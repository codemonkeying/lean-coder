import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Groq - an OpenAI-compatible inference API (very fast; free tier is generous but
# rate-limited). docs: https://console.groq.com/docs/openai  # sweep-ok
# Auth is a Bearer key (gsk_...). Wire format is OpenAI-compatible, so conversion
# from lean-coder's internal format is nearly an identity.
_API_BASE    = "https://api.groq.com/openai/v1"  # sweep-ok
_MAX_TOKENS  = 4096
_OUT_CAP     = 4096        # free tier per-minute token budget is tiny (as low as 8k TPM
                           # on gpt-oss-20b) and counts max_tokens; keep the cap small so
                           # a turn fits the budget. Large contexts may still 413 (clear msg)
_CTX_DEFAULT = 131_072
_PROVIDER    = "groq"  # sweep-ok

_CONFIG_DIR = None
_MODEL_DATA = {}
_lc         = {}

# Groq's edge (Cloudflare) 403s the default Python-urllib User-Agent (error 1010);
# any real UA passes.  # sweep-ok
_UA = "lean-coder"  # sweep-ok

# Privacy posture shown at the point of use (on connect + in /usage).
_PRIVACY    = "prompts are not used for training (API traffic under a separate DPA)"
_PRIVACY_HI = False   # low risk -> dim, not a warning

# Non-chat models the /models endpoint returns (speech, TTS, safety classifiers).
# They can't drive a coding agent, so they're kept out of the picker.  # sweep-ok
_SKIP_KEYS = ("whisper", "orpheus", "guard")  # sweep-ok

# Models that generate text but reject custom tool definitions (Groq's "compound"
# agentic systems run their own built-in tools). Kept in the list but flagged.  # sweep-ok
_NO_TOOLS_KEYS = ("compound",)  # sweep-ok


# ----------------------------------------------------------------------------
# Credentials: ~/.config/leancoder/groq.json   {"key": "gsk_...", "active": false}
# Key resolution order: GROQ_API_KEY > keyfile  # sweep-ok
# ----------------------------------------------------------------------------

def _cred_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "groq.json"  # sweep-ok

def _models_cache_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "model-cache-groq.json"

def _api_key():
    v = os.environ.get("GROQ_API_KEY", "")  # sweep-ok
    if v:
        return v
    try:
        return json.loads(_cred_path().read_text()).get("key", "")
    except Exception:
        return ""

def _cred_data():
    try:
        return json.loads(_cred_path().read_text())
    except Exception:
        return {}

def _save_cred(data):
    _cred_path().parent.mkdir(parents=True, exist_ok=True)
    _cred_path().write_text(json.dumps(data, indent=2))
    try:
        _cred_path().chmod(0o600)
    except OSError:
        pass

def _clear_cred():
    if _cred_path().is_file():
        _cred_path().unlink()


# ----------------------------------------------------------------------------
# Transient-error retry (same shape as the other providers)
# ----------------------------------------------------------------------------

_MAX_RETRIES = 3
_MAX_WAIT    = 75
_RETRY_CODES = (429, 502, 503, 504)   # 429 retried only with a delay hint; 5xx via backoff


def _retry_after_secs(headers, body):
    ra = headers.get("Retry-After") or headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    try:                                   # Groq embeds "try again in 7.66s" in the message
        msg = json.loads(body).get("error", {}).get("message", "")
        m = re.search(r"in ([\d.]+)s", msg)  # sweep-ok
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _backoff(attempt):
    return min(2.0 * (2 ** attempt), _MAX_WAIT)


def _interruptible_sleep(secs, should_abort):
    waited = 0.0
    while waited < secs:
        if should_abort and should_abort():
            return False
        chunk = min(0.5, secs - waited)
        time.sleep(chunk)
        waited += chunk
    return True


# ----------------------------------------------------------------------------
# Model cache
# ----------------------------------------------------------------------------

def _load_models_cache():
    try:
        return json.loads(_models_cache_path().read_text()).get("models", [])
    except Exception:
        return []


def _fetch_models(key):
    """GET /models (OpenAI-shaped). Keeps active text->text chat models; drops
    speech/TTS/classifier models that can't drive a coding agent."""
    try:
        req = urllib.request.Request(
            f"{_API_BASE}/models",
            headers={"Authorization": f"Bearer {key}", "User-Agent": _UA},  # sweep-ok
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        out = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            if not mid or not m.get("active", True):
                continue
            low = mid.lower()  # sweep-ok
            if any(k in low for k in _SKIP_KEYS):
                continue
            imod = m.get("input_modalities")  or ["text"]
            omod = m.get("output_modalities") or ["text"]
            if "text" not in imod or "text" not in omod:
                continue
            out.append({
                "id":               mid,
                "max_input_tokens": m.get("context_window") or m.get("context_length") or _CTX_DEFAULT,
                "max_tokens":       m.get("max_completion_tokens") or m.get("max_output_length") or _MAX_TOKENS,
                # Only gpt-oss takes reasoning_effort low/med/high. qwen3 et al accept
                # only none/default (a different scale) and 400 on low/med/high, so we
                # don't expose the effort knob for them - they just reason by default.
                "reason":           "gpt-oss" in low,  # sweep-ok
            })
        if out:
            try:
                _models_cache_path().parent.mkdir(parents=True, exist_ok=True)
                _models_cache_path().write_text(json.dumps({"models": out}))
            except Exception:
                pass
            return out
    except Exception:
        pass
    return _load_models_cache()


def _ensure_models(force=False):
    global _MODEL_DATA
    if _MODEL_DATA and not force:
        return _MODEL_DATA
    key    = _api_key()
    models = _fetch_models(key) if key else _load_models_cache()
    _MODEL_DATA = {m["id"]: m for m in models}
    return _MODEL_DATA


def _list_models():
    return list(_ensure_models())


def _context_window(model):
    return _ensure_models().get(model, {}).get("max_input_tokens", _CTX_DEFAULT)


def _capabilities(model):
    caps = {}
    if _ensure_models().get(model, {}).get("reason"):
        caps["effort"] = ["low", "med", "high"]
    return caps


def _is_no_tools(model):
    m = model.lower()  # sweep-ok
    return any(k in m for k in _NO_TOOLS_KEYS)


def _model_status(model):
    # Shown dimmed next to the model in /model. Flags models that generate text but
    # reject custom tool definitions, so the picker says so up front.
    return "chat-only (no tool calling)" if _is_no_tools(model) else None


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------

def _sn(n):
    try:
        n = int(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.0f}k"
        return str(n)
    except Exception:
        return str(n)


def _privacy_note():
    if _PRIVACY_HI:
        paint = _lc.get("yellow") or _lc.get("dim") or (lambda s: s)
    else:
        paint = _lc.get("dim") or (lambda s: s)
    return paint("  privacy: " + _PRIVACY)


# ----------------------------------------------------------------------------
# Message / tool conversion (lean-coder internal -> OpenAI/Groq; near identity)
# ----------------------------------------------------------------------------

def _cvt_tools(tools):
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "type": "function",
            "function": {
                "name":        fn["name"],
                "description": fn.get("description", ""),
                "parameters":  fn.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return out


def _cvt_messages(messages):
    out = []
    i   = 0
    while i < len(messages):
        m    = messages[i]
        role = m.get("role")
        if role in ("system", "user"):
            out.append({"role": role, "content": m.get("content", "")})
            i += 1
        elif role == "assistant":
            msg = {"role": "assistant", "content": m.get("content", "") or ""}
            tcs = []
            for tc in (m.get("tool_calls") or []):
                fn   = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                tcs.append({
                    "id":       tc.get("id", ""),
                    "type":     "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
            if tcs:
                msg["tool_calls"] = tcs
                if not msg["content"]:
                    msg["content"] = None
            out.append(msg)
            i += 1
        elif role == "tool":
            # lean-coder tool results carry `tool_name`, not `tool_call_id`; OpenAI
            # needs the id, so correlate positionally to the preceding assistant's
            # tool_calls (their ids round-tripped from our streamed response).
            prev_calls = []
            for prev in reversed(out):
                if prev.get("role") == "assistant":
                    prev_calls = prev.get("tool_calls") or []
                    break
            ci = 0
            while i < len(messages) and messages[i].get("role") == "tool":
                r    = messages[i]
                tcid = r.get("tool_call_id", "")
                if not tcid and ci < len(prev_calls):
                    tcid = prev_calls[ci].get("id", "")
                out.append({
                    "role":         "tool",
                    "tool_call_id": tcid,
                    "content":      r.get("content", ""),
                })
                ci += 1
                i  += 1
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------

class _GroqClient:
    uncapped_ctx = True

    def __init__(self, cfg):
        self.cfg               = cfg
        self.last_out_tokens   = None
        self.last_cache_read   = 0
        self.last_cache_write  = 0
        self._sess_cache_read  = 0
        self._sess_cache_write = 0

    def _model(self):
        return self.cfg.active_model()

    def list_models(self):
        return _list_models()

    def running_models(self):
        return []

    def detect_num_ctx(self):
        return _context_window(self._model())

    def _consume(self, resp, should_abort):
        content_parts = []
        tool_acc      = {}
        order         = []
        prompt_total  = None
        cached        = 0
        out_tokens    = None
        printed       = False
        aborted       = False
        md            = _lc["MarkdownStream"](sys.stdout.write)

        spin = _lc["Spinner"]("thinking", _lc["THINK_FRAMES"]).start()
        try:
            for raw in resp:
                if should_abort and should_abort():
                    aborted = True
                    break
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    txt   = delta.get("content")
                    if txt:
                        if not printed:
                            spin.stop()
                            sys.stdout.write(_lc["blue"]("● "))
                            printed = True
                        md.feed(txt)
                        content_parts.append(txt)
                    for tcd in (delta.get("tool_calls") or []):
                        idx = tcd.get("index", 0)
                        if idx not in tool_acc:
                            tool_acc[idx] = {"id": "", "name": "", "args": []}
                            order.append(idx)
                        slot = tool_acc[idx]
                        if tcd.get("id"):
                            slot["id"] = tcd["id"]
                        fn = tcd.get("function", {})
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"].append(fn["arguments"])

                # Groq returns usage in the final chunk under x_groq.usage (and
                # sometimes a top-level usage); check both.
                um = obj.get("usage") or (obj.get("x_groq") or {}).get("usage")
                if um:
                    prompt_total = um.get("prompt_tokens", prompt_total)
                    out_tokens   = um.get("completion_tokens", out_tokens)
                    det          = um.get("prompt_tokens_details") or {}
                    cached       = det.get("cached_tokens", cached) or 0
        finally:
            spin.stop()

        if printed:
            md.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

        tool_calls = []
        for idx in order:
            slot = tool_acc[idx]
            try:
                args = json.loads("".join(slot["args"])) if slot["args"] else {}
            except Exception:
                args = {}
            tool_calls.append({
                "id":       slot["id"] or f"call_{idx}",
                "function": {"name": slot["name"], "arguments": args},
            })

        if out_tokens:
            self.last_out_tokens = out_tokens
        # prompt_tokens includes any cache hit; core sums prompt_eval + cache, so
        # return only the uncached part.
        self.last_cache_read  = cached
        self.last_cache_write = 0
        prompt_eval = None
        if prompt_total is not None:
            prompt_eval = max(prompt_total - cached, 0)

        content   = "".join(content_parts)
        assistant = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, prompt_eval, aborted

    def _send(self, key, payload, should_abort):
        data    = json.dumps(payload).encode()
        model   = payload.get("model", "")
        attempt = 0
        while True:
            req = urllib.request.Request(
                f"{_API_BASE}/chat/completions",
                data=data, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}",
                         "User-Agent": _UA},  # sweep-ok
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return self._consume(resp, should_abort)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                    hint = _retry_after_secs(e.headers or {}, body)
                    if e.code == 429:
                        wait = (hint + 1) if hint is not None else None
                    else:
                        wait = (hint + 1) if hint is not None else _backoff(attempt)
                    if wait is not None and wait <= _MAX_WAIT:
                        attempt += 1
                        reason = "rate limited" if e.code == 429 else f"unavailable ({e.code})"
                        print(_lc["red"](f"[!] {model} {reason} - retry {attempt}/{_MAX_RETRIES} "
                                         f"in {wait:.0f}s"))
                        if not _interruptible_sleep(wait, should_abort):
                            raise RuntimeError("aborted during retry wait") from None
                        continue
                raise RuntimeError(f"API {e.code}: {body[:400]}") from None

    def chat(self, messages, tools, should_abort=None):
        key = _api_key()
        if not key:
            raise RuntimeError("no API key - use /provider login")

        cur_model = self._model()
        minfo     = _ensure_models().get(cur_model, {})
        max_out   = min(minfo.get("max_tokens", _MAX_TOKENS), _OUT_CAP)
        max_in    = minfo.get("max_input_tokens", _CTX_DEFAULT)
        _est_toks = len(json.dumps(messages)) // 4
        if _est_toks > max_in:
            raise RuntimeError(
                f"context too large for {cur_model} "
                f"(~{_est_toks:,} tokens vs {max_in:,} limit) - run /compact first"
            )

        payload = {
            "model":          cur_model,
            "messages":       _cvt_messages(messages),
            "max_tokens":     max_out,
            "stream":         True,
            "stream_options": {"include_usage": True},
        }
        # compound-style models reject custom tools; send none so they 200 as
        # chat-only instead of 400ing. The /model picker flags them via model_status.
        if tools and not _is_no_tools(cur_model):
            payload["tools"] = _cvt_tools(tools)

        if minfo.get("reason"):
            eff = self.cfg.setting("effort")
            if eff:
                payload["reasoning_effort"] = "medium" if eff == "med" else eff

        result = self._send(key, payload, should_abort)
        self._sess_cache_read  += self.last_cache_read
        self._sess_cache_write += self.last_cache_write
        return result


def _make_client(cfg):
    return _GroqClient(cfg)


# ----------------------------------------------------------------------------
# Provider hooks
# ----------------------------------------------------------------------------

def _available():
    return bool(_api_key())


def _autostart():
    return bool(_api_key()) and _cred_data().get("active", False)


def _on_activate(agent, cfg):
    _ensure_models(force=True)
    if cfg.setting("effort") is None and _ensure_models().get(cfg.active_model(), {}).get("reason"):
        cfg.set_setting("effort", "med")
    key    = _api_key()
    masked = f"{key[:8]}...{key[-4:]}" if key and len(key) > 12 else "(env)"
    print(_lc["dim"](f"  Groq API key  {masked}"))
    print(_privacy_note())
    d = _cred_data()
    d["active"] = True
    if d.get("key") or _cred_path().is_file():
        _save_cred(d)


def _on_deactivate(agent, cfg):
    d = _cred_data()
    d["active"] = False
    if d.get("key"):
        _save_cred(d)


def _login(agent, cfg):
    key = _api_key()
    if not key:
        print(_lc["dim"]("Paste your Groq API key (gsk_...) and press Enter:"))
        try:
            raw = getpass.getpass(prompt="").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        m   = re.search(r"gsk_\S+", raw)  # sweep-ok - extract key from surrounding noise
        key = m.group(0) if m else raw
        if not key.startswith("gsk_"):  # sweep-ok
            print(_lc["red"]("[!] does not look like a Groq key"))
            return False
        _save_cred({"key": key, "active": True})
        print(_lc["dim"](f"  saved to {_cred_path()}"))
    else:
        d = _cred_data()
        d["active"] = True
        if d.get("key"):
            _save_cred(d)
    return True


def _clear(agent, cfg):
    _clear_cred()


def _usage(agent, cfg):
    return None


def _detail(agent, cfg):
    bold = _lc["bold"]
    dim  = _lc["dim"]

    cur_model = cfg.active_model()
    can_eff   = _ensure_models().get(cur_model, {}).get("reason")
    cur_eff   = cfg.setting("effort") or ("med" if can_eff else "na")
    lines = [bold(cur_model) + dim(f"   effort: {cur_eff}")]

    si = getattr(agent, "session_in",  0) or 0
    so = getattr(agent, "session_out", 0) or 0
    cr = getattr(agent.client, "_sess_cache_read",  0) or 0
    cw = getattr(agent.client, "_sess_cache_write", 0) or 0
    cache_s = f"   cache r {_sn(cr)} w {_sn(cw)}" if (cr or cw) else ""
    lines.append(dim(f"\n  session   in {si:,}   out {so:,}   total {si + so:,}{cache_s}"))
    lines.append("\n" + _privacy_note())
    lines.append(dim("  (Groq - see console.groq.com for limits)"))  # sweep-ok
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

PROVIDER = {
    "name":           _PROVIDER,
    "description":    "Groq (fast OpenAI-compatible inference; API key)",  # sweep-ok
    "make_client":    _make_client,
    "list_models":    _list_models,
    "context_window": _context_window,
    "capabilities":   _capabilities,
    "model_status":   _model_status,
    "tool_support":   lambda m: not _is_no_tools(m),   # core forces chat-only for these
    "available":      _available,
    "autostart":      _autostart,
    "on_activate":    _on_activate,
    "on_deactivate":  _on_deactivate,
    "usage":          _usage,
    "login":          _login,
    "clear":          _clear,
    "detail":         _detail,
    "tag":            "groq",
}


def setup(lc, cfg):
    global _CONFIG_DIR
    _lc.update(lc)
    _CONFIG_DIR = lc.get("CONFIG_DIR")
