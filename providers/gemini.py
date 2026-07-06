import datetime
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Google Generative Language (Gemini) REST API.
#   docs: https://ai.google.dev/api/generate-content  # sweep-ok
# Auth is an API key sent as the x-goog-api-key header (NOT Bearer; the key is
# also accepted as a ?key= query param, but the header keeps it out of URLs/logs).
_API_BASE    = "https://generativelanguage.googleapis.com/v1beta"  # sweep-ok
_MAX_TOKENS  = 8192
_CTX_DEFAULT = 1_048_576   # Gemini chat models are ~1M; real value comes from /models
_PROVIDER    = "gemini"    # sweep-ok

_CONFIG_DIR = None   # set in setup() from lc["CONFIG_DIR"]
_MODEL_DATA = {}     # model_id -> {max_input_tokens, max_tokens, can_think, paid}
_lc         = {}     # display helpers injected by setup()
_cfg        = None   # read-only cfg ref (for the free/all tier filter)

# Models the /models endpoint returns that can't drive a coding agent (image /
# audio / tts / embedding / robotics output). They still list generateContent but
# aren't text chat models, so they'd only clutter /model.  # sweep-ok
_SKIP_KEYS = ("tts", "image", "embedding", "lyria", "robotics", "banana", "-clip")  # sweep-ok

# Paid-tier families: NOT on the Gemini free tier (free_tier limit:0 -> a hard
# billing-wall 429, never a transient throttle). This key is free-tier, so they're
# kept out of the picker entirely. (If a paid key is ever used, drop this filter.)
_PAID_KEYS = ("pro", "deep-research", "computer-use", "antigravity")  # sweep-ok

# Privacy posture shown at the point of use (on connect + in /usage).
_PRIVACY    = "free-tier prompts ARE used to train Google models & may be human-reviewed - don't send confidential code"
_PRIVACY_HI = True   # high risk -> yellow


# ----------------------------------------------------------------------------
# Credentials: ~/.config/leancoder/gemini.json
# {"key": "...", "active": false}
# Key resolution order: GEMINI_API_KEY > GOOGLE_API_KEY > keyfile  # sweep-ok
# ----------------------------------------------------------------------------

def _cred_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "gemini.json"  # sweep-ok

def _models_cache_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "model-cache-gemini.json"

def _api_key():
    for ev in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):  # sweep-ok
        v = os.environ.get(ev, "")
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
# Capability heuristics (the /models endpoint has no "thinking" flag; infer it
# from the family). Gemini 2.5 and 3.x reason; 2.0 / gemma do not.  # sweep-ok
# ----------------------------------------------------------------------------

def _can_think(model):
    m = model.lower()  # sweep-ok
    return "gemini-3" in m or "2.5" in m  # sweep-ok


def _thinking_config(model, setting):
    """Map lean-coder's off/dynamic/max onto Gemini's per-family thinking knob.
    Returns a thinkingConfig dict, or None to let the model use its default
    (dynamic) budget. Avoids values a family rejects (e.g. budget 0 on *-pro)."""
    if setting in (None, "", "dynamic"):
        return None
    m = model.lower()  # sweep-ok
    if "gemini-3" in m:                      # gemini 3.x uses thinkingLevel  # sweep-ok
        if setting == "off":  return {"thinkingLevel": "low"}   # 3.x can't fully disable
        if setting == "max":  return {"thinkingLevel": "high"}
        return None
    if "2.5" in m:                           # gemini 2.5 uses an int budget  # sweep-ok
        is_pro = "pro" in m
        if setting == "off":  return {"thinkingBudget": 128 if is_pro else 0}
        if setting == "max":  return {"thinkingBudget": 32768 if is_pro else 24576}
        return {"thinkingBudget": -1}        # explicit dynamic
    return None


# ----------------------------------------------------------------------------
# Model cache
# ----------------------------------------------------------------------------

def _load_models_cache():
    try:
        return json.loads(_models_cache_path().read_text()).get("models", [])
    except Exception:
        return []


def _fetch_models(key):
    """GET /models with API-key auth. Keeps text chat models that support
    generateContent. Writes a disk cache on success; falls back to it on failure."""
    try:
        req = urllib.request.Request(
            f"{_API_BASE}/models?pageSize=1000",
            headers={"x-goog-api-key": key},  # sweep-ok
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")            # "models/gemini-2.5-pro"
            mid  = name.split("/", 1)[1] if "/" in name else name
            if not mid:
                continue
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            low = mid.lower()  # sweep-ok
            if any(k in low for k in _SKIP_KEYS):
                continue                          # non-chat models: never useful, always dropped
            out.append({
                "id":               mid,
                "max_input_tokens": m.get("inputTokenLimit", _CTX_DEFAULT),
                "max_tokens":       m.get("outputTokenLimit", _MAX_TOKENS),
                "can_think":        _can_think(mid),
                "paid":             any(k in low for k in _PAID_KEYS),  # hidden unless tier=all
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
    # All tiers by default: paid families (pro/deep-research/...) are listed too.
    # `/set tier free` hides them (they hard-wall 429 on a free-tier key). New
    # models surface automatically (live fetch).
    show_all = True
    try:
        show_all = ((_cfg.setting("tier") or "all") != "free") if _cfg else True  # sweep-ok
    except Exception:
        show_all = True
    return [mid for mid, m in _ensure_models().items() if show_all or not m.get("paid")]


def _context_window(model):
    return _ensure_models().get(model, {}).get("max_input_tokens", _CTX_DEFAULT)


def _capabilities(model):
    caps = {"tier": ["all", "free"]}     # all (default) shows paid + free; free hides paid
    if _ensure_models().get(model, {}).get("can_think") or _can_think(model):
        caps["thinking"] = ["off", "dynamic", "max"]
    return caps


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
# 429 retry: free-tier models are rate-limited. A genuine throttle ships a retry
# delay (Retry-After header, or RetryInfo.retryDelay like "26s" in the body); we
# honor it (+1s) and retry. A hard quota/billing block (e.g. paid-only models on
# a no-billing key) ships NO delay - we surface that instantly instead of waiting.
# ----------------------------------------------------------------------------

_MAX_RETRIES = 3
_MAX_WAIT    = 75   # seconds; don't hang the REPL on a long throttle
_RETRY_CODES = (429, 502, 503, 504)   # 429 throttle + transient server/gateway errors


def _backoff(attempt):
    """Exponential backoff (2s, 4s, 8s, ...) for transient errors with no
    server-supplied delay. Capped at _MAX_WAIT."""
    return min(2.0 * (2 ** attempt), _MAX_WAIT)


def _retry_after_secs(headers, body):
    """Seconds to wait before a 429 retry, or None when retrying is futile.

    Gemini attaches a RetryInfo (e.g. "57s") even to a HARD wall - a free-tier
    limit of 0 (model not on the free tier, like gemini-2.5-pro) or a per-DAY
    quota. Those never clear on a short retry, so honoring their delay just hangs
    the REPL ~1min per attempt. We only return a delay for a genuine per-minute
    throttle (limit > 0, not a daily cap)."""
    try:
        err = json.loads(body).get("error", {})
        msg = err.get("message", "")
        if "limit: 0" in msg or "billing" in msg.lower():
            return None                                # hard wall - never retry
        for d in err.get("details", []):
            t = d.get("@type", "")
            if "QuotaFailure" in t:
                for v in d.get("violations", []):
                    if "PerDay" in v.get("quotaId", ""):
                        return None                    # daily cap - a short retry won't help
        for d in err.get("details", []):
            if "RetryInfo" in d.get("@type", ""):
                rd = str(d.get("retryDelay", ""))      # e.g. "26s"
                if rd.endswith("s"):
                    return float(rd[:-1] or 0)
    except Exception:
        pass
    ra = headers.get("Retry-After") or headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    return None


def _interruptible_sleep(secs, should_abort):
    """Sleep in small slices so should_abort()/Ctrl-C can break the wait.
    Returns False if aborted, True if the full wait elapsed."""
    waited = 0.0
    while waited < secs:
        if should_abort and should_abort():
            return False
        chunk = min(0.5, secs - waited)
        time.sleep(chunk)
        waited += chunk
    return True


# ----------------------------------------------------------------------------
# Message / tool format conversion (lean-coder internal -> Gemini)
#   Gemini roles are "user" / "model"; there is no system role (systemInstruction
#   carries it) and tool results are functionResponse parts in a user turn,
#   matched to the call by function NAME (Gemini has no tool-call ids).
# ----------------------------------------------------------------------------

def _clean_schema(node):
    """Strip JSON-Schema keys the Gemini function-declaration parser rejects
    ($schema / additionalProperties / title), recursively. Leaves the OpenAPI
    subset Gemini accepts (type, description, properties, items, enum, required,
    nullable, anyOf, ...) untouched."""
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items()
                if k not in ("$schema", "additionalProperties", "title")}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def _cvt_tools(tools):
    decls = []
    for t in tools:
        fn     = t.get("function", {})
        params = _clean_schema(fn.get("parameters", {}) or {})
        decl   = {"name": fn["name"], "description": fn.get("description", "")}
        # Only attach parameters when there is at least one property; an empty
        # schema is better omitted than sent as an empty object.
        if isinstance(params, dict) and params.get("properties"):
            decl["parameters"] = params
        decls.append(decl)
    return [{"functionDeclarations": decls}] if decls else []


def _cvt_messages(messages):
    # tool-call id -> function name, so a tool result can name its call.
    id2name = {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                id2name[tc.get("id", "")] = tc.get("function", {}).get("name", "")

    system   = []
    contents = []
    i        = 0
    while i < len(messages):
        m    = messages[i]
        role = m.get("role")

        if role == "system":
            if m.get("content"):
                system.append(m["content"])
            i += 1

        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": m.get("content", "")}]})
            i += 1

        elif role == "assistant":
            parts = []
            if m.get("content"):
                parts.append({"text": m["content"]})
            for tc in (m.get("tool_calls") or []):
                fn   = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                fc_part = {"functionCall": {"name": fn.get("name", ""), "args": args or {}}}
                sig = tc.get("_thought_sig")
                if sig:                                # replay Gemini 3.x thought-signature
                    fc_part["thoughtSignature"] = sig
                parts.append(fc_part)
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            i += 1

        elif role == "tool":
            # lean-coder tags each tool result with `tool_name`; Gemini matches a
            # functionResponse to its call by NAME. Fall back to the preceding model
            # turn's functionCalls positionally, then to the id map, so the name is
            # never empty (an empty name 400s).
            prev_calls = []
            for prev in reversed(contents):
                if prev.get("role") == "model":
                    prev_calls = [p["functionCall"] for p in prev["parts"] if "functionCall" in p]
                    break
            parts = []
            ci    = 0
            while i < len(messages) and messages[i].get("role") == "tool":
                r    = messages[i]
                name = r.get("tool_name") or ""
                if not name and ci < len(prev_calls):
                    name = prev_calls[ci].get("name", "")
                if not name:
                    name = id2name.get(r.get("tool_call_id", ""), "") or r.get("name", "")
                content = r.get("content", "")
                try:
                    parsed = json.loads(content)
                    resp   = parsed if isinstance(parsed, dict) else {"output": parsed}
                except Exception:
                    resp = {"output": content}
                parts.append({"functionResponse": {"name": name, "response": resp}})
                ci += 1
                i  += 1
            contents.append({"role": "user", "parts": parts})

        else:
            i += 1

    return "\n".join(system), contents


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------

class _GeminiClient:
    uncapped_ctx = True

    def __init__(self, cfg):
        self.cfg               = cfg
        self.last_out_tokens   = None
        self.last_cache_read   = 0    # cachedContentTokenCount from last turn
        self.last_cache_write  = 0    # Gemini implicit caching reports no write
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
        tool_calls    = []
        prompt_total  = None
        cached        = 0
        out_tokens    = None
        printed       = False
        aborted       = False
        tc_idx        = 0
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
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                for cand in obj.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("thought"):          # reasoning summary - not visible output
                            continue
                        if "text" in part:
                            chunk = part["text"]
                            if chunk:
                                if not printed:
                                    spin.stop()
                                    sys.stdout.write(_lc["blue"]("● "))
                                    printed = True
                                md.feed(chunk)
                                content_parts.append(chunk)
                        elif "functionCall" in part:
                            fc = part["functionCall"]
                            tc = {
                                "id": fc.get("id") or f"call_{tc_idx}_{fc.get('name', '')}",
                                "function": {"name": fc.get("name", ""),
                                             "arguments": fc.get("args", {}) or {}},
                            }
                            sig = part.get("thoughtSignature")
                            if sig:
                                # Gemini 3.x returns an opaque thought-signature on the
                                # functionCall part and REQUIRES it replayed next turn
                                # (else 400). Stash it on the tool_call so it round-trips
                                # through lean-coder's history + autosave.
                                tc["_thought_sig"] = sig
                            tool_calls.append(tc)
                            tc_idx += 1

                um = obj.get("usageMetadata")
                if um:
                    prompt_total = um.get("promptTokenCount", prompt_total)
                    cached       = um.get("cachedContentTokenCount", cached) or 0
                    _out         = um.get("candidatesTokenCount", 0) or 0
                    _out        += um.get("thoughtsTokenCount", 0) or 0
                    if _out:
                        out_tokens = _out
        finally:
            spin.stop()

        if printed:
            md.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

        if out_tokens:
            self.last_out_tokens = out_tokens
        # Gemini's promptTokenCount INCLUDES cached tokens; core sums
        # prompt_eval + cache_read + cache_write, so return only the uncached
        # portion as prompt_eval to avoid double-counting the cache hit.
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

    def _send(self, key, model, payload, should_abort):
        url  = f"{_API_BASE}/models/{model}:streamGenerateContent?alt=sse"
        data = json.dumps(payload).encode()
        attempt = 0
        while True:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json", "x-goog-api-key": key},  # sweep-ok
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return self._consume(resp, should_abort)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                    hint = _retry_after_secs(e.headers or {}, body)  # None on a hard wall
                    if e.code == 429:
                        wait = (hint + 1) if hint is not None else None  # retry only with a hint
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

        system, contents = _cvt_messages(messages)
        gem_tools        = _cvt_tools(tools) if tools else []

        cur_model = self._model()
        minfo     = _ensure_models().get(cur_model, {})
        max_out   = minfo.get("max_tokens", _MAX_TOKENS)
        max_in    = minfo.get("max_input_tokens", _CTX_DEFAULT)
        _est_toks = len(json.dumps(messages)) // 4
        if _est_toks > max_in:
            raise RuntimeError(
                f"context too large for {cur_model} "
                f"(~{_est_toks:,} tokens vs {max_in:,} limit) - run /compact first"
            )

        gen_cfg = {"maxOutputTokens": max_out}
        if _can_think(cur_model):
            tcfg = _thinking_config(cur_model, self.cfg.setting("thinking"))
            if tcfg:
                gen_cfg["thinkingConfig"] = tcfg

        payload = {"contents": contents, "generationConfig": gen_cfg}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if gem_tools:
            payload["tools"] = gem_tools

        result = self._send(key, cur_model, payload, should_abort)
        self._sess_cache_read  += self.last_cache_read
        self._sess_cache_write += self.last_cache_write
        return result


def _make_client(cfg):
    return _GeminiClient(cfg)


# ----------------------------------------------------------------------------
# Provider hooks
# ----------------------------------------------------------------------------

def _available():
    return bool(_api_key())


def _autostart():
    return bool(_api_key()) and _cred_data().get("active", False)


def _on_activate(agent, cfg):
    global _cfg
    _cfg = cfg
    _ensure_models(force=True)
    if cfg.setting("thinking") is None and _can_think(cfg.active_model()):
        cfg.set_setting("thinking", "dynamic")
    if cfg.setting("tier") is None:
        cfg.set_setting("tier", "all")
    key    = _api_key()
    masked = f"{key[:6]}...{key[-4:]}" if key and len(key) > 12 else "(env)"
    print(_lc["dim"](f"  Gemini API key  {masked}"))
    if (cfg.setting("tier") or "all") != "all":
        print(_lc["dim"]("  free tier: paid models (pro/deep-research/...) hidden; /set tier all to show"))
    else:
        print(_lc["dim"]("  tier all: paid models (pro/deep-research/...) shown; /set tier free to hide (they 429 on a free key)"))
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
        print(_lc["dim"]("Paste your Gemini API key and press Enter:"))
        try:
            raw = getpass.getpass(prompt="").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        m   = re.search(r"\S+", raw)  # sweep-ok - strip surrounding whitespace/noise
        key = m.group(0) if m else raw
        if len(key) < 20:
            print(_lc["red"]("[!] does not look like an API key"))
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
    cur_think = cfg.setting("thinking") or ("na" if not _can_think(cur_model) else "dynamic")
    lines = [bold(cur_model) + dim(f"   thinking: {cur_think}")]

    si = getattr(agent, "session_in",  0) or 0
    so = getattr(agent, "session_out", 0) or 0
    cr = getattr(agent.client, "_sess_cache_read",  0) or 0
    cw = getattr(agent.client, "_sess_cache_write", 0) or 0
    cache_s = f"   cache r {_sn(cr)} w {_sn(cw)}" if (cr or cw) else ""
    lines.append(dim(f"\n  session   in {si:,}   out {so:,}   total {si + so:,}{cache_s}"))
    lines.append("\n" + _privacy_note())
    lines.append(dim("  (Gemini API key - no subscription quota endpoint)"))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

PROVIDER = {
    "name":           _PROVIDER,
    "description":    "Google Gemini (API key)",  # sweep-ok
    "make_client":    _make_client,
    "list_models":    _list_models,
    "context_window": _context_window,
    "capabilities":   _capabilities,
    "available":      _available,
    "autostart":      _autostart,
    "on_activate":    _on_activate,
    "on_deactivate":  _on_deactivate,
    "usage":          _usage,
    "login":          _login,
    "clear":          _clear,
    "detail":         _detail,
    "tag":            "gem",
}


def setup(lc, cfg):
    global _CONFIG_DIR, _cfg
    _lc.update(lc)
    _CONFIG_DIR = lc.get("CONFIG_DIR")
    _cfg = cfg
