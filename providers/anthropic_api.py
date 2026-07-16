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

_API_BASE       = "https://api.anthropic.com"  # sweep-ok
_MAX_TOKENS     = 16384
_CTX_DEFAULT    = 200_000
_SONNET_CTX_CAP = 200_000   # sonnet 1M draws credits on Max; cap it
_PROVIDER       = "anthropic_api"  # sweep-ok

_CONFIG_DIR = None   # set in setup() from lc["CONFIG_DIR"]
_MODEL_DATA = {}     # model_id -> {max_input_tokens, max_tokens, can_think, ...}
_lc         = {}     # display helpers injected by setup()

# Transient-error retry. 429 is handled separately (Haiku fallback in chat()); here
# we only back off on Anthropic's 529 "Overloaded" + gateway/5xx errors, which the
# provider previously surfaced straight to the user.
_MAX_RETRIES = 3
_MAX_WAIT    = 75   # seconds; cap a single wait so the REPL never hangs
_RETRY_CODES = (500, 502, 503, 504, 529)   # 529 = Anthropic "Overloaded"

# Privacy posture shown at the point of use (on connect + in /usage).
_PRIVACY    = "commercial API - inputs & outputs are not used to train Anthropic's models"
_PRIVACY_HI = False   # low risk -> dim
# Show the on-activate identity + privacy banner? Off by default (the core
# 'provider: <name>  model: <model>' line already announces the backend).
_SHOW_BANNER = False


def _retry_after_secs(headers):
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    return None


def _backoff(attempt):
    """Exponential backoff (2s, 4s, 8s, ...), capped at _MAX_WAIT."""
    return min(2.0 * (2 ** attempt), _MAX_WAIT)


def _interruptible_sleep(secs, should_abort):
    """Sleep in slices so should_abort()/Ctrl-C breaks the wait. False if aborted."""
    waited = 0.0
    while waited < secs:
        if should_abort and should_abort():
            return False
        chunk = min(0.5, secs - waited)
        time.sleep(chunk)
        waited += chunk
    return True


# ----------------------------------------------------------------------------
# Credentials: ~/.config/leancoder/anthropic_api.json  # sweep-ok
# {"key": "sk-ant-api03-...", "active": false}
# Key resolution order: LEANCODER_ANTHROPIC_API_KEY > ANTHROPIC_API_KEY > keyfile  # sweep-ok
# ----------------------------------------------------------------------------

def _cred_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "anthropic_api.json"  # sweep-ok

def _models_cache_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "model-cache-api.json"

def _api_key():
    for ev in ("LEANCODER_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"):  # sweep-ok
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
# Model cache
# ----------------------------------------------------------------------------

def _load_models_cache():
    try:
        return json.loads(_models_cache_path().read_text()).get("models", [])
    except Exception:
        return []


def _fetch_models(key):
    """GET /v1/models with API key auth. Writes disk cache on success."""
    try:
        req = urllib.request.Request(
            f"{_API_BASE}/v1/models",
            headers={
                "x-api-key":         key,  # sweep-ok
                "anthropic-version": "2023-06-01",  # sweep-ok
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        out = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            if not mid:
                continue
            caps  = m.get("capabilities", {})
            think = caps.get("thinking", {})
            out.append({
                "id":               mid,
                "max_input_tokens": m.get("max_input_tokens", _CTX_DEFAULT),
                "max_tokens":       m.get("max_tokens", _MAX_TOKENS),
                "can_think":    think.get("types", {}).get("adaptive", {}).get("supported", False),
                "can_effort":   caps.get("effort", {}).get("supported", False),
                "can_ctx_mgmt": caps.get("context_management", {}).get(
                    "clear_thinking_20251015", {}).get("supported", False),
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
    limit = _ensure_models().get(model, {}).get("max_input_tokens", _CTX_DEFAULT)
    if "sonnet" in model:  # sweep-ok
        limit = min(limit, _SONNET_CTX_CAP)
    return limit


def _capabilities(model):
    info = _ensure_models().get(model, {})
    caps = {}
    if info.get("can_think"):
        caps["thinking"] = ["off", "adaptive", "max"]
    if info.get("can_effort"):
        caps["effort"] = ["low", "med", "high"]
    return caps


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------

def _fmt_reset(iso):
    try:
        dt  = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        if dt.date() == now.date():
            return dt.strftime("%H:%M")
        return dt.strftime("%-d%b")
    except Exception:
        return iso[:16] if iso else ""


def _bar(pct, width=10):
    filled = min(int(pct / 100 * width), width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


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
# Message / tool format conversion (API format matches the OAuth provider)
# ----------------------------------------------------------------------------

def _cvt_tools(tools):
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "name":         fn["name"],
            "description":  fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _tool_result_content(r):
    """tool_result.content for one role:tool message. Plain string when there is
    no image (legacy path, unchanged). When the message carries image_path AND
    the core image encoder is available, return an ARRAY of blocks - text first,
    then an image block - per the Anthropic vision spec (content MUST be a list,
    never a stringified list). Any encode failure falls back to the text string,
    so a broken/missing image never breaks the turn."""
    text = r.get("content", "")
    ip   = r.get("image_path")
    if not ip:
        return text
    enc = _lc.get("_encode_image_block")
    got = enc(ip) if enc else None
    if not got:
        return text
    media, data = got
    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.append({
        "type": "image",
        "source": {"type": "base64", "media_type": media, "data": data},
    })
    return blocks


def _cvt_messages(messages):
    system = ""
    out    = []
    i      = 0
    while i < len(messages):
        m    = messages[i]
        role = m.get("role")

        if role == "system":
            system = m.get("content", "")
            i += 1

        elif role == "user":
            um = {"role": "user", "content": m["content"]}
            if m.get("cache_boundary"):
                um["cache_boundary"] = True     # core's stable-prefix signal (see chat())
            out.append(um)
            i += 1

        elif role == "assistant":
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in (m.get("tool_calls") or []):
                fn   = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                content.append({
                    "type":  "tool_use",
                    "id":    tc.get("id", f"call_{i}"),
                    "name":  fn.get("name", ""),
                    "input": args,
                })
            out.append({"role": "assistant", "content": content or ""})
            i += 1

        elif role == "tool":
            j = i - 1
            while j >= 0 and messages[j].get("role") == "tool":
                j -= 1
            prev_calls = []
            if j >= 0 and messages[j].get("role") == "assistant":
                prev_calls = messages[j].get("tool_calls") or []
            blocks = []
            ci     = 0
            while i < len(messages) and messages[i].get("role") == "tool":
                r    = messages[i]
                tcid = r.get("tool_call_id", "")
                if not tcid and ci < len(prev_calls):
                    tcid = prev_calls[ci].get("id", "")
                blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": tcid,
                    "content":     _tool_result_content(r),
                })
                ci += 1
                i  += 1
            out.append({"role": "user", "content": blocks})

        else:
            i += 1

    return system, out


def _mark_cache(msg):
    """Add cache_control breakpoint (1h TTL, session-scoped) to last content block."""
    cc = {"type": "ephemeral", "ttl": "1h"}
    content = msg.get("content")
    if isinstance(content, list) and content:
        last = dict(content[-1])
        last["cache_control"] = cc
        return dict(msg, content=[*content[:-1], last])
    if isinstance(content, str) and content:
        return dict(msg, content=[{"type": "text", "text": content, "cache_control": cc}])
    return msg


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------

_API_BETAS = (  # sweep-ok - oauth-2025-04-20 deliberately absent (API key path)
    "claude-code-20250219,"  # sweep-ok
    "interleaved-thinking-2025-05-14,"
    "thinking-token-count-2026-05-13,"
    "context-management-2025-06-27,"
    "prompt-caching-scope-2026-01-05,"
    "advisor-tool-2026-03-01,"
    "effort-2025-11-24,"
    "cache-diagnosis-2026-04-07,"
    "extended-cache-ttl-2025-04-11"
)


class _ApiKeyClient:
    uncapped_ctx = True

    def __init__(self, cfg):
        self.cfg              = cfg
        self.last_out_tokens  = None
        self._last_rl         = {}   # anthropic-ratelimit-* response headers  # sweep-ok
        self.last_cache_read  = 0    # cache_read_input_tokens from last turn
        self.last_cache_write = 0    # cache_creation_input_tokens from last turn
        self.last_cache_write_1h = 0 # ephemeral_1h_input_tokens from last turn
        self._sess_cache_read  = 0   # session totals
        self._sess_cache_write = 0
        self._sess_cache_write_1h = 0

    def _model(self):
        return self.cfg.active_model()

    def _haiku_model(self):
        haikus = [m for m in _list_models() if "haiku" in m]  # sweep-ok
        return haikus[0] if haikus else None

    def list_models(self):
        return _list_models()

    def running_models(self):
        return []

    def detect_num_ctx(self):
        return _context_window(self._model())

    def _consume_urllib(self, resp, should_abort):
        content_parts = []
        tool_calls    = []
        cur_tool      = None
        cur_json      = []
        prompt_eval   = None
        output_eval   = None
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

                etype = obj.get("type")

                if etype == "content_block_start":
                    blk = obj.get("content_block", {})
                    if blk.get("type") == "tool_use":
                        cur_tool = {"id": blk.get("id", ""), "name": blk.get("name", "")}
                        cur_json = []

                elif etype == "content_block_delta":
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta":
                        chunk = delta.get("text", "")
                        if chunk:
                            if not printed:
                                spin.stop()
                                sys.stdout.write(_lc["blue"]("● "))
                                printed = True
                            md.feed(chunk)
                            content_parts.append(chunk)
                    elif delta.get("type") == "input_json_delta":
                        cur_json.append(delta.get("partial_json", ""))

                elif etype == "content_block_stop":
                    if cur_tool:
                        try:
                            args = json.loads("".join(cur_json)) if cur_json else {}
                        except Exception:
                            args = {}
                        tool_calls.append({
                            "id": cur_tool["id"],
                            "function": {"name": cur_tool["name"], "arguments": args},
                        })
                        cur_tool = None
                        cur_json = []

                elif etype == "message_start":
                    _u = obj.get("message", {}).get("usage", {})
                    prompt_eval = _u.get("input_tokens")
                    self.last_cache_read  = _u.get("cache_read_input_tokens",  0) or 0
                    self.last_cache_write = _u.get("cache_creation_input_tokens", 0) or 0
                    self.last_cache_write_1h = (_u.get("cache_creation") or {}).get(
                        "ephemeral_1h_input_tokens", 0) or 0

                elif etype == "message_delta":
                    output_eval = obj.get("usage", {}).get("output_tokens")

                elif etype == "error":
                    raise RuntimeError(obj.get("error", {}).get("message", "API error"))

        finally:
            spin.stop()

        if printed:
            md.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

        if output_eval:
            self.last_out_tokens = output_eval
        content   = "".join(content_parts)
        assistant = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, prompt_eval, aborted

    def _send(self, key, payload, should_abort):
        hdrs = {
            "Content-Type":      "application/json",
            "x-api-key":         key,  # sweep-ok
            "anthropic-version": "2023-06-01",  # sweep-ok
            "anthropic-beta":    _API_BETAS,  # sweep-ok
        }
        data    = json.dumps(payload).encode()
        attempt = 0
        while True:
            req = urllib.request.Request(
                f"{_API_BASE}/v1/messages",
                data=data,
                method="POST",
                headers=hdrs,
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    self._last_rl = {k.lower(): v for k, v in resp.headers.items()}
                    return self._consume_urllib(resp, should_abort)
            except urllib.error.HTTPError as e:
                self._last_rl = {k.lower(): v for k, v in (e.headers or {}).items()}
                body = e.read().decode(errors="replace")
                # 429 -> chat()'s Haiku fallback; here only transient overloads.
                if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                    hint = _retry_after_secs(e.headers or {})
                    wait = (hint + 1) if hint is not None else _backoff(attempt)
                    if wait <= _MAX_WAIT:
                        attempt += 1
                        print(_lc["red"](f"[!] {payload.get('model', '')} overloaded ({e.code}) - "
                                         f"retry {attempt}/{_MAX_RETRIES} in {wait:.0f}s"))
                        if not _interruptible_sleep(wait, should_abort):
                            raise RuntimeError("aborted during retry wait") from None
                        continue
                raise RuntimeError(f"API {e.code}: {body[:400]}") from None
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                # Transport failure (dropped connection, TLS reset, DNS blip) - no
                # HTTP status. Retry with backoff like a 5xx; a bad message never
                # surfaces empty.
                if attempt < _MAX_RETRIES:
                    wait = _backoff(attempt)
                    attempt += 1
                    detail = str(getattr(e, "reason", None) or e) or type(e).__name__
                    print(_lc["red"](f"[!] {payload.get('model', '')} connection error "
                                     f"({detail[:80]}) - retry {attempt}/{_MAX_RETRIES} in {wait:.0f}s"))
                    if not _interruptible_sleep(wait, should_abort):
                        raise RuntimeError("aborted during retry wait") from None
                    continue
                raise RuntimeError(
                    f"request failed: {str(getattr(e, 'reason', None) or e) or type(e).__name__}"
                ) from None

    def _on_error(self, key, payload, should_abort):
        """429 fallback: swap to Haiku, strip params the fallback can't use, retry once."""
        haiku = self._haiku_model()
        orig  = payload["model"]
        if not haiku or orig == haiku:
            print(_lc["red"]("[!] rate limited on all models - check your API key limits"))
            raise RuntimeError("rate limited") from None

        retry_s = self._last_rl.get("retry-after", "")  # sweep-ok
        retry_note = f" (retry after {retry_s}s)" if retry_s else ""
        print(_lc["red"](f"[!] {orig} rate limited{retry_note} - retrying with Haiku"))  # sweep-ok

        hinfo = _ensure_models().get(haiku, {})
        payload = dict(payload)
        payload["model"] = haiku
        if not hinfo.get("can_think", False):
            payload.pop("thinking", None)
            payload.pop("context_management", None)
        if not hinfo.get("can_effort", False):
            payload.pop("output_config", None)
        try:
            return self._send(key, payload, should_abort)
        except RuntimeError as e2:
            if "429" in str(e2):
                print(_lc["red"]("[!] rate limited on all models - check your API key limits"))
                raise RuntimeError("rate limited") from None
            raise

    def chat(self, messages, tools, should_abort=None):
        key = _api_key()
        if not key:
            raise RuntimeError("no API key - use /provider login")

        system, api_msgs = _cvt_messages(messages)
        api_tools        = sorted(_cvt_tools(tools), key=lambda t: t.get("name", "")) if tools else []

        cur_model  = self._model()
        minfo      = _ensure_models().get(cur_model, {})
        max_out    = minfo.get("max_tokens", _MAX_TOKENS)
        max_in     = minfo.get("max_input_tokens", _CTX_DEFAULT)
        _est_toks  = len(json.dumps(messages)) // 4
        if _est_toks > max_in:
            raise RuntimeError(
                f"context too large for {cur_model} "
                f"(~{_est_toks:,} tokens vs {max_in:,} limit) - run /compact first"
            )

        # Prompt caching: stable breakpoints so repeat input re-reads at ~0.1x cost.
        # All session-scoped + 1h TTL. scope:"global" was removed - with tools present
        # the render order (tools -> system -> messages) means a global breakpoint on
        # system is not a true global prefix, and the API 400s. Session scope still
        # caches the re-read within a session (the main win). See the OAuth client.
        _cc_stable = {"type": "ephemeral", "ttl": "1h"}
        if system:
            sys_blocks = [{"type": "text", "text": system, "cache_control": _cc_stable}]
        else:
            sys_blocks = None
        if api_tools:
            cached_tools = list(api_tools)
            cached_tools[-1] = dict(cached_tools[-1], cache_control=_cc_stable)
        else:
            cached_tools = api_tools
        cached_msgs = list(api_msgs)
        if len(cached_msgs) >= 2:
            cached_msgs[-2] = _mark_cache(cached_msgs[-2])   # rolling recent breakpoint
        # 4th breakpoint: core tags the frozen post-handover summary message with
        # cache_boundary=True. The [system][summary] prefix is stable until the next
        # handover, so anchoring here caches the biggest static block at ~0.1x re-read.
        # Only the LAST-flagged message gets it (the vendor caps at 4 breakpoints; a 5th
        # 400s), and we strip the internal flag from every message before send (it is not
        # part of the API schema).
        _bnd = max((k for k, mm in enumerate(cached_msgs) if mm.get("cache_boundary")),
                   default=None)
        if _bnd is not None and _bnd != len(cached_msgs) - 2:
            cached_msgs[_bnd] = _mark_cache(cached_msgs[_bnd])
        cached_msgs = [{k: v for k, v in mm.items() if k != "cache_boundary"}
                       for mm in cached_msgs]

        payload = {
            "model":      cur_model,
            "max_tokens": max_out,
            "stream":     True,
            "messages":   cached_msgs,
        }
        if sys_blocks:   payload["system"] = sys_blocks
        if cached_tools: payload["tools"]  = cached_tools

        _can_think  = minfo.get("can_think",    False)
        _can_effort = minfo.get("can_effort",   False)
        _can_ctx    = minfo.get("can_ctx_mgmt", False)
        _thinking   = self.cfg.setting("thinking") or "off"
        if _can_think and _thinking == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
            if _can_ctx:
                payload["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}}]}
        elif _can_think and _thinking == "max":
            payload["thinking"] = {"type": "enabled", "budget_tokens": 16000}
            if _can_ctx:
                payload["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}}]}
        if _can_effort:
            payload["output_config"] = {"effort": self.cfg.setting("effort") or "low"}

        try:
            result = self._send(key, payload, should_abort)
        except RuntimeError as e:
            if "429" in str(e):
                result = self._on_error(key, payload, should_abort)
            else:
                raise
        self._sess_cache_read  += self.last_cache_read
        self._sess_cache_write += self.last_cache_write
        self._sess_cache_write_1h += self.last_cache_write_1h
        return result


def _make_client(cfg):
    return _ApiKeyClient(cfg)


# ----------------------------------------------------------------------------
# Provider hooks
# ----------------------------------------------------------------------------

def _available():
    return bool(_api_key())


def _autostart():
    return bool(_api_key()) and _cred_data().get("active", False)


def _on_activate(agent, cfg):
    _ensure_models(force=True)
    info = _ensure_models().get(cfg.active_model(), {})
    if cfg.setting("thinking") is None and info.get("can_think"):
        cfg.set_setting("thinking", "adaptive")
    if cfg.setting("effort") is None and info.get("can_effort"):
        cfg.set_setting("effort", "low")
    if _SHOW_BANNER:
        key     = _api_key()
        masked  = f"{key[:12]}...{key[-4:]}" if key and len(key) > 16 else "(env)"
        print(_lc["dim"](f"  API key auth  {masked}"))
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
        print(_lc["dim"]("Paste your API key (sk-ant-api03-...) and press Enter:"))
        try:
            raw = getpass.getpass(prompt="").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        m = re.search(r"sk-ant-\S+", raw)  # sweep-ok - extract key from any surrounding noise
        key = m.group(0) if m else raw
        if not key.startswith("sk-ant-"):  # sweep-ok
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
    red  = _lc["red"]

    def _co(s, pct):
        if pct >= 90: return f"\033[31m{s}\033[0m"
        if pct >= 50: return f"\033[33m{s}\033[0m"
        return f"\033[2;34m{s}\033[0m"

    cur_model  = cfg.active_model()
    cur_think  = cfg.setting("thinking") or "off"
    cur_effort = cfg.setting("effort")   or "low"
    lines = [bold(cur_model) + dim(f"   thinking: {cur_think}   effort: {cur_effort}")]

    si = getattr(agent, "session_in",  0) or 0
    so = getattr(agent, "session_out", 0) or 0
    cr = getattr(agent.client, "_sess_cache_read",  0) or 0
    cw = getattr(agent.client, "_sess_cache_write", 0) or 0
    cw1h = getattr(agent.client, "_sess_cache_write_1h", 0) or 0
    w1h_s = f" (1h {_sn(cw1h)})" if cw1h else ""
    cache_s = f"   cache r {_sn(cr)} w {_sn(cw)}{w1h_s}" if (cr or cw) else ""
    lines.append(dim(f"\n  session   in {si:,}   out {so:,}   total {si + so:,}{cache_s}"))

    rl = getattr(agent.client, "_last_rl", {})
    if rl:
        lines.append("")
        for label, lim_key, rem_key, rst_key in (
            ("requests",     "anthropic-ratelimit-requests-limit",      # sweep-ok
                             "anthropic-ratelimit-requests-remaining",  # sweep-ok
                             "anthropic-ratelimit-requests-reset"),     # sweep-ok
            ("input tok",    "anthropic-ratelimit-input-tokens-limit",    # sweep-ok
                             "anthropic-ratelimit-input-tokens-remaining",# sweep-ok
                             "anthropic-ratelimit-input-tokens-reset"),   # sweep-ok
            ("output tok",   "anthropic-ratelimit-output-tokens-limit",   # sweep-ok
                             "anthropic-ratelimit-output-tokens-remaining",# sweep-ok
                             "anthropic-ratelimit-output-tokens-reset"),  # sweep-ok
        ):
            lim = rl.get(lim_key, "")
            rem = rl.get(rem_key, "")
            rst = rl.get(rst_key, "")
            if lim and rem:
                try:
                    lv, rv = int(lim), int(rem)
                    pct    = max(0, (lv - rv) / lv * 100) if lv else 0
                    rst_s  = f"  resets {_fmt_reset(rst)}" if rst else ""
                    lines.append(
                        f"  {dim(f'{label:<12}')}  {_co(_bar(pct), pct)}"
                        f"  {_co(f'{pct:>3.0f}%', pct)}"
                        f"  {dim(f'{_sn(rv)}/{_sn(lv)} remaining{rst_s}')}"
                    )
                except (ValueError, ZeroDivisionError):
                    pass

    if not rl:
        lines.append(dim("\n  (rate limit data available after first inference)"))

    lines.append("\n" + _privacy_note())
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

PROVIDER = {
    "name":           _PROVIDER,
    "description":    "Anthropic API key (direct; any tier)",  # sweep-ok
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
    "tag":            "api",
}


def setup(lc, cfg):
    global _CONFIG_DIR
    _lc.update(lc)
    _CONFIG_DIR = lc.get("CONFIG_DIR")
