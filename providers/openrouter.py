import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _dbg_swallow():
    """Debug bucket shim: hands a swallowed exception to lean-coder's _dbg_swallow
    (a no-op unless LEANCODER_DEBUG=1). Never raises."""
    for _m in ("__main__", "lean_coder"):
        _f = getattr(sys.modules.get(_m), "_dbg_swallow", None)
        if _f and _f is not _dbg_swallow:
            try:
                _f(2)
            except Exception:
                pass
            return


# OpenRouter - an OpenAI-compatible Chat Completions gateway over many providers.
#   docs: https://openrouter.ai/docs/api-reference/chat-completion  # sweep-ok
# Auth is a Bearer key (sk-or-v1-...). Because the wire format is OpenAI-compatible
# and lean-coder's internal message format already is too, conversion is nearly
# an identity (just stringify tool-call args + tag tool defs as type:function).
_API_BASE    = "https://openrouter.ai/api/v1"  # sweep-ok
_MAX_TOKENS  = 8192
_MIN_ROOM    = 1024        # below this much reply room, ask for /compact instead
_CTX_DEFAULT = 128_000
_PROVIDER    = "openrouter"  # sweep-ok

_CONFIG_DIR = None   # set in setup() from lc["CONFIG_DIR"]
_MODEL_DATA = {}     # model_id -> {max_input_tokens, max_tokens, tools, reason}
_lc         = {}     # display helpers injected by setup()
_cfg        = None   # read-only cfg ref (for the /model listing filter)

# Privacy posture shown at the point of use (on connect + in /usage).
_PRIVACY    = "OpenRouter won't train on prompts, but free models route to upstream providers whose terms vary - avoid sensitive code"
_PRIVACY_HI = True   # medium risk -> yellow
# Show the on-activate identity + privacy banner? Off by default (the core
# 'provider: <name>  model: <model>' line already announces the backend). NOTE:
# this provider's privacy note is medium-risk - consider True to keep it visible.
_SHOW_BANNER = False

# Published free-tier caps (openrouter.ai/docs/api_reference/limits): every
# :free model shares 20 req/min and 50 req/day per account, 1000/day once the
# account has ever bought $10+ of credits. The x-ratelimit-limit header reports
# the real daily figure and overrides this. Paid models: no OpenRouter cap.
_RATE_DEFAULTS = {":free": {"rpm": 20, "rpd": 50}, "openrouter/free": {"rpm": 20, "rpd": 50}}
_FREE_ROUTER   = "openrouter/free"   # routes to whichever free model is up
_EMPTY_RETRIES = 1        # re-asks after an empty reply (on a free fallback model)
_REASON_MIN_OUT = 16384   # floor when /models under-reports a model's output limit
_LEDGER        = None


def _ledger():
    """Client-side request ledger (core RateLedger), created lazily after setup."""
    global _LEDGER
    if _LEDGER is None and _lc.get("RateLedger"):
        _LEDGER = _lc["RateLedger"](_PROVIDER, _RATE_DEFAULTS, day_reset="utc",
                                    shared=("rpm", "rpd"))
    return _LEDGER


def _free_fallback(model, cfg):
    """Next model to try when a FREE model is throttled upstream: the first
    or_fallback entry that is free, else the openrouter/free router. Paid models
    never hop (that would change cost); returns None for them."""
    if not (model.endswith(":free") or model == _FREE_ROUTER):
        return None
    try:
        fb = (cfg.setting("or_fallback") or "").replace(",", " ").split()
    except Exception:
        fb = []
    for m in fb:
        if m != model and (m.endswith(":free") or m == _FREE_ROUTER):
            return m
    return _FREE_ROUTER if model != _FREE_ROUTER else None


class _UpstreamError(Exception):
    """An error delivered INSIDE a 200 stream (OpenRouter forwards an upstream
    provider's 429/5xx as an error chunk / finish_reason "error")."""
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


# ----------------------------------------------------------------------------
# Credentials: ~/.config/leancoder/openrouter.json
# {"key": "sk-or-v1-...", "active": false}
# Key resolution order: OPENROUTER_API_KEY > keyfile  # sweep-ok
# ----------------------------------------------------------------------------

def _cred_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "openrouter.json"  # sweep-ok

def _models_cache_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "model-cache-openrouter.json"

def _api_key():
    v = os.environ.get("OPENROUTER_API_KEY", "")  # sweep-ok
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
        _dbg_swallow()

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
    """GET /models. No auth needed to list, but we send the key anyway. Stores the
    served context length (top_provider.context_length) - it can be far smaller than
    the headline context_length, and is what the endpoint actually accepts."""
    try:
        req = urllib.request.Request(
            f"{_API_BASE}/models",
            headers={"Authorization": f"Bearer {key}"} if key else {},  # sweep-ok
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        out = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            if not mid:
                continue
            tp      = m.get("top_provider") or {}
            ctx     = tp.get("context_length") or m.get("context_length") or _CTX_DEFAULT
            max_out = tp.get("max_completion_tokens") or _MAX_TOKENS
            params  = m.get("supported_parameters") or []
            out.append({
                "id":               mid,
                "max_input_tokens": ctx,
                "max_tokens":       max_out,
                "tools":            "tools" in params,
                "reason":           "reasoning" in params,
                "free":             mid.endswith(":free") or mid == _FREE_ROUTER,
            })
        if out:
            try:
                _models_cache_path().parent.mkdir(parents=True, exist_ok=True)
                _models_cache_path().write_text(json.dumps({"models": out}))
            except Exception:
                _dbg_swallow()
            return out
    except Exception:
        _dbg_swallow()
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
    """The /model picker. OpenRouter has 300+ models, ~250 tool-capable. A coding
    agent needs tool calling, so only tool-capable models are listed. Free tier by
    default (`tier=free`): only `:free` models - `/set tier all` adds paid
    models. `/set or_filter <substring>` narrows WITHIN the current tier. New models
    surface automatically (live fetch)."""
    show_all = True
    try:
        show_all = ((_cfg.setting("tier") or "free") != "free") if _cfg else False  # sweep-ok
    except Exception:
        show_all = True
    sel = [m for m in _ensure_models().values()
           if m.get("tools") and (show_all or m.get("free"))]
    flt = ""
    try:
        flt = (_cfg.setting("or_filter") or "").strip().lower() if _cfg else ""  # sweep-ok
    except Exception:
        flt = ""
    if flt:
        sel = [m for m in sel if flt in m["id"].lower()] or sel
    return [m["id"] for m in sel]


def _tool_support(model):
    """True unless /models says the model lacks tool calling. Unknown ids (not in
    the list, e.g. offline) are assumed capable rather than silently degraded."""
    m = _ensure_models().get(model)
    return True if m is None else bool(m.get("tools"))


def _model_status(model):
    # Shown dimmed next to the model in /model.
    return None if _tool_support(model) else "chat-only (no tool calling)"


def _context_window(model):
    return _ensure_models().get(model, {}).get("max_input_tokens", _CTX_DEFAULT)


def _capabilities(model):
    caps = {"tier": ["free", "all"]}     # free (default) limits to :free only; all adds paid
    if _ensure_models().get(model, {}).get("reason"):
        caps["effort"] = ["off", "low", "med", "high"]
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
# 429 retry: free models are heavily throttled upstream. A genuine throttle ships
# a retry delay (Retry-After header, or error.metadata.retry_after_seconds in the
# body); we honor it (+1s) and retry. No delay hint -> surface immediately rather
# than guess-and-hammer.
# ----------------------------------------------------------------------------

_MAX_RETRIES = 3
_MAX_WAIT    = 75   # seconds; don't hang the REPL on a long throttle
_RETRY_CODES = (429, 502, 503, 504)   # 429 throttle + transient server/gateway errors


def _backoff(attempt):
    """Exponential backoff (2s, 4s, 8s, ...) for transient errors with no
    server-supplied delay. Capped at _MAX_WAIT."""
    return min(2.0 * (2 ** attempt), _MAX_WAIT)


def _daily_cap_msg(body):
    """If a 429 is OpenRouter's account-wide free-models-per-day cap (not a transient
    upstream throttle), return a clean, actionable message with the reset time.
    None otherwise. The daily cap won't clear on a retry - only at reset."""
    try:
        err  = json.loads(body).get("error", {})
        msg  = err.get("message", "")
        hdrs = (err.get("metadata", {}) or {}).get("headers", {}) or {}
        if "per-day" not in msg and "free-models-per-day" not in msg:
            return None
        lim   = hdrs.get("X-RateLimit-Limit", "")
        reset = hdrs.get("X-RateLimit-Reset", "")
        when  = ""
        if reset:
            try:
                when = " - resets " + time.strftime("%H:%M %d%b", time.localtime(int(reset) / 1000.0))
            except Exception:
                _dbg_swallow()
        cap = f" ({lim}/day)" if lim else ""
        return (f"OpenRouter free daily limit reached{cap}{when}. Add credits at "
                f"openrouter.ai/credits for 1000/day, or switch to groq (/provider groq).")
    except Exception:
        return None


def _retry_after_secs(headers, body):
    ra = headers.get("Retry-After") or headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except (TypeError, ValueError):
            _dbg_swallow()
    try:
        meta = json.loads(body).get("error", {}).get("metadata", {}) or {}
        if meta.get("retry_after_seconds") is not None:
            return float(meta["retry_after_seconds"])
        h = meta.get("headers", {}) or {}
        if h.get("Retry-After"):
            return float(h["Retry-After"])
    except Exception:
        _dbg_swallow()
    return None


def _interruptible_sleep(secs, should_abort, why="waiting"):
    vw = _lc.get("_visible_wait")
    if vw:      # live countdown, so a wait is never a silent blank screen
        return vw(secs, why, should_abort)
    return _plain_sleep(secs, should_abort)


def _plain_sleep(secs, should_abort):
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
# Message / tool conversion (lean-coder internal -> OpenAI/OpenRouter)
#   The internal format already IS OpenAI-shaped; we only normalise tool-call
#   arguments to a JSON string and tag tool defs with type:"function".
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
                _c = r.get("content", "")
                if not isinstance(_c, str):    # a loaded/synthesised history may carry
                    _c = "" if _c is None else str(_c)   # a non-string; the API wants text
                out.append({
                    "role":         "tool",
                    "tool_call_id": tcid,
                    "content":      _c,
                })
                ci += 1
                i  += 1
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------

class _OpenRouterClient:
    uncapped_ctx = True

    def __init__(self, cfg):
        self.cfg               = cfg
        self.last_out_tokens   = None
        self.last_cache_read   = 0
        self.last_cache_write  = 0    # OpenRouter reports cache reads, not writes
        self._sess_cache_read  = 0
        self._sess_cache_write = 0
        self._last_rl          = {}   # x-ratelimit-{limit,remaining,reset} - free daily cap

    def _capture_rl(self, headers):
        """Stash the free-models-per-day counter from response headers (present on
        both 200s and the 429 cap). No extra API call - just reads what we got."""
        if not headers:
            return
        def _i(k):
            try:
                v = headers.get(k)
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        lim = _i("x-ratelimit-limit")
        if lim:
            self._last_rl = {"limit": lim,
                             "remaining": _i("x-ratelimit-remaining") or 0,
                             "reset": _i("x-ratelimit-reset")}
            # The header carries the ACCOUNT's real free daily cap (50 or 1000);
            # teach the ledger so its predictions use the real number.
            led = _ledger()
            if led and lim >= 50:
                led.learn("*", rpd=lim)

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
        tool_acc      = {}     # index -> {id, name, args}
        order         = []     # tool-call indices in arrival order
        prompt_total  = None
        cached        = 0
        out_tokens    = None
        printed       = False
        aborted       = False
        md            = _lc["MarkdownStream"](sys.stdout.write)

        spin = _lc["Spinner"]("thinking", _lc["THINK_FRAMES"]).start()
        reason_chars = 0
        self.last_finish = None
        self.last_served = None
        try:
            for raw in _lc["stream_tiered"](resp, self.cfg):
                if should_abort and should_abort():
                    aborted = True
                    break
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):     # skip ": OPENROUTER PROCESSING" keep-alives
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                # Upstream failure smuggled into a 200 stream: a top-level "error"
                # chunk, or a choice with finish_reason "error" + choice.error.
                err = obj.get("error")
                if not err:
                    for ch in obj.get("choices", []):
                        if ch.get("finish_reason") == "error" or ch.get("error"):
                            err = ch.get("error") or {"message": "upstream error"}
                            break
                if err:
                    try:
                        code = int(err.get("code")) if isinstance(err, dict) else 0
                    except (TypeError, ValueError):
                        code = 0
                    msg = err.get("message", "") if isinstance(err, dict) else str(err)
                    if not printed and not tool_acc:
                        raise _UpstreamError(code or 502, msg or "upstream error")
                    # Already streamed part of a reply: keep it, tell the operator.
                    print(_lc["yellow"](f"\n[!] stream cut by upstream error {code}: {msg[:120]}"))
                    break

                if obj.get("model"):
                    self.last_served = obj["model"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    if ch.get("finish_reason"):
                        self.last_finish = ch["finish_reason"]
                    # Reasoning streams before any visible text; show progress on
                    # the spinner so a long think isn't a frozen screen.
                    rz = delta.get("reasoning") or delta.get("reasoning_content")
                    if rz and not printed:
                        reason_chars += len(rz)
                        spin.label = f"reasoning ~{reason_chars // 4:,} tok"
                    txt   = delta.get("content")
                    if txt:
                        if not printed:
                            spin.stop()
                            sys.stdout.write(_lc["blue"]((_lc.get("GLYPH") or {}).get("bullet", "*") + " "))
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

                um = obj.get("usage")
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
        self._last_in_total = prompt_total or 0
        # prompt_tokens is the FULL input incl. cache hits; core sums
        # prompt_eval + cache_read + cache_write, so return only the uncached part.
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
        led     = _ledger()
        while True:
            model = payload.get("model", "")
            data  = json.dumps(payload).encode()
            if led:
                warn = led.preflight(model, self.cfg)
                if warn and attempt == 0:
                    print(_lc["yellow"](f"[!] {model}: {warn}"))
            self._last_in_total = 0
            self.last_out_tokens = None
            req = urllib.request.Request(
                f"{_API_BASE}/chat/completions",
                data=data, method="POST",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {key}",  # sweep-ok
                },
            )
            try:
                with urllib.request.urlopen(
                        req,
                        timeout=(getattr(self.cfg, "gen_connect_timeout", None) or 600)) as resp:
                    self._capture_rl(resp.headers)
                    result = self._consume(resp, should_abort)
                    if led:
                        led.record(model, self._last_in_total, self.last_out_tokens or 0, 200)
                    return result
            except _UpstreamError as e:
                # Upstream provider throttled/failed behind a 200. Not the account
                # cap, so retrying (or hopping to another free model) usually works.
                if led:
                    led.record(model, 0, 0, 429 if e.code == 429 else e.code)
                if attempt < _MAX_RETRIES:
                    attempt += 1
                    nxt = _free_fallback(model, self.cfg)
                    if nxt and nxt != model:
                        print(_lc["red"](f"[!] {model} upstream {e.code} ({str(e)[:80]}) - "
                                         f"retry {attempt}/{_MAX_RETRIES} on {nxt}"))
                        payload = dict(payload, model=nxt)
                        payload.pop("models", None)
                        continue
                    wait = _backoff(attempt - 1)
                    print(_lc["red"](f"[!] {model} upstream {e.code} ({str(e)[:80]}) - "
                                     f"retry {attempt}/{_MAX_RETRIES} in {wait:.0f}s"))
                    if not _interruptible_sleep(wait, should_abort, f"{model} retry {attempt}/{_MAX_RETRIES}"):
                        raise RuntimeError("aborted during retry wait") from None
                    continue
                raise RuntimeError(f"{model} upstream error {e.code}: {str(e)[:300]} - "
                                   f"try another :free model or {_FREE_ROUTER}") from None
            except urllib.error.HTTPError as e:
                self._capture_rl(e.headers)
                body = e.read().decode(errors="replace")
                if led:
                    led.record(model, 0, 0, e.code)
                if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                    hint = _retry_after_secs(e.headers or {}, body)
                    if e.code == 429:
                        wait = (hint + 1) if hint is not None else None  # retry only with a hint
                    else:
                        wait = (hint + 1) if hint is not None else _backoff(attempt)
                    if wait is not None and wait <= _MAX_WAIT:
                        attempt += 1
                        reason = "rate limited" if e.code == 429 else f"unavailable ({e.code})"
                        print(_lc["red"](f"[!] {model} {reason} - retry {attempt}/{_MAX_RETRIES} "
                                         f"in {wait:.0f}s"))
                        if not _interruptible_sleep(wait, should_abort, f"{model} retry {attempt}/{_MAX_RETRIES}"):
                            raise RuntimeError("aborted during retry wait") from None
                        continue
                if e.code == 429:
                    dm = _daily_cap_msg(body)
                    if dm:
                        raise RuntimeError(dm) from None
                    # Per-model upstream throttle with no retry hint: hop to
                    # another free model rather than fail the turn.
                    nxt = _free_fallback(model, self.cfg)
                    if nxt and nxt != model and attempt < _MAX_RETRIES:
                        attempt += 1
                        print(_lc["red"](f"[!] {model} rate limited upstream - "
                                         f"retry {attempt}/{_MAX_RETRIES} on {nxt}"))
                        payload = dict(payload, model=nxt)
                        payload.pop("models", None)
                        continue
                raise RuntimeError(f"API {e.code}: {body[:400]}") from None
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                # Transport failure (dropped connection, TLS reset, DNS blip) - no
                # HTTP status. Retry with backoff; a bad message never surfaces empty.
                if attempt < _MAX_RETRIES:
                    wait = _backoff(attempt)
                    attempt += 1
                    detail = str(getattr(e, "reason", None) or e) or type(e).__name__
                    print(_lc["red"](f"[!] {model} connection error ({detail[:80]}) - "
                                     f"retry {attempt}/{_MAX_RETRIES} in {wait:.0f}s"))
                    if not _interruptible_sleep(wait, should_abort, f"{model} retry {attempt}/{_MAX_RETRIES}"):
                        raise RuntimeError("aborted during retry wait") from None
                    continue
                raise RuntimeError(
                    f"request failed: {str(getattr(e, 'reason', None) or e) or type(e).__name__}"
                ) from None

    def chat(self, messages, tools, should_abort=None):
        key = _api_key()
        if not key:
            raise RuntimeError("no API key - use /provider login")

        cur_model = self._model()
        minfo     = _ensure_models().get(cur_model, {})
        # Free-tier lock (default): refuse a known paid model so a credited key can't
        # spend by accident. Lift it deliberately with `/set tier all`.
        show_all  = (self.cfg.setting("tier") or "free") != "free"
        is_free   = (cur_model == _FREE_ROUTER or cur_model.endswith(":free")
                     or bool(minfo.get("free")))
        if not is_free and not show_all:
            raise RuntimeError(
                f"{cur_model} is a paid model and tier=free - run '/set tier all' to "
                f"allow paid models, or pick a :free model with /model"
            )
        # Never cap a reply below what the model can produce: a truncated reply
        # wastes the whole turn (the only fix is to ask again). Ask for the
        # model's full output limit, shrunk only to fit the context window.
        max_out   = max(minfo.get("max_tokens") or _MAX_TOKENS, _REASON_MIN_OUT)
        max_in    = minfo.get("max_input_tokens", _CTX_DEFAULT)
        send_tools = tools if (tools and _tool_support(cur_model)) else None
        _mt       = _lc.get("messages_tokens")
        _est_toks = _mt(messages, send_tools) if _mt else len(json.dumps(messages)) // 4
        if _mt is None and send_tools:
            _est_toks += len(json.dumps(send_tools)) // 4
        max_out = min(max_out, max_in - _est_toks)
        if max_out < _MIN_ROOM:
            raise RuntimeError(
                f"context too large for {cur_model} "
                f"(~{_est_toks:,} of {max_in:,} tokens used, no room to reply) - run /compact first"
            )

        payload = {
            "model":         cur_model,
            "messages":      _cvt_messages(messages),
            "max_tokens":    max_out,
            "stream":        True,
            "stream_options": {"include_usage": True},
        }
        # Models whose /models entry lacks "tools" in supported_parameters 400 on a
        # tools array; send none so they answer chat-only (flagged in /model).
        if send_tools:
            payload["tools"] = _cvt_tools(send_tools)

        is_free_m = cur_model.endswith(":free") or cur_model == _FREE_ROUTER
        if minfo.get("reason") or cur_model == _FREE_ROUTER:
            eff = self.cfg.setting("effort") or ("low" if is_free_m else None)
            if eff and eff != "off":
                payload["reasoning"] = {"effort": "medium" if eff == "med" else eff}
            elif eff == "off":
                payload["reasoning"] = {"enabled": False}

        # Opt-in cross-model fallback (OpenRouter-native): `/set or_fallback
        # "openai/gpt-5,google/gemini-3-pro"` -> OpenRouter walks the list top to
        # bottom when the active model 5xx/429s/refuses, no app-side retry needed.
        # The active model leads the chain; you pay whichever model actually serves.
        # Off by default (unset): swapping models mid-session changes quality/cost,
        # so it must be chosen deliberately.
        fb = (self.cfg.setting("or_fallback") or "").strip()
        if fb:
            extra = [m.strip() for m in fb.replace(",", " ").split() if m.strip()
                     and m.strip() != cur_model]
            if extra:
                payload["models"] = [cur_model] + extra
                payload["route"]  = "fallback"

        # Empty-reply recovery. Free reasoning models can spend the whole output
        # budget thinking and return nothing (measured: 3/12 turns on
        # openrouter/free). Retry once on a free model, saying so out loud.
        tried = {cur_model}
        for _ in range(_EMPTY_RETRIES + 1):
            result = self._send(key, payload, should_abort)
            self._sess_cache_read  += self.last_cache_read
            self._sess_cache_write += self.last_cache_write
            a, _pe, ab = result
            if ab or (a.get("content") or "").strip() or a.get("tool_calls"):
                return result
            served = self.last_served or payload["model"]
            why = ("used its whole output budget on reasoning"
                   if self.last_finish == "length" else
                   f"returned nothing (finish: {self.last_finish or 'none'})")
            nxt = _free_fallback(payload["model"], self.cfg)
            if not nxt or (nxt in tried and nxt != _FREE_ROUTER):
                print(_lc["yellow"](f"[!] {served} {why} - no free fallback to try; "
                                    f"retry, or /model to switch"))
                return result
            tried.add(nxt)
            print(_lc["yellow"](f"[!] {served} {why} - retrying on {nxt}"))
            payload = dict(payload, model=nxt)
            payload.pop("models", None)
        return result


def _make_client(cfg):
    return _OpenRouterClient(cfg)


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
    if cfg.setting("effort") is None and _ensure_models().get(cfg.active_model(), {}).get("reason"):
        # Free reasoning models at default effort can loop for 4k+ tokens and
        # return nothing; low effort measured 6/6 clean answers. Paid: med.
        m = cfg.active_model()
        cfg.set_setting("effort", "low" if (m.endswith(":free") or m == _FREE_ROUTER) else "med")
    if cfg.setting("tier") is None:
        cfg.set_setting("tier", "free")
    if _SHOW_BANNER:
        key    = _api_key()
        masked = f"{key[:12]}...{key[-4:]}" if key and len(key) > 16 else "(env)"
        print(_lc["dim"](f"  OpenRouter key  {masked}"))
    if (cfg.setting("tier") or "free") != "all":
        print(_lc["dim"]("  free tier: only :free models listed; /set tier all for paid, /set or_filter <text> to narrow"))
    else:
        print(_lc["dim"]("  tier all: paid + free models listed; /set tier free for :free only, /set or_filter <text> to narrow"))
    fb = (cfg.setting("or_fallback") or "").strip()
    if fb:
        print(_lc["dim"](f"  fallback chain: {cfg.active_model()} -> {fb}  (/set or_fallback '' to disable)"))
    else:
        print(_lc["dim"]("  /set or_fallback 'anthropic/claude-sonnet-4.5,openai/gpt-5' for auto cross-model fallback on outage/ratelimit"))
    if _SHOW_BANNER:
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
        print(_lc["dim"]("Paste your OpenRouter API key (sk-or-v1-...) and press Enter:"))
        try:
            raw = getpass.getpass(prompt="").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        m   = re.search(r"sk-or-\S+", raw)  # sweep-ok - extract key from surrounding noise
        key = m.group(0) if m else raw
        if not key.startswith("sk-or-"):  # sweep-ok
            print(_lc["red"]("[!] does not look like an OpenRouter key"))
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
    # Free-tier meters (min / day) from the client-side ledger. The server's own
    # x-ratelimit-* daily counter, when fresh, is the source of truth for "day".
    model = cfg.active_model()
    led   = _ledger()
    out   = led.meters(model, cfg) if led else None
    rl    = getattr(agent.client, "_last_rl", {}) or {}
    lim, reset = rl.get("limit"), rl.get("reset")
    if lim and not (reset and reset / 1000.0 < time.time()):
        rem = rl.get("remaining") or 0
        resets_at = ""
        if reset:
            try:
                resets_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(reset / 1000.0))
            except Exception:
                _dbg_swallow()
        day = {"label": "day", "pct": max(0.0, (lim - rem) / lim * 100.0),
               "resets_at": resets_at, "reset_on_warn": True}
        ms = [m for m in (out or {}).get("meters", []) if m["label"] != "day"] + [day]
        out = {"meters": ms}
    return out


def _detail(agent, cfg, arg=""):
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
    led = _ledger()
    if led and led.caps(cur_model, cfg)["rpd"]:
        lines.append(dim("\n" + "\n".join(led.detail_lines(cur_model, cfg))))
        lines.append(dim("  free caps: 20/min, 50/day (1000/day after a one-time $10 top-up);"
                         " upstream hosts may also throttle"))
    lines.append(dim("  (OpenRouter - see openrouter.ai/credits for balance)"))  # sweep-ok
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

PROVIDER = {
    "name":           _PROVIDER,
    "description":    "OpenRouter (OpenAI-compatible gateway; API key)",  # sweep-ok
    "make_client":    _make_client,
    "list_models":    _list_models,
    "context_window": _context_window,
    "capabilities":   _capabilities,
    "model_status":   _model_status,
    "tool_support":   _tool_support,   # core forces chat-only for these
    "available":      _available,
    "autostart":      _autostart,
    "on_activate":    _on_activate,
    "on_deactivate":  _on_deactivate,
    "usage":          _usage,
    "login":          _login,
    "clear":          _clear,
    "detail":         _detail,
    "tag":            "or",
}


def setup(lc, cfg):
    global _CONFIG_DIR, _cfg
    _lc.update(lc)
    _CONFIG_DIR = lc.get("CONFIG_DIR")
    _cfg = cfg
