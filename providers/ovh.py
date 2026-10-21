import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# OVHcloud AI Endpoints - OpenAI-compatible Chat Completions, ANONYMOUS free lane.
#   No account and NO API KEY, ever: this provider never sends an Authorization
#   header, so nothing can be billed (a keyed OVH project is pay-as-you-go; the
#   /models prices are that keyed rate and do not apply to the anonymous lane).
#   Measured 21 Oct 2026: 2 requests/minute PER MODEL per IP (each model has its own
#   bucket), 429 carries retry-after + x-ratelimit-limit-minute headers. Served by
#   vLLM; streaming + tool calls work. Hosted in EU data centres.
#   Built from providers/kilo.py (same wire format).
_API_BASE      = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"  # sweep-ok
_MAX_TOKENS    = 8192
_MIN_ROOM      = 1024        # below this much reply room, ask for /compact instead
_OUT_CAP       = 32768       # per-turn reply ceiling even when /models reports far more
_CTX_DEFAULT   = 131_072
_PROVIDER      = "ovh"  # sweep-ok
_DEFAULT_MODEL = "Qwen3.5-397B-A17B"     # biggest coder in the catalogue; ~2.6s tool call
_PER_MIN       = 2                       # anonymous: requests per minute per model per IP
_WINDOW        = 60.0

# Rotation when NOT pinned: a 429 / full minute on the active model moves to the next
# one with a free slot. Order = coding strength first (OVH lists no benchmarks; ranked
# by model size/family). Override with /provider set ovh_fallback "a,b,c".
_FALLBACK_CHAIN = ["Qwen3.5-397B-A17B", "gpt-oss-120b", "Qwen3.6-27B", "Qwen3.8-27B",
                   "Meta-Llama-3_3-70B-Instruct", "Mistral-Small-3.2-24B-Instruct-2506"]
# /models has no capability flags. Measured 21 Oct 2026: every chat model calls tools
# except the vision model (400 "feature 'tool calls' is not currently supported").
_NO_TOOLS  = {"Qwen2.5-VL-72B-Instruct"}
# Not chat models (embeddings, speech, image, safety classifiers): never listed.
_NOT_CHAT  = re.compile(r"embed|bge|whisper|tts|stable-diffusion|guard", re.I)

_CONFIG_DIR = None   # set in setup() from lc["CONFIG_DIR"]
_MODEL_DATA = {}     # model_id -> {max_input_tokens, max_tokens, tools}
_lc         = {}     # display helpers injected by setup()
_cfg        = None

_PRIVACY    = ("anonymous free lane, EU-hosted (OVHcloud); no account ties prompts to you,"
               " but avoid sensitive code on any free endpoint")
_PRIVACY_HI = False
_LEDGER     = None


def _ledger():
    """Client-side request ledger (core RateLedger), shared on disk
    (ratelog-ovh.json) so parallel lean-coder processes on this IP see each other."""
    global _LEDGER
    if _LEDGER is None and _lc.get("RateLedger"):
        _LEDGER = _lc["RateLedger"](_PROVIDER, {"*": {"rpm": _PER_MIN}}, day_reset="utc")
    return _LEDGER


def _sync_ledger():
    led = _ledger()
    if led:
        try:
            led._load()
        except Exception:
            pass
    return led


def _record(model, in_tok, out_tok, status):
    led = _sync_ledger()          # merge before append, so we don't clobber peers
    if led:
        led.record(model, in_tok, out_tok, status)


def _minute(model):
    """Accepted (non-429) requests to `model` in the last 60s -> (count, oldest ts)."""
    led = _sync_ledger()
    if not led:
        return 0, None
    cut = time.time() - _WINDOW
    ts = sorted(e[0] for e in led.events if e[0] >= cut and e[1] == model and e[4] != 429)
    return len(ts), (ts[0] if ts else None)


def _slot_wait_secs(model):
    """Seconds until `model` has a free slot (0 = send now). Rolling 60s from our own
    log - never earlier than OVH's real reset, so pacing cannot trip a 429 itself."""
    n, oldest = _minute(model)
    if n < _PER_MIN or oldest is None:
        return 0
    return max(0.0, oldest + _WINDOW - time.time()) + 1


def _pinned(cfg):
    """ovh_pin on: never switch models (benchmarks/certs). A full minute or a 429
    waits for the slot on the SAME model; a withdrawn model fails loudly."""
    try:
        return str(cfg.setting("ovh_pin") or "off").lower() in ("on", "true", "1", "yes")
    except Exception:
        return False


def _chain(cfg):
    try:
        fb = (cfg.setting("ovh_fallback") or "").replace(",", " ").split()
    except Exception:
        fb = []
    return fb or _FALLBACK_CHAIN


def _next_free_model(model, cfg, tried=()):
    """Unpinned rotation: the next chain model (after `model`) that has a free slot
    right now and is still in the live list. None = everyone is busy, or pinned."""
    if _pinned(cfg):
        return None
    for m in _chain(cfg):
        if m == model or m in tried:
            continue
        if _MODEL_DATA and m not in _MODEL_DATA:
            continue
        if not _tool_support(m):
            continue
        if _slot_wait_secs(m) <= 0:
            return m
    return None


class _UpstreamError(Exception):
    """An error delivered INSIDE a 200 stream (the gateway forwards an upstream
    provider's 429/5xx as an error chunk / finish_reason "error")."""
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code



# ----------------------------------------------------------------------------
# State: ~/.config/leancoder/ovh.json  {"active": bool}  - NO key is ever stored
# ----------------------------------------------------------------------------

def _state_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "ovh.json"  # sweep-ok

def _models_cache_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "model-cache-ovh.json"

def _state():
    try:
        return json.loads(_state_path().read_text())
    except Exception:
        return {}

def _save_state(data):
    try:
        _state_path().parent.mkdir(parents=True, exist_ok=True)
        _state_path().write_text(json.dumps(data, indent=2))
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Model list
# ----------------------------------------------------------------------------

def _load_models_cache():
    try:
        return json.loads(_models_cache_path().read_text()).get("models", [])
    except Exception:
        return []


def _fetch_models():
    """GET /models (public). Keeps chat models only; context_length and
    max_completion_tokens are what vLLM actually serves."""
    try:
        req = urllib.request.Request(f"{_API_BASE}/models", headers={"User-Agent": "lean-coder"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        out = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            ctx = m.get("context_length") or 0
            if not mid or _NOT_CHAT.search(mid) or not ctx or not m.get("max_completion_tokens"):
                continue
            out.append({
                "id":               mid,
                "max_input_tokens": ctx,
                "max_tokens":       m.get("max_completion_tokens") or _MAX_TOKENS,
                "tools":            mid not in _NO_TOOLS,
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
    _MODEL_DATA = {m["id"]: m for m in _fetch_models()}
    return _MODEL_DATA


def _list_models():
    """Tool-capable chat models, strongest first (chain order), then the rest."""
    ms = [m for m in _ensure_models().values() if m.get("tools")]
    rank = {m: i for i, m in enumerate(_FALLBACK_CHAIN)}
    ms.sort(key=lambda m: (rank.get(m["id"], 99), m["id"]))
    return [m["id"] for m in ms]


def _tool_support(model):
    m = _ensure_models().get(model)
    return (model not in _NO_TOOLS) if m is None else bool(m.get("tools"))


def _model_status(model):
    return None if _tool_support(model) else "chat-only (no tool calling)"


def _context_window(model):
    return _ensure_models().get(model, {}).get("max_input_tokens", _CTX_DEFAULT)


def _capabilities(model):
    # ovh_pin: on = never switch models (a busy minute waits for the slot) - for
    # benchmark/cert runs. off (default) = rotate down the chain to a model with a slot.
    return {"ovh_pin": ["off", "on"]}


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------

def _privacy_note():
    paint = (_lc.get("yellow") if _PRIVACY_HI else _lc.get("dim")) or (lambda s: s)
    return paint("  privacy: " + _PRIVACY)


def _fmt_secs(s):
    s = int(max(0, s))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


# ----------------------------------------------------------------------------
# Retry helpers
# ----------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_CODES = (502, 503, 504)


def _backoff(attempt):
    return min(2.0 * (2 ** attempt), 60.0)


def _retry_after_secs(headers):
    for h in ("retry-after", "Retry-After", "ratelimit-reset", "RateLimit-Reset"):
        v = headers.get(h)
        if v:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
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
# Message / tool conversion (lean-coder internal -> OpenAI wire format)
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

class _OvhClient:
    uncapped_ctx = True

    def __init__(self, cfg):
        self.cfg               = cfg
        self.last_out_tokens   = None
        self.last_cache_read   = 0
        self.last_cache_write  = 0
        self._sess_cache_read  = 0
        self._sess_cache_write = 0
        self.last_served       = None

    def _model(self):
        return self.cfg.active_model() or _DEFAULT_MODEL

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
                if not line.startswith("data:"):     # skip keep-alives / comments
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

    def _wait(self, secs, why, should_abort):
        print(_lc["yellow"](f"[!] {why} - waiting {_fmt_secs(secs)}"))
        if not _interruptible_sleep(max(1, secs), should_abort, why):
            raise RuntimeError("aborted while waiting for an OVH rate-limit slot") from None

    def _send(self, payload, should_abort):
        """Pace -> send -> on 429 wait for retry-after (pinned) or rotate (unpinned).
        Pinned 429s never give up (a cert would rather stall than switch); Ctrl-C
        aborts. Transient 5xx / transport errors retry with backoff."""
        attempt = 0
        tried   = set()
        while True:
            model = payload.get("model", "")
            tried.add(model)
            # Pace BEFORE sending: never hold an open request through a wait.
            secs = _slot_wait_secs(model)
            if secs > 0:
                nxt = _next_free_model(model, self.cfg, tried)
                if nxt:
                    print(_lc["dim"](f"  {model} minute full ({_PER_MIN}/min) - using {nxt}"))
                    payload = dict(payload, model=nxt)
                    continue
                self._wait(secs, f"{model} {_PER_MIN}/min slot", should_abort)
                continue
            self._last_in_total  = 0
            self.last_out_tokens = None
            req = urllib.request.Request(
                f"{_API_BASE}/chat/completions",
                data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "lean-coder"},
            )
            try:
                with urllib.request.urlopen(
                        req, timeout=(getattr(self.cfg, "gen_connect_timeout", None) or 600)) as resp:
                    result = self._consume(resp, should_abort)
                    _record(model, self._last_in_total, self.last_out_tokens or 0, 200)
                    self.last_served = model
                    return result
            except _UpstreamError as e:
                _record(model, 0, 0, e.code)
                if attempt < _MAX_RETRIES:
                    attempt += 1
                    self._wait(_backoff(attempt - 1),
                               f"{model} upstream {e.code} ({str(e)[:60]}) retry {attempt}/{_MAX_RETRIES}",
                               should_abort)
                    continue
                raise RuntimeError(f"{model} upstream error {e.code}: {str(e)[:300]}") from None
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                _record(model, 0, 0, e.code)
                if e.code == 429:
                    hint = _retry_after_secs(e.headers or {})
                    secs = (hint + 1) if hint is not None else _WINDOW
                    nxt  = _next_free_model(model, self.cfg, tried)
                    if nxt:
                        print(_lc["yellow"](f"[!] {model} rate limited - rotating to {nxt}"))
                        payload = dict(payload, model=nxt)
                        continue
                    tried = set()              # everyone busy: wait on this model
                    self._wait(secs, f"{model} rate limited ({_PER_MIN}/min per model)"
                               + (" - pinned, no model switch" if _pinned(self.cfg) else ""),
                               should_abort)
                    continue
                if e.code in (401, 403):
                    raise RuntimeError(f"API {e.code}: OVH refused the anonymous request "
                                       f"({body[:200]})") from None
                if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                    attempt += 1
                    hint = _retry_after_secs(e.headers or {})
                    self._wait((hint + 1) if hint else _backoff(attempt - 1),
                               f"{model} unavailable ({e.code}) retry {attempt}/{_MAX_RETRIES}",
                               should_abort)
                    continue
                raise RuntimeError(f"API {e.code}: {body[:400]}") from None
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                if attempt < _MAX_RETRIES:
                    attempt += 1
                    detail = str(getattr(e, "reason", None) or e) or type(e).__name__
                    self._wait(_backoff(attempt - 1),
                               f"{model} connection error ({detail[:60]}) retry {attempt}/{_MAX_RETRIES}",
                               should_abort)
                    continue
                raise RuntimeError(
                    f"request failed: {str(getattr(e, 'reason', None) or e) or type(e).__name__}"
                ) from None

    def chat(self, messages, tools, should_abort=None):
        cur_model = self._model()
        minfo     = _ensure_models().get(cur_model, {})
        if _MODEL_DATA and cur_model not in _MODEL_DATA:
            raise RuntimeError(f"{cur_model} is not in OVH's live model list (withdrawn?) - "
                               f"pick another with /model")
        # Never cap a reply below what the model can produce; shrink only to fit.
        max_out    = minfo.get("max_tokens") or _MAX_TOKENS
        max_in     = minfo.get("max_input_tokens", _CTX_DEFAULT)
        send_tools = tools if (tools and _tool_support(cur_model)) else None
        _mt        = _lc.get("messages_tokens")
        _est_toks  = _mt(messages, send_tools) if _mt else len(json.dumps(messages)) // 4
        if _mt is None and send_tools:
            _est_toks += len(json.dumps(send_tools)) // 4
        # vLLM rejects prompt + max_tokens > window with a 400 that arrives over SSE as a
        # bare [DONE] (empty stream). The local estimate runs low - up to ~1.8x on dense
        # digit/punctuation text (measured: est ~4.5k, real 8,095) - so: cap the reply at
        # _OUT_CAP (no turn needs the whole window; a /models max of 256k made us ask for
        # 254k) AND keep a 2x + 4k margin under the window.
        max_out = min(max_out, _OUT_CAP, max_in - _est_toks * 2 - 4096)
        if max_out < _MIN_ROOM:
            raise RuntimeError(
                f"context too large for {cur_model} "
                f"(~{_est_toks:,} of {max_in:,} tokens used, no room to reply) - run /compact first")
        payload = {
            "model":          cur_model,
            "messages":       _cvt_messages(messages),
            "max_tokens":     max_out,
            "stream":         True,
            "stream_options": {"include_usage": True},
        }
        if send_tools:
            payload["tools"] = _cvt_tools(send_tools)
        def _empty(r):
            a, _pe, ab = r
            return (not ab and _pe is None and not (a.get("content") or "").strip()
                    and not a.get("tool_calls") and not self.last_finish)
        result = self._send(payload, should_abort)
        if _empty(result) and payload["max_tokens"] > _MIN_ROOM * 4:
            # A bare [DONE] is almost always a rejected request (context overflow), not a
            # silent model: retry once with a much smaller reply budget.
            payload["max_tokens"] = max(_MIN_ROOM, payload["max_tokens"] // 8)
            result = self._send(payload, should_abort)
        if _empty(result):
            raise RuntimeError(f"{self.last_served or cur_model} returned an empty stream "
                               f"(no tokens, no finish reason) - likely the request was "
                               f"rejected (context too large?); try /compact, then retry")
        a, _pe, ab = result
        served = self.last_served or cur_model
        if served != cur_model:
            print(_lc["dim"](f"  (answered by {served}; ovh_pin on keeps one model)"))
        self._sess_cache_read += self.last_cache_read
        return result


def _make_client(cfg):
    return _OvhClient(cfg)


# ----------------------------------------------------------------------------
# Provider hooks
# ----------------------------------------------------------------------------

def _available():
    return True          # keyless: always offerable


def _autostart():
    return _state().get("active", False)


def _on_activate(agent, cfg):
    global _cfg
    _cfg = cfg
    _ensure_models(force=True)
    m = cfg.active_model()
    if not m:
        m = _DEFAULT_MODEL
        cfg.set_setting("model", m)
    if _MODEL_DATA and m not in _MODEL_DATA:
        if _pinned(cfg):
            print(_lc["red"](f"[!] {m} is no longer served by OVH and ovh_pin is on - NOT switching;"
                             " requests will fail. /provider set ovh_pin off to allow a switch"))
        else:
            nxt = next((c for c in [_DEFAULT_MODEL] + _FALLBACK_CHAIN if c in _MODEL_DATA), None)
            if nxt:
                print(_lc["yellow"](f"[!] {m} is no longer served by OVH - switched to {nxt}"))
                cfg.set_setting("model", nxt)
                m = nxt
    if _pinned(cfg):
        print(_lc["yellow"](f"  ovh_pin ON: {m} only - no rotation; a busy minute waits for the slot"
                            "  (/provider set ovh_pin off)"))
    else:
        print(_lc["dim"](f"  OVH free lane: {_PER_MIN} req/min per model per IP, no key;"
                         " busy model rotates to the next  (/provider set ovh_pin on = one model)"))
    d = _state()
    d["active"] = True
    _save_state(d)


def _on_deactivate(agent, cfg):
    d = _state()
    d["active"] = False
    _save_state(d)


def _login(agent, cfg):
    print(_lc["dim"]("  OVH's free lane needs no account or key (and this provider never sends one)."))
    d = _state()
    d["active"] = True
    _save_state(d)
    return True


def _clear(agent, cfg):
    try:
        if _state_path().is_file():
            _state_path().unlink()
    except OSError:
        pass


def _usage(agent, cfg):
    m = cfg.active_model() or _DEFAULT_MODEL
    n, oldest = _minute(m)
    rst = ""
    if oldest is not None:
        import datetime
        rst = datetime.datetime.fromtimestamp(oldest + _WINDOW, datetime.timezone.utc).isoformat()
    return {"meters": [{"label": "min", "pct": 100.0 * n / _PER_MIN,
                        "resets_at": rst, "reset_on_warn": True}]}


def _detail(agent, cfg, arg=""):
    bold, dim, red, green = _lc["bold"], _lc["dim"], _lc["red"], _lc["green"]
    cur  = cfg.active_model() or _DEFAULT_MODEL
    mode = "pinned" if _pinned(cfg) else "rotation on"
    lines = [bold(cur) + dim(f"   mode: {mode}   no key (cannot bill)")]
    si = getattr(agent, "session_in",  0) or 0
    so = getattr(agent, "session_out", 0) or 0
    lines.append(dim(f"\n  session   in {si:,}   out {so:,}   total {si + so:,}"))
    lines.append("")
    lines.append(dim(f"  per-model minute ({_PER_MIN} req/min each, per IP):"))
    led  = _sync_ledger()
    cut  = time.time() - 3600
    show = [cur] + [m for m in _chain(cfg) if m != cur]
    for m in show:
        if _MODEL_DATA and m not in _MODEL_DATA:
            continue
        n, _ = _minute(m)
        w    = _slot_wait_secs(m)
        hr   = [e for e in (led.events if led else []) if e[0] >= cut and e[1] == m]
        bad  = sum(1 for e in hr if e[4] == 429)
        st   = red(f"full, slot in {_fmt_secs(w)}") if w > 0 else green("free")
        mark = "*" if m == cur else " "
        lines.append(f"  {mark} {m[:36].ljust(36)} {n}/{_PER_MIN}  {st}"
                     + dim(f"   last hour: {len(hr) - bad} ok, {bad} x 429"))
    lines.append("\n" + _privacy_note())
    lines.append(dim("  (OVHcloud AI Endpoints - anonymous lane; models may change without notice)"))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

PROVIDER = {
    "name":           _PROVIDER,
    "description":    "OVHcloud AI Endpoints (free, no account or key; 2 req/min per model)",  # sweep-ok
    "make_client":    _make_client,
    "list_models":    _list_models,
    "context_window": _context_window,
    "capabilities":   _capabilities,
    "model_status":   _model_status,
    "tool_support":   _tool_support,
    "available":      _available,
    "autostart":      _autostart,
    "on_activate":    _on_activate,
    "on_deactivate":  _on_deactivate,
    "usage":          _usage,
    "login":          _login,
    "clear":          _clear,
    "detail":         _detail,
    "tag":            "ovh",
}


def setup(lc, cfg):
    global _CONFIG_DIR, _cfg
    _lc.update(lc)
    _CONFIG_DIR = lc.get("CONFIG_DIR")
    _cfg = cfg
