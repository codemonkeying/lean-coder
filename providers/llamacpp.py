"""lean-coder provider plugin: llama.cpp (via llama-server).

llama.cpp's `llama-server` is the most widely-used self-hosted OpenAI-compatible
inference server (the engine Ollama itself wraps). Many people run raw llama-server
directly - for GGUF models, custom quants, or hardware Ollama doesn't package well.
This provider drives any llama-server exactly like the bundled ollama backend,
without going through Ollama.

BACKEND: `llama-server -m model.gguf --port 8080 --jinja`
  - OpenAI shape: POST {base}/v1/chat/completions  (messages, tools, stream)
  - Model list:  GET  {base}/v1/models  (returns the one loaded model)
  - Health:      GET  {base}/health
  - Local, NO AUTH by default (optional --api-key; honoured via LLAMACPP_API_KEY).
  - Tool calling: supported since PR #9639, but REQUIRES the server started with
    `--jinja` (uses the model's chat template to format tool calls). A server
    started WITHOUT --jinja silently ignores the tools array; small GGUF models
    may also text-encode calls, so we run core's parse_text_tool_calls fallback
    exactly like OllamaClient.chat does.

SERVER-SIDE TOOL-USE NOTES (stock llama-server):
  - Multi-turn tool use needs each replayed assistant tool_call to carry the strict
    OpenAI shape {"id","type":"function","function":{...}} or --jinja 500s with
    "Missing tool call type" on turn 2+. We emit that shape (see _consume).
  - Verify the loaded template actually supports tools: GET {base}/props ->
    chat_template_tool_use should be present.
  - Qwen models: if calls come back as literal `<function=...>` text instead of
    structured tool_calls, start the server with
    --chat-template-kwargs '{"tool_call_format":"json"}' (else our text fallback
    catches them, but structured is cleaner).
  - arguments type (llama.cpp #20198): recent llama-server returns
    tool_calls[].function.arguments as a JSON object, older/OpenAI/ollama as a
    string - _consume accepts either.
  - thinking control: stock llama-server IGNORES a top-level enable_thinking/think
    key; Qwen-class templates only honour it via chat_template_kwargs. chat() maps
    cfg.think (bool|None|level str) -> chat_template_kwargs.enable_thinking (+ reasoning_effort for a level) so --think off/on
    actually takes effect (None = leave the template default). Without it, a
    thinking model dumps everything into reasoning_content and a pure-text turn
    (e.g. auto_compact) returns EMPTY content. _consume streams the reasoning_content
    channel dimmed (like ollama) and keeps it out of the content body.

Free + local, like ollama - not gated behind EVAL_ALLOW_PAID.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


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


_PROVIDER    = "llamacpp"
_DEFAULT_HOST = "http://127.0.0.1:8080"

_CTX_DEFAULT = 32_768        # fallback when /props can't give a real n_ctx
# cfg.think level -> chat_template_kwargs.reasoning_effort (qwen3.8: low/medium/xhigh).
_EFFORT    = {"low": "low", "medium": "medium", "med": "medium",
              "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
_THINK_OFF = {"off", "false", "no", "0", "none", ""}


# Reply-length guard for LOCAL models. A paid API is never capped (a cut reply
# wastes the turn), but locally a runaway/looping model can grind the GPU for
# many minutes with nothing else able to run. Default 32k is far above any real
# reply; /set max_out <n> to change, /set max_out 0 for unlimited.
_MAX_OUT_DEFAULT = 32_768


def _max_out(cfg):
    """Configured local reply cap: None = unlimited, else a positive int."""
    try:
        v = cfg.setting("max_out") if cfg is not None else None
    except Exception:
        v = None
    if v in (None, ""):
        return _MAX_OUT_DEFAULT
    try:
        v = int(v)
    except (TypeError, ValueError):
        return _MAX_OUT_DEFAULT
    return v if v > 0 else None


def _note_cut(cfg, finish):
    """Tell the operator when OUR cap cut the reply (never silent truncation)."""
    cap = _max_out(cfg)
    if finish == "length" and cap:
        print(_lc["yellow"](f"[!] reply hit max_out ({cap:,} tokens) and was cut off - "
                            f"/set max_out <higher> or /set max_out 0 for unlimited"))

_lc = {}
_CFG = None      # the live cfg, kept from setup()
# Active base URL, resolved at setup() from (in precedence order) the provider-scoped
# 'host' setting (config.toml [providers.llamacpp].host, or a session override), then
# the LLAMACPP_HOST env, then the default. /llamacpp host updates this + persists.
_HOST = None


def _entry_status(m):
    """A /v1/models entry's router status as a lowercase string, or None. Accepts
    both the object form {"status": {"value": "loaded", ...}} and a bare string."""
    st = m.get("status")
    if isinstance(st, dict):
        st = st.get("value")
    return st.lower() if isinstance(st, str) and st else None


# Last model this process sent to the server: a different one means a possible
# router swap, so we probe its status once and warn about the load wait.
_LAST_SENT = None


# Context window per (host, model), read from /props. Real values cache for good;
# a miss (unloaded router model, old server, network) caches the fallback for
# _CTX_RETRY s so the per-render status bar never stalls on a down host. A reply
# from the model (it's loaded now) drops a fallback entry so the next read is real.
_CTX_CACHE = {}
_CTX_RETRY = 300.0


def _props_ctx(base, model, timeout=3):
    """Per-conversation window from GET /props?model=<id> (plain /props fallback):
    default_generation_settings.n_ctx. That is ALREADY per slot (llama-server reports
    n_ctx_slot() = llama_n_ctx_seq, capped by --kv-unified-per-slot), so it is NOT
    divided by total_slots. None when unreadable. Router: an UNLOADED model is
    never probed, so reading the window can't trigger a 30-65s swap."""
    q = f"?model={urllib.parse.quote(model)}" if model else ""
    for url in ((f"{base}/props{q}", f"{base}/props") if q else (f"{base}/props",)):
        try:
            req = urllib.request.Request(url, headers=_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        n = (d.get("default_generation_settings") or {}).get("n_ctx") or d.get("n_ctx")
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return None


def _ctx_window(model):
    """Cached real window for `model` on the active host, else _CTX_DEFAULT."""
    import time as _t
    base = _base()
    key = (base, model or "")
    hit = _CTX_CACHE.get(key)
    if hit and (hit[1] is None or _t.monotonic() < hit[1]):
        return hit[0]
    n = None
    cfg = _CFG
    st = LlamaCppClient(cfg).model_status(timeout=3).get(model) if (cfg and model) else None
    if st in (None, "loaded"):          # single-model server (no status) or loaded
        n = _props_ctx(base, model)
    if n:
        _CTX_CACHE[key] = (n, None)
        return n
    _CTX_CACHE[key] = (_CTX_DEFAULT, _t.monotonic() + _CTX_RETRY)
    return _CTX_DEFAULT


# Sampling card per (host, model): what llama-server applies when a request OMITS a
# field (router preset / CLI flags / GGUF general.sampling.*), read from
# /props default_generation_settings. Values equal to llama.cpp's compiled built-ins
# mean "nobody set it" and are not reported as a card. Cached like _CTX_CACHE.
_SAMP_CACHE = {}
_LLAMA_BUILTIN = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                  "presence_penalty": 0.0, "repeat_penalty": 1.0}
_PROPS_KEYS = {"temperature": "temperature", "temp": "temperature", "top_k": "top_k",
               "top_p": "top_p", "min_p": "min_p", "presence_penalty": "presence_penalty",
               "repeat_penalty": "repeat_penalty"}


def _props_card(base, model, timeout=3):
    """dict of non-built-in sampling defaults the server applies, {} if it only has
    built-ins, None if /props is unreadable (unloaded router model / old server)."""
    q = f"?model={urllib.parse.quote(model)}" if model else ""
    for url in ((f"{base}/props{q}", f"{base}/props") if q else (f"{base}/props",)):
        try:
            req = urllib.request.Request(url, headers=_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            continue
        g = (d.get("default_generation_settings") or {}) if isinstance(d, dict) else {}
        g = g.get("params") if isinstance(g.get("params"), dict) else g
        card = {}
        for pk, k in _PROPS_KEYS.items():
            v = g.get(pk)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if abs(float(v) - _LLAMA_BUILTIN[k]) > 1e-6:
                    card[k] = v
        return card
    return None


def _server_card(model):
    import time as _t
    key = (_base(), model or "")
    hit = _SAMP_CACHE.get(key)
    if hit and (hit[1] is None or _t.monotonic() < hit[1]):
        return hit[0]
    cfg = _CFG
    st = LlamaCppClient(cfg).model_status(timeout=3).get(model) if (cfg and model) else None
    card = _props_card(_base(), model) if st in (None, "loaded") else None
    # A router preset that isn't loaded yet can't be read without a swap: retry later.
    _SAMP_CACHE[key] = (card, None if card is not None else _t.monotonic() + _CTX_RETRY)
    return card


def _base():
    """Active llama-server base URL. Precedence: the /llamacpp-set host (module state,
    seeded from cfg at setup) -> LLAMACPP_HOST env -> the compiled default. rstrip so
    '{base}/v1/...' never doubles a slash."""
    return (_HOST or os.environ.get("LLAMACPP_HOST") or _DEFAULT_HOST).rstrip("/")


def _api_key():
    # llama-server is usually keyless; honour an optional --api-key deployment.
    return os.environ.get("LLAMACPP_API_KEY", "")


def _headers():
    h = {"Content-Type": "application/json"}
    k = _api_key()
    if k:
        h["Authorization"] = f"Bearer {k}"
    return h


class LlamaCppClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_out_tokens = None
        # Decode telemetry (core reads last_out_tokens + last_eval_ns for tok/s), and
        # prompt tokens served from the slot cache (timings.cache_n), per call.
        self.last_eval_tokens = self.last_eval_ns = None
        self.last_cache_n = None
        self._models = None

    def _model_entries(self, timeout=10):
        """Raw /v1/models entries; [] on failure. Single-model llama-server lists one
        id; router mode (--models-preset) lists every preset with a status."""
        try:
            req = urllib.request.Request(f"{_base()}/v1/models", headers=_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            return [m for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        except (urllib.error.URLError, OSError, ValueError, KeyError, AttributeError):
            return []

    def list_models(self):
        """GET /v1/models (cached) -> model ids. [] on failure so menu/availability
        checks degrade gracefully."""
        if self._models is None:
            self._models = [m["id"] for m in self._model_entries()]
        return self._models

    def model_status(self, timeout=10):
        """{id: 'loaded'|'loading'|'unloaded'|...} for router-mode entries that carry a
        status; {} for a single-model server (no status field) or on failure."""
        out = {}
        for m in self._model_entries(timeout):
            st = _entry_status(m)
            if st:
                out[m["id"]] = st
        return out

    def running_models(self):
        """Router mode: the model(s) currently loaded (green dot in /model). Live, not
        cached - loads change under us. [] on a single-model server."""
        return [mid for mid, st in self.model_status().items() if st == "loaded"]

    def detect_num_ctx(self):
        return _ctx_window(self.cfg.active_model() if self.cfg else None)

    def resolve_sampling(self, full=False):
        """Effective sampling for the active model ({'params','source','sent'});
        full=True returns the (send, effective, source) triple used to build a request.
        Also stored as self.last_sampling (read via lean_coder.effective_sampling)."""
        model = self.cfg.active_model() if self.cfg else ""
        sf = _lc.get("sampling_for")
        if sf is None:
            return ({}, {}, "server built-in") if full else None
        try:
            card = _server_card(model)
        except Exception:
            card = None
        send, eff, src = sf(self.cfg, model, card, "llama-server /props")
        self.last_sampling = {"params": eff, "source": src, "sent": dict(send)}
        return (send, eff, src) if full else self.last_sampling

    def _consume(self, resp, should_abort, tools=None):
        """Read the streamed reply with core's shared OpenAI-style reader, plus
        llama-server's per-chunk `timings` (decode speed, cache hits, and a token count
        when the server omits `usage`). Returns (assistant_dict, prompt_tokens, aborted)."""
        self.last_eval_tokens = self.last_eval_ns = self.last_cache_n = None
        timings = {}

        def on_obj(obj, st):
            tm = obj.get("timings")
            if isinstance(tm, dict):
                timings.clear()
                timings.update(tm)               # last one wins

        def on_done(st):
            if not timings:
                return
            self.last_eval_tokens = timings.get("predicted_n") or None
            self.last_eval_ns = int((timings.get("predicted_ms") or 0) * 1e6) or None
            self.last_cache_n = timings.get("cache_n")
            if not st["out_tokens"]:
                st["out_tokens"] = self.last_eval_tokens
            if st["prompt_total"] is None and timings.get("prompt_n") is not None:
                st["prompt_total"] = (timings.get("prompt_n") or 0) + (timings.get("cache_n") or 0)

        out = _lc["consume_openai_stream"](self, resp, should_abort, tools,
                                           on_obj=on_obj, on_done=on_done)
        if not out[2]:
            _note_cut(self.cfg, self.last_finish)
        return out

    def _swap_hint(self, model):
        """Router mode: when the model changes and the server reports it isn't loaded
        yet, say so - a swap is a silent 30-65s wait otherwise. One quick status probe
        per model change; a single-model server (no status) prints nothing."""
        global _LAST_SENT
        if not model or model == _LAST_SENT:
            return
        _LAST_SENT = model
        st = self.model_status(timeout=3).get(model)
        if st and st != "loaded":
            print(_lc["dim"](f"  loading {model} on llama-server (router swap, can take ~1 min)..."))

    def chat(self, messages, tools, should_abort=None):
        """POST /v1/chat/completions (streamed). Returns
        (assistant_dict, prompt_tokens, aborted)."""
        if should_abort and should_abort():
            return {"role": "assistant", "content": ""}, None, True

        body = {
            "model":       self.cfg.active_model(),   # router mode (--models-preset): selects/loads the model
            "messages":    messages,
            "stream":      True,
            # Usage chunk (prompt/completion tokens) is opt-in, as on OpenAI; timings
            # carry decode time + slot-cache hits. Unknown keys are ignored server-side.
            "stream_options":    {"include_usage": True},
            "timings_per_token": True,
        }
        _cap = _max_out(self.cfg)
        if _cap:
            body["max_tokens"] = _cap
        if tools:
            body["tools"] = tools
        # Sampling: only what the user set, or the family table when the server has
        # no card of its own (unset fields fall through to preset / GGUF metadata).
        send, _eff, _src = self.resolve_sampling(full=True)
        body.update(send)
        self._swap_hint(body["model"])
        # Thinking control. cfg.think is the same bool|None the ollama provider sends
        # (True/False = force on/off; None = don't touch, let the template default
        # decide). BUT stock llama-server IGNORES a top-level enable_thinking / think
        # key - Qwen-class templates only honour it via chat_template_kwargs. Without
        # this, a thinking model dumps everything into reasoning_content and a
        # pure-TEXT turn (e.g. auto_compact's no-tools summary) comes back with EMPTY
        # content. So map the toggle into chat_template_kwargs.enable_thinking; leave
        # it off entirely when think is None so we don't override the model default.
        # A level string (headless callers set cfg.think="low"|"medium"|"high", as
        # ollama takes it) ALSO goes in as reasoning_effort, mapped to the qwen3.8
        # vocabulary: its template defaults to xhigh and RAISES on unknown values, so
        # an unmapped level drops the key rather than forwarding it. Templates without
        # reasoning_effort ignore the extra kwarg. "off"-style strings mean False.
        t = self.cfg.think
        if t is not None:
            if isinstance(t, str) and t.strip().lower() in _THINK_OFF:
                t = False
            kw = {"enable_thinking": bool(t)}
            if isinstance(t, str):
                eff = _EFFORT.get(t.strip().lower())
                if eff:
                    kw["reasoning_effort"] = eff
            body["chat_template_kwargs"] = kw

        req = urllib.request.Request(
            f"{_base()}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=_headers(),
            method="POST")
        try:
            with urllib.request.urlopen(
                    req,
                    timeout=(getattr(self.cfg, "gen_connect_timeout", None) or 600)) as resp:
                out = self._consume(resp, should_abort, tools)
                # It answered, so it's loaded: drop a fallback window cached while it
                # was unloaded so the next read gets the real n_ctx.
                _ck = (_base(), body["model"] or "")
                if _CTX_CACHE.get(_ck, (0, None))[1] is not None:
                    _CTX_CACHE.pop(_ck, None)
                return out
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (401, 403):
                raise RuntimeError(f"llama-server {e.code}: key rejected - set LLAMACPP_API_KEY\n{detail}") from None
            raise RuntimeError(f"API {e.code}: {detail}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(
                f"request to {_base()} failed: {e}\n"
                f"start it with: llama-server -m model.gguf --port 8080 --jinja") from None


def _make_client(cfg):
    return LlamaCppClient(cfg)


PROVIDER = {
    "name":           _PROVIDER,
    "description":    "llama.cpp (llama-server) - local GGUF, OpenAI-compatible",
    "make_client":    _make_client,
    "list_models":    lambda: LlamaCppClient(_CFG).list_models() if _CFG else [],
    "context_window": lambda model: _ctx_window(model),
    "available":      lambda: True,    # local + free
    "autostart":      lambda: False,   # explicit selection only
    "tag":            "gguf",
    "capabilities":   lambda model: {"thinking": ["off", "on"]},
    # Router mode: loaded model(s) get the green dot in /model; [] otherwise.
    "warm_models":    lambda: LlamaCppClient(_CFG).running_models() if _CFG else [],
}


# ----------------------------------------------------------------------------
# /llamacpp host command
# ----------------------------------------------------------------------------

def _host_alive(base, timeout=2.0):
    """Quick reachability probe: HTTP 200 from {base}/v1/models (the endpoint the
    client itself uses to enumerate the loaded model). False on any error/timeout."""
    try:
        req = urllib.request.Request(f"{base.rstrip('/')}/v1/models",
                                     headers=_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _norm_url(s):
    """Normalize a user-typed host into a base URL: add http:// if no scheme, strip a
    trailing slash. A bare 'box:8080' or 'http://box:8080' both become a usable base.
    Returns '' for empty/whitespace."""
    s = (s or "").strip()
    if not s:
        return ""
    if any(c.isspace() for c in s):
        return ""
    if "://" not in s:
        s = "http://" + s
    try:
        p = urllib.parse.urlparse(s)
        p.port  # raises ValueError on a bad port
    except ValueError:
        return ""
    if p.scheme not in ("http", "https") or not p.hostname:
        return ""
    return s.rstrip("/")


def _set_host(agent, cfg, url, persist=True):
    """Commit `url` as the active llama-server base: update module state so _base()
    (and thus the live client) uses it immediately, record it as the provider-scoped
    'host' setting, and persist to config.toml. Clears the client's cached model list
    so the next read re-resolves against the new host."""
    global _HOST, _LAST_SENT
    _HOST = url
    _LAST_SENT = None   # new server: the swap hint must fire again
    try:
        cfg.provider_settings.setdefault("llamacpp", {})["host"] = url
    except Exception:
        _dbg_swallow()
    # Drop any cached model list on the live llama.cpp client so /model re-reads.
    client = getattr(agent, "client", None)
    if isinstance(client, LlamaCppClient) and client._models is not None:
        client._models = None
    if persist:
        save = _lc.get("save_config") or _lc.get("autosave_config")
        if not save:
            print(_lc["yellow"]("no config saver available; host set for this session only."))
        else:
            try:
                save(cfg)
            except Exception as e:
                print(_lc["yellow"](f"could not save config ({e}); host set for this session only."))
    print(_lc["green"](f"llama-server host -> {url}"))


def _switch_host(agent, cfg, arg):
    """Set the active llama-server base URL. With no arg, show the current host and
    prompt for a new one. Probes the target BEFORE committing so a typo or a dead box
    isn't silently adopted: an unreachable host asks 'use anyway?' (it may just not be
    up yet), otherwise the host is left unchanged."""
    ask = _lc.get("ask")
    if not arg:
        print(_lc["dim"](f"current llama-server host: {_base()}"))
        if not ask:
            print(_lc["dim"]("usage: /llamacpp host <url>"))
            return
        arg = ask("new host (blank = keep current)?", default="") if _accepts_default(ask) \
            else ask("new host (blank = keep current)?")
        if not (arg or "").strip():
            print(_lc["dim"]("host unchanged."))
            return
    url = _norm_url(arg)
    if not url:
        print(_lc["red"](f"'{arg}' is not a usable host (need a name, host:port, or URL)."))
        return
    if url.rstrip("/") == _base():
        print(_lc["dim"](f"already on {url}."))
        return
    if not _host_alive(url):
        print(_lc["yellow"](f"{url} isn't responding (no /v1/models)."))
        if not (ask and ask("use it anyway (e.g. it's not up yet)?")):
            print(_lc["dim"]("host unchanged."))
            return
    _set_host(agent, cfg, url)


def _accepts_default(fn):
    """True if the injected _ask supports a default= kwarg (varies by core version)."""
    try:
        import inspect
        return "default" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _handle_llamacpp(agent, cfg, arg):
    """/llamacpp [host [url]] - manage the llama-server connection.

    bare /llamacpp        -> show current host, prompt for a new one
    /llamacpp host        -> same (explicit)
    /llamacpp host <url>  -> set the active base URL (probed, persisted)
    /llamacpp <url>       -> shorthand: treat a bare URL/name as a host set

    Host precedence: the host set here (saved as provider_settings.llamacpp.host)
    wins over the LLAMACPP_HOST env var, which wins over http://127.0.0.1:8080.
    """
    arg = (arg or "").strip()
    parts = arg.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    if sub == "host":
        _switch_host(agent, cfg, rest)
    elif not arg:
        _switch_host(agent, cfg, "")
    else:
        # no recognised subcommand -> treat the whole arg as a host (URL/name), so
        # `/llamacpp gpu-box:8080` and `/llamacpp http://...` Just Work.
        _switch_host(agent, cfg, arg)


def _llamacpp_completer(agent, cfg):
    """Tab targets for /llamacpp: the 'host' verb (the only subcommand for now)."""
    return ["host"]


def setup(lc, cfg):
    """Helpers-only hook: stash the core namespace + cfg, seed the active host from the
    provider-scoped 'host' setting (falls back to LLAMACPP_HOST env / default via
    _base), and register the /llamacpp command. The manager registers PROVIDER."""
    global _lc, _HOST, _CFG
    _lc = dict(lc)
    _CFG = cfg
    try:
        # Read the llamacpp bucket BY NAME, not cfg.setting() (which reads the
        # active provider's bucket): setup() runs for every enabled provider, so if
        # llamacpp is enabled but not active at launch, cfg.setting would read the
        # wrong provider's 'host'.
        stored = (cfg.provider_settings.get("llamacpp", {}).get("host") or "").strip()
    except Exception:
        stored = ""
    if stored:
        _HOST = _norm_url(stored)
    reg = lc.get("register_command")
    if reg:
        reg("/llamacpp", _handle_llamacpp,
            "llama-server: host [url] (no arg = show/set the base URL)",
            completer=_llamacpp_completer)
