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
    cfg.think (bool|None) -> chat_template_kwargs.enable_thinking so --think off/on
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
import urllib.request

_PROVIDER    = "llamacpp"
_DEFAULT_HOST = "http://127.0.0.1:8080"
_API_BASE    = os.environ.get("LLAMACPP_HOST", _DEFAULT_HOST)
_CTX_DEFAULT = 32_768        # llama-server's -c; we can't read it back reliably, so default
_OUT_CAP     = 4096

_lc = {}
# Active base URL, resolved at setup() from (in precedence order) the provider-scoped
# 'host' setting (config.toml [providers.llamacpp].host, or a session override), then
# the LLAMACPP_HOST env, then the default. /llamacpp host updates this + persists.
_HOST = None


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
        self._models = None

    def list_models(self):
        """GET /v1/models (cached). llama-server reports the single loaded model.
        [] on failure so menu/availability checks degrade gracefully."""
        if self._models is None:
            try:
                req = urllib.request.Request(f"{_base()}/v1/models", headers=_headers(), method="GET")
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read())
                self._models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            except (urllib.error.URLError, OSError, ValueError, KeyError):
                self._models = []
        return self._models

    def running_models(self):
        return []

    def detect_num_ctx(self):
        return _CTX_DEFAULT

    def _consume(self, resp, should_abort, tools=None):
        """Parse a streamed OpenAI SSE response, printing content as it arrives and
        accumulating any tool_calls. Returns (assistant_dict, prompt_tokens, aborted).
        Mirrors providers/openai.py::_consume."""
        content_parts = []
        tool_acc      = {}
        order         = []
        prompt_total  = None
        out_tokens    = None
        printed       = False
        printed_think = False
        aborted       = False
        md            = _lc["MarkdownStream"](sys.stdout.write)

        spin = _lc["Spinner"]("thinking", _lc["THINK_FRAMES"]).start()
        try:
            for raw in _lc["stream_tiered"](resp, self.cfg):
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
                    # Thinking channel: stock llama-server streams reasoning on
                    # delta.reasoning_content (some builds/models use delta.thinking).
                    # Dim it like ollama, and keep it OUT of content_parts so an
                    # all-reasoning span never inflates the (possibly empty) content.
                    think = delta.get("reasoning_content") or delta.get("thinking")
                    if think:
                        if not printed_think and not printed:
                            spin.stop()
                            sys.stdout.write(_lc["dim"](_lc["GLYPH"]["think"] + " "))
                            printed_think = True
                        sys.stdout.write(_lc["dim"](think))
                        sys.stdout.flush()
                    txt   = delta.get("content")
                    if txt:
                        if not printed:
                            spin.stop()
                            if printed_think:
                                sys.stdout.write("\n")
                            sys.stdout.write(_lc["blue"]("● "))
                            printed = True
                        md.feed(txt)
                        content_parts.append(txt)
                    for tcd in (delta.get("tool_calls") or []):
                        idx = tcd.get("index", 0)
                        if idx not in tool_acc:
                            tool_acc[idx] = {"id": "", "name": "", "args": [], "args_obj": None}
                            order.append(idx)
                        slot = tool_acc[idx]
                        if tcd.get("id"):
                            slot["id"] = tcd["id"]
                        fn = tcd.get("function", {})
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        _a = fn.get("arguments")
                        if _a is not None and _a != "":
                            # llama.cpp #20198: recent llama-server returns arguments
                            # as an already-parsed JSON OBJECT, whereas OpenAI/ollama
                            # stream it as string fragments. Keep string deltas for
                            # concatenation; stash a dict/list whole (last-wins).
                            if isinstance(_a, str):
                                slot["args"].append(_a)
                            else:
                                slot["args_obj"] = _a

                um = obj.get("usage")
                if um:
                    prompt_total = um.get("prompt_tokens", prompt_total)
                    out_tokens   = um.get("completion_tokens", out_tokens)
        finally:
            spin.stop()

        if printed:
            md.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif printed_think:
            # thinking-only turn (no content channel yet): close the dim reasoning
            # line so the next thing printed isn't glued to it.
            sys.stdout.write("\n")
            sys.stdout.flush()

        tool_calls = []
        for idx in order:
            slot = tool_acc[idx]
            if slot.get("args_obj") is not None:
                # Server already handed us a parsed object (llama.cpp #20198).
                args = slot["args_obj"]
            else:
                try:
                    args = json.loads("".join(slot["args"])) if slot["args"] else {}
                except Exception:
                    args = {}
            tool_calls.append({
                "id":       slot["id"] or f"call_{idx}",
                # OpenAI spec requires "type":"function" on assistant tool_calls.
                # Stock llama-server (--jinja) strictly re-validates this when the
                # assistant message is REPLAYED in history on turn 2+; omitting it
                # 500s ("Missing tool call type"). A fresh turn-1 call works either
                # way, which is why the bug only shows on multi-turn tool use.
                "type":     "function",
                "function": {"name": slot["name"], "arguments": args},
            })

        if out_tokens:
            self.last_out_tokens = out_tokens

        content = "".join(content_parts)
        # Text-tool-call fallback: only when the model emitted NO native tool_calls
        # (small GGUF models, or a server started without --jinja). Matches
        # OllamaClient.chat - guard against firing on a tool merely quoted in prose.
        if not tool_calls and not aborted:
            parse = _lc.get("parse_text_tool_calls")
            if parse:
                known = {(_t.get("function") or _t).get("name") for _t in (tools or [])}
                known.discard(None)
                _schemas = {}
                for _t in (tools or []):
                    _fn = _t.get("function") or _t
                    _nm = _fn.get("name")
                    _props = (_fn.get("parameters") or {}).get("properties")
                    if _nm and isinstance(_props, dict):
                        _schemas[_nm] = _props
                parsed, cleaned = parse(content, known_names=known or None,
                                        schemas=_schemas or None)
                if parsed:
                    print(_lc["dim"](f"(parsed {len(parsed)} tool call(s) from text)"))
                    tool_calls = parsed
                    content = cleaned

        assistant = {"role": "assistant", "content": content}
        if tool_calls:
            # Normalize EVERY tool_call (native + text-fallback) to the strict OpenAI
            # shape - id + type:function - so it survives being replayed in history to
            # a --jinja llama-server on later turns (see the type note above).
            for _i, _tc in enumerate(tool_calls):
                _tc.setdefault("id", f"call_{_i}")
                _tc.setdefault("type", "function")
            assistant["tool_calls"] = tool_calls
        return assistant, prompt_total, aborted

    def chat(self, messages, tools, should_abort=None):
        """POST /v1/chat/completions (streamed). Returns
        (assistant_dict, prompt_tokens, aborted)."""
        if should_abort and should_abort():
            return {"role": "assistant", "content": ""}, None, True

        body = {
            "model":       self.cfg.active_model(),   # llama-server ignores it (single model) but harmless
            "messages":    messages,
            "max_tokens":  _OUT_CAP,
            "temperature": getattr(self.cfg, "temperature", 0.0),
            "stream":      True,
        }
        if tools:
            body["tools"] = tools
        # Thinking control. cfg.think is the same bool|None the ollama provider sends
        # (True/False = force on/off; None = don't touch, let the template default
        # decide). BUT stock llama-server IGNORES a top-level enable_thinking / think
        # key - Qwen-class templates only honour it via chat_template_kwargs. Without
        # this, a thinking model dumps everything into reasoning_content and a
        # pure-TEXT turn (e.g. auto_compact's no-tools summary) comes back with EMPTY
        # content. So map the toggle into chat_template_kwargs.enable_thinking; leave
        # it off entirely when think is None so we don't override the model default.
        if self.cfg.think is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(self.cfg.think)}

        req = urllib.request.Request(
            f"{_base()}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=_headers(),
            method="POST")
        try:
            with urllib.request.urlopen(
                    req,
                    timeout=(getattr(self.cfg, "gen_connect_timeout", None) or 600)) as resp:
                return self._consume(resp, should_abort, tools)
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
    "list_models":    lambda: LlamaCppClient(_lc.get("_cfg")).list_models() if _lc.get("_cfg") else [],
    "context_window": lambda model: _CTX_DEFAULT,
    "available":      lambda: True,    # local + free
    "autostart":      lambda: False,   # explicit selection only
    "tag":            "gguf",
    "capabilities":   lambda model: {},
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
    if "://" not in s:
        s = "http://" + s
    return s.rstrip("/")


def _set_host(agent, cfg, url, persist=True):
    """Commit `url` as the active llama-server base: update module state so _base()
    (and thus the live client) uses it immediately, record it as the provider-scoped
    'host' setting, and persist to config.toml. Clears the client's cached model list
    so the next read re-resolves against the new host."""
    global _HOST
    _HOST = url
    try:
        cfg.set_setting("host", url)
    except Exception:
        pass
    # Drop any cached model list on the live client so /model re-reads the new host.
    client = getattr(agent, "client", None)
    if client is not None and getattr(client, "_models", None) is not None:
        client._models = None
    if persist:
        save = _lc.get("save_config") or _lc.get("autosave_config")
        if save:
            try:
                save(cfg)
            except Exception:
                pass
    print(_lc["green"](f"llama-server host -> {url}"))


def _switch_host(agent, cfg, arg):
    """Set the active llama-server base URL. With no arg, show the current host and
    prompt for a new one. Probes the target BEFORE committing so a typo or a dead box
    isn't silently adopted: an unreachable host asks 'use anyway?' (it may just not be
    up yet), otherwise the host is left unchanged."""
    ask = _lc.get("_ask")
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
    global _lc, _HOST
    _lc = dict(lc)
    _lc["_cfg"] = cfg
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
