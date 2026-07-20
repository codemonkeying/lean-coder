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

Free + local, like ollama - not gated behind EVAL_ALLOW_PAID.
"""
import json
import os
import sys
import urllib.error
import urllib.request

_PROVIDER    = "llamacpp"
_API_BASE    = os.environ.get("LLAMACPP_HOST", "http://127.0.0.1:8080")
_CTX_DEFAULT = 32_768        # llama-server's -c; we can't read it back reliably, so default
_OUT_CAP     = 4096

_lc = {}


def _base():
    return os.environ.get("LLAMACPP_HOST", _API_BASE).rstrip("/")


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


def setup(lc, cfg):
    """Helpers-only hook; stash core namespace + cfg. Manager registers PROVIDER."""
    global _lc
    _lc = dict(lc)
    _lc["_cfg"] = cfg
