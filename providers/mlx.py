"""lean-coder provider plugin: MLX (Apple Silicon native, via mlx_lm.server).

Apple Silicon (M-series) runs models fastest through MLX (unified memory, no
copy). mlx_lm.server exposes an OpenAI-compatible HTTP API, so lean-coder can
drive an MLX model on a Mac exactly like any OpenAI-shaped backend. Target box
in our fleet: the M4 Mac (216) - serves qwen3-14b MLX-4bit natively instead of
the slower GGUF/Ollama path.

BACKEND: `mlx_lm.server --model <hf-repo-or-local> --port 8080`
  - OpenAI shape: POST {base}/v1/chat/completions  (messages, tools, stream)
  - Model list:  GET  {base}/v1/models
  - Local, NO AUTH (basic security only; not for production/public exposure).
  - Default host/port: 127.0.0.1:8080. Base is configurable via MLX_HOST.
  - Tool calling: mlx_lm returns tool_calls in OpenAI shape; smaller models may
    text-encode calls, so we run core's parse_text_tool_calls fallback exactly
    like OllamaClient.chat does.

Free + local, like ollama - not gated behind EVAL_ALLOW_PAID.
"""
import json
import os
import sys
import urllib.error
import urllib.request

_PROVIDER    = "mlx"
# Base URL of the running mlx_lm.server. Override per-box via env.
_API_BASE    = os.environ.get("MLX_HOST", "http://127.0.0.1:8080")
_CTX_DEFAULT = 32_768        # match the harness AUTO_CTX_CAP; MLX models often 256K-capable
_OUT_CAP     = 4096

_lc = {}


def _base():
    # Re-read env each call so a per-run MLX_HOST (e.g. the Mac's LAN IP) is honoured.
    return os.environ.get("MLX_HOST", _API_BASE).rstrip("/")


class MLXClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_out_tokens = None
        self._models = None

    def list_models(self):
        """Available model ids from GET /v1/models (cached). [] on failure.
        Duck-types OllamaClient.list_models so ensure_model()-style checks work."""
        if self._models is None:
            try:
                req = urllib.request.Request(f"{_base()}/v1/models", method="GET")
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
                    # Reasoning trace (think ON): show it dimmed, do NOT store it -
                    # mirrors OllamaClient's handling of message.thinking. mlx_lm
                    # streams it as delta.reasoning_content (some builds:
                    # reasoning). Never accumulated into content_parts.
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if rc:
                        if not printed_think:
                            spin.stop()
                            sys.stdout.write(_lc["dim"](_lc.get("GLYPH", {}).get("think", "◇") + " "))
                            printed_think = True
                        sys.stdout.write(_lc["dim"](rc))
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
        elif printed_think:
            # reasoning shown but no content followed (e.g. a pure tool-call turn)
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
        # (smaller MLX models sometimes text-encode). Matches OllamaClient.chat.
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
        """POST /v1/chat/completions (streamed). messages/tools are already
        OpenAI-shaped (lean-coder's internal format IS OpenAI-compatible).
        Returns (assistant_dict, prompt_tokens_or_None, aborted)."""
        if should_abort and should_abort():
            return {"role": "assistant", "content": ""}, None, True

        # Sampling options: mirror OllamaClient's cfg.options() so a run behaves
        # identically across backends. mlx_lm.server reads these from the request
        # body; OpenAI-shape names it does not understand are simply ignored, so it
        # is safe to send the full set.
        opts = self.cfg.options() if hasattr(self.cfg, "options") else {}
        body = {
            "model":          self.cfg.active_model(),
            "messages":       messages,
            "max_tokens":     _OUT_CAP,
            "temperature":    opts.get("temperature", getattr(self.cfg, "temperature", 0.0)),
            "stream":         True,
            "stream_options": {"include_usage": True},
        }
        if opts.get("top_p") is not None:
            body["top_p"] = opts["top_p"]
        if opts.get("top_k") is not None:
            body["top_k"] = opts["top_k"]
        if opts.get("repeat_penalty") is not None:
            body["repetition_penalty"] = opts["repeat_penalty"]
        # Think toggle. The clean cross-backend way to disable thinking is
        # chat_template_kwargs={"enable_thinking": False} in the request body (the
        # server merges it into apply_chat_template). Newer mlx_lm honours this
        # per-request; older builds only read the --chat-template-args launch flag,
        # so ALSO launch the server with '{"enable_thinking":false}' for gold certs.
        if getattr(self.cfg, "think", None) is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(self.cfg.think)}
        if tools:
            body["tools"] = tools

        req = urllib.request.Request(
            f"{_base()}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            # Generous timeout: MLX cold-loads a model on first request (downloads
            # from HF if absent), and a heavy multi-turn probe can take minutes.
            with urllib.request.urlopen(req, timeout=600) as resp:
                return self._consume(resp, should_abort, tools)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"API {e.code}: {detail}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(
                f"request to {_base()} failed: {e}\n"
                f"start it with: mlx_lm.server --model <repo> --port 8080") from None


def _make_client(cfg):
    return MLXClient(cfg)


PROVIDER = {
    "name":           _PROVIDER,
    "description":    "MLX (Apple Silicon) via mlx_lm.server - local, OpenAI-compatible",
    "make_client":    _make_client,
    "list_models":    lambda: MLXClient(_lc.get("_cfg")).list_models() if _lc.get("_cfg") else [],
    "context_window": lambda model: _CTX_DEFAULT,
    "available":      lambda: True,   # local + free; always offerable (like ollama)
    "autostart":      lambda: False,  # don't auto-activate; operator selects it explicitly
    "tag":            "mlx",
    "capabilities":   lambda model: {},
}


def setup(lc, cfg):
    """Helpers-only hook. Stash core namespace + live cfg so list_models can build a
    client outside a chat turn. Do NOT call register_provider here (the manager does)."""
    global _lc
    _lc = dict(lc)
    _lc["_cfg"] = cfg
