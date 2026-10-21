"""lean-coder provider plugin: MLX (Apple Silicon native, via mlx_lm.server).

Apple Silicon (M-series) runs models fastest through MLX (unified memory, no
copy). mlx_lm.server exposes an OpenAI-compatible HTTP API, so lean-coder can
drive an MLX model on a Mac exactly like any OpenAI-shaped backend. Target box
in our lineup: the M4 Mac (216) - serves qwen3-14b MLX-4bit natively instead of
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
from pathlib import Path

_PROVIDER    = "mlx"
# Base URL of the running mlx_lm.server. Override per-box via env.
_API_BASE    = os.environ.get("MLX_HOST", "http://127.0.0.1:8080")
_CTX_DEFAULT = 32_768        # match the harness AUTO_CTX_CAP; MLX models often 256K-capable
# cfg.think level -> chat_template_kwargs.reasoning_effort (same map as llamacpp).
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


def _headers():
    h = {"Content-Type": "application/json"}
    key = os.environ.get("MLX_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _ctx_window(cfg=None):
    """Context window: provider_settings.mlx 'ctx' or 'max_kv_size' (matching the
    server's --max-kv-size), else _CTX_DEFAULT."""
    cfg = cfg or _lc.get("_cfg")
    try:
        ps = (getattr(cfg, "provider_settings", None) or {}).get(_PROVIDER, {})
        for k in ("ctx", "max_kv_size"):
            v = ps.get(k)
            if v:
                return int(v)
    except (TypeError, ValueError, AttributeError):
        pass
    return _CTX_DEFAULT


def _base():
    # Re-read env each call so a per-run MLX_HOST (e.g. the Mac's LAN IP) is honoured.
    return os.environ.get("MLX_HOST", _API_BASE).rstrip("/")


def _gen_config_card(model):
    """Sampling defaults from the checkpoint's generation_config.json when the model
    is a local path or a cached HF repo - what mlx_lm.server applies for omitted
    fields. None when not found (remote server, unknown layout)."""
    import glob
    cands = []
    if model:
        p = Path(os.path.expanduser(model))
        cands.append(p / "generation_config.json")
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        cands += [Path(x) for x in glob.glob(
            str(hub / ("models--" + model.replace("/", "--")) / "snapshots" / "*" /
                "generation_config.json"))]
    for c in cands:
        try:
            g = json.loads(c.read_text())
        except (OSError, ValueError):
            continue
        card = {}
        for src, k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                       ("min_p", "min_p"), ("presence_penalty", "presence_penalty"),
                       ("repetition_penalty", "repeat_penalty")):
            v = g.get(src)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                card[k] = v
        return card or None
    return None


class MLXClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_out_tokens = None
        self.last_eval_ns = None
        self.last_eval_tokens = None
        self._models = None

    def list_models(self):
        """Available model ids from GET /v1/models (cached). [] on failure.
        Duck-types OllamaClient.list_models so ensure_model()-style checks work."""
        if self._models is None:
            try:
                req = urllib.request.Request(f"{_base()}/v1/models", headers=_headers(),
                                             method="GET")
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read())
                self._models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            except (urllib.error.URLError, OSError, ValueError, KeyError):
                self._models = []
        return self._models

    def running_models(self):
        return []

    def detect_num_ctx(self):
        return _ctx_window(self.cfg)

    def _consume(self, resp, should_abort, tools=None):
        """Parse a streamed OpenAI SSE response, printing content as it arrives and
        accumulating any tool_calls. Returns (assistant_dict, prompt_tokens, aborted).
        Mirrors providers/openai.py::_consume."""
        finish        = None
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
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
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
                            sys.stdout.write(_lc["blue"]((_lc.get("GLYPH") or {}).get("bullet", "*") + " "))
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
                            # Some servers return arguments as a parsed object rather
                            # than string fragments: concat strings, keep a dict whole.
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
            # reasoning shown but no content followed (e.g. a pure tool-call turn)
            sys.stdout.write("\n")
            sys.stdout.flush()

        tool_calls = []
        for idx in order:
            slot = tool_acc[idx]
            if slot.get("args_obj") is not None:
                args = slot["args_obj"]
            else:
                try:
                    args = json.loads("".join(slot["args"])) if slot["args"] else {}
                except Exception:
                    args = {}
            tool_calls.append({
                "id":       slot["id"] or f"call_{idx}",
                "type":     "function",
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
        if not aborted:
            _note_cut(self.cfg, finish)
        return assistant, prompt_total, aborted

    def resolve_sampling(self, full=False):
        """Effective sampling ({'params','source','sent'}); see llamacpp twin."""
        model = self.cfg.active_model() if self.cfg else ""
        sf = _lc.get("sampling_for")
        if sf is None:
            return ({}, {}, "server built-in") if full else None
        try:
            card = _gen_config_card(model)
        except Exception:
            card = None
        send, eff, src = sf(self.cfg, model, card, "generation_config.json")
        self.last_sampling = {"params": eff, "source": src, "sent": dict(send)}
        return (send, eff, src) if full else self.last_sampling

    def chat(self, messages, tools, should_abort=None):
        """POST /v1/chat/completions (streamed). messages/tools are already
        OpenAI-shaped (lean-coder's internal format IS OpenAI-compatible).
        Returns (assistant_dict, prompt_tokens_or_None, aborted)."""
        if should_abort and should_abort():
            return {"role": "assistant", "content": ""}, None, True

        # Sampling: send only what the user set (or the family table when no card is
        # known). Current mlx_lm / mlx-vlm servers apply the checkpoint's
        # generation_config.json for omitted fields - forcing a global value here
        # overrode the model card (pre-0.10.86 sent 0.7/0.8/20/1.05 every request).
        send, _eff, _src = self.resolve_sampling(full=True)
        body = {
            "model":          self.cfg.active_model(),
            "messages":       messages,
            # Always send it: mlx_lm.server's own default when omitted is small.
            # Unlimited (max_out 0) = the context window, the real ceiling.
            "max_tokens":     _max_out(self.cfg) or _ctx_window(self.cfg),
            "stream":         True,
            "stream_options": {"include_usage": True},
        }
        for k, v in send.items():
            body["repetition_penalty" if k == "repeat_penalty" else k] = v
        # Think toggle. The clean cross-backend way to disable thinking is
        # chat_template_kwargs={"enable_thinking": False} in the request body (the
        # server merges it into apply_chat_template). Newer mlx_lm honours this
        # per-request; older builds only read the --chat-template-args launch flag,
        # so ALSO launch the server with '{"enable_thinking":false}' for gold certs.
        t = getattr(self.cfg, "think", None)
        if t is not None:
            if isinstance(t, str) and t.strip().lower() in _THINK_OFF:
                t = False
            kw = {"enable_thinking": bool(t)}
            if isinstance(t, str):
                eff = _EFFORT.get(t.strip().lower())
                if eff:
                    kw["reasoning_effort"] = eff
            body["chat_template_kwargs"] = kw
        if tools:
            body["tools"] = tools

        req = urllib.request.Request(
            f"{_base()}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=_headers(),
            method="POST")
        try:
            # Generous timeout: MLX cold-loads a model on first request (downloads
            # from HF if absent), and a heavy multi-turn probe can take minutes.
            with urllib.request.urlopen(
                    req,
                    timeout=(getattr(self.cfg, "gen_connect_timeout", None) or 600)) as resp:
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
    "context_window": lambda model: _ctx_window(),
    "available":      lambda: True,   # local + free; always offerable (like ollama)
    "autostart":      lambda: False,  # don't auto-activate; operator selects it explicitly
    "tag":            "mlx",
    "capabilities":   lambda model: {"thinking": ["off", "on"]},
    "warm_models":    lambda: [],
}


def setup(lc, cfg):
    """Helpers-only hook. Stash core namespace + live cfg so list_models can build a
    client outside a chat turn. Do NOT call register_provider here (the manager does)."""
    global _lc
    _lc = dict(lc)
    _lc["_cfg"] = cfg
