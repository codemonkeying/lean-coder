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
        _dbg_swallow()
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
        """Read the streamed reply with core's shared OpenAI-style reader (mlx_lm
        streams reasoning as delta.reasoning_content, some builds delta.reasoning).
        Returns (assistant_dict, prompt_tokens, aborted)."""
        out = _lc["consume_openai_stream"](self, resp, should_abort, tools)
        if not out[2]:
            _note_cut(self.cfg, self.last_finish)
        return out

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
