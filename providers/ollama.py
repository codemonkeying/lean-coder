# Bundled provider: Ollama (the out-of-the-box default backend).
#
# Ollama is a provider like any other - this file owns its HTTP client, tiered
# host failover, per-host default models, num_ctx auto-detect + OOM cap, and warm
# ("loaded") model hints. Core stays backend-agnostic: it knows the provider spec,
# not Ollama.
#
# Shared low-level helpers (colours, Spinner, the markdown styler, the menu
# primitives, host config helpers, save_config, _ask) live in core and are handed
# to setup(lc, cfg) as the `lc` namespace; setup() injects the stable ones into
# this module's globals so the moved code resolves them as plain names, and reads
# genuinely-mutable core state (_active_composer) live via `_lc`.
#
# Transport config (host / hosts / machines / num_ctx / host_models) stays
# top-level in config.toml and is read straight off cfg here - it is connection
# infrastructure, not a flat per-provider knob.

import json
import urllib.request
import urllib.error
import re
import sys
import threading

# --- core namespace, populated by setup(lc, cfg) ----------------------------
# _cfg: the live Config (PROVIDER callables with no cfg arg read it).
# _lc:  core's globals() dict - the live source for mutable state (_active_composer).
_cfg = None
_lc = None

# Names injected from core into this module's globals at setup() time. Stable
# (functions / constants / one-shot flags); mutable state is read via _lc instead.
_INJECT = (
    "green", "red", "yellow", "dim", "bold", "blue",
    "Spinner", "THINK_FRAMES", "GLYPH", "MarkdownStream",
    "parse_text_tool_calls", "_TurnAbort",
    "_menu_index", "pick_model_menu", "model_available",
    "_norm_host", "resolve_host", "host_label", "valid_host",
    "_ask", "save_config", "_TTY",
    "stream_tiered", "StreamStall",
)


# ----------------------------------------------------------------------------
# num_ctx detection + OOM cap
# ----------------------------------------------------------------------------

AUTO_CTX_CAP = 32768           # auto-detect never raises num_ctx above this (OOM guard)


def _parse_num_ctx(show: dict):
    """Extract a context length from an /api/show response, and whether it came
    from an explicit num_ctx PIN in the model's parameters (a deliberate
    Modelfile choice) vs. the architecture context_length (the raw max).
    Pure function (no I/O) for testability. Returns (int or None, pinned: bool)."""
    params = show.get("parameters")
    if isinstance(params, str):
        m = re.search(r"(?m)^\s*num_ctx\s+(\d+)", params)
        if m:
            return int(m.group(1)), True
    info = show.get("model_info") or {}
    for k, v in info.items():
        if k.endswith(".context_length") and isinstance(v, int):
            return v, False
    return None, False


def cap_num_ctx(detected, cap: int = AUTO_CTX_CAP, pinned: bool = False):
    """Apply the auto-detect safety policy: never raise num_ctx above `cap`
    (a model's architectural max can be huge, e.g. 262144 -> OOM); only lower it
    when the model's own window is smaller. An explicit --num-ctx bypasses this,
    and so does an explicit num_ctx PIN baked into the model's Modelfile
    (`pinned=True`): the operator already chose a deliberate window there, so we
    honour it rather than clamp it to the generic OOM cap.
    Returns (value_or_None, was_capped). Pure function for testability."""
    if detected is None:
        return None, False
    if pinned:
        return detected, False
    return min(detected, cap), detected > cap


# ----------------------------------------------------------------------------
# Small-model classification (measured, not name-matched)
# ----------------------------------------------------------------------------
# Research (Aider leaderboard, Cline#11272, Goose#6882, llama.cpp#9639) is
# conclusive: small local models fall out of the native tool_calls channel and
# emit calls as TEXT. They need a little extra tool-channel steering; big models
# don't. We classify PURELY by measured parameter size from /api/show - never by
# model name/family. No table to maintain, no greedy substring traps (a
# "qwen3-coder:30b" is not the same beast as "qwen3:4b"). Unknown size -> we
# assume small: the only consequence is a short prompt addendum, which is a
# harmless no-op on a capable model.
SMALL_PARAM_CEILING_B = 8.0        # <= this many billion params = "small"


def parse_param_b(param_size):
    """'4.0B' / '30.5B' / '540M' -> float billions, or None. Pure."""
    if not param_size or not isinstance(param_size, str):
        return None
    m = re.match(r"\s*([\d.]+)\s*([BM])", param_size.strip(), re.I)
    if not m:
        return None
    val = float(m.group(1))
    return val / 1000.0 if m.group(2).upper() == "M" else val


def is_small_model(param_b) -> bool:
    """True if a model of `param_b` billion params is 'small' (needs tool-channel
    steering). Unknown size (None) -> True (safe: the addendum is a no-op on a
    strong model). Measured-size only; no name matching. Pure/testable."""
    if param_b is None:
        return True
    return param_b <= SMALL_PARAM_CEILING_B


# ----------------------------------------------------------------------------
# Host reachability + tiered failover
# ----------------------------------------------------------------------------

def host_alive(host: str, timeout: float = 2.0) -> bool:
    """Quick reachability probe - HTTP 200 from {host}/api/tags."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def select_host(hosts, timeout: float = 2.0, probe=host_alive, on_probe=None):
    """Probe a tiered host list in priority order; return (host, alive) for the
    first that responds. If none do, fall back to the highest-priority host with
    alive=False. `probe`/`on_probe` are injectable for testing."""
    for h in hosts:
        alive = probe(h, timeout)
        if on_probe:
            on_probe(h, alive)
        if alive:
            return h, True
    return (hosts[0] if hosts else None), False


def _status_dot(alive: bool) -> str:
    """Green dot for reachable/warm, red for not. Pure (color only when TTY)."""
    return green("*") if alive else red("*")


def _probe_all(urls, probe=host_alive, timeout: float = 2.0) -> dict:
    """Probe urls concurrently; return {url: alive}. Parallel so a few dead
    hosts don't stall the menu by their timeouts. `probe` injectable for tests."""
    results = {}

    def work(u):
        results[u] = probe(u, timeout)

    threads = [threading.Thread(target=work, args=(u,), daemon=True) for u in urls]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 1)
    return results


def pick_host_menu(machines, current, probe=host_alive, prompt=input, pool=None):
    """Numbered host picker with a live reachability dot. Lists known machines,
    the priority-pool hosts (cfg.hosts - a bare `hosts = [...]` list with no
    [machines] names), and the current host; probes them in parallel and returns
    the chosen base URL (or None on cancel). `probe`/`prompt` injectable for tests."""
    items, seen = [], set()
    for name, url in machines.items():
        nu = _norm_host(url)
        if nu not in seen:
            items.append((name, nu))
            seen.add(nu)
    # Pool hosts (cfg.hosts): the priority list may hold URLs that aren't named in
    # [machines] - include them so the picker never hides a configured host. Reuse a
    # machines name if one maps to the same URL; otherwise label with the bare URL.
    url_to_name = {_norm_host(u): n for n, u in machines.items()}
    for url in (pool or []):
        nu = _norm_host(url)
        if nu not in seen:
            items.append((url_to_name.get(nu, nu), nu))
            seen.add(nu)
    cur = _norm_host(current) if current else None
    if cur and cur not in seen:
        items.append(("(current)", cur))
        seen.add(cur)
    if not items:
        print(dim("no known hosts; use /ollama host <url> or add [machines] to the config."))
        return None
    status = _probe_all([u for _, u in items], probe)
    print(bold("hosts:") + dim("  (" + green("*") + " up  " + red("*") + " down)"))
    for i, (name, url) in enumerate(items, 1):
        mark = dim("  (current)") if url == cur else ""
        print(f"  {i}) {_status_dot(status.get(url, False))} {name}  {dim(url)}{mark}")
    print("  0) cancel")
    try:
        sel = prompt("choice: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    idx = _menu_index(sel, len(items))
    return items[idx][1] if idx is not None else None


# ----------------------------------------------------------------------------
# Per-host default model + availability check
# ----------------------------------------------------------------------------

def remember_host_model(cfg, model=None):
    """Silently record `model` (default: the active model) as the last-used model for
    cfg.host, Anthropic-style - no 'make default?' prompt. On the next connect to this
    host it's adopted automatically. Keyed by the SAME normalized url as cfg.host so
    host <-> host_models can't drift out of sync. A --model/env pin (model_explicit) is
    a one-off override, NOT a preference - it never clobbers the host's remembered model."""
    if cfg.model_explicit:
        return
    m = model or cfg.active_model()
    key = _norm_host(cfg.host)
    if m and cfg.host_models.get(key) != m:
        cfg.host_models[key] = m
        save_config(cfg)


def ensure_model(client, cfg, interactive=None):
    """If cfg.model isn't available on the selected host, list what is and (when
    interactive) let the user pick one, remembering it as the host's last-used model.
    No-op when the model is present or the host can't be queried - crucially we only
    ever query the ACTIVE host (never a dead remote), so a switch can't reach the wrong
    box's model list."""
    if interactive is None:
        interactive = _TTY and sys.stdin.isatty()
    available = client.list_models()
    if not available:
        return                      # host unreachable / no /api/tags -> can't verify
    cur = cfg.active_model()
    if model_available(cur, available):
        remember_host_model(cfg, cur)   # confirmed present here -> remember silently
        return                      # present (exact, or only differs by implicit :latest)
    print(yellow(f"model '{cur}' is not available on {cfg.host}."))
    if not interactive:
        print(dim("available: " + ", ".join(available)))
        print(yellow(f"select with /model <name>, pass --model, or: ollama pull {cur}"))
        return
    chosen = pick_model_menu(available)
    if not chosen:
        print(dim(f"keeping '{cur}' (not installed; sends will fail until pulled)."))
        return
    cfg.set_active_model(chosen)
    remember_host_model(cfg, chosen)    # last-used, saved silently


def apply_host_default_model(client, cfg, respect_explicit=True, announce=False):
    """Adopt the host's last-used model for cfg.host (if remembered) and verify it's
    installed there, offering a picker when it's missing. Used at startup and on a
    runtime host switch. `respect_explicit` keeps a --model/env pin in place;
    `announce` prints when the model changes. host_models is keyed by the normalized
    host so lookup matches whatever _norm_host produced for cfg.host. ensure_model
    only ever queries the ACTIVE host, so this never reaches through a dead remote."""
    if not (respect_explicit and cfg.model_explicit):
        m = cfg.host_models.get(_norm_host(cfg.host))
        if m and m != cfg.active_model():
            cfg.set_active_model(m)
            if announce:
                print(dim(f"model -> {m} (last used on this host)"))
    ensure_model(client, cfg)


# ----------------------------------------------------------------------------
# The Ollama client (stdlib urllib, streaming NDJSON)
# ----------------------------------------------------------------------------

def _friendly_error(err, cfg):
    """Turn a raw Ollama /api/chat error string into an actionable message.

    The common trap: the selected model can't tool-call, so Ollama rejects the
    request's `tools` array with '<model> does not support tools'. lean-coder is
    tool-first, so a bare error leaves the user stuck. Point them at the two real
    fixes: switch to a tool-capable model, or run chat-only (/leash chat). Any
    other error is passed through unchanged."""
    s = str(err)
    if "does not support tools" in s.lower():
        model = cfg.active_model()
        return (
            f"the model '{model}' doesn't support tool calling, which lean-coder "
            f"uses for its file/command tools.\n"
            f"  - for pure conversation (no tools): run  /leash chat  "
            f"(or start with --chat-only), then retry;\n"
            f"  - to keep the tools: switch to a tool-capable model with  /ollama model  "
            f"(e.g. a qwen3-coder / llama3.1 / mistral-nemo build).")
    return s


class OllamaClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self._models = None  # cached model-name list for completion
        self._no_think = set()  # models that rejected `think`; don't re-send this session

    def _notify_no_think(self, model):
        """Record that `model` rejected `think` and, the first time, tell the user
        plainly (so a manual /think on a non-thinking model doesn't just vanish).
        Silent on repeats."""
        first = model not in self._no_think
        self._no_think.add(model)
        if first:
            try:
                print(dim(f"note: {model} doesn't support thinking - "
                          f"continuing without it (think stays off for it)."))
            except Exception:
                pass

    def _get_json(self, path, body=None, timeout=10):
        """Non-streaming JSON request; GET if body is None, else POST."""
        url = f"{self.cfg.host}{path}"
        if body is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def detect_num_ctx(self):
        """Read the model's context window from `ollama show`. None on failure."""
        try:
            return _parse_num_ctx(self._get_json("/api/show",
                                                 {"model": self.cfg.active_model()}))[0]
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            return None

    def list_models(self):
        """Available model names from /api/tags (cached). [] on failure."""
        if self._models is None:
            try:
                data = self._get_json("/api/tags")
                self._models = [m["name"] for m in data.get("models", [])]
            except (urllib.error.URLError, OSError, ValueError,
                    json.JSONDecodeError, KeyError):
                self._models = []
        return self._models

    def running_models(self):
        """Currently loaded ('warm') model names from /api/ps. [] on failure.
        Not cached: warm state changes as the server loads/evicts."""
        try:
            data = self._get_json("/api/ps")
            return [m["name"] for m in data.get("models", [])]
        except (urllib.error.URLError, OSError, ValueError,
                json.JSONDecodeError, KeyError):
            return []

    def chat(self, messages, tools, should_abort=None):
        """Stream /api/chat. Prints assistant content as it arrives; reasoning
        (message.thinking, when /think is on) is shown dimmed and NOT stored.
        Returns (assistant_message_dict, prompt_eval_count_or_None, aborted)."""
        model = self.cfg.active_model()
        content, tool_calls, prompt_eval, aborted = self._stream_chat(
            model, messages, tools, should_abort)
        content = "".join(content)
        # Fallback: only when the model emitted NO native tool_calls, look for
        # text-encoded ones in the content (leaves the native path untouched).
        if not tool_calls and not aborted:
            # Only accept a parsed call whose name is an ACTIVE tool (guards a
            # model that merely lists/quotes a tool in prose from being fired).
            known = {(_t.get("function") or _t).get("name") for _t in (tools or [])}
            known.discard(None)
            # name -> param properties, so the text parser can coerce Qwen-XML
            # string args to their declared boolean/integer/number types.
            _schemas = {}
            for _t in (tools or []):
                _fn = _t.get("function") or _t
                _nm = _fn.get("name")
                _props = (_fn.get("parameters") or {}).get("properties")
                if _nm and isinstance(_props, dict):
                    _schemas[_nm] = _props
            parsed, cleaned = parse_text_tool_calls(
                content, known_names=known or None, schemas=_schemas or None)
            if parsed:
                print(dim(f"(parsed {len(parsed)} tool call(s) from text)"))
                tool_calls = parsed
                content = cleaned
        assistant = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, prompt_eval, aborted

    def _preload(self, model):
        """Block until `model` is resident in the server's memory. Ollama loads a
        cold model on an empty-messages request (stream:false), returning only once
        it's in VRAM (done_reason == "load"); a streaming turn fired against a cold
        model bails empty immediately instead of waiting, so we preload here before
        retrying. Best-effort: any failure just means the retry runs unwarmed."""
        try:
            body = {"model": model, "messages": [], "stream": False}
            if self.cfg.keep_alive:
                body["keep_alive"] = self.cfg.keep_alive
            self._get_json("/api/chat", body=body, timeout=300)
            return True
        except (urllib.error.URLError, OSError, ValueError,
                json.JSONDecodeError) as e:
            return False

    def _build_payload(self, model, messages, tools):
        payload = {
            "model": model,
            "stream": True,
            "options": self.cfg.options(),
            "messages": messages,
            "tools": tools,
        }
        # `think` is a global toggle, but some models reject it outright
        # ("does not support thinking"). Skip it for any model we've seen
        # reject it this session so it Just Works without user fiddling.
        if self.cfg.think is not None and model not in self._no_think:
            payload["think"] = self.cfg.think
        if self.cfg.keep_alive:
            payload["keep_alive"] = self.cfg.keep_alive
        return payload

    def _stream_chat(self, model, messages, tools, should_abort, _cold_retried=False):
        """Stream one /api/chat request. Returns
        (content_parts, tool_calls, prompt_eval, aborted). Retries once without
        `think` if the model reports it doesn't support thinking, and retries once
        on a cold-load blank (see below)."""
        url = f"{self.cfg.host}/api/chat"
        payload = self._build_payload(model, messages, tools)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})

        content_parts: list[str] = []
        tool_calls: list = []
        prompt_eval = None
        printed_any = False
        printed_think = False
        md = None                         # markdown styler, created on first content
        aborted = False
        saw_done = False                  # did a done:true frame ever arrive?
        done_reason = None                # 'load' = cold-load ack, not a real turn
        self.last_out_tokens = None       # set on `done`; cleared so a failed call can't restack
        spin = Spinner("thinking", THINK_FRAMES).start()  # until first token
        # 3-tier timeout (shared core helper): open with the connect deadline, then
        # stream_tiered re-arms the socket to the TTFT deadline until the first line
        # and to the inter-token idle deadline after, raising StreamStall (a
        # ConnectionError, so core's next-model fallback still fires) on a stall.
        # should_abort semantics are untouched. Prod defaults (ttft=600, idle=None)
        # reproduce the old single 600s socket timeout.
        connect_to = getattr(self.cfg, "gen_connect_timeout", None) or 600
        try:
            with urllib.request.urlopen(req, timeout=connect_to) as resp:
                for raw in stream_tiered(resp, self.cfg, self.cfg.host):
                    if should_abort and should_abort():
                        aborted = True
                        break
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        err = obj["error"]
                        # Model rejects `think`: remember it, retry without it.
                        if ("think" in err.lower()
                                and "support" in err.lower()
                                and "think" in payload
                                and not printed_any and not printed_think):
                            self._notify_no_think(model)
                            spin.stop()
                            return self._stream_chat(
                                model, messages, tools, should_abort)
                        raise RuntimeError(_friendly_error(err, self.cfg))
                    msg = obj.get("message", {})
                    think = msg.get("thinking")
                    if think:                       # reasoning - show, don't store
                        if not printed_think:
                            if _lc["_active_composer"] is None:   # composer: keep status up
                                spin.stop()
                            sys.stdout.write(dim(GLYPH["think"] + " "))
                            printed_think = True
                        sys.stdout.write(dim(think))
                        sys.stdout.flush()
                    chunk = msg.get("content")
                    if chunk:
                        if not printed_any:
                            if _lc["_active_composer"] is None:   # composer: keep status up
                                spin.stop()
                            if printed_think:
                                sys.stdout.write("\n")
                            sys.stdout.write(blue(GLYPH["bullet"] + " "))
                            printed_any = True
                            md = MarkdownStream(sys.stdout.write)
                        md.feed(chunk)
                        sys.stdout.flush()
                        content_parts.append(chunk)
                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])
                    if obj.get("done"):
                        saw_done = True
                        done_reason = obj.get("done_reason")
                        prompt_eval = obj.get("prompt_eval_count")
                        # report output tokens for the core session counter
                        self.last_out_tokens = obj.get("eval_count")
        except _TurnAbort:
            aborted = True
        except urllib.error.HTTPError as e:
            # Some servers reject an unsupported option up front with an HTTP 4xx
            # (body carries {"error": "...does not support thinking"}) rather than
            # a streamed error line - so we never entered the loop. Read the body,
            # apply the same think-retry, else surface a friendly error.
            try:
                body = e.read().decode("utf-8", "replace")
                err = json.loads(body).get("error", body)
            except Exception:
                err = getattr(e, "reason", str(e))
            spin.stop()
            if ("think" in err.lower() and "support" in err.lower()
                    and "think" in payload):
                self._notify_no_think(model)
                return self._stream_chat(model, messages, tools, should_abort)
            raise RuntimeError(_friendly_error(err, self.cfg)) from None
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"can't reach Ollama at {self.cfg.host} ({e.reason}). "
                f"Is it running? Try: curl {self.cfg.host}/api/tags") from None
        finally:
            spin.stop()
        if md and not aborted:
            md.flush()                    # write any styled partial last line
        if printed_any or printed_think:
            print()
        # Cold-load blank: on a cold model, ollama can emit a placeholder frame and
        # end the stream WITHOUT any content/tool_calls - either a done:true with
        # done_reason == "load" (per the API docs) or the stream just closing before
        # done:true (the streaming variant, seen on remote hosts). That first call is
        # what triggers the VRAM load; the immediate retry hits a warm model and
        # actually answers. Distinguishable from a genuine empty turn (which carries
        # done:true + done_reason == "stop"). Retry once, guarded so it can't loop.
        cold_blank = (not aborted and not content_parts and not tool_calls
                      and (not saw_done or done_reason == "load"))
        if cold_blank and not _cold_retried:
            # The cold call already kicked off the load but returned before it
            # finished; block until the model is resident, then retry the stream once.
            self._preload(model)
            return self._stream_chat(model, messages, tools, should_abort,
                                     _cold_retried=True)
        return content_parts, tool_calls, prompt_eval, aborted


# ----------------------------------------------------------------------------
# PROVIDER spec hooks
# ----------------------------------------------------------------------------

def _context_window(model):
    """Honour an explicit num_ctx pin; otherwise detect from `ollama show` and
    apply the OOM cap. Records the model's max on cfg._ollama_capped_from when it
    had to cap, so the startup banner can mention it."""
    cfg = _cfg
    cfg._ollama_capped_from = None        # banner note: model max when capped
    if not cfg.auto_num_ctx and cfg.num_ctx:
        return cfg.num_ctx
    detected, pinned = None, False
    try:
        detected, pinned = _parse_num_ctx(
            OllamaClient(cfg)._get_json("/api/show", {"model": model}))
    except Exception:
        detected, pinned = None, False
    value, was_capped = cap_num_ctx(detected, pinned=pinned)
    if was_capped:
        cfg._ollama_capped_from = detected
    return value or cfg.num_ctx or AUTO_CTX_CAP


def _location(cfg):
    """Status/info label for the active backend: 'name (url)' when the host has a
    machine alias, else the bare URL."""
    return host_label(cfg.host, cfg.machines)


def _endpoints():
    """Tiered failover host list (empty when not configured)."""
    return list(_cfg.hosts)


def _failover(agent, cfg, announce=True):
    """Re-probe the priority pool in order and switch the ACTIVE host to the first
    live one (skipping the current, dead one). Returns True on a successful switch.
    The pool order is never changed - this only moves the active pick. Used by the
    on_conn_error hook (mid-session recovery) and /ollama failover."""
    pool = list(cfg.hosts) or [cfg.host]
    for h in pool:
        if _norm_host(h) == _norm_host(cfg.host):
            continue                 # the one that just failed - skip
        if host_alive(h):
            if announce:
                print(yellow(f"~ {host_label(cfg.host, cfg.machines)} unreachable "
                             f"-> failed over to {host_label(h, cfg.machines)}"))
            _activate_host(agent, cfg, h, announce=False)
            return True
    return False


def _on_conn_error(agent, cfg, err):
    """Transport-recovery hook: on a send failure, fail over to the next live host in
    the priority pool and ask core to retry the turn. No live alternative -> False
    (core prints the original error)."""
    return _failover(agent, cfg)


def _prepare(agent, cfg):
    """Pre-activation transport setup, run by core before activate_provider():
    probe the priority pool in order and pick the first live host (best-in-priority),
    then adopt the host's last-used model + availability check. Multi-host announces
    the probe; a single-host pool stays quiet unless it's down. Builds its own client
    (it owns the class); the client reads cfg.host live."""
    if len(cfg.hosts) > 1:
        print(dim("selecting host (priority order)..."))

        def _rep(h, up):
            lbl = host_label(h, cfg.machines)
            print((green(f"  {GLYPH['ok']} ") + dim(lbl)) if up
                  else (dim(f"  {GLYPH['no']} " + lbl + "  no response")))
        chosen, alive = select_host(cfg.hosts, on_probe=_rep)
        cfg.host = chosen
        if not alive:
            print(yellow(f"  no host responded - using {host_label(chosen, cfg.machines)} "
                         f"(may be offline; /ollama failover to retry, /ollama host to switch)"))
    elif cfg.hosts:
        cfg.host = cfg.hosts[0]
    apply_host_default_model(OllamaClient(cfg), cfg)


# Small-model tool-calling guidance appended to the system prompt. Research
# (Unsloth #4783: few-shot +15-30%; hive#4222: "don't write calls as text" cut
# leakage ~80%; hermes#5867: explicit channel stabilises qwen3). Kept short so it
# barely dents the context of a small model. Note: no edit-format steer here - we
# keep apply_diff for every model (its guardrail stops a botched edit from nuking
# a file; whole-file rewrites risk silent content loss), so we don't push small
# models off it.
_SMALL_TOOL_ADDENDUM = (
    "\n\nTOOL USE (important): when you need to act, CALL THE TOOL via the tool "
    "interface - do NOT write the call as JSON or XML text in your reply, and do "
    "not print it. One tool call per step; wait for its result before the next. "
    "Only use a tool that is actually in your tool list."
)


def _system_addendum(agent, cfg):
    """Extra system-prompt text for the ACTIVE model. Small models (by measured
    parameter size) get explicit tool-channel guidance; capable models get nothing.
    Size is read from /api/show (best-effort; ollama caches it). Cache-safe for
    core: only varies with model."""
    model = cfg.active_model()
    return _SMALL_TOOL_ADDENDUM if is_small_model(_param_b_for(cfg, model)) else ""


def _param_b_for(cfg, model):
    """Best-effort parameter size (billions) for `model` via /api/show. None on any
    failure. Not cached here - called rarely (prompt rebuild on model change)."""
    try:
        client = OllamaClient(cfg)
        show = client._get_json("/api/show", body={"model": model}, timeout=8)
        return parse_param_b((show.get("details") or {}).get("parameter_size"))
    except Exception:
        return None


PROVIDER = {
    "name": "ollama",
    "description": "Ollama - local/self-hosted models (the bundled default backend)",
    "make_client": lambda c: OllamaClient(c),
    "list_models": lambda: OllamaClient(_cfg).list_models(),
    "context_window": _context_window,
    "available": lambda: True,         # localhost is the floor; activation/host probe handle reachability
    "warm_models": lambda: OllamaClient(_cfg).running_models(),
    "model_status": lambda m: None,    # availability handled by the startup host probe
    "location": _location,             # status/info label (host alias + url)
    "endpoints": _endpoints,           # multi-host failover list
    "prepare": _prepare,               # pre-activation host failover + per-host model
    "on_conn_error": _on_conn_error,   # mid-session transport recovery (pool failover)
    "system_addendum": _system_addendum,  # small-model tool-calling guidance
    "tag": "ollama",
}


# ----------------------------------------------------------------------------
# /ollama command (registered by this provider, not a core built-in)
#   /ollama                 -> host picker, then a model picker for it (two-step)
#   /ollama host [name|url] -> switch the ACTIVE host (session override; no arg = picker)
#   /ollama model [name]    -> switch the model on the active host (no arg = picker)
#   /ollama failover        -> re-probe the pool + jump to the best live host (recover)
#   /ollama reset           -> rebuild the pool from config + re-probe from scratch
#   /ollama priority [a..]  -> show / persist the priority order (deliberate reorder)
# The priority pool (cfg.hosts) is config order and never auto-reordered; the active
# host (cfg.host) is a session pick. Launch + failover choose best-in-priority-order.
# ----------------------------------------------------------------------------

def _activate_host(agent, cfg, target, announce=True):
    """Make `target` the active ollama host for this session: adopt its remembered
    model (verify/pick if missing), re-activate ollama so the client + context window
    track it, and autosave.

    A MULTI-host pool (a real priority list) is left intact - the active pick is a
    session override, not a priority change. But a SINGLE-host pool has no priority
    order to preserve: it's just 'the host', so switching updates the pool entry too -
    otherwise save_config would persist the pool's stale host and the switch would be
    silently lost on the next launch (cfg.host moved, cfg.hosts didn't)."""
    cfg.host = target
    if len(cfg.hosts) <= 1:
        cfg.hosts = [target]
    if announce:
        print(dim(f"host -> {host_label(cfg.host, cfg.machines)}"))
    apply_host_default_model(OllamaClient(cfg), cfg,
                             respect_explicit=False, announce=announce)
    agent.activate_provider("ollama")
    _lc["autosave_config"](cfg)


def _switch_host(agent, cfg, arg):
    """Switch the ACTIVE ollama host (a session override; the priority pool is left
    intact). Validates + normalizes the target, then probes it BEFORE committing so a
    typo or a dead box is never silently adopted: an unreachable host asks 'use anyway?'
    (for one that just isn't up yet) and otherwise leaves cfg untouched. No arg opens
    the host picker."""
    if arg:
        if not valid_host(arg) and arg not in cfg.machines:
            print(red(f"'{arg}' is not a usable host (need a name, host:port, or URL)."))
            return
        target = resolve_host(arg, cfg.machines)
    else:
        target = pick_host_menu(cfg.machines, cfg.host, pool=cfg.hosts)
    if not target or _norm_host(target) == _norm_host(cfg.host):
        return
    target = _norm_host(target)
    if not host_alive(target):
        print(yellow(f"{host_label(target, cfg.machines)} isn't responding."))
        if not _ask("switch to it anyway (e.g. it's not up yet)?"):
            print(dim("host unchanged."))
            return
    _activate_host(agent, cfg, target)


def _switch_model(agent, cfg, arg):
    """Pick/switch the model on the current ollama host. Delegates to core's unified
    /model handler so behaviour (loaded-dot, per-host default save) stays identical."""
    _lc["handle_model_command"](agent, cfg, arg)


def _do_failover(agent, cfg):
    """/ollama failover - re-probe the priority pool now and jump to the best live
    host. The recover button: heals a fractured state (stuck on a dead host) with no
    config editing. No-op message when the current host is the only live one."""
    if _failover(agent, cfg, announce=True):
        return
    if host_alive(cfg.host):
        print(dim(f"staying on {host_label(cfg.host, cfg.machines)} "
                  f"(no other host in the pool responded)."))
    else:
        print(yellow("no host in the priority pool responded - "
                     "/ollama host <url> to point somewhere, or start ollama."))


def _do_reset(agent, cfg):
    """/ollama reset - rebuild the priority pool from the configured machines +
    current host, then re-probe from scratch and activate the best live host. Nukes a
    fractured RUNTIME state without hand-editing config."""
    pool = list(cfg.hosts) or [cfg.host]
    for u in cfg.machines.values():           # fold in every known machine
        if u not in pool:
            pool.append(u)
    cfg.hosts = pool
    print(dim("re-probing hosts..."))

    def _rep(h, up):
        lbl = host_label(h, cfg.machines)
        print((green(f"  {GLYPH['ok']} ") + dim(lbl)) if up
              else (dim(f"  {GLYPH['no']} " + lbl + "  no response")))
    chosen, alive = select_host(cfg.hosts, on_probe=_rep)
    if chosen:
        _activate_host(agent, cfg, chosen, announce=True)
        if not alive:
            print(yellow(f"  no host responded - using {host_label(chosen, cfg.machines)} "
                         f"(may be offline)."))


def _print_priority(order, cfg, status, header="priority order:"):
    """Render a numbered priority list with live dots + the active marker. Pure I/O."""
    print(bold(header) + dim("  (" + green("*") + " up  " + red("*") + " down)"))
    for i, h in enumerate(order, 1):
        mark = dim("  (active)") if _norm_host(h) == _norm_host(cfg.host) else ""
        print(f"  {i}) {_status_dot(status.get(h, False))} {host_label(h, cfg.machines)}{mark}")


def _persist_priority(agent, cfg, order):
    """Commit a new priority order to cfg + save. Keeps the active pick when it's still
    in the pool, else drops it to the head (highest priority)."""
    cfg.hosts = list(order)
    if _norm_host(cfg.host) not in [_norm_host(h) for h in order]:
        cfg.host = order[0]
    _lc["autosave_config"](cfg)
    print(dim("priority order -> " + ", ".join(host_label(h, cfg.machines) for h in order)))


def _edit_priority_menu(agent, cfg, prompt=input):
    """Interactive priority editor (BIOS-style). Move a slot up/down, remove one, or
    add a host; nothing is persisted until you save. An empty line is a no-op (discard
    + exit) exactly like the model/host pickers - so a stray Enter never rewrites the
    order. `prompt` injectable for tests."""
    order = list(cfg.hosts) or [cfg.host]
    status = _probe_all(order)
    dirty = False
    while True:
        print()
        _print_priority(order, cfg, status)
        print(dim("  <n> u/d move up/down   -<n> remove   +<host> add   "
                  "s save   Enter cancel"))
        try:
            raw = prompt("edit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:                                   # bare Enter -> no-op discard (house style)
            if dirty:
                print(dim("cancelled - priority unchanged."))
            return
        low = raw.lower()
        if low in ("s", "save"):
            if dirty:
                _persist_priority(agent, cfg, order)
            else:
                print(dim("no change."))
            return
        # move: "<n> u" / "<n> d" / "<n>u" / "<n>d"
        m = re.match(r"^(\d+)\s*([ud])$", low)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(order):
                j = i - 1 if m.group(2) == "u" else i + 1
                if 0 <= j < len(order):
                    order[i], order[j] = order[j], order[i]
                    dirty = True
                else:
                    print(dim("already at the edge."))
            else:
                print(red("no such slot."))
            continue
        # remove: "-<n>"
        m = re.match(r"^-\s*(\d+)$", raw)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(order):
                if len(order) == 1:
                    print(red("can't remove the last host."))
                else:
                    removed = order.pop(i)
                    status.pop(removed, None)
                    dirty = True
            else:
                print(red("no such slot."))
            continue
        # add: "+<host>"
        if raw.startswith("+"):
            tok = raw[1:].strip()
            if not tok or (not valid_host(tok) and tok not in cfg.machines):
                print(red(f"'{tok}' is not a usable host."))
                continue
            u = resolve_host(tok, cfg.machines)
            if u in order:
                print(dim("already in the list."))
            else:
                order.append(u)
                status[u] = host_alive(u)
                dirty = True
            continue
        print(dim("commands: <n> u | <n> d | -<n> | +<host> | s | Enter"))


def _set_priority(agent, cfg, arg):
    """/ollama priority [a b c] - manage the PERSISTED priority order (distinct from
    `/ollama host`, a session-only active pick). No args on a TTY opens the interactive
    editor (move/add/remove; Enter = no-op); no args without a TTY just prints the
    order. With args it rewrites the order to exactly those hosts. Tokens are
    validated/normalized like any host."""
    toks = (arg or "").split()
    if not toks:
        if _TTY and sys.stdin.isatty():
            _edit_priority_menu(agent, cfg)
        else:
            _print_priority(list(cfg.hosts) or [cfg.host], cfg,
                            _probe_all(list(cfg.hosts) or [cfg.host]))
            print(dim("  set with: /ollama priority <name|url> <name|url> ..."))
        return
    new = []
    for t in toks:
        if not valid_host(t) and t not in cfg.machines:
            print(red(f"'{t}' is not a usable host - priority unchanged."))
            return
        u = resolve_host(t, cfg.machines)
        if u not in new:
            new.append(u)
    _persist_priority(agent, cfg, new)


def _handle_ollama(agent, cfg, arg):
    """/ollama [sub] [value] - manage the ollama backend:
      /ollama                 host picker -> then a model picker for it (two-step)
      /ollama host [name|url] switch the ACTIVE host (session override; no arg = picker)
      /ollama model [name]    switch the model on the active host (no arg = picker)
      /ollama failover        re-probe the pool + jump to the best live host (recover)
      /ollama reset           rebuild the pool from config + re-probe from scratch
      /ollama priority [a..]  show / persist the priority order (deliberate reorder)
    A bare value that isn't a subcommand is treated as a host name/url."""
    if cfg.provider != "ollama":
        # Auto-switch rather than refuse: managing the ollama backend implies
        # you want to use it, so activate it here (falls back to a hint if the
        # local ollama isn't reachable).
        try:
            model = agent.activate_provider("ollama")
        except Exception as e:
            print(dim(f"/ollama needs the ollama backend, but it's not usable "
                      f"({e}). Start ollama, then retry."))
            return
        print(dim(f"switched to ollama ({model})."))
    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    if sub == "model":
        _switch_model(agent, cfg, rest)
    elif sub == "host":
        _switch_host(agent, cfg, rest)
    elif sub == "failover":
        _do_failover(agent, cfg)
    elif sub == "reset":
        _do_reset(agent, cfg)
    elif sub == "priority":
        _set_priority(agent, cfg, rest)
    elif sub == "":
        # two-step: pick a host, then pick a model on it.
        _switch_host(agent, cfg, "")
        _switch_model(agent, cfg, "")
    else:
        # no recognised subcommand -> treat the whole arg as a host (a URL/name),
        # so `/ollama gpu-box` and `/ollama http://...` still Just Work.
        _switch_host(agent, cfg, arg)


def _ollama_completer(agent, cfg):
    """Tab targets: the subcommands, then known host names/urls (so `/ollama <Tab>`
    offers hosts directly - the common case - alongside the verbs)."""
    return ["host", "model", "failover", "reset", "priority"] + list(cfg.machines)


def setup(lc, cfg):
    """Helpers-only startup hook (the manager calls this before register_provider).
    Inject the stable core helpers into this module's globals so the moved code
    resolves them as plain names, stash the live cfg + core namespace, and register
    the /ollama command. ollama is default-enabled in core (it seeds providers_enabled
    before the dir scan), since the manager only runs setup() for already-enabled
    providers - so enablement can't be done from here."""
    global _cfg, _lc
    _cfg, _lc = cfg, lc
    g = globals()
    for k in _INJECT:
        if k in lc:
            g[k] = lc[k]
    lc["register_command"]("/ollama", _handle_ollama,
                           "ollama: host|model|failover|reset|priority (no arg = pick host+model)",
                           completer=_ollama_completer)
