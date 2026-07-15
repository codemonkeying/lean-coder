# Provider API

A **provider** teaches lean-coder to talk to a model backend other than the bundled
Ollama - a hosted API, an OAuth subscription, anything. It is its own plugin type
(NOT a lean-tool): a `.py` file with a module-level `PROVIDER` dict, dropped in the
`providers/` dir and toggled with `/provider enable|disable`. A provider never ships
to a `/connect` remote and exposes no model tool; it wires in through a native
registry, so it never has to monkey-patch core or mutate `cfg`.

Ollama is the bundled default (`provider = "ollama"`); several hosted providers ship
bundled too (`anthropic_api`, `gemini`, `groq`, `openrouter`), disabled until you
enable and key them. Config **autosaves** - no manual `/save` step.

To start your own, copy the annotated template at
[`examples/providers/example.py`](examples/providers/example.py).

## Registering

A file in the providers dir (`~/.config/leancoder/providers/`, or the repo's
`providers/`) declares a module-level `PROVIDER` dict and an optional, **helpers-only**
`setup(lc, cfg)`. The `ProviderManager` reads the dict and registers it for you - the
file does **not** call `register_provider` itself.

```python
# ~/.config/leancoder/providers/cloud.py
PROVIDER = {                       # module-level; the ProviderManager reads + registers it
    "name":           "cloud",
    "description":    "my hosted backend",         # shown in the /provider menu
    "make_client":    lambda cfg: ApiClient(cfg),  # required
    "list_models":    lambda: [...ids...],         # required; safe unauthed - cached list or [], never throws
    "context_window": lambda model: 200000,        # required
    # --- everything below is optional and absent-safe ---
    "capabilities":   lambda model: {              # declare settable knobs (see below)
        "thinking": ["off", "adaptive", "max"],    # [] / absent = unsupported
        "effort":   ["low", "med", "high"],
    },
    "available":      lambda: _Auth().is_logged_in(),  # provider offered only when True (default True)
    "autostart":      lambda: _Auth().is_active(),     # activate on launch when True (default False)
    "on_activate":    lambda agent, cfg: ...,          # called when switched to
    "on_deactivate":  lambda agent, cfg: ...,          # called when switched away
    "usage":          lambda agent, cfg: {...},        # quota meters; numbers, core renders (below)
    "status_line":    lambda agent, cfg: "raw string", # raw escape hatch if usage's model doesn't fit
    "login":          lambda agent, cfg: True,         # enables /provider login (returns truthy on success)
    "clear":          lambda agent, cfg: None,         # enables /provider clear (wipes credentials)
    "detail":         lambda agent, cfg: "full view",  # full account/billing view for /usage
    "tag":            "cloud",                         # short label in the unified /model row (default: name)
    "warm_models":    lambda: ["m1"],                  # loaded/instant models (green dot in /model, /usage)
    "model_status":   lambda model: None,              # install/availability hint in /model (e.g. "pull")
    "endpoints":      lambda: ["http://host:11434"],   # multi-endpoint failover (Ollama uses this)
}

def setup(lc, cfg):               # OPTIONAL, helpers-only - do NOT register_provider here
    _lc.update(lc)                # grab display helpers (Spinner, colours, MarkdownStream, ...)
    lc["register_command"]("/cloud_push", _push)   # bespoke provider commands are still fine
```

**Opt-in hooks are absent-safe:** implement only what you want; omitting a hook costs
nothing (core degrades gracefully). A hosted single-endpoint API returns `[]` from
`warm_models()` and omits `endpoints()`; the bundled Ollama provider implements both.
There is **no `on_error` hook** - retry/fallback lives inside the client
(`client.chat()`), with core's cross-provider next-model fallback as the backstop.

### First-class onboarding: `login` + recognizable auth errors

Two small conventions give your provider the same smooth OBE as the bundled ones:

1. **Implement `login`** (and `clear`). Core wires `login` to `/provider login <name>`,
   and `/provider login <name>` will even **enable** a bundled-but-disabled provider on
   the spot, so a new user goes straight from key to running. Read the secret with
   `getpass` (never echo it), store it `chmod 600` under `lc["CONFIG_DIR"]`, and return
   truthy on success / `False` if the user cancelled.
2. **Raise a recognizable auth error from `chat()`** when the key is missing or the API
   rejects it (HTTP 401/403). Core inspects the exception text and, if it looks like an
   auth failure, **offers the login flow inline and retries the turn once** - so a stale
   or absent key never dead-ends at a red error. Make the `RuntimeError` message contain
   one of: `no API key`, `API key`, `401`, `403`, or `unauthorized`. For example:

   ```python
   def chat(self, messages, tools, should_abort=None):
       key = _api_key()
       if not key:
           raise RuntimeError("no API key - use /provider login")   # -> core offers login + retries
       ...
       except urllib.error.HTTPError as e:
           if e.code in (401, 403):
               raise RuntimeError(f"API {e.code} - key rejected; /provider login")
   ```

   Returning a normal assistant message on an auth failure would look like a real reply
   and strand the user - **raise instead.** See
   [`examples/providers/example.py`](examples/providers/example.py) for the full pattern.

Secrets go under `lc["CONFIG_DIR"]` (`~/.config/leancoder`), `chmod 600` - never in
`config.toml` (it autosaves and would round-trip your key back to disk). Resolve keys
from an env var first, then a keyfile (see the bundled providers for the pattern).

## The client (`make_client(cfg)` returns)

Same duck type as `OllamaClient` - the contract the agent loop already speaks:

- `chat(messages, tools, should_abort) -> (assistant_dict, prompt_eval|None, aborted_bool)` **(required)**.
  `should_abort()` returns `True` when the user hits Ctrl-C; poll it during streaming
  and terminate early (set `aborted=True`) if it fires.
- `uncapped_ctx = True` (optional attr) opts out of the 32k local-VRAM OOM cap - a
  hosted client has no local memory limit, so it should trust its own window.
- `list_models()` / `running_models()` - optional; core falls back to the spec's
  `list_models` and `[]`.
- **Per-call observability attributes** the client MAY set after each `chat()` (core
  reads them off the client; all optional, treated as 0 if absent):
  - `last_out_tokens` - output tokens for the call; feeds `agent.session_out`.
  - `last_cache_read` / `last_cache_write` - prompt-cache read/write tokens. **A
    caching backend MUST set these.** Core's context meter (and the auto-handover
    zones, and the status row) compute the true fill as
    `prompt_eval + last_cache_read + last_cache_write` - with caching, `prompt_eval`
    is just the *uncached* input and alone under-reports real context size by ~10x,
    which misleads the meter and stops compaction firing. A non-caching backend leaves
    them 0 (the meter falls back to `prompt_eval`).

The client holds the `cfg` it was built with and reads, at send time:

- `cfg.active_model()` - the model to send (the provider's, not Ollama's).
- `cfg.setting("thinking")`, `cfg.setting("effort")`, `cfg.setting("cache")`, ... -
  any knob the provider declared, plus any free-form one the user set via `/provider
  set`. The client
  interprets these; core only stores them.

It does **not** mutate `cfg`.

## Settings & capabilities

`capabilities(model)` returns `{setting_key: [choices]}`. `thinking` and `effort` are
the well-known keys (they get the friendly `/think` / `/effort` commands); any other
key you declare is a first-class knob too - core lists it under `/provider set`,
validates it against its choices, persists it, and menus it. A knob with no declared
choices is still settable free-form via `/provider set <key> <value>` and read with
`cfg.setting(key)`,
so a provider can wire in something core has never heard of - **no core change**.

```python
"capabilities": lambda model: {
    "thinking": ["off", "adaptive", "max"],
    "effort":   ["low", "med", "high"],
    "cache":    ["off", "on"],          # custom: core persists/sets, client reads
},
```

This is the extensibility seam: a provider adds new behaviour (prompt caching, a
tool-calling dialect, a beta header) by declaring and reading a setting key.

## Config & lifecycle

Provider state is core-owned and autosaved to `config.toml`:

- `provider` - the active backend name (default `"ollama"`).
- `[providers.<name>]` table - per-provider `model` (and optionally `thinking` /
  `effort`). The bundled Ollama provider's model lives in `[providers.ollama].model`;
  top-level `model` is kept as a back-compat alias. Ollama's transport config
  (`host` / `hosts` / `[machines]` / `num_ctx`) stays top-level.
- `num_ctx` is set from `context_window(model)` on activate / `/model`, and is
  **never** written to `config.toml` by a provider.

At launch, core activates a provider if `cfg.provider` names a registered +
`available()` one, or some provider's `autostart()` returns true. Activation builds
the client via `make_client`, swaps it in (re-running ctx detection), applies
`thinking` / `effort` against `capabilities`, and calls `on_activate`. Switching away
(or `/provider off`) calls `on_deactivate` and falls back to Ollama.

## Commands core provides

So each backend stops re-implementing them:

- `/provider list | enable <name> | disable <name>` - manage provider plugins in the
  dir (no arg = toggle menu). Enabling registers it; disabling deactivates + drops it.
- `/provider [name]` - list registered providers (available ones marked) and switch
  the active one. No arg = menu.
- `/provider login [name]` / `/provider clear [name]` - run the provider's `login` /
  `clear` hooks. `login` does the auth flow and returns truthy on success (core then
  activates it); `clear` wipes credentials (core drops back to Ollama if it was
  active). A provider that omits a hook simply isn't offered it; the name is optional
  when exactly one provider implements the hook.
- `/model` - shows the active provider's `list_models()`.
- `/think` / `/effort` - capability-aware pickers over the declared choices when the
  active provider declares them; a clear "not supported" otherwise. (Ollama keeps its
  boolean `/think`.)
- `/provider set [key [value]]` - list the active provider's declared settings +
  values (no args), menu a declared key (key only), or set any key including a
  free-form one. (App-config knobs live in the separate `/set`.)
- `/usage` - prints the active provider's `detail(agent, cfg) -> str` (its full
  account/billing view), or a core default (model, session tokens, context %, warm
  models) for a provider without `detail`.

Auth is the provider's business, but its *command surface* is core: `/provider
login|clear` replace per-backend login commands. Only genuinely bespoke actions stay
as the provider's own `register_command`.

## The usage / status line

The per-response line under each turn is rendered entirely by core - the provider
feeds **numbers**, so styling stays consistent across every backend and there's one
place to change it. After the context meter, core calls `usage(agent, cfg)` on the
active provider if defined:

```python
def usage(agent, cfg):
    return {
        "meters": [                                   # rendered as "label pct% (resets)"
            {"label": "5h", "pct": 12, "resets_at": "2026-06-27T20:00:00Z"},
            {"label": "wk", "pct": 40, "resets_at": "2026-07-01T00:00:00Z"},
        ],
        "note": "",                                   # optional trailing text
    }
```

Core owns the colour ramp (green/orange/red by pct), the `resets_at` -> `"20:00"` /
`"01Jul"` formatting, and `FULL` handling at >=100%. Returning `None` / `{}` keeps the
plain context meter. The same output also renders as the tail of the condensed status
row above the prompt - fill in `usage` and you get quota meters in both places for
free. `status_line(agent, cfg) -> str` is a raw-string escape hatch for a backend
whose display genuinely doesn't fit the meter model; if both are defined, `usage`
wins.

## Session token counter

Core accumulates `agent.session_in` / `agent.session_out` across the session (zeroed
by `/clear`), so `/usage` shows a token count for *any* backend without the provider
tracking it. Input comes from `chat()`'s returned `prompt_eval`; output from the
optional `client.last_out_tokens` the client sets per call.
