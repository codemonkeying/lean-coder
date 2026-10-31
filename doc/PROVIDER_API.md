# Provider API

A provider is a `.py` file in `providers/` (repo) or `~/.config/leancoder/providers/`
with a module-level `PROVIDER` dict, toggled with `/provider enable|disable`. The
`ProviderManager` registers it; the file does not call `register_provider`. Bundled
providers: [providers/README.md](../providers/README.md). Template:
[examples/providers/example.py](../examples/providers/example.py).

## The `PROVIDER` dict

```python
PROVIDER = {
    "name":           "cloud",
    "description":    "my hosted backend",         # /provider menu
    "make_client":    lambda cfg: ApiClient(cfg),  # required
    "list_models":    lambda: [...ids...],         # required; safe unauthed, never throws
    "context_window": lambda model: 200000,        # required
    # optional, absent-safe:
    "capabilities":   lambda model: {"thinking": ["off", "max"], "effort": ["low", "high"]},
    "available":      lambda: True,                # offered only when True (default True)
    "autostart":      lambda: False,               # activate on launch when True
    "on_activate":    lambda agent, cfg: ...,
    "on_deactivate":  lambda agent, cfg: ...,
    "usage":          lambda agent, cfg: {...},    # quota meters (below)
    "status_line":    lambda agent, cfg: "raw",    # used only if usage is absent
    "login":          lambda agent, cfg: True,     # enables /provider login; truthy on success
    "clear":          lambda agent, cfg: None,     # enables /provider clear
    "detail":         lambda agent, cfg: "text",   # /usage full view
    "tag":            "cloud",                     # /model row label (default: name)
    "warm_models":    lambda: ["m1"],              # loaded models (green dot)
    "model_status":   lambda model: None,          # hint in /model, e.g. "pull"
    "endpoints":      lambda: ["http://host:11434"],
    "restore_settings": lambda agent, cfg: ...,    # re-sync module state after /load
    "worker_inherit": ("account",),                # settings keys workers inherit
    "location":       lambda cfg: "box (http://...)",  # status label
    "prepare":        lambda agent, cfg: None,     # runs before activation
    "on_conn_error":  lambda agent, cfg, err: False,   # True = fixed, retry the turn
    "system_addendum": lambda agent, cfg: "",      # extra system-prompt text, deterministic per model
    "login_files":    "auth*.json",                # listed by /push and /pull
}

def setup(lc, cfg):               # optional, helpers only
    _lc.update(lc)
    lc["register_command"]("/cloud_push", _push)
```

`restore_settings` must not write `config.toml`; settings are session-scoped.

## What `lc` contains

A read-only mapping; unknown names raise `KeyError`. Bundled providers use only these
(a test enforces it).

| Group | Names |
|---|---|
| Display | `green` `red` `yellow` `dim` `bold` `blue` `glyph` `GLYPH` `Spinner` `THINK_FRAMES` `MarkdownStream` |
| Streaming | `consume_openai_stream` `upstream_error_of` `stream_tiered` `open_tiered` `StreamStall` `parse_text_tool_calls` `messages_tokens` `encode_image_block` |
| Config + limits | `CONFIG_DIR` `save_config` `autosave_config` `sampling_for` `hosted_sampling` `RateLedger` `log_system_activity` |
| Commands + prompts | `register_command` `register_provider` `pick_one` `pick_model_menu` `model_available` `handle_model_command` `handle_sync_command` `ask` `menu_index` `visible_wait` `TurnAbort` `TTY` |
| Hosts | `resolve_host` `host_label` `valid_host` `norm_host` |
| Live state | `active_composer()` |

## Auth

- Implement `login` / `clear`. Read secrets with `getpass`, store `chmod 600` under
  `lc["CONFIG_DIR"]`, never in `config.toml`. Resolve an env var first, then the keyfile.
- On a missing or rejected key, raise from `chat()` a `RuntimeError` whose text contains
  `no API key`, `API key`, `401`, `403` or `unauthorized`. Core then offers login and
  retries the turn once.

```python
if not key:
    raise RuntimeError("no API key - use /provider login")
```

## The client

Returned by `make_client(cfg)`:

- `chat(messages, tools, should_abort) -> (assistant_dict, prompt_eval|None, aborted)`
  (required). Poll `should_abort()` while streaming.
- `uncapped_ctx = True` - opt out of the 32k local OOM cap (hosted clients).
- `list_models()` / `running_models()` - optional.
- After each call, optionally set `last_out_tokens`, `last_cache_read`,
  `last_cache_write`. A caching backend must set the cache counts: context fill is
  `prompt_eval + last_cache_read + last_cache_write`.

Read `cfg.active_model()` and `cfg.setting(key)` at send time. Do not mutate `cfg`.

### Shared stream reader

For OpenAI-style SSE (`data: {...}` ... `data: [DONE]`):

```python
return lc["consume_openai_stream"](client, resp, should_abort, tools,
                                   on_line=None, on_obj=None, on_done=None,
                                   text_fallback=False, split_think=False)
```

Sets `client.last_finish`, `last_truncated`, `last_out_tokens`; reasoning is shown, not stored.

- `text_fallback` - parse text-encoded tool calls (local servers only).
- `split_think` - move inline `<think>` tags to the reasoning line (local servers only).
- `on_line(line, obj, st)` - each line; return `True` to stop, raise to fail. Use with
  `lc["upstream_error_of"](obj)`.
- `on_obj(obj, st)` - each parsed chunk, after core.
- `on_done(st)` - adjust `st["prompt_total"]`, `st["out_tokens"]`, `st["finish"]`.

`st` keys: `finish`, `content_parts`, `prompt_total`, `out_tokens`, `aborted`,
`printed`, `printed_think`, `chunks`, `tool_acc`.

## Settings

`capabilities(model)` returns `{key: [choices]}`. `thinking` and `effort` map to
`/think` and `/effort`; other keys appear under `/provider set`. Undeclared keys can be
set free-form with `/provider set <key> <value>`. Read with `cfg.setting(key)`.

## Config

- `provider` - active backend (default `"ollama"`).
- `[providers.<name>]` - per-provider `model`, `thinking`, `effort`.
- Ollama transport keys (`host`, `hosts`, `[machines]`, `num_ctx`) stay top-level.
- `num_ctx` comes from `context_window(model)`; providers never write it.

## Usage line

```python
def usage(agent, cfg):
    return {"meters": [{"label": "5h", "pct": 12, "resets_at": "2026-06-27T20:00:00Z"}],
            "note": ""}
```

Core renders colours and reset times. `None` / `{}` keeps the plain context meter.
Session token totals are tracked by core.

## Sampling (local providers)

Call `lc["sampling_for"](cfg, model, server_card, "<source>")` and send only the
returned `send` dict (`server_card` = the server's defaults, or `None`). Store
`{"params": effective, "source": source, "sent": send}` on `client.last_sampling`.
Hosted providers send no sampling parameters.
