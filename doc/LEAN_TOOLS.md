# Lean-tools

A lean-tool is one `.py` file with a `TOOL` schema and a `run` function, and/or a
`setup(lc, cfg)` hook. For MCP servers see [MCP.md](MCP.md); for model backends see
[PROVIDER_API.md](PROVIDER_API.md).

## Contract

```python
TOOL = {
    "name": "line_count",
    "description": "Count lines in a file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "safe": True,
}

def run(args, cwd):
    from pathlib import Path
    return f"{len((Path(cwd) / args['path']).read_text().splitlines())} lines"
```

### `TOOL`

| key | required | notes |
|-----|----------|-------|
| `name` | yes | Unique; must not collide with a built-in. |
| `description` | yes | Sent every request; keep it short. |
| `parameters` | yes | JSON Schema; list only what `run` reads. |
| `safe` | no | `True` = read-only: no confirmation, may run concurrently. Must be side-effect-free and thread-safe. |
| `tier` | no | `"read"`, `"write"` or `"exec"` (leash level). Default: `read` if `safe`, else `exec`. |
| `glyph` | no | Icon for the call line. |
| `driver_only` | no | `True` = always runs locally, never pushed to a remote. |

Leash tiers: [COMMANDS.md](COMMANDS.md#6-capability-leash).

### `run(args, cwd)`

- Resolve paths against `cwd` (the remote dir under `/connect`).
- Return a string. Errors: return `"error: ..."` or raise.
- Never print to stdout (it is the remote protocol channel); use stderr.

### Output sizing

Core sets `LEANCODER_CTX_MAX` (context window) and `LEANCODER_CTX_USED` before each call.
Cap large output to a fraction of the free space and offer an offset argument. Both
are unset in tests and on remotes. See `web_fetch.py` (`free_tokens()`, `char_budget()`).

### Shaping parameters

- One tool with an `action` enum, defaulted from which args are present.
- Minimal `required`; optional knobs default in `run()`.
- Put fixed defaults in a `"default"` key and in the description. Prefer `enum`.

### Built-in tools

`read_file`, `list_files`, `search_files`, `apply_diff`, `write_file`, `run_command`
live in [lean-tools/builtins.py](../lean-tools/builtins.py); always loaded.

## Bundled lean-tools

In [lean-tools/](../lean-tools/), listed by `/tools`. On by default except
`diagnostics`, `notify`, `symbols`, `word_count`.

`git_summary`, `symbols`, `diagnostics`, `word_count`, `web_fetch`, `brave_search`,
`view`, `ssh`, `shell_session`, `dispatch_worker` and `board` ([WORKERS.md](WORKERS.md)),
`notify`, `provision`, `update`.

Templates in [examples/lean-tools/](../examples/lean-tools/): `reverse_file.py`,
`weather.py`, `session_info.py`.

## `setup(lc, cfg)` and slash commands

```python
def setup(lc, cfg):
    lc["register_command"]("/whoami", _handler, "show session info")
```

- `register_command(name, handler, help_line, completer=None)`; `handler(agent, cfg, arg)`;
  `completer(agent, cfg) -> [str]`. Built-in names are refused; first registration wins.
- A setup-only file adds no model tool and is listed by filename.
- Runs once at startup, driver only; `/reload` does not re-run it. Exceptions are
  caught (`LEANCODER_DEBUG=1` for tracebacks).
- State set in `setup` does not reach `run` (separate loads).

### Pickers

- `lc["pick_one"](header, choices, current=None, labels=None)` - value or `None`.
- `lc["pick_grouped"](header, lines, index_of, num_of)`.
- `lc["run_picker"](render, on_key)`, `read_key(stdin)`, `picker_capable()`.
- `lc["glyph"](unicode, ascii_fallback)` for symbols with an ASCII fallback.

Without a TTY, or with a leading `--plain` arg, menus fall back to a numbered prompt.

## Enabling

- User tools go in `~/.config/leancoder/lean-tools/` (or `lean_tools_dir`).
- `/tools` toggles; the set is saved as `lean_tools_enabled`. New user tools start off.
- Subdirectories are groups; names must be unique across groups.
- On `/connect`, enabled tool files sync to the remote by hash; `/reload` resyncs.
  Third-party imports must exist where the tool runs.

## Testing

```python
import importlib.util
spec = importlib.util.spec_from_file_location("t", "my_tool.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
assert mod.run({}, ".").startswith("error")
```

Secrets and caches go in `~/.config/leancoder/` (`chmod 600` for secrets), not `config.toml`.
