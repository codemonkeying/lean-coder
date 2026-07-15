# lean-tools/

Tool plugins. Each `*.py` here defines a `TOOL` schema and a `run` function; an
enabled lean-tool appears to the model as a normal tool, local and remote alike. No
protocol, no server, no core dependency - the lean alternative to MCP.

## What's here

- **`builtins.py`** - the bundled, always-on file-editing + command surface
  (`read_file`, `list_files`, `search_files`, `apply_diff`, `write_file`,
  `run_command`). Ships with the code, not toggleable. Each entry declares a leash
  `tier` (read / write / exec).
- **`dispatch_worker.py`, `shell_session.py`, `update.py`** - other bundled tools.
- The rest (`web_fetch`, `web_screenshot`, `brave_search`, `git_summary`,
  `diagnostics`, `note`, `word_count`, `ssh`, `notify`, `provision`) are bundled
  extras, toggled with `/tools`.

## Writing one

The full contract (`TOOL` schema, `run` signature + return shape, the `safe` flag and
confirm/leash interaction, the optional helpers-only `setup(lc, cfg)`, and the
hermetic `--tool-exec` role) is in **[`../LEAN_TOOLS.md`](../LEAN_TOOLS.md)**. Copyable
examples live in **[`../examples/lean-tools/`](../examples/lean-tools/)**.

Drop your file here (or in `~/.config/leancoder/lean-tools/`), then enable it with
`/tools`. Config autosaves - no manual save step.
