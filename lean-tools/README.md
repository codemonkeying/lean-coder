# lean-tools/

Tool plugins. Each `*.py` here defines a `TOOL` schema and a `run` function; an
enabled lean-tool appears to the model as a normal tool, locally and on a remote.

## Files

- `builtins.py` - the always-on file and command tools (`read_file`, `list_files`,
  `search_files`, `apply_diff`, `write_file`, `run_command`). Not toggleable. Each
  entry declares a leash `tier` (read / write / exec).
- `board.py`, `brave_search.py`, `diagnostics.py`, `dispatch_worker.py`,
  `git_summary.py`, `notify.py`, `provision.py`, `shell_session.py`, `ssh.py`,
  `symbols.py`, `update.py`, `view.py`, `web_fetch.py`, `word_count.py` - bundled
  extras, toggled with `/tools`. All are on by default except `diagnostics`,
  `notify`, `symbols` and `word_count`.

Short descriptions are in [LEAN_TOOLS.md](../doc/LEAN_TOOLS.md#bundled-lean-tools);
workers and boards in [WORKERS.md](../doc/WORKERS.md).

## Writing one

The contract (`TOOL` schema, `run` signature, `safe` and leash tiers, `setup(lc, cfg)`)
is in [LEAN_TOOLS.md](../doc/LEAN_TOOLS.md). Examples are in
[`../examples/lean-tools/`](../examples/lean-tools/).

Put your file in `~/.config/leancoder/lean-tools/` (or here), then enable it with
`/tools`.
