# Slash-command / completion / menu contract

Every built-in `/command` follows this so completion, help, and menus behave
uniformly — learn the rules once, drive every command. A new command that obeys
this doc wires into all the shared machinery with no bolt-on special-casing.

## 1. Registration (single source of truth)

- **Dispatch:** add `"/name": handle_name_command` to `_BUILTIN_COMMANDS_TABLE`.
  This table is the *only* authority — `_BUILTIN_COMMANDS` (shadow-protection set
  that stops a lean-tool claiming a built-in name) is derived from it, so it can
  never drift.
- **Aliases:** extra keys in the same table pointing at the same handler
  (`"/models": handle_model_command`). `/cmd ?` lists them automatically.
- **Completion list:** add the *canonical* name to `SLASH_COMMANDS` (not aliases —
  they're reachable but not offered for Tab). This is what `/na<Tab>` completes.
- **Help:** add `("/name [args]", "one-liner")` to `HELP_COMMANDS`.
- **Per-command help:** write a docstring on the handler — `/cmd ?` prints it. No
  separate help registry to keep in sync.

A parity check (see the smoketest) asserts: nothing in `SLASH_COMMANDS`/`HELP` is
undispatchable, and every dispatchable canonical name appears in `HELP`.

## 2. Argument completion

First-arg Tab completion lives in `_arg_completions(agent, cfg, cmd)`, keyed on the
command path:

- `cmd == "/name"` → return the subcommand/value list (e.g. `["list","add",…]`).
- `cmd == "/name sub"` → return completions for that subcommand's own argument
  (e.g. a server/host/session name). Falls back to the bare command if empty.
- `?` is appended automatically as a universal first arg (per-command help).

Reads live state (models, hosts, enabled servers) — never a frozen snapshot.

## 3. Value/argument shape

Handlers take `(agent, cfg, arg)` where `arg` is the raw remainder string. Parse
with `arg.split(None, N)`. Subcommands are lower-cased verbs from a closed set:
`list | add | remove | reconnect | …`. No-arg usually opens the menu (§4).

## 4. Menu types

Three interactive shapes, all built on `run_picker(render, on_key)` and all
degrading to a plain listing when `not picker_capable()` (no TTY):

- **Enable/disable multi-select** (`lean_tools_menu`, `mcp_servers_menu`):
  up/down move, **space** toggles one, **`a`** toggles ALL (all on, or all off if
  already all on), **enter** saves, **q**/Esc cancels. Returns the new enabled set
  or `None` on cancel. Group headers (lean-tools) toggle a whole group.
- **Single-choice pick** (think/effort/leash/approve levels): highlight + enter.
- **Confirm** (`ask_action`): y/n/always-style prompt.

Keys come from the shared `_K_*` constants. Every menu header advertises its keys.

## 5. Persistence (mirror, don't duplicate)

A command that changes saved state writes it back to `cfg` and calls
`save_config(cfg)` — it does NOT invent its own storage:

- Enabled lean-tools: `cfg.lean_tools_enabled = sorted(new)` → `save_config`.
- Enabled MCP servers: `_mcp_persist` sets `cfg.mcp_servers` + `cfg.mcp_enabled`
  → `save_config` — the exact same shape as lean-tools.
- Pure config knobs (`/set`, `/provider`) autosave via `_CMD_AUTOSAVE` in the
  dispatch (handlers with many early-returns don't save themselves).

`load_config` reads each field back with the same key. New persisted state adds a
field to `Config`, a reader in `load_config`, and a writer in `save_config` — three
matching points, never a side file.

## 6. Capability leash (tools & MCP ride the same pipe)

The model's tool surface is gated once, in `active_tools`, by the `/leash` ceiling:

| leash | tools offered |
|-------|---------------|
| `chat` | none (conversation only) |
| `r`    | read-only core + safe lean-tools |
| `rw`   | + write tools |
| `rwe`  | + exec, ask_user_to_run, **MCP tools, non-safe lean-tools** |

MCP tool schemas fold into the same `lean_tool_schemas` stream as lean-tools
(`_all_lean_schemas` = lean-tool schemas + `mcp.schemas()`), marked `is_safe=False`,
so they surface only at `rwe` and vanish at `chat`/`r`/`rw` — no MCP-specific gate.
`_leash_allows_tool` re-checks at dispatch (defense in depth): a call that reaches
the executor above the ceiling is refused there too. So **`/leash chat` disables MCP
along with every other tool**, and a chat-only model (provider `tool_support=False`)
has no tools regardless of leash.
