# Slash-command contract

## 1. Registration

- Dispatch: `"/name": handle_name_command` in `_BUILTIN_COMMANDS_TABLE` (the only
  authority; `_BUILTIN_COMMANDS` is derived). Aliases are extra keys to the same handler.
- Completion: add the name to `SLASH_COMMANDS`.
- Help: `("/name [args]", "one-liner")` in `HELP_COMMANDS`; the handler docstring is `/name ?`.

The smoketest checks parity: everything listed is dispatchable, and every canonical
name is in `HELP_COMMANDS`.

## 2. Argument completion

`_arg_completions(agent, cfg, cmd)`: `cmd == "/name"` returns subcommands;
`cmd == "/name sub"` returns that subcommand's argument values. `?` is appended
automatically. Read live state.

## 3. Handlers

`handler(agent, cfg, arg)`, `arg` is the raw remainder; parse with `arg.split(None, N)`.
Subcommands are lower-case verbs; no arg usually opens the menu.

## 4. Menus

Built on `run_picker(render, on_key)`, falling back to a plain listing when
`not picker_capable()`. Keys come from `_K_*` constants.

- Multi-select (`lean_tools_menu`, `mcp_servers_menu`): space toggles, `a` all, enter saves, q/Esc cancels.
- Single choice: highlight + enter.
- Confirm (`ask_action`).

## 5. Persistence

Write to `cfg` and call `save_config(cfg)`. `/set` and `/provider` autosave via
`_CMD_AUTOSAVE`. New persisted state needs a `Config` field, a `load_config` reader
and a `save_config` writer.

## 6. Capability leash

Gated in `active_tools` and re-checked in `_leash_allows_tool`:

| leash | tools offered |
|-------|---------------|
| `chat` | none |
| `r`    | read-only core + lean-tools of tier `read` |
| `rw`   | + write tools, tier `write` |
| `rwe`  | + exec, `ask_user_to_run` (if enabled), MCP tools, tier `exec` |

`/approve` sets when to confirm: `ask`, `session`, `auto`. `/askread` also confirms
reads. A chat-only model (`tool_support=False`) gets no tools.
