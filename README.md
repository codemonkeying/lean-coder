<div align="center">

# lean-coder

**A small terminal coding agent. Stdlib-only Python core, minimal context overhead.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
![Core dependencies: none](https://img.shields.io/badge/core%20dependencies-none%20(stdlib%20only)-brightgreen.svg)

[Install](#install) &middot; [Quick start](#quick-start) &middot; [Providers](#providers) &middot; [Safety](#safety) &middot; [Tools](#tools) &middot; [Context](#context-management)

![lean-coder auditing and fixing a real bug on a local model](demos/demo.gif)

<sub>A local model (Qwen3-Coder:30b via Ollama) auditing a Python package, finding the root cause of a bug, and fixing it.</sub>

</div>

## What it is

lean-coder reads, edits, and runs code in your project through a model's native
tool-calling API. It keeps the system prompt and tool descriptions short, so more of the
context window is left for your code.

Context overhead (system prompt + tool schemas), checked by the test suite:

<!-- overhead-gate: tests/_smoketest.py reads these two numbers; they are the pass/fail limits -->
| Surface | Max tokens |
|---|---|
| Always-on (core tools) | 2500 |
| Every bundled lean-tool enabled | 6500 |

Optional lean-tools are off by default and add nothing until enabled.

- Runs against local models (Ollama, llama.cpp, MLX) or hosted APIs.
- When the context window fills, the agent updates its docs, pins a goal and plan, and
  continues from a summary ([compaction](#compaction)).
- Can dispatch background worker agents for sub-tasks (`dispatch_worker` lean-tool).
- Runs on Termux, or drives a remote box over SSH with `/connect`.
- Generic MCP client; no servers are configured by default.

Layout: `lean_coder.py` (core), `lean-tools/builtins.py` (required file/shell tools),
`lean-tools/` (optional tools), `providers/` (model backends). Every change must pass three
test gates; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

Requires `python3` 3.11+ and `curl` or `git`. Minimal images may need these first:

| Platform | Prerequisite install |
|---|---|
| Debian / Ubuntu / **WSL** | `sudo apt update && sudo apt install -y python3 curl` |
| Fedora / RHEL | `sudo dnf install -y python3 curl` |
| Arch | `sudo pacman -S --noconfirm python curl` |
| Alpine | `sudo apk add python3 curl` |
| Android / Termux | `pkg install -y python curl` |
| macOS | `python3` + `curl` ship with the OS (or `brew install python`) |

Linux / WSL / macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/codemonkeying/lean-coder/main/install.sh | bash
```

Android / Termux:

```bash
pkg install -y python curl && curl -fsSL https://raw.githubusercontent.com/codemonkeying/lean-coder/main/install.sh | bash
```

`install.sh` installs the code and puts `lean_coder` on your `PATH`. Add
`--with-ollama --pull` for a local Ollama on Linux. On Termux, point it at a remote Ollama
(`lean_coder --host http://HOST:11434`). The installer is idempotent; `./uninstall.sh`
removes everything.

Or run from a clone without installing:

```bash
git clone https://github.com/codemonkeying/lean-coder
cd lean-coder
./install.sh --dry-run                          # show what it would do, change nothing
python3 lean_coder.py                            # local Ollama, default model
python3 lean_coder.py --host http://box:11434 --model qwen3-coder:30b
```

**Runtime:** Python 3.11+, no third-party packages for the core. Some opt-in lean-tools
have their own deps (e.g. `web_screenshot` needs Playwright) and say so when enabled.
`/connect` needs an `ssh` client. The Anthropic subscription provider needs `node`.

**Updating:** re-run the installer, or `git pull && ./install.sh`. With the `update`
lean-tool enabled, `/update` pulls a newer `lean_coder.py` (`update_track` = `stable` or
`beta`; `auto_update = true` checks at launch). From a shell (or over ssh),
`lean_coder --update [check|force]` does the same and exits; it doesn't need the tool enabled.

## Quick start

```bash
lean_coder                       # uses your config, or localhost Ollama + default model
```

Type a request. The agent shows a diff before each edit and confirms before running
commands (unless approval is `session` or `auto`). `/help` lists commands.

```
$ lean_coder
lean-coder  <your-model> @ <your-provider>
› add a --json flag to the export command and update the tests
● I'll look at the export command first.
  ⚙ read_file(path=src/export.py)
  ⚙ search_files(pattern=def export)
  ...
```

## Providers

A provider connects lean-coder to a model backend. Ollama is enabled by default and uses
`localhost:11434`. Hosted providers ship disabled until you log in:

| Provider        | Backend | Get a key |
|-----------------|---------|-----------|
| `ollama`        | Local / self-hosted Ollama (default) | none needed |
| `anthropic_api` | Anthropic API (Claude) | [console.anthropic.com](https://console.anthropic.com) |
| `gemini`        | Google Gemini | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `groq`          | Groq | [console.groq.com](https://console.groq.com/keys) |
| `openai`        | OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) |
| `openrouter`    | OpenRouter | [openrouter.ai](https://openrouter.ai/keys) |

```
/provider login anthropic_api      # prompts for the key, saves it, switches to it
```

Keys are stored in a `chmod 600` file under `~/.config/leancoder/`, never in
`config.toml`. An env var (e.g. `ANTHROPIC_API_KEY`) is used instead if set. `/provider`
switches backends; `/model` lists and switches models across enabled providers.

For another backend, copy [`examples/providers/example.py`](examples/providers/example.py)
into `providers/`; see [PROVIDER_API.md](PROVIDER_API.md). Sessions work across providers.

## Safety

Two independent settings:

- **`/leash`** - which tools the model gets (default `rwe`):
  `chat` (none) · `r` (read files) · `rw` (+ edit files) · `rwe` (+ run commands).
  The model is told its level.
- **`/approve`** - when to confirm (default `session`):
  `ask` (every edit/command) · `session` (once per run) · `auto` (never).

Examples: `leash r` for unattended analysis; `leash rw` + `approve auto` for unattended
editing. OS file permissions still apply. `/incognito` writes nothing to disk;
`/askread` also confirms read tools.

## Tools

### Always-on

Eight file/shell tools in `lean-tools/builtins.py`:

| Tool            | What it does |
|-----------------|--------------|
| `read_file`     | Line-numbered file contents; optional line range; large files truncated. |
| `list_files`    | Directory listing, honoring ignore rules. |
| `search_files`  | Regex search -> `file:line` matches (capped). |
| `apply_diff`    | Preferred edit tool: SEARCH/REPLACE blocks. |
| `replace_lines` | Replace a line range by number. |
| `write_file`    | Create or overwrite a whole file. |
| `run_command`   | Run a foreground shell command; output truncated. |
| `background`    | Run and manage long-lived tasks (`run` / `status` / `kill`). |

Also always present: `update_plan` (a pinned goal + TODO that survives compaction),
`note` (a session notebook), and, by default, `ask_user_to_run`, which hands a command to
you when it needs sudo, an interactive prompt, or a password. Only the command and exit
code return to the model. Read-only calls in one turn run concurrently.

### Opt-in lean-tools

A lean-tool is one `.py` file with a `TOOL` schema and a `run` function. Bundled ones are
in [`lean-tools/`](lean-tools/), off by default; enable them with `/tools`.

| Lean-tool         | Adds |
|-------------------|------|
| `dispatch_worker` | Run a sub-task in a background worker agent and collect its result. Workers can be steered, restricted to fewer tools, and assigned to a `board`. With `worker_checkpoint` on, a stopped worker can be resumed, and `pause` saves one as a session to continue later. Adds `/worker`. |
| `board`           | A task board: a dependency graph of tasks the driver assigns to workers. Stored on disk. |
| `web_fetch`       | Read a URL as text. |
| `web_screenshot`  | Screenshot a URL with a headless browser and return the page text. Needs [Playwright](https://playwright.dev/python/). |
| `brave_search`    | Web search (Brave API key). |
| `git_summary`     | Read-only git status, diffstat, recent commits. |
| `diagnostics`     | Lint/typecheck with whatever is installed (pyright, ruff, tsc, eslint, shellcheck, …), falling back to `py_compile` / `bash -n`. |
| `symbols`         | Outline Python classes/functions or find a definition (stdlib `ast`). |
| `shell_session`   | A persistent interactive shell (REPL, ssh, …). |
| `ssh`             | One-shot `ssh host cmd`. |
| `notify`          | Desktop notification when a long task finishes. |
| `provision`       | `/provision`: install lean-coder on another box over SSH. |
| `update`          | `/update`: self-update `lean_coder.py`. |
| `word_count`      | Count lines / words / chars in a file. |

`brave_search` needs a [Brave Search API key](https://search.brave.com/app/keys) in
`~/.config/leancoder/brave.key` or `LEANCODER_BRAVE_KEY`. See [LEAN_TOOLS.md](LEAN_TOOLS.md);
[`examples/lean-tools/`](examples/lean-tools/) has templates. There is no LSP integration.

### MCP servers

lean-coder is an MCP client. Added servers' tools appear as `mcp__<server>__<tool>`:

```
/mcp add fs npx -y @modelcontextprotocol/server-filesystem /some/dir   # stdio server
/mcp add gw https://mcp-gateway.example.com/mcp/handbook/mcp           # HTTP server
/mcp                       # enable/disable menu
/mcp list                  # servers + connection state
/mcp reconnect [name]
/mcp remove <name>
```

Transports: stdio and HTTP (streamable, SSE or JSON). HTTP auth is a Bearer token, or
OAuth 2.1 client credentials (fetched and refreshed automatically). Configure in
`config.toml`:

```toml
[mcp_servers.gw]
transport = "http"
url = "https://mcp-gateway.example.com/mcp/handbook/mcp"
auth = { type = "bearer", token_env = "GW_KEY" }
# or: auth = { type = "oauth", token_url = "…/oauth/token", client_id = "…", client_secret_env = "GW_SECRET", scope = "mcp:access" }
```

MCP tools run locally (never on a `/connect` remote), need leash `rwe`, and confirm like
other non-read tools. See [MCP.md](MCP.md).

### `apply_diff` format

```
<<<<<<< SEARCH
exact existing text (must match the file verbatim)
=======
replacement text
>>>>>>> REPLACE
```

Blocks apply in order. If any SEARCH text isn't found, nothing is written and the model is
told to re-read the file.

## Context management

- **Truncation and ignore rules.** Large reads and command output are cut head/tail with a
  `…[truncated N …]…` notice. `.gitignore`, `.leancoderignore`, and built-in defaults
  (`.git/`, `node_modules/`, …) are honored.
- **Result cap.** Every tool result is capped on the way in (head + tail kept), sized to
  the free window.
- **`/trim [keep]`** replaces old tool results with one-line stubs, keeping the newest
  `keep`. No LLM call.
- **Meter.** Context use prints after each turn. `/usage` shows it; `/activity` lists
  automatic actions (compaction, trim, caps).

### Compaction

On `/compact`, or automatically, the model gets one turn to:

1. Update the project's docs (and commit them in a git repo).
2. Write a summary: goal, decisions, state, next step.
3. Pin a goal + TODO.
4. Write the next instruction for itself.

Older history is replaced by the summary (the last `compact_keep` turns, default 3, stay
verbatim). The next instruction is then sent as a new turn after a 5-second `^C` window.

Thresholds: a nudge near ~70% full, forced compaction at `compact_at` (~90%), emergency
compaction at ~100%. A loop guard limits it to about once a minute. `auto_compact`,
`autostart_after_compact`, `compact_emergency`, and the prompts are configurable.

- **Wake on background finish** (on by default): a finished background task or worker
  starts a turn when the prompt is idle. `wake_on_bg_finish = false` disables it.
- **Send window** (off by default): `window_messages = N` sends only the last N messages,
  cut at a turn boundary. For very small models.

## Configuration

Precedence: **CLI flag > env var > config file > default**.

| Setting         | Flag                   | Env               | Default                   |
|-----------------|------------------------|-------------------|---------------------------|
| Ollama endpoint | `--host`               | `OLLAMA_HOST`     | `http://localhost:11434`  |
| Model           | `--model`              | `LEANCODER_MODEL` | `qwen3-coder:30b`         |
| Context window  | `--num-ctx`            | -                 | auto-detect, capped 32768 |
| Project dir     | `--cwd`                | -                 | current directory         |
| Approval mode   | `--approval` / `--auto`| -                 | `session` (ask once)      |
| Capability      | `--leash`              | -                 | `rwe`                     |
| Resume session  | `--resume <name>`      | -                 | auto-load last for cwd    |

Config is `~/.config/leancoder/config.toml` and saves automatically on any change. It also
holds host failover, named machines, per-machine default models, and saved `/connect`
targets. Auto-detect only lowers the context window (capped 32768); use `--num-ctx` to go
higher.

**What stays on disk.** Only things you named: config, sessions (`~/.config/leancoder/sessions`),
and task boards. Everything else a running lean-coder creates (background task logs, worker
files, file-claim boards) lives in one folder per process under `$XDG_RUNTIME_DIR` (else
`$TMPDIR` or `/tmp`). It is removed when that process exits; after a crash the next start on
the same machine removes any folder whose process is gone. A remote executor does the same on
its machine and deletes its copied code when the session closes or the link has been down
longer than `LEANCODER_REMOTE_IDLE_TTL` (default 30 min).

## Slash commands

```
/clear             wipe conversation, stay in this session
/new [name]        start a separate session
/trim [keep]       stub old tool outputs, keep newest [keep] in full (no LLM)
/compact [k|to]    agent commits durable docs, writes a future-self summary, replaces history
                   (k = compact to ~k thousand tokens this once; `to` prompts for it)
/save [name]       name the current session
/load [name]       resume a session (no arg = picker)
/session           list | delete <name>
/prompt [name]     view/edit prompt files (/prompt use <name> = fire one as a turn)
/sh [cmd]          run a command yourself in a terminal (no arg = your $SHELL)
/connect [host]    run tools on a remote box over SSH (no arg = pick saved/open)
/local [host]      detach the active remote (keep it open to switch back)
/machines          manage saved remote hosts (list/add/remove)
/tools             enable/disable lean-tools
/mcp               manage MCP servers (add/remove/reconnect; no arg = enable/disable menu)
/reload            reload lean-tools + pick up prompt edits
/model [name]      switch model across enabled providers (no arg = list)
/provider [name]   switch/manage the model provider
/usage             session tokens + context / provider usage
/think [level]     set thinking level (no arg = menu)
/effort [level]    set reasoning effort (no arg = menu)
/set [key val]     edit app config (config.toml knobs; no arg = menu)
/provider set [k v] get/set a backend-specific provider knob
/approve [mode]    confirm cadence: ask | session | auto
/leash [level]     capability ceiling: chat | r | rw | rwe
/autosave [on|off] autosave + auto-load last on start
/incognito [on|off]don't save the session locally
/askread [on|off]  confirm read tools too
/bg [kill <pid>]   list/kill background tasks
/info              live session read-out
/activity [n|all]  what the system did automatically (compaction, trim, fallback, …)
/expand [N]        show a tool call's full (untruncated) args
/help              list commands
/quit              exit
```

`/<cmd> ?` shows help for a command. Tab completion and history work when `readline` is
available; menus are arrow-key navigable.

`/prompt` edits the built-in prompts (`system`, `compact`, `auto_compact`,
`compact_nudge`; `/prompt reset <name>` reverts). `/prompt new <name>` creates your own,
and `/prompt use <name>` sends one as a turn.

## Remote workspace

`/connect <[user@]host> [path]` runs all file and shell tools on a remote box over SSH.
The prompt shows `[remote: host] ›`.

- The remote only needs `python3`; the script is copied over (skipped if unchanged).
- The remote executor has no config, model, secrets, or network access; it only runs
  approved tool calls in that directory.
- Previews and confirmations happen locally; remote edits show a real diff.
- SSH authenticates once per connection; its output never enters the model's context.

The session stays the same when switching between local and remote.

## Agent loop

1. Send `[system] + history + latest turn` with the tool list.
2. Stream the reply.
3. Run any tool calls, append results, repeat.
4. A reply with no tool calls ends the turn.
5. Tool rounds per turn are capped.

A batch of only read-only calls runs concurrently; any batch with a write or command runs
in order, as does everything on a remote.

## Development

Three gates, from the repo root (each exits non-zero on failure):

```bash
python3 tests/_smoketest.py     # unit suite (incl. the overhead limits in this README)
python3 tests/_mocktest.py      # scripted end-to-end suite
bash tests/_sweep.sh            # hygiene lint (stray unicode, likely secrets/PII, etc.)
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [LEAN_TOOLS.md](LEAN_TOOLS.md), and
[PROVIDER_API.md](PROVIDER_API.md).

## License

[MIT](LICENSE).
