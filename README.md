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

A lean-tool adds nothing to the context until it is enabled.

- Runs against local models or hosted APIs, through bundled or your own providers.
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
python3 lean_coder.py
```

**Runtime:** Python 3.11+, no third-party packages for the core. Some opt-in lean-tools
have their own deps (e.g. `web_screenshot` needs Playwright) and say so when enabled.
`/connect` needs an `ssh` client. A provider that needs more says so when enabled.

**Updating:** re-run the installer, or `git pull && ./install.sh`. With the `update`
lean-tool enabled, `/update` pulls a newer `lean_coder.py` (`update_track` = `stable` or
`beta`; `auto_update = true` checks at launch). From a shell (or over ssh),
`lean_coder --update [check|force]` does the same and exits; it doesn't need the tool enabled.

## Quick start

```bash
lean_coder
```

On first start, pick a backend with `/provider` (hosted ones ask you to log in the first
time they're used), then a model with `/model`. Type a request. The agent shows a diff
before each edit and confirms per your `/approve` setting. `/help` lists commands.

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

A provider connects lean-coder to a model backend: a local server or a hosted API. Use one
of the bundled providers (`/provider` lists them and switches; `/provider login <name>`
stores a key) or write your own: copy
[`examples/providers/example.py`](examples/providers/example.py) into `providers/` and see
[PROVIDER_API.md](PROVIDER_API.md). `/model` lists and switches models across enabled
providers. Sessions work across providers.

Keys are stored in a `chmod 600` file under `~/.config/leancoder/`, never in
`config.toml`; a provider's env var is used instead if set.

## Safety

Two independent settings:

- **`/leash`** - which tools the model gets:
  `chat` (none) · `r` (read files) · `rw` (+ edit files) · `rwe` (+ run commands).
  The model is told its level.
- **`/approve`** - when to confirm:
  `ask` (every edit/command) · `session` (once per run) · `auto` (never).

Examples: `leash r` for unattended analysis; `leash rw` + `approve auto` for unattended
editing. OS file permissions still apply. `/incognito` writes nothing to disk;
`/askread` also confirms read tools.

## Tools

### Core

File, search, edit and shell tools are always present, along with a pinned plan
(`update_plan`), a session notebook (`note`), and `ask_user_to_run`, which hands you a
command that needs sudo, a prompt or a password; only the command and exit code return to
the model. Read-only calls in one turn run concurrently.

### Lean-tools

A lean-tool is one `.py` file with a `TOOL` schema and a `run` function. Bundled ones live
in [`lean-tools/`](lean-tools/); `/tools` shows them and turns them on or off. Write your
own from [`examples/lean-tools/`](examples/lean-tools/); see [LEAN_TOOLS.md](LEAN_TOOLS.md).

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

1. Update the project's docs.
2. Write a summary: goal, decisions, state, next step.
3. Pin a goal + TODO.
4. Write the next instruction for itself.

Older history is replaced by the summary (the most recent turns stay verbatim), and the
next instruction is sent as a new turn after a short `^C` window. Compaction is nudged,
then forced, as the window fills; thresholds and prompts are settings (`/set`, `/prompt`).

- **Wake on background finish:** a finished background task or worker can start a turn
  when the prompt is idle (`wake_on_bg_finish`).
- **Send window:** `window_messages = N` sends only the last N messages, cut at a turn
  boundary. For very small models.

## Configuration

Precedence: **CLI flag > env var > config file > default**. `lean_coder --help` lists the
flags; `/set` shows and edits every setting with its current value.

Config is `~/.config/leancoder/config.toml` and saves automatically on any change. The
context window is read from the backend; `--num-ctx` overrides it.

**What stays on disk.** Only things you named: config, sessions (`~/.config/leancoder/sessions`),
and task boards. Everything else a running lean-coder creates (background task logs, worker
files, file-claim boards) lives in one folder per process under `$XDG_RUNTIME_DIR` (else
`$TMPDIR` or `/tmp`). It is removed when that process exits; after a crash the next start on
the same machine removes any folder whose process is gone. A remote executor does the same on
its machine and deletes its copied code when the session closes or the link has been down
longer than `LEANCODER_REMOTE_IDLE_TTL` (default 30 min).

## Slash commands

`/help` lists every command; `/<cmd> ?` explains one. Tab completion and history work when
`readline` is available; menus are arrow-key navigable.

`/prompt` edits the built-in prompts (`/prompt reset <name>` reverts). `/prompt new <name>` creates your own,
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
