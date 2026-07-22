<div align="center">

# lean-coder

**A small terminal coding agent, dependency-free at its core, that treats context as the scarce resource it is.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
![Core dependencies: none](https://img.shields.io/badge/core%20dependencies-none%20(stdlib%20only)-brightgreen.svg)
![Baseline overhead: ~2.5k tokens](https://img.shields.io/badge/baseline%20overhead-~2.5k%20tokens-orange.svg)

[Install](#install) &middot; [Quick start](#quick-start) &middot; [Why it exists](#why-it-exists) &middot; [Providers](#providers-pick-a-model-backend) &middot; [Safety](#safety-two-axes) &middot; [Tools](#tools) &middot; [Compact](#compact-the-agent-documents-its-work-before-a-memory-wipe)

![lean-coder auditing and fixing a real bug on a local model](demos/demo.gif)

<sub>A local model (Qwen3-Coder:30b via Ollama) auditing a Python package, finding the root cause of a bug, and fixing it.</sub>

</div>

It reads, edits, and runs code in your project through a model's native tool-calling
API, and it's built around one idea most agents ignore: **your context window is the
scarce resource, so don't waste it describing the tool.**

Mainstream coding agents spend a lot of your window on themselves before you type a
word. The big ones run to tens of thousands of tokens of system prompt and built-in
tool scaffolding per session, and that's *before* you connect any MCP servers, which
add anywhere from hundreds to tens of thousands of tokens each on top. That's context
that can't hold your actual code. lean-coder's *entire* shipped tool surface plus its
system prompt costs about **~2.5k tokens**. It's just a different
category: the platform stays out of the way so the window holds your work, and it's
usable on small local models, not just frontier ones.

And when a long task *does* fill the window, it doesn't truncate and forget. Instead
the agent **documents its own work, pins a goal + plan, and hands over to a clean
slate**, so the job continues instead of dying mid-run.

If your model handles MCP well, brilliant: add all the servers you like on top, we're
big fans. We just didn't want the platform *itself* to be the thing eating your
context. This is open source, built for the folks running local and open models first.

> **Why bet on local?** Two curves are bending our way at once. Open-weight models now
> trail the closed frontier by only ~3-4 months on [Epoch AI's Capability Index](https://epoch.ai/data-insights/open-weights-vs-closed-weights-models)
> (down from years) - and the [*densing law*](https://arxiv.org/abs/2412.04315) finds
> capability density doubling roughly every ~3.5 months, i.e. the same ability in
> exponentially fewer parameters over time. Together that means the models keep getting
> both **smarter and smaller**: what needs a datacenter today runs on a consumer GPU
> tomorrow. A lean platform that already runs the small/open models is a bet on where
> that curve is heading, not just where it is.

The same tiny codebase scales across the whole range:

- drive a **small local model with a few thousand tokens of context** - the ~2.5k
  baseline leaves room to actually work;
- point it at a **frontier model and let it spawn parallel background workers** on
  scoped sub-tasks, running a job far bigger than one context window;
- run it **on your phone in [Termux](https://termux.dev)** - it's just `python3`,
  nothing to compile;
- or **`/connect` to a beefier box over SSH** and run every tool *there* through a
  hermetic, secret-free executor, from the same terminal.

It's a compact, **stdlib-only** Python codebase, no third-party packages. The core
is one script (`lean_coder.py`); alongside it ship a required builtin-tools module
(`lean-tools/builtins.py`) and a set of model provider adapters in `providers/`
(Ollama, llama.cpp, MLX, Anthropic, Gemini, Groq, OpenAI, OpenRouter). Point it at a local model via
[Ollama](https://ollama.com) (bundled and default-enabled) and you get an interactive
REPL that edits code, runs commands, and shows a diff before it touches anything.

What a session looks like:

```
$ lean_coder
lean-coder  <your-model> @ <your-provider>
  cwd: ~/myproject   ·   baseline overhead (system + always-on tools): ~2.5k tokens
› add a --json flag to the export command and update the tests
● I'll look at the export command first.
  ⚙ read_file(path=src/export.py)
  ⚙ search_files(pattern=def export)
  ...
```

## Install

On Linux / WSL:

```bash
curl -fsSL https://raw.githubusercontent.com/codemonkeying/lean-coder/main/install.sh | bash
```

On Android / Termux:

```bash
pkg install -y python curl && curl -fsSL https://raw.githubusercontent.com/codemonkeying/lean-coder/main/install.sh | bash
```

## Quick start

```bash
lean_coder                       # uses your config, or localhost Ollama + default model
```

You get an interactive REPL. Type a request; the agent reads, edits, and runs as
needed, **showing a diff before applying** and **confirming before running** shell
commands (unless approval is `session`/`auto`). Type `/help` for the full command list.

## Requirements

- **Python 3.11+** (uses the stdlib `tomllib`). No third-party packages for the
  core; a couple of opt-in lean-tools have their own (e.g. `web_screenshot` needs
  Playwright) and are disabled by default.
- A tool-calling model behind a provider:
  - **Ollama** (local or self-hosted) - e.g. `ollama pull qwen3-coder:30b` - works
    out of the box, or
  - a **hosted API** - Anthropic, Gemini, Groq, OpenAI, or OpenRouter all ship
    bundled; enable one with `/provider login <name>` (see [Providers](#providers-pick-a-model-backend)).

## Install options

The one-liners above fetch and run `install.sh`, which installs the code and
symlinks `lean_coder` onto your `PATH`. On WSL it's a normal Linux install. On Termux
there's no sudo/systemd, so it points you at a remote Ollama instead
(`lean_coder --host http://HOST:11434`), so you don't need anything beyond Python
there. If you want a local Ollama on Linux, pass `--with-ollama --pull` (see below).

Prefer to inspect first? Clone and run it yourself:

```bash
git clone https://github.com/codemonkeying/lean-coder
cd lean-coder
./install.sh                     # install the code, symlink `lean_coder` onto PATH
./install.sh --with-ollama --pull --model qwen3-coder:30b   # also install Ollama + pull a model
./install.sh --dry-run           # show what it would do, change nothing
./install.sh --help              # all options
```

It's idempotent and only does the heavy / privileged steps when you ask.
`./uninstall.sh` does a full teardown.

Or run it in place with no install at all (from a clone, so the bundled
`lean-tools/` and `providers/` sit beside the script):

```bash
python3 lean_coder.py                         # local Ollama, default model
python3 lean_coder.py --host http://box:11434 --model qwen3-coder:30b
python3 lean_coder.py --cwd ~/myproject
```

**Updating:** re-run the one-liner (or `git pull && ./install.sh`) any time; it
updates an install in place. Or enable the `update` lean-tool and run `/update`
from inside the REPL: it compares this build's version against the published
`VERSION` and only pulls a newer `lean_coder.py` (`/update check` reports without
changing anything; `/update force` re-pulls regardless). Set `auto_update = true`
to have that check run once at launch (off by default). Versions follow SemVer;
`update_track` picks the `stable` (default) or `beta` release line.

## Why it exists

Every agentic turn re-sends the whole conversation (system prompt, tool schemas,
and the growing history) to the model. On a small local model with a 32k window,
a bloated system prompt or a verbose tool schema is context that can't hold your
actual code. lean-coder treats context as the scarce resource it is:

- **Baseline overhead around ~2.5k tokens** for the system prompt *and* the entire
  always-on tool surface combined (a test enforces the ceiling). The surface scales
  with the leash, so you only pay for what you've enabled:

  | tier | ~tokens | what's in it |
  |---|---|---|
  | chat | ~500 | system prompt alone, no tools |
  | read (`r`) | ~1.2k | + update_plan, note, read_file, list_files, search_files |
  | write (`rw`) | ~1.65k | + apply_diff, replace_lines, write_file |
  | exec (`rwe`) | **~2.4k** | + run_command, background, ask_user_to_run - the fresh-session floor |

  That **~2.4k** is the whole shipped agent: system prompt plus all eleven always-on
  builtin tools. On top, the **optional bundled lean-tools** are off by default and
  cost nothing until you `/tools` them on. Roughly:

  | lean-tool | ~tokens | | lean-tool | ~tokens |
  |---|---|---|---|---|
  | dispatch_worker | ~860 | | web_fetch | ~180 |
  | web_screenshot | ~525 | | brave_search | ~165 |
  | shell_session | ~360 | | ssh | ~145 |
  | symbols | ~335 | | diagnostics | ~65 |
  | git_summary | ~65 | | word_count | ~55 |

  Turn on **every** one and the total is still **under ~5.5k tokens**. The meter always
  shows the real current figure.

  **For scale:** a *single* MCP server's tool definitions are commonly
  [300-710 tokens **per tool**](https://dev.to/piotr_hajdas/mcp-token-limits-the-hidden-cost-of-tool-overload-2d5),
  and a typical server (e.g. Slack, ~10-15 tools)
  [runs ~2,000 tokens](https://www.mindstudio.ai/blog/claude-code-mcp-server-token-overhead) -
  as much as lean-coder's *entire* shipped surface. Connect a few and you can burn
  [50k+ tokens before the first question](https://dev.to/kenimo49/your-mcp-server-eats-55000-tokens-before-your-agent-says-a-word-i-measured-the-real-cost-19l8).
  lean-coder is a generic MCP client too (see below). The difference is you pay that
  cost only for what you deliberately add.
- **Truncated results** - a single big file read or noisy command can't blow the budget.
- **Layered context management** - documented compaction (automatic, on by default)
  plus ingestion-time output caps, reclaiming space mid-task
  without losing the thread (`/compact` and `/trim` are the manual levers). More below.

The payoff: it stays usable on **small local models**, not just frontier hosted
ones. The tools are deliberately small and few, because that is what a 7B-30B model
can drive reliably.

## Highlights

- **No third-party dependencies in the core.** `python3` (3.11+) is the entire
  runtime, nothing to `pip install` to edit, run, and drive a model. (A few opt-in
  lean-tools bring their own deps (e.g. `web_screenshot` needs Playwright) and say
  so; they're disabled by default and cost nothing until you enable them.)
- **Roll your own tools - one drop-in file.** A small always-on set covers
  read/edit/run. Everything else is an **opt-in "lean-tool"**: a single `.py` file
  with a `TOOL` schema and a `run` function, dropped in a directory and toggled on
  with `/tools`, costing zero context until enabled. No server, no schema registry,
  no build step, no dependency. Write it, drop it, `/reload`.
- **Roll your own provider just as easily.** A model backend is a `.py` file with a
  `PROVIDER` dict in `providers/`. Ollama is bundled and default (a fresh install
  just works against `localhost:11434`); drop your own file in to talk to any other
  backend, and `/model` lists models across every enabled provider and switches in
  one step.
- **Self-managing context via compaction (on by default).** The standout feature and a
  main reason this exists: rather than blindly truncating, the agent *documents its
  own work before a memory wipe*. As context fills (a **soft ~70%** nudge, a **hard
  ~90%** force, an **emergency ~100%** stop) it commits durable docs to disk, writes a
  summary for its future self, **pins a goal + TODO that survive the wipe**, and leaves
  a self-prompt, then compacts and, by default, **auto-continues the same task from a
  clean slate** via that self-prompt. So a long-running job keeps going far past the
  window, with documentation that never goes stale and a plan that never gets lost.
  `/trim` complements it for lighter, no-LLM stubbing. (More below.)
- **Two independent safety axes:** a capability ceiling (`/leash`) and a confirm
  cadence (`/approve`) that compose from supervised editing to full autonomy. See
  [Safety](#safety-two-axes) below.
- **Remote workspace (the executor).** `/connect host` pushes a hermetic,
  secret-free **executor** onto another box over SSH; every tool then runs *there*,
  on the remote's filesystem, transparently to the model, with no install beyond
  `python3`, no config/model/keys/network on the remote. Edit from a phone in
  Termux while the code, builds and commands all execute on a workstation or server.
  Keep several boxes open and switch instantly; confirmations and diffs stay local.
- **Parallel background workers.** With a capable model, `dispatch_worker` hands a
  scoped sub-task to a headless worker agent that runs in the background on its own
  context, then reports just its result back, so the main session spends its window
  on the plan, not the raw output. Run several at once.
- **Any tool can return an image.** On a vision-capable model, a tool result can
  carry a picture (a screenshot, a rendered chart, a diff image) and lean-coder
  feeds it to the model the right way for each backend (Anthropic, OpenAI, Gemini).
  The image rides only in the request, never bloating saved history, and it's gated
  to models that can actually see. `web_screenshot` is the ready-made example.
- **One conversation, any model.** The full history is re-sent each turn, so you can
  switch model or provider mid-session with `/model` (lists every enabled backend)
  without starting over: drive a local Ollama for the cheap steps, jump to a
  frontier model for the hard one, and back.
- **Nothing happens off-screen.** `/activity` replays what the system did on its own -
  compaction, trim, model fallback, ingestion caps, so the automatic context
  management is auditable, not magic.
- **Updates and spreads itself.** `/update` self-updates the script to the latest
  published build (`stable` or `beta` track; `auto_update` checks at launch), and
  `/provision` installs lean-coder onto another box over SSH.
- **Sessions.** Conversations autosave each turn and the last one auto-loads on
  start, so a relaunch picks up where you left off. Keep as many as you like -
  `/save <name>`, `/load` between them, `/session` to list, and because
  `/connect` moves only the *executor*, a single saved session can drive your
  laptop one moment and a remote box the next without losing a thing.
- **Pinned input.** Type your next message while the model works - output scrolls
  above a fixed input line. Pure ANSI + stdlib; falls back on terminals that can't
  support it.

## Providers: pick a model backend

A **provider** is the adapter that connects lean-coder to a model backend.
**Ollama ships bundled and default-enabled.** A fresh install talks to a local
Ollama at `localhost:11434` with zero config, so if you have Ollama running you're
already done.

To use a **hosted API** instead, several ship bundled (disabled until you enable
one and add a key):

| Provider        | Backend | Get a key |
|-----------------|---------|-----------|
| `ollama`        | Local / self-hosted Ollama (default) | none needed |
| `anthropic_api` | Anthropic API (Claude) | [console.anthropic.com](https://console.anthropic.com) |
| `gemini`        | Google Gemini | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `groq`          | Groq (fast, free tier) | [console.groq.com](https://console.groq.com/keys) |
| `openai`        | OpenAI (gpt / o-series; paid) | [platform.openai.com](https://platform.openai.com/api-keys) |
| `openrouter`    | OpenRouter (gateway to many models) | [openrouter.ai](https://openrouter.ai/keys) |

**Setup is one step. Just log in and paste your key:**

```
/provider login anthropic_api      # prompts for the key, saves it, switches to it
```

That's it. `/provider login <name>` enables the backend, securely stores the key
(a `chmod 600` file under `~/.config/leancoder/`, **never** in `config.toml`), and
makes it active. If you'd rather use an environment variable, export the provider's
key var (e.g. `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`,
`OPENROUTER_API_KEY`) and lean-coder picks it up automatically.

Then:

```
/provider            list backends + status, switch between enabled ones
/model               list models across every enabled provider, switch in one step
```

If you launch with no backend reachable, lean-coder tells you exactly these
commands. And if a turn ever fails because a key is missing or rejected, it offers
the login prompt right there and retries, so you're never stranded.

To wire up any other backend, copy
[`examples/providers/example.py`](examples/providers/example.py), an annotated
OpenAI-compatible template, into `providers/`. See
[PROVIDER_API.md](PROVIDER_API.md) for the full interface. Sessions are portable
across providers (the full history is re-sent each turn), so a conversation saved
against one resumes against another.

## Safety: two axes

**What it *can* do** and **whether it *asks*** are separate knobs that compose.

- **`/leash` - capability ceiling** (`chat` | `r` | `rw` | `rwe`, default `rwe`).
  Bounds the tools the model is even *given*:
  - `chat` - no tools at all (pure conversation).
  - `r` - **read-only**: read/list/search files. Safe to walk away from.
  - `rw` - read **+ edit files** (`apply_diff`, `write_file`).
  - `rwe` - read + edit **+ run shell commands** (full agent).

  The ceiling bounds what the model can even attempt, so it *cannot* act outside the
  fence. Drop to `r` or `rw` and walk away. The model is told its ceiling, so it says
  "I'm read-only; `/leash rw` to let me edit" rather than failing opaquely.
- **`/approve` - confirm cadence** (`ask` | `session` | `auto`, default `ask`).
  *When* to confirm within the ceiling:
  - `ask` - confirm every edit/command (the default; you see a diff / the command first).
  - `session` - confirm once, then auto-approve the rest of this run.
  - `auto` - never ask (full autonomy).

Compose them: `leash r` = walk-away-safe analysis; `leash rw` + `approve auto` =
unattended editing; `leash rwe` + `approve auto` = full autonomy. The leash bounds
what the agent *attempts*; the **OS** (its file perms) bounds what it *can* do.

- **`/incognito`** writes nothing to disk for the run and tells the model - honest
  about its limits (it only avoids a *local* trace, not provider-side processing).
- **`/askread`** extends confirmation to read tools too.

## Tools

### The always-on set

Exposed to the model via native tool calling. Descriptions are one line each,
because they are serialized into **every** request, so keeping them terse is part of
the context budget. Eight file/shell tools live in the required builtin-tools module
(`lean-tools/builtins.py`):

| Tool            | What it does |
|-----------------|--------------|
| `read_file`     | Line-numbered file contents; optional line range; large files truncated. |
| `list_files`    | Directory / shallow project tree, honoring ignore rules. |
| `search_files`  | Regex search -> `file:line` matches (capped). |
| `apply_diff`    | **Preferred edit tool.** SEARCH/REPLACE blocks - sends/returns only changed lines. |
| `replace_lines` | Replace a line range by number (simpler than a diff when you have the line numbers). |
| `write_file`    | Create or overwrite a whole file (mainly for new files). |
| `run_command`   | Run a foreground shell command in the project dir; stdout/stderr truncated. |
| `background`    | Run + manage long-lived tasks (dev servers, watchers, builds): `run` starts one detached, `status` lists/inspects, `kill` stops one - with optional notify/heartbeat/max-runtime watchdog. |

Alongside these the model always has **`update_plan`** (maintains a pinned goal +
TODO that survives compaction) and **`note`** (a session notebook - episodic memory
that survives compaction and travels with the session), plus, by default,
**`ask_user_to_run`**, the escape hatch for anything `run_command` can't do itself: a
command needing **sudo/root**, an
**interactive prompt**, or a **typed password or secret**. Instead of running it, the
agent hands the exact command back to *you* to run in your own terminal; only the
command and its exit code return to the agent. **Whatever you type (the password,
the secret) never reaches the model**. You can also edit the command before running
it, or decline. On by default; toggle with `/set`. A batch of read-only calls in
one turn runs **concurrently**.

### Opt-in lean-tools

Anything beyond local edit + shell is a **lean-tool**: a single `.py` file with a
`TOOL` schema and a `run` function, discovered but **disabled by default**. Turn one
on with `/tools` and it costs context only from that point. These ship bundled in
[`lean-tools/`](lean-tools/), ready to enable:

| Lean-tool         | Adds |
|-------------------|------|
| `dispatch_worker` | Hand a scoped sub-task to a background worker agent; collect its result. Adds `/worker` (list workers / `/worker <pid>` = full result / cancel). |
| `web_fetch`       | Read a URL as clean text. |
| `web_screenshot`  | Screenshot a URL with a headless browser + return the page text (and, on a vision model, the image itself). **Needs [Playwright](https://playwright.dev/python/) + a browser** (`pip install playwright && playwright install firefox`); says so if absent. Disabled by default. |
| `brave_search`    | Web search (Brave API). |
| `git_summary`     | Read-only git snapshot (branch, status, diffstat, recent commits). |
| `diagnostics`     | Lint/typecheck a file with **whatever's already installed** on the box (or the remote executor when `/connect`'d). Zero deps of its own: it uses a richer checker if present (pyright, ruff, tsc, eslint, phpstan, clippy, shellcheck…) and falls through to always-available basics (`py_compile`, `bash -n`, `gcc -fsyntax-only`, stdlib JSON…). Never installs or pushes anything. |
| `symbols`         | Navigate Python code without grepping: outline a file/dir's classes+defs, or locate a definition by name (zero-dep, stdlib `ast`). |
| `shell_session`   | A persistent interactive shell the model holds open across calls (REPL, ssh, etc.). |
| `ssh`             | One-shot `ssh host cmd` (network egress, kept out of core). |
| `notify`          | Desktop notification when a long task finishes. |
| `provision`       | `/provision` wizard: install lean-coder onto another box over SSH. |
| `update`          | `/update` - self-update `lean_coder.py` to the latest published build. |
| `word_count`      | Count lines / words / chars in a file. |

**Dependencies:** almost all of these are stdlib-only. Two need a one-time setup, and
each prints these exact steps if you enable it without them:

- **`web_screenshot`** needs [Playwright](https://playwright.dev/python/) and a browser:
  ```bash
  pip3 install --user --break-system-packages playwright
  python3 -m playwright install firefox        # or chromium / webkit
  python3 -m playwright install-deps           # if the host is missing system libs (may need sudo)
  ```
- **`brave_search`** needs a free Brave Search API key (2,000 queries/month, no card)
  from [search.brave.com/app/keys](https://search.brave.com/app/keys). Put it in
  `~/.config/leancoder/brave.key` (chmod 600) or set `LEANCODER_BRAVE_KEY`.

`diagnostics` is the special case: no deps of its own, but it opportunistically *uses*
external linters if the box already has them (see its row). Every dep-bearing tool is
disabled by default, costs nothing until enabled, and says so clearly if the dep is
missing. Full per-tool notes live in [LEAN_TOOLS.md](LEAN_TOOLS.md).

**Not built (yet):** there's no persistent **LSP** (Language Server Protocol)
integration. That would be a heavy, dependency-per-language addition (great for
big PHP/Java/TS projects with go-to-definition, project-wide rename, live type
errors). We haven't built it because we don't need it: `diagnostics` (one-shot lint)
and `symbols` (stdlib `ast` navigation) cover our day-to-day. If you want proper LSP
and would use it, open an issue and we'll build it.

Each is small enough to read in a sitting and usable by small models. For learning
the pattern, [`examples/lean-tools/`](examples/lean-tools/) has two annotated
templates: `reverse_file.py` (a minimal local read-only tool) and `weather.py` (an
API-backed tool with a key + network egress). See [LEAN_TOOLS.md](LEAN_TOOLS.md);
writing your own is a drop-in `.py`; [LEAN_TOOLS.md](LEAN_TOOLS.md) walks the pattern.

### MCP servers (Model Context Protocol)

lean-coder is a **generic MCP client**, builtin, with **no servers shipped by default** (zero
context, zero surface until you add one). Point it at any MCP server and its tools join
the model's surface, namespaced `mcp__<server>__<tool>`:

```
/mcp add fs npx -y @modelcontextprotocol/server-filesystem /some/dir   # stdio server
/mcp add gw https://mcp-gateway.example.com/mcp/handbook/mcp           # HTTP server
/mcp                       # enable/disable menu (per server)
/mcp list                  # servers + connection state
/mcp reconnect [name]      # (re)connect
/mcp remove <name>
```

Two transports, both stdlib-only: **stdio** (a spawned subprocess, JSON-RPC over its
pipes, the Claude-Desktop shape: `command` + `args` + `env`), and **HTTP** (streamable
MCP, tolerating SSE or plain-JSON responses). HTTP auth is one `Authorization: Bearer`
header: a static token/env, or an **OAuth 2.1** client-credentials JWT fetched + cached
+ refreshed automatically. Prefer **OAuth 2.1** where the gateway offers it (short-lived
tokens that self-expire and auto-refresh); use a static bearer only when there is no OAuth
endpoint. For auth/env, edit the `mcp_servers` table in `config.toml`:

```toml
[mcp_servers.gw]
transport = "http"
url = "https://mcp-gateway.example.com/mcp/handbook/mcp"
auth = { type = "bearer", token_env = "GW_KEY" }
# or: auth = { type = "oauth", token_url = "…/oauth/token", client_id = "…", client_secret_env = "GW_SECRET", scope = "mcp:access" }
```

MCP tools run on the **driver** (never a connected remote) and ride the **`rwe`** leash
tier (they may have side effects), confirming like any non-safe tool unless approval is
armed. Enabled servers connect at launch; a dead one just contributes no tools (its
error shows in `/mcp list`). Full guide (transports, auth (bearer + OAuth 2.1), config,
troubleshooting) in [MCP.md](MCP.md).

### `apply_diff` format

The `diff` argument is one or more SEARCH/REPLACE blocks:

```
<<<<<<< SEARCH
exact existing text (must match the file verbatim)
=======
replacement text
>>>>>>> REPLACE
```

Blocks apply in order. If any SEARCH text isn't found, **nothing is written** and
the model is told to re-read and match exactly. The markers are word-bearing so a
model emits them reliably and they never collide with a plain row of `=` or a git
conflict marker in the file.

## Context discipline (this is the product)

- **Prefer `apply_diff`** over `write_file` for existing files - results stay minimal.
- **Ignore rules:** reads `.gitignore` and an optional `.leancoderignore`, plus
  built-in defaults (`.git/`, `node_modules/`, `dist/`, `*.lock`, binaries…). Ignored
  paths are never listed, searched, or walked.
- **Truncation:** large reads and command outputs are clipped head/tail with a clear
  `…[truncated N …]…` notice.
- **Budget meter:** after each turn a context-token figure (the real count from the
  provider when available, else an estimate) prints against the window, colored as it
  climbs. `/ctx` and `/usage` report it on demand.
- **`/trim [keep]`:** stubs old tool results (file dumps, command output) to
  one-line placeholders, keeping the newest `keep` in full. No LLM call. The lighter
  lever, reclaiming the biggest context consumer without touching the conversation or
  any edits. Fires manually or, in an emergency, automatically.

### Compact: the agent documents its work before a memory wipe

Compaction is the feature that keeps a long-running task alive. The insight: **the
agent knows what's in its own head**, so the sensible moment to update documentation
is *right before* that memory is compacted, not after it's already gone stale.

On `/compact` (or automatically, see below) the model gets a full tool-capable turn
to:

1. **Persist durable docs.** Update whatever the project already uses (a design doc,
   README, notes) with its current understanding, and commit them if the project is
   already a git repo.
2. **Write a summary for its future self.** The goal, key decisions, current state
   (done / in progress / next), *where* the durable docs live, and a prompt to
   continue from.
3. **Pin a goal + TODO** that survives the wipe.
4. **Write a self-prompt.** The single next instruction.

The older conversation is then replaced with just that summary block (the most recent
turns are kept verbatim), a "smart `/clear`" that keeps the thread; and, unless
disabled, the self-prompt is **fed straight back in as the next turn**, so the agent
continues the same task from a clean slate (a 5-second `^C`-to-cancel beat precedes
it). Durable state lives on disk and in the pinned plan; the context window resets.
That's how a task outruns the window while its documentation stays current instead of
rotting.

(Note: "handover" is reserved for a future feature that transfers a live session to
another model or human; today's self-shrinking behaviour is `/compact`.)

- **Ingestion-time output caps.** Every tool result passes through one cap on the way
  in: a runaway `mcp.call` or lean-tool result is sized to a share of the free window
  (head + tail kept, middle marked) so no single result can blow the context, with a
  loud notice to both you and the model when it truncates. Results are born small
  rather than clawed back later.
- **Self-managing (on by default).** As context fills, the agent manages it in tiers:
  a **soft zone** (~70% of the window) where it's nudged to wrap up at a *clean break*
  and compact tidily; a **hard threshold** (~90%) that forces a compaction at a
  boundary; and an **emergency** stop (~100%) that compacts immediately. A loop guard
  boundary; and an **emergency** stop (~100%) that compacts immediately. A loop guard
  (~1/min) stops a compact->continue->compact spin. After a compaction the last few
  turns are kept verbatim (`compact_keep`, default 3; tool payloads in that tail are
  stubbed so it stays cheap); the emergency overflow path uses a hardcoded minimal
  backstop. All thresholds are tunable per model via `/set` (`compact_soft`,
  `compact_hard`, `compact_emergency`, `compact_keep`, `auto_compact`,
  `autostart_after_compact`), and the prompts themselves are editable.
- **Autonomous wake on background finish (off by default).** With
  `wake_on_bg_finish = true` (via `/set`), a finished background task or worker
  wakes the agent with a synthesised turn so it reacts to the result with no operator
  input; otherwise the finish notice waits passively for your next turn. A single
  job can opt in on its own with `run_command`'s `notify_on_exit` / `heartbeat_timeout`
  / `max_runtime` args, which wake the agent for *that* job even when the global
  setting is off. See LEAN_TOOLS.md for details.
- **Bounded send-window (off by default).** For a very small local model, even the
  compaction flow can be too much history to hold. `window_messages = N` (via `/set`)
  caps each request to the last N messages, cut at a *whole-turn boundary* so the
  current task is never truncated, giving a hard token bound every turn, at the cost of the
  model seeing only recent turns. Most models are better served leaving this off and
  letting compaction manage size; reach for it on tight local windows.

## Configuration

Precedence: **CLI flag > env var > config file > default**.

| Setting         | Flag                   | Env               | Default                   |
|-----------------|------------------------|-------------------|---------------------------|
| Ollama endpoint | `--host`               | `OLLAMA_HOST`     | `http://localhost:11434`  |
| Model           | `--model`              | `LEANCODER_MODEL` | `qwen3-coder:30b`         |
| Context window  | `--num-ctx`            | -                 | auto-detect, capped 32768 |
| Project dir     | `--cwd`                | -                 | current directory         |
| Approval mode   | `--approval` / `--auto`| -                 | `ask` (confirm each)      |
| Capability      | `--leash`              | -                 | `rwe`                     |
| Resume session  | `--resume <name>`      | -                 | auto-load last for cwd    |

Config lives in `~/.config/leancoder/config.toml` and **autosaves**: any change
(`/set`, `/model`, `/provider`, `/approve`, …) is written back immediately. It also
supports tiered host failover, memorable machine names, per-machine default models,
and saved `/connect` targets; run the tool and see the config for the full set.

**Context auto-detection:** without `--num-ctx`, and when `auto_num_ctx` is on,
lean-coder reads the model's window from the provider at startup. Auto-detect is only
allowed to *lower* the window (capped at 32768) so a model advertising a huge max
can't exhaust memory; pass `--num-ctx` to go higher explicitly.

## Slash commands

```
/clear             wipe conversation, stay in this session
/new [name]        start a separate session
/trim [keep]       stub old tool outputs, keep newest [keep] in full (no LLM)
/compact           agent commits durable docs, writes a future-self summary, replaces history
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
/ctx               context-token estimate
/info              live session read-out
/activity [n|all]  what the system did automatically (compaction, trim, fallback, …)
/expand [N]        show a tool call's full (untruncated) args
/help              list commands
/quit              exit
```

Any command answers **`/<cmd> ?`** (or `/<cmd> help`) with its own detailed help -
args, aliases, and behaviour. When stdlib `readline` is available, **Tab** completes
commands and their arguments (and offers `?`), and you get line history; it degrades
to plain input if `readline` is missing. Menu pickers (`/set`, `/model`, `/think`, …)
are arrow-key navigable with type-to-filter on a real terminal, and fall back to a
numbered prompt when headless.

### Editable prompts

`/prompt` opens the prompt files in your editor. The built-ins (`system` (the system
prompt), `compact`, `auto_compact`, `compact_nudge`) can be tuned to taste and take
effect live (`/prompt reset <name>` reverts to the baked default). Overrides live in
`~/.config/leancoder/prompts/`.

You can also save your **own** named prompts (`/prompt <newname>` creates one) and fire
one as a one-shot turn with **`/prompt use <name>`**; it's injected as your next
message. Handy for reusable instructions you'd otherwise retype: a refactor brief, a
review checklist, a commit-message style.

## Remote workspace

`/connect <[user@]host> [path]` runs lean-coder's tools on a remote box over SSH.
Every file/exec tool then runs *there*, on the remote's own filesystem, while the
model's tool surface stays identical. The prompt shows `[remote: host] ›` so you
always know where you are.

- **No install on the remote beyond `python3`** - the script is pushed into throwaway
  space; the push is hash-skipped when a matching build is already there.
- **The executor is hermetic:** no config, no model, no secrets, no network egress -
  it only runs approved tool calls against the one directory.
- **Confirmations stay local:** the preview and `y/N` happen on your machine before
  anything is sent; remote edits still show a real unified diff.
- Auth happens once via a multiplexed SSH master socket; later calls reuse it, and
  none of the connection/install output ever enters the model's context.

## Agent loop

1. Build messages: `[system] + history + latest turn`; attach the tools.
2. Stream a chat completion from the provider.
3. If the reply has tool calls, execute each, append each result as a `tool`
   message, and loop.
4. Content with no tool calls is the final answer.
5. Capped tool rounds per turn prevent runaway loops.

A batch of tool calls that are **all read-only** runs concurrently (wall time = the
slowest call); any batch containing a writer or command runs sequentially, and a
connected session is always sequential. Results are appended in call order.

## Development

Three gates, run bare from the repo root (each exits non-zero on failure):

```bash
python3 tests/_smoketest.py     # offline unit suite (incl. the fixed-overhead budget check)
python3 tests/_mocktest.py      # scripted end-to-end suite
bash tests/_sweep.sh            # hygiene lint (stray unicode, likely secrets/PII, etc.)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the bar, [LEAN_TOOLS.md](LEAN_TOOLS.md)
for writing tools, and [PROVIDER_API.md](PROVIDER_API.md) for providers.

## License

[MIT](LICENSE).
