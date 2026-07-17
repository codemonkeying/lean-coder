# Writing lean-coder lean-tools

A lean-tool adds a tool to lean-coder. It is one `.py` file that defines a `TOOL`
schema and a `run` function. Enabled lean-tools appear to the model as normal
tools, local and remote alike. No protocol, no server, no dependency in the core
- the lean alternative to MCP. (Want to connect an *existing* MCP server instead of
writing a tool? lean-coder is also a generic MCP client - see [MCP.md](MCP.md).)

## The contract

```python
TOOL = {
    "name": "word_count",                      # unique; becomes the tool name
    "description": "Count lines/words/chars.",  # ONE line (serialized every request)
    "parameters": {                            # JSON Schema for the arguments
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "safe": True,                              # optional; True = read-only, no prompt
}

def run(args, cwd):                            # -> str (the model receives this)
    from pathlib import Path
    p = Path(cwd) / args["path"]
    return f"{len(p.read_text().splitlines())} lines"
```

That is the whole interface. Many runnable lean-tools ship bundled in
[`lean-tools/`](lean-tools/) (disabled until you `/tools`-enable them):
`word_count.py` (read-only), `note.py` (mutating), `diagnostics.py` (runs the
installed linter/typechecker for a file's language - the agent's feedback loop),
`git_summary.py` (a one-call read-only repo snapshot: branch, status, diffstat,
recent commits), `web_fetch.py` (GET an http(s) URL - network egress, never
`safe`, off by default), `symbols.py` (navigate Python code without grepping:
`mode='outline'` lists a file/dir's classes+defs with line numbers, `mode='def'`
locates a definition by name - zero-dep, stdlib `ast`), and `ssh.py` (one-shot
`ssh host cmd`, moved out of the core tool set because it's network egress rather
than local editing). For learning the pattern, annotated templates live in
[`examples/lean-tools/`](examples/lean-tools/): `reverse_file.py` (minimal local
read-only) and `weather.py` (API-backed with a key + network egress).

### `TOOL`
| key | required | notes |
|-----|----------|-------|
| `name` | yes | The tool name the model calls. Must be unique and must not collide with a built-in (`read_file`, `run_command`, ...). |
| `description` | yes | **Keep it to one line.** It is sent in *every* request, so it is a direct context cost. |
| `parameters` | yes | A JSON Schema object describing the arguments. Match what `run` reads. |
| `safe` | no | `True` for a read-only tool: it runs without a confirmation prompt **and becomes eligible to run concurrently** (see below). Omit (or `False`) for anything that writes, runs commands, or holds state - lean-coder will ask first (auto-approve mode waives that) and always runs it serially. |
| `glyph` | no | A short display string used as this tool's icon on its call line. Overrides the auto-derived category icon (see below). Give it an ASCII-safe or well-supported symbol; omit to use the category default. |

### `safe` also means concurrency-eligible

When the model emits a batch of tool calls in one turn and **every** call is
read-only, lean-coder runs them **concurrently on threads** (local only; a
`/connect`ed session stays serial). A `safe: True` lean-tool is part of that set, so
marking a lean-tool safe asserts more than "no prompt":

- **No side effects** - no writes, no shared-state mutation, no commands.
- **Thread-safe / reentrant** - it may run at the same instant as siblings (or
  another instance of itself). No mutable module globals, no shared handles.
- **Return, don't print** - concurrent stdout interleaves; the model only sees
  the return value anyway.

If you can't guarantee all three, **omit `safe`**: a non-safe lean-tool always runs
serially through the confirm gate, which is the correct default for anything with
state or I/O beyond a pure read.

### Call-line icons (glyphs)

Every tool call prints a dim one-line summary led by an **icon**. lean-coder picks it
automatically by **category**, so your lean-tool gets a sensible icon with no effort:

| Category | Icon | Chosen when |
|----------|------|-------------|
| read  | `◎` | your tool is `safe: True` (read-only) |
| write | `✎` | (built-in writers) |
| exec  | `»` | (built-in `run_command`) / non-safe by default |
| net   | `⇅` | the tool name is a known network tool (`web_fetch`, `ssh`, ...) |
| meta  | `◇` | agent-internal (`update_plan`, `note`, ...) |
| gear  | `⚙` | anything uncategorised (the fallback) |

A **backgrounded** `run_command` (trailing `&`) always shows `⚡` regardless.

To give your tool a **signature icon**, set `TOOL["glyph"]` - it wins over the derived
category:

```python
TOOL = {
    "name": "weather",
    "description": "Current weather for a city.",
    "glyph": "☂",            # shown on the call line instead of the category icon
    "parameters": {...},
    "safe": True,
}
```

Pick a symbol your users' terminals can render. For output *inside* your tool's return
value, the helper **`lc.g(unicode, ascii_fallback)`** and the **`lc.GLYPH`** dict are
available via the `setup(lc, cfg)` hook (below) - use them so a non-UTF terminal gets a
clean ASCII fallback instead of mojibake:

```python
def setup(lc, cfg):
    global _tick
    _tick = lc.g("✓", "+")   # ✓ on UTF-8 terminals, + otherwise
```

### Built-in tools (`lean-tools/builtins.py`)

The always-on tools the model gets for free - `read_file`, `list_files`,
`search_files`, `apply_diff`, `write_file`, `run_command` - are themselves a
**bundled lean-tool module**, [`lean-tools/builtins.py`](lean-tools/builtins.py), that
ships with the code (like `providers/`). They are loaded at startup and can't be
toggled off, and the core derives its `/leash` ceilings from them.

You write user lean-tools with the binary `safe` flag above; the builtins use a finer
internal field, **`tier`: `"read" | "write" | "exec"`**, because they span all three
(reads, edits, and command execution). The core maps `safe: True` user tools onto the
`read` tier. You don't set `tier` in your own lean-tool - it's the builtins' way of
declaring the three-rung leash; `safe` is the user-tool equivalent. Read
`builtins.py` if you want a worked example of the same `setup(lc, cfg)` inject seam.

### `run(args, cwd)`
- `args` is the dict the model passed, matching `parameters`.
- `cwd` is the working directory: the local cwd, or the **remote** working dir
  when you are connected with `/connect`. Always resolve paths against `cwd` -
  never hardcode an absolute path, or your lean-tool breaks when run remotely.
- Return a **string**. Anything non-string is `str()`-ed. Very long output is
  truncated by the host, so summarise where you can.
- Errors: just `return "error: ..."`, or raise - exceptions are caught and
  returned to the model as `error in lean-tool <name>: ...`. Your lean-tool can never
  crash lean-coder.
- **Return, do not print** to **stdout**. The model sees the return value; on a
  `/connect` remote, the executor's **stdout is the JSON protocol channel**, so a
  stray `print()` there corrupts it. For an out-of-band note to the *user* (a
  warning, a progress line), write to **stderr** - safe locally and remote.

### Sizing output to the context window

A big return value can blow a small or nearly-full context window - on a weak model
this evicts its own history and it starts hallucinating. If your lean-tool can emit a
lot (a web page, a big file, a long query result), **size the output to the space
available.** Before each dispatch the core publishes the live budget in two env vars:

| env var | meaning |
|---------|---------|
| `LEANCODER_CTX_MAX`  | the active model's context window, in tokens |
| `LEANCODER_CTX_USED` | tokens already in use this turn |

`free = MAX - USED`. A read-heavy tool should cap its output to a fraction of `free`
(a useful rule of thumb: `free × 0.4 × ~3 chars/token`), and page the rest behind a
`start=`/offset argument so the model can ask for more deliberately. When the budget
is so low you've clamped to a floor, write a one-line stderr notice (`/compact for
fuller results`). Both vars are **unset** in tests and on a remote executor - treat
unset as "no limit / fall back to your hard ceiling". See `web_fetch.py` for a
worked implementation (`free_tokens()` / `char_budget()`).

## Startup hooks: `setup(lc, cfg)` and slash commands

A `TOOL`+`run` lean-tool adds a tool the *model* calls. To extend lean-coder
itself - add a slash command, set a config default, or swap the HTTP client for
a different backend - define `setup`:

```python
def setup(lc, cfg):                 # called once at startup, on the driver only
    cfg.greeting = "hi"             # cfg is live: set defaults / read config
    lc["register_command"]("/whoami", _handler, "show session info")
```

- `lc` is lean-coder's module globals: read or replace anything in it
  (`lc["Agent"]`, `lc["load_config"]`, ...). `cfg` is the live `Config`.
- `register_command(name, handler, help_line, completer=None)` adds a real slash
  command. `handler(agent, cfg, arg)` runs when the user types it; `arg` is the rest
  of the line. Built-in names are refused (a lean-tool can never shadow `/model`
  etc.); if two different plugins register the SAME custom name, the first wins and the
  second is warned + ignored (re-registering your own command still replaces it). The
  name joins Tab-completion and `/help`. Optional
  `completer(agent, cfg) -> [str]` supplies Tab-completions for the command's first
  argument (e.g. a list of known hosts).
- A lean-tool may define `setup` **instead of** or **alongside** `TOOL`+`run`. A
  setup-only lean-tool adds nothing to the model's tools.

See the example [`examples/lean-tools/session_info.py`](examples/lean-tools/session_info.py)
for a `/whoami` command built this way,
[`lean-tools/notify.py`](lean-tools/notify.py) for a driver-only
lean-tool (no model tool: it hooks the turn/confirm flow to ring the bell, so it is
deliberately setup-only rather than a pushed TOOL),
[`lean-tools/provision.py`](lean-tools/provision.py) for a
`/provision` wizard (multi-select pickers, reusing lean-coder's SSH helpers and
`install.sh` to set up another box), and
[`lean-tools/update.py`](lean-tools/update.py) for a `/update`
command that self-updates `lean_coder.py` from the published build.

### Interactive pickers (reusable menu API)

A `setup()` lean-tool (or any feature) can offer the SAME native-feeling menu the
built-ins use - arrow-keys, type-to-filter, number-type-then-Enter to jump-select,
a scrolling viewport that never wraps/smears, and an automatic numbered-prompt
fallback on a dumb/no-tty terminal. Reach them through `lc` in `setup(lc, cfg)`:

```python
def setup(lc, cfg):
    def _handler(agent, cfg, arg):
        colour = lc["pick_one"]("pick a colour:", ["red", "green", "blue"],
                                current="green")
        if colour is None:
            return                      # user cancelled (esc / q / 0)
        print(f"you chose {colour}")
    lc["register_command"]("/colour", _handler, "pick a colour")
```

The public picker API (all in `lc`):

- **`pick_one(header, choices, current=None, labels=None)`** - single-select. Returns
  the chosen value from `choices`, or `None` on cancel. `current` is marked; `labels`
  optionally supplies pre-styled display strings parallel to `choices` (e.g. a green
  dot) while selection still keys off `choices`.
- **`pick_grouped(header, lines, index_of, num_of)`** - the grouped variant (section
  headers + selectable items), as `/model` uses across providers.
- **`run_picker(render, on_key)`**, **`read_key(stdin)`**, **`picker_capable()`** - the
  low-level engine, if you need a bespoke menu: `picker_capable()` gates rich vs. the
  numbered fallback, `run_picker` drives the scrolling render+input loop, `read_key`
  normalises one keypress to a token. Most tools only need `pick_one`.

A user can force the numbered fallback on any menu command with a leading `--plain`
(e.g. `/colour --plain`) - handled centrally, so your handler needs no extra code.

### Adding a model backend? That's a provider, not a lean-tool

Talking to a different model backend (a hosted API, an OAuth subscription,
anything non-Ollama) is a **provider**, which is its own plugin type - a `.py` file
with a `PROVIDER` dict in `providers/`, not a lean-tool. It never ships to a
`/connect` remote and exposes no model tool. See **[PROVIDER_API.md](PROVIDER_API.md)**
for the full contract, and [`examples/providers/example.py`](examples/providers/example.py)
for a copy-and-edit template.

Rules that matter:

- **Driver-only.** `setup` never runs in the remote executor - it patches the
  thing you interact with. Only `TOOL` lean-tools are pushed to a remote; a
  setup-only lean-tool stays local.
- **Startup-only.** `setup` runs once, before the session starts, and is **not**
  re-run by `/reload` (it may monkeypatch, which is not safe to repeat). Tool
  lean-tools still hot-reload; changing a `setup` hook needs a restart.
- **Enable like any lean-tool.** A setup lean-tool is gated by `/tools` the same way
  (a setup-only lean-tool is listed by its filename). Disabled = `setup` never runs.
- **Contained failures.** A `setup` that raises is caught and warned; startup
  continues. A command handler that raises is caught; the session survives. A
  throw mid-`setup` silently strips any `register_command` calls after it, so set
  `LEANCODER_DEBUG=1` to print the full traceback when a command goes missing.

## Installing and enabling

1. The bundled lean-tools in [`lean-tools/`](lean-tools/) are discovered
   automatically - just `/tools`-enable them. To add your own, drop a `.py` in
   `~/.config/leancoder/lean-tools/` (or set `lean_tools_dir` in the config to point
   elsewhere):
   ```bash
   mkdir -p ~/.config/leancoder/lean-tools
   cp my_tool.py ~/.config/leancoder/lean-tools/
   ```
2. Start lean-coder and run **`/tools`**: up/down to move, **space** to toggle,
   enter to save, q to cancel.
3. Lean-tools are **disabled by default** (zero context cost until you opt in). The
   enabled set is saved to `lean_tools_enabled` in the config and shown in the
   startup banner.

## Grouping by subdirectory

Tools can be organised into groups by putting them in subdirectories of
`lean_tools_dir`, e.g. `lean-tools/web_dev/`, `lean-tools/db/`. A tool's group is
its first subdirectory under the root; files directly in the root are the
ungrouped **general** group. In `/tools`, each group gets a header row whose own
**space** toggles the entire group at once (`[x]` all on, `[-]` partial, `[ ]`
off) - so you can switch on a whole `web_dev` set when you move to a more capable
model or a specific task, then switch it back off. Grouping is purely a UI
convenience: enabling is still per tool **name** (`lean_tools_enabled` and the
remote sync are unchanged), and tool names must be unique across groups. With no
subdirectories present, `/tools` stays the flat list it always was.

## Local and remote

Enabled lean-tools are surfaced identically whether you are local or connected to a
remote workspace. On `/connect`, the enabled lean-tool files are synced to the
remote executor and run there, so the model uses them the same way. Because of
that:

- Resolve paths against `cwd` (it is the remote dir when connected).
- Any third-party imports your lean-tool uses must be available **on whichever
  machine runs it** - the lean-coder core needs only `python3`, but your
  lean-tool's dependencies are your responsibility on both ends. Prefer the stdlib.
- The sync is **by content hash**: only changed/new files are sent, and a file
  you've since disabled is removed from the remote - so a tool is identified by
  its filename. Keep filenames stable across edits.
- After toggling `/tools` while already connected, run `/reload` to sync the
  change to the remote (it relaunches the executor over the same SSH session, no
  reconnect or re-auth).

## Worker agents (`dispatch_worker`)

The bundled `dispatch_worker` lean-tool lets the model hand a self-contained
sub-task to a **background worker agent** - its own session, own context, own model
choice - and collect the result later. Useful for scoped errands whose raw output
would bloat the main session ("scout this large module and report where X lives",
"run the tests and summarise the failures").

How it works (it piggybacks the background-task machinery, no new transport):

- The model calls `dispatch_worker(task=..., [model=...], [provider=...], [cwd=...],
  [leash=...], [host=...])` to launch one. The same tool also manages workers via
  `action`: `action='status'` reports dispatched workers, `action='cancel'` (with
  `pid=...`) kills one; the default `action='dispatch'` launches from `task`.
- The tool writes a **brief file** (the task + a grant header) and launches
  `lean_coder --agent-run` as a detached background task tagged `kind="worker"`,
  with a **lease** (it self-terminates if this session stops attending it, so a
  dropped parent can't leave a worker burning quota).
- The worker runs the one brief on **this box** (the driver's provider + auth - no
  per-box install or login), writes its answer between `===RESULT===` markers, and
  exits. Its file/command tools ride the same executor path this session uses.
- On finish you get a notice injected into the next turn (and a desktop ping if the
  `notify` lean-tool is also enabled); when *all* dispatched workers are done, an
  "all N finished" line too.

Collecting results: `/worker` lists dispatched workers with state + whether a result
is ready; `/worker <pid>` prints that worker's full result. Workers are deliberately
hidden from `/bg` and the `bg_status` tool, so a session without this tool never sees
them.


## Autonomous wake on background finish (`wake_on_bg_finish`)

By default a finished background job (a plain `' &'` task **or** a dispatched worker)
surfaces **passively**: its notice rides the *next turn you type*. If you never type,
the agent never sees it. That's fine when you're driving, but it means a long job
that finishes while you're away just waits.

Turn on **`wake_on_bg_finish`** (config, or `/set wake_on_bg_finish
true`) and a finish **wakes the agent itself** - it synthesises a turn carrying the
result and an instruction to react, with **no operator input**. The agent inspects
the result, decides the next step, and either acts or hands back. Off by default: an
idle-wake loop changes REPL semantics and can burn quota unattended, so it's opt-in.

How it works:

- **Idle poll.** While waiting at the prompt, the input path polls ~4×/s. The
  composer (`Composer.read_line`) and the classic no-composer path
  (`_input_or_wake`, a `select()` on stdin) both check for a finished job each tick;
  a hit returns a synthesised line exactly as if you'd typed it.
- **One source, both kinds.** `Agent.bg_wake_turn()` aggregates plain bg tasks (via
  the same `_bg_finished_here` + `_bg_announced` dedup the passive notice uses) and
  workers (via the `_WAKE_HOOKS` registry - `dispatch_worker` registers its
  `_finished_notice`, whose own `announced` dedup applies). A job is claimed by
  whichever fires first (wake when idle, or the passive turn-rider if you type
  first), so it is **never double-reported**.
- **Where it works.** Any real terminal - a plain TTY, tmux/screen, SSH, Termux -
  and even a non-composer session with live (still-open) stdin. It does **not** wake
  in a headless one-shot (piped-then-EOF stdin, cron) because there is no idle loop
  to wake into; a fire-and-forget `--serve` daemon mode for that is on the backlog.
- **Desktop ping** still comes from the `notify` lean-tool if enabled (workers ping
  on finish); wake is the turn-level reaction, `notify` is the OS-level nudge - they
  compose.

**Capability (leash).** A worker defaults to **read-only**. Pass `leash="rw"` (may
edit files) or `leash="rwe"` (may edit + run commands), but the grant is always
capped at **your own current leash** - a worker can never exceed its parent's
authority.

**Operator ceiling** (the cost lever) - each an env var, set once:

| env var | default | meaning |
|---|---|---|
| `LEANCODER_WORKER_MAX_CONCURRENT` | 10 | most workers alive at once |
| `LEANCODER_WORKER_IDLE_TIMEOUT` | 1800 | lease seconds; unattended worker self-terminates |
| `LEANCODER_WORKER_MAX_ITER` | 30 | agentic-loop cap per worker |
| `LEANCODER_WORKER_MODELS` | *(any)* | comma-list allowlist of dispatchable models |

`dispatch_worker` is a **`driver_only`** tool (`TOOL["driver_only"] = True`): it
spawns the worker process on the driver, so it is never pushed to or run on a remote
executor. Set that flag on any tool that must act on the local machine even when the
session is connected to a remote.

> Note: a worker's *brain* always runs on the local (driver) side. Running the brain
> natively on a remote box is a possible future extension (it would require a
> provisioned, authed lean-coder there); for now the worker's tools reach a remote via
> the executor while inference stays local.

## Context discipline

Every enabled lean-tool's schema is in every request. Keep `description` to one
line and `parameters` minimal, and only enable the lean-tools you are actually
using. An empty/disabled lean-tool set costs nothing.

## Gotchas (read before you debug)

- **`setup()` state does NOT reach `run()`.** Each load `exec`s a *fresh* module, and
  `setup()` runs on a different load than the one that dispatches `run()`. So a module
  global you set in `setup()` is `None` inside `run()`. `setup()` is for mutating the
  **core** (`register_command`, monkeypatching `Agent`), not for initialising your
  tool's own state. To pass live data from the core into `run()`, use the env-var
  channel (see *Sizing output*), not a module global.
- **`setup()` is startup-only and driver-only.** It does not re-run on `/reload`
  (restart to pick up `setup` changes), and it never runs in the remote executor.
- **Don't keep mutable module globals in a `safe` tool.** `safe` tools can run
  concurrently on threads; shared state races. Keep `run()` pure/reentrant or omit
  `safe`.
- **Resolve every path against `cwd`.** A hardcoded absolute path breaks the moment
  the tool runs on a `/connect` remote.
- **Identity is the `TOOL["name"]`** (or the filename for a setup-only tool), *not*
  the filename for tool lean-tools - `lean_tools_enabled` and `/tools` list that name.
  Two tools with the same `name` collide; the second is skipped with a warning.
- **A missing slash command** usually means a `setup()` raised before its
  `register_command` line. Run with `LEANCODER_DEBUG=1` to see the traceback.

## Testing your lean-tool

A lean-tool is plain Python, so test it without launching lean-coder:

```python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("t", "my_tool.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

assert mod.TOOL["name"] == "my_tool"
assert mod.run({"path": "README.md"}, ".").startswith("...")     # happy path
assert mod.run({}, ".").startswith("error")                       # missing arg
```

Favour **pure helper functions** (parse, format, budget) that you can assert on
directly, with `run()` a thin shell over them - that's how the bundled tools keep
their logic unit-testable with no network or filesystem. Test the error paths: a
good lean-tool returns a clear `error: ...` string for every bad input rather than
raising.

## Quality checklist

Before you ship a lean-tool:

- [ ] `description` is **one line**; `parameters` only lists args `run()` actually reads.
- [ ] `name` is unique and doesn't shadow a built-in.
- [ ] `safe` set **only** if read-only, side-effect-free, and thread-safe.
- [ ] All paths resolved against `cwd`; nothing hardcoded.
- [ ] Every bad input returns `error: ...` (no uncaught raise for expected cases).
- [ ] Returns a **string**; large output is sized to the context window and pageable.
- [ ] No stdout prints; user-facing notes go to **stderr**.
- [ ] Stdlib-only if you can; any third-party import is installed on **both** ends.
- [ ] Secrets/caches live in `~/.config/leancoder/` (`chmod 600` for secrets), never
      in `config.toml` (it autosaves / gets rewritten on any config change).
- [ ] Tested: happy path + error paths, ideally via pure helpers.
