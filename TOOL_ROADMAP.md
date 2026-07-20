# Tool roadmap

Candidate capabilities for lean-coder, ranked by value-for-leanness: how much
they help a capable model against their context cost and complexity. The bar for
adding anything is the project's: every enabled tool's schema rides in *every*
request, so the default answer is "no" unless the payoff is clear. Most of these
are lean-tools (opt-in, zero cost until enabled); only #1 is a core-loop change.

Status legend: **done** · **planned** · **deferred**.

## 1. Parallel tool calls - *done*

Run a batch of read-only tool calls concurrently instead of one-at-a-time, so
"read these four files" costs one wait, not four.

- **Why:** the model already emits multiple `tool_calls` in one turn; the loop
  used to run them serially. Read-heavy exploration is the common case for a
  strong model, and it was paying sum-of-latencies for no reason.
- **How:** `Agent._parallel_safe` classifies a call as concurrency-safe (core
  read tools, or a `safe:True` lean-tool). `Agent._run_parallel` runs the batch on
  daemon threads and appends results **in call order**. The loop only takes this
  path when `len(calls) > 1 and not remote and all(parallel_safe)`.
- **Constraints that shaped it** (each killed a bug class up front):
  1. results pair to calls positionally -> preserve append order;
  2. writers / `run_command` / `ask_user_to_run` need an interactive confirm and
     can race -> the *whole* batch must be read-only or it stays serial;
  3. the remote executor is a single serial request/response pipe -> a
     `/connect`ed session is always sequential.
- **Honest scope:** this is a throughput win for *local* read-heavy turns. It
  does nothing for remote (see constraint 3). No new tool surface, no context cost.

## 2. `git_summary` - *done* (lean-tool, `safe:True`)

One read-only tool returning a pre-digested repo snapshot: branch, status,
staged/unstaged diffstat, and the last few commits.

- **Why:** git is reached for constantly via `run_command`; this collapses 3-4
  shell round-trips into one structured result, and being read-only it never
  prompts. Cheap, high-frequency payoff.
- **Cost:** one schema line, pure stdlib (`subprocess` to `git`). Returns a
  string; truncate long diffstats.
- **Teaches:** the lean-tool contract and the read-only/context-cost tradeoff.

## 3. Semantic navigation - *partly done* (`symbols.py`, `safe:True`); full LSP deferred

Go-to-definition and file outline, sibling to `diagnostics.py`.

- **Shipped (`symbols.py`):** a dependency-free, Python-only navigator - `outline`
  (structure of a file) and `def` (locate a definition by name), built on the stdlib
  `ast`, so it fits the zero-dep ethos and ships bundled (toggle with `/tools`).
- **Still deferred (multi-language LSP):** find-references/rename and non-Python
  languages need a real language server (JSON-RPC over a subprocess) or tree-sitter -
  the highest build cost here, and the value only shows up on big/polyglot repos.
  Add unless zero-dep Python outline+def already covers the target.

## 4. `notify` - *done* (setup-hook lean-tool, not a model tool)

Ping the operator when a long task finishes or needs input.

- **Why:** quality-of-life for autonomy on long runs.
- **Wrinkle:** it must fire on the **driver** (the machine you're sitting at),
  not the remote executor - so it's a `setup()` hook / slash command, *not* a
  pushed `TOOL`. This is the executor-vs-driver distinction in miniature.

## 5. Sub-agent / delegate - *done* (`dispatch_worker`, never `safe:True`)

Spawn a scoped background child that does a fan-out sub-task and returns only the
conclusion, keeping the driver's context clean.

- **Shipped (`dispatch_worker`):** hands a self-contained task to a background worker
  agent with its own session + context; the parent is notified on completion and pulls
  the (optionally full) result, so a long errand's intermediate output never bloats the
  driver. Own capability leash (`r`/`rw`/`rwe`, capped at the parent's), optional
  separate model/host, and mid-task `inject`/`cancel`. This is the context-economy win
  the entry called for - the machinery earns its keep by *removing* tokens from the
  driver, in line with the lean philosophy rather than against it.

## 6. `web_fetch` - *done* (lean-tool, never `safe:True`); `http_request` deferred

Fetch a URL (read docs, hit a local dev server / API).

- **Shipped (`web_fetch`):** GET an http(s) URL; never `safe` so every call shows
  the URL and confirms; off by default. The control is the confirm gate, **not**
  an IP blocklist - hitting `localhost`/LAN dev servers is a primary use, so it
  is intentionally not blocked (a single-user local tool; the visible-URL confirm
  is the check against an injected/unintended fetch). GET-only, size-capped,
  timed out.
- **Still deferred (`http_request`):** arbitrary method/body is the riskier
  sibling (a POST body can exfiltrate). Add only on explicit demand.

## Recommended cut

Most of this list has shipped: #1 (parallel calls), #2 (`git_summary`), #4 (`notify`),
#5 (`dispatch_worker`), and #6 (`web_fetch`) are all done, plus a zero-dep Python slice
of #3 (`symbols`). What remains genuinely deferred: **multi-language LSP** (#3, the
build-cost item, worth it only on big/polyglot repos) and **`http_request`** (#6's
riskier method/body sibling - on explicit demand).
