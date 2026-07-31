"""dispatch_worker - let the model hand a scoped sub-task to a headless WORKER
agent that runs in the background, then collect its result later.

WHY: some work is a self-contained errand - "read this large module and tell me
where X is handled", "run the test suite and summarise the failures". Doing it
inline burns the main session's context on raw output. A worker does it in its
OWN session (own context, own model choice) and hands back only the answer.

HOW (piggybacks the background-task machinery, adds no new transport):
  - the model calls dispatch_worker(task=..., [model=...], [cwd=...]);
  - we write a BRIEF file (the task + a GRANT header: leash, model, cwd, limits);
  - we bg_launch `lean_coder --agent-run --brief-file B --result-file R` as a
    detached task tagged kind="worker", with a lease (self-terminates if this
    session dies / stops attending - caps runaway quota);
  - the worker runs one brief on THIS box (the driver's provider + auth - no
    per-box install/oauth), writes its answer between ===RESULT=== markers to R,
    exits. Its file/command tools ride the same executor path this session uses.
  - the finished worker is surfaced via /worker (workers are deliberately hidden
    from /bg + the bg_status tool, so a session without this tool never sees them).

LEASH: the worker's grant is set by the driver (the main AI) via the `leash` arg
and capped at the driver's OWN live leash - MIN(requested, parent), the leash
principle: a grant is never above the grantor's authority (see _capped_leash).
So read/write/exec workers are ALL supported; the default is 'r' (a scout), and a
write/exec worker inherits approval=auto headless (nobody's there to confirm), so
grant rw/rwe deliberately. This is a driver_only tool: it spawns the worker process
on the driver, so it is never pushed to / run on a remote executor.

WORKER CEILINGS (the cost lever) - each resolves in priority order:
  1. the env var (operator LOCKDOWN - still wins, a running session can't move it);
  2. else the live cfg value (editable in /settings, saved to config, usable at once);
  3. else the built-in default (shown below).
  cfg field              /settings + config      env var                          default
  worker_max_concurrent  most workers at once     LEANCODER_WORKER_MAX_CONCURRENT   10
  worker_idle_timeout    lease secs; self-kill    LEANCODER_WORKER_IDLE_TIMEOUT     1800
  worker_max_iterations  agentic-loop cap/worker  LEANCODER_WORKER_MAX_ITER         30
  model_allowlist        (env only)               LEANCODER_WORKER_MODELS           any

RESUME: with cfg.worker_checkpoint on, a worker dumps its transcript to a
'<brief>.checkpoint' sidecar each iteration; action='resume' (pid=, text=) relaunches a
DEAD/timed-out/incomplete worker from that checkpoint (a NEW pid - resume is a fresh OS
process; lineage kept in meta['resumed_from']). Off by default = no checkpoint file.
"""
import json
import os
import shlex
import sys
import time
from pathlib import Path

# Built-in ceiling defaults (last resort: env var > live cfg > this - see docstring).
_DEFAULTS = {"max_concurrent": 10, "idle_timeout": 1800, "max_iterations": 30,
             "max_depth": 1, "max_children": 0}
_ENV = {"max_concurrent": "LEANCODER_WORKER_MAX_CONCURRENT",
        "idle_timeout": "LEANCODER_WORKER_IDLE_TIMEOUT",
        "max_iterations": "LEANCODER_WORKER_MAX_ITER",
        "max_depth": "LEANCODER_WORKER_MAX_DEPTH",
        "max_children": "LEANCODER_WORKER_MAX_CHILDREN"}

TOOL = {
    "name": "dispatch_worker",
    "glyph": "\u2691",   # flag: a dispatched sub-agent (distinct from bg's bolt; bigger blast radius - own context/leash, tracked via /worker)
    "description": (
        "Hand a self-contained sub-task to a background worker agent (own session + "
        "context); returns a pid. It runs in the BACKGROUND and you're NOTIFIED "
        "AUTOMATICALLY when it finishes (a notice on a later turn) - don't poll it, just "
        "carry on or end your turn. For scoped errands whose output would bloat this "
        "session, e.g. 'scout this module, report where X is handled'. Default READ-ONLY; "
        "leash='rw'/'rwe' to let it edit/run, never above YOUR capability. Manage/steer a "
        "worker (and list models, resume a dead one) via `action`."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["dispatch", "models", "status", "result", "transcript",
                                "cancel", "pause", "pause_all", "stop_all", "resume",
                                "resume_all", "inject", "set_plan", "add_note", "promote",
                                "board_claim", "board_release", "board_list"],
                       "default": "dispatch",
                       "description": "Default 'dispatch' (launch from `task`). 'models'=list "
                                      "models you can dispatch on. 'status' [pid]=worker state/"
                                      "runtime/ready. 'result' (pid=)=FULL untruncated result "
                                      "(the finish notice truncates a long one). 'transcript' "
                                      "(pid=, [page=])=read the worker's OWN full reasoning/tool "
                                      "trail, paged (needs worker_checkpoint on). 'cancel' "
                                      "(pid=)=kill+discard it. FLEET lifecycle: 'pause' (pid=)="
                                      "kill but KEEP its checkpoint so 'resume' (pid=) can "
                                      "relaunch it later (also relaunches a DIED/capped worker); "
                                      "'pause_all'/'stop_all'/'resume_all' ([taskboard=])=do it to "
                                      "every worker (pause=resumable, stop=discarded), optionally "
                                      "just one board's team. Steer a RUNNING worker: 'inject' "
                                      "(text=) a correction, 'set_plan' (plan=) replace its plan "
                                      "(omit plan to READ it first), 'add_note' (notes=). "
                                      "'board_claim'/'board_release' (text=<path>)/'board_list'="
                                      "shared-file mutex so peer workers don't clash (no-op for a "
                                      "lone worker)."},
            "pid": {"type": "integer",
                    "description": "Worker to act on (required for 'cancel'; optional for 'status' "
                                   "to show one). From the dispatch return line."},
            "task": {"type": "string",
                     "description": "Full self-contained instruction: what to do and exactly "
                                    "what to report back (the worker has no other context). "
                                    "Required for action='dispatch'."},
            "text": {"type": "string",
                     "description": "For 'inject' (pid=): a mid-task correction/steer the running "
                                    "worker sees on its next reasoning step (does NOT interrupt a "
                                    "running command). Also the fresh steer for 'resume'."},
            "model": {"type": "string",
                      "description": "Optional worker brain - ANY model on an enabled provider "
                                     "(see action='models'); its provider is inferred. Default: "
                                     "inherit your current model."},
            "provider": {"type": "string",
                         "description": "Optional backend for the worker's brain (must be enabled). "
                                        "Only needed to disambiguate when the same model name is on "
                                        "more than one provider; else inferred from `model`."},
            "brain_host": {"type": "string",
                           "description": "Optional ollama endpoint (URL or /machines alias) to run "
                                          "the worker's BRAIN on - distinct from `host` (its TOOLS). "
                                          "Runs the model on a different box (ollama only)."},
            "host": {"type": "string",
                     "description": "Optional box to run the worker's TOOLS on ([user@]host or a "
                                    "/connect name); defaults to this session's target. A password "
                                    "host is prompted ONCE here - a worker can't answer prompts."},
            "cwd": {"type": "string",
                    "description": "Optional working directory (defaults to current)."},
            "from_session": {"type": "string",
                             "description": "Optional: seed this worker from a saved SESSION (name "
                                            "under CONFIG_DIR/sessions, or a path). Its transcript "
                                            "becomes the worker's history and `task` runs as a steer "
                                            "turn on top; checkpointing is forced on so it can be "
                                            "promoted back to a session later."},
            "name": {"type": "string",
                     "description": "For action='promote' (pid=): the session name to save the "
                                    "worker's transcript under (loadable with /load)."},
            "taskboard": {"type": "string",
                          "description": "Optional: assign this worker to a named task-board (see "
                                         "the `board` tool). The worker auto-gets the board tool and "
                                         "is told to report its task done there. Usually add+assign "
                                         "the task first, then dispatch with taskboard=<name>."},
            "iterations": {"type": "integer",
                           "description": "Optional tool-call budget, capped at the operator ceiling "
                                          "(omit = ceiling). On 'dispatch' sets the worker's cap; on "
                                          "'resume' grants a FRESH budget (use it when the worker "
                                          "died by hitting its cap)."},
            "leash": {"type": "string", "enum": ["r", "rw", "rwe"], "default": "r",
                      "description": "Worker capability: r=read-only (default), rw=edit, rwe=edit+run. "
                                     "Capped at your own leash."},
            "tools": {"type": "array", "items": {"type": "string"},
                      "description": "Optional ALLOWLIST of tool names the worker may use (e.g. "
                                     "[\"read_file\",\"web_fetch\"] for a scout). Omit = your full "
                                     "leash-permitted set. A worker never gets a tool you lack, and "
                                     "the grant is still leash-capped; its plan/note meta tools are "
                                     "always kept."},
            "context": {"type": "string",
                        "description": "Optional CURATED background the worker needs but that isn't "
                                       "the task itself (e.g. 'auth was just refactored; tokens now "
                                       "live in x'). Share the STATE it needs, not your whole "
                                       "transcript (not truncated)."},
            "plan": {"type": "string",
                     "description": "Optional starting plan (GOAL + '- [ ]' TODO) seeding the "
                                    "worker's pinned plan. Also the payload for 'set_plan'."},
            "notes": {"type": "string",
                      "description": "Optional seed notes for the worker's notebook, one per line "
                                     "(tagged as from you). Also the payload for 'add_note'."},
            "page": {"type": "integer",
                     "description": "For 'transcript' (pid=): which page of the worker's trail to "
                                    "show (1-based, default 1). The reply tells you the page count "
                                    "and whether more remain."},
        },
        "required": [],
    },
    "driver_only": True,   # spawns the worker process on the driver; never pushed remotely
    "no_timeout": True,    # launches/awaits a background worker agent; long-by-design
    # NOT safe: dispatching a worker spends quota + runs commands, so it confirms
    # unless approval is auto/armed (same policy as any non-read tool).
}

# Captured from core in setup(). A tool's run() doesn't get lc, so we stash the
# hooks + helpers here at startup (setup runs driver-only, which is where a worker
# is launched - so these are always present when run() fires).
_H = {}


def _ceiling(key):
    """An integer worker ceiling, resolved in priority order:
      1. the env var (LEANCODER_WORKER_*) if set + valid - an operator LOCKDOWN that
         a running session can't move;
      2. else the live cfg value (worker_<key>), editable in /settings + persisted to
         config - the user's own knob, usable straight away;
      3. else the built-in default.
    So users get simple, persistent control via /settings, while an env var still wins
    as a hard ceiling when the operator wants one."""
    env = os.environ.get(_ENV.get(key, ""), "")
    if env.strip():
        try:
            return int(env)
        except ValueError:
            pass
    cfg = _H.get("cfg")
    val = getattr(cfg, f"worker_{key}", None) if cfg else None
    if isinstance(val, int):
        return val
    return _DEFAULTS[key]


def _governor_check(my_depth):
    """Safe-recursion gate for the process trying to spawn a worker. `my_depth` is this
    process's depth in the worker tree (driver = 0, its workers = 1, ...). Returns
    (ok, error_str, granted_child_budget):
      - the DRIVER (depth 0) may always spawn (only max_concurrent bounds it); the child
        it makes is depth 1 and gets a child budget of worker_max_children.
      - a WORKER (depth >= 1) may spawn ONLY IF depth < worker_max_depth AND it still has
        child budget left (granted by its parent, decremented per spawn). The child it
        makes is depth my_depth+1 and inherits a SHRINKING budget (its own remaining
        minus the one it's spending), so a subtree is bounded by depth x count, never a
        fork-bomb. Both gates REFUSE loudly rather than silently succeed."""
    cfg = _H.get("cfg")
    max_depth = _ceiling("max_depth")
    if my_depth <= 0:
        # The driver: ungated here (max_concurrent is its bound). Its children are depth 1
        # and get the operator's per-worker child allowance.
        return True, "", _ceiling("max_children")
    if my_depth >= max_depth:
        return (False, (f"error: recursion depth limit reached (this worker is at depth "
                        f"{my_depth}, worker_max_depth is {max_depth}). A worker at the max "
                        f"depth cannot spawn its own workers. Raise worker_max_depth / "
                        f"LEANCODER_WORKER_MAX_DEPTH to allow deeper recursion."), 0)
    remaining = int(getattr(cfg, "worker_child_budget", 0) or 0)
    if remaining <= 0:
        return (False, ("error: this worker was granted no child budget (worker_max_children "
                        "is 0, or its budget is spent), so it cannot spawn workers. This is the "
                        "safe default - recursion is opt-in. Raise worker_max_children / "
                        "LEANCODER_WORKER_MAX_CHILDREN to grant one."), 0)
    # Spend one from this worker's budget; the child inherits what's left after this spawn.
    cfg.worker_child_budget = remaining - 1
    return True, "", remaining - 1

def _model_allowlist():
    """The model allowlist (LEANCODER_WORKER_MODELS, comma-separated), or None = any."""
    raw = os.environ.get("LEANCODER_WORKER_MODELS", "").strip()
    if not raw:
        return None
    return [m.strip() for m in raw.split(",") if m.strip()]


def _capped_leash(requested):
    """The worker's granted leash = MIN(requested, the parent's LIVE leash). A worker
    can NEVER exceed its parent's authority (the leash principle: a grant is <= the
    grantor's own). requested defaults to 'r' (read-only). Junk -> 'r'. Reads the
    parent's leash off the live cfg captured in setup(), so a runtime /leash change is
    respected. Returns (granted_leash, was_downgraded)."""
    levels = _H["LEASH_LEVELS"]                       # ("chat","r","rw","rwe")
    req = _H["_norm_leash"](requested) or "r"
    if req == "chat":                                 # a worker with no tools is pointless
        req = "r"
    parent = getattr(_H.get("cfg"), "leash", "rwe") or "rwe"
    granted = req if levels.index(req) <= levels.index(parent) else parent
    return granted, (granted != req)


def _workers_dir():
    return Path(_H["CONFIG_DIR"]) / "workers"


def _compose_brief(task, model, cwd, max_iter, leash="r", provider="", brain_host="",
                   tools="", context="", plan="", notes="", depth=0, child_budget=0,
                   checkpoint=False, resume="", board=0, taskboard=""):
    """Build the brief file text: the task wrapped in BRIEF markers + a GRANT header.
    `leash` is the already-capped grant (see _capped_leash). `provider` (optional) is
    the backend the worker must activate for `model`; absent = inherit the driver's.
    `brain_host` (optional) is an ollama inference endpoint for the worker's BRAIN,
    distinct from where its tools run; absent = the driver's own host. `tools`
    (optional) is a comma-separated allowlist of tool names; absent = full toolset.
    `context`/`plan`/`notes` (optional) seed the worker with curated STATE - a
    background blob, a starting plan, notebook entries - each emitted as its own marker
    block the worker parses in run_agent_brief. Not truncated: the parent decides how
    much state the job needs. `depth` is this worker's depth in the tree (driver->worker
    = 1); `child_budget` is how many children it may itself spawn (0 = none) - both drive
    the safe-recursion governor."""
    B, G = _H["BRIEF_MARK"], _H["GRANT_MARK"]
    grant = [f"leash: {leash}", f"cwd: {cwd}", f"max_iterations: {max_iter}"]
    if model:
        grant.append(f"model: {model}")
    if provider:
        grant.append(f"provider: {provider}")
    if brain_host:
        grant.append(f"brain_host: {brain_host}")
    if tools:
        grant.append(f"tools: {tools}")
    if depth:
        grant.append(f"depth: {depth}")
    if child_budget:
        grant.append(f"child_budget: {child_budget}")
    if checkpoint:
        grant.append("checkpoint: 1")
    if board:
        grant.append(f"board: {board}")
    if taskboard:
        grant.append(f"taskboard: {taskboard}")
    out = [f"{B}\n{task.strip()}\n{B}", f"{G}\n" + "\n".join(grant) + f"\n{G}"]
    for text, mark_key in ((context, "SEED_CONTEXT_MARK"), (plan, "SEED_PLAN_MARK"),
                           (notes, "SEED_NOTES_MARK")):
        if text and text.strip():
            m = _H[mark_key]
            out.append(f"{m}\n{text.strip()}\n{m}")
    # A resumed worker: point it at the dead worker's transcript checkpoint. run_agent_brief
    # reloads that transcript, then runs `task` as a fresh steer turn on top of it.
    if resume:
        rm = _H["RESUME_MARK"]
        out.append(f"{rm}\n{resume.strip()}\n{rm}")
    return "\n".join(out) + "\n"


def _self_argv():
    """The command that launches THIS lean-coder as a headless worker. Uses the same
    interpreter + the CORE module file (captured as lc["__file__"] in setup()), so the
    worker is the exact build the parent runs (no PATH lookup, no version drift)."""
    core = Path(_H["__file__"]).resolve()
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(core))}"


def _tb_arg(args):
    """The optional taskboard scope for a team action (pause_all/stop_all/resume_all):
    args['taskboard'] stripped, or None = all workers this session (unscoped)."""
    tb = (args.get("taskboard") or "").strip()
    return tb or None


def run(args, cwd):
    if "bg_launch" not in _H:
        return ("error: dispatch_worker is not initialised (its setup() did not run; "
                "it must be enabled as a driver lean-tool).")
    action = (args.get("action") or "dispatch").strip().lower()
    if action == "status":
        return _worker_status(args.get("pid"))
    if action == "models":
        return _worker_models()
    if action == "result":
        return _worker_result(args.get("pid"))
    if action == "cancel":
        return _worker_cancel(args.get("pid"))
    if action == "pause":
        return _worker_pause(args.get("pid"))
    if action == "pause_all":
        return _worker_pause_all(_tb_arg(args))
    if action == "stop_all":
        return _worker_stop_all(_tb_arg(args))
    if action == "resume_all":
        return _worker_resume_all(cwd, _tb_arg(args))
    if action == "inject":
        return _worker_inject(args.get("pid"), args.get("text"), source="parent-agent")
    if action == "set_plan":
        return _worker_set_plan(args.get("pid"), args.get("plan") or args.get("text"))
    if action == "add_note":
        return _worker_add_note(args.get("pid"), args.get("notes") or args.get("text"))
    if action == "resume":
        return _worker_resume(args.get("pid"), args.get("text") or args.get("task"), cwd,
                              iterations=args.get("iterations"))
    if action == "transcript":
        return _worker_transcript(args.get("pid"), args.get("page"))
    if action == "promote":
        return _worker_promote(args.get("pid"), args.get("name") or args.get("text"))
    if action in ("board_claim", "board_release", "board_list"):
        return _worker_board(action, args.get("text") or args.get("task"))
    if action != "dispatch":
        return ("error: unknown action %r (use dispatch | models | status | result | "
                "transcript | cancel | pause | pause_all | stop_all | resume | resume_all | "
                "inject | set_plan | add_note | promote | board_claim | board_release | "
                "board_list)." % action)
    task = (args.get("task") or "").strip()
    if not task:
        return "error: dispatch_worker action='dispatch' needs a non-empty task."
    # Optional: SEED this fresh worker from a saved SESSION (or any checkpoint-shaped
    # file) - the unified {meta,messages} envelope means a session file is a valid
    # worker checkpoint. Given a bare name, resolve it under CONFIG_DIR/sessions.
    # run_agent_brief reloads that transcript, then runs `task` as a steer turn on top.
    seed_session = (args.get("from_session") or "").strip()
    seed_resume = ""
    if seed_session:
        p = Path(seed_session)
        if not p.is_file():
            cand = Path(_H["CONFIG_DIR"]) / "sessions" / (seed_session + ".json")
            if cand.is_file():
                p = cand
        if not p.is_file():
            return (f"error: from_session {seed_session!r} not found "
                    f"(looked for a file and under CONFIG_DIR/sessions/).")
        seed_resume = str(p.resolve())
    # SAFE-RECURSION governor: is THIS process allowed to spawn a worker at all, and if
    # so how many? The driver (worker_depth 0) is governed only by max_concurrent. A
    # WORKER (depth >= 1) must pass two gates before it can spawn a child:
    #   (a) DEPTH: my depth < worker_max_depth (default 1 -> a depth-1 worker is blocked;
    #       recursion is opt-IN by raising the ceiling).
    #   (b) CHILD BUDGET: I was granted a child budget by my parent, and I have some left
    #       (each spawn decrements it). Grandchildren get a SHRINKING budget so a subtree
    #       is bounded by depth x a decrementing count, not a flat number.
    # Both refuse rather than silently succeed - a misfire can't fork-bomb the API.
    _cfg = _H.get("cfg")
    my_depth = int(getattr(_cfg, "worker_depth", 0) or 0)
    ok, why, child_budget = _governor_check(my_depth)
    if not ok:
        return why

    # Operator ceiling: concurrency.
    bg_list = _H["bg_list"]
    max_conc = _ceiling("max_concurrent")
    live = bg_list(kind="worker")
    if max_conc and len(live) >= max_conc:
        return (f"error: at the worker limit ({max_conc} running). Wait for one to "
                f"finish (its result reaches you automatically) or raise "
                f"LEANCODER_WORKER_MAX_CONCURRENT.")

    # Model + provider. Default: inherit the driver's current model+provider (both
    # blank -> the worker keeps whatever it activates). A `model` may be ANY model on
    # ANY enabled provider; we infer its provider from the enabled-provider->models
    # map. An explicit `provider` disambiguates (same model name on two backends) or
    # forces a backend. Validated against the worker allowlist + real availability so
    # a bad name fails HERE (loud) rather than silently falling back inside the worker.
    model = (args.get("model") or "").strip()
    provider = (args.get("provider") or "").strip()
    brain_host = (args.get("brain_host") or "").strip()
    # brain_host relocates INFERENCE to a different ollama endpoint. The model then
    # lives on THAT box, not the driver's, so we can't validate it against the driver's
    # enabled-provider models - it's implicitly the ollama provider on the remote
    # endpoint. Resolve the alias and default provider to ollama.
    if brain_host:
        resolve = _H.get("resolve_host") or (lambda h, m=None: h)
        machines = getattr(_H.get("cfg"), "machines", {}) or {}
        brain_host = _H.get("_norm_host", lambda h: h)(resolve(brain_host, machines))
        if not provider:
            provider = "ollama"
        if provider != "ollama":
            return "error: brain_host is ollama-only (inference endpoints are ollama URLs)."
    if (model or provider) and not brain_host:
        pmodels = _H.get("enabled_provider_models", lambda: {})()
        if provider and provider not in pmodels:
            return (f"error: provider '{provider}' is not enabled. Enabled: "
                    f"{', '.join(pmodels) or '(none)'}.")
        if model:
            allow = _model_allowlist()
            if allow and model not in allow:
                return f"error: model '{model}' is not in the worker allowlist ({', '.join(allow)})."
            # Which enabled provider(s) offer this model?
            hosts = [p for p, ms in pmodels.items() if model in (ms or [])]
            if provider:
                if model not in (pmodels.get(provider) or []):
                    avail = ', '.join(pmodels.get(provider) or []) or '(unknown)'
                    return (f"error: model '{model}' is not available on provider "
                            f"'{provider}'. available there: {avail}")
            elif len(hosts) == 1:
                provider = hosts[0]                      # unambiguous -> infer it
            elif len(hosts) > 1:
                return (f"error: model '{model}' exists on multiple enabled providers "
                        f"({', '.join(hosts)}); pass provider= to disambiguate.")
            else:
                allm = ', '.join(sorted({m for ms in pmodels.values() for m in ms})) or '(unknown)'
                return (f"error: model '{model}' is not available on any enabled "
                        f"provider. available: {allm}")
    elif brain_host and model:
        # Still honour the allowlist (operator lockdown) even for a remote-brain model.
        allow = _model_allowlist()
        if allow and model not in allow:
            return f"error: model '{model}' is not in the worker allowlist ({', '.join(allow)})."

    # Where the worker's TOOLS run. Precedence:
    #   1. an explicit `host` arg -> run the worker's tools on THAT machine. We ensure a
    #      live ssh master to it HERE, in the foreground (the one place a password can be
    #      entered - a detached worker can't answer a prompt). Key auth is silent.
    #   2. else the PARENT's live remote -> ride the same executor path the parent uses.
    #   3. else local on the driver.
    want_host = (args.get("host") or "").strip()
    if want_host:
        ensure = _H.get("ensure_worker_master")
        if ensure is None:
            return "error: this build can't target a host for a worker (update lean-coder)."
        remote = ensure(want_host)
        if "error" in remote:
            return (f"error: {remote['error']}. The worker was NOT dispatched - "
                    f"connect/authenticate to '{want_host}' works from your terminal, "
                    f"so try again once it's reachable.")
        wcwd = (args.get("cwd") or "").strip() or remote["cwd"]
    else:
        remote = _H.get("active_remote", lambda: None)()
        if remote:
            # A detached worker can't answer an auth prompt, so the master MUST be live NOW
            # (we're in the foreground dispatch call - the one place a password could be
            # entered). Verify it; if it's down, tell the operator to reconnect.
            alive = _H.get("_ssh_master_alive", lambda *a: False)(remote["host"], remote["ctl"])
            if not alive:
                return (f"error: connected to {remote['host']} but its ssh master isn't live "
                        f"- a worker can't authenticate on its own. Run a remote command (or "
                        f"/connect {remote['host']}) to re-establish it, then retry.")
            # cwd is on the REMOTE: default to the parent's remote cwd, don't stat locally.
            wcwd = (args.get("cwd") or "").strip() or remote["cwd"]
        else:
            wcwd = (args.get("cwd") or "").strip() or cwd
            if not Path(wcwd).is_dir():
                return f"error: cwd is not a directory: {wcwd}"

    # Leash: cap the requested grant at the parent's live authority (never exceed it).
    leash, downgraded = _capped_leash(args.get("leash") or "r")
    cap_note = ""
    if downgraded:
        cap_note = (f" (capped to '{leash}' - a worker can't exceed your current "
                    f"'{getattr(_H.get('cfg'), 'leash', '?')}' leash)")

    # Optional per-worker tool allowlist. Accept a list or a comma string; validate every
    # name against what the PARENT actually exposes so a typo fails loud HERE, not silently
    # inside the worker. Meta tools (plan/note/compaction) are always kept, so a caller
    # need not list them. Empty after filtering -> treat as no restriction.
    tools_arg = args.get("tools")
    tools_csv = ""
    if tools_arg:
        if isinstance(tools_arg, str):
            want_tools = [t.strip() for t in tools_arg.split(",") if t.strip()]
        else:
            want_tools = [str(t).strip() for t in tools_arg if str(t).strip()]
        _META = {"update_plan", "note", "request_compact"}
        want_tools = [t for t in want_tools if t not in _META]
        if want_tools:
            parent_tools = _H.get("active_tool_names", lambda: set())()
            if parent_tools:
                bad = [t for t in want_tools if t not in parent_tools]
                if bad:
                    return (f"error: tool(s) not available to you: {', '.join(bad)}. "
                            f"You can only grant tools you have. Available: "
                            f"{', '.join(sorted(parent_tools))}.")
            tools_csv = ",".join(want_tools)

    # Optional named task-board (Phase 2b): assign this worker to a task DAG board by NAME.
    # The worker's run_agent_brief auto-loads the `board` lean-tool when this is set (so it
    # can report done/list) and injects the report-done contract into its brief. Distinct
    # from the int `board` claim-session above. The driver has usually already added+assigned
    # the task on the board; this just tells the worker WHICH board to report to.
    taskboard = (args.get("taskboard") or "").strip()

    # Optional seed STATE the parent curated: a background blob, a starting plan, and
    # notebook lines. NOT size-capped - "share state, not context" is a discipline for
    # the parent to exercise, not something to enforce by clipping the payload to a
    # useless state (an emergency brief may legitimately need a lot of context; a hard
    # cap could silently truncate the one step that matters). Passed through whole.
    seed_context = (args.get("context") or "").strip()
    seed_plan = (args.get("plan") or "").strip()
    seed_notes = (args.get("notes") or "").strip()
    # Per-worker iteration budget: the driver MAY request a cap at dispatch (e.g. a tight
    # leash for a scout, a longer one for a refactor), but it is capped at the operator
    # ceiling (env > cfg > default) - a grant is never above the grantor's authority, and
    # an env lockdown stays a hard limit. Junk/non-positive -> silently fall back to the
    # ceiling (never crash on a bad arg).
    _ceil_iter = _ceiling("max_iterations")
    _req_iter = args.get("iterations")
    max_iter = _ceil_iter
    if _req_iter is not None:
        try:
            _ri = int(_req_iter)
            if _ri > 0:
                max_iter = min(_ri, _ceil_iter)
        except (TypeError, ValueError):
            pass
    idle_timeout = _ceiling("idle_timeout")
    # Progress-staleness watchdog: bark (alert the parent), never bite. The worker bumps
    # its .progress file every iteration, so if that mtime goes stale the worker is stuck
    # (model looping / hung tool) rather than just quiet - distinct from idle_timeout,
    # which fires on the PARENT dying. Default = half the idle timeout, floored at 300s;
    # silence != dead, so it only alerts and the operator decides (cancel/inject).
    hb_timeout = max(300, idle_timeout // 2) if idle_timeout else 0

    # Write the brief + result-file targets under CONFIG_DIR/workers.
    wdir = _workers_dir()
    try:
        wdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"error: cannot create workers dir: {e}"
    # Unique per worker: two workers dispatched in the SAME second must not share a
    # brief/result path (os.getpid() is the parent's - identical for all of them), or
    # the second's ===RESULT=== overwrites the first's. A per-session monotonic counter
    # guarantees uniqueness even within one second.
    _H["_seq"] = _H.get("_seq", 0) + 1
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{_H['_seq']}"
    brief_file = wdir / f"{stamp}.brief"
    result_file = wdir / f"{stamp}.result"
    # Checkpoint the transcript (opt-in) so a dead/timed-out worker can be action='resume'd.
    # A session-seeded worker turns it ON regardless (so it can be promoted back later).
    do_checkpoint = bool(getattr(_cfg, "worker_checkpoint", False)) or bool(seed_resume)
    # Shared swarm board session: peers coordinate on the driver's board (keyed by the
    # driver's pid). A worker dispatching a child passes its own inherited board session
    # DOWN so a whole subtree shares one board; the driver seeds it with its own pid.
    board_session = int(getattr(_cfg, "worker_board_session", 0) or 0) or os.getpid()
    try:
        brief_file.write_text(_compose_brief(task, model, wcwd, max_iter, leash=leash,
                                             provider=provider, brain_host=brain_host,
                                             tools=tools_csv, context=seed_context,
                                             plan=seed_plan, notes=seed_notes,
                                             depth=my_depth + 1, child_budget=child_budget,
                                             checkpoint=do_checkpoint, board=board_session,
                                             taskboard=taskboard, resume=seed_resume))
    except OSError as e:
        return f"error: cannot write brief file: {e}"

    # The worker PROCESS always runs on the driver; when attached to a remote its
    # driver-local cwd is irrelevant (tools run remotely), so launch it from the parent's
    # cwd there. --cwd is the worker's LOCAL cfg.cwd (driver); when remote, run_agent_brief
    # leaves cfg.cwd at the driver default and the remote workspace carries wcwd instead.
    launch_cwd = cwd if remote else wcwd
    cmd = (f"{_self_argv()} --agent-run "
           f"--brief-file {shlex.quote(str(brief_file))} "
           f"--result-file {shlex.quote(str(result_file))} "
           f"--cwd {shlex.quote(launch_cwd)}")
    if remote:
        cmd += (f" --remote-host {shlex.quote(remote['host'])} "
                f"--remote-ctl {shlex.quote(remote['ctl'])}")
    if brain_host:
        # Point the worker's INFERENCE at a different ollama endpoint than the driver.
        # (Distinct from --remote-host, which relocates the worker's TOOLS.)
        cmd += f" --host {shlex.quote(brain_host)}"
    launched = _H["bg_launch"](cmd, cwd=launch_cwd, kind="worker", idle_timeout=idle_timeout,
                               heartbeat_timeout=hb_timeout,
                               heartbeat_file=str(brief_file) + ".progress")
    if "error" in launched:
        return f"error: could not launch worker: {launched['error']}"

    # Remember the result-file per pid so /worker can harvest it.
    where = remote["host"] if remote else "local"
    _H["workers"][launched["pid"]] = {"result": str(result_file), "task": task,
                                      "started": time.time(), "model": model or "(default)",
                                      "where": where, "log": launched.get("log"),
                                      "brief": str(brief_file),
                                      "checkpoint": do_checkpoint, "cmd": cmd,
                                      "launch_cwd": launch_cwd, "idle_timeout": idle_timeout,
                                      "hb_timeout": hb_timeout, "taskboard": taskboard}
    mstr = model or "current model"
    lstr = {"r": "read-only", "rw": "read+write", "rwe": "read+write+exec"}.get(leash, leash)
    _H["workers"][launched["pid"]]["leash"] = leash
    wnote = f"on {remote['host']}" if remote else "local (driver)"
    tnote = f", tools limited to [{tools_csv}]" if tools_csv else ""
    seeds = [n for n, v in (("context", seed_context), ("plan", seed_plan),
                            ("notes", seed_notes)) if v]
    snote = f", seeded {'+'.join(seeds)}" if seeds else ""
    return (f"dispatched worker pid {launched['pid']} ({mstr}, {lstr}, tools {wnote}"
            f"{tnote}{snote}, cwd {wcwd}){cap_note}.\n"
            f"It runs in the BACKGROUND. You do NOT need to poll for it: when it "
            f"finishes, its result is delivered to you automatically on a later turn. "
            f"Do NOT call bg_status (workers are hidden from it by design) and do NOT "
            f"sleep/cat the result file by hand - just carry on with other work, or if "
            f"you have nothing else to do, end your turn and wait for the finish notice. "
            f"(The human can run /worker to inspect it.)")


def _read_result(path):
    """The worker's ===RESULT=== block from its result file, or None if not written
    yet / unreadable."""
    try:
        return _H["_extract_marked"](Path(path).read_text(), _H["RESULT_MARK"])
    except OSError:
        return None


def _read_usage(result_path):
    """A finished worker's token spend from its '<brief>.usage' sidecar (written on exit
    by run_agent_brief). Returns (in_tokens, out_tokens); (0, 0) if absent/unreadable.
    Observability: summed for /worker status, nothing gates on it."""
    try:
        txt = Path(_brief_from_result(result_path) + ".usage").read_text()
    except OSError:
        return 0, 0
    vin = vout = 0
    for tok in txt.split():
        k, _, v = tok.partition("=")
        try:
            if k == "in":
                vin = int(v)
            elif k == "out":
                vout = int(v)
        except ValueError:
            pass
    return vin, vout


def _accrue_usage(meta):
    """Add a finished worker's token spend to the session's shared totals, ONCE. Guarded
    by meta['usage_counted'] so a re-scan can't double-count. Best-effort - a worker with
    no .usage sidecar (crashed before writing) simply contributes 0."""
    if meta.get("usage_counted"):
        return
    meta["usage_counted"] = True
    vin, vout = _read_usage(meta["result"])
    meta["tokens_in"], meta["tokens_out"] = vin, vout
    _H["tokens_in"] = _H.get("tokens_in", 0) + vin
    _H["tokens_out"] = _H.get("tokens_out", 0) + vout


def _worker_tokens_spent():
    """Cumulative tokens (in+out) spent by all FINISHED workers this session (their .usage
    sidecars, accrued once each in _accrue_usage). Still-running workers count 0 until they
    exit. Pure observability - surfaced in /worker status; nothing gates on it."""
    return _H.get("tokens_in", 0) + _H.get("tokens_out", 0)

def _worker_stderr_tail(result_path, n=15):
    """Last few lines of a FAILED worker's combined stdout+stderr log (bg_launch runs it
    with stderr merged into the log), so a crash/early death leaves a visible trace in the
    failure notice instead of vanishing. "" if we have no log path or it's unreadable."""
    meta = None
    for m in _H["workers"].values():
        if m.get("result") == result_path:
            meta = m
            break
    log = (meta or {}).get("log")
    tail = _H.get("_bg_log_tail")
    if not (log and tail):
        return ""
    try:
        return tail(log, n)
    except Exception:
        return ""


def _read_progress(result_path):
    """The worker's deterministic progress heartbeat (iter/elapsed/last-tool/intent),
    written each iteration to <brief>.progress by run_agent_brief. Returns the text
    (stripped) or "" if none yet. Facts only - no tokens, no model cooperation."""
    try:
        return Path(_brief_from_result(result_path) + ".progress").read_text().strip()
    except OSError:
        return ""


def _read_planview(result_path):
    """The worker's CURRENT pinned plan, mirrored to <brief>.planview each iteration by
    run_agent_brief. Lets a parent read the live plan (with the worker's own checkbox
    progress) so action='set_plan' can send back an EDITED copy instead of a blind
    overwrite. Returns the plan text (stripped) or "" if the worker has no pinned plan."""
    try:
        return Path(_brief_from_result(result_path) + ".planview").read_text().strip()
    except OSError:
        return ""

def _inject_count(result_path):
    """How many injects this worker has CONSUMED (lines in <brief>.injects.log), and the
    last one's teaser - the delivery ack for action='inject'. (0, "") if none."""
    try:
        lines = [l for l in Path(_brief_from_result(result_path) + ".injects.log")
                 .read_text().splitlines() if l.strip()]
    except OSError:
        return 0, ""
    if not lines:
        return 0, ""
    last = lines[-1].split("consumed:", 1)[-1].strip()
    return len(lines), last


def _finished_notice():
    """Scan tracked workers for ones that FINISHED since we last looked, and build the
    notice to inject into the model's next turn. A worker finishes one of two ways:
      - SUCCESS: its result file got a ===RESULT=== block -> report the (capped) result;
      - FAILED:  the process is GONE but no result was ever written (crash / early death /
        lease-kill before writing) -> report the failure + any stderr tail, so the model
        is never left waiting on a worker that will never report. Without this second
        branch a dead-without-result worker sits in silent limbo forever (the bug this
        fixes).
    Reports each newly-finished worker once (dedup via meta['announced']); when that clears
    the LAST outstanding worker, appends an 'all N finished' line. Fires a desktop ping per
    finish if the notify lean-tool's _ping is present. "" when nothing new finished."""
    workers = _H["workers"]
    if not workers:
        return ""
    ping = _H.get("ping")
    rows = _worker_rows()               # {pid: bg_status row}; absent row => proc gone
    lines = []
    just_finished = []
    for pid, meta in workers.items():
        if meta.get("announced"):
            continue
        res = _read_result(meta["result"])
        if res is not None:
            meta["announced"] = True
            just_finished.append(pid)
            _accrue_usage(meta)          # meter this worker's token spend into the session
            tail = res.strip()
            if len(tail) > 1500:         # keep a long result from flooding the turn
                tail = tail[:1500] + " ...[truncated; /worker %d for the full result]" % pid
            lines.append(f"worker {pid} finished ({meta.get('leash','r')}, {meta['model']}) - "
                         f"task: {meta['task'][:100]}\nresult:\n{tail}")
            if ping:
                try:
                    ping("lean-coder worker", f"worker {pid} finished")
                except Exception:
                    pass
            continue
        # No result. Only a FAILURE if the process is actually gone (a running worker
        # just hasn't written yet - leave it). A missing bg_status row => proc gone.
        if rows.get(pid) is not None:
            continue                     # still running; wait
        meta["announced"] = True
        just_finished.append(pid)
        errtail = _worker_stderr_tail(meta.get("result"))
        msg = (f"worker {pid} FAILED ({meta.get('leash','r')}, {meta['model']}) - "
               f"it exited without writing a result "
               f"(crash / early death / lease-kill).\ntask: {meta['task'][:100]}")
        if errtail:
            msg += f"\nstderr tail:\n{errtail}"
        lines.append(msg)
        if ping:
            try:
                ping("lean-coder worker", f"worker {pid} FAILED (no result)")
            except Exception:
                pass
    if not just_finished:
        return ""
    total = len(workers)
    outstanding = sum(1 for m in workers.values() if not m.get("announced"))
    if outstanding == 0 and total > 1:
        lines.append(f"all {total} dispatched workers have finished - collect any "
                     f"you haven't used with /worker.")
        if ping:
            try:
                ping("lean-coder workers", f"all {total} workers finished")
            except Exception:
                pass
    return "[" + "\n\n".join(lines) + "]"


def _worker_rows():
    """{pid: bg_status row} for this session's workers, keyed by pid. Empty if none."""
    return {r["pid"]: r for r in _H["bg_status"](os.getpid(), kind="worker")}


def _worker_status(pid=None):
    """MODEL-facing status of dispatched workers (the tool's action='status'). One
    line per worker: state, runtime, result-ready flag, task teaser. `pid` narrows to
    one. Note: finished results reach the model automatically - this is for the rare
    case it needs to check a slow/suspect one."""
    workers = _H["workers"]
    if not workers:
        return "no workers dispatched this session."
    rows = _worker_rows()
    if pid is not None:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return f"error: bad pid {pid!r}."
        meta = workers.get(pid)
        if not meta:
            return f"no worker with pid {pid} this session."
        workers = {pid: meta}
    out = []
    for wp, meta in workers.items():
        row = rows.get(wp)
        state = row["state"] if row else "gone"
        runtime = row["runtime"] if row else "?"
        ready = _read_result(meta["result"]) is not None
        if ready and not meta.get("usage_counted"):
            _accrue_usage(meta)          # keep the shared spend current when status is read
        line = (f"pid {wp}  {state}  (ran {runtime})  "
                f"result {'ready' if ready else 'pending'}  {meta['model']}  "
                f"task: {meta['task'][:80]}")
        if ready and meta.get("tokens_in") is not None:
            _t = meta.get("tokens_in", 0) + meta.get("tokens_out", 0)
            if _t:
                line += f"  ({_t:,} tok)"
        prog = _read_progress(meta["result"])
        if prog and not ready:
            line += "\n  progress: " + " | ".join(prog.splitlines())
        if not ready and pid is not None:
            # Only when narrowed to one worker (the plan can be many lines) - show its live
            # pinned plan so the parent can steer/edit it via set_plan.
            plan = _read_planview(meta["result"])
            if plan:
                line += "\n  plan:\n    " + "\n    ".join(plan.splitlines())
        n_inj, last_inj = _inject_count(meta["result"])
        if n_inj:
            line += f"\n  injects delivered: {n_inj} (last: {last_inj[:80]})"
        out.append(line)
    spent = _worker_tokens_spent()
    if spent:
        out.append(f"total worker spend: {spent:,} tokens (finished workers)")
    return "\n".join(out)


def _worker_models():
    """List the models available to dispatch a worker on, grouped by enabled provider,
    so the driver can pick a cheap leaf model without guessing an exact id. Honours the
    operator worker-model allowlist when one is set. Read-only, no side effects."""
    pmodels = _H.get("enabled_provider_models", lambda: {})()
    if not pmodels:
        return ("no enabled providers report models (a remote-brain ollama worker can "
                "still be dispatched with brain_host= + model=).")
    allow = _model_allowlist()
    cur = getattr(_H.get("cfg"), "model", "") or ""
    lines = [_H["bold"]("models you can dispatch a worker on:")]
    for prov, models in pmodels.items():
        models = list(models or [])
        if allow:
            models = [m for m in models if m in allow]
        head = _H["cyan"](prov)
        if not models:
            lines.append(f"  {head}: (none available)")
            continue
        lines.append(f"  {head}:")
        for m in models:
            tag = _H["dim"]("  (current default)") if m == cur else ""
            lines.append(f"    {m}{tag}")
    lines.append(_H["dim"]("dispatch with model=<id> (provider= only if the id is on "
                           "more than one). Omit model= to use your current default."))
    if allow:
        lines.append(_H["dim"](f"note: operator allowlist active - only these are grantable."))
    return "\n".join(lines)



def _worker_result(pid):
    """MODEL-facing full result (the tool's action='result'): return a worker's
    COMPLETE ===RESULT=== block, untruncated. The turn-rider finish notice caps the
    result at 1500 chars to avoid flooding the turn; this is how the model pulls the
    rest without a human running /worker. `pid` is required (which worker); with no
    pid, if exactly one worker exists, use it, else ask which."""
    workers = _H["workers"]
    if not workers:
        return "no workers dispatched this session."
    if pid is None:
        if len(workers) == 1:
            pid = next(iter(workers))
        else:
            return ("error: action='result' needs a pid (which worker). "
                    "Use action='status' to list them.")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"error: bad pid {pid!r}."
    meta = workers.get(pid)
    if not meta:
        return f"no worker with pid {pid} this session."
    res = _read_result(meta["result"])
    if res is None:
        return (f"worker {pid} has no result yet (still running or terminated before "
                f"writing one). Use action='status' to check its state.")
    return f"worker {pid} result (task: {meta['task'][:100]}):\n{res.strip()}"


_TRANSCRIPT_PAGE = 6000   # chars of rendered trail per page (keeps one read digestible)


def _render_transcript_turn(m):
    """One transcript message -> readable lines. Assistant reasoning, its tool calls
    (name + compact args), and tool results are each shown with a role tag; the raw
    provider shapes (list content, tool_calls arrays) are flattened to text."""
    role = m.get("role", "?")
    content = m.get("content")
    if isinstance(content, list):   # provider block form -> join text parts
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or "")
            else:
                parts.append(str(b))
        content = "\n".join(p for p in parts if p)
    content = "" if content is None else str(content).strip()
    tag = {"user": "STEER", "assistant": "worker", "tool": "tool"}.get(role, role)
    # The first STEER turn is the fixed worker preamble (~2KB of boilerplate the driver
    # already knows) followed by 'TASK:\n<the task>'. Collapse it to just the task so the
    # transcript shows the worker's ACTUAL work, not the rubric.
    if role == "user" and "\nTASK:\n" in content:
        content = "TASK: " + content.split("\nTASK:\n", 1)[1].strip()
    out = []
    if content:
        out.append(f"[{tag}] {content}")
    for tc in m.get("tool_calls", []) or []:
        fn = tc.get("function") or tc
        name = fn.get("name", "?")
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args)
        args = (args or "").strip()
        if len(args) > 300:
            args = args[:300] + "..."
        out.append(f"[worker>tool] {name}({args})")
    if role == "tool" and not content:
        out.append(f"[tool] {m.get('tool_name', '')}: (no output)")
    return "\n".join(out)


def _worker_transcript(pid, page):
    """MODEL-facing transcript (the tool's action='transcript'): render a worker's OWN
    full reasoning + tool trail from its checkpoint, paged, so the driver can inspect HOW
    a worker reached its result (or where a dead one got stuck) without resuming it. Needs
    the worker to have been dispatched with worker_checkpoint on (the same sidecar resume
    uses). `page` is 1-based (default 1); the reply reports the page count and whether more
    remain. Read-only - never mutates the worker or its checkpoint."""
    workers = _H["workers"]
    if not workers:
        return "no workers dispatched this session."
    if pid is None:
        if len(workers) == 1:
            pid = next(iter(workers))
        else:
            return ("error: action='transcript' needs a pid (which worker). "
                    "Use action='status' to list them.")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"error: bad pid {pid!r}."
    meta = workers.get(pid)
    if not meta:
        return f"no worker with pid {pid} this session."
    ckpt = _brief_from_result(meta["result"]) + ".checkpoint"
    if not Path(ckpt).exists():
        return (f"worker {pid} has no transcript checkpoint (dispatched with checkpointing "
                f"off, so nothing was recorded). Turn on worker_checkpoint (/set or config) "
                f"before dispatching workers whose trail you may want to read.")
    try:
        data = json.loads(Path(ckpt).read_text())
    except Exception as e:
        return f"worker {pid} checkpoint is unreadable ({e})."
    body = [m for m in data.get("messages", []) if m.get("role") != "system"]
    if not body:
        return f"worker {pid} checkpoint has no recorded turns yet."
    blocks = [b for b in (_render_transcript_turn(m) for m in body) if b]
    full = "\n\n".join(blocks)
    # Paginate on rendered chars, breaking only at turn boundaries so no turn is split.
    pages, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > _TRANSCRIPT_PAGE:
            pages.append(cur)
            cur = b
        else:
            cur = (cur + "\n\n" + b) if cur else b
    if cur:
        pages.append(cur)
    npages = len(pages) or 1
    try:
        p = max(1, int(page or 1))
    except (TypeError, ValueError):
        p = 1
    if p > npages:
        p = npages
    head = (f"worker {pid} transcript (task: {meta['task'][:80]}) - "
            f"page {p}/{npages}, {len(body)} turns:")
    footer = ("" if p >= npages
              else f"\n\n[more: {npages - p} page(s) left - action='transcript' page={p + 1}]")
    return f"{head}\n\n{pages[p - 1]}{footer}"


def _worker_promote(pid, name):
    """PROMOTE a worker's transcript into a saved SESSION you can /load. Reads the
    worker's checkpoint (the unified {meta,messages} envelope) and writes it under
    CONFIG_DIR/sessions/<name>.json via the core save_session. Needs worker_checkpoint
    (the sidecar must exist). Works for a running, finished, or paused worker - it's a
    snapshot of the trail so far."""
    if pid is None:
        return "error: action='promote' needs a pid (which worker to save)."
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"error: bad pid {pid!r}."
    name = (name or "").strip()
    if not name:
        return "error: action='promote' needs name=<session name> to save under."
    meta = _H["workers"].get(pid)
    if not meta:
        return f"no worker with pid {pid} this session."
    ckpt = _brief_from_result(meta["result"]) + ".checkpoint"
    if not Path(ckpt).exists():
        return (f"worker {pid} has no checkpoint to promote (dispatched with checkpointing "
                f"off). Dispatch with worker_checkpoint on (session-seeded workers force it).")
    save_session = _H.get("save_session")
    load_envelope = _H.get("load_envelope")
    if not save_session or not load_envelope:
        return "error: core save_session/load_envelope not available (old core?)."
    try:
        data = json.loads(Path(ckpt).read_text())
    except Exception as e:
        return f"worker {pid} checkpoint is unreadable ({e})."
    body, cmeta = load_envelope(data)
    if not body:
        return f"worker {pid} checkpoint has no recorded turns yet - nothing to promote."
    # Rebuild a full message list (system + body) as save_session expects, and carry the
    # worker's pinned_plan + notes across from the checkpoint meta. Stamp origin so the
    # promoted session is identifiable as an ex-worker (the worker's brief stamp).
    messages = [{"role": "system", "content": ""}] + body
    stamp = Path(_brief_from_result(meta["result"])).stem
    try:
        path, _ = save_session(messages, _H["cfg"], name,
                               pinned_plan=cmeta.get("pinned_plan", ""),
                               notes=cmeta.get("notes", []),
                               origin=f"worker:{stamp}")
    except ValueError:
        return f"error: invalid session name {name!r}."
    except Exception as e:
        return f"error: could not save session: {e}"
    return (f"promoted worker {pid} -> session '{name}' ({len(body)} turns). "
            f"/load {name} to resume it here.")


def _worker_cancel(pid):
    """MODEL-facing cancel (the tool's action='cancel'): kill a still-running worker
    by pid. An already-finished worker is a no-op (its result stands - use it). Reuses
    the same process-group kill the human /bg kill uses."""
    workers = _H["workers"]
    if pid is None:
        return "error: action='cancel' needs a pid (the worker to kill)."
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"error: bad pid {pid!r}."
    meta = workers.get(pid)
    if not meta:
        return f"no worker with pid {pid} this session (nothing to cancel)."
    if _read_result(meta["result"]) is not None:
        return (f"worker {pid} already finished - its result stands; nothing to cancel. "
                f"Use action='status' or read the result it delivered.")
    rows = _worker_rows()
    row = rows.get(pid)
    if row and row["state"] != "running":
        return f"worker {pid} is not running ({row['state']}); nothing to cancel."
    # Kill the worker AND every descendant it spawned. A worker-spawned grandchild is
    # setsid'd into its own process group, so a single killpg would orphan it; _bg_kill_tree
    # walks the registry's owner graph to reach every level. Fall back to the flat kill if
    # core predates the tree helper.
    kill = _H.get("_bg_kill_tree") or _H.get("_bg_kill")
    if not kill:
        return "error: no kill hook available (core too old)."
    kill(pid)
    meta["announced"] = True    # suppress a finish notice for a worker we deliberately killed
    return (f"cancelled worker {pid} (SIGTERM to it and any workers it spawned). "
            f"task: {meta['task'][:80]}")


def _live_worker_pids(taskboard=None):
    """pids of THIS session's workers that are still running (a bg row in 'running' state)
    and have not delivered a result. Optionally narrowed to one taskboard (the team
    gesture: 'pause the mining team'). Order: dispatch order (dict insertion)."""
    workers = _H["workers"]
    rows = _worker_rows()
    out = []
    for pid, meta in workers.items():
        if taskboard is not None and meta.get("taskboard", "") != taskboard:
            continue
        if _read_result(meta["result"]) is not None:
            continue
        row = rows.get(pid)
        if row is not None and row.get("state") != "running":
            continue
        out.append(pid)
    return out


def _pause_one(pid):
    """PAUSE a running worker: kill its process but KEEP its transcript checkpoint so it
    can be action='resume'd later. Writes a '<brief>.suspended' sentinel first (so the
    reaper leaves the checkpoint intact - a paused worker is parked, not orphaned), then
    SIGTERMs the pid. Requires worker_checkpoint (else there's nothing to resume from);
    the worker checkpoints each iteration, so a checkpoint already exists on disk. Returns
    (ok, message)."""
    workers = _H["workers"]
    meta = workers.get(pid)
    if not meta:
        return False, f"no worker with pid {pid} this session."
    if _read_result(meta["result"]) is not None:
        return False, f"worker {pid} already finished - nothing to pause (its result stands)."
    if not meta.get("checkpoint"):
        return False, (f"worker {pid} has checkpointing off, so it can't be paused+resumed "
                       f"(nothing would survive the kill). Dispatch with worker_checkpoint on.")
    brief = meta.get("brief") or _brief_from_result(meta["result"])
    ckpt = brief + ".checkpoint"
    if not Path(ckpt).exists():
        return False, (f"worker {pid} has not written a checkpoint yet (too early to pause - "
                       f"let it run one iteration first).")
    # Sentinel BEFORE the kill: if the reaper races in the instant after SIGTERM, the
    # sentinel is already there to protect the checkpoint family.
    try:
        Path(brief + ".suspended").write_text(str(time.time()))
    except OSError as e:
        return False, f"error: could not mark worker {pid} suspended: {e}"
    kill = _H.get("_bg_kill_tree") or _H.get("_bg_kill")
    if not kill:
        return False, "error: no kill hook available (core too old)."
    kill(pid)
    meta["announced"] = True          # a deliberate pause is not a failure - no finish notice
    meta["paused"] = True
    return True, f"paused worker {pid} (checkpoint parked; task: {meta['task'][:60]})"


def _worker_pause(pid):
    """MODEL-facing pause (the tool's action='pause'): kill a running worker but KEEP its
    checkpoint parked on disk so action='resume' can bring it back later. Unlike 'cancel'
    (which lets the reaper wipe everything), a paused worker is resumable indefinitely -
    the controller's 'stop everyone, resume later' gesture. Needs worker_checkpoint on."""
    if pid is None:
        return "error: action='pause' needs a pid (which worker to pause)."
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"error: bad pid {pid!r}."
    ok, msg = _pause_one(pid)
    if ok:
        msg += ". Bring it back with action='resume' pid=%d (a resume is a new pid)." % pid
    return msg


def _worker_pause_all(taskboard=None):
    """MODEL-facing pause-all (the tool's action='pause_all'): pause EVERY running worker
    this session (optionally only those on taskboard=<name>) in one call, each keeping its
    checkpoint for a later resume. The 'freeze the whole team' gesture."""
    pids = _live_worker_pids(taskboard)
    scope = f" on board '{taskboard}'" if taskboard else ""
    if not pids:
        return f"no running workers{scope} to pause."
    done, failed = [], []
    for pid in pids:
        ok, msg = _pause_one(pid)
        (done if ok else failed).append((pid, msg))
    lines = [f"paused {len(done)} worker(s){scope} (each resumable via action='resume'):"]
    for pid, _ in done:
        lines.append(f"  pid {pid} paused")
    for pid, msg in failed:
        lines.append(f"  pid {pid} NOT paused: {msg}")
    return "\n".join(lines)


def _worker_stop_all(taskboard=None):
    """MODEL-facing stop-all (the tool's action='stop_all'): kill EVERY running worker
    this session (optionally only taskboard=<name>) and DISCARD it (the reaper wipes its
    sidecars - not resumable). The hard 'shut the whole team down' gesture; use pause_all to
    keep them resumable."""
    pids = _live_worker_pids(taskboard)
    scope = f" on board '{taskboard}'" if taskboard else ""
    if not pids:
        return f"no running workers{scope} to stop."
    stopped = []
    for pid in pids:
        out = _worker_cancel(pid)
        stopped.append((pid, out))
    return (f"stopped {len(stopped)} worker(s){scope} (killed + discarded, not resumable; "
            f"use pause_all to keep them resumable):\n"
            + "\n".join(f"  pid {pid}" for pid, _ in stopped))


def _worker_resume_all(cwd, taskboard=None):
    """MODEL-facing resume-all (the tool's action='resume_all'): relaunch EVERY paused
    worker this session (optionally only taskboard=<name>), each from its parked
    checkpoint. The counterpart to pause_all - 'unfreeze the team'. Each resume is a new
    pid; skips workers that aren't paused."""
    paused = [(pid, meta) for pid, meta in _H["workers"].items()
              if meta.get("paused")
              and (taskboard is None or meta.get("taskboard", "") == taskboard)]
    scope = f" on board '{taskboard}'" if taskboard else ""
    if not paused:
        return f"no paused workers{scope} to resume."
    lines = [f"resuming {len(paused)} paused worker(s){scope}:"]
    for pid, _ in paused:
        out = _worker_resume(pid, "", cwd)
        first = out.splitlines()[0] if out else out
        lines.append(f"  {first}")
    return "\n".join(lines)

def _brief_from_result(result_path):
    """The worker's brief-file path from its result path (they share a stamp: the tool
    writes <stamp>.brief + <stamp>.result). The inject sidecar lives beside the brief
    (<stamp>.brief.inject), driver-side even for a remote worker."""
    r = str(result_path)
    return r[:-len(".result")] + ".brief" if r.endswith(".result") else r + ".brief"


def _worker_inject(pid, text, source="operator"):
    """Deliver a mid-task message to a RUNNING worker. Writes (append-safe, NUL-
    separated so two quick injects can't clobber) to the worker's <brief>.inject
    sidecar; the worker drains it at its next between-iteration boundary and appends it
    as a user turn (see run_agent_brief's _drain_injects). `source` prefixes the message
    ([operator inject] / [parent-agent inject]) so the worker can weight it. Does NOT
    interrupt a running tool call - it lands on the next reasoning step; use cancel to
    stop hard. A finished/gone worker can't receive one."""
    workers = _H["workers"]
    if pid is None:
        return "error: action='inject' needs a pid (which worker) and text."
    msg = (text or "").strip()
    if not msg:
        return "error: action='inject' needs non-empty text (the message to send)."
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"error: bad pid {pid!r}."
    meta = workers.get(pid)
    if not meta:
        return f"no worker with pid {pid} this session (nothing to inject into)."
    if _read_result(meta["result"]) is not None:
        return (f"worker {pid} already finished - too late to inject; its result stands. "
                f"Read it, or dispatch a fresh worker for the change.")
    rows = _worker_rows()
    row = rows.get(pid)
    if row and row["state"] != "running":
        return f"worker {pid} is not running ({row['state']}); can't inject."
    inject_path = Path(_brief_from_result(meta["result"]) + ".inject")
    labelled = f"[{source} inject] {msg}"
    try:
        with inject_path.open("a") as f:
            f.write(labelled + "\0")
    except OSError as e:
        return f"error: could not write inject for worker {pid}: {e}"
    return (f"injected into worker {pid} (as [{source} inject]). It arrives on the worker's "
            f"NEXT reasoning step (not an interrupt - a running command finishes first); the "
            f"worker is prompted to briefly acknowledge it. Check delivery with action='status'.")


def _running_worker_sidecar(pid, action, suffix):
    """Shared guard for the live-steer actions (inject/set_plan/add_note): validate that
    `pid` names a RUNNING worker and return (sidecar_path, None), else (None, error_str).
    `suffix` is the sidecar to target ('.inject'/'.plan'/'.note')."""
    workers = _H["workers"]
    if pid is None:
        return None, f"error: action='{action}' needs a pid (which worker)."
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None, f"error: bad pid {pid!r}."
    meta = workers.get(pid)
    if not meta:
        return None, f"no worker with pid {pid} this session (nothing to steer)."
    if _read_result(meta["result"]) is not None:
        return None, (f"worker {pid} already finished - too late to {action}; its result "
                      f"stands. Read it, or dispatch a fresh worker for the change.")
    row = _worker_rows().get(pid)
    if row and row["state"] != "running":
        return None, f"worker {pid} is not running ({row['state']}); can't {action}."
    return Path(_brief_from_result(meta["result"]) + suffix), None


def _worker_set_plan(pid, text):
    """Live-steer a RUNNING worker's PINNED PLAN (the tool's action='set_plan'). Writes
    the full plan (GOAL + '- [ ]' TODO) to the worker's <brief>.plan sidecar; the worker
    drains it at its next between-iteration boundary and REPLACES its pinned plan via
    agent._update_plan (see run_agent_brief's _drain_plan), and gets an inline inject so
    it can't miss the change. A full replacement, not an append - the last set_plan wins.

    Call with NO plan text to READ the worker's current plan (with its own checkbox
    progress) - do that first, edit the returned plan, and send it back, so you EDIT the
    plan rather than blindly clobbering the worker's progress. Does not interrupt a
    running tool call. A finished/gone worker can't receive one."""
    path, err = _running_worker_sidecar(pid, "set_plan", ".plan")
    if err:
        return err
    plan = (text or "").strip()
    if not plan:
        # No plan supplied -> READ affordance: hand back the live plan for editing.
        meta = _H["workers"].get(int(pid))
        cur = _read_planview(meta["result"]) if meta else ""
        if not cur:
            return (f"worker {pid} has no pinned plan yet (nothing to edit). Send a full plan "
                    f"as `plan` to set one.")
        return (f"worker {pid} current plan (edit this and send it back as `plan` to update "
                f"it - tick/add boxes rather than overwrite its progress):\n{cur}")
    try:
        with path.open("a") as f:
            f.write(plan + "\0")
    except OSError as e:
        return f"error: could not write plan for worker {pid}: {e}"
    return (f"queued a plan update for worker {pid}. It replaces the worker's pinned plan on "
            f"its NEXT reasoning step and the worker is pinged inline to acknowledge it. "
            f"Confirm with action='status'.")


def _worker_add_note(pid, text):
    """Live-steer a RUNNING worker's NOTEBOOK (the tool's action='add_note'). Appends the
    note (NUL-separated so two quick notes can't clobber) to the worker's <brief>.note
    sidecar; the worker drains it at its next between-iteration boundary and appends it to
    agent.notes, tagged 'parent:' so it reads as handed down (see _drain_notes). Does not
    interrupt a running tool call. A finished/gone worker can't receive one."""
    msg = (text or "").strip()
    if not msg:
        return "error: action='add_note' needs non-empty text (the note to add)."
    path, err = _running_worker_sidecar(pid, "add_note", ".note")
    if err:
        return err
    try:
        with path.open("a") as f:
            f.write(msg + "\0")
    except OSError as e:
        return f"error: could not write note for worker {pid}: {e}"
    return (f"queued a note for worker {pid}'s notebook. It lands on the worker's NEXT "
            f"reasoning step (not an interrupt). Confirm with action='status'.")


def _worker_resume(pid, text, cwd, iterations=None):
    """RELAUNCH a worker that DIED without finishing (max_iterations / lease-kill / crash /
    stopped incomplete), reloading its transcript checkpoint so it continues from where it
    left off instead of cold. `text` is a fresh steer (what to fix/do next). `iterations`
    (optional) grants the resumed worker a FRESH max_iterations budget - important when the
    worker died BY hitting its cap, or it would just hit the same cap again; absent = reuse
    the original grant's budget. Returns a NEW pid (the old process is dead - resume is a
    new OS process; lineage is tracked via meta['resumed_from']). Requirements: the worker
    was dispatched with worker_checkpoint on (so a '<brief>.checkpoint' exists), it is NOT
    still running (use inject for that), and it has not already delivered a result (that
    stands - dispatch fresh instead)."""
    workers = _H["workers"]
    if pid is None:
        return "error: action='resume' needs a pid (which dead worker to relaunch)."
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"error: bad pid {pid!r}."
    meta = workers.get(pid)
    if not meta:
        return f"no worker with pid {pid} this session (nothing to resume)."
    if _read_result(meta["result"]) is not None:
        return (f"worker {pid} already finished with a result - nothing to resume (its "
                f"result stands; read it, or dispatch a fresh worker for new work).")
    row = _worker_rows().get(pid)
    if row is not None and row.get("state") == "running":
        return (f"worker {pid} is still running - don't resume a live worker. Use "
                f"action='inject' to steer it, or action='cancel' to stop it first.")
    ckpt = _brief_from_result(meta["result"]) + ".checkpoint"
    if not Path(ckpt).exists():
        return (f"worker {pid} has no transcript checkpoint, so it can't be resumed - it "
                f"was dispatched with checkpointing off. Turn on worker_checkpoint (/set "
                f"or config) before dispatching workers you may want to resume.")
    steer = (text or "").strip() or ("Continue your task from where you left off and "
                                     "finish it.")
    # Concurrency ceiling still applies (a resume is a new running worker).
    max_conc = _ceiling("max_concurrent")
    if max_conc and len(_H["bg_list"](kind="worker")) >= max_conc:
        return (f"error: at the worker limit ({max_conc} running); can't resume now. Wait "
                f"for one to finish or raise LEANCODER_WORKER_MAX_CONCURRENT.")

    # Build a FRESH brief that reuses the dead worker's GRANT verbatim (same leash/model/
    # tools/cwd), plus a RESUME marker pointing at its checkpoint and checkpoint:1 so the
    # resumed worker is itself resumable. New stamp -> new brief/result files (the old
    # ones are swept normally); the steer becomes the brief's task (its first turn on top
    # of the reloaded transcript).
    try:
        orig = Path(meta["brief"]).read_text()
    except OSError as e:
        return f"error: cannot read worker {pid}'s brief to resume it: {e}"
    grant = _H["_extract_marked"](orig, _H["GRANT_MARK"]) or ""
    B, G, RM = _H["BRIEF_MARK"], _H["GRANT_MARK"], _H["RESUME_MARK"]
    # Optionally bump the iteration budget (a worker that died ON its cap needs a bigger
    # one, or it dies on it again). Strip the old max_iterations line and re-add it.
    new_iter = None
    if iterations is not None:
        try:
            new_iter = int(iterations)
        except (TypeError, ValueError):
            return f"error: bad iterations {iterations!r} (want an integer)."
        if new_iter <= 0:
            return "error: iterations must be a positive integer."
    drop = ("checkpoint:",) + (("max_iterations:",) if new_iter is not None else ())
    grant_lines = [l for l in grant.splitlines() if l.strip()
                   and not l.strip().lower().startswith(drop)]
    if new_iter is not None:
        grant_lines.append(f"max_iterations: {new_iter}")
    grant_lines.append("checkpoint: 1")
    new_brief_text = "\n".join([
        f"{B}\n{steer}\n{B}",
        f"{G}\n" + "\n".join(grant_lines) + f"\n{G}",
        f"{RM}\n{ckpt}\n{RM}"]) + "\n"

    wdir = _workers_dir()
    try:
        wdir.mkdir(parents=True, exist_ok=True)   # may have been swept since dispatch
    except OSError as e:
        return f"error: cannot create workers dir: {e}"
    _H["_seq"] = _H.get("_seq", 0) + 1
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{_H['_seq']}"
    brief_file = wdir / f"{stamp}.brief"
    result_file = wdir / f"{stamp}.result"
    try:
        brief_file.write_text(new_brief_text)
    except OSError as e:
        return f"error: cannot write resume brief: {e}"

    # Relaunch: reuse the dead worker's launch command verbatim but swap in the new brief/
    # result paths (this preserves its --cwd / --remote-host / --remote-ctl / --host wiring,
    # so a remote/remote-brain worker resumes on the same target).
    old_cmd = meta.get("cmd", "")
    cmd = old_cmd.replace(shlex.quote(str(meta["brief"])), shlex.quote(str(brief_file)))
    cmd = cmd.replace(shlex.quote(meta["result"]), shlex.quote(str(result_file)))
    if str(brief_file) not in cmd or str(result_file) not in cmd:
        return ("error: could not reconstruct the resume launch command (the worker's "
                "original launch is unavailable). Dispatch a fresh worker instead.")
    launch_cwd = meta.get("launch_cwd") or cwd
    launched = _H["bg_launch"](cmd, cwd=launch_cwd, kind="worker",
                               idle_timeout=meta.get("idle_timeout") or _ceiling("idle_timeout"),
                               heartbeat_timeout=meta.get("hb_timeout") or 0,
                               heartbeat_file=str(brief_file) + ".progress")
    if "error" in launched:
        return f"error: could not relaunch worker: {launched['error']}"
    new_pid = launched["pid"]
    workers[new_pid] = {"result": str(result_file), "task": meta["task"],
                        "started": time.time(), "model": meta.get("model", "(default)"),
                        "where": meta.get("where", "local"), "log": launched.get("log"),
                        "brief": str(brief_file), "checkpoint": True, "cmd": cmd,
                        "launch_cwd": launch_cwd,
                        "idle_timeout": meta.get("idle_timeout"),
                        "hb_timeout": meta.get("hb_timeout"),
                        "leash": meta.get("leash", "r"), "resumed_from": pid,
                        "taskboard": meta.get("taskboard", "")}
    # The old worker is retired - suppress any lingering failure notice for it.
    meta["announced"] = True
    # If this worker was deliberately PAUSED, clear its sentinel now that we've read the
    # checkpoint + relaunched: the old sidecars are no longer parked and reap normally.
    if meta.get("paused"):
        try:
            Path(_brief_from_result(meta["result"]) + ".suspended").unlink()
        except OSError:
            pass
        meta["paused"] = False
    return (f"resumed worker {pid} as NEW pid {new_pid} (a resume is a fresh process; the "
            f"old pid is dead). It reloaded the saved transcript and continues from where "
            f"it stopped, with your steer on top. Its result reaches you automatically when "
            f"it finishes - don't poll.")


def _board_session_owner():
    """This process's identity ON the shared board: (session_id, owner_tag).
      - the DRIVER coordinates its OWN board, keyed by its pid; its owner tag is 'driver'.
      - a WORKER uses the board session handed down in its grant (worker_board_session)
        and tags claims with its own pid so peers can see who holds what.
    Returns (0, tag) when there is no board (a lone worker), so board actions no-op safely."""
    cfg = _H.get("cfg")
    sess = int(getattr(cfg, "worker_board_session", 0) or 0)
    if sess:
        return sess, f"worker:{os.getpid()}"
    # The driver: it IS the board root (its pid). Depth 0 = driver (workers carry a
    # board session and never reach here with sess 0 unless boardless).
    if int(getattr(cfg, "worker_depth", 0) or 0) == 0:
        return os.getpid(), "driver"
    return 0, f"worker:{os.getpid()}"


def _worker_board(action, path):
    """Shared swarm-board coordination (first-claim-wins + TTL). Lets peer workers on one
    repo avoid editing the same file:
      - board_claim  (text=<path>): claim a path before editing it. Succeeds if free (or
        already yours); REFUSED if a live peer holds it (you pick other work / wait).
      - board_release(text=<path>): release your claim when done, freeing it for peers.
      - board_list   : the currently-held claims (path -> owner).
    A claim carries a TTL, so a dead worker's claims expire and never deadlock the board.
    No-op with a clear note when there is no board (a lone worker)."""
    claim = _H.get("_board_claim")
    if not claim:
        return "error: this build has no shared board (update lean-coder)."
    sess, owner = _board_session_owner()
    if not sess:
        return ("no shared board is active (you are a lone worker). Board coordination only "
                "applies when several workers run on one repo.")
    if action == "board_list":
        live = _H["_board_read_claims"](sess)
        if not live:
            return "board: no paths are currently claimed."
        rows = "\n".join(f"  {p}  ->  {v.get('owner')}" for p, v in sorted(live.items()))
        return f"board: {len(live)} path(s) claimed:\n{rows}"
    p = (path or "").strip()
    if not p:
        return f"error: action='{action}' needs text=<path> (the file to claim/release)."
    if action == "board_release":
        _H["_board_release"](sess, p, owner)
        return f"released your claim on {p} (free for peers)."
    # board_claim
    ok, holder = claim(sess, p, owner)
    if ok:
        return (f"claimed {p} (yours until you release it or its TTL lapses). Edit it, then "
                f"action='board_release' text='{p}' when done.")
    return (f"REFUSED: {p} is already claimed by {holder}. Pick different work or wait for "
            f"them to release it; do NOT edit it (a concurrent edit would clash).")


def _worker_cmd(agent, cfg, arg):
    """/worker - human command, at PARITY with the model's dispatch_worker actions:
      /worker                    list dispatched workers (state + result-ready)
      /worker <pid>              print that worker's full result
      /worker transcript <pid> [page]  read a worker's own reasoning/tool trail (paged)
      /worker status [pid]       the model-facing status view (state/runtime/ready)
      /worker cancel <pid>       kill + discard a still-running worker
      /worker pause <pid>        kill but KEEP its checkpoint (resumable later)
      /worker pause_all [board]   pause every worker (or one board's team) - resumable
      /worker stop_all [board]    kill + discard every worker (or one board's team)
      /worker resume_all [board]  relaunch every paused worker (or one board's team)
      /worker inject <pid> <msg> mid-task message to a running worker
      /worker set_plan <pid> <plan>  replace a running worker's pinned plan
      /worker add_note <pid> <note>  add a note to a running worker's notebook
      /worker resume <pid> <steer>   relaunch a DEAD worker from its saved transcript
      /worker board                  show the shared swarm board's held file claims
      /worker models                 list the models you can dispatch a worker on
    Subcommands reuse the same helpers the tool uses, so the human and the model see
    identical behaviour."""
    workers = _H["workers"]
    if not workers:
        print(_H["dim"]("no workers dispatched this session."))
        return
    bg_status = _H["bg_status"]
    rows = {r["pid"]: r for r in bg_status(os.getpid(), kind="worker")}
    arg = (arg or "").strip()
    parts = arg.split()

    # Subcommands (parity with the model tool). A bare pid stays the result shortcut.
    if parts and parts[0].lower() == "board":
        print(_worker_board("board_list", None))
        return
    if parts and parts[0].lower() == "models":
        print(_worker_models())
        return
    # Fleet lifecycle (optional taskboard scope): /worker pause_all [board]
    if parts and parts[0].lower() in ("pause_all", "stop_all", "resume_all"):
        tb = parts[1] if len(parts) > 1 else None
        if parts[0].lower() == "pause_all":
            print(_worker_pause_all(tb))
        elif parts[0].lower() == "stop_all":
            print(_worker_stop_all(tb))
        else:
            print(_worker_resume_all(str(getattr(cfg, "cwd", ".")), tb))
        return
    if parts and parts[0].lower() in ("status", "cancel", "pause", "result", "transcript",
                                      "inject", "set_plan", "add_note", "resume"):
        sub = parts[0].lower()
        pid = parts[1] if len(parts) > 1 else None
        if sub == "status":
            print(_worker_status(pid))
        if sub == "status":
            print(_worker_status(pid))
        elif sub == "result":
            print(_worker_result(pid))
        elif sub == "transcript":
            # /worker transcript <pid> [page]
            page = parts[2] if len(parts) > 2 else None
            print(_worker_transcript(pid, page))
        elif sub == "inject":
            # /worker inject <pid> <message...> - the rest of the line is the message.
            text = arg.split(None, 2)[2] if len(parts) > 2 else ""
            print(_worker_inject(pid, text, source="operator"))
        elif sub == "set_plan":
            # /worker set_plan <pid> <plan...> - the rest of the line is the plan.
            text = arg.split(None, 2)[2] if len(parts) > 2 else ""
            print(_worker_set_plan(pid, text))
        elif sub == "add_note":
            # /worker add_note <pid> <note...> - the rest of the line is the note.
            text = arg.split(None, 2)[2] if len(parts) > 2 else ""
            print(_worker_add_note(pid, text))
        elif sub == "resume":
            # /worker resume <pid> <steer...> - the rest of the line is the steer.
            text = arg.split(None, 2)[2] if len(parts) > 2 else ""
            print(_worker_resume(pid, text, str(getattr(cfg, "cwd", "."))))
        elif sub == "pause":
            print(_worker_pause(pid))
        else:  # cancel
            print(_worker_cancel(pid))
        return

    if arg.isdigit():
        pid = int(arg)
        meta = workers.get(pid)
        if not meta:
            print(_H["dim"](f"no worker with pid {pid} this session."))
            return
        res = _read_result(meta["result"])
        print(_H["bold"](f"worker {pid}") + _H["dim"](f"  {meta['task'][:80]}"))
        if res:
            print(res)
        else:
            print(_H["dim"]("  (running - no result yet)"))
            prog = _read_progress(meta["result"])
            if prog:
                for pl in prog.splitlines():
                    if pl.lower().startswith("intent:"):
                        # Intent is prose; print label dim, value normal so it reads.
                        print(_H["dim"]("  intent:  ") + pl.split(":", 1)[1].strip())
                    else:
                        print(_H["dim"]("  " + pl))
        n_inj, last_inj = _inject_count(meta["result"])
        if n_inj:
            print(_H["dim"](f"  injects: {n_inj} delivered (last: {last_inj[:80]})"))
        return
    print(_H["bold"]("workers:"))
    for pid, meta in workers.items():
        row = rows.get(pid)
        state = row["state"] if row else "gone"
        runtime = row["runtime"] if row else "?"
        done = _read_result(meta["result"]) is not None
        tag = _H["green"]("result ready") if done else _H["dim"](state)
        print(f"  {_H['cyan'](str(pid))}  {tag}  (ran {runtime})  "
              f"{meta['model']}  {_H['dim'](meta['task'][:56])}")
        if not done:
            # Compact heartbeat only (iter + last tool); full progress + intent via
            # /worker <pid>. Keep it to one short line so the list stays scannable.
            prog = _read_progress(meta["result"])
            if prog:
                pl = {k.strip().lower(): v.strip()
                      for k, v in (x.split(":", 1) for x in prog.splitlines() if ":" in x)}
                bits = []
                first = prog.splitlines()[0] if prog.splitlines() else ""
                if first.startswith("iter "):
                    bits.append(first)
                if pl.get("last"):
                    bits.append("last " + pl["last"][:40])
                n_inj, _last = _inject_count(meta["result"])
                if n_inj:
                    bits.append(f"{n_inj} inject" + ("s" if n_inj > 1 else ""))
                if bits:
                    print(_H["dim"]("        " + "  ".join(bits)))
    print(_H["dim"]("  /worker <pid> = full result + progress/intent · status [pid] · "
                    "cancel <pid> · inject/set_plan/add_note <pid> <text> · resume <pid> <steer>"))


def _worker_completer(agent, cfg):
    """Tab-completion for /worker's first argument: the subcommand verbs plus every
    live worker pid (so `cancel <Tab>` / a bare `<Tab>` offers real pids). Matches the
    menu contract of other multi-verb commands (e.g. /mcp)."""
    opts = ["status", "result", "transcript", "cancel", "pause", "pause_all", "stop_all",
            "resume", "resume_all", "inject", "set_plan", "add_note", "board", "models"]
    opts += [str(pid) for pid in _H.get("workers", {})]
    return opts


def setup(lc, cfg):
    # Capture the core hooks + helpers run()/the command need (a tool's run() gets
    # no lc). setup() is driver-only = exactly where a worker is launched, so these
    # are always present when run() fires.
    for k in ("bg_launch", "bg_list", "bg_status", "_bg_kill", "_bg_kill_tree", "_bg_log_tail", "_extract_marked", "CONFIG_DIR",
              "BRIEF_MARK", "GRANT_MARK", "RESULT_MARK", "RESUME_MARK", "LEASH_LEVELS", "_norm_leash",
              "SEED_CONTEXT_MARK", "SEED_PLAN_MARK", "SEED_NOTES_MARK",
              "active_remote", "_ssh_master_alive", "ensure_worker_master",
              "active_tool_names", "resolve_host", "_norm_host",
              "_board_claim", "_board_release", "_board_read_claims",
              "save_session", "load_envelope",
              "dim", "bold", "green", "cyan"):
        if k in lc:
            _H[k] = lc[k]
    _H["__file__"] = lc.get("__file__")     # the CORE module file, for the worker argv
    _H["cfg"] = cfg
    _H["workers"] = {}                       # pid -> {result, task, started, model, announced}
    _H.setdefault("tokens_in", 0)            # observability: cumulative finished-worker spend
    _H.setdefault("tokens_out", 0)           # (accrued as each worker finishes; see _accrue_usage)

    # Desktop ping on a worker finish, IF the notify lean-tool is also enabled (its
    # _ping is exposed on lc when it runs). Optional - absent => notices are text-only.
    _H["ping"] = lc.get("_ping")

    # Surface a worker's finished result to the model: wrap run_turn so each turn, any
    # tracked worker that finished since last turn injects a finish notice (per-worker
    # + an 'all done' line when the last one clears). Workers are hidden from the core
    # bg finished-notify (kind filter), so this is their notification path.
    Agent = lc["Agent"]
    _orig_run_turn = Agent.run_turn

    def run_turn(self, user_input):
        try:
            notice = _finished_notice()
        except Exception:
            notice = ""
        if notice:
            user_input = user_input + "\n\n" + notice
        return _orig_run_turn(self, user_input)
    Agent.run_turn = run_turn

    # AUTONOMY: also feed worker finishes into the event-driven WAKE path so a finished
    # worker wakes the agent with no operator input (when cfg.wake_on_bg_finish is on).
    # Workers are hidden from the core bg wake (kind filter), so this hook is their only
    # wake route. _finished_notice owns its dedup (meta['announced']), so a worker
    # surfaced by wake won't re-report via the run_turn wrapper above. Best-effort: if
    # the core is older and lacks the registry, wake simply stays task-only.
    _reg = lc.get("register_wake_hook")
    if _reg:
        _reg(lambda: _finished_notice() or "")

    # A small helper to list models available on this box (for validation), resolved
    # lazily so it reflects the live provider.
    def _available_models():
        try:
            get_provider = lc["get_provider"]
            spec = get_provider(cfg.provider) if cfg.provider else None
            return list(spec["list_models"]() or []) if spec else []
        except Exception:
            return []
    _H["available_models"] = _available_models

    # Map every ENABLED provider -> its available models, so a worker `model` can be
    # ANY model on any enabled backend (its provider inferred). Resolved lazily so it
    # reflects the live registry. Returns an ordered dict {provider_name: [models]}.
    def _enabled_provider_models():
        get_provider = lc["get_provider"]
        out = {}
        for name in (cfg.providers_enabled or []):
            try:
                spec = get_provider(name)
                out[name] = list(spec["list_models"]() or []) if spec else []
            except Exception:
                out[name] = []
        return out
    _H["enabled_provider_models"] = _enabled_provider_models

    lc["register_command"]("/worker", _worker_cmd,
                           "list workers (+ results); /worker <pid> = full result, "
                           "/worker status [pid], /worker cancel <pid>, "
                           "/worker inject <pid> <message>",
                           _worker_completer)
