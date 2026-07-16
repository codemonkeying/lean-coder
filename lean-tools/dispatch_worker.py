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
"""
import os
import shlex
import sys
import time
from pathlib import Path

# Built-in ceiling defaults (last resort: env var > live cfg > this - see docstring).
_DEFAULTS = {"max_concurrent": 10, "idle_timeout": 1800, "max_iterations": 30}
_ENV = {"max_concurrent": "LEANCODER_WORKER_MAX_CONCURRENT",
        "idle_timeout": "LEANCODER_WORKER_IDLE_TIMEOUT",
        "max_iterations": "LEANCODER_WORKER_MAX_ITER"}

TOOL = {
    "name": "dispatch_worker",
    "description": (
        "Hand a self-contained sub-task to a background worker agent (own session + context); "
        "returns a pid. It runs in the BACKGROUND and its result is delivered to you "
        "AUTOMATICALLY on a later turn - do NOT poll it (bg_status won't show workers; "
        "don't sleep/cat its file). For scoped errands whose output would bloat this "
        "session - e.g. 'scout this module, report where X is handled'. Default READ-ONLY; "
        "leash='rw'/'rwe' to let it edit/run, never above YOUR capability."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["dispatch", "status", "cancel"],
                       "description": "What to do (default 'dispatch'): 'dispatch' launches a "
                                      "new worker from `task`; 'status' reports your dispatched "
                                      "workers (state, runtime, whether a result is ready) - use "
                                      "it if you must check on one, though finished results reach "
                                      "you automatically; 'cancel' kills the worker named by `pid`."},
            "pid": {"type": "integer",
                    "description": "Worker pid to act on (required for action='cancel'; optional "
                                   "for 'status' to show just one). From the dispatch return line."},
            "task": {"type": "string",
                     "description": "Full self-contained instruction: what to do and exactly "
                                    "what to report back (the worker has no other context). "
                                    "Required for action='dispatch'."},
            "model": {"type": "string",
                      "description": "Optional model for the worker's brain - ANY model on an "
                                     "ENABLED provider (see /models); its provider is inferred "
                                     "automatically. Defaults to inheriting your current model."},
            "provider": {"type": "string",
                         "description": "Optional backend to run the worker's brain on (must be "
                                        "enabled). Only needed to disambiguate when the same model "
                                        "name exists on more than one enabled provider; otherwise "
                                        "inferred from `model`."},
            "brain_host": {"type": "string",
                           "description": "Optional ollama endpoint (URL or a /machines alias) to "
                                          "run the worker's BRAIN (inference) on - distinct from "
                                          "`host` (which is where its TOOLS run). Defaults to the "
                                          "driver's own ollama host. Use it to run the worker's "
                                          "model on a different box than the driver (ollama only)."},
            "host": {"type": "string",
                     "description": "Optional box to run the worker's TOOLS on ([user@]host or a "
                                    "/connect name); defaults to this session's target. A password "
                                    "host is prompted ONCE here - a worker can't answer prompts."},
            "cwd": {"type": "string",
                    "description": "Optional working directory (defaults to current)."},
            "leash": {"type": "string", "enum": ["r", "rw", "rwe"],
                      "description": "Worker capability: r=read-only (default), rw=edit, rwe=edit+run. "
                                     "Capped at your own leash."},
        },
        "required": [],
    },
    "driver_only": True,   # spawns the worker process on the driver; never pushed remotely
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


def _compose_brief(task, model, cwd, max_iter, leash="r", provider="", brain_host=""):
    """Build the brief file text: the task wrapped in BRIEF markers + a GRANT header.
    `leash` is the already-capped grant (see _capped_leash). `provider` (optional) is
    the backend the worker must activate for `model`; absent = inherit the driver's.
    `brain_host` (optional) is an ollama inference endpoint for the worker's BRAIN,
    distinct from where its tools run; absent = the driver's own host."""
    B, G = _H["BRIEF_MARK"], _H["GRANT_MARK"]
    grant = [f"leash: {leash}", f"cwd: {cwd}", f"max_iterations: {max_iter}"]
    if model:
        grant.append(f"model: {model}")
    if provider:
        grant.append(f"provider: {provider}")
    if brain_host:
        grant.append(f"brain_host: {brain_host}")
    return (f"{B}\n{task.strip()}\n{B}\n{G}\n" + "\n".join(grant) + f"\n{G}\n")


def _self_argv():
    """The command that launches THIS lean-coder as a headless worker. Uses the same
    interpreter + the CORE module file (captured as lc["__file__"] in setup()), so the
    worker is the exact build the parent runs (no PATH lookup, no version drift)."""
    core = Path(_H["__file__"]).resolve()
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(core))}"


def run(args, cwd):
    if "bg_launch" not in _H:
        return ("error: dispatch_worker is not initialised (its setup() did not run; "
                "it must be enabled as a driver lean-tool).")
    action = (args.get("action") or "dispatch").strip().lower()
    if action == "status":
        return _worker_status(args.get("pid"))
    if action == "cancel":
        return _worker_cancel(args.get("pid"))
    if action != "dispatch":
        return f"error: unknown action {action!r} (use dispatch | status | cancel)."
    task = (args.get("task") or "").strip()
    if not task:
        return "error: dispatch_worker action='dispatch' needs a non-empty task."

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

    max_iter = _ceiling("max_iterations")
    idle_timeout = _ceiling("idle_timeout")

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
    try:
        brief_file.write_text(_compose_brief(task, model, wcwd, max_iter, leash=leash,
                                             provider=provider, brain_host=brain_host))
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
    launched = _H["bg_launch"](cmd, cwd=launch_cwd, kind="worker", idle_timeout=idle_timeout)
    if "error" in launched:
        return f"error: could not launch worker: {launched['error']}"

    # Remember the result-file per pid so /worker can harvest it.
    where = remote["host"] if remote else "local"
    _H["workers"][launched["pid"]] = {"result": str(result_file), "task": task,
                                      "started": time.time(), "model": model or "(default)",
                                      "where": where}
    mstr = model or "current model"
    lstr = {"r": "read-only", "rw": "read+write", "rwe": "read+write+exec"}.get(leash, leash)
    _H["workers"][launched["pid"]]["leash"] = leash
    wnote = f"on {remote['host']}" if remote else "local (driver)"
    return (f"dispatched worker pid {launched['pid']} ({mstr}, {lstr}, tools {wnote}, "
            f"cwd {wcwd}){cap_note}.\n"
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


def _finished_notice():
    """Scan tracked workers for ones that FINISHED since we last looked (result file
    written), and build the notice to inject into the model's next turn. Reports each
    newly-finished worker once (dedup via meta['announced']); when that clears the LAST
    outstanding worker, appends an 'all N finished' line. Fires a desktop ping per
    finish if the notify lean-tool's _ping is present. "" when nothing new finished."""
    workers = _H["workers"]
    if not workers:
        return ""
    ping = _H.get("ping")
    lines = []
    just_finished = []
    for pid, meta in workers.items():
        if meta.get("announced"):
            continue
        res = _read_result(meta["result"])
        if res is None:
            continue                     # still running / not written yet
        meta["announced"] = True
        just_finished.append(pid)
        tail = res.strip()
        if len(tail) > 1500:             # keep a long result from flooding the turn
            tail = tail[:1500] + " …[truncated; /worker %d for the full result]" % pid
        lines.append(f"worker {pid} finished ({meta.get('leash','r')}, {meta['model']}) - "
                     f"task: {meta['task'][:100]}\nresult:\n{tail}")
        if ping:
            try:
                ping("lean-coder worker", f"worker {pid} finished")
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
        out.append(f"pid {wp}  {state}  (ran {runtime})  "
                   f"result {'ready' if ready else 'pending'}  {meta['model']}  "
                   f"task: {meta['task'][:80]}")
    return "\n".join(out)


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
    kill = _H.get("_bg_kill")
    if not kill:
        return "error: no kill hook available (core too old)."
    kill(pid)
    meta["announced"] = True    # suppress a finish notice for a worker we deliberately killed
    return f"cancelled worker {pid} (SIGTERM to its process group). task: {meta['task'][:80]}"


def _worker_cmd(agent, cfg, arg):
    """/worker - list dispatched workers with state + result; /worker <pid> prints
    that worker's full result."""
    workers = _H["workers"]
    if not workers:
        print(_H["dim"]("no workers dispatched this session."))
        return
    bg_status = _H["bg_status"]
    rows = {r["pid"]: r for r in bg_status(os.getpid(), kind="worker")}
    arg = (arg or "").strip()
    if arg.isdigit():
        pid = int(arg)
        meta = workers.get(pid)
        if not meta:
            print(_H["dim"](f"no worker with pid {pid} this session."))
            return
        res = _read_result(meta["result"])
        print(_H["bold"](f"worker {pid}") + _H["dim"](f"  {meta['task'][:80]}"))
        print(res if res else _H["dim"]("(no result yet - still running or terminated)"))
        return
    print(_H["bold"]("workers:"))
    for pid, meta in workers.items():
        row = rows.get(pid)
        state = row["state"] if row else "gone"
        runtime = row["runtime"] if row else "?"
        done = _read_result(meta["result"]) is not None
        tag = _H["green"]("result ready") if done else _H["dim"](state)
        print(f"  {_H['cyan'](str(pid))}  {tag}  (ran {runtime})  "
              f"{meta['model']}  {_H['dim'](meta['task'][:60])}")
    print(_H["dim"]("  /worker <pid> prints that worker's full result"))


def setup(lc, cfg):
    # Capture the core hooks + helpers run()/the command need (a tool's run() gets
    # no lc). setup() is driver-only = exactly where a worker is launched, so these
    # are always present when run() fires.
    for k in ("bg_launch", "bg_list", "bg_status", "_bg_kill", "_extract_marked", "CONFIG_DIR",
              "BRIEF_MARK", "GRANT_MARK", "RESULT_MARK", "LEASH_LEVELS", "_norm_leash",
              "active_remote", "_ssh_master_alive", "ensure_worker_master",
              "resolve_host", "_norm_host",
              "dim", "bold", "green", "cyan"):
        if k in lc:
            _H[k] = lc[k]
    _H["__file__"] = lc.get("__file__")     # the CORE module file, for the worker argv
    _H["cfg"] = cfg
    _H["workers"] = {}                       # pid -> {result, task, started, model, announced}

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
                           "list dispatched worker agents (+ results); /worker <pid> "
                           "prints a worker's full result")
