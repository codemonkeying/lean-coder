"""board - a driver-orchestrated TASK BOARD (a named dependency DAG of tasks).

The swarm coordination surface: the DRIVER lays out a graph of tasks (with
dependencies), assigns a worker to each, and marks progress; workers report their
own task done and read the board to see what's going on. This is the PUSH model -
the driver schedules; workers are hands, they do NOT self-select their next task
(a dumb worker picking its own work is how you get the UI built around the wrong
database). A board is a NAMED, on-disk, session-shaped JSON doc: it survives a
crash and can be handed to a different lean-coder session to keep driving.

Distinct from a worker's file CLAIM (dispatch_worker board_claim): a claim is the
"don't both edit auth.py" MUTEX; this is the "task C waits for task D" DAG.

Least privilege - three separate capabilities, each capped at the grantor:
  - assign is DRIVER-ONLY (it directs a worker); a worker cannot assign.
  - done / fail a worker may call for the task IT was assigned.
  - create / add / block are driver actions; list / find are read, open to anyone.
The board grants COORDINATION, not spawn (the recursion governor gates spawn) and
not act (the leash gates read/edit/run).

Safe (no edit/run of the user's tree; it only reads+writes board JSON under the
lean-coder config dir). The heavy lifting lives in the core _taskboard_* helpers;
this file is a thin action router over them.
"""

_H = {}   # core hooks captured in setup(): the _taskboard_* primitives + cfg + colours


TOOL = {
    "name": "board",
    "glyph": "\u25a4",   # a ruled square: the task board / map
    "description": (
        "Driver task board: a named dependency DAG of tasks you (the driver) schedule "
        "workers over. action='create' a board, 'add' tasks (with deps=[t1,t2] that must "
        "finish first), 'assign' a worker pid to a ready task, 'list' to see every task "
        "with a computed ready/blocked flag (assign only READY ones - a blocked task's "
        "deps aren't done yet), 'done'/'fail' to record an outcome, 'block' to park a "
        "running task, 'find' to search, 'reconcile' to collect every finished task's "
        "result in dependency order (dep before dependent) once the DAG is done. Workers "
        "report their OWN task done/fail and can list/find/reconcile; only the driver "
        "creates/adds/assigns/blocks. The board is the map: hand out work whose deps are "
        "done, mark each finish, re-check what is ready, reconcile the results at the end."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["create", "add", "assign", "block", "done", "fail",
                                "list", "find", "reconcile"],
                       "description": "create=new board; add=append a task; assign=put a "
                                      "worker on a ready task (driver); block=park a task; "
                                      "done/fail=record an outcome; list=all tasks + ready/"
                                      "blocked; find=search tasks by text; reconcile=finished "
                                      "results in dependency order."},
            "board": {"type": "string",
                      "description": "The board name (like a session name). Required by every "
                                     "action."},
            "task": {"type": "string",
                     "description": "For add: the new task's name/description. For assign/block/"
                                    "done/fail: the task id (e.g. 't3', from add or list)."},
            "deps": {"type": "array", "items": {"type": "string"},
                     "description": "For add: task ids this task depends on (must all be 'done' "
                                    "before it is ready). Omit for no deps."},
            "worker": {"type": "string",
                       "description": "For assign: the worker pid (or label) to put on the task."},
            "note": {"type": "string",
                     "description": "Optional free text: driver->worker context on assign/block "
                                    "(e.g. 'the DB is postgres'), or worker->driver why on fail."},
            "result_ref": {"type": "string",
                           "description": "For done: a pointer to the work product (e.g. the "
                                          "worker's result file path). Used by the reconciler."},
            "title": {"type": "string",
                      "description": "For create: an optional human title for the board."},
            "query": {"type": "string",
                      "description": "For find: match tasks whose id/name/note contains this text."},
            "status": {"type": "string",
                       "description": "For list: optionally filter to one status "
                                      "(open|assigned|blocked|done|failed), or 'ready' for only "
                                      "the ready-to-assign tasks."},
        },
        "required": ["action", "board"],
    },
    "safe": True,
}


def setup(lc, cfg):
    """Capture the core task-board primitives + cfg (for the driver gate) + colours. A
    tool's run() gets no lc, so everything it needs is stashed on _H here."""
    for k in ("_taskboard_create", "_taskboard_load", "_taskboard_save", "_taskboard_add",
              "_taskboard_ready", "_taskboard_set_status", "_taskboard_reconcile",
              "_taskboards_list", "_tb_task", "dim", "bold", "green", "cyan"):
        if k in lc:
            _H[k] = lc[k]
    _H["cfg"] = cfg


def _is_driver():
    """True when this process is the DRIVER (tree depth 0), False for a spawned worker.
    Gates the assign/create/add/block actions (a worker directs no one)."""
    cfg = _H.get("cfg")
    try:
        return int(getattr(cfg, "worker_depth", 0) or 0) == 0
    except (TypeError, ValueError):
        return True


def _driver_only(action):
    return (f"error: action='{action}' is driver-only - a worker cannot direct the board "
            f"(it may only list/find, and done/fail its own task).")


def _fmt_task(t, ready_ids):
    """One task as a compact line for list/find output."""
    tid = t.get("id", "?")
    st = t.get("status", "?")
    flag = " READY" if tid in ready_ids else ""
    who = f" @{t['assignee']}" if t.get("assignee") not in (None, "") else ""
    deps = t.get("deps") or []
    dep = f" deps={','.join(deps)}" if deps else ""
    note = f"  - {t['note']}" if t.get("note") else ""
    rr = f"  ->{t['result_ref']}" if t.get("result_ref") else ""
    return f"  {tid} [{st}{flag}]{who}{dep} {t.get('name','')}{rr}{note}"


def _render(name, board):
    """A board's full task list with computed ready/blocked, newest concerns first."""
    ready, blocked = _H["_taskboard_ready"](board)
    ready_ids = {t.get("id") for t in ready}
    counts = board.get("meta", {}).get("counts", {})
    head = _H["bold"](f"board '{name}'") + _H["dim"](
        f"  ({', '.join(f'{k}:{v}' for k, v in counts.items() if v)})" if counts else "  (empty)")
    lines = [head]
    for t in board.get("tasks", []):
        lines.append(_fmt_task(t, ready_ids))
    if ready:
        lines.append(_H["green"](f"ready to assign: {', '.join(sorted(ready_ids))}"))
    else:
        lines.append(_H["dim"]("ready to assign: (none)"))
    return "\n".join(lines)


def run(args, cwd):
    if "_taskboard_load" not in _H:
        return "error: board tool is not initialised (its setup() did not run)."
    action = (args.get("action") or "").strip().lower()
    name = (args.get("board") or "").strip()
    if not name:
        return "error: board tool needs a 'board' name."

    if action == "create":
        if not _is_driver():
            return _driver_only(action)
        board, err = _H["_taskboard_create"](name, title=(args.get("title") or "").strip(),
                                              owner=str(getattr(_H.get("cfg"), "worker_board_session", "") or ""))
        if err:
            return f"error: {err}"
        return f"created board '{name}'. Add tasks with action='add'."

    # Every other action operates on an existing board.
    board = _H["_taskboard_load"](name)
    if board is None:
        return f"error: no board named '{name}' (create it first with action='create')."

    if action == "list":
        flt = (args.get("status") or "").strip().lower()
        if flt == "ready":
            ready, _ = _H["_taskboard_ready"](board)
            if not ready:
                return f"board '{name}': no tasks are ready to assign."
            ready_ids = {t.get("id") for t in ready}
            return "\n".join([_H["bold"](f"board '{name}' ready:")]
                             + [_fmt_task(t, ready_ids) for t in ready])
        if flt:
            ready, _ = _H["_taskboard_ready"](board)
            ready_ids = {t.get("id") for t in ready}
            sel = [t for t in board.get("tasks", []) if t.get("status") == flt]
            if not sel:
                return f"board '{name}': no tasks with status '{flt}'."
            return "\n".join([_H["bold"](f"board '{name}' [{flt}]:")]
                             + [_fmt_task(t, ready_ids) for t in sel])
        return _render(name, board)

    if action == "find":
        q = (args.get("query") or args.get("task") or "").strip().lower()
        if not q:
            return "error: action='find' needs a 'query'."
        ready, _ = _H["_taskboard_ready"](board)
        ready_ids = {t.get("id") for t in ready}
        hits = [t for t in board.get("tasks", [])
                if q in str(t.get("id", "")).lower()
                or q in str(t.get("name", "")).lower()
                or q in str(t.get("note", "")).lower()]
        if not hits:
            return f"board '{name}': no task matches '{q}'."
        return "\n".join([_H["bold"](f"board '{name}' matches for '{q}':")]
                         + [_fmt_task(t, ready_ids) for t in hits])

    if action == "reconcile":
        # Read action, open to anyone: the result_refs of every DONE task in dependency
        # (topological) order, dep-before-dependent - what the driver concatenates to
        # collect the swarm's finished work once the DAG has run.
        pairs = _H["_taskboard_reconcile"](board)
        tasks = board.get("tasks", [])
        done_no_ref = [t for t in tasks
                       if t.get("status") == "done" and not t.get("result_ref")]
        pending = [t for t in tasks if t.get("status") not in ("done", "failed")]
        failed = [t for t in tasks if t.get("status") == "failed"]
        if not pairs:
            base = f"board '{name}': nothing to reconcile - no done task has a result_ref yet."
            if pending:
                base += f" ({len(pending)} task(s) still unfinished.)"
            return base
        lines = [_H["bold"](f"board '{name}' reconcile (dependency order):")]
        for i, (t, rr) in enumerate(pairs, 1):
            lines.append(f"  {i}. {t.get('id','?')} {t.get('name','')}"
                         + _H["cyan"](f"  -> {rr}"))
        tail = []
        if done_no_ref:
            tail.append(f"{len(done_no_ref)} done task(s) had no result_ref (skipped)")
        if pending:
            tail.append(f"{len(pending)} still unfinished")
        if failed:
            tail.append(f"{len(failed)} failed")
        if tail:
            lines.append(_H["dim"]("note: " + "; ".join(tail) + "."))
        return "\n".join(lines)

    if action == "add":
        if not _is_driver():
            return _driver_only(action)
        task_name = (args.get("task") or "").strip()
        if not task_name:
            return "error: action='add' needs a 'task' (its name/description)."
        deps = [str(d).strip() for d in (args.get("deps") or []) if str(d).strip()]
        task, err = _H["_taskboard_add"](board, task_name, deps=deps,
                                         note=(args.get("note") or "").strip())
        if err:
            return f"error: {err}"
        _H["_taskboard_save"](name, board)
        ready, _ = _H["_taskboard_ready"](board)
        rflag = "READY" if task["id"] in {t.get("id") for t in ready} else "blocked (deps not done)"
        return f"added {task['id']} '{task_name}'{(' deps=' + ','.join(deps)) if deps else ''} -> {rflag}."

    if action == "assign":
        if not _is_driver():
            return _driver_only(action)
        tid = (args.get("task") or "").strip()
        worker = (args.get("worker") or "").strip()
        if not tid or not worker:
            return "error: action='assign' needs a 'task' id and a 'worker'."
        t = _H["_tb_task"](board, tid)
        if not t:
            return f"error: unknown task id '{tid}'."
        ready, _ = _H["_taskboard_ready"](board)
        if tid not in {r.get("id") for r in ready} and t.get("status") == "open":
            # open but not ready = deps unmet; refuse (assigning it wastes a worker that
            # would sit on unmet deps - the whole point of the DAG).
            unmet = [d for d in t.get("deps", [])
                     if (_H["_tb_task"](board, d) or {}).get("status") != "done"]
            return (f"error: {tid} is not ready - it depends on {', '.join(unmet)} which "
                    f"is not done. Assign a READY task (action='list' status='ready').")
        note = (args.get("note") or "").strip()
        _H["_taskboard_set_status"](board, tid, "assigned", assignee=worker,
                                    note=note or None)
        _H["_taskboard_save"](name, board)
        return f"assigned {tid} '{t.get('name','')}' to worker {worker}."

    if action in ("done", "fail", "block"):
        tid = (args.get("task") or "").strip()
        if not tid:
            return f"error: action='{action}' needs a 'task' id."
        t = _H["_tb_task"](board, tid)
        if not t:
            return f"error: unknown task id '{tid}'."
        # block is a driver action (parking a task); done/fail a worker may call for the
        # task it was assigned (or the driver, for any task).
        if action == "block" and not _is_driver():
            return _driver_only(action)
        note = (args.get("note") or "").strip()
        if action == "done":
            _H["_taskboard_set_status"](board, tid, "done", note=note or None,
                                        result_ref=(args.get("result_ref") or "").strip() or None)
            msg = f"marked {tid} DONE."
        elif action == "fail":
            _H["_taskboard_set_status"](board, tid, "failed", note=note or None)
            msg = f"marked {tid} FAILED."
        else:  # block
            _H["_taskboard_set_status"](board, tid, "blocked", note=note or None)
            msg = f"parked {tid} as BLOCKED."
        _H["_taskboard_save"](name, board)
        # On a done, surface what that unblocked so the driver knows what to assign next.
        if action == "done":
            ready, _ = _H["_taskboard_ready"](board)
            newly = sorted(r.get("id") for r in ready)
            if newly:
                msg += f" ready now: {', '.join(newly)}."
        return msg

    return f"error: unknown action '{action}'."
