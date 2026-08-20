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

Least privilege - separate capabilities, each capped at the grantor:
  - create / add / assign are DRIVER-ONLY (they direct the swarm); a worker cannot.
  - done / fail a worker may call for the task IT was assigned.
  - list / reconcile are read, open to anyone.
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
        "A named task board: a dependency DAG of tasks you (the driver) schedule workers "
        "over. Lay out tasks with deps, assign each ready one to a worker, mark finishes, and "
        "the board recomputes what's ready next; 'reconcile' collects the results in "
        "dependency order at the end. A task is READY only once its deps are all done. Push "
        "model: the driver schedules; workers report their own task + read the board, but "
        "never self-select. Lifecycle: open -> assigned -> done|failed."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["create", "add", "assign", "done", "fail",
                                "list", "reconcile", "participant"],
                       "description": "create=new board; add=append a task (deps=[...]); "
                                      "assign=put a worker/participant on a READY task; done/fail="
                                      "record an outcome; list=tasks (status= or query= filters); "
                                      "reconcile=done results in dependency order; participant="
                                      "register/list peer agents (pre-declare an expected role, or "
                                      "list who is registered). Driver-only: create/add/assign/"
                                      "participant. Workers: done/fail their task + list/reconcile."},
            "board": {"type": "string",
                      "description": "Board name (like a session name). Required by every action."},
            "task": {"type": "string",
                     "description": "add: the new task's name. assign/done/fail: the task id "
                                    "(e.g. 't3', from add or list)."},
            "deps": {"type": "array", "items": {"type": "string"},
                     "description": "add: task ids that must be 'done' before this is ready. Omit "
                                    "for none."},
            "worker": {"type": "string",
                       "description": "assign: the worker pid, OR a registered participant name/"
                                      "role (a peer session). A live peer is pinged; a dormant one "
                                      "is spawned from its session file; a peer live elsewhere gets "
                                      "a board note. A bare label with no participant just records "
                                      "the assignee."},
            "role": {"type": "string",
                     "description": "participant: the peer's role label (defaults to its name). "
                                    "assign uses 'worker' as the participant name; this is only "
                                    "for action='participant' registration."},
            "note": {"type": "string",
                     "description": "Optional free text: driver->worker context on assign, or "
                                    "worker->driver why on fail."},
            "result_ref": {"type": "string",
                           "description": "done: pointer to the work product (e.g. the worker's "
                                          "result file). Collected by reconcile."},
            "status": {"type": "string",
                       "description": "list: filter to one status (open|assigned|done|failed), or "
                                      "'ready' for the assignable ones."},
            "query": {"type": "string",
                      "description": "list: only tasks whose id/name/note contains this text."},
        },
        "required": ["action", "board"],
    },
    "safe": True,
    # driver_only: the board is pure driver-orchestration state - a JSON doc under the
    # DRIVER's config dir (workers/taskboards/<name>.json) that reconcile + any peer
    # session read from THERE. It has no dependency on a connected workspace. Without
    # this flag a /connect routes the tool to the remote --tool-exec executor, which (a)
    # never runs setup() so the tool's core hooks (_H) are unset -> "board tool is not
    # initialised", and (b) would write the JSON to the REMOTE box, invisible to the
    # driver's reconcile. Pin it to the driver, like brave_search and dispatch_worker.
    "driver_only": True,
}


def setup(lc, cfg):
    """Capture the core task-board primitives + cfg (for the driver gate) + colours. A
    tool's run() gets no lc, so everything it needs is stashed on _H here."""
    for k in ("_taskboard_create", "_taskboard_load", "_taskboard_save", "_taskboard_add",
              "_taskboard_ready", "_taskboard_set_status", "_taskboard_reconcile",
              "_taskboard_mutate", "_taskboards_list", "_tb_task", "worker_inject",
              "_taskboard_participant_upsert", "_taskboard_participant",
              "_participant_resolve", "spawn_peer", "peer_inject", "_my_session_name",
              "dim", "bold", "green", "cyan"):
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


def _push(worker, text):
    """Best-effort ping to a live worker (assign/done auto-notify so it need not poll the
    board). No-op unless dispatch_worker registered its inject bridge and `worker` is a
    live pid. Returns True if delivered. Never raises."""
    fn = _H.get("worker_inject")
    if not fn or not worker:
        return False
    try:
        return bool(fn(str(worker), text))
    except Exception:
        return False


def _notify_assigner(board_name, tid, action, note, assigned_by, task_name):
    """Wake whoever assigned `tid` that it is now done/failed - the completion return leg
    (mirrors assign's outbound wake). Returns a short tail describing what happened (for the
    tool's reply), "" if there was nobody/nothing to do. Never raises. A spawned assigner is
    a WORKER (worker_depth>0) and assign is driver-only, so waking a dormant assigner can
    never cascade into a fresh round of auto-assigns - the driver-gate bounds it."""
    handle = (assigned_by or "").strip()
    if not handle:
        return ""                              # no return address recorded (pre-0.10.31 task)
    me = (_H.get("_my_session_name") or (lambda: ""))()
    if handle == me:
        return ""                              # finisher IS the assigner - don't wake yourself
    verb = "done" if action == "done" else "FAILED"
    ping = (f"[board '{board_name}'] task {tid} ({task_name}) you assigned is now {verb}."
            + (f" Note: {note}" if note else "")
            + " React: fold it into your report / decide the next step (action='list' to see the board).")
    resolve = _H.get("_participant_resolve")
    res = resolve(handle) if resolve else {"state": "missing"}
    state = res.get("state")
    if state == "live-here":
        pinj = _H.get("peer_inject")
        ok = bool(pinj and pinj(handle, ping)) or _push(res.get("pid"), ping)
        return f" (assigner '{handle}' pinged)" if ok else f" (assigner '{handle}' live but push failed; it will read the board)"
    if state == "live-elsewhere":
        return f" (assigner '{handle}' is live on {res.get('host')}; not woken - it will see it on the board)"
    if state == "missing":
        return f" (assigner '{handle}' has no session file; cannot notify - it will see it on the board if reopened)"
    # dormant: spawn it to conduct (the AFK relay - the assigner wakes, fields the result,
    # drives the next step). Bounded by the driver-gate as noted above.
    spawn = _H.get("spawn_peer")
    out = spawn(handle, ping) if spawn else ""
    return (f" (assigner '{handle}' was dormant; spawned to conduct)"
            if out and "error" not in out.lower()
            else f" (assigner '{handle}' is dormant; could not spawn - {out or 'dispatch_worker not enabled'})")


# Status -> checkbox glyph, so a board reads like the pinned PLAN (GOAL + a '- [ ]' list):
#   [ ] not started (open)   [~] in flight (assigned)   [x] done   [!] failed
# A blocked task (open but deps unmet) keeps [ ] and is tagged '(blocked: ...)'.
_BOX = {"done": "[x]", "assigned": "[~]", "failed": "[!]", "open": "[ ]", "blocked": "[ ]"}


def _fmt_task(t, ready_ids, blocked_ids=frozenset()):
    """One task as a plan-style checkbox line for list/find output:
        - [x] t1  <name>  @annota (by unity)  -> ref   - note
    """
    tid = t.get("id", "?")
    st = t.get("status", "?")
    box = _BOX.get(st, "[ ]")
    who = f"  @{t['assignee']}" if t.get("assignee") not in (None, "") else ""
    by = f" (by {t['assigned_by']})" if t.get("assigned_by") else ""
    deps = t.get("deps") or []
    if tid in blocked_ids and deps:
        state = f"  (blocked: deps {','.join(deps)})"
    elif deps:
        state = f"  (deps {','.join(deps)})"
    else:
        state = ""
    if tid in ready_ids:
        state += "  READY"
    rr = f"  -> {t['result_ref']}" if t.get("result_ref") else ""
    note = f"\n        - {t['note']}" if t.get("note") else ""
    return f"  - {box} {tid}  {t.get('name','')}{who}{by}{state}{rr}{note}"


def _render(name, board):
    """A board's full task list, plan-style (checkbox lines + GOAL/counts header)."""
    ready, blocked = _H["_taskboard_ready"](board)
    ready_ids = {t.get("id") for t in ready}
    blocked_ids = {t.get("id") for t in blocked}
    meta = board.get("meta", {})
    counts = meta.get("counts", {})
    head = _H["bold"](f"board '{name}'") + _H["dim"](
        f"  ({', '.join(f'{k}:{v}' for k, v in counts.items() if v)})" if counts else "  (empty)")
    lines = [head]
    if meta.get("title"):
        lines.append(_H["bold"](f"GOAL: {meta['title']}"))
    for t in board.get("tasks", []):
        lines.append(_fmt_task(t, ready_ids, blocked_ids))
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
        board, err = _H["_taskboard_create"](name,
                                              owner=str(getattr(_H.get("cfg"), "worker_board_session", "") or ""))
        if err:
            return f"error: {err}"
        return f"created board '{name}'. Add tasks with action='add'."

    # Every other action operates on an existing board.
    board = _H["_taskboard_load"](name)
    if board is None:
        return f"error: no board named '{name}' (create it first with action='create')."

    if action == "list":
        ready, _ = _H["_taskboard_ready"](board)
        ready_ids = {t.get("id") for t in ready}
        q = (args.get("query") or "").strip().lower()
        if q:
            hits = [t for t in board.get("tasks", [])
                    if q in str(t.get("id", "")).lower()
                    or q in str(t.get("name", "")).lower()
                    or q in str(t.get("note", "")).lower()]
            if not hits:
                return f"board '{name}': no task matches '{q}'."
            return "\n".join([_H["bold"](f"board '{name}' matches for '{q}':")]
                             + [_fmt_task(t, ready_ids) for t in hits])
        flt = (args.get("status") or "").strip().lower()
        if flt == "ready":
            if not ready:
                return f"board '{name}': no tasks are ready to assign."
            return "\n".join([_H["bold"](f"board '{name}' ready:")]
                             + [_fmt_task(t, ready_ids) for t in ready])
        if flt:
            sel = [t for t in board.get("tasks", []) if t.get("status") == flt]
            if not sel:
                return f"board '{name}': no tasks with status '{flt}'."
            return "\n".join([_H["bold"](f"board '{name}' [{flt}]:")]
                             + [_fmt_task(t, ready_ids) for t in sel])
        return _render(name, board)

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

    if action == "participant":
        # Register/refresh a peer, or list who's registered. Driver-only to WRITE (a driver
        # pre-declares an expected role; a peer self-registers via the core hook). The LIST
        # form (no name given) is open, like list/reconcile.
        pname = (args.get("worker") or args.get("task") or "").strip()
        role = (args.get("role") or "").strip() or None
        if not pname:
            parts = board.get("participants", [])
            if not parts:
                return f"board '{name}': no participants registered."
            lines = [_H["bold"](f"board '{name}' participants:")]
            for p in parts:
                # Liveness is resolved on demand (never stored): live-here/elsewhere/dormant/missing.
                res = _H["_participant_resolve"](p.get("name", ""))
                note = f"  - {p['note']}" if p.get("note") else ""
                lines.append(f"  {p.get('name')} role={p.get('role')} "
                             f"({res.get('state', '?')}){note}")
            return "\n".join(lines)
        if not _is_driver():
            return _driver_only(action)

        def _do_reg(bd):
            return _H["_taskboard_participant_upsert"](bd, pname, role=role,
                                                       note=(args.get("note") or "").strip() or None)

        board2, rec, err = _H["_taskboard_mutate"](name, _do_reg)
        if err:
            return f"error: {err}"
        return (f"registered participant '{rec['name']}' (role={rec['role']}). "
                f"Assign it work with action='assign' worker='{rec['name']}'.")

    if action == "add":
        if not _is_driver():
            return _driver_only(action)
        task_name = (args.get("task") or "").strip()
        if not task_name:
            return "error: action='add' needs a 'task' (its name/description)."
        deps = [str(d).strip() for d in (args.get("deps") or []) if str(d).strip()]
        note = (args.get("note") or "").strip()

        def _do_add(bd):
            # Mutate the FRESHLY-reloaded board under lock so a concurrent add can't hand us a
            # stale id (the t1/t1 collision) or lose our task on save.
            return _H["_taskboard_add"](bd, task_name, deps=deps, note=note)

        board2, task, err = _H["_taskboard_mutate"](name, _do_add)
        if err:
            return f"error: {err}"
        ready, _ = _H["_taskboard_ready"](board2)
        rflag = "READY" if task["id"] in {t.get("id") for t in ready} else "blocked (deps not done)"
        return f"added {task['id']} '{task_name}'{(' deps=' + ','.join(deps)) if deps else ''} -> {rflag}."

    if action == "assign":
        if not _is_driver():
            return _driver_only(action)
        tid = (args.get("task") or "").strip()
        worker = (args.get("worker") or "").strip()
        if not tid or not worker:
            return "error: action='assign' needs a 'task' id and a 'worker'."
        note = (args.get("note") or "").strip()
        me = (_H.get("_my_session_name") or (lambda: ""))()

        def _do_assign(bd):
            t = _H["_tb_task"](bd, tid)
            if not t:
                return None, f"unknown task id '{tid}'."
            ready, _ = _H["_taskboard_ready"](bd)
            if tid not in {r.get("id") for r in ready} and t.get("status") == "open":
                # open but not ready = deps unmet; refuse (assigning it wastes a worker that
                # would sit on unmet deps - the whole point of the DAG).
                unmet = [d for d in t.get("deps", [])
                         if (_H["_tb_task"](bd, d) or {}).get("status") != "done"]
                return None, (f"{tid} is not ready - it depends on {', '.join(unmet)} which "
                              f"is not done. Assign a READY task (action='list' status='ready').")
            # Stamp assigned_by = the session doing the assigning (our own lock name), the
            # return address a later done/fail wakes back. "" (incognito/unnamed) -> no stamp.
            _H["_taskboard_set_status"](bd, tid, "assigned", assignee=worker,
                                        note=note or None, assigned_by=me or None)
            return t.get("name", ""), ""
        board2, tname, err = _H["_taskboard_mutate"](name, _do_assign)
        if err:
            return f"error: {err}"
        ping = (f"[board '{name}'] You are assigned {tid}: {tname}."
                + (f" Note: {note}" if note else "")
                + " Do this task, then mark it done on the board (action='done').")
        # If 'worker' names a registered PARTICIPANT (a peer session), resolve its live
        # address and act on the peer's lifecycle; else fall back to the plain pid push (1a).
        part = _H.get("_taskboard_participant")
        prec = part(board2, worker) if part else None
        if prec:
            handle = prec.get("name")
            res = _H["_participant_resolve"](handle)
            state = res.get("state")
            if state == "live-here":
                # A live-here participant is an independent PEER session, not a worker we
                # dispatched - so reach it via its inbox (peer_inject -> its wake hook),
                # NOT worker_inject (which only knows THIS session's workers). Fall back to
                # a worker push in case the handle happens to name one of our own workers.
                pinj = _H.get("peer_inject")
                pushed = bool(pinj and pinj(handle, ping)) or _push(res.get("pid"), ping)
                tail = " (peer pinged)" if pushed else " (peer live but push failed; it will read the board)"
            elif state == "live-elsewhere":
                # Never spawn from a live session file (double-writer lobotomy) - leave a note.
                tail = (f" (peer '{handle}' is live on {res.get('host')}; not spawned - it will "
                        f"see the assignment on the board)")
            elif state == "missing":
                # Session file is gone (user deleted it). Keep the roster entry (a silent prune
                # would confuse the model) but say plainly why the peer can't be reached.
                tail = (f" (peer '{handle}' has no session file - deleted? cannot spawn it; "
                        f"re-create the session or assign a different peer)")
            else:  # dormant -> wake it by spawning a worker from its session file
                spawn = _H.get("spawn_peer")
                out = spawn(handle, ping) if spawn else ""
                tail = (f" (peer '{handle}' was dormant; spawned from its session)"
                        if out and "error" not in out.lower()
                        else f" (peer '{handle}' is dormant; could not spawn - {out or 'dispatch_worker not enabled'})")
            return f"assigned {tid} '{tname}' to {worker}.{tail}"
        # Not a participant: plain worker pid push (1a).
        pushed = _push(worker, ping)
        tail = " (worker pinged)" if pushed else ""
        return f"assigned {tid} '{tname}' to worker {worker}.{tail}"

    if action in ("done", "fail"):
        tid = (args.get("task") or "").strip()
        if not tid:
            return f"error: action='{action}' needs a 'task' id."
        # done/fail: a worker may call for the task it was assigned (or the driver, for any).
        note = (args.get("note") or "").strip()
        result_ref = (args.get("result_ref") or "").strip() or None

        def _do_finish(bd):
            t = _H["_tb_task"](bd, tid)
            if not t:
                return None, f"unknown task id '{tid}'."
            if action == "done":
                _H["_taskboard_set_status"](bd, tid, "done", note=note or None,
                                            result_ref=result_ref)
            else:  # fail
                _H["_taskboard_set_status"](bd, tid, "failed", note=note or None)
            # Hand back what the notify-back needs: who assigned it + the task name.
            return {"assigned_by": t.get("assigned_by") or "",
                    "name": t.get("name", "")}, ""

        board2, meta, err = _H["_taskboard_mutate"](name, _do_finish)
        if err:
            return f"error: {err}"
        msg = f"marked {tid} DONE." if action == "done" else f"marked {tid} FAILED."
        # On a done, surface what that unblocked so the driver knows what to assign next.
        if action == "done":
            ready, _ = _H["_taskboard_ready"](board2)
            newly = sorted(r.get("id") for r in ready)
            if newly:
                msg += f" ready now: {', '.join(newly)}."
        # Notify-back: a status change MUST reach whoever assigned the task - they're busy
        # doing other work and need it to assemble their report (a done that's one line of a
        # report, or a fail that leaves a hole, is worthless if it never wakes them). Wake the
        # assigner the same way an assign wakes an assignee: peer_inject if live-here, spawn if
        # dormant, passive board note if live-elsewhere/missing. Skip if the finisher IS the
        # assigner (don't wake yourself), or if nobody was recorded.
        msg += _notify_assigner(name, tid, action, note,
                                (meta or {}).get("assigned_by", ""),
                                (meta or {}).get("name", ""))
        return msg

    return f"error: unknown action '{action}'."
