# Bundled BUILTIN tools - the core file-editing + command surface (read_file,
# list_files, search_files, apply_diff, write_file, run_command). Extracted from
# lean_coder.py so the core stays the contract (leash tiers, dispatch, confirm policy,
# remote protocol) while the weight lives in the tool dir. This file is bundled
# (shipped with the code), always-on, NOT toggleable in /tools.
#
# Shape mirrors providers/ollama.py: the module declares its data + impl, and a
# helpers-only setup(lc, cfg) injects the stable core helpers this code resolves as
# plain module-level names (see _INJECT). The Tools INSTANCE (held by core as
# agent.tools) carries changed_files + the ignore matcher, exactly as before.
#
# Each entry in BUILTIN_TOOLS declares a `tier`:
#   read  -> rides at /leash r+   (read-only, no side effects)
#   write -> rides at /leash rw+  (edits files)
#   exec  -> rides at /leash rwe  (runs commands)
# Core DERIVES its SAFE_TOOLS / _WRITE_TIER / _EXEC_TIER / EXEC_TOOLS tuples from these.

import os
import re
import time
import subprocess
from pathlib import Path

# --- injected by setup() from core (resolved as plain names below) -----------
# Stable core helpers/constants this module's code reaches for. setup() copies each
# from the core module globals so the moved code runs unchanged. Mutable state is
# never snapshotted here - everything below is a function or an immutable constant.
_INJECT = (
    "IgnoreMatcher", "resolve_in_project", "_parse_search_replace", "_confirm_action",
    "_bg_running", "_bg_register", "_bg_status_items", "_bg_status_msg",
    "_bg_spawn_detached",
    "READ_MAX_LINES", "READ_HEAD", "READ_TAIL", "TREE_DEFAULT_DEPTH", "TREE_MAX_ENTRIES",
    "SEARCH_MAX_MATCHES", "SEARCH_LINE_MAX", "SEARCH_MAX_FILE_BYTES",
    "OUTPUT_MAX_CHARS", "OUTPUT_HEAD", "OUTPUT_TAIL",
    "CONFIG_DIR",
)


class Tools:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ignore = IgnoreMatcher(cfg.cwd)
        self.changed_files: set = set()

    # --- read-only -------------------------------------------------------
    def read_file(self, path: str, start=None, end=None, lines=None) -> str:
        p = resolve_in_project(self.cfg, path)
        if not p.is_file():
            return f"error: no such file: {path}"
        try:
            text = p.read_text(errors="replace")
        except Exception as e:
            return f"error reading {path}: {e}"
        alllines = text.splitlines()
        n = len(alllines)
        # tolerate a `lines=[start, end]` form (some models emit that)
        if lines and isinstance(lines, (list, tuple)) and len(lines) == 2:
            start, end = lines[0], lines[1]
        # explicit line range (1-based, inclusive)
        if start is not None or end is not None:
            try:
                s = max(1, int(start)) if start is not None else 1
                e = min(n, int(end)) if end is not None else n
            except (TypeError, ValueError):
                return "error: start/end must be integers"
            if s > e:
                return f"error: start ({s}) > end ({e})"
            note = ""
            if e - s + 1 > READ_MAX_LINES:
                e = s + READ_MAX_LINES - 1
                note = f"\n…[range capped at {READ_MAX_LINES} lines; request a smaller range]…"
            body = _number(alllines[s - 1:e], s)
            return (body + note) if body else "(no lines in range)"
        # whole file, head/tail truncated if very large
        if n > READ_MAX_LINES:
            head = _number(alllines[:READ_HEAD], 1)
            tail = _number(alllines[-READ_TAIL:], n - READ_TAIL + 1)
            return head + f"\n…[truncated {n - READ_HEAD - READ_TAIL} lines]…\n" + tail
        return _number(alllines, 1) or "(empty file)"

    def list_files(self, path: str = "") -> str:
        base = resolve_in_project(self.cfg, path) if path else self.cfg.cwd
        if not base.exists():
            return f"error: no such path: {path}"
        if base.is_file():
            return path
        out, count = [], 0
        root = self.cfg.cwd.resolve()

        def _rel(p):
            # cwd-relative when the entry is inside the project (the common case);
            # otherwise the absolute path. A naive rootlen-slice mangled names for a
            # listing OUTSIDE cwd (e.g. an absolute /var/log) into garbage like 'g/f'.
            rp = p.resolve()
            try:
                return rp.relative_to(root).as_posix()
            except ValueError:
                return rp.as_posix()

        def walk(d, depth: int):
            nonlocal count
            if depth > TREE_DEFAULT_DEPTH or count >= TREE_MAX_ENTRIES:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return
            for e in entries:
                if count >= TREE_MAX_ENTRIES:
                    out.append("…[more entries truncated]…")
                    return
                if self.ignore.ignored(e):
                    continue
                rel = _rel(e)
                out.append(rel + ("/" if e.is_dir() else ""))
                count += 1
                if e.is_dir():
                    walk(e, depth + 1)

        walk(base, 1)
        return "\n".join(out) if out else "(empty)"

    def search_files(self, pattern: str, path: str = "", context: int = 0) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"error: bad regex: {e}"
        try:
            ctx_lines = max(0, min(int(context), 10))  # cap at 10 to avoid bloat
        except (TypeError, ValueError):
            ctx_lines = 0
        base = resolve_in_project(self.cfg, path) if path else self.cfg.cwd
        hits, count = [], 0
        files = [base] if base.is_file() else _iter_files(base, self.ignore)
        for f in files:
            if count >= SEARCH_MAX_MATCHES:
                hits.append("…[more matches truncated]…")
                break
            try:
                # Never slurp a whole file into RAM (a multi-GB VM image / DB / log
                # under a broad base like $HOME would exhaust memory -> swap death).
                # Skip oversize files, and stream line-by-line instead of read_text().
                if f.stat().st_size > SEARCH_MAX_FILE_BYTES:
                    continue
                with f.open("r", errors="replace") as fh:
                    # Cheap binary sniff: a NUL byte in the first chunk -> skip.
                    head = fh.read(8192)
                    if "\x00" in head:
                        continue
                    fh.seek(0)
                    all_lines = fh.readlines() if ctx_lines else None
                    if all_lines is None:
                        fh.seek(0)
                    source = enumerate(all_lines, 1) if all_lines else enumerate(fh, 1)
                    # Track which context lines we've already emitted per file to
                    # avoid duplicating when hits are close together.
                    emitted = set()
                    for i, line in source:
                        line = line.rstrip("\n")
                        if rx.search(line):
                            fp = f.resolve()
                            try:
                                rel = fp.relative_to(self.cfg.cwd.resolve()).as_posix()
                            except ValueError:
                                rel = fp.as_posix()
                            if ctx_lines and all_lines:
                                # Emit context window around this hit.
                                start_c = max(1, i - ctx_lines)
                                end_c = min(len(all_lines), i + ctx_lines)
                                if emitted and start_c - 1 > max(emitted):
                                    hits.append(f"  ...")
                                for c in range(start_c, end_c + 1):
                                    if c not in emitted:
                                        marker = ">" if c == i else " "
                                        ctxt = all_lines[c - 1].rstrip("\n")[:SEARCH_LINE_MAX]
                                        hits.append(f"{marker} {rel}:{c}: {ctxt}")
                                        emitted.add(c)
                            else:
                                txt = line.strip()[:SEARCH_LINE_MAX]
                                hits.append(f"{rel}:{i}: {txt}")
                            count += 1
                            if count >= SEARCH_MAX_MATCHES:
                                break
            except Exception:
                continue
        return "\n".join(hits) if hits else "(no matches)"

    # --- mutating (gated by confirmation) --------------------------------
    def replace_lines(self, path: str, start: int, end: int, new_text: str = "") -> str:
        """Replace an inclusive line range (1-based) with new_text. Sidesteps
        apply_diff's delimiter-matching entirely: read_file gives you line numbers,
        this takes line numbers. Staleness guard: the range is validated against the
        file's CURRENT line count (end > n is refused), so an edit computed from an
        out-of-date read fails loudly instead of clobbering the wrong lines."""
        p = resolve_in_project(self.cfg, path)
        if not p.is_file():
            return f"error: no such file: {path}"
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            return "error: start and end must be integers"
        original = p.read_text(errors="replace")
        lines = original.splitlines(keepends=True)
        n = len(lines)
        if start < 1 or end < start:
            return f"error: invalid range (start={start}, end={end}, file has {n} lines)"
        if end > n:
            return f"error: end ({end}) exceeds file length ({n} lines)"
        # Build the new file: everything before start, the new text, everything after end.
        # Ensure new_text ends with a newline if replacing mid-file (keeps line structure).
        replacement = new_text
        if replacement and not replacement.endswith("\n") and end < n:
            replacement += "\n"
        before = lines[:start - 1]
        after = lines[end:]
        new = "".join(before) + replacement + "".join(after)
        if new == original:
            return "no changes (result identical to original)"
        if not _confirm_action(self.cfg, "write", path=path, original=original, new=new):
            return "user declined the edit"
        p.write_text(new)
        self.changed_files.add(path)
        new_count = len(replacement.splitlines()) if replacement else 0
        old_count = end - start + 1
        return (f"replaced lines {start}-{end} ({old_count} lines) with "
                f"{new_count} lines in {path}")

    def apply_diff(self, path: str, diff: str) -> str:
        p = resolve_in_project(self.cfg, path)
        if not p.is_file():
            return f"error: no such file: {path} (use write_file for new files)"
        blocks, perr = _parse_search_replace(diff)
        if perr:
            # Actionable failure: the block didn't PARSE (delimiter recognition),
            # so re-submitting the same thing loops. Echo back what the tool actually
            # received (head-capped) so the model can SEE the mangling, then point at
            # the escape hatch (rewrite the whole file) rather than a dead-end format
            # reprint. Markers must each be ALONE on their own line.
            got = diff if len(diff) <= 400 else diff[:400] + " …[+%d chars]" % (len(diff) - 400)
            return ("error: " + perr + "\n"
                    "Expected one or more blocks, each marker alone on its own line:\n"
                    "<<<<<<< SEARCH\n<exact old text>\n=======\n<new text>\n>>>>>>> REPLACE\n"
                    "--- what apply_diff received (verify the markers are on their own "
                    "lines, not indented or fenced) ---\n" + got + "\n"
                    "If the markers keep failing to parse, rewrite the whole file with "
                    "write_file instead.")
        original = p.read_text(errors="replace")
        new = original
        applied = 0
        for idx, (search, replace) in enumerate(blocks, 1):
            cnt = new.count(search)
            if cnt == 1:
                new = new.replace(search, replace, 1)
                applied += 1
                continue
            if cnt > 1:
                return (f"error: block {idx} SEARCH text is ambiguous in {path} "
                        f"(matched {cnt} times); no changes written. Add surrounding "
                        f"lines to the SEARCH text so it matches exactly one place.")
            # Exact match failed: try a whitespace-tolerant line match (small models
            # rarely reproduce indentation/trailing space byte-for-byte). Never
            # loosens uniqueness - we still require exactly one fuzzy hit.
            fnew, ferr = _fuzzy_apply(new, search, replace, idx, path)
            if ferr:
                return ferr
            new = fnew
            applied += 1
        if new == original:
            return "no changes (result identical to original)"
        if not _confirm_action(self.cfg, "write", path=path, original=original, new=new):
            return "user declined the edit"
        p.write_text(new)
        self.changed_files.add(path)
        return f"applied {applied} block(s) to {path}"

    def write_file(self, path: str, content: str) -> str:
        p = resolve_in_project(self.cfg, path)
        original = p.read_text(errors="replace") if p.is_file() else None
        if original == content:
            return "no changes (content identical)"
        if not _confirm_action(self.cfg, "write", path=path, original=original, new=content):
            return "user declined the write"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.changed_files.add(path)
        verb = "overwrote" if original is not None else "created"
        return f"{verb} {path} ({len(content.splitlines())} lines)"

    def run_command(self, cmd: str, notify_on_exit=None, heartbeat_timeout=None,
                    max_runtime=None, kill_on_max=None) -> str:
        bg = cmd.rstrip().endswith("&")        # trailing & = "background this"
        if not _confirm_action(self.cfg, "exec", cmd=cmd):
            return "user declined to run the command"
        if bg:
            return self._run_background(
                cmd.rstrip()[:-1].rstrip(),
                notify_on_exit=notify_on_exit, heartbeat_timeout=heartbeat_timeout,
                max_runtime=max_runtime, kill_on_max=kill_on_max)
        to = self.cfg.command_timeout
        try:
            # stdin=DEVNULL: never let a child read the live terminal. With the
            # composer's reader thread on cbreak stdin, an inherited fd makes a
            # subshell (or any stdin-reading command) hang fighting for keystrokes.
            r = subprocess.run(cmd, shell=True, cwd=self.cfg.cwd,
                               capture_output=True, text=True, timeout=to,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return (f"error: command timed out after {to}s (no output captured). "
                    f"For a long-running command, end it with ' &' to run it in the "
                    f"background - returns a pid immediately; poll with bg_status.")
        except Exception as e:
            return f"error running command: {e}"
        out = ""
        if r.stdout:
            out += r.stdout
        if r.stderr:
            out += ("\n[stderr]\n" if out else "") + r.stderr
        out = _truncate_output(out.rstrip())
        return f"exit {r.returncode}\n{out}" if out else f"exit {r.returncode} (no output)"

    def _run_background(self, cmd: str, kind: str = "task", idle_timeout=None,
                        notify_on_exit=None, heartbeat_timeout=None,
                        max_runtime=None, kill_on_max=None) -> str:
        """Launch a detached background process and return immediately. Plain
        run_command captures stdout/stderr, so a backgrounded child keeps the pipe
        open until it dies and run_command would block until the 300s timeout - the
        bug behind 'I told it to start a daemon and the turn froze'. Here we detach
        (own session, no controlling tty), redirect output to a log, and hand back
        the pid + log path so the model can track or stop it.

        `idle_timeout` (seconds) arms a self-terminating LEASE: an in-band watchdog
        kills the whole group if the '<log>.lease' file isn't bumped for that long
        (the parent bumps it once per turn - see _bg_bump_session). Caps runaway
        quota burn by a task whose parent dropped. None = no lease (runs to done).

        WATCHDOG/NOTIFY args (all optional, all file-based so they survive a reconnect):
          notify_on_exit  - push exit code + tail on completion even if the global
                            wake_on_bg_finish is off (recorded on the registry row).
          heartbeat_timeout (secs) - one-time '<log>.silent' alert if the log goes
                            quiet that long; NEVER kills (silence != dead).
          max_runtime (secs) - wall-clock ceiling; notify on trip, auto-kill unless
                            kill_on_max is False (then notify-only, keep running)."""
        if not cmd:
            return "error: nothing to run in the background"
        cap = self.cfg.bg_max_concurrent
        if cap and len(_bg_running(kind=None)) >= cap:
            return (f"error: at the background-task limit ({cap} running). Free a slot by "
                    f"stopping a finished/unneeded one with `kill <pid>` (use bg_status to "
                    f"list them), then retry.")
        def _posint(v):
            try:
                n = int(v)
                return n if n > 0 else None
            except (TypeError, ValueError):
                return None
        heartbeat_timeout = _posint(heartbeat_timeout)
        max_runtime = _posint(max_runtime)
        kill_on_max = True if kill_on_max is None else bool(kill_on_max)
        try:
            logdir = CONFIG_DIR / "bg"
            logdir.mkdir(parents=True, exist_ok=True)
            log = logdir / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
            exitf = str(log) + ".exit"
            p = _bg_spawn_detached(cmd, log, exitf, str(log) + ".lease",
                                   logf=str(log), idle_timeout=idle_timeout,
                                   max_runtime=max_runtime, kill_on_max=kill_on_max,
                                   heartbeat_timeout=heartbeat_timeout,
                                   cwd=str(self.cfg.cwd))
        except Exception as e:
            return f"error backgrounding command: {e}"
        _bg_register(p.pid, cmd, log, kind=kind, idle_timeout=idle_timeout,
                     notify_on_exit=notify_on_exit, heartbeat_timeout=heartbeat_timeout,
                     max_runtime=max_runtime, kill_on_max=kill_on_max)
        notes = []
        if idle_timeout:
            notes.append(f"lease: self-terminates if unattended > {idle_timeout}s")
        if heartbeat_timeout:
            notes.append(f"heartbeat: alerts if silent > {heartbeat_timeout}s (won't kill)")
        if max_runtime:
            notes.append(f"max_runtime {max_runtime}s"
                         + ("; auto-kills on trip" if kill_on_max else "; notify-only"))
        if notify_on_exit:
            notes.append("notify_on_exit: pushes result on completion")
        note = ("  ·  " + "  ·  ".join(notes)) if notes else ""
        return (f"started in background: pid {p.pid} (detached; killed when you exit "
                f"lean-coder)\noutput -> {log}\n"
                f"track: bg_status  (exit code -> '{exitf}')  ·  list/stop: /bg  "
                f"·  stop now: kill {p.pid}{note}")

    @staticmethod
    def _bg_wrap(cmd: str, exitf: str, leasef: str, idle_timeout=None,
                 logf=None, max_runtime=None, kill_on_max=True,
                 heartbeat_timeout=None) -> str:
        """Build the shell wrapper for a backgrounded command.

        Base: run cmd in a SUBSHELL '( ... )' (contains the user's '&&'/pipes AND any
        'exit' they run, so $? is still captured), then write the exit code to the
        '<log>.exit' sidecar - completion + pass/fail become discoverable by READING
        A FILE (survives an executor death / ssh reconnect; identical local/remote).

        With any WATCHDOG feature set, also run an in-band poller alongside the child.
        All are FILE-based (sidecars), so they survive a reconnect and read identically
        local/remote; all shell built-ins + `stat -c %Y` are on dash/bash/busybox/toybox
        and a `|| echo now` fallback means a stat failure reads as 'just now' (never a
        false trip). The watchdog channels:
          - LEASE (idle_timeout): the parent bumps '<log>.lease' once per turn; if its
            mtime goes unbumped for idle_timeout the parent is gone -> write exit 137 and
            KILL the group. Caps runaway quota by an orphaned task.
          - MAX_RUNTIME (max_runtime): wall-clock ceiling. On trip, write '<log>.maxrun'
            (the elapsed secs, for the notice) and, if kill_on_max, write exit 137 + KILL;
            else leave it running (notify-only). The ONLY output-independent kill.
          - HEARTBEAT (heartbeat_timeout): if the LOG file's mtime (bumped whenever the
            child writes) goes stale for heartbeat_timeout, write '<log>.silent' (the
            silence secs) ONCE and keep running - a watchdog that BARKS, never bites.
            Silence != dead (a long compile/network wait goes quiet), so it never kills;
            the human/agent decides. Re-arms if output resumes.
        $$ is the detached session leader => its pid is the pgid for the group kills."""
        # our own generated log path (no shell metachars) - safe to single-quote.
        def _int(v, d):
            try:
                return int(v)
            except (TypeError, ValueError):
                return d
        lease = _int(idle_timeout, 0) if idle_timeout else 0
        maxrt = _int(max_runtime, 0) if max_runtime else 0
        hb = _int(heartbeat_timeout, 0) if heartbeat_timeout else 0
        if not (lease or maxrt or hb):
            return f"( {cmd} ); __rc=$?; printf %s \"$__rc\" > '{exitf}'; exit $__rc"
        silentf = str(logf) + ".silent" if logf else exitf + ".silent"
        maxrunf = str(logf) + ".maxrun" if logf else exitf + ".maxrun"
        wins = [w for w in (lease or None, maxrt or None, hb or None) if w]
        chk = max(5, min(30, min(wins) // 4))     # a few checks per shortest window
        checks = ["now=$(date +%s); "]
        if lease:
            checks.append(
                f"m=$(stat -c %Y '{leasef}' 2>/dev/null || echo $now); "
                f"if [ $((now - m)) -ge {lease} ]; then "
                f"echo \"[lease expired: unattended > {lease}s; terminated]\"; "
                f"printf %s 137 > '{exitf}'; kill -KILL -$__pg 2>/dev/null; exit; fi; ")
        if maxrt:
            # On kill: write the exit sidecar + KILL - the finish scan reports it, so
            # do NOT also write '.maxrun' (that's a still-running alert; a dead job with
            # a leftover .maxrun would double-report as "still running"). On notify-only:
            # write '.maxrun' (the alert channel) and let it keep running.
            act = (f"echo \"[max_runtime {maxrt}s reached; terminated]\"; "
                   f"printf %s 137 > '{exitf}'; kill -KILL -$__pg 2>/dev/null; exit; "
                   if kill_on_max else
                   f"printf %s {maxrt} > '{maxrunf}'; "
                   f"echo \"[max_runtime {maxrt}s reached; still running (kill_on_max=false)]\"; ")
            checks.append(
                f"if [ $((now - __t0)) -ge {maxrt} ] && [ $__mr -eq 0 ]; then __mr=1; "
                f"{act}fi; ")
        if hb:
            checks.append(
                f"lm=$(stat -c %Y '{logf or exitf}' 2>/dev/null || echo $now); "
                f"if [ $((now - lm)) -ge {hb} ]; then "
                f"if [ $__hb -eq 0 ]; then __hb=1; printf %s $((now - lm)) > '{silentf}'; fi; "
                f"else __hb=0; fi; ")     # re-arm once output resumes
        loop = (
            f"__t0=$(date +%s); __pg=$$; touch '{leasef}'; "
            f"( {cmd} ) & __cpid=$!; "
            f"( __hb=0; __mr=0; while kill -0 $__cpid 2>/dev/null; do "
            + "".join(checks)
            + f"sleep {chk}; done ) & __wpid=$!; "
            f"wait $__cpid; __rc=$?; kill $__wpid 2>/dev/null; "
            f"[ -f '{exitf}' ] || printf %s \"$__rc\" > '{exitf}'; exit $__rc")
        return loop

    # --- read-only: check on backgrounded tasks ---------------------------
    def bg_status(self, pid=None) -> str:
        """Report on background tasks THIS session started (state / runtime / recent
        output). No pid -> list them all; a pid -> just that one. Plain tasks only -
        the filtering lives in the core helper."""
        owner = os.getpid()
        rows = _bg_status_items(owner)
        if pid is not None:
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                return "error: pid must be an integer"
            rows = [r for r in rows if r.get("pid") == pid]
        return _bg_status_msg(rows, pid=pid)


# --- tool helpers (moved with the class; only the Tools methods use them) -----

def _nearest_match(file_lines, search_lines):
    """Find the file region most similar to search_lines. Returns (start_index,
    similarity_percent) or None. Uses a cheap line-level similarity score to avoid
    pulling in difflib for every edit."""
    w = len(search_lines)
    if w == 0 or len(file_lines) < w:
        return None
    s_stripped = [l.strip() for l in search_lines]
    best_at, best_score = 0, -1
    for i in range(0, len(file_lines) - w + 1):
        matches = sum(1 for k in range(w) if file_lines[i + k].strip() == s_stripped[k])
        if matches > best_score:
            best_score = matches
            best_at = i
    if best_score < 1:
        return None
    pct = int(best_score * 100 / w)
    return (best_at, pct)


def _fuzzy_apply(text: str, search: str, replace: str, idx: int, path: str):
    """Whitespace-tolerant fallback for apply_diff when an exact SEARCH match
    fails. Small models rarely reproduce indentation/trailing whitespace
    byte-for-byte, so match SEARCH against the file line-by-line ignoring each
    line's leading/trailing whitespace. Uniqueness is still enforced (exactly one
    window may match) so we never silently edit the wrong place; the REPLACE text
    is re-indented to the file's actual indentation of the matched region.
    Returns (new_text, None) on success, or (None, error_str) on failure."""
    def norm(ln):
        return ln.strip()
    file_lines = text.split("\n")
    s_lines = search.split("\n")
    # drop a trailing empty line artifact from a trailing newline in SEARCH
    if s_lines and s_lines[-1] == "":
        s_lines = s_lines[:-1]
    if not s_lines:
        return None, (f"error: block {idx} SEARCH text not found in {path} "
                      f"(matched 0 times); no changes written.")
    s_norm = [norm(l) for l in s_lines]
    w = len(s_norm)
    hits = []
    for i in range(0, len(file_lines) - w + 1):
        if [norm(file_lines[i + k]) for k in range(w)] == s_norm:
            hits.append(i)
    if len(hits) == 0:
        # Near-miss: find the closest region so the model can see the drift.
        near = _nearest_match(file_lines, s_lines)
        hint = ""
        if near:
            at, score = near
            # Show a compact side-by-side of SEARCH vs actual.
            actual = file_lines[at:at + w]
            diff_lines = []
            for j, (sl, al) in enumerate(zip(s_lines, actual)):
                if sl.rstrip() != al.rstrip():
                    diff_lines.append(f"  line {at+j+1}:")
                    diff_lines.append(f"    yours:  {sl.rstrip()!r}")
                    diff_lines.append(f"    actual: {al.rstrip()!r}")
            if diff_lines:
                hint = ("\nNearest match at lines " + str(at + 1) + "-" +
                        str(at + w) + f" ({score}% similar):\n" +
                        "\n".join(diff_lines[:15]))  # cap output
        return None, (f"error: block {idx} SEARCH text not found in {path} "
                      f"(matched 0 times, even ignoring whitespace); no changes "
                      f"written. Re-read the file and copy the exact lines." + hint)
    if len(hits) > 1:
        return None, (f"error: block {idx} SEARCH text is ambiguous in {path} "
                      f"(matched {len(hits)} times ignoring whitespace); no changes "
                      f"written. Add surrounding lines so it matches one place.")
    at = hits[0]
    # Re-indent REPLACE to the matched region: use the leading whitespace of the
    # first matched file line as the base so the edit keeps the file's own style.
    matched_indent = file_lines[at][:len(file_lines[at]) - len(file_lines[at].lstrip())]
    r_lines = replace.split("\n")
    if r_lines and r_lines[-1] == "":
        r_lines = r_lines[:-1]
    # Preserve the REPLACE block's OWN internal indentation (nested blocks, continued
    # statements) - only SHIFT the whole block to the matched region's indent. Strip
    # the block's common leading whitespace, then prepend matched_indent. Stripping
    # per-line to bare (the old bug) flattened nesting and wrote broken code.
    _base = min((len(l) - len(l.lstrip()) for l in r_lines if l.strip()), default=0)
    reindented = [(matched_indent + l[_base:]) if l.strip() else l for l in r_lines]
    new_lines = file_lines[:at] + reindented + file_lines[at + w:]
    return "\n".join(new_lines), None


def _number(lines, start: int) -> str:
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{str(start + i).rjust(width)}\t{ln}"
                     for i, ln in enumerate(lines))


def _iter_files(base, ignore):
    stack = [base]
    while stack:
        d = stack.pop()
        try:
            for e in d.iterdir():
                if ignore.ignored(e):
                    continue
                if e.is_dir():
                    stack.append(e)
                elif e.is_file():
                    yield e
        except (PermissionError, OSError):
            continue


def _truncate_output(s: str) -> str:
    if len(s) <= OUTPUT_MAX_CHARS:
        return s
    dropped = len(s) - OUTPUT_HEAD - OUTPUT_TAIL
    return (s[:OUTPUT_HEAD] + f"\n…[truncated {dropped} chars]…\n" + s[-OUTPUT_TAIL:])


# --- the tool contract: schema + capability tier per builtin -----------------
# (description text is the model-facing tool doc; kept verbatim from the old core
#  TOOLS list. tier maps to the /leash ladder - see header.)
BUILTIN_TOOLS = [
    {"name": "read_file", "tier": "read",
     "description": "Read a file (line-numbered); read before editing. start/end read a slice; large reads truncate - use start/end to page a specific portion, or search_files to locate first. Re-reading an unchanged file wastes context; only re-read if it may have changed.",
     "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "File path, relative to the project dir."},
                       "start": {"type": "integer", "description": "First line to read (1-based, optional)."},
                       "end": {"type": "integer", "description": "Last line to read, inclusive (optional)."}},
        "required": ["path"]}},
    {"name": "list_files", "tier": "read",
     "description": "List a directory (shallow, ignore-aware) to discover structure. No path = project root.",
     "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "Directory to list (optional; defaults to project root)."}},
        "required": []}},
    {"name": "search_files", "tier": "read",
     "description": "Find text across files by regex; returns file:line matches (capped). Locate where something is defined/used.",
     "parameters": {"type": "object",
        "properties": {"pattern": {"type": "string", "description": "Regex to search for."},
                       "path": {"type": "string", "description": "File or directory to search (optional; defaults to project root)."},
                       "context": {"type": "integer", "description": "Lines of context around each match (0-10, like grep -C). Default 0."}},
        "required": ["pattern"]}},
    {"name": "apply_diff", "tier": "write",
     "description": ("Preferred way to edit a file: replace exact spans via SEARCH/REPLACE blocks. SEARCH must "
                     "match verbatim (read_file first); on mismatch nothing is written - re-read and retry. "
                     "Format per block, each marker ALONE on its own line: '<<<<<<< SEARCH', old text, "
                     "'=======', new text, '>>>>>>> REPLACE'. Multiple blocks applied in order."),
     "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "File to edit (must already exist)."},
                       "diff": {"type": "string", "description": "One or more SEARCH/REPLACE blocks."}},
        "required": ["path", "diff"]}},
    {"name": "replace_lines", "tier": "write",
     "description": ("Replace a line range in a file. Simpler than apply_diff: read_file shows line numbers, "
                     "pass them here. Use for 'I just read lines N-M, replace them with this'. "
                     "1-based inclusive range. Omit new_text to delete lines."),
     "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "File to edit (must already exist)."},
                       "start": {"type": "integer", "description": "First line to replace (1-based)."},
                       "end": {"type": "integer", "description": "Last line to replace (inclusive)."},
                       "new_text": {"type": "string", "description": "Replacement text (omit or empty string to delete the lines)."}},
        "required": ["path", "start", "end"]}},
    {"name": "write_file", "tier": "write",
     "description": "Create or overwrite a whole file. Use for NEW files or a full rewrite; to change an existing file prefer apply_diff.",
     "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "File path to create or overwrite."},
                       "content": {"type": "string", "description": "Full new contents of the file."}},
        "required": ["path", "content"]}},
    {"name": "run_command", "tier": "exec",
     "description": "Run a non-interactive shell command (builds, tests, git, file ops); returns stdout/stderr (truncated - if clipped, pipe to a file or use grep/head/tail to read specific parts). Fresh shell each call - cd/env don't persist, chain with '&&'. Times out after a fixed limit - for a long-running or daemon process end the command with ' &' to background it (returns a pid immediately; poll with bg_status). For sudo/password/interactive use ask_user_to_run; for a shell that stays open use shell_session.",
     "parameters": {"type": "object",
        "properties": {"cmd": {"type": "string", "description": "The shell command to run."},
                       "notify_on_exit": {"type": "boolean", "description": "Bg jobs (' &') only: push exit code + tail back to you on completion."},
                       "heartbeat_timeout": {"type": "integer", "description": "Bg jobs only: alert once if no output for N secs. Never kills (silence != dead)."},
                       "max_runtime": {"type": "integer", "description": "Bg jobs only: wall-clock ceiling (secs); notify + auto-kill on trip."},
                       "kill_on_max": {"type": "boolean", "description": "With max_runtime: false = notify only, keep running (default true = kill)."}},
        "required": ["cmd"]}},
    {"name": "bg_status", "tier": "read",
     "description": "Poll background tasks (started with a trailing ' &' on run_command): state, runtime, last output. No pid = all; a pid = just that one.",
     "parameters": {"type": "object",
        "properties": {"pid": {"type": "integer", "description": "A specific task's pid (optional; omit to list all)."}},
        "required": []}},
]


def _schema(t):
    """The function-call tool schema the model sees (drops the internal `tier`)."""
    return {"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}


# What core consumes: the model-facing schemas (in declared order) + a name->tier map.
TOOL_SCHEMAS = [_schema(t) for t in BUILTIN_TOOLS]
TIERS = {t["name"]: t["tier"] for t in BUILTIN_TOOLS}


def setup(lc, cfg=None):
    """Helpers-only: inject the stable core helpers/constants this module resolves as
    plain names (see _INJECT). Called once by core after its helpers are defined, and
    by the remote executor. Absent names are skipped (guarded), so a partial core
    (e.g. a test harness) still loads the data + schemas."""
    g = globals()
    for k in _INJECT:
        if k in lc:
            g[k] = lc[k]
