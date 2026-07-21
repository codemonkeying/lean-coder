"""shell_session - a PERSISTENT interactive shell the model can hold open across
many tool calls, instead of the one-shot, state-less run_command (fresh shell each
call, cwd/env gone the moment it returns).

WHY: some work needs a shell that STAYS open - an interactive prompt you keep
typing into: a reverse/bind shell caught on a listener, an `ssh` into another box,
`python -i` / a db shell / `mysql`, `sudo -s`, a REPL, watching an interactive
installer. run_command can LAUNCH such a thing (even detached with ' &') but it
can't type into it afterwards - each call is a new process with no handle to the
last. This tool keeps one child process behind a pty for its whole life, so the
model can send input, read what comes back, and keep going turn after turn.

HOW (stdlib ONLY - pty + os + select + subprocess; NO tmux/screen, so it works on
a phone / Termux / a stripped box just the same):
  - `start`  opens a pty (pty.openpty), spawns a child shell (or any command) on
             the far end, keeps the master fd + Popen in a per-session dict that
             lives in THIS process for the session's lifetime;
  - `send`   writes text (+ newline) into the pty, then drains the reply;
  - `read`   drains any output the child produced since last time;
  - `close`  ends the child and frees the fd;
  - `list`   shows live sessions.
Output handed to the model is shaped through core's _clean_captured (ANSI stripped;
a full-screen TUI becomes a short note, not escape-paint).

REMOTE / HTB: to hold an interactive shell on another box, START it with
command="ssh user@target" (or any reverse-shell handler like "nc -lvnp 4444").
The pty + all state stay HERE on the driver, so this tool is driver_only - it is
never pushed to a remote executor (that would put the fd on the wrong machine).

CAVEAT: a session lives only as long as this lean-coder process (the held fd is
the persistence - there's no external multiplexer to outlive a restart). For an
agent driving a shell across turns that's exactly the right lifetime. Output that
piles up between reads sits in the OS pty buffer (~64k); drain with `read` if a
child is chatty. This is a run-commands capability, so it is NOT safe (confirms
unless approval is auto/armed), and needs the 'rwe' leash like run_command.
"""
import os
import re
import select
import shlex
import signal
import subprocess
import time

TOOL = {
    "name": "shell_session",
    "description": (
        "A shell that stays open across tool calls (unlike run_command's fresh shell each "
        "call). For anything you keep typing into: an ssh/reverse shell, python -i, a REPL, "
        "sudo -s. Actions: start (optional command, e.g. 'ssh user@host'; returns an id), "
        "send (input + reply), read, close, list."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "send", "read", "close", "list"],
                       "description": "start | send | read | close | list"},
            "id": {"type": "string",
                   "description": "Session id (from start). Required for send/read/close."},
            "command": {"type": "string",
                        "description": "start only: the command to run in the session "
                                       "(e.g. 'ssh user@host', 'python3 -i', 'nc -lvnp 4444'). "
                                       "Omit for a plain interactive shell."},
            "input": {"type": "string",
                      "description": "send only: text typed into the session. A newline is added "
                                     "unless you end with one (end with '' to send a bare keystroke)."},
            "timeout": {"type": "number", "default": 4,
                        "description": "send/read only: seconds to wait for output "
                                       "(default 4). Raise it for a slow command."},
            "raw": {"type": "boolean",
                    "description": "start only: force drain mode. Omit to auto-detect "
                                   "(a shell/ssh gets deterministic marker draining; a "
                                   "REPL/listener like python -i or nc uses timing). Set "
                                   "true for a REPL the auto-detect missed, false to force "
                                   "marker mode."},
        },
        "required": ["action"],
    },
    "driver_only": True,   # the pty fd lives in the driver process; never run on a remote executor
    # NOT safe: this runs commands / drives a live shell, so it confirms unless
    # approval is auto/armed (same policy as run_command).
}

# Captured from core in setup(); a tool's run() gets no lc. setup() is driver-only,
# which is where run() also fires (driver_only tool), so these are always present.
_H = {}
# Live sessions: id -> {proc, fd, cmd, started, where}. Lives for this process's life.
_SESSIONS = {}
_SEQ = 0

# Tunables.
_READ_QUIET = 0.3      # once output has arrived, stop after this long with none more
_READ_DEFAULT = 4.0    # default overall wait for a send/read
_MAX_SESSIONS = 8      # guard against runaway session count

# Sentinel-marker draining (the pexpect / Ansible-shell technique). For a POSIX
# shell/ssh session we append a command that prints a UNIQUE end-marker plus the
# exit code AFTER the real command finishes; _drain then reads until it sees that
# marker instead of guessing with a quiet-gap timer. Completion becomes
# DETERMINISTIC: a command that pauses mid-flight (sleep, a slow remote, bursty
# output) no longer returns early or lands a call late, and we get the exit code
# for free. The nonce is per-session so a marker can't collide with real output.
# A raw/interactive session (python -i, nc, a REPL, a password prompt) has no
# shell to run the marker, so it keeps the quiet-gap/timeout path (mode="raw").
_MARK_PREFIX = "__LC_DONE_"

# Interactive/raw children we must NOT inject a shell marker into - they have no
# POSIX printf/$?. Matched against the basename of the start command's first word.
_RAW_HINTS = ("python", "python3", "python2", "node", "irb", "ruby", "psql",
              "mysql", "sqlite3", "mongo", "redis-cli", "nc", "ncat", "netcat",
              "telnet", "gdb", "lldb", "julia", "ghci", "iex", "php")


def _mode_for(command):
    """Decide a new session's drain mode from its start command. A plain shell or
    an ssh/sh -c lands on a POSIX prompt -> "shell" (marker draining). A REPL / raw
    listener has no shell to run the marker -> "raw" (quiet-gap draining). An
    unknown command errs to "raw" (never inject a marker where it'd echo as literal
    text). ssh is "shell": the far end is a login shell."""
    if not command:
        return "shell"                     # bare interactive shell
    try:
        first = os.path.basename(shlex.split(command)[0])
    except (ValueError, IndexError):
        return "raw"
    if first in ("ssh", "sh", "bash", "zsh", "dash", "ksh", "sudo", "su", "fish"):
        return "shell"
    return "raw"                           # REPL / listener / unknown -> no marker


def _collapse_cr(text):
    """Apply carriage-return overwrite semantics per line, the way a terminal shows
    it: an interactive shell's readline reprints the WHOLE line after every keystroke
    (each redraw prefixed by '\\r'), so a raw capture is '\\rp\\rpr\\rpri...'. Core's
    _clean_captured turns each '\\r' into a newline, exploding that into a wall of
    partial lines. We instead keep only the FINAL state of each line (the text after
    its last '\\r'), collapsing the redraws back to what the operator actually saw.
    A CRLF ('\\r\\n') is a real line break, not a redraw, so normalise it FIRST -
    else the trailing '\\r' would make us keep the empty string after it and drop
    the line's text entirely."""
    text = text.replace("\r\n", "\n")
    out = []
    for line in text.split("\n"):
        out.append(line.rsplit("\r", 1)[-1] if "\r" in line else line)
    return "\n".join(out)


def _clean(raw_bytes):
    """Shape raw pty bytes for the model: collapse readline CR-redraws (so a live
    shell's per-keystroke reprints don't explode), then core's _clean_captured (ANSI
    strip / TUI -> note). Falls back to a plain decode if the hook is missing."""
    text = _collapse_cr(raw_bytes.decode("utf-8", "replace"))
    cc = _H.get("_clean_captured")
    if cc:
        out = cc(text)
        return "" if out is None else out
    return text


def _beyond_echo(acc, echo):
    """True once `acc` (bytes read so far) contains anything BEYOND the pty's echo
    of the just-sent input. The pty echoes a typed line back within milliseconds;
    that echo must NOT be mistaken for the command's actual output (which, over ssh
    / a slow remote, lands hundreds of ms later). We strip one leading copy of the
    echoed bytes (CR-insensitive) and report whether non-whitespace remains."""
    a = acc.replace(b"\r", b"")
    e = (echo or b"").replace(b"\r", b"")
    if e:
        idx = a.find(e)
        if idx != -1:
            a = a[:idx] + a[idx + len(e):]
    return bool(a.strip())


def _drain(fd, timeout, echo=b""):
    """Read whatever the child has produced, up to `timeout` seconds. Returns bytes.

    Waits up to `timeout` for REAL output (anything past the pty echo of `echo`),
    then stops after a short quiet gap (_READ_QUIET) once that output has settled -
    so an idle prompt doesn't burn the whole timeout. Passing `echo` (the bytes just
    written by _send) is what fixes the latency race: the instant command echo no
    longer trips the quiet-gap early-exit before the command's output arrives, which
    made a slow/remote command return only its own echo (output landing a call late).
    With echo=b"" (a bare `read`) the first byte of anything counts, as before."""
    chunks = []
    deadline = time.time() + max(0.0, timeout)
    seen_real = False
    while time.time() < deadline:
        wait = _READ_QUIET if seen_real else (deadline - time.time())
        try:
            r, _, _ = select.select([fd], [], [], max(0.0, wait))
        except (OSError, ValueError):
            break
        if r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:                 # EOF - child's pty closed
                break
            chunks.append(data)
            if not seen_real and _beyond_echo(b"".join(chunks), echo):
                seen_real = True
        elif seen_real:                  # had real output, now quiet -> done
            break
    return b"".join(chunks)


def _drain_until(fd, mark, timeout, echo=b""):
    """Marker-aware drain: read until the end-marker `mark` appears (deterministic
    completion) OR `timeout` elapses (a hard safety ceiling, NOT the primary signal).
    Returns (raw_bytes, exit_code_or_None, done). `done` is True when the marker was
    seen; on a marker hit exit_code is parsed from the `<mark><rc>` line. On timeout
    we return what we have with done=False so the caller can say 'still running'.
    The echoed marker command (the line WE injected) is only ever printed once by the
    shell after the real command; the LAST occurrence is the true terminator, so we
    match on that and ignore the pty echo of our own injected text earlier in the
    stream."""
    buf = b""
    deadline = time.time() + max(0.0, timeout)
    mark_b = mark.encode()
    # <mark><rc> on its own line; rc is the shell's $? (digits, possibly 130 etc).
    pat = re.compile(re.escape(mark_b) + rb"(\d{1,3})")
    while time.time() < deadline:
        try:
            r, _, _ = select.select([fd], [], [], max(0.0, deadline - time.time()))
        except (OSError, ValueError):
            break
        if not r:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:                        # EOF - child's pty closed
            break
        buf += data
        # The marker text appears TWICE in the raw stream: once as the pty echo of
        # our injected command, once when the shell actually runs it. Only the run
        # emits <mark><DIGITS>; the echo is <mark>$? (literal). Match the numeric one.
        m = None
        for m in pat.finditer(buf):
            pass                            # keep the last match (the real terminator)
        if m is not None:
            # Marker seen. Briefly consume the tail (rest of the marker line + the
            # next shell prompt the pty is about to echo) so it doesn't leak into the
            # NEXT command's output. Best-effort, short and bounded.
            tail_deadline = time.time() + 0.3
            while time.time() < tail_deadline:
                try:
                    r2, _, _ = select.select([fd], [], [], 0.1)
                except (OSError, ValueError):
                    break
                if not r2:
                    break
                try:
                    more = os.read(fd, 65536)
                except OSError:
                    break
                if not more:
                    break
                buf += more
            return buf, int(m.group(1)), True
    return buf, None, False


def _strip_marker(text, mark):
    """Return just the command's own output from a marker-drained capture. The
    cleaned stream is:  <echo of our injected line (contains mark)> / OUTPUT /
    <mark><rc> terminator / <next shell prompt>. The output is exactly the lines
    BETWEEN the first mark-bearing line (the echo) and the last one (the
    terminator) - so slicing there drops both the echoed command AND the trailing
    prompt in one move. Falls back to a plain mark-line filter if we can't find the
    two anchors (e.g. a partial capture)."""
    lines = text.split("\n")
    marked = [i for i, ln in enumerate(lines) if mark in ln]
    if len(marked) >= 2:
        return "\n".join(lines[marked[0] + 1:marked[-1]])
    return "\n".join(ln for ln in lines if mark not in ln)

def _alive(sess):
    return sess["proc"].poll() is None


def _reap(sid, sess):
    """Close a session's fd + drop it from the table. Returns any final output."""
    tail = b""
    try:
        tail = _drain(sess["fd"], 0.2)
    except Exception:
        pass
    try:
        os.close(sess["fd"])
    except OSError:
        pass
    _SESSIONS.pop(sid, None)
    return tail


def _start(args, cwd):
    global _SEQ
    if len(_SESSIONS) >= _MAX_SESSIONS:
        return (f"error: too many live sessions ({_MAX_SESSIONS}). Close one first "
                f"(shell_session action=close).")
    try:
        import pty
    except ImportError:
        return "error: no pty support on this platform; shell_session is unavailable."
    command = (args.get("command") or "").strip()
    if command:
        try:
            child = shlex.split(command)
        except ValueError as e:
            return f"error: bad command: {e}"
        where = command
    else:
        shell = os.environ.get("SHELL") or "/bin/sh"
        child = [shell, "-i"]            # interactive so it shows a prompt + line-edits
        where = shell
    master, slave = pty.openpty()
    # TERM=dumb so interactive programs (python's pyrepl, bash's readline, etc.)
    # don't emit cursor-control / line-redraw escapes we'd only have to scrub back
    # out. Kills the noise at the source, so captured output reads like a plain log.
    env = dict(os.environ, TERM="dumb")
    try:
        proc = subprocess.Popen(
            child, stdin=slave, stdout=slave, stderr=slave,
            cwd=str(cwd) if cwd and not command.startswith("ssh") else None,
            env=env, start_new_session=True, close_fds=True)
    except (OSError, ValueError) as e:
        os.close(master); os.close(slave)
        return f"error: could not start session: {e}"
    os.close(slave)                      # parent keeps only the master end
    _SEQ += 1
    sid = f"s{_SEQ}"
    # Drain mode: "shell" -> deterministic marker draining; "raw" -> quiet-gap.
    # An explicit `raw` arg overrides the command-based guess (force one or the other).
    raw = args.get("raw")
    if raw is True:
        mode = "raw"
    elif raw is False:
        mode = "shell"
    else:
        mode = _mode_for(command)
    _SESSIONS[sid] = {"proc": proc, "fd": master, "cmd": where,
                      "started": time.time(), "where": where,
                      "mode": mode, "nonce": f"{os.getpid()}_{_SEQ}"}
    banner = _clean(_drain(master, 1.5)) # let a prompt/banner appear
    head = f"session {sid} started ({where}, pid {proc.pid})."
    return head + (f"\n{banner}" if banner.strip() else "")


def _get(args):
    sid = (args.get("id") or "").strip()
    if not sid:
        return None, "error: this action needs an id (from a start)."
    sess = _SESSIONS.get(sid)
    if not sess:
        return None, f"error: no live session '{sid}' (start one, or list to see live ones)."
    return (sid, sess), None


def _send(args, cwd):
    got, err = _get(args)
    if err:
        return err
    sid, sess = got
    if not _alive(sess):
        tail = _clean(_reap(sid, sess))
        return (f"session {sid} has exited (rc {sess['proc'].returncode}); closed it."
                + (f"\n{tail}" if tail.strip() else ""))
    text = args.get("input", "")
    try:
        timeout = float(args.get("timeout") or _READ_DEFAULT)
    except (TypeError, ValueError):
        timeout = _READ_DEFAULT

    # SHELL mode: append a marker command so completion is deterministic. `printf`
    # runs AFTER the real command (on the same line via ';'), emitting <mark><$?>
    # once the command - however slow or bursty - has actually finished. We drain
    # until that marker instead of guessing with a quiet-gap timer, then strip our
    # injected line + the marker line, leaving just the command's own output plus a
    # real exit code. A bare keystroke (input ends without newline, e.g. sending a
    # ctrl-char or answering a prompt) skips the marker - there's no command to
    # terminate. RAW mode always uses the quiet-gap path (no shell to run printf).
    # Marker only for a normal command line: shell mode, non-empty, single line, and
    # WE own the newline (caller didn't pre-terminate or send a bare keystroke).
    marker_ok = (sess.get("mode") == "shell" and text.strip() != ""
                 and "\n" not in text.rstrip("\n"))
    if marker_ok:
        mark = _MARK_PREFIX + sess["nonce"] + "_"
        line = text.rstrip("\n")
        mark = _MARK_PREFIX + sess["nonce"] + "_"
        line = text.rstrip("\n")
        full = f"{line}; printf '\\n{mark}%s\\n' \"$?\"\n"
        echo = full.encode("utf-8", "replace")
        try:
            os.write(sess["fd"], echo)
        except OSError as e:
            return f"error: write to session {sid} failed: {e}"
        raw, rc, done = _drain_until(sess["fd"], mark, timeout, echo=echo)
        out = _strip_marker(_clean(raw), mark).strip("\n")
        if not _alive(sess):
            _reap(sid, sess)
            return (out + f"\n[session {sid} exited (rc {sess['proc'].returncode})]").strip()
        if not done:
            tail = f"[session {sid}: still running after {timeout:g}s - use read to get the rest]"
            return (out + "\n" + tail).strip() if out.strip() else tail
        if rc == 0:
            return out.strip() if out.strip() else f"[session {sid}: ok, no output]"
        return (out + f"\n[exit {rc}]").strip()

    # RAW mode (or a bare keystroke): quiet-gap drain, echo-aware.
    if not text.endswith("\n") and text != "":
        text += "\n"
    echo = text.encode("utf-8", "replace")
    try:
        os.write(sess["fd"], echo)
    except OSError as e:
        return f"error: write to session {sid} failed: {e}"
    out = _clean(_drain(sess["fd"], timeout, echo=echo))
    if not _alive(sess):
        _reap(sid, sess)
        return (out + f"\n[session {sid} exited (rc {sess['proc'].returncode})]").strip()
    return out if out.strip() else f"[session {sid}: no output within {timeout:g}s]"


def _read(args, cwd):
    got, err = _get(args)
    if err:
        return err
    sid, sess = got
    try:
        timeout = float(args.get("timeout") or _READ_DEFAULT)
    except (TypeError, ValueError):
        timeout = _READ_DEFAULT
    out = _clean(_drain(sess["fd"], timeout))
    if not _alive(sess):
        _reap(sid, sess)
        return (out + f"\n[session {sid} exited (rc {sess['proc'].returncode})]").strip()
    return out if out.strip() else f"[session {sid}: no new output within {timeout:g}s]"


def _close(args, cwd):
    got, err = _get(args)
    if err:
        return err
    sid, sess = got
    proc = sess["proc"]
    if _alive(sess):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            try:
                proc.terminate()
            except OSError:
                pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
    _reap(sid, sess)
    return f"session {sid} closed."


def _list(args, cwd):
    if not _SESSIONS:
        return "no live sessions."
    now = time.time()
    lines = []
    for sid, sess in list(_SESSIONS.items()):
        state = "live" if _alive(sess) else f"exited(rc {sess['proc'].returncode})"
        age = int(now - sess["started"])
        lines.append(f"{sid}  {state}  (up {age}s)  {sess['cmd']}")
    return "\n".join(lines)


_ACTIONS = {"start": _start, "send": _send, "read": _read, "close": _close, "list": _list}


def run(args, cwd):
    action = (args.get("action") or "").strip()
    fn = _ACTIONS.get(action)
    if not fn:
        return f"error: unknown action '{action}' (start|send|read|close|list)."
    return fn(args, cwd)


def _sessions_cmd(agent, cfg, arg):
    print(_list({}, None))


def _close_all():
    for sid in list(_SESSIONS):
        try:
            _close({"id": sid}, None)
        except Exception:
            pass


def setup(lc, cfg):
    # lc IS core's globals(); grab the output shaper (optional - we degrade to a
    # plain decode without it). setup runs driver-only = where run() also fires.
    if "_clean_captured" in lc:
        _H["_clean_captured"] = lc["_clean_captured"]
    # Operator-facing listing; tidy up any live sessions at exit.
    reg = lc.get("register_command")
    if reg:
        reg("/sessions", _sessions_cmd, "list live shell_session shells")
    try:
        import atexit
        atexit.register(_close_all)
    except Exception:
        pass
