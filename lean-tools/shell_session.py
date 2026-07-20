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
            "timeout": {"type": "number",
                        "description": "send/read only: seconds to wait for output "
                                       "(default 4). Raise it for a slow command."},
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
    _SESSIONS[sid] = {"proc": proc, "fd": master, "cmd": where,
                      "started": time.time(), "where": where}
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
    if not text.endswith("\n") and text != "":
        text += "\n"
    echo = text.encode("utf-8", "replace")
    try:
        os.write(sess["fd"], echo)
    except OSError as e:
        return f"error: write to session {sid} failed: {e}"
    try:
        timeout = float(args.get("timeout") or _READ_DEFAULT)
    except (TypeError, ValueError):
        timeout = _READ_DEFAULT
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
