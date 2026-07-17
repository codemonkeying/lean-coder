"""ssh - run a non-interactive command on a remote host over SSH.

Moved out of the core tool set: it's network egress (reaching another machine,
not editing this repo), so it's opt-in. The model never sees it unless you
enable it via /tools. For an interactive remote *workspace* use /connect; this
tool is just one-shot `ssh host cmd`.

Not marked safe (it runs a remote command), so lean-coder confirms before each
call unless auto-approved. Enabling this also makes it available inside a /connect
workspace (it's pushed like any tool lean-tool) - the old built-in hid itself there;
as an opt-in lean-tool, that's now your choice.
"""
import subprocess

TOOL = {
    "name": "ssh",
    "description": "Run a non-interactive command on a remote host over SSH; returns stdout/stderr.",
    "parameters": {
        "type": "object",
        "properties": {
            "host": {"type": "string"},
            "cmd": {"type": "string"},
            "port": {"type": "integer", "description": "SSH port (default 22)."},
            "identity": {"type": "string", "description": "Path to private key file (passed as -i)."},
            "timeout": {"type": "integer", "description": "Command timeout in seconds (default 300)."},
        },
        "required": ["host", "cmd"],
    },
    # no "safe": runs a remote command, so it goes through the confirm gate
}


def run(args, cwd):
    host, cmd = args.get("host", ""), args.get("cmd", "")
    if not host or not cmd:
        return "error: ssh needs both 'host' and 'cmd'"
    port = args.get("port")
    identity = (args.get("identity") or "").strip()
    try:
        cmd_timeout = max(10, int(args.get("timeout") or 300))
    except (TypeError, ValueError):
        cmd_timeout = 300
    # one-shot only: BatchMode fails fast instead of hanging on a prompt; no PTY.
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new"]
    if port:
        try:
            argv += ["-p", str(int(port))]
        except (TypeError, ValueError):
            return f"error: invalid port {port!r}"
    if identity:
        argv += ["-i", identity]
    argv += [host, cmd]
    try:
        r = subprocess.run(argv, shell=False, capture_output=True,
                           text=True, timeout=cmd_timeout)
    except subprocess.TimeoutExpired:
        return f"error: ssh timed out after {cmd_timeout}s"
    except Exception as e:
        return f"error running ssh: {e}"
    out = ""
    if r.stdout:
        out += r.stdout
    if r.stderr:
        out += ("\n[stderr]\n" if out else "") + r.stderr
    out = out.rstrip()
    if len(out) > 4000:
        out = out[:4000] + "\n... (truncated)"
    return f"exit {r.returncode}\n{out}" if out else f"exit {r.returncode} (no output)"
