# Example lean-coder lean-tool: reverse_file  (simple, local, read-only)
#
# The minimal shape of a lean-tool. One .py file exposing:
#   TOOL           - the schema the model sees (name, description, parameters, safe?)
#   run(args, cwd) - the handler; returns a string the model receives
#
# Copy this into ~/.config/leancoder/lean-tools/, run /tools, and enable it.
# See LEAN_TOOLS.md for the full guide.

TOOL = {
    "name": "reverse_file",
    # one line: it is serialized into every request, so keep it terse
    "description": "Return a file's lines in reverse order.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "safe": True,   # read-only -> runs without a confirmation prompt
}


def run(args, cwd):
    # args matches `parameters`; cwd is the working dir (local, or the remote dir
    # when connected - so resolve paths against cwd, never hardcode them).
    from pathlib import Path
    p = Path(cwd) / args["path"]
    if not p.is_file():
        return f"error: no such file: {args['path']}"
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(reversed(lines)) or "(empty file)"
