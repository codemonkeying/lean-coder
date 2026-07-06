# Example lean-coder lean-tool: word_count  (read-only)
#
# A lean-tool is one .py file exposing:
#   TOOL  - the tool schema the model sees (name, description, parameters, safe?)
#   run(args, cwd) -> str  - the handler; returns a string the model receives
#
# To try it: copy this file into ~/.config/leancoder/lean-tools/ then run /tools
# and enable it. See LEAN_TOOLS.md for the full guide.

TOOL = {
    "name": "word_count",
    # one line: it is serialized into every request, so keep it terse
    "description": "Count lines, words, and characters in a file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "safe": True,   # read-only -> runs without a confirmation prompt
}


def run(args, cwd):
    # args matches `parameters`; cwd is the working dir (local, or the remote
    # dir when connected - so resolve paths against cwd, never hardcode them).
    from pathlib import Path
    p = Path(cwd) / args["path"]
    if not p.is_file():
        return f"error: no such file: {args['path']}"
    text = p.read_text(errors="replace")
    return f"{len(text.splitlines())} lines, {len(text.split())} words, {len(text)} chars"
