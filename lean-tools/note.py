# Example lean-coder lean-tool: note  (mutating -> NOT marked safe)
#
# Because `safe` is omitted, lean-coder confirms before running this (and runs
# it on the remote box when you are connected). Compare with word_count.py,
# which is read-only and runs without a prompt.

TOOL = {
    "name": "note",
    "description": "Append a line to NOTES.md in the working dir.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    # no "safe" key -> treated as mutating -> asks before running (auto-approve waives)
}


def run(args, cwd):
    from pathlib import Path
    note = (args.get("text") or "").strip()
    if not note:
        return "error: empty note"
    f = Path(cwd) / "NOTES.md"
    try:
        with f.open("a", encoding="utf-8") as fh:
            fh.write(f"- {note}\n")
    except OSError as e:
        return f"error: could not write {f.name}: {e}"
    return f"appended to {f.name}"
