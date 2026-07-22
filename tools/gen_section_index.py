#!/usr/bin/env python3
"""Regenerate the SECTION index block inside lean_coder.py's module docstring.

The file is a deliberately single, zero-dependency module (curl-and-run). To keep it
navigable like a package, each major region carries a unique '# SECTION: <title>'
banner. This script scans those banners and rewrites the '=== FILE MAP ===' block near
the top of the module docstring so a reader (or a tool) can `read_file end=40` and see
the whole layout, and `search_files "^# SECTION:"` jumps to any region.

Run from the repo root:  python3 tools/gen_section_index.py
It edits lean_coder.py in place and is idempotent. Fails (nonzero) if the markers are
missing so CI/gate can catch drift.
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "lean_coder.py"
BEGIN = "=== FILE MAP (regen: tools/gen_section_index.py) ==="
END = "=== END FILE MAP ==="


def build_index(lines):
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith("# SECTION:"):
            title = ln[len("# SECTION:"):].strip()
            out.append(f"  L{i + 1:<6} {title}")
    return out


def main():
    text = SRC.read_text()
    lines = text.split("\n")
    index = build_index(lines)
    if not index:
        print("no '# SECTION:' banners found", file=sys.stderr)
        return 1
    block = [BEGIN] + index + [END]
    # Locate an existing block (between BEGIN/END) and replace it; else insert just
    # before the closing docstring triple-quote of the module docstring (line find).
    joined = "\n".join(block)
    if BEGIN in text and END in text:
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                     joined, text, count=1, flags=re.S)
    else:
        # insert before the first standalone '"""' that closes the module docstring
        # (the second triple-quote in the file).
        tq = [m.start() for m in re.finditer(r'"""', text)]
        if len(tq) < 2:
            print("could not find module docstring close", file=sys.stderr)
            return 1
        at = tq[1]
        new = text[:at] + "\n" + joined + "\n" + text[at:]
    SRC.write_text(new)
    print(f"wrote {len(index)} SECTION entries into the file map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
