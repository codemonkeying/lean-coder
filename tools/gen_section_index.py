#!/usr/bin/env python3
"""Regenerate the SECTION index block inside each big single-file module's docstring.

lean_coder.py is a deliberately single, zero-dependency module (curl-and-run); its test
harness tests/_smoketest.py is likewise one long flat script. To keep both navigable
like a package, each major region carries a unique '# SECTION: <title>' banner. This
script scans those banners in each target file and rewrites its '=== FILE MAP ===' block
(kept in the module docstring) so a reader - or a tool - can `read_file end=NN` for the
whole layout, and `search_files "^# SECTION:"` jumps to any region.

Run from the repo root:  python3 tools/gen_section_index.py
It edits each target in place and is idempotent. Fails (nonzero) if a target has no
banners or no docstring to hold the map, so CI/gate can catch drift.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "lean_coder.py", ROOT / "tests" / "_smoketest.py"]
BEGIN = "=== FILE MAP (regen: tools/gen_section_index.py) ==="
END = "=== END FILE MAP ==="


def build_index(lines):
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith("# SECTION:"):
            title = ln[len("# SECTION:"):].strip()
            out.append(f"  L{i + 1:<6} {title}")
    return out


def _render(text):
    """One pass: rebuild the map block from the banner lines in `text`. Returns the new
    text, or None on a structural error."""
    lines = text.split("\n")
    index = build_index(lines)
    if not index:
        return None, 0
    joined = "\n".join([BEGIN] + index + [END])
    if BEGIN in text and END in text:
        return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                      lambda _m: joined, text, count=1, flags=re.S), len(index)
    # insert before the module docstring's closing triple-quote (the 2nd in the file)
    tq = [m.start() for m in re.finditer(r'"""', text)]
    if len(tq) < 2:
        return None, -1
    return text[:tq[1]] + "\n" + joined + "\n" + text[tq[1]:], len(index)


def process(src: Path):
    # The map sits ABOVE the banners, so a map that grows/shrinks by N lines shifts every
    # banner by N - the numbers just written go stale. Re-render until a pass is a no-op
    # (a fixed point), so ONE run is always enough. Bounded: it converges in <=2 passes
    # (the second pass only fixes the shift, it doesn't change the entry count).
    text = src.read_text()
    new, n = text, 0
    for _ in range(5):
        nxt, n = _render(new)
        if nxt is None:
            why = "no '# SECTION:' banners found" if n == 0 else "could not find module docstring close"
            print(f"{src.name}: {why}", file=sys.stderr)
            return 1
        if nxt == new:
            break
        new = nxt
    else:
        print(f"{src.name}: file map did not converge", file=sys.stderr)
        return 1
    if new != text:
        src.write_text(new)
    print(f"{src.name}: wrote {n} SECTION entries into the file map")
    return 0


def main():
    rc = 0
    for t in TARGETS:
        if not t.is_file():
            print(f"{t}: not found", file=sys.stderr)
            rc = 1
            continue
        rc = process(t) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
