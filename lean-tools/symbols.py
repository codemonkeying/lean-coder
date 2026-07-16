"""symbols - a dependency-free code navigator for PYTHON files, built on the
stdlib `ast`. Gives the model the two things a grep can't do cleanly: a structural
OUTLINE (every class / function / method with its line number and signature) and a
DEFINITION lookup by name - so "where is X defined / what's the shape of this file"
is one call instead of a scattershot of search_files + read_file round-trips.

Python only, by design: it leans on the stdlib `ast`, so it adds NO dependency and
fits the zero-dep ethos. For multi-language navigation (PHP/JS/etc.) a tree-sitter
tool is the future upgrade (see backlog); this covers the Python case dep-free.

Modes (via `mode`):
  outline (default) - the structure of one file OR every .py under a dir: each
                      class/def with line number, signature, and nesting.
  def <name>        - locate where `name` (a class/function/method) is DEFINED;
                      searches the given file, or every .py under a dir.

Read-only (safe). Resolves paths against cwd (local, or the remote dir when
/connect'd), like every builtin.
"""
import ast
from pathlib import Path

TOOL = {
    "name": "symbols",
    "description": ("Navigate PYTHON code without grepping: mode='outline' lists every "
                    "class/def (line + signature) in a file or dir; mode='def' finds where "
                    "a named class/function/method is defined. Zero-dep (stdlib ast); Python "
                    "only. Faster + more exact than search_files for 'where is X / what's in "
                    "this file'. Just pass a path - the defaults do the sensible thing; on a "
                    "big file outline auto-summarizes to top-level classes/functions."),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "A .py file or a directory to scan (recursively)."},
            "mode": {"type": "string", "enum": ["outline", "def"],
                     "description": "outline (default) = structure; def = locate a definition."},
            "name": {"type": "string",
                     "description": "For mode='def': the class/function/method name to locate."},
            "match": {"type": "string",
                      "description": "Optional (outline): only show symbols whose name contains "
                                     "this text (case-insensitive). Use it to focus a big file."},
            "depth": {"type": "string", "enum": ["top", "full"],
                      "description": "Optional (outline): 'top' = classes + module-level defs only "
                                     "(compact); 'full' = include nested methods. Default: auto - "
                                     "'full', but falls back to 'top' if the output is too big."},
        },
        "required": ["path"],
    },
    "safe": True,
}

_MAX = 4000     # output cap so a huge tree can't flood the model's context


def _sig(node):
    """A compact signature string for a def/class node (best-effort, never raises)."""
    try:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            return f"class {node.name}({bases})" if bases else f"class {node.name}"
        a = node.args
        parts = [ast.unparse(x) for x in a.posonlyargs] if getattr(a, "posonlyargs", None) else []
        parts += [ast.unparse(x) for x in a.args]
        if a.vararg:
            parts.append("*" + a.vararg.arg)
        parts += [ast.unparse(x) for x in a.kwonlyargs]
        if a.kwarg:
            parts.append("**" + a.kwarg.arg)
        kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{kw} {node.name}({', '.join(parts)})"
    except Exception:
        return getattr(node, "name", "?")


def _walk(tree):
    """Yield (depth, node) for every class/def in source order, tracking nesting."""
    def rec(body, depth):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield depth, n
                yield from rec(n.body, depth + 1)
    yield from rec(tree.body, 0)


def _py_files(target):
    if target.is_dir():
        return sorted(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
    return [target] if target.suffix in (".py", ".pyi") else []


def _outline_one(path, rel, top_only=False, match=None):
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError as e:
        return [f"{rel}: syntax error line {e.lineno}: {e.msg}"]
    m = match.lower() if match else None
    lines = [f"{rel}:"]
    for depth, node in _walk(tree):
        if top_only and depth > 0:
            continue
        if m and m not in node.name.lower():
            continue
        lines.append(f"  {'  ' * depth}{node.lineno}: {_sig(node)}")
    if len(lines) == 1:
        if m:
            lines.append(f"  (no class/function name contains {match!r})")
        else:
            lines.append("  (no classes or functions)")
    return lines


def _def_one(path, rel, name):
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    hits = []
    for _depth, node in _walk(tree):
        if node.name == name:
            hits.append(f"{rel}:{node.lineno}: {_sig(node)}")
    return hits


def _disp(f, cwd):
    """A short display path for a file: relative to cwd when it's under it, else the
    path as-is. relative_to() raises for a file OUTSIDE cwd (e.g. an absolute /tmp
    path while cwd is /home/user), so guard it - that ValueError was crashing the tool."""
    try:
        return str(f.relative_to(cwd))
    except ValueError:
        return str(f)


def run(args, cwd):
    rel = (args.get("path") or "").strip()
    target = Path(cwd) / rel
    if not target.exists():
        return f"error: {rel or '(empty)'} not found"
    files = _py_files(target)
    if not files:
        return f"symbols: no .py files at {rel} (Python only)"
    mode = (args.get("mode") or "outline").strip().lower()

    if mode == "def":
        name = (args.get("name") or "").strip()
        if not name:
            return "error: mode='def' needs a name (the definition to locate)."
        hits = []
        for f in files:
            hits += _def_one(f, _disp(f, cwd), name)
        if not hits:
            return f"symbols: no definition of '{name}' in {rel}"
        return "\n".join(hits)
    if mode != "outline":
        return f"error: unknown mode {mode!r} (use outline | def)."

    match = (args.get("match") or "").strip() or None
    depth = (args.get("depth") or "auto").strip().lower()
    top_only = depth == "top"

    def build(top):
        out = []
        for f in files:
            out += _outline_one(f, _disp(f, cwd), top_only=top, match=match)
        return "\n".join(out)

    body = build(top_only)
    # auto mode: if the full tree overflows, retry top-level-only rather than
    # blunt-truncating mid-symbol - a compact whole beats a chopped fragment.
    if depth == "auto" and len(body) > _MAX:
        compact = build(True)
        if len(compact) <= _MAX:
            return (compact + "\n... (methods hidden to fit; re-run with depth='full' "
                    "on a single class/file, or use match='name' to focus)")
        body = compact
    if len(body) > _MAX:
        body = body[:_MAX] + ("\n... (truncated; narrow the path, add match='name', "
                              "or use mode='def')")
    return body
