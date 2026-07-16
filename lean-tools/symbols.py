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
                    "this file'."),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "A .py file or a directory to scan (recursively)."},
            "mode": {"type": "string", "enum": ["outline", "def"],
                     "description": "outline (default) = structure; def = locate a definition."},
            "name": {"type": "string",
                     "description": "For mode='def': the class/function/method name to locate."},
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


def _outline_one(path, rel):
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError as e:
        return [f"{rel}: syntax error line {e.lineno}: {e.msg}"]
    lines = [f"{rel}:"]
    for depth, node in _walk(tree):
        lines.append(f"  {'  ' * depth}{node.lineno}: {_sig(node)}")
    if len(lines) == 1:
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
            hits += _def_one(f, str(f.relative_to(cwd) if f.is_absolute() else f), name)
        if not hits:
            return f"symbols: no definition of '{name}' in {rel}"
        return "\n".join(hits)

    if mode != "outline":
        return f"error: unknown mode {mode!r} (use outline | def)."

    out = []
    for f in files:
        out += _outline_one(f, str(f.relative_to(cwd) if f.is_absolute() else f))
    body = "\n".join(out)
    if len(body) > _MAX:
        body = body[:_MAX] + "\n... (truncated; narrow the path or use mode='def')"
    return body
