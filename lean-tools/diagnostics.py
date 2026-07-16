"""diagnostics - run the available linter / typechecker on a path and return its
findings, so the model SEES errors instead of parsing raw build output. This is
the feedback-loop tool: lint the file you just edited, fix what it reports.

Picks the first installed linter for the file's language:
  Python: ruff -> pyflakes -> py_compile   JS/TS: tsc / eslint
  Go: go vet                               Rust: cargo clippy
  PHP: php -l                              Shell: shellcheck -> bash -n
  Ruby: ruby -c                            C/C++: gcc/g++ -fsyntax-only
  Java: javac -Xlint                       CSS: stylelint
  JSON: python -m json.tool                YAML: yamllint
  Markdown: markdownlint
Read-only (safe): it never modifies files. Whatever linters are installed on the
machine that runs the tool (local, or the remote executor when /connect'd) are
what it uses - no linters are pushed or installed, only what's already there.
"""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

TOOL = {
    "name": "diagnostics",
    "description": "Lint/typecheck a file or dir with the installed tool; return errors + warnings.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "safe": True,
}

_PY = sys.executable
_LINTERS = {
    # Python
    ".py":   [["ruff", "check"], [_PY, "-m", "pyflakes"], [_PY, "-m", "py_compile"]],
    ".pyi":  [["ruff", "check"]],
    # JavaScript / TypeScript
    ".ts":   [["npx", "--no-install", "tsc", "--noEmit"]],
    ".tsx":  [["npx", "--no-install", "tsc", "--noEmit"]],
    ".js":   [["npx", "--no-install", "eslint"], ["node", "--check"]],
    ".jsx":  [["npx", "--no-install", "eslint"]],
    # Go / Rust
    ".go":   [["go", "vet"]],
    ".rs":   [["cargo", "clippy", "-q"]],
    # PHP
    ".php":  [["php", "-l"]],
    # Shell
    ".sh":   [["shellcheck"], ["bash", "-n"]],
    ".bash": [["shellcheck"], ["bash", "-n"]],
    ".zsh":  [["shellcheck", "-s", "bash"], ["zsh", "-n"]],
    # Ruby
    ".rb":   [["ruby", "-c"]],
    # C / C++
    ".c":    [["gcc", "-fsyntax-only", "-Wall"]],
    ".h":    [["gcc", "-fsyntax-only", "-Wall"]],
    ".cpp":  [["g++", "-fsyntax-only", "-Wall"]],
    ".cc":   [["g++", "-fsyntax-only", "-Wall"]],
    ".hpp":  [["g++", "-fsyntax-only", "-Wall"]],
    # Java
    ".java": [["javac", "-Xlint", "-d", "/tmp"]],
    # CSS
    ".css":  [["npx", "--no-install", "stylelint"]],
    ".scss": [["npx", "--no-install", "stylelint"]],
    # Data formats
    ".json": [[_PY, "-m", "json.tool"]],
    ".yaml": [["yamllint"]],
    ".yml":  [["yamllint"]],
    # Markdown
    ".md":   [["markdownlint"]],
}
# tried when the target is a directory (project-wide)
_DIR_LINTERS = [["ruff", "check"], ["npx", "--no-install", "eslint", "."],
                ["php", "-l"], ["shellcheck"]]


def _module_available(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def pick_linter(ext, is_dir, which=shutil.which, have_module=_module_available):
    """First linter that is actually runnable for this extension / dir. For a
    'python -m MODULE' entry the binary (python) always exists, so we must check
    the MODULE is importable - otherwise we'd pick e.g. pyflakes when it isn't
    installed and fail at runtime instead of falling through to py_compile (which
    is stdlib and always available). `which`/`have_module` are injectable for
    tests."""
    for cmd in (_DIR_LINTERS if is_dir else _LINTERS.get(ext, [])):
        if len(cmd) >= 3 and cmd[0] == _PY and cmd[1] == "-m":
            if have_module(cmd[2]):
                return cmd
        elif which(cmd[0]):
            return cmd
    return None


def run(args, cwd):
    rel = (args.get("path") or "").strip()
    target = Path(cwd) / rel
    if not target.exists():
        return f"error: {rel or '(empty)'} not found"
    is_dir = target.is_dir()
    cmd = pick_linter(target.suffix, is_dir)
    if not cmd:
        what = "directory" if is_dir else f"'{target.suffix or 'this file type'}'"
        return f"diagnostics: no linter installed for {what}"
    try:
        r = subprocess.run(cmd + [str(target)], capture_output=True, text=True,
                           cwd=cwd, timeout=120)
    except Exception as e:
        return f"diagnostics: failed to run {cmd[0]}: {e}"
    out = (r.stdout + r.stderr).strip()
    if not out:
        return (f"{cmd[0]}: no issues in {rel}" if r.returncode == 0
                else f"{cmd[0]}: exit {r.returncode} (no output)")
    if len(out) > 4000:
        out = out[:4000] + "\n... (truncated)"
    return f"{cmd[0]} [{'clean' if r.returncode == 0 else 'issues'}]:\n{out}"
