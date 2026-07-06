"""git_summary - a one-call, read-only snapshot of the working dir's git state.

Collapses the usual status -> branch -> log -> diff round-trips into a single
tool call, so the model spends one turn instead of four to orient in a repo.
Read-only (only `git` query commands, no writes, no shared state), so it's
marked safe: it runs without a confirmation prompt and is safe to run
concurrently with other read-only tools.

Each section is capped independently so one pathological section (thousands of
untracked files, a giant diffstat) can't crowd out the others, and the most
useful section -- recent commits -- is emitted first so it always survives.
A single wall-clock deadline is shared across the underlying git calls so a
slow repo can't stack up many separate timeouts.
"""
import subprocess
import time

TOOL = {
    "name": "git_summary",
    "description": "Read-only git snapshot of the working dir: branch (+ ahead/behind), status, staged/unstaged diffstat, and recent commits.",
    "parameters": {"type": "object", "properties": {}},
    "safe": True,
}

_CAP = 4000          # overall result ceiling
_SECTION_CAP = 1500  # per-section ceiling, so no one section dominates
_DEADLINE = 20.0     # shared wall-clock budget across all git calls (seconds)


def _clip(text, label):
    """Trim a section to _SECTION_CAP on a line boundary, noting how much was cut."""
    if len(text) <= _SECTION_CAP:
        return text
    head = text[:_SECTION_CAP]
    nl = head.rfind("\n")
    if nl > 0:
        head = head[:nl]
    dropped = text.count("\n", len(head)) + 1
    return head + f"\n... ({label}: {dropped} more line(s) truncated)"


def run(args, cwd):
    deadline = time.monotonic() + _DEADLINE

    def _git(*gitargs):
        """Run a git query within the shared deadline; stdout or None on failure."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            r = subprocess.run(["git", *gitargs], cwd=cwd, capture_output=True,
                               text=True, timeout=remaining)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.rstrip() if r.returncode == 0 else None

    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside != "true":
        return "not a git repository (no snapshot)"

    parts = []
    # Commits first: most useful for orienting, and guaranteed to survive the cap.
    log = _git("log", "--oneline", "-5")
    if log:
        parts.append(_clip("recent commits:\n" + log, "commits"))
    status = _git("status", "--short", "--branch")
    if status is not None:
        # Branch/ahead-behind reflects the last fetch; run `git fetch` for live remote state.
        parts.append(_clip(status or "(clean)", "status"))
    staged = _git("diff", "--cached", "--stat")
    if staged:
        parts.append(_clip("staged:\n" + staged, "staged"))
    unstaged = _git("diff", "--stat")
    if unstaged:
        parts.append(_clip("unstaged:\n" + unstaged, "unstaged"))

    out = "\n\n".join(parts) if parts else "(no git output)"
    if len(out) > _CAP:
        out = out[:_CAP] + "\n... (truncated)"
    return out
