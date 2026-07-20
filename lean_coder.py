#!/usr/bin/env python3
"""lean-coder - a minimal terminal coding agent driving an Ollama model via
native tool calling. Zero third-party dependencies: Python stdlib only.

Design priority: lean context usage. Small system prompt, one-line tool
schemas, truncated tool results. See README.md.
"""

import argparse
import base64
import datetime
import hashlib
import atexit
import fnmatch
import inspect
import itertools
import json
import os
import re
import select
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import shutil
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path

try:
    import readline  # stdlib; gives Tab-completion + line history where available
except ImportError:  # pragma: no cover - not present on some platforms
    readline = None

# ----------------------------------------------------------------------------
# Constants / tunables
# ----------------------------------------------------------------------------

_HOSTNAME = socket.gethostname()                 # cached: never changes in-process
CONFIG_PATH = Path.home() / ".config" / "leancoder" / "config.toml"
CONFIG_DIR = CONFIG_PATH.parent                  # ~/.config/leancoder; providers read
                                                 # this via lc["CONFIG_DIR"] for secrets
                                                 # /caches instead of hardcoding the path
SESSIONS_DIR = CONFIG_DIR / "sessions"           # saved + autosaved conversations
HISTORY_PATH = CONFIG_DIR / "history"            # composer input history (one line per entry)
HISTORY_MAX = 1000                               # trailing entries kept on disk
AUTOSAVE_KEEP = 10               # rolling autosave sessions kept (oldest pruned)
AUTOSAVE_PREFIX = "auto-"        # autosave session names; named snapshots have none
# Pre-handover safety snapshots: written just before a handover REPLACES history, so
# the full pre-handover conversation stays recoverable (via /load, and the grep-my-own-
# context recall path). Named <origin-session>-prehandover-<N> so the snapshot is tied
# to WHAT it snapshots. (Historically mislabelled "pre-compact-*" - it's a handover, not
# a compact: compact() only stubs tool results in place and takes no snapshot.)
PREHANDOVER_MARK = "-prehandover-"   # infix marking a pre-handover snapshot
PREHANDOVER_KEEP = 3             # cap them: safety nets, not named saves - newest few suffice


def _is_snapshot(name: str) -> bool:
    """True for a rolling pre-handover safety snapshot (not a user/auto session)."""
    return PREHANDOVER_MARK in name


def _prehandover_name(origin: str, existing) -> str:
    """A pre-handover snapshot name tied to its origin session: <origin>-prehandover-<N>,
    N = the next free index for that origin among `existing` names. So a session's
    snapshots read origin-prehandover-1, -2, ... and the recall path can find them by
    globbing '<origin>-prehandover-*'."""
    base = f"{origin}{PREHANDOVER_MARK}"
    used = []
    for nm in existing:
        if nm.startswith(base):
            tail = nm[len(base):]
            if tail.isdigit():
                used.append(int(tail))
    return f"{base}{(max(used) + 1) if used else 1}"


# Human-readable release version, SemVer (MAJOR.MINOR.PATCH[-prerelease]). Bump on a
# published change; the /update lean-tool probes the repo's VERSION file against this to
# decide whether to pull. A pre-release suffix (e.g. 1.2.0-beta.1) marks the beta track:
# it has LOWER precedence than the same core release (1.2.0), per SemVer. source_hash()
# (below) is the exact-content fingerprint /connect uses to skip a redundant re-push -
# a different axis (any byte change), so the two are intentionally separate.
__version__ = "0.9.1"


def _prerelease_key(pre):
    """SemVer pre-release precedence key for the dot-separated identifiers after '-'.
    Numeric identifiers compare numerically and rank below alphanumerics; each is
    wrapped as (is_alpha, num, str) so ints and strings never compare directly."""
    key = []
    for ident in pre.split("."):
        if ident.isdigit():
            key.append((0, int(ident), ""))
        else:
            key.append((1, 0, ident))
    return tuple(key)


def version_tuple(s=None):
    """Parse a SemVer string into a comparable key. Returns (major, minor, patch,
    is_release, prerelease_key): is_release=1 for a plain release, 0 for a
    pre-release, so 1.2.0 > 1.2.0-beta.1 (SemVer precedence). Non-numeric core
    parts and junk degrade to 0 so a malformed string never raises. Defaults to
    this build's __version__."""
    s = str(__version__ if s is None else s).strip()
    core, _, pre = s.partition("-")           # split off any -prerelease
    nums = []
    for part in core.split("."):
        digits = "".join(c for c in part if c.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:                        # pad to (major, minor, patch)
        nums.append(0)
    if pre:
        return (nums[0], nums[1], nums[2], 0, _prerelease_key(pre))
    return (nums[0], nums[1], nums[2], 1, ())


def source_hash() -> str:
    """sha256 of this build's own source - the 'version' /connect compares against
    a provisioned box's installed copy to decide whether a re-push is needed."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return ""
COMPACT_KEEP = 3               # /compact: tool results kept in full (rest stubbed)
NUDGE_EVERY = 5                # soft-zone: nudge the model to wrap up at most every N turns
AUTO_CTX_CAP = 32768           # auto-detect never raises num_ctx above this (OOM guard)

READ_MAX_LINES = 600           # line-number'd file read cap
READ_HEAD = 450
READ_TAIL = 100
OUTPUT_MAX_CHARS = 6000        # run_command result cap (sent to model)
OUTPUT_HEAD = 3500
OUTPUT_TAIL = 2000
RAW_READ_MAX = 1_000_000       # byte ceiling for remote diff-preview fetches
SEARCH_MAX_MATCHES = 200
SEARCH_LINE_MAX = 200          # max chars of a matched line
SEARCH_MAX_FILE_BYTES = 5_000_000   # skip files bigger than this - never slurp a
                                    # multi-GB VM image / DB / log into RAM (swap death)
TREE_MAX_ENTRIES = 400
TREE_DEFAULT_DEPTH = 3

DEFAULT_IGNORES = [
    ".git/", "node_modules/", "dist/", "build/", "out/", ".next/",
    "__pycache__/", ".venv/", "venv/", ".mypy_cache/", ".pytest_cache/",
    ".cache/", ".mozilla/", ".config/",   # avoid flooding ctx when cwd is $HOME
    "*.lock", "*.log", "*.min.js", "*.map", "*.pyc", "*.so", "*.o",
    "*.bin", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.pdf", "*.zip",
]

SYSTEM_PROMPT = (
    "You are a precise code editor. Read files, make targeted edits with "
    "apply_diff, and run commands when needed. Be concise - output only what "
    "changed; don't explain unless asked. "
    "The operator sees your prose and a ONE-LINE truncated preview of each tool "
    "result - not the full output. When they ask for information a tool produced "
    "(a status, a value, a file's contents), relay the relevant part in your reply; "
    "never run a tool and then answer as if they saw its output. "
    # Diagnose-before-switching (highest-ROI anti-thrash rule).
    "If an approach fails, diagnose why before switching tactics - read the error, "
    "check your assumptions, try a focused fix. Don't retry the identical action "
    "blindly, but don't abandon a viable approach after a single failure either. "
    "Escalate to the operator only when genuinely stuck after investigation. "
    # Actions-with-care / reversibility.
    "Weigh reversibility before you act. Irreversible or far-reaching actions - "
    "git push (especially --force), rm -rf, dropping or migrating a database, "
    "anything visible to third parties or that spends money - warrant extra care: "
    "confirm intent first unless the operator has clearly asked for exactly that. "
    "The operator approving an action once does NOT mean it's approved in every "
    "later context. Never commit unless asked. "
    # Function-result clearing (standing, not just at compaction).
    "Older tool results may be cleared from context as the conversation grows, so "
    "copy any critical values - file paths, command output, error messages, IDs - "
    "into your prose while you still have them; don't rely on a past tool result "
    "still being visible later."
)

HANDOVER_MARK = "===HANDOVER==="    # the visible delimiter the model wraps its handover in
SELFPROMPT_MARK = "===NEXT==="      # the model wraps its post-compaction self-prompt in these
PLAN_MARK = "===PLAN==="            # the model wraps its pinned goal + TODO in these
BRIEF_MARK = "===BRIEF==="          # a worker's task (its first user turn) - see --agent-run
GRANT_MARK = "===GRANT==="          # a worker's grant header (leash/model/cwd/limits)
RESULT_MARK = "===RESULT==="        # a worker wraps its final answer in these; the parent harvests it

HANDOVER_INSTR = (
    "You are wrapping up this session (/handover). Two steps, in order:\n"
    "1. DURABLE DOCS: if there are docs/notes worth persisting (a plan, a design doc, a "
    "README, a HANDOVER.md - whatever this project already uses), update them now with "
    "your tools, and if this is ALREADY a git repo, commit them. Do NOT initialise a new "
    "repo, and do NOT commit if there isn't one. If you have no write/command tools right "
    "now (read-only leash), skip this step.\n"
    "2. HANDOVER: write a concise handover for your future self (memory will be wiped) - "
    "the goal, key decisions, current state, WHERE the durable docs live, and a clear "
    "prompt to continue from. Wrap ONLY that handover between two " + HANDOVER_MARK +
    " markers as visible text, like:\n" + HANDOVER_MARK + "\n<your handover here>\n" +
    HANDOVER_MARK + "\n"
    "3. OPTIONAL PLAN: you may write your goal + TODO between two " + PLAN_MARK +
    " markers; it stays pinned for whoever resumes:\n" + PLAN_MARK +
    "\nGOAL: ...\nTODO:\n- [ ] ...\n" + PLAN_MARK + "\n"
    "4. OPTIONAL SELF-PROMPT: if there's a clear next action, write it between two " +
    SELFPROMPT_MARK + " markers (your continue-from prompt):\n" + SELFPROMPT_MARK +
    "\n<what to do next>\n" + SELFPROMPT_MARK + "\n"
    "Write these blocks as visible text and then STOP - the handover text becomes the "
    "ENTIRE continuing context; everything else is discarded. No tool call is needed to "
    "finish; just end your reply after the blocks."
)


def _extract_marked(text: str, mark: str):
    """Text between the LAST pair of `mark` delimiters (so an echoed instruction
    example doesn't win). None when there isn't a complete pair - the caller then
    refuses to act, so a bad parse never nukes the session. Pure -> tested.

    Belt-and-braces fallback: a small model often opens with the fence but closes
    with an XML-style tag derived from the mark (e.g. the HANDOVER fence, then the
    body, then a </handover> close), leaving only ONE fence - so the strict pair
    match misses a perfectly good block and the whole handover is silently
    discarded. When there is exactly one fence, accept a matching close tag whose
    letters equal the mark's letters (case-insensitive). Only that XML-close case
    is rescued; we never greedily take everything after a lone fence (that would
    swallow later blocks)."""
    if not text:
        return None
    close = text.rfind(mark)
    open_ = text.rfind(mark, 0, close) if close != -1 else -1
    if open_ != -1:
        block = text[open_ + len(mark):close].strip()
        return block or None
    if close == -1:
        return None
    tag = re.sub(r"[^a-z0-9]", "", mark.lower())
    if tag:
        m = re.search(re.escape(mark) + r"(.*?)</\s*" + re.escape(tag) + r"\s*>",
                      text, re.IGNORECASE | re.DOTALL)
        if m:
            body = m.group(1)
            body = re.sub(r"^\s*<\s*" + re.escape(tag) + r"\s*>\s*", "", body,
                          flags=re.IGNORECASE)
            return body.strip() or None
    return None


def _extract_handover_block(text: str):
    """The handover block (between the last HANDOVER_MARK pair)."""
    return _extract_marked(text, HANDOVER_MARK)


def _extract_plan(text: str):
    """The pinned goal + TODO (between the last PLAN_MARK pair), or None."""
    return _extract_marked(text, PLAN_MARK)


def _extract_selfprompt(text: str):
    """The post-compaction self-prompt (between the last SELFPROMPT_MARK pair). The
    model writes this to itself; autostart submits it as the next turn so work
    continues seamlessly after the memory wipe."""
    return _extract_marked(text, SELFPROMPT_MARK)


# --- tolerant handover parsing (research-backed, model-eval/) --------------------
# A model under peak cognitive load (the handover turn is the highest-load turn in a
# session) frequently drops marker discipline: markdown headers instead of fences, a
# JSON object, fuzzy field names, or just prose. The strict fence parser then finds
# nothing and the OLD behaviour no-op'd - which on a FULL context leaves the model
# stuck (can't compact, can't continue). The fix (per the research handover): try
# tiers, and ACCEPT a degraded handover over a no-op. Only truly-empty output no-ops.
_HANDOVER_MIN = 30    # a block shorter than this is treated as empty (parse failure)
_HANDOVER_MAX_TRIES = 3   # total handover-turn attempts before accepting what we have
                          # (completeness retry; separate from _loop's send/429 recovery)

# Escalating corrective retries for a handover the tolerant parser couldn't salvage
# (research: accept a degraded handover over a no-op; only truly-empty output no-ops).
# Retry 1 = terse "start with the marker"; retry 2 = the exact template to copy.
HANDOVER_RETRY_1 = (
    "Your handover blocks were not found. Start your reply with " + HANDOVER_MARK +
    " immediately - no preamble, no other tool. The parser needs the exact fence markers."
)
HANDOVER_RETRY_2 = (
    "Still no usable blocks. Copy this EXACT shape, filling the brackets, nothing "
    "before the first marker:\n" +
    HANDOVER_MARK + "\n<what you were doing, what changed, what's next, open questions>\n" + HANDOVER_MARK + "\n" +
    PLAN_MARK + "\nGOAL: <one line>\nTODO:\n- [ ] <item>\n" + PLAN_MARK + "\n" +
    SELFPROMPT_MARK + "\n<the single next instruction>\n" + SELFPROMPT_MARK + "\nBegin now:"
)

# fuzzy field-name aliases for the JSON/markdown tiers.
_HANDOVER_ALIASES = ("handover", "summary", "context")
_PLAN_ALIASES     = ("plan", "todo", "checklist", "goal")
_NEXT_ALIASES     = ("next", "selfprompt", "self_prompt", "next_step", "next_prompt", "prompt")


def _md_section(text, names):
    """Tier 3: content under a markdown header (## Handover / **Plan** / etc.) up to
    the next header or end. `names` is the alias tuple. Returns the section or None."""
    if not text:
        return None
    pat = (r"(?im)^\s*(?:#{1,6}\s*|\*\*)\s*(?:" + "|".join(names) +
           r")\b[^\n]*\**\s*$")
    m = re.search(pat, text)
    if not m:
        return None
    start = m.end()
    nxt = re.search(r"(?im)^\s*(?:#{1,6}\s*|\*\*)\s*\w", text[start:])
    section = text[start: start + nxt.start()] if nxt else text[start:]
    return section.strip() or None


def _json_field(text, names):
    """Tier 4: a top-level JSON object anywhere in the text, read a fuzzy-named field
    (case-insensitive). Value may be str or list (a TODO). Returns str or None."""
    if not text or "{" not in text:
        return None
    # try the largest brace-balanced span
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    low = {k.lower(): v for k, v in obj.items()}
    for nm in names:
        if nm in low:
            v = low[nm]
            if isinstance(v, list):
                return "\n".join(str(x) for x in v).strip() or None
            return (str(v).strip() or None) if v is not None else None
    return None


def _parse_handover(text: str) -> dict:
    """Tolerant multi-tier parse of a handover turn -> {handover, plan, next, tier}.
    Each field is a string or None. `tier` records the HIGHEST (most-degraded) tier
    any field needed, for observability. Tiers, tried per-field in order:
      1 fences  - the canonical ===HANDOVER=== pair (also XML-close, via _extract_marked)
      3 md      - a markdown header section
      4 json    - a fuzzy-named field in a JSON object
      5 prose   - last resort: split raw content into summary / plan-ish / last line
    (Tier 2 = single-fence+XML, folded into _extract_marked.)"""
    text = text or ""
    tier = 0
    def _pick(fence_mark, md_names, json_names):
        nonlocal tier
        v = _extract_marked(text, fence_mark)
        if v:
            tier = max(tier, 1); return v
        v = _md_section(text, md_names)
        if v:
            tier = max(tier, 3); return v
        v = _json_field(text, json_names)
        if v:
            tier = max(tier, 4); return v
        return None
    handover = _pick(HANDOVER_MARK, _HANDOVER_ALIASES, _HANDOVER_ALIASES)
    plan     = _pick(PLAN_MARK, _PLAN_ALIASES, _PLAN_ALIASES)
    nxt      = _pick(SELFPROMPT_MARK, _NEXT_ALIASES, _NEXT_ALIASES)
    # Tier 5 (semantic): no handover block found at all, but there IS real prose - use
    # it wholesale as the summary so a full-context session is never left stuck.
    if not handover:
        stripped = re.sub(r"(?m)^\s*(?:#{1,6}.*|=+.*=+)\s*$", "", text).strip()
        if len(stripped) >= _HANDOVER_MIN:
            handover = stripped
            tier = max(tier, 5)
    # min-content guard: a too-short block is treated as absent (empty != usable).
    if handover and len(handover) < _HANDOVER_MIN:
        handover = None
    return {"handover": handover, "plan": plan, "next": nxt, "tier": tier}


def _handover_missing(parsed) -> list:
    """Which required handover blocks are still missing (in emit order). The model
    writes all three as marked text; the turn ending is the trigger (no finalize
    tool), and any missing block is re-solicited before we accept what we have.
    This completeness retry recovers an incomplete-but-successful ANSWER - a
    DIFFERENT layer from the send-level overload/429 recovery inside _loop (which
    recovers a failed SEND). Separate budgets; they compose, never share a counter."""
    missing = []
    if not parsed.get("handover"):
        missing.append("handover")
    if not parsed.get("plan"):
        missing.append("plan")
    if not parsed.get("next"):
        missing.append("next-step")
    return missing


def _handover_nudge_missing(missing) -> str:
    """A corrective re-prompt naming exactly which block(s) to emit, and how."""
    how = {
        "handover":  f"the SUMMARY between two {HANDOVER_MARK} markers",
        "plan":      f"the PLAN (GOAL + TODO) between two {PLAN_MARK} markers",
        "next-step": f"the NEXT-STEP self-prompt between two {SELFPROMPT_MARK} markers",
    }
    want = "; ".join(how[m] for m in missing)
    return ("Your handover is incomplete - still missing " + want + ". Emit ONLY the "
            "missing block(s) now as visible text between the exact markers - start with "
            "the marker, no preamble, no other tool.")


# Auto-compaction instruction: the model gets a full tool-capable turn to wrap up,
# then writes BOTH a handover block (@#!, kept as continuing context) and a
# self-prompt (%%^^, autostarted as the next turn). Recent turns are kept verbatim.
# Forced (hard/emergency) handover. IMPORTANT ordering lesson (measured, model-eval/
# tune_handover.py): the OLD version opened with an optional "update docs with your
# tools" step, which invited a chatty preamble ("I'll compact the context...") that
# smaller models emitted and then STOPPED - no blocks, failed handover. The fix that
# tested best (qwen3-coder:30b 4/4 vs a flaky ~3/4 before) is blocks-FIRST, an explicit
# "start your reply with the first marker - no preamble", and durable-doc saving demoted
# to an AFTER option. Keep that shape if you edit this.
AUTO_HANDOVER_INSTR = (
    "Your context is full - compact it yourself now (recent turns are kept; older ones "
    "get replaced by what you write here). You are writing a note to YOURSELF in a fresh "
    "session that has NO memory of this conversation - capture everything that future-you "
    "needs to continue without losing the thread. Write these three blocks IN THIS REPLY, "
    "using the exact markers, then STOP. Start your reply with " +
    HANDOVER_MARK + " - no preamble, no narration, no tool call.\n"
    "1. HANDOVER: a concise summary of the older context for your future self - the goal, "
    "key decisions, current task state (done / in progress / next), WHERE things live "
    "(files, commands), and open threads. Put the summary text BETWEEN two identical " +
    HANDOVER_MARK + " lines (repeat the marker exactly - do NOT invent an XML-style "
    "closing tag):\n" +
    HANDOVER_MARK + "\nyour summary here\n" + HANDOVER_MARK + "\n"
    "2. PLAN: your current goal + TODO (mark done vs next) - this stays PINNED across the "
    "compaction, so keep it accurate. Again between two identical " + PLAN_MARK + " lines:\n" +
    PLAN_MARK + "\nGOAL: ...\nTODO:\n- [x] done\n- [ ] next\n" + PLAN_MARK + "\n"
    "3. SELF-PROMPT: the single next instruction to yourself - the very next thing to do "
    "after compaction, phrased as a prompt you'd act on immediately. Between two identical " +
    SELFPROMPT_MARK + " lines:\n" +
    SELFPROMPT_MARK + "\nwhat to do next\n" + SELFPROMPT_MARK + "\n"
    "Then STOP - just end your reply after the blocks (no tool call). The handover becomes "
    "your kept context; the self-prompt runs as your next turn. If durable notes need "
    "saving, do that AFTER the blocks (and commit ONLY if this is already a git repo - "
    "never init one)."
)

# Soft-zone nudge: gentle, injected inline into the user turn while context is in the
# soft zone. DELIBERATELY soft - "finish, then wrap at a clean break" - not "compact
# now"; a harsh nudge just makes the model stop mid-thought, which is what the hard
# auto-handover is for. {used}/{limit}/{pct} are filled in at injection time.
HANDOVER_NUDGE = (
    "context {used}/{limit} tokens ({pct}%). At {hard} tokens ({hard_pct}%) a handover is "
    "FORCED - history gets compacted whether or not you're at a clean point. To land it "
    "tidily instead, steer toward a clean break; and IF the current task/step is fully "
    "done (nothing half-finished) and the next thing is unrelated, you may call "
    "request_handover NOW to compact at this clean boundary. Don't interrupt work in "
    "progress for it - only at a genuine boundary."
)

# Editable prompts live as text files under the config dir. The built-in prompts
# (system, handover, auto_handover, handover_nudge) live in a protected `system/`
# subdir so a user browsing the
# prompts dir doesn't break them by accident; custom prompts live in the root.
# read_prompt() returns the override file's text if present, else the baked default.
# Reloaded on /reload (and read live by _system()/handover()).
PROMPTS_DIR = CONFIG_PATH.parent / "prompts"
SYSTEM_PROMPTS_DIR = PROMPTS_DIR / "system"
BUILTIN_PROMPTS = {
    "system":         SYSTEM_PROMPT,
    "handover":       HANDOVER_INSTR,
    "auto_handover":  AUTO_HANDOVER_INSTR,     # the lever-pull instruction on an auto-handover turn
    "handover_nudge": HANDOVER_NUDGE,    # the gentle soft-zone "wrap up at a break" nudge
}
EDITOR_FALLBACKS = ("nano", "vi", "vim")


def prompt_file(name):
    """Path of a prompt's override file: built-ins under system/, custom in the root."""
    if name in BUILTIN_PROMPTS:
        return SYSTEM_PROMPTS_DIR / f"{name}.txt"
    return PROMPTS_DIR / f"{name}.txt"


def read_prompt(name):
    """Effective text of a prompt: the override file if it exists, else the baked-in
    default for a built-in, else None for an unknown custom name."""
    f = prompt_file(name)
    try:
        if f.is_file():
            return f.read_text()
    except OSError:
        pass
    return BUILTIN_PROMPTS.get(name)


def list_prompts(all_builtins=False):
    """(builtin_names, custom_names): customs are the .txt files in the prompts root
    (the system/ subdir is excluded by glob's non-recursion). Built-ins are the
    advanced system prompts - by default only the ones the user has ALREADY overridden
    are surfaced (so a fresh menu shows just the user's own prompts, not internal
    machinery); pass all_builtins=True to list every one (they stay editable by name
    regardless, e.g. `/prompt system`)."""
    custom = []
    try:
        custom = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
    except OSError:
        pass
    builtins = sorted(BUILTIN_PROMPTS) if all_builtins \
        else sorted(n for n in BUILTIN_PROMPTS if prompt_file(n).is_file())
    return builtins, custom


def restore_prompt(name):
    """Revert a built-in prompt to its baked default by removing its override file.
    Returns True if an override existed and was removed."""
    if name not in BUILTIN_PROMPTS:
        return False
    f = prompt_file(name)
    try:
        if f.is_file():
            f.unlink()
            return True
    except OSError:
        pass
    return False


def seed_prompt_file(name):
    """Ensure a prompt's override file exists (seeded with the current effective
    text) so an editor has something to open. Returns the Path."""
    f = prompt_file(name)
    try:
        if not f.is_file():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(read_prompt(name) or "")
    except OSError:
        pass
    return f

# The built-in tool schemas (read_file..run_command) + their implementations now live
# in the bundled lean-tools/builtins.py (canon, shipped + always-on). Core loads that
# module once its helpers are defined and DERIVES the surfaces below from it:
#   TOOLS        - the model-facing schema list (active_tools assembles from it)
#   _WRITE_TIER  - names that ride at /leash rw+   (declared tier=="write")
#   _EXEC_TIER   - names that ride at /leash rwe   (declared tier=="exec"); ask_user_to_run too
#   SAFE_TOOLS   - read-only names (tier=="read"): parallel-safe, ask-on-read
#   EXEC_TOOLS   - every builtin name (the remote executor allowlist)
# See "# --- builtin tools: load + derive ---" further down (after _confirm_write).
# `ssh` (run a command on another host) used to live in TOOLS; it moved to an opt-in
# lean-tool (examples/lean-tools/ssh.py) - network egress, not local editing, so it is
# off by default. /connect (remote workspace) and /sh are unaffected.
# The leash ladder, least -> most capable. chat = no tools; r = read only; rw = read +
# write files; rwe = + run commands. A missing cfg.leash defaults to rwe (today's full
# surface), so old configs are unchanged.
LEASH_LEVELS = ("chat", "r", "rw", "rwe")
_LEASH_ALIASES = {"none": "chat", "off": "chat", "chat": "chat",
                  "read": "r", "ro": "r", "r": "r",
                  "write": "rw", "rw": "rw", "readwrite": "rw",
                  "exec": "rwe", "rwe": "rwe", "all": "rwe", "full": "rwe"}
_LEASH_GRANTS = {
    "chat": "no tools - conversation only",
    "r":    "read only - read/list/search; cannot edit or run commands",
    "rw":   "read + write files; cannot run commands",
    "rwe":  "read, write files, AND run commands (full reach)",
}

# The model's permissions note. Injected into the NEXT user turn on a leash change (and
# at startup if restricted) - NOT baked into the cached system prompt, so toggling the
# leash never rewrites the global system cache. The tool surface already reflects the
# tier; this just explains it + how the user raises it.
_LEASH_NOTES = {
    "chat": "Permissions changed: chat-only, no tools. If a task needs one, say tools are "
            "off and the user can enable them with /leash rwe - don't claim you can't.",
    "r":    "Permissions changed: read-only - read/list/search only, no edits or commands. "
            "If an edit or command is needed, say you're read-only and the user can raise it "
            "with /leash rw or /leash rwe.",
    "rw":   "Permissions changed: read-write - read and edit files, but no commands. If a "
            "command is needed, say so; the user can allow it with /leash rwe.",
    "rwe":  "Permissions changed: read-write-execute - full tool surface (read, edit, run "
            "commands) is available now. Use the tools you have.",
}

def _leash_note(leash):
    return _LEASH_NOTES.get(leash, "")


def _norm_leash(s):
    """Map a leash arg (level or alias, any case, -/_ stripped) to a canonical level,
    or None if unrecognized. Pure -> tested."""
    return _LEASH_ALIASES.get((s or "").strip().lower().replace("-", "").replace("_", ""))

# Optional handoff tool (added by active_tools when cfg.ask_user_to_run is on).
# Lets the model ask the operator to run a command (e.g. one needing sudo); the
# operator can edit it, runs it in a real terminal, and the result returns here.
ASK_USER_TOOL = {"type": "function", "function": {
    "name": "ask_user_to_run",
    "description": "Hand a command to the operator to run in their own terminal; the exact command and its exit code return to you (never any password). Use when run_command can't: needs sudo/root, an interactive prompt, a typed password or secret, or your judgement before acting. Also the right fallback when run_command failed because it required one of those. They may edit the command before running. Don't use it for ordinary commands run_command can handle.",
    "parameters": {"type": "object",
        "properties": {"cmd": {"type": "string", "description": "The command to hand over."},
                       "reason": {"type": "string", "description": "Brief why this needs the operator (e.g. 'needs sudo')."}},
        "required": ["cmd"]}}}

# Surfaced ONLY during /handover (added to the tool surface for that window, removed
# after). Calling it is the explicit "I'm done preparing - finalize" trigger: core
# extracts the @#! ... @#! block from the handover turn and replaces history with it.
UPDATE_PLAN_TOOL = {"type": "function", "function": {
    "name": "update_plan",
    "description": ("Set/replace your PINNED plan: the GOAL plus a TODO checklist. It stays "
                    "pinned to the system prompt, visible every turn, and SURVIVES "
                    "compaction/handover - so it's how you keep long-horizon work on track. "
                    "Call it whenever the plan changes: at the start to lay out the GOAL and "
                    "tasks, and again as you finish steps (flip '- [ ]' to '- [x]') or "
                    "re-scope. Pass the FULL plan each time - it replaces the previous one. "
                    "Pass an empty string to clear it. Keep it short. Use it for tasks with "
                    "3+ steps or several distinct sub-tasks; keep only ONE step in progress "
                    "at a time. Don't use it for a single trivial task or a conversational reply."),
    "parameters": {"type": "object",
        "properties": {"plan": {"type": "string", "description":
            "The full plan text, e.g.\nGOAL: ship feature X\nTODO:\n- [x] done step\n- [ ] next step"}},
        "required": ["plan"]}}}


# Surfaced ONLY in the soft context zone (the user has, via handover_soft, opened the
# spend window). Lets the model ELECT a handover at a clean semantic boundary - a
# finished task, before switching to an unrelated one - instead of being nagged with
# no lever, or force-wiped mid-thought when context later hits the hard threshold.
# Cost is NOT the model's call: it can never hand over before the soft zone (the
# user's threshold), only pick the tidiest MOMENT inside the window the user opened.
REQUEST_HANDOVER_TOOL = {"type": "function", "function": {
    "name": "request_handover",
    "description": ("Electively hand over NOW, at a clean stopping point. Only call this "
                    "when the CURRENT task/step is genuinely complete (work committed, "
                    "nothing half-done) and the next thing is a fresh, unrelated task that "
                    "won't need this history - a natural boundary to summarize + reset at. "
                    "It runs the same handover the automatic one does: you summarize the "
                    "older context and write your next-step self-prompt, then history is "
                    "replaced by that summary. Do NOT call it mid-task, and do NOT call it "
                    "just because context is large - the automatic handover covers pure "
                    "size. Prefer finishing cleanly first. Takes no arguments."),
    "parameters": {"type": "object", "properties": {}, "required": []}}}


# Session-scoped episodic memory. A core agent tool (handled in _run_tool like
# update_plan - never routed to a remote executor), so notes always live with the
# LOCAL session and travel with it: they persist in the session JSON and restore 1:1
# on /load or when a saved session is pushed to another leancoder. Entries are
# DTG-stamped and kept like a spooling log (trimmed to cfg.notes_spool lines); reads
# are hard-capped so a recall can never flood context.
NOTE_TOOL = {"type": "function", "function": {
    "name": "note",
    "description": (
        "Your persistent session NOTEBOOK - episodic memory that survives compaction and "
        "travels with the session (saved/restored on /load). Jot decisions, findings, "
        "gotchas you couldn't re-derive from code or git. Actions: add (default; bare `text` "
        "appends a timestamped entry) | recent (newest `n`, default 40) | grep (`pattern`) | "
        "range (`from` alone = since then; `from`+`to` = between; timestamps 'YYYY-MM-DD "
        "HH:MM', bare date = whole day). No args shows recent; reads are capped."),
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "description": "Note to append (add)."},
        "action": {"type": "string", "enum": ["add", "recent", "grep", "range"]},
        "pattern": {"type": "string", "description": "Substring/regex (grep)."},
        "n": {"type": "integer", "description": "How many (recent)."},
        "from": {"type": "string", "description": "Start ts (range)."},
        "to": {"type": "string", "description": "End ts (range; omit = now)."}},
        "required": []}}}


def active_tools(cfg, remote=False, lean_tool_schemas=(), model_tools=True,
                 offer_handover=False):
    """The tool list the model actually sees, gated by the /leash capability ceiling:
    chat -> none; r -> read tools only; rw -> + write tools; rwe -> + run_command +
    ask_user_to_run. Lean-tools are tiered by their `safe` flag: a safe (read-only)
    lean-tool rides at r+, anything else only at rwe (it may write or execute).
    `lean_tool_schemas` is an iterable of (schema, is_safe). `model_tools` is whether the
    ACTIVE MODEL can tool-call at all (a provider may report a chat-only model via its
    tool_support() hook); False drops the whole tool block regardless of leash - a model
    with no tools can't be granted any by raising the ceiling. `offer_handover` adds the
    elective request_handover tool (surfaced only in the soft context zone - the caller
    decides). (`remote` is kept for call-site symmetry; the core tools are identical
    local or remote.)"""
    leash = getattr(cfg, "leash", "rwe")
    if leash == "chat" or not model_tools:
        return []                    # chat-only (user ceiling OR model can't tool-call)
    allow_write = leash in ("rw", "rwe")
    allow_exec = leash == "rwe"
    tools = [UPDATE_PLAN_TOOL, NOTE_TOOL]  # plan upkeep + session notebook: no fs/exec, every non-chat tier
    if offer_handover:               # soft zone: let the model elect a handover at a break
        tools.append(REQUEST_HANDOVER_TOOL)
    for t in TOOLS:
        nm = t["function"]["name"]
        if nm in _WRITE_TIER and not allow_write:
            continue
        if nm in _EXEC_TIER and not allow_exec:
            continue
        # run_command's description carries a {command_timeout} placeholder so the model
        # sees the LIVE foreground timeout (respects /settings changes). Fill it on a
        # shallow copy - never mutate the shared schema dict.
        if nm == "run_command" and "{command_timeout}" in t["function"].get("description", ""):
            t = {**t, "function": {**t["function"],
                 "description": t["function"]["description"].format(
                     command_timeout=getattr(cfg, "command_timeout", 300))}}
        tools.append(t)
    if cfg.ask_user_to_run and allow_exec:   # a handoff to RUN a command -> exec tier
        tools.append(ASK_USER_TOOL)
    # A lean-tool must not shadow a core tool name (or another lean-tool): duplicate
    # tool names make the API 400 ("Tool names must be unique"). Core wins - a
    # lean-tool that reuses a builtin name (e.g. todo.py's `update_plan`, which is
    # already a core tool) is silently dropped from the surface here.
    seen = {t["function"]["name"] for t in tools}
    for sch, is_safe in lean_tool_schemas:
        if not (is_safe or allow_exec):      # non-safe lean-tools may write/run -> rwe only
            continue
        nm = sch.get("function", {}).get("name")
        if nm in seen:                       # collides with a core tool / earlier lean-tool
            continue
        seen.add(nm)
        tools.append(sch)
    return tools


def _min_leash_for(name, lean_safe=None):
    """The lowest /leash level that permits `name`. `lean_safe`: a lean-tool's `safe`
    flag, or None for a core tool. Used to tell the model exactly how to get unblocked."""
    if lean_safe is not None:
        return "r" if lean_safe else "rwe"
    if name in _WRITE_TIER:
        return "rw"
    if name in _EXEC_TIER or name == ASK_USER_TOOL["function"]["name"]:
        return "rwe"
    return "r"                               # read tools (and anything unclassified)


def _leash_allows_tool(leash, name, lean_safe=None):
    """Authoritative DISPATCH-time capability check - the hard lock BEHIND the
    active_tools() surface filter. active_tools decides what the model is OFFERED;
    this decides what the executor will actually RUN. Even if a call for an above-tier
    tool reaches dispatch anyway - a hallucinated or injected tool_call, a replayed or
    loaded message, a tool wrongly leaked onto the surface - it is refused here unless
    `leash` truly permits it. Mirrors active_tools()'s tiering exactly. update_plan /
    request_handover is agent-internal (no fs/exec) and allowed at any tier."""
    if name in ("update_plan", "request_handover", "note"):
        return True
    if leash == "chat":
        return False                         # no tools at all
    allow_write = leash in ("rw", "rwe")
    allow_exec = leash == "rwe"
    if lean_safe is not None:                # a lean-tool: safe -> r+, else rwe only
        return bool(lean_safe) or allow_exec
    if name in _WRITE_TIER:
        return allow_write
    if name in _EXEC_TIER or name == ASK_USER_TOOL["function"]["name"]:
        return allow_exec
    return True                              # read tools (and anything else) ride at r+


def _leash_block_msg(leash, name, lean_safe=None):
    """The tool-result returned to the model when the dispatch gate refuses a call:
    explains it's a HARD lock (not run), why, and how the user can lift it."""
    need = _min_leash_for(name, lean_safe)
    return (f"BLOCKED: '{name}' is above the current '{leash}' capability leash "
            f"({_LEASH_GRANTS.get(leash, leash)}) and was NOT run. This is a hard lock "
            f"enforced at execution - being handed the tool, or guessing/forcing its name, "
            f"does not bypass it. You CAN do this once the user raises the ceiling to "
            f"'{need}' or higher (e.g. /leash {need}): explain what you need to do and why, "
            f"and let them decide. Do not retry this tool until the leash is raised.")


def _model_no_tools_msg(model, name):
    """Returned to the model when the active model itself can't tool-call (provider
    tool_support=False). Distinct from the leash block: raising the leash won't help -
    the limitation is the model, so the fix is to switch models, not change a setting."""
    return (f"BLOCKED: the active model '{model}' does not support tool calling, so '{name}' "
            f"was not run and no tools are available. This is the MODEL's limitation, not a "
            f"permission setting - raising the leash will NOT help. Switch to a tool-capable "
            f"model (/model) to use tools; otherwise continue in conversation only.")


# ----------------------------------------------------------------------------
# Lean-tools: drop a .py in the lean-tools dir to add a tool. Each file defines
#   TOOL = {"name", "description", "parameters", "safe"?}   and   def run(args, cwd)
# Enabled lean-tools surface as normal tools (local AND remote - they're pushed to
# the remote executor on /connect, so the model's surface is identical either
# way). Disabled by default so context stays opt-in; toggle with /tools.
# ----------------------------------------------------------------------------

def _load_lean_tool(path: Path):
    # compile+exec (not importlib) so a reload always re-reads the source -
    # importlib's .pyc cache can serve stale bytecode after a same-second edit.
    mod = types.ModuleType(f"lcleantool_{path.stem}")
    mod.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return mod


def _same_file(a: Path, b: Path) -> bool:
    """True if two paths have byte-identical content. Used to decide whether a
    shadowed duplicate across the [bundled, user] search path is worth warning about:
    an identical copy (e.g. the same plugin present in both dirs on a mirrored box) is
    silently skipped; only a DIFFERENT file under a name already loaded is flagged."""
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _dup_name_skip(prev: Path, f: Path, label: str, name: str) -> bool:
    """First-dir-wins guard shared by the lean-tool and provider loaders: a name
    already loaded from `prev` is skipped; an identical copy across dirs is silent,
    a DIFFERENT file under the same name warns. Always returns True (caller continues)."""
    if not _same_file(prev, f):
        print(yellow(f"{label} {f.name}: name '{name}' already loaded "
                     f"(from {prev.name}); skipping"))
    return True


class LeanToolManager:
    """Discovers lean-tool files and exposes their schemas/handlers. `enabled` is a
    set of lean-tool names (None = all, used by the hermetic executor which only
    ever receives the lean-tools the driver chose to push)."""

    def __init__(self, lean_tools_dir, enabled=None):
        # Accept one dir (str/Path) or a list of dirs (bundled first, then user),
        # mirroring ProviderManager. First dir to claim a tool NAME wins, so a user
        # drop-in can't silently shadow a bundled builtin (e.g. read_file).
        if isinstance(lean_tools_dir, (str, Path)):
            raw = [lean_tools_dir]
        else:
            raw = list(lean_tools_dir or [])
        self.dirs = [Path(d) for d in raw if d]
        self.dir = self.dirs[0] if self.dirs else None   # back-compat (display/first)
        self.enabled = set(enabled) if enabled is not None else None
        self.lean_tools = {}        # name -> {"schema", "run", "safe", "path"}
        self._load()

    def _load(self):
        # Recurse so tools can be grouped in subdirs (lean-tools/web_dev/foo.py).
        # A tool's GROUP is its first path component under its dir ("" = the
        # ungrouped root, shown as "general"); deeper nesting still groups by the
        # top-level subdir. Groups are a UI grouping only - enabling is still by
        # tool name, so the config and remote sync are unchanged.
        for d in self.dirs:
            if not d.is_dir():
                continue
            self._load_dir(d)

    def _load_dir(self, d):
        for f in sorted(d.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            # builtins.py is the BUNDLED builtin tool surface (read_file..run_command):
            # core loads it directly via _load_builtins() (it exports a Tools class +
            # tier schemas, not the TOOL+run shape), so the manager must NOT also scan
            # it as a plugin - that would double-load it and list "builtins" as a tool.
            if f.name == "builtins.py":
                continue
            rel = f.relative_to(d).parts
            group = rel[0] if len(rel) > 1 else ""
            try:
                mod = _load_lean_tool(f)
                spec, run = getattr(mod, "TOOL", None), getattr(mod, "run", None)
                setup = getattr(mod, "setup", None)
                has_tool = isinstance(spec, dict) and callable(run) and spec.get("name")
                if not has_tool and not callable(setup):
                    continue            # neither a tool nor a startup hook: skip
                # identity: the TOOL name for tool lean-tools, else the file stem
                name = spec["name"] if has_tool else f.stem
                if name in self.lean_tools:
                    _dup_name_skip(self.lean_tools[name]["path"], f, "lean-tool", name)
                    continue
                entry = {"path": f, "group": group,
                         "setup": setup if callable(setup) else None}
                if has_tool:
                    entry["schema"] = {"type": "function", "function": {
                        "name": name,
                        "description": spec.get("description", ""),
                        "parameters": spec.get("parameters",
                                               {"type": "object", "properties": {}})}}
                    entry["run"], entry["safe"] = run, bool(spec.get("safe"))
                    # driver_only: a model-facing tool that must run on the DRIVER even
                    # when the session is connected to a remote (it acts on the local
                    # machine / spawns a local process - e.g. dispatch_worker, whose
                    # worker brain runs on the driver). Never pushed to the executor,
                    # never routed remotely; see enabled_paths() + _run_tool routing.
                    entry["driver_only"] = bool(spec.get("driver_only"))
                    # Optional call-line icon: a lean-tool may set TOOL["glyph"] to a
                    # short display string (its own signature icon); it's registered so
                    # _tool_glyph can honour it over the derived category. No glyph = the
                    # category default (from `safe`/tier) still applies.
                    gl = spec.get("glyph")
                    if isinstance(gl, str) and gl.strip():
                        entry["glyph"] = gl.strip()
                        _LEAN_TOOL_GLYPHS[name] = gl.strip()
                self.lean_tools[name] = entry
            except Exception as e:
                print(yellow(f"lean-tool {f.name} failed to load: {e}"))

    def _is_on(self, name):
        return self.enabled is None or name in self.enabled

    def names(self):
        return list(self.lean_tools)

    def group_of(self, name):
        return (self.lean_tools.get(name) or {}).get("group", "")

    def groups(self):
        """Group names present, sorted ("" = root/general sorts first)."""
        return sorted({p.get("group", "") for p in self.lean_tools.values()})

    def names_in_group(self, group):
        return [n for n, p in self.lean_tools.items() if p.get("group", "") == group]

    def desc(self, name):
        """One-line description for the menu: the tool description, or a marker
        for a setup-only (startup hook) lean-tool."""
        p = self.lean_tools.get(name) or {}
        if "schema" in p:
            return p["schema"]["function"]["description"]
        return dim("(startup hook, no tool)")

    def schemas(self):
        """(schema, is_safe) for each enabled tool lean-tool. The safe flag tiers it
        for the /leash ceiling (safe = read-only -> rides at r+; else rwe only)."""
        return [(p["schema"], bool(p.get("safe"))) for n, p in self.lean_tools.items()
                if self._is_on(n) and "schema" in p]

    def enabled_paths(self):
        # only tool lean-tools are pushed to the remote executor (setup() is
        # driver-only and never runs there, so a setup-only lean-tool has nothing
        # to do remotely). A driver_only tool is also skipped: it acts on the local
        # machine (e.g. dispatch_worker spawns the worker brain here), so it must
        # never run in the remote executor.
        return [p["path"] for n, p in self.lean_tools.items()
                if self._is_on(n) and "schema" in p and not p.get("driver_only")]

    def setups(self):
        """(name, setup) for every enabled lean-tool that defines setup()."""
        return [(n, p["setup"]) for n, p in self.lean_tools.items()
                if self._is_on(n) and p.get("setup")]

    def get(self, name):
        return self.lean_tools.get(name)


def _lean_tools_dir(cfg):
    return cfg.lean_tools_dir or str(CONFIG_PATH.parent / "lean-tools")


def _bundled_lean_tools_dir():
    """The code-relative lean-tools dir (bundled with the code, like providers/).
    Holds bundled BUILTIN tools - e.g. read_file..run_command. None when absent
    (e.g. a bare checkout without it)."""
    d = Path(__file__).resolve().parent / "lean-tools"
    return d if d.is_dir() else None


def _lean_tools_dirs(cfg):
    """Lean-tool search path, in order: the bundled dir first, then the user dir.
    First file to claim a tool name wins, so a user drop-in can't shadow a bundled tool.
    Mirrors _provider_dirs EXACTLY - tools and providers now discover the same way: the
    driver scans both the code-relative bundled dir (lean-tools/, bundled with the code)
    and cfg.lean_tools_dir. (builtins.py is the one exception - it's loaded directly by
    _load_builtins() and skipped by the manager scan; see LeanToolManager._load_dir.)"""
    dirs = []
    bundled = _bundled_lean_tools_dir()
    if bundled:
        dirs.append(bundled)
    user = Path(_lean_tools_dir(cfg))
    if user not in dirs:
        dirs.append(user)
    return dirs


# ----------------------------------------------------------------------------
# MCP (Model Context Protocol) client - builtin, generic, zero servers shipped
# ----------------------------------------------------------------------------

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_NS = "mcp__"                 # tool-name namespace: mcp__<server>__<tool>
_MCP_CONNECT_TIMEOUT = 30
_MCP_CALL_TIMEOUT = 120


def _mcp_tool_name(server, tool):
    """Namespaced surface name for an MCP tool: mcp__<server>__<tool>. Keeps MCP
    tools from colliding with core tools / lean-tools / each other."""
    return f"{MCP_NS}{server}__{tool}"


def _mcp_parse_tool_name(name):
    """(server, tool) from a namespaced mcp__<server>__<tool>, or (None, None)."""
    if not name.startswith(MCP_NS):
        return None, None
    rest = name[len(MCP_NS):]
    server, _, tool = rest.partition("__")
    return (server, tool) if server and tool else (None, None)


def _mcp_result_text(result):
    """Normalise an MCP tools/call result to the tool-result string contract.
    Prefers structuredContent, then joins text content blocks, else JSON dumps."""
    if not isinstance(result, dict):
        return str(result)
    if "structuredContent" in result and result["structuredContent"] is not None:
        sc = result["structuredContent"]
        return sc if isinstance(sc, str) else json.dumps(sc, indent=2, ensure_ascii=False)
    content = result.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block)); continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        joined = "\n".join(p for p in parts if p)
        if joined:
            return ("[tool error] " + joined) if result.get("isError") else joined
    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MCP OAuth: sidecar credential store + discovery + dynamic client registration
#
# A DCR-minted client_secret and cached tokens are MACHINE-obtained, so - matching
# the anthropic_plan provider's auth.json - they live in a mode-0600 sidecar under the
# config dir, NOT in config.toml and NOT in an env var. config.toml just carries
# `auth = { type = "oauth" }`; the sidecar mcp_auth/<name>.json holds
# {client_id, client_secret, token_endpoint, registration_endpoint, scope} plus a
# cached {access_token, expires_at}. Human-supplied secrets (static bearer, a
# pre-existing oauth secret) still use *_env and are never written by us.
# ---------------------------------------------------------------------------

def _mcp_auth_dir():
    return CONFIG_DIR / "mcp_auth"


def _mcp_sidecar_path(name):
    return _mcp_auth_dir() / f"{name}.json"


def load_mcp_client(name):
    p = _mcp_sidecar_path(name)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_mcp_client(name, data):
    d = _mcp_auth_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = _mcp_sidecar_path(name)
    p.write_text(json.dumps(data, indent=2))
    try:
        p.chmod(0o600)
    except OSError:
        pass


def delete_mcp_client(name):
    p = _mcp_sidecar_path(name)
    if p.is_file():
        try:
            p.unlink()
        except OSError:
            pass


_CURL_PATH = None   # cached: curl's presence on PATH is stable for the process lifetime


def _require_curl(msg):
    """Cached shutil.which('curl'); raise `msg` if absent. Avoids a PATH scan per call."""
    global _CURL_PATH
    if _CURL_PATH is None:
        _CURL_PATH = shutil.which("curl") or ""
    if not _CURL_PATH:
        raise RuntimeError(msg)


def _mcp_curl_json(args, timeout=15):
    """Run curl with the given extra args, return parsed JSON or None. stdlib-only."""
    _require_curl("MCP OAuth needs curl on PATH")
    r = subprocess.run(["curl", "-s", "--max-time", str(timeout)] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except (json.JSONDecodeError, ValueError):
        return None


def mcp_discover_auth(mcp_url, timeout=10):
    """RFC 9728 -> RFC 8414 discovery. From an MCP endpoint URL, find the authorization
    server metadata (token_endpoint, registration_endpoint). Tries the URL's root origin
    for the well-known docs (gateway URLs look like http://host/mcp/svc/mcp). Returns the
    auth-server metadata dict, or None if discovery fails."""
    parsed = urllib.parse.urlparse(mcp_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    stripped = mcp_url.rstrip("/").rsplit("/mcp", 1)[0]
    as_url = root
    # RFC 9728: protected-resource metadata points at the authorization server(s).
    for base in dict.fromkeys([root, stripped]):
        meta = _mcp_curl_json(["-L", f"{base}/.well-known/oauth-protected-resource"], timeout)
        if isinstance(meta, dict):
            servers = meta.get("authorization_servers") or []
            if servers:
                as_url = servers[0].rstrip("/")
            break
    # RFC 8414: authorization server metadata (token + registration endpoints).
    for base in dict.fromkeys([as_url, root]):
        meta = _mcp_curl_json(["-L", f"{base}/.well-known/oauth-authorization-server"], timeout)
        if isinstance(meta, dict) and meta.get("token_endpoint"):
            return meta
    return None


def mcp_register_client(reg_endpoint, registration_key=None, client_name="lean-coder",
                        scope="mcp:access", timeout=10):
    """RFC 7591 dynamic client registration for the client-credentials grant. Returns the
    registration response dict ({client_id, client_secret, ...}) or None. If the endpoint
    is guarded, `registration_key` is sent as a Bearer header."""
    body = {"client_name": client_name,
            "grant_types": ["client_credentials"],
            "token_endpoint_auth_method": "client_secret_post"}
    if scope:
        body["scope"] = scope
    args = ["-X", "POST", reg_endpoint,
            "-H", "Content-Type: application/json",
            "-d", json.dumps(body)]
    if registration_key:
        args += ["-H", f"Authorization: Bearer {registration_key}"]
    resp = _mcp_curl_json(args, timeout)
    if isinstance(resp, dict) and resp.get("client_id"):
        return resp
    return None


class MCPConnection:
    """One live connection to a single MCP server. Two transports, both stdlib-only:
      - stdio: a spawned subprocess, JSON-RPC framed newline-delimited over its stdin/
        stdout (the Claude-Desktop shape: command + args + env).
      - http:  streamable-HTTP via curl (initialize -> Mcp-Session-Id header ->
        notifications/initialized -> tools/list|tools/call), tolerating an SSE or plain
        JSON body. Auth is one Authorization: Bearer header - a static token/env, or an
        OAuth2 client-credentials JWT fetched + cached + refreshed on 401.
    Never raises into the agent loop: connect()/list_tools()/call() return ([]/error
    string) on failure and stash .error. Proven shape lifted from mcp-probe.py."""

    def __init__(self, name, config):
        self.name = name
        self.config = config or {}
        self.transport = (self.config.get("transport") or "").lower() or (
            "http" if self.config.get("url") else "stdio")
        self.tools = []              # raw MCP tool dicts from tools/list
        self.error = ""
        self.connected = False
        self._proc = None            # stdio subprocess
        self._rpc_id = 0
        self._session_id = None      # http: Mcp-Session-Id
        self._token = None           # http: cached bearer/oauth token

    # -- shared ---------------------------------------------------------------
    def _next_id(self):
        self._rpc_id += 1
        return self._rpc_id

    def connect(self, timeout=_MCP_CONNECT_TIMEOUT):
        """Handshake + tools/list. Returns True on success. Idempotent-ish: a second
        call reconnects. Sets .error (and returns False) on any failure."""
        self.error = ""
        try:
            if self.transport == "stdio":
                self._connect_stdio(timeout)
            else:
                self._connect_http(timeout)
            self.connected = True
            return True
        except Exception as e:
            self.error = str(e)
            self.connected = False
            self.close()
            return False

    def list_tools(self):
        return list(self.tools)

    def call(self, tool, arguments, timeout=_MCP_CALL_TIMEOUT):
        """Invoke one tool. Returns the normalised result string. Never raises."""
        if not self.connected and not self.connect():
            return f"error: MCP server '{self.name}' not connected ({self.error})"
        try:
            if self.transport == "stdio":
                res = self._rpc_stdio("tools/call",
                                      {"name": tool, "arguments": arguments or {}}, timeout)
            else:
                res = self._rpc_http("tools/call",
                                     {"name": tool, "arguments": arguments or {}}, timeout)
            return _mcp_result_text(res)
        except Exception as e:
            return f"error calling MCP {self.name}.{tool}: {e}"

    def close(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
            except Exception:
                pass
            self._proc = None
        self.connected = False

    # -- stdio transport ------------------------------------------------------
    def _connect_stdio(self, timeout):
        cmd = self.config.get("command")
        if not cmd:
            raise ValueError("stdio server needs a 'command'")
        argv = [cmd] + list(self.config.get("args") or [])
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (self.config.get("env") or {}).items()})
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
        self._rpc_stdio("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "lean-coder", "version": __version__}}, timeout)
        self._notify_stdio("notifications/initialized")
        res = self._rpc_stdio("tools/list", {}, timeout)
        self.tools = res.get("tools", []) if isinstance(res, dict) else []

    def _send_stdio(self, payload):
        if not self._proc or not self._proc.stdin:
            raise ConnectionError("stdio server not running")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _notify_stdio(self, method, params=None):
        self._send_stdio({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _rpc_stdio(self, method, params, timeout):
        rid = self._next_id()
        self._send_stdio({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.time() + timeout
        # Read newline-delimited JSON until we see our id (skip notifications / other ids).
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                raise ConnectionError("stdio server closed the stream")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                raise RuntimeError(json.dumps(msg["error"])[:240])
            return msg.get("result", {})
        raise TimeoutError(f"no response to {method} within {timeout}s")

    # -- http transport (streamable-HTTP via curl) ----------------------------
    def _auth_header(self):
        auth = self.config.get("auth") or {}
        atype = (auth.get("type") or ("bearer" if (auth.get("token") or auth.get("token_env"))
                                      else "")).lower()
        if atype == "bearer":
            tok = auth.get("token") or os.environ.get(auth.get("token_env", ""), "")
            if not tok:
                raise ValueError("bearer auth: no token / token_env")
            return f"Bearer {tok}"
        if atype == "oauth":
            return f"Bearer {self._oauth_token(auth)}"
        # No auth block, but a DCR sidecar exists -> treat as oauth (creds live in sidecar).
        if not atype and load_mcp_client(self.name):
            return f"Bearer {self._oauth_token({'type': 'oauth'})}"
        return ""     # no auth

    def _oauth_token(self, auth, force=False):
        """OAuth2 client-credentials: POST client_id/secret to the token endpoint, cache
        the access_token (in-memory + a 0600 sidecar with its expiry). Credentials come
        from the sidecar (DCR-minted, preferred) or from config (client_id + a literal/
        client_secret_env). Refetched on force (a 401 retry) or when the cached token is
        near expiry."""
        side = load_mcp_client(self.name) or {}
        # In-memory cache first, then a still-valid sidecar token.
        if not force:
            if self._token:
                return self._token
            cached = side.get("access_token")
            exp = side.get("expires_at", 0)
            if cached and (not exp or time.time() < exp - 60):
                self._token = cached
                return cached
        token_url = (auth.get("token_url") or side.get("token_endpoint"))
        if not token_url:
            raise ValueError("oauth auth: no token_url (config) or token_endpoint (sidecar)")
        client_id = auth.get("client_id") or side.get("client_id", "")
        secret = (auth.get("client_secret") or os.environ.get(auth.get("client_secret_env", ""), "")
                  or side.get("client_secret", ""))
        scope = auth.get("scope") or side.get("scope")
        data = ("grant_type=client_credentials"
                f"&client_id={urllib.parse.quote(client_id)}"
                f"&client_secret={urllib.parse.quote(secret)}")
        if scope:
            data += f"&scope={urllib.parse.quote(scope)}"
        out = self._curl(["-X", "POST", token_url,
                          "-H", "Content-Type: application/x-www-form-urlencoded",
                          "-d", data], _MCP_CONNECT_TIMEOUT)
        try:
            parsed = json.loads(out)
            tok = parsed.get("access_token")
        except json.JSONDecodeError:
            parsed, tok = {}, None
        if not tok:
            raise RuntimeError(f"oauth token fetch failed: {out[:160]}")
        self._token = tok
        # Persist the token + expiry back to the sidecar (only if one already exists, i.e.
        # a DCR/oauth-managed server), so a restart reuses it until expiry.
        if side:
            side["access_token"] = tok
            if parsed.get("expires_in"):
                side["expires_at"] = time.time() + int(parsed["expires_in"])
            save_mcp_client(self.name, side)
        return tok

    def _curl(self, extra_args, timeout, capture_headers=None):
        _require_curl("http MCP transport needs curl on PATH")
        cmd = ["curl", "-s", "--max-time", str(timeout)] + extra_args
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise ConnectionError(r.stderr.strip() or f"curl exited {r.returncode}")
        return r.stdout

    def _http_url(self):
        return self.config["url"]

    def _post_http(self, payload, timeout):
        """POST a JSON-RPC payload; return (headers dict, parsed json or None). Captures
        the Mcp-Session-Id header. Tolerates an SSE ('data: {...}') or plain JSON body.
        Retries once on a 401 by forcing an OAuth token refresh."""
        for attempt in (0, 1):
            _hfd, hf = tempfile.mkstemp()
            os.close(_hfd)
            args = ["-X", "POST", self._http_url(),
                    "-H", "Content-Type: application/json",
                    "-H", "Accept: application/json, text/event-stream",
                    "-D", hf]
            ah = self._auth_header()
            if ah:
                args += ["-H", f"Authorization: {ah}"]
            for hk, hv in (self.config.get("headers") or {}).items():
                # A raw header only wins where 'auth' didn't already set one (auth
                # block takes precedence for Authorization), so both can coexist.
                if hk.lower() == "authorization" and ah:
                    continue
                args += ["-H", f"{hk}: {hv}"]
            if self._session_id:
                args += ["-H", f"Mcp-Session-Id: {self._session_id}"]
            args += ["-d", json.dumps(payload)]
            body = self._curl(args, timeout)
            headers, status = {}, 0
            try:
                with open(hf) as f:
                    for line in f:
                        if line.startswith("HTTP/"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                status = int(parts[1])
                        elif ":" in line:
                            k, _, v = line.partition(":")
                            headers[k.strip().lower()] = v.strip()
            except OSError:
                pass
            finally:
                try: os.remove(hf)
                except OSError: pass
            if status == 401 and attempt == 0 and (self.config.get("auth") or {}).get("type") == "oauth":
                self._oauth_token(self.config["auth"], force=True)
                continue
            return headers, self._sse_json(body)
        return {}, None

    @staticmethod
    def _sse_json(out):
        if not out:
            return None
        try:
            body = out.split("data: ", 1)[1] if "data: " in out else out
            return json.loads(body)
        except Exception:
            return None

    def _connect_http(self, timeout):
        if not self.config.get("url"):
            raise ValueError("http server needs a 'url'")
        self._session_id = None
        headers, j = self._post_http({
            "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize", "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {},
                "clientInfo": {"name": "lean-coder", "version": __version__}}}, timeout)
        if isinstance(j, dict) and j.get("error"):
            raise RuntimeError(json.dumps(j["error"])[:240])
        self._session_id = headers.get("mcp-session-id")
        if self._session_id:
            self._post_http({"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout)
        res = self._rpc_http("tools/list", {}, timeout)
        self.tools = res.get("tools", []) if isinstance(res, dict) else []

    def _rpc_http(self, method, params, timeout):
        _, j = self._post_http({"jsonrpc": "2.0", "id": self._next_id(),
                                "method": method, "params": params}, timeout)
        if j is None:
            raise RuntimeError("could not parse gateway/MCP response")
        if isinstance(j, dict) and "error" in j:
            raise RuntimeError(json.dumps(j["error"])[:240])
        return j.get("result", {}) if isinstance(j, dict) else {}


class MCPManager:
    """Owns the configured MCP servers and their live connections. Only ENABLED
    servers are connected + contribute tools to the surface. Namespaced tool names
    (mcp__<server>__<tool>) route back here via call(). Never raises into the loop -
    a dead server just contributes no tools and stashes its error for /mcp."""

    def __init__(self, servers=None, enabled=None):
        self.servers = dict(servers or {})           # name -> config
        self.enabled = set(enabled or [])
        self.conns = {}                              # name -> MCPConnection (enabled only)

    def _is_on(self, name):
        return name in self.enabled

    def names(self):
        return sorted(self.servers)

    def stale_enabled(self):
        """Names that are ENABLED but have no server DEFINITION - orphaned config (e.g. a
        server whose mcp_servers entry was removed/lost, leaving a dangling mcp_enabled
        name). These contribute zero tools but linger confusingly; /mcp surfaces them as
        'stale' and offers a one-touch nuke. Sorted for stable output."""
        return sorted(n for n in self.enabled if n not in self.servers)

    def drop_stale(self, only=None):
        """Forget enabled-but-undefined names. With `only` (a name or iterable of
        names), drop just those (ignoring any that aren't actually stale); otherwise
        drop every stale name. Returns the list actually dropped."""
        stale = self.stale_enabled()
        if only is not None:
            want = {only} if isinstance(only, str) else set(only)
            stale = [n for n in stale if n in want]
        for n in stale:
            self.enabled.discard(n)
        return stale

    def connect_enabled(self):
        """(Re)connect every enabled server; drop connections for disabled/removed ones.
        Returns {name: (ok, ntools_or_error)}. Safe to call repeatedly."""
        report = {}
        for name in list(self.conns):
            if name not in self.enabled or name not in self.servers:
                self.conns[name].close()
                del self.conns[name]
        for name in self.servers:
            if not self._is_on(name):
                continue
            conn = self.conns.get(name)
            if conn is None or not conn.connected:
                conn = MCPConnection(name, self.servers[name])
                ok = conn.connect()
                self.conns[name] = conn
                report[name] = (ok, len(conn.tools) if ok else conn.error)
            else:
                report[name] = (True, len(conn.tools))
        return report

    def schemas(self):
        """(schema, is_safe) for every tool of every connected enabled server, in the
        tool-schema shape active_tools expects. MCP tools may have side effects, so
        they are NOT marked safe (ride at the rwe tier), same as a non-safe lean-tool."""
        out = []
        for name, conn in self.conns.items():
            if not self._is_on(name):
                continue
            for t in conn.list_tools():
                tn = t.get("name")
                if not tn:
                    continue
                out.append(({"type": "function", "function": {
                    "name": _mcp_tool_name(name, tn),
                    "description": (t.get("description") or "").strip(),
                    "parameters": t.get("inputSchema")
                    or {"type": "object", "properties": {}}}}, False))
        return out

    def has_tool(self, surface_name):
        server, tool = _mcp_parse_tool_name(surface_name)
        return bool(server) and server in self.conns and any(
            t.get("name") == tool for t in self.conns[server].list_tools())

    def call(self, surface_name, args):
        server, tool = _mcp_parse_tool_name(surface_name)
        if not server:
            return f"error: not an MCP tool name: {surface_name}"
        conn = self.conns.get(server)
        if conn is None:
            return f"error: MCP server '{server}' is not enabled/connected"
        return conn.call(tool, args)

    def close(self):
        for conn in self.conns.values():
            conn.close()
        self.conns.clear()


def _connect_mcp_startup(agent):
    """Connect the enabled MCP servers at launch and fold their tools into the
    surface. Best-effort + quiet: a dead server just contributes nothing (its error
    is visible via /mcp). No-op when nothing is enabled. Driver-only (never in the
    remote executor)."""
    mgr = getattr(agent, "mcp", None)
    if not mgr or not mgr.enabled:
        return
    report = mgr.connect_enabled()
    agent.refresh_tools()
    ok = [n for n, (good, _) in report.items() if good]
    bad = [(n, info) for n, (good, info) in report.items() if not good]
    if ok:
        ntools = sum(len(mgr.conns[n].list_tools()) for n in ok if n in mgr.conns)
        print(dim(f"MCP: {len(ok)} server(s) connected, {ntools} tool(s) "
                  f"({', '.join(ok)})"))
    for n, info in bad:
        print(yellow(f"MCP: '{n}' failed to connect: {info}"))


# Names of lean-tools whose setup() has already run this session. setup() may
# monkeypatch (not idempotent), so it runs AT MOST ONCE per tool per process -
# whether at startup or when enabled mid-session via /tools.
_setup_done = set()


def _run_lean_tool_setup(cfg, mgr=None):
    """Driver-only: call setup(lc, cfg) for each enabled lean-tool that defines it
    and hasn't been set up yet this session, passing the module globals (lc) and
    the live config so it can patch internals, register commands, or swap the
    client. NEVER called in the --tool-exec executor. Runs at startup AND on a
    /tools enable (a newly-enabled tool needs its hooks); each tool's setup runs
    at most once (tracked in _setup_done) since setup may monkeypatch."""
    global _registering_owner
    if mgr is None:
        mgr = LeanToolManager(_lean_tools_dirs(cfg), enabled=cfg.lean_tools_enabled)
    for name, setup in mgr.setups():
        if name in _setup_done:
            continue
        _registering_owner = name        # so register_command can attribute /commands
        try:
            setup(globals(), cfg)
            _setup_done.add(name)
        except Exception as e:
            # A throw here silently strips any register_command() calls after it,
            # so make it diagnosable: full traceback under LEANCODER_DEBUG, a
            # pointer otherwise.
            print(yellow(f"lean-tool setup '{name}' failed: {e}"))
            if os.environ.get("LEANCODER_DEBUG"):
                import traceback
                print(dim(traceback.format_exc().rstrip()))
            else:
                print(dim("  (set LEANCODER_DEBUG=1 for the full traceback)"))
    _registering_owner = None


def _providers_dir(cfg):
    return cfg.providers_dir or str(CONFIG_DIR / "providers")


def _bundled_providers_dir():
    """The code-relative providers dir (bundled with the code, like lean-tools).
    Holds bundled providers - e.g. ollama, the default backend. None when absent
    (e.g. a bare checkout without it)."""
    d = Path(__file__).resolve().parent / "providers"
    return d if d.is_dir() else None


def _provider_dirs(cfg):
    """Provider-plugin search path, in order: the bundled canon dir first, then the
    user dir. First file to claim a name wins, so a user drop-in can't silently shadow
    a bundled provider (e.g. ollama)."""
    dirs = []
    bundled = _bundled_providers_dir()
    if bundled:
        dirs.append(bundled)
    user = Path(_providers_dir(cfg))
    if user not in dirs:
        dirs.append(user)
    return dirs


class ProviderManager:
    """Discovers provider plugins in the providers/ dir - each a `.py` with a
    module-level PROVIDER dict (the spec) and an optional helpers-only
    `setup(lc, cfg)` (populate the module's display-helper namespace / register
    commands; it must NOT call register_provider - the manager does that). `enabled`
    is the set of provider names that are active. A provider is its own plugin type,
    NOT a lean-tool: it never ships to a /connect remote and exposes no model tool."""

    def __init__(self, providers_dir, enabled=None):
        # Accept one dir (str/Path) or a list of dirs (bundled first, then user).
        if isinstance(providers_dir, (str, Path)):
            raw = [providers_dir]
        else:
            raw = list(providers_dir or [])
        self.dirs = [Path(d) for d in raw if d]
        self.dir = self.dirs[0] if self.dirs else None   # back-compat (display/first)
        self.enabled = set(enabled or [])
        self.providers = {}        # name -> {"path", "spec", "module", "setup"}
        self._load()

    def _load(self):
        for d in self.dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                try:
                    mod = _load_lean_tool(f)    # same compile+exec loader (fresh module)
                    spec = getattr(mod, "PROVIDER", None)
                    if not (isinstance(spec, dict) and spec.get("name")):
                        print(yellow(f"provider {f.name}: no PROVIDER dict with a name; skipping"))
                        continue
                    name = spec["name"]
                    if name in self.providers:   # first dir wins (bundled shadows a user drop-in)
                        _dup_name_skip(self.providers[name]["path"], f, "provider", name)
                        continue
                    self.providers[name] = {"path": f, "spec": spec, "module": mod,
                                            "setup": getattr(mod, "setup", None)}
                except Exception as e:
                    print(yellow(f"provider {f.name} failed to load: {e}"))

    def names(self):
        return list(self.providers)

    def _is_on(self, name):
        return name in self.enabled

    def desc(self, name):
        return (self.providers.get(name, {}).get("spec") or {}).get("description", "")

    def register_enabled(self, cfg):
        """Register every ENABLED provider into the global registry. Calls each file's
        helpers-only setup(lc, cfg) first (so its display helpers / commands are live),
        then register_provider(PROVIDER). Returns the registered names. Never raises -
        a bad provider is reported and skipped."""
        global _registering_owner
        done = []
        for name, p in self.providers.items():
            if not self._is_on(name):
                continue
            _registering_owner = name      # attribute any /commands this provider adds
            try:
                if callable(p["setup"]):
                    p["setup"](globals(), cfg)     # helpers-only: populate _lc, register commands
                register_provider(p["spec"])       # the manager owns registration (Option A)
                done.append(name)
            except Exception as e:
                print(yellow(f"provider '{name}' setup/register failed: {e}"))
                if os.environ.get("LEANCODER_DEBUG"):
                    import traceback
                    print(dim(traceback.format_exc().rstrip()))
        _registering_owner = None
        return done


def _register_dir_providers(cfg):
    """Driver-only startup: discover the providers/ dir and register the enabled ones
    into the global registry, BEFORE _maybe_autostart_provider so a config-named or
    autostart()ing provider is available. Returns the ProviderManager."""
    mgr = ProviderManager(_provider_dirs(cfg), enabled=cfg.providers_enabled)
    mgr.register_enabled(cfg)
    return mgr


def _lean_tools_rows(manager, show_groups):
    """Build the menu's display rows: ("group", name) headers + ("tool", name)
    entries when grouping by subdir, else a flat list of ("tool", name)."""
    if not show_groups:
        return [("tool", n) for n in manager.names()]
    rows = []
    for g in manager.groups():
        rows.append(("group", g))
        rows += [("tool", n) for n in manager.names_in_group(g)]
    return rows


# ----------------------------------------------------------------------------
# Shared interactive-picker engine. ONE implementation of the raw-mode, scrolling,
# type-to-filter menu that every picker (/tools multiselect, /set + /model + /session
# single-select) drives - so there's no per-menu rewrite and no drift. Three layers:
#   picker_capable()  - the capability gate (real tty + termios + not TERM=dumb)
#   read_key()        - robust escape-tolerant single keypress -> a normalised token
#   run_picker()       - the scrolling-viewport render + input loop (bounded to the
#                        terminal height, so a long list can never overflow/corrupt)
# A caller that can't/shouldn't go interactive (gate false, or the user forced plain
# mode) uses its own numbered-prompt fallback - the bulletproof floor that works on
# ANY terminal, pipe, or OS.
# ----------------------------------------------------------------------------

# Set by a command's `--plain`/`--fallback` arg (see _wants_plain): a session-wide
# escape hatch so a user stuck in a garbled render on a mis-detected terminal can
# force the numbered fallback for the NEXT menu without restarting. One-shot: consumed
# by the next picker_capable() check.
_FORCE_PLAIN_ONCE = False


def _wants_plain(arg):
    """True if a command arg opts out of the rich picker for this invocation
    (`--plain`, `--fallback`, or bare `plain`). Lets any menu command offer an
    emergency escape from a broken render without a restart: e.g. `/tools --plain`,
    `/model --plain`. Returns (wants_plain, remaining_arg)."""
    toks = (arg or "").split()
    if toks and toks[0] in ("--plain", "--fallback", "plain"):
        return True, " ".join(toks[1:]).strip()
    return False, (arg or "").strip()


def picker_capable():
    """Can we safely run the rich raw-mode picker? Requires termios, both stdin AND
    stdout to be real ttys, and a non-dumb TERM. A one-shot force-plain flag (set by
    a `--plain` arg) overrides to False so a user on a mis-detected terminal can
    always fall back. Returns True for rich, False for the numbered fallback."""
    global _FORCE_PLAIN_ONCE
    if _FORCE_PLAIN_ONCE:
        _FORCE_PLAIN_ONCE = False
        return False
    try:
        import termios  # noqa: F401
    except ImportError:
        return False
    if not (sys.stdin.isatty() and _TTY):
        return False
    term = os.environ.get("TERM", "")
    if term in ("", "dumb"):
        return False
    return True


# Normalised key tokens returned by read_key.
_K_UP, _K_DOWN, _K_ENTER, _K_CANCEL, _K_BACK, _K_SPACE, _K_PGUP, _K_PGDN, _K_HOME, _K_END = (
    "up", "down", "enter", "cancel", "back", "space", "pgup", "pgdn", "home", "end")


def read_key(stdin):
    """Read ONE logical keypress from a raw-mode stdin and normalise it to a token.
    Tolerates partial/unknown escape sequences (an unrecognised sequence returns None
    -> caller just redraws, never crashes or mis-fires). Returns a token string, a
    single printable char (for filtering), or None. j/k also move (vim-style), but
    ONLY as bare keys - inside an active filter they're treated as text, so the caller
    passes `filtering` to say whether a bare letter should navigate or type."""
    ch = stdin.read(1)
    if not ch:
        return _K_CANCEL                       # EOF/^D
    if ch == "\x1b":                           # ESC or an escape sequence
        # A bare ESC and the start of a CSI/SS3 sequence both begin with \x1b; in
        # raw mode a blocking read(1) here would hang until the NEXT key, so a lone
        # ESC never fires. Peek with a tiny select() timeout: nothing waiting ==
        # the user pressed ESC on its own -> cancel.
        try:
            waiting = bool(select.select([stdin], [], [], 0.05)[0])
        except (ValueError, TypeError, OSError):
            waiting = True                     # non-selectable stream (tests): fall through
        if not waiting:
            return _K_CANCEL                   # bare ESC
        nxt = stdin.read(1)
        if nxt == "":
            return _K_CANCEL                   # bare ESC
        if nxt == "[" or nxt == "O":           # CSI / SS3
            code = stdin.read(1)
            # multi-char (e.g. ESC[5~ pgup, ESC[3~ del) - swallow to the final byte
            seq = code
            while code and not (code.isalpha() or code == "~"):
                code = stdin.read(1)
                seq += code
            return {
                "A": _K_UP, "B": _K_DOWN, "H": _K_HOME, "F": _K_END,
                "5~": _K_PGUP, "6~": _K_PGDN, "1~": _K_HOME, "4~": _K_END,
            }.get(seq)                          # unknown -> None (ignored)
        return None                             # ESC + other -> ignore (don't cancel)
    if ch in ("\r", "\n"):
        return _K_ENTER
    if ch == "\x03":                            # ^C
        return _K_CANCEL
    if ch in ("\x7f", "\x08"):
        return _K_BACK
    if ch == " ":
        return _K_SPACE
    return ch                                   # a printable char (or control) as-is


class _RawFdReader:
    """A minimal file-like over a raw tty fd: read()/select() go straight to the OS
    (os.read), so no bytes ever hide in Python's stdin buffer. That buffering was the
    arrow-key bug: read(1) on the buffered sys.stdin pulled a whole '\\x1b[A' burst into
    the Python buffer, then read_key's select() peek saw the OS fd empty and treated it
    as a bare ESC -> the menu cancelled on every arrow. Reading unbuffered fixes it."""
    def __init__(self, fd):
        self.fd = fd
    def fileno(self):
        return self.fd
    def read(self, n=1):
        try:
            return os.read(self.fd, n).decode("utf-8", "replace")
        except OSError:
            return ""


def run_picker(render, on_key, height_reserve=2):
    """The shared raw-mode render+input loop with a SCROLLING VIEWPORT. `render(top,
    rows_avail)` prints the menu (the callback decides what to draw within `rows_avail`
    visible rows starting at scroll offset `top`) and returns the number of terminal
    lines it emitted; on the next tick we jump the cursor back up exactly that many and
    redraw, so the region is bounded to the screen and a long list scrolls instead of
    overflowing. `on_key(token)` handles one normalised key and returns either None
    (keep looping) or a ("done", value) / ("cancel", None) result. Restores the tty on
    every exit. Assumes picker_capable() already passed."""
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    reader = _RawFdReader(fd)      # unbuffered fd reads: keeps escape bursts off the
                                   # Python stdin buffer so read_key's select() is honest
    prev_lines = [0]

    def _draw(first):
        if not first and prev_lines[0]:
            sys.stdout.write(f"\033[{prev_lines[0]}A")
        rows = max(1, _term_rows() - height_reserve)
        n = render(rows)
        sys.stdout.write("\033[J")             # clear anything below the region
        sys.stdout.flush()
        prev_lines[0] = n

    _draw(True)
    try:
        tty.setraw(fd)
        while True:
            token = read_key(reader)
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            if token is not None:
                res = on_key(token)
                if res is not None:
                    return res
            _draw(False)
            tty.setraw(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


def _term_rows():
    """Visible terminal height in rows (best-effort; 24 when unknown)."""
    try:
        return os.get_terminal_size().lines or 24
    except OSError:
        return 24


def _term_cols():
    """Visible terminal width in columns (best-effort; 80 when unknown)."""
    try:
        return os.get_terminal_size().columns or 80
    except OSError:
        return 80


def _fit_line(s, width=None):
    """Truncate a display string to `width` VISIBLE columns (ANSI colour codes don't
    count), so a picker row can NEVER wrap onto a second physical line - wrapping is
    what desyncs the cursor-up redraw math and smears the menu. Any colour left open
    at the cut is closed with a reset. `width` defaults to the terminal width minus 1
    (leave the last column free so no terminal auto-wraps on the final glyph)."""
    if width is None:
        width = max(1, _term_cols() - 1)
    out, vis, i, n, truncated = [], 0, 0, len(s), False
    while i < n:
        m = _ANSI_RE.match(s, i)
        if m:                                    # copy escape verbatim (0 visible width)
            out.append(m.group())
            i = m.end()
            continue
        if vis >= width:
            truncated = True
            break
        out.append(s[i])
        vis += 1
        i += 1
    res = "".join(out)
    if truncated and "\x1b[" in res:             # close any colour left open by the cut
        res += "\033[0m"
    return res


def _wrap_detail(text, max_lines=3, width=None):
    """Wrap a (plain, un-coloured) description to the terminal width for a picker's
    detail footer - how you read the full text of a row that's been truncated to fit.
    Capped at `max_lines`; the last shown line gets an ellipsis if more was cut."""
    text = (text or "").strip()
    if not text:
        return []
    if width is None:
        width = max(20, _term_cols() - 6)         # indent room
    lines = textwrap.wrap(text, width=width) or []
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit_line(lines[-1], width - 1) + "…"
    return lines


def lean_tools_menu(manager):
    """Interactive enable/disable: up/down move, space toggles, enter saves, q
    cancels. Tools in subdirs are shown under a group header whose own row toggles
    the WHOLE group ([x] all on, [-] partial, [ ] off) - so a 'web_dev' set flips
    in one keystroke. Returns the new enabled set, or None if cancelled. Falls
    back to a plain listing when there's no TTY (returns None - edit config)."""
    items = manager.names()
    enabled = set(n for n in items if manager._is_on(n))
    show_groups = len(manager.groups()) > 1
    rows = _lean_tools_rows(manager, show_groups)

    def gstate(g):
        ns = manager.names_in_group(g)
        on = sum(1 for n in ns if n in enabled)
        return "all" if ns and on == len(ns) else ("none" if on == 0 else "partial")

    if not picker_capable():                     # no tty / dumb term / forced plain
        print(bold("lean-tools:"))
        for kind, val in rows:
            if kind == "group":
                print(bold(f"  [{val or 'general'}]"))
            else:
                ind = "    " if show_groups else "  "
                print(f"{ind}[{'x' if val in enabled else ' '}] {val}")
        print(dim("  (no TTY: set lean_tools_enabled in the config file to change)"))
        return None

    st = {"cur": 0, "top": 0}

    def toggle(i):
        kind, val = rows[i]
        if kind == "group":
            ns = manager.names_in_group(val)
            if all(n in enabled for n in ns):
                enabled.difference_update(ns)
            else:
                enabled.update(ns)
        else:
            enabled.discard(val) if val in enabled else enabled.add(val)

    def render(rows_avail):
        cur, top = st["cur"], st["top"]
        # Reserve up to 3 rows at the bottom for the selected tool's full (wrapped)
        # description - the rows are truncated to width so this is how you read the rest.
        sel_kind, sel_val = rows[cur]
        detail = _wrap_detail(manager.desc(sel_val)) if sel_kind == "tool" else []
        body = rows_avail - 1 - len(detail)        # 1 header + detail footer
        body = max(1, body)
        if cur < top:                              # keep the cursor in view
            top = cur
        elif cur >= top + body:
            top = cur - body + 1
        st["top"] = top
        window = rows[top:top + body]
        more_up   = " ↑more" if top > 0 else ""      # sweep-ok
        more_down = " ↓more" if top + body < len(rows) else ""      # sweep-ok

        def row(content):                          # fit to width so nothing wraps
            return "\r" + _fit_line(content) + "\033[K"

        out = [row(bold("lean-tools  ")
                   + dim("(up/down move, space toggle, a all, enter save, q cancel)")
                   + dim(more_up + more_down))]
        for off, (kind, val) in enumerate(window):
            i = top + off
            pointer = cyan(">") if i == cur else " "
            if kind == "group":
                gs = gstate(val)
                mark = green("x") if gs == "all" else (yellow("-") if gs == "partial" else " ")
                out.append(row(f"{pointer} [{mark}] {bold(val or 'general')}"))
            else:
                mark = green("x") if val in enabled else " "
                ind = "  " if show_groups else ""
                out.append(row(f"{pointer} {ind}[{mark}] {val}  {dim(manager.desc(val))}"))
        for d in detail:                           # full desc of the selected tool
            out.append(row("    " + dim(d)))
        sys.stdout.write("\n".join(out))
        return len(out) - 1

    def on_key(k):
        if k in (_K_UP, "k"):
            st["cur"] = (st["cur"] - 1) % len(rows)
        elif k in (_K_DOWN, "j"):
            st["cur"] = (st["cur"] + 1) % len(rows)
        elif k == _K_HOME:
            st["cur"] = 0
        elif k == _K_END:
            st["cur"] = len(rows) - 1
        elif k == _K_PGUP:
            st["cur"] = max(0, st["cur"] - 10)
        elif k == _K_PGDN:
            st["cur"] = min(len(rows) - 1, st["cur"] + 10)
        elif k == _K_SPACE:
            toggle(st["cur"])
        elif k == "a":                           # bulk: all on, or all off if already all on
            if enabled >= set(items):
                enabled.clear()
            else:
                enabled.update(items)
        elif k == _K_ENTER:
            return ("done", enabled)
        elif k in (_K_CANCEL, "q"):
            return ("cancel", None)
        return None

    res = run_picker(render, on_key)
    return res[1] if res[0] == "done" else None


def mcp_servers_menu(manager):
    """Interactive per-SERVER enable/disable for MCP: up/down move, space toggles,
    enter saves, q cancels. Each row shows the server name, transport, connection
    state (tool count or error), and enabled mark. Returns the new enabled set, or
    None if cancelled. Falls back to a plain listing with no TTY."""
    names = manager.names()
    enabled = set(n for n in names if manager._is_on(n))

    def _row_desc(name):
        cfg = manager.servers.get(name, {})
        tp = (cfg.get("transport") or ("http" if cfg.get("url") else "stdio"))
        conn = manager.conns.get(name)
        if conn and conn.connected:
            state = f"{len(conn.list_tools())} tools"
        elif conn and conn.error:
            state = f"error: {conn.error[:40]}"
        else:
            state = "not connected"
        target = cfg.get("url") or cfg.get("command") or "?"
        return f"{tp} · {state} · {target}"

    if not picker_capable():
        print(bold("MCP servers:"))
        for n in names:
            print(f"  [{'x' if n in enabled else ' '}] {n}  {dim(_row_desc(n))}")
        print(dim("  (no TTY: set mcp_enabled in the config file to change)"))
        return None

    st = {"cur": 0, "top": 0}

    def render(rows_avail):
        cur, top = st["cur"], st["top"]
        body = max(1, rows_avail - 1)
        if cur < top:
            top = cur
        elif cur >= top + body:
            top = cur - body + 1
        st["top"] = top
        window = names[top:top + body]
        more_up = " ↑more" if top > 0 else ""      # sweep-ok
        more_down = " ↓more" if top + body < len(names) else ""      # sweep-ok

        def row(content):
            return "\r" + _fit_line(content) + "\033[K"

        out = [row(bold("MCP servers  ")
                   + dim("(up/down move, space toggle, a all, enter save, q cancel)")
                   + dim(more_up + more_down))]
        for off, name in enumerate(window):
            i = top + off
            pointer = cyan(">") if i == cur else " "
            mark = green("x") if name in enabled else " "
            out.append(row(f"{pointer} [{mark}] {name}  {dim(_row_desc(name))}"))
        sys.stdout.write("\n".join(out))
        return len(out) - 1

    def on_key(k):
        if k in (_K_UP, "k"):
            st["cur"] = (st["cur"] - 1) % len(names)
        elif k in (_K_DOWN, "j"):
            st["cur"] = (st["cur"] + 1) % len(names)
        elif k == _K_HOME:
            st["cur"] = 0
        elif k == _K_END:
            st["cur"] = len(names) - 1
        elif k == _K_SPACE:
            name = names[st["cur"]]
            enabled.discard(name) if name in enabled else enabled.add(name)
        elif k == "a":                           # bulk: all on, or all off if already all on
            if enabled >= set(names):
                enabled.clear()
            else:
                enabled.update(names)
        elif k == _K_ENTER:
            return ("done", enabled)
        elif k in (_K_CANCEL, "q"):
            return ("cancel", None)
        return None

    res = run_picker(render, on_key)
    return res[1] if res[0] == "done" else None

# ----------------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------------

_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def dim(s):    return _c("2", s)
def bold(s):   return _c("1", s)
def red(s):    return _c("31", s)
def green(s):  return _c("32", s)
def yellow(s): return _c("33", s)
def blue(s):   return _c("34", s)
def cyan(s):   return _c("36", s)
def italic(s): return _c("3", s)
def magenta(s): return _c("35", s)


def _pct_color(pct):
    """Threshold colour for a percentage meter: blue (ok) -> yellow (mid) -> red."""
    return red if pct >= 85 else (yellow if pct >= 60 else blue)


def _zone_color(zone):
    """Colour for a context-budget zone (see ContextMeter.zone). When auto-handover is
    on, the ctx meter colours by the zone that actually drives behaviour - soft (nudge)
    -> yellow, hard/emergency (forced handover) -> red - instead of raw % of the max
    window, which is misleading when the handover threshold is well below 100% (e.g.
    handover_hard=0.25 forces a handover while a %-of-max meter still reads calm blue)."""
    return {"ok": blue, "soft": yellow, "hard": red, "emergency": red}.get(zone, blue)


def hr() -> str:
    """A full-width horizontal rule used to separate turns. Dim box-draw line on a
    capable TTY (U+2500, or ASCII '-' when the terminal isn't UTF-8); a short ASCII
    dash run when stdout isn't a terminal (so logs/pipes still show a divider). One
    place to change the turn separator."""
    if not _TTY:
        return "-" * 8
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    return dim(GLYPH["rule"] * max(8, cols))


def _fmt_tokens(n) -> str:
    """Compact token count: 950 -> '950', 8237 -> '8.2k', 200000 -> '200k',
    1000000 -> '1M'. Scales to any window size."""
    n = int(n)
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1000:
        v = n / 1000
        return f"{v:.0f}k" if v == int(v) else f"{v:.1f}k"
    return str(n)


def _fmt_age(seconds) -> str:
    """Compact relative age: 5s / 5m / 2h / 3d / 2wk. Used by the /load picker."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m"
    h = m // 60
    if h < 24:
        return f"{h}h"
    d = h // 24
    if d < 14:
        return f"{d}d"
    return f"{d // 7}wk"


def _rl_wrap(s: str) -> str:
    r"""Wrap ANSI SGR codes in readline's \001/\002 'invisible' markers."""
    return re.sub(r"(\x1b\[[0-9;]*m)", "\x01\\1\x02", s)


def _rl_safe(prompt: str) -> str:
    r"""A colored input() prompt readline can measure correctly: ANSI codes wrapped
    in \001/\002 so they are not counted as visible width. Without this a colored
    prompt makes readline miscalculate wrapping, so long input overwrites itself /
    wraps a column early - on every terminal, incl. tmux and termux. No-op when not
    interactive (the markers would otherwise print as raw control chars)."""
    if readline is None or not (sys.stdout.isatty() and sys.stdin.isatty()):
        return prompt
    return _rl_wrap(prompt)


# ----------------------------------------------------------------------------
# Glyphs - define a symbol ONCE with an ASCII fallback; callers use the constant
# (or g()) and never repeat the "does this terminal support it?" check. We pick
# the Unicode form only when stdout's encoding is UTF-8 (Windows consoles and
# minimal TERMs get the ASCII version instead of mojibake). One place to change.
# ----------------------------------------------------------------------------

_UNICODE = "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()


def g(uni: str, alt: str) -> str:
    """The Unicode glyph if the terminal can render it, else the ASCII fallback.
    Exposed for ad-hoc/lean-tool use; most code uses the GLYPH constants below."""
    return uni if _UNICODE else alt


GLYPH = {
    "prompt":   g("›", ">"),
    "prompt_remote": g("»", ">>"),  # prompt glyph when tools run on a REMOTE (like root's $ vs #)
    "bullet":   g("●", "*"),       # assistant message
    "tool":     g("⚙", "*"),       # a tool call (uncategorised / fallback)
    "bg":       g("⚡", "&"),       # a backgrounded (detached) run_command
    "warn":     g("⚠", "!"),
    "edit":     g("✎", "*"),       # files changed
    # Per-CATEGORY call-line icons (derived from tool tiering, not per-tool): the eye
    # scans "read/write/run/net/mcp/meta" at a glance. Each has an ASCII fallback.
    "cat_read": g("◎", "r"),       # read_file, list_files, search_files, safe lean-tools
    "cat_write": g("✎", "w"),      # apply_diff, replace_lines, write_file
    "cat_exec": g("»", "$"),       # run_command, shell_session, ask_user_to_run
    "cat_net":  g("⇅", "@"),       # web_fetch, brave_search, ssh, web_screenshot  # sweep-ok
    "cat_mcp":  g("⧉", "m"),       # any mcp__* tool
    "cat_meta": g("◇", "-"),       # update_plan, note, request_handover
    "userbar":  g("┃", "|"),       # left accent bar down the operator's own turn
    "ok":       g("✓", "+"),
    "no":       g("✗", "x"),
    "think":    g("💭", "..."),    # reasoning
    "ghost":    g("👻", "~"),       # incognito session marker (ASCII fallback ~)
    "dot":      g("·", "-"),       # inline separator
    "ellipsis": g("…", "..."),
    "ret":      g("⏎", "<"),       # newline shown inline
    "rule":     g(chr(0x2500), "-"),  # U+2500 box-draw; chr() keeps it out of the sweep
}

# precompiled once (GLYPH["dot"] is fixed at import): strips the volatile 'N turns'
# token from the status key, which _status_key runs on every prompt.
_STATUS_TURNS_RE = re.compile(r"\s*" + re.escape(GLYPH["dot"]) + r"\s*\d+ turns")


# ----------------------------------------------------------------------------
# Spinner - simple animated activity indicator (distinct frames per phase).
# TTY-gated: a no-op when stdout isn't a terminal (pipes, tests), so it never
# spawns threads or writes control chars in non-interactive use.
# ----------------------------------------------------------------------------

THINK_FRAMES = g("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "|/-\\")   # braille when UTF-8, ASCII spinner otherwise
TOOL_FRAMES = g("◐◓◑◒", "|/-\\")             # moon when UTF-8, ASCII otherwise

# --- lightweight markdown styling for streamed model output --------------------
# The markers are ALWAYS stripped (so a minimal terminal shows clean "Title", not
# "## Title"), and bold/colour are layered on only when the terminal supports them
# (the color helpers are no-ops off a TTY). Line-level + a few inline forms only -
# enough to de-noise headings/bold/code without a real markdown engine.
_MD_H = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_CODE = re.compile(r"`([^`\n]+)`")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
# single-* / single-_ italic: only OUTSIDE the ** case (bold is stripped first),
# and not touching ** pairs, list-bullet "* ", or bare/unbalanced markers. The
# operands reject a marker adjacent to whitespace so "a * b" and "2 * 3" pass through.
_MD_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\*\w])"
                        r"|(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])")


def style_md_line(line: str) -> str:
    """Style one COMPLETE markdown line: strip heading hashes / ** / backticks
    (always), add bold/cyan when the terminal allows. Lists, italics, and other
    text pass through untouched."""
    h = _MD_H.match(line)
    if h:
        inner = _MD_CODE.sub(lambda m: m.group(1), h.group(2))      # strip backticks
        inner = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2), inner)  # strip **
        return bold(inner)
    line = _MD_CODE.sub(lambda m: cyan(m.group(1)), line)
    line = _MD_BOLD.sub(lambda m: bold(m.group(1) or m.group(2)), line)
    line = _MD_ITALIC.sub(lambda m: italic(m.group(1) or m.group(2)), line)
    return line


class StreamStall(ConnectionError):
    """A streaming model request stalled at the socket level. Subclasses
    ConnectionError so the agent loop's cross-provider next-model fallback (which
    catches ConnectionError) still fires - no per-provider `on_error` hook needed.
    `kind` distinguishes the two tiers for callers that care (the eval harness reads
    it off the exception): "prefill" = the server accepted the request but never
    emitted a first token (TTFT deadline tripped); "decode" = the stream went idle
    mid-flight (inter-token deadline tripped)."""
    def __init__(self, kind, where=""):
        self.kind = kind
        what = "first token" if kind == "prefill" else "next token"
        loc = f" from {where}" if where else ""
        super().__init__(
            f"stream stalled ({kind}) waiting for the {what}{loc}: the server "
            f"accepted the request but went idle "
            f"(tune gen_ttft_timeout / gen_idle_timeout).")


def _set_stream_read_timeout(resp, secs):
    """Best-effort: retarget a live urllib response socket's read timeout. urllib's
    single urlopen `timeout` scalar can't express connect-vs-read tiers, so we open
    with the connect deadline then re-arm the socket here for the TTFT / idle read
    phases. Silent no-op if the internals aren't reachable (never breaks a stream)."""
    if not secs:
        return
    try:
        resp.fp.raw._sock.settimeout(secs)
    except Exception:
        pass


def stream_tiered(resp, cfg, where=""):
    """Wrap a streaming urllib response in the 3-tier timeout model and yield its
    raw lines. The provider opens the connection with `cfg.gen_connect_timeout`,
    then iterates THIS instead of `resp` directly; its own should_abort poll and
    line parsing are unchanged. We re-arm the socket to `gen_ttft_timeout` until the
    first line arrives (a socket.timeout before then = "prefill" stall), then to
    `gen_idle_timeout` for the decode phase (a timeout after = "decode" stall).
    Prod defaults (ttft=600, idle=None) reproduce the old single 600s read timeout
    exactly - idle=None leaves the ttft deadline in place, so a live-but-slow stream
    is never tripped. Raises StreamStall (a ConnectionError) on a stall."""
    ttft = getattr(cfg, "gen_ttft_timeout", None) or 600
    idle = getattr(cfg, "gen_idle_timeout", None)
    _set_stream_read_timeout(resp, ttft)
    it = iter(resp)
    got_first = False
    while True:
        try:
            raw = next(it)
        except StopIteration:
            return
        except socket.timeout:
            raise StreamStall("prefill" if not got_first else "decode", where)
        if not got_first:
            got_first = True
            _set_stream_read_timeout(resp, idle)   # None = keep ttft (prod)
        yield raw


class MarkdownStream:
    """Line-buffered styler for streamed output: feed() chunks, it writes styled
    complete lines and holds the partial tail until flush(). The ``` fence
    delimiter lines are dropped (noise); lines INSIDE a fence are emitted verbatim
    (code must stay exact) - dimmed when colour is on.
    Always de-noises markers; tiny terminals just don't get the ANSI."""

    def __init__(self, write):
        self._write = write
        self._buf = ""
        self._fence = False

    def _emit(self, line, newline):
        if line.lstrip().startswith("```"):
            # toggle fence state but DON'T print the ``` delimiter itself - the
            # markers are noise on a terminal (consistent with stripping #, **).
            self._fence = not self._fence
            return
        elif self._fence:
            out = dim(line)
        else:
            out = style_md_line(line)
        self._write(out + ("\n" if newline else ""))

    def feed(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line, True)

    def flush(self):
        if self._buf:
            self._emit(self._buf, False)
            self._buf = ""

_active_spinner = None             # the TOP spinner (so _ask() can pause it before prompting)
_spinner_stack = []                # nesting stack: only the top spinner paints the status line


WATCHDOG_ELAPSED_AT = 6      # s before the status starts showing an elapsed counter
WATCHDOG_HINT_AT = 25        # s before it adds the "how to bail out" hint

# A remote round-trip (over the ssh master) can STALL - not die - when the link
# blips (you muck with network settings, roaming, NAT rebind): the socket neither
# delivers data nor EOFs, so the read just waits. We ride it out (a blip usually
# recovers, and the reconnect path only fires on a real drop/EOF), but after this
# many seconds we surface a spinner ('waiting on <host> ...') so a stall never
# looks like a dead blank screen, and ^C stays live to bail. Not a timeout - it
# never forces a teardown (that would fight reconnect on a link about to recover).
REMOTE_WAIT_GRACE = 5        # s of silence before showing the 'waiting on remote' spinner


def _activity_label(base, elapsed, pid=None):
    """The status text for an in-progress activity. Below the threshold it's just
    the label ('thinking', 'read_file', ...); once work runs long it grows an
    elapsed counter and then an escape hint - so a slow OR wedged step never looks
    like a dead blank screen, and the user always sees what's happening + how to
    get out. Pure (no I/O) so it's unit-tested headless."""
    base = base or "working"
    if elapsed < WATCHDOG_ELAPSED_AT:
        return base
    out = f"{base} ({int(elapsed)}s)"
    if elapsed >= WATCHDOG_HINT_AT:
        out += "  ^C to stop" + (f" (kill {pid} from another shell if stuck)" if pid else "")
    return out


class Spinner:
    def __init__(self, label="", frames=THINK_FRAMES, color=dim, interval=0.1):
        self.label, self.frames = label, frames
        self.color, self.interval = color, interval
        self._stop = None
        self._thread = None
    def start(self):
        global _active_spinner
        # A live elapsed/escape ticker runs in BOTH modes so a long or wedged step
        # keeps updating (the worker thread is separate from the stuck main thread).
        # Spinners NEST (a mid-tool-call reconnect starts connect()'s stage spinners
        # while the outer tool spinner is still alive). Only the TOP of the stack may
        # paint the shared status line - otherwise two ticker threads fight over it
        # every ~0.1s and the display flip-flops. On start we suspend the current top
        # (stop its thread, leave it on the stack) and become the new top; on stop we
        # pop and RESUME whoever was underneath.
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        pid = os.getpid()

        if _spinner_stack:                       # pause the current top's painter
            _spinner_stack[-1]._pause()
        _spinner_stack.append(self)
        _active_spinner = self

        if _active_composer is not None:     # composer owns the status line
            comp = _active_composer

            def tick_composer():
                comp.status = self.label
                while not self._stop.wait(0.5):
                    base = _activity_label(self.label, time.monotonic() - self._t0, pid)
                    q = len(getattr(comp, "queued", []))
                    comp.status = base + (f"  ·  {q} queued (^C to send)" if q else "")

            self._thread = threading.Thread(target=tick_composer, daemon=True)
            self._thread.start()
            return self
        if not _TTY:
            return self

        def run():
            for fr in itertools.cycle(self.frames):
                if self._stop.is_set():
                    break
                lbl = _activity_label(self.label, time.monotonic() - self._t0, pid)
                sys.stdout.write("\r\033[K" + self.color(f"{fr} {lbl}"))
                sys.stdout.flush()
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def _pause(self):
        """Stop this spinner's painter thread WITHOUT popping it from the stack -
        it stays suspended while a nested spinner is on top, and _resume() restarts
        it when the nested one pops."""
        if self._stop is not None:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        self._thread = None

    def _resume(self):
        """Restart this spinner's painter after a nested spinner popped off above
        it. Keeps its original t0 so the elapsed counter is continuous."""
        pid = os.getpid()
        self._stop = threading.Event()
        if _active_composer is not None:
            comp = _active_composer

            def tick_composer():
                comp.status = self.label
                while not self._stop.wait(0.5):
                    base = _activity_label(self.label, time.monotonic() - self._t0, pid)
                    q = len(getattr(comp, "queued", []))
                    comp.status = base + (f"  ·  {q} queued (^C to send)" if q else "")

            self._thread = threading.Thread(target=tick_composer, daemon=True)
            self._thread.start()
            return
        if not _TTY:
            return

        def run():
            for fr in itertools.cycle(self.frames):
                if self._stop.is_set():
                    break
                lbl = _activity_label(self.label, time.monotonic() - self._t0, pid)
                sys.stdout.write("\r\033[K" + self.color(f"{fr} {lbl}"))
                sys.stdout.flush()
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        global _active_spinner
        if self._stop is not None:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        self._stop = self._thread = None
        # Pop from the stack (we may not be the top if stops interleave oddly).
        if self in _spinner_stack:
            _spinner_stack.remove(self)
        if _active_composer is not None:
            if _spinner_stack:                   # resume whoever's underneath
                _active_spinner = _spinner_stack[-1]
                _active_spinner._resume()
            else:
                _active_spinner = None
                _active_composer.status = ""
            return
        if not _TTY:
            _active_spinner = _spinner_stack[-1] if _spinner_stack else None
            return
        sys.stdout.write("\r\033[K")          # clear the spinner line
        sys.stdout.flush()
        if _spinner_stack:                       # resume the one underneath
            _active_spinner = _spinner_stack[-1]
            _active_spinner._resume()
        else:
            _active_spinner = None
            _active_spinner = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class _StageHandle:
    """Yielded by _stage so the body can refine the completion word once it knows more
    (e.g. the probe learns the remote OS -> 'probed · Windows'). Set .done inside the
    block; whatever it holds at exit is what prints."""
    __slots__ = ("done",)
    def __init__(self, done):
        self.done = done


@contextmanager
def _stage(label, done=None):
    """A single connect/push STAGE with feedback that works in ANY terminal. On a TTY:
    an animated Spinner (elapsed ticker, so a slow/wedged step still visibly updates).
    On a dumb terminal / no-TTY (the "old shell" case that looked hung): a plain
    'label…' line up front, then 'done' - no \\r, no ansi. Best-effort: never raises
    into the connect path. `done` overrides the completion word (e.g. 'ready'). Yields a
    handle whose .done can be reassigned inside the block to append late-known detail."""
    h = _StageHandle(done or label)
    if _TTY and _active_composer is None:
        sp = Spinner(label, TOOL_FRAMES, cyan, interval=0.12).start()
        try:
            yield h
        finally:
            sp.stop()
            print(dim(f"  {GLYPH.get('ok', 'v')} {h.done}"))
    else:
        print(dim(f"  {label}…"))
        yield h
        print(dim(f"  {h.done} done"))

@contextmanager
def suppress_echo():
    """Stop the TTY echoing keystrokes for the duration of a turn. While the
    model streams and the spinner rewrites column 0, the terminal would
    otherwise paint the user's typeahead onto the same line and garble it ("I
    can't see my typing / it won't go to a new line"). We clear ECHO but keep
    ICANON, so a follow-up line the user types is still buffered by the line
    discipline and shows up cleanly at the next prompt. No-op without a TTY."""
    try:
        import termios
    except ImportError:
        termios = None
    if termios is None or not (sys.stdin.isatty() and _TTY):
        yield
        return
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except (termios.error, OSError):
        yield
        return
    new = termios.tcgetattr(fd)
    new[3] &= ~termios.ECHO          # c_lflag: drop echo, keep canonical buffering
    try:
        termios.tcsetattr(fd, termios.TCSANOW, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def flush_stdin():
    """Discard any unread terminal input. Called after a turn in the classic
    (non-composer) path so a multi-line paste the user dropped in WHILE the model
    was working can't auto-run line-by-line as a string of separate turns - the
    line discipline buffers each line, and without this each one would be read by
    the next prompt and fired unattended. No-op without a TTY."""
    try:
        import termios
    except ImportError:
        return
    if not (sys.stdin.isatty() and _TTY):
        return
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (termios.error, OSError):
        pass


_active_composer = None            # set while a Composer owns the terminal


class Composer:
    """A bottom-of-screen input line (+ a status line) that stays put while the
    model streams output above it, so the user can type a follow-up without
    fighting the stream. Owns the keyboard during a turn.

    A tool that needs approval never interrupts: it parks a pending confirmation
    as STATE and the agent thread blocks in ask() until it is answered. The
    confirmation is:
      - INERT while the input buffer has text (a keystroke can't silently action
        a y/n that appeared mid-sentence), and
      - ARMED (answerable by y/n, and Enter == yes) only once the buffer is empty
        - i.e. after the user sends the message or backspaces it away.

    This class is pure logic + string rendering, unit-tested headless. The TTY
    plumbing (raw-mode reader thread, scroll region) is layered on top and
    TTY-gated, so without a terminal the feature is simply dormant."""

    INPUT_ROWS = 5                       # the input box is >= 5 rows; wraps then scrolls

    def __init__(self):
        self.buf = ""
        self.cur = 0                     # cursor index into buf (0..len)
        self.queued = []                 # submitted lines, drained after the turn
        self.status = ""                 # activity label for the status line
        self._pending = None             # the confirm question, or None
        self._answer = None
        self._decided = threading.Event()
        self._pstate = ("", False, [])   # input-stream parser state (set here so
                                         # _consume_chunk is testable headless)
        # Line-editing extras (all pure/headless-testable):
        self.history = []                # submitted lines, oldest first
        self._hist_idx = None            # None = editing live buf; else index into history
        self._hist_stash = None          # live buf saved when we start browsing history
        self.completer = None            # optional fn(word)->[options]; set by wiring
        self.comp_display = []           # last completion candidates, for the renderer
        self._read_active = False        # True only inside read_line() (idle reader)
        self._read_eof = False           # Ctrl-D on an empty buffer during read_line
        self._read_wake = threading.Event()   # reader thread -> read_line: line ready/EOF
        self.rule_label = ""             # STATIC idle-rule label (e.g. remote host);
                                         # never animated, unlike self.status (spinner)
        self.remote = False              # True when tools run on a REMOTE: swaps the
                                         # prompt glyph (› -> ») so the input row itself
                                         # signals you're not local (like root's # vs $)
        self._resize_pending = False     # SIGWINCH sets this; the reader thread does
                                         # the actual resize (see on_resize / _reader)

    def armed(self) -> bool:
        """A pending confirm is answerable only when the buffer is empty."""
        return self._pending is not None and self.buf == ""

    @staticmethod
    def wrap_buffer(buf, prompt, cols, max_rows, cursor=None):
        """Pure: lay `prompt + buf` out across up to `max_rows` display rows of
        width `cols`. Returns (rows, cur_row, cur_col):
          - rows: list of strings (1..max_rows), each <= cols wide; the first is
            prefixed with `prompt`.
          - cur_row: 0-based index into `rows` where the cursor is.
          - cur_col: 1-based column (terminal coords) of the cursor on that row.
        `cursor` is the cursor's index into `buf` (0..len); None means end-of-buffer
        (the old behaviour). The display char at each buf index is 1:1 (callers
        substitute \\n with a single glyph before calling), so a buf index maps
        straight onto the laid-out `full` string. When the laid-out text needs more
        than max_rows, the EARLIEST rows are dropped (scroll internally) so the
        cursor's row stays visible. cols/max_rows are floored at 1."""
        cols = max(1, cols)
        max_rows = max(1, max_rows)
        full = prompt + buf
        # chunk into width-cols segments; an empty buffer is one (prompt) row.
        rows = [full[i:i + cols] for i in range(0, len(full), cols)] or [prompt]
        # cursor position in `full`: prompt width + cursor index into buf.
        pos = len(full) if cursor is None else len(prompt) + max(0, min(cursor, len(buf)))
        if pos > 0 and pos % cols == 0:
            # sits exactly at a row boundary -> start of the next row
            cur_row_abs, cur_col = pos // cols, 1
            if cur_row_abs >= len(rows):               # cursor past last char -> fresh row
                rows = rows + [""]
        else:
            cur_row_abs = pos // cols
            cur_col = (pos % cols) + 1
        if len(rows) > max_rows:                       # scroll: keep the tail
            drop = len(rows) - max_rows
            rows = rows[drop:]
            cur_row_abs -= drop
            # Signal that earlier text is scrolled off (so a long buffer doesn't
            # LOOK like it was clipped/lost): mark the top visible row with a leading
            # up-arrow. Only overlays the first char of that row; the full text is
            # intact in the buffer and submits whole on Enter.
            if rows:
                first = rows[0]
                rows[0] = "\u2191" + (first[1:] if first else "")
        cur_row = max(0, min(cur_row_abs, len(rows) - 1))
        return rows, cur_row, cur_col

    def feed(self, ch: str):
        """Process one printable keypress / Enter / backspace. Named navigation
        keys go through feed_key(). Returns 'submit' when a line was queued,
        'answer' when an armed confirm was resolved, else None."""
        if self.armed():                 # empty buffer + pending confirm
            if ch in ("y", "Y", "\r", "\n"):  # Enter on an empty buffer confirms (yes)
                return self._resolve(True)
            if ch in ("n", "N"):
                return self._resolve(False)
            # any other key falls through and starts a new message (de-arms)
        if ch in ("\r", "\n"):
            if self.buf:
                self.queued.append(self.buf)
                self._push_history(self.buf)
                self.buf = ""
                self.cur = 0
                self._hist_reset()
                return "submit"
            return None
        if ch in ("\x7f", "\b"):         # backspace: delete char before cursor
            if self.cur > 0:
                self.buf = self.buf[:self.cur - 1] + self.buf[self.cur:]
                self.cur -= 1
            return None
        if ch.isprintable():
            self.buf = self.buf[:self.cur] + ch + self.buf[self.cur:]
            self.cur += 1
        self.comp_display = []           # any keystroke dismisses the candidate list
        return None

    def feed_key(self, name: str):
        """Handle a recognised control/navigation key (name from _parse). Returns
        the same protocol as feed() ('submit'/'answer'/None). Pure: touches only
        buf/cur/history/queued so it is unit-tested headless."""
        if name != "tab":
            self.comp_display = []       # anything but re-Tab dismisses the list
        buf = self.buf
        if name == "left":
            self.cur = max(0, self.cur - 1)
        elif name == "right":
            self.cur = min(len(buf), self.cur + 1)
        elif name in ("home", "ctrl-a"):
            self.cur = 0
        elif name in ("end", "ctrl-e"):
            self.cur = len(buf)
        elif name == "ctrl-u":               # kill to start of line
            self.buf = buf[self.cur:]
            self.cur = 0
        elif name == "ctrl-k":               # kill to end of line
            self.buf = buf[:self.cur]
        elif name == "ctrl-w":               # delete word before cursor
            i = self.cur
            while i > 0 and buf[i - 1].isspace():
                i -= 1
            while i > 0 and not buf[i - 1].isspace():
                i -= 1
            self.buf = buf[:i] + buf[self.cur:]
            self.cur = i
        elif name == "delete":               # forward-delete
            if self.cur < len(buf):
                self.buf = buf[:self.cur] + buf[self.cur + 1:]
        elif name == "up":
            self._hist_step(-1)
        elif name == "down":
            self._hist_step(1)
        elif name == "newline":              # Ctrl-O / Alt-Enter: insert \n, never submit
            self.buf = buf[:self.cur] + "\n" + buf[self.cur:]
            self.cur += 1
        elif name == "tab":
            self._complete()
        return None

    def feed_paste(self, text: str):
        """Insert pasted text at the cursor as ONE literal edit. Newlines are KEPT
        (so pasted code survives) but normalized to '\\n'; crucially a paste NEVER
        submits - that is the bug that turned a 30-line paste into 30 API turns.
        The user reviews the buffer and presses Enter once to send. A paste also
        de-arms a pending confirm (buffer becomes non-empty), so pasted 'y' can't
        action a y/N. Returns 'paste'."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.buf = self.buf[:self.cur] + text + self.buf[self.cur:]
        self.cur += len(text)
        return "paste"

    # --- history + completion (pure) -----------------------------------------
    def load_history(self, path=None):
        """Load persisted input history (one entry per line; blank-separated
        multi-line entries are stored with literal \\n escaped as \\\\n). Silent
        best-effort - a missing/corrupt file just means an empty history."""
        path = Path(path) if path else HISTORY_PATH
        try:
            raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        self.history = [self._unescape(ln) for ln in raw if ln]

    @staticmethod
    def _unescape(s: str) -> str:
        r"""Reverse save_history's escaping: \\ -> \ and \n -> newline, one pass
        left-to-right so an escaped backslash can't swallow a following 'n'."""
        out, i, n = [], 0, len(s)
        while i < n:
            if s[i] == "\\" and i + 1 < n:
                nxt = s[i + 1]
                out.append("\n" if nxt == "n" else nxt)
                i += 2
            else:
                out.append(s[i])
                i += 1
        return "".join(out)

    def save_history(self, path=None):
        """Persist the trailing HISTORY_MAX entries. Newlines within an entry are
        escaped so each entry stays one physical line. Best-effort."""
        path = Path(path) if path else HISTORY_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [e.replace("\\", "\\\\").replace("\n", "\\n")
                     for e in self.history[-HISTORY_MAX:]]
            path.write_text("\n".join(lines) + ("\n" if lines else ""),
                            encoding="utf-8")
        except OSError:
            pass

    def _push_history(self, line: str):
        """Record a submitted line (skip blanks and immediate duplicates)."""
        if line and (not self.history or self.history[-1] != line):
            self.history.append(line)

    def _hist_reset(self):
        self._hist_idx = None
        self._hist_stash = None

    def _hist_step(self, delta: int):
        """Up/Down through history. Up from the live buffer stashes it; Down past
        the newest entry restores it."""
        if not self.history:
            return
        if self._hist_idx is None:
            if delta < 0:                    # entering history from the live buffer
                self._hist_stash = self.buf
                self._hist_idx = len(self.history) - 1
            else:
                return                       # Down while already live: no-op
        else:
            self._hist_idx += delta
            if self._hist_idx < 0:
                self._hist_idx = 0
            elif self._hist_idx >= len(self.history):   # past newest -> back to live
                self.buf = self._hist_stash or ""
                self.cur = len(self.buf)
                self._hist_reset()
                return
        self.buf = self.history[self._hist_idx]
        self.cur = len(self.buf)

    def _complete(self):
        """Tab-complete the token under the cursor. self.completer is called with
        the text left of the cursor and returns the candidate completions for the
        final token (each a full replacement for the token, e.g. '/model'). On a
        single match we insert the rest of it; on several we insert their common
        prefix and expose the list for the renderer. No completer/no match -> no-op."""
        self.comp_display = []
        if not self.completer:
            return
        left = self.buf[:self.cur]
        # the token being completed = text after the last whitespace in `left`
        sp = max(left.rfind(" "), left.rfind("\t"), left.rfind("\n"))
        word = left[sp + 1:]
        try:
            opts = [o for o in self.completer(left) if o.startswith(word)]
        except Exception:
            opts = []
        if not opts:
            return
        insert = (opts[0] if len(opts) == 1
                  else os.path.commonprefix(opts))[len(word):]
        if insert:
            self.buf = self.buf[:self.cur] + insert + self.buf[self.cur:]
            self.cur += len(insert)
        if len(opts) > 1:
            self.comp_display = opts

    def _resolve(self, val: bool):
        self._answer = val
        self._decided.set()
        return "answer"

    def ask(self, question: str) -> bool:
        """Park a confirmation and block (agent thread) until the user answers -
        which they can only do once their buffer is empty. The tool 'waits'
        here while the user keeps typing freely."""
        self._pending = question
        self._answer = None
        self._decided.clear()
        self._decided.wait()
        self._pending = None
        self.status = ""
        return bool(self._answer)

    def pop_queued(self):
        """Hand back the lines submitted during the turn (FIFO) and clear them."""
        out, self.queued = self.queued, []
        return out

    def status_text(self, frame: str = ""):
        """Line-2 content. A pending confirm wins over everything; Tab-completion
        candidates win over the activity status. Returns (text, armed): the
        renderer greys it when not armed, lights it when armed."""
        if self._pending is not None:
            return (f"{self._pending} [Y/n]", self.armed())
        if self.comp_display:
            return (self._prompt_glyph() + " " + "  ".join(self.comp_display), False)
        if self.status:
            return (f"{frame} {self.status}".strip(), False)
        if self.rule_label:                     # static label (no spinner frame)
            return (self.rule_label, False)
        return ("", False)

    def _prompt_glyph(self) -> str:
        """The input-row glyph: the remote variant when tools run on another box, so
        the prompt itself shows you're not local (falls back like every GLYPH)."""
        return GLYPH["prompt_remote"] if self.remote else GLYPH["prompt"]

    def input_text(self) -> str:
        return self._prompt_glyph() + " " + self.buf

    # --- Input stream parsing (pure; unit-tested headless) --------------------
    # Terminals in bracketed-paste mode wrap a paste in ESC[200~ ... ESC[201~.
    # We must (a) treat everything between as a single literal insert (no submit
    # per embedded newline - that was the runaway), and (b) parse/skip ordinary
    # escape sequences (arrows etc.) without chopping them. Markers and escapes
    # can straddle os.read() boundaries, so the parser carries residual + paste
    # state across chunks.
    PASTE_START = "\x1b[200~"
    PASTE_END = "\x1b[201~"

    # Escape sequences -> navigation key names. Both CSI (ESC [) and SS3 (ESC O)
    # arrow forms are covered because terminals disagree in application mode.
    _ESC_KEYS = {
        "\x1b[A": "up",    "\x1bOA": "up",
        "\x1b[B": "down",  "\x1bOB": "down",
        "\x1b[C": "right", "\x1bOC": "right",
        "\x1b[D": "left",  "\x1bOD": "left",
        "\x1b[H": "home",  "\x1bOH": "home", "\x1b[1~": "home", "\x1b[7~": "home",
        "\x1b[F": "end",   "\x1bOF": "end",  "\x1b[4~": "end",  "\x1b[8~": "end",
        "\x1b[3~": "delete",
        # Insert-a-newline (never submit). Two UNAMBIGUOUS sources only:
        #   - Kitty CSU shift-enter (ESC[13;2u); Konsole can be set to emit this.
        #   - Alt+Enter (ESC CR), which terminals send out of the box.
        # We deliberately DON'T map SS3 'ESC O M' - that's keypad-Enter on some
        # terminals, so treating it as newline would hijack a normal Enter.
        "\x1b[13;2u": "newline", "\x1b\r": "newline",
    }
    # Lone control bytes -> emacs-style editing keys. Ctrl-O is the portable
    # insert-newline that works in ANY terminal (no config, no collision).
    _CTRL_KEYS = {
        "\x01": "ctrl-a", "\x05": "ctrl-e", "\x15": "ctrl-u",
        "\x0b": "ctrl-k", "\x17": "ctrl-w", "\t": "tab",
        "\x0f": "newline",
    }

    @classmethod
    def _key_name(cls, seq: str):
        """Map a complete escape sequence OR a single control byte to a key name,
        or None if it isn't one we handle (caller drops it)."""
        return cls._ESC_KEYS.get(seq) or cls._CTRL_KEYS.get(seq)

    @staticmethod
    def _csi_end(s, i):
        """s[i] == ESC. Index just past a complete escape sequence at i, or -1 if
        it's incomplete (need more bytes before we can classify it)."""
        n = len(s)
        if i + 1 >= n:
            return -1
        c = s[i + 1]
        if c == "[":                         # CSI: params/intermediates, final 0x40-0x7e
            j = i + 2
            while j < n and "\x20" <= s[j] <= "\x3f":
                j += 1
            return j + 1 if j < n else -1
        if c == "O":                         # SS3: ESC O x
            return i + 3 if i + 2 < n else -1
        return i + 2                         # ESC + one char

    @staticmethod
    def _tail_prefix(seg, marker):
        """Length of the longest suffix of `seg` that is a proper prefix of
        `marker` (so a marker split across chunks is held back, not mis-parsed)."""
        for k in range(min(len(seg), len(marker) - 1), 0, -1):
            if marker.startswith(seg[-k:]):
                return k
        return 0

    @classmethod
    def _parse(cls, state, data):
        """Pure stream parser. `state` is (residual, in_paste, paste_chunks).
        Returns (events, new_state) where events is a list of ('char', c) for a
        normal keypress, ('key', name) for a recognised navigation/edit key, and
        ('paste', text) for a completed paste. Unrecognised escape sequences are
        dropped. Trailing bytes that could be an incomplete marker/escape are
        returned as residual to prepend next call."""
        residual, in_paste, paste = state
        s = residual + data
        events, i, n = [], 0, len(s)
        while i < n:
            if in_paste:
                j = s.find(cls.PASTE_END, i)
                if j == -1:                  # end marker not here yet
                    k = cls._tail_prefix(s[i:], cls.PASTE_END)
                    paste.append(s[i:n - k] if k else s[i:])
                    return events, (s[n - k:] if k else "", True, paste)
                paste.append(s[i:j])
                events.append(("paste", "".join(paste)))
                paste, in_paste, i = [], False, j + len(cls.PASTE_END)
                continue
            if s.startswith(cls.PASTE_START, i):
                in_paste, i = True, i + len(cls.PASTE_START)
                continue
            ch = s[i]
            if ch == "\x1b":                 # escape: an incomplete one (maybe a
                end = cls._csi_end(s, i)      # split paste-start marker) is held back
                if end == -1:
                    return events, (s[i:], False, paste)
                name = cls._key_name(s[i:end])   # recognised nav key -> event, else drop
                if name:
                    events.append(("key", name))
                i = end
                continue
            name = cls._CTRL_KEYS.get(ch)    # lone control byte (Tab, ^A, ^E, ...)
            if name:
                events.append(("key", name))
                i += 1
                continue
            events.append(("char", ch))
            i += 1
        return events, ("", in_paste, paste)

    # --- TTY layer (all gated; any failure leaves active False -> classic) ----
    # Mechanism: a scroll region (DECSTBM) confines the model's output to all but
    # the bottom two rows; those two rows are the status line and the input line.
    # The output cursor position is parked in the terminal's own save slot
    # (DECSC/DECRC, ESC 7 / ESC 8) so we never hand-track it across wrapping or
    # scrolling; _draw() only ever positions absolutely and never touches that
    # slot. A single reader thread does both keys (via select, so it can be told
    # to stop) and the status animation tick.

    active = False

    def start_tty(self):
        """Take over the terminal: scroll region + raw(cbreak) input + reader
        thread. Returns True if it engaged, False to fall back to the classic
        prompt (no TTY, no termios, or any setup error)."""
        try:
            import termios, tty
        except ImportError:
            return False
        if not (sys.stdout.isatty() and sys.stdin.isatty() and _TTY):
            return False
        self._out = sys.stdout
        self._fd = sys.stdin.fileno()
        try:
            self._termios = termios
            self._old_term = termios.tcgetattr(self._fd)
            sz = os.get_terminal_size()
            self._rows, self._cols = sz.lines, sz.columns
            self.active = True
            self._frame = 0
            self._lock = threading.RLock()
            reserve = self.INPUT_ROWS + 1                      # status line + input rows
            self._out.write("\n" * reserve + f"\033[1;{self._rows - reserve}r")
            self._out.write(f"\033[{self._rows - reserve};1H\0337")  # park output pos
            self._out.write("\033[?2004h")                     # bracketed paste on
            tty.setcbreak(self._fd)                            # non-canonical; ISIG kept (^C works)
            # tty.setcbreak only clears ECHO on Python 3.12+. On older versions the
            # terminal echoes each keystroke on top of the line we render ourselves
            # ("text types over itself, then wraps a line early"). Clear ECHO
            # explicitly so the composer is the sole renderer, on every version.
            _m = termios.tcgetattr(self._fd)
            _m[3] &= ~termios.ECHO
            termios.tcsetattr(self._fd, termios.TCSANOW, _m)
            self._pstate = ("", False, [])                     # input-parser state
            self._running = True
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            self._redraw_locked()
            self._out.flush()
            return True
        except Exception as e:
            self.active = False
            # Don't hide a fallback on a capable terminal: a silent return here was
            # making a real engagement bug indistinguishable from "no tty". Report
            # it to stderr (one line; the banner already shows the on/off state).
            try:
                sys.stderr.write(dim(f"  composer: TTY setup failed, using classic "
                                     f"input ({type(e).__name__}: {e})\n"))
            except Exception:
                pass
            try:
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_term)
            except Exception:
                pass
            return False

    def stop_tty(self):
        if not self.active:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.4)
        try:
            self._out.write("\033[?2004l")                         # bracketed paste off
            self._out.write("\033[r")                              # reset region
            top = self._rows - self.INPUT_ROWS - 1                 # first reserved (status) row
            for r in range(self.INPUT_ROWS + 1):                  # clear status + input rows
                self._out.write(f"\033[{top + r};1H\033[2K")
            self._out.write(f"\033[{top};1H")                      # cursor where output ended
            self._out.flush()
        except Exception:
            pass
        try:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        self.active = False

    def on_resize(self, *_):
        # SIGWINCH runs async in the MAIN thread and can interrupt a write()
        # mid-sequence (streaming output holds the RLock reentrantly). Doing the
        # terminal rewrite (scroll region + saved cursor) here would corrupt the
        # display and leave typed input invisible. So just flag it; the reader
        # thread (loops <=0.12s) performs the actual resize in a clean context.
        self._resize_pending = True

    def _apply_resize(self):
        """Perform a pending terminal resize. Called from the reader thread (never
        the signal handler) so it can't interrupt an in-flight write()."""
        if not self.active:
            return
        try:
            sz = os.get_terminal_size()
        except OSError:
            return
        with self._lock:
            self._rows, self._cols = sz.lines, sz.columns
            reserve = self.INPUT_ROWS + 1
            try:
                self._out.write(f"\033[1;{self._rows - reserve}r")
                self._out.write(f"\033[{self._rows - reserve};1H\0337")
                self._redraw_locked()
                self._out.flush()
            except Exception:
                pass

    def read_line(self, prompt_status=None, wake_check=None):
        """Block at the IDLE prompt reading ONE submitted line, using the same
        proven scroll-region + reader-thread + _redraw_locked path the mid-turn
        composer uses - so a multi-line paste renders in the bounded input box
        (never the readline taller-than-screen mis-draw), and Tab/history/emacs
        keys/shift-enter all work. `prompt_status` is an optional label for the
        status line (e.g. remote indicator). Returns the submitted string, or
        raises EOFError (Ctrl-D on empty buffer). Raises RuntimeError if it can't
        take the TTY so the caller falls back to input().

        `wake_check` (optional, no-arg callable) is polled once per idle tick
        (~4x/s) while waiting: if it returns a truthy string, read_line abandons
        the wait and returns THAT string as if submitted - the event-driven wake
        path (a finished bg task synthesises a turn with NO operator input). The
        partial buffer being typed is preserved (self.buf untouched)."""
        # Preserve any partial (unsubmitted) buffer carried over from the turn
        # that just ended: keystrokes typed mid-turn but NOT terminated with Enter
        # live in self.buf. pop_queued() only drains Enter-submitted lines, so
        # clearing here silently ate whatever the user was still typing when the
        # turn handed control back. Keep it and park the cursor at its end.
        self.cur = len(self.buf)
        self._read_eof = False
        self._read_wake.clear()
        self._read_active = True
        self.rule_label = prompt_status or ""   # static (no spinner) idle-rule label
        if not self.start_tty():
            self._read_active = False
            self.rule_label = ""
            raise RuntimeError("no tty")
        try:
            while True:
                self._read_wake.wait(0.25)
                if self._read_eof:
                    raise EOFError
                if self.queued:
                    return self.queued.pop(0)
                self._read_wake.clear()
                if wake_check is not None:
                    woke = wake_check()
                    if woke:
                        return woke
        finally:
            self._read_active = False
            self.rule_label = ""
            self.stop_tty()

    def _consume_chunk(self, data: str) -> bool:
        """Turn one raw read into buffer edits and return whether to redraw.

        The runaway-killer: a chunk longer than one character is a paste or an
        escape burst, NOT a deliberate Enter - so its characters (including any
        newlines) are inserted literally and NEVER submit. Only a lone keypress
        goes through feed(), so a single Enter is the only thing that submits a
        line. This holds even when the terminal/multiplexer strips bracketed-paste
        markers (the case that defeated the marker-only fix), because it keys off
        burst length, not the markers. Bracketed pastes are still parsed out and
        treated as the literal paste text; stray escape sequences are dropped.
        Pure w.r.t. the TTY (only touches buf/queued/_pstate), so it is unit
        tested headless."""
        events, self._pstate = self._parse(self._pstate, data)
        # A ('key', ...) event is an escape/control the parser already recognised
        # in full, so it's a real keypress even though its bytes make the chunk
        # long - don't let the burst-heuristic swallow it as literal paste text.
        # The burst rule still applies to plain typed characters: a multi-char run
        # of them is a paste (or bracketed markers were stripped), inserted whole,
        # never submitting per embedded newline.
        keyed = any(k == "key" for k, _ in events)
        burst = len(data) > 1 and not keyed
        literal, changed = [], False
        for kind, payload in events:
            if kind == "paste":
                literal.append(payload)
            elif kind == "key":
                self.feed_key(payload)           # arrows / edit keys / Tab
                changed = True
            elif burst:
                literal.append(payload)          # bursted plain chars -> literal
            else:
                self.feed(payload)               # lone keypress: Enter submits, etc.
                changed = True
        if literal:
            self.feed_paste("".join(literal))
            changed = True
        return changed

    def _reader(self):
        while self._running:
            try:
                r, _, _ = select.select([self._fd], [], [], 0.12)
            except (OSError, ValueError):
                break
            if self._resize_pending:         # SIGWINCH flagged a resize; do it here
                self._resize_pending = False # (clean context, not the signal handler)
                self._apply_resize()
            if r:
                try:
                    data = os.read(self._fd, 4096).decode("utf-8", "ignore")
                except OSError:
                    continue
                if not data:                     # stdin closed
                    if self._read_active:
                        self._read_eof = True
                        self._read_wake.set()
                    continue
                # Idle reader: Ctrl-D on an empty buffer is EOF (matches input()).
                if self._read_active and data == "\x04" and not self.buf:
                    self._read_eof = True
                    self._read_wake.set()
                    continue
                # Hold the lock across the buffer mutation: _consume_chunk edits
                # self.buf/queued while write()/_redraw_locked read them under the
                # same lock. Without this, fast typing during heavy model-output
                # streaming can race and drop/garble a keystroke.
                with self._lock:
                    had = len(self.queued)
                    changed = self._consume_chunk(data)
                    if changed:
                        self._redraw_locked()
                        try:
                            self._out.flush()
                        except Exception:
                            pass
                # Idle reader: a newly queued line ends this read_line() call.
                if self._read_active and len(self.queued) > had:
                    self._read_wake.set()
            elif self.status:                    # idle tick -> animate status
                self._frame += 1
                self._redraw()

    def _redraw(self):
        with self._lock:
            self._redraw_locked()
            try:
                self._out.flush()
            except Exception:
                pass

    def _redraw_locked(self):
        if not self.active:
            return
        rows, cols = self._rows, self._cols
        # status line: a dim rule (with any activity label) or a parked confirm.
        frame = THINK_FRAMES[self._frame % len(THINK_FRAMES)] if self.status else ""
        text, armed = self.status_text(frame)
        if self._pending is not None:               # a confirm replaces the rule
            text = text[:cols]
            sline = bold(yellow(text)) if armed else dim(text)
        else:                                       # a dim rule splits output/input
            ch = GLYPH["rule"]
            label = text.strip()
            if label:                               # embed activity in the rule
                seg = f"{ch}{ch} {label} "
                sline = dim(seg[:cols] + ch * max(0, cols - len(seg)))
            else:
                sline = dim(ch * cols)
        # input box: the buffer wrapped across INPUT_ROWS rows (grows to the cap
        # then scrolls to keep the cursor visible). Newlines render as a glyph so a
        # pasted multi-line buffer wraps by width instead of breaking the layout.
        prompt = self._prompt_glyph() + " "
        disp = self.buf.replace("\n", GLYPH["ret"])
        irows, cur_row, cur_col = self.wrap_buffer(disp, prompt, cols, self.INPUT_ROWS,
                                                   cursor=self.cur)
        status_row = rows - self.INPUT_ROWS         # status sits above the input rows
        try:
            self._out.write(f"\033[{status_row};1H\033[2K{sline}")
            for idx in range(self.INPUT_ROWS):
                line = irows[idx] if idx < len(irows) else ""
                self._out.write(f"\033[{status_row + 1 + idx};1H\033[2K{line}")
            self._out.write(f"\033[{status_row + 1 + cur_row};{cur_col}H")
        except Exception:
            pass

    # file-like: the agent's stdout is pointed here for the turn, so every print
    # lands above the input line (restore output pos -> write -> re-park -> redraw)
    def write(self, text):
        if not self.active:
            return self._out.write(text)
        with self._lock:
            try:
                self._out.write("\0338")      # restore to output position
                self._out.write(text)
                self._out.write("\0337")      # save new output position
                self._redraw_locked()
                self._out.flush()
            except Exception:
                pass
        return len(text)

    def flush(self):
        try:
            self._out.flush()
        except Exception:
            pass


@contextmanager
def composer_session(composer):
    """Run a turn with the composer owning the terminal. Redirects stdout through
    it, routes SIGWINCH to it, and restores everything on exit (including on
    exceptions / ^C). Yields True if it engaged, False if it fell back."""
    global _active_composer
    if not composer.start_tty():
        yield False
        return
    _active_composer = composer
    old_stdout = sys.stdout
    sys.stdout = composer
    old_winch = None
    if hasattr(signal, "SIGWINCH"):
        try:
            old_winch = signal.getsignal(signal.SIGWINCH)
            signal.signal(signal.SIGWINCH, composer.on_resize)
        except (ValueError, OSError):
            old_winch = None
    try:
        yield True
    finally:
        sys.stdout = old_stdout
        _active_composer = None
        if old_winch is not None:
            try:
                signal.signal(signal.SIGWINCH, old_winch)
            except (ValueError, OSError):
                pass
        composer.stop_tty()


@contextmanager
def composer_suspended():
    """Hand the real terminal back to a plain cooked prompt/subprocess while a
    composer is engaged: stop its reader thread, restore the terminal + stdout,
    then re-engage afterwards. Without this, an interactive handoff (readline in
    ask_user_to_run, a live sudo prompt) and the composer's reader BOTH read
    stdin - so only every Nth keystroke lands and echo is off (the command is
    invisible). No-op when no composer owns the terminal."""
    global _active_composer
    comp = _active_composer
    if comp is None or not getattr(comp, "active", False):
        yield
        return
    real_stdout = comp._out
    sys.stdout = real_stdout
    _active_composer = None
    comp.stop_tty()                      # cooked mode, echo on, reader thread joined
    try:
        yield
    finally:
        if comp.start_tty():             # re-pin the input box and resume capture
            _active_composer = comp
            sys.stdout = comp
        # if re-engage fails, stay on the real stdout: the turn degrades to the
        # classic (unpinned) output rather than losing the terminal entirely.


def open_in_editor(path, cfg=None) -> bool:
    """Open `path` in an editor and block until it exits. Editor preference:
    cfg.editor, then $EDITOR, then nano/vi/vim. Returns True if one ran; False if
    none is installed or there's no TTY - the caller should then print the path so
    the user can edit it manually. Hands the terminal over via composer_suspended."""
    cands = []
    if cfg is not None and getattr(cfg, "editor", ""):
        cands.append(cfg.editor)
    if os.environ.get("EDITOR"):
        cands.append(os.environ["EDITOR"])
    cands += list(EDITOR_FALLBACKS)
    editor = next((e for e in cands if shutil.which(shlex.split(e)[0])), None)
    if not editor or not sys.stdin.isatty():
        return False
    try:
        with composer_suspended():
            subprocess.run(shlex.split(editor) + [str(path)])
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Token estimate (no tokenizer dep) - char heuristic, SELF-CALIBRATING
#
# Base heuristic is chars/4, but two things made the raw estimate under-report
# (worst on a restored/fresh session, before the first real prompt_eval lands):
#   1. no PER-MESSAGE framing - every message costs the provider a few tokens of
#      role/delimiter wrapping the char count misses; on a long history this
#      compounds into the ~2x gap seen on restore.
#   2. JSON (tool schemas, tool_call args, tool results) tokenizes DENSER than
#      prose - lots of punctuation/short tokens - so chars/4 undercounts it.
# Rather than hard-code a magic divisor (wrong per model/tokenizer), we CALIBRATE:
# each turn the provider returns a real prompt token count for a history we can
# also estimate, so we learn the true estimate->actual ratio (EMA, clamped) and
# apply it everywhere messages_tokens/est_tokens are used - including the pre-first
# -turn estimate, since the factor is persisted with the session. Converges to the
# real tokenizer within a turn or two and travels across restores.
# ----------------------------------------------------------------------------

_PER_MSG_OVERHEAD = 4        # ~role + delimiter framing tokens the char count misses
_TOK_FACTOR = 1.0            # learned estimate->actual multiplier (see calibrate_tokens)
_TOK_FACTOR_MIN, _TOK_FACTOR_MAX = 0.7, 3.0


def _tok_factor() -> float:
    return _TOK_FACTOR


def set_tok_factor(f) -> None:
    """Restore a persisted calibration factor (e.g. on session load), clamped."""
    global _TOK_FACTOR
    try:
        _TOK_FACTOR = max(_TOK_FACTOR_MIN, min(_TOK_FACTOR_MAX, float(f)))
    except (TypeError, ValueError):
        pass


def calibrate_tokens(raw_estimate: int, actual: int) -> None:
    """Nudge the calibration factor toward actual/raw_estimate for the same history.
    EMA-smoothed (slow, so a single cached/sliced turn can't swing it) and clamped.
    Called with the provider's real prompt count (incl. cache) vs our raw estimate of
    what was sent. No-op on a degenerate input."""
    global _TOK_FACTOR
    if raw_estimate <= 0 or actual <= 0:
        return
    observed = actual / raw_estimate
    observed = max(_TOK_FACTOR_MIN, min(_TOK_FACTOR_MAX, observed))
    _TOK_FACTOR = round(0.7 * _TOK_FACTOR + 0.3 * observed, 4)


def est_tokens(text: str) -> int:
    """Calibrated token estimate for a plain string (chars/4 * learned factor)."""
    return int(((len(text) + 3) // 4) * _TOK_FACTOR)


# Tool-schema token cost is STATIC within a session load (the tool list only changes
# on refresh_tools(), which builds a fresh list object). Re-json.dumps'ing the whole
# schema list on every _raw_messages_tokens call - which runs per tool call, many times
# a turn - is pure waste, so memoize it on the list's identity (id + len guards against
# id reuse after a rebuild). Cheap drop-in; no API change.
_TOOLS_TOK_CACHE = (None, 0, 0)   # (id(tools), len(tools), token_cost)


def _tools_tokens(tools) -> int:
    global _TOOLS_TOK_CACHE
    key_id, key_len = id(tools), len(tools or [])
    cid, clen, cost = _TOOLS_TOK_CACHE
    if cid == key_id and clen == key_len:
        return cost
    cost = (len(json.dumps(tools)) + 3) // 4
    _TOOLS_TOK_CACHE = (key_id, key_len, cost)
    return cost


# User-turn count feeds the status line + /info, both HOT (status prints many times a
# turn). Re-scanning the whole message list each time is an O(n) waste; memoize on the
# list identity + length so it recomputes only when history actually grows/shrinks.
_TURNS_CACHE = (None, 0, 0)   # (id(messages), len(messages), user_turn_count)


def _user_turns(messages) -> int:
    global _TURNS_CACHE
    key_id, key_len = id(messages), len(messages)
    cid, clen, count = _TURNS_CACHE
    if cid == key_id and clen == key_len:
        return count
    count = sum(1 for m in messages if m.get("role") == "user")
    _TURNS_CACHE = (key_id, key_len, count)
    return count


# The message-content scan is O(n) over the whole history plus a json.dumps per stored
# tool_call, and runs several times a turn (ctx meter, zones, status line). Memoize on
# the list identity + length like _tools_tokens/_user_turns. Messages are otherwise
# append-only; the ONE in-place content mutation (_trim_tool_indices stubbing an evicted
# tool result, same length) explicitly busts this cache alongside last_prompt_tokens.
_MSG_TOK_CACHE = (None, 0, None, 0, 0)  # (id(msgs), len(msgs), id(tools), len(tools), cost)


def _invalidate_msg_tokens():
    global _MSG_TOK_CACHE
    _MSG_TOK_CACHE = (None, 0, None, 0, 0)


def _raw_messages_tokens(messages, tools) -> int:
    """UNCALIBRATED char-based size of a full send (messages + tool schemas), including
    per-message framing. This is the quantity we compare against the provider's real
    count to learn the calibration factor, so it must NOT itself be calibrated."""
    global _MSG_TOK_CACHE
    mid, mlen, tid, tlen = id(messages), len(messages), id(tools), len(tools or [])
    cmid, cmlen, ctid, ctlen, cost = _MSG_TOK_CACHE
    if cmid == mid and cmlen == mlen and ctid == tid and ctlen == tlen:
        return cost
    total = 0
    for m in messages:
        c = m.get("content")
        # content is normally a string, but can be a block LIST (image parts) or, from
        # a loaded/synthesised history, a non-string. Size a string by chars; anything
        # else by its JSON length - so len() never crashes on an int and a list of
        # blocks isn't mis-measured as its element COUNT (2) instead of its size.
        if isinstance(c, str):
            clen = len(c)
        elif c is None:
            clen = 0
        else:
            try:
                clen = len(json.dumps(c))
            except (TypeError, ValueError):
                clen = len(str(c))
        total += (clen + 3) // 4
        total += _PER_MSG_OVERHEAD
        for tc in m.get("tool_calls", []) or []:
            total += (len(json.dumps(tc.get("function", {}))) + 3) // 4
    total += _tools_tokens(tools)
    _MSG_TOK_CACHE = (mid, mlen, tid, tlen, total)
    return total


def messages_tokens(messages, tools) -> int:
    """Calibrated estimate of a full send (messages + tool schemas)."""
    return int(_raw_messages_tokens(messages, tools) * _TOK_FACTOR)


def repair_tool_pairs(messages):
    """Guarantee every assistant tool_call is followed by a matching tool result.

    An interrupted turn - ^C between tool calls, a crash, or a 400 mid-batch - can
    leave an assistant message holding tool_calls with no `tool` messages after it
    (the next thing appended is the user's next message). The API then rejects the
    WHOLE history with 'tool_use ids were found without tool_result blocks', and an
    autosave of that state bricks the session permanently (every resume replays it).

    For any assistant whose tool_calls outnumber the tool messages immediately
    following it, synthesize stub results (carrying the matching tool_call_id) for
    the missing ones. Returns a NEW list; the input is untouched. Idempotent on
    already-valid history (returns an equivalent list)."""
    out = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        out.append(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = m.get("tool_calls") or []
            j, got = i + 1, 0
            while j < n and messages[j].get("role") == "tool":
                out.append(messages[j])
                got += 1
                j += 1
            for c in calls[got:]:                 # backfill un-resulted calls
                out.append({
                    "role": "tool",
                    "tool_name": (c.get("function") or {}).get("name", ""),
                    "tool_call_id": c.get("id", ""),
                    "content": "[tool not run: the previous turn was interrupted]",
                })
            i = j
            continue
        i += 1
    return out


# ----------------------------------------------------------------------------
# Ignore matching (.gitignore + .leancoderignore + defaults)
# ----------------------------------------------------------------------------

class IgnoreMatcher:
    def __init__(self, root: Path):
        self.root = root
        self._root_resolved = root.resolve()   # cached: root never changes after init
        self.patterns = list(DEFAULT_IGNORES)
        for name in (".gitignore", ".leancoderignore"):
            f = root / name
            if f.is_file():
                for line in f.read_text(errors="replace").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("!"):
                        self.patterns.append(line)

    def ignored(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self._root_resolved).as_posix()
        except ValueError:
            return False
        except OSError:
            # e.g. a Windows junction / untrusted mount point (WinError 448)
            return True
        if not rel or rel == ".":
            return False
        try:
            is_dir = path.is_dir()
        except OSError:
            is_dir = False
        parts = rel.split("/")
        for pat in self.patterns:
            anchored = pat.startswith("/")
            dironly = pat.endswith("/")
            p = pat.strip("/")
            if not p:
                continue
            if "/" in p:                       # path-ish pattern, match from root
                if _fnmatch(rel, p) or rel.startswith(p + "/"):
                    if not dironly or is_dir or (p in parts):
                        return True
            else:                              # name pattern, match any component
                if anchored:
                    if _fnmatch(parts[0], p):
                        return True
                else:
                    for i, comp in enumerate(parts):
                        if _fnmatch(comp, p):
                            # dir-only pattern must match a directory component
                            if dironly and i == len(parts) - 1 and not is_dir:
                                continue
                            return True
        return False


def _fnmatch(name: str, pat: str) -> bool:
    return fnmatch.fnmatch(name, pat)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class Config:
    host: str = "http://localhost:11434"
    model: str = "qwen3-coder:30b"
    num_ctx: int = 32768
    max_iterations: int = 0          # tool-call rounds per user turn before the loop
                                     # stops and hands control back. 0 = unlimited (the
                                     # default): long agentic runs aren't cut off; set a
                                     # 0 = unlimited) via /set or config for fully
                                     # autonomous runs. This is a power tool - the cap is
                                     # a guardrail, not a wall.
    cwd: Path = field(default_factory=Path.cwd)
    approval: str = "ask"            # ask (confirm each) | session (ask once) | auto (never)
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repeat_penalty: float = 1.05
    think: "bool | None" = None      # None = don't send; True/False = send think param
    auto_num_ctx: bool = False       # detect num_ctx from `ollama show` at startup
    keep_alive: str = "10m"          # ollama: how long the model stays resident after a
                                     # request (keeps the prefill KV cache warm between
                                     # turns). Modest default since the box may be shared;
                                     # set "-1" to pin indefinitely, "0" to unload at once.
    # --- ollama streaming timeouts (3-tier). Prod defaults preserve the old
    # single 600s socket timeout byte-for-byte; raise/lower per box or in eval. ---
    gen_connect_timeout: float = 10.0  # ollama: TCP connect-phase deadline (s)
    gen_ttft_timeout: float = 600.0    # ollama: read deadline until the FIRST token
                                       # arrives (catches a prefill stall - server
                                       # accepts then emits nothing). 600 = old default.
    gen_idle_timeout: "float | None" = None  # ollama: max gap BETWEEN tokens once
                                       # streaming (catches a decode stall). None =
                                       # disabled (old behaviour); a slow-but-live
                                       # stream must not trip it, so set generously.
    auto_evict: bool = False         # continuous tool-output eviction: each round, stub
                                     # acted-on tool results (keep last auto_evict_keep in
                                     # full). A raw-token quota lever - BUT it rewrites
                                     # history mid-prefix, busting the cache it shrinks, so
                                     # it only wins if cache reads are NOT discounted on the
                                     # plan's quota. Defaults OFF until measured. See
                                     # docs/cost-reduction-roadmap.md.
    auto_evict_keep: int = 3         # tool results kept verbatim by auto_evict
    window_messages: int = 0         # bounded context window: send only the last N
                                     # messages (cut at a clean turn boundary) instead
                                     # of the full history. 0 = unbounded (OFF, the
                                     # default) - a loaded session is fully in play and
                                     # the model sees all of it; auto-handover handles
                                     # the size. Set >0 to opt into a hard send-bound
                                     # (cuts tokens every turn, but the model then only
                                     # sees the last N messages of a long session).
                                     # On by default. The clean cut only drops WHOLE prior
                                     # turns (never mid-turn), so the current task is never
                                     # truncated; older turns are simply dropped until
                                     # summarizing compaction backfills them. See
                                     # docs/cost-reduction-roadmap.md.
    window_tokens: "int | str" = "auto"  # bounded context window by TOKENS (the overflow
                                     # backstop): cap each send so it can never exceed the
                                     # model's window, measured with the calibrated meter
                                     # (messages_tokens). System + tool schemas + the env/plan
                                     # tail are ALWAYS kept + counted first; the conversation
                                     # tail is then walked back whole-turn-by-whole-turn until
                                     # the next older turn wouldn't fit, cut at a clean user
                                     # boundary (a tool result is never orphaned).
                                     #   "auto" (DEFAULT): budget = detected ctx_window minus a
                                     #     reply reserve, recomputed each send. A pure BACKSTOP
                                     #     - a no-op every turn EXCEPT the one that would
                                     #     otherwise overflow the server (which ollama silently
                                     #     top-clips + a hosted API 400s). So it never moves the
                                     #     cache boundary in normal use; it only trims the turn
                                     #     that was going over anyway. auto_handover still fires
                                     #     FIRST at its % (the cache-friendly path); this is the
                                     #     last line only. Loses old tail, never sys/tools/env/
                                     #     plan/recent turns - vs the server beheading the top.
                                     #   an int N: a HARD fixed cap (e.g. 4000 for a 4k model).
                                     #   0: OFF. Takes precedence over window_messages when on.
                                     # Non-destructive (shapes the send only).
    auto_handover: bool = True       # self-managing context: when on, the model is shown a
                                     # ctx-budget line and may hand over at a natural break in
                                     # the soft zone; a hard threshold force-hands-over (the
                                     # model pulls the same lever /handover does, then its
                                     # self-prompt is fed back to it). See cost-reduction-roadmap.
    handover_soft: float = 0.70       # soft zone start, as a fraction of the context window:
                                     # nudge the model to wrap up + hand over at a clean break
    handover_hard: float = 0.95       # hard threshold: force a handover at a clean boundary
                                     # (respects the min-interval loop guard)
    handover_emergency: float = 1.00  # emergency stop: compact regardless, bypassing the clean
                                     # boundary AND the loop guard (about to overflow the window)
    handover_min_interval: float = 60.0  # loop guard: min seconds between compactions (~1/min),
                                     # bypassed only by the emergency threshold
    autostart_after_handover: bool = True   # after a compaction, auto-run the model's
                                     # self-prompt as the next turn so work continues (a
                                     # 5s ^C-to-cancel beat precedes it; off = park at prompt)
    handover_overrides: dict = field(default_factory=dict)  # per-model override of the
                                     # handover settings (models fill context at different
                                     # rates): model_id -> {auto, soft, hard, autostart}.
                                     # Anything absent falls back to the global default above.
    auto_compact_interval: int = 0   # NEW auto-COMPACT (distinct from auto-handover): the
                                     # cheap programmatic strip (/compact - stub old tool
                                     # outputs) fired automatically each time context crosses
                                     # a k*interval TOKEN boundary going up. 0 = off (default).
                                     # A first line of defence BEFORE a full handover: e.g.
                                     # 100000 -> strip at 100k, 200k, ... Hysteresis (below)
                                     # stops it re-firing every turn once it strips just under
                                     # a boundary. Keeps the last handover_compact_keep tool
                                     # results in full.
    auto_compact_hysteresis: float = 0.25  # re-arm margin as a fraction of the interval: after
                                     # firing at boundary B, don't re-arm that boundary until
                                     # context drops below B - hysteresis*interval (so a strip
                                     # to just under B can't retrigger next turn). Schmitt trigger.
    auto_compact_keep: int = COMPACT_KEEP  # tool results kept in full by an auto-compact strip
    wake_on_bg_finish: bool = False  # AUTONOMY: when a background task THIS session started
                                     # finishes, WAKE the agent with a synthesised turn (react
                                     # to the result with NO operator input) instead of only
                                     # surfacing the notice on the next human turn. OFF by
                                     # default: an idle-wake loop changes REPL semantics and can
                                     # burn quota unattended. Composer (idle-poll) path only.
    notes_spool: int = 2000          # session-scoped notes (the note tool) are kept like a
                                     # spooling log: append DTG-stamped entries, trim to the
                                     # newest N LINES on write. ~2000 lines ~= 760KB ~= one full
                                     # 200k window; raise for a heavy-logging session. Reads are
                                     # separately hard-capped (400 lines) so a recall can't flood
                                     # context. /set notes_spool <n>.
    ask_user_to_run: bool = True     # expose the ask_user_to_run handoff tool
    composer: bool = True            # pinned bottom input line (type while it works)
    statusline: bool = True          # status rows above each prompt (perms, context
                                     # mgmt, session, model/provider) - so the live
                                     # state is always visible. Auto-off on a non-TTY.
    statusline_every: int = 1        # how often to reprint the full status block:
                                     # 1 = every prompt (default), N = every N prompts,
                                     # 0 = only when it CHANGES (keeps chat history clear).
                                     # A real state change always reprints regardless.
    statusline_iter: int = 0         # reprint the status block every N model iterations
                                     # WITHIN a single long turn (so a constantly-iterating
                                     # model still surfaces ctx/quota/perms). 0 = off (default).
    autosave: bool = True            # autosave the session each turn; auto-load last on start
    auto_update: bool = False        # on launch, the /update lean-tool checks the published
                                     # VERSION and self-updates if newer. OFF by default;
                                     # only acts when the 'update' lean-tool is enabled.
    update_track: str = "stable"     # which /update track to follow: 'stable' (the main
                                     # branch) or 'beta' (pre-release branch). Selects the
                                     # branch /update fetches VERSION + lean_coder.py from.
    command_timeout: int = 300       # foreground run_command timeout (s); long tasks use ' &'
    bg_max_concurrent: int = 5       # max background tasks at once (0 = unlimited)
    worker_max_concurrent: int = 10   # dispatch_worker: max worker agents alive at once (0 = unlimited)
    worker_idle_timeout: int = 1800  # dispatch_worker: worker lease secs; unattended self-kill
    worker_max_iterations: int = 30  # dispatch_worker: agentic-loop cap per worker
    editor: str = ""                 # preferred editor for /prompt etc.; "" -> $EDITOR/nano
    user_name: str = "operator"      # label on the operator's own turn in scrollback
                                     # (yellow left-bar marker so a human turn is easy to
                                     # find in a sea of AI + tool lines)
    leash: str = "rwe"               # capability ceiling: chat (no tools) | r | rw | rwe.
                                     # bounds what the agent is given AT ALL (never blocks
                                     # on something above the ceiling). chat folds in the
                                     # old chat-only mode. The model is told its ceiling.
    confirm_reads: bool = False      # ask-on-read: read tools confirm too, not just writes
    incognito: bool = False          # no session persistence this run; the model is told
                                     # (ephemeral - never written to config; --incognito to start)
    ephemeral: bool = False          # default for /connect: FORCE the embeddable-Python
                                     # runtime on a remote (even if it has a real one) and
                                     # WIPE the cached runtime on teardown - a zero-trace,
                                     # no-install-left-behind mode. Persisted so a user can
                                     # opt into always-wipe; a per-/connect --ephemeral flag
                                     # overrides it for one connection.
    hosts: list = field(default_factory=list)  # tiered failover list (priority order)
    host_models: dict = field(default_factory=dict)  # per-host default model (url -> model)
    model_explicit: bool = False     # model came from --model / env (skip per-host override)
    machines: dict = field(default_factory=dict)  # memorable name -> url alias map
    connect_hosts: dict = field(default_factory=dict)  # name -> ssh target for /connect
    auto_reconnect: bool = False     # on /load of a session that ran remote, reconnect
                                     # to its host automatically (off = ask first)
    lean_tools_dir: str = ""            # "" -> ~/.config/leancoder/lean-tools
    lean_tools_enabled: list = field(default_factory=list)  # enabled lean-tool names
    always_expand: list = field(default_factory=list)  # tool names
                                     # whose full args auto-print after the call line
                                     # (see /expand); default = none (diffs stay
                                     # collapsed to the call line; /expand shows them)
    show_snapshots: bool = False     # show pre-handover safety snapshots in the /load
                                     # picker + /session list (off = hidden so they don't
                                     # clog the menu; /load <exact-name> always works)
    providers_dir: str = ""             # "" -> ~/.config/leancoder/providers
    # MCP (Model Context Protocol) client - builtin, ZERO servers shipped by default.
    # mcp_servers: name -> config dict. A stdio server: {"transport":"stdio","command":..,
    # "args":[..],"env":{..}}. An HTTP server: {"transport":"http","url":..,"auth":{..}}
    # where auth is {"type":"bearer","token":..|"token_env":..} or {"type":"oauth",
    # "token_url":..,"client_id":..,"client_secret":..|"client_secret_env":..,"scope":..}.
    # mcp_enabled: server names whose tools are live on the surface (per-server gate).
    mcp_servers: dict = field(default_factory=dict)
    mcp_enabled: list = field(default_factory=list)
    providers_enabled: list = field(default_factory=list)  # enabled provider plugin names
    provider: str = "ollama"         # active backend - ALWAYS a registered provider now
    provider_settings: dict = field(default_factory=dict)  # name -> {model, num_ctx, thinking, effort, <custom>}
    # --- the DEFAULTS / SESSION-OVERRIDE layer (uniform working-state model) --------
    # config.toml holds DEFAULTS; a session holds per-key OVERRIDES. The live cfg
    # attribute is always the EFFECTIVE value (override if present, else default), so
    # every read site stays `cfg.<key>` unchanged. `_defaults` snapshots the persisted
    # scalars as loaded from config.toml (populated in load_config); save_config writes
    # the scalar lines FROM `_defaults`, never the live/effective value - so a session
    # override (e.g. approval flipped by a loaded worker session) can NEVER leak into
    # config.toml. `session_overrides` is the per-key override map, persisted in the
    # SESSION meta (not config.toml) and re-applied onto the live cfg at load time,
    # RUNTIME-only. Neither is a config.toml scalar - both are classified EPHEMERAL.
    _defaults: dict = field(default_factory=dict)          # config.toml default snapshot
    session_overrides: dict = field(default_factory=dict)  # per-key session overrides

    # --- provider-aware accessors ---------------------------------------------
    # There is no "native" backend any more: the active model is ALWAYS a provider's
    # model, kept in provider_settings[provider]. Ollama is just the bundled,
    # default-enabled provider (provider == "ollama"). Its model/settings live in
    # provider_settings["ollama"] like any other; its transport config (host/hosts/
    # machines/num_ctx) stays top-level, read directly by the built-in ollama
    # provider. `model`/`num_ctx` top-level remain as migration seeds + back-compat
    # aliases (see load_config), so the accessors fall back to them.
    def active_model(self) -> str:
        return self.provider_settings.get(self.provider, {}).get("model") or self.model

    def ctx_window(self) -> int:
        # Prefer the provider's LIVE context_window() hook so the window can't go
        # stale after a models-cache refresh or a code/cap change (the stored num_ctx
        # is only a snapshot taken at activation). Fall back to the stored value
        # (ollama pins a detected/capped num_ctx there; also covers no-provider).
        stored = self.provider_settings.get(self.provider, {}).get("num_ctx") or self.num_ctx
        # ollama pins a detected/capped num_ctx at activation and its context_window()
        # hook can shell out to `ollama show` - too costly for a per-turn status render,
        # and the stored value is already authoritative there. Only hosted providers
        # (cheap cache-backed hook) get the live read, which is where staleness bit.
        if self.provider and self.provider != "ollama":
            try:
                spec = get_provider(self.provider)
                model = self.provider_settings.get(self.provider, {}).get("model")
                if spec and spec.get("context_window") and model:
                    live = int(spec["context_window"](model))
                    if live > 0:
                        return live
            except Exception:
                pass
        return stored

    def handover_for(self, model: "str | None" = None):
        """Resolve compaction settings for `model` (default: the active model).
        Precedence (uniform working-state layer): SESSION override > per-model
        handover_overrides > global default. A /set of a handover_* key is a session
        override and must win over a per-model override; a /set --config writes the
        global default. Models fill context at different rates, so the per-model layer
        sits between. Returns {auto, soft, hard, autostart}."""
        ov = self.handover_overrides.get(model or self.active_model(), {})
        so = self.session_overrides
        def pick(mkey, gkey, gval):
            if gkey in so:                       # session override wins outright
                return getattr(self, gkey)
            return ov.get(mkey, gval)
        return {
            "auto":      pick("auto",      "auto_handover",            self.auto_handover),
            "soft":      pick("soft",      "handover_soft",            self.handover_soft),
            "hard":      pick("hard",      "handover_hard",            self.handover_hard),
            "autostart": pick("autostart", "autostart_after_handover", self.autostart_after_handover),
        }

    def set_active_model(self, name):
        """Set the active backend's model (provider_settings[provider]["model"]).
        For ollama, mirror to the top-level `model` alias so anything still reading
        cfg.model (and the back-compat config line) stays in sync."""
        self.provider_settings.setdefault(self.provider, {})["model"] = name
        if self.provider == "ollama":
            self.model = name

    def setting(self, key, default=None):
        """A per-provider knob (thinking/effort/or anything the provider declares).
        Only meaningful while a provider is active; the provider's client reads
        these to shape its request. Core persists them but never interprets the
        custom ones."""
        return self.provider_settings.get(self.provider, {}).get(key, default)

    def set_setting(self, key, val):
        self.provider_settings.setdefault(self.provider, {})[key] = val

    def pin_model(self, name):
        """Set the active model and mark it as explicitly chosen, so the per-host
        [models] default won't override it. For a lean-tool that selects a model in
        setup() (which runs before the startup host-default adoption)."""
        self.set_active_model(name)
        self.model_explicit = True

    def options(self) -> dict:
        return {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }


def _norm_host(h: str) -> str:
    """Accept a full base URL or a bare host[:port]; return a normalized URL.

    'localhost' is rewritten to 127.0.0.1: on a dual-stack box getaddrinfo returns
    IPv6 ::1 first, but Ollama binds IPv4 127.0.0.1 by default - so urllib connects
    to ::1, gets refused, and reports 'can't reach' even though `curl` (Happy-Eyeballs,
    falls back to IPv4) works fine. Pinning IPv4 matches curl + Ollama's default bind.
    Only the bare hostname is swapped, so an explicit IPv6 URL or a real DNS name is
    left untouched.

    A missing port gets Ollama's default :11434 appended - a bare 'gpu-box' would
    otherwise resolve to port 80 and silently never connect. An explicit/custom port
    (:8080) is left as-is, so non-default Ollama binds keep working. IPv6 literals
    ([::1]) and hosts that already carry a port are untouched."""
    h = h.strip().rstrip("/")
    if not re.match(r"^https?://", h):
        h = "http://" + h
    h = re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", h)
    # Append the default port when the authority has none. Split off scheme, then
    # look at only the host[:port] authority (no path), skipping IPv6 [..] literals.
    m = re.match(r"^(https?://)([^/]+)(.*)$", h)
    if m:
        scheme, authority, rest = m.groups()
        if not authority.startswith("[") and ":" not in authority:
            authority += ":11434"
        h = scheme + authority + rest
    return h


def valid_host(token: str) -> bool:
    """Whether a token can become a usable host URL. Rejects empty/whitespace and
    obvious junk (spaces, no host part). Used to refuse persisting a broken host."""
    t = (token or "").strip()
    if not t or any(c.isspace() for c in t):
        return False
    try:
        u = _norm_host(t)
    except Exception:
        return False
    m = re.match(r"^https?://([^/:]+|\[[^\]]+\])", u)
    return bool(m and m.group(1))


def resolve_host(token: str, machines: dict) -> str:
    """Map a memorable machine name to its URL, else normalize a raw host/URL."""
    return machines[token] if token in machines else _norm_host(token)


def host_label(url: str, machines: dict) -> str:
    """Display form: 'name (url)' when the url has an alias, else the url."""
    for name, u in machines.items():
        if u == url:
            return f"{name} ({url})"
    return url


# The scalar Config fields that round-trip through the TOML config file: load reads
# each (if present) into cfg; save writes each back (default-valued ones are omitted
# for a clean file, but the value still round-trips). This is the ONE source of truth
# for "is field X persisted?" - the load loop and the round-trip test both read it, so
# a new persisted knob is added in exactly one place. (Collections - machines, hosts,
# provider_settings, mcp_*, etc - have bespoke table serialisers and are handled
# separately in load/save; they are intentionally NOT here.)
_PERSISTED_SCALAR_KEYS = (
    "model", "num_ctx", "max_iterations", "keep_alive", "temperature", "top_p", "top_k",
    "repeat_penalty", "ask_user_to_run", "autosave", "auto_update", "update_track",
    "command_timeout", "bg_max_concurrent",
    "worker_max_concurrent", "worker_idle_timeout", "worker_max_iterations", "editor",
    "approval", "confirm_reads", "auto_reconnect", "statusline", "statusline_every",
    "statusline_iter", "auto_evict", "auto_evict_keep",
    "window_messages", "window_tokens", "auto_handover", "handover_soft", "handover_hard",
    "handover_emergency", "handover_min_interval", "autostart_after_handover",
    "auto_compact_interval", "auto_compact_hysteresis", "auto_compact_keep",
    "wake_on_bg_finish", "notes_spool",
    "gen_connect_timeout", "gen_ttft_timeout", "gen_idle_timeout",
    "lean_tools_dir", "providers_dir", "user_name", "ephemeral",
)
# EPHEMERAL: deliberately NEVER persisted. These are per-launch state - saving them
# would let one session silently narrow/alter the NEXT launch's default. `composer` is
# read from the file on load (a user CAN pin composer=false by hand) but is never
# WRITTEN, since --no-composer / worker mode flip it transiently. `leash` is a
# runtime-only ceiling; `incognito` a session trace we won't record; `think` a
# per-session send flag; `cwd`/`model_explicit`/`auto_num_ctx` are derived at launch.
_EPHEMERAL_KEYS = frozenset((
    "composer", "leash", "incognito", "think", "cwd", "model_explicit", "auto_num_ctx",
    # Not config.toml scalars: `_defaults` is the DEFAULTS snapshot save_config writes
    # FROM; `session_overrides` is the per-key override map persisted in the SESSION
    # meta (not config.toml) and re-applied at load runtime-only. Classified here so
    # the config-drift guard accounts for them without treating them as saved scalars.
    "_defaults", "session_overrides",
))
# DERIVED / bespoke: persisted, but via their own table serialisers (not scalar lines),
# or reconstructed at load. Listed so the round-trip test can account for every field.
_BESPOKE_KEYS = frozenset((
    "host", "hosts", "machines", "connect_hosts", "host_models", "provider",
    "provider_settings", "providers_enabled", "lean_tools_enabled", "mcp_servers",
    "mcp_enabled", "always_expand", "show_snapshots", "handover_overrides",
))


def load_config(args) -> Config:
    cfg = Config()
    # config file (lowest, above defaults)
    file_vals = {}
    if CONFIG_PATH.is_file():
        try:
            import tomllib  # lazy: the hermetic --tool-exec role never needs it
            file_vals = tomllib.loads(CONFIG_PATH.read_text())
        except Exception as e:
            print(yellow(f"warning: could not parse {CONFIG_PATH}: {e}"))
    for key in _PERSISTED_SCALAR_KEYS + ("composer",):  # composer: read-only (see _EPHEMERAL_KEYS)
        if key in file_vals:
            setattr(cfg, key, file_vals[key])
    # Snapshot the config.toml DEFAULTS layer: for each persisted scalar, the value as
    # configured (file value if present, else the dataclass default). save_config writes
    # scalar lines FROM this, so a per-session OVERRIDE later applied onto the live cfg
    # never leaks back into config.toml. A /set --config <k> <v> mutates _defaults[k]
    # AND the live attribute together (new default shows through once the override is
    # cleared). CLI-flag overrides below are session-ish but pre-date this layer; they
    # DO update the live value only (not _defaults), matching prior behaviour.
    cfg._defaults = {k: getattr(cfg, k) for k in _PERSISTED_SCALAR_KEYS}
    # The self-managing mechanism is a HANDOVER (the model pulls the same lever
    # /handover does); "compact" now means only the programmatic /compact strip.
    # The knobs are handover_* accordingly - no compact_* back-compat aliases (they'd
    # be one more thing to track); a pre-rename config just falls back to defaults.
    _ov = file_vals.get("handover_overrides")
    if isinstance(_ov, dict):
        cfg.handover_overrides = {k: dict(v) for k, v in _ov.items() if isinstance(v, dict)}
    cfg.leash = _norm_leash(cfg.leash) or "rwe"     # validate the ceiling; junk -> full
    if cfg.approval not in APPROVAL_MODES:          # validate the cadence; junk -> ask
        cfg.approval = "ask"
    if isinstance(file_vals.get("lean_tools_enabled"), list):
        cfg.lean_tools_enabled = [p for p in file_vals["lean_tools_enabled"] if isinstance(p, str)]
    if isinstance(file_vals.get("mcp_servers"), dict):
        cfg.mcp_servers = {k: dict(v) for k, v in file_vals["mcp_servers"].items()
                           if isinstance(v, dict)}
    if isinstance(file_vals.get("mcp_enabled"), list):
        cfg.mcp_enabled = [p for p in file_vals["mcp_enabled"] if isinstance(p, str)]
    if isinstance(file_vals.get("always_expand"), list):
        cfg.always_expand = [p for p in file_vals["always_expand"] if isinstance(p, str)]
    if isinstance(file_vals.get("show_snapshots"), bool):
        cfg.show_snapshots = file_vals["show_snapshots"]
    if isinstance(file_vals.get("providers_enabled"), list):
        cfg.providers_enabled = [p for p in file_vals["providers_enabled"] if isinstance(p, str)]
    else:
        # No providers_enabled key = fresh install or a legacy pre-provider config: seed the
        # bundled ollama default so it works out of the box. Once the key EXISTS we respect
        # it verbatim (even empty) - so disabling ollama STICKS instead of being re-seeded
        # every launch, and a box set up with only a hosted provider is never pushed to ollama.
        cfg.providers_enabled = ["ollama"]
    if isinstance(file_vals.get("provider"), str):
        cfg.provider = file_vals["provider"].strip()
    # Migration: legacy configs used provider = "" to mean the built-in (native)
    # Ollama backend. Ollama is now a real provider, so "" -> "ollama".
    if not cfg.provider:
        cfg.provider = "ollama"
    # An explicit --provider overrides the config's active backend (used by --agent-run
    # workers and anyone pinning a backend at launch). Validated against enabled set
    # after providers_enabled is resolved below.
    if getattr(args, "provider", None):
        cfg.provider = args.provider.strip()
    # [providers.<name>] tables: each is that provider's saved model / window /
    # settings (thinking, effort, and any custom knobs). Kept verbatim so a
    # provider can persist things core doesn't need to understand.
    if isinstance(file_vals.get("providers"), dict):
        # Drop any blank/whitespace provider name (a "" leftover from the old native
        # `provider = ""` era) so it can't round-trip back to an invalid `[providers.]`
        # header that would break the NEXT parse. Self-heals an already-corrupted file.
        cfg.provider_settings = {n: dict(v) for n, v in file_vals["providers"].items()
                                 if isinstance(v, dict) and isinstance(n, str) and n.strip()}
    if os.environ.get("LEANCODER_MODEL"):
        cfg.model = os.environ["LEANCODER_MODEL"]
    if args.model:
        cfg.model = args.model
    cfg.model_explicit = bool(args.model or os.environ.get("LEANCODER_MODEL"))
    # Seed the ollama provider's model from the legacy top-level `model` so a
    # pre-migration config (or a --model/env override) carries onto the ollama
    # provider. An existing [providers.ollama].model is kept unless explicitly
    # overridden on the command line.
    _oll = cfg.provider_settings.setdefault("ollama", {})
    if cfg.model_explicit or not _oll.get("model"):
        _oll["model"] = cfg.model

    # per-host default models: [models] table mapping host URL -> model name
    if isinstance(file_vals.get("models"), dict):
        cfg.host_models = {_norm_host(k): v for k, v in file_vals["models"].items()
                           if isinstance(v, str) and v.strip()}

    # memorable machine names: [machines] table mapping name -> url
    if isinstance(file_vals.get("machines"), dict):
        cfg.machines = {n: _norm_host(u) for n, u in file_vals["machines"].items()
                        if isinstance(u, str) and u.strip()}

    # saved /connect targets: [connect] table mapping name -> ssh destination
    if isinstance(file_vals.get("connect"), dict):
        cfg.connect_hosts = {n: t.strip() for n, t in file_vals["connect"].items()
                             if isinstance(t, str) and t.strip()}

    # Host resolution (precedence: --host > OLLAMA_HOST > config). Names resolve
    # via [machines]. cfg.hosts is ALWAYS the priority pool (config order, never
    # auto-reordered); cfg.host is the active pick within it. A lone `host = "..."`
    # loads as a one-element pool so there is a single source of truth. `hosts = [...]`
    # is the multi-host priority list. An explicit --host/OLLAMA_HOST is a session
    # override of the ACTIVE host: it moves the pick to that host (adding it to the
    # front of the pool if new) but leaves the configured priority order intact.
    if isinstance(file_vals.get("hosts"), list) and file_vals["hosts"]:
        cfg.hosts = [resolve_host(h, cfg.machines) for h in file_vals["hosts"]
                     if isinstance(h, str) and h.strip()]
    elif "host" in file_vals:
        cfg.hosts = [resolve_host(file_vals["host"], cfg.machines)]
    else:
        cfg.hosts = [_norm_host(cfg.host)]
    explicit = args.host or os.environ.get("OLLAMA_HOST")
    if explicit:
        eh = resolve_host(explicit, cfg.machines)
        cfg.host = eh
        if eh not in cfg.hosts:
            cfg.hosts.insert(0, eh)          # session override; priority order otherwise kept
    else:
        cfg.host = cfg.hosts[0]              # provisional; repl probes the pool and picks

    # remaining CLI flags
    if args.num_ctx:        cfg.num_ctx = args.num_ctx
    if getattr(args, "keep_alive", None) is not None: cfg.keep_alive = args.keep_alive
    if getattr(args, "auto_evict", None): cfg.auto_evict = True
    if getattr(args, "window_messages", None) is not None: cfg.window_messages = args.window_messages
    if getattr(args, "window_tokens", None) is not None:
        try:
            cfg.window_tokens = _coerce_setting(args.window_tokens, "int_or_auto")
        except ValueError:
            pass                              # keep the default on a junk --window-tokens
    if args.temperature is not None:    cfg.temperature = args.temperature
    if args.top_p is not None:          cfg.top_p = args.top_p
    if args.top_k is not None:          cfg.top_k = args.top_k
    if args.repeat_penalty is not None: cfg.repeat_penalty = args.repeat_penalty
    if getattr(args, "approval", None):
        cfg.approval = args.approval
    elif args.auto or args.yolo:        # --auto (and the kept --yolo alias) = auto
        cfg.approval = "auto"
    if getattr(args, "no_composer", False):
        cfg.composer = False
    if getattr(args, "no_autosave", False):
        cfg.autosave = False
    if getattr(args, "incognito", False):
        cfg.incognito = True
    if getattr(args, "leash", None):
        cfg.leash = args.leash
    if getattr(args, "chat_only", False):
        cfg.leash = "chat"
    cfg.cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
    if args.think is not None:
        cfg.think = args.think

    # Auto-detect num_ctx at startup only when it wasn't pinned explicitly
    # (no --num-ctx flag and no num_ctx in the config file).
    cfg.auto_num_ctx = not (args.num_ctx or "num_ctx" in file_vals)
    return cfg


def _toml_value(v) -> str:
    """Serialize a scalar/list for a provider-settings table. Lean: strings,
    bools, numbers, and string lists - all a provider knob should ever need."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


class _DefaultsView:
    """Read-only view over a Config that returns the config.toml DEFAULT for any
    persisted scalar (cfg._defaults[key]) instead of the live/effective value, and
    falls through to the live attribute for everything else (bespoke tables, and any
    scalar not yet in _defaults - e.g. a Config built directly in a test). save_config
    reads through this so a per-session OVERRIDE held on the live cfg can never leak
    into config.toml."""
    __slots__ = ("_cfg",)

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, name):
        cfg = object.__getattribute__(self, "_cfg")
        d = cfg._defaults
        if name in _PERSISTED_SCALAR_KEYS and name in d:
            return d[name]
        return getattr(cfg, name)


def save_config(cfg: Config, quiet: bool = False):
    cfg = _DefaultsView(cfg)          # persisted scalars read from _defaults, not live
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # One-time safety net: before the first write that introduces the new
    # provider-based format, snapshot the legacy config so the ollama-as-provider
    # migration can be rolled back by hand. Kept forever (only written once).
    try:
        bak = CONFIG_PATH.with_name(CONFIG_PATH.name + ".pre-migration.bak")
        if CONFIG_PATH.is_file() and not bak.exists():
            old = CONFIG_PATH.read_text()
            if "[providers.ollama]" not in old:
                bak.write_text(old)
    except OSError:
        pass
    name_of = {u: n for n, u in cfg.machines.items()}  # url -> name, for readability
    # cfg.hosts is the priority pool (config order). Persist the ORDER only - the
    # active pick (cfg.host) is a session override and is intentionally not saved as
    # priority. A single-host pool emits the scalar `host = "..."` for clean
    # back-compat; a multi-host pool emits the `hosts = [...]` priority list.
    pool = cfg.hosts or [cfg.host]
    if len(pool) > 1:
        host_line = "hosts = [\n" + "".join(
            f'    "{name_of.get(h, h)}",\n' for h in pool) + "]"
    else:
        host_line = f'host = "{name_of.get(pool[0], pool[0])}"'
    # Top-level `model` stays as a back-compat alias of the ollama provider's model
    # (so an older lean-coder, or anything still reading cfg.model, keeps working).
    # The canonical store is [providers.ollama].model, written below.
    ollama_model = cfg.provider_settings.get("ollama", {}).get("model") or cfg.model
    lines = [
        host_line,
        f'model = "{ollama_model}"',
    ]
    # Only persist num_ctx if it was explicitly pinned (--num-ctx/env/file). When
    # it came from auto-detect, omit it so detection stays active per-host on the
    # next launch - otherwise saving after a session on a 16k host would bake that
    # in and silently cap a 32k host.
    if not cfg.auto_num_ctx:
        lines.append(f"num_ctx = {cfg.num_ctx}")
    if cfg.keep_alive and cfg.keep_alive != "10m":
        lines.append(f'keep_alive = "{cfg.keep_alive}"')
    # ollama streaming timeouts: persist only a non-default (prod defaults stay implicit).
    if cfg.gen_connect_timeout != 10.0:
        lines.append(f"gen_connect_timeout = {cfg.gen_connect_timeout}")
    if cfg.gen_ttft_timeout != 600.0:
        lines.append(f"gen_ttft_timeout = {cfg.gen_ttft_timeout}")
    if cfg.gen_idle_timeout is not None:
        lines.append(f"gen_idle_timeout = {cfg.gen_idle_timeout}")
    if cfg.auto_evict:
        lines.append("auto_evict = true")
    if cfg.auto_evict_keep != 3:
        lines.append(f"auto_evict_keep = {cfg.auto_evict_keep}")
    if cfg.max_iterations != 0:
        lines.append(f"max_iterations = {cfg.max_iterations}")
    if cfg.window_messages != 0:              # default is off (0) now
        lines.append(f"window_messages = {cfg.window_messages}")
    if cfg.window_tokens != "auto":           # default is 'auto'; persist an int cap or 0 (off)
        lines.append(f"window_tokens = {_toml_value(cfg.window_tokens)}")
    if not cfg.auto_handover:
        lines.append("auto_handover = false")
    if cfg.handover_soft != 0.70:
        lines.append(f"handover_soft = {cfg.handover_soft}")
    if cfg.handover_hard != 0.95:
        lines.append(f"handover_hard = {cfg.handover_hard}")
    if cfg.handover_emergency != 1.00:
        lines.append(f"handover_emergency = {cfg.handover_emergency}")
    if cfg.handover_min_interval != 60.0:
        lines.append(f"handover_min_interval = {cfg.handover_min_interval}")
    if not cfg.autostart_after_handover:       # default is on now; persist an opt-OUT
        lines.append("autostart_after_handover = false")
    if cfg.auto_compact_interval != 0:         # default 0 (off); persist when opted in
        lines.append(f"auto_compact_interval = {cfg.auto_compact_interval}")
    if cfg.auto_compact_hysteresis != 0.25:
        lines.append(f"auto_compact_hysteresis = {cfg.auto_compact_hysteresis}")
    if cfg.auto_compact_keep != COMPACT_KEEP:
        lines.append(f"auto_compact_keep = {cfg.auto_compact_keep}")
    if cfg.wake_on_bg_finish:                  # default off; persist an opt-IN
        lines.append("wake_on_bg_finish = true")
    if cfg.notes_spool != 2000:                # default 2000 lines; persist when changed
        lines.append(f"notes_spool = {cfg.notes_spool}")
    if cfg.always_expand:                       # default = [] (diffs stay collapsed)
        lines.append("always_expand = [" + ", ".join(_toml_value(x) for x in cfg.always_expand) + "]")
    if cfg.show_snapshots:                      # default off (snapshots hidden in /load)
        lines.append("show_snapshots = true")
    lines += [
        f"temperature = {cfg.temperature}",
        f"top_p = {cfg.top_p}",
        f"top_k = {cfg.top_k}",
        f"repeat_penalty = {cfg.repeat_penalty}",
        f"ask_user_to_run = {str(cfg.ask_user_to_run).lower()}",
        f"autosave = {str(cfg.autosave).lower()}",
        f"command_timeout = {cfg.command_timeout}",
        f"bg_max_concurrent = {cfg.bg_max_concurrent}",
        f"worker_max_concurrent = {cfg.worker_max_concurrent}",
        f"worker_idle_timeout = {cfg.worker_idle_timeout}",
        f"worker_max_iterations = {cfg.worker_max_iterations}",
    ]
    # leash is deliberately EPHEMERAL - it is NOT persisted. /leash is a runtime-only
    # ceiling (see the load path, which also treats it as runtime-only): a fresh launch
    # must default to the full surface (rwe), never silently boot chat-only/read-only
    # because some earlier session narrowed the leash and then any unrelated setting
    # change snapshotted it to disk. Pass --leash / --chat-only to start narrowed.
    # incognito is ephemeral too (persisting it would be a local trace).
    if cfg.approval != "ask":              # default is ask; persist a non-default cadence
        lines.append(f'approval = "{cfg.approval}"')
    if cfg.confirm_reads:
        lines.append("confirm_reads = true")
    if cfg.auto_reconnect:
        lines.append("auto_reconnect = true")
    if cfg.ephemeral:                         # default off; persist an opt-IN (always-wipe)
        lines.append("ephemeral = true")
    if not cfg.statusline:                    # default on; persist only when turned off
        lines.append("statusline = false")
    if cfg.statusline_every != 1:             # default 1 (every prompt); persist if changed
        lines.append(f"statusline_every = {cfg.statusline_every}")
    if cfg.statusline_iter != 0:              # default 0 (off); persist if changed
        lines.append(f"statusline_iter = {cfg.statusline_iter}")
    if cfg.auto_update:                       # default off; persist an opt-IN
        lines.append("auto_update = true")
    if cfg.update_track != "stable":          # default stable; persist a non-default track
        lines.append(f'update_track = "{cfg.update_track}"')
    if cfg.editor:
        lines.append(f'editor = "{cfg.editor}"')
    if cfg.user_name and cfg.user_name != "operator":
        lines.append(f'user_name = "{cfg.user_name}"')
    if cfg.lean_tools_dir:
        lines.append(f'lean_tools_dir = "{cfg.lean_tools_dir}"')
    if cfg.lean_tools_enabled:
        lines.append("lean_tools_enabled = ["
                     + ", ".join(f'"{p}"' for p in cfg.lean_tools_enabled) + "]")
    if cfg.providers_dir:
        lines.append(f'providers_dir = "{cfg.providers_dir}"')
    if cfg.mcp_enabled:
        lines.append("mcp_enabled = ["
                     + ", ".join(f'"{p}"' for p in cfg.mcp_enabled) + "]")
    # Always persist the enabled set (even empty): an absent key is treated as "fresh -
    # seed ollama" on load, so an explicit empty/ollama-disabled set MUST be written to
    # stick (otherwise disabling your last provider would silently re-seed ollama).
    lines.append("providers_enabled = ["
                 + ", ".join(f'"{p}"' for p in cfg.providers_enabled) + "]")
    if cfg.machines:
        lines.append("")
        lines.append("[machines]")
        lines += [f'"{n}" = "{u}"' for n, u in cfg.machines.items()]
    if cfg.connect_hosts:
        lines.append("")
        lines.append("[connect]")
        lines += [f'"{n}" = "{t}"' for n, t in cfg.connect_hosts.items()]
    if cfg.host_models:
        lines.append("")
        lines.append("[models]")
        lines += [f'"{h}" = "{m}"' for h, m in cfg.host_models.items()]
    # Per-model handover overrides: [handover_overrides.<model>] tables. Declared a
    # BESPOKE (own-serializer) key and READ back in load_config - but historically had
    # NO writer here, so every autosave_config() (fired by /model, /connect, and the
    # flows around a handover) rewrote config.toml WITHOUT them: the overrides silently
    # vanished and the next load fell back to the global handover_soft/hard defaults
    # (0.70/0.95), which the status line shows as "handover 95%". Write them back.
    if cfg.handover_overrides:
        for _m, _ov in cfg.handover_overrides.items():
            if not (isinstance(_m, str) and _m.strip() and isinstance(_ov, dict) and _ov):
                continue
            lines.append("")
            # Quote the model segment: names carry ':' / '.' (qwen3:32b, claude-opus-4-8)
            # which are INVALID as a bare TOML key - an unquoted header would make the
            # whole config.toml unparseable, and the next load would silently fall back to
            # ALL defaults (wiping global handover_soft/hard too -> the "95%" reset).
            lines.append(f"[handover_overrides.{_toml_value(_m)}]")
            lines += [f"{k} = {_toml_value(v)}" for k, v in _ov.items()]
    if cfg.provider and cfg.provider != "ollama":
        lines.insert(2, f'provider = "{cfg.provider}"')   # just after model line
    # ollama is the default provider; omit the line for it (a bare config still
    # boots on ollama). A non-default active provider is written so it comes back.
    # [providers.<name>] tables: the per-backend model + settings. num_ctx is
    # deliberately NOT persisted - it is derived from the provider's
    # context_window() hook at activation, so saving it would pin a stale value
    # and override an intentional cap (e.g. a plugin capping below the raw API
    # limit) on the next load. Top-level model/num_ctx stay the native Ollama ones.
    for name, settings in cfg.provider_settings.items():
        # Guard against a blank/whitespace provider name (a "" leftover from the old
        # native-ollama `provider = ""` era): it would emit an invalid `[providers.]`
        # header, and the NEXT load would fail to parse and silently fall back to
        # defaults - wiping providers_enabled + lean_tools_enabled. Skip it.
        if not (isinstance(name, str) and name.strip()):
            continue
        saved = {k: v for k, v in settings.items() if k != "num_ctx"}
        if not saved:
            continue
        lines.append("")
        lines.append(f"[providers.{name}]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in saved.items()]
    CONFIG_PATH.write_text("\n".join(lines) + "\n")
    if not quiet:
        print(green(f"saved config -> {CONFIG_PATH}"))


def autosave_config(cfg: Config):
    """Persist config silently after a config-mutating command (full autosave -
    settings stick across launches without a manual save). Never breaks a turn."""
    try:
        save_config(cfg, quiet=True)
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Session persistence (autosave on by default)
#
# A conversation is just agent.messages - a list of dicts on the driver. Both
# the local and the hosted-API backends are stateless: the full history is
# re-sent every turn and the server stores nothing. So save = dump the list to a
# file, resume = load it back, and it works identically across backends because
# the history is kept in lean-coder's own canonical shape (each client translates at
# send time).
#
# By default the live conversation autosaves each turn to a rolling "auto-<time>"
# session (crash-safe) and the most-recent session auto-loads on launch, so a
# restart picks up where you left off. `/save [name]` freezes a named snapshot;
# `/load` resumes any session. Turn autosave off (cfg.autosave / `/autosave off`)
# for amnesic behavior - then nothing is written and nothing auto-loads.
# ----------------------------------------------------------------------------

def _session_name_ok(name: str) -> str:
    """Sanitize a session name to a safe single filename component (no path
    traversal, no separators). Returns the cleaned name, or '' if nothing usable
    remains."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return cleaned


def _session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def _session_title(messages) -> str:
    """A short label for a session: the first user message, one line, truncated."""
    for m in messages:
        if m.get("role") == "user":
            line = " ".join(str(m.get("content", "")).split())
            return (line[:60] + "…") if len(line) > 60 else line
    return "(no user messages)"


def _atomic_write_text(path: Path, text: str):
    """Write text to `path` atomically: write a temp file in the same dir, fsync it,
    then os.replace() over the target (a single rename - never a half-written file).
    Matters for the per-turn autosave: a reboot/crash MID-WRITE otherwise truncates
    the session JSON, and the next launch's auto-resume fails to parse it. os.replace
    is atomic on the same filesystem, so a reader always sees the old OR new file
    whole, never a torn one. Falls back to a plain write if the temp/replace dance
    itself errors (better a non-atomic write than no write)."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        path.write_text(text)


def save_session(messages, cfg, name: str, remote=None, pinned_plan="", notes=None):
    """Write the conversation to ~/.config/leancoder/sessions/<name>.json with a
    metadata header. Returns (path, meta). Raises ValueError on a bad name.

    The meta captures the session's full working state - backend, model, host,
    leash/approval ceiling, thinking/effort, the pinned plan (GOAL+TODO), and the
    remote host it ran on (if any) - so /load can restore it. `remote` is the active
    remote host, or None for local. A field absent from an older session's meta falls
    back to the live config on load (and is then re-captured here on the next save)."""
    safe = _session_name_ok(name)
    if not safe:
        raise ValueError("invalid session name")
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    body = [m for m in messages if m.get("role") != "system"]
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M"),
        "provider": cfg.provider,            # "" = native Ollama; else the backend name
        "model": cfg.active_model(),          # the model actually in use (provider or native)
        "host": cfg.host,
        "cwd": str(cfg.cwd),
        "leash": cfg.leash,                  # capability ceiling (per-session perms)
        "approval": cfg.approval,            # confirm cadence
        "thinking": cfg.setting("thinking"),  # provider-scoped; None on ollama / unset
        "effort": cfg.setting("effort"),
        "remote": remote,                    # ssh host tools ran on, or None (local)
        "title": _session_title(messages),
        "turns": len(body),
        "pinned_plan": pinned_plan or "",
        "notes": list(notes or []),          # session-scoped episodic memory (note tool)
        "session_overrides": dict(getattr(cfg, "session_overrides", {}) or {}),
        "tok_factor": _tok_factor(),         # learned estimate->actual factor (calibrated meter)
    }
    path = _session_path(safe)
    _atomic_write_text(path, json.dumps({"meta": meta, "messages": messages}, indent=2))
    return path, meta


def load_session(name: str):
    """Return the stored {'meta', 'messages'} dict for a saved session.
    Raises FileNotFoundError if it doesn't exist."""
    safe = _session_name_ok(name)
    path = _session_path(safe)
    if not path.is_file():
        raise FileNotFoundError(name)
    return json.loads(path.read_text())


def list_sessions():
    """All saved sessions as (name, meta) pairs, newest first."""
    if not SESSIONS_DIR.is_dir():
        return []
    out = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            meta = json.loads(p.read_text()).get("meta", {})
        except (json.JSONDecodeError, OSError):
            meta = {}
        out.append((p.stem, meta))
    out.sort(key=lambda nm: nm[1].get("saved_at", ""), reverse=True)
    return out


def delete_session(name: str) -> bool:
    """Remove a saved session. Returns True if a file was deleted."""
    path = _session_path(_session_name_ok(name))
    if path.is_file():
        path.unlink()
        return True
    return False


def _session_rows():
    """(name, meta, mtime) for all sessions, most-recently-used (mtime) first.
    Used by the /load picker and the settings sessions view."""
    rows = []
    if not SESSIONS_DIR.is_dir():
        return rows
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            meta = json.loads(p.read_text()).get("meta", {})
        except (json.JSONDecodeError, OSError):
            meta = {}
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0
        rows.append((p.stem, meta, mtime))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def _auto_resume_pick(rows):
    """The session that auto-load resumes on launch: the most-recently-used one
    that isn't a pre-handover safety snapshot. `rows` is _session_rows() output
    (already most-recent first). Returns (name, meta, mtime) or None. Deliberately
    NOT cwd-scoped - scoping silently skipped the genuinely-most-recent session
    when it was last used from another dir. Pure -> unit-tested."""
    return next((r for r in rows if not _is_snapshot(r[0])), None)


_autosave_seq = 0


def _new_autosave_name() -> str:
    """A unique rolling-autosave session id for this launch (or post-/clear). The
    sequence suffix keeps it distinct even when two are minted in the same second
    (e.g. /clear then immediately /handover) so neither clobbers the other."""
    global _autosave_seq
    _autosave_seq += 1
    # Short + readable: auto-<MMDD-HHMMSS>-<seq>. No PID (the seq keeps two names minted
    # in the same second distinct within a launch; the session-lock system handles real
    # cross-instance collisions). MMDD keeps it day-safe without a full timestamp.
    return f"{AUTOSAVE_PREFIX}{time.strftime('%m%d-%H%M%S')}-{_autosave_seq}"


# --- session locks: one live writer per session ----------------------------
# A sidecar "<name>.lock" marks who is actively writing a session: our pid + host
# + a heartbeat bumped on every autosave. The contract:
#   - Loading a session CLAIMS its lock immediately (not on the first autosave) -
#     so the loader owns it from the start and is never forked off the session it
#     just opened (the bug that lobotomised a session mid-conversation).
#   - Auto-load on startup SKIPS a session that's live in another instance and
#     starts fresh instead (never steals a session another instance is using).
#   - Explicit /load (or --resume) of a live session PROMPTS to take it over; on
#     yes it claims the lock, and only THEN does the previous owner notice the
#     steal on its next autosave and fork to a fresh auto- session (with a clear
#     note + how to take it back) - history intact, just renamed.
# So a fork only ever happens after an explicit, consenting take-over.
_LOCK_STALE_SECS = 120  # a heartbeat older than this means the writer is gone

def _lock_path(name: str) -> Path:
    return SESSIONS_DIR / f"{_session_name_ok(name)}.lock"

def _read_lock(name: str):
    try:
        return json.loads(_lock_path(name).read_text())
    except (OSError, ValueError):
        return None

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

def _lock_is_live(name: str) -> bool:
    """True if some OTHER running instance currently holds this session.

    On the SAME host the pid is authoritative: a live holder owns the session no
    matter how long it has sat idle. This is deliberate - an active instance that
    simply hasn't autosaved in a while (>_LOCK_STALE_SECS) must NOT look dead, or a
    second launch would silently steal the session out from under it (the "opened a
    new session and it stole my other one" bug). A dead pid frees the lock at once.
    Cross-host we can't probe the pid, so we fall back to heartbeat freshness."""
    info = _read_lock(name)
    if not info or info.get("pid") == os.getpid():
        return False
    if info.get("host") == _HOSTNAME:
        return _pid_alive(info.get("pid", -1))
    return (time.time() - info.get("ts", 0)) <= _LOCK_STALE_SECS

def _write_lock(name: str):
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        _lock_path(name).write_text(json.dumps(
            {"pid": os.getpid(), "host": _HOSTNAME, "ts": time.time()}))
    except OSError:
        pass

def _own_lock(name: str) -> bool:
    info = _read_lock(name)
    return bool(info and info.get("pid") == os.getpid()
                and info.get("host") == _HOSTNAME)

def _release_lock(name: str):
    if _own_lock(name):
        try:
            _lock_path(name).unlink()
        except OSError:
            pass


def _prune_stale_locks():
    """Sweep session .lock files whose holder is provably gone: a stale heartbeat (older
    than _LOCK_STALE_SECS) or a dead pid on this host. A clean exit unlinks its own lock,
    but a KILLED instance (or a relaunch) leaves one behind - it ages out logically yet
    the file lingers in the sessions dir forever. Never removes a LIVE lock (another
    running instance) or our own. Called at startup so dead locks don't accumulate."""
    if not SESSIONS_DIR.is_dir():
        return
    me, host = os.getpid(), _HOSTNAME
    for lk in SESSIONS_DIR.glob("*.lock"):
        info = _read_lock(lk.stem)
        if info and info.get("pid") == me and info.get("host") == host:
            continue                       # our own lock - keep
        if not _lock_is_live(lk.stem):     # nobody live holds it -> safe to remove
            try:
                lk.unlink()
            except OSError:
                pass


def autosave_session(agent, cfg):
    """Write the live conversation to its rolling autosave session. No-op when
    autosave is off or the conversation is empty. Must never raise into the turn
    loop - a failed autosave should not kill a working session."""
    if cfg.incognito or not cfg.autosave:
        return
    if not any(m.get("role") != "system" for m in agent.messages):
        return
    # If another instance has taken over our session (claimed the lock), don't
    # fight over it: announce it once and fork to a fresh auto- session.
    name = agent.autosave_name
    info = _read_lock(name)
    if info and not _own_lock(name) and _lock_is_live(name):
        old = name
        agent.autosave_name = _new_autosave_name()
        print(yellow(f"\n  {GLYPH.get('warn', '!')} session '{old}' resumed elsewhere"
                     f" - /load {old} to take it back."))
        print(dim(f"  continuing here in a new session '{agent.autosave_name}'."))
        name = agent.autosave_name
    try:
        rhost = agent.remote.host if getattr(agent, "remote", None) else None
        save_session(agent.messages, cfg, name, remote=rhost,
                     pinned_plan=getattr(agent, "pinned_plan", ""),
                     notes=getattr(agent, "notes", None))
        _write_lock(name)
        _prune_autosaves()
    except (ValueError, OSError):
        pass


def _prune_autosaves(keep: int = AUTOSAVE_KEEP):
    """Keep only the newest `keep` autosave sessions, and separately cap the rolling
    pre-handover safety snapshots (a handover writes one every fire) to
    PREHANDOVER_KEEP. Both are rolling/disposable; only USER-named sessions are kept
    forever. Without the snapshot cap they pile up unbounded and flood /load (a busy
    session can leave dozens)."""
    if not SESSIONS_DIR.is_dir():
        return

    def _cap(match, n):
        files = [p for p in SESSIONS_DIR.glob("*.json") if match(p.stem)]
        try:
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return
        for p in files[n:]:
            try:
                p.unlink()
            except OSError:
                pass

    # A prehandover snapshot of an auto- session starts with AUTOSAVE_PREFIX too, so
    # match snapshots FIRST and exclude them from the autosave cap (else a snapshot
    # would count against the autosave quota and skew both).
    _cap(_is_snapshot, PREHANDOVER_KEEP)
    _cap(lambda s: s.startswith(AUTOSAVE_PREFIX) and not _is_snapshot(s), keep)


# ----------------------------------------------------------------------------
# Path safety
# ----------------------------------------------------------------------------

def resolve_in_project(cfg: Config, path: str) -> Path:
    # expanduser FIRST so a leading ~ (or ~user) is a real home dir, not a literal
    # 'cwd/~/...' child - a model that writes "~/notes.txt" means its home, not a
    # folder called '~'. After expansion an absolute path (incl. an expanded ~) is
    # taken as-is; anything still relative is anchored to the project cwd.
    p = Path(path).expanduser()
    p = p.resolve() if p.is_absolute() else (cfg.cwd / p).resolve()
    return p


# ----------------------------------------------------------------------------
# Background tasks (a trailing '&' on run_command detaches; tracked + reaped)
#
# A backgrounded job is detached (own session) so it survives the turn, BUT it is
# pegged to the lean-coder session: a registry under bg/tasks.jsonl records the pid +
# the owning lean-coder pid, so a clean exit kills this session's jobs and a later
# launch reaps orphans left by a crashed session. /bg lists + kills them.
# ----------------------------------------------------------------------------

def _bg_registry_path():
    return CONFIG_DIR / "bg" / "tasks.jsonl"


def _bg_register(pid, cmd, log, kind="task", idle_timeout=None,
                 notify_on_exit=False, heartbeat_timeout=None,
                 max_runtime=None, kill_on_max=True):
    """Record a backgrounded task (owner = this process's pid, host = this box).
    A bg task runs on whichever box the executor is on, so the registry lives on
    THAT box; the host field lets a reader tell 'my box' (pid is authoritative)
    from a foreign record (never mine to reap) - see _proc_alive_here.

    `kind` tags what the record is: "task" (a plain backgrounded run_command, the
    default) or "worker" (a dispatched agent). Both share this lifecycle, but the
    plain bg surface (/bg, the bg_status tool, non-agent finished-notify) FILTERS
    to kind="task" so a session without the worker tool never sees - or is
    confused by - a worker. Workers surface only through the worker tool's path.

    `idle_timeout` (seconds, or None) records the lease: the task self-terminates
    if left UN-attended (its '<log>.lease' file not touched) for this long. The
    parent attends by bumping the lease (once per turn while it lives - see
    _bg_bump_lease); when the parent is gone the lease goes stale and the task's
    own watchdog kills it. Inverse of session-lock staleness; caps runaway quota
    burn by a worker whose parent dropped. None = no lease (runs until done)."""
    try:
        p = _bg_registry_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"pid": pid, "cmd": cmd, "log": str(log), "owner": os.getpid(),
               "host": _HOSTNAME, "kind": kind,
               "idle_timeout": idle_timeout,
               "notify_on_exit": notify_on_exit,
               "heartbeat_timeout": heartbeat_timeout,
               "max_runtime": max_runtime, "kill_on_max": kill_on_max,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "started_at": time.time()}
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        global _BG_LOAD_CACHE
        _BG_LOAD_CACHE = (None, [])   # invalidate: next _bg_load re-reads
    except OSError:
        pass


def _bg_lease_path(log):
    """The lease file for a bg task with the given log path. Its mtime is 'last
    attended'; the task's watchdog compares it against idle_timeout."""
    return Path(str(log) + ".lease")


def _bg_bump_lease(rec):
    """Attend a leased task: bump its lease file mtime to now, so its watchdog sees
    the parent is alive and doesn't self-terminate. No-op for a task with no lease
    (idle_timeout None) or a foreign-host record (its lease lives on that box, not
    ours to touch - the executor there bumps it). Best-effort."""
    if not rec.get("idle_timeout") or not _bg_here(rec):
        return
    log = rec.get("log")
    if not log:
        return
    try:
        _bg_lease_path(log).touch()
    except OSError:
        pass


_BG_LOAD_CACHE = (None, [])   # (mtime, records); registry read 2-5x/turn otherwise


def _bg_load():
    # Cache on the registry file's mtime: it changes only when a task starts/ends/is
    # killed, so the cache is hot for the vast majority of the per-turn reads. A stale
    # mtime just re-reads (always valid), so a cross-process write is never missed.
    global _BG_LOAD_CACHE
    p = _bg_registry_path()
    try:
        mtime = p.stat().st_mtime
        if mtime == _BG_LOAD_CACHE[0]:
            return list(_BG_LOAD_CACHE[1])
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        _BG_LOAD_CACHE = (mtime, recs)
        return list(recs)
    except (OSError, json.JSONDecodeError):
        return []


def _bg_save(recs):
    global _BG_LOAD_CACHE
    try:
        p = _bg_registry_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        _BG_LOAD_CACHE = (None, [])   # invalidate: next _bg_load re-reads
    except OSError:
        pass


def _proc_alive(pid):
    # Zombie-aware: a killed/finished child we never waitpid()'d still has a pid that
    # os.kill(pid, 0) reports as alive, so check /proc state and treat 'Z' as dead.
    # Falls back to signal-0 where /proc isn't available (non-Linux).
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            state = f.read().rpartition(")")[2].split()[0]
        return state != "Z"
    except (OSError, ValueError, TypeError, IndexError):
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True            # exists but not ours
    except (OSError, ValueError, TypeError):
        return False


# --- the ONLY two OS primitives with no cross-platform stdlib call ------------
# Detaching a child into its own session/group and killing that whole group have
# no unified API: POSIX uses setsid + os.killpg, Windows uses process-group
# creation flags + taskkill /T. Every other bg/watchdog behaviour (sidecars,
# lease/heartbeat/max_runtime logic) is platform-agnostic and sits on top of
# these two. Kept here as the single source of truth so core (_bg_kill) and the
# pure-Python bg-runner share one implementation of each (no drift).
def _detached_popen_kwargs():
    """Popen kwargs that put the child in its OWN session/process group, so a
    group kill takes the whole tree. POSIX: start_new_session (setsid), making
    pgid == the child pid. Windows: CREATE_NEW_PROCESS_GROUP."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_tree(pid, sig=None):
    """Kill a detached child AND its whole group/tree. POSIX: os.killpg on the
    pgid (== pid, since it was start_new_session'd). Windows: os.killpg is absent
    (AttributeError) and groups differ, so use taskkill /T to walk the tree. On
    any failure, fall back to a direct os.kill. Best-effort; never raises.

    `sig` defaults to None (resolved to SIGTERM) rather than signal.SIGKILL: on
    Windows signal.SIGKILL DOESN'T EXIST, and a signal.SIGKILL default arg is
    evaluated at IMPORT time - so importing this module (e.g. the pushed executor
    agent.py) crashed with AttributeError before the handshake. SIGTERM exists on
    both; os.kill on Windows treats any non-CTRL_* sig as a hard TerminateProcess."""
    if sig is None:
        sig = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        os.killpg(int(pid), sig)
        return
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (OSError, ValueError, TypeError):
            pass
    try:
        os.kill(int(pid), sig)
    except (OSError, ValueError, TypeError):
        pass


def _bg_kill(pid, sig=signal.SIGTERM):
    """Kill a backgrounded task's whole group (start_new_session -> pgid == pid).
    Delegates to the shared _kill_tree primitive so Linux and Windows share one
    implementation."""
    _kill_tree(pid, sig)


def _bg_here(rec) -> bool:
    """True if this bg record was registered on THIS box (so its pid is ours to
    probe / reap). A record with no host predates the host field - treat as local
    (back-compat: old registries only ever held this-box tasks)."""
    h = rec.get("host")
    return h is None or h == _HOSTNAME


def _bg_alive(rec) -> bool:
    """Whether a task looks alive. Same-host: the pid is authoritative. Foreign
    host (a task on another box we can't probe): fall back to 'has it written its
    exit sidecar yet' - no sidecar => assume still running. Mirrors _lock_is_live:
    trust the pid only where we can see it, else infer."""
    if _bg_here(rec):
        return _proc_alive(rec.get("pid"))
    return _bg_exit_code(rec) is None


# Wake-hook registry: extra no-arg callables that return a finish-notice string (or "")
# for the event-driven wake path. A lean-tool whose jobs are HIDDEN from the core bg
# notify (e.g. dispatch_worker's workers, kind-filtered out) registers its own
# _finished_notice here so its finishes also wake the agent. Each hook owns its dedup,
# so a job surfaced by wake won't re-report on the next real turn. Empty by default.
_WAKE_HOOKS = []


def register_wake_hook(fn):
    """Register a no-arg callable fn() -> str for Agent.bg_wake_turn to aggregate.
    Idempotent (won't add the same fn twice). Returns fn (usable as a decorator)."""
    if fn not in _WAKE_HOOKS:
        _WAKE_HOOKS.append(fn)
    return fn


def _bg_is_worker(rec) -> bool:
    """A dispatched-agent record (kind='worker'). Everything else - incl. records
    that predate the field - is a plain task. Plain-bg surfaces filter workers OUT
    (see _bg_register); the worker tool filters them IN."""
    return rec.get("kind") == "worker"


def _fmt_runtime(secs) -> str:
    """A short elapsed-time string (e.g. '5s', '3m12s', '1h04m') for how long a bg
    task has run. '?' when we can't tell (a record older than the started_at field)."""
    try:
        secs = int(secs)
    except (TypeError, ValueError):
        return "?"
    if secs < 0:
        return "?"
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _bg_runtime(rec):
    """Seconds a task has been running (now - started_at) if still alive, or its
    total lifetime is unknown once finished (we only stamp start). None if the
    record predates started_at."""
    t0 = rec.get("started_at")
    if not isinstance(t0, (int, float)):
        return None
    return max(0, int(time.time() - t0))


def _bg_running(owner=None, kind="task"):
    """Registered tasks still alive (optionally only this session's). Filters to
    `kind` (default 'task' = the plain-bg surface, workers excluded); pass
    kind=None to include every kind."""
    out = []
    for r in _bg_load():
        if owner is not None and r.get("owner") != owner:
            continue
        if kind is not None and r.get("kind", "task") != kind:
            continue
        if _bg_alive(r):
            out.append(r)
    return out


def _bg_finished_here(owner):
    """This-box bg tasks launched by `owner` (a pid) that have FINISHED and left an
    exit sidecar - as [{pid, cmd, code, tail}]. Shared by the local finished-notify
    scan and the remote BG_POLL verb (which runs in the executor, owner = its own
    pid), so both report identically. Dedup by pid is the caller's job."""
    out = []
    for r in _bg_load():
        pid = r.get("pid")
        if r.get("owner") != owner or not _bg_here(r) or _proc_alive(pid):
            continue
        if r.get("kind", "task") != "task":     # workers notify via their own path
            continue
        code = _bg_exit_code(r)
        if code is None:
            continue
        out.append({"pid": pid, "cmd": r.get("cmd", "?"), "code": code,
                    "tail": _bg_log_tail(r.get("log"), 20),
                    "notify_on_exit": bool(r.get("notify_on_exit"))})
    return out


def _bg_finished_msg(items):
    """Format finished-task dicts (from _bg_finished_here / a BG_POLL reply) into
    the notice injected on the next turn. "" for an empty list."""
    lines = []
    for it in items:
        head = f"background task finished: `{it.get('cmd', '?')}` exited {it.get('code')}"
        tail = it.get("tail") or ""
        lines.append(head + (f"\nlast output:\n{tail}" if tail else ""))
    return "\n\n".join(lines)


def _bg_exit_code(rec):
    """The exit code a finished bg task wrote to its '<log>.exit' sidecar, as a
    string ('0', '137', ...), or None if it hasn't finished / wrote nothing. The
    sidecar is a plain file, so this works whether the task ran locally or on a
    remote executor - no live process handle needed."""
    log = rec.get("log")
    if not log:
        return None
    try:
        code = Path(str(log) + ".exit").read_text().strip()
        return code or None
    except (OSError, ValueError):
        return None


def _bg_log_tail(log, n=20) -> str:
    """Last n lines of a bg task's log (for the finished-job notice), truncated
    per line so a runaway line can't flood context. "" if unreadable/empty."""
    if not log:
        return ""
    try:
        lines = Path(log).read_text(errors="replace").splitlines()
    except (OSError, ValueError):
        return ""
    tail = [ln[:500] for ln in lines[-n:]]
    return "\n".join(tail).strip()


def _bg_alerts_here(owner):
    """This-box RUNNING bg tasks launched by `owner` that have tripped a watchdog
    ALERT sidecar ('<log>.silent' from a heartbeat, '<log>.maxrun' from a notify-only
    max_runtime) - as [{pid, cmd, kind ('silent'|'maxrun'), secs, tail}]. Unlike a
    finish, these fire while the job is STILL ALIVE (heartbeat never kills; a
    kill_on_max=false ceiling notifies but keeps running), so they're a separate scan
    from _bg_finished_here. Dedup (report each once) is the caller's job."""
    out = []
    for r in _bg_load():
        pid = r.get("pid")
        if r.get("owner") != owner or not _bg_here(r):
            continue
        if r.get("kind", "task") != "task":
            continue
        log = r.get("log")
        if not log:
            continue
        alive = _proc_alive(pid)
        for suffix, akind in ((".silent", "silent"), (".maxrun", "maxrun")):
            sc = Path(str(log) + suffix)
            try:
                secs = sc.read_text().strip()
            except (OSError, ValueError):
                continue
            if not secs:
                continue
            # An alert means "still running". If the job is already dead, its finish
            # notice covers it - drop the stale sidecar so it can't double-report.
            if not alive:
                try:
                    sc.unlink()
                except OSError:
                    pass
                continue
            out.append({"pid": pid, "cmd": r.get("cmd", "?"), "kind": akind,
                        "secs": secs, "tail": _bg_log_tail(log, 20), "log": log})
    return out


def _bg_alert_msg(items):
    """Format watchdog-alert dicts (from _bg_alerts_here) into the notice injected on
    the next turn. "" for an empty list. These are ALERTS about a still-running job,
    not a finish - the wording makes that explicit so the agent doesn't assume it's
    done."""
    lines = []
    for it in items:
        pid, cmd = it.get("pid"), it.get("cmd", "?")
        if it.get("kind") == "silent":
            head = (f"background task STILL RUNNING but silent: `{cmd}` (pid {pid}) has "
                    f"produced no output for ~{it.get('secs')}s. It was NOT killed. "
                    f"Check on it (bg_status), keep waiting, or `kill {pid}`.")
        else:
            head = (f"background task hit max_runtime: `{cmd}` (pid {pid}) ran past its "
                    f"{it.get('secs')}s ceiling and is STILL RUNNING (kill_on_max=false). "
                    f"Decide: let it continue or `kill {pid}`.")
        tail = it.get("tail") or ""
        lines.append(head + (f"\nlast output:\n{tail}" if tail else ""))
    return "\n\n".join(lines)


def _bg_clear_alert(log, akind):
    """Remove a watchdog-alert sidecar once reported, so it fires once per trip. The
    in-band watchdog re-writes it if the condition recurs (heartbeat re-arms when the
    log resumes then goes quiet again). Best-effort."""
    suffix = ".silent" if akind == "silent" else ".maxrun"
    try:
        Path(str(log) + suffix).unlink()
    except OSError:
        pass


def _bg_status_items(owner, kind="task"):
    """Status rows for this-box bg tasks owned by `owner` (a pid), for the
    model-facing bg_status tool. Each row: {pid, cmd, state ('running' | 'exit N'),
    runtime (human elapsed), tail (last lines)}. Filtered to `kind` (default 'task',
    so workers stay off the plain-bg surface). Only records on THIS box - a foreign
    record's pid isn't ours to probe."""
    rows = []
    for r in _bg_load():
        if r.get("owner") != owner or not _bg_here(r):
            continue
        if kind is not None and r.get("kind", "task") != kind:
            continue
        code = _bg_exit_code(r)
        alive = _proc_alive(r.get("pid"))
        if alive and code is None:
            state = "running"
        elif code is not None:
            state = f"exit {code}"
        else:
            state = "gone"        # not alive, never wrote a sidecar (killed/crashed)
        rt = _bg_runtime(r)
        rows.append({"pid": r.get("pid"), "cmd": r.get("cmd", "?"), "state": state,
                     "runtime": _fmt_runtime(rt) if rt is not None else "?",
                     "started": r.get("started", "?"),
                     "tail": _bg_log_tail(r.get("log"), 20)})
    return rows


def _bg_status_msg(rows, pid=None) -> str:
    """Render _bg_status_items rows into the string the bg_status tool returns to
    the model. pid given but no row -> a clear 'no such task'. Empty -> a plain note."""
    if pid is not None and not rows:
        return (f"no background task with pid {pid} owned by this session on this box "
                f"(it may have been reaped, or belongs to another session).")
    if not rows:
        return "no background tasks from this session.  (end a run_command with ' &' to start one)"
    out = []
    for r in rows:
        head = f"pid {r['pid']}  {r['state']}  (ran {r['runtime']})  {r['cmd']}"
        tail = r.get("tail") or ""
        out.append(head + (f"\n  last output:\n{tail}" if tail else ""))
    return "\n\n".join(out)


def _bg_poll_interval(lease, maxrt, hb):
    """Watchdog poll cadence: a few checks per shortest active window, clamped
    5..30s. Identical to _bg_wrap's `chk` so shell and Python paths trip alike."""
    wins = [w for w in (lease or None, maxrt or None, hb or None) if w]
    return max(5, min(30, min(wins) // 4)) if wins else 30


def run_bg_child(cfg):
    """Pure-Python backgrounded-command runner (the cross-platform twin of the
    POSIX shell wrapper built by builtins Tools._bg_wrap). Spawned detached as
    `python lean_coder.py --bg-run <json>` so it survives the executor/ssh drop,
    exactly like the old subshell. Reproduces the sidecar contract byte-for-byte:
      <log>.exit   : child exit code on completion; 137 on lease-expiry / max-kill
      <log>.lease  : touched at launch; parent bumps mtime; stale>idle -> 137 + kill
      <log>.maxrun : elapsed secs, written ONCE on max trip when kill_on_max=False
      <log>.silent : silence secs, written ONCE when log mtime stale>hb (bark, re-arms)
    All watchdog channels are file-based so they read identically local/remote and
    survive a reconnect. Used on Windows (no POSIX shell); POSIX keeps the shell path.
    `cfg` keys: cmd, exitf, leasef, logf, idle_timeout, max_runtime, heartbeat_timeout,
    kill_on_max."""
    cmd = cfg["cmd"]
    exitf, leasef, logf = cfg["exitf"], cfg["leasef"], cfg.get("logf") or cfg["exitf"]
    lease = int(cfg.get("idle_timeout") or 0)
    maxrt = int(cfg.get("max_runtime") or 0)
    hb = int(cfg.get("heartbeat_timeout") or 0)
    kill_on_max = bool(cfg.get("kill_on_max", True))
    silentf, maxrunf = logf + ".silent", logf + ".maxrun"

    def _write(path, text):
        try:
            with open(path, "w") as f:
                f.write(str(text))
        except OSError:
            pass

    def _mtime(path, default):
        try:
            return os.path.getmtime(path)
        except OSError:
            return default

    # lease sidecar exists from t0 (shell path: touch '{leasef}')
    try:
        open(leasef, "a").close()
        os.utime(leasef, None)
    except OSError:
        pass
    logdir = cfg.get("cwd")
    proc = subprocess.Popen(cmd, shell=True, cwd=logdir,
                            stdin=subprocess.DEVNULL, **_detached_popen_kwargs())

    if not (lease or maxrt or hb):
        rc = proc.wait()
        if not os.path.exists(exitf):
            _write(exitf, rc)
        return rc

    chk = _bg_poll_interval(lease, maxrt, hb)
    t0 = time.time()
    state = {"mr": 0, "hb": 0, "done": False}

    def _watch():
        while not state["done"]:
            now = time.time()
            if lease and now - _mtime(leasef, now) >= lease:
                _write(exitf, 137)
                _kill_tree(proc.pid)
                return
            if maxrt and state["mr"] == 0 and now - t0 >= maxrt:
                state["mr"] = 1
                if kill_on_max:
                    _write(exitf, 137)
                    _kill_tree(proc.pid)
                    return
                _write(maxrunf, maxrt)
            if hb:
                if now - _mtime(logf, now) >= hb:
                    if state["hb"] == 0:
                        state["hb"] = 1
                        _write(silentf, int(now - _mtime(logf, now)))
                else:
                    state["hb"] = 0
            time.sleep(chk)

    threading.Thread(target=_watch, daemon=True).start()
    rc = proc.wait()
    state["done"] = True
    if not os.path.exists(exitf):
        _write(exitf, rc)
    return rc


def _bg_spawn_detached(cmd, log, exitf, leasef, logf=None, idle_timeout=None,
                       max_runtime=None, kill_on_max=True, heartbeat_timeout=None,
                       cwd=None):
    """The single detached-bg spawn chokepoint shared by core (_bg_launch) and the
    run_command ' &' path. POSIX: build the proven shell wrapper (Tools._bg_wrap)
    and Popen it via /bin/sh - unchanged, battle-tested. Windows (no POSIX shell):
    spawn `python lean_coder.py --bg-run <json>` running run_bg_child, which
    reproduces the identical sidecar contract. Returns the detached Popen handle.
    Its .pid is the group leader (start_new_session / CREATE_NEW_PROCESS_GROUP), so
    _kill_tree(pid) reaps the whole tree on either OS. Caller opens `log` for stdout."""
    f = open(log, "ab")
    try:
        if os.name == "nt":
            cfg = {"cmd": cmd, "exitf": exitf, "leasef": leasef,
                   "logf": logf or str(log), "idle_timeout": idle_timeout,
                   "max_runtime": max_runtime, "kill_on_max": kill_on_max,
                   "heartbeat_timeout": heartbeat_timeout,
                   "cwd": str(cwd) if cwd else None}
            argv = [sys.executable, os.path.abspath(__file__), "--bg-run",
                    json.dumps(cfg)]
            return subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,
                                    **_detached_popen_kwargs())
        wrapped = _BUILTINS.Tools._bg_wrap(cmd, exitf, leasef, idle_timeout,
                                           logf=logf, max_runtime=max_runtime,
                                           kill_on_max=kill_on_max,
                                           heartbeat_timeout=heartbeat_timeout)
        return subprocess.Popen(wrapped, shell=True,
                                cwd=str(cwd) if cwd else None,
                                stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                **_detached_popen_kwargs())
    finally:
        f.close()

def _bg_launch(cmd, cwd=None, kind="task", idle_timeout=None,
               heartbeat_timeout=None, heartbeat_file=None):
    """Launch a detached background process on THIS box and register it, returning
    {pid, log, exitf, leasef} or {"error": msg}. The lean-tool-facing bg spawn hook
    (exposed as lc["bg_launch"]): a tool - e.g. dispatch_worker - can start + track a
    bg task via the SAME machinery as a plain backgrounded run_command (sidecar exit,
    optional lease, kind tag) without reaching into core internals or the leash/confirm
    path (a worker's authority is set by its GRANT, gated at dispatch by the tool).

    Reuses the builtins shell wrapper so completion/lease semantics are identical to
    run_command's ' &'. LOCAL only by design: a worker's brain runs on the driver
    (see the worker-agents design); its TOOLS reach a remote via the executor."""
    if not cmd:
        return {"error": "nothing to launch"}
    try:
        logdir = CONFIG_DIR / "bg"
        logdir.mkdir(parents=True, exist_ok=True)
        log = logdir / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{kind}.log"
        exitf = str(log) + ".exit"
        leasef = str(log) + ".lease"
        # heartbeat_file lets a worker point the (bark-not-bite) staleness watchdog at
        # its OWN progress file (bumped each iteration) instead of the bg log, so a worker
        # that's genuinely stuck (model looping, hung tool) - not merely quiet - trips the
        # alert. logf carries that path into the HEARTBEAT channel (shell or Python path).
        hb_watch = heartbeat_file or log
        p = _bg_spawn_detached(cmd, log, exitf, leasef, logf=str(hb_watch),
                               idle_timeout=idle_timeout,
                               heartbeat_timeout=heartbeat_timeout, cwd=cwd)
    except Exception as e:
        return {"error": f"launch failed: {e}"}
    _bg_register(p.pid, cmd, log, kind=kind, idle_timeout=idle_timeout,
                 heartbeat_timeout=heartbeat_timeout)
    return {"pid": p.pid, "log": str(log), "exitf": exitf, "leasef": leasef}


# Named bg hooks for lean-tools. The lc dict passed to a lean-tool's setup() IS the
# core module globals(), so a tool resolves these by name (lc["bg_launch"] etc.) - a
# stable, intention-revealing surface over the _bg_* internals, so dispatch_worker
# never reaches into private helpers. bg_launch: spawn+register a detached task;
# bg_status: status rows (state/runtime/tail); bg_list: live records of a kind.
bg_launch = _bg_launch
bg_status = _bg_status_items
bg_list = _bg_running

# The live Agent, set at repl() startup. A lean-tool's setup() runs before the agent
# exists, so it can't capture it directly; active_remote() reads it lazily at call time.
_active_agent = None


def log_system_activity(kind, summary, detail=""):
    """Module-level bridge so a provider (which has no Agent handle, only the _lc
    dict) can record an automatic-behaviour event - e.g. the 429->Haiku fallback
    or a token refresh. No-op if there's no active agent. Reachable from a
    provider as _lc["log_system_activity"]."""
    ag = _active_agent
    if ag is not None and hasattr(ag, "_log_activity"):
        ag._log_activity(kind, summary, detail)


def active_remote():
    """The parent session's LIVE remote as {host, ctl, cwd}, or None when local.
    dispatch_worker uses this to make a worker ride the SAME executor path the parent
    uses (its tools run where the parent's do), reusing the parent's ControlMaster."""
    ag = _active_agent
    ws = getattr(ag, "remote", None) if ag else None
    if ws is None:
        return None
    return {"host": ws.host, "ctl": ws.ctl,
            "cwd": ws.remote_cwd or ws.cwd}


# ControlMasters opened SOLELY to carry workers to a host the parent isn't itself
# connected to (host -> ctl path). Kept alive for the workers' lifetime and torn down
# at session exit. A host the parent IS connected to reuses that master instead (never
# tracked here) so we don't double-open or tear down a socket the parent is using.
_worker_masters = {}


def ensure_worker_master(host):
    """Foreground-only: make sure a LIVE ssh master exists for `host` so a detached
    worker can ride it (BatchMode, no auth prompt of its own). Returns
    {host, ctl, cwd} on success or {"error": msg}. Resolution order:
      1. a /connect alias (cfg.connect_hosts) resolves to its real target;
      2. if the parent is connected to that host (active or pooled + alive), reuse ITS
         master - the worker rides the exact channel the parent uses;
      3. else open a DEDICATED master with KEY AUTH ONLY (BatchMode). This runs inside
         a tool call - the terminal isn't ours to prompt on - so it must never block on
         a password. Keys -> silent + fast; a password-only host fails fast with a
         'run /connect <host> first' error (that command owns the terminal and prompts
         safely), then the dispatch is retried. A detached worker never authenticates.
    The worker's cwd defaults to '.' on the target (the tool can override)."""
    ag = _active_agent
    if ag is None:
        return {"error": "no active session"}
    cfg = ag.cfg
    host = (cfg.connect_hosts.get(host, host) if getattr(cfg, "connect_hosts", None)
            else host)
    # Parent already on this host (active or pooled + alive): reuse its master + cwd.
    for ws in ([ag.remote] if ag.remote else []) + list(ag.remotes.values()):
        if ws is not None and ws.host == host and _remote_alive(ws):
            return {"host": host, "ctl": ws.ctl, "cwd": ws.remote_cwd or ws.cwd}
    # A dedicated worker master we opened earlier and is still live.
    ctl = _worker_masters.get(host)
    if ctl and _ssh_master_alive(host, ctl):
        return {"host": host, "ctl": ctl, "cwd": "."}
    # Open one now - KEY AUTH ONLY (BatchMode). This runs inside a tool call, where
    # the terminal belongs to the agent, so we must NEVER sit on an invisible password
    # prompt (that hangs the whole session). If keys work it's silent + fast; if a
    # password would be needed, ssh fails fast and we send the operator to /connect,
    # which owns the terminal properly and surfaces the prompt - then they retry.
    ctl = _ctl_path(host)
    _clear_master(host, ctl)
    print(dim(f"opening ssh master to {host} for a worker (key auth) ..."))
    argv = ["ssh", "-o", "ControlMaster=yes", "-o", f"ControlPath={ctl}",
            "-o", "ControlPersist=yes", "-o", "BatchMode=yes",
            # accept-new: BatchMode can't answer a first-contact host-key prompt, so
            # without this a key-auth host that just isn't in known_hosts yet FAILS and
            # gets misreported as 'needs a password'. accept-new trusts a new key (TOFU)
            # but still refuses a CHANGED one (MITM guard), same as interactive /connect.
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8",
            "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR", "-fN", host]
    try:
        rc = subprocess.run(argv, timeout=25).returncode
    except subprocess.TimeoutExpired:
        return {"error": f"ssh to {host} timed out (no worker dispatched)"}
    except Exception as e:
        return {"error": f"ssh to {host} failed: {e} (no worker dispatched)"}
    if rc != 0:
        return {"error": (f"can't reach {host} with key auth. If it needs a password, "
                          f"run '/connect {host}' first (that prompts you safely), then "
                          f"retry the dispatch. No worker dispatched.")}
    _worker_masters[host] = ctl
    return {"host": host, "ctl": ctl, "cwd": "."}


def _close_worker_masters():
    """Tear down every dedicated worker master at session exit (a host the parent is
    connected to isn't in here, so its master is left alone)."""
    for host, ctl in list(_worker_masters.items()):
        _clear_master(host, ctl)
    _worker_masters.clear()


# Worker sidecar scheme (MIRRORS dispatch_worker's file naming - a stable contract):
# a worker with brief path '<stamp>.brief' owns exactly this family, all driver-side
# under CONFIG_DIR/workers (never remote - the worker process is local, only its tools
# reach an executor). Centralised here so every cleanup path nukes the WHOLE family and
# leaves zero mess/trace.
def _worker_brief_from_cmd(cmd):
    """The '--brief-file' path from a worker record's launch cmd, or None. This is the
    crash-survivable link from a bg record to its on-disk worker files."""
    try:
        parts = shlex.split(cmd or "")
        if "--brief-file" in parts:
            return parts[parts.index("--brief-file") + 1]
    except (ValueError, IndexError):
        pass
    return None


def _clean_worker_sidecars(brief):
    """Unlink a worker's entire sidecar family given its '<stamp>.brief' path: the brief,
    its .result, and the .progress/.inject/.injects.log heartbeat+inject files. Best-
    effort; a missing file is fine."""
    if not brief:
        return
    b = str(brief)
    base = b[:-len(".brief")] if b.endswith(".brief") else b
    for p in (b, base + ".result", b + ".progress", b + ".inject", b + ".injects.log"):
        try:
            Path(p).unlink()
        except OSError:
            pass


def _bg_clean_sidecars(rec):
    """Remove a finished/killed task's sidecars so CONFIG_DIR doesn't grow without
    bound. Covers the bg family (log/.exit/.lease and the watchdog .silent/.maxrun
    alert files) AND, for a worker record, its full worker sidecar family. Best-effort;
    a missing file is fine."""
    log = rec.get("log")
    if log:
        for p in (Path(log), Path(str(log) + ".exit"), _bg_lease_path(log),
                  Path(str(log) + ".silent"), Path(str(log) + ".maxrun")):
            try:
                p.unlink()
            except OSError:
                pass
    if rec.get("kind") == "worker":
        _clean_worker_sidecars(_worker_brief_from_cmd(rec.get("cmd")))


def _worker_dir_sweep(grace=3600):
    """Startup: remove ORPHANED worker sidecar files under CONFIG_DIR/workers - the
    residue of a worker hard-killed by a reboot/SIGKILL where even the bg registry was
    lost, so the normal reap paths never saw it. Two guards make this safe when dide is
    running many concurrent sessions on the box:
      1. a file still referenced by a LIVE bg worker record (any owner, THIS box) is
         kept - never delete another session's running worker's files;
      2. a `grace` window (default 1h) protects a worker just launched by another
         session whose record this process hasn't observed yet.
    Zero remote concern: these are driver-side only; the executor dir is wiped by
    _wipe_remote. Host-gated via _bg_here on the reference check. Best-effort."""
    wdir = CONFIG_DIR / "workers"
    if not wdir.is_dir():
        return
    referenced = set()
    for r in _bg_load():
        if r.get("kind") == "worker" and _bg_here(r):
            b = _worker_brief_from_cmd(r.get("cmd"))
            if b:
                referenced.add(os.path.abspath(b))
    now = time.time()
    # Anchor on the brief: clean the whole family for any unreferenced, aged-out brief.
    for f in wdir.glob("*.brief"):
        if os.path.abspath(str(f)) in referenced:
            continue
        try:
            if now - f.stat().st_mtime < grace:
                continue
        except OSError:
            continue
        _clean_worker_sidecars(str(f))
    # Belt-and-braces: a stray .result/.progress/.inject/.injects.log whose .brief is
    # already gone (partial prior cleanup) - drop it once aged out.
    for suf in (".brief.injects.log", ".brief.inject", ".brief.progress", ".result"):
        for f in wdir.glob("*" + suf):
            stamp = f.name[:-len(suf)]
            if (wdir / (stamp + ".brief")).exists():
                continue
            try:
                if now - f.stat().st_mtime >= grace:
                    f.unlink()
            except OSError:
                pass


def _bg_reap_orphans():
    """Startup: on THIS box only, kill background tasks whose owning lean-coder is
    gone and prune dead entries; keeps a crashed session from leaving daemons
    running forever. Foreign-host records (a task on another box) are left
    untouched - not ours to probe or reap; they age out via /bg kill. A record
    with no host predates the field and is treated as local (back-compat)."""
    recs = _bg_load()
    if not recs:
        return
    keep, reaped = [], 0
    for r in recs:
        if not _bg_here(r):                       # another box's task -> leave it alone
            keep.append(r)
            continue
        if not _proc_alive(r.get("pid")):
            _bg_clean_sidecars(r)                 # already gone -> drop + tidy its files
            continue
        owner = r.get("owner")
        if owner and not _proc_alive(owner):      # orphan from a dead session -> kill
            _bg_kill(r.get("pid"))
            _bg_clean_sidecars(r)
            reaped += 1
            continue
        keep.append(r)
    _bg_save(keep)
    if reaped:
        print(dim(f"reaped {reaped} orphaned background task(s) from a previous session."))


def _bg_adopt():
    """Executor startup (a fresh executor after a reconnect): ADOPT this-box bg
    tasks whose owning process is gone by re-stamping owner = my pid - instead of
    reaping them. A remote bg task is detached and SURVIVES the ssh drop that
    killed the prior executor; if the new executor ran the owner-based reap it
    would see the dead owner and kill the still-running task (the foot-gun). Adopt
    keeps it tracked to a live process so /bg still lists it after a reconnect.
    Host-gated: foreign records are left alone. Dead (already-finished) same-box
    tasks are dropped + their sidecars tidied on the way through."""
    me = os.getpid()
    recs = _bg_load()
    if not recs:
        return
    keep, changed = [], False
    for r in recs:
        if not _bg_here(r):
            keep.append(r); continue
        if not _proc_alive(r.get("pid")):        # finished while we were away -> drop
            _bg_clean_sidecars(r); changed = True; continue
        owner = r.get("owner")
        if owner and owner != me and not _proc_alive(owner):
            r["owner"] = me                      # re-parent the survivor to me
            changed = True
        keep.append(r)
    if changed:
        _bg_save(keep)


def _bg_kill_session():
    """Clean exit: kill the background tasks THIS process started (owner pid on
    THIS box), drop them + tidy sidecars. Pegs background jobs to the lean-coder
    that launched them. Host-gated so a pid that collides with a foreign record
    can't make us kill/drop another box's task."""
    me, host = os.getpid(), _HOSTNAME
    recs = _bg_load()
    if not recs:
        return
    def mine(r):
        return r.get("owner") == me and _bg_here(r)
    for r in recs:
        if mine(r):
            if _proc_alive(r.get("pid")):
                _bg_kill(r.get("pid"))
            _bg_clean_sidecars(r)
    remaining = [r for r in recs if not mine(r)]
    if len(remaining) != len(recs):
        _bg_save(remaining)


# Marker lines (a whole line that is only the marker, modulo surrounding space).
# We use the git-conflict-marker SEARCH/REPLACE format (<<<<<<< SEARCH / ======= /
# >>>>>>> REPLACE) that Aider/Cline/Roo-Code use: these markers occur millions of
# times in pretraining corpora (every committed merge conflict), so a model emits
# them near-perfectly zero-shot - unlike a bespoke syntax it must be taught.
# Tradeoff: a file whose CONTENT literally contains these conflict markers can
# confuse the parser; that is rare and accepted. This is the ONE format; no alias.
_RE_OPEN = re.compile(r"^<{3,}\s*SEARCH\s*$")
_RE_DIV = re.compile(r"^={3,}\s*$")
_RE_CLOSE = re.compile(r"^>{3,}\s*REPLACE\s*$")


def _parse_search_replace(diff: str):
    """Line-based, marker-anchored parse. Returns (blocks, error).

    Block format (the git-conflict-marker SEARCH/REPLACE format, ubiquitous in
    pretraining so a model emits it reliably):
        <<<<<<< SEARCH
        <exact old text>
        =======
        <new text>
        >>>>>>> REPLACE
    Only a line that is WHOLLY a marker delimits a block, so content lines that merely
    contain marker characters are kept verbatim. A malformed structure yields a
    specific error string instead of silently dropping the block (which could corrupt
    a file)."""
    OPEN = "<<<<<<< SEARCH"; DIV = "======="; CLOSE = ">>>>>>> REPLACE"
    lines = diff.split("\n")
    blocks = []
    i, n = 0, len(lines)
    saw_open = False
    while i < n:
        if not _RE_OPEN.match(lines[i]):
            i += 1
            continue
        saw_open = True
        i += 1
        search = []
        while i < n and not _RE_DIV.match(lines[i]):
            if _RE_OPEN.match(lines[i]):
                return [], ("malformed block: a new '%s' line appeared before "
                            "the '%s' divider (near line %d of diff)"
                            % (OPEN, DIV, i + 1))
            if _RE_CLOSE.match(lines[i]):
                return [], ("malformed block: '%s' appeared before the '%s' "
                            "divider (near line %d of diff)"
                            % (CLOSE, DIV, i + 1))
            search.append(lines[i]); i += 1
        if i >= n:
            return [], "malformed block: missing '%s' divider after SEARCH" % DIV
        i += 1
        replace = []
        while i < n and not _RE_CLOSE.match(lines[i]):
            if _RE_OPEN.match(lines[i]) or _RE_DIV.match(lines[i]):
                return [], ("malformed block: expected '%s' to close the block "
                            "but found another marker (near line %d of diff)"
                            % (CLOSE, i + 1))
            replace.append(lines[i]); i += 1
        if i >= n:
            return [], "malformed block: missing '%s' to close the block" % CLOSE
        i += 1
        blocks.append(("\n".join(search), "\n".join(replace)))
    if not blocks:
        if saw_open:
            return [], "found '%s' but the block was never completed" % OPEN
        return [], "no SEARCH/REPLACE blocks found"
    return blocks, None

# ----------------------------------------------------------------------------
# Text-encoded tool-call fallback
#
# Some models (e.g. qwen3-coder:30b) emit tool calls as text in the assistant
# content instead of populating Ollama's native message.tool_calls. We parse
# that text into the SAME {"function": {"name", "arguments"}} shape the native
# path produces, so Agent.run_turn dispatches it unchanged. Engaged ONLY when
# no native tool_calls are present.
# ----------------------------------------------------------------------------

# Format A - Qwen XML: <function=NAME><parameter=KEY>VALUE</parameter>…</function>
_FN_RE = re.compile(r"<function=([^>]+?)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+?)\s*>(.*?)</parameter>", re.DOTALL)
# Format B - JSON-in-tags (Hermes/Nous): <tool_call>{"name":…,"arguments":…}</tool_call>
_JSON_TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TC_TAG_RE = re.compile(r"</?tool_call>")
# Fenced code markers (```json ... ``` or bare ```) - stripped after we've pulled
# the JSON object out from between them by brace-scan, so backticks inside a string
# arg never break boundary detection.
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_]*")
# Max leftover non-whitespace chars allowed around a bare-JSON tool call before we
# treat the message as prose (a final answer) instead of a call. Small - a real
# tool-call message is essentially just the object.
_TEXT_TC_PROSE_SLACK = 24


def _strip_one_nl(s: str) -> str:
    """Strip a single leading and trailing newline; preserve inner content
    (matters for multi-line diff/content args)."""
    if s.startswith("\r\n"):
        s = s[2:]
    elif s[:1] in ("\n", "\r"):
        s = s[1:]
    if s.endswith("\r\n"):
        s = s[:-2]
    elif s[-1:] in ("\n", "\r"):
        s = s[:-1]
    return s


def _tc_from_obj(obj):
    """Coerce one parsed object into {"function":{"name","arguments"}} or None.
    Accepts the bare shape {"name","arguments"} and the OpenAI wrapper
    {"function":{"name","arguments"}}. arguments may be a JSON string."""
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not name or not isinstance(name, str):
        return None
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"function": {"name": name, "arguments": args}}


def _coerce_args_to_schema(name, args, schemas):
    """Coerce STRING arg values to the type the tool's JSON schema declares, for
    calls parsed from a text channel that stringifies everything (Qwen's XML
    <parameter=> form). Only boolean/integer/number params are touched, and only
    when the value is a string that cleanly converts - a string param (e.g.
    write_file's content="true") is NEVER coerced, and an unconvertible value is
    left as-is for the tool's own validation to report. `schemas` maps tool name ->
    properties dict; a missing tool/param is a no-op (pass-through)."""
    if not schemas or not isinstance(args, dict):
        return args
    props = schemas.get(name)
    if not isinstance(props, dict):
        return args
    for k, v in list(args.items()):
        if not isinstance(v, str):
            continue
        spec = props.get(k)
        typ = spec.get("type") if isinstance(spec, dict) else None
        s = v.strip()
        if typ == "boolean":
            low = s.lower()
            if low in ("true", "false"):
                args[k] = (low == "true")
        elif typ == "integer":
            try:
                args[k] = int(s)
            except ValueError:
                pass
        elif typ == "number":
            try:
                args[k] = float(s)
            except ValueError:
                pass
    return args


def _json_objects_from_text(text):
    """Yield (obj, start, end) for every complete top-level JSON object in `text`,
    scanning brace-depth while RESPECTING string literals + escapes so a '{' or a
    stray ``` inside a string value never confuses the boundary (the Cline
    triple-backtick gotcha). Only depth-0 objects are yielded."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        blob = text[i:j + 1]
                        try:
                            yield json.loads(blob), i, j + 1
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            j += 1
        i = j + 1


def parse_text_tool_calls(content: str, known_names=None, schemas=None):
    """Parse text-encoded tool calls from assistant content (small local models
    that fall out of the native tool_calls channel and emit the call as TEXT).
    Returns (calls, cleaned_content) where calls is a list of
    {"function": {"name": str, "arguments": dict}}. If nothing valid parses,
    returns ([], content) unchanged so the caller treats it as a final answer.

    `known_names`, if given, is the set of ACTIVE tool names; any parsed call
    whose name isn't in it is rejected (guards against listing/demonstrating a
    tool in prose being mistaken for a call). Covers: Qwen XML <function=>,
    Hermes <tool_call>{...}</tool_call>, bare JSON {"name","arguments"}, the
    OpenAI {"function":{...}} wrapper, a tool_calls:[...] array, and fenced
    ```json blocks - all scanned brace/string-aware so backticks or braces inside
    a string arg can't break boundary detection.

    `schemas`, if given, maps tool name -> its parameter `properties` dict; it is
    used to coerce STRING args from the Qwen XML form (which stringifies every
    value) back to the declared boolean/integer/number type - so e.g.
    notify_on_exit=false is a real False, not the truthy string 'false'."""
    calls = []
    if not content:
        return calls, content
    known = set(known_names) if known_names is not None else None

    def _accept(tc):
        if tc and (known is None or tc["function"]["name"] in known):
            calls.append(tc)
            return True
        return False

    # Format A: Qwen XML function/parameter tags. This channel stringifies every
    # value, so coerce against the schema (booleans/ints/numbers) - the JSON
    # formats below already carry real types and are left untouched.
    for m in _FN_RE.finditer(content):
        name = m.group(1).strip()
        if not name:
            continue
        args = {pm.group(1).strip(): _strip_one_nl(pm.group(2))
                for pm in _PARAM_RE.finditer(m.group(2))}
        args = _coerce_args_to_schema(name, args, schemas)
        _accept({"function": {"name": name, "arguments": args}})

    # Format B: JSON object inside <tool_call> tags.
    for m in _JSON_TC_RE.finditer(content):
        try:
            obj = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        _accept(_tc_from_obj(obj))

    if calls:
        cleaned = _FN_RE.sub("", content)
        cleaned = _JSON_TC_RE.sub("", cleaned)
        cleaned = _TC_TAG_RE.sub("", cleaned).strip()
        return calls, cleaned

    # Formats C/D/E: bare JSON, OpenAI wrapper, tool_calls array, and fenced
    # ```json blocks (a fence just wraps a JSON object we'd find anyway). We
    # ONLY treat these as a call when the content is ESSENTIALLY nothing but the
    # object(s) - prose that merely mentions/quotes JSON is left as a final
    # answer. "Essentially" = after removing the matched object spans (and fence
    # markers) little non-whitespace remains.
    spans = []
    for obj, s, e in _json_objects_from_text(content):
        # tool_calls array wrapper: {"tool_calls":[ {...}, ... ]}
        arr = obj.get("tool_calls") if isinstance(obj, dict) else None
        got_here = False
        if isinstance(arr, list):
            for item in arr:
                if _accept(_tc_from_obj(item)):
                    got_here = True
        elif _accept(_tc_from_obj(obj)):
            got_here = True
        if got_here:
            spans.append((s, e))
    if not calls:
        return [], content
    # Build cleaned content: drop the object spans + surrounding fence markers.
    keep = []
    prev = 0
    for s, e in spans:
        keep.append(content[prev:s])
        prev = e
    keep.append(content[prev:])
    cleaned = "".join(keep)
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    # Guard against a false positive: if meaningful prose remains, this wasn't a
    # bare tool-call message - discard the parse and treat it as a final answer.
    if len(cleaned) > _TEXT_TC_PROSE_SLACK:
        return [], content
    return calls, cleaned


# ----------------------------------------------------------------------------
# Confirmation + diff display
# ----------------------------------------------------------------------------

def _ask(question: str) -> bool:
    if _active_composer is not None:  # park the confirm as state; never interrupt
        return _active_composer.ask(question)
    if _active_spinner:               # don't animate over an interactive prompt
        _active_spinner.stop()
    try:
        ans = input(_rl_safe(bold(f"{question} [Y/n] "))).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("", "y", "yes")


# Approval mode: how writes/commands/non-safe tools are gated.
#   ask     - confirm every action (default)
#   session - confirm once; a 'yes' then auto-approves the rest of this run
#   auto    - never confirm (the old --yolo)
_session_approved = False    # armed once the user approves under approval="session"
APPROVAL_MODES = ("ask", "session", "auto")


def auto_approved(cfg) -> bool:
    """True when an action may run WITHOUT prompting: 'auto' always; 'session' once
    the user has approved one action this run."""
    return cfg.approval == "auto" or (cfg.approval == "session" and _session_approved)


def ask_action(cfg, question: str) -> bool:
    """Confirm a write/command/tool action under the approval mode. Auto-approves
    when auto_approved(); else prompts, and a 'yes' under 'session' arms the rest
    of the run ('ask once')."""
    global _session_approved
    if auto_approved(cfg):
        return True
    ok = _ask(question)
    if ok and cfg.approval == "session":
        _session_approved = True
    return ok


def approval_badge(cfg) -> str:
    """Styled indicator for the active approval mode; '' for the default 'ask'."""
    if cfg.approval == "auto":
        return red("AUTO")
    if cfg.approval == "session":
        return yellow("SESSION-AUTO" if _session_approved else "SESSION")
    return ""


def set_approval(cfg, mode):
    """Set the approval mode and clear any session-arming (so a fresh 'session' must
    be re-approved once)."""
    global _session_approved
    cfg.approval = mode
    _session_approved = False


def _confirm_action(cfg, kind: str, **info) -> bool:
    """The single confirm POLICY for a tool's side effect, keyed by capability kind.
    A tool supplies only the PREVIEW content; this owns the auto-approve short-circuit,
    the preview rendering, and the prompt (with session-arming via ask_action). The
    per-tool methods no longer carry any of this - they just call here. Returns True to
    proceed. Kinds:
      write: path, original (str|None), new (str) -> diff / new-file preview, 'apply changes to PATH?'
      exec:  cmd (str)                            -> '$ cmd', 'run this command?'
      read:  name (str), args_fmt (str)           -> 'NAME(args)', 'allow read NAME?'
    (The remote transport has its own pre-send confirm in confirm_remote_tool; this is
    the LOCAL policy.)"""
    if kind == "write":
        path, original, new = info["path"], info["original"], info["new"]
        # Auto-approved: no preview here. The uniform result-preview line (printed
        # under the call line for EVERY tool) already shows the outcome + /expand, so a
        # separate pencil note would just double up. Only the interactive prompt below
        # renders the full diff (you must SEE a change to approve it).
        if auto_approved(cfg):
            return True
        print()
        if original is None:
            print(cyan(f"+++ new file: {path}"))
            _print_new_file(new)
        else:
            _print_diff(path, original, new)
        return ask_action(cfg, f"apply changes to {path}?")
    if kind == "exec":
        if auto_approved(cfg):
            return True
        print(yellow(f"\n$ {info['cmd']}"))
        return ask_action(cfg, "run this command?")
    if kind == "read":
        if auto_approved(cfg):
            return True
        print(yellow(f"\n{info['name']}({info.get('args_fmt', '')})"))
        return ask_action(cfg, f"allow read {info['name']}?")
    return ask_action(cfg, "proceed?")          # unknown kind: prompt to be safe


def _print_diff(path: str, a: str, b: str):
    diff = unified_diff(a.splitlines(), b.splitlines(),
                        fromfile=path, tofile=path, lineterm="")
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            print(green(line))
        elif line.startswith("-") and not line.startswith("---"):
            print(red(line))
        elif line.startswith("@@"):
            print(cyan(line))
        else:
            print(dim(line))


def _print_new_file(text: str, cap: int = 40):
    """Preview a created file with no prior content: the first `cap` lines as
    +adds, then a '(N more lines)' note so a long file never silently truncates.
    Shared by the local (_confirm_action) and remote (confirm_remote_tool) write
    confirms - the caller prints its own header above this body."""
    lines = text.splitlines()
    for ln in lines[:cap]:
        print(green("+ " + ln))
    if len(lines) > cap:
        print(dim(f"  …({len(lines) - cap} more lines)"))


# ----------------------------------------------------------------------------
# Inference control signal (shared by core and provider clients)
# ----------------------------------------------------------------------------

class _TurnAbort(Exception):
    """Raised (or signalled) to stop streaming mid-inference on SIGINT."""


def _menu_index(sel: str, count: int):
    """Parse a 1-based menu selection; return 0-based index or None. Pure."""
    sel = sel.strip()
    if sel.isdigit():
        n = int(sel)
        if 1 <= n <= count:
            return n - 1
    return None


def menu_resolve(arg, ordered):
    """MENU CONTRACT (single source of truth). A command that opens a numbered menu on
    bare invocation MUST accept a bare 1-based integer arg as "pick that item from the
    SAME ordered list the menu shows" - so `/connect 8` == the 8th menu row, never a
    host literally named 8. Returns the chosen item, or None when `arg` isn't a valid
    in-range index (caller then falls through to its name/literal handling). Pure; the
    ordering passed in must match what the command's picker renders. Built on
    _menu_index so the CLI-arg path and the interactive picker share one index rule."""
    idx = _menu_index(str(arg).strip(), len(ordered))
    return ordered[idx] if idx is not None else None


def _connect_menu_items(connect_hosts, open_hosts=()):
    """MENU CONTRACT: the ordered (name, target) list backing BOTH the bare-/connect
    picker AND the `/connect <N>` numeric arg path - saved [connect] targets first (in
    config order), then open-but-unsaved sessions. One source so the two can never
    drift (a numeric arg must select exactly the row the menu renders). Pure."""
    items, seen = [], set()                       # (name, target)
    for name, target in connect_hosts.items():
        items.append((name, target)); seen.add(target)
    for host in open_hosts:                        # open-but-unsaved targets too
        if host not in seen:
            items.append((host, host)); seen.add(host)
    return items


def _connect_menu_order(cfg, agent):
    """The list of ssh TARGETS in menu order (see _connect_menu_items) - what a bare
    integer /connect arg indexes into. Kept beside the picker so both share it."""
    return [t for _n, t in _connect_menu_items(cfg.connect_hosts, list(agent.remotes))]


def pick_connect_menu(connect_hosts, open_hosts=(), active=None, prompt=input):
    """Numbered picker over saved [connect] targets plus any currently-open
    sessions (marked). Returns the chosen ssh target, or None on cancel/empty.
    No liveness probe - SSH handles auth (passwordless straight in, else prompts)."""
    items = _connect_menu_items(connect_hosts, open_hosts)
    if not items:
        print(dim("no saved connect targets; use /connect <[user@]host>, or add a "
                  "[connect] section to the config."))
        return None
    open_set = set(open_hosts)
    targets = [t for _n, t in items]
    labels = []
    for name, target in items:
        dot = green("*") if target in open_set else " "
        extra = dim("  " + target) if name != target else ""
        labels.append(f"{dot} {name}{extra}")
    return pick_one("connect to:" + dim("  (" + green("*") + " open)"),
                       targets, current=active, prompt=prompt, labels=labels)


def pick_model_menu(available, current=None, warm=(), prompt=input):
    """Model picker; returns the chosen model name or None (cancel). `warm` = names
    currently loaded in the server (green dot = instant first token); `current` is
    marked. Rich arrow/#-jump/filter picker on a real tty, numbered fallback otherwise
    (or when a test passes its own `prompt`). `prompt` injectable for tests."""
    if not available:
        print(dim("no models on this host."))
        return None
    warm = set(warm)
    legend = dim("  (" + green("*") + " loaded  " + dim("o") + " on disk)")
    labels = []
    for m in available:
        dot = green("*") if m in warm else dim("o")
        labels.append(f"{dot} {m}")
    return pick_one("models on this host:" + legend, list(available),
                       current=current, prompt=prompt, labels=labels)


def model_available(want: str, available) -> bool:
    """Tag-aware membership: a bare name implies `:latest` (and vice-versa), so
    `demo-coder` matches an installed `demo-coder:latest`. Pure."""
    if want in available:
        return True
    if ":" not in want and f"{want}:latest" in available:
        return True
    if want.endswith(":latest") and want[:-len(":latest")] in available:
        return True
    return False


# ----------------------------------------------------------------------------
# Remote execution (staged). Stage 1: a hermetic tool executor + a driver-side
# client, both speaking newline-delimited JSON. Stage 1 runs the executor as a
# local subprocess (no SSH) so the protocol is fully testable; later stages run
# the same executor on a remote box over an SSH pipe.
#
# The executor is deliberately dumb and secret-free: no config, no model, no
# network egress, no prompts. The driver confirms writes/commands BEFORE sending
# them, so the executor just runs approved tool calls against one directory.
# ----------------------------------------------------------------------------

# --- builtin tools: load + derive -------------------------------------------
# The built-in tool surface (read_file..run_command) lives in the bundled
# lean-tools/builtins.py (canon, always-on). Load it ONCE here - after every helper
# it needs is defined (resolve_in_project, IgnoreMatcher, _parse_search_replace,
# _confirm_action, _bg_*, the READ_/TREE_/SEARCH_/OUTPUT_ constants) - and
# derive the model-facing schema list + the /leash capability tuples from the tiers it
# declares. This is the single source of truth: a tier change in builtins.py reshapes
# the leash here with no core edit. Mirrors how providers/ollama.py owns the backend.

def _load_builtins():
    """Load + inject the bundled builtin-tools module. Required (a coder with no
    read_file is broken), so a missing file is a hard, explained error rather than a
    silent empty tool surface."""
    d = _bundled_lean_tools_dir()
    f = (d / "builtins.py") if d else None
    if not f or not f.is_file():
        raise RuntimeError(
            "bundled builtin tools missing (lean-tools/builtins.py); cannot start - "
            "this file ships with the code and must be present.")
    mod = _load_lean_tool(f)
    mod.setup(globals())            # inject the stable core helpers it resolves by name
    return mod

_BUILTINS = _load_builtins()
Tools = _BUILTINS.Tools             # core holds agent.tools = Tools(cfg) (carries changed_files)
TOOLS = _BUILTINS.TOOL_SCHEMAS      # model-facing schemas, in declared order
_TIERS = _BUILTINS.TIERS            # name -> "read"|"write"|"exec"
_WRITE_TIER = tuple(n for n, t in _TIERS.items() if t == "write")
_EXEC_TIER = tuple(n for n, t in _TIERS.items() if t == "exec")  # ask_user_to_run too (exec)
# Read-only builtins: no side effects, no confirm prompt, so a batch of them can run
# concurrently (see Agent._run_parallel); also the ask-on-read set.
SAFE_TOOLS = tuple(n for n, t in _TIERS.items() if t == "read")
# Every builtin name - the remote executor allowlist (no nested ssh; ssh is a lean-tool).
EXEC_TOOLS = tuple(_TIERS)


def _emit_json(stream, obj):
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


# Driver-only verb (never in the model's tool surface): fetch a file's exact
# bytes for diff previews, base64-encoded so it bypasses run_command's output
# cap/rstrip and is binary-safe.
RAW_READ = "__read_raw__"
BG_POLL  = "__bg_poll__"        # driver-only: finished bg tasks the executor owns
BG_LIST  = "__bg_list__"        # driver-only: all bg tasks (running+finished) it owns, for /bg


def _read_raw_b64(path: Path, max_bytes=RAW_READ_MAX):
    """Return (b64, clipped, total, shown) for path's bytes. Raises OSError if
    unreadable. No truncation marker is ever spliced in; if the file exceeds
    max_bytes the prefix is returned and `clipped` is True (caller warns)."""
    data = path.read_bytes()
    payload = data[:max_bytes]
    return base64.b64encode(payload).decode(), len(data) > max_bytes, len(data), len(payload)


def _sniff_image_media_type(data):
    """Media type from magic bytes (stdlib only). None if not a supported image."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _encode_image_block(path, max_edge=1568):
    """Return (media_type, base64_str) for an image file, or None on any failure
    (missing file, not an image) so the caller falls back to text-only.

    PIL is OPTIONAL, not a hard dep. When present, the image is downscaled so its
    long edge is <= max_edge - keeps Anthropic on the cheap standard vision tier
    (the cost win). When PIL is absent, we send the raw bytes as-is: Anthropic
    auto-resizes oversized images server-side, so it still WORKS - it just costs
    a few more tokens on a big screenshot. Format is detected from magic bytes."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None

    # Best-effort downscale + re-encode when Pillow is available.
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        img.load()
        w, h = img.size
        if max(w, h) > max_edge:
            scale = max_edge / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            media, fmt = "image/png", "PNG"
        else:
            img = img.convert("RGB")
            media, fmt = "image/jpeg", "JPEG"
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return media, base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    # PIL absent (or failed): send raw bytes, detecting the type from magic bytes.
    media = _sniff_image_media_type(raw)
    if not media:
        return None
    return media, base64.b64encode(raw).decode()


def _tool_result_msg(name, result, tool_call_id=""):
    """Build a role:tool message from a tool's return. A str is stored as-is
    (unchanged legacy path). A dict {text, image_path?} stores the text as
    `content` (so eviction/compaction, which key on a string content, keep
    working untouched) and carries `image_path` as a sibling key that only the
    Anthropic serializers read. On a non-vision model those serializers ignore
    it, so images auto-strip on a mid-session model switch.

    `tool_call_id` (the PROVIDER's call id, e.g. 'toolu_…'/'call_…', taken from
    the assistant's tool_calls entry - NOT the internal /expand #N) binds this
    result to its call EXPLICITLY. Every serializer already prefers it and only
    falls back to positional order when it's absent, so setting it makes the
    result<->call pairing order-independent (a batched/parallel result can never
    mis-map) instead of relying on append order. Empty for a backend (ollama/text
    fallback) whose calls carry no id - the positional path still covers those."""
    msg = {"role": "tool", "tool_name": name}
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    if isinstance(result, dict) and "text" in result:
        msg["content"] = str(result.get("text", ""))
        ip = result.get("image_path")
        if ip:
            msg["image_path"] = ip
    else:
        msg["content"] = result if isinstance(result, str) else str(result)
    return msg


def _remote_idle_ttl() -> int:
    """Self-wipe lease TTL (seconds) for a pushed remote executor. Env-tunable;
    defaults to 30min (matches the worker lease). Floor of 60s."""
    try:
        return max(60, int(os.environ.get("LEANCODER_REMOTE_IDLE_TTL", "1800")))
    except (TypeError, ValueError):
        return 1800


def _arm_self_wipe():
    """Fault-tolerance for a PUSHED remote executor: leave nothing behind if the
    driver never gets to run a clean disconnect (broken pipe, crash, hard-killed
    worker). The pushed copy is always '<runtime>/leancoder/agent.py'; we ONLY arm
    when running from exactly that (basename agent.py, parent dir 'leancoder') so a
    provisioned `lean_coder` install or a local dev run is never touched.

    Two parts, both self-contained on the remote (no per-turn driver ssh):
      - a daemon BUMP thread touches '<dir>/.lease' every TTL/4 while THIS process
        lives, so lease-freshness == executor-alive (survives long model 'thinking'
        gaps; stops the instant the process dies);
      - a DETACHED shell WATCHDOG (own session, so it outlives this process and the
        ssh channel) that `rm -rf`s the dir once the lease is unbumped for TTL, or
        exits early if the dir is already gone (a clean disconnect wiped it).
    Reconnect within TTL relaunches the executor, whose new bump thread refreshes the
    lease -> no wipe (self-heal). Only one watchdog runs: a prior one (from a died
    executor reusing this fixed dir) is killed via '<dir>/.watchdog.pid' first.
    Best-effort throughout; any failure just means we fall back to the OS clearing
    the throwaway dir, and the files are secret-free code anyway."""
    try:
        mydir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return
    if os.path.basename(__file__) != "agent.py" or os.path.basename(mydir) != "leancoder":
        return                              # not a pushed executor - do nothing
    ttl   = _remote_idle_ttl()
    chk   = max(5, min(30, ttl // 4))
    lease = os.path.join(mydir, ".lease")
    pidf  = os.path.join(mydir, ".watchdog.pid")
    try:
        Path(lease).touch()
    except OSError:
        pass
    # kill any prior watchdog bound to this reused dir (one watchdog invariant)
    try:
        old = int(Path(pidf).read_text().strip())
        if old != os.getpid():
            _kill_tree(old, signal.SIGTERM)
    except (OSError, ValueError):
        pass
    # Detached watchdog: a Python child (own session/process group, so it outlives
    # this executor and the ssh channel) that rm -rf's the dir once the lease goes
    # unbumped for TTL. Pure-Python + shutil.rmtree so it runs identically on POSIX
    # and native Windows (no sh/stat/date/rm dependency).
    try:
        argv = [sys.executable, os.path.abspath(__file__), "--wipe-watch",
                json.dumps({"dir": mydir, "ttl": ttl, "chk": chk})]
        wp = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              **_detached_popen_kwargs())
        Path(pidf).write_text(str(wp.pid))
    except (OSError, ValueError):
        return
    def _bump():
        while True:
            try:
                os.utime(lease, None)
            except OSError:
                pass
            time.sleep(chk)
    threading.Thread(target=_bump, daemon=True).start()


def run_wipe_watchdog(cfg):
    """Detached self-wipe watchdog (cross-platform twin of the old POSIX sh loop).
    Spawned as `python lean_coder.py --wipe-watch <json>`, own session/group, so it
    outlives the executor + ssh channel. Polls the pushed dir's '.lease' mtime; once
    unbumped for `ttl` it shutil.rmtree's the dir and exits, or exits early if the dir
    is already gone (a clean disconnect wiped it). `cfg`: dir, ttl, chk."""
    d = cfg["dir"]
    ttl = int(cfg["ttl"])
    chk = int(cfg["chk"])
    lease = os.path.join(d, ".lease")
    while True:
        if not os.path.isdir(d):
            return
        now = time.time()
        try:
            m = os.path.getmtime(lease)
        except OSError:
            m = now
        if now - m >= ttl:
            shutil.rmtree(d, ignore_errors=True)
            return
        time.sleep(chk)


def run_tool_executor(cwd, lean_tools_dir=None):
    """Hermetic executor loop: read JSON tool requests on stdin, run them against
    `cwd`, write JSON results on stdout. Tool prints (diffs, notices) are sent to
    stderr so they never pollute the protocol channel. Runs until stdin closes.
    Any lean-tools under `lean_tools_dir` (pushed by the driver) are also dispatchable."""
    # Pin the protocol stdio to UTF-8 (the driver's ExecutorClient reads UTF-8). On a
    # Windows executor - esp. embeddable Python under cmd.exe - stdout/stdin default to
    # cp1252, so a non-ASCII char in a tool result (the '·' separators) would mojibake to
    # 'Â·' on the driver. reconfigure() is a no-op if already UTF-8; guarded for any
    # stream that doesn't support it (test doubles).
    for _s in (sys.stdout, sys.stdin, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    proto = sys.stdout
    sys.stdout = sys.stderr            # keep real stdout clean for the protocol
    _arm_self_wipe()   # pushed executor: self-delete if the driver never cleanly disconnects
    cfg = Config(cwd=Path(cwd).expanduser().resolve(), approval="auto")  # driver pre-confirms
    tools = Tools(cfg)
    lean_tools = LeanToolManager(lean_tools_dir) if lean_tools_dir else None  # enabled=None -> all
    _bg_adopt()   # reconnect: re-parent this-box bg tasks that survived the prior executor
    _emit_json(proto, {"ready": True, "cwd": str(cfg.cwd)})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid, tool, args = req.get("id"), req.get("tool"), (req.get("args") or {})
        if tool == RAW_READ:               # driver-only: exact bytes for previews
            try:
                b64, clipped, total, shown = _read_raw_b64(
                    resolve_in_project(cfg, args.get("path", "")))
            except (OSError, ValueError):
                _emit_json(proto, {"id": rid, "ok": False, "error": "cannot read"})
                continue
            _emit_json(proto, {"id": rid, "ok": True, "result": b64,
                               "clipped": clipped, "total": total, "shown": shown})
            continue
        if tool == BG_POLL:                # driver-only: finished bg tasks WE own here
            # Attend our leased tasks here too: a remote task's lease lives on THIS
            # (executor) box, so bumping it must happen where the file is - each
            # driver turn calls BG_POLL, so this is the remote per-turn heartbeat.
            _me = os.getpid()
            for _r in _bg_load():
                if _r.get("owner") == _me and _r.get("idle_timeout"):
                    _bg_bump_lease(_r)
            _emit_json(proto, {"id": rid, "ok": True,
                               "result": _bg_finished_here(_me)})
            continue
        if tool == BG_LIST:                # driver-only: all bg tasks (running+finished) WE own here
            _emit_json(proto, {"id": rid, "ok": True,
                               "result": _bg_status_items(os.getpid())})
            continue
        plug = lean_tools.get(tool) if lean_tools else None
        if tool not in EXEC_TOOLS and not plug:
            _emit_json(proto, {"id": rid, "ok": False, "error": f"tool not allowed: {tool}"})
            continue
        try:
            if plug:
                result = str(plug["run"](args, str(cfg.cwd)))
            else:
                result = getattr(tools, tool)(**args)
            _emit_json(proto, {"id": rid, "ok": True, "result": result})
        except Exception as e:
            _emit_json(proto, {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


class ExecutorClient:
    """Driver-side handle to a tool executor subprocess (stage 1: local; later:
    over SSH). Sends a tool request and returns its result string."""

    def __init__(self, argv, stderr=None):
        # Pin the protocol channel to UTF-8 on BOTH ends (the executor reconfigures its
        # own stdio to match - see run_tool_executor). Without this, text=True decodes
        # with the driver's locale while a Windows executor emits cp1252, so any non-ASCII
        # char in a tool result (e.g. the '·' separators) mojibakes to 'Â·'. errors=replace
        # keeps a stray undecodable byte from tearing the whole channel down.
        # start_new_session: run the ssh channel in its OWN process group so a
        # terminal ^C (SIGINT to the whole foreground group) does NOT reach the ssh
        # child. Without this, ^C to abort a turn killed the exec channel -> the next
        # request hit a dead pipe -> _route_remote misread it as a link drop and
        # reconnected. Now ^C only trips the cooperative self._abort path (clean
        # abort, master + channel intact). Windows: its own group, same effect.
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, bufsize=1,
            encoding="utf-8", errors="replace",
            **_detached_popen_kwargs())
        self._id = 0
        self.host = None                # set by RemoteWorkspace for wait-spinner labels
        self.ready = self._read_obj()   # consume the {"ready": ...} handshake

    def _read_obj(self, should_abort=None, deadline=None):
        """Read the next JSON object from the executor's stdout.

        Blocks until a line arrives - but WAKES periodically (via select) so a
        stalled read isn't a dead, uninterruptible freeze. A remote round-trip
        can STALL (not die) when the link blips: the socket neither delivers nor
        EOFs. We ride that out (a blip recovers, and the caller's reconnect only
        fires on a real drop/EOF) while staying responsive:
          - after REMOTE_WAIT_GRACE seconds of silence, show a 'waiting on <host>'
            spinner so it never looks like a hung blank screen;
          - if should_abort() goes true (^C during the wait), raise _TurnAbort to
            drop cleanly back to the prompt - the master is left intact;
          - if deadline (monotonic) passes, raise TimeoutError - a soft, caller-set
            budget for non-critical calls (e.g. bg_poll) that must never block a turn.
        None of these tear the connection down; that stays the caller's job."""
        fd = self.proc.stdout.fileno()
        t0 = time.monotonic()
        spin = None
        try:
            while True:
                if should_abort is not None and should_abort():
                    raise _TurnAbort()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("remote did not respond within the budget")
                # select() when the stream is a real fd (remote pipe / local proc);
                # a non-selectable stream (test doubles) falls through to a plain read.
                try:
                    ready = select.select([fd], [], [], 0.25)[0]
                except (OSError, ValueError):
                    ready = True
                if not ready:
                    if spin is None and time.monotonic() - t0 >= REMOTE_WAIT_GRACE:
                        where = f" on {self.host}" if self.host else ""
                        spin = Spinner(f"waiting for remote{where}",
                                       TOOL_FRAMES, yellow, interval=0.12).start()
                    continue
                line = self.proc.stdout.readline()
                if not line:
                    raise ConnectionError("executor closed before responding")
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        finally:
            if spin is not None:
                spin.stop()

    def request(self, tool, args=None, should_abort=None, deadline=None):
        """Send a tool request; return the full response object. should_abort/deadline
        are threaded to _read_obj so a stalled reply stays interruptible + bounded."""
        self._id += 1
        _emit_json(self.proc.stdin, {"id": self._id, "tool": tool, "args": args or {}})
        while True:
            obj = self._read_obj(should_abort=should_abort, deadline=deadline)
            if obj.get("id") == self._id:
                return obj

    def call(self, tool, args=None, should_abort=None):
        obj = self.request(tool, args, should_abort=should_abort)
        return obj["result"] if obj.get("ok") else f"error: {obj.get('error')}"

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ----------------------------------------------------------------------------
# Remote workspace (SSH transport, staged). Bootstraps lean_coder onto a host
# and runs the hermetic --tool-exec executor there, driven by the same
# ExecutorClient. Auth happens once via a multiplexed master socket
# (ControlMaster); later calls reuse it. The script lives in throwaway space
# ($XDG_RUNTIME_DIR or /tmp/leancoder-<uid>) and is re-pushed each connect, so
# nothing accumulates. All bootstrap output is driver-side, never in context.
# ----------------------------------------------------------------------------

# remote shell snippet: pick/create the throwaway dir, write the script from
# stdin, echo the dir back on stdout.
# Throwaway dir picker: XDG_RUNTIME_DIR, else TMPDIR ($PREFIX/tmp on Termux, where
# /tmp doesn't exist), else /tmp/leancoder-<uid>. Same precedence as the local
# _runtime_dir(), so a Termux box works as both driver AND remote executor.
_REMOTE_TMP = '${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp/leancoder-$(id -u)}}/leancoder'
_REMOTE_PUSH = (f'd="{_REMOTE_TMP}"; '
                'mkdir -p "$d" && chmod 700 "$d" 2>/dev/null; '
                'cat > "$d/agent.py"; printf %s "$d"')
# Same stable dir, but make it WITHOUT pushing the script - used when we can reuse
# an existing matching-hash executor (installed lean_coder, or a prior push).
_REMOTE_MKDIR = (f'd="{_REMOTE_TMP}"; '
                 'mkdir -p "$d" && chmod 700 "$d" 2>/dev/null; printf %s "$d"')

# Windows/PowerShell equivalents (used when the target's DefaultShell is PowerShell,
# which has no sh/mkdir -p/chmod/cat/printf). Throwaway dir: %TEMP%\leancoder. PUSH
# reads the piped agent.py bytes from stdin and writes them, then echoes the dir with
# NO trailing newline (the driver .strip()s it anyway). -Encoding Byte would corrupt a
# utf-8 source, so we read the raw stdin stream and write bytes verbatim. `chmod 700`
# has no analogue; the per-user %TEMP% is already ACL'd to the user.
_WIN_REMOTE_TMP = '$d = Join-Path $env:TEMP "leancoder"'
_WIN_REMOTE_MKDIR = (
    f'{_WIN_REMOTE_TMP}; '
    'New-Item -ItemType Directory -Force -Path $d | Out-Null; '
    '[Console]::Out.Write($d)')

# Embeddable-Python fallback for a Windows host with NO real interpreter. Stock Win11
# ships none (only a Store stub), and we won't silently winget-install. Instead we
# provision the official python-<ver>-embed-amd64.zip - a self-contained embeddable CPython
# with a FULL stdlib (the import spike confirmed our executor imports cleanly under it:
# all stdlib, all lazy imports guarded). It is CACHED under %LOCALAPPDATA% in a neutral
# dir (survives our %TEMP% wipe, so we download once), and its python<ver>._pth is
# rewritten to un-comment `import site` so the interpreter behaves normally. Prints the
# path to python.exe on success, or 'ERR:<reason>'. Idempotent: reuses a cached unzip.
_WIN_EMBED_VER = "3.12.7"
_WIN_EMBED_URL = (f"https://www.python.org/ftp/python/{_WIN_EMBED_VER}/"
                  f"python-{_WIN_EMBED_VER}-embed-amd64.zip")
_WIN_PROVISION_EMBED = (
    "$ErrorActionPreference='Stop'; try {"
    "  $root = Join-Path $env:LOCALAPPDATA 'lc-runtime';"
    "  $dir  = Join-Path $root 'py-embed-amd64';"
    "  $exe  = Join-Path $dir 'python.exe';"
    "  if (-not (Test-Path $exe)) {"
    "    New-Item -ItemType Directory -Force -Path $root | Out-Null;"
    "    $zip = Join-Path $root 'embed.zip';"
    f"    Invoke-WebRequest -UseBasicParsing -Uri '{_WIN_EMBED_URL}' -OutFile $zip;"
    "    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }"
    "    Expand-Archive -Path $zip -DestinationPath $dir -Force;"
    "    Remove-Item -Force $zip;"
    "    Get-ChildItem -Path $dir -Filter 'python*._pth' | ForEach-Object {"
    "      (Get-Content $_.FullName) -replace '^#\\s*import\\s+site','import site'"
    "        | Set-Content $_.FullName };"
    "  }"
    "  [Console]::Out.Write($exe);"
    "} catch { [Console]::Out.Write('ERR:' + $_.Exception.Message) }")

# ---------------------------------------------------------------------------
# POSIX embeddable Python (no-interpreter Linux/BSD boxes + zero-trace mode)
# ---------------------------------------------------------------------------
# Mirror of the Windows embed path for a POSIX host with no python3 (and for the
# --ephemeral dogfood/zero-trace mode). We ship a self-contained CPython from
# astral-sh/python-build-standalone: the `install_only` tarball is a normal
# ./python/bin/python3 tree with a FULL stdlib (our executor is all-stdlib +
# guarded lazy imports, so it runs cleanly under it). We pick the build by the
# remote's C library: the `install_only` tarballs are DYNAMICALLY linked against
# their libc's loader (gnu -> /lib64/ld-linux..., musl -> /lib/ld-musl-...), so a
# musl binary will NOT run on a glibc box and vice-versa (only the '+static-full'
# variants are truly static, and those aren't published as install_only). So we
# probe the remote libc (glibc vs musl) and fetch the matching build. Release +
# CPython version + per-(arch,libc) sha256 are PINNED here (reproducible,
# verifiable) - never "latest". The driver DOWNLOADS the tarball once (local
# cache, sha256-verified), then PUSHES it to the remote over the existing ssh
# master - the remote never reaches the internet, so an air-gapped box works and
# the transfer is auditable. Uncovered arch -> a hard, actionable error (never a
# silent wrong-arch binary).
_POSIX_EMBED_RELEASE = "20260718"        # python-build-standalone release tag (pinned)
_POSIX_EMBED_PYVER = "3.10.20"           # CPython version in that release (pinned)
# (arch slug, libc) -> sha256 of that install_only tarball. `libc` is "gnu" (glibc)
# or "musl", chosen per-remote by _posix_embed_libc. uname -m tokens map to a slug.
_POSIX_EMBED_UNAME = {
    "x86_64": "x86_64", "amd64": "x86_64",
    "aarch64": "aarch64", "arm64": "aarch64",
}
_POSIX_EMBED_SHA = {
    ("x86_64", "gnu"):   "9c28d8017eeaf692f24dbaf26fd4679ce496c7f58e48b897d278739661794e37",
    ("x86_64", "musl"):  "013ff8ab30357820d99df80745ea58b49fb693165a27b599b9fec09c4780dd20",
    ("aarch64", "gnu"):  "506d003732d99a1598b63a40b53fa9359460a3be7488b9a3123ea6cf8ed1627b",
    ("aarch64", "musl"): "1e3a67cd2a207b281815bfe7c9e7bc41b2881dd7460e7efb3c3705f651bdf731",
}


def _posix_embed_asset(uname_m: str, libc: str = "gnu"):
    """Map a remote `uname -m` token + libc ('gnu'|'musl') to (asset_filename, url,
    sha256) for the pinned install_only tarball, or None if the (arch, libc) pair
    isn't covered (caller hard-errors). Normalises case/whitespace so 'X86_64\\n'
    resolves. libc defaults to 'gnu' (the common case) so callers/tests can omit it."""
    slug = _POSIX_EMBED_UNAME.get((uname_m or "").strip().lower())
    if not slug:
        return None
    libc = (libc or "gnu").strip().lower()
    sha = _POSIX_EMBED_SHA.get((slug, libc))
    if not sha:
        return None
    fn = (f"cpython-{_POSIX_EMBED_PYVER}+{_POSIX_EMBED_RELEASE}-"
          f"{slug}-unknown-linux-{libc}-install_only.tar.gz")
    url = (f"https://github.com/astral-sh/python-build-standalone/releases/download/"
           f"{_POSIX_EMBED_RELEASE}/{fn}")
    return fn, url, sha


def _posix_embed_cache_download(fn: str, url: str, sha: str) -> Path:
    """Fetch the pinned tarball to the LOCAL driver cache (~/.cache/leancoder/
    lc-runtime), verifying sha256; reuse a cached copy that already matches. Returns
    the local Path. Raises ConnectionError on download/verify failure. Download-once:
    a second POSIX embed connect reuses the file (no re-fetch)."""
    cache = Path(os.path.expanduser("~/.cache/leancoder/lc-runtime"))
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / fn

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if dest.is_file() and _sha(dest) == sha:
        return dest                                # cached + verified: reuse
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except (urllib.error.URLError, OSError) as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ConnectionError(f"failed to download embeddable Python ({url}): {e}")
    got = _sha(tmp)
    if got != sha:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ConnectionError(
            f"embeddable Python tarball sha256 mismatch: got {got}, expected {sha}")
    tmp.replace(dest)
    return dest


# Remote cache layout for the pushed embed runtime. Kept UNDER a fixed lc-runtime
# leaf (like Windows' %LOCALAPPDATA%\\lc-runtime) so ephemeral teardown can wipe
# exactly it and never an unrelated path. XDG_CACHE_HOME honoured, else ~/.cache.
_POSIX_EMBED_ROOT = '${XDG_CACHE_HOME:-$HOME/.cache}/leancoder/lc-runtime'

# Probe run over ssh at connect to detect the remote OS/shell family. A bare `uname`
# is the most portable discriminator: a POSIX shell prints a known kernel token
# ("Linux"/"Darwin"/"FreeBSD"/etc); Windows cmd/PowerShell have no `uname`, so the
# channel prints an error / nothing recognisable. Detection is done driver-side by
# _remote_is_posix() on the captured output (NOT on shell operators, which differ
# between sh and PowerShell < 7). Kept trivial so it behaves in ANY remote shell.
_OS_PROBE = "uname"
_POSIX_UNAMES = ("linux", "darwin", "bsd", "sunos", "aix", "cygwin", "msys", "gnu")


def _remote_is_posix(uname_out) -> bool:
    """Classify a remote from the _OS_PROBE (`uname`) output. True = POSIX shell
    (Linux/macOS/BSD/Termux): use the ControlMaster + sh push path unchanged. False =
    Windows (uname absent -> error/empty/'not recognized'): use the non-multiplexed
    ssh + PowerShell push path. Errs toward POSIX only on a clear kernel token, so an
    ambiguous/empty answer is treated as Windows (the safer branch to probe further)."""
    s = (uname_out or "").strip().lower()
    return any(tok in s for tok in _POSIX_UNAMES)


def _remote_os_label(uname_out) -> str:
    """A short human name for the probed remote OS, from the _OS_PROBE (`uname`) output -
    surfaced on the connect 'probed' line so a misdetect is visible immediately. POSIX
    kernels map to a friendly name. Anything without a recognisable kernel token is
    'unknown' - we do NOT assume Windows here (that call belongs to the caller, which
    knows the box connected fine and simply lacks `uname`); an empty/garbled answer
    that could be anything must not masquerade as a confident 'Windows'."""
    s = (uname_out or "").strip().lower()
    if "linux" in s:
        return "Linux"
    if "darwin" in s:
        return "macOS"
    if "bsd" in s:
        return "BSD"
    if any(tok in s for tok in ("sunos", "aix", "cygwin", "msys", "gnu")):
        return s.split()[0].strip() or "POSIX"
    return "unknown"


# ssh exits 255 for its OWN failures (can't connect, auth refused, timeout) as
# opposed to relaying the remote command's exit code. That's how we tell "the box
# is genuinely Windows (connected fine, no `uname`)" from "we never reached a
# shell at all" - the latter must NOT be misread as Windows (it was, and the real
# ssh error got buried under a doomed embeddable-Python push).
_SSH_UNREACHABLE = (
    "network is unreachable", "no route to host", "connection refused",
    "connection timed out", "connection closed", "operation timed out",
    "could not resolve", "name or service not known", "timed out",
    "host is down", "no address associated",
)
_SSH_AUTH_DENIED = (
    "permission denied", "authentication failed", "too many authentication",
    "password", "publickey", "host key verification failed",
)


def _classify_ssh_probe(rc, stdout, stderr):
    """Classify the connect-time OS probe. Returns (kind, detail):
      "posix"       - reached a POSIX shell (uname kernel token in stdout)
      "windows"     - CONNECTED (rc 0) but no uname -> a real Windows box
      "auth"        - reachable but ssh couldn't authenticate (needs a password/key;
                      BatchMode refused it). Caller retries WITHOUT BatchMode.
      "unreachable" - never reached the host (network/route/DNS/refused/timeout).
    rc 255 is ssh's own-failure code; anything else means the remote shell ran."""
    if _remote_is_posix(stdout):
        return "posix", ""
    err = (stderr or "").strip()
    low = err.lower()
    if rc == 255 or rc is None:
        if any(tok in low for tok in _SSH_AUTH_DENIED) and not any(
                tok in low for tok in _SSH_UNREACHABLE):
            return "auth", err
        if any(tok in low for tok in _SSH_UNREACHABLE) or not err:
            return "unreachable", err or "ssh could not connect"
        # An unrecognised 255 is a connection-level failure too - surface it.
        return "unreachable", err
    # rc != 255: ssh connected and ran the command; no uname => genuine Windows.
    return "windows", ""


def _runtime_dir() -> Path:
    """Local ephemeral dir for control sockets. Prefers XDG_RUNTIME_DIR, then TMPDIR
    ($PREFIX/tmp on Termux, where /tmp doesn't exist), then /tmp - so it works on
    Android/Termux and other non-FHS layouts, not just standard Linux. os.getuid is
    absent on some platforms (Windows), so the uid suffix is best-effort."""
    uid = os.getuid() if hasattr(os, "getuid") else os.environ.get("USER", "user")
    base = (os.environ.get("XDG_RUNTIME_DIR")
            or os.environ.get("TMPDIR")
            or "/tmp")
    d = Path(base) / f"leancoder-{uid}" / "leancoder"
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)
    except OSError:
        pass
    _reap_dead_sockets(d)
    return d


def _reap_dead_sockets(d: Path) -> None:
    """Unlink orphan control sockets left by instances that are no longer running.
    Control paths are per-PID (cm-<host>-<pid>.sock) so two instances can't evict
    each other's master; the cost is that a crashed/killed instance leaves its
    socket behind. Reap any whose PID is dead - os.kill(pid, 0) raises for a gone
    process and is a no-op for a live one, so a running instance's socket is never
    touched. Best-effort: any error is ignored (e.g. no os.kill on the platform)."""
    try:
        mypid = os.getpid()
        for p in d.glob("cm-*.sock"):
            m = re.search(r"-(\d+)\.sock$", p.name)
            if not m:
                continue
            pid = int(m.group(1))
            if pid == mypid:
                continue
            try:
                os.kill(pid, 0)          # alive -> leave it
            except ProcessLookupError:
                p.unlink(missing_ok=True)  # dead -> reap
            except (PermissionError, OSError):
                pass                     # exists but not ours to signal -> leave
    except Exception:
        pass


def _ctl_path(host: str) -> str:
    # Per-PID so concurrent instances on one box don't share a master socket:
    # a second instance's /connect (_clear_master) or /disconnect does `ssh -O
    # exit` on the ControlPath, which would otherwise tear down the FIRST
    # instance's live master -> its next remote tool call hits a dead socket ->
    # "Broken pipe" -> session drops to LOCAL. Isolated paths stop the stomping.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", host)
    return str(_runtime_dir() / f"cm-{safe}-{os.getpid()}.sock")


def _ssh_master_argv(host, ctl, connect_timeout=10, batch=False):
    # opens a backgrounded master; auth (key/password/passphrase) happens here,
    # interactively if needed (no BatchMode so the user can be prompted once).
    #
    # batch=True: BatchMode + accept-new, for a reconnect that runs WHERE WE CAN'T
    # PROMPT (mid-tool-call, or a detached worker). It must never sit on an invisible
    # password prompt (that hangs the session forever), so it either succeeds via keys
    # /a live master or fails fast; accept-new so a not-yet-known host key doesn't stall
    # it either (TOFU on new, refuse changed).
    #
    # ControlPersist=yes (not a timeout): the master is -fN, so it carries NO session
    # traffic and sits fully idle during a long model turn (e.g. minutes of "thinking"
    # with no tool call). A finite ControlPersist reaped it in that gap, so the next
    # tool call hit a dead master -> "Broken pipe" -> the session dropped to LOCAL.
    # `yes` keeps it up until we explicitly tear it down (drop_remote does `-O exit`).
    # ServerAlive* sends TCP keepalives so a NAT/firewall idle timeout can't silently
    # kill the socket either (~15s x 8 = ~2min of dead link before ssh gives up).
    batch_opts = (["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
                  if batch else [])
    return ["ssh", "-o", "ControlMaster=yes", "-o", f"ControlPath={ctl}",
            "-o", "ControlPersist=yes"] + batch_opts + [
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8",
            "-o", f"ConnectTimeout={connect_timeout}",
            "-o", "LogLevel=ERROR", "-fN", host]


def _clear_master(host, ctl):
    """Tear down any existing/stale ControlMaster before a fresh /connect. A
    leftover socket from a killed session makes ControlMaster=yes refuse to
    multiplex and silently fall back to a per-command (re-auth every time)
    connection, breaking the auth-once guarantee. `-O exit` stops a live master
    (no-op + silent if there's none); the unlink clears a stale socket file.
    (This also drops a master shared by another session on the same host - the
    already-unsupported case of two sessions connected to one remote.)"""
    subprocess.run(["ssh", "-o", f"ControlPath={ctl}", "-O", "exit", host],
                   capture_output=True)
    try:
        Path(ctl).unlink()
    except OSError:
        pass


def _ssh_master_alive(host, ctl) -> bool:
    """True if a LIVE ControlMaster socket exists at ctl for host (`ssh -O check`).
    Used by dispatch_worker to verify the parent's shared master is up BEFORE it
    launches a detached worker that will reuse it (a bg worker can't answer an auth
    prompt, so a dead master must be caught in the foreground dispatch call).
    False when there's no ctl (Windows / non-multiplexed transport): there is no
    master to check, so a worker can't ride one - callers must handle that."""
    if not ctl:
        return False
    try:
        return subprocess.run(["ssh", "-o", f"ControlPath={ctl}", "-O", "check", host],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ctl_opts(ctl):
    """SSH options binding a call to a shared ControlMaster socket. Empty when ctl is
    falsy: Windows OpenSSH has NO ControlMaster support (client or server), so the
    Windows transport runs each ssh WITHOUT multiplexing (a fresh auth per channel, or
    key-auth). On POSIX ctl is always set, so behaviour is byte-identical to before."""
    return ["-o", f"ControlPath={ctl}"] if ctl else []


def _scp_argv(host, ctl, local_path, remote_path):
    """scp a local file to `remote_path` on `host`. Reuses the ControlMaster socket
    when `ctl` is set (POSIX); on Windows (ctl falsy) it's a fresh key-auth transfer.
    MEASURED on Win11 OpenSSH (192.0.2.16): a full 648KB agent.py lands byte-identical
    in ~4s - so scp is the Windows push (the old 'scp truncates large files' claim was
    wrong; it replaces the ~3-4min, deadlock-prone chunked base64 path)."""
    return (["scp"] + _ctl_opts(ctl) + ["-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
             "-p", local_path, f"{host}:{remote_path}"])


def _ssh_run_argv(host, ctl, remote_cmd, tty=False):
    # reuses the master socket (BatchMode: auth is already done, so fail fast).
    # LogLevel=ERROR silences the "Shared connection to <host> closed." notice a
    # -tt channel prints when it ends (the master stays up; nothing real closes).
    return (["ssh"] + _ctl_opts(ctl) + ["-o", "BatchMode=yes",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8",
             "-o", "LogLevel=ERROR"]
            + (["-tt"] if tty else ["-T"]) + [host, remote_cmd])


def _win_quote(s):
    """Quote an argument for a Windows PowerShell command line (the DefaultShell on the
    Win11 hosts). PowerShell single-quotes are literal (only '' escapes a quote), which
    is exactly what we want for paths - and unlike shlex.quote (POSIX) an EMPTY string
    becomes a real '' token PowerShell passes through, instead of being dropped so the
    next flag swallows the following arg (that dropped-empty-cwd bug exited the executor
    with an argparse usage error before the handshake)."""
    return "'" + str(s).replace("'", "''") + "'"


def _win_ps(cmd):
    """Wrap a PowerShell command so it runs regardless of the remote sshd DefaultShell.
    Windows OpenSSH runs an exec channel under whatever DefaultShell is configured -
    cmd.exe by DEFAULT, PowerShell only if an admin set the registry key. Our Windows
    commands are all PowerShell syntax, which cmd.exe can't parse (`'$d' is not
    recognized`), so a stock box silently failed every remote command. base64-UTF16LE
    -EncodedCommand is pure [A-Za-z0-9+/=] - it needs NO quoting and passes through ssh,
    cmd.exe AND PowerShell byte-identically, so we no longer depend on the DefaultShell
    being PowerShell. (-NonInteractive so it can never block on a prompt; the JSON stdio
    protocol stays transparent - verified live on a cmd.exe-default Win11 box.)"""
    b64 = base64.b64encode(cmd.encode("utf-16-le")).decode()
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {b64}"


def _ssh_exec_argv(host, ctl, exec_cmd, cwd, lean_tools_dir=None, posix=True):
    # long-lived executor channel: -T (no PTY) keeps stdio clean for the protocol.
    # exec_cmd is the executor invocation prefix - "python3 '<dir>/agent.py'" for a
    # pushed copy, or "lean_coder" for a matching provisioned install. Quoting is
    # per-OS: POSIX shell (shlex) vs Windows PowerShell (_win_quote) - the target runs
    # PowerShell, where POSIX single-quoting mis-parses (and an empty cwd vanishes).
    q = shlex.quote if posix else _win_quote
    inner = f"{exec_cmd} --tool-exec --cwd {q(cwd or '.')}"
    if lean_tools_dir:
        inner += f" --lean-tools {q(lean_tools_dir)}"
    if not posix:
        # Windows: wrap the whole launch in a base64 -EncodedCommand so the long-lived
        # executor channel runs regardless of the sshd DefaultShell (cmd.exe by default).
        # The JSON stdio protocol stays transparent through it (verified live).
        inner = _win_ps(inner)
    # ServerAlive on this long-lived channel too: it multiplexes over the master
    # (which also has keepalives) but a multi-minute model turn with no traffic is
    # exactly when a stateful firewall/NAT drops the data channel - belt-and-braces.
    return (["ssh", "-T"] + _ctl_opts(ctl) + ["-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8",
            "-o", "LogLevel=ERROR", host, inner])


def _tools_to_sync(local, remote):
    """Pure: given {name: sha256} for the locally-enabled lean-tools and {name:
    sha256} already on the remote, return (to_push, to_remove) - tools whose hash
    differs or is missing, and remote tools no longer enabled."""
    to_push = sorted(n for n, h in local.items() if remote.get(n) != h)
    to_remove = sorted(n for n in remote if n not in local)
    return to_push, to_remove


class RemoteWorkspace:
    """A bootstrapped remote executor reached over SSH. Same call() surface as
    ExecutorClient, so the agent loop (stage 3) can route tools through it."""

    def __init__(self, host, cwd=".", lean_tool_paths=(), ctl=None, ephemeral=False):
        self.host = host
        self.cwd = cwd
        # ctl given = ATTACH mode: reuse a master socket someone else already
        # opened+authenticated (a worker reusing its PARENT's ControlMaster). The
        # worker is a detached bg proc that can't answer an auth prompt, so it must
        # NEVER open/clear a master - only ride an existing one. ctl None = own it
        # (the per-PID default): open + tear down our own master as usual.
        self.owns_master = ctl is None
        self.ctl = ctl or _ctl_path(host)
        self.lean_tool_paths = list(lean_tool_paths)
        # ephemeral (Windows): FORCE the embeddable-Python fallback even when the host
        # has a real interpreter, and WIPE the cached %LOCALAPPDATA% runtime on teardown -
        # a zero-trace, no-install-left-behind mode (also the way to dogfood the embed path
        # on a box that already has Python).
        self.ephemeral = ephemeral
        self.remote_dir = None
        self.remote_cwd = None        # executor's resolved working dir (from handshake)
        self.remote_posix = True      # set by the uname probe in connect(); False = Windows
        self.client = None

    def _run(self, remote_cmd, tty=False):
        # Windows: every remote command we emit is PowerShell, but sshd runs the exec
        # channel under its DefaultShell (cmd.exe by default). Wrap in a base64
        # -EncodedCommand so it runs no matter the DefaultShell (see _win_ps). tty is a
        # POSIX-only path (the sudo prompt), so it's never wrapped.
        if not getattr(self, "remote_posix", True) and not tty:
            remote_cmd = _win_ps(remote_cmd)
        if tty:                                    # interactive (sudo prompt is the user's)
            return subprocess.run(_ssh_run_argv(self.host, self.ctl, remote_cmd, tty=True)).returncode, "", ""
        r = subprocess.run(_ssh_run_argv(self.host, self.ctl, remote_cmd),
                           capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout, r.stderr

    def connect(self, batch=False):
        """Open the ssh master + bootstrap the executor. batch=True forces a
        NON-INTERACTIVE open (BatchMode + a timeout) - for a reconnect that runs
        where we can't prompt (mid-tool-call, or a detached worker): it must fail
        fast rather than hang on an invisible 'password:' prompt. A password host
        then needs an interactive /connect. batch=False (default) is the interactive
        open where ssh may prompt ONCE for a password/passphrase.

        Order: open the ControlMaster FIRST (the single auth - key OR one interactive
        password prompt), THEN detect the OS by running `uname` OVER that master
        (BatchMode reuse, no second auth). A Linux client can drive a shared master
        to both POSIX and Windows sshd, so we don't need to know the OS to open it -
        and this is the one place auth happens, so a password host is asked exactly
        once (the old probe-then-master order asked twice). If ssh never reaches the
        host, or auth fails, we surface the REAL ssh error here instead of misreading
        empty output as 'Windows' and marching into a doomed embeddable-Python push.

        Windows/unknown boxes: not every Windows sshd honours session-reuse, so we
        keep the master only if `ssh -O check` confirms it live+reusable; otherwise
        we drop to per-call connections (self.ctl=None, the old always-correct
        behaviour). All Windows commands are stdin-free (powershell -EncodedCommand /
        scp over its own channel), so the historic stdin-piped deadlock can't recur."""
        print(dim(f"connecting to {self.host} ..."))
        with _stage("opening ssh", done="ssh ready") as _st:
            _clear_master(self.host, self.ctl)     # drop any stale socket first
            argv = _ssh_master_argv(self.host, self.ctl, batch=batch)
            try:
                # capture_output so we can classify a failure from ssh's own stderr;
                # an interactive password prompt still reaches the tty regardless.
                r = subprocess.run(argv, capture_output=True, text=True,
                                   timeout=30 if batch else None)
            except subprocess.TimeoutExpired:
                _clear_master(self.host, self.ctl)
                raise ConnectionError(f"ssh connection to {self.host} timed out")
            if r.returncode != 0:
                kind, detail = _classify_ssh_probe(r.returncode, "", r.stderr)
                _clear_master(self.host, self.ctl)
                if kind in ("unreachable", "auth"):
                    _st.done = "unreachable" if kind == "unreachable" else "auth failed"
                    raise ConnectionError(f"can't reach {self.host}: {detail}")
                # Connected + authed but the shared master didn't hold (a Windows
                # sshd that won't multiplex): fall back to per-call connections.
                self.ctl = None
                _st.done = "ssh ready (per-call)"
        # Master (or per-call fallback) is up. Detect the OS by running `uname` over
        # it - reuses the master when we have one (BatchMode, no re-auth).
        with _stage("probing host", done="probed") as _st:
            uname_out = uname_err = ""
            try:
                _p = subprocess.run(_ssh_run_argv(self.host, self.ctl, _OS_PROBE),
                                    capture_output=True, text=True, timeout=30)
                uname_out, uname_err = _p.stdout, _p.stderr
            except (subprocess.SubprocessError, OSError):
                pass
            self.remote_posix = _remote_is_posix(uname_out)
            if self.remote_posix:
                label = _remote_os_label(uname_out)
            elif any(t in (uname_err or "").lower() for t in
                     ("not recognized", "commandnotfound", "cmdlet")):
                label = "Windows"          # ran the channel, no uname -> genuine Windows
            else:
                label = "unknown"          # connected but couldn't tell (never assume)
            _st.done = f"probed · {label}"
            if self.remote_posix:
                self._check_posix_python()
            else:
                # Windows/unknown: keep the master only if it's genuinely reusable; some
                # Windows sshd builds don't honour session-reuse. Else drop to per-call.
                if self.ctl and not _ssh_master_alive(self.host, self.ctl):
                    _clear_master(self.host, self.ctl)
                    self.ctl = None
                self._check_windows_python()
            if getattr(self, "remote_py_desc", ""):
                _st.done = f"probed · {label} · {self.remote_py_desc}"
        self._bootstrap_executor()
        return self

    def attach(self):
        """ATTACH to a master a PARENT already opened (worker reuse): DON'T open or
        clear the master - just verify it's live, then bootstrap our own executor
        channel over it. A worker is detached and can't answer an auth prompt, so a
        dead master is a hard error (the parent must ensure it's up before spawning).
        Python is assumed present (the parent already connected here)."""
        if not _ssh_master_alive(self.host, self.ctl):
            raise ConnectionError(
                f"no live ssh master for {self.host} at {self.ctl} (parent's "
                f"ControlMaster is down - a worker can't authenticate on its own)")
        self._bootstrap_executor()
        return self

    def _reuse_executor(self, want):
        """Find an EXISTING remote executor whose source hash == `want`, so we can
        skip the ~50KB push. Prefers a provisioned `lean_coder` on PATH, then a
        copy pushed earlier into the stable runtime dir. Returns (exec_cmd, dir),
        or (None, dir) when nothing matches (caller pushes). Stateless: it only
        ever runs `--version`, never leaves anything behind beyond the dir."""
        if not want:
            return None, None
        if not self.remote_posix:
            return self._reuse_executor_win(want)
        _, d, _ = self._run(_REMOTE_MKDIR)
        rdir = d.strip() or None
        if rdir is None:
            return None, None
        _, out, _ = self._run("command -v lean_coder >/dev/null 2>&1 && "
                              "lean_coder --version 2>/dev/null || true")
        if want in out.split():
            self.reused = "installed"
            return "lean_coder", rdir
        agent_py = rdir + "/agent.py"
        py = getattr(self, "posix_python", None) or "python3"
        _, out, _ = self._run(f"[ -f {shlex.quote(agent_py)} ] && "
                              f"{py} {shlex.quote(agent_py)} --version 2>/dev/null || true")
        if want in out.split():
            self.reused = "pushed"
            return f"{py} " + shlex.quote(agent_py), rdir
        return None, rdir

    def _reuse_executor_win(self, want):
        """Windows variant of _reuse_executor: make the throwaway dir via the
        PowerShell mkdir snippet and check for a prior pushed agent.py whose
        --version matches. No provisioned `lean_coder` on PATH is assumed on
        Windows, so we only ever reuse a prior push."""
        _, d, _ = self._run(_WIN_REMOTE_MKDIR)
        rdir = d.strip() or None
        if rdir is None:
            return None, None
        agent_py = rdir + "\\agent.py"
        py = getattr(self, "win_python", None) or "python"
        # PowerShell: run --version only if the file exists; compare driver-side. The
        # leading '&' (call operator) is REQUIRED when win_python is a quoted absolute
        # path (the embeddable-runtime case) - PowerShell parses a statement that starts
        # with a "quoted string" as a string LITERAL, not a command, without it. Harmless
        # for a bare 'python'/'py -3'.
        _, out, _ = self._run(
            f'if (Test-Path "{agent_py}") {{ & {py} "{agent_py}" --version }}')
        if want in out.split():
            self.reused = "pushed"
            return f'& {py} "{agent_py}"', rdir
        return None, rdir

    def _bootstrap_executor(self):
        """Launch the executor, pushing this build only if the box doesn't already
        have a matching-hash copy. Used by connect() and reload() - reload reuses
        the live master socket, so it never re-authenticates."""
        self.reused = None
        exec_cmd, self.remote_dir = self._reuse_executor(source_hash())
        if exec_cmd is None:                       # no match - push a fresh copy
            with open(__file__, "rb") as f:
                src = f.read()
            if self.remote_posix:
                with _stage("pushing agent", done="agent pushed"):
                    r = subprocess.run(_ssh_run_argv(self.host, self.ctl, _REMOTE_PUSH),
                                       input=src, capture_output=True, timeout=60)
                    self.remote_dir = r.stdout.decode(errors="replace").strip()
                    if r.returncode != 0 or not self.remote_dir:
                        raise ConnectionError(
                            f"failed to push agent to {self.host}: "
                            f"{r.stderr.decode(errors='replace')[:200]}")
                py = getattr(self, "posix_python", None) or "python3"
                exec_cmd = f"{py} " + shlex.quote(self.remote_dir + "/agent.py")
            else:
                # Windows: make the throwaway dir, then scp the agent in (stdin-piped
                # push deadlocks vs Windows sshd; scp lands the file byte-exact via its
                # own channel).
                if not self.remote_dir:
                    _, d, _ = self._run(_WIN_REMOTE_MKDIR)
                    self.remote_dir = d.strip()
                if not self.remote_dir:
                    raise ConnectionError(f"failed to make remote dir on {self.host}")
                with _stage("pushing agent", done="agent pushed"):
                    self._win_scp_write(self.remote_dir + "\\agent.py", src)
                py = getattr(self, "win_python", None) or "python"
                # '&' call operator: required when py is a quoted embed-runtime path (see
                # _reuse_executor_win); harmless for a bare python.
                exec_cmd = f'& {py} "{self.remote_dir}\\agent.py"'
        # The single-file agent.py push is NOT self-contained: at import core runs
        # _load_builtins() from <dir>/lean-tools/builtins.py (the builtin tool surface was
        # extracted out of core). Ship that file beside agent.py so the remote import
        # succeeds - without it the executor dies before the handshake ("executor closed
        # before responding"). A reused provisioned `lean_coder` carries its own copy.
        if self.reused != "installed":
            self._push_builtins()
        lean_tools_remote = self._push_lean_tools() if self.lean_tool_paths else None
        with _stage("starting executor", done="ready"):
            self.client = ExecutorClient(
                _ssh_exec_argv(self.host, self.ctl, exec_cmd, self.cwd, lean_tools_remote,
                               posix=self.remote_posix),
                stderr=subprocess.DEVNULL)
            self.client.host = self.host          # label the wait-spinner with the box
            self.remote_cwd = self.client.ready.get("cwd")

    def read_raw(self, path):
        """Fetch exact remote file content for diff previews; None if unreadable.
        Byte-exact (base64 over a dedicated verb) - not subject to run_command's
        output cap or rstrip, so the previewed diff matches the real file."""
        obj = self.client.request(RAW_READ, {"path": path})
        if not obj.get("ok"):
            return None
        if obj.get("clipped"):
            print(yellow(f"  ... preview clipped: {obj.get('shown', 0):,} of "
                         f"{obj.get('total', 0):,} bytes shown (file too large) ..."))
        return base64.b64decode(obj["result"]).decode("utf-8", "replace")

    def bg_poll(self):
        """Finished bg tasks the REMOTE executor owns (its sidecars live on that
        box), as [{pid, cmd, code, tail}] - so a long-running remote job surfaces
        its result on the next turn just like a local one. [] on any error (never
        break a turn over a poll). This is NON-CRITICAL housekeeping that runs
        BEFORE inference every turn, so it gets a short soft deadline: if the link
        is stalled (a blip), skip the note this turn rather than block the user's
        actual message behind it - it'll report on the next turn instead."""
        try:
            obj = self.client.request(BG_POLL, deadline=time.monotonic() + 3)
        except (ConnectionError, OSError, TimeoutError):
            return []
        return obj.get("result") or [] if obj.get("ok") else []

    def bg_list(self):
        """All bg tasks (running + finished) the REMOTE executor owns, as
        _bg_status_items rows [{pid, cmd, state, runtime, started, tail}] - the
        data behind /bg for a connected session (the registry lives on that box).
        [] on any error (never break the command over a poll)."""
        try:
            obj = self.client.request(BG_LIST, deadline=time.monotonic() + 3)
        except (ConnectionError, OSError, TimeoutError):
            return []
        return obj.get("result") or [] if obj.get("ok") else []

    def reload(self, lean_tool_paths=()):
        """Re-push current code + lean-tools and relaunch the executor over the
        SAME ssh master (no re-auth, connection preserved)."""
        if self.client:
            self.client.close()
        self.lean_tool_paths = list(lean_tool_paths)
        self._bootstrap_executor()
        return self

    def _remote_sha256(self, remote_path):
        """Return the remote file's sha256 hex, or '' if absent/unreadable. POSIX:
        `sha256sum`. Windows: PowerShell `Get-FileHash -Algorithm SHA256` (its output
        is UPPERCASE hex, so lowercase to match hashlib.hexdigest())."""
        if self.remote_posix:
            _, out, _ = self._run(f"sha256sum {shlex.quote(remote_path)} 2>/dev/null || true")
            return out.split()[0] if out.split() else ""
        _, out, _ = self._run(
            f'if (Test-Path "{remote_path}") {{ (Get-FileHash -Algorithm SHA256 '
            f'"{remote_path}").Hash }}')
        return out.strip().lower()

    def _remote_mkdir(self, remote_dir):
        if self.remote_posix:
            self._run(f"mkdir -p {shlex.quote(remote_dir)}")
        else:
            self._run(f'New-Item -ItemType Directory -Force -Path "{remote_dir}" | Out-Null')

    def _remote_write(self, remote_path, data):
        """Write bytes to a remote file over ssh. POSIX: `cat > path` over piped
        stdin. Windows: scp (Windows sshd deadlocks on a stdin-piped channel, but scp
        transfers a large file byte-exact in a few seconds - see _win_scp_write)."""
        if self.remote_posix:
            subprocess.run(_ssh_run_argv(self.host, self.ctl, f'cat > {shlex.quote(remote_path)}'),
                           input=data, capture_output=True, timeout=60)
        else:
            self._win_scp_write(remote_path, data)

    def _win_scp_write(self, remote_path, data):
        """Byte-exact large-file push to a Windows remote via scp. Windows sshd
        DEADLOCKS on a stdin-piped exec channel (so `cat > path` can't be used) and the
        old command-line base64 chunking was slow (~3-4 min, 54 calls) and could hang on
        a chunk. scp goes through the same sshd over its own channel: MEASURED on Win11
        OpenSSH (192.0.2.16) a full 648KB agent.py lands byte-identical in ~4s. We write
        the bytes to a local temp file, scp it across, verify sha256 (raises on
        mismatch), and always clean the temp file up."""
        import tempfile
        want = hashlib.sha256(data).hexdigest()
        tf = tempfile.NamedTemporaryFile(prefix="lc_push_", delete=False)
        try:
            tf.write(data)
            tf.close()
            # scp needs a forward-slash / drive-style remote path; the caller passes a
            # backslash Windows path (\\...\\agent.py) - OpenSSH scp accepts it quoted.
            r = subprocess.run(
                _scp_argv(self.host, self.ctl, tf.name, remote_path.replace("\\", "/")),
                capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise ConnectionError(
                    f"scp push to {self.host} failed: {(r.stderr or '')[:200]}")
        finally:
            try:
                os.unlink(tf.name)
            except OSError:
                pass
        got = self._remote_sha256(remote_path)
        if got != want:
            raise ConnectionError(
                f"scp push to {self.host} corrupted {remote_path}: "
                f"sha256 {got or 'missing'} != {want}")

    def _remote_sep(self):
        return "/" if self.remote_posix else "\\"

    def _push_builtins(self):
        """Ship the bundled builtins module into <remote_dir>/lean-tools/builtins.py - the
        exact path the remote import's _load_builtins() reads (next to the pushed
        agent.py). Hash-checked: skip when the remote copy already matches, so a reused
        push doesn't re-transfer it. The file ships with the code, so a missing local copy
        is a build problem, not a remote one - quietly do nothing in that case."""
        src = _bundled_lean_tools_dir()
        local = (src / "builtins.py") if src else None
        if not local or not local.is_file():
            return
        data = local.read_bytes()
        want = hashlib.sha256(data).hexdigest()
        sep = self._remote_sep()
        bdir = self.remote_dir + sep + "lean-tools"
        target = bdir + sep + "builtins.py"
        if self._remote_sha256(target) == want:
            return                                  # already current on the remote
        self._remote_mkdir(bdir)
        self._remote_write(target, data)

    def _push_lean_tools(self):
        """Sync enabled lean-tool files into <remote_dir>/tools and return it.
        Hash-checks what's already there: pushes only changed/new tools and
        removes ones no longer enabled (the stable dir persists between connects,
        so this both saves transfers and prevents stale tools loading remotely)."""
        sep = self._remote_sep()
        pdir = self.remote_dir + sep + "tools"
        self._remote_mkdir(pdir)
        remote = {}
        if self.remote_posix:
            _, out, _ = self._run(f"sha256sum {shlex.quote(pdir)}/*.py 2>/dev/null || true")
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2:
                    remote[Path(parts[1]).name] = parts[0]
        else:
            # PowerShell: name + hash per .py file, one per line
            _, out, _ = self._run(
                f'Get-ChildItem -Path "{pdir}\\*.py" -ErrorAction SilentlyContinue | '
                f'ForEach-Object {{ $_.Name + " " + (Get-FileHash -Algorithm SHA256 '
                f'$_.FullName).Hash }}')
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2:
                    remote[parts[0]] = parts[1].lower()
        local, blobs = {}, {}
        for p in self.lean_tool_paths:
            p = Path(p)
            data = p.read_bytes()
            local[p.name] = hashlib.sha256(data).hexdigest()
            blobs[p.name] = data
        to_push, to_remove = _tools_to_sync(local, remote)
        if to_push:
            n = len(to_push)
            with _stage("syncing tools",
                        done=f"{n} tool{'s' if n != 1 else ''} synced") as _st:
                for i, name in enumerate(to_push, 1):
                    _st.done = f"{i}/{n} synced"
                    self._remote_write(pdir + sep + name, blobs[name])
                _st.done = f"{n} tool{'s' if n != 1 else ''} synced"
        if to_remove:
            if self.remote_posix:
                self._run("rm -f " + " ".join(shlex.quote(pdir + "/" + n) for n in to_remove))
            else:
                for n in to_remove:
                    self._run(f'Remove-Item -Force "{pdir}\\{n}" -ErrorAction SilentlyContinue')
        return pdir

    def _posix_python(self):
        """Return 'python3' if the POSIX remote already has a usable interpreter on
        PATH, else None. A bare probe - no install, nothing left behind. Records the
        version string on self.remote_py_ver for the connect readout."""
        _, out, _ = self._run(
            'command -v python3 >/dev/null 2>&1 && '
            'python3 -c "import sys;print(\\"%d.%d.%d\\"%sys.version_info[:3])" || true')
        out = (out or "").strip()
        if out:
            self.remote_py_ver = out
            return "python3"
        return None

    def _posix_embed_libc(self):
        """Detect the remote's C library so we fetch a runnable install_only build:
        'musl' if a musl loader (/lib/ld-musl-*) is present, else 'gnu' (glibc). The
        install_only tarballs are dynamically linked to their libc's loader, so a musl
        binary won't run on glibc and vice-versa - this picks the matching one. `ldd
        --version` prints 'musl' on Alpine; the loader glob is the robust fallback."""
        _, out, _ = self._run(
            "if ls /lib/ld-musl-* >/dev/null 2>&1; then echo musl; "
            "elif ldd --version 2>&1 | grep -qi musl; then echo musl; "
            "else echo gnu; fi")
        return "musl" if "musl" in (out or "").lower() else "gnu"

    def _provision_posix_embed(self, forced=False):
        """Fallback for a POSIX host with no python3 (and the mechanism behind
        --ephemeral): provision a self-contained CPython from python-build-standalone.
        The driver DOWNLOADS the pinned `install_only` tarball for the remote's
        arch+libc once (local cache, sha256-verified), PUSHES it over the live master,
        extracts it under a fixed lc-runtime cache dir and returns the absolute path to
        its python3, or None on a soft failure. An UNCOVERED arch is a HARD error (never
        a silent wrong-arch binary). Idempotent: a matching extracted runtime is reused.
        `forced`=True means the caller CHOSE the embed runtime (--ephemeral) rather than
        the remote lacking python3 - so the notice must not falsely claim 'no python3'."""
        _, machine, _ = self._run("uname -m || true")
        libc = self._posix_embed_libc()
        asset = _posix_embed_asset(machine, libc)
        if asset is None:
            raise ConnectionError(
                f"no python3 on {self.host} and no pinned embeddable build for its "
                f"architecture/libc ({(machine or '').strip() or 'unknown'}/{libc}). "
                f"Install python3 there (any 3.x), or connect to a covered arch "
                f"({', '.join(sorted(set(_POSIX_EMBED_UNAME)))}).")
        fn, url, sha = asset
        # Resolve the remote cache dir + expected python3 path; reuse if already extracted.
        _, root, _ = self._run(f'printf %s "{_POSIX_EMBED_ROOT}"')
        root = root.strip()
        if not root:
            return None
        # Extracted tree is ./python/bin/python3 (the install_only layout). Tag the dir
        # with the pinned version so a version bump lands beside the old one, not over it.
        rdir = f"{root}/py-{_POSIX_EMBED_PYVER}-{_POSIX_EMBED_RELEASE}"
        py = f"{rdir}/python/bin/python3"
        _, have, _ = self._run(
            f'[ -x {shlex.quote(py)} ] && {shlex.quote(py)} -c '
            f'"import sys;print(sys.executable)" 2>/dev/null || true')
        if have.strip():
            print(dim(f"  reusing embeddable Python at {py}"))
            return py
        if forced:
            print(dim(f"  --ephemeral: pushing a private Python to remote {self.host} "
                      f"(CPython {_POSIX_EMBED_PYVER} {libc}, one-time, cached)."))
        else:
            print(dim(f"no python3 on remote {self.host}; pushing a private one "
                      f"(CPython {_POSIX_EMBED_PYVER} {libc}, one-time, cached). "
                      f"Install python3 ON THE REMOTE to skip this next time."))
        try:
            tarball = _posix_embed_cache_download(fn, url, sha)
        except ConnectionError as e:
            print(yellow(f"  embeddable provision failed: {e}"))
            return None
        # Push the tarball to a temp path on the remote, extract, then remove it.
        remote_tar = f"{root}/{fn}"
        self._run(f"mkdir -p {shlex.quote(rdir)}")
        try:
            self._remote_write(remote_tar, tarball.read_bytes())
        except (OSError, ConnectionError) as e:
            print(yellow(f"  embeddable push failed: {e}"))
            return None
        rc, _, err = self._run(
            f"tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(rdir)} && "
            f"rm -f {shlex.quote(remote_tar)}")
        if rc != 0:
            self._run(f"rm -f {shlex.quote(remote_tar)}")
            print(yellow(f"  embeddable extract failed: {(err or '').strip()[:200]}"))
            return None
        _, ver, _ = self._run(
            f'{shlex.quote(py)} -c "import sys;print(sys.executable)" 2>/dev/null || true')
        if not ver.strip():
            print(yellow("  provisioned python3 did not run; falling back to error."))
            return None
        size = ""
        _, mb, _ = self._run(
            f"du -sm {shlex.quote(rdir)} 2>/dev/null | cut -f1 || true")
        if mb.strip():
            size = f" ({mb.strip()} MB on disk)"
        print(dim(f"  provisioned embeddable Python at {py}{size}"))
        return py

    def _check_posix_python(self):
        """POSIX target: use the system python3 if present; otherwise provision the
        pinned embeddable CPython for its arch+libc (downloaded by the driver, pushed + extracted
        under a fixed lc-runtime cache) so a box with NO interpreter works with zero
        manual setup and NO sudo/package install. ephemeral=True SKIPS the system probe
        and forces the embeddable runtime (wiped on teardown) - a zero-trace mode + the
        way to dogfood the embed path on a box that already has Python. Caches the
        working `python3` invocation on self.posix_python. An uncovered-arch host with
        no python3 is a hard error (raised by _provision_posix_embed)."""
        self.remote_py_ver = ""
        if getattr(self, "ephemeral", False):
            prefix = self._provision_posix_embed(forced=True)
            src = "provisioned"
        elif (prefix := self._posix_python()):
            src = "system"
        else:
            prefix = self._provision_posix_embed()
            src = "provisioned"
        if src == "provisioned" and prefix and not self.remote_py_ver:
            self.remote_py_ver = _POSIX_EMBED_PYVER   # embed distro version is known
        if not prefix:
            raise ConnectionError(
                f"no python3 on {self.host} and the embeddable-runtime fallback failed. "
                f"Install python3 there (any 3.x) and reconnect.")
        self.posix_python = prefix
        ver = f"python {self.remote_py_ver}" if self.remote_py_ver else "python3"
        self.remote_py_desc = f"{ver} ({src})"

    def _win_python(self):
        """Return the Windows Python invocation prefix ('python' or 'py -3'), or None
        if neither is present. Stock Win11 ships NO Python (only a Store stub that
        prints usage and exits nonzero), so we probe for a REAL interpreter by running
        it. `py -3` (the launcher) is preferred when present; else `python`. Records the
        version on self.remote_py_ver for the connect readout."""
        for cand in ("py -3", "python"):
            _, out, _ = self._run(
                f'{cand} -c "import sys;print(sys.executable);'
                f'print(\'%d.%d.%d\'%sys.version_info[:3])"')
            lines = [l for l in (out or "").splitlines() if l.strip()]
            if lines and "\\" in lines[0]:            # a real path back == a real interpreter
                if len(lines) > 1:
                    self.remote_py_ver = lines[1].strip()
                return cand
        return None

    def _provision_win_embed(self, forced=False):
        """Fallback for a Windows host with no real Python: provision the official
        embeddable-amd64 zip (cached under %LOCALAPPDATA%\\lc-runtime, downloaded once)
        and return a quoted 'python.exe' path to invoke it, or None on failure. The
        import spike confirmed our --tool-exec executor runs cleanly under the embed
        distro (all-stdlib, guarded lazy imports). `forced`=True means the caller CHOSE
        the embed runtime (--ephemeral), not that the box lacks Python - so the notice
        must not falsely claim 'no Python'."""
        if forced:
            print(dim(f"  windows - ephemeral - pushing python {_WIN_EMBED_VER} "
                      f"(one-time, cached)"))
        else:
            print(dim(f"  windows - no python - pushing python {_WIN_EMBED_VER} "
                      f"(one-time, cached); install python to skip this next time"))
        rc, out, err = self._run(_WIN_PROVISION_EMBED)
        out = (out or "").strip()
        if rc != 0 or not out or out.startswith("ERR:"):
            reason = out[4:] if out.startswith("ERR:") else (err or "").strip()[:200]
            print(yellow(f"  embeddable provision failed: {reason or 'unknown error'}"))
            return None
        # Verify the provisioned interpreter actually runs before we commit to it. The
        # '&' call operator is needed because `quoted` is a "quoted path" (see
        # _reuse_executor_win) - PowerShell would treat it as a string literal without it.
        quoted = f'"{out}"'
        _, ver, _ = self._run(f'& {quoted} -c "import sys;print(sys.executable)"')
        if not (ver.strip() and "\\" in ver):
            print(yellow("  provisioned python.exe did not run; falling back to error."))
            return None
        # Report the ACTUAL on-disk footprint (no hardcoded transfer size): measure the
        # provisioned dir. Best-effort - a failure just drops the size, never blocks.
        size = ""
        try:
            _, mb, _ = self._run(
                "$d = Split-Path -Parent '" + out + "'; "
                "$b = (Get-ChildItem -Recurse -File $d | Measure-Object -Sum Length).Sum; "
                "[Console]::Out.Write([math]::Round($b/1MB,1))")
            if mb.strip():
                size = f" ({mb.strip()} MB on disk)"
        except Exception:
            pass
        print(dim(f"  provisioned embeddable Python at {out}{size}"))
        return quoted

    def _check_windows_python(self):
        """Windows target: use a real Python if present ('py -3' or 'python'); otherwise
        provision the official embeddable-amd64 zip (cached under %LOCALAPPDATA%) so a
        stock Win11 box works with zero manual setup. Only if BOTH fail do we error with
        an actionable message. Caches the working invocation prefix on self.win_python.
        ephemeral=True SKIPS the installed-Python probe and forces the embeddable runtime
        (which is wiped on teardown) - a zero-trace mode + the way to dogfood the embed
        path on a box that already has Python."""
        self.remote_py_ver = ""
        if getattr(self, "ephemeral", False):
            prefix = self._provision_win_embed(forced=True)
            src = "provisioned"
        elif (prefix := self._win_python()):
            src = "system"
        else:
            prefix = self._provision_win_embed()
            src = "provisioned"
        if src == "provisioned" and prefix and not self.remote_py_ver:
            self.remote_py_ver = _WIN_EMBED_VER   # embed distro version is known
        if not prefix:
            raise ConnectionError(
                f"no Python on the Windows host {self.host} and the embeddable-runtime "
                f"fallback failed. Install one (e.g. `winget install Python.Python.3`) or "
                f"enable the py launcher, then reconnect.")
        self.win_python = prefix
        ver = f"python {self.remote_py_ver}" if self.remote_py_ver else "python"
        self.remote_py_desc = f"{ver} ({src})"

    def call(self, tool, args=None, should_abort=None):
        return self.client.call(tool, args, should_abort=should_abort)

    def close(self):
        if self.client:
            self.client.close()
        # Only tear down a master we OWN. An attached worker rides its parent's
        # master - killing it would drop the parent's live connection.
        if self.owns_master:
            self._wipe_remote()            # zero-trace: remove the pushed code first
            if self.ephemeral:             # ephemeral: also remove the cached embed runtime
                if self.remote_posix:
                    self._wipe_posix_embed()
                else:
                    self._wipe_win_embed()
            if self.ctl:                   # Windows has no master socket to exit
                subprocess.run(["ssh", "-o", f"ControlPath={self.ctl}", "-O", "exit", self.host],
                               capture_output=True)

    def _wipe_win_embed(self):
        """Ephemeral teardown (Windows only): remove the cached embeddable-Python runtime
        under %LOCALAPPDATA%\\lc-runtime so an --ephemeral session leaves NOTHING behind
        (the interpreter we provisioned included). Best-effort, guarded to the fixed
        lc-runtime leaf so it can never remove an unrelated path. Non-ephemeral sessions
        KEEP the cache (that's the point - download once, reuse)."""
        cmd = ('$p = Join-Path $env:LOCALAPPDATA "lc-runtime"; '
               'if (Test-Path $p) { Remove-Item -Recurse -Force $p '
               '-ErrorAction SilentlyContinue }')
        try:
            subprocess.run(_ssh_run_argv(self.host, self.ctl, _win_ps(cmd)),
                           capture_output=True, timeout=30, start_new_session=True)
        except BaseException:
            pass

    def _wipe_posix_embed(self):
        """Ephemeral teardown (POSIX only): remove the pushed embeddable-Python runtime
        under the fixed lc-runtime cache leaf so an --ephemeral session leaves NOTHING
        behind (the provisioned interpreter included). Best-effort, guarded to the fixed
        '.../leancoder/lc-runtime' path so it can never remove an unrelated dir. A
        non-ephemeral session KEEPS the cache (download once, reuse). The LOCAL driver
        cache (~/.cache/leancoder/lc-runtime tarballs) is never touched - it's the reuse
        source, not a remote trace."""
        cmd = (f'p="{_POSIX_EMBED_ROOT}"; '
               'case "$p" in */leancoder/lc-runtime) rm -rf "$p" ;; esac')
        try:
            subprocess.run(_ssh_run_argv(self.host, self.ctl, cmd),
                           capture_output=True, timeout=30, start_new_session=True)
        except BaseException:
            pass

    def _wipe_remote(self):
        """Zero-trace teardown: delete the pushed executor dir (agent.py + bundled
        builtins + synced lean-tools) over the still-live master, so nothing we put
        on the remote outlives the session. Only the master-OWNER wipes: an attached
        worker shares the SAME fixed dir as its parent, so a worker wiping would nuke
        the parent's live executor. Skipped for a REUSED provisioned `lean_coder`
        install (we didn't push it, and remote_dir is just the empty runtime dir).
        Guarded to a '.../leancoder' path so a bad remote_dir can never rm elsewhere.
        Best-effort: runs over the master before it's torn down; any failure is
        swallowed (the files are secret-free code, and the dir is throwaway anyway)."""
        d = (self.remote_dir or "").rstrip("/\\")
        if getattr(self, "reused", None) == "installed":
            return
        # Separator-agnostic basename guard: the driver is POSIX but a Windows remote_dir
        # uses backslashes, and os.path.basename (posixpath here) does NOT split on '\\',
        # so it returned the whole "C:\...\leancoder" string - the guard then always
        # failed and the Windows wipe silently returned EARLY (the pushed dir leaked every
        # session). Split on BOTH separators so the leaf name is checked on either OS.
        leaf = d.replace("\\", "/").rsplit("/", 1)[-1]
        if not d or leaf != "leancoder":
            return
        # Remove the dir FIRST, then kill the watchdog. Order matters for robustness:
        # `rm -rf` drops the .lease/.watchdog.pid sidecars, so ANY watchdog (including
        # stale ones from earlier reconnects that .watchdog.pid no longer names) exits
        # on its next poll ([ -d "$d" ] || exit 0) - the kill is then just a courtesy
        # to reap the current one immediately. Doing rm first means a partial/aborted
        # run still guarantees the trace is gone.
        # Run in its OWN session (start_new_session) so a terminal ^C during a graceful
        # ^C^C/EOF quit - which SIGINTs the whole foreground process group - can't kill
        # this wipe's ssh mid-flight. Catch BaseException for the same reason (a pending
        # KeyboardInterrupt must not abort the wipe before rm lands).
        if self.remote_posix:
            q = shlex.quote(d)
            cmd = (f'rm -rf {q}; '
                   f'p=$(cat {shlex.quote(d + "/.watchdog.pid")} 2>/dev/null) && '
                   f'kill "$p" 2>/dev/null; true')
        else:
            # PowerShell: kill the watchdog by its pid file, then Remove-Item with a
            # short RETRY loop. client.close() already ended the executor (stdin close),
            # so the only process still holding agent.py is the detached self-wipe
            # watchdog (`python agent.py --wipe-watch`, named by .watchdog.pid). Killing
            # it releases the lock, but Windows holds it for a beat after the process
            # exits, so a single Remove-Item races and loses - loop until the dir is gone.
            # (Deliberately NO per-iteration Get-CimInstance process scan: it's ~1-2s each
            # on Windows, so a 20x loop blew the 30s ssh timeout and got killed mid-wipe,
            # leaking the dir - the pid-file kill is enough.)
            cmd = (f'$pf="{d}\\.watchdog.pid"; '
                   f'if (Test-Path $pf) {{ $p=Get-Content $pf; '
                   f'taskkill /F /T /PID $p 2>$null }}; '
                   f'foreach ($i in 1..15) {{ '
                   f'Remove-Item -Recurse -Force "{d}" -ErrorAction SilentlyContinue; '
                   f'if (-not (Test-Path "{d}")) {{ break }}; '
                   f'Start-Sleep -Milliseconds 300 }}')
            cmd = _win_ps(cmd)   # run PowerShell regardless of the sshd DefaultShell
        try:
            subprocess.run(_ssh_run_argv(self.host, self.ctl, cmd),
                           capture_output=True, timeout=30, start_new_session=True)
        except BaseException:
            pass


# ----------------------------------------------------------------------------
# Direct mode: the operator runs a command in a real terminal (so sudo prompts
# work live, never entering context). Reached two ways: the model's
# ask_user_to_run handoff, or the user's /sh command. The command is shown with
# a loud LOCAL/REMOTE label and is editable before it runs.
# ----------------------------------------------------------------------------

def _direct_prompt(cfg, remote=None) -> str:
    # Loud, worded label so the operator always knows WHERE Enter sends it.
    if remote is not None:
        return bold(yellow(f"[remote: {remote.host}] $ "))
    return bold(cyan("[local] $ "))


def confirm_remote_tool(cfg, host, name, args, safe=False, fetch=None):
    """Driver-side confirmation for tools routed to a remote executor (which
    runs non-interactively, so the human-in-the-loop must stay here). Read-only
    tools, safe lean-tools, and auto-approval skip it. `fetch(path)` returns the remote
    file's current content (or None) so we can show a real unified diff, just
    like local. Returns True to proceed."""
    if auto_approved(cfg) or safe or name in ("read_file", "list_files", "search_files"):
        return True
    label = bold(yellow(f"[remote: {host}]"))
    if name == "run_command":
        print(f"\n{label} $ {args.get('cmd', '')}")
        return ask_action(cfg, "run this remote command?")
    if name == "write_file":
        path, content = args.get("path", ""), args.get("content", "")
        old = fetch(path) if fetch else None
        print(f"\n{label} {cyan('write ' + path)} ({len(content.splitlines())} lines)")
        if old is not None:
            _print_diff(path, old, content)
        else:
            _print_new_file(content)
        return ask_action(cfg, f"write {path} on {host}?")
    if name == "replace_lines":
        path = args.get("path", "")
        start, end = args.get("start", "?"), args.get("end", "?")
        new_text = args.get("new_text", "")
        old = fetch(path) if fetch else None
        print(f"\n{label} {cyan('replace lines ' + str(start) + '-' + str(end) + ' in ' + path)}:")
        if old is not None:
            lines = old.splitlines(keepends=True)
            try:
                s, e = int(start), int(end)
                replacement = new_text
                if replacement and not replacement.endswith("\n") and e < len(lines):
                    replacement += "\n"
                new = "".join(lines[:s-1]) + replacement + "".join(lines[e:])
                _print_diff(path, old, new)
            except (TypeError, ValueError):
                print(dim(f"  new_text: {new_text[:500]}"))
        else:
            print(dim(f"  lines {start}-{end} -> {len(new_text.splitlines())} new lines"))
        return ask_action(cfg, f"replace lines in {path} on {host}?")
    if name == "apply_diff":
        path, diff = args.get("path", ""), args.get("diff", "")
        old = fetch(path) if fetch else None
        print(f"\n{label} {cyan('apply diff to ' + path)}:")
        if old is not None:
            new = old
            parsed, _perr = _parse_search_replace(diff)
            for s, r in parsed:
                new = new.replace(s, r, 1)
            _print_diff(path, old, new)
        else:
            print(dim(diff[:1000]))      # fallback: show the raw blocks
        return ask_action(cfg, f"apply this diff on {host}?")
    print(f"\n{label} {cyan(name)}({_fmt_args(args)})")
    return ask_action(cfg, f"run {name} on {host}?")


def operator_note(cmd: str, exit_code, proposed: str = None, output: str = None) -> str:
    """The note fed back to the model after the operator runs something, so it
    stays in sync. Shows the actual command (and any edit) and, when captured, the
    command's OUTPUT so the model can act on it (e.g. a systemctl status / log dump
    it asked the operator to run). Never any password - the pty capture excludes the
    echo-off password entry. Pure function for testability."""
    line = f"$ {cmd}"
    if proposed is not None and proposed.strip() != cmd.strip():
        line += f"   (edited from: {proposed})"
    tail = f"exit {exit_code}" if exit_code is not None else "(not run)"
    note = f"[operator ran a command directly]\n{line}\n{tail}"
    if output is not None:
        out = output.strip()
        note += ("\noutput:\n" + _clip_output(out) if out else "\noutput: (none)")
    return note


def _clip_output(s: str) -> str:
    """Head/tail truncate captured command output to OUTPUT_MAX_CHARS, same shape as
    the run_command cap, so a long log dump can't flood the model's context."""
    if len(s) <= OUTPUT_MAX_CHARS:
        return s
    dropped = len(s) - OUTPUT_HEAD - OUTPUT_TAIL
    return s[:OUTPUT_HEAD] + f"\n…[truncated {dropped} chars]…\n" + s[-OUTPUT_TAIL:]


# CSI / OSC / other ANSI escape sequences - stripped from CAPTURED pty output before
# it goes to the model (the operator's live view keeps them; the model wants text).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]"      # CSI (cursor/colour/mode)
                      r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC (title etc.)
                      r"|\x1b[()][A-Za-z0-9]"           # charset select
                      r"|\x1b[=>NOME78]")               # misc single-char escapes
# Alternate-screen enter (?1049h / ?47h / ?1047h): a full-screen TUI (nano/vim/less/
# htop) took over the terminal. Its screen-paint isn't meaningful "output", so we
# don't feed the model a wall of escape noise - just a note that a TUI was used.
_ALT_SCREEN_RE = re.compile(r"\x1b\[\?(?:1049|1047|47)h")


def _clean_captured(text: str):
    """Turn raw pty bytes-as-text into what the MODEL should see. Returns:
      - a short NOTE string if a full-screen TUI ran (nano/vim/less/htop): its
        screen-paint isn't meaningful output, so we don't feed a wall of escapes;
      - else the ANSI-stripped text.
    None only ever means 'no capture at all' (the pty fallback), kept distinct so the
    caller can tell 'TUI ran' from 'couldn't capture'. The operator already saw the
    live, un-stripped stream; this only shapes the model's copy."""
    if text is None:
        return None
    if _ALT_SCREEN_RE.search(text):
        return "[operator used a full-screen program; its screen output was not captured]"
    cleaned = _ANSI_RE.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


def _tee_capture(argv, cwd=None) -> tuple:
    """Run argv attached to a pty so the operator sees output LIVE and can answer an
    interactive prompt (sudo), while we also CAPTURE the output to hand back to the
    model. The password itself is never captured: it's typed with echo off, so it
    isn't in the pty's output stream. Returns (exit_code, captured_text). Falls back to
    an uncaptured terminal-inherit run if pty isn't available (returns text=None)."""
    try:
        import pty
    except ImportError:
        code = subprocess.run(argv, cwd=cwd).returncode
        return code, None
    buf = []

    def _read(fd):
        data = os.read(fd, 4096)
        buf.append(data)
        return data          # pty.spawn writes this to our stdout -> operator sees it live

    # pty.spawn has no cwd arg; wrap a local command in a cd. A remote argv (ssh ...)
    # carries its own cwd on the far side, so it's passed through untouched.
    status = pty.spawn(argv, _read)
    code = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status
    raw = b"".join(buf).decode("utf-8", "replace")
    # Shape for the MODEL: strip ANSI escapes; a full-screen TUI (nano/vim/htop) ->
    # None, signalling the caller to attach a short note instead of screen-paint noise.
    # The operator's live view above was the raw, un-stripped stream, untouched.
    return code, _clean_captured(raw)


def _edit_prompt(prompt: str, prefill: str) -> str:
    """Editable input pre-filled with `prefill` (via readline). Empty = cancel."""
    if readline is not None:
        # insert_text alone: the startup hook fires inside readline's initial
        # display, so a manual redisplay() here draws the prefilled line twice.
        readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    finally:
        if readline is not None:
            readline.set_startup_hook()


def run_direct_command(cfg, cmd: str, reason: str = None, remote=None,
                       editable=True) -> str:
    """Run a command in a real terminal (local, or on `remote` over the ssh
    master so sudo prompts live); return an operator_note for the model.
    `editable` (default) shows an editable, location-labelled line for the
    operator to confirm/edit - right for the model-proposed ask_user_to_run
    path. `/sh` passes editable=False: the operator already typed it, so just
    label it and run on a single Enter (no second prompt, no re-edit)."""
    if reason:
        print(dim(f"  (model asks the operator to run this: {reason})"))
    if _active_spinner:
        _active_spinner.stop()
    if editable:
        print(dim("  Enter to run (edit first if you like); empty line or ^C to decline."))
        final = _edit_prompt(_direct_prompt(cfg, remote), cmd)
    else:
        print(_direct_prompt(cfg, remote) + cmd)   # show where it runs, then run
        final = cmd
    if not final:
        return "[operator declined to run the command]"
    # Run under a pty so the operator sees output LIVE and any prompt (sudo) works,
    # while we CAPTURE the output for the model to act on. The password isn't captured
    # (typed echo-off, not in the pty stream). Remote: the ssh -tt argv carries its own
    # pty on the far side; local: run via the shell inside our pty.
    if remote is not None:
        argv = _ssh_run_argv(remote.host, remote.ctl, final, tty=True)
    else:
        argv = ["/bin/sh", "-c", f"cd {shlex.quote(str(cfg.cwd))} && {final}"]
    try:
        code, output = _tee_capture(argv)
    except KeyboardInterrupt:
        print()
        return operator_note(final, None, proposed=cmd) + "\n(interrupted)"
    except Exception as e:
        return f"[operator command error] {e}"
    return operator_note(final, code, proposed=cmd, output=output)


def run_sh_command(agent, cfg, arg: str):
    """/sh: run a command in a real terminal where the session points - on the
    remote over the ssh master when connected, else local (same rule as the
    model-driven ask_user_to_run path). Records an operator note so the model
    stays in sync. Returns the note, or None when no command was given."""
    if not arg:
        # No command: drop into a real interactive shell so the operator gets the
        # shell's own line-editing, tab-completion + history. It inherits the
        # terminal; on exit we hand the model a brief note that a shell was used
        # (the individual commands aren't captured - that's the point of a real
        # subshell). When connected, ride the ssh master with a real tty so the
        # REMOTE shell's completion/history work exactly like a local one.
        note = run_interactive_shell(cfg, remote=agent.remote)
        agent.messages.append({"role": "user", "content": note})
        agent.dirty = True
        return note
    # the operator already typed the command: run it on one Enter, no re-edit
    note = run_direct_command(cfg, arg, remote=agent.remote, editable=False)
    agent.messages.append({"role": "user", "content": note})
    agent.dirty = True
    return note


def run_interactive_shell(cfg, remote=None) -> str:
    """Drop into a real interactive shell, inheriting the terminal so its own
    line-editing, tab-completion + history all work. Local: spawn the operator's
    $SHELL. Remote (connected): ride the ssh master with a real tty (-tt) into the
    remote login shell - completion/history work exactly as a local shell. Returns
    an operator_note; the commands run inside aren't captured (a real shell owns
    the tty), so the model is told a shell session happened, not what ran."""
    if _active_spinner:
        _active_spinner.stop()
    if remote is not None:
        # Interactive login shell on the remote over the existing master (no
        # re-auth), started in the session's remote cwd when we know it.
        where = f"{remote.host}"
        rcwd = getattr(remote, "remote_cwd", None) or getattr(remote, "cwd", None)
        launch = "exec ${SHELL:-/bin/sh} -l"
        if rcwd:
            launch = f"cd {shlex.quote(str(rcwd))}; " + launch
        print(dim(f"  entering a shell on {where} (exit or Ctrl-D to return "
                  f"to lean-coder)"))
        try:
            code = subprocess.run(
                _ssh_run_argv(remote.host, remote.ctl, launch, tty=True)).returncode
        except KeyboardInterrupt:
            print()
            code = None
        except Exception as e:
            return f"[interactive shell error] {e}"
        print(dim("  back in lean-coder."))
        tail = f" (last exit {code})" if code is not None else ""
        return (f"[operator dropped into an interactive shell on {where} and "
                f"returned{tail}; the individual commands run there were not "
                f"captured]")
    shell = os.environ.get("SHELL") or "/bin/sh"
    print(dim(f"  entering {shell} (exit or Ctrl-D to return to lean-coder)"))
    try:
        code = subprocess.run([shell], cwd=cfg.cwd).returncode
    except KeyboardInterrupt:
        print()
        code = None
    except Exception as e:
        return f"[interactive shell error] {e}"
    print(dim("  back in lean-coder."))
    tail = f" (last exit {code})" if code is not None else ""
    return (f"[operator dropped into an interactive {shell} session and returned"
            f"{tail}; the individual commands run there were not captured]")


# ----------------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------------

class ContextMeter:
    """Read-only context-window measurement + compaction policy for an Agent. Computes
    how full the window is (used tokens), which compaction zone that puts us in, whether
    a compaction is allowed under the loop guard, and prints the ctx meter line. Holds a
    back-reference to its Agent but only READS from it (last_prompt_tokens, the client's
    cache counters, messages, tool_defs, cfg) - it never mutates conversation state. The
    one writable bit, the last-compaction timestamp, stays owned by the Agent (set in
    auto_handover) and is read here as agent._last_compact_ts. Pulling this cohesive
    read-mostly cluster out of the Agent god-class makes it unit-testable in isolation."""

    def __init__(self, agent):
        self.agent = agent

    @property
    def cfg(self):
        return self.agent.cfg

    def used(self) -> int:
        """Current context in use (tokens) = the FULL prompt size of the last turn:
        uncached input + cache reads + cache writes. Prompt caching makes the bare
        input_tokens tiny (the rest is served from cache), so last_prompt_tokens ALONE
        badly under-reports true context fill - which would both mislead the ctx meter
        and stop the compaction zones from ever firing. Falls back to an estimate when
        there's no real count yet. Also published to lean-tools (see _dispatch)."""
        a = self.agent
        est = messages_tokens(a.messages, a.tool_defs)
        if a.last_prompt_tokens:
            client = getattr(a, "client", None)
            cr = getattr(client, "last_cache_read", 0) or 0
            cw = getattr(client, "last_cache_write", 0) or 0
            reported = a.last_prompt_tokens + cr + cw
            # Ollama (and any server with a fixed num_ctx) SILENTLY truncates a prompt
            # that overflows its window, then reports prompt_eval for only the part that
            # survived. That post-truncation count looks healthy (e.g. 7k of a 32.8k
            # window), so the compaction zones never fire while the model has actually
            # lost most of history. When our own estimate exceeds the reported count,
            # trust the estimate - it reflects what we TRIED to send, so emergency still
            # trips and the fallback strip runs. (Caching legitimately makes reported >
            # est via cr/cw; that direction we keep as-is.)
            return max(reported, est)
        return est

    def zone(self):
        """Context-budget zone for self-managing compaction. Returns (zone, used,
        limit), zone in 'ok' | 'soft' | 'hard' | 'emergency' by fraction of window:
          >= handover_emergency -> 'emergency' (compact NOW, bypass boundary + guard)
          >= hard (per-model)   -> 'hard'      (force at a clean boundary, respect guard)
          >= soft (per-model)   -> 'soft'      (nudge the model to compact at a break)
          else                  -> 'ok'.
        Used only when cfg.auto_handover (per-model) is on."""
        used  = self.used()
        limit = self.cfg.ctx_window() or 1
        frac  = used / limit
        c     = self.cfg.handover_for()        # per-model soft/hard/auto/autostart
        if frac >= self.cfg.handover_emergency:
            return ("emergency", used, limit)
        if frac >= c["hard"]:
            return ("hard", used, limit)
        if frac >= c["soft"]:
            return ("soft", used, limit)
        return ("ok", used, limit)

    def compact_allowed(self, zone) -> bool:
        """Loop guard: emergency always proceeds; otherwise require at least
        handover_min_interval seconds since the last compaction, so a self-driving
        compact->autostart->compact cycle can't burn quota in a tight loop."""
        if zone == "emergency":
            return True
        return (time.time() - self.agent._last_compact_ts) >= self.cfg.handover_min_interval

    def print_meter(self):
        used = self.used()
        window = self.cfg.ctx_window() or 1
        pct = used / window * 100
        # Colour by the auto-handover zone (soft->yellow, hard/emergency->red) when it's
        # on, matching the status-row meter; plain %-of-max colour when handover is off.
        if self.cfg.handover_for().get("auto"):
            zlabel = self.zone()[0]
            col = _zone_color(zlabel)
        else:
            col = _pct_color(pct)
        line = col(f"  ctx: ~{_fmt_tokens(used)}/{_fmt_tokens(window)} ({pct:.0f}%)")
        extra = _provider_usage_str(self.agent, self.cfg)   # backend quota tail (shared)
        badge = approval_badge(self.cfg)      # persistent reminder when not "ask"
        if badge:
            line += dim(f"  {GLYPH['dot']}  ") + badge
        if not extra:
            line += dim(f"  {GLYPH['dot']}  /clear to start fresh")
        print(line + extra)


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ctx = ContextMeter(self)        # read-only ctx measurement + compaction policy
        # No bootstrap client: the active backend is always a provider, and
        # _maybe_autostart_provider -> activate_provider builds the real client at
        # launch (its prepare() hook runs first for transport setup, e.g. ollama
        # host failover). Stays None only if no provider could be started.
        self.client = None
        self.tools = Tools(cfg)
        self.remote = None                   # the ACTIVE RemoteWorkspace, or None (local)
        self.remotes = {}                    # host -> RemoteWorkspace: the live pool
        self.lean_tools = LeanToolManager(_lean_tools_dirs(cfg), enabled=cfg.lean_tools_enabled)
        self.mcp = MCPManager(cfg.mcp_servers, enabled=cfg.mcp_enabled)
        self.tool_defs = self._provider_tool_filter(
            active_tools(cfg, lean_tool_schemas=self._all_lean_schemas(),
                         model_tools=self._model_tool_support()))
        self.messages = [{"role": "system", "content": self._system()}]
        self.last_prompt_tokens = None
        self.session_in = 0              # tokens sent/received this session (any backend);
        self.session_out = 0            # accumulated per chat() call, zeroed on /clear
        self._last_provider = None       # last active provider, for '/provider on'
        self._abort = False
        self._handover_capture = None    # set by finalize_handover -> stops the loop
        self.handovers = 0               # completed handovers this session (manual + auto)
        self._handover_mark = 0          # index where the current /handover turn starts
        self._elected_handover = False   # set by request_handover -> _maybe_handover runs it
        self._last_compact_ts = 0.0      # time of last auto-handover (loop guard)
        self._autostart_pending = None   # self-prompt queued to run as the next turn
        self._queued_turns = []          # user turns queued by a command (e.g. /prompt use);
                                         # the REPL drains these as full turns before prompting
        self._activity = []              # system-activity log: automatic behaviours
                                         # (auto-evict, auto-compact, handover, 429->Haiku,
                                         # token refresh) recorded with a reason so /activity
                                         # shows "what the system did for you, and why".
        self._ac_armed_boundary = 0      # highest k*interval boundary auto-compact has FIRED
                                         # at; hysteresis re-arms it only after ctx drops back
                                         # under (boundary - hysteresis*interval). Schmitt state.
        self._handover_boundary = 0      # message index just past the cache-STABLE prefix left
                                         # by the last handover (0 = none yet). A provider client
                                         # may pin a cache breakpoint here. See auto_handover().
        # permissions note for the model, injected into the next user turn (not the cached
        # system prompt). Seeded at startup only if the leash is restricted (rwe = full = no note).
        self._pending_ai_note = _leash_note(cfg.leash) if cfg.leash != "rwe" else None
        self.notes = []                  # session-scoped episodic memory: list of {dtg, text}
                                         # (the note tool). Persisted in the session JSON, so it
                                         # travels with /load and a pushed session - resume 1:1.
        self.pinned_plan = ""            # goal + TODO the model maintains; rides an uncached
                                         # prompt tail and survives compaction (===PLAN=== block)
        self._tool_calls = []            # ring (cap 100) of {id,name,args} for /expand:
                                         # the full args of recent tool calls, so the
                                         # operator can view what was truncated on-screen
        self._tool_call_seq = 0          # monotonic id assigned to each printed tool call
        self._turn_count = 0             # turns this session (for the soft-zone nudge throttle)
        self._last_nudge_turn = -10**9   # turn of the last soft-zone nudge
        self._bg_announced = set()       # (host,pid) of finished bg tasks already surfaced to the
                                         # model, so a completed long-run job is reported once
        self.dirty = False               # conversation changed since last save/load
        self.autosave_name = _new_autosave_name()   # rolling autosave target this launch

    def use_client(self, client):
        """Swap the active model client and refresh num_ctx for it via the active
        provider's context_window() (ollama detects+caps in there; a hosted backend
        returns its cached window). Lean-tools that replace the client mid-session -
        e.g. from a slash command - should call this rather than assigning `.client`
        directly, so the ctx meter tracks the new backend. Clears the stale token
        estimate too."""
        self.client = client
        self.last_prompt_tokens = None
        spec = self.active_provider()
        if spec is not None:
            try:
                st = self.cfg.provider_settings.setdefault(self.cfg.provider, {})
                st["num_ctx"] = int(spec["context_window"](self.cfg.active_model()))
            except Exception:
                pass

    def active_provider(self):
        """The normalized spec of the live backend, or None only when NO provider is
        enabled at all (every backend, including the bundled ollama, was disabled)."""
        return get_provider(self.cfg.provider) if self.cfg.provider else None

    def activate_provider(self, name):
        """Switch the active backend to a registered provider: build its client,
        restore its saved model, set its context window from context_window(), and
        run its on_activate hook. The model/window live in
        cfg.provider_settings[name] (for ollama, the model also mirrors to the
        top-level alias). Returns the active model id. Raises on an unknown or
        unavailable provider."""
        spec = get_provider(name)
        if spec is None:
            raise ValueError(f"no such provider '{name}'")
        if not spec["available"]():
            raise RuntimeError(f"provider '{name}' is not available")
        self.cfg.provider = name
        st = self.cfg.provider_settings.setdefault(name, {})
        models = list(spec["list_models"]() or [])
        model = st.get("model")
        if model not in models:
            model = models[0] if models else (model or "")
        self.cfg.set_active_model(model)          # writes st["model"] (+ ollama alias)
        # The provider's context_window() is the single source of the window: ollama
        # detects+caps in there, a hosted backend returns its (possibly capped)
        # cached window. No detect_and_cap dance, no native/provider split.
        try:
            st["num_ctx"] = int(spec["context_window"](model))
        except Exception:
            pass
        self.client = spec["make_client"](self.cfg)
        self.last_prompt_tokens = None
        spec["on_activate"](self, self.cfg)
        return model

    def deactivate_provider(self):
        """Switch back to the bundled ollama provider (the default backend). Runs the
        leaving provider's on_deactivate hook. No-op if already on ollama. If ollama
        itself isn't registered, leaves no backend active (cfg.provider = "")."""
        spec = self.active_provider()
        if spec is None or self.cfg.provider == "ollama":
            return
        try:
            spec["on_deactivate"](self, self.cfg)
        except Exception:
            pass
        self._last_provider = self.cfg.provider   # remember it for '/provider on'
        if get_provider("ollama"):
            self.activate_provider("ollama")
        else:
            self.cfg.provider = ""

    def set_remote(self, ws):
        """Make `ws` the active workspace (or None = local) and refresh the tool
        surface + working-directory line. A non-None ws joins the live pool so it
        can be switched back to without reconnecting."""
        self.remote = ws
        if ws is not None:
            self.remotes[ws.host] = ws
        self.tool_defs = self._provider_tool_filter(
            active_tools(self.cfg, remote=bool(ws),
                         lean_tool_schemas=self._all_lean_schemas(),
                         model_tools=self._model_tool_support(),
                         offer_handover=self._offer_handover()))
        self.messages[0] = {"role": "system", "content": self._system()}

    def drop_remote(self, host):
        """Close and forget a pooled connection; if it was active, fall back to
        local. Used on a clean /local <host> and on a detected mid-session drop."""
        ws = self.remotes.pop(host, None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self.remote is not None and self.remote.host == host:
            self.set_remote(None)            # back to local (rebuilds surface + cwd)

    def close_all_remotes(self):
        """Tear down every pooled connection (on quit). Nothing remote survives a
        restart, so there is never stale remote code to worry about."""
        for ws in list(self.remotes.values()):
            try:
                ws.close()
            except Exception:
                pass
        self.remotes.clear()
        self.remote = None

    def _model_tool_support(self) -> bool:
        """Whether the ACTIVE model can use tools at all. A provider may flag a chat-only
        model (e.g. one that 400s on a tools param) via its optional tool_support(model)
        hook. Absent-safe -> True (every current backend supports tools)."""
        spec = self.active_provider()
        if not spec:
            return True
        try:
            return bool(spec["tool_support"](self.cfg.active_model()))
        except Exception:
            return True

    def _provider_tool_filter(self, tools):
        """Last-pass provider filter of the tool surface for the ACTIVE model
        (ollama drops apply_diff for small models). Absent-safe -> unchanged."""
        spec = self.active_provider()
        if not spec or not spec.get("tool_filter"):
            return tools
        try:
            out = spec["tool_filter"](self.cfg, tools)
            return out if isinstance(out, list) else tools
        except Exception:
            return tools

    def _offer_handover(self) -> bool:
        """Whether to surface the elective request_handover tool this turn: only in the
        soft context zone (auto-handover on), i.e. once context has crossed the user's
        handover_soft threshold - the point at which the user has decided spending on a
        handover is acceptable. The model then picks the tidy MOMENT within that window;
        it can never elect one BELOW soft, so it never decides how much the user pays."""
        if not self.cfg.handover_for()["auto"]:
            return False
        return self._ctx_zone()[0] == "soft"

    def _all_lean_schemas(self):
        """The lean-tool-shaped (schema, is_safe) list active_tools consumes: real
        lean-tools PLUS every enabled+connected MCP server's tools (namespaced,
        marked non-safe so they ride the rwe tier). One list so the surface, the
        leash tiering, and dispatch all agree on what MCP contributes."""
        return list(self.lean_tools.schemas()) + list(self.mcp.schemas())

    def refresh_tools(self):
        """Rebuild the model's tool surface (after a /tools toggle, leash or model change)."""
        self.tool_defs = self._provider_tool_filter(
            active_tools(self.cfg, remote=bool(self.remote),
                         lean_tool_schemas=self._all_lean_schemas(),
                         model_tools=self._model_tool_support(),
                         offer_handover=self._offer_handover()))

    def reload_lean_tools(self):
        """Re-scan the lean-tool dir (picks up new/edited/removed files) keeping the
        enabled set, and rebuild the tool surface. Session + ssh untouched."""
        self.lean_tools = LeanToolManager(_lean_tools_dirs(self.cfg), enabled=self.lean_tools.enabled)
        _run_lean_tool_setup(self.cfg, self.lean_tools)   # setup() any newly-enabled tool (once each)
        self.refresh_tools()
        return self.lean_tools.names()

    def _gate_tool(self, name, args, plug, safe, is_mcp):
        """Pre-dispatch gates common to every tool call. Returns a block-message
        string (and prints the operator-facing block) when the call is refused,
        else None to let it through. Order: model-capability lock -> hard leash
        lock -> ask-on-read confirm."""
        # Model-level lock: a chat-only model (provider tool_support=False) has NO tools,
        # and that wins over the leash - raising the ceiling can't grant tools it lacks.
        if not self._model_tool_support():
            model = self.cfg.active_model()
            print(red(f"\n{GLYPH['warn']} blocked {name}: model '{model}' has no tool calling "
                      f"(not run; switch models with /model)"))
            return _model_no_tools_msg(model, name)
        # HARD capability lock (defense in depth behind the active_tools surface filter):
        # refuse to RUN anything above the current /leash, however the call arrived - a
        # hallucinated or injected tool_call, a replayed/loaded message, a leaked surface.
        # The model gets an instructive result; the operator sees the block.
        # MCP tools carry no lean-tool plug; they ride the rwe tier (may have side
        # effects), so pass lean_safe=False for them - matching the surface filter.
        _leash_safe = (safe if plug else (False if is_mcp else None))
        if not _leash_allows_tool(self.cfg.leash, name, lean_safe=_leash_safe):
            print(red(f"\n{GLYPH['warn']} blocked {name}: above the '{self.cfg.leash}' leash "
                      f"(not run; raise with /leash {_min_leash_for(name, _leash_safe)})"))
            return _leash_block_msg(self.cfg.leash, name, lean_safe=_leash_safe)
        # ask-on-read: when confirm_reads is on, read-only tools (core reads + safe
        # lean-tools) confirm too, not just writers - for anyone who doesn't want it
        # reading files without an OK. auto/session-armed approval still skips it.
        if (self.cfg.confirm_reads and (name in SAFE_TOOLS or safe)
                and not auto_approved(self.cfg)):
            if not _confirm_action(self.cfg, "read", name=name, args_fmt=_fmt_args(args)):
                return "operator declined the read"
        return None

    def _route_remote(self, name, args, safe):
        """Run a tool on the connected remote executor (which can't prompt, so the
        confirm happens locally first). Survives a transient exec-channel drop with a
        single silent reconnect+retry; a reconnect that also fails drops to LOCAL and
        tells the model. Tracks written paths for the changed-files set."""
        host = self.remote.host
        try:
            if not confirm_remote_tool(self.cfg, host, name, args,
                                       safe=safe, fetch=self.remote.read_raw):
                return "operator declined the remote command"
            try:
                result = self.remote.call(name, args,
                                          should_abort=lambda: self._abort)
            except (ConnectionError, OSError) as e:
                # The exec channel died - almost always a transient master drop
                # (idle NAT/keepalive timeout, a brief link blip), NOT a reboot.
                # It surfaces on the NEXT request's write to a dead pipe. Try a
                # single silent reconnect (re-establish master + re-bootstrap the
                # executor over the same host) and retry once; only a reconnect
                # that ALSO fails means the box is really gone -> fall to LOCAL.
                # Reconnect NON-INTERACTIVELY (batch): this runs mid-tool-call, so
                # it must never hang on an invisible password prompt. Keys/a revived
                # master -> silent success; a password host or dead box -> fail fast,
                # caught below and dropped to LOCAL (operator re-/connect to prompt).
                print(dim(f"\n  remote {host} link lost ({e}); reconnecting ..."))
                self.remote.connect(batch=True)
                result = self.remote.call(name, args,
                                          should_abort=lambda: self._abort)
        except (ConnectionError, OSError) as e:
            # reconnect failed too: the box rebooted or the link is truly down.
            # Fall back to local and tell the model, instead of crashing.
            self.drop_remote(host)
            print(red(f"\n! remote {host} dropped ({e}); switched to LOCAL."))
            return (f"error: remote {host} dropped mid-command (rebooted or link "
                    f"lost); the session is now LOCAL - retry the command.")
        if name in ("apply_diff", "write_file", "replace_lines") and not any(
                result.startswith(p) for p in
                ("error", "operator declined", "user declined", "no changes")):
            if args.get("path"):
                self.tools.changed_files.add(args["path"])
        return result

    def _run_tool(self, name, args):
        """Route a tool call: to the remote executor when connected (with a
        local confirmation first, since the executor can't prompt), else local.
        The model is unaware of any of this - identical surface either way."""
        if name == "request_handover":       # elective soft-zone handover (agent-internal)
            return self._request_handover()
        if name == "update_plan":            # pinned-plan upkeep (local agent state, never remote)
            return self._update_plan(args.get("plan", ""))
        if name == "note":                   # session notebook (local agent state, never remote)
            return self._note(args)
        is_mcp = name.startswith(MCP_NS)
        plug = self.lean_tools.get(name)
        safe = bool(plug and plug["safe"])
        blocked = self._gate_tool(name, args, plug, safe, is_mcp)
        if blocked is not None:
            return blocked
        if self.remote and (name in EXEC_TOOLS or plug) and not (plug and plug.get("driver_only")):
            return self._route_remote(name, args, safe)
        # local: a non-safe lean-tool still confirms (core tools confirm in Tools)
        if plug and not safe and not auto_approved(self.cfg):
            print(yellow(f"\nlean-tool {name}({_fmt_args(args)})"))
            if not ask_action(self.cfg, f"run lean-tool {name}?"):
                return "operator declined the lean-tool"
        # MCP tools run on the DRIVER (never the remote executor) and may have side
        # effects, so they confirm like a non-safe lean-tool unless approval is armed.
        if is_mcp and not auto_approved(self.cfg):
            print(yellow(f"\nMCP {name}({_fmt_args(args)})"))
            if not ask_action(self.cfg, f"run MCP tool {name}?"):
                return "operator declined the MCP tool"
        return self._dispatch(name, args)

    def _parallel_safe(self, call) -> bool:
        """True iff this call is a read-only tool with no side effect and no
        confirm prompt, so it may run concurrently with its batch siblings. Core
        read tools qualify; a lean-tool qualifies only if it declares safe=True.
        Writers, run_command, and ask_user_to_run never do."""
        # ask-on-read serializes reads (each needs its own confirm), so no fast-path
        # unless approval is auto/armed (then reads don't prompt anyway).
        if self.cfg.confirm_reads and not auto_approved(self.cfg):
            return False
        name = call.get("function", {}).get("name", "")
        if name in SAFE_TOOLS:
            return True
        plug = self.lean_tools.get(name)
        return bool(plug and plug["safe"])

    def _run_parallel(self, calls):
        """Run a batch of read-only tool calls on daemon threads, then append
        their results in call order (the model pairs results to calls
        positionally, so execution order is free but append order is not).
        Callers guarantee every call is _parallel_safe and that we are local."""
        parsed = []
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            cid = self.record_tool_call(name, args)
            print(_tool_call_line(name, args, cid))
            self._maybe_auto_expand(name, cid)
            parsed.append((name, args, cid, call.get("id", "")))
        results = [None] * len(parsed)

        def work(idx, name, args):
            results[idx] = self._run_tool(name, args)  # already exception-guarded

        spin = Spinner(f"running {len(parsed)} in parallel", TOOL_FRAMES,
                       cyan, interval=0.12).start()
        try:
            threads = [threading.Thread(target=work, args=(i, n, a), daemon=True)
                       for i, (n, a, _cid, _tcid) in enumerate(parsed)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            spin.stop()
        for (name, _args, cid, tcid), result in zip(parsed, results):
            self.record_tool_result(cid, result)
            if _TTY:
                print(_tool_result_preview(name, result, cid))
            self.messages.append(_tool_result_msg(name, result, tool_call_id=tcid))

    @contextmanager
    def _sigint_guard(self):
        """Install a clean SIGINT handler for the duration of a turn: ^C stops
        inference and drops back to the prompt without a traceback, session
        intact. Degrades to cooperative abort if not on the main thread."""
        self._abort = False
        prev, installed = None, False

        def handler(signum, frame):
            self._abort = True
            raise _TurnAbort()

        try:
            prev = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handler)
            installed = True
        except (ValueError, OSError):
            pass  # non-main thread / unsupported - cooperative check still works
        try:
            yield
        except _TurnAbort:
            print(yellow("\n^C - stopped; back to prompt"))
        finally:
            if installed:
                signal.signal(signal.SIGINT, prev)

    def _system(self) -> str:
        # NOTE: the leash/permissions note is deliberately NOT here. It varies with the
        # leash, and the system block is a global cache breakpoint - baking it in rewrote
        # that cache on every /leash toggle. Instead the tool surface reflects the tier and
        # _leash_note() is injected into the next user turn on change (see run_turn).
        # NOTE: the run_command timeout/'&' note and the incognito mode note used to live
        # here. Removed: models don't read/act on the bg note (it duplicates the tool
        # schema), and neither is worth a line in the cache-pinned block.
        # NOTE: the pinned plan is deliberately NOT here. It's the goal+TODO the model
        # rewrites often (~every turn on an active task), and the system block is a cache
        # breakpoint - baking a volatile block in would rewrite system + everything after
        # it (summary + tail) at 2x on every plan edit. Instead it rides an uncached
        # send-time tail after the last breakpoint (see _plan_reminder / the send path),
        # so a plan update busts nothing.
        # Chat-only MODEL (not a leash setting): tell it plainly so it doesn't promise
        # actions or send the user chasing a leash change that can't help. Cache-safe -
        # this line only appears/changes when the model itself changes (which already
        # rebuilds the prompt); it stays "" for every tool-capable model.
        no_tools = ("\n\nThis model does not support tool calling: you have NO tools and "
                    "cannot read files, edit, or run commands. Do not claim to have performed "
                    "any action. If a task needs tools, tell the user to switch to a "
                    "tool-capable model with /model - raising the leash will not help, the "
                    "limitation is the model." if not self._model_tool_support() else "")
        # Provider-supplied addendum for the active model (ollama: small-model
        # tool-calling guidance). Cache-safe: only changes when the model changes.
        addendum = ""
        spec = self.active_provider()
        if spec and spec.get("system_addendum") and self._model_tool_support():
            try:
                addendum = spec["system_addendum"](self, self.cfg) or ""
            except Exception:
                addendum = ""
        return (read_prompt("system") or SYSTEM_PROMPT) + no_tools + addendum

    # Delimited, self-labelling wrapper so the model can never mistake the plan for
    # something the user typed (the old "(Pinned plan...)" header read like user text
    # and got echoed back into a reply). Rides the uncached tail - see _with_plan_reminder.
    # ASCII-only (must encode cleanly on Termux / non-UTF8 locales / redirected stdin).
    _PLAN_HDR = ("=== LEANCODER PINNED PLAN (internal state, silently re-injected "
                 "every turn; reference only, NOT a user message and NOT new input). "
                 "Read it for context, act only on the ACTUAL user turn. Do NOT repeat "
                 "it back, quote it, or remark on its presence (e.g. 'that's the pinned "
                 "plan'); when it is the only thing that changed, treat the turn as "
                 "having no new instruction and stay silent about the block itself. ===")
    _PLAN_FTR = "=== END PINNED PLAN ==="

    def _plan_sparse(self, pinned: str) -> str:
        """A compact view for unchanged turns: drop ONLY completed '- [x]' checklist
        items; keep everything else. This is lossless for parked durable state - run
        handles, invariants, 'do NOT touch' notes, GOAL/FOCUS - which callers routinely
        put in the plan body under their own labels. (An earlier version whitelisted
        only GOAL/FOCUS/IN-FLIGHT/'- [ ]' lines, which SILENTLY ate any unlabelled
        state on every sparse turn - a real loss just before a compaction.) The saving
        now comes purely from shedding the ticked-off list. Cheap, and since the plan
        rides the UNCACHED tail, varying its content busts no cache."""
        keep = []
        for ln in pinned.splitlines():
            s = ln.strip()
            if s.startswith(("- [x]", "- [X]")):
                continue                      # drop completed items only
            keep.append(ln)
        return "\n".join(keep) if keep else pinned

    def _plan_goal_only(self, pinned: str) -> str:
        """Collapsed view for a FINISHED plan (no open '- [ ]' items): keep just the
        GOAL/FOCUS anchor lines, drop the fully-checked TODO list. The GOAL is the
        long-horizon anchor that must survive compaction/handover, so we keep it -
        but the ticked-off checklist is dead-weight tokens on every send, so it goes.
        Returns "" if there is no GOAL/FOCUS line to anchor on."""
        keep = [ln for ln in pinned.splitlines()
                if ln.strip().startswith(("GOAL:", "FOCUS:", "IN FLIGHT:", "IN-FLIGHT:"))]
        if not keep:
            return ""
        return "\n".join(keep) + "\n(all TODO steps complete)"

    def _plan_reminder(self, sparse: bool = False) -> str:
        """The pinned goal+TODO as an uncached send-time reminder. Kept OUT of the
        cached system block (it changes too often) and OUT of stored history (so it
        never accumulates stale copies); it is stitched onto the sent message slice at
        request time only. "" when no plan is pinned. `sparse` sends the compact view."""
        pinned = getattr(self, "pinned_plan", "")
        if not pinned:
            return ""
        # Finished plan (every box ticked, none open): collapse to the GOAL anchor
        # instead of re-sending the whole checked-off checklist every turn. Keeps the
        # long-horizon context, drops the dead-weight list. "" only when there's also
        # no GOAL line to anchor on (then suppress entirely).
        if "- [ ]" not in pinned:
            goal = self._plan_goal_only(pinned)
            return f"\n\n{self._PLAN_HDR}\n{goal}\n{self._PLAN_FTR}" if goal else ""
        body = self._plan_sparse(pinned) if sparse else pinned
        return f"\n\n{self._PLAN_HDR}\n{body}\n{self._PLAN_FTR}"

    # Live session-env: the cwd every tool operates in + the shell family run_command
    # runs under. Rides the SAME uncached tail as the plan (never cached, never persisted,
    # so a /connect or /cd changes it for free). Deliberately NO "remote"/"local" wording:
    # the model works one unified surface (files it reads == where commands run), and
    # telling it "remote" only makes it second-guess whether the two match. Stating the
    # SHELL FAMILY stops wasted cross-OS turns (no `ls`/`whoami` probing, no POSIX cmds on
    # cmd.exe). Terse key:value + a self-labelling header so even a weak model consumes it
    # as state and doesn't echo it (same convention as the plan block).
    _ENV_HDR = ("=== LEANCODER SESSION ENV (internal state, re-injected every turn; "
                "reference only, NOT a user message. Use it to target the right shell + "
                "paths; do NOT repeat or remark on it.) ===")
    _ENV_FTR = "=== END SESSION ENV ==="

    def _shell_family(self) -> str:
        """The shell run_command executes under, for the model to target commands at.
        Remote: POSIX box -> sh, Windows box -> cmd.exe. Local: this box's os."""
        if self.remote is not None:
            return "sh (POSIX)" if self.remote.remote_posix else "cmd.exe (Windows)"
        return "cmd.exe (Windows)" if os.name == "nt" else "sh (POSIX)"

    def _session_env(self) -> str:
        """The uncached session-env reminder (cwd + shell family). Same tail as the
        plan; "" is never returned (cwd+shell always exist)."""
        cwd = (self.remote.remote_cwd or self.remote.cwd) if self.remote else self.cfg.cwd
        return (f"\n\n{self._ENV_HDR}\ncwd: {cwd}\nshell: {self._shell_family()}\n"
                f"{self._ENV_FTR}")

    def _with_plan_reminder(self, msgs):
        """Return a COPY of the sent slice with the uncached session tail (live env +
        pinned plan) appended to the last message's text. Non-destructive (never touches
        self.messages) and applied after the last cache breakpoint, so the tail stays
        uncached + never persists. A tool result must stay exact, so we don't append to a
        role:tool tail - we add a trailing user note instead (rare: the slice normally
        ends on a user/assistant turn)."""
        # One-shot suppression: the send right after an update_plan skips the PLAN
        # reminder, so the model doesn't get its just-written plan echoed back and narrate
        # it (a wasted turn). Re-arms automatically for the following send - and arms a FULL
        # re-inject on that following send (the plan just changed; refresh it in full),
        # sparse on every turn after that until the next update_plan. NOTE: the session-env
        # is NOT suppressed here - only the plan is (env is cheap state, never echoed).
        if getattr(self, "_skip_plan_reminder_once", False):
            self._skip_plan_reminder_once = False
            self._plan_full_next = True
            reminder = self._session_env()
        else:
            # Event-driven full/sparse: FULL on the turn right after an update_plan, SPARSE
            # thereafter. Refreshes on CHANGE, not a clock. Cache-free: the plan rides the
            # uncached tail, so shrinking it busts nothing (see _with_plan_reminder docstring).
            full = getattr(self, "_plan_full_next", True)
            self._plan_full_next = False
            # The live session-env ALWAYS rides the tail (cwd + shell family); the plan
            # rides it only when pinned. Env first so the plan (the richer block) sits last
            # = most recent = highest attention.
            reminder = self._session_env() + self._plan_reminder(sparse=not full)
        if not reminder or not msgs:
            return msgs
        out = list(msgs)
        last = dict(out[-1])
        if last.get("role") == "tool":
            out.append({"role": "user", "content": reminder.lstrip("\n")})
            return out
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = content + reminder
        elif isinstance(content, list) and content and content[-1].get("type") == "text":
            blocks = list(content)
            blocks[-1] = dict(blocks[-1], text=blocks[-1].get("text", "") + reminder)
            last["content"] = blocks
        elif isinstance(content, list):
            last["content"] = content + [{"type": "text", "text": reminder.lstrip("\n")}]
        else:
            return msgs
        out[-1] = last
        return out

    def _bg_bump_session(self):
        """Attend this session's leased bg tasks so their watchdogs keep them alive.
        Called once per turn: while we live we bump each leased task's lease file;
        when we're gone the leases go stale and the tasks self-terminate. LOCAL only -
        a remote task's lease lives on the executor box, bumped there (the remote
        keeps its own tasks attended); a foreign-host record is skipped by
        _bg_bump_lease. Best-effort, never raises into the turn."""
        try:
            me = os.getpid()
            for r in _bg_load():
                if r.get("owner") == me and r.get("idle_timeout"):
                    _bg_bump_lease(r)
        except Exception:
            pass

    def _bg_finished_note(self, only_notify=False) -> str:
        """A one-time notice for background tasks THIS session started that have
        finished since we last looked - so a long-running job the model fired and
        moved on from surfaces its result ('job done, exit N, last lines') on the
        next turn instead of needing a manual /bg poll. Works both LOCAL (scan our
        own sidecars) and REMOTE (ask the executor via BG_POLL - the sidecars live
        on that box). Reported once per pid, keyed by (host,pid) in _bg_announced
        so a local and a remote task can't collide on a bare pid.

        Also surfaces WATCHDOG ALERTS for still-running jobs (LOCAL only): a heartbeat
        silence ('.silent') or a notify-only max_runtime ('.maxrun') trip. These are
        opt-in (only exist if the job set those args), fire while the job is ALIVE, and
        re-arm - so they dedup by unlinking the sidecar after reporting (not _bg_announced).

        `only_notify` (for the wake path when global wake is OFF): restrict the FINISHED
        notices to jobs that set notify_on_exit. Alerts are always included when set
        (they're an explicit per-job opt-in). "" when nothing new to report."""
        remote = getattr(self, "remote", None)
        if remote is not None:
            host = remote.host
            items = remote.bg_poll()
            alerts = []                       # remote watchdog alerts: follow-up
        else:
            host = _HOSTNAME
            items = _bg_finished_here(os.getpid())
            alerts = _bg_alerts_here(os.getpid())
        lines = []
        for it in items:
            key = (host, it.get("pid"))
            if key in self._bg_announced:
                continue
            if only_notify and not it.get("notify_on_exit"):
                continue
            self._bg_announced.add(key)
            lines.append(_bg_finished_msg([it]))
        for al in alerts:
            lines.append(_bg_alert_msg([al]))
            _bg_clear_alert(al.get("log"), al.get("kind"))
        return "\n\n".join(l for l in lines if l)

    def _has_bg_optins(self) -> bool:
        """True if any live bg job THIS session owns needs the idle wake path armed:
          - a plain TASK that opted into a per-job push (notify_on_exit /
            heartbeat_timeout / max_runtime), OR
          - ANY live WORKER. A worker ALWAYS needs the idle path: its finish wakes the
            agent via the wake-hook registry, AND - critically - the idle path is where
            its LEASE gets bumped while the parent sits idle (see bg_wake_turn). Without
            counting workers here, a parent with only workers live never arms the wake
            path, never bumps the lease, and the worker self-terminates (exit 137) ~30min
            into an AFK stretch - the 'worker died with no output' bug.
        Lets the wake path stay armed even when global wake_on_bg_finish is off - cheap
        registry read, best-effort."""
        try:
            me = os.getpid()
            for r in _bg_load():
                if r.get("owner") != me:
                    continue
                if r.get("kind", "task") == "worker":
                    return True
                if (r.get("notify_on_exit") or r.get("heartbeat_timeout")
                        or r.get("max_runtime")):
                    return True
        except Exception:
            pass
        return False

    def bg_wake_turn(self):
        """AUTONOMY: a synthesised user turn describing any background job THIS session
        started that has finished / tripped a watchdog since we last looked - or "" if
        none. Aggregates two sources, both using their OWN dedup so a job is claimed by
        whichever fires first (this wake path when idle, or the passive turn-rider when
        the operator types):
          1. plain bg TASKS + watchdog alerts via _bg_finished_note;
          2. WORKERS via the wake-hook registry (_WAKE_HOOKS) - the dispatch_worker
             lean-tool registers its _finished_notice there, so worker finishes wake
             the agent too. Workers are hidden from the core bg notify (kind filter),
             so the hook is their only wake path; its own meta['announced'] dedup means
             a worker surfaced by wake won't re-report on the next real turn.
        When global cfg.wake_on_bg_finish is ON, every finished plain task wakes. When
        it's OFF this may still fire for a job that opted in per-job (notify_on_exit /
        heartbeat_timeout / max_runtime) - only_notify restricts the finished notices to
        those. Wrapped in autonomous-wake framing + an explicit react instruction, since
        no operator prompt accompanies it."""
        # Attend our leased bg tasks/workers HERE too, not just in run_turn: a parent
        # sitting IDLE at the prompt (operator AFK) is still alive, but run_turn - the
        # only other lease-bump site - doesn't fire without a turn. Without this, a
        # dispatched worker (idle_timeout=1800) self-terminates ~30min into an AFK stretch
        # with exit 137 = the "worker died with no output" bug. This wake_check runs every
        # idle tick (~0.25s); throttle the bump to once per ~30s so it's ~free.
        now = time.monotonic()
        if now - getattr(self, "_last_idle_lease_bump", 0.0) >= 30:
            self._last_idle_lease_bump = now
            self._bg_bump_session()
        global_wake = bool(getattr(self.cfg, "wake_on_bg_finish", False))
        parts = []
        note = self._bg_finished_note(only_notify=not global_wake)
        if note:
            parts.append(note)
        for hook in list(_WAKE_HOOKS):
            try:
                h = hook()
            except Exception:
                h = ""
            if h:
                parts.append(h)
        if not parts:
            return ""
        return ("[autonomous wake - a background job you started finished; no operator "
                "input. React to the result now: inspect it, decide the next step, and "
                "either act or hand back.]\n\n" + "\n\n".join(parts))

    def reset(self):
        self.messages = [{"role": "system", "content": self._system()}]
        self.last_prompt_tokens = None
        self.session_in = self.session_out = 0
        self.tools.changed_files.clear()
        self.dirty = False
        # A fresh session starts with no goal. The pinned plan is separate agent
        # state (not in self.messages), so it would otherwise survive /new and
        # mislead the next task with a stale GOAL. /load restores its own plan.
        self.pinned_plan = ""
        # Drop the /expand ring too: its calls (+ captured output) belong to the
        # session we're leaving. Keeping them would linger in memory and let
        # /expand N surface a stale call whose output already scrolled off.
        self._tool_calls, self._tool_call_seq = [], 0

    def restore(self, messages):
        """Replace the live conversation with a loaded session. The stored system
        prompt is dropped and rebuilt for the *current* context (cwd / remote
        state may differ from when it was saved), and the cached token count is
        cleared so the meter re-estimates instead of showing the old session's."""
        # Defensive: a session file could carry a non-list, or a non-dict entry (hand
        # edit, partial write, foreign format). Keep only real dict messages with a
        # role, so one bad entry can't crash the whole resume via .get on a non-dict.
        if not isinstance(messages, list):
            messages = []
        body = [m for m in messages
                if isinstance(m, dict) and m.get("role") and m.get("role") != "system"]
        self.messages = [{"role": "system", "content": self._system()}] + body
        self.last_prompt_tokens = None
        self.last_prompt_tokens = None
        self.tools.changed_files.clear()
        self.dirty = False               # now showing a saved session, not unsaved work
        # The /expand ring is not persisted, so a loaded session brings no tool
        # calls with it - clear the outgoing session's so /expand N can't surface
        # a stale call from the conversation we just replaced.
        self._tool_calls, self._tool_call_seq = [], 0
        return len(body)

    def _trim_tool_indices(self, indices):
        """Stub the tool messages at `indices` to short placeholders. Skips any
        already-stubbed message. Returns (trimmed_count, tokens_freed)."""
        trimmed, freed = 0, 0
        for i in indices:
            m = self.messages[i]
            # Content is normally a string, but a loaded/older/hand-edited session
            # could carry a non-string (or missing) tool content. Coerce defensively
            # so compaction (which auto-fires under handover) can never crash the
            # whole turn on one odd message.
            content = m.get("content")
            if not isinstance(content, str):
                content = "" if content is None else str(content)
                m["content"] = content
            if content.startswith("[trimmed"):
                continue
            n = len(content.splitlines())
            stub = f"[trimmed {m.get('tool_name', 'tool')} result - {n} lines]"
            freed += est_tokens(content) - est_tokens(stub)
            m["content"] = stub
            # An evicted result drops its image too - the big cost. The file
            # stays on disk (path is in the stubbed text), so it can be re-fetched
            # if needed; we just stop re-sending the base64 every round.
            if "image_path" in m:
                m["image_path"] = None
                m.pop("image_path")
            trimmed += 1
        if trimmed:
            # The cached actual prompt_eval_count is now stale (it reflects the
            # pre-trim prompt). Clear it so the meter shows the trimmed estimate
            # instead of a misleading unchanged figure until the next model turn.
            self.last_prompt_tokens = None
            # This is the one in-place content mutation (stubs keep list length), so
            # the (id,len)-keyed message-token cache would go stale - bust it here.
            _invalidate_msg_tokens()
        return trimmed, freed

    def compact(self, keep: int = COMPACT_KEEP):
        """Trim old tool results to short stubs, keeping the last `keep` in
        full. The cheapest big context win - old file dumps/command output are
        rarely needed once acted on. Returns (trimmed_count, tokens_freed)."""
        tool_idx = [i for i, m in enumerate(self.messages)
                    if m["role"] == "tool"]
        to_trim = tool_idx[:-keep] if keep > 0 else tool_idx
        return self._trim_tool_indices(to_trim)

    def _window_messages(self, msgs, max_msgs):
        """Return the system message plus a bounded tail of `msgs`, cut at a clean
        turn boundary (the first user message in range) so a tool result is never
        orphaned from its tool_call. Non-destructive: `msgs` itself is untouched -
        this only shapes what a single request SENDS. max_msgs <= 0, or a history
        already within the bound, returns `msgs` unchanged. If no clean boundary
        falls within range, returns `msgs` unchanged (skip windowing this round
        rather than risk an orphaned/!user-leading send). See
        docs/cost-reduction-roadmap.md.

        A TOKEN budget (cfg.window_tokens) takes precedence over the message-count bound:
        it keeps as many whole recent turns as fit under the token ceiling. 'auto' resolves
        to the detected context window minus a reply reserve (see _resolve_window_tokens),
        so it's a pure overflow backstop. See _window_by_tokens."""
        tok_budget = self._resolve_window_tokens()
        if tok_budget > 0:
            return self._window_by_tokens(msgs, tok_budget)
        start = self._keep_tail_start(msgs, max_msgs)
        if start is None:
            return msgs
        head = msgs[:1] if msgs and msgs[0].get("role") == "system" else []
        return head + msgs[start:]

    # Reply reserve for 'auto': room the model needs to GENERATE its answer, kept out of
    # the prompt budget so an auto-window never fills the whole context and starves output
    # (on ollama num_ctx is shared prompt+generation; a 100% prompt = no room to reply).
    # max(fixed floor, fraction of the window) so it scales from a 4k model up to a 200k one.
    _AUTO_WINDOW_RESERVE = 1024
    _AUTO_WINDOW_RESERVE_FRAC = 0.15

    def _resolve_window_tokens(self) -> int:
        """The numeric per-send token budget from cfg.window_tokens: a positive int is used
        as-is (hard cap); 'auto' -> detected ctx_window minus a reply reserve (the overflow
        backstop); 0 / off / anything else -> 0 (windowing off). Best-effort: never raises."""
        cfg = getattr(self, "cfg", None)
        raw = getattr(cfg, "window_tokens", 0)
        if isinstance(raw, str):
            if raw.strip().lower() != "auto":
                return 0
            try:
                ctx = int(cfg.ctx_window())
            except Exception:
                return 0
            if ctx <= 0:
                return 0
            reserve = max(self._AUTO_WINDOW_RESERVE, int(ctx * self._AUTO_WINDOW_RESERVE_FRAC))
            return max(0, ctx - reserve)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _window_by_tokens(self, msgs, budget):
        """System message + the largest clean-boundary TAIL of `msgs` that keeps the whole
        send at/under `budget` CALIBRATED tokens (messages_tokens - the same fixed meter the
        ctx line/handover use, so the bound matches reality). ALWAYS kept + counted first:
        the system message, the tool schemas, and a reserve for the env/plan tail that
        _with_plan_reminder appends AFTER this (so the real send never blows past budget).
        The conversation tail is then grown one WHOLE turn at a time (each user boundary)
        from newest back until the next older turn wouldn't fit, cut at that user boundary
        so a tool result is never orphaned. Non-destructive. Returns `msgs` unchanged when
        it already fits, or when even the last turn alone can't fit (we never send an empty
        or orphaned slice - overflow is then the model/provider's own truncation to handle,
        exactly as with no window)."""
        head = msgs[:1] if msgs and msgs[0].get("role") == "system" else []
        body = msgs[len(head):]
        if not body:
            return msgs
        # Fixed overhead that is ALWAYS on the wire: system + tools + the env/plan tail
        # reserve. Count them against the budget up front so the conversation only gets
        # what's left. messages_tokens is the calibrated meter (never the raw char count).
        reserve = head + [{"role": "user", "content": self._reminder_probe()}]
        overhead = messages_tokens(reserve, self.tool_defs)
        remaining = budget - overhead
        if remaining <= 0:
            # Budget can't even hold the fixed surface; keep the last turn so we still send
            # something coherent (provider truncates if it must) rather than an empty body.
            start = self._first_user_from(body, len(body) - 1)
            return (head + body[start:]) if start is not None else msgs
        # Grow the tail newest-first, snapping each candidate start to a user boundary, and
        # keep the largest slice that fits. Walk user boundaries from the end backwards.
        best = None
        for i in range(len(body) - 1, -1, -1):
            if body[i].get("role") != "user":
                continue
            cand = body[i:]
            if messages_tokens(cand, None) <= remaining:
                best = i
            else:
                break                          # older turns only get bigger; stop
        if best is None:
            # Not even the last turn fits the remaining budget: send the last turn anyway
            # (coherent over empty); the provider handles the overflow like the no-window case.
            start = self._first_user_from(body, len(body) - 1)
            return (head + body[start:]) if start is not None else msgs
        if best == 0:
            return msgs                         # whole history fits -> unchanged
        return head + body[best:]

    def _first_user_from(self, body, from_idx):
        """Index of the nearest user message at or before `from_idx` in `body` (so a slice
        starts on a clean turn), or None if there is none."""
        for i in range(min(from_idx, len(body) - 1), -1, -1):
            if body[i].get("role") == "user":
                return i
        return None

    def _reminder_probe(self):
        """A worst-case-ish stand-in for the env/plan tail _with_plan_reminder appends after
        windowing, so _window_by_tokens can RESERVE room for it. Uses the live env + a FULL
        (non-sparse) plan reminder - the largest form - so the reserve never under-counts.
        Best-effort: any failure returns '' (reserve just 0, still safe-ish)."""
        try:
            return self._session_env() + self._plan_reminder(sparse=False)
        except Exception:
            return ""

    def _keep_tail_start(self, msgs, max_msgs):
        """Index into `msgs` where the kept tail begins: the first user-message
        boundary within the last `max_msgs` (so a tool_result is never orphaned from
        its tool_call, and the kept slice starts a clean turn). None when no windowing
        applies - disabled, history already within bound, or no clean boundary in
        range. Shared by the send-window and by compaction (what to keep vs summarize)."""
        if max_msgs <= 0 or len(msgs) <= max_msgs:
            return None
        head = 1 if (msgs and msgs[0].get("role") == "system") else 0
        tail = msgs[head:][-max_msgs:]
        base = len(msgs) - len(tail)          # index in msgs of tail[0]
        for i, m in enumerate(tail):
            if m.get("role") == "user":
                return base + i
        return None

    def _auto_evict(self):
        """Continuous tool-output eviction - the main quota lever. Each agentic
        round, stub the bodies of tool results the model has ALREADY responded to,
        keeping the last `auto_evict_keep` of them in full. The in-flight batch
        (tool results AFTER the last assistant message - what the model is about to
        read this round) is never touched, so a parallel read batch stays intact
        until it has been acted on. Quota counts raw tokens and the full history is
        re-sent every round, so shrinking acted-on results shrinks every later call.
        See docs/cost-reduction-roadmap.md. Returns (trimmed_count, tokens_freed)."""
        keep = self.cfg.auto_evict_keep
        last_asst = max((i for i, m in enumerate(self.messages)
                         if m["role"] == "assistant"), default=-1)
        acted = [i for i, m in enumerate(self.messages)
                 if m["role"] == "tool" and i < last_asst]
        to_trim = acted[:-keep] if keep > 0 else acted
        return self._trim_tool_indices(to_trim)

    def _handover_turn_text(self):
        """All assistant content produced since the handover instruction was injected
        (the marked block lives here, possibly alongside the finalize tool call)."""
        return "\n".join(m.get("content") or "" for m in self.messages[self._handover_mark:]
                         if m.get("role") == "assistant")

    def _solicit_handover(self, instr):
        """Shared core of both compaction paths: inject the handover instruction `instr`,
        give the model a turn to write the marked blocks (@#! summary + plan + next-step),
        and - since there is no finalize tool, the turn ENDING is the trigger - tolerantly
        parse it, re-soliciting any MISSING block BY NAME up to _HANDOVER_MAX_TRIES.
        Returns (parsed, missing, attempt, turn_text). ONE place so handover() and
        auto_handover() can't drift (they did: one used the tolerant parse, one a strict
        re-extract, silently dropping the salvaged plan/next). Callers decide what to do
        with the result (replace history now, snapshot first, autostart, etc.)."""
        self.messages.append({"role": "user", "content": instr})
        self.dirty = True
        self._handover_capture = None
        self._handover_mark = len(self.messages)
        parsed, turn_text = None, ""
        missing = ["handover"]
        attempt = 0
        for attempt in range(_HANDOVER_MAX_TRIES):
            if attempt > 0:
                nudge = (_handover_nudge_missing(missing) if attempt == 1
                         else HANDOVER_RETRY_2)   # last try: the exact full template
                self.messages.append({"role": "user", "content": nudge})
            with self._sigint_guard():
                self._loop()
            turn_text = self._handover_turn_text()
            parsed = _parse_handover(turn_text)
            missing = _handover_missing(parsed)
            if not missing:
                break
        return parsed, missing, attempt, turn_text

    def _finish_handover(self, parsed, *, summary_prefix, tail=None,
                         autostart=False, stamp_clock=False):
        """Shared epilogue for handover() + auto_handover(): once the marked blocks
        are parsed, install the pinned plan, rebuild history as [system, summary(+tail)],
        set the stable cache boundary, and reset the per-turn counters. The two callers
        differ only in the four knobs:
          - summary_prefix: how the compacted summary is introduced;
          - tail: recent verbatim messages kept AFTER the summary (auto only; None = a
            clean [system, summary] cut for an operator-driven /handover);
          - autostart: queue the model's next-step self-prompt (auto only - nobody's
            watching to type it);
          - stamp_clock: bump the anti-thrash loop guard (auto only).
        Returns the handover text block."""
        block = parsed["handover"]
        if parsed["plan"]:
            self.pinned_plan = parsed["plan"]   # survives handover (uncached send-time reminder)
        self.messages = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": summary_prefix + block, "cache_boundary": True},
        ] + (tail or [])
        # Cache-boundary signal: after a handover the prefix [system][summary] is STABLE,
        # so a client with a spare breakpoint can pin it and re-read the prefix at ~0.1x.
        # 2 = system + the one compacted-summary user message.
        self._handover_boundary = 2
        self.last_prompt_tokens = None
        self.tools.changed_files.clear()
        if stamp_clock:
            self._last_compact_ts = time.time()
        if autostart and parsed["next"] and self.cfg.handover_for()["autostart"]:
            self._autostart_pending = parsed["next"]
        self.handovers += 1
        return block

    def handover(self):
        """Agentic /handover: the model gets a real (tool-capable) turn to update +
        commit any durable docs, then write its future-self handover as marked text
        (@#! block + plan + next-step). No finalize tool - the turn ending is the
        trigger; we parse the marked blocks, re-solicit any missing one, then replace
        history with the summary - a "smart /clear" that keeps the thread. A bad/absent
        block does NOT nuke history (returns None). Returns the handover text, or None."""
        if len([m for m in self.messages if m["role"] != "system"]) == 0:
            return None
        instr = read_prompt("handover") or HANDOVER_INSTR
        parsed, missing, attempt, _turn_text = self._solicit_handover(instr)
        if not parsed["handover"]:   # no usable handover (truly empty) -> leave history untouched
            return None
        if parsed["tier"] >= 3 or missing:
            self._log_activity("handover",
                               f"degraded parse (tier {parsed['tier']}, missing {missing})",
                               "model dropped marker discipline; salvaged via tolerant parser")
        return self._finish_handover(
            parsed, summary_prefix="Session handover (prior context summarized):\n")

    def _update_plan(self, plan: str):
        """The update_plan tool body: replace the pinned GOAL + TODO and refresh the
        live system prompt so the change is visible from the very next turn (and rides
        compaction/handover). Cheap + cache-stable: the plan only moves when rewritten."""
        plan = (plan or "").strip()
        self.pinned_plan = plan
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self._system()
        self.last_prompt_tokens = None       # prompt changed; let the meter re-estimate
        # Suppress the plan reminder on the NEXT send only: the model just wrote the plan
        # (this call), so re-appending the full plan onto the update_plan tool result makes
        # it echo/narrate it back - a wasted turn. It re-arms on the following send.
        self._skip_plan_reminder_once = True
        if not plan:
            return "pinned plan cleared."
        done = plan.count("[x]") + plan.count("[X]")
        todo = plan.count("[ ]")
        return f"plan pinned ({done} done, {todo} to go)."

    # Session notebook (the note tool). Notes are {dtg, text} dicts kept on self.notes,
    # persisted in the session JSON so they travel 1:1 with /load and a pushed session.
    # A fixed read cap keeps any recall from flooding context; the spool cap
    # (cfg.notes_spool) trims the oldest on write so the log can't grow unbounded.
    _NOTES_READ_CAP = 400                     # max LINES any read (recent/grep/range) returns

    @staticmethod
    def _note_lines(entries):
        """Render entries as '- [dtg] text' lines (text may itself be multi-line)."""
        return [f"- [{e.get('dtg', '?')}] {e.get('text', '')}" for e in entries]

    def _note_cap(self, entries, label):
        """Cap a rendered read to _NOTES_READ_CAP LINES, keeping the MOST RECENT. Returns
        the capped string with an elision note when it overflowed."""
        lines = self._note_lines(entries)
        total = len(lines)
        if total <= self._NOTES_READ_CAP:
            return "\n".join(lines) if lines else f"no notes {label}."
        kept = lines[-self._NOTES_READ_CAP:]
        return (f"[showing the latest {self._NOTES_READ_CAP} of {total} lines {label}; "
                f"narrow with grep/range]\n" + "\n".join(kept))

    def _note(self, args):
        """The note tool body. action=add (default when text given) appends a DTG-stamped
        entry and spool-trims; recent/grep/range are capped reads. Pure agent state - never
        touches the filesystem or the remote."""
        action = (args.get("action") or "").strip().lower()
        text = (args.get("text") or "").strip()
        if not action:
            action = "add" if text else "recent"
        if action == "add":
            if not text:
                return "error: nothing to note (pass text)."
            dtg = time.strftime("%Y-%m-%d %H:%M")
            self.notes.append({"dtg": dtg, "text": text})
            # spool-trim to the newest notes_spool LINES (an entry may span several lines)
            cap = max(1, int(getattr(self.cfg, "notes_spool", 2000) or 2000))
            lines = 0
            keep_from = len(self.notes)
            for i in range(len(self.notes) - 1, -1, -1):
                lines += (self.notes[i].get("text", "").count("\n") + 2)  # text + dtg line
                if lines > cap:
                    break
                keep_from = i
            trimmed = keep_from
            if trimmed:
                self.notes = self.notes[trimmed:]
            n = len(self.notes)
            extra = f" ({trimmed} old trimmed)" if trimmed else ""
            return f"noted [{dtg}]{extra}. {n} entr{'y' if n == 1 else 'ies'} in the notebook."
        if not self.notes:
            return "the notebook is empty."
        if action == "recent":
            n = args.get("n")
            try:
                n = int(n) if n is not None else 40
            except (TypeError, ValueError):
                n = 40
            n = max(1, n)
            return self._note_cap(self.notes[-n:], "recent")
        if action == "grep":
            pat = args.get("pattern") or ""
            if not pat:
                return "error: grep needs a pattern."
            try:
                rx = re.compile(pat, re.IGNORECASE)
                match = lambda e: rx.search(f"[{e.get('dtg','')}] {e.get('text','')}")
            except re.error:
                match = lambda e: pat.lower() in f"[{e.get('dtg','')}] {e.get('text','')}".lower()
            hits = [e for e in self.notes if match(e)]
            if not hits:
                return f"no notes match {pat!r}."
            return self._note_cap(hits, f"matching {pat!r}")
        if action == "range":
            frm = (args.get("from") or "").strip()
            to = (args.get("to") or "").strip()
            if not frm:
                return "error: range needs a 'from' timestamp (YYYY-MM-DD [HH:MM])."
            # a bare date is inclusive of the whole day; pad to/from accordingly.
            lo = frm if len(frm) > 10 else frm + " 00:00"
            hi = (to if to and len(to) > 10 else (to + " 23:59") if to else "9999-12-31 23:59")
            sel = [e for e in self.notes if lo <= e.get("dtg", "") <= hi]
            if not sel:
                return f"no notes in range {lo} .. {hi}."
            return self._note_cap(sel, f"in {lo} .. {hi}")
        return f"error: unknown note action {action!r} (add|recent|grep|range)."

    def _request_handover(self):
        """The request_handover tool body: the model elects a handover at a clean break.
        We don't compact here (mid-turn, inside a tool batch) - we set a flag that
        _maybe_handover honours AFTER the turn, running the same auto_handover machinery.
        Guard: only honour it in the soft-or-higher zone. The tool is only offered in the
        soft zone, but a stale/replayed/hallucinated call could arrive below it - refuse
        those so an elected handover can never happen before the user's threshold (the
        model never gets to spend on a handover the user hasn't opened the window for)."""
        if not self.cfg.handover_for()["auto"]:
            return "handover is off (auto_handover disabled) - not handing over."
        if self._ctx_zone()[0] == "ok":
            return ("not yet: context is still light (below the handover threshold), so a "
                    "handover would just discard fidelity for no gain. Keep going - the "
                    "option reappears once context is heavier.")
        self._elected_handover = True
        return ("handover queued - finish this turn cleanly (no half-done work); the "
                "handover runs right after, and you'll write the summary + next-step then.")

    def auto_handover(self, zone: str = "hard"):
        """Self-managing compaction. Snapshot the full history (recoverable), then an
        agentic turn where the model summarizes the OLDER context (between
        HANDOVER_MARK markers) and writes its next-step self-prompt (between
        SELFPROMPT_MARK markers), then ends the turn (no finalize tool). We keep the recent tail
        verbatim, replace the older part with the summary, queue the self-prompt for
        autostart, and stamp the loop-guard clock. A bad/absent summary leaves history
        untouched (returns None). Returns the summary block, or None."""
        if not [m for m in self.messages if m["role"] != "system"]:
            return None
        pre       = list(self.messages)       # capture BEFORE the compaction turn
        pre_plan  = getattr(self, "pinned_plan", "")   # for the recovery snapshot below
        keep_from = self._keep_tail_start(pre, self.cfg.window_messages)
        tail      = pre[keep_from:] if keep_from is not None else []

        saved = self.tool_defs
        # Shared retry-parse core (see _solicit_handover): give the model a turn to write
        # the marked blocks, tolerantly parse, re-solicit any missing one. The completeness
        # retry (an incomplete-but-successful ANSWER) is a DIFFERENT layer from the send-
        # level overload/429 recovery inside _loop (a failed SEND): separate budgets.
        instr = read_prompt("auto_handover") or AUTO_HANDOVER_INSTR
        parsed, missing, attempt, _turn_text = self._solicit_handover(instr)
        block = parsed["handover"]
        self.tool_defs = saved
        if not block:                         # no usable summary -> don't nuke history
            self.messages = pre
            # Stamp the loop guard even on failure so a model that can't produce a
            # summary doesn't re-attempt (and re-snapshot) compaction every single turn.
            self._last_compact_ts = time.time()
            return None
        if parsed["tier"] >= 3 or attempt > 0:
            self._log_activity(
                "handover",
                f"degraded parse (tier {parsed['tier']}, {attempt} retr{'y' if attempt == 1 else 'ies'})",
                "model dropped marker discipline; salvaged via tolerant parser + retry")
        # Snapshot the PRE-HANDOVER history ONLY now that we have a usable summary and are
        # about to replace it - so it stays recoverable via /load (and greppable by the
        # recall path). Doing it here (not at the top) means a FAILED/futile handover
        # writes no snapshot: that unbounded per-turn snapshotting by a model that never
        # summarized is what flooded /load. Named <origin>-prehandover-<N> so the snapshot
        # is tied to the session it came from.
        try:
            rhost = self.remote.host if self.remote else None
            origin = getattr(self, "autosave_name", "") or "session"
            existing = [nm for nm, _ in list_sessions()]
            save_session(pre, self.cfg, _prehandover_name(origin, existing),
                         remote=rhost, pinned_plan=pre_plan)
        except Exception:
            pass
        # Consume the already-computed tolerant parse (from the retry loop above), NOT a
        # second strict fence-only re-extract: _parse_handover salvages plan/next from a
        # degraded tier (## Plan header, JSON field) too, and _handover_missing already
        # accepted that tier - so a strict _extract_* here would silently throw away the
        # salvage for exactly the models the tolerant parser exists to rescue, dropping
        # both the pinned plan and the autostart self-prompt. Same source as handover().
        # Auto differs from /handover only in the four knobs below: keep the recent
        # verbatim tail, queue the next-step self-prompt, and stamp the loop guard.
        return self._finish_handover(
            parsed, summary_prefix="Earlier context (compacted summary):\n",
            tail=tail, autostart=True, stamp_clock=True)

    def _dispatch(self, name: str, args: dict) -> str:
        if name.startswith(MCP_NS):
            return self.mcp.call(name, args)
        plug = self.lean_tools.get(name)
        if plug:
            # Publish the live context budget so a lean-tool can size its output to
            # the active model (e.g. web_fetch caps a read to the FREE window, and
            # warns + goes minimal when little is left). MAX = window, USED = current;
            # free/pct derive. A fresh module per load means a tool's own setup()
            # state wouldn't reach run(), so these env vars are the carry. Any tool
            # may read them. Not set on the remote executor (hermetic, no model).
            os.environ["LEANCODER_CTX_MAX"] = str(self.cfg.ctx_window())
            os.environ["LEANCODER_CTX_USED"] = str(self._ctx_used())
            try:
                out = plug["run"](args, str(self.cfg.cwd))
                # Image tool-result contract: a tool may return a dict
                # {text, image_path?} instead of a str. Preserve it here; the
                # append sites split it into content(+image_path). Anything else
                # is stringified (unchanged behaviour for the thousands of
                # str-returning tools).
                if isinstance(out, dict) and "text" in out:
                    return out
                return str(out)
            except Exception as e:
                return f"error in lean-tool {name}: {e}"
        fn = getattr(self.tools, name, None)
        if fn is None:
            return f"error: unknown tool {name}"
        try:
            return fn(**args)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"error in {name}: {e}"

    def _soft_nudge(self) -> str:
        """A throttled one-line context nudge while in the soft zone, injected into the
        user turn (uncached). Tells the model context is filling so it wraps up toward a
        clean stopping point before the hard auto-compaction. "" when not due."""
        if not self.cfg.handover_for()["auto"]:
            return ""
        zone, used, limit = self._ctx_zone()
        if zone != "soft":
            return ""
        if self._turn_count - self._last_nudge_turn < NUDGE_EVERY:
            return ""
        self._last_nudge_turn = self._turn_count
        pct = int(used / limit * 100) if limit else 0
        hard_frac = self.cfg.handover_for()["hard"]
        hard = int(limit * hard_frac) if limit else 0
        hard_pct = int(hard_frac * 100)
        tmpl = read_prompt("handover_nudge") or HANDOVER_NUDGE
        try:
            return tmpl.format(used=f"{used:,}", limit=f"{limit:,}", pct=pct,
                               hard=f"{hard:,}", hard_pct=hard_pct)
        except (KeyError, IndexError, ValueError):
            # a user-edited nudge with a bad placeholder must not break the turn
            return tmpl

    def run_turn(self, user_input: str):
        self._turn_count += 1
        # Rebuild the surface so the elective request_handover tool appears/disappears
        # as context crosses the soft threshold (offer_handover tracks the zone). Cheap;
        # keeps the offered tools honest to the current zone without a separate trigger.
        self.refresh_tools()
        notes = []
        if self._pending_ai_note:        # leash/permissions note (on change/startup)
            notes.append(self._pending_ai_note)
            self._pending_ai_note = None
        nudge = self._soft_nudge()       # throttled soft-zone context nudge
        if nudge:
            notes.append(nudge)
        if notes:                        # both ride the (uncached) user turn, never the prompt
            user_input = "(" + "  ".join(notes) + ")\n\n" + user_input
        self._bg_bump_session()              # attend our leased bg tasks (keeps them alive)
        bg_note = self._bg_finished_note()   # long-run bg tasks that finished since last turn
        if bg_note:
            user_input = user_input + "\n\n[" + bg_note + "]"
        self.messages.append({"role": "user", "content": user_input})
        self.tools.changed_files.clear()
        self._handover_capture = None    # clear any prior /handover trigger (loop guard)
        self._elected_handover = False   # clear any prior elective request (fires once)
        self.dirty = True                # conversation now has unsaved changes
        with self._sigint_guard():
            self._loop()
        # An interrupted tool batch (^C, crash) can leave a dangling tool_use; heal
        # it in place so the autosave + next turn aren't poisoned (see repair_tool_pairs).
        self.messages = repair_tool_pairs(self.messages)
        self._maybe_auto_compact()        # cheap programmatic strip at token intervals (first)
        self._maybe_handover()            # then the agentic handover (hard/emergency)

    def record_tool_call(self, name, args):
        """Store a tool call's FULL args in the ring and return its monotonic id.
        Called at print time (the instant the call starts, before it runs), so an
        in-progress call already has its number. The on-screen line truncates args
        to 50 chars; /expand id (or bare /expand = newest) re-renders the full args
        from here. Ring-buffered (cap 100) so it never grows unbounded; lazily inits
        so a test-double Agent that skipped __init__ can't crash on it."""
        if not hasattr(self, "_tool_calls") or self._tool_calls is None:
            self._tool_calls, self._tool_call_seq = [], 0
        self._tool_call_seq += 1
        self._tool_calls.append({"id": self._tool_call_seq,
                                 "name": name,
                                 "args": dict(args) if isinstance(args, dict) else {},
                                 "result": None})
        if len(self._tool_calls) > 100:
            del self._tool_calls[:-100]
        return self._tool_call_seq

    def record_tool_result(self, call_id, result):
        """Attach a tool call's RESULT to its ring entry so `/expand` can re-show
        output the user scrolled past (a run_command's stdout, a status dump, ...).
        Stored in the ring ONLY - never in self.messages - so it costs zero model
        context and compaction/handover never touch it. Capped at EXPAND_MAX_CHARS
        (head/tail) so a huge dump can't bloat local memory; the ring's own 100-entry
        cap bounds the total. `result` may be a str or the {text,image_path} dict a
        tool can return - we keep the text. Best-effort; never raises into the loop."""
        if not call_id or not getattr(self, "_tool_calls", None):
            return
        text = result.get("text", "") if isinstance(result, dict) else str(result or "")
        if len(text) > EXPAND_MAX_CHARS:
            head, tail = EXPAND_MAX_CHARS * 2 // 3, EXPAND_MAX_CHARS // 3
            text = (text[:head]
                    + f"\n…[{len(text) - head - tail} chars hidden]…\n"
                    + text[-tail:])
        for e in reversed(self._tool_calls):
            if e["id"] == call_id:
                e["result"] = text
                return

    def _maybe_auto_expand(self, name, call_id):
        """Auto-print the full args right after the call line when `name` is in the
        cfg.always_expand list (default [] - nothing auto-expands; diffs stay collapsed
        to the call line and /expand shows them on demand). No-op otherwise."""
        if name not in getattr(self.cfg, "always_expand", []):
            return
        entry = next((e for e in self._tool_calls if e["id"] == call_id), None)
        if entry:
            print(_render_tool_call(entry))

    def _log_activity(self, kind, summary, detail=""):
        """Record one automatic-behaviour event for /activity. `kind` is a short
        tag (auto-compact, handover, 429-fallback, token-refresh, ...), `summary`
        the one-line "what happened", `detail` any extra "why". These are the
        smart-but-invisible things the system does on the user's behalf; the log
        makes them scrollable after the fact instead of a scroll-past warning.
        Ring-buffered (cap 200) so it never grows unbounded. Lazily inits the
        buffer so a partially-built Agent (test doubles that skip __init__) can't
        crash a compaction/handover path on it."""
        if not hasattr(self, "_activity") or self._activity is None:
            self._activity = []
        self._activity.append({
            "t": time.time(),
            "kind": kind,
            "summary": summary,
            "detail": detail,
        })
        if len(self._activity) > 200:
            del self._activity[:-200]

    def _maybe_auto_compact(self):
        """Auto-COMPACT: the cheap, instant, programmatic strip (like /compact - stub old
        tool outputs) fired when context crosses a k*interval TOKEN boundary going UP. A
        first line of defence before a full agentic handover: no model turn, no summary,
        just drops stale tool dumps. Hysteresis (a Schmitt trigger) stops it re-firing
        every turn once a strip lands just under a boundary: after firing at boundary B it
        won't re-arm until context falls below B - hysteresis*interval. Off when
        auto_compact_interval is 0."""
        interval = getattr(self.cfg, "auto_compact_interval", 0)
        if interval <= 0:
            return
        used = self._ctx_used()
        hyst = interval * getattr(self.cfg, "auto_compact_hysteresis", 0.25)
        # re-arm: if we've dropped a full margin below the last-fired boundary, clear it
        if self._ac_armed_boundary and used < self._ac_armed_boundary - hyst:
            self._ac_armed_boundary = 0
        boundary = (used // interval) * interval        # highest boundary at/below `used`
        if boundary <= 0 or boundary <= self._ac_armed_boundary:
            return                                       # not past a NEW boundary yet
        keep = getattr(self.cfg, "auto_compact_keep", COMPACT_KEEP)
        trimmed, freed = self.compact(keep)
        self._ac_armed_boundary = boundary               # arm this boundary (don't refire)
        if trimmed:
            print(yellow(f"  {GLYPH['warn']} auto-compact at {used:,} tokens: stripped "
                         f"{trimmed} old tool result(s), ~{freed:,} tokens freed")
                  + dim(f"  (every {interval:,} tokens; change: /set auto_compact_interval)"))
            self._log_activity(
                "auto-compact",
                f"stripped {trimmed} old tool result(s), ~{freed:,} tokens freed",
                f"context crossed {boundary:,} (at {used:,}); fires every {interval:,} tokens")

    def _maybe_handover(self):
        """After a turn, run a handover when either (a) context is in the hard/emergency
        zone - forced, threshold-driven; or (b) the model ELECTED one this turn via
        request_handover (soft zone, a clean semantic boundary). Then optionally AUTOSTART
        the model's queued self-prompt. The loop guard (handover_min_interval) caps a
        FORCED compaction to ~1/min so the autostart->turn->compact cycle can't spin; we
        also refuse to autostart if compaction didn't actually drop below hard."""
        if not self.cfg.handover_for()["auto"]:
            return
        zone, used, limit = self._ctx_zone()
        elected = getattr(self, "_elected_handover", False)
        self._elected_handover = False        # consume it (fires at most once)
        forced = zone in ("hard", "emergency")
        if not (forced or elected):
            return
        if forced and not self._compact_allowed(zone):
            return
        if not self._model_tool_support():
            # Compaction summarizes via a tool-capable turn; a chat-only model can
            # never complete it, so attempting it just burns a turn + snapshot every turn.
            # Skip, and say why once (the model, not the user, is the limitation).
            if not getattr(self, "_compact_unsupported_warned", False):
                print(yellow(f"  {GLYPH['warn']} context is full ({used:,}/{limit:,}) but "
                             f"'{self.cfg.active_model()}' can't tool-call, so auto-compaction "
                             f"can't run - switch to a tool-capable model, /compact, or /clear."))
                self._compact_unsupported_warned = True
            return
        pct = int(used / limit * 100) if limit else 0
        kind = "elective handover (clean break)" if elected and not forced else "auto-handover"
        print(yellow(f"  {GLYPH['warn']} context {used:,}/{limit:,} ({pct}%, {zone}) - "
                     f"{kind} (summarizing + resetting the cache prefix)…")
              + dim("  (change: /set handover_hard)"))
        if self.auto_handover(zone) is None:
            # The agentic summary failed (model couldn't produce a usable handover -
            # common when history already overflows the window so the summarizing turn
            # itself is truncated). For a FORCED handover, leaving history intact would
            # re-fire emergency every turn forever (see the 232% loop) - so fall back to a
            # programmatic strip. For an ELECTED one, context isn't full: just report and
            # carry on (no strip needed). The loop guard is already stamped on failure.
            if not forced:
                print(yellow(f"  {GLYPH['warn']} elective handover produced no summary - "
                             f"nothing changed, carrying on."))
                return
            trimmed, freed = self.compact(self.cfg.auto_evict_keep)
            if trimmed:
                print(yellow(f"  {GLYPH['warn']} agentic handover failed; stripped {trimmed} old "
                             f"tool result(s) instead, ~{freed:,} tokens freed  ")
                      + dim("(/compact or /clear if still full)"))
                self._log_activity(
                    "handover-failed",
                    f"summary call failed; fell back to stripping {trimmed} tool result(s), ~{freed:,} tokens freed",
                    "usually a backend overload/429 during the summarizing turn; history kept, raw tool output dropped")
            else:
                print(yellow(f"  {GLYPH['warn']} agentic handover failed and nothing to strip - "
                             f"/clear or switch to a larger-window model."))
                self._log_activity(
                    "handover-failed",
                    "summary call failed and nothing left to strip",
                    "/clear or switch to a larger-window model")
            return
        print(yellow(f"  {GLYPH['warn']} handover done -> ~{self._ctx_used():,} tokens kept; "
                     f"new cache boundary set on the summary"))
        self._log_activity(
            "handover",
            f"summarized the session -> ~{self._ctx_used():,} tokens kept",
            f"{kind}; new cache boundary set on the summary")
        sp = self._autostart_pending
        self._autostart_pending = None
        if not sp:
            return
        if self._ctx_zone()[0] in ("hard", "emergency"):
            # compaction didn't free enough; autostarting would immediately re-compact.
            # Surface the planned step instead of running it.
            self.messages.append({"role": "user", "content":
                "[autostart skipped - context still full after compaction] planned next step:\n" + sp})
            return
        if self._autostart_countdown(sp):
            self.run_turn(sp)
        else:
            # cancelled: drop the self-prompt into history as a visible note, do NOT run it
            self.messages.append({"role": "user", "content":
                "[autostart cancelled by user] the planned next step was:\n" + sp})

    def _autostart_countdown(self, sp, secs: int = 5) -> bool:
        """Visible 'autostarting in 5..1, ^C to cancel' beat before running the model's
        own self-prompt. Returns True to proceed, False if the user interrupts. Thin +
        interactive (the decision logic in _maybe_compact is what's unit-tested).

        Runs under composer_suspended(): the countdown happens at the tail of run_turn
        while the composer still owns the TTY (cbreak, reader thread draining stdin), so
        without handing the real terminal back a ^C would be swallowed by the composer
        (never reaching us as KeyboardInterrupt) AND the \\r redraw would garble through
        the pinned-line renderer. Suspending restores cooked mode + echo + real stdout."""
        preview = sp.replace("\n", " ")
        preview = preview[:80] + "…" if len(preview) > 80 else preview
        with composer_suspended():
            print(dim(f"  {GLYPH['dot']} autostarting (^C to cancel): {preview}"))
            try:
                for n in range(secs, 0, -1):
                    sys.stdout.write(dim(f"\r  {n}… "))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\r" + " " * 24 + "\r")
                sys.stdout.flush()
                return True
            except KeyboardInterrupt:
                sys.stdout.write(dim("\r  autostart cancelled.        \n"))
                sys.stdout.flush()
                return False

    def _chat_with_model_fallback(self):
        """Send to the model; if it 404s because the active model is unavailable
        (e.g. a tier-gated model left in the config), print a
        clear message and retry with the next model the backend offers instead of
        failing the whole turn. The fallback autosaves with the rest of config."""
        tried = set()
        while True:
            try:
                # repair AFTER windowing so we send a tool-pair-consistent slice -
                # heals an already-poisoned (resumed) session that would 400 forever.
                msgs = repair_tool_pairs(
                    self._window_messages(self.messages, self.cfg.window_messages))
                # Stitch the pinned plan onto the SENT slice only (a copy) - never into
                # stored history, and after the last cache breakpoint (messages[-2]) so it
                # stays uncached and a plan edit busts nothing. See _plan_reminder.
                msgs = self._with_plan_reminder(msgs)
                return self.client.chat(
                    msgs, self.tool_defs, should_abort=lambda: self._abort)
            except Exception as e:
                # A 404 from the chat endpoint means the model path is bad (unknown
                # / not entitled), not a transport failure - anything else re-raises
                # to the normal handler.
                if not (getattr(e, "code", None) == 404 or "404" in str(e)):
                    raise
                current = self.cfg.active_model()
                tried.add(current)
                nxt = self._next_available_model(tried)
                if not nxt:
                    raise RuntimeError(
                        f"model '{current}' is unavailable on this account (404) and "
                        f"no alternative is offered - use /model to pick one, or "
                        f"/provider to switch.")
                print(yellow(f"\n{GLYPH['warn']} model '{current}' unavailable (404); "
                             f"falling back to '{nxt}'  (/model to pick another)"))
                self._apply_model(nxt)

    def _next_available_model(self, tried):
        """First model the active backend offers that we haven't already tried."""
        try:
            models = list(self.client.list_models() or [])
        except Exception:
            models = []
        return next((m for m in models if m and m not in tried), None)

    def _apply_model(self, name):
        """Switch the active backend's model at runtime; the client reads
        cfg.active_model() on the next send."""
        spec = self.active_provider()
        self.cfg.set_active_model(name)
        if spec and spec.get("context_window"):
            try:
                self.cfg.set_setting("num_ctx", int(spec["context_window"](name)))
            except Exception:
                pass
        # The new model may differ in tool-support (a chat-only model exposes NO tools
        # regardless of leash; a tool-capable one restores them). Rebuild the surface so
        # a same-provider /model switch - or a 404 fallback - doesn't leave it stale
        # (e.g. switching off a chat-only model used to keep showing "0 tools").
        self.refresh_tools()

    def _offer_login_on_auth_error(self, err) -> bool:
        """First-class onboarding: when a turn fails because the active provider has no
        (or a rejected) API key, run that provider's login hook inline and re-activate,
        so the user can paste a key and continue rather than hitting a dead red error.
        Returns True if login succeeded and the turn should be retried; False otherwise
        (unrecognized error, no login hook, or the user declined/cancelled)."""
        msg = str(err).lower()
        auth = ("no api key" in msg or "api key" in msg
                or " 401" in msg or "401:" in msg or " 403" in msg or "403:" in msg
                or "unauthorized" in msg or ("invalid" in msg and "key" in msg))
        if not auth:
            return False
        spec = self.active_provider()
        if not spec or not spec.get("login"):
            return False
        name = self.cfg.provider
        print(yellow(f"\n{GLYPH['warn']} {name}: authentication failed - "
                     f"you need to log in."))
        if not _ask(f"log in to '{name}' now?"):
            print(dim(f"  skipped. Run /provider login {name} when ready, "
                      f"or /provider to switch backend."))
            return False
        try:
            ok = spec["login"](self, self.cfg)
        except Exception as e:
            print(red(f"login failed: {e}"))
            return False
        if ok is False:
            print(dim("login cancelled."))
            return False
        try:
            model = self.activate_provider(name)
            print(green(f"provider: {name}") + dim(f"  model: {model}  - retrying..."))
        except Exception as e:
            print(red(f"/provider {name} failed: {e}"))
            return False
        return True

    def _recover_and_should_retry(self, err) -> bool:
        """One-shot recovery for a failed send. Tries transport failover first (the
        provider's on_conn_error hook - ollama re-probes its priority pool and jumps to
        the next live host), then the auth/login flow. Returns True when something
        recovered and the turn should be retried, else False (caller prints the error)."""
        spec = self.active_provider()
        recover = spec.get("on_conn_error") if spec else None
        if isinstance(err, ConnectionError) and recover:
            try:
                if recover(self, self.cfg, err):
                    return True
            except Exception:
                pass
        return self._offer_login_on_auth_error(err)

    def _loop(self):
        # max_iterations <= 0 means unlimited (count up forever).
        cap = self.cfg.max_iterations
        i = 0
        while cap <= 0 or i < cap:
            i += 1
            # Pre-iteration hook (worker inject path only; None in interactive sessions).
            # A headless worker sets this to a poller that, at this clean between-iteration
            # boundary, drains any pending inject sidecar and appends it as a user turn so
            # the worker sees an operator/parent mid-task message on its next think. Never
            # interrupts a running tool call; best-effort, never raises into the loop.
            _hook = getattr(self, "_pre_iter_hook", None)
            if _hook:
                try:
                    _hook()
                except Exception:
                    pass
            # Periodic in-turn status: a constantly-iterating model can run many
            # loops without ever returning to the prompt (where the status block
            # normally prints), so ctx/quota/perms go stale on screen. When
            # statusline_iter is set, reprint the block every N iterations.
            _si = getattr(self.cfg, "statusline_iter", 0)
            if (_si and i > 1 and (i - 1) % _si == 0
                    and getattr(self.cfg, "statusline", True) and _TTY):
                for _row in _status_rows(self, self.cfg):
                    print(_row)
            if self.cfg.auto_evict:
                _ev_trimmed, _ev_freed = self._auto_evict()
                if _ev_trimmed:
                    self._log_activity(
                        "auto-evict",
                        f"stubbed {_ev_trimmed} acted-on tool result(s), ~{_ev_freed:,} tokens freed",
                        f"continuous eviction; keeps last {self.cfg.auto_evict_keep} in full")
            try:
                assistant, prompt_eval, aborted = self._chat_with_model_fallback()
            except (ConnectionError, RuntimeError) as e:
                # A failed send gets ONE recovery attempt then a retry:
                #  - transport recovery: a provider may fail over (ollama re-probes its
                #    priority pool and jumps to the next live host), so a dropped LAN box
                #    / roaming off wifi doesn't dead-end;
                #  - auth recovery (OBE): a missing/invalid key (401/403) on a provider
                #    that can log in offers the login flow inline instead of a red error.
                if not self._recover_and_should_retry(e):
                    print(red(f"\n{e}"))
                    return
                try:
                    assistant, prompt_eval, aborted = self._chat_with_model_fallback()
                except (ConnectionError, RuntimeError) as e2:
                    print(red(f"\n{e2}"))
                    return
            if aborted:
                # Drop the half-streamed turn; history stays consistent.
                print(yellow("\n^C - stopped mid-inference; back to prompt"))
                return
            if prompt_eval:
                # Self-calibrate the estimator: compare the provider's REAL prompt count
                # (incl. cache read/write, so it reflects the whole context) against our
                # uncalibrated char estimate of the same send, and nudge the factor. This
                # is what fixes the pre-first-turn / restore under-report going forward.
                client = getattr(self, "client", None)
                cr = getattr(client, "last_cache_read", 0) or 0
                cw = getattr(client, "last_cache_write", 0) or 0
                raw = _raw_messages_tokens(self.messages, self.tool_defs)
                calibrate_tokens(raw, prompt_eval + cr + cw)
                self.last_prompt_tokens = prompt_eval
                self.session_in += prompt_eval
            # Output tokens are reported by the client (Ollama eval_count / API
            # output_tokens) on an optional attribute, so any backend contributes
            # to the session count without a chat() signature change.
            self.session_out += getattr(self.client, "last_out_tokens", 0) or 0
            self.messages.append(assistant)
            calls = assistant.get("tool_calls")
            if not calls:
                self._end_of_turn()
                return
            # Parallel fast-path: when the model batches several independent
            # read-only tools (and we're local), run them at once - N reads/greps
            # in the wall time of the slowest, not the sum. Whole batch must be
            # safe: a writer/command needs an interactive confirm (can't prompt
            # for several at once) and could race. Remote is excluded - its
            # executor is a single serial pipe (one stdin/stdout), so concurrent
            # calls would corrupt the stream.
            if (len(calls) > 1 and not self.remote
                    and all(self._parallel_safe(c) for c in calls)):
                self._run_parallel(calls)
                if self._abort:
                    return
                continue
            for call in calls:
                if self._abort:                 # ^C between tool calls
                    return
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                cid = self.record_tool_call(name, args)
                print(_tool_call_line(name, args, cid))
                self._maybe_auto_expand(name, cid)
                if name == "ask_user_to_run":
                    # Interactive handoff: no spinner, operator runs it directly
                    # (on the remote box when connected). Suspend the composer so
                    # the editable prompt owns stdin/echo (no reader-thread race).
                    with composer_suspended():
                        result = run_direct_command(
                            self.cfg, args.get("cmd", ""), args.get("reason"),
                            remote=self.remote)
                else:
                    # Tool-running indicator (distinct from the model spinner),
                    # labelled with the tool so you always see WHAT is running.
                    # _ask() pauses it for confirmation prompts; in auto mode it
                    # animates (with an elapsed counter) across the whole execution.
                    spin = Spinner(name, TOOL_FRAMES, cyan, interval=0.12).start()
                    try:
                        result = self._run_tool(name, args)
                    finally:
                        spin.stop()
                self.record_tool_result(cid, result)
                if _TTY:
                    print(_tool_result_preview(name, result, cid))
                self.messages.append(_tool_result_msg(name, result,
                                                      tool_call_id=call.get("id", "")))
        print(yellow(f"\n{GLYPH['warn']} hit {cap}-iteration cap; stopping this turn. "
                     f"Refine your request or continue, or raise the limit: "
                     f"/set -> max_iterations (0 = unlimited), or set "
                     f"max_iterations in {CONFIG_PATH}."))
        self._end_of_turn()

    def _end_of_turn(self):
        changed = sorted(self.tools.changed_files)
        if changed:
            print(green(f"\n{GLYPH['edit']} changed {len(changed)} file(s): "
                        + ", ".join(changed)))
        # The status rows (printed before the next prompt) already carry ctx + quota,
        # so only print the standalone meter when the status line is off (fallback).
        if not (getattr(self.cfg, "statusline", True) and _TTY):
            self._print_ctx()

    # The context measurement / compaction-policy methods live on self.ctx
    # (ContextMeter) now; these stay as thin delegators so the ~14 internal + external
    # call sites - and the suite's monkeypatch seam (tests override self._ctx_zone to
    # drive auto_handover) - keep working unchanged.
    def _ctx_used(self) -> int:
        return self.ctx.used()

    def _ctx_zone(self):
        return self.ctx.zone()

    def _ctx_summary(self, cfg):
        """Shared context read-out math for /info and the status meter (kept in one
        place so their used/window/zone/pct never drift). Returns
        (used, window, zone, pct). Callers pick their own label strings."""
        used, window = self._ctx_used(), cfg.ctx_window() or 1
        try:
            zone = self._ctx_zone()[0]
        except Exception:
            zone = "ok"
        return used, window, zone, used / window * 100

    def _compact_allowed(self, zone) -> bool:
        return self.ctx.compact_allowed(zone)

    def _print_ctx(self):
        self.ctx.print_meter()


def _fmt_reset(iso) -> str:
    """ISO timestamp -> short local reset with a countdown so a bare clock time
    can't be misread as a number: 'HH:MM(Nh)' within a day, else 'ddMon(Nd)'
    (e.g. '15:00(18h)', '05Jul(3d)'). Under an hour reads 'HH:MM(Nm)'. Core owns
    this so every provider's reset times read the same way."""
    if not iso:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone()
    except ValueError:
        return ""
    secs = (dt - datetime.datetime.now(dt.tzinfo)).total_seconds()
    if secs < 0:
        return dt.strftime("%H:%M(now)")
    if secs < 86400:
        cd = f"{int(secs // 3600)}h" if secs >= 3600 else f"{int(secs // 60)}m"
        return dt.strftime("%H:%M") + f"({cd})"
    return dt.strftime("%d%b") + f"({int(secs // 86400)}d)"


def render_usage_meters(usage, verbose=False) -> str:
    """Render a provider's usage dict into a status suffix, with core's own colour
    ramp / reset formatting (the provider supplies only numbers). Shape:
    {"meters": [{"label": "5h", "pct": 12, "resets_at": iso, "day": "D6",
    "tag": {...}}, ...], "note": ""}.
    The status line stays TIDY: reset time/countdown is shown only when verbose
    (i.e. /info), never on the row-3 meter. Returns "" for an empty/None usage so
    the caller can fall back to the plain ctx line."""
    if not usage:
        return ""
    parts = []
    for m in usage.get("meters", []):
        try:
            pct = float(m.get("pct", 0))
        except (TypeError, ValueError):
            continue
        label = str(m.get("label", "")).strip()
        body = (f"{label} FULL" if pct >= 100 else f"{label} {pct:.0f}%")
        if verbose:                      # reset lives in /info, not the tidy status line
            rst = _fmt_reset(m.get("resets_at", ""))
            if rst:
                body += f" {rst}"
        body = _pct_color(pct)(body)
        # Optional provider tag rendered bracketed AFTER the pct with its own colour
        # ramp (e.g. today's spend '(0%/day6)').
        # {"text": str, "level": "ok"|"warn"|"crit"} - provider owns the level,
        # core owns the colour so every provider's tags read the same.
        tag = m.get("tag")
        if isinstance(tag, dict) and str(tag.get("text", "")).strip():
            tclr = {"crit": red, "warn": yellow}.get(tag.get("level"), blue)
            body += tclr(f" ({str(tag['text']).strip()})")
        parts.append(body)
    note = str(usage.get("note", "")).strip()
    if note:
        parts.append(dim(note))
    if not parts:
        return ""
    sep = dim(f" {GLYPH['dot']} ")          # match the status rows' separator spacing
    return sep + sep.join(parts)


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", GLYPH["ret"])
        if len(s) > 50:
            s = s[:50] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


# Arg keys that carry a large content BLOB (file bodies, diffs, plans). On a call
# line these are pure noise - the value is huge and the RESULT preview already
# confirms the outcome. Collapsed to a compact size hint (…Nc/ML) for EVERY tool,
# so a custom lean-tool with a 'content'/'body' arg benefits too. /expand shows full.
_BLOB_ARGS = frozenset({
    "content", "diff", "plan", "new_str", "old_str", "text", "body", "patch", "new_text"})
# Where a write-tier tool keeps its target path (checked in order). The call line
# shows just the BASENAME - the result preview carries the authoritative full path.
_PATH_ARGS = ("path", "file_path", "filename", "file", "target")


def _fmt_call_args(name: str, args: dict) -> str:
    """Category-aware TERSE args for a tool CALL line (not /expand, which is full).
    WRITE-tier tools (write_file, apply_diff, …) collapse to just the target's
    basename - the path + fact-of-write live on the result line, repeating them on
    the call line is noise. Every other tier keeps its args (a read path, a
    run_command cmd, a url are all load-bearing) but any large content BLOB
    (content=/diff=/plan=…) collapses to a size hint so it can't swamp the line.
    Uniform: driven by _tool_category + the blob-key set, no per-tool special-case."""
    if _tool_category(name, args) == "write":
        for k in _PATH_ARGS:
            if args.get(k):
                return os.path.basename(str(args[k]).rstrip("/")) or str(args[k])
        # no recognisable path arg - fall through to the generic (blob-collapsed) form
    parts = []
    for k, v in args.items():
        if k in _BLOB_ARGS:
            s = str(v)
            parts.append(f"{k}=…{len(s)}c/{s.count(chr(10)) + 1}L")
            continue
        s = str(v).replace("\n", GLYPH["ret"])
        if len(s) > 50:
            s = s[:50] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _is_bg_call(name: str, args: dict) -> bool:
    """True when this is a backgrounded (detached) run_command - a trailing '&' on
    the cmd."""
    return name == "run_command" and str(args.get("cmd", "")).rstrip().endswith("&")


# name -> display glyph, populated from a lean-tool's optional TOOL["glyph"] as it
# loads (see LeanToolManager._load_dir). An explicit glyph wins over the derived
# category in _tool_glyph. Module-level so the pure _tool_glyph(name, args) can read
# it without an agent handle (both print paths agree).
_LEAN_TOOL_GLYPHS = {}

# Lean-tools whose job is network egress (no core tier says so - they're plugins), so
# the call line can show the 'net' category icon. A name not here just falls through to
# its tier / the gear; missing one only costs a less-specific icon, never correctness.
_NET_TOOLS = frozenset({
    "web_fetch", "web_search", "brave_search", "web_screenshot", "render_url", "ssh",
    "fetch", "curl", "http_get"})
# Agent-internal / bookkeeping tools: no fs, no exec, no net - a distinct 'meta' icon.
_META_TOOLS = frozenset({"update_plan", "request_handover", "note", "bg_status"})


def _tool_category(name: str, args: dict) -> str:
    """Coarse CATEGORY for a tool call's icon, derived from the same tiering the leash
    uses (single source of truth) plus two small plugin-name sets. Returns one of:
    'mcp' | 'net' | 'meta' | 'write' | 'exec' | 'read' | ''. '' = uncategorised (gear).
    Pure (name/args only) so both the sync and parallel print paths agree."""
    if name.startswith(MCP_NS):
        return "mcp"
    if name in _NET_TOOLS:
        return "net"
    if name in _META_TOOLS:
        return "meta"
    if name in _WRITE_TIER:
        return "write"
    if name in _EXEC_TIER or name == "ask_user_to_run":
        return "exec"
    if name in SAFE_TOOLS:
        return "read"
    return ""


def _is_wake_turn(text: str) -> bool:
    """A synthesised autonomous-wake turn (a bg task / worker finished with no
    operator input) - it is NOT something the human typed. bg_wake_turn() always
    frames it with this leading marker; the repl uses this to echo it distinctly
    instead of dressing it up as an operator turn."""
    return (text or "").lstrip().startswith("[autonomous wake")


_WAKE_DISPLAY_CAP = 200   # chars of a wake/finish turn shown inline; full via /expand


def _wake_display_body(text: str) -> str:
    """The USER-facing body of an autonomous-wake turn: strip the model-facing
    '[autonomous wake ... react now]' instruction preamble (that's an instruction to
    the MODEL, not something the human needs to read), leaving just the event notice
    (e.g. 'worker N finished: ...'). Also peels one pair of wrapping [] off the notice
    so it reads as plain text, not a bracketed blob."""
    t = (text or "").lstrip()
    if t.startswith("[autonomous wake"):
        end = t.find("]")
        if end != -1:
            t = t[end + 1:].lstrip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1].strip()
    return t


def _wake_label(body: str) -> str:
    """The accent-bar label for a wake turn - names WHAT finished rather than the
    generic 'auto', so the human reads 'worker' when a worker finished (the common
    case), 'bg' for a background task, else 'auto'."""
    head = body.lstrip().lower()
    if head.startswith("worker"):
        return "worker"
    if head.startswith("bg ") or head.startswith("background") or "task" in head[:40]:
        return "bg"
    return "auto"


def _wake_turn_lines(text: str, agent=None) -> str:
    """Echo an autonomous-wake turn into scrollback with a dim cyan accent bar,
    labelled by WHAT finished ('worker'/'bg'/'auto') - visually distinct from an
    operator turn (yellow, named) and AI/tool lines, so it reads as 'the system woke
    me', not 'the user said this'. The model-facing wake preamble is stripped (the
    human just sees 'worker N finished: ...'), the body is truncated HARD inline, and
    - when an agent is passed - the FULL text is stashed in the /expand ring so
    `/expand N` shows all of it, exactly like a tool-call result."""
    body = _wake_display_body(text)
    label = _wake_label(body)
    cid = None
    if agent is not None:
        try:
            cid = agent.record_tool_call(f"{label} finished", {})
            agent.record_tool_result(cid, body)
        except Exception:
            cid = None
    # Show only the FIRST line, hard-capped - a wake turn is a headline, not the whole
    # payload (that's what /expand is for). Keeps it to one tidy row in scrollback.
    first = body.splitlines()[0] if body.splitlines() else body
    shown = first
    extra_lines = max(0, len(body.splitlines()) - 1)
    if len(shown) > _WAKE_DISPLAY_CAP:
        shown = shown[:_WAKE_DISPLAY_CAP].rstrip()
    hidden = len(body) - len(shown)
    if hidden > 0:
        # Point at the command that renders the FULL result. A worker's full output
        # lives in its result file (not this notice), so /worker <pid> is the real
        # source; fall back to /expand for a plain (non-worker) wake turn.
        m = re.match(r"worker\s+(\d+)", body.lstrip())
        if m:
            full_cmd = f"/worker {m.group(1)}"
        elif cid:
            full_cmd = f"/expand {cid}"
        else:
            full_cmd = None
        tail = f" …[+{hidden} chars; {full_cmd} for the full result]" if full_cmd \
            else f" …[+{hidden} chars]"
        shown = shown + tail
    bar = cyan(GLYPH["userbar"])
    tag = dim(f"  #{cid}") if cid else ""
    head = f"{bar} {cyan(bold(label + ' finished'))}{tag}"
    return f"{head}\n{bar} {dim(shown)}"


def _operator_turn_lines(name: str, text: str, agent=None) -> str:
    """Echo the operator's own submitted turn into scrollback with a yellow left
    accent bar + their name, so a human turn is easy to spot in a sea of AI (blue ●)
    and tool lines. The bar runs down every line of a multi-line paste. ASCII '|'
    fallback on a non-UTF terminal; colour is _TTY-gated by the yellow() helper.
    A synthesised autonomous-wake turn is echoed distinctly (see _wake_turn_lines)
    so it never masquerades as something the human typed; pass `agent` to make that
    turn /expand-able."""
    if _is_wake_turn(text):
        return _wake_turn_lines(text, agent)
    bar = yellow(GLYPH["userbar"])
    head = f"{bar} {yellow(bold(name))}"
    body = "\n".join(f"{bar} {ln}" for ln in (text or "").splitlines()) or bar
    return f"{head}\n{body}"


def _tool_glyph(name: str, args: dict) -> str:
    """Icon for a tool call line: a distinct 'bg' glyph when this is a backgrounded
    (detached) run_command, else a per-CATEGORY icon (read/write/exec/net/mcp/meta)
    derived from the tool's tier, falling back to the gear for anything uncategorised."""
    if _is_bg_call(name, args):
        return GLYPH["bg"]
    override = _LEAN_TOOL_GLYPHS.get(name)   # a lean-tool's own TOOL["glyph"] wins
    if override:
        return override
    cat = _tool_category(name, args)
    return GLYPH.get(f"cat_{cat}", GLYPH["tool"]) if cat else GLYPH["tool"]


def _tool_call_line(name: str, args: dict, call_id: int = 0) -> str:
    """The dim one-line summary printed for a tool call. A backgrounded
    run_command gets an explicit ' (background)' text tag so it reads as detached
    even without colour/glyph support - not just a different icon. `call_id` (when
    >0) is shown as a '#N' tag so `/expand N` can re-render the full (untruncated)
    args; bare `/expand` targets the newest."""
    tag = "  (background)" if _is_bg_call(name, args) else ""
    body = dim(f"  {_tool_glyph(name, args)} {name}({_fmt_call_args(name, args)}){tag}")
    if not call_id:
        return _fit_line(body)
    # Pin '#N' to the right INSIDE the width budget: reserve its visible width
    # ("  #N"), fit the rest to width-minus-that, then append it. So on a narrow
    # line the args truncate with '…' but the '#N' (how /expand targets a call)
    # always survives - it never orphans onto a wrapped second physical line.
    idt = dim(f"  #{call_id}")
    reserve = len(f"  #{call_id}")
    width = max(1, _term_cols() - 1)
    return _fit_line(body, max(1, width - reserve)) + idt


def _result_status(text: str):
    """(glyph_key, color_fn) for a tool result's outcome, from its text. error/exit!=0/
    [tool error] -> 'no' (red); everything else -> 'ok' (green). Pure -> tested."""
    s = (text or "").lstrip().lower()
    if (s.startswith(("error", "[tool error]", "operator declined", "user declined"))
            or s.startswith("exit ") and not s.startswith("exit 0")
            or "traceback (most recent call last)" in s):
        return "no", red
    return "ok", green


# A run_command result opens with an 'exit N' / 'exit N (no output)' status line; the
# status glyph already conveys pass/fail, so the preview skips this line to surface the
# actual OUTPUT instead of a redundant 'exit 0'.
_EXIT_LINE_RE = re.compile(r"^exit -?\d+(?: \(no output\))?$")


def _first_meaningful_line(text: str) -> str:
    """The first non-empty, non-noise line of a tool result - the teaser shown under
    the call line. Skips blank lines and a leading run_command 'exit N' status line
    (the glyph already shows pass/fail), so the preview surfaces real output; collapses
    inner whitespace. '' for an empty result (caller shows '(no output)')."""
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln and not _EXIT_LINE_RE.match(ln):
            return " ".join(ln.split())
    return ""


def _tool_result_preview(name: str, result, call_id: int = 0) -> str:
    """The uniform ONE-LINE result preview printed UNDER a tool's call line, for EVERY
    tool (the operator never sees full tool output otherwise). Shape: 2-space indent,
    a status glyph (✓/✗, coloured), then the first meaningful line of the result,
    truncated to the terminal width so it never wraps (right edge lands like the call
    line's). `/expand N` shows the full result. `result` may be the {text,image_path}
    dict a tool can return. Returns "" only when there is genuinely nothing to show."""
    text = result.get("text", "") if isinstance(result, dict) else str(result or "")
    key, color = _result_status(text)
    first = _first_meaningful_line(text)
    # Count lines the preview isn't showing, ignoring a skipped leading 'exit N'
    # status line so the '+N more' tally matches what /expand actually reveals.
    lines = [ln for ln in text.splitlines() if not _EXIT_LINE_RE.match(ln.strip())]
    line_count = len(lines)
    body = first if first else dim("(no output)")
    if first and line_count > 1:
        body += dim(f"  (+{line_count - 1} more; /expand{f' {call_id}' if call_id else ''})")
    lead = "  " + color(GLYPH[key]) + " "
    return "\r" + _fit_line(lead + body) + "\033[K"


EXPAND_MAX_CHARS = 4000     # cap on /expand output so it can't flood/hang the terminal


def _render_tool_call(entry: dict, cap: int = EXPAND_MAX_CHARS) -> str:
    """Full, human-readable view of a recorded tool call's args (for /expand).
    A diff (apply_diff) renders as a colorized unified diff; everything else is
    printed as key: value blocks. Total output is head/tail capped at `cap` chars
    so a huge write_file/diff can't flood the terminal (the flood-hang we fixed)."""
    name, args = entry.get("name", ""), entry.get("args", {}) or {}
    out = []
    for k, v in args.items():
        s = v if isinstance(v, str) else json.dumps(v, indent=2, ensure_ascii=False)
        out.append(f"{bold(k)}:\n{s}" if "\n" in str(s) else f"{bold(k)}: {s}")
    body = "\n".join(out) if out else dim("(no args)")
    if len(body) > cap:
        head, tail = cap * 2 // 3, cap // 3
        body = body[:head] + dim(f"\n…[{len(body) - head - tail} chars hidden]…\n") + body[-tail:]
    # The stored RESULT (output), if we captured one - so /expand re-shows what
    # scrolled past, not just the call's args. Already capped at record time.
    if "result" in entry:
        res = entry.get("result")
        if res:
            body += "\n" + dim("─ result ─") + "\n" + res      # sweep-ok
        elif res == "":
            body += "\n" + dim("─ result ─ (empty)")      # sweep-ok
        else:
            body += "\n" + dim("─ result ─ (not captured)")      # sweep-ok
    return body


# ----------------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------------

SLASH_COMMANDS = ["/clear", "/new", "/compact", "/handover", "/session", "/save", "/load",
                  "/prompt", "/sh", "/connect", "/machines", "/local", "/disconnect", "/tools", "/reload",
                  "/model", "/provider", "/think", "/effort",
                  "/set", "/usage", "/approve", "/leash", "/autosave", "/incognito",
                  "/askread", "/bg", "/note", "/mcp", "/info", "/ctx", "/activity", "/expand", "/help", "/quit"]

# Built-in command names are the shadow-protection set: lean-tool commands can't claim
# any of them. It is DERIVED from _BUILTIN_COMMANDS_TABLE (every builtin command + alias)
# right after that table is defined, below - so the protected set can never drift from
# what's actually dispatched. SLASH_COMMANDS (above) is the curated completion list
# (canonical names only; aliases like /disconnect are reachable but not offered for
# completion). /reset is gone for good - /clear replaced it.

# Slash commands contributed by lean-tools via setup() -> register_command.
# name -> (handler(agent, cfg, arg), help_line). repl() consults this after its
# built-in chain; populated only on the driver (setup() never runs remotely).
_lean_tool_commands = {}
# Optional Tab-completion for lean-tool commands: name -> completer(agent, cfg) ->
# [str]. Lets a lean-tool (e.g. /update) complete its first arg the same way the
# built-ins do; absence just means no inline completion (the no-arg picker, if the
# handler implements one, still works).
_lean_tool_completers = {}
# Which plugin (tool/provider name) registered each command, so a SECOND plugin
# claiming the same /command is warned + ignored (first wins) while a plugin
# re-registering its own command still replaces. Set around each setup() call.
_lean_tool_cmd_owner = {}
_registering_owner = None


# External model providers contributed by lean-tools via setup() ->
# register_provider. name -> normalized spec. The built-in Ollama backend is NOT
# in here; "" / "ollama" mean native. Populated only on the driver.
_providers = {}


def register_provider(spec):
    """Register an external model backend (called from a lean-tool's setup(),
    usually as lc['register_provider']). `spec` is a dict; see PROVIDER_API.md.

    Required keys:
      name            unique id (string)
      make_client     cfg -> client with .chat(messages, tools, should_abort)
      list_models     () -> [model_id, ...]
      context_window  model_id -> int

    Optional keys (sensible defaults filled in):
      capabilities    model_id -> {setting_key: [choices]}   (default {})
      available       () -> bool   gate for activation/menus  (default True)
      autostart       () -> bool   activate on launch         (default False)
      on_activate     (agent, cfg) -> None
      on_deactivate   (agent, cfg) -> None
      usage           (agent, cfg) -> {"meters":[...], "note": str} or None
      status_line     (agent, cfg) -> str   raw-string escape hatch
      login           (agent, cfg) -> bool  auth flow; enables /provider login
      clear           (agent, cfg) -> None  wipe credentials; enables /provider clear
      detail          (agent, cfg) -> str   full usage view for /usage
      tag             str   short label for the unified /model row (default: name)

    Returns the normalized spec. Re-registering a name replaces it."""
    name = spec.get("name", "")
    if not name:
        raise ValueError("provider needs a unique name")
    for req in ("make_client", "list_models", "context_window"):
        if not callable(spec.get(req)):
            raise ValueError(f"provider '{name}': '{req}' must be callable")
    norm = {
        "name": name,
        "make_client": spec["make_client"],
        "list_models": spec["list_models"],
        "context_window": spec["context_window"],
        "capabilities": spec.get("capabilities") or (lambda m: {}),
        "available": spec.get("available") or (lambda: True),
        "autostart": spec.get("autostart") or (lambda: False),
        "on_activate": spec.get("on_activate") or (lambda a, c: None),
        "on_deactivate": spec.get("on_deactivate") or (lambda a, c: None),
        "usage": spec.get("usage"),
        "status_line": spec.get("status_line"),
        "login": spec.get("login"),
        "clear": spec.get("clear"),
        "detail": spec.get("detail"),
        # short label shown next to each model row in the unified /model -
        # e.g. "plan" -> "model-x  [plan]". Defaults to the provider name.
        "tag": (spec.get("tag") or name),
        # opt-in hooks (each absent-safe; non-implementers pay nothing):
        #   warm_models()        -> [model_id] loaded/instant models (ollama green dots)
        #   model_status(model)  -> str|None   install/availability hint for /model
        #   tool_support(model)  -> bool        False = chat-only model (no tool calling);
        #                                       absent -> True (every current backend supports tools)
        #   endpoints()          -> [url]      multi-endpoint failover (ollama hosts)
        #   location(cfg)        -> str         status/info label for the backend (ollama:
        #                                       host alias + url); absent -> the provider name
        #   prepare(agent, cfg)  -> None        pre-activation setup run by core BEFORE
        #                                       activate_provider (ollama: tiered host failover
        #                                       + per-host default model). Absent -> nothing.
        "warm_models": spec.get("warm_models"),
        "model_status": spec.get("model_status"),
        "tool_support": spec.get("tool_support") or (lambda m: True),
        "endpoints": spec.get("endpoints"),
        "location": spec.get("location"),
        "prepare": spec.get("prepare"),
        # on_conn_error(agent, cfg, err) -> bool: last-chance recovery when a send
        # fails with a transport error. Ollama re-probes its priority pool and fails
        # over to the next live host; return True to retry the turn. Absent -> no retry.
        "on_conn_error": spec.get("on_conn_error"),
        # system_addendum(agent, cfg) -> str: extra text appended to the system
        # prompt for the ACTIVE model (ollama: small-model tool-calling guidance -
        # use the tool_calls channel, a few-shot, don't write calls as text).
        # Cache-safe: only changes when the model changes. Absent/"" -> nothing.
        "system_addendum": spec.get("system_addendum"),
        # tool_filter(cfg, tools) -> tools: last-pass filter/reorder of the tool
        # surface for the ACTIVE model (ollama: drop apply_diff for small models
        # that can't produce byte-exact diffs - they use write_file instead).
        # Absent -> tools unchanged. Must return a list.
        "tool_filter": spec.get("tool_filter"),
    }
    _providers[name] = norm
    return norm


def get_provider(name):
    return _providers.get(name)


def provider_names():
    return list(_providers)


def _provider_label(cfg) -> str:
    """The display label for the active backend (status row / info / usage).
    A provider may supply location(cfg) for a richer label (ollama: host alias +
    url); otherwise it's the provider name, or 'no provider' when none is on."""
    spec = get_provider(cfg.provider) if cfg.provider else None
    if spec is not None and spec.get("location"):
        try:
            lbl = spec["location"](cfg)
            if lbl:
                return lbl
        except Exception:
            pass
    return cfg.provider or "no provider"


def _provider_uses_settings(cfg) -> bool:
    """True when the active provider drives reasoning via the 'thinking'/'effort'
    per-provider settings (hosted APIs that declare those capabilities); False when
    it uses the plain boolean `think` toggle (ollama). Replaces the old on_ollama
    branch so think/effort display the right knob for any backend."""
    spec = get_provider(cfg.provider) if cfg.provider else None
    if spec is None:
        return False
    try:
        caps = spec["capabilities"](cfg.active_model()) or {}
    except Exception:
        caps = {}
    return "thinking" in caps or "effort" in caps


def register_command(cmd, handler, help_line="", completer=None):
    """Register a lean-tool slash command. Called from a lean-tool's setup() (usually
    as lc['register_command']). Built-in names are refused (built-ins win), and
    the name is added to SLASH_COMMANDS for Tab-completion. Idempotent-ish: a
    re-register of the same name replaces its handler. Pass `completer(agent, cfg)
    -> [str]` to Tab-complete the command's first argument."""
    if not cmd.startswith("/"):
        cmd = "/" + cmd
    if cmd in _BUILTIN_COMMANDS:
        print(yellow(f"lean-tool command {cmd} ignored: shadows a built-in"))
        return
    owner, prev = _registering_owner, _lean_tool_cmd_owner.get(cmd)
    if cmd in _lean_tool_commands and owner is not None and prev is not None and owner != prev:
        # a DIFFERENT plugin wants a name another already took: keep the first, warn
        # (matches duplicate tool/provider NAMES). Same owner re-registering (e.g. its
        # setup re-runs) still replaces. Direct calls with no owner context replace too.
        print(yellow(f"command {cmd} from '{owner}' ignored: already registered by '{prev}'"))
        return
    _lean_tool_commands[cmd] = (handler, help_line)
    _lean_tool_cmd_owner[cmd] = owner if owner is not None else prev
    if completer is not None:
        _lean_tool_completers[cmd] = completer
    elif cmd in _lean_tool_completers:        # re-register without one clears it
        del _lean_tool_completers[cmd]
    if cmd not in SLASH_COMMANDS:
        SLASH_COMMANDS.append(cmd)

# (command, description) pairs. Rendered with the command bold+cyan and the
# description dimmed, so the two are easy to tell apart even when a narrow
# (mobile) terminal wraps the line.
HELP_COMMANDS = [
    ("/clear", "wipe conversation, stay in this session"),
    ("/new [name]", "start a separate session"),
    ("/compact [keep]", "programmatic: stub old tool outputs, keep newest [keep]"),
    ("/handover", "agentic: summarize + commit docs, replace history (the lever auto-handover pulls)"),
    ("/save [name]", "name the current session"),
    ("/load [name]", "resume a session (no arg = picker)"),
    ("/session", "list | delete <name>"),
    ("/prompt [name]", "view/edit prompt files"),
    ("/sh [cmd]", "run a command in a terminal (no arg: drop into your $SHELL)"),
    ("/connect [host]", "run tools on a remote over SSH"),
    ("/machines", "manage saved remote hosts (list; remove <name>)"),
    ("/local [host]", "detach the active remote"),
    ("/tools", "enable/disable lean-tools"),
    ("/mcp", "manage MCP servers (list/add/remove/reconnect; no arg = enable/disable menu)"),
    ("/reload", "reload lean-tools"),
    ("/model [name]", "switch model (no arg = list)"),
    ("/provider [name]", "switch/manage the model backend"),
    ("/usage", "session tokens + context / provider usage ('raw' dumps the full payload)"),
    ("/think [level]", "set thinking (no arg = menu)"),
    ("/effort [level]", "set reasoning effort (no arg = menu)"),
    ("/set [key val]", "edit app config (config.toml knobs; no arg = menu)"),
    ("/approve [mode]", "ask | session | auto"),
    ("/leash [level]", "ceiling: chat | r | rw | rwe"),
    ("/autosave [on|off]", "autosave + auto-load last on start"),
    ("/incognito [on|off]", "don't save the session locally"),
    ("/askread [on|off]", "confirm read tools too"),
    ("/bg [kill <pid>]", "list/kill background tasks"),
    ("/note [grep|range|add|clear]", "the session notebook (episodic memory); no arg = recent"),
    ("/info", "live session read-out"),
    ("/ctx", "context-token estimate"),
    ("/activity [n|all]", "what the system did automatically (compaction, handover, fallback, ...)"),
    ("/expand [N]", "show a tool call's full args + captured output; bare = newest, N = the #id"),
    ("/help", "this help"),
    ("/quit", "exit"),
]

HELP_FOOTER = ("Tab completes commands and their inline arguments. `/<cmd> ?` shows that "
               "command's own help (args, behaviour). ^C stops a running turn; "
               "at the prompt it cancels the line, and ^C twice in a row exits. "
               "Anything else goes to the model.\n"
               "App-config knobs (context/handover gates, sampling, etc.) live in /set - "
               "e.g. handover_soft/hard/emergency, auto_handover, auto_evict_keep; "
               "backend-specific knobs live in /provider set.")


def render_help(extra=()):
    """Build the /help text: command column bold+cyan, descriptions dimmed.
    `extra` is (command, help_line) pairs for lean-tool-registered commands. The
    description column is aligned, but padding is computed on the raw command
    length so the ANSI color codes don't throw the alignment off."""
    rows = list(HELP_COMMANDS) + list(extra)
    # Align the description column to the longest command, but CAP it: one very long
    # command (e.g. /provider's full arg list) must not pad every other row out to its
    # width and wrap the whole table. Commands over the cap just get 2 trailing spaces.
    ALIGN_CAP = 24
    width = min(max((len(c) for c, _ in rows), default=0), ALIGN_CAP)

    def row(c, d):
        pad = max(2, width - len(c) + 2)
        return f"  {bold(cyan(c))}{' ' * pad}{dim(d)}"

    out = [bold("commands:")] + [row(c, d) for c, d in HELP_COMMANDS]
    if extra:
        out += [bold("lean-tool commands:")] + [row(c, d) for c, d in extra]
    out.append(dim(HELP_FOOTER))
    return "\n".join(out)


_PROMPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _refresh_system_prompt(agent, name):
    """If the live system prompt changed, rebuild messages[0] so it applies now."""
    if name == "system" and agent.messages and agent.messages[0].get("role") == "system":
        agent.messages[0] = {"role": "system", "content": agent._system()}


def _edit_prompt_file(agent, cfg, name):
    if not _PROMPT_NAME_RE.match(name):
        print(yellow(f"invalid prompt name '{name}' (use letters, digits, . _ -)"))
        return
    f = seed_prompt_file(name)
    if open_in_editor(f, cfg):
        print(dim(f"saved {f}"))
    else:
        print(yellow(f"no editor available - edit this file directly:\n  {f}"))
    _refresh_system_prompt(agent, name)


def _use_prompt(agent, name):
    """/prompt use <name> - inject a saved prompt's text as the NEXT user turn (one-shot).
    Lets you keep reusable instructions (a refactor brief, a review checklist, a commit-
    message style) as a named prompt and fire one on demand without retyping it."""
    text = read_prompt(name)
    if not text or not text.strip():
        f = prompt_file(name)
        if name in BUILTIN_PROMPTS:
            print(yellow(f"'{name}' is a system prompt, not a one-shot message - "
                         f"/prompt use is for your own custom prompts."))
        else:
            print(yellow(f"no prompt '{name}' to use (create it with /prompt {name}, "
                         f"or drop a {f.name} in {PROMPTS_DIR})."))
        return
    agent._queued_turns.append(text.rstrip("\n"))
    print(dim(f"queued prompt '{name}' ({len(text.split())} words) -> sending as the next turn."))


def handle_prompt_command(agent, cfg, arg):
    """/prompt: view/edit prompt files, or fire one as a one-shot turn.
      /prompt                 menu
      /prompt <name>          edit (create a custom prompt if absent)
      /prompt use <name>      inject a saved custom prompt as the next turn (one-shot)
      /prompt reset <name>    restore a built-in to its baked default"""
    arg = arg.strip()
    if arg.startswith(("reset ", "restore ")):
        name = arg.split(maxsplit=1)[1].strip()
        if restore_prompt(name):
            print(dim(f"restored '{name}' to its default."))
            _refresh_system_prompt(agent, name)
        else:
            print(dim(f"'{name}' has no override to restore (already default)."))
        return
    if arg.startswith("use "):
        _use_prompt(agent, arg[4:].strip())
        return
    if arg.startswith("edit "):
        arg = arg[5:].strip()
    if arg:
        _edit_prompt_file(agent, cfg, arg)
        return
    # no arg -> menu. Only the user's own prompts (+ any system prompt they've already
    # overridden) are listed; the untouched internal system prompts stay hidden but
    # remain editable by name (e.g. /prompt system) - hinted below.
    builtins, custom = list_prompts()
    names = builtins + custom
    print(bold("prompts:"))
    if not names:
        print(dim("  (none yet - type a new name to create a custom prompt)"))
    for i, n in enumerate(names, 1):
        kind = "system" if n in BUILTIN_PROMPTS else "custom"
        edited = ", edited" if prompt_file(n).is_file() else ""
        print(f"  {i}) {n}" + dim(f"  ({kind}{edited})"))
    print(dim("  number = edit  |  'u <n>' = use as a one-shot turn  |  "
              "'r <n>' = restore a system default  |  a new name = create  |  blank = cancel"))
    _sys_hidden = sorted(n for n in BUILTIN_PROMPTS if not prompt_file(n).is_file())
    if _sys_hidden:
        print(dim(f"  advanced: system prompts ({', '.join(_sys_hidden)}) - "
                  f"edit by name, e.g. /prompt {_sys_hidden[0]}"))
    try:
        sel = input("prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not sel:
        return
    if sel.startswith("r ") and sel[2:].strip().isdigit():
        i = int(sel[2:].strip())
        if 1 <= i <= len(names):
            nm = names[i - 1]
            if restore_prompt(nm):
                print(dim(f"restored '{nm}'."))
                _refresh_system_prompt(agent, nm)
            else:
                print(dim(f"'{nm}' already default."))
        return
    if sel.startswith("u ") and sel[2:].strip().isdigit():
        i = int(sel[2:].strip())
        if 1 <= i <= len(names):
            _use_prompt(agent, names[i - 1])
        return
    name = names[int(sel) - 1] if (sel.isdigit() and 1 <= int(sel) <= len(names)) else sel
    _edit_prompt_file(agent, cfg, name)


# Config fields /set can edit directly. (model/host/num_ctx have dedicated
# live commands - /model, /ollama host - that also re-init the client; /set is the
# granular editor for the rest of the persisted knobs.) kind: str|int|float|bool, or
# a tuple of allowed values (picker).
_SETTINGS_FIELDS = [
    ("temperature", "temperature", "float"),
    ("top_p", "top_p", "float"),
    ("top_k", "top_k", "int"),
    ("repeat_penalty", "repeat_penalty", "float"),
    ("editor", "editor (for /prompt)", "str"),
    ("user_name", "your name on your own turn in scrollback", "str"),
    ("approval", "approval mode", tuple(APPROVAL_MODES)),
    ("composer", "composer (pinned input)", "bool"),
    ("autosave", "autosave session (auto-load last on start)", "bool"),
    ("auto_update", "on launch, self-update to the latest published build (needs the 'update' lean-tool)", "bool"),
    ("update_track", "which /update track to follow", ("stable", "beta")),
    ("command_timeout", "run_command timeout (s)", "int"),
    ("max_iterations", "max tool-call rounds per turn (0 = unlimited)", "int"),
    ("bg_max_concurrent", "max background tasks (0 = unlimited)", "int"),
    ("worker_max_concurrent", "max worker agents at once (0 = unlimited)", "int"),
    ("worker_idle_timeout", "worker lease timeout (s); unattended self-kill", "int"),
    ("worker_max_iterations", "max tool-call rounds per worker", "int"),
    ("ask_user_to_run", "ask_user_to_run tool", "bool"),
    ("leash", "capability ceiling", tuple(LEASH_LEVELS)),
    ("confirm_reads", "ask-on-read (confirm read tools too)", "bool"),
    ("auto_reconnect", "reconnect to a session's remote on load (off = ask)", "bool"),
    ("ephemeral", "/connect default: force embed runtime + wipe on teardown (zero-trace)", "bool"),
    ("show_snapshots", "show pre-handover safety snapshots in the /load picker + /session list", "bool"),
    ("statusline", "status rows above the prompt", "bool"),
    ("statusline_every", "reprint status every N prompts (1 = every turn, 0 = only on change)", "int"),
    ("statusline_iter", "reprint status every N model iterations within a long turn (0 = off)", "int"),
    # --- context management (send-window + auto-handover) ---
    ("window_messages", "send-window size in messages (0 = off, full history)", "int"),
    ("window_tokens", "send-window token cap: 'auto' (=ctx-reserve, default), an int (hard cap), or 0 (off)", "int_or_auto"),
    ("auto_handover", "auto-handover (self-managing context)", "bool"),
    ("handover_soft", "handover soft-zone start (fraction of ctx)", "float"),
    ("handover_hard", "handover hard threshold (fraction of ctx)", "float"),
    ("auto_num_ctx", "ollama: detect num_ctx at startup", "bool"),
    ("gen_connect_timeout", "ollama: TCP connect deadline (s)", "float"),
    ("gen_ttft_timeout", "ollama: read deadline until first token (s)", "float"),
    ("gen_idle_timeout", "ollama: max gap between tokens (s; blank = off)", "float"),
    ("handover_min_interval", "min seconds between handovers", "float"),
    ("autostart_after_handover", "auto-continue the turn after a handover (on by default; 5s ^C to cancel)", "bool"),
    ("wake_on_bg_finish", "wake + react autonomously when a background task finishes (off by default)", "bool"),
    ("auto_compact_interval", "auto-compact: strip old tool outputs every N tokens (0 = off)", "int"),
    ("auto_compact_hysteresis", "auto-compact re-arm margin (fraction of interval)", "float"),
    ("auto_compact_keep", "tool results kept in full by an auto-compact strip", "int"),
    ("notes_spool", "note memory: keep the newest N lines (spooling log; default 2000)", "int"),
    ("auto_evict", "continuous tool-output eviction", "bool"),
    ("auto_evict_keep", "tool results kept verbatim by auto_evict", "int"),
    ("keep_alive", "ollama: model keep-alive (e.g. 10m, -1, 0)", "str"),
    ("auto_num_ctx", "ollama: detect num_ctx at startup", "bool"),
]
_SETTINGS_BY_KEY = {k: (label, kind) for k, label, kind in _SETTINGS_FIELDS}
# Settings that change the LIVE agent (its tool surface / system prompt), not just a
# cfg value read each turn - editing these via /set must re-apply them, exactly
# as the dedicated command does, or the change is cosmetic (e.g. /set leash r
# would show 'r' but leave the model holding the full tool surface). They announce
# themselves on apply, so the generic "key -> value" echo is skipped for them.
_LIVE_APPLY = {"leash"}


def _coerce_setting(raw, kind):
    """Parse a raw string to the field's type; raise ValueError on a bad value."""
    if kind == "int":
        return int(raw)
    if kind == "int_or_auto":              # 'auto' sentinel OR an int (e.g. window_tokens)
        if isinstance(raw, str) and raw.strip().lower() == "auto":
            return "auto"
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return raw.strip().lower() in ("1", "true", "yes", "on", "y")
    return raw


def _set_setting_field(agent, cfg, key, raw, scope="session"):
    """Set one /set config field from a raw string, APPLYING it to the live agent when
    the field affects its tool surface or system prompt (leash, ask_user_to_run) -
    not just the cfg value. Returns True on success. `agent` may be None for headless
    callers (then live-apply is skipped).

    scope controls where the value lands (the uniform DEFAULTS/OVERRIDE model):
      "session" (default): a per-session OVERRIDE - live cfg + cfg.session_overrides[key]
                 (carried in the session meta by autosave). config.toml is NOT touched.
      "config":  the config.toml DEFAULT - live cfg + cfg._defaults[key], AND the session
                 override for that key is CLEARED so the new default shows through. The
                 caller persists via save_config."""
    if key not in _SETTINGS_BY_KEY:
        print(yellow(f"unknown setting '{key}' (try: {', '.join(_SETTINGS_BY_KEY)})"))
        return False
    _, kind = _SETTINGS_BY_KEY[key]
    if isinstance(kind, tuple) and raw not in kind:
        print(yellow(f"{key} must be one of: {', '.join(kind)}"))
        return False
    try:
        val = raw if isinstance(kind, tuple) else _coerce_setting(raw, kind)
    except ValueError:
        print(yellow(f"invalid value for {key}: {raw!r}"))
        return False
    if key == "approval":
        set_approval(cfg, val)
    elif key == "leash" and agent is not None:
        _apply_leash(agent, cfg, val, persist=(scope == "config"))  # live rebuild
    else:
        setattr(cfg, key, val)
        # ask_user_to_run adds/removes a tool, so the live surface must rebuild too.
        if key == "ask_user_to_run" and agent is not None:
            agent.refresh_tools()
    # Record where the value belongs in the DEFAULTS/OVERRIDE layer.
    if scope == "config":
        cfg._defaults[key] = getattr(cfg, key)   # the new config.toml default
        cfg.session_overrides.pop(key, None)      # let the new default show through
    else:
        cfg.session_overrides[key] = getattr(cfg, key)  # per-session override
    return True


def _edit_one_setting(agent, cfg, k, label, kind):
    """Interactively edit a single /set config field: flip a bool, pick a tuple value,
    prompt a scalar. Routes through _set_setting_field (so live-apply settings take
    effect), then persists + echoes. Shared by the /set menu and `/set <key>`
    with no value (so a bare key edits like picking it in the menu)."""
    cur = getattr(cfg, k)
    if kind == "bool":
        ok = _set_setting_field(agent, cfg, k, "off" if cur else "on")
    elif isinstance(kind, tuple):
        choice = pick_one(label + ":", list(kind), cur)
        if not choice:
            return
        ok = _set_setting_field(agent, cfg, k, choice)
    else:
        try:
            raw = input(f"{label} [{cur}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            return
        ok = _set_setting_field(agent, cfg, k, raw)
    if ok:
        save_config(cfg, quiet=True)
        if k not in _LIVE_APPLY:             # live-apply keys announce themselves
            print(dim(f"{k} -> {getattr(cfg, k)}"))


def _settings_sessions_view(agent, cfg):
    """The settings 'sessions' item: list saved sessions most-recently-used first
    with timestamps + relative age, then offer the load picker."""
    rows = _session_rows()
    if not rows:
        print(dim(f"  no saved sessions yet.  dir: {SESSIONS_DIR}"))
        return
    now = time.time()
    d = GLYPH["dot"]
    state = "on" if cfg.autosave else "off (amnesic)"
    print(bold("sessions") + dim(f"  (most-recently-used first; autosave {state})"))
    for name, meta, mtime in rows:
        when = meta.get("saved_at", "?")
        turns = meta.get("turns", "?")
        age = f"{_fmt_age(now - mtime)} ago" if mtime else "?"
        print(f"  {bold(cyan(name))}  {dim(f'{when} {d} {age} {d} {turns} turns')}")
    name = _session_picker("load session (enter to skip):")
    if name:
        _load_session_into(agent, cfg, name)


def handle_settings_command(agent, cfg, arg):
    """/set - granular editor over the working state. No arg: interactive menu.
    `key value`: a per-SESSION override (live + carried in the session, config.toml
    untouched). `--config key value`: write that ONE key as the config.toml DEFAULT and
    clear its session override (so the new default shows through). `--config key` with
    no value: clear the session override for that key (revert to the config default).
    (Provider-specific backend knobs live in /provider set.)"""
    if arg:
        to_config = False
        if arg.split(maxsplit=1)[0] in ("--config", "--default"):
            to_config = True
            arg = arg.split(maxsplit=1)[1] if len(arg.split(maxsplit=1)) > 1 else ""
        parts = arg.split(maxsplit=1)
        if to_config and len(parts) == 1 and parts[0] in _SETTINGS_BY_KEY:
            # `/set --config key` (no value): drop the session override so the live cfg
            # reverts to the config.toml default for that key.
            key = parts[0]
            cfg.session_overrides.pop(key, None)
            if key in cfg._defaults:
                if key == "leash" and agent is not None:
                    _apply_leash(agent, cfg, cfg._defaults[key], persist=False)
                elif key == "approval":
                    set_approval(cfg, cfg._defaults[key])
                else:
                    setattr(cfg, key, cfg._defaults[key])
                    if key == "ask_user_to_run" and agent is not None:
                        agent.refresh_tools()
            print(dim(f"{key} -> {getattr(cfg, key)}  (session override cleared)"))
            return
        scope = "config" if to_config else "session"
        if len(parts) == 2 and _set_setting_field(agent, cfg, parts[0], parts[1], scope=scope):
            if to_config:
                save_config(cfg, quiet=True)
            else:
                autosave_session(agent, cfg)      # carry the override in the session
            if parts[0] not in _LIVE_APPLY:       # live-apply keys announce themselves
                tag = "" if to_config else "  (session override)"
                print(dim(f"{parts[0]} -> {getattr(cfg, parts[0])}{tag}"))
        elif len(parts) == 1:                     # bare key -> edit that one field
            if parts[0] in _SETTINGS_BY_KEY:
                label, kind = _SETTINGS_BY_KEY[parts[0]]
                _edit_one_setting(agent, cfg, parts[0], label, kind)
            else:
                print(yellow(f"unknown setting '{parts[0]}' "
                             f"(try: {', '.join(_SETTINGS_BY_KEY)})"))
        return
    sessions_item = len(_SETTINGS_FIELDS) + 1
    while True:
        print(bold("settings") + dim("  (config.toml; model/host via /model, /ollama host)"))
        for i, (k, label, kind) in enumerate(_SETTINGS_FIELDS, 1):
            print(f"  {i}) {label}: " + cyan(str(getattr(cfg, k))))
        print(f"  {sessions_item}) sessions" + dim("  (view recent / load)"))
        print("  0) done")
        try:
            sel = input("edit #: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if sel in ("", "0"):
            break
        if sel == str(sessions_item):
            _settings_sessions_view(agent, cfg)
            continue
        if not (sel.isdigit() and 1 <= int(sel) <= len(_SETTINGS_FIELDS)):
            continue
        k, label, kind = _SETTINGS_FIELDS[int(sel) - 1]
        _edit_one_setting(agent, cfg, k, label, kind)


def _toggle_provider_plugin(agent, cfg, mgr, name):
    """Enable/disable one provider plugin: enabling runs its helpers-only setup() +
    register_provider now; disabling deactivates it (if active) and drops it from the
    registry. Persists providers_enabled."""
    if name in cfg.providers_enabled:                       # -> disable
        cfg.providers_enabled = [n for n in cfg.providers_enabled if n != name]
        if cfg.provider == name:
            agent.deactivate_provider()
        _providers.pop(name, None)
        print(dim(f"{name}: off"))
        if name == "ollama":                                # the bundled default floor
            print(dim("  (ollama is the bundled default; it re-enables on next launch)"))
    else:                                                   # -> enable
        p = mgr.providers.get(name)
        try:
            if callable(p["setup"]):
                p["setup"](globals(), cfg)
            register_provider(p["spec"])
        except Exception as e:
            print(yellow(f"enable failed: {e}"))
            return
        cfg.providers_enabled = sorted(set(cfg.providers_enabled) | {name})
        print(dim(f"{name}: on  (/model to use it)"))
    autosave_config(cfg)


def handle_providers_command(agent, cfg, arg):
    """The providers/ catalog manager (reached via `/provider list` and, with a name,
    `/provider enable|disable`). No arg: a toggle menu; `<name>` toggles one. Enabling
    registers it now; disabling deactivates it (if active) and removes it."""
    mgr = ProviderManager(_provider_dirs(cfg), enabled=cfg.providers_enabled)
    if not mgr.names():
        print(dim(f"no provider plugins in {_providers_dir(cfg)} "
                  f"(drop a .py with a PROVIDER dict there)."))
        return
    if arg.strip():
        name = arg.strip()
        if name not in mgr.names():
            print(dim(f"no such provider '{name}' (have: {', '.join(mgr.names())})"))
            return
        _toggle_provider_plugin(agent, cfg, mgr, name)
        return
    while True:
        print(bold("provider plugins") + dim(f"  ({_providers_dir(cfg)})"))
        names = mgr.names()
        for i, n in enumerate(names, 1):
            mark = green("on ") if n in cfg.providers_enabled else dim("off")
            d = mgr.desc(n)
            # mark the backend actually RUNNING now: [on]/[off] is the enabled set, which
            # doesn't tell you which enabled provider is active (the /model choice does).
            active = green("  (active)") if n == cfg.provider else ""
            print(f"  {i}) [{mark}] {bold(cyan(n))}" + (dim(f"  {d}") if d else "") + active)
        print("  0) done")
        try:
            sel = input("toggle #: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if sel in ("", "0"):
            break
        if sel.isdigit() and 1 <= int(sel) <= len(names):
            _toggle_provider_plugin(agent, cfg, mgr, names[int(sel) - 1])


def handle_bg_command(agent, cfg, arg):
    """/bg - list background tasks; /bg kill <pid|all> - stop them. When the session
    is /connect'ed the tasks run in the REMOTE executor and its registry lives on that
    box, so both list + kill are routed there (via BG_LIST / a remote kill); a LOCAL
    session reads its own registry directly."""
    remote = getattr(agent, "remote", None)
    parts = arg.split()
    if parts and parts[0].lower() == "kill":
        tgt = parts[1] if len(parts) > 1 else ""
        if remote is not None:
            rows = [r for r in remote.bg_list() if r.get("state") == "running"]
            if tgt == "all":
                pids = [r.get("pid") for r in rows]
            elif tgt.isdigit() and any(r.get("pid") == int(tgt) for r in rows):
                pids = [int(tgt)]
            else:
                print(dim("usage: /bg kill <pid|all>"
                          + ("" if rows else "  (no background tasks running)")))
                return
            for pid in pids:
                remote.call("run_command", {"cmd": f"kill {int(pid)} 2>/dev/null || true"})
            print(dim(f"killed {len(pids)} background task(s) on {remote.host}."))
            return
        running = _bg_running()
        if tgt == "all":
            killed = [r.get("pid") for r in running]
        elif tgt.isdigit() and any(r.get("pid") == int(tgt) for r in running):
            killed = [int(tgt)]
        else:
            print(dim("usage: /bg kill <pid|all>"
                      + ("" if running else "  (no background tasks running)")))
            return
        for pid in killed:
            _bg_kill(pid)
        _bg_save([r for r in _bg_load() if r.get("pid") not in set(killed)])
        print(dim(f"killed {len(killed)} background task(s)."))
        return
    if remote is not None:
        rows = remote.bg_list()
        if not rows:
            print(dim(f"no background tasks on {remote.host}.  "
                      f"(end a command with ' &' to start one)"))
            return
        print(bold(f"background tasks (remote {remote.host}):"))
        for r in rows:
            st = r.get("state", "?")
            label = green("running") if st == "running" else (
                green(st) if st == "exit 0" else red(st))
            print(f"  {bold(cyan(str(r.get('pid'))))}  {label}  "
                  f"(ran {r.get('runtime', '?')})  {r.get('cmd', '?')}")
            if st == "running":
                print(dim(f"      /bg kill {r.get('pid')}"))
        return
    me = os.getpid()
    recs = _bg_load()
    running = [r for r in recs if _proc_alive(r.get("pid"))]
    # finished = registered, not alive, but wrote an exit sidecar (so we can report
    # pass/fail). Entries with no sidecar have simply gone and aren't worth listing.
    finished = [r for r in recs if not _proc_alive(r.get("pid"))
                and _bg_exit_code(r) is not None]
    if not running and not finished:
        print(dim("no background tasks running.  (end a command with ' &' to start one)"))
        return
    print(bold("background tasks:"))
    for r in running:
        tag = "" if r.get("owner") == me else dim("  (other session)")
        print(f"  {bold(cyan(str(r.get('pid'))))}  {green('running')}  "
              f"{dim(r.get('started', '?'))}  {r.get('cmd', '?')}{tag}")
        print(dim(f"      log: {r.get('log', '?')}   ·   /bg kill {r.get('pid')}"))
    for r in finished:
        code = _bg_exit_code(r)
        stat = green("exit 0") if code == "0" else red(f"exit {code}")
        print(f"  {dim(str(r.get('pid')))}  {stat}  "
              f"{dim(r.get('started', '?'))}  {r.get('cmd', '?')}")
        print(dim(f"      log: {r.get('log', '?')}"))


def handle_note_command(agent, cfg, arg):
    """/note - show the session notebook (recent entries); the same episodic memory
    the model keeps via the note tool. Subcommands:
      /note                 recent entries (capped)
      /note <text>          add a timestamped entry yourself
      /note add <text>      add (explicit)
      /note grep <pattern>  entries matching a substring/regex
      /note range <from> [to]   entries in a time window (ts 'YYYY-MM-DD HH:MM'; bare date = whole day)
      /note clear           wipe the notebook for this session
    Notes live with the session (saved + restored on /load), so /note lets you read +
    curate what the agent has recorded - the same way /bg surfaces background tasks."""
    parts = arg.split()
    sub = parts[0].lower() if parts else ""
    if sub == "clear":
        n = len(agent.notes)
        agent.notes = []
        print(dim(f"notebook cleared ({n} entr{'y' if n == 1 else 'ies'} removed)."))
        return
    if sub == "grep":
        out = agent._note({"action": "grep", "pattern": " ".join(parts[1:])})
    elif sub == "range":
        out = agent._note({"action": "range",
                           "from": parts[1] if len(parts) > 1 else "",
                           "to": parts[2] if len(parts) > 2 else ""})
    elif sub == "add":
        out = agent._note({"action": "add", "text": "operator: " + arg[len("add"):].strip()})
    elif sub == "recent" or not arg.strip():
        out = agent._note({"action": "recent",
                           "n": parts[1] if sub == "recent" and len(parts) > 1 else None})
    else:                                     # bare text -> add it. Prefix 'operator: ' in
        # plain language (not a coded tag) so even a small model reads it as "the operator
        # said this" - and can weight it - while the model's OWN notes stay bare.
        out = agent._note({"action": "add", "text": "operator: " + arg.strip()})
    print(out)


def _provider_usage_str(agent, cfg, verbose=False) -> str:
    """The active backend's usage/quota tail (e.g. the '5h .. wk ..' meters), already
    styled, or '' if the backend reports none. Core does the rendering so every
    backend reads the same. Shared by the ctx meter and the status rows. verbose=True
    (used by /info) also shows the reset time/countdown, kept off the tidy status line."""
    spec = agent.active_provider()
    if spec is None:
        return ""
    try:
        if spec["usage"]:
            return render_usage_meters(spec["usage"](agent, cfg), verbose=verbose)
        if spec["status_line"]:
            s = (spec["status_line"](agent, cfg) or "").strip()
            return dim("  " + GLYPH["dot"] + "  ") + s if s else ""
    except Exception:
        return ""
    return ""


def _short_model(name, n=20):
    """Truncate a long model name for the status row (full name stays in /info)."""
    name = name or "?"
    return name if len(name) <= n else name[:n - 1] + GLYPH["ellipsis"]


def _status_rows(agent, cfg):
    """The status rows printed above each prompt so the live state is never invisible:
      1. session (name + autosave / incognito / amnesic) + condensed model @ provider, + pinned plan
      2. perms + context management (leash, approve, round-cap in auto, window, compact, tools)
      3. ctx + the backend quota meters (5h / wk) + think/effort
    Toggles render only when active so the rows stay short. Returns [] when disabled
    (cfg.statusline off) or off a TTY. Just strings - colour helpers no-op off a TTY."""
    if not getattr(cfg, "statusline", True) or not _TTY:
        return []
    d = dim(f" {GLYPH['dot']} ")
    rows = []
    model_at = f"{_short_model(cfg.active_model())} @ {_provider_label(cfg)}"

    # 1) session + model (at the top - what/where you're running). The live turn
    # counter sits where 'autosaving' used to (autosave state -> /info now); it's a
    # volatile token _status_key strips so 'changed-only' mode still gates correctly.
    turns = _user_turns(agent.messages)
    if cfg.incognito:
        s = [magenta(f"{GLYPH['ghost']} incognito")]
    elif cfg.autosave:
        s = [f"session: {agent.autosave_name}"]
    else:
        s = ["amnesic"]
    s.append(f"{turns} turns")
    s.append(model_at)
    rows.append("  " + d.join(s))

    # 2) perms + context management
    if not agent._model_tool_support():       # model can't tool-call -> chat-only, leash moot
        # yellow so the USER notices it's the model (not a setting) and stops pushing it to tool
        p = [yellow("chat-only (model)")]
        if cfg.leash != "chat":
            p.append(dim(f"leash {cfg.leash} n/a"))   # the ceiling can't grant tools the model lacks
        p.append(f"approve: {cfg.approval}")
    else:
        p = [bold(cfg.leash), f"approve: {cfg.approval}"]
    if cfg.approval == "auto":                # the round-cap only bites when unattended
        mi = cfg.max_iterations
        p.append(f"max {mi} rounds" if mi else "no round cap")
    if cfg.confirm_reads:
        p.append("ask-read")
    _wt = cfg.window_tokens
    if isinstance(_wt, str) and _wt.strip().lower() == "auto":
        p.append("window auto")               # backstop: ctx - reply reserve, recomputed
    elif isinstance(_wt, int) and _wt > 0:    # a hard token cap
        p.append(f"window {_wt}tok")
    elif cfg.window_messages > 0:             # only show when on - don't nag with 'window off'
        p.append(f"window {cfg.window_messages}")
    c = cfg.handover_for()
    p.append(f"handover {c['hard'] * 100:.0f}%" if c.get("auto") else "handover off")
    p.append(f"{len(agent.tool_defs)} tools")
    rows.append("  " + d.join(p))

    # 3) context (+ think/effort + backend quota tail)
    if _provider_uses_settings(cfg):
        think, effort = cfg.setting("thinking") or "na", cfg.setting("effort") or "na"
    else:
        think, effort = {True: "on", False: "off"}.get(cfg.think, "na"), "na"
    used, window, zone, pct = agent._ctx_summary(cfg)
    # '~' marks a pre-call ESTIMATE (no live API measurement yet, e.g. right after a
    # load): _ctx_used falls back to a transcript estimate that reads lower than the
    # cache-inclusive measured size. Without the marker a fresh load looks like ctx
    # dropped/was lost. It firms up to the measured value on the next turn.
    est = "~" if not getattr(agent, "last_prompt_tokens", 0) else ""
    # drop the 'ok' zone label - only surface a zone once it's noteworthy (soft/hard/emergency)
    zlabel = "" if zone == "ok" else f" {zone}"
    # Colour by the auto-handover ZONE (what actually drives behaviour) when it's on -
    # so the meter goes yellow/red at the soft/hard thresholds, not at a fixed % of the
    # max window. Falls back to plain %-of-max colour when auto-handover is off.
    col = _zone_color(zone) if c.get("auto") else _pct_color(pct)
    ctx = col(f"ctx {est}{_fmt_tokens(used)}/{_fmt_tokens(window)} ({pct:.0f}%{zlabel})")
    quota = _provider_usage_str(agent, cfg)     # leading separator already, or "" if none
    # handovers is a live counter (row 3 reprints every turn); shown once any happened.
    hov = (d + f"{agent.handovers} handovers") if agent.handovers else ""
    # ctx -> handover count -> the backend quota meters (5h / wk) -> think / effort
    rows.append("  " + ctx + hov + quota + d + d.join([f"think {think}", f"effort {effort}"]))
    return rows


def _status_key(rows):
    """Signature the repl uses to decide when to REPRINT the stable status rows
    (session/model + perms/tools). The LAST row - the live context meter - is excluded
    because it prints every turn anyway; so per-turn ctx/token/zone drift never triggers
    a reprint, only a real settings/model/tools/plan change (in rows 1-2) does. The live
    'N turns' token on row 1 is also stripped (it changes every turn, and would defeat
    the changed-only gate otherwise). Pure."""
    if not rows:
        return ()
    stable = list(rows[:-1])
    if stable:                       # drop the volatile 'N turns' token from row 1
        stable[0] = _STATUS_TURNS_RE.sub("", stable[0])
    return tuple(stable)


def handle_info_command(agent, cfg, arg=""):
    """/info - full live read-out: model, backend, context + how it's managed
    (send-window / auto-compaction), session + which instance holds the write-lock,
    pinned plan, perms (leash/approval), thinking/effort, and the enabled modes. The
    point is that nothing about the session's behaviour is invisible."""
    where = f"remote {agent.remote.host}" if agent.remote else "local"
    backend = _provider_label(cfg)
    used, window, zone, _pct = agent._ctx_summary(cfg)
    turns = _user_turns(agent.messages)
    if _provider_uses_settings(cfg):
        think, effort = cfg.setting("thinking") or "off", cfg.setting("effort") or "-"
    else:
        think, effort = {True: "on", False: "off"}.get(cfg.think, "unset"), "-"
    dot = f"  {GLYPH['dot']}  "
    badge = approval_badge(cfg)
    print(f"  version:  {__version__}{dot}track: {getattr(cfg, 'update_track', 'stable')}")
    print(f"  model:    {cfg.active_model()}")
    print(f"  provider: {backend}  ({where})")
    hov = f", {agent.handovers} handovers" if agent.handovers else ""
    print(f"  context:  ~{_fmt_tokens(used)}/{_fmt_tokens(window)} "
          f"({used / window * 100:.0f}%, {zone}){dot}{len(agent.messages)} msgs, {turns} turns{hov}")
    # how context is managed - the two things that surprised people
    wm = cfg.window_messages
    print(f"  window:   {'off (full history sent)' if wm <= 0 else f'last {wm} messages sent'}")
    c = cfg.handover_for()
    if c.get("auto"):
        comp = (f"auto on{dot}soft {c['soft']:.0%}  hard {c['hard']:.0%}"
                + (f"{dot}autostart" if c.get("autostart") else ""))
    else:
        comp = "off"
    print(f"  handover: {comp}")
    # session + which instance owns the write-lock (so a fork is never a surprise)
    if cfg.incognito:
        sess = magenta(f"{GLYPH['ghost']} incognito") + dim(" (not saved locally; provider may still log)")
    elif cfg.autosave:
        if _own_lock(agent.autosave_name):
            lock = "you hold the lock"
        elif _lock_is_live(agent.autosave_name):
            lock = yellow("held by another instance")
        else:
            lock = "unlocked"
        sess = f"autosaving -> {agent.autosave_name}{dot}{lock}"
    else:
        sess = "amnesic (autosave off)"
    print(f"  session:  {sess}")
    if getattr(agent, "pinned_plan", ""):
        print(f"  plan:     pinned (GOAL+TODO, carried across compaction + load)")
    print(f"  usage:    in {agent.session_in:,}  out {agent.session_out:,}")
    _quota = _provider_usage_str(agent, cfg, verbose=True)   # backend quota incl. reset times
    if _quota:
        print(f"  quota:    {_quota.lstrip()}")
    print(f"  thinking: {think}{dot}effort: {effort}")
    print(f"  approval: {cfg.approval}" + (f"  {badge}" if badge else ""))
    print(f"  leash:    {cfg.leash}  ({_LEASH_GRANTS[cfg.leash]})")
    modes = []
    if cfg.confirm_reads:
        modes.append("ask-on-read")
    if cfg.auto_reconnect:
        modes.append("auto-reconnect")
    if cfg.incognito:
        modes.append("incognito")
    if modes:
        print(f"  modes:    {', '.join(modes)}")


# Per-command flags a user can add anywhere in the arg (order-agnostic, matching how
# the handlers parse them - e.g. handle_connect_command pulls --ephemeral from any
# position). Kept as completion candidates so the "Tab completes inline arguments"
# contract (HELP_FOOTER) actually covers a command's own flags. `--plain`/`--fallback`
# are universal (dispatch_command strips them from any command) so they're offered too.
_COMMAND_FLAGS = {
    "/connect": ["--ephemeral", "--no-ephemeral"],
}
_UNIVERSAL_FLAGS = ["--plain", "--fallback"]


def _command_flags(cmd):
    """Flags offerable at ANY arg position for `cmd` (its own + the universal ones)."""
    return _COMMAND_FLAGS.get(cmd, []) + _UNIVERSAL_FLAGS


def _arg_completions(agent, cfg, cmd):
    """Inline first-arg Tab completions for a slash command. Reads live state
    (active backend's models, registered providers, hosts, open remotes). Returns
    [] for commands that take no completable arg (the no-arg picker still works).
    Pure w.r.t. the terminal, so it is unit-tested headless."""
    if cmd in ("/model", "/models"):
        try:
            return list(agent.client.list_models())
        except Exception:
            return []
    if cmd in ("/provider", "/providers"):
        return (["off", "on", "login", "clear", "list", "enable", "disable", "set"]
                + provider_names())
    if cmd in ("/think", "/effort"):
        key = "thinking" if cmd == "/think" else "effort"
        default = ["off", "adaptive", "max"] if key == "thinking" else ["low", "med", "high"]
        spec = agent.active_provider()
        return list(_safe_caps(spec, cfg).get(key, default)) if spec else default
    if cmd == "/session":
        return ["list", "delete", "save", "load"]
    if cmd in ("/load", "/delete", "/session delete", "/session load", "/session save"):
        return [n for n, _, _ in _session_rows()]      # saved session names
    if cmd in ("/save", "/new"):
        return [n for n, _, _ in _session_rows()]      # offer existing names (overwrite / reuse)
    if cmd in ("/autosave", "/incognito", "/askread"):
        return ["on", "off"]
    if cmd == "/leash":
        return list(LEASH_LEVELS)
    if cmd == "/bg":
        return ["kill"]
    if cmd == "/note":
        return ["recent", "grep", "range", "add", "clear"]
    if cmd == "/mcp":
        return ["list", "add", "remove", "reconnect", "clean"]
    if cmd in ("/mcp remove", "/mcp reconnect"):   # complete a configured server name
        try:
            return list(agent.mcp.names())
        except Exception:
            return []
    if cmd == "/mcp clean":                         # complete a stale (orphaned) server name
        try:
            return list(agent.mcp.stale_enabled())
        except Exception:
            return []
    if cmd == "/expand":                            # msg -> conversation view (else tool-call ids)
        return ["msg"]
    if cmd == "/connect":
        return ["remove"] + sorted(set(list(cfg.connect_hosts) + list(getattr(agent, "remotes", {}))))
    if cmd == "/connect remove":
        return sorted(cfg.connect_hosts)
    if cmd == "/machines":
        return ["remove"]
    if cmd == "/machines remove":
        return sorted(cfg.machines)
    if cmd in ("/local", "/disconnect"):
        return list(getattr(agent, "remotes", {}))
    if cmd == "/prompt":
        b, c = list_prompts(all_builtins=True)   # completion offers every system prompt by name
        return ["edit", "use", "reset"] + b + c
    if cmd == "/approve":
        return list(APPROVAL_MODES)
    if cmd == "/set":                             # app-config keys (was /settings)
        return [k for k, _, _ in _SETTINGS_FIELDS]
    if cmd.startswith("/set "):                   # value completion for a chosen config key
        toks = cmd.split()
        _, kind = _SETTINGS_BY_KEY.get(toks[1], (None, None)) if len(toks) > 1 else (None, None)
        if isinstance(kind, tuple):
            return list(kind)
        return ["on", "off"] if kind == "bool" else []
    if cmd in ("/provider set", "/providers set"):  # provider-declared backend knobs
        spec = agent.active_provider()
        return list(_safe_caps(spec, cfg)) if spec is not None else []
    if cmd in _lean_tool_completers:          # lean-tool-contributed completion
        try:
            return list(_lean_tool_completers[cmd](agent, cfg))
        except Exception:
            return []
    return []


def _completion_options(agent, cfg, left):
    """Shared Tab-completion: given the text LEFT of the cursor, return the full
    candidate completions for the final token (each a whole token, e.g. '/model').
    Backs both readline's completer and the Composer's. Pure w.r.t. the terminal."""
    buf = left.lstrip()
    tokens = buf.split()
    # the token being completed = "" if the line ends in whitespace, else last token
    text = "" if (buf and buf[-1].isspace()) else (tokens[-1] if tokens else "")
    # the "path" is every token already entered before the one being typed
    path = tokens if (buf and buf[-1].isspace()) else tokens[:-1]
    if path:                                 # past the command -> complete this arg,
        key = " ".join(path)                 # keyed on the full command path
        opts = _arg_completions(agent, cfg, key)
        if not opts and len(path) > 1:       # fall back to the bare command
            opts = _arg_completions(agent, cfg, path[0])
        # `?` is a universal first-arg on every command (per-command help), so offer it
        # alongside a command's own completions when the cursor is on its first argument.
        if len(path) == 1 and path[0] != "/help":
            opts = list(opts) + ["?"]
        # Flags (a command's own + the universal --plain/--fallback) complete at ANY arg
        # position, mirroring the handlers that accept them order-agnostically - so e.g.
        # `/connect 8 --ephe<Tab>` offers --ephemeral. Skip already-present ones.
        if text.startswith("-"):
            flags = [f for f in _command_flags(path[0]) if f not in tokens]
            opts = list(opts) + flags
        return [o for o in opts if o.startswith(text)]
    if buf.startswith("/"):                  # still on the command itself
        return [c for c in SLASH_COMMANDS if c.startswith(text)]
    return []


def _make_composer_completer(agent, cfg):
    """A completer(left_text)->[options] bound to this agent/cfg, for the Composer."""
    return lambda left: _completion_options(agent, cfg, left)


def _setup_readline(agent, cfg):
    """stdlib readline Tab-completion: slash commands, and each command's inline
    first argument (model/provider/effort/session/host/connect names, etc.)."""
    if readline is None:
        return
    readline.set_completer_delims(" \t\n")  # keep "/" and ":" inside one token

    def complete(text, state):
        opts = _completion_options(agent, cfg, readline.get_line_buffer())
        # readline shows the command itself with a trailing space to advance to args
        if opts and opts[0].startswith("/") and " " not in readline.get_line_buffer().strip():
            opts = [o + " " for o in opts]
        return opts[state] if state < len(opts) else None

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")
    # Bracketed paste: make readline treat a pasted block as one literal insert
    # (newlines kept in the line buffer, submitted only on a real Enter) instead
    # of executing it line-by-line - which truncates a multi-line paste to its
    # first line. Only helps if the terminal/multiplexer actually emits the
    # ESC[200~ / ESC[201~ markers; harmless no-op if not. parse_and_bind is
    # wrapped because some libedit builds reject the directive.
    try:
        readline.parse_and_bind("set enable-bracketed-paste on")
    except Exception:
        pass


def handle_save_command(agent, cfg, arg):
    """/save [name] - DEPRECATED alias: autosave already persists the conversation
    (and now its per-session overrides) each turn, so a manual save is rarely needed.
    Still works: it snapshots + names the session (autosave then continues into it).
    No arg prompts for a name. Config persists automatically - /save never writes it."""
    name = arg.strip()
    if not any(m.get("role") != "system" for m in agent.messages):
        # Empty conversation: a bare `/save` has nothing to snapshot, but an
        # explicit `/save <name>` CLAIMS that name - retarget autosave so the next
        # real turn (and the on-exit autosave) lands under it. No empty file written.
        if not name:
            print(yellow("nothing to save (conversation is empty)."))
            return
        if not _session_name_ok(name):
            print(red(f"invalid session name: {name!r}"))
            return
        if cfg.incognito:
            print(yellow("incognito: not autosaving; name not claimed."))
            return
        agent.autosave_name = name
        print(green(f"will autosave this session as '{name}' once it has content."))
        return
    if not name:
        default = time.strftime("%Y-%m-%d-%H%M")
        try:
            name = input(f"save session as [{default}]: ").strip() or default
        except (EOFError, KeyboardInterrupt):
            print()
            return
    try:
        rhost = agent.remote.host if getattr(agent, "remote", None) else None
        path, meta = save_session(agent.messages, cfg, name, remote=rhost,
                                  pinned_plan=getattr(agent, "pinned_plan", ""),
                                  notes=getattr(agent, "notes", None))
    except ValueError:
        print(red(f"invalid session name: {name!r}"))
        return
    agent.dirty = False              # this conversation is now snapshotted
    # Continue autosaving INTO this name, so further work + the on-exit autosave land
    # here too - loading it later resumes where you left off, not the save-time snapshot.
    # (To branch off an immutable checkpoint, /save under a new name.)
    if not cfg.incognito:
        agent.autosave_name = path.stem
    print(green(f"saved session '{path.stem}' ({meta['turns']} turns) -> {path}"))
    if cfg.autosave and not cfg.incognito:
        print(dim(f"  now autosaving into '{path.stem}'"))


def _restore_backend_for(agent, cfg, meta):
    """Switch the live backend to match a loaded session's saved provider + model,
    so a chat loads back onto the backend it ran on. Returns a short human note, or
    "" if nothing changed. Never raises - a provider that won't activate
    (unregistered, logged out, down) degrades to a note telling the user to /model.
    Legacy sessions saved provider="" (the old native Ollama); that maps to the
    bundled "ollama" provider now."""
    want = (meta.get("provider") or "").strip() or "ollama"
    model = (meta.get("model") or "").strip()
    saved_host = (meta.get("host") or "").strip()
    cur = cfg.provider or ""
    # For ollama, the SESSION HOST is part of the backend identity: a chat on
    # 192.0.2.28 must reload onto 192.0.2.28, not snap back to whatever host is
    # current. Restore it before activation so the model list / ctx detection
    # run against the right box. (Hosted providers have no per-session host.)
    host_note = ""
    if want == "ollama" and saved_host and saved_host != cfg.host:
        cfg.hosts = [saved_host] + [h for h in cfg.hosts if h != saved_host]
        host_note = f" @ {saved_host}"
    if want != cur or host_note:                   # different backend OR different ollama host
        spec = get_provider(want)
        if spec is None:
            return f"note: saved on provider '{want}' (not registered now) - /model to pick."
        try:
            if not spec["available"]():
                return f"note: saved on provider '{want}' (not available) - login or /model."
            if model:
                cfg.provider_settings.setdefault(want, {})["model"] = model
                if want == "ollama":
                    cfg.model = model
            m = agent.activate_provider(want)
            autosave_config(cfg)
            label = "ollama" if want == "ollama" else f"provider '{want}'"
            return f"backend -> {label} (model {m})."
        except Exception as e:
            return f"note: couldn't restore backend '{want}' ({e}) - /model to pick."
    if model and model != cfg.active_model():       # same backend, model drifted -> sync it
        # Soft preference: adopt the session's model, but if it's gone from the live
        # backend fall back to what's actually available rather than pinning a dead
        # name. Keep it silent-graceful; sends + failover recover the rest.
        try:
            avail = list(get_provider(want)["list_models"]() or [])
        except Exception:
            avail = []
        if avail and not model_available(model, avail):
            alt = avail[0]
            cfg.set_active_model(alt)
            autosave_config(cfg)
            return f"note: saved model '{model}' not available now - using '{alt}' (/model to pick)."
        cfg.set_active_model(model)
        autosave_config(cfg)
        return f"model -> {model}."
    return ""


def _render_message_tail(messages, n: int = 4, width: int = 100, hint: bool = False):
    """Echo the last `n` non-system messages so a resumed (or /expand msg) view shows
    what you were doing. Roles are labelled + coloured (you / lc / tool); long lines
    collapse so it stays scannable. `n` is clamped to [1, len] so a silly count never
    errors or under-shows. `hint` appends the '/expand msg N' pointer when the view is
    partial. No-op for an empty conversation."""
    body = [m for m in messages if m.get("role") != "system"]
    if not body:
        return False
    n = max(1, min(int(n), len(body)))               # clamp: never error, never over-read
    tail = body[-n:]
    print(yellow(f"  recent context ({len(tail)} of {len(body)} messages):"))
    for m in tail:
        role = m.get("role", "?")
        if role == "user":
            label = cyan("you")
        elif role == "assistant":
            label = blue("llm")
        else:
            label = dim("tool")  # tool / other output
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        text = " ".join(content.split())              # collapse whitespace/newlines
        if not text and m.get("tool_calls"):
            text = "(tool call)"
        if len(text) > width:
            text = text[:width] + GLYPH["ellipsis"]
        print(f"    {label} {dim(GLYPH['dot'])} {dim(text) if role != 'user' else text}")
    if hint and len(tail) < len(body):
        print(dim(f"    … /expand msg {min(len(body), max(n * 2, 20))} to see further back"))
    print()
    return True


def _print_session_tail(messages, n: int = 4, width: int = 100):
    """Resume-banner view: the last few messages plus the '/expand msg N' hint so a
    user knows they can scroll further back without a reload. Thin wrapper over
    _render_message_tail (shared with /expand msg)."""
    _render_message_tail(messages, n=n, width=width, hint=True)


def _restore_session_state(agent, cfg, meta):
    """Restore a loaded session's full working state onto the live config: backend +
    model (via _restore_backend_for), then the per-session perms/settings - leash,
    approval, thinking, effort. A field MISSING from meta (older or brand-new session)
    falls back to the current config value, which save_session then re-captures, so
    the session adopts it from here on. leash/approval are applied at RUNTIME only
    (not persisted to config.toml) so two windows on one box can hold different
    ceilings. Returns a summary dict for the load read-out, flagging any leash
    escalation (design: apply + loud notice, never a silent raise)."""
    note = _restore_backend_for(agent, cfg, meta)          # backend + model

    # Restore the pinned plan (GOAL + TODO) BEFORE agent.restore() rebuilds the
    # system prompt, so the plan rides the rebuilt prompt. Absent in older sessions
    # -> keep whatever's live (don't clobber a plan with nothing).
    plan = meta.get("pinned_plan")
    if plan is not None:
        agent.pinned_plan = plan

    # Restore session-scoped episodic notes (the note tool). Absent in older sessions
    # -> keep whatever's live (don't clobber). Coerced to a clean list of {dtg,text}.
    notes = meta.get("notes")
    if isinstance(notes, list):
        agent.notes = [n for n in notes if isinstance(n, dict) and n.get("text")]

    # Restore the learned token calibration factor so the meter is accurate on the very
    # first (pre-prompt_eval) render after a load, instead of under-reporting until the
    # first turn re-learns it. Absent in older sessions -> keep the live factor.
    tf = meta.get("tok_factor")
    if tf is not None:
        set_tok_factor(tf)

    for key in ("thinking", "effort"):                     # provider-scoped settings
        val = meta.get(key)
        if val is not None and val != cfg.setting(key):
            cfg.set_setting(key, val)

    # Per-key SESSION OVERRIDES (the uniform working-state layer): re-apply the loaded
    # session's overrides onto the LIVE cfg at RUNTIME only - never save_config here, so
    # a worker/auto session can't clobber the config.toml defaults (the old approval->
    # auto root bug). The legacy top-level meta "approval"/"leash" (below) still restore
    # for sessions saved before this layer; going forward they ride session_overrides.
    overrides = meta.get("session_overrides")
    if isinstance(overrides, dict):
        cfg.session_overrides = dict(overrides)
        for k, v in overrides.items():
            if k not in _SETTINGS_BY_KEY or k == "leash":  # leash handled below (escalation notice)
                continue
            if k == "approval":
                if v in APPROVAL_MODES:
                    set_approval(cfg, v)
            else:
                setattr(cfg, k, v)
        if "ask_user_to_run" in overrides:
            agent.refresh_tools()

    approval = meta.get("approval")
    if approval in APPROVAL_MODES and approval != cfg.approval:
        set_approval(cfg, approval)                        # runtime only (not autosaved)

    escalated_from = None
    # A leash session-override takes precedence over the legacy top-level meta field.
    want = (meta.get("session_overrides") or {}).get("leash") or meta.get("leash")
    if want in LEASH_LEVELS and want != cfg.leash:
        if LEASH_LEVELS.index(want) > LEASH_LEVELS.index(cfg.leash):
            escalated_from = cfg.leash                     # this load RAISES the ceiling
        cfg.leash = want                                   # runtime only (not autosaved)
        agent.messages[0] = {"role": "system", "content": agent._system()}
        agent.refresh_tools()
        agent._pending_ai_note = _leash_note(want)         # tell the model next turn
    return {
        "backend_note": note,
        "leash": cfg.leash,
        "leash_escalated_from": escalated_from,
        "approval": cfg.approval,
        "thinking": cfg.setting("thinking"),
        "effort": cfg.setting("effort"),
        "remote": meta.get("remote"),
    }


def _print_session_state(state):
    """Render the restored working state on load (perms always visible, per the
    'show them on load' rule). A leash that this load RAISED is shouted in yellow."""
    grant = _LEASH_GRANTS.get(state["leash"], "")
    leash_line = (f"  leash    {state['leash']} ({grant})"
                  f"   approve  {state['approval']}")
    if state["leash_escalated_from"]:
        print(yellow(leash_line))
        print(yellow(f"  {GLYPH['warn']} leash raised from "
                     f"'{state['leash_escalated_from']}' - this session grants more"))
    else:
        print(dim(leash_line))
    print(dim(f"  thinking {state['thinking'] or 'off'}   effort   {state['effort'] or '-'}"))
    if state["backend_note"]:
        print(dim(f"  {state['backend_note']}"))


def _maybe_reconnect(agent, cfg, meta):
    """Realign the connection to where the loaded session last ran. If it ran on a
    remote host, restore that connection; if it ran LOCAL but we're currently on a
    remote, offer to disconnect back to local. Automatic when cfg.auto_reconnect is
    on, otherwise a y/N prompt (default off). No-op when we're already in the right
    place."""
    host = meta.get("remote")
    cur = agent.remote.host if agent.remote else None
    if (host or None) == cur:
        return                                 # already where the session ran
    if not host:                               # session was local, we're on a remote
        if cfg.auto_reconnect:
            print(dim(f"  detaching from {cur} - this session ran local "
                      f"(auto_reconnect on)…"))
            agent.set_remote(None)             # keep it open in the pool
        elif _ask(f"this session ran local but you're on remote {cur} - "
                  f"return to local?"):
            agent.set_remote(None)
            print(dim(f"  tools run locally again ({cur} still open - "
                      f"/connect {cur} to switch back)."))
        else:
            print(dim(f"  staying on {cur} - /local to return to local."))
        return
    if cfg.auto_reconnect:
        print(dim(f"  reconnecting to remote {host} (auto_reconnect on)…"))
        try:
            _do_connect(agent, cfg, host)
        except Exception as e:
            print(red(f"  reconnect failed: {e} - /connect {host} to retry."))
    elif _ask(f"this session ran on remote {host} - reconnect?"):
        _do_connect(agent, cfg, host)
    else:
        print(dim(f"  staying local - /connect {host} to reconnect."))


def _load_session_into(agent, cfg, name):
    """Restore a named session into the live conversation (with an unsaved-work
    guard). Shared by /load, the picker, and /session load."""
    try:
        data = load_session(name)
    except FileNotFoundError:
        print(red(f"no such session: {name}  (/load to list)"))
        return
    except (json.JSONDecodeError, OSError) as e:
        print(red(f"session '{name}' is unreadable ({type(e).__name__}) - "
                  f"the file may be corrupt (e.g. a crash mid-save). Not loaded; "
                  f"your current session is untouched."))
        return
    # Only guard when the work would actually be LOST: autosave (non-incognito)
    # has already persisted the current session to disk each turn, so there is
    # nothing to discard - don't nag in that case.
    persisted = cfg.autosave and not cfg.incognito
    if (agent.dirty and not persisted
            and any(m.get("role") != "system" for m in agent.messages)):
        if not _ask("current session has unsaved changes - discard them and load anyway?"):
            print(dim("load cancelled - '/save [name]' to keep them first."))
            return
    # If the session is live in another instance, confirm the take-over (the lock
    # gets claimed by the first autosave into it here).
    if not cfg.incognito and _lock_is_live(name):
        if not _ask(f"session '{_session_name_ok(name)}' is live in another instance - "
                    f"take it over?"):
            print(dim("load cancelled - leaving the other instance alone."))
            return
    meta = data.get("meta", {})
    state = _restore_session_state(agent, cfg, meta)   # backend, model, leash, approve, think/effort
    msgs = data.get("messages", [])
    n = agent.restore(msgs)
    # Continue working IN the loaded session - further turns autosave back into it,
    # so it stays current (no silent fork into a new auto- name). Claim its lock NOW
    # (and drop the one we're leaving) so the first autosave owns it - this is what a
    # take-over actually does, and what stops the loader being forked straight off it.
    if not cfg.incognito:
        _release_lock(agent.autosave_name)
        agent.autosave_name = _session_name_ok(name)
        _write_lock(agent.autosave_name)
    print(green(f"resumed '{_session_name_ok(name)}'  {GLYPH['dot']}  {n} turns  "
                f"{GLYPH['dot']}  saved {meta.get('saved_at', '?')}"))
    _print_session_tail(msgs)
    _print_session_state(state)
    _maybe_reconnect(agent, cfg, meta)
    # The status rows (printed before the next prompt) already carry ctx + quota,
    # so only print the standalone meter here when the status line is off (fallback).
    if not (getattr(cfg, "statusline", True) and _TTY):
        agent._print_ctx()


def _session_picker(prompt="load session:", show_snapshots=False):
    """Recent-first session picker (most-recently-used first, with a relative-age
    column). Returns the chosen session name, or None if cancelled/empty. Pre-handover
    safety snapshots are hidden unless `show_snapshots` (they clog the menu; /load
    <exact-name> reaches them regardless)."""
    rows = _session_rows()
    if not show_snapshots:
        rows = [r for r in rows if not _is_snapshot(r[0])]
    if not rows:
        print(dim(f"no saved sessions (/save [name]).  dir: {SESSIONS_DIR}"))
        return None
    now = time.time()
    d = GLYPH["dot"]
    labels = []
    for nm, meta, mtime in rows:
        age = f"{_fmt_age(now - mtime)} ago" if mtime else "?"
        labels.append(f"{nm}  ({meta.get('saved_at', '?')} {d} "
                      f"{meta.get('turns', '?')} turns {d} {age})")
    choice = pick_one(prompt, labels)
    return rows[labels.index(choice)][0] if choice else None


def handle_load_command(agent, cfg, arg):
    """/load [name] - resume a session. No arg opens the recent-first picker;
    a name (Tab-completes) loads directly."""
    name = arg.strip() or _session_picker(show_snapshots=cfg.show_snapshots)
    if name:
        _load_session_into(agent, cfg, name)


def start_new_session(agent, cfg, name=""):
    """Finalize the current session (kept on disk under its own name) and switch to
    a fresh, empty one. name='' -> a rolling auto- target; a non-empty name -> that
    named session. Returns the new autosave_name. The caller handles any confirm
    prompts (overwriting an existing name, discarding un-autosaved work)."""
    autosave_session(agent, cfg)          # keep the session we're leaving on disk
    agent.reset()
    agent.autosave_name = (_session_name_ok(name) if name.strip()
                           else _new_autosave_name())
    return agent.autosave_name


def handle_session_command(agent, cfg, arg):
    """/session list | delete <name>. save -> /save, load -> /load (no arg here
    opens the same load picker)."""
    parts = arg.split(maxsplit=1)
    if not parts:
        sub = pick_one("/session:", ["list", "load", "save", "delete"])
        if not sub:
            return
        rest = ""
    else:
        sub = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "save":
        handle_save_command(agent, cfg, rest)
        return
    if sub == "load":
        handle_load_command(agent, cfg, rest)
        return

    if sub == "list":
        rows = _session_rows()
        hidden = 0
        if not cfg.show_snapshots:
            kept = [r for r in rows if not _is_snapshot(r[0])]
            hidden = len(rows) - len(kept)
            rows = kept
        if not rows:
            print(dim(f"no saved sessions (/save [name]).  dir: {SESSIONS_DIR}"))
            return
        now = time.time()
        d = GLYPH["dot"]
        print(bold("sessions") + dim("  (most-recently-used first)"))
        for name, meta, mtime in rows:
            when = meta.get("saved_at", "?")
            turns = meta.get("turns", "?")
            title = meta.get("title", "")
            age = f"{_fmt_age(now - mtime)} ago" if mtime else "?"
            print(f"  {bold(cyan(name))}  {dim(f'{when} {d} {age} {d} {turns} turns {d} {title}')}")
        if hidden:
            print(dim(f"  ({hidden} pre-handover snapshot(s) hidden; "
                      f"/set show_snapshots true to show)"))
        return

def _remote_alive(ws) -> bool:
    """Best-effort: is this pooled connection still usable? A rebooted/dropped box
    makes the executor process exit or its pipe error, which we detect here so a
    switch can transparently reconnect instead of failing on the next tool call."""
    c = getattr(ws, "client", None)
    if c is None or c.proc.poll() is not None:
        return False
    try:
        c.request(RAW_READ, {"path": ".__lc_alive__"})   # any response = alive
        return True
    except (ConnectionError, OSError, ValueError):
        return False


def _do_connect(agent, cfg, rhost, rpath=".", offer_save=False, ephemeral=False):
    """Connect to / switch to `rhost`, routing tools there. Switching to an
    already-open box is instant (no re-push); a pooled box that died (reboot) is
    transparently reconnected. Push is implicit (no prompt); any failure leaves
    the session where it was. `offer_save` remembers an ad-hoc host. `ephemeral`
    (Windows) forces the embeddable-Python runtime and wipes it on teardown."""
    if agent.remote and agent.remote.host == rhost:
        print(dim(f"already active on {rhost}."))
        return
    pooled = agent.remotes.get(rhost)
    if pooled is not None and not ephemeral:
        if _remote_alive(pooled):
            agent.set_remote(pooled)              # instant switch
            print(green(f"switched to {rhost} (tools run there)."))
            return
        print(yellow(f"{rhost} had dropped (rebooted?) - reconnecting ..."))
        agent.drop_remote(rhost)                  # clear the dead one, then reopen
    where = f"on {agent.remote.host}" if agent.remote else "local"
    try:
        ws = RemoteWorkspace(
            rhost, rpath, lean_tool_paths=agent.lean_tools.enabled_paths(),
            ephemeral=ephemeral).connect()
    except (ConnectionError, OSError) as e:
        print(red(f"connect failed (staying {where}): {e}"))
        return
    agent.set_remote(ws)
    print(green(f"connected: tools now run on {rhost} (agent at {ws.remote_dir})"))
    # /sh drops into the box's LOGIN shell (its sshd DefaultShell), which on Windows is
    # cmd.exe unless an admin set PowerShell - distinct from run_command's shell. Say so
    # once so a /sh isn't a surprise.
    if not ws.remote_posix:
        print(dim("  note: /sh here opens the host's default shell (cmd.exe unless "
                  "PowerShell is set); run_command runs cmd.exe commands."))
    print(dim("  /local detaches (keeps it open to switch back); /connect <host> switches."))
    if (offer_save and rhost not in cfg.connect_hosts
            and rhost not in cfg.connect_hosts.values()):
        if _ask(f"save '{rhost}' to your connect list?"):
            cfg.connect_hosts[rhost] = rhost
            save_config(cfg)


_NO_TTY = object()   # _pick_one_tty sentinel: raw mode unavailable -> numbered fallback


def _pick_one_tty(header, choices, current=None, labels=None):
    """Inline arrow-key single-select picker (fzf/gum-style): up/down move, type to
    filter, enter selects, esc/^C cancels, with a SCROLLING VIEWPORT so a long list
    never overflows the screen. Drawn below the cursor, no alternate screen - degrades
    identically in any terminal; the caller falls back to the numbered prompt when this
    returns _NO_TTY (no tty / dumb term / forced plain). Built on the shared run_picker
    engine. `labels` optionally supplies pre-styled display strings (e.g. model rows
    with warm-dots/current marks) parallel to `choices`; filtering matches the plain
    label text. Rows are numbered - typing digits jumps to that row (Enter selects it);
    a non-digit character switches to filter mode. Returns the chosen value, None, or
    _NO_TTY."""
    if not picker_capable():
        return _NO_TTY
    plain = [str(c) for c in choices]                     # for filtering + current-match
    disp = [str(l) for l in labels] if labels is not None else plain
    st = {"query": "", "cur": 0, "top": 0}
    if current is not None and str(current) in plain:     # start on the current value
        st["cur"] = plain.index(str(current))
    labels = plain                                        # _filtered() matches plain text

    def _filtered():
        q = st["query"].lower()
        if not q:
            return list(range(len(choices)))
        return [i for i, lbl in enumerate(labels) if q in lbl.lower()]

    def render(rows_avail):
        fi = _filtered()
        if fi:
            st["cur"] %= len(fi)
        body = max(1, rows_avail - 2)              # 2 rows: header + filter line
        cur, top = st["cur"], st["top"]
        if not fi:
            top = 0
        elif cur < top:
            top = cur
        elif cur >= top + body:
            top = cur - body + 1
        st["top"] = top
        hint = dim("(up/down, #=jump, type to filter, enter select, esc cancel)")
        q = (cyan(st["query"]) + dim("_")) if st["query"] else dim("(type to filter)")
        more_up   = " ↑more" if top > 0 else ""      # sweep-ok
        more_down = " ↓more" if fi and top + body < len(fi) else ""      # sweep-ok

        def row(content):                          # fit to width so nothing wraps
            return "\r" + _fit_line(content) + "\033[K"

        out = [row(bold(header) + "  " + hint),
               row("  " + dim("filter: ") + q + dim(more_up + more_down))]
        for pos in range(top, min(top + body, len(fi))):
            i = fi[pos]
            sel = (pos == cur)
            pointer = cyan(">") if sel else " "
            num = dim(f"{pos + 1:>2}) ")            # 1-based row number for #-jump
            mark = dim("  (current)") if current is not None and plain[i] == str(current) else ""
            line = f"{disp[i]}{mark}"
            out.append(row(pointer + " " + num + (cyan(line) if sel else line)))
        if not fi:
            out.append(row("  " + dim("(no match)")))
        sys.stdout.write("\n".join(out))
        return len(out) - 1

    def on_key(k):
        fi = _filtered()
        # Number-jump: while the query is empty, a digit moves the cursor to that
        # 1-based visible row (multi-digit accumulates); Enter then selects it. Once
        # a non-digit is typed the query is text and digits become filter characters.
        if not st["query"] and isinstance(k, str) and k.isdigit():
            st["numjump"] = (st.get("numjump", "") + k)[-3:]
            n = int(st["numjump"])
            if fi and 1 <= n <= len(fi):
                st["cur"] = n - 1
            return None
        st["numjump"] = ""
        if k == _K_UP and fi:
            st["cur"] = (st["cur"] - 1) % len(fi)
        elif k == _K_DOWN and fi:
            st["cur"] = (st["cur"] + 1) % len(fi)
        elif k == _K_PGUP:
            st["cur"] = max(0, st["cur"] - 10)
        elif k == _K_PGDN and fi:
            st["cur"] = min(len(fi) - 1, st["cur"] + 10)
        elif k == _K_HOME:
            st["cur"] = 0
        elif k == _K_END and fi:
            st["cur"] = len(fi) - 1
        elif k == _K_ENTER:
            return ("done", choices[fi[st["cur"]]] if fi else None)
        elif k == _K_CANCEL:
            return ("cancel", None)
        elif k == _K_BACK:
            st["query"] = st["query"][:-1]
            st["cur"] = 0
        elif k == _K_SPACE:
            st["query"] += " "
            st["cur"] = 0
        elif isinstance(k, str) and len(k) == 1 and k >= " ":   # printable -> filter
            st["query"] += k
            st["cur"] = 0
        return None

    res = run_picker(render, on_key)
    return res[1] if res[0] == "done" else None


def pick_one(header, choices, current=None, prompt=input, labels=None):
    """PUBLIC single-select picker (reusable by features + lean-tools via lc['pick_one']).
    Returns the chosen value or None (cancel). Rich inline picker on a real terminal
    (up/down + #-to-jump + type-to-filter + enter, scrolling, no-wrap); falls back to
    the classic numbered prompt when there's no tty, a dumb TERM, forced --plain, or a
    caller passes its own `prompt` (tests, headless). `labels` optionally supplies
    pre-styled display strings parallel to `choices` (e.g. model rows with warm-dots) -
    both the rich picker and the numbered fallback render them, while selection/
    current-match still key off `choices`."""
    disp = [str(l) for l in labels] if labels is not None else [str(c) for c in choices]
    if prompt is input:                              # only the default interactive path
        chosen = _pick_one_tty(header, choices, current, labels=labels)
        if chosen is not _NO_TTY:
            return chosen
    print(bold(header))
    for i, c in enumerate(choices, 1):
        mark = dim("  (current)") if current is not None and str(c) == str(current) else ""
        print(f"  {i}) {disp[i - 1]}{mark}")
    print("  0) cancel")
    try:
        sel = prompt("choice: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    idx = _menu_index(sel, len(choices))
    return choices[idx] if idx is not None else None


def _safe_caps(spec, cfg) -> dict:
    try:
        return spec["capabilities"](cfg.active_model()) or {}
    except Exception:
        return {}


def _warm_models(spec) -> list:
    """A provider's loaded/instant models (its opt-in warm_models() hook), absent-
    safe. [] when the provider doesn't implement it (e.g. a hosted API) or it
    errors. Used for the green 'loaded' dots in /model and /usage."""
    hook = spec and spec.get("warm_models")
    if not hook:
        return []
    try:
        return list(hook() or [])
    except Exception:
        return []


def _apply_setting(cfg, key, val, choices):
    """Set (or menu-pick) a per-provider setting. `choices` None = free-form."""
    val = (val or "").strip()
    if val:
        if choices and val.lower() not in [str(c).lower() for c in choices]:
            print(yellow(f"{key}: '{val}' invalid - choose from "
                         f"{', '.join(map(str, choices))}"))
            return
        cfg.set_setting(key, val)
        print(dim(f"{key}: {val}"))
    elif choices:
        chosen = pick_one(f"{key}:", list(choices), current=cfg.setting(key))
        if chosen is not None:
            cfg.set_setting(key, chosen)
            print(dim(f"{key}: {chosen}"))
    else:
        cur = cfg.setting(key)
        print(dim(f"{key}: {cur if cur is not None else '(unset)'}  "
                  f"(free-form; pass a value to set)"))


def _set_known(agent, cfg, key, arg) -> bool:
    """Capability-gated set for a well-known knob (thinking/effort) on the active
    provider. Returns False for the bundled ollama provider (or no backend), so the
    caller falls back to ollama's simple think=true/false toggle; True once handled
    by a provider that declares the knob in capabilities() (including 'not
    supported')."""
    spec = agent.active_provider()
    if spec is None or spec["name"] == "ollama":
        return False
    choices = _safe_caps(spec, cfg).get(key)
    if not choices:
        print(yellow(f"{key}: not supported by provider '{cfg.provider}'"))
        return True
    _apply_setting(cfg, key, arg, choices)
    return True


def handle_provider_set_command(agent, cfg, arg):
    """/provider set [key [value]] - the extensible escape hatch. Lists/sets any knob
    the active provider declares in capabilities(), and allows free-form keys the
    provider reads itself (tool format, prompt caching, whatever) without core
    needing to know what they mean. (App-config knobs live in /set.)"""
    spec = agent.active_provider()
    if spec is None:
        print(dim("/provider set applies to an active provider; none active (see /provider)"))
        return
    caps = _safe_caps(spec, cfg)
    parts = arg.split(maxsplit=1)
    if not parts:
        cur = cfg.provider_settings.get(cfg.provider, {})
        if not caps:
            print(dim(f"provider '{cfg.provider}' declares no settings "
                      f"(you can still /provider set <key> <value> a custom one)"))
            return
        print(bold(f"{cfg.provider} settings:"))
        for k, ch in caps.items():
            val = cur.get(k)
            shown = val if val is not None else "(unset)"
            print(f"  {k} = {shown}   " + dim("choices: " + ", ".join(map(str, ch))))
        return
    key = parts[0]
    val = parts[1] if len(parts) > 1 else ""
    _apply_setting(cfg, key, val, caps.get(key))


def _resolve_provider_for(hook, name):
    """Pick the provider a login/clear sub-command targets: an explicit name, or
    the sole registered provider that implements `hook`. Returns the name or
    None (with a printed reason)."""
    if name:
        spec = get_provider(name)
        if spec is None:
            print(yellow(f"no such provider '{name}'."))
            return None
        if not spec.get(hook):
            print(dim(f"provider '{name}' has no {hook}."))
            return None
        return name
    cands = [n for n in provider_names() if (get_provider(n) or {}).get(hook)]
    if not cands:
        print(dim(f"no provider supports {hook}."))
        return None
    if len(cands) > 1:
        print(dim(f"specify which: /provider {hook} <{' | '.join(cands)}>"))
        return None
    return cands[0]


def _provider_login(agent, cfg, name):
    """/provider login [name] - run the provider's auth hook, then activate it.

    First-class OBE: if `name` is a bundled-but-not-yet-enabled provider plugin, enable
    it first so a brand-new user can go straight from `/provider login groq` to using
    it, without a separate `/provider enable` step."""
    if name and get_provider(name) is None:
        mgr = ProviderManager(_provider_dirs(cfg), enabled=cfg.providers_enabled)
        if name in mgr.names():
            _toggle_provider_plugin(agent, cfg, mgr, name)   # registers + persists
    target = _resolve_provider_for("login", name)
    if target is None:
        return
    spec = get_provider(target)
    try:
        ok = spec["login"](agent, cfg)
    except Exception as e:
        print(red(f"login failed: {e}"))
        return
    if ok is False:
        print(dim("login cancelled."))
        return
    try:
        model = agent.activate_provider(target)
        print(green(f"provider: {target}") + dim(f"  model: {model}"))
    except Exception as e:
        print(red(f"/provider {target} failed: {e}"))


def _provider_clear(agent, cfg, name):
    """/provider clear [name] - wipe a provider's stored credentials (and drop
    back to Ollama if it was active)."""
    target = _resolve_provider_for("clear", name or cfg.provider)
    if target is None:
        return
    spec = get_provider(target)
    try:
        spec["clear"](agent, cfg)
    except Exception as e:
        print(red(f"clear failed: {e}"))
        return
    if cfg.provider == target:
        agent.deactivate_provider()
    print(dim(f"cleared credentials for '{target}'."))


def _manage_provider_plugin(agent, cfg, name, action):
    """enable/disable/toggle one providers/ dir plugin (the catalog), respecting the
    requested state so an explicit enable/disable is idempotent."""
    mgr = ProviderManager(_provider_dirs(cfg), enabled=cfg.providers_enabled)
    if name not in mgr.names():
        print(dim(f"no such provider plugin '{name}' "
                  f"(have: {', '.join(mgr.names()) or 'none'})"))
        return
    is_on = name in cfg.providers_enabled
    want_on = action == "enable" or (action == "toggle" and not is_on)
    if want_on == is_on:
        print(dim(f"{name}: already {'on' if is_on else 'off'}"))
        return
    _toggle_provider_plugin(agent, cfg, mgr, name)


def handle_provider_command(agent, cfg, arg):
    """/provider - manage the model backends (folds in the old /providers):
      /provider                 the on/off catalog menu (toggle backends, incl. ollama)
      /provider list            alias for the catalog menu
      /provider <name>          switch the active backend to a registered one
      /provider off             back to the bundled ollama default
      /provider on              re-activate the last-used backend
      /provider login [name]    run a provider's auth flow, then activate
      /provider clear [name]    wipe a provider's credentials
      /provider enable <name>   enable a providers/ dir plugin
      /provider disable <name>  disable one
      /provider set [key [val]] get/set a backend-specific knob (was /set)
    Switching the active MODEL (and its backend) is /model; bare /provider is the
    catalog (what's on/off), matching the old /provider list. App-config knobs
    (handover gates, sampling, etc.) live in /set."""
    parts = arg.split()
    sub = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    if sub == "login":
        _provider_login(agent, cfg, rest)
        return
    if sub == "clear":
        _provider_clear(agent, cfg, rest)
        return
    if sub == "set":                        # /provider set [key [value]] - backend knobs
        handle_provider_set_command(agent, cfg, arg.split(maxsplit=1)[1] if len(parts) > 1 else "")
        return
    if sub in ("", "list"):                 # bare /provider == the catalog on/off menu
        handle_providers_command(agent, cfg, "")
        return
    if sub in ("enable", "disable", "toggle"):
        if not rest:
            print(dim(f"usage: /provider {sub} <plugin-name>"))
            return
        _manage_provider_plugin(agent, cfg, rest, sub)
        return
    if sub in ("off", "none"):
        # "off" means back to the bundled ollama default (there is no other "native").
        if cfg.provider == "ollama":
            print(dim("already on the ollama provider."))
        else:
            name = cfg.provider
            agent.deactivate_provider()
            print(dim(f"left provider '{name}' - back on ollama ({cfg.active_model()})."))
        return
    names = provider_names()                    # includes the bundled "ollama"
    if not names:
        print(dim("no provider enabled - /provider to add one."))
        return
    if sub == "on":
        # mirror of '/provider off': re-activate the last-used provider, or the
        # sole non-ollama one. With several and none remembered, ask for a name.
        others = [n for n in names if n != "ollama"]
        sub = agent._last_provider or (others[0] if len(others) == 1 else "")
        if not sub:
            print(dim(f"which provider? /provider <{' | '.join(names)}>"))
            return
    if sub == cfg.provider:
        print(dim(f"already on '{sub}' ({cfg.active_model()})."))
        return
    try:
        model = agent.activate_provider(sub)   # named activation: /provider <name>
        where = "ollama" if sub == "ollama" else f"provider: {sub}"
        print(green(where) + dim(f"  model: {model}"))
    except Exception as e:
        print(red(f"/provider {sub} failed: {e}"))


def handle_usage_command(agent, cfg, arg=""):
    """/usage - the active provider's full usage view (its detail() hook), or a
    core default (model, session tokens, ctx%, warm models) on Ollama / when a
    provider declares no detail()."""
    spec = agent.active_provider()
    if spec is not None and spec.get("detail"):
        try:
            fn = spec["detail"]
            # detail() is (agent, cfg); some providers accept an extra arg string
            # (e.g. the subscription provider's 'raw' dump). Pass it only if taken.
            try:
                nparams = fn.__code__.co_argcount
            except Exception:
                nparams = 2
            out = fn(agent, cfg, arg) if nparams >= 3 else fn(agent, cfg)
        except Exception as e:
            print(red(f"usage failed: {e}"))
            return
        if out:
            print(out if isinstance(out, str) else str(out))
            return
    # core default - works for any backend
    loc = _provider_label(cfg)
    print(bold(cfg.active_model()) + dim(f"  @ {loc}"))
    si, so = agent.session_in, agent.session_out
    print(dim(f"  session   in {si:,}   out {so:,}   total {si + so:,}"))
    used = agent.last_prompt_tokens or messages_tokens(agent.messages, agent.tool_defs)
    window = cfg.ctx_window() or 1
    print(dim(f"  context   ~{used:,}/{window:,} ({used / window * 100:.0f}%)"))
    warm = _warm_models(spec)              # loaded/instant models (ollama green dots)
    if warm:
        print(dim(f"  loaded    {', '.join(warm)}"))


def _index_pick(arg, items):
    """If `arg` is a 1-based index into `items` (matching the numbered menu),
    return that item; else None. Lets '/model 7' select option 7."""
    if arg.isdigit() and 1 <= int(arg) <= len(items):
        return items[int(arg) - 1]
    return None


def _model_rows(agent, cfg):
    """Every selectable backend/model across all enabled providers, with display
    metadata. Each row: {provider, model, tag, warm, status, available, current}.
    An unavailable (e.g. logged-out) provider yields a single model=None row so the
    menu can offer to log it in. Pure w.r.t. the terminal - the provider hooks are
    faked in tests - so it's unit-tested headless."""
    rows = []
    cur_p, cur_m = cfg.provider, cfg.active_model()
    for pname in provider_names():
        spec = get_provider(pname)
        tag = (spec.get("tag") if spec else None) or pname
        try:
            avail = bool(spec["available"]())
        except Exception:
            avail = False
        if not avail:
            rows.append({"provider": pname, "model": None, "tag": tag, "warm": False,
                         "status": "logged out - select to log in",
                         "available": False, "current": False})
            continue
        try:
            models = list(spec["list_models"]() or [])
        except Exception:
            models = []
        warm = set(_warm_models(spec))
        # An enabled provider that lists NO models (its host/endpoint is unreachable -
        # e.g. ollama pointed at a down host) must not silently vanish from /model: show
        # a single hint row so the user sees it's on but can't be reached, with a nudge
        # to fix the host. `endpoints` marks the multi-host (ollama-style) providers.
        if not models:
            hint = ("host unreachable - /ollama host to point it, or start ollama"
                    if spec.get("endpoints") else "no models available")
            rows.append({"provider": pname, "model": None, "tag": tag, "warm": False,
                         "status": hint, "available": True, "current": False})
            continue
        for m in models:
            status = None
            if spec.get("model_status"):
                try:
                    status = spec["model_status"](m)
                except Exception:
                    status = None
            rows.append({"provider": pname, "model": m, "tag": tag,
                         "warm": m in warm, "status": status, "available": True,
                         "current": pname == cur_p and m == cur_m})
    return rows


def _resolve_model_arg(arg, sel_rows):
    """Map a /model argument to a selectable row. Accepts a 1-based index into the
    listed rows, a 'provider:model' qualifier, or a bare model name (first match
    wins, preferring the active provider). Returns a row or None. Pure."""
    arg = arg.strip()
    # MENU CONTRACT: a bare 1-based integer selects the Nth listed row (see menu_resolve).
    picked = menu_resolve(arg, sel_rows)
    if picked is not None:
        return picked
    if ":" in arg:
        p, _, m = arg.partition(":")
        # ollama model tags contain ':' (e.g. qwen3:30b), so only treat the prefix
        # as a provider when it actually names one.
        if any(r["provider"] == p for r in sel_rows):
            cand = [r for r in sel_rows if r["provider"] == p and r["model"] == m]
            if cand:
                return cand[0]
    exact = [r for r in sel_rows if r["model"] == arg]
    if exact:
        return exact[0]
    # tag-aware (bare name vs :latest), preferring the current provider
    loose = [r for r in sel_rows if model_available(arg, [r["model"]])]
    return loose[0] if loose else None


def _select_model_row(agent, cfg, row):
    """Activate a chosen row's backend (switching providers if needed) and model."""
    p, m = row["provider"], row["model"]
    if p != cfg.provider:
        try:
            agent.activate_provider(p)
        except Exception as e:
            print(red(f"/model: couldn't switch to '{p}' ({e})"))
            return
    agent._apply_model(m)
    agent.last_prompt_tokens = None
    # ollama: remember this as the host's last-used model (Anthropic-style, silent),
    # keyed by the normalized active host so host <-> host_models never drift.
    if p == "ollama":
        cfg.host_models[_norm_host(cfg.host)] = m
    autosave_config(cfg)
    where = "ollama" if p == "ollama" else f"[{row.get('tag') or p}]"
    print(dim(f"model -> {m}  {where}"))


def _unified_model_lines(rows):
    """Build the display model for the grouped model picker: a flat list of
    (kind, payload) display rows - ("group", header) non-selectable headers,
    ("info", text) non-selectable info lines, ("item", row) selectable entries -
    plus index_of (selectable rows in order, for number-jump). Shared by the rich
    picker and the numbered fallback so both render identically. Returns
    (lines, index_of)."""
    lines, index_of, last_p = [], [], None
    for r in rows:
        if r["provider"] != last_p:
            last_p = r["provider"]
            head = r["provider"] + (f"  [{r['tag']}]" if r["tag"] != r["provider"] else "")
            lines.append(("group", head))
        if not r["model"]:
            if not r["available"]:                # logged-out -> selectable to log in
                index_of.append(r)
                lines.append(("item", r))
            else:                                 # enabled but unreachable -> info only
                lines.append(("info", r["status"]))
            continue
        index_of.append(r)
        lines.append(("item", r))
    return lines, index_of


def _fmt_model_row(r, num):
    """One selectable model row's text (number + dot + name + marks). `num` is its
    1-based selection number (for number-jump), or None to omit."""
    npart = dim(f"{num:>2}) ") if num is not None else ""
    if not r["model"]:                            # a (login) row for a logged-out backend
        return npart + dim(f"(login) {r['status']}")
    dot = green("*") if r["warm"] else " "
    mark = dim("  (current)") if r["current"] else ""
    hint = dim(f"   {r['status']}") if r["status"] else ""
    return f"{npart}{dot} {r['model']}{mark}{hint}"


def pick_unified_model_menu(rows, prompt=input):
    """Picker over all backends' models, grouped by provider with its tag, a green dot
    for warm/loaded models, and any model_status() hint dimmed. Rich picker on a real
    tty (arrow-keys + scrolling + number-type-then-Enter to jump-select fast + filter),
    numbered fallback otherwise (or when a test passes its own `prompt`). Returns the
    chosen row or None."""
    lines, index_of = _unified_model_lines(rows)
    if not index_of:
        legend = dim("  (" + green("*") + " loaded)")
        print(bold("models") + legend)
        for kind, payload in lines:
            if kind == "group":
                print(dim(f"  {payload}:"))
            elif kind == "info":
                print(dim(f"       {payload}"))
        print(dim("  (no selectable models)"))
        return None
    # number of each selectable item, keyed by id() of its row
    num_of = {id(r): i + 1 for i, r in enumerate(index_of)}

    if prompt is input and picker_capable():
        return pick_grouped(rows_header="models" + dim("  (" + green("*")
                             + " loaded, #=jump, arrows, type to filter)"),
                             lines=lines, index_of=index_of, num_of=num_of)

    # numbered fallback
    legend = dim("  (" + green("*") + " loaded)")
    print(bold("models") + legend)
    for kind, payload in lines:
        if kind == "group":
            print(dim(f"  {payload}:"))
        elif kind == "info":
            print(dim(f"       {payload}"))
        else:
            print("  " + _fmt_model_row(payload, num_of[id(payload)]))
    print("  0) cancel")
    try:
        s = prompt("choice: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    idx = _menu_index(s, len(index_of))
    return index_of[idx] if idx is not None else None


def pick_grouped(rows_header, lines, index_of, num_of):
    """Rich raw-mode picker for a GROUPED list (group headers + selectable items),
    on the shared run_picker engine. Cursor moves only over selectable items; group/
    info rows scroll with them. Fast path: type a number (multi-digit accumulates) to
    move the cursor to that item, Enter selects; a non-digit starts a text filter.
    Arrows/pgup/pgdn/home/end also move. Returns the chosen row, or None."""
    # positions of selectable lines within `lines`
    item_pos = [i for i, (k, _) in enumerate(lines) if k == "item"]
    st = {"sel": 0, "top": 0, "num": "", "query": ""}
    # start the cursor on the current model if present
    for si, r in enumerate(index_of):
        if r.get("current"):
            st["sel"] = si
            break

    def _visible():
        """The lines to show given the filter: keep group headers whose section has a
        matching item; drop non-matching items. Returns (vis_lines, vis_items) where
        vis_items maps to index_of rows in display order."""
        q = st["query"].lower()
        if not q:
            return lines, index_of
        vis, vis_items, pending_group = [], [], None
        for k, payload in lines:
            if k == "group":
                pending_group = (k, payload)
            elif k == "item":
                name = (payload.get("model") or payload.get("status") or "")
                if q in name.lower():
                    if pending_group:
                        vis.append(pending_group); pending_group = None
                    vis.append((k, payload)); vis_items.append(payload)
        return vis, vis_items

    def render(rows_avail):
        vis_lines, vis_items = _visible()
        if vis_items:
            st["sel"] %= len(vis_items)
        # map cursor (index into vis_items) to a line offset in vis_lines
        sel_line = 0
        seen = -1
        for li, (k, _p) in enumerate(vis_lines):
            if k == "item":
                seen += 1
                if seen == st["sel"]:
                    sel_line = li
                    break
        body = max(1, rows_avail - 2)
        top = st["top"]
        if sel_line < top:
            top = sel_line
        elif sel_line >= top + body:
            top = sel_line - body + 1
        st["top"] = top
        q = (cyan(st["query"]) + dim("_")) if st["query"] else dim("(type to filter)")
        more_up   = " ↑more" if top > 0 else ""      # sweep-ok
        more_down = " ↓more" if top + body < len(vis_lines) else ""      # sweep-ok

        def row(content):
            return "\r" + _fit_line(content) + "\033[K"

        # precompute each line's item-index (into vis_items), -1 for non-items
        item_index = []
        seen = -1
        for k, _p in vis_lines:
            if k == "item":
                seen += 1
                item_index.append(seen)
            else:
                item_index.append(-1)

        out = [row(bold(rows_header)),
               row("  " + dim("filter: ") + q + dim(more_up + more_down))]
        for li in range(top, min(top + body, len(vis_lines))):
            k, payload = vis_lines[li]
            if k == "group":
                out.append(row("  " + dim(payload + ":")))
            elif k == "info":
                out.append(row("       " + dim(payload)))
            else:
                is_sel = (item_index[li] == st["sel"])
                pointer = cyan(">") if is_sel else " "
                text = _fmt_model_row(payload, num_of[id(payload)])
                out.append(row(pointer + " " + (cyan(text) if is_sel else text)))
        if not vis_items:
            out.append(row("  " + dim("(no match)")))
        sys.stdout.write("\n".join(out))
        return len(out) - 1

    def on_key(k):
        _vl, vis_items = _visible()
        n = len(vis_items)
        if not st["query"] and isinstance(k, str) and k.isdigit():
            st["num"] = (st["num"] + k)[-3:]
            j = int(st["num"])
            if n and 1 <= j <= n:
                st["sel"] = j - 1
            return None
        st["num"] = ""
        if k == _K_UP and n:
            st["sel"] = (st["sel"] - 1) % n
        elif k == _K_DOWN and n:
            st["sel"] = (st["sel"] + 1) % n
        elif k == _K_PGUP:
            st["sel"] = max(0, st["sel"] - 10)
        elif k == _K_PGDN and n:
            st["sel"] = min(n - 1, st["sel"] + 10)
        elif k == _K_HOME:
            st["sel"] = 0
        elif k == _K_END and n:
            st["sel"] = n - 1
        elif k == _K_ENTER:
            return ("done", vis_items[st["sel"]] if n else None)
        elif k == _K_CANCEL:
            return ("cancel", None)
        elif k == _K_BACK:
            st["query"] = st["query"][:-1]; st["sel"] = 0
        elif isinstance(k, str) and len(k) == 1 and k >= " ":
            st["query"] += k; st["sel"] = 0
        return None

    res = run_picker(render, on_key)
    return res[1] if res[0] == "done" else None


def handle_model_command(agent, cfg, arg):
    """/model - unified across every enabled+available backend. Lists each
    provider's models (tagged), with a green dot for loaded models and any
    model_status() hint; selecting one activates its backend + model. With an arg:
    a 1-based index, a 'provider:model' qualifier, or a model name."""
    rows = _model_rows(agent, cfg)
    sel = [r for r in rows if r["model"]]
    if arg:
        chosen = _resolve_model_arg(arg, sel)
        if chosen is None:
            print(yellow(f"no model matching '{arg}'  (/model with no arg to list)"))
            return
        _select_model_row(agent, cfg, chosen)
        return
    if not rows:
        print(dim("no provider enabled - /provider to add one."))
        return
    if not sel and not any(not r["available"] for r in rows):
        loc = _provider_label(cfg)
        print(yellow(f"no models available from any provider "
                     f"(current: {cfg.active_model()} @ {loc})."))
        return
    if not (_TTY and sys.stdin.isatty()):
        for r in rows:
            if r["model"]:
                tag = f"[{r['tag']}]" if r["tag"] != r["provider"] else ""
                cur = " (current)" if r["current"] else ""
                print(f"  {r['provider']}: {r['model']} {tag}{cur}".rstrip())
        print(dim("  (no tty: /model <name> or provider:model to switch)"))
        return
    chosen = pick_unified_model_menu(rows)
    if chosen is None:
        return
    if chosen.get("model"):
        _select_model_row(agent, cfg, chosen)
    elif not chosen["available"]:                 # logged-out provider row -> log in
        _provider_login(agent, cfg, chosen["provider"])


def _maybe_autostart_provider(agent, cfg):
    """Activate the configured backend at launch (cfg.provider, default 'ollama'),
    or a registered provider whose autostart() asks for it. Returns the active
    spec, or None only if no backend could be started (every provider disabled).
    Never raises: on a failure it falls back to the bundled ollama provider."""
    # Precedence: an explicit non-ollama config provider, then a non-ollama
    # autostart() (so a hosted provider with live auth beats the ollama default),
    # then the configured provider (incl. ollama), then the bundled ollama floor.
    want = cfg.provider if (cfg.provider and cfg.provider != "ollama"
                            and cfg.provider in _providers) else ""
    if not want:
        for name, spec in _providers.items():
            if name == "ollama":
                continue
            try:
                if spec["autostart"]() and spec["available"]():
                    want = name
                    break
            except Exception:
                continue
    if not want:
        if cfg.provider in _providers:
            want = cfg.provider
        elif "ollama" in _providers:
            want = "ollama"               # the bundled, out-of-the-box default
    if not want:
        cfg.provider = ""
        agent._provider_fail_reason = ("no backend could be started - every provider "
                                       "is disabled/unconfigured on this box")
        return None                       # no provider at all -> no backend
    try:
        prep = get_provider(want).get("prepare")
        if prep:                          # transport setup before model/ctx pickup
            prep(agent, cfg)              # (ollama: tiered host failover + per-host model)
        model = agent.activate_provider(want)
        if want != "ollama":              # ollama is the quiet default; named ones announce
            print(green(f"provider: {want}") + dim(f"  model: {model}"))
        return agent.active_provider()
    except Exception as e:
        print(yellow(f"provider '{want}' not started: {e}"))
        if want != "ollama" and "ollama" in _providers:
            try:
                agent.activate_provider("ollama")
                return agent.active_provider()
            except Exception as e2:
                agent._provider_fail_reason = (f"provider '{want}' failed to start ({e}); "
                                               f"ollama fallback also failed ({e2})")
        else:
            agent._provider_fail_reason = f"provider '{want}' failed to start: {e}"
        cfg.provider = ""
        return None


def _parse_toggle(arg, current):
    """Map an [on|off] arg to a bool: on/off/true/false/yes/no/1/0; empty = flip the
    current value; anything else = None (invalid). Pure -> tested."""
    a = (arg or "").strip().lower()
    if a in ("on", "true", "yes", "1"):
        return True
    if a in ("off", "false", "no", "0"):
        return False
    if a == "":
        return not current
    return None


def handle_incognito_command(agent, cfg, arg):
    """/incognito [on|off] - no local session persistence this run, and the model is
    told. Deliberately honest: it does NOT hide anything from the model provider."""
    want = _parse_toggle(arg, cfg.incognito)
    if want is None:
        print(dim("usage: /incognito [on|off]"))
        return
    cfg.incognito = want
    agent.messages[0] = {"role": "system", "content": agent._system()}   # tell/untell the model
    if want:
        print(magenta(f"{GLYPH['ghost']} incognito: on")
              + dim(" - nothing is written to disk here (no autosave, no auto-load). "
                    "NOTE: this does not hide anything from the model provider; "
                    "requests may still be processed/retained per its policy."))
    else:
        print(dim(f"{GLYPH['ghost']} incognito: off - session persistence follows your "
                  "autosave setting again."))


def _apply_leash(agent, cfg, want, persist=True):
    """Set the capability ceiling, rebuild the system prompt + tool surface live, and
    announce the grant. Used by /leash. `persist=True` autosaves the leash as a
    config.toml DEFAULT (the /leash command's behaviour); `persist=False` applies it
    live only (a /set session override records it in session_overrides itself)."""
    cfg.leash = want
    agent.messages[0] = {"role": "system", "content": agent._system()}
    agent.refresh_tools()
    agent._pending_ai_note = _leash_note(want)   # tell the model on its next turn (not via the cached prompt)
    if persist:
        cfg._defaults["leash"] = want
        autosave_config(cfg)
    print(bold(f"leash: {want}") + dim(f"  - {_LEASH_GRANTS[want]}  "
                                       f"({len(agent.tool_defs)} tools)"))
    if want == "rwe":
        print(dim("  note: the leash bounds what the agent will ATTEMPT; the OS bounds what "
                  "it CAN do. Running lean-coder as root removes that bound - power-user only."))


def handle_leash_command(agent, cfg, arg):
    """/leash [chat|r|rw|rwe] - the capability ceiling: what the agent is given AT ALL,
    so it never blocks on (or reaches for) anything above it. Pairs with /approve (the
    confirm cadence): e.g. /leash r = walk-away-safe analysis; /leash rw + /approve auto
    = unattended editing; /leash rwe = full reach. No arg = a menu."""
    if not arg.strip():
        print(bold("leash") + dim(f"  (current: {cfg.leash} - {_LEASH_GRANTS[cfg.leash]})"))
        chosen = pick_one("leash:", list(LEASH_LEVELS), current=cfg.leash)
        if chosen is None:
            return
        want = chosen
    else:
        want = _norm_leash(arg)
        if want is None:
            print(dim("usage: /leash <chat|r|rw|rwe>  (aliases: read/write/exec, none/all)"))
            return
    _apply_leash(agent, cfg, want)


def handle_askread_command(agent, cfg, arg):
    """/askread [on|off] - ask-on-read: read tools (read_file/list_files/search_files
    + safe lean-tools) confirm too, not just writers. For anyone who doesn't want it
    reading without an OK. Orthogonal to /approve."""
    want = _parse_toggle(arg, cfg.confirm_reads)
    if want is None:
        print(dim("usage: /askread [on|off]"))
        return
    cfg.confirm_reads = want
    autosave_config(cfg)
    print(dim("ask-on-read: " + ("on - reads now confirm too (auto/session approval still skips)"
                                 if want else "off - reads run without a prompt")))


def _split_queued_commands(queued):
    """Partition drained mid-turn lines into (commands, messages), preserving order.
    A line is a command iff its first non-whitespace char is '/' (same rule the main
    prompt uses). Commands are dispatched as commands and never sent to the model;
    only messages go through the combine/separate/discard drain. Pure -> tested."""
    cmds = [q for q in queued if q.strip().startswith("/")]
    msgs = [q for q in queued if not q.strip().startswith("/")]
    return cmds, msgs


def _drain_choice(raw):
    """Map a queue-drain keypress to combine / separate / discard. Pure -> tested.
    Default (Enter or anything unrecognized) is 'combine' - one turn carrying all the
    queued lines, the least-surprising "I typed more while you worked, just take it"."""
    r = (raw or "").strip().lower()
    if r in ("s", "separate", "split"):
        return "separate"
    if r in ("d", "n", "discard", "no", "drop"):
        return "discard"
    return "combine"


def handle_clear_command(agent, cfg, arg):
    """/clear - wipe the conversation but STAY in the current session (same autosave
    name), so you can clear and keep working without forking off. /new starts a separate
    session. With autosave on the cleared conversation is /load-able until you continue."""
    nonempty = any(m.get("role") != "system" for m in agent.messages)
    will_persist = cfg.autosave and not cfg.incognito
    if nonempty:
        # Staying in the same name, continuing will overwrite this conversation. With
        # autosave on it's still /load-able until the next turn; with autosave off it's
        # gone now - say which.
        if will_persist:
            msg = (f"clear the conversation in session "
                   f"'{agent.autosave_name}'? (recoverable with /load "
                   f"until you continue)")
        else:
            msg = "current conversation is NOT autosaved - clear and lose it?"
        if not _ask(msg):
            print(dim("clear cancelled - /save [name] to keep it, "
                      "/new for a separate session."))
            return
    autosave_session(agent, cfg)     # finalize (no-op if off/empty)
    agent.reset()                    # keep the SAME autosave_name
    print(dim(f"history cleared (still in session '{agent.autosave_name}')."))


def handle_new_command(agent, cfg, arg):
    """/new [name] - start a separate session, keeping the current one on disk. /new with
    a name targets that named session directly; bare /new -> rolling auto-name."""
    want = arg.strip()
    will_persist = cfg.autosave and not cfg.incognito
    nonempty = any(m.get("role") != "system" for m in agent.messages)
    if want:
        safe = _session_name_ok(want)
        if not safe:
            print(red(f"invalid session name: {want!r}"))
            return
        if _session_path(safe).is_file():
            if not _ask(f"session '{safe}' already exists - reuse the "
                        f"name for a new empty session?"):
                print(dim(f"cancelled - /load {safe} to resume it instead."))
                return
    # /new keeps the current session under its own name, so loss only happens when it
    # isn't being autosaved at all.
    if nonempty and not will_persist:
        if not _ask("current conversation is NOT autosaved - start a new "
                    "one and lose it?"):
            print(dim("cancelled - /save [name] to keep it first."))
            return
    new = start_new_session(agent, cfg, want)
    print(dim(f"started new session '{new}'."))


def handle_connect_command(agent, cfg, arg):
    """/connect <[user@]host> [remote-path] [--ephemeral] - enter/switch a remote
    workspace: all file/exec tools then run there, transparently to the model. Bare
    /connect with saved [connect] hosts (or open ones) opens a menu. /connect remove
    <name> forgets a saved target. --ephemeral (Windows) forces the embeddable-Python
    runtime and wipes it on teardown (zero-trace; also dogfoods the embed path). Any
    failure leaves you where you were."""
    # Pull an --ephemeral / -e flag out of anywhere in the arg (order-agnostic). The
    # flag is a per-connect OVERRIDE of the cfg.ephemeral default (which a user can set
    # to always-wipe via /set ephemeral on); --no-ephemeral forces off for one connect.
    toks = arg.split()
    ephemeral = getattr(cfg, "ephemeral", False)
    if any(t in ("--ephemeral", "-e") for t in toks):
        ephemeral = True
    if "--no-ephemeral" in toks:
        ephemeral = False
    arg = " ".join(t for t in toks
                   if t not in ("--ephemeral", "-e", "--no-ephemeral"))
    if not arg:
        if cfg.connect_hosts or agent.remotes:
            active = agent.remote.host if agent.remote else None
            target = pick_connect_menu(cfg.connect_hosts,
                                       open_hosts=list(agent.remotes),
                                       active=active)
            if target:
                _do_connect(agent, cfg, target, ephemeral=ephemeral)
        else:
            print(dim("usage: /connect <[user@]host> [remote-path] [--ephemeral]  "
                      "(or add a [connect] section to the config)"))
    elif arg.split(maxsplit=1)[0] == "remove":
        _connect_remove(agent, cfg, arg.split(maxsplit=1)[1].strip() if len(arg.split(maxsplit=1)) > 1 else "")
    else:
        sp = arg.split(maxsplit=1)
        rpath = sp[1] if len(sp) > 1 else "."
        # MENU CONTRACT: a bare 1-based integer selects from the SAME ordered list the
        # bare-/connect menu shows (saved [connect] targets, then open-but-unsaved
        # sessions) - so `/connect 8` means "the 8th menu row", NOT a host literally
        # named 8 (which ssh would dial as a bare-integer address). See menu_resolve.
        picked = menu_resolve(sp[0], _connect_menu_order(cfg, agent))
        if picked is not None:
            _do_connect(agent, cfg, picked, rpath, offer_save=True, ephemeral=ephemeral)
            return
        # a bare token may be a saved [connect] name -> resolve to its target
        rhost = cfg.connect_hosts.get(sp[0], sp[0])
        _do_connect(agent, cfg, rhost, rpath, offer_save=True, ephemeral=ephemeral)


def _connect_remove(agent, cfg, name):
    """/connect remove <name> - forget a saved [connect] target. If it's the host the
    session is CURRENTLY connected to, the live connection stays up (tools keep running
    there); only the saved alias is dropped, with a note saying so."""
    if not name:
        if not cfg.connect_hosts:
            print(dim("no saved connect targets to remove."))
            return
        print(dim("usage: /connect remove <name>   (saved: "
                  + ", ".join(sorted(cfg.connect_hosts)) + ")"))
        return
    if name not in cfg.connect_hosts:
        print(red(f"no saved connect target named {name!r} "
                  f"(saved: {', '.join(sorted(cfg.connect_hosts)) or 'none'})."))
        return
    target = cfg.connect_hosts.pop(name)
    save_config(cfg)
    active = agent.remote.host if getattr(agent, "remote", None) else None
    if active and active in (name, target):
        print(green(f"removed saved connect target '{name}'."))
        print(yellow(f"  note: still CONNECTED to {active} for this session "
                     f"(tools keep running there); the alias is just no longer saved. "
                     f"/local to detach."))
    else:
        print(green(f"removed saved connect target '{name}' -> {target}."))


def handle_machines_command(agent, cfg, arg):
    """/machines - list the saved [machines] name->url aliases; /machines remove <name>
    forgets one. Aliases are added by editing the config's [machines] table; this command
    exists so a saved alias can be dropped without hand-editing the TOML."""
    sp = arg.split(maxsplit=1)
    sub = sp[0] if sp else ""
    if sub == "remove":
        name = sp[1].strip() if len(sp) > 1 else ""
        if not name:
            print(dim("usage: /machines remove <name>   (saved: "
                      + (", ".join(sorted(cfg.machines)) or "none") + ")"))
            return
        if name not in cfg.machines:
            print(red(f"no machine alias named {name!r} "
                      f"(saved: {', '.join(sorted(cfg.machines)) or 'none'})."))
            return
        url = cfg.machines.pop(name)
        save_config(cfg)
        print(green(f"removed machine alias '{name}' -> {url}."))
        return
    if not cfg.machines:
        print(dim("no saved machine aliases. Add them via a [machines] table in the config."))
        return
    print(bold("machines:"))
    for name, url in sorted(cfg.machines.items()):
        print(f"  {cyan(name)}  {dim(url)}")
    print(dim("  /machines remove <name> forgets one"))


def handle_tools_command(agent, cfg, arg):
    """/tools - menu to enable/disable the discovered lean-tools; persists the selection,
    refreshes the live tool set, and flags if a connected remote needs /reload."""
    if not agent.lean_tools.names():
        print(dim(f"no lean-tools in {agent.lean_tools.dir} "
                  f"(drop a .py with TOOL + run there)"))
    else:
        new = lean_tools_menu(agent.lean_tools)
        if new is not None:
            agent.lean_tools.enabled = new
            cfg.lean_tools_enabled = sorted(new)
            _run_lean_tool_setup(cfg, agent.lean_tools)   # run setup() for newly-enabled tools
            agent.refresh_tools()
            save_config(cfg)
            print(dim(f"enabled: {', '.join(sorted(new)) or '(none)'}"))
            if agent.remote:
                print(yellow(f"connected - run /reload to push to {agent.remote.host}."))


def _mcp_persist(agent, cfg):
    """Push the manager's live server map + enabled set back into cfg and save."""
    cfg.mcp_servers = dict(agent.mcp.servers)
    cfg.mcp_enabled = sorted(agent.mcp.enabled)
    save_config(cfg)


def _mcp_looks_unauthorized(info):
    """Heuristic: does a connect error string smell like a missing/bad credential
    (401/403/unauthorized/forbidden/auth) rather than a network/parse failure? Used to
    decide whether to offer the guided auth setup, same shape as provider login recovery."""
    s = (info or "").lower()
    return any(k in s for k in ("401", "403", "unauthorized", "forbidden",
                                "authentication", "not authenticated", "invalid token",
                                "missing token", "no token"))


def _mcp_guided_auth(agent, cfg, name):
    """Walk the user through adding auth to an HTTP MCP server that connected but was
    refused (or that they know needs a credential). Mirrors the provider login flow:
    offer bearer vs OAuth 2.1, write the auth block into the server config, remind them
    to export the secret env var (never store it in the file), then reconnect. Returns
    True if it reconnected with tools, False otherwise. No-op (returns None) with no TTY."""
    server = agent.mcp.servers.get(name)
    if not server or (server.get("transport") or "http") != "http":
        return None
    if not sys.stdin.isatty():
        print(dim(f"  auth needed for '{name}'. Add an auth block to "
                  "[mcp_servers." + name + "] in config.toml (see MCP.md), "
                  "then /mcp reconnect " + name + "."))
        return None
    print(yellow(f"  '{name}' needs authentication."))
    choice = pick_one("Auth method:",
                      ["OAuth 2.1 (recommended - short-lived, auto-refreshed)",
                       "Static bearer token",
                       "skip (set it up later in config.toml)"])
    if not choice or choice.startswith("skip"):
        print(dim("  skipped. Edit [mcp_servers." + name + "] in config.toml when ready "
                  "(guide: MCP.md), then /mcp reconnect " + name + "."))
        return None

    if choice.startswith("Static bearer"):
        env = _ask_line("  env var holding the token [UNE_PPT_KEY-style name]: ").strip()
        if not env:
            print(dim("  no env var given; aborted."))
            return None
        server["auth"] = {"type": "bearer", "token_env": env}
        if not os.environ.get(env):
            print(yellow(f"  reminder: export {env}=<token> in your shell "
                         "(never stored in config.toml), then rerun this."))
    else:
        url = server.get("url", "")
        print(dim("  discovering OAuth endpoints (.well-known)..."))
        meta = None
        try:
            meta = mcp_discover_auth(url)
        except Exception as e:
            print(dim(f"  discovery error: {e}"))
        reg_endpoint = (meta or {}).get("registration_endpoint")
        token_endpoint = (meta or {}).get("token_endpoint")

        # Path 1: dynamic client registration (DCR) - mint a client inline, no hand-curl.
        if reg_endpoint:
            print(dim(f"  registration endpoint: {reg_endpoint}"))
            regkey = _ask_line("  registration key if the endpoint is guarded "
                               "(blank if open): ").strip()
            scope = _ask_line("  scope [mcp:access]: ").strip() or "mcp:access"
            print(dim("  registering client..."))
            try:
                reg = mcp_register_client(reg_endpoint, registration_key=regkey or None,
                                          client_name=f"lean-coder ({name})", scope=scope)
            except Exception as e:
                reg = None
                print(dim(f"  registration error: {e}"))
            if not reg:
                print(yellow("  DCR failed (bad/missing registration key, or endpoint "
                             "refused). Falling back to manual client entry."))
            else:
                side = {"client_id": reg["client_id"],
                        "client_secret": reg.get("client_secret", ""),
                        "token_endpoint": reg.get("token_endpoint") or token_endpoint,
                        "registration_endpoint": reg_endpoint,
                        "scope": scope}
                if not side["token_endpoint"]:
                    side["token_endpoint"] = _ask_line("  token endpoint: ").strip()
                save_mcp_client(name, side)   # 0600 sidecar; secret never in config.toml
                server["auth"] = {"type": "oauth"}   # creds live in the sidecar
                print(green(f"  registered client {side['client_id']} "
                            f"(stored 0600 in mcp_auth/{name}.json)"))

        # Path 2: no DCR (or it failed) - collect the OAuth fields manually. A machine-
        # minted secret goes in the sidecar; a human-supplied one stays in an env var.
        if server.get("auth", {}).get("type") != "oauth" or not load_mcp_client(name):
            token_url = (token_endpoint or _ask_line("  OAuth token_url: ").strip())
            client_id = _ask_line("  client_id: ").strip()
            env = _ask_line("  env var holding the client secret "
                            "(blank to store the secret in the 0600 sidecar instead): ").strip()
            scope = _ask_line("  scope [mcp:access]: ").strip() or "mcp:access"
            if not (token_url and client_id):
                print(dim("  token_url and client_id are required; aborted."))
                return None
            if env:
                auth = {"type": "oauth", "token_url": token_url, "client_id": client_id,
                        "client_secret_env": env, "scope": scope}
                server["auth"] = auth
                if not os.environ.get(env):
                    print(yellow(f"  reminder: export {env}=<client-secret> in your shell "
                                 "(never stored in config.toml), then rerun this."))
            else:
                secret = _ask_line("  client secret (stored 0600 in the sidecar): ").strip()
                save_mcp_client(name, {"client_id": client_id, "client_secret": secret,
                                       "token_endpoint": token_url, "scope": scope})
                server["auth"] = {"type": "oauth"}

    _mcp_persist(agent, cfg)
    c = agent.mcp.conns.pop(name, None)
    if c:
        c.close()
    report = agent.mcp.connect_enabled()
    agent.refresh_tools()
    ok, info = report.get(name, (False, "?"))
    print(green(f"  ✓ {name}: {info} tools") if ok
          else yellow(f"  ✗ {name}: {info}  (fix env/config + /mcp reconnect {name})"))
    return ok


def _ask_line(prompt):
    try:
        return input(_rl_safe(prompt))
    except (EOFError, KeyboardInterrupt):
        print()
        return ""

def handle_mcp_command(agent, cfg, arg):
    """/mcp - manage MCP (Model Context Protocol) servers. No servers ship by default.
    Subcommands:
      /mcp                     enable/disable menu (per server; live)
      /mcp list                show configured servers + connection state
      /mcp add <name> <spec>   add a server, then enable it. <spec> is either
                                 a URL (https://…  -> http transport) or a shell
                                 command (stdio transport, e.g. `npx -y foo-mcp`).
                                 For auth/env, edit mcp_servers in config.toml.
      /mcp remove <name>       forget a server
      /mcp reconnect [name]    (re)connect all enabled servers, or just <name>
      /mcp clean               drop stale entries (enabled but no definition)
    Enabled servers' tools appear to the model namespaced mcp__<server>__<tool>; they
    run on the DRIVER (never a connected remote) and ride the rwe leash tier."""
    parts = arg.split(None, 2)
    sub = parts[0].lower() if parts else ""

    # Surface orphaned enables (enabled but no server definition) on every /mcp entry -
    # they contribute nothing and confuse ("config says enabled, /mcp shows none").
    stale = agent.mcp.stale_enabled()
    if stale and sub not in ("add", "remove", "reconnect", "clean"):
        print(yellow(f"  stale (enabled but not defined): {', '.join(stale)}"))
        print(dim(f"  -> /mcp clean [name]  removes them  {GLYPH['dot']}  or re-add with "
                  f"/mcp add <name> <url|command> to keep using it"))

    if sub == "clean":
        if not stale:
            print(dim("no stale MCP entries to clean."))
            return
        name = parts[1] if len(parts) > 1 else None
        if name is None and len(stale) > 1 and picker_capable():
            name = _pick_one_tty("Clean which stale MCP entry?", stale)
            if name in (None, _NO_TTY):
                return                                # cancelled / no tty
        if name is not None and name not in stale:
            print(yellow(f"'{name}' is not a stale MCP entry. stale: {', '.join(stale)}"))
            return
        dropped = agent.mcp.drop_stale(only=name)     # name=None -> drop all
        if not dropped:
            print(dim("no stale MCP entries to clean."))
            return
        agent.refresh_tools()
        _mcp_persist(agent, cfg)
        print(green(f"cleaned stale MCP entries: {', '.join(dropped)}"))
        return

    if not sub:
        if not agent.mcp.names():
            if not stale:
                print(dim("no MCP servers configured. Add one: /mcp add <name> <url|command>"))
            return
        new = mcp_servers_menu(agent.mcp)
        if new is not None:
            agent.mcp.enabled = set(new)
            report = agent.mcp.connect_enabled()
            agent.refresh_tools()
            _mcp_persist(agent, cfg)
            for n, (ok, info) in sorted(report.items()):
                print((green(f"  ✓ {n}: {info} tools") if ok
                       else yellow(f"  ✗ {n}: {info}")))
            print(dim(f"enabled: {', '.join(sorted(agent.mcp.enabled)) or '(none)'}"))
        return

    if sub == "list":
        if not agent.mcp.names() and not stale:
            print(dim("no MCP servers configured. /mcp add <name> <url|command>"))
            return
        print(bold("MCP servers:"))
        for n in stale:
            print(f"  {yellow('!')} {bold(n)}  {yellow('stale (enabled, not defined)')}  "
                  f"{dim('/mcp clean to remove, or /mcp add to re-add')}")
        for n in agent.mcp.names():
            on = n in agent.mcp.enabled
            c = agent.mcp.servers[n]
            tp = c.get("transport") or ("http" if c.get("url") else "stdio")
            conn = agent.mcp.conns.get(n)
            if conn and conn.connected:
                state = green(f"{len(conn.list_tools())} tools")
            elif conn and conn.error:
                state = red(f"error: {conn.error[:50]}")
            else:
                state = dim("not connected")
            tgt = c.get("url") or c.get("command") or "?"
            mark = green("●") if on else dim("○")
            print(f"  {mark} {bold(n)}  {dim(tp)}  {state}  {dim(tgt)}")
        return

    if sub == "add":
        if len(arg.split(None, 2)) < 3:
            print(dim("usage: /mcp add <name> <url|command>"))
            return
        name = parts[1]
        spec = parts[2]
        if name in agent.mcp.servers:
            print(yellow(f"'{name}' already exists - /mcp remove {name} first."))
            return
        if spec.startswith(("http://", "https://")):
            server = {"transport": "http", "url": spec}
            print(dim(f"added http server '{name}' -> {spec}"))
        else:
            argv = spec.split()
            server = {"transport": "stdio", "command": argv[0], "args": argv[1:]}
            print(dim(f"added stdio server '{name}' -> {spec}"))
        agent.mcp.servers[name] = server
        agent.mcp.enabled.add(name)
        report = agent.mcp.connect_enabled()
        agent.refresh_tools()
        _mcp_persist(agent, cfg)
        good, info = report.get(name, (False, "?"))
        if good:
            print(green(f"  ✓ connected, {info} tools"))
        elif server.get("transport", "http") == "http" and _mcp_looks_unauthorized(info):
            # Connected but refused -> offer the guided auth setup inline, the same way
            # a provider that 401s offers its login flow (rather than a dead red error).
            print(yellow(f"  ✗ {info}"))
            _mcp_guided_auth(agent, cfg, name)
        else:
            print(yellow(f"  ✗ {info}  (still saved + enabled; fix + /mcp reconnect {name})"))
        return

    if sub == "remove":
        if len(parts) < 2:
            print(dim("usage: /mcp remove <name>"))
            return
        name = parts[1]
        if name not in agent.mcp.servers:
            print(yellow(f"no such server: {name}"))
            return
        conn = agent.mcp.conns.pop(name, None)
        if conn:
            conn.close()
        del agent.mcp.servers[name]
        agent.mcp.enabled.discard(name)
        agent.refresh_tools()
        _mcp_persist(agent, cfg)
        print(dim(f"removed '{name}'."))
        return

    if sub == "reconnect":
        target = parts[1] if len(parts) > 1 else None
        if target and target not in agent.mcp.servers:
            print(yellow(f"no such server: {target}"))
            return
        if target:                       # force this one to reconnect
            c = agent.mcp.conns.pop(target, None)
            if c:
                c.close()
        report = agent.mcp.connect_enabled()
        agent.refresh_tools()
        for n, (ok, info) in sorted(report.items()):
            if target and n != target:
                continue
            if ok:
                print(green(f"  ✓ {n}: {info} tools"))
            else:
                print(yellow(f"  ✗ {n}: {info}"))
                srv = agent.mcp.servers.get(n, {})
                if (srv.get("transport") or "http") == "http" and _mcp_looks_unauthorized(info):
                    _mcp_guided_auth(agent, cfg, n)
        return

    print(dim(f"unknown /mcp subcommand: {sub}  (try /mcp ?)"))


def handle_reload_command(agent, cfg, arg):
    """/reload - re-scan lean-tools and pick up system-prompt edits; if connected,
    re-push code + enabled tools to the remote (ssh kept). Local core changes still need
    a restart."""
    names = agent.reload_lean_tools()
    # prompts are file-backed; pick up any edits to the system prompt now
    if agent.messages and agent.messages[0].get("role") == "system":
        agent.messages[0] = {"role": "system", "content": agent._system()}
    msg = f"reloaded lean-tools ({len(names)} found)"
    if agent.remote:
        try:
            agent.remote.reload(agent.lean_tools.enabled_paths())
            msg += f"; re-pushed code + lean-tools to {agent.remote.host} (ssh kept)"
        except (ConnectionError, OSError) as e:
            print(red(f"remote reload failed: {e}"))
    print(dim(msg + ". (local core changes still need a restart; /handover first)"))


def handle_local_command(agent, cfg, arg):
    """/local | /disconnect [host] - detach from the active remote (tools run locally
    again) but KEEP it open in the pool; /local <host> closes that pooled connection."""
    if arg:
        # /local <host> (or saved name): close that pooled connection
        target = cfg.connect_hosts.get(arg, arg)
        if target in agent.remotes:
            agent.drop_remote(target)
            print(dim(f"closed {target}."))
        else:
            print(dim(f"no open connection to {target}."))
    elif agent.remote:
        host = agent.remote.host
        agent.set_remote(None)        # detach but KEEP it open in the pool
        note = (f"  (still open: {', '.join(agent.remotes)} - /connect <host> to switch back)"
                if agent.remotes else "")
        print(dim(f"detached from {host}; tools run locally again.{note}"))
    else:
        print(dim("already local."))


def handle_compact_command(agent, cfg, arg):
    """/compact [keep] - summarize/drop old tool results, keeping the last `keep` in full
    (default COMPACT_KEEP)."""
    keep = COMPACT_KEEP
    if arg.isdigit():
        keep = int(arg)
    n, freed = agent.compact(keep)
    print(dim(f"compacted {n} tool result(s), ~{freed:,} tokens freed "
              f"(kept last {keep} in full)."))
    agent._print_ctx()


def handle_handover_command(agent, cfg, arg):
    """/handover - agentic handover: the model updates+commits durable docs, writes the
    handover between the markers, and calls the transiently surfaced finalize tool;
    history is then replaced with the handover."""
    print(dim("handover: update + commit durable docs, then write the "
              f"handover between {HANDOVER_MARK} markers and finalize…"))
    # Snapshot the full pre-handover conversation as a <name>-prehandover-N sidecar
    # (NOT under the live name) so it stays loadable AND the live session keeps its
    # name. Mirrors auto_handover: an explicit /handover must not fork a named session
    # into a fresh auto- one (the "it's auto again" bug) - you continue IN the session
    # you handed over, now carrying the compacted summary.
    try:
        rhost = agent.remote.host if agent.remote else None
        origin = getattr(agent, "autosave_name", "") or "session"
        existing = [nm for nm, _ in list_sessions()]
        save_session(list(agent.messages), cfg, _prehandover_name(origin, existing),
                     remote=rhost, pinned_plan=getattr(agent, "pinned_plan", ""),
                     notes=getattr(agent, "notes", None))
    except Exception:
        pass
    summary = agent.handover()
    if summary is None:
        print(yellow("\nno handover captured - history left intact. (Write it "
                     f"between {HANDOVER_MARK} markers as visible text.)"))
    else:
        # Keep the live autosave_name: the next autosave writes the compacted history
        # back into the SAME session, so a named session stays named.
        print(dim("\nhistory replaced with the handover above; "
                  "new cache boundary set on the summary."))
        agent._print_ctx()


def handle_autosave_command(agent, cfg, arg):
    """/autosave [on|off] - toggle persisting the conversation; no arg flips it."""
    want = _parse_toggle(arg, cfg.autosave)
    if want is None:
        print(dim("usage: /autosave [on|off]"))
        return
    cfg.autosave = want
    autosave_config(cfg)
    print(dim(f"autosave -> {'on' if cfg.autosave else 'off (amnesic)'}"))


def _apply_think(agent, cfg, arg, key):
    """Shared body for /think | /effort. `key` is 'thinking' or
    'effort'. No arg -> a menu of valid values. With a provider active these are
    capability-gated settings; on native Ollama thinking stays the think=true/false
    toggle."""
    if not arg:
        choices = _arg_completions(agent, cfg,
                                   "/effort" if key == "effort" else "/think")
        _spec = agent.active_provider()
        _native_think = key == "thinking" and (_spec is None or _spec["name"] == "ollama")
        if _native_think:
            current = "off" if not cfg.think else "on"
        else:
            current = cfg.setting(key)
        arg = pick_one(f"{key}:", choices, current) or ""
    if arg:
        if not _set_known(agent, cfg, key, arg):
            if key == "thinking":
                cfg.think = arg.lower() not in ("off", "false", "no", "0", "none")
                print(dim(f"think -> {cfg.think}"))
            else:
                print(dim("/effort applies to a provider that supports it "
                          "(not the ollama backend)."))
        autosave_config(cfg)


def handle_think_command(agent, cfg, arg):
    """/think [on|off|...] - thinking level. No arg -> a menu."""
    _apply_think(agent, cfg, arg, "thinking")


def handle_effort_command(agent, cfg, arg):
    """/effort [low|medium|...] - reasoning effort for providers that support it. No arg
    -> a menu."""
    _apply_think(agent, cfg, arg, "effort")


def handle_approve_command(agent, cfg, arg):
    """/approve [mode] - set the confirm cadence (one of APPROVAL_MODES); no arg -> menu."""
    mode = arg.strip().lower() or (
        pick_one("approval:", list(APPROVAL_MODES), cfg.approval) or "")
    if mode in APPROVAL_MODES:
        set_approval(cfg, mode)
        autosave_config(cfg)
        print(dim(f"approval -> {cfg.approval}"))
    elif mode:
        print(dim(f"approval modes: {', '.join(APPROVAL_MODES)}"))


def handle_ctx_command(agent, cfg, arg):
    """/ctx - print the context-window meter."""
    agent._print_ctx()


def handle_activity_command(agent, cfg, arg):
    """/activity [n] - show the system-activity log: the automatic behaviours the
    system ran on your behalf (auto-compact, handover, auto-evict, 429->Haiku
    fallback, token refresh), newest last, each with a reason. `n` limits to the
    last n entries (default 20; `all` for everything)."""
    log = getattr(agent, "_activity", [])
    if not log:
        print(dim("  no automatic activity yet this session."))
        return
    arg = (arg or "").strip()
    if arg == "all":
        shown = log
    else:
        try:
            n = int(arg)
        except (TypeError, ValueError):
            n = 20
        shown = log[-n:]
    hidden = len(log) - len(shown)
    print(bold(f"system activity ({len(log)} event(s)"
               + (f", showing last {len(shown)}" if hidden > 0 else "") + "):"))
    for e in shown:
        ts = time.strftime("%H:%M:%S", time.localtime(e["t"]))
        print(f"  {dim(ts)}  {yellow(e['kind'])}  {e['summary']}")
        if e.get("detail"):
            print(dim(f"           {e['detail']}"))
    if hidden > 0:
        print(dim(f"  ... {hidden} older; /activity all for everything."))


def handle_expand_command(agent, cfg, arg):
    """/expand [N] - re-print the FULL (untruncated) args AND captured result of a
    recent tool call. Bare /expand = the most recent call; /expand N = the call
    tagged '#N' on its line. On-screen tool-call lines truncate args to 50 chars;
    this shows all of it (a diff renders colorized) plus the tool's output, capped
    so it can't flood the terminal.

    /expand msg [N] (alias: messages) - re-print the last N conversation messages
    (default 10) in the resume-banner style, so you can scroll back without a reload.
    N is clamped to what exists (a silly count never errors)."""
    arg = (arg or "").strip()
    # /expand msg [N] - conversation view (distinct from the tool-call ring below).
    parts = arg.split()
    if parts and parts[0].lower() in ("msg", "messages", "m"):
        n = 10
        if len(parts) > 1:
            try:
                n = int(parts[1])
            except ValueError:
                print(dim("  usage: /expand msg [N]  (N = how many recent messages; default 10)"))
                return
        if not _render_message_tail(agent.messages, n=n, width=100):
            print(dim("  no messages yet this session."))
        return
    calls = getattr(agent, "_tool_calls", None)
    if not calls:
        print(dim("  no tool calls to expand yet this session."))
        return
    arg = arg.lstrip("#")
    if not arg:
        entry = calls[-1]
    else:
        try:
            want = int(arg)
        except ValueError:
            print(dim("  usage: /expand [N]  (N = the #id on a tool-call line; bare = newest)"))
            return
        entry = next((e for e in calls if e["id"] == want), None)
        if entry is None:
            lo, hi = calls[0]["id"], calls[-1]["id"]
            print(dim(f"  no tool call #{want} (have #{lo}..#{hi}, last 100)."))
            return
    print(dim(f"  {GLYPH['tool']} #{entry['id']} {entry['name']}"))
    print(_render_tool_call(entry))


def handle_help_command(agent, cfg, arg):
    """/help | /h | /? - list builtin + lean-tool-registered commands."""
    extra = [(c, h) for c, (_, h) in sorted(_lean_tool_commands.items())]
    print(render_help(extra))


# Loop-control sentinel: a builtin handler returns this to tell the repl dispatch to
# leave the loop (only /quit does). Every other handler returns None -> dispatch
# continues. Keeps handlers pure (agent, cfg, arg) -> control|None, no exception-for-flow.
REPL_EXIT = object()


def handle_quit_command(agent, cfg, arg):
    """/quit | /exit | /q - close any open remotes and leave the repl."""
    agent.close_all_remotes()
    print("bye")
    return REPL_EXIT


# The builtin slash-command dispatch table: command-or-alias -> handler. ONE lookup for
# every builtin (aliases map to the same handler), replacing the former if/elif chain.
# All handlers share the (agent, cfg, arg) signature; lean-tool-registered commands keep
# their own `_lean_tool_commands` registry, consulted after this table (A3 unifies them).
_BUILTIN_COMMANDS_TABLE = {
    "/quit": handle_quit_command, "/exit": handle_quit_command, "/q": handle_quit_command,
    "/clear": handle_clear_command,
    "/new": handle_new_command,
    "/prompt": handle_prompt_command,
    "/sh": run_sh_command,
    "/connect": handle_connect_command,
    "/machines": handle_machines_command,
    "/tools": handle_tools_command,
    "/mcp": handle_mcp_command,
    "/reload": handle_reload_command,
    "/local": handle_local_command, "/disconnect": handle_local_command,
    "/compact": handle_compact_command,
    "/handover": handle_handover_command,
    "/session": handle_session_command,
    "/save": handle_save_command,
    "/load": handle_load_command,
    "/bg": handle_bg_command,
    "/note": handle_note_command,
    "/autosave": handle_autosave_command,
    "/incognito": handle_incognito_command,
    "/leash": handle_leash_command,
    "/askread": handle_askread_command,
    "/think": handle_think_command,
    "/effort": handle_effort_command,
    "/provider": handle_provider_command,
    "/providers": handle_provider_command,
    "/usage": handle_usage_command,
    "/info": handle_info_command,
    "/set": handle_settings_command,
    "/model": handle_model_command,
    "/models": handle_model_command,
    "/approve": handle_approve_command,
    "/ctx": handle_ctx_command,
    "/activity": handle_activity_command,
    "/expand": handle_expand_command,
    "/help": handle_help_command, "/h": handle_help_command, "/?": handle_help_command,
}

# Commands that persist a config change: the dispatch autosaves the config AFTER the
# handler returns (these handlers have many early-return paths, so the autosave belongs
# in the dispatch, exactly where the former chain ran it - not inside the handler).
_CMD_AUTOSAVE = frozenset({"/provider", "/providers", "/set"})

# Single source of truth for shadow protection: every builtin command + alias, derived
# from the dispatch table so the protected set can never drift from what's dispatched
# (this also covers aliases like /disconnect that the old hand-maintained list missed).
_BUILTIN_COMMANDS = frozenset(_BUILTIN_COMMANDS_TABLE)


def _command_help(cmd, handler, short):
    """Per-command help for `/cmd ?` (or `/cmd help`). Single source of truth: the
    handler's own docstring - no separate registry to drift. Prints the canonical
    name + any aliases (both derived from the dispatch table), the one-line /help
    description, then the full dedented docstring. `short` is the /help one-liner
    (None for lean-tool commands - we fall back to the docstring alone)."""
    aliases = sorted(a for a, h in _BUILTIN_COMMANDS_TABLE.items()
                     if h is handler and a != cmd)
    head = bold(cyan(cmd)) + (dim("  (aliases: " + ", ".join(aliases) + ")") if aliases else "")
    print(head)
    if short:
        print("  " + dim(short))
    doc = inspect.getdoc(handler) or ""
    if doc:
        print()
        for ln in doc.splitlines():
            print("  " + ln)


def dispatch_command(agent, cfg, cmd, arg):
    """The one slash-command lookup: builtin table first, then the lean-tool registry,
    else 'unknown command'. Returns True iff the repl should exit (only /quit signals
    it, via the REPL_EXIT sentinel). Builtin handlers run unguarded (core code); a
    lean-tool handler is wrapped so a buggy plugin can't take the loop down. Folds in
    the /provider + /set post-handler config autosave (app-config + provider knobs).

    Uniform command contract: `/<cmd> ?` (or `/<cmd> help`) prints that command's
    own help (its docstring) instead of running it - works for every builtin and
    lean-tool command, so a user never has to guess a command's args.

    Universal emergency escape: a leading `--plain`/`--fallback` on ANY command forces
    the next interactive menu to use the bulletproof numbered fallback instead of the
    raw-mode picker - the way out of a garbled render on a mis-detected terminal, with
    no restart. The flag is stripped before the handler sees the arg."""
    global _FORCE_PLAIN_ONCE
    _plain, arg = _wants_plain(arg)
    if _plain:
        _FORCE_PLAIN_ONCE = True
    if arg.strip().lower() in ("?", "help") and cmd not in ("/help", "/h", "/?"):
        handler = _BUILTIN_COMMANDS_TABLE.get(cmd)
        if handler is not None:
            short = dict(HELP_COMMANDS).get(cmd) or next(
                (d for c, d in HELP_COMMANDS if c.split()[0] == cmd), None)
            _command_help(cmd, handler, short)
            return False
        if cmd in _lean_tool_commands:
            fn, hl = _lean_tool_commands[cmd]
            _command_help(cmd, fn, hl)
            return False
        print(yellow(f"unknown command {cmd} - /help"))
        return False
    handler = _BUILTIN_COMMANDS_TABLE.get(cmd)
    if handler is not None:
        if handler(agent, cfg, arg) is REPL_EXIT:
            return True
        if cmd in _CMD_AUTOSAVE:      # persist a settings change across launches
            autosave_config(cfg)
        return False
    if cmd in _lean_tool_commands:        # lean-tool-registered command
        try:
            _lean_tool_commands[cmd][0](agent, cfg, arg)
        except Exception as e:
            print(red(f"error in {cmd}: {e}"))
        return False
    print(yellow(f"unknown command {cmd} - /help"))
    return False


def _input_or_wake(prompt, wake_check=None, tick=0.25):
    """input() that can also WAKE on a background event, for the non-composer path
    (no-TTY-but-live-stdin: piped-but-open stdin, some Termux/ssh setups). When
    wake_check is None it's a plain blocking input() (identical behaviour - the
    common case). When set, stdin is polled with select() every `tick` seconds; on a
    ready line we read + return it, and on each idle tick we call wake_check() - a
    truthy return is a synthesised turn we return as if typed (NO operator input).
    Falls back to a plain blocking input() where select-on-stdin isn't possible
    (Windows, no fileno) so nothing regresses. Raises EOFError on stdin close, as
    input() does, so the caller's ^D handling is unchanged."""
    if wake_check is None:
        return input(prompt).strip()
    try:
        fd = sys.stdin.fileno()
    except (ValueError, OSError, AttributeError):
        return input(prompt).strip()        # no pollable stdin -> plain blocking read
    sys.stdout.write(prompt)
    sys.stdout.flush()
    while True:
        try:
            r, _, _ = select.select([fd], [], [], tick)
        except (OSError, ValueError):
            return input("").strip()        # select unusable -> block for the rest
        if r:
            data = sys.stdin.readline()
            if data == "":                  # EOF (stdin closed) - match input()
                raise EOFError
            return data.strip()
        woke = wake_check()
        if woke:
            sys.stdout.write("\n")          # leave the prompt line cleanly
            sys.stdout.flush()
            return woke


def repl(cfg: Config, resume=None):
    global _active_agent
    agent = Agent(cfg)
    _active_agent = agent                 # expose to lean-tools via active_remote()
    # Safety net: the explicit exit paths (/quit, ^D, ^C^C) call close_all_remotes()
    # to wipe pushed remotes, but a non-graceful death (SIGTERM, window/terminal close,
    # an uncaught error) skips them. Register it with atexit too so ANY interpreter exit
    # attempts the clean wipe over still-live masters; if the process is hard-killed
    # (SIGKILL) or the link is already gone, the remote self-wipe watchdog is the final
    # backstop. close_all_remotes is idempotent (empties the pool), so double-firing on a
    # graceful exit is harmless.
    atexit.register(agent.close_all_remotes)
    # Driver-only startup hooks: run each enabled lean-tool's setup() on the AGENT's
    # own manager instance (not a throwaway) - a tool that stashes hooks in its own
    # module globals (e.g. dispatch_worker's _H) needs the SAME module instance whose
    # run() later fires. _setup_done keeps it at most once per tool per process.
    _run_lean_tool_setup(cfg, agent.lean_tools)
    _connect_mcp_startup(agent)   # connect enabled MCP servers + fold their tools in
    # Clear .lock files left by instances that were killed without a clean exit (they
    # age out logically after _LOCK_STALE_SECS but the files otherwise pile up forever).
    # Before we claim any session below, so it can never remove our own fresh lock.
    _prune_stale_locks()
    # Release our session lock on exit so another instance doesn't see a stale
    # "live elsewhere" and prompt to take over a session nobody is holding.
    atexit.register(lambda: _release_lock(getattr(agent, "autosave_name", "")))

    # The active backend is ALWAYS a provider now. Any pre-activation transport
    # setup (e.g. the ollama provider's tiered host failover + per-host default
    # model) runs inside _maybe_autostart_provider via the chosen provider's
    # prepare() hook, BEFORE activate_provider picks the model + context window up.
    prov = _maybe_autostart_provider(agent, cfg)

    if prov is None:
        print(yellow("No model backend is active yet. To get started:"))
        print(dim("  /provider            list backends and enable one (Anthropic, "
                  "Gemini, Groq, OpenRouter, ...)"))
        print(dim("  /provider login <name>   paste an API key for a hosted backend"))
        print(dim("  or run a local model with Ollama (https://ollama.com), then relaunch"))
        print(dim("  /model               pick a model once a backend is up   "
                  f"{GLYPH['dot']}  /help  all commands"))
        ctx_note, capped_from = "-", None
    elif prov["name"] == "ollama":
        ctx_note = "auto-detected" if cfg.auto_num_ctx else "pinned"
        capped_from = getattr(cfg, "_ollama_capped_from", None)
    else:
        ctx_note, capped_from = "provider", None

    _setup_readline(agent, cfg)

    location = _provider_label(cfg)

    def _print_full_banner():
        """The orientation banner (model, cwd, num_ctx, composer). Printed on a FRESH
        start; SUPPRESSED when a session is resumed, since the always-on status rows
        already carry model @ provider / session / tools / ctx - so a resume ends on the
        'resumed …' line + the 3 status rows, not a wall of repeated state."""
        overhead = messages_tokens([agent.messages[0]], agent.tool_defs)
        print(bold("lean-coder") + dim(f"  {cfg.active_model()} @ {location}"))
        print(dim(f"  cwd: {cfg.cwd}"))
        _badge = approval_badge(cfg)
        print(dim(f"  num_ctx: {cfg.ctx_window():,} ({ctx_note})  {GLYPH['dot']}  baseline overhead "
                  f"(system + {len(agent.tool_defs)} tools): ~{overhead} tokens")
              + (dim(f"  {GLYPH['dot']}  ") + _badge if _badge else ""))
        if capped_from:
            print(yellow(f"  note: model max is {capped_from:,}; capped to "
                         f"{AUTO_CTX_CAP:,} to avoid OOM - pass --num-ctx to raise it."))
        # Composer state: a broken input channel can't paste diagnostics back, so the
        # program self-reports WHY the pinned input is or isn't active.
        if not cfg.composer:
            _comp_state = "off (composer=false / --no-composer)"
        elif not (_TTY and sys.stdin.isatty()):
            _comp_state = "off (no tty: stdout/stdin not a terminal)"
        else:
            _comp_state = "on (pinned input)"
        print(dim(f"  composer: {_comp_state}"))

    def _take_over_or_fork(name) -> bool:
        """If `name` is held live by another instance, ask whether to take it over.
        Returns True to load it (the lock gets claimed by the first autosave),
        False to leave it alone and start a fresh auto- session here."""
        if cfg.incognito or not _lock_is_live(name):
            return True
        if _ask(f"session '{_session_name_ok(name)}' is live in another instance - "
                f"take it over?"):
            return True
        print(dim("  starting a fresh session here instead."))
        return False

    # A resumed session ENDS on the 'resumed …' instruction line (the last thing the
    # user is likely to read), then the always-on 3-row status bar carries model /
    # session / tools / ctx / perms - so we skip the full orientation banner, the
    # per-session state read-out, and the trailing session/help lines (all of which
    # the status rows already show). A FRESH start prints the full banner instead.
    resumed = False
    if resume:
        try:
            if not _take_over_or_fork(resume):
                raise FileNotFoundError  # decline -> fall through to a clean start
            data = load_session(resume)
            meta = data.get("meta", {})
            _restore_session_state(agent, cfg, meta)  # backend/model/perms
            msgs = data.get("messages", [])
            n = agent.restore(msgs)
            # Continue IN the resumed session (named or auto-): further work autosaves
            # back into it, so it stays current instead of forking to a new auto- name.
            agent.autosave_name = _session_name_ok(resume)
            if not cfg.incognito:
                _write_lock(agent.autosave_name)   # claim it now (take-over is real)
            _maybe_reconnect(agent, cfg, meta)
            _print_session_tail(msgs)
            print(yellow(f"  resumed '{_session_name_ok(resume)}' ({n} turns) - "
                         f"/clear for a fresh one, /load to switch.") + "\n")
            resumed = True
        except FileNotFoundError:
            print(yellow(f"--resume: no such session '{resume}' "
                         f"(starting fresh; /load to see saved ones)"))
        except (json.JSONDecodeError, OSError) as e:
            print(yellow(f"--resume: session '{resume}' is unreadable "
                         f"({type(e).__name__}; file may be corrupt) - starting fresh."))
    elif cfg.autosave and not cfg.incognito:
        # Auto-load the session you were last actually in: the most-recently-used one,
        # EXCLUDING pre-handover snapshots (those are safety checkpoints, not the live
        # thread). cwd is a HINT, not a hard filter - scoping auto-load by cwd silently
        # skipped the genuinely-most-recent session whenever it was last used from a
        # different dir (e.g. a session worked over a remote connection, or a synced
        # one), resuming a stale same-cwd session instead. If the resumed session was
        # last used in another cwd, we say so rather than hide it.
        # (Incognito starts clean and leaves no trace, so it never auto-loads.)
        here = str(cfg.cwd)
        # Most-recent-first, excluding snapshots. A session that's live in ANOTHER
        # instance is skipped (never silently stolen) and we fall through to the next
        # candidate - so restarting while an OLDER session is open elsewhere still
        # resumes YOUR genuinely-most-recent one, not a fresh blank. Only start fresh
        # when every candidate is either gone or locked.
        _rows = [r for r in _session_rows() if not _is_snapshot(r[0])]
        _busy = next((r for r in _rows if _lock_is_live(r[0])), None)
        cand = next((r for r in _rows if not _lock_is_live(r[0])), None)
        if not cand and _busy:
            # Nothing free to resume; the only candidate(s) are open elsewhere.
            print(dim(f"last session '{_busy[0]}' is open in another instance - "
                      f"starting fresh here (/load {_busy[0]} to take it over)."))
        elif cand:
            last = cand[0]
            try:
                data = load_session(last)
                meta = data.get("meta", {})
                _restore_session_state(agent, cfg, meta)
                msgs = data.get("messages", [])
                n = agent.restore(msgs)
                # Continue IN the resumed session (named or auto-) so it stays current;
                # claim its lock now so the first autosave owns it.
                agent.autosave_name = _session_name_ok(last)
                _write_lock(agent.autosave_name)
                _maybe_reconnect(agent, cfg, meta)
                _print_session_tail(msgs)
                other = meta.get("cwd")
                cwd_note = f"  (last used in {other})" if other and other != here else ""
                print(yellow(f"  resumed last session '{last}' ({n} turns) - "
                             f"/clear for a fresh one, /load to switch.") + dim(cwd_note) + "\n")
                resumed = True
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                pass

    if not resumed:                              # FRESH start: full orientation banner
        _print_full_banner()
        if cfg.incognito:
            print(magenta(f"  session: {GLYPH['ghost']} incognito")
                  + dim(" (not saved locally; provider may still log)"))
        else:
            _sess_state = (f"autosaving -> {agent.autosave_name}" if cfg.autosave
                           else "off (amnesic)  /autosave on to enable")
            print(dim(f"  session: {_sess_state}"))
        if cfg.leash != "rwe":
            print(dim(f"  leash: {cfg.leash} ({_LEASH_GRANTS[cfg.leash]})"))
        print(dim("  /help for commands, /quit to exit\n"))

    pending_exit = False           # armed by one ^C at the prompt; a second exits
    pending_inputs = []            # lines the user typed via the composer mid-turn
    use_composer = cfg.composer and _TTY and sys.stdin.isatty()
    # Persistent idle-prompt composer: owns the input line at idle (correct
    # multi-line paste rendering + Tab/history/emacs keys), and its history
    # persists to disk across the session. readline stays the no-TTY fallback.
    idle_comp = None
    if use_composer:
        idle_comp = Composer()
        idle_comp.completer = _make_composer_completer(agent, cfg)
        idle_comp.load_history()
    seen_turn = False              # draw a turn-separator rule before every prompt but the first
    status_key = object()          # last-shown status signature; sentinel forces the first print
    prompt_no = 0                  # prompts drawn, for the "reprint every N" cadence
    status_shown_at = 0            # prompt_no when the block was last printed
    while True:
        # a full-width rule between turns so user vs model turns are easy to tell apart
        # (the prompt's own colour/dot distinguishes who is speaking within a turn)
        if seen_turn:
            print(hr())
        # Status block above the prompt. statusline_every controls the cadence: 1 (default)
        # = every prompt, N = every N prompts, 0 = only when it changes. A real state change
        # (settings/model/tools/plan - the ctx row's per-turn token drift is excluded from
        # the signature) ALWAYS reprints regardless of cadence. /info and /ctx show it too.
        prompt_no += 1
        _rows = _status_rows(agent, cfg)
        _key = _status_key(_rows)
        _every = getattr(cfg, "statusline_every", 1)
        _due = _every >= 1 and (prompt_no - status_shown_at) >= _every
        if _rows and (_key != status_key or _due):
            for _row in _rows:
                print(_row)
            status_key = _key
            status_shown_at = prompt_no
        # prompt shows the active location so you always know where you are
        indicator = f"[remote: {agent.remote.host}] " if agent.remote else ""
        _pg = GLYPH["prompt_remote"] if agent.remote else GLYPH["prompt"]
        if idle_comp is not None:
            idle_comp.remote = bool(agent.remote)   # swap the input-row glyph when remote
        if agent._queued_turns:    # a command queued a full turn (e.g. /prompt use) - run it
            line = agent._queued_turns.pop(0)
            print(bold(cyan(indicator + _pg + " ")) + dim("(queued prompt)"))
        elif pending_inputs:       # drain composer typeahead before prompting anew
            line = pending_inputs.pop(0).strip()
            if line:
                print(_operator_turn_lines(cfg.user_name, line, agent))
        else:
            try:
                _wake = (agent.bg_wake_turn
                         if (cfg.wake_on_bg_finish or agent._has_bg_optins())
                         else None)
                if idle_comp is not None:
                    # Idle composer owns the line: correct multi-line paste render
                    # + Tab/history/emacs. Falls back to input() if it can't take
                    # the TTY (RuntimeError). The prompt glyph is drawn by the
                    # composer's own input row, so pass only the remote indicator.
                    try:
                        line = idle_comp.read_line(
                            prompt_status=(indicator.strip() or None),
                            wake_check=_wake).strip()
                        # stop_tty() cleared the pinned input rows; echo the
                        # submitted line into scrollback with the operator's yellow
                        # accent bar so a human turn stands out from AI + tool lines.
                        if line:
                            print(_operator_turn_lines(cfg.user_name, line, agent))
                    except RuntimeError:
                        line = _input_or_wake(
                            _rl_safe(bold(cyan(indicator + _pg + " "))), _wake)
                else:
                    line = _input_or_wake(
                        _rl_safe(bold(cyan(indicator + _pg + " "))), _wake)
                pending_exit = False   # any input (even empty) disarms the exit
            except EOFError:           # Ctrl-D
                agent.close_all_remotes()
                if idle_comp is not None and not cfg.incognito:
                    idle_comp.save_history()   # incognito leaves no on-disk input trace
                print("\nbye")
                return
            except KeyboardInterrupt:  # ^C at the prompt: first cancels, second exits
                if pending_exit:
                    agent.close_all_remotes()
                    if idle_comp is not None and not cfg.incognito:
                        idle_comp.save_history()   # incognito leaves no on-disk input trace
                    print("\nbye")
                    return
                pending_exit = True
                print(yellow("\npress ^C again to exit"))
                continue
        if not line:
            continue
        seen_turn = True           # next loop draws a separator before the prompt
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1].strip() if len(parts) > 1 else ""
            if dispatch_command(agent, cfg, cmd, arg):    # True only on /quit
                return
            continue
        if use_composer:
            comp = Composer()
            comp.remote = bool(agent.remote)   # mid-turn input row shows » when remote
            interrupted = False
            with composer_session(comp) as engaged:
                try:
                    if engaged:
                        agent.run_turn(line)
                    else:
                        # Composer couldn't take the TTY. Fall back to the SAME
                        # protection as the classic path - without this the turn
                        # ran with echo on, garbling the stream and line-buffering
                        # typeahead that then auto-fired as turns.
                        with suppress_echo():
                            agent.run_turn(line)
                except KeyboardInterrupt:
                    interrupted = True
            queued = comp.pop_queued() if engaged else []
            # Carry any UNSUBMITTED partial line (typed mid-turn but not Enter'd) from
            # the throwaway turn-composer into the persistent idle composer, so it
            # survives the turn->idle handoff instead of being eaten with `comp`. The
            # idle composer's read_line() preserves + redraws whatever is in its buf.
            _partial = comp.buf if engaged else ""
            if _partial and idle_comp is not None:
                idle_comp.buf = (idle_comp.buf + _partial) if idle_comp.buf else _partial
                idle_comp.cur = len(idle_comp.buf)
            # Slash-command lines typed mid-turn are NEVER sent to the model: peel them
            # off and run them as commands (they dispatch at the top of the loop). Only
            # plain messages go through the combine/separate/discard drain. This stops a
            # mid-turn '/help' (etc.) from being folded into a model turn as literal text.
            cmds, queued = _split_queued_commands(queued)
            if interrupted:
                if queued:
                    # ^C WITH something typed = "stop this task and send what I typed
                    # now" (steering): combine the typed lines into one redirect, drop
                    # any older backlog (runaway can't outlive ^C).
                    pending_inputs.clear()
                    pending_inputs.append("\n".join(queued))
                    print(yellow("\n^C - stopped; sending your message now"
                                 + (f" ({len(queued)} lines combined)" if len(queued) > 1 else "")))
                else:
                    # ^C with nothing typed = just stop; clear any waiting backlog.
                    dropped = len(pending_inputs)
                    pending_inputs.clear()
                    print(yellow("\n^C - stopped"
                                 + (f"; cleared {dropped} queued message(s)" if dropped else "")))
            elif len(queued) == 1:
                # one typed-ahead message auto-runs next - tell the user so (no silent magic)
                pending_inputs.extend(queued)
                print(dim(f"  {GLYPH['dot']} 1 message queued -> sending as the next turn."))
            elif queued:
                # several queued -> explicit 3-way (never a silent burst, never an
                # ambiguous "send all now?"). Default (Enter) = combine into one turn.
                print(dim(f"{len(queued)} messages queued while working:"))
                for q in queued:
                    print(dim(f"  {GLYPH['dot']} {q}"))
                try:
                    raw = input(_rl_safe(bold(
                        "[c]ombine into 1 turn  ·  [s]eparate turns (one each)  ·  [d]iscard?  [c] ")))
                except (EOFError, KeyboardInterrupt):
                    raw = "d"
                choice = _drain_choice(raw)
                if choice == "combine":
                    pending_inputs.append("\n".join(queued))
                    print(dim(f"combined {len(queued)} into one turn."))
                elif choice == "discard":
                    print(dim("discarded queued messages."))
                else:
                    pending_inputs.extend(queued)
                    print(dim(f"queued {len(queued)} as separate turns (one per turn)."))
            # queued commands run as commands, after any drained messages, in order typed
            if cmds:
                pending_inputs.extend(cmds)
                print(dim(f"  {GLYPH['dot']} {len(cmds)} command(s) queued -> running next: "
                          + ", ".join(c.split()[0] for c in cmds)))
            autosave_session(agent, cfg)  # crash-safe: persist after every turn
            flush_stdin()                # drop any stray typeahead left in the tty
        else:
            interrupted = False
            try:
                with suppress_echo():    # don't let typeahead garble the stream
                    agent.run_turn(line)
            except KeyboardInterrupt:
                interrupted = True
            if interrupted:
                dropped = len(pending_inputs)
                pending_inputs.clear()
                print(yellow("\n^C - stopped"
                             + (f"; cleared {dropped} queued message(s)" if dropped else "")))
            autosave_session(agent, cfg)  # crash-safe: persist after every turn
            flush_stdin()                # drop typeahead buffered during the turn


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------

def _parse_grant(text: str) -> dict:
    """Parse the ===GRANT=== block's simple 'key: value' lines into a dict. Unknown
    keys are ignored; junk lines are skipped. Pure -> tested."""
    grant = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k and v:
            grant[k] = v
    return grant


def _brief_result_path(brief_file, result_file):
    return result_file or (str(brief_file) + ".result")


def run_agent_brief(args) -> int:
    """Headless WORKER role (`--agent-run`): read a brief file, run ONE brief to
    completion, write the model's final answer (its ===RESULT=== block, or the whole
    final message) to the result file, exit. No composer, no readline, no banner, no
    interactive confirm (approval=auto - nobody's there), no session autoload, no
    ===NEXT=== autostart. The brain runs HERE (this box's activated provider + auth);
    the brief's GRANT sets leash/model/cwd/limits (the dispatch tool caps them <= the
    parent's). Returns a process exit code: 0 ok, non-zero on a setup/run failure so
    the parent's '<log>.exit' sidecar reflects pass/fail. See dispatch_worker."""
    brief_file = args.brief_file
    resultf = _brief_result_path(brief_file, args.result_file)

    def _fail(msg, code=1):
        print(red(f"agent-run: {msg}"))
        try:
            Path(resultf).write_text(f"{RESULT_MARK}\nERROR: {msg}\n{RESULT_MARK}\n")
        except OSError:
            pass
        return code

    if not brief_file:
        return _fail("no --brief-file given")
    try:
        raw = Path(brief_file).read_text()
    except OSError as e:
        return _fail(f"cannot read brief file: {e}")
    brief = _extract_marked(raw, BRIEF_MARK)
    if not brief:
        return _fail(f"brief file has no {BRIEF_MARK} block")
    grant = _parse_grant(_extract_marked(raw, GRANT_MARK) or "")

    # Build the worker's config from CLI + grant (grant wins - it's the parent's grant).
    cfg = load_config(args)
    cfg.approval = "auto"                 # headless: no operator to confirm
    cfg.composer = False
    cfg.autosave = False                  # a worker doesn't autosave/auto-load sessions
    cfg.auto_handover = False             # one brief, no self-compaction handover
    # Mark this process a headless worker: no human is watching, so a provider that
    # would fast-fail an interactive turn on a transient throttle should instead wait
    # it out (see anthropic_plan _send_with_retry - it backs off 429 in worker mode).
    os.environ["LEANCODER_HEADLESS_WORKER"] = "1"
    if grant.get("leash"):
        cfg.leash = _norm_leash(grant["leash"]) or cfg.leash
    # The worker attaches its TOOLS to the parent's remote when one is passed. In that
    # case the grant 'cwd' is a path on the REMOTE (not the driver), so it must NOT
    # constrain the driver-local cfg.cwd - leave cfg.cwd at the driver default and hand
    # the remote cwd to the workspace below. Local worker: grant cwd is driver-local.
    remote_host = getattr(args, "remote_host", None)
    remote_ctl = getattr(args, "remote_ctl", None)
    if grant.get("cwd") and not remote_host:
        cfg.cwd = Path(grant["cwd"]).expanduser()
    if grant.get("max_iterations"):
        try:
            cfg.max_iterations = int(grant["max_iterations"])
        except ValueError:
            pass
    if not cfg.cwd.is_dir():
        return _fail(f"cwd is not a directory: {cfg.cwd}")

    try:
        _register_dir_providers(cfg)
        agent = Agent(cfg)
    except Exception as e:
        return _fail(f"worker failed to initialise (Agent build): {e}")
    prov = _maybe_autostart_provider(agent, cfg)
    if prov is None or agent.client is None:
        why = getattr(agent, "_provider_fail_reason", "") or (
            "worker needs the driver's activated provider + auth")
        return _fail(f"no provider/model available on this box: {why}")

    # Attach to the parent's remote (its brain runs here on the driver, but its TOOLS
    # run where the PARENT's do - same box, same codebase). Reuse the parent's live ssh
    # master via the passed ControlPath (BatchMode, no re-auth); a detached worker can't
    # answer a prompt, so a dead master is a hard fail (the parent verified it live
    # before spawning). Local parent -> no host passed -> worker stays local.
    if remote_host and remote_ctl:
        rcwd = grant.get("cwd") or "."
        try:
            ws = RemoteWorkspace(remote_host, rcwd,
                                 lean_tool_paths=agent.lean_tools.enabled_paths(),
                                 ctl=remote_ctl).attach()
        except (ConnectionError, OSError) as e:
            return _fail(f"cannot attach to parent's remote {remote_host}: {e}")
        agent.set_remote(ws)
        print(dim(f"agent-run: tools attached to remote {remote_host} (cwd {ws.remote_cwd or rcwd})"))
    # Grant's provider: if the dispatch tool pinned a backend (a `model` on a
    # DIFFERENT enabled provider than the driver's), activate it first so the model
    # switch below lands on the right client. Only switch if it's enabled + available;
    # else keep the activated default and note it - a worker never dies on a bad grant.
    want_provider = grant.get("provider")
    if want_provider and want_provider != cfg.provider:
        if want_provider in cfg.providers_enabled and want_provider in _providers:
            try:
                agent.activate_provider(want_provider)
            except Exception as e:
                print(yellow(f"agent-run: provider '{want_provider}' not activated ({e}); "
                             f"using '{cfg.provider}'"))
        else:
            print(yellow(f"agent-run: provider '{want_provider}' not enabled here; "
                         f"using '{cfg.provider}'"))

    # Grant's model: applied AFTER activation (active_model reads provider_settings,
    # not cfg.model). Only switch if it's actually available on this box; else keep
    # the activated default and note it - a worker must never die on a bad model name.
    want_model = grant.get("model")
    if want_model and want_model != cfg.active_model():
        try:
            spec = agent.active_provider()
            avail = list(spec["list_models"]() or []) if spec else []
            if want_model in avail:
                cfg.set_active_model(want_model)
                agent.use_client(spec["make_client"](cfg))
                agent.messages[0] = {"role": "system", "content": agent._system()}
            else:
                print(yellow(f"agent-run: model '{want_model}' not available here; "
                             f"using '{cfg.active_model()}'"))
        except Exception as e:
            print(yellow(f"agent-run: could not switch to model '{want_model}' ({e}); "
                         f"using '{cfg.active_model()}'"))

    # Preamble: tell the worker it's headless + how to return its answer.
    preamble = (
        "You are a headless worker agent. You have ONE task (below). Do it using your "
        "tools, then STOP. When finished, write your final answer as your last message, "
        f"wrapped between two {RESULT_MARK} markers, e.g.\n{RESULT_MARK}\n<your result>\n"
        f"{RESULT_MARK}\nNothing after the closing marker. There is no user to ask - if "
        "you get stuck, put what you found + what blocked you inside the result block. "
        "Your leash is fixed; do not ask to raise it.\n"
        "IMPORTANT: do NOT put a tool call and your " + RESULT_MARK + " block in the same "
        "message - a message that also calls a tool is NOT your final answer, and the "
        f"{RESULT_MARK} would make the tool call be ignored (your action would not run). "
        "Finish every tool call FIRST, wait for its result, confirm the change (re-read if "
        f"you edited), and only THEN write the {RESULT_MARK} block on its own.\n"
        "HOW YOUR RUN ENDS: a message from you that calls NO tool ENDS your run immediately. "
        "So never send a bare transition like 'now I'll compile the findings' or 'let me write "
        "that up' on its own - that message has no tool call, so your run stops right there and "
        f"whatever you meant to write next is lost. Every message must either (a) call a tool to "
        f"keep working, or (b) BE your final answer in the {RESULT_MARK} block. There is no "
        "in-between 'thinking out loud' message.\n"
        "MID-TASK MESSAGES: the operator or the agent that dispatched you may send you a "
        "message while you work (it arrives as a new turn prefixed [operator inject] or "
        "[parent-agent inject]). Treat it as a correction/steer that OUTRANKS your original "
        "task where they conflict (e.g. a changed constraint, a stop-and-redirect). When one "
        "arrives, briefly acknowledge it - aim for about one sentence confirming you got it "
        "and what you'll change - then carry on. (Keep the ack short, but do not truncate a "
        "thought to hit a length.)\n"
        "FIRST, before your first tool call, state your understanding of the task in about one "
        "sentence (what you're going to do). This stated intent lets whoever dispatched you "
        "catch a misread early and stop or redirect you. Keep it short, but don't truncate a "
        "thought to hit a length.\n\nTASK:\n"
        + brief)

    # Wire the inject poller: at each between-iteration boundary the worker drains any
    # pending inject sidecar (written by dispatch_worker action='inject' / '/worker inject')
    # and appends it as a user turn, so an operator/parent mid-task message lands on the
    # worker's next think without interrupting a running tool call. Sidecar lives beside the
    # brief file (driver-side, even for a remote worker). Consumption is logged to
    # '<brief>.injects.log' so status can confirm delivery. Best-effort; never raises.
    _inject_path = Path(str(brief_file) + ".inject")
    _inject_log = Path(str(brief_file) + ".injects.log")
    _progress_path = Path(str(brief_file) + ".progress")
    _wstart = time.time()
    _witer = {"n": 0}
    _intent_cache = {"v": ""}   # the first assistant prose line; set once, never changes

    def _drain_injects():
        try:
            if not _inject_path.exists():
                return
            raw = _inject_path.read_text()
            _inject_path.unlink()
        except OSError:
            return
        for block in raw.split("\0"):
            msg = block.strip()
            if not msg:
                continue
            agent.messages.append({"role": "user", "content": msg})
            try:
                with _inject_log.open("a") as f:
                    f.write(f"{int(time.time())} consumed: {msg[:120]}\n")
            except OSError:
                pass

    def _write_progress():
        # Deterministic heartbeat (facts the harness already has - zero tokens, zero
        # model cooperation): iteration N, elapsed, the last tool it ran, and its stated
        # INTENT (the one-line understanding the worker writes alongside its first tool
        # call - see preamble). Overwritten each iteration; surfaced via /worker status
        # so an operator/parent can see where it is and catch a total misread early.
        try:
            intent = _intent_cache["v"]
            if not intent:                          # scan only until first found, then cache
                for m in agent.messages:
                    if (m.get("role") == "assistant" and isinstance(m.get("content"), str)
                            and m["content"].strip()):
                        intent = " ".join(m["content"].split())[:200]
                        _intent_cache["v"] = intent
                        break
            last_tool = ""
            for m in reversed(agent.messages):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    fn = (m["tool_calls"][0].get("function") or {})
                    a = fn.get("arguments")
                    astr = ""
                    if isinstance(a, dict):
                        astr = ", ".join(f"{k}={str(v)[:30]}" for k, v in list(a.items())[:2])
                    last_tool = f"{fn.get('name', '?')}({astr})"
                    break
            el = int(time.time() - _wstart)
            elapsed = f"{el}s" if el < 60 else f"{el // 60}m{el % 60:02d}s"
            capstr = f"/{cfg.max_iterations}" if getattr(cfg, "max_iterations", 0) else ""
            lines = [f"iter {_witer['n']}{capstr} - {elapsed}"]
            if last_tool:
                lines.append(f"last: {last_tool}")
            if intent:
                lines.append(f"intent: {intent}")
            _progress_path.write_text("\n".join(lines) + "\n")
        except Exception:
            pass

    def _pre_iter():
        _witer["n"] += 1
        _drain_injects()
        _write_progress()
    agent._pre_iter_hook = _pre_iter

    try:
        agent.run_turn(preamble)
    except Exception as e:
        return _fail(f"worker run failed: {e}")

    # Guard against the "act + finish in one message" trap: a worker that emits a tool
    # call AND a ===RESULT=== in the same message ends the loop without running the
    # call (the L2a text parser sees the RESULT prose and discards the call as a final
    # answer), yet the RESULT claims success - the action silently never happened. We
    # strip the RESULT block, re-parse the remainder (dodging the prose-slack guard);
    # any tool call found there was NEVER executed, so nudge the worker to actually
    # finish it. At most 2 corrective turns, then give up and report honestly.
    def _final_asst_content():
        for m in reversed(agent.messages):
            if (m.get("role") == "assistant" and isinstance(m.get("content"), str)
                    and m["content"].strip()):
                return m["content"]
        return ""

    known = {(t.get("function") or t).get("name") for t in (agent.tool_defs or [])}
    known.discard(None)
    for _ in range(2):
        content = _final_asst_content()
        # Fire on ANY RESULT marker, not just a complete pair: a worker that emits a
        # single UNCLOSED ===RESULT=== (opening marker + a bundled call, no close) would
        # slip a pair-only check AND then the final harvest's `final.strip()` fallback
        # returns the message verbatim - i.e. the unexecuted call reported as success.
        if RESULT_MARK not in content:
            break                                     # no RESULT yet -> loop ended normally
        # Strip a complete RESULT pair if present, else everything from the last lone
        # marker onward, so the remainder is just the pre-RESULT body we re-parse.
        without_result = re.sub(
            re.escape(RESULT_MARK) + r".*?" + re.escape(RESULT_MARK), "",
            content, flags=re.S)
        if RESULT_MARK in without_result:             # a lone/odd marker survived the pair strip
            without_result = without_result[:without_result.rfind(RESULT_MARK)]
        orphaned, _ = parse_text_tool_calls(without_result, known_names=known or None)
        if not orphaned:
            break                                     # RESULT with no bundled call -> clean
        names = ", ".join((c.get("function") or {}).get("name", "?") for c in orphaned)
        try:
            agent.run_turn(
                f"Your last message put a tool call ({names}) in the SAME message as your "
                f"{RESULT_MARK} block, so the tool call did NOT run - your action was NOT "
                f"applied. Make that tool call now, on its own (no {RESULT_MARK} in the same "
                f"message). Wait for the result, confirm the change, and only then write the "
                f"{RESULT_MARK} block alone.")
        except Exception as e:
            return _fail(f"worker corrective turn failed: {e}")

    # Missing-RESULT rescue: the loop ends the moment the model emits an assistant
    # message with no tool call - but a worker sometimes stops on a bare "let me compile
    # the findings" preamble WITHOUT ever writing the RESULT block, so the harvest below
    # would capture that stub as the whole answer. If the final text carries no marker,
    # nudge it to actually produce the RESULT (at most 2 turns), then re-harvest.
    for _ in range(2):
        if RESULT_MARK in _final_asst_content():
            break
        try:
            agent.run_turn(
                f"You stopped without writing your {RESULT_MARK} block, so no answer was "
                f"captured. Write your COMPLETE final answer now, wrapped between two "
                f"{RESULT_MARK} markers, as your entire message (no tool call in it). Do not "
                f"summarize that you will - write the actual content.")
        except Exception as e:
            return _fail(f"worker result-rescue turn failed: {e}")

    # Harvest: the last assistant text -> its RESULT block, or the whole message.
    final = ""
    for m in reversed(agent.messages):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
            final = m["content"]
            break
    block = _extract_marked(final, RESULT_MARK) or final.strip() or "(worker produced no output)"
    try:
        Path(resultf).write_text(f"{RESULT_MARK}\n{block}\n{RESULT_MARK}\n")
    except OSError as e:
        return _fail(f"cannot write result file: {e}")
    print(dim(f"agent-run: done -> {resultf}"))
    return 0


def main():
    ap = argparse.ArgumentParser(
        prog="lean-coder",
        description="Lean terminal coding agent over Ollama native tool calling.")
    ap.add_argument("--host", help="Ollama base URL (env OLLAMA_HOST)")
    ap.add_argument("--model", help="model name (env LEANCODER_MODEL)")
    ap.add_argument("--provider", help="active backend provider (must be enabled)")
    ap.add_argument("--num-ctx", type=int, dest="num_ctx",
                    help="context window (default 32768)")
    ap.add_argument("--keep-alive", dest="keep_alive",
                    help="ollama model resident time, e.g. 10m, 30m, -1 (default 10m)")
    ap.add_argument("--auto-evict", dest="auto_evict", action="store_true", default=None,
                    help="continuously stub acted-on tool results to cut tokens sent (see docs)")
    ap.add_argument("--window-messages", type=int, dest="window_messages",
                    help="send only the last N messages per request (default 30; 0 = unbounded)")
    ap.add_argument("--window-tokens", dest="window_tokens",
                    help="cap each send by tokens: 'auto' (=ctx-reserve), an int (hard cap), or 0 (off)")
    ap.add_argument("--cwd", help="project directory (default: current dir)")
    ap.add_argument("-r", "--resume", metavar="NAME",
                    help="resume a saved session by name (see /load)")
    ap.add_argument("--approval", choices=APPROVAL_MODES,
                    help="approval mode: ask (default) | session | auto")
    ap.add_argument("--auto", action="store_true",
                    help="auto-approve writes and commands (same as --approval auto)")
    ap.add_argument("--yolo", action="store_true", help=argparse.SUPPRESS)  # alias for --auto
    ap.add_argument("--no-composer", dest="no_composer", action="store_true",
                    help="disable the pinned bottom input line (use the classic prompt)")
    ap.add_argument("--no-autosave", dest="no_autosave", action="store_true",
                    help="amnesic: don't autosave the session or auto-load the last one")
    ap.add_argument("--incognito", action="store_true",
                    help="incognito: no session saved locally + the model is told "
                         "(does not stop the provider from logging)")
    ap.add_argument("--leash", choices=LEASH_LEVELS,
                    help="capability ceiling: chat (no tools) | r | rw | rwe (default)")
    ap.add_argument("--chat-only", dest="chat_only", action="store_true",
                    help="alias for --leash chat (send no tools, pure conversation)")
    ap.add_argument("--think", dest="think", action="store_const", const=True,
                    default=None, help="send think=true (extended reasoning)")
    ap.add_argument("--no-think", dest="think", action="store_const", const=False,
                    help="send think=false")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float, dest="top_p")
    ap.add_argument("--top-k", type=int, dest="top_k")
    ap.add_argument("--repeat-penalty", type=float, dest="repeat_penalty")
    ap.add_argument("--bg-run", dest="bg_run",
                    help="(internal) detached backgrounded-command runner: JSON cfg; "
                         "the cross-platform twin of the POSIX shell bg wrapper")
    ap.add_argument("--wipe-watch", dest="wipe_watch",
                    help="(internal) detached self-wipe watchdog for a pushed remote "
                         "executor: JSON cfg (dir/ttl/chk); cross-platform")
    ap.add_argument("--tool-exec", action="store_true", dest="tool_exec",
                    help="run as a hermetic tool executor (JSON over stdio); internal")
    ap.add_argument("--lean-tools", dest="lean_tools_exec",
                    help="(executor) directory of lean-tools to load; internal")
    ap.add_argument("--agent-run", action="store_true", dest="agent_run",
                    help="run headless as a worker agent: execute one brief, write the "
                         "result, exit; internal (spawned by the dispatch_worker tool)")
    ap.add_argument("--brief-file", dest="brief_file",
                    help="(agent-run) path to the brief file to execute; internal")
    ap.add_argument("--result-file", dest="result_file",
                    help="(agent-run) where to write the ===RESULT===; "
                         "default <brief-file>.result; internal")
    ap.add_argument("--remote-host", dest="remote_host",
                    help="(agent-run) attach the worker's tools to this remote host, "
                         "reusing the parent's ssh master; internal")
    ap.add_argument("--remote-ctl", dest="remote_ctl",
                    help="(agent-run) ControlPath of the parent's ssh master to reuse; "
                         "internal")
    ap.add_argument("--version", action="store_true",
                    help="print this build's source hash and exit "
                         "(used by /connect to skip a redundant re-push)")
    args = ap.parse_args()

    if args.version:                   # version probe: print self-hash, do nothing else
        print(source_hash())           # /connect parses this exact line (hash only)
        return
    if args.bg_run:                    # detached bg-runner role: run one command, write sidecars, exit
        try:
            sys.exit(run_bg_child(json.loads(args.bg_run)) or 0)
        except Exception:
            sys.exit(1)
    if args.wipe_watch:                # detached self-wipe watchdog for a pushed executor
        try:
            run_wipe_watchdog(json.loads(args.wipe_watch))
        except Exception:
            pass
        return
    if args.tool_exec:                 # hermetic executor role: no config, no model
        run_tool_executor(args.cwd or ".", args.lean_tools_exec)
        return
    if args.agent_run:                 # headless worker role: run one brief, write result, exit
        sys.exit(run_agent_brief(args))
        return

    # Reattach stdin to the controlling terminal when a wrapper/launcher left it
    # piped (stdout IS a tty but stdin is not) - otherwise the composer's gate
    # (and any interactive prompt) correctly refuses, and input only flushes at
    # end-of-turn. Reopening /dev/tty restores the pinned input for these
    # wrapper-launched sessions. Best-effort and quiet: if /dev/tty isn't there
    # (no controlling terminal - cron, true pipe), we leave stdin as-is.
    if sys.stdout.isatty() and not sys.stdin.isatty():
        try:
            _tty_in = open("/dev/tty")
            if _tty_in.isatty():
                sys.stdin = _tty_in
        except OSError:
            pass

    cfg = load_config(args)
    if not cfg.cwd.is_dir():
        print(red(f"error: --cwd is not a directory: {cfg.cwd}"))
        sys.exit(1)
    _register_dir_providers(cfg)          # register enabled providers/ plugins (bundled ollama + user)
    _bg_reap_orphans()                    # kill background tasks left by a crashed session
    _worker_dir_sweep()                   # nuke orphaned worker sidecars (reboot/SIGKILL residue)
    atexit.register(_bg_kill_session)     # peg this session's bg tasks to it: kill on exit
    atexit.register(_close_worker_masters)  # tear down masters opened just to carry workers
    try:
        repl(cfg, resume=args.resume)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
