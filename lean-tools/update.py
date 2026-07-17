"""/update - self-update lean_coder to the latest published build.

A driver-side lean-tool (setup() hook, like /provision): it acts on THIS box,
adds no model tool. Run `/update` and it:

  1. reads the published VERSION (a tiny text file in the repo) and compares it to
     this build's lean_coder.__version__ - a cheap probe, so an up-to-date install
     downloads nothing;
  2. only when the published version is newer (or /update force), downloads the repo
     TARBALL for the track's branch;
  3. byte-compiles every .py in the overlay set to make sure none is truncated /
     corrupt (validates ALL before writing ANY - a partial swap could break startup);
  4. backs up your current lean_coder.py to lean_coder.py.bak, then OVERLAYS the
     bundled files in place: lean_coder.py + VERSION + shipped docs + providers/*.py|md
     + lean-tools/*.py|md. Overlay-only: it never deletes, so your own drop-in
     providers / lean-tools (and any file we removed upstream) are left untouched;
  5. tells you to relaunch to run the new build.

`/update check` probes and reports without changing anything. `/update force`
re-pulls even when the version looks current (e.g. to repair a local edit).

If `auto_update` is on in your config (off by default), the check+update runs
once automatically at launch, non-interactively, whenever this tool is enabled.

Whole-tree overlay (not just lean_coder.py) so a single feature that spans core +
a bundled provider can't leave a version-mismatched install. Needs network access
and curl (or a working urllib). Nothing is changed if any step fails.
"""
import io
import os
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = "codemonkeying/lean-coder"
# The branch each update track fetches from: stable = main, beta = a pre-release branch.
TRACK_BRANCH = {"stable": "main", "beta": "beta"}

# The overlay set - what a runtime install actually consists of. Top-level files
# refreshed if present in the tarball; the two bundled plugin dirs get their *.py and
# *.md refreshed. OVERLAY-ONLY: we only ever overwrite a path the tarball ships; we
# never delete, so user drop-ins (and upstream-removed files) survive.
_ROOT_FILES = ("lean_coder.py", "VERSION", "README.md", "LEAN_TOOLS.md", "MCP.md",
               "PROVIDER_API.md", "CONTRIBUTING.md", "install.sh", "uninstall.sh")
_OVERLAY_DIRS = ("providers", "lean-tools", "demos")
# .gif ships the README demo asset (under demos/); .py/.md cover the plugin dirs.
_DIR_SUFFIXES = (".py", ".md", ".gif")


# Env flag set on the self-heal re-exec so the freshly-overlaid updater doesn't
# recurse: a second overlay pass finds nothing changed and stops, but we belt-and-
# brace it so a bug can never spin.
_REEXEC_ENV = "LEANCODER_UPDATE_REEXEC"


def _reexec_fresh_updater(dest_root: Path, track: str) -> bool:
    """After an overlay that replaced update.py ITSELF, run the freshly-installed
    updater once in a subprocess so the job finishes under the NEW overlay logic -
    not the stale in-memory code that started it. This is what makes a jump ACROSS
    an updater change (e.g. the pre-0.2.6 'only lean_coder.py' bug) self-heal in a
    single /update, instead of leaving a half-applied tree that needs a second run.
    Returns True if the re-exec ran and reported it applied more files."""
    fresh = dest_root / "lean-tools" / "update.py"
    if os.environ.get(_REEXEC_ENV) or not fresh.is_file():
        return False
    # Load the fresh module by path, rebuild a minimal cfg (only .update_track is
    # read), pass identity color fns, no prompt. Any files the new logic knows about
    # but the old one skipped get applied now; then it's a byte-identical no-op.
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('_lc_fresh_update', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "class C: pass\n"
        "cfg = C(); cfg.update_track = sys.argv[2]\n"
        "idn = lambda s: s\n"
        "ok = m._apply(cfg, __import__('pathlib').Path(sys.argv[3]), idn, idn, idn, idn, ask=None)\n"
        "sys.exit(0 if ok else 3)\n"
    )
    env = dict(os.environ, **{_REEXEC_ENV: "1"})
    try:
        r = subprocess.run([sys.executable, "-c", code, str(fresh), track, str(dest_root)],
                           env=env, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def _raw_base(track):
    branch = TRACK_BRANCH.get(track, "main")
    return f"https://raw.githubusercontent.com/{REPO}/{branch}"


def _tarball_url(track):
    branch = TRACK_BRANCH.get(track, "main")
    return f"https://github.com/{REPO}/archive/refs/heads/{branch}.tar.gz"


def _get_text(url):
    """Fetch a small text resource -> (text, None) or (None, error). Prefers curl
    (honours the same proxies/certs as the installer), falls back to urllib."""
    if shutil.which("curl"):
        r = subprocess.run(["curl", "-fsSL", url], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None, (r.stderr.strip() or f"curl exited {r.returncode}")
        return r.stdout, None
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace"), None
    except Exception as e:
        return None, str(e)


def _download_bytes(url, timeout=120):
    """Fetch url -> (bytes, None) or (None, error). curl first, urllib fallback."""
    if shutil.which("curl"):
        r = subprocess.run(["curl", "-fsSL", url], capture_output=True, timeout=timeout)
        if r.returncode != 0:
            return None, (r.stderr.decode("utf-8", "replace").strip() or f"curl exited {r.returncode}")
        return r.stdout, None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read(), None
    except Exception as e:
        return None, str(e)


def _overlay_rel_paths(src_root: Path):
    """The relative paths in an extracted source tree that make up the overlay set:
    the top-level shipped files plus *.py/*.md under the bundled plugin dirs. Only
    paths that actually exist in `src_root` are returned. Pure (fs read of src)."""
    rels = []
    for name in _ROOT_FILES:
        if (src_root / name).is_file():
            rels.append(name)
    for d in _OVERLAY_DIRS:
        sub = src_root / d
        if sub.is_dir():
            for f in sorted(sub.iterdir()):
                if f.is_file() and f.suffix in _DIR_SUFFIXES:
                    rels.append(f"{d}/{f.name}")
    return rels


def _track(cfg):
    return (getattr(cfg, "update_track", "stable") or "stable").strip().lower()


def _probe(cfg, local_ver, version_tuple):
    """(published_str, is_newer, err). is_newer is None on error. Module-level +
    dependency-injected (local_ver, version_tuple from core) so it's unit-testable."""
    text, err = _get_text(f"{_raw_base(_track(cfg))}/VERSION")
    if err:
        return None, None, err
    published = (text or "").strip().splitlines()[0].strip() if text.strip() else ""
    if not published:
        return None, None, "published VERSION is empty"
    return published, version_tuple(published) > version_tuple(local_ver), None


def _apply(cfg, dest_root, dim, green, yellow, red, ask=None):
    """Download the track tarball, validate, and OVERLAY the bundled files onto
    dest_root. `ask` gates the write (None = no prompt, for auto-update). Returns
    True on a successful update. Nothing is written unless every .py compiles and at
    least one file differs from what's installed. Module-level so it's unit-testable
    (the tests stub _download_bytes)."""
    url = _tarball_url(_track(cfg))
    blob, err = _download_bytes(url)
    if err:
        print(red(f"download failed: {err}")); return False
    if not blob:
        print(red("download failed: empty archive")); return False
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                tar.extractall(tdp, filter="data")
        except (tarfile.TarError, OSError) as e:
            print(red(f"could not unpack archive: {e}")); return False
        # GitHub wraps everything in a single 'lean-coder-<branch>/' dir.
        roots = [p for p in tdp.iterdir() if p.is_dir()]
        src_root = roots[0] if len(roots) == 1 else tdp
        if not (src_root / "lean_coder.py").is_file():
            print(red("archive is missing lean_coder.py; not applying.")); return False

        rels = _overlay_rel_paths(src_root)
        # Validate ALL python before writing ANY (never a partial, breakage-causing swap).
        for rel in rels:
            if rel.endswith(".py"):
                try:
                    py_compile.compile(str(src_root / rel), doraise=True)
                except py_compile.PyCompileError as e:
                    print(red(f"downloaded {rel} is not valid Python; not applying.\n{e}"))
                    return False
        # Which files actually differ from what's installed?
        changed = []
        for rel in rels:
            dst = dest_root / rel
            new = (src_root / rel).read_bytes()
            if not dst.exists() or dst.read_bytes() != new:
                changed.append(rel)
        if not changed:
            print(green("already up to date (all bundled files byte-identical).")); return False
        n = len(changed)
        core = " (incl. lean_coder.py)" if "lean_coder.py" in changed else ""
        if ask is not None and not ask(f"overlay {n} updated file(s){core} into {dest_root}?"):
            print(dim("cancelled.")); return False
        # Back up the core file for a hand rollback, then overlay every changed file.
        try:
            core_dst = dest_root / "lean_coder.py"
            if core_dst.exists():
                shutil.copy2(core_dst, core_dst.with_suffix(core_dst.suffix + ".bak"))
            for rel in changed:
                dst = dest_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_root / rel, dst)
                if rel.endswith(".py"):
                    dst.chmod(dst.stat().st_mode | 0o111)
        except OSError as e:
            print(red(f"could not write update: {e}")); return False
    print(green(f"updated {n} file(s) to the latest build "
                f"(lean_coder.py backup at lean_coder.py.bak)."))
    for rel in changed:
        print(dim(f"  + {rel}"))
    # Self-heal: if THIS overlay replaced the updater itself, the code that just ran
    # is now stale - re-run the freshly-installed updater once so the whole job
    # finishes under the new logic (a jump across an updater change lands complete in
    # ONE /update, never "great, now run it again"). Guarded against recursion.
    if "lean-tools/update.py" in changed and not os.environ.get(_REEXEC_ENV):
        print(dim("updater changed - re-running the new updater to finish..."))
        _reexec_fresh_updater(dest_root, _track(cfg))
    print(yellow("relaunch lean_coder to run the new build."))
    return True


def setup(lc, cfg):
    register_command = lc["register_command"]
    _vt = lc["version_tuple"]
    _local_ver = lc["__version__"]

    def _update(agent, cfg, arg):
        dim, green = lc["dim"], lc["green"]
        yellow, red = lc["yellow"], lc["red"]
        ask = lc["_ask"]
        mode = arg.strip().lower()
        dest_root = Path(lc["__file__"]).resolve().parent

        if mode not in ("", "check", "force"):
            print(yellow("usage: /update [check|force]")); return

        track = _track(cfg)
        print(dim(f"current version {_local_ver}; track '{track}'; "
                  f"checking {_raw_base(track)}/VERSION"))
        published, newer, err = _probe(cfg, _local_ver, _vt)
        if err and mode != "force":
            print(red(f"version check failed: {err}")); return
        if err:
            print(yellow(f"version check failed ({err}); force-pulling anyway."))
        else:
            print(dim(f"published version {published}"))

        if mode == "check":
            print(green("update available." if newer else "up to date.")); return
        if mode != "force" and not newer:
            print(green("up to date - nothing to do.")); return

        _apply(cfg, dest_root, dim, green, yellow, red, ask=ask)

    def _auto_update_on_launch():
        """Non-interactive launch check, gated by cfg.auto_update. Quiet on the
        common 'already current' / offline path so startup isn't noisy."""
        if not getattr(cfg, "auto_update", False):
            return
        dim, green = lc["dim"], lc["green"]
        yellow, red = lc["yellow"], lc["red"]
        published, newer, err = _probe(cfg, _local_ver, _vt)
        if err:
            print(dim(f"auto-update: version check skipped ({err})")); return
        if not newer:
            return
        print(yellow(f"auto-update: newer build {published} available (have {_local_ver}, "
                     f"track '{_track(cfg)}'); updating..."))
        _apply(cfg, Path(lc["__file__"]).resolve().parent, dim, green, yellow, red, ask=None)

    register_command("/update", _update,
                     "self-update lean_coder + bundled plugins to the latest build (check | force)")
    _auto_update_on_launch()
