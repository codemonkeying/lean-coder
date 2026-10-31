"""Test isolation: import this FIRST, before lean_coder, in every test entry point.

Points HOME (and the XDG dirs) at a fresh /tmp folder for the whole run, so every path
lean-coder builds from the home dir - config.toml, sessions, keys, history, caches, the
debug log - lands there instead of the developer's real ~/.config/leancoder. Child
processes the tests spawn inherit it. The folder is removed when the run exits.
A test can therefore never read or clobber the real config, even if it forgets to
redirect a path itself.
"""
import atexit
import os
import shutil
import tempfile

REAL_HOME = os.path.expanduser("~")
TEST_HOME = tempfile.mkdtemp(prefix="lc-test-")

for _k in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
    os.environ.pop(_k, None)
os.environ["HOME"] = TEST_HOME
# git refuses to commit without an identity; give tests one that isn't the real user's.
os.environ.setdefault("GIT_AUTHOR_NAME", "lc-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "lc-test@example.invalid")
os.environ.setdefault("GIT_COMMITTER_NAME", "lc-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "lc-test@example.invalid")

atexit.register(shutil.rmtree, TEST_HOME, True)
