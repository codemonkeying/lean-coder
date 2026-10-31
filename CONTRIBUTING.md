# Contributing

- The core is one stdlib-only file; new capability goes in a lean-tool or provider
  ([doc/LEAN_TOOLS.md](doc/LEAN_TOOLS.md), [doc/PROVIDER_API.md](doc/PROVIDER_API.md)).
- The context overhead limit in the README is enforced by a test.
- Keep tool descriptions short.

## The gates

Run all three from the repo root before opening a PR. Each exits non-zero on
failure, so they're safe to chain:

```bash
python3 tests/_smoketest.py     # offline unit suite (includes the fixed-overhead budget check)
python3 tests/_mocktest.py      # scripted end-to-end suite
bash tests/_sweep.sh            # hygiene lint (stray unicode, likely secrets/PII, etc.)
```

`tests/_sweep.sh` scans tracked `*.py` / `*.md` / `*.sh`. A line that legitimately needs a
flagged token can be exempted by ending it with a `# sweep-ok` marker - use sparingly.

A change is ready when all three pass.

### Git hooks

The repo ships git hooks in `.githooks/`. Enable them once per clone:

```bash
git config core.hooksPath .githooks
```

`pre-push` then runs all three gates (plus a `VERSION` / `__version__` match check)
and refuses the push if any fail; it takes about half a minute. `pre-commit` blocks
the local-only trees. The hook exists so the gates run on every push without relying
on memory. `git push --no-verify` bypasses it.

## Versioning

SemVer. Bump `__version__` in `lean_coder.py` and `VERSION` together (a test checks
they match). A pre-release suffix (e.g. `1.2.0-beta.1`) marks the beta track.

## Pull requests

Keep diffs focused, update the relevant doc, and confirm the gates pass.

## License

lean-coder is all rights reserved; see the License section of the [README](README.md#license).
By contributing you assign or grant codemonkeying, without restriction or payment, all
rights in your contribution, including the right to use, modify, relicense and
distribute it in any way.
