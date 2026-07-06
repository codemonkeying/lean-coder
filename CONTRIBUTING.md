# Contributing

Thanks for taking a look. lean-coder has a deliberately small surface and a firm
design bias - keeping both intact is the main thing a change is judged on.

## The design bar

- **Zero dependencies, one file.** The core is a single stdlib-only `python3` file.
  A change that adds a third-party import, or splits the core into multiple modules,
  is a no unless there's an exceptional reason.
- **Context is the scarce resource.** The baseline overhead (system prompt + always-on
  tool schemas, before any opt-in lean-tools) is budgeted, and there's a test that
  enforces it. New always-on
  surface has to earn its tokens; most capabilities belong in an **opt-in lean-tool**,
  not the core.
- **Small and few beats big and many.** The tools are terse so a small local model
  can drive them reliably. Keep descriptions to one line.
- **New capability = a plugin, not a core edit.** Adding a tool or a backend should
  be a drop-in `.py` file. See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md),
  [LEAN_TOOLS.md](LEAN_TOOLS.md), and [PROVIDER_API.md](PROVIDER_API.md).

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

A change is ready when all three are green.

## Versioning

Releases follow [SemVer](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **PATCH** - bug fixes, no behaviour change (safe to auto-update into).
- **MINOR** - new, backward-compatible features.
- **MAJOR** - breaking changes.

A pre-release suffix (e.g. `1.2.0-beta.1`) marks the **beta** track and has lower
precedence than the same core release. On a published change, bump both the
`__version__` string in `lean_coder.py` and the top-level `VERSION` file (they must
match - a test enforces it); the `/update` lean-tool probes `VERSION` to decide
whether to pull.

## Pull requests

- Keep the diff focused; prefer small, reviewable changes.
- Update the relevant doc when you change behaviour (README, LEAN_TOOLS,
  PROVIDER_API, BUILD_GUIDE).
- Describe what you changed and why, and note that the three gates pass.

## License

By contributing you agree your contributions are licensed under the repo's
[MIT License](LICENSE).
