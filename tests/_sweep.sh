#!/usr/bin/env bash
# Hygiene lint. One command, run before a commit.
# Flags things that shouldn't ship in source:
#   - dash-like unicode (em/en/figure/bar/box) and stray unicode arrows
#     (keep source ASCII so diffs and terminals stay clean)
#   - likely secrets / PII: non-example IPv4, real-looking emails, private keys,
#     MAC addresses
# Exits 0 when clean, 1 when anything is found. Add patterns here, not in your head.
set -u
cd "$(dirname "$0")/.." || exit 2   # repo root (this script lives in tests/)

# Lines marked "sweep-ok" are excluded from all checks - for a line that
# legitimately needs a flagged token (a load-bearing TUI glyph, a protocol
# constant, an example address). Works from a code comment (`# sweep-ok`) or a
# markdown/HTML one (`<!-- sweep-ok -->`).
nosweep() { grep -Fv 'sweep-ok'; }

# Files to scan: git-tracked source/docs (includes anything staged for the commit).
# Falls back to a full working-tree scan outside a git repo. The sweep excludes
# itself (it holds the patterns).
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mapfile -t FILES < <(git ls-files -- '*.py' '*.md' '*.sh' | grep -vx 'tests/_sweep.sh' | sort)
else
  mapfile -t FILES < <(find . \
    \( -name '*.py' -o -name '*.md' -o -name '*.sh' \) \
    -not -name '_sweep.sh' \
    -not -path '*/tests/_smoketest_local.py' \
    -not -path '*/__pycache__/*' | sort)
fi

fails=0
report() {  # name, hits  (hits passed as an arg, not piped: keeps the counter
            # in this shell - a piped last stage would run in a subshell)
  local name="$1" hits="$2"
  if [ -n "$hits" ]; then
    echo "FAIL: $name"
    printf '%s\n' "$hits" | sed 's/^/    /'
    fails=$((fails + 1))
  else
    echo "ok:   $name"
  fi
}

# --- ASCII hygiene ---------------------------------------------------------
# No -o: keep the whole line so a `sweep-ok` marker on it survives to nosweep
# (grep -o would emit only the glyph, and the marker filter could never see it).
report "dash-like unicode (use ASCII -)" \
  "$(grep -rnP "[\x{2012}-\x{2015}\x{2212}\x{2500}]" "${FILES[@]}" | nosweep)"

report "unicode arrows (use ASCII -> )" \
  "$(grep -rnP "[\x{2190}-\x{21FF}\x{2794}\x{27A1}]" "${FILES[@]}" | nosweep)"

# --- likely secrets / PII --------------------------------------------------
# Any dotted-quad that is NOT localhost, 0.0.0.0, or an RFC 5737 doc range
# (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24). Use the doc ranges in examples.
report "non-example IPv4 (use 192.0.2.x / 198.51.100.x / 203.0.113.x)" \
  "$(grep -rnoE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' "${FILES[@]}" \
     | grep -vE ':(127\.0\.0\.1|0\.0\.0\.0|192\.0\.2\.[0-9]{1,3}|198\.51\.100\.[0-9]{1,3}|203\.0\.113\.[0-9]{1,3})$' \
     | nosweep)"

# Real-looking email addresses (allow example.com and *.example placeholders).
report "email addresses (use user@host or *.example)" \
  "$(grep -rnoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' "${FILES[@]}" \
     | grep -viE '@example\.|\.example(:|$)|@host' \
     | nosweep)"

report "private keys / public-key material" \
  "$(grep -rniE 'BEGIN [A-Z0-9 ]*PRIVATE KEY|ssh-rsa AAAA|ssh-ed25519 AAAA' "${FILES[@]}" | nosweep)"

report "MAC addresses" \
  "$(grep -rniE '\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b' "${FILES[@]}" | nosweep)"

echo
if [ "$fails" -eq 0 ]; then
  echo "SWEEP CLEAN"
  exit 0
fi
echo "SWEEP FOUND $fails categor(y/ies) of issues"
exit 1
