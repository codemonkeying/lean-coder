#!/usr/bin/env bash
# Debug-bucket run: the smoke + mock suites with LEANCODER_DEBUG=1, every swallowed
# exception logged to a throwaway temp file, then a per-site summary. Informational:
# exits non-zero only if a suite fails. Pass a path to keep the raw log there.
set -u
cd "$(dirname "$0")/.."
log=$(mktemp "${TMPDIR:-/tmp}/lc-debug.XXXXXX")
trap 'rm -f "$log"' EXIT
export LEANCODER_DEBUG=1 LEANCODER_DEBUG_LOG="$log"
rc=0
python3 tests/_smoketest.py >/dev/null 2>&1 || { echo "smoketest FAILED under debug"; rc=1; }
python3 tests/_mocktest.py  >/dev/null 2>&1 || { echo "mocktest FAILED under debug"; rc=1; }
echo "debug bucket: $(wc -l < "$log") swallowed exception(s), by site:"
sed -E 's/^.* swallowed at ([^ ]+ [^:]+): ([A-Za-z_.]+).*/\1 \2/' "$log" | sort | uniq -c | sort -rn
[ -n "${1:-}" ] && cp "$log" "$1"
exit $rc
