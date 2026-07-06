#!/usr/bin/env bash
#
# uninstall.sh - remove lean-coder from this machine. By default it NUKES
# everything install.sh created: the symlink, the install dir, the config dir
# (config + model cache + saved sessions + the OAuth token in auth.json), the
# Ollama systemd bind drop-in, and the PATH line added to your shell rc.
#
# It does NOT uninstall Ollama itself or delete any models - that is a separate
# tool you may want to keep. Use --keep-config to preserve your settings/secrets,
# or --repo to also delete the source tree you are running this from.
#
# Usage:
#   ./uninstall.sh [options]
#     -y, --yes        don't prompt; just do it
#     -n, --dry-run    show what would be removed, change nothing
#         --keep-config  leave ~/.config/leancoder (settings, sessions, auth) intact
#         --repo         also delete the source repo this script lives in
#         --bin DIR    symlink dir on PATH      (default: ~/.local/bin)
#     -h, --help       this help
#
# Honors LEANCODER_DIR / LEANCODER_BIN, same as install.sh.

set -euo pipefail

# ---- defaults (match install.sh) -------------------------------------------
DIR="${LEANCODER_DIR:-$HOME/lean-coder}"
BIN="${LEANCODER_BIN:-$HOME/.local/bin}"
CONFIG_DIR="$HOME/.config/leancoder"
DROPIN="/etc/systemd/system/ollama.service.d/zz-leancoder-bind.conf"
ASSUME_YES=0
DRY=0
KEEP_CONFIG=0
NUKE_REPO=0
DEV_INSTALL=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- pretty output (match install.sh) --------------------------------------
if [ -t 1 ]; then
  B=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; CYN=$'\e[36m'; Z=$'\e[0m'
else
  B=""; DIM=""; RED=""; GRN=""; YEL=""; CYN=""; Z=""
fi
say()  { printf '%s\n' "${CYN}==>${Z} $*"; }
ok()   { printf '%s\n' "${GRN}  ok${Z} $*"; }
warn() { printf '%s\n' "${YEL}  ! ${Z} $*" >&2; }
die()  { printf '%s\n' "${RED}error:${Z} $*" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then printf '%s\n' "${DIM}  would: $*${Z}"; else eval "$@"; fi; }
usage() { sed -n '3,21p' "$0" | sed 's/^#\( \|$\)//'; }

# ---- args -------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift;;
    -n|--dry-run) DRY=1; shift;;
    --keep-config) KEEP_CONFIG=1; shift;;
    --repo) NUKE_REPO=1; shift;;
    --bin) BIN="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "unknown option: $1  (--help)";;
  esac
done

LINK="$BIN/lean_coder"

# Resolve the REAL install dir from the symlink when present, so a custom
# --dir/LEANCODER_DIR install is found without the user re-specifying it.
if [ -L "$LINK" ]; then
  target="$(readlink -f "$LINK" 2>/dev/null || true)"
  [ -n "$target" ] && DIR="$(dirname "$target")"
fi

# ---- build the kill list (only things that actually exist) ------------------
# NB: use explicit if-blocks, not `[ test ] && {...}` - under `set -e` a
# statement-level test that returns false would abort the whole script.
declare -a TARGETS LABELS
if [ -L "$LINK" ] || [ -e "$LINK" ]; then
  TARGETS+=("$LINK"); LABELS+=("symlink        $LINK")
fi
# A dev/in-place install symlinks straight at the source tree, so the resolved
# "install dir" can BE the repo. Never auto-delete the repo: that is gated behind
# --repo. Only treat DIR as a removable install dir when it's a separate copy.
if [ -d "$DIR" ] && [ "$DIR" != "$SCRIPT_DIR" ]; then
  TARGETS+=("$DIR"); LABELS+=("install dir    $DIR")
elif [ "$DIR" = "$SCRIPT_DIR" ]; then
  DEV_INSTALL=1
fi
if [ "$KEEP_CONFIG" = 0 ] && [ -d "$CONFIG_DIR" ]; then
  TARGETS+=("$CONFIG_DIR")
  LABELS+=("config+secrets $CONFIG_DIR  ${RED}(config.toml, sessions, auth.json)${Z}")
fi
HAVE_DROPIN=0
if [ -f "$DROPIN" ]; then
  HAVE_DROPIN=1; LABELS+=("ollama drop-in $DROPIN  ${DIM}(sudo; restarts ollama)${Z}")
fi
if [ "$NUKE_REPO" = 1 ]; then
  LABELS+=("source repo    $SCRIPT_DIR  ${RED}(the tree you are running from)${Z}")
fi

# PATH line in the shell rc (added only for the default ~/.local/bin)
RC=""
case "$(basename "${SHELL:-}")" in
  zsh) RC="$HOME/.zshrc";; bash) RC="$HOME/.bashrc";; *) RC="$HOME/.profile";;
esac
PATH_LINE_PRESENT=0
if [ -f "$RC" ] && grep -qsF "# added by lean-coder install.sh" "$RC"; then
  PATH_LINE_PRESENT=1; LABELS+=("PATH line       $RC  ${DIM}(the lean-coder export)${Z}")
fi

if [ ${#LABELS[@]} -eq 0 ]; then
  ok "nothing to remove - lean-coder isn't installed here (checked $LINK, $DIR, $CONFIG_DIR)."
  exit 0
fi

# ---- confirm ----------------------------------------------------------------
say "lean-coder uninstall - this will remove:"
for l in "${LABELS[@]}"; do printf '    %s\n' "$l"; done
[ "$KEEP_CONFIG" = 1 ] && warn "keeping config dir ($CONFIG_DIR) - settings, sessions and auth.json stay."
if [ "$DEV_INSTALL" = 1 ] && [ "$NUKE_REPO" = 0 ]; then
  warn "the command points at the source repo ($SCRIPT_DIR); it is kept. Use --repo to delete it too."
fi
warn "Ollama itself and its models are left untouched."
if [ "$DRY" = 0 ] && [ "$ASSUME_YES" = 0 ]; then
  [ -t 0 ] || die "non-interactive; pass --yes to confirm (or --dry-run to preview)."
  read -r -p "${B}Proceed? [y/N]${Z} " a || true
  case "$a" in y|Y|yes|YES) ;; *) die "aborted; nothing changed.";; esac
fi

# ---- remove -----------------------------------------------------------------
say "Removing lean-coder"
for i in "${!TARGETS[@]}"; do
  run "rm -rf '${TARGETS[$i]}'"; ok "removed ${TARGETS[$i]}"
done

if [ "$HAVE_DROPIN" = 1 ]; then
  say "Reverting Ollama bind drop-in (needs sudo)"
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${DIM}  would: sudo rm -f $DROPIN; daemon-reload; restart ollama${Z}"
  elif sudo rm -f "$DROPIN" && sudo systemctl daemon-reload && sudo systemctl restart ollama; then
    ok "removed drop-in and restarted ollama (back to its default bind)"
  else
    warn "couldn't fully revert the drop-in - check: journalctl -u ollama -e"
  fi
fi

if [ "$PATH_LINE_PRESENT" = 1 ]; then
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${DIM}  would: strip the lean-coder PATH line from $RC${Z}"
  else
    # drop the marker comment and the export line that follows it
    tmp="$(mktemp)"
    awk '
      /# added by lean-coder install.sh/ { skip=1; next }
      skip && /export PATH=.*\.local\/bin/ { skip=0; next }
      { skip=0; print }
    ' "$RC" > "$tmp" && cat "$tmp" > "$RC" && rm -f "$tmp"
    ok "removed the PATH line from $RC (open a new shell to refresh)"
  fi
fi

ok "lean-coder removed."

if [ "$NUKE_REPO" = 1 ]; then
  say "Deleting source repo $SCRIPT_DIR"
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${DIM}  would: rm -rf $SCRIPT_DIR${Z}"
  else
    # last, and from outside the tree, since we're deleting the dir we live in
    cd / && rm -rf "$SCRIPT_DIR" && ok "removed $SCRIPT_DIR"
  fi
fi
