#!/usr/bin/env bash
#
# lean-coder installer - sets lean-coder up on a Linux box (see README "Setup").
#
# Core (always): verify Python 3.11+, copy the script to an install dir, make it
# executable, and symlink it onto your PATH as `lean_coder`.
# Optional (opt-in): install Ollama, pull a model, seed a config file.
#
# Re-running is safe - it updates an existing install in place.
#
# Usage:
#   ./install.sh [options]
#     --dir DIR        install dir            (default: ~/lean-coder)
#     --bin DIR        symlink dir on PATH    (default: ~/.local/bin)
#     --host URL       seed config 'host'     (writes config if none exists)
#     --model NAME     seed config 'model'    (and pull target for --pull)
#     --with-ollama    install Ollama if missing (no prompt)
#     --no-ollama      never install Ollama
#     --expose         bind Ollama to 0.0.0.0:11434 for LAN access (sudo, systemd)
#     --bind ADDR      bind Ollama to ADDR, e.g. 192.0.2.42:11434 (sudo, systemd)
#     --no-expose      never touch Ollama's network bind
#     --pull           pull --model (or the config model) via ollama
#     -y, --yes        assume "yes" to prompts (non-interactive)
#     --dry-run        print what would happen, change nothing
#     --uninstall      remove the symlink and install dir
#     --remote HOST    install onto a remote box over SSH (runs this installer there;
#                      other flags are forwarded, e.g. --remote box --with-ollama)
#     -h, --help       this help

set -euo pipefail

# ---- defaults ---------------------------------------------------------------
DIR="${LEANCODER_DIR:-$HOME/lean-coder}"
BIN="${LEANCODER_BIN:-$HOME/.local/bin}"
CONFIG_DIR="$HOME/.config/leancoder"
CONFIG="$CONFIG_DIR/config.toml"
HOST=""
MODEL=""
OLLAMA_MODE="ask"   # ask | yes | no
EXPOSE_MODE="ask"   # ask | yes | no
BIND_ADDR="0.0.0.0:11434"
PULL=0
ASSUME_YES=0
DRY=0
UNINSTALL=0
REMOTE_HOST=""

# ${BASH_SOURCE[0]:-$0}: when piped (curl | bash) the script has no BASH_SOURCE, and
# set -u would abort on the bare reference - fall back to $0 (self-fetch resets this anyway).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# GitHub source, for a self-fetch when the installer is run standalone (e.g. the
# curl | bash one-liner: install.sh arrives alone, without the script + bundled dirs).
REPO="codemonkeying/lean-coder"
BRANCH="${LEANCODER_BRANCH:-main}"
TARBALL_URL="https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz"

# Termux (Android) has no sudo/systemd and a non-FHS prefix ($PREFIX). Detect it so
# the Ollama/systemd steps degrade to a clear warning instead of failing.
IS_TERMUX=0
case "${PREFIX:-}" in *com.termux*) IS_TERMUX=1;; esac
[ -d /data/data/com.termux ] && IS_TERMUX=1

# ---- pretty output ----------------------------------------------------------
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

ask_yn() {  # ask_yn "question" -> returns 0 for yes (honors --yes)
  [ "$ASSUME_YES" = 1 ] && return 0
  [ -t 0 ] || return 1            # non-interactive and no --yes => no
  local a; read -r -p "${B}$1 [y/N]${Z} " a || true
  case "$a" in y|Y|yes|YES) return 0;; *) return 1;; esac
}

usage() { sed -n '3,28p' "$0" | sed 's/^#\( \|$\)//'; }

list_ipv4() {  # emit "ifname ip" for each global-scope IPv4 address
  if command -v ip >/dev/null 2>&1; then
    ip -o -4 addr show scope global 2>/dev/null | awk '{sub(/\/[0-9]+$/,"",$4); print $2" "$4}'
  elif command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | awk '{print "iface "$1}'
  fi
}

pick_bind_addr() {  # interactive menu -> prints "host:port" on stdout, or fails if cancelled
  local port="${1:-11434}"
  local -a opts labels
  opts=("127.0.0.1:$port"); labels=("127.0.0.1   local only (same machine)")
  opts+=("0.0.0.0:$port");  labels+=("0.0.0.0     all interfaces (LAN + everything)")
  local ifn ip
  while read -r ifn ip; do
    [ -n "$ip" ] || continue
    opts+=("$ip:$port"); labels+=("$ip   interface $ifn")
  done < <(list_ipv4)
  {
    printf '%s\n' "${B}Bind Ollama to which address?${Z}"
    local i
    for i in "${!opts[@]}"; do printf '  %d) %s\n' "$((i + 1))" "${labels[$i]}"; done
    printf '  0) cancel (leave it on localhost)\n'
  } >&2
  local sel; read -r -p "choice [0-${#opts[@]}]: " sel >&2 || true
  case "$sel" in
    ''|0|*[!0-9]*) return 1;;
  esac
  [ "$sel" -ge 1 ] && [ "$sel" -le "${#opts[@]}" ] || return 1
  printf '%s' "${opts[$((sel - 1))]}"
}

# ---- Ollama network-bind helpers -------------------------------------------
normalize_addr() { case "$1" in *:*) printf '%s' "$1";; *) printf '%s:11434' "$1";; esac; }

host_is_public() {  # $1 = host:port or scheme://host:port ; 0 if reachable off-box
  local h="${1#*://}"; h="${h%%:*}"
  case "$h" in ""|127.0.0.1|localhost|::1) return 1;; *) return 0;; esac
}

systemd_has_ollama() {
  command -v systemctl >/dev/null 2>&1 || return 1
  [ "$(systemctl show ollama -p LoadState --value 2>/dev/null)" = "loaded" ]
}

ollama_host_addr() {  # configured OLLAMA_HOST from the systemd unit (may be empty)
  command -v systemctl >/dev/null 2>&1 || return 0
  systemctl show ollama -p Environment --value 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^OLLAMA_HOST=//p' | head -1
}

do_expose() {  # $1 = addr (host:port) - write a systemd drop-in and restart
  local addr; addr="$(normalize_addr "$1")"
  local dropin=/etc/systemd/system/ollama.service.d/zz-leancoder-bind.conf
  say "Binding Ollama -> $addr  (systemd drop-in; needs sudo)"
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${DIM}  would: write $dropin (OLLAMA_HOST=$addr), daemon-reload, restart ollama${Z}"
    return
  fi
  if ! { sudo mkdir -p /etc/systemd/system/ollama.service.d \
      && printf '[Service]\nEnvironment="OLLAMA_HOST=%s"\n' "$addr" | sudo tee "$dropin" >/dev/null \
      && sudo systemctl daemon-reload && sudo systemctl restart ollama; }; then
    warn "failed to apply bind (sudo/systemd) - check: journalctl -u ollama -e"
    return
  fi
  ok "OLLAMA_HOST=$(ollama_host_addr) (drop-in: $dropin)"
  if command -v ss >/dev/null 2>&1; then
    if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -q ":${addr##*:}$"; then
      ok "listening on port ${addr##*:}"
    else
      warn "not yet listening on ${addr##*:} - check: journalctl -u ollama -e"
    fi
  fi
  warn "Ollama is now exposed with NO authentication - restrict via firewall / trusted LAN."
}

# ---- remote install: stage this installer + script to a box, run it there ---
remote_install() {  # $1 = host (user@host or an ssh alias)
  local host="$1"
  [ -f "$SCRIPT_DIR/lean_coder.py" ] || die "lean_coder.py not found next to this installer ($SCRIPT_DIR)."
  [ -f "$SCRIPT_DIR/install.sh" ]   || die "install.sh not found ($SCRIPT_DIR)."

  # Forward every original flag except --remote and its host value.
  local fwd=() skip=0 a
  for a in "${ORIG_ARGS[@]}"; do
    if [ "$skip" = 1 ]; then skip=0; continue; fi
    if [ "$a" = "--remote" ]; then skip=1; continue; fi
    fwd+=("$a")
  done
  local q="" x                       # quote forwarded args for the remote shell
  for x in "${fwd[@]}"; do q="$q $(printf '%q' "$x")"; done

  local files="lean_coder.py install.sh"
  [ -f "$SCRIPT_DIR/README.md" ]  && files="$files README.md"
  [ -f "$SCRIPT_DIR/LEAN_TOOLS.md" ] && files="$files LEAN_TOOLS.md"
  [ -f "$SCRIPT_DIR/VERSION" ] && files="$files VERSION"
  # the bundled plugin dirs must travel too: core loads lean-tools/builtins.py at import
  # and the default ollama backend from providers/ - an install missing them won't start.
  [ -d "$SCRIPT_DIR/lean-tools" ] && files="$files lean-tools"
  [ -d "$SCRIPT_DIR/providers" ] && files="$files providers"

  say "Remote install -> $host"
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${DIM}  would: tar [$files] | ssh $host (extract to a temp dir)${Z}"
    printf '%s\n' "${DIM}  would: ssh -tt $host bash <tmp>/install.sh$q${Z}"
    ok "dry-run: nothing changed on $host"
    return 0
  fi

  # Authenticate once: open (or reuse) an SSH ControlMaster and route every step
  # through it, so password-auth boxes prompt a single time instead of per call.
  # LEANCODER_SSH_CONTROL lets a caller (e.g. the /provision lean-tool) hand us a
  # master it already opened, so the whole provision is one prompt.
  local ctl own=0
  if [ -n "${LEANCODER_SSH_CONTROL:-}" ]; then
    ctl="$LEANCODER_SSH_CONTROL"
  else
    ctl="${TMPDIR:-/tmp}/lc-cm-$$"
    ssh -o ControlMaster=yes -o ControlPath="$ctl" -o ControlPersist=60 \
        -o ConnectTimeout=10 -o LogLevel=ERROR -fN "$host" \
      || die "ssh to $host failed (check connectivity / auth)."
    own=1
  fi
  rsh() { ssh -o ControlPath="$ctl" "$@"; }    # reuses the master (no re-auth)
  close_master() { [ "$own" = 1 ] && ssh -o ControlPath="$ctl" -O exit "$host" 2>/dev/null || true; }

  local tmp
  tmp="$(rsh "$host" 'mktemp -d "${TMPDIR:-/tmp}/lc-install.XXXXXX"')" \
    || { close_master; die "ssh to $host failed (check connectivity / auth)."; }
  say "Staging to $host:$tmp"
  if ! tar -C "$SCRIPT_DIR" -cf - $files | rsh "$host" "tar -xf - -C '$tmp'"; then
    rsh "$host" "rm -rf '$tmp'" 2>/dev/null || true
    close_master
    die "copying files to $host failed."
  fi
  ok "copied installer + script"

  say "Running installer on $host"
  local rc
  if rsh -tt "$host" "bash '$tmp/install.sh'$q"; then rc=0; else rc=$?; fi
  rsh "$host" "rm -rf '$tmp'" 2>/dev/null || true
  close_master
  [ "$rc" = 0 ] || die "remote install exited $rc"
  ok "lean_coder installed on $host"
}

# ---- args -------------------------------------------------------------------
ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2;;
    --bin) BIN="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --with-ollama) OLLAMA_MODE="yes"; shift;;
    --no-ollama) OLLAMA_MODE="no"; shift;;
    --expose) EXPOSE_MODE="yes"; shift;;
    --bind) EXPOSE_MODE="yes"; BIND_ADDR="$2"; shift 2;;
    --no-expose) EXPOSE_MODE="no"; shift;;
    --pull) PULL=1; shift;;
    -y|--yes) ASSUME_YES=1; shift;;
    --dry-run) DRY=1; shift;;
    --uninstall) UNINSTALL=1; shift;;
    --remote) REMOTE_HOST="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "unknown option: $1 (try --help)";;
  esac
done

# Self-fetch: when run standalone (the curl | bash one-liner ships install.sh
# ALONE, with no lean_coder.py or bundled dirs beside it), pull the repo tarball
# and repoint SCRIPT_DIR at the extracted tree so the rest of the flow is a normal
# local install. A remote install self-fetches on the remote, so skip it here.
if [ -z "$REMOTE_HOST" ] && [ "$UNINSTALL" != 1 ] && [ ! -f "$SCRIPT_DIR/lean_coder.py" ]; then
  command -v curl >/dev/null 2>&1 || die "curl not found - needed to fetch lean-coder."
  command -v tar  >/dev/null 2>&1 || die "tar not found - needed to unpack lean-coder."
  say "Fetching lean-coder ($REPO@$BRANCH)"
  FETCH_TMP="$(mktemp -d "${TMPDIR:-/tmp}/lc-fetch.XXXXXX")"
  trap 'rm -rf "$FETCH_TMP"' EXIT
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${DIM}  would: curl $TARBALL_URL | tar -xz -> use as source${Z}"
  else
    curl -fsSL "$TARBALL_URL" | tar -xz -C "$FETCH_TMP" \
      || die "download failed ($TARBALL_URL) - check network / branch name."
    SRC="$(find "$FETCH_TMP" -maxdepth 1 -type d -name 'lean-coder-*' | head -1)"
    [ -n "$SRC" ] && [ -f "$SRC/lean_coder.py" ] \
      || die "fetched archive is missing lean_coder.py."
    SCRIPT_DIR="$SRC"
    ok "fetched into $SCRIPT_DIR"
  fi
fi

# Remote install short-circuits the whole local flow: stage to the box and run
# this same installer there with the remaining flags.
if [ -n "$REMOTE_HOST" ]; then
  remote_install "$REMOTE_HOST"
  exit $?
fi

LINK="$BIN/lean_coder"

# ---- uninstall --------------------------------------------------------------
if [ "$UNINSTALL" = 1 ]; then
  say "Uninstalling lean-coder"
  run "rm -f '$LINK'";       ok "removed symlink $LINK"
  run "rm -rf '$DIR'";       ok "removed install dir $DIR"
  warn "left your config ($CONFIG) and Ollama untouched."
  exit 0
fi

# ---- 1. Python 3.11+ --------------------------------------------------------
say "Checking Python"
command -v python3 >/dev/null 2>&1 || die "python3 not found - install Python 3.11+ first."
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)' \
  || die "Python $PYV found, but 3.11+ is required (needs tomllib)."
ok "python3 $PYV"

# ---- 2. install the script + symlink ---------------------------------------
[ -f "$SCRIPT_DIR/lean_coder.py" ] || die "lean_coder.py not found next to this installer ($SCRIPT_DIR)."
say "Installing lean_coder -> $DIR"
run "mkdir -p '$DIR'"
# ALL copies below are skipped when installing IN PLACE (SCRIPT_DIR == DIR): the
# source IS the dest, so `cp -f x x` would error ("same file") and abort under set -e -
# this was a real install failure when re-running from the install dir. In place there
# is nothing to copy (the files are already there), so we just re-link + chmod below.
if [ "$SCRIPT_DIR" != "$DIR" ]; then
  run "cp -f '$SCRIPT_DIR/lean_coder.py' '$DIR/'"
  [ -f "$SCRIPT_DIR/README.md" ]  && run "cp -f '$SCRIPT_DIR/README.md' '$DIR/'"
  [ -f "$SCRIPT_DIR/LEAN_TOOLS.md" ] && run "cp -f '$SCRIPT_DIR/LEAN_TOOLS.md' '$DIR/'"
  [ -f "$SCRIPT_DIR/VERSION" ] && run "cp -f '$SCRIPT_DIR/VERSION' '$DIR/'"
  # keep the installer alongside the script so an installed box can re-provision
  # others (the /provision lean-tool shells to it).
  [ -f "$SCRIPT_DIR/install.sh" ]   && run "cp -f '$SCRIPT_DIR/install.sh' '$DIR/'"
  [ -f "$SCRIPT_DIR/uninstall.sh" ] && run "cp -f '$SCRIPT_DIR/uninstall.sh' '$DIR/'"
  # Ship the bundled plugin dirs. Core loads lean-tools/builtins.py at import (the builtin
  # tool surface: read_file..run_command) and the default ollama backend from providers/ -
  # an install missing them won't even start. OVERLAY the bundled files on top (never
  # `rm -rf` the dir first): a re-install must NOT delete user-added providers/lean-tools -
  # dropping your own plugin files in these dirs is a supported feature, and wiping them
  # cost a user their custom provider once. Bundled files are still refreshed (cp -f
  # overwrites), we just don't remove anything we didn't ship.
  for d in lean-tools providers; do
    if [ -d "$SCRIPT_DIR/$d" ]; then
      run "mkdir -p '$DIR/$d'"
      run "cp -f '$SCRIPT_DIR/$d/'*.py '$DIR/$d/'"
    fi
  done
  # demo asset for the README (the top-of-page GIF). Best-effort; not load-bearing.
  if [ -d "$SCRIPT_DIR/demos" ]; then
    run "mkdir -p '$DIR/demos'"
    run "cp -f '$SCRIPT_DIR/demos/'*.gif '$DIR/demos/' 2>/dev/null || true"
  fi
fi
run "chmod +x '$DIR/lean_coder.py'"
ok "copied and made executable"

say "Linking command -> $LINK"
run "mkdir -p '$BIN'"
run "ln -sf '$DIR/lean_coder.py' '$LINK'"
ok "lean_coder -> $DIR/lean_coder.py"

# PATH check (only manage the default bin dir; never touch rc files for custom --bin)
case ":$PATH:" in
  *":$BIN:"*) ok "$BIN already on PATH";;
  *)
    if [ "$BIN" = "$HOME/.local/bin" ]; then
      case "$(basename "${SHELL:-}")" in
        zsh) RC="$HOME/.zshrc";; bash) RC="$HOME/.bashrc";; *) RC="$HOME/.profile";;
      esac
      LINE='export PATH="$HOME/.local/bin:$PATH"'
      if [ "$DRY" = 1 ]; then
        printf '%s\n' "${DIM}  would: append PATH line to $RC${Z}"
      elif ! grep -qsF "$LINE" "$RC" 2>/dev/null; then
        printf '\n# added by lean-coder install.sh\n%s\n' "$LINE" >> "$RC"
        warn "added $BIN to PATH in $RC - run: source $RC  (or open a new shell)"
      fi
    else
      warn "$BIN is not on your PATH - add it yourself to run 'lean_coder' bare."
    fi
    ;;
esac

# ---- 3. Ollama (optional) ---------------------------------------------------
say "Checking Ollama"
if command -v ollama >/dev/null 2>&1; then
  ok "ollama present ($(ollama --version 2>/dev/null | head -1))"
elif [ "$IS_TERMUX" = 1 ]; then
  warn "Termux detected - Ollama's Linux installer (sudo/systemd) doesn't apply here."
  warn "Point lean_coder at an Ollama on another box: --host http://HOST:11434 (or set 'host' in the config)."
else
  DO_OLLAMA=0
  case "$OLLAMA_MODE" in
    yes) DO_OLLAMA=1;;
    no)  warn "ollama not installed (skipped by --no-ollama).";;
    ask) if ask_yn "Ollama not found. Install it now (downloads + sudo, systemd service)?"; then DO_OLLAMA=1; fi;;
  esac
  if [ "$DO_OLLAMA" = 1 ]; then
    run "curl -fsSL https://ollama.com/install.sh | sh"
    ok "ollama installed (service: systemctl status ollama)"
  else
    warn "without Ollama, lean_coder has nothing to talk to - install later: curl -fsSL https://ollama.com/install.sh | sh"
  fi
fi

# ---- 4. Ollama network bind (optional) -------------------------------------
# Detect current state first; only offer/act when it isn't already exposed.
if [ "$EXPOSE_MODE" != "no" ]; then
  say "Checking Ollama network bind"
  CUR_HOST="$(ollama_host_addr || true)"
  if ! command -v ollama >/dev/null 2>&1; then
    [ "$EXPOSE_MODE" = "yes" ] && warn "ollama not installed - can't bind yet."
  elif ! systemd_has_ollama; then
    warn "ollama isn't a systemd service here (manual/Docker?) - set OLLAMA_HOST=$BIND_ADDR yourself."
  elif [ -n "$CUR_HOST" ] && host_is_public "$CUR_HOST"; then
    ok "already bound to $CUR_HOST - nothing to do"
  else
    case "$EXPOSE_MODE" in
      yes) do_expose "$BIND_ADDR";;
      ask) # interactive only - pick an interface; a blanket --yes never auto-exposes
        if [ -t 0 ]; then
          warn "Ollama is listening on localhost only."
          sel_addr="$(pick_bind_addr "${BIND_ADDR##*:}")" || sel_addr=""
          if [ -n "$sel_addr" ]; then do_expose "$sel_addr"
          else ok "left Ollama on localhost."; fi
        fi;;
    esac
  fi
fi

# ---- 5. pull a model (optional) --------------------------------------------
if [ "$PULL" = 1 ]; then
  PULL_MODEL="${MODEL:-qwen3-coder:30b}"
  if command -v ollama >/dev/null 2>&1; then
    say "Pulling model: $PULL_MODEL (may be large)"
    run "ollama pull '$PULL_MODEL'"
    ok "pulled $PULL_MODEL"
  else
    warn "--pull requested but ollama isn't installed; skipping."
  fi
fi

# ---- 6. seed config (only if none exists) ----------------------------------
if [ -e "$CONFIG" ]; then
  ok "config exists ($CONFIG) - left untouched"
elif [ -n "$HOST" ] || [ -n "$MODEL" ]; then
  say "Writing starter config -> $CONFIG"
  C_HOST="${HOST:-http://localhost:11434}"
  C_MODEL="${MODEL:-qwen3-coder:30b}"
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${DIM}  would: write host=$C_HOST model=$C_MODEL (num_ctx omitted for auto-detect)${Z}"
  else
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG" <<EOF
# lean-coder defaults - a bare \`lean_coder\` connects here.
# num_ctx intentionally omitted so auto-detect (with the safe 32768 cap) stays active.
host = "$C_HOST"
model = "$C_MODEL"
temperature = 0.7
top_p = 0.8
top_k = 20
repeat_penalty = 1.05
EOF
  fi
  ok "config seeded (host=$C_HOST, model=$C_MODEL)"
else
  warn "no config written (pass --host/--model to seed one); lean_coder will use built-in defaults."
fi

# ---- done -------------------------------------------------------------------
echo
say "${B}Done.${Z}"
echo "  Run:        ${B}lean_coder${Z}            ${DIM}(or: $DIR/lean_coder.py)${Z}"
echo "  Override:   ${B}lean_coder --host http://HOST:11434 --model NAME${Z}"
echo "  Help:       ${B}lean_coder --help${Z}   ·   in-REPL: ${B}/help${Z}"
# NB: keep this an `if` (not `... && echo`) so a non-dry-run leaves $? == 0 -
# a trailing `&&` whose left side is false would make the script exit 1 on success.
if [ "$DRY" = 1 ]; then echo "  ${YEL}(dry-run: nothing was changed)${Z}"; fi
