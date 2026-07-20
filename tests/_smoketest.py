"""Offline smoke test of lean_coder internals (no Ollama required)."""
import shutil, sys, tempfile, atexit, os, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (this file lives in tests/)
import lean_coder as lc

# Safety net: tests reassign lc.CONFIG_PATH and some call save_config/autosave_config.
# Snapshot the user's REAL config.toml now and restore it on exit, so a stray test can
# never leave the real config clobbered (e.g. providers_dir pointing at a scratch dir).
_REAL_CFG = lc.CONFIG_PATH
_REAL_CFG_BAK = _REAL_CFG.read_text() if _REAL_CFG.is_file() else None
@atexit.register
def _restore_real_cfg():
    try:
        if _REAL_CFG_BAK is not None:
            _REAL_CFG.write_text(_REAL_CFG_BAK)
        elif _REAL_CFG.is_file():
            _REAL_CFG.unlink()
    except OSError:
        pass

FIX = Path(tempfile.mkdtemp(prefix="leancoder_smoke_"))
ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    ok = ok and cond

# 0. command/completion/help contract parity (see COMMANDS.md). MUST run BEFORE any
# lean-tool setup() below, which register_command()s and so mutates SLASH_COMMANDS.
_tbl = set(lc._BUILTIN_COMMANDS_TABLE)
_slash = set(lc.SLASH_COMMANDS)
_help = set(c.split()[0] for c, _ in lc.HELP_COMMANDS)
_aliases = {"/exit", "/q", "/h", "/?", "/models", "/providers", "/disconnect"}
check("every SLASH_COMMAND is dispatchable", not (_slash - _tbl - set(lc._lean_tool_commands)))
check("every HELP command is dispatchable", not (_help - _tbl))
check("every canonical command is in HELP",
      not [c for c in _tbl if c not in _help and c not in _aliases])
check("/mcp fully wired (dispatch+slash+help)",
      "/mcp" in _tbl and "/mcp" in _slash and "/mcp" in _help)

# 1. Fixed overhead budget (system prompt + always-on tool schemas). README claims
# ~2k core; ceiling gives headroom for later additions without lying in the README.
sys_msg = {"role": "system", "content": lc.SYSTEM_PROMPT + "\nWorking directory: /x"}
overhead = lc.messages_tokens([sys_msg], lc.active_tools(lc.Config()))
check("fixed overhead < 2500 tokens (~2k README claim + headroom)", overhead < 2500, f"~{overhead} tokens")
# the hot metering path must survive non-string content (int/None from a loaded or
# synthesised history) and size a block LIST by its JSON length, not its element count
# (a crash/skew here would feed a bad ctx meter into handover decisions).
_mt_msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": 42},
            {"role": "assistant", "content": None}]
check("_raw_messages_tokens: non-string content doesn't crash the meter",
      isinstance(lc._raw_messages_tokens(_mt_msgs, []), int))
check("_raw_messages_tokens: block list sized by JSON, not element count",
      lc._raw_messages_tokens([{"role": "user", "content": [{"type": "text", "text": "x" * 400}]}], [])
      > lc._raw_messages_tokens([{"role": "user", "content": [{"type": "text", "text": "x"}]}], []) + 50)
check("8 base tools (ssh moved to a lean-tool; bg_status added)", len(lc.TOOLS) == 8, f"{len(lc.TOOLS)}")
check("bg_status is a core tool", "bg_status" in [t["function"]["name"] for t in lc.TOOLS])
check("active_tools adds ask_user_to_run by default", len(lc.active_tools(lc.Config())) == 11)
check("active_tools drops it when disabled",
      len(lc.active_tools(lc.Config(ask_user_to_run=False))) == 10)
check("update_plan is on the surface at every non-chat tier",
      all("update_plan" in [t["function"]["name"] for t in lc.active_tools(lc.Config(leash=l))]
          for l in ("r", "rw", "rwe")))
check("ssh is not a core tool", "ssh" not in [t["function"]["name"] for t in lc.TOOLS])
# elective handover: request_handover is surfaced ONLY when offer_handover (soft zone)
_no_ho = [t["function"]["name"] for t in lc.active_tools(lc.Config())]
_yes_ho = [t["function"]["name"] for t in lc.active_tools(lc.Config(), offer_handover=True)]
check("request_handover absent below the soft zone", "request_handover" not in _no_ho)
check("request_handover offered when in the soft zone", "request_handover" in _yes_ho)
check("request_handover allowed at any tier (agent-internal)",
      lc._leash_allows_tool("r", "request_handover") and lc._leash_allows_tool("chat", "request_handover"))
# _request_handover guards: below soft (zone 'ok') it refuses; soft+ it queues
_rh = lc.Agent.__new__(lc.Agent)
_rh.cfg = lc.Config(handover_soft=0.70, handover_hard=0.95,
                    provider_settings={"ollama": {"num_ctx": 1000}}, provider="ollama")
_rh.messages = []; _rh.tool_defs = []; _rh.last_prompt_tokens = 100  # 10% -> 'ok'
_rh.ctx = lc.ContextMeter(_rh); _rh._elected_handover = False
_msg = _rh._request_handover()
check("request_handover refuses below soft (no elect)",
      _rh._elected_handover is False and "not yet" in _msg.lower())
_rh.last_prompt_tokens = 800   # 80% -> 'soft'
_msg2 = _rh._request_handover()
check("request_handover queues in the soft zone", _rh._elected_handover is True and "queued" in _msg2.lower())
check("operator_note shows command + exit",
      lc.operator_note("ls", 0, proposed="ls") == "[operator ran a command directly]\n$ ls\nexit 0")
check("operator_note flags an edit",
      "(edited from: ls)" in lc.operator_note("ls -a", 1, proposed="ls"))
check("operator_note: not run -> (not run)", "(not run)" in lc.operator_note("x", None))

# --- tool-schema SHAPE contract: every model-facing schema (core builtins + the
# always-on agent tools) must be a well-formed OpenAI function tool. A malformed
# schema 400s the API, so this is a hard shape gate covering NOTE_TOOL too.
_all_schemas = list(lc.TOOLS) + [lc.UPDATE_PLAN_TOOL, lc.REQUEST_HANDOVER_TOOL, lc.NOTE_TOOL]
def _schema_wellformed(t):
    if t.get("type") != "function":
        return False
    fn = t.get("function")
    if not isinstance(fn, dict):
        return False
    if not (isinstance(fn.get("name"), str) and fn["name"]):
        return False
    if not (isinstance(fn.get("description"), str) and fn["description"]):
        return False
    p = fn.get("parameters")
    if not (isinstance(p, dict) and p.get("type") == "object" and isinstance(p.get("properties"), dict)):
        return False
    req = p.get("required", [])
    if not isinstance(req, list) or not set(req).issubset(set(p["properties"])):
        return False   # required must be a subset of declared properties
    return True
_bad_schema = [t.get("function", {}).get("name", "?") for t in _all_schemas if not _schema_wellformed(t)]
check("tool schema shape: every core/agent schema is well-formed", not _bad_schema, f"malformed: {_bad_schema}")
check("NOTE_TOOL is a well-formed schema with the four actions",
      _schema_wellformed(lc.NOTE_TOOL)
      and set(lc.NOTE_TOOL["function"]["parameters"]["properties"]["action"]["enum"])
          == {"add", "recent", "grep", "range"})
check("note is agent-internal (allowed at every tier, never a routed builtin)",
      lc._leash_allows_tool("r", "note") and lc._leash_allows_tool("chat", "note")
      and "note" not in lc.EXEC_TOOLS)
check("note is on the surface at every non-chat tier",
      all("note" in [t["function"]["name"] for t in lc.active_tools(lc.Config(leash=l))]
          for l in ("r", "rw", "rwe")))

# --- note tool BEHAVIOUR: add/recent/grep/range, default action, spool trim, read cap
_nag = lc.Agent(lc.Config(cwd=FIX, notes_spool=2000))
check("note: fresh notebook is empty list", _nag.notes == [])
_nag._note({"text": "first finding about X"})          # bare text -> add
check("note add (default action) appends a dtg entry",
      len(_nag.notes) == 1 and _nag.notes[0]["text"] == "first finding about X"
      and "-" in _nag.notes[0]["dtg"])
_nag._note({"action": "add", "text": "second thing"})
check("note recent (default read) shows entries",
      "first finding" in _nag._note({}) and "second thing" in _nag._note({}))
check("note recent honours n", "second thing" in _nag._note({"action": "recent", "n": 1})
      and "first finding" not in _nag._note({"action": "recent", "n": 1}))
check("note grep matches", "second" in _nag._note({"action": "grep", "pattern": "second"})
      and "first" not in _nag._note({"action": "grep", "pattern": "second"}))
check("note grep no-match message", "no notes match" in _nag._note({"action": "grep", "pattern": "zzzzz"}))
# range: force known dtgs, then query
_nag.notes = [{"dtg": "2026-01-01 09:00", "text": "jan entry"},
              {"dtg": "2026-06-15 12:00", "text": "jun entry"}]
check("note range from-only (since)", "jun entry" in _nag._note({"action": "range", "from": "2026-03-01"})
      and "jan entry" not in _nag._note({"action": "range", "from": "2026-03-01"}))
check("note range between two dates",
      "jan entry" in _nag._note({"action": "range", "from": "2026-01-01", "to": "2026-02-01"})
      and "jun entry" not in _nag._note({"action": "range", "from": "2026-01-01", "to": "2026-02-01"}))
check("note range needs a from", "needs a 'from'" in _nag._note({"action": "range"}))
# spool trim: a tiny spool keeps only the newest lines
_sag = lc.Agent(lc.Config(cwd=FIX, notes_spool=3))
for _i in range(20):
    _sag._note({"text": f"entry {_i}"})
check("note spool trims to newest ~notes_spool lines",
      1 <= len(_sag.notes) <= 3 and _sag.notes[-1]["text"] == "entry 19")
# read cap: many entries, recall is capped to _NOTES_READ_CAP lines
_cag = lc.Agent(lc.Config(cwd=FIX, notes_spool=100000))
for _i in range(1000):
    _cag._note({"text": f"line {_i}"})
_dump = _cag._note({"action": "recent", "n": 1000})
check("note read is line-capped (recall can't flood context)",
      _dump.count("\n") <= lc.Agent._NOTES_READ_CAP + 2 and "showing the latest" in _dump)

# --- /note command dispatch + tab-completion contract
check("/note is a registered builtin command", "/note" in lc._BUILTIN_COMMANDS and "/note" in lc.SLASH_COMMANDS)
check("/note is in the help table", any(c[0].startswith("/note") for c in lc.HELP_COMMANDS))
check("argcomp: /note subcommands",
      lc._arg_completions(object(), lc.Config(cwd=FIX), "/note") == ["recent", "grep", "range", "add", "clear"])
import io as _io_note, contextlib as _ctx_note
_ncmdag = lc.Agent(lc.Config(cwd=FIX))
with _ctx_note.redirect_stdout(_io_note.StringIO()):
    lc.handle_note_command(_ncmdag, _ncmdag.cfg, "a manual operator note")   # bare text -> add
check("/note <text> adds an entry tagged operator: via the shared _note path",
      len(_ncmdag.notes) == 1 and _ncmdag.notes[0]["text"] == "operator: a manual operator note")
with _ctx_note.redirect_stdout(_io_note.StringIO()):
    lc.handle_note_command(_ncmdag, _ncmdag.cfg, "clear")
check("/note clear empties the notebook", _ncmdag.notes == [])

# /sh must run where the session points: forward agent.remote into
# run_direct_command (regression - the handler used to drop it and run local).
_cap = {}
_orig_rdc = lc.run_direct_command
lc.run_direct_command = lambda cfg, cmd, reason=None, remote=None, editable=True: (
    _cap.update(cmd=cmd, remote=remote, editable=editable)
    or "[operator ran a command directly]\n$ x\nexit 0")
try:
    _ag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
    _ag.remote = "SENTINEL_REMOTE"            # stand-in for a RemoteWorkspace
    _note = lc.run_sh_command(_ag, _ag.cfg, "whoami")
    check("/sh forwards agent.remote (connected -> runs remote)",
          _cap.get("remote") == "SENTINEL_REMOTE" and _cap.get("cmd") == "whoami")
    check("/sh runs without the edit prompt (editable=False, one Enter)",
          _cap.get("editable") is False)
    check("/sh appends the operator note for the model",
          _ag.messages[-1]["role"] == "user" and "operator ran" in _ag.messages[-1]["content"])
    _ag.remote = None
    _cap.clear()
    lc.run_sh_command(_ag, _ag.cfg, "ls")
    check("/sh local when not connected -> remote=None",
          _cap.get("remote") is None and _cap.get("cmd") == "ls")
    # empty arg + connected -> refuse interactive remote shell, no call, None
    _cap.clear()
    _ag.remote = type("R", (), {"host": "SENTINEL_REMOTE"})()
    _orig_ris = lc.run_interactive_shell
    _ris_calls = []
    lc.run_interactive_shell = (
        lambda cfg, remote=None: _ris_calls.append(remote)
        or "[operator dropped into an interactive shell on SENTINEL_REMOTE]")
    try:
        _ret = lc.run_sh_command(_ag, _ag.cfg, "")
    finally:
        lc.run_interactive_shell = _orig_ris
    check("/sh empty arg while remote -> interactive shell passed the remote",
          _ris_calls and getattr(_ris_calls[0], "host", None) == "SENTINEL_REMOTE"
          and "cmd" not in _cap and _ret and "interactive" in _ret)
    _ag.remote = None
    # empty arg + local -> drop into interactive $SHELL (stubbed), append note
    _cap.clear()
    _orig_ris = lc.run_interactive_shell
    lc.run_interactive_shell = lambda cfg, remote=None: "[operator dropped into an interactive shell]"
    try:
        _ret = lc.run_sh_command(_ag, _ag.cfg, "")
    finally:
        lc.run_interactive_shell = _orig_ris
    check("/sh empty arg local -> interactive shell, no run_direct_command call",
          "cmd" not in _cap and _ret and "interactive" in _ret)
    check("/sh interactive appends the operator note for the model",
          _ag.messages[-1]["role"] == "user" and "interactive" in _ag.messages[-1]["content"])
finally:
    lc.run_direct_command = _orig_rdc

# run_direct_command(editable=False): no edit prompt, runs the cmd as typed
_edit_called = {"n": 0}
_ran = {}
class _RC0:
    returncode = 0
_oe, _otc = lc._edit_prompt, lc._tee_capture
lc._edit_prompt = lambda *a, **k: _edit_called.update(n=_edit_called["n"] + 1) or ""
lc._tee_capture = lambda argv, cwd=None: (_ran.update(argv=argv) or (0, "some-output"))
try:
    note = lc.run_direct_command(lc.Config(cwd=FIX), "echo hi", editable=False)
    check("editable=False skips the edit prompt", _edit_called["n"] == 0)
    check("editable=False runs the command as typed", "echo hi" in " ".join(_ran.get("argv") or []))
    check("editable=False note reflects exit 0", "exit 0" in note)
    check("run_direct_command feeds captured output to the model", "some-output" in note)
finally:
    lc._edit_prompt, lc._tee_capture = _oe, _otc

# captured-output cleaning: ANSI stripped, TUI -> note, no-capture -> None
check("_clean_captured strips ANSI", lc._clean_captured("\x1b[31mred\x1b[0m x") == "red x")
check("_clean_captured TUI -> note",
      "full-screen" in (lc._clean_captured("\x1b[?1049h\x1b[2Hjunk") or ""))
check("_clean_captured None stays None", lc._clean_captured(None) is None)
check("_clean_captured normalizes CR", lc._clean_captured("a\r\nb\rc") == "a\nb\nc")

# operator_note with output: appends an output section (never a password)
check("operator_note with output appends it",
      "output:" in lc.operator_note("systemctl status x", 0, output="active (running)")
      and "active (running)" in lc.operator_note("systemctl status x", 0, output="active (running)"))
check("operator_note with empty output -> (none)",
      "(none)" in lc.operator_note("x", 0, output="   "))
check("operator_note no output arg -> no output section (back-compat)",
      "output:" not in lc.operator_note("ls", 0, proposed="ls"))

# _clear_master (#3): issues `-O exit` and removes a stale socket file
_argv = {}
_clr_sock = Path(tempfile.mktemp(suffix=".sock"))
_clr_sock.write_text("")                       # stand-in stale socket
_or3 = lc.subprocess.run
lc.subprocess.run = lambda a, **k: _argv.update(a=a) or _RC0()
try:
    lc._clear_master("user@host", str(_clr_sock))
    check("_clear_master sends ssh -O exit", "-O" in _argv["a"] and "exit" in _argv["a"])
    check("_clear_master removes the stale socket file", not _clr_sock.exists())
finally:
    lc.subprocess.run = _or3

# 2. Ignore matcher
(FIX / "src").mkdir()
(FIX / "node_modules" / "x").mkdir(parents=True)
(FIX / "src" / "app.py").write_text("print('hi')\n")
(FIX / "pkg.lock").write_text("x")
(FIX / ".git").mkdir()
ig = lc.IgnoreMatcher(FIX)
check("ignores node_modules/", ig.ignored(FIX / "node_modules"))
check("ignores *.lock", ig.ignored(FIX / "pkg.lock"))
check("ignores .git/", ig.ignored(FIX / ".git"))
check("keeps src/app.py", not ig.ignored(FIX / "src" / "app.py"))
(FIX / ".cache").mkdir()
check("ignores .cache/ (HOME-flood guard)", lc.IgnoreMatcher(FIX).ignored(FIX / ".cache"))

# extra .leancoderignore pattern
(FIX / ".leancoderignore").write_text("secret.txt\n")
(FIX / "secret.txt").write_text("nope")
ig2 = lc.IgnoreMatcher(FIX)
check("honors .leancoderignore", ig2.ignored(FIX / "secret.txt"))

# 3. SEARCH/REPLACE parser (multi-block) - the ONE format: git-conflict markers
#    <<<<<<< SEARCH / ======= / >>>>>>> REPLACE
diff = ("<<<<<<< SEARCH\nold one\n=======\nnew one\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE")
blocks, perr = lc._parse_search_replace(diff)
check("parses 2 SEARCH/REPLACE blocks",
      perr is None and blocks == [("old one", "new one"), ("foo", "bar")], str(blocks))

# 3b. malformed block -> specific error, no crash
_bad = "<<<<<<< SEARCH\nfoo\n>>>>>>> REPLACE"
_, _be = lc._parse_search_replace(_bad)
check("parser reports malformed block", _be is not None and "before the" in _be, _be)

# 3c. the OLD bespoke ===SEARCH=== markers are NO LONGER a valid format
check("old ===SEARCH=== markers are rejected (format switched to conflict markers)",
      lc._parse_search_replace("===SEARCH===\nx\n===DIVIDER===\ny\n===REPLACE===")[1] is not None)

# 4. truncation helper (moved into the bundled builtins module with run_command)
big = "\n".join(str(i) for i in range(5000))
trunc = lc._BUILTINS._truncate_output(big)
check("run_command output truncated", "truncated" in trunc and len(trunc) < len(big))

# 4b. _confirm_action: the single tier-keyed confirm policy (tools supply only the
# preview; this owns auto-approve + the prompt). One function for write/exec/read.
_save_ask = lc._ask
try:
    _ca = lc.Config(cwd=FIX, approval="ask")
    lc._ask = lambda q: True
    check("_confirm_action write + yes -> True",
          lc._confirm_action(_ca, "write", path="x", original="a\n", new="b\n") is True)
    check("_confirm_action exec + yes -> True",
          lc._confirm_action(_ca, "exec", cmd="echo hi") is True)
    check("_confirm_action read + yes -> True",
          lc._confirm_action(_ca, "read", name="read_file", args_fmt="path=x") is True)
    lc._ask = lambda q: False
    check("_confirm_action write + no -> False",
          lc._confirm_action(_ca, "write", path="x", original=None, new="b\n") is False)
    check("_confirm_action exec + no -> False",
          lc._confirm_action(_ca, "exec", cmd="rm -rf /") is False)
    _auto = lc.Config(cwd=FIX, approval="auto")
    lc._ask = lambda q: (_ for _ in ()).throw(AssertionError("auto must not prompt"))
    check("_confirm_action auto -> True, no prompt (write)",
          lc._confirm_action(_auto, "write", path="x", original="a\n", new="b\n") is True)
    check("_confirm_action auto -> True, no prompt (exec)",
          lc._confirm_action(_auto, "exec", cmd="echo") is True)
finally:
    lc._ask = _save_ask

# 4c. _print_new_file: the new-file preview body shared by the local + remote write
# confirms. Short content prints all lines with no note; long content caps at 40 and
# appends a "(N more lines)" note so a big remote write never silently truncates.
import io as _io_pnf, contextlib as _ctx_pnf
_pnf_short = _io_pnf.StringIO()
with _ctx_pnf.redirect_stdout(_pnf_short):
    lc._print_new_file("a\nb\nc\n")
check("_print_new_file: short content -> all lines, no truncation note",
      _pnf_short.getvalue().count("\n") == 3 and "more lines" not in _pnf_short.getvalue())
_pnf_long = _io_pnf.StringIO()
with _ctx_pnf.redirect_stdout(_pnf_long):
    lc._print_new_file("\n".join(str(i) for i in range(100)))
check("_print_new_file: >40 lines -> caps at 40 + '(60 more lines)' note",
      "60 more lines" in _pnf_long.getvalue() and _pnf_long.getvalue().count("+ ") == 40,
      repr(_pnf_long.getvalue()[-40:]))

# 4d. repl command handlers (Stage A1 extraction): the inline slash-command bodies are
# now module-level handle_*(agent, cfg, arg). Spot-check the pure-ish ones resolve their
# module-level helpers and preserve behavior. CONFIG_PATH is isolated (these autosave).
import io as _io_h, contextlib as _ctx_h
_h_cp = lc.CONFIG_PATH
lc.CONFIG_PATH = Path(tempfile.mkdtemp(prefix="lc_h_")) / "cfg.toml"
try:
    _hc = lc.Config(cwd=FIX, autosave=False)
    with _ctx_h.redirect_stdout(_io_h.StringIO()):
        lc.handle_autosave_command(None, _hc, "on")
    check("handle_autosave on -> True", _hc.autosave is True)
    with _ctx_h.redirect_stdout(_io_h.StringIO()):
        lc.handle_autosave_command(None, _hc, "off")
    check("handle_autosave off -> False", _hc.autosave is False)
    with _ctx_h.redirect_stdout(_io_h.StringIO()):
        lc.handle_autosave_command(None, _hc, "")
    check("handle_autosave no-arg toggles", _hc.autosave is True)
    _hc2 = lc.Config(cwd=FIX, approval="ask")
    with _ctx_h.redirect_stdout(_io_h.StringIO()):
        lc.handle_approve_command(None, _hc2, "auto")
    check("handle_approve sets a valid mode", _hc2.approval == "auto")
    _hb = _io_h.StringIO()
    with _ctx_h.redirect_stdout(_hb):
        lc.handle_help_command(None, lc.Config(cwd=FIX), "")
    check("handle_help renders the command list", "/help" in _hb.getvalue())
    _hag = lc.Agent(lc.Config(cwd=FIX))
    _hcx = _io_h.StringIO()
    with _ctx_h.redirect_stdout(_hcx):
        lc.handle_ctx_command(_hag, _hag.cfg, "")
    check("handle_ctx prints the context meter", len(_hcx.getvalue()) > 0)
finally:
    lc.CONFIG_PATH = _h_cp

# 4e. repl command dispatch table (Stage A2): the elif chain is now a cmd -> handler
# table. Assert every command + alias resolves to a callable, aliases share a handler,
# /quit returns the REPL_EXIT sentinel, and unknown commands miss the table.
_tbl = lc._BUILTIN_COMMANDS_TABLE
check("command table: every entry is callable", all(callable(h) for h in _tbl.values()))
check("command table: quit aliases share one handler",
      _tbl["/quit"] is _tbl["/exit"] is _tbl["/q"] is lc.handle_quit_command)
check("command table: help aliases share one handler",
      _tbl["/help"] is _tbl["/h"] is _tbl["/?"] is lc.handle_help_command)
check("command table: /local and /disconnect share one handler",
      _tbl["/local"] is _tbl["/disconnect"] is lc.handle_local_command)
check("command table: all expected commands present",
      {"/clear", "/new", "/connect", "/tools", "/reload", "/compact", "/handover",
       "/session", "/save", "/load", "/bg", "/autosave", "/incognito", "/leash",
       "/askread", "/think", "/effort", "/provider", "/usage",
       "/info", "/set", "/model", "/approve", "/ctx", "/prompt",
       "/sh"}.issubset(_tbl))
check("command table: /settings renamed to /set (no /settings)", "/settings" not in _tbl)
check("command table: removed rot-prone aliases are gone",
      not any(c in _tbl for c in ("/chat", "/yolo", "/nothink")))
check("command table: unknown command not in table", _tbl.get("/nope") is None)
check("autosave-after set names only settings-persisting commands",
      lc._CMD_AUTOSAVE == frozenset({"/provider", "/providers", "/set"}) and lc._CMD_AUTOSAVE <= set(_tbl))
class _QuitAgent:
    def close_all_remotes(self):
        pass
with _ctx_h.redirect_stdout(_io_h.StringIO()):
    _qret = lc.handle_quit_command(_QuitAgent(), lc.Config(cwd=FIX), "")
check("handle_quit returns REPL_EXIT sentinel", _qret is lc.REPL_EXIT)
with _ctx_h.redirect_stdout(_io_h.StringIO()):
    _hret = lc.handle_help_command(None, lc.Config(cwd=FIX), "")
check("non-exit handler returns None (dispatch continues, not REPL_EXIT)",
      _hret is None and _hret is not lc.REPL_EXIT)

# 4f. dispatch_command (Stage A3): ONE slash-command lookup (builtin table, then the
# lean-tool registry, then unknown). Returns True only on /quit. _BUILTIN_COMMANDS is
# DERIVED from the table (single source of truth - closes the old /disconnect gap).
check("_BUILTIN_COMMANDS derived from the table (single source of truth)",
      lc._BUILTIN_COMMANDS == frozenset(lc._BUILTIN_COMMANDS_TABLE))
check("derived set covers the /disconnect alias (old hand-list gap)",
      "/disconnect" in lc._BUILTIN_COMMANDS)
_da = lc.Agent(lc.Config(cwd=FIX))
with _ctx_h.redirect_stdout(_io_h.StringIO()):
    _dexit = lc.dispatch_command(_da, _da.cfg, "/quit", "")
check("dispatch_command: /quit -> True (exit)", _dexit is True)
with _ctx_h.redirect_stdout(_io_h.StringIO()):
    _dctx = lc.dispatch_command(_da, _da.cfg, "/ctx", "")
check("dispatch_command: a builtin -> False (continue)", _dctx is False)
_dunk = _io_h.StringIO()
with _ctx_h.redirect_stdout(_dunk):
    _druk = lc.dispatch_command(_da, _da.cfg, "/nope", "")
check("dispatch_command: unknown -> False + a notice",
      _druk is False and "unknown command" in _dunk.getvalue())
# a buggy lean-tool handler is caught (loop survives); routes through the registry
lc._lean_tool_commands.clear()
def _boom(agent, cfg, arg):
    raise RuntimeError("kaboom")
lc._lean_tool_commands["/boom"] = (_boom, "explodes")
_dbe = _io_h.StringIO()
with _ctx_h.redirect_stdout(_dbe):
    _dbr = lc.dispatch_command(_da, _da.cfg, "/boom", "")
check("dispatch_command: lean-tool exception is caught, not fatal",
      _dbr is False and "error in /boom" in _dbe.getvalue())
lc._lean_tool_commands.clear()
# a lean-tool routes normally when it doesn't throw
_routed = {}
lc._lean_tool_commands["/route"] = (lambda a, c, arg: _routed.__setitem__("arg", arg), "r")
with _ctx_h.redirect_stdout(_io_h.StringIO()):
    lc.dispatch_command(_da, _da.cfg, "/route", "hi there")
check("dispatch_command: routes a lean-tool command with its arg", _routed.get("arg") == "hi there")
lc._lean_tool_commands.clear()

# tools in auto-approve mode (no prompts)
cfg = lc.Config(cwd=FIX, approval="auto")
tools = lc.Tools(cfg)

# 5. read_file line numbering + truncation notice
(FIX / "big.txt").write_text("\n".join(f"line{i}" for i in range(2000)))
r = tools.read_file("big.txt")
check("read_file numbers + truncates", "\t" in r and "truncated" in r)

# 5b. read_file line range (start/end, 1-based inclusive) + lines=[a,b] alias
(FIX / "rng.txt").write_text("\n".join(f"L{i}" for i in range(1, 21)))  # 20 lines
rr = tools.read_file("rng.txt", start=5, end=7)
check("read_file range returns only L5..L7",
      "5\tL5" in rr and "7\tL7" in rr and "L4" not in rr and "L8" not in rr, repr(rr))
check("read_file accepts lines=[a,b] alias",
      tools.read_file("rng.txt", lines=[5, 7]) == rr)
check("read_file range: start>end errors", "start (7) > end (5)" in tools.read_file("rng.txt", start=7, end=5))
check("read_file range clamps to file end", "20\tL20" in tools.read_file("rng.txt", start=18, end=999))

# 6. write_file (new file)
w = tools.write_file("src/test_new.py", "def t():\n    return 1\n")
check("write_file creates file", (FIX / "src/test_new.py").is_file(), w)

# 7. apply_diff edits existing file
tools.write_file("src/edit.py", "a = 1\nb = 2\n")
d = "<<<<<<< SEARCH\nb = 2\n=======\nb = 99\n>>>>>>> REPLACE"
res = tools.apply_diff("src/edit.py", d)
content = (FIX / "src/edit.py").read_text()
check("apply_diff applies block", "b = 99" in content and "a = 1" in content, res)

# apply_diff with non-matching SEARCH must NOT write
res2 = tools.apply_diff("src/edit.py", "<<<<<<< SEARCH\nZZZ\n=======\nq\n>>>>>>> REPLACE")
check("apply_diff rejects bad SEARCH, no write",
      "not found" in res2 and "b = 99" in (FIX / "src/edit.py").read_text(), res2)

# apply_diff with SEARCH matching multiple times must NOT write (ambiguous)
tools.write_file("src/dup.py", "z = 1\nz = 1\n")
res3 = tools.apply_diff("src/dup.py", "<<<<<<< SEARCH\nz = 1\n=======\nz = 2\n>>>>>>> REPLACE")
check("apply_diff rejects ambiguous SEARCH, no write",
      "ambiguous" in res3 and "matched 2" in res3 and (FIX / "src/dup.py").read_text() == "z = 1\nz = 1\n", res3)

# apply_diff whitespace-tolerant fallback: SEARCH indentation differs from file
tools.write_file("src/fuzz.py", "def f():\n        return 1\n")   # 8-space indent
_fz = "<<<<<<< SEARCH\nreturn 1\n=======\nreturn 42\n>>>>>>> REPLACE"  # no indent
res_fz = tools.apply_diff("src/fuzz.py", _fz)
_fzc = (FIX / "src/fuzz.py").read_text()
check("apply_diff fuzzy-matches on whitespace + re-indents",
      "return 42" in _fzc and _fzc.startswith("def f():\n        return 42"), res_fz + " :: " + repr(_fzc))

# fuzzy still enforces uniqueness: ambiguous whitespace match must NOT write
tools.write_file("src/fuzzdup.py", "  q = 1\n\tq = 1\n")   # same stripped, diff ws
res_fzd = tools.apply_diff("src/fuzzdup.py", "<<<<<<< SEARCH\nq = 1\n=======\nq = 2\n>>>>>>> REPLACE")
check("apply_diff fuzzy rejects ambiguous whitespace match",
      "ambiguous" in res_fzd and "q = 2" not in (FIX / "src/fuzzdup.py").read_text(), res_fzd)

# fuzzy re-indent must PRESERVE the REPLACE block's internal nesting, only shifting
# the whole block to the matched region's indent (regression: it used to strip every
# line to bare + prepend one indent, flattening nested blocks into broken code).
tools.write_file("src/fuzznest.py", "def f():\n  old_a()\n  old_b()\n")   # 2-space region
_fzn = ("<<<<<<< SEARCH\nold_a()\nold_b()\n=======\n"
        "if cond:\n    new_a()\n    new_b()\n>>>>>>> REPLACE")            # nested replace
tools.apply_diff("src/fuzznest.py", _fzn)
_fznc = (FIX / "src/fuzznest.py").read_text()
check("apply_diff fuzzy preserves nested REPLACE indentation",
      _fznc == "def f():\n  if cond:\n      new_a()\n      new_b()\n", repr(_fznc))

# 7b. tool-call line: backgrounded run_command is labelled in TEXT (not just glyph)
check("tool line: bg run_command shows (background) tag",
      "(background)" in lc._tool_call_line("run_command", {"cmd": "x.py &"}))
check("tool line: blocking run_command has no bg tag",
      "(background)" not in lc._tool_call_line("run_command", {"cmd": "ls"}))
check("tool line: non-run_command tool has no bg tag",
      "(background)" not in lc._tool_call_line("read_file", {"path": "a"}))

# 7c. P2: '#N' survives inside the width budget even on a narrow line (never orphans)
_narrow = lc._tool_call_line("read_file", {"path": "x/" + "y" * 400}, 42)
check("tool line: '#N' pinned right, survives long args",
      _narrow.rstrip().endswith("#42"))
check("tool line: '#N' fits one physical line (no wrap)",
      "\n" not in _narrow)
check("tool line: no call_id -> no '#'",
      "#" not in lc._tool_call_line("read_file", {"path": "a"}))

# 7d. P1: WRITE-tier call line is terse (basename only, no content= / full path);
# the result preview carries the authoritative full path.
_wl = lc._tool_call_line("write_file",
                         {"path": "docs/planning/01-strategy.md", "content": "# Hi\n" * 200}, 9)
check("write call line: shows basename", "01-strategy.md" in _wl)
check("write call line: hides full dir path", "docs/planning" not in _wl)
check("write call line: hides content= blob", "content=" not in _wl and "# Hi" not in _wl)
check("write result: keeps full path",
      "docs/planning/01-strategy.md" in lc._tool_result_preview(
          "write_file", "created docs/planning/01-strategy.md (200 lines)", 9))
# non-writer keeps its load-bearing args; blob args collapse to a size hint everywhere
check("read call line: keeps path arg", "path=" in lc._tool_call_line("read_file", {"path": "a.py"}))
check("blob arg collapses to size hint",
      "content=…" in lc._tool_call_line("write_file", {"cmd": "x", "content": "a\nb\nc"}, 0))

# 8. search_files
s = tools.search_files("b = 99")
check("search_files finds match", "src/edit.py:" in s, s.splitlines()[0] if s else "")
check("search_files skips ignored", "node_modules" not in tools.search_files("x"))
# searching a path OUTSIDE cwd must not silently drop matches (relative_to ValueError
# was swallowed by the broad except -> "no matches" even when matches exist)
_outside = Path(tempfile.mkdtemp(prefix="leancoder_outside_"))
(_outside / "note.txt").write_text("hello Maya\nbye\nMaya again\n")
_so = tools.search_files("Maya", str(_outside))
check("search_files finds matches outside cwd (abs path, not dropped)",
      _so.count("Maya") == 2 and _outside.as_posix() in _so, _so.splitlines()[0] if _so else "(none)")
shutil.rmtree(_outside, ignore_errors=True)

# 9. list_files respects ignore
lst = tools.list_files()
check("list_files hides node_modules", "node_modules" not in lst)
check("list_files shows src/", any(l.startswith("src") for l in lst.splitlines()))
# listing an ABSOLUTE path OUTSIDE cwd must yield usable absolute names, not a
# rootlen-sliced mangle (regression: '/tmp/x/g.txt' under cwd '/tmp/y' -> 'g.txt').
_ext = Path(tempfile.mkdtemp(prefix="lc_extlist_"))
(_ext / "deep").mkdir()
(_ext / "top.txt").write_text("x")
(_ext / "deep" / "inner.txt").write_text("y")
_extlst = tools.list_files(str(_ext))
check("list_files outside cwd yields real absolute paths (no mangle)",
      all(l.rstrip("/").startswith(str(_ext)) for l in _extlst.splitlines() if "truncated" not in l)
      and f"{_ext}/top.txt" in _extlst, repr(_extlst))
shutil.rmtree(_ext, ignore_errors=True)

# 10. run_command
rc = tools.run_command("echo hello && exit 0")
check("run_command returns output", "hello" in rc and "exit 0" in rc, rc.replace("\n", " "))

# 10a. trailing '&' backgrounds (detached) instead of hanging on the captured pipe
_orig_cfgdir = lc.CONFIG_DIR
lc.CONFIG_DIR = Path(tempfile.mkdtemp(prefix="lc_bg_"))
bgr = tools.run_command("sleep 30 &")           # would block ~30s/timeout if not detached
check("run_command: trailing & launches in background (pid returned)",
      "background: pid" in bgr and "kill " in bgr, bgr.replace("\n", " "))
check("run_command: background writes a log path under CONFIG_DIR/bg",
      "/bg/" in bgr and (lc.CONFIG_DIR / "bg").is_dir())
import re as _re_bg, os as _os_bg
_m = _re_bg.search(r"pid (\d+)", bgr)
if _m:
    try:
        _os_bg.kill(int(_m.group(1)), 9)        # clean up the detached sleep
    except OSError:
        pass
lc.CONFIG_DIR = _orig_cfgdir

# 10b. _activity_label: watchdog text (plain -> elapsed -> escape hint)
check("_activity_label: below threshold is just the label",
      lc._activity_label("read_file", 1) == "read_file")
check("_activity_label: long-running shows elapsed",
      lc._activity_label("read_file", 10) == "read_file (10s)")
check("_activity_label: very long adds the escape hint + pid",
      "^C to stop" in lc._activity_label("x", 99, 4242) and "kill 4242" in lc._activity_label("x", 99, 4242))
check("_activity_label: empty base falls back to 'working'",
      lc._activity_label("", 1) == "working")

# 10c. background registry: track, enforce max-concurrent, /bg list+kill, session reap
import io as _io_bgr, contextlib as _ctx_bgr
_orig_cd2 = lc.CONFIG_DIR
lc.CONFIG_DIR = Path(tempfile.mkdtemp(prefix="lc_bgreg_"))
_bgcfg = lc.Config(cwd=FIX, approval="auto", bg_max_concurrent=2)
_bgtools = lc.Tools(_bgcfg)
_r1 = _bgtools.run_command("sleep 30 &")
check("bg: first task registered + running",
      "background: pid" in _r1 and len(lc._bg_running()) == 1)
_bgtools.run_command("sleep 30 &")
check("bg: second task under the limit", len(lc._bg_running()) == 2)
_r3 = _bgtools.run_command("sleep 30 &")
check("bg: max_concurrent refuses the 3rd with a clear message",
      "background-task limit" in _r3 and len(lc._bg_running()) == 2)
_b = _io_bgr.StringIO()
with _ctx_bgr.redirect_stdout(_b):
    lc.handle_bg_command(None, _bgcfg, "")            # /bg list
check("/bg lists running tasks", "background tasks:" in _b.getvalue())
with _ctx_bgr.redirect_stdout(_io_bgr.StringIO()):
    lc.handle_bg_command(None, _bgcfg, "kill all")    # /bg kill all
check("/bg kill all stops them", len(lc._bg_running()) == 0)
_bgtools.run_command("sleep 30 &")
check("bg: a task runs before session-kill", len(lc._bg_running()) == 1)
lc._bg_kill_session()                                  # exit hook pegs tasks to the session
check("bg: _bg_kill_session reaps this session's tasks", len(lc._bg_running()) == 0)
# <log>.exit sidecar: a finished task records its exit code in a file (survives a
# reconnect / executor death; works local + remote). /bg then reports pass/fail.
_r_exit = _bgtools.run_command("sh -c 'exit 7' &")
_exlog = _r_exit.split("output -> ", 1)[1].splitlines()[0].strip()
import time as _t_bg
for _ in range(50):
    if Path(_exlog + ".exit").exists(): break
    _t_bg.sleep(0.1)
check("bg: finished task writes exit code to <log>.exit sidecar",
      Path(_exlog + ".exit").read_text().strip() == "7")
_rec7 = {"pid": -1, "cmd": "x", "log": _exlog, "owner": lc.os.getpid()}
check("bg: _bg_exit_code reads the sidecar", lc._bg_exit_code(_rec7) == "7")
_bexit = _io_bgr.StringIO()
with _ctx_bgr.redirect_stdout(_bexit):
    lc.handle_bg_command(None, _bgcfg, "")             # /bg lists finished w/ exit code
check("/bg surfaces a finished task's exit code", "exit 7" in _bexit.getvalue())
lc._bg_clean_sidecars(_rec7)
check("bg: _bg_clean_sidecars removes log + .exit", not Path(_exlog + ".exit").exists())
# bg_status tool: model-facing pid->{state,runtime,tail}, kind-filtered so workers
# never leak onto the plain-bg surface.
check("_fmt_runtime: seconds", lc._fmt_runtime(5) == "5s")
check("_fmt_runtime: minutes", lc._fmt_runtime(192) == "3m12s")
check("_fmt_runtime: hours", lc._fmt_runtime(3840) == "1h04m")
check("_fmt_runtime: bad input -> ?", lc._fmt_runtime(None) == "?")
lc._bg_save([])                                        # clean registry for the status subtests
_bgtools.run_command("sleep 30 &")                     # a live plain task, this session
_st_rows = [r for r in lc._bg_status_items(lc.os.getpid()) if r["state"] == "running"]
check("bg_status: lists this session's live task", len(_st_rows) == 1)
check("bg_status: task carries a runtime", _st_rows[0]["runtime"] != "?")
check("bg_status tool: running task via the Tools method", "running" in _bgtools.bg_status())
# a worker record must NOT appear on the plain-bg surface (a live one: reuse the sleep pid)
_wpid = _st_rows[0]["pid"]
lc._bg_register(_wpid, "lean_coder --agent-run x", _exlog + ".worker", kind="worker")
check("bg_status: worker filtered out of the plain surface",
      all(r["cmd"] != "lean_coder --agent-run x" for r in lc._bg_status_items(lc.os.getpid())))
check("_bg_running: worker excluded from default (task) kind",
      all("agent-run" not in r.get("cmd", "") for r in lc._bg_running()))
check("_bg_running(kind=worker): worker included when asked",
      any("agent-run" in r.get("cmd", "") for r in lc._bg_running(kind="worker")))
check("bg_status: no such pid -> clear message", "no background task with pid" in _bgtools.bg_status(pid=123456))
lc._bg_kill_session()
lc._bg_save([])                                        # drop the fake foreign-pid worker record too
_missmsg = _bgtools.bg_status()
check("bg_status: empty -> plain note", "no background tasks" in _missmsg)
# idle-timeout / lease: a leased task self-terminates when left unattended, and its
# lease file is bumped by _bg_bump_lease (the per-turn heartbeat).
lc._bg_save([])
_lr = _bgtools._run_background("sleep 30", idle_timeout=2)   # 2s lease, never attended
_llog = _lr.split("output -> ", 1)[1].splitlines()[0].strip()
check("lease: launch note advertises self-terminate", "self-terminates if unattended" in _lr)
check("lease: record carries idle_timeout",
      any(r.get("idle_timeout") == 2 for r in lc._bg_load()))
for _ in range(120):                                    # wait out the 2s window + poll cadence
    if Path(_llog + ".exit").exists(): break
    _t_bg.sleep(0.1)
check("lease: unattended task self-terminates (writes exit sidecar)", Path(_llog + ".exit").exists())
check("lease: expiry recorded as exit 137", Path(_llog + ".exit").read_text().strip() == "137")
check("lease: watchdog logged the expiry", "lease expired" in Path(_llog).read_text())
# _bg_bump_lease touches the lease file (attend); no-op without a lease / foreign host
_lrec = {"log": _llog, "idle_timeout": 5, "host": lc.socket.gethostname()}
Path(_llog + ".lease").unlink(missing_ok=True)
lc._bg_bump_lease(_lrec)
check("lease: _bg_bump_lease touches the lease file", Path(_llog + ".lease").exists())
lc._bg_bump_lease({"log": _llog, "idle_timeout": None})     # no lease -> no-op (no raise)
Path(_llog + ".lease").unlink(missing_ok=True)
lc._bg_bump_lease({"log": _llog, "idle_timeout": 5, "host": "some-other-box"})
check("lease: _bg_bump_lease skips a foreign-host record", not Path(_llog + ".lease").exists())
lc._bg_clean_sidecars({"log": _llog})
check("lease: _bg_clean_sidecars removes the .lease too",
      not Path(_llog + ".lease").exists() and not Path(_llog + ".exit").exists())
lc._bg_save([])

# AFK lease-bump regression (the "worker died with no output" bug, commit 93d489b):
# a parent idle at the prompt must keep a leased WORKER alive via bg_wake_turn's idle
# tick, not just per-turn run_turn. Guards the exact seam that once had no test.
lc._bg_save([])
_afklog = str(lc.CONFIG_DIR / "bg" / "afk-worker.log")
Path(_afklog).parent.mkdir(parents=True, exist_ok=True)
Path(_afklog).write_text("")
_afk_lease = Path(_afklog + ".lease"); _afk_lease.write_text("")
_afk_old = _t_bg.time() - 100                                # 100s-stale lease (unattended)
_os_bg.utime(_afk_lease, (_afk_old, _afk_old))
lc._bg_register(424242, "fake worker", _afklog, kind="worker", idle_timeout=1800)
_afk_ag = lc.Agent.__new__(lc.Agent)
_afk_ag.cfg = lc.Config(model="m"); _afk_ag.remote = None; _afk_ag._bg_announced = set()
# 1. a worker-only session arms the idle-wake path (the root-cause miss)
check("afk lease: _has_bg_optins arms for a worker-only session",
      lc.Agent._has_bg_optins(_afk_ag) is True)
# 2. bg_wake_turn bumps the stale lease -> the AFK worker survives (no exit 137)
_afk_before = _afk_lease.stat().st_mtime
lc.Agent.bg_wake_turn(_afk_ag)
_afk_after = _afk_lease.stat().st_mtime
check("afk lease: bg_wake_turn bumps a stale worker lease (survives AFK)",
      _afk_after > _afk_before)
# 3. throttle: a second immediate call does NOT re-bump (~free per idle tick)
_afk_m2 = _afk_lease.stat().st_mtime
lc.Agent.bg_wake_turn(_afk_ag)
check("afk lease: bump throttled within 30s", _afk_lease.stat().st_mtime == _afk_m2)
# 4. past the throttle window it bumps again (steady-state keep-alive)
_afk_ag._last_idle_lease_bump = _t_bg.monotonic() - 31
_t_bg.sleep(0.02)
lc.Agent.bg_wake_turn(_afk_ag)
check("afk lease: bump resumes after the 30s throttle window",
      _afk_lease.stat().st_mtime > _afk_m2)
lc._bg_clean_sidecars({"log": _afklog})
lc._bg_save([])

# 10d. cross-platform bg-runner (run_bg_child / --bg-run). This is the pure-Python
# twin of the POSIX shell wrapper used on Windows (no /bin/sh). Assert it reproduces
# the sidecar contract: plain exit code, and the 137 kill on max_runtime.
import json as _json_bgr, subprocess as _sp_bgr
check("bg: _detached_popen_kwargs posix uses start_new_session",
      lc.os.name == "nt" or lc._detached_popen_kwargs() == {"start_new_session": True})
check("bg: _kill_tree never raises on a dead/absent pid", lc._kill_tree(2 ** 31 - 1) is None)
_bgc_dir = Path(tempfile.mkdtemp(prefix="lc_bgchild_"))
_clog = str(_bgc_dir / "c.log"); Path(_clog).touch()
_ccfg = {"cmd": "exit 5", "exitf": _clog + ".exit", "leasef": _clog + ".lease", "logf": _clog}
_crc = lc.run_bg_child(_ccfg)
check("bg-runner: plain exit code returned + written to sidecar",
      _crc == 5 and Path(_clog + ".exit").read_text().strip() == "5")
_mlog = str(_bgc_dir / "m.log"); Path(_mlog).touch()
_mcfg = {"cmd": "sleep 30", "exitf": _mlog + ".exit", "leasef": _mlog + ".lease",
         "logf": _mlog, "max_runtime": 2, "kill_on_max": True}
_argv = [sys.executable, str(Path(lc.__file__).resolve()), "--bg-run", _json_bgr.dumps(_mcfg)]
_p = _sp_bgr.Popen(_argv, stdout=_sp_bgr.DEVNULL, stderr=_sp_bgr.DEVNULL,
                   stdin=_sp_bgr.DEVNULL, **lc._detached_popen_kwargs())
_p.wait()
check("bg-runner: --bg-run entrypoint enforces max_runtime (exit 137 sidecar)",
      Path(_mlog + ".exit").read_text().strip() == "137")
shutil.rmtree(_bgc_dir, ignore_errors=True)

# 10e. self-wipe watchdog (run_wipe_watchdog / --wipe-watch): the cross-platform twin
# of the old POSIX sh loop. A stale lease -> the pushed dir is rmtree'd; a bumped lease
# self-heals (no wipe). Uses the real detached --wipe-watch subprocess.
_ww_dir = Path(tempfile.mkdtemp(prefix="lc_wipe_")) / "leancoder"
_ww_dir.mkdir(parents=True); (_ww_dir / ".lease").touch(); (_ww_dir / "agent.py").touch()
_wwargv = [sys.executable, str(Path(lc.__file__).resolve()), "--wipe-watch",
           _json_bgr.dumps({"dir": str(_ww_dir), "ttl": 2, "chk": 1})]
_wwp = _sp_bgr.Popen(_wwargv, stdin=_sp_bgr.DEVNULL, stdout=_sp_bgr.DEVNULL,
                     stderr=_sp_bgr.DEVNULL, **lc._detached_popen_kwargs())
for _ in range(80):                                     # stale lease -> wipe within TTL+poll
    if not _ww_dir.is_dir(): break
    _t_bg.sleep(0.1)
check("wipe-watch: stale lease -> pushed dir is wiped", not _ww_dir.is_dir())

# 10f. cross-platform ssh transport groundwork: OS-detect + multiplex-aware argv.
# Windows OpenSSH has NO ControlMaster, so with no ctl the argv builders must drop
# the ControlPath option; POSIX (ctl set) stays byte-identical. _remote_is_posix
# classifies the `uname` probe output.
check("_remote_is_posix: Linux -> True", lc._remote_is_posix("Linux"))
check("_remote_is_posix: Darwin -> True", lc._remote_is_posix("Darwin"))
check("_remote_is_posix: empty -> False (Windows)", not lc._remote_is_posix(""))
check("_remote_is_posix: cmd error -> False",
      not lc._remote_is_posix("'uname' is not recognized as an internal or external command"))
check("_remote_is_posix: None -> False", not lc._remote_is_posix(None))
check("_ctl_opts: with ctl -> ControlPath", lc._ctl_opts("/x") == ["-o", "ControlPath=/x"])
check("_ctl_opts: no ctl -> empty (no multiplexing)", lc._ctl_opts(None) == [])
check("_ssh_run_argv: posix carries ControlPath",
      "ControlPath=/x" in " ".join(lc._ssh_run_argv("h", "/x", "cmd")))
check("_ssh_run_argv: windows (no ctl) drops ControlPath",
      "ControlPath" not in " ".join(lc._ssh_run_argv("h", None, "cmd")))
check("_ssh_exec_argv: windows (no ctl) drops ControlPath, keeps --tool-exec",
      "ControlPath" not in " ".join(lc._ssh_exec_argv("h", None, "python agent.py", "."))
      and "--tool-exec" in " ".join(lc._ssh_exec_argv("h", None, "python agent.py", ".")))
check("_ssh_master_alive: no ctl -> False (no master to check)",
      lc._ssh_master_alive("h", None) is False)
# --- worker engine (--agent-run): brief/grant parsing + result harvest --------
# The full headless run is hand-tested (needs a live provider); here we cover the
# pure logic that shapes a worker run so a regression is caught offline.
_brief_raw = (f"lead-in\n{lc.BRIEF_MARK}\nDo the thing.\n{lc.BRIEF_MARK}\n"
              f"{lc.GRANT_MARK}\nleash: r\nmodel: some-model\ncwd: /tmp\n"
              f"max_iterations: 12\nbudget_tokens: 5000\n{lc.GRANT_MARK}\n")
check("agent-run: BRIEF block extracted", lc._extract_marked(_brief_raw, lc.BRIEF_MARK) == "Do the thing.")
_g = lc._parse_grant(lc._extract_marked(_brief_raw, lc.GRANT_MARK))
check("agent-run: grant parses leash", _g.get("leash") == "r")
check("agent-run: grant parses model", _g.get("model") == "some-model")
check("agent-run: grant parses cwd", _g.get("cwd") == "/tmp")
check("agent-run: grant parses max_iterations", _g.get("max_iterations") == "12")
check("agent-run: grant ignores junk lines", "lead-in" not in _g and len(_g) == 5)
check("agent-run: _parse_grant on empty -> {}", lc._parse_grant("") == {})
check("agent-run: result path defaults to <brief>.result",
      lc._brief_result_path("/tmp/b.txt", None) == "/tmp/b.txt.result")
check("agent-run: result path honours explicit --result-file",
      lc._brief_result_path("/tmp/b.txt", "/tmp/out") == "/tmp/out")
# harvest: a RESULT block wins; a bare answer falls back to the whole message
check("agent-run: RESULT block harvested",
      lc._extract_marked(f"prose\n{lc.RESULT_MARK}\nthe answer\n{lc.RESULT_MARK}", lc.RESULT_MARK) == "the answer")
check("agent-run: no RESULT block -> None (caller falls back to full text)",
      lc._extract_marked("just a plain answer, no markers", lc.RESULT_MARK) is None)
# _extract_marked XML-close fallback: a model that opens with the fence but closes
# with an XML-style </tag> (a common small-model slip) must NOT lose its block.
check("_extract_marked: strict fence pair still wins",
      lc._extract_marked(f"{lc.HANDOVER_MARK}\nsummary\n{lc.HANDOVER_MARK}", lc.HANDOVER_MARK) == "summary")
check("_extract_marked: fence + <tag>..</tag> XML-close rescued",
      lc._extract_marked(f"{lc.HANDOVER_MARK}\n<handover>\nsummary text\n</handover>", lc.HANDOVER_MARK) == "summary text")
check("_extract_marked: fence + bare </tag> close rescued",
      lc._extract_marked(f"{lc.HANDOVER_MARK}\nbody only\n</HANDOVER>", lc.HANDOVER_MARK) == "body only")
check("_extract_marked: lone fence, no close -> None (safe, no session nuke)",
      lc._extract_marked(f"{lc.HANDOVER_MARK}\ndangling text", lc.HANDOVER_MARK) is None)
# _parse_handover: tolerant multi-tier parse (canonical -> md -> json -> prose).
_ph1 = lc._parse_handover(f"{lc.HANDOVER_MARK}\nfixed add() in calc.py, verified\n{lc.HANDOVER_MARK}\n"
                          f"{lc.PLAN_MARK}\nGOAL: x\nTODO:\n- [ ] test\n{lc.PLAN_MARK}\n"
                          f"{lc.SELFPROMPT_MARK}\nwrite the test\n{lc.SELFPROMPT_MARK}")
check("_parse_handover: tier1 canonical fences", _ph1["tier"] == 1 and _ph1["handover"] and _ph1["plan"] and _ph1["next"])
_ph3 = lc._parse_handover("## Handover\nfixed add() in calc.py and verified it prints 5\n## Plan\nGOAL: t\n## Next\nwrite unit test")
check("_parse_handover: tier3 markdown headers", _ph3["tier"] == 3 and "calc.py" in _ph3["handover"] and _ph3["next"] == "write unit test")
_ph4 = lc._parse_handover('{"summary": "fixed the add bug in calc.py and verified output", "todo": ["write test"], "next": "add the unit test"}')
check("_parse_handover: tier4 json fuzzy fields", _ph4["tier"] == 4 and "calc.py" in _ph4["handover"] and _ph4["next"] == "add the unit test")
_ph5 = lc._parse_handover("I fixed the subtraction bug in calc.py by changing minus to plus, ran it, got 5. Next add a test.")
check("_parse_handover: tier5 prose salvage", _ph5["tier"] == 5 and _ph5["handover"])
check("_parse_handover: empty content -> no handover (safe no-op)", lc._parse_handover("ok")["handover"] is None)

# --- bg hooks for lean-tools (bg_launch/bg_status/bg_list on the lc dict) ------
check("bg hooks: bg_launch is exposed", callable(lc.bg_launch))
check("bg hooks: bg_status is _bg_status_items", lc.bg_status is lc._bg_status_items)
check("bg hooks: bg_list is _bg_running", lc.bg_list is lc._bg_running)
lc._bg_save([])
_hl = lc.bg_launch("sleep 30", cwd=str(FIX), kind="worker", idle_timeout=60)
check("bg_launch: returns pid+log+exitf+leasef",
      all(k in _hl for k in ("pid", "log", "exitf", "leasef")) and "error" not in _hl)
check("bg_launch: registered as kind=worker",
      any(r.get("pid") == _hl["pid"] and r.get("kind") == "worker" for r in lc._bg_load()))
check("bg_launch: carries the lease (idle_timeout)",
      any(r.get("pid") == _hl["pid"] and r.get("idle_timeout") == 60 for r in lc._bg_load()))
check("bg_list(kind=worker) sees the launched worker",
      any(r.get("pid") == _hl["pid"] for r in lc.bg_list(kind="worker")))
check("bg_list default (task) excludes the worker",
      all(r.get("pid") != _hl["pid"] for r in lc.bg_list()))
check("bg_launch: empty cmd -> error", "error" in lc.bg_launch(""))
lc._bg_kill(_hl["pid"]); lc._bg_save([])

# --- dispatch_worker lean-tool: brief compose + ceiling + driver_only ---------
import os as _os_dw
_dw = lc._load_lean_tool(Path(__file__).resolve().parent.parent / "lean-tools" / "dispatch_worker.py")
check("dispatch_worker: is a driver_only tool", _dw.TOOL.get("driver_only") is True)
check("dispatch_worker: not marked safe (dispatch spends quota)", not _dw.TOOL.get("safe"))
_dwcfg = lc.Config(cwd=FIX)
_dw.setup(lc.__dict__, _dwcfg)
check("dispatch_worker: setup captured bg_launch hook", "bg_launch" in _dw._H)
check("dispatch_worker: setup captured core file for the worker argv",
      _dw._H.get("__file__", "").endswith("lean_coder.py"))

# --- shell_session _drain: the echo-race fix. A pty echoes the typed line back
# within ms; the command's real output can lag (ssh RTT / slow remote). _drain must
# NOT let that instant echo trip the quiet-gap early-exit and return only the echo
# (the bug: output landed a call late / "no output"). _beyond_echo is the pure core.
_ss = lc._load_lean_tool(Path(__file__).resolve().parent.parent / "lean-tools" / "shell_session.py")
check("shell_session: is driver_only (pty fd lives on the driver)",
      _ss.TOOL.get("driver_only") is True)
check("_beyond_echo: bare echo alone is NOT real output",
      _ss._beyond_echo(b"qm list\n", b"qm list\n") is False)
check("_beyond_echo: echo + CR variations still counted as echo-only",
      _ss._beyond_echo(b"qm list\r\n", b"qm list\n") is False)
check("_beyond_echo: output past the echo IS real",
      _ss._beyond_echo(b"qm list\nVMID NAME\n", b"qm list\n") is True)
check("_beyond_echo: no echo passed -> any non-space is real (bare read)",
      _ss._beyond_echo(b"data\n", b"") is True and _ss._beyond_echo(b"  \n", b"") is False)
# --- shell_session sentinel-marker draining (deterministic completion) ----------
# _mode_for: shells/ssh get marker mode; REPLs/listeners/unknown get raw.
check("_mode_for: bare shell -> shell", _ss._mode_for("") == "shell")
check("_mode_for: ssh -> shell (far end is a login shell)", _ss._mode_for("ssh user@host") == "shell")
check("_mode_for: sudo/su -> shell", _ss._mode_for("sudo -s") == "shell")
check("_mode_for: python -i -> raw", _ss._mode_for("python3 -i") == "raw")
check("_mode_for: nc listener -> raw", _ss._mode_for("nc -lvnp 4444") == "raw")
check("_mode_for: unknown command -> raw (never inject a marker blindly)",
      _ss._mode_for("some-weird-repl") == "raw")
# _strip_marker: output is exactly the lines BETWEEN the echoed injected line and the
# <mark><rc> terminator - so echo AND trailing prompt both drop.
_MK = "__LC_DONE_1_2_"
_cap = ("echo hi; printf '\\n%s' \"$?\"\n"   # echo of our injected line (has mark)
        "hi\n"                                # the real output
        f"{_MK}0\n"                           # the terminator
        "user@box:~$ ")                       # next prompt (must be dropped)
_cap = _cap.replace("%s", _MK)
check("_strip_marker: keeps only output between echo and terminator",
      _ss._strip_marker(_cap, _MK).strip() == "hi")
check("_strip_marker: fallback filters mark lines when anchors missing",
      _ss._strip_marker(f"just output\n{_MK}0", _MK).strip() == "just output")
# End-to-end through a real pty (POSIX only): the cases that broke before -
# lagged/bursty output, exit code surfaced, clean-exit stays quiet, prompt/echo
# never leak. Guarded so a no-pty platform just skips.
try:
    import pty as _pty_probe  # noqa: F401
    _has_pty = True
except ImportError:
    _has_pty = False
if _has_pty:
    _ss._H["_clean_captured"] = getattr(lc, "_clean_captured", lambda t: t)
    def _sss(inp, timeout=6):
        return _ss.run({"action": "send", "id": _sid, "input": inp, "timeout": timeout}, "/tmp")
    _startres = _ss._start({}, "/tmp")
    _sid = _startres.split()[1]              # "session sN started ..."
    # bursty: output split by a mid-command sleep longer than the quiet-gap
    _r = _sss("echo A; sleep 1; echo B_LATE")
    check("shell_session e2e: bursty output fully captured (marker drain)",
          "A" in _r and "B_LATE" in _r and "__LC_DONE" not in _r, repr(_r))
    # exit code surfaced on failure, quiet on clean success
    check("shell_session e2e: non-zero exit surfaced", "[exit 1]" in _sss("false"), )
    check("shell_session e2e: clean exit stays quiet (no [exit 0])",
          "[exit" not in _sss("true"))
    # no prompt / echo / marker leakage
    _r = _sss("echo HELLO")
    check("shell_session e2e: output is clean (no prompt/echo/marker leak)",
          _r.strip() == "HELLO", repr(_r))
    # state persists across sends
    _sss("cd /etc && export _LCT=zzz")
    check("shell_session e2e: cwd + env persist across sends",
          _ss.run({"action": "send", "id": _sid, "input": 'echo "$PWD/$_LCT"'}, "/tmp").strip()
          == "/etc/zzz")
    # timeout < command -> 'still running', rest retrievable via read
    _r = _sss("sleep 3; echo TARDY", timeout=1)
    check("shell_session e2e: timeout reports still-running (not a false empty)",
          "still running" in _r)
    check("shell_session e2e: the late output is retrievable via read",
          "TARDY" in _ss.run({"action": "read", "id": _sid, "timeout": 4}, "/tmp"))
    _ss.run({"action": "close", "id": _sid}, "/tmp")

# --- web_screenshot lean-tool: MUST be driver_only ----------------------------
# The browser + playwright live on the driver; if this tool is pushed to a remote
# executor it tries to apt-install pip+playwright+a browser onto the user's box.
_ws_path = Path(__file__).resolve().parent.parent / "lean-tools" / "web_screenshot.py"
if _ws_path.is_file():
    _ws = lc._load_lean_tool(_ws_path)
    check("web_screenshot: is a driver_only tool (never pushed to remote)",
          _ws.TOOL.get("driver_only") is True)
_brief = _dw._compose_brief("scout the module", "some-model", "/tmp", 20)
check("dispatch_worker: brief has BRIEF + GRANT blocks",
      lc._extract_marked(_brief, lc.BRIEF_MARK) == "scout the module"
      and "leash: r" in lc._extract_marked(_brief, lc.GRANT_MARK))
check("dispatch_worker: Phase-1 leash is read-only in the grant",
      "leash: r\n" in _brief and "leash: rw" not in _brief)
# ceiling: env override wins over the default
_os_dw.environ.pop("LEANCODER_WORKER_MAX_CONCURRENT", None)
check("dispatch_worker: default max_concurrent", _dw._ceiling("max_concurrent") == 10)
_os_dw.environ["LEANCODER_WORKER_MAX_CONCURRENT"] = "4"
check("dispatch_worker: env overrides max_concurrent", _dw._ceiling("max_concurrent") == 4)
_os_dw.environ.pop("LEANCODER_WORKER_MAX_CONCURRENT", None)
_os_dw.environ.pop("LEANCODER_WORKER_MODELS", None)
check("dispatch_worker: no allowlist -> None", _dw._model_allowlist() is None)
_os_dw.environ["LEANCODER_WORKER_MODELS"] = "model-x, model-y"
check("dispatch_worker: allowlist parsed", _dw._model_allowlist() == ["model-x", "model-y"])
check("dispatch_worker: empty task refused", "error" in _dw.run({"task": "  "}, str(FIX)))
check("dispatch_worker: model outside allowlist refused",
      "not in the worker allowlist" in _dw.run({"task": "x", "model": "banned"}, str(FIX)))
_os_dw.environ.pop("LEANCODER_WORKER_MODELS", None)
# Phase 2: leash cap = MIN(requested, parent's live leash) - never exceeds the parent
_dwcfg.leash = "rwe"
check("dispatch_worker: parent rwe grants requested rw", _dw._capped_leash("rw") == ("rw", False))
check("dispatch_worker: parent rwe grants rwe", _dw._capped_leash("rwe") == ("rwe", False))
check("dispatch_worker: default request is read-only", _dw._capped_leash("r") == ("r", False))
_dwcfg.leash = "r"
check("dispatch_worker: parent r caps rwe -> r (downgraded)", _dw._capped_leash("rwe") == ("r", True))
check("dispatch_worker: parent r caps rw -> r (downgraded)", _dw._capped_leash("rw") == ("r", True))
_dwcfg.leash = "rw"
check("dispatch_worker: parent rw caps rwe -> rw", _dw._capped_leash("rwe") == ("rw", True))
_dwcfg.leash = "rwe"
check("dispatch_worker: chat request coerced to r (no-tool worker is pointless)",
      _dw._capped_leash("chat") == ("r", False))
check("dispatch_worker: junk leash -> r", _dw._capped_leash("xyz") == ("r", False))
check("dispatch_worker: capped leash flows into the GRANT block",
      "leash: rw\n" in _dw._compose_brief("t", "", "/tmp", 10, leash="rw"))
_dwcfg.leash = "rwe"
# Phase 3: finish-notice - fires once per worker, 'all done' when the last clears
_dwtmp = Path(tempfile.mkdtemp(prefix="lc_wnotify_"))
_r1p, _r2p = _dwtmp / "w1.result", _dwtmp / "w2.result"
_dw._H["workers"] = {
    101: {"result": str(_r1p), "task": "scout A", "model": "m", "leash": "r"},
    102: {"result": str(_r2p), "task": "scout B", "model": "m", "leash": "r"},
}
# Both procs stay alive (a present bg_status row) for the whole success run, so finish
# is signaled purely by the result file appearing - not by the proc-gone FAILED path.
# (_worker_rows() reads _H["bg_status"]; without this it returns {} and every result-less
# worker is wrongly declared FAILED on the first scan.)
_dw_bg_status_real = _dw._H.get("bg_status")
_dw._H["bg_status"] = lambda owner, kind=None: [{"pid": 101}, {"pid": 102}]
check("finish-notice: nothing finished -> empty", _dw._finished_notice() == "")
_r1p.write_text(f"{lc.RESULT_MARK}\nA is here\n{lc.RESULT_MARK}\n")
_n1 = _dw._finished_notice()
check("finish-notice: reports the finished worker", "worker 101 finished" in _n1 and "A is here" in _n1)
check("finish-notice: does NOT announce all-done while one is outstanding",
      "all 2" not in _n1)
check("finish-notice: dedups (already-announced worker not repeated)", _dw._finished_notice() == "")
_r2p.write_text(f"{lc.RESULT_MARK}\nB is here\n{lc.RESULT_MARK}\n")
_n2 = _dw._finished_notice()
check("finish-notice: reports the second worker", "worker 102 finished" in _n2)
check("finish-notice: announces all-done when the last clears", "all 2 dispatched workers" in _n2)
# FAILED path: a worker with no result whose proc is GONE (no bg_status row) is announced
# as failed, not left in limbo (regression guard for the silent-limbo bug).
_r3p = _dwtmp / "w3.result"
_dw._H["workers"] = {103: {"result": str(_r3p), "task": "scout C", "model": "m", "leash": "r"}}
_dw._H["bg_status"] = lambda owner, kind=None: []   # proc gone, no result written
_n3 = _dw._finished_notice()
check("finish-notice: dead worker with no result -> FAILED", "worker 103 FAILED" in _n3)
_dw._H["bg_status"] = _dw_bg_status_real
_dw._H["workers"] = {}
shutil.rmtree(_dwtmp, ignore_errors=True)
lc._bg_save([])                                        # tidy the registry for later checks
# Phase 2: host-gating + adopt-on-reconnect. A foreign-host record is never ours
# to probe/reap; same-host trusts the pid; no host => local (back-compat).
_HN = lc.socket.gethostname()
check("bg: _bg_here true for this host + for a host-less (legacy) record",
      lc._bg_here({"host": _HN}) and lc._bg_here({}))
check("bg: _bg_here false for a foreign host", not lc._bg_here({"host": _HN + "-elsewhere"}))
# foreign task with no exit sidecar => assumed alive; with one => finished.
check("bg: _bg_alive foreign infers from sidecar (no sidecar -> alive)",
      lc._bg_alive({"host": "otherbox", "pid": 999999, "log": "/nope/x.log"}))
# adopt: a survived same-host task owned by a DEAD prior executor is re-parented,
# not killed. Simulate with a real detached sleep and a bogus dead owner.
_r_ad = _bgtools.run_command("sleep 30 &")
_adlog = _r_ad.split("output -> ", 1)[1].splitlines()[0].strip()
_recs = lc._bg_load()
for _r in _recs:
    if _r.get("log") == _adlog:
        _r["owner"] = 999999                           # a dead owner (survived a reconnect)
lc._bg_save(_recs)
lc._bg_adopt()                                         # executor startup re-parents it
_after = [r for r in lc._bg_load() if r.get("log") == _adlog]
check("bg: _bg_adopt re-parents a survivor to this pid (not killed)",
      len(_after) == 1 and _after[0].get("owner") == lc.os.getpid()
      and lc._proc_alive(_after[0].get("pid")))
lc._bg_kill_session(); lc._bg_save([])                 # cleanup
# Phase 3: long-run notify. A this-session task that finished since last turn is
# surfaced ONCE (cmd, exit code, log tail) via _bg_finished_note; reported again
# is suppressed. _bg_log_tail returns trailing lines.
_bgpag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_bgpag.tools = _bgtools                                 # share the launcher's Tools
_r_fin = _bgtools.run_command("sh -c 'echo line-a; echo line-b; exit 5' &")
_finlog = _r_fin.split("output -> ", 1)[1].splitlines()[0].strip()
for _ in range(50):
    if Path(_finlog + ".exit").exists(): break
    _t_bg.sleep(0.1)
check("bg: _bg_log_tail returns trailing log lines",
      "line-b" in lc._bg_log_tail(_finlog, 20))
_note1 = _bgpag._bg_finished_note()
check("bg: _bg_finished_note reports a finished task once (cmd + exit + tail)",
      "exited 5" in _note1 and "line-b" in _note1)
check("bg: _bg_finished_note suppresses an already-announced task",
      _bgpag._bg_finished_note() == "")
# a /connected (remote) session polls the executor (BG_POLL) instead of scanning
# local sidecars: a finished REMOTE task surfaces once, then is suppressed.
class _BgRemote:
    host = "remotebox"
    def __init__(self): self._polls = 0
    def bg_poll(self):
        self._polls += 1
        return [{"pid": 4242, "cmd": "make", "code": "0", "tail": "done-remote"}] if self._polls == 1 else []
_rpag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_rpag.remote = _BgRemote()
_rnote = _rpag._bg_finished_note()
check("bg: remote _bg_finished_note reports a polled task (cmd + exit + tail)",
      "make" in _rnote and "exited 0" in _rnote and "done-remote" in _rnote)
check("bg: remote finished task announced once (dedup by host,pid)",
      _rpag._bg_finished_note() == "")
lc._bg_kill_session(); lc._bg_save([])                 # cleanup
_tcf = Path(tempfile.mktemp(suffix=".toml")); _ocp_bg = lc.CONFIG_PATH
lc.CONFIG_PATH = _tcf
lc.save_config(lc.Config(model="m", command_timeout=120, bg_max_concurrent=3), quiet=True)
import tomllib as _tt_bg
_rtc = _tt_bg.loads(_tcf.read_text())
check("save_config: command_timeout round-trip", _rtc.get("command_timeout") == 120)
check("save_config: bg_max_concurrent round-trip", _rtc.get("bg_max_concurrent") == 3)
# non-default context knobs must round-trip (a guard referencing a stale default
# silently drops the user's value - bit us when window default went 30 -> 0)
lc.save_config(lc.Config(model="m", window_messages=50, handover_soft=0.4, handover_hard=0.6),
               quiet=True)
_rtw = _tt_bg.loads(_tcf.read_text())
check("save_config: window_messages round-trip", _rtw.get("window_messages") == 50)
check("save_config: handover_soft/hard round-trip",
      _rtw.get("handover_soft") == 0.4 and _rtw.get("handover_hard") == 0.6)
# the default value is NOT written (stays default on reload)
lc.save_config(lc.Config(model="m"), quiet=True)
check("save_config: default window_messages omitted",
      "window_messages" not in _tt_bg.loads(_tcf.read_text()))
# statusline_every: persists when set, omitted at the default 0, and reloads via load_config
lc.save_config(lc.Config(model="m", statusline_every=5), quiet=True)
check("save_config: statusline_every round-trip", _tt_bg.loads(_tcf.read_text()).get("statusline_every") == 5)
lc.save_config(lc.Config(model="m", statusline_every=0), quiet=True)   # 0 = on-change, non-default
check("save_config: non-default statusline_every=0 persisted",
      _tt_bg.loads(_tcf.read_text()).get("statusline_every") == 0)
lc.save_config(lc.Config(model="m"), quiet=True)
check("save_config: default statusline_every (1) omitted",
      "statusline_every" not in _tt_bg.loads(_tcf.read_text()))
check("Config default statusline_every is 1 (every turn)", lc.Config().statusline_every == 1)
# statusline_iter: default 0 (off), persists when set, omitted at default
check("Config default statusline_iter is 0 (off)", lc.Config().statusline_iter == 0)
lc.save_config(lc.Config(model="m", statusline_iter=30), quiet=True)
check("save_config: statusline_iter round-trip", _tt_bg.loads(_tcf.read_text()).get("statusline_iter") == 30)
lc.save_config(lc.Config(model="m", statusline_iter=0), quiet=True)
check("save_config: default statusline_iter (0) omitted",
      "statusline_iter" not in _tt_bg.loads(_tcf.read_text()))
# --- config drift guard: every Config field is classified exactly once, and every
# PERSISTED scalar actually round-trips through save -> load. This is the insurance
# against the load/save mirror silently dropping a knob (bit us: auto_update /
# update_track were loadable + /set-able but never saved). If you add a Config field,
# you MUST add it to _PERSISTED_SCALAR_KEYS, _EPHEMERAL_KEYS, or _BESPOKE_KEYS - this
# fails until you do, so the classification can't rot.
import dataclasses as _dc
_all_fields = set(f.name for f in _dc.fields(lc.Config))
_classified = set(lc._PERSISTED_SCALAR_KEYS) | lc._EPHEMERAL_KEYS | lc._BESPOKE_KEYS
check("config: every Config field is classified (persist/ephemeral/bespoke)",
      _all_fields == _classified,
      f"unclassified={sorted(_all_fields - _classified)} stale={sorted(_classified - _all_fields)}")
# Each persisted scalar, set to a non-default, must come back through load_config.
_nondefault = {
    "num_ctx": 8192, "max_iterations": 7, "keep_alive": "30m", "temperature": 0.33,
    "top_p": 0.55, "top_k": 42, "repeat_penalty": 1.11, "ask_user_to_run": False,
    "autosave": False, "auto_update": True, "update_track": "beta", "command_timeout": 99,
    "bg_max_concurrent": 2, "worker_max_concurrent": 4, "worker_idle_timeout": 900,
    "worker_max_iterations": 15, "editor": "vim", "approval": "auto", "confirm_reads": True,
    "auto_reconnect": True, "statusline": False, "statusline_every": 3, "statusline_iter": 20,
    "auto_evict": True, "auto_evict_keep": 5, "window_messages": 12, "window_tokens": 4000,
    "auto_handover": False,
    "handover_soft": 0.42, "handover_hard": 0.61, "handover_emergency": 0.99,
    "handover_min_interval": 33.0, "autostart_after_handover": False,
    "auto_compact_interval": 8, "auto_compact_hysteresis": 0.4, "auto_compact_keep": 6,
    "wake_on_bg_finish": True, "notes_spool": 500, "lean_tools_dir": "/tmp/lt", "providers_dir": "/tmp/pv",
    "user_name": "dide", "model": "some-model:1b", "ephemeral": True,
    "gen_connect_timeout": 7.0, "gen_ttft_timeout": 120.0, "gen_idle_timeout": 30.0,
}
# every persisted scalar must have a probe value here (so nothing is left untested)
_missing_probe = [k for k in lc._PERSISTED_SCALAR_KEYS if k not in _nondefault]
check("config: round-trip probe covers every persisted scalar", not _missing_probe,
      f"no probe value for {_missing_probe}")
class _Args:  # minimal args stub for load_config (no CLI overrides)
    def __getattr__(self, _n): return None
_cfg_rt = lc.Config(**{k: _nondefault[k] for k in lc._PERSISTED_SCALAR_KEYS})
lc.save_config(_cfg_rt, quiet=True)
_ocp_rt = lc.CONFIG_PATH  # load_config reads CONFIG_PATH (already the scratch _tcf here)
_loaded = lc.load_config(_Args())
_rt_bad = [k for k in lc._PERSISTED_SCALAR_KEYS
           if k != "model" and getattr(_loaded, k) != _nondefault[k]]
check("config: every persisted scalar round-trips save -> load", not _rt_bad,
      f"dropped/changed: {_rt_bad}")

# 10d. queue-drain 3-way choice (default combine)
check("_drain_choice: s -> separate", lc._drain_choice("s") == "separate"
      and lc._drain_choice("separate") == "separate")
check("_drain_choice: d/n -> discard", lc._drain_choice("d") == "discard"
      and lc._drain_choice("n") == "discard")
check("_drain_choice: default (Enter / c / junk) -> combine",
      lc._drain_choice("") == "combine" and lc._drain_choice("c") == "combine"
      and lc._drain_choice("zzz") == "combine")
# text-tool-call schema coercion: the Qwen XML channel stringifies every arg, so a
# declared boolean/integer must be coerced back (else bool('false') is truthy -> a
# notify_on_exit=false silently becomes ON). string params must NOT be coerced.
_tt_schemas = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in lc.TOOLS}
_tt_xml = ("<function=run_command><parameter=cmd>sleep 5 &</parameter>"
           "<parameter=notify_on_exit>false</parameter>"
           "<parameter=heartbeat_timeout>30</parameter></function>")
_tt_calls, _ = lc.parse_text_tool_calls(_tt_xml, known_names={"run_command"}, schemas=_tt_schemas)
_tt_a = _tt_calls[0]["function"]["arguments"]
check("text-tool-call: XML bool 'false' coerced to real False",
      _tt_a.get("notify_on_exit") is False, repr(_tt_a))
check("text-tool-call: XML integer arg coerced to int", _tt_a.get("heartbeat_timeout") == 30)
_tt_wf, _ = lc.parse_text_tool_calls(
    "<function=write_file><parameter=path>x</parameter><parameter=content>true</parameter></function>",
    known_names={"write_file"}, schemas=_tt_schemas)
check("text-tool-call: string param NEVER coerced (content stays 'true')",
      _tt_wf[0]["function"]["arguments"].get("content") == "true")
_tt_np, _ = lc.parse_text_tool_calls(_tt_xml, known_names={"run_command"})  # no schemas
check("text-tool-call: no-schema pass-through unchanged (back-compat)",
      _tt_np[0]["function"]["arguments"].get("notify_on_exit") == "false")
# mid-turn slash commands are peeled out (run as commands, never sent to the model)
_qc, _qm = lc._split_queued_commands(["/help", "fix the bug", "  /tools", "run /help please"])
check("_split_queued_commands: commands peeled (incl. leading-space)",
      _qc == ["/help", "  /tools"])
check("_split_queued_commands: only true messages remain (mid-sentence / stays text)",
      _qm == ["fix the bug", "run /help please"])
check("_split_queued_commands: order preserved, no overlap",
      lc._split_queued_commands(["a", "/x", "b", "/y"]) == (["/x", "/y"], ["a", "b"]))
shutil.rmtree(lc.CONFIG_DIR, ignore_errors=True)
lc.CONFIG_DIR = _orig_cd2

# 10b. ssh moved to an opt-in lean-tool (lean-tools/ssh.py): run() argv + shape
sshp = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/ssh.py")
class _CP:
    returncode = 0; stdout = "Linux box 6.1 x86_64\n"; stderr = ""
scap = {}
_orig_run = sshp.subprocess.run
sshp.subprocess.run = lambda argv, **kw: (scap.update(argv=argv, kw=kw) or _CP())
try:
    sres = sshp.run({"host": "user@host", "cmd": "uname -a"}, str(FIX))
finally:
    sshp.subprocess.run = _orig_run
sargv = scap.get("argv", [])
check("ssh lean-tool: argv list, shell=False", scap["kw"].get("shell") is False)
check("ssh lean-tool: BatchMode=yes (non-interactive)", "BatchMode=yes" in sargv)
check("ssh lean-tool: ConnectTimeout + accept-new",
      "ConnectTimeout=10" in sargv and "StrictHostKeyChecking=accept-new" in sargv)
check("ssh lean-tool: host then cmd are the final args",
      sargv[-2:] == ["user@host", "uname -a"], str(sargv))
check("ssh lean-tool: timeout=300", scap["kw"].get("timeout") == 300)
check("ssh lean-tool: 'exit N' shaped result",
      sres.startswith("exit 0") and "Linux box" in sres, sres.replace("\n", " "))
check("ssh lean-tool: missing host/cmd -> error", "needs both" in sshp.run({"host": "h"}, str(FIX)))
check("ssh lean-tool is not safe (goes through the confirm gate)", not sshp.TOOL.get("safe"))

# 11. host normalization (full base URL, no hardcoded localhost)
class A:  # fake args
    host = "myserver:11434"; model = None; num_ctx = None; cwd = str(FIX)
    approval = "ask"; temperature = None; top_p = None; top_k = None
    repeat_penalty = None; think = None
import os, tempfile as _tfA, pathlib as _plA
os.environ.pop("OLLAMA_HOST", None); os.environ.pop("LEANCODER_MODEL", None)
# Isolate from the machine's real config.toml: load_config reads CONFIG_PATH, and
# a box that pins num_ctx there would otherwise flip auto_num_ctx and fail this.
_orig_cpA = lc.CONFIG_PATH
lc.CONFIG_PATH = _plA.Path(_tfA.mkdtemp()) / "absent.toml"
try:
    c = lc.load_config(A())
    check("host accepts bare host:port -> http URL", c.host == "http://myserver:11434", c.host)
    check("auto_num_ctx on when --num-ctx absent", c.auto_num_ctx is True)
    class A2(A):
        num_ctx = 8192
    check("auto_num_ctx off when --num-ctx given", lc.load_config(A2()).auto_num_ctx is False)
finally:
    lc.CONFIG_PATH = _orig_cpA

# handover_overrides load from the [handover_overrides.<model>] table
_hop = _plA.Path(_tfA.mkdtemp()) / "ho.toml"
_hop.write_text("[handover_overrides.alpha-1]\nsoft = 0.6\nhard = 0.85\n")
_orig_cpB = lc.CONFIG_PATH
lc.CONFIG_PATH = _hop
try:
    _lc = lc.load_config(A())
    check("handover_overrides load per-model", _lc.handover_overrides.get("alpha-1", {}) ==
          {"soft": 0.6, "hard": 0.85})
finally:
    lc.CONFIG_PATH = _orig_cpB

# handover_overrides + non-default global thresholds SURVIVE a save_config round-trip.
# Regression: save_config had no serializer for handover_overrides (a BESPOKE key), so
# every autosave_config() rewrote config.toml WITHOUT them - they vanished and the next
# load fell back to the global defaults (0.70/0.95 -> the "handover 95%" reset). Model
# names carry ':'/'.' so the table header MUST be quoted or the whole file won't parse.
_hop2 = _plA.Path(_tfA.mkdtemp()) / "ho2.toml"
_orig_cpC = lc.CONFIG_PATH
lc.CONFIG_PATH = _hop2
try:
    _cfg = lc.Config()
    _cfg.handover_soft = 0.2
    _cfg.handover_hard = 0.25
    _cfg.handover_overrides = {"qwen3:32b": {"soft": 0.6, "hard": 0.85},
                               "claude-opus-4-8": {"auto": False}}
    lc.save_config(_cfg, quiet=True)
    _rl = lc.load_config(A())
    check("save_config preserves non-default global thresholds",
          (_rl.handover_soft, _rl.handover_hard) == (0.2, 0.25),
          (_rl.handover_soft, _rl.handover_hard))
    check("save_config preserves handover_overrides (colon/dot model names)",
          _rl.handover_overrides == {"qwen3:32b": {"soft": 0.6, "hard": 0.85},
                                     "claude-opus-4-8": {"auto": False}},
          _rl.handover_overrides)
finally:
    lc.CONFIG_PATH = _orig_cpC

# 12. /compact - stub old tool results, keep last N in full
ag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
for i in range(4):
    ag.messages.append({"role": "user", "content": f"q{i}"})
    ag.messages.append({"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": "read_file",
                                        "arguments": {"path": f"f{i}"}}}]})
    ag.messages.append({"role": "tool", "tool_name": "read_file",
                        "content": "\n".join(["data"] * 100)})
before = lc.messages_tokens(ag.messages, lc.TOOLS)
n, freed = ag.compact(keep=1)
after = lc.messages_tokens(ag.messages, lc.TOOLS)
tmsgs = [m for m in ag.messages if m["role"] == "tool"]
check("compact trims all but last N", n == 3, f"trimmed={n}")
check("compact stubs old tool results", tmsgs[0]["content"].startswith("[trimmed"))
check("compact keeps last N in full", not tmsgs[-1]["content"].startswith("[trimmed")
      and len(tmsgs[-1]["content"].splitlines()) == 100)
check("compact frees tokens", freed > 0 and after < before, f"freed~{freed}")
check("compact is idempotent (re-run trims 0)", ag.compact(keep=1)[0] == 0)
# compact must survive a tool message with non-string content (an older/hand-edited
# session, or an image-only result) - it auto-fires under handover, so a crash here
# would take down the whole turn.
_ncx = lc.Agent.__new__(lc.Agent)
_ncx.messages = [{"role": "system", "content": "s"},
                 {"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "", "tool_calls": [{}]},
                 {"role": "tool", "tool_name": "read_file", "content": None},
                 {"role": "assistant", "content": "", "tool_calls": [{}]},
                 {"role": "tool", "tool_name": "x", "content": "a\nb\nc"}]
_ncx.last_prompt_tokens = None
_nc_tr, _ = _ncx.compact(keep=1)
check("compact survives non-string tool content (no crash, coerced)",
      _nc_tr == 1 and _ncx.messages[3]["content"].startswith("[trimmed"))

# 12b. auto_compact (NEW): programmatic strip fired at token intervals, w/ hysteresis
acx = lc.Agent.__new__(lc.Agent)
acx.cfg = lc.Config(auto_compact_interval=100000, auto_compact_hysteresis=0.25, auto_compact_keep=1)
acx._ac_armed_boundary = 0
_acu = {"used": 0}
acx._ctx_used = lambda: _acu["used"]
_stripped = {"n": 0}
def _fake_strip(keep):
    _stripped["n"] += 1
    return (2, 5000)
acx.compact = _fake_strip
import io as _io_ac, contextlib as _ctx_ac
def _run_ac():
    with _ctx_ac.redirect_stdout(_io_ac.StringIO()):
        acx._maybe_auto_compact()
_acu["used"] = 50000; _run_ac()
check("auto_compact: does not fire below the first boundary", _stripped["n"] == 0)
_acu["used"] = 100500; _run_ac()
check("auto_compact: fires when crossing a boundary", _stripped["n"] == 1
      and acx._ac_armed_boundary == 100000)
_acu["used"] = 99000; _run_ac()   # strip landed just under 100k - must NOT refire
check("auto_compact: hysteresis blocks refire just under the boundary", _stripped["n"] == 1)
_acu["used"] = 70000; _run_ac()   # dropped below 100k - 0.25*100k=75k -> re-arms, but no new boundary
check("auto_compact: re-arms below the margin but no new boundary -> no fire", _stripped["n"] == 1
      and acx._ac_armed_boundary == 0)
_acu["used"] = 205000; _run_ac()  # crossed 200k
check("auto_compact: fires again at the next boundary", _stripped["n"] == 2
      and acx._ac_armed_boundary == 200000)
# off by default
acx.cfg.auto_compact_interval = 0
_acu["used"] = 999999; _stripped["n"] = 0; _run_ac()
check("auto_compact: off when interval is 0", _stripped["n"] == 0)
check("Config default auto_compact_interval is 0 (off)", lc.Config().auto_compact_interval == 0)

# 12b2. handover marks the stable-prefix summary message with a cache_boundary signal
_cb = lc.Agent.__new__(lc.Agent)
_cb.cfg = lc.Config()
_cb._handover_boundary = 0
_cb.messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": "Session handover (prior context summarized):\nX",
                 "cache_boundary": True}]
check("handover summary message carries cache_boundary + boundary index makes sense",
      _cb.messages[1].get("cache_boundary") is True)

# 12c. _auto_evict: stub acted-on tool results, NEVER the in-flight batch
ae = lc.Agent.__new__(lc.Agent)
ae.cfg = lc.Config(auto_evict_keep=1)
ae.last_prompt_tokens = None
ae.messages = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
    {"role": "tool", "tool_name": "read_file", "content": "A\n" * 50},   # acted-on
    {"role": "assistant", "content": "", "tool_calls": [{"id": "2"}]},
    {"role": "tool", "tool_name": "read_file", "content": "B\n" * 50},   # acted-on (last kept)
    {"role": "assistant", "content": "", "tool_calls": [{"id": "3"}, {"id": "4"}]},
    {"role": "tool", "tool_name": "read_file", "content": "C\n" * 50},   # in-flight - protect
    {"role": "tool", "tool_name": "read_file", "content": "D\n" * 50},   # in-flight - protect
]
n_ev, _ = ae._auto_evict()
check("_auto_evict stubs acted-on beyond keep", n_ev == 1, f"evicted={n_ev}")
check("_auto_evict stubbed the OLDEST acted-on", ae.messages[2]["content"].startswith("[trimmed"))
check("_auto_evict keeps last N acted-on full", ae.messages[4]["content"].startswith("B"))
check("_auto_evict NEVER touches the in-flight batch",
      ae.messages[6]["content"].startswith("C") and ae.messages[7]["content"].startswith("D"))
check("auto_evict defaults off", lc.Config().auto_evict is False)

# 12d. _window_messages: bounded send, clean turn-boundary cut, non-destructive
wm = lc.Agent.__new__(lc.Agent)
_wmsgs = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
    {"role": "tool", "tool_name": "read_file", "content": "r1"},
    {"role": "assistant", "content": "done1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
    {"role": "tool", "tool_name": "read_file", "content": "r2"},
    {"role": "assistant", "content": "done2"},
]
check("_window off (0) returns history unchanged", wm._window_messages(_wmsgs, 0) is _wmsgs)
check("_window within bound returns unchanged", wm._window_messages(_wmsgs, 99) is _wmsgs)
_w = wm._window_messages(_wmsgs, 4)
check("_window keeps the system message", _w[0]["role"] == "system")
check("_window cuts at a user boundary (no orphaned tool result)", _w[1]["role"] == "user")
check("_window starts the turn at u2", _w[1]["content"] == "u2")
check("_window never starts body on a tool message",
      all(m["role"] != "tool" for m in _w[1:2]))
check("_window is non-destructive (history intact)", len(_wmsgs) == 9)
# no user boundary within range -> skip windowing rather than orphan
_noturn = [{"role": "system", "content": "S"},
           {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
           {"role": "tool", "tool_name": "t", "content": "r"},
           {"role": "assistant", "content": "z"}]
check("_window with no user boundary in range returns unchanged",
      wm._window_messages(_noturn, 2) is _noturn)
check("window_messages defaults to 0 (off - full context)", lc.Config().window_messages == 0)

# 12d'. _window_by_tokens: HARD token cap wins over window_messages, keeps whole recent
# turns under budget, always keeps system + reserves the env/plan tail, non-destructive.
check("window_tokens defaults to 'auto' (overflow backstop on)", lc.Config().window_tokens == "auto")
wt = lc.Agent.__new__(lc.Agent)
wt.cfg = lc.Config(model="m")
wt.cfg.window_tokens = 0                      # tests below drive the numeric path explicitly
wt.tool_defs = []
wt.tool_defs = []
wt.pinned_plan = ""
wt._skip_plan_reminder_once = False
wt._plan_full_next = True
wt.remote = None
_big = [{"role": "system", "content": "S"}]
for _i in range(1, 21):
    _big.append({"role": "user", "content": f"user turn number {_i} " * 4})
    _big.append({"role": "assistant", "content": f"assistant reply number {_i} " * 4})
check("window_tokens: ample budget returns history unchanged",
      wt._window_by_tokens(_big, 100000) is _big)
_wt = wt._window_by_tokens(_big, 400)
check("window_tokens: keeps the system message first", _wt[0]["role"] == "system")
check("window_tokens: body starts on a clean user boundary (no orphan)",
      _wt[1]["role"] == "user")
check("window_tokens: actually trimmed the history", len(_wt) < len(_big))
check("window_tokens: kept slice is the most RECENT turns",
      _wt[-1]["content"] == _big[-1]["content"])
check("window_tokens: non-destructive (history intact)", len(_big) == 41)
wt.cfg.window_tokens = 400
wt.cfg.window_messages = 99
check("_window_messages: window_tokens takes precedence over window_messages",
      len(wt._window_messages(_big, 99)) < len(_big))
wt.cfg.window_tokens = 0
check("_window_messages: falls back to window_messages when window_tokens is 0",
      wt._window_messages(_big, 99) is _big)
# _resolve_window_tokens: 'auto' -> ctx_window minus a reply reserve; int passthrough; off-cases
wt.cfg.window_messages = 0
wt.cfg.window_tokens = "auto"
wt.cfg.num_ctx = 8192                          # ollama path: ctx_window() reads num_ctx
_auto = wt._resolve_window_tokens()
check("_resolve_window_tokens: 'auto' = ctx minus reply reserve (< ctx, > 0)",
      0 < _auto < 8192)
check("_resolve_window_tokens: 'auto' leaves a sane reply reserve (>= 15% of ctx)",
      8192 - _auto >= int(8192 * 0.15))
wt.cfg.window_tokens = 4000
check("_resolve_window_tokens: an int passes through as the hard cap",
      wt._resolve_window_tokens() == 4000)
wt.cfg.window_tokens = 0
check("_resolve_window_tokens: 0 -> off", wt._resolve_window_tokens() == 0)
wt.cfg.window_tokens = "off"
check("_resolve_window_tokens: junk string -> off", wt._resolve_window_tokens() == 0)
wt.cfg.window_tokens = "auto"                 # backstop no-op on a small history (fits ctx)
check("window_tokens 'auto': no-op when the history fits the window",
      wt._window_messages(_big, 0) is _big)

# 12d-bis. _auto_resume_pick: most-recent session wins, NOT cwd-scoped, snapshots skipped
_ar_rows = [
    ("lcdev-prehandover-1", {"cwd": "/x"}, 300),   # newest, but a snapshot
    ("lcdev",                          {"cwd": "/elsewhere"}, 200),  # genuinely-most-recent thread
    ("auto-0628-120000-1",             {"cwd": "/home/user"}, 100),  # stale, same-cwd
]
check("_auto_resume_pick skips pre-handover snapshots",
      lc._auto_resume_pick(_ar_rows)[0] == "lcdev")
check("_auto_resume_pick is NOT cwd-scoped (most-recent thread wins regardless of cwd)",
      lc._auto_resume_pick(_ar_rows)[1]["cwd"] == "/elsewhere")
check("_auto_resume_pick None when only snapshots exist",
      lc._auto_resume_pick([("lcdev-prehandover-1", {}, 9)]) is None)

# 12e. ContextMeter.zone: context-budget zones for self-managing compaction. The
# measurement/policy cluster lives on a focused ContextMeter (B1 extraction) - test it
# directly against a bare agent stub; Agent._ctx_zone/_ctx_used/etc. delegate to it.
cz = lc.Agent.__new__(lc.Agent)
cz.cfg = lc.Config(handover_soft=0.40, handover_hard=0.60,
                   provider_settings={"ollama": {"num_ctx": 1000}})
cz.last_prompt_tokens = None
cz.messages = []; cz.tool_defs = []
czm = lc.ContextMeter(cz)
check("ContextMeter is what Agent delegates to", lc.Agent(lc.Config()).ctx.__class__ is lc.ContextMeter)
cz.last_prompt_tokens = 300
check("zone ok below soft", czm.zone()[0] == "ok")
cz.last_prompt_tokens = 500
check("zone soft in [soft,hard)", czm.zone()[0] == "soft")
cz.last_prompt_tokens = 700
check("zone hard at/over hard", czm.zone()[0] == "hard")
check("zone reports used+limit", czm.zone()[1] == 700 and czm.zone()[2] == 1000)
cz.last_prompt_tokens = 1000
check("zone emergency at/over window", czm.zone()[0] == "emergency")
# Agent delegator routes to the meter (same answer through self._ctx_zone)
cz.ctx = czm
check("Agent._ctx_zone delegates to the meter", cz._ctx_zone()[0] == czm.zone()[0] == "emergency")
# loop guard: emergency always allowed; non-emergency gated by min interval
import time as _time_cz
cz._last_compact_ts = _time_cz.time()      # just compacted
check("compact_allowed blocks rapid non-emergency", czm.compact_allowed("hard") is False)
check("compact_allowed lets emergency through", czm.compact_allowed("emergency") is True)
cz._last_compact_ts = _time_cz.time() - 999
check("compact_allowed ok after interval", czm.compact_allowed("hard") is True)
# used() must count CACHED tokens as context (else caching hides true fill from
# the meter AND the compaction zones)
cu = lc.Agent.__new__(lc.Agent)
cu.cfg = lc.Config(); cu.messages = []; cu.tool_defs = []
cu.last_prompt_tokens = 12
cu.client = type("C", (), {"last_cache_read": 1700, "last_cache_write": 300})()
cum = lc.ContextMeter(cu)
check("used = input + cache_read + cache_write", cum.used() == 2012)
cu.client = None      # ollama-style client with no cache fields
check("used falls back to bare input when no cache fields", cum.used() == 12)
# silent server-side truncation: a fixed-num_ctx server (Ollama) drops the overflow
# and reports prompt_eval for only the surviving part, so the reported count looks
# healthy while history was actually lost. When our own estimate exceeds the reported
# count we trust the estimate, so emergency still trips. (see ContextMeter.used)
ct = lc.Agent.__new__(lc.Agent)
ct.cfg = lc.Config(); ct.tool_defs = []; ct.client = None
ct.messages = [{"role": "user", "content": "word " * 40000}]  # big estimate (~40k tokens)
ct.last_prompt_tokens = 7000                                  # server's post-truncation count
ctm = lc.ContextMeter(ct)
check("used trusts estimate over a truncated server count", ctm.used() > 7000)
# leash note injects into the NEXT user turn (uncached tail), then clears
pn = lc.Agent.__new__(lc.Agent)
pn.messages = [{"role": "system", "content": "S"}]
pn.tool_defs = []; pn.cfg = lc.Config()
pn._pending_ai_note = "Permissions changed: read-only"
pn.tools = type("T", (), {"changed_files": set()})()
pn._handover_capture = None; pn.dirty = False
pn._turn_count = 0
pn._elected_handover = False
pn._soft_nudge = lambda: ""       # isolate this test to the pending-note path
pn.refresh_tools = lambda: None   # surface rebuild is out of scope for this test
pn._sigint_guard = lambda: __import__("contextlib").nullcontext()
pn._loop = lambda: None
pn._maybe_handover = lambda: None
pn.run_turn("do the thing")
_last = pn.messages[-1]["content"]
check("run_turn prepends pending note to the user turn",
      "read-only" in _last and "do the thing" in _last)
check("run_turn clears the pending note after injecting", pn._pending_ai_note is None)

# autostart decision: after a hard-zone compaction, run the queued self-prompt (or, on
# cancel, drop it into history un-run). Interactive countdown stubbed; logic isolated.
import io as _io_mc, contextlib as _ctx_mc
mc = lc.Agent.__new__(lc.Agent)
mc.cfg = lc.Config()                       # auto_handover on by default
_st = {"compacted": False}
mc._ctx_used = lambda: 0
mc._ctx_zone = lambda: ("ok", 0, 1000) if _st["compacted"] else ("hard", 600, 1000)
mc._compact_allowed = lambda z: True
def _fake_compact(zone):
    _st["compacted"] = True
    mc._autostart_pending = "do step 2"
    return "BLOCK"
mc.auto_handover = _fake_compact
mc._autostart_pending = None
mc._autostart_countdown = lambda sp, secs=5: True
_ran = []
mc.run_turn = lambda ui: _ran.append(ui)
mc.messages = []
with _ctx_mc.redirect_stdout(_io_mc.StringIO()):
    mc._maybe_handover()
check("autostart runs the self-prompt when countdown proceeds", _ran == ["do step 2"])
# cancel branch: drop into history, do NOT run
_st["compacted"] = False; mc._autostart_pending = None; _ran.clear(); mc.messages = []
mc._autostart_countdown = lambda sp, secs=5: False
with _ctx_mc.redirect_stdout(_io_mc.StringIO()):
    mc._maybe_handover()
check("autostart cancel: self-prompt dropped into history, not run",
      _ran == [] and any("autostart cancelled" in (m.get("content") or "") for m in mc.messages))
# guard: if compaction didn't drop below hard, don't autostart (would loop)
_st["compacted"] = False; mc._autostart_pending = None; _ran.clear(); mc.messages = []
mc._ctx_zone = lambda: ("hard", 600, 1000)      # stays hard even after compact
mc._autostart_countdown = lambda sp, secs=5: True
with _ctx_mc.redirect_stdout(_io_mc.StringIO()):
    mc._maybe_handover()
check("autostart skipped when still full after compaction",
      _ran == [] and any("still full" in (m.get("content") or "") for m in mc.messages))
check("auto_handover defaults on", lc.Config().auto_handover is True)
check("compact thresholds default 0.7/0.95 + emergency 1.0",
      lc.Config().handover_soft == 0.70 and lc.Config().handover_hard == 0.95
      and lc.Config().handover_emergency == 1.00)
check("autostart_after_handover defaults ON", lc.Config().autostart_after_handover is True)

# 12e2. per-model compaction overrides (models fill context differently)
cmc = lc.Config(provider="testprov", handover_soft=0.40, handover_hard=0.60,
                autostart_after_handover=True,    # opt in, to test fallback to the global
                provider_settings={"testprov": {"model": "alpha-1"}},
                handover_overrides={"alpha-1": {"soft": 0.5, "hard": 0.75}})
_cf = cmc.handover_for()
check("handover_for applies per-model override", _cf["soft"] == 0.5 and _cf["hard"] == 0.75)
check("handover_for falls back to global for absent keys",
      _cf["auto"] is True and _cf["autostart"] is True)
check("handover_for uses global when no override for model",
      cmc.handover_for("beta-2")["hard"] == 0.60)

# 12f. self-prompt extractor (post-compaction autostart) + shared marker extractor
check("_extract_selfprompt pulls last NEXT pair",
      lc._extract_selfprompt("noise ===NEXT===\ndo the next thing\n===NEXT=== tail") == "do the next thing")
check("_extract_selfprompt None when no pair", lc._extract_selfprompt("no markers here") is None)
check("_extract_handover_block still works via shared helper",
      lc._extract_handover_block("x ===HANDOVER===\nHANDOVER\n===HANDOVER=== y") == "HANDOVER")
check("_extract_marked None on empty", lc._extract_marked("", "===NEXT===") is None)
check("_extract_plan pulls the ===PLAN=== block",
      lc._extract_plan("x ===PLAN===\nGOAL: ship\nTODO:\n- [ ] a\n===PLAN=== y") == "GOAL: ship\nTODO:\n- [ ] a")
# pinned plan is an UNCACHED send-time reminder, NOT in the cached system block
_pp = lc.Agent(lc.Config(cwd=FIX))
check("system prompt never carries the plan (unset)", "PINNED PLAN" not in _pp.messages[0]["content"])
_pp.pinned_plan = "GOAL: bind ollama\nTODO:\n- [ ] restart\n- [x] done"
check("plan stays OUT of _system() (cache-stable)", "GOAL: bind ollama" not in _pp._system())
check("_plan_reminder carries the plan + self-labelling header",
      "GOAL: bind ollama" in _pp._plan_reminder()
      and "PINNED PLAN" in _pp._plan_reminder()
      and "not a user message" in _pp._plan_reminder().lower())
check("_plan_reminder empty when no plan",
      lc.Agent(lc.Config(cwd=FIX))._plan_reminder() == "")
# sparse view: keeps GOAL + open [ ] items, drops done [x] items
_sp = _pp._plan_reminder(sparse=True)
check("_plan_reminder sparse keeps GOAL + open items",
      "GOAL: bind ollama" in _sp and "- [ ] restart" in _sp)
check("_plan_reminder sparse drops completed [x] items", "- [x] done" not in _sp)
# _with_plan_reminder stitches onto the SENT slice only, never into stored history
_pp.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
_sent = _pp._with_plan_reminder(_pp.messages)
check("_with_plan_reminder appends the plan to the sent slice's last message",
      "GOAL: bind ollama" in _sent[-1]["content"])
check("_with_plan_reminder does NOT mutate stored history",
      "GOAL: bind ollama" not in _pp.messages[-1]["content"])
# a tool-tailed slice gets a trailing user note (never appends to a role:tool result)
_ppt = lc.Agent(lc.Config(cwd=FIX)); _ppt.pinned_plan = "GOAL: x"
_st = _ppt._with_plan_reminder([{"role": "tool", "tool_name": "read_file", "content": "DATA"}])
check("_with_plan_reminder keeps a tool result exact, adds a trailing user note",
      _st[0]["content"] == "DATA" and _st[-1]["role"] == "user" and "GOAL: x" in _st[-1]["content"])
# With no plan pinned the PLAN block is absent, but the live session-env ALWAYS rides
# the uncached tail (cwd + shell family) - so the last message gains the ENV block only.
_npr = lc.Agent(lc.Config(cwd=FIX))._with_plan_reminder([{"role": "user", "content": "q"}])
check("_with_plan_reminder: no plan -> env rides, no plan block",
      _npr[-1]["content"].startswith("q")
      and "SESSION ENV" in _npr[-1]["content"]
      and "PINNED PLAN" not in _npr[-1]["content"])

# soft-zone nudge: throttled, soft zone only, injected (not in the cached prompt)
sn = lc.Agent.__new__(lc.Agent)
sn.cfg = lc.Config()                 # auto_compact on by default
sn._turn_count = 10; sn._last_nudge_turn = -999
sn._ctx_zone = lambda: ("soft", 500, 1000)
_snout = sn._soft_nudge()
check("_soft_nudge fires in soft zone when due", "context" in _snout and "%" in _snout)
check("_soft_nudge throttled within NUDGE_EVERY", sn._soft_nudge() == "")
sn._turn_count = 16
_snout = sn._soft_nudge()
check("_soft_nudge fires again after NUDGE_EVERY turns", "context" in _snout and "%" in _snout)
sn._ctx_zone = lambda: ("ok", 100, 1000); sn._turn_count = 999
check("_soft_nudge silent outside the soft zone", sn._soft_nudge() == "")
# the auto-handover instruction + soft nudge are editable builtin prompts
check("auto_handover is a builtin prompt", "auto_handover" in lc.BUILTIN_PROMPTS)
check("handover_nudge is a builtin prompt", "handover_nudge" in lc.BUILTIN_PROMPTS)
check("read_prompt('auto_handover') returns the compaction instruction",
      lc.read_prompt("auto_handover") == lc.AUTO_HANDOVER_INSTR)
check("read_prompt('handover_nudge') returns the nudge template",
      lc.read_prompt("handover_nudge") == lc.HANDOVER_NUDGE)
check("handover_nudge template has ctx placeholders",
      "{used}" in lc.HANDOVER_NUDGE and "{pct}" in lc.HANDOVER_NUDGE)
import tomllib as _tae
_aef = Path(tempfile.mktemp(suffix=".toml"))
_orig_aecp = lc.CONFIG_PATH; lc.CONFIG_PATH = _aef
lc.save_config(lc.Config(model="m", auto_evict=True, auto_evict_keep=5), quiet=True)
_aev = _tae.loads(_aef.read_text())
check("save_config persists auto_evict on + keep", _aev.get("auto_evict") is True and _aev.get("auto_evict_keep") == 5)
lc.CONFIG_PATH = _orig_aecp; _aef.unlink(missing_ok=True)

# 12b. compact clears the stale actual prompt_eval count (the "looked broken" bug)
ag2 = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
for i in range(3):
    ag2.messages.append({"role": "tool", "tool_name": "read_file",
                         "content": "\n".join(["x"] * 50)})
ag2.last_prompt_tokens = 9999
ag2.compact(keep=1)
check("compact clears stale last_prompt_tokens when it trims",
      ag2.last_prompt_tokens is None)
ag2.last_prompt_tokens = 1234
check("compact leaves token count alone when nothing trims",
      ag2.compact(keep=1)[0] == 0 and ag2.last_prompt_tokens == 1234)

# 12c. Spinner is a safe no-op when stdout isn't a TTY (tests/pipes)
sp = lc.Spinner("x").start()
sp.stop()
check("spinner no-op offline: no thread, no active spinner",
      sp._thread is None and lc._active_spinner is None)
# 12c'. Nested spinners stack: only the TOP is active; stopping it resumes the one
# underneath; stopping the last clears. (Guards the flip-flop bug: two painters on
# one status line.) Force the TTY path so start()/stop() manage the stack.
_otty = lc._TTY
lc._TTY = False   # keep it painter-free but still stack-managed (the no-TTY branch pushes too)
try:
    _s1 = lc.Spinner("outer").start()
    check("spinner stack: first push is active + on stack",
          lc._active_spinner is _s1 and lc._spinner_stack == [_s1])
    _s2 = lc.Spinner("inner").start()
    check("spinner stack: nested push becomes top",
          lc._active_spinner is _s2 and lc._spinner_stack == [_s1, _s2])
    _s2.stop()
    check("spinner stack: pop resumes the one underneath",
          lc._active_spinner is _s1 and lc._spinner_stack == [_s1])
    _s1.stop()
    check("spinner stack: last pop clears",
          lc._active_spinner is None and lc._spinner_stack == [])
finally:
    lc._TTY = _otty

# --- the bundled ollama provider lives in providers/ollama.py now (extracted
# from core). Load it the way the manager does (compile+exec) and run its
# helpers-only setup() with the core namespace so the moved functions resolve
# their injected helpers. The moved code is tested on this module handle (olm).
import pathlib as _plib
olm = lc._load_lean_tool(_plib.Path(__file__).resolve().parent.parent / "providers" / "ollama.py")
olm.setup(lc.__dict__, lc.Config(cwd=FIX))
check("ollama provider: PROVIDER dict has the required hooks",
      all(callable(olm.PROVIDER[k]) for k in ("make_client", "list_models", "context_window")))

# 13. _parse_num_ctx - prefer explicit num_ctx param, else arch context_length.
# Returns (value, pinned): pinned=True iff an explicit Modelfile num_ctx set it.
check("num_ctx: explicit param wins",
      olm._parse_num_ctx({"parameters": "num_ctx 8192\n",
                          "model_info": {"qwen3.context_length": 262144}}) == (8192, True))
check("num_ctx: falls back to context_length",
      olm._parse_num_ctx({"model_info": {"qwen3moe.context_length": 262144}}) == (262144, False))
check("num_ctx: None when unknown", olm._parse_num_ctx({"model_info": {}}) == (None, False))

# 14. cap_num_ctx - auto-detect only lowers, never raises above the cap
check("cap: big model max -> capped to 32768", olm.cap_num_ctx(262144) == (32768, True))
check("cap: small model max -> used as-is", olm.cap_num_ctx(16384) == (16384, False))
check("cap: exactly at cap -> not flagged capped", olm.cap_num_ctx(32768) == (32768, False))
check("cap: None detected -> None", olm.cap_num_ctx(None) == (None, False))
check("cap: custom ceiling honored", olm.cap_num_ctx(262144, cap=131072) == (131072, True))

# 15. tiered host failover
P_ALL = lambda h, t=2: True
P_NONE = lambda h, t=2: False
check("select_host: all alive -> first",
      olm.select_host(["http://A", "http://B"], probe=P_ALL) == ("http://A", True))
check("select_host: first dead -> second",
      olm.select_host(["http://A", "http://B"], probe=lambda h, t=2: h.endswith("B"))
      == ("http://B", True))
check("select_host: none alive -> first, False",
      olm.select_host(["http://A", "http://B"], probe=P_NONE) == ("http://A", False))
check("select_host: empty -> (None, False)", olm.select_host([], probe=P_NONE) == (None, False))
check("_norm_host adds scheme + strips slash",
      lc._norm_host("box:11434") == "http://box:11434" and lc._norm_host("http://x/") == "http://x:11434")
check("_norm_host appends default port when missing",
      lc._norm_host("gpu-box") == "http://gpu-box:11434")
check("_norm_host preserves an explicit custom port",
      lc._norm_host("http://gpu-box:8080") == "http://gpu-box:8080")
check("_norm_host localhost -> 127.0.0.1 with default port",
      lc._norm_host("localhost") == "http://127.0.0.1:11434")
check("valid_host rejects junk / accepts real hosts",
      lc.valid_host("") is False and lc.valid_host("a b") is False
      and lc.valid_host("gpu-box") is True and lc.valid_host("http://x:8080") is True)

# 15a. _failover: re-probe the priority pool, jump to the next live host, keep order.
class _FoAgent:
    def __init__(self): self.activated = []
    def activate_provider(self, n): self.activated.append(n)
_orig_alive = olm.host_alive
try:
    # current (A) dead, B dead, C live -> switch to C; pool order untouched.
    olm.host_alive = lambda h, t=2.0: "C" in h
    fc = lc.Config(cwd=FIX, host="http://A:11434",
                   hosts=["http://A:11434", "http://B:11434", "http://C:11434"])
    fa = _FoAgent()
    import io, contextlib, pathlib
    _b = io.StringIO()
    with contextlib.redirect_stdout(_b):
        ok = olm._failover(fa, fc, announce=True)
    check("_failover: jumps to next live host", ok is True and fc.host == "http://C:11434", fc.host)
    check("_failover: priority pool order unchanged",
          fc.hosts == ["http://A:11434", "http://B:11434", "http://C:11434"], str(fc.hosts))
    check("_failover: re-activates ollama", fa.activated == ["ollama"], str(fa.activated))
    # nothing else alive -> False, active host untouched.
    olm.host_alive = lambda h, t=2.0: False
    fc2 = lc.Config(cwd=FIX, host="http://A:11434",
                    hosts=["http://A:11434", "http://B:11434"])
    with contextlib.redirect_stdout(io.StringIO()):
        ok2 = olm._failover(_FoAgent(), fc2)
    check("_failover: none live -> False, host kept", ok2 is False and fc2.host == "http://A:11434")
finally:
    olm.host_alive = _orig_alive

# 15b'. _set_priority: persisted reorder; unknown token rejected, order kept.
_pcfg = pathlib.Path(tempfile.mktemp(suffix=".toml"))
_pcp = lc.CONFIG_PATH
lc.CONFIG_PATH = _pcfg
pc = lc.Config(cwd=FIX, host="http://a:11434", hosts=["http://a:11434", "http://b:11434"])
with contextlib.redirect_stdout(io.StringIO()):
    olm._set_priority(_FoAgent(), pc, "b a")
check("_set_priority: rewrites pool in given order",
      pc.hosts == ["http://b:11434", "http://a:11434"], str(pc.hosts))
pc2 = lc.Config(cwd=FIX, host="http://a:11434", hosts=["http://a:11434"])
with contextlib.redirect_stdout(io.StringIO()):
    olm._set_priority(_FoAgent(), pc2, "b ///")   # '///' has no host part -> rejected
check("_set_priority: junk token -> pool unchanged", pc2.hosts == ["http://a:11434"], str(pc2.hosts))
lc.CONFIG_PATH = _pcp
_pcfg.unlink(missing_ok=True)

# 15b. interactive pickers (probe + prompt injected; no network, no TTY)
pr = olm._probe_all(["http://A", "http://B"], probe=lambda h, t=2: h.endswith("A"))
check("_probe_all maps each url to its liveness",
      pr == {"http://A": True, "http://B": False}, str(pr))
MACH = {"gpu": "192.0.2.28:11434", "mac": "http://mac:11434"}
hsel = olm.pick_host_menu(MACH, "http://mac:11434", probe=P_ALL, prompt=lambda _: "1")
check("pick_host_menu: choice 1 -> first machine url", hsel == "http://192.0.2.28:11434", str(hsel))
check("pick_host_menu: 0 cancels -> None",
      olm.pick_host_menu(MACH, None, probe=P_ALL, prompt=lambda _: "0") is None)
check("pick_host_menu: empty + no current -> None",
      olm.pick_host_menu({}, None, probe=P_ALL, prompt=lambda _: "1") is None)
# current host not among machines is appended as a selectable row
csel = olm.pick_host_menu({}, "http://solo:11434", probe=P_NONE, prompt=lambda _: "1")
check("pick_host_menu: lone current host is selectable", csel == "http://solo:11434", str(csel))
# priority-pool hosts (cfg.hosts) with NO [machines] entry must still be listed and
# selectable - the bug where a bare `hosts=[...]` host (e.g. 203.0.113.216) was hidden.
psel = olm.pick_host_menu({}, "http://127.0.0.1:11434", probe=P_ALL, prompt=lambda _: "2",
                          pool=["http://127.0.0.1:11434", "http://203.0.113.216:11434"])
check("pick_host_menu: pool host with no [machines] entry is selectable",
      psel == "http://203.0.113.216:11434", str(psel))
# a pool url that duplicates a machines url is not listed twice (dedup, machines name kept)
pdd = olm.pick_host_menu({"gpu": "http://203.0.113.216:11434"}, None, probe=P_ALL,
                         prompt=lambda _: "1", pool=["http://203.0.113.216:11434"])
check("pick_host_menu: pool url dupes a machines url -> single row",
      pdd == "http://203.0.113.216:11434", str(pdd))
# pick_model_menu now takes current/warm/prompt without breaking selection
msel = lc.pick_model_menu(["a:latest", "b:7b"], current="a", warm=["a:latest"],
                          prompt=lambda _: "2")
check("pick_model_menu: choice 2 -> second model", msel == "b:7b", str(msel))
check("pick_model_menu: 0 cancels -> None",
      lc.pick_model_menu(["a", "b"], prompt=lambda _: "0") is None)

# 15c. _edit_priority_menu: BIOS-style editor (move/add/remove; Enter = no-op)
import io as _io_ep, contextlib as _ctx_ep
class _EpAgent:
    def activate_provider(self, n): pass
def _run_editor(hosts, inputs):
    """Drive the editor with a scripted prompt; returns the config after it exits."""
    _cfg = lc.Config(cwd=FIX, host=hosts[0], hosts=list(hosts))
    _it = iter(inputs)
    _pcp = lc.CONFIG_PATH
    lc.CONFIG_PATH = pathlib.Path(tempfile.mktemp(suffix=".toml"))
    _oa = olm.host_alive
    olm.host_alive = lambda h, t=2.0: True
    try:
        with _ctx_ep.redirect_stdout(_io_ep.StringIO()):
            olm._edit_priority_menu(_EpAgent(), _cfg, prompt=lambda _: next(_it))
    finally:
        olm.host_alive = _oa
        lc.CONFIG_PATH = _pcp
    return _cfg
HS = ["http://a:11434", "http://b:11434", "http://c:11434"]
# bare Enter -> no-op even after a move (discard)
r = _run_editor(HS, ["2 u", ""])
check("_edit_priority: Enter discards (no persist)", r.hosts == HS, str(r.hosts))
# move slot 2 up, then save
r = _run_editor(HS, ["2 u", "s"])
check("_edit_priority: move up + save persists",
      r.hosts == ["http://b:11434", "http://a:11434", "http://c:11434"], str(r.hosts))
# remove slot 3, save
r = _run_editor(HS, ["-3", "s"])
check("_edit_priority: remove + save",
      r.hosts == ["http://a:11434", "http://b:11434"], str(r.hosts))
# add a host, save
r = _run_editor(HS, ["+d", "s"])
check("_edit_priority: add + save",
      r.hosts == HS + ["http://d:11434"], str(r.hosts))
# save with no change -> unchanged
r = _run_editor(HS, ["s"])
check("_edit_priority: save w/o change -> unchanged", r.hosts == HS, str(r.hosts))
# junk add rejected, then Enter -> unchanged
r = _run_editor(HS, ["+a b", ""])
check("_edit_priority: junk add rejected", r.hosts == HS, str(r.hosts))

# load_config parses a tiered hosts[] list from the config file
import tempfile, pathlib
cfgf = pathlib.Path(tempfile.mktemp(suffix=".toml"))
cfgf.write_text('hosts = ["192.0.2.42:11434", "http://mac:11434"]\nmodel = "demo-coder"\n')
_orig_cp = lc.CONFIG_PATH
lc.CONFIG_PATH = cfgf
class HArgs:
    host = None; model = None; num_ctx = None; cwd = str(FIX); approval = "ask"
    temperature = None; top_p = None; top_k = None; repeat_penalty = None; think = None
hc = lc.load_config(HArgs())
check("config hosts[] -> cfg.hosts normalized",
      hc.hosts == ["http://192.0.2.42:11434", "http://mac:11434"], str(hc.hosts))
check("config hosts[] -> host = first (provisional)", hc.host == "http://192.0.2.42:11434")
class HArgs2(HArgs):
    host = "http://explicit:11434"
ho = lc.load_config(HArgs2())
check("--host overrides active pick + joins pool, priority order kept",
      ho.host == "http://explicit:11434"
      and ho.hosts == ["http://explicit:11434", "http://192.0.2.42:11434", "http://mac:11434"],
      str((ho.host, ho.hosts)))
class HArgs3(HArgs):
    host = "http://192.0.2.42:11434"     # already in the pool -> pick it, no reorder
ho2 = lc.load_config(HArgs3())
check("--host of an existing pool member: picks it, pool unchanged",
      ho2.host == "http://192.0.2.42:11434"
      and ho2.hosts == ["http://192.0.2.42:11434", "http://mac:11434"], str((ho2.host, ho2.hosts)))
lc.CONFIG_PATH = _orig_cp
cfgf.unlink(missing_ok=True)

# 16. per-host model: ensure_model + [models] parse/save
check("_menu_index parses valid", lc._menu_index("3", 4) == 2)
check("_menu_index rejects 0/oob/nonnum",
      lc._menu_index("0", 4) is None and lc._menu_index("5", 4) is None
      and lc._menu_index("x", 4) is None)

# --- MENU CONTRACT: a bare 1-based integer arg selects the Nth item from the SAME
# ordered list the command's picker renders. One resolver (menu_resolve) backs every
# menu-fronting command; these guard against a command re-drifting to a literal (the
# `/connect 8` -> bare-integer-address bug).
_mc = ["user@a", "user@b", "user@c"]
check("menu_resolve: 1-based index -> item", lc.menu_resolve("2", _mc) == "user@b")
check("menu_resolve: out-of-range -> None", lc.menu_resolve("9", _mc) is None)
check("menu_resolve: 0 -> None", lc.menu_resolve("0", _mc) is None)
check("menu_resolve: non-numeric -> None (caller does literal)",
      lc.menu_resolve("user@a", _mc) is None)
check("menu_resolve: empty list -> None", lc.menu_resolve("1", []) is None)
# _connect_menu_order must match the picker ordering (saved targets, then open-unsaved)
class _FakeAg:
    def __init__(self, remotes): self.remotes = remotes
_cord = lc._connect_menu_order(
    lc.Config(connect_hosts={"box1": "user@192.0.2.1", "box2": "user@192.0.2.2"}),
    _FakeAg(["user@open"]))
check("connect menu order: saved-then-open",
      _cord == ["user@192.0.2.1", "user@192.0.2.2", "user@open"])
check("connect menu order: numeric arg picks the same row menu shows",
      lc.menu_resolve("3", _cord) == "user@open")
# the picker renders from the SAME items builder -> can't drift from the arg path
check("connect: picker + arg path share one order",
      [t for _n, t in lc._connect_menu_items(
          {"box1": "user@192.0.2.1", "box2": "user@192.0.2.2"}, ["user@open"])] == _cord)
# every menu-fronting command's numeric branch routes through menu_resolve
import inspect as _insp
_src_rma = _insp.getsource(lc._resolve_model_arg)
check("contract: _resolve_model_arg uses menu_resolve (no ad-hoc isdigit index)",
      "menu_resolve(" in _src_rma and "int(arg) - 1" not in _src_rma)
_src_conn = _insp.getsource(lc.handle_connect_command)
check("contract: handle_connect_command uses menu_resolve",
      "menu_resolve(" in _src_conn)

# --- /expand msg [N]: conversation view with clamp safeguards (never errors on a silly
# count, never under-reads). _render_message_tail is the shared renderer.
import io as _io_em, contextlib as _cl_em
_em_msgs = ([{"role": "system", "content": "sys"}]
            + [{"role": ("user" if i % 2 == 0 else "assistant"), "content": f"m{i}"}
               for i in range(6)])                 # 6 non-system messages
def _em_render(n):
    with _cl_em.redirect_stdout(_io_em.StringIO()) as _b:
        ok = lc._render_message_tail(_em_msgs, n=n)
    return ok, _b.getvalue()
_ok3, _o3 = _em_render(3)
check("expand msg: shows last N", _ok3 and "3 of 6 messages" in _o3
      and "m5" in _o3 and "m3" in _o3 and "m2" not in _o3)
_okBig, _oBig = _em_render(324221341341323)         # absurd count -> clamp to all, no error
check("expand msg: absurd N clamps to all (no error)", _okBig and "6 of 6 messages" in _oBig)
_ok0, _o0 = _em_render(0)                           # 0/negative -> clamp up to 1
check("expand msg: 0 clamps up to 1", _ok0 and "1 of 6 messages" in _o0)
_okNeg, _oNeg = _em_render(-5)
check("expand msg: negative clamps up to 1", _okNeg and "1 of 6 messages" in _oNeg)
with _cl_em.redirect_stdout(_io_em.StringIO()):     # empty conversation -> False, no crash
    _okEmpty = lc._render_message_tail([{"role": "system", "content": "s"}], n=10)
check("expand msg: empty conversation -> no-op False", _okEmpty is False)
# the resume banner appends the /expand msg hint when the view is partial
with _cl_em.redirect_stdout(_io_em.StringIO()) as _bh:
    lc._print_session_tail(_em_msgs, n=2)
check("expand msg: resume banner hints /expand msg when partial",
      "/expand msg" in _bh.getvalue())

class FakeClient:
    def __init__(self, models): self._m = models
    def list_models(self): return self._m
cm1 = lc.Config(cwd=FIX, model="demo-coder")
olm.ensure_model(FakeClient(["demo-coder", "x"]), cm1, interactive=False)
check("ensure_model: present -> unchanged", cm1.model == "demo-coder")
cm2 = lc.Config(cwd=FIX, model="ghost")
olm.ensure_model(FakeClient(["a", "b"]), cm2, interactive=False)
check("ensure_model: absent + non-interactive -> unchanged (warns only)", cm2.model == "ghost")
cm3 = lc.Config(cwd=FIX, model="ghost")
olm.ensure_model(FakeClient([]), cm3, interactive=False)
check("ensure_model: host unreachable (no models) -> no-op", cm3.model == "ghost")

# tag-aware availability (bare name <-> :latest)
check("model_available: bare matches :latest", lc.model_available("demo-coder", ["demo-coder:latest"]))
check("model_available: :latest matches bare", lc.model_available("demo-coder:latest", ["demo-coder"]))
check("model_available: exact match", lc.model_available("m:7b", ["m:7b"]))
check("model_available: genuine miss", not lc.model_available("x", ["y", "z:latest"]))
# ensure_model must stay SILENT when the only diff is the implicit :latest
import io, contextlib
cm4 = lc.Config(cwd=FIX, model="demo-coder")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    olm.ensure_model(FakeClient(["demo-coder:latest"]), cm4, interactive=False)
check("ensure_model: no false warning on implicit :latest",
      "not available" not in _buf.getvalue() and cm4.model == "demo-coder", repr(_buf.getvalue()))

# [models] table parses into host_models (normalized keys)
mcfg = pathlib.Path(tempfile.mktemp(suffix=".toml"))
mcfg.write_text('model = "qwen3-coder:30b"\n[models]\n"192.0.2.42:11434" = "demo-coder"\n')
lc.CONFIG_PATH = mcfg
mc = lc.load_config(HArgs())
check("config [models] -> host_models normalized",
      mc.host_models == {"http://192.0.2.42:11434": "demo-coder"}, str(mc.host_models))

# save_config round-trips [models]
outf = pathlib.Path(tempfile.mktemp(suffix=".toml"))
lc.CONFIG_PATH = outf
sc = lc.Config(host="http://h:11434", model="m",
               host_models={"http://h:11434": "demo-coder"})
lc.save_config(sc)
import tomllib as _t
rt = _t.loads(outf.read_text())
check("save_config writes [models] table",
      rt.get("models", {}).get("http://h:11434") == "demo-coder", str(rt.get("models")))
check("save_config writes ask_user_to_run", rt.get("ask_user_to_run") is True)
check("save_config writes autosave (on by default)", rt.get("autosave") is True)
lc.save_config(lc.Config(host="http://h:11434", model="m", autosave=False))
check("save_config persists autosave off",
      _t.loads(outf.read_text()).get("autosave") is False)
_qf = pathlib.Path(tempfile.mktemp(suffix=".toml")); lc.CONFIG_PATH = _qf
import io as _io, contextlib as _ctx
_buf = _io.StringIO()
with _ctx.redirect_stdout(_buf):
    lc.save_config(lc.Config(model="m"), quiet=True)
check("save_config quiet=True prints nothing", _buf.getvalue() == "")
lc.autosave_config(lc.Config(model="m"))
check("autosave_config writes config silently", _qf.is_file())
check("CONFIG_DIR exposed for providers (~/.config/leancoder)",
      lc.CONFIG_DIR == pathlib.Path.home() / ".config" / "leancoder")
lc.CONFIG_PATH = outf

# /save: persist num_ctx only when explicitly pinned, not when auto-detected
# (otherwise saving after a 16k host would bake in and cap a 32k host).
nf = pathlib.Path(tempfile.mktemp(suffix=".toml"))
lc.CONFIG_PATH = nf
lc.save_config(lc.Config(model="m", num_ctx=16384, auto_num_ctx=True))
check("save_config omits auto-detected num_ctx (detection stays per-host)",
      "num_ctx" not in _t.loads(nf.read_text()))
lc.save_config(lc.Config(model="m", num_ctx=8192, auto_num_ctx=False))
check("save_config writes an explicitly-pinned num_ctx",
      _t.loads(nf.read_text()).get("num_ctx") == 8192)

# keep_alive: default omitted, non-default written + loaded, payload-gated
check("keep_alive defaults to 10m", lc.Config().keep_alive == "10m")
lc.save_config(lc.Config(model="m"))
check("save_config omits keep_alive at default", "keep_alive" not in _t.loads(nf.read_text()))
lc.save_config(lc.Config(model="m", keep_alive="-1"))
check("save_config writes a non-default keep_alive",
      _t.loads(nf.read_text()).get("keep_alive") == "-1")
_kaf = pathlib.Path(tempfile.mktemp(suffix=".toml"))
_kaf.write_text('model = "m"\nkeep_alive = "30m"\nauto_evict = true\nauto_evict_keep = 5\n')
_orig_ka = lc.CONFIG_PATH; lc.CONFIG_PATH = _kaf
_kac = lc.load_config(HArgs())
check("load_config reads keep_alive", _kac.keep_alive == "30m")
check("load_config reads auto_evict + keep", _kac.auto_evict is True and _kac.auto_evict_keep == 5)
lc.CONFIG_PATH = _orig_ka; _kaf.unlink(missing_ok=True)

# round-trip the toggle off
af = pathlib.Path(tempfile.mktemp(suffix=".toml"))
af.write_text('model = "m"\nask_user_to_run = false\n')
lc.CONFIG_PATH = af
check("load_config reads ask_user_to_run=false", lc.load_config(HArgs()).ask_user_to_run is False)
check("ask_user_to_run defaults on when absent", lc.Config().ask_user_to_run is True)
lc.CONFIG_PATH = _orig_cp
mcfg.unlink(missing_ok=True); outf.unlink(missing_ok=True); af.unlink(missing_ok=True)

# 17. apply_host_default_model - shared by startup and /host
HM = {"http://h:11434": "demo-coder"}
a1 = lc.Config(cwd=FIX, host="http://h:11434", model="global", host_models=HM)
a1.model_explicit = True
olm.apply_host_default_model(FakeClient(["global", "demo-coder"]), a1, respect_explicit=True)
check("apply_host_default_model: explicit --model not overridden", a1.model == "global")
a2 = lc.Config(cwd=FIX, host="http://h:11434", model="global", host_models=HM)
olm.apply_host_default_model(FakeClient(["global", "demo-coder"]), a2, respect_explicit=False)
check("apply_host_default_model: host default applied (/host path)", a2.model == "demo-coder")
a3 = lc.Config(cwd=FIX, host="http://other:11434", model="global", host_models=HM)
olm.apply_host_default_model(FakeClient(["global"]), a3, respect_explicit=False)
check("apply_host_default_model: no entry for host -> keeps current", a3.model == "global")

# 18. memorable machine names ([machines] aliases)
MACH = {"gpu": "http://192.0.2.42:11434", "mini": "http://mac:11434"}
check("resolve_host: name -> url", lc.resolve_host("gpu", MACH) == "http://192.0.2.42:11434")
check("resolve_host: raw -> normalized", lc.resolve_host("box:11434", MACH) == "http://box:11434")
check("host_label: known -> 'name (url)'",
      lc.host_label("http://mac:11434", MACH) == "mini (http://mac:11434)")
check("host_label: unknown -> url", lc.host_label("http://x:11434", MACH) == "http://x:11434")

# load_config: [machines] parsed; hosts list + host resolved by name
amf = pathlib.Path(tempfile.mktemp(suffix=".toml"))
amf.write_text('hosts = ["gpu", "mini", "http://raw:11434"]\n\n'
               '[machines]\ngpu = "192.0.2.42:11434"\nmini = "http://mac:11434"\n')
lc.CONFIG_PATH = amf
am = lc.load_config(HArgs())
check("config [machines] parsed + normalized",
      am.machines == {"gpu": "http://192.0.2.42:11434", "mini": "http://mac:11434"})
check("hosts[] resolved via names",
      am.hosts == ["http://192.0.2.42:11434", "http://mac:11434", "http://raw:11434"], str(am.hosts))
# --host accepts a name too
class NArgs(HArgs):
    host = "mini"
check("--host name resolves to url",
      lc.load_config(NArgs()).host == "http://mac:11434")
lc.CONFIG_PATH = _orig_cp
amf.unlink(missing_ok=True)

# save_config writes [machines] and tiered hosts by name
sf = pathlib.Path(tempfile.mktemp(suffix=".toml"))
lc.CONFIG_PATH = sf
scm = lc.Config(model="m", machines={"gpu": "http://g:11434", "mini": "http://m:11434"},
                hosts=["http://g:11434", "http://m:11434"])
lc.save_config(scm)
back = _t.loads(sf.read_text())
check("save_config writes [machines]", back.get("machines", {}).get("gpu") == "http://g:11434")
check("save_config writes hosts by name", back.get("hosts") == ["gpu", "mini"], str(back.get("hosts")))
lc.CONFIG_PATH = _orig_cp
sf.unlink(missing_ok=True)

# 19. remote executor protocol (stage 1: local subprocess, no SSH)
import subprocess as _sp, sys as _sys
exdir = pathlib.Path(tempfile.mkdtemp(prefix="lc_exec_"))
(exdir / "a.py").write_text("x = 1\ny = 2\n")
(exdir / "node_modules").mkdir()
(exdir / "node_modules" / "junk.js").write_text("nope\n")
client = lc.ExecutorClient(
    [_sys.executable, lc.__file__, "--tool-exec", "--cwd", str(exdir)],
    stderr=_sp.DEVNULL)  # tool prints (diffs) go to executor stderr; silence them
try:
    check("executor handshake ready", client.ready.get("ready") is True)
    check("executor read_file", "x = 1" in client.call("read_file", {"path": "a.py"}))
    check("executor read_file range",
          "2\ty = 2" in client.call("read_file", {"path": "a.py", "start": 2, "end": 2}))
    client.call("write_file", {"path": "b.txt", "content": "hi\n"})
    check("executor write_file (driver pre-approved -> no prompt)", (exdir / "b.txt").is_file())
    client.call("apply_diff", {"path": "a.py",
                "diff": "<<<<<<< SEARCH\ny = 2\n=======\ny = 99\n>>>>>>> REPLACE"})
    check("executor apply_diff applied", "y = 99" in (exdir / "a.py").read_text())
    check("executor search_files (honors ignore)",
          "a.py:" in client.call("search_files", {"pattern": "x ="})
          and "node_modules" not in client.call("search_files", {"pattern": "x ="}))
    check("executor run_command", "hello" in client.call("run_command", {"cmd": "echo hello"}))
    check("executor list_files hides ignored",
          "node_modules" not in client.call("list_files", {}))
    check("executor rejects disallowed tool",
          "not allowed" in client.call("ssh", {"host": "h", "cmd": "x"}))
finally:
    client.close()
shutil.rmtree(exdir, ignore_errors=True)

# shadowed duplicate across the [bundled, user] search path: an IDENTICAL copy in both
# dirs is skipped SILENTLY (the mirrored-box case - no noise); a DIFFERENT file under a
# name already loaded is still flagged. Guards the dup-provider/tool warning quieting.
_sa = Path(tempfile.mkdtemp(prefix="lc_shadow_a_")); _sb = Path(tempfile.mkdtemp(prefix="lc_shadow_b_"))
_TOOLSRC = ('TOOL = {"name": "dup", "description": "d", "safe": True,\n'
            '        "parameters": {"type": "object", "properties": {}}}\n'
            'def run(args, cwd):\n    return "%s"\n')
(_sa / "dup.py").write_text(_TOOLSRC % "A")
(_sb / "dup.py").write_text(_TOOLSRC % "A")           # byte-identical
_io_sh = io.StringIO()
with contextlib.redirect_stdout(_io_sh):
    _mid = lc.LeanToolManager([str(_sa), str(_sb)], enabled=["dup"])
check("identical shadowed copy across dirs loads once, SILENTLY",
      _mid.names() == ["dup"] and "already loaded" not in _io_sh.getvalue(), _io_sh.getvalue())
(_sb / "dup.py").write_text(_TOOLSRC % "B")           # now DIFFERENT content
_io_sh2 = io.StringIO()
with contextlib.redirect_stdout(_io_sh2):
    lc.LeanToolManager([str(_sa), str(_sb)], enabled=["dup"])
check("divergent shadowed file under a taken name is still flagged",
      "already loaded" in _io_sh2.getvalue(), _io_sh2.getvalue())
shutil.rmtree(_sa, ignore_errors=True); shutil.rmtree(_sb, ignore_errors=True)

# 19b. driver_only tool flag: not pushed to the executor, not routed remotely
_sd = Path(tempfile.mkdtemp(prefix="lc_drv_"))
(_sd / "drv.py").write_text(
    'TOOL={"name":"drv","description":"d","driver_only":True,'
    '"parameters":{"type":"object","properties":{}}}\n'
    'def run(args,cwd):\n    return "ran on driver"\n')
(_sd / "reg.py").write_text(
    'TOOL={"name":"reg","description":"r",'
    '"parameters":{"type":"object","properties":{}}}\n'
    'def run(args,cwd):\n    return "ok"\n')
_dm = lc.LeanToolManager([str(_sd)], enabled=["drv", "reg"])
check("driver_only: flag captured on the tool entry", _dm.get("drv").get("driver_only") is True)
check("driver_only: absent flag defaults False", not _dm.get("reg").get("driver_only"))
_paths = [p.name for p in _dm.enabled_paths()]
check("driver_only: NOT pushed to the executor (enabled_paths)", "drv.py" not in _paths)
check("driver_only: an ordinary tool IS pushed", "reg.py" in _paths)
shutil.rmtree(_sd, ignore_errors=True)

# 19c. image tool-results: OpenAI + Gemini slices inject an image ONLY in the
# transient _cvt_messages output (never self.messages), gated on model vision.
import importlib as _il
_png = Path(tempfile.mkdtemp(prefix="lc_img_")) / "shot.png"
# 1x1 PNG (valid magic bytes so _encode_image_block's raw fallback accepts it)
_png.write_bytes(bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c6360000002000100"
    "05fe02fea7b5c0e30000000049454e44ae426082"))
_ihist = [
    {"role": "user", "content": "shot"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "c1", "function": {"name": "web_screenshot", "arguments": "{}"}}]},
    lc._tool_result_msg("web_screenshot", {"text": "captured", "image_path": str(_png)}),
]
# _tool_result_msg threads the provider tool_call_id when given, omits it otherwise
check("_tool_result_msg carries tool_call_id when set",
      lc._tool_result_msg("read_file", "ok", tool_call_id="toolu_9").get("tool_call_id") == "toolu_9")
check("_tool_result_msg omits tool_call_id when empty",
      "tool_call_id" not in lc._tool_result_msg("read_file", "ok"))

import copy as _copy
_ihist_copy = _copy.deepcopy(_ihist)
# OpenAI
_oa = _il.import_module("providers.openai")
_oa._lc["_encode_image_block"] = lc._encode_image_block
check("openai vision detection: gpt-4o True, gpt-3.5 False",
      _oa._model_supports_vision("gpt-4o") and not _oa._model_supports_vision("gpt-3.5-turbo"))
_ov = _oa._cvt_messages(_ihist, vision=True)
check("openai vision=True: emits a following user image_url message",
      any(x["role"] == "user" and isinstance(x.get("content"), list)
          and x["content"][0].get("type") == "image_url" for x in _ov))
_ont = _oa._cvt_messages(_ihist, vision=False)
check("openai vision=False: no image message, tool content stays str",
      not any(isinstance(x.get("content"), list) for x in _ont)
      and isinstance([x for x in _ont if x["role"] == "tool"][0]["content"], str))
# Gemini
_ge = _il.import_module("providers.gemini")
_ge._lc["_encode_image_block"] = lc._encode_image_block
check("gemini vision detection: gemini-2.0-flash True, embedding-001 False",
      _ge._model_supports_vision("gemini-2.0-flash") and not _ge._model_supports_vision("embedding-001"))
_, _gc = _ge._cvt_messages(_ihist, vision=True)
check("gemini vision=True: emits a following user inlineData part",
      any(any("inlineData" in p for p in x["parts"]) for x in _gc))
_, _gcn = _ge._cvt_messages(_ihist, vision=False)
check("gemini vision=False: no inlineData part",
      not any(any("inlineData" in p for p in x["parts"]) for x in _gcn))
check("image slices: self.messages history NOT mutated by conversion",
      _ihist == _ihist_copy)
shutil.rmtree(_png.parent, ignore_errors=True)
# non-string tool content (a loaded/synthesised history) must not reach the wire as a
# non-string - each bundled provider coerces it so one odd message can't 400 the call.
_badhist = [{"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "c1", "tool_name": "read_file", "content": None}]
for _pn in ("groq", "openrouter"):
    _pm = _il.import_module(f"providers.{_pn}")
    _conv = _pm._cvt_messages(_badhist)
    _tc = [x for x in _conv if x.get("role") == "tool"][0]
    check(f"{_pn}: non-string tool content coerced to str (no wire type error)",
          isinstance(_tc["content"], str))
_api_m = _il.import_module("providers.anthropic_api")
check("anthropic_api: _tool_result_content coerces None -> ''",
      _api_m._tool_result_content({"content": None}) == "")

# 20. remote SSH transport: pure builders
check("ctl_path sanitizes host + is per-PID + lives in runtime dir",
      lc._ctl_path("user@gpu.example").endswith(f"cm-user_gpu.example-{os.getpid()}.sock"))
check("reap_dead_sockets unlinks dead-PID sockets, keeps live/own",
      (lambda d: (
          (d / "cm-h-999999.sock").write_text(""),
          (d / f"cm-h-{os.getpid()}.sock").write_text(""),
          lc._reap_dead_sockets(d),
          not (d / "cm-h-999999.sock").exists()
          and (d / f"cm-h-{os.getpid()}.sock").exists(),
      )[-1])(Path(tempfile.mkdtemp())))
marg = lc._ssh_master_argv("h", "/c")
check("master argv has ControlMaster + persist=yes + keepalive + -fN",
      "ControlMaster=yes" in marg and "ControlPersist=yes" in marg
      and "ServerAliveInterval=15" in marg and "-fN" in marg)
rarg = lc._ssh_run_argv("h", "/c", "echo hi")
check("run argv reuses ControlPath + BatchMode + -T",
      "ControlPath=/c" in rarg and "BatchMode=yes" in rarg and "-T" in rarg and rarg[-1] == "echo hi")
check("run argv tty uses -tt", "-tt" in lc._ssh_run_argv("h", "/c", "x", tty=True))
check("run + exec argv carry ServerAlive keepalives (drop guard)",
      "ServerAliveInterval=15" in rarg
      and "ServerAliveInterval=15" in lc._ssh_exec_argv("h", "/c", "x", "/p"))
earg = lc._ssh_exec_argv("h", "/c", "python3 /run/lc/agent.py", "/proj dir")
check("exec argv launches the given executor --tool-exec with quoted cwd, no PTY",
      "-T" in earg and earg[-1] == "python3 /run/lc/agent.py --tool-exec --cwd '/proj dir'")
check("exec argv accepts an installed-executor command",
      lc._ssh_exec_argv("h", "/c", "lean_coder", "/p")[-1] == "lean_coder --tool-exec --cwd /p")
# tool hash-sync: push changed/new, remove de-selected, leave unchanged alone
_l2s_push, _l2s_rm = lc._tools_to_sync(
    {"a.py": "h1", "b.py": "h2", "c.py": "hNEW"},      # local enabled
    {"b.py": "h2", "c.py": "hOLD", "d.py": "hx"})      # already on remote
check("_tools_to_sync: pushes new + changed only",
      _l2s_push == ["a.py", "c.py"])                    # a missing, c changed; b unchanged skipped
check("_tools_to_sync: removes de-selected remote tools", _l2s_rm == ["d.py"])
check("_tools_to_sync: nothing to do when in sync",
      lc._tools_to_sync({"a.py": "h"}, {"a.py": "h"}) == ([], []))
check("posix_embed_asset x86_64 defaults to gnu",
      lc._posix_embed_asset("x86_64")[0].endswith("x86_64-unknown-linux-gnu-install_only.tar.gz"))
check("posix_embed_asset musl build selectable",
      lc._posix_embed_asset("x86_64", "musl")[0].endswith("x86_64-unknown-linux-musl-install_only.tar.gz"))
check("posix_embed_asset gnu != musl (different sha)",
      lc._posix_embed_asset("x86_64", "gnu")[2] != lc._posix_embed_asset("x86_64", "musl")[2])
check("posix_embed_asset maps aarch64 (arm64 alias)",
      lc._posix_embed_asset("arm64") == lc._posix_embed_asset("aarch64")
      and "aarch64" in lc._posix_embed_asset("arm64")[0])
check("posix_embed_asset normalises case/whitespace",
      lc._posix_embed_asset(" X86_64\n") == lc._posix_embed_asset("x86_64"))
check("posix_embed_asset uncovered arch -> None (caller hard-errors)",
      lc._posix_embed_asset("mips") is None)
check("posix_embed_asset uncovered libc -> None",
      lc._posix_embed_asset("x86_64", "uclibc") is None)
check("posix_embed_asset returns (fn, url, 64-hex sha256)",
      len(lc._posix_embed_asset("x86_64")[2]) == 64
      and lc._posix_embed_asset("x86_64")[1].startswith("https://github.com/astral-sh/"))

# 20b. remote tool gating + driver-side confirm (stage 3)
c0 = lc.Config()
check("active_tools: same core surface local or remote (ssh is a lean-tool now)",
      [t["function"]["name"] for t in lc.active_tools(c0, remote=True)]
      == [t["function"]["name"] for t in lc.active_tools(c0)])
check("confirm_remote_tool: read-only auto-ok", lc.confirm_remote_tool(c0, "h", "list_files", {}))
check("confirm_remote_tool: auto-approval bypass",
      lc.confirm_remote_tool(lc.Config(approval="auto"), "h", "run_command", {"cmd": "x"}))
_oa = lc._ask
lc._ask = lambda q: True
with contextlib.redirect_stdout(io.StringIO()):
    yes = lc.confirm_remote_tool(c0, "h", "run_command", {"cmd": "x"})
lc._ask = lambda q: False
with contextlib.redirect_stdout(io.StringIO()):
    no = lc.confirm_remote_tool(c0, "h", "apply_diff", {"path": "f", "diff": "d"})
lc._ask = _oa
check("confirm_remote_tool: mutating asks (yes->True, no->False)", yes is True and no is False)

# minor 2: real unified diff in remote previews when fetch() yields old content
_oa = lc._ask
lc._ask = lambda q: True
_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    lc.confirm_remote_tool(c0, "h", "write_file", {"path": "f", "content": "new\n"},
                           fetch=lambda p: "old\n")
check("remote write_file shows unified diff", "-old" in _b.getvalue() and "+new" in _b.getvalue())
_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    lc.confirm_remote_tool(c0, "h", "apply_diff",
                           {"path": "f", "diff": "<<<<<<< SEARCH\nb\n=======\nB\n>>>>>>> REPLACE"},
                           fetch=lambda p: "a\nb\nc\n")
check("remote apply_diff shows unified diff (-b/+B)", "-b" in _b.getvalue() and "+B" in _b.getvalue())
lc._ask = _oa

# raw-read helper: byte-exact prefix + clip flag (never a spliced marker)
import base64 as _b64mod
_rt = pathlib.Path(tempfile.mkdtemp())
(_rt / "x").write_bytes(b"ABCDE")
_e, _clip, _tot, _shown = lc._read_raw_b64(_rt / "x", max_bytes=3)
check("_read_raw_b64 clips with real counts (no marker)",
      _clip and _tot == 5 and _shown == 3 and _b64mod.b64decode(_e) == b"ABC")
# minor 1: the live session-env (uncached tail) names the dir tools act on (remote when
# connected) + the shell family, and never leaks "remote"/"local" to the model.
class _WS:
    host = "gpu"; cwd = "."; remote_cwd = "/srv/app"; remote_posix = True
    def read_raw(self, p): return None
agm = lc.Agent(lc.Config(cwd=FIX))
check("session-env cwd is local by default", str(FIX) in agm._session_env())
check("cached system block no longer carries the cwd", str(FIX) not in agm.messages[0]["content"])
agm.set_remote(_WS())
check("session-env cwd switches to remote dir on connect", "/srv/app" in agm._session_env())
check("session-env does not leak 'remote'/'local' to the model",
      "remote" not in agm._session_env().lower() and "local" not in agm._session_env().lower())
check("session-env states the shell family (POSIX remote)", "sh (POSIX)" in agm._session_env())
agm.set_remote(None)
check("session-env cwd returns to local on disconnect", str(FIX) in agm._session_env())

# 22. lean-tools: load, dispatch (safe vs confirm), config round-trip
plugdir = pathlib.Path(tempfile.mkdtemp(prefix="lc_plug_"))
(plugdir / "shout.py").write_text(
    'TOOL = {"name": "shout", "description": "Uppercase text.", "safe": True,\n'
    '        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},\n'
    '                       "required": ["text"]}}\n'
    'def run(args, cwd):\n    return args.get("text", "").upper()\n')
(plugdir / "danger.py").write_text(
    'TOOL = {"name": "danger", "description": "Echo.",\n'
    '        "parameters": {"type": "object", "properties": {}}}\n'
    'def run(args, cwd):\n    return "DANGER_RAN"\n')
pm = lc.LeanToolManager(str(plugdir), enabled=["shout", "danger"])
check("lean-tool loader discovers both", set(pm.names()) == {"shout", "danger"})
check("lean-tool safe flag parsed", pm.get("shout")["safe"] and not pm.get("danger")["safe"])
check("active_tools appends enabled lean-tool schemas",
      "shout" in [t["function"]["name"] for t in lc.active_tools(lc.Config(), lean_tool_schemas=pm.schemas())])
check("active_tools omits lean-tools when none enabled",
      "shout" not in [t["function"]["name"]
                      for t in lc.active_tools(lc.Config(), lean_tool_schemas=lc.LeanToolManager(str(plugdir), enabled=[]).schemas())])
# a lean-tool that reuses a CORE tool name (e.g. todo.py's update_plan) must NOT be
# added a second time - duplicate names 400 the API ("Tool names must be unique").
_dup = ({"type": "function", "function": {"name": "update_plan", "description": "x", "parameters": {}}}, True)
_names = [t["function"]["name"] for t in lc.active_tools(lc.Config(), lean_tool_schemas=[_dup])]
check("active_tools dedups a lean-tool that shadows a core tool name",
      _names.count("update_plan") == 1)

agp = lc.Agent(lc.Config(cwd=FIX, lean_tools_dir=str(plugdir), lean_tools_enabled=["shout"]))
check("agent surfaces enabled lean-tool",
      "shout" in [t["function"]["name"] for t in agp.tool_defs])
check("agent dispatches safe lean-tool (no confirm)",
      agp._run_tool("shout", {"text": "hi"}) == "HI")
agd = lc.Agent(lc.Config(cwd=FIX, lean_tools_dir=str(plugdir), lean_tools_enabled=["danger"]))
_oap = lc._ask
lc._ask = lambda q: False
with contextlib.redirect_stdout(io.StringIO()):
    pdecl = agd._run_tool("danger", {})
lc._ask = lambda q: True
with contextlib.redirect_stdout(io.StringIO()):
    pran = agd._run_tool("danger", {})
lc._ask = _oap
check("non-safe lean-tool declined -> not run", "declined" in pdecl)
check("non-safe lean-tool confirmed -> runs", pran == "DANGER_RAN")

pf = pathlib.Path(tempfile.mktemp(suffix=".toml"))
lc.CONFIG_PATH = pf
lc.save_config(lc.Config(model="m", lean_tools_enabled=["shout", "danger"]))
check("config round-trips lean_tools_enabled",
      _t.loads(pf.read_text()).get("lean_tools_enabled") == ["shout", "danger"])
lc.CONFIG_PATH = _orig_cp
pf.unlink(missing_ok=True)

# /reload: re-scan dir, pick up NEW and EDITED lean-tool code (no pyc staleness)
rdir = pathlib.Path(tempfile.mkdtemp(prefix="lc_reload_"))
agr = lc.Agent(lc.Config(cwd=FIX, lean_tools_dir=str(rdir), lean_tools_enabled=["rl"]))
check("reload: user-dir tool absent before its file exists", "rl" not in agr.reload_lean_tools())
def _wp(v):
    (rdir / "rl.py").write_text(
        'TOOL = {"name": "rl", "description": "x", "safe": True,\n'
        '        "parameters": {"type": "object", "properties": {}}}\n'
        f'def run(a, c):\n    return "{v}"\n')
_wp("ONE")
check("reload picks up a new lean-tool", "rl" in agr.reload_lean_tools())
check("reload dispatch v1", agr._run_tool("rl", {}) == "ONE")
_wp("TWO")
agr.reload_lean_tools()
check("reload picks up edited code (no pyc staleness)", agr._run_tool("rl", {}) == "TWO")
shutil.rmtree(rdir, ignore_errors=True)

# 20b. setup(lc, cfg) startup hooks + slash-command registry
hdir = pathlib.Path(tempfile.mkdtemp(prefix="lc_hooks_"))
(hdir / "hook_only.py").write_text(           # setup-only: no TOOL/run
    "def _h(agent, cfg, arg):\n"
    "    cfg.probe_arg = arg\n"
    "    cfg.probe_remote = agent.remote\n"
    "def setup(lc, cfg):\n"
    "    cfg.hooked = True\n"
    "    lc['register_command']('/probe', _h, 'probe cmd')\n")
(hdir / "tooled.py").write_text(              # TOOL + setup together
    "TOOL = {'name': 'echo_t', 'description': 'echo',\n"
    "        'parameters': {'type': 'object', 'properties': {}}}\n"
    "def run(args, cwd):\n    return 'ok'\n"
    "def setup(lc, cfg):\n    cfg.tooled_setup = True\n")

pm = lc.LeanToolManager(hdir, enabled={"hook_only", "echo_t"})
check("[hook] loads setup-only lean-tool by file stem", "hook_only" in pm.names())
check("[hook] loads TOOL+setup lean-tool by tool name", "echo_t" in pm.names())
check("[hook] setup-only adds no model tool (no schema)", "schema" not in pm.get("hook_only"))
check("[hook] schemas() excludes setup-only", len(pm.schemas()) == 1)
check("[hook] enabled_paths() excludes setup-only (driver-only, not pushed)",
      all(p.name != "hook_only.py" for p in pm.enabled_paths()))
check("[hook] setups() returns both enabled hooks",
      set(n for n, _ in pm.setups()) == {"hook_only", "echo_t"})
check("[hook] disabled hook omitted from setups()",
      [n for n, _ in lc.LeanToolManager(hdir, enabled={"echo_t"}).setups()] == ["echo_t"])

# register_command: refuses built-ins, normalizes a missing leading slash
lc._lean_tool_commands.clear()
lc.register_command("/model", lambda *a: None, "x")
check("[hook] register_command refuses a built-in (no shadow)", "/model" not in lc._lean_tool_commands)
lc.register_command("ping", lambda *a: None, "p")
check("[hook] register_command normalizes missing slash", "/ping" in lc._lean_tool_commands)
check("[hook] registered name joins SLASH_COMMANDS (completion)", "/ping" in lc.SLASH_COMMANDS)
# optional completer hook: lean-tool can Tab-complete its first arg (e.g. /update targets)
lc.register_command("/synctest", lambda *a: None, "s", completer=lambda agent, cfg: ["user@a", "dev"])
check("[hook] register_command stores a completer",
      lc._arg_completions(None, None, "/synctest") == ["user@a", "dev"])
lc.register_command("/synctest", lambda *a: None, "s")    # re-register without one
check("[hook] re-register without completer clears it",
      lc._arg_completions(None, None, "/synctest") == [])

# owner-aware collision: a DIFFERENT plugin can't shadow a taken command (first wins);
# the SAME plugin re-registering its own command still replaces.
lc._lean_tool_commands.clear(); lc._lean_tool_cmd_owner.clear()
lc._registering_owner = "toolA"; lc.register_command("/dup", lambda *a: "A", "from A")
lc._registering_owner = "toolB"; lc.register_command("/dup", lambda *a: "B", "from B")
check("register_command: a second plugin can't shadow (first wins)",
      lc._lean_tool_commands["/dup"][1] == "from A")
lc._registering_owner = "toolA"; lc.register_command("/dup", lambda *a: "A2", "from A v2")
check("register_command: same plugin re-registering replaces",
      lc._lean_tool_commands["/dup"][1] == "from A v2")
lc._registering_owner = None
lc._lean_tool_commands.clear(); lc._lean_tool_cmd_owner.clear()

# _run_lean_tool_setup drives setup(lc, cfg): mutates live cfg + registers command
lc._lean_tool_commands.clear()
scfg = lc.Config(cwd=FIX, lean_tools_dir=str(hdir), lean_tools_enabled=["hook_only"])
lc._run_lean_tool_setup(scfg)
check("[hook] _run_lean_tool_setup ran setup (live cfg mutated)", getattr(scfg, "hooked", False) is True)
check("[hook] only enabled hook ran (tooled disabled here)", not hasattr(scfg, "tooled_setup"))
check("[hook] setup registered the slash command", "/probe" in lc._lean_tool_commands)
# the registered handler dispatches with (agent, cfg, arg) - same call repl() makes
_hag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
lc._lean_tool_commands["/probe"][0](_hag, scfg, "hello world")
check("[hook] handler receives arg + agent", scfg.probe_arg == "hello world" and scfg.probe_remote is None)
lc._lean_tool_commands.clear()
shutil.rmtree(hdir, ignore_errors=True)

# 20c. render_help: command column styled distinctly from descriptions
import re as _re
_tty0 = lc._TTY
lc._TTY = True                                  # force ANSI to check styling
help_txt = lc.render_help([("/whoami", "show session info")])
plain = _re.sub(r"\x1b\[[0-9;]*m", "", help_txt)   # strip ANSI for content checks
check("render_help lists every built-in command",
      all(c in plain for c, _ in lc.HELP_COMMANDS))
check("render_help includes descriptions + footer",
      "this help" in plain and "Tab completes" in plain)
check("render_help styles the command (bold+cyan), not the description",
      lc.bold(lc.cyan("/clear")) in help_txt and lc.dim("this help") in help_txt)
check("render_help shows lean-tool commands under their own header",
      "lean-tool commands:" in plain and "/whoami" in plain)
# regression: one very long command (/provider) must NOT pad every row out to its
# width (that wrapped the whole menu). Measure the GAP between a short command and
# its description - it must be capped, not stretched to the longest command's length.
_short = [ln for ln in plain.splitlines() if ln.lstrip().startswith("/clear ")][0]
_gap = len(_short[_short.index("/clear")+len("/clear"):]) - len(_short[_short.index("/clear")+len("/clear"):].lstrip())
check("render_help caps the alignment gap (long command doesn't blow up the table)",
      _gap <= 24)
lc._TTY = _tty0

# --- /prompt: file-backed prompts (system/ subdir + custom root) ---
import tempfile as _tf, pathlib as _pl
_pdir = _pl.Path(_tf.mkdtemp())
_orig_pd, _orig_spd = lc.PROMPTS_DIR, lc.SYSTEM_PROMPTS_DIR
lc.PROMPTS_DIR, lc.SYSTEM_PROMPTS_DIR = _pdir, _pdir / "system"
check("prompt_file: built-in routes to system/ subdir",
      lc.prompt_file("system").parent == lc.SYSTEM_PROMPTS_DIR)
check("prompt_file: custom routes to prompts root",
      lc.prompt_file("research").parent == lc.PROMPTS_DIR)
check("read_prompt: built-in falls back to baked default (no override)",
      lc.read_prompt("system") == lc.SYSTEM_PROMPT)
check("read_prompt: unknown custom -> None", lc.read_prompt("nope") is None)
_pf = lc.seed_prompt_file("system")
check("seed_prompt_file: creates override seeded with the default",
      _pf.is_file() and _pf.read_text() == lc.SYSTEM_PROMPT)
_pf.write_text("CUSTOM SYS")
check("read_prompt: override file wins over default", lc.read_prompt("system") == "CUSTOM SYS")
check("restore_prompt: removes override -> back to default",
      lc.restore_prompt("system") is True and lc.read_prompt("system") == lc.SYSTEM_PROMPT)
check("restore_prompt: nothing to restore -> False", lc.restore_prompt("system") is False)
(lc.PROMPTS_DIR / "research.txt").write_text("do research")
_pb, _pc = lc.list_prompts()
check("list_prompts: default hides un-overridden system prompts, lists custom",
      "system" not in _pb and "handover" not in _pb and "research" in _pc)
_pba, _pca = lc.list_prompts(all_builtins=True)
check("list_prompts(all_builtins=True): every system prompt listed",
      "system" in _pba and "handover" in _pba and "research" in _pca)
lc.seed_prompt_file("handover")   # override one -> it surfaces in the default list
_pb2, _ = lc.list_prompts()
check("list_prompts: an overridden system prompt IS surfaced",
      "handover" in _pb2 and "system" not in _pb2)
lc.restore_prompt("handover")
check("open_in_editor: no editor/TTY -> False (graceful)", lc.open_in_editor(_pf) is False)
lc.PROMPTS_DIR, lc.SYSTEM_PROMPTS_DIR = _orig_pd, _orig_spd

# --- command canon: /clear (was /reset), /prompt; rot-prone aliases removed ---
check("/clear advertised; /reset removed entirely (no alias)",
      "/clear" in lc.SLASH_COMMANDS and "/reset" not in lc.SLASH_COMMANDS
      and "/reset" not in lc._BUILTIN_COMMANDS)
check("/prompt advertised", "/prompt" in lc.SLASH_COMMANDS)
check("/nothink removed entirely (use /think off)",
      "/nothink" not in lc.SLASH_COMMANDS and "/nothink" not in lc._BUILTIN_COMMANDS)
_stubA = type("A", (), {"active_provider": lambda self: None})()
check("/think completion offers values for the no-arg menu",
      "off" in lc._arg_completions(_stubA, lc.Config(), "/think"))
check("/effort completion offers values for the no-arg menu",
      len(lc._arg_completions(_stubA, lc.Config(), "/effort")) > 0)

# 20d. client-abstraction hooks for lean-tools that swap the client
class _FakeClient:
    def __init__(self, ctx=None, uncapped=False, models=()):
        self._ctx = ctx
        self.uncapped_ctx = uncapped
        self._models = list(models)
    def detect_num_ctx(self): return self._ctx
    def list_models(self): return self._models
    def running_models(self): return []

# #1 the OOM-cap behaviour now lives in the ollama provider's context_window()
# (the old detect_and_cap is gone). Monkeypatch the client's /api/show so the
# detection path runs without a network.
_orig_getjson = olm.OllamaClient._get_json
def _fake_show(window):
    return lambda self, path, body=None, timeout=10: {"model_info": {"arch.context_length": window}}
olm._cfg = lc.Config(auto_num_ctx=True, num_ctx=32768)
olm.OllamaClient._get_json = _fake_show(200000)
cw = olm.PROVIDER["context_window"]("any")
check("context_window: big model max capped to AUTO_CTX_CAP",
      cw == olm.AUTO_CTX_CAP and olm._cfg._ollama_capped_from == 200000)
olm._cfg = lc.Config(auto_num_ctx=True, num_ctx=32768)
olm.OllamaClient._get_json = _fake_show(16384)
check("context_window: smaller window used as-is",
      olm.PROVIDER["context_window"]("any") == 16384 and olm._cfg._ollama_capped_from is None)
olm._cfg = lc.Config(auto_num_ctx=False, num_ctx=8192)
check("context_window: explicitly-pinned num_ctx untouched",
      olm.PROVIDER["context_window"]("any") == 8192)
olm.OllamaClient._get_json = _orig_getjson
olm._cfg = lc.Config(cwd=FIX)        # restore for any later provider-module test

# #2 Agent.use_client: swaps client, refreshes num_ctx via the active provider's
# context_window(), clears stale estimate. Uses a registered fake provider.
lc.register_provider({
    "name": "fake_uc",
    "make_client": lambda c: _FakeClient(ctx=128000),
    "list_models": lambda: ["m"],
    "context_window": lambda m: 128000,
})
uag = lc.Agent(lc.Config(cwd=FIX, provider="fake_uc"))
uag.cfg.provider_settings.setdefault("fake_uc", {})["model"] = "m"
uag.last_prompt_tokens = 123
ufc = _FakeClient(ctx=128000, uncapped=True)
uag.use_client(ufc)
check("use_client swaps the active client", uag.client is ufc)
check("use_client refreshes num_ctx via provider context_window",
      uag.cfg.provider_settings["fake_uc"]["num_ctx"] == 128000)
check("use_client clears the stale token estimate", uag.last_prompt_tokens is None)

# #3 pin_model: survives the per-host [models] override
pm = lc.Config(host="http://h:11434", model="orig", host_models={"http://h:11434": "hostdefault"})
pm.pin_model("plugmodel")
check("pin_model sets model + marks explicit", pm.model == "plugmodel" and pm.model_explicit is True)
olm.apply_host_default_model(_FakeClient(models=[]), pm)   # respect_explicit=True (default)
check("pin_model survives apply_host_default_model", pm.model == "plugmodel")
pm2 = lc.Config(host="http://h:11434", model="orig", host_models={"http://h:11434": "hostdefault"})
olm.apply_host_default_model(_FakeClient(models=[]), pm2)
check("without pin, host default IS adopted (shows the clobber it prevents)",
      pm2.model == "hostdefault")

# #4 setup() failure: contained, pre-throw commands survive, traceback under debug
import os as _os, io as _io, contextlib as _ctx
sfdir = pathlib.Path(tempfile.mkdtemp(prefix="lc_setupfail_"))
(sfdir / "boom.py").write_text(
    "def setup(lc, cfg):\n"
    "    lc['register_command']('/early', lambda a, c, r: None, 'before the throw')\n"
    "    raise ValueError('boom in setup')\n")
lc._lean_tool_commands.clear()
bcfg = lc.Config(cwd=FIX, lean_tools_dir=str(sfdir), lean_tools_enabled=["boom"])
_prev_dbg = _os.environ.pop("LEANCODER_DEBUG", None)
buf = _io.StringIO()
with _ctx.redirect_stdout(buf):
    lc._run_lean_tool_setup(bcfg)        # must not raise
out = buf.getvalue()
check("setup failure is contained + reported", "boom in setup" in out)
check("setup: command registered before the throw survives", "/early" in lc._lean_tool_commands)
check("setup: points at LEANCODER_DEBUG when off", "LEANCODER_DEBUG" in out)
_os.environ["LEANCODER_DEBUG"] = "1"
buf2 = _io.StringIO()
with _ctx.redirect_stdout(buf2):
    lc._run_lean_tool_setup(bcfg)
check("setup: full traceback under LEANCODER_DEBUG",
      "Traceback (most recent call last)" in buf2.getvalue())
if _prev_dbg is None:
    _os.environ.pop("LEANCODER_DEBUG", None)
else:
    _os.environ["LEANCODER_DEBUG"] = _prev_dbg
lc._lean_tool_commands.clear()
shutil.rmtree(sfdir, ignore_errors=True)

# 20z. REGRESSION (remote executor must ship builtins.py beside the pushed agent.py).
# The single-file push is NOT self-contained: core imports the bundled builtins at load
# (_load_builtins). A pushed agent.py with no sibling lean-tools/builtins.py dies at
# import -> the executor closes before the handshake ("executor closed before
# responding"). RemoteWorkspace._push_builtins ships it; this guards both ends ssh-free.
import subprocess as _subp_be
_bedir = Path(tempfile.mkdtemp(prefix="lc_be_"))
(_bedir / "agent.py").write_bytes(Path(lc.__file__).read_bytes())
_r1 = _subp_be.run([sys.executable, str(_bedir / "agent.py"), "--tool-exec",
                    "--cwd", str(_bedir)], stdin=_subp_be.DEVNULL,
                   capture_output=True, text=True, timeout=20)
check("executor without bundled builtins.py fails to start (the connect bug)",
      _r1.returncode != 0 and "builtin tools missing" in _r1.stderr, _r1.stderr[-120:])
(_bedir / "lean-tools").mkdir()
(_bedir / "lean-tools" / "builtins.py").write_bytes(
    (Path(lc.__file__).parent / "lean-tools" / "builtins.py").read_bytes())
_r2 = _subp_be.run([sys.executable, str(_bedir / "agent.py"), "--tool-exec",
                    "--cwd", str(_bedir)], stdin=_subp_be.DEVNULL,
                   capture_output=True, text=True, timeout=20)
check("executor WITH builtins.py beside agent.py handshakes ({ready})",
      _r2.returncode == 0 and '"ready": true' in _r2.stdout, _r2.stderr[-120:])
shutil.rmtree(_bedir, ignore_errors=True)

# 21. RemoteWorkspace end-to-end via a fake `ssh` shim (runs the "remote" locally)
shimdir = pathlib.Path(tempfile.mkdtemp(prefix="lc_ssh_"))
(shimdir / "ssh").write_text(
    "#!/bin/bash\n"
    'for a in "$@"; do [ "$a" = "-fN" ] && exit 0; done\n'      # master: succeed
    'case "$*" in *"-O exit"*) exit 0;; esac\n'                  # close master
    'exec bash -c "${@: -1}"\n')                                 # else run trailing cmd locally
(shimdir / "ssh").chmod(0o755)
proj = pathlib.Path(tempfile.mkdtemp(prefix="lc_proj_"))
(proj / "hello.py").write_text("print('hi')\n")
# a file well past OUTPUT_MAX_CHARS (6000), with a marker line beyond that point
BIG = "\n".join("x" * 20 for _ in range(400)) + "\nMARKER_PAST_6K\n"
(proj / "big.txt").write_text(BIG)
_oldpath = os.environ.get("PATH", "")
_oldxdg = os.environ.pop("XDG_RUNTIME_DIR", None)   # force /tmp paths for determinism
os.environ["PATH"] = str(shimdir) + os.pathsep + _oldpath
try:
    ws = lc.RemoteWorkspace("fakehost", str(proj)).connect()
    check("remote connect resolves a remote dir", bool(ws.remote_dir), str(ws.remote_dir))
    check("remote list_files round-trips", "hello.py" in ws.call("list_files", {}))
    check("remote read_file round-trips", "print('hi')" in ws.call("read_file", {"path": "hello.py"}))
    check("remote run_command round-trips", "ok" in ws.call("run_command", {"cmd": "echo ok"}))
    # stage 3: route an Agent's tools through the remote workspace
    ag = lc.Agent(lc.Config(cwd=FIX, approval="ask"))
    ag.set_remote(ws)
    check("agent remote mode: no ssh in surface (it's an opt-in lean-tool now)",
          "ssh" not in [t["function"]["name"] for t in ag.tool_defs])
    check("session-env picks up the remote cwd (handshake)",
          bool(ws.remote_cwd) and ws.remote_cwd in ag._session_env())
    check("remote read_raw fetches file content",
          "print('hi')" in (ws.read_raw("hello.py") or ""))
    check("router: read-only runs remote (no confirm)",
          "print('hi')" in ag._run_tool("read_file", {"path": "hello.py"}))
    _oa2 = lc._ask
    lc._ask = lambda q: True
    with contextlib.redirect_stdout(io.StringIO()):
        rw = ag._run_tool("write_file", {"path": "made.txt", "content": "yo\n"})
    check("router: confirmed remote write executes + tracked",
          (proj / "made.txt").is_file() and "made.txt" in ag.tools.changed_files, rw)
    lc._ask = lambda q: False
    with contextlib.redirect_stdout(io.StringIO()):
        rd = ag._run_tool("run_command", {"cmd": "echo nope"})
    lc._ask = _oa2
    check("router: declined remote command does not run", "declined" in rd)
    # byte-exact remote fetch for previews (the >6KB fidelity fix)
    got = ws.read_raw("big.txt")
    check("read_raw is byte-exact past 6KB (no truncation marker)",
          got == BIG and "[truncated" not in got and got.endswith("\n"),
          f"len={len(got) if got else None} vs {len(BIG)}")
    check("read_raw missing file -> None", ws.read_raw("nope.txt") is None)
    # apply_diff preview against a SEARCH block PAST the old 6KB boundary
    lc._ask = lambda q: True
    with contextlib.redirect_stdout(io.StringIO()) as _pb:
        lc.confirm_remote_tool(lc.Config(), ws.host, "apply_diff",
            {"path": "big.txt",
             "diff": "<<<<<<< SEARCH\nMARKER_PAST_6K\n=======\nMARKER_REPLACED\n>>>>>>> REPLACE"},
            fetch=ws.read_raw)
    lc._ask = _oa2
    check("apply_diff preview diffs full file past 6KB boundary",
          "-MARKER_PAST_6K" in _pb.getvalue() and "+MARKER_REPLACED" in _pb.getvalue())
    ws.close()
    # lean-tools pushed to + dispatched on the remote executor
    wsp = lc.RemoteWorkspace("fakehost", str(proj),
                             lean_tool_paths=[str(plugdir / "shout.py")]).connect()
    check("remote lean-tool dispatch round-trips",
          wsp.call("shout", {"text": "hey"}) == "HEY")
    # /reload over the same (fake) master: relaunch executor, still works
    wsp.reload([str(plugdir / "shout.py")])
    check("remote reload relaunches executor (session kept)",
          wsp.call("shout", {"text": "yo"}) == "YO"
          and "hello.py" in wsp.call("list_files", {}))
    wsp.close()
except Exception as e:
    check("remote workspace e2e (fake ssh)", False, f"{type(e).__name__}: {e}")
finally:
    os.environ["PATH"] = _oldpath
    if _oldxdg is not None:
        os.environ["XDG_RUNTIME_DIR"] = _oldxdg
shutil.rmtree(shimdir, ignore_errors=True)
shutil.rmtree(proj, ignore_errors=True)
shutil.rmtree(plugdir, ignore_errors=True)

# 22b. Windows transport branches (remote_posix=False): the remote file helpers must
# emit PowerShell, not POSIX sh. We drive a RemoteWorkspace whose _run captures the
# command string (and returns canned output) instead of touching ssh - so the branch
# selection is asserted offline, before a real Win11 box exists.
_wsw = lc.RemoteWorkspace.__new__(lc.RemoteWorkspace)
_wsw.host = "winbox"; _wsw.ctl = "/tmp/x.sock"; _wsw.remote_posix = False
_wsw.remote_dir = r"C:\Users\me\AppData\Local\Temp\leancoder"
_wcaps = []
# Simulate the Windows box for the scp push: capture the scp argv, "receive" the file
# by reading the local temp file scp would have sent, and answer Get-FileHash with the
# sha256 of those bytes so the end-to-end verify passes.
_wstate = {"files": {}}
def _wsw_run(cmd, tty=False, _caps=_wcaps):
    import re as _re
    _caps.append(cmd)
    if "New-Item" in cmd and "Directory" in cmd and "agent.py" not in cmd:
        return 0, _wsw.remote_dir, ""
    if "Get-FileHash" in cmd:
        m = _re.search(r'Test-Path "([^"]*)"', cmd)
        data = _wstate["files"].get(m.group(1)) if m else None
        return (0, lc.hashlib.sha256(data).hexdigest().upper(), "") if data is not None else (0, "", "")
    return 0, "", ""
_wsw._run = _wsw_run
_wwrites = []
_orig_ssh_run = lc.subprocess.run
def _cap_ssh_run(argv, *a, **k):
    # scp argv is ["scp", ..., <localtmp>, "winbox:C:/.../big"]: emulate the transfer by
    # reading the local temp file and recording it under the (backslash) remote path.
    _wwrites.append((argv, k.get("input")))
    if argv and argv[0] == "scp":
        localtmp = argv[-2]
        remote = argv[-1].split(":", 1)[1].replace("/", "\\")
        try:
            _wstate["files"][remote] = open(localtmp, "rb").read()
        except OSError:
            pass
    class _R:  # noqa
        returncode = 0; stdout = b""; stderr = b""
    return _R()
# _remote_write / push go through subprocess.run(_scp_argv(...)); capture those
lc.subprocess.run = _cap_ssh_run
try:
    _wsw._remote_mkdir(r"C:\tmp\d")
    check("win: _remote_mkdir emits PowerShell New-Item",
          "New-Item -ItemType Directory" in _wcaps[-1] and "mkdir" not in _wcaps[-1])
    _wsw._remote_sha256(r"C:\tmp\f")
    check("win: _remote_sha256 emits Get-FileHash SHA256",
          "Get-FileHash -Algorithm SHA256" in _wcaps[-1] and "sha256sum" not in _wcaps[-1])
    check("win: _remote_sep is backslash", _wsw._remote_sep() == "\\")
    _wsw._remote_write(r"C:\tmp\big", b"data" * 20000)
    check("win: _remote_write pushes via scp (no stdin, no cat >, no Add-Content)",
          any(a[0] and a[0][0] == "scp" for a in _wwrites)
          and all("cat >" not in c for c in _wcaps)
          and all("Add-Content" not in c for c in _wcaps))
    check("win: scp push reconstructs bytes end-to-end (sha256 verified)",
          _wstate["files"].get(r"C:\tmp\big") == b"data" * 20000)
    check("win: scp argv reuses the ControlPath + BatchMode",
          any("BatchMode=yes" in a[0] and any("ControlPath" in x for x in a[0])
              for a in _wwrites if a[0] and a[0][0] == "scp"))
    _wsw.win_python = "py -3"
    _ex, _rd = _wsw._reuse_executor_win("deadbeef")
    check("win: _reuse_executor_win probes agent.py via Test-Path + py",
          any("Test-Path" in c and "--version" in c for c in _wcaps))
finally:
    lc.subprocess.run = _orig_ssh_run
# _check_windows_python: a real interpreter path passes; a bare stub fails
_wsp = lc.RemoteWorkspace.__new__(lc.RemoteWorkspace)
_wsp.host = "winbox"
_wsp._run = lambda cmd, tty=False: (0, r"C:\Python312\python.exe", "")   # real path back
_wsp._check_windows_python()
check("win: _check_windows_python accepts a real interpreter", getattr(_wsp, "win_python", None) in ("py -3", "python"))
_wsp2 = lc.RemoteWorkspace.__new__(lc.RemoteWorkspace)
_wsp2.host = "winbox"
_wsp2._run = lambda cmd, tty=False: (0, "", "")                          # no interpreter
try:
    _wsp2._check_windows_python()
    check("win: _check_windows_python errors when Python absent", False)
except ConnectionError as _e:
    check("win: _check_windows_python errors when Python absent", "no Python" in str(_e))

# 22c. Regression cover for the Windows live-connect fixes (all found on 192.0.2.16):
# (i) _kill_tree default sig must NOT be signal.SIGKILL (absent on Windows -> import
#     crash). Resolves to SIGTERM when SIGKILL is missing; here (POSIX) SIGKILL exists.
import inspect as _inspect
check("win: _kill_tree default sig is a None sentinel (no import-time SIGKILL)",
      _inspect.signature(lc._kill_tree).parameters["sig"].default is None)
# (ii) _win_quote makes an EMPTY arg a real '' token (shlex.quote drops it, so PowerShell
#      swallowed the next flag -> the empty-cwd argparse crash that killed the executor).
check("win: _win_quote('') -> real empty quoted token", lc._win_quote("") == "''")
check("win: _win_quote escapes a literal quote by doubling", lc._win_quote("a'b") == "'a''b'")
# Windows commands are now wrapped as `powershell -NoProfile -NonInteractive
# -EncodedCommand <base64-UTF16LE>` (shell-agnostic: runs under cmd.exe OR PowerShell,
# so a stock box whose sshd DefaultShell is cmd.exe no longer fails every command).
# Decode the payload back to the PowerShell text so the branch asserts still inspect it.
import base64 as _b64
def _decode_ps(s):
    """Return the decoded PowerShell text if s is an -EncodedCommand wrapper, else s."""
    m = _re.search(r'-EncodedCommand\s+([A-Za-z0-9+/=]+)', s)
    return _b64.b64decode(m.group(1)).decode("utf-16-le") if m else s
check("win: _win_ps wraps as base64 -EncodedCommand (shell-agnostic)",
      "-EncodedCommand" in lc._win_ps("Write-Output hi")
      and _decode_ps(lc._win_ps("Write-Output hi")) == "Write-Output hi")
# (iii) _ssh_exec_argv quotes per-OS: POSIX shlex vs Windows PowerShell (_win_quote), and
#       a None/empty cwd becomes '.' not a dropped token. (Windows: inside the wrapper.)
_ewin = _decode_ps(lc._ssh_exec_argv("h", None, 'python "a.py"', None, posix=False)[-1])
check("win: _ssh_exec_argv (posix=False) uses '.' for a None cwd, PowerShell-quoted",
      "--cwd '.'" in _ewin and "--cwd ''" not in _ewin)
_eposix = lc._ssh_exec_argv("h", "/c", "python3 agent.py", "/p", posix=True)[-1]
check("win: _ssh_exec_argv (posix=True) keeps POSIX shlex quoting", "--cwd /p" in _eposix)
# (iv) _wipe_remote basename guard is separator-agnostic: a backslash Windows remote_dir
#      whose leaf is 'leancoder' must ACTUALLY wipe (posixpath.basename returned the whole
#      string -> early return -> the pushed dir leaked every session). Capture the emitted
#      command; a non-leancoder leaf must still be refused.
_wr = lc.RemoteWorkspace.__new__(lc.RemoteWorkspace)
_wr.host = "winbox"; _wr.ctl = None; _wr.remote_posix = False; _wr.reused = None
_wr.remote_dir = r"C:\Users\me\AppData\Local\Temp\leancoder"
_wcmd = {}
def _cap_wipe(argv, *a, **k):
    _wcmd["cmd"] = argv[-1]
    class _R:  # noqa
        returncode = 0; stdout = b""; stderr = b""
    return _R()
_orig_wipe_run = lc.subprocess.run
lc.subprocess.run = _cap_wipe
try:
    _wr._wipe_remote()
    _wcmd_ps = _decode_ps(_wcmd.get("cmd", ""))
    check("win: _wipe_remote runs on a backslash 'leancoder' path (guard not early-return)",
          bool(_wcmd) and "Remove-Item" in _wcmd_ps and ".watchdog.pid" in _wcmd_ps)
    _wcmd.clear()
    _wr.remote_dir = r"C:\Users\me\AppData\Local\Temp\notours"
    _wr._wipe_remote()
    check("win: _wipe_remote REFUSES a non-leancoder leaf (safety guard holds)", not _wcmd)
finally:
    lc.subprocess.run = _orig_wipe_run

# 23. install.sh --remote: stage + run the installer on another box over SSH
# 23. install.sh --remote: stage + run the installer on another box over SSH
import subprocess as _subp
INSTALL = str(Path(lc.__file__).parent / "install.sh")

# (a) dry-run: prints the plan and forwards the other flags, needs no ssh
dr = _subp.run(["bash", INSTALL, "--remote", "user@box", "--with-ollama", "--dry-run"],
               capture_output=True, text=True)
check("install --remote --dry-run: plans + forwards flags (no ssh)",
      dr.returncode == 0 and "Remote install -> user@box" in dr.stdout
      and "--with-ollama --dry-run" in dr.stdout, dr.stdout[-160:])

# (b) end-to-end through a fake `ssh` that runs the "remote" side locally
rsh = pathlib.Path(tempfile.mkdtemp(prefix="lc_rinstall_"))
(rsh / "ssh").write_text(                              # fake ssh: run the trailing cmd locally
    '#!/bin/bash\n'
    'for a in "$@"; do [ "$a" = "-fN" ] && exit 0; done\n'   # ControlMaster open: succeed
    'case "$*" in *"-O exit"*) exit 0;; esac\n'              # ControlMaster close
    'exec bash -c "${@: -1}"\n')
(rsh / "ssh").chmod(0o755)
rhome = pathlib.Path(tempfile.mkdtemp(prefix="lc_rhome_"))
renv = dict(os.environ, PATH=f"{rsh}:" + os.environ["PATH"], HOME=str(rhome))
ee = _subp.run(["bash", INSTALL, "--remote", "fakehost",
                "--dir", str(rhome / "tools"), "--bin", str(rhome / "bin"),
                "--no-ollama", "--no-expose", "-y"],
               capture_output=True, text=True, env=renv)
link = rhome / "bin" / "lean_coder"
check("install --remote end-to-end: stages + installs on the 'remote'",
      ee.returncode == 0 and link.is_symlink(),
      (ee.stdout + ee.stderr)[-200:])
shutil.rmtree(rsh, ignore_errors=True)
shutil.rmtree(rhome, ignore_errors=True)

# 24. provision.py example lean-tool: pure helpers + command registration
import tomllib as _tl
prov = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/provision.py")

# _build_config: single named host -> host=name + [machines]; valid TOML
t1 = _tl.loads(prov._build_config("m", [("gpu", "http://g:11434")], ["word_count"]))
check("provision._build_config: single host + lean-tool + machine",
      t1.get("host") == "gpu" and t1.get("model") == "m"
      and t1.get("lean_tools_enabled") == ["word_count"]
      and t1.get("machines", {}).get("gpu") == "http://g:11434", str(t1))
# multiple addresses -> tiered hosts by name
t2 = _tl.loads(prov._build_config("m", [("gpu", "http://g:11434"), ("mini", "http://h:11434")], []))
check("provision._build_config: multiple -> hosts[] by name",
      t2.get("hosts") == ["gpu", "mini"] and len(t2.get("machines", {})) == 2, str(t2))
# an unnamed address (name == url) -> bare host, no [machines]
t3 = _tl.loads(prov._build_config("m", [("http://x:11434", "http://x:11434")], []))
check("provision._build_config: unnamed addr -> bare host, no machines table",
      t3.get("host") == "http://x:11434" and "machines" not in t3, str(t3))

# _multiselect: toggle by number, blank confirms; q cancels
_seq = iter(["1 3", ""])
ms = prov._multiselect("pick:", [("a", "a"), ("b", "b"), ("c", "c")], prompt=lambda _: next(_seq))
check("provision._multiselect: toggles selected keys", ms == {"a", "c"}, str(ms))
check("provision._multiselect: q cancels -> None",
      prov._multiselect("pick:", [("a", "a")], prompt=lambda _: "q") is None)

# _lan_ips: best-effort, never raises, returns a list
check("provision._lan_ips: returns a list (best-effort)", isinstance(prov._lan_ips(), list))

# setup registers the /provision command (shows in /help)
lc._lean_tool_commands.clear()
prov.setup(vars(lc), lc.Config(cwd=FIX))
check("provision.setup registers /provision", "/provision" in lc._lean_tool_commands)
lc._lean_tool_commands.clear()

# 25. Personal provider-plugin tests (hosted APIs) live in a local-only
# companion that ships with canon but NOT the public repo. Run it if present;
# a public checkout without the file simply skips these checks.
_local_suite = Path(__file__).with_name("_smoketest_local.py")  # local-only companion, lives beside this file in tests/
if _local_suite.is_file():
    exec(compile(_local_suite.read_text(), str(_local_suite), "exec"), globals())

# 26. diagnostics.py example lean-tool (pure logic). (todo.py removed - its tool
# name `update_plan` collides with the core builtin; core supersedes it.)
diag = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/diagnostics.py")
check("diagnostics.pick_linter: prefers ruff for .py",
      diag.pick_linter(".py", False, which=lambda b: b == "ruff") == ["ruff", "check"])
check("diagnostics.pick_linter: falls back to pyflakes when its module is importable",
      diag.pick_linter(".py", False, which=lambda b: False,
                       have_module=lambda m: m == "pyflakes")[-1] == "pyflakes")
# the bug: pyflakes module absent must skip it (not pick python-m-pyflakes and fail
# at runtime) and fall through to py_compile, which is always importable
check("diagnostics.pick_linter: pyflakes absent -> falls through to py_compile",
      diag.pick_linter(".py", False, which=lambda b: False,
                       have_module=lambda m: m == "py_compile")[-1] == "py_compile")
check("diagnostics.pick_linter: nothing available -> None",
      diag.pick_linter(".py", False, which=lambda b: False, have_module=lambda m: False) is None)
check("diagnostics.pick_linter: unknown ext -> None",
      diag.pick_linter(".xyz", False, which=lambda b: True) is None)
check("diagnostics.pick_linter: dir uses dir linters",
      diag.pick_linter("", True, which=lambda b: b == "ruff") == ["ruff", "check"])
check("diagnostics.run: missing path -> error",
      "not found" in diag.run({"path": "no_such_file.py"}, str(FIX)))
check("diagnostics TOOL is safe (read-only)", diag.TOOL.get("safe") is True)

# 26b. web_fetch: render (HTML->text+links, prefers <main>), find, paging, DDG search
_wf = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/web_fetch.py")
_html = ('<html><head><style>a{}</style></head><body>'
         '<nav><a href="/skip">Nav</a></nav>'
         '<main><h2>Title</h2><p>Body <a href="https://e.com/p">link</a> here.</p>'
         '<script>bad()</script></main></body></html>')
_txt, _links = _wf.render(_html, "https://e.com/")
check("web_fetch.render keeps visible text + heading prefix",
      "## Title" in _txt and "Body" in _txt)
check("web_fetch.render keeps links inline [text](url)",
      "[link](https://e.com/p)" in _txt)
check("web_fetch.render drops scripts/styles/tags",
      "bad()" not in _txt and "<script" not in _txt and "a{}" not in _txt)
check("web_fetch.render prefers <main> (nav dropped)",
      "Nav" not in _txt and ("link", "https://e.com/p") in _links and len(_links) == 1)
_findr = _wf.apply_find("l0\nl1\nl2\nbeta here\nl4\nl5\nFARAWAY\nl7", "beta")
check("web_fetch.apply_find returns only matching section + context",
      "beta here" in _findr and "FARAWAY" not in _findr)
check("web_fetch.apply_find no match -> note",
      "no sections match" in _wf.apply_find("a\nb", "zzz"))
_c, _tot, _more, _st = _wf.window("z" * 100_000, 0)
check("web_fetch.window pages a long doc", len(_c) == _wf.MAX_TEXT and _tot == 100_000 and _more)
_c2, _t2, _m2, _s2 = _wf.window("z" * 100_000, _wf.MAX_TEXT)
check("web_fetch.window honors start offset", _s2 == _wf.MAX_TEXT and not _m2)
_c3, _t3, _m3, _s3 = _wf.window("z" * 100_000, 0, 5_000)
check("web_fetch.window honors a custom limit", len(_c3) == 5_000 and _m3)
# char_budget: sizes the read to FREE context (MAX - USED), which the core publishes
# in env before each dispatch. free/pct derive from the two values.
import os as _os
for _k in (_wf.CTX_MAX_ENV, _wf.CTX_USED_ENV):
    _os.environ.pop(_k, None)
check("web_fetch.char_budget: env unset -> MAX_TEXT", _wf.char_budget() == _wf.MAX_TEXT)
check("web_fetch.free_tokens: unset -> None", _wf.free_tokens() is None)
check("web_fetch.is_minimal: unset -> False", _wf.is_minimal() is False)
_os.environ[_wf.CTX_MAX_ENV] = "16400"; _os.environ[_wf.CTX_USED_ENV] = "2000"
check("web_fetch.free_tokens: MAX - USED", _wf.free_tokens() == 14400)
check("web_fetch.char_budget: fraction of FREE window",
      _wf.char_budget() == int(14_400 * _wf.CHARS_PER_TOKEN * _wf.CTX_FRACTION))
_os.environ[_wf.CTX_MAX_ENV] = "200000"; _os.environ[_wf.CTX_USED_ENV] = "0"
check("web_fetch.char_budget: big free -> capped at MAX_TEXT", _wf.char_budget() == _wf.MAX_TEXT)
_os.environ[_wf.CTX_MAX_ENV] = "16000"; _os.environ[_wf.CTX_USED_ENV] = "15500"
check("web_fetch.char_budget: little free -> floored at MIN_TEXT", _wf.char_budget() == _wf.MIN_TEXT)
check("web_fetch.is_minimal: little free -> True", _wf.is_minimal() is True)
_os.environ[_wf.CTX_USED_ENV] = "99999"   # used > max -> free clamps to 0
check("web_fetch.free_tokens: never negative", _wf.free_tokens() == 0)
_os.environ[_wf.CTX_MAX_ENV] = "garbage"; _os.environ.pop(_wf.CTX_USED_ENV, None)
check("web_fetch.char_budget: bad env -> MAX_TEXT", _wf.char_budget() == _wf.MAX_TEXT)
for _k in (_wf.CTX_MAX_ENV, _wf.CTX_USED_ENV):
    _os.environ.pop(_k, None)   # restore (no leakage into later tests)
check("web_fetch.format_links empty -> note", _wf.format_links([]) == "(no links found)")
# web_fetch is read-only now (search is the separate brave_search tool)
check("web_fetch.normalize_url: real url passes through",
      _wf.normalize_url("https://x.com/a") == "https://x.com/a")
check("web_fetch.normalize_url: bare domain gets https://",
      _wf.normalize_url("example.com/a") == "https://example.com/a")
check("web_fetch.normalize_url: a phrase is not a url -> None",
      _wf.normalize_url("python json indent") is None)
check("web_fetch.normalize_url: non-http scheme -> None (not fetched)",
      _wf.normalize_url("ftp://x/y") is None and _wf.normalize_url("file:///etc/passwd") is None)
check("web_fetch.normalize_url: empty -> None", _wf.normalize_url("") is None)

# brave_search tool: result formatting (pure) + no-key guidance
_bs = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/brave_search.py")
_bres = _bs.format_results({"web": {"results": [
    {"title": "T1", "url": "https://a", "description": "snip one"},
    {"title": "T2", "url": "https://b", "description": ""}]}}, 5)
check("brave_search.format_results renders title/url/snippet",
      "T1" in _bres and "https://a" in _bres and "snip one" in _bres)
check("brave_search.format_results empty -> (no results)",
      _bs.format_results({"web": {"results": []}}, 5) == "(no results)")
check("brave_search.format_results strips highlight tags",
      "<strong>" not in _bs.format_results({"web": {"results": [
          {"title": "A <strong>B</strong>", "url": "https://x", "description": "see <strong>this</strong>"}]}}, 5))
check("brave_search.run: needs a query", "needs a 'query'" in _bs.run({}, str(FIX)))
import os as _os_bs, pathlib as _pl_bs
_saved_key = _os_bs.environ.pop(_bs.KEY_ENV, None)
_saved_kf = _bs.KEY_FILE
_bs.KEY_FILE = _pl_bs.Path("/nonexistent/brave.key")   # isolate from any real key file
check("brave_search.run: no key -> clear instruction",
      "Brave API key" in _bs.run({"query": "x"}, str(FIX)))
_bs.KEY_FILE = _saved_kf
if _saved_key is not None:
    _os_bs.environ[_bs.KEY_ENV] = _saved_key

# markdown styling for streamed output (tests run headless: ANSI off, so these
# verify markers are STRIPPED to clean text - no raw '## stuff' on a tiny terminal)
# hr(): non-TTY (these tests) -> plain ASCII dash run, no ANSI, no box-draw chars
_hr = lc.hr()
check("hr: non-TTY is plain ASCII dashes (no ANSI / no unicode rule)",
      set(_hr) == {"-"} and "\033" not in _hr and chr(0x2500) not in _hr)
check("style_md_line strips heading hashes", lc.style_md_line("## Title ##") == "Title")
check("style_md_line strips **bold**", lc.style_md_line("a **b** c") == "a b c")
check("style_md_line strips inline `code`", lc.style_md_line("use `x` now") == "use x now")
check("style_md_line strips ** and ` inside a heading too",
      lc.style_md_line("## Title **b** `c`") == "Title b c")
check("style_md_line leaves list/plain untouched",
      lc.style_md_line("- item") == "- item" and lc.style_md_line("plain") == "plain")
# single-* / single-_ italic: markers stripped (styling is a no-op off a TTY)
check("style_md_line strips *italic*", lc.style_md_line("a *b* c") == "a b c")
check("style_md_line strips _italic_", lc.style_md_line("a _b_ c") == "a b c")
check("style_md_line italic leaves a list bullet alone",
      lc.style_md_line("* item") == "* item")
check("style_md_line italic ignores spaced/unbalanced asterisks",
      lc.style_md_line("2 * 3 = 6") == "2 * 3 = 6" and lc.style_md_line("a * b") == "a * b")
check("style_md_line **bold** wins over * (bold stripped, no stray italic)",
      lc.style_md_line("a **b** c") == "a b c")
check("style_md_line strips *italic* and identifiers with _ untouched",
      lc.style_md_line("call foo_bar_baz here") == "call foo_bar_baz here")
# _tool_glyph: a backgrounded run_command gets the distinct bg glyph; else the gear
check("_tool_glyph: backgrounded run_command -> bg glyph",
      lc._tool_glyph("run_command", {"cmd": "sleep 100 &"}) == lc.GLYPH["bg"])
check("_tool_glyph: normal run_command -> exec category glyph",
      lc._tool_glyph("run_command", {"cmd": "ls"}) == lc.GLYPH["cat_exec"])
check("_tool_glyph: read_file -> read category glyph",
      lc._tool_glyph("read_file", {"path": "x"}) == lc.GLYPH["cat_read"])
check("_tool_glyph: apply_diff -> write category glyph",
      lc._tool_glyph("apply_diff", {"path": "x"}) == lc.GLYPH["cat_write"])
check("_tool_glyph: mcp tool -> mcp category glyph",
      lc._tool_glyph("mcp__srv__tool", {}) == lc.GLYPH["cat_mcp"])
check("_tool_glyph: net lean-tool -> net category glyph",
      lc._tool_glyph("web_fetch", {"url": "x"}) == lc.GLYPH["cat_net"])
check("_tool_glyph: update_plan -> meta category glyph",
      lc._tool_glyph("update_plan", {"plan": "x"}) == lc.GLYPH["cat_meta"])
check("_tool_glyph: unknown tool -> gear fallback",
      lc._tool_glyph("some_random_tool", {}) == lc.GLYPH["tool"])
# _tool_category
check("_tool_category: exec", lc._tool_category("run_command", {"cmd": "ls"}) == "exec")
check("_tool_category: mcp", lc._tool_category("mcp__x__y", {}) == "mcp")
check("_tool_category: unknown -> ''", lc._tool_category("nope", {}) == "")
# result status + preview
check("_result_status: ok", lc._result_status("214 lines")[0] == "ok")
check("_result_status: error text", lc._result_status("error: boom")[0] == "no")
check("_result_status: nonzero exit", lc._result_status("exit 1")[0] == "no")
check("_result_status: exit 0 is ok", lc._result_status("exit 0")[0] == "ok")
check("_result_status: traceback", lc._result_status("Traceback (most recent call last):")[0] == "no")
check("_first_meaningful_line skips blanks",
      lc._first_meaningful_line("\n\n  hi there \n more") == "hi there")
check("_first_meaningful_line empty", lc._first_meaningful_line("   \n ") == "")
_pv = lc._tool_result_preview("read_file", "line1\nline2\nline3", 7)
check("_tool_result_preview shows first line", "line1" in _pv)
check("_tool_result_preview flags more + expand id", "+2 more" in _pv and "/expand 7" in _pv)
check("_tool_result_preview empty -> (no output)",
      "(no output)" in lc._tool_result_preview("read_file", "", 0))
# operator turn rendering
_ot = lc._operator_turn_lines("operator", "do a status")
check("_operator_turn_lines has name + bar", "operator" in _ot and lc.GLYPH["userbar"] in _ot)
check("_operator_turn_lines one body line", _ot.count("\n") == 1)
_ot2 = lc._operator_turn_lines("dide", "line one\nline two")
check("_operator_turn_lines bars every paste line", _ot2.count(lc.GLYPH["userbar"]) == 3)
check("_operator_turn_lines custom name", "dide" in _ot2)
# TOOL["glyph"] override wins over category
lc._LEAN_TOOL_GLYPHS["myfancy"] = "★"
check("_tool_glyph honours TOOL glyph override",
      lc._tool_glyph("myfancy", {}) == "★")
del lc._LEAN_TOOL_GLYPHS["myfancy"]
check("user_name config default", lc.Config().user_name == "operator")
_mbuf = []
_ms = lc.MarkdownStream(_mbuf.append)
_ms.feed("# H1\nhello **world**\n`code` ")   # last bit has no newline -> held for flush
_ms.flush()
check("MarkdownStream line-buffers + styles complete lines, flushes the tail",
      "".join(_mbuf) == "H1\nhello world\ncode ")
_fbuf = []
_fs = lc.MarkdownStream(_fbuf.append)
_fs.feed("```\n**keep me raw**\nx = 1\n```\ndone\n")
_fs.flush()
_fout = "".join(_fbuf)
check("MarkdownStream keeps code-fence content verbatim, drops the ``` delimiters",
      "**keep me raw**" in _fout and "x = 1" in _fout and "```" not in _fout and "done" in _fout)

# core-side 404 model fallback: a tier-gated model saved in config 404s on send;
# the turn should fall back to the next offered model, not die.
import types as _types
_fcfg = lc.Config()
_fcfg.provider = "demoprov"
_fcfg.provider_settings = {"demoprov": {"model": "gated-x"}}   # generic, sweep-safe names
class _F404Client:
    def __init__(self): self.calls = 0; self.seen = []
    def list_models(self): return ["gated-x", "model-a", "model-b"]
    def chat(self, messages, tools, should_abort=None):
        self.calls += 1; self.seen.append(_fself.cfg.active_model())
        if _fself.cfg.active_model() == "gated-x":
            raise RuntimeError("HTTP 404 model: gated-x not found")
        return ({"role": "assistant", "content": "ok"}, 7, False)
_fself = _types.SimpleNamespace(client=_F404Client(), cfg=_fcfg, messages=[],
                                tool_defs=[], _abort=False, active_provider=lambda: None,
                                refresh_tools=lambda: None,
                                _window_messages=lambda m, n: m,
                                _with_plan_reminder=lambda m: m)
_fself._next_available_model = lambda tried: lc.Agent._next_available_model(_fself, tried)
_fself._apply_model = lambda name: lc.Agent._apply_model(_fself, name)
_fr = lc.Agent._chat_with_model_fallback(_fself)
check("404 fallback: turn succeeds after switching models",
      _fr[0]["content"] == "ok" and _fself.client.calls == 2)
check("404 fallback: active model moved off the unavailable one",
      _fcfg.active_model() == "model-a")
# a non-404 error must propagate (not be swallowed as a model fallback)
class _FBoomClient:
    def list_models(self): return ["m1", "m2"]
    def chat(self, *a, **k): raise RuntimeError("connection reset by peer")
_fb = _types.SimpleNamespace(client=_FBoomClient(), cfg=lc.Config(), messages=[],
                             tool_defs=[], _abort=False, active_provider=lambda: None,
                             refresh_tools=lambda: None,
                             _window_messages=lambda m, n: m,
                             _with_plan_reminder=lambda m: m)
_fb._next_available_model = lambda tried: lc.Agent._next_available_model(_fb, tried)
_fb._apply_model = lambda name: lc.Agent._apply_model(_fb, name)
_boom = False
try:
    lc.Agent._chat_with_model_fallback(_fb)
except RuntimeError:
    _boom = True
check("404 fallback: non-404 errors still propagate", _boom)
# all models exhausted -> clear RuntimeError, not an infinite loop
class _FAll404:
    def list_models(self): return ["only-one"]
    def chat(self, *a, **k): raise RuntimeError("HTTP 404 nope")
_fe = _types.SimpleNamespace(client=_FAll404(), cfg=lc.Config(), messages=[],
                             tool_defs=[], _abort=False, active_provider=lambda: None,
                             refresh_tools=lambda: None,
                             _window_messages=lambda m, n: m,
                             _with_plan_reminder=lambda m: m)
_fe.cfg.model = "only-one"
_fe._next_available_model = lambda tried: lc.Agent._next_available_model(_fe, tried)
_fe._apply_model = lambda name: lc.Agent._apply_model(_fe, name)
_exhausted = False
try:
    lc.Agent._chat_with_model_fallback(_fe)
except RuntimeError as e:
    _exhausted = "unavailable" in str(e)
check("404 fallback: exhausted models -> clear error, no infinite loop", _exhausted)

# parallel tool calls: classification + order-preserving concurrent execution
_mkcall = lambda n, a=None: {"function": {"name": n, "arguments": a or {}}}
_psa = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
check("_parallel_safe: read_file qualifies", _psa._parallel_safe(_mkcall("read_file")))
check("_parallel_safe: search_files qualifies", _psa._parallel_safe(_mkcall("search_files")))
check("_parallel_safe: list_files qualifies", _psa._parallel_safe(_mkcall("list_files")))
check("_parallel_safe: write_file does not", not _psa._parallel_safe(_mkcall("write_file")))
check("_parallel_safe: run_command does not", not _psa._parallel_safe(_mkcall("run_command")))
check("_parallel_safe: apply_diff does not", not _psa._parallel_safe(_mkcall("apply_diff")))
check("_parallel_safe: ask_user_to_run does not", not _psa._parallel_safe(_mkcall("ask_user_to_run")))
_ppd = pathlib.Path(tempfile.mkdtemp(prefix="lc_pp_"))   # plugdir is rmtree'd earlier
(_ppd / "shout.py").write_text(
    'TOOL = {"name": "shout", "description": "u", "safe": True,\n'
    '        "parameters": {"type": "object", "properties": {}}}\n'
    'def run(args, cwd):\n    return "X"\n')
(_ppd / "danger.py").write_text(
    'TOOL = {"name": "danger", "description": "e",\n'
    '        "parameters": {"type": "object", "properties": {}}}\n'
    'def run(args, cwd):\n    return "Y"\n')
_psp = lc.Agent(lc.Config(cwd=FIX, lean_tools_dir=str(_ppd), lean_tools_enabled=["shout", "danger"]))
check("_parallel_safe: safe lean-tool qualifies", _psp._parallel_safe(_mkcall("shout")))
check("_parallel_safe: unsafe lean-tool does not", not _psp._parallel_safe(_mkcall("danger")))
shutil.rmtree(_ppd, ignore_errors=True)

import time as _time
_pcalls = [_mkcall("read_file", {"path": f"f{i}"}) for i in range(3)]
_pag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_pag.messages = [{"role": "system", "content": "x"}]
def _fake_run(name, args):
    if args.get("path") == "f0":
        _time.sleep(0.05)            # finishes last though dispatched first
    return f"result:{args.get('path')}"
_pag._run_tool = _fake_run
with contextlib.redirect_stdout(io.StringIO()):
    _pag._run_parallel(_pcalls)
_tmsgs = [m for m in _pag.messages if m["role"] == "tool"]
check("_run_parallel: one tool result appended per call", len(_tmsgs) == 3)
check("_run_parallel: results kept in call order despite slow first call",
      [m["content"] for m in _tmsgs] == ["result:f0", "result:f1", "result:f2"])
check("_run_parallel: tool_name tagged on each result",
      all(m["tool_name"] == "read_file" for m in _tmsgs))
# gating: a remote session is never eligible (single serial executor pipe)
_rag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_rag.remote = object()
_batch = [_mkcall("read_file"), _mkcall("read_file")]
check("parallel gate: all-safe local batch is eligible",
      len(_batch) > 1 and not lc.Agent(lc.Config(cwd=FIX)).remote
      and all(_psa._parallel_safe(c) for c in _batch))
check("parallel gate: remote batch is NOT eligible (serial pipe)",
      not (not _rag.remote and all(_rag._parallel_safe(c) for c in _batch)))

# session persistence: manual save/load/list/delete, amnesic by default
def _raises(fn, exc):
    try:
        fn(); return False
    except exc:
        return True
lc.SESSIONS_DIR = Path(tempfile.mkdtemp(prefix="lc_sess_"))
_scfg = lc.Config(cwd=FIX, model="m1", host="h1")
_smsgs = [{"role": "system", "content": "OLD SYS"},
          {"role": "user", "content": "first   message  here"},
          {"role": "assistant", "content": "ok"},
          {"role": "tool", "tool_name": "read_file", "content": "x"}]
check("session: no sessions initially", lc.list_sessions() == [])
_sp, _smeta = lc.save_session(_smsgs, _scfg, "alpha")
check("save_session: writes a file", _sp.is_file() and _sp.stem == "alpha")
check("save_session: meta turns excludes system", _smeta["turns"] == 3)
check("save_session: title is first user line", _smeta["title"] == "first message here")
check("save_session: records model/host/cwd",
      _smeta["model"] == "m1" and _smeta["host"] == "h1" and _smeta["cwd"] == str(FIX))
_sd = lc.load_session("alpha")
check("load_session: round-trips messages", _sd["messages"] == _smsgs)
check("load_session: missing -> FileNotFoundError",
      _raises(lambda: lc.load_session("nope"), FileNotFoundError))
# atomic save: no torn file left behind, and no leftover .tmp sidecar (matters for
# per-turn autosave across dide's mid-session reboots - a truncated JSON breaks resume)
check("save_session: no leftover .tmp sidecar after save",
      not any(p.name.startswith("alpha.json.tmp") for p in lc.SESSIONS_DIR.iterdir()))
# a corrupt session file must raise JSONDecodeError (the callers catch it and degrade
# to a fresh start / 'unreadable' notice rather than crashing the launch)
(lc.SESSIONS_DIR / "torn.json").write_text('{"meta": {"model": "x"')  # truncated mid-write
check("load_session: corrupt file -> JSONDecodeError (callers degrade gracefully)",
      _raises(lambda: lc.load_session("torn"), __import__("json").JSONDecodeError))
# name sanitization: no path separators survive, traversal impossible
_sp2, _ = lc.save_session(_smsgs, _scfg, "a/b/../c")
check("save_session: sanitized name has no separators",
      "/" not in _sp2.stem and _sp2.parent == lc.SESSIONS_DIR)
check("save_session: empty/dots name rejected",
      _raises(lambda: lc.save_session(_smsgs, _scfg, "///"), ValueError)
      and _raises(lambda: lc.save_session(_smsgs, _scfg, "..."), ValueError))
check("list_sessions: returns saved names", "alpha" in [n for n, _ in lc.list_sessions()])
check("delete_session: removes file", lc.delete_session("alpha") and not _sp.is_file())
check("delete_session: missing -> False", lc.delete_session("ghost") is False)

# agent.restore: drop saved system, rebuild for current cwd, clear token cache
_resag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_resag.last_prompt_tokens = 9999
_rn = _resag.restore(_smsgs)
check("agent.restore: returns body turn count", _rn == 3)
check("agent.restore: system rebuilt for current cwd (not 'OLD SYS')",
      _resag.messages[0]["role"] == "system"
      and "OLD SYS" not in _resag.messages[0]["content"]
      and str(FIX) in _resag._session_env())
check("agent.restore: only one system message", sum(
    1 for m in _resag.messages if m["role"] == "system") == 1)
check("agent.restore: body preserved after system",
      [m["role"] for m in _resag.messages] == ["system", "user", "assistant", "tool"])
check("agent.restore: clears stale token estimate", _resag.last_prompt_tokens is None)
# restore must survive a malformed message list (corrupt/hand-edited/partial session):
# non-dicts, role-less entries, and a non-list input are all filtered, not crashed on.
_junkag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_jn = _junkag.restore([{"role": "user", "content": "ok"}, "junk", None,
                       {"content": "no role"}, {"role": "assistant", "content": "a"}])
check("agent.restore: filters non-dict/role-less junk (keeps 2 valid)",
      _jn == 2 and all(isinstance(m, dict) and m.get("role") for m in _junkag.messages))
check("agent.restore: a non-list message payload -> empty body (no crash)",
      _junkag.restore("not a list") == 0)
check("/session is a known built-in command", "/session" in lc._BUILTIN_COMMANDS)

# unsaved-changes tracking for /session load guard
_dag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
check("agent: fresh conversation is not dirty", _dag.dirty is False)
_dag.dirty = True
_dag.restore(_smsgs)
check("agent.restore: clears dirty (now showing a saved session)", _dag.dirty is False)
_dag.dirty = True
_dag.reset()
check("agent.reset: clears dirty", _dag.dirty is False)

# --- _fmt_age: relative-age formatter for the /load picker -------------------
check("_fmt_age: seconds", lc._fmt_age(0) == "0s" and lc._fmt_age(59) == "59s")
check("_fmt_age: minutes", lc._fmt_age(60) == "1m" and lc._fmt_age(3599) == "59m")
check("_fmt_age: hours", lc._fmt_age(3600) == "1h" and lc._fmt_age(86399) == "23h")
check("_fmt_age: days", lc._fmt_age(86400) == "1d" and lc._fmt_age(13 * 86400) == "13d")
check("_fmt_age: weeks", lc._fmt_age(14 * 86400) == "2wk")
check("_fmt_age: negative clamps to 0s", lc._fmt_age(-5) == "0s")

# --- autosave session lifecycle ---------------------------------------------
lc.SESSIONS_DIR = Path(tempfile.mkdtemp(prefix="lc_auto_"))
check("_new_autosave_name: carries the auto- prefix",
      lc._new_autosave_name().startswith(lc.AUTOSAVE_PREFIX))
_aag = lc.Agent(lc.Config(cwd=FIX, approval="auto", autosave=True))
_aag.messages = [{"role": "system", "content": "S"},
                 {"role": "user", "content": "hello autosave"}]
lc.autosave_session(_aag, _aag.cfg)
check("autosave_session: writes the rolling target",
      lc._session_path(_aag.autosave_name).is_file())
# off -> amnesic, nothing new written
_aag2 = lc.Agent(lc.Config(cwd=FIX, approval="auto", autosave=False))
_aag2.messages = [{"role": "system", "content": "S"},
                  {"role": "user", "content": "should not persist"}]
lc.autosave_session(_aag2, _aag2.cfg)
check("autosave_session: no-op when autosave off",
      not lc._session_path(_aag2.autosave_name).is_file())
# empty conversation -> no file
_aag3 = lc.Agent(lc.Config(cwd=FIX, approval="auto", autosave=True))
_aag3.messages = [{"role": "system", "content": "S"}]
lc.autosave_session(_aag3, _aag3.cfg)
check("autosave_session: no-op on an empty conversation",
      not lc._session_path(_aag3.autosave_name).is_file())

# _session_rows: most-recently-used (mtime) first (clean dir first)
shutil.rmtree(lc.SESSIONS_DIR, ignore_errors=True)
lc.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
for _i, _nm in enumerate(["auto-old", "auto-mid", "auto-new"]):
    _p, _ = lc.save_session([{"role": "user", "content": f"m{_i}"}], _aag.cfg, _nm)
    os.utime(_p, (1000 + _i * 100, 1000 + _i * 100))
_rows = lc._session_rows()
check("_session_rows: returns (name, meta, mtime) triples", len(_rows[0]) == 3)
check("_session_rows: most-recent first",
      [r[0] for r in _rows[:3]] == ["auto-new", "auto-mid", "auto-old"])

# _prune_autosaves: keep only the newest N auto- sessions, never named ones
lc.save_session([{"role": "user", "content": "keep me"}], _aag.cfg, "named-snapshot")
for _i in range(15):
    _p, _ = lc.save_session([{"role": "user", "content": f"a{_i}"}], _aag.cfg, f"auto-bulk-{_i:02d}")
    os.utime(_p, (2000 + _i, 2000 + _i))
lc._prune_autosaves(keep=10)
_autos_left = list(lc.SESSIONS_DIR.glob(f"{lc.AUTOSAVE_PREFIX}*.json"))
check("_prune_autosaves: keeps only N newest auto- sessions", len(_autos_left) == 10)
check("_prune_autosaves: keeps the newest, drops the oldest",
      (lc.SESSIONS_DIR / "auto-bulk-14.json").is_file()
      and not (lc.SESSIONS_DIR / "auto-bulk-00.json").is_file())
check("_prune_autosaves: never prunes a named snapshot",
      (lc.SESSIONS_DIR / "named-snapshot.json").is_file())
# pre-handover safety snapshots are rolling too: capped to PREHANDOVER_KEEP, not kept
# forever (else a busy session floods /load with dozens). Newest kept, oldest dropped.
# Named <origin>-prehandover-<N>; the origin here is a user session ("sess").
for _i in range(12):
    _pp, _ = lc.save_session([{"role": "user", "content": f"pc{_i}"}], _aag.cfg,
                             f"sess{lc.PREHANDOVER_MARK}{_i}")
    os.utime(_pp, (3000 + _i, 3000 + _i))
lc._prune_autosaves(keep=10)
_pc_left = sorted(p.name for p in lc.SESSIONS_DIR.glob(f"*{lc.PREHANDOVER_MARK}*.json"))
check("_prune_autosaves: caps pre-handover snapshots to PREHANDOVER_KEEP",
      len(_pc_left) == lc.PREHANDOVER_KEEP, _pc_left)
check("_prune_autosaves: keeps the NEWEST pre-handover snapshots, drops the oldest",
      (lc.SESSIONS_DIR / f"sess{lc.PREHANDOVER_MARK}11.json").is_file()
      and not (lc.SESSIONS_DIR / f"sess{lc.PREHANDOVER_MARK}0.json").is_file())
check("_prune_autosaves: snapshot cap leaves named sessions + auto- count intact",
      (lc.SESSIONS_DIR / "named-snapshot.json").is_file()
      and len([p for p in lc.SESSIONS_DIR.glob(f"{lc.AUTOSAVE_PREFIX}*.json")
               if lc.PREHANDOVER_MARK not in p.stem]) == 10)

# handle_save_command / handle_load_command round-trip (session verbs)
shutil.rmtree(lc.SESSIONS_DIR, ignore_errors=True)
lc.SESSIONS_DIR = Path(tempfile.mkdtemp(prefix="lc_sl_"))
_slag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_slag.messages = [{"role": "system", "content": "S"},
                  {"role": "user", "content": "save then load me"},
                  {"role": "assistant", "content": "done"}]
lc.handle_save_command(_slag, _slag.cfg, "snap1")
check("handle_save_command: writes a named snapshot",
      lc._session_path("snap1").is_file())
check("handle_save_command: clears dirty after snapshot", _slag.dirty is False)
# /save continues autosaving INTO the named session (no silent fork to auto-) so the
# on-exit autosave + later /load land where you left off, not at the save-time snapshot.
check("handle_save_command: autosave continues into the named session",
      _slag.autosave_name == "snap1")
_slag.reset()
check("after reset: snapshot body gone from live convo",
      not any("save then load me" in str(m.get("content")) for m in _slag.messages))
_slag.autosave_name = "auto-something-else"          # simulate a fresh/other target
lc.handle_load_command(_slag, _slag.cfg, "snap1")
check("handle_load_command: restores the named snapshot",
      any("save then load me" in str(m.get("content")) for m in _slag.messages))
check("handle_load_command: continues autosaving into the loaded session",
      _slag.autosave_name == "snap1")
# start_new_session: keeps the current session on disk, switches to a fresh target
_slag.messages = [{"role": "system", "content": "S"},
                  {"role": "user", "content": "in snap1 now"},
                  {"role": "assistant", "content": "ok"}]
_slag.autosave_name = "snap1"
_new_auto = lc.start_new_session(_slag, _slag.cfg)        # bare /new -> rolling auto-
check("start_new_session (no name): switches to an auto- target",
      _new_auto.startswith(lc.AUTOSAVE_PREFIX) and _slag.autosave_name == _new_auto)
check("start_new_session: leaves the prior session on disk",
      lc._session_path("snap1").is_file())
check("start_new_session: new conversation is empty",
      not any(m.get("role") != "system" for m in _slag.messages))
_named = lc.start_new_session(_slag, _slag.cfg, "backend")  # /new <name> -> named target
check("start_new_session (name): targets that named session", _named == "backend"
      and _slag.autosave_name == "backend")

# per-session state: save captures perms/remote; restore applies with config fallback
_psag = lc.Agent(lc.Config(cwd=FIX, approval="auto", leash="rw"))
_psag.messages = [{"role": "system", "content": "S"},
                  {"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "ok"}]
_pp, _pm = lc.save_session(_psag.messages, _psag.cfg, "permsess", remote="boxA")
check("save_session captures leash", _pm["leash"] == "rw")
check("save_session captures approval", _pm["approval"] == "auto")
check("save_session captures remote host", _pm["remote"] == "boxA")
# restore onto a LOWER current ceiling -> applies the session's leash + flags escalation
_psag.cfg.leash = "r"
_psag.cfg.approval = "ask"
_pst = lc._restore_session_state(_psag, _psag.cfg, _pm)
check("restore applies the session leash", _psag.cfg.leash == "rw")
check("restore flags a leash escalation", _pst["leash_escalated_from"] == "r")
check("restore applies the session approval", _psag.cfg.approval == "auto")
# config fallback: a meta missing leash leaves the current value untouched
_psag.cfg.leash = "rwe"
lc._restore_session_state(_psag, _psag.cfg, {"provider": "ollama"})
check("restore falls back to config when a field is absent", _psag.cfg.leash == "rwe")
# a same-or-lower restore does NOT flag escalation
_psag.cfg.leash = "rwe"
_pst2 = lc._restore_session_state(_psag, _psag.cfg, {"provider": "ollama", "leash": "r"})
check("restore: downgrade is not flagged as escalation",
      _psag.cfg.leash == "r" and _pst2["leash_escalated_from"] is None)

# session locks: own pid is never "live" to us; releasing clears ownership
lc._write_lock("locksess")
check("_write_lock then _own_lock is True", lc._own_lock("locksess") is True)
check("_lock_is_live is False for our OWN lock", lc._lock_is_live("locksess") is False)
lc._release_lock("locksess")
check("_release_lock drops the lock", not lc._lock_path("locksess").exists()
      and lc._own_lock("locksess") is False)
# a stale heartbeat (old ts) is not live, even from a foreign pid/host
lc._lock_path("stalesess").write_text(lc.json.dumps(
    {"pid": 999999, "host": "otherbox", "ts": 0}))
check("_lock_is_live is False for a stale heartbeat", lc._lock_is_live("stalesess") is False)
# same-host: a LIVE pid owns the session even with an OLD heartbeat (an active
# instance that just hasn't autosaved recently must NOT be stealable - the bug)
lc._lock_path("idlelive").write_text(lc.json.dumps(
    {"pid": lc.os.getppid(), "host": lc.socket.gethostname(), "ts": 0}))  # ancient ts, live pid
check("_lock_is_live is True for a live same-host pid with a stale heartbeat",
      lc._lock_is_live("idlelive") is True)
lc._lock_path("idlelive").unlink(missing_ok=True)
# same-host: a DEAD pid frees the lock immediately regardless of a fresh heartbeat
lc._lock_path("deadfresh").write_text(lc.json.dumps(
    {"pid": 999999, "host": lc.socket.gethostname(), "ts": lc.time.time()}))
check("_lock_is_live is False for a dead same-host pid even with a fresh heartbeat",
      lc._lock_is_live("deadfresh") is False)
lc._lock_path("deadfresh").unlink(missing_ok=True)
# a fresh heartbeat from another HOST is taken as live (cross-host trusts the ts)
lc._lock_path("livesess").write_text(lc.json.dumps(
    {"pid": 999999, "host": "otherbox", "ts": lc.time.time()}))
check("_lock_is_live is True for a fresh foreign-host lock",
      lc._lock_is_live("livesess") is True)
# _prune_stale_locks: sweep .lock files nobody live holds (killed instances leave them),
# but keep a LIVE foreign lock and our OWN. stalesess (ts=0) + a dead same-host pid go;
# livesess (fresh foreign) + our own stay.
lc._lock_path("deadpid").write_text(lc.json.dumps(
    {"pid": 999999, "host": lc.socket.gethostname(), "ts": lc.time.time()}))  # fresh ts, dead pid, our host
lc._write_lock("mine")                     # our own live lock
lc._prune_stale_locks()
check("_prune_stale_locks removes a stale-heartbeat lock", not lc._lock_path("stalesess").exists())
check("_prune_stale_locks removes a dead-pid lock on this host", not lc._lock_path("deadpid").exists())
check("_prune_stale_locks keeps a live foreign-host lock", lc._lock_path("livesess").exists())
check("_prune_stale_locks keeps our OWN lock", lc._lock_path("mine").exists())
for _ln in ("stalesess", "livesess", "deadpid", "mine"):
    lc._lock_path(_ln).unlink(missing_ok=True)

# _short_model: long names truncate with an ellipsis; short ones pass through
check("_short_model passes a short name through", lc._short_model("opus-4-8") == "opus-4-8")
_sm = lc._short_model("really-long-model-name-20251101", 20)
check("_short_model truncates a long name", len(_sm) == 20 and _sm.endswith(lc.GLYPH["ellipsis"]))

# _status_rows: 3 rows above the prompt, with the live state; [] when off / non-TTY
_tty_save = lc._TTY
lc._TTY = True
_srag = lc.Agent(lc.Config(cwd=FIX, approval="auto", leash="rw", max_iterations=25))
_srag.messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
_srag.autosave_name = "sess-x"
_srows = lc._status_rows(_srag, _srag.cfg)
_plain = "\n".join(_re.sub(r"\x1b\[[0-9;]*m", "", r) for r in _srows)
check("_status_rows returns 3 rows", len(_srows) == 3)
check("_status_rows shows leash + approve: + handover + tools",
      "rw" in _plain and "approve: auto" in _plain
      and "handover" in _plain and "tools" in _plain)
check("_status_rows hides 'window off' when window is off (default)",
      "window off" not in _plain)
check("_status_rows shows live turn counter", "1 turns" in _plain)
check("_status_rows handover shows only the hard pct", "handover 95%" in _plain)
check("_status_rows drops the 'ok' zone label from ctx", "ok)" not in _plain)
check("_status_rows hides handovers count when none have happened",
      "handovers" not in _plain)
check("_status_rows shows the round-cap only in auto mode", "max 25 rounds" in _plain)
check("_status_rows row1 shows the session name with a colon", "session: sess-x" in _plain)
# window ON -> the size is shown; handovers count appears once any have happened.
# window_tokens (default 'auto') takes precedence in the row, so turn it off to exercise
# the message-count display.
_srag.cfg.window_tokens = 0
_srag.cfg.window_messages = 40
_srag.handovers = 2
_plain_w = "\n".join(_re.sub(r"\x1b\[[0-9;]*m", "", r) for r in lc._status_rows(_srag, _srag.cfg))
check("_status_rows shows 'window N' when window is on", "window 40" in _plain_w)
check("_status_rows shows handovers count once any happened", "2 handovers" in _plain_w)
# window_tokens 'auto' renders as 'window auto' and wins over a message-count window
_srag.cfg.window_tokens = "auto"
_plain_a = "\n".join(_re.sub(r"\x1b\[[0-9;]*m", "", r) for r in lc._status_rows(_srag, _srag.cfg))
check("_status_rows shows 'window auto' when window_tokens is auto", "window auto" in _plain_a)
_srag.cfg.window_tokens = 0
_srag.cfg.window_messages = 0
_srag.handovers = 0
_srag.cfg.statusline = False
check("_status_rows is empty when statusline is off", lc._status_rows(_srag, _srag.cfg) == [])
lc._TTY = _tty_save

# _status_key: the repl reprints the status block only when it MEANINGFULLY changes.
# Same settings + only ctx tokens/pct drifting -> SAME key (no reprint, keeps history
# clean); a settings/zone change -> DIFFERENT key (reprints).
_k_base = ["  session s " + lc.GLYPH["dot"] + " m @ ollama",
           "  rw " + lc.GLYPH["dot"] + " approve auto " + lc.GLYPH["dot"] + " 6 tools",
           "  ctx 2,600/131,072 (2% ok) " + lc.GLYPH["dot"] + " think na " + lc.GLYPH["dot"] + " effort na"]
_k_drift = _k_base[:2] + ["  ctx 5,100/131,072 (4% ok) " + lc.GLYPH["dot"] + " think na " + lc.GLYPH["dot"] + " effort na"]
_k_zone = _k_base[:2] + ["  ctx 5,100/131,072 (4% hard) " + lc.GLYPH["dot"] + " think na " + lc.GLYPH["dot"] + " effort na"]
_k_leash = ["  session s " + lc.GLYPH["dot"] + " m @ ollama",
            "  chat " + lc.GLYPH["dot"] + " approve auto " + lc.GLYPH["dot"] + " 0 tools", _k_base[2]]
check("_status_key ignores the ctx row entirely (drift never forces a reprint)",
      lc._status_key(_k_base) == lc._status_key(_k_drift))
check("_status_key ignores a zone flip too (the ctx row prints every turn anyway)",
      lc._status_key(_k_base) == lc._status_key(_k_zone))
check("_status_key changes on a settings change (leash/tools in rows 1-2)",
      lc._status_key(_k_base) != lc._status_key(_k_leash))
check("_status_key excludes the last (ctx) row", lc._status_key(_k_base) == tuple(_k_base[:-1]))
check("_status_key of empty rows is empty (statusline off)", lc._status_key([]) == ())

shutil.rmtree(lc.SESSIONS_DIR, ignore_errors=True)
lc.SESSIONS_DIR = Path(tempfile.mkdtemp(prefix="lc_sess_"))

# suppress_echo: clean no-op without a TTY (the test env), body still runs
_echo_ran = []
with lc.suppress_echo():
    _echo_ran.append(1)
check("suppress_echo: no-TTY no-op runs the body", _echo_ran == [1])
_echo_exc = False
try:
    with lc.suppress_echo():
        raise ValueError("x")
except ValueError:
    _echo_exc = True
check("suppress_echo: does not swallow exceptions", _echo_exc)

# Composer: input buffer, queue, and the greyed/lit confirm-as-state model
import threading as _cth, time as _ct
_c = lc.Composer()
for _ch in "hello":
    _c.feed(_ch)
check("composer: typing builds the buffer", _c.buf == "hello")
check("composer: input_text shows prompt + buffer",
      _c.input_text() == lc.GLYPH["prompt"] + " hello")
_c.feed("\x7f")
check("composer: backspace deletes", _c.buf == "hell")
check("composer: enter submits (queues + clears)",
      _c.feed("\r") == "submit" and _c.buf == "" and _c.queued == ["hell"])
check("composer: pop_queued drains", _c.pop_queued() == ["hell"] and _c.pop_queued() == [])
check("composer: enter on empty buffer is a no-op", _c.feed("\r") is None and _c.queued == [])

_c2 = lc.Composer(); _res = {}
_th = _cth.Thread(target=lambda: _res.update(a=_c2.ask("apply changes to x?")))
_th.start()
for _ in range(200):
    if _c2._pending is not None:
        break
    _ct.sleep(0.005)
check("composer: confirm armed when buffer empty", _c2.armed())
_t, _lit = _c2.status_text()
check("composer: armed confirm renders lit with [Y/n]", _lit and "[Y/n]" in _t)
# a non-y/n key starts a message and de-arms; the confirm stays pending, greyed
_c2.feed("h")
check("composer: typing de-arms (buffer non-empty)", (not _c2.armed()) and _c2.buf == "h")
check("composer: confirm greyed (not lit) while typing", _c2.status_text()[1] is False)
check("composer: confirm still pending while typing", _c2._pending is not None)
# sending the message clears the buffer -> re-arms (the 'lights up' moment)
check("composer: submitting the message re-arms the confirm",
      _c2.feed("\r") == "submit" and _c2.armed())
check("composer: submitted line is queued", _c2.queued == ["h"])
# backspace path also re-arms (already empty here, just assert it stays armed)
_c2.feed("\x7f")
check("composer: empty buffer stays armed", _c2.armed())
_c2.feed("y"); _th.join(timeout=1)
check("composer: armed 'y' resolves ask() -> True", _res.get("a") is True)

_c3 = lc.Composer(); _res3 = {}
_th3 = _cth.Thread(target=lambda: _res3.update(a=_c3.ask("q?"))); _th3.start()
for _ in range(200):
    if _c3._pending is not None:
        break
    _ct.sleep(0.005)
_c3.feed("n"); _th3.join(timeout=1)
check("composer: armed 'n' resolves ask() -> False", _res3.get("a") is False)

# Enter on an empty (armed) buffer confirms == yes, same as 'y'
_c4 = lc.Composer(); _res4 = {}
_th4 = _cth.Thread(target=lambda: _res4.update(a=_c4.ask("q?"))); _th4.start()
for _ in range(200):
    if _c4._pending is not None:
        break
    _ct.sleep(0.005)
check("composer: armed Enter resolves ask() -> True", _c4.armed())
_c4.feed("\r"); _th4.join(timeout=1)
check("composer: armed '\\r' resolves ask() -> True (Enter == yes)", _res4.get("a") is True)

# Enter while typing must NOT confirm - it submits the message, confirm stays parked
_c5 = lc.Composer(); _res5 = {}
_th5 = _cth.Thread(target=lambda: _res5.update(a=_c5.ask("q?"))); _th5.start()
for _ in range(200):
    if _c5._pending is not None:
        break
    _ct.sleep(0.005)
_c5.feed("h")                       # de-arm via non-empty buffer
check("composer: typed buffer de-arms before Enter", not _c5.armed())
check("composer: Enter while typing submits, does not confirm",
      _c5.feed("\r") == "submit" and _res5.get("a") is None and _c5._pending is not None)
_c5.feed("\r"); _th5.join(timeout=1)   # now empty + armed -> Enter confirms
check("composer: Enter after submit (empty) then confirms", _res5.get("a") is True)

_c4 = lc.Composer(); _c4.status = "running apply_diff"
_t4, _lit4 = _c4.status_text(frame="*")
check("composer: status line shows animated activity when no confirm",
      "running apply_diff" in _t4 and "*" in _t4 and not _lit4)
# rule_label is STATIC: shown without the spinner frame (the idle-remote-indicator
# bug was routing it through self.status, which animates every tick)
_c4b = lc.Composer(); _c4b.rule_label = "[remote: box]"
_t4b, _lit4b = _c4b.status_text(frame="*")
check("composer: rule_label shows without a spinner frame",
      _t4b == "[remote: box]" and "*" not in _t4b and not _lit4b)
# self.status (activity) still wins over a static rule_label when both are set
_c4c = lc.Composer(); _c4c.status = "thinking"; _c4c.rule_label = "[remote: box]"
_t4c, _ = _c4c.status_text(frame="*")
check("composer: animated status wins over static rule_label",
      "thinking" in _t4c and "*" in _t4c)

# --- paste handling: the fix for the runaway (a paste must NOT auto-submit) ---
_cp = lc.Composer()
check("composer: feed_paste inserts literally and returns 'paste'",
      _cp.feed_paste("do x") == "paste" and _cp.buf == "do x")
_cp.feed_paste("\nline2\nline3")
check("composer: feed_paste keeps newlines but never submits (no queue)",
      _cp.buf == "do x\nline2\nline3" and _cp.queued == [])
_cp2 = lc.Composer()
_cp2.feed_paste("a\r\nb\rc")          # CRLF and bare CR normalize to \n
check("composer: feed_paste normalizes CR/CRLF to \\n",
      _cp2.buf == "a\nb\nc" and _cp2.queued == [])
# a pasted 'y' must not action an armed confirm (paste de-arms via non-empty buf)
_cp3 = lc.Composer(); _cp3._pending = "ok?"
check("composer: armed before paste", _cp3.armed())
_cp3.feed_paste("y")
check("composer: paste de-arms a pending confirm (no accidental y/N)",
      (not _cp3.armed()) and _cp3._pending == "ok?")

# Composer._parse: pure bracketed-paste / escape stream parser
_P = lc.Composer._parse
_S = ("", False, [])                  # fresh parser state
# a complete bracketed paste -> one ('paste', text) event, newlines preserved
_ev, _st = _P(_S, "\x1b[200~hello\nworld\x1b[201~")
check("composer._parse: bracketed paste -> single paste event",
      _ev == [("paste", "hello\nworld")] and _st == ("", False, []))
# plain typing -> per-char events, no residual
_ev2, _ = _P(("", False, []), "ab")
check("composer._parse: plain chars -> char events",
      _ev2 == [("char", "a"), ("char", "b")])
# an Enter keypress comes through as a char (so feed() can submit it)
_ev3, _ = _P(("", False, []), "\r")
check("composer._parse: CR is a normal char event", _ev3 == [("char", "\r")])
# arrow key (CSI) is parsed into a named key event, not injected as text
_ev4, _st4 = _P(("", False, []), "\x1b[A")
check("composer._parse: arrow-key escape -> ('key','up')",
      _ev4 == [("key", "up")] and _st4 == ("", False, []))
# an unrecognised escape (e.g. F5 = ESC[15~) is still dropped, not injected
_evx, _stx = _P(("", False, []), "\x1b[15~")
check("composer._parse: unknown escape is dropped", _evx == [] and _stx == ("", False, []))
# a lone control byte (Tab) -> a named key event
_evt, _ = _P(("", False, []), "\t")
check("composer._parse: Tab -> ('key','tab')", _evt == [("key", "tab")])
# paste split across two reads: start in chunk 1, end in chunk 2
_ev5, _st5 = _P(("", False, []), "\x1b[200~par")
check("composer._parse: open paste holds across chunks (no event yet)",
      _ev5 == [] and _st5[1] is True)
_ev6, _st6 = _P(_st5, "tial\x1b[201~done")
check("composer._parse: paste completes on next chunk + trailing keys",
      _ev6 == [("paste", "partial"), ("char", "d"), ("char", "o"),
               ("char", "n"), ("char", "e")] and _st6 == ("", False, []))
# a paste-start marker split mid-marker across chunks is held back, not mangled
_ev7, _st7 = _P(("", False, []), "x\x1b[20")
check("composer._parse: partial start marker held as residual",
      _ev7 == [("char", "x")] and _st7[0] == "\x1b[20")
_ev8, _st8 = _P(_st7, "0~hi\x1b[201~")
check("composer._parse: held marker completes into a paste",
      _ev8 == [("paste", "hi")] and _st8 == ("", False, []))
# end-marker split across chunks while in paste is held back too
_ev9, _st9 = _P(("", False, []), "\x1b[200~ab\x1b[20")
check("composer._parse: partial end marker inside paste held back",
      _ev9 == [] and _st9[1] is True and _st9[0] == "\x1b[20")
_ev10, _st10 = _P(_st9, "1~")
check("composer._parse: held end marker completes the paste",
      _ev10 == [("paste", "ab")] and _st10 == ("", False, []))

# flush_stdin: safe no-op without a TTY (just must not raise)
lc.flush_stdin()
check("flush_stdin: no-op without a tty (no raise)", True)

# _consume_chunk: the runaway-killer. A multi-char read is a paste/burst and
# NEVER submits (independent of bracketed-paste markers); only a lone keypress can.
_cc = lc.Composer()
_cc._consume_chunk("do a\ndo b\ndo c\n")     # unmarkered multiline paste (burst)
check("composer: unmarkered multiline paste -> literal, never submits",
      _cc.buf == "do a\ndo b\ndo c\n" and _cc.queued == [])
_cc2 = lc.Composer(); _cc2.buf = "hello"
check("composer: a lone Enter (1-char chunk) submits",
      _cc2._consume_chunk("\r") and _cc2.queued == ["hello"] and _cc2.buf == "")
_cc3 = lc.Composer(); _cc3._consume_chunk("h"); _cc3._consume_chunk("i")
check("composer: lone printable keys append", _cc3.buf == "hi")
_cc4 = lc.Composer(); _cc4._consume_chunk("\x1b[200~a\nb\x1b[201~")
check("composer: bracketed-paste chunk -> literal, no submit",
      _cc4.buf == "a\nb" and _cc4.queued == [])
_cc5 = lc.Composer(); _cc5.buf = "abc"; _cc5.cur = 3; _cc5._consume_chunk("\x1b[D")
check("composer: Left arrow moves cursor, buffer unchanged",
      _cc5.buf == "abc" and _cc5.cur == 2 and _cc5.queued == [])
_cc6 = lc.Composer(); _cc6._consume_chunk("ab\r")
check("composer: coalesced 'type+Enter' burst is literal, no submit",
      _cc6.queued == [] and _cc6.buf == "ab\n")
# 100-line paste in one burst -> exactly zero submits (the cost-blowup scenario)
_cc7 = lc.Composer(); _cc7._consume_chunk("\n".join(f"line{i}" for i in range(100)) + "\n")
check("composer: 100-line paste burst -> 0 queued (no API-cost blowup)",
      _cc7.queued == [] and _cc7.buf.count("\n") == 100)

# cursor-aware editing: insert mid-buffer, backspace before cursor
_ce = lc.Composer()
for _ch in "helo":
    _ce.feed(_ch)
_ce.feed_key("left"); _ce.feed("l")          # fix "helo" -> "hello"
check("composer: insert at cursor (left then type)",
      _ce.buf == "hello" and _ce.cur == 4)
_ce.feed_key("home"); _ce.feed_key("delete")  # delete first char
check("composer: home + forward-delete", _ce.buf == "ello" and _ce.cur == 0)
_ce.feed_key("end")
check("composer: end moves cursor to buffer end", _ce.cur == 4)
_ce.feed("\x7f")                               # backspace at end
check("composer: backspace before cursor", _ce.buf == "ell" and _ce.cur == 3)
# ctrl-a/e/u/k/w
_ck = lc.Composer(); [_ck.feed(c) for c in "one two three"]
_ck.feed_key("ctrl-w")
check("composer: ctrl-w deletes word before cursor", _ck.buf == "one two ")
_ck.feed_key("ctrl-u")
check("composer: ctrl-u kills to line start", _ck.buf == "")
_ck2 = lc.Composer(); [_ck2.feed(c) for c in "abcdef"]; _ck2.feed_key("home")
_ck2.feed_key("ctrl-k")
check("composer: ctrl-k kills to line end", _ck2.buf == "")
# newline key (Ctrl-O / Alt-Enter) inserts a newline, never submits
_cse = lc.Composer(); [_cse.feed(c) for c in "ab"]
_r_se = _cse.feed_key("newline")
check("composer: newline key inserts \\n, no submit",
      _cse.buf == "ab\n" and _cse.cur == 3 and _r_se is None and _cse.queued == [])
# Ctrl-O (0x0f) routes through _parse -> a 'newline' key event
_evn, _ = lc.Composer._parse(("", False, []), "\x0f")
check("composer._parse: Ctrl-O -> ('key','newline')", _evn == [("key", "newline")])
# Alt-Enter (ESC CR) maps to newline too
_eva, _ = lc.Composer._parse(("", False, []), "\x1b\r")
check("composer._parse: Alt-Enter (ESC CR) -> ('key','newline')",
      _eva == [("key", "newline")])
# SS3 'ESC O M' must NOT be treated as newline (keypad-Enter collision)
_evm, _ = lc.Composer._parse(("", False, []), "\x1bOM")
check("composer._parse: ESC O M is not a newline (dropped)", _evm == [])
# history: submit builds history; Up/Down browse and restore the live buffer
_ch2 = lc.Composer()
for _line in ("first", "second"):
    [_ch2.feed(c) for c in _line]; _ch2.feed("\r")
check("composer: submit records history", _ch2.history == ["first", "second"])
[_ch2.feed(c) for c in "draft"]              # a live, unsubmitted buffer
_ch2.feed_key("up")
check("composer: Up recalls newest history entry", _ch2.buf == "second")
_ch2.feed_key("up")
check("composer: Up again -> older entry", _ch2.buf == "first")
_ch2.feed_key("up")
check("composer: Up clamps at oldest", _ch2.buf == "first")
_ch2.feed_key("down"); _ch2.feed_key("down")
check("composer: Down past newest restores the live draft",
      _ch2.buf == "draft" and _ch2._hist_idx is None)
# no dup + no blank in history
_ch3 = lc.Composer()
[_ch3.feed(c) for c in "x"]; _ch3.feed("\r")
[_ch3.feed(c) for c in "x"]; _ch3.feed("\r")
check("composer: history skips immediate duplicates", _ch3.history == ["x"])
# tab-completion via an injected completer
_ct = lc.Composer()
_ct.completer = lambda w: ["/model", "/models"] if w.startswith("/m") else []
[_ct.feed(c) for c in "/mo"]
_ct.feed_key("tab")                          # common prefix "/model", 2 candidates
check("composer: Tab inserts common prefix + exposes candidates",
      _ct.buf == "/model" and _ct.comp_display == ["/model", "/models"])
_ct2 = lc.Composer()
_ct2.completer = lambda w: ["/quit"] if w.startswith("/q") else []
[_ct2.feed(c) for c in "/q"]
_ct2.feed_key("tab")                         # single match -> full insert, no display
check("composer: Tab single match inserts whole word",
      _ct2.buf == "/quit" and _ct2.comp_display == [])
# shared _completion_options: command name, then arg completion off live state
_co = lc._completion_options(_ag, _ag.cfg, "/he")
check("_completion_options: partial slash cmd -> matching commands",
      all(o.startswith("/he") for o in _co) and "/help" in _co)
_co2 = lc._completion_options(_ag, _ag.cfg, "/leash ")
check("_completion_options: arg completion after a space",
      set(lc.LEASH_LEVELS) | {"?"} == set(_co2))
check("_completion_options: universal ? help arg offered",
      "?" in lc._completion_options(_ag, _ag.cfg, "/leash "))
_co3 = lc._completion_options(_ag, _ag.cfg, "plain text")
check("_completion_options: non-slash line -> no completions", _co3 == [])
# per-command `?`/`help`: dispatch intercepts it, prints the handler's help, does NOT run it
import io as _io, contextlib as _cl
def _cap_dispatch(_c, _a):
    _b = _io.StringIO()
    with _cl.redirect_stdout(_b):
        _ex = lc.dispatch_command(_ag, _ag.cfg, _c, _a)
    return _ex, _b.getvalue()
_qex, _qtxt = _cap_dispatch("/leash", "?")
check("dispatch: /cmd ? prints per-command help, does not exit",
      _qex is False and "/leash" in _qtxt and "ceiling" in _qtxt.lower())
_hex, _htxt = _cap_dispatch("/leash", "help")
check("dispatch: /cmd help == /cmd ?", "ceiling" in _htxt.lower())
_aex, _atxt = _cap_dispatch("/model", "?")
check("dispatch: /cmd ? shows aliases", "aliases" in _atxt and "/models" in _atxt)
# the composer completer wraps it and inserts a whole token via feed_key('tab')
_cw = lc.Composer(); _cw.completer = lc._make_composer_completer(_ag, _ag.cfg)
[_cw.feed(c) for c in "/leas"]; _cw.feed_key("tab")
check("composer completer: Tab completes a slash command", _cw.buf == "/leash")
# history persistence: save + load round-trips, incl. newlines and backslashes
import tempfile as _tf
_hp = _tf.mktemp()
_chs = lc.Composer(); _chs.history = ["hello", "multi\nline", "back\\slash", "a\tb"]
_chs.save_history(_hp)
_chl = lc.Composer(); _chl.load_history(_hp)
check("composer: history save/load round-trips (newlines, backslashes)",
      _chl.history == _chs.history)
_chl2 = lc.Composer(); _chl2.load_history(_tf.mktemp())  # missing file
check("composer: load_history on a missing file -> empty, no raise",
      _chl2.history == [])
# candidate list shows in the status line, and any non-Tab key dismisses it
_ct.feed("x")
check("composer: typing dismisses the candidate list", _ct.comp_display == [])
_ct3 = lc.Composer()
_ct3.completer = lambda w: ["/model", "/models"] if w.startswith("/m") else []
[_ct3.feed(c) for c in "/mo"]; _ct3.feed_key("tab")
_txt, _armed = _ct3.status_text()
check("composer: candidates render in the status line",
      "/model" in _txt and "/models" in _txt and _armed is False)
_ct3.feed_key("left")
check("composer: an edit key dismisses the candidate list", _ct3.comp_display == [])

# composer TTY layer: headless must no-op safely and fall back to the classic path
_before_stdout = sys.stdout
with lc.composer_session(lc.Composer()) as _cact:
    pass
check("composer_session: headless yields False (fallback)", _cact is False)
check("composer_session: restores stdout", sys.stdout is _before_stdout)
check("composer_session: clears _active_composer", lc._active_composer is None)
check("composer.start_tty: False without a TTY", lc.Composer().start_tty() is False)

# composer.wrap_buffer: pure multi-line layout (grows then scrolls); cursor coords
_wb = lc.Composer.wrap_buffer
# short buffer: one row, cursor right after the text
_r, _cr, _cc = _wb("hi", "> ", 80, 5)
check("wrap_buffer: short buffer is one row", _r == ["> hi"] and _cr == 0 and _cc == 5)
# empty buffer: just the prompt row, cursor after prompt
_r, _cr, _cc = _wb("", "> ", 80, 5)
check("wrap_buffer: empty buffer is the prompt row", _r == ["> "] and _cr == 0 and _cc == 3)
# explicit mid-buffer cursor: "hi", cursor at index 1 -> after the 'h'
_r, _cr, _cc = _wb("hi", "> ", 80, 5, cursor=1)
check("wrap_buffer: mid-buffer cursor lands after that char",
      _r == ["> hi"] and _cr == 0 and _cc == 4)
# cursor at start of buffer -> right after the prompt
_r, _cr, _cc = _wb("hi", "> ", 80, 5, cursor=0)
check("wrap_buffer: cursor=0 sits just after the prompt", _cr == 0 and _cc == 3)
# mid-buffer cursor across a wrap boundary
_r, _cr, _cc = _wb("abcdefg", "> ", 5, 5, cursor=1)  # prompt(2)+1 = pos 3, row 0
check("wrap_buffer: mid-buffer cursor on the first wrapped row",
      _cr == 0 and _cc == 4)
# wrapping: prompt(2)+7 chars at cols=5 -> rows of width 5; cursor on last row
_r, _cr, _cc = _wb("abcdefg", "> ", 5, 5)
check("wrap_buffer: wraps into width-cols rows",
      _r == ["> abc", "defg"] and all(len(x) <= 5 for x in _r) and _cr == 1 and _cc == 5)
# exact fill (prompt+8 = 10 = 2*5) wraps the cursor to a fresh empty row
_r, _cr, _cc = _wb("abcdefgh", "> ", 5, 5)
check("wrap_buffer: exact fill wraps cursor to a fresh row",
      _r == ["> abc", "defgh", ""] and _cr == 2 and _cc == 1)
# growth then scroll: 30 chars at cols=5, cap 3 rows -> keep the tail 3, cursor visible
_r, _cr, _cc = _wb("x" * 28, ">>", 5, 3)
check("wrap_buffer: caps at max_rows (scrolls)", len(_r) == 3)
check("wrap_buffer: cursor stays on a visible row after scroll", 0 <= _cr < len(_r))
# floors: cols/max_rows < 1 are clamped, never crash
_r, _cr, _cc = _wb("abc", "", 0, 0)
check("wrap_buffer: cols/max_rows floored at 1", len(_r) == 1 and _cr == 0)

# glyph mechanism: one place, with ASCII fallback (no per-site capability checks)
_usave = lc._UNICODE
lc._UNICODE = True
check("g(): picks the Unicode form when supported", lc.g("X", "Y") == "X")
lc._UNICODE = False
check("g(): falls back to ASCII when not", lc.g("X", "Y") == "Y")
lc._UNICODE = _usave
check("GLYPH: every entry has a value", all(v for v in lc.GLYPH.values()))
check("GLYPH['rule'] is box-draw or ASCII dash", lc.GLYPH["rule"] in (chr(0x2500), "-"))
check("THINK_FRAMES is capability-aware and non-empty", len(lc.THINK_FRAMES) >= 1)
check("TOOL_FRAMES is capability-aware and non-empty", len(lc.TOOL_FRAMES) >= 1)

# git_summary lean-tool: read-only snapshot of a repo
_gs = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/git_summary.py")
check("git_summary TOOL is safe (read-only)", _gs.TOOL.get("safe") is True)
check("git_summary: no params (operates on cwd)", _gs.TOOL["parameters"]["properties"] == {})
_nogit = Path(tempfile.mkdtemp(prefix="lc_nogit_"))
check("git_summary: non-repo -> clear message",
      "not a git repository" in _gs.run({}, str(_nogit)))
shutil.rmtree(_nogit, ignore_errors=True)
import subprocess as _gsp
_grepo = Path(tempfile.mkdtemp(prefix="lc_grepo_"))
_genv = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
if shutil.which("git"):
    def _g(*a):
        _gsp.run(["git", *a], cwd=_grepo, env=_genv, capture_output=True, text=True)
    _g("init", "-q", "-b", "main")
    _g("config", "user.email", "t@example.com")
    _g("config", "user.name", "t")
    (_grepo / "a.txt").write_text("one\n")
    _g("add", "a.txt"); _g("commit", "-qm", "first commit")
    (_grepo / "a.txt").write_text("one\ntwo\n")      # unstaged change
    (_grepo / "b.txt").write_text("new\n"); _g("add", "b.txt")   # staged change
    (_grepo / "c.txt").write_text("untracked\n")     # untracked
    _out = _gs.run({}, str(_grepo))
    check("git_summary: shows branch line", "## main" in _out)
    check("git_summary: shows the recent commit", "first commit" in _out)
    check("git_summary: shows staged section", "staged:" in _out and "b.txt" in _out)
    check("git_summary: shows unstaged section", "unstaged:" in _out and "a.txt" in _out)
    check("git_summary: flags the untracked file", "c.txt" in _out)
else:
    check("git_summary: (git not installed - repo test skipped)", True)
shutil.rmtree(_grepo, ignore_errors=True)

# notify lean-tool: setup-only (driver), no model tool; pings on finish/needs-input
_nf = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/notify.py")
check("notify: is setup-only (no model TOOL)", not hasattr(_nf, "TOOL"))
check("notify: has a setup() hook", hasattr(_nf, "setup"))
class _FakeAg:
    def run_turn(self, x): return "ran"
    def _end_of_turn(self): pass
_fa_run, _fa_end = _FakeAg.run_turn, _FakeAg._end_of_turn
class _FakeTools:
    def run_command(self, cmd): return "ok"
_ft_rc = _FakeTools.run_command
def _fake_rdc(cfg, cmd, reason=None, remote=None, editable=True): return "ran"
_ncmds = {}
_fakelc = {
    "Agent": _FakeAg,
    "Tools": _FakeTools,
    "_ask": lambda q: True,
    "run_direct_command": _fake_rdc,
    "register_command": lambda n, h, hl="": _ncmds.__setitem__(n, (h, hl)),
}
_orig_fake_ask = _fakelc["_ask"]
_nf.setup(_fakelc, lc.Config(cwd=FIX))           # patches the FAKE lc only
check("notify: registers the /notify command", "/notify" in _ncmds)
check("notify: wraps Agent.run_turn (driver hook)", _FakeAg.run_turn is not _fa_run)
check("notify: wraps Agent._end_of_turn", _FakeAg._end_of_turn is not _fa_end)
check("notify: wraps _ask in the module dict", _fakelc["_ask"] is not _orig_fake_ask)
check("notify: does NOT wrap Tools.run_command (turn-end only, not per-command)",
      _FakeTools.run_command is _ft_rc)
check("notify: does NOT wrap run_direct_command (turn-end only)",
      _fakelc["run_direct_command"] is _fake_rdc)
check("notify: audible bell is OFF by default (no sound)", _nf._state["bell"] is False)
_nf._state["on"] = False                          # silence subprocess in tests
check("notify: wrapped _ask still calls through", _fakelc["_ask"]("ok?") is True)
# _ping would otherwise fire a REAL notify-send toast on the tester's desktop
# (visible as summary "t" / body "b"). Stub the
# module's shutil.which -> None so it takes the no-op path but still runs.
_nf_which = _nf.shutil.which
_nf.shutil.which = lambda *_a, **_k: None
try:
    check("notify: _ping never raises", (_nf._ping("t", "b") or True))
finally:
    _nf.shutil.which = _nf_which
_ntoggle = _ncmds["/notify"][0]
_nf._state["on"] = False; _ntoggle(None, None, "")
check("notify: /notify toggles on", _nf._state["on"] is True)
_ntoggle(None, None, "")
check("notify: /notify toggles off", _nf._state["on"] is False)
_nf._state["bell"] = False; _ntoggle(None, None, "bell")
check("notify: '/notify bell' opts into the audible bell", _nf._state["bell"] is True)
_ntoggle(None, None, "bell")
check("notify: '/notify bell' toggles it back off", _nf._state["bell"] is False)

# multi-session pool: set_remote pools, detach keeps it, drop/close_all
class _FakeWS:
    def __init__(self, h): self.host = h; self.closed = False; self.remote_cwd = "/r"; self.cwd = "."
    def close(self): self.closed = True
    def read_raw(self, p): return None
    def call(self, name, args, should_abort=None): return "ok"
_pag2 = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_w1, _w2 = _FakeWS("boxA"), _FakeWS("boxB")
_pag2.set_remote(_w1); _pag2.set_remote(_w2)
check("pool: both connections pooled", set(_pag2.remotes) == {"boxA", "boxB"})
check("pool: last set_remote is active", _pag2.remote.host == "boxB")
_pag2.set_remote(None)
check("pool: /local detach keeps the pool alive", _pag2.remote is None and len(_pag2.remotes) == 2)
_pag2.drop_remote("boxA")
check("pool: drop_remote closes + removes", _w1.closed and "boxA" not in _pag2.remotes)
_pag2.set_remote(_w2); _pag2.close_all_remotes()
check("pool: close_all_remotes tears everything down",
      _w2.closed and _pag2.remotes == {} and _pag2.remote is None)

# mid-session drop: a dead executor triggers ONE silent reconnect+retry; only a
# reconnect that ALSO fails falls back to local + error result (no crash).
class _DropWS(_FakeWS):
    def call(self, name, args, should_abort=None): raise ConnectionError("pipe closed")
    def connect(self, batch=False): raise ConnectionError("reconnect failed")   # box really gone
_dpag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_dpag.set_remote(_DropWS("boxX"))
with contextlib.redirect_stdout(io.StringIO()):
    _dres = _dpag._run_tool("read_file", {"path": "x"})
check("drop: dead remote + failed reconnect -> error mentioning LOCAL", "LOCAL" in _dres)
check("drop: falls back to local (active cleared)", _dpag.remote is None)
check("drop: dead connection removed from pool", "boxX" not in _dpag.remotes)

# transient blip: first call dies, reconnect succeeds, retry returns the result
# (no drop to LOCAL, remote stays active).
class _BlipWS(_FakeWS):
    def __init__(self, host):
        super().__init__(host); self._n = 0
    def call(self, name, args, should_abort=None):
        self._n += 1
        if self._n == 1: raise ConnectionError("pipe closed")   # transient blip
        return "recovered"
    def connect(self, batch=False): return self                 # reconnect ok
_bpag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_bpag.set_remote(_BlipWS("boxY"))
with contextlib.redirect_stdout(io.StringIO()):
    _bres = _bpag._run_tool("read_file", {"path": "x"})
check("blip: reconnect+retry recovers the result", _bres == "recovered")
check("blip: remote stays active (no drop to LOCAL)",
      _bpag.remote is not None and _bpag.remote.host == "boxY")

# _do_connect: switching to an already-open (alive) box is instant, no reconnect
_spag = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_spag.set_remote(_FakeWS("boxA")); _spag.set_remote(_FakeWS("boxB"))   # active boxB
_orig_alive = lc._remote_alive
lc._remote_alive = lambda ws: True
with contextlib.redirect_stdout(io.StringIO()):
    lc._do_connect(_spag, lc.Config(cwd=FIX), "boxA")
check("switch: _do_connect activates a pooled box without reconnecting",
      _spag.remote.host == "boxA" and set(_spag.remotes) == {"boxA", "boxB"})
with contextlib.redirect_stdout(io.StringIO()):
    lc._do_connect(_spag, lc.Config(cwd=FIX), "boxA")    # already active -> no-op
check("switch: connecting to the active box is a no-op", _spag.remote.host == "boxA")
lc._remote_alive = _orig_alive

# picker shows open-but-unsaved targets and marks the active one
with contextlib.redirect_stdout(io.StringIO()) as _mbuf:
    _mpick = lc.pick_connect_menu({"box1": "user@a"}, open_hosts=["user@b"],
                                  active="user@b", prompt=lambda p: "2")
check("picker: includes an open-but-unsaved target", _mpick == "user@b")
check("picker: marks the active session", "(current)" in _mbuf.getvalue())

# /connect saved addresses: [connect] config round-trip + the picker
_chosts = {"box1": "user@192.0.2.1", "box2": "user@192.0.2.2"}
with contextlib.redirect_stdout(io.StringIO()):
    _cpick = lc.pick_connect_menu(_chosts, prompt=lambda p: "2")
    _ccancel = lc.pick_connect_menu(_chosts, prompt=lambda p: "0")
    _cempty = lc.pick_connect_menu({}, prompt=lambda p: "1")
check("pick_connect_menu: returns chosen target", _cpick == "user@192.0.2.2")
check("pick_connect_menu: 0 cancels -> None", _ccancel is None)
check("pick_connect_menu: empty list -> None", _cempty is None)
_ccfgp = Path(tempfile.mktemp(prefix="lc_conncfg_"))
_orig_cp = lc.CONFIG_PATH
lc.CONFIG_PATH = _ccfgp
with contextlib.redirect_stdout(io.StringIO()):
    lc.save_config(lc.Config(connect_hosts={"box1": "user@192.0.2.1"}))
import tomllib as _ctoml
_cdat = _ctoml.loads(_ccfgp.read_text())
check("save_config: writes [connect] table", _cdat.get("connect") == {"box1": "user@192.0.2.1"})
_ccfgp.unlink(missing_ok=True)
lc.CONFIG_PATH = _orig_cp

# --version self-hash (the version probe /connect uses to skip a redundant push)
_sh = lc.source_hash()
check("source_hash: 64-char hex", len(_sh) == 64 and all(c in "0123456789abcdef" for c in _sh))
import subprocess as _vsp
_vout = _vsp.run([sys.executable, str(Path(lc.__file__)), "--version"],
                 capture_output=True, text=True, timeout=30)
check("--version prints the source hash and exits 0",
      _vout.returncode == 0 and _vout.stdout.strip() == _sh)

# __version__ + version_tuple: the queryable release version the /update lean-tool
# probes (distinct axis from source_hash's exact-content fingerprint).
check("__version__ is a dotted string", isinstance(lc.__version__, str) and "." in lc.__version__)
check("version_tuple defaults to this build", lc.version_tuple() == lc.version_tuple(lc.__version__))
check("version_tuple orders numerically", lc.version_tuple("0.2.0") > lc.version_tuple("0.1.9"))
check("version_tuple: broad versions order (1.12 > 1.9)", lc.version_tuple("1.12.0") > lc.version_tuple("1.9.0"))
check("version_tuple tolerates junk (no raise)", lc.version_tuple("v1.x.foo")[:3] == (1, 0, 0))
# SemVer pre-release precedence: a -beta build ranks BELOW the same core release,
# so a beta user isn't 'newer' than stable at the same version (the beta-track gotcha).
check("version_tuple: release > its pre-release", lc.version_tuple("1.2.0") > lc.version_tuple("1.2.0-beta.1"))
check("version_tuple: beta.2 > beta.1", lc.version_tuple("1.2.0-beta.2") > lc.version_tuple("1.2.0-beta.1"))
check("version_tuple: numeric pre-release < alphanumeric", lc.version_tuple("1.2.0-1") < lc.version_tuple("1.2.0-alpha"))
# the bundled VERSION file matches __version__ (install ships it for the probe)
_vf = Path(lc.__file__).parent / "VERSION"
check("VERSION file matches __version__",
      _vf.is_file() and _vf.read_text().strip() == lc.__version__)

# web_fetch lean-tool: egress, confirm-gated (not safe), validates without network
_wf = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/web_fetch.py")
check("web_fetch: NOT safe (egress always confirms)", not _wf.TOOL.get("safe"))
check("web_fetch: requires a url", "needs a valid" in _wf.run({}, str(FIX)))
check("web_fetch: a phrase is not a url -> error pointing to brave_search",
      "brave_search" in _wf.run({"url": "not a url"}, str(FIX)))
check("web_fetch: does NOT pre-block localhost (local dev is a use case)",
      "SSRF" not in _wf.run({"url": "http://127.0.0.1:9/"}, str(FIX)))

# ============================================================================
# External provider registration API
# ============================================================================
class _FakeClient:
    uncapped_ctx = True
    def __init__(self, cfg): self.cfg = cfg
    def chat(self, messages, tools, should_abort=None):
        return ({"role": "assistant", "content": "ok"}, 1, False)
    def list_models(self): return ["cloud-a", "cloud-b"]
    def running_models(self): return []
    def detect_num_ctx(self): return 200000

_prov_log = []
_fake_spec = {
    "name": "fake",
    "make_client": lambda cfg: _FakeClient(cfg),
    "list_models": lambda: ["cloud-a", "cloud-b"],
    "context_window": lambda m: 200000 if m == "cloud-a" else 100000,
    "capabilities": lambda m: {"thinking": ["off", "adaptive", "max"],
                               "effort": ["low", "med", "high"]},
    "available": lambda: True,
    "autostart": lambda: False,
    "on_activate":   lambda a, c: _prov_log.append(("act", c.provider)),
    "on_deactivate": lambda a, c: _prov_log.append(("deact", c.provider)),
    "usage": lambda a, c: {"meters": [{"label": "5h", "pct": 12, "resets_at": ""},
                                      {"label": "wk", "pct": 100, "resets_at": ""}],
                           "note": "ok"},
}

# A deterministic, offline stand-in for the bundled ollama provider. The real one
# (providers/ollama.py) wraps OllamaClient and hits the network; in the sandbox
# that's unreachable, so the suite registers this fake to exercise the collapsed
# "ollama is just a provider" dispatch without a server.
class _FakeOllamaClient:
    def __init__(self, cfg): self.cfg = cfg
    def chat(self, messages, tools, should_abort=None):
        return ({"role": "assistant", "content": "ok"}, 1, False)
    def list_models(self): return ["qwen", "qwen2"]
    def running_models(self): return ["qwen"]
    def detect_num_ctx(self): return 32768

_ollama_spec = {
    "name": "ollama",
    "make_client": lambda cfg: _FakeOllamaClient(cfg),
    "list_models": lambda: ["qwen", "qwen2"],
    "context_window": lambda m: 32768,
    "available": lambda: True,
    "warm_models": lambda: ["qwen"],
    "tag": "ollama",
}

def _raises(fn, exc=(ValueError, RuntimeError)):
    try:
        fn(); return False
    except exc:
        return True

_norm = lc.register_provider(_fake_spec)
check("register_provider: stored + retrievable",
      lc.get_provider("fake") is _norm and "fake" in lc.provider_names())
check("register_provider: tag defaults to the provider name", _norm["tag"] == "fake")
check("register_provider: explicit tag kept",
      lc.register_provider({**_fake_spec, "name": "tagged", "tag": "plan"})["tag"] == "plan")
check("register_provider: rejects missing make_client",
      _raises(lambda: lc.register_provider({"name": "x", "list_models": lambda: [],
                                            "context_window": lambda m: 1})))
check("register_provider: 'ollama' is registerable (no longer reserved)",
      lc.register_provider(_ollama_spec)["name"] == "ollama" and lc.get_provider("ollama") is not None)

# Config accessors: ollama (default) reads through provider_settings, falling back
# to the top-level model/num_ctx seed; a provider reads its own settings.
pc = lc.Config(model="qwen", num_ctx=32768)
check("default provider is ollama", pc.provider == "ollama")
check("active_model: ollama -> falls back to seed model", pc.active_model() == "qwen")
check("ctx_window: ollama -> falls back to seed num_ctx", pc.ctx_window() == 32768)
pc.set_active_model("qwen3")
check("set_active_model: writes provider_settings + ollama alias",
      pc.provider_settings["ollama"]["model"] == "qwen3" and pc.model == "qwen3"
      and pc.active_model() == "qwen3")
pc = lc.Config(model="qwen", num_ctx=32768)
pc.provider = "fake"; pc.provider_settings["fake"] = {"model": "cloud-a", "num_ctx": 200000}
check("active_model: provider -> provider model", pc.active_model() == "cloud-a")
check("ctx_window: provider -> provider window", pc.ctx_window() == 200000)
check("provider active leaves native model/window untouched",
      pc.model == "qwen" and pc.num_ctx == 32768)
pc.set_setting("thinking", "max")
check("setting/set_setting roundtrip", pc.setting("thinking") == "max")

# activate / deactivate through the Agent.
# NOTE: an earlier test ran backend.setup(), which monkey-patches Agent.__init__
# to activate the API client when the live auth is active (the exact coupling the
# provider API replaces). Neutralize that here so a fresh Agent honors the Config
# we pass instead of being swapped onto the hosted model.
# `be` only exists when the local-only provider suite ran backend.setup() (which
# monkeypatches Agent.__init__); on a public checkout it never ran, so there is
# nothing to neutralize. Guard so this works with or without the local suite.
_be = globals().get("be")
_be_isactive = _be._Auth.is_active if _be else None
if _be:
    _be._Auth.is_active = lambda self: False
try:
    _prov_log.clear()
    pag = lc.Agent(lc.Config(cwd=FIX, model="qwen", num_ctx=32768))
    _m = pag.activate_provider("fake")
    check("activate_provider: picks default model", _m == "cloud-a")
    check("activate_provider: sets cfg.provider", pag.cfg.provider == "fake")
    check("activate_provider: swaps in provider client", isinstance(pag.client, _FakeClient))
    check("activate_provider: records provider window",
          pag.cfg.provider_settings["fake"]["num_ctx"] == 200000)
    check("activate_provider: native model/window intact",
          pag.cfg.model == "qwen" and pag.cfg.num_ctx == 32768)
    check("activate_provider: on_activate ran", ("act", "fake") in _prov_log)
    check("activate_provider: unknown raises", _raises(lambda: pag.activate_provider("nope")))
    lc.register_provider({**_fake_spec, "name": "down", "available": lambda: False})
    check("activate_provider: unavailable raises", _raises(lambda: pag.activate_provider("down")))
    pag.deactivate_provider()
    check("deactivate_provider: back to the ollama default",
          pag.cfg.provider == "ollama" and isinstance(pag.client, _FakeOllamaClient))
    check("deactivate_provider: on_deactivate ran", any(x[0] == "deact" for x in _prov_log))

    # unknown config-named provider -> falls back to the bundled ollama default
    cag = lc.Agent(lc.Config(cwd=FIX, model="qwen", provider="ghost"))
    csp = lc._maybe_autostart_provider(cag, cag.cfg)
    check("_maybe_autostart_provider: unknown named -> ollama fallback",
          csp is not None and cag.cfg.provider == "ollama")

    # autostart resolution: a non-ollama autostart() beats the ollama default,
    # but an explicit non-ollama config provider beats autostart.
    lc.register_provider({**_fake_spec, "name": "auto1", "autostart": lambda: True})
    aag = lc.Agent(lc.Config(cwd=FIX, model="qwen"))   # provider defaults to "ollama"
    asp = lc._maybe_autostart_provider(aag, aag.cfg)
    check("_maybe_autostart_provider: autostart()=True beats ollama default",
          asp is not None and aag.cfg.provider == "auto1")
    bag = lc.Agent(lc.Config(cwd=FIX, model="qwen", provider="fake"))
    lc._maybe_autostart_provider(bag, bag.cfg)
    check("_maybe_autostart_provider: explicit config provider wins", bag.cfg.provider == "fake")

    # /provider (incl. its `set` subcommand) handlers (no-arg menus would block, so pass args)
    hag = lc.Agent(lc.Config(cwd=FIX, model="qwen"))
    lc.handle_provider_command(hag, hag.cfg, "fake")
    check("handle_provider_command: activates named provider", hag.cfg.provider == "fake")
    lc.handle_provider_command(hag, hag.cfg, "set thinking max")
    check("/provider set: sets a declared setting", hag.cfg.setting("thinking") == "max")
    lc.handle_provider_command(hag, hag.cfg, "set thinking bogus")
    check("/provider set: rejects an invalid choice (unchanged)", hag.cfg.setting("thinking") == "max")
    lc.handle_provider_command(hag, hag.cfg, "set tool_format json")
    check("/provider set: free-form custom knob allowed (extensibility)",
          hag.cfg.setting("tool_format") == "json")
    lc.handle_provider_command(hag, hag.cfg, "off")
    check("handle_provider_command: off -> ollama", hag.cfg.provider == "ollama")
    # bare /provider is the on/off catalog now (the old switch-picker is gone): it
    # lists plugins and does NOT change the active backend. Patch input so the menu
    # loop can't block, and capture the header.
    import builtins as _bi, io as _io3, contextlib as _ctx3
    _orig_input = _bi.input
    _bi.input = lambda *a, **k: "0"          # immediately "done"
    try:
        hag.cfg.provider = "fake"
        _pb = _io3.StringIO()
        with _ctx3.redirect_stdout(_pb):
            lc.handle_provider_command(hag, hag.cfg, "")
        check("bare /provider opens the catalog (does not switch the active backend)",
              hag.cfg.provider == "fake" and "provider plugins" in _pb.getvalue(),
              repr(_pb.getvalue()[:60]))
    finally:
        _bi.input = _orig_input
    # the catalog marks the ACTIVE backend, so with several enabled you can tell which
    # one is running (enabled != active; ollama used to show [off] yet be the one in use)
    _padir = pathlib.Path(tempfile.mkdtemp(prefix="lc_provcat_"))
    (_padir / "myactive.py").write_text('PROVIDER = {"name": "myactive", "description": "d"}\n')
    _pacfg = lc.Config(cwd=FIX, providers_dir=str(_padir),
                       providers_enabled=["myactive"], provider="myactive")
    _orig_input = _bi.input
    _bi.input = lambda *a, **k: "0"
    try:
        _pab = _io3.StringIO()
        with _ctx3.redirect_stdout(_pab):
            lc.handle_providers_command(None, _pacfg, "")
    finally:
        _bi.input = _orig_input
    _paout = _re.sub(r"\x1b\[[0-9;]*m", "", _pab.getvalue())
    check("provider catalog marks (active) on the running provider's row",
          any("myactive" in ln and "(active)" in ln for ln in _paout.splitlines()), repr(_paout))
    check("provider catalog does NOT mark a non-active provider active",
          not any("ollama" in ln and "(active)" in ln for ln in _paout.splitlines()))
    shutil.rmtree(_padir, ignore_errors=True)
    hag.cfg.provider = "ollama"               # restore for the tests that follow

    # _restore_backend_for: a loaded session switches the live backend to match its
    # saved provider + model, degrading gracefully on failure. Legacy provider=""
    # (the old native Ollama) maps onto the bundled "ollama" provider.
    rbag = lc.Agent(lc.Config(cwd=FIX, model="qwen", num_ctx=32768))
    lc._restore_backend_for(rbag, rbag.cfg, {"provider": "", "model": "qwen2"})
    check("_restore_backend_for: legacy ''->ollama syncs the model",
          rbag.cfg.provider == "ollama" and rbag.cfg.active_model() == "qwen2")
    lc._restore_backend_for(rbag, rbag.cfg, {"provider": "fake", "model": "cloud-b"})
    check("_restore_backend_for: activates the saved provider", rbag.cfg.provider == "fake")
    check("_restore_backend_for: restores the exact saved provider model",
          rbag.cfg.active_model() == "cloud-b" and isinstance(rbag.client, _FakeClient))
    lc._restore_backend_for(rbag, rbag.cfg, {"provider": "", "model": "qwen"})
    check("_restore_backend_for: provider->ollama switches back",
          rbag.cfg.provider == "ollama" and isinstance(rbag.client, _FakeOllamaClient))
    _nu = lc._restore_backend_for(rbag, rbag.cfg, {"provider": "ghostp", "model": "z"})
    check("_restore_backend_for: unregistered provider -> note, no switch",
          rbag.cfg.provider == "ollama" and "ghostp" in _nu)
    _nd = lc._restore_backend_for(rbag, rbag.cfg, {"provider": "down", "model": "z"})
    check("_restore_backend_for: unavailable provider -> note, no switch",
          rbag.cfg.provider == "ollama" and "down" in _nd)
    # save_session records the active backend + the active model
    _ps_dir = lc.SESSIONS_DIR
    lc.SESSIONS_DIR = Path(tempfile.mkdtemp(prefix="lc_provmeta_"))
    rbag.cfg.provider = "fake"
    rbag.cfg.provider_settings.setdefault("fake", {})["model"] = "cloud-a"
    _pp, _pmeta = lc.save_session([{"role": "user", "content": "hi"}], rbag.cfg, "psave")
    check("save_session: meta records the provider", _pmeta.get("provider") == "fake")
    check("save_session: meta model is the active model", _pmeta.get("model") == "cloud-a")
    shutil.rmtree(lc.SESSIONS_DIR, ignore_errors=True)
    lc.SESSIONS_DIR = _ps_dir
    rbag.cfg.provider = ""

    # ProviderManager: discover a providers/ dir plugin (module-level PROVIDER +
    # helpers-only setup()), register the enabled one, toggle on/off via /providers.
    _pdir = Path(tempfile.mkdtemp(prefix="lc_providers_"))
    (_pdir / "plugp.py").write_text(
        'PROVIDER = {\n'
        '    "name": "plugp",\n'
        '    "description": "test provider plugin",\n'
        '    "make_client": lambda cfg: None,\n'
        '    "list_models": lambda: ["pm-a", "pm-b"],\n'
        '    "context_window": lambda m: 4096,\n'
        '    "available": lambda: True,\n'
        '}\n'
        '_setup_calls = []\n'
        'def setup(lc, cfg):\n'
        '    _setup_calls.append(1)\n')
    (_pdir / "broken.py").write_text('x = 1  # no PROVIDER dict\n')
    _pm = lc.ProviderManager(str(_pdir), enabled=["plugp"])
    check("ProviderManager: discovers PROVIDER plugins", _pm.names() == ["plugp"])
    check("ProviderManager: skips files without a PROVIDER dict", "broken" not in _pm.names())
    check("ProviderManager: desc from the spec", _pm.desc("plugp") == "test provider plugin")
    lc._providers.pop("plugp", None)
    _reg = _pm.register_enabled(lc.Config(cwd=FIX))
    check("ProviderManager: registers the enabled provider", lc.get_provider("plugp") is not None)
    check("ProviderManager: register_enabled ran the helpers-only setup",
          _pm.providers["plugp"]["module"]._setup_calls == [1])
    check("ProviderManager: returns the registered names", _reg == ["plugp"])
    # disabled providers are not registered
    lc._providers.pop("plugp", None)
    lc.ProviderManager(str(_pdir), enabled=[]).register_enabled(lc.Config(cwd=FIX))
    check("ProviderManager: a disabled provider is NOT registered", lc.get_provider("plugp") is None)
    # the providers/ catalog is now managed via /provider enable|disable|list (the
    # standalone /providers command is GONE - folded in, no alias).
    # handle_provider_command autosaves config -> redirect CONFIG_PATH to a temp first
    # so the suite NEVER writes the user's real ~/.config/leancoder/config.toml.
    lc.CONFIG_PATH = pathlib.Path(tempfile.mktemp(suffix=".toml"))
    _pcfg = lc.Config(cwd=FIX, providers_dir=str(_pdir))
    _pag2 = lc.Agent(_pcfg)
    check("/providers is an alias of /provider (same handler)",
          lc._BUILTIN_COMMANDS_TABLE.get("/providers") is lc.handle_provider_command
          and lc._BUILTIN_COMMANDS_TABLE.get("/models") is lc.handle_model_command)
    lc.handle_provider_command(_pag2, _pcfg, "enable plugp")
    check("/provider enable <name>: enables + registers", lc.get_provider("plugp") is not None
          and "plugp" in _pcfg.providers_enabled)
    lc.handle_provider_command(_pag2, _pcfg, "enable plugp")    # idempotent
    check("/provider enable <name>: idempotent (already on)", "plugp" in _pcfg.providers_enabled)
    lc.handle_provider_command(_pag2, _pcfg, "disable plugp")
    check("/provider disable <name>: disables + drops from registry",
          lc.get_provider("plugp") is None and "plugp" not in _pcfg.providers_enabled)
    check("/provider completion includes the catalog subverbs",
          set(["list", "enable", "disable"]).issubset(set(lc._arg_completions(_pag2, _pcfg, "/provider"))))
    # config round-trips providers_dir + providers_enabled
    _pcf = pathlib.Path(tempfile.mktemp(suffix=".toml")); lc.CONFIG_PATH = _pcf
    lc.save_config(lc.Config(model="m", providers_dir="/x/prov",
                             providers_enabled=["plugp", "plugq"]), quiet=True)
    import tomllib as _toml_p
    _prt = _toml_p.loads(_pcf.read_text())
    check("save_config: writes providers_dir", _prt.get("providers_dir") == "/x/prov")
    check("save_config: writes providers_enabled", _prt.get("providers_enabled") == ["plugp", "plugq"])
    shutil.rmtree(_pdir, ignore_errors=True)
    lc._providers.pop("plugp", None)
finally:
    if _be:
        _be._Auth.is_active = _be_isactive

# usage meter rendering (provider feeds numbers, core renders) + reset formatting
_um = lc.render_usage_meters(_fake_spec["usage"](None, pc))
check("render_usage_meters: labels + FULL + note present",
      "5h 12%" in _um and "wk FULL" in _um and "ok" in _um)
check("render_usage_meters: empty -> ''",
      lc.render_usage_meters({}) == "" and lc.render_usage_meters(None) == "")
# tidy line hides the reset; verbose (/info) shows it; a 'day' token + tag render inline
_fut = (lc.datetime.datetime.now(lc.datetime.timezone.utc) + lc.datetime.timedelta(hours=5)).isoformat()
_um_day = {"meters": [{"label": "wk", "pct": 58, "resets_at": _fut,
                       "tag": {"text": "0%/day6", "level": "ok"}}]}
_tidy = _re.sub(r"\x1b\[[0-9;]*m", "", lc.render_usage_meters(_um_day))
_verb = _re.sub(r"\x1b\[[0-9;]*m", "", lc.render_usage_meters(_um_day, verbose=True))
check("render_usage_meters: tidy shows 'wk 58% (0%/day6)'",
      "wk 58% (0%/day6)" in _tidy)
# on the LAST day of the weekly window the tag appends the reset clock time
_d7 = be._wk_daily(90, (lc.datetime.datetime.now(lc.datetime.timezone.utc)
                        + lc.datetime.timedelta(hours=3)).isoformat())
check("_wk_daily: day 7 tag appends '(reset@HH:MM)'",
      "/day7" in _d7["text"] and ")(reset@" in _d7["text"])
_d1 = be._wk_daily(5, (lc.datetime.datetime.now(lc.datetime.timezone.utc)
                       + lc.datetime.timedelta(days=6, hours=3)).isoformat())
check("_wk_daily: earlier day has NO reset clock", "reset@" not in _d1["text"])
check("render_usage_meters: tidy HIDES the reset time/countdown", "(5h)" not in _tidy and ":" not in _tidy)
check("render_usage_meters: verbose SHOWS the reset countdown", "h)" in _verb and ":" in _verb)
check("_fmt_reset: bad input -> ''", lc._fmt_reset("") == "" and lc._fmt_reset("garbage") == "")
# countdown suffix so a bare clock time can't be misread as a number (e.g. '15:00(18h)')
_r_soon = lc._fmt_reset((lc.datetime.datetime.now(lc.datetime.timezone.utc)
                         + lc.datetime.timedelta(hours=18)).isoformat())
check("_fmt_reset: within-a-day shows HH:MM(Nh) countdown",
      ":" in _r_soon and _r_soon.endswith("h)"))
_r_far = lc._fmt_reset((lc.datetime.datetime.now(lc.datetime.timezone.utc)
                        + lc.datetime.timedelta(days=3)).isoformat())
check("_fmt_reset: >1 day shows ddMon(Nd) countdown", _r_far.endswith("d)"))
check("_toml_value: scalars + list",
      lc._toml_value("a") == '"a"' and lc._toml_value(True) == "true"
      and lc._toml_value(200000) == "200000" and lc._toml_value(["x", "y"]) == '["x", "y"]')

# display fixes (dev feedback): compact tokens, colour ramp, readline-safe prompt
check("_fmt_tokens: compact units", lc._fmt_tokens(950) == "950" and lc._fmt_tokens(8237) == "8.2k"
      and lc._fmt_tokens(200000) == "200k" and lc._fmt_tokens(1_000_000) == "1M")
check("_pct_color: blue ok / yellow mid / red high",
      lc._pct_color(10) is lc.blue and lc._pct_color(70) is lc.yellow and lc._pct_color(95) is lc.red)
# _rl_wrap puts the ANSI in \001/\002 so readline doesn't count it as width.
# (Use a literal ANSI string: bold()/cyan() are no-ops when stdout isn't a TTY.)
_wrapped = lc._rl_wrap("\x1b[1m\x1b[36m> \x1b[0m")
check("_rl_wrap: wraps each ANSI code in \\001/\\002 ignore markers",
      _wrapped.count("\x01") == 3 and _wrapped.count("\x02") == 3
      and "\x01\x1b[1m\x02" in _wrapped)
check("_rl_wrap: visible text preserved", "> " in _wrapped)
check("_rl_wrap: no ANSI -> unchanged", lc._rl_wrap("plain") == "plain")

# save_config / load_config round-trip for provider state (use a temp config path)
import tempfile as _tf, pathlib as _pl
_tmpcfg = _pl.Path(_tf.mkdtemp()) / "config.toml"
_orig_cp = lc.CONFIG_PATH
lc.CONFIG_PATH = _tmpcfg
try:
    sc = lc.Config(model="qwen", num_ctx=32768, provider="fake")
    sc.provider_settings = {"fake": {"model": "cloud-a", "num_ctx": 200000,
                                     "thinking": "max", "effort": "low"}}
    lc.save_config(sc)
    _txt = _tmpcfg.read_text()
    check("save_config: writes provider line", 'provider = "fake"' in _txt)
    check("save_config: writes [providers.fake] table (model + settings)",
          "[providers.fake]" in _txt and 'model = "cloud-a"' in _txt and 'thinking = "max"' in _txt)
    check("save_config: does NOT persist num_ctx (context_window hook is source of truth)",
          "num_ctx = 200000" not in _txt)
    class _AC:
        host = None; model = None; num_ctx = None; cwd = str(FIX); approval = "ask"
        temperature = None; top_p = None; top_k = None; repeat_penalty = None; think = None
    _rt = lc.load_config(_AC())
    check("load_config: reads provider", _rt.provider == "fake")
    check("load_config: reads [providers.fake] model/settings (no stale num_ctx)",
          _rt.provider_settings.get("fake", {}).get("model") == "cloud-a"
          and "num_ctx" not in _rt.provider_settings["fake"])
finally:
    lc.CONFIG_PATH = _orig_cp

# ============================================================================
# Provider API additions: login/clear/detail hooks, /usage, session tokens
# ============================================================================
import io as _io, contextlib as _ctxl

check("register_provider: login/clear/detail default to None",
      lc.get_provider("fake")["login"] is None and lc.get_provider("fake")["detail"] is None)

_hook_log = []
lc.register_provider({**_fake_spec, "name": "hooked",
                      "login": lambda a, c: (_hook_log.append("login") or True),
                      "clear": lambda a, c: _hook_log.append("clear"),
                      "detail": lambda a, c: "FULL USAGE VIEW"})
check("register_provider: hooks stored when given",
      callable(lc.get_provider("hooked")["login"]) and callable(lc.get_provider("hooked")["detail"]))

# backend.setup() (local suite only) may have patched Agent.__init__; neutralize
# so Agents stay clean. No-op on a public checkout where the local suite is absent.
_be2 = globals().get("be")
_be_isactive2 = _be2._Auth.is_active if _be2 else None
if _be2:
    _be2._Auth.is_active = lambda self: False
try:
    # session-token accumulation (any backend, via client.last_out_tokens)
    class _FakeChatClient:
        def __init__(self): self.last_out_tokens = None
        def chat(self, messages, tools, should_abort=None):
            self.last_out_tokens = 20
            return ({"role": "assistant", "content": "hi"}, 100, False)
        def list_models(self): return []
        def running_models(self): return []
    tag = lc.Agent(lc.Config(cwd=FIX, model="qwen"))
    tag.client = _FakeChatClient()
    tag.run_turn("hello")
    check("session_in accumulates prompt_eval", tag.session_in == 100)
    check("session_out accumulates client.last_out_tokens", tag.session_out == 20)
    tag.run_turn("again")
    check("session tokens accumulate across turns", tag.session_in == 200 and tag.session_out == 40)
    tag.reset()
    check("reset zeroes session tokens", tag.session_in == 0 and tag.session_out == 0)

    # /usage renders a provider's detail()
    uag = lc.Agent(lc.Config(cwd=FIX, model="qwen"))
    lc.handle_provider_command(uag, uag.cfg, "hooked")
    check("provider with hooks activates", uag.cfg.provider == "hooked")
    _b = _io.StringIO()
    with _ctxl.redirect_stdout(_b):
        lc.handle_usage_command(uag, uag.cfg)
    check("/usage renders provider detail()", "FULL USAGE VIEW" in _b.getvalue())

    # /provider clear runs the hook and drops back to the ollama default
    lc.handle_provider_command(uag, uag.cfg, "clear")
    check("/provider clear runs clear hook + deactivates",
          "clear" in _hook_log and uag.cfg.provider == "ollama")

    # /provider login [name] runs the hook then activates
    lc.handle_provider_command(uag, uag.cfg, "login hooked")
    check("/provider login runs login hook + activates",
          "login" in _hook_log and uag.cfg.provider == "hooked")

    # /usage default view (no detail / Ollama) shows session + context
    oag = lc.Agent(lc.Config(cwd=FIX, model="qwen"))
    oag.session_in, oag.session_out = 123, 45
    _b2 = _io.StringIO()
    with _ctxl.redirect_stdout(_b2):
        lc.handle_usage_command(oag, oag.cfg)
    _o2 = _b2.getvalue()
    check("/usage default shows session tokens + context",
          "session" in _o2 and "123" in _o2 and "context" in _o2)
    # /provider on + named re-activation (testing-day fix #1)
    pon = lc.Agent(lc.Config(cwd=FIX, model="qwen"))
    lc.handle_provider_command(pon, pon.cfg, "fake")          # named activation
    check("/provider <name> activates directly", pon.cfg.provider == "fake")
    lc.handle_provider_command(pon, pon.cfg, "off")
    check("/provider off remembers last provider",
          pon._last_provider == "fake" and pon.cfg.provider == "ollama")
    lc.handle_provider_command(pon, pon.cfg, "on")            # re-activate last
    check("/provider on re-activates the last provider", pon.cfg.provider == "fake")

    # activate must NOT clobber the top-level (ollama) num_ctx with a provider's
    # window (the leak that left ollama showing /200,000 after /provider off).
    pnx = lc.Agent(lc.Config(cwd=FIX, model="qwen", num_ctx=32768))
    pnx.activate_provider("fake")                             # fake's context_window = 200000
    check("activate keeps the top-level num_ctx intact (no provider-window leak)",
          pnx.cfg.num_ctx == 32768)
    check("ctx_window() reflects the provider while active",
          pnx.cfg.ctx_window() == 200000)
    pnx.deactivate_provider()                                 # back to ollama (fake ollama ctx 32768)
    check("after /provider off, ollama window is back (not the provider's)",
          pnx.cfg.ctx_window() == 32768)
finally:
    if _be2:
        _be2._Auth.is_active = _be_isactive2

# /model N numeric selection (testing-day clarification: '7' meant option 7)
check("_index_pick: number maps to 1-based menu row",
      lc._index_pick("2", ["a", "b", "c"]) == "b")
check("_index_pick: out-of-range -> None", lc._index_pick("9", ["a", "b"]) is None)
check("_index_pick: non-number -> None", lc._index_pick("haiku", ["a", "b"]) is None)

# ============================================================================
# Ollama-as-provider: built-in registration, config migration, unified /model
# ============================================================================
# Bundled ollama provider: now a real file (providers/ollama.py), loaded above as
# `olm`. The client wraps OllamaClient (network); here we check the PROVIDER dict
# has the right shape + honours an explicit context-window pin, no server needed.
_bcfg = lc.Config(cwd=FIX, model="qwen", num_ctx=16384)   # num_ctx given -> auto off -> pin path
olm.setup(lc.__dict__, _bcfg)                              # rebind the module's _cfg to this config
_bo = olm.PROVIDER
check("bundled ollama: PROVIDER tag 'ollama'", _bo["tag"] == "ollama")
check("bundled ollama: make_client builds an OllamaClient",
      isinstance(_bo["make_client"](_bcfg), olm.OllamaClient))
check("bundled ollama: available() is True (localhost floor)", _bo["available"]() is True)
check("bundled ollama: context_window honours an explicit num_ctx pin",
      _bo["context_window"]("anything") == 16384)
check("bundled ollama: warm_models is callable (green dots)", callable(_bo.get("warm_models")))
check("bundled ollama: location() labels host via machine alias",
      olm.PROVIDER["location"](lc.Config(host="http://h:1", machines={"box": "http://h:1"})) == "box (http://h:1)")
olm._cfg = lc.Config(cwd=FIX)                              # restore for any later use

# Bundled hosted providers: each ships as a providers/ file with a module-level
# PROVIDER dict. Verify shape + that the unauthed-safe hooks don't throw (no network,
# no keys). anthropic_plan is NOT bundled publicly (OAuth/subscription) - its deeper
# tests live in the local-only _smoketest_local.py.
for _pname, _ptag in (("anthropic_api", "api"), ("gemini", "gem"),
                      ("groq", "groq"), ("openrouter", "or"),
                      ("llamacpp", "gguf"), ("mlx", "mlx")):
    _pp = lc._load_lean_tool(Path(lc.__file__).parent / "providers" / f"{_pname}.py")
    _pd = _pp.PROVIDER
    check(f"provider {_pname}: name matches file", _pd["name"] == _pname)
    check(f"provider {_pname}: tag '{_ptag}'", _pd.get("tag") == _ptag)
    check(f"provider {_pname}: required hooks present + callable",
          all(callable(_pd.get(h)) for h in ("make_client", "list_models", "context_window")))
    check(f"provider {_pname}: list_models() safe unauthed (list, no throw)",
          isinstance(_pd["list_models"](), list))
    check(f"provider {_pname}: context_window() returns a positive int",
          isinstance(_pd["context_window"]("x"), int) and _pd["context_window"]("x") > 0)
    check(f"provider {_pname}: available() callable + returns a bool",
          callable(_pd.get("available")) and isinstance(_pd["available"](), bool))

# Example provider template (examples/providers/example.py): must be a valid,
# loadable provider so anyone copying it gets a working starting point.
_ex = lc._load_lean_tool(Path(lc.__file__).parent / "examples/providers/example.py")
check("example provider: PROVIDER name 'example'", _ex.PROVIDER["name"] == "example")
check("example provider: list_models() returns a non-empty list",
      isinstance(_ex.PROVIDER["list_models"](), list) and _ex.PROVIDER["list_models"]())
check("example provider: make_client builds its client",
      isinstance(_ex.PROVIDER["make_client"](lc.Config(cwd=FIX)), _ex.ExampleClient))

# Config migration: a legacy config (no provider, top-level model) loads onto the
# bundled ollama provider; the first save writes the new format + a backup.
class _MigArgs:
    host = None; model = None; num_ctx = None; cwd = str(FIX)
    approval = "ask"; temperature = None; top_p = None; top_k = None
    repeat_penalty = None; think = None
_mig_cp = lc.CONFIG_PATH
_mtmp = pathlib.Path(tempfile.mkdtemp(prefix="lc_mig_")) / "config.toml"
_mtmp.write_text('host = "http://localhost:11434"\nmodel = "legacy-model"\n')
lc.CONFIG_PATH = _mtmp
try:
    _mc = lc.load_config(_MigArgs())
    check("migration: legacy provider '' -> 'ollama'", _mc.provider == "ollama")
    check("migration: top-level model seeds [providers.ollama].model",
          _mc.provider_settings.get("ollama", {}).get("model") == "legacy-model")
    check("migration: active_model() reads the migrated model",
          _mc.active_model() == "legacy-model")
    lc.save_config(_mc, quiet=True)
    _txt = _mtmp.read_text()
    check("save: writes the [providers.ollama] canonical block", "[providers.ollama]" in _txt)
    check("save: keeps the top-level model alias", 'model = "legacy-model"' in _txt)
    _bakp = _mtmp.with_name(_mtmp.name + ".pre-migration.bak")
    check("save: one-time pre-migration backup written (legacy format preserved)",
          _bakp.is_file() and "legacy-model" in _bakp.read_text()
          and "[providers.ollama]" not in _bakp.read_text())
    # seed-once: a config with NO providers_enabled key (fresh/legacy) seeds ollama on...
    check("seed: absent providers_enabled seeds ollama (out-of-the-box default)",
          _mc.providers_enabled == ["ollama"])
    # ...but once the key EXISTS it is respected verbatim: disabling ollama STICKS (not
    # re-seeded), and it isn't re-added on the next load. Also an explicit empty set holds.
    _mtmp.write_text('providers_enabled = ["acme"]\nmodel = "m"\n')
    _off = lc.load_config(_MigArgs())
    check("disable sticks: ollama NOT re-seeded when the key excludes it",
          _off.providers_enabled == ["acme"] and "ollama" not in _off.providers_enabled)
    _mtmp.write_text('providers_enabled = []\nmodel = "m"\n')
    check("explicit empty providers_enabled is respected (no re-seed)",
          lc.load_config(_MigArgs()).providers_enabled == [])
    # and save persists the enabled set even when empty, so the disable round-trips
    lc.save_config(lc.Config(model="m", providers_enabled=[]), quiet=True)
    check("save: persists an empty providers_enabled (so 'all off' sticks)",
          "providers_enabled = []" in _mtmp.read_text())
finally:
    lc.CONFIG_PATH = _mig_cp

# Unified /model across every enabled backend (rows + arg resolution + selection).
_snap = dict(lc._providers)
_umc_cp = lc.CONFIG_PATH
lc.CONFIG_PATH = pathlib.Path(tempfile.mktemp(suffix=".toml"))   # autosave must not touch real config
lc._providers.clear()
lc.register_provider(_ollama_spec)                          # qwen, qwen2 (warm: qwen)
lc.register_provider(_fake_spec)                            # cloud-a, cloud-b (tag "fake")
lc.register_provider({**_fake_spec, "name": "lo", "available": lambda: False})
try:
    _uag = lc.Agent(lc.Config(cwd=FIX, model="qwen"))
    _uag.activate_provider("ollama")
    _rows = lc._model_rows(_uag, _uag.cfg)
    _ms = [r["model"] for r in _rows]
    check("model_rows: lists every backend's models", "qwen" in _ms and "cloud-a" in _ms)
    check("model_rows: warm flag from warm_models()",
          any(r["model"] == "qwen" and r["warm"] for r in _rows))
    check("model_rows: current flag on the active model",
          any(r["current"] and r["model"] == _uag.cfg.active_model() for r in _rows))
    check("model_rows: unavailable provider -> a login row (model None)",
          any(r["provider"] == "lo" and r["model"] is None and not r["available"] for r in _rows))
    check("model_rows: tag carried from spec",
          any(r["provider"] == "fake" and r["tag"] == "fake" for r in _rows))
    _sel = [r for r in _rows if r["model"]]
    check("resolve_model_arg: 1-based index", lc._resolve_model_arg("1", _sel) is _sel[0])
    check("resolve_model_arg: bare name", lc._resolve_model_arg("cloud-b", _sel)["model"] == "cloud-b")
    _q = lc._resolve_model_arg("fake:cloud-a", _sel)
    check("resolve_model_arg: provider:model qualifier",
          _q is not None and _q["model"] == "cloud-a" and _q["provider"] == "fake")
    check("resolve_model_arg: no match -> None", lc._resolve_model_arg("nope", _sel) is None)
    _row = next(r for r in _sel if r["provider"] == "fake" and r["model"] == "cloud-b")
    with _ctxl.redirect_stdout(_io.StringIO()):
        lc._select_model_row(_uag, _uag.cfg, _row)
    check("select_model_row: switches provider + model across backends",
          _uag.cfg.provider == "fake" and _uag.cfg.active_model() == "cloud-b")
    with _ctxl.redirect_stdout(_io.StringIO()):
        lc.handle_model_command(_uag, _uag.cfg, "ollama:qwen2")
    check("handle_model_command arg: cross-backend select switches back to ollama",
          _uag.cfg.provider == "ollama" and _uag.cfg.active_model() == "qwen2")
    with _ctxl.redirect_stdout(_io.StringIO()):
        _ch = lc.pick_unified_model_menu(_rows, prompt=lambda *_: "1")
    check("pick_unified_model_menu: numeric pick returns a row", _ch is not None and "provider" in _ch)
    with _ctxl.redirect_stdout(_io.StringIO()):
        _cn = lc.pick_unified_model_menu(_rows, prompt=lambda *_: "0")
    check("pick_unified_model_menu: cancel -> None", _cn is None)
    # grouped display model: headers + selectable items, index_of only counts selectables
    _glines, _gidx = lc._unified_model_lines(_rows)
    check("_unified_model_lines: index_of are all selectable rows",
          all(k in ("group", "info", "item") for k, _ in _glines)
          and _gidx == [p for k, p in _glines if k == "item"]
          and len(_gidx) >= 1)
    check("_fmt_model_row: numbered + name present",
          "1)" in lc._ANSI_RE.sub("", lc._fmt_model_row(_gidx[0], 1)))
finally:
    lc._providers.clear(); lc._providers.update(_snap)
    lc.CONFIG_PATH = _umc_cp

# ============================================================================
# Session modes: incognito / chat-only / ask-on-read
# ============================================================================
_modes_cp = lc.CONFIG_PATH
lc.CONFIG_PATH = pathlib.Path(tempfile.mktemp(suffix=".toml"))   # autosave must not touch real config
try:
    check("_parse_toggle: on", lc._parse_toggle("on", False) is True)
    check("_parse_toggle: off", lc._parse_toggle("off", True) is False)
    check("_parse_toggle: empty flips current",
          lc._parse_toggle("", True) is False and lc._parse_toggle("", False) is True)
    check("_parse_toggle: invalid -> None", lc._parse_toggle("maybe", False) is None)

    # /handover block parser (the @#! delimiters) - must never grab the wrong text
    _M = lc.HANDOVER_MARK
    check("handover block: extracts between markers",
          lc._extract_handover_block(f"prose\n{_M}\nthe handover\n{_M}\ntrailing") == "the handover")
    check("handover block: LAST pair wins (ignores an echoed example)",
          lc._extract_handover_block(f"{_M}example{_M} then real {_M}real one{_M}") == "real one")
    check("handover block: no markers -> None (history NOT nuked)",
          lc._extract_handover_block("just a summary, no markers") is None)
    check("handover block: single marker -> None", lc._extract_handover_block(f"{_M} only one") is None)
    check("handover block: empty between -> None", lc._extract_handover_block(f"{_M}{_M}") is None)

    # /leash capability ceiling: bounds the tool surface by level
    check("_norm_leash: aliases", lc._norm_leash("READ") == "r" and lc._norm_leash("exec") == "rwe"
          and lc._norm_leash("rw") == "rw" and lc._norm_leash("none") == "chat")
    check("_norm_leash: junk -> None", lc._norm_leash("wat") is None)
    _names = lambda c: [t["function"]["name"] for t in lc.active_tools(c)]
    check("leash chat -> no tools", lc.active_tools(lc.Config(leash="chat")) == [])
    _rn = _names(lc.Config(leash="r"))
    check("leash r -> read tools only",
          set(_rn) == set(lc.SAFE_TOOLS) | {"update_plan", "note"} and "write_file" not in _rn and "run_command" not in _rn)
    _rwn = _names(lc.Config(leash="rw"))
    check("leash rw -> + write, no exec",
          "write_file" in _rwn and "apply_diff" in _rwn and "run_command" not in _rwn
          and "ask_user_to_run" not in _rwn)
    _rwen = _names(lc.Config(leash="rwe"))
    check("leash rwe -> full surface incl. run_command + ask_user_to_run",
          "run_command" in _rwen and "ask_user_to_run" in _rwen)
    check("leash default is rwe (back-compat full surface)", lc.Config().leash == "rwe")
    # lean-tools tier by their safe flag: safe rides at r+, non-safe only at rwe
    _safe_t = ({"type": "function", "function": {"name": "peek"}}, True)
    _unsafe_t = ({"type": "function", "function": {"name": "zap"}}, False)
    _rl = [t["function"]["name"] for t in lc.active_tools(lc.Config(leash="r"), lean_tool_schemas=[_safe_t, _unsafe_t])]
    check("leash r: safe lean-tool in, non-safe out", "peek" in _rl and "zap" not in _rl)
    _el = [t["function"]["name"] for t in lc.active_tools(lc.Config(leash="rwe"), lean_tool_schemas=[_safe_t, _unsafe_t])]
    check("leash rwe: both lean-tools in", "peek" in _el and "zap" in _el)

    # chat rung tells the model + hides tools
    _mag = lc.Agent(lc.Config(cwd=FIX, leash="chat"))
    check("leash chat Agent: tool_defs empty", _mag.tool_defs == [])
    # the permissions note now rides the NEXT user turn (not the cached system prompt)
    check("leash chat: model told via pending note, NOT system prompt",
          "chat-only" in (_mag._pending_ai_note or "")
          and "Chat-only" not in _mag.messages[0]["content"])
    with _ctxl.redirect_stdout(_io.StringIO()):
        lc.handle_leash_command(_mag, _mag.cfg, "rwe")         # back to full
    check("/leash rwe -> tools restored",
          len(_mag.tool_defs) > 0 and _mag.cfg.leash == "rwe")
    with _ctxl.redirect_stdout(_io.StringIO()):
        lc.handle_leash_command(_mag, _mag.cfg, "rw")
    check("/leash rw: tools rebuilt without run_command + on-change note set",
          "run_command" not in [t["function"]["name"] for t in _mag.tool_defs]
          and "read-write" in (_mag._pending_ai_note or "")
          and "Read-write" not in _mag.messages[0]["content"])
    # incognito: no local session persistence. (The incognito NOTICE was removed from
    # the system prompt - the model is no longer told; only the on-disk behaviour matters.)
    _iag = lc.Agent(lc.Config(cwd=FIX, incognito=True))
    _iag.autosave_name = "auto-incog-test"
    check("incognito: notice NOT in the (trimmed) system prompt",
          "Incognito" not in _iag.messages[0]["content"])
    _isd = lc.SESSIONS_DIR
    lc.SESSIONS_DIR = Path(tempfile.mkdtemp(prefix="lc_incog_"))
    _iag.messages.append({"role": "user", "content": "secret"})
    lc.autosave_session(_iag, _iag.cfg)
    check("incognito: autosave writes nothing", not list(lc.SESSIONS_DIR.glob("*.json")))
    _iag.cfg.incognito = False; _iag.cfg.autosave = True
    lc.autosave_session(_iag, _iag.cfg)
    check("non-incognito: autosave writes a session", bool(list(lc.SESSIONS_DIR.glob("*.json"))))
    shutil.rmtree(lc.SESSIONS_DIR, ignore_errors=True); lc.SESSIONS_DIR = _isd

    # ask-on-read: reads confirm too + leave the parallel fast-path
    _rag = lc.Agent(lc.Config(cwd=FIX, confirm_reads=True, approval="ask"))
    _readcall = {"function": {"name": "read_file", "arguments": {"path": "x"}}}
    check("ask-on-read: read leaves the parallel fast-path", _rag._parallel_safe(_readcall) is False)
    _rag.cfg.approval = "auto"
    check("ask-on-read + auto: read can still parallelize", _rag._parallel_safe(_readcall) is True)
    _rag.cfg.approval = "ask"
    _orig_ask = lc.ask_action
    lc.ask_action = lambda cfg, q: False           # decline at the gate
    try:
        with _ctxl.redirect_stdout(_io.StringIO()):
            _res = _rag._run_tool("read_file", {"path": "anything"})
        check("ask-on-read: declining a read blocks it", "declined" in _res)
    finally:
        lc.ask_action = _orig_ask
    check("ask-on-read off by default", lc.Agent(lc.Config(cwd=FIX)).cfg.confirm_reads is False)

    # config round-trips confirm_reads; leash and incognito are NEVER persisted (both
    # ephemeral - a fresh launch must default to the full surface, never silently boot
    # narrowed because an earlier session's leash got snapshotted to disk).
    _mc2 = pathlib.Path(tempfile.mktemp(suffix=".toml")); lc.CONFIG_PATH = _mc2
    lc.save_config(lc.Config(model="m", leash="rw", confirm_reads=True, incognito=True), quiet=True)
    _mt = _mc2.read_text()
    check("save: leash NOT persisted (ephemeral)", "leash" not in _mt)
    check("save: confirm_reads persisted", "confirm_reads = true" in _mt)
    check("save: incognito NOT persisted (ephemeral)", "incognito" not in _mt)
    lc.save_config(lc.Config(model="m", leash="chat"), quiet=True)
    check("save: chat leash also omitted", "leash" not in _mc2.read_text())
    # approval cadence PERSISTS (a sticky preference, unlike leash): a non-default mode
    # is written and reloaded so every new/auto-spawned session keeps it; default "ask"
    # is omitted; junk in the file falls back to "ask".
    lc.save_config(lc.Config(model="m", approval="session"), quiet=True)
    check("save: non-default approval persisted", 'approval = "session"' in _mc2.read_text())
    lc.save_config(lc.Config(model="m"), quiet=True)
    check("save: default approval (ask) omitted", "approval" not in _mc2.read_text())
finally:
    lc.CONFIG_PATH = _modes_cp

# arg/subcommand tab completion (testing-day fix #2) - _arg_completions is pure
class _FakeArgAgent:
    class _Cl:
        def list_models(self): return ["m1", "m2"]
    client = _Cl()
    remotes = {"box1": object()}
    def active_provider(self): return None
_fa = _FakeArgAgent()
_fc = lc.Config(cwd=FIX, connect_hosts={"dev": "user@dev"},
                machines={"mac": "http://mac:11434"})
check("argcomp: /model from active client", lc._arg_completions(_fa, _fc, "/model") == ["m1", "m2"])
check("argcomp: /ollama offers subcommands + machines",
      lc._arg_completions(_fa, _fc, "/ollama")
      == ["host", "model", "failover", "reset", "priority", "mac"])
check("argcomp: /provider has off/on/login/clear + names",
      {"off", "on", "login", "clear"}.issubset(set(lc._arg_completions(_fa, _fc, "/provider")))
      and "fake" in lc._arg_completions(_fa, _fc, "/provider"))
check("argcomp: /effort levels", lc._arg_completions(_fa, _fc, "/effort") == ["low", "med", "high"])
check("argcomp: /think modes", lc._arg_completions(_fa, _fc, "/think") == ["off", "adaptive", "max"])
check("argcomp: /session subcommands",
      lc._arg_completions(_fa, _fc, "/session") == ["list", "delete", "save", "load"])
# sub-path completion: each level of a command tree resolves its own arg. The
# completer keys _arg_completions on the full path (e.g. '/session delete'), so a
# saved-session list is offered there - not the subcommand list again.
_sess_arg = lc._arg_completions(_fa, _fc, "/session delete")
check("argcomp: /session delete completes session names (not subcommands again)",
      "delete" not in _sess_arg and "save" not in _sess_arg)
check("argcomp: /session load + /session save key to the same name list",
      lc._arg_completions(_fa, _fc, "/session load") == _sess_arg
      and lc._arg_completions(_fa, _fc, "/session save") == _sess_arg)
check("argcomp: /autosave on/off", lc._arg_completions(_fa, _fc, "/autosave") == ["on", "off"])
check("argcomp: /connect saved + open hosts",
      "dev" in lc._arg_completions(_fa, _fc, "/connect") and "box1" in lc._arg_completions(_fa, _fc, "/connect"))
# CLI contract: a command's flags Tab-complete at ANY arg position (mirrors the handlers
# that parse them order-agnostically) - the `/connect 8 --ephe<Tab>` gap.
check("flagcomp: --ephemeral offered on /connect first arg",
      "--ephemeral" in lc._completion_options(_fa, _fc, "/connect --eph"))
check("flagcomp: --ephemeral offered after a positional arg",
      lc._completion_options(_fa, _fc, "/connect 8 --ephe") == ["--ephemeral"])
check("flagcomp: universal --plain/--fallback offered on any command",
      set(["--plain", "--fallback"]).issubset(set(lc._completion_options(_fa, _fc, "/connect 8 --"))))
check("flagcomp: an already-typed flag is not re-offered",
      "--ephemeral" not in lc._completion_options(_fa, _fc, "/connect --ephemeral 8 --"))
check("argcomp: /local open remotes", lc._arg_completions(_fa, _fc, "/local") == ["box1"])
check("argcomp: unknown command -> no completions", lc._arg_completions(_fa, _fc, "/quit") == [])
check("argcomp: /approve modes", lc._arg_completions(_fa, _fc, "/approve") == list(lc.APPROVAL_MODES))

# --- approval modes (ask | session | auto) replacing the yolo bool ---
check("/approve advertised; /yolo removed entirely",
      "/approve" in lc.SLASH_COMMANDS and "/yolo" not in lc.SLASH_COMMANDS
      and "/yolo" not in lc._BUILTIN_COMMANDS)
lc._session_approved = False
check("auto_approved: auto -> True", lc.auto_approved(lc.Config(approval="auto")) is True)
check("auto_approved: ask -> False", lc.auto_approved(lc.Config(approval="ask")) is False)
check("auto_approved: session unarmed -> False", lc.auto_approved(lc.Config(approval="session")) is False)
_oask = lc._ask
lc._ask = lambda q: True
_csess = lc.Config(approval="session")
check("ask_action: session 'yes' returns True and arms the run",
      lc.ask_action(_csess, "?") is True and lc._session_approved is True)
check("auto_approved: session armed -> True", lc.auto_approved(_csess) is True)
lc.set_approval(_csess, "ask")                       # set_approval clears the arming
check("set_approval resets session arming", lc._session_approved is False)
lc._ask = lambda q: False
check("ask_action: ask 'no' -> False, no arming",
      lc.ask_action(lc.Config(approval="ask"), "?") is False and lc._session_approved is False)
lc._ask = _oask
check("approval_badge: auto shows, ask blank",
      "AUTO" in _re.sub(r"\x1b\[[0-9;]*m", "", lc.approval_badge(lc.Config(approval="auto")))
      and lc.approval_badge(lc.Config(approval="ask")) == "")
lc._session_approved = False                          # no leakage into later tests

# --- /set app-config field editor ---
_scfg = lc.Config()
check("/set advertised + completes its field keys",
      "/set" in lc.SLASH_COMMANDS
      and lc._arg_completions(_fa, _scfg, "/set") == [k for k, _, _ in lc._SETTINGS_FIELDS])
check("/provider set completes provider caps (not app-config keys)",
      "set" in lc._arg_completions(_fa, _scfg, "/provider"))
# arrow-key picker: falls back to the numbered prompt for headless/custom-prompt callers
check("pick_one: numbered fallback honours a custom prompt (index select)",
      lc.pick_one("pick:", ["a", "b", "c"], prompt=lambda _p: "2") == "b")
check("pick_one: numbered fallback cancel (0)",
      lc.pick_one("pick:", ["a", "b"], prompt=lambda _p: "0") is None)
import io as _io4, contextlib as _cl4
_lb = _io4.StringIO()
with _cl4.redirect_stdout(_lb):
    _pv = lc.pick_one("pick:", ["x", "y"], prompt=lambda _p: "2",
                         labels=["* x (styled)", "o y (styled)"])
check("pick_one: labels shown in numbered fallback, value from choices",
      _pv == "y" and "y (styled)" in _lb.getvalue())
check("pick_model_menu: styled label path returns chosen model",
      lc.pick_model_menu(["a:latest", "b:7b"], current="a", warm=["a:latest"],
                         prompt=lambda _p: "2") == "b:7b")
check("_pick_one_tty: returns _NO_TTY when stdin/stdout not a terminal (headless)",
      lc._pick_one_tty("pick:", ["a", "b"]) is lc._NO_TTY)
# shared picker engine: capability gate + --plain emergency escape
check("picker_capable: False when headless (no tty)", lc.picker_capable() is False)
check("_wants_plain: strips --plain and flags it",
      lc._wants_plain("--plain foo") == (True, "foo")
      and lc._wants_plain("--fallback") == (True, "")
      and lc._wants_plain("plain") == (True, "")
      and lc._wants_plain("foo") == (False, "foo"))
# read_key normalises raw bytes/escapes to tokens (feed it a fake raw stdin)
class _FakeIn:
    def __init__(self, s): self._it = iter(s)
    def read(self, n=1):
        try: return next(self._it)
        except StopIteration: return ""
check("read_key: arrow up/down", lc.read_key(_FakeIn("\x1b[A")) == lc._K_UP
      and lc.read_key(_FakeIn("\x1b[B")) == lc._K_DOWN)
check("read_key: enter / ^C / backspace",
      lc.read_key(_FakeIn("\r")) == lc._K_ENTER
      and lc.read_key(_FakeIn("\x03")) == lc._K_CANCEL
      and lc.read_key(_FakeIn("\x7f")) == lc._K_BACK)
check("read_key: pgup/pgdn via CSI ~", lc.read_key(_FakeIn("\x1b[5~")) == lc._K_PGUP
      and lc.read_key(_FakeIn("\x1b[6~")) == lc._K_PGDN)
check("read_key: printable passes through", lc.read_key(_FakeIn("x")) == "x")
check("read_key: unknown escape ignored (None), does NOT cancel",
      lc.read_key(_FakeIn("\x1b[Z")) is None)
# _fit_line: truncate to VISIBLE width so a picker row can never wrap (the smear bug)
check("_fit_line: plain string truncated to width",
      len(lc._ANSI_RE.sub("", lc._fit_line("x" * 100, 10))) == 10)
check("_fit_line: ANSI codes don't count toward width + reset appended on cut",
      (lambda f: len(lc._ANSI_RE.sub("", f)) == 8 and f.endswith("\033[0m"))(
          lc._fit_line("\033[1mhello\033[0m" + "y" * 50, 8)))
check("_fit_line: short string unchanged", lc._fit_line("hi", 40) == "hi")
# _wrap_detail: full row text for the picker detail footer (read the truncated rest)
check("_wrap_detail: wraps + caps at max_lines with ellipsis",
      (lambda d: len(d) == 3 and d[-1].endswith("…"))(
          lc._wrap_detail("a " * 80, max_lines=3, width=40)))
check("_wrap_detail: empty -> no lines", lc._wrap_detail("", width=40) == [])
check("_wrap_detail: short text fits one line", lc._wrap_detail("hi there", width=40) == ["hi there"])
check("_set_setting_field: float", lc._set_setting_field(None, _scfg, "temperature", "0.3") and _scfg.temperature == 0.3)
check("_set_setting_field: int", lc._set_setting_field(None, _scfg, "top_k", "40") and _scfg.top_k == 40)
check("_set_setting_field: bool off", lc._set_setting_field(None, _scfg, "composer", "off") and _scfg.composer is False)
check("_set_setting_field: approval enum routes via set_approval",
      lc._set_setting_field(None, _scfg, "approval", "auto") and _scfg.approval == "auto")
check("_set_setting_field: bad enum rejected", lc._set_setting_field(None, _scfg, "approval", "bogus") is False)
check("_set_setting_field: bad number rejected", lc._set_setting_field(None, _scfg, "temperature", "abc") is False)
check("_set_setting_field: unknown key rejected", lc._set_setting_field(None, _scfg, "nope", "x") is False)
# leash edited via /set must APPLY live (rebuild the tool surface), not just set cfg
_lag = lc.Agent(lc.Config(cwd=FIX, approval="auto", leash="rwe"))
_n_rwe = len(_lag.tool_defs)
lc._set_setting_field(_lag, _lag.cfg, "leash", "r")
check("_set_setting_field: leash applies live (cfg + reduced tool surface)",
      _lag.cfg.leash == "r" and len(_lag.tool_defs) < _n_rwe)
check("_set_setting_field: leash with agent=None just sets cfg (headless-safe)",
      lc._set_setting_field(None, lc.Config(leash="rwe"), "leash", "r") is True)

# --- GOAL C: uniform DEFAULTS / SESSION-OVERRIDE layer -----------------------
# (a) /set (session scope) records a session override, NOT a config default.
_ovc = lc.Config(model="m", temperature=0.7)
_ovc._defaults = {k: getattr(_ovc, k) for k in lc._PERSISTED_SCALAR_KEYS}
lc._set_setting_field(None, _ovc, "temperature", "0.2", scope="session")
check("goal C: /set session override sets live cfg", _ovc.temperature == 0.2)
check("goal C: /set session override recorded in session_overrides",
      _ovc.session_overrides.get("temperature") == 0.2)
check("goal C: /set session override does NOT touch _defaults",
      _ovc._defaults.get("temperature") == 0.7)
# (b) save_config writes scalar lines FROM _defaults, so the override never leaks.
import tomllib as _ttc
_ocp = lc.CONFIG_PATH
_ocf = pathlib.Path(tempfile.mktemp(suffix=".toml")); lc.CONFIG_PATH = _ocf
lc.save_config(_ovc, quiet=True)
check("goal C: save_config writes the DEFAULT, not the session override",
      _ttc.loads(_ocf.read_text()).get("temperature") == 0.7)
# (c) /set --config writes ONE key to the default AND clears the session override.
lc._set_setting_field(None, _ovc, "temperature", "0.9", scope="config")
check("goal C: /set --config updates _defaults", _ovc._defaults.get("temperature") == 0.9)
check("goal C: /set --config clears the session override",
      "temperature" not in _ovc.session_overrides)
lc.save_config(_ovc, quiet=True)
check("goal C: /set --config persisted the new default",
      _ttc.loads(_ocf.read_text()).get("temperature") == 0.9)
# (d) session load applies overrides at RUNTIME only, never writing config.toml.
_load_cfg = lc.Config(model="m", approval="ask")
_load_cfg._defaults = {k: getattr(_load_cfg, k) for k in lc._PERSISTED_SCALAR_KEYS}
class _StubAg:
    def __init__(self): self.messages = [{"role": "system", "content": ""}]; self.pinned_plan = ""
    def refresh_tools(self): pass
    def _system(self): return ""
_meta = {"session_overrides": {"approval": "auto"}, "leash": "rwe"}
_before = _ocf.read_text()
lc._restore_backend_for = (lambda *a, **k: "")  # neutralize backend switch for this stub
lc._restore_session_state(_StubAg(), _load_cfg, _meta)
check("goal C: session load applies the override to live cfg", _load_cfg.approval == "auto")
check("goal C: session load leaves _defaults (config) unchanged",
      _load_cfg._defaults.get("approval") == "ask")
check("goal C: session load did NOT rewrite config.toml", _ocf.read_text() == _before)
# (e) handover_for: session override > per-model handover_overrides > global.
_hc = lc.Config(model="m", handover_soft=0.70,
                handover_overrides={"m": {"soft": 0.50}})
_hc.set_active_model("m")
check("goal C: handover_for uses per-model override when no session override",
      _hc.handover_for("m")["soft"] == 0.50)
_hc.session_overrides["handover_soft"] = 0.33; _hc.handover_soft = 0.33
check("goal C: handover_for session override beats per-model",
      _hc.handover_for("m")["soft"] == 0.33)
lc.CONFIG_PATH = _ocp; _ocf.unlink(missing_ok=True)

# --- leash HARD dispatch lock (defense in depth behind the surface filter) ---
check("_leash_allows_tool: read rides at r", lc._leash_allows_tool("r", "read_file") is True)
check("_leash_allows_tool: write blocked at r", lc._leash_allows_tool("r", "write_file") is False)
check("_leash_allows_tool: write ok at rw", lc._leash_allows_tool("rw", "write_file") is True)
check("_leash_allows_tool: run_command blocked at rw", lc._leash_allows_tool("rw", "run_command") is False)
check("_leash_allows_tool: run_command ok at rwe", lc._leash_allows_tool("rwe", "run_command") is True)
check("_leash_allows_tool: chat blocks even reads", lc._leash_allows_tool("chat", "read_file") is False)
check("_leash_allows_tool: update_plan allowed at any tier", lc._leash_allows_tool("chat", "update_plan") is True)
check("_leash_allows_tool: safe lean-tool rides at r", lc._leash_allows_tool("r", "x", lean_safe=True) is True)
check("_leash_allows_tool: unsafe lean-tool needs rwe", lc._leash_allows_tool("r", "x", lean_safe=False) is False
      and lc._leash_allows_tool("rwe", "x", lean_safe=False) is True)
check("_min_leash_for: write->rw, exec->rwe, read->r",
      lc._min_leash_for("write_file") == "rw" and lc._min_leash_for("run_command") == "rwe"
      and lc._min_leash_for("read_file") == "r")
# the gate must actually REFUSE execution: a run_command at leash r does NOT run
_leak = FIX / "leash_lock_proof.txt"
_leak.unlink(missing_ok=True)
_glag = lc.Agent(lc.Config(cwd=FIX, approval="auto", leash="r"))
_gres = _glag._run_tool("run_command", {"command": f"echo pwned > {_leak}"})
check("_run_tool: above-leash call is hard-blocked (not executed)",
      _gres.startswith("BLOCKED") and not _leak.exists())

# --- per-model tool support: a chat-only model is forced chat regardless of leash ---
check("register_provider: tool_support defaults to True when absent",
      lc.register_provider({**_fake_spec, "name": "ts_default"})["tool_support"]("any") is True)
check("register_provider: tool_support hook respected",
      lc.register_provider({**_fake_spec, "name": "ts_hook",
                            "tool_support": lambda m: m != "compound"})["tool_support"]("compound") is False)
check("active_tools: model_tools=False drops the whole surface (even at rwe)",
      lc.active_tools(lc.Config(leash="rwe"), model_tools=False) == [])
check("active_tools: model_tools=True unchanged at rwe",
      len(lc.active_tools(lc.Config(leash="rwe"), model_tools=True)) > 0)
_mnt = lc._model_no_tools_msg("acme/compound", "read_file")
check("_model_no_tools_msg: says it's the model, not the leash",
      _mnt.startswith("BLOCKED") and "acme/compound" in _mnt and "NOT help" in _mnt)
# end-to-end: chat-only model empties the surface, blocks dispatch, tells the model, shows chat-only
_csa = lc.Agent(lc.Config(cwd=FIX, approval="auto", leash="rwe"))
_csa._model_tool_support = lambda: False          # simulate a provider-reported chat-only model
_csa.refresh_tools()
check("chat-only model: tool surface empties even at leash rwe", _csa.tool_defs == [])
_csres = _csa._run_tool("read_file", {"path": "x"})
check("chat-only model: dispatch blocked at the model level", _csres.startswith("BLOCKED") and "NOT help" in _csres)
check("chat-only model: system prompt tells the model it has no tools",
      "does not support tool calling" in _csa._system())
import re as _re_cs
_save_tty_cs = lc._TTY
lc._TTY = True
_csraw = " ".join(lc._status_rows(_csa, _csa.cfg))
_csrow = _re_cs.sub(r"\x1b\[[0-9;]*m", "", _csraw)
lc._TTY = _save_tty_cs
check("chat-only model: status row shows 'chat-only (model)', not a tool count",
      "chat-only (model)" in _csrow)
check("chat-only model: the chat-only marker is yellow (user-facing heads-up)",
      "chat-only (model)" in lc.yellow("chat-only (model)") and lc.yellow("chat-only (model)") in _csraw)
# REGRESSION: a SAME-PROVIDER /model switch must rebuild the tool surface. A chat-only
# model leaves 0 tools; switching to a tool-capable model on the same provider via
# _apply_model used to leave the surface stale at 0 (showed "0 tools" at rw until a
# /leash forced a refresh). _apply_model now calls refresh_tools.
_swag = lc.Agent(lc.Config(cwd=FIX, leash="rw"))
_swag.active_provider = lambda: {"name": "grq", "tool_support": lambda m: "cap" in m,
                                 "context_window": lambda m: 131072}
_swag.cfg.set_active_model("grq/chatonly"); _swag.refresh_tools()
check("same-provider switch: chat-only model -> 0 tools", len(_swag.tool_defs) == 0)
_swag._apply_model("grq/cap-model")               # same provider, tool-capable
check("same-provider switch: _apply_model rebuilds the surface (no stale 0 at rw)",
      len(_swag.tool_defs) >= 6, [t["function"]["name"] for t in _swag.tool_defs])

# REGRESSION: _maybe_compact must SKIP when the model can't tool-call - it summarizes
# via finalize_handover (a tool), so a chat-only model can never compact; attempting it
# burned a turn + a pre-compact snapshot EVERY turn (the /load flood).
import io as _io_cc, contextlib as _ctx_cc
_cc = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_cc._ctx_zone = lambda: ("hard", 9000, 10000)
_cc._compact_allowed = lambda z: True
_cc_called = []
_cc.auto_handover = lambda z="hard": _cc_called.append(z)
_cc._model_tool_support = lambda: False
with _ctx_cc.redirect_stdout(_io_cc.StringIO()):
    _cc._maybe_handover()
check("_maybe_handover skips compaction when the model can't tool-call", _cc_called == [])
_cc._model_tool_support = lambda: True
with _ctx_cc.redirect_stdout(_io_cc.StringIO()):
    _cc._maybe_handover()
check("_maybe_handover proceeds once the model can tool-call", _cc_called == ["hard"])

# REGRESSION: a FAILED compaction (no summary block) writes NO pre-compact snapshot and
# stamps the loop guard - so a model that never summarizes can't snapshot every turn.
_ncd = Path(tempfile.mkdtemp(prefix="lc_nocompact_")); _sd_save = lc.SESSIONS_DIR
lc.SESSIONS_DIR = _ncd
_nca = lc.Agent(lc.Config(cwd=FIX, approval="auto"))
_nca.messages = [{"role": "system", "content": "S"},
                 {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
_nca._loop = lambda: None              # summary turn yields nothing -> no handover block
_nca._last_compact_ts = 0.0
with _ctx_cc.redirect_stdout(_io_cc.StringIO()):
    _ncr = _nca.auto_handover("hard")
check("failed compaction returns None, history untouched",
      _ncr is None and [m["content"] for m in _nca.messages if m["role"] != "system"] == ["hi", "yo"])
check("failed compaction writes NO pre-handover snapshot (kills the /load flood)",
      list(lc.SESSIONS_DIR.glob(f"*{lc.PREHANDOVER_MARK}*.json")) == [])
check("failed compaction stamps the loop guard (no retry every turn)", _nca._last_compact_ts > 0)
lc.SESSIONS_DIR = _sd_save; shutil.rmtree(_ncd, ignore_errors=True)

# session_info /info reads the active backend, not always cfg.host (fix #3)
_si = lc._load_lean_tool(Path(lc.__file__).parent / "examples/lean-tools/session_info.py")
import io as _io3, contextlib as _cx3
_sicfg = lc.Config(cwd=FIX, host="http://localhost:11434", model="qwen",
                   provider="cloud", provider_settings={"cloud": {"model": "big-model", "num_ctx": 200000}})
class _SiAgent:
    remote = None
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]
_sb = _io3.StringIO()
with _cx3.redirect_stdout(_sb):
    _si._whoami(_SiAgent(), _sicfg, "")
_so = _sb.getvalue()
check("/whoami shows active provider model (not native)", "big-model" in _so and "qwen" not in _so)
check("/whoami endpoint shows provider when active", "cloud" in _so and "localhost" not in _so)
check("/whoami window from provider ctx", "200,000" in _so)

# ============================================================================
# Grouped lean-tools (subdir = group) + group-aware menu rows
# ============================================================================
import tempfile as _tf2, pathlib as _pl2

def _wtool(p, name):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f'TOOL = {{"name": "{name}", "description": "d", '
                 f'"parameters": {{"type": "object", "properties": {{}}}}}}\n'
                 f'def run(args, cwd): return "ok"\n')

_td = _pl2.Path(_tf2.mkdtemp())
_wtool(_td / "aa.py", "aa")
_wtool(_td / "web_dev" / "wd1.py", "wd1")
_wtool(_td / "web_dev" / "wd2.py", "wd2")
_wtool(_td / "db" / "db1.py", "db1")
_gm = lc.LeanToolManager(str(_td), enabled=None)
check("grouped: groups() = root + subdirs, sorted ('' first)",
      _gm.groups() == ["", "db", "web_dev"])
check("grouped: group_of() subdir tool", _gm.group_of("wd1") == "web_dev")
check("grouped: group_of() root tool is ''", _gm.group_of("aa") == "")
check("grouped: names_in_group()", sorted(_gm.names_in_group("web_dev")) == ["wd1", "wd2"])
_rows = lc._lean_tools_rows(_gm, True)
check("grouped: rows start with the root group header", _rows[0] == ("group", ""))
check("grouped: rows interleave group headers + tools",
      ("group", "web_dev") in _rows and ("tool", "wd1") in _rows and ("tool", "db1") in _rows)
check("grouped: flat rows when show_groups is False",
      all(k == "tool" for k, _ in lc._lean_tools_rows(_gm, False)))
# duplicate tool name across groups -> loaded once (warns, keeps first)
_wtool(_td / "web_dev" / "c1.py", "dup")
_wtool(_td / "db" / "c2.py", "dup")
_gm2 = lc.LeanToolManager(str(_td), enabled=None)
check("grouped: duplicate tool name loaded once", _gm2.names().count("dup") == 1)
# a flat dir (no subdirs) still groups as a single root -> menu renders flat
_tdflat = _pl2.Path(_tf2.mkdtemp())
_wtool(_tdflat / "only.py", "only")
_gmf = lc.LeanToolManager(str(_tdflat), enabled=None)
check("grouped: single root group -> show_groups would be False",
      _gmf.groups() == [""])
# multi-dir search path (bundled + user): manager accepts a LIST of dirs, first dir to
# claim a tool name wins (a user drop-in can't shadow a bundled builtin). Mirrors
# ProviderManager / _provider_dirs.
_tdb = _pl2.Path(_tf2.mkdtemp())   # "bundled"
_tdu = _pl2.Path(_tf2.mkdtemp())   # "user"
_wtool(_tdb / "shared.py", "shared")
_tdb_shared = (_tdb / "shared.py")
(_tdu / "shared.py").write_text(
    'TOOL = {"name": "shared", "description": "USERVER", '
    '"parameters": {"type": "object", "properties": {}}}\n'
    'def run(args, cwd): return "user"\n')
_wtool(_tdu / "useronly.py", "useronly")
_gml = lc.LeanToolManager([str(_tdb), str(_tdu)], enabled=None)
check("multi-dir: both dirs scanned (union of names)",
      set(_gml.names()) == {"shared", "useronly"})
check("multi-dir: first dir (bundled) wins on name collision",
      _gml.get("shared")["path"] == _tdb_shared)
check("multi-dir: back-compat single str dir still works",
      set(lc.LeanToolManager(str(_tdu), enabled=None).names()) == {"shared", "useronly"})
check("_lean_tools_dirs: bundled (code-relative) first when present, then user",
      (lambda ds: ds[-1] == _pl2.Path(lc._lean_tools_dir(lc.Config())) and
                  all(isinstance(d, _pl2.Path) for d in ds))(
          lc._lean_tools_dirs(lc.Config())))
# The driver now scans the bundled lean-tools/ dir just like providers/ (list form):
# a bundled tool file is discovered without being dropped in the user dir, and
# builtins.py is EXCLUDED from the scan (it's loaded by _load_builtins, not a plugin).
_bm = lc.LeanToolManager(lc._lean_tools_dirs(lc.Config(cwd=FIX)), enabled=[])
check("bundled lean-tools dir is scanned (dispatch_worker discovered)", "dispatch_worker" in _bm.names())
check("builtins.py is skipped by the manager scan", "builtins" not in _bm.names())
import shutil as _sh2
_sh2.rmtree(_tdb, ignore_errors=True)
_sh2.rmtree(_tdu, ignore_errors=True)
_sh2.rmtree(_td, ignore_errors=True)
_sh2.rmtree(_tdflat, ignore_errors=True)

# ============================================================================
# update.py overlay set: _overlay_rel_paths (what /update refreshes) + apply
# semantics (overlay-only, changed-only, validate-before-write). Headless: build
# a fake "downloaded source tree" and a fake install dir, no network.
# ============================================================================
_upd = lc._load_lean_tool(Path(lc.__file__).parent / "lean-tools/update.py")
_usrc = _pl2.Path(_tf2.mkdtemp())          # pretend extracted tarball root
(_usrc / "lean_coder.py").write_text("# core\n")
(_usrc / "VERSION").write_text("9.9.9\n")
(_usrc / "README.md").write_text("readme\n")
(_usrc / "providers").mkdir()
(_usrc / "providers" / "ollama.py").write_text("# ollama\n")
(_usrc / "providers" / "README.md").write_text("prov readme\n")
(_usrc / "providers" / "notes.txt").write_text("ignore me\n")   # non-.py/.md -> excluded
(_usrc / "lean-tools").mkdir()
(_usrc / "lean-tools" / "update.py").write_text("# update\n")
(_usrc / "lean-tools" / "builtins.py").write_text("# builtins\n")
_rels = set(_upd._overlay_rel_paths(_usrc))
check("update overlay: includes core + shipped root files",
      {"lean_coder.py", "VERSION", "README.md"} <= _rels, str(_rels))
check("update overlay: includes providers/ + lean-tools/ .py and .md",
      {"providers/ollama.py", "providers/README.md", "lean-tools/update.py",
       "lean-tools/builtins.py"} <= _rels, str(_rels))
check("update overlay: excludes non-.py/.md dir files",
      "providers/notes.txt" not in _rels)
check("update overlay: excludes root files absent from the source",
      "install.sh" not in _rels and "uninstall.sh" not in _rels)

# apply(): overlay-only + changed-only. Fake install dir seeded with an OLD core, an
# up-to-date file, and a USER drop-in that must survive untouched.
_udest = _pl2.Path(_tf2.mkdtemp())
(_udest / "lean_coder.py").write_text("# OLD core\n")
(_udest / "VERSION").write_text("9.9.9\n")               # already current -> unchanged
(_udest / "providers").mkdir()
(_udest / "providers" / "ollama.py").write_text("# OLD ollama\n")
(_udest / "providers" / "my_secret.py").write_text("# USER drop-in\n")   # not in src
_ucfg = lc.Config(cwd=FIX)
# stub the tarball download to return our fake src as a gz tarball
import io as _uio, tarfile as _utar
_buf = _uio.BytesIO()
with _utar.open(fileobj=_buf, mode="w:gz") as _tf:
    _tf.add(str(_usrc), arcname="lean-coder-beta")       # GitHub-style single wrapper dir
_blob = _buf.getvalue()
_orig_dl = _upd._download_bytes
_upd._download_bytes = lambda url, timeout=120: (_blob, None)
import io as _uio2, contextlib as _uctx
try:
    _ub = _uio2.StringIO()
    with _uctx.redirect_stdout(_ub):
        _ok = _upd._apply(_ucfg, _udest, lc.dim, lc.green, lc.yellow, lc.red, ask=lambda q: True)
finally:
    _upd._download_bytes = _orig_dl
check("update apply: reports success", _ok is True)
check("update apply: OLD core overwritten", (_udest / "lean_coder.py").read_text() == "# core\n")
check("update apply: backup written", (_udest / "lean_coder.py.bak").read_text() == "# OLD core\n")
check("update apply: bundled provider refreshed", (_udest / "providers" / "ollama.py").read_text() == "# ollama\n")
check("update apply: new bundled file created", (_udest / "providers" / "README.md").read_text() == "prov readme\n")
check("update apply: USER drop-in untouched (overlay-only, never deletes)",
      (_udest / "providers" / "my_secret.py").read_text() == "# USER drop-in\n")
check("update apply: does NOT create excluded non-.py/.md", not (_udest / "providers" / "notes.txt").exists())
# second run: nothing changed -> no-op, returns False
_upd._download_bytes = lambda url, timeout=120: (_blob, None)
try:
    with _uctx.redirect_stdout(_uio2.StringIO()):
        _ok2 = _upd._apply(_ucfg, _udest, lc.dim, lc.green, lc.yellow, lc.red, ask=lambda q: True)
finally:
    _upd._download_bytes = _orig_dl
check("update apply: byte-identical second run is a no-op", _ok2 is False)

# self-heal: when the overlay replaces update.py ITSELF, _apply re-runs the fresh
# updater once so the job finishes under the new logic (no "run it again"). We stub
# _reexec_fresh_updater to record the call rather than actually spawn a subprocess.
_usrc2 = _pl2.Path(_tf2.mkdtemp())
(_usrc2 / "lean_coder.py").write_text("# core2\n")
(_usrc2 / "VERSION").write_text("9.9.9\n")
(_usrc2 / "lean-tools").mkdir()
(_usrc2 / "lean-tools" / "update.py").write_text("# NEW updater\n")
_udest2 = _pl2.Path(_tf2.mkdtemp())
(_udest2 / "lean_coder.py").write_text("# old\n")
(_udest2 / "lean-tools").mkdir()
(_udest2 / "lean-tools" / "update.py").write_text("# OLD updater\n")
_buf2 = _uio.BytesIO()
with _utar.open(fileobj=_buf2, mode="w:gz") as _tf2b:
    _tf2b.add(str(_usrc2), arcname="lean-coder-beta")
_blob2 = _buf2.getvalue()
_reexec_calls = {"n": 0}
_orig_reexec = _upd._reexec_fresh_updater
_upd._reexec_fresh_updater = lambda dest, track: _reexec_calls.update(n=_reexec_calls["n"] + 1) or True
_upd._download_bytes = lambda url, timeout=120: (_blob2, None)
try:
    with _uctx.redirect_stdout(_uio2.StringIO()):
        _upd._apply(_ucfg, _udest2, lc.dim, lc.green, lc.yellow, lc.red, ask=lambda q: True)
finally:
    _upd._download_bytes = _orig_dl
check("update self-heal: re-runs the fresh updater when update.py itself changed",
      _reexec_calls["n"] == 1, f"calls={_reexec_calls['n']}")
# guarded on re-entry: the re-exec sets LEANCODER_UPDATE_REEXEC, so a nested _apply
# does NOT recurse (would spin forever otherwise).
import os as _uos
_reexec_calls["n"] = 0
_uos.environ[_upd._REEXEC_ENV] = "1"
(_udest2 / "lean-tools" / "update.py").write_text("# OLDER updater\n")   # make it differ again
_upd._download_bytes = lambda url, timeout=120: (_blob2, None)
try:
    with _uctx.redirect_stdout(_uio2.StringIO()):
        _upd._apply(_ucfg, _udest2, lc.dim, lc.green, lc.yellow, lc.red, ask=lambda q: True)
finally:
    _upd._download_bytes = _orig_dl
    _uos.environ.pop(_upd._REEXEC_ENV, None)
    _upd._reexec_fresh_updater = _orig_reexec
check("update self-heal: does NOT re-run when already inside a re-exec (loop guard)",
      _reexec_calls["n"] == 0, f"calls={_reexec_calls['n']}")
_sh2.rmtree(_usrc2, ignore_errors=True)
_sh2.rmtree(_udest2, ignore_errors=True)
_sh2.rmtree(_usrc, ignore_errors=True)
_sh2.rmtree(_udest, ignore_errors=True)

# repair_tool_pairs: a dangling tool_use (interrupted turn) must not 400 forever
_rtc = lambda i: {"function": {"name": "read_file", "arguments": {"path": f"f{i}"}},
                  "id": f"toolu_{i}"}
# poisoned: assistant tool_use followed by a user message (no tool_result) - the real bug
poisoned = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "do a thing"},
    {"role": "assistant", "content": "", "tool_calls": [_rtc(1)]},
    {"role": "user", "content": "hi"},                 # <- poison: should be a tool result
]
fixed = lc.repair_tool_pairs(poisoned)
_t_after_asst = fixed[fixed.index(poisoned[2]) + 1]
check("repair backfills a dangling tool_use", _t_after_asst.get("role") == "tool")
check("repair stub carries the tool_call_id", _t_after_asst.get("tool_call_id") == "toolu_1")
check("repair keeps the trailing user message", fixed[-1]["content"] == "hi")
check("repair does not mutate the input", len(poisoned) == 4)
# partial batch: 2 calls, only 1 result -> backfill the 2nd
partial = [
    {"role": "assistant", "content": "", "tool_calls": [_rtc(1), _rtc(2)]},
    {"role": "tool", "tool_name": "read_file", "tool_call_id": "toolu_1", "content": "ok"},
    {"role": "user", "content": "next"},
]
pf = lc.repair_tool_pairs(partial)
_tools = [m for m in pf if m["role"] == "tool"]
check("repair backfills a partial tool batch", len(_tools) == 2
      and _tools[1]["tool_call_id"] == "toolu_2")
# trailing dangling tool_use (last message) -> backfilled too
trailing = lc.repair_tool_pairs(
    [{"role": "assistant", "content": "", "tool_calls": [_rtc(9)]}])
check("repair backfills a trailing dangling tool_use",
      len(trailing) == 2 and trailing[1]["role"] == "tool")
# well-formed history is unchanged (idempotent)
good = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "", "tool_calls": [_rtc(1)]},
    {"role": "tool", "tool_name": "read_file", "tool_call_id": "toolu_1", "content": "ok"},
    {"role": "assistant", "content": "done"},
]
gf = lc.repair_tool_pairs(good)
check("repair is a no-op on well-formed history", gf == good)

# --- MCP client -------------------------------------------------------------
check("mcp tool name namespaces", lc._mcp_tool_name("srv", "tool") == "mcp__srv__tool")
check("mcp name parse round-trips", lc._mcp_parse_tool_name("mcp__srv__tool") == ("srv", "tool"))
check("mcp name parse rejects non-mcp", lc._mcp_parse_tool_name("read_file") == (None, None))
check("mcp name parse rejects malformed", lc._mcp_parse_tool_name("mcp__onlyserver") == (None, None))
check("mcp _sse_json parses plain json",
      lc.MCPConnection._sse_json('{"a":1}') == {"a": 1})
check("mcp _sse_json parses SSE frame",
      lc.MCPConnection._sse_json('event: message\ndata: {"a":2}\n') == {"a": 2})
check("mcp _sse_json None on junk", lc.MCPConnection._sse_json("not json") is None)
check("mcp result: text content", lc._mcp_result_text(
    {"content": [{"type": "text", "text": "hi"}]}) == "hi")
check("mcp result: structuredContent wins", lc._mcp_result_text(
    {"structuredContent": {"x": 1}, "content": [{"type": "text", "text": "hi"}]}).strip().startswith("{"))
check("mcp result: isError flagged", lc._mcp_result_text(
    {"isError": True, "content": [{"type": "text", "text": "boom"}]}).startswith("[tool error]"))
_echo = Path(__file__).resolve().parent / "fixtures" / "echo_mcp_server.py"
if _echo.is_file():
    _srv = {"transport": "stdio", "command": sys.executable, "args": [str(_echo)]}
    _m = lc.MCPManager({"echo": _srv}, enabled=["echo"])
    _rep = _m.connect_enabled()
    check("mcp stdio connects + lists 1 tool", _rep.get("echo") == (True, 1))
    check("mcp stdio schema is namespaced + non-safe",
          [(s["function"]["name"], safe) for s, safe in _m.schemas()] == [("mcp__echo__echo", False)])
    check("mcp stdio call round-trips", _m.call("mcp__echo__echo", {"text": "yo"}) == "echo: yo")
    check("mcp call on disabled server errors",
          _m.call("mcp__nope__x", {}).startswith("error"))
    _m.close()
else:
    check("mcp echo server present (skipped: ref file absent)", True)
_m0 = lc.MCPManager({}, enabled=[])
check("mcp empty manager: no schemas", _m0.schemas() == [])
_c = lc.Config()
check("mcp config defaults empty", _c.mcp_servers == {} and _c.mcp_enabled == [])

shutil.rmtree(FIX, ignore_errors=True)

# Hygiene sweep as a HARD gate: the leak of a real internal IP into tracked source
# slipped through because tests/_sweep.sh (the lint that catches exactly that) was
# never run - no pre-commit hook invoked it, and this smoketest didn't either. Fold
# it in here so the one command we DO run before every commit also fails on a stray
# non-example IP / email / key / MAC / unicode dash. Skipped only if bash is absent.
_sweep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sweep.sh")
if os.path.isfile(_sweep) and shutil.which("bash"):
    _sw = subprocess.run(["bash", _sweep], capture_output=True, text=True)
    if _sw.returncode == 0:
        check("hygiene sweep clean (no stray IP/email/key/unicode in tracked source)", True)
    else:
        _tail = (_sw.stdout or "").strip().splitlines()
        check("hygiene sweep clean (no stray IP/email/key/unicode in tracked source)",
              False, "\n    " + "\n    ".join(_tail[-12:]))

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
