"""End-to-end tests of the agent loop against a mock Ollama /api/chat server,
plus unit tests of the text-encoded tool-call fallback parser.

Covers three first-turn shapes:
  - native  : model populates message.tool_calls (unchanged native path)
  - xml      : model emits Qwen <function=…><parameter=…> text, empty tool_calls
  - json     : model emits <tool_call>{json}</tool_call> text, empty tool_calls
"""
import json, shutil, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (this file lives in tests/)
import lean_coder as lc

# Redirect CONFIG_PATH to a throwaway temp file BEFORE any test runs. Scenarios build
# real Agents/Configs and run turns; anything that triggers autosave_config/save_config
# would otherwise write to the developer's REAL ~/.config/leancoder/config.toml - which
# leaked the mock server's ephemeral host port into it and corrupted it every run.
lc.CONFIG_PATH = Path(tempfile.mktemp(suffix=".toml"))

ok = True
def check(name, cond, extra=""):
    global ok; ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))

# --- ground-truth text the real model emitted (incl. stray </tool_call>) -----
QWEN_XML = ("<function=read_file>\n"
            "<parameter=path>\n"
            "calc.py\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>")
JSON_TC = '<tool_call>{"name": "read_file", "arguments": {"path": "foo.txt"}}</tool_call>'

# ============================================================================
# Unit tests: parse_text_tool_calls (no server)
# ============================================================================
calls, cleaned = lc.parse_text_tool_calls(QWEN_XML)
check("xml: parses exactly 1 call", len(calls) == 1, str(calls))
check("xml: name == read_file", calls and calls[0]["function"]["name"] == "read_file")
check("xml: args path == calc.py (newlines stripped)",
      calls and calls[0]["function"]["arguments"] == {"path": "calc.py"},
      str(calls[0]["function"]["arguments"]) if calls else "")
check("xml: content cleaned (no <function=, no stray </tool_call>)",
      "<function=" not in cleaned and "tool_call" not in cleaned, repr(cleaned))

# multi-line value: only outer newline stripped, inner preserved (diff/content args)
multi = ("<function=apply_diff><parameter=path>a.py</parameter>"
         "<parameter=diff>\nline1\nline2\n</parameter></function>")
c2, _ = lc.parse_text_tool_calls(multi)
check("xml: multi-param parsed", len(c2) == 1 and len(c2[0]["function"]["arguments"]) == 2)
check("xml: inner newlines preserved in multi-line value",
      c2 and c2[0]["function"]["arguments"]["diff"] == "line1\nline2",
      repr(c2[0]["function"]["arguments"]["diff"]) if c2 else "")

# prose around the call is preserved
c3, clean3 = lc.parse_text_tool_calls("Sure, reading it.\n" + QWEN_XML)
check("xml: surrounding prose preserved", clean3 == "Sure, reading it.", repr(clean3))

# JSON-in-tags: dict args
c4, _ = lc.parse_text_tool_calls(JSON_TC)
check("json: parses 1 call w/ dict args",
      len(c4) == 1 and c4[0]["function"]["arguments"] == {"path": "foo.txt"}, str(c4))

# JSON-in-tags: arguments as a JSON *string* must be json.loads'd to a dict
json_str_args = '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"foo.txt\\"}"}</tool_call>'
c5, _ = lc.parse_text_tool_calls(json_str_args)
check("json: string-encoded arguments decoded to dict",
      len(c5) == 1 and c5[0]["function"]["arguments"] == {"path": "foo.txt"}, str(c5))

# malformed / partial markup -> no calls, content unchanged (acts as final answer)
c6, clean6 = lc.parse_text_tool_calls("here is <function=read_file> but no close")
check("malformed markup -> no calls, unchanged", c6 == [] and clean6.startswith("here is"))
check("plain prose -> no calls", lc.parse_text_tool_calls("just a normal answer")[0] == [])

# --- L2a: content-fallback scanner (small models that emit the call as TEXT) ---
_KN = {"read_file", "apply_diff", "write_file"}
# bare JSON (the exact qwen3:4b real-world failure) - whole content is the object
_bare = ('{"name":"apply_diff","arguments":{"path":"greet.py",'
         '"diff":"<<<<<<< SEARCH\\nprint(1)\\n>>>>>>> REPLACE"}}')
_cb, _clb = lc.parse_text_tool_calls(_bare, _KN)
check("bare JSON -> promoted call, content emptied",
      len(_cb) == 1 and _cb[0]["function"]["name"] == "apply_diff" and _clb == "",
      str(_cb))
check("bare JSON: multiline diff arg preserved byte-exact",
      _cb and _cb[0]["function"]["arguments"]["diff"] == "<<<<<<< SEARCH\nprint(1)\n>>>>>>> REPLACE")
# fenced ```json block
_fenced = '```json\n{"name":"read_file","arguments":{"path":"x.py"}}\n```'
_cf, _ = lc.parse_text_tool_calls(_fenced, _KN)
check("fenced ```json -> parsed", len(_cf) == 1 and _cf[0]["function"]["name"] == "read_file")
# OpenAI wrapper shape
_wrap = '{"function":{"name":"read_file","arguments":{"path":"a"}}}'
check("OpenAI wrapper {function:{...}} -> parsed",
      lc.parse_text_tool_calls(_wrap, _KN)[0][0]["function"]["name"] == "read_file")
# tool_calls array wrapper
_arr = '{"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a"}}}]}'
check("tool_calls:[...] array -> parsed",
      lc.parse_text_tool_calls(_arr, _KN)[0][0]["function"]["name"] == "read_file")
# GUARD 1: prose that merely lists/mentions tools must NOT fire a call
_prose = ("I have these tools: read_file reads a file, write_file writes one. "
          "Which would you like me to use?")
check("guard: prose mentioning tools -> NO call (final answer)",
      lc.parse_text_tool_calls(_prose, _KN)[0] == [])
# GUARD 2: an unknown tool name is rejected even if perfectly shaped
_unk = '{"name":"delete_everything","arguments":{}}'
check("guard: unknown tool name rejected", lc.parse_text_tool_calls(_unk, _KN)[0] == [])
# GUARD 3: a bare object embedded in real prose (slack exceeded) is NOT a call
_embed = ('Here is an example you could send: {"name":"read_file",'
          '"arguments":{"path":"x"}} - but I need you to confirm the path first please.')
check("guard: JSON embedded in prose -> NO call",
      lc.parse_text_tool_calls(_embed, _KN)[0] == [])
# BACKTICK GOTCHA: write_file whose content arg contains ``` must not break parsing
_bt = ('{"name":"write_file","arguments":{"path":"r.md","content":'
       '"# Title\\n```python\\nx=1\\n```\\ndone"}}')
_cbt, _ = lc.parse_text_tool_calls(_bt, _KN)
check("backtick-in-arg: write_file content with ``` parses whole, arg intact",
      len(_cbt) == 1 and "```python" in _cbt[0]["function"]["arguments"]["content"])
# registry gate off (known_names=None) still parses (back-compat for callers w/o registry)
check("no registry -> still parses bare JSON",
      lc.parse_text_tool_calls(_bare)[0][0]["function"]["name"] == "apply_diff")

# ============================================================================
# End-to-end tests: drive Agent.run_turn through a mock server
# ============================================================================
FIX = Path(tempfile.mkdtemp(prefix="leancoder_mock_"))
(FIX / "foo.txt").write_text("hello world\nsecond line\n")
(FIX / "calc.py").write_text("def add(a, b):\n    return a - b\n")

MODE = "native"
counter = {"n": 0}
bodies = []   # every request body the server received (for payload assertions)
BIG_MODEL = "bigmystery:latest"   # unknown family + big /api/show size -> strong path
cold_state = {"warmed": False}   # coldload scenario: flips after the first (cold) turn

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass
    def do_GET(self):
        if self.path == "/api/tags":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models":[{"name":"demo-coder"}]}')
        elif self.path == "/api/ps":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models":[{"name":"demo-coder"}]}')
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        # /api/show: report a BIG parameter size for the unknown-family strong
        # model, else a small one - drives the size-based profile fallback.
        if self.path == "/api/show":
            psize = "70.0B" if body.get("model") == BIG_MODEL else "3.0B"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"details": {"parameter_size": psize}}).encode())
            return
        bodies.append(body)
        msgs = body["messages"]
        counter["n"] += 1
        if MODE in ("nothink", "nothink400") and body.get("think") is not None:
            # Model rejects `think`: the first request (carries think) is refused;
            # the client must retry WITHOUT think and get a real answer. Two refusal
            # shapes seen in the wild: a streamed {"error":...} line on a 200
            # (nothink), and an HTTP 400 whose body is {"error":...} (nothink400 -
            # what 203.0.113.216 actually does). Both must trigger the same retry.
            if MODE == "nothink400":
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(
                        b'{"error": "\\"kilo-coder:latest\\" does not support thinking"}')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self._emit({"error": '"kilo-coder:latest" does not support thinking'})
            return
        if MODE == "coldload":
            # Cold-model streaming quirk (measured on a real remote ollama): the first
            # stream:true call kicks off the VRAM load and CLOSES the stream with a
            # single empty placeholder frame (no done:true). The client must then do a
            # blocking preload (empty messages, stream:false -> done_reason:"load") and
            # retry the stream, which now answers. counter tracks call order.
            is_preload = not msgs and body.get("stream") is False
            if is_preload:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps({
                        "model": "kilo-coder:latest",
                        "message": {"role": "assistant", "content": ""},
                        "done": True, "done_reason": "load"}).encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            if not cold_state["warmed"]:
                # first real turn: cold -> one empty frame, stream closes (no done:true)
                cold_state["warmed"] = True
                self._emit({"model": "", "message": {"role": "", "content": ""},
                            "done": False})
                return
            # after preload: warm -> real answer
            self._emit({"message": {"role": "assistant",
                        "content": "warm answer"}, "done": False})
            self._emit({"message": {"role": "assistant", "content": ""},
                        "done": True, "done_reason": "stop",
                        "prompt_eval_count": 200, "eval_count": 3})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if MODE in ("nothink", "nothink400"):
            # retry (no think) - real answer
            self._emit({"message": {"role": "assistant",
                        "content": "no-think answer"}, "done": False})
            self._emit({"message": {"role": "assistant", "content": ""},
                        "done": True, "prompt_eval_count": 128, "eval_count": 4})
            return
        if msgs[-1].get("role") != "tool":
            # First model turn - shape depends on MODE.
            if MODE == "native":
                self._emit({"message": {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {
                                "name": "read_file",
                                "arguments": {"path": "foo.txt"}}}]}, "done": False})
            elif MODE == "xml":
                self._emit({"message": {"role": "assistant", "content": QWEN_XML},
                            "done": False})
            elif MODE == "json":
                self._emit({"message": {"role": "assistant", "content": JSON_TC},
                            "done": False})
            elif MODE == "think":
                # reasoning arrives in message.thinking, separate from content
                self._emit({"message": {"role": "assistant",
                            "thinking": "let me reason about this", "content": ""},
                            "done": False})
                self._emit({"message": {"role": "assistant",
                            "content": "final answer here"}, "done": False})
            elif MODE == "summary":
                self._emit({"message": {"role": "assistant",
                            "content": "GOAL: bind ollama. NEXT: restart svc."},
                            "done": False})
            elif MODE == "handover":
                # agentic /handover: write the three marked blocks as visible text and
                # STOP (no finalize tool - the turn ending is the trigger).
                self._emit({"message": {"role": "assistant",
                            "content": "Updated the docs.\n===HANDOVER===\nGOAL: bind ollama. "
                                       "NEXT: restart svc.\n===HANDOVER===\n"
                                       "===NEXT===\nrestart the ollama service\n===NEXT===\n"
                                       "===PLAN===\nGOAL: bind ollama to labnet\nTODO:\n"
                                       "- [x] edit config\n- [ ] restart svc\n===PLAN==="},
                            "done": False})
            elif MODE == "stream5":
                for k in range(5):
                    self._emit({"message": {"role": "assistant",
                                "content": f"tok{k} "}, "done": False})
            self._emit({"message": {"role": "assistant", "content": ""},
                        "done": True, "prompt_eval_count": 512, "eval_count": 8})
        else:
            for tok in ["Done", " - ", "summarized."]:
                self._emit({"message": {"role": "assistant", "content": tok},
                            "done": False})
            self._emit({"message": {"role": "assistant", "content": ""},
                        "done": True, "prompt_eval_count": 690, "eval_count": 12})
    def _emit(self, obj):
        # The client aborts the stream mid-flight in the abort scenario; writing
        # to the closed socket is expected, so swallow the drop instead of
        # letting it bubble up as a traceback.
        try:
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class QuietHTTPServer(HTTPServer):
    def handle_error(self, request, client_address):
        # Keep suite output clean: a dropped client connection is expected here
        # and must never print a traceback that could hide a real one. Re-raise
        # anything else by deferring to the default handler.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


srv = QuietHTTPServer(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
cfg = lc.Config(cwd=FIX, approval="auto", host=f"http://127.0.0.1:{port}")

# The ollama client + host machinery now live in the bundled provider file, not
# core. Load it the way the manager does and run setup() so its injected helpers
# bind; mk_agent() bootstraps an ollama client onto a fresh Agent (Agent.__init__
# no longer creates one - the provider does, at activation, in production).
import pathlib as _plib
olm = lc._load_lean_tool(_plib.Path(__file__).resolve().parent.parent / "providers" / "ollama.py")
olm.setup(lc.__dict__, cfg)
def mk_agent(c):
    a = lc.Agent(c)
    a.client = olm.OllamaClient(a.cfg)
    return a


def run_scenario(mode, marker):
    global MODE
    MODE = mode
    counter["n"] = 0
    agent = mk_agent(cfg)
    print(f"\n--- scenario: {mode} ---")
    agent.run_turn("read the file and summarize")
    check(f"[{mode}] tool round-trip happened (no early exit)", counter["n"] == 2,
          f"round-trips={counter['n']}")
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    check(f"[{mode}] history has a role:tool message", len(tool_msgs) == 1)
    check(f"[{mode}] tool result carries file contents",
          tool_msgs and marker in tool_msgs[0]["content"])
    tc_msgs = [m for m in agent.messages
               if m["role"] == "assistant" and m.get("tool_calls")]
    check(f"[{mode}] assistant message has tool_calls populated", len(tc_msgs) == 1)
    if tc_msgs:
        check(f"[{mode}] stored assistant content cleaned of <function=",
              "<function=" not in tc_msgs[0]["content"]
              and "<tool_call>" not in tc_msgs[0]["content"],
              repr(tc_msgs[0]["content"]))
    final = agent.messages[-1]
    check(f"[{mode}] final answer has content, no tool_calls",
          final["role"] == "assistant" and final["content"].strip() != ""
          and "tool_calls" not in final)


run_scenario("native", "hello world")   # existing native path must still pass
run_scenario("xml", "return a - b")      # Qwen XML fallback (reads calc.py)
run_scenario("json", "hello world")      # JSON-in-tags fallback (reads foo.txt)

# --- /think: sends think param; reasoning is shown but NOT stored ------------
print("\n--- scenario: think ---")
MODE = "think"; counter["n"] = 0; bodies.clear()
cfg.think = True
agent = mk_agent(cfg)
agent.run_turn("answer with reasoning")
cfg.think = None  # reset shared cfg
check("[think] payload carried think=true", bodies and bodies[-1].get("think") is True,
      str(bodies[-1].get("think")) if bodies else "no body")
final = agent.messages[-1]
check("[think] final answer content stored", final["content"] == "final answer here",
      repr(final["content"]))
check("[think] reasoning NOT stored in history",
      not any("reason about this" in (m.get("content") or "") for m in agent.messages))

# think param omitted entirely when unset (default behavior preserved)
MODE = "native"; counter["n"] = 0; bodies.clear()
mk_agent(lc.Config(cwd=FIX, approval="auto", host=cfg.host)).run_turn("hi")
check("[think] param omitted when unset", all("think" not in b for b in bodies))

# --- no-think retry: model that rejects `think` gets a transparent retry -------
print("\n--- scenario: nothink ---")
MODE = "nothink"; counter["n"] = 0; bodies.clear()
cfg.think = True
nt_agent = mk_agent(cfg)
nt_agent.run_turn("answer")
cfg.think = None  # reset shared cfg
check("[nothink] first request carried think, then retried without it",
      len(bodies) == 2 and bodies[0].get("think") is True
      and "think" not in bodies[1],
      f"bodies={[b.get('think') for b in bodies]}")
nt_final = nt_agent.messages[-1]
check("[nothink] retry produced the real answer",
      nt_final["role"] == "assistant" and nt_final["content"] == "no-think answer",
      repr(nt_final["content"]))
check("[nothink] model cached so think never re-sent",
      nt_agent.client._no_think == {cfg.active_model()},
      str(nt_agent.client._no_think))
# a subsequent turn on the cached model must NOT carry think at all
bodies.clear(); counter["n"] = 0
cfg.think = True
nt_agent.run_turn("again")
cfg.think = None
check("[nothink] cached model omits think on later turns",
      bodies and all("think" not in b for b in bodies),
      f"bodies={[b.get('think') for b in bodies]}")

# --- no-think retry, HTTP 400 shape: what 203.0.113.216 actually returns --------
# The refusal comes as an HTTP 400 (urlopen raises HTTPError) with the error in the
# body, NOT a streamed error line - the shape a live test caught that this mock had
# missed. Same transparent retry must fire.
print("\n--- scenario: nothink400 ---")
MODE = "nothink400"; counter["n"] = 0; bodies.clear()
cfg.think = True
nt4 = mk_agent(cfg)
nt4.run_turn("answer")
cfg.think = None
check("[nothink400] first request carried think, then retried without it",
      len(bodies) == 2 and bodies[0].get("think") is True
      and "think" not in bodies[1],
      f"bodies={[b.get('think') for b in bodies]}")
nt4_final = nt4.messages[-1]
check("[nothink400] retry produced the real answer",
      nt4_final["role"] == "assistant" and nt4_final["content"] == "no-think answer",
      repr(nt4_final["content"]))
check("[nothink400] model cached after the 400 retry",
      nt4.client._no_think == {cfg.active_model()},
      str(nt4.client._no_think))

# --- cold-load blank: first stream turn on a cold model closes empty; client
# must blocking-preload (empty messages, stream:false) then retry the stream ------
print("\n--- scenario: coldload ---")
MODE = "coldload"; counter["n"] = 0; bodies.clear()
cold_state["warmed"] = False
cfg.think = None
cl = mk_agent(cfg)
cl.run_turn("say hi")
cl_final = cl.messages[-1]
# three requests expected: cold stream (empty close), preload (stream:false, empty
# messages), warm stream (real answer).
_preloads = [b for b in bodies if not b.get("messages") and b.get("stream") is False]
check("[coldload] a blocking preload (empty messages, stream:false) was fired",
      len(_preloads) == 1, f"preloads={len(_preloads)} of {len(bodies)} reqs")
check("[coldload] retried the stream after preload and got the real answer",
      cl_final["role"] == "assistant" and cl_final["content"] == "warm answer",
      repr(cl_final["content"]))
check("[coldload] exactly 3 requests (cold stream, preload, warm stream)",
      len(bodies) == 3, f"n={len(bodies)}")
# a warm run must NOT preload or retry (no cold blank -> single request)
MODE = "coldload"; counter["n"] = 0; bodies.clear()
cold_state["warmed"] = True   # already warm
cl2 = mk_agent(cfg)
cl2.run_turn("again")
check("[coldload] warm run does not preload/retry (single stream request)",
      len(bodies) == 1 and cl2.messages[-1]["content"] == "warm answer",
      f"n={len(bodies)}")

# --- clean SIGINT path: chat() aborts cooperatively via should_abort ---------
print("\n--- scenario: abort ---")
MODE = "stream5"; counter["n"] = 0
client = olm.OllamaClient(cfg)
state = {"i": 0}
def abrt():
    state["i"] += 1
    return state["i"] > 2          # abort after a couple of streamed lines
asst, pe, aborted = client.chat([{"role": "user", "content": "x"}], lc.TOOLS,
                                should_abort=abrt)
check("[abort] chat reports aborted", aborted is True)
check("[abort] returns only partial content (stopped early)",
      asst["content"].count("tok") < 5, repr(asst["content"]))
check("[abort] no spurious tool_calls on abort", "tool_calls" not in asst)

# --- /handover: summarize the session, then replace history with the summary -
print("\n--- scenario: handover ---")
MODE = "handover"; counter["n"] = 0; bodies.clear()
agent = mk_agent(cfg)
agent.messages.append({"role": "user", "content": "bind ollama to labnet"})
agent.messages.append({"role": "tool", "tool_name": "run_command",
                       "content": "\n".join(["log line"] * 80)})
summary = agent.handover()
check("[handover] returns the marked block", bool(summary) and "GOAL" in summary,
      repr(summary))
check("[handover] block extracted, not the surrounding prose",
      summary and "Updated the docs" not in summary)
check("[handover] finalize_handover tool is NOT surfaced (dropped)",
      bodies and not any(t["function"]["name"] == "finalize_handover"
                         for t in bodies[-1].get("tools", [])))
check("[handover] history replaced with system + one summary turn",
      [m["role"] for m in agent.messages] == ["system", "user"]
      and "summarized" in agent.messages[1]["content"])
check("[handover] old tool dump is gone from history",
      not any("log line" in (m.get("content") or "") for m in agent.messages))
check("[handover] stale token count cleared", agent.last_prompt_tokens is None)

# --- auto-compaction: summarize OLD, KEEP recent tail, queue autostart self-prompt -
print("\n--- scenario: auto_handover ---")
MODE = "handover"; counter["n"] = 0; bodies.clear()
agent = mk_agent(cfg)
agent.cfg.autostart_after_handover = True   # opt in (default is off) to test the queue path
_saved_wm = agent.cfg.window_messages
agent.cfg.window_messages = 4          # keep a small recent tail
agent.messages.append({"role": "user", "content": "old goal"})
agent.messages.append({"role": "tool", "tool_name": "run_command",
                       "content": "\n".join(["OLDDUMPLINE"] * 80)})
agent.messages.append({"role": "assistant", "content": "did old thing"})
agent.messages.append({"role": "user", "content": "recent task"})
agent.messages.append({"role": "assistant", "content": "working on recent"})
block = agent.auto_handover("hard")
check("[compact] returns summary block", bool(block) and "GOAL" in block, repr(block))
check("[compact] keeps system + summary head", [m["role"] for m in agent.messages][:2] == ["system", "user"])
check("[compact] summary carries the cache_boundary signal",
      agent.messages[1].get("cache_boundary") is True and agent._handover_boundary == 2)
check("[compact] recent tail kept verbatim",
      any(m.get("content") == "recent task" for m in agent.messages))
check("[compact] old tool dump dropped",
      not any("OLDDUMPLINE" in (m.get("content") or "") for m in agent.messages))
check("[compact] self-prompt queued for autostart",
      agent._autostart_pending == "restart the ollama service", repr(agent._autostart_pending))
check("[compact] loop-guard clock stamped", agent._last_compact_ts > 0)
check("[compact] plan pinned from ===PLAN=== block",
      "bind ollama to labnet" in agent.pinned_plan and "restart svc" in agent.pinned_plan)
check("[compact] pinned plan stays OUT of the system prompt (uncached)",
      "bind ollama to labnet" not in agent.messages[0]["content"])
check("[compact] pinned plan available via _plan_reminder",
      "bind ollama to labnet" in agent._plan_reminder())
agent.cfg.window_messages = _saved_wm
check("[handover] finalize tool never on the surface",
      not any(t["function"]["name"] == "finalize_handover" for t in agent.tool_defs))

# handover on an empty session -> None (nothing to hand over)
check("[handover] empty session -> None",
      mk_agent(lc.Config(cwd=FIX, approval="auto", host=cfg.host)).handover() is None)

# --- tiered host failover: live probe against the mock + a dead port ---------
print("\n--- scenario: host failover ---")
DEAD = "http://127.0.0.1:1"
check("[hosts] host_alive true for live mock", olm.host_alive(cfg.host, timeout=2) is True)
check("[hosts] host_alive false for dead port", olm.host_alive(DEAD, timeout=2) is False)
chosen, alive = olm.select_host([DEAD, cfg.host], timeout=2)
check("[hosts] select_host skips dead, picks live", chosen == cfg.host and alive is True)
chosen2, alive2 = olm.select_host([DEAD], timeout=2)
check("[hosts] all dead -> top host, alive False", chosen2 == DEAD and alive2 is False)

# --- per-host model check against the mock's /api/tags (serves "demo-coder") -
print("\n--- scenario: model availability ---")
mc_ok = lc.Config(cwd=FIX, host=cfg.host, model="demo-coder")
olm.ensure_model(olm.OllamaClient(mc_ok), mc_ok, interactive=False)
check("[model] present on host -> unchanged", mc_ok.model == "demo-coder")
mc_bad = lc.Config(cwd=FIX, host=cfg.host, model="not-installed")
olm.ensure_model(olm.OllamaClient(mc_bad), mc_bad, interactive=False)
check("[model] absent + non-interactive -> unchanged (warned)", mc_bad.model == "not-installed")

# --- /api/ps -> running ('warm') models, for the /model picker's green dots ----
print("\n--- scenario: running models (/api/ps) ---")
warm = olm.OllamaClient(cfg).running_models()
check("[ps] running_models parses /api/ps", warm == ["demo-coder"])
dead_warm = olm.OllamaClient(lc.Config(cwd=FIX, host=DEAD)).running_models()
check("[ps] running_models -> [] when host unreachable", dead_warm == [])

# --- Small-model classification by MEASURED SIZE (no name matching) ---
check("[size] parse_param_b: B/M/None",
      olm.parse_param_b("4.0B") == 4.0 and olm.parse_param_b("30.5B") == 30.5
      and olm.parse_param_b("540M") == 0.54 and olm.parse_param_b(None) is None)
check("[size] small param_b -> small", olm.is_small_model(3.0) and olm.is_small_model(8.0))
check("[size] big param_b -> not small",
      not olm.is_small_model(30.5) and not olm.is_small_model(8.01))
check("[size] unknown size (None) -> assume small (safe default)",
      olm.is_small_model(None))
# The key regression: a big 'qwen3-coder' must NOT be classed small. With
# name matching gone, classification depends only on measured size, so a 30B
# model is strong regardless of its name.
check("[size] name is irrelevant: big size -> strong even if name says qwen3",
      not olm.is_small_model(30.5))
# --- system_addendum: size-gated, tool-channel only, NO write_file steer ---
_cfg_small = lc.Config(cwd=FIX, host=cfg.host); _cfg_small.set_active_model("qwen3:4b")
_add = olm._system_addendum(None, _cfg_small)
check("[addendum] small model (mock /api/show small) gets tool-channel guidance",
      "tool" in _add.lower() and len(_add) > 0)
check("[addendum] no write_file/anti-diff steer (apply_diff kept for all)",
      "write_file" not in _add)
_cfg_big = lc.Config(cwd=FIX, host=cfg.host); _cfg_big.set_active_model(BIG_MODEL)
check("[addendum] strong model (mock /api/show big) -> empty addendum",
      olm._system_addendum(None, _cfg_big) == "")
# --- name matching is GONE from the provider ---
check("[clean] MODEL_PROFILES / model_profile / _tool_filter removed",
      not hasattr(olm, "MODEL_PROFILES") and not hasattr(olm, "model_profile")
      and not hasattr(olm, "_tool_filter") and not hasattr(olm, "_family_known"))
check("[clean] no tool_filter hook advertised (apply_diff kept for everyone)",
      olm.PROVIDER.get("tool_filter") is None)

# ============================================================================
# Event-driven WAKE on bg-task finish (headless: dedup + framing, no TTY)
# ============================================================================
# bg_wake_turn shares source + dedup (_bg_finished_here + _bg_announced) with the
# passive _bg_finished_note, so a finished task is claimed by whichever fires first
# and is NEVER double-reported. We stub _bg_finished_note to a fixed notice and
# assert the wake framing + one-shot behaviour (the announce set is _bg_finished_note's
# job; here we prove bg_wake_turn is a thin, correct wrapper over it).
_wcfg = lc.Config(cwd=FIX, host=cfg.host)
_wagent = lc.Agent(_wcfg)
# no finished task -> empty (no spurious wake)
_wagent._bg_finished_note = lambda: ""
check("[wake] no finished task -> empty (agent stays asleep)", _wagent.bg_wake_turn() == "")
# a finished task -> framed autonomous-wake turn carrying the notice
_NOTICE = "background task finished: `eval run` exited 0\nlast output:\ndone 21/21"
_wagent._bg_finished_note = lambda: _NOTICE
_w = _wagent.bg_wake_turn()
check("[wake] finished task -> non-empty synthesised turn", bool(_w))
check("[wake] wake turn carries the finish notice", _NOTICE in _w)
check("[wake] wake turn is flagged autonomous (no operator input)",
      "autonomous wake" in _w.lower())
check("[wake] wake turn instructs the agent to react", "react" in _w.lower())
# opt-in gate is OFF by default (autonomy never fires unless explicitly enabled)
check("[wake] wake_on_bg_finish defaults OFF", lc.Config(cwd=FIX, host=cfg.host).wake_on_bg_finish is False)
# read_line accepts the wake_check kwarg (the idle-poll hook)
import inspect as _insp
check("[wake] Composer.read_line exposes wake_check hook",
      "wake_check" in _insp.signature(lc.Composer.read_line).parameters)
# --- worker wake via the hook registry (dispatch_worker registers its notice) ---
check("[wake] register_wake_hook exposed to lean-tools", callable(getattr(lc, "register_wake_hook", None)))
_saved_hooks = list(lc._WAKE_HOOKS)
lc._WAKE_HOOKS.clear()
_wagent2 = lc.Agent(lc.Config(cwd=FIX, host=cfg.host))
_wagent2._bg_finished_note = lambda: ""          # no plain-bg task finished
_WNOTE = "[worker 101 finished (r, devstral:24b) - task: scout\nresult:\nfound it]"
lc.register_wake_hook(lambda: _WNOTE)            # a worker finished
_wt = _wagent2.bg_wake_turn()
check("[wake] worker-finish hook wakes the agent", bool(_wt) and _WNOTE in _wt)
check("[wake] worker wake carries autonomous framing", "autonomous wake" in _wt.lower())
# aggregation: both a plain task AND a worker in one wake turn
_wagent2._bg_finished_note = lambda: "background task finished: `x` exited 0"
_wt2 = _wagent2.bg_wake_turn()
check("[wake] aggregates task + worker in one wake turn",
      "background task finished" in _wt2 and _WNOTE in _wt2)
# a raising hook must not break the wake turn (best-effort)
lc._WAKE_HOOKS.clear()
_wagent2._bg_finished_note = lambda: ""
lc.register_wake_hook(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
check("[wake] a raising hook is swallowed (no wake, no crash)", _wagent2.bg_wake_turn() == "")
# register is idempotent
lc._WAKE_HOOKS.clear()
_fn = lambda: ""
lc.register_wake_hook(_fn); lc.register_wake_hook(_fn)
check("[wake] register_wake_hook is idempotent", lc._WAKE_HOOKS.count(_fn) == 1)
lc._WAKE_HOOKS[:] = _saved_hooks                 # restore
# --- non-TTY-but-live-stdin wake: _input_or_wake polls + wakes without input ---
check("[wake] _input_or_wake plain input() when no wake_check", callable(lc._input_or_wake))
# wake_check fires -> returns the synthesised turn without any stdin line. Use a
# throwaway pipe as stdin so select() has a real (never-ready) fd to poll.
import os as _os
_r, _w = _os.pipe()
_old_stdin = sys.stdin
try:
    sys.stdin = _os.fdopen(_r)
    _fired = {"n": 0}
    def _wc():
        _fired["n"] += 1
        return "WOKE" if _fired["n"] >= 2 else ""   # fire on the 2nd idle tick
    _got = lc._input_or_wake("", wake_check=_wc, tick=0.01)
    check("[wake] _input_or_wake returns the wake turn with no stdin input", _got == "WOKE")
finally:
    sys.stdin = _old_stdin
    _os.close(_w)

srv.shutdown()
shutil.rmtree(FIX, ignore_errors=True)
print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
