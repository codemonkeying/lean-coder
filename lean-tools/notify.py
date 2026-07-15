"""notify - ping the operator when a whole TURN finishes (control returns to you),
or when the agent needs your input (a confirmation prompt). So you can look away
during a long task and get called back. It pings once at the end of the turn - NOT
per tool call/command the agent runs mid-turn.

This is the canonical *driver-only* lean-tool: it hooks the machine you're sitting
at, so it is a setup() lean-tool, NOT a pushed TOOL. A TOOL would be pushed to the
remote executor and fire on the wrong box; setup() runs only on the driver and
adds nothing to the model's tool surface. Contrast every other example, which
adds a tool the model calls.

How it pings: a best-effort desktop notification (notify-send on Linux,
osascript on macOS) - VISUAL. Any sound that makes is your OS's notification
setting, not ours. There is NO audible terminal bell by default; it's opt-in via
`/notify bell`. Toggle the whole thing with `/notify`.
"""
import shutil
import subprocess
import sys
import time

NOTIFY_AFTER = 8.0      # only ping for finished turns that actually took a while
_state = {"on": True, "start": 0.0, "bell": False}   # bell OFF by default (no sound)


def _ping(summary="lean-coder", body=""):
    """Notify without making noise by default: a desktop toast where available
    (its sound, if any, is your OS setting - not ours). The audible terminal bell
    is opt-in (_state['bell'] / `/notify bell`). All best-effort, never raises."""
    if _state["bell"]:
        try:
            out = sys.__stdout__        # the real terminal, not a composer redirect
            if out is not None:
                out.write("\a")
                out.flush()
        except Exception:
            pass
    try:
        if shutil.which("notify-send"):
            subprocess.run(["notify-send", summary, body],
                           timeout=5, capture_output=True)
        elif shutil.which("osascript"):
            subprocess.run(["osascript", "-e",
                            f"display notification {body!r} with title {summary!r}"],
                           timeout=5, capture_output=True)
    except Exception:
        pass


def setup(lc, cfg):
    Agent = lc["Agent"]

    # 1) note when a turn starts, ping if it ran long enough by the time it ends
    _orig_run = Agent.run_turn
    def run_turn(self, user_input):
        _state["start"] = time.monotonic()
        return _orig_run(self, user_input)
    Agent.run_turn = run_turn

    _orig_end = Agent._end_of_turn
    def _end_of_turn(self):
        _orig_end(self)
        # Only ping when the turn ACTUALLY ends - i.e. control returns to the
        # operator. If the agent has an autostart continuation queued (post-handover
        # self-prompt), the "turn" immediately continues into the next one, so
        # pinging here spams a "turn finished" for every intermediate continuation.
        # Suppress until we're really handing back.
        if getattr(self, "_autostart_pending", None):
            return
        if _state["on"] and (time.monotonic() - _state["start"]) >= NOTIFY_AFTER:
            _ping("lean-coder", "turn finished")
    Agent._end_of_turn = _end_of_turn

    # 2) ping the moment the agent blocks on a confirmation (needs your input).
    #    Reassigning the module global means callers that look up _ask at call
    #    time (apply_diff, write_file, run_command, the composer) get the wrapper.
    _orig_ask = lc["_ask"]
    def _ask(question):
        if _state["on"]:
            _ping("lean-coder", "needs your input")
        return _orig_ask(question)
    lc["_ask"] = _ask

    # 3) runtime toggle: "/notify" on/off; "/notify bell" opts into the sound
    def _toggle(agent, cfg, arg):
        if arg.strip().lower() == "bell":
            _state["bell"] = not _state["bell"]
            print("notify bell on (audible)" if _state["bell"]
                  else "notify bell off (silent)")
        else:
            _state["on"] = not _state["on"]
            print("notify on" if _state["on"] else "notify off")
    lc["register_command"]("/notify", _toggle,
                           "toggle notifications; '/notify bell' toggles the audible bell (off by default)")
