"""Example startup-hook lean-tool: adds a /whoami slash command (no model tool).

Shows the two extension points beyond a normal tool lean-tool:
  - setup(lc, cfg): called once at startup on the driver, with the module
    globals (lc) and the live config. Use it to register commands, set config
    defaults, or patch internals.
  - register_command: surface a real slash command with a handler.

(The core itself ships a richer /info; this example uses /whoami so it doesn't
collide with that built-in - a lean-tool can never shadow a built-in name.)

Enable it like any lean-tool (run /tools), then type /whoami. This file defines
no TOOL/run, so it adds nothing to the model's tool surface - it is a
driver-side hook only and is never pushed to a remote executor.
"""


def _whoami(agent, cfg, arg):
    """Handler for /whoami. Signature is handler(agent, cfg, arg)."""
    where = f"remote {agent.remote.host}" if agent.remote else "local"
    turns = sum(1 for m in agent.messages if m.get("role") == "user")
    # Report the ACTIVE backend: when a provider is live, model/endpoint/window
    # come from it (cfg.active_model()/ctx_window()), not the native Ollama host.
    endpoint = cfg.provider if cfg.provider else cfg.host
    print(f"  model:    {cfg.active_model()}")
    print(f"  endpoint: {endpoint}  ({where})")
    print(f"  messages: {len(agent.messages)} ({turns} user turns)")
    print(f"  num_ctx:  {cfg.ctx_window():,}")


def setup(lc, cfg):
    # Register a slash command. Built-in names (e.g. /model, /info) are refused
    # automatically so a lean-tool can never shadow a core command.
    lc["register_command"]("/whoami", _whoami, "show session model/endpoint/size")
