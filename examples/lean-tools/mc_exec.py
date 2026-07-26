# lean-coder lean-tool: mc_exec  (drive a Mineflayer bot in Minecraft)
#
# The brain->game bridge for the LLM bot swarm. Each bot is a Node/Mineflayer
# process exposing a tiny HTTP endpoint (see mc-bots/bot.js):
#   GET  /state       -> current world state
#   POST /exec  body=JS async snippet ('bot','goals','Movements' in scope)
#                     -> {result, ready, position, health, food, inventory,
#                         nearbyBlocks, ...}
#
# This tool POSTs a JS snippet to a bot's /exec (or GETs /state when no code),
# and returns the JSON result compactly so the model sees what happened and the
# resulting world state. One bot per HTTP port; address a bot by host:port.
#
# Copy into ~/.config/leancoder/lean-tools/, run /tools, enable it.

import json
import urllib.error
import urllib.request

TOOL = {
    "name": "mc_exec",
    "description": (
        "Drive a Minecraft bot via its Mineflayer bridge. Runs a JS snippet in "
        "the bot's context (async; 'bot', 'goals', 'Movements' in scope; use "
        "'return X' to report a value) and returns {result, ...world state}. "
        "Omit 'code' to just read current state. Example code: "
        "\"const p=bot.entity.position; await bot.pathfinder.goto(new "
        "goals.GoalNear(p.x+10,p.y,p.z,1)); return bot.entity.position\"."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "JS async snippet body. Omit to just GET /state.",
            },
            "host": {"type": "string", "description": "Bot bridge host (default 127.0.0.1)."},
            "port": {"type": "integer", "description": "Bot bridge HTTP port (default 3000)."},
            "timeout": {"type": "integer", "description": "Seconds to await (default 60)."},
        },
        "required": [],
    },
    # not "safe": it acts in the game world and hits the network -> confirm gate
}

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 3000
_DEFAULT_TIMEOUT = 60


def _fmt(payload):
    """Compact, model-friendly rendering of the bridge's JSON reply."""
    if not isinstance(payload, dict):
        return str(payload)
    lines = []
    if "error" in payload and payload["error"]:
        lines.append(f"error: {payload['error']}")
    if "result" in payload:
        lines.append(f"result: {json.dumps(payload['result'], default=str)}")
    if not payload.get("ready", True):
        lines.append("bot: NOT READY (connecting/reconnecting)")
    pos = payload.get("position")
    if pos:
        try:
            lines.append(f"pos: ({pos['x']:.1f},{pos['y']:.1f},{pos['z']:.1f})"
                         f" {payload.get('dimension','?')}")
        except (KeyError, TypeError, ValueError):
            lines.append(f"pos: {pos}")
    if "health" in payload:
        lines.append(f"health: {payload.get('health')}  food: {payload.get('food')}")
    inv = payload.get("inventory")
    if inv is not None:
        items = ", ".join(f"{i['name']}x{i['count']}" for i in inv) or "(empty)"
        lines.append(f"inventory: {items}")
    nb = payload.get("nearbyBlocks")
    if nb:
        top = sorted(nb.items(), key=lambda kv: -kv[1])[:12]
        lines.append("nearby: " + ", ".join(f"{n}:{c}" for n, c in top))
    return "\n".join(lines) if lines else json.dumps(payload, default=str)


def run(args, cwd):
    host = (args.get("host") or _DEFAULT_HOST).strip()
    port = int(args.get("port") or _DEFAULT_PORT)
    timeout = int(args.get("timeout") or _DEFAULT_TIMEOUT)
    code = args.get("code")

    base = f"http://{host}:{port}"
    if code is None or str(code).strip() == "":
        url, data, method = base + "/state", None, "GET"
    else:
        url, data, method = base + "/exec", str(code).encode("utf-8"), "POST"

    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "text/plain")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(2_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(200_000).decode("utf-8", "replace")
        except Exception:
            pass
        try:
            return _fmt(json.loads(body))
        except Exception:
            return f"error: mc_exec HTTP {e.code} {e.reason}: {body[:500]}"
    except urllib.error.URLError as e:
        return (f"error: mc_exec cannot reach bot at {base} ({e.reason}). "
                f"Is bot.js running on that port?")
    except Exception as e:
        return f"error: mc_exec request failed: {e}"

    try:
        return _fmt(json.loads(raw))
    except Exception:
        return raw[:2000]
