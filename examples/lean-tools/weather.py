# Example lean-coder lean-tool: weather  (API-backed, network egress)
#
# Shows the shape of a tool that calls an external HTTP API with a key:
#   - stdlib-only request (urllib)
#   - key from an env var or a chmod-600 file under ~/.config/leancoder/ (NEVER
#     config.toml - it autosaves and would round-trip your key back to disk)
#   - NOT marked "safe": network egress goes through the confirm gate, so each
#     call shows its args and asks first (auto-approve waives that)
#
# Copy this into ~/.config/leancoder/lean-tools/, run /tools, and enable it.
# See LEAN_TOOLS.md for the full guide; brave_search.py is a real bundled example
# of this same pattern.

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

TOOL = {
    "name": "weather",
    "description": "Current weather for a city (name or 'lat,lon').",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    # no "safe": network egress -> confirmed before each call
}

_API = "https://api.example-weather.com/v1/current"   # replace with a real endpoint
_KEY_ENV = "WEATHER_API_KEY"
_KEY_FILE = Path.home() / ".config" / "leancoder" / "weather.key"
_TIMEOUT = 20


def _api_key():
    # env var wins (override / CI), else the chmod-600 key file. Never config.toml.
    k = os.environ.get(_KEY_ENV)
    if k:
        return k.strip()
    try:
        return _KEY_FILE.read_text().strip() or None
    except OSError:
        return None


def run(args, cwd):
    location = (args.get("location") or "").strip()
    if not location:
        return "error: weather needs a 'location'"
    key = _api_key()
    if not key:
        return (f"error: weather needs an API key - put it in {_KEY_FILE} "
                f"(or set {_KEY_ENV}).")
    url = _API + "?" + urlencode({"q": location, "key": key})
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            payload = json.loads(r.read(1_000_000).decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        hint = f" - check {_KEY_ENV}" if e.code in (401, 403) else ""
        return f"error: weather HTTP {e.code} {e.reason}{hint}"
    except Exception as e:
        return f"error: weather request failed: {e}"
    # shape depends on the real API; this is illustrative
    cur = payload.get("current") or {}
    return f"{location}: {cur.get('temp_c', '?')}°C, {cur.get('summary', 'n/a')}"
