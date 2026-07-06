"""Example lean-coder provider plugin.

A provider is the adapter that connects lean-coder to a model backend other than
the bundled Ollama - it takes the conversation, sends it to the backend's API, and
returns the reply. Drop a file like this in your providers dir (`~/.config/leancoder/providers/`
or the repo's `providers/`), enable it with `/provider enable example`, then switch
to it with `/provider example`. See PROVIDER_API.md for the full contract.

This template targets a generic OpenAI-compatible chat API (the most common shape):
`POST {base}/chat/completions` with a Bearer key. To adapt it to a real backend you
usually only change:
  - _API_BASE / _PROVIDER / the model list, and
  - the credential env var name.
The message/tool format below is already OpenAI-compatible, so for most hosted APIs
it works unchanged.

Auth: set EXAMPLE_API_KEY in the environment, or write
`~/.config/leancoder/example.json` -> {"key": "...", "active": true}. Nothing here
carries a real key; keys are resolved at runtime, never stored in config.toml.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_PROVIDER    = "example"
_API_BASE    = "https://api.example.com/v1"   # replace with your backend's base URL
_CTX_DEFAULT = 128_000
_OUT_CAP     = 4096

# Populated by setup() from the host's config dir; falls back to the default path.
_CONFIG_DIR = None
_lc = {}


# ----------------------------------------------------------------------------
# Credentials: env var first, then a keyfile under the config dir. Never in
# config.toml (it autosaves and would round-trip your key back to disk in plain
# text alongside non-secret settings).
# ----------------------------------------------------------------------------
def _cred_path():
    return (_CONFIG_DIR or Path.home() / ".config" / "leancoder") / "example.json"


def _api_key():
    v = os.environ.get("EXAMPLE_API_KEY", "")
    if v:
        return v
    try:
        return json.loads(_cred_path().read_text()).get("key", "")
    except (OSError, ValueError):
        return ""


def _cred_data():
    try:
        return json.loads(_cred_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_cred(data):
    p = _cred_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    try:
        p.chmod(0o600)          # secrets are owner-only, never world-readable
    except OSError:
        pass


def _clear_cred():
    try:
        _cred_path().unlink()
    except OSError:
        pass


def _is_active():
    """True when we have a key AND the user hasn't parked the provider inactive."""
    if not _api_key():
        return False
    try:
        return json.loads(_cred_path().read_text()).get("active", True)
    except (OSError, ValueError):
        return True         # a bare env-var key with no file counts as active


# ----------------------------------------------------------------------------
# Onboarding hooks: login + clear. Implementing "login" is what makes the OBE
# first-class - core wires it to `/provider login example`, AND, when a chat call
# fails because the key is missing/rejected, core offers this same flow inline and
# retries. So the strings your chat() raises on an auth failure matter: raise a
# RuntimeError whose text contains "no API key" / "401" / "403" / "unauthorized"
# so core recognizes it as an auth error and prompts to log in (see chat() below).
# Return truthy on success; False to signal the user cancelled.
# ----------------------------------------------------------------------------
def _login(agent, cfg):
    key = _api_key()
    if not key:
        # No spinner, no echo: read the secret with getpass so it never hits the screen
        # or the scrollback. A key already in the env / keyfile skips straight to active.
        import getpass
        _lc.get("dim") and print(_lc["dim"]("Paste your API key and press Enter:"))
        try:
            key = getpass.getpass(prompt="").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if not key:
            return False
        _save_cred({"key": key, "active": True})
    else:
        d = _cred_data()
        d["active"] = True
        if d.get("key"):        # only persist if the key lives in the file (not a bare env key)
            _save_cred(d)
    return True


def _clear(agent, cfg):
    """/provider clear example - wipe the stored key (core drops back to Ollama if active)."""
    _clear_cred()


# ----------------------------------------------------------------------------
# The client. Duck-types OllamaClient: the agent loop only needs chat().
# ----------------------------------------------------------------------------
class ExampleClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_out_tokens = None      # core reads this after each call (optional)

    def chat(self, messages, tools, should_abort=None):
        """Return (assistant_message_dict, prompt_eval_tokens_or_None, aborted_bool).

        messages/tools are already in OpenAI shape. For a non-OpenAI backend,
        convert here and convert the response back into an assistant dict with
        `content` and optional `tool_calls`.
        """
        key = _api_key()
        if not key:
            # Raise (don't return a content string) with "no API key" in the text: core
            # recognizes an auth error and offers `/provider login` inline, then retries
            # this turn. Returning a plain assistant message would look like a normal
            # reply and the user would be stuck. See the "login" hook + PROVIDER_API.md.
            raise RuntimeError("no API key - use /provider login (or set EXAMPLE_API_KEY)")

        body = {
            "model":      self.cfg.active_model(),
            "messages":   messages,
            "max_tokens": _OUT_CAP,
        }
        if tools:
            body["tools"] = tools

        req = urllib.request.Request(
            f"{_API_BASE}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            # 401/403 = bad/expired key: raise so core offers login + retries this turn.
            if e.code in (401, 403):
                raise RuntimeError(f"API {e.code} {e.reason} - key rejected; /provider login\n{detail}")
            return {"role": "assistant",
                    "content": f"example provider: HTTP {e.code} {e.reason}\n{detail}"}, None, False
        except (urllib.error.URLError, OSError, ValueError) as e:
            return {"role": "assistant", "content": f"example provider: request failed: {e}"}, None, False

        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {"role": "assistant", "content": ""}
        usage = payload.get("usage") or {}
        self.last_out_tokens = usage.get("completion_tokens")
        return msg, usage.get("prompt_tokens"), False


# ----------------------------------------------------------------------------
# The PROVIDER dict: what the ProviderManager reads + registers. Only make_client,
# list_models and context_window are required; everything else is optional and
# absent-safe (see PROVIDER_API.md).
# ----------------------------------------------------------------------------
PROVIDER = {
    "name":           _PROVIDER,
    "description":    "Example OpenAI-compatible provider (template - edit before use)",
    "make_client":    lambda cfg: ExampleClient(cfg),
    "list_models":    lambda: ["example-small", "example-large"],   # cached/static; never throws
    "context_window": lambda model: _CTX_DEFAULT,
    "available":      _is_active,          # provider offered/auto-startable only when keyed
    "autostart":      _is_active,          # activate on launch when a key is present + active
    "login":          _login,              # enables `/provider login example` + inline auth-fail retry
    "clear":          _clear,              # enables `/provider clear example` (wipe the key)
    "tag":            "ex",                # short label in the /model row
    "capabilities":   lambda model: {},    # declare {"thinking": [...], "effort": [...]} if supported
}


def setup(lc, cfg):
    """OPTIONAL, helpers-only. The manager calls this then registers PROVIDER itself -
    do NOT call register_provider here. Grab the config dir so secrets resolve to the
    right place; you could also register bespoke slash commands (e.g. a /example_login)."""
    global _CONFIG_DIR
    _lc.update(lc)
    _CONFIG_DIR = lc.get("CONFIG_DIR")
