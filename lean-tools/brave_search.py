"""brave_search - web search via the Brave Search API; returns the top results
(title, url, snippet) so the model can then web_fetch the ones worth reading.

Stdlib only: a GET to Brave's JSON API. Needs a key (free tier: 2,000
queries/month, no card, at search.brave.com/app/keys) - put it in the chmod-600
file ~/.config/leancoder/brave.key, or set the LEANCODER_BRAVE_KEY env var (which
overrides the file). NOT config.toml: /save rewrites that and would drop it.
Keyless scraping of the big engines is blocked, so a key is the reliable lean
path. Pairs with web_fetch: brave_search (find URLs) -> web_fetch (read one).

NETWORK EGRESS - not marked safe, so each search shows the query and confirms
(auto-approve waives).
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

TOOL = {
    "name": "brave_search",
    "description": "Web search: returns title/url/snippet for the top hits; web_fetch one to read it.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "description": "max results, default 8"},
            "freshness": {"type": "string", "description": "time filter: 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year), or a date range 'YYYY-MM-DDtoYYYY-MM-DD'."},
            "offset": {"type": "integer", "description": "result offset for paging (default 0). Use to get the next page of results."},
        },
        "required": ["query"],
    },
    # no "safe": egress goes through the confirm gate
    # driver_only: pure network egress with NO dependency on the connected workspace's
    # filesystem. It must run on the DRIVER, not the /connect remote: the API key lives
    # in the driver's ~/.config/leancoder/brave.key, and search traffic should egress
    # from the driver, never from the target box. Never pushed to the remote executor.
    "driver_only": True,
}

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
KEY_ENV = "LEANCODER_BRAVE_KEY"
KEY_FILE = Path.home() / ".config" / "leancoder" / "brave.key"
TIMEOUT = 20
UA = "lean-coder/brave_search"


def api_key():
    """The Brave key: env var wins (override / CI), else the chmod-600 key file in
    the config dir. NOT config.toml - /save rewrites that and would drop it."""
    k = os.environ.get(KEY_ENV)
    if k:
        return k.strip()
    try:
        return KEY_FILE.read_text().strip() or None
    except OSError:
        return None


def _clean(s):
    """Strip Brave's <strong> highlight tags (and any other) from a field."""
    return re.sub(r"<[^>]+>", "", s or "")


def format_results(payload, count):
    """Pure: Brave API JSON -> a compact, LLM-friendly result block."""
    results = (payload.get("web") or {}).get("results") or []
    if not results:
        return "(no results)"
    out = []
    for r in results[:count]:
        title = _clean(r.get("title")).strip() or "(untitled)"
        url = (r.get("url") or "").strip()
        desc = " ".join(_clean(r.get("description")).split())
        block = f"- {title}  {url}"
        if desc:
            block += f"\n  {desc}"
        out.append(block)
    return "\n".join(out)


def run(args, cwd):
    query = (args.get("query") or "").strip()
    if not query:
        return "error: brave_search needs a 'query'"
    try:
        count = max(1, min(int(args.get("count") or 8), 20))
    except (TypeError, ValueError):
        count = 8
    key = api_key()
    if not key:
        return (f"error: brave_search needs a Brave API key - put it in "
                f"{KEY_FILE} (or set {KEY_ENV}). Free tier at "
                f"search.brave.com/app/keys.")
    params = {"q": query, "count": count}
    freshness = (args.get("freshness") or "").strip()
    if freshness:
        params["freshness"] = freshness
    offset = args.get("offset")
    if offset:
        try:
            params["offset"] = max(0, int(offset))
        except (TypeError, ValueError):
            pass
    url = BRAVE_URL + "?" + urlencode(params)
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "X-Subscription-Token": key,
        "User-Agent": UA,
    })
    payload = None
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                payload = json.loads(r.read(2_000_000).decode("utf-8", "replace"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                last_err = e
                continue
            hint = f" - check {KEY_ENV}" if e.code in (401, 403, 422) else ""
            return f"error: Brave search HTTP {e.code} {e.reason}{hint}"
        except Exception as e:
            return f"error: Brave search failed: {e}"
    if payload is None:
        return f"error: Brave search rate-limited after retries (HTTP {last_err.code})"
    return f"[search] {query}\n{format_results(payload, count)}"
