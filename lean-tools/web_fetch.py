"""web_fetch - read a web page (or local dev server) as clean text.

  - read a page:   web_fetch(url=...)            HTML -> readable text, links kept
                                                 inline as [text](url); JSON pretty
  - just links:    web_fetch(url=..., links=true)   the page's links only
  - find in page:  web_fetch(url=..., find="...")    only sections matching a regex
  - page long doc: web_fetch(url=..., start=N)       char offset (reply says where)

A page's <main>/<article> is preferred when present (nav/footer dropped). For
discovery use the brave_search tool to find URLs, then web_fetch one.

NETWORK EGRESS - not marked safe, so every call shows the URL and confirms
(auto-approve waives). It does NOT block private/localhost addresses - hitting your own
dev servers (http://localhost:3000, a LAN box) is a primary use. If the model has
read untrusted content it could *propose* a fetch you didn't intend; the visible-
URL confirm is your check. GET only, size-capped, timed out. Off by default.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

TOOL = {
    "name": "web_fetch",
    "description": "Read a URL as readable text (HTML stripped, links kept inline); find/links/start tune the read.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "page to fetch; a bare domain gets https:// prepended"},
            "find": {"type": "string", "description": "return only sections matching this regex/substring"},
            "links": {"type": "boolean", "description": "return just the page's links"},
            "start": {"type": "integer", "description": "char offset into the text (page long docs)"},
        },
        "required": ["url"],
    },
    # no "safe": egress always goes through the confirm gate
    # driver_only: pure network fetch with NO dependency on the connected workspace's
    # filesystem. Runs on the DRIVER, not the /connect remote, so egress is always from
    # the driver (never the target box) and the core's LEANCODER_CTX_MAX/USED env (set
    # per-dispatch on the driver only) is present, so the read is context-sized correctly
    # instead of falling back to full MAX_TEXT on a remote executor. Never pushed remote.
    "driver_only": True,
}

MAX_BYTES = 400_000     # raw bytes read
MAX_TEXT = 80_000       # hard ceiling on chars returned per page (start= pages on)
MIN_TEXT = 2_000        # never return less than this, even on a tiny window
CHARS_PER_TOKEN = 3     # conservative chars/token for budgeting
CTX_FRACTION = 0.4      # share of the context window one read may use
LINK_CAP = 300
FIND_CTX = 2            # lines of context around a find match
TIMEOUT = 20
UA = "Mozilla/5.0 (X11; Linux x86_64) lean-coder/web_fetch"

# The core publishes the live context budget (tokens) before each lean-tool dispatch:
#   LEANCODER_CTX_MAX  - the active model's context window (num_ctx)
#   LEANCODER_CTX_USED - tokens already in use this turn
# free = MAX - USED. A weak model (small window) or a nearly-full big one won't add
# find=/start= itself, so the tool sizes the read to FREE or a full page evicts the
# model's own context (and it then hallucinates from memory). A fresh module is exec'd
# per load, so a setup()-set global wouldn't reach run() - the env vars are the carry.
# Any tool may read them. Unset on a remote executor / tests -> full MAX_TEXT.
CTX_MAX_ENV = "LEANCODER_CTX_MAX"
CTX_USED_ENV = "LEANCODER_CTX_USED"


def _env_int(name):
    try:
        return int(os.environ.get(name) or 0)
    except ValueError:
        return 0


def free_tokens():
    """Free context tokens (MAX - USED) the core published, or None if it didn't."""
    mx = _env_int(CTX_MAX_ENV)
    if not mx:
        return None
    return max(0, mx - _env_int(CTX_USED_ENV))


def char_budget():
    """Chars to return per page: CTX_FRACTION of the FREE context window, bounded
    MIN_TEXT..MAX_TEXT. Falls back to MAX_TEXT when the core published nothing
    (tests / remote executor). Big free window -> full chunk; little free -> safe slice."""
    free = free_tokens()
    if free is None:
        return MAX_TEXT
    return max(MIN_TEXT, min(int(free * CHARS_PER_TOKEN * CTX_FRACTION), MAX_TEXT))


def is_minimal():
    """True when free context is so low the budget hit the MIN_TEXT floor - the read
    may still strain the window, so run() warns the user to /compact."""
    free = free_tokens()
    return free is not None and int(free * CHARS_PER_TOKEN * CTX_FRACTION) < MIN_TEXT


class _Extract(HTMLParser):
    """HTML -> readable text + links. Headings get '#' prefixes, anchors become
    [text](url) inline, scripts/styles are dropped, and if any <main>/<article>
    is present only its content is kept (nav/footer boilerplate falls away)."""
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BREAK = {"p", "br", "div", "li", "tr", "section", "header", "footer",
              "ul", "ol", "table", "blockquote"}
    _H = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _MAIN = {"main", "article"}

    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.chunks = []          # (in_main, text)
        self.links = []           # (in_main, text, url)
        self.saw_main = False
        self._skip = 0
        self._main = 0
        self._a = None
        self._abuf = []

    def _push(self, text):
        self.chunks.append((self._main > 0, text))

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in self._MAIN:
            self._main += 1
            self.saw_main = True
        elif tag == "a":
            href = dict(attrs).get("href")
            self._a = urljoin(self.base, href) if href else None
            self._abuf = []
        elif tag in self._H:
            self._push("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "br":
            self._push("\n")
        elif tag in self._BREAK:
            self._push("\n")

    def handle_startendtag(self, tag, attrs):
        if not self._skip and tag == "br":
            self._push("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if tag in self._MAIN and self._main:
            self._main -= 1
        elif tag == "a":
            txt = " ".join("".join(self._abuf).split())
            if self._a:
                self.links.append((self._main > 0, txt or self._a, self._a))
                self._push(f"[{txt}]({self._a})" if txt else f"({self._a})")
            elif txt:
                self._push(txt)
            self._a, self._abuf = None, []
        elif tag in self._H:
            self._push("\n")
        elif tag in self._BREAK:
            self._push("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._a is not None:
            self._abuf.append(data)
        elif data.strip():
            self._push(data)


def render(html, base):
    """Pure: HTML -> (readable text, [(text, url)] links). Prefers <main>/<article>
    content when present. On parse error, returns (html, [])."""
    p = _Extract(base)
    try:
        p.feed(html)
        p.close()
    except Exception:
        return html, []
    keep_main = p.saw_main
    raw = "".join(t for (m, t) in p.chunks if m or not keep_main)
    lines, out, blank = [ln.strip() for ln in raw.splitlines()], [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    text = "\n".join(out).strip()
    seen, links = set(), []
    for m, t, u in p.links:
        if (m or not keep_main) and u not in seen:
            seen.add(u)
            links.append((t, u))
    return text, links


def format_links(links):
    if not links:
        return "(no links found)"
    return "\n".join(f"- {t}  {u}" for t, u in links[:LINK_CAP])


def apply_find(text, pattern):
    """Return only the lines matching pattern (regex, else literal) + context."""
    try:
        rx = re.compile(pattern, re.I)
    except re.error:
        rx = re.compile(re.escape(pattern), re.I)
    lines = text.splitlines()
    keep = set()
    for i, ln in enumerate(lines):
        if rx.search(ln):
            keep.update(range(max(0, i - FIND_CTX), min(len(lines), i + FIND_CTX + 1)))
    if not keep:
        return f"(no sections match find={pattern!r})"
    out, prev = [], -2
    for i in sorted(keep):
        if i > prev + 1:
            out.append("...")
        out.append(lines[i])
        prev = i
    return "\n".join(out)


def window(text, start, limit=MAX_TEXT):
    """Slice up to `limit` chars from `start`; report total + whether more remains."""
    total = len(text)
    start = max(0, min(start, total))
    chunk = text[start:start + limit]
    return chunk, total, start + len(chunk) < total, start


def normalize_url(url):
    """A real http(s) URL passes through; a bare domain gets https:// prepended;
    anything else (a phrase, a non-http scheme) returns None (not fetchable - use
    brave_search for queries)."""
    url = (url or "").strip()
    if not url:
        return None
    if urlparse(url).scheme in ("http", "https"):
        return url
    if ("://" not in url and " " not in url and "." in url
            and not url.startswith(("/", "?"))):
        return "https://" + url
    return None


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read(MAX_BYTES + 1), r.headers.get("Content-Type", "?"), r.geturl()


def run(args, cwd):
    url = normalize_url(args.get("url"))
    if not url:
        return ("error: web_fetch needs a valid http(s) url "
                "(use brave_search to find one).")
    try:
        data, ctype, final = _fetch(url)
    except urllib.error.HTTPError as e:
        return f"error: HTTP {e.code} {e.reason}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"error fetching {url}: {e}"
    body = data[:MAX_BYTES].decode("utf-8", "replace")
    ct = ctype.lower()
    links = []
    if "html" in ct:
        text, links = render(body, final)
    elif "json" in ct:
        try:
            text = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
        except Exception:
            text = body
    else:
        text = body

    where = url if final == url else f"{url} -> {final}"
    budget = char_budget()
    if is_minimal():
        # stderr, not stdout: safe even on the remote executor (stdout is the protocol
        # channel there); lands above the input/ctx bar locally.
        sys.stderr.write(f"[web_fetch] low free context (~{free_tokens()} tok) - "
                         f"minimal read; /compact for fuller results.\n")
    if args.get("links"):
        return f"[{ctype}] links from {where}\n{format_links(links)}"
    if args.get("find"):
        found = apply_find(text, str(args['find']))
        if len(found) > budget:
            found = found[:budget] + f"\n... (truncated to {budget} chars; narrow find= or use start=)"
        return f"[{ctype}] from {where}  (find={args['find']!r})\n{found}"
    try:
        start = int(args.get("start") or 0)
    except (TypeError, ValueError):
        start = 0
    chunk, total, more, start = window(text, start, budget)
    if more or start:
        head = (f"[{ctype}] from {where} ({total} chars, showing {start}-{start + len(chunk)}"
                + (f"; more at start={start + len(chunk)})" if more else ")"))
    else:
        head = f"[{ctype}] from {where} ({total} chars)"
    return f"{head}\n{chunk}"
