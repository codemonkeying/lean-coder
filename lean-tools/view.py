"""view - see a web page, a local file or the screen as an image.

  target forms:
    web:<url>  (or a bare url/domain)  render a page; returns image + visible text
    file:<path>                        png/jpg/gif/webp (bmp with PIL) returned directly;
                                       html/htm/svg/pdf rendered in a browser
    screen:                            native screenshot of the desktop

Web renderer chain: approved Playwright engines if importable (full features:
devices, breakpoints, dark, wait_for, a11y dump; bounded by PW_DEADLINE) -> if none
launches, or Playwright is absent, an approved system browser found on the box
(BROWSERS order) in basic headless mode -> else a clear install hint.

Images go to a per-process temp dir (lc-view-*) removed at exit. Providers read
image_path lazily on every request, so the file must live until then.

NETWORK EGRESS - not marked safe; every call shows the target and confirms
(auto-approve waives).
"""
import atexit
import itertools
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


def _dbg_swallow():
    """Debug bucket shim: hands a swallowed exception to lean-coder's _dbg_swallow
    (a no-op unless LEANCODER_DEBUG=1). Never raises."""
    for _m in ("__main__", "lean_coder"):
        _f = getattr(sys.modules.get(_m), "_dbg_swallow", None)
        if _f and _f is not _dbg_swallow:
            try:
                _f(2)
            except Exception:
                pass
            return


TOOL = {
    "name": "view",
    "description": "View a web page, file or the screen as an image.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "web:<url> (bare url ok), file:<path>, or screen:",
            },
            "full_page": {"type": "boolean", "default": False,
                          "description": "Whole scrollable page (Playwright)."},
            "width": {"type": "integer", "default": 1280, "description": "Viewport width."},
            "height": {"type": "integer", "default": 800, "description": "Viewport height."},
            "mobile": {"type": "boolean", "default": False,
                       "description": "Mobile emulation (Playwright)."},
            "device": {"type": "string", "description": "Device name, e.g. 'iPhone 14' (Playwright)."},
            "dark": {"type": "boolean", "default": False, "description": "Dark color scheme (Playwright)."},
            "breakpoints": {"type": "array", "items": {"type": "integer"},
                            "description": "Widths, one shot each (Playwright)."},
            "wait_for": {"type": "string", "description": "CSS selector to await (Playwright)."},
            "dom": {"type": "boolean", "default": False,
                    "description": "Add accessibility tree (Playwright)."},
        },
        "required": ["target"],
    },
    # no "safe": egress goes through the confirm gate.
    # Runs wherever the session runs (local or a connected remote). A remote image
    # is fetched back to the driver by core (any lean-tool returning image_path).
    # tier read: it only reads, so it rides at /leash r; still confirms each call.
    "tier": "read",
    "no_timeout": True,   # browser launch + render is long-by-design (has its own TIMEOUT)
}

TIMEOUT = 30_000   # ms for playwright operations
PROC_TIMEOUT = 30  # s for a system-browser / capture subprocess (incl. kill wait)
KILL_WAIT = 2      # s reaping a killed browser; counted inside PROC_TIMEOUT
MAX_TEXT = 6_000   # chars of visible text returned
MAX_A11Y = 10_000  # chars of accessibility tree if dom=true
MAX_AGGREGATE = 20_000  # total text cap across all breakpoints

PW_ONLY = ("full_page", "mobile", "device", "dark", "breakpoints", "wait_for", "dom")
FILE_IMAGE_MAX = 20_000_000  # bytes; larger file: images are refused
IMAGE_EXTS = ("png", "jpg", "jpeg", "gif", "webp")   # + bmp only when PIL is importable
RENDER_EXTS = ("html", "htm", "svg", "pdf")
# Renderers TESTED per OS (Playwright + system browsers). Only these are used;
# anything else installed is ignored. Add a name only after it passes a live test.
# Entries are per renderer: "playwright:<engine>" or a system browser name (BROWSERS).
# Playwright engines are tried in the order listed here.
APPROVED = {
    "linux":  ("playwright:chromium", "playwright:firefox", "chromium", "firefox"),
    "win":    ("playwright:chromium", "playwright:firefox", "chrome", "firefox"),
    "darwin": ("playwright:chromium", "playwright:firefox", "firefox"),
}
_LABEL = {"chromium": "Chromium", "chrome": "Chrome", "edge": "Edge",
          "firefox": "Firefox", "brave": "Brave", "webkit": "WebKit"}
PW_DEADLINE = 60   # s: hard cap on one whole Playwright run (launch + all captures)

# System-browser detection data, in preference order. Reorder here only.
# (name, family, linux PATH names, macOS app/binary, windows relative paths)
BROWSERS = [
    ("chromium", "chromium", ("chromium", "chromium-browser"),
     ("Chromium", "Chromium"), (r"Chromium\Application\chrome.exe",)),
    ("chrome", "chromium", ("google-chrome", "google-chrome-stable", "chrome"),
     ("Google Chrome", "Google Chrome"), (r"Google\Chrome\Application\chrome.exe",)),
    ("edge", "chromium", ("microsoft-edge", "msedge"),
     ("Microsoft Edge", "Microsoft Edge"), (r"Microsoft\Edge\Application\msedge.exe",)),
    ("firefox", "firefox", ("firefox",),
     ("Firefox", "firefox"), (r"Mozilla Firefox\firefox.exe",)),
    # brave: hangs headless (no PNG) on Linux and Windows tests; kept for detection only
    ("brave", "chromium", ("brave-browser", "brave"),
     ("Brave Browser", "Brave Browser"), (r"BraveSoftware\Brave-Browser\Application\brave.exe",)),
]


def _os_key(plat=None):
    plat = plat or _platform()
    return "darwin" if plat == "darwin" else "win" if plat.startswith("win") else "linux"


def _approved():
    return APPROVED.get(_os_key(), ())


def _pw_approved():
    """Approved Playwright engines for this OS, in try order."""
    return [n.split(":", 1)[1] for n in _approved() if n.startswith("playwright:")]


def no_browser_msg():
    pw = _pw_approved()
    names = [f"Playwright ({'/'.join(_LABEL.get(e, e) for e in pw)}: pip install playwright && "
              f"python -m playwright install {pw[0]})"] if pw else []
    names += [_LABEL.get(n, n) for n in _approved() if ":" not in n]
    files = "Image files (file:) still work."
    if not names:
        return ("error: no tested browser on this OS yet, so web pages and html/svg/pdf "
                f"files can't be rendered. {files}")
    hint = " or ".join(names)
    return ("error: no tested browser here, so web pages and html/svg/pdf files can't be "
            f"rendered. Install {hint}. {files}")


_OUT = {"dir": None}
_SEQ = itertools.count()


def _have_pil():
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _image_exts():
    return IMAGE_EXTS + (("bmp",) if _have_pil() else ())


def _out_dir():
    """Per-process temp dir for images (shared with core when loaded in it), removed
    at exit. Never ~/screenshots."""
    for _m in ("__main__", "lean_coder"):
        _f = getattr(sys.modules.get(_m), "_lc_view_dir", None)
        if callable(_f):
            try:
                return Path(_f())
            except Exception:
                _dbg_swallow()
    if _OUT["dir"] is None:
        d = tempfile.mkdtemp(prefix=f"lc-view-{os.getpid()}-")
        atexit.register(shutil.rmtree, d, True)
        _OUT["dir"] = d
    return Path(_OUT["dir"])


def _normalise_url(raw):
    raw = raw.strip()
    if raw and "://" not in raw:
        raw = "https://" + raw
    return raw


def _slug(url):
    """A filesystem-safe slug from a URL for naming the screenshot."""
    p = urlparse(url)
    host = re.sub(r"[^\w.-]", "_", p.netloc or "page")
    path = re.sub(r"[^\w-]", "_", p.path.strip("/"))[:40]
    # microseconds + pid + a process counter: names never collide
    ts = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}_{next(_SEQ)}"
    name = f"{host}_{path}_{ts}" if path else f"{host}_{ts}"
    return re.sub(r"_+", "_", name).strip("_")


def parse_target(raw):
    """-> (kind, value): kind in web|file|screen. Bare url/domain = web."""
    t = (raw or "").strip()
    low = t.lower()
    if low.startswith("screen:") or low == "screen":
        return "screen", ""
    if low.startswith("file:") and not low.startswith("file://"):
        return "file", t[5:].strip()
    if low.startswith("file://"):
        return "file", t[7:]
    if low.startswith("web:"):
        return "web", _normalise_url(t[4:])
    return "web", _normalise_url(t)

def _visible_text(page, max_chars):
    """Extract visible body text via JS - clean readable snapshot."""
    script = """
    () => {
        function walk(node) {
            if (!node) return '';
            const skip = ['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','HEAD',
                          'IFRAME','svg','path','symbol'];
            if (skip.includes(node.nodeName)) return '';
            if (node.nodeType === 3) {
                const t = node.textContent || '';
                return t.trim() ? t.trim() + ' ' : '';
            }
            let out = '';
            const block = ['DIV','P','H1','H2','H3','H4','H5','H6','LI',
                           'TD','TH','HEADER','FOOTER','SECTION','ARTICLE',
                           'ASIDE','MAIN','NAV','BLOCKQUOTE','PRE','FIGCAPTION'];
            if (block.includes(node.nodeName)) out += '\\n';
            for (const child of node.childNodes) out += walk(child);
            if (block.includes(node.nodeName)) out += '\\n';
            return out;
        }
        const raw = walk(document.body);
        return raw.replace(/[ \\t]+/g,' ').replace(/\\n{3,}/g,'\\n\\n').trim();
    }
    """
    try:
        text = page.evaluate(script)
        return (text or "")[:max_chars]
    except Exception as e:
        return f"(could not extract text: {e})"


def _a11y_snapshot(page, max_chars):
    """Accessibility tree snapshot - structural + semantic, good for debugging."""
    script = """
    () => {
        function node(el, depth) {
            if (!el) return '';
            const role = el.getAttribute ? (el.getAttribute('role') || el.tagName.toLowerCase()) : '';
            const label = el.getAttribute ? (
                el.getAttribute('aria-label') ||
                el.getAttribute('alt') ||
                el.getAttribute('placeholder') ||
                (el.textContent || '').trim().slice(0, 60)
            ) : '';
            const indent = '  '.repeat(depth);
            let out = `${indent}[${role}] ${label}\\n`;
            for (const child of (el.children || [])) {
                if (depth < 6) out += node(child, depth + 1);
            }
            return out;
        }
        return node(document.body, 0);
    }
    """
    try:
        text = page.evaluate(script)
        return (text or "")[:max_chars]
    except Exception as e:
        return f"(could not extract a11y tree: {e})"


ENGINES = ("chromium", "firefox", "webkit")

# Generic mobile defaults (when mobile=true but no device specified).
_MOBILE_DEFAULTS = {
    "viewport": {"width": 390, "height": 844},
    "user_agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
        "Mobile/15E148 Safari/604.1"
    ),
    "is_mobile": True,
    "has_touch": True,
    "device_scale_factor": 2,
}


def _launch(p, engine):
    """Launch one engine. chromium takes hardening flags; others don't."""
    launcher = getattr(p, engine)
    if engine == "chromium":
        args = ["--disable-dev-shm-usage"]
        if _need_no_sandbox():
            args.append("--no-sandbox")
        return launcher.launch(args=args)
    return launcher.launch()


def _page_context_args(p, args):
    """Build the kwargs dict for browser.new_context() from tool args.
    Handles device, mobile, dark, and explicit width/height."""
    ctx = {}

    device_name = (args.get("device") or "").strip()
    if device_name:
        # Playwright ships a device registry; look up by name.
        devices = p.devices
        if device_name not in devices:
            # Try case-insensitive match.
            lower_map = {k.lower(): k for k in devices}
            match = lower_map.get(device_name.lower())
            if match:
                device_name = match
            else:
                return None, f"error: unknown device {device_name!r}. Examples: 'iPhone 14', 'Pixel 7', 'iPad Mini'."
        ctx.update(devices[device_name])
    elif args.get("mobile"):
        ctx.update(_MOBILE_DEFAULTS)

    # Explicit width/height override device/mobile viewport.
    w = args.get("width")
    h = args.get("height")
    if w or h:
        vp = dict(ctx.get("viewport") or {"width": 1280, "height": 800})
        if w: vp["width"] = int(w)
        if h: vp["height"] = int(h)
        ctx["viewport"] = vp
    elif "viewport" not in ctx:
        ctx["viewport"] = {"width": 1280, "height": 800}

    if args.get("dark"):
        ctx["color_scheme"] = "dark"

    return ctx, None


def _capture_one(browser, ctx_args, url, full_page, wait_for, want_dom, out_path, timeout):
    """Capture a single screenshot with the given context args. Returns (result_dict, error_str)."""
    context = browser.new_context(**ctx_args)
    try:
        return _capture_in(context, ctx_args, url, full_page, wait_for, want_dom, out_path, timeout)
    finally:
        try:
            context.close()     # also on failure (finding 7)
        except Exception:
            _dbg_swallow()


def _capture_in(context, ctx_args, url, full_page, wait_for, want_dom, out_path, timeout):
    from playwright.sync_api import TimeoutError as PWTimeout

    page = context.new_page()
    page.set_default_timeout(timeout)

    try:
        page.goto(url, wait_until="networkidle", timeout=timeout)
    except PWTimeout:
        try:
            page.goto(url, wait_until="load", timeout=timeout)
        except PWTimeout:
            _dbg_swallow()

    wait_missed = False
    if wait_for:
        try:
            page.wait_for_selector(wait_for, timeout=5_000)
        except PWTimeout:
            wait_missed = True

    title = page.title() or "(no title)"
    final_url = page.url
    vp = ctx_args.get("viewport", {})

    page.screenshot(path=str(out_path), full_page=full_page)

    text = _visible_text(page, MAX_TEXT)
    a11y = _a11y_snapshot(page, MAX_A11Y) if want_dom else None

    return {
        "title": title, "final_url": final_url,
        "text": text, "a11y": a11y, "out_path": out_path,
        "viewport": vp, "full_page": full_page, "wait_missed": wait_missed,
    }, None



class _NoEngine(Exception):
    """Playwright is importable but no approved engine launched (-> system fallback)."""


class _PWDriver:
    """Popen-like handle on the Playwright driver (node) so core's signal cleanup
    (_kill_view_browser -> _kill_tree) and our deadline can kill it. The driver is
    not its own process group, so _kill_tree's killpg(pid) is a harmless ESRCH and
    kill() below does the work; on Windows taskkill /T takes the whole tree. On
    POSIX the browsers Playwright launched talk to the driver over a pipe and exit
    when it dies."""
    def __init__(self, aproc):
        self._p, self.pid = aproc, aproc.pid

    def poll(self):
        return self._p.returncode

    def kill(self):
        try:
            os.kill(self.pid, 9) if os.name != "nt" else self._p.kill()
        except Exception:
            _dbg_swallow()


def _pw_driver(p):
    """The driver subprocess behind a sync_playwright() handle (private API; None
    if the layout changes - then only the thread deadline applies)."""
    try:
        aproc = p._impl_obj._connection._transport._proc
        return _PWDriver(aproc) if aproc is not None and aproc.pid else None
    except Exception:
        return None


def _run_playwright(url, args, order):
    """Run the Playwright render in a daemon thread bounded by PW_DEADLINE. On
    expiry, kill the driver (tracked in _CUR_PROC while running) and return an
    error. Raises _NoEngine when no engine in `order` launched."""
    global _CUR_PROC
    box = {}

    def _work():
        try:
            box["r"] = _run_playwright_inner(url, args, order)
        except BaseException as e:   # noqa: BLE001 - handed to the caller
            box["e"] = e

    t = threading.Thread(target=_work, name="lc-view-playwright", daemon=True)
    t.start()
    t.join(PW_DEADLINE)
    if t.is_alive():
        drv = _CUR_PROC
        if drv is not None:
            _kill_tree(drv)
        t.join(KILL_WAIT)
        _CUR_PROC = None
        return f"error: Playwright run exceeded {PW_DEADLINE}s; browser stopped"
    _CUR_PROC = None
    if "e" in box:
        raise box["e"]
    return box["r"]


def _run_playwright_inner(url, args, order):
    from playwright.sync_api import sync_playwright
    global _CUR_PROC


    wait_for = args.get("wait_for") or None
    want_dom = bool(args.get("dom", False))
    breakpoints = args.get("breakpoints") or None

    # full_page: default false normally, default true for breakpoints.
    if "full_page" in args:
        full_page = bool(args["full_page"])
    else:
        full_page = bool(breakpoints)

    out_dir = _out_dir()
    slug = _slug(url)

    try:
        with sync_playwright() as p:
            _CUR_PROC = _pw_driver(p)
            # Validate context args before launching browser.
            ctx_args, err = _page_context_args(p, args)
            if err:
                return err

            browser = None
            engine_used = None
            launch_errors = []
            for engine in order:
                try:
                    browser = _launch(p, engine)
                    engine_used = engine
                    break
                except Exception as e:
                    launch_errors.append(f"  {engine}: {e}")
            if browser is None:
                detail = "\n".join(launch_errors)
                blob = detail.lower()
                if "missing dependencies" in blob or "missing librar" in blob:
                    hint = (
                        "The host is missing system libraries the browser needs. Install them with:\n"
                        "  python3 -m playwright install-deps    # may need sudo\n"
                        "then install a browser binary:\n"
                        "  python3 -m playwright install firefox   # or chromium / webkit"
                    )
                else:
                    hint = (
                        "A browser binary may not be installed. Install one with:\n"
                        "  python3 -m playwright install firefox   # or chromium / webkit"
                    )
                raise _NoEngine(
                    "error: could not launch any browser engine.\n"
                    f"{hint}\n"
                    f"launch attempts:\n{detail}"
                )

            try:
                captures = []

                if breakpoints:
                    # Multi-breakpoint: one screenshot per width.
                    for bp_width in breakpoints:
                        bp_args = dict(ctx_args)
                        vp = dict(bp_args.get("viewport") or {"width": 1280, "height": 800})
                        vp["width"] = int(bp_width)
                        bp_args["viewport"] = vp
                        bp_path = out_dir / f"{slug}_{bp_width}w.png"
                        result, cap_err = _capture_one(
                            browser, bp_args, url, full_page, wait_for, want_dom, bp_path, TIMEOUT)
                        if cap_err:
                            return cap_err
                        captures.append(result)
                else:
                    # Single screenshot.
                    out_path = out_dir / f"{slug}.png"
                    result, cap_err = _capture_one(
                        browser, ctx_args, url, full_page, wait_for, want_dom, out_path, TIMEOUT)
                    if cap_err:
                        return cap_err
                    captures.append(result)
            finally:
                browser.close()
    except _NoEngine:
        raise
    except Exception as e:
        return f"error running headless browser: {e}"

    # Build result text.
    all_lines = []
    image_path = None  # for single-shot, return the image; for multi, return the first

    for i, cap in enumerate(captures):
        vp = cap["viewport"]
        w, h = vp.get("width", "?"), vp.get("height", "?")
        if len(captures) > 1:
            all_lines.append(f"=== breakpoint {i+1}/{len(captures)}: {w}px ===")
        all_lines.append("image attached" if i == 0 else "screenshot captured (not attached)")
        all_lines.append(f"title: {cap['title']}")
        all_lines.append(f"url: {cap['final_url']}")
        vp_note = f"viewport: {w}x{h}"
        if cap["full_page"]:
            vp_note += " (full page)"
        if ctx_args.get("is_mobile") or (args.get("device") or args.get("mobile")):
            vp_note += " (mobile)"
        if args.get("dark"):
            vp_note += " (dark)"
        all_lines.append(vp_note)
        all_lines.append(f"engine: {engine_used}")
        if args.get("device"):
            all_lines.append(f"device: {args['device']}")
        if wait_for:
            all_lines.append(f"wait_for: {wait_for!r}" + (
                " (not found within 5s; captured anyway)" if cap.get("wait_missed") else ""))

        if cap["text"]:
            all_lines.append("")
            all_lines.append("--- visible text ---")
            for para in cap["text"].split("\n"):
                para = para.strip()
                if not para:
                    all_lines.append("")
                elif len(para) > 120:
                    all_lines.extend(textwrap.wrap(para, 120))
                else:
                    all_lines.append(para)

        if cap["a11y"]:
            all_lines.append("")
            all_lines.append("--- accessibility tree ---")
            all_lines.append(cap["a11y"])

        if i < len(captures) - 1:
            all_lines.append("")

        if image_path is None:
            image_path = str(cap["out_path"])

        # Aggregate cap: stop adding text if we've blown the budget.
        if len(captures) > 1 and len("\n".join(all_lines)) > MAX_AGGREGATE:
            remaining = len(captures) - i - 1
            if remaining > 0:
                all_lines.append(f"\n... ({remaining} more breakpoint(s) truncated from text; "
                                 "screenshots still captured)")
            break

    # Image tool-result contract: return {text, image_path}. On a vision model
    # (Anthropic v1) core base64s the PNG into the tool_result so the model SEES
    # the page; on any other model core uses `text` only (the path + extracted
    # text). A plain string would also work - the dict just adds the pixels.
    return {"text": "\n".join(all_lines), "image_path": image_path}


# --------------------------------------------------------------- system browser
def _platform():
    return sys.platform


def find_browser(family_only=None):
    """First installed, APPROVED browser in BROWSERS order -> (name, family, path) or None."""
    plat = _platform()
    ok = _approved()
    for name, family, path_names, mac, win in BROWSERS:
        if name not in ok or (family_only and family != family_only):
            continue
        if plat == "darwin":
            for root in ("/Applications", os.path.expanduser("~/Applications")):
                p = f"{root}/{mac[0]}.app/Contents/MacOS/{mac[1]}"
                if os.path.exists(p):
                    return name, family, p
        elif plat.startswith("win"):
            roots = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
            roots = [r for r in roots if r] or [r"C:\Program Files", r"C:\Program Files (x86)"]
            for root in roots:
                for rel in win:
                    p = root.rstrip("\\/") + "\\" + rel
                    if os.path.exists(p):
                        return name, family, p
        for n in path_names:
            p = shutil.which(n)
            if p:
                return name, family, p
    return None


def _kill_tree(proc):
    """Kill proc and everything it spawned (browsers fork helpers/renderers)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, 9)     # own session => pgid == pid
    except (OSError, subprocess.SubprocessError):
        _dbg_swallow()
    try:
        proc.kill()
    except OSError:
        _dbg_swallow()


_lc_view_tool = True   # marker: core's signal cleanup kills _CUR_PROC's tree
_CUR_PROC = None       # the browser Popen running right now (None when idle)


def _run_proc(argv, timeout=PROC_TIMEOUT, env=None):
    """Run in its own process group with a hard timeout; on timeout kill the whole
    tree and don't block on pipes held by orphans. env: extra variables merged over
    os.environ (None = inherit). -> (rc, stdout_text)."""
    kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    if env:
        kw["env"] = {**os.environ, **env}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
    else:
        kw["start_new_session"] = True
    global _CUR_PROC
    try:
        proc = subprocess.Popen(argv, **kw)
    except OSError:
        return -2, ""
    _CUR_PROC = proc
    try:
        # The post-kill reap wait (KILL_WAIT) comes out of the same budget.
        out, _ = proc.communicate(timeout=max(1, timeout - KILL_WAIT))
        return proc.returncode, (out or b"").decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=KILL_WAIT)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            _dbg_swallow()
        return -1, ""
    finally:
        _CUR_PROC = None


def _need_no_sandbox():
    """Chromium's sandbox fails only as root or inside a container."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def _chromium_argv(path, prof):
    a = [path, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
         "--disable-crash-reporter", f"--crash-dumps-dir={prof}"]
    if _need_no_sandbox():
        a.append("--no-sandbox")
    return a


def _chromium_env(prof):
    """Linux: keep Chromium's config/cache writes inside the temp profile."""
    if _os_key() != "linux":
        return None
    env = {}
    for var, sub in (("XDG_CONFIG_HOME", "xdg-config"), ("XDG_CACHE_HOME", "xdg-cache")):
        d = os.path.join(prof, sub)
        os.makedirs(d, exist_ok=True)
        env[var] = d
    return env


_FF_USER_JS = (
    'user_pref("toolkit.telemetry.enabled", false);\n'
    'user_pref("datareporting.healthreport.uploadEnabled", false);\n'
    'user_pref("datareporting.policy.dataSubmissionEnabled", false);\n'
)
_FF_ENV = {"MOZ_CRASHREPORTER_DISABLE": "1"}


def _firefox_profile_prep(prof):
    with open(os.path.join(prof, "user.js"), "w") as f:
        f.write(_FF_USER_JS)


# macOS system Firefox creates these under the real home even with a temp profile.
# We remove only those that didn't exist before and are empty after (117a).
_MAC_FF_SUBDIRS = ("Firefox", "Firefox/Pending Pings", "Firefox/Crash Reports",
                   "Firefox/Crash Reports/events")
# In-process lock: two concurrent view calls must not see each other's dirs as
# "absent before". Another process running Firefox at the same moment can still
# race us (residual, accepted: rmdir only ever removes empty dirs).
_MAC_FF_LOCK = threading.Lock()


def _home():
    return os.path.expanduser("~")


def _mac_ff_paths():
    base = os.path.join(_home(), "Library", "Application Support")
    return [os.path.join(base, *d.split("/")) for d in _MAC_FF_SUBDIRS]


def _mac_ff_snapshot():
    return {p for p in _mac_ff_paths() if os.path.isdir(p)}


def _mac_ff_cleanup(before):
    """rmdir (never rmtree) deepest-first the dirs absent before; rmdir fails on a
    non-empty dir, which is exactly the "keep it" case. Errors ignored."""
    for p in sorted(_mac_ff_paths(), key=lambda x: x.count(os.sep), reverse=True):
        if p in before:
            continue
        try:
            os.rmdir(p)
        except OSError:
            pass


class _TextParser(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "head", "svg"}

    def __init__(self):
        super().__init__()
        self.out, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.out.append(data.strip() + " ")


def _html_text(html, max_chars=MAX_TEXT):
    p = _TextParser()
    try:
        p.feed(html)
    except Exception:
        _dbg_swallow()
    txt = re.sub(r"\n\s*\n+", "\n\n", "".join(p.out)).strip()
    return txt[:max_chars]


def _run_basic(url, args, found):
    name, family, path = found
    w, h = int(args.get("width") or 1280), int(args.get("height") or 800)
    out = _out_dir() / f"{_slug(url)}.png"
    text = ""
    if family == "firefox":
        prof = tempfile.mkdtemp(prefix="ff-profile-", dir=str(_out_dir()))
        mac = _platform() == "darwin"
        lock = _MAC_FF_LOCK if mac else None
        if lock:
            lock.acquire()
        before = _mac_ff_snapshot() if mac else None
        try:
            _firefox_profile_prep(prof)
            _run_proc([path, "--headless", "-no-remote", "-profile", prof, "--screenshot", str(out),
                       f"--window-size={w},{h}", url], env=dict(_FF_ENV))
        finally:
            shutil.rmtree(prof, ignore_errors=True)
            if mac:
                _mac_ff_cleanup(before)
            if lock:
                lock.release()
    else:
        # Two loads (screenshot, then --dump-dom) share ONE PROC_TIMEOUT budget; the
        # DOM run gets what the screenshot left and is skipped if under 3 s.
        prof = tempfile.mkdtemp(prefix="cr-profile-", dir=str(_out_dir()))
        try:
            base = _chromium_argv(path, prof)
            env = _chromium_env(prof)
            t0 = time.monotonic()
            _run_proc(base + [f"--screenshot={out}", f"--window-size={w},{h}", url], env=env)
            left = PROC_TIMEOUT - (time.monotonic() - t0)
            if out.is_file() and left >= 3 + KILL_WAIT:
                rc, dom = _run_proc(base + ["--dump-dom", url], timeout=left, env=env)
                if rc == 0 and dom:
                    text = _html_text(dom)
        finally:
            shutil.rmtree(prof, ignore_errors=True)
    if not out.is_file():
        return f"error: {name} could not render {url} (timed out or failed)"
    ignored = [k for k in PW_ONLY if args.get(k)]
    if url.lower().split("?")[0].endswith(".pdf"):
        note = f"PDF shown in {name}'s viewer (first page)."
    else:
        note = f"basic browser mode ({name}); install Playwright for full features."
    if ignored:
        note += f" Ignored: {', '.join(ignored)}."
    lines = ["image attached", f"url: {url}", f"viewport: {w}x{h}", note]
    if text:
        lines += ["", "--- visible text ---"]
        for para in text.split("\n"):
            para = para.strip()
            lines.extend(textwrap.wrap(para, 120) if len(para) > 120 else [para])
    return {"text": "\n".join(lines), "image_path": str(out)}


def _have_playwright():
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


_PW_PY = {}   # 119b: cached system Python with Playwright ("py": argv list or None)


def _mac_clt():
    """True if macOS Command Line Tools/Xcode is installed (so /usr/bin/python3 is real)."""
    if Path("/Library/Developer/CommandLineTools/usr/bin/python3").exists():
        return True
    try:
        r = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and Path(r.stdout.strip()).is_dir()
    except (OSError, subprocess.SubprocessError):
        return False


def _pw_py_candidates():
    """System Python argv candidates (sys.executable itself excluded)."""
    win = _os_key() == "win"
    names = ("py", "python", "python3") if win else ("python3", "python")
    out, me = [], os.path.normcase(os.path.abspath(sys.executable or ""))
    mac_stub = _os_key() == "darwin" and not _mac_clt()
    def _add(argv):
        a = os.path.normcase(os.path.abspath(argv[0]))
        if "\\windowsapps\\" in a.lower().replace("/", "\\"):   # Store App Execution Alias stub
            return
        if mac_stub and a.startswith("/usr/bin/"):   # CLT stub: running it pops a GUI dialog
            return
        if a != me and argv not in out:
            out.append(argv)
    for n in names:
        w = shutil.which(n)
        if w:
            _add([w, "-3"] if n == "py" else [w])
    if _os_key() == "darwin":   # Homebrew may sit behind a skipped /usr/bin stub on PATH
        for b in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3"):
            if os.path.exists(b):
                _add([b])
    if win:
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
        if os.environ.get("LOCALAPPDATA") and base.is_dir():
            for d in sorted(base.glob("Python3*"), reverse=True):
                if (d / "python.exe").exists():
                    _add([str(d / "python.exe")])
    return out


def _pw_python():
    """First candidate that can import playwright.sync_api (cached per process)."""
    if "py" not in _PW_PY:
        _PW_PY["py"] = None
        for argv in _pw_py_candidates():
            rc, _ = _run_proc(argv + ["-c", "import playwright.sync_api"], timeout=10)
            if rc == 0:
                _PW_PY["py"] = argv
                break
    return _PW_PY["py"]


def _run_playwright_sub(py, url, args, order):
    """Run the Playwright path via this same file in another interpreter (119b).
    -> result, or None on failure (caller falls back to system browsers)."""
    import json
    d = _out_dir()
    n = next(_SEQ)
    req, res = d / f"pwreq-{n}.json", d / f"pwres-{n}.json"
    req.write_text(json.dumps({"url": url, "args": args, "order": order,
                               "dir": str(d), "out": str(res)}), encoding="utf-8")
    try:
        _run_proc(py + [str(Path(__file__).resolve()), "--pw-run", str(req)],
                  timeout=PW_DEADLINE + 15)
        r = json.loads(res.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        for f in (req, res):
            try:
                f.unlink()
            except OSError:
                pass
    if r.get("noengine"):
        return None
    return r.get("result")


def _pw_child(req_path):
    """Hidden CLI entry: `python view.py --pw-run <req.json>`."""
    import json
    q = json.loads(Path(req_path).read_text(encoding="utf-8"))
    _OUT["dir"] = q["dir"]   # write images into the parent's lc-view dir; no atexit rmtree
    try:
        out = {"result": _run_playwright(q["url"], q["args"], q["order"])}
    except _NoEngine as e:
        out = {"noengine": str(e)}
    except Exception as e:   # keep the parent informed instead of a silent missing file
        out = {"error": f"{type(e).__name__}: {e}"[:300]}
    Path(q["out"]).write_text(json.dumps(out), encoding="utf-8")


def _render(url, args):
    # Headless Chromium has no PDF viewer (blank page; under Playwright it's a download).
    # System Firefox's --screenshot renders PDFs with its built-in viewer.
    if url.lower().split("?")[0].endswith(".pdf"):
        found = find_browser("firefox")
        if not found:
            return ("error: PDFs need Firefox installed (headless Chrome/Chromium can't "
                    "display PDFs).")
        return _run_basic(url, args, found)
    pw = _pw_approved()
    if pw and _have_playwright():
        try:
            return _run_playwright(url, args, pw)
        except _NoEngine as e:
            pw_err = str(e)   # finding 5: fall back to the approved system browser
            found = find_browser()
            if not found:
                return pw_err
            return _run_basic(url, args, found)
    py = _pw_python() if pw else None
    if py:
        r = _run_playwright_sub(py, url, args, pw)
        if r is not None:
            return r
        _PW_PY["py"] = None   # don't pay the deadline again on every call
    found = find_browser()
    if not found:
        return no_browser_msg()
    return _run_basic(url, args, found)


# ------------------------------------------------------------------- file
def _sniff_image(data):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    return None


def _image_size(kind, data):
    """(w, h) if cheaply known from the header, else None."""
    try:
        if kind == "png":
            return struct.unpack(">II", data[16:24])
        if kind == "gif":
            return struct.unpack("<HH", data[6:10])
        if kind == "bmp":
            w, h = struct.unpack("<ii", data[18:26])
            return w, abs(h)
        if kind == "jpeg":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    return None
                m = data[i + 1]
                seg = struct.unpack(">H", data[i + 2:i + 4])[0]
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                i += 2 + seg
    except struct.error:
        _dbg_swallow()
    return None


def _view_file(raw, args, cwd):
    if not raw:
        return "error: file: needs a path"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(cwd) / p
    if not p.is_file():
        return f"error: file not found: {p}"
    with open(p, "rb") as f:
        head = f.read(65536)
    kind = _sniff_image(head)
    if kind == "bmp" and not _have_pil():
        kind = None
    if kind:
        n = p.stat().st_size
        if n > FILE_IMAGE_MAX:
            return (f"error: image too large: {p} is {n} bytes "
                    f"(limit {FILE_IMAGE_MAX // 1_000_000} MB)")
        size = _image_size(kind, head)
        dims = f" {size[0]}x{size[1]}" if size else ""
        return {"text": f"{p}{dims} {p.stat().st_size} bytes", "image_path": str(p)}
    ext = p.suffix.lower().lstrip(".")
    if ext in RENDER_EXTS:
        return _render(p.resolve().as_uri(), args)
    return (f"error: unsupported file type {p.suffix or '(none)'}; supported: "
            f"{', '.join(_image_exts() + RENDER_EXTS)}")


# ------------------------------------------------------------------- screen
def _is_ssh():
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


_WIN_PS = (
    "Add-Type -Name D -Namespace W -MemberDefinition "
    "'[DllImport(\"user32.dll\")] public static extern bool SetProcessDPIAware();'; "
    "[void][W.D]::SetProcessDPIAware(); "
    "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
    "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen; "
    "$i=New-Object System.Drawing.Bitmap $b.Width,$b.Height; "
    "$g=[System.Drawing.Graphics]::FromImage($i); "
    "$g.CopyFromScreen($b.Left,$b.Top,0,0,$i.Size); "
    "$i.Save('{out}',[System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $i.Dispose()")


def _screen_cmd(out):
    """-> (argv, None) or (None, reason)."""
    plat = _platform()
    if plat.startswith("win"):
        if _is_ssh():
            return None, "no desktop session (SSH runs outside the user's desktop)"
        return ["powershell", "-NoProfile", "-Command",
                _WIN_PS.replace("{out}", str(out).replace("'", "''"))], None
    if plat == "darwin":
        return ["screencapture", "-x", str(out)], None
    way, x11 = os.environ.get("WAYLAND_DISPLAY"), os.environ.get("DISPLAY")
    if not (way or x11):
        return None, "no display"
    cands = []
    if way:
        cands.append(("grim", [str(out)]))
    if x11:
        cands += [("scrot", ["-o", str(out)]), ("import", ["-window", "root", str(out)])]
    cands.append(("gnome-screenshot", ["-f", str(out)]))
    for tool, rest in cands:
        exe = shutil.which(tool)
        if exe:
            return [exe] + rest, None
    return None, "no capture tool (install grim or scrot)"


def _view_screen():
    out = _out_dir() / f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}_{next(_SEQ)}.png"
    argv, reason = _screen_cmd(out)
    if argv:
        rc, _ = _run_proc(argv)
        if rc == 0 and out.is_file() and out.stat().st_size > 0:
            return {"text": "screen captured; image attached", "image_path": str(out)}
        plat = _platform()
        if plat == "darwin":
            reason = "capture failed; grant Screen Recording to sshd/Terminal"
        elif plat.startswith("win"):
            reason = "capture failed; no interactive desktop session"
        else:
            reason = f"capture failed ({Path(argv[0]).name} exit {rc}); no display access"
    return f"error: screen: {reason}"


def run(args, cwd):
    kind, val = parse_target(args.get("target") or args.get("url") or "")
    if kind == "screen":
        return _view_screen()
    if kind == "file":
        return _view_file(val, args, cwd)
    if not val:
        return "error: view needs a target (web:<url>, file:<path> or screen:)"
    return _render(val, args)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--pw-run":
    _pw_child(sys.argv[2])
