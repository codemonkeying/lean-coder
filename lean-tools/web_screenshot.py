"""web_screenshot - capture a screenshot + text snapshot of a URL using a headless browser.

  - basic:         web_screenshot(url=...)
                       saves screenshot to ~/screenshots/<slug>.png
                       returns path + visible text snapshot
  - full page:     web_screenshot(url=..., full_page=true)
                       captures the full scrollable page, not just the viewport
  - custom size:   web_screenshot(url=..., width=1440, height=900)
  - js wait:       web_screenshot(url=..., wait_for=".my-selector")
                       waits for a CSS selector to appear before capturing
  - dom snapshot:  web_screenshot(url=..., dom=true)
                       includes the accessibility tree text alongside the screenshot

Requires: playwright + at least one browser engine.
  Install: pip3 install --user --break-system-packages playwright
           python3 -m playwright install firefox   # or: chromium / webkit

Engine: auto-detects a working engine (tries chromium, then firefox, then
webkit), or force one with browser="firefox". On some boxes the bundled
chromium is broken (SIGTRAPs headless) - firefox is the reliable fallback.

Primary use: visually check a WordPress site / theme / local dev server.
Private/localhost URLs are allowed (http://localhost:8080, LAN IPs, etc.).

NETWORK EGRESS - not marked safe; every call shows the URL and confirms
(auto-approve waives).
"""
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

TOOL = {
    "name": "web_screenshot",
    "description": (
        "Capture a screenshot of a URL using a headless browser (Playwright; chromium/firefox/webkit). "
        "Saves a PNG to ~/screenshots/ and returns the file path + visible text snapshot. "
        "Works with localhost, LAN addresses, and public URLs. "
        "Use for checking WordPress themes, layouts, and frontend rendering."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to capture. Bare domain gets https:// prepended. localhost/LAN allowed.",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture the full scrollable page (default false = viewport only).",
            },
            "width": {
                "type": "integer",
                "description": "Viewport width in pixels (default 1280).",
            },
            "height": {
                "type": "integer",
                "description": "Viewport height in pixels (default 800).",
            },
            "wait_for": {
                "type": "string",
                "description": "CSS selector to wait for before capturing. Useful for JS-rendered pages.",
            },
            "dom": {
                "type": "boolean",
                "description": "Include a full accessibility-tree text dump alongside the screenshot (default false).",
            },
            "browser": {
                "type": "string",
                "description": "Force an engine: 'chromium', 'firefox', or 'webkit'. Default: auto (tries each until one launches).",
            },
        },
    "required": ["url"],
    },
    # no "safe": egress goes through the confirm gate
    # driver_only: the browser + playwright live on the DRIVER (set up once), and
    # the image-return pipeline base64s the PNG from the driver's ~/screenshots.
    # Without this the tool gets pushed to a connected remote executor and tries to
    # apt-install pip + playwright + a browser (~hundreds of MB) onto the user's
    # server just to screenshot a URL - which it can reach over the network anyway.
    "driver_only": True,
}

TIMEOUT = 30_000   # ms for playwright operations
SCREENSHOT_DIR = Path.home() / "screenshots"
MAX_TEXT = 6_000   # chars of visible text returned
MAX_A11Y = 10_000  # chars of accessibility tree if dom=true


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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{host}_{path}_{ts}" if path else f"{host}_{ts}"
    return re.sub(r"_+", "_", name).strip("_")


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


def _launch(p, engine):
    """Launch one engine. chromium takes hardening flags; others don't."""
    launcher = getattr(p, engine)
    if engine == "chromium":
        return launcher.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    return launcher.launch()


def run(args, cwd):
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return (
            "error: playwright not installed.\n"
            "Install with:\n"
            "  pip3 install --user --break-system-packages playwright\n"
            "  python3 -m playwright install firefox   # or chromium / webkit"
        )

    forced = (args.get("browser") or "").strip().lower() or None
    if forced and forced not in ENGINES:
        return f"error: unknown browser {forced!r}; choose one of {', '.join(ENGINES)}"
    order = [forced] if forced else list(ENGINES)

    url = _normalise_url(args.get("url") or "")
    if not url:
        return "error: web_screenshot needs a url"

    full_page = bool(args.get("full_page", False))
    width = int(args.get("width") or 1280)
    height = int(args.get("height") or 800)
    wait_for = args.get("wait_for") or None
    want_dom = bool(args.get("dom", False))

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(url)
    out_path = SCREENSHOT_DIR / f"{slug}.png"

    try:
        with sync_playwright() as p:
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
                return (
                    "error: could not launch any browser engine.\n"
                    f"{hint}\n"
                    f"launch attempts:\n{detail}"
                )
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.set_default_timeout(TIMEOUT)

                try:
                    page.goto(url, wait_until="networkidle", timeout=TIMEOUT)
                except PWTimeout:
                    # networkidle can time out on heavy WP pages - fall back to load
                    try:
                        page.goto(url, wait_until="load", timeout=TIMEOUT)
                    except PWTimeout:
                        pass  # capture whatever we have

                if wait_for:
                    try:
                        page.wait_for_selector(wait_for, timeout=5_000)
                    except PWTimeout:
                        pass  # capture anyway; note below

                title = page.title() or "(no title)"
                final_url = page.url

                page.screenshot(path=str(out_path), full_page=full_page)

                text = _visible_text(page, MAX_TEXT)
                a11y = _a11y_snapshot(page, MAX_A11Y) if want_dom else None
            finally:
                browser.close()
    except Exception as e:
        return f"error running headless browser: {e}"

    # Build result
    lines = [
        f"screenshot: {out_path}",
        f"title: {title}",
        f"url: {final_url}",
        f"viewport: {width}x{height}" + (" (full page)" if full_page else ""),
        f"engine: {engine_used}",
    ]
    if wait_for:
        lines.append(f"wait_for: {wait_for!r}")

    if text:
        lines.append("")
        lines.append("--- visible text ---")
        # wrap long lines for readability
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                lines.append("")
            elif len(para) > 120:
                lines.extend(textwrap.wrap(para, 120))
            else:
                lines.append(para)

    if a11y:
        lines.append("")
        lines.append("--- accessibility tree ---")
        lines.append(a11y)

    # Image tool-result contract: return {text, image_path}. On a vision model
    # (Anthropic v1) core base64s the PNG into the tool_result so the model SEES
    # the page; on any other model core uses `text` only (the path + extracted
    # text). A plain string would also work - the dict just adds the pixels.
    return {"text": "\n".join(lines), "image_path": str(out_path)}
