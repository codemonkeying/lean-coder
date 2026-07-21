"""web_screenshot - capture a screenshot + text snapshot of a URL using a headless browser.

  - basic:         web_screenshot(url=...)
                       saves screenshot to ~/screenshots/<slug>.png
                       returns path + visible text snapshot
  - full page:     web_screenshot(url=..., full_page=true)
                       captures the full scrollable page, not just the viewport
  - custom size:   web_screenshot(url=..., width=1440, height=900)
  - mobile:        web_screenshot(url=..., mobile=true)
                       390x844 viewport, mobile user-agent, touch enabled
  - device:        web_screenshot(url=..., device="iPhone 14")
                       full Playwright device emulation (viewport, UA, scale)
  - dark mode:     web_screenshot(url=..., dark=true)
                       emulates prefers-color-scheme: dark
  - breakpoints:   web_screenshot(url=..., breakpoints=[375, 768, 1280])
                       one call, multiple screenshots at different widths
                       (auto-enables full_page unless explicitly false)
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
                "default": False,
                "description": "Capture the full scrollable page (default false = viewport only).",
            },
            "width": {
                "type": "integer",
                "default": 1280,
                "description": "Viewport width in pixels (default 1280).",
            },
            "height": {
                "type": "integer",
                "default": 800,
                "description": "Viewport height in pixels (default 800).",
            },
            "mobile": {
                "type": "boolean",
                "default": False,
                "description": "Emulate a mobile device: 390x844 viewport, mobile user-agent, touch, 2x scale (default false). Overridden by device.",
            },
            "device": {
                "type": "string",
                "description": "Playwright device name for full emulation (e.g. 'iPhone 14', 'Pixel 7', 'iPad Mini'). Sets viewport, UA, scale, touch. See playwright.dev/python/docs/emulation#devices.",
            },
            "dark": {
                "type": "boolean",
                "default": False,
                "description": "Emulate prefers-color-scheme: dark (default false = light).",
            },
            "breakpoints": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of viewport widths to capture (e.g. [375, 768, 1280]). One screenshot per width. Auto-enables full_page unless full_page is explicitly false.",
            },
            "wait_for": {
                "type": "string",
                "description": "CSS selector to wait for before capturing. Useful for JS-rendered pages.",
            },
            "dom": {
                "type": "boolean",
                "default": False,
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
MAX_AGGREGATE = 20_000  # total text cap across all breakpoints


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
        return launcher.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
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
    from playwright.sync_api import TimeoutError as PWTimeout

    context = browser.new_context(**ctx_args)
    page = context.new_page()
    page.set_default_timeout(timeout)

    try:
        page.goto(url, wait_until="networkidle", timeout=timeout)
    except PWTimeout:
        try:
            page.goto(url, wait_until="load", timeout=timeout)
        except PWTimeout:
            pass

    if wait_for:
        try:
            page.wait_for_selector(wait_for, timeout=5_000)
        except PWTimeout:
            pass

    title = page.title() or "(no title)"
    final_url = page.url
    vp = ctx_args.get("viewport", {})

    page.screenshot(path=str(out_path), full_page=full_page)

    text = _visible_text(page, MAX_TEXT)
    a11y = _a11y_snapshot(page, MAX_A11Y) if want_dom else None

    context.close()

    return {
        "title": title, "final_url": final_url,
        "text": text, "a11y": a11y, "out_path": out_path,
        "viewport": vp, "full_page": full_page,
    }, None


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

    wait_for = args.get("wait_for") or None
    want_dom = bool(args.get("dom", False))
    breakpoints = args.get("breakpoints") or None

    # full_page: default false normally, default true for breakpoints.
    if "full_page" in args:
        full_page = bool(args["full_page"])
    else:
        full_page = bool(breakpoints)

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(url)

    try:
        with sync_playwright() as p:
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
                return (
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
                        bp_path = SCREENSHOT_DIR / f"{slug}_{bp_width}w.png"
                        result, cap_err = _capture_one(
                            browser, bp_args, url, full_page, wait_for, want_dom, bp_path, TIMEOUT)
                        if cap_err:
                            return cap_err
                        captures.append(result)
                else:
                    # Single screenshot.
                    out_path = SCREENSHOT_DIR / f"{slug}.png"
                    result, cap_err = _capture_one(
                        browser, ctx_args, url, full_page, wait_for, want_dom, out_path, TIMEOUT)
                    if cap_err:
                        return cap_err
                    captures.append(result)
            finally:
                browser.close()
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
        all_lines.append(f"screenshot: {cap['out_path']}")
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
            all_lines.append(f"wait_for: {wait_for!r}")

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
                                 f"screenshots still saved to ~/screenshots/)")
            break

    # Image tool-result contract: return {text, image_path}. On a vision model
    # (Anthropic v1) core base64s the PNG into the tool_result so the model SEES
    # the page; on any other model core uses `text` only (the path + extracted
    # text). A plain string would also work - the dict just adds the pixels.
    return {"text": "\n".join(all_lines), "image_path": image_path}
