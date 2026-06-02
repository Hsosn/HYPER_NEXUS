"""
Browser automation tools using Playwright.

CRITICAL FIX (v2): Now uses the shared BrowserManager from browser_routes
instead of maintaining a separate Playwright instance. This ensures:
- The live preview viewer shows the SAME page the agent is interacting with
- Cookies, sessions, and login state persist between tool calls
- No duplicate browser processes consuming memory

v3 FIX: Falls back to a standalone Playwright instance when browser_routes
is unavailable (e.g., running without the API server). This makes browser
tools work in both web-ui mode and standalone/CLI mode.

The BrowserManager provides a persistent page that the live viewer streams
from, so what the agent sees is exactly what the user sees in the preview.

Install: pip install playwright && playwright install chromium

Falls back gracefully if Playwright is not installed.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime
from pathlib import Path

from ..registry import tool
from ...events import emit
from ...config import BASE_DIR

logger = logging.getLogger("nexus.browser_tools")

_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

_INSTALL_MSG = (
    "Playwright is not installed. Run: pip install playwright && playwright install chromium"
)

# ── Standalone Playwright manager (fallback when browser_routes unavailable) ──

_standalone_pw = None
_standalone_browser = None
_standalone_page = None


async def _get_standalone_page():
    """Get or create a standalone Playwright page for use without browser_routes."""
    global _standalone_pw, _standalone_browser, _standalone_page
    if _standalone_page and not _standalone_page.is_closed():
        return _standalone_page
    # Clean up old instances
    if _standalone_browser:
        try:
            await _standalone_browser.close()
        except Exception:
            pass
    if _standalone_pw:
        try:
            await _standalone_pw.stop()
        except Exception:
            pass
    _standalone_pw = await async_playwright().start()
    _standalone_browser = await _standalone_pw.chromium.launch(headless=True)
    _standalone_page = await _standalone_browser.new_page(viewport={"width": 1280, "height": 720})
    return _standalone_page


async def _cleanup_standalone():
    """Clean up standalone browser resources."""
    global _standalone_pw, _standalone_browser, _standalone_page
    for resource in (_standalone_page, _standalone_browser, _standalone_pw):
        if resource:
            try:
                await resource.close() if hasattr(resource, 'close') else await resource.stop()
            except Exception:
                pass
    _standalone_page = _standalone_browser = _standalone_pw = None


# ── Unified page getter with fallback ──────────────────────────────────────

_USE_BROWSER_ROUTES = None  # lazy-detected


async def _get_page_and_manager():
    """Get a Playwright page and optional manager (with 20s overall timeout).

    Returns (page, manager_or_none). If browser_routes is available,
    returns the shared manager's page. Otherwise, returns a standalone page.
    """
    global _USE_BROWSER_ROUTES
    if _USE_BROWSER_ROUTES is None:
        try:
            from ...api.browser_routes import get_manager
            # Try to get the manager — if it works, use it
            mgr = get_manager()
            _USE_BROWSER_ROUTES = True
            page = await asyncio.wait_for(mgr.get_page(), timeout=15)
            return page, mgr
        except (ImportError, Exception):
            _USE_BROWSER_ROUTES = False
            logger.info("browser_routes unavailable, using standalone Playwright")

    if _USE_BROWSER_ROUTES:
        from ...api.browser_routes import get_manager
        mgr = get_manager()
        page = await asyncio.wait_for(mgr.get_page(), timeout=15)
        return page, mgr
    else:
        page = await asyncio.wait_for(_get_standalone_page(), timeout=20)
        return page, None


@tool(
    name="browser_open",
    description=(
        "Open a URL in a headless browser and return the page title and visible text. "
        "More reliable than fetch_url for JavaScript-heavy pages. "
        "The page stays open in the Browser Live viewer for real-time preview."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "url":       {"type": "string"},
            "max_chars": {"type": "integer", "default": 8000},
            "wait_ms":   {"type": "integer", "default": 1500,
                          "description": "ms to wait after page load for JS to settle"},
        },
        "required": ["url"],
    },
    risk="medium",
    category="browser",
    timeout=30,  # 30 second timeout (browser launch can take ~10s + 15s nav)
)
async def browser_open(params):
    if not _PLAYWRIGHT_AVAILABLE:
        return f"BROWSER NOT AVAILABLE.\n\nTo fix, run in terminal:\npip install playwright\nplaywright install chromium\n\nFor now, use fetch_url to get website content."
    url = params["url"]
    # FIX: Default values now match the schema definitions (1500ms and 8000 chars)
    # Previously the code used 1000ms and 5000 chars, which didn't match the schema.
    wait_ms = params.get("wait_ms", 1500)
    max_chars = params.get("max_chars", 8000)
    try:
        page, mgr = await _get_page_and_manager()

        # Navigate — use manager.navigate() if available, else page.goto()
        if mgr:
            result = await asyncio.wait_for(mgr.navigate(url), timeout=15)
            if result.get("status") == "error":
                err = result.get('error', 'Navigation failed')
                return f"BROWSER NAVIGATION FAILED: {err}\n\nTry fetch_url instead."
            title = result.get("title", "")
        else:
            response = await asyncio.wait_for(page.goto(url, wait_until="domcontentloaded", timeout=15000), timeout=15)
            title = await page.title()

        await page.wait_for_timeout(wait_ms)

        if not title:
            title = await page.title()
        text = await asyncio.wait_for(
            page.evaluate(f"() => document.body.innerText.slice(0, {max_chars})"),
            timeout=5,
        )

        await emit("browser_action", action="open", url=url, title=title, source="agent_tool")
        return f"Opened {url}\nTitle: {title}\nContent preview:\n{text[:500]}"
    except asyncio.TimeoutError:
        return "BROWSER TIMEOUT - Page took too long to load. Site may be slow or down."
    except Exception as e:
        return f"BROWSER ERROR: {str(e)[:200]}\n\nFallback: use fetch_url tool instead."


@tool(
    name="browser_screenshot",
    description=(
        "Take a screenshot of the current browser page and save to workspace. "
        "Returns the filename where screenshot is saved. "
        "The page stays open in the Browser Live viewer. "
        "Optional param: url - navigate to URL before screenshot."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "url":     {"type": "string", "description": "Optional URL to navigate to before screenshot"},
            "full_page": {"type": "boolean", "description": "Capture full page", "default": False},
            "filename": {"type": "string", "description": "Optional output filename"},
        },
        "required": [],
    },
    risk="medium",
    category="browser",
)
async def browser_screenshot(params):
    if not _PLAYWRIGHT_AVAILABLE:
        return _INSTALL_MSG

    url = params.get("url")
    full = bool(params.get("full_page", False))
    custom_filename = params.get("filename", "")

    try:
        page, mgr = await _get_page_and_manager()

        # Navigate if URL provided
        if url:
            if mgr:
                result = await mgr.navigate(url)
                if result.get("status") == "error":
                    return f"Navigation error: {result.get('error', 'Failed')}"
            else:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)

        await page.wait_for_timeout(1000)

        # Take screenshot
        shot = await page.screenshot(full_page=full)

        if not shot or len(shot) == 0:
            return "Error: Screenshot capture returned empty data"

        b64 = base64.b64encode(shot).decode()

        base_dir = BASE_DIR
        current_url = page.url
        if base_dir:
            workspace = base_dir / "data" / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            host_part = current_url.split("://")[-1].split("/")[0][:50] if current_url else "page"

            if custom_filename:
                filename = custom_filename if custom_filename.endswith(".png") else custom_filename + ".png"
            else:
                filename = f"screenshot_{ts}_{host_part}.png"

            filepath = workspace / filename
            filepath.write_bytes(shot)

            await emit("browser_action", action="screenshot", url=current_url, filepath=str(filepath), size=len(shot), image=f"data:image/png;base64,{b64}", source="agent_tool")

            return f"Screenshot: {filename} ({len(shot)} bytes) - check Browser Live popup!"
        else:
            await emit("browser_action", action="screenshot", url=current_url, size=len(shot), image=f"data:image/png;base64,{b64}", source="agent_tool")
            return f"Screenshot ({len(shot)} bytes) - check Browser Live popup!"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Screenshot error: {e}\n\nDetails:\n{tb}"


@tool(
    name="browser_click",
    description="Click an element on the current browser page by CSS selector or text content.",
    parameters_schema={
        "type": "object",
        "properties": {
            "url":      {"type": "string", "description": "Optional URL to navigate to first"},
            "selector": {"type": "string", "description": "CSS selector or text to click"},
            "by_text":  {"type": "boolean", "default": False,
                         "description": "If true, match by visible text instead of CSS selector"},
        },
        "required": ["selector"],
    },
    risk="medium",
    category="browser",
)
async def browser_click(params):
    if not _PLAYWRIGHT_AVAILABLE:
        return _INSTALL_MSG
    url      = params.get("url")
    selector = params["selector"]
    by_text  = bool(params.get("by_text", False))
    try:
        page, mgr = await _get_page_and_manager()

        if url:
            if mgr:
                await mgr.navigate(url)
            else:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)

        if by_text:
            await page.get_by_text(selector).first.click()
        else:
            await page.click(selector, timeout=5000)
        await page.wait_for_timeout(1000)

        title = await page.title()
        current_url = page.url
        await emit("browser_action", action="click", url=current_url, selector=selector, source="agent_tool")
        return f"Clicked '{selector}'. New page title: {title}"
    except Exception as e:
        return f"Click error: {e}"


@tool(
    name="browser_fill_form",
    description=(
        "Fill in form fields on the current browser page and optionally submit. "
        "Provide fields as a dict of {selector: value}."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "url":    {"type": "string", "description": "Optional URL to navigate to first"},
            "fields": {"type": "object",
                       "description": "CSS selector → value pairs",
                       "additionalProperties": {"type": "string"}},
            "submit_selector": {"type": "string",
                                "description": "CSS selector of submit button (optional)"},
        },
        "required": ["fields"],
    },
    risk="high",
    category="browser",
)
async def browser_fill_form(params):
    if not _PLAYWRIGHT_AVAILABLE:
        return _INSTALL_MSG
    url     = params.get("url")
    fields  = params.get("fields", {})
    submit  = params.get("submit_selector")
    try:
        page, mgr = await _get_page_and_manager()

        if url:
            if mgr:
                await mgr.navigate(url)
            else:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)

        for selector, value in fields.items():
            await page.fill(selector, str(value))
        if submit:
            await page.click(submit)
            await page.wait_for_timeout(1500)
        title = await page.title()
        current_url = page.url
        filled = list(fields.keys())
        await emit("browser_action", action="fill_form", url=current_url, fields=filled, source="agent_tool")
        return f"Filled {len(filled)} field(s): {filled}. Page title after: {title}"
    except Exception as e:
        return f"Form fill error: {e}"
