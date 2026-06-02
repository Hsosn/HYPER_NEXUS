"""
Browser Live Routes — REST API + WebSocket screenshot streaming for live browser preview.

Provides:
- REST endpoints for browser page interaction (navigate, click, type, keyboard, scroll)
- /api/browser/stream  — WebSocket endpoint that streams JPEG screenshots at ~8 fps
- BrowserManager — manages a persistent Playwright page for live interaction

The BrowserManager mirrors agent actions: when the agent's browser_* tools emit
browser_action events via EventBus, the manager navigates the persistent page to
follow along. Users can also interact directly via REST endpoints and the live viewer.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel

from .. import config
from ..events import emit, BUS
from ..auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/browser", tags=["browser-live"])

_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
    print("[browser] [OK] Playwright available")
except ImportError as e:
    print(f"[browser] [FAIL] Playwright NOT installed: {e}")
    print("[browser] To fix, run: pip install playwright && playwright install chromium")


# ── BrowserManager ────────────────────────────────────────────────────────

class BrowserManager:
    """
    Manages a persistent Playwright browser + page for live streaming.

    The manager keeps a single page open. When the agent uses browser_* tools
    and emits browser_action events, the manager mirrors the navigation so
    the live stream stays in sync. Users can also interact directly.

    Thread-safe: all operations use asyncio locks.
    """

    def __init__(self):
        self._browser = None
        self._pw = None
        self._page = None
        self._ctx = None
        self._lock = asyncio.Lock()
        self._url = ""
        self._title = ""
        self._is_open = False
        self._viewport = {"width": 1280, "height": 720}
        # No subscribers list needed — streaming is direct via WebSocket

    async def get_browser(self):
        """Get or create the shared browser instance. Protected against concurrent calls."""
        async with self._lock:
            if self._browser and self._browser.is_connected():
                return self._browser

            if not _PLAYWRIGHT_AVAILABLE:
                raise RuntimeError("Playwright not installed. Fix: pip install playwright && playwright install chromium")

            print("[browser] Starting Playwright...")
            self._pw = await async_playwright().start()

            launch_args = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ]
            }

            http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
            https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            if http_proxy or https_proxy:
                proxy = https_proxy or http_proxy  # Prefer HTTPS proxy
                launch_args["proxy"] = {"server": proxy}
                print(f"[browser] Using proxy: {proxy}")

            print("[browser] Launching Chromium...")
            self._browser = await self._pw.chromium.launch(**launch_args)
            print("[browser] Chromium launched [OK]")
            return self._browser

    async def get_page(self):
        """Get or create the persistent page."""
        async with self._lock:
            if self._page and not self._page.is_closed():
                return self._page

            browser = await self.get_browser()
            self._ctx = await browser.new_context(
                viewport=self._viewport,
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self._page = await self._ctx.new_page()
            # Start with a blank page
            await self._page.goto("about:blank")
            self._url = "about:blank"
            self._title = ""
            self._is_open = True
            return self._page

    @staticmethod
    def _validate_url(url: str) -> str:
        """Validate URL to prevent SSRF attacks. Only http/https schemes allowed."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL scheme '{parsed.scheme}' not allowed. Use http(s)://")

        hostname = parsed.hostname or ""

        # Block localhost and loopback
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"):
            raise ValueError(f"Access to '{hostname}' is not allowed (loopback address)")

        # Block AWS/GCP cloud metadata endpoint
        if hostname == "169.254.169.254":
            raise ValueError("Access to cloud metadata endpoint (169.254.169.254) is not allowed")

        # Block private network ranges (10.x, 172.16-31.x, 192.168.x)
        import ipaddress
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private:
                raise ValueError(f"Access to private network address '{hostname}' is not allowed")
            if ip.is_loopback:
                raise ValueError(f"Access to loopback address '{hostname}' is not allowed")
            if ip.is_link_local:
                raise ValueError(f"Access to link-local address '{hostname}' is not allowed")
            if ip.is_reserved:
                raise ValueError(f"Access to reserved address '{hostname}' is not allowed")
        except ValueError:
            # Not an IP address (could be a domain) — check for common patterns
            lower_host = hostname.lower()
            if lower_host.endswith(".local") or lower_host.endswith(".internal"):
                raise ValueError(f"Access to internal domain '{hostname}' is not allowed")

        return url

    async def navigate(self, url: str) -> dict:
        """Navigate the persistent page to a URL."""
        self._validate_url(url)  # SSRF protection
        async with self._lock:
            page = await self._get_page_internal()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await page.wait_for_timeout(500)
                self._url = page.url
                self._title = await page.title()
                await emit("browser_action", action="open", url=self._url, title=self._title, source="live_viewer")
                return {"url": self._url, "title": self._title, "status": "ok"}
            except Exception as e:
                await emit("browser_action", action="error", error=str(e), url=url, source="live_viewer")
                return {"url": self._url, "title": self._title, "status": "error", "error": str(e)}

    async def click(self, x: float, y: float, button: str = "left") -> dict:
        """Click at viewport coordinates on the persistent page."""
        async with self._lock:
            page = await self._get_page_internal()
            try:
                if button == "right":
                    await page.mouse.click(x, y, button="right")
                elif button == "middle":
                    await page.mouse.click(x, y, button="middle")
                else:
                    await page.mouse.click(x, y)
                self._url = page.url
                self._title = await page.title()
                await emit("browser_action", action="click", url=self._url, x=x, y=y, button=button, source="live_viewer")
                return {"status": "ok", "url": self._url}
            except Exception as e:
                return {"status": "error", "error": str(e)}

    async def type_text(self, text: str) -> dict:
        """Type text into the currently focused element on the persistent page."""
        async with self._lock:
            page = await self._get_page_internal()
            try:
                await page.keyboard.type(text, delay=20)
                self._url = page.url
                self._title = await page.title()
                return {"status": "ok"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

    async def press_key(self, key: str) -> dict:
        """Press a special key on the persistent page."""
        async with self._lock:
            page = await self._get_page_internal()
            try:
                await page.keyboard.press(key)
                self._url = page.url
                self._title = await page.title()
                return {"status": "ok"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

    async def scroll(self, x: float = 0, y: float = 300, delta_x: float = 0, delta_y: float = 0) -> dict:
        """Scroll the persistent page."""
        async with self._lock:
            page = await self._get_page_internal()
            try:
                if delta_x != 0 or delta_y != 0:
                    await page.mouse.wheel(delta_x, delta_y)
                else:
                    # Parameterized evaluation to prevent JS injection
                    await page.evaluate("([sx, sy]) => window.scrollBy(sx, sy)", [x, y])
                return {"status": "ok"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

    async def mouse_move(self, x: float, y: float) -> dict:
        """Move the mouse to viewport coordinates on the persistent page."""
        async with self._lock:
            page = await self._get_page_internal()
            try:
                await page.mouse.move(x, y)
                return {"status": "ok"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

    async def _get_page_internal(self) -> Any:
        """Get the page without acquiring the lock. Caller MUST hold self._lock."""
        if self._page and not self._page.is_closed():
            return self._page
        # Close old context if it exists (prevent context leak on page recreation)
        if self._ctx:
            try:
                await self._ctx.close()
            except Exception:
                pass
            self._ctx = None
        # Lazily create page (still under caller's lock)
        browser = await self.get_browser()
        self._ctx = await browser.new_context(
            viewport=self._viewport,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self._page = await self._ctx.new_page()
        await self._page.goto("about:blank")
        self._url = "about:blank"
        self._title = ""
        self._is_open = True
        return self._page

    async def screenshot_jpeg(self, quality: int = 75) -> bytes | None:
        """Capture a JPEG screenshot of the persistent page. No lock held to avoid blocking stream."""
        try:
            # Access page directly (no lock) — Playwright handles concurrency internally
            page = self._page
            if not page or page.is_closed() or page.url == "about:blank":
                return None
            shot = await page.screenshot(type="jpeg", quality=quality)
            return shot if shot else None
        except Exception as e:
            logger.debug("[browser-manager] Screenshot error: %s", e)
            return None

    async def screenshot_png(self) -> bytes | None:
        """Capture a PNG screenshot of the persistent page. No lock held to avoid blocking stream."""
        try:
            page = self._page
            if not page or page.is_closed() or page.url == "about:blank":
                return None
            shot = await page.screenshot(type="png")
            return shot if shot else None
        except Exception as e:
            logger.debug("[browser-manager] Screenshot error: %s", e)
            return None

    async def status(self) -> dict:
        """Get current browser status."""
        is_open = self._page is not None and not self._page.is_closed()
        return {
            "available": _PLAYWRIGHT_AVAILABLE,
            "open": is_open,
            "url": self._url if is_open else "",
            "title": self._title if is_open else "",
            "viewport": self._viewport,
        }

    async def close_page(self) -> dict:
        """Close the persistent page."""
        async with self._lock:
            if self._page and not self._page.is_closed():
                try:
                    await self._page.close()
                except Exception:
                    pass
            self._page = None
            self._url = ""
            self._title = ""
            self._is_open = False
            await emit("browser_action", action="close", source="live_viewer")
            return {"status": "ok"}

    async def cleanup(self):
        """Cleanup all resources."""
        await self.close_page()
        if self._ctx:
            try:
                await self._ctx.close()
            except Exception:
                pass
            self._ctx = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None


# Global singleton
_manager: BrowserManager | None = None


def get_manager() -> BrowserManager:
    global _manager
    if _manager is None:
        _manager = BrowserManager()
    return _manager


async def mirror_browser_action(event):
    """
    EventBus subscriber: mirror agent browser actions to the persistent page.

    When the agent uses browser_open, browser_screenshot, browser_click, etc.,
    this handler navigates the persistent page so the live stream stays in sync.
    """
    mgr = get_manager()

    kind = getattr(event, 'kind', '')
    data = getattr(event, 'data', {})

    if kind != 'browser_action':
        return

    action = data.get('action', '')
    url = data.get('url', '')
    source = data.get('source', '')

    # Don't mirror our own events
    if source == 'live_viewer':
        return

    # Mirror navigation
    if action in ('open', 'screenshot', 'click', 'fill_form') and url:
        try:
            page = await mgr.get_page()
            current_url = page.url
            # Only navigate if it's a different URL or page is blank
            if url and current_url != url:
                await mgr.navigate(url)
        except Exception as e:
            logger.debug("[browser-mirror] Failed to mirror navigation: %s", e)


# ── Request models ────────────────────────────────────────────────────────

class NavigateRequest(BaseModel):
    url: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_url_field

    @classmethod
    def validate_url_field(cls, v):
        BrowserManager._validate_url(v)
        return v

class ClickRequest(BaseModel):
    x: float
    y: float
    button: str = "left"

class TypeRequest(BaseModel):
    text: str

class KeyRequest(BaseModel):
    key: str  # e.g. "Enter", "Tab", "Backspace", "ArrowDown", "ctrl+a"

class ScrollRequest(BaseModel):
    x: float = 0
    y: float = 300
    delta_x: float = 0
    delta_y: float = 0

class MouseMoveRequest(BaseModel):
    x: float
    y: float


# ══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def browser_status(auth: dict = Depends(require_auth)):
    """Get current browser live status."""
    mgr = get_manager()
    return await mgr.status()


@router.post("/navigate")
async def browser_navigate(req: NavigateRequest, auth: dict = Depends(require_auth)):
    """Navigate the live browser to a URL."""
    mgr = get_manager()
    return await mgr.navigate(req.url)


@router.post("/click")
async def browser_click(req: ClickRequest, auth: dict = Depends(require_auth)):
    """Click at coordinates on the live browser page."""
    mgr = get_manager()
    return await mgr.click(req.x, req.y, req.button)


@router.post("/type")
async def browser_type(req: TypeRequest, auth: dict = Depends(require_auth)):
    """Type text into the focused element on the live browser page."""
    mgr = get_manager()
    return await mgr.type_text(req.text)


@router.post("/keyboard")
async def browser_keyboard(req: KeyRequest, auth: dict = Depends(require_auth)):
    """Press a special key on the live browser page."""
    mgr = get_manager()
    return await mgr.press_key(req.key)


@router.post("/scroll")
async def browser_scroll(req: ScrollRequest, auth: dict = Depends(require_auth)):
    """Scroll the live browser page."""
    mgr = get_manager()
    return await mgr.scroll(req.x, req.y, req.delta_x, req.delta_y)


@router.post("/mouse-move")
async def browser_mouse_move(req: MouseMoveRequest, auth: dict = Depends(require_auth)):
    """Move mouse to coordinates on the live browser page."""
    mgr = get_manager()
    return await mgr.mouse_move(req.x, req.y)


@router.post("/close")
async def browser_close(auth: dict = Depends(require_auth)):
    """Close the live browser page."""
    mgr = get_manager()
    return await mgr.close_page()


@router.get("/screenshot")
async def browser_screenshot_endpoint(auth: dict = Depends(require_auth)):
    """Capture a single JPEG screenshot and return as base64 data URI."""
    mgr = get_manager()
    shot = await mgr.screenshot_jpeg()
    if not shot:
        return {"image": None, "error": "Browser is not open or page is blank"}
    b64 = base64.b64encode(shot).decode("utf-8")
    return {"image": f"data:image/jpeg;base64,{b64}", "size_kb": len(shot) / 1024}


# ════════════════════════════════════════════════════════════════════════════
# WebSocket: Browser Screenshot Stream
# ════════════════════════════════════════════════════════════════════════════

@router.websocket("/stream")
async def browser_stream_ws(websocket: WebSocket):
    """WebSocket endpoint that streams JPEG screenshots from the browser at ~8 fps.

    NOTE: The router prefix is "/api/browser", so the full path is /api/browser/stream.
    The frontend connects to exactly this path.

    Protocol:
    - Server sends binary frames (JPEG bytes) continuously
    - First message: JSON {"type": "info", ...} with browser metadata
    - Client can send JSON messages for input:
      {"type": "click", "x": 100, "y": 200, "button": "left"}
      {"type": "type", "text": "hello"}
      {"type": "keyboard", "key": "Enter"}
      {"type": "navigate", "url": "https://example.com"}
      {"type": "scroll", "delta_x": 0, "delta_y": 300}
      {"type": "mouse_move", "x": 100, "y": 200}
    - Server responds to input messages with {"type": "input_result", "success": true}
    - Server sends {"type": "page_info", "url": "...", "title": "..."} on navigation
    """
    # Auth: enforce token validation when auth is enabled
    auth_enabled = config.get("enable_auth", False)
    token = websocket.query_params.get("token")
    if auth_enabled:
        if not token:
            await websocket.close(code=4001, reason="Authentication required")
            return
        try:
            from ..auth import verify_token
            payload = verify_token(token)
            if not payload:
                await websocket.close(code=4001, reason="Invalid token")
                return
        except Exception:
            await websocket.close(code=4001, reason="Authentication failed")
            return
    elif token:
        # Auth disabled but token provided — validate anyway for consistency
        try:
            from ..auth import verify_token
            verify_token(token)
        except Exception:
            pass
    await websocket.accept()

    mgr = get_manager()

    # Send initial info
    status = await mgr.status()
    try:
        info = json.dumps({
            "type": "info",
            "available": status.get("available", False),
            "open": status.get("open", False),
            "url": status.get("url", ""),
            "title": status.get("title", ""),
            "viewport": status.get("viewport", {"width": 1280, "height": 720}),
            "timestamp": time.time(),
        })
        await websocket.send_text(info)
    except Exception:
        pass

    if not _PLAYWRIGHT_AVAILABLE:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Playwright not installed. Run: pip install playwright && playwright install chromium"
            }))
        except Exception:
            pass
        return

    logger.info("[browser-stream] Client connected, starting screenshot stream")

    # Track current URL to detect navigation changes from the manager
    last_url = status.get("url", "")
    last_title = status.get("title", "")

    # Stream loop
    frame_interval = 0.08  # ~12 fps (faster streaming)
    consecutive_errors = 0
    max_errors = 20

    # For receiving client input messages concurrently with streaming
    recv_task = None

    async def receive_input():
        """Receive and process client input messages."""
        nonlocal last_url, last_title
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    msg_type = msg.get("type", "")

                    result = {"type": "input_result", "success": False, "action": msg_type}

                    if msg_type == "click":
                        r = await mgr.click(msg.get("x", 0), msg.get("y", 0), msg.get("button", "left"))
                        result.update({"success": r.get("status") == "ok", "url": r.get("url", "")})
                        last_url = r.get("url", last_url)
                    elif msg_type == "type":
                        r = await mgr.type_text(msg.get("text", ""))
                        result["success"] = r.get("status") == "ok"
                    elif msg_type == "keyboard":
                        r = await mgr.press_key(msg.get("key", ""))
                        result["success"] = r.get("status") == "ok"
                    elif msg_type == "navigate":
                        r = await mgr.navigate(msg.get("url", ""))
                        result.update({"success": r.get("status") == "ok", "url": r.get("url", ""), "title": r.get("title", "")})
                        last_url = r.get("url", last_url)
                        last_title = r.get("title", last_title)
                    elif msg_type == "scroll":
                        r = await mgr.scroll(
                            x=msg.get("x", 0), y=msg.get("y", 300),
                            delta_x=msg.get("delta_x", 0), delta_y=msg.get("delta_y", 0)
                        )
                        result["success"] = r.get("status") == "ok"
                    elif msg_type == "mouse_move":
                        r = await mgr.mouse_move(msg.get("x", 0), msg.get("y", 0))
                        result["success"] = r.get("status") == "ok"
                    else:
                        result["message"] = f"Unknown input type: {msg_type}"

                    try:
                        await websocket.send_text(json.dumps(result))
                    except Exception:
                        break

                except json.JSONDecodeError:
                    pass

        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("[browser-stream] Receive error: %s", e)

    try:
        recv_task = asyncio.create_task(receive_input())

        while True:
            try:
                # Capture screenshot
                jpeg_bytes = await mgr.screenshot_jpeg(quality=55)  # Lower quality = faster encoding
                if jpeg_bytes:
                    await websocket.send_bytes(jpeg_bytes)
                    consecutive_errors = 0

                    # Check if page URL/title changed (from agent mirroring)
                    current_status = await mgr.status()
                    current_url = current_status.get("url", "")
                    current_title = current_status.get("title", "")
                    if current_url != last_url or current_title != last_title:
                        last_url = current_url
                        last_title = current_title
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "page_info",
                                "url": current_url,
                                "title": current_title,
                            }))
                        except Exception:
                            break
                else:
                    # No screenshot (page might be blank)
                    consecutive_errors += 1
                    if consecutive_errors < 5:
                        await asyncio.sleep(0.1)
                        continue

                await asyncio.sleep(frame_interval)

            except WebSocketDisconnect:
                logger.info("[browser-stream] Client disconnected")
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("[browser-stream] Frame error: %s", exc)
                await asyncio.sleep(0.2)
                if consecutive_errors > max_errors:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error", "message": "Internal stream error. Please reconnect."
                        }))
                    except Exception:
                        pass
                    break
    except Exception as exc:
        logger.error("[browser-stream] Fatal error: %s", exc)
    finally:
        if recv_task:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass
        logger.info("[browser-stream] Connection closed")


# ── EventBus subscription for mirroring agent browser actions ──────────

_mirror_task: asyncio.Task | None = None
_mirror_queue: asyncio.Queue | None = None


async def start_event_mirror():
    """Subscribe to EventBus and mirror agent browser actions to the persistent page."""
    global _mirror_task, _mirror_queue
    _mirror_queue = BUS.subscribe()
    _mirror_task = asyncio.create_task(_event_mirror_loop(_mirror_queue))


async def stop_event_mirror():
    """Cancel event mirror task and unsubscribe from EventBus."""
    global _mirror_task, _mirror_queue
    if _mirror_task:
        _mirror_task.cancel()
        try:
            await _mirror_task
        except asyncio.CancelledError:
            pass
        _mirror_task = None
    if _mirror_queue:
        BUS.unsubscribe(_mirror_queue)
        _mirror_queue = None


async def _event_mirror_loop(q: asyncio.Queue):
    """Process events from the bus and mirror browser actions."""
    while True:
        try:
            event = await q.get()
            await mirror_browser_action(event)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug("[browser-mirror] Event processing error: %s", e)
