"""
VM Routes — REST API + WebSocket screenshot streaming + VNC WebSocket proxy.

Provides:
- REST endpoints for VM lifecycle control and input forwarding
- /ws/vm-stream  — WebSocket endpoint that streams JPEG screenshots at ~10 fps
- /ws/vnc        — WebSocket-to-TCP proxy for full VNC interactive mode

v22: New module extracted from server.py for clean separation.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import config
from pathlib import Path
from ..virtual_computer.container import get_vm, VNC_PORT, _run_raw, _docker, _run, SCREENSHOT_PATH, XVFB_DISPLAY
from ..events import emit
from ..auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vm", tags=["virtual-computer"])


# ── Request models ────────────────────────────────────────────────────────

class MouseInput(BaseModel):
    action: str  # "move" | "click"
    x: int | None = None
    y: int | None = None
    button: str = "left"


class KeyboardInput(BaseModel):
    action: str  # "type" | "press"
    text: str | None = None
    key: str | None = None


class VMCommand(BaseModel):
    command: str
    timeout: int = 30


# ══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def vm_status(auth: dict = Depends(require_auth)):
    """Get current VM status."""
    vm = get_vm()
    return await vm.status()


@router.post("/start")
async def vm_start(auth: dict = Depends(require_auth)):
    """Start the virtual computer."""
    vm = get_vm()
    await emit("vm_action", action="start", status="starting")
    result = await vm.start()
    if result.get("status") == "error":
        await emit("vm_action", action="start", status="error", message=result.get("message", ""))
        return result
    await emit("vm_action", action="start", status="running", message="VM started")
    return result


@router.post("/stop")
async def vm_stop(auth: dict = Depends(require_auth)):
    """Stop the virtual computer."""
    vm = get_vm()
    result = await vm.stop()
    await emit("vm_action", action="stop", status="stopped", message="VM stopped")
    return result


@router.post("/restart")
async def vm_restart(auth: dict = Depends(require_auth)):
    """Restart the virtual computer."""
    vm = get_vm()
    await emit("vm_action", action="restart", status="restarting")
    result = await vm.restart()
    await emit("vm_action", action="restart", status="running", message="VM restarted")
    return result


@router.post("/destroy")
async def vm_destroy(auth: dict = Depends(require_auth)):
    """Destroy the virtual computer container."""
    vm = get_vm()
    result = await vm.destroy()
    await emit("vm_action", action="destroy", status="destroyed", message="VM destroyed")
    return result


@router.get("/screenshot")
async def vm_screenshot(auth: dict = Depends(require_auth)):
    """Capture a PNG screenshot and return base64-encoded image."""
    vm = get_vm()
    status = await vm.status()
    if status.get("status") != "running":
        return {"image": None, "error": "VM is not running", "status": status.get("status")}

    import base64
    png_bytes = await vm.screenshot()
    if not png_bytes:
        return {"image": None, "error": "Failed to capture screenshot"}

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    await emit("vm_screenshot", size_bytes=len(png_bytes))
    return {"image": f"data:image/png;base64,{b64}", "size_kb": len(png_bytes) / 1024}


@router.post("/mouse")
async def vm_mouse(input: MouseInput, auth: dict = Depends(require_auth)):
    """Forward mouse action to the VM."""
    vm = get_vm()

    if input.action == "click":
        x, y = input.x, input.y
        if x is not None and y is not None:
            result = await vm.mouse_click(x=x, y=y, button={"left": 1, "middle": 2, "right": 3}.get(input.button, 1))
        else:
            result = await vm.mouse_click(button={"left": 1, "middle": 2, "right": 3}.get(input.button, 1))
    elif input.action == "move":
        if input.x is not None and input.y is not None:
            result = await vm.mouse_move(x=input.x, y=input.y)
        else:
            return {"error": "x and y are required for move"}
    else:
        return {"error": f"Unknown mouse action: {input.action}"}

    await emit("vm_action", action="mouse", mouse_input=input.model_dump())
    return {"result": result}


@router.post("/keyboard")
async def vm_keyboard(input: KeyboardInput, auth: dict = Depends(require_auth)):
    """Forward keyboard action to the VM."""
    vm = get_vm()

    if input.action == "type":
        if not input.text:
            return {"error": "text is required for type action"}
        result = await vm.type_text(input.text)
    elif input.action == "press":
        if not input.key:
            return {"error": "key is required for press action"}
        result = await vm.press_key(input.key)
    else:
        return {"error": f"Unknown keyboard action: {input.action}"}

    await emit("vm_action", action="keyboard", keyboard_input=input.model_dump())
    return {"result": result}


@router.post("/execute")
async def vm_execute(cmd: VMCommand, auth: dict = Depends(require_auth)):
    """Execute a shell command inside the VM container."""
    vm = get_vm()
    output = await vm.exec_command(cmd.command, timeout=cmd.timeout)
    await emit("vm_action", action="execute", command=cmd.command[:200])
    return {"output": output}


@router.post("/upload")
async def vm_upload(
    local_path: str = Body(...),
    container_path: str = Body(...),
    auth: dict = Depends(require_auth),
):
    """Upload a file to the VM container."""
    vm = get_vm()
    result = await vm.upload_file(local_path, container_path)
    return {"result": result}


@router.post("/download")
async def vm_download(
    container_path: str = Body(...),
    local_path: str = Body(...),
    auth: dict = Depends(require_auth),
):
    """Download a file from the VM container."""
    vm = get_vm()
    result = await vm.download_file(container_path, local_path)
    return {"result": result}


@router.post("/install-package")
async def vm_install(packages: str, auth: dict = Depends(require_auth)):
    """Install apt packages inside the VM."""
    vm = get_vm()
    pkg_list = [p.strip() for p in packages.split(",") if p.strip()]
    if not pkg_list:
        return {"error": "No packages specified"}
    result = await vm.install_package(pkg_list)
    return {"result": result}


@router.post("/vision-loop")
async def vm_vision_loop(
    task: str,
    max_iterations: int = 20,
    auth: dict = Depends(require_auth),
):
    """Start a vision loop for autonomous GUI interaction."""
    await emit("vm_vision_loop_started", task=task, max_iterations=max_iterations)
    # Vision loop runs as a background task in the agent
    # The actual implementation is in the coordinator/executor
    return {"status": "started", "task": task, "max_iterations": max_iterations}


# ════════════════════════════════════════════════════════════════════════════
# WebSocket: Screenshot Stream (Lightweight, no VNC dependency)
# ════════════════════════════════════════════════════════════════════════════

@router.websocket("/ws/vm-stream")
async def vm_stream_ws(websocket: WebSocket):
    """WebSocket endpoint that streams JPEG screenshots from the VM at ~8 fps.

    Protocol:
    - Server sends binary frames (JPEG bytes) continuously
    - First message: JSON {"type": "info", ...} with VM metadata
    - Client can send JSON messages for input:
      {"type": "mouse", "x": 100, "y": 200, "button": "left", "action": "click"}
      {"type": "keyboard", "action": "type", "text": "hello"}
      {"type": "keyboard", "action": "press", "key": "Return"}
    - Server responds to input messages with {"type": "input_result", "success": True}
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

    vm = get_vm()

    # Send initial info
    status = await vm.status()
    try:
        info = json.dumps({
            "type": "info",
            "status": status.get("status", "unknown"),
            "resolution": "1280x720",
            "vnc_port": VNC_PORT,
            "timestamp": time.time(),
        })
        await websocket.send_text(info)
    except Exception:
        pass

    # Check if VM is running
    if not status.get("running"):
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "VM is not running"}))
        except Exception:
            pass
        # Wait for VM to start (with timeout)
        waited = 0
        while waited < 30:
            await asyncio.sleep(1)
            status = await vm.status()
            if status.get("running"):
                break
            waited += 1
        if not status.get("running"):
            try:
                await websocket.send_text(json.dumps({"type": "error", "message": "VM failed to start within 30s"}))
            except Exception:
                pass
            return

    logger.info("[vm-stream] streaming...")

    # ── Input receiver task (handles mouse/keyboard from client via WS) ──
    async def receive_vm_input():
        """Receive and process client input messages via WebSocket."""
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    msg_type = msg.get("type", "")
                    result = {"type": "input_result", "success": False, "action": msg_type}

                    if msg_type == "mouse":
                        action = msg.get("action", "click")
                        x, y = msg.get("x"), msg.get("y")
                        button_str = msg.get("button", "left")
                        button = {"left": 1, "middle": 2, "right": 3}.get(button_str, 1)
                        if action == "click" and x is not None and y is not None:
                            r = await vm.mouse_click(x=int(x), y=int(y), button=button)
                            result.update({"success": True})
                        elif action == "move" and x is not None and y is not None:
                            r = await vm.mouse_move(x=int(x), y=int(y))
                            result.update({"success": True})
                        else:
                            result["message"] = f"Unknown mouse action or missing coords: {action}"

                    elif msg_type == "keyboard":
                        kb_action = msg.get("action", "")
                        if kb_action == "type" and msg.get("text"):
                            r = await vm.type_text(msg["text"])
                            result.update({"success": True})
                        elif kb_action == "press" and msg.get("key"):
                            r = await vm.press_key(msg["key"])
                            result.update({"success": True})
                        else:
                            result["message"] = f"Unknown keyboard action: {kb_action}"

                    else:
                        result["message"] = f"Unknown input type: {msg_type}"

                    try:
                        await websocket.send_text(json.dumps(result))
                    except Exception:
                        break

                except json.JSONDecodeError:
                    pass
        except Exception:
            pass  # WebSocket closed or cancelled

    recv_task = asyncio.create_task(receive_vm_input())

    try:
        # Send initial info with FPS target
        try:
            await websocket.send_text(json.dumps({"type": "info", "fps_target": 5, "status": "running"}))
        except Exception:
            pass

        consecutive_errors = 0
        max_errors = 20

        while True:
            try:
                if websocket.client_state.name != "CONNECTED":
                    break
            except Exception:
                break

            try:
                jpeg_bytes = b""
                if vm._use_docker:
                    # Capture JPEG via Docker exec (much smaller than PNG)
                    jpeg_bytes, stderr, rc = await _run_raw(
                        _docker(
                            "exec", vm.container_name,
                            "/bin/bash", "-c",
                            "DISPLAY=:99 import -window root -quality 65 "
                            "-sampling-factor 4x2,4x2,1 -interlace Plane "
                            "jpeg:/tmp/vm_stream.jpg 2>/dev/null && cat /tmp/vm_stream.jpg",
                        ),
                        timeout=5,
                    )
                else:
                    # Subprocess mode: capture JPEG using host ImageMagick
                    import_path = SCREENSHOT_PATH.replace(".png", ".jpg")
                    stdout, stderr, rc = await _run(
                        ["env", f"DISPLAY={XVFB_DISPLAY}",
                         "import", "-window", "root", "-quality", "65",
                         f"jpeg:{import_path}"],
                        timeout=15, check=False,
                    )
                    if rc == 0:
                        jpeg_bytes = Path(import_path).read_bytes() if Path(import_path).exists() else b""

                if rc == 0 and jpeg_bytes and len(jpeg_bytes) > 500:
                    await websocket.send_bytes(jpeg_bytes)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        try:
                            png_bytes = await vm.screenshot()
                            if png_bytes and len(png_bytes) > 500:
                                await websocket.send_bytes(png_bytes)
                                consecutive_errors = 0
                        except Exception:
                            pass

                    if consecutive_errors > max_errors:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "error", "message": "Too many capture failures. Please reconnect."
                            }))
                        except Exception:
                            pass
                        break

            except Exception as e:
                consecutive_errors += 1
                logger.debug("[vm-stream] Capture error: %s", e)
                if consecutive_errors > max_errors:
                    break

            await asyncio.sleep(0.15)  # ~6-7 fps with JPEG

    except Exception as e:
        logger.warning("[vm-stream] %s", e)
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        logger.info("[vm-stream] done")


# ════════════════════════════════════════════════════════════════════════════
# WebSocket: VNC Proxy (Full interactive VNC via noVNC client)
# ════════════════════════════════════════════════════════════════════════════

@router.websocket("/ws/vnc")
async def vnc_proxy_ws(websocket: WebSocket):
    """WebSocket-to-TCP proxy for full VNC interactive mode.

    Bridges the browser's WebSocket connection directly to the VNC server
    running inside the Docker container on port 5900. This enables using
    the noVNC client library for full interactive VNC (pixel-perfect,
    low-latency, mouse/keyboard built-in).

    Note: The VNC server (x11vnc) runs with -nopw -shared, so no authentication
    is required for the VNC connection itself.
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

    vm = get_vm()

    # Verify VM is running
    status = await vm.status()
    if not status.get("running"):
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "VM is not running"}))
        except Exception:
            pass
        return

    # Connect to VNC server via TCP
    vnc_host = "127.0.0.1"
    vnc_port = 5900

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(vnc_host, vnc_port), timeout=5
        )
    except Exception as exc:
        logger.error("[vnc-proxy] Failed to connect to VNC: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Cannot connect to VNC: {exc}"}))
        except Exception:
            pass
        return

    logger.info("[vnc-proxy] VNC TCP connection established to %s:%d", vnc_host, vnc_port)

    async def ws_to_vnc():
        """Forward WebSocket frames to VNC TCP socket."""
        try:
            async for data in websocket.iter_bytes():
                writer.write(data)
                await writer.drain()
        except Exception:
            pass

    async def vnc_to_ws():
        """Forward VNC TCP data to WebSocket."""
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                await websocket.send_bytes(data)
        except Exception:
            pass

    try:
        await asyncio.gather(ws_to_vnc(), vnc_to_ws())
    except Exception as exc:
        logger.debug("[vnc-proxy] Proxy ended: %s", exc)
    finally:
        logger.info("[vnc-proxy] Connection closed")
        try:
            writer.close()
        except Exception:
            pass
