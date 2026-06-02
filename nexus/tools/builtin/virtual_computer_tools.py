"""
Virtual Computer Tools — Docker-based headless desktop for NEXUS.

Provides 14 tools for managing and interacting with a virtual computer
environment powered by Docker, Xvfb, x11vnc, and xdotool.

Container: nexus-vm (ubuntu:22.04, 2GB RAM, 1280x720 display)
VNC: port 5900 (no password)
"""
from __future__ import annotations

import base64
import json

from ..registry import tool
from ...events import emit

from ...virtual_computer.container import get_vm


_PREVIEW_B64_LEN = 200


async def _vm_running() -> str | None:
    """Return None if VM is running, or an error string if not."""
    try:
        status = await get_vm().status()
        if status.get("status") != "running":
            return (
                f"Virtual computer is not running (status: {status.get('status')}). "
                f"Use vm_start first."
            )
        return None
    except Exception as e:
        return f"Virtual computer unavailable: {e}"


async def _vm_result(coro, *, emit_event: str = "", **emit_kw) -> str:
    """Execute a VM coroutine with standard error wrapping."""
    err = await _vm_running()
    if err:
        # Close the un-awaited coroutine to prevent "coroutine was never awaited" warning
        if hasattr(coro, 'close'):
            coro.close()
        return err
    try:
        await emit("vm_action", action=emit_event, **emit_kw)
        return await coro
    except Exception as e:
        return f"[VM Error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 1. vm_start — Start the virtual computer
# ---------------------------------------------------------------------------

@tool(
    name="vm_start",
    description=(
        "Start the virtual computer (Docker container with Ubuntu desktop). "
        "Creates the container if it doesn't exist, or resumes it if stopped. "
        "Provides a headless GUI environment with Xvfb (1280x720), x11vnc (port 5900), "
        "and xdotool for mouse/keyboard input. 2GB RAM, workspace bind-mounted."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="virtual_computer",
    risk="medium",
    timeout=120,
)
async def vm_start(params: dict) -> str:
    try:
        vm = get_vm()
        await emit("vm_action", action="start")
        result = await vm.start()
        if result.get("status") == "error":
            return f"Error: {result.get('message', 'Unknown error')}"
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"[VM Error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 2. vm_stop — Stop the virtual computer
# ---------------------------------------------------------------------------

@tool(
    name="vm_stop",
    description=(
        "Stop the virtual computer container. The container is preserved and can be "
        "restarted with vm_start. No data inside the container is lost."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="virtual_computer",
    risk="low",
    timeout=60,
)
async def vm_stop(params: dict) -> str:
    try:
        vm = get_vm()
        await emit("vm_action", action="stop")
        result = await vm.stop()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[VM Error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 3. vm_restart — Restart the virtual computer
# ---------------------------------------------------------------------------

@tool(
    name="vm_restart",
    description=(
        "Restart the virtual computer container. Re-ensures all display services "
        "(Xvfb, x11vnc) are running after restart."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="virtual_computer",
    risk="medium",
    timeout=90,
)
async def vm_restart(params: dict) -> str:
    try:
        vm = get_vm()
        await emit("vm_action", action="restart")
        result = await vm.restart()
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"[VM Error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 4. vm_destroy — Destroy and remove the container
# ---------------------------------------------------------------------------

@tool(
    name="vm_destroy",
    description=(
        "Permanently destroy and remove the virtual computer container. "
        "All data INSIDE the container is lost. The workspace directory on the host "
        "is preserved. Use vm_start to create a fresh container afterward."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="virtual_computer",
    risk="high",
    timeout=60,
)
async def vm_destroy(params: dict) -> str:
    try:
        vm = get_vm()
        await emit("vm_action", action="destroy")
        result = await vm.destroy()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[VM Error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 5. vm_status — Get container status
# ---------------------------------------------------------------------------

@tool(
    name="vm_status",
    description=(
        "Get the current status of the virtual computer. Returns information about "
        "whether the container is running, its IP address, uptime, display settings, "
        "and whether required services (Xvfb, x11vnc, xdotool) are available."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="virtual_computer",
    risk="low",
    timeout=15,
)
async def vm_status(params: dict) -> str:
    try:
        vm = get_vm()
        result = await vm.status()
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"[VM Error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 6. vm_screenshot — Capture the virtual display
# ---------------------------------------------------------------------------

@tool(
    name="vm_screenshot",
    description=(
        "Take a screenshot of the virtual computer's display (1280x720). "
        "Returns a base64-encoded PNG image that can be viewed or analyzed. "
        "The virtual computer must be running (use vm_start first)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="virtual_computer",
    risk="low",
    timeout=30,
)
async def vm_screenshot(params: dict) -> str:
    err = await _vm_running()
    if err:
        return err

    try:
        png_bytes = await get_vm().screenshot()
        if not png_bytes:
            return "Error: Failed to capture screenshot. The display may not be ready yet."

        b64 = base64.b64encode(png_bytes).decode("utf-8")
        size_kb = len(png_bytes) / 1024

        await emit(
            "vm_screenshot",
            size_bytes=len(png_bytes),
            image=f"data:image/png;base64,{b64}",
        )

        return (
            f"Screenshot captured ({size_kb:.1f} KB, 1280x720).\n"
            f"Base64 PNG preview: {b64[:_PREVIEW_B64_LEN]}..."
        )
    except Exception as e:
        return f"[VM Error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 7. vm_mouse_move — Move mouse pointer
# ---------------------------------------------------------------------------

@tool(
    name="vm_mouse_move",
    description=(
        "Move the mouse pointer to specific coordinates on the virtual display. "
        "The display is 1280x720 pixels. Coordinates (0,0) is top-left."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "description": "X coordinate (0-1279)",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate (0-719)",
            },
        },
        "required": ["x", "y"],
    },
    category="virtual_computer",
    risk="low",
    timeout=10,
)
async def vm_mouse_move(params: dict) -> str:
    x = max(0, min(1279, int(params.get("x", 0))))
    y = max(0, min(719, int(params.get("y", 0))))
    return await _vm_result(
        get_vm().mouse_move(x, y),
        emit_event="mouse_move", x=x, y=y,
    )


# ---------------------------------------------------------------------------
# 8. vm_mouse_click — Click mouse button
# ---------------------------------------------------------------------------

@tool(
    name="vm_mouse_click",
    description=(
        "Click a mouse button on the virtual display. Optionally move to coordinates "
        "before clicking. Default is left click at current position."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "description": "X coordinate to move to before clicking (optional)",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate to move to before clicking (optional)",
            },
            "button": {
                "type": "string",
                "enum": ["left", "middle", "right"],
                "default": "left",
                "description": "Mouse button to click",
            },
        },
        "required": [],
    },
    category="virtual_computer",
    risk="low",
    timeout=10,
)
async def vm_mouse_click(params: dict) -> str:
    button_map = {"left": 1, "middle": 2, "right": 3}
    button_str = params.get("button", "left")
    button = button_map.get(button_str, 1)

    x = params.get("x")
    y = params.get("y")
    if x is not None:
        x = max(0, min(1279, int(x)))
    if y is not None:
        y = max(0, min(719, int(y)))

    return await _vm_result(
        get_vm().mouse_click(x=x, y=y, button=button),
        emit_event="mouse_click", x=x, y=y, button=button_str,
    )


# ---------------------------------------------------------------------------
# 9. vm_type_text — Type text
# ---------------------------------------------------------------------------

@tool(
    name="vm_type_text",
    description=(
        "Type text into the virtual computer using the keyboard. The text is typed "
        "character by character at the current cursor position. Use vm_press_key for "
        "special keys (Enter, Tab, etc.)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to type",
            },
        },
        "required": ["text"],
    },
    category="virtual_computer",
    risk="low",
    timeout=15,
)
async def vm_type_text(params: dict) -> str:
    text = params.get("text", "")
    if not text:
        return "Error: No text provided."
    return await _vm_result(
        get_vm().type_text(text),
        emit_event="type_text", text=text[:100],
    )


# ---------------------------------------------------------------------------
# 10. vm_press_key — Press a key or key combination
# ---------------------------------------------------------------------------

@tool(
    name="vm_press_key",
    description=(
        "Press a key or key combination on the virtual keyboard. "
        "Examples: 'Return', 'Tab', 'Escape', 'ctrl+c', 'ctrl+v', 'alt+Tab', "
        "'ctrl+alt+t' (open terminal), 'super' (Windows/Meta key), 'F5'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Key name or combination. Examples: 'Return', 'Tab', 'Escape', "
                    "'ctrl+c', 'ctrl+v', 'alt+Tab', 'ctrl+alt+t', 'super', 'F1-F12'"
                ),
            },
        },
        "required": ["key"],
    },
    category="virtual_computer",
    risk="low",
    timeout=10,
)
async def vm_press_key(params: dict) -> str:
    key = params.get("key", "")
    if not key:
        return "Error: No key specified."
    return await _vm_result(
        get_vm().press_key(key),
        emit_event="press_key", key=key,
    )


# ---------------------------------------------------------------------------
# 11. vm_execute — Execute a shell command
# ---------------------------------------------------------------------------

@tool(
    name="vm_execute",
    description=(
        "Execute a shell command inside the virtual computer container. "
        "Returns stdout and stderr. Useful for installing software, running scripts, "
        "managing files, etc. Runs as the default container user."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute inside the container",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
                "default": 30,
            },
        },
        "required": ["command"],
    },
    category="virtual_computer",
    risk="high",
    timeout=120,
)
async def vm_execute(params: dict) -> str:
    cmd = params.get("command", "")
    if not cmd.strip():
        return "Error: No command provided."
    timeout = int(params.get("timeout", 30))
    return await _vm_result(
        get_vm().exec_command(cmd, timeout=timeout),
        emit_event="execute", command=cmd[:200],
    )


# ---------------------------------------------------------------------------
# 12. vm_upload — Upload a file to the container
# ---------------------------------------------------------------------------

@tool(
    name="vm_upload",
    description=(
        "Upload a file from the host workspace into the virtual computer container. "
        "The local_path is relative to the workspace directory. "
        "The container_path is the absolute destination path inside the container."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "local_path": {
                "type": "string",
                "description": "Path relative to the workspace directory on the host",
            },
            "container_path": {
                "type": "string",
                "description": "Absolute destination path inside the container",
            },
        },
        "required": ["local_path", "container_path"],
    },
    category="virtual_computer",
    risk="low",
    timeout=30,
)
async def vm_upload(params: dict) -> str:
    local_path = params.get("local_path", "")
    container_path = params.get("container_path", "")
    if not local_path or not container_path:
        return "Error: Both local_path and container_path are required."
    return await _vm_result(
        get_vm().upload_file(local_path, container_path),
        emit_event="upload", local_path=local_path, container_path=container_path,
    )


# ---------------------------------------------------------------------------
# 13. vm_download — Download a file from the container
# ---------------------------------------------------------------------------

@tool(
    name="vm_download",
    description=(
        "Download a file from the virtual computer container to the host workspace. "
        "The container_path is the absolute path inside the container. "
        "The local_path is relative to the workspace directory on the host."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "container_path": {
                "type": "string",
                "description": "Absolute path of the file inside the container",
            },
            "local_path": {
                "type": "string",
                "description": "Destination path relative to the workspace directory",
            },
        },
        "required": ["container_path", "local_path"],
    },
    category="virtual_computer",
    risk="low",
    timeout=30,
)
async def vm_download(params: dict) -> str:
    container_path = params.get("container_path", "")
    local_path = params.get("local_path", "")
    if not container_path or not local_path:
        return "Error: Both container_path and local_path are required."
    return await _vm_result(
        get_vm().download_file(container_path, local_path),
        emit_event="download", container_path=container_path, local_path=local_path,
    )


# ---------------------------------------------------------------------------
# 14. vm_install_package — Install apt packages
# ---------------------------------------------------------------------------

@tool(
    name="vm_install_package",
    description=(
        "Install apt packages inside the virtual computer container. "
        "Runs apt-get update first, then installs the specified packages. "
        "Requires root access inside the container. Examples: 'vim', 'firefox', 'python3-pip'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "packages": {
                "type": "string",
                "description": (
                    "Comma-separated list of apt packages to install. "
                    "Example: 'vim,htop,python3-pip'"
                ),
            },
        },
        "required": ["packages"],
    },
    category="virtual_computer",
    risk="medium",
    timeout=300,
)
async def vm_install_package(params: dict) -> str:
    packages_str = params.get("packages", "")
    if not packages_str.strip():
        return "Error: No packages specified."
    packages = [p.strip() for p in packages_str.split(",") if p.strip()]
    return await _vm_result(
        get_vm().install_package(packages),
        emit_event="install_package", packages=packages,
    )
