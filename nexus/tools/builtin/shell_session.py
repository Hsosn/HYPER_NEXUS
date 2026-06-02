"""
Persistent shell sessions — tmux-backed with async subprocess pool.

Each session maps to a tmux session (``nexus-{name}``) so state (cwd, env,
background processes) survives server restarts. Falls back to an in-memory
subprocess approach when ``tmux`` is unavailable.

Optimizations:
- Cached tmux detection (lazy, 60s TTL)
- Async subprocess pool for fallback execution
- Connection pooling via semaphore for subprocess limits
"""
from __future__ import annotations

import asyncio
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ...config import BASE_DIR
from ..registry import tool

# Dev server auto-detection — imported lazily to avoid circular imports
_dev_server_tracker: Any | None = None

def _get_dev_server_tracker():
    """Lazy import of dev server tracking utilities."""
    global _dev_server_tracker
    if _dev_server_tracker is None:
        try:
            from .fullstack_tools import track_dev_server, untrack_dev_server
            _dev_server_tracker = {"track": track_dev_server, "untrack": untrack_dev_server}
        except ImportError:
            _dev_server_tracker = False
    return _dev_server_tracker if _dev_server_tracker else None

# Compile dev server detection patterns once
_DEV_SERVER_PORT_RE = re.compile(r"(?:https?://)?localhost[^\s]*?(\d{4,5})", re.IGNORECASE)
_DEV_SERVER_PATTERNS = [
    re.compile(r"(?:local|ready|started|running|listening)\s+(?:at|on)\s+(?:https?://)?localhost", re.IGNORECASE),
    re.compile(r"Server running on", re.IGNORECASE),
    re.compile(r"Uvicorn running on", re.IGNORECASE),
    re.compile(r"dev server", re.IGNORECASE),
    re.compile(r"npm run dev|next dev|vite|uvicorn|gunicorn", re.IGNORECASE),
    re.compile(r"(?:Application|Server) start", re.IGNORECASE),
]


def _detect_dev_server(output: str, command: str) -> tuple[str | None, int | None]:
    """Detect if a command started a dev server.
    Returns (project_name, port) or (None, None).
    """
    if not output:
        return None, None
    
    combined = output + "\n" + command
    
    # Check if this is a dev server start command
    if not any(p.search(combined) for p in _DEV_SERVER_PATTERNS):
        return None, None
    
    # Extract port from output
    port_match = _DEV_SERVER_PORT_RE.search(output)
    if not port_match:
        return None, None
    
    port = int(port_match.group(1))
    
    # Extract project name from cwd path or command
    # Try to get project name from the cwd path
    try:
        cwd = _SESSIONS.get("default", {}).get("cwd", "")
        if cwd:
            project = Path(cwd).name
            if project and project != "workspace":
                return project, port
    except Exception:
        pass
    
    return "project", port

_WORKSPACE = str((Path(BASE_DIR) / "data" / "workspace").resolve())


_RESET_COOLDOWN = 5.0
_RESET_WINDOW = 10.0
_RESET_MAX = 2
_reset_timestamps: dict[str, list[float]] = {}

# Async subprocess semaphore (max 4 concurrent subprocess calls)
_subprocess_sem = asyncio.Semaphore(4)

# ---------------------------------------------------------------------------
# Cached tmux detection (with TTL)
# ---------------------------------------------------------------------------

_TMUX_AVAILABLE: bool | None = None
_TMUX_CHECKED_AT: float = 0.0
_TMUX_CACHE_TTL = 60.0


def _check_tmux() -> bool:
    """Check whether ``tmux`` is installed, cached for 60s."""
    global _TMUX_AVAILABLE, _TMUX_CHECKED_AT
    now = time.monotonic()
    if _TMUX_AVAILABLE is not None and (now - _TMUX_CHECKED_AT) < _TMUX_CACHE_TTL:
        return _TMUX_AVAILABLE
    try:
        result = subprocess.run(["tmux", "-V"], capture_output=True, timeout=5)
        _TMUX_AVAILABLE = result.returncode == 0
    except Exception:
        _TMUX_AVAILABLE = False
    _TMUX_CHECKED_AT = now
    return _TMUX_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmux_session_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return f"nexus-{safe}"


def _check_allowed(cmd: str) -> str | None:
    """No commands are blocked — all commands are allowed."""
    return None


def _check_windows_bash_syntax(cmd: str) -> str | None:
    """Check if a command uses bash-only syntax that won't work on Windows cmd/PowerShell 5.1.

    Returns a helpful message if detected, None otherwise.
    """
    if platform.system().lower() != "windows":
        return None

    if "&&" in cmd:
        return (
            "`&&` (bash chaining) is not supported on Windows PowerShell 5.1 or cmd.exe. "
            "Use `; ` (semicolon + space) to chain commands sequentially, or use python_exec instead."
        )
    if "||" in cmd:
        return (
            "`||` (bash OR chaining) is not supported on Windows. "
            "Use python_exec with conditional logic instead."
        )
    return None


# ---------------------------------------------------------------------------
# tmux-backed implementation
# ---------------------------------------------------------------------------

async def _tmux_ensure_session(session_name: str) -> str:
    tmux_name = _tmux_session_name(session_name)
    check = subprocess.run(
        ["tmux", "has-session", "-t", tmux_name],
        capture_output=True, timeout=5,
    )
    if check.returncode != 0:
        os.makedirs(_WORKSPACE, exist_ok=True)
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", tmux_name, "-x", "200", "-y", "50", "-c", _WORKSPACE],
            capture_output=True, timeout=10,
        )
        await asyncio.sleep(0.3)
    return tmux_name


async def _tmux_run(session_name: str, command: str, timeout: int = 120) -> str:
    tmux_name = await _tmux_ensure_session(session_name)
    escaped = command.replace('"', '\\"')
    subprocess.run(["tmux", "send-keys", "-t", tmux_name, escaped, "Enter"],
                   capture_output=True, timeout=5)
    wait_time = min(timeout, 3)
    await asyncio.sleep(wait_time)
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", tmux_name, "-p", "-S", "-200"],
        capture_output=True, timeout=5, encoding='utf-8', errors='replace',
    )
    output = (result.stdout or "").rstrip()
    lines = output.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) if lines else "(no output)"


def _tmux_kill_session(session_name: str) -> bool:
    tmux_name = _tmux_session_name(session_name)
    result = subprocess.run(["tmux", "kill-session", "-t", tmux_name],
                            capture_output=True, timeout=5)
    return result.returncode == 0


def _tmux_list_sessions() -> list[dict[str, str]]:
    result = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                            capture_output=True, timeout=5, encoding='utf-8', errors='replace')
    sessions: list[dict[str, str]] = []
    if result.returncode != 0:
        return sessions
    for name in (result.stdout or "").splitlines():
        name = name.strip()
        if not name.startswith("nexus-"):
            continue
        sessions.append({"name": name[len("nexus-"):], "tmux_session": name})
    return sessions


def _tmux_capture(session_name: str, lines: int = 200) -> str:
    tmux_name = _tmux_session_name(session_name)
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", tmux_name, "-p", "-S", f"-{lines}"],
        capture_output=True, timeout=5, encoding='utf-8', errors='replace',
    )
    if result.returncode != 0:
        return f"Error: tmux session '{session_name}' not found"
    output = (result.stdout or "").rstrip()
    buf_lines = output.splitlines()
    while buf_lines and not buf_lines[-1].strip():
        buf_lines.pop()
    return "\n".join(buf_lines) if buf_lines else "(session is empty)"


# ---------------------------------------------------------------------------
# Fallback subprocess implementation (async with semaphore)
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, dict[str, Any]] = {}


def _get_or_create(name: str) -> dict[str, Any]:
    if name not in _SESSIONS:
        _SESSIONS[name] = {"cwd": _WORKSPACE, "env": {**os.environ}}
    else:
        current = Path(_SESSIONS[name]["cwd"]).resolve()
        ws_root = Path(_WORKSPACE).resolve()
        if current != ws_root and ws_root in current.parents:
            rel = current.relative_to(ws_root)
            parts = rel.parts
            if parts == ("workspace",) or parts == ("workspace", "workspace") or all(p == "workspace" for p in parts):
                _SESSIONS[name]["cwd"] = _WORKSPACE
    return _SESSIONS[name]


async def _subprocess_run(session_name: str, command: str, timeout: int = 120) -> str:
    async with _subprocess_sem:
        session = _get_or_create(session_name)
        cwd = session["cwd"]
        env = session["env"]

        if command.strip().startswith("cd "):
            target = command.strip()[3:].strip().strip('"').strip("'")
            ws_root = Path(_WORKSPACE).resolve()
            if target == "workspace" and Path(cwd).resolve() == ws_root:
                return f"cd: Already in workspace root. Use 'cd ..' to go up, or specify a subdirectory name."
            if target == "workspace":
                candidate = (Path(cwd) / target).resolve()
                if candidate.parent == ws_root and candidate != ws_root:
                    return f"cd: workspace nesting blocked."
            new_path = Path(cwd) / target if not Path(target).is_absolute() else Path(target)
            new_path = new_path.resolve()
            if not new_path.exists():
                return f"cd: {target}: No such file or directory"
            if ws_root not in new_path.parents and new_path != ws_root:
                return f"cd: {target}: Permission denied — path outside workspace"
            session["cwd"] = str(new_path)
            return f"Changed directory to: {new_path}"

        err = _check_allowed(command)
        if err:
            return f"Error: {err}"

        # Check for bash-only syntax on Windows
        bash_err = _check_windows_bash_syntax(command)
        if bash_err:
            return f"Error: {bash_err}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            try:
                out, err_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return f"Error: command timed out after {timeout}s"

            stdout = out.decode("utf-8", errors="replace")
            stderr = err_bytes.decode("utf-8", errors="replace")

            for line in stdout.splitlines() + stderr.splitlines():
                m = re.match(r"^export\s+(\w+)=(.+)$", line)
                if m:
                    session["env"][m.group(1)] = m.group(2)

            result = ""
            if stdout.strip():
                result += stdout[:6000]
            if stderr.strip():
                result += f"\n[stderr]\n{stderr[:2000]}"
            result += f"\n[exit {proc.returncode}] cwd: {cwd}"
            return result.strip()
        except FileNotFoundError:
            return (
                "Shell error: The shell binary was not found on this platform. "
                "On Windows, bash/sh may not be available. Use python_exec instead "
                "with the 'os', 'subprocess', 'pathlib', or 'shutil' modules "
                "to accomplish filesystem operations."
            )
        except Exception as e:
            return f"Shell error: {e}"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool(
    name="shell_run",
    description=(
        "Run a command in a named persistent shell session. "
        "When tmux is available the session survives server restarts and "
        "supports backgrounding processes (Ctrl+Z, bg, fg, jobs). "
        "Falls back to an in-memory subprocess if tmux is missing."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "command":         {"type": "string", "description": "Command to run"},
            "cmd":             {"type": "string", "description": "Alias for command"},
            "shell_cmd":       {"type": "string", "description": "Alias for command"},
            "session":         {"type": "string", "default": "default",
                                "description": "Session name (each session has its own cwd)"},
            "session_name":    {"type": "string", "default": "default",
                                "description": "Alias for session"},
            "timeout_seconds": {"type": "integer", "default": 120,
                                "description": "Timeout in seconds (default 120)"},
            "timeout":         {"type": "integer", "default": 120,
                                "description": "Alias for timeout_seconds"},
        },
        "required": ["command"],
    },
    risk="high",
    category="shell",
)
async def shell_run(params):
    cmd     = params.get("command", "").strip()
    name    = params.get("session", "default")
    timeout = int(params.get("timeout_seconds", 120))
    if not cmd:
        return "Error: no command provided."
    err = _check_allowed(cmd)
    if err:
        return f"Error: {err}"
    if _check_tmux():
        try:
            output = await _tmux_run(name, cmd, timeout)
        except Exception as exc:
            return f"tmux error: {exc}"
    else:
        output = await _subprocess_run(name, cmd, timeout)
    
    # Auto-detect and track dev servers
    tracker = _get_dev_server_tracker()
    if tracker:
        try:
            project, port = _detect_dev_server(output, cmd)
            if project and port:
                tracker["track"](project, port, server_type=cmd.split()[0] if cmd.split() else "unknown")
                output += f"\n[ℹ️ Dev server detected: {project} running on port {port}]"
        except Exception:
            pass
    
    return output


@tool(
    name="shell_sessions",
    description="List all active persistent shell sessions and their status.",
    parameters_schema={"type": "object", "properties": {}, "required": []},
    category="shell",
)
async def shell_sessions(_params):
    if _check_tmux():
        sessions = _tmux_list_sessions()
        if not sessions:
            return "No active tmux sessions."
        lines = ["## Active Shell Sessions (tmux-backed)"]
        for s in sessions:
            lines.append(f"  [{s['name']}] tmux: {s['tmux_session']}")
        lines.append(f"\nTotal: {len(sessions)} session(s)")
        return "\n".join(lines)
    if not _SESSIONS:
        return "No active sessions."
    lines = ["## Active Shell Sessions (in-memory)"]
    for name, s in _SESSIONS.items():
        lines.append(f"  [{name}] cwd: {s['cwd']}")
    return "\n".join(lines)


@tool(
    name="shell_reset",
    description="Kill a shell session. When tmux is available this destroys the "
                "underlying tmux session; otherwise clears the in-memory session.",
    parameters_schema={
        "type": "object",
        "properties": {
            "session": {"type": "string", "default": "default",
                        "description": "Session name to reset"},
        },
        "required": [],
    },
    category="shell",
)
async def shell_reset(params):
    name = params.get("session", "default")
    now = time.monotonic()
    if name not in _reset_timestamps:
        _reset_timestamps[name] = []
    _reset_timestamps[name] = [t for t in _reset_timestamps[name] if now - t < _RESET_WINDOW]
    if len(_reset_timestamps[name]) >= _RESET_MAX:
        return f"Rate limit: too many resets for '{name}' within {_RESET_WINDOW}s. Wait and retry."
    _reset_timestamps[name].append(now)
    if _check_tmux():
        killed = _tmux_kill_session(name)
        if killed:
            return f"Session '{name}' killed (tmux session removed)."
        return f"Session '{name}' not found."
    _SESSIONS.pop(name, None)
    return f"Session '{name}' reset. Next command will start from workspace root."


@tool(
    name="shell_attach",
    description="Get the recent output buffer from a persistent shell session. "
                "Useful for checking the output of long-running processes without "
                "sending a new command.",
    parameters_schema={
        "type": "object",
        "properties": {
            "session":  {"type": "string", "default": "default",
                         "description": "Session name"},
            "lines":    {"type": "integer", "default": 200,
                         "description": "Number of recent lines to capture"},
        },
        "required": [],
    },
    category="shell",
)
async def shell_attach(params):
    name  = params.get("session", "default")
    lines = int(params.get("lines", 200))
    if _check_tmux():
        output = _tmux_capture(name, lines)
        return f"## Session '{name}' — last {lines} lines\n\n{output}"
    if name in _SESSIONS:
        return f"Session '{name}' exists (cwd: {_SESSIONS[name]['cwd']}) but has no output buffer (tmux not available)."
    return f"Session '{name}' not found."
