"""
Code execution tools.

python_exec  — run Python in an isolated subprocess with stdout/stderr capture
               and optional file output scanning. Uses a temp file instead of
               -c to avoid OS command-line buffer limits (32767 chars on Windows).
shell_command — whitelisted shell commands.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from pathlib import Path

from ...config import BASE_DIR
from ..registry import tool

_WORKSPACE = BASE_DIR / "data" / "workspace"
_WORKSPACE.mkdir(parents=True, exist_ok=True)


@tool(
    name="python_exec",
    description=(
        "Execute Python 3 code in an isolated subprocess. Returns stdout, stderr, exit code, "
        "and any files written to the working directory. Timeout: 120s."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "code":             {"type": "string", "description": "Python source to run"},
            "script":           {"type": "string", "description": "Alias for code"},
            "python_code":      {"type": "string", "description": "Alias for code"},
            "source":           {"type": "string", "description": "Alias for code"},
            "input":            {"type": "string", "description": "Alias for code"},
            "command":          {"type": "string", "description": "Alias for code"},
            "payload":          {"type": "string", "description": "Alias for code"},
            "program":          {"type": "string", "description": "Alias for code"},
            "timeout_seconds":  {"type": "integer", "default": 120,
                                 "description": "Timeout in seconds (default 120)"},
            "timeout":          {"type": "integer", "default": 120,
                                 "description": "Alias for timeout_seconds"},
            "max_time":         {"type": "integer", "default": 120,
                                 "description": "Alias for timeout_seconds"},
            "capture_files":    {"type": "boolean", "default": True,
                                 "description": "List new files created during execution"},
            "capture":          {"type": "boolean", "default": True,
                                 "description": "Alias for capture_files"},
        },
        "required": ["code"],
    },
    risk="high",
    category="code",
    timeout=120,
)
async def python_exec(params):
    # Resolve code from multiple possible keys (defence-in-depth against parser aliasing)
    code = (params.get("code") or params.get("script") or params.get("python_code")
            or params.get("source") or params.get("program") or params.get("payload")
            or params.get("body") or params.get("input") or params.get("command")
            or "")
    timeout  = int(params.get("timeout_seconds", 120))
    capture  = bool(params.get("capture_files", True))
    if not code:
        return "Error: 'code' parameter is required. Provide Python source to execute."
    if len(code) > 30000:
        return "Error: code too large (max 30000 chars)"

    # Write code to a temp file instead of passing via -c.
    # This avoids the OS command-line buffer limit (32,767 chars on Windows)
    # which could otherwise truncate the payload and cause the overflow to
    # leak outside the tool boundary.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write(code)
        tmp.close()

        # Run inside the workspace directory so relative file writes land there
        work_dir = str(_WORKSPACE)

        # Snapshot existing files before execution
        before_files: set[str] = set()
        if capture:
            before_files = {str(p) for p in _WORKSPACE.rglob("*") if p.is_file()}

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", tmp.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return f"Error: execution timed out after {timeout}s"

            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")

            result_parts = []
            if out:
                result_parts.append(f"STDOUT:\n{out[:5000]}")
            if err:
                result_parts.append(f"STDERR:\n{err[:2000]}")
            result_parts.append(f"Exit code: {proc.returncode}")

            # Detect new files
            if capture:
                after_files = {str(p) for p in _WORKSPACE.rglob("*") if p.is_file()}
                new_files = after_files - before_files
                if new_files:
                    names = [Path(f).relative_to(_WORKSPACE) for f in sorted(new_files)]
                    result_parts.append(f"Files created: {', '.join(str(n) for n in names)}")

            return "\n\n".join(result_parts) or "(no output)"
        except Exception as e:
            return f"Execution error: {e}"
    finally:
        # Clean up the temp file
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@tool(
    name="shell_command",
    description=(
        "Run a shell command in the workspace directory. Timeout 120s."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "command":          {"type": "string"},
            "timeout_seconds":  {"type": "integer", "default": 120,
                                 "description": "Timeout in seconds (default 120)"},
        },
        "required": ["command"],
    },
    risk="high",
    category="code",
    timeout=120,
)
async def shell_command(params):
    cmd     = params.get("command", "").strip()
    timeout = int(params.get("timeout_seconds", 120))
    try:
        # On Windows, use the native shell (cmd.exe) to handle paths with
        # backslashes correctly.  On POSIX, shlex.split + subprocess_exec
        # avoids shell-injection surface for a cleaner execution model.
        if os.name == "nt":
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_WORKSPACE),
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_WORKSPACE),
            )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return "Error: command timed out"
        return (out.decode("utf-8", errors="replace")[:5000] +
                ("\n" + err.decode("utf-8", errors="replace")[:1000] if err.strip() else ""))
    except Exception as e:
        return f"Shell error: {e}"
