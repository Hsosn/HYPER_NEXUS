"""
VirtualComputer — Docker-based headless desktop environment.

Provides a clean Python API for managing a virtual computer container
with Xvfb, x11vnc, and xdotool for GUI interaction.

Container name: nexus-vm
Base image: ubuntu:22.04
RAM: 2GB
Display: :99 (1280x720x24)
VNC: port 5900 (no password)

Fallback: When Docker is unavailable, a subprocess-based sandbox is used
instead.  Xvfb runs directly on the host with xdotool for input, and
commands are executed via asyncio.create_subprocess_exec.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.virtual_computer")


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "nexus-vm"
BASE_IMAGE = "ubuntu:22.04"
MEMORY_LIMIT = "2g"
DISPLAY_NUM = ":99"
DISPLAY_RESOLUTION = "1280x720x24"
VNC_PORT = "5900"

# Subprocess-mode specific constants
XVFB_DISPLAY = ":99"
XVFB_RESOLUTION = "1280x720x24"
SCREENSHOT_PATH = "/tmp/vm_screenshot.png"

# Host paths (relative to project root)
_HOST_BASE: Path | None = None


def _get_base_dir() -> Path:
    """Lazily resolve the project base directory."""
    global _HOST_BASE
    if _HOST_BASE is None:
        try:
            from nexus.config import BASE_DIR as _BD
            _HOST_BASE = _BD
        except Exception:
            _HOST_BASE = Path.cwd()
    return _HOST_BASE


def _get_workspace_host() -> Path:
    """Return the host-side workspace directory to bind-mount."""
    return _get_base_dir() / "data" / "workspace"


def _get_setup_script() -> Path:
    """Return the path to the setup-vm.sh script."""
    return _get_base_dir() / "nexus" / "virtual_computer" / "setup-vm.sh"


# ---------------------------------------------------------------------------
# Helper: run a subprocess asynchronously
# ---------------------------------------------------------------------------

async def _run(
    cmd: list[str],
    timeout: float = 60,
    check: bool = True,
) -> tuple[str, str, int]:
    """Run *cmd* via ``asyncio.create_subprocess_exec``.

    Returns ``(stdout, stderr, returncode)`` — all decoded as UTF-8.
    """
    logger.debug("exec: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return "", f"Process timed out after {timeout}s", -1

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return stdout, stderr, proc.returncode


async def _run_raw(
    cmd: list[str],
    timeout: float = 60,
) -> tuple[bytes, bytes, int]:
    """Run *cmd* and return raw bytes without decoding.

    Use this for binary outputs (e.g. extracting PNG screenshots).

    Returns ``(stdout_bytes, stderr_bytes, returncode)``.
    """
    logger.debug("exec (raw): %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return b"", f"Process timed out after {timeout}s".encode(), -1

    return stdout_bytes, stderr_bytes, proc.returncode


import sys as _sys


def _docker(*args: str) -> list[str]:
    """Build a docker CLI command list.

    On Windows (Docker Desktop), we must specify the ``desktop-linux`` context
    so that commands are routed to the WSL2 backend instead of the default
    Windows daemon.  On Linux/macOS the context flag is omitted entirely —
    passing ``-c desktop-linux`` on those platforms causes every Docker call
    to fail with "context not found".
    """
    if _sys.platform == "win32":
        return ["docker", "-c", "desktop-linux", *args]
    return ["docker", *args]


# ---------------------------------------------------------------------------
# VirtualComputer class
# ---------------------------------------------------------------------------

class VirtualComputer:
    """High-level interface to the NEXUS virtual computer.

    When Docker is available the original container-based implementation is
    used.  When Docker is **not** available the class transparently falls back
    to a *subprocess mode* that runs Xvfb, xdotool and commands directly on
    the host.

    Usage::

        vc = VirtualComputer()
        await vc.start()
        png_bytes = await vc.screenshot()
        await vc.mouse_move(640, 360)
        await vc.mouse_click()
        await vc.type_text("hello world")
        await vc.stop()
    """

    def __init__(
        self,
        container_name: str = CONTAINER_NAME,
        base_image: str = BASE_IMAGE,
        workspace_host: Path | None = None,
        workspace_container: str = "/workspace",
    ) -> None:
        self.container_name = container_name
        self.base_image = base_image
        self.workspace_host = workspace_host or _get_workspace_host()
        self.workspace_container = workspace_container
        self._created: bool = False

        # Docker availability flag — None means "not yet checked".
        self._use_docker: bool | None = None

        # Subprocess-mode state
        self._xvfb_proc: asyncio.subprocess.Process | None = None
        self._xvfb_pid: int | None = None  # OS-level PID for bookkeeping

    # ------------------------------------------------------------------
    # Docker detection
    # ------------------------------------------------------------------

    async def _detect_docker(self) -> bool:
        """Check whether Docker is available and set ``_use_docker``."""
        _, _, rc = await _run(_docker("info"), timeout=10, check=False)
        available = rc == 0
        self._use_docker = available
        if available:
            logger.info("Docker detected — using container mode.")
        else:
            logger.info(
                "Docker NOT detected — falling back to subprocess mode."
            )
        return available

    async def _ensure_docker(self) -> str | None:
        """Return an error message if Docker is unavailable, else ``None``.

        As a side-effect this sets ``_use_docker`` on the first call.
        """
        if self._use_docker is None:
            await self._detect_docker()

        if not self._use_docker:
            return (
                "Docker is not available or not running. "
                "Subprocess fallback mode will be used instead."
            )
        return None

    # ------------------------------------------------------------------
    # Docker helpers (unchanged from original)
    # ------------------------------------------------------------------

    async def _get_container_id(self) -> str:
        """Get the full container hash ID from Docker."""
        try:
            stdout, _, rc = await _run(
                _docker("ps", "-a", "--filter", f"name={self.container_name}",
                        "--format", "{{.ID}}"),
                timeout=10,
            )
            if rc == 0 and stdout.strip():
                return stdout.strip()[:12]
        except Exception:
            pass
        return ""

    async def _docker_exec(
        self, cmd: str, timeout: float = 30
    ) -> tuple[str, str, int]:
        """Execute a shell command inside the container."""
        args = _docker(
            "exec", self.container_name,
            "/bin/bash", "-c", cmd,
        )
        return await _run(args, timeout=timeout)

    async def _docker_exec_root(
        self, cmd: str, timeout: float = 30
    ) -> tuple[str, str, int]:
        """Execute a shell command inside the container as root."""
        args = _docker(
            "exec", "-u", "root", self.container_name,
            "/bin/bash", "-c", cmd,
        )
        return await _run(args, timeout=timeout)

    async def _container_exists(self) -> bool:
        """Check if the container exists (running or stopped)."""
        stdout, _, rc = await _run(
            _docker("ps", "-a", "--filter", f"name={self.container_name}",
                    "--format", "{{.Names}}"),
            timeout=10,
            check=False,
        )
        return self.container_name in stdout

    async def _container_running(self) -> bool:
        """Check if the container is currently running."""
        stdout, _, _ = await _run(
            _docker("ps", "--filter", f"name={self.container_name}",
                    "--format", "{{.Names}}"),
            timeout=10,
            check=False,
        )
        return self.container_name in stdout

    # ------------------------------------------------------------------
    # Subprocess-mode helpers
    # ------------------------------------------------------------------

    async def _subprocess_ensure_xdotool(self) -> bool:
        """Ensure xdotool is available on the host (install if needed)."""
        if shutil.which("xdotool"):
            return True
        # Try to install via apt-get (works when running as root)
        logger.info("xdotool not found — attempting to install via apt-get…")
        stdout, stderr, rc = await _run(
            ["apt-get", "install", "-y", "--no-install-recommends", "xdotool"],
            timeout=120,
            check=False,
        )
        if rc == 0 and shutil.which("xdotool"):
            logger.info("xdotool installed successfully.")
            return True
        # Try with sudo
        stdout, stderr, rc = await _run(
            ["sudo", "apt-get", "install", "-y", "--no-install-recommends", "xdotool"],
            timeout=120,
            check=False,
        )
        if rc == 0 and shutil.which("xdotool"):
            logger.info("xdotool installed successfully (via sudo).")
            return True
        logger.warning("Could not install xdotool: %s", stderr[:300])
        return False

    async def _subprocess_ensure_imagemagick(self) -> bool:
        """Ensure ImageMagick (import command) is available for screenshots."""
        if shutil.which("import"):
            return True
        # import is part of imagemagick package
        logger.info("ImageMagick not found — attempting to install…")
        stdout, stderr, rc = await _run(
            ["apt-get", "install", "-y", "--no-install-recommends", "imagemagick"],
            timeout=120,
            check=False,
        )
        if rc == 0 and shutil.which("import"):
            return True
        stdout, stderr, rc = await _run(
            ["sudo", "apt-get", "install", "-y", "--no-install-recommends", "imagemagick"],
            timeout=120,
            check=False,
        )
        if rc == 0 and shutil.which("import"):
            return True
        logger.warning("Could not install ImageMagick: %s", stderr[:300])
        return False

    async def _subprocess_is_xvfb_running(self) -> bool:
        """Check if our Xvfb process is still alive."""
        if self._xvfb_proc is not None:
            if self._xvfb_proc.returncode is None:
                return True
            # Process has exited
            self._xvfb_proc = None
            self._xvfb_pid = None
            return False
        # Try to detect a running Xvfb on our display by other means
        stdout, _, rc = await _run(
            ["pgrep", "-f", f"Xvfb {XVFB_DISPLAY}"],
            timeout=5,
            check=False,
        )
        return bool(stdout.strip())

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def start(self) -> dict[str, Any]:
        """Create and start the virtual computer.

        If Docker is available the original container-based start is used.
        If Docker is NOT available a subprocess-based Xvfb session is started
        on the host instead.

        Returns a status dict.
        """
        # Detect Docker on first call
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._start_docker()
        else:
            return await self._start_subprocess()

    # -- Docker start (original logic, unchanged) -------------------------

    async def _start_docker(self) -> dict[str, Any]:
        """Create and start the virtual computer container (Docker mode)."""
        # If container is running, just return status
        if await self._container_running():
            logger.info("Container '%s' is already running.", self.container_name)
            return await self.status()

        # If container exists but stopped, try to start it - if it fails (e.g., stale
        # device config), destroy and recreate it.
        if await self._container_exists():
            logger.info("Container '%s' exists but stopped — starting.", self.container_name)
            stdout, stderr, rc = await _run(
                _docker("start", self.container_name), timeout=60
            )
            if rc != 0:
                logger.warning(
                    "Failed to start existing container (%s), removing and recreating.",
                    stderr.strip()[:200],
                )
                await _run(_docker("rm", "-f", self.container_name), timeout=10)
            else:
                # Ensure services are running after start
                await self._ensure_services()
                return await self.status()

        # Create new container
        return await self._create_and_start()

    async def _create_and_start(self) -> dict[str, Any]:
        """Create the container from scratch and start it."""
        workspace = self.workspace_host
        workspace.mkdir(parents=True, exist_ok=True)

        setup_script = _get_setup_script()

        docker_args = [
            "run", "-d",
            "--name", self.container_name,
            "--memory", MEMORY_LIMIT,
            "--shm-size", "1g",
            "-v", f"{workspace}:{self.workspace_container}",
            "-e", "DISPLAY=:99",
            "-p", f"{VNC_PORT}:5900",
            "--cap-add", "SYS_ADMIN",
            self.base_image,
            "/bin/bash", "-c", "tail -f /dev/null",
        ]

        args = _docker(*docker_args)
        stdout, stderr, rc = await _run(args, timeout=120)

        if rc != 0:
            # Retry with minimal options on ANY container creation failure.
            # We drop --cap-add and --shm-size extras that some Docker/OS
            # combinations reject.
            logger.warning(
                "Container creation failed (%s), retrying with minimal options",
                stderr.strip()[:200],
            )
            fallback_args = [
                "run", "-d",
                "--name", self.container_name,
                "--memory", MEMORY_LIMIT,
                "--shm-size", "512m",
                "-v", f"{workspace}:{self.workspace_container}",
                "-e", "DISPLAY=:99",
                "-p", f"{VNC_PORT}:5900",
                self.base_image,
                "/bin/bash", "-c", "tail -f /dev/null",
            ]
            args = _docker(*fallback_args)
            stdout, stderr, rc = await _run(args, timeout=120)

            if rc != 0:
                return {
                    "status": "error",
                    "message": f"Failed to create container: {stderr.strip()}",
                }

        self._created = True
        logger.info("Container '%s' created.", self.container_name)

        # Wait a moment for container to be ready
        await asyncio.sleep(2)

        # Copy and run the setup script
        if setup_script.exists():
            logger.info("Running setup script inside container...")
            cp_out, cp_err, cp_rc = await _run(
                _docker("cp", str(setup_script),
                        f"{self.container_name}:/tmp/setup-vm.sh"),
                timeout=30,
            )
            if cp_rc != 0:
                logger.warning("Failed to copy setup script: %s", cp_err)

            # Run the setup script
            setup_out, setup_err, setup_rc = await self._docker_exec_root(
                "chmod +x /tmp/setup-vm.sh && bash /tmp/setup-vm.sh",
                timeout=180,
            )
            if setup_rc != 0:
                logger.warning("Setup script had errors: %s", setup_err[-500:])
            else:
                logger.info("Setup script completed successfully.")
        else:
            # Fallback: manually install and start services
            logger.info("Setup script not found, installing services manually...")
            await self._install_services_manual()

        # Verify display is working
        await asyncio.sleep(2)
        self._created = True

        return await self.status()

    async def _install_services_manual(self) -> None:
        """Manually install and configure virtual display services."""
        await self._docker_exec_root(
            "export DEBIAN_FRONTEND=noninteractive && "
            "apt-get update -qq && "
            "apt-get install -y --no-install-recommends "
            "xvfb x11vnc xdotool imagemagick 2>/dev/null && "
            "apt-get clean && "
            "mkdir -p /workspace && chmod 777 /workspace",
            timeout=180,
        )
        await self._ensure_services()

    async def _ensure_services(self) -> None:
        """Make sure Xvfb and x11vnc are running inside the container."""
        # Check if Xvfb is running
        stdout, _, _ = await self._docker_exec(
            "pgrep -x Xvfb", timeout=5
        )
        if not stdout.strip():
            await self._docker_exec_root(
                f"Xvfb {DISPLAY_NUM} -screen 0 {DISPLAY_RESOLUTION} "
                f"-ac -extension GLX -render -noreset &>/dev/null & "
                f"sleep 1",
                timeout=10,
            )

        # Check if x11vnc is running
        stdout, _, _ = await self._docker_exec(
            "pgrep -x x11vnc", timeout=5
        )
        if not stdout.strip():
            await self._docker_exec_root(
                f"DISPLAY={DISPLAY_NUM} x11vnc -display {DISPLAY_NUM} "
                f"-rfbport {VNC_PORT} -forever -nopw -shared "
                f"-bg -o /tmp/x11vnc.log 2>/dev/null",
                timeout=10,
            )

        # Ensure DISPLAY env is set
        await self._docker_exec_root(
            f"echo 'DISPLAY={DISPLAY_NUM}' >> /etc/environment",
            timeout=5,
        )

    # -- Subprocess start -------------------------------------------------

    async def _start_subprocess(self) -> dict[str, Any]:
        """Start Xvfb and prepare the subprocess-based virtual computer."""
        # If Xvfb is already running under our management, return status
        if await self._subprocess_is_xvfb_running():
            logger.info("Xvfb is already running on display %s.", XVFB_DISPLAY)
            return await self.status()

        # Ensure workspace directory exists
        self.workspace_host.mkdir(parents=True, exist_ok=True)

        # Ensure Xvfb binary exists
        xvfb_bin = shutil.which("Xvfb")
        if not xvfb_bin:
            # Try to install it
            logger.info("Xvfb not found — attempting to install via apt-get…")
            stdout, stderr, rc = await _run(
                ["apt-get", "install", "-y", "--no-install-recommends", "xvfb"],
                timeout=120,
                check=False,
            )
            if rc != 0:
                stdout, stderr, rc = await _run(
                    ["sudo", "apt-get", "install", "-y", "--no-install-recommends", "xvfb"],
                    timeout=120,
                    check=False,
                )
            xvfb_bin = shutil.which("Xvfb")
            if not xvfb_bin:
                return {
                    "status": "error",
                    "message": (
                        "Xvfb is not installed and could not be installed. "
                        "Please install xvfb: apt-get install xvfb"
                    ),
                }

        # Kill any existing Xvfb on our display to avoid conflicts
        await _run(
            ["pkill", "-f", f"Xvfb {XVFB_DISPLAY}"],
            timeout=5,
            check=False,
        )
        await asyncio.sleep(0.5)

        # Start Xvfb
        logger.info(
            "Starting Xvfb on display %s with resolution %s…",
            XVFB_DISPLAY, XVFB_RESOLUTION,
        )
        env = os.environ.copy()
        # Ensure no existing display interferes
        env.pop("DISPLAY", None)

        proc = await asyncio.create_subprocess_exec(
            xvfb_bin, XVFB_DISPLAY,
            "-screen", "0", XVFB_RESOLUTION,
            "-ac", "-render", "-noreset",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._xvfb_proc = proc
        self._xvfb_pid = proc.pid

        # Wait for Xvfb to start up
        await asyncio.sleep(1.5)

        # Verify Xvfb is still alive
        if proc.returncode is not None:
            self._xvfb_proc = None
            self._xvfb_pid = None
            return {
                "status": "error",
                "message": f"Xvfb exited immediately with code {proc.returncode}.",
            }

        logger.info("Xvfb started (PID: %s).", proc.pid)

        # Ensure xdotool is available
        xdotool_ok = await self._subprocess_ensure_xdotool()
        if not xdotool_ok:
            logger.warning(
                "xdotool is not available — mouse/keyboard input will not work."
            )

        # Ensure ImageMagick is available for screenshots
        imagemagick_ok = await self._subprocess_ensure_imagemagick()
        if not imagemagick_ok:
            logger.warning(
                "ImageMagick is not available — screenshots may not work."
            )

        # Try to start a window manager (fluxbox if available) for a better
        # visual experience, but don't fail if it's missing.
        fluxbox_bin = shutil.which("fluxbox")
        if fluxbox_bin:
            wm_env = os.environ.copy()
            wm_env["DISPLAY"] = XVFB_DISPLAY
            try:
                wm_proc = await asyncio.create_subprocess_exec(
                    fluxbox_bin,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=wm_env,
                )
                logger.info("Fluxbox window manager started (PID: %s).", wm_proc.pid)
            except Exception as exc:
                logger.debug("Could not start fluxbox: %s", exc)

        self._created = True
        return await self.status()

    # ------------------------------------------------------------------
    # stop
    # ------------------------------------------------------------------

    async def stop(self) -> dict[str, Any]:
        """Stop the virtual computer.

        In Docker mode the container is stopped (preserving state).
        In subprocess mode the Xvfb process is killed.
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._stop_docker()
        else:
            return await self._stop_subprocess()

    async def _stop_docker(self) -> dict[str, Any]:
        """Stop the Docker container (original logic, unchanged)."""
        if not await self._container_exists():
            return {"status": "not_found", "message": f"Container '{self.container_name}' does not exist."}

        if not await self._container_running():
            return {"status": "stopped", "message": f"Container '{self.container_name}' is already stopped."}

        stdout, stderr, rc = await _run(
            _docker("stop", self.container_name), timeout=60
        )
        if rc != 0:
            return {
                "status": "error",
                "message": f"Failed to stop container: {stderr.strip()}",
            }

        logger.info("Container '%s' stopped.", self.container_name)
        return {"status": "stopped", "message": "Virtual computer stopped. Container preserved."}

    async def _stop_subprocess(self) -> dict[str, Any]:
        """Stop the subprocess-based Xvfb session."""
        if self._xvfb_proc is not None and self._xvfb_proc.returncode is None:
            self._xvfb_proc.terminate()
            try:
                await asyncio.wait_for(self._xvfb_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._xvfb_proc.kill()
                await self._xvfb_proc.wait()
            logger.info("Xvfb process (PID %s) terminated.", self._xvfb_pid)
            self._xvfb_proc = None
            self._xvfb_pid = None
            self._created = False
            return {"status": "stopped", "message": "Virtual computer stopped (subprocess mode). Xvfb terminated."}

        # Check if Xvfb is running even if not under our management
        stdout, _, rc = await _run(
            ["pgrep", "-f", f"Xvfb {XVFB_DISPLAY}"],
            timeout=5,
            check=False,
        )
        if stdout.strip():
            pids = stdout.strip().splitlines()
            for pid in pids:
                await _run(["kill", pid.strip()], timeout=5, check=False)
            logger.info("Killed orphaned Xvfb process(es): %s", pids)
            self._xvfb_proc = None
            self._xvfb_pid = None
            self._created = False
            return {"status": "stopped", "message": "Virtual computer stopped (subprocess mode). Orphaned Xvfb killed."}

        return {"status": "stopped", "message": "Virtual computer is not running (subprocess mode)."}

    # ------------------------------------------------------------------
    # restart
    # ------------------------------------------------------------------

    async def restart(self) -> dict[str, Any]:
        """Restart the virtual computer."""
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._restart_docker()
        else:
            await self._stop_subprocess()
            await asyncio.sleep(1)
            return await self._start_subprocess()

    async def _restart_docker(self) -> dict[str, Any]:
        """Restart the Docker container (original logic, unchanged)."""
        if not await self._container_exists():
            return await self.start()

        stdout, stderr, rc = await _run(
            _docker("restart", self.container_name), timeout=90
        )
        if rc != 0:
            return {
                "status": "error",
                "message": f"Failed to restart container: {stderr.strip()}",
            }

        # Re-ensure services after restart
        await asyncio.sleep(3)
        await self._ensure_services()

        logger.info("Container '%s' restarted.", self.container_name)
        return await self.status()

    # ------------------------------------------------------------------
    # destroy
    # ------------------------------------------------------------------

    async def destroy(self) -> dict[str, Any]:
        """Stop and remove the virtual computer.

        In Docker mode the container is removed (data inside container is lost,
        workspace bind-mount is preserved).
        In subprocess mode Xvfb is killed.
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._destroy_docker()
        else:
            # In subprocess mode "destroy" is equivalent to a hard stop
            return await self._stop_subprocess()

    async def _destroy_docker(self) -> dict[str, Any]:
        """Destroy the Docker container (original logic, unchanged)."""
        if not await self._container_exists():
            return {"status": "not_found", "message": f"Container '{self.container_name}' does not exist."}

        stdout, stderr, rc = await _run(
            _docker("rm", "-f", self.container_name), timeout=60
        )
        if rc != 0:
            return {
                "status": "error",
                "message": f"Failed to destroy container: {stderr.strip()}",
            }

        self._created = False
        logger.info("Container '%s' destroyed.", self.container_name)
        return {"status": "destroyed", "message": "Virtual computer destroyed. Workspace files preserved on host."}

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """Return detailed status information about the virtual computer."""
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._status_docker()
        else:
            return await self._status_subprocess()

    async def _status_docker(self) -> dict[str, Any]:
        """Return status for Docker mode (original logic, unchanged)."""
        exists = await self._container_exists()
        if not exists:
            return {
                "running": False,
                "status": "not_created",
                "container_name": self.container_name,
                "docker_available": True,
                "base_image": self.base_image,
                "message": "Container does not exist. Use vm_start to create it.",
            }

        is_running = await self._container_running()
        if not is_running:
            return {
                "running": False,
                "status": "stopped",
                "container_name": self.container_name,
                "docker_available": True,
                "base_image": self.base_image,
                "message": "Container exists but is not running. Use vm_start to start it.",
            }

        # Container is running - get detailed info via inspect
        stdout, _, _ = await _run(
            _docker(
                "inspect", self.container_name,
                "--format", "{{json .State}}",
            ),
            timeout=15,
        )

        try:
            info = json.loads(stdout.strip()) if stdout.strip() else {}
        except json.JSONDecodeError:
            info = {}

        running = info.get("Running") or str(info.get("running", "")).lower() == "true"
        state_status = info.get("Status", info.get("status", "unknown"))

        result: dict[str, Any] = {
            "running": running,
            "status": "running" if running else "stopped",
            "container_name": self.container_name,
            "base_image": self.base_image,
            "docker_available": True,
            "container_id": "",
            "image": "",
            "created": "",
            "state": state_status,
            "pid": "",
            "ip_address": "N/A",
            "display": DISPLAY_NUM,
            "resolution": DISPLAY_RESOLUTION,
            "vnc_port": VNC_PORT,
            "workspace_host": str(self.workspace_host),
            "workspace_container": self.workspace_container,
            "memory_limit": MEMORY_LIMIT,
        }

        # Get additional details if running
        if running:
            stdout_info, _, _ = await _run(
                _docker(
                    "inspect", self.container_name,
                    "--format", "{{json .Config.Image}} {{.Created}} {{.State.Pid}} {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                ),
                timeout=15,
            )
            parts = stdout_info.strip().split()
            if len(parts) >= 1:
                result["image"] = parts[0]
            if len(parts) >= 2:
                result["created"] = " ".join(parts[1:4]) if len(parts) > 4 else " ".join(parts[1:])
            if len(parts) >= 3:
                result["pid"] = parts[-2] if len(parts) > 2 else ""
            if len(parts) >= 4:
                result["ip_address"] = parts[-1]
            result["container_id"] = await self._get_container_id()

            # Get uptime
            uptime_out, _, _ = await self._docker_exec(
                "cat /proc/uptime", timeout=5
            )
            if uptime_out.strip():
                uptime_secs = float(uptime_out.strip().split()[0])
                result["uptime_seconds"] = uptime_secs
                hours, remainder = divmod(int(uptime_secs), 3600)
                minutes, seconds = divmod(remainder, 60)
                result["uptime"] = f"{hours}h {minutes}m {seconds}s"

            # Check services
            for svc in ("Xvfb", "x11vnc", "xdotool"):
                svc_out, _, _ = await self._docker_exec(
                    f"which {svc.lower() if svc != 'Xvfb' else 'Xvfb'} 2>/dev/null",
                    timeout=5,
                )
                result[f"{svc.lower()}_available"] = bool(svc_out.strip())

        return result

    async def _status_subprocess(self) -> dict[str, Any]:
        """Return status for subprocess mode."""
        is_running = await self._subprocess_is_xvfb_running()

        result: dict[str, Any] = {
            "running": is_running,
            "status": "running" if is_running else "stopped",
            "mode": "subprocess",
            "docker_available": False,
            "display": XVFB_DISPLAY,
            "resolution": XVFB_RESOLUTION,
            "workspace_host": str(self.workspace_host),
        }

        if is_running:
            # Report Xvfb PID
            result["xvfb_pid"] = self._xvfb_pid or "unknown"

            # Check tool availability
            result["xdotool_available"] = bool(shutil.which("xdotool"))
            result["imagemagick_available"] = bool(shutil.which("import"))
            result["xvfb_available"] = True  # it's running

            # Uptime (based on /proc if available)
            if self._xvfb_pid:
                try:
                    proc_stat = Path(f"/proc/{self._xvfb_pid}/stat")
                    if proc_stat.exists():
                        stat_text = proc_stat.read_text()
                        starttime_ticks = int(stat_text.split()[21])
                        # Get system clock ticks per second
                        ticks_per_sec = os.sysconf("SC_CLK_TCK")
                        boot_time = starttime_ticks / ticks_per_sec
                        uptime_secs = time.time() - boot_time
                        result["uptime_seconds"] = uptime_secs
                        hours, remainder = divmod(int(uptime_secs), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        result["uptime"] = f"{hours}h {minutes}m {seconds}s"
                except Exception:
                    pass

            result["message"] = (
                "Virtual computer running in subprocess mode. "
                "Docker is not available — using host Xvfb directly."
            )
        else:
            result["message"] = (
                "Virtual computer is not running (subprocess mode). "
                "Use vm_start to start it."
            )

        return result

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def screenshot(self) -> bytes:
        """Capture the virtual display as PNG bytes.

        In Docker mode uses ImageMagick ``import`` inside the container.
        In subprocess mode uses ImageMagick ``import`` on the host.

        Returns:
            PNG image bytes, or empty bytes on failure.
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._screenshot_docker()
        else:
            return await self._screenshot_subprocess()

    async def _screenshot_docker(self) -> bytes:
        """Screenshot in Docker mode (original logic, unchanged)."""
        # Try ImageMagick import first
        stdout, stderr, rc = await self._docker_exec(
            f'DISPLAY={DISPLAY_NUM} import -window root png:/tmp/vm_screenshot.png 2>&1',
            timeout=15,
        )

        if rc != 0:
            # Fallback: use xwd + pnmtopng
            stdout2, stderr2, rc2 = await self._docker_exec(
                f'DISPLAY={DISPLAY_NUM} xwd -root -out /tmp/vm_screen.xwd 2>&1 && '
                f'DISPLAY={DISPLAY_NUM} xwd -root | pnmtopng > /tmp/vm_screenshot.png 2>&1',
                timeout=15,
            )
            if rc2 != 0:
                logger.error("Screenshot failed: %s / %s", stderr, stderr2)
                return b""

        # Extract raw PNG bytes from container via docker exec cat (not docker cp,
        # which wraps output in a tar stream that would corrupt the binary data).
        png_bytes, stderr, rc = await _run_raw(
            _docker("exec", self.container_name, "cat", "/tmp/vm_screenshot.png"),
            timeout=10,
        )

        if rc != 0 or not png_bytes:
            logger.error("Failed to extract screenshot: %s", stderr.decode("utf-8", errors="replace"))
            return b""

        return png_bytes

    async def _screenshot_subprocess(self) -> bytes:
        """Screenshot in subprocess mode using host ImageMagick."""
        if not await self._subprocess_is_xvfb_running():
            logger.error("Cannot take screenshot: Xvfb is not running.")
            return b""

        # Remove old screenshot file if it exists
        screenshot_path = Path(SCREENSHOT_PATH)
        if screenshot_path.exists():
            screenshot_path.unlink()

        # Try ImageMagick import first
        stdout, stderr, rc = await _run(
            ["env", f"DISPLAY={XVFB_DISPLAY}", "import", "-window", "root",
             f"png:{SCREENSHOT_PATH}"],
            timeout=15,
            check=False,
        )

        if rc != 0 or not screenshot_path.exists():
            # Fallback: use xwd + convert
            logger.debug(
                "ImageMagick import failed (%s), trying xwd fallback…",
                stderr[:200],
            )
            # Clean up partial file
            if screenshot_path.exists():
                screenshot_path.unlink()

            stdout2, stderr2, rc2 = await _run(
                ["env", f"DISPLAY={XVFB_DISPLAY}",
                 "bash", "-c",
                 f"xwd -root | convert xwd:- png:{SCREENSHOT_PATH}"],
                timeout=15,
                check=False,
            )

            if rc2 != 0 or not screenshot_path.exists():
                # Third fallback: xdotool + scrot if available
                scrot_bin = shutil.which("scrot")
                if scrot_bin:
                    stdout3, stderr3, rc3 = await _run(
                        ["env", f"DISPLAY={XVFB_DISPLAY}", scrot_bin,
                         SCREENSHOT_PATH],
                        timeout=15,
                        check=False,
                    )
                    if rc3 != 0 or not screenshot_path.exists():
                        logger.error(
                            "All screenshot methods failed. import: %s; xwd: %s; scrot: %s",
                            stderr[:200], stderr2[:200], stderr3[:200],
                        )
                        return b""
                else:
                    logger.error(
                        "Screenshot failed. import: %s; xwd+convert: %s",
                        stderr[:200], stderr2[:200],
                    )
                    return b""

        # Read the PNG bytes from disk
        try:
            png_bytes = screenshot_path.read_bytes()
            # Clean up
            screenshot_path.unlink(missing_ok=True)
            return png_bytes
        except OSError as exc:
            logger.error("Failed to read screenshot file: %s", exc)
            return b""

    # ------------------------------------------------------------------
    # Mouse input
    # ------------------------------------------------------------------

    async def mouse_move(self, x: int, y: int) -> str:
        """Move the mouse pointer to screen coordinates ``(x, y)``."""
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._mouse_move_docker(x, y)
        else:
            return await self._mouse_move_subprocess(x, y)

    async def _mouse_move_docker(self, x: int, y: int) -> str:
        """Mouse move in Docker mode (original logic, unchanged)."""
        stdout, stderr, rc = await self._docker_exec(
            f'DISPLAY={DISPLAY_NUM} xdotool mousemove {int(x)} {int(y)}',
            timeout=10,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return f"Mouse moved to ({x}, {y})"

    async def _mouse_move_subprocess(self, x: int, y: int) -> str:
        """Mouse move in subprocess mode using host xdotool."""
        stdout, stderr, rc = await _run(
            ["env", f"DISPLAY={XVFB_DISPLAY}", "xdotool", "mousemove",
             str(int(x)), str(int(y))],
            timeout=10,
            check=False,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return f"Mouse moved to ({x}, {y})"

    async def mouse_click(
        self, x: int | None = None, y: int | None = None, button: int = 1
    ) -> str:
        """Click a mouse button.

        Args:
            x: Optional X coordinate to move to before clicking.
            y: Optional Y coordinate to move to before clicking.
            button: Mouse button number (1=left, 2=middle, 3=right).
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._mouse_click_docker(x, y, button)
        else:
            return await self._mouse_click_subprocess(x, y, button)

    async def _mouse_click_docker(
        self, x: int | None, y: int | None, button: int
    ) -> str:
        """Mouse click in Docker mode (original logic, unchanged)."""
        parts: list[str] = [f"DISPLAY={DISPLAY_NUM}"]
        if x is not None and y is not None:
            parts.append(f"xdotool mousemove {int(x)} {int(y)}")
            parts.append("sleep 0.05")
        parts.append(f"xdotool click {int(button)}")

        cmd = " && ".join(parts)
        stdout, stderr, rc = await self._docker_exec(cmd, timeout=10)
        if rc != 0:
            return f"Error: {stderr.strip()}"

        if x is not None and y is not None:
            return f"Clicked button {button} at ({x}, {y})"
        return f"Clicked button {button} at current position"

    async def _mouse_click_subprocess(
        self, x: int | None, y: int | None, button: int
    ) -> str:
        """Mouse click in subprocess mode using host xdotool."""
        if x is not None and y is not None:
            # Move first, then click
            stdout, stderr, rc = await _run(
                ["env", f"DISPLAY={XVFB_DISPLAY}", "xdotool", "mousemove",
                 str(int(x)), str(int(y))],
                timeout=10,
                check=False,
            )
            if rc != 0:
                return f"Error moving mouse: {stderr.strip()}"
            await asyncio.sleep(0.05)

        stdout, stderr, rc = await _run(
            ["env", f"DISPLAY={XVFB_DISPLAY}", "xdotool", "click",
             str(int(button))],
            timeout=10,
            check=False,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"

        if x is not None and y is not None:
            return f"Clicked button {button} at ({x}, {y})"
        return f"Clicked button {button} at current position"

    # ------------------------------------------------------------------
    # Keyboard input
    # ------------------------------------------------------------------

    async def type_text(self, text: str) -> str:
        """Type text using the virtual keyboard."""
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._type_text_docker(text)
        else:
            return await self._type_text_subprocess(text)

    async def _type_text_docker(self, text: str) -> str:
        """Type text in Docker mode (original logic, unchanged)."""
        # Escape single quotes for bash
        escaped = text.replace("'", "'\\''")
        stdout, stderr, rc = await self._docker_exec(
            f"DISPLAY={DISPLAY_NUM} xdotool type -- '{escaped}'",
            timeout=15,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return f"Typed: {text}"

    async def _type_text_subprocess(self, text: str) -> str:
        """Type text in subprocess mode using host xdotool."""
        # xdotool type reads from args, but special characters need care.
        # We pass the text directly as arguments (no shell escaping needed
        # since we use create_subprocess_exec).
        stdout, stderr, rc = await _run(
            ["env", f"DISPLAY={XVFB_DISPLAY}", "xdotool", "type", "--", text],
            timeout=15,
            check=False,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return f"Typed: {text}"

    async def press_key(self, key: str) -> str:
        """Press a key or key combination.

        Examples: ``"Return"``, ``"ctrl+c"``, ``"alt+Tab"``, ``"super"``
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._press_key_docker(key)
        else:
            return await self._press_key_subprocess(key)

    async def _press_key_docker(self, key: str) -> str:
        """Press key in Docker mode (original logic, unchanged)."""
        stdout, stderr, rc = await self._docker_exec(
            f"DISPLAY={DISPLAY_NUM} xdotool key '{key}'",
            timeout=10,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return f"Pressed key: {key}"

    async def _press_key_subprocess(self, key: str) -> str:
        """Press key in subprocess mode using host xdotool."""
        stdout, stderr, rc = await _run(
            ["env", f"DISPLAY={XVFB_DISPLAY}", "xdotool", "key", key],
            timeout=10,
            check=False,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return f"Pressed key: {key}"

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    async def exec_command(self, cmd: str, timeout: float = 30) -> str:
        """Execute a shell command.

        In Docker mode the command runs inside the container.
        In subprocess mode the command runs directly on the host via
        ``asyncio.create_subprocess_exec`` (through bash -c).

        Returns the combined stdout + stderr output.
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._exec_command_docker(cmd, timeout)
        else:
            return await self._exec_command_subprocess(cmd, timeout)

    async def _exec_command_docker(self, cmd: str, timeout: float) -> str:
        """Execute command in Docker mode (original logic, unchanged)."""
        stdout, stderr, rc = await self._docker_exec(cmd, timeout=timeout)

        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            parts.append(f"[stderr] {stderr.strip()}")
        if rc != 0:
            parts.append(f"[exit code: {rc}]")

        return "\n".join(parts) if parts else "(no output)"

    @staticmethod
    def _get_shell_cmd(cmd: str) -> list[str]:
        """Return platform-appropriate shell invocation for host-level commands.

        Docker container commands (where /bin/bash is guaranteed) should
        keep using ["/bin/bash", "-c", cmd] directly.  This helper is only
        for subprocess-fallback host execution.
        """
        if sys.platform == "win32":
            return ["cmd.exe", "/c", cmd]
        return ["/bin/bash", "-c", cmd]

    async def _exec_command_subprocess(self, cmd: str, timeout: float) -> str:
        """Execute command on the host via bash."""
        env = os.environ.copy()
        env["DISPLAY"] = XVFB_DISPLAY
        env["TERM"] = env.get("TERM", "xterm-256color")

        shell_cmd = self._get_shell_cmd(cmd)
        proc = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return ""


        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        rc = proc.returncode if proc.returncode is not None else -1

        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            parts.append(f"[stderr] {stderr.strip()}")
        if rc != 0:
            parts.append(f"[exit code: {rc}]")

        return "\n".join(parts) if parts else "(no output)"

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    async def upload_file(
        self, local_path: str, container_path: str
    ) -> str:
        """Copy a file from the host workspace into the virtual computer.

        *local_path* is relative to the host workspace directory.
        *container_path* is the absolute path inside the container
        (or destination path in subprocess mode).
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._upload_file_docker(local_path, container_path)
        else:
            return await self._upload_file_subprocess(local_path, container_path)

    async def _upload_file_docker(
        self, local_path: str, container_path: str
    ) -> str:
        """Upload file in Docker mode (original logic, unchanged)."""
        host_file = Path(self.workspace_host) / local_path
        if not host_file.exists():
            return f"Error: File not found: {host_file}"

        stdout, stderr, rc = await _run(
            _docker("cp", str(host_file), f"{self.container_name}:{container_path}"),
            timeout=30,
        )
        if rc != 0:
            return f"Error uploading file: {stderr.strip()}"
        return f"Uploaded {local_path} → {container_path}"

    async def _upload_file_subprocess(
        self, local_path: str, container_path: str
    ) -> str:
        """Upload file in subprocess mode (simple file copy).

        In subprocess mode, ``container_path`` is treated as a path inside
        the workspace directory (or an absolute host path).
        """
        host_file = Path(self.workspace_host) / local_path
        if not host_file.exists():
            return f"Error: File not found: {host_file}"

        # Determine destination: if container_path starts with /workspace,
        # map it to the host workspace; otherwise treat as absolute.
        if container_path.startswith("/workspace"):
            rel = container_path[len("/workspace"):]
            if rel.startswith("/"):
                rel = rel[1:]
            dest = self.workspace_host / rel
        else:
            dest = Path(container_path)

        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(str(host_file), str(dest))
        except OSError as exc:
            return f"Error copying file: {exc}"
        return f"Uploaded {local_path} → {container_path}"

    async def download_file(
        self, container_path: str, local_path: str
    ) -> str:
        """Copy a file from the virtual computer to the host workspace.

        *container_path* is the absolute path inside the container
        (or source path in subprocess mode).
        *local_path* is relative to the host workspace directory.
        """
        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._download_file_docker(container_path, local_path)
        else:
            return await self._download_file_subprocess(container_path, local_path)

    async def _download_file_docker(
        self, container_path: str, local_path: str
    ) -> str:
        """Download file in Docker mode (original logic, unchanged)."""
        host_file = Path(self.workspace_host) / local_path
        host_file.parent.mkdir(parents=True, exist_ok=True)

        stdout, stderr, rc = await _run(
            _docker("cp", f"{self.container_name}:{container_path}", str(host_file)),
            timeout=30,
        )
        if rc != 0:
            return f"Error downloading file: {stderr.strip()}"
        return f"Downloaded {container_path} → {local_path}"

    async def _download_file_subprocess(
        self, container_path: str, local_path: str
    ) -> str:
        """Download file in subprocess mode (simple file copy)."""
        # Resolve source: if container_path starts with /workspace, map it
        if container_path.startswith("/workspace"):
            rel = container_path[len("/workspace"):]
            if rel.startswith("/"):
                rel = rel[1:]
            src = self.workspace_host / rel
        else:
            src = Path(container_path)

        if not src.exists():
            return f"Error: File not found: {src}"

        dest = Path(self.workspace_host) / local_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(str(src), str(dest))
        except OSError as exc:
            return f"Error copying file: {exc}"
        return f"Downloaded {container_path} → {local_path}"

    # ------------------------------------------------------------------
    # Package installation
    # ------------------------------------------------------------------

    async def install_package(self, packages: list[str]) -> str:
        """Install apt packages.

        In Docker mode packages are installed inside the container.
        In subprocess mode packages are installed on the host via apt-get
        (requires root/sudo).

        Args:
            packages: List of package names to install.
        """
        if not packages:
            return "Error: No packages specified."

        if self._use_docker is None:
            await self._detect_docker()

        if self._use_docker:
            return await self._install_package_docker(packages)
        else:
            return await self._install_package_subprocess(packages)

    async def _install_package_docker(self, packages: list[str]) -> str:
        """Install packages in Docker mode (original logic, unchanged)."""
        pkg_str = " ".join(packages)
        stdout, stderr, rc = await self._docker_exec_root(
            f'export DEBIAN_FRONTEND=noninteractive && '
            f'apt-get update -qq && '
            f'apt-get install -y --no-install-recommends {pkg_str} 2>&1 && '
            f'apt-get clean',
            timeout=300,
        )

        if rc != 0:
            return f"Error installing packages: {stderr.strip()[-500:]}"
        return f"Installed packages: {pkg_str}"

    async def _install_package_subprocess(self, packages: list[str]) -> str:
        """Install packages on the host system via apt-get."""
        pkg_str = " ".join(packages)

        # Check if we're running as root (Unix-only; Windows falls back to sudo path)
        try:
            is_root = os.geteuid() == 0
        except AttributeError:
            is_root = False

        if is_root:
            cmd = [
                "apt-get", "update", "-qq",
                "&&",
                "DEBIAN_FRONTEND=noninteractive",
                "apt-get", "install", "-y", "--no-install-recommends",
                *packages,
                "&&",
                "apt-get", "clean",
            ]
            # Use bash -c for the chained command
            full_cmd = (
                f"export DEBIAN_FRONTEND=noninteractive && "
                f"apt-get update -qq && "
                f"apt-get install -y --no-install-recommends {pkg_str} 2>&1 && "
                f"apt-get clean"
            )
            stdout, stderr, rc = await _run(
                self._get_shell_cmd(full_cmd),
                timeout=300,
                check=False,
            )
        else:
            # Try with sudo
            full_cmd = (
                f"export DEBIAN_FRONTEND=noninteractive && "
                f"sudo apt-get update -qq && "
                f"sudo apt-get install -y --no-install-recommends {pkg_str} 2>&1 && "
                f"sudo apt-get clean"
            )
            stdout, stderr, rc = await _run(
                self._get_shell_cmd(full_cmd),
                timeout=300,
                check=False,
            )

        if rc != 0:
            error_detail = stderr.strip()[-500:] if stderr.strip() else stdout.strip()[-500:]
            return (
                f"Error installing packages: {error_detail}\n"
                f"Note: In subprocess mode, package installation requires root "
                f"privileges or passwordless sudo."
            )
        return f"Installed packages: {pkg_str}"


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

_vm_instance: VirtualComputer | None = None


def get_vm() -> VirtualComputer:
    """Return the shared VirtualComputer instance."""
    global _vm_instance
    if _vm_instance is None:
        _vm_instance = VirtualComputer()
    return _vm_instance
