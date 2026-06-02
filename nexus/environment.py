"""
Environment awareness system (v8 — Comprehensive Runtime Awareness).

Tracks global system state across all sessions, monitors agent health,
performance trends, and resource usage.  Provides contextual awareness
summaries for the system prompt and detects anomalies / alerts.

v8 additions:
- OS & system detection (type, platform, hostname, Python, CPU)
- Runtime context (working directory, process uptime, disk, memory)
- Capabilities detection (available tools, configured API keys, services)
- Filesystem awareness (workspace summary, workspace size)
- Network context (local vs. deployed, port info)
- Rich get_context_for_prompt() for full agent self-awareness
- get_state() now includes all runtime information
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import time
from datetime import datetime, timezone
from typing import Any

from . import config
from .events import emit
from .memory import db

logger = logging.getLogger(__name__)


# ── State tracking ────────────────────────────────────────────────────────────

class EnvironmentState:
    """Tracks the agent's environment state for awareness.

    Unlike reactive systems that only respond to the current message,
    this system maintains a global picture of:
    - What OS / runtime the agent is executing on
    - What goals exist and their progress
    - How many memories are stored and their health
    - Tool performance trends and which tools are available
    - Which API keys and external services are configured
    - Recent activity patterns
    - Quality trends over time
    - Resource usage patterns (disk, memory, CPU)
    - Filesystem layout and workspace awareness
    - Network context (local vs deployed)
    """

    def __init__(self) -> None:
        self._last_state: dict[str, Any] = {}
        self._state_update_interval = 60  # seconds
        self._last_refresh: float = 0.0
        self._anomaly_flags: list[dict] = []
        self._process_start_time: float = time.time()
        self._memory_tree_stats: dict = {}

    # ===================================================================
    # State computation
    # ===================================================================

    async def get_state(self, force_refresh: bool = False) -> dict[str, Any]:
        """Compute a comprehensive environment state snapshot.

        This is called by the reasoning engine to build context-aware
        system prompts, and by the heartbeat for monitoring.
        """
        now = time.time()
        if not force_refresh and (now - self._last_refresh) < self._state_update_interval:
            return self._last_state

        self._last_refresh = now
        state: dict[str, Any] = {}

        # ── OS & System ──────────────────────────────────────────────
        state["system"] = self._detect_system()

        # ── Runtime Context ──────────────────────────────────────────
        state["runtime"] = self._detect_runtime()

        # ── Capabilities ─────────────────────────────────────────────
        state["capabilities"] = await self._detect_capabilities()

        # ── Filesystem ───────────────────────────────────────────────
        state["filesystem"] = self._detect_filesystem()

        # ── Network ──────────────────────────────────────────────────
        state["network"] = self._detect_network()

        # ── Goals Overview ───────────────────────────────────────────
        all_goals = await db.list_goals()
        state["goals"] = {
            "total": len(all_goals),
            "pending": sum(1 for g in all_goals if g.get("status") == "pending"),
            "active": sum(1 for g in all_goals if g.get("status") == "active"),
            "done": sum(1 for g in all_goals if g.get("status") == "done"),
            "failed": sum(1 for g in all_goals if g.get("status") == "failed"),
            "avg_progress": self._avg_goal_progress(all_goals),
            "stale_goals": self._count_stale_goals(all_goals),
        }

        # ── Tasks Overview ───────────────────────────────────────────
        all_tasks = await db.list_tasks()
        state["tasks"] = {
            "total": len(all_tasks),
            "pending": sum(1 for t in all_tasks if t.get("status") == "pending"),
            "active": sum(1 for t in all_tasks if t.get("status") == "active"),
            "done": sum(1 for t in all_tasks if t.get("status") == "done"),
            "failed": sum(1 for t in all_tasks if t.get("status") == "failed"),
        }

        # ── Memory Health ────────────────────────────────────────────
        memories = await db.all_memories(limit=1000)
        state["memory"] = {
            "total": len(memories),
            "factual": sum(1 for m in memories if m.get("kind") == "factual"),
            "semantic": sum(1 for m in memories if m.get("kind") == "semantic"),
            "procedural": sum(1 for m in memories if m.get("kind") == "procedural"),
            "episodic": sum(1 for m in memories if m.get("kind") == "episodic"),
            "learned_strategies": sum(
                1 for m in memories
                if "Learned Strategy" in (m.get("content") or "")
            ),
            "avg_importance": self._avg_memory_importance(memories),
        }

        # ── Memory Tree Stats ────────────────────────────────────────
        try:
            from .memory.memory_tree import MemoryTree
            tree = MemoryTree()
            self._memory_tree_stats = await tree.get_stats()
        except Exception:
            self._memory_tree_stats = {}
        state["memory_tree"] = self._memory_tree_stats

        # ── Tool Intelligence ────────────────────────────────────────
        try:
            tool_profiles = await db.get_tool_profiles()
            state["tools"] = {
                "total_profiles": len(tool_profiles),
                "avg_effectiveness": self._avg_tool_effectiveness(tool_profiles),
                "underperforming": [
                    p["tool_name"] for p in tool_profiles
                    if (p.get("success_rate") or 1) < 0.5 and (p.get("total_calls") or 0) >= 3
                ],
                "top_performers": [
                    p["tool_name"] for p in tool_profiles[:5]
                    if (p.get("effectiveness_score") or 0) >= 0.7
                ],
            }
        except Exception as exc:
            logger.warning("get_state: tool_profiles unavailable (%s), using empty defaults", exc)
            state["tools"] = {
                "total_profiles": 0,
                "avg_effectiveness": 0.0,
                "underperforming": [],
                "top_performers": [],
            }

        # ── Quality Trends ───────────────────────────────────────────
        quality_scores = await db.recent_task_scores(limit=20)
        state["quality"] = {
            "recent_avg": self._avg_score(quality_scores),
            "trend": self._quality_trend(quality_scores),
            "approvals": sum(1 for s in quality_scores if s.get("approved")),
            "rejections": sum(1 for s in quality_scores if not s.get("approved")),
        }

        # ── Activity ─────────────────────────────────────────────────
        recent_execs = await db.recent_tool_executions(limit=100)
        state["activity"] = {
            "recent_tool_calls": len(recent_execs),
            "last_activity": self._last_activity_time(recent_execs),
            "sessions_count": len(await db.list_sessions()),
        }

        # ── Anomaly Detection ────────────────────────────────────────
        self._anomaly_flags = self._detect_anomalies(state)
        state["anomalies"] = self._anomaly_flags

        # ── Background Systems ───────────────────────────────────────
        state["systems"] = {
            "file_watches": len(await db.list_file_watches()),
            "web_monitors": len(await db.list_web_monitors()),
            "schedules": len(await db.list_nl_schedules()),
            "notifications_unread": await db.unread_notification_count(),
        }

        self._last_state = state
        return state

    # ===================================================================
    # Context for prompt — the rich self-awareness string
    # ===================================================================

    def get_context_for_prompt(self) -> str:
        """Generate environment context for injection into the system prompt.

        Makes the agent fully aware of its environment — OS, runtime,
        workspace, capabilities, active goals, memory stats, quality,
        and any anomalies — enabling proactive, context-grounded responses.
        """
        state = self._last_state
        if not state:
            return ""

        lines = ["# Environment Awareness"]

        # ── System & OS ──────────────────────────────────────────────
        sys_info = state.get("system", {})
        if sys_info:
            os_name = sys_info.get("os_type", "unknown")
            os_release = sys_info.get("os_release", "")
            machine = sys_info.get("machine", "")
            hostname = sys_info.get("hostname", "")
            py_version = sys_info.get("python_version", "")
            cpu_count = sys_info.get("cpu_count", 0)

            parts = [f"OS: {os_name}"]
            if os_release:
                parts.append(f"Release: {os_release}")
            if machine:
                parts.append(f"Arch: {machine}")
            if hostname:
                parts.append(f"Host: {hostname}")
            lines.append(f"- System: {' | '.join(parts)}")
            if py_version:
                lines.append(f"- Python: {py_version} ({cpu_count} CPU cores detected)")

        # ── Runtime ──────────────────────────────────────────────────
        runtime = state.get("runtime", {})
        if runtime:
            workdir = runtime.get("working_directory", "")
            uptime_s = runtime.get("uptime_seconds", 0)
            if uptime_s > 0:
                uptime_str = self._format_duration(uptime_s)
                lines.append(f"- Process uptime: {uptime_str}")

            disk_total = runtime.get("disk_total_gb", 0)
            disk_free = runtime.get("disk_free_gb", 0)
            disk_pct = runtime.get("disk_usage_percent", 0)
            if disk_total > 0:
                lines.append(
                    f"- Disk: {disk_free:.1f} GB free / {disk_total:.1f} GB total ({disk_pct:.0f}% used)"
                )

            mem_info = runtime.get("memory_usage", {})
            if mem_info and mem_info.get("rss_mb"):
                lines.append(
                    f"- Memory: RSS {mem_info['rss_mb']:.0f} MB"
                    + (f", VMS {mem_info['vms_mb']:.0f} MB" if mem_info.get("vms_mb") else "")
                )

            now_utc = runtime.get("timestamp_utc", "")
            if now_utc:
                lines.append(f"- Current time (UTC): {now_utc}")

        # ── Filesystem / Workspace ───────────────────────────────────
        fs = state.get("filesystem", {})
        if fs:
            workspace = fs.get("workspace_directory", "")
            if workspace:
                lines.append(f"- Workspace: {workspace}")
            ws_size = fs.get("workspace_size_mb", 0)
            if ws_size > 0:
                lines.append(f"- Workspace size: {ws_size:.1f} MB")
            item_count = fs.get("workspace_item_count", 0)
            if item_count > 0:
                lines.append(f"- Workspace items: {item_count}")
            top_dirs = fs.get("top_directories", [])
            if top_dirs:
                lines.append(f"- Workspace layout: {', '.join(top_dirs[:6])}")

        # ── Network ──────────────────────────────────────────────────
        net = state.get("network", {})
        if net:
            env_type = net.get("environment_type", "unknown")
            lines.append(f"- Environment: {env_type}")
            if net.get("is_docker"):
                lines.append("- Running inside Docker container")
            if net.get("detected_ports"):
                lines.append(f"- Detected services on ports: {', '.join(str(p) for p in net['detected_ports'][:8])}")

        # ── Capabilities ─────────────────────────────────────────────
        caps = state.get("capabilities", {})
        if caps:
            configured_keys = caps.get("configured_api_keys", [])
            if configured_keys:
                lines.append(f"- Configured API keys: {', '.join(configured_keys)}")
            else:
                lines.append("- Configured API keys: none")

            available_tools = caps.get("available_tools_summary", {})
            if available_tools:
                tool_items = []
                for category, count in available_tools.items():
                    if count > 0:
                        tool_items.append(f"{category}({count})")
                if tool_items:
                    lines.append(f"- Available tools: {', '.join(tool_items)}")

            vision_model = caps.get("vision_model", "")
            if vision_model:
                lines.append(f"- Vision model: {vision_model} (you can process images)")

            services = caps.get("external_services", {})
            ready = [name for name, status in services.items() if status == "ready"]
            missing = [name for name, status in services.items() if status == "missing"]
            if ready:
                lines.append(f"- Ready services: {', '.join(ready)}")
            if missing:
                lines.append(f"- Missing services: {', '.join(missing)}")

        # ── Goals status ─────────────────────────────────────────────
        goals = state.get("goals", {})
        if goals.get("total", 0) > 0:
            active = goals.get("active", 0)
            pending = goals.get("pending", 0)
            done = goals.get("done", 0)
            failed = goals.get("failed", 0)
            progress = goals.get("avg_progress", 0)
            lines.append(
                f"- Goals: {active} active, {pending} pending, {done} done, {failed} failed "
                f"(avg progress: {progress:.0%})"
            )
            if goals.get("stale_goals", 0) > 0:
                lines.append(
                    f"- WARNING: {goals['stale_goals']} goal(s) have been stale for >24h"
                )

        # ── Quality awareness ────────────────────────────────────────
        quality = state.get("quality", {})
        avg_q = quality.get("recent_avg", 0)
        if avg_q > 0:
            trend = quality.get("trend", "stable")
            approvals = quality.get("approvals", 0)
            rejections = quality.get("rejections", 0)
            lines.append(
                f"- Response quality: {avg_q:.1f}/10 (trend: {trend}, "
                f"{approvals} approved, {rejections} rejected)"
            )
            if trend == "declining":
                lines.append("- ACTION NEEDED: Quality is declining — be more careful and deliberate")

        # ── Memory awareness ─────────────────────────────────────────
        memory = state.get("memory", {})
        total_mem = memory.get("total", 0)
        if total_mem > 0:
            learned = memory.get("learned_strategies", 0)
            factual = memory.get("factual", 0)
            semantic = memory.get("semantic", 0)
            procedural = memory.get("procedural", 0)
            episodic = memory.get("episodic", 0)
            avg_imp = memory.get("avg_importance", 0)
            lines.append(
                f"- Memory: {total_mem} total "
                f"(factual: {factual}, semantic: {semantic}, "
                f"procedural: {procedural}, episodic: {episodic})"
            )
            if learned > 0:
                lines.append(f"- Learned strategies available: {learned}")
            if avg_imp > 0:
                lines.append(f"- Average memory importance: {avg_imp:.2f}")

        # ── Memory Tree awareness ────────────────────────────────────
        tree_stats = state.get("memory_tree", {})
        if tree_stats.get("roots", 0) > 0:
            lines.append(
                f"- Memory Tree: {tree_stats['roots']} roots, {tree_stats['nodes']} nodes, "
                f"{tree_stats['chunks']} unique chunks, {tree_stats['entities']} entities"
            )
            if tree_stats.get("pending_jobs", 0) > 0:
                lines.append(f"- Memory Tree jobs pending: {tree_stats['pending_jobs']}")

        # ── Activity ─────────────────────────────────────────────────
        activity = state.get("activity", {})
        if activity:
            recent_calls = activity.get("recent_tool_calls", 0)
            sessions = activity.get("sessions_count", 0)
            if recent_calls > 0:
                lines.append(f"- Recent tool calls (last 100): {recent_calls}")
            if sessions > 0:
                lines.append(f"- Total sessions: {sessions}")

        # ── Tool intelligence summary ────────────────────────────────
        tools = state.get("tools", {})
        underperforming = tools.get("underperforming", [])
        if underperforming:
            lines.append(f"- Underperforming tools (success < 50%): {', '.join(underperforming[:5])}")
        top = tools.get("top_performers", [])
        if top:
            lines.append(f"- Top-performing tools: {', '.join(top[:5])}")

        # ── Background Systems ───────────────────────────────────────
        systems = state.get("systems", {})
        active_systems = []
        if systems.get("file_watches", 0) > 0:
            active_systems.append(f"{systems['file_watches']} file watches")
        if systems.get("web_monitors", 0) > 0:
            active_systems.append(f"{systems['web_monitors']} web monitors")
        if systems.get("schedules", 0) > 0:
            active_systems.append(f"{systems['schedules']} schedules")
        if active_systems:
            lines.append(f"- Background systems: {', '.join(active_systems)}")
        unread = systems.get("notifications_unread", 0)
        if unread > 0:
            lines.append(f"- Unread notifications: {unread}")

        # ── Anomalies ────────────────────────────────────────────────
        anomalies = state.get("anomalies", [])
        for anomaly in anomalies[:5]:
            severity = anomaly.get("severity", "info").upper()
            lines.append(f"- ANOMALY [{severity}]: {anomaly.get('message', '')}")

        return "\n".join(lines)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive state data for the metrics dashboard."""
        return self._last_state or {}

    # ===================================================================
    # OS & System Detection
    # ===================================================================

    @staticmethod
    def _detect_system() -> dict[str, Any]:
        """Detect operating system, platform, hostname, Python, CPU."""
        info: dict[str, Any] = {}

        try:
            info["os_type"] = platform.system()  # Linux, Darwin, Windows, etc.
        except Exception:
            info["os_type"] = "unknown"

        try:
            info["os_release"] = platform.release()
        except Exception:
            info["os_release"] = ""

        try:
            info["os_version"] = platform.version()
        except Exception:
            info["os_version"] = ""

        try:
            info["platform"] = platform.platform(aliased=True, terse=True)
        except Exception:
            info["platform"] = ""

        try:
            info["machine"] = platform.machine()  # x86_64, aarch64, etc.
        except Exception:
            info["machine"] = ""

        try:
            info["processor"] = platform.processor()
        except Exception:
            info["processor"] = ""

        try:
            info["hostname"] = platform.node()
        except Exception:
            info["hostname"] = ""

        try:
            info["python_version"] = platform.python_version()
        except Exception:
            info["python_version"] = ""

        try:
            info["python_implementation"] = platform.python_implementation()
        except Exception:
            info["python_implementation"] = ""

        try:
            cpu = os.cpu_count()
            info["cpu_count"] = cpu if cpu is not None else 0
        except Exception:
            info["cpu_count"] = 0

        return info

    # ===================================================================
    # Runtime Context
    # ===================================================================

    def _detect_runtime(self) -> dict[str, Any]:
        """Detect working directory, process uptime, disk usage, memory."""
        info: dict[str, Any] = {}

        # Working directory
        try:
            info["working_directory"] = os.getcwd()
        except Exception:
            info["working_directory"] = ""

        # Process uptime
        try:
            info["uptime_seconds"] = time.time() - self._process_start_time
        except Exception:
            info["uptime_seconds"] = 0

        # Current timestamp
        try:
            info["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            info["timestamp_utc"] = ""

        # Disk usage (root or workspace)
        try:
            target_path = str(config.BASE_DIR) if hasattr(config, "BASE_DIR") else "/"
            if not os.path.exists(target_path):
                target_path = "/"
            usage = shutil.disk_usage(target_path)
            info["disk_total_gb"] = usage.total / (1024 ** 3)
            info["disk_used_gb"] = usage.used / (1024 ** 3)
            info["disk_free_gb"] = usage.free / (1024 ** 3)
            info["disk_usage_percent"] = (usage.used / usage.total * 100) if usage.total > 0 else 0
        except Exception:
            info["disk_total_gb"] = 0
            info["disk_used_gb"] = 0
            info["disk_free_gb"] = 0
            info["disk_usage_percent"] = 0

        # Memory usage (process-level RSS / VMS)
        info["memory_usage"] = self._detect_process_memory()

        return info

    @staticmethod
    def _detect_process_memory() -> dict[str, Any]:
        """Attempt to read process RSS / VMS using /proc (Linux) or psutil."""
        mem: dict[str, Any] = {"rss_mb": 0, "vms_mb": 0, "method": "none"}

        # Try /proc/self/status first (Linux)
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        mem["rss_mb"] = kb / 1024
                        mem["method"] = "proc"
                    elif line.startswith("VmSize:"):
                        kb = int(line.split()[1])
                        mem["vms_mb"] = kb / 1024
            if mem["method"] == "proc":
                return mem
        except Exception:
            pass

        # Try psutil as fallback
        try:
            import resource  # noqa: F811 — stdlib, safe
            ru = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss on Linux is in KB; on macOS it's in bytes
            maxrss = ru.ru_maxrss
            if platform.system() == "Linux":
                mem["rss_mb"] = maxrss / 1024
            else:
                mem["rss_mb"] = maxrss / (1024 * 1024)
            mem["method"] = "resource"
        except Exception:
            pass

        return mem

    # ===================================================================
    # Capabilities Detection
    # ===================================================================

    @staticmethod
    async def _detect_capabilities() -> dict[str, Any]:
        """Detect which tools, API keys, and external services are available."""
        caps: dict[str, Any] = {}

        # API keys configured?
        configured_keys = []
        key_checks = [
            ("openrouter_api_key", "OpenRouter"),
            ("openai_api_key", "OpenAI"),
            ("custom_api_key", "Custom/Local LLM"),
            ("groq_api_key", "Groq"),
            ("github_token", "GitHub"),
        ]
        for setting_key, label in key_checks:
            val = config.get(setting_key, "")
            if val and val.strip():
                configured_keys.append(label)
        caps["configured_api_keys"] = configured_keys

        # Vision model configured?
        vision_model = config.get("vision_model", "")
        caps["vision_model"] = vision_model if vision_model and vision_model.strip() else ""

        # Which tools are actually available (from registry)?
        try:
            from .tools.registry import REGISTRY
            all_tools = REGISTRY.all_tools()
            category_counts: dict[str, int] = {}
            for t in all_tools:
                cat = t.category or "general"
                category_counts[cat] = category_counts.get(cat, 0) + 1
            caps["available_tools_summary"] = category_counts
            caps["total_tools_registered"] = len(all_tools)
            caps["tool_names"] = [t.name for t in all_tools]
        except Exception:
            caps["available_tools_summary"] = {}
            caps["total_tools_registered"] = 0
            caps["tool_names"] = []

        # External services readiness
        services: dict[str, str] = {}

        # LLM provider
        provider = config.get("provider", "openrouter")
        has_key = bool(config.get(f"{provider}_api_key", ""))
        if provider == "custom" and config.get("custom_no_auth", False):
            has_key = True
        services["LLM Provider"] = "ready" if has_key else "no_key"
        services["LLM Provider (type)"] = provider

        # Database — SQLite via aiosqlite
        try:
            from .memory import database as _db
            if _db._db is not None:
                services["Database"] = "ready"
            else:
                services["Database"] = "not initialized"
        except Exception:
            services["Database"] = "missing"

        # Celery / Redis
        if config.get("enable_celery", False):
            try:
                import redis  # type: ignore[import-untyped]
                r_url = config.get("redis_url", "redis://localhost:6379/0")
                r = redis.from_url(r_url, socket_connect_timeout=2)
                r.ping()
                services["Redis"] = "ready"
                r.close()
            except Exception:
                services["Redis"] = "missing"
        else:
            services["Redis"] = "disabled"

        # Git
        try:
            import shutil as _shutil
            git_path = _shutil.which("git")
            services["Git CLI"] = "ready" if git_path else "missing"
        except Exception:
            services["Git CLI"] = "missing"

        # Browser (Playwright / Chromium)
        try:
            import shutil as _shutil
            playwright = _shutil.which("playwright")
            chromium = _shutil.which("chromium") or _shutil.which("chromium-browser") or _shutil.which("google-chrome")
            services["Browser"] = "ready" if (playwright or chromium) else "missing"
        except Exception:
            services["Browser"] = "missing"

        # Docker
        try:
            import shutil as _shutil
            docker = _shutil.which("docker")
            services["Docker"] = "ready" if docker else "missing"
        except Exception:
            services["Docker"] = "missing"

        caps["external_services"] = services

        return caps

    # ===================================================================
    # Filesystem Awareness
    # ===================================================================

    def _detect_filesystem(self) -> dict[str, Any]:
        """Detect workspace layout, item counts, and sizes."""
        fs: dict[str, Any] = {}

        # Workspace directory
        try:
            ws = config.BASE_DIR / "data" / "workspace"
            fs["workspace_directory"] = str(ws)
        except Exception:
            fs["workspace_directory"] = ""

        # Workspace size
        try:
            ws_path = config.BASE_DIR / "data" / "workspace"
            if os.path.exists(ws_path):
                total_bytes = 0
                count = 0
                for dirpath, dirnames, filenames in os.walk(ws_path):
                    # Skip hidden directories, node_modules, .git, __pycache__, venv
                    dirnames[:] = [
                        d for d in dirnames
                        if not d.startswith((".", "__"))
                        and d not in ("node_modules", "venv", ".venv", "env")
                    ]
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        try:
                            total_bytes += os.path.getsize(fp)
                            count += 1
                        except OSError:
                            pass
                fs["workspace_size_mb"] = total_bytes / (1024 * 1024)
                fs["workspace_item_count"] = count
            else:
                fs["workspace_size_mb"] = 0
                fs["workspace_item_count"] = 0
        except Exception:
            fs["workspace_size_mb"] = 0
            fs["workspace_item_count"] = 0

        # Top-level directory summary
        try:
            ws_path = config.BASE_DIR / "data" / "workspace"
            if os.path.exists(ws_path):
                top_dirs = []
                for entry in sorted(os.listdir(ws_path)):
                    full = os.path.join(ws_path, entry)
                    if os.path.isdir(full) and not entry.startswith((".", "__")):
                        top_dirs.append(entry)
                fs["top_directories"] = top_dirs[:20]
            else:
                fs["top_directories"] = []
        except Exception:
            fs["top_directories"] = []

        # Data directory size
        try:
            data_dir = config.DATA_DIR
            if os.path.exists(data_dir):
                data_bytes = 0
                for dirpath, dirnames, filenames in os.walk(data_dir):
                    for f in filenames:
                        try:
                            data_bytes += os.path.getsize(os.path.join(dirpath, f))
                        except OSError:
                            pass
                fs["data_dir_size_mb"] = data_bytes / (1024 * 1024)
            else:
                fs["data_dir_size_mb"] = 0
        except Exception:
            fs["data_dir_size_mb"] = 0

        return fs

    # ===================================================================
    # Network Context
    # ===================================================================

    @staticmethod
    def _detect_network() -> dict[str, Any]:
        """Detect whether running locally or deployed, port info."""
        net: dict[str, Any] = {}

        # Docker detection
        is_docker = False
        try:
            if os.path.exists("/.dockerenv"):
                is_docker = True
            elif os.path.exists("/proc/1/cgroup"):
                try:
                    with open("/proc/1/cgroup", "r") as f:
                        for line in f:
                            if "docker" in line.lower() or "kubepods" in line.lower():
                                is_docker = True
                                break
                except Exception:
                    pass
        except Exception:
            pass
        net["is_docker"] = is_docker

        # Environment type heuristic
        env_type = "local"
        env_indicators = [
            ("VERCEL", "Vercel"),
            ("RENDER", "Render"),
            ("RAILWAY", "Railway"),
            ("FLY_APP_NAME", "Fly.io"),
            ("HEROKU_APP_NAME", "Heroku"),
            ("AWS_LAMBDA_FUNCTION_NAME", "AWS Lambda"),
            ("GOOGLE_CLOUD_PROJECT", "GCP"),
            ("AZURE_FUNCTIONS_ENVIRONMENT", "Azure Functions"),
            ("KUBERNETES_SERVICE_HOST", "Kubernetes"),
            ("DIGITALOCEAN_APP_HOSTNAME", "DigitalOcean App Platform"),
        ]
        for env_var, label in env_indicators:
            if os.environ.get(env_var):
                env_type = f"deployed ({label})"
                break

        if is_docker and env_type == "local":
            # Could be a local Docker container or deployed
            env_type = "containerized (likely local or CI)"

        net["environment_type"] = env_type

        # Detected ports — check if well-known services are listening
        detected_ports: list[int] = []
        port_checks = [
            (3000, "127.0.0.1"),   # common dev server
            (5000, "127.0.0.1"),   # Flask default
            (8000, "127.0.0.1"),   # common API server
            (8080, "127.0.0.1"),   # alternative HTTP
            (6379, "127.0.0.1"),   # Redis
            (5432, "127.0.0.1"),   # PostgreSQL (legacy, superseded by SQLite)
            (8765, "127.0.0.1"),   # Nexus WebUI default
            (27017, "127.0.0.1"),  # MongoDB
        ]
        for port, host in port_checks:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.1)
                result = s.connect_ex((host, port))
                s.close()
                if result == 0:
                    detected_ports.append(port)
            except Exception:
                pass

        net["detected_ports"] = detected_ports

        # Host IP addresses
        try:
            hostname = socket.gethostname()
            try:
                ip = socket.gethostbyname(hostname)
                net["host_ip"] = ip
            except Exception:
                net["host_ip"] = "unknown"
        except Exception:
            net["host_ip"] = "unknown"

        return net

    # ===================================================================
    # Internal helpers
    # ===================================================================

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into a human-readable duration string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        hours = seconds / 3600
        if hours < 48:
            return f"{hours:.1f}h"
        return f"{hours / 24:.1f} days"

    @staticmethod
    def _avg_goal_progress(goals: list[dict]) -> float:
        active = [g for g in goals if g.get("status") in ("active", "pending")]
        if not active:
            return 0.0
        return sum(g.get("progress", 0) or 0 for g in active) / len(active)

    @staticmethod
    def _count_stale_goals(goals: list[dict]) -> int:
        now = time.time()
        stale_threshold = now - 86400  # 24h
        return sum(
            1 for g in goals
            if g.get("status") in ("active", "pending")
            and (g.get("updated_at") or 0) < stale_threshold
        )

    @staticmethod
    def _avg_memory_importance(memories: list[dict]) -> float:
        if not memories:
            return 0.0
        return sum(m.get("importance", 0) or 0 for m in memories) / len(memories)

    @staticmethod
    def _avg_tool_effectiveness(profiles: list[dict]) -> float:
        if not profiles:
            return 0.0
        scores = [p.get("effectiveness_score") or 0 for p in profiles]
        return sum(scores) / len(scores)

    @staticmethod
    def _avg_score(scores: list[dict]) -> float:
        if not scores:
            return 0.0
        return sum(s.get("score", 0) or 0 for s in scores) / len(scores)

    @staticmethod
    def _quality_trend(scores: list[dict]) -> str:
        if len(scores) < 6:
            return "stable"
        mid = len(scores) // 2
        first = sum(s.get("score", 0) or 0 for s in scores[:mid]) / mid
        second = sum(s.get("score", 0) or 0 for s in scores[mid:]) / (len(scores) - mid)
        diff = second - first
        if diff > 0.5:
            return "improving"
        elif diff < -0.5:
            return "declining"
        return "stable"

    @staticmethod
    def _last_activity_time(executions: list[dict]) -> float:
        if not executions:
            return 0
        return executions[0].get("created_at", 0) if executions else 0

    def _detect_anomalies(self, state: dict) -> list[dict]:
        """Detect unusual conditions that need attention."""
        anomalies = []
        now = time.time()

        # Stale goals
        stale = state.get("goals", {}).get("stale_goals", 0)
        if stale > 0:
            anomalies.append({
                "type": "stale_goals",
                "severity": "warning",
                "message": f"{stale} goal(s) inactive for >24h",
            })

        # Quality declining
        quality = state.get("quality", {})
        if quality.get("trend") == "declining" and quality.get("recent_avg", 10) < 6:
            anomalies.append({
                "type": "quality_decline",
                "severity": "critical",
                "message": f"Response quality declining (avg: {quality.get('recent_avg', 0):.1f}/10)",
            })

        # Too many pending tasks
        pending_tasks = state.get("tasks", {}).get("pending", 0)
        if pending_tasks > 10:
            anomalies.append({
                "type": "task_backlog",
                "severity": "warning",
                "message": f"{pending_tasks} pending tasks — consider prioritizing",
            })

        # Underperforming tools
        underperforming = state.get("tools", {}).get("underperforming", [])
        if len(underperforming) >= 3:
            anomalies.append({
                "type": "tool_health",
                "severity": "warning",
                "message": f"{len(underperforming)} tools have low success rates",
            })

        # Memory bloat
        total_memories = state.get("memory", {}).get("total", 0)
        if total_memories > 500:
            anomalies.append({
                "type": "memory_bloat",
                "severity": "info",
                "message": f"{total_memories} memories stored — consider cleanup",
            })

        # Inactivity
        last_activity = state.get("activity", {}).get("last_activity", 0)
        if last_activity > 0 and (now - last_activity) > 3600:
            anomalies.append({
                "type": "inactivity",
                "severity": "info",
                "message": f"No tool activity for {(now - last_activity) / 60:.0f} minutes",
            })

        # Disk space low (< 10% free)
        disk_pct = state.get("runtime", {}).get("disk_usage_percent", 0)
        if disk_pct > 90:
            anomalies.append({
                "type": "disk_space",
                "severity": "critical",
                "message": f"Disk usage at {disk_pct:.0f}% — storage critically low",
            })
        elif disk_pct > 80:
            anomalies.append({
                "type": "disk_space",
                "severity": "warning",
                "message": f"Disk usage at {disk_pct:.0f}% — consider cleanup",
            })

        # No API keys configured
        configured = state.get("capabilities", {}).get("configured_api_keys", [])
        if not configured:
            anomalies.append({
                "type": "no_api_keys",
                "severity": "critical",
                "message": "No API keys configured — LLM provider will not work",
            })

        # LLM provider not ready
        services = state.get("capabilities", {}).get("external_services", {})
        llm_status = services.get("LLM Provider", "")
        if llm_status == "no_key":
            anomalies.append({
                "type": "llm_unavailable",
                "severity": "critical",
                "message": "LLM provider has no API key — agent cannot generate responses",
            })

        return anomalies


# Global singleton
ENVIRONMENT = EnvironmentState()
