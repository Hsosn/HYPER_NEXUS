"""
System introspection tools — comprehensive diagnostics for Hyper Nexus state.
Lets the agent query its own health, performance, and system configuration.
"""
from __future__ import annotations

import gc
import os
import platform
import sys
import time
from typing import Any

from ... import config
from ...events import BUS
from ...memory import db
from ...tasks.task_registry import list_tasks as _list_bg_tasks, cancel_task as _cancel_bg_task
from ..registry import REGISTRY, tool


@tool(
    name="system_status",
    description=(
        "Full system diagnostic: heartbeat, memory, registered tools, goals, "
        "active processes, database health, and module versions."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "detailed": {
                "type": "boolean",
                "default": False,
                "description": "If True, includes memory usage and GC stats",
            },
        },
        "required": [],
    },
    category="system",
    cacheable=True,
    cache_ttl=15,
)
async def system_status(params: dict[str, Any]) -> str:
    detailed = params.get("detailed", False)
    now = time.time()

    # Event bus heartbeat info
    history = BUS.history(limit=300)
    recent_heartbeats = [e for e in history if e.get("kind") == "heartbeat"]
    last_hb = recent_heartbeats[-1] if recent_heartbeats else None
    last_hb_ago = (now - last_hb["timestamp"]) if last_hb else None
    hb_rate = (
        f"active ({sum(1 for e in recent_heartbeats[-30:] if now - e['timestamp'] < 120)} ticks / 2min)"
        if recent_heartbeats else "never fired"
    )

    # Database
    mem_count = await db.count_memories()
    goals = await db.list_goals()

    # Tool registry
    all_tools = REGISTRY.all_tools()

    # System info
    process = psutil_info() if detailed else {}

    lines = [
        "# Hyper Nexus System Status",
        "",
        "## Core",
        f"- Heartbeat: {hb_rate}",
        f"- Interval: {config.get('heartbeat_interval_seconds', 30)}s",
        f"- Last tick: {f'{last_hb_ago:.0f}s ago' if last_hb else 'NEVER'}",
        f"- Reasoning: Nexus Framework (adaptive)",
        f"- Model: {config.get('default_model', 'default')}",
        "",
        "## Memory",
        f"- Long-term memory: {'ON' if config.get('enable_long_term_memory', True) else 'OFF'}",
        f"- Memories stored: {mem_count}",
        "",
        "## Tools & Goals",
        f"- Tools registered: {len(all_tools)}",
        f"- Goals: {len(goals)} ({sum(1 for g in goals if g.get('status')=='pending')} pending, "
        f"{sum(1 for g in goals if g.get('status')=='active')} active, "
        f"{sum(1 for g in goals if (g.get('progress') or 0) >= 1.0)} completed)",
        "",
        "## Features",
        f"- Self-reflection: {'ON' if config.get('enable_self_reflection', True) else 'OFF'}",
        f"- Celery: {'ON' if config.get('celery_broker_url') else 'OFF'}",
        f"- 3D Engine: {'ON' if config.get('enable_3d', True) else 'OFF'}",
    ]

    if detailed:
        lines.extend([
            "",
            "## Process",
            f"- PID: {process.get('pid', os.getpid())}",
            f"- Memory: {process.get('memory_rss', 'N/A')} MB RSS",
            f"- CPU: {process.get('cpu_percent', 'N/A')}%",
            f"- Python: {sys.version.split()[0]}",
            f"- Platform: {platform.system()} {platform.release()}",
            f"- GC objects: {len(gc.get_objects()) if 'gc' in dir() else 'N/A'}",
        ])

    return "\n".join(lines)


def psutil_info() -> dict[str, Any]:
    """Get process resource usage without psutil dependency (fallback to basic)."""
    info: dict[str, Any] = {"pid": os.getpid()}
    try:
        import psutil
        p = psutil.Process()
        mem = p.memory_info()
        info["memory_rss"] = round(mem.rss / (1024 * 1024), 1)
        info["cpu_percent"] = p.cpu_percent(interval=0.1)
        info["memory_percent"] = round(p.memory_percent(), 1)
        info["created"] = p.create_time()
        info["threads"] = p.num_threads()
        info["connections"] = len(p.connections())
    except ImportError:
        # Fallback: basic resource module
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            info["memory_rss"] = round(rss / 1024, 1)  # KB → MB (Linux) or already MB (macOS)
        except (ImportError, AttributeError):
            info["memory_rss"] = "psutil not installed"
    except Exception:
        info["memory_rss"] = "error reading"
    return info


@tool(
    name="recent_events",
    description="List recent internal events from the event bus with optional filtering by kind.",
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 30},
            "kind": {
                "type": "string",
                "description": "Optional filter by event kind (heartbeat, tool_start, tool_end, user_message, agent_message, etc.)",
            },
            "since_seconds": {
                "type": "integer",
                "description": "Only events within this many seconds",
            },
        },
        "required": [],
    },
    category="system",
)
async def recent_events(params: dict[str, Any]) -> str:
    limit = int(params.get("limit", 30))
    kind_filter = params.get("kind")
    since = params.get("since_seconds")

    try:
        history = BUS.history(limit=500)
    except Exception:
        return "(event bus unavailable)"

    if kind_filter:
        history = [e for e in history if e.get("kind") == kind_filter]
    if since:
        cutoff = time.time() - int(since)
        history = [e for e in history if e.get("timestamp", 0) > cutoff]

    history = history[-limit:]
    if not history:
        return "(no matching events)"

    lines = [f"Recent events ({len(history)} shown):", ""]
    for e in history:
        try:
            ts = e.get("timestamp", time.time())
            ago = time.time() - ts if isinstance(ts, (int, float)) else 0
            kind = str(e.get("kind", "?"))
            # Fix: name is inside data dict, not at event top level
            data = e.get("data", {}) or {}
            name = str(data.get("name", e.get("session_id", "")) or "")
            raw_preview = str(data)[:100]
            # Sanitize preview: replace control chars and curly braces (safe for f-strings)
            preview = "".join(c if c.isprintable() else "�" for c in raw_preview)
            lines.append(f"  [{ago:>6.1f}s] {kind:<20} {name:<25} {preview}")
        except Exception:
            continue  # Skip malformed events

    return "\n".join(lines)


@tool(
    name="schedule_task",
    description="Schedule a Celery task to be executed at a future time. The heartbeat will dispatch it when due.",
    parameters_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Task description"},
            "delay_seconds": {
                "type": "integer",
                "description": "Seconds from now until execution",
                "minimum": 5,
            },
            "goal_id": {"type": "integer", "description": "Optional goal ID this task belongs to"},
            "priority": {
                "type": "integer",
                "default": 5,
                "description": "Priority 1-10 (higher = more urgent)",
            },
        },
        "required": ["title", "delay_seconds"],
    },
    category="system",
)
async def schedule_task(params: dict[str, Any]) -> str:
    delay = int(params["delay_seconds"])
    when = time.time() + delay
    priority = int(params.get("priority", 5))

    tid = await db.add_task(
        goal_id=params.get("goal_id"),
        title=params["title"],
        scheduled_for=when,
    )
    return (
        f"Task #{tid} '{params['title']}' scheduled {delay}s from now (priority {priority}). "
        f"Will auto-fire at {time.strftime('%H:%M:%S', time.localtime(when))}."
    )


@tool(
    name="list_schedules",
    description="List all scheduled tasks, NL schedules, and active cron jobs.",
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="system",
    cacheable=True,
    cache_ttl=30,
)
async def list_schedules(_params: dict[str, Any]) -> str:
    now = time.time()
    lines = ["## Scheduled Tasks & Cron Jobs", ""]

    # Pending timed tasks
    try:
        pending = await db.list_tasks(status="pending")
        due_now = [t for t in pending if t.get("scheduled_for", 0) <= now]
        future = [t for t in pending if t.get("scheduled_for", 0) > now]
        if due_now:
            lines.append(f"### Due Now ({len(due_now)})")
            for t in due_now[:10]:
                lines.append(f"  #{t['id']} {t.get('title', '?')} "
                             f"(due {time.strftime('%H:%M', time.localtime(t['scheduled_for']))})")
        if future:
            lines.append(f"### Future ({len(future)})")
            for t in sorted(future, key=lambda x: x.get("scheduled_for", 0))[:15]:
                eta = t.get("scheduled_for", 0) - now
                lines.append(f"  #{t['id']} {t.get('title', '?')} (in {eta/60:.0f}m)")
        if not due_now and not future:
            lines.append("  (no pending tasks)")
    except Exception as e:
        lines.append(f"  Error reading tasks: {e}")

    # NL schedules
    try:
        nl = await db.list_nl_schedules(enabled_only=True)
        if nl:
            lines.extend(["", "### NL Schedules"])
            for s in nl:
                next_str = time.strftime("%H:%M %b %d", time.localtime(s.get("next_run", 0))) if s.get("next_run") else "?"
                lines.append(f"  '{s.get('expression', '?')}' → {s.get('task_title', '?')} (next: {next_str})")
    except Exception:
        pass

    # Active watches
    try:
        watches = await db.list_file_watches(enabled_only=True)
        if watches:
            lines.extend(["", "### File Watches"])
            for w in watches[:10]:
                lines.append(f"  {w.get('path', '?')} (last: {w.get('last_checked', 0):.0f})")
    except Exception:
        pass

    return "\n".join(lines)


@tool(
    name="remove_schedule",
    description="Cancel a scheduled task by its task ID. Also cancels any running background tasks for this schedule.",
    parameters_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "Task ID to cancel"},
        },
        "required": ["task_id"],
    },
    category="system",
)
async def remove_schedule(params: dict[str, Any]) -> str:
    tid = int(params["task_id"])
    try:
        await db.update_task(tid, status="cancelled", updated_at=time.time())
        # Also cancel any running background tasks matching this schedule
        bt_title = f"task_{tid}"
        cancelled_bg = 0
        for bt in _list_bg_tasks():
            if bt_title in bt.get("title", "") or bt.get("description", "") == f"schedule:{tid}":
                if _cancel_bg_task(bt["task_id"]):
                    cancelled_bg += 1
        msg = f"Task #{tid} cancelled."
        if cancelled_bg:
            msg += f" Also cancelled {cancelled_bg} running background task(s)."
        return msg
    except Exception as e:
        return f"Error cancelling task #{tid}: {e}"


@tool(
    name="list_background_tasks",
    description="List all currently running background tasks (periodic jobs, scheduled agent runs, trigger-fired tasks, etc.).",
    parameters_schema={
        "type": "object",
        "properties": {
            "detailed": {
                "type": "boolean",
                "default": False,
                "description": "If True, include full descriptions",
            },
        },
        "required": [],
    },
    category="system",
)
async def list_background_tasks(params: dict[str, Any]) -> str:
    """List all background tasks currently tracked by the system.

    Returns a formatted table of task_id, title, type, age, and status.
    This gives the agent full visibility into what's running behind the scenes.
    """
    detailed = params.get("detailed", False)
    tasks = _list_bg_tasks()
    if not tasks:
        return "(no background tasks running)"

    lines = [f"## Background Tasks ({len(tasks)} running)", ""]
    lines.append(f"  {'Task ID':<22} {'Title':<30} {'Type':<18} {'Age':<8} {'Status'}")
    lines.append(f"  {'-'*22} {'-'*30} {'-'*18} {'-'*8} {'-'*8}")

    for t in tasks:
        tid = t.get("task_id", "")[:20]
        title = t.get("title", "")[:28]
        typ = t.get("type", "")[:16]
        age = f"{t.get('age_seconds', 0):.0f}s"
        status = "done" if t.get("done") else "running"
        lines.append(f"  {tid:<22} {title:<30} {typ:<18} {age:<8} {status}")

    if detailed:
        lines.append("")
        for t in tasks:
            desc = t.get("description", "")
            if desc:
                lines.append(f"  {t['task_id']}: {desc[:200]}")

    return "\n".join(lines)


@tool(
    name="cancel_background_task",
    description="Cancel a specific running background task by its task ID.",
    parameters_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task ID to cancel (use list_background_tasks to get running IDs)",
            },
        },
        "required": ["task_id"],
    },
    category="system",
)
async def cancel_background_task(params: dict[str, Any]) -> str:
    """Cancel a specific background task by ID.

    Cancels the running asyncio task and removes it from the registry.
    Use list_background_tasks to find the task_id.
    """
    tid = params.get("task_id", "")
    if not tid:
        return "Error: task_id is required"
    if _cancel_bg_task(tid):
        return f"Background task {tid} cancelled."
    return f"Task {tid} not found or already completed."
