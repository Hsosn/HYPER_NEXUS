"""
Monitor and scheduling tools.

Allows Nexus (or the user) to:
- Watch workspace paths for file changes
- Monitor URLs for content changes
- Schedule recurring tasks using natural language
- List and remove active monitors/schedules
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any

from ...memory import db
from ..registry import tool


# ── File Watcher ─────────────────────────────────────────────────────────────

@tool(
    name="watch_path",
    description=(
        "Start watching a workspace path for file changes (created, modified, deleted). "
        "Hyper Nexus will emit events whenever files change. Use for keeping aware of evolving work."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path":  {"type": "string", "description": "Absolute or workspace-relative path to watch"},
            "label": {"type": "string", "description": "Human-readable label for this watch"},
        },
        "required": ["path"],
    },
    category="monitor",
)
async def watch_path(params):
    from ...config import BASE_DIR
    from pathlib import Path
    raw = params.get("path", "")
    p = Path(raw) if Path(raw).is_absolute() else BASE_DIR / "data" / "workspace" / raw
    if not p.exists():
        return f"Path does not exist: {p}"
    wid = await db.add_file_watch(str(p), params.get("label", raw))
    return f"Watching '{p}' (watch #{wid}). File changes will be logged and emitted as events."


@tool(
    name="list_watches",
    description="List all active file watches and web monitors.",
    parameters_schema={"type": "object", "properties": {}, "required": []},
    category="monitor",
)
async def list_watches(_params):
    watches  = await db.list_file_watches(enabled_only=False)
    monitors = await db.list_web_monitors(enabled_only=False)
    lines = ["## File Watches"]
    for w in watches:
        status = "ON" if w.get("enabled") else "OFF"
        lines.append(f"  #{w['id']} [{status}] {w.get('label','')} → {w.get('path','')}")
    lines.append("\n## Web Monitors")
    for m in monitors:
        status = "ON" if m.get("enabled") else "OFF"
        interval = m.get("check_interval_seconds", 3600)
        lines.append(f"  #{m['id']} [{status}] {m.get('label','')} → {m.get('url','')} (every {interval}s)")
    if not watches and not monitors:
        return "No active watches or monitors."
    return "\n".join(lines)


@tool(
    name="remove_watch",
    description="Remove a file watch or web monitor by ID.",
    parameters_schema={
        "type": "object",
        "properties": {
            "watch_id":   {"type": "integer", "description": "File watch ID to remove"},
            "monitor_id": {"type": "integer", "description": "Web monitor ID to remove"},
        },
        "required": [],
    },
    category="monitor",
)
async def remove_watch(params):
    results = []
    if wid := params.get("watch_id"):
        await db.delete_file_watch(int(wid))
        results.append(f"File watch #{wid} removed.")
    if mid := params.get("monitor_id"):
        await db.delete_web_monitor(int(mid))
        results.append(f"Web monitor #{mid} removed.")
    return "\n".join(results) if results else "Nothing removed — provide watch_id or monitor_id."


# ── Web Monitor ───────────────────────────────────────────────────────────────

@tool(
    name="monitor_url",
    description=(
        "Monitor a URL for content changes. Hyper Nexus will check it periodically "
        "and emit an event whenever the page content changes."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "url":               {"type": "string"},
            "label":             {"type": "string", "description": "Human-readable name"},
            "interval_seconds":  {"type": "integer", "default": 3600,
                                  "description": "How often to check (seconds). Min 300."},
        },
        "required": ["url"],
    },
    category="monitor",
)
async def monitor_url(params):
    url      = params.get("url", "")
    label    = params.get("label", url)
    interval = max(300, int(params.get("interval_seconds", 3600)))
    mid = await db.add_web_monitor(url, label, interval)
    return (f"Monitoring '{label}' every {interval}s (monitor #{mid}). "
            f"You'll be notified when content changes.")


# ── NL Scheduler ─────────────────────────────────────────────────────────────

@tool(
    name="schedule",
    description=(
        "Schedule a recurring task using natural language. "
        "Examples: 'every day at 9am', 'every friday at 6pm', 'every hour', 'in 2 hours', "
        "'every monday and wednesday at 10am', 'every 30 minutes'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Natural language schedule expression"},
            "task":       {"type": "string", "description": "What to do when the schedule fires"},
            "goal_id":    {"type": "integer", "description": "Optional goal to attach this schedule to"},
        },
        "required": ["expression", "task"],
    },
    category="monitor",
)
async def schedule(params):
    expr  = params.get("expression", "").strip().lower()
    task  = params.get("task", "")
    gid   = params.get("goal_id")

    next_run = _parse_nl_schedule(expr)
    if next_run is None:
        return (f"Could not parse schedule expression: '{expr}'. "
                f"Try: 'every day at 9am', 'every friday', 'in 2 hours', 'every 30 minutes'.")

    sid = await db.add_nl_schedule(expr, task, next_run, goal_id=gid)
    dt  = datetime.fromtimestamp(next_run).strftime("%Y-%m-%d %H:%M")
    return f"Scheduled '{task}' — expression: '{expr}' — next run: {dt} (schedule #{sid})."


@tool(
    name="list_nl_schedules",
    description="List all active NL schedules.",
    parameters_schema={"type": "object", "properties": {}, "required": []},
    category="monitor",
)
async def list_nl_schedules(_params):
    schedules = await db.list_nl_schedules(enabled_only=False)
    if not schedules:
        return "No schedules configured."
    lines = ["## Schedules"]
    for s in schedules:
        status   = "ON" if s.get("enabled") else "OFF"
        next_dt  = datetime.fromtimestamp(s["next_run"]).strftime("%Y-%m-%d %H:%M") if s.get("next_run") else "?"
        runs     = s.get("run_count", 0)
        lines.append(f"  #{s['id']} [{status}] '{s['expression']}' → {s['task_title']}")
        lines.append(f"         next: {next_dt} | runs: {runs}")
    return "\n".join(lines)


@tool(
    name="remove_nl_schedule",
    description="Remove a schedule by ID.",
    parameters_schema={
        "type": "object",
        "properties": {"schedule_id": {"type": "integer"}},
        "required": ["schedule_id"],
    },
    category="monitor",
)
async def remove_nl_schedule(params):
    sid = int(params["schedule_id"])
    await db.delete_nl_schedule(sid)
    return f"Schedule #{sid} removed."


# ── Improvement Log ───────────────────────────────────────────────────────────

@tool(
    name="improvement_log",
    description="Show Hyper Nexus's self-improvement log — detected failure patterns, stalled goals, and learnings.",
    parameters_schema={
        "type": "object",
        "properties": {
            "include_resolved": {"type": "boolean", "default": False},
        },
        "required": [],
    },
    category="system",
)
async def get_improvement_log(params):
    # Safely coerce the resolved flag — handle string values from query params
    _raw = params.get("include_resolved", False)
    if isinstance(_raw, str):
        resolved = _raw.lower() in ("true", "1", "yes")
    else:
        resolved = bool(_raw)
    entries  = await db.list_improvement_log(resolved=resolved, limit=50)
    if not entries:
        return "No improvement log entries." + (" (all resolved)" if not resolved else "")
    lines = [f"## Self-Improvement Log ({'all' if resolved else 'unresolved'})"]
    for e in entries:
        lines.append(
            f"  #{e['id']} [{e['category']}] {e.get('tool_name','')} "
            f"— freq:{e.get('frequency',1)} — {e.get('detail','')[:120]}"
        )
    return "\n".join(lines)


# ── NL Schedule Parser ────────────────────────────────────────────────────────

_DAYS = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
         "friday":4,"saturday":5,"sunday":6}
_MONTH_SECONDS = 30 * 86400

def _parse_nl_schedule(expr: str) -> float | None:
    """Return Unix timestamp of the next scheduled run, or None if unparseable."""
    now = datetime.now()
    expr = expr.lower().strip()

    # "in N minutes/hours/days"
    m = re.match(r"in (\d+)\s*(minute|hour|day|second)s?", expr)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        deltas = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
        return time.time() + n * deltas[unit]

    # "every N minutes/hours"
    m = re.match(r"every (\d+)\s*(minute|hour|second)s?", expr)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        deltas = {"second": 1, "minute": 60, "hour": 3600}
        return time.time() + n * deltas[unit]

    # "every hour / every day / every week"
    if expr in ("every hour",):
        return time.time() + 3600
    if expr in ("every day", "daily"):
        return time.time() + 86400
    if expr in ("every week", "weekly"):
        return time.time() + 7 * 86400

    # Extract optional time "at HH:MM" or "at Xam/pm"
    time_match = re.search(r"at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?", expr)
    target_hour, target_min = now.hour, 0
    if time_match:
        h = int(time_match.group(1))
        target_min = int(time_match.group(2)) if time_match.group(2) else 0
        period = time_match.group(3)
        if period == "pm" and h != 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        target_hour = h

    # "every day at X"
    if re.search(r"every day|daily", expr):
        candidate = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate.timestamp()

    # "every weekday"
    if "weekday" in expr:
        candidate = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate.timestamp()

    # "every monday", "every friday and wednesday", etc.
    matched_days = [d for d in _DAYS if d in expr]
    if matched_days:
        best = None
        for day_name in matched_days:
            day_num = _DAYS[day_name]
            days_ahead = (day_num - now.weekday()) % 7
            candidate = (now + timedelta(days=days_ahead)).replace(
                hour=target_hour, minute=target_min, second=0, microsecond=0)
            if candidate <= now:
                candidate += timedelta(days=7)
            if best is None or candidate.timestamp() < best:
                best = candidate.timestamp()
        return best

    return None
