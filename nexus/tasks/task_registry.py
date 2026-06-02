"""
Background task registry — tracks running asyncio tasks so the agent can query
and cancel them. This is a standalone module (no imports from heartbeat or
scheduler_tasks) to avoid circular imports.

Usage:
    from .task_registry import register_task, cancel_task, list_tasks

    # Register when creating a task:
    task = asyncio.create_task(my_coro())
    register_task(task, title="My task", type="periodic")

    # List all running tasks:
    for t in list_tasks():
        print(t["task_id"], t["title"], t["done"])

    # Cancel by ID:
    cancel_task("bt_1234567890_1")
"""
from __future__ import annotations

import time
import threading
from typing import Any

# ── Background Task Registry ───────────────────────────────────────────────
# Tracks all running background asyncio tasks so the agent can:
# 1. See what's running via the list_background_tasks tool
# 2. Cancel specific tasks by ID

_registry: dict[str, dict] = {}
_registry_lock = threading.Lock()
_counter: int = 0


def register_task(
    task: Any,
    title: str,
    task_type: str,
    description: str = "",
) -> str:
    """Register a background task and return its ID.

    Args:
        task: The asyncio.Task object.
        title: Short human-readable title.
        task_type: Category (e.g. 'periodic', 'scheduled', 'trigger', 'autorun').
        description: Longer description of what this task does.

    Returns:
        A unique task ID string like "bt_1234567890_1".
    """
    global _counter
    _counter += 1
    tid = f"bt_{int(time.time())}_{_counter}"
    now = time.time()
    # Prune stale completed entries to prevent unbounded growth
    stale = [k for k, v in list(_registry.items()) if v.get("done") and now - v.get("done_at", now) > 60]
    for k in stale:
        _registry.pop(k, None)
    with _registry_lock:
        _registry[tid] = {
            "task": task,
            "title": title,
            "type": task_type,
            "description": description,
            "created_at": time.time(),
            "done": task.done(),
        }
    task.add_done_callback(lambda _t: _mark_done(tid))
    return tid


def _mark_done(tid: str) -> None:
    """Mark a background task as done (called by done_callback)."""
    with _registry_lock:
        entry = _registry.get(tid)
        if entry:
            entry["done"] = True
            entry["done_at"] = time.time()


def unregister_task(tid: str) -> bool:
    """Remove a task from the registry entirely."""
    with _registry_lock:
        return bool(_registry.pop(tid, None))


def cancel_task(tid: str) -> bool:
    """Cancel a specific background task by ID. Returns True if cancelled."""
    with _registry_lock:
        entry = _registry.get(tid)
    if not entry or entry.get("done"):
        return False
    task = entry.get("task")
    if task and hasattr(task, "cancel") and not task.done():
        task.cancel()
        entry["done"] = True
        entry["done_at"] = time.time()
        return True
    return False


def list_tasks() -> list[dict]:
    """List all tracked background tasks (metadata only, no task objects).

    Automatically cleans up completed tasks older than 60 seconds.
    """
    now = time.time()
    with _registry_lock:
        result = []
        dead_tids = []
        for tid, entry in list(_registry.items()):
            if entry.get("done"):
                if now - entry.get("done_at", now) > 60:
                    dead_tids.append(tid)
                    continue
            result.append({
                "task_id": tid,
                "title": entry.get("title", ""),
                "type": entry.get("type", ""),
                "description": entry.get("description", ""),
                "age_seconds": now - entry.get("created_at", now),
                "done": entry.get("done", False),
            })
        for tid in dead_tids:
            _registry.pop(tid, None)
        return result


def cancel_tasks_matching(prefix: str = "", exact_title: str = "", exact_desc: str = "") -> int:
    """Cancel all tasks matching criteria. Returns count cancelled."""
    cancelled = 0
    for entry in list_tasks():
        tid = entry["task_id"]
        match = False
        if prefix and prefix in entry.get("title", ""):
            match = True
        if exact_title and entry.get("title", "") == exact_title:
            match = True
        if exact_desc and entry.get("description", "") == exact_desc:
            match = True
        if match and cancel_task(tid):
            cancelled += 1
    return cancelled


def clear_all() -> int:
    """Remove all completed tasks from registry. Returns count removed."""
    now = time.time()
    with _registry_lock:
        removed = 0
        for tid, entry in list(_registry.items()):
            if entry.get("done"):
                if now - entry.get("done_at", now) > 60:
                    _registry.pop(tid, None)
                    removed += 1
    return removed
