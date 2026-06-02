"""
Celery tasks for all periodic Nexus scheduler work.

Each function is a plain synchronous Celery task that delegates to the
underlying async Nexus APIs via ``_async_helper.run_async``.  Events are
published through the Redis event bridge so the main-process event
consumer can relay them to WebSocket clients.

Intervals are configured in ``heartbeat.start()`` via the Celery Beat
schedule; the tasks themselves are stateless.

v2: Added batch task processing, parallel dispatch optimization, and
better error recovery with event-driven notifications.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from ..celery_app import celery_app
from .event_bridge import publish_event
from .task_registry import register_task as _register_task
from . import _async_helper as _ah

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe(task_name: str, fn, *args, **kwargs):
    """Run *fn* inside run_async; swallow & log exceptions."""
    try:
        _ah.run_async(fn(*args, **kwargs))
    except Exception as exc:
        logger.exception("Celery task %s failed: %s", task_name, exc)
        publish_event("heartbeat_step_error", step=task_name, error=str(exc))


# ═════════════════════════════════════════════════════════════════════════
# 1. Data integrity check
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.check_data_integrity",
    max_retries=1,
)
def check_data_integrity() -> None:
    """Periodically verify that the SQLite database is healthy."""
    _safe("data_integrity", _do_check_data_integrity)


async def _do_check_data_integrity() -> None:
    from .event_bridge import get_state, set_state
    from ..memory.database import _connect, _SCHEMA_STATEMENTS
    import re

    now = time.time()

    last_check = float(get_state("last_integrity_check", "0"))
    if now - last_check < 300:
        return
    set_state("last_integrity_check", str(now))

    issues: list[str] = []

    # Check 1: Connection + query
    try:
        async with _connect() as conn:
            c = await conn.execute("SELECT 1")
            val = (await c.fetchone())[0]
            if val != 1:
                issues.append("SELECT 1 returned unexpected value")
    except Exception as e:
        issues.append(f"Database connection error: {e}")

    if not issues:
        # Check 2: Expected tables exist (SQLite)
        expected_tables: set[str] = set()
        for stmt in _SCHEMA_STATEMENTS:
            m = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", stmt, re.IGNORECASE)
            if m:
                expected_tables.add(m.group(1).lower())

        try:
            async with _connect() as conn:
                c = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                rows = await c.fetchall()
                existing = {r[0].lower() for r in rows if r[0]}
                missing = expected_tables - existing
                if missing:
                    issues.append(f"Missing tables: {', '.join(sorted(missing))}")

                # Check 3: Can read critical tables
                for table in ("sessions", "memories", "goals", "tasks"):
                    if table in existing:
                        try:
                            c2 = await conn.execute(f"SELECT count(*) FROM {table}")
                            await c2.fetchone()
                        except Exception as e:
                            issues.append(f"Cannot read table '{table}': {e}")
        except Exception as e:
            issues.append(f"Table-existence check failed: {e}")

    is_ok = len(issues) == 0
    prev_ok = get_state("last_integrity_ok", "")

    if prev_ok == "" or is_ok != (prev_ok == "1"):
        if is_ok:
            publish_event("integrity_check", status="ok",
                          message="All data is safe — database integrity verified")
        else:
            publish_event("integrity_check", status="corrupted",
                          message="Data integrity issues detected", issues=issues)
            # Auto-repair
            try:
                async with _connect() as conn:
                    for stmt in _SCHEMA_STATEMENTS:
                        try:
                            await conn.execute(stmt)
                        except Exception:
                            pass
                publish_event("integrity_repair", status="repaired",
                              message="Missing tables have been recreated")
            except Exception as repair_err:
                publish_event("integrity_repair", status="failed",
                              message=f"Auto-repair failed: {repair_err}")

    set_state("last_integrity_ok", "1" if is_ok else "0")


# ═════════════════════════════════════════════════════════════════════════
# 2. Process scheduled tasks (batch-optimized)
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.process_scheduled_tasks",
)
def process_scheduled_tasks() -> None:
    """Find tasks whose ``scheduled_for`` has passed and dispatch the agent."""
    _safe("scheduled_tasks", _do_process_scheduled_tasks)


async def _do_process_scheduled_tasks() -> None:
    from ..memory import db
    from .agent_tasks import run_autorun_task_inline

    now = time.time()
    pending = [t for t in await db.list_tasks(status="pending")
               if (t.get("scheduled_for") or 0) <= now]

    if not pending:
        return

    # Batch update all matching tasks to 'active' status
    for t in pending:
        await db.update_task(t["id"], status="active", updated_at=now)

    # Dispatch each pending task inline (works w/ or w/o Celery worker)
    for t in pending:
        publish_event(
            "scheduled_task_ready", task_id=t["id"], title=t.get("title", ""),
            message=f"Scheduled task is now due: {t.get('title', '')}",
        )
        _bt = asyncio.create_task(run_autorun_task_inline(
            t["id"], t.get("title", "Untitled task"),
        ))
        _register_task(_bt, f"task_{t['id']}", "scheduled",
                        f"schedule:{t['id']}")


# ═════════════════════════════════════════════════════════════════════════
# 3. Process NL schedules
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.process_nl_schedules",
)
def process_nl_schedules() -> None:
    """Fire NL schedules whose ``next_run`` has passed and compute next run."""
    _safe("nl_schedules", _do_process_nl_schedules)


async def _do_process_nl_schedules() -> None:
    from ..memory import db
    from ..heartbeat import _next_run_from_expr
    from .agent_tasks import autorun_task

    now = time.time()
    schedules = await db.list_nl_schedules(enabled_only=True)

    for s in schedules:
        if (s.get("next_run") or 0) > now:
            continue

        task_title = s.get("task_title", "Scheduled task")
        schedule_id = s["id"]
        expr = (s.get("expression") or "").lower()

        tid = await db.add_task(goal_id=s.get("goal_id"), title=task_title,
                                scheduled_for=now)
        await db.update_task(tid, status="active", updated_at=now)

        publish_event(
            "nl_schedule_fired", schedule_id=schedule_id, task_id=tid,
            expression=expr,
            message=f"Schedule fired: '{task_title}' ('{expr}')",
        )

        try:
            await db.add_notification(
                title="Schedule fired",
                body=f"{task_title} — '{expr}'",
                kind="schedule",
            )
        except Exception:
            pass

        from .agent_tasks import run_autorun_task_inline as _run_autorun
        _bt = asyncio.create_task(_run_autorun(tid, task_title))
        _register_task(_bt, f"task_{tid}", "nl_schedule",
                        f"schedule:{tid}")

        next_run = _next_run_from_expr(expr, now)
        run_count = (s.get("run_count") or 0) + 1
        await db.update_nl_schedule(
            schedule_id, next_run=next_run, last_run=now, run_count=run_count,
        )


# ═════════════════════════════════════════════════════════════════════════
# 4. Process incomplete goals (batch-optimized)
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.process_incomplete_goals",
)
def process_incomplete_goals() -> None:
    """Scan for incomplete goals/tasks and dispatch the agent."""
    _safe("incomplete_goals", _do_process_incomplete_goals)


async def _do_process_incomplete_goals() -> None:
    from ..memory import db
    from .. import config
    from .event_bridge import (
        get_last_goal_dispatch, set_last_goal_dispatch,
        cleanup_goal_dispatches, get_goal_dispatch_cleanup_ts,
        set_goal_dispatch_cleanup_ts,
    )
    from .agent_tasks import run_agent_on_goal, autorun_task

    now = time.time()
    goal_retry_interval = int(config.get("goal_retry_interval_seconds", 600))
    stale_task_threshold = int(config.get("stale_task_threshold_seconds", 3600))
    goal_dispatch_cleanup_interval = 86400

    # Periodic cleanup
    last_cleanup = get_goal_dispatch_cleanup_ts()
    if now - last_cleanup > goal_dispatch_cleanup_interval:
        removed = cleanup_goal_dispatches(now - goal_dispatch_cleanup_interval)
        set_goal_dispatch_cleanup_ts(now)
        if removed:
            logger.debug("Cleaned up %d stale entries from goal_dispatches", removed)

    # 1. Pending goals → dispatch agent
    pending_goals = await db.list_goals(status="pending")
    pending_goals.sort(key=lambda g: g.get("priority", 5), reverse=True)

    for goal in pending_goals:
        gid = goal["id"]
        if now - get_last_goal_dispatch(gid) < goal_retry_interval:
            continue

        set_last_goal_dispatch(gid, now)
        title = goal.get("title", "Untitled goal")
        await db.update_goal(gid, status="active")
        publish_event(
            "goal_started", goal_id=gid, title=title,
            message=f"Agent starting work on goal: {title}",
        )
        run_agent_on_goal.delay(gid, title, goal.get("description", ""),
                                goal.get("priority", 5), "")

    # 2. Active goals with pending sub-tasks → nudge agent
    active_goals = await db.list_goals(status="active")
    all_pending_tasks = await db.list_tasks(status="pending")

    for goal in active_goals:
        gid = goal["id"]
        if now - get_last_goal_dispatch(gid) < goal_retry_interval:
            continue

        goal_tasks = [t for t in all_pending_tasks if t.get("goal_id") == gid]
        if goal_tasks:
            set_last_goal_dispatch(gid, now)
            publish_event(
                "goal_needs_work", goal_id=gid, title=goal.get("title", ""),
                progress=goal.get("progress", 0), pending_tasks=len(goal_tasks),
                message=f"Goal '{goal.get('title')}' has {len(goal_tasks)} pending task(s)",
            )
            run_agent_on_goal.delay(
                gid, goal.get("title", ""), goal.get("description", ""),
                goal.get("priority", 5),
                f"Continue working — {len(goal_tasks)} sub-task(s) still pending",
            )

    # 3. Orphan pending tasks → dispatch agent
    orphans = [t for t in all_pending_tasks if not t.get("goal_id")]
    orphan_batch = []
    for task in orphans[:3]:
        tid, ttitle = task["id"], task.get("title", "Pending task")
        orphan_batch.append((tid, ttitle))
    if orphan_batch:
        from .agent_tasks import autorun_batch
        autorun_batch.delay(orphan_batch)

    # 4. Stale active tasks
    active_tasks = await db.list_tasks(status="active")
    stale_threshold = now - stale_task_threshold
    stale_tasks = [
        t for t in active_tasks
        if (t.get("updated_at") or t.get("created_at") or 0) < stale_threshold
    ]
    if stale_tasks:
        publish_event(
            "stale_tasks_detected",
            count=len(stale_tasks),
            tasks=[{"id": t["id"], "title": t.get("title")} for t in stale_tasks],
            message=f"{len(stale_tasks)} task(s) stale for >{stale_task_threshold // 3600}h",
        )


# ═════════════════════════════════════════════════════════════════════════
# 5. File watcher tick
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.file_watcher_tick",
)
def file_watcher_tick() -> None:
    """Run one pass of the file watcher."""
    _safe("file_watcher", _do_file_watcher_tick)


async def _do_file_watcher_tick() -> None:
    from ..watchers import FileWatcher
    watcher = FileWatcher()
    await watcher.tick()


# ═════════════════════════════════════════════════════════════════════════
# 6. Web monitor tick
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.web_monitor_tick",
)
def web_monitor_tick() -> None:
    """Run one pass of the web monitor."""
    _safe("web_monitor", _do_web_monitor_tick)


async def _do_web_monitor_tick() -> None:
    from ..watchers import WebMonitor
    monitor = WebMonitor()
    await monitor.tick()


# ═════════════════════════════════════════════════════════════════════════
# 7. Self-improvement run
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.self_improvement_run",
    max_retries=0,
)
def self_improvement_run() -> None:
    """Execute the full self-improvement analysis cycle."""
    _safe("self_improvement", _do_self_improvement_run)


async def _do_self_improvement_run() -> None:
    logger.info("[self_improve] Starting pipeline")
    from .. import self_improve
    try:
        await self_improve.run()
        logger.info("[self_improve] Pipeline completed successfully")
    except Exception as exc:
        logger.exception("[self_improve] Pipeline failed: %s", exc)
        raise


# ═════════════════════════════════════════════════════════════════════════
# 8. Environment awareness check
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.environment_check",
)
def environment_check() -> None:
    """Check environment for anomalies and emit alerts."""
    _safe("environment", _do_environment_check)


async def _do_environment_check() -> None:
    from ..environment import ENVIRONMENT
    from ..memory import db

    env_state = await ENVIRONMENT.get_state()
    anomalies = env_state.get("anomalies", [])
    for anomaly in anomalies:
        publish_event("anomaly", **anomaly)
        if anomaly.get("severity") == "critical":
            try:
                await db.add_notification(
                    title="System Alert",
                    body=anomaly.get("message", ""),
                    kind="warning",
                )
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════
# 9. Health summary
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.health_summary",
)
def health_summary() -> None:
    """Compile and emit system health statistics."""
    _safe("health_summary", _do_health_summary)


async def _do_health_summary() -> None:
    from ..memory import db
    from .event_bridge import get_state

    now = time.time()

    # Gather all stats in parallel (direct await, not nested run_async)
    pending_goals, active_goals, pending_tasks, file_watches, web_monitors, nl_schedules, triggers = await _gather_health_stats()

    integrity_ok_str = get_state("last_integrity_ok", "")
    integrity_ok = integrity_ok_str == "1"
    last_integrity_check = float(get_state("last_integrity_check", "0"))

    health = {
        "uptime_tick": now,
        "integrity_ok": integrity_ok,
        "integrity_last_checked": last_integrity_check,
        "pending_goals": len(pending_goals) + len(active_goals),
        "pending_tasks": len(pending_tasks),
        "file_watches": len(file_watches),
        "web_monitors": len(web_monitors),
        "nl_schedules": len(nl_schedules),
        "active_triggers": len(triggers),
    }
    publish_event("heartbeat_health", **health)


async def _gather_health_stats() -> tuple:
    """Gather all health statistics in parallel for faster health summary."""
    from ..memory import db

    async def _pg():
        return await db.list_goals(status="pending")
    async def _ag():
        return await db.list_goals(status="active")
    async def _pt():
        return await db.list_tasks(status="pending")
    async def _fw():
        return await db.list_file_watches(enabled_only=True)
    async def _wm():
        return await db.list_web_monitors(enabled_only=True)
    async def _ns():
        return await db.list_nl_schedules(enabled_only=True)
    async def _tr():
        return await db.list_triggers(enabled_only=True)

    import asyncio
    return await asyncio.gather(_pg(), _ag(), _pt(), _fw(), _wm(), _ns(), _tr())


# ═════════════════════════════════════════════════════════════════════════
# 10. Process active triggers (batch-optimized)
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.process_active_triggers",
)
def process_active_triggers() -> None:
    """Poll enabled triggers, evaluate conditions, and dispatch agent runs."""
    _safe("active_triggers", _do_process_active_triggers)


async def _do_process_active_triggers() -> None:
    from ..memory import db
    from .. import config
    from .agent_tasks import autorun_task
    from .event_bridge import get_state, set_state

    now = time.time()
    trigger_cooldown = int(config.get("trigger_cooldown_seconds", 60))

    triggers = await db.list_triggers(enabled_only=True)
    if not triggers:
        return

    fired: list[tuple[int, str, str]] = []  # (task_id, trigger_name, reason)
    _webhook_event_id: int | None = None

    for trigger in triggers:
        trigger_id = trigger["id"]
        trigger_type = trigger.get("type", "custom")
        trigger_name = trigger.get("name", trigger_id)
        trigger_config: dict = {}
        if trigger.get("config"):
            try:
                trigger_config = json.loads(trigger["config"]) if isinstance(trigger["config"], str) else trigger["config"]
            except Exception:
                logger.debug("Failed to parse trigger config for %s", trigger_id, exc_info=True)
                trigger_config = {}

        last_fired = trigger.get("last_fired") or 0
        if last_fired and (now - last_fired) < trigger_cooldown:
            continue

        should_fire = False
        fire_reason = ""
        _webhook_event_id = None

        if trigger_type == "schedule":
            interval = trigger_config.get("interval_seconds", 0)
            if interval > 0 and (now - last_fired) >= interval:
                should_fire = True
                fire_reason = f"Scheduled interval reached ({interval}s)"
            else:
                expression = trigger_config.get("expression", "")
                if expression:
                    from ..heartbeat import _next_run_from_expr
                    next_run = trigger_config.get("_next_run", 0)
                    if next_run and now >= next_run:
                        should_fire = True
                        fire_reason = f"Schedule expression '{expression}' triggered"
                    elif not next_run:
                        next_run = _next_run_from_expr(expression.lower(), now)
                        trigger_config["_next_run"] = next_run
                        await db.upsert_trigger(
                            trigger_id=trigger_id, name=trigger_name,
                            trigger_type=trigger_type,
                            integration_id=trigger.get("integration_id", ""),
                            config=trigger_config, enabled=True,
                        )

        elif trigger_type == "webhook":
            service_id = trigger.get("integration_id", "") or trigger_config.get("service_id", "")
            if service_id:
                try:
                    events = await db.list_webhook_events(service_id=service_id, unprocessed_only=True, limit=1)
                    if events:
                        _webhook_event_id = events[0]["id"]
                        should_fire = True
                        fire_reason = f"Unprocessed webhook from {service_id}"
                except Exception:
                    logger.debug("Failed to list webhook events for %s", service_id, exc_info=True)

        elif trigger_type == "event":
            event_type = trigger_config.get("event_type", "")
            if event_type:
                last_event_ts = float(get_state(f"trigger_event_{trigger_id}_{event_type}", "0"))
                if last_event_ts > (last_fired or 0):
                    should_fire = True
                    fire_reason = f"Event '{event_type}' detected"

        elif trigger_type == "integration":
            integration_id = trigger.get("integration_id", "")
            if integration_id:
                try:
                    recent = await db.recent_tool_executions(limit=10)
                    for exe in recent:
                        exe_tool = exe.get("tool_name", "")
                        exe_created = (exe.get("created_at", 0) or 0)
                        if exe_tool == "call_integration_api":
                            exe_params = exe.get("params", {})
                            if isinstance(exe_params, str):
                                exe_params = json.loads(exe_params)
                            if exe_params.get("service_id", "") == integration_id or exe_params.get("service", "") == integration_id:
                                if exe_created > (last_fired or 0):
                                    should_fire = True
                                    fire_reason = f"Integration '{integration_id}' activity detected"
                                    break
                except Exception:
                    logger.debug("Failed to check integration activity for %s", integration_id, exc_info=True)

        elif trigger_type == "file_change":
            watch_path = trigger_config.get("path", "")
            if watch_path:
                try:
                    watches = await db.list_file_watches(enabled_only=True)
                    for w in watches:
                        if w.get("path") == watch_path and (w.get("last_changed", 0) or 0) > (last_fired or 0):
                            should_fire = True
                            fire_reason = f"File change detected at {watch_path}"
                            break
                except Exception:
                    logger.debug("Failed to check file watches for %s", watch_path, exc_info=True)

        elif trigger_type == "manual":
            continue

        elif trigger_type == "custom":
            condition = trigger_config.get("condition", "")
            if condition and condition.lower() == "always":
                should_fire = True
                fire_reason = "Custom 'always' condition met"

        if should_fire:
            await db.fire_trigger(trigger_id)
            task_title = f"[Trigger] {trigger_name}"
            prompt_context = trigger_config.get("prompt", "")
            if not prompt_context:
                prompt_context = f"Trigger '{trigger_name}' of type '{trigger_type}' has fired. {fire_reason}."

            task_id = await db.add_task(goal_id=None, title=task_title, scheduled_for=now)
            await db.update_task(task_id, status="active", updated_at=now)
            fired.append((task_id, task_title, fire_reason))

            publish_event(
                "trigger_fired",
                trigger_id=trigger_id, trigger_name=trigger_name,
                trigger_type=trigger_type, task_id=task_id,
                reason=fire_reason,
                message=f"Trigger fired: {trigger_name} — {fire_reason}",
            )

            try:
                await db.add_notification(
                    title="Trigger Fired",
                    body=f"{trigger_name}: {fire_reason}",
                    kind="trigger",
                )
            except Exception:
                logger.debug("Failed to add trigger notification for %s", trigger_name, exc_info=True)

            # Update schedule expression next_run
            if trigger_type == "schedule" and trigger_config.get("expression"):
                from ..heartbeat import _next_run_from_expr
                next_run = _next_run_from_expr(trigger_config["expression"].lower(), now)
                trigger_config["_next_run"] = next_run
                await db.upsert_trigger(
                    trigger_id=trigger_id, name=trigger_name,
                    trigger_type=trigger_type,
                    integration_id=trigger.get("integration_id", ""),
                    config=trigger_config, enabled=True,
                )

            # Mark webhook event as processed
            if trigger_type == "webhook" and _webhook_event_id is not None:
                try:
                    await db.mark_webhook_event_processed(_webhook_event_id)
                except Exception:
                    logger.debug("Failed to mark webhook event %s processed", _webhook_event_id, exc_info=True)

    # Fire all triggered tasks inline (works w/ or w/o Celery worker)
    if fired:
        from .agent_tasks import run_autorun_task_inline
        for tid, title, _ in fired:
            _bt = asyncio.create_task(run_autorun_task_inline(tid, title))
            _register_task(_bt, f"task_{tid}", "trigger",
                            f"Trigger → task {tid}: {title[:60]}")



async def _do_tree_maintenance_run() -> dict:
    """Run memory tree maintenance (sealing, summarization, cleanup)."""
    from ..memory_maintenance import run_tree_maintenance
    result = await run_tree_maintenance()
    sealed = result.get("sealed", 0)
    summarized = result.get("summarized", 0)
    cleaned = result.get("orphans_removed", 0)
    if sealed or summarized or cleaned:
        logger.info("Tree maintenance: sealed=%d, summarized=%d, cleaned=%d",
                    sealed, summarized, cleaned)
    return result

# ═════════════════════════════════════════════════════════════════════════
# 11. Memory curve pruning — Ebbinghaus forgetting curve
# ═════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="nexus.tasks.scheduler_tasks.memory_curve_pruning",
)
def memory_curve_pruning() -> None:
    """Periodic pruning of flat memories and tree nodes using the
    Ebbinghaus forgetting curve.  Low-retention memories are archived;
    low-retention tree nodes are hard-deleted.
    """
    _safe("memory_curve_pruning", _do_memory_curve_pruning)


async def _do_memory_curve_pruning() -> dict:
    """Run the forgetting-curve pruning on both flat memories and tree nodes."""
    from .. import config
    from ..memory import database as _db
    from ..memory.tree_db import tree_prune_by_forgetting_curve

    base_half_life = float(config.get("pruning_half_life_hours", 720.0))
    retention_thr = float(config.get("pruning_retention_threshold", 0.08))
    max_prune_flat = int(config.get("pruning_max_per_run", 500))
    max_prune_tree = int(config.get("pruning_tree_max_per_run", 200))

    logger.info(
        "[pruning] Starting forgetting-curve pruning "
        "(half_life=%.1fh, threshold=%.3f, max_flat=%d, max_tree=%d)",
        base_half_life, retention_thr, max_prune_flat, max_prune_tree,
    )

    result_flat = await _db.prune_memories_by_forgetting_curve(
        base_half_life_hours=base_half_life,
        retention_threshold=retention_thr,
        max_prune=max_prune_flat,
    )

    result_tree = await tree_prune_by_forgetting_curve(
        base_half_life_hours=base_half_life,
        retention_threshold=retention_thr,
        max_prune=max_prune_tree,
    )

    pruned_total = result_flat.get("pruned", 0) + result_tree.get("pruned_nodes", 0)
    remaining = result_flat.get("remaining", 0)

    logger.info(
        "[pruning] Done: pruned %d flat + %d tree nodes, "
        "%d flat remaining, %d tree orphans cleaned",
        result_flat.get("pruned", 0), result_tree.get("pruned_nodes", 0),
        remaining, result_tree.get("orphans_removed", 0),
    )

    # Emit event for the WebUI
    try:
        from ..events import emit
        await emit(
            "memory_curves_pruned",
            pruned_flat=result_flat.get("pruned", 0),
            pruned_tree=result_tree.get("pruned_nodes", 0),
            flat_remaining=remaining,
            method="forgetting_curve",
        )
    except Exception:
        pass

    # ── Hard cap: enforce max memory limit regardless of retention ──
    try:
        hard_pruned = await _db.prune_stale_memories(
            max_age_days=90,
            min_importance=0.1,
            min_access_count=0,
            max_memories=1000,
        )
        if hard_pruned:
            logger.info(
                "[pruning] Hard cap pruned %d additional memories",
                hard_pruned,
            )
            remaining = remaining - hard_pruned
    except Exception as exc:
        logger.warning("[pruning] Hard cap pruning failed: %s", exc)

    return {
        "flat": result_flat,
        "tree": result_tree,
    }


