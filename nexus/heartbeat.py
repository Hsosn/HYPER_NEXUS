"""
Heartbeat — thin scheduler that wires Celery Beat to the Nexus periodic
task system and starts the Redis→asyncio event bridge consumer.

Responsibilities
----------------
* On ``start()``: configure Celery Beat with the desired schedule,
  launch the embedded beat service in a daemon thread, and start the
  asyncio event-bridge consumer.
* On ``stop()``: cancel the beat service and event consumer.
* Expose ``_next_run_from_expr()`` for NL schedule processing in the
  Celery tasks.

The heavy lifting (data-integrity checks, scheduled-task processing,
goal scanning, file/web watching, self-improvement, environment checks,
health summaries) has been moved to Celery tasks in ``nexus.tasks.*``.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta
from typing import Any

from . import config
from .celery_app import celery_app
from .tasks.event_bridge import start_event_consumer

logger = logging.getLogger(__name__)

# ── State ──────────────────────────────────────────────────────────────────
_running = False
_beat_service: Any = None
_beat_thread: threading.Thread | None = None
_event_consumer_task: asyncio.Task | None = None
_heartbeat_task: asyncio.Task | None = None

# Mapping of Celery task name → interval in seconds, used to build the
# Beat schedule.  Intentionally kept here (not in Celery conf) so the
# heartbeat module remains the single source of truth for scheduling.
_SCHEDULE: dict[str, int] = {
    "nexus.tasks.scheduler_tasks.check_data_integrity":     300,   # 5 min
    "nexus.tasks.scheduler_tasks.process_scheduled_tasks":   30,   # 30 s
    "nexus.tasks.scheduler_tasks.process_nl_schedules":      30,   # 30 s
    "nexus.tasks.scheduler_tasks.process_incomplete_goals": 120,   # 2 min
    "nexus.tasks.scheduler_tasks.file_watcher_tick":         30,   # 30 s
    "nexus.tasks.scheduler_tasks.web_monitor_tick":          60,   # 60 s
    "nexus.tasks.scheduler_tasks.self_improvement_run":     1800,  # 30 min
    "nexus.tasks.scheduler_tasks.environment_check":        300,   # 5 min
    "nexus.tasks.scheduler_tasks.health_summary":            30,   # 30 s
    "nexus.tasks.scheduler_tasks.process_active_triggers":   60,   # 60 s
    "nexus.tasks.scheduler_tasks.memory_curve_pruning":              1800, # 30 min
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API — start / stop
# ═══════════════════════════════════════════════════════════════════════════

async def start() -> None:
    """Activate periodic heartbeat and the Redis event consumer.

    Emits a heartbeat event directly from the main asyncio loop every
    30 seconds so the WebUI stays connected.  This bypasses the need
    for a separate Celery worker on single-machine deployments.
    """
    global _running, _beat_service, _beat_thread, _event_consumer_task, _heartbeat_task

    if _running:
        return

    _running = True

    from .tasks.task_registry import register_task as _register_task

    # ── 1. Start the Redis→asyncio event consumer ─────────────────────
    try:
        _event_consumer_task = asyncio.create_task(
            start_event_consumer(), name="event-bridge-consumer",
        )
        _register_task(_event_consumer_task, "Event bridge consumer", "system",
                        "Redis→asyncio event bridge consumer")
        logger.info("Event bridge consumer started")
    except Exception as exc:
        logger.warning("Event bridge consumer failed to start (non-fatal): %s", exc)

    # ── 2. Direct asyncio heartbeat loop (no Celery worker needed) ─────
    from .events import emit
    import time as _time

    async def _heartbeat_loop():
        """Emit heartbeat events and run periodic tasks every 30 seconds.
        
        v32: Runs critical periodic tasks directly in the asyncio loop
        so that scheduled tasks, health summaries, and goal processing
        work without requiring a separate Celery worker process.
        """
        _tick_count = 0
        while _running:
            try:
                await emit("heartbeat", ts=_time.time())
                if _tick_count % 2 == 0:
                    logger.info("[heartbeat] tick %d", _tick_count)
            except Exception as exc:
                logger.warning("Heartbeat emit failed: %s", exc)
            
            _tick_count += 1
            
            # Run periodic tasks inline every tick
            # (These tasks are idempotent — each checks DB state before acting)
            try:
                # Every tick (30s): health summary
                if _tick_count % 1 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_health_summary
                        await _do_health_summary()
                    except Exception as exc:
                        logger.warning("Inline health_summary failed: %s", exc)
                
                # Every 1 tick (30s): process scheduled tasks
                if _tick_count % 1 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_process_scheduled_tasks
                        await _do_process_scheduled_tasks()
                    except Exception as exc:
                        logger.warning("Inline scheduled_tasks failed: %s", exc)
                
                # Every 1 tick (30s): process NL schedules
                if _tick_count % 1 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_process_nl_schedules
                        await _do_process_nl_schedules()
                    except Exception as exc:
                        logger.warning("Inline nl_schedules failed: %s", exc)
                
                # Every 4 ticks (2 min): process incomplete goals
                if _tick_count % 4 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_process_incomplete_goals
                        await _do_process_incomplete_goals()
                    except Exception as exc:
                        logger.warning("Inline incomplete_goals failed: %s", exc)
                
                # Every 2 ticks (1 min): process active triggers
                if _tick_count % 2 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_process_active_triggers
                        await _do_process_active_triggers()
                    except Exception as exc:
                        logger.warning("Inline active_triggers failed: %s", exc)
                
                # Every 2 ticks (1 min): web monitor tick
                if _tick_count % 2 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_web_monitor_tick
                        await _do_web_monitor_tick()
                    except Exception as exc:
                        logger.warning("Inline web_monitor failed: %s", exc)
                
                # Every 1 tick (30s): file watcher tick
                if _tick_count % 1 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_file_watcher_tick
                        await _do_file_watcher_tick()
                    except Exception as exc:
                        logger.warning("Inline file_watcher failed: %s", exc)
                
                # Every 10 ticks (5 min): environment check
                if _tick_count % 10 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_environment_check
                        await _do_environment_check()
                    except Exception as exc:
                        logger.warning("Inline environment_check failed: %s", exc)
                
                # Every 60 ticks (30 min): self-improvement (configurable)
                si_interval_ticks = max(1, config.get("self_improvement_interval_seconds", 1800) // 30)
                if _tick_count % si_interval_ticks == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_self_improvement_run
                        logger.info("[heartbeat] Starting self-improvement cycle")
                        await _do_self_improvement_run()
                        logger.info("[heartbeat] Self-improvement cycle completed")
                    except Exception as exc:
                        logger.warning("[heartbeat] Self-improvement failed: %s", exc)
                
                # Every 10 ticks (5 min): data integrity check
                if _tick_count % 10 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_check_data_integrity
                        await _do_check_data_integrity()
                    except Exception as exc:
                        logger.warning("Inline data_integrity failed: %s", exc)

                # Every 60 ticks (30 min): forgetting-curve memory pruning
                if _tick_count % 60 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_memory_curve_pruning
                        await _do_memory_curve_pruning()
                    except Exception as exc:
                        logger.warning("Inline memory_curve_pruning failed: %s", exc)

                # Every 60 ticks (30 min): tree maintenance
                if _tick_count % 60 == 0:
                    try:
                        from .tasks.scheduler_tasks import _do_tree_maintenance_run
                        await _do_tree_maintenance_run()
                    except Exception as exc:
                        logger.warning("Inline tree_maintenance failed: %s", exc)

                # Every 20 ticks (10 min): notification cleanup
                if _tick_count % 20 == 0:
                    try:
                        from .memory.database import cleanup_old_notifications
                        await cleanup_old_notifications(max_age_days=30)
                    except Exception as exc:
                        logger.warning("Inline notification_cleanup failed: %s", exc)
                        
            except Exception as exc:
                logger.warning("Inline periodic task error: %s", exc)
            
            await asyncio.sleep(30)

        # Clean up all tracked background tasks when the loop stops
        from .tasks.task_registry import clear_all as _clear_tasks
        _clear_tasks()

    _heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="heartbeat-loop")
    _register_task(_heartbeat_task, "Heartbeat loop", "system",
                    "Main heartbeat loop — runs periodic tasks every 30s")
    logger.info("Direct heartbeat loop started (30s interval)")

    # Emit the first heartbeat immediately so the WebUI knows Nexus is alive.
    try:
        await emit("heartbeat", ts=_time.time())
    except Exception:
        pass

    # ── 3. Optionally start Celery Beat for advanced scheduling ───────
    if config.get("enable_celery", False):
        try:
            beat_schedule: dict[str, dict[str, Any]] = {}
            for task_name, interval_secs in _SCHEDULE.items():
                beat_schedule[task_name.rsplit(".", 1)[-1]] = {
                    "task": task_name,
                    "schedule": interval_secs,
                }
            celery_app.conf.beat_schedule = beat_schedule

            try:
                from celery.beat import EmbeddedService as BeatService
                _beat_service = BeatService(app=celery_app)
            except ImportError:
                from celery.beat import Service as BeatService
                _beat_service = BeatService(app=celery_app, max_interval=30)

            beat_target = None
            if hasattr(_beat_service, 'run'):
                beat_target = _beat_service.run
            elif hasattr(_beat_service, 'start_sync'):
                beat_target = _beat_service.start_sync
            elif hasattr(_beat_service, 'start'):
                import asyncio as _aio
                loop = _aio.new_event_loop()
                def _run_start():
                    _aio.set_event_loop(loop)
                    loop.run_until_complete(_beat_service.start())
                beat_target = _run_start

            if beat_target:
                _beat_thread = threading.Thread(
                    target=beat_target, daemon=True, name="celery-beat",
                )
                _beat_thread.start()
                logger.info("Celery Beat scheduler started (embedded)")
        except Exception as exc:
            logger.warning("Celery Beat failed to start (non-fatal): %s", exc)
    else:
        logger.info("Celery Beat disabled (enable_celery=false) — using direct heartbeat")


async def stop() -> None:
    """Shut down the heartbeat loop, Beat scheduler, and event consumer."""
    global _running, _beat_service, _beat_thread, _event_consumer_task, _heartbeat_task

    _running = False

    # Cancel the direct heartbeat loop
    if _heartbeat_task is not None and not _heartbeat_task.done():
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
        _heartbeat_task = None
        logger.info("Heartbeat loop stopped")

    # Stop Beat
    if _beat_service is not None:
        try:
            _beat_service.stop()
            logger.info("Celery Beat scheduler stopped")
        except AttributeError as exc:
            # Celery BeatService may internally call .terminate() on a None
            # attribute during shutdown (e.g. self._process). Catch gracefully.
            logger.warning("Error stopping Celery Beat (internal state): %s", exc)
        except Exception as exc:
            logger.warning("Error stopping Celery Beat: %s", exc)
        _beat_service = None
        _beat_thread = None

    # Cancel the event consumer
    if _event_consumer_task is not None and not _event_consumer_task.done():
        _event_consumer_task.cancel()
        try:
            await _event_consumer_task
        except asyncio.CancelledError:
            pass
        _event_consumer_task = None
        logger.info("Event bridge consumer stopped")


# ═══════════════════════════════════════════════════════════════════════════
# NL schedule expression parser (used by the Celery task)
# ═══════════════════════════════════════════════════════════════════════════

def _next_run_from_expr(expr: str, base: float) -> float:
    """Compute next scheduled time from expression and a base timestamp.

    Supports:
      - "every N minutes/hours/seconds"
      - "every hour", "every day", "daily", "every week", "weekly"
      - "every weekday at HH:MM" → next Mon–Fri at that time
      - "every <day> at HH:MM" → next occurrence of that day at that time
      - "every N days/weeks/months"
    Uses local time for day-of-week scheduling.
    """
    dt = datetime.fromtimestamp(base)

    # Pattern: "every N minutes/hours/seconds"
    m = re.match(r"every (\d+)\s*(minute|hour|second)s?", expr)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        deltas = {"second": 1, "minute": 60, "hour": 3600}
        return base + n * deltas[unit]

    # Pattern: "every N days/weeks/months"
    m = re.match(r"every (\d+)\s*(day|week|month)s?", expr)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        deltas = {"day": 86400, "week": 7 * 86400, "month": 30 * 86400}
        return base + n * deltas[unit]

    # Pattern: "every weekday at HH:MM" — Monday through Friday only
    m = re.match(r"every weekday at (\d{1,2}):(\d{2})", expr)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        candidate = dt.replace(hour=hour, minute=minute, second=0,
                               microsecond=0)
        if candidate <= dt:
            candidate += timedelta(days=1)
        # Skip to Monday if we land on Sat (5) or Sun (6)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate.timestamp()

    # Pattern: "every <day_of_week> at HH:MM"
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    days_lower = {d: v for d, v in day_map.items()}
    days_lower.update({d + "s": v for d, v in day_map.items()})
    m = re.match(r"every (\w+) at (\d{1,2}):(\d{2})", expr)
    if m:
        day_name = m.group(1).lower()
        hour, minute = int(m.group(2)), int(m.group(3))
        if day_name in days_lower:
            target_wday = days_lower[day_name]
            candidate = dt.replace(hour=hour, minute=minute, second=0,
                                   microsecond=0)
            days_ahead = (target_wday - candidate.weekday()) % 7
            if days_ahead == 0 and candidate <= dt:
                days_ahead = 7
            candidate += timedelta(days=days_ahead)
            return candidate.timestamp()

    # Simple keyword fallbacks
    if "every hour" in expr:
        return base + 3600
    if "every day" in expr or "daily" in expr:
        return base + 86400
    if "every week" in expr or "weekly" in expr:
        return base + 7 * 86400

    # Bare day-of-week mentions (no time): schedule 7 days out
    bare_days = {
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "mondays", "tuesdays", "wednesdays",
        "thursdays", "fridays", "saturdays", "sundays",
    }
    if any(d in expr for d in bare_days):
        return base + 7 * 86400

    # Default: daily
    return base + 86400
