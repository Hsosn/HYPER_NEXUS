"""
Celery tasks for autonomous agent execution.

These tasks replace the old ``_run_agent_on_goal()`` and ``_autorun_task()``
coroutines that were previously fire-and-forget ``asyncio.create_task()`` calls
inside the heartbeat loop.  By running as proper Celery tasks they benefit from:
  * retry / error handling via Celery
  * distributed locking via Redis (``event_bridge.acquire_autorun_lock``)
  * visibility into execution via Flower / Celery result backend
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..celery_app import celery_app
from .event_bridge import publish_event, acquire_autorun_lock, release_autorun_lock
from . import _async_helper as _ah
from .event_bridge import _get_sync_redis

logger = logging.getLogger(__name__)

_CHAT_LOCK_KEY = "nexus:chat_lock"
_CHAT_LOCK_TIMEOUT = 120  # seconds


def acquire_chat_lock(session_id: str, timeout: int = _CHAT_LOCK_TIMEOUT) -> bool:
    """Acquire a session-specific chat lock to prevent parallel messages corrupting state."""
    client = _get_sync_redis()
    key = f"{_CHAT_LOCK_KEY}:{session_id}"
    return bool(client.set(key, "1", nx=True, ex=timeout))


def release_chat_lock(session_id: str) -> None:
    """Release the chat lock for a session."""
    try:
        client = _get_sync_redis()
        key = f"{_CHAT_LOCK_KEY}:{session_id}"
        client.delete(key)
    except Exception:
        pass


_AUTORUN_SESSION_ID = "autorun"
_AUTORUN_LOCK_TIMEOUT = 300  # seconds
_autorun_inline_lock = asyncio.Lock()


# ═════════════════════════════════════════════════════════════════════════
# run_agent_on_goal
# ═════════════════════════════════════════════════════════════════════════


@celery_app.task(
    name="nexus.tasks.agent_tasks.run_agent_on_goal",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=1,
    default_retry_delay=60,
)
def run_agent_on_goal(
    goal_id: int,
    title: str,
    description: str,
    priority: int,
    context: str = "",
) -> None:
    """Run the agent on a goal and update goal progress.

    Uses a Redis distributed lock so only *one* autonomous agent run is
    active across all Celery workers at any given time.
    """
    if not acquire_autorun_lock(timeout=_AUTORUN_LOCK_TIMEOUT):
        logger.warning(
            "Autorun lock busy for >%ds, skipping goal %d ('%s')",
            _AUTORUN_LOCK_TIMEOUT, goal_id, title,
        )
        return

    try:
        _ah.run_async(_do_run_agent_on_goal(
            goal_id, title, description, priority, context,
        ))
    except Exception as exc:
        logger.exception("run_agent_on_goal failed for goal %d", goal_id)
        publish_event(
            "goal_agent_error", goal_id=goal_id, error=str(exc),
            message=f"Agent error on goal '{title}': {exc}",
        )
        raise
    finally:
        release_autorun_lock()


async def _do_run_agent_on_goal(
    goal_id: int,
    title: str,
    description: str,
    priority: int,
    context: str = "",
) -> None:
    from ..reasoning import ReasoningEngine
    from ..memory import db

    await db.create_session(_AUTORUN_SESSION_ID, "autonomous")

    context_line = f"\nContext: {context}" if context else ""
    prompt = (
        f"[AUTONOMOUS GOAL — ID {goal_id}] Priority: {priority}/10\n"
        f"Title: {title}\n"
        f"Description: {description or '(none provided)'}"
        f"{context_line}\n\n"
        "Work on this goal using all available tools. Break it into steps, "
        "execute them one by one, and update the goal progress when sub-tasks "
        "are completed. Be thorough and autonomous."
    )

    publish_event(
        "autorun_start", goal_id=goal_id, title=title,
        session_id=_AUTORUN_SESSION_ID,
        message=f"Agent autonomously working on goal: {title}",
    )

    engine = ReasoningEngine(_AUTORUN_SESSION_ID)
    try:
        await engine._load_cross_session_memories()
    except Exception:
        pass

    try:
        result = await engine.respond(prompt)

        # Parallel: calculate goal progress + persist notification concurrently
        goal_tasks, _ = await asyncio.gather(
            db.list_tasks(goal_id=goal_id),
            db.add_notification(
                title="Goal progress",
                body=f"{title}: {(result or '').strip()[:120]}",
                kind="goal",
            ),
            return_exceptions=True,
        )

        if isinstance(goal_tasks, list):
            total = len(goal_tasks)
            done = sum(1 for t in goal_tasks if t.get("status") == "done")
            new_progress = done / total if total > 0 else (
                0.25 if "error" in (result or "").lower() or "failed" in (result or "").lower() else 0.5
            )
        else:
            new_progress = 0.5

        await db.update_goal(goal_id, progress=new_progress)

        publish_event(
            "goal_progress", goal_id=goal_id, title=title,
            result=(result or "")[:400],
            message=f"Goal progress: {title}",
        )
    except Exception as e:
        publish_event(
            "goal_agent_error", goal_id=goal_id, error=str(e),
            message=f"Agent error on goal '{title}': {e}",
        )


# ═════════════════════════════════════════════════════════════════════════
# autorun_task
# ═════════════════════════════════════════════════════════════════════════


@celery_app.task(
    name="nexus.tasks.agent_tasks.autorun_task",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=1,
    default_retry_delay=60,
)
def autorun_task(task_id: int, title: str) -> None:
    """Execute a scheduled / NL-schedule task through the ReasoningEngine."""
    if not acquire_autorun_lock(timeout=_AUTORUN_LOCK_TIMEOUT):
        logger.warning(
            "Autorun lock busy for >%ds, skipping task %d ('%s')",
            _AUTORUN_LOCK_TIMEOUT, task_id, title,
        )
        return

    try:
        _ah.run_async(_do_autorun_task(task_id, title))
    except Exception as exc:
        logger.exception("autorun_task failed for task %d", task_id)
        publish_event(
            "autorun_error", task_id=task_id, title=title,
            session_id=_AUTORUN_SESSION_ID,
            error=str(exc),
            message=f"Scheduled task failed: {title}",
        )
        raise
    finally:
        release_autorun_lock()


async def run_autorun_task_inline(task_id: int, title: str) -> None:
    """Run autorun task inside the inline heartbeat (no Celery worker).

    Serializes via an in-process asyncio.Lock since all inline tasks
    share the same ``_AUTORUN_SESSION_ID`` and would corrupt each
    other's session state if run concurrently.
    """
    async with _autorun_inline_lock:
        await _do_autorun_task(task_id, title)


async def _do_autorun_task(task_id: int, title: str) -> None:
    from ..reasoning import ReasoningEngine
    from ..memory import db

    await db.create_session(_AUTORUN_SESSION_ID, "autonomous")

    publish_event(
        "autorun_start", task_id=task_id, title=title,
        session_id=_AUTORUN_SESSION_ID,
        message=f"Agent running scheduled task: {title}",
    )

    engine = ReasoningEngine(_AUTORUN_SESSION_ID)
    try:
        await engine._load_cross_session_memories()
    except Exception:
        pass

    try:
        result = await engine.respond(title)

        # Parallel update: task status + notification
        await asyncio.gather(
            db.update_task(
                task_id, status="done",
                result=(result or "")[:4000],
                updated_at=time.time(),
            ),
            db.add_notification(
                title="Task completed",
                body=(
                    f"{title} — "
                    f"{(result or '').strip().splitlines()[0][:160] if result else 'done'}"
                ),
                kind="task_done",
            ),
            return_exceptions=True,
        )

        publish_event(
            "autorun_done", task_id=task_id, title=title,
            session_id=_AUTORUN_SESSION_ID,
            preview=(result or "")[:240],
            message=f"Scheduled task completed: {title}",
        )
    except Exception as e:
        await asyncio.gather(
            db.update_task(
                task_id, status="failed",
                result=f"[error] {type(e).__name__}: {e}"[:4000],
                updated_at=time.time(),
            ),
            db.add_notification(
                title="Task failed",
                body=f"{title} — {e}",
                kind="task_failed",
            ),
            return_exceptions=True,
        )
        publish_event(
            "autorun_error", task_id=task_id, title=title,
            session_id=_AUTORUN_SESSION_ID,
            error=str(e),
            message=f"Scheduled task failed: {title}",
        )


# ═════════════════════════════════════════════════════════════════════════
# Batch task execution (new)
# ═════════════════════════════════════════════════════════════════════════


@celery_app.task(
    name="nexus.tasks.agent_tasks.autorun_batch",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=0,
)
def autorun_batch(tasks: list[tuple[int, str]]) -> None:
    """Execute multiple tasks in batch.
    
    Tasks that are independent can run concurrently. Each task
    acquires and releases the autorun lock independently.
    """
    for task_id, title in tasks:
        try:
            autorun_task.delay(task_id, title)
        except Exception as exc:
            logger.error("Failed to dispatch batch task %d: %s", task_id, exc)


# ═════════════════════════════════════════════════════════════════════════
# chat_message task
# ═════════════════════════════════════════════════════════════════════════


@celery_app.task(
    name="nexus.tasks.agent_tasks.chat_message",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=2,
    default_retry_delay=30,
    task_soft_time_limit=600,
)
def chat_message(
    session_id: str,
    message: str,
    images: list[str] | None = None,
) -> str:
    """Process a user chat message through the ReasoningEngine via Celery.

    Uses a session lock to prevent parallel messages corrupting state.
    """
    logger.info("chat_message task received for session %s", session_id)

    if not acquire_chat_lock(session_id):
        logger.warning("Chat lock failed for session %s", session_id)
        publish_event(
            "chat_busy",
            session_id=session_id,
            message="Another message is still processing. Please wait.",
        )
        return

    try:
        logger.info("Calling _do_chat_message for session %s", session_id)
        _ah.run_async(_do_chat_message(session_id, message, images or []))
        logger.info("_do_chat_message completed for session %s", session_id)
    except Exception as exc:
        logger.exception("chat_message failed for session %s", session_id)
        publish_event(
            "chat_error",
            session_id=session_id,
            error=str(exc),
            message=f"Chat processing failed: {exc}",
        )
        raise
    finally:
        release_chat_lock(session_id)


async def _do_chat_message(session_id: str, message: str, images: list[str]) -> None:
    from pathlib import Path as _P
    from ..reasoning import ReasoningEngine
    from ..memory import db
    from .. import config

    logger.info("_do_chat_message starting for session %s", session_id)

    await db.create_session(session_id, "chat")

    publish_event(
        "chat_started",
        session_id=session_id,
        preview=message[:200] if len(message) > 200 else message,
        message="Processing your message...",
    )

    workspace = _P(config.BASE_DIR) / "data" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    def _snapshot(ws: _P) -> dict[str, float]:
        snap: dict[str, float] = {}
        for entry in ws.rglob("*"):
            if entry.is_file():
                try:
                    snap[str(entry.relative_to(ws))] = entry.stat().st_mtime
                except Exception:
                    pass
        return snap

    before = _snapshot(workspace)

    logger.info("Creating ReasoningEngine for session %s", session_id)
    engine = ReasoningEngine(session_id)
    try:
        await engine._load_cross_session_memories()
    except Exception as e:
        logger.warning("Failed to load cross-session memories: %s", e)

    try:
        logger.info("Calling engine.respond for session %s", session_id)
        result = await engine.respond(message, images=images if images else None)
        logger.info("engine.respond completed for session %s", session_id)

        # Parallel: save message + snapshot workspace files
        after = _snapshot(workspace)
        new_or_modified: list[dict] = []
        for rel_path, mtime in after.items():
            if rel_path not in before or before[rel_path] != mtime:
                try:
                    size = (workspace / rel_path).stat().st_size
                except Exception:
                    size = 0
                new_or_modified.append({
                    "name": _P(rel_path).name,
                    "path": rel_path,
                    "size": size,
                })

        # Direct publish_event call (not async, but non-blocking for the event bus)
        publish_event(
            "final_answer",
            content=result,
            session_id=session_id,
            workspace_files=new_or_modified,
            message="Response ready",
        )

        await db.add_message(session_id, "assistant", result)
        logger.info("Final answer published for session %s", session_id)
    except Exception as e:
        logger.exception("Error in _do_chat_message for session %s", session_id)
        publish_event(
            "chat_error",
            session_id=session_id,
            error=str(e),
            message=f"Chat error: {e}",
        )
        raise
