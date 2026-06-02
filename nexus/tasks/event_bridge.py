"""
Event bridge between Celery worker processes and the main asyncio event bus.

Celery tasks run in separate (possibly remote) processes and therefore
cannot call ``events.emit()`` directly — the asyncio event loop and the
WebSocket subscribers live only in the main process.

Solution
--------
* **Worker side (sync):** ``publish_event(kind, **data)`` serialises the
  event to JSON and appends it to a Redis list ``nexus:events``.
* **Main process (async):** ``start_event_consumer()`` runs as a
  long-lived asyncio task, blocking on ``BLPOP`` for that same list.
  When a message arrives it is deserialised and ``events.emit()`` is
  called on the main event loop.

Distributed lock
----------------
``acquire_autorun_lock`` / ``release_autorun_lock`` provide a Redis-based
mutex that replaces the in-process ``asyncio.Lock`` used by the old
heartbeat, allowing multiple Celery workers to safely share the autorun
concurrency guarantee.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

EVENT_LIST_KEY = "nexus:events"
AUTORUN_LOCK_KEY = "nexus:autorun_lock"
AUTORUN_LOCK_TTL = 300  # seconds – same as the old asyncio.Lock timeout

# Goal-dispatch tracking (Redis hash)
GOAL_DISPATCH_KEY = "nexus:goal_dispatches"
GOAL_DISPATCH_CLEANUP_KEY = "nexus:goal_dispatch_cleanup"

# Shared state keys (for cross-task state like integrity-check status)
STATE_PREFIX = "nexus:state:"


# ── Redis connection helpers ──────────────────────────────────────────────

_sync_redis = None


def _get_sync_redis():
    """Return (and cache) a synchronous ``redis.Redis`` client."""
    global _sync_redis
    if _sync_redis is None:
        import redis as _redis_mod
        from .. import config
        url = config.get(
            "celery_broker_url",
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        )
        _sync_redis = _redis_mod.from_url(url, decode_responses=True)
    return _sync_redis


# ═════════════════════════════════════════════════════════════════════════
# Publisher (called from synchronous Celery tasks)
# ═════════════════════════════════════════════════════════════════════════

def publish_event(kind: str, **data: Any) -> None:
    """Push an event onto the Redis list so the main process can emit it.

    This function is **synchronous** and safe to call from any Celery task.
    """
    try:
        client = _get_sync_redis()
        payload = json.dumps({
            "kind": kind,
            "data": data,
            "timestamp": time.time(),
        })
        client.rpush(EVENT_LIST_KEY, payload)
    except Exception as exc:
        logger.warning("event_bridge: failed to publish '%s': %s", kind, exc)


# ═════════════════════════════════════════════════════════════════════════
# Consumer (runs as an asyncio task in the main process)
# ═════════════════════════════════════════════════════════════════════════

async def start_event_consumer() -> None:
    """Block forever, reading events from Redis and calling ``emit()``.

    Intended to be launched as::

        asyncio.create_task(start_event_consumer())

    It exits cleanly on ``asyncio.CancelledError``.
    """
    import redis.asyncio as redis_asyncio
    from .. import config
    from ..events import emit

    url = config.get(
        "celery_broker_url",
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    )
    client = redis_asyncio.from_url(url, decode_responses=True)

    try:
        while True:
            try:
                result = await client.blpop(EVENT_LIST_KEY, timeout=2)
                if result is None:
                    continue
                _, raw = result
                event = json.loads(raw)
                kind: str = event.get("kind", "")
                data: dict = event.get("data", {})
                await emit(kind, **data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("event_bridge consumer error: %s", exc)
                await asyncio.sleep(1)
    finally:
        await client.close()
        logger.info("event_bridge consumer stopped")


# ═════════════════════════════════════════════════════════════════════════
# Distributed lock (replaces asyncio.Lock for autorun serialisation)
# ═════════════════════════════════════════════════════════════════════════

def acquire_autorun_lock(timeout: int = AUTORUN_LOCK_TTL) -> bool:
    """Try to set the autorun lock in Redis.  Returns ``True`` on success."""
    client = _get_sync_redis()
    return bool(client.set(AUTORUN_LOCK_KEY, "1", nx=True, ex=timeout))


def release_autorun_lock() -> None:
    """Release the autorun lock."""
    try:
        _get_sync_redis().delete(AUTORUN_LOCK_KEY)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════
# Goal-dispatch tracking (Redis hash – works across workers)
# ═════════════════════════════════════════════════════════════════════════

def get_last_goal_dispatch(goal_id: int) -> float:
    """Return the Unix timestamp of the last dispatch for *goal_id*, or 0."""
    client = _get_sync_redis()
    val = client.hget(GOAL_DISPATCH_KEY, str(goal_id))
    return float(val) if val else 0.0


def set_last_goal_dispatch(goal_id: int, ts: float) -> None:
    """Record the most-recent dispatch timestamp for *goal_id*."""
    _get_sync_redis().hset(GOAL_DISPATCH_KEY, str(goal_id), str(ts))


def cleanup_goal_dispatches(cutoff: float) -> int:
    """Remove dispatch entries older than *cutoff*.  Returns count removed."""
    client = _get_sync_redis()
    all_entries = client.hgetall(GOAL_DISPATCH_KEY)
    removed = 0
    pipe = client.pipeline()
    for gid_str, ts_str in all_entries.items():
        try:
            if float(ts_str) < cutoff:
                pipe.hdel(GOAL_DISPATCH_KEY, gid_str)
                removed += 1
        except ValueError:
            pipe.hdel(GOAL_DISPATCH_KEY, gid_str)
            removed += 1
    pipe.execute()
    return removed


def get_goal_dispatch_cleanup_ts() -> float:
    """Return the last cleanup timestamp, or 0."""
    client = _get_sync_redis()
    val = client.get(GOAL_DISPATCH_CLEANUP_KEY)
    return float(val) if val else 0.0


def set_goal_dispatch_cleanup_ts(ts: float) -> None:
    _get_sync_redis().set(GOAL_DISPATCH_CLEANUP_KEY, str(ts))


# ═════════════════════════════════════════════════════════════════════════
# Simple shared state helpers (Redis strings)
# ═════════════════════════════════════════════════════════════════════════

def get_state(key: str, default: str = "0") -> str:
    return _get_sync_redis().get(STATE_PREFIX + key) or default


def set_state(key: str, value: str) -> None:
    _get_sync_redis().set(STATE_PREFIX + key, value)
