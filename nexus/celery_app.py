"""
Celery application configuration for Nexus periodic task scheduling.

Uses Redis as both the broker and the result backend.  Worker processes
initialise the SQLite database on startup so that every async
DB function can be called safely from the synchronous Celery task body
via the ``_async_helper.run_async`` bridge.
"""
from __future__ import annotations

import os
import logging

from celery import Celery, signals

logger = logging.getLogger(__name__)


def _redis_url() -> str:
    """Resolve the Redis URL from Nexus config or the REDIS_URL env-var."""
    from . import config
    return config.get(
        "celery_broker_url",
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    )


# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------
celery_app = Celery("nexus")

try:
    _rurl = _redis_url()
except Exception:
    _rurl = "redis://localhost:6379/0"
celery_app.conf.update(
    broker_url=_rurl,
    result_backend=_rurl,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,      # one task at a time per worker
    task_acks_late=True,
    task_track_started=True,
    task_default_max_retries=2,
    task_default_retry_delay=30,
    # soft / hard time-limits (seconds)
    task_soft_time_limit=600 if os.name != 'nt' else None,  # SIGUSR1 missing on Windows
    task_time_limit=660,
)

# Auto-discover task modules inside nexus.tasks
celery_app.autodiscover_tasks(["nexus.tasks"])

# Explicitly import task modules to ensure they are registered
# This is a fallback in case autodiscover doesn't work on this platform
from .tasks import scheduler_tasks, agent_tasks, event_bridge, _async_helper

# Beat schedule – populated dynamically by heartbeat.start()
celery_app.conf.beat_schedule = {}


# ---------------------------------------------------------------------------
# Worker lifecycle signals
# ---------------------------------------------------------------------------
@signals.worker_process_init.connect
def _on_worker_process_init(**kwargs) -> None:
    """Create a persistent asyncio event loop, init the DB pool, and
    monkey-patch ``events.emit`` so every event produced inside a worker
    is also pushed to the Redis event bridge for the main process to
    consume.

    Also ensures the fast_trace_task optimization is disabled on Windows
    / Python 3.14 before ``setup_worker_optimizations`` runs, preventing
    ``TypeError: 'NoneType' object is not callable`` from cached ``__trace__``
    lookups that return None in spawned worker processes.
    """
    import celery.app.trace as _celery_trace

    # Windows + Python 3.14: disable fast_trace_task FIRST so that
    # setup_worker_optimizations uses the safe trace_task_ret path.
    if os.name == 'nt':
        celery_app.use_fast_trace_task = False

    try:
        _celery_trace.setup_worker_optimizations(celery_app)
    except Exception as _exc:
        logger.warning(
            "Celery worker: setup_worker_optimizations failed — %s", _exc
        )

    from .tasks._async_helper import get_loop, run_async
    from .memory import database as _db
    from . import events as _evt
    from .tasks.event_bridge import publish_event

    # Ensure a usable event loop exists in this worker process.
    get_loop()

    # Initialise the SQLite database connection.
    if _db._db is None:
        try:
            run_async(_db.init())
            logger.info("Celery worker: DB initialised")
        except Exception as exc:
            logger.error("Celery worker: DB init failed — %s", exc)

    # Bridge emit() → Redis so the main-process event consumer can relay.
    async def _bridged_emit(kind: str, **data):
        publish_event(kind, **data)

    _evt.emit = _bridged_emit
    logger.info("Celery worker: emit() bridged to Redis")


@signals.worker_process_shutdown.connect
def _on_worker_process_shutdown(**kwargs) -> None:
    """Gracefully close the DB pool."""
    from .tasks._async_helper import run_async
    from .memory import database as _db

    try:
        run_async(_db.close())
        logger.info("Celery worker: DB pool closed")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Celery worker optimisation workaround for Python 3.14 compatibility
# ---------------------------------------------------------------------------
# Windows + Python 3.14: disable the broken fast_trace_task optimization FIRST.
# The cached __trace__ lookups can return None in spawned worker processes,
# causing ``TypeError: 'NoneType' object is not callable`` on every task.
# The safe trace_task_ret path is slower but reliable.
if os.name == 'nt':
    celery_app.use_fast_trace_task = False

# Now call setup_worker_optimizations — with use_fast_trace_task=False
# on Windows the cached references use the safe trace_task_ret path.
try:
    import celery.app.trace as _celery_trace
    _celery_trace.setup_worker_optimizations(celery_app)
except Exception as _exc:
    logger.debug("Module-level setup_worker_optimizations failed: %s", _exc)


# ---------------------------------------------------------------------------
# Additional safety net: if fast_trace_task is still active and __trace__ is
# None, fall back to the normal task execution path.  This guards against
# Celery versions or codepaths where the flag above is not sufficient.
# ---------------------------------------------------------------------------
def _safe_fast_trace_task_wrapper(self, *args, **kwargs):
    """Wrap fast_trace_task so None __trace__ falls back to trace_task_ret."""
    try:
        return _celery_trace.fast_trace_task(self, *args, **kwargs)
    except TypeError as exc:
        if "'NoneType' object is not callable" in str(exc):
            return _celery_trace.trace_task_ret(self, *args, **kwargs)
        raise


if os.name == 'nt' and hasattr(_celery_trace, 'fast_trace_task'):
    _celery_trace.fast_trace_task = _safe_fast_trace_task_wrapper
