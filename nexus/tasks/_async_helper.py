"""
Async helper – lets synchronous Celery task bodies call the Nexus async APIs.

A single *persistent* event loop is kept alive for the lifetime of each
worker process.  ``run_async(coro)`` schedules the coroutine on that loop
and blocks until it completes, making it safe to call from any Celery task
without creating/destroying loops.
"""
from __future__ import annotations

import asyncio

_loop: asyncio.AbstractEventLoop | None = None


def get_loop() -> asyncio.AbstractEventLoop:
    """Return (or create) the persistent event loop for this process."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


def run_async(coro):
    """Run an *async* coroutine from a synchronous context."""
    loop = get_loop()
    return loop.run_until_complete(coro)
