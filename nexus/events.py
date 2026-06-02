"""
Event bus for broadcasting agent activity to the WebUI in real time.
Any subsystem can publish events; the WebSocket layer subscribes and streams them.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    kind: str                         # "thought" | "action" | "observation" | "message" | ...
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._history: deque[Event] = deque(maxlen=500)
        self._lock = asyncio.Lock()

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def publish(self, event: Event) -> None:
        async with self._lock:
            self._history.append(event)
            # Clean up stale subscribers (queues with closed/blocked tasks)
            dead: list[asyncio.Queue[Event]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Try to remove full/drained items to make room
                    try:
                        q.get_nowait()
                        q.task_done()
                        q.put_nowait(event)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        logger.warning(
                            "Event dropped for subscriber %s: queue full (size=%d)",
                            id(q), q.qsize(),
                        )
                except (RuntimeError, AttributeError):
                    # Queue is closed or in invalid state — mark for removal
                    dead.append(q)
            # Remove dead subscribers
            for q in dead:
                self._subscribers.remove(q)
                logger.debug("Removed dead subscriber %s", id(q))

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        # Use itertools.islice to avoid converting the entire deque to a list
        # before slicing — O(limit) instead of O(len(history)).
        return [e.to_dict() for e in itertools.islice(self._history, max(0, len(self._history) - limit), None)]


BUS = EventBus()


# Pre-compiled DSML stripping patterns for system-level defense-in-depth.
# Strips DSML/XML markup from event payloads going to the frontend WebSocket.
# This catches any DSML that might leak past backend-level stripping.
_DSML_STRIP_PATTERNS = [
    re.compile(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>[\s\S]*?<\/[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?\/>', re.IGNORECASE),
    re.compile(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>', re.IGNORECASE),
    re.compile(r'<DSML_\w+[^>]*>[\s\S]*?</DSML_\w+>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<tool_call>[\s\S]*?<\/tool_call>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<invoke[^>]*>[\s\S]*?<\/invoke>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<tool_calls>[\s\S]*?<\/tool_calls>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<(parameter|tool_response|tool_result|error|result|response|tool_input|input|output|status|invoke)[^>]*>[\s\S]*?<\/\1>', re.DOTALL | re.IGNORECASE),
    # Catch-all for DSML/XML tags — only strip known DSML tag patterns, not generic angle-bracket content
    re.compile(r'<(?:[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+|DSML_\w+)(?:\s+[^>]*)?\/?>', re.IGNORECASE),
]

def _strip_dsml_from_data(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip DSML markup from all string values in a dict.
    System-level defense-in-depth — catches any DSML that escapes per-source stripping.
    """
    if not data:
        return data
    result = {}
    for k, v in data.items():
        if isinstance(v, str):
            val = v
            for p in _DSML_STRIP_PATTERNS:
                val = p.sub('', val)
            val = re.sub(r'\n{3,}', '\n\n', val)
            result[k] = val.strip()
        elif isinstance(v, dict):
            result[k] = _strip_dsml_from_data(v)
        elif isinstance(v, list):
            items = []
            for item in v:
                if isinstance(item, str):
                    val = item
                    for p in _DSML_STRIP_PATTERNS:
                        val = p.sub('', val)
                    val = re.sub(r'\n{3,}', '\n\n', val)
                    items.append(val.strip())
                elif isinstance(item, dict):
                    items.append(_strip_dsml_from_data(item))
                else:
                    items.append(item)
            result[k] = items
        else:
            result[k] = v
    return result


async def emit(kind: str, **data: Any) -> None:
    """Emit an event to all subscribers.
    System-level DSML stripping: strips DSML/XML markup from ALL string event data
    before publishing, as defense-in-depth against DSML leaking to the frontend.
    """
    stripped = _strip_dsml_from_data(data) if data else data
    # Extract session_id from data if present and pass it to Event constructor
    session_id = stripped.pop('session_id', None) if stripped else None
    await BUS.publish(Event(kind=kind, data=stripped, session_id=session_id))
