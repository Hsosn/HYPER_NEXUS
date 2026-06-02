"""
Notification system.

Stores notifications in the DB and broadcasts them over the WS event bus.
The WebUI picks them up and shows a persistent bell with unread count.
"""
from __future__ import annotations

import time
from .events import emit
from .memory import db


async def notify(title: str, body: str = "", kind: str = "info") -> int:
    """Create a notification, persist it, and broadcast via WS."""
    nid = await db.add_notification(title=title, body=body, kind=kind)
    await emit("notification", id=nid, title=title, body=body, kind=kind)
    return nid
