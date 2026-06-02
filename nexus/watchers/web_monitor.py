"""
Web monitor: periodically fetches URLs and compares content hashes.
Emits a 'web_change_detected' event when content changes.
"""
from __future__ import annotations

import hashlib
import time

import httpx

from ..events import emit
from ..memory import db


class WebMonitor:

    async def tick(self) -> None:
        monitors = await db.list_web_monitors(enabled_only=True)
        now = time.time()
        for mon in monitors:
            interval = int(mon.get("check_interval_seconds") or 3600)
            last_checked = mon.get("last_checked") or 0
            if now - last_checked < interval:
                continue
            try:
                await self._check_monitor(mon)
            except Exception as e:
                await emit("web_monitor_error", monitor_id=mon["id"], error=str(e))

    async def _check_monitor(self, mon: dict) -> None:
        url = mon["url"]
        mon_id = mon["id"]
        now = time.time()

        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                          headers={"User-Agent": "NexusMonitor/1.0"}) as client:
                r = await client.get(url)
                r.raise_for_status()
                content_hash = hashlib.sha256(r.content).hexdigest()
        except Exception as e:
            await db.update_web_monitor(mon_id, last_checked=now)
            await emit("web_monitor_error", monitor_id=mon_id, url=url, error=str(e))
            return

        prev_hash = mon.get("last_hash")
        changed = prev_hash is not None and content_hash != prev_hash

        await db.update_web_monitor(mon_id,
            last_hash=content_hash,
            last_checked=now,
            **({"last_changed": now} if changed else {}),
        )

        if changed:
            await emit("web_change_detected",
                       monitor_id=mon_id,
                       url=url,
                       label=mon.get("label", url),
                       message=f"Content changed at: {mon.get('label', url)}")
