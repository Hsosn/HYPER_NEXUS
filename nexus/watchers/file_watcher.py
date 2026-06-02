"""
Polling-based file system watcher.
No external dependencies — uses os.stat snapshots compared each heartbeat tick.
Detects: created, modified, deleted files within watched paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .. import config
from ..events import emit
from ..memory import db


class FileWatcher:
    """Compares directory snapshots each tick to detect changes."""

    async def tick(self) -> None:
        watches = await db.list_file_watches(enabled_only=True)
        for watch in watches:
            try:
                await self._check_watch(watch)
            except Exception as e:
                await emit("file_watcher_error", watch_id=watch["id"], error=str(e))

    async def _check_watch(self, watch: dict) -> None:
        watch_id  = watch["id"]
        watch_path = Path(watch["path"])

        if not watch_path.exists():
            return

        # Build current snapshot: {rel_path_str: "mtime:size"}
        current: dict[str, str] = {}
        try:
            if watch_path.is_file():
                st = watch_path.stat()
                current[watch_path.name] = f"{st.st_mtime:.3f}:{st.st_size}"
            else:
                for root, _dirs, files in os.walk(watch_path):
                    for fname in files:
                        fpath = Path(root) / fname
                        try:
                            st = fpath.stat()
                            rel = str(fpath.relative_to(watch_path))
                            current[rel] = f"{st.st_mtime:.3f}:{st.st_size}"
                        except OSError:
                            pass
        except OSError:
            return

        # Compare with previous snapshot
        prev_raw = watch.get("last_snapshot")
        prev: dict[str, str] = json.loads(prev_raw) if prev_raw else {}

        created  = [p for p in current if p not in prev]
        deleted  = [p for p in prev    if p not in current]
        modified = [p for p in current if p in prev and current[p] != prev[p]]

        # Persist new snapshot (always, regardless of changes)
        await db.update_file_watch(watch_id,
            last_snapshot=json.dumps(current),
        )

        # Skip first run (prev was empty) — just take baseline
        if not prev:
            await db.update_file_watch(watch_id, last_checked=time.time())
            return

        for path in created:
            await db.add_workspace_event("created", path, watch_id)
            await emit("workspace_change", event_type="created", path=path,
                       watch_id=watch_id, label=watch.get("label", ""))

        for path in deleted:
            await db.add_workspace_event("deleted", path, watch_id)
            await emit("workspace_change", event_type="deleted", path=path,
                       watch_id=watch_id, label=watch.get("label", ""))

        for path in modified:
            await db.add_workspace_event("modified", path, watch_id)
            await emit("workspace_change", event_type="modified", path=path,
                       watch_id=watch_id, label=watch.get("label", ""))

        # Track when the last check happened and when a change was last detected
        now = time.time()
        update: dict[str, Any] = {"last_checked": now}
        if created or deleted or modified:
            update["last_changed"] = now
        await db.update_file_watch(watch_id, **update)
