"""
Task State Checkpointing & Resume System.

Provides interval-based checkpoint persistence for long-running agent tasks.
Enables the agent to:
- Save execution progress at configurable intervals (auto-save)
- Resume from the last checkpoint after interruption
- Track task progress with rich metadata (steps done, tools called, tokens used)
- Handle graceful shutdown / Celery worker restart without losing state

Architecture:
    ┌─ CheckpointManager ──────────────────────────────────────────┐
    │  - auto_save_interval: how often to persist (seconds)        │
    │  - _checkpoint_dir: data/checkpoints/<session_id>/           │
    │  - save() / load() / resume_all() / prune_stale()            │
    │  - Thread-safe via asyncio.Lock                              │
    └──────────────────────────────────────────────────────────────┘

Usage:
    from .task_state import CheckpointManager
    mgr = CheckpointManager(session_id)
    await mgr.update_progress(steps_done=3, total_steps=10, current_step="Building API routes")
    # On resume:
    state = await mgr.load()
    if state and not state.completed:
        await mgr.resume_last_tool_call()  # Replays last incomplete step

All data is stored as JSON in data/checkpoints/<session_id>/.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..config import BASE_DIR

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"
_DEFAULT_AUTO_SAVE_INTERVAL = 30  # seconds — save every 30s by default
_MAX_CHECKPOINT_AGE = 86400 * 7   # 7 days — prune checkpoints older than this
_MAX_CHECKPOINTS_PER_SESSION = 50  # max checkpoints per session (ring buffer)
_STALE_CHECK_INTERVAL = 3600       # 1h — how often to scan for stale checkpoints

# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class TaskState:
    """Complete snapshot of agent execution state at a point in time.

    This is persisted as JSON and loaded on resume.
    """
    session_id: str
    user_request: str
    task_type: str = "general"

    # Progress tracking
    steps_planned: list[str] = field(default_factory=list)
    steps_completed: list[dict] = field(default_factory=list)
    current_step: str = ""
    current_step_index: int = 0

    # Tool execution history (abbreviated for checkpoint size)
    tool_calls: list[dict] = field(default_factory=list)
    last_tool_result: str = ""

    # Resource tracking
    token_usage: dict = field(default_factory=dict)
    total_duration_ms: float = 0.0

    # Context
    recent_messages: list[dict] = field(default_factory=list)

    # Metadata
    started_at: float = 0.0
    last_save_at: float = 0.0
    completed: bool = False
    error: str | None = None
    version: int = 1  # schema version for forward-compat

    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if not self.steps_planned:
            return 0.0
        total = len(self.steps_planned)
        done = len(self.steps_completed)
        return min(done / total, 1.0) if total > 0 else 0.0

    def elapsed_seconds(self) -> float:
        """Seconds since task started."""
        if self.started_at <= 0:
            return 0.0
        return time.time() - self.started_at

    def estimated_remaining(self) -> float:
        """Estimate remaining time based on progress so far."""
        pct = self.progress_pct()
        if pct <= 0:
            return 0.0
        elapsed = self.elapsed_seconds()
        return (elapsed / pct) - elapsed

    def summary(self) -> str:
        """Human-readable summary of current state."""
        lines = [
            f"Task: {self.user_request[:80]}",
            f"Progress: {len(self.steps_completed)}/{len(self.steps_planned)} steps ({self.progress_pct():.0%})",
            f"Elapsed: {self.elapsed_seconds():.0f}s",
        ]
        if self.current_step:
            lines.append(f"Current: {self.current_step}")
        if self.steps_planned:
            remaining = self.steps_planned[self.current_step_index:]
            if remaining:
                lines.append(f"Remaining: {len(remaining)} steps")
        return " | ".join(lines)


# ── Checkpoint Manager ─────────────────────────────────────────────────────────


class CheckpointManager:
    """Persists and restores agent task execution state.

    Usage:
        mgr = CheckpointManager(
            session_id="abc123",
            auto_save_interval=30,  # checkpoint every 30s
        )

        # During execution:
        await mgr.init(request="Build user API")
        await mgr.plan_steps(["Step 1: Create model", "Step 2: Create route", ...])
        for i, step in enumerate(plan):
            await mgr.begin_step(i, step)
            # ... execute step ...
            await mgr.complete_step(result="...")
            # Auto-save happens automatically at interval

        # On resume:
        state = await mgr.load()
        if state and not state.completed:
            await mgr.resume(state)
            # Agent continues from state.current_step_index
    """

    def __init__(
        self,
        session_id: str,
        auto_save_interval: int = _DEFAULT_AUTO_SAVE_INTERVAL,
        checkpoint_dir: Path | None = None,
    ):
        self.session_id = session_id
        self.auto_save_interval = auto_save_interval
        self._base_dir = checkpoint_dir or _DEFAULT_CHECKPOINT_DIR
        self._session_dir = self._base_dir / session_id
        self._state: TaskState | None = None
        self._lock = asyncio.Lock()
        self._last_save_time: float = 0.0
        self._dirty: bool = False
        self._auto_save_task: asyncio.Task | None = None
        self._running: bool = False

        # Ensure directories exist
        self._session_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    async def init(self, request: str, task_type: str = "general") -> TaskState:
        """Initialize a new task state checkpoint."""
        async with self._lock:
            self._state = TaskState(
                session_id=self.session_id,
                user_request=request,
                task_type=task_type,
                started_at=time.time(),
                last_save_at=time.time(),
            )
            self._dirty = True
            await self._persist()
            self._running = True
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())
            return self._state

    async def plan_steps(self, steps: list[str]) -> None:
        """Record the plan of steps."""
        async with self._lock:
            if self._state:
                self._state.steps_planned = steps
                self._dirty = True

    async def begin_step(self, index: int, description: str) -> None:
        """Mark a step as in-progress."""
        async with self._lock:
            if self._state:
                self._state.current_step_index = index
                self._state.current_step = description
                self._dirty = True

    async def complete_step(
        self,
        result: str = "",
        tool_calls: list[dict] | None = None,
        tokens: dict | None = None,
    ) -> None:
        """Mark the current step as complete and record results."""
        async with self._lock:
            if not self._state:
                return
            self._state.steps_completed.append({
                "index": self._state.current_step_index,
                "description": self._state.current_step,
                "result": result[:500],
                "tool_calls": (tool_calls or [])[:5],  # Keep last 5 tool calls per step
                "timestamp": time.time(),
            })
            if tokens:
                for k, v in tokens.items():
                    self._state.token_usage[k] = self._state.token_usage.get(k, 0) + v
            self._dirty = True
            # Persist immediately on step completion for safety
            await self._persist()

    async def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a tool call in the checkpoint (abbreviated)."""
        async with self._lock:
            if not self._state:
                return
            self._state.tool_calls.append({
                "tool": tool_name,
                "args_preview": str(args)[:100],
                "result_preview": str(result)[:200],
                "success": success,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            })
            # Keep only the last N tool calls to limit checkpoint size
            if len(self._state.tool_calls) > 20:
                self._state.tool_calls = self._state.tool_calls[-20:]
            self._state.last_tool_result = str(result)[:300]
            self._state.total_duration_ms += duration_ms
            self._dirty = True

    async def update_progress(
        self,
        steps_done: int | None = None,
        total_steps: int | None = None,
        current_step: str | None = None,
    ) -> None:
        """Update progress metadata without completing a step."""
        async with self._lock:
            if not self._state:
                return
            if current_step is not None:
                self._state.current_step = current_step
            self._dirty = True

    async def set_error(self, error: str) -> None:
        """Record an error state."""
        async with self._lock:
            if self._state:
                self._state.error = error
                self._dirty = True
                await self._persist()

    async def mark_completed(self) -> None:
        """Mark the task as completed."""
        async with self._lock:
            if self._state:
                self._state.completed = True
                self._state.last_save_at = time.time()
                self._dirty = True
                await self._persist()
            self._running = False
            if self._auto_save_task:
                self._auto_save_task.cancel()
                self._auto_save_task = None

    async def load(self) -> TaskState | None:
        """Load the latest checkpoint for this session.

        Returns None if no checkpoint exists.
        """
        latest = self._find_latest_checkpoint()
        if latest is None:
            return None
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            self._state = TaskState(**data)
            return self._state
        except Exception as e:
            logger.warning("Failed to load checkpoint %s: %s", latest, e)
            return None

    async def resume(self, state: TaskState | None = None) -> TaskState | None:
        """Resume execution from a loaded state.

        Returns the state to resume from, or None if nothing to resume.
        """
        if state is None:
            state = await self.load()
        if state is None or state.completed:
            return None

        self._state = state
        self._running = True
        self._auto_save_task = asyncio.create_task(self._auto_save_loop())

        logger.info(
            "Resumed task %s at step %d/%d (%.0f%%)",
            self.session_id,
            state.current_step_index,
            len(state.steps_planned),
            state.progress_pct() * 100,
        )
        return state

    async def resume_last_tool_call(self) -> dict | None:
        """Get the last incomplete tool call (if any) for replay on resume.

        Useful when the agent was mid-tool-call when interrupted.
        """
        if not self._state or not self._state.tool_calls:
            return None
        last_call = self._state.tool_calls[-1]
        if last_call.get("success"):
            return None  # Last call succeeded, nothing to resume
        return last_call

    def get_state(self) -> TaskState | None:
        """Get current in-memory state (thread-safe read without lock)."""
        return self._state

    # ── Cleanup / Maintenance ──────────────────────────────────────────────

    async def cleanup(self) -> None:
        """Clean up checkpoint files for this session."""
        if self._session_dir.exists():
            shutil.rmtree(self._session_dir, ignore_errors=True)
        self._state = None
        self._running = False
        if self._auto_save_task:
            self._auto_save_task.cancel()
            self._auto_save_task = None

    @classmethod
    async def prune_stale_checkpoints(
        cls,
        max_age: int = _MAX_CHECKPOINT_AGE,
        base_dir: Path | None = None,
    ) -> int:
        """Remove all checkpoints older than max_age seconds.

        Returns count of pruned sessions.
        """
        base = base_dir or _DEFAULT_CHECKPOINT_DIR
        if not base.exists():
            return 0

        now = time.time()
        pruned = 0
        for session_dir in base.iterdir():
            if not session_dir.is_dir():
                continue
            # Check if any checkpoint file is recent enough
            latest = 0.0
            for f in session_dir.iterdir():
                if f.suffix == ".json":
                    try:
                        mtime = f.stat().st_mtime
                        latest = max(latest, mtime)
                    except Exception:
                        pass
            if latest > 0 and (now - latest) > max_age:
                shutil.rmtree(session_dir, ignore_errors=True)
                pruned += 1

        if pruned:
            logger.info("Pruned %d stale checkpoint sessions", pruned)
        return pruned

    # ── Internal ───────────────────────────────────────────────────────────

    async def _auto_save_loop(self) -> None:
        """Background loop that auto-saves at the configured interval."""
        while self._running:
            try:
                await asyncio.sleep(self.auto_save_interval)
                if self._dirty and self._state:
                    async with self._lock:
                        if self._dirty:
                            await self._persist()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Auto-save error: %s", e)

    async def _persist(self) -> None:
        """Write current state to disk."""
        if not self._state:
            return
        try:
            self._state.last_save_at = time.time()
            data = asdict(self._state)
            # Use timestamp as filename for historical tracking
            filename = f"checkpoint_{int(time.time())}.json"
            path = self._session_dir / filename
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            self._last_save_time = time.time()
            self._dirty = False

            # Prune old checkpoints (ring buffer)
            self._prune_old_checkpoints()
        except Exception as e:
            logger.warning("Checkpoint persist error: %s", e)

    def _find_latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint file."""
        if not self._session_dir.exists():
            return None
        json_files = sorted(
            [f for f in self._session_dir.iterdir() if f.suffix == ".json"],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return json_files[0] if json_files else None

    def _prune_old_checkpoints(self) -> None:
        """Remove oldest checkpoints when exceeding max count."""
        if not self._session_dir.exists():
            return
        json_files = sorted(
            [f for f in self._session_dir.iterdir() if f.suffix == ".json"],
            key=lambda f: f.stat().st_mtime,
        )
        while len(json_files) > _MAX_CHECKPOINTS_PER_SESSION:
            oldest = json_files.pop(0)
            try:
                oldest.unlink()
            except Exception:
                pass


# ── Module-level convenience functions ────────────────────────────────────────

_checkpoint_managers: dict[str, CheckpointManager] = {}


def get_checkpoint_manager(
    session_id: str,
    auto_save_interval: int = _DEFAULT_AUTO_SAVE_INTERVAL,
) -> CheckpointManager:
    """Get or create a CheckpointManager for a session."""
    if session_id not in _checkpoint_managers:
        _checkpoint_managers[session_id] = CheckpointManager(
            session_id=session_id,
            auto_save_interval=auto_save_interval,
        )
    return _checkpoint_managers[session_id]


async def prune_all_stale_checkpoints() -> int:
    """Prune all stale checkpoints across all sessions."""
    return await CheckpointManager.prune_stale_checkpoints()


# ── Background pruning task ──────────────────────────────────────────────────

async def periodic_checkpoint_prune() -> None:
    """Run periodically to clean up old checkpoints."""
    while True:
        try:
            await asyncio.sleep(_STALE_CHECK_INTERVAL)
            await prune_all_stale_checkpoints()
        except asyncio.CancelledError:
            break
        except Exception:
            pass
