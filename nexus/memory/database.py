"""
SQLite-backed persistent storage for memory, goals, tasks, sessions, and logs.
Uses aiosqlite for async access with a single-writer connection.

Desktop-optimized SQLite storage:
- Single connection with asyncio.Lock for thread safety
- SQLite positional placeholders (?)
- INTEGER PRIMARY KEY AUTOINCREMENT for auto-ID columns
- REAL for floating-point columns
- INSERT OR IGNORE/REPLACE for upserts
- cursor.lastrowid for RETURNING semantics
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
_DB_PATH = os.environ.get("DATABASE_URL", "").replace("sqlite:///", "")
if not _DB_PATH:
    _DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nexus.db")
_DB_PATH = os.path.abspath(_DB_PATH)

_db: aiosqlite.Connection | None = None
_write_lock = asyncio.Lock()

# ── Schema (SQLite syntax) ───────────────────────────────────────────────────
_SCHEMA_STATEMENTS: list[str] = [
    # ── Core tables ────────────────────────────────────────────────────────
    """CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        created_at REAL,
        updated_at REAL,
        meta TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        meta TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT,
        content TEXT,
        importance REAL,
        embedding TEXT,
        session_id TEXT,
        tags TEXT,
        mood TEXT,
        permanent INTEGER DEFAULT 0,
        created_at REAL,
        last_accessed REAL,
        access_count INTEGER DEFAULT 0
    )""",

    """CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id INTEGER,
        title TEXT,
        description TEXT,
        status TEXT,
        priority INTEGER,
        progress REAL,
        session_id TEXT,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id INTEGER,
        title TEXT,
        status TEXT,
        result TEXT,
        scheduled_for REAL,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tool_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        tool_name TEXT,
        input TEXT,
        output TEXT,
        success INTEGER,
        duration_ms REAL,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS custom_tools (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        description TEXT,
        "schema" TEXT,
        code TEXT,
        enabled INTEGER DEFAULT 1,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS custom_skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        description TEXT,
        category TEXT DEFAULT 'custom',
        icon TEXT DEFAULT 'PLG',
        skill_md TEXT,
        code TEXT,
        file_path TEXT,
        enabled INTEGER DEFAULT 1,
        source_type TEXT DEFAULT 'upload',
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS reflections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        content TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS nexus_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        task TEXT,
        steps TEXT,
        branches TEXT,
        chosen_path_id INTEGER,
        final_answer TEXT,
        reflection TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS file_watches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL,
        label TEXT,
        enabled INTEGER DEFAULT 1,
        last_snapshot TEXT,
        last_checked REAL,
        last_changed REAL,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS web_monitors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        label TEXT,
        enabled INTEGER DEFAULT 1,
        check_interval_seconds INTEGER DEFAULT 3600,
        last_hash TEXT,
        last_checked REAL,
        last_changed REAL,
        notify_on_change INTEGER DEFAULT 1,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS nl_schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        expression TEXT NOT NULL,
        task_title TEXT NOT NULL,
        goal_id INTEGER,
        enabled INTEGER DEFAULT 1,
        next_run REAL,
        last_run REAL,
        run_count INTEGER DEFAULT 0,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS improvement_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        tool_name TEXT,
        detail TEXT,
        frequency INTEGER DEFAULT 1,
        resolved INTEGER DEFAULT 0,
        memory_id INTEGER,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS workspace_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT,
        path TEXT,
        watch_id INTEGER,
        meta TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        body TEXT,
        kind TEXT,
        "read" INTEGER DEFAULT 0,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        mood TEXT,
        tags TEXT,
        people TEXT,
        importance REAL DEFAULT 0.7,
        permanent INTEGER DEFAULT 1,
        session_id TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS task_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT,
        answer TEXT,
        score REAL,
        dimensions TEXT,
        issues TEXT,
        approved INTEGER,
        session_id TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tool_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_name TEXT UNIQUE,
        total_calls INTEGER DEFAULT 0,
        success_count INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        consecutive_failures INTEGER DEFAULT 0,
        avg_duration_ms REAL,
        avg_output_len REAL,
        success_rate REAL,
        effectiveness_score REAL,
        last_used REAL,
        last_success REAL,
        last_failure REAL,
        learned_strategy TEXT,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS connected_integrations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        description TEXT,
        config TEXT DEFAULT '{}',
        connected INTEGER DEFAULT 1,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS mcp_servers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        transport TEXT DEFAULT 'stdio',
        command TEXT DEFAULT '',
        args TEXT DEFAULT '[]',
        env TEXT DEFAULT '{}',
        url TEXT DEFAULT '',
        headers TEXT DEFAULT '{}',
        description TEXT DEFAULT '',
        category TEXT DEFAULT 'custom',
        status TEXT DEFAULT 'disconnected',
        connected INTEGER DEFAULT 0,
        tools TEXT DEFAULT '[]',
        config TEXT DEFAULT '{}',
        auto_connect INTEGER DEFAULT 0,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS active_triggers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        integration_id TEXT DEFAULT '',
        config TEXT DEFAULT '{}',
        enabled INTEGER DEFAULT 1,
        last_fired REAL,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tool_affinities (
        tool_name TEXT NOT NULL,
        task_type TEXT NOT NULL,
        success_count INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        total_duration_ms REAL DEFAULT 0,
        avg_score REAL DEFAULT 0,
        last_used REAL,
        PRIMARY KEY (tool_name, task_type)
    )""",

    """CREATE TABLE IF NOT EXISTS task_analysis_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_hash INTEGER NOT NULL,
        predicted_complexity TEXT,
        predicted_strategy TEXT,
        signals_count INTEGER DEFAULT 0,
        actual_score REAL,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS task_trajectories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_request TEXT,
        task_type TEXT,
        reasoning_strategy TEXT,
        plan_steps TEXT,
        tool_calls TEXT,
        final_answer TEXT,
        quality_score REAL,
        token_usage TEXT,
        duration_seconds REAL,
        error TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS learning_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        task_type TEXT,
        content TEXT,
        confidence REAL DEFAULT 0.5,
        evidence_count INTEGER DEFAULT 1,
        status TEXT DEFAULT 'proposed',
        effectiveness REAL DEFAULT 0.0,
        created_at REAL,
        last_validated REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tool_chains (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chain_hash TEXT NOT NULL UNIQUE,
        tools TEXT NOT NULL,
        task_type TEXT,
        success INTEGER DEFAULT 0,
        execution_count INTEGER DEFAULT 1,
        avg_quality_score REAL DEFAULT 0.0,
        avg_duration_ms REAL DEFAULT 0.0,
        last_seen REAL,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS reasoning_preferences (
        task_type TEXT PRIMARY KEY,
        preferred_strategy TEXT,
        confidence REAL DEFAULT 0.5,
        sample_count INTEGER DEFAULT 0,
        scores_by_strategy TEXT,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS webhook_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT,
        processed INTEGER DEFAULT 0,
        error TEXT,
        created_at REAL,
        processed_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS webhook_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service TEXT NOT NULL,
        url TEXT,
        secret TEXT,
        events TEXT,
        enabled INTEGER DEFAULT 1,
        last_triggered REAL,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS global_usage (
        id INTEGER PRIMARY KEY DEFAULT 1,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0.0,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS memory_archived (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_id INTEGER,
        kind TEXT,
        content TEXT,
        importance REAL,
        embedding TEXT,
        session_id TEXT,
        tags TEXT,
        mood TEXT,
        archived_reason TEXT,
        archived_at REAL,
        original_created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS weakness_tracker (
        weakness_key TEXT PRIMARY KEY,
        count INTEGER DEFAULT 0,
        first_seen REAL,
        last_seen REAL,
        addressed INTEGER DEFAULT 0,
        updated_at REAL
    )""",

    # ── Indexes ───────────────────────────────────────────────────────────
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)",
    "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)",
    "CREATE INDEX IF NOT EXISTS idx_file_watches_enabled ON file_watches(enabled)",
    "CREATE INDEX IF NOT EXISTS idx_web_monitors_enabled ON web_monitors(enabled)",
    "CREATE INDEX IF NOT EXISTS idx_nl_schedules_next_run ON nl_schedules(next_run)",
    "CREATE INDEX IF NOT EXISTS idx_improvement_log_tool ON improvement_log(tool_name)",
    "CREATE INDEX IF NOT EXISTS idx_improvement_log_category_resolved ON improvement_log(category, resolved)",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)",
    "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)",
    "CREATE INDEX IF NOT EXISTS idx_memories_permanent ON memories(permanent)",
    "CREATE INDEX IF NOT EXISTS idx_workspace_events_time ON workspace_events(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(\"read\")",
    "CREATE INDEX IF NOT EXISTS idx_journal_created ON journal_entries(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_task_scores_approved ON task_scores(approved)",
    "CREATE INDEX IF NOT EXISTS idx_tool_profiles_name ON tool_profiles(tool_name)",
    "CREATE INDEX IF NOT EXISTS idx_tool_profiles_effectiveness ON tool_profiles(effectiveness_score DESC)",
    "CREATE INDEX IF NOT EXISTS idx_connected_integrations ON connected_integrations(connected)",
    "CREATE INDEX IF NOT EXISTS idx_task_trajectories_type ON task_trajectories(task_type)",
    "CREATE INDEX IF NOT EXISTS idx_task_trajectories_session ON task_trajectories(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_task_trajectories_score ON task_trajectories(quality_score)",
    "CREATE INDEX IF NOT EXISTS idx_task_trajectories_created ON task_trajectories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_learning_insights_category ON learning_insights(category)",
    "CREATE INDEX IF NOT EXISTS idx_learning_insights_task_type ON learning_insights(task_type)",
    "CREATE INDEX IF NOT EXISTS idx_learning_insights_status ON learning_insights(status)",
    "CREATE INDEX IF NOT EXISTS idx_tool_chains_task_type ON tool_chains(task_type)",
    "CREATE INDEX IF NOT EXISTS idx_tool_chains_success ON tool_chains(success)",
    "CREATE INDEX IF NOT EXISTS idx_webhook_events_pending ON webhook_events(processed)",
    "CREATE INDEX IF NOT EXISTS idx_webhook_events_service ON webhook_events(service)",
    "CREATE INDEX IF NOT EXISTS idx_memory_archived_kind ON memory_archived(kind)",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ LIFECYCLE ═════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def init(database_url: str | None = None) -> None:
    """Open the SQLite database and ensure all tables exist.

    Must be called once during application startup before any other DB function.
    """
    global _db, _DB_PATH
    if _db is not None:
        return
    if database_url:
        _path = database_url.replace("sqlite:///", "")
        if _path:
            _DB_PATH = os.path.abspath(_path)
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    _db = await aiosqlite.connect(_DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=OFF")
    await _db.execute("PRAGMA busy_timeout=5000")
    for stmt in _SCHEMA_STATEMENTS:
        try:
            await _db.execute(stmt)
        except Exception as exc:
            logger.warning("Schema statement failed: %s", exc)
    await _db.commit()

    # ── Memory Tree schema ───────────────────────────────────────────────
    try:
        from .tree_db import TREE_SCHEMA
        for stmt in TREE_SCHEMA:
            try:
                await _db.execute(stmt)
            except Exception as exc:
                logger.warning("Tree schema statement failed: %s", exc)
        await _db.commit()
    except ImportError:
        logger.warning("tree_db module not available; skipping tree schema")


async def close() -> None:
    """Close the SQLite database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ INTERNAL HELPERS ══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _connect():
    """Provide the database connection (autocommit on exit). Raises if not initialised."""
    if _db is None:
        raise RuntimeError("Database not initialised. Call init() first.")
    yield _db
    await _db.commit()


def _sanitize_str(value: str | None) -> str | None:
    """Sanitise a string for JSON serialisation."""
    if value is None:
        return None
    sanitised = value.strip().replace("\x00", "")
    return sanitised if sanitised else None


def _row_to_dict(row: aiosqlite.Row | None) -> dict:
    """Convert an aiosqlite Row to a plain dict."""
    if row is None:
        return {}
    return dict(row)


async def _execute(sql: str, params: tuple = ()) -> aiosqlite.Cursor:
    """Execute a write query with lock protection."""
    async with _write_lock:
        c = await _db.execute(sql, params)
        await _db.commit()
        return c


async def _fetchall(sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
    """Fetch multiple rows."""
    c = await _db.execute(sql, params)
    return await c.fetchall()


async def _fetchone(sql: str, params: tuple = ()) -> aiosqlite.Row | None:
    """Fetch a single row."""
    c = await _db.execute(sql, params)
    return await c.fetchone()


async def _fetchval(sql: str, params: tuple = ()) -> Any:
    """Fetch first column of first row."""
    c = await _db.execute(sql, params)
    row = await c.fetchone()
    return row[0] if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ SESSIONS API ══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def create_session(session_id: str, name: str = "New Session") -> None:
    now = time.time()
    await _execute(
        "INSERT OR IGNORE INTO sessions (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, name, now, now),
    )


async def list_sessions(limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


async def touch_session(session_id: str) -> None:
    await _execute(
        "UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), session_id)
    )


async def delete_session(session_id: str) -> None:
    async with _write_lock:
        await _db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        await _db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        await _db.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ MESSAGES API ══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_message(
    session_id: str, role: str, content: str, meta: dict | None = None,
    *, tool_calls: list | None = None, tool_call_id: str | None = None,
    name: str | None = None,
) -> int:
    msg_meta = dict(meta or {})
    if tool_calls:
        msg_meta["tool_calls"] = tool_calls
    if tool_call_id:
        msg_meta["tool_call_id"] = tool_call_id
    if name:
        msg_meta["name"] = name
    c = await _execute(
        "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, json.dumps(msg_meta) if msg_meta else None, time.time()),
    )
    return c.lastrowid or 0


async def get_messages(session_id: str, limit: int = 200) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM messages WHERE session_id=? ORDER BY id ASC LIMIT ?",
        (session_id, limit),
    )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ MEMORIES API ══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_memory(
    kind: str, content: str, importance: float,
    embedding: list[float] | None = None, session_id: str | None = None,
    tags: list[str] | None = None,
) -> int:
    c = await _execute(
        """INSERT INTO memories (kind, content, importance, embedding, session_id, tags, created_at, last_accessed, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (kind, content, importance, json.dumps(embedding) if embedding else None,
         session_id, json.dumps(tags) if tags else None, time.time(), time.time(), 0),
    )
    return c.lastrowid or 0


async def count_memories() -> int:
    val = await _fetchval("SELECT COUNT(*) FROM memories")
    return val or 0


async def all_memories(kind: str | None = None, limit: int = 500) -> list[dict]:
    if kind:
        rows = await _fetchall(
            "SELECT * FROM memories WHERE kind=? ORDER BY created_at DESC LIMIT ?",
            (kind, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    return [_row_to_dict(r) for r in rows]


async def search_memories(query: str, limit: int = 50, kind: str | None = None) -> list[dict]:
    like = f"%{query}%"
    if kind:
        rows = await _fetchall(
            "SELECT * FROM memories WHERE kind=? AND content LIKE ? ORDER BY importance DESC, created_at DESC LIMIT ?",
            (kind, like, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY importance DESC, created_at DESC LIMIT ?",
            (like, limit),
        )
    return [_row_to_dict(r) for r in rows]


async def delete_memory(mem_id: int) -> None:
    await _execute("DELETE FROM memories WHERE id=?", (mem_id,))


async def top_factual_memories(limit: int = 8) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM memories WHERE kind='factual' ORDER BY importance DESC, last_accessed DESC LIMIT ?",
        (limit,),
    )
    return [_row_to_dict(r) for r in rows]


async def find_memory_by_content(content: str) -> dict | None:
    row = await _fetchone(
        "SELECT * FROM memories WHERE content=? LIMIT 1", (content,)
    )
    return _row_to_dict(row) if row else None


async def find_similar_memory(content: str, threshold: float = 0.85) -> dict | None:
    # Simple content match fallback (no embedding similarity in SQLite)
    row = await _fetchone(
        "SELECT * FROM memories WHERE content=? LIMIT 1", (content,)
    )
    return _row_to_dict(row) if row else None


async def update_memory_access(mem_id: int) -> None:
    await _execute(
        "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
        (time.time(), mem_id),
    )


async def batch_update_memory_access(mem_ids: list[int]) -> None:
    now = time.time()
    async with _write_lock:
        for mid in mem_ids:
            await _db.execute(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
                (now, mid),
            )
        await _db.commit()


async def memory_count(kind: str | None = None) -> int:
    if kind:
        val = await _fetchval("SELECT COUNT(*) FROM memories WHERE kind=?", (kind,))
    else:
        val = await _fetchval("SELECT COUNT(*) FROM memories")
    return val or 0


async def memory_counts_by_kind() -> dict[str, int]:
    rows = await _fetchall("SELECT kind, COUNT(*) as cnt FROM memories GROUP BY kind")
    return {r["kind"]: r["cnt"] for r in rows}


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ GOALS API ═════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_goal(
    title: str, description: str, priority: int = 5,
    parent_id: int | None = None, session_id: str | None = None,
) -> int:
    now = time.time()
    c = await _execute(
        "INSERT INTO goals (parent_id, title, description, status, priority, session_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (parent_id, title, description, "active", priority, session_id, now, now),
    )
    return c.lastrowid or 0


async def list_goals(status: str | None = None) -> list[dict]:
    if status:
        rows = await _fetchall(
            "SELECT * FROM goals WHERE status=? ORDER BY priority DESC, created_at DESC",
            (status,),
        )
    else:
        rows = await _fetchall("SELECT * FROM goals ORDER BY priority DESC, created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def update_goal(goal_id: int, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [goal_id]
    await _execute(f"UPDATE goals SET {sets} WHERE id=?", tuple(vals))


async def delete_goal(goal_id: int) -> None:
    await _execute("DELETE FROM goals WHERE id=?", (goal_id,))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TASKS API ═════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_task(goal_id: int | None, title: str, scheduled_for: float | None = None) -> int:
    now = time.time()
    c = await _execute(
        "INSERT INTO tasks (goal_id, title, status, scheduled_for, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (goal_id, title, "pending", scheduled_for, now, now),
    )
    return c.lastrowid or 0


async def list_tasks(status: str | None = None, goal_id: int | None = None) -> list[dict]:
    if status and goal_id is not None:
        rows = await _fetchall(
            "SELECT * FROM tasks WHERE status=? AND goal_id=? ORDER BY created_at DESC",
            (status, goal_id),
        )
    elif status:
        rows = await _fetchall(
            "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC", (status,)
        )
    elif goal_id is not None:
        rows = await _fetchall(
            "SELECT * FROM tasks WHERE goal_id=? ORDER BY created_at DESC", (goal_id,)
        )
    else:
        rows = await _fetchall("SELECT * FROM tasks ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def update_task(task_id: int, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [task_id]
    await _execute(f"UPDATE tasks SET {sets} WHERE id=?", tuple(vals))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TOOL EXECUTIONS API ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def log_tool_execution(
    session_id: str, tool_name: str, input_data: dict,
    output: str, success: bool, duration_ms: float,
) -> None:
    await _execute(
        "INSERT INTO tool_executions (session_id, tool_name, input, output, success, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, tool_name, json.dumps(input_data), output, 1 if success else 0, duration_ms, time.time()),
    )


async def recent_tool_executions(limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM tool_executions ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ CUSTOM TOOLS API ══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_custom_tool(name: str, description: str, schema: dict, code: str) -> int:
    c = await _execute(
        "INSERT INTO custom_tools (name, description, \"schema\", code, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, description, json.dumps(schema), code, time.time()),
    )
    return c.lastrowid or 0


async def list_custom_tools() -> list[dict]:
    rows = await _fetchall("SELECT * FROM custom_tools ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def delete_custom_tool(tool_id: int) -> None:
    await _execute("DELETE FROM custom_tools WHERE id=?", (tool_id,))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ CUSTOM SKILLS API ═════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_custom_skill(
    name: str, description: str, category: str = "custom", icon: str = "PLG",
    skill_md: str = "", code: str = "", file_path: str = "",
    source_type: str = "upload",
) -> int:
    now = time.time()
    c = await _execute(
        """INSERT INTO custom_skills (name, description, category, icon, skill_md, code, file_path, source_type, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, description, category, icon, skill_md, code, file_path, source_type, now, now),
    )
    return c.lastrowid or 0


async def list_custom_skills() -> list[dict]:
    rows = await _fetchall("SELECT * FROM custom_skills ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def get_custom_skill(skill_id: int) -> dict | None:
    row = await _fetchone("SELECT * FROM custom_skills WHERE id=?", (skill_id,))
    return _row_to_dict(row) if row else None


async def get_custom_skill_by_name(name: str) -> dict | None:
    row = await _fetchone("SELECT * FROM custom_skills WHERE name=?", (name,))
    return _row_to_dict(row) if row else None


async def delete_custom_skill(skill_id: int) -> None:
    await _execute("DELETE FROM custom_skills WHERE id=?", (skill_id,))


async def toggle_custom_skill(skill_id: int, enabled: bool) -> bool:
    await _execute(
        "UPDATE custom_skills SET enabled=? WHERE id=?", (1 if enabled else 0, skill_id),
    )
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ REFLECTIONS API ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_reflection(session_id: str, content: str) -> int:
    c = await _execute(
        "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
        (session_id, content, time.time()),
    )
    return c.lastrowid or 0


async def recent_reflections(limit: int = 20) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ NEXUS TRACES API ══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_nexus_trace(
    session_id: str, task: str, steps: list, branches: list,
    chosen_path_id: int, final_answer: str, reflection: str,
) -> int:
    c = await _execute(
        """INSERT INTO nexus_traces (session_id, task, steps, branches, chosen_path_id, final_answer, reflection, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, task, json.dumps(steps), json.dumps(branches),
         chosen_path_id, final_answer, reflection, time.time()),
    )
    return c.lastrowid or 0


async def recent_nexus_traces(limit: int = 20) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM nexus_traces ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ FILE WATCHES & WORKSPACE EVENTS ═══════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_file_watch(path: str, label: str = "") -> int:
    c = await _execute(
        "INSERT INTO file_watches (path, label, created_at) VALUES (?, ?, ?)",
        (path, label, time.time()),
    )
    return c.lastrowid or 0


async def list_file_watches(enabled_only: bool = True) -> list[dict]:
    if enabled_only:
        rows = await _fetchall(
            "SELECT * FROM file_watches WHERE enabled=1 ORDER BY created_at DESC"
        )
    else:
        rows = await _fetchall("SELECT * FROM file_watches ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def update_file_watch(watch_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [watch_id]
    await _execute(f"UPDATE file_watches SET {sets} WHERE id=?", tuple(vals))


async def delete_file_watch(watch_id: int) -> None:
    await _execute("DELETE FROM file_watches WHERE id=?", (watch_id,))


async def add_workspace_event(
    event_type: str, path: str, watch_id: int | None = None,
    meta: dict | None = None,
) -> int:
    c = await _execute(
        "INSERT INTO workspace_events (event_type, path, watch_id, meta, created_at) VALUES (?, ?, ?, ?, ?)",
        (event_type, path, watch_id, json.dumps(meta) if meta else None, time.time()),
    )
    return c.lastrowid or 0


async def recent_workspace_events(limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM workspace_events ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ WEB MONITORS API ══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_web_monitor(url: str, label: str = "", interval_seconds: int = 3600) -> int:
    c = await _execute(
        "INSERT INTO web_monitors (url, label, check_interval_seconds, created_at) VALUES (?, ?, ?, ?)",
        (url, label, interval_seconds, time.time()),
    )
    return c.lastrowid or 0


async def list_web_monitors(enabled_only: bool = True) -> list[dict]:
    if enabled_only:
        rows = await _fetchall(
            "SELECT * FROM web_monitors WHERE enabled=1 ORDER BY created_at DESC"
        )
    else:
        rows = await _fetchall("SELECT * FROM web_monitors ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def update_web_monitor(monitor_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [monitor_id]
    await _execute(f"UPDATE web_monitors SET {sets} WHERE id=?", tuple(vals))


async def delete_web_monitor(monitor_id: int) -> None:
    await _execute("DELETE FROM web_monitors WHERE id=?", (monitor_id,))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ NL SCHEDULES API ══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_nl_schedule(
    expression: str, task_title: str, next_run: float,
    goal_id: int | None = None,
) -> int:
    c = await _execute(
        """INSERT INTO nl_schedules (expression, task_title, goal_id, next_run, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (expression, task_title, goal_id, next_run, time.time()),
    )
    return c.lastrowid or 0


async def list_nl_schedules(enabled_only: bool = True) -> list[dict]:
    if enabled_only:
        rows = await _fetchall(
            "SELECT * FROM nl_schedules WHERE enabled=1 ORDER BY created_at DESC"
        )
    else:
        rows = await _fetchall("SELECT * FROM nl_schedules ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def update_nl_schedule(schedule_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [schedule_id]
    await _execute(f"UPDATE nl_schedules SET {sets} WHERE id=?", tuple(vals))


async def delete_nl_schedule(schedule_id: int) -> None:
    await _execute("DELETE FROM nl_schedules WHERE id=?", (schedule_id,))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ IMPROVEMENT LOG API ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def log_improvement(
    category: str, detail: str, tool_name: str | None = None,
    frequency: int = 1,
) -> int:
    now = time.time()
    c = await _execute(
        "INSERT INTO improvement_log (category, tool_name, detail, frequency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (category, tool_name, detail, frequency, now, now),
    )
    return c.lastrowid or 0


async def list_improvement_log(resolved: bool = False, limit: int = 100) -> list[dict]:
    if resolved:
        rows = await _fetchall(
            "SELECT * FROM improvement_log WHERE resolved=1 ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM improvement_log WHERE resolved=0 ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
    return [_row_to_dict(r) for r in rows]


async def resolve_improvement(log_id: int, memory_id: int | None = None) -> None:
    now = time.time()
    if memory_id is not None:
        await _execute(
            "UPDATE improvement_log SET resolved=1, memory_id=?, updated_at=? WHERE id=?",
            (memory_id, now, log_id),
        )
    else:
        await _execute(
            "UPDATE improvement_log SET resolved=1, updated_at=? WHERE id=?", (now, log_id),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ NOTIFICATIONS API ═════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_notification(title: str, body: str = "", kind: str = "info") -> int:
    c = await _execute(
        "INSERT INTO notifications (title, body, kind, created_at) VALUES (?, ?, ?, ?)",
        (title, body, kind, time.time()),
    )
    return c.lastrowid or 0


async def list_notifications(unread_only: bool = False, limit: int = 50) -> list[dict]:
    if unread_only:
        rows = await _fetchall(
            "SELECT * FROM notifications WHERE \"read\"=0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM notifications ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    return [_row_to_dict(r) for r in rows]


async def mark_notifications_read(ids: list[int] | None = None) -> None:
    if ids:
        placeholders = ",".join("?" for _ in ids)
        await _execute(
            f"UPDATE notifications SET \"read\"=1 WHERE id IN ({placeholders})",
            tuple(ids),
        )
    else:
        await _execute("UPDATE notifications SET \"read\"=1")


async def unread_notification_count() -> int:
    val = await _fetchval("SELECT COUNT(*) FROM notifications WHERE \"read\"=0")
    return val or 0


async def cleanup_old_notifications(max_age_days: float = 30) -> int:
    cutoff = time.time() - max_age_days * 86400
    c = await _execute(
        "DELETE FROM notifications WHERE created_at<? AND \"read\"=1", (cutoff,)
    )
    return _db.total_changes if _db else 0


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ MEMORY CLEANUP / PRUNING ══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def cleanup_duplicate_memories() -> int:
    # Find duplicate content entries, keep the one with highest importance
    async with _write_lock:
        rows = await _fetchall(
            """SELECT m1.id FROM memories m1
               WHERE EXISTS (SELECT 1 FROM memories m2
                             WHERE m2.content=m1.content AND m2.id<m1.id)"""
        )
        ids = [r["id"] for r in rows]
        for mid in ids:
            await _db.execute("DELETE FROM memories WHERE id=?", (mid,))
        await _db.commit()
    return len(ids)


async def prune_stale_memories(
    max_age_days: float = 90, min_importance: float = 0.2,
    min_access_count: int = 1, max_memories: int = 1000,
) -> int:
    cutoff = time.time() - max_age_days * 86400
    total = await _fetchval("SELECT COUNT(*) FROM memories WHERE permanent=0") or 0
    if total <= max_memories:
        return 0
    excess = total - max_memories
    async with _write_lock:
        rows = await _fetchall(
            """SELECT id FROM memories
               WHERE permanent=0 AND created_at<? AND importance<? AND access_count<? AND kind NOT IN ('core', 'identity')
               ORDER BY importance ASC, last_accessed ASC LIMIT ?""",
            (cutoff, min_importance, min_access_count, excess),
        )
        ids = [r["id"] for r in rows]
        for mid in ids:
            await _db.execute("DELETE FROM memories WHERE id=?", (mid,))
        await _db.commit()
    return len(ids)


async def prune_memories_by_forgetting_curve(
    base_half_life_hours: float = 720.0,
    retention_threshold: float = 0.08,
    max_prune: int = 500,
) -> dict:
    now = time.time()
    rows = await _fetchall(
        """SELECT id, importance, last_accessed, access_count FROM memories
           WHERE permanent=0 AND kind NOT IN ('core', 'identity')
           ORDER BY importance ASC, last_accessed ASC"""
    )
    pruned = 0
    async with _write_lock:
        for row in rows:
            if pruned >= max_prune:
                break
            mem_id = row["id"]
            importance = row["importance"] or 0.5
            last_accessed = row["last_accessed"] or now
            access_count = row["access_count"] or 0
            hours_elapsed = (now - last_accessed) / 3600
            retention = 2 ** (-hours_elapsed / base_half_life_hours)
            retention *= 1 + 0.2 * min(access_count, 10)
            retention = min(retention, 1.0)
            if retention < retention_threshold:
                await _db.execute("DELETE FROM memories WHERE id=?", (mem_id,))
                pruned += 1
        await _db.commit()
    return {"pruned": pruned, "threshold": retention_threshold, "half_life_hours": base_half_life_hours}


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ JOURNAL API ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def add_journal_entry(
    content: str, mood: str = "", tags: list[str] | None = None,
    people: list[str] | None = None, importance: float = 0.7,
    permanent: bool = True, session_id: str | None = None,
) -> int:
    c = await _execute(
        """INSERT INTO journal_entries (content, mood, tags, people, importance, permanent, session_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, mood, json.dumps(tags) if tags else None,
         json.dumps(people) if people else None,
         importance, 1 if permanent else 0, session_id, time.time()),
    )
    return c.lastrowid or 0


async def list_journal_entries(limit: int = 50, mood: str | None = None) -> list[dict]:
    if mood:
        rows = await _fetchall(
            "SELECT * FROM journal_entries WHERE mood=? ORDER BY created_at DESC LIMIT ?",
            (mood, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM journal_entries ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    return [_row_to_dict(r) for r in rows]


async def delete_journal_entry(entry_id: int) -> None:
    await _execute("DELETE FROM journal_entries WHERE id=?", (entry_id,))


async def get_mood_summary(days: int = 7) -> list[dict]:
    cutoff = time.time() - days * 86400
    rows = await _fetchall(
        "SELECT mood, COUNT(*) as cnt FROM journal_entries WHERE created_at>? AND mood!='' GROUP BY mood ORDER BY cnt DESC",
        (cutoff,),
    )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TASK SCORES API ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def log_task_score(
    task: str, answer: str, score: float, dimensions: str = "",
    issues: str = "", approved: bool = True,
    session_id: str | None = None,
) -> int:
    c = await _execute(
        "INSERT INTO task_scores (task, answer, score, dimensions, issues, approved, session_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (task, answer, score, dimensions, issues, 1 if approved else 0, session_id, time.time()),
    )
    return c.lastrowid or 0


async def recent_task_scores(limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM task_scores ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


async def avg_quality_score() -> float:
    val = await _fetchval("SELECT AVG(score) FROM task_scores WHERE score IS NOT NULL")
    return val or 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TOOL PROFILES API ═════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def update_tool_profile(
    tool_name: str, success: bool, duration_ms: float, output_len: int = 0,
) -> None:
    now = time.time()
    existing = await _fetchone(
        "SELECT * FROM tool_profiles WHERE tool_name=?", (tool_name,)
    )
    if existing:
        tc = existing["total_calls"] + 1
        sc = existing["success_count"] + (1 if success else 0)
        fc = existing["failure_count"] + (0 if success else 1)
        cf = (existing["consecutive_failures"] + 1) if not success else 0
        sr = sc / tc if tc > 0 else 0
        avg_dur = ((existing["avg_duration_ms"] or 0) * (tc - 1) + duration_ms) / tc
        avg_len = ((existing["avg_output_len"] or 0) * (tc - 1) + output_len) / tc
        eff = sr * 0.7 + (1.0 - (avg_dur / 60000)) * 0.3 if avg_dur else sr
        await _execute(
            """UPDATE tool_profiles SET total_calls=?, success_count=?, failure_count=?, consecutive_failures=?,
               avg_duration_ms=?, avg_output_len=?, success_rate=?, effectiveness_score=?,
               last_used=?, last_success=?, last_failure=?, updated_at=? WHERE tool_name=?""",
            (tc, sc, fc, cf, avg_dur, avg_len, sr, eff, now,
             now if success else existing["last_success"],
             now if not success else existing["last_failure"], now, tool_name),
        )
    else:
        sr = 1.0 if success else 0.0
        eff = sr * 0.7 + 0.3
        await _execute(
            """INSERT INTO tool_profiles (tool_name, total_calls, success_count, failure_count, consecutive_failures,
               avg_duration_ms, avg_output_len, success_rate, effectiveness_score, last_used, last_success, last_failure, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tool_name, 1, 1 if success else 0, 0 if success else 1, 0 if success else 1,
             duration_ms, output_len, sr, eff, now,
             now if success else None, now if not success else None, now),
        )


async def get_tool_profiles() -> list[dict]:
    try:
        rows = await _fetchall(
            "SELECT * FROM tool_profiles ORDER BY effectiveness_score DESC"
        )
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.warning("get_tool_profiles query failed: %s", exc)
        return []


async def get_tool_profile(tool_name: str) -> dict | None:
    row = await _fetchone(
        "SELECT * FROM tool_profiles WHERE tool_name=?", (tool_name,)
    )
    return _row_to_dict(row) if row else None


async def get_tool_strategy(tool_name: str) -> str | None:
    row = await _fetchone(
        "SELECT learned_strategy FROM tool_profiles WHERE tool_name=?", (tool_name,)
    )
    return row["learned_strategy"] if row else None


async def update_tool_strategy(tool_name: str, strategy: str) -> None:
    await _execute(
        "UPDATE tool_profiles SET learned_strategy=?, updated_at=? WHERE tool_name=?",
        (strategy, time.time(), tool_name),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ CONNECTED INTEGRATIONS API ════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def upsert_integration(
    intg_id: str, name: str, category: str = "", description: str = "",
    connected: bool = True, config: dict | None = None,
) -> None:
    now = time.time()
    await _execute(
        """INSERT OR REPLACE INTO connected_integrations (id, name, category, description, config, connected, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM connected_integrations WHERE id=?), ?), ?)""",
        (intg_id, name, category, description, json.dumps(config or {}),
         1 if connected else 0, intg_id, now, now),
    )


async def add_integration(
    id: str, name: str, category: str = "", description: str = "",
    config: str = "", connected: int = 1,
) -> None:
    now = time.time()
    await _execute(
        "INSERT OR IGNORE INTO connected_integrations (id, name, category, description, config, connected, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (id, name, category, description, config, connected, now, now),
    )


async def toggle_integration(intg_id: str, connected: bool) -> bool:
    await _execute(
        "UPDATE connected_integrations SET connected=?, updated_at=? WHERE id=?",
        (1 if connected else 0, time.time(), intg_id),
    )
    return True


async def list_connected_integrations(connected_only: bool = True) -> list[dict]:
    if connected_only:
        rows = await _fetchall(
            "SELECT * FROM connected_integrations WHERE connected=1 ORDER BY created_at DESC"
        )
    else:
        rows = await _fetchall("SELECT * FROM connected_integrations ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def get_integration(intg_id: str) -> dict | None:
    row = await _fetchone(
        "SELECT * FROM connected_integrations WHERE id=?", (intg_id,)
    )
    return _row_to_dict(row) if row else None


async def get_integration_config(intg_id: str) -> dict:
    row = await _fetchone(
        "SELECT config FROM connected_integrations WHERE id=?", (intg_id,)
    )
    if row and row["config"]:
        try:
            return json.loads(row["config"])
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


async def update_integration_config(intg_id: str, config: dict) -> None:
    await _execute(
        "UPDATE connected_integrations SET config=?, updated_at=? WHERE id=?",
        (json.dumps(config), time.time(), intg_id),
    )


async def delete_integration(intg_id: str) -> None:
    await _execute("DELETE FROM connected_integrations WHERE id=?", (intg_id,))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ MCP SERVERS API ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def upsert_mcp_server(
    mcp_id: str, name: str, transport: str = "stdio", command: str = "",
    args: list | None = None, env: dict | None = None, url: str = "",
    headers: dict | None = None, description: str = "", category: str = "custom",
    status: str = "disconnected", connected: bool = False,
    tools: list | None = None, config: dict | None = None,
    auto_connect: bool = False,
) -> None:
    now = time.time()
    await _execute(
        """INSERT OR REPLACE INTO mcp_servers
           (id, name, transport, command, args, env, url, headers, description, category, status, connected, tools, config, auto_connect, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   COALESCE((SELECT created_at FROM mcp_servers WHERE id=?), ?), ?)""",
        (mcp_id, name, transport, command, json.dumps(args or []),
         json.dumps(env or {}), url, json.dumps(headers or {}), description,
         category, status, 1 if connected else 0, json.dumps(tools or []),
         json.dumps(config or {}), 1 if auto_connect else 0,
         mcp_id, now, now),
    )


async def toggle_mcp_server(mcp_id: str, connected: bool) -> bool:
    await _execute(
        "UPDATE mcp_servers SET connected=?, updated_at=? WHERE id=?",
        (1 if connected else 0, time.time(), mcp_id),
    )
    return True


async def update_mcp_server_status(
    mcp_id: str, status: str, tools: list | None = None,
) -> None:
    await _execute(
        "UPDATE mcp_servers SET status=?, tools=?, updated_at=? WHERE id=?",
        (status, json.dumps(tools or []), time.time(), mcp_id),
    )


async def list_mcp_servers(connected_only: bool = False) -> list[dict]:
    if connected_only:
        rows = await _fetchall(
            "SELECT * FROM mcp_servers WHERE connected=1 ORDER BY created_at DESC"
        )
    else:
        rows = await _fetchall("SELECT * FROM mcp_servers ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def get_mcp_server(mcp_id: str) -> dict | None:
    row = await _fetchone("SELECT * FROM mcp_servers WHERE id=?", (mcp_id,))
    return _row_to_dict(row) if row else None


async def delete_mcp_server(mcp_id: str) -> None:
    await _execute("DELETE FROM mcp_servers WHERE id=?", (mcp_id,))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ ACTIVE TRIGGERS API ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def upsert_trigger(
    trigger_id: str, name: str, trigger_type: str,
    integration_id: str = "", config: dict | None = None,
    enabled: bool = True,
) -> None:
    now = time.time()
    await _execute(
        """INSERT OR REPLACE INTO active_triggers
           (id, name, type, integration_id, config, enabled, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?,
                   COALESCE((SELECT created_at FROM active_triggers WHERE id=?), ?), ?)""",
        (trigger_id, name, trigger_type, integration_id, json.dumps(config or {}),
         1 if enabled else 0, trigger_id, now, now),
    )


async def toggle_trigger(trigger_id: str, enabled: bool) -> bool:
    await _execute(
        "UPDATE active_triggers SET enabled=?, updated_at=? WHERE id=?",
        (1 if enabled else 0, time.time(), trigger_id),
    )
    return True


async def list_triggers(enabled_only: bool = False) -> list[dict]:
    if enabled_only:
        rows = await _fetchall(
            "SELECT * FROM active_triggers WHERE enabled=1 ORDER BY created_at DESC"
        )
    else:
        rows = await _fetchall("SELECT * FROM active_triggers ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def delete_trigger(trigger_id: str) -> None:
    await _execute("DELETE FROM active_triggers WHERE id=?", (trigger_id,))


async def fire_trigger(trigger_id: str) -> None:
    await _execute(
        "UPDATE active_triggers SET last_fired=? WHERE id=?",
        (time.time(), trigger_id),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TASK ANALYSIS TRACKING ════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def log_task_analysis(
    task_hash: int, predicted_complexity: str, predicted_strategy: str,
    signals_count: int,
) -> None:
    await _execute(
        "INSERT INTO task_analysis_log (task_hash, predicted_complexity, predicted_strategy, signals_count, created_at) VALUES (?, ?, ?, ?, ?)",
        (task_hash, predicted_complexity, predicted_strategy, signals_count, time.time()),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TOOL-TASK AFFINITY ════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def record_tool_affinity(
    tool_name: str, task_type: str, success: bool, duration_ms: float,
) -> None:
    now = time.time()
    existing = await _fetchone(
        "SELECT * FROM tool_affinities WHERE tool_name=? AND task_type=?",
        (tool_name, task_type),
    )
    if existing:
        sc = existing["success_count"] + (1 if success else 0)
        fc = existing["failure_count"] + (0 if success else 1)
        td = existing["total_duration_ms"] + duration_ms
        avg = existing["avg_score"]
        await _execute(
            """UPDATE tool_affinities SET success_count=?, failure_count=?, total_duration_ms=?,
               avg_score=?, last_used=? WHERE tool_name=? AND task_type=?""",
            (sc, fc, td, avg, now, tool_name, task_type),
        )
    else:
        await _execute(
            "INSERT INTO tool_affinities (tool_name, task_type, success_count, failure_count, total_duration_ms, avg_score, last_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tool_name, task_type, 1 if success else 0, 0 if success else 1, duration_ms, 0.0, now),
        )


async def get_tool_affinities(task_type: str = "", top_k: int = 5) -> list[dict]:
    if task_type:
        rows = await _fetchall(
            "SELECT * FROM tool_affinities WHERE task_type=? ORDER BY success_count DESC, total_duration_ms ASC LIMIT ?",
            (task_type, top_k),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM tool_affinities ORDER BY success_count DESC, total_duration_ms ASC LIMIT ?",
            (top_k,),
        )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ EXPERIENCE REPLAY / GOLDEN TRACES ═════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def store_golden_trace(
    session_id: str, task_type: str, tools_used: list[str],
    answer_summary: str, score: float,
) -> None:
    await _execute(
        "INSERT INTO task_trajectories (session_id, task_type, tool_calls, final_answer, quality_score, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, task_type, json.dumps(tools_used), answer_summary, score, time.time()),
    )


async def get_golden_traces(task_type: str = "", limit: int = 3) -> list[dict]:
    if task_type:
        rows = await _fetchall(
            "SELECT * FROM task_trajectories WHERE task_type=? AND quality_score>=0.8 ORDER BY quality_score DESC, created_at DESC LIMIT ?",
            (task_type, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM task_trajectories WHERE quality_score>=0.8 ORDER BY quality_score DESC, created_at DESC LIMIT ?",
            (limit,),
        )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ SELF-IMPROVEMENT HELPERS ══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def get_reflections(limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


async def get_task_scores(limit: int = 20) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM task_scores ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


async def get_tool_executions(session_id: str = "", limit: int = 50) -> list[dict]:
    if session_id:
        rows = await _fetchall(
            "SELECT * FROM tool_executions WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM tool_executions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    return [_row_to_dict(r) for r in rows]


async def update_memory_importance(mem_id: int, new_importance: float) -> None:
    await _execute(
        "UPDATE memories SET importance=? WHERE id=?", (new_importance, mem_id),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TASK TRAJECTORIES API ═════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def store_trajectory(trajectory: dict) -> int:
    c = await _execute(
        """INSERT INTO task_trajectories (session_id, user_request, task_type, reasoning_strategy, plan_steps,
           tool_calls, final_answer, quality_score, token_usage, duration_seconds, error, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trajectory.get("session_id"), trajectory.get("user_request"),
         trajectory.get("task_type"), trajectory.get("reasoning_strategy"),
         json.dumps(trajectory.get("plan_steps", [])),
         json.dumps(trajectory.get("tool_calls", [])),
         trajectory.get("final_answer"), trajectory.get("quality_score"),
         json.dumps(trajectory.get("token_usage", {})),
         trajectory.get("duration_seconds"),
         trajectory.get("error"), time.time()),
    )
    return c.lastrowid or 0


async def retrieve_similar_trajectories(task_type: str, limit: int = 5) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM task_trajectories WHERE task_type=? AND quality_score>=0.5 ORDER BY quality_score DESC, created_at DESC LIMIT ?",
        (task_type, limit),
    )
    return [_row_to_dict(r) for r in rows]


async def get_recent_trajectories(limit: int = 10, min_score: float = 0.0) -> list[dict]:
    if min_score > 0:
        rows = await _fetchall(
            "SELECT * FROM task_trajectories WHERE quality_score>=? ORDER BY created_at DESC LIMIT ?",
            (min_score, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM task_trajectories ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    return [_row_to_dict(r) for r in rows]


async def get_trajectory_by_session(session_id: str) -> dict | None:
    row = await _fetchone(
        "SELECT * FROM task_trajectories WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    )
    return _row_to_dict(row) if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ LEARNING INSIGHTS API ═════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def store_learning_insight(insight: dict) -> int:
    now = time.time()
    c = await _execute(
        """INSERT INTO learning_insights (category, task_type, content, confidence, evidence_count, status, effectiveness, created_at, last_validated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (insight.get("category"), insight.get("task_type"), insight.get("content"),
         insight.get("confidence", 0.5), insight.get("evidence_count", 1),
         insight.get("status", "proposed"), insight.get("effectiveness", 0.0),
         now, now),
    )
    return c.lastrowid or 0


async def get_active_insights(category: str = "", limit: int = 10) -> list[dict]:
    if category:
        rows = await _fetchall(
            "SELECT * FROM learning_insights WHERE status='active' AND category=? ORDER BY effectiveness DESC, confidence DESC LIMIT ?",
            (category, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM learning_insights WHERE status='active' ORDER BY effectiveness DESC, confidence DESC LIMIT ?",
            (limit,),
        )
    return [_row_to_dict(r) for r in rows]


async def update_insight_effectiveness(insight_id: int, effectiveness: float) -> None:
    await _execute(
        "UPDATE learning_insights SET effectiveness=?, last_validated=? WHERE id=?",
        (effectiveness, time.time(), insight_id),
    )


async def get_insights_for_task_type(task_type: str, limit: int = 5) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM learning_insights WHERE task_type=? AND status='active' ORDER BY effectiveness DESC, confidence DESC LIMIT ?",
        (task_type, limit),
    )
    return [_row_to_dict(r) for r in rows]


async def decay_insight_confidence() -> int:
    now = time.time()
    # Decay confidence for insights not validated in 30 days
    cutoff = now - 30 * 86400
    c = await _execute(
        "UPDATE learning_insights SET confidence=MAX(confidence*0.95, 0.1) WHERE last_validated<? AND status='active'",
        (cutoff,),
    )
    return _db.total_changes if _db else 0


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ TOOL CHAINS / WEAKNESS TRACKER ════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def save_weakness_tracker(weakness_data: dict[str, dict]) -> None:
    now = time.time()
    async with _write_lock:
        for key, data in weakness_data.items():
            existing = await _fetchone(
                "SELECT * FROM weakness_tracker WHERE weakness_key=?", (key,)
            )
            if existing:
                await _db.execute(
                    "UPDATE weakness_tracker SET count=?, last_seen=?, addressed=?, updated_at=? WHERE weakness_key=?",
                    (data.get("count", existing["count"]), now,
                     data.get("addressed", existing["addressed"]), now, key),
                )
            else:
                await _db.execute(
                    "INSERT INTO weakness_tracker (weakness_key, count, first_seen, last_seen, addressed, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (key, data.get("count", 1), now, now, data.get("addressed", 0), now),
                )
        await _db.commit()


async def load_weakness_tracker() -> dict[str, dict]:
    rows = await _fetchall("SELECT * FROM weakness_tracker")
    result = {}
    for r in rows:
        d = dict(r)
        key = d.pop("weakness_key")
        result[key] = d
    return result


async def record_tool_chain(
    chain: list[str], success: bool, task_type: str,
    quality_score: float = 0.0, duration_ms: float = 0.0,
) -> None:
    chain_hash = hashlib.md5(json.dumps(chain, sort_keys=True).encode()).hexdigest()
    now = time.time()
    existing = await _fetchone(
        "SELECT * FROM tool_chains WHERE chain_hash=?", (chain_hash,)
    )
    if existing:
        ec = existing["execution_count"] + 1
        sc = existing["success"] + (1 if success else 0)
        aq = ((existing["avg_quality_score"] or 0) * (ec - 1) + quality_score) / ec
        ad = ((existing["avg_duration_ms"] or 0) * (ec - 1) + duration_ms) / ec
        await _execute(
            """UPDATE tool_chains SET tools=?, task_type=?, success=?, execution_count=?,
               avg_quality_score=?, avg_duration_ms=?, last_seen=?, created_at=? WHERE chain_hash=?""",
            (json.dumps(chain), task_type, sc, ec, aq, ad, now,
             existing["created_at"], chain_hash),
        )
    else:
        await _execute(
            "INSERT INTO tool_chains (chain_hash, tools, task_type, success, execution_count, avg_quality_score, avg_duration_ms, last_seen, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chain_hash, json.dumps(chain), task_type, 1 if success else 0, 1, quality_score, duration_ms, now, now),
        )


async def get_successful_chains(task_type: str = "", limit: int = 10) -> list[dict]:
    if task_type:
        rows = await _fetchall(
            "SELECT * FROM tool_chains WHERE task_type=? AND success>0 ORDER BY avg_quality_score DESC, execution_count DESC LIMIT ?",
            (task_type, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM tool_chains WHERE success>0 ORDER BY avg_quality_score DESC, execution_count DESC LIMIT ?",
            (limit,),
        )
    return [_row_to_dict(r) for r in rows]


async def get_anti_pattern_chains(task_type: str = "", limit: int = 10) -> list[dict]:
    if task_type:
        rows = await _fetchall(
            "SELECT * FROM tool_chains WHERE task_type=? AND success=0 AND execution_count>=2 ORDER BY execution_count DESC LIMIT ?",
            (task_type, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM tool_chains WHERE success=0 AND execution_count>=2 ORDER BY execution_count DESC LIMIT ?",
            (limit,),
        )
    return [_row_to_dict(r) for r in rows]


async def get_tool_cooccurrence(limit: int = 20) -> list[dict]:
    rows = await _fetchall(
        "SELECT tools, execution_count, avg_quality_score FROM tool_chains ORDER BY execution_count DESC LIMIT ?",
        (limit,),
    )
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ REASONING PREFERENCES API ═════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def get_reasoning_preference(task_type: str) -> dict | None:
    row = await _fetchone(
        "SELECT * FROM reasoning_preferences WHERE task_type=?", (task_type,)
    )
    return _row_to_dict(row) if row else None


async def update_reasoning_preference(
    task_type: str, mode: str, confidence: float,
    sample_count: int = 1, scores_by_strategy: dict | None = None,
) -> None:
    existing = await _fetchone(
        "SELECT * FROM reasoning_preferences WHERE task_type=?", (task_type,)
    )
    now = time.time()
    if existing:
        sc = existing["sample_count"] + sample_count
        c = (existing["confidence"] * (sc - sample_count) + confidence * sample_count) / sc
        sbs = scores_by_strategy or {}
        old_sbs = {}
        if existing["scores_by_strategy"]:
            try:
                old_sbs = json.loads(existing["scores_by_strategy"])
            except (json.JSONDecodeError, TypeError):
                pass
        merged = {**old_sbs, **sbs}
        await _execute(
            "UPDATE reasoning_preferences SET preferred_strategy=?, confidence=?, sample_count=?, scores_by_strategy=?, updated_at=? WHERE task_type=?",
            (mode, c, sc, json.dumps(merged), now, task_type),
        )
    else:
        await _execute(
            "INSERT INTO reasoning_preferences (task_type, preferred_strategy, confidence, sample_count, scores_by_strategy, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (task_type, mode, confidence, sample_count, json.dumps(scores_by_strategy or {}), now),
        )


async def get_all_reasoning_preferences() -> list[dict]:
    rows = await _fetchall("SELECT * FROM reasoning_preferences ORDER BY sample_count DESC")
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ WEBHOOK EVENTS & CONFIGS ══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def store_webhook_event(service: str, event_type: str, payload: dict) -> int:
    c = await _execute(
        "INSERT INTO webhook_events (service, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
        (service, event_type, json.dumps(payload), time.time()),
    )
    return c.lastrowid or 0


async def get_pending_webhook_events(limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM webhook_events WHERE processed=0 ORDER BY created_at ASC LIMIT ?",
        (limit,),
    )
    return [_row_to_dict(r) for r in rows]


async def list_webhook_events(
    service_id: str = "", processed_only: bool = False,
    unprocessed_only: bool = False, limit: int = 50,
) -> list[dict]:
    if processed_only:
        rows = await _fetchall(
            "SELECT * FROM webhook_events WHERE processed=1 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
    elif unprocessed_only:
        rows = await _fetchall(
            "SELECT * FROM webhook_events WHERE processed=0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
    elif service_id:
        rows = await _fetchall(
            "SELECT * FROM webhook_events WHERE service=? ORDER BY created_at DESC LIMIT ?",
            (service_id, limit),
        )
    else:
        rows = await _fetchall(
            "SELECT * FROM webhook_events ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    return [_row_to_dict(r) for r in rows]


async def mark_webhook_event_processed(event_id: int, error: str | None = None) -> None:
    await _execute(
        "UPDATE webhook_events SET processed=1, error=?, processed_at=? WHERE id=?",
        (error, time.time(), event_id),
    )


async def register_webhook_config(
    service: str, url: str = "", secret: str = "",
    events: list[str] | None = None,
) -> int:
    now = time.time()
    c = await _execute(
        "INSERT INTO webhook_configs (service, url, secret, events, last_triggered, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (service, url, secret, json.dumps(events or []), now, now, now),
    )
    return c.lastrowid or 0


async def get_webhook_configs(service: str = "", enabled_only: bool = True) -> list[dict]:
    if service and enabled_only:
        rows = await _fetchall(
            "SELECT * FROM webhook_configs WHERE service=? AND enabled=1 ORDER BY created_at DESC",
            (service,),
        )
    elif service:
        rows = await _fetchall(
            "SELECT * FROM webhook_configs WHERE service=? ORDER BY created_at DESC",
            (service,),
        )
    elif enabled_only:
        rows = await _fetchall(
            "SELECT * FROM webhook_configs WHERE enabled=1 ORDER BY created_at DESC"
        )
    else:
        rows = await _fetchall("SELECT * FROM webhook_configs ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


async def update_webhook_config(config_id: int, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [config_id]
    await _execute(f"UPDATE webhook_configs SET {sets} WHERE id=?", tuple(vals))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ MEMORY INTELLIGENCE / GLOBAL USAGE ════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def get_memory_usage_stats() -> dict:
    total = await _fetchval("SELECT COUNT(*) FROM memories") or 0
    by_kind = await memory_counts_by_kind()
    by_permanent = await _fetchall(
        "SELECT permanent, COUNT(*) as cnt FROM memories GROUP BY permanent"
    )
    perm_map = {r["permanent"]: r["cnt"] for r in by_permanent}
    return {
        "total": total,
        "by_kind": by_kind,
        "permanent": perm_map.get(1, 0),
        "non_permanent": perm_map.get(0, 0),
    }


async def archive_memory(mem_id: int, reason: str = "manual") -> None:
    row = await _fetchone("SELECT * FROM memories WHERE id=?", (mem_id,))
    if not row:
        return
    d = dict(row)
    await _execute(
        "INSERT INTO memory_archived (original_id, kind, content, importance, embedding, session_id, tags, mood, archived_reason, archived_at, original_created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (d["id"], d["kind"], d["content"], d["importance"], d["embedding"],
         d["session_id"], d["tags"], d["mood"], reason, time.time(), d["created_at"]),
    )
    await _execute("DELETE FROM memories WHERE id=?", (mem_id,))


async def promote_memory(mem_id: int, importance_boost: float = 0.1) -> None:
    await _execute(
        "UPDATE memories SET importance=MIN(importance+?, 1.0), last_accessed=? WHERE id=?",
        (importance_boost, time.time(), mem_id),
    )


async def get_frequently_accessed_memories(
    min_access_count: int = 5, max_importance: float = 0.9, limit: int = 20,
) -> list[int]:
    rows = await _fetchall(
        "SELECT id FROM memories WHERE access_count>=? AND importance<=? ORDER BY access_count DESC, importance DESC LIMIT ?",
        (min_access_count, max_importance, limit),
    )
    return [r["id"] for r in rows]


async def get_archived_memories(limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM memory_archived ORDER BY archived_at DESC LIMIT ?", (limit,)
    )
    return [_row_to_dict(r) for r in rows]


async def get_global_usage() -> dict:
    row = await _fetchone("SELECT * FROM global_usage WHERE id=1")
    if row:
        return dict(row)
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}


async def increment_global_usage(
    prompt_tokens: int = 0, completion_tokens: int = 0,
    total_tokens: int = 0, cost_usd: float = 0.0,
) -> None:
    now = time.time()
    existing = await _fetchone("SELECT * FROM global_usage WHERE id=1")
    if existing:
        await _execute(
            "UPDATE global_usage SET prompt_tokens=prompt_tokens+?, completion_tokens=completion_tokens+?, total_tokens=total_tokens+?, cost_usd=cost_usd+?, updated_at=? WHERE id=1",
            (prompt_tokens, completion_tokens, total_tokens, cost_usd, now),
        )
    else:
        await _execute(
            "INSERT INTO global_usage (id, prompt_tokens, completion_tokens, total_tokens, cost_usd, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
            (prompt_tokens, completion_tokens, total_tokens, cost_usd, now),
        )


async def restore_memory(archived_id: int) -> int | None:
    row = await _fetchone("SELECT * FROM memory_archived WHERE id=?", (archived_id,))
    if not row:
        return None
    d = dict(row)
    c = await _execute(
        "INSERT INTO memories (kind, content, importance, embedding, session_id, tags, mood, created_at, last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (d["kind"], d["content"], d["importance"], d["embedding"],
         d["session_id"], d["tags"], d["mood"], d["original_created_at"] or time.time(),
         time.time(), 0),
    )
    await _execute("DELETE FROM memory_archived WHERE id=?", (archived_id,))
    return c.lastrowid
