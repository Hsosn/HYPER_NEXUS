"""
Memory Tree database schema and helper functions (SQLite/aiosqlite).

Contains:
- CREATE TABLE statements for tree_chunks, tree_roots, tree_nodes, tree_jobs, tree_entities
- Indexes for fast tree lookups
- All CRUD/fetch functions for tree operations
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Schema statements (executed by database.py init()) ─────────────────────────

TREE_CREATE_TABLES: list[str] = [
    """CREATE TABLE IF NOT EXISTS tree_chunks (
        chunk_hash TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        token_count INTEGER DEFAULT 0,
        source TEXT,
        created_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tree_roots (
        root_id TEXT PRIMARY KEY,
        root_type TEXT NOT NULL DEFAULT 'source',
        title TEXT,
        summary TEXT,
        metadata TEXT DEFAULT '{}',
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tree_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        root_id TEXT NOT NULL REFERENCES tree_roots(root_id) ON DELETE CASCADE,
        parent_id INTEGER REFERENCES tree_nodes(id) ON DELETE CASCADE,
        tree_type TEXT NOT NULL DEFAULT 'source',
        level INTEGER DEFAULT 0,
        state TEXT NOT NULL DEFAULT 'buffered',
        content TEXT NOT NULL,
        summary TEXT,
        chunk_hash TEXT REFERENCES tree_chunks(chunk_hash),
        metadata TEXT DEFAULT '{}',
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tree_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id INTEGER REFERENCES tree_nodes(id) ON DELETE CASCADE,
        root_id TEXT REFERENCES tree_roots(root_id) ON DELETE CASCADE,
        job_type TEXT NOT NULL,
        state TEXT DEFAULT 'pending',
        error TEXT,
        created_at REAL,
        updated_at REAL
    )""",

    """CREATE TABLE IF NOT EXISTS tree_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        root_id TEXT,
        metadata TEXT DEFAULT '{}',
        created_at REAL,
        updated_at REAL
    )""",
]

TREE_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_tree_nodes_root ON tree_nodes(root_id)",
    "CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent ON tree_nodes(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_tree_nodes_state ON tree_nodes(state)",
    "CREATE INDEX IF NOT EXISTS idx_tree_nodes_type ON tree_nodes(tree_type)",
    "CREATE INDEX IF NOT EXISTS idx_tree_nodes_level ON tree_nodes(level)",
    "CREATE INDEX IF NOT EXISTS idx_tree_chunks_hash ON tree_chunks(chunk_hash)",
    "CREATE INDEX IF NOT EXISTS idx_tree_jobs_state ON tree_jobs(state)",
    "CREATE INDEX IF NOT EXISTS idx_tree_jobs_type ON tree_jobs(job_type)",
    "CREATE INDEX IF NOT EXISTS idx_tree_entities_name ON tree_entities(name)",
    "CREATE INDEX IF NOT EXISTS idx_tree_roots_type ON tree_roots(root_type)",
    "CREATE INDEX IF NOT EXISTS idx_tree_roots_created ON tree_roots(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tree_nodes_created ON tree_nodes(created_at)",
]

TREE_SCHEMA = TREE_CREATE_TABLES + TREE_INDEXES

# ── Tree DB helper functions ──────────────────────────────────────────────────

from nexus.memory.database import _connect, _row_to_dict, _sanitize_str


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ Tree Roots ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def tree_create_root(
    root_id: str, root_type: str = "source", title: str = "",
    summary: str = "", metadata: dict | None = None,
) -> None:
    now = time.time()
    async with _connect() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO tree_roots(root_id, root_type, title, summary, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (root_id, root_type, title, summary, _sanitize_str(json.dumps(metadata or {})), now, now),
        )


async def tree_get_root(root_id: str) -> dict | None:
    async with _connect() as conn:
        c = await conn.execute("SELECT * FROM tree_roots WHERE root_id=?", (root_id,))
        row = await c.fetchone()
        return _row_to_dict(row) if row else None


async def tree_list_roots(root_type: str | None = None, limit: int = 50) -> list[dict]:
    async with _connect() as conn:
        if root_type:
            c = await conn.execute(
                "SELECT * FROM tree_roots WHERE root_type=? ORDER BY created_at DESC LIMIT ?",
                (root_type, limit),
            )
        else:
            c = await conn.execute(
                "SELECT * FROM tree_roots ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await c.fetchall()
        return [_row_to_dict(r) for r in rows]


async def tree_update_root(root_id: str, **fields) -> None:
    allowed = {"title", "summary", "metadata"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    now = time.time()
    if "metadata" in updates and isinstance(updates["metadata"], dict):
        updates["metadata"] = json.dumps(updates["metadata"])
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [now, root_id]
    async with _connect() as conn:
        await conn.execute(f"UPDATE tree_roots SET {sets}, updated_at=? WHERE root_id=?", tuple(vals))


async def tree_delete_root(root_id: str) -> None:
    async with _connect() as conn:
        await conn.execute("DELETE FROM tree_roots WHERE root_id=?", (root_id,))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ Tree Nodes ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def tree_insert_node(
    root_id: str, content: str, parent_id: int | None = None,
    tree_type: str = "source", level: int = 0, state: str = "buffered",
    summary: str | None = None, chunk_hash: str | None = None,
    metadata: dict | None = None,
) -> int:
    now = time.time()
    async with _connect() as conn:
        c = await conn.execute(
            "INSERT INTO tree_nodes(root_id, parent_id, tree_type, level, state, content, summary, chunk_hash, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (root_id, parent_id, tree_type, level, state, _sanitize_str(content), summary, chunk_hash, _sanitize_str(json.dumps(metadata or {})), now, now),
        )
        return c.lastrowid or 0


async def tree_get_node(node_id: int) -> dict | None:
    async with _connect() as conn:
        c = await conn.execute("SELECT * FROM tree_nodes WHERE id=?", (node_id,))
        row = await c.fetchone()
        return _row_to_dict(row) if row else None


async def tree_get_children(parent_id: int, state: str | None = None) -> list[dict]:
    async with _connect() as conn:
        if state:
            c = await conn.execute(
                "SELECT * FROM tree_nodes WHERE parent_id=? AND state=? ORDER BY id ASC",
                (parent_id, state),
            )
        else:
            c = await conn.execute(
                "SELECT * FROM tree_nodes WHERE parent_id=? ORDER BY id ASC",
                (parent_id,),
            )
        rows = await c.fetchall()
        return [_row_to_dict(r) for r in rows]


async def tree_get_root_nodes(root_id: str, state: str | None = None) -> list[dict]:
    async with _connect() as conn:
        if state:
            c = await conn.execute(
                "SELECT * FROM tree_nodes WHERE root_id=? AND state=? ORDER BY id ASC",
                (root_id, state),
            )
        else:
            c = await conn.execute(
                "SELECT * FROM tree_nodes WHERE root_id=? ORDER BY id ASC",
                (root_id,),
            )
        rows = await c.fetchall()
        return [_row_to_dict(r) for r in rows]


async def tree_update_node(node_id: int, **fields) -> None:
    allowed = {"state", "summary", "content", "metadata", "chunk_hash", "parent_id"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    if "metadata" in updates and isinstance(updates["metadata"], dict):
        updates["metadata"] = json.dumps(updates["metadata"])
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [time.time(), node_id]
    async with _connect() as conn:
        await conn.execute(f"UPDATE tree_nodes SET {sets}, updated_at=? WHERE id=?", tuple(vals))


async def tree_delete_node(node_id: int) -> None:
    async with _connect() as conn:
        await conn.execute("DELETE FROM tree_nodes WHERE id=?", (node_id,))


async def tree_count_nodes(root_id: str | None = None, state: str | None = None) -> int:
    async with _connect() as conn:
        conditions = []
        params = []
        if root_id:
            conditions.append("root_id=?")
            params.append(root_id)
        if state:
            conditions.append("state=?")
            params.append(state)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        c = await conn.execute(f"SELECT COUNT(*) FROM tree_nodes{where}", tuple(params))
        row = await c.fetchone()
        return row[0] if row else 0


async def tree_get_daily_nodes(days_ago: int = 0, state: str | None = None) -> list[dict]:
    day_start = int(time.time()) - (days_ago * 86400)
    day_end = day_start + 86400
    async with _connect() as conn:
        if state:
            c = await conn.execute(
                "SELECT * FROM tree_nodes WHERE created_at >= ? AND created_at < ? AND state=? ORDER BY id ASC",
                (day_start, day_end, state),
            )
        else:
            c = await conn.execute(
                "SELECT * FROM tree_nodes WHERE created_at >= ? AND created_at < ? ORDER BY id ASC",
                (day_start, day_end),
            )
        rows = await c.fetchall()
        return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ Tree Chunks ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def compute_chunk_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def tree_upsert_chunk(content: str, token_count: int = 0, source: str | None = None) -> str:
    chunk_hash = compute_chunk_hash(content)
    now = time.time()
    async with _connect() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO tree_chunks(chunk_hash, content, token_count, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (chunk_hash, _sanitize_str(content), token_count, source, now),
        )
    return chunk_hash


async def tree_get_chunk(chunk_hash: str) -> dict | None:
    async with _connect() as conn:
        c = await conn.execute("SELECT * FROM tree_chunks WHERE chunk_hash=?", (chunk_hash,))
        row = await c.fetchone()
        return _row_to_dict(row) if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ Tree Jobs ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def tree_create_job(node_id: int, root_id: str | None, job_type: str) -> int:
    now = time.time()
    async with _connect() as conn:
        c = await conn.execute(
            "INSERT INTO tree_jobs(node_id, root_id, job_type, state, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (node_id, root_id, job_type, now, now),
        )
        return c.lastrowid or 0


async def tree_list_pending_jobs(job_type: str | None = None, limit: int = 50) -> list[dict]:
    async with _connect() as conn:
        if job_type:
            c = await conn.execute(
                "SELECT * FROM tree_jobs WHERE state='pending' AND job_type=? ORDER BY created_at ASC LIMIT ?",
                (job_type, limit),
            )
        else:
            c = await conn.execute(
                "SELECT * FROM tree_jobs WHERE state='pending' ORDER BY created_at ASC LIMIT ?",
                (limit,),
            )
        rows = await c.fetchall()
        return [_row_to_dict(r) for r in rows]


async def tree_update_job(job_id: int, state: str, error: str | None = None) -> None:
    now = time.time()
    async with _connect() as conn:
        if error:
            await conn.execute(
                "UPDATE tree_jobs SET state=?, error=?, updated_at=? WHERE id=?",
                (state, error, now, job_id),
            )
        else:
            await conn.execute(
                "UPDATE tree_jobs SET state=?, updated_at=? WHERE id=?",
                (state, now, job_id),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ Tree Entities ════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def tree_upsert_entity(name: str, root_id: str | None = None, metadata: dict | None = None) -> None:
    now = time.time()
    async with _connect() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO tree_entities(name, root_id, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (name, root_id, _sanitize_str(json.dumps(metadata or {})), now, now),
        )


async def tree_list_entities(limit: int = 100) -> list[dict]:
    async with _connect() as conn:
        c = await conn.execute(
            "SELECT * FROM tree_entities ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        rows = await c.fetchall()
        return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ Stats & Maintenance ═════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def tree_get_stats() -> dict:
    async with _connect() as conn:
        def fetchval(sql, params=()):
            return conn.execute_fetchval(sql, params) if hasattr(conn, 'execute_fetchval') else None

        c = await conn.execute("SELECT COUNT(*) FROM tree_roots")
        root_count = (await c.fetchone())[0] or 0

        c = await conn.execute("SELECT COUNT(*) FROM tree_nodes")
        node_count = (await c.fetchone())[0] or 0

        c = await conn.execute("SELECT COUNT(*) FROM tree_chunks")
        chunk_count = (await c.fetchone())[0] or 0

        c = await conn.execute("SELECT COUNT(*) FROM tree_entities")
        entity_count = (await c.fetchone())[0] or 0

        c = await conn.execute("SELECT COUNT(*) FROM tree_jobs WHERE state='pending'")
        pending_jobs = (await c.fetchone())[0] or 0

        c = await conn.execute("SELECT state, COUNT(*) as cnt FROM tree_nodes GROUP BY state")
        state_rows = await c.fetchall()
        state_counts = {r["state"]: r["cnt"] for r in state_rows}

        c = await conn.execute("SELECT tree_type, COUNT(*) as cnt FROM tree_nodes GROUP BY tree_type")
        type_rows = await c.fetchall()
        type_counts = {r["tree_type"]: r["cnt"] for r in type_rows}

        return {
            "roots": root_count,
            "nodes": node_count,
            "chunks": chunk_count,
            "entities": entity_count,
            "pending_jobs": pending_jobs,
            "nodes_by_state": state_counts,
            "nodes_by_type": type_counts,
        }


async def tree_seal_old_buckets(max_age_days: int = 1, batch_size: int = 100) -> int:
    cutoff = time.time() - (max_age_days * 86400)
    sealed = 0
    async with _connect() as conn:
        # Find buffered nodes older than cutoff
        c = await conn.execute(
            "SELECT id FROM tree_nodes WHERE state='buffered' AND created_at < ? LIMIT ?",
            (cutoff, batch_size),
        )
        rows = await c.fetchall()

        for row in rows:
            node_id = row["id"]
            await conn.execute(
                "UPDATE tree_nodes SET state='sealed', updated_at=? WHERE id=?",
                (time.time(), node_id),
            )
            # Get root_id
            c2 = await conn.execute("SELECT root_id FROM tree_nodes WHERE id=?", (node_id,))
            node = await c2.fetchone()
            if node:
                await conn.execute(
                    "INSERT INTO tree_jobs(node_id, root_id, job_type, state, created_at, updated_at) VALUES (?, ?, 'summarize', 'pending', ?, ?)",
                    (node_id, node["root_id"], time.time(), time.time()),
                )
            sealed += 1
    return sealed


async def tree_cleanup_orphans() -> int:
    removed = 0
    async with _connect() as conn:
        # Orphaned chunks
        c = await conn.execute(
            "DELETE FROM tree_chunks WHERE chunk_hash NOT IN (SELECT DISTINCT chunk_hash FROM tree_nodes WHERE chunk_hash IS NOT NULL)"
        )
        removed += c.rowcount or 0

        # Old failed jobs (>7 days)
        c = await conn.execute(
            "DELETE FROM tree_jobs WHERE state='failed' AND updated_at <?",
            (time.time() - (7 * 86400),),
        )
        removed += c.rowcount or 0

        # Completed jobs older than 30 days
        c = await conn.execute(
            "DELETE FROM tree_jobs WHERE state='completed' AND updated_at <?",
            (time.time() - (30 * 86400),),
        )
        removed += c.rowcount or 0

    return removed


async def tree_compute_summary(node_id: int) -> str | None:
    node = await tree_get_node(node_id)
    if not node:
        return None
    content = node["content"]
    if len(content) < 50:
        summary = content[:200]
        await tree_update_node(node_id, summary=summary)
        return summary
    await tree_create_job(node_id, node["root_id"], "summarize")
    return None


async def tree_prune_by_forgetting_curve(
    base_half_life_hours: float = 720.0,
    retention_threshold: float = 0.08,
    max_prune: int = 200,
) -> dict:
    pruned_nodes = 0
    now = time.time()

    async with _connect() as conn:
        # Phase 1: old buffered nodes
        c = await conn.execute(
            "SELECT id, created_at, metadata FROM tree_nodes WHERE state='buffered' AND created_at < ? LIMIT ?",
            (now - (base_half_life_hours * 3600), max_prune // 2),
        )
        old_buffered = await c.fetchall()

        for row in old_buffered:
            age_hours = max(0.0, (now - row["created_at"]) / 3600.0)
            meta = row.get("metadata") or "{}"
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            imp = float(meta.get("importance", 0.3))
            retention = imp / (1.0 + age_hours / base_half_life_hours)

            if retention < retention_threshold:
                await conn.execute("DELETE FROM tree_nodes WHERE id=?", (row["id"],))
                pruned_nodes += 1

        # Phase 2: old sealed nodes
        c = await conn.execute(
            "SELECT id, updated_at, created_at, metadata FROM tree_nodes WHERE state='sealed' AND updated_at < ? LIMIT ?",
            (now - (base_half_life_hours * 2 * 3600), max_prune // 2),
        )
        old_sealed = await c.fetchall()

        for row in old_sealed:
            ts = row.get("updated_at") or row.get("created_at") or now
            age_hours = max(0.0, (now - ts) / 3600.0)
            meta = row.get("metadata") or "{}"
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            imp = float(meta.get("importance", 0.5))
            retention = imp / (1.0 + age_hours / (base_half_life_hours * 2))

            if retention < retention_threshold * 0.5:
                await conn.execute("DELETE FROM tree_nodes WHERE id=?", (row["id"],))
                pruned_nodes += 1

    orphans_removed = await tree_cleanup_orphans()

    return {
        "pruned_nodes": pruned_nodes,
        "orphans_removed": orphans_removed,
    }
