"""Tools that let the agent read/write its own memory explicitly."""
from __future__ import annotations

from ..registry import tool
from ...memory import db


@tool(
    name="remember",
    description="Explicitly save an important fact/preference/insight to long-term memory.",
    parameters_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": ["factual", "semantic", "procedural", "episodic", "profile", "preference", "golden_trace"],
                "default": "factual",
            },
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["content"],
    },
    category="memory",
)
async def remember(params):
    """Save a memory with auto-embedding. Uses MemoryManager for dedup + embedding."""
    from ...memory.memory import MemoryManager

    content = params["content"][:2000]
    if not content.strip():
        return "Error: content is empty."

    mm = MemoryManager(session_id="")  # tool-level calls have no session context
    mem_id = await mm.remember(
        content=content,
        kind=params.get("kind", "factual"),
        importance=float(params.get("importance", 0.7)),
        tags=params.get("tags") or [],
    )
    return f"Saved memory #{mem_id} (with embedding)"


@tool(
    name="recall_memories",
    description="Search long-term memories by relevance. Optionally filter by kind.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories via semantic similarity.",
            },
            "kind": {
                "type": "string",
                "enum": ["factual", "semantic", "procedural", "episodic", "profile", "preference", "golden_trace"],
            },
            "limit": {"type": "integer", "default": 10},
        },
        "required": [],
    },
    category="memory",
)
async def recall_memories(params):
    """Recall memories using full semantic scoring (cosine + Ebbinghaus + importance + frequency)."""
    from ...memory.memory import MemoryManager

    query = params.get("query", "")
    kind = params.get("kind")
    limit = int(params.get("limit", 10))

    # Use MemoryManager.recall() for full semantic scoring when a query is provided
    if query.strip():
        mm = MemoryManager(session_id="")
        mems = await mm.recall(query, k=limit, kind=kind)
    else:
        # No query → fall back to raw DB listing (most important first)
        rows = await db.all_memories(kind=kind, limit=limit)
        mems = [__row_to_mem(r) for r in rows]

    if not mems:
        return "(no memories found)"

    lines = []
    for m in mems:
        content = m.content if hasattr(m, "content") else m.get("content", "")
        kind_val = m.kind if hasattr(m, "kind") else m.get("kind", "?")
        imp = m.importance if hasattr(m, "importance") else m.get("importance", 0.5)
        lines.append(f"[#{m.id if hasattr(m, 'id') else m.get('id', '?')} {kind_val} imp={imp:.2f}] {content[:300]}")
    return "\n".join(lines)


def __row_to_mem(row: dict):
    """Shallow Memory-like object from a DB row (for no-query fallback)."""
    from ...memory.memory import Memory
    return Memory.from_row(row)


@tool(
    name="forget_memories",
    description="Delete or archive memories by ID, or prune old/low-importance memories. "
                "Use this to intentionally forget outdated information, clean up clutter, "
                "or enforce the forgetting mechanism.",
    parameters_schema={
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "ID of a single memory to delete permanently.",
            },
            "archive": {
                "type": "boolean",
                "description": "If true, archive the memory instead of hard-deleting (recoverable). Default false (hard delete).",
                "default": False,
            },
            "prune_older_than_days": {
                "type": "integer",
                "description": "Prune (archive) memories older than this many days with importance below min_importance.",
                "minimum": 1,
            },
            "min_importance": {
                "type": "number",
                "description": "Importance threshold for pruning (0.0–1.0). Only memories below this are pruned. Default 0.3.",
                "minimum": 0,
                "maximum": 1,
                "default": 0.3,
            },
        },
        "required": [],
    },
    category="memory",
)
async def forget_memories(params: dict) -> str:
    """Intentionally forget/delete/prune memories."""
    mem_id = params.get("memory_id")
    archive_flag = params.get("archive", False)
    prune_days = params.get("prune_older_than_days")
    min_imp = float(params.get("min_importance", 0.3))

    if mem_id:
        if archive_flag:
            await db.archive_memory(int(mem_id), reason="forgotten_by_agent")
            return f"Memory #{mem_id} archived (recoverable)."
        else:
            await db.delete_memory(int(mem_id))
            return f"Memory #{mem_id} permanently deleted."

    if prune_days:
        pruned = await db.prune_stale_memories(
            max_age_days=float(prune_days),
            min_importance=min_imp,
            min_access_count=0,
            max_memories=10000,
        )
        if pruned > 0:
            return f"Pruned {pruned} old/low-importance memories."
        return "No memories matched the pruning criteria."

    return "Error: provide memory_id or prune_older_than_days."



# ═══════════════════════════════════════════════════════════════════════════════
# ═══ Memory Tree Tools ══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════


@tool(
    name="tree_ingest",
    description="Ingest content into the Memory Tree for hierarchical storage, deduplication, and summarization.",
    parameters_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Text content to store"},
            "source": {"type": "string", "description": "Source description"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        },
        "required": ["content"],
    },
    category="memory",
)
async def tree_ingest(params):
    """Ingest content into the memory tree for hierarchical recall."""
    from ...memory.memory_tree import MemoryTree
    
    content = params["content"][:5000]
    source = params.get("source", "")
    tags = params.get("tags") or []
    importance = float(params.get("importance", 0.5))
    
    tree = MemoryTree()
    result = await tree.ingest(
        content=content,
        source=source,
        tags=tags,
        importance=importance,
    )
    return f"Ingested into Memory Tree: {result['node_count']} nodes in root {result['root_id'][:20]}..."


@tool(
    name="tree_browse",
    description="Browse the memory tree: search for nodes, view daily digests, get topic trees, or list roots.",
    parameters_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "daily_digest", "topic", "roots", "root"],
                "description": "What to browse: search=keyword search, daily_digest=view a day's memories, topic=topic tree, roots=list all roots, root=view specific root",
            },
            "query": {"type": "string", "description": "Search query or topic name"},
            "root_id": {"type": "string", "description": "Root ID to browse"},
            "days_ago": {"type": "integer", "description": "Days ago for daily digest (0=today)", "default": 0},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["action"],
    },
    category="memory",
)
async def tree_browse(params):
    """Browse the memory tree with various views."""
    from ...memory.memory_tree import MemoryTree
    
    action = params["action"]
    tree = MemoryTree()
    
    if action == "search":
        query = params.get("query", "")
        limit = int(params.get("limit", 20))
        nodes = await tree.recall(query=query, limit=limit)
        if not nodes:
            return "(no matching memories found)"
        lines = []
        for n in nodes:
            summary = n.get("summary", "") or n.get("content", "")[:100]
            state = n.get("state", "?")
            rid = n.get("root_id", "?")[:16]
            lines.append(f"[#{n['id']} {state} root={rid}] {summary[:200]}")
        return '\n'.join(lines)

    elif action == "daily_digest":
        days_ago = int(params.get("days_ago", 0))
        digest = await tree.get_daily_digest(days_ago=days_ago)
        sep = '\n'
        return f"=== Daily Digest: {digest['date']} ==={sep}Roots: {digest['root_count']}, Nodes: {digest['node_count']}{sep}{digest['summary']}"

    elif action == "topic":
        topic = params.get("query", "")
        if not topic:
            return "Error: provide a topic name in 'query'"
        nodes = await tree.get_topic_tree(topic, limit=int(params.get("limit", 20)))
        if not nodes:
            return f"(no memories found for topic: {topic})"
        lines = [f"=== Topic: {topic} ==="]
        for n in nodes:
            lines.append(f"  - {n.get('content', '')[:150]}")
        return '\n'.join(lines)

    elif action == "roots":
        from ...memory.tree_db import tree_list_roots
        roots = await tree_list_roots(limit=int(params.get("limit", 20)))
        if not roots:
            return "(no roots in memory tree)"
        lines = [f"=== Memory Tree Roots ==="]
        for r in roots:
            title = r.get("title", "Untitled")
            rtype = r.get("root_type", "?")
            rid = r.get("root_id", "?")[:20]
            lines.append(f"  {rid} [{rtype}] {title}")
        return '\n'.join(lines)

    elif action == "root":
        root_id = params.get("root_id", "")
        if not root_id:
            return "Error: provide root_id"
        nodes = await tree.recall_by_root(root_id)
        if not nodes:
            return f"(root not found: {root_id})"
        lines = [f"=== Root: {root_id} ==="]
        for n in nodes[:30]:
            summary = n.get("summary", "") or n.get("content", "")[:100]
            lines.append(f"  [#{n['id']} lv{n['level']} {n['state']}] {summary}")
        if len(nodes) > 30:
            lines.append(f"  ... and {len(nodes)-30} more nodes")
        return '\n'.join(lines)

    return f"Unknown action: {action}"


@tool(
    name="tree_status",
    description="Get memory tree statistics: roots, nodes, chunks, entities, pending jobs.",
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="memory",
)
async def tree_status(params):
    """Show memory tree statistics."""
    from ...memory.memory_tree import MemoryTree
    
    tree = MemoryTree()
    stats = await tree.get_stats()
    
    sep = '\n'
    return f"=== Memory Tree Status ==={sep}Roots: {stats['roots']}{sep}Nodes: {stats['nodes']}{sep}Chunks: {stats['chunks']}{sep}Entities: {stats['entities']}{sep}Pending jobs: {stats['pending_jobs']}{sep}Nodes by state: {stats.get('nodes_by_state', {})}{sep}Nodes by type: {stats.get('nodes_by_type', {})}"
