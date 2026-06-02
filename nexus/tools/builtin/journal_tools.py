"""Journal tools — structured journaling with mood, tags, and permanence (#21, #22, #25)."""
from __future__ import annotations

import json
from typing import Any

from ... import config
from ...memory import db
from ..registry import tool


@tool(
    name="journal_write",
    description="Write a journal entry with optional mood, tags, and people. "
                "Journal entries are permanent by default and won't be pruned.",
    parameters_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The journal entry content"},
            "mood": {"type": "string", "description": "Current mood (happy, sad, anxious, excited, calm, frustrated, grateful, tired, etc.)"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization (work, personal, health, relationship, etc.)"},
            "people": {"type": "array", "items": {"type": "string"}, "description": "People mentioned or involved"},
        },
        "required": ["content"],
    },
    category="journal",
)
async def journal_write(params: dict[str, Any]) -> str:
    """Write a journal entry."""
    content = params.get("content", "").strip()
    if not content:
        return "Error: Content is required"
    if len(content) > 5000:
        content = content[:5000]

    entry_id = await db.add_journal_entry(
        content=content,
        mood=params.get("mood", ""),
        tags=params.get("tags", []),
        people=params.get("people", []),
        importance=0.7,
        permanent=True,
    )
    mood_str = f" (mood: {params['mood']})" if params.get("mood") else ""
    return f"Journal entry #{entry_id} saved{mood_str}"


@tool(
    name="journal_read",
    description="Read recent journal entries. Optionally filter by mood.",
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Number of entries to return (default 20, max 100)"},
            "mood": {"type": "string", "description": "Filter by mood (optional)"},
        },
    },
    category="journal",
)
async def journal_read(params: dict[str, Any]) -> str:
    """Read journal entries."""
    limit = min(int(params.get("limit", 20)), 100)
    mood = params.get("mood")
    entries = await db.list_journal_entries(limit=limit, mood=mood)
    if not entries:
        return "No journal entries found."
    result = []
    for e in entries:
        import time as _time
        ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(e.get("created_at", 0)))
        mood_str = f" [{e['mood']}]" if e.get("mood") else ""
        tags_str = f" #{','.join(json.loads(e.get('tags', '[]'))[:3])}" if e.get("tags") else ""
        result.append(f"#{e['id']} | {ts}{mood_str}{tags_str}\n{e['content'][:500]}")
    return "\n---\n".join(result)


@tool(
    name="journal_mood_summary",
    description="Get a summary of mood trends over the last N days. Useful for "
                "reflecting on emotional patterns.",
    parameters_schema={
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "Number of days to analyze (default 7)"},
        },
    },
    category="journal",
)
async def journal_mood_summary(params: dict[str, Any]) -> str:
    """Get mood summary."""
    days = min(int(params.get("days", 7)), 365)
    summary = await db.get_mood_summary(days=days)
    if not summary:
        return f"No mood data found in the last {days} days."
    result = [f"Mood trends (last {days} days):"]
    for row in summary:
        result.append(f"  {row['mood']}: {row['count']} entries (avg importance: {row['avg_importance']:.2f})")
    return "\n".join(result)


@tool(
    name="memory_set_permanent",
    description="Mark a memory as permanent so it won't be pruned. "
                "Use this for important life events, critical facts, or journal-like entries.",
    parameters_schema={
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer", "description": "The memory ID to protect"},
            "permanent": {"type": "boolean", "description": "True to protect, False to unprotect (default True)"},
        },
        "required": ["memory_id"],
    },
    category="memory",
)
async def memory_set_permanent(params: dict[str, Any]) -> str:
    """Set memory permanence flag (#22)."""
    mem_id = params.get("memory_id")
    permanent = params.get("permanent", True)
    if not mem_id:
        return "Error: memory_id is required"
    async with db._connect() as conn:
        await conn.execute(
            "UPDATE memories SET permanent=? WHERE id=?", (int(permanent), int(mem_id))
        )
    return f"Memory #{mem_id} {'protected' if permanent else 'unprotected'} from pruning"
