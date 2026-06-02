from . import database as db  # noqa: F401
from . import tree_db  # noqa: F401


async def cleanup_duplicates() -> int:
    """Remove duplicate memories. Returns count of deleted items."""
    return await db.cleanup_duplicate_memories()


async def prune_stale_memories() -> int:
    """Prune old, low-importance memories to prevent memory bloat."""
    return await db.prune_stale_memories()


# Lazy imports to avoid circular dependency between database.py and memory.py
def __getattr__(name: str):
    if name in ("Memory", "MemoryManager"):
        from .memory import Memory, MemoryManager
        globals()["Memory"] = Memory
        globals()["MemoryManager"] = MemoryManager
        return {"Memory": Memory, "MemoryManager": MemoryManager}[name]
    if name == "MemoryTree":
        from .memory_tree import MemoryTree
        globals()["MemoryTree"] = MemoryTree
        return MemoryTree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
