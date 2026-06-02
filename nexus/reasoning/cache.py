"""
Prompt and prefilter caches for the Nexus reasoning engine.
"""
from __future__ import annotations
import time as _time
import threading
import logging
logger = logging.getLogger(__name__)


import time as _time


_prompt_caches: dict[str, tuple[str, float]] = {}  # session_id -> (prompt, timestamp)


_prompt_cache_compressed_caches: dict[str, tuple[str, float]] = {}  # session_id -> (compressed_prompt, timestamp)


_PROMPT_CACHE_TTL = 60  # v38: Increased from 30 to 60  - system prompt rarely changes within a minute


_db_prompt_sub_cache: dict[str, tuple[str, float]] = {}


_self_awareness_cache: dict[str, tuple[str, float]] = {}


_DB_PROMPT_SUB_CACHE_TTL = 60  # v38: Increased from 30 to 60


_SELF_AWARENESS_CACHE_TTL = 30  # v38: Increased from 10 to 30  - DB state (watches, schedules) changes slowly


# v20: Tool pre-filtering cache  - keyed by first 200 chars of lowercased input, 30s TTL


_prefilter_cache: dict[str, tuple[list[dict], float]] = {}

_CACHE_LOCK = threading.Lock()

_PREFILTER_CACHE_TTL = 60  # v38: Increased from 30 to 60  - same input rarely changes tools within a minute


# FIX: Track tool count to invalidate cache when tools change


_prefilter_cache_tool_count = 0


def _invalidate_prompt_cache(session_id: str) -> None:


    """Invalidate all per-session prompt caches after settings change."""


    _prompt_caches.pop(session_id, None)


    _prompt_cache_compressed_caches.pop(session_id, None)


    _db_prompt_sub_cache.pop(session_id, None)


def _build_prompt_cache_key(session_id: str, messages: list[dict] | None = None) -> str:
    """Build a cache key that changes whenever the conversation changes.

    This prevents the agent from reusing a stale system prompt after the user
    sends a new message such as "I'll do it myself" or changes the context.
    """
    if not messages:
        return session_id
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = str(msg.get("content", ""))[:200]
            break
    return f"{session_id}:{last_user_msg}"


def _invalidate_all_prompt_caches() -> None:


    """Invalidate ALL prompt caches (used when settings change globally)."""


    _prompt_caches.clear()


    _prompt_cache_compressed_caches.clear()


    _db_prompt_sub_cache.clear()


    _self_awareness_cache.clear()