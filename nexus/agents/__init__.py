"""
Agents module — single-agent architecture with lifecycle management.

All agent logic flows through the ReasoningEngine's single-agent
Nexus Framework Reasoning loop. This module provides a factory
interface for creating, tracking, and managing agent sessions.

Usage:
    from nexus.agents import create_agent, get_agent, cleanup_agent

    agent = await create_agent(session_id="my_session")
    response = await agent.respond("Hello!")
    await cleanup_agent(session_id="my_session")
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Agent registry ───────────────────────────────────────────────────────────
# Tracks all active ReasoningEngine instances session_id → (engine, created_at)
_active_agents: dict[str, tuple[Any, float]] = {}

# Sessions marked for cleanup (avoid double-cleanup race)
_cleanup_pending: set[str] = set()

# Max age for an agent session before it's eligible for automatic cleanup (seconds)
_MAX_AGENT_AGE = 3600  # 1 hour


async def create_agent(session_id: str) -> Any:
    """Create and register a new ReasoningEngine agent for the given session.

    Returns the ReasoningEngine instance. If an agent already exists for this
    session, returns the existing one (idempotent).

    Example:
        agent = await create_agent("session_123")
        response = await agent.respond("What's the weather?")
    """
    if session_id in _active_agents:
        agent, _ = _active_agents[session_id]
        logger.debug("Reusing existing agent for session %s", session_id)
        return agent

    from ..reasoning.engine import ReasoningEngine

    engine = ReasoningEngine(session_id)
    _active_agents[session_id] = (engine, time.time())
    logger.debug("Created new agent for session %s", session_id)
    return engine


def get_agent(session_id: str) -> Any | None:
    """Get the agent for a session, or None if not active."""
    entry = _active_agents.get(session_id)
    if entry:
        return entry[0]
    return None


async def cleanup_agent(session_id: str) -> None:
    """Clean up and deregister an agent session.

    This removes the agent from the registry and clears any pending
    cleanup flags. Call this when a session ends or times out.
    """
    if session_id in _cleanup_pending:
        return
    _cleanup_pending.add(session_id)

    try:
        _active_agents.pop(session_id, None)
        logger.debug("Cleaned up agent for session %s", session_id)
    except Exception as e:
        logger.warning("Error cleaning up agent session %s: %s", session_id, e)
    finally:
        _cleanup_pending.discard(session_id)


async def cleanup_stale_agent_sessions() -> int:
    """Remove agent sessions older than _MAX_AGENT_AGE.

    Returns the number of sessions cleaned up.
    """
    now = time.time()
    stale = [
        sid for sid, (_, created_at) in _active_agents.items()
        if (now - created_at) > _MAX_AGENT_AGE
    ]
    for sid in stale:
        await cleanup_agent(sid)
    if stale:
        logger.info("Cleaned up %d stale agent sessions", len(stale))
    return len(stale)


def active_agents_count() -> int:
    """Return the number of currently active agent sessions."""
    return len(_active_agents)


def active_agent_ids() -> list[str]:
    """Return the list of active session IDs for monitoring/debugging."""
    return list(_active_agents.keys())


def get_agent_stats() -> dict[str, Any]:
    """Get basic stats about active agents for observability."""
    now = time.time()
    stats = {
        "active_count": len(_active_agents),
        "agents": [],
    }
    for sid, (_, created_at) in _active_agents.items():
        stats["agents"].append({
            "session_id": sid,
            "age_seconds": round(now - created_at, 1),
        })
    return stats
