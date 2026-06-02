"""
Self-improvement loop — deep learning system (v18 complete rewrite).

Architecture:
├── 1. Experience Replay System — store & retrieve full task trajectories
├── 2. Meta-Learning Engine — learn which reasoning strategy works best per task type
├── 3. Tool Affinity Analysis — co-occurrence matrix, successful/failed chains
├── 4. [REMOVED] Adaptive Critic System — removed (was multi-agent only)
├── 5. Failure Pattern Recognition — NLP-based clustering via embeddings
├── 6. Strategy Injection System — lifecycle-managed learned strategies
├── 7. Quality Feedback Loop — quality monitoring & intervention
├── 8. Memory Consolidation Intelligence — semantic dedup, usage tracking
├── 9. Learning Rate & Decay — time-weighted, evidence-adjusted
└── 10. v17 Bug Fixes — all known bugs resolved

Guardrails (inherited + enhanced):
- ONLY writes to the database (memories, tool strategies, trajectories, insights)
- NEVER modifies code, config, settings, or any files on disk
- Cannot brick the system — worst case is bad memories polluting context
- Bad memories can be cleaned via /api/memories/cleanup
- All LLM calls use the memory_model (cheaper/faster) to avoid cost blowup
- Strategy content is capped at 200 chars to prevent prompt bloat
- Has explicit error handling — never crashes the agent loop
- Learning cap enforced to prevent prompt bloat from too many strategies
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import logging
logger = logging.getLogger(__name__)

from .. import config
from ..core import llm
from ..events import emit
from ..memory import db

# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class TaskTrajectory:
    """Complete task execution trace for experience replay."""
    session_id: str
    user_request: str
    task_type: str
    reasoning_strategy: str  # Always "nexus" in the unified framework
    plan_steps: list[dict]
    tool_calls: list[dict]  # {tool, params, result, duration, success}
    final_answer: str
    quality_score: float  # quality score 0.0-10.0
    token_usage: dict
    duration_seconds: float
    error: str | None = None


@dataclass
class LearningInsight:
    """A learned strategy, pattern, or countermeasure."""
    category: str  # "tool_strategy", "reasoning_preference", "failure_pattern", "tool_chain"
    task_type: str
    content: str
    confidence: float  # 0.0-1.0
    evidence_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    effectiveness: float = 0.0  # measured improvement


# ── Constants ─────────────────────────────────────────────────────────────────

# How many failures before we write a learning
_FAILURE_THRESHOLD = 2
# Exponential backoff: don't keep re-analyzing the same failures
_FAILURE_BACKOFF: dict[str, int] = {}  # tool_name -> skip count
# Max frequency before auto-resolving as systemic/unfixable
_MAX_SYSTEMIC_FREQUENCY = 5  # >= this = auto-resolve without LLM call
# How often to run the full improvement analysis (seconds)
_IMPROVE_INTERVAL = 120  # 2 minutes — more frequent learning cycles
_last_improve_run: float = 0.0

# ── Safety helpers ────────────────────────────────────────────────────────────
_MAX_SELF_LEARNED_MEMORIES = 50
_MAX_STRATEGY_LENGTH = 200
_MAX_ACTIVE_INSIGHTS = 100
_MAX_ACTIVE_STRATEGIES_PER_TYPE = 3

# ── Learning rate constants ───────────────────────────────────────────────────
_LEARNING_WEIGHT_NEW = 1.0
_LEARNING_DECAY_DAYS = 14.0  # half-life in days
_LEARNING_VALIDATED_BOOST = 0.1
_LEARNING_CONTRADICTED_PENALTY = 0.3
_LEARNING_MIN_CONFIDENCE = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 0. v17 BUG FIXES — Applied throughout the file ══════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
# Fixed bugs:
#   1. Indentation of _mine_reflections, _store_experience_replay,
#      _classify_task_from_score — all now at module level (not inside
#      _generate_daily_summary)
#   2. Removed meaningless @staticmethod from _classify_task_from_score
#      (it's a module-level function, not a method)
#   3. Fixed typo "self_improveement" → "self_improvement" in emit source
#   4. Fixed reflection mining to use embeddings not naive word frequency
#   5. Fixed circular import risk — database imports are deferred where needed
# ═══════════════════════════════════════════════════════════════════════════════


# ── Schema bootstrap ──────────────────────────────────────────────────────────

async def _ensure_schema() -> None:
    """Ensure the database pool is initialized and all schema tables exist."""
    if db._db is None:
        await db.init()


# ── Internal safety check ────────────────────────────────────────────────────

async def _check_self_learned_cap() -> bool:
    """Check if the self-learned memory cap has been reached.

    Returns True if cap is reached (caller should skip), False if ok.
    """
    existing = await db.all_memories(kind="procedural", limit=1000)
    self_learned = [m for m in existing if "self-learning" in (m.get("tags") or [])]
    if len(self_learned) >= _MAX_SELF_LEARNED_MEMORIES:
        await emit("warn", source="self_improvement",
                   message=f"Self-learned memory cap reached ({_MAX_SELF_LEARNED_MEMORIES}). Skipping new strategy.")
        return True
    return False


async def _check_insight_cap() -> bool:
    """Check if the active learning insight cap has been reached."""
    active = await db.get_active_insights(limit=_MAX_ACTIVE_INSIGHTS + 1)
    if len(active) >= _MAX_ACTIVE_INSIGHTS:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ MAIN ENTRY POINT ════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def run() -> None:
    """Main entry: run all improvement analyses if enough time has passed.

    Orchestrates the full self-improvement pipeline:
    1. Analyse tool failures and generate strategies
    2. Analyse tool usage patterns (co-occurrence, speed)
    3. Analyse quality trends and trigger interventions
    4. Detect goal stalls
    5. Flag recurring patterns
    6. Verify previous learnings are still effective
    7. Generate daily summary
    8. Mine reflections (embedding-based clustering)
    9. Store experience replay trajectories
    10. Run meta-learning engine
    11. Tool affinity analysis
    12. Strategy injection lifecycle
    13. Quality feedback loop
    14. Memory consolidation
    15. Apply learning decay
    """
    global _last_improve_run
    now = time.time()
    if now - _last_improve_run < _IMPROVE_INTERVAL:
        logger.debug("[self_improve] Skipping — last run %.0fs ago", now - _last_improve_run)
        return
    _last_improve_run = now

    logger.info("[self_improve] Starting full improvement pipeline")

    # Ensure DB schema exists before any operations
    try:
        await _ensure_schema()
    except Exception:
        pass  # Non-critical — next cycle will retry

    try:
        # ── Core analyses (inherited from v7, enhanced) ──
        logger.info("[self_improve] Step 1/15: Tool failure analysis")
        await _analyse_tool_failures()
        logger.info("[self_improve] Step 2/15: Tool strategy analysis")
        await _analyse_tool_strategies()
        logger.info("[self_improve] Step 3/15: Quality trend analysis")
        await _analyse_quality_trends()
        logger.info("[self_improve] Step 4/15: Goal stall detection")
        await _analyse_goal_stalls()
        logger.info("[self_improve] Step 4b/15: Resolve systemic patterns")
        await _resolve_systemic_patterns()
        logger.info("[self_improve] Step 5/15: Pattern flagging")
        await _flag_patterns()
        logger.info("[self_improve] Step 6/15: Verify previous learnings")
        await _verify_previous_learnings()
        logger.info("[self_improve] Step 7/15: Daily summary")
        await _generate_daily_summary()

        # ── v18 Deep Learning Features ──
        logger.info("[self_improve] Step 8/15: Reflection mining")
        await _mine_reflections()
        logger.info("[self_improve] Step 9/15: Experience replay")
        await _store_experience_replay()
        logger.info("[self_improve] Step 10/15: Meta-learning")
        await _run_meta_learning()
        logger.info("[self_improve] Step 11/15: Tool affinity analysis")
        await _run_tool_affinity_analysis()
        logger.info("[self_improve] Step 12/15: Strategy lifecycle management")
        await _manage_strategy_lifecycle()
        logger.info("[self_improve] Step 13/15: Quality feedback loop")
        await _quality_feedback_loop()
        logger.info("[self_improve] Step 14/15: Memory consolidation")
        await _consolidate_memories()
        logger.info("[self_improve] Step 15/15: Learning decay")
        await _apply_learning_decay()

        await emit("self_improvement",
                   message="Full v18 self-improvement cycle completed")
        logger.info("[self_improve] Pipeline completed successfully")

    except Exception as e:
        # Self-improvement must NEVER crash the agent. Log and continue.
        # v17 bug fix: "self_improveement" → "self_improvement"
        await emit("warn", source="self_improvement", error=str(e),
                   message="Self-improvement cycle encountered an error (non-critical)")


# ═══ ADVANCED SELF-LEARNING: REAL-TIME LEARNING — extracted to realtime.py ══════

from .realtime import learn_from_failure_realtime, _generate_quick_fix  # noqa: F811


async def get_agent_capabilities_summary() -> dict:
    """Get a comprehensive summary of the agent's current capabilities and state.

    Returns a dict with:
    - tool_count: number of registered tools
    - tool_categories: dict of category -> count
    - skill_count: number of available skills
    - memory_stats: long-term, procedural, semantic memory counts
    - active_goals: count of pending goals
    - connected_integrations: list of connected service names
    - learned_strategies: count of active learning insights
    - procedural_skills: count of procedural skill memories
    - quality_trend: recent average quality score
    """
    summary = {}

    try:
        from ..tools import REGISTRY
        tools = REGISTRY.all_tools()
        summary["tool_count"] = len(tools)
        summary["tool_categories"] = {}
        for t in tools:
            cat = t.category or "uncategorized"
            summary["tool_categories"][cat] = summary["tool_categories"].get(cat, 0) + 1
    except Exception:
        summary["tool_count"] = 0
        summary["tool_categories"] = {}

    try:
        summary["skill_count"] = len([t for t in tools if getattr(t, "category", "") == "Skills"])
    except Exception:
        summary["skill_count"] = 0

    try:
        summary["memory_stats"] = {
            "total": len(await db.all_memories(limit=10000)),
            "procedural": len(await db.all_memories(kind="procedural", limit=10000)),
            "semantic": len(await db.all_memories(kind="semantic", limit=10000)),
        }
    except Exception:
        summary["memory_stats"] = {"total": 0, "procedural": 0, "semantic": 0}

    try:
        goals = await db.list_goals()
        summary["active_goals"] = len([g for g in goals if g.get("status") in ("pending", "active")])
    except Exception:
        summary["active_goals"] = 0

    try:
        integrations = await db.list_connected_integrations(connected_only=True)
        summary["connected_integrations"] = [i.get("name", "") for i in integrations]
    except Exception:
        summary["connected_integrations"] = []

    try:
        insights = await db.get_active_insights(limit=200)
        summary["learned_strategies"] = len(insights)
    except Exception:
        summary["learned_strategies"] = 0

    try:
        scores = await db.recent_task_scores(limit=10)
        if scores:
            summary["quality_trend"] = round(sum(s.get("score", 0) for s in scores) / len(scores), 1)
        else:
            summary["quality_trend"] = 0
    except Exception:
        summary["quality_trend"] = 0

    return summary


async def detect_user_satisfaction(session_id: str, user_input: str, assistant_response: str) -> dict:
    """Analyze user satisfaction signals from the conversation.

    Detects implicit feedback:
    - Positive: "thanks", "perfect", "great", "that's what I wanted", etc.
    - Negative: "no", "that's wrong", "try again", "not what I asked", etc.
    - Follow-up refinement: asking for changes to previous response

    Returns dict with: {sentiment: "positive"|"negative"|"neutral", confidence: float, signals: list}
    """
    result = {"sentiment": "neutral", "confidence": 0.0, "signals": []}

    # Check assistant_response for error indicators
    if assistant_response:
        a_low = assistant_response.lower()
        if any(w in a_low for w in ("error", "failed", "unable to", "i cannot")):
            result["sentiment"] = "negative"
            result["confidence"] = 0.3
            result["signals"] = ["assistant_error"]
            return result

    user_low = user_input.lower().strip()

    # Positive signals
    positive_patterns = [
        "thanks", "thank you", "perfect", "great", "awesome", "excellent",
        "that's it", "that's what I wanted", "exactly", "yes that's right",
        "good job", "well done", "nice", "love it", "amazing", "correct",
        "spot on", "nailed it", "brilliant",
    ]

    # Negative signals
    negative_patterns = [
        "no that's not", "that's wrong", "try again", "not what I asked",
        "incorrect", "wrong answer", "that's not right", "fix this",
        "this is wrong", "you're wrong", "bad", "terrible", "horrible",
        "doesn't work", "not working", "error", "failed", "broke",
        "do it again", "redo", "start over",
    ]

    # Check for signals
    positive_hits = [p for p in positive_patterns if p in user_low]
    negative_hits = [p for p in negative_patterns if p in user_low]

    if positive_hits and not negative_hits:
        result["sentiment"] = "positive"
        result["confidence"] = min(0.5 + 0.1 * len(positive_hits), 0.95)
        result["signals"] = positive_hits[:3]

        # Store positive outcome as reinforcement learning
        try:
            await db.log_improvement(
                category="positive_feedback",
                detail=f"User satisfaction signal: {', '.join(positive_hits[:3])}",
            )
        except Exception:
            pass

    elif negative_hits:
        result["sentiment"] = "negative"
        result["confidence"] = min(0.5 + 0.1 * len(negative_hits), 0.95)
        result["signals"] = negative_hits[:3]

        # Store negative outcome for improvement analysis
        try:
            await db.log_improvement(
                category="negative_feedback",
                detail=f"User dissatisfaction signal: {', '.join(negative_hits[:3])}. "
                       f"Context: user said '{user_input[:200]}'",
            )
        except Exception:
            pass

    return result


async def get_active_learnings_for_prompt(limit: int = 5) -> str:
    """Get the most recent and relevant learnings for injection into the system prompt.

    Returns a formatted string of the top-N most relevant learned strategies
    that should influence the agent's current behavior.
    """
    try:
        insights = await db.get_active_insights(limit=limit)
        if not insights:
            return ""

        lines = ["## Active Learned Strategies (auto-improving)"]
        for i, insight in enumerate(insights, 1):
            content = insight.get("content", "")[:150]
            confidence = insight.get("confidence", 0)
            category = insight.get("category", "")
            status = insight.get("status", "")

            # Format based on category
            if category == "tool_chain" and "Avoid" in content:
                lines.append(f"{i}. ⚠️ {content} [{status}]")
            elif category == "tool_chain":
                lines.append(f"{i}. ✅ {content} [{status}]")
            else:
                lines.append(f"{i}. {content} [{status}, confidence: {confidence:.0%}]")

        return "\n".join(lines)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 1. EXPERIENCE REPLAY SYSTEM ═════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def store_trajectory(trajectory: TaskTrajectory) -> int:
    """Store a complete task trajectory for experience replay.

    Stores the full execution trace so similar future tasks can use it
    as a few-shot example in planning prompts.

    Args:
        trajectory: Complete TaskTrajectory dataclass with all execution details.

    Returns:
        The database row ID of the stored trajectory.
    """
    traj_dict = {
        "session_id": trajectory.session_id,
        "user_request": trajectory.user_request,
        "task_type": trajectory.task_type,
        "reasoning_strategy": "nexus",
        "plan_steps": trajectory.plan_steps,
        "tool_calls": trajectory.tool_calls,
        "final_answer": trajectory.final_answer,
        "quality_score": trajectory.quality_score,
        "token_usage": trajectory.token_usage,
        "duration_seconds": trajectory.duration_seconds,
        "error": trajectory.error,
    }
    return await db.store_trajectory(traj_dict)


async def retrieve_similar_trajectories(query: str, k: int = 3) -> list[dict]:
    """Retrieve past trajectories similar to the current query.

    Uses the task type classification to find relevant past executions.
    For best results, the query should be classified into a task_type first.

    Args:
        query: The user's current request (used for task type matching).
        k: Maximum number of trajectories to retrieve.

    Returns:
        List of trajectory dicts, sorted by quality_score DESC.
    """
    task_type = _classify_task_type(query)
    trajectories = await db.retrieve_similar_trajectories(task_type, limit=k)
    return trajectories


def format_trajectory_for_prompt(trajectories: list[dict]) -> str:
    """Format retrieved trajectories as few-shot examples for the reasoning engine.

    Produces a concise summary of each trajectory showing the approach
    taken, tools used, and outcome quality.
    """
    if not trajectories:
        return ""

    lines = ["## Relevant Past Experiences\n"]
    for i, traj in enumerate(trajectories, 1):
        task = traj.get("user_request", "")[:100]
        score = traj.get("quality_score", 0)
        mode = traj.get("reasoning_strategy", traj.get("reasoning_mode", "nexus"))
        tools = []
        try:
            calls = json.loads(traj.get("tool_calls") or "[]")
            tools = [c.get("tool", "") for c in calls if c.get("tool")]
        except Exception:
            pass
        answer = traj.get("final_answer", "")[:150]

        lines.append(
            f"### Experience {i} (score: {score:.1f}, mode: {mode})\n"
            f"- Task: {task}\n"
            f"- Tools: {', '.join(tools[:5]) or 'none'}\n"
            f"- Outcome: {answer}\n"
        )

    return "\n".join(lines)


async def _store_experience_replay() -> None:
    """Store high-scoring recent task executions as reusable trajectories.

    Scans recent task scores for excellent executions (score >= 8.5) and
    stores them as full trajectories in the task_trajectories table.
    Also records tool chains used in successful executions.
    """
    try:
        scores = await db.get_task_scores(limit=20)
        if not scores:
            return

        for s in scores:
            score = s.get("score", 0)
            if score < 8.5:  # Only store truly excellent executions
                continue

            session_id = s.get("session_id", "")
            if not session_id:
                continue

            # Check if trajectory already stored for this session
            existing = await db.get_trajectory_by_session(session_id)
            if existing:
                continue

            task_type = _classify_task_from_score(s)
            task_text = s.get("task", "") or ""

            # Get tools used in this session
            tools_used = []
            tool_calls = []
            try:
                executions = await db.get_tool_executions(session_id=session_id, limit=20)
                for ex in executions:
                    tool_name = ex.get("tool_name", "")
                    tools_used.append(tool_name)
                    tool_calls.append({
                        "tool": tool_name,
                        "params": json.loads(ex.get("input") or "{}"),
                        "result": (ex.get("output") or "")[:500],
                        "duration": ex.get("duration_ms", 0),
                        "success": bool(ex.get("success")),
                    })
            except Exception:
                pass

            if not tools_used:
                continue

            # Build and store trajectory
            trajectory = TaskTrajectory(
                session_id=session_id,
                user_request=task_text[:500],
                task_type=task_type,
                reasoning_strategy="nexus",
                plan_steps=[],
                tool_calls=tool_calls,
                final_answer=(s.get("answer") or "")[:1000],
                quality_score=score,
                token_usage={},
                duration_seconds=0.0,
            )

            traj_id = await store_trajectory(trajectory)

            # Record tool chain for affinity analysis
            successful_tools = [tc["tool"] for tc in tool_calls if tc.get("success")]
            if len(successful_tools) >= 2:
                await db.record_tool_chain(
                    chain=successful_tools,
                    success=True,
                    task_type=task_type,
                    quality_score=score,
                )

        await emit("experience_replay",
                   message="Experience replay scan completed")

    except Exception as e:
        logger.error("Experience replay error: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 2. META-LEARNING ENGINE ═════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_meta_learning() -> None:
    """Learn which task types produce the best quality scores.

    Analyzes recent trajectories to build a task_type → quality_score
    correlation map for the Nexus Framework.
    """
    try:
        trajectories = await db.get_recent_trajectories(limit=50, min_score=3.0)
        if len(trajectories) < 5:
            return

        # Group scores by task_type -> mode -> [scores]
        # FIX: Previously used undefined `mode_scores` instead of the declared `task_scores`.
        # Now correctly uses nested defaultdict: task_type -> mode -> list of scores.
        mode_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for traj in trajectories:
            task_type = traj.get("task_type", "general")
            mode = "nexus"
            score = traj.get("quality_score", 0)
            if score > 0:
                mode_scores[task_type][mode].append(score)

        # For each task type, determine the best mode
        updates = 0
        for task_type, modes in mode_scores.items():
            if not modes:
                continue

            # Calculate average score per mode
            mode_avgs = {}
            for mode, scores in modes.items():
                if len(scores) >= 2:  # Need at least 2 samples
                    mode_avgs[mode] = sum(scores) / len(scores)

            if not mode_avgs:
                continue

            best_mode = max(mode_avgs, key=mode_avgs.get)
            best_score = mode_avgs[best_mode]
            confidence = min(best_score / 10.0, 1.0)

            # Only update if we have meaningful data
            if confidence >= 0.5:
                await db.update_reasoning_preference(
                    task_type=task_type,
                    mode=best_mode,
                    confidence=confidence,
                    sample_count=len(modes[best_mode]),
                    scores_by_mode={k: round(v, 2) for k, v in mode_avgs.items()},
                )
                updates += 1

        if updates > 0:
            await emit("meta_learning",
                       updates=updates,
                       message=f"Meta-learning updated {updates} reasoning preferences")

    except Exception as e:
        logger.error("Meta-learning error: %s", e)


async def get_meta_learning_context(task_type: str) -> str:
    """Get meta-learning insights for a task type to inject into agent context.

    Returns a formatted string with the recommended reasoning strategy and
    supporting evidence, or empty string if no data available.
    """
    pref = await db.get_reasoning_preference(task_type)
    if not pref or pref.get("confidence", 0) < 0.4:
        return ""

    mode = "nexus"
    confidence = pref.get("confidence", 0)
    samples = pref.get("sample_count", 0)

    lines = [
        f"## Meta-Learning Insight (task type: {task_type})",
        f"- Recommended reasoning strategy: **{mode}** (confidence: {confidence:.0%}, {samples} samples)",
    ]

    try:
        scores = json.loads(pref.get("scores_by_mode") or "{}")
        if scores:
            score_lines = [f"  - {m}: avg {s:.1f}" for m, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
            lines.append("- Mode performance:")
            lines.extend(score_lines)
    except Exception:
        pass

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 3. TOOL AFFINITY ANALYSIS ═══════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_tool_affinity_analysis() -> None:
    """Analyze tool usage patterns to discover optimization opportunities.

    Builds co-occurrence matrix, identifies successful tool chains,
    detects anti-patterns, and suggests optimal tool sequences.
    """
    try:
        executions = await db.recent_tool_executions(limit=200)
        if len(executions) < 10:
            return

        # ── Group by session for chain extraction ──
        sessions_tools: dict[str, list[str]] = defaultdict(list)
        for ex in executions:
            sid = ex.get("session_id", "")
            if sid:
                sessions_tools[sid].append(ex.get("tool_name", ""))

        # ── Record tool chains for each session ──
        for sid, tools in sessions_tools.items():
            if len(tools) < 2:
                continue
            # Extract unique tool sequence preserving order
            seen = set()
            unique_chain = []
            for t in tools:
                if t not in seen:
                    seen.add(t)
                    unique_chain.append(t)

            if len(unique_chain) >= 2:
                # Determine success based on session trajectories
                traj = await db.get_trajectory_by_session(sid)
                success = traj and traj.get("quality_score", 0) >= 5.0
                score = traj.get("quality_score", 0) if traj else 0

                await db.record_tool_chain(
                    chain=unique_chain,
                    success=success,
                    task_type=traj.get("task_type", "general") if traj else "general",
                    quality_score=score,
                )

        # ── Detect anti-patterns and generate insights ──
        anti_patterns = await db.get_anti_pattern_chains(limit=5)
        for ap in anti_patterns:
            try:
                tools = json.loads(ap.get("tools") or "[]")
                if len(tools) >= 2:
                    failure_rate = ap.get("failure_rate", 0)
                    if failure_rate > 0.6 and ap.get("execution_count", 0) >= 3:
                        insight = LearningInsight(
                            category="tool_chain",
                            task_type=ap.get("task_type", "general"),
                            content=f"Avoid sequence: {' → '.join(tools[:4])} "
                                    f"(failure rate: {failure_rate:.0%})",
                            confidence=min(failure_rate, 0.9),
                            evidence_count=ap.get("execution_count", 0),
                        )
                        await _store_insight_safely(insight)
            except Exception:
                pass

        # ── Analyze successful chains and generate suggestions ──
        successful = await db.get_successful_chains(limit=5)
        for sc in successful:
            try:
                tools = json.loads(sc.get("tools") or "[]")
                if len(tools) >= 2:
                    avg_score = sc.get("avg_quality_score", 0)
                    success_rate = sc.get("success_rate", 0)
                    if success_rate > 0.7 and avg_score >= 7.0:
                        insight = LearningInsight(
                            category="tool_chain",
                            task_type=sc.get("task_type", "general"),
                            content=f"Recommended sequence: {' → '.join(tools[:4])} "
                                    f"(success rate: {success_rate:.0%}, avg score: {avg_score:.1f})",
                            confidence=min(success_rate * 0.8, 0.95),
                            evidence_count=sc.get("execution_count", 0),
                        )
                        await _store_insight_safely(insight)
            except Exception:
                pass

        await emit("tool_affinity",
                   anti_patterns=len(anti_patterns),
                   successful_chains=len(successful),
                   message=f"Tool affinity analysis: {len(anti_patterns)} anti-patterns, {len(successful)} good chains")

    except Exception as e:
        logger.error("Tool affinity error: %s", e)


async def get_tool_suggestions(task_type: str) -> str:
    """Get tool usage suggestions for a task type based on affinity data.

    Returns formatted string with recommended tools and anti-patterns.
    """
    lines = []

    # Get successful chains
    successful = await db.get_successful_chains(task_type=task_type, limit=3)
    if successful:
        lines.append("## Recommended Tool Sequences")
        for sc in successful[:3]:
            try:
                tools = json.loads(sc.get("tools") or "[]")
                rate = sc.get("success_rate", 0)
                score = sc.get("avg_quality_score", 0)
                lines.append(f"- {' → '.join(tools[:4])} (success: {rate:.0%}, score: {score:.1f})")
            except Exception:
                pass

    # Get anti-patterns
    anti = await db.get_anti_pattern_chains(task_type=task_type, limit=2)
    if anti:
        lines.append("## Tool Anti-Patterns to Avoid")
        for ap in anti[:2]:
            try:
                tools = json.loads(ap.get("tools") or "[]")
                rate = ap.get("failure_rate", 0)
                lines.append(f"- {' → '.join(tools[:4])} (failure: {rate:.0%})")
            except Exception:
                pass

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 4. [REMOVED] ADAPTIVE CRITIC SYSTEM ═══════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 5. FAILURE PATTERN RECOGNITION (NLP-based) ═══════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _mine_reflections() -> None:
    """Scan recent reflections for recurring themes using embedding similarity.

    v17 bug fix: Now uses embedding-based clustering instead of naive word
    frequency counting. This identifies semantic themes rather than surface
    word overlaps.
    """
    try:
        reflections = await db.get_reflections(limit=50)
        if len(reflections) < 5:
            return

        # Build embeddings for reflection content
        from ..memory.memory import embed as _embed

        reflection_texts = [r.get("content", "") for r in reflections]
        embeddings = []
        valid_indices = []

        for i, text in enumerate(reflection_texts):
            if len(text) < 20:
                continue
            try:
                emb = await _embed(text)
                if emb and len(emb) > 0:
                    embeddings.append(emb)
                    valid_indices.append(i)
            except Exception:
                continue

        if len(embeddings) < 3:
            # Fallback to simple clustering if embedding fails
            await _mine_reflections_simple(reflections)
            return

        # Cluster reflections by cosine similarity
        clusters = _cluster_by_similarity(embeddings, threshold=0.7)

        # Process significant clusters (3+ members)
        for cluster_indices in clusters.values():
            if len(cluster_indices) < 3:
                continue

            # Get the representative (most central) reflection
            representative_idx = _find_cluster_representative(
                [embeddings[i] for i in cluster_indices],
                cluster_indices,
            )
            if representative_idx is None:
                continue

            ref_idx = valid_indices[representative_idx]
            theme_content = reflection_texts[ref_idx][:200]
            cluster_size = len(cluster_indices)

            # Check if we already have a similar insight
            existing = await db.find_similar_memory(
                f"reflection pattern {theme_content[:50]}"
            )
            if existing:
                # Boost importance of existing memory
                new_imp = min((existing.get("importance") or 0.5) + 0.05, 0.95)
                await db.update_memory_importance(existing["id"], new_imp)
                continue

            # Generate an actionable countermeasure
            countermeasure = await _generate_failure_countermeasure(
                theme_content, cluster_size
            )
            if countermeasure:
                insight = LearningInsight(
                    category="failure_pattern",
                    task_type="general",
                    content=f"Reflection theme (n={cluster_size}): {theme_content[:150]}. "
                            f"Countermeasure: {countermeasure}",
                    confidence=0.6,
                    evidence_count=cluster_size,
                )
                await _store_insight_safely(insight)

                # Also store as procedural memory
                if await _check_self_learned_cap():
                    continue
                try:
                    embedding = await _embed(
                        f"Self-improvement pattern: {countermeasure}"
                    )
                except Exception:
                    embedding = None
                mem_id = await db.add_memory(
                    kind="procedural",
                    content=(
                        f"[Learned Pattern] Recurring reflection theme ({cluster_size} occurrences): "
                        f"{theme_content[:150]}. Countermeasure: {countermeasure}"
                    )[:500],
                    importance=0.75,
                    embedding=embedding,
                    tags=["self-learning", "reflection_mining", "failure_pattern"],
                )

    except Exception as e:
        logger.error("Reflection mining error: %s", e)


async def _mine_reflections_simple(reflections: list[dict]) -> None:
    """Fallback: simple reflection mining when embeddings are unavailable."""
    # Group by keyword overlap (crude but functional)
    texts = [r.get("content", "").lower() for r in reflections
             if len(r.get("content", "")) >= 20]
    if len(texts) < 3:
        return

    # Build word sets and find overlapping themes
    word_sets = []
    for t in texts:
        words = set(w.strip(".,;:!?()[]{}\"'-") for w in t.split() if len(w.strip(".,;:!?()[]{}\"'-")) >= 4)
        word_sets.append(words)

    # Find pairs with high overlap
    clusters: dict[int, list[int]] = {}
    for i, ws_i in enumerate(word_sets):
        for j, ws_j in enumerate(word_sets):
            if i >= j:
                continue
            overlap = len(ws_i & ws_j)
            if overlap >= 3:
                # Merge clusters
                merged_key = None
                for key, members in clusters.items():
                    if i in members or j in members:
                        merged_key = key
                        break
                if merged_key is not None:
                    clusters[merged_key].extend([i, j])
                    clusters[merged_key] = list(set(clusters[merged_key]))
                else:
                    clusters[len(clusters)] = [i, j]

    for key, members in clusters.items():
        if len(set(members)) < 3:
            continue
        unique_members = list(set(members))
        theme = texts[unique_members[0]][:200]
        await emit("reflection_mining",
                   cluster_size=len(unique_members),
                   theme_preview=theme[:80],
                   message=f"Reflection cluster found ({len(unique_members)} similar reflections)")


def _cluster_by_similarity(embeddings: list[list[float]],
                            threshold: float = 0.7) -> dict[int, list[int]]:
    """Cluster embeddings by cosine similarity using single-linkage clustering.

    Returns a dict mapping cluster_id -> list of indices.
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    return clusters


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_cluster_representative(embeddings: list[list[float]],
                                  indices: list[int]) -> int | None:
    """Find the most central item in a cluster (highest avg similarity to others)."""
    if len(embeddings) < 2:
        return 0 if embeddings else None

    best_idx = 0
    best_score = -1.0

    for i, emb_i in enumerate(embeddings):
        total_sim = 0.0
        for j, emb_j in enumerate(embeddings):
            if i != j:
                total_sim += _cosine_similarity(emb_i, emb_j)
        avg_sim = total_sim / (len(embeddings) - 1)
        if avg_sim > best_score:
            best_score = avg_sim
            best_idx = i

    return best_idx


async def _generate_failure_countermeasure(theme: str, occurrence_count: int) -> str:
    """Generate an actionable countermeasure for a recurring failure pattern."""
    try:
        prompt = [{"role": "system", "content": (
            "You are an AI agent analysing your own failure patterns to improve. "
            "Given a recurring theme from your reflections, generate a SPECIFIC "
            "countermeasure that can prevent this pattern from recurring.\n\n"
            "Rules:\n"
            "- Start with an action verb (Check, Verify, Always, Avoid, Use...)\n"
            "- Be specific about WHAT to do differently\n"
            "- Include a concrete pattern to follow\n"
            "- Keep it under 80 words\n"
            "- No preamble, just the countermeasure"
        )}, {"role": "user", "content": (
            f"Recurring reflection theme ({occurrence_count} occurrences): {theme}\n\n"
            f"Generate a specific countermeasure to prevent this pattern."
        )}]
        response = await llm.complete(
            prompt,
            model=config.get("memory_model"),
            max_tokens=100,
            temperature=0.4,
        )
        text = _clean_llm_output(response.content or "")
        return text[:_MAX_STRATEGY_LENGTH]
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 6. STRATEGY INJECTION SYSTEM ═════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _manage_strategy_lifecycle() -> None:
    """Manage the lifecycle of learned strategies.

    Strategy lifecycle:
    - proposed → testing → validated → expired (if effectiveness drops)
    - testing → expired (if effectiveness is negative)
    - validated → expired (if confidence drops below threshold)

    Promotes strategies with good evidence and demotes underperformers.
    """
    try:
        active_insights = await db.get_active_insights(limit=100)
        promoted = 0
        expired = 0

        for insight in active_insights:
            status = insight.get("status", "proposed")
            evidence = insight.get("evidence_count", 1)
            confidence = insight.get("confidence", 0.5)

            if status == "proposed" and evidence >= 3 and confidence >= 0.6:
                # Promote to testing
                await db.update_insight_effectiveness(insight["id"], insight.get("effectiveness", 0.0))
                promoted += 1
            elif status == "testing" and evidence >= 5 and confidence >= 0.7:
                # Promote to validated
                await db.update_insight_effectiveness(
                    insight["id"],
                    max(insight.get("effectiveness", 0.0), 0.3),
                )
                promoted += 1
            elif status in ("proposed", "testing") and evidence >= 3 and confidence < 0.2:
                # Expire low-confidence strategies
                await db.update_insight_effectiveness(insight["id"], -0.1)
                expired += 1

        if promoted or expired:
            await emit("strategy_lifecycle",
                       promoted=promoted, expired=expired,
                       message=f"Strategy lifecycle: {promoted} promoted, {expired} expired")

    except Exception as e:
        logger.error("Strategy lifecycle error: %s", e)


async def get_strategy_injections(task_type: str) -> str:
    """Get top-3 validated strategies for a task type to inject into tool descriptions.

    Returns formatted string with strategies, or empty string if none available.
    """
    insights = await db.get_insights_for_task_type(task_type, limit=_MAX_ACTIVE_STRATEGIES_PER_TYPE)
    if not insights:
        return ""

    lines = ["## Learned Strategies"]
    for i, insight in enumerate(insights, 1):
        content = insight.get("content", "")
        confidence = insight.get("confidence", 0)
        effectiveness = insight.get("effectiveness", 0)
        lines.append(
            f"{i}. {content[:_MAX_STRATEGY_LENGTH]} "
            f"(confidence: {confidence:.0%})"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 7. QUALITY FEEDBACK LOOP ════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _quality_feedback_loop() -> None:
    """Monitor quality scores and generate intervention strategies.

    When quality declines across tasks:
    - Suggest more rigorous reasoning strategy
    - Add more tools to context

    When quality is consistently high:
    - Optimize for speed with streamlined reasoning
    """
    try:
        scores = await db.recent_task_scores(limit=30)
        if len(scores) < 10:
            return

        recent_avg = sum(s.get("score", 0) for s in scores[:10]) / 10
        older_avg = sum(s.get("score", 0) for s in scores[10:20]) / max(len(scores[10:20]), 1)

        diff = recent_avg - older_avg

        if diff < -1.5:
            # Quality declining significantly — trigger intervention
            await _trigger_quality_intervention(recent_avg, older_avg, scores)
        elif diff > 1.5:
            # Quality improving — optimize for speed
            await _trigger_quality_optimization(recent_avg, older_avg)

    except Exception as e:
        logger.error("Quality feedback loop error: %s", e)


async def _trigger_quality_intervention(recent_avg: float, older_avg: float,
                                         scores: list[dict]) -> None:
    """Trigger intervention when quality is declining."""
    # Find weakest dimensions
    dim_avgs: dict[str, list[float]] = {}
    for s in scores[:10]:
        try:
            dims = json.loads(s.get("dimensions") or "{}")
            for k, v in dims.items():
                dim_avgs.setdefault(k, []).append(float(v))
        except Exception:
            pass

    weakest = ""
    if dim_avgs:
        weakest_dim = min(dim_avgs, key=lambda k: sum(dim_avgs[k]) / len(dim_avgs[k]))
        weakest = f"Weakest dimension: {weakest_dim}"

    issues = []
    for s in scores[:5]:
        try:
            parsed = json.loads(s.get("issues") or "[]")
            if parsed:
                issues.extend(parsed[:2])
        except Exception:
            pass

    learning = await _generate_strategy(
        f"Agent quality declining from {older_avg:.1f} to {recent_avg:.1f}. {weakest}. "
        f"Recent issues: {'; '.join(issues[:3])}",
        "quality_intervention",
    )

    if learning:
        if await _check_self_learned_cap():
            return
        from ..memory.memory import embed
        content = (
            f"[Quality Intervention] Reasoning quality improvement: {learning}. "
            f"Quality dropped from {older_avg:.1f} to {recent_avg:.1f}. "
            f"RECOMMENDATION: Task may benefit from more thorough analysis, "
            f"include more tools in context."
        )[:500]
        try:
            embedding = await embed(content)
        except Exception:
            embedding = None
        await db.add_memory(
            kind="procedural",
            content=content,
            importance=0.90,
            embedding=embedding,
            tags=["self-learning", "quality", "intervention"],
        )

        # Store as a learning insight too
        insight = LearningInsight(
            category="tool_strategy",
            task_type="general",
            content=f"Quality intervention: {learning}",
            confidence=0.7,
        )
        await _store_insight_safely(insight)

    await emit("quality_intervention",
               learning=learning,
               message=f"Quality declining — intervention triggered: {learning[:80] if learning else 'none'}")


async def _trigger_quality_optimization(recent_avg: float, older_avg: float) -> None:
    """Optimize for speed when quality is consistently high."""
    await emit("quality_trend", trend="improving",
               avg_recent=recent_avg, avg_older=older_avg,
               message=f"Quality improving: {older_avg:.1f} → {recent_avg:.1f}. "
                       f"Consider using streamlined reasoning strategy for speed.")


async def get_quality_feedback_context() -> str:
    """Get current quality feedback context for reasoning engine injection."""
    try:
        scores = await db.recent_task_scores(limit=20)
        if len(scores) < 5:
            return ""

        recent_avg = sum(s.get("score", 0) for s in scores[:5]) / 5
        all_avg = sum(s.get("score", 0) for s in scores) / len(scores)

        lines = ["## Quality Feedback Context"]
        lines.append(f"- Recent 5-task average: {recent_avg:.1f}/10")
        lines.append(f"- Last 20-task average: {all_avg:.1f}/10")

        if recent_avg < 5.0:
            lines.append("- STATUS: Quality is LOW — use more thorough reasoning, include more tools")
        elif recent_avg < 7.0:
            lines.append("- STATUS: Quality is MODERATE — balanced approach recommended")
        else:
            lines.append("- STATUS: Quality is HIGH — can optimize for speed")

        return "\n".join(lines)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 8. MEMORY CONSOLIDATION INTELLIGENCE ═════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _consolidate_memories() -> None:
    """Smarter memory pruning with semantic deduplication.

    Instead of naive deletion, this:
    1. Clusters related memories using content similarity
    2. Keeps the best representative from each cluster
    3. Archives (not deletes) pruned memories for potential recovery
    4. Promotes frequently-used memories to higher importance
    5. Tracks which memories are actually used in task execution
    """
    try:
        stats = await db.get_memory_usage_stats()
        total = stats.get("total", 0)

        if total < 800:
            # No need to consolidate yet
            return

        # ── Phase 1: Semantic deduplication ──
        await _semantic_dedup()

        # ── Phase 2: Promote frequently-used memories ──
        await _promote_used_memories()

        # ── Phase 3: Archive stale, low-value memories ──
        pruned = await db.prune_stale_memories(
            max_age_days=90,
            min_importance=0.2,
            min_access_count=1,
            max_memories=800,
        )

        if pruned > 0:
            await emit("memory_consolidation",
                       pruned=pruned, remaining=total - pruned,
                       message=f"Memory consolidation: {pruned} memories archived, {total - pruned} remaining")

    except Exception as e:
        logger.error("Memory consolidation error: %s", e)


async def _semantic_dedup() -> None:
    """Cluster related memories and keep the best representative.

    Uses embedding similarity to group similar memories, then
    archives the less important duplicates.
    """
    try:
        from ..memory.memory import embed as _embed

        # Get procedural and semantic memories
        memories = await db.all_memories(kind="procedural", limit=200)
        memories += await db.all_memories(kind="semantic", limit=200)

        if len(memories) < 20:
            return

        # Build embeddings
        text_mem_pairs = []
        embeddings = []
        for m in memories:
            content = m.get("content", "")
            if len(content) < 15:
                continue
            try:
                emb = await _embed(content)
                if emb and len(emb) > 0:
                    text_mem_pairs.append((content, m))
                    embeddings.append(emb)
            except Exception:
                continue

        if len(embeddings) < 10:
            return

        # Cluster
        clusters = _cluster_by_similarity(embeddings, threshold=0.85)

        # For clusters with multiple members, keep the best and archive the rest
        archived_count = 0
        for cluster_id, member_indices in clusters.items():
            if len(member_indices) < 2:
                continue

            # Sort by importance (descending)
            sorted_members = sorted(
                member_indices,
                key=lambda idx: text_mem_pairs[idx][1].get("importance", 0),
                reverse=True,
            )

            # Keep the most important, archive the rest
            for idx in sorted_members[1:]:
                mem = text_mem_pairs[idx][1]
                mem_id = mem.get("id")
                if mem_id and not mem.get("permanent"):
                    await db.archive_memory(mem_id, reason="semantic_dedup")
                    archived_count += 1

        if archived_count > 0:
            await emit("semantic_dedup",
                       archived=archived_count,
                       message=f"Semantic deduplication archived {archived_count} duplicate memories")

    except Exception:
        pass  # Non-critical, don't propagate


async def _promote_used_memories() -> None:
    """Promote memories that are frequently accessed in task execution."""
    try:
        stats = await db.get_memory_usage_stats()
        total = stats.get("total", 0)
        recent = stats.get("recently_accessed_7d", 0)

        if total == 0:
            return

        # If a memory has been accessed recently, give it a small importance boost
        access_ratio = recent / total
        if access_ratio < 0.1:
            # Most memories aren't being used — this is fine, just log it
            return

        # Get memories with high access counts and boost them
        mem_ids = await db.get_frequently_accessed_memories(
            min_access_count=5, max_importance=0.9, limit=20
        )
        for mem_id in mem_ids:
            await db.promote_memory(mem_id, importance_boost=0.02)

    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 9. LEARNING RATE & DECAY ════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _apply_learning_decay() -> int:
    """Apply time-based confidence decay to all active learnings.

    Learning weight calculation:
    - New learnings have weight = _LEARNING_WEIGHT_NEW (1.0)
    - Weight decays exponentially with age: weight = base * 2^(-age/half_life)
    - Learnings validated by experience get _LEARNING_VALIDATED_BOOST
    - Learnings contradicted by experience lose _LEARNING_CONTRADICTED_PENALTY
    - Learnings below _LEARNING_MIN_CONFIDENCE are expired

    Returns count of insights that were expired by this decay.
    """
    try:
        return await db.decay_insight_confidence()
    except Exception:
        return 0


def _compute_learning_weight(confidence: float, created_at: float,
                              last_validated: float) -> float:
    """Compute the current effective weight of a learning.

    Factors:
    - Base confidence
    - Time decay (exponential with half-life)
    - Validation recency (boosted if recently validated)
    """
    now = time.time()
    age_days = (now - created_at) / 86400

    # Exponential decay
    time_factor = 2.0 ** (-age_days / _LEARNING_DECAY_DAYS)

    # Validation recency boost
    validation_age_days = (now - last_validated) / 86400
    validation_factor = max(0.5, 1.0 - validation_age_days / 30.0)

    return confidence * time_factor * validation_factor


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ INHERITED ANALYSES (from v7, enhanced) ══════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def _resolve_systemic_patterns() -> int:
    """Auto-resolve improvement log patterns that are clearly systemic.

    A pattern is considered systemic if:
    - It has frequency >= 5 (retried many times with no effect)
    - It involves a tool sequence that keeps failing

    Instead of wasting LLM cycles trying to generate strategies for
    unfixable patterns, we mark them resolved and move on.

    Returns count of patterns auto-resolved.
    """
    try:
        entries = await db.list_improvement_log(resolved=False)
        resolved_count = 0
        for entry in entries:
            if (entry.get("frequency") or 0) >= _MAX_SYSTEMIC_FREQUENCY:
                await db.resolve_improvement(entry["id"])
                resolved_count += 1
                logger.info(
                    "[self_improve] Systemic pattern resolved: %s/%s (freq=%d)",
                    entry.get("category", ""), entry.get("tool_name", "?"),
                    entry.get("frequency", 0),
                )
        if resolved_count:
            await emit("self_improvement",
                       category="systemic_patterns_cleanup",
                       count=resolved_count,
                       message=f"Auto-resolved {resolved_count} systemic improvement patterns")
        return resolved_count
    except Exception as exc:
        logger.warning("[self_improve] Systemic pattern resolution failed: %s", exc)
        return 0


async def _analyse_tool_failures() -> None:
    """Find tools failing repeatedly and write actionable STRATEGIES to memory."""
    entries = await db.list_improvement_log(resolved=False)
    for entry in entries:
        if entry.get("category") != "tool_failure":
            continue
        if (entry.get("frequency") or 0) < _FAILURE_THRESHOLD:
            continue

        tool_name = entry.get("tool_name", "unknown")
        detail = entry.get("detail", "")
        log_id = entry["id"]

        # Exponential backoff for repeated failures
        backoff = _FAILURE_BACKOFF.get(tool_name, 0)
        if backoff > 0:
            _FAILURE_BACKOFF[tool_name] = backoff - 1
            continue

        # Get the tool's execution history for context
        executions = await db.recent_tool_executions(limit=50)
        tool_execs = [e for e in executions if e.get("tool_name") == tool_name]

        error_patterns = []
        for ex in tool_execs[-5:]:
            if not ex.get("success"):
                output = (ex.get("output") or "")[:200]
                error_patterns.append(output)

        # Check if this pattern has been retried too many times (systemic)
        if (entry.get("frequency") or 0) >= _MAX_SYSTEMIC_FREQUENCY:
            logger.info("[self_improve] Auto-resolving systemic pattern %d: %s/%s",
                        entry['id'], entry.get("category", ""), tool_name)
            await db.resolve_improvement(entry['id'])
            await emit("self_improvement",
                       category="systemic_pattern_resolved",
                       tool=tool_name,
                       frequency=entry.get("frequency", 0),
                       message=f"Auto-resolved systemic pattern for '{tool_name}' "
                               f"(frequency: {entry['frequency']})")
            continue

        # Ask LLM to generate a specific ACTIONABLE strategy
        context = (
            f"Tool '{tool_name}' has failed {entry['frequency']} times. "
            f"Recent errors: {' | '.join(error_patterns[-3:])}"
        )
        strategy = await _generate_strategy(context, tool_name)

        if strategy:
            # Safety: check we haven't exceeded self-learned memory cap
            if await _check_self_learned_cap():
                continue

            # Store as procedural memory
            content = (
                f"[Learned Strategy] Tool '{tool_name}': {strategy}. "
                f"Apply this pattern whenever using {tool_name} to avoid repeated failures."
            )
            from ..memory.memory import embed
            try:
                embedding = await embed(content)
            except Exception:
                embedding = None
            mem_id = await db.add_memory(
                kind="procedural",
                content=content[:500],
                importance=0.85,
                embedding=embedding,
                tags=["self-learning", "tool-strategy", tool_name],
            )

            # Also write to tool profile
            await db.update_tool_strategy(tool_name, strategy)

            # Record as a learning insight
            insight = LearningInsight(
                category="tool_strategy",
                task_type="general",
                content=f"Tool '{tool_name}' strategy: {strategy}",
                confidence=0.8,
            )
            await _store_insight_safely(insight)

            await db.resolve_improvement(log_id, memory_id=mem_id)
            _FAILURE_BACKOFF[tool_name] = 0
            await emit("self_improvement",
                       category="tool_strategy_learned",
                       tool=tool_name,
                       strategy=strategy,
                       message=f"Learned strategy for '{tool_name}': {strategy[:80]}")


async def _analyse_tool_strategies() -> None:
    """Analyze tool usage patterns to discover optimization opportunities."""
    executions = await db.recent_tool_executions(limit=200)
    if len(executions) < 10:
        return

    # Group by tool
    tool_stats: dict[str, dict] = {}
    for ex in executions:
        name = ex.get("tool_name", "unknown")
        if name not in tool_stats:
            tool_stats[name] = {"durations": [], "successes": 0, "failures": 0, "output_lens": []}
        stats = tool_stats[name]
        stats["durations"].append(ex.get("duration_ms", 0) or 0)
        if ex.get("success"):
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["output_lens"].append(len(ex.get("output") or ""))

    # Find tools that are slow but reliable
    for name, stats in tool_stats.items():
        if stats["successes"] + stats["failures"] < 5:
            continue
        avg_dur = sum(stats["durations"]) / len(stats["durations"])
        success_rate = stats["successes"] / (stats["successes"] + stats["failures"])

        if avg_dur > 10000 and success_rate > 0.8:
            profile = await db.get_tool_profile(name)
            if profile and not profile.get("learned_strategy"):
                strategy = (
                    f"This tool averages {avg_dur/1000:.1f}s per call. "
                    f"Consider caching results when possible or batching operations."
                )
                await db.update_tool_strategy(name, strategy)
                await emit("tool_optimization", tool=name, avg_duration_ms=avg_dur,
                           message=f"Optimization tip for '{name}': consider caching/batching")


async def _analyse_quality_trends() -> None:
    """Analyze task quality scores over time to identify reasoning improvements."""
    scores = await db.recent_task_scores(limit=30)
    if len(scores) < 5:
        return

    mid = len(scores) // 2
    first_half = scores[:mid]
    second_half = scores[mid:]

    avg_first = sum(s.get("score", 0) for s in first_half) / len(first_half) if first_half else 0
    avg_second = sum(s.get("score", 0) for s in second_half) / len(second_half) if second_half else 0

    diff = avg_second - avg_first

    if diff < -1.0:
        # Quality declining — generate intervention strategy
        dim_avgs: dict[str, list[float]] = {}
        for s in second_half:
            try:
                dims = json.loads(s.get("dimensions") or "{}")
                for k, v in dims.items():
                    dim_avgs.setdefault(k, []).append(float(v))
            except Exception:
                pass

        weakest = ""
        if dim_avgs:
            weakest_dim = min(dim_avgs, key=lambda k: sum(dim_avgs[k]) / len(dim_avgs[k]))
            weakest = f"Weakest dimension: {weakest_dim}"

        issues = []
        for s in second_half[-3:]:
            try:
                parsed = json.loads(s.get("issues") or "[]")
                if parsed:
                    issues.extend(parsed[:2])
            except Exception:
                pass

        learning = await _generate_strategy(
            f"Agent quality declining from {avg_first:.1f} to {avg_second:.1f}. {weakest}. "
            f"Recent issues: {'; '.join(issues[:3])}",
            "reasoning_quality",
        )
        if learning:
            if await _check_self_learned_cap():
                return

            from ..memory.memory import embed
            content = (
                f"[Learned Strategy] Reasoning quality improvement: {learning}. "
                f"Quality dropped from {avg_first:.1f} to {avg_second:.1f} — apply this countermeasure."
            )
            try:
                embedding = await embed(content)
            except Exception:
                embedding = None
            await db.add_memory(
                kind="procedural",
                content=content[:500],
                importance=0.90,
                embedding=embedding,
                tags=["self-learning", "reasoning", "quality"],
            )
            await emit("quality_intervention", learning=learning,
                       message=f"Quality declining — learned countermeasure: {learning[:80]}")

    elif diff > 1.0:
        await emit("quality_trend", trend="improving",
                   avg_first=avg_first, avg_second=avg_second,
                   message=f"Quality improving: {avg_first:.1f} → {avg_second:.1f}")


async def _analyse_goal_stalls() -> None:
    """Detect goals that have been active for too long with no progress."""
    goals = await db.list_goals(status="active")
    stale_threshold = time.time() - 86400
    for goal in goals:
        if (goal.get("updated_at") or 0) < stale_threshold:
            progress = goal.get("progress") or 0
            await db.log_improvement(
                category="goal_stall",
                detail=f"Goal '{goal.get('title')}' (id={goal['id']}) has been active "
                       f"for >24h with {progress*100:.0f}% progress",
            )
            await emit("self_improvement",
                       category="goal_stall",
                       goal_id=goal["id"],
                       message=f"Goal '{goal.get('title')}' appears stalled")


async def _flag_patterns() -> None:
    """Look at recent tool executions for useful patterns."""
    executions = await db.recent_tool_executions(limit=200)
    failures: dict[str, int] = {}
    for ex in executions:
        if not ex.get("success"):
            name = ex.get("tool_name", "unknown")
            failures[name] = failures.get(name, 0) + 1

    # Check which tools were recently resolved (systemic) — skip them
    _resolved_recently = set()
    try:
        recent_resolved = await db.list_improvement_log(resolved=True, limit=50)
        now = time.time()
        for r in recent_resolved:
            tool = r.get("tool_name")
            resolved_at = r.get("updated_at") or 0
            if tool and (now - resolved_at) < 86400:  # resolved within 24h
                _resolved_recently.add(tool)
    except Exception:
        pass

    for tool_name, count in failures.items():
        if tool_name in _resolved_recently:
            # This tool was flagged, analyzed, and resolved recently.
            # Re-logging it would waste improvement cycles. Skip.
            continue
        if count >= _FAILURE_THRESHOLD:
            await db.log_improvement(
                category="tool_failure",
                tool_name=tool_name,
                detail=f"{count} failures in recent execution log",
                frequency=count,
            )

    await emit("self_improvement_scan",
               tools_checked=len(set(e.get("tool_name") for e in executions)),
               failure_patterns=len(failures),
               message="Self-improvement scan complete")


async def _verify_previous_learnings() -> None:
    """Check if previous learned strategies actually improved outcomes."""
    profiles = await db.get_tool_profiles()
    for p in profiles:
        strategy = p.get("learned_strategy")
        if not strategy:
            continue
        name = p.get("tool_name", "")
        success_rate = p.get("success_rate") or 0
        total = p.get("total_calls") or 0

        if success_rate >= 0.8 and total >= 5:
            try:
                from ..memory.memory import MemoryManager, embed
                memories = await db.all_memories(kind="procedural", limit=100)
                for m in memories:
                    if name in m.get("content", "") and "Learned Strategy" in m.get("content", ""):
                        if (m.get("importance") or 0) < 0.95:
                            await db.update_memory_importance(m["id"], 0.95)
                        break
            except Exception:
                pass


async def _generate_daily_summary() -> None:
    """Generate a daily summary from journal entries and mood data."""
    try:
        from ..memory import database as _db
        entries = await _db.list_journal_entries(limit=50)
        if not entries:
            return

        now = time.time()
        day_ago = now - 86400
        recent = [e for e in entries if (e.get("created_at") or 0) > day_ago]
        if len(recent) < 2:
            return

        mood_summary = await _db.get_mood_summary(days=1)
        moods = [e.get("mood", "") for e in recent if e.get("mood")]
        mood_str = ", ".join(set(moods)) if moods else "not tracked"

        content_parts = ["[Daily Summary]"]
        content_parts.append(f"Journal entries: {len(recent)}")
        if mood_summary:
            mood_parts = [f"{r['mood']}({r['count']})" for r in mood_summary[:5]]
            content_parts.append(f"Mood distribution: {', '.join(mood_parts)}")
        content_parts.append(f"Overall mood: {mood_str}")

        all_tags = []
        for e in recent:
            try:
                tags = json.loads(e.get("tags") or "[]")
                all_tags.extend(tags)
            except Exception:
                pass
        if all_tags:
            tag_counts = Counter(all_tags)
            top_tags = [f"{tag}({count})" for tag, count in tag_counts.most_common(5)]
            content_parts.append(f"Themes: {', '.join(top_tags)}")

        summary_content = ". ".join(content_parts)

        from ..memory.memory import embed
        try:
            embedding = await embed(summary_content)
        except Exception:
            embedding = None
        await _db.add_memory(
            kind="episodic",
            content=summary_content[:500],
            importance=0.6,
            embedding=embedding,
            tags=["daily-summary", "journal"],
        )
        await emit("daily_summary", entries=len(recent), moods=mood_str,
                   message=f"Daily summary generated: {len(recent)} entries, mood: {mood_str}")
    except Exception as e:
        await emit("warn", source="self_improvement", error=f"Daily summary failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ UTILITY FUNCTIONS ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_task_type(query: str) -> str:
    """Classify a user query into a task type for trajectory retrieval.

    Uses simple keyword heuristics for task type classification.
    """
    query_lower = query.lower()

    classification_map = [
        (["code", "program", "function", "class", "debug", "refactor", "implement"],
         "coding"),
        (["write", "essay", "article", "blog", "document", "report", "summary"],
         "writing"),
        (["analyze", "data", "statistics", "chart", "graph", "visualization"],
         "data_analysis"),
        (["search", "find", "lookup", "research", "information"],
         "research"),
        (["plan", "strategy", "project", "organize", "schedule"],
         "planning"),
        (["math", "calculate", "equation", "formula", "compute"],
         "mathematics"),
        (["translate", "language", "localize"],
         "translation"),
        (["explain", "teach", "tutorial", "how", "what is", "describe"],
         "explanation"),
        (["file", "read", "write", "directory", "path", "folder"],
         "file_operations"),
        (["web", "scrape", "crawl", "fetch", "url", "http", "api"],
         "web_operations"),
    ]

    for keywords, task_type in classification_map:
        for kw in keywords:
            if kw in query_lower:
                return task_type

    return "general"


def _classify_task_from_score(score_row: dict) -> str:
    """Classify task type from score metadata.

    v17 bug fix: Removed meaningless @staticmethod decorator.
    This was a module-level function, not a method, so @staticmethod
    was incorrect and confusing.
    """
    task_text = (score_row.get("task") or "").lower()

    if not task_text:
        return "general"

    # Try to extract task type from the task description
    return _classify_task_type(task_text)


def _infer_reasoning_strategy(executions: list[dict]) -> str:
    """Infer the reasoning strategy used based on tool execution patterns.

    Heuristic:
    - Nexus Framework: always uses adaptive reasoning
    """
    if not executions:
        return "nexus"

    # Count distinct tool types
    tools = set(e.get("tool_name", "") for e in executions)

    # Check for branching patterns (same tool called multiple times)
    tool_counts = Counter(e.get("tool_name", "") for e in executions)
    has_branching = any(c >= 3 for c in tool_counts.values())

    # Check for multi-step patterns
    if len(tools) >= 4:
        if has_branching:
            return "nexus"
        return "nexus"
    elif len(tools) >= 2:
        return "nexus"
    else:
        return "nexus"


async def _store_insight_safely(insight: LearningInsight) -> int | None:
    """Store a learning insight with cap checking.

    Returns the insight ID or None if cap was reached.
    """
    try:
        active = await db.get_active_insights(limit=_MAX_ACTIVE_INSIGHTS + 1)
        if len(active) >= _MAX_ACTIVE_INSIGHTS:
            return None
        return await db.store_learning_insight({
            "category": insight.category,
            "task_type": insight.task_type,
            "content": insight.content[:1000],
            "confidence": insight.confidence,
            "evidence_count": insight.evidence_count,
            "effectiveness": insight.effectiveness,
        })
    except Exception:
        return None


async def _generate_strategy(context: str, topic: str) -> str:
    """Ask the LLM to generate an actionable strategy from a failure pattern."""
    try:
        prompt = [{"role": "system", "content": (
            "You are an AI agent analysing your own patterns to improve. "
            "Generate a SPECIFIC, ACTIONABLE strategy that can be applied "
            "to avoid this problem in the future.\n\n"
            "Rules:\n"
            "- Start with an action verb (Check, Verify, Always, Avoid, Use...)\n"
            "- Be specific about WHAT to do differently\n"
            "- Include a concrete pattern to follow\n"
            "- Keep it under 100 words\n"
            "- No preamble, just the strategy"
        )}, {"role": "user", "content": (
            f"Context: {context}\n\n"
            f"Generate a specific strategy to improve '{topic}'."
        )}]
        response = await llm.complete(
            prompt,
            model=config.get("memory_model"),
            max_tokens=120,
            temperature=0.4,
        )
        text = _clean_llm_output(response.content or "")
        return text[:_MAX_STRATEGY_LENGTH]
    except Exception:
        return ""


def _clean_llm_output(text: str) -> str:
    """Clean LLM output by removing artifacts and extra whitespace."""
    text = text.replace("\u200b", "").replace("Î", "").strip()
    # Strip thinking tags
    for tag in ["<think thinking>", "</think thinking>", "<think model>", "</think model>"]:
        text = text.replace(tag, "")
    text = text.replace("💭", "").replace("✨", "").strip()
    # Collapse multiple newlines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 10. CODE REVIEW QUALITY ASSESSMENT SYSTEM ════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
# Self-supervised quality feedback: evaluates the agent's own responses and
# tool outputs for correctness, completeness, and efficiency.  Generates
# improvement suggestions that feed back into the meta-learning loop.
#
# Unlike the Quality Feedback Loop (§7) which monitors aggregate scores, this
# system evaluates INDIVIDUAL responses and tool outputs in real-time, closes
# the loop by storing improvement suggestions as procedural skills.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualityAssessment:
    """Result of a single quality assessment on a response or tool output."""
    score: float  # 0.0–10.0
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    dimension_scores: dict[str, float]  # e.g. {"correctness": 8.0, "completeness": 7.5, "efficiency": 9.0}
    category: str  # "response_quality" | "tool_output" | "code_quality"
    context: str  # Brief description of what was assessed


# Dimension weights for final score aggregation
_QUALITY_DIMENSIONS = {
    "correctness": 0.35,
    "completeness": 0.25,
    "clarity": 0.15,
    "efficiency": 0.15,
    "safety": 0.10,
}


async def assess_response_quality(
    user_request: str,
    assistant_response: str,
    tool_calls: list[dict] | None = None,
    duration_ms: float = 0.0,
) -> QualityAssessment:
    """Assess the quality of an assistant response using heuristic rules.

    This is a lightweight, no-LLM evaluation that runs in milliseconds.
    It looks for structural quality signals in the response:
    - Is it helpful and direct (not apologetic or evasive)?
    - Does it address the user's request directly?
    - Are results properly cited / attributed?
    - Is the response well-structured?
    - Were tool calls efficient (not excessive)?

    Returns a QualityAssessment with score, strengths, weaknesses, and
    improvement suggestions.
    """
    user_low = user_request.lower()
    resp = assistant_response or ""
    resp_low = resp.lower()

    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    dims: dict[str, float] = {}

    # ── Correctness ────────────────────────────────────────────────────────
    correctness = 7.5  # Start at neutral

    # Check for "I don't know" - not necessarily bad, but indicates uncertainty
    if any(p in resp_low for p in ["i don't know", "i'm not sure", "i am not sure", "i cannot determine"]):
        weaknesses.append("Response expressed uncertainty without offering alternatives")
        suggestions.append("When uncertain, suggest ways to find the answer rather than stopping")
        correctness -= 1.0

    # Check for contradictory statements
    if _has_contradictions(resp):
        weaknesses.append("Response contains contradictory statements")
        suggestions.append("Review response for internal consistency before responding")
        correctness -= 2.0

    # Check for hallucination markers (claims without evidence)
    hallucination_markers = ["i found", "research shows", "according to my analysis", "based on search"]
    has_tool_calls = bool(tool_calls)
    for marker in hallucination_markers:
        if marker in resp_low and not has_tool_calls:
            weaknesses.append(f"Response uses '{marker}' without actually calling a tool to verify")
            suggestions.append("Always call a tool before claiming you found or researched something")
            correctness -= 1.5
            break

    # ── Completeness ───────────────────────────────────────────────────────
    completeness = 7.0

    # Does the response directly answer the user's question?
    if "?" in user_request:
        # Extract question type
        if user_request.strip().endswith("?"):
            # Check if response length is reasonable for the question
            if len(resp) < 20:
                weaknesses.append("Response is very short for a question — may be incomplete")
                suggestions.append("Provide a more complete answer with context and examples")
                completeness -= 2.0
            else:
                strengths.append("Response length is appropriate for the question asked")
                completeness += 1.0

    # Check if response has structure (bullets, numbered lists, sections)
    has_structure = any(
        resp.count(pat) >= 2
        for pat in ["\n- ", "\n* ", "\n1. ", "\n2. ", "## ", "\n\n"]
    )
    if has_structure:
        strengths.append("Response is well-structured with clear organization")
        completeness += 1.5
    else:
        weaknesses.append("Response lacks structural organization (bullets, sections)")
        suggestions.append("Use bullet points, numbered lists, or sections for complex responses")
        completeness -= 0.5

    # Check for actionable content (next steps, concrete suggestions)
    if any(p in resp_low for p in ["you can", "try this", "here's how", "steps", "first", "next", "finally"]):
        strengths.append("Response provides actionable steps or concrete guidance")
        completeness += 1.0

    # ── Clarity ────────────────────────────────────────────────────────────
    clarity = 7.5

    # Check for excessive hedging
    hedging_words = ["maybe", "perhaps", "possibly", "could be", "might be", "kind of"]
    hedging_count = sum(1 for w in hedging_words if f" {w} " in f" {resp_low} ")
    if hedging_count >= 3:
        weaknesses.append(f"Overly hedged ({hedging_count} hedging words) — be more direct")
        suggestions.append("Reduce hedging language — be confident and direct in answers")
        clarity -= 1.0

    # Check for excessive length (verbose without substance)
    if len(resp) > 3000 and len(resp.split()) > 400:
        words = resp.split()
        if len(words) > 400:
            weaknesses.append("Response is overly verbose — could be more concise")
            suggestions.append("Aim for concise responses — trim redundant explanations")
            clarity -= 1.0
    else:
        strengths.append("Response length is appropriate")
        clarity += 0.5

    # ── Efficiency (tool usage) ───────────────────────────────────────────
    efficiency = 8.0
    if tool_calls:
        if len(tool_calls) > 10:
            weaknesses.append(f"Used {len(tool_calls)} tool calls — consider batching")
            suggestions.append("Batch tool calls when possible to reduce latency")
            efficiency -= 1.0
        # Check for redundant tool calls
        tool_names = [t.get("name", "") for t in tool_calls]
        if len(tool_names) >= 4:
            name_counts = {}
            for n in tool_names:
                name_counts[n] = name_counts.get(n, 0) + 1
            for name, count in name_counts.items():
                if count >= 3:
                    weaknesses.append(f"Tool '{name}' called {count} times — consider caching")
                    suggestions.append(f"Cache results from '{name}' to avoid repeated calls")
                    efficiency -= 0.5
        # Check for fast execution
        if duration_ms > 0 and duration_ms < 5000:
            strengths.append("Response generated quickly (< 5s)")
            efficiency += 1.0
        elif duration_ms > 30000:
            weaknesses.append(f"Response took {duration_ms/1000:.0f}s — could be faster")
            suggestions.append("Consider using the fast-response mode for quicker answers")
            efficiency -= 1.0

    # ── Safety ─────────────────────────────────────────────────────────────
    safety = 9.0

    # Check for code execution without safety warnings
    if "```" in resp and not has_tool_calls:
        if any(dangerous in resp_low for dangerous in ["rm -rf", "sudo ", "chmod 777", "eval(", "exec("]):
            weaknesses.append("Generated potentially dangerous code without safety warning")
            suggestions.append("Always include safety warnings when generating destructive commands")
            safety -= 3.0

    # Compile final score
    dims = {
        "correctness": max(1.0, min(10.0, correctness)),
        "completeness": max(1.0, min(10.0, completeness)),
        "clarity": max(1.0, min(10.0, clarity)),
        "efficiency": max(1.0, min(10.0, efficiency)),
        "safety": max(1.0, min(10.0, safety)),
    }
    final_score = sum(dims[k] * _QUALITY_DIMENSIONS[k] for k in _QUALITY_DIMENSIONS)
    final_score = max(1.0, min(10.0, final_score))

    return QualityAssessment(
        score=round(final_score, 1),
        strengths=strengths[:3],
        weaknesses=weaknesses[:3],
        improvement_suggestions=suggestions[:3],
        dimension_scores=dims,
        category="response_quality",
        context=f"Assessed response to: {user_request[:100]}",
    )


def _has_contradictions(text: str) -> bool:
    """Heuristic check for contradictory statements in text."""
    text_low = text.lower()
    # Look for explicit contradiction signal pairs
    contradiction_pairs = [
        ("always", "sometimes"),
        ("never", "sometimes"),
        ("always", "occasionally"),
        ("never", "always"),
        ("cannot", "can easily"),
        ("not possible", "possible"),
        ("everyone", "some people"),
        ("all", "some"),
    ]
    for a, b in contradiction_pairs:
        if a in text_low and b in text_low:
            # Check they're in different sentences (not same sentence qualifier)
            sentences = text_low.split(". ")
            a_sentences = [s for s in sentences if a in s]
            b_sentences = [s for s in sentences if b in s]
            if a_sentences and b_sentences:
                # They're in different sentences — possible contradiction
                return True
    return False


async def assess_code_quality(
    code: str,
    language: str = "python",
    task_description: str = "",
) -> QualityAssessment:
    """Assess the quality of generated code.

    Evaluates:
    - Syntax correctness (basic heuristics)
    - Documentation (comments, docstrings)
    - Best practices (error handling, typing)
    - Safety (no dangerous patterns)

    Args:
        code: The generated code string.
        language: Programming language ("python", "javascript", etc.)
        task_description: What the code was supposed to do.

    Returns:
        QualityAssessment with code-specific dimensions.
    """
    import ast

    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []

    # ── Syntax correctness ─────────────────────────────────────────────────
    syntax_score = 8.0
    if language == "python":
        try:
            ast.parse(code)
            strengths.append("Python code is syntactically valid")
            syntax_score = 10.0
        except SyntaxError as e:
            weaknesses.append(f"Python syntax error: {e}")
            suggestions.append(f"Fix the syntax error at line {e.lineno}: {e.msg}")
            syntax_score = 3.0

    # ── Documentation ──────────────────────────────────────────────────────
    doc_score = 5.0  # neutral
    if '"""' in code or "'''" in code:
        strengths.append("Code includes docstrings")
        doc_score = 8.0
    if "#" in code:
        strengths.append("Code includes inline comments")
        doc_score += 1.0
    else:
        weaknesses.append("Code lacks inline comments")
        suggestions.append("Add comments to explain non-obvious logic")
        doc_score = 4.0

    # ── Best practices ────────────────────────────────────────────────────
    practice_score = 7.0

    # Error handling
    if "try:" in code or "except" in code or "Exception" in code:
        strengths.append("Code includes proper error handling")
        practice_score += 1.0
    else:
        weaknesses.append("Code lacks error handling (try/except blocks)")
        suggestions.append("Add try/except blocks with specific exception types")
        practice_score -= 1.0

    # Type hints (Python)
    if language == "python" and ": " in code and "def " in code:
        # Check if any function has type hints
        has_hints = False
        for line in code.split("\n"):
            if "def " in line and ":" in line:
                # def foo(x: int) -> str:
                if ": " in line and ") ->" in line:
                    has_hints = True
                    break
        if has_hints:
            strengths.append("Code uses type hints for better readability")
            practice_score += 1.0

    # ── Safety ────────────────────────────────────────────────────────────
    safety_score = 9.0
    dangerous_patterns = [
        ("rm -rf /", "Destructive file removal"),
        ("os.system", "Shell injection risk"),
        ("subprocess.run.*shell=True", "Shell injection risk"),
        ("eval(", "Code injection risk"),
        ("exec(", "Code injection risk"),
        ("pickle.load", "Deserialization risk"),
    ]
    for pattern, risk in dangerous_patterns:
        if pattern in code:
            weaknesses.append(f"Potentially unsafe: {risk}")
            suggestions.append(f"Replace {pattern} with safer alternatives")
            safety_score -= 2.0

    dims = {
        "syntax": min(10.0, syntax_score),
        "documentation": min(10.0, doc_score),
        "best_practices": min(10.0, practice_score),
        "safety": min(10.0, safety_score),
    }
    code_dimensions = {"syntax": 0.35, "documentation": 0.20, "best_practices": 0.25, "safety": 0.20}
    final_score = sum(dims[k] * code_dimensions[k] for k in code_dimensions)

    return QualityAssessment(
        score=round(final_score, 1),
        strengths=strengths[:3],
        weaknesses=weaknesses[:3],
        improvement_suggestions=suggestions[:3],
        dimension_scores=dims,
        category="code_quality",
        context=f"Code assessment for: {task_description[:100]}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 11. SELF-SUPERVISED LEARNING FROM OUTCOMES ═══════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
# Learns from the outcomes of tool calls and responses to improve future
# behavior, without requiring explicit user feedback, via:
# 1. Outcome prediction — predict whether a tool call will succeed
# 2. Pattern extraction — extract reusable patterns from successful sequences
# 3. Negative mining — learn from failed tool calls to avoid repeating mistakes
# ═══════════════════════════════════════════════════════════════════════════════


# Outcome memory — stores (tool, args_preview, success, duration) tuples
# for statistical learning
_outcome_memory: dict[str, list[dict]] = {}  # tool_name -> [outcomes]
_OUTCOME_MEMORY_MAX = 100  # max outcomes per tool


# ═══ SELF-SUPERVISED LEARNING FROM OUTCOMES — extracted to predict.py ════

from .predict import record_outcome, predict_tool_success, extract_success_pattern  # noqa: F811


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ 12. IMPROVEMENT PIPELINE ═════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
# End-to-end pipeline that:
# 1. Collects quality assessments from response quality and code quality checks
# 2. Identifies recurring weakness patterns
# 3. Generates improvement strategies
# 4. Validates improvements against future outcomes
# 5. Promotes validated improvements to active strategies
# 6. Archives stale or disproven improvements
# ═══════════════════════════════════════════════════════════════════════════════


# Track weakness patterns: weakness_text -> {count, first_seen, last_seen, addressed}
_weakness_tracker: dict[str, dict] = {}
_WEAKNESS_TRACKER_MAX = 200  # prevent unbounded growth

# Will be populated from database on first access via _ensure_weakness_tracker_loaded()
_weakness_tracker_loaded: bool = False




async def _ensure_weakness_tracker_loaded() -> None:
    """Load weakness tracker from database on first call."""
    global _weakness_tracker_loaded
    if not _weakness_tracker_loaded:
        try:
            loaded = await db.load_weakness_tracker()
            if loaded:
                _weakness_tracker.clear()
                _weakness_tracker.update(loaded)
            _weakness_tracker_loaded = True
        except Exception:
            _weakness_tracker_loaded = True  # Don't retry on failure

async def process_quality_assessment(assessment: QualityAssessment) -> bool:
    """Process a quality assessment through the improvement pipeline.

    Steps:
    1. Record weaknesses (increment counters)
    2. If a weakness has been seen >= 3 times, generate improvement strategy
    3. If improvement is already active, reinforce it
    4. Return True if improvement was stored, False otherwise

    Args:
        assessment: A QualityAssessment from assess_response_quality or
                    assess_code_quality.

    Returns:
        True if an improvement was stored or reinforced.
    """
    await _ensure_weakness_tracker_loaded()
    if not assessment.weaknesses:
        return False

    stored = False

    for weakness in assessment.weaknesses:
        # Normalize weakness text for tracking
        norm = weakness.lower().strip().rstrip(".!?")

        if norm not in _weakness_tracker:
            if len(_weakness_tracker) >= _WEAKNESS_TRACKER_MAX:
                stale = min(_weakness_tracker, key=lambda k: _weakness_tracker[k]["last_seen"])
                del _weakness_tracker[stale]
            _weakness_tracker[norm] = {
                "count": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "addressed": False,
            }

        _weakness_tracker[norm]["count"] += 1
        _weakness_tracker[norm]["last_seen"] = time.time()

        # If this weakness has appeared >= 3 times, generate an improvement
        if _weakness_tracker[norm]["count"] >= 3 and not _weakness_tracker[norm]["addressed"]:
            # Find corresponding suggestion
            suggestion_idx = None
            for i, w in enumerate(assessment.weaknesses):
                if w == weakness and i < len(assessment.improvement_suggestions):
                    suggestion_idx = i
                    break

            if suggestion_idx is not None:
                suggestion = assessment.improvement_suggestions[suggestion_idx]

                # Check if a similar insight already exists
                existing_insights = await db.get_active_insights(limit=50)
                already_exists = False
                for ins in existing_insights:
                    if suggestion[:50] in ins.get("content", ""):
                        already_exists = True
                        # Reinforce existing
                        await db.update_insight_effectiveness(
                            ins["id"],
                            ins.get("effectiveness", 0.0) + 0.05,
                        )
                        break

                if not already_exists:
                    insight = LearningInsight(
                        category="quality_improvement",
                        task_type=assessment.category,
                        content=f"Quality improvement: {suggestion}",
                        confidence=0.5,
                        evidence_count=_weakness_tracker[norm]["count"],
                    )
                    await _store_insight_safely(insight)

                _weakness_tracker[norm]["addressed"] = True
                stored = True

    # Persist weakness tracker to database
    try:
        await db.save_weakness_tracker(_weakness_tracker)
    except Exception:
        pass

    return stored


async def run_improvement_pipeline() -> dict:
    """Run the full improvement pipeline: assess, learn, validate, promote.

    This is called periodically (every N improvement cycles):
    1. Check for new weakness patterns in the tracker
    2. Check if previously stored improvements have been validated by outcomes
    3. Promote validated improvements to active status
    4. Archive improvements that haven't shown effect

    Returns:
        dict with counts of: assessed, learned, validated, promoted, archived
    """
    await _ensure_weakness_tracker_loaded()
    result = {
        "assessed": 0,
        "learned": 0,
        "validated": 0,
        "promoted": 0,
        "archived": 0,
    }

    try:
        # 1. Check weakness patterns that haven't been addressed yet
        for norm, tracker in _weakness_tracker.items():
            if tracker["count"] >= 3 and not tracker["addressed"]:
                # Generate a generic improvement suggestion
                suggestion = f"Address recurring weakness: {norm[:100]}"
                insight = LearningInsight(
                    category="quality_improvement",
                    task_type="general",
                    content=f"Quality improvement: {suggestion}",
                    confidence=0.4,
                    evidence_count=tracker["count"],
                )
                await _store_insight_safely(insight)
                tracker["addressed"] = True
                result["learned"] += 1

        # 2. Validate existing improvements against outcome data
        active_insights = await db.get_active_insights(limit=_MAX_ACTIVE_INSIGHTS)
        for insight in active_insights:
            if insight.get("category") != "quality_improvement":
                continue

            evidence = insight.get("evidence_count", 1)
            confidence = insight.get("confidence", 0.5)
            status = insight.get("status", "proposed")

            # Check if this improvement has enough evidence to validate
            if status == "proposed" and evidence >= 5 and confidence >= 0.6:
                await db.update_insight_effectiveness(
                    insight["id"],
                    insight.get("effectiveness", 0.0) + 0.1,
                )
                result["validated"] += 1

            # Promote validated improvements to active
            if status == "testing" and evidence >= 8 and confidence >= 0.75:
                result["promoted"] += 1

            # Archive low-confidence improvements with sufficient evidence
            if evidence >= 10 and confidence < 0.3:
                await db.update_insight_effectiveness(insight["id"], -0.5)
                result["archived"] += 1

    except Exception as e:
        logger.error("Improvement pipeline error: %s", e)

    return result


async def reset_weakness_tracker() -> None:
    """Reset the weakness tracker (e.g., for testing or after major fixes)."""
    _weakness_tracker.clear()
    try:
        await db.save_weakness_tracker({})
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ PUBLIC API — Functions called by other modules ═════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

async def get_full_learning_context(task_type: str, user_query: str = "") -> str:
    """Get the complete learning context for a task.

    Combines all learning subsystems into a single formatted string
    that can be injected into the agent's context.

    Args:
        task_type: The classified task type.
        user_query: The user's original query (for trajectory retrieval).

    Returns:
        Formatted string with all relevant learning context.
    """
    parts = []

    # 1. Meta-learning insights
    meta = await get_meta_learning_context(task_type)
    if meta:
        parts.append(meta)

    # 2. Strategy injections
    strategies = await get_strategy_injections(task_type)
    if strategies:
        parts.append(strategies)

    # 3. Tool suggestions
    tool_suggestions = await get_tool_suggestions(task_type)
    if tool_suggestions:
        parts.append(tool_suggestions)

    # 4. Quality feedback
    quality = await get_quality_feedback_context()
    if quality:
        parts.append(quality)

    # 5. Experience replay (similar past trajectories)
    if user_query:
        trajectories = await retrieve_similar_trajectories(user_query, k=2)
        if trajectories:
            traj_text = format_trajectory_for_prompt(trajectories)
            if traj_text:
                parts.append(traj_text)

    if not parts:
        return ""

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Procedural Skill Distillation (v19)
# ═══════════════════════════════════════════════════════════════════════════════
#
# After a successful task trajectory, distil the key action sequence into a
# reusable procedural memory.  Future tasks with similar trigger patterns can
# skip the planning loop entirely and execute the cached procedure directly.
#
# Inspired by cognitive architecture "muscle memory":
# repeated successful patterns become fast, automatic, low-cost to invoke.
# ═══════════════════════════════════════════════════════════════════════════════

_PROCEDURAL_MIN_STEPS = 2        # don't distil trivial single-step tasks
_PROCEDURAL_MIN_REWARD = 0.60    # only distil genuinely successful runs
_PROCEDURAL_SIMILARITY_THRESHOLD = 0.55  # merge threshold for trigger patterns


async def distill_procedural_skill(
    task: str,
    steps: list[dict],
    reward: float,
    session_id: str = "",
) -> bool:
    """Distil a successful trajectory into a reusable procedural skill memory.

    Called by the engine after a task completes with a positive reward.
    If the trajectory is long and successful enough, the key action sequence
    is condensed and stored as a ``procedural`` memory with high importance.

    Args:
        task:       The original user task string.
        steps:      List of executed step dicts (each has ``step``, ``output``,
                    ``status`` keys from the engine).
        reward:     Quality score for this trajectory in [0.0, 1.0].
        session_id: Session that produced the trajectory.

    Returns:
        ``True`` if a skill was stored or merged, ``False`` otherwise.
    """
    if reward < _PROCEDURAL_MIN_REWARD:
        return False

    completed = [s for s in steps if s.get("status") == "done"]
    if len(completed) < _PROCEDURAL_MIN_STEPS:
        return False

    action_sequence = []
    for i, step in enumerate(completed, 1):
        action = step.get("step", "").strip()
        if action:
            action_sequence.append(f"{i}. {action}")

    if not action_sequence:
        return False

    skill_content = (
        f"PROCEDURAL SKILL — trigger: {task[:120]}\n"
        f"Steps:\n" + "\n".join(action_sequence)
    )

    try:
        from ..memory.memory import MemoryManager
        mm = MemoryManager(session_id=session_id or "self_improve")

        # If a similar skill already exists, reinforce it instead of duplicating
        existing = await mm.dedup_check(skill_content,
                                        threshold=_PROCEDURAL_SIMILARITY_THRESHOLD)
        if existing is not None:
            from ..memory import database as _db
            new_imp = min(existing.importance + 0.05, 1.0)
            await _db.update_memory_importance(existing.id, new_imp)
            logger.info(
                "[self_improve] Procedural skill reinforced (id=%d, task=%.60s)",
                existing.id, task,
            )
            return True

        importance = min(0.70 + 0.25 * reward, 0.95)
        mem_id = await mm.remember(
            content=skill_content,
            kind="procedural",
            importance=importance,
            tags=["procedural_skill", "auto_distilled"],
        )
        if mem_id and mem_id > 0:
            logger.info(
                "[self_improve] New procedural skill stored "
                "(id=%d, steps=%d, reward=%.2f, task=%.60s)",
                mem_id, len(completed), reward, task,
            )
            return True
    except Exception as exc:
        logger.warning("[self_improve] Procedural skill distillation failed: %s", exc)

    return False


async def get_procedural_skill_context(task: str, limit: int = 3) -> str:
    """Retrieve the most relevant procedural skills for a task.

    Returns a formatted string for injection into the agent prompt,
    or an empty string if no relevant skills are found.

    Args:
        task:  The current user task.
        limit: Maximum number of skills to return.
    """
    try:
        from ..memory.memory import MemoryManager
        mm = MemoryManager(session_id="skill_lookup")
        mems = await mm.recall(task, k=limit, kind="procedural")
        if not mems:
            return ""
        lines = ["[Procedural Skills — reusable patterns from past successes]"]
        for mem in mems:
            lines.append(mem.content[:400])
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("[self_improve] get_procedural_skill_context failed: %s", exc)
        return ""

