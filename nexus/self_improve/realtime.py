"""Real-time failure learning — called immediately on tool failure."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("nexus.self_improve.realtime")


def _generate_quick_fix(tool_name: str, error_message: str, args: dict) -> str:
    """Generate an instant heuristic fix without LLM call.

    Uses pattern matching on common error types to provide immediate workarounds.
    """
    error_low = error_message.lower()

    # Pattern 1: File not found → suggest checking path
    if "not found" in error_low or "no such file" in error_low or "file does not exist" in error_low:
        return f"Check file path exists before calling {tool_name}. Use file_list to verify directory contents first."

    # Pattern 2: Permission denied
    if "permission" in error_low or "access denied" in error_low or "forbidden" in error_low:
        return f"Permission issue with {tool_name}. Check credentials or use shell_run with appropriate permissions."

    # Pattern 3: Timeout
    if "timeout" in error_low or "timed out" in error_low:
        return f"Timeout in {tool_name}. Try reducing scope (smaller limit/max_results) or retrying."

    # Pattern 4: Authentication
    if "auth" in error_low or "unauthorized" in error_low or "401" in error_low or "403" in error_low:
        return f"Authentication failed for {tool_name}. Check API key/token configuration in Settings."

    # Pattern 5: Rate limit
    if "rate limit" in error_low or "too many requests" in error_low or "429" in error_low:
        return f"Rate limited by {tool_name}. Wait a moment before retrying, or reduce request frequency."

    # Pattern 6: Connection error
    if "connection" in error_low or "network" in error_low or "dns" in error_low:
        return f"Network error in {tool_name}. Check internet connectivity and retry."

    # Pattern 7: JSON parse error
    if "json" in error_low or "parse error" in error_low:
        return f"Data format error in {tool_name}. The response wasn't valid JSON. Try with different parameters."

    # Pattern 8: Module not found
    if "module" in error_low and "not found" in error_low:
        return f"Missing dependency for {tool_name}. Use shell_run to install required package first."

    # Pattern 9: Invalid arguments — give tool-specific alias suggestions
    if "invalid" in error_low or "required" in error_low or "missing" in error_low:
        _TOOL_PARAM_ALIASES = {
            "file_write":   ("Use 'path' for the file path (also: filepath, filename, dest). "
                             "Use 'content' for the data (also: data, text, html, code). "
                             "Example: file_write(path='file.txt', content='hello')"),
            "file_read":    "Use 'path' for the file path (also: file, filename). Example: file_read(path='file.txt')",
            "file_list":    "Use 'path' for the directory (also: dir, directory). Example: file_list(path='.')",
            "python_exec":  "Use 'code' for the Python source (also: script, python_code). Example: python_exec(code='print(1)')",
            "shell_run":    "Use 'command' for the shell command (also: cmd, shell_cmd). Example: shell_run(command='ls')",
            "web_search":   "Use 'query' for the search term (also: q, search). Example: web_search(query='python tutorials')",
        }
        specific = _TOOL_PARAM_ALIASES.get(tool_name)
        if specific:
            return f"Invalid arguments for {tool_name}. {specific}"
        param_hint = ", ".join(list(args.keys())[:3]) if args else "N/A"
        return f"Invalid arguments for {tool_name}. Check required parameters. Provided: {param_hint}."

    return ""


async def learn_from_failure_realtime(
    tool_name: str,
    error_message: str,
    args: dict,
    session_id: str = "",
) -> str | None:
    """Real-time failure learning — called immediately on tool failure.
    
    Returns the quick-fix strategy string if generated, or None.
    """
    from . import _ensure_schema, _store_insight_safely, _generate_strategy, LearningInsight
    from ..events import emit
    from ..memory import db

    try:
        logger.info("[learn] Real-time failure learning triggered: tool=%s error=%.80s", tool_name, error_message)

        try:
            await _ensure_schema()
        except Exception:
            pass
        await db.log_improvement(
            category="realtime_tool_failure",
            tool_name=tool_name,
            detail=f"Tool '{tool_name}' failed: {error_message[:300]}",
        )

        existing_strategy = await db.get_tool_strategy(tool_name)
        if existing_strategy:
            await emit("known_workaround_available",
                       tool=tool_name,
                       strategy=existing_strategy[:100],
                       message=f"Known workaround exists for '{tool_name}': {existing_strategy[:80]}")
            return existing_strategy

        quick_fix = _generate_quick_fix(tool_name, error_message, args)

        if quick_fix:
            await db.update_tool_strategy(tool_name, quick_fix)
            try:
                from ..memory.memory import embed
                embedding = await embed(f"Tool fix: {tool_name} - {quick_fix}")
            except Exception:
                embedding = None

            await db.add_memory(
                kind="procedural",
                content=f"[Realtime Fix] When {tool_name} fails with '{error_message[:100]}': {quick_fix}",
                importance=0.80,
                embedding=embedding,
                tags=["self-learning", "realtime_fix", tool_name],
            )

            insight = LearningInsight(
                category="tool_strategy",
                task_type="general",
                content=f"Realtime fix for '{tool_name}': {quick_fix}",
                confidence=0.7,
                evidence_count=1,
            )
            await _store_insight_safely(insight)

            await emit("realtime_fix_applied",
                       tool=tool_name,
                       fix=quick_fix,
                       message=f"Real-time fix applied for '{tool_name}': {quick_fix[:80]}")

        if not quick_fix:
            llm_fix = await _generate_strategy(
                f"Tool '{tool_name}' failed: {error_message[:200]}. Args: {args}",
                tool_name,
            )
            if llm_fix:
                await db.update_tool_strategy(tool_name, llm_fix)
                try:
                    from ..memory.memory import embed
                    llm_embedding = await embed(f"Tool fix: {tool_name} - {llm_fix}")
                except Exception:
                    llm_embedding = None
                await db.add_memory(
                    kind="procedural",
                    content=f"[Realtime Fix] When {tool_name} fails with '{error_message[:100]}': {llm_fix}",
                    importance=0.80,
                    embedding=llm_embedding,
                    tags=["self-learning", "realtime_fix", tool_name, "llm_generated"],
                )

                llm_insight = LearningInsight(
                    category="tool_strategy",
                    task_type="general",
                    content=f"Realtime fix for '{tool_name}': {llm_fix}",
                    confidence=0.75,
                    evidence_count=1,
                )
                await _store_insight_safely(llm_insight)

                await emit("realtime_fix_applied",
                           tool=tool_name,
                           fix=llm_fix,
                           message=f"LLM-generated fix for '{tool_name}': {llm_fix[:80]}")
                return llm_fix

        return quick_fix

    except Exception as e:
        logger.error("Realtime failure learning error: %s", e)
        return None
