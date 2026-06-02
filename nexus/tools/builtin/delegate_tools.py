"""Delegate tools - spawn sub-agents for parallel task execution.

Optimizations (v2):
  - BaseException catch (CancelledError in Python 3.9+) to prevent orphaned sessions
  - return_exceptions=True so one task failure doesn't cancel the entire batch
  - Timeout alignment: per-task 300s matches batch timeout; batch timeout increased to 600s
    to account for semaphore(5) queuing delays with large batches
  - Lightweight sub-agent fast path: sub-agents skip memory loading, current-events fetch,
    self-awareness construction, and other overhead (already in engine.py via _is_sub_agent)
  - Focused system prompt for file-creation tasks (minimize LLM round-trips)
  - Structured batch results with per-task status and error isolation
  - Batch-level retry via configurable _BATCH_RETRY_LIMIT
"""
from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid

from ..registry import tool

logger = logging.getLogger(__name__)

# Track active sub-agent sessions for cleanup
_active_sub_agents: set[str] = set()
_SUB_AGENT_SEMAPHORE = asyncio.Semaphore(5)  # Max 5 concurrent sub-agents

# Timeouts: per-task timeout (120s) prevents a single stuck sub-agent from
# consuming too much of the batch budget.  Batch timeout is 360s to
# accommodate N tasks flowing through the semaphore(5) sequentially.
_DELEGATE_BATCH_TIMEOUT = 1800.0
_DELEGATE_TASK_TIMEOUT = 600.0
_BATCH_RETRY_LIMIT = 1  # Number of times to retry failed tasks within a batch


def _is_low_quality_response(response: str | None) -> bool:
    """Check if a sub-agent response is too brief or uninformative."""
    stripped = response.strip() if response else ""
    if len(stripped) < 20:
        return True
    lower = stripped.lower()
    if lower.startswith(("error:", "failed:", "unable to", "could not", "i don't know", "i cannot", "i'm not sure")):
        return True
    return False


def active_sub_agent_count() -> int:
    return len(_active_sub_agents)


def active_sub_agent_ids() -> list[str]:
    return list(_active_sub_agents)


async def _ensure_db() -> bool:
    try:
        from ...memory import db
        if not hasattr(db, '_db') or db._db is None:
            await db.init()
        return True
    except Exception:
        return False


async def _cleanup_sub_agent(session_id: str) -> None:
    try:
        from ...memory import db
        if hasattr(db, 'delete_session') and callable(db.delete_session):
            await db.delete_session(session_id)
    except Exception:
        pass
    _active_sub_agents.discard(session_id)


# ── Focused system prompt for sub-agents ────────────────────────────────────
# Kept minimal to reduce token consumption and LLM round-trips.
_SUB_AGENT_SYSTEM_PROMPT = """You are a sub-agent completing a delegated task.

CRITICAL RULES:
1. Complete the task in as FEW tool calls as possible
2. For file creation: use file_write directly, then stop - no verification needed
3. Be decisive - don't ask for permission, don't over-analyze, just do it
4. Return a concise summary of what you did and the result

Available tools: file_read, file_write, shell_run, python_exec, web_search, web_fetch

TOOL GUIDANCE:
- file_read / file_write / python_exec are the most reliable tools
- shell_run may fail with 'Shell error' on some platforms — prefer python_exec for running commands
- If a tool fails with the same error twice, switch to a different approach immediately
- file_list may return empty for non-existent paths — always check if a file exists with file_read first"""

# Build platform-aware system prompt for sub-agents
def _build_sub_agent_prompt(task: str, context: str = "") -> str:
    """Build a sub-agent system prompt with platform-aware guidance."""
    import platform
    _system = platform.system().lower()
    _guidance = ""
    if _system == "windows":
        _guidance = (
            "\n\nPLATFORM NOTES (Windows):\n"
            "- shell_run may fail because bash is not natively available\n"
            "- Use python_exec with os/path/shutil modules instead of shell commands\n"
            "- File paths use backslashes (\\\\) — prefer forward slashes (/) which also work\n"
            "- **Do NOT use `&&` or `||`** in shell commands — those are bash operators, not supported on Windows.\n"
            "  Use python_exec with sequential calls or `; ` (semicolon space) instead.\n"
            "- python_exec is the most reliable way to inspect or modify the filesystem"
        )
    prompt = _SUB_AGENT_SYSTEM_PROMPT + _guidance + f"""

CONTEXT:
{context}

TASK:
{task}

Report your result concisely."""
    return prompt


@tool(
    name="delegate_task",
    description="Delegate a subtask to a sub-agent for parallel execution. "
                "Use this when a task requires creating multiple files or doing "
                "independent work that can run in parallel. The sub-agent will "
                "complete the task and return results.",
    parameters_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "What the sub-agent should do"},
            "context": {"type": "string", "description": "Relevant context/files for the task"},
            "workspace": {"type": "string", "description": "Workspace folder to work in"},
        },
        "required": ["task"],
    },
    category="agent",
    timeout=_DELEGATE_TASK_TIMEOUT + 30,
)
async def delegate_task(params):
    from ...reasoning.engine import ReasoningEngine
    from ...memory import db

    await _ensure_db()

    task = params.get("task", "").strip()
    context = params.get("context", "")
    workspace = params.get("workspace", "default")

    if not task:
        return "Error: task description required"

    sub_agent_id = f"delegate_{uuid.uuid4().hex[:8]}"
    _active_sub_agents.add(sub_agent_id)

    async with _SUB_AGENT_SEMAPHORE:
        try:
            await db.create_session(sub_agent_id, "delegate")

            system_context = _build_sub_agent_prompt(task, context)

            await db.add_message(sub_agent_id, "system", system_context)

            engine = ReasoningEngine(sub_agent_id, is_sub_agent=True)

            response = await asyncio.wait_for(
                engine.respond(task),
                timeout=_DELEGATE_TASK_TIMEOUT
            )

            await db.add_message(sub_agent_id, "assistant", response)

            # Inline cleanup (avoid fire-and-forget task that might never run)
            await _cleanup_sub_agent(sub_agent_id)

            # vFIX: Quality gate — detect sub-agents that returned minimal output
            if _is_low_quality_response(response):
                _stripped = response.strip() if response else ""
                logger.warning("Sub-agent %s returned low-quality response (%d chars): %s",
                               sub_agent_id[:8], len(_stripped), _stripped[:100])
                return f"[SUB-AGENT {sub_agent_id[:8]} LOW-QUALITY] Response too brief or uninformative ({len(_stripped)} chars): {_stripped[:200]}"

            return f"[SUB-AGENT {sub_agent_id[:8]} COMPLETED]\n\n{response[:2000]}"

        except asyncio.CancelledError:
            await _cleanup_sub_agent(sub_agent_id)
            raise
        except asyncio.TimeoutError:
            await _cleanup_sub_agent(sub_agent_id)
            return f"[SUB-AGENT {sub_agent_id[:8]} TIMEOUT] Task took too long"
        except BaseException as e:
            await _cleanup_sub_agent(sub_agent_id)
            tb = traceback.format_exc()[:200]
            logger.warning("Sub-agent %s failed: %s | %s", sub_agent_id[:8], e, tb)
            return f"[SUB-AGENT {sub_agent_id[:8]} ERROR] {str(e)}"


@tool(
    name="delegate_batch",
    description="Delegate MULTIPLE independent subtasks to sub-agents in PARALLEL. "
                "Use when you need to create multiple files or do parallel work. "
                "Each task runs concurrently. Maximum 5 concurrent sub-agents. "
                "Batch timeout: 600s. Per-task timeout: 300s.",
    parameters_schema={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "task": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["name", "task"]
                },
                "description": "List of tasks to delegate in parallel"
            },
        },
        "required": ["tasks"],
    },
    category="agent",
    timeout=_DELEGATE_BATCH_TIMEOUT + 30,
)
async def delegate_batch(params):
    from ...reasoning.engine import ReasoningEngine
    from ...memory import db

    await _ensure_db()

    tasks = params.get("tasks", [])

    # Issue #1 fix: normalize nested array parameter if it arrives as a string
    if isinstance(tasks, str):
        try:
            parsed = json.loads(tasks)
            if not isinstance(parsed, list):
                return "Error: tasks parameter must be a JSON array"
            tasks = parsed
        except (json.JSONDecodeError, TypeError):
            return "Error: tasks parameter is malformed — expected an array"

    if not tasks:
        return "Error: at least one task required"

    # Build the list of task descriptors upfront
    task_meta = [
        {
            "name": t.get("name", f"task{i}"),
            "task": t.get("task", ""),
            "context": t.get("context", ""),
        }
        for i, t in enumerate(tasks)
    ]

    # Issue #2 fix: pre-create all DB sessions SEQUENTIALLY to avoid
    # connection-pool races when asyncio.gather fires them concurrently.
    # Each entry carries its pre-allocated sub_id so run_single doesn't
    # need to call create_session.
    for m in task_meta:
        sub_id = f"batch_{m['name']}_{uuid.uuid4().hex[:6]}"
        m["_sub_id"] = sub_id
        _active_sub_agents.add(sub_id)
        try:
            await db.create_session(sub_id, "delegate")
        except Exception:
            pass  # session may already exist (ON CONFLICT DO NOTHING)

    async def run_single(meta: dict):
        name = meta["name"]
        task_text = meta["task"]
        context = meta["context"]
        sub_id = meta["_sub_id"]
        async with _SUB_AGENT_SEMAPHORE:
            try:
                system_msg = _build_sub_agent_prompt(task_text, context)
                await db.add_message(sub_id, "system", system_msg)
                engine = ReasoningEngine(sub_id, is_sub_agent=True)
                response = await asyncio.wait_for(engine.respond(task_text), timeout=_DELEGATE_TASK_TIMEOUT)
                await _cleanup_sub_agent(sub_id)
                # vFIX: Quality gate for batch sub-agents
                if _is_low_quality_response(response):
                    _stripped = response.strip() if response else ""
                    logger.warning("Batch sub-agent '%s' returned low-quality response (%d chars)", name, len(_stripped))
                    return {"name": name, "status": "low_quality", "result": f"Low-quality response ({len(_stripped)} chars): {_stripped[:200]}"}
                return {"name": name, "status": "ok", "result": response[:1000]}
            except asyncio.TimeoutError:
                await _cleanup_sub_agent(sub_id)
                return {"name": name, "status": "timeout", "result": "Task took too long"}
            except BaseException as e:
                await _cleanup_sub_agent(sub_id)
                logger.warning("Batch sub-agent %s failed: %s", name, e)
                return {"name": name, "status": "error", "result": str(e)[:200]}

    all_results: list = []

    for attempt in range(_BATCH_RETRY_LIMIT + 1):
        try:
            raw_results = await asyncio.wait_for(
                asyncio.gather(
                    *[run_single(m) for m in task_meta],
                    return_exceptions=True,
                ),
                timeout=_DELEGATE_BATCH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return f"[BATCH TIMEOUT] Entire batch exceeded {_DELEGATE_BATCH_TIMEOUT}s timeout"
        except asyncio.CancelledError:
            # Clean up sessions before re-raising
            for meta in task_meta:
                await _cleanup_sub_agent(meta["_sub_id"])
            raise

        # Process results: filter out BaseException instances from return_exceptions=True
        failures = []
        for i, r in enumerate(raw_results):
            task_meta_entry = task_meta[i]
            task_name = task_meta_entry["name"]
            if isinstance(r, BaseException):
                logger.warning("Batch sub-agent %s raised unhandled: %s", task_name, r)
                failures.append(task_meta_entry)
                all_results.append({"name": task_name, "status": "error", "result": str(r)[:200]})
            elif isinstance(r, dict) and r.get("status") in ("timeout", "error"):
                failures.append(task_meta_entry)
                all_results.append(r)
            else:
                all_results.append(r)

        # If no failures or retries exhausted, return accumulated results
        if not failures or attempt >= _BATCH_RETRY_LIMIT:
            lines = []
            for r in all_results:
                if isinstance(r, dict):
                    status_icon = {"ok": "✓", "timeout": "⏱", "error": "✗", "low_quality": "⚠"}.get(r.get("status", ""), "?")
                    lines.append(f"[{status_icon} {r['name']}]: {r['result']}")
                else:
                    lines.append(str(r)[:1000])
            return "\n\n".join(lines)

        # Remove old error entries so retries can replace them
        failed_names = {f["name"] for f in failures}
        all_results = [r for r in all_results if not (isinstance(r, dict) and r.get("name") in failed_names)]

        # Retry failed tasks only
        logger.info("Retrying %d failed sub-agents (attempt %d/%d)", len(failures), attempt + 1, _BATCH_RETRY_LIMIT)
        task_meta = failures
        # Re-create sessions that were deleted by _cleanup_sub_agent on failure
        for m in task_meta:
            try:
                await db.create_session(m["_sub_id"], "delegate")
            except Exception:
                pass

    return "\n\n".join(
        f"[✗ {r['name']}] {r['result']}" if isinstance(r, dict) else str(r)[:1000]
        for r in all_results
    )