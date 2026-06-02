"""Tool outcome tracking and success prediction for self-supervised learning."""
from __future__ import annotations

import time
import logging
from typing import Any

logger = logging.getLogger("nexus.self_improve.predict")

_OUTCOME_MEMORY_MAX = 200
_outcome_memory: dict[str, list[dict]] = {}


async def record_outcome(
    tool_name: str,
    args: dict,
    success: bool,
    duration_ms: float,
    output_preview: str = "",
) -> None:
    """Record a tool execution outcome for self-supervised learning."""
    if tool_name not in _outcome_memory:
        _outcome_memory[tool_name] = []
    _outcome_memory[tool_name].append({
        "args_preview": str(args)[:100],
        "success": success,
        "duration_ms": duration_ms,
        "timestamp": time.time(),
    })
    if len(_outcome_memory[tool_name]) > _OUTCOME_MEMORY_MAX:
        _outcome_memory[tool_name] = _outcome_memory[tool_name][-_OUTCOME_MEMORY_MAX:]

    try:
        from ..memory import db
        await db.log_improvement(
            category="tool_failure" if not success else "self_supervised_outcome",
            tool_name=tool_name,
            detail=(
                f"{'SUCCESS' if success else 'FAILURE'}: {tool_name} "
                f"completed in {duration_ms:.0f}ms. "
                f"Args: {str(args)[:100]}. "
                f"Output: {output_preview[:100]}"
            ),
        )
    except Exception:
        pass


async def predict_tool_success(tool_name: str, args: dict) -> dict:
    """Predict whether a tool call will succeed based on past outcomes.

    Returns dict with predicted_success, confidence, success_probability, advice.
    """
    _ALIAS_SAFE_TOOLS = {"file_write", "file_read", "file_list", "python_exec", "shell_run"}
    if tool_name in _ALIAS_SAFE_TOOLS:
        return {"predicted_success": True, "confidence": 1.0,
                "similar_successes": 0, "similar_failures": 0,
                "success_probability": 0.9, "advice": ""}

    result = {
        "predicted_success": True,
        "success_probability": 0.5,
        "confidence": 0.0,
        "similar_successes": 0,
        "similar_failures": 0,
        "advice": "",
    }

    outcomes = _outcome_memory.get(tool_name, [])
    if not outcomes:
        return result

    arg_keys = set(args.keys())
    similar_ok = 0
    similar_fail = 0

    for out in outcomes:
        out_args_str = out.get("args_preview", "")
        try:
            if all(k in out_args_str for k in list(arg_keys)[:3]):
                if out["success"]:
                    similar_ok += 1
                else:
                    similar_fail += 1
        except Exception:
            pass

    total_similar = similar_ok + similar_fail
    if total_similar == 0:
        return result

    success_rate = similar_ok / total_similar
    result["similar_successes"] = similar_ok
    result["similar_failures"] = similar_fail
    result["predicted_success"] = success_rate >= 0.5
    result["success_probability"] = max(success_rate, 0.4)
    result["confidence"] = min(1.0, total_similar * 0.1 + success_rate * 0.5)

    if success_rate < 0.4 and total_similar >= 3:
        result["advice"] = (
            f"{tool_name} has a {1-success_rate:.0%} failure rate in similar calls. "
            f"Consider checking parameters or retrying with a simpler approach."
        )
    elif success_rate > 0.9 and total_similar >= 3:
        result["advice"] = f"{tool_name} has a strong track record ({success_rate:.0%} success)"

    return result


async def extract_success_pattern(tool_name: str) -> str | None:
    """Extract a reusable success pattern from past outcomes."""
    outcomes = _outcome_memory.get(tool_name, [])
    if len(outcomes) < 5:
        return None

    successes = [o for o in outcomes if o["success"]]
    failures = [o for o in outcomes if not o["success"]]

    if not failures:
        return None

    success_rate = len(successes) / len(outcomes)
    if success_rate > 0.95:
        return None

    if len(failures) >= 3:
        return (
            f"[Self-Learned] {tool_name} success rate: {success_rate:.0%} "
            f"({len(successes)} ok, {len(failures)} fail). "
            f"Consider reducing complexity of parameters for better reliability."
        )

    return None
