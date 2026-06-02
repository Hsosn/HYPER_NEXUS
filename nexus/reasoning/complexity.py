"""
Complexity assessment for the Nexus reasoning engine.
Determines task complexity (high/medium/low) using structural signals.
"""
from __future__ import annotations
import re
import logging
logger = logging.getLogger(__name__)

from .tool_parsing import _SIMPLE_NUMBERED_LIST_RE


_HIGH_COMPLEXITY_INTENTS = frozenset({


    "research and write", "investigate and report", "analyze and create",


    "plan and build", "design and implement", "build a complete",


    "create a comprehensive", "develop a full",


    "from scratch", "end to end", "end-to-end",


    "multi-step", "multi step", "full stack", "full-stack",


    "deep dive", "deep analysis", "thorough analysis",


})


# Action verbs that suggest complex, multi-phase work when ≥3 appear in a task.

_HIGH_COMPLEXITY_VERBS = frozenset({

    "architect", "orchestrate", "coordinate", "synthesize",

    "integrate", "deploy", "migrate", "refactor",

    "optimize", "containerize", "provision", "configure",

    "automate", "scaffold", "compose", "establish",

})


# Connectors that join separate sub-tasks  - their presence means the user


# is asking for multiple things that likely need different tools or phases.


_MULTI_TASK_CONNECTORS = (


    " and then ", " after that ", " as well as ", " along with ",


    " in addition to ", " followed by ", " and also ",


    " first ", " then ", " finally ", " secondly ", " lastly ",


    " next, ", " after which ",


)

# Words that indicate design/creative/coordination work requiring non-trivial reasoning
# even without explicit question marks or connectors.
_MEDIUM_COMPLEXITY_DESIGN = frozenset({
    "design", "build", "create", "coordinate", "strategy", "architect",
    "architecture", "compose", "composing", "compose a", "compose an",
    "draft", "outline", "plan", "compare", "contrast", "evaluate",
    "best", "optimal", "efficient",
})


def _assess_complexity(task: str) -> str:


    """Assess task complexity using structural and intent signals.


    Unlike keyword counting, this analyzes the *shape* of the request:


    - How many separate sub-tasks are implied (connectors, list markers)?


    - How long is the message (longer ≈ more detail ≈ more work)?


    - Does it contain explicit high-complexity intent phrases?


    - Does it ask for multiple distinct operations?


    Returns 'high', 'medium', or 'low'.


    """


    t = task.strip()


    low = t.lower()


    words = t.split()


    word_count = len(words)


    #  -  -  High complexity: structural multi-task signals  -  - 


    # Explicit multi-step intent phrases (word-boundary matching to avoid false partial matches)
    if any(re.search(r'\b' + re.escape(intent) + r'\b', low) for intent in _HIGH_COMPLEXITY_INTENTS):
        return "high"

    # High-complexity verb count: unconventionally-worded complex tasks often use ≥3 action verbs
    _verbs_found = sum(1 for v in _HIGH_COMPLEXITY_VERBS if re.search(r'\b' + re.escape(v) + r'\b', low))
    if _verbs_found >= 3:
        return "high"
    if _verbs_found == 2 and word_count > 30:
        return "high"

    # Multiple sub-tasks joined by connectors


    connector_count = sum(1 for c in _MULTI_TASK_CONNECTORS if c in low)


    if connector_count >= 2:


        return "high"


    # Numbered steps (1. 2. 3.) or bullet patterns indicate structured multi-step


    numbered_steps = len(_SIMPLE_NUMBERED_LIST_RE.findall(low))


    if numbered_steps >= 3:


        return "high"


    # Very long messages with questions  - usually detailed requests


    if word_count > 80 and "?" in t:


        return "high"


    #  -  -  Medium complexity: single non-trivial task  -  - 


    # One connector = two sub-tasks


    if connector_count == 1:


        return "medium"


    # Question about how/why  - needs reasoning, not just tool call


    has_deep_question = any(


        low.startswith(q) or f" {q}" in low


        for q in ("how to", "how do", "how can", "why does", "why is",


                   "what would", "what if", "which approach", "what's the best")


    )


    if has_deep_question:


        return "medium"


    # Moderate length with a question mark


    if word_count > 20 and "?" in t:


        return "medium"


    # Two numbered steps


    if numbered_steps == 2:


        return "medium"


    # Design/creative/coordination words indicate non-trivial reasoning work
    task_words = set(low.split())
    if task_words & _MEDIUM_COMPLEXITY_DESIGN:
        return "medium"

    #  -  -  Low complexity: single direct action  -  - 


    # Short, direct requests like "search for X", "run command Y"

    return "low"