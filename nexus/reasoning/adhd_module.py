"""
ADHD Cross-Domain Reasoning Module
===================================

Boosts agent creativity by mimicking the ADHD brain's superpower:
hyper-connecting seemingly unrelated knowledge domains to find
non-obvious solutions.

How it works:
  1. Strip the task to its structural problem type (search? optimization? coordination?)
  2. Fire that structure across 8 knowledge domains simultaneously
  3. Spawn N parallel reasoning streams seeded from different domains
  4. Cross-pollinate (bleed) the strongest stream into the weakest
  5. Lock onto any stream that exceeds the hyperfocus threshold
  6. Optionally inject a wild-card domain (serendipity) to escape local optima
  7. Collapse everything into one actionable synthesis

Adaptive learning:
  - Domain utility is tracked via exponential moving average
  - High-confidence domains get higher weight in future fires
  - Serendipity excludes recently-used domains to maximize diversity
  - Stream bleeding cross-pollinates actual reasoning content, not just confidence

Config: enable_adhd_reasoning (bool), adhd_complexity_min (str, default "medium"),
        adhd_use_llm (bool), adhd_cache_ttl (int, default 30),
        adhd_max_analogies (int, default 3)
"""

import asyncio
import math
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("nexus.adhd")


# =============================================================================
# DOMAIN KNOWLEDGE BASE — 8 domains × 8 problem types
# =============================================================================

DOMAIN_PATTERNS: dict[str, dict[str, str]] = {
    "biology": {
        "search":       "Evolution: explore many random variants in parallel, ruthlessly kill weak ones. The search space is sampled stochastically, not exhaustively.",
        "optimization": "Enzyme catalysis: don't brute-force the reaction. Find the binding site that lowers activation energy so the optimal path is also the easiest path.",
        "coordination": "Ant colony: no central controller. Each agent follows pheromone gradients and deposits its own. Global order emerges from local stigmergic rules.",
        "flow":         "Circulatory system: pressure gradient drives unidirectional flow. Valves prevent backflow. Capillaries slow flow at leaves to allow exchange.",
        "memory":       "Immune system: first exposure is slow (primary response). Every subsequent exposure is exponentially faster via memory B-cells. Recognition is pattern-based, not exact.",
        "pruning":      "Synaptic pruning: the brain removes 50% of synaptic connections in adolescence. Removing weak connections strengthens the signal-to-noise of important ones.",
        "parallelism":  "Brain hemispheres: two specialized processors running in parallel, syncing via corpus callosum only for integration, not constant coordination.",
        "deadlock":     "Apoptosis: when a cell is irreparably damaged, it executes a controlled self-destruct rather than lingering and corrupting neighbors. Fail fast, fail clean.",
    },
    "physics": {
        "search":       "Least action: nature doesn't try all paths — it finds the one where action is stationary. The optimal path is also the most probable path.",
        "optimization": "Simulated annealing: start hot (random, exploratory), cool slowly. Rapid cooling = local minimum. Gradual cooling = global minimum.",
        "coordination": "Resonance: systems synchronize when their natural frequencies align. Don't force synchronization — tune the frequencies instead.",
        "flow":         "Ohm's law: current = voltage / resistance. Bottleneck is always the highest resistance node. Reducing it anywhere else is waste.",
        "memory":       "Hysteresis: the current state depends on the history of inputs, not just the current input. The path matters as much as the destination.",
        "pruning":      "Phase transition: water doesn't gradually become ice. At a critical threshold the whole system snaps to a new configuration. Prune at the threshold.",
        "parallelism":  "Superposition: exist in multiple states simultaneously until forced to collapse by observation. Delay commitment as long as possible.",
        "deadlock":     "Entropy: closed systems always tend toward disorder unless energy is supplied. A stalled process needs an energy injection, not more waiting.",
    },
    "music": {
        "search":       "Jazz improvisation: explore freely but within harmonic constraints. The key signature is the constraint set. Violation of it is signal, not noise.",
        "optimization": "Counterpoint: find voice leading that resolves tension via minimal motion. The most elegant solution moves each voice as little as possible.",
        "coordination": "Ensemble timing: nobody watches a conductor exclusively. Each player listens to the whole and micro-adjusts. Coherence is emergent.",
        "flow":         "Tension and release: build dissonance (complexity), then resolve to tonic (simplicity). Every problem has a natural resolution if you build toward it.",
        "memory":       "Leitmotif: a recurring short theme that carries meaning across a 3-hour opera. Compress key state into a retrievable pattern, not a full replay.",
        "pruning":      "Silence: the notes you don't play define the ones you do. Remove everything that isn't load-bearing.",
        "parallelism":  "Polyphony: multiple independent melodic lines coexist without collapsing into each other. Independence + awareness = harmony, not cacophony.",
        "deadlock":     "Fermata: hold the note beyond its written duration, then release. Sometimes the solution is to pause everything and wait for resolution.",
    },
    "economics": {
        "search":       "Price discovery: markets aggregate distributed information into a single signal no central planner could compute. Let agents bid — the price is the answer.",
        "optimization": "Comparative advantage: don't do what you're best at in absolute terms. Do what you give up the least to do. Opportunity cost, not raw cost.",
        "coordination": "Mechanism design: you can't control what agents do, but you can design the incentive structure so their self-interest produces your desired outcome.",
        "flow":         "Theory of constraints: throughput of the whole system equals throughput at the bottleneck. Optimize the constraint, not the non-constraints.",
        "memory":       "Credit score: compress an agent's entire financial history into one number. Use the number to predict future behavior without replaying the history.",
        "pruning":      "Creative destruction: inefficient incumbents must die for better structures to emerge. Protecting them delays the inevitable and has compounding cost.",
        "parallelism":  "Division of labor: break the task into components each agent can specialize in. Total output vastly exceeds what any single generalist could produce.",
        "deadlock":     "Bank run: rational individual responses produce irrational collective outcome. Coordination failure — requires a credible commitment device to break.",
    },
    "architecture": {
        "search":       "Iterative design: generate many rough sketches before committing to structure. The first sketch is never the solution — it's the question.",
        "optimization": "Load path: the shortest, most direct route from force to ground. Every structural element should be on this path or removed.",
        "coordination": "Modular grid: standardized units enable infinite recombination. Constraint at the module level creates freedom at the system level.",
        "flow":         "Circulation: how people move through the space determines everything else. Design the path before the rooms. The path is the program.",
        "memory":       "Vernacular tradition: local buildings encode centuries of environmental adaptation without a single engineer. The form is the stored solution.",
        "pruning":      "Less is more: keep subtracting until the next removal breaks the thing. That's the minimum viable structure.",
        "parallelism":  "Mixed use: multiple programs in one building — office, retail, residential. Each serves different users simultaneously without interfering.",
        "deadlock":     "Temporary structure: when you can't build the permanent solution yet, build a scaffold. The scaffold enables the permanent without being it.",
    },
    "game_theory": {
        "search":       "Minimax: at every node, assume the adversary plays perfectly against you. Search the tree that maximizes your minimum guaranteed payoff.",
        "optimization": "Nash equilibrium: find the state where no single agent has unilateral incentive to deviate. That's the stable solution, even if it's not globally optimal.",
        "coordination": "Schelling point: when agents can't communicate, they converge on the most salient focal point. Salience is the coordination mechanism.",
        "flow":         "Information cascade: early public signals disproportionately influence all subsequent decisions, regardless of private information. Order matters.",
        "memory":       "Reputation: past defection changes future opponent strategies. A reputation for keeping commitments is a strategic asset.",
        "pruning":      "Iterated elimination of dominated strategies: remove any strategy that is never optimal. Repeat until no further removal is possible. What remains is solvable.",
        "parallelism":  "Simultaneous game: all players move at once without seeing others' choices. You must reason about distributions over actions, not responses to actions.",
        "deadlock":     "Chicken game: mutual destruction if neither yields. Pre-commit to not yielding (burn the steering wheel) to force the other to yield.",
    },
    "neuroscience": {
        "search":       "Hippocampal replay: during rest, the brain replays recent trajectories at high speed and simulates counterfactual paths. Exploration happens offline.",
        "optimization": "Hebbian learning: neurons that fire together wire together. Repeated co-activation strengthens the connection without explicit reward.",
        "coordination": "Default mode network: background process that integrates disparate inputs into a coherent narrative. Runs best when task-focused networks are idle.",
        "flow":         "Predictive coding: the brain predicts incoming input and only propagates the error (surprise). Most of the signal is suppressed because it was expected.",
        "memory":       "Consolidation: transfer from fast hippocampus to slow cortex during sleep. Working memory is volatile; long-term memory requires a consolidation pass.",
        "pruning":      "Lateral inhibition: activated neurons suppress their neighbors. Competition for activation is the pruning mechanism. Winner-take-all is common.",
        "parallelism":  "Dual processing: System 1 (fast, automatic, parallel) runs constantly beneath System 2 (slow, deliberate, serial). Most of cognition is System 1.",
        "deadlock":     "Cognitive dissonance: two incompatible beliefs cause psychological tension that must be resolved by updating one of them. Contradiction forces update.",
    },
    "military": {
        "search":       "Reconnaissance in force: commit small units to provoke enemy response, revealing dispositions you couldn't observe from afar.",
        "optimization": "Schwerpunkt: concentrate overwhelming force at the decisive point. Be weak everywhere else. Superiority at the point that matters.",
        "coordination": "OODA loop: Observe, Orient, Decide, Act. Win by completing your loop faster than the enemy, collapsing their ability to respond coherently.",
        "flow":         "Logistics: amateurs talk tactics, professionals talk logistics. The throughput of supply is the actual constraint on operational tempo.",
        "memory":       "After-action review: every operation produces a doctrine update. Institutional memory is the accumulated output of structured debriefs.",
        "pruning":      "Economy of force: the minimum force on secondary efforts, maximum on primary. Every resource on a non-decisive axis is waste.",
        "parallelism":  "Combined arms: armor, infantry, artillery, and air operating simultaneously in a mutually supporting way. Each covers the others' weaknesses.",
        "deadlock":     "Envelopment: when frontal assault stalls, go around the flanks. Attack the enemy's line of communication, not his front.",
    },
}

PROBLEM_TYPE_SIGNALS: dict[str, list[str]] = {
    "search":       ["find", "locate", "discover", "search", "explore", "identify", "detect", "scan", "look for", "determine", "where is"],
    "optimization": ["best", "optimal", "minimize", "maximize", "improve", "efficient", "fastest", "cheapest", "reduce", "increase", "trade-off", "balance", "design", "build", "create", "compose", "construct", "architect", "strategy", "approach", "framework", "architecture", "solution"],
    "coordination": ["coordinate", "synchronize", "align", "orchestrate", "schedule", "assign", "manage", "organize", "multi-agent", "distributed", "fair", "fairness", "collaborate", "consensus"],
    "flow":         ["pipeline", "process", "sequence", "order", "chain", "workflow", "route", "pipeline", "stage", "step by step", "then", "after"],
    "memory":       ["remember", "store", "retrieve", "cache", "persist", "recall", "history", "context", "state", "track", "log", "record"],
    "pruning":      ["filter", "remove", "eliminate", "select", "reduce", "trim", "prioritize", "rank", "clean", "simplify", "cut", "drop"],
    "parallelism":  ["parallel", "concurrent", "simultaneous", "multiple", "batch", "distribute", "split", "shard", "async", "fork", "scale"],
    "deadlock":     ["stuck", "blocked", "deadlock", "loop", "circular", "conflict", "race condition", "contention", "retry", "fail", "error"],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Analogy:
    domain: str
    problem_type: str
    pattern: str
    translation: str = ""
    relevance: float = 0.0


@dataclass
class Stream:
    stream_id: str
    domain: str
    problem_type: str
    seed_thought: str
    confidence: float = 0.0
    is_hyperfocused: bool = False


# =============================================================================
# ADHD-TOOL DOMAIN MAP — maps ADHD domains/problem_types to relevant tools
# =============================================================================
# Enables the ADHD module to suggest concrete tools based on cross-domain
# analogies, making ADHD reasoning directly actionable.

_ADHD_TOOL_DOMAIN_MAP: dict[str, dict[str, list[str]]] = {
    "biology": {
        "search":       ["web_search", "deep_research", "file_search", "github_list_repos"],
        "optimization": ["shell_run", "python_exec", "pt_hyperparameter_tune", "pt_model_convert"],
        "coordination": ["delegate_task", "delegate_batch", "schedule_task", "goal_create"],
        "flow":         ["file_read", "file_write", "shell_run", "git_status"],
        "memory":       ["recall", "recall_memories", "memory_set_permanent", "journal_search"],
        "pruning":      ["file_delete", "git_branch", "shell_reset", "settings_update"],
        "parallelism":  ["delegate_batch", "shell_run", "python_exec", "browser_open"],
        "deadlock":     ["shell_reset", "git_merge", "git_push", "vm_restart"],
    },
    "physics": {
        "search":       ["deep_research", "web_search", "fetch_url", "file_search"],
        "optimization": ["shell_run", "python_exec", "pt_hyperparameter_tune", "pt_model_convert"],
        "coordination": ["delegate_task", "goal_create", "schedule_task", "settings_update"],
        "flow":         ["git_log", "watch_url", "watch_path", "monitor_url"],
        "memory":       ["recall", "journal_search", "memory_set_permanent", "recall_memories"],
        "pruning":      ["file_delete", "shell_reset", "git_branch", "goal_update"],
        "parallelism":  ["delegate_batch", "shell_run", "python_exec", "vm_execute"],
        "deadlock":     ["shell_reset", "git_merge", "email_send", "vm_restart"],
    },
    "music": {
        "search":       ["web_search", "deep_research", "fetch_url", "file_search"],
        "optimization": ["python_exec", "shell_run", "file_write", "code_write"],
        "coordination": ["delegate_task", "delegate_batch", "schedule_task", "goal_create"],
        "flow":         ["git_log", "git_status", "file_read", "file_list"],
        "memory":       ["recall", "journal_search", "journal_add", "recall_memories"],
        "pruning":      ["file_delete", "git_branch", "shell_reset", "goal_update"],
        "parallelism":  ["delegate_batch", "browser_open", "shell_run", "vm_execute"],
        "deadlock":     ["email_send", "shell_reset", "git_merge", "vm_restart"],
    },
    "economics": {
        "search":       ["web_search", "deep_research", "fetch_url", "file_search"],
        "optimization": ["python_exec", "pt_hyperparameter_tune", "shell_run", "file_write"],
        "coordination": ["delegate_task", "delegate_batch", "schedule_task", "goal_create"],
        "flow":         ["git_log", "git_status", "watch_url", "monitor_url"],
        "memory":       ["recall", "recall_memories", "memory_set_permanent", "journal_search"],
        "pruning":      ["file_delete", "goal_update", "shell_reset", "git_branch"],
        "parallelism":  ["delegate_batch", "shell_run", "python_exec", "vm_execute"],
        "deadlock":     ["email_send", "shell_reset", "git_merge", "vm_restart"],
    },
    "architecture": {
        "search":       ["web_search", "deep_research", "file_search", "fetch_url"],
        "optimization": ["file_write", "shell_run", "python_exec", "code_write"],
        "coordination": ["delegate_task", "delegate_batch", "goal_create", "schedule_task"],
        "flow":         ["file_list", "git_log", "git_status", "file_read"],
        "memory":       ["recall", "recall_memories", "journal_search", "journal_read"],
        "pruning":      ["file_delete", "git_branch", "shell_reset", "goal_update"],
        "parallelism":  ["delegate_batch", "shell_run", "python_exec", "vm_execute"],
        "deadlock":     ["shell_reset", "git_merge", "git_push", "vm_restart"],
    },
    "game_theory": {
        "search":       ["web_search", "deep_research", "file_search", "fetch_url"],
        "optimization": ["python_exec", "shell_run", "pt_hyperparameter_tune", "file_write"],
        "coordination": ["delegate_task", "delegate_batch", "schedule_task", "goal_create"],
        "flow":         ["git_log", "git_status", "watch_url", "monitor_url"],
        "memory":       ["recall", "recall_memories", "memory_set_permanent", "journal_search"],
        "pruning":      ["file_delete", "goal_update", "git_branch", "shell_reset"],
        "parallelism":  ["delegate_batch", "shell_run", "python_exec", "vm_execute"],
        "deadlock":     ["email_send", "shell_reset", "git_merge", "vm_restart"],
    },
    "neuroscience": {
        "search":       ["web_search", "deep_research", "file_search", "fetch_url"],
        "optimization": ["python_exec", "shell_run", "pt_hyperparameter_tune", "file_write"],
        "coordination": ["delegate_task", "delegate_batch", "schedule_task", "goal_create"],
        "flow":         ["git_log", "git_status", "file_read", "file_list"],
        "memory":       ["recall", "recall_memories", "memory_set_permanent", "journal_search"],
        "pruning":      ["file_delete", "git_branch", "shell_reset", "goal_update"],
        "parallelism":  ["delegate_batch", "shell_run", "python_exec", "vm_execute"],
        "deadlock":     ["email_send", "shell_reset", "git_merge", "vm_restart"],
    },
    "military": {
        "search":       ["web_search", "deep_research", "file_search", "fetch_url"],
        "optimization": ["shell_run", "python_exec", "pt_hyperparameter_tune", "file_write"],
        "coordination": ["delegate_task", "delegate_batch", "goal_create", "schedule_task"],
        "flow":         ["git_log", "git_status", "watch_url", "monitor_url"],
        "memory":       ["recall", "recall_memories", "memory_set_permanent", "journal_search"],
        "pruning":      ["file_delete", "git_branch", "shell_reset", "goal_update"],
        "parallelism":  ["delegate_batch", "shell_run", "python_exec", "vm_execute"],
        "deadlock":     ["email_send", "shell_reset", "git_merge", "vm_restart"],
    },
}


@dataclass
class ADHDContext:
    task: str
    problem_types: list[str]
    analogies: list[Analogy]
    streams: list[Stream]
    hyperfocus_hint: str = ""
    serendipity_pattern: str = ""
    synthesis: str = ""
    generation_time: float = 0.0

    def get_tool_suggestions(self) -> dict[str, float]:
        """Map ADHD analogies to concrete tool suggestions with confidence scores.

        Returns a dict of {tool_name: confidence_score} where confidence is
        weighted by the analogy's relevance, domain utility, and match specificity.
        """
        suggestions: dict[str, float] = {}
        for analogy in self.analogies:
            domain_tools = _ADHD_TOOL_DOMAIN_MAP.get(analogy.domain, {}).get(analogy.problem_type, [])
            for tool in domain_tools:
                # Weight: relevance of the analogy * decay for lower-ranked matches
                boost = analogy.relevance * 0.8
                suggestions[tool] = max(suggestions.get(tool, 0.0), boost)

        # Include hyperfocus hint tool if available
        if self.hyperfocus_hint:
            # Parse domain/problem_type from hyperfocus hint like "[biology/search] ..."
            import re
            m = re.match(r'\[(\w+)/(\w+)\]', self.hyperfocus_hint)
            if m:
                hf_domain, hf_ptype = m.group(1), m.group(2)
                hf_tools = _ADHD_TOOL_DOMAIN_MAP.get(hf_domain, {}).get(hf_ptype, [])
                for tool in hf_tools:
                    suggestions[tool] = max(suggestions.get(tool, 0.0), 1.0)  # Hyperfocus = max confidence

        return suggestions

    def get_tool_suggestions_text(self) -> str:
        """Get a formatted string of tool suggestions for the prompt block."""
        suggestions = self.get_tool_suggestions()
        if not suggestions:
            return ""
        sorted_tools = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
        lines = ["[ADHD Tool Suggestions]"]
        for tool, confidence in sorted_tools[:5]:
            label = "★ " if confidence >= 0.9 else "  "
            lines.append(f"  {label}{tool} (confidence: {confidence:.2f})")
        return "\n".join(lines)

    def to_prompt_block(self, max_analogies: int = 3) -> str:
        lines = ["[ADHD Cross-Domain Reasoning]"]
        lines.append(f"Problem structure: {', '.join(self.problem_types) or 'general'}")
        if self.analogies:
            lines.append("Analogies:")
            for a in self.analogies[:max_analogies]:
                lines.append(f"  [{a.domain}/{a.problem_type}] {a.pattern[:150]}")
                if a.translation:
                    lines.append(f"  -> {a.translation[:150]}")
        if self.hyperfocus_hint:
            lines.append(f"[Focus] {self.hyperfocus_hint}")
        if self.serendipity_pattern:
            lines.append(f"[Serendipity] {self.serendipity_pattern[:150]}")
        if self.synthesis:
            lines.append(f"[Synthesis] {self.synthesis}")
        # Append tool suggestions
        tool_text = self.get_tool_suggestions_text()
        if tool_text:
            lines.append("")
            lines.append(tool_text)
        return "\n".join(lines)


# =============================================================================
# RESULT CACHE
# =============================================================================

_CACHE_MAX_SIZE = 32
_CACHE_TTL_DEFAULT = 30  # seconds


class _ADHDResultCache:
    """LRU cache for ADHD results keyed by task+complexity."""

    def __init__(self, max_size: int = _CACHE_MAX_SIZE, ttl: float = _CACHE_TTL_DEFAULT):
        self._max_size = max_size
        self._ttl = ttl
        self._cache: dict[str, tuple[float, str]] = {}  # key -> (timestamp, result)
        self._access_order: list[str] = []  # LRU tracking
        self._lock = asyncio.Lock()

    def _make_key(self, task: str, complexity: str) -> str:
        return f"{task.strip()[:200].lower()}::{complexity}"

    async def get(self, task: str, complexity: str) -> str | None:
        async with self._lock:
            key = self._make_key(task, complexity)
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, result = entry
            if time.time() - ts > self._ttl:
                self._cache.pop(key, None)
                self._access_order = [k for k in self._access_order if k != key]
                return None
            # Move to end (most recently used)
            self._access_order = [k for k in self._access_order if k != key]
            self._access_order.append(key)
            return result

    async def set(self, task: str, complexity: str, result: str) -> None:
        async with self._lock:
            key = self._make_key(task, complexity)
            self._cache[key] = (time.time(), result)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            # Evict LRU if over max size
            while len(self._cache) > self._max_size:
                lru_key = self._access_order.pop(0)
                self._cache.pop(lru_key, None)

    async def invalidate(self, task: str | None = None, complexity: str | None = None) -> None:
        async with self._lock:
            if task is not None:
                if complexity is None:
                    # Require complexity to avoid ambiguous key construction
                    raise ValueError("complexity is required when invalidating a specific task")
                key = self._make_key(task, complexity)
                self._cache.pop(key, None)
                self._access_order = [k for k in self._access_order if k != key]
            else:
                self._cache.clear()
                self._access_order.clear()


# =============================================================================
# CORE MODULE
# =============================================================================

class ADHDReasoningModule:
    def __init__(
        self,
        llm_module: Any = None,
        hyperfocus_threshold: float = 0.68,
        serendipity_rate: float = 0.20,
        max_streams: int = 4,
    ):
        self.llm = llm_module
        self.hyperfocus_threshold = hyperfocus_threshold
        self.serendipity_rate = serendipity_rate
        self.max_streams = max_streams
        self.domains = list(DOMAIN_PATTERNS.keys())

        # ── Adaptive domain utility (exponential moving average) ──
        self._domain_utility: dict[str, float] = {d: 0.5 for d in self.domains}
        self._domain_utility_decay: float = 0.5  # moderate = perceptible adaptation within 1-2 fires

        # ── Domain fatigue — temporary dip after use (interest-based fading) ──
        # Each domain starts at 0 fatigue. After use, fatigue jumps to 0.6 and decays by 0.15 per fire.
        self._domain_fatigue: dict[str, float] = {d: 0.0 for d in self.domains}
        self._domain_fatigue_initial: float = 0.6   # fatigue spike after use
        self._domain_fatigue_decay: float = 0.15    # how fast fatigue fades per fire

        # ── Novelty tracker — how many fires since each domain was last used ──
        self._fires_since_use: dict[str, int] = {d: 5 for d in self.domains}  # start with moderate novelty

        # ── ADHD noise — stochastic perturbation to simulate non-linear processing ──
        self._adhd_noise_std: float = 0.12  # standard deviation of Gaussian noise added to relevance

        # ── Hyperfocus state — tracks escalating lock-on ──
        self._hyperfocus_domain: str | None = None   # domain currently in hyperfocus, if any
        self._hyperfocus_streak: int = 0              # consecutive fires in hyperfocus
        self._hyperfocus_trigger: float = 0.50        # lower threshold to ENTER hyperfocus
        self._hyperfocus_max_streak: int = 5            # max consecutive fires before hyperfocus burnout

        # ── Racing thoughts — probability of randomly derailing to a different domain ──
        self._derail_rate: float = 0.15               # chance to replace a domain with a random one

        # ── Serendipity state ──
        self._fire_count: int = 0
        self._last_used_domains: set[str] = set()  # domains used in the most recent fire

        # ── Result cache ──
        self._cache = _ADHDResultCache(max_size=_CACHE_MAX_SIZE, ttl=_CACHE_TTL_DEFAULT)

        # ── Last context (for engine to access tool suggestions) ──
        self.last_context: ADHDContext | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    async def fire(
        self,
        task: str,
        complexity: str = "medium",
        use_llm: bool = False,
    ) -> ADHDContext | None:
        """Run the full ADHD pipeline on a task.

        Args:
            task: The user's input/task description.
            complexity: From _assess_complexity() — "low", "medium", or "high".
            use_llm: If True, uses LLM calls for translations and synthesis
                     (only recommended for high complexity tasks to save tokens).

        Returns:
            ADHDContext ready for to_prompt_block(), or None if task is too simple.
        """
        t0 = time.time()
        self._fire_count += 1

        problem_types = _extract_problem_types(task)
        if not problem_types:
            return None

        analogies = self._cross_domain_fire(problem_types)
        if not analogies:
            return None

        streams = self._build_streams(task, analogies)

        # ── Adaptive learning: update domain utility after each fire ──
        self._update_domain_utility(streams)
        self._last_used_domains = {s.domain for s in streams}

        hyperfocus_hint = self._detect_hyperfocus(streams)

        serendipity_pattern = ""
        serendipity = self._maybe_serendipity()
        if serendipity:
            serendipity_pattern = f"[{serendipity[0]}/{serendipity[1]}] {serendipity[2]}"

        synthesis = ""
        if use_llm and self.llm is not None and complexity == "high":
            synthesis = await self._synthesize(task, streams, hyperfocus_hint)

        ctx = ADHDContext(
            task=task,
            problem_types=problem_types,
            analogies=analogies,
            streams=streams,
            hyperfocus_hint=hyperfocus_hint,
            serendipity_pattern=serendipity_pattern,
            synthesis=synthesis,
            generation_time=time.time() - t0,
        )
        self.last_context = ctx
        return ctx

    # ──────────────────────────────────────────────────────────────────────────
    # ADAPTIVE DOMAIN UTILITY — learns which domains produce the best streams
    # ──────────────────────────────────────────────────────────────────────────

    def _update_domain_utility(self, streams: list[Stream]) -> None:
        """Update domain utility via exponential moving average.

        A domain's score is the average confidence of its streams (0 if unused).
        The EMA smooths across fires so recent performance has more weight.

        Also updates domain fatigue: used domains get a fatigue spike,
        which decays each fire. This creates the ADHD pattern of
        intense interest -> abrupt disinterest -> recovery.
        """
        if not streams:
            return

        # ── Update fatigue: decay all, spike used domains ──
        used_domains = {s.domain for s in streams}
        for domain in self._domain_fatigue:
            old_fatigue = self._domain_fatigue[domain]
            if domain in used_domains:
                # Spike fatigue (interest abruptly drops after use)
                self._domain_fatigue[domain] = min(old_fatigue + self._domain_fatigue_initial, 0.9)
            else:
                # Fatigue decays naturally
                self._domain_fatigue[domain] = max(old_fatigue - self._domain_fatigue_decay, 0.0)

        # ── Update novelty tracker ──
        for domain in self._fires_since_use:
            if domain in used_domains:
                self._fires_since_use[domain] = 0  # reset — just used
            else:
                self._fires_since_use[domain] += 1  # not used — novelty grows

        # Compute per-domain average confidence from this fire
        domain_scores: dict[str, list[float]] = {}
        for s in streams:
            domain_scores.setdefault(s.domain, []).append(s.confidence)

        decay = self._domain_utility_decay
        for domain in self._domain_utility:
            scores = domain_scores.get(domain)
            if scores:
                avg_confidence = sum(scores) / len(scores)
                # Scale confidence from [0, 1] to a score around 0.5
                new_score = 0.3 + 0.7 * avg_confidence
            else:
                # Domain was not selected — slight decay
                new_score = 0.4

            # EMA update: new_utility = decay * old + (1-decay) * new_score
            old = self._domain_utility.get(domain, 0.5)
            self._domain_utility[domain] = decay * old + (1.0 - decay) * new_score

    # ──────────────────────────────────────────────────────────────────────────
    # CROSS-DOMAIN FIRE — query all domains for relevant patterns
    # ──────────────────────────────────────────────────────────────────────────

    def _cross_domain_fire(self, problem_types: list[str]) -> list[Analogy]:
        candidates: list[Analogy] = []
        for domain in self.domains:
            domain_data = DOMAIN_PATTERNS.get(domain, {})
            utility = self._domain_utility.get(domain, 0.5)

            # ── Interest-based relevance (not just importance) ──
            # ADHD brain: interest = familiarity (utility) + novelty (time since use) - fatigue
            novelty_bonus = min(self._fires_since_use.get(domain, 0) * 0.04, 0.25)
            fatigue_penalty = self._domain_fatigue.get(domain, 0.0)
            interest = max(utility + novelty_bonus - fatigue_penalty, 0.05)

            for i, ptype in enumerate(problem_types):
                pattern = domain_data.get(ptype)
                if not pattern:
                    continue
                type_weight = 1.0 / (1.0 + i)

                # ── ADHD noise: stochastic perturbation to simulate non-linear processing ──
                noise = random.gauss(0, self._adhd_noise_std)

                # ── Hyperfocus escalation: if already in hyperfocus, amp the same domain ──
                hf_boost = 0.0
                if self._hyperfocus_domain == domain and self._hyperfocus_streak > 0:
                    # Hyperfocus escalates: each consecutive fire in the same domain amplifies it
                    hf_boost = min(self._hyperfocus_streak * 0.08, 0.35)

                relevance = max(interest * type_weight + noise + hf_boost, 0.0)
                candidates.append(Analogy(
                    domain=domain,
                    problem_type=ptype,
                    pattern=pattern,
                    relevance=relevance,
                ))
        candidates.sort(key=lambda a: a.relevance, reverse=True)
        return candidates[:6]

    # ──────────────────────────────────────────────────────────────────────────
    # STREAM BUILDING & BLEEDING — cross-pollinate reasoning content
    # ──────────────────────────────────────────────────────────────────────────

    def _build_streams(self, task: str, analogies: list[Analogy]) -> list[Stream]:
        # ── Apply racing thoughts derail BEFORE building streams ──
        analogies = self._maybe_derail(analogies)

        streams: list[Stream] = []
        for i, a in enumerate(analogies[:self.max_streams]):
            stream = Stream(
                stream_id=f"s{i}_{a.domain}",
                domain=a.domain,
                problem_type=a.problem_type,
                seed_thought=f"Through the lens of {a.domain}: {a.pattern[:100]}",
                confidence=a.relevance,
            )
            streams.append(stream)
        self._bleed_streams(streams)
        return streams

    def _bleed_streams(self, streams: list[Stream]) -> None:
        """Cross-pollinate the weakest stream with content from the strongest.

        Instead of just boosting confidence by 0.1, this actually blends
        the seed_thought from the highest-confidence stream into the
        lowest-confidence one, creating a hybrid reasoning fragment.
        """
        if len(streams) < 2:
            return

        streams.sort(key=lambda s: s.confidence, reverse=True)
        best = streams[0]
        weakest = streams[-1]

        # Blend actual reasoning content: prepend the best stream's insight
        # to the weakest stream's seed thought
        bleed_fragment = f"[Bleed from {best.domain}] {best.seed_thought[:80]}"
        if bleed_fragment not in weakest.seed_thought:
            weakest.seed_thought = f"{bleed_fragment} | {weakest.seed_thought}"

        # Boost weakest stream's confidence meaningfully
        conf_boost = min(0.15 * (best.confidence / max(weakest.confidence, 0.01)), 0.3)
        weakest.confidence = min(weakest.confidence + conf_boost, 1.0)

        # Slight decay to best stream to prevent runaway dominance
        best.confidence = max(best.confidence - 0.02, 0.1)

    # ──────────────────────────────────────────────────────────────────────────
    # RACING THOUGHTS — random domain derail to simulate ADHD thought jumps
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_derail(self, analogies: list[Analogy]) -> list[Analogy]:
        """Randomly replace one analogy with a different domain to simulate
        racing thoughts / tangential thinking.

        Real ADHD: your brain randomly jumps to a completely different topic
        mid-stream. This injects that chaos into the reasoning.
        """
        if len(analogies) < 2:
            return analogies

        if random.random() > self._derail_rate:
            return analogies

        # Pick a domain not currently in the analogies
        current_domains = {a.domain for a in analogies}
        other_domains = [d for d in self.domains if d not in current_domains]
        if not other_domains:
            return analogies

        # Pick a random other domain and a random problem type
        derail_domain = random.choice(other_domains)
        # Pick the problem type most aligned with existing analogies or random
        existing_types = [a.problem_type for a in analogies]
        derail_type = random.choice(existing_types + list(DOMAIN_PATTERNS.get(derail_domain, {}).keys()))

        pattern = DOMAIN_PATTERNS.get(derail_domain, {}).get(derail_type, "")
        if not pattern:
            return analogies

        # Replace a random analogy (not the top one, to keep some coherence)
        replace_idx = random.randint(1, len(analogies) - 1)
        derail_analogy = Analogy(
            domain=derail_domain,
            problem_type=derail_type,
            pattern=f"[Derail] {pattern}",
            relevance=analogies[replace_idx].relevance * 0.8,  # slightly lower to not dominate
        )
        analogies[replace_idx] = derail_analogy
        return analogies

    # ──────────────────────────────────────────────────────────────────────────
    # HYPERFOCUS DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_hyperfocus(self, streams: list[Stream]) -> str:
        """Detect hyperfocus with dynamic escalation.

        Real ADHD hyperfocus:
        - Enters at a lower threshold than neurotypical focus
        - Escalates: the longer you're in it, the harder to disengage
        - Eventually becomes so intense it breaks (hyperfocus escape)

        Tracks streak across fires for the same domain.
        """
        if not streams:
            # Reset hyperfocus if no streams
            self._hyperfocus_domain = None
            self._hyperfocus_streak = 0
            return ""

        best = max(streams, key=lambda s: s.confidence)
        domain = best.domain

        # ── Check if we should enter hyperfocus ──
        # Lower trigger threshold than neurotypical, with fatigue override
        fatigue = self._domain_fatigue.get(domain, 0.0)
        effective_threshold = self.hyperfocus_threshold
        if self._hyperfocus_domain == domain:
            # Already in hyperfocus — use lower threshold to stay locked
            effective_threshold = self._hyperfocus_trigger
        elif fatigue < 0.3 and best.confidence >= self._hyperfocus_trigger:
            # Not in hyperfocus but interest is high and fatigue is low — enter
            effective_threshold = self._hyperfocus_trigger
        else:
            effective_threshold = self.hyperfocus_threshold

        # ── Hyperfocus escape: if streak exceeds limit, force break ──
        if self._hyperfocus_domain == domain and self._hyperfocus_streak >= self._hyperfocus_max_streak:
            # Hyperfocus burnout — force break
            self._hyperfocus_domain = None
            self._hyperfocus_streak = 0
            return ""

        if best.confidence >= effective_threshold:
            best.is_hyperfocused = True

            # Track hyperfocus streak
            if self._hyperfocus_domain == domain:
                self._hyperfocus_streak += 1
            else:
                self._hyperfocus_domain = domain
                self._hyperfocus_streak = 1

            # Escalating confidence for hyperfocus — makes it self-reinforcing
            best.confidence = min(best.confidence + self._hyperfocus_streak * 0.04, 1.0)

            return f"[Hyperfocus/{domain}/{best.problem_type}] {best.seed_thought}"

        # ── Not in hyperfocus — decay streak ──
        if self._hyperfocus_domain == domain:
            self._hyperfocus_streak = max(self._hyperfocus_streak - 1, 0)
            if self._hyperfocus_streak == 0:
                self._hyperfocus_domain = None
        else:
            self._hyperfocus_domain = None
            self._hyperfocus_streak = 0

        return ""

    # ──────────────────────────────────────────────────────────────────────────
    # SERENDIPITY — wild-card injection to escape local optima
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_serendipity(self) -> tuple[str, str, str] | None:
        """Maybe inject a wild-card domain not recently used.

        Real ADHD brain: higher baseline serendipity (racing thoughts),
        increases with fire count, and preferentially jumps to domains
        that are maximally different from current focus.

        The serendipity rate increases with fire count (up to ~36% with defaults, 70% cap if configured higher).
        Domains used in the most recent fire are excluded to maximize diversity.
        High-utility domains that have dominated recent fires are also deprioritized.
        High-fatigue domains (recently hyperfocused on) are preferred — novelty seeking.
        """
        adjusted = min(self.serendipity_rate * (1 + 0.08 * min(self._fire_count, 10)), 0.7)
        if random.random() > adjusted:
            return None

        # Exclude domains used in the most recent fire(s)
        excluded = self._last_used_domains.copy()

        # Also deprioritize high-utility domains (they're already well-represented)
        high_utility = {
            d for d, u in self._domain_utility.items()
            if u > 0.65 and d not in excluded
        }
        # Only exclude high-utility if we'd still have domain choices
        tentative_pool = [d for d in self.domains if d not in excluded]
        if len(tentative_pool) > 2:
            # We have enough alternatives — exclude high-utility too
            pool = [d for d in self.domains if d not in excluded | high_utility]
            if not pool:
                pool = tentative_pool
        else:
            pool = tentative_pool

        if not pool:
            pool = list(self.domains)

        # ── ADHD twist: prefer high-fatigue domains (recently abandoned interests) ──
        # This mimics jumping back to something you were obsessed with before
        fatigued = [(d, self._domain_fatigue.get(d, 0.0)) for d in pool]
        fatigued.sort(key=lambda x: x[1], reverse=True)

        # 40% chance to pick the most fatigued domain (was interested before)
        if fatigued and fatigued[0][1] > 0.2 and random.random() < 0.4:
            domain = fatigued[0][0]
        else:
            domain = random.choice(pool)

        patterns = DOMAIN_PATTERNS.get(domain, {})
        if not patterns:
            return None
        ptype = random.choice(list(patterns.keys()))
        return domain, ptype, patterns[ptype]

    # ──────────────────────────────────────────────────────────────────────────
    # LLM SYNTHESIS (optional, high-complexity only)
    # ──────────────────────────────────────────────────────────────────────────

    async def _synthesize(self, task: str, streams: list[Stream], hint: str) -> str:
        if not streams:
            return ""
        fragments = "\n".join(f"[{s.domain}] {s.seed_thought}" for s in streams if s.seed_thought)
        if hint:
            fragments += f"\nStrongest: {hint}"
        prompt = (
            f"Task: {task}\n\n"
            f"Cross-domain reasoning:\n{fragments}\n\n"
            "Synthesize these into ONE concrete, actionable strategy sentence "
            "for solving the task. No preamble. Be direct."
        )
        try:
            resp = await self.llm.complete(prompt, max_tokens=100, temperature=0.3)
            return (resp.content or "").strip() if hasattr(resp, "content") else str(resp).strip()
        except Exception as e:
            logger.debug("ADHD synthesis LLM call failed: %s", e)
            return ""


# =============================================================================
# PROBLEM TYPE EXTRACTION
# =============================================================================

def _extract_problem_types(task: str) -> list[str]:
    """Extract up to 3 problem types from the task using keyword signals.

    Normalized by sqrt(len(signals)) to avoid bias toward types with more keywords.
    Falls back to ["search"] if nothing matches.
    """
    task_lower = task.lower()
    scores: dict[str, float] = {}
    for ptype, signals in PROBLEM_TYPE_SIGNALS.items():
        hits = sum(1 for s in signals if s in task_lower)
        if hits > 0:
            scores[ptype] = hits / math.sqrt(len(signals))
    if not scores:
        return ["search"]
    sorted_types = sorted(scores, key=lambda t: scores[t], reverse=True)
    return sorted_types[:3]


# =============================================================================
# CONVENIENCE HELPER — called from ReasoningEngine
# =============================================================================

async def maybe_fire_adhd(
    task: str,
    complexity: str = "medium",
    module: ADHDReasoningModule | None = None,
) -> str | None:
    """Convenience helper: return ADHD prompt block or None.

    Call this from ReasoningEngine._nexus_reason() after complexity assessment:
        adhd_block = await maybe_fire_adhd(user_input, complexity, self._adhd_module)
        if adhd_block:
            messages.append({"role": "system", "content": adhd_block})

    Config keys used:
        enable_adhd_reasoning (bool, default True)
        adhd_complexity_min (str, default "medium")
        adhd_use_llm (bool, default False)
        adhd_cache_ttl (int, default 30)
        adhd_max_analogies (int, default 3)
    """
    cfg = None
    try:
        from ..config import get as cfg_get
        cfg = cfg_get
    except ImportError:
        return None

    if not cfg("enable_adhd_reasoning", True):
        return None

    min_complexity = cfg("adhd_complexity_min", "medium")
    complexity_order = {"low": 0, "medium": 1, "high": 2}
    if complexity_order.get(complexity, 0) < complexity_order.get(min_complexity, 1):
        return None

    use_llm = cfg("adhd_use_llm", False) and complexity == "high"

    if module is None:
        return None

    # ── Check cache ──
    cache_ttl = cfg("adhd_cache_ttl", _CACHE_TTL_DEFAULT)
    if module._cache._ttl != cache_ttl:
        module._cache._ttl = cache_ttl

    cached = await module._cache.get(task, complexity)
    if cached is not None:
        return cached

    # ── Fire the module ──
    ctx = await module.fire(task, complexity=complexity, use_llm=use_llm)
    if ctx is None:
        return None

    max_analogies = cfg("adhd_max_analogies", 3)
    result = ctx.to_prompt_block(max_analogies=max_analogies)

    # ── Cache result ──
    await module._cache.set(task, complexity, result)

    return result


# =============================================================================
# TOOL BOOSTING — get ADHD-recommended tool names for prefilter boosting
# =============================================================================

def get_adhd_tool_boost(module: ADHDReasoningModule | None) -> dict[str, float]:
    """Get tool boosting weights from the last ADHD context.

    Returns a dict of {tool_name: confidence} that the engine can use
    to boost relevant tools in _prefilter_tools.
    """
    if module is None or module.last_context is None:
        return {}
    return module.last_context.get_tool_suggestions()
