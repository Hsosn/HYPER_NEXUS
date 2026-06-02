"""
Tool Registry v18 — Deep Optimized (v20 optimizations)
====================================

Production-grade tool execution engine with:

1. Middleware pipeline (pre/post/error hooks with priority ordering)
2. Tool composition & chaining (sequential, parallel, fan-out)
3. Semantic tool discovery (embedding similarity with cache)
4. LRU result caching (OrderedDict, per-tool TTL, hit/miss metrics)
5. Deep profiling (latency percentiles, per-param success, error categorization)
6. Dependency resolution (DAG validation, auto-load)
7. Rate limiting (sliding window, per-tool configurable)
8. Health checking (circuit breaker, auto-disable/recover)

v17 bugs fixed:
- Parameter mutation: params dict is now deep-copied before modification
- hashlib import moved to top level
- Cache pruning: O(1) via OrderedDict.move_to_end instead of O(n log n) sort

v20 optimizations:
- Lazy schema caching with dirty-flag invalidation
- DB profile cache (60s TTL) in to_openai_format()
- Cached health status per tool (avoids repeated dict lookups)
- Early return in _validate_params for tools with no required params
- Batch execution helper with dependency-aware parallel execution
- Removed dead no-op setdefault call in register()

Usage:
    from nexus.tools.registry import REGISTRY, tool

    @tool("my_tool", "Does X", {...})
    async def my_tool_handler(params: dict) -> str:
        return "result"
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import heapq
import inspect
import json
import logging
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
)

from ..events import emit
from ..memory import db

# DSML/XML tool call markup stripping — strips <|DSML|tag>...</|DSML|tag> blocks
# and other tool call markup formats from tool output before sending to frontend.
# Unified character set matches engine.py's _strip_dsml_from_text().
import re as _re
_DSML_STRIP_PATTERNS = [
    _re.compile(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>[\s\S]*?<\/[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+>', _re.DOTALL | _re.IGNORECASE),
    _re.compile(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?\/>', _re.IGNORECASE),
    _re.compile(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>', _re.IGNORECASE),
    _re.compile(r'<DSML_\w+[^>]*>[\s\S]*?</DSML_\w+>', _re.DOTALL | _re.IGNORECASE),
    _re.compile(r'<tool_call>[\s\S]*?<\/tool_call>', _re.DOTALL | _re.IGNORECASE),
    _re.compile(r'<invoke[^>]*>[\s\S]*?<\/invoke>', _re.DOTALL | _re.IGNORECASE),
    _re.compile(r'<tool_calls>[\s\S]*?<\/tool_calls>', _re.DOTALL | _re.IGNORECASE),
    _re.compile(r'<(parameter|tool_response|tool_result|error|result|response|tool_input|input|output|status|invoke)[^>]*>[\s\S]*?<\/\1>', _re.DOTALL | _re.IGNORECASE),
    _re.compile(r'<[\w-]+(?:\s+[^>]*)?\/?>'),
]
def _strip_dsml(text: str) -> str:
    """Strip DSML/XML tool call markup from tool output.
    Prevents DSML tags from leaking into the frontend WebSocket payload.
    """
    if not text:
        return text
    for p in _DSML_STRIP_PATTERNS:
        text = p.sub('', text)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

__all__ = [
    "Tool",
    "ToolResult",
    "ToolMiddleware",
    "ToolProfile",
    "HealthStatus",
    "ComposeStep",
    "ComposeResult",
    "ToolMatch",
    "ErrorCategory",
    "ToolRegistry",
    "REGISTRY",
    "tool",
]

logger = logging.getLogger("nexus.registry")


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Handler = Callable[[dict[str, Any]], Awaitable[str]]


# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    """Categories for classifying tool execution errors."""

    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RUNTIME = "runtime"
    NETWORK = "network"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """Represents a single tool that the agent can invoke.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description (used for LLM function-calling
            and semantic discovery).
        parameters_schema: JSON Schema dict describing expected parameters.
        handler: Async function ``async (params: dict) -> str``.
        risk: Risk level — ``low``, ``medium``, ``high``, or ``critical``.
        category: Logical grouping (e.g. ``"file"``, ``"web"``, ``"git"``).
        cacheable: When ``True`` and the handler succeeds, the result is cached
            keyed by a hash of ``name + sorted params``.
        cache_ttl: Per-tool cache TTL in seconds (only effective when
            ``cacheable=True``).
        rate_limit: Max invocations per ``rate_window`` seconds.  ``0`` means
            unlimited.
        rate_window: Sliding window size in seconds for rate limiting.
        depends_on: List of tool names that must be available (registered)
            before this tool can execute.
        timeout: Maximum execution time in seconds before cancellation.
        version: Semantic version string for the tool implementation.
        tags: Free-form tags useful for filtering / discovery.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Handler
    risk: str = "low"
    category: str = "general"
    cacheable: bool = False
    cache_ttl: int = 300
    rate_limit: int = 0
    rate_window: int = 60
    depends_on: list[str] = field(default_factory=list)
    timeout: int = 30
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)

    async def to_openai_format(self) -> dict[str, Any]:
        # v20: Optimized — caches DB profile (60s TTL) and health status
        """Convert to OpenAI function-calling format, with intelligence hints."""
        desc = self.description

        # v20: Use cached DB profile with 60s TTL to avoid repeated DB queries
        profile = None
        registry = _REGISTRY_SINGLETON_REF if _REGISTRY_SINGLETON_REF else None
        if registry:
            cached_entry = registry._db_profile_cache.get(self.name)
            now = time.time()
            if cached_entry and (now - cached_entry[0]) < registry._DB_PROFILE_TTL:
                profile = cached_entry[1]
            else:
                profile = await db.get_tool_profile(self.name)
                registry._db_profile_cache[self.name] = (now, profile or {})
        else:
            profile = await db.get_tool_profile(self.name)

        if profile and profile.get("learned_strategy"):
            desc += f"\n\n[Learned best practice]: {profile['learned_strategy']}"
        if profile and profile.get("success_rate") is not None:
            rate = profile["success_rate"]
            if rate < 0.5:
                desc += f"\n[Note: This tool has a {rate:.0%} success rate — use with caution]"

        # Append health warning if unhealthy — v20: direct dict access instead of method call
        if registry:
            health = registry._health.get(self.name)
        else:
            health = None
        if health and not health.healthy:
            desc += f"\n[WARNING: Tool health degraded — {health.reason}]"

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": self.parameters_schema,
            },
        }


@dataclass
class ToolResult:
    """Standardized result from tool execution.

    Supports tuple unpacking for backward compatibility:
        output, success, duration_ms = result  # 3-tuple (required)
    This prevents ``TypeError: cannot unpack non-iterable ToolResult object``
    if any caller accidentally treats the result as a tuple.
    """

    success: bool
    output: str
    tool_name: str = ""
    duration_ms: float = 0.0
    cached: bool = False
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    error_detail: str = ""
    params_snapshot: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Allow tuple unpacking: (output, success, duration_ms)."""
        return iter((self.output, self.success, self.duration_ms))

    def __len__(self) -> int:
        return 3


@dataclass
class ToolProfile:
    """Deep profiling data for a single tool.

    Tracks:
    - Call counts and success rates
    - Latency percentiles (p50 / p90 / p99)
    - Per-parameter-hash success rates
    - Error categorization counts
    """

    tool_name: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    latencies: list[float] = field(default_factory=list)  # all durations in ms
    param_success: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"success": {}, "total": {}}
    )  # hash -> count
    error_categories: dict[str, int] = field(default_factory=dict)
    last_used: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    learned_strategy: str = ""

    # --- Derived properties ---

    @property
    def success_rate(self) -> float:
        """Fraction of calls that succeeded (0.0 – 1.0)."""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls

    @property
    def avg_duration_ms(self) -> float:
        """Mean latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def percentile(self, p: int) -> float:
        """Return the *p*-th percentile latency (e.g. ``percentile(90)``).

        Uses the "nearest-rank" method which requires no interpolation.
        Returns 0.0 when there is no data.
        """
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        k = max(0, min(int(math.ceil(p / 100.0 * len(sorted_lat))) - 1, len(sorted_lat) - 1))
        return sorted_lat[k]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p90(self) -> float:
        return self.percentile(90)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def effectiveness_score(self) -> float:
        """Composite effectiveness score combining success rate, speed, and volume.

        Formula: ``0.5 * success_rate + 0.3 * speed_score + 0.2 * confidence``
        where speed_score penalizes tools slower than 5s, and confidence
        ramps up with more calls (full at 20+).
        """
        sr = self.success_rate
        speed = max(0.0, 1.0 - self.avg_duration_ms / 5000.0)
        confidence = min(self.total_calls / 20.0, 1.0)
        return round(0.5 * sr + 0.3 * speed + 0.2 * confidence, 3)

    def record(
        self,
        success: bool,
        duration_ms: float,
        params: dict[str, Any] | None = None,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    ) -> None:
        """Record a single execution."""
        self.total_calls += 1
        now = time.time()
        self.last_used = now

        if success:
            self.success_count += 1
            self.last_success = now
        else:
            self.failure_count += 1
            self.last_failure = now
            self.error_categories[error_category.value] = (
                self.error_categories.get(error_category.value, 0) + 1
            )

        # Keep latency ring buffer bounded (last 500 samples)
        self.latencies.append(duration_ms)
        if len(self.latencies) > 500:
            self.latencies = self.latencies[-500:]

        # Per-parameter-hash tracking
        if params:
            param_hash = hashlib.md5(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()[:12]
            self.param_success["total"][param_hash] = (
                self.param_success["total"].get(param_hash, 0) + 1
            )
            if success:
                self.param_success["success"][param_hash] = (
                    self.param_success["success"].get(param_hash, 0) + 1
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for external consumption."""
        return {
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 3),
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "p50_ms": round(self.p50, 1),
            "p90_ms": round(self.p90, 1),
            "p99_ms": round(self.p99, 1),
            "effectiveness_score": self.effectiveness_score,
            "error_categories": dict(self.error_categories),
            "last_used": self.last_used,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "learned_strategy": self.learned_strategy,
        }


# ---------------------------------------------------------------------------
# Health & circuit breaker
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    """Tracks the health of a tool for circuit-breaking decisions."""

    healthy: bool = True
    consecutive_failures: int = 0
    total_failures: int = 0
    last_check: float = 0.0
    reason: str = ""
    disabled_at: float = 0.0  # when the circuit opened


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@dataclass
class ToolMiddleware:
    """Base middleware descriptor.

    Subclass or instantiate with the hook callables you need.
    Only the hooks that are not ``None`` will be invoked.
    """

    name: str = "unnamed"

    async def before(
        self,
        tool: Tool,
        params: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any] | None:
        """Pre-execution hook.

        Return a (possibly modified) params dict to continue execution,
        or ``None`` to **abort** the call (the tool will not run).
        """
        return params

    async def after(
        self,
        tool: Tool,
        result: ToolResult,
        session_id: str,
    ) -> ToolResult | None:
        """Post-execution hook.

        Return a (possibly modified) ``ToolResult`` to pass it along,
        or ``None`` to swallow the result (the original is still returned
        to the caller).
        """
        return result

    async def on_error(
        self,
        tool: Tool,
        error: Exception,
        params: dict[str, Any],
        session_id: str,
    ) -> ToolResult | None:
        """Error hook.

        Return a ``ToolResult`` to **replace** the error result,
        or ``None`` to let the default error handling proceed.
        """
        return None


# ---------------------------------------------------------------------------
# Composition types
# ---------------------------------------------------------------------------

@dataclass
class ComposeStep:
    """A single step in a composed tool pipeline.

    Attributes:
        tool_name: Name of the tool to invoke.
        params: Static parameters.  Use ``"$output"`` as a placeholder
            to receive the previous step's output string under the key
            ``_prev_output``.
        output_key: If set, the result will be stored in the
            ``ComposeResult.outputs`` dict under this key for later steps.
    """

    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    output_key: str = ""


@dataclass
class ComposeResult:
    """Result of a composed tool pipeline."""

    success: bool
    results: list[ToolResult] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    error: str = ""


@dataclass
class ToolMatch:
    """Result from semantic tool discovery."""

    tool: Tool
    score: float
    reason: str = ""


# ---------------------------------------------------------------------------
# Internal rate-limit tracker (sliding window)
# ---------------------------------------------------------------------------

class _SlidingWindowCounter:
    """Track call timestamps in a sliding window for rate-limiting."""

    def __init__(self, window_seconds: int) -> None:
        self._window = window_seconds
        self._timestamps: list[float] = []

    def record(self) -> None:
        """Record a call at the current time."""
        now = time.time()
        self._timestamps.append(now)
        self._evict(now)

    def count(self) -> int:
        """Return the number of calls in the current window."""
        self._evict(time.time())
        return len(self._timestamps)

    def _evict(self, now: float) -> None:
        """Remove timestamps outside the sliding window."""
        cutoff = now - self._window
        # Binary search for the first timestamp still in window
        idx = 0
        lo, hi = 0, len(self._timestamps)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._timestamps[mid] <= cutoff:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            self._timestamps = self._timestamps[lo:]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for all agent tools.

    Provides:
    - Registration / unregistration with dependency graph validation
    - Middleware pipeline (pre/post/error) with priority ordering
    - Result caching with per-tool TTL and O(1) LRU eviction
    - Tool composition (sequential / parallel / fan-out)
    - Semantic discovery via TF-IDF cosine similarity
    - Sliding-window rate limiting per tool
    - Circuit-breaker health checking
    - Deep profiling with latency percentiles
    """

    # Maximum cache entries before eviction kicks in.
    _CACHE_MAX_SIZE: int = 512

    # Circuit breaker thresholds.
    _CIRCUIT_FAILURE_THRESHOLD: int = 5
    _CIRCUIT_COOLDOWN_SECONDS: float = 60.0

    def __init__(self) -> None:
        # Core tool storage.
        self._tools: dict[str, Tool] = {}

        # Middleware pipeline — sorted by priority (lower runs first).
        self._middlewares: list[tuple[int, ToolMiddleware]] = []

        # --- Result cache (OrderedDict for O(1) LRU eviction) ---
        # key = "tool_name:param_hash", value = (timestamp, result_string)
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # --- Rate limiters (per-tool) ---
        self._rate_counters: dict[str, _SlidingWindowCounter] = {}

        # --- Health / circuit breaker ---
        self._health: dict[str, HealthStatus] = {}

        # --- Deep profiling (in-memory, persisted periodically) ---
        self._profiles: dict[str, ToolProfile] = {}

        # --- Semantic discovery cache ---
        # tool_name -> list[float]  (unit-normalized TF-IDF vector)
        self._desc_vectors: dict[str, list[float]] = {}

        # --- TF-IDF vocabulary (built lazily) ---
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._vocab_built: bool = False

        # --- v20: Lazy schema cache ---
        self._schemas_cache: list[dict] | None = None
        self._schemas_dirty: bool = True

        # --- v20: DB profile cache (60s TTL) for to_openai_format() ---
        self._db_profile_cache: dict[str, tuple[float, dict]] = {}
        self._DB_PROFILE_TTL: float = 60.0

    # ===================================================================
    # Registration
    # ===================================================================

    def register(self, tool: Tool) -> None:
        """Register a tool, validating its dependency graph for cycles."""
        # Validate dependencies exist (warn, don't block — tools may register later).
        for dep in tool.depends_on:
            if dep not in self._tools:
                logger.debug(
                    "Tool '%s' depends on '%s' which is not yet registered",
                    tool.name, dep,
                )

        # Cycle detection: would adding this tool create a cycle?
        # Temporarily add the tool so _has_cycle can discover it when
        # traversing back from a dependency (the tool isn't registered yet,
        # so self._tools.get(tool.name) would otherwise return None).
        if tool.depends_on:
            self._tools[tool.name] = tool  # temp add for cycle check
            visited: set[str] = set()
            has_cycle = self._has_cycle(tool.name, tool.depends_on, visited, set())
            if has_cycle:
                del self._tools[tool.name]  # remove temp entry
                raise ValueError(
                    f"Circular dependency detected involving tool '{tool.name}' "
                    f"and dependencies {tool.depends_on}"
                )
            # Keep the temporary entry — it's the actual registration.

        self._tools[tool.name] = tool

        # Initialise supporting structures.
        self._profiles.setdefault(tool.name, ToolProfile(tool_name=tool.name))
        self._health.setdefault(tool.name, HealthStatus())
        if tool.rate_limit > 0:
            self._rate_counters.setdefault(
                tool.name, _SlidingWindowCounter(tool.rate_window)
            )
        # v20: Removed no-op setdefault — was dead code accessing method without calling it

        # Invalidate semantic cache since tool set changed.
        self._vocab_built = False
        # v20: Invalidate schema cache on register.
        self._schemas_dirty = True

        logger.info(
            "Registered tool '%s' (category=%s, risk=%s, cacheable=%s, "
            "rate_limit=%s/%ss, deps=%s, timeout=%ss)",
            tool.name, tool.category, tool.risk, tool.cacheable,
            tool.rate_limit, tool.rate_window, tool.depends_on, tool.timeout,
        )

    def unregister(self, name: str) -> None:
        """Remove a tool and all its associated state."""
        self._tools.pop(name, None)
        self._rate_counters.pop(name, None)
        self._health.pop(name, None)
        self._desc_vectors.pop(name, None)
        self._vocab_built = False  # invalidate
        # v20: Invalidate caches on unregister.
        self._schemas_dirty = True
        self._db_profile_cache.pop(name, None)
        # Remove any dependents that would be broken.
        broken = [
            tname for tname, t in self._tools.items()
            if name in t.depends_on
        ]
        if broken:
            logger.warning(
                "Unregistered '%s' — tools that depend on it: %s", name, broken
            )
        logger.info("Unregistered tool '%s'", name)

    def get_tool(self, name: str) -> Tool | None:
        """Retrieve a tool by name, or ``None`` if not registered."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return a snapshot of all registered tools."""
        return list(self._tools.values())

    # ===================================================================
    # Dependency helpers
    # ===================================================================

    def _has_cycle(self, node: str, neighbors: list[str], visited: set[str], rec_stack: set[str]) -> bool:
        if node in rec_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        rec_stack.add(node)
        dep_tool = self._tools.get(node)
        if dep_tool:
            for dep in dep_tool.depends_on:
                if self._has_cycle(dep, dep_tool.depends_on, visited, rec_stack):
                    return True
        rec_stack.discard(node)
        return False

    def _resolve_dependencies(self, tool: Tool) -> list[str]:
        """Topologically sort dependencies of a tool.

        Returns ordered list of dependency names (deepest deps first).
        """
        ordered: list[str] = []
        visited: set[str] = set()

        def _visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            dep_tool = self._tools.get(name)
            if dep_tool:
                for d in dep_tool.depends_on:
                    _visit(d)
            ordered.append(name)

        for dep in tool.depends_on:
            _visit(dep)
        return ordered

    # ===================================================================
    # Middleware
    # ===================================================================

    def add_middleware(self, middleware: ToolMiddleware, priority: int = 100) -> None:
        """Register a middleware.  Lower priority values execute first."""
        self._middlewares.append((priority, middleware))
        # Re-sort by priority.
        self._middlewares.sort(key=lambda pair: pair[0])
        logger.info(
            "Added middleware '%s' with priority %d", middleware.name, priority
        )

    def remove_middleware(self, name: str) -> None:
        """Remove a middleware by name."""
        self._middlewares = [
            (p, m) for p, m in self._middlewares if m.name != name
        ]

    # ===================================================================
    # Execution
    # ===================================================================

    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        session_id: str = "",
    ) -> ToolResult:
        """Execute a registered tool with the full middleware pipeline.

        Steps:
        1. Look up tool, auto-load dependencies
        2. Health check & circuit breaker
        3. Rate limiting
        4. Pre-execution middleware hooks
        5. Parameter validation (deep-copied — caller's dict is untouched)
        6. Result cache lookup (if cacheable)
        7. Execute handler (with timeout)
        8. Post-execution middleware hooks
        9. Cache write, profiling, DB logging

        Returns:
            A ``ToolResult`` with success/output/metadata.
        """
        t0 = time.time()

        # --- 1. Look up tool ---
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=f"Tool '{name}' not found.",
                tool_name=name,
                duration_ms=(time.time() - t0) * 1000,
                error_category=ErrorCategory.VALIDATION,
                error_detail="tool_not_found",
            )

        # --- Auto-load dependencies ---
        dep_order = self._resolve_dependencies(tool)
        for dep_name in dep_order:
            if dep_name not in self._tools:
                logger.warning(
                    "Dependency '%s' of '%s' is not registered — skipping",
                    dep_name, name,
                )

        # --- 2. Health check / circuit breaker ---
        health = self._health.get(name, HealthStatus())
        if not health.healthy:
            # Check if cooldown has elapsed → attempt recovery.
            if (time.time() - health.disabled_at) >= self._CIRCUIT_COOLDOWN_SECONDS:
                health.healthy = True
                health.consecutive_failures = 0
                health.reason = "auto_recovered_after_cooldown"
                logger.info("Circuit breaker auto-recovered for '%s'", name)
            else:
                return ToolResult(
                    success=False,
                    output=(
                        f"Tool '{name}' is temporarily disabled due to "
                        f"repeated failures: {health.reason}. "
                        f"Will retry after cooldown."
                    ),
                    tool_name=name,
                    duration_ms=(time.time() - t0) * 1000,
                    error_category=ErrorCategory.RUNTIME,
                    error_detail="circuit_open",
                )

        # --- 3. Rate limiting ---
        if tool.rate_limit > 0:
            counter = self._rate_counters.get(name)
            if counter and counter.count() >= tool.rate_limit:
                retry_after = tool.rate_window
                return ToolResult(
                    success=False,
                    output=(
                        f"Rate limit exceeded for '{name}'. "
                        f"Limit: {tool.rate_limit} calls per {tool.rate_window}s. "
                        f"Retry after {retry_after}s."
                    ),
                    tool_name=name,
                    duration_ms=(time.time() - t0) * 1000,
                    error_category=ErrorCategory.UNKNOWN,
                    error_detail="rate_limited",
                )

        # --- 4. Pre-execution middleware ---
        # Deep-copy params so the caller's dict is NEVER mutated (v17 bug fix).
        working_params = copy.deepcopy(params)

        for _priority, mw in self._middlewares:
            try:
                result_params = await mw.before(tool, working_params, session_id)
            except Exception as exc:
                logger.exception("Middleware '%s' before-hook error", mw.name)
                continue
            if result_params is None:
                # Middleware aborted the call.
                return ToolResult(
                    success=False,
                    output=f"Execution aborted by middleware '{mw.name}'.",
                    tool_name=name,
                    duration_ms=(time.time() - t0) * 1000,
                    error_category=ErrorCategory.PERMISSION,
                    error_detail=f"middleware_abort:{mw.name}",
                    params_snapshot=working_params,
                )
            working_params = result_params

        # --- 5. Normalize alias parameters before validation ---
        # Map alias keys (marked with "Alias for X" description) to their primary key.
        _props = (tool.parameters_schema or {}).get("properties", {})
        for _pkey, _pval in list(working_params.items()):
            if _pkey in _props and _pval is not None:
                _desc = _props[_pkey].get("description", "")
                if _desc.startswith("Alias for ") and _pval:
                    _target = _desc[len("Alias for "):]
                    if _target in _props and _target not in working_params:
                        working_params[_target] = _pval
                        del working_params[_pkey]

        # --- 6. Parameter validation ---
        validation_errors = self._validate_params(tool, working_params)
        if validation_errors:
            return ToolResult(
                success=False,
                output=f"Parameter validation error: {'; '.join(validation_errors)}",
                tool_name=name,
                duration_ms=(time.time() - t0) * 1000,
                error_category=ErrorCategory.VALIDATION,
                error_detail="param_validation",
                params_snapshot=working_params,
            )

        # --- 7. Cache lookup ---
        cache_key: str | None = None
        if tool.cacheable:
            cache_key = self._make_cache_key(name, working_params)
            cached_entry = self._cache.get(cache_key)
            if cached_entry is not None:
                cached_ts, cached_result = cached_entry
                if (time.time() - cached_ts) < tool.cache_ttl:
                    self._cache_hits += 1
                    self._cache.move_to_end(cache_key)  # O(1) LRU refresh
                    duration_ms = (time.time() - t0) * 1000
                    result = ToolResult(
                        success=True,
                        output=cached_result,
                        tool_name=name,
                        duration_ms=duration_ms,
                        cached=True,
                        params_snapshot=working_params,
                    )
                    # Run post-middleware even on cache hit.
                    result = await self._run_post_middleware(tool, result, session_id)
                    return result
                else:
                    # Expired — remove it.
                    del self._cache[cache_key]
            self._cache_misses += 1

        # Record rate-limit usage BEFORE execution.
        if tool.rate_limit > 0:
            counter = self._rate_counters.get(name)
            if counter:
                counter.record()

        # --- 8. Execute handler (with timeout) ---
        # NOTE: tool_start is emitted by the reasoning engine (engine.py) before
        # calling registry.execute(). Don't emit it again here to avoid duplicates.

        output = ""
        success = True
        error_category = ErrorCategory.UNKNOWN
        error_detail = ""

        try:
            coro_or_val = tool.handler(working_params)
            if inspect.isawaitable(coro_or_val):
                output = await asyncio.wait_for(
                    coro_or_val, timeout=tool.timeout
                )
            else:
                output = str(coro_or_val)
            output = str(output)
        except asyncio.TimeoutError:
            success = False
            error_category = ErrorCategory.TIMEOUT
            error_detail = f"timeout_after_{tool.timeout}s"
            output = f"[Timeout] Tool '{name}' exceeded {tool.timeout}s limit."
        except asyncio.CancelledError:
            success = False
            error_category = ErrorCategory.RUNTIME
            error_detail = "cancelled"
            output = f"[Cancelled] Tool '{name}' was cancelled."
            # Update profiling, health, DB, and events before propagating
            duration_ms = (time.time() - t0) * 1000
            profile = self._profiles.get(name)
            if profile:
                profile.record(
                    success=False,
                    duration_ms=duration_ms,
                    params=working_params,
                    error_category=error_category,
                )
            health.consecutive_failures += 1
            health.total_failures += 1
            if health.consecutive_failures >= self._CIRCUIT_FAILURE_THRESHOLD:
                health.healthy = False
                health.disabled_at = time.time()
                health.reason = (
                    f"{health.consecutive_failures} consecutive failures "
                    f"(last: {error_detail})"
                )
                logger.warning(
                    "Circuit breaker OPEN for '%s': %s", name, health.reason,
                )
            try:
                await db.update_tool_profile(
                    tool_name=name,
                    success=False,
                    duration_ms=duration_ms,
                    output_len=len(output),
                )
            except Exception:
                pass
            try:
                await db.log_tool_execution(session_id, name, working_params, output, False, duration_ms)
            except Exception:
                pass
            await emit(
                "tool_end",
                name=name,
                params=working_params,
                success=False,
                output=_strip_dsml(output[:2000] if output else ""),
                output_full=_strip_dsml(output) if output and len(output) <= 10000 else None,
                session_id=session_id,
                cached=False,
            )
            raise  # propagate cancellation
        except Exception as exc:
            success = False
            error_category = self._categorize_error(exc)
            error_detail = f"{type(exc).__name__}: {exc}"
            output = f"[Error] {type(exc).__name__}: {exc}"

            # Run error middleware — may recover.
            for _priority, mw in self._middlewares:
                try:
                    recovery = await mw.on_error(
                        tool, exc, working_params, session_id,
                    )
                except Exception:
                    continue
                if recovery is not None:
                    output = recovery.output
                    success = recovery.success
                    error_category = recovery.error_category
                    error_detail = recovery.error_detail
                    break

        duration_ms = (time.time() - t0) * 1000

        result = ToolResult(
            success=success,
            output=output,
            tool_name=name,
            duration_ms=duration_ms,
            error_category=error_category,
            error_detail=error_detail,
            params_snapshot=working_params,
        )

        # --- 9. Post-execution middleware ---
        result = await self._run_post_middleware(tool, result, session_id)

        # --- 10. Cache write (only on success for cacheable tools) ---
        if success and tool.cacheable and cache_key is not None:
            self._cache[cache_key] = (time.time(), output)
            self._cache.move_to_end(cache_key)
            # O(1) LRU eviction: pop oldest until under max size.
            while len(self._cache) > self._CACHE_MAX_SIZE:
                self._cache.popitem(last=False)

        # --- 10. Profiling & health ---
        profile = self._profiles.get(name)
        if profile:
            profile.record(
                success=success,
                duration_ms=duration_ms,
                params=working_params,
                error_category=error_category,
            )

        # Update health / circuit breaker.
        if success:
            health.consecutive_failures = 0
            health.healthy = True
            health.reason = ""
        else:
            health.consecutive_failures += 1
            health.total_failures += 1
            if health.consecutive_failures >= self._CIRCUIT_FAILURE_THRESHOLD:
                health.healthy = False
                health.disabled_at = time.time()
                health.reason = (
                    f"{health.consecutive_failures} consecutive failures "
                    f"(last: {error_detail})"
                )
                logger.warning(
                    "Circuit breaker OPEN for '%s': %s", name, health.reason,
                )

        # Persist to DB (best-effort).
        try:
            await db.update_tool_profile(
                tool_name=name,
                success=success,
                duration_ms=duration_ms,
                output_len=len(output),
            )
        except Exception:
            pass

        await db.log_tool_execution(session_id, name, working_params, output, success, duration_ms)
        await emit(
            "tool_end",
            name=name,
            params=working_params,
            success=success,
            duration=duration_ms,
            output=_strip_dsml(output[:2000] if output else ""),  # Show more in thinking block
            output_full=_strip_dsml(output) if output and len(output) <= 10000 else None,
            session_id=session_id,
            cached=result.cached,
        )

        return result

    async def _run_post_middleware(
        self, tool: Tool, result: ToolResult, session_id: str,
    ) -> ToolResult:
        """Run post-execution middleware in priority order."""
        for _priority, mw in self._middlewares:
            try:
                maybe_result = await mw.after(tool, result, session_id)
            except Exception:
                logger.exception("Middleware '%s' after-hook error", mw.name)
                continue
            if maybe_result is not None:
                result = maybe_result
        return result

    # ===================================================================
    # Tool Composition / Chaining
    # ===================================================================

    async def compose(
        self,
        steps: list[ComposeStep],
        strategy: str = "sequential",
        session_id: str = "",
    ) -> ComposeResult:
        """Run multiple tools in a pipeline.

        Strategies:
        - ``"sequential"``: Run steps one-by-one, piping output forward.
        - ``"parallel"``: Run all steps concurrently, collect results.
        - ``"fan_out"``: Run steps concurrently (each with its own params),
          merge all results into ``outputs`` dict.

        For sequential mode, use ``"$output"`` in a step's ``params`` dict
        to receive the previous step's output under the key ``_prev_output``.
        """
        t0 = time.time()
        results: list[ToolResult] = []
        outputs: dict[str, str] = {}

        if strategy == "parallel":
            # Run all steps concurrently.
            coros = [
                self.execute(
                    step.tool_name,
                    step.params,
                    session_id=session_id,
                )
                for step in steps
            ]
            raw_results = await asyncio.gather(*coros, return_exceptions=True)

            # Process results — handle exceptions gracefully so one tool
            # failure doesn't kill the entire compose.
            results = []
            for raw in raw_results:
                if isinstance(raw, Exception):
                    # Wrap the exception as a failed ToolResult so the
                    # caller can inspect it alongside normal results.
                    results.append(ToolResult(
                        success=False,
                        output=f"[Error] {type(raw).__name__}: {raw}",
                        tool_name="",
                        error_category=ErrorCategory.RUNTIME,
                        error_detail=f"{type(raw).__name__}: {raw}",
                    ))
                else:
                    results.append(raw)

            for step, result in zip(steps, results):
                if isinstance(result, ToolResult) and result.success:
                    if step.output_key:
                        outputs[step.output_key] = result.output

            all_ok = all(r.success for r in results if isinstance(r, ToolResult))

        elif strategy == "fan_out":
            # Fan-out: run each step with its own params in parallel.
            coros = [
                self.execute(
                    step.tool_name,
                    step.params,
                    session_id=session_id,
                )
                for step in steps
            ]
            raw_results = await asyncio.gather(*coros, return_exceptions=True)

            # Process results — handle exceptions gracefully so one tool
            # failure doesn't kill the entire fan_out.
            results = []
            for raw in raw_results:
                if isinstance(raw, Exception):
                    results.append(ToolResult(
                        success=False,
                        output=f"[Error] {type(raw).__name__}: {raw}",
                        tool_name="",
                        error_category=self._categorize_error(raw) if isinstance(raw, Exception) else ErrorCategory.UNKNOWN,
                        error_detail=f"fan_out_exception:{type(raw).__name__}",
                    ))
                else:
                    results.append(raw)

            for step, result in zip(steps, results):
                if isinstance(result, ToolResult) and result.success:
                    outputs[step.tool_name] = result.output

            all_ok = all(r.success for r in results if isinstance(r, ToolResult))

        else:
            # Sequential — pipe output forward.
            all_ok = True
            prev_output = ""
            for step in steps:
                step_params = copy.deepcopy(step.params)
                # Replace "$output" placeholder with previous step's output.
                for k, v in step_params.items():
                    if v == "$output":
                        step_params[k] = prev_output
                    elif isinstance(v, str) and "$output" in v:
                        step_params[k] = v.replace("$output", prev_output)
                step_params["_prev_output"] = prev_output

                result = await self.execute(
                    step.tool_name, step_params, session_id=session_id,
                )
                results.append(result)

                if result.success:
                    prev_output = result.output
                    if step.output_key:
                        outputs[step.output_key] = result.output
                else:
                    all_ok = False
                    # Optionally break on first failure (default: continue).
                    break

        return ComposeResult(
            success=all_ok,
            results=results,
            outputs=outputs,
            total_duration_ms=(time.time() - t0) * 1000,
        )

    # ===================================================================
    # Semantic Tool Discovery
    # ===================================================================

    def discover(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ToolMatch]:
        """Find tools semantically similar to *query*.

        Uses TF-IDF cosine similarity between the query and each tool's
        description + tags + category.  Results are ranked by descending
        similarity score.

        Embeddings are cached and lazily recomputed when the tool set
        changes (i.e. after register / unregister).
        """
        if not query.strip():
            return []

        self._ensure_vocab()

        query_vec = self._text_to_vector(query)
        if not query_vec or not self._tools:
            return []

        matches: list[ToolMatch] = []

        for name, tool in self._tools.items():
            tool_text = f"{tool.description} {' '.join(tool.tags)} {tool.category}"
            tool_vec = self._desc_vectors.get(name)
            if tool_vec is None:
                tool_vec = self._text_to_vector(tool_text)
            if not tool_vec:
                continue
            score = self._cosine_sim(query_vec, tool_vec)
            if score > 0.0:
                # Build a human-readable reason.
                keywords = self._matching_keywords(query, tool_text)
                reason = f"matched keywords: {', '.join(keywords[:5])}" if keywords else "semantic match"
                matches.append(ToolMatch(tool=tool, score=round(score, 4), reason=reason))

        # Sort by score descending, take top_k.
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def _ensure_vocab(self) -> None:
        """Build (or rebuild) the TF-IDF vocabulary from all tool descriptions."""
        if self._vocab_built:
            return

        documents: dict[str, str] = {}
        for name, tool in self._tools.items():
            documents[name] = f"{tool.description} {' '.join(tool.tags)} {tool.category}"

        if not documents:
            return

        # Tokenize all documents.
        all_tokens: list[str] = []
        doc_token_sets: dict[str, set[str]] = {}
        for name, text in documents.items():
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
            doc_token_sets[name] = set(tokens)

        # Build vocabulary (words appearing in >= 1 doc).
        self._vocab = sorted(set(all_tokens))
        n_docs = len(documents)

        # Compute IDF: log(N / df) where df = number of docs containing term.
        self._idf = {}
        for term in self._vocab:
            df = sum(1 for tokens in doc_token_sets.values() if term in tokens)
            self._idf[term] = math.log(n_docs / (1 + df)) + 1.0  # smoothed IDF

        # Pre-compute and cache description vectors.
        self._desc_vectors = {}
        for name, text in documents.items():
            self._desc_vectors[name] = self._text_to_vector(text)

        self._vocab_built = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        # Split on non-alphanumeric characters.
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert text to a TF-IDF vector aligned with ``_vocab``."""
        if not self._vocab:
            return []
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(self._vocab)

        # Term frequency.
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        # Normalize TF.
        max_tf = max(tf.values())
        vec = []
        for term in self._vocab:
            norm_tf = tf.get(term, 0) / max_tf if max_tf > 0 else 0.0
            vec.append(norm_tf * self._idf.get(term, 0.0))

        return vec

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two equal-length vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _matching_keywords(self, query: str, tool_text: str) -> list[str]:
        """Return tokens that appear in both query and tool text, ranked by IDF."""
        q_tokens = set(self._tokenize(query))
        t_tokens = set(self._tokenize(tool_text.lower()))
        common = q_tokens & t_tokens
        # Sort by IDF (higher = more distinctive).
        scored = [(t, self._idf.get(t, 0.0)) for t in common]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored]

    # ===================================================================
    # Caching helpers
    # ===================================================================

    @staticmethod
    def _make_cache_key(name: str, params: dict[str, Any]) -> str:
        """Build a deterministic cache key from tool name + params.

        Uses SHA-256 for uniform distribution and collision resistance.
        """
        param_str = json.dumps(params, sort_keys=True, default=str)
        raw = f"{name}:{param_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Return cache hit/miss/size statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "size": len(self._cache),
            "max_size": self._CACHE_MAX_SIZE,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / total, 3) if total > 0 else 0.0,
        }

    def clear_cache(self) -> int:
        """Clear the entire result cache.  Returns number of entries evicted."""
        count = len(self._cache)
        self._cache.clear()
        return count

    # ===================================================================
    # Parameter validation
    # ===================================================================

    @staticmethod
    def _validate_params(tool: Tool, params: dict[str, Any]) -> list[str]:
        # v20: Optimized — early return for tools with no required params
        """Validate *params* against the tool's JSON Schema.

        Returns a list of human-readable error strings (empty = valid).

        Note: ``params`` may be modified in-place for type coercion since
        we already deep-copied in ``execute()``.
        """
        errors: list[str] = []
        schema = tool.parameters_schema or {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # v20: Early return — skip validation if no params provided and no required fields
        if not required and not params:
            return errors

        # Check required parameters.
        for req in required:
            val = params.get(req)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(f"'{req}' is required")

        # Type coercion for provided parameters.
        for key, value in params.items():
            if key not in properties:
                continue
            expected_type = properties[key].get("type")
            if not expected_type or value is None:
                continue
            try:
                if expected_type == "integer" and not isinstance(value, (int, float)):
                    params[key] = int(value)
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    params[key] = float(value)
                elif expected_type == "string" and not isinstance(value, str):
                    params[key] = str(value)
                elif expected_type == "boolean" and not isinstance(value, bool):
                    params[key] = str(value).lower() in ("true", "1", "yes")
                elif expected_type == "array" and isinstance(value, str):
                    params[key] = json.loads(value)
            except (ValueError, TypeError, json.JSONDecodeError):
                errors.append(f"'{key}' should be {expected_type}")

        return errors

    # ===================================================================
    # Error categorization
    # ===================================================================

    @staticmethod
    def _categorize_error(exc: Exception) -> ErrorCategory:
        """Classify an exception into a high-level error category."""
        exc_name = type(exc).__name__.lower()
        exc_msg = str(exc).lower()

        if "timeout" in exc_name or "timeout" in exc_msg:
            return ErrorCategory.TIMEOUT
        if "valid" in exc_name or "valid" in exc_msg:
            return ErrorCategory.VALIDATION
        if any(kw in exc_msg for kw in (
            "connection", "network", "socket", "dns", "resolve", "refused",
        )):
            return ErrorCategory.NETWORK
        if any(kw in exc_msg for kw in (
            "permission", "forbidden", "unauthorized", "access denied", "403",
        )):
            return ErrorCategory.PERMISSION
        return ErrorCategory.RUNTIME

    # ===================================================================
    # Health checking
    # ===================================================================

    def get_health(self, name: str) -> HealthStatus | None:
        """Return the health status for a tool, or ``None``."""
        return self._health.get(name)

    def check_health(self, name: str) -> HealthStatus:
        """Perform a health check for a tool (lightweight — just reads state).

        For more thorough external health checks, integrate with a
        periodic scheduler that calls this.
        """
        health = self._health.get(name, HealthStatus())
        health.last_check = time.time()

        # If currently unhealthy, check cooldown for auto-recovery.
        if not health.healthy:
            if (time.time() - health.disabled_at) >= self._CIRCUIT_COOLDOWN_SECONDS:
                health.healthy = True
                health.consecutive_failures = 0
                health.reason = "auto_recovered_after_cooldown"
                logger.info("Health check auto-recovered '%s'", name)

        return health

    def reset_health(self, name: str) -> bool:
        """Manually reset the circuit breaker for a tool.  Returns ``True`` if the tool exists."""
        if name in self._health:
            self._health[name] = HealthStatus()
            return True
        return False

    # ===================================================================
    # Profiling
    # ===================================================================

    def get_profiles(self) -> dict[str, ToolProfile]:
        """Return a snapshot of all in-memory tool profiles."""
        return {name: copy.deepcopy(profile) for name, profile in self._profiles.items()}

    def get_profile(self, name: str) -> ToolProfile | None:
        """Return the profile for a specific tool."""
        return copy.deepcopy(self._profiles.get(name))

    # ===================================================================
    # Intelligence summary (for system prompt injection)
    # ===================================================================

    async def intelligence_summary(self) -> str:
        """Generate a tool intelligence summary for the system prompt.

        Includes: most reliable tools, tools to use with caution,
        learned strategies, health warnings, and profiling highlights.
        """
        # Merge DB profiles with in-memory profiles.
        profiles = await db.get_tool_profiles()
        if not profiles and not self._profiles:
            return ""

        lines = ["# Tool Intelligence (learned from usage)"]

        # Top performers.
        top = [
            p for p in profiles
            if (p.get("effectiveness_score") or 0) >= 0.8
        ]
        if top:
            names = ", ".join(
                f"{p['tool_name']} ({p['effectiveness_score']:.0%})" for p in top[:8]
            )
            lines.append(f"- **Most reliable**: {names}")

        # Tools to use with caution.
        caution = [
            p for p in profiles
            if (p.get("success_rate") or 1) < 0.6
            and (p.get("total_calls") or 0) >= 3
        ]
        if caution:
            c_names = ", ".join(
                f"{p['tool_name']} ({p['success_rate']:.0%} success)"
                for p in caution[:5]
            )
            lines.append(f"- **Use with caution**: {c_names}")

        # Learned strategies.
        learned = [p for p in profiles if p.get("learned_strategy")]
        if learned:
            for p in learned[:3]:
                lines.append(
                    f"- **{p['tool_name']}** tip: {p['learned_strategy'][:100]}"
                )

        # Health warnings.
        unhealthy = [
            (name, h) for name, h in self._health.items()
            if not h.healthy
        ]
        if unhealthy:
            for name, h in unhealthy[:5]:
                lines.append(f"- **UNHEALTHY**: '{name}' — {h.reason}")

        # Profiling highlights (fastest/slowest tools).
        active_profiles = {
            n: p for n, p in self._profiles.items() if p.total_calls >= 3
        }
        if active_profiles:
            fastest = min(
                active_profiles.items(), key=lambda x: x[1].p50
            )
            slowest = max(
                active_profiles.items(), key=lambda x: x[1].p99
            )
            lines.append(
                f"- **Fastest tool**: {fastest[0]} (p50={fastest[1].p50:.0f}ms)"
            )
            lines.append(
                f"- **Slowest tool**: {slowest[0]} (p99={slowest[1].p99:.0f}ms)"
            )

        # Cache stats.
        stats = self.cache_stats
        if stats["hits"] + stats["misses"] > 0:
            lines.append(
                f"- **Cache**: {stats['hit_rate']:.0%} hit rate "
                f"({stats['size']}/{stats['max_size']} entries)"
            )

        return "\n".join(lines)

    # ===================================================================
    # OpenAI schema generation
    # ===================================================================

    async def schemas(
        self, enabled_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # v20: Lazy schema cache — avoids re-converting on every LLM call
        """Generate OpenAI function-calling schemas for enabled tools.

        Skips unhealthy tools (circuit breaker open).
        When called with default (None) names, uses a cached result.
        """
        names = enabled_names
        if names is None and not self._schemas_dirty and self._schemas_cache is not None:
            return self._schemas_cache

        names = names if names is not None else list(self._tools.keys())
        result = []
        for name in names:
            tool = self._tools.get(name)
            if not tool:
                continue
            # Skip unhealthy tools.
            health = self._health.get(name)
            if health and not health.healthy:
                continue
            result.append(await tool.to_openai_format())

        # Cache only when called with default (all tools) names.
        if enabled_names is None:
            self._schemas_cache = result
            self._schemas_dirty = False

        return result

    # ===================================================================
    # Strategic ranking (backward-compatible API from v17)
    # ===================================================================

    async def strategic_tool_ranking(self, task_context: str = "") -> list[dict]:
        """Return tools ranked by likely effectiveness.

        Combines DB profiles and in-memory profiles. Useful for
        agent planning and tool selection.
        """
        profiles = await db.get_tool_profiles()
        if not profiles:
            return []

        ranked: list[dict] = []
        for p in profiles:
            name = p.get("tool_name", "")
            tool = self._tools.get(name)
            if not tool:
                continue

            score = p.get("effectiveness_score") or 0.5
            success_rate = p.get("success_rate") or 0.5
            total_calls = p.get("total_calls") or 0

            # In-memory profile overrides if available.
            mem_profile = self._profiles.get(name)
            if mem_profile and mem_profile.total_calls > 0:
                score = mem_profile.effectiveness_score
                success_rate = mem_profile.success_rate
                total_calls = mem_profile.total_calls

            confidence = min(total_calls / 10.0, 1.0)
            adjusted = score * 0.6 + confidence * 0.4

            ranked.append({
                "name": name,
                "category": tool.category,
                "description": tool.description[:80],
                "effectiveness": round(score, 2),
                "success_rate": round(success_rate, 2),
                "total_calls": total_calls,
                "adjusted_score": round(adjusted, 2),
                "avg_duration_ms": round(
                    mem_profile.avg_duration_ms if mem_profile
                    else (p.get("avg_duration_ms") or 0), 0
                ),
                "p50_ms": round(mem_profile.p50, 1) if mem_profile else 0.0,
                "p99_ms": round(mem_profile.p99, 1) if mem_profile else 0.0,
                "learned_strategy": p.get("learned_strategy", ""),
                "healthy": bool(
                    self._health.get(name, HealthStatus()).healthy
                ),
            })

        ranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return ranked

    # ===================================================================
    # Tool-task affinity (backward-compatible from v17)
    # ===================================================================

    async def record_tool_affinity(
        self,
        tool_name: str,
        task_type: str,
        success: bool,
        duration_ms: float,
        session_id: str = "",
    ) -> None:
        """Record which tools work best for which task types."""
        try:
            await db.record_tool_affinity(
                tool_name=tool_name,
                task_type=task_type,
                success=success,
                duration_ms=duration_ms,
            )
        except Exception:
            pass

    async def get_best_tools_for_task(self, task_type: str, top_k: int = 5) -> list[dict]:
        """Get the most effective tools for a given task type."""
        try:
            return await db.get_tool_affinities(task_type=task_type, top_k=top_k)
        except Exception:
            return []

    async def tool_intelligence_summary(self) -> str:
        """Alias for ``intelligence_summary`` (v17 backward compatibility)."""
        return await self.intelligence_summary()

    # ===================================================================
    # Backward-compatible aliases
    # ===================================================================

    def get(self, name: str) -> Tool | None:
        """Alias for ``get_tool`` (v17 compat)."""
        return self.get_tool(name)

    def all(self) -> list[Tool]:
        """Alias for ``all_tools`` (v17 compat)."""
        return self.all_tools()

    # ===================================================================
    # v20: Batch execution with dependency-aware parallel execution
    # ===================================================================

    async def execute_batch(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        session_id: str = "",
    ) -> list[ToolResult]:
        # v20: Batch execution helper — runs independent tools in parallel
        """Execute multiple tools efficiently, respecting dependencies.

        Args:
            calls: List of (tool_name, params) tuples.
            session_id: Optional session identifier.

        Returns:
            List of ToolResult in the same order as *calls*.

        Tools are executed concurrently if they have no inter-dependencies.
        If a tool depends on others (via ``depends_on``), those are run first.
        """
        if not calls:
            return []

        # Build index and dependency graph
        indexed: list[tuple[int, str, dict[str, Any]]] = [(i, n, p) for i, (n, p) in enumerate(calls)]
        results: list[ToolResult | None] = [None] * len(calls)
        completed: set[str] = set()
        remaining = list(indexed)
        max_rounds = len(calls) + 1  # prevent infinite loops

        for _round in range(max_rounds):
            if not remaining:
                break

            # Partition into ready (deps satisfied) vs blocked
            ready: list[tuple[int, str, dict[str, Any]]] = []
            blocked: list[tuple[int, str, dict[str, Any]]] = []
            for idx, name, params in remaining:
                tool = self._tools.get(name)
                deps = tool.depends_on if tool else []
                if all(d in completed for d in deps):
                    ready.append((idx, name, params))
                else:
                    blocked.append((idx, name, params))

            if ready:
                # Execute all ready calls in parallel
                coros = [self.execute(name, params, session_id=session_id) for _, name, params in ready]
                batch_results = await asyncio.gather(*coros, return_exceptions=True)
                for (idx, tool_name, _), raw in zip(ready, batch_results):
                    if isinstance(raw, Exception):
                        results[idx] = ToolResult(
                            success=False,
                            output=f"[Error] {type(raw).__name__}: {raw}",
                            tool_name=tool_name,
                            error_category=self._categorize_error(raw),
                            error_detail=f"batch_exception:{type(raw).__name__}",
                        )
                    else:
                        results[idx] = raw
                    completed.add(calls[idx][0])

            remaining = blocked

        # Fill any remaining blocked items as errors
        for idx, name, _ in remaining:
            results[idx] = ToolResult(
                success=False,
                output=f"Tool '{name}' could not execute due to unresolved dependencies.",
                tool_name=name,
                error_category=ErrorCategory.VALIDATION,
                error_detail="dependency_unresolved",
            )

        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Singleton (created at module level)
# ---------------------------------------------------------------------------

REGISTRY = ToolRegistry()
"""Global tool registry singleton."""

# Reference for use inside Tool.to_openai_format (avoids circular import).
_REGISTRY_SINGLETON_REF = REGISTRY


# ---------------------------------------------------------------------------
# @tool decorator (backward-compatible)
# ---------------------------------------------------------------------------

def tool(
    name: str,
    description: str,
    parameters_schema: dict[str, Any],
    risk: str = "low",
    category: str = "general",
    cacheable: bool = False,
    cache_ttl: int = 300,
    rate_limit: int = 0,
    rate_window: int = 60,
    depends_on: list[str] | None = None,
    timeout: int = 30,
    version: str = "1.0",
    tags: list[str] | None = None,
) -> Callable[[Handler], Handler]:
    """Decorator to register an async function as a tool.

    All v18 parameters are supported with backward-compatible defaults.

    Usage::

        @tool("search", "Search the web", {"type": "object", "properties": {"query": {"type": "string"}}})
        async def search_handler(params: dict) -> str:
            return f"Results for {params['query']}"
    """

    def decorator(fn: Handler) -> Handler:
        REGISTRY.register(
            Tool(
                name=name,
                description=description,
                parameters_schema=parameters_schema,
                handler=fn,
                risk=risk,
                category=category,
                cacheable=cacheable,
                cache_ttl=cache_ttl,
                rate_limit=rate_limit,
                rate_window=rate_window,
                depends_on=depends_on or [],
                timeout=timeout,
                version=version,
                tags=tags or [],
            )
        )
        return fn

    return decorator
