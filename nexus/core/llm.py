"""
LLM client — v21: NVIDIA NIM + robust multi-provider support.

What's new in v21:
  - "provider" config key: "openrouter" (default) | "openai" | "together" | "nvidia" | "custom" | "groq"
  - "nvidia" is now a first-class provider with its own base URL, API key, model
    catalog, auto-routing, and fallback chain.  Previously NVIDIA NIM was treated
    as a "custom" provider, causing 404 errors when the default model
    (anthropic/claude-3.5-sonnet) was sent to the NVIDIA API.
  - NVIDIA NIM model catalog: short names (e.g. "llama-3.1-70b-instruct") are
    auto-resolved to fully-qualified names (e.g. "meta/llama-3.1-70b-instruct").
  - Provider/model mismatch guard extended to "nvidia" provider: non-NVIDIA
    models are automatically replaced with the default NVIDIA model.
  - All v20 features preserved: retry, circuit breaker, fallback chain,
    connection pooling, token tracking, cost-aware routing, streaming.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
import traceback
import weakref
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from .. import config
from ..events import emit


_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 30.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_DEFAULT_FALLBACK_CHAIN = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct:free",
]

_NVIDIA_DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

_NVIDIA_FALLBACK_CHAIN = [
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
]

_CIRCUIT_FAILURE_THRESHOLD = 3
_CIRCUIT_COOLDOWN_SECONDS = 60


# ── NVIDIA NIM Model Catalog ─────────────────────────────────────────────────
# Maps short/user-friendly model names to their fully-qualified NVIDIA NIM IDs.
# When provider is "nvidia" and a model name does not contain "/", we look it
# up here.  If the model is already fully-qualified (contains "/"), it is used
# as-is unless it belongs to a different provider (e.g. "anthropic/...").

_NVIDIA_MODELS: dict[str, str] = {
    "llama-3.1-nemotron-70b-instruct": "nvidia/llama-3.1-nemotron-70b-instruct",
    "llama-3.1-405b-instruct": "meta/llama-3.1-405b-instruct",
    "llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
    "llama3-8b-instruct": "meta/llama3-8b-instruct",
    "llama3-70b-instruct": "meta/llama3-70b-instruct",
    "mistral-large": "nv-mistralai/mistral-large-24b-instruct-v1",
    "mistral-7b-instruct": "mistralai/mistral-7b-instruct-v0.3",
    "mixtral-8x7b-instruct": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b-instruct": "mistralai/mixtral-8x22b-instruct-v0.1",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
    "codellama-34b-instruct": "codellama/codellama-34b-instruct",
    "deepseek-r1": "deepseek-ai/deepseek-r1",
    "qwen2.5-72b-instruct": "qwen/qwen2.5-72b-instruct",
    "starcoder2-15b": "bigcode/starcoder2-15b",
    "arctic": "snowflake/arctic",
}

# Set of prefixes that indicate a fully-qualified NVIDIA NIM model name
_NVIDIA_MODEL_PREFIXES = frozenset({
    "nvidia/", "meta/", "mistralai/", "nv-mistralai/",
    "google/", "codellama/", "deepseek-ai/", "qwen/",
    "bigcode/", "snowflake/",
})

# Prefixes that are clearly NOT NVIDIA models (belong to other providers)
_NON_NVIDIA_PREFIXES = frozenset({
    "anthropic/", "openai/", "cohere/", "meta-llama/",
})


def _resolve_nvidia_model(model: str | None) -> str:
    """Resolve a model name for the NVIDIA NIM provider.

    1. If model is None or empty, return the default NVIDIA model.
    2. If the model name is clearly from another provider (e.g. "anthropic/claude-3.5-sonnet"),
       emit a warning and fall back to the default NVIDIA model.
    3. If the model name already has a recognized NVIDIA prefix (e.g. "nvidia/llama-3.1-nemotron-70b-instruct"),
       return it as-is.
    4. If the model name is a short name (no "/"), look it up in _NVIDIA_MODELS.
    5. If not found in the catalog, return as-is and let the API validate it.
    """
    if not model:
        return _NVIDIA_DEFAULT_MODEL

    model_lower = model.lower()

    # Check if model is clearly from a different provider
    for prefix in _NON_NVIDIA_PREFIXES:
        if model_lower.startswith(prefix):
            # This model doesn't exist on NVIDIA NIM — fall back
            import asyncio as _aio
            try:
                loop = _aio.get_running_loop()
                loop.create_task(emit(
                    "warning", source="llm.nvidia_model_fallback",
                    original_model=model,
                    fallback_model=_NVIDIA_DEFAULT_MODEL,
                    message=f"Model '{model}' is not available on NVIDIA NIM — "
                            f"falling back to '{_NVIDIA_DEFAULT_MODEL}'",
                ))
            except RuntimeError:
                pass  # No event loop — can't emit async event here
            return _NVIDIA_DEFAULT_MODEL

    # Check if model already has a recognized NVIDIA prefix
    for prefix in _NVIDIA_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return model  # Already fully-qualified — use as-is

    # Try to resolve short name from catalog
    if "/" not in model:
        resolved = _NVIDIA_MODELS.get(model_lower)
        if resolved:
            return resolved
        # Also try exact (case-sensitive) match
        resolved = _NVIDIA_MODELS.get(model)
        if resolved:
            return resolved

    # Model has a "/" but doesn't match any known NVIDIA prefix.
    # Could be a user-provided custom model ID.  Pass it through and let the
    # API validate it.  If it's something like "meta-llama/llama-3.1-8b-instruct"
    # (OpenRouter naming), it will likely 404 on NVIDIA NIM, but we let it fail
    # naturally rather than silently replacing it.
    return model


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd
        return self


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)
    tools_stripped: bool = False  # True when model doesn't support tools API


GLOBAL_USAGE = Usage()
_usage_loaded = False  # lazily restored from DB on first increment

_shared_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _persist_usage(usage: Usage) -> None:
    """Fire-and-forget: write usage increment to the DB (single row UPSERT)."""
    try:
        from ..memory import db
        await db.increment_global_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
        )
    except Exception:
        pass  # never block the caller


async def load_global_usage() -> None:
    """Load GLOBAL_USAGE from the DB (call once at startup).

    Idempotent — safe to call multiple times.  Subsequent calls are no-ops
    unless the DB row was written after the last load.
    """
    global GLOBAL_USAGE, _usage_loaded
    if _usage_loaded:
        return
    _usage_loaded = True
    try:
        from ..memory import db
        row = await db.get_global_usage()
        GLOBAL_USAGE.prompt_tokens = row["prompt_tokens"]
        GLOBAL_USAGE.completion_tokens = row["completion_tokens"]
        GLOBAL_USAGE.total_tokens = row["total_tokens"]
        GLOBAL_USAGE.cost_usd = row["cost_usd"]
    except Exception:
        pass  # first boot or DB not ready yet — start from zero


async def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        async with _client_lock:
            if _shared_client is None or _shared_client.is_closed:
                _shared_client = httpx.AsyncClient(
                    # No total timeout — outer asyncio.wait_for(600s) in server.py
                    # is the sole timeout authority, preventing race conditions
                    # between httpx and asyncio timeouts.
                    timeout=httpx.Timeout(None, connect=30.0),
                    limits=httpx.Limits(max_connections=50, max_keepalive_connections=10, keepalive_expiry=30.0),
                    http2=False,  # Disable HTTP/2 - causes issues with some providers
                )
    return _shared_client


async def shutdown_client() -> None:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


# ── Provider resolution ───────────────────────────────────────────────────────

def _get_provider() -> str:
    """Return active provider: 'openrouter' | 'openai' | 'together' | 'nvidia' | 'custom' | 'groq'."""
    return (config.get("provider") or "openrouter").lower().strip()


def _get_base_url() -> str:
    """Return the base URL for the active provider."""
    provider = _get_provider()
    if provider == "openai":
        return (config.get("openai_base_url") or "https://api.openai.com/v1").rstrip("/")
    if provider == "together":
        return (config.get("together_base_url") or "https://api.together.xyz/v1").rstrip("/")
    if provider == "nvidia":
        return (config.get("nvidia_base_url") or "https://integrate.api.nvidia.com/v1").rstrip("/")
    if provider == "groq":
        return (config.get("groq_base_url") or "https://api.groq.com/openai/v1").rstrip("/")
    if provider == "custom":
        url = config.get("custom_base_url")
        # Default to localhost if custom provider is selected but no URL set
        if not url:
            url = "127.0.0.1:11434"
        if not url.startswith(("http://", "https://")):
            # Convert localhost to 127.0.0.1 to avoid DNS issues on Windows
            if "localhost" in url.lower():
                url = url.replace("localhost", "127.0.0.1", 1)
            url = "http://" + url
        return url.rstrip("/")
    # openrouter (default)
    return (config.get("openrouter_base_url") or "https://openrouter.ai/api/v1").rstrip("/")


def _get_provider_for_model(model: str | None = None) -> str:
    """Determine which provider to use for a specific model.

    v21: Now handles "nvidia" as a first-class provider. When provider is
    "nvidia", always returns "nvidia" — the model name is resolved separately
    by _resolve_nvidia_model().
    """
    active = _get_provider()
    if active == "together":
        return "together"
    if active == "groq":
        return "groq"
    if active == "openai":
        return "openai"
    if active == "nvidia":
        return "nvidia"
    if active == "custom":
        if model and model.startswith("nvidia/"):
            return "custom"
        if model and "/" in model:
            # model looks like openrouter format but provider is custom
            # -> if it's clearly an OpenRouter-only model, route to openrouter
            model_lower = model.lower()
            if any(x in model_lower for x in ["anthropic/", "openai/", "cohere/", "google/"]):
                return "openrouter"
        return "custom"
    # Default provider is openrouter
    if model and "/" in model and not model.startswith("nvidia/"):
        return "openrouter"
    return "openrouter"


def _get_base_url_for_model(model: str | None = None) -> str:
    """Get base URL for a specific model - auto-routes based on model name.

    v21: Now handles "nvidia" as a first-class provider.
    v26: Custom provider no longer falls back to NVIDIA URL — raises a clear
         error or falls back to OpenRouter default when no custom_base_url is set.
    """
    resolved_provider = _get_provider_for_model(model)
    if resolved_provider == "together":
        return (config.get("together_base_url") or "https://api.together.xyz/v1").rstrip("/")
    if resolved_provider == "groq":
        return (config.get("groq_base_url") or "https://api.groq.com/openai/v1").rstrip("/")
    if resolved_provider == "openai":
        return (config.get("openai_base_url") or "https://api.openai.com/v1").rstrip("/")
    if resolved_provider == "nvidia":
        return (config.get("nvidia_base_url") or "https://integrate.api.nvidia.com/v1").rstrip("/")
    if resolved_provider == "custom":
        custom_url = config.get("custom_base_url")
        if custom_url:
            return custom_url.rstrip("/")
        # No custom_base_url configured — fall back to OpenRouter default URL
        # rather than the nonsensical NVIDIA URL. The user chose "custom"
        # provider but hasn't configured a URL, so OpenRouter is a safer
        # fallback than sending requests to an NVIDIA endpoint that would
        # reject them with auth errors.
        return (config.get("openrouter_base_url") or "https://openrouter.ai/api/v1").rstrip("/")
    # openrouter (default)
    return (config.get("openrouter_base_url") or "https://openrouter.ai/api/v1").rstrip("/")


def _get_api_key_for_model(model: str | None = None) -> str:
    """Return the API key for a specific model - auto-routes based on model name.

    v21: Now handles "nvidia" as a first-class provider.
    """
    resolved_provider = _get_provider_for_model(model)
    if resolved_provider == "together":
        return config.get("together_api_key") or ""
    if resolved_provider == "groq":
        return config.get("groq_api_key") or ""
    if resolved_provider == "openai":
        return config.get("openai_api_key") or ""
    if resolved_provider == "nvidia":
        return config.get("nvidia_api_key") or ""
    if resolved_provider == "custom":
        return config.get("custom_api_key") or ""
    # openrouter
    return config.get("openrouter_api_key") or ""


def _get_api_key() -> str:
    """Return the API key for the active provider."""
    provider = _get_provider()
    if provider == "openai":
        return config.get("openai_api_key") or ""
    if provider == "groq":
        return config.get("groq_api_key") or ""
    if provider == "together":
        return config.get("together_api_key") or ""
    if provider == "nvidia":
        return config.get("nvidia_api_key") or ""
    if provider == "custom":
        return config.get("custom_api_key") or ""
    return config.get("openrouter_api_key") or ""


def _validate_api_key(model: str | None = None) -> str:
    """Return validated API key. Raises RuntimeError on missing/placeholder.

    v21: Now validates the key for the model's resolved provider, including
    "nvidia" as a first-class provider.
    """
    resolved_provider = _get_provider_for_model(model) if model else _get_provider()
    api_key = _get_api_key_for_model(model) if model else _get_api_key()

    # Custom providers may not need a key (e.g. local LM Studio)
    if resolved_provider == "custom":
        allow_no_key = config.get("custom_no_auth", False)
        if allow_no_key:
            return api_key or ""

    if not api_key:
        provider_label = {
            "openai": "OpenAI",
            "together": "Together AI",
            "groq": "Groq",
            "nvidia": "NVIDIA NIM",
            "custom": "Custom endpoint",
        }.get(resolved_provider, "OpenRouter")
        raise RuntimeError(
            f"{provider_label} API key not set. Open Settings -> API Credentials and add your key."
        )

    _placeholders = (
        "your-api-key-here", "your_api_key_here", "placeholder",
        "replace-me", "your-key-here", "sk-placeholder",
        "sk-xxxx", "sk-xxxxxxxx",
        "***", "****", "*****", "xxxxxx", "xxxx",
        "your-openrouter-key", "your-nvidia-key", "your-custom-key",
        "nvapi-xxxx", "nvapi-placeholder", "your-together-key",
    )
    if api_key.lower().strip() in _placeholders or api_key.strip() == "":
        raise RuntimeError(
            "API key is still a placeholder. Open Settings -> API Credentials and set your real key."
        )
    return api_key


def _headers(
    *,
    api_key: str | None = None,
    provider: str | None = None,
) -> dict[str, str]:
    """Build Authorization headers for the given provider and API key.

    v21: Now handles "nvidia" as a first-class provider.
    NVIDIA NIM requires: Authorization: Bearer nvapi-xxxxx
    NVIDIA NIM accepts: content-type: application/json
    The invoke-model header is NOT needed for the chat/completions endpoint.
    """
    effective_provider = provider or _get_provider()
    effective_key      = api_key or _get_api_key()

    headers = {"Content-Type": "application/json"}

    if effective_provider == "nvidia":
        # NVIDIA NIM: Bearer auth with nvapi- prefixed key
        if effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"
        # No extra headers needed — invoke-model is NOT for chat/completions
    elif effective_provider == "custom":
        auth_type = (config.get("custom_auth_type") or "bearer").lower()
        if auth_type == "basic":
            import base64
            encoded = base64.b64encode(effective_key.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        elif auth_type == "apikey":
            headers["X-API-Key"] = effective_key
        elif effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"
        # Merge any extra custom headers
        extra = config.get("custom_extra_headers")
        if isinstance(extra, dict):
            headers.update(extra)
    elif effective_provider == "openai":
        if effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"
        org = config.get("openai_org")
        if org:
            headers["OpenAI-Organization"] = org
    elif effective_provider == "together":
        if effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"
    elif effective_provider == "groq":
        if effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"
    else:
        # OpenRouter (default)
        if effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"
        headers["HTTP-Referer"] = "https://nexus.local"
        headers["X-Title"] = "Hyper Nexus Agent"

    return headers


# ── Token ceiling ─────────────────────────────────────────────────────────────

_DEFAULT_TOKEN_LIMITS: dict[str, int] = {"default": 8192, "nvidia": 4096}


def _get_token_ceiling(model: str) -> int:
    limits = config.get("model_token_limits", _DEFAULT_TOKEN_LIMITS)
    if not isinstance(limits, dict):
        limits = _DEFAULT_TOKEN_LIMITS
    model_lower = (model or "").lower()
    if model in limits:
        return int(limits[model])
    for key, value in limits.items():
        if key == "default":
            continue
        if key in model_lower:
            return int(value)
    return int(limits.get("default", 8192))


def _clamp_max_tokens(value: int | None, model: str = "") -> int:
    if value is None or value <= 0:
        value = int(config.get("max_tokens", 1024))
    ceiling = _get_token_ceiling(model)
    return max(64, min(int(value), ceiling))


# ── Circuit breaker ───────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    model: str
    failures: int = 0
    opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self.failures >= _CIRCUIT_FAILURE_THRESHOLD:
            return time.time() - self.opened_at < _CIRCUIT_COOLDOWN_SECONDS
        return False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= _CIRCUIT_FAILURE_THRESHOLD:
            self.opened_at = time.time()

    def record_success(self) -> None:
        self.failures = 0


_circuits: dict[str, CircuitBreaker] = {}


def _get_circuit(model: str) -> CircuitBreaker:
    if model not in _circuits:
        _circuits[model] = CircuitBreaker(model)
    return _circuits[model]


def get_circuit_breaker_status() -> dict:
    return {
        m: {"failures": cb.failures, "is_open": cb.is_open, "opened_at": cb.opened_at}
        for m, cb in _circuits.items()
    }


def reset_circuit_breaker(model: str | None = None) -> int:
    if model:
        if model in _circuits:
            _circuits[model] = CircuitBreaker(model)
            return 1
        return 0
    count = len(_circuits)
    _circuits.clear()
    return count


# ── Session tracker ───────────────────────────────────────────────────────────

class SessionTracker:
    def __init__(self):
        self._data: dict[str, Usage] = {}

    def record(self, session_id: str, usage: Usage) -> None:
        if session_id not in self._data:
            self._data[session_id] = Usage()
        self._data[session_id] += usage

    def get(self, session_id: str) -> Usage:
        return self._data.get(session_id, Usage())


_session_tracker = SessionTracker()


def get_session_usage(session_id: str) -> Usage:
    return _session_tracker.get(session_id)


# ── Fallback chain ────────────────────────────────────────────────────────────

def _get_fallback_chain(model: str | None) -> list[str]:
    """Return the fallback chain for the current provider and model.

    v21: When the active provider is "nvidia", returns the NVIDIA-specific
    fallback chain instead of the default OpenRouter-oriented chain.
    """
    active_provider = _get_provider()

    if active_provider == "nvidia":
        default = model or config.get("default_model") or _NVIDIA_DEFAULT_MODEL
        # Resolve the default model through the NVIDIA catalog
        default = _resolve_nvidia_model(default)
        chain = config.get("model_fallback_chain", [])
        if not chain:
            # Use the built-in NVIDIA fallback chain
            chain = list(_NVIDIA_FALLBACK_CHAIN)
        if default not in chain:
            chain = [default] + list(chain)
        return chain

    # Default (OpenRouter / other providers)
    default = model or config.get("default_model") or _DEFAULT_FALLBACK_CHAIN[0]
    chain   = config.get("model_fallback_chain", [])
    if not chain:
        return [default]
    if default not in chain:
        chain = [default] + list(chain)
    return chain


def _pick_model(chain: list[str]) -> tuple[str, bool]:
    for i, m in enumerate(chain):
        if not _get_circuit(m).is_open:
            return m, i > 0
    return chain[0], False


# ── Cost model ────────────────────────────────────────────────────────────────

_MODEL_COST_TIERS: dict[str, float] = {
    "meta-llama/llama-3.1-8b-instruct": 0.00,
    "google/gemma-2-9b-it": 0.00,
    "mistralai/mistral-7b-instruct": 0.00,
    "anthropic/claude-3-haiku": 0.25,
    "google/gemini-flash-1.5": 0.075,
    "openai/gpt-4o-mini": 0.15,
    "anthropic/claude-3-5-sonnet": 3.0,
    "openai/gpt-4o": 2.5,
    "google/gemini-pro-1.5": 1.25,
    "anthropic/claude-3-opus": 15.0,
    "openai/o1-preview": 15.0,
    "openai/o1": 15.0,
    # NVIDIA NIM models
    "nvidia/llama-3.1-nemotron-70b-instruct": 0.00,
    "meta/llama-3.1-70b-instruct": 0.00,
    "meta/llama-3.1-8b-instruct": 0.00,
    "mistralai/mixtral-8x7b-instruct-v0.1": 0.00,
}

_TIER_CHEAP = 0.5
_TIER_MID   = 5.0


def _get_model_cost(model: str) -> float:
    extra = config.get("model_costs")
    if isinstance(extra, dict) and model in extra:
        return float(extra[model])
    model_lower = model.lower()
    for key, cost in _MODEL_COST_TIERS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return cost
    return 3.0


def _get_all_models_with_costs() -> list[tuple[str, float]]:
    costs: dict[str, float] = {}
    costs.update(_MODEL_COST_TIERS)
    extra = config.get("model_costs")
    if isinstance(extra, dict):
        costs.update({k: float(v) for k, v in extra.items()})
    for m in config.get("model_fallback_chain", []):
        if m not in costs:
            costs[m] = _get_model_cost(m)
    default = config.get("default_model", "")
    if default and default not in costs:
        costs[default] = _get_model_cost(default)
    return sorted(costs.items(), key=lambda x: x[1])


def suggest_model(task_complexity: str = "medium", budget: float | None = None) -> str:
    candidates = _get_all_models_with_costs()
    if budget is not None:
        candidates = [(m, c) for m, c in candidates if c <= budget]
    if not candidates:
        all_m = _get_all_models_with_costs()
        return all_m[0][0] if all_m else config.get("default_model", "openai/gpt-4o-mini")
    if task_complexity == "simple":
        cheap = [(m, c) for m, c in candidates if c < _TIER_CHEAP]
        if cheap:
            return cheap[0][0]
    for model_name, _ in candidates:
        if not _get_circuit(model_name).is_open:
            return model_name
    return candidates[0][0]


# ── Retry-After ───────────────────────────────────────────────────────────────

def _parse_retry_after(response: httpx.Response) -> float | None:
    val = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            import datetime
            dt = parsedate_to_datetime(val)
            remaining = (dt - datetime.datetime.now(dt.tzinfo)).total_seconds()
            return max(0, min(remaining, 120))
        except Exception:
            return None


# ── list_models ───────────────────────────────────────────────────────────────

async def list_models() -> list[dict[str, Any]]:
    """Fetch available models from the active provider.

    v21: When provider is "nvidia", returns the known NVIDIA models list
    from _NVIDIA_MODELS instead of hitting the /v1/models endpoint
    (which may not be available or may return incomplete data on NVIDIA NIM).

    For other providers, handles two response shapes:
      - OpenAI / OpenRouter style: {"data": [...]}
      - Together AI: raw JSON array [...]
    """
    provider = _get_provider()

    # NVIDIA NIM: return known model catalog
    if provider == "nvidia":
        return [
            {"id": fully_qualified, "object": "model", "owned_by": fully_qualified.split("/")[0]}
            for fully_qualified in _NVIDIA_MODELS.values()
        ]

    base = _get_base_url()
    api_key = _get_api_key()

    # v25 FIX: Skip the API call entirely if there's no API key configured.
    # Previously, calling /v1/models with no auth header produced a confusing
    # 401 error that looked like a broken key even when the key simply hadn't
    # been saved yet.  Now we return an empty list with a clear event.
    if not api_key:
        if provider == "custom":
            allow_no_key = config.get("custom_no_auth", False)
            if allow_no_key:
                pass  # OK — local endpoint with no auth
            else:
                await emit("warning", source="llm.list_models",
                           error="No API key configured — enter your key in Settings")
                return []
        else:
            await emit("warning", source="llm.list_models",
                       error=f"No {provider} API key configured — enter your key in Settings")
            return []

    # Catch placeholder/example keys before making the HTTP request.
    # Without this check the user gets a confusing 401 from the remote API
    # instead of a clear local message about their key being a placeholder.
    _placeholders = (
        "your-api-key-here", "your_api_key_here", "placeholder",
        "replace-me", "your-key-here", "sk-placeholder",
        "sk-xxxx", "sk-xxxxxxxx",
        "***", "****", "*****", "xxxxxx", "xxxx",
        "your-openrouter-key", "your-nvidia-key", "your-custom-key",
        "nvapi-xxxx", "nvapi-placeholder", "your-together-key",
    )
    if api_key.lower().strip() in _placeholders:
        await emit("warning", source="llm.list_models",
                   error=f"API key is still a placeholder ('{api_key.strip()}'). "
                         f"Open Settings -> API Credentials and set your real key.")
        return []

    try:
        client = await _get_client()
        r = await client.get(f"{base}/models", headers=_headers(api_key=api_key, provider=provider))
        r.raise_for_status()
        data = r.json()
        # Together AI returns a raw list; OpenAI/OpenRouter wrap in {"data": [...]}
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", [])
        return []
    except Exception as e:
        await emit("error", source="llm.list_models", error=str(e))
        return []


def _supports_vision(model: str) -> bool:
    model_lower = model.lower()
    vision_keywords = [
        "vision", "vl", "claude", "gpt-4v", "4v", "llama3.2", "llama-4",
        "llama4", "scout", "qwen2.5-vl", "qwen-vl", "glm4v", "glm-4v",
        "gemini-2", "gemini-1.5", "gemini-pro-vision",
    ]
    return any(kw in model_lower for kw in vision_keywords)


# ── Model tool-calling capability ─────────────────────────────────────────────

# Models known to NOT support the `tools` parameter in chat completions.
# When these models are active, we strip `tools` from the API payload and
# instead embed tool descriptions in the system prompt, relying on the
# text-based tool-call parser in the reasoning engine.
_NO_TOOL_CALL_MODELS: list[str] = [
    "deepseek-r1",          # DeepSeek R1 (reasoning model, no function calling)
    "deepseek-r1-distill",  # Distilled variants
    "o1-preview",           # OpenAI o1-preview (no tools)
    "o1-mini",              # OpenAI o1-mini (no tools)
    "deepseek/deepseek-r1", # Via OpenRouter
    "starcoder2-15b",       # Code model, no function calling
    "codellama",            # Code models generally don't support tools
]

# Runtime set: models that have been auto-detected as not supporting tools
# (populated when a 400/500 error mentioning "tools" or "function" is returned).
_detected_no_tool_models: set[str] = set()


def _supports_tool_calling(model: str) -> bool:
    """Return True if the model is known to support the `tools` API parameter.

    Some reasoning models (DeepSeek-R1, OpenAI o1-preview) do not support
    function calling via the API.  Sending `tools` causes 400/500 errors or
    the parameter is silently ignored and the model responds with text
    describing what it *would* do instead of actually calling tools.
    """
    if not model:
        return True  # assume yes when model is unknown
    model_lower = model.lower()
    # Check auto-detected models first
    if model in _detected_no_tool_models:
        return False
    # Check known list
    for pattern in _NO_TOOL_CALL_MODELS:
        if pattern in model_lower:
            return False
    return True


def mark_model_no_tool_calling(model: str) -> None:
    """Mark a model as not supporting tool calling (auto-detected at runtime)."""
    if model:
        _detected_no_tool_models.add(model)


def reset_detected_tool_models() -> None:
    """Clear the runtime auto-detected set (for testing)."""
    _detected_no_tool_models.clear()


def _strip_images_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed = []
    for msg in messages:
        # Handle both string content and dict content
        content = ""
        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif isinstance(msg, str):
            content = msg
        else:
            content = str(msg)

        if isinstance(content, str) and "data:image" in content:
            new_content = content[:content.find("data:image")]
            if new_content.strip():
                msg = {**msg, "content": new_content} if isinstance(msg, dict) else new_content
            else:
                msg = {**msg, "content": "[Screenshot unavailable - model does not support image input]"} if isinstance(msg, dict) else "[Screenshot unavailable]"
        elif isinstance(content, list):
            new_content = [c for c in content if not isinstance(c, dict) or c.get("type") != "image_url"]
            msg = {**msg, "content": new_content}
        fixed.append(msg)
    return fixed


# ── complete ──────────────────────────────────────────────────────────────────

async def complete(
    prompt: str,
    model: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
    session_id: str | None = None,
    images: list[str] | None = None,  # v22: Base64 images for vision
) -> "LLMResponse | AsyncIterator[dict[str, Any]]":
    global GLOBAL_USAGE

    # Restore persisted counters from DB on first call after restart
    await load_global_usage()

    resolved_model = model or config.get("default_model")
    temperature = temperature if temperature is not None else config.get("temperature")

    # ── NVIDIA model auto-resolution ──────────────────────────────────────
    # When provider is "nvidia", resolve short model names and detect
    # non-NVIDIA models before making the API call.
    _active_provider = _get_provider()
    if _active_provider == "nvidia":
        resolved_model = _resolve_nvidia_model(resolved_model)

    base = _get_base_url_for_model(resolved_model)
    api_key = _get_api_key_for_model(resolved_model)

    # v24: Validate the API key for the model's resolved provider
    _validate_api_key(resolved_model)

    # v22: Auto-select vision model when images are present in messages
    if images:
        vision_model = config.get("vision_model", "")
        if vision_model:
            # Explicitly configured vision_model is always trusted
            resolved_model = vision_model
            # Re-resolve if NVIDIA provider
            if _active_provider == "nvidia":
                resolved_model = _resolve_nvidia_model(resolved_model)
        elif _supports_vision(resolved_model):
            pass  # default_model already supports vision - use it
        else:
            # Fallback: check if planner_model supports vision
            planner = config.get("planner_model", "")
            if planner and _supports_vision(planner):
                resolved_model = planner

    # ── Guard: detect provider/model mismatch ────────────────────────────────
    # Check mismatch for "nvidia" and "custom" providers; Together AI and
    # OpenAI use their own named providers so they never hit this guard.
    if _active_provider == "nvidia" and resolved_model:
        # After _resolve_nvidia_model(), the model should be valid for NVIDIA.
        # But if somehow a non-NVIDIA model slipped through, catch it here.
        _nvidia_url = config.get("nvidia_base_url") or "https://integrate.api.nvidia.com/v1"
        _looks_like_non_nvidia = (
            "anthropic/" in resolved_model.lower() or
            "openai/" in resolved_model.lower() or
            "cohere/" in resolved_model.lower() or
            "meta-llama/" in resolved_model.lower()  # OpenRouter naming, not NVIDIA
        )
        if _looks_like_non_nvidia:
            await emit("warning", source="llm.nvidia_model_fallback",
                       original_model=resolved_model,
                       fallback_model=_NVIDIA_DEFAULT_MODEL,
                       message=f"Model '{resolved_model}' is not available on NVIDIA NIM — "
                               f"falling back to '{_NVIDIA_DEFAULT_MODEL}'")
            resolved_model = _NVIDIA_DEFAULT_MODEL

    elif _active_provider == "custom" and resolved_model and "/" in resolved_model:
        _custom_url = config.get("custom_base_url") or "https://integrate.api.nvidia.com/v1"
        _looks_like_openrouter_only = (
            "anthropic/" in resolved_model.lower() or
            "openai/" in resolved_model.lower() or
            "cohere/" in resolved_model.lower()
        )
        if _looks_like_openrouter_only:
            raise RuntimeError(
                f"Provider/model mismatch: provider is 'custom' ({_custom_url}) but "
                f"model '{resolved_model}' is only available on OpenRouter. "
                f"Fix: either set provider to 'openrouter' in Settings, or change the "
                f"model to one supported by your custom endpoint."
            )

    # Build messages - convert simple string to proper format
    if messages is None:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

    # v22: Inject images into the last user message if images provided
    if images:
        has_user_msg = False
        for msg in messages[::-1]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg["content"] = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    pass  # Already in correct format
                else:
                    msg["content"] = [{"type": "text", "text": str(content)}]
                for img in images:
                    msg["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                    })
                has_user_msg = True
                break
        if not has_user_msg:
            content_parts = [{"type": "text", "text": str(prompt)}]
            for img in images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
            messages.append({"role": "user", "content": content_parts})
        # NOTE: Image injection into messages is done above.
        # The actual API call headers are built later by _headers().
        # Do NOT return here — that was the original bug that caused vision
        # requests to return a dict of headers instead of an LLMResponse.

    chain = _get_fallback_chain(resolved_model)
    active_model, is_fallback = (resolved_model, False) if len(chain) <= 1 else _pick_model(chain)

    if is_fallback:
        await emit("model_fallback", from_model=resolved_model, to_model=active_model,
                   message=f"Switched to fallback model: {active_model}")

    clamped_max = _clamp_max_tokens(max_tokens, active_model)

    # ── Single-pass message normalization ──────────────────────────────────
    # Previously, messages were normalized three times (convert to dict,
    # filter to dict-only, validate roles).  Now consolidated into one pass
    # that: (1) converts to dict format, (2) filters to valid dicts,
    # (3) validates roles and handles tool/assistant messages.
    validated_messages = []
    for msg in messages:
        # Step 1: Convert non-dict messages to dict format
        if not isinstance(msg, dict):
            validated_messages.append({"role": "user", "content": str(msg)})
            continue

        role = msg.get("role", "user")

        # Step 2: Handle tool result messages — keep tool_call_id + name intact
        # Stripping these fields causes OpenAI-compatible APIs to return 400
        # because a tool message references an ID with no matching tool_call.
        if role == "tool":
            out: dict[str, Any] = {"role": "tool"}
            if "tool_call_id" in msg:
                out["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                out["name"] = msg["name"]
            # Tool results are almost always JSON — do NOT replace
            # content that starts with "{" with a placeholder.
            out["content"] = msg.get("content") or ""
            validated_messages.append(out)
            continue

        # Step 3: Handle assistant messages — keep tool_calls list
        # Without it the follow-up tool result messages reference ghost IDs.
        if role == "assistant":
            out = {"role": "assistant"}
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and (part.get("type") == "text" or part.get("text"))
                ]
                content = " ".join(text_parts)
            out["content"] = content if isinstance(content, str) else str(content)
            if msg.get("tool_calls"):
                out["tool_calls"] = msg["tool_calls"]
            validated_messages.append(out)
            continue

        # Step 4: Regular user / system messages
        content = msg.get("content", "")
        if isinstance(content, list):
            # Check if content has image_url parts (vision format — preserve as-is)
            has_images = any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in content
            )
            if has_images:
                # Preserve the full list with image_url parts so vision data
                # survives to the API call. Non-vision models strip images
                # later via _strip_images_from_messages().
                validated_messages.append({"role": role, "content": content})
                continue
            # No images — flatten to text as before
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("text"):
                        text_parts.append(part.get("text", ""))
            content = " ".join(text_parts) if text_parts else ""

        if not isinstance(content, str):
            content = str(content) if content else ""

        # Only fall back to placeholder when content is genuinely empty.
        # Never replace content just because it starts with "{" — that
        # pattern matches legitimate JSON payloads (tool results, etc.).
        if not content:
            content = "[Message content]"

        validated_messages.append({"role": role, "content": content})

    messages = validated_messages

    if not _supports_vision(active_model):
        messages = _strip_images_from_messages(messages)

    if stream:
        return _stream(messages, model=active_model, tools=tools,
                       temperature=temperature, max_tokens=clamped_max,
                       base=base, session_id=session_id)

    # ── Non-streaming with retry + circuit breaker ─────────────────────────
    last_exc: Exception | None = None
    last_resp_body: str = ""

    # ── Tool-calling capability check ───────────────────────────────────────
    # Some models (e.g., DeepSeek-R1, o1-preview) don't support the `tools`
    # parameter.  When detected, we strip `tools` from the API payload and
    # instead inject a tool-use instruction block into the system prompt so
    # the model outputs tool calls in a text format that the reasoning
    # engine's _parse_text_tool_calls() can parse.
    _tools_stripped = False
    _original_tools = tools  # keep reference for the response
    if tools and not _supports_tool_calling(active_model):
        _tools_stripped = True
        tools = None  # will NOT be sent in the API payload
        # Build a text block describing available tools and how to call them
        _tool_lines = []
        for t in _original_tools:
            func = t.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            param_str = ""
            if params and isinstance(params, dict):
                props = params.get("properties", {})
                required = params.get("required", [])
                parts = []
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "string")
                    req = " (required)" if pname in required else ""
                    parts.append(f'"{pname}": <{ptype}{req}>')
                param_str = ", ".join(parts)
            _tool_lines.append(f'- {name}({param_str}): {desc}')
        _tool_instruction = (
            "\n\n# CRITICAL: Tool Use Instructions\n"
            "You have the following tools available. You MUST use them when the user's request "
            "requires action (searching, reading files, running code, etc.).\n\n"
            "## How to Call Tools\n"
            "Output a tool call in this EXACT format — a JSON code block:\n"
            '```json\n{"name": "tool_name", "arguments": {"param": "value"}}\n```\n\n'
            "## Examples\n"
            '```json\n{"name": "web_search", "arguments": {"query": "latest AI news"}}\n```\n'
            '```json\n{"name": "file_read", "arguments": {"path": "/tmp/data.txt"}}\n```\n'
            '```json\n{"name": "shell_run", "arguments": {"command": "ls -la"}}\n```\n\n'
            "## Rules\n"
            "1. You can call multiple tools by outputting multiple JSON blocks.\n"
            "2. Do NOT describe what you would do in prose — ACTUALLY output the JSON tool call.\n"
            "3. Do NOT say 'I will search for...' — instead, output the search JSON directly.\n"
            "4. Do NOT say 'Let me check...' — instead, output the tool JSON directly.\n"
            "5. Every action you want to take MUST be a JSON tool call, not text description.\n\n"
            "Available tools:\n" + "\n".join(_tool_lines)
        )
        # Inject into the first system message
        messages = list(messages)  # FIX: copy before mutating (Bug 2)
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i] = {**msg, "content": msg.get("content", "") + _tool_instruction}
                break
        else:
            # No system message found — prepend one
            messages.insert(0, {"role": "system", "content": _tool_instruction})

    for attempt in range(_MAX_RETRIES):
        if _get_circuit(active_model).is_open:
            # Try next in chain
            active_model, _ = _pick_model(chain)

        # v24 FIX: Recalculate base URL, API key, and provider for the
        # ACTIVE model (which may have changed due to fallback). Previously
        # `base` and `api_key` were computed once from `resolved_model` and
        # never updated, so a fallback model that needs a different provider
        # would still use the original provider's URL and key -> 401 errors.
        base = _get_base_url_for_model(active_model)
        api_key = _get_api_key_for_model(active_model)
        resolved_provider = _get_provider_for_model(active_model)

        payload: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.7,
            "max_tokens": clamped_max,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        hdrs = _headers(api_key=api_key, provider=resolved_provider)

        # Debug gated behind config flag
        if config.get("debug_llm_requests", False):
            import sys as _sys
            print(f"[LLM REQ] URL: {base}/chat/completions", file=_sys.stderr)
            print(f"[LLM REQ] Model: {active_model}", file=_sys.stderr)
            print(f"[LLM REQ] Messages: {len(messages)} msg(s)", file=_sys.stderr)
            print(f"[LLM REQ] Tools: {'yes' if tools else 'no (stripped)' if _tools_stripped else 'no'}", file=_sys.stderr)

        try:
            client = await _get_client()
            endpoint = "/chat/completions"
            r = await client.post(f"{base}{endpoint}", headers=hdrs, json=payload)

            # Handle 400 errors with more info (debug gated)
            if r.status_code == 400 and config.get("debug_llm_requests", False):
                import sys as _sys
                print(f"[LLM 400 ERROR] Model: {active_model}", file=_sys.stderr)
                print(f"[LLM 400 ERROR] Response: {r.text[:500]}", file=_sys.stderr)

            # ── Auto-detect models that don't support tools ──────────────
            # If a 400 error mentions "tools" or "function_calling", the
            # model likely doesn't support the tools parameter.  Mark it and
            # retry WITHOUT tools (injected into system prompt instead).
            if r.status_code == 400 and _original_tools and not _tools_stripped:
                resp_text = r.text.lower()
                if any(kw in resp_text for kw in ("tool", "function_call", "function calling", "tools parameter", "does not support")):
                    mark_model_no_tool_calling(active_model)
                    _tools_stripped = True
                    tools = None
                    # Re-inject tool instructions into system prompt
                    _tool_lines = []
                    for t in _original_tools:
                        func = t.get("function", {})
                        name = func.get("name", "unknown")
                        desc = func.get("description", "")
                        params = func.get("parameters", {})
                        param_str = ""
                        if params and isinstance(params, dict):
                            props = params.get("properties", {})
                            required = params.get("required", [])
                            parts = []
                            for pname, pdef in props.items():
                                ptype = pdef.get("type", "string")
                                req = " (required)" if pname in required else ""
                                parts.append(f'"{pname}": <{ptype}{req}>')
                            param_str = ", ".join(parts)
                        _tool_lines.append(f'- {name}({param_str}): {desc}')
                    _tool_instruction = (
                        "\n\n# CRITICAL: Tool Use Instructions\n"
                        "You have the following tools available. You MUST use them when the user's request "
                        "requires action (searching, reading files, running code, etc.).\n\n"
                        "## How to Call Tools\n"
                        "Output a tool call in this EXACT format — a JSON code block:\n"
                        '```json\n{"name": "tool_name", "arguments": {"param": "value"}}\n```\n\n'
                        "## Examples\n"
                        '```json\n{"name": "web_search", "arguments": {"query": "latest AI news"}}\n```\n'
                        '```json\n{"name": "file_read", "arguments": {"path": "/tmp/data.txt"}}\n```\n'
                        '```json\n{"name": "shell_run", "arguments": {"command": "ls -la"}}\n```\n\n'
                        "## Rules\n"
                        "1. You can call multiple tools by outputting multiple JSON blocks.\n"
                        "2. Do NOT describe what you would do in prose — ACTUALLY output the JSON tool call.\n"
                        "3. Do NOT say 'I will search for...' — instead, output the search JSON directly.\n"
                        "4. Do NOT say 'Let me check...' — instead, output the tool JSON directly.\n"
                        "5. Every action you want to take MUST be a JSON tool call, not text description.\n\n"
                        "Available tools:\n" + "\n".join(_tool_lines)
                    )
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            messages[i] = {**msg, "content": msg.get("content", "") + _tool_instruction}
                            break
                    else:
                        messages.insert(0, {"role": "system", "content": _tool_instruction})
                    # Retry immediately without incrementing attempt counter
                    continue

            # ── NVIDIA NIM 404 handling ──────────────────────────────────
            # If we get a 404 on the NVIDIA endpoint, it usually means the
            # model ID is invalid.  Try to fall back to the default model.
            if r.status_code == 404 and resolved_provider == "nvidia":
                if active_model != _NVIDIA_DEFAULT_MODEL:
                    await emit("warning", source="llm.nvidia_404_fallback",
                               failed_model=active_model,
                               fallback_model=_NVIDIA_DEFAULT_MODEL,
                               message=f"NVIDIA NIM returned 404 for model '{active_model}' — "
                                       f"falling back to '{_NVIDIA_DEFAULT_MODEL}'")
                    active_model = _NVIDIA_DEFAULT_MODEL
                    # Retry with default model (don't count as a retry attempt)
                    continue
                else:
                    raise RuntimeError(
                        f"NVIDIA NIM returned 404 even for default model '{_NVIDIA_DEFAULT_MODEL}'. "
                        f"The NVIDIA API endpoint may be down or your API key may lack access. "
                        f"URL: {base}/chat/completions"
                    )

            if r.status_code in _RETRYABLE_STATUS:
                last_resp_body = r.text[:500]
                ra = _parse_retry_after(r)
                delay = ra if ra else min(_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), _RETRY_MAX_DELAY)
                await asyncio.sleep(delay)
                continue

            if r.status_code == 401:
                raise RuntimeError(
                    f"API key rejected (401). Check your API key in Settings. "
                    f"Provider: {_get_provider()}, URL: {base}"
                )
            if r.status_code == 403:
                raise RuntimeError(f"Access forbidden (403). Your API key may lack permissions. URL: {base}")
            if r.status_code == 400:
                raise RuntimeError(f"Bad request (400). Details: {r.text[:300]}")

            r.raise_for_status()

            data = r.json()

            msg = data["choices"][0]["message"]
            # Handle null/empty content
            raw_content = msg.get("content")
            msg_content = raw_content if raw_content else ""
            # Strip null bytes — SQLite rejects 0x00 in TEXT columns
            if msg_content and "\x00" in msg_content:
                msg_content = msg_content.replace("\x00", "")
            # FIX: NEVER use reasoning_content as visible response content.
            # reasoning_content contains the model's internal chain-of-thought
            # (e.g., DeepSeek-R1's reasoning tokens, OpenAI o1's internal monologue)
            # and should NEVER be shown to the user. Only use it for debugging.
            # If content is empty AND there are no tool_calls, use a placeholder.
            # Models that return tool_calls via the native API commonly set
            # content to null/empty — that's normal, not an error.
            if not msg_content and not msg.get("tool_calls"):
                # Model returned an empty content with no tool calls.
                # This can happen when the model is purely reasoning (DeepSeek-R1
                # style) or hit a content filter. Instead of injecting an ugly
                # placeholder that leaks to the user, log the reasoning_content
                # (if any) for debugging and return empty — the caller's
                # retry/stuck logic handles this gracefully.
                _rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if _rc:
                    if config.get("debug_llm_requests", False):
                        import sys as _sys
                        print(f"[LLM DEBUG] Model returned reasoning_content but empty content: {_rc[:200]}...", file=_sys.stderr)
                msg_content = ""
            if "does not support image" in msg_content.lower() or "model does not support image" in msg_content.lower():
                raise RuntimeError(
                    f"Model {active_model} does not support image input. "
                    "Use a vision-capable model (e.g., Claude, GPT-4V, Llama with vision) or analyze screenshots manually."
                )
            usage_data = data.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            GLOBAL_USAGE += usage
            if session_id:
                _session_tracker.record(session_id, usage)
            # Persist to DB so a crash doesn't lose the counters
            await _persist_usage(usage)



            _get_circuit(active_model).record_success()

            tool_calls = []
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args)
                    except Exception:
                        # Try to extract JSON from malformed string
                        try:
                            match = re.search(r'\{.*\}', str(args), re.DOTALL)
                            if match:
                                args = json.loads(match.group())
                            else:
                                args = {"input": str(args)}  # last resort
                        except Exception:
                            args = {"input": str(args)}
                    tool_calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})

            return LLMResponse(
                content=msg_content,
                tool_calls=tool_calls,
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
                model=data.get("model", active_model),
                usage=usage,
                raw=data,
                tools_stripped=_tools_stripped,
            )

        except asyncio.CancelledError:
            # Propagate cancellation immediately — do not retry.
            # The outer asyncio.wait_for() in server.py converts this to
            # TimeoutError, which is handled at the caller level.
            raise
        except RuntimeError:
            raise
        except httpx.TimeoutException as e:
            last_exc = e
            _get_circuit(active_model).record_failure()
            delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
            await asyncio.sleep(delay)
        except Exception as e:
            last_exc = e
            _get_circuit(active_model).record_failure()
            if attempt < _MAX_RETRIES - 1:
                delay = min(_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5), _RETRY_MAX_DELAY)
                await asyncio.sleep(delay)

    raise RuntimeError(f"LLM request failed after {_MAX_RETRIES} attempts: {last_exc or f'HTTP retryable status (see response: {last_resp_body})'}") from last_exc


# ── Streaming ─────────────────────────────────────────────────────────────────

async def _stream(
    messages, *, model, tools, temperature, max_tokens, base, session_id
) -> AsyncIterator[dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature is not None else 0.7,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resolved_provider = _get_provider_for_model(model)
    api_key = _get_api_key_for_model(model)
    client = await _get_client()
    total_content = 0

    async with client.stream("POST", f"{base}/chat/completions",
                              headers=_headers(api_key=api_key, provider=resolved_provider),
                              json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except Exception:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            # Strip null bytes — SQLite rejects 0x00 in TEXT columns
            if content and "\x00" in content:
                content = content.replace("\x00", "")
            total_content += len(content)
            yield {"type": "delta", "content": content, "raw": chunk}

    if session_id:
        approx = Usage(completion_tokens=total_content // 4, total_tokens=total_content // 4)
        _session_tracker.record(session_id, approx)
        GLOBAL_USAGE += approx
        # Persist streaming approximation to DB
        await _persist_usage(approx)
