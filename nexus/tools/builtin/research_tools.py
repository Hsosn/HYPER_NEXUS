"""
Deep research tool — parallel multi-source synthesis with connection pooling + caching.

Takes a question, searches 5-10 sources, fetches each one in parallel,
then synthesises a structured report with an LLM call.
"""
from __future__ import annotations

import asyncio
import hashlib
import re as _re
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

from ...config import get as cfg
from ...core import llm
from ..registry import tool

# ---------------------------------------------------------------------------
# Connection pool (shared across fetches)
# ---------------------------------------------------------------------------
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client(timeout: float = 15.0) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
                _client = httpx.AsyncClient(
                    timeout=timeout, follow_redirects=True, limits=limits,
                    headers={"User-Agent": "NexusResearch/2.0"},
                )
    return _client


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 600.0  # 10 minutes, matches tool-level cache_ttl


def _cache_key(prefix: str, *args: str) -> str:
    return hashlib.md5((prefix + "|".join(args)).encode()).hexdigest()


def _cache_get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < _CACHE_TTL:
        return entry[1]
    if entry:
        del _cache[key]
    return None


def _cache_set(key: str, value: Any) -> None:
    if len(_cache) > 128:
        # Use heapq.nsmallest to evict oldest 16 entries in O(n log k) instead of O(n log n)
        import heapq as _heapq
        oldest = _heapq.nsmallest(16, _cache.items(), key=lambda x: x[1][0])
        for k, _ in oldest:
            _cache.pop(k, None)
    _cache[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# Rate limiter — DDGS is sensitive to parallel requests, but full serialisation
# is too slow (4-6 queries × 10-15s each = 60-90s).  Allow 2 concurrent DDGS
# calls; the rate limiter's retry logic handles transient 429s.
# ---------------------------------------------------------------------------
_ddgs_semaphore = asyncio.Semaphore(2)

# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

@tool(
        name="deep_research",
        description=(
            "Conduct thorough research on a topic. Searches multiple sources in parallel, "
            "fetches full content, and synthesises a structured report with key findings, "
            "analysis, and source citations. Use for complex questions needing depth."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "question":    {"type": "string", "description": "Research question or topic"},
                "query":       {"type": "string", "description": "Alias for question"},
                "q":           {"type": "string", "description": "Alias for question"},
                "num_sources": {"type": "integer", "default": 4,
                                "description": "Number of sources to fetch and synthesise (3–10)"},
                "sources":     {"type": "integer", "default": 4,
                                "description": "Alias for num_sources"},
                "focus":       {"type": "string", "default": "",
                                "description": "Optional: specific angle or focus for the report"},
            },
            "required": ["question"],
        },
        category="research",
        cacheable=True,
        cache_ttl=600,
        timeout=90,
    )
async def deep_research(params):
    question   = params.get("question") or params.get("query") or params.get("q", "")
    n_sources  = max(3, min(int(params.get("num_sources") or params.get("sources") or 4), 10))
    focus      = params.get("focus", "")

    ck = _cache_key("research:", question, str(n_sources), focus)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    try:
        # Step 1: Generate diverse search queries
        queries = await _generate_queries(question, n_sources)

        # Step 2: Search and fetch all sources in parallel
        # Each _search_and_fetch has its own per-query timeout (search 10s + fetch 10s).
        # Allow up to 75s total to accommodate 4-6 queries × ~12s each with semaphore=2.
        fetch_tasks = [_search_and_fetch(q) for q in queries]
        results     = await asyncio.wait_for(
            asyncio.gather(*fetch_tasks, return_exceptions=True),
            timeout=75.0,
        )

        sources = []
        errors = []
        for r in results:
            if isinstance(r, Exception):
                errors.append(str(r))
            elif isinstance(r, dict) and r.get("content"):
                sources.append(r)
            if len(sources) >= n_sources:
                break

        if not sources:
            error_detail = "; ".join(errors[:3]) if errors else "all searches returned empty results"
            return (
                f"Research failed: could not retrieve any sources ({error_detail}). "
                f"Try a simpler question or check API availability."
            )

        # Step 3: Synthesise with LLM
        report = await _synthesise(question, sources, focus)
    except asyncio.TimeoutError:
        return (
            f"Research timed out while searching for sources on: {question}. "
            f"The web searches took too long. Try rephrasing or being more specific."
        )

    _cache_set(ck, report)
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _generate_queries(question: str, n: int) -> list[str]:
    """Generate diverse search queries for a topic."""
    try:
        prompt = [{
            "role": "user",
            "content": (
                f"Generate {n} diverse web search queries to thoroughly research this topic:\n\n"
                f"Topic: {question}\n\n"
                f"Return ONLY the queries, one per line, no numbering, no explanations."
            )
        }]
        resp = await llm.complete(prompt, model=cfg("memory_model"),
                                  max_tokens=300, temperature=0.4)
        raw = (resp.content or "").strip()
        lines = [_re.sub(r"^[\s>\-*\d.)]+", "", ln).strip() for ln in raw.splitlines()]
        queries = [q for q in lines if q]
        return queries[:n] if queries else [question]
    except Exception:
        return [question]


def _ddgs_search_sync(query: str, max_results: int) -> list:
    """Synchronous DDGS search helper (run in a thread) with retry.

    NOTE: Import order is CRITICAL. `ddgs` is the current package name;
    `duckduckgo_search` is deprecated and silently returns 0 results.
    Always try `ddgs` first.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception:
            if attempt == 2:
                return []
            time.sleep(0.5 * (attempt + 1))  # linear backoff


async def _search_and_fetch(query: str) -> dict | None:
    """Search DuckDuckGo for a query and fetch the top result.

    Uses a semaphore to serialise DDGS calls (parallel requests get
    rate-limited).  Each search+fetch has its own per-query timeout.
    """
    try:
        ck = _cache_key("search_fetch:", query)
        cached = _cache_get(ck)
        if cached is not None:
            return cached

        # Limit DDGS concurrency — 2 parallel calls is safe; full serialisation is too slow
        async with _ddgs_semaphore:
            results = await asyncio.wait_for(
                asyncio.to_thread(_ddgs_search_sync, query, 2),
                timeout=10.0,
            )

        if not results:
            return None
        url = results[0].get("href", "") or results[0].get("link", "")
        if not url:
            return None
        content = await _fetch_text(url)
        if not content or content.startswith("[FETCH ERROR]"):
            return None
        result = {
            "query":   query,
            "url":     url,
            "title":   results[0].get("title", url),
            "content": content[:3000],
        }
        _cache_set(ck, result)
        return result
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


async def _fetch_text(url: str) -> str:
    """Fetch and clean text from a URL using shared connection pool."""
    try:
        client = await _get_client()
        r = await asyncio.wait_for(client.get(url), timeout=10.0)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:4000]
    except asyncio.TimeoutError:
        return f"[FETCH ERROR] Timeout for {url}"
    except httpx.HTTPStatusError as e:
        return f"[FETCH ERROR] HTTP {e.response.status_code} for {url}"
    except httpx.TimeoutException:
        return f"[FETCH ERROR] Timeout for {url}"
    except httpx.ConnectError as e:
        err_str = str(e).lower()
        if "getaddrinfo" in err_str or "name or service not known" in err_str:
            return f"[FETCH ERROR] DNS resolution failed for {url}"
        return f"[FETCH ERROR] Connection failed for {url}"
    except Exception as e:
        return f"[FETCH ERROR] {e}"


async def _synthesise(question: str, sources: list[dict], focus: str) -> str:
    """Ask the LLM to synthesise all sources into a structured report."""
    sources_text = ""
    for i, s in enumerate(sources, 1):
        sources_text += f"\n---\nSource {i}: {s['title']}\nURL: {s['url']}\n{s['content']}\n"

    focus_note = f"\nFocus angle: {focus}" if focus else ""

    prompt = [{
        "role": "user",
        "content": (
            f"You are a research analyst. Using ONLY the sources below, write a thorough "
            f"research report answering this question:\n\n"
            f"Question: {question}{focus_note}\n\n"
            f"Sources:{sources_text}\n\n"
            f"Structure your report as:\n"
            f"## Summary\n(3-4 sentence overview)\n\n"
            f"## Key Findings\n(bullet points with the most important facts)\n\n"
            f"## Analysis\n(your synthesis and interpretation)\n\n"
            f"## Sources\n(numbered list with URLs)\n\n"
            f"Be specific, cite sources by number [1], [2] etc. "
            f"Do not include information not found in the sources."
        )
    }]

    try:
        report = await llm.complete(
            prompt,
            model=cfg("default_model"),
            max_tokens=1500,
            temperature=0.3,
        )
        text = (report.content or "").strip()
        if not text:
            raise RuntimeError("Empty synthesis from model")
        return text
    except Exception as e:
        lines = [f"# Research: {question}\n"]
        for i, s in enumerate(sources, 1):
            lines.append(f"## [{i}] {s['title']}\n{s['url']}\n{s['content'][:500]}\n")
        return "\n".join(lines)
