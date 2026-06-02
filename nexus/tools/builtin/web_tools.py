"""Web tools: search, fetch URL, summarize — optimized with connection pooling + caching."""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import time
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ...config import get as cfg_get
from ..registry import tool

# ---------------------------------------------------------------------------
# Connection pool (shared across all tools)
# ---------------------------------------------------------------------------
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client(timeout: float = 20.0) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
                _client = httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=True,
                    limits=limits,
                    headers={"User-Agent": "NexusAgent/2.0"},
                )
    return _client


# ---------------------------------------------------------------------------
# Result cache (LRU-like, TTL-based)
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 60.0  # 1 minute default


def _cache_key(prefix: str, *args: str) -> str:
    raw = prefix + "|".join(args)
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < _CACHE_TTL:
        return entry[1]
    if entry:
        del _cache[key]
    return None


def _cache_set(key: str, value: str, ttl: float = _CACHE_TTL) -> None:
    if len(_cache) > 256:
        # Use heapq.nsmallest to evict oldest 32 entries in O(n log 32) instead of O(n log n)
        import heapq
        oldest = heapq.nsmallest(32, _cache.items(), key=lambda x: x[1][0])
        for k, _ in oldest:
            _cache.pop(k, None)
    _cache[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------

def _is_safe_url(url: str) -> bool:
    """Block requests to private/internal networks."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            return False
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return False
        except ValueError:
            pass
        if hostname.startswith(("169.254.", "10.", "172.16.", "192.168.")):
            return False
        return True
    except Exception:
        return False


def _search_sync(query: str, max_results: int) -> list:
    """Synchronous DDGS search helper (run in a thread) with retry."""
    from ddgs import DDGS
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))  # linear backoff, uses module-level `time`
    return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    name="web_search",
    description="Search the web using DuckDuckGo. Returns top results with title, URL, and snippet.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 6},
        },
        "required": ["query"],
    },
    category="web",
    cacheable=True,
    cache_ttl=120,
)
async def web_search(params):
    q = params.get("query", "")
    n = int(params.get("max_results", 6))

    ck = _cache_key("search:", q, str(n))
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    try:
        results = await asyncio.to_thread(_search_sync, q, n)
        if not results:
            return "No results."
        out = []
        for r in results:
            title = r.get('title') or r.get('title') or ''
            href = r.get('href') or r.get('link') or ''
            body = r.get('body') or r.get('snippet') or ''
            out.append(f"• {title}\n  {href}\n  {body[:280]}")
        result = "\n\n".join(out)
        _cache_set(ck, result, ttl=120.0)
        return result
    except Exception as e:
        return f"Search error: {e}"


@tool(
    name="fetch_url",
    description="Fetch the textual content of a URL. Returns cleaned plain text.",
    parameters_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 6000},
        },
        "required": ["url"],
    },
    risk="medium",
    category="web",
    cacheable=True,
    cache_ttl=30,
)
async def fetch_url(params):
    url = params.get("url", "")
    max_chars = int(params.get("max_chars", 6000))

    ck = _cache_key("fetch:", url, str(max_chars))
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    try:
        client = await _get_client()
        r = await client.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        result = f"Title: {title}\n\n{text}"
        _cache_set(ck, result, ttl=30.0)
        return result
    except httpx.HTTPStatusError as e:
        return (
            f"Fetch error for {url}: HTTP {e.response.status_code} "
            f"({e.response.reason_phrase})\n"
            f"The server returned an error status. The URL may be incorrect or the page is unavailable."
        )
    except httpx.TimeoutException:
        return (
            f"Fetch error for {url}: Connection timed out.\n"
            f"The server took too long to respond. Try again later or check if the site is accessible."
        )
    except httpx.ConnectError as e:
        err_str = str(e).lower()
        if "getaddrinfo" in err_str or "name or service not known" in err_str or "nodename nor servname" in err_str:
            return (
                f"Fetch error for {url}: Could not resolve the domain name (DNS lookup failed).\n"
                f"Check the URL for typos, or verify your internet/DNS connection."
            )
        return (
            f"Fetch error for {url}: Could not connect to the server.\n"
            f"The site may be down, blocked, or unreachable. Try again later."
        )
    except Exception as e:
        return f"Fetch error for {url}: {e}"


@tool(
    name="http_request",
    description="Make a generic HTTP request (GET/POST/PUT/DELETE). Use with caution.",
    parameters_schema={
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
            "url": {"type": "string"},
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "json_body": {"type": "object"},
        },
        "required": ["method", "url"],
    },
    risk="high",
    category="web",
)
async def http_request(params):
    method = params.get("method", "GET").upper()
    url = params.get("url", "")
    headers = params.get("headers") or {}
    json_body = params.get("json_body")
    if not _is_safe_url(url):
        return "Error: requests to private/internal networks are not allowed"
    try:
        client = await _get_client(timeout=30.0)
        r = await client.request(method, url, headers=headers, json=json_body)
        return f"Status: {r.status_code}\n\n{r.text[:4000]}"
    except httpx.HTTPStatusError as e:
        return (
            f"HTTP error for {method} {url}: HTTP {e.response.status_code} "
            f"({e.response.reason_phrase})"
        )
    except httpx.TimeoutException:
        return (
            f"HTTP error for {method} {url}: Connection timed out.\n"
            f"The server took too long to respond."
        )
    except httpx.ConnectError as e:
        err_str = str(e).lower()
        if "getaddrinfo" in err_str or "name or service not known" in err_str or "nodename nor servname" in err_str:
            return (
                f"HTTP error for {method} {url}: Could not resolve the domain name (DNS lookup failed).\n"
                f"Check the URL for typos, or verify your internet/DNS connection."
            )
        return (
            f"HTTP error for {method} {url}: Could not connect to the server.\n"
            f"The site may be down, blocked, or unreachable."
        )
    except Exception as e:
        return f"HTTP error for {method} {url}: {e}"
