"""
Memory manager v18 — deeply optimized.

Unifies three tiers of memory:
- Working memory: current reasoning context (in-process dict, per session).
- Short-term: recent conversation turns (from messages table, last N).
- Long-term: persistent memories with vector embeddings + importance scores.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path as _Path
from typing import Any

import httpx
import numpy as np

from .. import config
from ..events import emit
from . import database as db

# ── Local embedding backend (sentence-transformers) ───────────────────────────
# Activated when ``embedding_model`` starts with ``local/`` or equals ``local``.
# Uses a small CPU-friendly model (default: all-MiniLM-L6-v2, ~80MB, 384-dim).
_LOCAL_EMBEDDER = None
_LOCAL_EMBEDDER_LOCK = asyncio.Lock()


def _get_local_embedder():
    """Lazy-load and cache the sentence-transformers model."""
    global _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER is not None:
        return _LOCAL_EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        model_name = (
            config.get("local_embedding_model") or "all-MiniLM-L6-v2"
        )
        _LOCAL_EMBEDDER = SentenceTransformer(model_name)
        logger.info(
            "Local embedding model loaded: %s (dim=%d)",
            model_name,
            _LOCAL_EMBEDDER.get_sentence_embedding_dimension(),
        )
        return _LOCAL_EMBEDDER
    except Exception as exc:
        logger.warning("Failed to load local embedding model: %s", exc)
        return None

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Memory:
    """Single long-term memory item.

    Attributes:
        id:             Primary key from the ``memories`` table.
        kind:           One of ``factual``, ``semantic``, ``episodic``, ``procedural``,
                        ``profile``, ``preference``.
        content:        The human-readable text of the memory.
        importance:     Manual / auto-assessed importance in [0.0, 1.0].
        embedding:      Dense vector (list of floats) produced by the embedding model,
                        or ``None`` if not yet embedded.
        tags:           Free-form tags for categorisation.
        session_id:     The session that created this memory (may be ``None``).
        created_at:     Unix timestamp of creation.
        last_accessed:  Unix timestamp of last recall access.
        access_count:   Number of times this memory has been recalled.
    """

    id: int
    kind: str
    content: str
    importance: float
    embedding: list[float] | None = None
    tags: list[str] = field(default_factory=list)
    session_id: str | None = None
    created_at: float | None = None
    last_accessed: float | None = None
    access_count: int = 0

    @classmethod
    def from_row(cls, row: dict) -> Memory:
        emb = row.get("embedding")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except Exception:
                emb = None
        # Tags are stored as JSON array (json.dumps in db.add_memory).
        # Must json.loads, NOT comma-split.
        raw_tags = row.get("tags")
        if isinstance(raw_tags, str) and raw_tags:
            try:
                tags = json.loads(raw_tags)
                if not isinstance(tags, list):
                    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            except Exception:
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif isinstance(raw_tags, list):
            tags = raw_tags
        else:
            tags = []
        return cls(
            id=row["id"],
            kind=row.get("kind", "semantic"),
            content=row.get("content", ""),
            importance=float(row.get("importance", 0.5)),
            embedding=emb,
            tags=tags,
            session_id=row.get("session_id"),
            created_at=row.get("created_at"),
            last_accessed=row.get("last_accessed"),
            access_count=int(row.get("access_count", 0)),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding cache
# ═══════════════════════════════════════════════════════════════════════════════

_EMBEDDING_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_EMBEDDING_CACHE_MAX = 200
_EXPECTED_EMBEDDING_DIM: int | None = None
_EMBED_DIRTY = False
_EMBED_SAVE_INTERVAL = 30  # seconds between disk saves

_CACHE_FILE = config.BASE_DIR / "data" / "embedding_cache.json"


def _load_cache_from_disk() -> None:
    global _EMBEDDING_CACHE, _EXPECTED_EMBEDDING_DIM
    if not _CACHE_FILE.exists():
        return
    try:
        data = json.loads(_CACHE_FILE.read_text())
        _EMBEDDING_CACHE = OrderedDict(data.get("cache", {}))
        _EXPECTED_EMBEDDING_DIM = data.get("dim")
    except Exception:
        pass


def _save_cache_to_disk() -> None:
    """Persist embedding cache to disk (non-blocking best-effort)."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Use atomic write to avoid corruption on crash
        tmp = _CACHE_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps({
            "cache": dict(_EMBEDDING_CACHE),
            "dim": _EXPECTED_EMBEDDING_DIM,
        }))
        tmp.replace(_CACHE_FILE)
    except Exception:
        pass


_load_cache_from_disk()


async def _embedding_cache_saver():
    """Periodically save the embedding cache to disk (debounced)."""
    global _EMBED_DIRTY
    while True:
        await asyncio.sleep(_EMBED_SAVE_INTERVAL)
        if _EMBED_DIRTY:
            _EMBED_DIRTY = False
            _save_cache_to_disk()

# Start the saver in the background when the event loop is available
import asyncio as _asyncio_mod
try:
    _loop = _asyncio_mod.get_running_loop()
    _loop.create_task(_embedding_cache_saver())
except RuntimeError:
    pass  # No event loop yet; will be saved on next embed() or shutdown


def _validate_embedding_dim(vec: list[float]) -> list[float] | None:
    global _EXPECTED_EMBEDDING_DIM
    if _EXPECTED_EMBEDDING_DIM is None:
        _EXPECTED_EMBEDDING_DIM = len(vec)
        return vec
    if len(vec) != _EXPECTED_EMBEDDING_DIM:
        logger.warning(
            "Embedding dimension mismatch: got %d, expected %d. Falling back to hash embedding.",
            len(vec), _EXPECTED_EMBEDDING_DIM,
        )
        return None
    return vec


async def embed(text: str) -> list[float] | None:
    """Get embedding for text, with caching and fallback."""
    if not text:
        return None
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE.move_to_end(key)
        return _EMBEDDING_CACHE[key]

    # Priority: local sentence-transformers → API → hash fallback
    model_cfg = config.get("embedding_model", "")
    if model_cfg.startswith("local/") or model_cfg == "local":
        async with _LOCAL_EMBEDDER_LOCK:
            model = _get_local_embedder()
        if model is not None:
            try:
                vec = model.encode(text[:8000], normalize_embeddings=True).tolist()
                vec = _validate_embedding_dim(vec)
                if vec is not None:
                    _EMBEDDING_CACHE[key] = vec
                    while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
                        _EMBEDDING_CACHE.popitem(last=False)
                    _EMBED_DIRTY = True
                    if len(_EMBEDDING_CACHE) % 50 == 0:
                        _save_cache_to_disk()
                    return vec
            except Exception as exc:
                logger.warning("Local embedding failed, falling back: %s", exc)

    vec = await _fetch_embedding(text)
    if vec is None:
        vec = _hash_embedding(text, dim=_EXPECTED_EMBEDDING_DIM or 256)

    _EMBEDDING_CACHE[key] = vec
    while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
        _EMBEDDING_CACHE.popitem(last=False)
    _EMBED_DIRTY = True
    if len(_EMBEDDING_CACHE) % 50 == 0:
        _save_cache_to_disk()
    return vec


async def _fetch_embedding(text: str) -> list[float] | None:
    """Attempt to fetch embedding from API."""
    model = config.get("embedding_model")
    truncated = text[:8000]
    provider = config.get("provider", "").lower()

    # Try Together AI first (user already has together_api_key)
    if provider == "together":
        together_key = config.get("together_api_key", "")
        together_url = config.get("together_base_url", "https://api.together.xyz/v1").rstrip("/")
        logger.debug("[memory] Together API key present: %s, URL: %s", bool(together_key), together_url)
        if together_key and together_url:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(
                        f"{together_url}/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {together_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": model, "input": truncated},
                    )
                    if r.status_code == 200:
                        data = r.json()
                        raw_vec = data["data"][0]["embedding"]
                        return _validate_embedding_dim(raw_vec)
            except Exception as e:
                logger.warning("Together embedding failed: %s (model: %s)", e, model)

    # Fallback to OpenRouter
    base = config.get("openrouter_base_url")
    api_key = config.get("openrouter_api_key", "")
    if api_key and "openrouter" in base.lower():
        try:
            embedding_base = base
            if embedding_base.endswith("/v1"):
                embedding_base = embedding_base[:-3]
            embedding_base = embedding_base.rstrip("/")
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    f"{embedding_base}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://nexus.local",
                        "X-Title": "Hyper Nexus Agent",
                    },
                    json={"model": model, "input": truncated},
                )
                if r.status_code == 200:
                    data = r.json()
                    raw_vec = data["data"][0]["embedding"]
                    return _validate_embedding_dim(raw_vec)
        except Exception as e:
            logger.warning(
                "Embedding API failed, using hash fallback — semantic search will be degraded: %s",
                e,
            )

    return None


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic hash-based pseudo-embedding (NOT semantic)."""
    words = text.lower().split()
    v = np.zeros(dim, dtype=np.float32)
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


# Cosine similarity helpers
# ═══════════════════════════════════════════════════════════════════════════════


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 if vectors have mismatched dimensions (instead of crashing).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    na = np.asarray(a, dtype=np.float32)
    nb = np.asarray(b, dtype=np.float32)
    da = np.linalg.norm(na)
    db_ = np.linalg.norm(nb)
    if da == 0 or db_ == 0:
        return 0.0
    return float(np.dot(na, nb) / (da * db_))


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Manager
# ═══════════════════════════════════════════════════════════════════════════════

def _string_overlap(a: str, b: str) -> float:
    """Simple word-overlap similarity between two strings."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa | wb), 1)


_DEFAULT_WEIGHTS = {
    "semantic": 0.60,
    "importance": 0.25,
    "recency": 0.10,
    "frequency": 0.05,
}


class MemoryManager:
    DEFAULT_W_SEMANTIC = 0.60
    DEFAULT_W_IMPORTANCE = 0.25
    DEFAULT_W_RECENCY = 0.10
    DEFAULT_W_FREQUENCY = 0.05
    DEFAULT_EBBINGHAUS_HALF_LIFE = 24.0 * 7  # 1 week in hours
    DEFAULT_EBBINGHAUS_K = 0.5

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self._usage_tracker = _UsageTracker()

    async def remember(
        self,
        content: str,
        kind: str = "semantic",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> int:
        """Store a memory with auto-embedding and deduplication."""
        existing = await db.find_memory_by_content(content)
        if existing:
            return existing["id"]

        embedding = await embed(content)
        mem_id = await db.add_memory(
            kind=kind,
            content=content,
            importance=importance,
            embedding=embedding,
            session_id=self.session_id,
            tags=tags,
        )
        return mem_id

    async def remember_fact(
        self,
        content: str,
        importance: float = 0.9,
        kind: str = "factual",
        tags: list[str] | None = None,
    ) -> int:
        """Store a factual memory with deduplication."""
        existing = await db.find_memory_by_content(content)
        if existing:
            return existing["id"]

        embedding = await embed(content)
        mem_id = await db.add_memory(
            kind=kind,
            content=content,
            importance=importance,
            embedding=embedding,
            session_id=self.session_id,
            tags=tags or ["user_profile"],
        )
        return mem_id

    async def dedup_check(self, content: str, threshold: float = 0.92) -> "Memory | None":
        """Check if a similar memory already exists.

        Returns the existing Memory if its content similarity exceeds threshold,
        otherwise None. Used by self_improve to avoid duplicate procedural skills.
        """
        from . import db as _db
        rows = await _db.find_similar_memory(content)
        if not rows:
            return None
        # Score each candidate
        try:
            query_vec = await embed(content)
        except Exception:
            # Fallback: simple string overlap
            low = content.lower()
            for r in rows:
                if _string_overlap(low, r.get("content", "").lower()) > threshold:
                    return Memory.from_row(r) if hasattr(Memory, "from_row") else None
            return None
        best = None
        best_score = 0.0
        for r in rows:
            mem = Memory.from_row(r)
            if mem.embedding:
                try:
                    mem_vec = [float(x) for x in (mem.embedding if isinstance(mem.embedding, list) else mem.embedding.split(","))]
                    score = cosine(query_vec, mem_vec)
                except Exception:
                    score = 0.0
            else:
                score = _string_overlap(content.lower(), (r.get("content") or "").lower())
            if score > best_score:
                best_score = score
                best = mem
        return best if best_score >= threshold else None

    @staticmethod
    def _ebbinghaus(hours_since_access: float,
                    half_life: float = 168.0, k: float = 0.5) -> float:
        """Ebbinghaus forgetting curve — recency component in (0, 1], decays with time.

        Returns a value in (0, 1] representing memory retention strength.
        *half_life* is in hours (default 168 = 7 days).
        """
        if hours_since_access <= 0:
            return 1.0
        return k / (1.0 + (hours_since_access / half_life))

    async def recall(
        self,
        query: str,
        k: int | None = None,
        kind: str | None = None,
    ) -> list[Memory]:
        """Retrieve top-k relevant long-term memories using full v8 scoring.

        Scoring formula (restored from v8):
            score = w_semantic  × cosine_similarity(query_emb, mem_emb)
                 + w_importance × mem.importance
                 + w_recency    × Ebbinghaus(hours_since_access)
                 + w_frequency  × normalised_access_count

        Memories without embeddings fall back to keyword overlap for the
        semantic component (same as the hash-embedding approximation).
        """
        k = k or config.get("memory_retrieval_k", 6)
        k = max(1, min(100, int(k) if k else 6))

        # Load scoring weights (allow user override via config)
        weights = config.get("memory_scoring_weights", _DEFAULT_WEIGHTS)
        w_sem  = float(weights.get("semantic",  self.DEFAULT_W_SEMANTIC))
        w_imp  = float(weights.get("importance", self.DEFAULT_W_IMPORTANCE))
        w_rec  = float(weights.get("recency",    self.DEFAULT_W_RECENCY))
        w_freq = float(weights.get("frequency",  self.DEFAULT_W_FREQUENCY))

        half_life = float(config.get("ebbinghaus_half_life_hours",
                                     self.DEFAULT_EBBINGHAUS_HALF_LIFE))
        ebbing_k  = float(config.get("ebbinghaus_k", self.DEFAULT_EBBINGHAUS_K))

        # v32: Try keyword-based search first for faster recall on specific queries
        all_rows = None
        query_words_set = set(query.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'what', 'who', 'how', 'why', 'when', 'where', 'can', 'could', 'would', 'should', 'i', 'me', 'my', 'you', 'your'}
        if len(query_words_set) >= 2 and not kind:
            try:
                # Try a targeted search with search_memories if available
                keyword_rows = await db.search_memories(query, limit=50)
                if keyword_rows and len(keyword_rows) >= 3:
                    all_rows = keyword_rows
            except (AttributeError, Exception):
                pass
        if all_rows is None:
            all_rows = await db.all_memories(kind=kind, limit=200)
        if not all_rows:
            return []

        # Compute query embedding once for cosine comparison
        query_emb = await embed(query)

        query_words = set(query.lower().split())
        now = time.time()
        max_access = 0  # track for normalising frequency

        # v28: Batch cosine computation — create numpy matrix of all embeddings
        # then compute similarities in one vectorized operation instead of per-memory loops
        _memories_with_emb = []  # (index, Memory, access_count, sim)
        _embedding_matrix = []

        for row in all_rows:
            mem = Memory.from_row(row)
            access_count = mem.access_count or 0
            if access_count > max_access:
                max_access = access_count

            # ── Semantic similarity ──────────────────────────
            if query_emb and mem.embedding:
                _memories_with_emb.append(None)  # placeholder
                _embedding_matrix.append(mem.embedding)
            else:
                # Fallback: keyword overlap (same behaviour as hash embedding)
                content_words = set(mem.content.lower().split())
                overlap = len(query_words & content_words)
                sim = min(1.0, overlap / max(1, len(query_words)))
                _memories_with_emb.append((mem, access_count, sim))
                _embedding_matrix.append(None)

        # Batch cosine: vectorized numpy operation for all embeddings at once
        if query_emb and _embedding_matrix:
            try:
                import numpy as _np
                q_vec = _np.asarray(query_emb, dtype=_np.float32)
                q_norm = _np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                emb_indices = []
                emb_vecs = []
                for idx, emb in enumerate(_embedding_matrix):
                    if emb is not None:
                        emb_indices.append(idx)
                        emb_vecs.append(emb)

                if emb_vecs:
                    mat = _np.asarray(emb_vecs, dtype=_np.float32)
                    # Normalize each row
                    norms = _np.linalg.norm(mat, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    mat_normed = mat / norms
                    # Batch cosine similarity
                    sims = mat_normed @ q_vec  # (n,)

                    for i, mat_idx in enumerate(emb_indices):
                        mem_row = all_rows[mat_idx]
                        mem = Memory.from_row(mem_row)
                        access_count = mem.access_count or 0
                        _memories_with_emb[mat_idx] = (mem, access_count, float(sims[i]))
            except Exception:
                # Fallback: compute individually
                for idx, emb in enumerate(_embedding_matrix):
                    if emb is not None:
                        try:
                            sim = cosine(query_emb, emb)
                        except Exception:
                            content_words = set(Memory.from_row(all_rows[idx]).content.lower().split())
                            overlap = len(query_words & content_words)
                            sim = min(1.0, overlap / max(1, len(query_words)))
                        mem = Memory.from_row(all_rows[idx])
                        access_count = mem.access_count or 0
                        _memories_with_emb[idx] = (mem, access_count, sim)

        # --- Single pass: compute final scores ---
        import heapq as _heapq
        scored_heap = []

        for item in _memories_with_emb:
            if item is None:
                continue
            mem, access_count, sim = item

            # ── Importance (direct) ─────────────────────────────────
            importance = mem.importance

            # ── Ebbinghaus recency ──────────────────────────────────
            age_seconds = now - (mem.last_accessed or mem.created_at or now)
            age_hours = max(0.0, age_seconds / 3600.0)
            recency = self._ebbinghaus(age_hours, half_life, ebbing_k)

            # ── Frequency (normalised) ──────────────────────────────
            freq_norm = (access_count / max_access) if max_access > 0 else 0.0

            # ── Composite score ─────────────────────────────────────
            score = (
                w_sem  * sim
                + w_imp  * importance
                + w_rec  * recency
                + w_freq * freq_norm
            )

            # v28: Use heapq.nlargest instead of full sort — O(n log k) vs O(n log n)
            if len(scored_heap) < k:
                _heapq.heappush(scored_heap, (score, id(mem), mem))
            elif score > scored_heap[0][0]:
                _heapq.heapreplace(scored_heap, (score, id(mem), mem))

        top_mems = [m for _, _, m in sorted(scored_heap, key=lambda x: x[0], reverse=True)]

        # v28: Update access counts in parallel instead of sequentially
        await asyncio.gather(
            *(db.update_memory_access(m.id) for m in top_mems),
            return_exceptions=True,
        )
        for m in top_mems:
            self._usage_tracker.record(m.id, task_type="recall")

        # ── Memory Tree hierarchical recall ──────────────────────────
        if config.get("enable_memory_tree_recall", True):
            try:
                from .memory_tree import MemoryTree
                _tree = MemoryTree()
                _tree_nodes = await _tree.recall(query=query, limit=max(2, k))
                if _tree_nodes:
                    for _tn in _tree_nodes:
                        _content = _tn.get("content", "") or ""
                        _summary = _tn.get("summary", "") or ""
                        _meta = _tn.get("metadata", {})
                        if isinstance(_meta, str):
                            try:
                                _meta = json.loads(_meta)
                            except Exception:
                                _meta = {}
                        _imp = float(_meta.get("importance", 0.5))
                        _root_title = _tn.get("root_title", "") or ""

                        _mem = Memory(
                            id=_tn.get("id", 0) + 1000000,
                            kind="tree_hierarchical",
                            content=f"[{_root_title}] {_summary or _content[:200]}",
                            importance=_imp,
                            tags=["memory_tree"],
                            created_at=_tn.get("created_at"),
                        )
                        top_mems.append(_mem)  # Append tree node
            except Exception:
                logger.debug("Memory tree recall failed (non-fatal)", exc_info=True)

        return top_mems

    async def short_term(self, limit: int = 15) -> list[dict]:
        """Get recent conversation messages."""
        if not self.session_id:
            return []
        rows = await db.get_messages(self.session_id, limit=limit)
        return rows

    async def user_profile(self, limit: int = 15) -> list[Memory]:
        """Get factual memories about the user.

        v31: Increased default limit from 8 to 15 for better cross-session recall.
        """
        rows = await db.top_factual_memories(limit=limit)
        return [Memory.from_row(r) for r in rows]

    async def consolidate(self, conversation: list[dict]) -> None:
        """After a task/conversation, distill key facts into long-term memory.

        Uses the LLM to extract important facts, preferences, decisions, and
        outcomes from the conversation. Deduplicates against existing
        memories before storing.
        """
        if not config.get("enable_long_term_memory", True):
            logger.debug("[memory] consolidate skipped - long term memory disabled")
            return
        if not conversation or len(conversation) < 2:
            logger.debug("[memory] consolidate skipped - not enough conversation messages")
            return

        # Ingest into Memory Tree for hierarchical storage
        if config.get("enable_memory_tree", True) and config.get("memory_tree_auto_ingest", True):
            try:
                transcript_parts = []
                for m in conversation[-8:]:
                    if m.get('content') and len(str(m.get('content', ''))) > 3:
                        transcript_parts.append(f"{m.get('role', '')}: {m.get('content', '')[:500]}")
                transcript_text = '\n'.join(transcript_parts)
                if transcript_text:
                    from .memory_tree import MemoryTree
                    tree = MemoryTree()
                    await tree.ingest(
                        content=transcript_text,
                        source="conversation",
                        metadata={"session_id": ""},
                        tags=["conversation", "auto"],
                    )
            except Exception as e:
                logger.debug("Memory tree ingest failed (non-fatal): %s", e)

        try:
            from ..core import llm
        except Exception as e:
            logger.warning("[memory] consolidate skipped - llm import failed: %s", e)
            return

        logger.debug("[memory] Starting consolidation with %d messages", len(conversation))

        # Build a compact transcript of the conversation
        transcript_parts = []
        for msg in conversation[-16:]:  # v32: Increased from 8 to 16 for better consolidation
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle content-parts format
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_parts)
            if not content or len(content) < 3:
                continue
            # Truncate long messages to keep the prompt manageable
            transcript_parts.append(f"{role}: {content[:500]}")

        if not transcript_parts:
            return

        transcript = "\n".join(transcript_parts)

        consolidation_prompt = (
            "Analyze this conversation and extract ALL important facts, preferences, "
            "decisions, user details, project context, or actionable conclusions.\n\n"
            "Pay SPECIAL attention to:\n"
            "- User's name, preferences, role, location (importance >= 0.9)\n"
            "- Project context, tech stack, work details (importance >= 0.8)\n"
            "- Technical decisions and procedures (importance >= 0.7)\n\n"
            "Return a JSON array of objects with these fields:\n"
            '- "content": the fact (string)\n'
            '- "kind": one of "semantic", "factual", "preference", "procedural", "episodic" (use "semantic" for user facts and preferences)\n'
            '- "importance": 0.0-1.0 (how important is this long-term)\n\n'
            "Rules:\n"
            "- Extract EVERY important piece of information — be thorough, not conservative\n"
            "- User preferences, name, role, projects are HIGH priority (importance >= 0.8)\n"
            "- Technical decisions and procedures are worth saving\n"
            "- Project context (what user is working on, their tech stack) is critical\n"
            "- User statements like 'remember this' or 'I always...' are VERY important\n"
            "- Skip greetings, filler, and already-known facts\n"
            "- Return empty array [] if nothing is worth remembering\n\n"
            f"Conversation:\n{transcript}\n\n"
            'Respond ONLY with the JSON array, no other text.'
        )

        try:
            logger.debug("[memory] Calling LLM for consolidation...")
            response = await llm.complete(
                prompt=consolidation_prompt,
                model=config.get("memory_model"),
                max_tokens=512,
                temperature=0.1,
            )
            raw = response.content.strip() if response.content else ""
            logger.debug("[memory] LLM response length: %d", len(raw))
            if not raw:
                logger.warning("[memory] Consolidation LLM returned empty response")
                return

            # Extract JSON - handle multiple formats
            json_match = None
            # Try: ```json ... ``` code block
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw, re.IGNORECASE)
            if json_match:
                raw = json_match.group(1)
            # Try: ``` ... ``` any code block
            if not json_match:
                json_match = re.search(r'```\s*([\s\S]*?)\s*```', raw)
                if json_match:
                    raw = json_match.group(1)
            # Try: raw JSON array
            if not json_match:
                json_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if json_match:
                    raw = json_match.group()

            if not raw.strip():
                logger.warning("[memory] Could not extract JSON from LLM response. Response preview: %s", raw[:300])
                return

            facts = json.loads(raw)
            if not isinstance(facts, list) or len(facts) == 0:
                logger.debug("[memory] No facts extracted from conversation")
                return

            saved_count = 0
            for fact in facts[:12]:  # v32: Increased from 8 to 12 for better fact extraction
                content = fact.get("content", "").strip()
                if not content or len(content) < 5:
                    continue
                kind = fact.get("kind", "semantic")
                # Normalize to semantic if unknown
                if kind not in ("semantic", "factual", "preference", "procedural", "episodic"):
                    kind = "semantic"
                importance = float(fact.get("importance", 0.6))
                importance = max(0.1, min(1.0, importance))

                mem_id = await self.remember(
                    content=content,
                    kind=kind,
                    importance=importance,
                    tags=["consolidated", "auto"],
                )
                saved_count += 1

            if saved_count > 0:
                await emit(
                    "memory_consolidated",
                    session_id=self.session_id,
                    facts_extracted=saved_count,
                )

        except Exception as e:
            logger.debug("Consolidation failed (non-fatal): %s", e)

    # Usage tracker ───────────────────────────────────────────────

    def record_usage(self, memory_id: int, task_type: str = "general") -> None:
        self._usage_tracker.record(memory_id, task_type)


class _UsageTracker:
    def __init__(self):
        self._data: dict[int, dict[str, Any]] = {}

    def record(self, memory_id: int, task_type: str = "general") -> None:
        if memory_id not in self._data:
            self._data[memory_id] = {"count": 0, "tasks": set()}
        self._data[memory_id]["count"] += 1
        self._data[memory_id]["tasks"].add(task_type)

    def flush(self) -> None:
        self._data.clear()