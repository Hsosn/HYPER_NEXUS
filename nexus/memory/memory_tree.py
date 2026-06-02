"""
Memory Tree — hierarchical long-term memory system.

Replaces flat long-term memory with a tree structure inspired by OpenHuman:
- Automatic content chunking and tree ingestion
- State machine: buffered → sealed (age-based)
- Hierarchical summarization (topic trees)
- Daily digest for each day's activity
- Source trees, topic trees, and global trees
- Entity extraction and cross-linking
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class MemoryTree:
    """Hierarchical long-term memory with tree-based organization.

    Design:
    - Content is chunked into nodes organized under roots (documents/conversations).
    - Nodes progress through states: buffered → sealed (after age threshold).
    - Summarization happens asynchronously via background jobs.
    - Topic trees group related content across roots.
    - Daily digest provides a day-level summary of activity.
    """

    def __init__(self, llm_func=None, chunk_size: int | None = None):
        """
        Args:
            llm_func: Optional async callable(prompt: str) -> str for summarization.
            chunk_size: Target token size for content chunking. Falls back to
                config key ``memory_tree_max_chunk_tokens`` (default 500).
        """
        from .. import config as _cfg
        self._llm_func = llm_func
        self._chunk_size = chunk_size or _cfg.get("memory_tree_max_chunk_tokens", 500)

    # ── Content Ingestion ─────────────────────────────────────────────────────

    async def ingest(self, content: str, source: str = "",
                     root_id: str | None = None,
                     metadata: dict | None = None,
                     tags: list[str] | None = None,
                     importance: float = 0.5) -> dict[str, Any]:
        """Ingest content into the memory tree.

        Creates or reuses a root node, chunks content, and inserts nodes.

        Args:
            content: The text content to store.
            source: Source description (e.g. "conversation", "journal").
            root_id: Optional explicit root ID. Auto-generated if not provided.
            metadata: Optional metadata dict.
            tags: Optional tags for the content.
            importance: Importance score 0-1.

        Returns:
            dict with root_id, node_count, chunk_count
        """
        from nexus.memory.tree_db import (
            tree_create_root, tree_insert_node, tree_upsert_chunk,
            tree_get_root, tree_update_root, compute_chunk_hash,
        )

        if not root_id:
            root_id = f"tree_{uuid.uuid4().hex[:12]}_{int(time.time())}"

        if not content or not content.strip():
            return {"root_id": root_id, "node_count": 0, "chunk_count": 0}

        # Create or update root
        existing_root = await tree_get_root(root_id)
        if existing_root:
            await tree_update_root(
                root_id,
                metadata={**(json.loads(existing_root.get("metadata", "{}") or "{}")),
                          **(metadata or {}),
                          "tags": tags or [],
                          "importance": max(
                              importance,
                              json.loads(existing_root.get("metadata", "{}") or "{}").get("importance", 0),
                          )},
            )
        else:
            root_metadata = dict(metadata or {})
            if tags:
                root_metadata["tags"] = tags
            root_metadata["importance"] = importance
            await tree_create_root(
                root_id=root_id,
                root_type="source",
                title=source or f"Memory {time.strftime('%Y-%m-%d %H:%M')}",
                metadata=root_metadata,
            )

        # Chunk content
        chunks = self._chunk_content(content)
        chunk_hashes = set()
        node_ids = []

        for i, chunk_text in enumerate(chunks):
            # Deduplicate by content hash
            chunk_hash = await tree_upsert_chunk(
                chunk_text,
                token_count=len(chunk_text.split()),
                source=source or None,
            )

            if chunk_hash in chunk_hashes and i > 0:
                continue
            chunk_hashes.add(chunk_hash)

            # Determine parent (chain chunks sequentially)
            parent_id = node_ids[-1] if node_ids else None

            node_id = await tree_insert_node(
                root_id=root_id,
                content=chunk_text,
                parent_id=parent_id,
                tree_type="source",
                level=i,
                state="buffered",
                chunk_hash=chunk_hash,
                metadata={
                    "source": source,
                    "tags": tags or [],
                    "importance": importance,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            node_ids.append(node_id)

        # Create summarization job for the root
        if node_ids:
            from nexus.memory.tree_db import tree_create_job
            await tree_create_job(node_ids[0], root_id, "summarize")

        return {
            "root_id": root_id,
            "node_count": len(node_ids),
            "chunk_count": len(chunk_hashes),
        }

    # ── Content Chunking ──────────────────────────────────────────────────────

    def _chunk_content(self, content: str) -> list[str]:
        """Split content into chunks at sentence/paragraph boundaries.

        Targets chunks around self._chunk_size tokens (words for simplicity).
        """
        if not content:
            return []

        # Try splitting by paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content.strip()]

        chunks = []
        current = []

        for para in paragraphs:
            words = para.split()
            current_words = sum(len(c.split()) for c in current)

            if current_words + len(words) <= self._chunk_size:
                current.append(para)
            else:
                if current:
                    chunks.append("\n\n".join(current))
                # If a single paragraph is longer than chunk_size, split by sentences
                if len(words) > self._chunk_size:
                    sentences = self._split_sentences(para)
                    sent_buf = []
                    for sent in sentences:
                        if sum(len(s.split()) for s in sent_buf) + len(sent.split()) <= self._chunk_size:
                            sent_buf.append(sent)
                        else:
                            if sent_buf:
                                chunks.append(" ".join(sent_buf))
                            sent_buf = [sent] if len(sent.split()) <= self._chunk_size * 2 else [sent]
                    if sent_buf:
                        current = sent_buf
                    else:
                        current = []
                else:
                    current = [para]

        if current:
            chunks.append("\n\n".join(current))

        return chunks if chunks else [content]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    # ── Hierarchical Recall ───────────────────────────────────────────────────

    async def recall(self, query: str = "", limit: int = 20,
                     state: str | None = "sealed") -> list[dict[str, Any]]:
        """Recall memories from the tree with optional query matching.

        Args:
            query: Search query for keyword matching.
            limit: Maximum results.
            state: Filter by node state (None = all states).

        Returns:
            List of node dicts with content, summary, root info, etc.
        """
        from nexus.memory.tree_db import (
            _connect, _row_to_dict,
        )

        async with _connect() as conn:
            conditions = []
            params = []

            if state:
                conditions.append("n.state = ?")
                params.append(state)

            if query and query.strip():
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
                              'what', 'who', 'how', 'why', 'when', 'where', 'can', 'could',
                              'i', 'me', 'my', 'you', 'your', 'this', 'that', 'it', 'and',
                              'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                words = [w.lower() for w in query.split()
                         if w.lower() not in stop_words and len(w) >= 2]
                if words:
                    word_conditions = []
                    for word in words[:5]:
                        word_conditions.append("n.content LIKE ?")
                        params.append(f"%{word}%")
                    if word_conditions:
                        conditions.append(f"({' OR '.join(word_conditions)})")

            where_clause = " AND ".join(conditions) if conditions else "TRUE"

            sql = f"""
                SELECT n.*, r.title as root_title, r.summary as root_summary,
                       r.root_type, r.metadata as root_metadata
                FROM tree_nodes n
                JOIN tree_roots r ON n.root_id = r.root_id
                WHERE {where_clause}
                ORDER BY n.level ASC, n.id DESC
                LIMIT ?
            """
            params.append(limit)
            c = await conn.execute(sql, tuple(params))
            rows = await c.fetchall()

        results = []
        for row in rows:
            d = _row_to_dict(row)
            # Parse metadata JSON
            for field in ("metadata", "root_metadata"):
                if isinstance(d.get(field), str):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)

        return results

    async def recall_summaries(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get top-level summaries from sealed nodes.
        
        Returns nodes that have summaries, for compact context building.
        """
        from nexus.memory.tree_db import _connect, _row_to_dict

        async with _connect() as conn:
            c = await conn.execute(
                """SELECT n.id, n.root_id, n.summary, n.tree_type, n.level,
                          r.title as root_title, r.summary as root_summary
                   FROM tree_nodes n
                   JOIN tree_roots r ON n.root_id = r.root_id
                   WHERE n.state = 'sealed' AND n.summary IS NOT NULL
                   ORDER BY n.updated_at DESC
                   LIMIT ?""",
                (limit,),
            )
            rows = await c.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def recall_by_root(self, root_id: str) -> list[dict[str, Any]]:
        """Get all nodes for a specific root, in tree order."""
        from nexus.memory.tree_db import tree_get_root_nodes, tree_get_root

        root = await tree_get_root(root_id)
        if not root:
            return []

        nodes = await tree_get_root_nodes(root_id)
        return nodes

    # ── Topic Trees ───────────────────────────────────────────────────────────

    async def get_topic_tree(self, topic: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find memories related to a topic across all roots.

        Uses keyword matching on content and summaries.
        """
        from nexus.memory.tree_db import _connect, _row_to_dict

        async with _connect() as conn:
            c = await conn.execute(
                """SELECT n.*, r.title as root_title, r.summary as root_summary
                   FROM tree_nodes n
                   JOIN tree_roots r ON n.root_id = r.root_id
                   WHERE (n.content LIKE ? OR n.summary LIKE ?)
                     AND n.state = 'sealed'
                   ORDER BY n.updated_at DESC
                   LIMIT ?""",
                (f"%{topic}%", f"%{topic}%", limit),
            )
            rows = await c.fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row)
            if isinstance(d.get("metadata"), str):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except Exception:
                    pass
            results.append(d)
        return results

    async def extract_entities(self, content: str,
                               root_id: str | None = None) -> list[str]:
        """Extract named entities from content (simple heuristic).

        Returns list of capitalized multi-word phrases that may be entities.
        """
        import re
        # Find capitalized phrases (simple heuristic for entity extraction)
        candidates = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', content)
        # Filter out common false positives
        common = {"I Am", "I Have", "It Is", "This Is", "There Is", "One Of",
                  "Most Of", "Some Of", "A Lot", "Due To", "Based On"}
        entities = list(set(c for c in candidates if c not in common and len(c) > 5))

        # Store entities in tree_entities
        if entities:
            from nexus.memory.tree_db import tree_upsert_entity
            for entity in entities[:20]:  # Limit to 20 per call
                await tree_upsert_entity(entity, root_id=root_id)

        return entities

    # ── Daily Digest ──────────────────────────────────────────────────────────

    async def get_daily_digest(self, days_ago: int = 0) -> dict[str, Any]:
        """Get a digest of all memories created on a specific day.

        Args:
            days_ago: 0 = today, 1 = yesterday, etc.

        Returns:
            dict with date, root_count, node_count, summary, and nodes.
        """
        from nexus.memory.tree_db import tree_get_daily_nodes, tree_get_root

        nodes = await tree_get_daily_nodes(days_ago)
        if not nodes:
            return {
                "date": time.strftime("%Y-%m-%d", time.gmtime(time.time() - days_ago * 86400)),
                "root_count": 0,
                "node_count": 0,
                "summary": "No memories recorded.",
                "nodes": [],
            }

        # Group by root
        root_ids = set(n["root_id"] for n in nodes)
        roots = {}
        for rid in root_ids:
            root = await tree_get_root(rid)
            if root:
                roots[rid] = root

        # Build digest text
        summaries = []
        for rid in root_ids:
            root = roots.get(rid)
            if root and root.get("summary"):
                summaries.append(f"- {root.get('title', 'Untitled')}: {root['summary']}")

        all_text = "\n".join(n["content"] for n in nodes)
        digest = {
            "date": time.strftime("%Y-%m-%d", time.gmtime(
                nodes[0].get("created_at", time.time()))),
            "root_count": len(root_ids),
            "node_count": len(nodes),
            "summary": "\n".join(summaries) if summaries else f"{len(nodes)} memories recorded.",
            "char_count": len(all_text),
            "roots": {
                rid: {
                    "title": roots[rid].get("title", "Untitled") if rid in roots else "Unknown",
                    "node_count": sum(1 for n in nodes if n["root_id"] == rid),
                }
                for rid in root_ids
            },
        }
        return digest

    # ── Summarization (via LLM callback) ──────────────────────────────────────

    async def summarize_node(self, node_id: int) -> str | None:
        """Generate a summary for a node using the LLM callback."""
        if not self._llm_func:
            logger.warning("No LLM function provided; cannot summarize node %d", node_id)
            return None

        from nexus.memory.tree_db import tree_get_node, tree_update_node

        node = await tree_get_node(node_id)
        if not node:
            return None

        content = node["content"]
        if not content or len(content.strip()) < 20:
            summary = content[:200]
            await tree_update_node(node_id, summary=summary)
            return summary

        prompt = (
            f"Summarize the following memory content in 1-2 sentences:\n\n"
            f"{content[:2000]}\n\nSummary:"
        )
        try:
            summary = await self._llm_func(prompt)
            summary = summary.strip()[:500]
            await tree_update_node(node_id, summary=summary)
            return summary
        except Exception as e:
            logger.error("Failed to summarize node %d: %s", node_id, e)
            return None

    async def summarize_root(self, root_id: str) -> str | None:
        """Generate a summary for an entire root from its node summaries."""
        if not self._llm_func:
            return None

        from nexus.memory.tree_db import tree_get_root_nodes, tree_get_root, tree_update_root

        nodes = await tree_get_root_nodes(root_id, state="sealed")
        if not nodes:
            nodes = await tree_get_root_nodes(root_id)

        summaries = []
        for node in nodes:
            s = node.get("summary") or node["content"][:100]
            summaries.append(s)

        if not summaries:
            return None

        if len(summaries) == 1 and len(summaries[0]) < 200:
            await tree_update_root(root_id, summary=summaries[0])
            return summaries[0]

        combined = "\n".join(f"- {s}" for s in summaries[:10])
        prompt = (
            f"Summarize the following collection of memory fragments in 2-3 sentences:\n\n"
            f"{combined}\n\nOverall summary:"
        )
        try:
            summary = await self._llm_func(prompt)
            summary = summary.strip()[:1000]
            await tree_update_root(root_id, summary=summary)
            return summary
        except Exception as e:
            logger.error("Failed to summarize root %s: %s", root_id, e)
            return None

    async def run_summarization_jobs(self, max_jobs: int = 10) -> int:
        """Process pending summarization jobs. Returns count processed."""
        from nexus.memory.tree_db import (
            tree_list_pending_jobs, tree_update_job, tree_get_node,
        )

        jobs = await tree_list_pending_jobs(job_type="summarize", limit=max_jobs)
        processed = 0

        for job in jobs:
            job_id = job["id"]
            node_id = job["node_id"]
            root_id = job.get("root_id")

            await tree_update_job(job_id, "running")

            try:
                summary = await self.summarize_node(node_id)
                if summary:
                    # Also try to update root summary
                    if root_id:
                        await self.summarize_root(root_id)
                    await tree_update_job(job_id, "completed")
                    processed += 1
                else:
                    await tree_update_job(job_id, "completed")
                    processed += 1
            except Exception as e:
                logger.error("Job %d failed: %s", job_id, e)
                await tree_update_job(job_id, "failed", str(e))

        return processed

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Get memory tree statistics."""
        from nexus.memory.tree_db import tree_get_stats

        stats = await tree_get_stats()
        stats["chunk_size"] = self._chunk_size
        stats["has_llm"] = self._llm_func is not None
        return stats
