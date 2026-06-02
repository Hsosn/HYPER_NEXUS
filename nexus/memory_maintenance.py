"""
Memory Tree Maintenance — seals old buffered nodes and processes pending
summarization jobs to keep the memory tree healthy.

Called by the periodic heartbeat; does not depend on Celery.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


async def run_tree_maintenance() -> dict:
    """Run one pass of memory tree maintenance.

    Steps:
    1. Seal buffered nodes older than 24h → creates 'summarize' jobs
    2. Process pending summarization jobs (use content as summary for now)
    3. Process pending seal jobs (mark nodes as sealed)
    4. Clean up orphaned chunks and old failed jobs

    Returns a dict with sealed, summarized, and orphans_removed counts.
    """
    from .memory import tree_db
    from .memory import db  # noqa: F401 — ensure pool is initialized

    sealed = 0
    summarized = 0
    skipped = 0
    failed = 0

    try:
        # ── Step 1: Seal old buffered nodes ────────────────────────────
        sealed = await tree_db.tree_seal_old_buckets(
            max_age_days=1,
            batch_size=100,
        )
        if sealed:
            logger.info("[tree_maint] Sealed %d buffered nodes", sealed)

        # ── Step 2: Process pending jobs ───────────────────────────────
        pending_jobs = await tree_db.tree_list_pending_jobs(limit=50)
        for job in pending_jobs:
            job_id = job["id"]
            job_type = job.get("job_type", "")
            node_id = job.get("node_id")

            try:
                if job_type == "summarize":
                    ok = await _process_summarize_job(tree_db, node_id, job_id)
                    if ok:
                        summarized += 1
                    else:
                        skipped += 1
                elif job_type == "seal":
                    # Mark the node as sealed
                    if node_id:
                        await tree_db.tree_update_node(node_id, state="sealed")
                    await tree_db.tree_update_job(job_id, "completed")
                    sealed += 1
                else:
                    # Unknown job type — mark as completed to prevent backlog
                    await tree_db.tree_update_job(
                        job_id, "failed",
                        error=f"Unknown job type: {job_type}",
                    )
                    failed += 1
            except Exception as exc:
                logger.warning(
                    "[tree_maint] Job %d (type=%s) failed: %s",
                    job_id, job_type, exc,
                )
                try:
                    await tree_db.tree_update_job(
                        job_id, "failed", error=str(exc)[:200],
                    )
                except Exception:
                    pass
                failed += 1

        # ── Step 3: Clean up orphaned chunks and old jobs ───────────────
        orphans_removed = await tree_db.tree_cleanup_orphans()

        total = sealed + summarized + skipped
        if total:
            logger.info(
                "[tree_maint] Done: %d sealed, %d summarized, %d skipped, "
                "%d failed, %d orphans cleaned",
                sealed, summarized, skipped, failed, orphans_removed,
            )

        return {
            "sealed": sealed,
            "summarized": summarized,
            "skipped": skipped,
            "failed": failed,
            "orphans_removed": orphans_removed,
        }

    except Exception as exc:
        logger.error("[tree_maint] Maintenance run failed: %s", exc)
        return {
            "sealed": sealed,
            "summarized": summarized,
            "skipped": skipped,
            "failed": failed,
            "orphans_removed": 0,
            "error": str(exc)[:200],
        }


async def _process_summarize_job(
    tree_db: Any, node_id: int | None, job_id: int,
) -> bool:
    """Process a single summarization job: mark node summary and complete job.

    For now, uses the first 200 chars of the node content as a quick summary.
    Can be upgraded to use LLM summarization later.
    """
    if node_id is None:
        await tree_db.tree_update_job(
            job_id, "failed", error="node_id is None",
        )
        return False

    node = await tree_db.tree_get_node(node_id)
    if not node:
        await tree_db.tree_update_job(
            job_id, "failed", error=f"Node {node_id} not found",
        )
        return False

    content: str = node.get("content") or ""
    existing_summary: str | None = node.get("summary")

    if existing_summary:
        # Already summarized — just mark complete
        await tree_db.tree_update_job(job_id, "completed")
        return True

    if len(content) < 50:
        # Too short — use content as summary
        summary = content[:200]
    else:
        # Use first meaningful sentence(s) as quick summary
        # Try to extract the first 1-2 sentences
        summary = _extract_summary(content)

    await tree_db.tree_update_node(node_id, summary=summary)
    await tree_db.tree_update_job(job_id, "completed")
    return True


def _extract_summary(text: str, max_chars: int = 200) -> str:
    """Extract a concise summary from text using the first meaningful sentences."""
    # Strip markdown headings
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip code blocks, empty lines, and heading markers
        if stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            # Include the heading text as context
            clean_lines.append(stripped.lstrip("#").strip())
            continue
        if stripped:
            clean_lines.append(stripped)

    full_text = " ".join(clean_lines)

    if len(full_text) <= max_chars:
        return full_text

    # Try to split at sentence boundaries
    sentences = full_text.replace("! ", ". ").replace("? ", ". ").split(". ")
    summary = ""
    for sent in sentences:
        if len(summary) + len(sent) + 2 <= max_chars:
            if summary:
                summary += ". " + sent
            else:
                summary = sent
        else:
            if not summary:
                summary = sent[:max_chars]
            break

    if summary:
        return summary.strip()
    return full_text[:max_chars]
