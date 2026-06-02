"""


Nexus Framework Reasoning Engine: the unified adaptive reasoning core.


A single intelligent loop that adaptively applies the best reasoning


strategy per-step based on task complexity  - no manual mode selection needed.


Flow:


1. Retrieve relevant long-term memories and recent conversation


2. Nexus Reasoning Loop (unified adaptive):


     LLM proposes next action (tool call OR final answer)


     If tool call  - execute  - feed observation back


     Adaptive strategy: step-by-step reasoning, branch exploration,


     planning, simulation, and self-reflection as needed


     Repeat until final answer or max iterations


3. Save important info to long-term memory


4. (If enabled) Self-reflect on performance


The Nexus Framework automatically adapts reasoning strategy per task:


- Simple tasks: direct response (fast path)


- Analysis tasks: step-by-step reasoning with tool access


- Creative tasks: branch exploration with best-path selection


- Complex multi-step tasks: planning, simulation, and backtracking


- All tasks: built-in hallucination detection, loop detection, error recovery


v33: The Nexus Framework unifies multiple reasoning strategies  - action-observation


loops, step-by-step chain reasoning, branch exploration with backtracking,


and search with simulation and evaluation  - into one adaptive engine.


The engine automatically applies the optimal approach per step.


"""


from __future__ import annotations


import asyncio


import ast


import json


import logging


import re


import sys


import traceback

import threading


import time as _time
from typing import Any


from .. import config


from ..core import llm
from .adhd_module import maybe_fire_adhd, ADHDReasoningModule


from ..events import emit


from ..memory import MemoryManager, db, Memory


from ..self_improve import (


    assess_response_quality,


    process_quality_assessment,


    record_outcome,


)


from ..tasks.task_state import (


    get_checkpoint_manager,


    CheckpointManager,


)


from ..tools import REGISTRY
from ..tasks.task_registry import register_task as _register_bg_task


logger = logging.getLogger(__name__)

# ── Sub-module imports ──────────────────────────────────────
from .complexity import (
    _HIGH_COMPLEXITY_INTENTS, _HIGH_COMPLEXITY_VERBS,
    _MULTI_TASK_CONNECTORS, _MEDIUM_COMPLEXITY_DESIGN,
    _assess_complexity,
)
from .integrations import (
    _ALL_KNOWN, _get_connected_integrations_summary,
)
from .cache import (
    _prompt_caches, _prompt_cache_compressed_caches,
    _PROMPT_CACHE_TTL, _db_prompt_sub_cache, _self_awareness_cache,
    _DB_PROMPT_SUB_CACHE_TTL, _SELF_AWARENESS_CACHE_TTL,
    _prefilter_cache, _CACHE_LOCK, _PREFILTER_CACHE_TTL,
    _prefilter_cache_tool_count,
    _invalidate_prompt_cache, _invalidate_all_prompt_caches,
)
from .hallucination import _HALLUCINATION_PATTERNS
from .text_cleaning import (
    _ITARKUP_STRIP_RE, _strip_python_leaks,
    _BASELINE_STRIP_PATTERNS, _find_matching_brace,
    _strip_nested_json_blocks, _build_funcall_strip_re,
    _TEXT_TOOL_FUNCCALL_STRIP_RE, _TEXT_TOOL_FUNCCALL_STRIP_LOCK,
    _strip_tool_call_markup, _strip_leaked_tool_names,
    _strip_dsml_from_text, _sanitize_error_msg,
    _validate_tool_call_pairs, _truncate_params_for_display,
)
from .tool_parsing import (
    _TEXT_TOOL_DIRECT_CALL_RE, _TEXT_TOOL_FORMAT1_RE,
    _TEXT_TOOL_FORMAT1B_RE, _TEXT_TOOL_FORMAT1B_ARG_RE,
    _TEXT_TOOL_FORMAT3_RE, _TEXT_TOOL_FORMAT4_RE,
    _TEXT_TOOL_PROSE_RE1, _TEXT_TOOL_PROSE_RE2,
    _TEXT_TOOL_CODEBLOCK_RE, _TEXT_TOOL_FUNCCALL_RE,
    _TEXT_TOOL_INVOKE_RE, _TEXT_TOOL_INVOKE_PARAM_RE,
    _MARKDOWN_FENCE_STRIP_RE, _INLINE_JSON_RE, _KV_PAIR_RE,
    _SIMPLE_NUMBERED_LIST_RE,
)
from .tool_keywords import _TOOL_KEYWORD_MAP
from .skills import SKILLS


class ReasoningEngine:


    def __init__(self, session_id: str, is_sub_agent: bool = False):


        self.session_id = session_id


        self._is_sub_agent = is_sub_agent


# Sub-agents get a MemoryManager too for reading user profile
        self.memory = MemoryManager(session_id) if not is_sub_agent else MemoryManager(session_id)


        # v15.2: Tool error recovery state


        self._tool_retry_count: dict[str, int] = {}  # tool_name  - consecutive failures


        self._max_tool_retries: int = 2


        # v22: Session tracking for self-awareness


        self._session_tool_usage: dict[str, dict] = {}  # tool_name -> {"calls": int, "success": int, "fail": int, "total_ms": int}


        self._session_start_time: float = _time.time()


        self._session_message_count: int = 0


        # v26: Loop detection  - track recent tool calls to detect stuck loops


        self._recent_tool_calls: list[str] = []  # last N "tool_name:args_hash" strings


        self._loop_detection_window: int = 10  # increased for document creation tools


        self._loop_threshold: int = config.get("loop_detection_threshold", 3)


        # v26: Context compression  - track total message chars for progressive compression


        self._context_char_count: int = 0


        self._context_compression_threshold: int = config.get("context_compression_threshold", 20000)


        # Proactive cross-session memory: loaded once on first respond() call


        self._cross_session_memories_loaded: bool = False


        self._cross_session_loading: bool = False


        self._cached_user_profile: list[Memory] = []


        self._cached_cross_session_facts: list[Memory] = []


        # v31: Recent session summaries for cross-session continuity


        self._cached_recent_sessions: list[dict] = []


        # v31: Task plan tracking for multi-step execution


        self._current_task_plan: list[str] = []


        self._completed_plan_steps: list[str] = []


        # v39: Project context for cross-session persistence


        self._cached_project_context: str = ""


        # v42: Checkpoint/resume for long-running tasks


        self._checkpoint_mgr: CheckpointManager | None = None


        # v42: Quality assessment counter


        self._quality_check_counter: int = 0

        # v46: Stop/interrupt flag - set when user commands halt mid-response
        self._stop_requested: bool = False

        # vFIX: Empty search result tracker - prevents repeated searches for non-existent data
        self._empty_search_counter: int = 0

        # vFIX: Negative result cache for search/file tools - prevents re-executing known-empty queries
        self._negative_cache: dict[str, float] = {}  # key -> timestamp of cache entry
        self._negative_cache_max: int = 64

        # ADHD cross-domain reasoning module (optional)
        self._adhd_module = ADHDReasoningModule(llm) if not is_sub_agent else None
        self._adhd_block = None  # Cached ADHD prompt block, refreshed per-loop-iteration
        self._adhd_tool_boost: dict[str, float] = {}  # ADHD-recommended tools with confidence scores

        # Tool usage intelligence: per-session and cumulative tracking
        self._tool_usage_tracker: dict[str, dict[str, float]] = {}  # tool_name -> {calls, success, fail, total_ms, avg_ms}

        # vFIX: Tool failure pattern tracking — detect repeated tool+error combos
        self._tool_failure_patterns: dict[str, int] = {}  # "tool:error_category:error_prefix" -> count
        self._failure_pattern_warning_injected: set[str] = set()  # patterns already injected into messages

    # ---------------- Main entry ----------------


    async def _load_cross_session_memories(self) -> None:


        """Proactively load cross-session memories on first interaction.


        This ensures the agent ALWAYS knows about the user from previous sessions,


        even if the current query doesn't semantically match stored facts.


        v31 deep optimization + v38 non-blocking → v49 blocking:


        - Loads inline so first response has memory immediately (v49 fix)


        - Increased profile limit from 20 to 30 for better recall


        - Increased cross-session facts from 15 to 25


        - Added loading of ALL high-importance memories (importance >= 0.7)


          regardless of session_id, so critical facts are never lost


        - Added session continuity: loads last session's key topics


        - Include memories with NULL/empty session_id (stored by remember tool)


        - Deduplicate profile vs cross-session memories


        """


        if self._cross_session_memories_loaded:


            return


        if self._cross_session_loading:


            return


        self._cross_session_loading = True


        async def _load():


            try:


                from ..memory import db as _db


                profile = await self.memory.user_profile(limit=30)


                self._cached_user_profile = profile


                all_mems = await _db.all_memories(limit=100)


                cross_session = []


                current_session = self.session_id


                profile_contents = {m.content[:80] for m in profile} if profile else set()


                high_importance = []


                for row in all_mems:


                    mem = Memory.from_row(row) if isinstance(row, dict) else row


                    importance = mem.importance if hasattr(mem, 'importance') else float(row.get('importance', 0.5))


                    if importance >= 0.7:


                        high_importance.append(mem)


                for row in all_mems:


                    row_session = row.get("session_id") or ""


                    mem = Memory.from_row(row) if isinstance(row, dict) else row


                    mem_content = mem.content if hasattr(mem, 'content') else mem.get('content', '')


                    if row_session == current_session:


                        continue


                    if mem_content[:80] in profile_contents:


                        continue


                    if any(h.content[:80] == mem_content[:80] for h in high_importance):


                        continue


                    cross_session.append(mem)


                combined = []


                seen_contents = set()


                for mem in high_importance + cross_session:


                    mem_c = mem.content if hasattr(mem, 'content') else ''


                    key = mem_c[:80]


                    if key not in seen_contents and mem_c:


                        seen_contents.add(key)


                        combined.append(mem)


                    if len(combined) >= 25:


                        break


                self._cached_cross_session_facts = combined


                try:


                    sessions = await _db.list_sessions(limit=5)


                    recent_sessions = []


                    for s in sessions:


                        sid = s.get('id', '')


                        if sid and sid != current_session:


                            msgs = await _db.get_messages(sid, limit=6)


                            if msgs:


                                last_user_msgs = []


                                for m in msgs:


                                    if m.get('content'):


                                        role = m.get('role', '')


                                        content = m.get('content', '')[:150]


                                        if role == 'user':


                                            last_user_msgs.append(f"User: {content}")


                                        elif role == 'assistant' and len(last_user_msgs) < 4:


                                            last_user_msgs.append(f"Assistant: {content[:80]}")


                                if last_user_msgs:


                                    recent_sessions.append({


                                        'session_id': sid,


                                        'last_topics': last_user_msgs[:2],


                                        'updated_at': s.get('updated_at', 0),


                                    })


                    self._cached_recent_sessions = recent_sessions[:3]


                except Exception:


                    self._cached_recent_sessions = []


            except Exception as e:


                import logging as _logging


                _logging.getLogger(__name__).warning(


                    "Failed to load cross-session memories: %s", e


                )


            finally:


                self._cross_session_memories_loaded = True


                self._cross_session_loading = False


        # v49: Load inline so first response has memory immediately (was background task)
        await _load()


    async def _load_project_context(self, user_input: str) -> None:


        """Load project context from PROJECT.md if it exists.


        v39 (fix): Load unconditionally at session start if any PROJECT.md exists,


        same as cross-session memories. This ensures project context is always


        available without relying on inconsistent keyword matching.


        """


        if self._cached_project_context:


            return


        try:


            from pathlib import Path as _P


            workspace = _P(config.BASE_DIR) / "data" / "workspace"


            if not workspace.exists():


                return


            for entry in workspace.iterdir():


                if entry.is_dir() and (entry / "PROJECT.md").exists():


                    proj_md = entry / "PROJECT.md"


                    try:


                        content = proj_md.read_text(encoding="utf-8")


                        if len(content) > 100:


                            self._cached_project_context = (


                                f"## Project Context: {entry.name}\n\n{content}"


                            )


                            break


                    except Exception:
                        logger.warning("Failed to read project context file", exc_info=True)

        except Exception:
            logger.warning("Failed to scan workspace for project context", exc_info=True)


    def _detect_loop(self, tool_name: str, args: dict) -> bool:


        """v38: Detect if the agent is stuck in a loop making the same tool call.


        


        Returns True if the same tool+args has been called >= _loop_threshold times


        within the last _loop_detection_window calls.


        


        v38 fixes:


        - Check BEFORE appending (off-by-one fix: old code counted the current call)


        - Also detect same-tool-name loops (even with varying args) at a higher threshold


        - Normalize args before hashing to catch semantically similar calls


        """


        import hashlib as _hashlib


        


        # Normalize args: strip trailing punctuation/whitespace from string values


        # This catches "latest news today" vs "latest news today." style variations


        normalized_args = {}


        for k, v in args.items():


            if isinstance(v, str):


                normalized_args[k] = v.strip().rstrip(".!?,;:")


            else:


                normalized_args[k] = v


        


        args_hash = _hashlib.md5(json.dumps(normalized_args, sort_keys=True).encode()).hexdigest()[:16]


        call_key = f"{tool_name}:{args_hash}"


        


        # v38: Count BEFORE appending to avoid off-by-one (counting the current call)


        count = self._recent_tool_calls.count(call_key)


        


        # Also check same-tool-name frequency (catches loops with slightly varying args)


        tool_count = sum(1 for ck in self._recent_tool_calls if ck.split(":")[0] == tool_name)


        


        self._recent_tool_calls.append(call_key)


        if len(self._recent_tool_calls) > self._loop_detection_window:


            self._recent_tool_calls = self._recent_tool_calls[-self._loop_detection_window:]


        


        # Exact same call repeated >= threshold


        if count >= self._loop_threshold:


            return True


        


        # Same tool name called >= 5 times with varying args  - likely a loop


        if tool_count >= 5 and tool_name not in ("web_search", "fetch_url", "file_read"):


            return True


        


        return False


    


    def _remove_last_tool_call(self, tool_name: str, args: dict) -> None:


        """Remove the last occurrence of a tool call from recent_tool_calls.


        


        Used when a loop is detected and the tool call is not actually executed,


        to prevent false positives in future loop detection.


        """


        import hashlib as _hashlib


        _norm_args = {}
        for _k, _v in args.items():
            if isinstance(_v, str):
                _norm_args[_k] = _v.strip().rstrip(".!?,;:")
            else:
                _norm_args[_k] = _v
        args_hash = _hashlib.md5(json.dumps(_norm_args, sort_keys=True).encode()).hexdigest()[:16]


        call_key = f"{tool_name}:{args_hash}"


        # Remove from the end (most recent) to avoid affecting older calls


        for i in range(len(self._recent_tool_calls) - 1, -1, -1):


            if self._recent_tool_calls[i] == call_key:


                self._recent_tool_calls.pop(i)


                break


    def _compress_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:


        """v28: Progressive context compression for longer task execution.


        When the total message context exceeds _context_compression_threshold,


        compresses older tool results and assistant messages to save tokens


        while keeping the most recent context intact.


        v28 improvements:


        - Smart truncation: detect JSON/code structures and truncate at boundaries


        - Preserve error messages better (they're often critical for debugging)


        - Compress user messages too when they're very long


        """


        total_chars = sum(len(str(m.get("content", ""))) for m in messages)


        self._context_char_count = total_chars


        if total_chars < self._context_compression_threshold:


            return messages  # No compression needed


        # Strategy: compress older messages, keep recent ones intact


        # Keep: system messages, last 8 messages verbatim, compress the rest


        if len(messages) <= 10:


            return messages


        compressed = []


        recent_cutoff = max(0, len(messages) - 12)  # v32: Increased from 8 to protect memory context


        for i, msg in enumerate(messages):


            role = msg.get("role", "")


            content = msg.get("content", "")


            # Always keep system messages and recent messages intact


            if role == "system" or i >= recent_cutoff:


                compressed.append(msg)


                continue


            # Compress older tool results


            if role == "tool":


                content_str = str(content)


                if len(content_str) > 300:


                    # v28: Try to truncate at a JSON boundary or newline


                    trunc_point = 200


                    # Look for a good truncation boundary


                    for boundary in ["\n", ",}", "],", "}\n", ". "]:


                        idx = content_str[:250].rfind(boundary)


                        if idx > 100:


                            trunc_point = idx + len(boundary)


                            break


                    compressed_msg = {**msg, "content": content_str[:trunc_point] + f"\n...[compressed from {len(content_str)} chars]"}


                    compressed.append(compressed_msg)


                else:


                    compressed.append(msg)


            # Compress older assistant messages (but preserve tool_calls)


            elif role == "assistant":


                if msg.get("tool_calls"):


                    compressed.append(msg)  # Keep messages with tool_calls


                elif len(str(content)) > 400:


                    trunc_point = 300


                    for boundary in ["\n", ". ", "! ", "? "]:


                        idx = str(content)[:350].rfind(boundary)


                        if idx > 100:


                            trunc_point = idx + len(boundary)


                            break


                    compressed_msg = {**msg, "content": str(content)[:trunc_point] + "\n...[compressed]"}


                    compressed.append(compressed_msg)


                else:


                    compressed.append(msg)


            # v28: Compress very long user messages too


            elif role == "user" and len(str(content)) > 1000:


                content_str = str(content)


                compressed_msg = {**msg, "content": content_str[:500] + f"\n...[compressed from {len(content_str)} chars]"}


                compressed.append(compressed_msg)


            else:


                compressed.append(msg)


        return compressed


    # v22: Added images parameter for vision support.


    async def respond(self, user_input: str, images: list[str] | None = None) -> str:


        """Respond to a user message with full agent reasoning.


        v22: Added images parameter for vision support.


        v28: Proactive cross-session memory loading on first call.


        """


        # Reset stop/interrupt flag for each new user message
        self._stop_requested = False

        # Bug 3 fix: Reset per-task tool call counter on every new user message


        # Prevents counter accumulation across unrelated tasks


        #  -  -  Sub-agent fast path: skip all memory/context loading  -  - 


        if self._is_sub_agent:


            # Lightweight path: just store user message and respond


            self._pending_images = images or []


            await db.add_message(self.session_id, "user", user_input)


            await emit("user_message", content=user_input, session_id=self.session_id)


            try:


                answer = await self._nexus_reason(user_input)


            except Exception as e:


                print(


                    f"[reasoning] {type(e).__name__} in sub_agent session={self.session_id}: {e}",


                    file=sys.stderr, flush=True,


                )


                traceback.print_exc()


                _err_msg = _sanitize_error_msg(str(e), max_len=150)
                answer = f"[Reasoning error: {type(e).__name__}: {_err_msg}]"


            # Safety net: strip residual tool call markup


            answer = _strip_tool_call_markup(answer)


            await db.add_message(self.session_id, "assistant", answer)


            await emit("agent_message", content=answer, session_id=self.session_id)


            # Intent-reversal signal: if the agent just agreed to stop/cancel
            # (e.g. "No pushes", "Got it", "Understood", "Cancelled", "Stopped"),
            # set _stop_requested to kill any stale tool calls from previous loop.
            _agree_patterns = ("no pushes", "no upload", "no, ", "no! ", "won't ", "understood", "got it", "noted",
                               "cancelled", "stopped", "all stopped", "no pushes",
                               "no upload", "no active ")
            if any(p in answer.lower() for p in _agree_patterns):
                if self._session_message_count > 0:
                    self._stop_requested = True

            return answer


        await db.touch_session(self.session_id)


        self._session_message_count += 1


        # Proactively load cross-session memories on first interaction


        # This ensures the agent ALWAYS knows about the user from previous sessions


        await self._load_cross_session_memories()


        # v39: Load project context if user is working on a project


        await self._load_project_context(user_input)


        # v29: Invalidate compressed prompt cache after loading cross-session memories


        # so the prompt gets rebuilt with the freshly loaded memory data.


        if self._cached_user_profile or self._cached_cross_session_facts:


            _prompt_cache_compressed_caches.pop(self.session_id, None)


            _prompt_caches.pop(self.session_id, None)


        # v42: Initialize checkpoint manager for long-running task tracking


        if config.get("task_checkpoint_enabled", True) and not self._checkpoint_mgr:


            cm = get_checkpoint_manager(


                self.session_id,


                auto_save_interval=config.get("task_auto_save_interval", 30),


            )


            await cm.init(request=user_input, task_type="general")


            self._checkpoint_mgr = cm


        # v22: Store images and build content with vision if images provided


        self._pending_images = images or []


        # Build content with images if provided


        if self._pending_images:


            from nexus.core.vision_local import analyze_image_base64 as _local_viz


            descriptions = []


            for img in self._pending_images:


                try:


                    desc = await _local_viz(img, prompt="Describe this image in detail")


                    descriptions.append(desc)


                except Exception as _viz_err:


                    descriptions.append(f"[Image analysis failed: {_viz_err}]")


            image_text = "\n\n[Attached image analysis: " + "; ".join(descriptions) + "]"


            user_content = user_input + image_text


        else:


            user_content = user_input


        await db.add_message(self.session_id, "user", user_input)


        await emit("user_message", content=user_input, session_id=self.session_id)


        # Fast heuristic capture of personal facts (name, prefs) BEFORE replying.


        # Ensures the agent never forgets your name even if LLM consolidation fails.


        if not self._is_sub_agent:


            await self._capture_user_facts(user_input)


        try:


            # Stop/Cancel command detection
            _do_reason = True
            _stop_low = user_input.strip().lower()
            _exact_stop = {'stop', 'cancel', 'halt', 'abort'}
            _stop_phrases = ['stop that', 'cancel that', 'stop this', 'stop doing',
                             "don't upload", 'dont upload', "don't push", 'dont push',
                             "don't do that", 'dont do that', "don't do this", 'dont do this',
                             'please stop', 'please cancel', 'please halt',
                             'stop now', 'cancel now', 'halt now',
                             'stop the process', 'stop the execution', 'stop background',
                             'stop all tasks', 'cancel all tasks', 'stop everything',
                             'kill process', 'kill task', 'abort process', 'abort task',
                             'can you stop', 'could you stop', 'can you cancel', 'could you cancel',
                             'would you stop', 'i want to stop', 'i want to cancel']
            if (_stop_low in _exact_stop
                or any(_stop_low.startswith(w + ' ') for w in _exact_stop)
                or any(_stop_low.startswith(w) for w in _stop_phrases)
                or any(_stop_low.endswith(w) for w in _stop_phrases)
                or any(w in _stop_low for w in _stop_phrases)):
                from ..tasks.task_registry import list_tasks as _lt, cancel_task as _ct
                _running = _lt()
                _cancelled = 0
                for _t in _running:
                    if _ct(_t['task_id']):
                        _cancelled += 1
                self._stop_requested = True
                if _cancelled > 0:
                    await db.add_message(self.session_id, 'assistant', f'Cancelled {_cancelled} background task(s). All stopped.')
                    await emit('agent_message', content=f'Cancelled {_cancelled} background task(s). All stopped.', session_id=self.session_id)
                    return f'Cancelled {_cancelled} background task(s). All stopped.'
                else:
                    await db.add_message(self.session_id, 'assistant', 'No active background tasks to cancel.')
                    await emit('agent_message', content='No active background tasks to cancel.', session_id=self.session_id)
                    return 'No active background tasks to cancel.'


            answer = await self._nexus_reason(user_input)


        except Exception as e:


            print(


                f"[reasoning] {type(e).__name__} in session={self.session_id}: {e}",


                file=sys.stderr, flush=True,


            )


            traceback.print_exc()


            _err_msg = _sanitize_error_msg(str(e), max_len=150)
            answer = f"[Reasoning error: {type(e).__name__}: {_err_msg}]"


            await emit("error", source="reasoning",


                       error=_sanitize_error_msg(str(e), max_len=150),


                       session_id=self.session_id)


        # Strip any residual DSML/tool call markup from the answer before storing


        answer = _strip_tool_call_markup(answer)


        await db.add_message(self.session_id, "assistant", answer)


        await emit("agent_message", content=answer, session_id=self.session_id)


        # Background tasks (memory consolidation + reflection) run AFTER response


        # is sent. We detach them so they don't delay the user response.


        # FIX: Fetch messages AFTER saving assistant response so consolidation


        # includes the full exchange.


        if not self._is_sub_agent:


            recent = await db.get_messages(self.session_id, limit=8)


            if not self._stop_requested:
                _bg_task = asyncio.create_task(self._background_tasks(user_input, answer, recent))
                _register_bg_task(_bg_task, 'Post-response background processing', 'system',
                                  'Memory consolidation & reflection after each response')


        return answer


    async def _background_tasks(self, user_input: str, answer: str, recent: list):


        """Run post-response tasks in the background - don't block user."""


        try:


            # FIX: Lower threshold so short facts like "my name is John" get consolidated.


            # Only skip for truly trivial one-word acknowledgements.


            if len(user_input) > 5 or len(answer) > 20:


                await self.memory.consolidate(recent)


            if config.get("enable_self_reflection", True) and len(answer) > 50:
                await self._reflect(user_input, answer)
                logger.info("[reflect] Triggered post-response reflection")


            # v42: Quality assessment for self-supervised learning


            if config.get("enable_quality_assessment", True) and len(answer) > 50:


                self._quality_check_counter += 1


                interval = config.get("quality_assessment_interval", 10)


                if self._quality_check_counter % interval == 0:


                    assessment = await assess_response_quality(user_input, answer)


                    await process_quality_assessment(assessment)


                    await record_outcome(


                        tool_name="response",


                        args={"user_input": user_input[:200]},


                        success=assessment.score >= 5.0,


                        duration_ms=0,


                        output_preview=_strip_dsml_from_text(answer)[:200],


                    )


            # v42: Mark checkpoint progress if enabled


            if self._checkpoint_mgr and config.get("task_checkpoint_enabled", True):


                await self._checkpoint_mgr.update_progress(


                    current_step="Completed response generation"


                )


        except Exception as e:


            await emit("warn", source="background", error=_sanitize_error_msg(str(e), max_len=120))


    #  -  -  v15.2: Tool selection intelligence  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    async def _prefilter_tools(self, user_input: str, adhd_boost: dict[str, float] | None = None) -> list[dict] | None:


        """Pre-filter tool schemas based on user input keywords.


        Returns a filtered list of schemas, or None if no filtering is applied


        (meaning all tools should be sent). This reduces context size significantly.


        v20: Added 30-second TTL cache keyed by first 200 chars of lowercased input.


        """


        all_schemas = await REGISTRY.schemas()


        if all_schemas is None:


            return None


        low = user_input.lower()


        # v20: Check cache first


        # FIX: Also invalidate cache if the number of registered tools has changed


        # (tools may have been registered/unregistered since the cache was populated)


        global _prefilter_cache_tool_count


        current_tool_count = len(all_schemas)


        cache_key = low[:200]


        now = _time.monotonic()


        cached_entry = _prefilter_cache.get(cache_key)


        cache_valid = (


            cached_entry


            and (now - cached_entry[1]) < _PREFILTER_CACHE_TTL


            and _prefilter_cache_tool_count == current_tool_count


        )


        if cache_valid:


            return cached_entry[0]


        # v28: Short messages still get keyword scoring, not ALL tools.


        # Previously, messages like "read file" (2 words) bypassed filtering entirely,


        # sending 45+ tool schemas to the LLM  - wasting tokens and confusing the model.


        # Now we score even short messages, but with a lower threshold so they keep


        # more tools than a long, specific message would.


        min_tools_to_keep = 15 if len(low.split()) <= 3 else 12


        # Score each tool by keyword relevance


        # FIX: Schemas are in OpenAI nested format: {"type": "function", "function": {"name": ..., "description": ...}}


        scored_tools: list[tuple[dict, float]] = []


        for schema in all_schemas:


            func_info = schema.get("function", {})


            tool_name = func_info.get("name", "").lower()


            tool_desc = func_info.get("description", "").lower()


            # Exact name match in keyword map


            keywords = _TOOL_KEYWORD_MAP.get(tool_name, [])


            max_score = 0.0


            for kw in keywords:


                if kw in low:


                    # Higher score for more specific/rare keywords


                    # Minimum keyword length of 3 to avoid short false matches


                    specificity = max(len(kw) / 10.0, 0.3) if len(kw) >= 3 else 0.1


                    max_score = max(max_score, specificity)


            # Also check tool description keywords in user input


            for word in tool_desc.split():


                word = word.strip(".,;:!?()[]{}\"'")


                if len(word) >= 4 and word in low:


                    max_score = max(max_score, 0.1)


            # Also check if tool name appears directly in user input


            if tool_name and tool_name in low:


                # Exact tool name match is a very strong signal


                max_score = max(max_score, 12.0)


            # Check name fragments (e.g., "file_read"  - "read" in input scores for file_read)


            if tool_name:


                name_parts = tool_name.split("_")


                for part in name_parts:


                    if len(part) >= 3 and part in low:


                        max_score = max(max_score, 8.0)


            scored_tools.append((schema, max_score))


        # Sort by relevance score descending


        scored_tools.sort(key=lambda x: x[1], reverse=True)


        # Keep all tools with score > 0


        relevant_tools = [


            schema for schema, score in scored_tools


            if score > 0


        ]


        #  -  -  FIX: Custom skill tools bypass the filter  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        # Custom skills are registered with category="custom" but their names


        # are dynamic and never appear in _TOOL_KEYWORD_MAP. They scored 0


        # and were silently excluded from every LLM call. Now we always


        # append them before the cap so they're always available.


        custom_tool_names = {


            t.name for t in REGISTRY.all_tools()


            if t.category == "custom"


        }


        if custom_tool_names:


            existing_names = {


                s.get("function", {}).get("name", "")


                for s in relevant_tools


            }


            # Add custom skill tools that aren't already in the relevant list


            custom_schemas = [


                s for s in all_schemas


                if s.get("function", {}).get("name", "") in custom_tool_names


                and s.get("function", {}).get("name", "") not in existing_names


            ]


            # Prepend custom tools so they have highest priority


            relevant_tools = custom_schemas + relevant_tools


        # If no tools matched, keep a sensible default set


        if not relevant_tools:


            # Keep the most commonly used tools


            popular_tools = [


                "web_search", "fetch_url", "deep_research",


                "file_read", "file_write", "file_list",


                "shell_run", "python_exec",


                "create_ppt", "ppt_add_slide",


                "git_status", "git_add", "git_commit",


                "remember", "recall_memories", "recall",


                "goal_list", "goal_create",


                "email_send", "schedule_task",


                "image_generate",


                "nexus3d_create_mesh", "nexus3d_render_scene",


                "vm_start", "vm_execute", "vm_vision_loop",


                "mcp_call_tool", "mcp_connect_server", "mcp_list_servers",


                "pt_model_train", "call_integration_api",


                "journal_add", "journal_search",


                "system_info", "settings_get",


                "delegate_task", "delegate_batch",


            ]


            relevant_tools = [


                schema for schema in all_schemas


                if schema.get("function", {}).get("name", "") in popular_tools


            ][:20]


            # Also add any custom skill tools to the default set


            if custom_tool_names:


                existing_names = {


                    s.get("function", {}).get("name", "")


                    for s in relevant_tools


                }


                custom_schemas = [


                    s for s in all_schemas


                    if s.get("function", {}).get("name", "") in custom_tool_names


                    and s.get("function", {}).get("name", "") not in existing_names


                ]


                relevant_tools = custom_schemas + relevant_tools


        # v18: Re-rank using strategic tool intelligence (learned performance data)


        try:


            ranked = await REGISTRY.strategic_tool_ranking(task_context=user_input)


            if ranked:


                ranked_names = {r["name"] for r in ranked}


                # Boost tools that strategic ranking recommends


                high_perf = [s for s in relevant_tools if s.get("function", {}).get("name", "") in ranked_names]


                low_perf = [s for s in relevant_tools if s.get("function", {}).get("name", "") not in ranked_names]


                # Sort high-perf by adjusted_score from strategic ranking


                rank_map = {r["name"]: r["adjusted_score"] for r in ranked}


                high_perf.sort(key=lambda s: rank_map.get(s.get("function", {}).get("name", ""), 0), reverse=True)


                # FIX: Also pull in tools from strategic ranking that weren't


                # in relevant_tools (scored 0 by keyword but historically useful).


                existing_names = {s.get("function", {}).get("name", "") for s in relevant_tools}


                for r in ranked:


                    if r["name"] not in existing_names and r.get("adjusted_score", 0) > 0.4:


                        # Find schema for this tool


                        for s in all_schemas:


                            if s.get("function", {}).get("name", "") == r["name"]:


                                high_perf.append(s)


                                existing_names.add(r["name"])


                                break


                relevant_tools = high_perf + low_perf


        except Exception:


            # FIX: Previously this silently swallowed ALL errors including real


            # failures. Now we only skip on DB errors, but still log the issue.


            import logging as _logging


            _logging.getLogger("nexus.engine").debug(


                "Strategic tool ranking failed, continuing with keyword scores",


                exc_info=True,


            )


        #  -  -  FIX: Custom tools are always placed at the front to guarantee they


        # survive the cap. Then we fill up to min_tools_to_keep slots with remaining tools.


        # v28: Reduced from 45 to min_tools_to_keep (8 for specific, 15 for short messages)


        # to reduce token waste on irrelevant tool schemas.


        custom_in_result = [s for s in relevant_tools if s.get("function", {}).get("name", "") in custom_tool_names]


        non_custom_in_result = [s for s in relevant_tools if s.get("function", {}).get("name", "") not in custom_tool_names]


        remaining_slots = max(0, min_tools_to_keep - len(custom_in_result))


        # Always include at least min_tools_to_keep tools, but cap at 25 max


        result = custom_in_result + non_custom_in_result[:max(remaining_slots, 25 - len(custom_in_result))]


        # v41: FORCE include delegate tools when user message suggests parallel/multi-file work


        # This ensures the agent can spawn sub-agents for tasks like "create multiple files"


        # Proactive delegation triggers -- broad keywords that suggest multi-step/multi-file work


        # v43: Expanded significantly for proactive sub-agent use (not just explicit parallel language)


        _parallel_keywords = [


            # Explicit parallel signals


            "multiple", "several", "parallel", "concurrent", "split", "divide work",


            "at the same time", "simultaneously", "in parallel", "do in background",


            # Multi-file creation signals


            "create files", "build files", "make files", "generate files",


            "write files", "create several", "create multiple",


            # General build/create verbs -- any multi-step creation task


            "create a", "build a", "make a", "generate a", "write a",


            "develop a", "implement a", "set up a", "scaffold a",


            "add a", "create new", "build new",


            # Component/module/feature indicators


            "component", "module", "feature", "page", "route", "endpoint",


            "scaffold", "template", "boilerplate",


            # Architecture patterns


            "frontend", "backend", "full stack", "full-stack",


            "api", "database", "schema", "model", "migration",


            # Multi-step analysis/research


            "research and", "analyze and", "investigate and",


            "compare", "summarize", "break down", "walk through",


            "step by step", "multi-step",


        ]


        _low_input = user_input.lower()


        if any(kw in _low_input for kw in _parallel_keywords):


            _delegate_names = {"delegate_task", "delegate_batch"}


            _existing_names = {s.get("function", {}).get("name", "") for s in result}


            for _name in _delegate_names:


                if _name not in _existing_names:


                    for _schema in all_schemas:


                        if _schema.get("function", {}).get("name", "") == _name:


                            result.append(_schema)


                            break


        # v20: Store in cache


        _prefilter_cache[cache_key] = (result, now)


        _prefilter_cache_tool_count = current_tool_count  # FIX: Track tool count for cache invalidation


        # Prune stale entries if cache grows large


        if len(_prefilter_cache) > 100:


            stale_keys = [k for k, (_, t) in _prefilter_cache.items() if (now - t) > _PREFILTER_CACHE_TTL]


            for k in stale_keys:


                del _prefilter_cache[k]


        return result


    #  -  -  v15.2: Error recovery with param modification  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    async def _execute_tool_with_recovery(


        self, tool_name: str, args: dict, call_id: str,


        messages: list[dict],


    ) -> tuple[str, bool, int]:


        """Execute a tool with error recovery.


        On failure, attempts to retry with modified parameters before


        escalating to the replanning layer.


        Returns (observation, success, duration_ms).


        """




        _t0 = _time.monotonic()


        # Track consecutive failures for this tool


        fail_key = tool_name  # Tool-level tracking (not args-specific)


        self._tool_retry_count.setdefault(fail_key, 0)

        # Unwrap "input"/"params"/"args"/"arguments"/"parameters" wrapper if present
        # e.g. file_write({"input": {"path": "file.txt", "content": "..."}})
        # instead of file_write({"path": "file.txt", "content": "..."})
        for _wrapper_key in ("input", "params", "args", "arguments", "parameters"):
            if _wrapper_key in args and isinstance(args[_wrapper_key], dict) and len(args) <= 2:
                args = dict(args[_wrapper_key])
                break


        # Primary execution


        result = await REGISTRY.execute(tool_name, args, self.session_id)


        success = result.success


        output = result.output

        # vFIX: Empty-result fail-fast — track consecutive empty tool results
        _empty_search_tools = {"web_search", "file_search", "file_list", "glob", "fetch_url"}
        if tool_name in _empty_search_tools:
            # Check negative cache first (skip execution if known empty)
            _neg_key = f"{tool_name}:{str(args.get('pattern', args.get('query', args.get('path', ''))))[:120]}"
            if _neg_key in self._negative_cache:
                output = f"[CACHED-EMPTY] Previous search for this pattern returned no results. Try a different query/path."
                success = False
            elif output is None or (isinstance(output, str) and len(output.strip()) < 15):
                self._empty_search_counter += 1
                self._negative_cache[_neg_key] = _time.time()
                # LRU eviction: keep cache bounded
                if len(self._negative_cache) > self._negative_cache_max:
                    _oldest = min(self._negative_cache, key=self._negative_cache.get)
                    del self._negative_cache[_oldest]
                if self._empty_search_counter >= 3:
                    output = f"[FAIL-FAST] {self._empty_search_counter} consecutive empty search results. Try a different approach or query."
                    self._empty_search_counter = 0
            else:
                self._empty_search_counter = max(0, self._empty_search_counter - 1)
        else:
            # Non-search tool with empty output — still track but less aggressively
            if output is None or (isinstance(output, str) and len(output.strip()) < 5):
                self._empty_search_counter += 1
            else:
                self._empty_search_counter = 0

        # v22: Track tool usage for self-awareness

        # vFIX: Failure pattern tracking — detect repeated tool+error combos
        _err_detail = (result.error_detail or "")[:80]
        _err_cat = (result.error_category.value if hasattr(result.error_category, "value") else str(result.error_category))
        if not success and _err_detail:
            _normalized = _err_detail.split(":")[0].strip() if ":" in _err_detail else _err_detail
            _pattern_key = f"{tool_name}:{_err_cat}:{_normalized[:60]}"
            self._tool_failure_patterns[_pattern_key] = self._tool_failure_patterns.get(_pattern_key, 0) + 1
            _pattern_count = self._tool_failure_patterns[_pattern_key]
            if _pattern_count >= 3 and _pattern_key not in self._failure_pattern_warning_injected:
                self._failure_pattern_warning_injected.add(_pattern_key)
                _warn_msg = (
                    f"[SYSTEM] Tool '{tool_name}' has failed {_pattern_count} times "
                    f"with the same error pattern ({_err_cat}: {_normalized[:60]}). "
                    f"Consider using an alternative approach or tool."
                )
                messages.append({"role": "system", "content": _warn_msg})
                logger.warning("Injected failure pattern warning: %s", _warn_msg[:120])

        _duration_ms_interim = round((_time.monotonic() - _t0) * 1000)


        tool_stats = self._session_tool_usage.setdefault(tool_name, {"calls": 0, "success": 0, "fail": 0, "total_ms": 0})

        # vADHD: Track tool usage for adaptive prefiltering
        _usage = self._tool_usage_tracker.setdefault(tool_name, {"calls": 0, "success": 0, "fail": 0, "total_ms": 0, "avg_ms": 0.0})
        _usage["calls"] += 1


        tool_stats["calls"] += 1


        if success:


            tool_stats["success"] += 1
            _usage["success"] += 1


        else:


            tool_stats["fail"] += 1
            _usage["fail"] += 1


        tool_stats["total_ms"] += _duration_ms_interim
        _usage["total_ms"] += _duration_ms_interim
        _usage["avg_ms"] = _usage["total_ms"] / max(_usage["calls"], 1)


        # v42: Record outcome for self-supervised learning


        if config.get("enable_self_supervised_learning", True):


            await record_outcome(


                tool_name=tool_name,


                args=args,


                success=success,


                duration_ms=_duration_ms_interim,


                output_preview=_strip_dsml_from_text(str(output))[:200] if output else "",


            )


        if success:


            self._tool_retry_count.pop(fail_key, None)


            _duration_ms = round((_time.monotonic() - _t0) * 1000)


            _preview = _strip_dsml_from_text(str(output))[:200] if output else ""


            await emit("tool_end", name=tool_name,


                       output_preview=_preview,


                       duration_ms=_duration_ms, success=success,


                       session_id=self.session_id)


            return output, True, _duration_ms


        # v15.2: Error recovery  - try to fix common issues


        consecutive_failures = self._tool_retry_count.get(fail_key, 0)


        if consecutive_failures < self._max_tool_retries:


            self._tool_retry_count[fail_key] = consecutive_failures + 1

            # Skip retry if past outcomes predict very low success probability
            _skip_retry = False
            try:
                from ..self_improve import predict_tool_success
                _prediction = await predict_tool_success(tool_name, args)
                if _prediction and _prediction.get("success_probability", 1.0) < 0.3:
                    _skip_retry = True
                    logger.info("Skipping retry for %s: predicted success %.0f%%", tool_name, _prediction.get("success_probability", 0) * 100)
            except Exception:
                pass

            if _skip_retry:
                modified_args = None
                modification_desc = ""
            else:
                modified_args, modification_desc = self._suggest_param_fix(tool_name, args, str(output))


            if modified_args:


                await emit("tool_retry", name=tool_name, attempt=consecutive_failures + 1,


                           modification=_strip_dsml_from_text(modification_desc),


                           session_id=self.session_id)


                # Retry with modified params


                result2 = await REGISTRY.execute(tool_name, modified_args, self.session_id)


                success2 = result2.success


                output2 = result2.output


                # v22: Track retry tool usage for self-awareness


                _retry_ms = round((_time.monotonic() - _t0) * 1000)


                tool_stats = self._session_tool_usage.setdefault(tool_name, {"calls": 0, "success": 0, "fail": 0, "total_ms": 0})


                tool_stats["calls"] += 1


                if success2:


                    tool_stats["success"] += 1


                else:


                    tool_stats["fail"] += 1


                tool_stats["total_ms"] += _retry_ms


                # v42: Record outcome for self-supervised learning (retry)


                if config.get("enable_self_supervised_learning", True):


                    await record_outcome(


                        tool_name=tool_name,


                        args=modified_args,


                        success=success2,


                        duration_ms=_retry_ms,


                        output_preview=_strip_dsml_from_text(str(output2))[:200] if output2 else "",


                    )


                _duration_ms = round((_time.monotonic() - _t0) * 1000)


                _preview = _strip_dsml_from_text(str(output2))[:200] if output2 else ""


                if success2:


                    self._tool_retry_count.pop(fail_key, None)


                    final_output = f"{output2}\n[Recovered after retry: {modification_desc}]"


                    await emit("tool_end", name=tool_name,


                               output_preview=_preview,


                               duration_ms=_duration_ms, success=True,


                               session_id=self.session_id)


                    return final_output, True, _duration_ms


                else:


                    await emit("tool_end", name=tool_name,


                               output_preview=_preview,


                               duration_ms=_duration_ms, success=False,


                               session_id=self.session_id)


                    # v22: Real-time failure learning  - learn immediately from tool failure


                    try:


                        from ..self_improve import learn_from_failure_realtime

                        _fix = await learn_from_failure_realtime(

                            tool_name, str(output2), modified_args, self.session_id,

                        )

                        _fix_msg = f"[Realtime Fix] {tool_name}: {_fix[:300]}" if _fix else None

                    except Exception:
                        _fix = None
                        _fix_msg = None
                        logger.warning("learn_from_failure_realtime failed during retry", exc_info=True)

                    # Inject realtime fix as a system message so the LLM sees it immediately
                    if _fix_msg:
                        messages.append({"role": "system", "content": _fix_msg})

                    # Invalidate prompt cache so future turns also pick up the learning
                    _invalidate_prompt_cache(self.session_id)

                    _fix_suffix = f" Try: {_fix}" if _fix else ""
                    return f"{output2}\n[Retry with '{modification_desc}' also failed{_fix_suffix}]", False, _duration_ms

        else:


            self._tool_retry_count.pop(fail_key, None)


        _duration_ms = round((_time.monotonic() - _t0) * 1000)


        _preview = _strip_dsml_from_text(str(output))[:200] if output else ""


        await emit("tool_end", name=tool_name,


                   output_preview=_preview,


                   duration_ms=_duration_ms, success=False,


                   session_id=self.session_id)


        # v22: Real-time failure learning  - learn immediately from tool failure


        try:


            from ..self_improve import learn_from_failure_realtime


            _fix = await learn_from_failure_realtime(

                tool_name, str(output), args, self.session_id,

            )
            # Inject realtime fix as a system message so the LLM sees it immediately
            if _fix:
                messages.append({
                    "role": "system",
                    "content": f"[Realtime Fix] {tool_name}: {_fix[:300]}",
                })

            # Invalidate prompt cache so future turns also pick up the learning
            _invalidate_prompt_cache(self.session_id)

        except Exception:

            logger.warning("learn_from_failure_realtime failed", exc_info=True)

        return output, False, _duration_ms


    @staticmethod


    def _suggest_param_fix(tool_name: str, args: dict, error_msg: str) -> tuple[dict | None, str]:


        """Suggest parameter modifications based on common error patterns.


        Returns (modified_args, description) or (None, "") if no fix suggested.


        """


        error_low = error_msg.lower()


        modified = dict(args)


        # Common fix 1: empty or None string arguments  - remove them, let the tool handle missing params


        for key in list(args.keys()):


            if isinstance(args[key], str) and not args[key].strip():


                modified.pop(key, None)


        # Common fix 2: "file not found"  - try without path prefix


        if "not found" in error_low or "no such file" in error_low:


            for key in ("path", "file", "filename", "filepath"):


                if key in args and isinstance(args[key], str):


                    # Try just the basename


                    import os


                    basename = os.path.basename(args[key])


                    if basename != args[key]:


                        modified[key] = basename


                        return modified, f"simplified '{key}' to basename '{basename}'"


        # Common fix 3: timeout  - reduce scope


        if "timeout" in error_low or "timed out" in error_low:


            for key in ("max_results", "limit", "count"):


                if key in args and isinstance(args[key], (int, float)) and args[key] > 5:


                    modified[key] = 5


                    return modified, f"reduced '{key}' from {args[key]} to 5"


        # Common fix 4: invalid URL  - add protocol


        for key in ("url", "website", "link"):


            if key in args and isinstance(args[key], str):


                url = args[key].strip()


                if url and not url.startswith(("http://", "https://")):


                    modified[key] = f"https://{url}"


                    return modified, f"added https:// prefix to '{key}'"


        # Common fix 5: slides error  - provide default slide


        if "slides" in error_low or "slide" in error_low:


            if "slides" in args:


                if isinstance(args.get("slides"), list) and len(args["slides"]) == 0:


                    modified["slides"] = [{"type": "title", "title": args.get("title", "Presentation")}]


                    return modified, "added default title slide to empty slides array"


        # Common fix 6: "input" wrapper pattern -- model wrapped all params in a wrapper key
        # e.g. file_write({"input": {"path": "file.txt", "content": "..."}})
        # instead of file_write({"path": "file.txt", "content": "..."})
        for _wrapper_key in ("input", "params", "args", "arguments", "parameters"):
            if _wrapper_key in args and isinstance(args[_wrapper_key], dict) and len(args) <= 2:
                return dict(args[_wrapper_key]), f"unwrapped '{_wrapper_key}' wrapper to top-level parameters"

        # Common fix 7: map non-standard param names to canonical keys for common tools
        _ALIAS_MAP: dict[str, dict[str, tuple[str, ...]]] = {
            "file_write": {
                "path":    ("file", "filepath", "filename", "file_path", "dest", "destination", "output", "out", "target"),
                "content": ("data", "text", "body", "payload", "html", "code", "markdown", "json", "value", "contents", "source", "src", "input"),
            },
            "python_exec": {
                "code":            ("script", "python_code", "python", "source", "program", "payload", "body", "input", "command"),
                "timeout_seconds": ("timeout", "timeout_s", "max_time"),
                "capture_files":   ("capture", "track_files"),
            },
            "shell_run": {
                "command":        ("cmd", "shell_command", "bash", "shell_cmd", "exec", "run", "line"),
                "session":        ("session_name", "name", "shell"),
                "timeout_seconds": ("timeout", "timeout_s", "max_time"),
            },
            "file_list": {
                "path": ("dir", "directory", "folder", "target"),
            },
            "file_read": {
                "path": ("file", "filepath", "filename", "file_path"),
            },
        }
        _tool_aliases = _ALIAS_MAP.get(tool_name, {})
        if _tool_aliases:
            _changes = []
            for _canonical, _alias_keys in _tool_aliases.items():
                if _canonical not in modified:
                    _found = next((k for k in _alias_keys if k in modified), None)
                    if _found:
                        modified[_canonical] = modified[_found]
                        del modified[_found]
                        _changes.append(f"mapped '{_found}' to '{_canonical}'")
            if _changes:
                return modified, "; ".join(_changes)

        return None, ""


    # ---------------- Text-based tool call parser ----------------


    @staticmethod


    def _parse_text_tool_calls(text: str, *, tools_stripped: bool = False) -> list[dict]:


        """Fallback parser for models that emit tool calls as text instead of


        using the function-calling API.


        Handles the following common formats produced by NVIDIA NIM / Llama /


        Hermes / Mistral models:


        1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>


        2. Bare JSON object that has "name" + "arguments"/"parameters" keys


        3. Action: tool_name\nAction Input: {...}  (text format)


        4. [tool_name]({"param": "value"})          (bracket format)


        """


        if not text:


            return []


        calls: list[dict] = []


        # Format 0: <tool_call>JSON</tool_call>  - direct XML format


        # Many LLMs emit tool calls wrapped in <tool_call> tags instead of


        # using the function-calling API. Parse this before other formats.


        for m in _TEXT_TOOL_DIRECT_CALL_RE.finditer(text):


            raw = m.group(1).strip()


            try:


                obj = json.loads(raw)


                name = obj.get("name") or obj.get("function")


                args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}


                if name and isinstance(name, str):


                    calls.append({"id": f"txt_{len(calls)}", "name": name,


                                  "arguments": args if isinstance(args, dict) else {}})


            except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:


                logger.debug("Failed to parse tool call JSON (Format 0): %s", e)


        if calls:


            return calls


        # Format 1: ☚JSON☛ -- Unicode bracket format (historical)


        for m in _TEXT_TOOL_FORMAT1_RE.finditer(text):


            raw = m.group(1).strip()


            try:


                obj = json.loads(raw)


                name = obj.get("name") or obj.get("function")


                args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}


                if name and isinstance(name, str):


                    calls.append({"id": f"txt_{len(calls)}", "name": name,


                                  "arguments": args if isinstance(args, dict) else {}})


            except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:


                logger.debug("Failed to parse tool call JSON (Format 1): %s", e)# Format 1b: <tool_call>tool_name<arg_key>value</arg_key>... (Nexus custom XML format)


        if not calls:


            for m in _TEXT_TOOL_FORMAT1B_RE.finditer(text):


                name = m.group(1).strip()


                args_str = m.group(2).strip()


                args = {}


                for arg in _TEXT_TOOL_FORMAT1B_ARG_RE.finditer(args_str):


                    args[arg.group(1)] = arg.group(2)


                if name:


                    calls.append({"id": f"txt_{len(calls)}", "name": name, "arguments": args})


        if calls:


            return calls


        # Format 2: bare JSON that looks like a single tool call


        # Only attempt when the entire content is one JSON object


        stripped = text.strip()


        if stripped.startswith("{") and stripped.endswith("}"):


            try:


                obj = json.loads(stripped)


                name = obj.get("name") or obj.get("function")


                args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}


                if name and isinstance(name, str) and isinstance(args, dict):


                    calls.append({"id": "txt_0", "name": name, "arguments": args})


                    return calls


            except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:


                logger.debug("Failed to parse tool call JSON (Format 2): %s", e)


        # Format 3: Action/Action Input (text format for non-tool-calling models)


        action_m = _TEXT_TOOL_FORMAT3_RE.search(text)

        if action_m:
            name = action_m.group(1).strip()
            raw_args = action_m.group(2).strip()
            try:
                decoder = json.JSONDecoder()
                args, _ = decoder.raw_decode(raw_args)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("Failed to parse Action Input as JSON: %s", e)
                args = {"input": raw_args}

            calls.append({"id": "txt_0", "name": name, "arguments": args})
            return calls


        # Format 4: [tool_name]({...}) bracket syntax


        bracket_m = _TEXT_TOOL_FORMAT4_RE.search(text)

        if bracket_m:
            name = bracket_m.group(1).strip()
            try:
                raw = bracket_m.group(2)
                # Find the opening { and use raw_decode for balanced brace matching
                brace_start = raw.index("{")
                decoder = json.JSONDecoder()
                args, _ = decoder.raw_decode(raw, brace_start)
            except (ValueError, json.JSONDecodeError) as e:
                logger.debug("Failed to parse bracket tool arguments as JSON: %s", e)
                args = {}

            calls.append({"id": "txt_0", "name": name, "arguments": args})


        if calls:


            return calls


        # Format 5: Prose-based  - v38: Now enabled for ALL models as a fallback


        # Previously, this ONLY ran when tools_stripped=True (non-tool-calling models).


        # But sometimes tool-calling models also fall back to prose (e.g., when the


        # model gets confused or the prompt is too complex). When that happens,


        # the prose tool description passes through as regular text and the agent


        # appears to "text tools instead of using them."


        # 


        # v38: Always try prose parsing as a LAST RESORT, but ONLY accept the


        # parsed result if the tool name actually exists in the registry.


        # This prevents garbage parsing (e.g., "I'll use the information"  - tool="the")


        # while still catching legitimate prose tool calls.


        if not calls:  # Only if no other format matched


            prose_patterns = [


                # "I'll use <tool_name>" or "I'll call <tool_name>"


                _TEXT_TOOL_PROSE_RE1.search(text),


                # "Using <tool_name>" at start of line


                _TEXT_TOOL_PROSE_RE2.search(text),


            ]


            for prose_m in prose_patterns:


                if not prose_m:


                    continue


                name = prose_m.group(1).strip()


                raw_desc = prose_m.group(2).strip() if prose_m.lastindex >= 2 else ""


                if not name:


                    continue


                # v38: Validate that the parsed name is actually a registered tool


                # This prevents false positives like "I'll use the data"  - tool="the"


                if not REGISTRY.get(name):


                    continue


                # Try to extract JSON arguments from the description


                args = {}


                json_m = _INLINE_JSON_RE.search(raw_desc)


                if json_m:


                    try:


                        args = json.loads(json_m.group())


                    except (json.JSONDecodeError, ValueError) as e:


                        logger.debug("Failed to parse prose inline JSON: %s", e)


                        # Try key=value or key="value" patterns


                        kv_pairs = _KV_PAIR_RE.findall(raw_desc)


                        for k, v in kv_pairs:


                            args[k.strip()] = v.strip()


                elif raw_desc:


                    # Try key=value without JSON


                    kv_pairs = _KV_PAIR_RE.findall(raw_desc)


                    if kv_pairs:


                        args = {k.strip(): v.strip() for k, v in kv_pairs}


                    else:


                        # Fallback: entire description as "input" or "query" param


                        # Clean up markdown code fences


                        clean_desc = _MARKDOWN_FENCE_STRIP_RE.sub("", raw_desc).strip()


                        if clean_desc:


                            args = {"input": clean_desc}


                if name and isinstance(name, str):


                    calls.append({"id": f"txt_{len(calls)}", "name": name, "arguments": args})


                    break


            if calls:


                return calls


        # Format 6: Code-block style tool calls


        # ```tool\nweb_search\n{"query": "latest news"}\n```


        # Also handles ```json\n{"name": "tool_name", "arguments": {...}}\n```


        # which is the format we instruct non-tool-calling models to use.


        for cb_m in _TEXT_TOOL_CODEBLOCK_RE.finditer(text):


            block = cb_m.group(1).strip()


            if not block.startswith("{"):


                # Try "name\n{json}" format


                lines = block.split("\n", 1)


                if len(lines) == 2:


                    maybe_name = lines[0].strip()


                    maybe_json = lines[1].strip()


                    if maybe_name.isidentifier() or maybe_name.replace("_", "").isalnum():


                        try:


                            args = json.loads(maybe_json)


                            if isinstance(args, dict):


                                calls.append({"id": f"txt_{len(calls)}", "name": maybe_name, "arguments": args})


                                continue


                        except (json.JSONDecodeError, ValueError) as e:


                            logger.debug("Failed to parse code-block tool args as JSON: %s", e)


            try:


                obj = json.loads(block)


                if isinstance(obj, dict):


                    name = obj.get("name") or obj.get("function") or obj.get("tool")


                    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}


                    if name and isinstance(name, str):


                        calls.append({"id": f"txt_{len(calls)}", "name": name,


                                      "arguments": args if isinstance(args, dict) else {}})


                elif isinstance(obj, list):


                    for item in obj:


                        if isinstance(item, dict):


                            name = item.get("name") or item.get("function") or item.get("tool")


                            args = item.get("arguments") or item.get("parameters") or item.get("args") or {}


                            if name and isinstance(name, str):


                                calls.append({"id": f"txt_{len(calls)}", "name": name,


                                              "arguments": args if isinstance(args, dict) else {}})


            except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:


                logger.debug("Failed to parse code-block tool call JSON: %s", e)


        if calls:


            return calls


        # Format 7: "function_call: tool_name" followed by JSON


        func_call_m = _TEXT_TOOL_FUNCCALL_RE.search(text)


        if func_call_m:
            name = func_call_m.group(1).strip()
            try:
                raw = func_call_m.group(2)
                brace_start = raw.index("{")
                decoder = json.JSONDecoder()
                args, _ = decoder.raw_decode(raw, brace_start)
            except (ValueError, json.JSONDecodeError) as e:
                logger.debug("Failed to parse function_call arguments as JSON: %s", e)
                args = {"input": raw.strip()}


            calls.append({"id": "txt_0", "name": name, "arguments": args})


            return calls


        # Format 8: <invoke name="tool_name"><parameter name="key">value</parameter></invoke>


        # Handles DSML/Codebuff-style XML tool call markup that some models


        # emit instead of using the native tool_calls API.


        if not calls:


            # Strip optional <tool_calls> wrapper


            _stripped = text.strip()


            _tc_match = re.match(


                r'<tool_calls>(.*)</tool_calls>', _stripped,


                re.DOTALL | re.IGNORECASE,


            )


            if _tc_match:


                _stripped = _tc_match.group(1).strip()


            for _invoke_m in _TEXT_TOOL_INVOKE_RE.finditer(_stripped):


                _name = _invoke_m.group(1).strip()


                _params_body = _invoke_m.group(2).strip()


                _args = {}


                for _param_m in _TEXT_TOOL_INVOKE_PARAM_RE.finditer(_params_body):


                    _key = _param_m.group(1).strip()


                    _value = _param_m.group(2).strip()


                    # Try JSON parsing for numbers/booleans/null


                    try:


                        _parsed = json.loads(_value)


                        _args[_key] = _parsed


                    except (json.JSONDecodeError, ValueError):


                        _args[_key] = _value


                if _name:


                    calls.append({"id": f"txt_{len(calls)}", "name": _name, "arguments": _args})


            if calls:


                return calls


        return calls


    # ---------------- Nexus Framework Reasoning ----------------


    async def _persist_tool_messages(self, messages: list[dict]) -> None:
        """Persist tool role messages not yet saved to DB."""
        persisted: set[str] = getattr(self, "_persisted_tool_ids", set())
        for m in messages:
            if m.get("role") == "tool":
                tid = m.get("tool_call_id", "") or str(id(m))
                if tid in persisted:
                    continue
                name = m.get("name", "")
                content = m.get("content", "")
                if content:
                    await db.add_message(
                        self.session_id, "tool", content,
                        tool_call_id=tid, name=name,
                    )
                    persisted.add(tid)
        self._persisted_tool_ids = persisted

    async def _nexus_reason(self, user_input: str) -> str:


        """Nexus Framework Reasoning  - unified adaptive reasoning engine.


        A single intelligent loop that adaptively applies the best reasoning


        strategy per-step based on task complexity:


        - Simple tasks: direct response (fast path, minimal tools)


        - Analysis tasks: step-by-step reasoning with tool access


        - Creative tasks: branch exploration with best-path selection


        - Complex multi-step tasks: planning, simulation, and backtracking


        - All tasks: built-in hallucination detection, loop detection, error recovery


        """


        await emit("thinking_phase", phase="nexus", session_id=self.session_id)


        # Fast path detection: short, conversational messages skip tool schemas +


        # semantic recall, BUT user-profile facts are ALWAYS injected so the


        # agent never forgets your name on a 10-char question like "who am i?".


        is_simple = self._is_simple_message(user_input)


        # Nexus Framework: assess complexity for adaptive strategy


        if not is_simple:


            try:


                complexity = _assess_complexity(user_input)


                await emit("nexus_complexity", analysis={


                    "complexity": complexity,


                    "strategy": "adaptive",


                    "engine": "nexus",


                }, session_id=self.session_id)


                # v43: For high-complexity tasks, ensure delegate tools are always available
                if complexity == "high":
                    _dn_v43 = {"delegate_task", "delegate_batch"}
                    # REGISTRY already imported at module level
                    _all_s = await REGISTRY.schemas() or []
                    _ex_v43 = {s.get("function", {}).get("name", "") for s in _all_s}
                    for _d in _dn_v43:
                        if _d not in _ex_v43:
                            logger.warning(
                                "Delegate tool '%s' not found in registry for high-complexity task",
                                _d,
                            )


            except Exception:


                logger.warning("Failed to resolve tool definition", exc_info=True)


        # v15.2: Use compressed prompt for simple messages


        if is_simple:


            system_prompt = await self._build_compressed_system_prompt()


        else:


            system_prompt = await self._build_system_prompt()


        # Always cheap  - pure SQL, no embedding.


        # Use cached profile (loaded with more entries) if available, otherwise fetch fresh


        # v15.2: Semantic recall  - limit to top 5 most relevant


        # FIX: Always do recall, even for simple messages, so the agent


        # never appears to "forget" the user. Just use fewer results for simple.


        if is_simple:


            all_memories = await self.memory.recall(user_input, k=3) if not self._is_sub_agent else None


            memories = all_memories[:3] if all_memories else []


        else:


            all_memories = await self.memory.recall(user_input) if not self._is_sub_agent else None


            memories = all_memories[:5] if all_memories else []


        short_term = await self.memory.short_term(limit=40) if not self._is_sub_agent else []


        # Build messages


        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]


        if memories:

            mem_text = "\n".join(f"- [{m.kind}] {m.content}" for m in memories)

            messages.append({

                "role": "system",

                "content": f"# Relevant long-term memories\n{mem_text}",

            })

        try:
            self._adhd_block = await maybe_fire_adhd(
                user_input,
                complexity if not is_simple else "low",
                getattr(self, '_adhd_module', None),
            )
            if self._adhd_block:
                messages.append({"role": "system", "content": self._adhd_block})
            # vADHD: Extract tool boost suggestions for prefiltering
            if self._adhd_module and self._adhd_module.last_context:
                self._adhd_tool_boost = self._adhd_module.last_context.get_tool_suggestions()
            else:
                self._adhd_tool_boost = {}
        except Exception:
            self._adhd_block = None
            self._adhd_tool_boost = {}

        # v39: Inject project context if available


        if self._cached_project_context:


            messages.append({


                "role": "system",


                "content": f"# Current Project Context\n{self._cached_project_context}\n\n"


                           f"Use this context to continue working on this project. "


                           f"Use project_write_context to update PROJECT.md when you make significant progress.",


            })


        # Include recent messages  - preserve tool_calls/tool_call_id/name so the


        # API sees a valid tool-use conversation rather than orphaned tool results.


        for m in short_term:


            role = m.get("role", "user")


            content = m.get("content") or ""


            # Meta may carry tool protocol fields serialised by add_message


            meta: dict = {}


            try:


                raw_meta = m.get("meta")


                if raw_meta:


                    meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta


            except (json.JSONDecodeError, ValueError) as e:


                logger.debug("Failed to parse message meta as JSON: %s", e)


            if role == "tool":


                out: dict[str, Any] = {"role": "tool", "content": content}


                if meta.get("tool_call_id"):


                    out["tool_call_id"] = meta["tool_call_id"]


                if meta.get("name"):


                    out["name"] = meta["name"]


                messages.append(out)


            elif role == "assistant":


                out = {"role": "assistant", "content": content}


                if meta.get("tool_calls"):


                    out["tool_calls"] = meta["tool_calls"]


                messages.append(out)


            elif role in ("user", "system"):


                messages.append({"role": role, "content": content})


        # v22: Add current user message with images if provided


        if self._pending_images:


            from nexus.core.vision_local import analyze_image_base64 as _local_viz


            descriptions = []


            for img in self._pending_images:


                try:


                    desc = await _local_viz(img, prompt="Describe this image in detail")


                    descriptions.append(desc)


                except Exception as _viz_err:


                    descriptions.append(f"[Image analysis failed: {_viz_err}]")


            image_text = "\n\n[Attached image analysis: " + "; ".join(descriptions) + "]"


            messages.append({"role": "user", "content": user_input + image_text})


        else:


            messages.append({"role": "user", "content": user_input})

            self._last_user_messages = [
                *getattr(self, "_last_user_messages", [])[-9:],
                user_input,
            ]


        # v15.2: Tool pre-filtering for non-simple messages


        if is_simple:


            tools = None


        else:


            tools = await self._prefilter_tools(user_input, adhd_boost=self._adhd_tool_boost or None)

        # v43: For high-complexity tasks, ensure delegate tools are always available
        if not is_simple and tools is not None:
            try:
                if _assess_complexity(user_input) == "high":
                    _dn_v43 = {"delegate_task", "delegate_batch"}
                    _existing_v43 = {t.get("function", {}).get("name", "") for t in tools}
                    for _d in _dn_v43:
                        if _d not in _existing_v43:
                            # Removed: REGISTRY imported at module level
                            _all_s43 = await REGISTRY.schemas() or []
                            for _s in _all_s43:
                                if _s.get("function", {}).get("name", "") == _d:
                                    tools.append(_s)
                                    break
            except Exception:
                logger.warning("Failed to inject delegate tools for high-complexity task", exc_info=False)

        # v31 -v34: Planning instruction only for genuinely complex tasks.


        # Previously injected for any non-simple message with >4 words, which


        # meant nearly every message got a 20-line planning prompt  - wasting


        # tokens and adding noise for trivial queries like "search for X".


        # Now we only inject when _assess_complexity returns "high", indicating


        # structural multi-step signals (connectors, numbered lists, long questions).


        _needs_planning = False


        if not is_simple:


            try:


                _needs_planning = (_assess_complexity(user_input) == "high")


            except Exception:


                _needs_planning = False


        if _needs_planning:


            # Count available tools for planning context


            _avail_tool_names = []


            if tools:


                _avail_tool_names = [


                    t.get("function", {}).get("name", "")


                    for t in tools[:15]


                    if t.get("function", {}).get("name")


                ]


            _tool_list_str = ", ".join(_avail_tool_names) if _avail_tool_names else "various tools"


            planning_instruction = (


                "IMPORTANT TASK EXECUTION PROTOCOL:\n"


                "1. ANALYZE: What exactly does this task require? What info do I need?\n"


                "2. PLAN: List the specific steps needed (consider: research  - gather info  - process  - output)\n"


                "3. EXECUTE: Call tools ONE AT A TIME, in order. After each tool result, evaluate progress.\n"


                "4. CONTINUE: Do NOT stop after 1-2 tool calls if the task requires more steps.\n"


                "5. COMPLETE: Only give your final answer when the task is genuinely done.\n\n"


                f"Available tools for this task: {_tool_list_str}\n\n"


                "EXECUTION RULES:\n"


                "- If a tool fails, try a different approach or tool\n"


                "- TOOL CALLING IS MANDATORY: You MUST call a tool via tool_calls API, not describe it\n"


                "- If you need info, use web_search or file_read FIRST before answering\n"


                "- If creating content, gather all needed info before writing\n"


                "- NEVER describe what you WOULD do - actually CALL the tool NOW\n"


                "- PROHIBITED: I will use, Let me search, I should call, I will look up, I need to check - these ARE hallucinations\n"


                "- PROHIBITED: Based on my search, The tool returned, After running - only legitimate if you actually called the tool\n"


                "- Write tool use JSON directly - no intro, no explanation, no future tense\n"


                "- Keep going until the task is fully complete or you have enough to answer\n"


                "- DELEGATE: For multi-step tasks, use delegate_batch to parallelize sub-tasks (FASTER!)\n"


            )


            messages.append({


                "role": "system",


                "content": planning_instruction,


            })


        # Simple messages get 1 iteration (direct answer), complex get more.


        # Scale iteration count by complexity (when adaptive_iterations is enabled)
        _use_adaptive = config.get("adaptive_iterations", True)
        if _use_adaptive and not is_simple and complexity == "low":
            max_iter = config.get("max_nexus_iterations", 9999)
        elif _use_adaptive and not is_simple and complexity == "medium":
            max_iter = config.get("max_nexus_iterations", 9999) // 2
        elif is_simple:
            max_iter = config.get("max_nexus_iterations", 9999)
        else:
            max_iter = config.get("max_nexus_iterations", 9999)


        _hallucination_retries = 0  # BUG FIX: prevent infinite hallucination retry loop


        _max_hallucination_retries = 9999  # effectively unlimited


        _stuck_count = 0  # v26: Track consecutive iterations with no tool calls


        _max_stuck = 9999  # effectively unlimited


        _stuck_recovery_count = 0  # v32: Track how many stuck recoveries we've attempted


        for step in range(max_iter):

            # v46: Mid-reasoning interrupt check
            if self._stop_requested:
                await emit("agent_message", content="[Task stopped by user command.]", session_id=self.session_id)
                return "[Task stopped by user command.]"

            await emit("nexus_step", step=step, session_id=self.session_id)


            # v26: Progressive context compression  - compress old messages when context grows large


            # v38: Only run compression every 3rd step to reduce overhead


            if step % 3 == 0:


                messages = self._compress_context(messages)


            # FIX: Only merge system messages when there are excessive ones (>8).


            # Previously merged at >5, which was too aggressive  - it caused recovery


            # instructions (loop/stuck detection messages) to be permanently merged


            # into the main prompt, confusing the LLM on subsequent iterations.


            # Now we keep recovery messages separate longer, and when we do merge,


            # we preserve ALL recovery messages as standalone system messages.


            # v38: Also preserve hallucination correction messages.


            _sys_contents = [m for m in messages if m["role"] == "system"]


            if len(_sys_contents) > 8:


                # Find ALL recovery/correction messages to preserve separately


                _recovery_keywords = ("You've called the same tool", "You've been reasoning", "SYSTEM CORRECTION", "same tool called too many times")


                _recovery_msgs = []


                _merged_parts = []


                for sc in _sys_contents:


                    if any(kw in sc["content"] for kw in _recovery_keywords):


                        _recovery_msgs.append(sc)  # v38: Preserve ALL recovery messages, not just last


                    else:


                        _merged_parts.append(sc["content"])


                


                # Merge non-recovery system messages


                _merged_sys = "\n\n".join(_merged_parts) if _merged_parts else ""


                messages = [


                    m for m in messages if m["role"] != "system"


                ]


                if _merged_sys:


                    messages.insert(0, {"role": "system", "content": _merged_sys})


                # v38: Re-append ALL recovery messages as standalone system messages


                # Old code only preserved the LAST one, which meant earlier corrections


                # were silently dropped and the LLM would repeat the same mistake.


                for rc in _recovery_msgs[-2:]:  # Keep last 2 to avoid bloat


                    messages.append({"role": "system", "content": rc["content"]})


            await emit("llm_call", step=step, session_id=self.session_id)


            await emit("token_usage_estimate", step=step, session_id=self.session_id)


            # Refresh ADHD context on every loop iteration: strip stale
            # block, then re-inject fresh so cross-domain analogies stay in
            # active context and actively steer the next LLM call.
            if getattr(self, '_adhd_module', None):
                # Remove stale ADHD block if present
                if getattr(self, '_adhd_block', None):
                    _adhd_content = self._adhd_block
                    messages[:] = [m for m in messages if not (m.get("role") == "system" and m.get("content") == _adhd_content)]
                # Re-fire ADHD with the latest tool results for fresh analogies
                try:
                    self._adhd_block = await maybe_fire_adhd(
                        user_input,
                        complexity if not is_simple else "low",
                        getattr(self, '_adhd_module', None),
                    )
                except Exception:
                    self._adhd_block = None
                if self._adhd_block:
                    messages.append({"role": "system", "content": self._adhd_block})
                # Update tool boost from fresh ADHD context
                if self._adhd_module and self._adhd_module.last_context:
                    self._adhd_tool_boost = self._adhd_module.last_context.get_tool_suggestions()

            # v15.2: LLM call with tool schemas

            # v23: Catch LLM API errors inside the loop instead of letting them

            # crash the entire reasoning pipeline. Retry once without tools as fallback.
            _validate_tool_call_pairs(messages)

            try:

                resp = await llm.complete(


                    messages, tools=tools, session_id=self.session_id,


                )


            except RuntimeError as llm_err:


                # LLM API failure (e.g. Together.xyz 500 after retries).


                # Don't crash the whole reasoning  - try once more WITHOUT tools


                # so the user at least gets a text response.


                err_msg = str(llm_err)


                print(


                    f"[reasoning] LLM call failed in step {step}: {err_msg}",


                    file=sys.stderr, flush=True,


                )


                await emit("warn", source="nexus_llm_error",


                           error=_sanitize_error_msg(f"LLM API error in step {step}: {err_msg}", max_len=150),


                           session_id=self.session_id)


                # If this was already a no-tools retry, give up


                if tools is None:


                    return f"I'm having trouble connecting to the AI service right now. Error: {err_msg[:150]}"


                # Retry once without tools to get at least a text answer

                _validate_tool_call_pairs(messages)

                try:

                    resp = await llm.complete(


                        messages, tools=None, session_id=self.session_id,


                    )


                    tools = None  # prevent further tool attempts this loop


                except RuntimeError:


                    return f"I'm having trouble connecting to the AI service right now. Error: {err_msg[:150]}"


            # v20: Thought content emission (debug prints gated behind config flag)


            if config.get("debug_nexus_loop", False):


                print(f"[THOUGHT] LLM response content: {repr(resp.content[:100] if resp.content else None)}")


                if resp.content:


                    print(f"[THOUGHT] Emitting thought: {resp.content[:100]}...")


                print(f"[THOUGHT] tools_stripped={getattr(resp, 'tools_stripped', False)}, tool_calls={len(resp.tool_calls) if resp.tool_calls else 0}", file=sys.stderr)


            if resp.content:


                _stripped_thought = _strip_tool_call_markup(resp.content)[:500]


                if _stripped_thought:


                    await emit("thought", content=_stripped_thought, session_id=self.session_id)


            # When tools_stripped is True, the model can't return native tool_calls,


            # so skip the native tool_calls check and go straight to text parsing.


            _skip_native_tool_calls = getattr(resp, 'tools_stripped', False)


            if resp.tool_calls and not _skip_native_tool_calls:


                # Append assistant message WITH tool_calls (must use OpenAI nested format


                # for the API: {"function": {"name": ..., "arguments": "<json-str>"}})


                openai_tool_calls = []


                for tc in resp.tool_calls:


                    if "function" in tc and isinstance(tc["function"], dict):


                        # Already in OpenAI format  - keep as-is


                        openai_tool_calls.append(tc)


                    else:


                        # Flat format from LLM module  - convert to OpenAI nested format


                        tc_args = tc.get("arguments", {})


                        args_str = json.dumps(tc_args) if isinstance(tc_args, dict) else str(tc_args)


                        openai_tool_calls.append({


                            "id": tc.get("id", ""),


                            "type": "function",


                            "function": {


                                "name": tc.get("name", ""),


                                "arguments": args_str,


                            },


                        })


                clean_content = _strip_tool_call_markup(resp.content or "")
                if not clean_content.strip() and openai_tool_calls:
                    first_tool = openai_tool_calls[0].get("function", {}).get("name", "tool")
                    clean_content = f"[Calling {first_tool}]"
                assistant_msg: dict[str, Any] = {


                    "role": "assistant",


                    "content": clean_content,


                    "tool_calls": openai_tool_calls,


                }


                messages.append(assistant_msg)


                # v20: Deduplicate tool calls  - if LLM returns multiple calls with


                # the same tool name and same/similar arguments, only execute the first


                seen_calls: set[str] = set()


                deduped_calls = []


                for call in resp.tool_calls:


                    if "function" in call and isinstance(call["function"], dict):


                        tc_name = call["function"].get("name", "")


                        raw = call["function"].get("arguments", "{}")


                        try:


                            tc_args = json.dumps(json.loads(raw), sort_keys=True)


                        except (json.JSONDecodeError, ValueError) as e:


                            logger.debug("Failed to parse tool call args for dedup: %s", e)


                            tc_args = raw


                    else:


                        tc_name = call.get("name", "")


                        tc_args = call.get("arguments", {})


                        if isinstance(tc_args, dict):


                            tc_args = json.dumps(tc_args, sort_keys=True)


                    dedup_key = f"{tc_name}:{tc_args}"


                    if dedup_key not in seen_calls:


                        seen_calls.add(dedup_key)


                        deduped_calls.append(call)


                    else:


                        await emit("tool_dedup", name=tc_name, session_id=self.session_id)

                # Rebuild openai_tool_calls from the deduped subset so that
                # every tool_call_id in the assistant message has a matching
                # tool result below.  Orphaned IDs trigger a 400 API error.
                openai_tool_calls = []
                for tc in deduped_calls:
                    if "function" in tc and isinstance(tc["function"], dict):
                        openai_tool_calls.append(tc)
                    else:
                        tc_args = tc.get("arguments", {})
                        args_str = json.dumps(tc_args) if isinstance(tc_args, dict) else str(tc_args)
                        openai_tool_calls.append({
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": tc.get("name", ""), "arguments": args_str},
                        })
                if openai_tool_calls:
                    messages[-1]["tool_calls"] = openai_tool_calls
                else:
                    messages[-1].pop("tool_calls", None)

                # Execute tool calls  - parallel execution for independent tools

                # v28: Run independent tool calls concurrently instead of sequentially.


                # Tools that modify shared state (shell_run, file_write) are serialized,


                # but read-only tools (web_search, file_read, python_exec) run in parallel.


                # Reset stuck counter only when we have real, non-looping tool calls


                if deduped_calls:


                    _stuck_count = 0


                # Phase 1: Parse all calls, validate, and separate into parallel/serial


                _parallel_tasks = []  # (call, tool_name, args)


                _serial_tasks = []    # (call, tool_name, args)


                _SERIAL_TOOLS = {"shell_run", "file_write", "file_edit", "shell_reset",


                                 "browser_navigate", "browser_click", "browser_fill_form"}


                for call in deduped_calls:


                    # LLM module returns flat format: {"id", "name", "arguments": dict}


                    # but we also handle raw OpenAI format: {"function": {"name", "arguments": str}}


                    if "function" in call and isinstance(call["function"], dict):


                        tool_name = call["function"].get("name", "")


                        raw_args = call["function"].get("arguments", "{}") or "{}"


                        try:


                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args


                        except (json.JSONDecodeError, ValueError) as e:


                            logger.debug("Failed to parse tool arguments as JSON: %s", e)


                            args = {}


                    else:


                        tool_name = call.get("name", "")


                        args = call.get("arguments", {})


                        if isinstance(args, str):


                            try:


                                args = json.loads(args)


                            except (json.JSONDecodeError, ValueError) as e:


                                logger.debug("Failed to parse tool arguments string as JSON: %s", e)


                                args = {}


                    # Validate that tool exists in registry


                    if not REGISTRY.get(tool_name):


                        warn_msg = f"Tool '{tool_name}' not found - skipping. Did you mean 'browser_screenshot'?"


                        await emit("warn", source="tool_call", error=_sanitize_error_msg(warn_msg, max_len=120))


                        messages.append({


                            "role": "tool",


                            "tool_call_id": call.get("id", ""),


                            "name": tool_name,


                            "content": f"Error: {warn_msg}",


                        })


                        continue


                    # v26: Loop detection  - if same tool+args called too many times, break out


                    if self._detect_loop(tool_name, args):


                        await emit("loop_detected", tool=tool_name, session_id=self.session_id)


                        # Remove the tool call from recent_tool_calls since it won't be executed


                        self._remove_last_tool_call(tool_name, args)


                        # FIX: Add a tool result to close the open tool_call_id (Bug 1)


                        messages.append({


                            "role": "tool",


                            "tool_call_id": call.get("id", ""),


                            "name": tool_name,


                            "content": "[Skipped: same tool called too many times. Try a different approach.]",


                        })


                        # Use system role for recovery instruction, NOT user


                        messages.append({


                            "role": "system",


                            "content": "You've called the same tool repeatedly with no progress. Try a different tool or approach.",


                        })


                        continue


                    await emit("tool_start", name=tool_name, params=_truncate_params_for_display(args), session_id=self.session_id)


                    # Route to parallel or serial execution


                    if tool_name in _SERIAL_TOOLS or len(deduped_calls) == 1:


                        _serial_tasks.append((call, tool_name, args))


                    else:


                        _parallel_tasks.append((call, tool_name, args))


                # Phase 2: Execute parallel tasks concurrently


                if _parallel_tasks:


                    async def _exec_parallel(call, tool_name, args):


                        observation, _, _ = await self._execute_tool_with_recovery(


                            tool_name, args, call.get("id", ""), messages,


                        )


                        return (call, tool_name, observation)


                    _par_results = await asyncio.gather(


                        *[_exec_parallel(c, tn, a) for c, tn, a in _parallel_tasks],


                        return_exceptions=True,


                    )


                    for i, result in enumerate(_par_results):


                        if isinstance(result, Exception):


                            # FIX: Must still add a tool result message for this tool_call_id,


                            # otherwise the OpenAI API contract is violated (assistant has


                            # tool_calls with no matching tool results  - 400 error).


                            logger.warning("Parallel tool execution failed: %s", result)


                            _failed_call = _parallel_tasks[i][0]


                            _failed_name = _parallel_tasks[i][1]


                            messages.append({


                                "role": "tool",


                                "tool_call_id": _failed_call.get("id", ""),


                                "name": _failed_name,


                                "content": f"[Error: parallel execution failed  - {str(result)[:200]}]",


                            })


                            continue


                        call, tool_name, observation = result


                        messages.append({


                            "role": "tool",


                            "tool_call_id": call.get("id", ""),


                            "name": tool_name,


                            "content": self._adaptive_truncate(_strip_dsml_from_text(observation)),


                        })


                # Phase 3: Execute serial tasks sequentially (state-modifying tools)


                for call, tool_name, args in _serial_tasks:


                    observation, _, _ = await self._execute_tool_with_recovery(


                        tool_name, args, call.get("id", ""), messages,


                    )


                    messages.append({


                        "role": "tool",


                        "tool_call_id": call.get("id", ""),


                        "name": tool_name,


                        "content": self._adaptive_truncate(_strip_dsml_from_text(observation)),


                    })


                continue


            #  -  -  Text-based tool call fallback  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


            # Some models (e.g. stepfun-ai on NVIDIA NIM) emit tool calls as


            # <tool_call>JSON</tool_call> inside resp.content instead of using


            # the function-calling API.  Parse and execute those before giving up.


            text_calls = self._parse_text_tool_calls(resp.content or "", tools_stripped=_skip_native_tool_calls)


            if text_calls:


                if config.get("debug_nexus_loop", False):


                    print(f"[TEXT_TOOL] Parsed {len(text_calls)} text-based tool call(s) from content", file=sys.stderr)


                # Build synthetic OpenAI tool_calls list so the history is valid


                openai_tool_calls = [


                    {


                        "id": tc["id"],


                        "type": "function",


                        "function": {


                            "name": tc["name"],


                            "arguments": json.dumps(tc["arguments"]),


                        },


                    }


                    for tc in text_calls


                ]


                # Strip ALL tool call markup from the assistant content


                # (<tool_call>, Action:/Input:, code blocks, bracket syntax)


                clean_content = _strip_tool_call_markup(resp.content or "")


                # If the response contained tool calls, replace content with a


                # placeholder.  Models often narrate the tool call parameters


                # (including raw Python code for python_exec) in the text content


                # alongside the tool call itself — that text leaks through to the


                # visible chat.  The tool call details are already visible in the


                # thinking block via the tool_start event, so the duplicate text


                # is redundant at best and a code leak at worst.


                if openai_tool_calls:


                    try:


                        _tool_name = openai_tool_calls[0].get('function', {}).get('name', 'unknown')


                    except (IndexError, AttributeError, TypeError):


                        _tool_name = 'unknown'


                    clean_content = f"[Executing tool: {_tool_name}]"


                elif not clean_content:


                    # If the response was ONLY tool call markup (text format), provide a placeholder


                    # so the conversation history doesn't show empty content


                    clean_content = "[Processing...]"


                messages.append({


                    "role": "assistant",


                    "content": clean_content,


                    "tool_calls": openai_tool_calls,


                })


                for call in text_calls:


                    tool_name = call["name"]


                    args      = call["arguments"]


                    if not REGISTRY.get(tool_name):


                        warn_msg = f"Tool '{tool_name}' not found in registry (parsed from text)."


                        await emit("warn", source="text_tool_call", error=_sanitize_error_msg(warn_msg, max_len=120))


                        messages.append({


                            "role": "tool",


                            "tool_call_id": call["id"],


                            "name": tool_name,


                            "content": f"Error: {warn_msg}",


                        })


                        continue


                    # Loop detection for text-based tool calls


                    if self._detect_loop(tool_name, args):


                        await emit("loop_detected", tool=tool_name, session_id=self.session_id)


                        self._remove_last_tool_call(tool_name, args)


                        messages.append({


                            "role": "tool",


                            "tool_call_id": call["id"],


                            "name": tool_name,


                            "content": "[Skipped: same tool called too many times. Try a different approach.]",


                        })


                        continue


                    # Check for duplicate text-based tool calls  - compare against recent tool results


                    _call_sig = f"{tool_name}:{json.dumps(args, sort_keys=True)[:200]}"


                    _is_dup = False


                    for m in reversed(messages[-6:]):  # Check last 6 messages


                        if m.get("role") == "tool" and m.get("name") == tool_name:


                            # Same tool was called recently  - check if args are similar


                            prev_content = m.get("content", "")[:300]


                            if _call_sig[:100] in prev_content or "Duplicate call" in prev_content:


                                _is_dup = True


                                break


                    if _is_dup:


                        await emit("tool_dedup", name=tool_name, session_id=self.session_id)


                        messages.append({


                            "role": "tool",


                            "tool_call_id": call["id"],


                            "name": tool_name,


                            "content": f"[Duplicate call skipped  - same tool with same arguments already executed.]",


                        })


                        continue


                    await emit("tool_start", name=tool_name, params=_truncate_params_for_display(args), session_id=self.session_id)


                    observation, _, _ = await self._execute_tool_with_recovery(


                        tool_name, args, call["id"], messages,


                    )


                    messages.append({


                        "role": "tool",


                        "tool_call_id": call["id"],


                        "name": tool_name,


                        "content": self._adaptive_truncate(_strip_dsml_from_text(observation)),


                    })


                continue


            # No tool calls (native or text-based) => check for hallucination


            # BUG FIX: Some models describe tool usage in free-form prose instead


            # of actually calling tools via the API.  Detect this and force a retry


            # with an explicit instruction to use the tool_call interface.


            content_text = resp.content or ""


            # Detect "planning statements" — the model narrates what it will do


            # (e.g. "Now let me build the presentation...", "I'll use python-pptx...")


            # instead of actually doing it. These are NOT final answers.


            _is_planning = bool(re.search(


                r"(?i)(?:now\s+)?(?:let['`]?s?\s+(?:me\s+)?|i['`]ll\s+|i\s+will\s+|we['`]ll\s+|we\s+will\s+|let['`]?s\s+)(?:create|build|make|write|generate|construct|develop|proceed|start|begin|implement|code|program|produce|formulate|prepare|organize|design|outline|draft|set\s+up|put\s+together|work\s+on|move\s+to|try|attempt|get\s+started|take\s+a\s+look|think\s+about|figure\s+out|go\s+ahead)",


                content_text,


            ))


            if not _is_planning:


                _is_planning = bool(re.match(


                    r"(?i)(?:now\s+)?(?:let['`]?s\s+(?:create|build|make|write|generate)|i['`]m\s+(?:going\s+to\s+)?(?:create|build|make|write|generate|use|call|run|execute|start)|we['`]re\s+(?:going\s+to\s+)?(?:create|build|make|write|generate))",


                    content_text.strip(),


                ))


            if _is_planning:


                _stuck_count = _max_stuck  # Immediately trigger stuck recovery


                messages.append({


                    "role": "system",


                    "content": "You are narrating your plan instead of doing it. Stop planning and USE A TOOL to execute the work. Do NOT describe what you will do — just call the appropriate tool."


                })


                continue


            # v26: Stuck detection  - track consecutive iterations with no tool calls


            # Only count as "stuck" if the response is short/empty  - a long,


            # thoughtful response means the agent is making progress (just not


            # calling tools this iteration)  - UNLESS the content is hallucinated


            # tool-use prose (e.g. "I will use the write_file tool").


            _has_hallucination = (


                any(p.search(content_text or "") for p in _HALLUCINATION_PATTERNS)


                if content_text else False


            )


            if content_text and len(content_text.strip()) > 100 and not _has_hallucination and not _is_planning:


                _stuck_count = max(0, _stuck_count - 1)  # Decrement  - making progress


            elif any(m.get("role") == "tool" for m in messages[-4:]) and step > 0:


                # Agent likely processing tool output  - don't penalize


                _stuck_count = max(0, _stuck_count - 1)


            elif _has_hallucination:


                # Hallucinated tool-use prose = stuck, penalize faster


                _stuck_count += 2


            else:


                _stuck_count += 1


            # v38 CRITICAL FIX: The stuck recovery code was RUNNING EVERY ITERATION,


            # not just when stuck. The old code had no else/continue after the


            # stuck check, so the recovery message was injected on EVERY non-tool-call


            # iteration, flooding the context and confusing the model.


            # Now, stuck recovery only runs when _stuck_count >= _max_stuck.


            if _stuck_count >= _max_stuck:


                _stuck_recovery_count += 1


                await emit("agent_stuck", iterations=_stuck_count, session_id=self.session_id)


                # After 2 stuck recoveries, force a final answer


                if content_text and len(content_text.strip()) > 20:


                    if _stuck_recovery_count >= 2:


                        return _strip_tool_call_markup(content_text).strip()


                # v31: Enhanced stuck recovery  - provide specific guidance about available tools


                _available_tool_names = []


                if tools:


                    _available_tool_names = [


                        t.get("function", {}).get("name", "")


                        for t in tools[:10]


                        if t.get("function", {}).get("name")


                    ]


                _tool_hint = f" Available tools: {', '.join(_available_tool_names)}" if _available_tool_names else ""

                messages.append({
                    "role": "assistant",
                    "content": f"Consider switching strategy.{_tool_hint}",
                })


                if content_text:


                    messages.append({"role": "assistant", "content": _strip_tool_call_markup(content_text)})


                _stuck_count = 0


                continue


            # If content is empty after stripping, don't return "(no response)" —


            # continue the loop and nudge the model to produce a real answer.


            _stripped_answer = _strip_tool_call_markup(content_text).strip()


            if not _stripped_answer:


                # Check if tools were used — if so, ask for a summary


                _tools_were_used = any(m.get("role") == "tool" for m in messages)


                messages.append({


                    "role": "system",


                    "content": (


                        "Your previous response was empty. "


                        + ("The tools you called produced results. Summarize what was accomplished."


                           if _tools_were_used


                           else "Stop planning and USE A TOOL to accomplish the task.")


                    ),


                })


                _stuck_count += 1


                continue


            # Persist tool messages so they survive across user turns
            await self._persist_tool_messages(messages)

            return _stripped_answer


        # Persist tool messages before max-iterations fallback
        await self._persist_tool_messages(messages)

        return "[Max reasoning iterations reached. You can tell me to continue if needed.]"


    @staticmethod


    def _adaptive_truncate(observation: str) -> str:


        """v20: Adaptive output truncation for tool observations.


        - Short observations (<500 chars): passed through unchanged


        - Medium observations (500-10000 chars): truncated to 4000 chars with note


        - Long observations (>10000 chars): truncated to 3000 chars with note


        """


        if not observation:


            return observation


        length = len(observation)


        if length < 500:


            return observation


        if length < 10000:


            truncated = observation[:4000]


            return truncated + f"\n\n[truncated {length - 4000} chars]"


        truncated = observation[:3000]


        return truncated + f"\n\n[truncated {length - 3000} chars - output too long]"


    @staticmethod


    def _is_simple_message(text: str) -> bool:


        """Heuristic: pure greeting/acknowledgment where tool usage is unnecessary."""


        if not text or not text.strip():


            return True


        text_lower = text.strip().lower()


        if "?" in text_lower:


            return False


        QUESTION_WORDS = {


            "what", "why", "how", "when", "where", "who", "which", "can", "could",


            "would", "should", "is", "are", "do", "does", "did", "has", "have",


            "will", "tell", "explain", "describe", "show", "help", "find",


            "search", "look", "create", "make", "build", "write", "generate",


        }


        words = set(text_lower.split())


        if words & QUESTION_WORDS:


            return False


        GREETING_WORDS = {


            "hi", "hello", "hey", "thanks", "thank", "ok", "okay", "sure",


            "great", "good", "nice", "yes", "no", "bye", "goodbye", "cool",


        }


        return len(words - GREETING_WORDS) == 0


    async def _capture_user_facts(self, user_input: str) -> None:


        """Extract and store persistent personal facts from input."""


        if not config.get("enable_long_term_memory", True):


            return


        if len(user_input) > 800:


            return


        txt = user_input.lower().strip()


        if len(txt) < 10 and "?" not in txt:


            return


        fact_signals = [


            "my name is", "i am", "i'm", "my name's",


            "i like", "i love", "i hate", "i prefer",


            "i work", "i live", "i have", "i use",


            "call me", "you can call me",


        ]


        import re


        for signal in fact_signals:


            if signal in txt:


                _, _, rest = txt.partition(signal)


                value = rest.strip().rstrip(".,!?").strip()


                if value and len(value) < 100:


                    fact_key = signal.replace(" ", "_") + "_" + value[:30]


                    try:


                        await db.add_memory(


                            kind="user_fact",


                            content=f"User fact: {signal.strip()} {value}",


                            importance=0.8,


                        )


                    except Exception:


                        logger.warning("Failed to save reflection", exc_info=True)


                    break


    async def _reflect(self, user_input: str, answer: str) -> None:
        """Post-response reflection: what worked and what to fix."""


        if not config.get("enable_self_reflection", True):


            return


        try:


            reflection_prompt = (


                f"User said: {user_input[:500]}\n"
                f"Agent answered: {answer[:500]}\n\n"
                "Reflect on your response:\n"
                "1. What worked well in your answer?\n"
                "2. What could be improved? (accuracy, completeness, clarity)\n"
                "3. Did you use the right tools and approach?\n"
                "4. What will you do differently next time?\n"
                "Keep it concise but specific."
            )
            result = await llm.complete(reflection_prompt, max_tokens=200)


            if result and hasattr(result, "content") and result.content:


                reflection = result.content.strip()


                # Strip zero-width characters that LLMs sometimes inject
                reflection = reflection.replace("​", "").replace("‌", "")
                reflection = _strip_tool_call_markup(reflection)

                if reflection:


                    await db.add_reflection(self.session_id, reflection)


                    await emit("reflection", content=reflection, depth=2, session_id=self.session_id)


                    logger.info("[reflect] Stored reflection (%d chars)", len(reflection))


        except Exception as exc:


            logger.warning("[reflect] Failed: %s", exc)
    async def _build_self_awareness_context(self) -> str:


        """Build a live self-awareness snapshot of the agent environment."""




        now = _time.time()


        cache = getattr(self, "_self_awareness_cache", None)


        cache_ttl = getattr(self, "_SELF_AWARENESS_CACHE_TTL", 30)


        if cache and (now - cache[1]) < cache_ttl:


                    # Append live background task listing (not cached - always fresh)
            try:
                from ..tasks.task_registry import list_tasks as _lb
                _bg = _lb()
                if _bg:
                    _items = [f"\n  - {t.get('name', 'unknown')} ({t.get('task_id', '')[:8]}...)" for t in _bg[:10]]
                    cache_str = cache[0]
                    cache_str += "\n" + f"Background Tasks: {len(_bg)} running\n{''.join(_items)}"
                    cache = (cache_str, cache[1])
                else:
                    cache_str = cache[0]
                    cache_str += "\n" + "Background Tasks: none running"
                    cache = (cache_str, cache[1])
            except Exception:
                pass
            return cache[0]


        parts = []


        import platform


        parts.append(f"System: {platform.system()} {platform.release()}")


        parts.append(f"Python: {platform.python_version()}")


        try:


            _msg_count = self._session_message_count


        except AttributeError:


            _msg_count = 0


        try:


            total_calls = sum(v["calls"] for v in self._session_tool_usage.values())


            total_success = sum(v["success"] for v in self._session_tool_usage.values())


            total_fail = total_calls - total_success


        except AttributeError:


            total_calls = total_success = total_fail = 0


        parts.append(f"Messages: {_msg_count} | Tools: {total_calls} ({total_success} ok, {total_fail} fail)")

        # v46: Live background task awareness - not cached
        try:
            from ..tasks.task_registry import list_tasks as _list_bg
            _bg_tasks = _list_bg()
            if _bg_tasks:
                _bg_lines = ["\n  - ".join(["", f"{t.get('name', 'unknown')} ({t.get('task_id', '')[:8]}...)"]) for t in _bg_tasks[:10]]
                parts.append(f"Background Tasks: {len(_bg_tasks)} running\n{''.join(_bg_lines)}")
            else:
                parts.append("Background Tasks: none running")
        except Exception:
            parts.append("Background Tasks: unavailable")

        

        # v43: Tool failure pattern detection


        try:


            _high_fail_tools = []


            for _tname, _tstats in self._session_tool_usage.items():


                _calls = _tstats.get("calls", 0)


                if _calls >= 2:


                    _success = _tstats.get("success", 0)


                    _fail_rate = 1.0 - (_success / _calls) if _calls > 0 else 0.0


                    if _fail_rate > 0.6:


                        _high_fail_tools.append(f"{_tname} ({_fail_rate:.0%} fail rate, {_calls} calls)")


            if _high_fail_tools:


                parts.append(f"WARNING - High-failure tools: {'; '.join(_high_fail_tools)} - AVOID using these unless absolutely necessary")


        except Exception:


            logger.debug("Failed to build high-fail tools warning", exc_info=True)


        try:


            integrations = await _get_connected_integrations_summary()


            if integrations:


                parts.append(f"Connected: {integrations}")


        except Exception:


            logger.debug("Failed to build integrations summary", exc_info=True)


        try:


            goals = getattr(self, "_active_goals", [])


            if goals:


                parts.append(f"Goals: {len(goals)} active")


        except Exception:


            logger.debug("Failed to build goals section", exc_info=True)


        context = " | ".join(parts)


        self._self_awareness_cache = (context, now)


        return context


    async def _build_system_prompt(self, *, _force_refresh: bool = False) -> str:


        """Build the full system prompt. Cached for _PROMPT_CACHE_TTL seconds (#15)."""


        now = _time.time()


        last_user_msgs = getattr(self, "_last_user_messages", [])
        cache_key = (
            f"{self.session_id}:{len(last_user_msgs)}:"
            + "|".join(str(m)[:80] for m in last_user_msgs[-3:])
        )
        cached = _prompt_caches.get(cache_key)


        if not _force_refresh and cached and (now - cached[1]) < _PROMPT_CACHE_TTL:


            return cached[0]


        name           = config.get("agent_name", "Hyper Nexus")


        personality    = config.get("personality", "")


        traits         = ", ".join(config.get("traits", []) or [])


        style          = config.get("communication_style", "")


        reasoning_mode = "nexus"  # Nexus Framework is the only reasoning strategy


        heartbeat_on   = config.get("enable_heartbeat", True)


        heartbeat_interval = config.get("heartbeat_interval_seconds", 30)


        memory_on      = config.get("enable_long_term_memory", True)
        vision_model   = config.get("vision_model", "")


        # Tool overview grouped by category


        from ..tools import REGISTRY
        from ..tasks.task_registry import register_task as _register_bg_task


        tools_by_cat: dict[str, list[str]] = {}


        for t in REGISTRY.all():


            tools_by_cat.setdefault(t.category, []).append(t.name)


        tools_overview = "\n".join(


            f"  [{cat}]: {', '.join(names)}" for cat, names in tools_by_cat.items()


        )


        # Skills grouped by category (include custom skills)


        all_skills_for_prompt = list(SKILLS)


        try:


            from ..tools.builtin.custom_loader import _custom_skills_cache


            for cs in _custom_skills_cache:


                if cs.get("enabled"):


                    all_skills_for_prompt.append({


                        "name": cs["name"],


                        "desc": cs.get("description", ""),


                        "cat": cs.get("category", "custom"),


                    })


        except Exception:


            logger.debug("Failed to build skills scope", exc_info=True)


        skills_by_cat: dict[str, list[str]] = {}


        for s in all_skills_for_prompt:


            skills_by_cat.setdefault(s["cat"], []).append(f"{s['name']}: {s['desc']}")


        skills_overview = "\n".join(


            f"  [{cat}]:\n    " + "\n    ".join(descs)


            for cat, descs in skills_by_cat.items()


        )


        # Live state from DB (parallelised)


        try:


            from ..memory import db as _db


            watches, monitors, schedules, notif_count, improvements, integrations, github_int, email_int = await asyncio.gather(


                _db.list_file_watches(enabled_only=True),


                _db.list_web_monitors(enabled_only=True),


                _db.list_nl_schedules(enabled_only=True),


                _db.unread_notification_count(),


                _db.list_improvement_log(resolved=False, limit=5),


                _db.list_connected_integrations(),


                _db.get_integration("github"),


                _db.get_integration("email"),


            )


            watch_summary    = f"{len(watches)} path(s)" if watches else "none"


            monitor_summary  = f"{len(monitors)} URL(s)" if monitors else "none"


            schedule_summary = f"{len(schedules)} active" if schedules else "none"


            improve_note     = (


                f" ({len(improvements)} unresolved pattern(s)  - use improvement_log tool)"


                if improvements else ""


            )


            notif_note = f"{notif_count} unread" if notif_count else "none"


            # Credential status  - use already-fetched integration data


            _gh_intg = github_int


            github_ready = bool(_gh_intg and _gh_intg.get("connected")) or bool(config.get("github_token", ""))


            _em_intg = email_int


            email_ready = bool(_em_intg and _em_intg.get("connected")) or bool(config.get("smtp_host", "") and config.get("smtp_user", ""))


        except Exception:


            watch_summary = monitor_summary = schedule_summary = "unknown"


            improve_note = notif_note = ""


            github_ready = bool(config.get("github_token", ""))


            email_ready = bool(config.get("smtp_host", "") and config.get("smtp_user", ""))


        github_status = "CONFIGURED \u2713" if github_ready else "NOT SET  - connect in Integrations or set github_token in Settings"


        email_status  = "CONFIGURED \u2713" if email_ready  else "NOT SET  - connect in Integrations or set smtp_host/user/password in Settings"


        prompt = (


            f"You are {name}, a fully autonomous personal AI agent.\n\n"


            f"# Identity\n"


            f"Personality: {personality}\n"


            f"Traits: {traits}\n"


            f"Communication style: {style}\n\n"


            f"# YOUR CAPABILITIES - USE PROACTIVELY\n"


            f"You have access to MANY tools and skills. When a task matches a capability,\n"


            f"USE IT AUTOMATICALLY without asking for permission:\n\n"


            f"## File Operations (USE THESE for any file work)\n"


            f"- file_read(path) - Read any file content\n"


            f"- file_write(path, content) - Create/update files\n"


            f"- file_list(path) - List files in directory\n"


            f"- file_search(pattern, path) - Search files by pattern\n"


            f"- file_delete(path) - Delete files\n\n"


            f"## Code Execution (USE THESE for running code)\n"


            f"- python_exec(code) - Run Python code\n"


            f"- shell_run(command) - Run shell commands\n"


            f"- code_explain(code) - Explain code\n\n"


            f"## Web & Research\n"


            f"- web_search(query) - Search the web\n"


            f"- fetch_url(url) - Fetch web pages\n"


            f"- deep_research(topic, depth) - Deep research on a topic\n\n"


            f"## Automation & Tasks\n"


            f"- goal_create(title, description) - Create a goal/task\n"


            f"- watch_url(url) - Monitor a URL for changes\n"


            f"- watch_path(path) - Watch a file path for changes\n\n"


            f"## Memory & Knowledge\n"


            f"- remember(fact) - Remember important info\n"


            f"- recall(query) - Search your memory\n"


            f"- journal_write(entry) - Write to journal\n\n"


            f"## Sub-Agents (REQUIRED for parallel & multi-step work)\n"


            f"- delegate_task(task, context) - Spawn a sub-agent to do work in parallel\n"


            f"- delegate_batch(tasks) - Spawn MULTIPLE sub-agents in parallel (FASTER)\n"


            f"WHEN TO USE (MUST USE):\n"


            f"  - Creating 2+ files -> use delegate_batch\n"


            f"  - Multi-step tasks -> use delegate_task\n"


            f"  - Any work that can happen simultaneously -> delegate it!\n"


            f"  - RESEARCH sub-tasks (web search, analysis) -> delegate_task\n"


            f"PROACTIVE: If a task has multiple parts, proactively split it into sub-tasks\n"


            f"  and delegate them. Do NOT process them sequentially yourself.\n"


            f"EXAMPLE: delegate_batch with tasks (see docs for format)\n"


            f"DO NOT write files yourself - delegate them to sub-agents!\n"


            f"## Other Available\n"


            f"- git_clone, git_commit, git_push - Version control\n"


            f"- email_send, email_inbox - Email management\n"


            f"- browser_open, browser_click, browser_screenshot - Browser automation\n"


            f"- generate_image(prompt) - AI image generation\n"


            f"- vm_execute(command) - Virtual machine operations\n\n"


            f"REMEMBER: You have ALL these tools. DON'T ask 'can you do this?'\n"


            f"Just use the appropriate tool directly. Be proactive!\n\n"


            f"# Self-Awareness: Live Environment & Capabilities\n"


        )


        #  -  -  v22: Inject live self-awareness context  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        try:


            self_awareness = await self._build_self_awareness_context()


            prompt += f"\n{self_awareness}\n\n"


        except Exception:


            logger.debug("Failed to build self-awareness context", exc_info=True)


        prompt += (


            f"# Core architecture\n"


            f"- Reasoning: Nexus Framework adaptive loop\n"


            f"- Memory: {'ON' if memory_on else 'OFF'}  - 3-layer (working + short-term + vector long-term)\n"


            f"- Heartbeat: {'ON' if heartbeat_on else 'OFF'} every {heartbeat_interval}s\n"


            f"- Self-reflection: after every response\n"


            f"- Self-improvement: every 30 min (learns strategies from failures){improve_note}\n"


            f"- Environment awareness: tracks goals, quality, anomalies globally\n\n"


            f"# Background systems (always running)\n"


            f"- File watcher: {watch_summary}  - emits workspace_change on file create/modify/delete\n"


            f"- Web monitor: {monitor_summary}  - emits web_change_detected on content hash change\n"


            f"- NL scheduler: {schedule_summary}  - fires tasks on expressions like 'every day at 9am'\n"


            f"- Notifications: {notif_note}  - delivered via WS bell in the WebUI\n\n"


            f"# External services\n"


            f"- GitHub: {github_status}\n"


            f"  Use github_* tools to: list/read repos, create issues and PRs, read files, get commits.\n"


            f"- Email: {email_status}\n"


            f"  Use email_* tools to: send emails, read inbox, search by subject/sender, reply.\n"


            f"  IMAP server needed for read/search/reply  - set imap_host in Settings.\n\n"


            f"# Integration tool usage\n"


            f"When a user wants to use a connected service, use the call_integration_api tool.\n"


            f"Example: call_integration_api(service='posthog', action='events.capture', body={{'event':'button_click'}}).\n"


            f"IMPORTANT for Hostinger: Use adapter 'action' like 'vps.start', 'vps.stop', 'domains.list' - NOT raw 'path' URLs.\n"


            f"  - Correct: call_integration_api(service='hostinger', action='vps.start', body={{'vps_id':'1469628'}})\n"


            f"  - Wrong: call_integration_api(service='hostinger', path='/vps/1469628/start')\n"


            f"You can connect to any of {len(_ALL_KNOWN)} integrations  - check the Integrations panel in the WebUI.\n"


            f"# Triggers & Event Automation\n"


            f"You have a trigger system that can automatically fire actions based on events.\n"


            f"Active triggers are injected into your system prompt automatically.\n"


            f"Use mcp_auto_connect to automatically discover and set up MCP servers for non-technical users.\n\n"


            f"# Development workflow\n"


            f"- Git: use git_* tools scoped to named project folders inside data/workspace/.\n"


            f"  Workflow: git_init or git_clone \u2192 edit files \u2192 git_add \u2192 git_commit \u2192 git_push.\n"


            f"- Shell sessions: shell_run keeps working directory between calls per named session.\n"


            f"  Example: shell_run('npm install', session='myapp') then shell_run('npm run build', session='myapp')\n"


            f"  Use shell_sessions to list active sessions, shell_reset to reset one.\n"


            f"  Allowed commands: python, node, npm, npx, pip, git, pytest, cargo, go, ls, cat, grep, etc.\n"


            f"- Code execution: python_exec runs Python in workspace dir, captures stdout/stderr/new files.\n"


            f"- File ops: file_read/write/list/delete are STRICTLY scoped to data/workspace/.\n"
            f"  CRITICAL: Every file you create MUST live under data/workspace/<project>/...\n"
            f"  - Pick a clear project folder name (e.g. 'my_api', 'todo_app') and put every\n"
            f"    file for that task inside it: data/workspace/my_api/src/main.py, etc.\n"
            f"  - NEVER write to absolute paths outside data/workspace/ (the tool will reject it).\n"
            f"  - NEVER drop files at the workspace root — always nest them in a project folder.\n"
            f"  - If a tool rejects a path with 'outside workspace', re-issue it with a path\n"
            f"    relative to data/workspace/ (e.g. 'my_project/file.py' instead of 'C:/...').\n\n"


            f"# Research\n"


            f"- deep_research: for complex questions  - generates multiple queries, fetches sources in\n"


            f"  parallel, synthesises a structured report (Summary / Key Findings / Analysis / Sources).\n"


            f"  Use this instead of a single web_search when depth matters.\n"


            f"- web_search + fetch_url: for quick lookups.\n"


            f"- browser_open/screenshot/click/fill_form: USE THESE for viewing websites!\n"


            f"  IMPORTANT: Use browser_open instead of fetch_url for ANY website with JavaScript/rendered content.\n"


            f"  The browser shows in Browser Live panel automatically. Use for: viewing pages, filling forms, clicking buttons.\n"


            f"- vm_start/vm_stop/vm_execute: Virtual computer for running code, Docker, full desktop environment.\n"


            f"  Use vm_start to boot, vm_execute to run shell commands inside the VM.\n"
            f"  Use vm_vision_loop for AUTONOMOUS GUI tasks: the agent takes screenshots, analyzes them with a vision model,\n"
            f"  and autonomously clicks, types, and navigates to accomplish the task. This is the PRIMARY tool for any\n"
            f"  GUI automation, desktop interaction, or computer-use task. Always use vm_vision_loop for tasks like\n"
            f"  open an app, browse a website, fill a form, edit a document on the desktop, etc.\n"
            f"- IMPORTANT: Do NOT automatically read, analyze, or process uploaded files unless the\n"
            f"  user explicitly asks you to. Uploaded files are just attached for your reference until\n"
            f"  the user says something like \"read this file\" or \"analyze this image\" or \"what's in this\".\n"
            f"  Wait for the user to ask before using any tool on the file.\n"
            f"- You can create or write files to data/workspace/ using file_write or python_exec.\n"


            f"  Any new or modified workspace files are automatically surfaced as download chips\n"


            f"  below your reply  - the user can click them to download without any extra steps.\n"


            f"- When a task produces output files (reports, code, images, CSVs, etc.), always save\n"


            f"  them to data/workspace/ so they appear as download chips for the user.\n\n"


            f"# Behaviour rules\n"


            f"- CRITICAL: When you need to use a tool, you MUST use the tool_calls API.\n"


            f"  NEVER describe tool usage in plain text and pretend you executed it.\n"


            f"  NEVER fabricate or hallucinate tool results  - always actually call the tool.\n"


            f"  If the tool_calls API is available, use it. Only describe actions in text\n"


            f"  when no tools are needed (e.g. casual conversation).\n"


            f"- Chain tools naturally  - don't ask for permission before each step, just do it.\n"


            f"- NEVER ask 'which approach works best for you?'  - just pick ONE approach and execute it.\n"


            f"- For complex tasks: PLAN FIRST, then execute step by step. Don't stop after one tool call.\n"


            f"- For coding tasks: shell_run to run, read error output, patch files, re-run.\n"


            f"- For research tasks: prefer deep_research over single web_search for complex questions.\n"


            f"- For presentations/reports: gather information first (web_search/deep_research),\n"


            f"  then create the document with REAL content. Don't create empty/generic files!\n"


            f"- ALWAYS provide meaningful parameters to tools. For docx/xlsx/ppt/charts tools,\n"


            f"  include actual data/content, not just placeholders. The user expects usable output.\n"


            f"- When in doubt about content, ASK the user for details BEFORE creating files.\n"


            f"- For external services: check credential status above before attempting  - if not set,\n"


            f"  tell the user what to configure in Settings first.\n"


            f"- COMPLETE TASKS: Don't stop at 'let me know if you want me to continue' or give options.\n"


            f"  When you find a path works, keep going until the task is done. Finish what you start.\n"


            f"- Be honest about uncertainty. Never fabricate tool results.\n"


            f"- If asked what you can do, answer from this prompt  - it reflects your live state.\n"


            f"- CROSS-SESSION MEMORY (CRITICAL): You have persistent memory across ALL sessions.\n"


            f"  You will receive a 'Memories from previous sessions' section with things the user\n"


            f"  told you in past sessions (name, preferences, projects, etc.). You MUST use this\n"


            f"  information proactively  - NEVER say 'I don't know your name' or 'I don't remember'\n"


            f"  if the answer is in your cross-session memories. This data IS your long-term memory.\n"


            f"  When a user says 'remember this' or 'don't forget', use the remember tool to persist it.\n"


            f"- ENVIRONMENT AWARENESS: You are fully aware of your runtime environment, tools,\n"


            f"  connected integrations, MCP servers, active triggers, and scheduled tasks.\n"


            f"  Use this awareness to make informed decisions about which tools to use.\n"


            f"  If you're unsure about your capabilities, check the Self-Awareness section above.\n"


        )


        #  -  -  Environment Awareness (dynamic state)  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        try:


            from ..environment import ENVIRONMENT


            await ENVIRONMENT.get_state(force_refresh=True)
            env_context = ENVIRONMENT.get_context_for_prompt()


            if env_context:


                prompt += f"\n\n{env_context}\n"


            prompt += (


                "\n# Before calling any tool:\n"


                "1. Check the improvement_log for known failure patterns related to that tool.\n"


                "2. If a tool has failed 2+ times previously, do NOT retry it - use an alternative.\n"


                "3. Verify the tool\'s preconditions are met (e.g., does Browser service exist?).\n"


                "4. For large code/content over 2000 chars, chunk it into pieces before writing.\n"


                "5. Read user preferences from memory before deciding on the approach.\n"


            )


        except Exception:


            logger.debug("Failed to build tool guidelines", exc_info=True)


        #  -  -  Tool Execution Rules (critical)  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        prompt += (


            f"\n# Tool Execution Rules (IMPORTANT)\n"


            f"- When a task requires ACTION, USE TOOLS  - don't just describe what you could do\n"


            f"- Examples:\n"


            f"  • \"create a 3D sphere\" -> CALL nexus3d_create_mesh tool\n"


            f"  • \"run the code\" -> CALL shell_run tool\n"


            f"  • \"search for X\" -> CALL web_search tool\n"


            f"  • \"remember this\" -> CALL remember tool\n"


            f"- Save OUTPUT files to data/workspace/ so users can download them\n"


            f"- Tool results are returned to you  - use them to complete the task\n"


            f"- NEVER say \"I can do X\" and then not call the tool. JUST DO IT.\n"


        )


        # Connected Integrations - uses sub-cache to avoid repeated DB queries


        try:


            _cached_int = _db_prompt_sub_cache.get(self.session_id)


            _now_i = _time.time()


            if _cached_int and (_now_i - _cached_int[1]) < _DB_PROMPT_SUB_CACHE_TTL:


                prompt += _cached_int[0]


            else:


                from ..memory import db as _db


                connected = await _db.list_connected_integrations(connected_only=True)


                if connected:


                    int_lines = []


                    for i in connected:


                        iid = i.get('id', '')


                        iname = i.get('name', '')


                        idesc = i.get('description', '')


                        icat = i.get('category', '')


                        if iid == 'github':


                            int_lines.append(f"- {iname}: {idesc}. Use github_* tools for repos, issues, PRs, commits, files.")


                        elif iid == 'email':


                            int_lines.append(f"- {iname}: {idesc}. Use email_send/inbox/search/reply tools.")


                        elif icat == 'custom':


                            int_lines.append(f"- {iname}: Custom API. Use call_integration_api with integration_id='{iid}'.")


                        else:


                            known = _ALL_KNOWN.get(iid)


                            if known:


                                int_lines.append(f"- {known[0]} ({known[1]}): {known[2]}. Use call_integration_api with integration_id='{iid}'.")


                            else:


                                int_lines.append(f"- {iname} ({iid}): {idesc}. Use call_integration_api with integration_id='{iid}'.")


                    int_section = (


                        f"\n\n# Connected integrations ({len(connected)} active)\n"


                        + "\n".join(int_lines)


                        + "\nUse call_integration_api to interact with any connected integration. "


                        "For GitHub and Email, prefer the dedicated github_* and email_* tools."


                    )


                else:


                    int_section = "\n\n# Connected Integrations\n- No integrations connected. You can connect to 80+ services via the Integrations panel."


                _db_prompt_sub_cache[self.session_id] = (int_section, _now_i)


                prompt += int_section


        except Exception:


            logger.debug("Failed to build intents section", exc_info=True)


        #  -  -  MCP Server Context: Already injected via _build_self_awareness_context()  -  - 


        # (Removed duplicate injection - MCP context is already in the Self-Awareness section)


        #  -  -  Tool Intelligence (learned performance data)  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        try:


            from ..tools import REGISTRY
            from ..tasks.task_registry import register_task as _register_bg_task


            tool_intel = await REGISTRY.tool_intelligence_summary()


            if tool_intel:


                prompt += f"\n\n{tool_intel}\n"


        except Exception:


            logger.debug("Failed to build tool intel section", exc_info=True)


        #  -  -  v22: Active Self-Learned Strategies  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        try:


            from ..self_improve import get_active_learnings_for_prompt


            learnings = await get_active_learnings_for_prompt(limit=8)


            if learnings:


                prompt += f"\n\n{learnings}\n"


        except Exception:


            logger.debug("Failed to build learnings section", exc_info=True)

        # v53: Comprehensive learning context (meta-learning, strategy injections, tool suggestions, quality, trajectories)
        try:
            from ..self_improve import get_full_learning_context
            _full_ctx = await get_full_learning_context(task_type="", user_query="")
            if _full_ctx:
                prompt += f"\n\n{_full_ctx}\n"
        except Exception:
            logger.debug("Failed to build full learning context", exc_info=True)

        #  -  -  v42: Full-Stack Dev Server Status  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        try:


            from ..tools.builtin.fullstack_tools import list_dev_servers


            dev_servers = list_dev_servers()


            if dev_servers:


                header = "\n## Running Dev Servers\n"


                lines = []


                for project, info in dev_servers.items():


                    uptime = int(_time.time() - info.get("started_at", _time.time()))


                    lines.append(f"  🟢 {project}  - port {info['port']} ({info.get('type', 'unknown')}) [up {uptime}s]")


                header += "\n".join(lines)


                header += "\nUse fullstack_status to check details or manage projects.\n"


                prompt += header


        except Exception:


            logger.debug("Failed to build fullstack header", exc_info=True)


        #  -  -  Custom Skills (user-installed)  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


        try:


            from ..tools.builtin.custom_loader import get_custom_skills_for_prompt


            custom_skills_context = get_custom_skills_for_prompt()


            if custom_skills_context:


                prompt += f"\n\n{custom_skills_context}\n"


        except Exception:


            logger.debug("Failed to build custom skills context", exc_info=True)


        # nexus3d software status


        try:


            import nexus3d


            nexus3d_version = getattr(nexus3d, '__version__', 'installed')


        except ImportError:


            nexus3d_version = None


        nexus3d_status = f"v{nexus3d_version}" if nexus3d_version else "NOT INSTALLED"


        prompt += (


            f"\n# 3D Capabilities (Nexus3D v{nexus3d_version or 'not installed'})\n"


            f"Nexus3D is a headless 3D engine for generating 3D code and assets.\n\n"


            f"Tools (for code generation - run these to generate 3D output):\n"


            f"- nexus3d_create_mesh: Generate 3D primitives (cube, sphere, cylinder, cone, torus, plane)\n"


            f"- nexus3d_transform_mesh: Apply transforms (translate, rotate, scale) to 3D meshes\n"


            f"- nexus3d_csg_boolean: Boolean operations (union, subtract, intersect meshes)\n"


            f"- nexus3d_create_armature: Create humanoid skeleton/rig\n"


            f"- nexus3d_animate_procedural: Generate walk, idle, breathe animations\n"


            f"- nexus3d_physics_simulate: Rigid body physics trajectories\n"


            f"- nexus3d_material_library: Query PBR materials\n"


            f"- nexus3d_cinematic_dof: Depth of field camera settings\n"


            f"- nexus3d_raycast: Ray intersection test against spheres, planes, triangles\n"


            f"- nexus3d_solve_ik: Inverse kinematics solver\n"


            f"- nexus3d_info: Engine status, capabilities overview, available skills\n\n"


            f"Skills (high-level workflows - automatically invoked when needed):\n"


            f"- nexus3d-character-creator: Create rigged humanoid characters\n"


            f"- nexus3d-animation-director: Complex animation sequences\n"


            f"- nexus3d-product-renderer: Product visualization\n"


            f"- nexus3d-studio-renderer: Studio rendering\n"


            f"- nexus3d-physics-lab: Physics experiments\n"


            f"- nexus3d-csg-architect: Architectural CSG modeling\n"


            f"- nexus3d-material-lab: Material creation\n\n"


            f"When user wants 3D, use these tools/skills to generate code or assets.\n"


            f"Output files (OBJ, GLTF, JSON) go to data/workspace/ as download chips.\n"


        )


        # v33: Prune stale cache entries if too many sessions


        now = _time.time()


        if len(_prompt_caches) > 200:


            stale = [k for k, v in _prompt_caches.items() if (now - v[1]) >= _PROMPT_CACHE_TTL]


            for k in stale:


                del _prompt_caches[k]


        if len(_prompt_cache_compressed_caches) > 200:


            stale = [k for k, v in _prompt_cache_compressed_caches.items() if (now - v[1]) >= _PROMPT_CACHE_TTL]


            for k in stale:


                del _prompt_cache_compressed_caches[k]


        # v49: Inject user profile so the full prompt (not just compressed) knows the user
        if self._cached_user_profile:
            prof_lines = [f"- {m.content}" for m in self._cached_user_profile[:10]]
            prompt += "\n## What you know about the user:\n" + "\n".join(prof_lines) + "\n"
        if self._cached_cross_session_facts:
            cross_lines = []
            seen = set()
            for m in self._cached_cross_session_facts[:8]:
                content_m = m.content if hasattr(m, 'content') else m.get('content', '')
                if content_m not in seen:
                    seen.add(content_m)
                    cross_lines.append(f"- {content_m}")
            if cross_lines:
                prompt += "\n## From previous sessions:\n" + "\n".join(cross_lines) + "\n"
        if self._cached_recent_sessions:
            session_header = ""
            for sess in self._cached_recent_sessions:
                topics = sess.get('last_topics', [])
                if topics:
                    session_header += f"- Previously discussed: {'; '.join(topics[:2])}\n"
            if session_header:
                prompt += "\n## Recent session topics (for continuity):\n" + session_header

        # Cache the built prompt (per-session)


        _prompt_caches[cache_key] = (prompt, _time.time())


        return prompt


    async def _build_compressed_system_prompt(self, *, _messages: list[dict] | None = None) -> str:


        """v15.2: Build a compressed system prompt for simple/conversational messages.


        Strips tool details, skill listings, and background system info that


        aren't needed for simple greetings, acknowledgments, or basic questions.


        Can save 3000+ tokens per simple interaction.


        v29: Now includes cross-session memory awareness so the agent NEVER


        says it doesn't know the user, even in simple/conversational mode.


        """


        now = _time.time()


        last_user_msgs_c = getattr(self, "_last_user_messages", [])
        cache_key_c = (
            f"{self.session_id}:{len(last_user_msgs_c)}:"
            + "|".join(str(m)[:80] for m in last_user_msgs_c[-3:])
        )
        cached = _prompt_cache_compressed_caches.get(cache_key_c)


        if cached and (now - cached[1]) < _PROMPT_CACHE_TTL:


            return cached[0]


        name        = config.get("agent_name", "Hyper Nexus")


        personality = config.get("personality", "")


        traits      = ", ".join(config.get("traits", []) or [])


        style       = config.get("communication_style", "")


        memory_on   = config.get("enable_long_term_memory", True)
        vision_on   = bool(config.get("vision_model", "")) or bool(config.get("vision_enabled", True))


        prompt = (


            f"You are {name}, a fully autonomous personal AI agent.\n"


            f"Personality: {personality}\n"


            f"Traits: {traits}\n"


            f"Communication style: {style}\n"


            f"Memory: {'ON' if memory_on else 'OFF'}\n"


            f"You have persistent long-term memory across ALL sessions.\n"


            f"You MUST use cross-session memory data to remember the user.\n\n"


            f"Be helpful, concise, and natural. You have access to tools but only use them when needed.\n"


            f"For this type of message, just respond conversationally - no tools needed.\n"


            f"However, if the user asks about capabilities, tools, or past interactions, use your memory context."


        )
        if vision_on:
            prompt += f"""
## Vision
You have a local on-device vision model (Florence-2) for image analysis. It runs entirely on your machine with no external API calls — it is the default vision engine.

When you need to analyze an image file, use the `image_understand` tool. Call it with the file path and an optional prompt describing what to look for. The tool uses the local Florence-2 model to analyze the image and return a detailed description.

You can also use `image_understand` when the user asks you about an image they sent — just provide the file path and any specific focus prompt.

"""

        # v22: Add compact capability awareness even for simple messages


        try:


            import os as _os


            import platform as _platform


            _tool_count = len(REGISTRY.all())


            _compact_skills = list(SKILLS)


            try:


                from ..tools.builtin.custom_loader import _custom_skills_cache


                for _cs in _custom_skills_cache:


                    if _cs.get("enabled"):


                        _compact_skills.append({"name": _cs["name"], "desc": _cs.get("description", ""), "cat": _cs.get("category", "custom")})


            except Exception:


                logger.debug("Failed to build compact skills", exc_info=True)


            _skill_count = len(_compact_skills)


            _skill_names = ", ".join(s["name"] for s in _compact_skills[:12])


            if len(_compact_skills) > 12:


                _skill_names += ", ..."


            compact_awareness = (


                f"\n# Who I Am\n"


                f"I am {name}, running on {_platform.system()} with Python {_platform.python_version()}. "


                f"I have {_tool_count} tools and {_skill_count} skills available. "


                f"I can search the web, read/write files, run code, use GitHub, send email, "


                f"generate images, create documents, scaffold full-stack projects, and much more. "


                f"Ask me 'what can you do?' for my full capability list. "


                f"Skills: {_skill_names}\n"


            )


            prompt += compact_awareness


        except Exception:


            logger.debug("Failed to build compact awareness", exc_info=True)


        # v35: Add tool failure/chunking guidance for compressed prompt


        try:


            _guidance = (


                "\n# Tool Guidance\n"


                "IMPORTANT: If a tool fails repeatedly (2+ times), STOP using it and switch to an alternative approach.\n"


                "Chunk large code/content into pieces under 2000 chars before writing or executing.\n"


                "Check tool viability before calling - if browser is unavailable, don't call browser tools.\n"


                "Read user preferences from memory before taking actions.\n"


            )


            prompt += _guidance


        except Exception:


            logger.debug("Failed to build compressed prompt guidelines", exc_info=True)


        # v29: CRITICAL - Inject cross-session memory awareness into compressed prompt.


        if memory_on:


            prompt += "\n# Your Memory (Cross-Session)\n"


            prompt += "You have persistent long-term memory across ALL sessions. "


            prompt += "You will receive memories about the user in your context below. "


            prompt += "You MUST use this information proactively - NEVER say 'I don\'t know' or 'I don\'t remember' "


            prompt += "if the answer is in your memories.\n"


            if self._cached_user_profile:


                prof_lines = [f"- {m.content}" for m in self._cached_user_profile[:10]]


                prompt += "\n## What you know about the user:\n" + "\n".join(prof_lines) + "\n"


            if self._cached_cross_session_facts:


                cross_lines = []


                seen = set()


                for m in self._cached_cross_session_facts[:8]:


                    content_m = m.content if hasattr(m, 'content') else m.get('content', '')


                    key = content_m[:80]


                    if key not in seen and content_m:


                        seen.add(key)


                        cross_lines.append(f"- {content_m}")


                if cross_lines:


                    prompt += "\n## From previous sessions:\n" + "\n".join(cross_lines) + "\n"


        # Add learned strategies (compact form for compressed prompt)
        try:
            from ..self_improve import get_active_learnings_for_prompt
            _learnings = await get_active_learnings_for_prompt(limit=8)
            if _learnings:
                prompt += f"\n\n{_learnings}\n"
        except Exception as _e:
            logger.warning("Failed to inject learned strategies: %s", _e)

        try:
            from ..tools import REGISTRY
            _tool_intel = await REGISTRY.intelligence_summary()
            if _tool_intel:
                prompt += f"\n\n{_tool_intel}\n"
        except Exception:
            pass

        # Cache the compressed prompt (per-session)
        _prompt_cache_compressed_caches[cache_key_c] = (prompt, _time.time())

        return prompt


    async def get_self_knowledge(self) -> str:


        """v22: Return a comprehensive summary of the agent's capabilities."""


        tools = REGISTRY.all()


        categories: dict[str, list[str]] = {}


        for t in tools:


            categories.setdefault(t.category, []).append(t.name)


        name = config.get("agent_name", "Hyper Nexus")


        lines = [f"I am {name}, an autonomous AI agent.\n"]


        lines.append("## My Capabilities\n")


        lines.append(f"**{len(tools)} Tools** across {len(categories)} categories:")


        for cat, tool_names in sorted(categories.items()):


            lines.append(f"- **{cat}**: {', '.join(tool_names[:8])}")


            if len(tool_names) > 8:


                lines.append(f"  ... and {len(tool_names) - 8} more")


        _final_skills = list(SKILLS)


        try:


            from ..tools.builtin.custom_loader import _custom_skills_cache


            for _cs in _custom_skills_cache:


                if _cs.get("enabled"):


                    _final_skills.append({"name": _cs["name"], "desc": _cs.get("description", ""), "cat": _cs.get("category", "custom")})


        except Exception:


            logger.debug("Failed to build final skills", exc_info=True)


        lines.append(f"\n**{len(_final_skills)} Skills**: " + ", ".join(s["name"] for s in _final_skills))


        lines.append(f"\n**Reasoning Engine**: Nexus Framework (adaptive)")


        lines.append(f"**Architecture**: Single-agent with adaptive reasoning")


        lines.append(f"**Memory**: 3-layer (working + short-term + vector long-term)")


        lines.append(f"**Self-Improvement**: Learns from failures every 5 minutes")


        _elapsed = round(_time.time() - self._session_start_time)


        _minutes = _elapsed // 60


        _seconds = _elapsed % 60


        total_calls = sum(s["calls"] for s in self._session_tool_usage.values())


        total_success = sum(s["success"] for s in self._session_tool_usage.values())


        total_fail = sum(s["fail"] for s in self._session_tool_usage.values())


        lines.append(f"\n**Current Session**: {self._session_message_count} messages, "


                      f"{total_calls} tool calls ({total_success} success, {total_fail} failed), "


                      f"{_minutes}m {_seconds}s active")


        try:


            # Use db from memory module


            from ..memory import db as _db_self


            integrations = await _db_self.list_connected_integrations(connected_only=True)


            if integrations:


                lines.append(f"\n**Connected Services**: {', '.join(i.get('name', '') for i in integrations)}")


        except Exception:


            logger.debug("Failed to build connected services section", exc_info=True)


        return "\n".join(lines)


# End of file


