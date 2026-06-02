"""
Text cleaning and DSML/XML/tool-call markup stripping utilities.
"""
from __future__ import annotations
import ast
import json
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

from ..tools import REGISTRY
from .tool_parsing import (
    _TEXT_TOOL_CODEBLOCK_RE, _TEXT_TOOL_INVOKE_RE,
)


_ITARKUP_STRIP_RE = re.compile(r"☚.*?☛", re.DOTALL)


def _strip_python_leaks(text: str) -> str:


    """Remove raw Python scripts that leaked into visible assistant text.


    Multi-strategy detection:


      1. Shebang line


      2. import/from block at line start


      3. ast.parse — syntactically valid Python with few NL markers


      4. High-density heuristic (covers bare calls like process(item))


    """


    if not text:


        return text


    # --- Strategy 1: shebang line ---


    # Matches any #! line containing "python" (handles /usr/bin/env python,


    # /usr/bin/python3, /usr/bin/python3.14, /bin/python, etc.)


    t = re.sub(


        r'(?:^|\n)#![^\n]*python[^\n]*\n[\s\S]*?(?=\n{3,}|\Z)',


        '',


        text,


        flags=re.MULTILINE,


    )


    if t != text:


        return t


    # --- Strategy 2: block starting with import/from at line start ---


    t = re.sub(


        r'(?:^|\n)(?:from\s+\S+\s+import|import\s+\S+)\s*[\s\S]*?(?=\n{3,}|\Z)',


        '',


        text,


        flags=re.MULTILINE,


    )


    if t != text and len(t.strip()) < len(text) * 0.3:


        while True:


            n = re.sub(


                r'(?:^|\n)(?:from\s+\S+\s+import|import\s+\S+)\s*[\s\S]*?(?=\n{3,}|\Z)',


                '',


                t,


                flags=re.MULTILINE,


            )


            if n == t:


                break


            t = n


        return t


    # --- Strategy 3: ast.parse — valid Python with few conversational words ---


    lines = text.strip().split('\n')


    non_blank = [L for L in lines if L.strip()]


    if len(non_blank) >= 2:


        _nl_words = len(re.findall(r'\b(the|is|are|was|were|this|that|these|those|have|has|been|been|will|would|could|should|might|may|shall|can|do|does|did|here|there|and|but|or|for|nor|yet|so|with|without|because|although|while|since|after|before|until|if|then|else|than|as|of|at|by|to|from|in|on|no|not|very|just|about|above|below|between|through|during|before|after|up|down|out|off|over|under|again|further|then|once|here|there|when|where|why|how|all|each|every|both|few|more|most|other|some|such|only|own|same|too|really|also|now)\b', text, re.IGNORECASE))


        try:


            ast.parse(text)


            # Valid Python with very few natural-language words = code leak


            if _nl_words < max(3, len(non_blank) * 0.3):


                return ""


        except SyntaxError:


            logger.debug("Failed to parse code block as multiline function call", exc_info=True)


        # Try parsing just the non-blank indented block (ignore surrounding text)


        try:


            _indented = '\n'.join(non_blank)


            ast.parse(_indented)


            if _nl_words < max(3, len(non_blank) * 0.3):


                return ""


        except SyntaxError:


            logger.debug("Failed to parse non-blank indented block", exc_info=True)


    # --- Strategy 4: high-density heuristic (lower threshold + bare calls) ---


    if len(non_blank) >= 2:


        py_score = 0


        for L in non_blank:


            s = L.strip()


            if re.match(


                r'^(import |from |def |class |return |if |elif |else:|for |while |with |try:|except |finally:|raise |yield |async |await |print\(|pass\b|break\b|continue\b|del\b|global\b|nonlocal\b)',


                s,


            ):


                py_score += 2


            elif ' = ' in s or s.endswith(':'):


                py_score += 1


            elif s.startswith(('#', '"""', "'''")):


                py_score += 0.5


            if re.search(r'\.[a-zA-Z_]\w*\s*\(', s):


                py_score += 0.5


            # Bare function call like process(item), save(), etc.


            if re.match(r'[a-z_]\w*\s*\(', s) and not re.match(r'(i|you|he|she|it|we|they)\b', s, re.IGNORECASE):


                py_score += 1


            # String literal detection


            if re.match(r'["\']', s):


                py_score += 0.5


        avg = py_score / len(non_blank)


        if avg >= 1.0:


            return ""


    return text


_BASELINE_STRIP_PATTERNS: tuple[re.Pattern, ...] = (


    re.compile(r"❌.*?(?=\n|$)", re.DOTALL),            # cross-icon metadata footers


    re.compile(r"✅.*?(?=\n|$)", re.DOTALL),


    re.compile(r"\ud83d\udd0d.*?(?=\n|$)", re.DOTALL),  # search emoji metadata


    re.compile(r"\ud83d\udcdd.*?(?=\n|$)", re.DOTALL),   # memo emoji metadata


    re.compile(


        r"`[^`]*?(?:No (?:new|recent|unread)|no (?:new|recent|unread))[^`]*?`",


        re.IGNORECASE,


    ),


    re.compile(r"❌\s*tool.*?(?=\n)", re.IGNORECASE | re.DOTALL),


)  # end _BASELINE_STRIP_PATTERNS


# v45: Broad tool-call markup stripping - removes ALL known text-based tool call


# formats from assistant content so they don't leak into visible chat as "thoughts".


def _find_matching_brace(text: str, start: int) -> int:
    """Find the matching closing brace for the brace at position start."""
    if start >= len(text) or text[start] != "{":
        return start
    depth = 0
    for pos in range(start, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return pos
    return start


def _strip_nested_json_blocks(text: str) -> str:
    """Strip Action:/function_call:/[tool] blocks with proper brace-counting.

    Replaces non-greedy regex patterns that break on nested JSON.
    """
    if not text:
        return text
    result = []
    i = 0
    while i < len(text):
        # Pattern 1: Action: name\nAction Input: {nested-json}
        m = re.match(
            r'Action:\s*\w+\s*\nAction\s*Input:\s*',
            text[i:], re.DOTALL
        )
        if m:
            json_start = i + m.end()
            if json_start < len(text) and text[json_start] == "{":
                json_end = _find_matching_brace(text, json_start)
                if json_end > json_start:
                    i = json_end + 1
                    continue
        # Pattern 2: function_call: tool_name {nested-json}
        m = re.match(
            r'function_call:\s*\w+\s*',
            text[i:], re.DOTALL | re.IGNORECASE
        )
        if m:
            json_start = i + m.end()
            if json_start < len(text) and text[json_start] == "{":
                json_end = _find_matching_brace(text, json_start)
                if json_end > json_start:
                    i = json_end + 1
                    continue
        # Pattern 3: [tool_name]({nested-json})
        m = re.match(
            r'\[[\w-]+\]\s*\(',
            text[i:], re.DOTALL
        )
        if m:
            json_start = i + m.end()
            if json_start < len(text) and text[json_start] == '{':
                json_end = _find_matching_brace(text, json_start)
                if json_end > json_start and json_end + 1 < len(text) and text[json_end + 1] == ')':
                    i = json_end + 2
                    continue
        result.append(text[i])
        i += 1
    return ''.join(result)


# Lazy-built regex for function-call style: tool_name(param='value', ...)
# Built on first use so REGISTRY.all_tools() is guaranteed populated.
_TEXT_TOOL_FUNCCALL_STRIP_RE: re.Pattern | None = None
_TEXT_TOOL_FUNCCALL_STRIP_LOCK = threading.Lock()

def _build_funcall_strip_re() -> re.Pattern | None:
    global _TEXT_TOOL_FUNCCALL_STRIP_RE
    if _TEXT_TOOL_FUNCCALL_STRIP_RE is not None:
        return _TEXT_TOOL_FUNCCALL_STRIP_RE
    with _TEXT_TOOL_FUNCCALL_STRIP_LOCK:
        if _TEXT_TOOL_FUNCCALL_STRIP_RE is not None:
            return _TEXT_TOOL_FUNCCALL_STRIP_RE
        try:
            _names = sorted({t.name for t in REGISTRY.all_tools()}, key=len, reverse=True)
            if not _names:
                return None
            # Uses balanced-parentheses pattern to handle nested parens in args
            _pattern = (
                r'\b(?:' + '|'.join(re.escape(n) for n in _names) +
                r')\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)'
            )
            _TEXT_TOOL_FUNCCALL_STRIP_RE = re.compile(_pattern, re.DOTALL | re.IGNORECASE)
            return _TEXT_TOOL_FUNCCALL_STRIP_RE
        except Exception:
            return None


def _strip_tool_call_markup(text: str) -> str:


    """Strip ALL tool call markup formats from visible assistant content.


    Prevents "thought leaking" where models output tool calls as visible text


    instead of using the function-calling API.


    """


    if not text:


        return ""


    # First: strip raw Python scripts that models emit instead of using tools


    text = _strip_nested_json_blocks(text)


    text = _strip_python_leaks(text)


    for _p in (


        _ITARKUP_STRIP_RE,


        _TEXT_TOOL_CODEBLOCK_RE,


        _TEXT_TOOL_INVOKE_RE,


    ):


        text = _p.sub("", text)


    text = re.sub(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>[\s\S]*?<\/[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+>', '', text, flags=re.DOTALL | re.IGNORECASE)


    text = re.sub(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?\/>', '', text, flags=re.IGNORECASE)


    text = re.sub(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\/[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+>', '', text, flags=re.IGNORECASE)


    text = re.sub(r'<DSML_\w+[^>]*>[\s\S]*?</DSML_\w+>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\/DSML_\w+>', '', text, flags=re.IGNORECASE)


    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL | re.IGNORECASE)


    text = re.sub(r'<tool_calls>.*?</tool_calls>', '', text, flags=re.DOTALL | re.IGNORECASE)


    # Strip <|python_tag|> and other <|...|> inline tags plus their content up to


    # the next paragraph break or end-of-string (common model leak format)


    text = re.sub(r'<\|[^|]+\|>[\s\S]*?(?=\n{3,}|\Z)', '', text, flags=re.DOTALL)


    # Strip ALL remaining DSML/XML elements (including their content) - catch-all


    # This handles <parameter>...</parameter>, <tool_response>...</tool_response>,


    # <tool_result>...</tool_result>, <error>...</error>, and any other DSML tags


    # that might leak through individual regex patterns. Uses backreference (\1) to


    # ensure opening and closing tags match.


    text = re.sub(r'<(parameter|tool_response|tool_result|error|result|response|tool_input|input|output|status|invoke)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Strip function-call style: tool_name(param='value', param2=42)
    # The system prompt documents tools as nexus3d_create_mesh(shape='sphere')
    # and text-only models output calls in this exact format as visible text.
    # Without this pattern, the tool call syntax leaks into the thought block
    # AND the final assistant answer.
    _funcall_re = _build_funcall_strip_re()
    if _funcall_re:
        text = _funcall_re.sub('', text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    # Second pass: strip Python leaks AFTER DSML tags are removed,


    # so patterns like <|python_tag|>from pptx import Presentation


    # are seen as bare "from pptx import Presentation" by the leak detector


    


    text = _strip_nested_json_blocks(text)


    text = _strip_python_leaks(text)

    # Third pass: strip standalone tool-name leaks
    text = _strip_leaked_tool_names(text)


    return text.strip()


def _strip_leaked_tool_names(text: str) -> str:
    _tool_names: list[str] | None = None
    try:
        _tool_names = sorted({t.name for t in REGISTRY.all_tools()}, key=len, reverse=True)
    except Exception:
        pass
    if not _tool_names:
        return text

    _joined = "|".join(re.escape(n) for n in _tool_names)

    # 1. [Calling tool_name] or [Calling tool_name(args)]
    text = re.sub(
        r"\[Calling\s+(?:" + _joined + r")(?:\([^)]*\))?\]",
        "", text, flags=re.IGNORECASE,
    )

    # 2. `tool_name` optionally followed by (args)
    text = re.sub(
        r"`(?:" + _joined + r")`\s*(?:\([^)]*\))?",
        "", text, flags=re.IGNORECASE,
    )

    # 3. [[tool_name]] or [tool_name] (bracketed reference)
    text = re.sub(
        r"\[\[(?:" + _joined + r")\]\]",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)\[(?:" + _joined + r")\](?!\w)",
        "", text, flags=re.IGNORECASE,
    )

    # 4. Line-level: standalone tool name on its own line,
    #    possibly followed by param-looking lines
    #    Handles: "tool_name\nParam = value", "tool_name\nParam:\n  value",
    #    and also "tool_name\nCode\nvalue" (no = sign)
    lines = text.split("\n")
    _tool_set = frozenset(n.lower() for n in _tool_names)
    _stripped: list[str] = []
    _skip_next = 0
    for idx, line in enumerate(lines):
        if _skip_next > 0:
            _skip_next -= 1
            continue
        stripped_line = line.strip()
        if stripped_line.lower() in _tool_set:
            _skip_next = 0
            for j in range(idx + 1, min(idx + 6, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    break
                # Skip param lines: "Key = value", "Key: value",
                # or bare param name (single word, no punctuation)
                if re.match(r"^[a-zA-Z_]\w*\s*[=:]\s*", next_line) or re.match(r"^[A-Z]\w*$", next_line):
                    _skip_next += 1
                else:
                    break
            continue
        _stripped.append(line)
    text = "\n".join(_stripped)

    # 5. YAML-style: tool_name:\n  param: value
    text = re.sub(
        r"(?m)^\s*(?:" + _joined + r")\s*:\s*\n(?:\s{2,}[^\n]*\n)*",
        "", text,
    )

    return text.strip()


def _strip_dsml_from_text(text: str) -> str:


    """Strip DSML/XML tool call markup from parameter strings.


    Prevents DSML tags embedded in tool parameters from leaking into the


    frontend WebSocket payload and being rendered visibly in the chat UI.


    Handles both standard XML format and the <|DSML|tag> pipe format.


    """


    if not text:


        return text


    text = re.sub(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>[\s\S]*?<\/[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+>', '', text, flags=re.DOTALL | re.IGNORECASE)


    text = re.sub(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?\/>', '', text, flags=re.IGNORECASE)


    text = re.sub(r'<[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+(?:\s+[^>]*)?>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\/[\|｜¦│❘ǀ]DSML[\|｜¦│❘ǀ][\w-]+>', '', text, flags=re.IGNORECASE)


    text = re.sub(r'<DSML_\w+[^>]*>[\s\S]*?</DSML_\w+>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\/DSML_\w+>', '', text, flags=re.IGNORECASE)


    text = re.sub(r'<tool_call>[\s\S]*?<\/tool_call>', '', text, flags=re.DOTALL | re.IGNORECASE)


    text = re.sub(r'<invoke[^>]*>[\s\S]*?<\/invoke>', '', text, flags=re.DOTALL | re.IGNORECASE)


    text = re.sub(r'<tool_calls>[\s\S]*?<\/tool_calls>', '', text, flags=re.DOTALL | re.IGNORECASE)


    text = re.sub(r'<(parameter|tool_response|tool_result|error|result|response|tool_input|input|output|status|invoke)[^>]*>[\s\S]*?<\/\1>', '', text, flags=re.DOTALL | re.IGNORECASE)


    # Only strip DSML-specific tags — not arbitrary XML-like tags like <string>, <module>
    # which commonly appear in Python traceback output from python_exec.
    text = re.sub(r'<(?:DSML_\w+|parameter|tool_response|tool_result|error|result|response|tool_input|input|output|status|invoke|tool_call|tool_calls)(?:\s+[^>]*)?\/?>', '', text)


    text = re.sub(r'\n{3,}', '\n\n', text)


    text = _strip_nested_json_blocks(text)


    text = _strip_leaked_tool_names(text)


    return text.strip()


def _sanitize_error_msg(msg: str, max_len: int = 250) -> str:
    """Sanitize error messages to prevent internal data leakage to the frontend.

    Strips file paths, masks API keys, long hex/digit strings, and truncates.
    """
    if not msg:
        return msg
    # Strip Windows absolute paths: C:\Users\...
    msg = re.sub(r'[A-Za-z]:\\(?:[^\s,\)\]\'"]{1,})', '<path>', msg)
    # Strip Unix absolute paths: /home/user/... (only matches actual file-system-looking paths)
    msg = re.sub(r'(?:/[a-zA-Z_][a-zA-Z0-9_./-]*){2,}', '<path>', msg)
    # Mask OpenAI-style API keys: sk-...
    msg = re.sub(r'sk-[A-Za-z0-9\\-]{8,}', 'sk-***...***', msg)
    # Mask long hex strings (16+ chars)
    msg = re.sub(r'[0-9a-fA-F]{16,}', '<hex>', msg)
    # Mask long digit strings (16+ digits, likely synthetic IDs rather than timestamps)
    msg = re.sub(r'\b\d{16,}\b', '<num>', msg)
    if len(msg) > max_len:
        msg = msg[:max_len] + '...'
    return msg


def _validate_tool_call_pairs(messages: list[dict]) -> None:
    """Fix orphaned tool_call_id / tool message pairs in-place.

    The OpenAI API requires that an assistant message with ``tool_calls`` be
    immediately followed by tool messages whose ``tool_call_id`` values
    exactly match those in the assistant message.  This function fixes both
    directions:

    * Removes ``tool_call`` entries from the assistant message that lack a
      matching tool result.
    * Removes **tool messages** whose ``tool_call_id`` has no matching entry
      in the preceding assistant message.

    Called right before each ``llm.complete()`` call as a safety net.
    """
    i = 0
    while i < len(messages):
        msg = messages[i]
        tool_calls = msg.get("tool_calls")
        if msg.get("role") != "assistant" or not tool_calls:
            i += 1
            continue

        # Collect tool_call_ids that actually have tool results right after
        paired: set[str] = set()
        valid_ids = {tc.get("id", "") for tc in tool_calls}
        tool_indices: list[int] = []
        for j in range(i + 1, len(messages)):
            if messages[j].get("role") == "tool":
                tid = messages[j].get("tool_call_id", "")
                if tid:
                    paired.add(tid)
                tool_indices.append(j)
            else:
                break  # Stop at first non-tool message after assistant

        # Remove tool_call_ids from assistant that have no matching result
        kept = [tc for tc in tool_calls if tc.get("id", "") in paired]
        if len(kept) != len(tool_calls):
            if kept:
                msg["tool_calls"] = kept
            else:
                # Empty tool_calls array causes OpenAI 400 error — remove key entirely
                msg.pop("tool_calls", None)

        # Remove tool messages whose tool_call_id has no matching assistant entry
        for idx in reversed(tool_indices):
            tid = messages[idx].get("tool_call_id", "")
            if tid not in valid_ids:
                del messages[idx]

        i += 1

    # Second pass: remove tool messages that have NO preceding
    # assistant message with matching tool_calls at all (edge case:
    # text-only assistant followed by orphaned tool messages).
    i = 0
    last_assistant_had_tool_calls = False
    last_valid_ids: set[str] = set()
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant":
            tc = msg.get("tool_calls")
            last_assistant_had_tool_calls = bool(tc)
            last_valid_ids = {t.get("id", "") for t in tc} if tc else set()
            i += 1
        elif msg.get("role") == "tool" and not last_assistant_had_tool_calls:
            del messages[i]
        elif msg.get("role") == "tool" and last_assistant_had_tool_calls:
            tid = msg.get("tool_call_id", "")
            if tid not in last_valid_ids:
                del messages[i]
            else:
                i += 1
        else:
            i += 1


def _truncate_params_for_display(args: dict, max_str_len: int = 500) -> dict:


    """Truncate large string values in tool args before sending to frontend.


    Also strips DSML/XML markup to prevent rendering artifacts in the UI.


    The params dict is ONLY used for the frontend UI preview (truncated to 60


    chars in the thinking block), so there is no need to send full code strings


    (which can be 10,000+ chars) over the WebSocket. This prevents JSON parser


    buffer boundary issues in the frontend when large code blocks are passed.


    Original args are NOT modified - only the emit payload is truncated.


    """


    if not args:


        return args


    truncated: dict[str, object] = {}


    for k, v in args.items():


        if isinstance(v, str):


            stripped = _strip_dsml_from_text(v)


            if len(stripped) > max_str_len:


                truncated[k] = stripped[:max_str_len] + "..."


            else:


                truncated[k] = stripped


        elif isinstance(v, dict):


            truncated[k] = _truncate_params_for_display(v, max_str_len)


        elif isinstance(v, list):


            items: list[object] = []


            for item in v:


                if isinstance(item, str):


                    stripped = _strip_dsml_from_text(item)


                    if len(stripped) > max_str_len:


                        items.append(stripped[:max_str_len] + "...")


                    else:


                        items.append(stripped)


                elif isinstance(item, dict):


                    items.append(_truncate_params_for_display(item, max_str_len))


                else:


                    items.append(item)


            truncated[k] = items


        else:


            truncated[k] = v


    return truncated