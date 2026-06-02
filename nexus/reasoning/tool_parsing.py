"""
Text-based tool call parsing regex patterns.
"""
from __future__ import annotations
import re


# Format 0: <tool_call>JSON</tool_call>  - direct XML format used by many LLMs


_TEXT_TOOL_DIRECT_CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)


# Text tool call parser  - Format 1: ☚JSON☛


_TEXT_TOOL_FORMAT1_RE = re.compile(r"☚\s*(.*?)\s*☛", re.DOTALL)


# Format 1b: ☚tool_name☛value☚...


_TEXT_TOOL_FORMAT1B_RE = re.compile(r"☚(\w+)(.*?)☛", re.DOTALL)


_TEXT_TOOL_FORMAT1B_ARG_RE = re.compile(r"☚(\w+)>(.*?)(?:☚|$)", re.DOTALL)


# Format 3: Action/Action Input (text format for non-tool-calling models)


_TEXT_TOOL_FORMAT3_RE = re.compile(
    r"Action:\s*(\w+)\s*\nAction\s*Input:\s*(.*)",
    re.DOTALL,
)


# Format 4: [tool_name]({...}) bracket syntax


_TEXT_TOOL_FORMAT4_RE = re.compile(r"\[(\w+)\]\((.*)\)", re.DOTALL)


# Format 4b: <invoke name="tool_name"> or <invoke name="tool_name">\n<parameter>value


_TEXT_TOOL_PROSE_RE1 = re.compile(


    r"(?:i'?ll|let\s+me|going\s+to|i\s+will)\s+(?:now\s+)?(?:use|call|run|execute|invoke)\s+"


    r"(?:the\s+)?(\w+)(?:\s+(?:tool|function|command))?"


    r"(?:\s+(?:to|with|for|using)\s+)?(.+?)(?:\.|$)",


    re.IGNORECASE | re.DOTALL,


)


_TEXT_TOOL_PROSE_RE2 = re.compile(


    r"(?:^|\n)\s*(?:using|calling)\s+(?:the\s+)?(\w+)"


    r"(?:\s+(?:tool|function|command))?\s*[:\s]\s*(.+?)(?:\n|$)",


    re.IGNORECASE | re.DOTALL,


)


# Format 6: code-block style tool calls


_TEXT_TOOL_CODEBLOCK_RE = re.compile(


    r"```\s*(?:json|tool|function|call)?\s*\n(.*?)\n\s*```",


    re.DOTALL | re.IGNORECASE,


)


# Format 7: "function_call: tool_name" followed by JSON


_TEXT_TOOL_FUNCCALL_RE = re.compile(
    r"function[_-]?call\s*:\s*(\w+)\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)


# Format 8: <invoke name="tool_name"><parameter name="key">value</parameter></invoke>  (DSML/Codebuff-style XML)


# Often wrapped in <tool_calls>...</tool_calls>


_TEXT_TOOL_INVOKE_RE = re.compile(


    r'<invoke\s+name=[\"\'](\w+)[\"\']>(.*?)</invoke>',


    re.DOTALL | re.IGNORECASE,


)


_TEXT_TOOL_INVOKE_PARAM_RE = re.compile(


    r'<parameter\s+name=[\"\'](\w+)[\"\'].*?>(.*?)</parameter>',


    re.DOTALL | re.IGNORECASE,


)


# Markdown code fence cleanup


_MARKDOWN_FENCE_STRIP_RE = re.compile(r"```\w*\n?")


# Inline JSON extraction


_INLINE_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


# Key=value extraction


_KV_PAIR_RE = re.compile(r"(\w+)\s*[:=]\s*['\"]?([^'\"},]+?)['\"]?\s*(?:,|\n|$)")


_SIMPLE_NUMBERED_LIST_RE = re.compile(r"\b\d+[.)]\s")


