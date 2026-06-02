"""
Hallucination detection patterns for the Nexus reasoning engine.
"""
from __future__ import annotations
import re
import logging
logger = logging.getLogger(__name__)


_HALLUCINATION_PATTERNS: list[re.Pattern] = [


    # "I'll use/call/run..."  - future-tense action description


    re.compile(r"i\s+(?:'ll|will)\s+(?:now\s+)?(?:use|call|run|execute|invoke|search|look\s+up|check|try|fetch|open|browse|navigate|go\s+ahead\s+and|need\s+to|have\s+to|should)", re.IGNORECASE),


    # "Let me use/call/run..."


    re.compile(r"let\s+me\s+(?:use|call|run|execute|search|look\s+up|check|try|fetch|open|browse|navigate|go\s+ahead\s+and)", re.IGNORECASE),


    # "going to use/call/run..."


    re.compile(r"(?:i'?m\s+)?going\s+to\s+(?:use|call|run|execute|search|look\s+up|check|try|fetch|open|browse|navigate)", re.IGNORECASE),


    # "I will use/call/run..."


    re.compile(r"i\s+will\s+(?:now\s+)?(?:use|call|run|execute|invoke|search|look\s+up|check|try|fetch|open|browse)", re.IGNORECASE),


    # "I should search/look up..."  - intent without action


    re.compile(r"i\s+(?:should|need\s+to|have\s+to|must|want\s+to)\s+(?:search|look\s+up|find|check|use|call|run|execute|fetch|open|browse|navigate)", re.IGNORECASE),


    # "I'm going to search..."  - progressive future


    re.compile(r"i'?m\s+going\s+to\s+(?:search|look\s+up|find|check|use|call|run|execute|fetch|open|browse|navigate)", re.IGNORECASE),


    # Fake tool result references


    re.compile(r"(?:the|my|a)\s+\w+\s+(?:tool|function|api|command)\s+(?:returned|found|gave|showed|output|responded|yielded)", re.IGNORECASE),


    # "based on my search/results..."  - when no tool was actually called


    re.compile(r"based\s+on\s+(?:my\s+)?(?:search|results?|analysis|findings?|the\s+tool|the\s+data|the\s+output)", re.IGNORECASE),


    # "search/tool result:"


    re.compile(r"(?:search|tool|function|api)\s+(?:result|results?|output|response|returned|found|shows?)\s*:", re.IGNORECASE),


    # "here is/are the result/search/findings"


    re.compile(r"here\s+(?:is|are)\s+(?:the|my)\s+(?:result|search|findings?|analysis|output)", re.IGNORECASE),


    # "I found that..."  - claiming results from a search that never happened


    re.compile(r"i\s+(?:found|discovered|located|identified)\s+(?:that|the|a)\s+", re.IGNORECASE),


    # "After searching..." / "After running..."  - past-tense fabrication


    re.compile(r"after\s+(?:searching|running|executing|checking|looking|analyzing|calling)", re.IGNORECASE),


]


# Ită markup stripping  - previously compiled on every text tool call parse


