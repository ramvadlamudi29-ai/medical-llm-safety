from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Route:
    intent: str  # 'calculator' | 'summarize' | 'search'
    confidence: float = 1.0


_MATH_HINT = re.compile(
    r"[0-9].*[\+\-\*\/%].*[0-9]|\d+\s*%\s*of\s*\d+", re.I
)
_SUMMARIZE = re.compile(r"\b(summari[sz]e|tl;dr|summary|brief)\b", re.I)
_CALC_VERB = re.compile(r"\bcalculate\b|\bcompute\b", re.I)


class AgentRouter:
    """Lightweight intent router."""

    def route(self, query: str) -> Route:
        q = (query or "").strip()
        if not q:
            return Route(intent="search", confidence=0.0)
        if _SUMMARIZE.search(q):
            return Route(intent="summarize", confidence=0.9)
        if _CALC_VERB.search(q) or _MATH_HINT.search(q):
            return Route(intent="calculator", confidence=0.9)
        return Route(intent="search", confidence=0.5)
