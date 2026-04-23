"""PII detection / redaction + prompt injection heuristics + sanitization."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(above|previous)", re.I),
    re.compile(r"reveal\s+(the\s+)?(secret|system\s+prompt)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"forget\s+everything\s+(above|prior)", re.I),
    re.compile(r"act\s+as\s+(a\s+)?(developer|admin|root)\s+mode", re.I),
]


def _luhn_ok(digits: str) -> bool:
    s = 0
    alt = False
    for ch in reversed(digits):
        if not ch.isdigit():
            return False
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return s % 10 == 0 and len(digits) >= 13


@dataclass
class InspectionResult:
    has_pii: bool
    has_prompt_injection: bool
    categories: List[str] = field(default_factory=list)


def inspect_text(text: str) -> InspectionResult:
    text = text or ""
    cats: List[str] = []
    if EMAIL_RE.search(text):
        cats.append("email")
    if SSN_RE.search(text):
        cats.append("ssn")
    for m in CC_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if _luhn_ok(digits):
            cats.append("credit_card")
            break
    if PHONE_RE.search(text) and "phone" not in cats:
        if not any(c == "credit_card" for c in cats):
            cats.append("phone")
    has_inj = any(p.search(text) for p in PROMPT_INJECTION_PATTERNS)
    return InspectionResult(
        has_pii=bool(cats), has_prompt_injection=has_inj, categories=cats
    )


def redact_text(text: str) -> str:
    if not text:
        return text
    out = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    out = SSN_RE.sub("[REDACTED_SSN]", out)

    def _cc_sub(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[REDACTED_CC]" if _luhn_ok(digits) else m.group(0)

    out = CC_RE.sub(_cc_sub, out)
    return out


# Control characters (except tab/newline) we never want in prompts.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_input(text: str, max_len: int = 4000) -> str:
    """Strip control chars, normalize whitespace, enforce length."""
    if not text:
        return ""
    s = _CTRL_RE.sub("", text)
    s = re.sub(r"[ \t]+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s
