"""Thin facade over security/pii_filter for nicer call sites.

Importing from ``core.security`` keeps pipeline-level code clean while
the heavy lifting still lives in ``security/pii_filter.py``.
"""
from __future__ import annotations
from typing import Tuple

from security.pii_filter import (
    InspectionResult,
    inspect_text,
    redact_text,
    sanitize_input,
)


def guard_input(text: str, *, max_len: int = 4000) -> Tuple[str, InspectionResult]:
    """Sanitize + inspect user input. Returns ``(clean_text, inspection)``."""
    clean = sanitize_input(text or "", max_len=max_len)
    inspection = inspect_text(clean)
    return clean, inspection


__all__ = [
    "InspectionResult",
    "inspect_text",
    "redact_text",
    "sanitize_input",
    "guard_input",
]
