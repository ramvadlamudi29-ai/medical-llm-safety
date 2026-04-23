"""Error taxonomy + helpers for consistent API responses.

Every public failure mode maps to one of these error codes so callers can
rely on a stable contract. Wire-format::

    {"ok": false, "error": "<code>", "message": "...", "trace_id": "..."}
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Stable error codes (do not rename without bumping API version).
VALIDATION_ERROR = "validation_error"
AUTH_ERROR = "auth_error"
RATE_LIMITED = "rate_limited"
PAYLOAD_TOO_LARGE = "payload_too_large"
NOT_FOUND = "not_found"
TIMEOUT_ERROR = "timeout_error"
INTERNAL_ERROR = "internal_error"
SECURITY_BLOCKED = "security_blocked"

STATUS_FOR: Dict[str, int] = {
    VALIDATION_ERROR: 422,
    AUTH_ERROR: 401,
    RATE_LIMITED: 429,
    PAYLOAD_TOO_LARGE: 413,
    NOT_FOUND: 404,
    TIMEOUT_ERROR: 504,
    INTERNAL_ERROR: 500,
    SECURITY_BLOCKED: 400,
}


class AppError(Exception):
    """Base class for typed application errors."""

    code: str = INTERNAL_ERROR

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message or (code or self.code))
        self.message = message or (code or self.code)
        if code:
            self.code = code
        self.details = details or {}


class ValidationError(AppError):
    code = VALIDATION_ERROR


class AuthError(AppError):
    code = AUTH_ERROR


class TimeoutError_(AppError):
    code = TIMEOUT_ERROR


class SecurityBlocked(AppError):
    code = SECURITY_BLOCKED


@dataclass
class ErrorEnvelope:
    code: str
    message: str
    details: Dict[str, Any]
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ok": False,
            "error": self.code,
            "message": self.message,
        }
        if self.details:
            d["details"] = self.details
        if self.trace_id:
            d["trace_id"] = self.trace_id
        return d


def envelope(
    code: str,
    message: str = "",
    details: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    return ErrorEnvelope(
        code=code,
        message=message or code,
        details=details or {},
        trace_id=trace_id,
    ).to_dict()


def status_for(code: str) -> int:
    return STATUS_FOR.get(code, 500)
