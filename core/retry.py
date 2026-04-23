"""Retry + timeout wrappers with graceful fallback.

These helpers are intentionally dependency-free so they work in offline
environments and inside tests without altering call timing meaningfully.
"""
from __future__ import annotations
import asyncio
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Iterable, Optional, Tuple, TypeVar

T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    *,
    retries: int = 2,
    base_delay: float = 0.1,
    exceptions: Tuple[type, ...] = (Exception,),
) -> T:
    """Run ``fn`` with exponential backoff. Re-raises the last exception."""
    last: Optional[BaseException] = None
    for attempt in range(max(0, retries) + 1):
        try:
            return fn()
        except exceptions as e:  # noqa: BLE001
            last = e
            if attempt >= retries:
                break
            time.sleep(base_delay * (2 ** attempt))
    assert last is not None
    raise last


def with_fallback(
    fn: Callable[[], T],
    *,
    fallback: T,
    exceptions: Tuple[type, ...] = (Exception,),
) -> T:
    """Run ``fn``; on any matching exception return ``fallback``."""
    try:
        return fn()
    except exceptions:  # noqa: BLE001
        return fallback


async def run_with_timeout(
    coro: Awaitable[T], *, timeout_s: float
) -> T:
    """Await ``coro`` with a hard timeout. Raises asyncio.TimeoutError on miss."""
    return await asyncio.wait_for(coro, timeout=timeout_s)


def timeout(seconds: float) -> Callable:
    """Decorator that enforces a timeout on async callables."""

    def deco(fn: Callable[..., Awaitable[T]]):
        @wraps(fn)
        async def w(*a: Any, **kw: Any) -> T:
            return await asyncio.wait_for(fn(*a, **kw), timeout=seconds)
        return w
    return deco


def safe_call(
    fn: Callable[..., T],
    *args: Any,
    fallback: T,
    log_with: Optional[Callable[[str], None]] = None,
    exceptions: Iterable[type] = (Exception,),
    **kwargs: Any,
) -> T:
    """Call ``fn`` and return ``fallback`` on failure (never raises)."""
    try:
        return fn(*args, **kwargs)
    except tuple(exceptions) as e:  # noqa: BLE001
        if log_with is not None:
            try:
                log_with(f"safe_call fallback: {e!r}")
            except Exception:  # noqa: BLE001
                pass
        return fallback
