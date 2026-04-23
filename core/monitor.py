"""Observability: counters, latency histograms, cost tracking."""
from __future__ import annotations
import math
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import Callable, Deque, Dict, Iterator, List


class Metrics:
    def __init__(self, max_samples: int = 1024) -> None:
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._costs: Dict[str, float] = defaultdict(float)
        self._tokens: Dict[str, int] = defaultdict(int)
        self._max_samples = max_samples

    # --- counters ---
    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    # --- latencies ---
    def observe_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            self._latencies[name].append(float(seconds))

    def latency_summary(self, name: str) -> Dict[str, float]:
        with self._lock:
            samples = list(self._latencies.get(name, ()))
        return _summarize(samples)

    # --- cost / tokens ---
    def add_cost(self, name: str, usd: float) -> None:
        with self._lock:
            self._costs[name] += float(usd)

    def add_tokens(self, name: str, count: int) -> None:
        with self._lock:
            self._tokens[name] += int(count)

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            counters = dict(self._counters)
            costs = dict(self._costs)
            tokens = dict(self._tokens)
            latencies = {k: list(v) for k, v in self._latencies.items()}
        latency_summary = {k: _summarize(v) for k, v in latencies.items()}
        return {
            "counters": counters,
            "tokens": tokens,
            "cost_usd": costs,
            "latency_ms": latency_summary,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()
            self._costs.clear()
            self._tokens.clear()


def _summarize(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}
    s = sorted(samples)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(math.ceil(p * n)) - 1))
        return round(s[idx] * 1000, 3)  # ms

    return {
        "count": n,
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "avg": round((sum(s) / n) * 1000, 3),
    }


metrics = Metrics()


# --- timing helpers ---
@contextmanager
def timed(name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        metrics.observe_latency(name, time.perf_counter() - t0)


def time_it(name: str) -> Callable:
    """Decorator that records latency for sync or async callables."""

    def deco(fn: Callable) -> Callable:
        import asyncio

        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def aw(*a, **kw):
                t0 = time.perf_counter()
                try:
                    return await fn(*a, **kw)
                finally:
                    metrics.observe_latency(name, time.perf_counter() - t0)
            return aw

        @wraps(fn)
        def w(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                metrics.observe_latency(name, time.perf_counter() - t0)
        return w

    return deco
