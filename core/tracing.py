"""Trace IDs + per-stage latency breakdown using contextvars.

Usage in pipeline::

    with new_trace() as trace:
        with trace.stage("retrieval"):
            ...
        with trace.stage("generation"):
            ...
        breakdown = trace.breakdown_ms()  # {"retrieval": 12.3, ...}
        trace_id = trace.trace_id
"""
from __future__ import annotations
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


_current: ContextVar[Optional["Trace"]] = ContextVar("trace", default=None)


@dataclass
class Trace:
    trace_id: str
    stages: Dict[str, float] = field(default_factory=dict)  # seconds
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            # Multiple calls accumulate (e.g. several retrieval rounds).
            self.stages[name] = self.stages.get(name, 0.0) + dt

    def total_s(self) -> float:
        return time.perf_counter() - self.started

    def breakdown_ms(self) -> Dict[str, float]:
        out = {k: round(v * 1000, 3) for k, v in self.stages.items()}
        out["total"] = round(self.total_s() * 1000, 3)
        return out


@contextmanager
def new_trace(trace_id: Optional[str] = None) -> Iterator[Trace]:
    t = Trace(trace_id=trace_id or uuid.uuid4().hex[:16])
    token = _current.set(t)
    try:
        yield t
    finally:
        _current.reset(token)


def current_trace() -> Optional[Trace]:
    return _current.get()


def current_trace_id() -> Optional[str]:
    t = _current.get()
    return t.trace_id if t else None
