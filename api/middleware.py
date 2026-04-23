from __future__ import annotations
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from core.errors import (
    AUTH_ERROR,
    PAYLOAD_TOO_LARGE,
    RATE_LIMITED,
    envelope,
)
from core.monitor import metrics


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Assign / propagate an ``X-Request-Id`` for every request."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        api_keys: list[str],
        protected_prefixes: tuple[str, ...] = ("/query",),
    ):
        super().__init__(app)
        self.api_keys = set(api_keys or [])
        self.protected_prefixes = protected_prefixes

    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(p) for p in self.protected_prefixes):
            key = request.headers.get("x-api-key")
            if not key or key not in self.api_keys:
                metrics.incr("api.unauthorized")
                rid = getattr(request.state, "request_id", None)
                # Keep legacy short shape AND new taxonomy fields.
                body = {"ok": False, "error": "unauthorized"}
                body.update(
                    envelope(
                        AUTH_ERROR, message="missing or invalid api key", trace_id=rid
                    )
                )
                # Preserve back-compat: legacy `error="unauthorized"` wins.
                body["error"] = "unauthorized"
                return JSONResponse(body, status_code=401)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, per_minute: int = 120):
        super().__init__(app)
        self.per_minute = max(1, int(per_minute))
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "anon"
        now = time.time()
        window = self._hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            metrics.incr("api.rate_limited")
            rid = getattr(request.state, "request_id", None)
            body = {"ok": False, "error": "rate_limited"}
            body.update(envelope(RATE_LIMITED, trace_id=rid))
            body["error"] = "rate_limited"
            return JSONResponse(body, status_code=429)
        window.append(now)
        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 64 * 1024):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max_bytes:
            metrics.incr("api.payload_too_large")
            rid = getattr(request.state, "request_id", None)
            body = {"ok": False, "error": "payload_too_large"}
            body.update(envelope(PAYLOAD_TOO_LARGE, trace_id=rid))
            body["error"] = "payload_too_large"
            return JSONResponse(body, status_code=413)
        return await call_next(request)


class TimingMiddleware(BaseHTTPMiddleware):
    """Records per-request latency and adds an X-Process-Time header."""

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        metrics.observe_latency(f"http.{request.method}.{request.url.path}", dt)
        response.headers["X-Process-Time"] = f"{dt*1000:.2f}ms"
        return response
