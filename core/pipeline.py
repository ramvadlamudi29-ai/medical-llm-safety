from __future__ import annotations
import ast
import json
import operator as op
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.agent import AgentRouter
from core.cache import LRUCache, make_key
from core.config import settings
from core.llm import LLMClient, LLMResponse, LocalHeuristicLLM, estimate_tokens
from core.logging_setup import get_logger
from core.monitor import metrics, timed
from core.rag import HybridRAG
from core.tracing import new_trace
from security.pii_filter import inspect_text, redact_text, sanitize_input

log = get_logger("pipeline")


# -----------------------------
# Safe calculator
# -----------------------------
_ALLOWED_BIN = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}


class CalculatorError(ValueError):
    pass


class SafeCalculator:
    """Evaluates arithmetic expressions safely. Supports natural-language hints."""

    def evaluate(self, expression: str) -> float:
        if not expression or not isinstance(expression, str):
            raise CalculatorError("empty expression")
        expr = self._extract_expression(expression)
        if not expr:
            raise CalculatorError("no math expression found")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise CalculatorError(f"invalid syntax: {e}") from e
        return float(self._eval(tree.body))

    def _extract_expression(self, text: str) -> str:
        s = text.strip()
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", s, re.I)
        if m:
            a, b = m.group(1), m.group(2)
            return f"({a}/100)*{b}"
        s = re.sub(
            r"^(please\s+)?(calculate|compute|what\s+is|how\s+much\s+is)\s*",
            "",
            s,
            flags=re.I,
        )
        s = s.rstrip("?.! ")
        if re.fullmatch(r"[\d\.\+\-\*\/\%\(\)\s]+", s):
            return s.strip()
        return ""

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise CalculatorError("only numeric constants allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN:
            return _ALLOWED_BIN[type(node.op)](
                self._eval(node.left), self._eval(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](self._eval(node.operand))
        raise CalculatorError(f"disallowed expression: {ast.dump(node)}")


# -----------------------------
# Pipeline response
# -----------------------------
@dataclass
class PipelineResponse:
    answer: str
    provider: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    route: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Main pipeline
# -----------------------------
class AIPlatformPipeline:
    def __init__(
        self,
        rag: HybridRAG | None = None,
        router: AgentRouter | None = None,
        calculator: SafeCalculator | None = None,
        llm: LocalHeuristicLLM | LLMClient | None = None,
        response_cache: LRUCache | None = None,
    ) -> None:
        self.rag = rag or HybridRAG()
        self.router = router or AgentRouter()
        self.calculator = calculator or SafeCalculator()
        self.llm = llm or LocalHeuristicLLM()
        self.client = (
            llm if isinstance(llm, LLMClient) else LLMClient.from_settings()
        )
        self.response_cache = response_cache or LRUCache(
            settings.response_cache_size
        )

    # --- helpers ---
    def _llm_generate(
        self, query: str, contexts: List[str], intent: str, offline_safe: bool
    ) -> LLMResponse:
        if offline_safe:
            return self.llm.generate(query, contexts, intent=intent)  # type: ignore[arg-type]
        return self.client.generate(
            query, contexts, intent=intent, offline_safe=False
        )

    async def run(
        self,
        query: str,
        top_k: int = 3,
        offline_safe: bool = True,
        structured: bool = False,
    ) -> PipelineResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        t0 = time.perf_counter()
        metrics.incr("pipeline.requests")

        with new_trace() as trace:
            # ---- input sanitization ----
            with trace.stage("sanitize"):
                sanitized = sanitize_input(query, max_len=settings.max_input_chars)

            # ---- security: prompt injection / PII inspection ----
            if settings.enable_security_guard:
                with trace.stage("security_guard"):
                    inspection = inspect_text(sanitized)
                if inspection.has_prompt_injection:
                    metrics.incr("security.prompt_injection_blocked")
                    return PipelineResponse(
                        answer="Request blocked: possible prompt injection detected.",
                        provider="security_guard",
                        citations=[],
                        route={"intent": "blocked", "confidence": 1.0},
                        meta={
                            "blocked": True,
                            "reason": "prompt_injection",
                            "trace_id": trace.trace_id,
                            "latency_breakdown_ms": trace.breakdown_ms(),
                        },
                    )

            clean_query = redact_text(sanitized)

            # ---- response cache ----
            cache_enabled = settings.enable_cache
            cache_key = make_key(
                "resp",
                clean_query,
                top_k,
                offline_safe,
                structured,
                settings.model_version,
                settings.prompt_version,
            )
            if cache_enabled:
                with trace.stage("cache_lookup"):
                    cached = self.response_cache.get(cache_key)
                if cached is not None:
                    metrics.incr("cache.response.hit")
                    # Refresh trace metadata for the cached response.
                    cached.meta = {
                        **cached.meta,
                        "cache": "hit",
                        "trace_id": trace.trace_id,
                    }
                    return cached
                metrics.incr("cache.response.miss")

            # ---- routing ----
            with trace.stage("route"):
                route = self.router.route(clean_query)
            route_dict: Dict[str, Any] = {
                "intent": route.intent,
                "confidence": route.confidence,
            }

            # ---- calculator branch ----
            if route.intent == "calculator":
                try:
                    with trace.stage("calculator"):
                        with timed("calculator.eval"):
                            value = self.calculator.evaluate(clean_query)
                    answer = (
                        str(int(value)) if value == int(value) else str(value)
                    )
                    resp = PipelineResponse(
                        answer=answer,
                        provider="calculator",
                        citations=[],
                        route=route_dict,
                        meta={
                            "latency_s": round(time.perf_counter() - t0, 4),
                            "trace_id": trace.trace_id,
                            "latency_breakdown_ms": trace.breakdown_ms(),
                            "cache": "miss",
                            "tokens": {"prompt": 0, "completion": 0},
                            "cost_usd": 0.0,
                        },
                    )
                    if cache_enabled:
                        self.response_cache.set(cache_key, resp)
                    metrics.incr("calculator.ok")
                    metrics.observe_latency(
                        "pipeline.run", time.perf_counter() - t0
                    )
                    return resp
                except CalculatorError:
                    route_dict = {
                        "intent": "search",
                        "confidence": 0.4,
                        "fallback_from": "calculator",
                    }

            # ---- RAG path ----
            intent = (
                route_dict["intent"]
                if route_dict["intent"] in {"search", "summarize"}
                else "search"
            )
            with trace.stage("retrieval"):
                with timed("rag.retrieve"):
                    results = self.rag.retrieve(clean_query, top_k=top_k)
            contexts = [r.document.text for r in results]
            citations = [
                {
                    "id": r.document.id,
                    "title": r.document.title,
                    "score": round(r.score, 4),
                }
                for r in results
            ]

            # Optional rerank stage (fires only if retriever has a reranker
            # and the user asked for one via top-level config).
            if settings.enable_reranker and self.rag.reranker is not None:
                with trace.stage("rerank"):
                    reranked = self.rag.reranker.rerank(
                        clean_query, results, top_k=top_k
                    )
                contexts = [r.document.text for r in reranked]
                citations = [
                    {
                        "id": r.document.id,
                        "title": r.document.title,
                        "score": round(r.score, 4),
                    }
                    for r in reranked
                ]

            with trace.stage("generation"):
                with timed("llm.generate"):
                    llm_resp = self._llm_generate(
                        clean_query,
                        contexts,
                        intent=intent,
                        offline_safe=offline_safe,
                    )

            metrics.add_tokens(
                "llm.total_tokens",
                llm_resp.prompt_tokens + llm_resp.completion_tokens,
            )
            if settings.enable_budget_tracking and llm_resp.cost_usd:
                metrics.add_cost("llm.total_cost", llm_resp.cost_usd)

            answer_text = llm_resp.text
            if structured:
                structured_payload = {
                    "answer": answer_text,
                    "confidence": float(route_dict.get("confidence", 0.5)),
                    "citations": citations,
                }
                answer_text = json.dumps(structured_payload, ensure_ascii=False)

            resp = PipelineResponse(
                answer=answer_text,
                provider=llm_resp.provider,
                citations=citations,
                route=route_dict,
                meta={
                    "offline_safe": offline_safe,
                    "model": llm_resp.model,
                    "model_version": settings.model_version,
                    "prompt_version": settings.prompt_version,
                    "dataset_version": settings.dataset_version,
                    "tokens": {
                        "prompt": llm_resp.prompt_tokens,
                        "completion": llm_resp.completion_tokens,
                    },
                    "cost_usd": round(llm_resp.cost_usd, 6),
                    "latency_s": round(time.perf_counter() - t0, 4),
                    "structured": structured,
                    "trace_id": trace.trace_id,
                    "latency_breakdown_ms": trace.breakdown_ms(),
                    "cache": "miss",
                },
            )
            if cache_enabled:
                self.response_cache.set(cache_key, resp)
            metrics.observe_latency("pipeline.run", time.perf_counter() - t0)
            metrics.incr("pipeline.ok")
            return resp
