"""Evaluation: relevance + faithfulness + must-contain checks."""
from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from core.rag import tokenize


@dataclass
class EvalCase:
    query: str
    must_contain: List[str] = field(default_factory=list)
    expected_intent: Optional[str] = None


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    answer: str
    relevance: float = 0.0
    faithfulness: float = 0.0
    intent_ok: bool = True
    provider: str = ""
    latency_ms: float = 0.0


def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def relevance_score(query: str, answer: str) -> float:
    return round(_overlap(query, answer), 4)


def faithfulness_score(answer: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
    joined = " ".join(contexts)
    return round(_overlap(answer, joined), 4)


class Evaluator:
    """Runs an offline evaluation. Compatible with the existing test suite."""

    def evaluate(self, pipeline, cases: Iterable[EvalCase]) -> List[EvalResult]:
        import asyncio

        results: List[EvalResult] = []
        for case in cases:
            t0 = time.perf_counter()
            resp = asyncio.run(pipeline.run(case.query, offline_safe=True))
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            ans = resp.answer or ""
            ans_l = ans.lower()
            ok = all(s.lower() in ans_l for s in case.must_contain)
            ctx_texts = [c.get("title", "") for c in resp.citations]
            rel = relevance_score(case.query, ans)
            faith = faithfulness_score(ans, ctx_texts) if ctx_texts else 0.0
            intent_ok = (
                case.expected_intent is None
                or resp.route.get("intent") == case.expected_intent
            )
            results.append(
                EvalResult(
                    case=case,
                    passed=ok and intent_ok,
                    answer=ans,
                    relevance=rel,
                    faithfulness=faith,
                    intent_ok=intent_ok,
                    provider=resp.provider,
                    latency_ms=latency_ms,
                )
            )
        return results

    def pass_rate(self, results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.passed) / len(results)

    def aggregate(self, results: List[EvalResult]) -> dict:
        n = len(results) or 1
        latencies = sorted(r.latency_ms for r in results) or [0.0]
        p95_idx = max(0, int(0.95 * len(latencies)) - 1)
        return {
            "n": len(results),
            "pass_rate": self.pass_rate(results),
            "avg_relevance": round(sum(r.relevance for r in results) / n, 4),
            "avg_faithfulness": round(sum(r.faithfulness for r in results) / n, 4),
            "intent_accuracy": round(
                sum(1 for r in results if r.intent_ok) / n, 4
            ),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            "p95_latency_ms": latencies[p95_idx],
        }

    def save_jsonl(self, results: List[EvalResult], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(
                    json.dumps(
                        {
                            "query": r.case.query,
                            "expected_intent": r.case.expected_intent,
                            "must_contain": r.case.must_contain,
                            "passed": r.passed,
                            "answer": r.answer,
                            "relevance": r.relevance,
                            "faithfulness": r.faithfulness,
                            "intent_ok": r.intent_ok,
                            "provider": r.provider,
                            "latency_ms": r.latency_ms,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return out


def load_cases(path: str | Path) -> List[EvalCase]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[EvalCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out.append(
            EvalCase(
                query=obj["query"],
                must_contain=list(obj.get("must_contain", [])),
                expected_intent=obj.get("expected_intent"),
            )
        )
    return out
