"""Benchmark the pipeline over N queries.

Prints average latency, p95, success rate, and average eval score.

Usage::

    python scripts/benchmark.py --n 50
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import asyncio
import json
import statistics
import time
from typing import List

from core.evaluator import Evaluator, load_cases
from core.experiment import log_run
from core.pipeline import AIPlatformPipeline


DEFAULT_QUERIES: List[str] = [
    "What is the zero trust policy?",
    "Summarize the incident response guide",
    "Calculate 12 * 11",
    "What is 25% of 1800?",
    "Remote work policy days",
    "What does the code review standard say?",
]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(p * len(s))) - 1))
    return s[idx]


async def _run_once(pipe: AIPlatformPipeline, q: str) -> tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        resp = await pipe.run(q, offline_safe=True)
        ok = bool(resp.answer)
    except Exception:  # noqa: BLE001
        ok = False
    return ok, (time.perf_counter() - t0) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Total queries to run")
    ap.add_argument(
        "--cases",
        default="data/eval_cases.jsonl",
        help="Optional eval JSONL to also score quality",
    )
    args = ap.parse_args()

    pipe = AIPlatformPipeline()
    queries = (DEFAULT_QUERIES * ((args.n // len(DEFAULT_QUERIES)) + 1))[: args.n]
    latencies: List[float] = []
    successes = 0

    async def go() -> None:
        nonlocal successes
        for q in queries:
            ok, ms = await _run_once(pipe, q)
            latencies.append(ms)
            successes += int(ok)

    asyncio.run(go())

    avg = round(statistics.mean(latencies), 3) if latencies else 0.0
    p95 = round(_percentile(latencies, 0.95), 3)
    p50 = round(_percentile(latencies, 0.50), 3)
    success_rate = round(successes / max(1, len(queries)), 4)

    eval_summary = {}
    cases = load_cases(args.cases)
    if cases:
        ev = Evaluator()
        eval_summary = ev.aggregate(ev.evaluate(pipe, cases))

    summary = {
        "n": len(queries),
        "avg_latency_ms": avg,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "success_rate": success_rate,
        "eval": eval_summary,
    }
    run_id = log_run("benchmark", {"n": args.n}, summary)
    print(json.dumps({"run_id": run_id, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
