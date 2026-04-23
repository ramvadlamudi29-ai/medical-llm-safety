"""Run evaluation and persist per-case results to JSONL.

This is a richer alternative to ``scripts/run_eval.py`` that also writes a
``experiments/eval_<run_id>.jsonl`` file with per-case detail.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import json
from pathlib import Path

from core.config import settings
from core.evaluator import EvalCase, Evaluator, load_cases
from core.experiment import log_run
from core.pipeline import AIPlatformPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/eval_cases.jsonl")
    ap.add_argument("--out-dir", default=settings.experiments_dir)
    args = ap.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        cases = [
            EvalCase(
                query="What is the zero trust policy?",
                must_contain=["zero trust"],
                expected_intent="search",
            ),
            EvalCase(
                query="Calculate 6 * 7",
                must_contain=["42"],
                expected_intent="calculator",
            ),
        ]

    pipe = AIPlatformPipeline()
    ev = Evaluator()
    results = ev.evaluate(pipe, cases)
    summary = ev.aggregate(results)
    run_id = log_run(
        "eval", {"cases_path": args.cases, "n": len(cases)}, summary
    )
    out_path = Path(args.out_dir) / f"eval_{run_id}.jsonl"
    ev.save_jsonl(results, out_path)
    print(json.dumps(
        {"run_id": run_id, "results_path": str(out_path), "summary": summary},
        indent=2,
    ))


if __name__ == "__main__":
    main()
