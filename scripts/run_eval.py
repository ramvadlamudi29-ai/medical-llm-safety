"""CLI: run the evaluator over data/eval_cases.jsonl and log a run."""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import json
from pathlib import Path

from core.evaluator import EvalCase, Evaluator, load_cases
from core.experiment import log_run
from core.pipeline import AIPlatformPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases", default="data/eval_cases.jsonl", help="Path to JSONL eval cases"
    )
    args = ap.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        # Fallback so the script is always runnable
        cases = [
            EvalCase(
                query="What is the zero trust policy?",
                must_contain=["zero trust"],
                expected_intent="search",
            ),
            EvalCase(
                query="Calculate 6 * 7", must_contain=["42"], expected_intent="calculator"
            ),
        ]

    pipe = AIPlatformPipeline()
    ev = Evaluator()
    results = ev.evaluate(pipe, cases)
    summary = ev.aggregate(results)
    run_id = log_run("eval", {"cases_path": args.cases}, summary)
    print(json.dumps({"run_id": run_id, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
