"""CLI: rebuild a HybridRAG index from a directory or files.

This is a convenience wrapper around ``core.ingest.build_rag``. The current
runtime keeps an in-memory index per process, so this script demonstrates
ingestion + retrieval and prints index stats.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import json

from core.ingest import build_rag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Files / directories to (re)index")
    ap.add_argument("--query", default="", help="Optional smoke query")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    rag = build_rag(args.paths)
    out = {"docs_indexed": len(rag.documents)}
    if args.query:
        out["results"] = [
            {
                "id": r.document.id,
                "title": r.document.title,
                "score": round(r.score, 4),
            }
            for r in rag.retrieve(args.query, top_k=args.top_k)
        ]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
