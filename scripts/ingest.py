"""CLI: ingest a directory or file into a HybridRAG index (in-memory demo)."""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import json

from core.ingest import build_rag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Files or directories to ingest")
    ap.add_argument("--query", default="", help="Optional sample query")
    args = ap.parse_args()

    rag = build_rag(args.paths)
    print(json.dumps({"docs_indexed": len(rag.documents)}))
    if args.query:
        for r in rag.retrieve(args.query, top_k=5):
            print(
                json.dumps(
                    {"id": r.document.id, "title": r.document.title, "score": r.score}
                )
            )


if __name__ == "__main__":
    main()
