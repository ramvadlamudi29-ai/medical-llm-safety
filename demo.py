"""End-to-end demo: run 3 queries and print answers, citations, latency."""
from __future__ import annotations
import asyncio
import json
import time

from core.pipeline import AIPlatformPipeline


DEMO_QUERIES = [
    "What is the zero trust policy?",
    "Summarize the incident response guide",
    "Calculate 25% of 1800",
]


async def main() -> None:
    pipe = AIPlatformPipeline()
    for q in DEMO_QUERIES:
        t0 = time.perf_counter()
        resp = await pipe.run(q, offline_safe=True)
        dt = round((time.perf_counter() - t0) * 1000, 2)
        print("=" * 72)
        print(f"Q: {q}")
        print(f"A: {resp.answer}")
        print(
            f"   provider={resp.provider}  "
            f"intent={resp.route.get('intent')}  latency={dt}ms"
        )
        if resp.citations:
            print("   citations:")
            for c in resp.citations:
                print(f"     - [{c['id']}] {c['title']} (score={c['score']})")
        print("   meta:", json.dumps(resp.meta, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
