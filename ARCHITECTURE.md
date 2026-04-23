# Architecture

```
                ┌────────────────────────────────────────────┐
                │                  Client                    │
                └───────────────┬────────────────────────────┘
                                │  HTTPS  (X-API-Key, JSON)
                                ▼
        ┌───────────────────────────────────────────────────────┐
        │                   FastAPI app (api/)                  │
        │  ┌───────────────────────────────────────────────┐    │
        │  │ Middleware stack (outer → inner)              │    │
        │  │  TraceId  →  APIKey  →  RequestSizeLimit  →   │    │
        │  │  RateLimit  →  Timing                          │   │
        │  └───────────────────────────────────────────────┘    │
        │  Routes: /health  /version  /metrics  /query  /eval   │
        └─────────────────────┬─────────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────────────────┐
        │            AIPlatformPipeline (core/)                 │
        │                                                       │
        │  sanitize → security_guard → cache_lookup → route     │
        │       │                                       │       │
        │       │            ┌──────────────────────────┘       │
        │       ▼            ▼                                  │
        │  Calculator   AgentRouter ── intent ──┐               │
        │                                       ▼               │
        │                              HybridRAG (BM25+TFIDF)   │
        │                                       │               │
        │                       (optional CrossEncoder rerank)  │
        │                                       ▼               │
        │                                 LLMClient             │
        │                  OpenAI → Ollama → LocalHeuristic     │
        │                                       │               │
        │                                       ▼               │
        │   Response: answer, citations, route, meta {trace_id, │
        │             latency_breakdown_ms, tokens, cost_usd}   │
        └───────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────────────────┐
        │   Cross-cutting: monitor (counters/p50/p95/p99,       │
        │   tokens, cost) · structured JSON logs · experiment   │
        │   tracking (experiments/runs.jsonl) · LRU caches      │
        └───────────────────────────────────────────────────────┘
```

## Request lifecycle

1. **TraceIdMiddleware** assigns/propagates `X-Request-Id`.
2. **APIKeyMiddleware** rejects `/query`/`/eval` calls without a valid key.
3. **RequestSizeLimit / RateLimit / Timing** middlewares enforce safety nets.
4. **Pipeline** sanitizes input, runs the security guard, checks the LRU
   cache, routes the intent, dispatches to calculator or RAG, optionally
   reranks, calls the LLM (with provider fallback), and returns a
   `PipelineResponse` whose `meta` carries `trace_id`,
   `latency_breakdown_ms` (sanitize / security / route / retrieval / rerank
   / generation / total), tokens, and `cost_usd`.
5. **Monitor** records counters, latency histograms, tokens, and cost.

## Failure handling

- LLM provider chain falls back automatically (OpenAI → Ollama → local).
- `core/retry.py` provides `retry`, `with_fallback`, `safe_call`, and a
  `timeout(seconds)` decorator.
- All API errors return a stable envelope:
  `{ok: false, error: <code>, message, details?, trace_id?}` while
  preserving the legacy `error` shape used by existing clients.

## Versioning

`/version` and every `meta` block expose `model_version`, `prompt_version`,
and `dataset_version` so experiment logs in `experiments/runs.jsonl` are
fully reproducible.
