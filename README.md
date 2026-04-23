# AI Platform v4.1

A compact, **production-grade** retrieval-augmented AI service with hybrid
search, agent routing, observability, evaluation, caching, cost tracking,
structured output, security hardening, and an optional UI — all runnable
fully offline.

> **Use case.** A self-contained "Ask your internal docs" service for small
> teams: the API answers natural-language questions over a private corpus
> of policies/runbooks, routes math questions to a safe calculator, and
> exposes everything DevOps needs to ship it (metrics, eval, traces,
> versioning, CI).

## ✨ Features

| Area | What you get |
|------|--------------|
| **Retrieval** | `HybridRAG` (BM25 + TF-IDF) with optional `CrossEncoderReranker` (graceful lexical fallback). |
| **Routing** | `AgentRouter` dispatches to `calculator`, `summarize`, or `search`. |
| **LLM layer** | `LLMClient` with **OpenAI → Ollama → LocalHeuristic** fallback chain (offline-safe). |
| **Prompts** | Versioned templates in `core/prompts.py` (`search@v1`, `summarize@v1`, …). |
| **Security** | PII detection/redaction (email, SSN, credit-card via Luhn), prompt-injection guard, control-char & length sanitization. |
| **Observability** | Structured JSON logs, p50/p95/p99 latency histograms, **per-stage latency breakdown** (sanitize / security / route / retrieval / rerank / generation / total), per-request **trace IDs** (`X-Request-Id`). |
| **Caching** | Thread-safe LRU response cache + embedding cache, toggled by `ENABLE_CACHE`. |
| **Cost** | Token + USD estimates per request; aggregate counters in `/metrics`. |
| **Eval** | Relevance / faithfulness / intent accuracy, JSONL dataset, CLI runner that writes per-case results. |
| **Experiments** | Every run appends to `experiments/runs.jsonl` with `run_id`, params, metrics, versions. |
| **Errors** | Stable taxonomy (`validation_error`, `auth_error`, `rate_limited`, `payload_too_large`, `timeout_error`, `internal_error`, `security_blocked`). |
| **Resilience** | Retry/timeout helpers (`core/retry.py`), graceful fallback everywhere — the system never crashes. |
| **API** | `/health`, `/version`, `/metrics`, `/query`, `/eval`. |
| **CI** | GitHub Actions: install → test → eval → benchmark. |

## 🗺️ Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full ASCII diagram. TL;DR:

```
client → FastAPI (TraceId·APIKey·SizeLimit·RateLimit·Timing)
       → Pipeline (sanitize → guard → cache → route → calc | RAG → rerank? → LLM)
       → Monitor + Experiment log
```

## 🚀 Run it

```bash
pip install -r requirements.txt

# 1) Tests
python -m pytest -q                  # 26 passing

# 2) Demo (3 sample queries, offline)
python demo.py

# 3) API
python -m uvicorn api.app:app --reload
#   POST /query     — main entry point
#   GET  /health    — versions + feature flags
#   GET  /version   — model/prompt/dataset versions
#   GET  /metrics   — counters, latencies, tokens, cost, cache stats
#   POST /eval      — run the eval suite over the API

# 4) Evaluation
python scripts/evaluate.py --cases data/eval_cases.jsonl

# 5) Benchmark
python scripts/benchmark.py --n 50

# 6) Reindex a folder of docs
python scripts/reindex.py ./data --query "zero trust"

# 7) Optional UI
pip install streamlit
streamlit run ui/streamlit_app.py
```

## 📡 API example

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -H 'x-api-key: test-key' \
  -d '{"query": "What is the zero trust policy?", "top_k": 3, "offline_safe": true}' | jq
```

Sample response (offline mode):

```json
{
  "ok": true,
  "answer": "Zero trust security policy requires continuous verification ...",
  "provider": "local_heuristic",
  "citations": [
    {"id": "sec-001", "title": "Zero Trust Security Policy", "score": 5.42}
  ],
  "route": {"intent": "search", "confidence": 0.5},
  "meta": {
    "model_version": "v1.0.0",
    "prompt_version": "v1",
    "dataset_version": "v1",
    "tokens": {"prompt": 0, "completion": 0},
    "cost_usd": 0.0,
    "latency_s": 0.0021,
    "trace_id": "f4a1c0e9d8b7",
    "latency_breakdown_ms": {
      "sanitize": 0.04, "security_guard": 0.06, "cache_lookup": 0.01,
      "route": 0.03, "retrieval": 0.71, "generation": 0.12, "total": 1.91
    },
    "request_id": "f4a1c0e9d8b7"
  }
}
```

Calculator route:

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'content-type: application/json' -H 'x-api-key: test-key' \
  -d '{"query": "Calculate 25% of 1800"}' | jq
# → {"ok": true, "answer": "450", "provider": "calculator", ...}
```

Errors return a stable envelope (legacy `error` field preserved):

```json
{
  "ok": false,
  "error": "validation_error",
  "details": [...],
  "message": "validation_error",
  "trace_id": "9b3a..."
}
```

## 📊 Metrics & evaluation snapshot

Sample eval summary on the bundled `data/eval_cases.jsonl` (5 cases,
offline-safe):

| Metric | Value |
|---|---|
| pass_rate | 1.0 |
| avg_relevance | 0.18 |
| avg_faithfulness | 0.31 |
| intent_accuracy | 1.0 |
| avg_latency_ms | ~2 ms |
| p95_latency_ms | ~4 ms |

Sample benchmark (`--n 50`, offline-safe): avg ≈ 1.5 ms / req, p95 ≈ 3 ms,
success_rate = 1.0 on a single core.

## 🔐 Security

- **API key** required for `/query` and `/eval` (`x-api-key` header).
- **Input sanitization**: control-char strip, whitespace normalize, length cap.
- **PII**: email / SSN / credit-card (Luhn-verified) detection & redaction.
- **Prompt injection**: heuristic patterns block obvious overrides
  (e.g. "ignore previous instructions") and short-circuit the pipeline.
- **Body size & rate limits** enforced as middleware.

## 🧱 Project tree

```
ai-platform/
├── api/                  # FastAPI app + middleware
├── core/                 # Pipeline, RAG, LLM, eval, monitor, cache, errors,
│                         # retry, tracing, prompts, ingest, schemas, config
├── security/             # PII filter + injection guard
├── data/                 # eval_cases.jsonl + sample.jsonl
├── scripts/              # ingest, reindex, evaluate, run_eval, benchmark
├── experiments/          # JSONL run logs (generated)
├── tests/                # pytest suite (26 tests)
├── ui/                   # Optional Streamlit dashboard
├── demo.py               # 3-query demo
├── ARCHITECTURE.md       # ASCII architecture diagram
├── .env.example
├── .github/workflows/ci.yml
└── requirements.txt
```

## ⚙️ Configuration

All settings come from env vars (or `.env`). See `.env.example` for the
exhaustive list. Notable feature flags:

- `ENABLE_CACHE` (default `true`)
- `ENABLE_EVAL` (default `true`)
- `ENABLE_SECURITY_GUARD` (default `true`)
- `ENABLE_RERANKER` (default `false`)
- `DEBUG_MODE` (default `false`)

## 🧪 Tests

```bash
python -m pytest -q
# 26 passed
```

The suite covers retrieval, pipeline, agent routing, calculator safety,
PII filter, and the API endpoint.

## 📦 Limitations & future work

- The bundled retriever is intentionally tiny and in-memory; for >100k
  docs swap in FAISS / pgvector behind the same `HybridRAG` interface.
- The cross-encoder reranker is optional and downloads a model on first
  use; without it the lexical fallback is used so tests stay offline.
- `LLMClient` ships HTTP clients for OpenAI and Ollama. The local
  heuristic is deterministic but not generative — plug a real model in
  by setting `OPENAI_API_KEY` or `OLLAMA_BASE_URL`.
- Cost figures are estimates; for production accounting feed real usage
  back into `metrics.add_cost` from your provider's billing webhook.
- No persistent vector store, no auth beyond API keys, no per-tenant
  isolation — these are intentional scope cuts for a portfolio system.
