# 🤖 AI Platform v4.1 — RAG + LLM System

## 👨‍💻 About This Project

I built this system to solve a real problem I faced while working with LLMs:

➡️ **Hallucination and unreliable answers**

To address this, I designed a **Retrieval-Augmented Generation (RAG) platform** that:
- Grounds responses using document retrieval
- Routes queries intelligently (search / calculator / summarize)
- Applies evaluation + retry logic to improve reliability

This project represents my transition from **prompt-based LLM usage → system-level AI engineering**.

---

## 🚀 Core Capabilities

- 📂 Document ingestion → chunking & indexing  
- 🔍 Hybrid retrieval (BM25 + TF-IDF fallback)  
- 🧠 Context-aware LLM responses (Ollama / OpenAI fallback)  
- 🔀 Intelligent routing (search / calculator / summarize)  
- 🛡️ Hallucination control via strict prompting  
- 📊 Evaluation + retry pipeline  
- ⚡ FastAPI backend + optional Streamlit UI  
- 📦 Offline-safe execution  

---

## 🧠 My Contribution

- Designed end-to-end RAG pipeline (retrieval → rerank → LLM → evaluation)  
- Implemented hybrid search system (BM25 + TF-IDF)  
- Built secure input pipeline (PII filtering + prompt injection guard)  
- Added observability (latency, traces, cost tracking)  
- Developed fallback LLM chain (OpenAI → Ollama → Local heuristic)  
- Ensured system reliability with caching, retries, and error handling  

---

## 🏗️ Architecture (Simplified)


Client
↓
FastAPI API Layer
↓
Pipeline
├── Input Sanitization + Security Guard
├── Cache Lookup
├── Agent Router (search / calculator / summarize)
├── RAG Retrieval (Hybrid Search)
├── Optional Reranking
├── LLM Generation
└── Evaluation + Retry
↓
Response + Metrics + Trace


---

## 🛠️ Tech Stack

- **Backend:** FastAPI  
- **UI:** Streamlit (optional)  
- **Retrieval:** BM25 + TF-IDF  
- **Vector / Search:** FAISS-style logic  
- **LLM:** Ollama / OpenAI / Local fallback  
- **Embeddings:** Sentence Transformers  
- **Monitoring:** Custom metrics + tracing  

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
Start API
uvicorn api.app:app --reload
Run Demo
python demo.py
Run Tests
pytest -q
Optional UI
pip install streamlit
streamlit run ui/streamlit_app.py
📡 API Usage
Query
curl -X POST http://localhost:8000/query \
-H "content-type: application/json" \
-H "x-api-key: test-key" \
-d '{"query": "What is zero trust?", "top_k": 3}'
Example Response
{
  "ok": true,
  "answer": "Zero trust requires continuous verification...",
  "provider": "local_heuristic",
  "route": {"intent": "search"},
  "meta": {
    "latency_s": 0.002,
    "tokens": {"prompt": 0, "completion": 0},
    "trace_id": "abc123"
  }
}
🔐 Security Features
API key authentication
Input sanitization (length + control chars)
PII detection (email, SSN, credit card)
Prompt injection protection
Rate limiting + request size limits
📊 Evaluation & Metrics
Relevance scoring
Faithfulness tracking
Intent accuracy
Latency metrics (p50 / p95 / p99)
Token + cost estimation
📁 Project Structure
ai-platform/
├── api/            # FastAPI app
├── core/           # pipeline, rag, llm, evaluator, etc.
├── security/       # guards and filters
├── data/           # datasets
├── scripts/        # ingestion, eval, benchmark
├── tests/          # test cases
├── ui/             # Streamlit UI
├── experiments/    # run logs
├── README.md
└── requirements.txt
⚠️ Limitations
In-memory retrieval (not optimized for large-scale data)
No user authentication (API key only)
Local heuristic fallback is not fully generative
Designed as a portfolio + learning system, not enterprise deployment
🎯 Key Outcome

Built a production-style AI system that moves beyond prompt engineering into:

reliable LLM outputs
grounded responses
system-level AI design
🚀 Future Improvements
Replace retrieval with FAISS / pgvector for scale
Add authentication + multi-user support
Deploy to cloud (Render / AWS / GCP)
Integrate real LLM APIs for production
💼 Author

AI Engineer (RAG + LLM Systems)
Focused on building reliable, real-world AI systems beyond prompt engineering
