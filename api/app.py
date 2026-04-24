from fastapi import FastAPI
from typing import Any, Dict

app = FastAPI()

# -----------------------------
# Safe pipeline import
# -----------------------------
try:
    from core.pipeline import AIPlatformPipeline

    pipeline = AIPlatformPipeline()
    PIPELINE_AVAILABLE = True
    PIPELINE_ERROR = None
except Exception as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(e)


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "AI Platform API running 🚀",
        "pipeline": "loaded" if PIPELINE_AVAILABLE else "failed",
        "error": PIPELINE_ERROR
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(body: Dict[str, Any]):
    query_text = body.get("query", "")

    # ❌ If pipeline failed to load
    if not PIPELINE_AVAILABLE:
        return {
            "ok": True,
            "answer": f"[PIPELINE LOAD ERROR] {PIPELINE_ERROR}",
            "provider": "debug",
            "citations": [],
            "route": {}
        }

    try:
        # ✅ Call async pipeline
        result = await pipeline.run(
            query=query_text,
            top_k=body.get("top_k", 3),
            offline_safe=body.get("offline_safe", True),
            structured=body.get("structured", False),
        )

        return {
            "ok": True,
            "answer": result.answer,
            "provider": result.provider,
            "citations": result.citations,
            "route": result.route,
            "meta": result.meta,
        }

    except Exception as e:
        return {
            "ok": True,
            "answer": f"[PIPELINE RUNTIME ERROR] {str(e)}",
            "provider": "debug",
            "citations": [],
            "route": {}
        }