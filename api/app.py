from fastapi import FastAPI

app = FastAPI()

# ✅ Safe import (prevents startup crash)
try:
    from core.pipeline import run_pipeline
    PIPELINE_AVAILABLE = True
except Exception as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(e)


@app.get("/")
def root():
    return {
        "message": "AI Platform API running 🚀",
        "pipeline": "loaded" if PIPELINE_AVAILABLE else "failed",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(body: dict):
    query_text = body.get("query", "")

    # 🚨 If pipeline failed → show real error instead of crash
    if not PIPELINE_AVAILABLE:
        return {
            "ok": True,
            "answer": f"[PIPELINE LOAD ERROR] {PIPELINE_ERROR}",
            "provider": "debug",
            "citations": [],
            "route": {}
        }

    try:
        result = run_pipeline(query_text)
        return result
    except Exception as e:
        return {
            "ok": True,
            "answer": f"[PIPELINE RUNTIME ERROR] {str(e)}",
            "provider": "debug",
            "citations": [],
            "route": {}
        }