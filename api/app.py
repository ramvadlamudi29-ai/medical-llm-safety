from fastapi import FastAPI
from core.pipeline import run_pipeline

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Platform API running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
async def query(body: dict):
    query_text = body.get("query", "")

    try:
        result = run_pipeline(query_text)
        return result
    except Exception as e:
        return {
            "ok": True,
            "answer": f"[PIPELINE ERROR] {str(e)}",
            "provider": "debug",
            "citations": [],
            "route": {}
        }