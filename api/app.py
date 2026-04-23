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

    result = run_pipeline(query_text)

    return result