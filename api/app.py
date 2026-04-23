from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Platform API running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
async def query():
    return {
        "ok": True,
        "answer": "API working",
        "provider": "render",
        "citations": [],
        "route": {}
    }