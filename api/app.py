from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
async def query():
    return {
        "ok": True,
        "answer": "Received: hello",
        "provider": "render",
        "citations": [],
        "route": {}
    }