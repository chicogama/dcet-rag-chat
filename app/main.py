
from fastapi import FastAPI
from app.routers import data, rag

app = FastAPI(
    title="RAG System API",
    description="An API for a RAG system that uses Elasticsearch, Qdrant, and Ollama.",
    version="1.0.0",
)

app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the RAG System API"}
