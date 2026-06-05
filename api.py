"""
api.py — FastAPI wrapper for the RAG pipeline.

- POST /query        — ask a question, get a grounded answer
- GET  /docs-list    — list all ingested documents
- GET  /health       — check Ollama is running
- Auto-generated interactive UI at http://localhost:8000/docs

Usage:
    py -m uvicorn api:app --reload
    Then open http://localhost:8000/docs in your browser
"""

import requests
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from query import query as rag_query
from agent import agent_query

app = FastAPI(
    title="Local RAG API",
    description="Built by Sahil Gheek — zero-cost local RAG pipeline",
    version="1.0.0",
)

# Allow browser requests from any origin (for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session_memories: dict[str, deque] = {}

CHROMA_PATH = r".\chroma_db"


# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    technique: str = "chain_of_thought"
    department: str = None
    session_id: str = "default"

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the main recommendations?",
                "technique": "chain_of_thought",
                "department": None,
                "session_id": "user_123",
            }
        }
    }


class QueryResponse(BaseModel):
    question: str
    answer: str
    technique: str
    department: str | None
    session_id: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if Ollama is running and reachable."""
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        ollama_status = "running" if r.status_code == 200 else "unreachable"
    except Exception:
        ollama_status = "unreachable"

    return {
        "status": "ok",
        "ollama": ollama_status,
    }


@app.post("/query", response_model=QueryResponse)
def ask(req: QueryRequest):
    """
    Ask a question against your ingested documents.

    - **question**: natural language question
    - **technique**: baseline | chain_of_thought | few_shot | self_critique
    - **department**: optional filter (matches filename prefix e.g. 'hr')
    - **session_id**: used to maintain conversation memory per user
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieve or create session memory
    if req.session_id not in session_memories:
        session_memories[req.session_id] = deque(maxlen=10)
    history = session_memories[req.session_id]

    answer = rag_query(
        question=req.question,
        history=history,
        technique=req.technique,
        department=req.department,
    )

    # Update memory
    history.append(req.question)
    history.append(answer)

    return QueryResponse(
        question=req.question,
        answer=answer,
        technique=req.technique,
        department=req.department,
        session_id=req.session_id,
    )


@app.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in session_memories:
        session_memories[session_id].clear()
    return {"message": f"Memory cleared for session {session_id}"}


@app.get("/docs-list")
def list_documents():
    """List all documents currently ingested in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection("documents")
        results = collection.get()
        sources = list({m["source"] for m in results["metadatas"]})
        return {
            "total_chunks": len(results["ids"]),
            "documents": sorted(sources),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))