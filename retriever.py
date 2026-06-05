"""
retriever.py — Retrieval module.
"""

import requests
import chromadb
from rank_bm25 import BM25Okapi

from embeddings import get_embedding

CHROMA_PATH = r".\chroma_db"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2"

def call_llm(prompt_text: str) -> str:
    """Call Ollama directly via REST — avoids httpx client lifecycle issues."""
    response = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": 0}
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def rewrite_query(question: str) -> str:
    prompt = f"""You are a search query optimizer.
Rewrite the user's question into a precise, keyword-rich search query for a document retrieval system.
Return ONLY the rewritten query. No explanation, no preamble.

User question: {question}
Rewritten query:"""

    rewritten = call_llm(prompt)
    print(f"  [Query rewritten]: {rewritten}")
    return rewritten

def vector_search(query_text: str, collection, n: int = 10, department: str = None) -> list[dict]:
    embedding = get_embedding(query_text)
    where_filter = {"department": department} if department else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        where=where_filter,
    )

    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

def bm25_search(query_text: str, all_docs: list[dict], n: int = 10) -> list[dict]:
    tokenized_corpus = [doc["text"].lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query_text.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:n]]

def hybrid_search(question: str, department: str = None, top_k: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("documents")

    rewritten = rewrite_query(question)

    all_results = collection.get()
    all_docs = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(all_results["documents"], all_results["metadatas"])
    ]

    vector_results = vector_search(rewritten, collection, n=10, department=department)
    bm25_results = bm25_search(rewritten, all_docs, n=10)

    seen = set()
    merged = []
    for doc in vector_results + bm25_results:
        if doc["text"] not in seen:
            seen.add(doc["text"])
            merged.append(doc)

    if merged:
        q_emb = get_embedding(question)

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x ** 2 for x in a) ** 0.5
            nb = sum(x ** 2 for x in b) ** 0.5
            return dot / (na * nb) if na and nb else 0

        scores = [cosine(q_emb, get_embedding(doc["text"])) for doc in merged]
        ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
        merged = [doc for _, doc in ranked[:top_k]]

    return merged
