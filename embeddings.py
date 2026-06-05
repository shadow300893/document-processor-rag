"""
embeddings.py — Shared embedding utility with disk caching.

Improvement #9: Caching
- Caches embeddings to .cache/ folder using MD5 hash of text
- Repeat queries are near-instant — no Ollama call needed
- Cache persists across runs
"""

import hashlib
import json
import os
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CACHE_DIR = ".cache"


def get_embedding(text: str) -> list[float]:
    """Get embedding for text, using disk cache if available."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = hashlib.md5(text.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # Return cached embedding if exists
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    # Call Ollama and cache result
    response = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    embedding = response.json()["embedding"]

    with open(cache_file, "w") as f:
        json.dump(embedding, f)

    return embedding
