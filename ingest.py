"""
ingest.py

Usage:
    py ingest.py
"""

import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embedding

DOCS_PATH = r".\docs"
CHROMA_PATH = r".\chroma_db"


def ingest():
    all_chunks = []
    all_ids = []
    all_embeddings = []
    all_metadatas = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len,
    )

    for filename in os.listdir(DOCS_PATH):
        filepath = os.path.join(DOCS_PATH, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            filetype = "txt"
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            filetype = "pdf"
        else:
            continue

        docs = loader.load()
        chunks = splitter.split_documents(docs)
        print(f"  {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{i}"

            embedding = get_embedding(chunk.page_content)

            all_chunks.append(chunk.page_content)
            all_ids.append(chunk_id)
            all_embeddings.append(embedding)

            all_metadatas.append({
                "source": filename,
                "page": str(chunk.metadata.get("page", 0)),
                "filetype": filetype,
                "department": get_department(filename),
            })

    print(f"\nEmbedding complete. Storing {len(all_chunks)} chunks...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to avoid duplicates on re-ingest
    try:
        client.delete_collection("documents")
    except Exception:
        pass

    collection = client.get_or_create_collection("documents")
    collection.upsert(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )
    print(f"Done. {len(all_chunks)} vectors stored in ChromaDB.")

def get_department(filename):
    """
    Extract department from filename prefix if it exists.
    Examples:
      hr_policy.txt        -> "hr"
      engineering_runbook  -> "engineering"
      test.txt             -> "general"
      random_file.txt      -> "general" (only known prefixes count)
    """
    known_departments = ["hr", "engineering", "finance", "ai", "general", "legal", "product", "sales"]
    prefix = filename.split("_")[0].lower()
    return prefix if prefix in known_departments else "general"


if __name__ == "__main__":
    ingest()
