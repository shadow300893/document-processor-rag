# Setup Guide

This is a local RAG system that runs entirely on your machine — no API keys, no cloud, no cost. Built and tested on Windows using PowerShell.

---

## Getting the code

```powershell
git clone https://github.com/shadow300893/document-processor-rag.git
cd document-processor-rag
mkdir docs
```

The `docs/` folder is where you put your own PDF or TXT files. It's gitignored so nothing in there gets committed.

---

## Step 1 — Install Python

Download from **python.org/downloads** and run the installer.

One thing that gets everyone: there's a checkbox at the bottom of the first screen that says **"Add Python to PATH"**. Tick that before clicking Install. If you miss it you'll get `py not recognised` errors and have to reinstall.

Check it worked:
```powershell
py --version
```

If pip is missing (happens on some Python installs):
```powershell
py -m ensurepip --upgrade
```

---

## Step 2 — Install Ollama

Download from **ollama.com/download/windows** and run the installer. It installs silently — no prompts, just wait a few seconds.

After installing, close and reopen PowerShell, then check:
```powershell
ollama --version
```

If it says `ollama not recognised`, the installer didn't add it to PATH. Fix it:
```powershell
# For the current session
$env:Path += ";C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama"

# Permanently (needs a new PowerShell window to take effect)
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path","User") + ";C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama", "User")
```

If PATH still doesn't work, you can call Ollama directly by full path — this works fine for pulling models:
```powershell
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull llama3.2
```

---

## Step 3 — Pull the models

These are the two models the system needs. Both run locally.

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

`llama3.2` is about 2GB, `nomic-embed-text` is around 274MB. Wait for both to finish before moving on.

---

## Step 4 — Virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You'll see `(.venv)` at the start of your prompt when it's active. You need this active every time you run the project.

If activation is blocked:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 5 — Install dependencies

```powershell
py -m pip install -r requirements.txt
```

Takes 3-5 minutes on first run. If any package fails, install it individually:
```powershell
py -m pip install <package-name>
```

---

## Step 6 — Start Ollama server

Ollama needs to be running in the background whenever you use the project. Open a **separate PowerShell window** and run:

```powershell
ollama serve
```

Leave that window open. You should see:
```
Ollama is running on http://127.0.0.1:11434
```

Quick check to confirm it's reachable:
```powershell
Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing
```

Should return `StatusCode: 200`. If it doesn't, the server isn't running — go back to the separate window.

---

## Step 7 — Add your documents

Drop PDF or TXT files into the `docs/` folder.

If you want department filtering to work, prefix filenames with the department name:
```
hr_policy.pdf            → filters as "hr"
engineering_runbook.txt  → filters as "engineering"
finance_policy.pdf       → filters as "finance"
report.pdf               → falls back to "general"
```

Files without a recognised prefix automatically get tagged as `general` — they still get ingested and searched, just not filterable by department.

No documents yet? Create a quick test file:
```powershell
Set-Content -Path ".\docs\test.txt" -Value "This is a test document about artificial intelligence."
```

---

## Step 8 — Ingest documents

```powershell
py ingest.py
```

This reads everything in `docs/`, splits it into chunks, embeds each chunk using nomic-embed-text, and stores the vectors in ChromaDB. Run this once at the start and again whenever you add or change documents.

Expected output:
```
  hr_policy.txt: 6 chunks
  engineering_runbook.txt: 8 chunks
Embedding complete. Storing 14 chunks...
Done. 14 vectors stored in ChromaDB.
```

Two folders get created automatically: `chroma_db/` for the vectors and `.cache/` for cached embeddings. Both are gitignored.

---

## Running the project

### Chat with your documents
```powershell
# Default (chain_of_thought prompt)
py query.py

# Step-by-step reasoning — best for complex questions
py query.py --technique chain_of_thought

# Only search HR documents
py query.py --department hr

# Combine both
py query.py --technique chain_of_thought --department engineering
```

Type `clear` to reset the conversation memory. Type `q` to quit.

### API
```powershell
py -m uvicorn api:app --reload
```

Open **http://localhost:8000/docs** in your browser. You get an interactive UI where you can test all endpoints without writing any code. Useful for demos.

---

## Troubleshooting

**`ollama` not recognised after install**
The PATH wasn't set. Run the PATH fix in Step 2, then open a new PowerShell window.

**Empty embeddings error / ChromaDB crash**
Ollama server is not running. Go to your separate window and run `ollama serve`.

**SSL certificate error when downloading models**
Usually a corporate network issue. Try on a different network or hotspot.

**`No module named X`**
The venv isn't active. Run `.\.venv\Scripts\Activate.ps1` first.

**Re-ingesting after changing documents**
```powershell
Remove-Item -Recurse -Force .\chroma_db
py ingest.py
```

**Clearing the embedding cache**
```powershell
Remove-Item -Recurse -Force .\.cache
```

---

## Project structure

```
document-processor-rag/
├── docs/           ← your documents go here (gitignored)
├── chroma_db/      ← auto-created vector store (gitignored)
├── .cache/         ← auto-created embedding cache (gitignored)
├── embeddings.py   ← ollama embedding calls with disk cache
├── ingest.py       ← document loading, chunking, indexing
├── retriever.py    ← hybrid search + query rewriting + reranking
├── prompts.py      ← four prompt engineering variants
├── query.py        ← interactive CLI with memory
├── api.py          ← fastapi wrapper
└── requirements.txt
```

---

## Hardware

Runs on CPU — no GPU needed. Minimum 8GB RAM, 16GB recommended. About 5GB of disk space for the models.

Inference is slow on CPU — expect 10-30 seconds per response depending on your machine. A GPU cuts that to 1-3 seconds but is not required.