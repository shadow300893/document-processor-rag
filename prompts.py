"""
prompts.py — Prompt engineering variants.

Four techniques:
  baseline       — simple grounded answer
  chain_of_thought — step-by-step reasoning
  few_shot       — example-guided output format
  self_critique  — draft → critique → final answer
"""

BASELINE = """
You are an assistant answering questions from documents.
Use ONLY the context below. If the answer is not in the context, say "I don't know."
Cite the source filename and page when possible.

Previous conversation:
{history}

Context:
{context}

Question: {question}
"""

CHAIN_OF_THOUGHT = """
You are an assistant answering questions from documents.
Use ONLY the context below. Think step by step before answering.

Step 1: Identify the relevant facts from the context.
Step 2: Reason through what they imply.
Step 3: State your final answer clearly, citing source and page.

If the answer is not in the context, say "I don't know."

Previous conversation:
{history}

Context:
{context}

Question: {question}

Step-by-step reasoning:
"""

FEW_SHOT = """
You are a document assistant. Answer using only the provided context.
Cite source and page. Be concise.

Previous conversation:
{history}

Context:
{context}

Examples of good answers:
Q: What is the document's main topic?
A: Based on [filename p.1], the document covers artificial intelligence systems.

Q: Who is the intended audience?
A: According to [filename p.2], the document targets senior software engineers.

Q: {question}
A:
"""

SELF_CRITIQUE = """
You are a careful document assistant. Use ONLY the context below.

Previous conversation:
{history}

Context:
{context}

Question: {question}

Step 1 — Draft answer:
Write an initial answer based on the context.

Step 2 — Critique:
Check your draft. Flag anything that:
- Goes beyond what the context actually says
- Assumes information not present
- Could mislead the reader

Step 3 — Final answer:
Write a corrected, conservative answer that only claims what the context supports.
Cite source and page.
"""

PROMPTS = {
    "baseline": BASELINE,
    "chain_of_thought": CHAIN_OF_THOUGHT,
    "few_shot": FEW_SHOT,
    "self_critique": SELF_CRITIQUE,
}
