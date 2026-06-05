import argparse
import requests
from collections import deque
from langchain_core.prompts import ChatPromptTemplate
from retriever import hybrid_search
from prompts import PROMPTS

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2"


def format_context(chunks):
    return "\n\n---\n".join(
        f"[{c['metadata']['source']} p.{c['metadata']['page']}]\n{c['text']}"
        for c in chunks
    )


def format_history(history):
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
        for i, msg in enumerate(history)
    )


def call_llm(prompt_text):
    response = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": 0}
    })
    response.raise_for_status()
    return response.json()["response"]


def query(question, history, technique="baseline", department=None):
    chunks = hybrid_search(question, department=department, top_k=5)

    if not chunks:
        return "No relevant documents found. Please check your docs folder and re-run ingest.py."

    context = format_context(chunks)
    history_text = format_history(history)

    prompt_template = PROMPTS.get(technique, PROMPTS["baseline"])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    filled = prompt.format_messages(
        context=context,
        question=question,
        history=history_text
    )
    # Convert to plain text for direct REST call
    prompt_text = "\n".join(m.content for m in filled)

    return call_llm(prompt_text)


def main():
    parser = argparse.ArgumentParser(description="RAG Query CLI")
    parser.add_argument("--technique",
        choices=["baseline", "chain_of_thought", "few_shot", "self_critique"],
        default="chain_of_thought")
    parser.add_argument("--department", default=None)
    args = parser.parse_args()

    history = deque(maxlen=10)

    print(f"\nRAG Query CLI")
    print(f"Technique : {args.technique}")
    print(f"Department: {args.department or 'all'}")
    print("Type 'q' to quit, 'clear' to reset memory\n")

    while True:
        q = input("Question: ").strip()
        if q == "q":
            break
        if q == "clear":
            history.clear()
            print("Memory cleared.\n")
            continue
        if not q:
            continue

        history.append(q)
        answer = query(q, history, technique=args.technique, department=args.department)
        history.append(answer)

        print(f"\nAnswer: {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()