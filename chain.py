# chain.py
# This is the brain of the RAG system.
# It takes a question, retrieves chunks,
# checks citations, and generates an answer.

import os
import time
from monitor import log_query
from langsmith import traceable
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import retrieve

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CITATION_THRESHOLD = -2.0


def check_citations(chunks: list) -> bool:
    """Only answer if retrieved chunks are relevant enough."""
    if not chunks:
        return False
    best_score = chunks[0].get("rerank_score", -99)
    return best_score >= CITATION_THRESHOLD


def build_prompt(query: str, chunks: list) -> str:
    """Build the prompt with context from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = os.path.basename(chunk.get("source", "unknown"))
        context_parts.append(f"[Source {i + 1} — {source}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)
    return f"""You are a helpful study assistant. Answer using ONLY the context below.

RULES:
1. Only use information from the context below.
2. Always mention which source you used.
3. If context is not enough, say "I don't have enough information."
4. Do NOT make up anything.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


@traceable
def ask(query: str) -> dict:
    """Full RAG chain — retrieve, check, answer."""
    start_time = time.time()

    chunks = retrieve(query)

    if not check_citations(chunks):
        latency = time.time() - start_time
        log_query(query, "Declined", latency, [], True, [])
        return {
            "answer": "I couldn't find relevant information in your study materials.",
            "sources": [],
            "chunks": [],
            "declined": True,
        }

    prompt = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
    )

    answer = response.choices[0].message.content.strip()
    sources = list({os.path.basename(c.get("source", "unknown")) for c in chunks})
    latency = time.time() - start_time

    log_query(query, answer, latency, chunks, False, sources)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "declined": False,
    }