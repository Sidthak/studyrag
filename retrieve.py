# retrieve.py
# This file searches your documents using TWO methods:
# 1. BM25 keyword search
# 2. Vector semantic search
# Then combines and reranks the results.
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import CrossEncoder

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BM25_PATH = "./bm25_index.pkl"
CHROMA_PATH = "./chroma_db"

print("Loading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def embed_query(query: str) -> list:
    """Convert the user's question into a vector."""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def vector_search(query: str, top_k: int = 15) -> list:
    """Search by MEANING — finds chunks similar to the question."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection("studyrag")

    embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "score": 1 - results["distances"][0][i],
            "method": "vector",
        })
    return chunks
def bm25_search(query: str, top_k: int = 15) -> list:
    """Search by KEYWORDS — finds chunks with exact words."""
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError("BM25 index not found. Run ingest.py first!")

    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    texts = data["texts"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]
    chunks = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunks.append({
                "text": texts[idx],
                "source": "bm25_index",
                "score": float(scores[idx]),
                "method": "bm25",
            })
    return chunks


def reciprocal_rank_fusion(vector_results: list, bm25_results: list, k: int = 60) -> list:
    """Combine BM25 and vector results into one ranked list."""
    scores = {}
    texts_map = {}

    for rank, chunk in enumerate(vector_results):
        key = chunk["text"][:120]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        texts_map[key] = chunk

    for rank, chunk in enumerate(bm25_results):
        key = chunk["text"][:120]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in texts_map:
            texts_map[key] = chunk

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [texts_map[k] for k in sorted_keys]


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """Pick the best 5 chunks using cross-encoder."""
    if not chunks:
        return []
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def retrieve(query: str, top_k: int = 5) -> list:
    """Full retrieval pipeline — runs all steps together."""
    vector_results = vector_search(query, top_k=15)
    bm25_results = bm25_search(query, top_k=15)
    merged = reciprocal_rank_fusion(vector_results, bm25_results)
    final = rerank(query, merged, top_k=top_k)
    return final