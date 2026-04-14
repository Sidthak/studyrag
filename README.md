# 📚 StudyRAG — Production RAG System

Ask questions about your study notes using AI.

## Features
- Hybrid retrieval (BM25 + Vector search)
- Cross-encoder reranking
- Citation enforcement
- Chat UI built with Streamlit

## Tech Stack
- OpenAI GPT-4o-mini
- ChromaDB
- LangChain
- Streamlit

## How to run
1. Add your documents to ./docs/
2. Run python ingest.py
3. Run streamlit run app.py