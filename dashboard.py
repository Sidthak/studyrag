# dashboard.py
# A live monitoring dashboard for StudyRAG.
# Run with: streamlit run dashboard.py

import streamlit as st
import sqlite3
import pandas as pd
from monitor import get_all_queries, DB_PATH, init_db

st.set_page_config(
    page_title="StudyRAG Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 StudyRAG — Monitoring Dashboard")
st.caption("Real-time observability for your RAG system")

init_db()

rows = get_all_queries()

if not rows:
    st.warning("No queries logged yet! Run the app and ask some questions first.")
    st.stop()

df = pd.DataFrame(rows, columns=[
    "id", "timestamp", "query", "answer",
    "latency_seconds", "num_chunks", "top_rerank_score",
    "declined", "sources"
])

# ── Top metrics ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Queries", len(df))
col2.metric("Avg Latency", f"{df['latency_seconds'].mean():.2f}s")
col3.metric("Declined Queries", int(df['declined'].sum()))
col4.metric("Avg Rerank Score", f"{df['top_rerank_score'].mean():.2f}")

st.divider()

# ── Latency chart ─────────────────────────────────────────
st.subheader("⏱️ Latency per Query")
st.line_chart(df[["latency_seconds"]].rename(
    columns={"latency_seconds": "seconds"}
))

st.divider()

# ── Rerank score chart ────────────────────────────────────
st.subheader("🎯 Top Rerank Score per Query")
st.bar_chart(df[["top_rerank_score"]].rename(
    columns={"top_rerank_score": "score"}
))

st.divider()

# ── Full query log ────────────────────────────────────────
st.subheader("📋 Full Query Log")
st.dataframe(
    df[["timestamp", "query", "latency_seconds",
        "top_rerank_score", "declined", "sources"]],
    use_container_width=True,
)