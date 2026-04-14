# monitor.py
# Tracks every query locally — latency, cost, quality.
# Stores everything in a simple SQLite database.

import sqlite3
import time
import os
from datetime import datetime

DB_PATH = "./monitoring.db"


def init_db():
    """Create the monitoring database if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            answer TEXT,
            latency_seconds REAL,
            num_chunks INTEGER,
            top_rerank_score REAL,
            declined INTEGER,
            sources TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_query(query, answer, latency, chunks, declined, sources):
    """Save one query's metrics to the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    top_score = chunks[0].get("rerank_score", 0) if chunks else 0
    cursor.execute("""
        INSERT INTO queries
        (timestamp, query, answer, latency_seconds, num_chunks, top_rerank_score, declined, sources)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        query,
        answer,
        round(latency, 2),
        len(chunks),
        round(top_score, 3),
        1 if declined else 0,
        ", ".join(sources),
    ))
    conn.commit()
    conn.close()


def get_all_queries():
    """Fetch all logged queries from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM queries ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows