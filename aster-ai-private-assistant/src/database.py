import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "aster_history.db"

def _connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_message(role, content):
    if role not in ("user", "assistant"):
        raise ValueError("role must be user or assistant")
    with _connect() as conn:
        conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))

def load_messages(limit=100):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]

def clear_messages():
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
