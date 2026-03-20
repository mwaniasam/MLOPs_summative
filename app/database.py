import sqlite3
import os
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "data/coffeeguard.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            accuracy REAL,
            images_used INTEGER,
            epochs INTEGER
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def log_upload(filename, label, file_path):
    conn = get_connection()
    conn.execute(
        "INSERT INTO uploads (filename, label, file_path, uploaded_at) VALUES (?, ?, ?, ?)",
        (filename, label, file_path, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_pending_uploads():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM uploads ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_retrain_start():
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO retrain_log (triggered_at, status, images_used, epochs) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), "running", 0, 0)
    )
    retrain_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return retrain_id


def log_retrain_complete(retrain_id, accuracy, images_used, epochs):
    conn = get_connection()
    conn.execute(
        """UPDATE retrain_log
           SET completed_at=?, status=?, accuracy=?, images_used=?, epochs=?
           WHERE id=?""",
        (datetime.utcnow().isoformat(), "completed", accuracy, images_used, epochs, retrain_id)
    )
    conn.commit()
    conn.close()


def log_retrain_failed(retrain_id, error):
    conn = get_connection()
    conn.execute(
        "UPDATE retrain_log SET completed_at=?, status=? WHERE id=?",
        (datetime.utcnow().isoformat(), f"failed: {error[:200]}", retrain_id)
    )
    conn.commit()
    conn.close()


def get_retrain_history():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM retrain_log ORDER BY triggered_at DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
