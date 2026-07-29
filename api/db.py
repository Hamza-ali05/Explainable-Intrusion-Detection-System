from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data/ids_history.db")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_timestamp TEXT NOT NULL,
              user_email TEXT,
              total_rows INTEGER,
              benign_count INTEGER,
              threat_count INTEGER,
              top_threat TEXT,
              top_threat_count INTEGER,
              model_used TEXT,
              input_method TEXT
            )
            """
        )
        conn.commit()


def save_session(record: dict[str, Any]) -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO analysis_sessions (
              session_timestamp,
              user_email,
              total_rows,
              benign_count,
              threat_count,
              top_threat,
              top_threat_count,
              model_used,
              input_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("session_timestamp"),
                record.get("user_email"),
                record.get("total_rows"),
                record.get("benign_count"),
                record.get("threat_count"),
                record.get("top_threat"),
                record.get("top_threat_count"),
                record.get("model_used"),
                record.get("input_method"),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_sessions(limit: int = 50) -> list[dict[str, Any]]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, session_timestamp, user_email, total_rows, benign_count,
                   threat_count, top_threat, top_threat_count, model_used, input_method
            FROM analysis_sessions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: int) -> bool:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "DELETE FROM analysis_sessions WHERE id = ?",
            (int(session_id),),
        )
        conn.commit()
        return cursor.rowcount > 0
