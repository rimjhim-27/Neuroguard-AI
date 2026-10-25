
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "neuroguard.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_rul REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_prediction(predicted_rul):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (predicted_rul)
            VALUES (?)
            """,
            (predicted_rul,),
        )
        return cursor.lastrowid


def get_predictions(limit=20):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, predicted_rul, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]
