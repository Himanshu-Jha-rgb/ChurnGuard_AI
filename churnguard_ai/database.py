from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PredictionRecord:
    customer_id: str
    churn_prediction: int
    churn_probability: float
    model_name: str


class PredictionDatabase:
    """Simple SQLite-backed store for prediction history."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    churn_prediction INTEGER NOT NULL,
                    churn_probability REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def insert_predictions(self, records: list[PredictionRecord]) -> None:
        if not records:
            return

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO prediction_history (
                    customer_id,
                    churn_prediction,
                    churn_probability,
                    model_name
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        record.customer_id,
                        record.churn_prediction,
                        record.churn_probability,
                        record.model_name,
                    )
                    for record in records
                ],
            )

    def fetch_recent_predictions(self, limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    customer_id,
                    churn_prediction,
                    churn_probability,
                    model_name,
                    created_at
                FROM prediction_history
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

