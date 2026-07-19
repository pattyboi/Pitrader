"""Small transactional runtime-state store backed by DuckDB.

Configuration and exported snapshots remain JSON files because humans and
external tools consume them. Restart-critical internal state belongs here so
all readers observe committed values and concurrent callbacks cannot see a
partially replaced file.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

import duckdb


class DuckDBStateStore:
    """Persist JSON-compatible values transactionally by namespace."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._schema_initialized = False
        self._lock = RLock()
        self._connection: duckdb.DuckDBPyConnection | None = None

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock, self._open() as conn:
            row = conn.execute(
                "SELECT payload FROM runtime_state WHERE state_key = ?", (key,)
            ).fetchone()
        if row is None:
            return False, None
        try:
            return True, json.loads(str(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False, None

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
        with self._lock, self._open() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state (state_key, payload, updated_at)
                VALUES (?, ?, current_timestamp)
                ON CONFLICT(state_key) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (key, payload),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        with self._lock, self._open() as conn:
            conn.execute("DELETE FROM runtime_state WHERE state_key = ?", (key,))
            conn.commit()

    @contextmanager
    def _open(self) -> Iterator[duckdb.DuckDBPyConnection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if self._connection is None:
            self._connection = duckdb.connect(str(self.database_path))
        if not self._schema_initialized:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_state (
                    state_key TEXT PRIMARY KEY,
                    payload JSON NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp
                )
                """
            )
            self._schema_initialized = True
        yield self._connection

    def close(self) -> None:
        """Release the process-local connection when its owner shuts down."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._schema_initialized = False
