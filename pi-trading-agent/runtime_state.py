"""Small transactional runtime-state store backed by DuckDB.

Configuration and exported snapshots remain JSON files because humans and
external tools consume them. Restart-critical internal state belongs here so
all readers observe committed values and concurrent callbacks cannot see a
partially replaced file.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

import duckdb


class DuckDBStateStore:
    """Persist JSON-compatible values transactionally by namespace."""

    _CACHE_TTL_SECONDS = 1.0

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._schema_initialized = False
        self._lock = RLock()
        self._cache: dict[str, tuple[float, bool, str | None]] = {}

    def get(self, key: str) -> tuple[bool, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] <= self._CACHE_TTL_SECONDS:
                return self._decode(cached[1], cached[2])
        with self._lock, self._open() as conn:
            row = conn.execute(
                "SELECT payload FROM runtime_state WHERE state_key = ?", (key,)
            ).fetchone()
            payload = str(row[0]) if row is not None else None
            self._cache[key] = (time.monotonic(), row is not None, payload)
        return self._decode(row is not None, payload)

    @staticmethod
    def _decode(found: bool, payload: str | None) -> tuple[bool, Any]:
        if not found or payload is None:
            return False, None
        try:
            return True, json.loads(payload)
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
            self._cache[key] = (time.monotonic(), True, payload)

    def delete(self, key: str) -> None:
        with self._lock, self._open() as conn:
            conn.execute("DELETE FROM runtime_state WHERE state_key = ?", (key,))
            conn.commit()
            self._cache[key] = (time.monotonic(), False, None)

    @contextmanager
    def _open(self) -> Iterator[duckdb.DuckDBPyConnection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.database_path)) as conn:
            if not self._schema_initialized:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_state (
                        state_key TEXT PRIMARY KEY,
                        payload JSON NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp
                    )
                    """
                )
                self._schema_initialized = True
            yield conn

    def close(self) -> None:
        """Discard cached values so the next read observes durable state."""
        with self._lock:
            self._cache.clear()
