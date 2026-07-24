"""Small transactional runtime-state stores for restart-critical state.

Configuration and exported snapshots remain JSON files because humans and
external tools consume them. Restart-critical internal state belongs here so
all readers observe committed values and concurrent callbacks cannot see a
partially replaced file.
"""

import json
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Protocol
from urllib.parse import unquote, urlparse

import duckdb


class RuntimeStateStore(Protocol):
    """Minimal interface shared by runtime-state backends."""

    identity: tuple[Any, ...]

    def get(self, key: str) -> tuple[bool, Any]: ...

    def set(self, key: str, value: Any) -> None: ...

    def delete(self, key: str) -> None: ...

    def close(self) -> None: ...


class DuckDBStateStore:
    """Persist JSON-compatible values transactionally by namespace."""

    _CACHE_TTL_SECONDS = 1.0

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.identity = ("duckdb", self.database_path)
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


class _RedisConnection:
    """Tiny RESP client for the small command set runtime state needs."""

    def __init__(self, url: str, *, timeout_seconds: float = 0.5):
        parsed = urlparse(url)
        if parsed.scheme != "redis":
            raise ValueError("runtime-state Redis URLs must use redis://")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 6379
        self.username = unquote(parsed.username) if parsed.username else None
        self.password = unquote(parsed.password) if parsed.password else None
        raw_db = parsed.path.strip("/")
        self.database = int(raw_db) if raw_db else 0
        self.timeout_seconds = timeout_seconds

    def command(self, *parts: str | bytes) -> Any:
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_seconds
        ) as sock:
            sock.settimeout(self.timeout_seconds)
            stream = sock.makefile("rb")
            if self.password:
                auth_parts = (
                    ("AUTH", self.username, self.password)
                    if self.username
                    else ("AUTH", self.password)
                )
                self._send(sock, *auth_parts)
                self._read_response(stream)
            if self.database:
                self._send(sock, "SELECT", str(self.database))
                self._read_response(stream)
            self._send(sock, *parts)
            return self._read_response(stream)

    @staticmethod
    def _send(sock: socket.socket, *parts: str | bytes) -> None:
        payload = [f"*{len(parts)}\r\n".encode("ascii")]
        for part in parts:
            data = part if isinstance(part, bytes) else part.encode("utf-8")
            payload.append(f"${len(data)}\r\n".encode("ascii"))
            payload.append(data)
            payload.append(b"\r\n")
        sock.sendall(b"".join(payload))

    def _read_response(self, stream: Any) -> Any:
        line = stream.readline()
        if not line:
            raise ConnectionError("Redis closed the connection")
        prefix, payload = line[:1], line[1:-2]
        if prefix == b"+":
            return payload.decode("utf-8")
        if prefix == b":":
            return int(payload)
        if prefix == b"$":
            length = int(payload)
            if length == -1:
                return None
            data = stream.read(length)
            stream.read(2)
            return data
        if prefix == b"-":
            raise RuntimeError(payload.decode("utf-8", errors="replace"))
        raise RuntimeError(f"Unsupported Redis response: {line!r}")


class RedisStateStore:
    """Redis-backed runtime state with DuckDB backup and migration fallback."""

    _CACHE_TTL_SECONDS = 1.0

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str,
        backup_store: DuckDBStateStore | None = None,
    ):
        if not key_prefix:
            raise ValueError("Redis runtime-state key_prefix must not be empty")
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.backup_store = backup_store
        self.identity = (
            "redis",
            self.redis_url,
            self.key_prefix,
            backup_store.database_path if backup_store is not None else None,
        )
        self._lock = RLock()
        self._cache: dict[str, tuple[float, bool, str | None]] = {}

    def get(self, key: str) -> tuple[bool, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] <= self._CACHE_TTL_SECONDS:
                return DuckDBStateStore._decode(cached[1], cached[2])
            try:
                payload = self._redis_get(key)
            except (OSError, RuntimeError, ValueError, ConnectionError):
                payload = None
            if payload is not None:
                self._cache[key] = (time.monotonic(), True, payload)
                return DuckDBStateStore._decode(True, payload)
            if self.backup_store is not None:
                found, value = self.backup_store.get(key)
                if found:
                    self.set(key, value)
                    return True, value
            self._cache[key] = (time.monotonic(), False, None)
            return False, None

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
        with self._lock:
            redis_written = False
            try:
                self._redis_command("SET", self._redis_key(key), payload)
                redis_written = True
            except (OSError, RuntimeError, ValueError, ConnectionError):
                pass
            if self.backup_store is not None:
                self.backup_store.set(key, value)
            elif not redis_written:
                raise ConnectionError("Redis runtime-state write failed")
            self._cache[key] = (time.monotonic(), True, payload)

    def delete(self, key: str) -> None:
        with self._lock:
            redis_deleted = False
            try:
                self._redis_command("DEL", self._redis_key(key))
                redis_deleted = True
            except (OSError, RuntimeError, ValueError, ConnectionError):
                pass
            if self.backup_store is not None:
                self.backup_store.delete(key)
            elif not redis_deleted:
                raise ConnectionError("Redis runtime-state delete failed")
            self._cache[key] = (time.monotonic(), False, None)

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
        if self.backup_store is not None:
            self.backup_store.close()

    def _redis_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def _redis_get(self, key: str) -> str | None:
        value = self._redis_command("GET", self._redis_key(key))
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _redis_command(self, *parts: str) -> Any:
        return _RedisConnection(self.redis_url).command(*parts)


def create_runtime_state_store(
    database_path: Path,
    *,
    redis_url: str | None = None,
    redis_key_prefix: str | None = None,
) -> RuntimeStateStore:
    """Create the configured runtime-state store.

    DuckDB remains the durable local backup when Redis is enabled. That lets a
    Redis-backed process lazily migrate existing DuckDB keys on first read and
    keeps restart-critical state recoverable if Redis persistence is disabled.
    """
    backup_store = DuckDBStateStore(database_path)
    if redis_url:
        return RedisStateStore(
            redis_url,
            key_prefix=redis_key_prefix or "pi-trading:runtime:",
            backup_store=backup_store,
        )
    return backup_store
