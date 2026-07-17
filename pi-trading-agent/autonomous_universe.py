"""Bounded, persistent discovery of Alpaca-tradable US equities.

Backed by embedded DuckDB, like ``trade_memory.py``, ``portfolio_memory.py``,
and ``symbol_reference.py`` -- this used to be the one remaining memory
module on plain JSON, rewriting the whole state file on every call. An
existing ``.autonomous_universe.json`` from before this migration is
imported once (mirroring ``trade_memory.py``'s ``_migrate_legacy_sqlite``
pattern) and never deleted.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import duckdb
import requests


class AutonomousUniverse:
    """Rotate through a small batch of active assets without an unbounded scan."""

    # The assets endpoint lives on a different host per trading mode; paper
    # keys are rejected by the live host and vice versa.
    ASSETS_URL_PAPER = "https://paper-api.alpaca.markets/v2/assets"
    ASSETS_URL_LIVE = "https://api.alpaca.markets/v2/assets"
    _SYMBOL = re.compile(r"^[A-Z]{1,5}$")

    def __init__(
        self,
        database_path: Path,
        refresh_days: int,
        batch_size: int,
        paper: bool = True,
        legacy_json_path: Path | None = None,
    ):
        self.database_path = database_path
        self.refresh_days = refresh_days
        self.batch_size = batch_size
        self.assets_url = self.ASSETS_URL_PAPER if paper else self.ASSETS_URL_LIVE
        self.legacy_json_path = legacy_json_path

    def next_batch(self, api_key: str, secret_key: str) -> list[str]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            self._migrate_legacy_json(conn)
            refreshed_value = self._get_state(conn, "refreshed")
            cursor_value = self._get_state(conn, "cursor")
            symbols = [
                row[0] for row in conn.execute("SELECT symbol FROM universe_symbols ORDER BY rank").fetchall()
            ]
            conn.commit()
        try:
            refreshed = date.fromisoformat(refreshed_value) if refreshed_value else date(1970, 1, 1)
        except ValueError:
            refreshed = date(1970, 1, 1)
        today = date.today()
        if not symbols or today - refreshed >= timedelta(days=self.refresh_days):
            response = requests.get(
                self.assets_url,
                params={"status": "active", "asset_class": "us_equity"},
                headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
                timeout=20,
            )
            response.raise_for_status()
            symbols = sorted(
                item["symbol"].upper()
                for item in response.json()
                if item.get("tradable") is True
                and item.get("fractionable") is True
                and self._SYMBOL.fullmatch(str(item.get("symbol", "")).upper())
            )
            with self._connect() as conn:
                self._create_schema(conn)
                conn.execute("DELETE FROM universe_symbols")
                if symbols:
                    conn.executemany(
                        "INSERT INTO universe_symbols (symbol, rank) VALUES (?, ?)",
                        [(symbol, rank) for rank, symbol in enumerate(symbols)],
                    )
                self._set_state(conn, "cursor", "0")
                self._set_state(conn, "refreshed", today.isoformat())
                conn.commit()
            cursor = 0
        else:
            cursor = int(cursor_value or 0) % len(symbols)
        if not symbols:
            return []
        batch = [symbols[(cursor + offset) % len(symbols)] for offset in range(min(self.batch_size, len(symbols)))]
        new_cursor = (cursor + len(batch)) % len(symbols)
        with self._connect() as conn:
            self._create_schema(conn)
            self._set_state(conn, "cursor", str(new_cursor))
            learned = [
                row[0]
                for row in conn.execute(
                    "SELECT symbol FROM learned_symbols ORDER BY last_seen_rank ASC"
                ).fetchall()
            ]
            conn.commit()
        return list(dict.fromkeys(learned + batch))

    def remember(self, symbols: list[str], limit: int = 30) -> None:
        """Keep historically qualifying symbols in future daily evaluations.

        Re-mentioning an already-learned symbol (e.g. a current holding,
        re-remembered every day) refreshes its recency so it is never
        trimmed out of the `limit`-sized window while it is still owned --
        the same guarantee the old JSON-backed recency trim provided.
        """
        valid = list(
            dict.fromkeys(
                str(symbol).upper() for symbol in symbols if self._SYMBOL.fullmatch(str(symbol).upper())
            )
        )
        if not valid:
            return
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            self._migrate_legacy_json(conn)
            next_rank = (
                conn.execute("SELECT COALESCE(MAX(last_seen_rank), 0) FROM learned_symbols").fetchone()[0] or 0
            ) + 1
            conn.executemany(
                """
                INSERT INTO learned_symbols (symbol, last_seen_rank) VALUES (?, ?)
                ON CONFLICT (symbol) DO UPDATE SET last_seen_rank = excluded.last_seen_rank
                """,
                [(symbol, next_rank) for symbol in valid],
            )
            conn.execute(
                """
                DELETE FROM learned_symbols WHERE symbol NOT IN (
                    SELECT symbol FROM learned_symbols ORDER BY last_seen_rank DESC LIMIT ?
                )
                """,
                (limit,),
            )
            conn.commit()

    def managed_symbols(self) -> list[str]:
        """Return the persisted discovery symbols the strategy is allowed to own.

        This is deliberately separate from the full Alpaca asset directory.
        The directory is only a source of candidates; a holding becomes managed
        only after it is in the configured watchlist or this persisted list.
        """
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                self._create_schema(conn)
                self._migrate_legacy_json(conn)
                rows = conn.execute(
                    "SELECT symbol FROM learned_symbols ORDER BY last_seen_rank ASC"
                ).fetchall()
                conn.commit()
            return list(dict.fromkeys(row[0] for row in rows))
        except Exception:
            return []

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database_path))

    @staticmethod
    def _create_schema(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS universe_state (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS universe_symbols (
                symbol TEXT PRIMARY KEY,
                rank INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_symbols (
                symbol TEXT PRIMARY KEY,
                last_seen_rank BIGINT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_state (
                name TEXT PRIMARY KEY
            )
            """
        )

    @staticmethod
    def _get_state(conn: duckdb.DuckDBPyConnection, name: str) -> str | None:
        row = conn.execute("SELECT value FROM universe_state WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _set_state(conn: duckdb.DuckDBPyConnection, name: str, value: str) -> None:
        conn.execute(
            "INSERT INTO universe_state VALUES (?, ?) ON CONFLICT (name) DO UPDATE SET value = excluded.value",
            (name, value),
        )

    def _migrate_legacy_json(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Copy the old JSON state once when upgrading an existing installation.

        Guarded by a migration_state marker, exactly like
        ``trade_memory.py``'s ``_migrate_legacy_sqlite``. Fails open: a
        missing or corrupt legacy file simply leaves the new database empty,
        never blocking discovery. The old file is never deleted.
        """
        migrated = conn.execute(
            "SELECT 1 FROM migration_state WHERE name = 'json_to_duckdb'"
        ).fetchone()
        if migrated or self.legacy_json_path is None or not self.legacy_json_path.is_file():
            return
        try:
            raw = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
            legacy_state = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            legacy_state = {}
        symbols = legacy_state.get("symbols", [])
        if isinstance(symbols, list) and symbols:
            rows = [
                (str(symbol).upper(), rank)
                for rank, symbol in enumerate(symbols)
                if self._SYMBOL.fullmatch(str(symbol).upper())
            ]
            if rows:
                conn.executemany(
                    "INSERT INTO universe_symbols (symbol, rank) VALUES (?, ?) ON CONFLICT (symbol) DO NOTHING",
                    rows,
                )
        try:
            cursor = int(legacy_state.get("cursor", 0))
            self._set_state(conn, "cursor", str(cursor))
        except (TypeError, ValueError):
            pass
        refreshed = legacy_state.get("refreshed")
        if isinstance(refreshed, str) and refreshed:
            self._set_state(conn, "refreshed", refreshed)
        learned = legacy_state.get("learned", [])
        if isinstance(learned, list) and learned:
            rows = [
                (str(symbol).upper(), rank)
                for rank, symbol in enumerate(learned)
                if self._SYMBOL.fullmatch(str(symbol).upper())
            ]
            if rows:
                conn.executemany(
                    "INSERT INTO learned_symbols (symbol, last_seen_rank) VALUES (?, ?) ON CONFLICT (symbol) DO NOTHING",
                    rows,
                )
        conn.execute("INSERT INTO migration_state VALUES ('json_to_duckdb')")
