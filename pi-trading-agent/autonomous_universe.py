"""Bounded, persistent discovery of Alpaca-tradable US equities.

Backed by embedded DuckDB, like ``trade_memory.py``, ``portfolio_memory.py``,
and ``symbol_reference.py``.
"""

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import requests


class AutonomousUniverse:
    """Rotate through a small batch of active assets without an unbounded scan."""

    # The assets endpoint lives on a different host per trading mode; paper
    # keys are rejected by the live host and vice versa.
    ASSETS_URL_PAPER = "https://paper-api.alpaca.markets/v2/assets"
    ASSETS_URL_LIVE = "https://api.alpaca.markets/v2/assets"
    _SYMBOL = re.compile(r"^[A-Z]{1,5}$")

    @staticmethod
    def _default_symbol_filter(item: dict[str, Any]) -> str | None:
        """Equity default: a plain 1-5 letter ticker, tradable and fractionable."""
        symbol = str(item.get("symbol", "")).upper()
        if (
            item.get("tradable") is True
            and item.get("fractionable") is True
            and AutonomousUniverse._SYMBOL.fullmatch(symbol)
        ):
            return symbol
        return None

    def __init__(
        self,
        database_path: Path,
        refresh_days: int,
        batch_size: int,
        paper: bool = True,
        asset_class: str = "us_equity",
        symbol_filter: Callable[[dict[str, Any]], str | None] | None = None,
    ):
        self.database_path = database_path
        self.refresh_days = refresh_days
        self.batch_size = batch_size
        self.assets_url = self.ASSETS_URL_PAPER if paper else self.ASSETS_URL_LIVE
        # Defaults preserve today's exact equity behavior. CryptoRotationStrategy
        # passes asset_class="crypto" and its own symbol_filter, since Alpaca's
        # crypto assets come back as "BASE/QUOTE" pairs (e.g. "BTC/USD") rather
        # than plain equity tickers, and have no "fractionable" field to check.
        self.asset_class = asset_class
        self._symbol_filter = symbol_filter or self._default_symbol_filter
        self._schema_initialized = False

    def next_batch(self, api_key: str, secret_key: str) -> list[str]:
        with self._open() as conn:
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
                params={"status": "active", "asset_class": self.asset_class},
                headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
                timeout=20,
            )
            response.raise_for_status()
            symbols = sorted(
                {
                    symbol
                    for symbol in (self._symbol_filter(item) for item in response.json())
                    if symbol
                }
            )
            with self._open() as conn:
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
        with self._open() as conn:
            self._set_state(conn, "cursor", str(new_cursor))
            learned = [
                row[0]
                for row in conn.execute(
                    "SELECT symbol FROM learned_symbols ORDER BY last_seen_rank ASC"
                ).fetchall()
            ]
            unpriceable = {row[0] for row in conn.execute("SELECT symbol FROM unpriceable_symbols").fetchall()}
            conn.commit()
        return [symbol for symbol in dict.fromkeys(learned + batch) if symbol not in unpriceable]

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
        with self._open() as conn:
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
        """Return discovery symbols confirmed by a strategy buy fill.

        This is deliberately separate from the full Alpaca asset directory.
        The directory is only a source of candidates; a holding becomes managed
        only after it is in the configured watchlist or this persisted list.
        """
        try:
            with self._open() as conn:
                rows = conn.execute("SELECT symbol FROM owned_symbols ORDER BY symbol").fetchall()
                conn.commit()
            return list(dict.fromkeys(row[0] for row in rows))
        except Exception:
            return []

    def remember_owned(self, symbols: list[str]) -> None:
        """Persist broker-confirmed strategy ownership separately from candidates."""
        valid = sorted(
            {
                str(symbol).upper()
                for symbol in symbols
                if self._SYMBOL.fullmatch(str(symbol).upper())
            }
        )
        if not valid:
            return
        with self._open() as conn:
            conn.executemany(
                "INSERT INTO owned_symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
                [(symbol,) for symbol in valid],
            )
            conn.commit()

    def exclude_unpriceable(self, symbols: list[str]) -> None:
        """Persist symbols confirmed to have no Alpaca price history at all.

        Such a symbol (e.g. a thinly traded OTC ADR that passes the
        tradable/fractionable asset filter in `next_batch` but has no
        historical bars) can never clear the dip-signal check, so this stops
        it from being handed back out -- and the same Alpaca round trip and
        warning repeated -- the next time the rotation cursor, or a
        `remember()`'d entry, would otherwise resurface it.
        """
        valid = sorted(
            {
                str(symbol).upper()
                for symbol in symbols
                if self._SYMBOL.fullmatch(str(symbol).upper())
            }
        )
        if not valid:
            return
        with self._open() as conn:
            conn.executemany(
                "INSERT INTO unpriceable_symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
                [(symbol,) for symbol in valid],
            )
            conn.execute(
                "DELETE FROM learned_symbols WHERE symbol IN (SELECT symbol FROM unpriceable_symbols)"
            )
            conn.commit()

    def forget_owned(self, symbols: list[str]) -> None:
        """Revoke management permission after a strategy-owned position is sold."""
        valid = sorted(
            {
                str(symbol).upper()
                for symbol in symbols
                if self._SYMBOL.fullmatch(str(symbol).upper())
            }
        )
        if not valid:
            return
        with self._open() as conn:
            placeholders = ", ".join("?" for _ in valid)
            conn.execute(
                f"DELETE FROM owned_symbols WHERE symbol IN ({placeholders})", valid
            )
            conn.commit()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database_path))

    @contextmanager
    def _open(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Ensure the directory/schema exist, then hand back a ready
        connection -- shared prologue for every call site below. Schema
        creation only runs once per instance (mirrors DuckDBStateStore),
        so callers should hold onto one instance across a process's calls
        rather than constructing a fresh one each time."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            if not self._schema_initialized:
                self._create_schema(conn)
                self._schema_initialized = True
            yield conn

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
            CREATE TABLE IF NOT EXISTS owned_symbols (
                symbol TEXT PRIMARY KEY
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unpriceable_symbols (
                symbol TEXT PRIMARY KEY
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
