"""Durable, pooled memory for evaluating dip signals across the whole portfolio.

Unlike ``trade_memory.py``'s ``TradeMemory`` (scoped to a single Asset-A/B
pair), this tracks every symbol the portfolio evaluates each day, keyed by
``(evaluation_date, symbol)``. One pooled regression is fit across all
symbols' history rather than a model per symbol -- a single symbol's dip
signals are too sparse to fit reliably on their own, while pooling lets every
qualifying symbol's daily observation contribute to the same model.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from trade_memory import RotationForecast
from market_sessions import is_next_trading_session
from ridge_regression import fit_two_feature_ridge_model


@dataclass(frozen=True)
class PortfolioMemoryInput:
    symbol: str
    price: float
    dip_percent: float
    news_score: int | None
    signal_present: bool = True
    live_spread_percent: float | None = None
    recent_avg_volume: float | None = None
    historical_expected_profit: float | None = None
    historical_win_probability: float | None = None
    historical_return_stdev: float | None = None


class PortfolioMemory:
    """Record per-symbol dip observations and forecast next-session return."""

    def __init__(
        self,
        database_path: Path,
        minimum_observations: int,
        maximum_observations: int,
        next_session_predicate: Callable[[str, str], bool] = is_next_trading_session,
    ):
        self.database_path = database_path
        self.minimum_observations = minimum_observations
        self.maximum_observations = maximum_observations
        self._schema_initialized = False
        # Defaults to NYSE-trading-session succession (equity's behavior,
        # unchanged). CryptoRotationStrategy passes
        # market_sessions.is_next_calendar_day instead, since crypto trades
        # every calendar day rather than skipping weekends/holidays.
        self._next_session_predicate = next_session_predicate

    def update_and_forecast(
        self,
        evaluation_date: str,
        symbol: str,
        price: float,
        dip_percent: float,
        news_score: int | None,
        signal_present: bool = True,
        live_spread_percent: float | None = None,
        recent_avg_volume: float | None = None,
        historical_expected_profit: float | None = None,
        historical_win_probability: float | None = None,
        historical_return_stdev: float | None = None,
    ) -> RotationForecast:
        """Settle this symbol's prior observation, record today's, and forecast.

        Settlement is scoped to this symbol only: a next-session return can
        only be measured from the same symbol's own later price. The fit
        that follows pools every *signal-present* symbol's settled history
        together -- `signal_present` marks whether today's dip actually
        cleared the live threshold, exactly like trade_memory.py's own
        column of the same name, so recording every evaluated symbol's daily
        context (not just qualifying ones) never dilutes the pooled forecast
        with ordinary non-dip market days. The extra fact columns are
        durable context for the day, not model inputs.
        """
        item = PortfolioMemoryInput(
            symbol=symbol,
            price=price,
            dip_percent=dip_percent,
            news_score=news_score,
            signal_present=signal_present,
            live_spread_percent=live_spread_percent,
            recent_avg_volume=recent_avg_volume,
            historical_expected_profit=historical_expected_profit,
            historical_win_probability=historical_win_probability,
            historical_return_stdev=historical_return_stdev,
        )
        return self.update_many_and_forecast(evaluation_date, [item])[symbol]

    def update_many_and_forecast(
        self,
        evaluation_date: str,
        observations: list[PortfolioMemoryInput],
    ) -> dict[str, RotationForecast]:
        """Settle, insert, and forecast a full iteration in one transaction."""
        if not observations:
            return {}
        with self._open() as conn:
            for item in observations:
                self._settle_prior_observations_with_predicate(
                    conn, item.symbol, evaluation_date, item.price
                )
            frame = pd.DataFrame(
                [
                    {
                        "evaluation_date": evaluation_date,
                        "symbol": item.symbol,
                        "price": item.price,
                        "dip_percent": item.dip_percent,
                        "news_score": item.news_score,
                        "signal_present": int(item.signal_present),
                        "live_spread_percent": item.live_spread_percent,
                        "recent_avg_volume": item.recent_avg_volume,
                        "historical_expected_profit": item.historical_expected_profit,
                        "historical_win_probability": item.historical_win_probability,
                        "historical_return_stdev": item.historical_return_stdev,
                    }
                    for item in observations
                ]
            )
            conn.register("portfolio_memory_inputs", frame)
            try:
                conn.execute(
                    """
                    INSERT INTO observations
                        (evaluation_date, symbol, price, dip_percent, news_score, signal_present,
                         live_spread_percent, recent_avg_volume, historical_expected_profit,
                         historical_win_probability, historical_return_stdev)
                    SELECT evaluation_date, symbol, price, dip_percent, news_score, signal_present,
                           live_spread_percent, recent_avg_volume, historical_expected_profit,
                           historical_win_probability, historical_return_stdev
                    FROM portfolio_memory_inputs
                    ON CONFLICT(evaluation_date, symbol) DO NOTHING
                    """
                )
            finally:
                conn.unregister("portfolio_memory_inputs")
            conn.commit()
            rows = conn.execute(
                """
                SELECT dip_percent, news_score, next_session_return_percent
                FROM observations
                WHERE next_session_return_percent IS NOT NULL AND signal_present = 1
                ORDER BY evaluation_date DESC LIMIT ?
                """,
                (self.maximum_observations,),
            ).fetchall()
        return self._fit_many(list(reversed(rows)), observations)

    def backfill_history(self, symbol: str, rows: list[tuple[str, float, float]]) -> int:
        """Import completed daily (date, dip, next_session_return) rows for one symbol.

        Deliberately price-only, like ``TradeMemory.backfill_history``:
        inventing historic news scores would make the learned relationship
        look more certain than it is.
        """
        return self.backfill_many({symbol: rows})

    def backfill_many(self, histories: dict[str, list[tuple[str, float, float]]]) -> int:
        """Bulk import settled histories for multiple symbols."""
        records = [
            {
                "evaluation_date": date,
                "symbol": symbol,
                "dip_percent": dip,
                "next_session_return_percent": max(-25.0, min(25.0, next_return)),
            }
            for symbol, rows in histories.items()
            for date, dip, next_return in rows
            if math.isfinite(dip) and math.isfinite(next_return)
        ]
        if not records:
            return 0
        frame = pd.DataFrame.from_records(records)
        with self._open() as conn:
            before = self._observation_count(conn)
            conn.register("portfolio_memory_backfill", frame)
            try:
                conn.execute(
                    """
                    INSERT INTO observations
                        (evaluation_date, symbol, price, dip_percent, news_score, next_session_return_percent)
                    SELECT evaluation_date, symbol, NULL, dip_percent, NULL,
                           next_session_return_percent
                    FROM portfolio_memory_backfill
                    ON CONFLICT(evaluation_date, symbol) DO NOTHING
                    """
                )
            finally:
                conn.unregister("portfolio_memory_backfill")
            inserted = self._observation_count(conn) - before
            conn.commit()
        return inserted

    @staticmethod
    def _create_schema(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                evaluation_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL,
                dip_percent REAL NOT NULL,
                news_score INTEGER,
                next_session_return_percent REAL,
                signal_present INTEGER NOT NULL DEFAULT 1,
                live_spread_percent REAL,
                recent_avg_volume REAL,
                historical_expected_profit REAL,
                historical_win_probability REAL,
                historical_return_stdev REAL,
                PRIMARY KEY (evaluation_date, symbol)
            )
            """
        )
        # An install created before these columns existed already has an
        # observations table without them. DuckDB's ADD COLUMN can't carry a
        # NOT NULL/DEFAULT constraint, so add each missing column plain, then
        # backfill signal_present (every pre-existing row predates broadened
        # coverage, so it was always a qualifying dip signal) and set its
        # default for future inserts that omit it -- exactly once, guarded by
        # column presence rather than re-scanning the table on every call.
        existing_columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'observations'"
            ).fetchall()
        }
        new_columns = {
            "signal_present": "INTEGER",
            "live_spread_percent": "REAL",
            "recent_avg_volume": "REAL",
            "historical_expected_profit": "REAL",
            "historical_win_probability": "REAL",
            "historical_return_stdev": "REAL",
        }
        for column, column_type in new_columns.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE observations ADD COLUMN {column} {column_type}")
        if "signal_present" not in existing_columns:
            conn.execute("UPDATE observations SET signal_present = 1 WHERE signal_present IS NULL")
            conn.execute("ALTER TABLE observations ALTER COLUMN signal_present SET DEFAULT 1")

    def _settle_prior_observations_with_predicate(
        self, conn: duckdb.DuckDBPyConnection, symbol: str, date: str, price: float
    ) -> None:
        rows = conn.execute(
            """
            SELECT evaluation_date, price FROM observations
            WHERE symbol = ? AND evaluation_date < ?
              AND next_session_return_percent IS NULL AND price IS NOT NULL
            ORDER BY evaluation_date DESC LIMIT 1
            """,
            (symbol, date),
        ).fetchall()
        for prior_date, prior_price in rows:
            if not self._next_session_predicate(str(prior_date), date):
                continue
            if prior_price is None or prior_price <= 0:
                continue
            next_return = ((price - prior_price) / prior_price) * 100.0
            # Check finiteness before clamping: min/max silently turn NaN
            # into the clamp bound, which would record a fabricated return.
            if math.isfinite(next_return):
                conn.execute(
                    "UPDATE observations SET next_session_return_percent = ? "
                    "WHERE evaluation_date = ? AND symbol = ?",
                    (max(-25.0, min(25.0, next_return)), prior_date, symbol),
                )

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database_path))

    @contextmanager
    def _open(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Ensure the directory/schema exist, then hand back a ready
        connection -- shared prologue for every public method above."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            if not self._schema_initialized:
                self._create_schema(conn)
                self._schema_initialized = True
            yield conn

    @staticmethod
    def _observation_count(conn: duckdb.DuckDBPyConnection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def _fit_many(
        self,
        rows: list[tuple[float, int | None, float]],
        observations: list[PortfolioMemoryInput],
    ) -> dict[str, RotationForecast]:
        count = len(rows)
        if count < self.minimum_observations:
            forecast = RotationForecast(
                count,
                False,
                None,
                None,
                f"Portfolio memory is warming up: {count}/{self.minimum_observations} pooled dip signals settled.",
            )
            return {item.symbol: forecast for item in observations}
        model = fit_two_feature_ridge_model(rows)
        if model is None:
            forecast = RotationForecast(
                count,
                False,
                None,
                None,
                "Portfolio memory lacks enough feature variation for a trustworthy forecast.",
            )
            return {item.symbol: forecast for item in observations}
        forecasts: dict[str, RotationForecast] = {}
        for item in observations:
            predicted = model.predict(item.dip_percent, item.news_score)
            forecasts[item.symbol] = RotationForecast(
                count,
                True,
                predicted,
                model.correlation,
                f"Portfolio memory used {count} pooled dip signals; predicted next-session return "
                f"{predicted:+.2f}% "
                f"(fit correlation {model.correlation:+.2f}).",
            )
        return forecasts
