"""Durable, pooled memory for evaluating dip signals across the whole portfolio.

Unlike ``trade_memory.py``'s ``TradeMemory`` (scoped to a single Asset-A/B
pair), this tracks every symbol the portfolio evaluates each day, keyed by
``(evaluation_date, symbol)``. One pooled regression is fit across all
symbols' history rather than a model per symbol -- a single symbol's dip
signals are too sparse to fit reliably on their own, while pooling lets every
qualifying symbol's daily observation contribute to the same model.
"""

import math
from pathlib import Path

import duckdb

from trade_memory import RotationForecast
from market_sessions import is_next_trading_session
from ridge_regression import fit_two_feature_ridge


class PortfolioMemory:
    """Record per-symbol dip observations and forecast next-session return."""

    def __init__(self, database_path: Path, minimum_observations: int, maximum_observations: int):
        self.database_path = database_path
        self.minimum_observations = minimum_observations
        self.maximum_observations = maximum_observations

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
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            self._settle_prior_observations(conn, symbol, evaluation_date, price)
            conn.execute(
                """
                INSERT INTO observations
                    (evaluation_date, symbol, price, dip_percent, news_score, signal_present,
                     live_spread_percent, recent_avg_volume, historical_expected_profit,
                     historical_win_probability, historical_return_stdev)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_date, symbol) DO NOTHING
                """,
                (
                    evaluation_date,
                    symbol,
                    price,
                    dip_percent,
                    news_score,
                    int(signal_present),
                    live_spread_percent,
                    recent_avg_volume,
                    historical_expected_profit,
                    historical_win_probability,
                    historical_return_stdev,
                ),
            )
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
        return self._fit(list(reversed(rows)), dip_percent, news_score)

    def backfill_history(self, symbol: str, rows: list[tuple[str, float, float]]) -> int:
        """Import completed daily (date, dip, next_session_return) rows for one symbol.

        Deliberately price-only, like ``TradeMemory.backfill_history``:
        inventing historic news scores would make the learned relationship
        look more certain than it is.
        """
        if not rows:
            return 0
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            before = self._observation_count(conn)
            for date, dip, next_return in rows:
                if not math.isfinite(dip) or not math.isfinite(next_return):
                    continue
                conn.execute(
                    """
                    INSERT INTO observations
                        (evaluation_date, symbol, price, dip_percent, news_score, next_session_return_percent)
                    VALUES (?, ?, NULL, ?, NULL, ?)
                    ON CONFLICT(evaluation_date, symbol) DO NOTHING
                    """,
                    (date, symbol, dip, max(-25.0, min(25.0, next_return))),
                )
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

    @staticmethod
    def _settle_prior_observations(
        conn: duckdb.DuckDBPyConnection, symbol: str, date: str, price: float
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
            if not is_next_trading_session(str(prior_date), date):
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

    @staticmethod
    def _observation_count(conn: duckdb.DuckDBPyConnection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def _fit(self, rows: list[tuple[float, int | None, float]], dip: float, score: int | None) -> RotationForecast:
        count = len(rows)
        if count < self.minimum_observations:
            return RotationForecast(
                count, False, None, None,
                f"Portfolio memory is warming up: {count}/{self.minimum_observations} pooled dip signals settled.",
            )
        # Same two-feature, ridge-stabilized regression as TradeMemory._fit,
        # pooled across every symbol's history rather than one A/B pair.
        fit = fit_two_feature_ridge(rows, dip, score)
        if fit is None:
            return RotationForecast(count, False, None, None, "Portfolio memory lacks enough feature variation for a trustworthy forecast.")
        predicted, correlation = fit
        return RotationForecast(
            count, True, predicted, correlation,
            f"Portfolio memory used {count} pooled dip signals; predicted next-session return {predicted:+.2f}% (fit correlation {correlation:+.2f}).",
        )
