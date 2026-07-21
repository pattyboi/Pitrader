"""Durable, explainable memory for evaluating rotation decisions.

The database deliberately contains only market observations and decisions.  It
never stores API credentials, account balances, or order identifiers.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb

from market_sessions import is_next_trading_session
from ridge_regression import fit_two_feature_ridge

@dataclass
class RotationForecast:
    observations: int
    ready: bool
    predicted_edge_percent: float | None
    correlation: float | None
    explanation: str


@dataclass
class OpportunityProbability:
    """Historical chance that a prior A-to-B dip rotation beat holding A."""

    observations: int
    wins: int
    probability: float | None


class TradeMemory:
    """Record decisions and estimate next-session Asset-B minus Asset-A edge."""

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
        # Defaults to NYSE-trading-session succession (equity's behavior,
        # unchanged). CryptoRotationStrategy passes
        # market_sessions.is_next_calendar_day instead, since crypto trades
        # every calendar day rather than skipping weekends/holidays.
        self._next_session_predicate = next_session_predicate
        self._schema_initialized = False

    def update_and_forecast(
        self,
        evaluation_date: str,
        price_a: float,
        price_b: float,
        dip_percent: float,
        llm_score: int | None,
        signal_present: bool,
    ) -> RotationForecast:
        """Settle prior observations, retain today's snapshot, and forecast.

        The target is the next-session relative return of B versus A, which is
        the quantity a rotation is trying to improve.  Only prior *dip signal*
        observations train the model, preventing ordinary market days from
        diluting the decision-specific evidence.
        """
        with self._open() as conn:
            self._settle_prior_observations_with_predicate(conn, evaluation_date, price_a, price_b)
            conn.execute(
                """
                INSERT INTO observations
                    (evaluation_date, price_a, price_b, dip_percent, llm_score, signal_present)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_date) DO NOTHING
                """,
                (evaluation_date, price_a, price_b, dip_percent, llm_score, int(signal_present)),
            )
            conn.commit()
            rows = conn.execute(
                """
                SELECT dip_percent, llm_score, relative_return_percent
                FROM observations
                WHERE signal_present = 1 AND relative_return_percent IS NOT NULL
                ORDER BY evaluation_date DESC LIMIT ?
                """,
                (self.maximum_observations,),
            ).fetchall()
        return self._fit(list(reversed(rows)), dip_percent, llm_score)

    def backfill_history(
        self,
        rows: list[tuple[str, float, float, float, bool]],
    ) -> int:
        """Import completed daily observations without changing existing records.

        Each row is ``(date, asset_a_close, asset_b_close, dip, signal)``.
        The following row supplies the already-known next-session outcome, so
        only rows with a subsequent valid close are stored as settled.  This is
        deliberately price-only: inventing historic LLM scores would make the
        learned relationship look more certain than it is.
        """
        if len(rows) < 2:
            return 0
        inserted = 0
        with self._open() as conn:
            before = self._observation_count(conn)
            for (date, price_a, price_b, dip, signal), (next_date, next_a, next_b, _, _) in zip(
                rows, rows[1:]
            ):
                # List-adjacency isn't session-adjacency: a source data gap
                # for just one of the two assets (a halt, delisting, missing
                # bar) can silently skip a date out of `rows` upstream, which
                # would otherwise pair two non-consecutive sessions here and
                # record a multi-day return as if it were a single session's.
                if not self._next_session_predicate(str(date), str(next_date)):
                    continue
                if min(price_a, price_b, next_a, next_b) <= 0:
                    continue
                edge = ((next_b - price_b) / price_b - (next_a - price_a) / price_a) * 100.0
                if not math.isfinite(edge):
                    continue
                conn.execute(
                    """
                    INSERT INTO observations
                        (evaluation_date, price_a, price_b, dip_percent, llm_score,
                         signal_present, relative_return_percent)
                    VALUES (?, ?, ?, ?, NULL, ?, ?)
                    ON CONFLICT(evaluation_date) DO NOTHING
                    """,
                    (
                        date,
                        price_a,
                        price_b,
                        dip,
                        int(signal),
                        max(-25.0, min(25.0, edge)),
                    ),
                )
            inserted = self._observation_count(conn) - before
            conn.commit()
        return inserted

    def record_decision(self, evaluation_date: str, decision: str, reason: str) -> None:
        """Attach the final decision to today's already-recorded snapshot."""
        with self._open() as conn:
            conn.execute(
                "UPDATE observations SET decision = ?, decision_reason = ? WHERE evaluation_date = ?",
                (decision[:40], reason[:500], evaluation_date),
            )
            conn.commit()

    def record_execution(self, evaluation_date: str, symbol: str, side: str, price: float, quantity: float) -> None:
        """Keep an immutable local record of broker-confirmed fills."""
        with self._open() as conn:
            conn.execute(
                """
                INSERT INTO executions (id, evaluation_date, symbol, side, price, quantity)
                VALUES ((SELECT COALESCE(MAX(id), 0) + 1 FROM executions), ?, ?, ?, ?, ?)
                """,
                (evaluation_date, symbol[:32], side[:12], price, quantity),
            )
            conn.commit()

    def opportunity_probability(self) -> OpportunityProbability:
        """Return a smoothed win probability using settled, prior dip signals.

        The estimate is (wins + 1) / (observations + 2), a simple Laplace
        correction that avoids claiming 0% or 100% certainty from a small
        sample. It only reads outcomes recorded before the current decision.
        """
        with self._open() as conn:
            observations, wins = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(CASE WHEN relative_return_percent > 0 THEN 1 ELSE 0 END), 0)
                FROM (
                    SELECT relative_return_percent
                    FROM observations
                    WHERE signal_present = 1 AND relative_return_percent IS NOT NULL
                    ORDER BY evaluation_date DESC
                    LIMIT ?
                )
                """,
                (self.maximum_observations,),
            ).fetchone()
        observations, wins = int(observations), int(wins)
        return OpportunityProbability(
            observations=observations,
            wins=wins,
            probability=(wins + 1) / (observations + 2) if observations else None,
        )

    @staticmethod
    def _create_schema(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                evaluation_date TEXT PRIMARY KEY,
                price_a REAL NOT NULL,
                price_b REAL NOT NULL,
                dip_percent REAL NOT NULL,
                llm_score INTEGER,
                signal_present INTEGER NOT NULL,
                decision TEXT,
                decision_reason TEXT,
                relative_return_percent REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                id BIGINT PRIMARY KEY,
                evaluation_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL
            )
            """
        )
        existing_columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'observations'"
            ).fetchall()
        }
        if "llm_score" not in existing_columns:
            conn.execute("ALTER TABLE observations ADD COLUMN llm_score INTEGER")

    def _settle_prior_observations_with_predicate(
        self, conn: duckdb.DuckDBPyConnection, date: str, price_a: float, price_b: float
    ) -> None:
        rows = conn.execute(
            """
            SELECT evaluation_date, price_a, price_b FROM observations
            WHERE evaluation_date < ?
              AND relative_return_percent IS NULL
            ORDER BY evaluation_date DESC LIMIT 1
            """,
            (date,),
        ).fetchall()
        for prior_date, prior_a, prior_b in rows:
            if not self._next_session_predicate(str(prior_date), date):
                continue
            if prior_a <= 0 or prior_b <= 0:
                continue
            return_a = ((price_a - prior_a) / prior_a) * 100.0
            return_b = ((price_b - prior_b) / prior_b) * 100.0
            edge = return_b - return_a
            # Check finiteness before clamping: min/max silently turn NaN
            # into the clamp bound, which would record a fabricated edge.
            if math.isfinite(edge):
                conn.execute(
                    "UPDATE observations SET relative_return_percent = ? WHERE evaluation_date = ?",
                    (max(-25.0, min(25.0, edge)), prior_date),
                )

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open the host-native analytical store used for decision memory."""
        return duckdb.connect(str(self.database_path))

    @contextmanager
    def _open(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Ensure the directory/schema exist, then hand back a ready
        connection -- shared prologue for every public method below. Schema
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
    def _observation_count(conn: duckdb.DuckDBPyConnection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def _fit(self, rows: list[tuple[float, int | None, float]], dip: float, score: int | None) -> RotationForecast:
        count = len(rows)
        if count < self.minimum_observations:
            return RotationForecast(
                count, False, None, None,
                f"Decision memory is warming up: {count}/{self.minimum_observations} comparable dip signals settled.",
            )
        fit = fit_two_feature_ridge(rows, dip, score)
        if fit is None:
            return RotationForecast(count, False, None, None, "Decision memory lacks enough feature variation for a trustworthy forecast.")
        predicted, correlation = fit
        return RotationForecast(
            count, True, predicted, correlation,
            f"Decision memory used {count} prior dip signals; predicted next-session B-minus-A edge {predicted:+.2f}% (fit correlation {correlation:+.2f}).",
        )
