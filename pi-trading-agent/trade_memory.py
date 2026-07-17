"""Durable, explainable memory for evaluating rotation decisions.

The database deliberately contains only market observations and decisions.  It
never stores API credentials, account balances, or order identifiers.
"""

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import duckdb

from market_sessions import is_next_trading_session

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

    def __init__(self, database_path: Path, minimum_observations: int, maximum_observations: int):
        self.database_path = database_path
        self.minimum_observations = minimum_observations
        self.maximum_observations = maximum_observations

    def update_and_forecast(
        self,
        evaluation_date: str,
        price_a: float,
        price_b: float,
        dip_percent: float,
        news_score: int | None,
        signal_present: bool,
    ) -> RotationForecast:
        """Settle prior observations, retain today's snapshot, and forecast.

        The target is the next-session relative return of B versus A, which is
        the quantity a rotation is trying to improve.  Only prior *dip signal*
        observations train the model, preventing ordinary market days from
        diluting the decision-specific evidence.
        """
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            self._migrate_legacy_sqlite(conn)
            self._settle_prior_observations(conn, evaluation_date, price_a, price_b)
            conn.execute(
                """
                INSERT INTO observations
                    (evaluation_date, price_a, price_b, dip_percent, news_score, signal_present)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_date) DO NOTHING
                """,
                (evaluation_date, price_a, price_b, dip_percent, news_score, int(signal_present)),
            )
            conn.commit()
            rows = conn.execute(
                """
                SELECT dip_percent, news_score, relative_return_percent
                FROM observations
                WHERE signal_present = 1 AND relative_return_percent IS NOT NULL
                ORDER BY evaluation_date DESC LIMIT ?
                """,
                (self.maximum_observations,),
            ).fetchall()
        return self._fit(list(reversed(rows)), dip_percent, news_score)

    def backfill_history(
        self,
        rows: list[tuple[str, float, float, float, bool]],
    ) -> int:
        """Import completed daily observations without changing existing records.

        Each row is ``(date, asset_a_close, asset_b_close, dip, signal)``.
        The following row supplies the already-known next-session outcome, so
        only rows with a subsequent valid close are stored as settled.  This is
        deliberately price-only: inventing historic news scores would make the
        learned relationship look more certain than it is.
        """
        if len(rows) < 2:
            return 0
        inserted = 0
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            self._migrate_legacy_sqlite(conn)
            before = self._observation_count(conn)
            for (date, price_a, price_b, dip, signal), (_, next_a, next_b, _, _) in zip(
                rows, rows[1:]
            ):
                if min(price_a, price_b, next_a, next_b) <= 0:
                    continue
                edge = ((next_b - price_b) / price_b - (next_a - price_a) / price_a) * 100.0
                if not math.isfinite(edge):
                    continue
                conn.execute(
                    """
                    INSERT INTO observations
                        (evaluation_date, price_a, price_b, dip_percent, news_score,
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
        with self._connect() as conn:
            self._create_schema(conn)
            self._migrate_legacy_sqlite(conn)
            conn.execute(
                "UPDATE observations SET decision = ?, decision_reason = ? WHERE evaluation_date = ?",
                (decision[:40], reason[:500], evaluation_date),
            )
            conn.commit()

    def record_execution(self, evaluation_date: str, symbol: str, side: str, price: float, quantity: float) -> None:
        """Keep an immutable local record of broker-confirmed fills."""
        with self._connect() as conn:
            self._create_schema(conn)
            self._migrate_legacy_sqlite(conn)
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
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            self._migrate_legacy_sqlite(conn)
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
                news_score INTEGER,
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_state (
                name TEXT PRIMARY KEY
            )
            """
        )

    @staticmethod
    def _settle_prior_observations(conn: duckdb.DuckDBPyConnection, date: str, price_a: float, price_b: float) -> None:
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
            if not is_next_trading_session(str(prior_date), date):
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

    @staticmethod
    def _observation_count(conn: duckdb.DuckDBPyConnection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def _migrate_legacy_sqlite(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Copy the old journal once when upgrading an existing installation.

        This uses Python's built-in SQLite reader rather than DuckDB extensions,
        so it works on the Pi without fetching or installing an extension.
        """
        legacy_path = self.database_path.with_suffix(".sqlite3")
        migrated = conn.execute(
            "SELECT 1 FROM migration_state WHERE name = 'sqlite_to_duckdb'"
        ).fetchone()
        if migrated or not legacy_path.is_file():
            return
        try:
            with sqlite3.connect(legacy_path) as legacy:
                observations = legacy.execute(
                    """
                    SELECT evaluation_date, price_a, price_b, dip_percent, news_score,
                           signal_present, decision, decision_reason, relative_return_percent
                    FROM observations
                    """
                ).fetchall()
                executions = legacy.execute(
                    "SELECT id, evaluation_date, symbol, side, price, quantity FROM executions"
                ).fetchall()
        except sqlite3.Error:
            # A malformed or incompatible old file must not stop trading. The
            # new database remains usable and can be reviewed independently.
            return
        if observations:
            conn.executemany(
                """
                INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_date) DO NOTHING
                """,
                observations,
            )
        if executions:
            conn.executemany(
                """
                INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                executions,
            )
        conn.execute("INSERT INTO migration_state VALUES ('sqlite_to_duckdb')")
        conn.commit()

    def _fit(self, rows: list[tuple[float, int | None, float]], dip: float, score: int | None) -> RotationForecast:
        count = len(rows)
        if count < self.minimum_observations:
            return RotationForecast(
                count, False, None, None,
                f"Decision memory is warming up: {count}/{self.minimum_observations} comparable dip signals settled.",
            )
        # Use a two-feature, ridge-stabilized linear regression implemented
        # directly to keep this Pi service dependency-free and auditable.
        xs = [(float(row[0]), float(row[1] if row[1] is not None else 0)) for row in rows]
        ys = [float(row[2]) for row in rows]
        mean_x = [sum(values[index] for values in xs) / count for index in range(2)]
        mean_y = sum(ys) / count
        centered = [[values[index] - mean_x[index] for index in range(2)] for values in xs]
        # Small ridge penalty makes a nearly constant news score harmless.
        a = sum(row[0] * row[0] for row in centered) + 1.0
        b = sum(row[0] * row[1] for row in centered)
        d = sum(row[1] * row[1] for row in centered) + 1.0
        determinant = a * d - b * b
        if determinant <= 1e-9:
            return RotationForecast(count, False, None, None, "Decision memory lacks enough feature variation for a trustworthy forecast.")
        target = [sum(row[index] * (y - mean_y) for row, y in zip(centered, ys)) for index in range(2)]
        beta_dip = (d * target[0] - b * target[1]) / determinant
        beta_news = (a * target[1] - b * target[0]) / determinant
        predicted = mean_y + beta_dip * (dip - mean_x[0]) + beta_news * ((score or 0) - mean_x[1])
        predicted = max(-10.0, min(10.0, predicted))
        fitted = [mean_y + beta_dip * row[0] + beta_news * row[1] for row in centered]
        mean_fit = sum(fitted) / count
        variance_y = sum((value - mean_y) ** 2 for value in ys)
        variance_fit = sum((value - mean_fit) ** 2 for value in fitted)
        covariance = sum((y - mean_y) * (fit - mean_fit) for y, fit in zip(ys, fitted))
        correlation = covariance / math.sqrt(variance_y * variance_fit) if variance_y > 0 and variance_fit > 0 else 0.0
        return RotationForecast(
            count, True, predicted, correlation,
            f"Decision memory used {count} prior dip signals; predicted next-session B-minus-A edge {predicted:+.2f}% (fit correlation {correlation:+.2f}).",
        )
