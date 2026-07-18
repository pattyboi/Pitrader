"""Shared once-only migration bookkeeping for the DuckDB memory modules
(`trade_memory.py`'s sqlite migration, `autonomous_universe.py`'s JSON
migration). Only the tracking-table plumbing is shared; each module's
actual migration body (different source format, different target tables)
stays local.
"""

import duckdb


def ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS migration_state (name TEXT PRIMARY KEY)")


def already_done(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM migration_state WHERE name = ?", (name,)).fetchone() is not None


def mark_done(conn: duckdb.DuckDBPyConnection, name: str) -> None:
    conn.execute("INSERT INTO migration_state VALUES (?)", (name,))
