"""Small, cached helpers for reasoning about NYSE trading sessions."""

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import pandas_market_calendars as market_calendars

_NYSE = market_calendars.get_calendar("NYSE")

# Keyed by calendar date; value is (market_open, market_close) as tz-aware
# datetimes, or None for a day with no session (weekend/holiday). Populated
# lazily by nyse_is_open() and pruned to the current day so it never grows
# unbounded across a long-running process.
_session_cache: dict[date, tuple[datetime, datetime] | None] = {}


@lru_cache(maxsize=512)
def is_next_trading_session(prior_date: str, current_date: str) -> bool:
    """Return whether ``current_date`` is the first NYSE session after ``prior_date``."""
    try:
        prior = date.fromisoformat(prior_date)
        current = date.fromisoformat(current_date)
    except (TypeError, ValueError):
        return False
    if current <= prior:
        return False

    sessions = _NYSE.valid_days(
        start_date=prior,
        end_date=current,
    )
    session_dates = [timestamp.date() for timestamp in sessions]
    return session_dates == [prior, current]


def is_next_calendar_day(prior_date: str, current_date: str) -> bool:
    """Return whether ``current_date`` is the calendar day right after ``prior_date``.

    PortfolioMemory's settlement step defaults to is_next_trading_session,
    which is correct for equities (a next-session return can only be measured
    against the next NYSE trading day) but wrong for crypto: crypto trades
    every calendar day, including weekends and NYSE holidays, so skipping to
    the next NYSE session would silently span multiple days of crypto price
    action as if it were a single next-day return. Used as
    CryptoRotationStrategy's next_session_predicate instead.
    """
    try:
        prior = date.fromisoformat(prior_date)
        current = date.fromisoformat(current_date)
    except (TypeError, ValueError):
        return False
    return current - prior == timedelta(days=1)


def nyse_is_open(now_utc: datetime) -> bool:
    """Return whether NYSE regular trading hours are open at ``now_utc``.

    Used by the crypto strategy to gate trading to NYSE-closed hours. Caches
    the day's open/close pair keyed by date and drops stale-day entries, the
    same discipline MarketOpenLoggingAlpaca.market_hours (main.py) uses to
    avoid rebuilding the NYSE holiday calendar from scratch on every poll --
    that rebuild is what pegged a Pi core at ~33% CPU before it was cached.
    Unlike that helper this needs no broker/Alpaca credential: it is pure
    pandas_market_calendars, so it can run in the crypto process's tight
    polling loop for free.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    effective_date = now_utc.date()
    for stale_date in [cached_date for cached_date in _session_cache if cached_date != effective_date]:
        del _session_cache[stale_date]
    if effective_date not in _session_cache:
        schedule = _NYSE.schedule(start_date=effective_date, end_date=effective_date)
        if schedule.empty:
            _session_cache[effective_date] = None
        else:
            market_open = schedule.iloc[0]["market_open"].to_pydatetime().astimezone(timezone.utc)
            market_close = schedule.iloc[0]["market_close"].to_pydatetime().astimezone(timezone.utc)
            _session_cache[effective_date] = (market_open, market_close)
    session = _session_cache[effective_date]
    if session is None:
        return False
    market_open, market_close = session
    return market_open <= now_utc <= market_close
