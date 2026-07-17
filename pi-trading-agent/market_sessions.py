"""Small, cached helpers for reasoning about NYSE trading sessions."""

from datetime import date
from functools import lru_cache

import pandas_market_calendars as market_calendars

_NYSE = market_calendars.get_calendar("NYSE")


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
