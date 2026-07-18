#!/usr/bin/env python3
"""Start the Alpaca-backed Lumibot asset-rotation strategy."""

import json
import logging
import os
import sys
from datetime import datetime, timedelta as datetime_timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from strategy import AssetRotationStrategy


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
EASTERN_TIME = ZoneInfo("America/New_York")


def format_market_open_time(open_time: datetime) -> str:
    """Format a market-calendar timestamp for the market-wait log message."""
    return open_time.astimezone(EASTERN_TIME).strftime("%-I:%M %p ET")


class MarketOpenLoggingAlpaca(Alpaca):
    """Alpaca broker with a more useful pre-market wait message.

    Lumibot normally logs only that it is sleeping.  Keep its wait behavior
    unchanged while including the next calendar-derived market-open time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._market_hours_cache: dict[tuple, Any] = {}

    def market_hours(self, market="NASDAQ", close=True, next=False, date=None):
        """Cache Lumibot's per-call market-calendar recomputation.

        Broker.market_hours() (lumibot/brokers/broker.py) rebuilds the whole
        trading calendar -- including the NYSE holiday list, which pandas
        regenerates via a slow dateutil.relativedelta loop -- from scratch on
        every call, with no memoization upstream. _handle_lifecycle_methods()
        calls this (via market_close_time()) on every tick of Lumibot's live
        loop, which only sleeps 1 second between ticks and runs continuously
        around the clock regardless of this strategy's own "1D" sleeptime.
        That pegged a Pi core at ~33% CPU 24/7 recomputing an answer that
        cannot change within a calendar day. Cache by (effective date,
        market, close, next); entries from a previous day are dropped since
        they're never queried again.
        """
        effective_date = (date if date is not None else datetime.now(timezone.utc)).date()
        cache_key = (effective_date, market, close, next)
        cache = self._market_hours_cache
        for stale_key in [key for key in cache if key[0] != effective_date]:
            del cache[stale_key]
        if cache_key not in cache:
            cache[cache_key] = super().market_hours(market=market, close=close, next=next, date=date)
        return cache[cache_key]

    def _await_market_to_open(self, timedelta=None, strategy=None):
        if self.is_market_open():
            return

        time_to_open = self.get_time_to_open()
        market_open = datetime.now(timezone.utc) + datetime_timedelta(
            seconds=max(0, time_to_open)
        )
        if timedelta is not None:
            time_to_open -= 60 * timedelta

        self.logger.info(
            "Sleeping until the market opens (%s)", format_market_open_time(market_open)
        )
        # Lumibot's own Broker.sleep() is a single uninterruptible time.sleep(),
        # which can block this thread for hours and made a clean process stop
        # impossible (systemd's SIGINT/SIGTERM never got noticed; the service
        # only ever exited via the stop timeout's SIGKILL). Wait on the
        # executor's stop_event instead, in short slices, so a requested stop
        # is picked up within a second instead of at the next market open.
        stop_event = getattr(getattr(strategy, "_executor", None), "stop_event", None)
        remaining = max(0, time_to_open)
        if stop_event is None:
            self.sleep(remaining)
            return
        while remaining > 0 and not stop_event.is_set():
            slice_seconds = min(1.0, remaining)
            stop_event.wait(slice_seconds)
            remaining -= slice_seconds

    def _await_market_to_close(self, timedelta=None, strategy=None):
        """Wait for market close the same interruptible way as market-open.

        Lumibot's own _await_market_to_close() (alpaca.py) ends in a single
        uninterruptible time.sleep(), which can block this thread for hours.
        Fixing the process_pending_orders AttributeError above (see that
        method's docstring) means this method is now actually reached and
        slept in, re-exposing the same clean-stop problem
        _await_market_to_open already had to solve: without this override,
        systemd's TimeoutStopUSec expires and the process gets SIGKILLed
        instead of exiting via the executor's stop_event.
        """
        self.process_pending_orders(strategy=strategy)

        time_to_close = self.get_time_to_close()
        if timedelta is not None:
            time_to_close -= 60 * timedelta
        if time_to_close <= 0:
            return

        self.logger.info(f"Sleeping {time_to_close:.0f} seconds until market close")
        stop_event = getattr(getattr(strategy, "_executor", None), "stop_event", None)
        remaining = max(0, time_to_close)
        if stop_event is None:
            self.sleep(remaining)
            return
        while remaining > 0 and not stop_event.is_set():
            slice_seconds = min(1.0, remaining)
            stop_event.wait(slice_seconds)
            remaining -= slice_seconds

    def process_pending_orders(self, strategy=None) -> None:
        """No-op: this is a backtesting-only concern Lumibot calls unconditionally.

        Lumibot's live Alpaca._await_market_to_close() calls
        self.process_pending_orders(strategy=strategy) every day near market
        close, but that method only exists on BacktestingBroker (it simulates
        fills bar-by-bar there). Live fills already arrive through the
        trade-event stream wired to on_filled_order, so there is nothing to
        do here; without this stub every trading day logs an AttributeError
        crash traceback (caught and harmless, but noisy). Upstream bug:
        https://github.com/Lumiwealth/lumibot/issues/1113
        """


def _require_booleans(config: dict[str, Any], *keys: str) -> None:
    invalid = [key for key in keys if not isinstance(config[key], bool)]
    if invalid:
        raise TypeError(f"{', '.join(invalid)} must be true or false")


def _require_range(
    name: str,
    value: float,
    minimum: float,
    maximum: float,
    *,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> None:
    lower_ok = value >= minimum if minimum_inclusive else value > minimum
    upper_ok = value <= maximum if maximum_inclusive else value < maximum
    if not lower_ok or not upper_ok:
        lower = "at least" if minimum_inclusive else "greater than"
        upper = "at most" if maximum_inclusive else "less than"
        raise ValueError(f"{name} must be {lower} {minimum:g} and {upper} {maximum:g}")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate configuration before connecting to the broker."""
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read valid JSON from {path}: {exc}") from exc

    required = {
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "IS_PAPER_TRADING",
        "ASSET_A",
        "ASSET_B",
        "DIP_THRESHOLD_PERCENT",
        "RECENT_HIGH_LOOKBACK_DAYS",
        "EMAIL_REPORT_ENABLED",
        "EMAIL_SMTP_HOST",
        "EMAIL_SMTP_PORT",
        "EMAIL_SMTP_USERNAME",
        "EMAIL_SMTP_PASSWORD",
        "EMAIL_FROM_ADDRESS",
        "EMAIL_TO_ADDRESS",
        "EMAIL_USE_TLS",
        "NEWS_CONTEXT_ENABLED",
        "NEWS_LOOKBACK_HOURS",
        "NEWS_MAX_ARTICLES",
        "NEWS_BLOCK_ON_HIGH_RISK",
        "NEWS_HIGH_RISK_SCORE",
        "NEWS_LEARNING_ENABLED",
        "NEWS_LEARNING_BLOCK_ENABLED",
        "NEWS_LEARNING_MIN_OBSERVATIONS",
        "NEWS_LEARNING_MAX_OBSERVATIONS",
        "NEWS_LEARNING_MIN_CORRELATION",
        "NEWS_PREDICTED_RETURN_BLOCK_PERCENT",
        "LLM_NEWS_ENABLED",
        "LLM_NEWS_MODEL",
        "LLM_NEWS_BASE_URL",
        "LLM_NEWS_BLOCK_ON_HIGH_RISK",
        "LLM_NEWS_BLOCK_SCORE",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")

    if config["ALPACA_API_KEY"].startswith("REPLACE_WITH_") or config[
        "ALPACA_SECRET_KEY"
    ].startswith("REPLACE_WITH_"):
        raise ValueError("Replace the Alpaca credential placeholders in config.json")
    _require_booleans(config, "IS_PAPER_TRADING")

    asset_a = str(config["ASSET_A"]).strip().upper()
    asset_b = str(config["ASSET_B"]).strip().upper()
    if not asset_a or not asset_b or asset_a == asset_b:
        raise ValueError("ASSET_A and ASSET_B must be different, non-empty symbols")

    dip_threshold = float(config["DIP_THRESHOLD_PERCENT"])
    lookback_days = int(config["RECENT_HIGH_LOOKBACK_DAYS"])
    _require_range(
        "DIP_THRESHOLD_PERCENT", dip_threshold, 0.0, 100.0,
        minimum_inclusive=False, maximum_inclusive=False,
    )
    _require_range("RECENT_HIGH_LOOKBACK_DAYS", lookback_days, 2, float("inf"))

    # Autonomous discovery, when enabled, expands the configured seed
    # watchlist through a bounded scan.
    portfolio_defaults = {
        "PORTFOLIO_SYMBOLS": [asset_a, asset_b],
        "PORTFOLIO_MAX_POSITIONS": 1,
        "PORTFOLIO_ANALYSIS_DAYS": 252,
        "PORTFOLIO_MIN_SIGNAL_OBSERVATIONS": 20,
        "PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT": 1.0,
        "PORTFOLIO_OOS_MIN_OBSERVATIONS": 10,
        "PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT": 0.0,
        "PORTFOLIO_ROUND_TRIP_COST_PERCENT": 0.20,
        "PORTFOLIO_TAKE_PROFIT_PERCENT": 1.0,
        "PORTFOLIO_STOP_LOSS_PERCENT": 0.5,
        "PORTFOLIO_HOLDING_HORIZON_MAX_DAYS": 15,
        "PORTFOLIO_AUTONOMOUS_DISCOVERY": False,
        "PORTFOLIO_DISCOVERY_BATCH_SIZE": 12,
        "PORTFOLIO_DISCOVERY_REFRESH_DAYS": 7,
        "PORTFOLIO_DISCOVERY_MIN_PRICE_DOLLARS": 5.0,
        "PORTFOLIO_DISCOVERY_MIN_AVG_VOLUME": 100000,
        "PORTFOLIO_FRACTIONAL_SHARES": True,
        "PORTFOLIO_CASH_RESERVE_DOLLARS": 2.0,
        "PORTFOLIO_MIN_ORDER_DOLLARS": 5.0,
        "PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY": 0.55,
        "PORTFOLIO_RISK_POSTURE": "conservative",
        "PORTFOLIO_DISCOVERY_LLM_BLOCK_ENABLED": False,
        "PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES": 210,
        "PORTFOLIO_NIGHTLY_PREEVAL_ENABLED": True,
    }
    for key, default in portfolio_defaults.items():
        config.setdefault(key, default)
    raw_symbols = config["PORTFOLIO_SYMBOLS"]
    if not isinstance(raw_symbols, list):
        raise TypeError("PORTFOLIO_SYMBOLS must be a JSON array of symbols")
    portfolio_symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()))
    if not portfolio_symbols:
        raise ValueError("PORTFOLIO_SYMBOLS must contain at least one symbol")
    if len(portfolio_symbols) > 30:
        raise ValueError("PORTFOLIO_SYMBOLS may contain at most 30 symbols")
    portfolio_max_positions = int(config["PORTFOLIO_MAX_POSITIONS"])
    portfolio_analysis_days = int(config["PORTFOLIO_ANALYSIS_DAYS"])
    portfolio_min_observations = int(config["PORTFOLIO_MIN_SIGNAL_OBSERVATIONS"])
    portfolio_min_profit = float(config["PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT"])
    portfolio_oos_min_observations = int(config["PORTFOLIO_OOS_MIN_OBSERVATIONS"])
    portfolio_oos_min_profit = float(config["PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT"])
    portfolio_round_trip_cost = float(config["PORTFOLIO_ROUND_TRIP_COST_PERCENT"])
    portfolio_take_profit_percent = float(config["PORTFOLIO_TAKE_PROFIT_PERCENT"])
    portfolio_stop_loss_percent = float(config["PORTFOLIO_STOP_LOSS_PERCENT"])
    portfolio_holding_horizon_max_days = int(config["PORTFOLIO_HOLDING_HORIZON_MAX_DAYS"])
    _require_booleans(config, "PORTFOLIO_AUTONOMOUS_DISCOVERY")
    discovery_batch_size = int(config["PORTFOLIO_DISCOVERY_BATCH_SIZE"])
    discovery_refresh_days = int(config["PORTFOLIO_DISCOVERY_REFRESH_DAYS"])
    discovery_min_price = float(config["PORTFOLIO_DISCOVERY_MIN_PRICE_DOLLARS"])
    discovery_min_avg_volume = float(config["PORTFOLIO_DISCOVERY_MIN_AVG_VOLUME"])
    _require_booleans(config, "PORTFOLIO_FRACTIONAL_SHARES")
    portfolio_cash_reserve = float(config["PORTFOLIO_CASH_RESERVE_DOLLARS"])
    portfolio_min_order = float(config["PORTFOLIO_MIN_ORDER_DOLLARS"])
    opportunity_min_probability = float(config["PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY"])
    second_iteration_offset_minutes = int(config["PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES"])
    # Autonomous discovery expands the daily candidate universe beyond the
    # static seed list, so the cap on concurrent positions shouldn't be tied
    # to len(portfolio_symbols) when discovery can supply the rest. Without
    # discovery, the static list is the only source of candidates, so the
    # original bound still applies.
    portfolio_max_positions_ceiling = (
        30 if config["PORTFOLIO_AUTONOMOUS_DISCOVERY"] else len(portfolio_symbols)
    )
    range_checks = (
        ("PORTFOLIO_MAX_POSITIONS", portfolio_max_positions, 1, portfolio_max_positions_ceiling),
        ("PORTFOLIO_ANALYSIS_DAYS", portfolio_analysis_days, 30, 2000),
        ("PORTFOLIO_MIN_SIGNAL_OBSERVATIONS", portfolio_min_observations, 5, 500),
        ("PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT", portfolio_min_profit, 0, 100),
        ("PORTFOLIO_OOS_MIN_OBSERVATIONS", portfolio_oos_min_observations, 5, 500),
        ("PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT", portfolio_oos_min_profit, 0, 100),
        ("PORTFOLIO_ROUND_TRIP_COST_PERCENT", portfolio_round_trip_cost, 0, 10),
        ("PORTFOLIO_TAKE_PROFIT_PERCENT", portfolio_take_profit_percent, 0.05, 100),
        ("PORTFOLIO_STOP_LOSS_PERCENT", portfolio_stop_loss_percent, 0.05, 100),
        ("PORTFOLIO_HOLDING_HORIZON_MAX_DAYS", portfolio_holding_horizon_max_days, 1, 60),
        ("PORTFOLIO_DISCOVERY_BATCH_SIZE", discovery_batch_size, 1, 30),
        ("PORTFOLIO_DISCOVERY_REFRESH_DAYS", discovery_refresh_days, 1, 90),
        ("PORTFOLIO_DISCOVERY_MIN_PRICE_DOLLARS", discovery_min_price, 0, 1000),
        ("PORTFOLIO_DISCOVERY_MIN_AVG_VOLUME", discovery_min_avg_volume, 0, 100_000_000),
        ("PORTFOLIO_CASH_RESERVE_DOLLARS", portfolio_cash_reserve, 0, 1000),
        ("PORTFOLIO_MIN_ORDER_DOLLARS", portfolio_min_order, 1, 1000),
        ("PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY", opportunity_min_probability, 0.5, 0.95),
        # Minutes after market open for the day's second evaluation. Lower
        # bound keeps it a meaningfully later, independent read rather than
        # a near-duplicate of the open window; upper bound keeps it inside
        # even a shortened trading session.
        ("PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES", second_iteration_offset_minutes, 30, 360),
    )
    for name, value, minimum, maximum in range_checks:
        _require_range(name, value, minimum, maximum)
    risk_posture = str(config["PORTFOLIO_RISK_POSTURE"]).strip().lower()
    if risk_posture not in ("conservative", "risky"):
        raise ValueError("PORTFOLIO_RISK_POSTURE must be conservative or risky")
    _require_booleans(
        config, "PORTFOLIO_DISCOVERY_LLM_BLOCK_ENABLED", "PORTFOLIO_NIGHTLY_PREEVAL_ENABLED"
    )
    config["PORTFOLIO_RISK_POSTURE"] = risk_posture
    config["PORTFOLIO_SYMBOLS"] = portfolio_symbols
    config["PORTFOLIO_MAX_POSITIONS"] = portfolio_max_positions
    config["PORTFOLIO_ANALYSIS_DAYS"] = portfolio_analysis_days
    config["PORTFOLIO_MIN_SIGNAL_OBSERVATIONS"] = portfolio_min_observations
    config["PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT"] = portfolio_min_profit
    config["PORTFOLIO_OOS_MIN_OBSERVATIONS"] = portfolio_oos_min_observations
    config["PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT"] = portfolio_oos_min_profit
    config["PORTFOLIO_ROUND_TRIP_COST_PERCENT"] = portfolio_round_trip_cost
    config["PORTFOLIO_TAKE_PROFIT_PERCENT"] = portfolio_take_profit_percent
    config["PORTFOLIO_STOP_LOSS_PERCENT"] = portfolio_stop_loss_percent
    config["PORTFOLIO_HOLDING_HORIZON_MAX_DAYS"] = portfolio_holding_horizon_max_days
    config["PORTFOLIO_DISCOVERY_BATCH_SIZE"] = discovery_batch_size
    config["PORTFOLIO_DISCOVERY_REFRESH_DAYS"] = discovery_refresh_days
    config["PORTFOLIO_DISCOVERY_MIN_PRICE_DOLLARS"] = discovery_min_price
    config["PORTFOLIO_DISCOVERY_MIN_AVG_VOLUME"] = discovery_min_avg_volume
    config["PORTFOLIO_CASH_RESERVE_DOLLARS"] = portfolio_cash_reserve
    config["PORTFOLIO_MIN_ORDER_DOLLARS"] = portfolio_min_order
    config["PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY"] = opportunity_min_probability
    config["PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES"] = second_iteration_offset_minutes

    _require_booleans(config, "EMAIL_REPORT_ENABLED", "EMAIL_USE_TLS")
    email_port = int(config["EMAIL_SMTP_PORT"])
    _require_range("EMAIL_SMTP_PORT", email_port, 1, 65535)
    if config["EMAIL_REPORT_ENABLED"]:
        email_fields = (
            "EMAIL_SMTP_HOST",
            "EMAIL_SMTP_USERNAME",
            "EMAIL_SMTP_PASSWORD",
            "EMAIL_FROM_ADDRESS",
            "EMAIL_TO_ADDRESS",
        )
        invalid_email_fields = [
            field
            for field in email_fields
            if not str(config[field]).strip()
            or str(config[field]).startswith("REPLACE_WITH_")
        ]
        if invalid_email_fields:
            raise ValueError(
                "Email reporting is enabled, but these settings are incomplete: "
                + ", ".join(invalid_email_fields)
            )

    _require_booleans(config, "NEWS_CONTEXT_ENABLED", "NEWS_BLOCK_ON_HIGH_RISK")
    news_lookback = int(config["NEWS_LOOKBACK_HOURS"])
    news_limit = int(config["NEWS_MAX_ARTICLES"])
    news_block_score = int(config["NEWS_HIGH_RISK_SCORE"])
    _require_range("NEWS_LOOKBACK_HOURS", news_lookback, 1, 168)
    _require_range("NEWS_MAX_ARTICLES", news_limit, 1, 50)
    if news_block_score >= 0:
        raise ValueError("NEWS_HIGH_RISK_SCORE must be a negative integer")
    _require_booleans(
        config, "NEWS_LEARNING_ENABLED", "NEWS_LEARNING_BLOCK_ENABLED"
    )
    learning_minimum = int(config["NEWS_LEARNING_MIN_OBSERVATIONS"])
    learning_maximum = int(config["NEWS_LEARNING_MAX_OBSERVATIONS"])
    predicted_return_block = float(config["NEWS_PREDICTED_RETURN_BLOCK_PERCENT"])
    minimum_correlation = float(config["NEWS_LEARNING_MIN_CORRELATION"])
    _require_range("NEWS_LEARNING_MIN_OBSERVATIONS", learning_minimum, 10, 500)
    _require_range(
        "NEWS_LEARNING_MAX_OBSERVATIONS", learning_maximum, learning_minimum, 1000
    )
    _require_range(
        "NEWS_PREDICTED_RETURN_BLOCK_PERCENT", predicted_return_block, -25, 0,
        maximum_inclusive=False,
    )
    _require_range("NEWS_LEARNING_MIN_CORRELATION", minimum_correlation, 0, 1)

    _require_booleans(config, "LLM_NEWS_ENABLED", "LLM_NEWS_BLOCK_ON_HIGH_RISK")
    llm_block_score = int(config["LLM_NEWS_BLOCK_SCORE"])
    _require_range("LLM_NEWS_BLOCK_SCORE", llm_block_score, -10, -1)
    llm_model = str(config["LLM_NEWS_MODEL"]).strip()
    if not llm_model:
        raise ValueError("LLM_NEWS_MODEL must be a non-empty model id")
    llm_base_url = str(config["LLM_NEWS_BASE_URL"]).strip()
    if llm_base_url and not llm_base_url.startswith(("http://", "https://")):
        raise ValueError("LLM_NEWS_BASE_URL must be an http(s) URL or empty")
    config["LLM_NEWS_BLOCK_SCORE"] = llm_block_score
    config["LLM_NEWS_MODEL"] = llm_model
    config["LLM_NEWS_BASE_URL"] = llm_base_url

    config["ASSET_A"] = asset_a
    config["ASSET_B"] = asset_b
    config["DIP_THRESHOLD_PERCENT"] = dip_threshold
    config["RECENT_HIGH_LOOKBACK_DAYS"] = lookback_days
    config["EMAIL_SMTP_PORT"] = email_port
    config["NEWS_LOOKBACK_HOURS"] = news_lookback
    config["NEWS_MAX_ARTICLES"] = news_limit
    config["NEWS_HIGH_RISK_SCORE"] = news_block_score
    config["NEWS_LEARNING_MIN_OBSERVATIONS"] = learning_minimum
    config["NEWS_LEARNING_MAX_OBSERVATIONS"] = learning_maximum
    config["NEWS_PREDICTED_RETURN_BLOCK_PERCENT"] = predicted_return_block
    config["NEWS_LEARNING_MIN_CORRELATION"] = minimum_correlation
    decision_defaults = {
        "DECISION_MEMORY_ENABLED": True,
        "DECISION_MEMORY_BLOCK_ENABLED": False,
        "DECISION_MEMORY_MIN_OBSERVATIONS": 40,
        "DECISION_MEMORY_MAX_OBSERVATIONS": 180,
        "DECISION_MEMORY_MIN_CORRELATION": 0.25,
        "DECISION_MEMORY_EDGE_BLOCK_PERCENT": -0.75,
        "DECISION_MEMORY_BACKFILL_DAYS": 1000,
    }
    for key, default in decision_defaults.items():
        config.setdefault(key, default)
    _require_booleans(
        config, "DECISION_MEMORY_ENABLED", "DECISION_MEMORY_BLOCK_ENABLED"
    )
    decision_minimum = int(config["DECISION_MEMORY_MIN_OBSERVATIONS"])
    decision_maximum = int(config["DECISION_MEMORY_MAX_OBSERVATIONS"])
    decision_correlation = float(config["DECISION_MEMORY_MIN_CORRELATION"])
    decision_edge = float(config["DECISION_MEMORY_EDGE_BLOCK_PERCENT"])
    decision_backfill_days = int(config["DECISION_MEMORY_BACKFILL_DAYS"])
    _require_range("DECISION_MEMORY_MIN_OBSERVATIONS", decision_minimum, 20, 500)
    _require_range(
        "DECISION_MEMORY_MAX_OBSERVATIONS", decision_maximum, decision_minimum, 1000
    )
    _require_range("DECISION_MEMORY_MIN_CORRELATION", decision_correlation, 0, 1)
    _require_range(
        "DECISION_MEMORY_EDGE_BLOCK_PERCENT", decision_edge, -25, 0,
        maximum_inclusive=False,
    )
    _require_range("DECISION_MEMORY_BACKFILL_DAYS", decision_backfill_days, 0, 5000)
    config["DECISION_MEMORY_MIN_OBSERVATIONS"] = decision_minimum
    config["DECISION_MEMORY_MAX_OBSERVATIONS"] = decision_maximum
    config["DECISION_MEMORY_MIN_CORRELATION"] = decision_correlation
    config["DECISION_MEMORY_EDGE_BLOCK_PERCENT"] = decision_edge
    config["DECISION_MEMORY_BACKFILL_DAYS"] = decision_backfill_days

    # Unlike DECISION_MEMORY (scoped to the single Asset-A/B pair), this pools
    # every portfolio symbol's daily dip signal into one model, so it warms up
    # much faster -- the max default is larger to match.
    portfolio_memory_defaults = {
        "PORTFOLIO_MEMORY_ENABLED": True,
        "PORTFOLIO_MEMORY_MIN_OBSERVATIONS": 20,
        "PORTFOLIO_MEMORY_MAX_OBSERVATIONS": 500,
    }
    for key, default in portfolio_memory_defaults.items():
        config.setdefault(key, default)
    _require_booleans(config, "PORTFOLIO_MEMORY_ENABLED")
    portfolio_memory_minimum = int(config["PORTFOLIO_MEMORY_MIN_OBSERVATIONS"])
    portfolio_memory_maximum = int(config["PORTFOLIO_MEMORY_MAX_OBSERVATIONS"])
    _require_range("PORTFOLIO_MEMORY_MIN_OBSERVATIONS", portfolio_memory_minimum, 20, 500)
    _require_range(
        "PORTFOLIO_MEMORY_MAX_OBSERVATIONS",
        portfolio_memory_maximum,
        portfolio_memory_minimum,
        5000,
    )
    config["PORTFOLIO_MEMORY_MIN_OBSERVATIONS"] = portfolio_memory_minimum
    config["PORTFOLIO_MEMORY_MAX_OBSERVATIONS"] = portfolio_memory_maximum

    symbol_reference_defaults = {
        "SYMBOL_REFERENCE_ENABLED": True,
        "SYMBOL_REFERENCE_REFRESH_DAYS": 7,
        "NEWS_SCORE_REFINEMENT_ENABLED": False,
    }
    for key, default in symbol_reference_defaults.items():
        config.setdefault(key, default)
    _require_booleans(
        config, "SYMBOL_REFERENCE_ENABLED", "NEWS_SCORE_REFINEMENT_ENABLED"
    )
    symbol_reference_refresh_days = int(config["SYMBOL_REFERENCE_REFRESH_DAYS"])
    _require_range("SYMBOL_REFERENCE_REFRESH_DAYS", symbol_reference_refresh_days, 1, 30)
    config["SYMBOL_REFERENCE_REFRESH_DAYS"] = symbol_reference_refresh_days

    # Free, no-API-key supplementary headlines (rss_news.py), merged into the
    # same article set news_context.py already builds from Alpaca. Off by
    # default, same posture as every other optional source in this pipeline.
    rss_defaults = {
        "NEWS_RSS_ENABLED": False,
        "NEWS_RSS_FEED_URLS": [
            "https://finance.yahoo.com/news/rssindex",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ],
    }
    for key, default in rss_defaults.items():
        config.setdefault(key, default)
    _require_booleans(config, "NEWS_RSS_ENABLED")
    raw_feed_urls = config["NEWS_RSS_FEED_URLS"]
    if not isinstance(raw_feed_urls, list):
        raise TypeError("NEWS_RSS_FEED_URLS must be a JSON array of feed URLs")
    feed_urls = list(dict.fromkeys(str(url).strip() for url in raw_feed_urls if str(url).strip()))
    invalid_feed_urls = [url for url in feed_urls if not url.startswith(("http://", "https://"))]
    if invalid_feed_urls:
        raise ValueError(f"NEWS_RSS_FEED_URLS entries must be http(s) URLs: {', '.join(invalid_feed_urls)}")
    if len(feed_urls) > 10:
        raise ValueError("NEWS_RSS_FEED_URLS may contain at most 10 feeds")
    config["NEWS_RSS_FEED_URLS"] = feed_urls

    return config


class _DropTelemetry(logging.Filter):
    """Drop Lumibot's five-minute telemetry lines; they bloat the SD-card journal."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "LUMIBOT_TELEMETRY" not in record.getMessage()
        except Exception:
            return True


class _DropOptionalLumiwealthWarning(logging.Filter):
    """Hide the optional Lumiwealth cloud-registration reminder."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "LUMIWEALTH_API_KEY not set." not in record.getMessage()
        except Exception:
            return True


class _DropLumibotDuplicates(logging.Filter):
    """Keep Lumibot records off the root handler; Lumibot's own handler prints them.

    Lumibot attaches a console handler to its "lumibot" logger and re-forces
    propagation on during its own setup, so without this filter every Lumibot
    line lands in the journal twice (Lumibot's format plus the root handler
    installed by basicConfig).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("lumibot")


def _tidy_logging() -> None:
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropLumibotDuplicates())
        handler.addFilter(_DropTelemetry())
    # Logger-level filters run before any handler (Lumibot's included) and
    # survive Lumibot recreating its handlers, so noise is silenced at the
    # source.
    logging.getLogger("lumibot.brokers.broker").addFilter(_DropTelemetry())
    logging.getLogger("lumibot.strategies._strategy").addFilter(
        _DropOptionalLumiwealthWarning()
    )


def build_strategy(
    config: dict[str, Any], base_dir: Path
) -> tuple["MarketOpenLoggingAlpaca", AssetRotationStrategy]:
    """Construct the broker and strategy from validated config, without
    starting the Trader loop. Shared by `main()` (which wraps the result in
    `Trader().add_strategy(...).run_all()`) and `scripts/nightly_preeval.py`
    (which calls a strategy method directly and exits), so the two never
    drift out of sync as config keys are added.
    """
    broker = MarketOpenLoggingAlpaca(
        {
            "API_KEY": config["ALPACA_API_KEY"],
            "API_SECRET": config["ALPACA_SECRET_KEY"],
            "PAPER": config["IS_PAPER_TRADING"],
        }
    )
    strategy = AssetRotationStrategy(
        broker=broker,
        parameters={
            "asset_a": config["ASSET_A"],
            "asset_b": config["ASSET_B"],
            "dip_threshold_percent": config["DIP_THRESHOLD_PERCENT"],
            "recent_high_lookback_days": config["RECENT_HIGH_LOOKBACK_DAYS"],
            "email_report_enabled": config["EMAIL_REPORT_ENABLED"],
            "email_smtp_host": config["EMAIL_SMTP_HOST"],
            "email_smtp_port": config["EMAIL_SMTP_PORT"],
            "email_smtp_username": config["EMAIL_SMTP_USERNAME"],
            "email_from_address": config["EMAIL_FROM_ADDRESS"],
            "email_to_address": config["EMAIL_TO_ADDRESS"],
            "email_use_tls": config["EMAIL_USE_TLS"],
            "email_state_file": str(base_dir / ".last_email_report"),
            "shutdown_diagnostic_file": str(base_dir / ".shutdown_diagnostic.log"),
            "portfolio_symbols": config["PORTFOLIO_SYMBOLS"],
            "portfolio_max_positions": config["PORTFOLIO_MAX_POSITIONS"],
            "portfolio_analysis_days": config["PORTFOLIO_ANALYSIS_DAYS"],
            "portfolio_min_signal_observations": config["PORTFOLIO_MIN_SIGNAL_OBSERVATIONS"],
            "portfolio_min_expected_profit_percent": config["PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT"],
            "portfolio_oos_min_observations": config["PORTFOLIO_OOS_MIN_OBSERVATIONS"],
            "portfolio_oos_min_net_profit_percent": config["PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT"],
            "portfolio_round_trip_cost_percent": config["PORTFOLIO_ROUND_TRIP_COST_PERCENT"],
            "portfolio_take_profit_percent": config["PORTFOLIO_TAKE_PROFIT_PERCENT"],
            "portfolio_stop_loss_percent": config["PORTFOLIO_STOP_LOSS_PERCENT"],
            "portfolio_holding_horizon_max_days": config["PORTFOLIO_HOLDING_HORIZON_MAX_DAYS"],
            "portfolio_holding_state_file": str(base_dir / ".portfolio_holding_state.json"),
            "portfolio_rotation_state_file": str(base_dir / ".portfolio_rotation_state.json"),
            "portfolio_iteration_state_file": str(base_dir / ".portfolio_iteration_state.json"),
            "portfolio_second_iteration_offset_minutes": config[
                "PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES"
            ],
            "portfolio_nightly_preeval_enabled": config["PORTFOLIO_NIGHTLY_PREEVAL_ENABLED"],
            "nightly_preeval_state_file": str(base_dir / ".nightly_preeval_state.json"),
            "portfolio_autonomous_discovery": config["PORTFOLIO_AUTONOMOUS_DISCOVERY"],
            "portfolio_discovery_batch_size": config["PORTFOLIO_DISCOVERY_BATCH_SIZE"],
            "portfolio_discovery_refresh_days": config["PORTFOLIO_DISCOVERY_REFRESH_DAYS"],
            "portfolio_discovery_min_price_dollars": config["PORTFOLIO_DISCOVERY_MIN_PRICE_DOLLARS"],
            "portfolio_discovery_min_avg_volume": config["PORTFOLIO_DISCOVERY_MIN_AVG_VOLUME"],
            "portfolio_universe_database_file": str(base_dir / ".autonomous_universe.duckdb"),
            "fractional_shares": config["PORTFOLIO_FRACTIONAL_SHARES"],
            "portfolio_cash_reserve_dollars": config["PORTFOLIO_CASH_RESERVE_DOLLARS"],
            "portfolio_min_order_dollars": config["PORTFOLIO_MIN_ORDER_DOLLARS"],
            "portfolio_opportunistic_min_probability": config["PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY"],
            "portfolio_risk_posture": config["PORTFOLIO_RISK_POSTURE"],
            "portfolio_discovery_llm_block_enabled": config["PORTFOLIO_DISCOVERY_LLM_BLOCK_ENABLED"],
            "news_context_enabled": config["NEWS_CONTEXT_ENABLED"],
            "news_lookback_hours": config["NEWS_LOOKBACK_HOURS"],
            "news_max_articles": config["NEWS_MAX_ARTICLES"],
            "news_block_on_high_risk": config["NEWS_BLOCK_ON_HIGH_RISK"],
            "news_high_risk_score": config["NEWS_HIGH_RISK_SCORE"],
            "news_learning_enabled": config["NEWS_LEARNING_ENABLED"],
            "news_learning_block_enabled": config[
                "NEWS_LEARNING_BLOCK_ENABLED"
            ],
            "news_learning_min_observations": config[
                "NEWS_LEARNING_MIN_OBSERVATIONS"
            ],
            "news_learning_max_observations": config[
                "NEWS_LEARNING_MAX_OBSERVATIONS"
            ],
            "news_learning_min_correlation": config[
                "NEWS_LEARNING_MIN_CORRELATION"
            ],
            "news_predicted_return_block_percent": config[
                "NEWS_PREDICTED_RETURN_BLOCK_PERCENT"
            ],
            "news_learning_state_file": str(base_dir / ".news_learning_state.json"),
            "news_learning_llm_state_file": str(base_dir / ".news_learning_state_llm.json"),
            "news_score_refinement_enabled": config["NEWS_SCORE_REFINEMENT_ENABLED"],
            "news_rss_enabled": config["NEWS_RSS_ENABLED"],
            "news_rss_feed_urls": config["NEWS_RSS_FEED_URLS"],
            "symbol_reference_enabled": config["SYMBOL_REFERENCE_ENABLED"],
            "symbol_reference_refresh_days": config["SYMBOL_REFERENCE_REFRESH_DAYS"],
            "symbol_reference_database_file": str(base_dir / ".symbol_reference.duckdb"),
            "decision_memory_enabled": config["DECISION_MEMORY_ENABLED"],
            "decision_memory_block_enabled": config["DECISION_MEMORY_BLOCK_ENABLED"],
            "decision_memory_min_observations": config["DECISION_MEMORY_MIN_OBSERVATIONS"],
            "decision_memory_max_observations": config["DECISION_MEMORY_MAX_OBSERVATIONS"],
            "decision_memory_min_correlation": config["DECISION_MEMORY_MIN_CORRELATION"],
            "decision_memory_edge_block_percent": config["DECISION_MEMORY_EDGE_BLOCK_PERCENT"],
            "decision_memory_backfill_days": config["DECISION_MEMORY_BACKFILL_DAYS"],
            "decision_memory_database_file": str(base_dir / ".trade_memory.duckdb"),
            "portfolio_memory_enabled": config["PORTFOLIO_MEMORY_ENABLED"],
            "portfolio_memory_min_observations": config["PORTFOLIO_MEMORY_MIN_OBSERVATIONS"],
            "portfolio_memory_max_observations": config["PORTFOLIO_MEMORY_MAX_OBSERVATIONS"],
            "portfolio_memory_database_file": str(base_dir / ".portfolio_memory.duckdb"),
            "llm_news_enabled": config["LLM_NEWS_ENABLED"],
            "llm_news_model": config["LLM_NEWS_MODEL"],
            "llm_news_base_url": config["LLM_NEWS_BASE_URL"],
            "llm_news_block_on_high_risk": config["LLM_NEWS_BLOCK_ON_HIGH_RISK"],
            "llm_news_block_score": config["LLM_NEWS_BLOCK_SCORE"],
        },
    )
    return broker, strategy


def main() -> int:
    """Configure Lumibot and run until the process receives a stop signal."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    _tidy_logging()
    logger = logging.getLogger("trading-agent")

    try:
        config = load_config(CONFIG_PATH)

        # Keep credentials out of command-line arguments and process listings.
        # The news layer (news_context.py) and email reporting read these
        # environment variables so secrets never travel through the Lumibot
        # parameters dict, which can end up in logs.
        os.environ["ALPACA_API_KEY"] = str(config["ALPACA_API_KEY"])
        os.environ["ALPACA_API_SECRET"] = str(config["ALPACA_SECRET_KEY"])
        os.environ["ALPACA_IS_PAPER"] = str(config["IS_PAPER_TRADING"]).lower()
        os.environ["EMAIL_SMTP_PASSWORD"] = str(config["EMAIL_SMTP_PASSWORD"])

        _, strategy = build_strategy(config, BASE_DIR)

        trader = Trader()
        trader.add_strategy(strategy)
        logger.info(
            "Starting %s portfolio trading (proxy assets %s/%s)",
            "paper" if config["IS_PAPER_TRADING"] else "LIVE",
            config["ASSET_A"],
            config["ASSET_B"],
        )
        trader.run_all()
        return 0
    except KeyboardInterrupt:
        logger.info("Trading agent stopped by operator")
        return 0
    except Exception:
        logger.exception("Trading agent stopped because of a fatal startup/runtime error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
