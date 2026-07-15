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
        "LLM_NEWS_PROVIDER",
        "LLM_NEWS_API_KEY",
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
    if not isinstance(config["IS_PAPER_TRADING"], bool):
        raise TypeError("IS_PAPER_TRADING must be true or false")

    asset_a = str(config["ASSET_A"]).strip().upper()
    asset_b = str(config["ASSET_B"]).strip().upper()
    if not asset_a or not asset_b or asset_a == asset_b:
        raise ValueError("ASSET_A and ASSET_B must be different, non-empty symbols")

    dip_threshold = float(config["DIP_THRESHOLD_PERCENT"])
    lookback_days = int(config["RECENT_HIGH_LOOKBACK_DAYS"])
    if not 0.0 < dip_threshold < 100.0:
        raise ValueError("DIP_THRESHOLD_PERCENT must be greater than 0 and less than 100")
    if lookback_days < 2:
        raise ValueError("RECENT_HIGH_LOOKBACK_DAYS must be at least 2")

    # Portfolio mode is the default. Autonomous discovery, when also enabled,
    # expands the configured seed watchlist through a bounded scan.
    portfolio_defaults = {
        "PORTFOLIO_ENABLED": True,
        "PORTFOLIO_SYMBOLS": [asset_a, asset_b],
        "PORTFOLIO_MAX_POSITIONS": 1,
        "PORTFOLIO_ANALYSIS_DAYS": 252,
        "PORTFOLIO_MIN_SIGNAL_OBSERVATIONS": 20,
        "PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT": 1.0,
        "PORTFOLIO_OOS_MIN_OBSERVATIONS": 10,
        "PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT": 0.0,
        "PORTFOLIO_ROUND_TRIP_COST_PERCENT": 0.20,
        "PORTFOLIO_MAX_HOLDING_DAYS": 1,
        "PORTFOLIO_AUTONOMOUS_DISCOVERY": False,
        "PORTFOLIO_DISCOVERY_BATCH_SIZE": 12,
        "PORTFOLIO_DISCOVERY_REFRESH_DAYS": 7,
        "PORTFOLIO_FRACTIONAL_SHARES": True,
        "PORTFOLIO_CASH_RESERVE_DOLLARS": 2.0,
        "PORTFOLIO_MIN_ORDER_DOLLARS": 5.0,
        "PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY": 0.55,
        "PORTFOLIO_RISK_POSTURE": "conservative",
        "WSB_CONTEXT_ENABLED": False,
        "WSB_DISCOVERY_ENABLED": False,
        "WSB_DISCOVERY_MAX_SYMBOLS": 10,
        "WSB_CONTEXT_TIMEOUT_SECONDS": 10.0,
        "CONGRESS_CONTEXT_ENABLED": False,
        "CONGRESS_CONTEXT_TIMEOUT_SECONDS": 10.0,
    }
    for key, default in portfolio_defaults.items():
        config.setdefault(key, default)
    if not isinstance(config["PORTFOLIO_ENABLED"], bool):
        raise TypeError("PORTFOLIO_ENABLED must be true or false")
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
    portfolio_max_holding_days = int(config["PORTFOLIO_MAX_HOLDING_DAYS"])
    if not isinstance(config["PORTFOLIO_AUTONOMOUS_DISCOVERY"], bool):
        raise TypeError("PORTFOLIO_AUTONOMOUS_DISCOVERY must be true or false")
    discovery_batch_size = int(config["PORTFOLIO_DISCOVERY_BATCH_SIZE"])
    discovery_refresh_days = int(config["PORTFOLIO_DISCOVERY_REFRESH_DAYS"])
    if not isinstance(config["PORTFOLIO_FRACTIONAL_SHARES"], bool):
        raise TypeError("PORTFOLIO_FRACTIONAL_SHARES must be true or false")
    portfolio_cash_reserve = float(config["PORTFOLIO_CASH_RESERVE_DOLLARS"])
    portfolio_min_order = float(config["PORTFOLIO_MIN_ORDER_DOLLARS"])
    opportunity_min_probability = float(config["PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY"])
    if not isinstance(config["WSB_CONTEXT_ENABLED"], bool):
        raise TypeError("WSB_CONTEXT_ENABLED must be true or false")
    if not isinstance(config["WSB_DISCOVERY_ENABLED"], bool):
        raise TypeError("WSB_DISCOVERY_ENABLED must be true or false")
    wsb_max_symbols = int(config["WSB_DISCOVERY_MAX_SYMBOLS"])
    wsb_timeout = float(config["WSB_CONTEXT_TIMEOUT_SECONDS"])
    if not 1 <= portfolio_max_positions <= len(portfolio_symbols):
        raise ValueError("PORTFOLIO_MAX_POSITIONS must be between 1 and the number of portfolio symbols")
    if not 30 <= portfolio_analysis_days <= 2000:
        raise ValueError("PORTFOLIO_ANALYSIS_DAYS must be between 30 and 2000")
    if not 5 <= portfolio_min_observations <= 500:
        raise ValueError("PORTFOLIO_MIN_SIGNAL_OBSERVATIONS must be between 5 and 500")
    if not 0.0 <= portfolio_min_profit <= 100.0:
        raise ValueError("PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT must be between 0 and 100")
    if not 5 <= portfolio_oos_min_observations <= 500:
        raise ValueError("PORTFOLIO_OOS_MIN_OBSERVATIONS must be between 5 and 500")
    if not 0.0 <= portfolio_oos_min_profit <= 100.0:
        raise ValueError("PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT must be between 0 and 100")
    if not 0.0 <= portfolio_round_trip_cost <= 10.0:
        raise ValueError("PORTFOLIO_ROUND_TRIP_COST_PERCENT must be between 0 and 10")
    if not 1 <= portfolio_max_holding_days <= 20:
        raise ValueError("PORTFOLIO_MAX_HOLDING_DAYS must be between 1 and 20")
    if not 1 <= discovery_batch_size <= 30:
        raise ValueError("PORTFOLIO_DISCOVERY_BATCH_SIZE must be between 1 and 30")
    if not 1 <= discovery_refresh_days <= 90:
        raise ValueError("PORTFOLIO_DISCOVERY_REFRESH_DAYS must be between 1 and 90")
    if not 0.0 <= portfolio_cash_reserve <= 1000.0:
        raise ValueError("PORTFOLIO_CASH_RESERVE_DOLLARS must be between 0 and 1000")
    if not 1.0 <= portfolio_min_order <= 1000.0:
        raise ValueError("PORTFOLIO_MIN_ORDER_DOLLARS must be between 1 and 1000")
    if not 0.5 <= opportunity_min_probability <= 0.95:
        raise ValueError("PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY must be between 0.5 and 0.95")
    risk_posture = str(config["PORTFOLIO_RISK_POSTURE"]).strip().lower()
    if risk_posture not in ("conservative", "risky"):
        raise ValueError("PORTFOLIO_RISK_POSTURE must be conservative or risky")
    config["PORTFOLIO_RISK_POSTURE"] = risk_posture
    if not 1 <= wsb_max_symbols <= 20:
        raise ValueError("WSB_DISCOVERY_MAX_SYMBOLS must be between 1 and 20")
    if not 1.0 <= wsb_timeout <= 30.0:
        raise ValueError("WSB_CONTEXT_TIMEOUT_SECONDS must be between 1 and 30")
    if not isinstance(config["CONGRESS_CONTEXT_ENABLED"], bool):
        raise TypeError("CONGRESS_CONTEXT_ENABLED must be true or false")
    congress_timeout = float(config["CONGRESS_CONTEXT_TIMEOUT_SECONDS"])
    if not 1.0 <= congress_timeout <= 30.0:
        raise ValueError("CONGRESS_CONTEXT_TIMEOUT_SECONDS must be between 1 and 30")
    config["PORTFOLIO_SYMBOLS"] = portfolio_symbols
    config["PORTFOLIO_MAX_POSITIONS"] = portfolio_max_positions
    config["PORTFOLIO_ANALYSIS_DAYS"] = portfolio_analysis_days
    config["PORTFOLIO_MIN_SIGNAL_OBSERVATIONS"] = portfolio_min_observations
    config["PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT"] = portfolio_min_profit
    config["PORTFOLIO_OOS_MIN_OBSERVATIONS"] = portfolio_oos_min_observations
    config["PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT"] = portfolio_oos_min_profit
    config["PORTFOLIO_ROUND_TRIP_COST_PERCENT"] = portfolio_round_trip_cost
    config["PORTFOLIO_MAX_HOLDING_DAYS"] = portfolio_max_holding_days
    config["PORTFOLIO_DISCOVERY_BATCH_SIZE"] = discovery_batch_size
    config["PORTFOLIO_DISCOVERY_REFRESH_DAYS"] = discovery_refresh_days
    config["PORTFOLIO_CASH_RESERVE_DOLLARS"] = portfolio_cash_reserve
    config["PORTFOLIO_MIN_ORDER_DOLLARS"] = portfolio_min_order
    config["PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY"] = opportunity_min_probability
    config["WSB_DISCOVERY_MAX_SYMBOLS"] = wsb_max_symbols
    config["WSB_CONTEXT_TIMEOUT_SECONDS"] = wsb_timeout
    config["CONGRESS_CONTEXT_TIMEOUT_SECONDS"] = congress_timeout

    if not isinstance(config["EMAIL_REPORT_ENABLED"], bool):
        raise TypeError("EMAIL_REPORT_ENABLED must be true or false")
    if not isinstance(config["EMAIL_USE_TLS"], bool):
        raise TypeError("EMAIL_USE_TLS must be true or false")
    email_port = int(config["EMAIL_SMTP_PORT"])
    if not 1 <= email_port <= 65535:
        raise ValueError("EMAIL_SMTP_PORT must be between 1 and 65535")
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

    if not isinstance(config["NEWS_CONTEXT_ENABLED"], bool):
        raise TypeError("NEWS_CONTEXT_ENABLED must be true or false")
    if not isinstance(config["NEWS_BLOCK_ON_HIGH_RISK"], bool):
        raise TypeError("NEWS_BLOCK_ON_HIGH_RISK must be true or false")
    news_lookback = int(config["NEWS_LOOKBACK_HOURS"])
    news_limit = int(config["NEWS_MAX_ARTICLES"])
    news_block_score = int(config["NEWS_HIGH_RISK_SCORE"])
    if not 1 <= news_lookback <= 168:
        raise ValueError("NEWS_LOOKBACK_HOURS must be between 1 and 168")
    if not 1 <= news_limit <= 50:
        raise ValueError("NEWS_MAX_ARTICLES must be between 1 and 50")
    if news_block_score >= 0:
        raise ValueError("NEWS_HIGH_RISK_SCORE must be a negative integer")
    if not isinstance(config["NEWS_LEARNING_ENABLED"], bool):
        raise TypeError("NEWS_LEARNING_ENABLED must be true or false")
    if not isinstance(config["NEWS_LEARNING_BLOCK_ENABLED"], bool):
        raise TypeError("NEWS_LEARNING_BLOCK_ENABLED must be true or false")
    learning_minimum = int(config["NEWS_LEARNING_MIN_OBSERVATIONS"])
    learning_maximum = int(config["NEWS_LEARNING_MAX_OBSERVATIONS"])
    predicted_return_block = float(config["NEWS_PREDICTED_RETURN_BLOCK_PERCENT"])
    minimum_correlation = float(config["NEWS_LEARNING_MIN_CORRELATION"])
    if not 10 <= learning_minimum <= 500:
        raise ValueError("NEWS_LEARNING_MIN_OBSERVATIONS must be between 10 and 500")
    if not learning_minimum <= learning_maximum <= 1000:
        raise ValueError(
            "NEWS_LEARNING_MAX_OBSERVATIONS must be at least the minimum and at most 1000"
        )
    if not -25.0 <= predicted_return_block < 0.0:
        raise ValueError("NEWS_PREDICTED_RETURN_BLOCK_PERCENT must be from -25 to below 0")
    if not 0.0 <= minimum_correlation <= 1.0:
        raise ValueError("NEWS_LEARNING_MIN_CORRELATION must be between 0 and 1")

    if not isinstance(config["LLM_NEWS_ENABLED"], bool):
        raise TypeError("LLM_NEWS_ENABLED must be true or false")
    if not isinstance(config["LLM_NEWS_BLOCK_ON_HIGH_RISK"], bool):
        raise TypeError("LLM_NEWS_BLOCK_ON_HIGH_RISK must be true or false")
    llm_block_score = int(config["LLM_NEWS_BLOCK_SCORE"])
    if not -10 <= llm_block_score <= -1:
        raise ValueError("LLM_NEWS_BLOCK_SCORE must be from -10 through -1")
    llm_provider = str(config["LLM_NEWS_PROVIDER"]).strip().lower()
    if llm_provider not in ("gemini", "openai_compatible", "anthropic"):
        raise ValueError(
            "LLM_NEWS_PROVIDER must be gemini, openai_compatible, or anthropic"
        )
    llm_model = str(config["LLM_NEWS_MODEL"]).strip()
    if not llm_model:
        raise ValueError("LLM_NEWS_MODEL must be a non-empty model id")
    llm_base_url = str(config["LLM_NEWS_BASE_URL"]).strip()
    if llm_base_url and not llm_base_url.startswith(("http://", "https://")):
        raise ValueError("LLM_NEWS_BASE_URL must be an http(s) URL or empty")
    if llm_provider == "openai_compatible" and not llm_base_url:
        raise ValueError(
            "LLM_NEWS_BASE_URL is required when LLM_NEWS_PROVIDER is "
            "openai_compatible"
        )
    if config["LLM_NEWS_ENABLED"]:
        llm_key = str(config["LLM_NEWS_API_KEY"]).strip()
        if not llm_key or llm_key.startswith("REPLACE_WITH_"):
            raise ValueError(
                "LLM news assessment is enabled, but LLM_NEWS_API_KEY is "
                "not set. For the free Gemini tier, create a key at "
                "aistudio.google.com, or set LLM_NEWS_ENABLED to false."
            )
    config["LLM_NEWS_BLOCK_SCORE"] = llm_block_score
    config["LLM_NEWS_PROVIDER"] = llm_provider
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
    if not isinstance(config["DECISION_MEMORY_ENABLED"], bool) or not isinstance(config["DECISION_MEMORY_BLOCK_ENABLED"], bool):
        raise TypeError("DECISION_MEMORY_ENABLED and DECISION_MEMORY_BLOCK_ENABLED must be true or false")
    decision_minimum = int(config["DECISION_MEMORY_MIN_OBSERVATIONS"])
    decision_maximum = int(config["DECISION_MEMORY_MAX_OBSERVATIONS"])
    decision_correlation = float(config["DECISION_MEMORY_MIN_CORRELATION"])
    decision_edge = float(config["DECISION_MEMORY_EDGE_BLOCK_PERCENT"])
    decision_backfill_days = int(config["DECISION_MEMORY_BACKFILL_DAYS"])
    if not 20 <= decision_minimum <= 500 or not decision_minimum <= decision_maximum <= 1000:
        raise ValueError("DECISION_MEMORY observation limits must be from 20 to 1000")
    if not 0.0 <= decision_correlation <= 1.0 or not -25.0 <= decision_edge < 0.0:
        raise ValueError("DECISION_MEMORY correlation must be 0..1 and edge block must be -25..<0")
    if not 0 <= decision_backfill_days <= 5000:
        raise ValueError("DECISION_MEMORY_BACKFILL_DAYS must be between 0 and 5000")
    config["DECISION_MEMORY_MIN_OBSERVATIONS"] = decision_minimum
    config["DECISION_MEMORY_MAX_OBSERVATIONS"] = decision_maximum
    config["DECISION_MEMORY_MIN_CORRELATION"] = decision_correlation
    config["DECISION_MEMORY_EDGE_BLOCK_PERCENT"] = decision_edge
    config["DECISION_MEMORY_BACKFILL_DAYS"] = decision_backfill_days

    symbol_reference_defaults = {
        "SYMBOL_REFERENCE_ENABLED": True,
        "SYMBOL_REFERENCE_REFRESH_DAYS": 7,
        "NEWS_SCORE_REFINEMENT_ENABLED": False,
    }
    for key, default in symbol_reference_defaults.items():
        config.setdefault(key, default)
    if not isinstance(config["SYMBOL_REFERENCE_ENABLED"], bool):
        raise TypeError("SYMBOL_REFERENCE_ENABLED must be true or false")
    if not isinstance(config["NEWS_SCORE_REFINEMENT_ENABLED"], bool):
        raise TypeError("NEWS_SCORE_REFINEMENT_ENABLED must be true or false")
    symbol_reference_refresh_days = int(config["SYMBOL_REFERENCE_REFRESH_DAYS"])
    if not 1 <= symbol_reference_refresh_days <= 30:
        raise ValueError("SYMBOL_REFERENCE_REFRESH_DAYS must be between 1 and 30")
    config["SYMBOL_REFERENCE_REFRESH_DAYS"] = symbol_reference_refresh_days

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
        llm_key = str(config["LLM_NEWS_API_KEY"]).strip()
        if llm_key and not llm_key.startswith("REPLACE_WITH_"):
            os.environ["LLM_NEWS_API_KEY"] = llm_key

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
                "email_state_file": str(BASE_DIR / ".last_email_report"),
                "rotation_state_file": str(BASE_DIR / ".rotation_state.json"),
                "portfolio_enabled": config["PORTFOLIO_ENABLED"],
                "portfolio_symbols": config["PORTFOLIO_SYMBOLS"],
                "portfolio_max_positions": config["PORTFOLIO_MAX_POSITIONS"],
                "portfolio_analysis_days": config["PORTFOLIO_ANALYSIS_DAYS"],
                "portfolio_min_signal_observations": config["PORTFOLIO_MIN_SIGNAL_OBSERVATIONS"],
                "portfolio_min_expected_profit_percent": config["PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT"],
                "portfolio_oos_min_observations": config["PORTFOLIO_OOS_MIN_OBSERVATIONS"],
                "portfolio_oos_min_net_profit_percent": config["PORTFOLIO_OOS_MIN_NET_PROFIT_PERCENT"],
                "portfolio_round_trip_cost_percent": config["PORTFOLIO_ROUND_TRIP_COST_PERCENT"],
                "portfolio_max_holding_days": config["PORTFOLIO_MAX_HOLDING_DAYS"],
                "portfolio_holding_state_file": str(BASE_DIR / ".portfolio_holding_state.json"),
                "portfolio_rotation_state_file": str(BASE_DIR / ".portfolio_rotation_state.json"),
                "portfolio_autonomous_discovery": config["PORTFOLIO_AUTONOMOUS_DISCOVERY"],
                "portfolio_discovery_batch_size": config["PORTFOLIO_DISCOVERY_BATCH_SIZE"],
                "portfolio_discovery_refresh_days": config["PORTFOLIO_DISCOVERY_REFRESH_DAYS"],
                "portfolio_universe_state_file": str(BASE_DIR / ".autonomous_universe.json"),
                "fractional_shares": config["PORTFOLIO_FRACTIONAL_SHARES"],
                "portfolio_cash_reserve_dollars": config["PORTFOLIO_CASH_RESERVE_DOLLARS"],
                "portfolio_min_order_dollars": config["PORTFOLIO_MIN_ORDER_DOLLARS"],
                "portfolio_opportunistic_min_probability": config["PORTFOLIO_OPPORTUNISTIC_MIN_PROBABILITY"],
                "portfolio_risk_posture": config["PORTFOLIO_RISK_POSTURE"],
                "wsb_context_enabled": config["WSB_CONTEXT_ENABLED"],
                "wsb_discovery_enabled": config["WSB_DISCOVERY_ENABLED"],
                "wsb_discovery_max_symbols": config["WSB_DISCOVERY_MAX_SYMBOLS"],
                "wsb_context_timeout_seconds": config["WSB_CONTEXT_TIMEOUT_SECONDS"],
                "wsb_context_state_file": str(BASE_DIR / ".wsb_context_snapshot.json"),
                "congress_context_enabled": config["CONGRESS_CONTEXT_ENABLED"],
                "congress_context_timeout_seconds": config["CONGRESS_CONTEXT_TIMEOUT_SECONDS"],
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
                "news_learning_state_file": str(BASE_DIR / ".news_learning_state.json"),
                "news_score_refinement_enabled": config["NEWS_SCORE_REFINEMENT_ENABLED"],
                "symbol_reference_enabled": config["SYMBOL_REFERENCE_ENABLED"],
                "symbol_reference_refresh_days": config["SYMBOL_REFERENCE_REFRESH_DAYS"],
                "symbol_reference_database_file": str(BASE_DIR / ".symbol_reference.duckdb"),
                "decision_memory_enabled": config["DECISION_MEMORY_ENABLED"],
                "decision_memory_block_enabled": config["DECISION_MEMORY_BLOCK_ENABLED"],
                "decision_memory_min_observations": config["DECISION_MEMORY_MIN_OBSERVATIONS"],
                "decision_memory_max_observations": config["DECISION_MEMORY_MAX_OBSERVATIONS"],
                "decision_memory_min_correlation": config["DECISION_MEMORY_MIN_CORRELATION"],
                "decision_memory_edge_block_percent": config["DECISION_MEMORY_EDGE_BLOCK_PERCENT"],
                "decision_memory_backfill_days": config["DECISION_MEMORY_BACKFILL_DAYS"],
                "decision_memory_database_file": str(BASE_DIR / ".trade_memory.duckdb"),
                "llm_news_enabled": config["LLM_NEWS_ENABLED"],
                "llm_news_provider": config["LLM_NEWS_PROVIDER"],
                "llm_news_model": config["LLM_NEWS_MODEL"],
                "llm_news_base_url": config["LLM_NEWS_BASE_URL"],
                "llm_news_block_on_high_risk": config["LLM_NEWS_BLOCK_ON_HIGH_RISK"],
                "llm_news_block_score": config["LLM_NEWS_BLOCK_SCORE"],
            },
        )

        trader = Trader()
        trader.add_strategy(strategy)
        logger.info(
            "Starting %s trading for %s/%s%s",
            "paper" if config["IS_PAPER_TRADING"] else "LIVE",
            config["ASSET_A"],
            config["ASSET_B"],
            " (portfolio mode enabled)" if config["PORTFOLIO_ENABLED"] else "",
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
