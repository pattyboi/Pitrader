#!/usr/bin/env python3
"""Start the Alpaca-backed Lumibot asset-rotation strategy."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from strategy import AssetRotationStrategy


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


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
    return config


def main() -> int:
    """Configure Lumibot and run until the process receives a stop signal."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
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

        broker = Alpaca(
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
            },
        )

        trader = Trader()
        trader.add_strategy(strategy)
        logger.info(
            "Starting %s trading for %s/%s",
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
