#!/usr/bin/env python3
"""Start the Alpaca-backed Lumibot crypto rotation strategy.

Runs as a separate systemd service/process from main.py: Lumibot's
Trader.add_strategy() raises NotImplementedError for a second live strategy
in the same Trader, so equity and crypto trading can't share one process.
Independent processes also mean a crash or restart in one can never take the
other down.

Reads the same config.json as main.py -- CRYPTO_* keys are validated by
main.load_config so both processes agree on the same values without a second
validation implementation to drift out of sync -- and shares the same Alpaca
account/credentials, but writes only its own, fully separate .crypto_* state
files (see crypto_strategy.py).
"""

import logging
import os
import sys

from lumibot.traders import Trader

from crypto_strategy import CryptoRotationStrategy
from main import BASE_DIR, CONFIG_PATH, LOG_FORMAT, MarketOpenLoggingAlpaca, _tidy_logging, load_config


def build_crypto_strategy(
    config: dict, base_dir
) -> tuple[MarketOpenLoggingAlpaca, CryptoRotationStrategy]:
    """Construct the broker and crypto strategy from validated config.

    Mirrors main.build_strategy's shape (broker + strategy, no Trader loop
    started) so the two entry points never drift apart as config keys are
    added, without duplicating main.py's config validation.
    """
    broker = MarketOpenLoggingAlpaca(
        {
            "API_KEY": config["ALPACA_API_KEY"],
            "API_SECRET": config["ALPACA_SECRET_KEY"],
            "PAPER": config["IS_PAPER_TRADING"],
        }
    )
    strategy = CryptoRotationStrategy(
        broker=broker,
        parameters={
            "crypto_enabled": config["CRYPTO_ENABLED"],
            "crypto_symbols": config["CRYPTO_SYMBOLS"],
            "crypto_max_positions": config["CRYPTO_MAX_POSITIONS"],
            "crypto_analysis_days": config["CRYPTO_ANALYSIS_DAYS"],
            "crypto_recent_high_lookback_days": config["CRYPTO_RECENT_HIGH_LOOKBACK_DAYS"],
            "crypto_min_signal_observations": config["CRYPTO_MIN_SIGNAL_OBSERVATIONS"],
            "crypto_dip_threshold_percent": config["CRYPTO_DIP_THRESHOLD_PERCENT"],
            "crypto_min_expected_profit_percent": config["CRYPTO_MIN_EXPECTED_PROFIT_PERCENT"],
            "crypto_oos_min_observations": config["CRYPTO_OOS_MIN_OBSERVATIONS"],
            "crypto_oos_min_net_profit_percent": config["CRYPTO_OOS_MIN_NET_PROFIT_PERCENT"],
            "crypto_round_trip_cost_percent": config["CRYPTO_ROUND_TRIP_COST_PERCENT"],
            "crypto_take_profit_percent": config["CRYPTO_TAKE_PROFIT_PERCENT"],
            "crypto_stop_loss_percent": config["CRYPTO_STOP_LOSS_PERCENT"],
            "crypto_holding_horizon_max_days": config["CRYPTO_HOLDING_HORIZON_MAX_DAYS"],
            "crypto_min_order_dollars": config["CRYPTO_MIN_ORDER_DOLLARS"],
            "crypto_iteration_interval_minutes": config["CRYPTO_ITERATION_INTERVAL_MINUTES"],
            "crypto_risk_posture": config["CRYPTO_RISK_POSTURE"],
            "crypto_holding_state_file": str(base_dir / ".crypto_holding_state.json"),
            "crypto_memory_enabled": config["CRYPTO_MEMORY_ENABLED"],
            "crypto_memory_min_observations": config["CRYPTO_MEMORY_MIN_OBSERVATIONS"],
            "crypto_memory_max_observations": config["CRYPTO_MEMORY_MAX_OBSERVATIONS"],
            "crypto_memory_database_file": str(base_dir / ".crypto_portfolio_memory.duckdb"),
            "crypto_trade_memory_database_file": str(base_dir / ".crypto_trade_memory.duckdb"),
            "crypto_email_report_enabled": config["CRYPTO_EMAIL_REPORT_ENABLED"],
            "email_smtp_host": config["EMAIL_SMTP_HOST"],
            "email_smtp_port": config["EMAIL_SMTP_PORT"],
            "email_smtp_username": config["EMAIL_SMTP_USERNAME"],
            "email_from_address": config["EMAIL_FROM_ADDRESS"],
            "email_to_address": config["EMAIL_TO_ADDRESS"],
            "email_use_tls": config["EMAIL_USE_TLS"],
            "crypto_email_state_file": str(base_dir / ".crypto_last_email_report"),
            "crypto_autonomous_discovery": config["CRYPTO_AUTONOMOUS_DISCOVERY"],
            "crypto_discovery_batch_size": config["CRYPTO_DISCOVERY_BATCH_SIZE"],
            "crypto_discovery_refresh_days": config["CRYPTO_DISCOVERY_REFRESH_DAYS"],
            "crypto_universe_database_file": str(base_dir / ".crypto_universe.duckdb"),
            "crypto_asset_a": config["CRYPTO_ASSET_A"],
            "crypto_asset_b": config["CRYPTO_ASSET_B"],
            "crypto_opportunistic_min_probability": config["CRYPTO_OPPORTUNISTIC_MIN_PROBABILITY"],
            "crypto_rotation_state_file": str(base_dir / ".crypto_rotation_state.json"),
            "crypto_opportunistic_swap_state_file": str(base_dir / ".crypto_opportunistic_swap_state.json"),
        },
    )
    return broker, strategy


def main() -> int:
    """Configure Lumibot and run until the process receives a stop signal."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    _tidy_logging()
    logger = logging.getLogger("trading-agent-crypto")

    try:
        config = load_config(CONFIG_PATH)

        # Same credential-handling rationale as main.py: keep secrets out of
        # the Lumibot parameters dict, which can end up in logs.
        os.environ["ALPACA_API_KEY"] = str(config["ALPACA_API_KEY"])
        os.environ["ALPACA_API_SECRET"] = str(config["ALPACA_SECRET_KEY"])
        os.environ["ALPACA_IS_PAPER"] = str(config["IS_PAPER_TRADING"]).lower()
        os.environ["EMAIL_SMTP_PASSWORD"] = str(config["EMAIL_SMTP_PASSWORD"])

        _, strategy = build_crypto_strategy(config, BASE_DIR)

        trader = Trader()
        trader.add_strategy(strategy)
        logger.info(
            "Starting %s crypto trading (%s; symbols %s), active only while NYSE is closed",
            "paper" if config["IS_PAPER_TRADING"] else "LIVE",
            "enabled" if config["CRYPTO_ENABLED"] else "disabled -- idling until CRYPTO_ENABLED is true",
            ", ".join(config["CRYPTO_SYMBOLS"]) or "(none configured)",
        )
        trader.run_all()
        return 0
    except KeyboardInterrupt:
        logger.info("Crypto trading agent stopped by operator")
        return 0
    except Exception:
        logger.exception("Crypto trading agent stopped because of a fatal startup/runtime error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
