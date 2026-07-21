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
from pathlib import Path

from lumibot.traders import Trader

from config_support import (
    EMAIL_CONFIG_KEYS,
    LLM_CONFIG_KEYS,
    NEWS_CONFIG_KEYS,
    resolve_state_paths,
    select_parameters,
)
from crypto_strategy import CryptoRotationStrategy
from main import (
    BASE_DIR,
    CONFIG_PATH,
    LOG_FORMAT,
    MarketOpenLoggingAlpaca,
    _tidy_logging,
    load_config,
    run_trader_until_stopped,
)


_CRYPTO_PARAMETER_KEYS = (
    "CRYPTO_ENABLED",
    "CRYPTO_SYMBOLS",
    "CRYPTO_MAX_POSITIONS",
    "CRYPTO_FILL_QUALIFIED_SLOTS",
    "CRYPTO_ANALYSIS_DAYS",
    "CRYPTO_RECENT_HIGH_LOOKBACK_DAYS",
    "CRYPTO_MIN_SIGNAL_OBSERVATIONS",
    "CRYPTO_DIP_THRESHOLD_PERCENT",
    "CRYPTO_MIN_EXPECTED_PROFIT_PERCENT",
    "CRYPTO_OOS_MIN_OBSERVATIONS",
    "CRYPTO_OOS_MIN_NET_PROFIT_PERCENT",
    "CRYPTO_ROUND_TRIP_COST_PERCENT",
    "CRYPTO_TAKE_PROFIT_PERCENT",
    "CRYPTO_STOP_LOSS_PERCENT",
    "CRYPTO_HOLDING_HORIZON_MAX_DAYS",
    "CRYPTO_MIN_ORDER_DOLLARS",
    "CRYPTO_ITERATION_INTERVAL_MINUTES",
    "CRYPTO_NEWS_REFRESH_MINUTES",
    "CRYPTO_RISK_POSTURE",
    "CRYPTO_MEMORY_ENABLED",
    "CRYPTO_MEMORY_MIN_OBSERVATIONS",
    "CRYPTO_MEMORY_MAX_OBSERVATIONS",
    "CRYPTO_MEMORY_MIN_CORRELATION",
    "CRYPTO_EMAIL_REPORT_ENABLED",
    "CRYPTO_AUTONOMOUS_DISCOVERY",
    "CRYPTO_DISCOVERY_BATCH_SIZE",
    "CRYPTO_DISCOVERY_REFRESH_DAYS",
    "CRYPTO_ASSET_A",
    "CRYPTO_ASSET_B",
    "CRYPTO_OPPORTUNISTIC_MIN_PROBABILITY",
)

_CRYPTO_STATE_FILES = {
    "crypto_holding_state_file": ".crypto_holding_state.json",
    "crypto_runtime_state_database_file": ".crypto_runtime_state.duckdb",
    "crypto_signal_snapshot_file": ".crypto_signal_snapshot.json",
    "crypto_trade_count_file": ".crypto_trade_count.json",
    "crypto_memory_database_file": ".crypto_portfolio_memory.duckdb",
    "crypto_trade_memory_database_file": ".crypto_trade_memory.duckdb",
    "crypto_email_state_file": ".crypto_last_email_report",
    "crypto_universe_database_file": ".crypto_universe.duckdb",
    "crypto_rotation_state_file": ".crypto_rotation_state.json",
    "crypto_opportunistic_swap_state_file": ".crypto_opportunistic_swap_state.json",
    "shutdown_diagnostic_file": ".crypto_shutdown_diagnostic.log",
}


def _crypto_strategy_parameters(config: dict, base_dir: Path) -> dict:
    parameters = select_parameters(
        config,
        _CRYPTO_PARAMETER_KEYS,
        EMAIL_CONFIG_KEYS,
        NEWS_CONFIG_KEYS,
        LLM_CONFIG_KEYS,
    )
    parameters.update(resolve_state_paths(base_dir, _CRYPTO_STATE_FILES))
    return parameters


def build_crypto_strategy(
    config: dict, base_dir: Path
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
        parameters=_crypto_strategy_parameters(config, base_dir),
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
        return run_trader_until_stopped(
            trader, strategy, logger, process_name="Crypto trading agent"
        )
    except KeyboardInterrupt:
        logger.info("Crypto trading agent stopped by operator")
        return 0
    except Exception:
        logger.exception("Crypto trading agent stopped because of a fatal startup/runtime error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
