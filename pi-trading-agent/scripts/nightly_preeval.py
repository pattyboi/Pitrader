#!/usr/bin/env python3
"""One nightly pass that pre-warms the per-symbol LLM article-verdict cache
(article_filter.py's `.article_verdicts.duckdb`) for every symbol tomorrow's
trading iteration might evaluate, so that iteration finds a same-day cache
hit instead of paying an Ollama round-trip live. See
AssetRotationStrategy._run_nightly_preevaluation for what actually runs.
Run once nightly, well after midnight, from trading-agent-nightly-preeval.timer.

Exits 0 on any failure -- a bad night must never block or delay trading.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import BASE_DIR, CONFIG_PATH, build_strategy, load_config  # noqa: E402

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("nightly-preeval")

    try:
        config = load_config(CONFIG_PATH)
        os.environ["ALPACA_API_KEY"] = str(config["ALPACA_API_KEY"])
        os.environ["ALPACA_API_SECRET"] = str(config["ALPACA_SECRET_KEY"])
        os.environ["ALPACA_IS_PAPER"] = str(config["IS_PAPER_TRADING"]).lower()
        os.environ["EMAIL_SMTP_PASSWORD"] = str(config["EMAIL_SMTP_PASSWORD"])

        broker, strategy = build_strategy(config, BASE_DIR)
        # This helper runs outside Lumibot's Trader lifecycle, so initialize
        # the strategy explicitly before reading persisted holdings/rotation
        # state. Without this, Vars lacks portfolio_pending_rotation and the
        # nightly pass exits before making a single LLM call.
        strategy.initialize()

        try:
            market_open = broker.market_hours(close=False, next=False)
            today = strategy.get_datetime().date()
            if market_open.date() != today:
                logger.info("Not a trading day today; skipping nightly pre-evaluation.")
                return 0
        except Exception as exc:
            # Can't confirm either way -- proceed rather than silently skip a
            # trading day; worst case is a few wasted Ollama calls tonight.
            logger.warning(
                "Could not confirm today is a trading day (%s: %s); proceeding anyway.",
                type(exc).__name__,
                exc,
            )

        report = strategy._run_nightly_preevaluation()
        logger.info("Nightly pre-evaluation done: %s", report or "nothing to do")
        return 0
    except Exception:
        logger.exception("Nightly pre-evaluation failed safely; trading is unaffected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
