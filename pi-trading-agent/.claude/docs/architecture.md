# Architecture

Referenced by `/mnt/dietpi_userdata/staging/pi-trading/CLAUDE.md`. Keep this in sync with any module/behavior change.

## Two independent processes

The repo runs as **two separate systemd services**, each its own OS process:

- `main.py` → `strategy.py`'s `AssetRotationStrategy` — equities, gated to run around NYSE open (twice a trading day).
- `main_crypto.py` → `crypto_strategy.py`'s `CryptoRotationStrategy` — crypto, gated to run only while NYSE is *closed*.

They are separate processes, not one process with two Lumibot strategies, because Lumibot's `Trader.add_strategy()` raises `NotImplementedError` for a second live strategy in one `Trader` (`lumibot/traders/trader.py`). Two processes also mean a crash or restart in one can never take the other down. Both read the same `config.json` and share the same Alpaca account/credentials, but write only their own, fully disjoint state files (`.portfolio_*`/`.trade_memory.duckdb`/etc. for equity, `.crypto_*` for crypto).

## Module map

**Entry points**
- `main.py` — loads and validates `config.json` (`load_config`), builds the broker + `AssetRotationStrategy` (`build_strategy`), starts Lumibot's `Trader`. Also defines `MarketOpenLoggingAlpaca`, an `Alpaca` broker subclass that caches `market_hours()` per calendar day (avoids a CPU-pegging bug from Lumibot recomputing the NYSE holiday calendar on every poll tick) and makes market-open/close waits interruptible so systemd stops cleanly.
- `main_crypto.py` — parallel entry point for crypto. Reuses `main.load_config`/`MarketOpenLoggingAlpaca` rather than duplicating validation; only `build_crypto_strategy` and its own `main()` are crypto-specific.

**Strategy classes**
- `strategy.py` — `AssetRotationStrategy(Strategy)`, ~3300 lines. Owns the entire equity decision pipeline: state-file helpers, email reporting, news/LLM integration, decision memory, portfolio memory, discovery, and the portfolio decision pipeline itself (see decision-pipeline.md).
- `crypto_strategy.py` — `CryptoRotationStrategy(Strategy)`. Deliberately its own class, not a subclass of `AssetRotationStrategy` — inheriting would couple crypto to equity-specific internals (email HTML, equity-tuned discovery heuristics) that would need silent overriding. Reimplements the same decision shape (dip signal, exits, build, discovery, Opportunistic Opportunity swap) narrowly for crypto, sharing only genuinely asset-class-agnostic code (below). No news/LLM integration — crypto has none of that layer yet.

**Shared, asset-class-agnostic modules** (imported by both strategy classes)
- `decision_math.py` — pure math with zero broker/`self` coupling: `walk_forward_net_returns` (chronological out-of-sample validation), `posture_adjusted_edge` (risky/conservative ranking reshaping), `optimal_position_count` (Sharpe-like position sizing). `AssetRotationStrategy` keeps its old method names (`_walk_forward_net_returns` etc.) as `staticmethod` aliases onto these functions so existing call sites/tests are unaffected.
- `email_render.py` — generic HTML-table-rendering helpers (`email_kv_section`, `email_bullet_section`, `email_status_theme`, `render_email_shell`) extracted from `strategy.py`'s email code. Both strategies build their own report-specific sections and pass them in; this module has no report-shape assumptions.
- `market_sessions.py` — `is_next_trading_session` (NYSE-session succession, used by equity's memory classes), `nyse_is_open` (cached NYSE-open predicate, used by crypto's gating — see decision-pipeline.md), `is_next_calendar_day` (plain calendar-day succession, used by crypto's memory classes since crypto trades every day, not just NYSE sessions).
- `portfolio_memory.py` / `trade_memory.py` — `PortfolioMemory`/`TradeMemory` both accept a `next_session_predicate` constructor parameter (default `is_next_trading_session`, preserving equity's exact behavior). `CryptoRotationStrategy` passes `is_next_calendar_day` instead — using the NYSE-session default for crypto would silently fail to settle a Friday observation until the following Monday, spanning a weekend of real crypto price action as if it were a single "next session" return.
- `autonomous_universe.py` — `AutonomousUniverse` takes `asset_class` (default `"us_equity"`) and `symbol_filter` (default: tradable + fractionable + plain-ticker regex) constructor parameters. `CryptoRotationStrategy` passes `asset_class="crypto"` and `_crypto_asset_symbol_filter` (defined in `crypto_strategy.py`), since Alpaca's crypto assets come back as `"BASE/QUOTE"` pairs (e.g. `"BTC/USD"`) with no `fractionable` field, not plain tickers.
- `ridge_regression.py` — the two-feature ridge fit both memory classes use.
- `signal_snapshot.py` / `trade_counter.py` — purely observational side channels, both called from each strategy's own decision/order-submission code but never read back into it. `signal_snapshot.write_snapshot` runs once per iteration (per-symbol opinions); `trade_counter.record_trade` runs from each strategy's `_submit_order_checked` choke point on every broker-accepted order, resetting to 1 on a new calendar day. Consumed by `scripts/view_signals.py` (terminal) and `scripts/web_dashboard.py` (browser) — see "Operating the service" in the README.

**Equity-only supporting modules** (no crypto equivalent yet)
- `adaptive_news_model.py`, `news_context.py`, `rss_news.py`, `article_filter.py`, `llm_news.py`, `symbol_reference.py` — the news/LLM/learning layer. `token_estimate.py` is a small LLM-prompt-sizing helper used by `llm_news.py`.

## Config

`main.py`'s `load_config` validates the entire `config.json` in one place — including all `CRYPTO_*` keys — even though only `main_crypto.py` consumes most of them. This is deliberate: `main.py` needs `CRYPTO_CASH_ALLOCATION_DOLLARS` itself (to compute `portfolio_crypto_reserve_dollars`, the equity-side cash reservation — see decision-pipeline.md), and a single validation implementation means the two processes can never silently disagree about what a config value means.
