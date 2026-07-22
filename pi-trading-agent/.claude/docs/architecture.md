# Architecture

Referenced by `/mnt/dietpi_userdata/staging/pi-trading/CLAUDE.md`. Keep this in sync with any module/behavior change.

## Two independent processes

The repo runs as **two separate systemd services**, each its own OS process:

- `main.py` → `strategy.py`'s `AssetRotationStrategy` — equities, gated to run around NYSE open (twice a trading day).
- `main_crypto.py` → `crypto_strategy.py`'s `CryptoRotationStrategy` — crypto, gated to run only while NYSE is *closed*.

They are separate processes, not one process with two Lumibot strategies, because Lumibot's `Trader.add_strategy()` raises `NotImplementedError` for a second live strategy in one `Trader` (`lumibot/traders/trader.py`). Two processes also isolate crashes and restarts. Both read the same `config.json` and Alpaca account, but use disjoint runtime-state, memory, and discovery files. They coordinate cash only by independently reading the shared account value: equity reserves half when crypto is enabled, and crypto targets the other half.

## Module map

**Entry points**
- `main.py` — loads and validates the complete `config.json`, builds the broker + `AssetRotationStrategy`, and starts Lumibot's `Trader`. `MarketOpenLoggingAlpaca` caches `market_hours()` per calendar day and makes waits interruptible so systemd stops cleanly.
- `main_crypto.py` — parallel crypto entry point. Reuses `main.load_config` and `MarketOpenLoggingAlpaca`; its parameter selection and process startup remain crypto-specific.
- `config_support.py` — declarative shared config groups, lowercase runtime-parameter selection, and explicit state-path resolution used by both entry points. Validation policy remains centralized in `main.load_config`.

**Strategy classes**
- `strategy.py` — `AssetRotationStrategy(BrokerRuntimeSupport, Strategy)`. Owns equity-specific universe, signals, exits, replacements/top-ups, reports, and orchestration. `_prepare_portfolio_iteration_context()` gathers all non-order decision inputs before `_run_portfolio_iteration()` ranks or submits orders.
- `crypto_strategy.py` — `CryptoRotationStrategy(BrokerRuntimeSupport, Strategy)`. It remains independent from `AssetRotationStrategy`; only asset-agnostic mechanics are shared. `_prepare_crypto_iteration_context()` gathers cached news/LLM context and signals after protective exits and before new entries.
- `strategy_support.py` — `BrokerRuntimeSupport` (restart-safe state access, safe quantity parsing, one-read-per-iteration order cache, active-order checks, and checked submission), the `IterationContext` value object, and shared pooled-memory input/update helpers. It contains no equity/crypto selection policy.

**Shared, asset-class-agnostic modules** (imported by both strategy classes)
- `decision_math.py` — pure math with zero broker/`self` coupling: chronological out-of-sample validation, posture/news ranking adjustments, LLM exposure scaling, `qualified_position_count` (default: fill every fundable qualified slot), and `optimal_position_count` (optional narrower variance-aware sizing). Equity keeps compatibility aliases for established call sites.
- `email_render.py` — generic HTML-table-rendering helpers (`email_kv_section`, `email_bullet_section`, `email_status_theme`, `render_email_shell`) extracted from `strategy.py`'s email code. Both strategies build their own report-specific sections and pass them in; this module has no report-shape assumptions.
- `market_sessions.py` — `is_next_trading_session` (NYSE-session succession, used by equity's memory classes), `nyse_is_open` (cached NYSE-open predicate, used by crypto's gating — see decision-pipeline.md), `is_next_calendar_day` (plain calendar-day succession, used by crypto's memory classes since crypto trades every day, not just NYSE sessions).
- `portfolio_memory.py` / `trade_memory.py` — `PortfolioMemory`/`TradeMemory` both accept a `next_session_predicate` constructor parameter (default `is_next_trading_session`, preserving equity's exact behavior) and use dip plus the LLM score as learned inputs. Upgraded databases gain a nullable `llm_score` column; the legacy keyword-based `news_score` column is ignored. `CryptoRotationStrategy` passes `is_next_calendar_day` instead.
- `autonomous_universe.py` — `AutonomousUniverse` takes `asset_class` (default `"us_equity"`) and `symbol_filter` (default: tradable + fractionable + plain-ticker regex) constructor parameters. `CryptoRotationStrategy` passes `asset_class="crypto"` and `_crypto_asset_symbol_filter` (defined in `crypto_strategy.py`), since Alpaca's crypto assets come back as `"BASE/QUOTE"` pairs (e.g. `"BTC/USD"`) with no `fractionable` field, not plain tickers.
- `ridge_regression.py` — the two-feature ridge fit both memory classes use.

**News/LLM supporting modules**
- `news_context.py` and `llm_news.py` are shared by equity and crypto; crypto uses Alpaca's symbol filter and excludes broad equity RSS feeds. `llm_news.purchase_veto_reason` is the single purchase guard. The Granite assessment uses strict schema fields (`score`, `reason_code`, `evidence`); risk level and readable evidence text are derived locally. Keyword scores remain preprocessing/reporting metadata. `rss_news.py`, `article_filter.py`, and `symbol_reference.py` are equity-specific; `token_estimate.py` bounds prompts.

## Config

`main.py`'s `load_config` validates the entire `config.json` in one place, including all `CRYPTO_*` keys. `config_support.select_parameters()` then maps validated uppercase keys to each strategy's lowercase runtime names, and `resolve_state_paths()` supplies disjoint file manifests. Equity reads `CRYPTO_ENABLED` to decide whether to reserve half the shared account value; there is no fixed `CRYPTO_CASH_ALLOCATION_DOLLARS` setting.
