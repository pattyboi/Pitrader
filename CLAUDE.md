# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Lumibot/Alpaca trading agent that runs as a systemd service on a Raspberry Pi. Once per trading day it checks whether Asset B (default QQQ) has dipped a configured percent from its recent high; if so and the account holds Asset A (default SPY), it sells all of A and buys as much B as cash allows (fractional shares by default via `PORTFOLIO_FRACTIONAL_SHARES`; whole shares when false). It does not rotate back from B to A. An opt-in portfolio mode (`PORTFOLIO_ENABLED`) replaces the A/B pipeline with a bounded dip-signal portfolio over a watchlist, optionally expanded by autonomous discovery. All code lives in `pi-trading-agent/`.

**This can place real orders.** `config.json` holds live-capable Alpaca credentials and `IS_PAPER_TRADING`. Never flip that to `false`, weaken a trade-blocking guard, or loosen config validation without the user explicitly asking. `config.json` must stay mode 600 and out of version control.

## Commands

There are no tests and no linter configured. Python 3.13 runs on the target; deps are in `requirements.txt` (lumibot, alpaca-py, pandas), installed into `.venv/` by the installer.

```bash
cd pi-trading-agent

# Install venv + systemd unit, enable and start the service
sudo ./setup_service.sh

# Run in the foreground (uses .venv, reads config.json from the script's own directory)
.venv/bin/python main.py

# Syntax-check without printing secrets
python3 -m json.tool config.json >/dev/null

# Service lifecycle — always stop before editing config.json or code, then restart
sudo systemctl stop|start|restart trading-agent.service
sudo journalctl -u trading-agent.service -f          # follow logs
sudo journalctl -u trading-agent.service -n 100 --no-pager
```

The service restarts on crash after 30s (`Restart=always`), so a config error shows up as a restart loop — read the journal instead of rerunning the installer.

## Architecture

Seven modules, one strategy class:

- `main.py` — loads and strictly validates every key in `config.json` (`load_config` raises on any missing/invalid value; new config keys must be added to the `required` set, validated, and threaded into the strategy `parameters` dict), builds the Lumibot `Alpaca` broker and `Trader`, and runs `AssetRotationStrategy`.
- `strategy.py` — `AssetRotationStrategy(Strategy)` with `sleeptime = "1D"`. `on_trading_iteration` is the whole decision pipeline; `on_filled_order` completes the rotation.
- `news_context.py` — `WorldEventAnalyzer` fetches Alpaca news and produces a deterministic keyword score (`SEVERE_RISK_TERMS` −3, `RISK_TERMS` −1, `OPPORTUNITY_TERMS` +1, each counted once per article) wrapped in a `NewsContext`, which also carries the raw articles and a per-symbol score breakdown (from Alpaca's own article `symbols` tags) for downstream analysis. The pure `score_articles` classmethod (article dicts in, no network) does the scoring and is what `analyze()` calls after fetching; `NEWS_SCORE_REFINEMENT_ENABLED` (off by default) turns on recency decay and duplicate-phrase dampening inside it — off reproduces the original unweighted score exactly, so this can't silently change `NEWS_HIGH_RISK_SCORE`'s behavior.
- `symbol_reference.py` — `SymbolReference`, a local ticker→company-name mapping cross-checked from Alpaca's per-symbol asset lookup and the SEC's public `company_tickers.json`, persisted in `.symbol_reference.duckdb` and refreshed at most every `SYMBOL_REFERENCE_REFRESH_DAYS`. `verified_symbols()` filters a spurious/malformed Alpaca symbol tag before it's trusted (empty result — nothing cached yet — means fail open, not reject-everything); `scan_text_for_symbols()` is a bounded company-name text scan, scoped only to the day's candidates, that catches a mention Alpaca's own tagging missed. Never creates a trade or veto on its own.
- `llm_news.py` — optional `LLMNewsAnalyzer` sends the same articles to an LLM (one call per trading day, score −10…+10). Provider-agnostic: `gemini` (default — free tier, OpenAI-compatible endpoint via `requests`), `openai_compatible` (custom `LLM_NEWS_BASE_URL`), or `anthropic` (SDK with a strict structured-outputs schema). Non-Anthropic providers get the JSON contract via prompt + `response_format: json_object`, and replies are repaired defensively (`_parse_assessment`: fence-stripping, score clamping, risk-level derivation). Disabled by default; requires `LLM_NEWS_API_KEY` (chat subscriptions like Claude Pro/ChatGPT Plus cannot be used). Advisory unless `LLM_NEWS_BLOCK_ON_HIGH_RISK` is true. Do not add `temperature` or `thinking` params to the Anthropic request — the model is user-configurable and those 400 or vary by model.
- `adaptive_news_model.py` — `AdaptiveNewsModel`, a ridge-stabilized one-variable regression of next-day Asset B return on the news score, persisted in `.news_learning_state.json`. It only becomes authoritative after `NEWS_LEARNING_MIN_OBSERVATIONS` samples *and* sufficient `|correlation|`. In portfolio mode it keeps learning from Asset B as a market proxy.
- `trade_memory.py` — `TradeMemory`, a DuckDB journal (`.trade_memory.duckdb`, migrated once from a legacy `.trade_memory.sqlite3` if present) plus a two-feature ridge regression of the next-session B-minus-A edge on dip size and news score. A/B mode only. Unsettled observations older than `MAX_SETTLEMENT_GAP_DAYS` (4 calendar days) are never settled — a longer gap would record a multi-day return as one session. Every call site opens/closes its own `duckdb.connect()`; concurrent access from the broker-callback thread (`on_filled_order`) and the main iteration thread is possible and can raise `TransactionException: Catalog write-write conflict` — every call site must wrap in its own `try/except`, not just rely on the outer iteration handler.
- `autonomous_universe.py` — `AutonomousUniverse`, bounded daily discovery over Alpaca's asset directory, persisted in `.autonomous_universe.json`. The assets host follows the trading mode (paper vs live keys are host-specific). `remember()` deduplicates keeping the most recent mention so re-confirmed symbols (e.g. current holdings) are never trimmed as stale.

### Decision pipeline order — A/B mode (in `on_trading_iteration`)

1. Fetch news context and prices; bail out (no trade) if prices are missing or non-positive.
2. Update the adaptive model with today's score/price.
3. If `vars.pending_rotation` is set, reconcile against live positions and open orders: still holding A with an active sell → wait; holding A with no active sell (order died / lost across restart) → clear the flag and fall through to a fresh evaluation; A gone → buy B (or finish the rotation if cash can't cover one share).
4. Compute dip from the max daily high over the lookback; require `dip >= threshold` and a long Asset A position.
5. Three independent vetoes, any of which blocks the trade: the fixed keyword news score (`score <= NEWS_HIGH_RISK_SCORE` when `NEWS_BLOCK_ON_HIGH_RISK`), the LLM assessment (`score <= LLM_NEWS_BLOCK_SCORE` when `LLM_NEWS_BLOCK_ON_HIGH_RISK`), and the mature learned forecast (`predicted return <= NEWS_PREDICTED_RETURN_BLOCK_PERCENT` when correlation qualifies).
6. Submit the A market sale and set the pending flag via `_set_pending_rotation(True)`, which persists to `.rotation_state.json` so restarts can't strand mid-rotation cash. The B buy happens in `on_filled_order` when the sale fills (next daily iteration as fallback); the flag clears only when the B **buy fills** (or spendable cash drops below the minimum purchase), never on submission. `on_canceled_order` resets a dead sell to idle and leaves a dead buy pending for retry. `_buy_asset_b_with_available_cash` holds back `CASH_BUFFER_FRACTION` (1%) of cash plus `PORTFOLIO_CASH_RESERVE_DOLLARS` so the market order can't be rejected on a price uptick, checks for an already-working buy order before submitting, and is serialized by `_rotation_lock` against the broker-callback thread.

### Portfolio mode pipeline (in `_run_portfolio_iteration`)

1. Read **all** account stock positions via `_portfolio_held_positions()`; a broker read failure aborts the iteration (an empty dict would look like an empty portfolio and trigger duplicate buys). Held symbols are always merged into the evaluation universe and re-`remember()`ed daily so an orphaned position is impossible.
2. Fetch news, the LLM assessment, and update adaptive learning (Asset B proxy). `_market_veto_reason` combines the three configured vetoes; it blocks **new** purchases/replacements only — completing an in-flight replacement is never vetoed. Decision memory does not run in portfolio mode. `_refresh_symbol_reference` runs here too (once the day's evaluation universe is final, before news scoring is consumed) — cheap and weekly-gated, never blocks trading on failure.
3. Reconcile `.portfolio_rotation_state.json` exactly like the A/B flag: waiting sell → wait; dead sell → reset; sale done → buy the target (also continued immediately in `on_filled_order` when the sale fills); the state clears only when the **replacement buy fills** or confirmed cash is below the minimum order. `on_canceled_order` resets a dead portfolio sell and leaves a dead buy pending for retry.
4. Signals come from `_portfolio_signal`: historical dips are measured against the *previous* lookback bars (excluding the event day's own high) to match the live check. A holding with no current dip signal is scored as a neutral 0% expected edge — never the old −100% that force-rotated recovered holdings; a replacement requires the target's historical edge to beat the weakest holding by `PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT`.
5. `_posture_adjusted_edge` reshapes each signal's raw historical edge through the configured `PORTFOLIO_RISK_POSTURE` (`conservative`, the default, or `risky`) before ranking candidates and picking the weakest holding: conservative penalizes return variance and a negative news-score day harder and ignores WSB mentions; risky barely discounts variance or bad news and adds a small bonus for WSB-bullish sentiment (a bearish WSB read is penalized under both postures). The news input is per-symbol via `_symbol_news_scores` (built from `NewsContext.per_symbol_scores`, filtered through `SymbolReference.verified_symbols()`, and extended with `scan_text_for_symbols()` for an untagged mention) rather than the market-wide score, falling back to the market-wide score only when a symbol has no dedicated coverage at all. Congress context is deliberately never an input here — it stays research-only, per the documented invariant that it cannot influence symbol choice. None of this changes which already-qualifying candidate looks best and which holding looks weakest — `PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT` itself, and the eligibility floor computed from the raw `expected_profit`/`oos_expected_profit`, never change with posture or per-symbol scoring.

### Fail-safe conventions — preserve these

- News/data outages **fail open**: the price strategy continues without the news veto rather than halting (`NewsContext.available = False`).
- Any exception in the iteration is caught, logged as "failed safely", and retried next cycle; the process must not die mid-market-day.
- The daily email report (`_send_daily_email`, gated by `.last_email_report` date file so it never duplicates a day) is sent from a `finally` block — every exit path fills `report["status"]` first, so new early returns must set it too.
- A corrupt `.news_learning_state.json` is renamed to `.corrupt` and learning restarts clean; malformed entries inside valid JSON are filtered out on load.
- Both two-phase rotations (A/B `pending_rotation` and the portfolio replacement state) are deliberately idempotent across restarts, rejections, and network drops; don't introduce paths that could submit duplicate orders or clear the flag before the buy actually fills.
- The daily email body is mode-aware (`_send_daily_email` renders portfolio holdings/candidates in portfolio mode, A/B prices otherwise); keep new report fields wired into it.
- Secrets never travel through Lumibot's `parameters` dict (it can be logged): `main.py` exports `ALPACA_API_KEY`/`ALPACA_API_SECRET` (read by `news_context.py` — alpaca-py has no env fallback of its own), `EMAIL_SMTP_PASSWORD` (read by `_send_daily_email`), and `LLM_NEWS_API_KEY` (read by `llm_news.py`). SMTP uses `ssl.create_default_context()` for STARTTLS — the stdlib default is unverified; don't regress it.

The README is user-facing documentation for a novice operator and describes behavior in detail — keep it in sync with any behavior or config change.
