# Decision pipeline

Referenced by `/mnt/dietpi_userdata/staging/pi-trading/CLAUDE.md`. Keep this in sync with any behavior change to `_run_portfolio_iteration` or `_run_crypto_iteration`.

## Equity: `AssetRotationStrategy._run_portfolio_iteration` (strategy.py)

Called from `on_trading_iteration`, gated to at most twice a trading day by `_due_iteration_window_now` (market open, and again `PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES` later — default 210). Phase order:

1. **Account/universe base** — reset per-iteration order/quote caches; `_managed_portfolio_symbols()` → `_portfolio_held_positions()` (stock/equity filter). Abort safely if positions are unavailable.
2. **Immutable decision context** — `_prepare_portfolio_iteration_context()` expands managed ∪ held ∪ discovery symbols, refreshes symbol references, fetches news, attributes per-symbol keyword scores, runs the strict-schema LLM assessment, records nightly pre-evaluation context, screens discovery-only article risks, and returns one `IterationContext` with the sole purchase `veto_reason`. Keyword scores only prioritize/attribute; the signed aggregate LLM score feeds ranking, learned memory, and bounded exposure.
3. **Opportunistic Opportunity** — `_opportunistic_opportunity(asset_a, asset_b, ...)` computes dip/forecast/probability via `TradeMemory`, evaluated exactly once as a single non-looped decision, before phase 5 gets to pick from `eligible`.
4. **Restart-safe rotation reconciliation** — `_reconcile_pending_portfolio_rotations(held)` completes or resets any sell-then-buy pair staged in a prior iteration/process life (`.portfolio_rotation_state.json`), before any new decision is made. Returns `claimed_symbols`, the single source of truth preventing a symbol from being touched twice in one pass.
5. **Exits** — `_submit_due_portfolio_exits()` (take-profit/stop-loss/holding-horizon, skipping anything in `claimed_symbols`), `_queue_exit_narratives()` (advisory LLM explanation, deferred).
6. **Signals** — threaded `_portfolio_signals()`, discovery-only unpriceable exclusion, batch historical-memory backfill/update, posture + aggregate/per-symbol-news adjustment, then hard eligibility filters (dip, observations, expected profit, OOS result, learned edge) and sort. Every usable symbol contributes a memory observation; only qualifying dip days train the fitted signal model.
7. **Opportunistic Opportunity execution** — if eligible (asset_a held & unclaimed, asset_b not held & unclaimed, forecast ready, dip/probability/edge thresholds met, not already swapped today): `_submit_portfolio_rotation_sell(asset_a, asset_b, ...)`, mark `opportunistic_swap_done` in `.portfolio_iteration_state.json` (survives a restart between the day's two windows), update `held_working`/`claimed_symbols`.
8. **Build / replace / top-up** — compute this pipeline's account allocation and effective slot ceiling. With `PORTFOLIO_FILL_QUALIFIED_SLOTS=true`, `qualified_position_count()` admits every fundable qualified slot up to the configured ceiling; `false` uses `_optimal_position_count()`'s narrower variance-aware heuristic. Then build empty slots, replace materially weaker unclaimed holdings, and top up the best already-held candidate when no build/replacement fired. Negative non-blocking LLM scores scale deployment (25-100%) but never expand eligibility or capacity.
9. **Reporting** — `_summarize_portfolio_actions()`, then (in `on_trading_iteration`'s `finally`) `_record_memory_decision()`, `_generate_daily_narrative()`, and `_send_daily_email()`.

## Crypto: `CryptoRotationStrategy._run_crypto_iteration` (crypto_strategy.py)

Called from `on_trading_iteration` every `sleeptime` tick (`CRYPTO_ITERATION_INTERVAL_MINUTES`), gated by `market_sessions.nyse_is_open` — no-ops whenever NYSE is open, and whenever `CRYPTO_ENABLED` is false. Deliberately narrower than equity's pipeline (no replace-weak-holding logic): phase order:

1. **Account/universe base** — reset caches, load managed crypto symbols and crypto-only positions, then abort safely if either is unavailable.
2. **Restart-safe rotation reconciliation** — `_reconcile_pending_crypto_rotation(held)`, same shape as equity's but scoped to a single pending entry (crypto only ever has one `CRYPTO_ASSET_A`→`CRYPTO_ASSET_B` swap in flight, never many simultaneous replacements).
3. **Exits** — `_crypto_exit_reasons()` (take-profit/stop-loss/holding-horizon, wider defaults than equity since crypto moves more), skipping anything in `claimed_symbols`. Symbols exited this pass are tracked separately (`exited_this_pass`) so a take-profit sale can't be immediately bought back in the same pass if its signal still reads "qualifies".
4. **Pre-purchase context** — `_prepare_crypto_iteration_context()` establishes the universe, fetches/caches `BASEUSD` news and the strict-schema LLM assessment, derives the shared purchase veto, computes signals, excludes only unpriceable discovery symbols, and returns one `IterationContext`. Exits already ran, so model latency cannot delay protection.
5. **Signals/memory** — batch backfill/update pooled memory, apply posture plus aggregate/per-symbol news, enforce raw dip/observation/profit/OOS/learned-edge floors, and sort.
6. **Opportunistic Opportunity** — `_crypto_opportunistic_opportunity(asset_a, asset_b, ...)` computed unconditionally (feeds the email report either way), then executed via `_submit_crypto_rotation_sell()` if eligible and the crypto news/LLM guard is clear (same shape as equity's phase 3/7, capped once/day via `.crypto_opportunistic_swap_state.json`).
7. **Build** — target half the shared account value, subtract deployed crypto, and apply the bounded LLM exposure multiplier. `CRYPTO_FILL_QUALIFIED_SLOTS=true` uses every fundable qualified slot up to `CRYPTO_MAX_POSITIONS`; `false` uses `optimal_position_count()`. Submit empty-slot buys only (no replacement/top-up), skipping claimed or just-exited symbols.
8. **Reporting** — `_send_crypto_email()` (own daily-dedup state file, own SMTP send, reusing `email_render.py`'s shared HTML helpers).

## Fail-safe conventions (both pipelines — preserve on any change)

- **Contain data/API errors.** Broker reads, quotes, news/LLM calls, and discovery outages are caught and logged rather than crashing the process. An enabled LLM follows `LLM_NEWS_FAIL_CLOSED_ON_UNAVAILABLE` for new purchases.
- **`claimed_symbols` is the single source of truth** preventing a symbol from being bought, sold, or rotated twice in one pass. Any new phase that touches positions must read and update it.
- **Restart-safety for multi-step trades.** A sell-then-buy rotation is persisted to disk (`.portfolio_rotation_state.json` / `.crypto_rotation_state.json`) *before* the sell is submitted, and only cleared once the buy leg is confirmed — so a process crash mid-rotation resumes correctly on the next iteration instead of stranding cash or double-selling.
- **One aggregate news guard.** Keyword scoring attributes/prioritizes articles and may adjust ordering for the affected symbol, but it never independently vetoes. The strict-schema LLM score is the sole aggregate news veto/exposure input. Protective exits and restart reconciliation run first and never depend on it.
- **Once-per-day caps are persisted, not just in-memory**, so a restart between iterations can't accidentally repeat a capped action (the Opportunistic Opportunity swap, in both pipelines).
- **A discovery-only symbol can be permanently excluded on missing price history; a config-listed symbol never can** — a transient data outage must not blacklist something the operator explicitly configured.
