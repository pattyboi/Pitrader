# Decision pipeline

Referenced by `/mnt/dietpi_userdata/staging/pi-trading/CLAUDE.md`. Keep this in sync with any behavior change to `_run_portfolio_iteration` or `_run_crypto_iteration`.

## Equity: `AssetRotationStrategy._run_portfolio_iteration` (strategy.py)

Called from `on_trading_iteration`, gated to at most twice a trading day by `_due_iteration_window_now` (market open, and again `PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES` later — default 210). Phase order:

1. **Universe** — `_managed_portfolio_symbols()` (static `PORTFOLIO_SYMBOLS` ∪ discovery-confirmed owned symbols) → `_portfolio_held_positions()` (broker read, filtered to `asset_type in ("stock", "us_equity")`) → `_portfolio_symbols()` (managed ∪ held ∪ one autonomous-discovery batch, if `PORTFOLIO_AUTONOMOUS_DISCOVERY`).
2. **Market-level context** — `_get_news_context()`, `_symbol_news_scores()`, `_llm_assessment_for_iteration()` (may defer the advisory LLM call to after trading — see fail-safes below), `_defer_or_run_discovery_analysis()`, `_update_adaptive_learning()` / `_update_llm_adaptive_learning()`, `_market_veto_reason()` (world-event/LLM/learned-model block check).
3. **Opportunistic Opportunity** — `_opportunistic_opportunity(asset_a, asset_b, ...)` computes dip/forecast/probability via `TradeMemory`, evaluated exactly once as a single non-looped decision, before phase 5 gets to pick from `eligible`.
4. **Restart-safe rotation reconciliation** — `_reconcile_pending_portfolio_rotations(held)` completes or resets any sell-then-buy pair staged in a prior iteration/process life (`.portfolio_rotation_state.json`), before any new decision is made. Returns `claimed_symbols`, the single source of truth preventing a symbol from being touched twice in one pass.
5. **Exits** — `_submit_due_portfolio_exits()` (take-profit/stop-loss/holding-horizon, skipping anything in `claimed_symbols`), `_queue_exit_narratives()` (advisory LLM explanation, deferred).
6. **Signals** — `_portfolio_signals(symbols)` (threaded per-symbol dip/edge computation), `_exclude_unpriceable_discovered_symbols()` (a discovery-only symbol with zero price history is permanently excluded; a config-listed symbol never is), then per evaluated symbol: `_backfill_portfolio_memory()` + `_update_portfolio_memory()` (every evaluated symbol contributes a learning observation, not just qualifying ones) → `_posture_adjusted_edge()` → eligibility filter (qualifies, min observations, min expected profit, out-of-sample floor) → sort.
7. **Opportunistic Opportunity execution** — if eligible (asset_a held & unclaimed, asset_b not held & unclaimed, forecast ready, dip/probability/edge thresholds met, not already swapped today): `_submit_portfolio_rotation_sell(asset_a, asset_b, ...)`, mark `opportunistic_swap_done` in `.portfolio_iteration_state.json` (survives a restart between the day's two windows), update `held_working`/`claimed_symbols`.
8. **Build / replace / top-up** — `_submit_portfolio_builds()` (empty slots, ranked candidates, `_optimal_position_count()`-sized), `_submit_portfolio_replacements()` (swap a weak unclaimed holding for a materially stronger candidate), `_maybe_top_up_portfolio()` (residual cash into the best already-held candidate).
9. **Reporting** — `_summarize_portfolio_actions()`, then (in `on_trading_iteration`'s `finally`) `_record_memory_decision()`, `_generate_daily_narrative()`, `_send_daily_email()`, and last `_start_deferred_llm_analysis()` (advisory-only; must never delay orders, state persistence, or the report).

## Crypto: `CryptoRotationStrategy._run_crypto_iteration` (crypto_strategy.py)

Called from `on_trading_iteration` every `sleeptime` tick (`CRYPTO_ITERATION_INTERVAL_MINUTES`), gated by `market_sessions.nyse_is_open` — no-ops whenever NYSE is open, and whenever `CRYPTO_ENABLED` is false. Deliberately narrower than equity's pipeline (no news/LLM layer, no replace-weak-holding logic): phase order:

1. **Universe** — `_managed_crypto_symbols()` (static `CRYPTO_SYMBOLS` ∪ discovery-confirmed owned) → `_crypto_held_positions()` (filtered to `asset_type == "crypto"`, the mirror-image filter of equity's, so the two never double-count the same shared Alpaca account's holdings).
2. **Restart-safe rotation reconciliation** — `_reconcile_pending_crypto_rotation(held)`, same shape as equity's but scoped to a single pending entry (crypto only ever has one `CRYPTO_ASSET_A`→`CRYPTO_ASSET_B` swap in flight, never many simultaneous replacements).
3. **Exits** — `_crypto_exit_reasons()` (take-profit/stop-loss/holding-horizon, wider defaults than equity since crypto moves more), skipping anything in `claimed_symbols`. Symbols exited this pass are tracked separately (`exited_this_pass`) so a take-profit sale can't be immediately bought back in the same pass if its signal still reads "qualifies".
4. **Signals** — `_crypto_symbols()` (managed ∪ held ∪ one discovery batch), `_crypto_signals()`, unpriceable-discovery exclusion, then per evaluated symbol: `_backfill_crypto_memory()` + `_update_crypto_memory()` → `posture_adjusted_edge()` (from `decision_math.py`) → eligibility filter → sort.
5. **Opportunistic Opportunity** — `_crypto_opportunistic_opportunity(asset_a, asset_b, ...)` computed unconditionally (feeds the email report either way), then executed via `_submit_crypto_rotation_sell()` if eligible (same shape as equity's phase 3/7, capped once/day via `.crypto_opportunistic_swap_state.json`).
6. **Build** — empty-slot buys only (no replace/top-up), sized by `decision_math.optimal_position_count()` against `min(CRYPTO_CASH_ALLOCATION_DOLLARS - deployed, real-time cash)` — the software-enforced soft cap that keeps crypto from spending into the equity strategy's reserve (see architecture.md's config section and the README's "Crypto trading mode").
7. **Reporting** — `_send_crypto_email()` (own daily-dedup state file, own SMTP send, reusing `email_render.py`'s shared HTML helpers).

## Fail-safe conventions (both pipelines — preserve on any change)

- **Fail open on data/API errors.** A broker read failure, a missing quote, a failed news/LLM call, a discovery outage — all are caught, logged, and degrade to "skip this signal" or "use the flat configured fallback," never raised into a crash. Search for `failed safely` in both strategy files for the pattern.
- **`claimed_symbols` is the single source of truth** preventing a symbol from being bought, sold, or rotated twice in one pass. Any new phase that touches positions must read and update it.
- **Restart-safety for multi-step trades.** A sell-then-buy rotation is persisted to disk (`.portfolio_rotation_state.json` / `.crypto_rotation_state.json`) *before* the sell is submitted, and only cleared once the buy leg is confirmed — so a process crash mid-rotation resumes correctly on the next iteration instead of stranding cash or double-selling.
- **Advisory work never blocks the trading path.** LLM analysis, narrative generation, and email sending are deliberately ordered last (or deferred to a background thread) so a slow or failed advisory call can never delay an order or the persisted decision.
- **Once-per-day caps are persisted, not just in-memory**, so a restart between iterations can't accidentally repeat a capped action (the Opportunistic Opportunity swap, in both pipelines).
- **A discovery-only symbol can be permanently excluded on missing price history; a config-listed symbol never can** — a transient data outage must not blacklist something the operator explicitly configured.
