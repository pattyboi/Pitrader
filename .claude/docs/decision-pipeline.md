# Decision pipeline

Referenced from `CLAUDE.md`. Portfolio mode is the only mode. `initialize()` sets `sleeptime = "10M"` — a cheap poll, not the evaluation cadence itself — and `on_trading_iteration` calls `_due_iteration_window_now()` first: it returns `None` (skip, cost-free) on most polls and only the label `"open"` or `"midday"` when that window's time has arrived, gated by `_due_portfolio_iteration_window` (pure/tested in `tests/test_safety.py`) against `.portfolio_iteration_state.json` (today's `windows_completed`, restart-safe, reset when the calendar date rolls over). Only then does `on_trading_iteration` build `report` and call `_run_portfolio_iteration(report)`, and in its `finally` block always call `_record_memory_decision`, then `_generate_daily_narrative` (a purely descriptive LLM recap of `report`, stored as `report["daily_narrative"]` — see `architecture.md`'s `llm_news.py` entry), then `_send_daily_email`. Net effect: the full pipeline below runs at most twice a trading day, at market open and `PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES` (default 210) after it.

## Diagram

```mermaid
flowchart TD
    Poll["Lumibot poll: every 10 minutes<br/>(self.sleeptime = '10M')"] --> Gate{"_due_iteration_window_now()<br/>is 'open' or 'midday' window due today?"}
    Gate -- "no (already run, before open, etc.)" --> Skip["Return: no evaluation this poll"]
    Gate -- "yes" --> Iter["on_trading_iteration()<br/>calls _run_portfolio_iteration(report)"]

    subgraph CTX[Context gathering]
        direction TB
        Managed["_managed_portfolio_symbols()<br/>static watchlist ∪ managed-discovered ∪ held ∪ pending-rotation"]
        Held["_portfolio_held_positions()<br/>broker cost basis + quantities"]
        HeldFail{"positions unavailable?"}
        Universe["_portfolio_symbols()<br/>managed ∪ held ∪ one discovery batch<br/>(mutates AutonomousUniverse's batch cursor)"]
        SymRef["_refresh_symbol_reference()<br/>(background thread)"]
        News["_get_news_context()<br/>Alpaca headlines + keyword score"]
        SymScores["_symbol_news_scores()<br/>per-symbol cross-checked coverage"]
        LLM["_get_llm_news_assessment()<br/>Ollama market-wide risk read"]
        Nightly["_load_nightly_preeval_learnings()<br/>today's overnight verdict-cache summary, if any<br/>(surfaced in the report only, never gates anything)"]
    end
    Iter --> Managed --> Held --> HeldFail
    HeldFail -- yes --> Abort["status = 'positions unavailable', stop"]
    HeldFail -- no --> Universe --> SymRef --> News --> SymScores --> LLM --> Nightly

    subgraph SCREEN[Discovery screening & learning]
        direction TB
        RedFlags["_check_discovery_red_flags()<br/>discovery-only symbols with negative score<br/>advisory, or excludes if<br/>PORTFOLIO_DISCOVERY_LLM_BLOCK_ENABLED"]
        ArticleCtx["_check_discovery_article_context()<br/>same discovery-only symbols, live —<br/>cache hit if the nightly pass already<br/>covered them; advisory only, never excludes"]
        AdaptiveKw["_update_adaptive_learning()<br/>keyword score → next-session return"]
        AdaptiveLLM["_update_llm_adaptive_learning()<br/>LLM score → next-session return<br/>(observational only, not yet wired to the veto)"]
    end
    Nightly --> RedFlags --> ArticleCtx --> AdaptiveKw --> AdaptiveLLM

    AdaptiveLLM --> Veto{"_market_veto_reason()<br/>news score ≤ threshold?<br/>LLM score ≤ threshold?<br/>adaptive forecast ≤ threshold (correlation-gated)?"}

    subgraph EXEC[Trade execution phases — one pass can run several]
        direction TB
        P0["Phase 0: reconcile pending rotations<br/>(never vetoed — the sale already happened)"]
        P1["Phase 1: exits<br/>take-profit / stop-loss / holding-horizon<br/>(never vetoed, single-leg sell)"]
        Opportunity["_opportunistic_opportunity()<br/>A-vs-B win probability + edge<br/>(trade_memory.py regression)"]
        Rank["Rank eligible signals by<br/>posture-adjusted edge<br/>(_posture_adjusted_edge)"]
        P2["Phase 2: Opportunistic Opportunity swap<br/>Asset A → Asset B, at most once/day"]
        P3a["Phase 3a: build empty slots"]
        P3b["Phase 3b: replace weakest holding(s)"]
        P3c["Phase 3c: top up if nothing else fired"]
    end
    Veto --> P0 --> P1 --> Opportunity --> Rank --> VetoGate{"veto_reason set?"}
    VetoGate -- "yes — new exposure blocked" --> SkipTrades["Phase 2 and 3 skipped this pass"]
    VetoGate -- no --> P2 --> P3a --> P3b --> P3c

    subgraph FIN["Finalization — always runs, even after a caught error"]
        direction TB
        Memory["_record_memory_decision()<br/>portfolio_memory.py / trade_memory.py"]
        Narrative["_generate_daily_narrative()<br/>LLM 2-3 sentence recap, if enabled"]
        Email["_send_daily_email()<br/>at most one per calendar day"]
    end
    P3c --> Memory --> Narrative --> Email
    SkipTrades --> Memory
    Abort --> Memory
```

### Off-hours side channel: the nightly pre-evaluation pass

`scripts/nightly_preeval.py`, run once at 03:00 ET by
`trading-agent-nightly-preeval.timer`, calls `_run_nightly_preevaluation` — a
separate process, a separate strategy instance, entirely outside the diagram
above. It reads `_managed_portfolio_symbols()` (read-only) ∪ currently-held
positions — deliberately **not** `_portfolio_symbols`, which calls
`AutonomousUniverse.next_batch()` and would consume a discovery batch the
live morning iteration should get instead — fetches a fresh `NewsContext`,
and calls `_check_discovery_article_context(..., require_negative_score=False)`
over every one of those symbols (not just discovery's negative-news ones, as
the live call below still does by default), populating
`.article_verdicts.duckdb`'s same-day cache. It persists a summary to
`.nightly_preeval_state.json`, which `_load_nightly_preeval_learnings` reads
back into the live report's "Learned at night" email line (see
`architecture.md`'s `article_filter.py` entry) — informational only, it
cannot itself gate a trade. The point is cache warmth, not a new signal: when
the live pipeline's own `_check_discovery_article_context` step reaches a
symbol the nightly pass already checked, `article_filter.extract_financial_context`'s
cache lookup hits and skips the Ollama round-trip.

## `_run_portfolio_iteration`

1. Read **all** account stock positions via `_portfolio_held_positions()`; a broker read failure aborts the iteration (an empty dict would look like an empty portfolio and trigger duplicate buys). Held symbols are always merged into the evaluation universe and re-`remember()`ed daily so an orphaned position is impossible.
2. Fetch news, the LLM assessment, and update adaptive learning (Asset B proxy). `_update_adaptive_learning` trains `adaptive_news_model.py`'s regression on the deterministic keyword `news_context.score`; `_update_llm_adaptive_learning` trains a second, separately-persisted instance of the same regression on `llm_assessment.score` instead, purely so the two signals' actual predictive value (next-session return) can be compared over time — its forecast (`llm_learned_forecast` in `report`) is observational only and does not feed `_market_veto_reason`. `_market_veto_reason` combines the three configured vetoes — fixed keyword news score (`NEWS_BLOCK_ON_HIGH_RISK`), LLM assessment (`LLM_NEWS_BLOCK_ON_HIGH_RISK`), and the mature keyword-trained learned forecast (`NEWS_PREDICTED_RETURN_BLOCK_PERCENT` once correlation qualifies) — and blocks **new** purchases/replacements only; completing an in-flight replacement or exit is never vetoed. `_refresh_symbol_reference` runs here too (once the day's evaluation universe is final, before news scoring is consumed) — cheap and weekly-gated, never blocks trading on failure. `_load_nightly_preeval_learnings` also runs here, surfacing last night's overnight pass (see "Off-hours side channel" above) into `report["nightly_learned_summary"]` for the email — read-only, it never affects ranking, screening, or the veto below. Right after, `_check_discovery_red_flags` screens only the symbols discovery itself just added to `symbols` this iteration (never held/static ones) that have negative `_symbol_news_scores` coverage; a flagged symbol is logged/reported always, and only actually dropped from `symbols` (before any phase below sees it) when `PORTFOLIO_DISCOVERY_LLM_BLOCK_ENABLED` is true — advisory-by-default, same posture as `LLM_NEWS_BLOCK_ON_HIGH_RISK`. `_check_discovery_article_context` runs immediately after over the same candidates (see `architecture.md`'s `article_filter.py` entry) — purely advisory, logged/reported only, never excludes a symbol; it takes a `require_negative_score` flag (default `True`, unchanged here) so the nightly pass below can reuse it over the full universe regardless of sentiment.
3. **Phase 0 — reconcile pending rotations**: `.portfolio_rotation_state.json` entries are resolved first. Still holding the source with an active sell → wait. Dead sell (order died / lost across restart) → reset and re-evaluate. Sale done → buy the target (also continued immediately in `on_filled_order` when the sale fills). The state only clears when the **replacement buy fills** or confirmed cash is below the minimum order. `on_canceled_order` resets a dead sell and leaves a dead buy pending for retry.
4. **Phase 1 — exit management**: every current holding (not just one) is checked by `_portfolio_exit_reasons` for take-profit/stop-loss against the broker's cost basis and the live bid, plus a holding-horizon backstop. This is a plain single-leg sell, never vetoed, exactly like completing a pending rotation. `_generate_exit_narrative` then adds one purely descriptive LLM sentence to the `portfolio_actions` log/email when the exited symbol has dedicated news coverage today — the sale itself already happened on price alone.
5. **Signals**: `_portfolio_signal` computes each candidate's historical edge, measuring dips against the *previous* lookback bars (excluding the event day's own high) to match the live check. A holding with no current dip signal scores a neutral 0% expected edge (never force-rotated just for lacking a signal). It also enforces a discovery liquidity floor — reject if price is below `PORTFOLIO_DISCOVERY_MIN_PRICE_DOLLARS` or the recent average volume is below `PORTFOLIO_DISCOVERY_MIN_AVG_VOLUME` (either 0 disables its check) — applied to the whole evaluated universe (watchlist, holdings, and discovery candidates alike), not just new discoveries, since the data is already in hand at that point.
6. `_portfolio_signal` now always returns a context dict once bars/price/liquidity-floor pass, whether or not today's dip clears `dip_threshold_percent` — a `qualifies` field (today's dip ≥ threshold *and* at least one historical comparable dip exists) records which. Every symbol reaching this point — every watched, held, or candidate symbol, not just one with a qualifying dip today — updates `portfolio_memory.py`'s pooled cross-symbol model (`_update_portfolio_memory`, plus a once-per-symbol price-only backfill via `_backfill_portfolio_memory`) with at least five daily facts (dip %, news score, live spread, recent average volume, historical backtest edge/win-probability/stdev); `signal_present=qualifies` keeps the pooled ridge regression trained only on decision-specific dip days — this broader daily context is what "learning" means across the whole evaluated universe now (see `architecture.md`). Ranking fields (`posture_adjusted_edge`, `learned_edge`) are computed only for `qualifies=True` signals; `_posture_adjusted_edge` then reshapes each one's raw historical edge through `PORTFOLIO_RISK_POSTURE` (`conservative`, default, or `risky`) before ranking candidates and picking the weakest holding: conservative penalizes return variance and a negative news-score day harder; risky barely discounts either. The news input is per-symbol via `_symbol_news_scores` (from `NewsContext.per_symbol_scores`, filtered through `SymbolReference.verified_symbols()`, extended with `scan_text_for_symbols()` for untagged mentions), falling back to the market-wide score only when a symbol has no dedicated coverage. When PortfolioMemory's forecast is ready, its disagreement with the raw historical `expected_profit` becomes one more ranking adjustment, weighted by posture (risky trusts it more) exactly like the variance/news terms. None of this changes eligibility itself — the `eligible` filter and the weakest-holding lookup (`held_signals`) both require `qualifies=True` explicitly (never inferred from absence, now that non-qualifying symbols are present in `signals` too), and `PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT`/the OOS floor are computed from the raw `expected_profit`/`oos_expected_profit` and never shift with posture, per-symbol scoring, or the learned edge.
7. **Phase 2 — Opportunistic Opportunity**: evaluated exactly once per iteration, before Phase 3 gets to pick from eligible candidates. This is `trade_memory.py`'s two-feature regression (dip size + news score) forecasting the next-session Asset-B-minus-Asset-A edge; if Asset A is held, Asset B isn't, and the forecast clears its probability/edge/dip thresholds, it submits an A→B swap via the same rotation-state mechanism as Phase 0. Reserving both legs in `claimed_symbols` keeps it structurally distinct from — never competing for a slot within — the Phase 3 batch below. Now that a trading day can run this function up to twice (see the cadence note above), "at most one swap per day" is enforced explicitly: `opportunity_is_eligible` also requires `.portfolio_iteration_state.json`'s `opportunistic_swap_done` to still be false, which is set `True` the moment a swap actually submits and persists across a restart between the day's two windows.
8. **Phase 3 — replace/top-up**: builds empty slots first, then replaces weak holdings, then tops up, looping over every remaining ranked candidate this iteration (not just the single best one). A replacement requires the target's posture-adjusted historical edge to beat the weakest holding by `PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT`. `_optimal_position_count` narrows the configured `PORTFOLIO_MAX_POSITIONS` ceiling to what today's total capital (holdings + spendable cash) and candidate quality actually support.

## Fail-safe conventions — preserve these

- News/data outages **fail open**: the strategy continues without the news veto rather than halting (`NewsContext.available = False`).
- Any exception in the iteration is caught, logged as "failed safely", and retried next cycle; the process must not die mid-market-day.
- The daily email report (`_send_daily_email`, gated by `.last_email_report` date file so it never duplicates a day) is sent from a `finally` block — every exit path fills `report["status"]` first, so new early returns must set it too. On a day with two iterations, this means only the `"open"` window's report gets emailed; the `"midday"` window still runs the full pipeline (and can still trade) but its report is silently not re-mailed.
- A corrupt `.news_learning_state.json` is renamed to `.corrupt` and learning restarts clean; malformed entries inside valid JSON are filtered out on load.
- The portfolio replacement state (`.portfolio_rotation_state.json`) is deliberately idempotent across restarts, rejections, and network drops; don't introduce paths that could submit duplicate orders or clear the flag before the buy actually fills.
- Secrets never travel through Lumibot's `parameters` dict (it can be logged): `main.py` exports `ALPACA_API_KEY`/`ALPACA_API_SECRET` (read by `news_context.py` — alpaca-py has no env fallback of its own) and `EMAIL_SMTP_PASSWORD` (read by `_send_daily_email`). `llm_news.py` needs no secret at all — it only ever talks to a local Ollama server. SMTP uses `ssl.create_default_context()` for STARTTLS — the stdlib default is unverified; don't regress it.
- The nightly pre-evaluation pass (`scripts/nightly_preeval.py` → `_run_nightly_preevaluation`) is a read-only, cache-only side channel: it must never call `_portfolio_symbols`/`AutonomousUniverse.next_batch` (that batch cursor is exclusively the live iteration's to consume), and its findings can only ever warm `.article_verdicts.duckdb`'s cache or annotate the email's "Learned at night" line — never gate a trade. It fails open like everything else here: any error inside it is caught and logged, and the script still exits `0` so a bad night never blocks the systemd timer from being considered healthy.
