# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Lumibot/Alpaca trading agent that runs as a systemd service on a Raspberry Pi. Once per trading day it checks whether Asset B (default QQQ) has dipped a configured percent from its recent high; if so and the account holds Asset A (default SPY), it sells all of A and buys as many whole shares of B as cash allows. It does not rotate back from B to A. All code lives in `pi-trading-agent/`.

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

Four modules, one strategy class:

- `main.py` — loads and strictly validates every key in `config.json` (`load_config` raises on any missing/invalid value; new config keys must be added to the `required` set, validated, and threaded into the strategy `parameters` dict), builds the Lumibot `Alpaca` broker and `Trader`, and runs `AssetRotationStrategy`.
- `strategy.py` — `AssetRotationStrategy(Strategy)` with `sleeptime = "1D"`. `on_trading_iteration` is the whole decision pipeline; `on_filled_order` completes the rotation.
- `news_context.py` — `WorldEventAnalyzer` fetches Alpaca news and produces a deterministic keyword score (`SEVERE_RISK_TERMS` −3, `RISK_TERMS` −1, `OPPORTUNITY_TERMS` +1, each counted once per article) wrapped in a `NewsContext`.
- `adaptive_news_model.py` — `AdaptiveNewsModel`, a ridge-stabilized one-variable regression of next-day Asset B return on the news score, persisted in `.news_learning_state.json`. It only becomes authoritative after `NEWS_LEARNING_MIN_OBSERVATIONS` samples *and* sufficient `|correlation|`.

### Decision pipeline order (in `on_trading_iteration`)

1. Fetch news context and prices; bail out (no trade) if prices are missing or non-positive.
2. Update the adaptive model with today's score/price.
3. If `vars.pending_rotation` is set, reconcile against live positions and open orders: still holding A with an active sell → wait; holding A with no active sell (order died / lost across restart) → clear the flag and fall through to a fresh evaluation; A gone → buy B (or finish the rotation if cash can't cover one share).
4. Compute dip from the max daily high over the lookback; require `dip >= threshold` and a long Asset A position.
5. Two independent vetoes, either blocks the trade: the fixed news score (`score <= NEWS_HIGH_RISK_SCORE` when `NEWS_BLOCK_ON_HIGH_RISK`) and the mature learned forecast (`predicted return <= NEWS_PREDICTED_RETURN_BLOCK_PERCENT` when correlation qualifies).
6. Submit the A market sale and set the pending flag via `_set_pending_rotation(True)`, which persists to `.rotation_state.json` so restarts can't strand mid-rotation cash. The B buy happens in `on_filled_order` when the sale fills (next daily iteration as fallback); the flag clears only when the B **buy fills** (or cash drops below one share), never on submission. `on_canceled_order` resets a dead sell to idle and leaves a dead buy pending for retry. `_buy_asset_b_with_available_cash` holds back `CASH_BUFFER_FRACTION` (1%) of cash so the market order can't be rejected on a price uptick, checks for an already-working buy order before submitting, and is serialized by `_rotation_lock` against the broker-callback thread.

### Fail-safe conventions — preserve these

- News/data outages **fail open**: the price strategy continues without the news veto rather than halting (`NewsContext.available = False`).
- Any exception in the iteration is caught, logged as "failed safely", and retried next cycle; the process must not die mid-market-day.
- The daily email report (`_send_daily_email`, gated by `.last_email_report` date file so it never duplicates a day) is sent from a `finally` block — every exit path fills `report["status"]` first, so new early returns must set it too.
- A corrupt `.news_learning_state.json` is renamed to `.corrupt` and learning restarts clean; malformed entries inside valid JSON are filtered out on load.
- The two-phase rotation (persisted `pending_rotation` + whole-share cash buy) is deliberately idempotent across restarts, rejections, and network drops; don't introduce paths that could submit duplicate orders or clear the flag before the buy actually fills.
- Secrets never travel through Lumibot's `parameters` dict (it can be logged): `main.py` exports `ALPACA_API_KEY`/`ALPACA_API_SECRET` (read by `news_context.py` — alpaca-py has no env fallback of its own) and `EMAIL_SMTP_PASSWORD` (read by `_send_daily_email`). SMTP uses `ssl.create_default_context()` for STARTTLS — the stdlib default is unverified; don't regress it.

The README is user-facing documentation for a novice operator and describes behavior in detail — keep it in sync with any behavior or config change.
