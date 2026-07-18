# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Lumibot/Alpaca trading agent that runs as a systemd service on a Raspberry Pi. Up to twice per trading day (at market open, and again `PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES` later, default 210) it evaluates a bounded dip-signal portfolio over a configured watchlist (default including SPY/QQQ), optionally expanded by autonomous discovery, and takes multiple actions per iteration: opening empty slots, replacing weak holdings, taking profit/cutting losses on existing ones, and an occasional Asset-A→Asset-B "Opportunistic Opportunity" swap trained on historical A-vs-B edge (capped at one swap per day regardless of how many iterations run — see `.claude/docs/decision-pipeline.md`). Fractional shares by default via `PORTFOLIO_FRACTIONAL_SHARES`; whole shares when false. All code lives in `pi-trading-agent/`.

**This can place real orders.** `config.json` holds live-capable Alpaca credentials and `IS_PAPER_TRADING`. Never flip that to `false`, weaken a trade-blocking guard, or loosen config validation without the user explicitly asking. `config.json` must stay mode 600 and out of version control.

## Commands

The pytest suite in `tests/test_safety.py` is the regression net for the decision logic (exit reasons, position-count math, walk-forward validation, posture adjustments, quote/spread fallbacks) — run it after any strategy change. Decision logic is deliberately extracted into small methods testable on a bare `AssetRotationStrategy.__new__` instance with stubbed collaborators; keep new logic in that shape. No linter is configured. Python 3.13 runs on the target; deps are in `requirements.txt` (lumibot, alpaca-py, pandas, pytest), installed into `.venv/` by the installer.

```bash
cd pi-trading-agent

# Install venv + systemd unit, enable and start the service
sudo ./setup_service.sh

# Run the test suite
.venv/bin/python -m pytest tests/ -q

# Run in the foreground (uses .venv, reads config.json from the script's own directory)
.venv/bin/python main.py

# Syntax-check without printing secrets
python3 -m json.tool config.json >/dev/null

# Service lifecycle — always stop before editing config.json or code, then restart
sudo systemctl stop|start|restart trading-agent.service
sudo journalctl -u trading-agent.service -f          # follow logs
sudo journalctl -u trading-agent.service -n 100 --no-pager
```

The service restarts on crash after 30s (`Restart=always`), so a config error shows up as a restart loop — read the journal instead of rerunning the installer. `setup_service.sh` also installs `trading-agent-cpu-watchdog.timer`, which samples the service's cgroup CPU usage every 5 minutes into `.cpu_watchdog.log` and logs a warning (journal tag `trading-agent-cpu-watchdog`) above 10% — a trail for spotting a future CPU regression like the one a stale market-hours cache once caused (`main.py`'s `MarketOpenLoggingAlpaca.market_hours` caches it per day for exactly this reason).

## Architecture

Eight supporting modules plus the one strategy class (portfolio mode is the only mode — the legacy Asset-A/B rotation pipeline, WallStreetBets context, and congressional-trading context have all been removed). Full per-module breakdown: `.claude/docs/architecture.md`.

### Decision pipeline

Portfolio mode's `_run_portfolio_iteration` is the entire decision pipeline: pending-rotation reconciliation, take-profit/stop-loss exits, dip signals with a discovery liquidity floor, posture-adjusted ranking, the Opportunistic Opportunity (occasional A→B swap), and replace/top-up. Full phase-by-phase trace plus the fail-safe conventions that must be preserved on any change: `.claude/docs/decision-pipeline.md`.

The README is user-facing documentation for a novice operator and describes behavior in detail — keep it, `.claude/docs/architecture.md`, and `.claude/docs/decision-pipeline.md` in sync with any behavior or config change.
