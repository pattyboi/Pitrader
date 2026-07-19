#!/usr/bin/env python3
"""CLI viewer for the per-symbol opinion snapshot (signal_snapshot.py).

Purely observational: reads the two small JSON snapshot files the live
equity and crypto iterations already write (.portfolio_signal_snapshot.json,
.crypto_signal_snapshot.json) and prints a colorized table, sectioned by
stocks and crypto. Never touches the broker, never imports lumibot, and
never affects the trading path -- stdlib only, so it's cheap enough to run
any time, including over a slow SSH session on the Pi, and works with any
Python 3 interpreter, not just the project's venv.

Usage: python3 scripts/view_signals.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_SNAPSHOT = BASE_DIR / ".portfolio_signal_snapshot.json"
CRYPTO_SNAPSHOT = BASE_DIR / ".crypto_signal_snapshot.json"

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

USE_COLOR = sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if USE_COLOR else text


def _load_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _age_description(generated_at: str) -> str:
    try:
        generated = datetime.fromisoformat(generated_at)
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        seconds = (datetime.now(timezone.utc) - generated).total_seconds()
    except ValueError:
        return "unknown age"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _print_section(title: str, snapshot: dict[str, Any] | None) -> None:
    print(_color(title, BOLD))
    if snapshot is None:
        print("  no data yet -- the agent hasn't completed an iteration since this was set up\n")
        return
    entries = snapshot.get("symbols") or []
    generated_at = str(snapshot.get("generated_at", "unknown"))
    risk_posture = snapshot.get("risk_posture", "unknown")
    print(f"  {DIM}posture: {risk_posture} | updated {_age_description(generated_at)} ({generated_at}){RESET}")
    if not entries:
        print("  no symbols evaluated yet\n")
        return
    print(f"  {'SYMBOL':<10}{'HELD':<6}{'DIP%':>8}{'EDGE%':>9}  OPINION")
    for entry in entries:
        symbol = str(entry.get("symbol", "?"))
        held = "yes" if entry.get("held") else "no"
        dip = float(entry.get("dip_percent", 0.0))
        edge = float(entry.get("edge_percent", 0.0))
        opinion = str(entry.get("opinion", "-"))
        marker = _color("+", GREEN) if opinion == "+" else _color("-", RED)
        print(f"  {symbol:<10}{held:<6}{dip:>7.2f}%{edge:>8.2f}%  {marker}")
    print()


def main() -> int:
    _print_section("STOCKS", _load_snapshot(PORTFOLIO_SNAPSHOT))
    _print_section("CRYPTO", _load_snapshot(CRYPTO_SNAPSHOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
