"""Per-symbol opinion snapshot shared by both strategy classes.

A small, asset-class-agnostic side channel: both AssetRotationStrategy
(strategy.py) and CryptoRotationStrategy (crypto_strategy.py) call
build_snapshot_entries/write_snapshot at the same point each iteration --
right after every evaluated symbol's posture-adjusted edge (or, for a symbol
not dipping today, its raw historical expected_profit) is known, but before
eligibility filtering trims the list down to actionable candidates -- so the
browser dashboard (scripts/web_dashboard.py) can show the agent's standing
opinion on every symbol it looked at, not just the ones it decided to act on.
Purely
observational: writing this snapshot can never affect a trading decision, and
a failed write is swallowed rather than raised, the same fail-open posture as
the rest of this codebase's advisory-only side channels (e.g. the nightly
pre-evaluation pass).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def build_snapshot_entries(
    signals: Iterable[dict[str, Any]], held: Iterable[str]
) -> list[dict[str, Any]]:
    """One opinion entry per evaluated symbol, sorted alphabetically.

    `edge_percent` is the posture-adjusted edge for a symbol whose dip
    cleared today's threshold, or the raw historical expected profit
    otherwise -- so a currently-idle symbol still carries the agent's
    standing opinion instead of dropping out of the view entirely. The sign
    of `edge_percent` alone decides `opinion` ("+" at or above zero, "-"
    below), matching the CLI viewer's two-state red/green display.
    """
    held_symbols = set(held)
    entries: list[dict[str, Any]] = []
    for signal in signals:
        # This function itself must never raise -- it's called inline as an
        # argument expression at the call site (before write_snapshot's own
        # try/except is even entered), and the module docstring promises
        # writing this snapshot can never affect a trading decision. A
        # malformed entry (missing/non-numeric field) is skipped rather than
        # aborting the rest of the trading iteration.
        try:
            symbol = str(signal["symbol"])
            edge = signal.get("posture_adjusted_edge")
            if edge is None:
                edge = signal.get("expected_profit")
            edge_percent = float(edge) if edge is not None else 0.0
            entries.append(
                {
                    "symbol": symbol,
                    "held": symbol in held_symbols,
                    "qualifies": bool(signal.get("qualifies")),
                    "dip_percent": float(signal.get("dip") or 0.0),
                    "edge_percent": edge_percent,
                    "opinion": "+" if edge_percent >= 0 else "-",
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    entries.sort(key=lambda entry: str(entry["symbol"]))
    return entries


def write_snapshot(
    path: str, generated_at: str, risk_posture: str, entries: list[dict[str, Any]]
) -> None:
    """Best-effort snapshot write. Never raises -- an empty path (feature not
    wired up) or a write failure are both silently skipped, the same
    fail-open posture callers of article_filter.extract_financial_context
    already rely on elsewhere in this codebase."""
    if not path:
        return
    try:
        target = Path(path)
        # Atomic write (temp file + replace), matching this codebase's other
        # state files (e.g. adaptive_news_model.py) -- scripts/web_dashboard.py
        # polls this file continuously, and a truncate-then-write left
        # half-written by a mid-write process kill (a real risk on a Pi) would
        # otherwise show as a momentary "no data" instead of the last good read.
        temporary_path = target.with_suffix(target.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "generated_at": generated_at,
                    "risk_posture": risk_posture,
                    "symbols": entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(target)
    except Exception:
        pass
