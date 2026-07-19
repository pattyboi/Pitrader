"""Daily trade counter shared by both strategy classes.

A companion side channel to signal_snapshot.py, same fail-open posture:
both AssetRotationStrategy (strategy.py) and CryptoRotationStrategy
(crypto_strategy.py) call record_trade from their single order-submission
choke point (_submit_order_checked) whenever the broker actually accepts an
order, so scripts/web_dashboard.py can show a running "trades today" count
without touching the broker or the decision path itself. Purely
observational: incrementing this counter can never affect a trading
decision, and a failed read/write is swallowed rather than raised.
"""

from __future__ import annotations

import json
from pathlib import Path


def record_trade(path: str, today: str) -> None:
    """Best-effort increment of today's trade count.

    Resets to 1 instead of incrementing when the stored date doesn't match
    `today`, so a stale count from a previous trading day never leaks into
    the current one. Never raises -- an empty path (feature not wired up),
    a missing/corrupt file, or a write failure are all silently skipped.
    """
    if not path:
        return
    try:
        file = Path(path)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("date") != today:
            data = {"date": today, "count": 0}
        data["count"] = int(data.get("count", 0)) + 1
        file.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
