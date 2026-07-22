#!/usr/bin/env python3
"""Keep the crypto service resident only while NYSE regular hours are closed."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from market_sessions import nyse_is_open


SERVICE_NAME = "trading-agent-crypto.service"


def reconcile_crypto_service(
    config_path: Path,
    *,
    now_utc: datetime | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Start or stop crypto according to configuration and the NYSE calendar."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    enabled = bool(config.get("CRYPTO_ENABLED", False))
    should_run = enabled and not nyse_is_open(now_utc or datetime.now(timezone.utc))
    active = run(
        ["systemctl", "is-active", "--quiet", SERVICE_NAME],
        check=False,
    ).returncode == 0
    if should_run and not active:
        run(["systemctl", "start", SERVICE_NAME], check=True)
        return "started"
    if not should_run and active:
        run(["systemctl", "stop", SERVICE_NAME], check=True)
        return "stopped"
    return "unchanged"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or [])
    config_path = Path(args[0]) if args else PROJECT_DIR / "config.json"
    reconcile_crypto_service(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
