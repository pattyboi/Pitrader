"""Bounded, persistent discovery of Alpaca-tradable US equities."""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import requests


class AutonomousUniverse:
    """Rotate through a small batch of active assets without an unbounded scan."""

    ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"
    _SYMBOL = re.compile(r"^[A-Z]{1,5}$")

    def __init__(self, state_path: Path, refresh_days: int, batch_size: int):
        self.state_path = state_path
        self.refresh_days = refresh_days
        self.batch_size = batch_size

    def next_batch(self, api_key: str, secret_key: str) -> list[str]:
        state = self._load()
        today = date.today()
        try:
            refreshed = date.fromisoformat(str(state.get("refreshed", "1970-01-01")))
        except ValueError:
            refreshed = date(1970, 1, 1)
        symbols = state.get("symbols", [])
        if not isinstance(symbols, list) or today - refreshed >= timedelta(days=self.refresh_days):
            response = requests.get(
                self.ASSETS_URL,
                params={"status": "active", "asset_class": "us_equity"},
                headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
                timeout=20,
            )
            response.raise_for_status()
            symbols = sorted(
                item["symbol"].upper()
                for item in response.json()
                if item.get("tradable") is True
                and item.get("fractionable") is True
                and self._SYMBOL.fullmatch(str(item.get("symbol", "")).upper())
            )
            state = {
                "symbols": symbols,
                "cursor": 0,
                "refreshed": today.isoformat(),
                "learned": state.get("learned", []),
            }
        if not symbols:
            return []
        cursor = int(state.get("cursor", 0)) % len(symbols)
        batch = [symbols[(cursor + offset) % len(symbols)] for offset in range(min(self.batch_size, len(symbols)))]
        state["cursor"] = (cursor + len(batch)) % len(symbols)
        self._save(state)
        learned = [symbol for symbol in state.get("learned", []) if self._SYMBOL.fullmatch(str(symbol))]
        return list(dict.fromkeys(learned + batch))

    def remember(self, symbols: list[str], limit: int = 30) -> None:
        """Keep historically qualifying symbols in future daily evaluations."""
        state = self._load()
        learned = [str(symbol).upper() for symbol in state.get("learned", [])]
        learned = [symbol for symbol in learned + symbols if self._SYMBOL.fullmatch(symbol)]
        state["learned"] = list(dict.fromkeys(learned))[-limit:]
        self._save(state)

    def _load(self) -> dict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict) -> None:
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)
