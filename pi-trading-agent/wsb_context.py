"""Public WallStreetBets mention context from AltIndex's live tracker."""

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen


ALTINDEX_WSB_URL = "https://altindex.com/wallstreetbets"
_ROW = re.compile(r"<tr>(.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
_TICKER = re.compile(r"/ticker/([A-Za-z0-9.\-]+)", flags=re.IGNORECASE)
_MENTIONS = re.compile(r"<td>\s*([\d,]+)\s*<br", flags=re.IGNORECASE)
_SENTIMENT = re.compile(r"badge--sentiment-([a-z]+)", flags=re.IGNORECASE)
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@dataclass(frozen=True)
class WSBMention:
    symbol: str
    mentions: int
    sentiment: str


@dataclass
class WSBContext:
    available: bool
    mentions: list[WSBMention] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    explanation: str = "WallStreetBets context was not evaluated."


class WallStreetBetsAnalyzer:
    """Read AltIndex's public rendered WSB tracker without using its paid API."""

    def __init__(
        self,
        url: str = ALTINDEX_WSB_URL,
        timeout_seconds: float = 10.0,
        fetcher: Callable[[str, float], str] | None = None,
    ):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._fetcher = fetcher or self._fetch_html

    @staticmethod
    def _fetch_html(url: str, timeout_seconds: float) -> str:
        request = Request(url, headers={"User-Agent": "pi-trading-agent/1.0"})
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: fixed public HTTPS URL
            return response.read().decode("utf-8", errors="replace")

    @classmethod
    def parse(cls, page: str) -> list[WSBMention]:
        """Extract the public tracker table, tolerating unrelated page markup."""
        records: list[WSBMention] = []
        seen: set[str] = set()
        for row in _ROW.findall(html.unescape(page)):
            ticker_match = _TICKER.search(row)
            mentions_match = _MENTIONS.search(row)
            sentiment_match = _SENTIMENT.search(row)
            if not ticker_match or not mentions_match or not sentiment_match:
                continue
            symbol = ticker_match.group(1).upper()
            if not _SYMBOL.fullmatch(symbol) or symbol in seen:
                continue
            seen.add(symbol)
            records.append(
                WSBMention(
                    symbol=symbol,
                    mentions=int(mentions_match.group(1).replace(",", "")),
                    sentiment=sentiment_match.group(1).lower(),
                )
            )
        return records

    def analyze(self, watched_symbols: list[str] | set[str] | tuple[str, ...]) -> WSBContext:
        try:
            return self.context_from_mentions(
                self.parse(self._fetcher(self.url, self.timeout_seconds)), watched_symbols
            )
        except Exception as exc:
            return WSBContext(
                available=False,
                explanation=f"WallStreetBets context unavailable: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def context_from_mentions(
        mentions: list[WSBMention], watched_symbols: list[str] | set[str] | tuple[str, ...], source: str = "live"
    ) -> WSBContext:
        watched = {str(symbol).upper() for symbol in watched_symbols}
        highlights = [
            f"{item.symbol}: {item.mentions} WSB mentions, {item.sentiment} sentiment."
            for item in mentions
            if item.symbol in watched
        ]
        return WSBContext(
            available=True,
            mentions=mentions,
            highlights=highlights[:5],
            explanation=(
                f"Used {len(mentions)} public AltIndex WallStreetBets tracker rows ({source}). "
                "Mentions are research and discovery context only, not an entry signal."
            ),
        )


class WallStreetBetsSnapshot:
    """Persist one WSB snapshot and refresh it no more often than every 24 hours."""

    REFRESH_INTERVAL = timedelta(hours=24)

    def __init__(self, state_path: Path, analyzer: WallStreetBetsAnalyzer):
        self.state_path = state_path
        self.analyzer = analyzer

    def refresh_if_due(self) -> bool:
        """Fetch once before trading when absent or at least 24 hours old."""
        fetched_at, mentions = self._load()
        now = datetime.now(timezone.utc)
        if fetched_at is not None and mentions and now - fetched_at < self.REFRESH_INTERVAL:
            return False
        page = self.analyzer._fetcher(self.analyzer.url, self.analyzer.timeout_seconds)
        mentions = self.analyzer.parse(page)
        if not mentions:
            raise ValueError("AltIndex page contained no parseable WSB tracker rows")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "fetched_at": now.isoformat(),
                    "mentions": [item.__dict__ for item in mentions],
                }
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)
        return True

    def context(self, watched_symbols: list[str] | set[str] | tuple[str, ...]) -> WSBContext:
        try:
            refreshed = self.refresh_if_due()
            _, mentions = self._load()
            return self.analyzer.context_from_mentions(
                mentions, watched_symbols, "refreshed 24-hour snapshot" if refreshed else "cached 24-hour snapshot"
            )
        except Exception as exc:
            _, mentions = self._load()
            if mentions:
                context = self.analyzer.context_from_mentions(mentions, watched_symbols, "stale cached snapshot")
                context.explanation += f" Refresh failed safely: {type(exc).__name__}."
                return context
            return WSBContext(
                available=False,
                explanation=f"WallStreetBets context unavailable: {type(exc).__name__}: {exc}",
            )

    def _load(self) -> tuple[datetime | None, list[WSBMention]]:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(str(state["fetched_at"])).astimezone(timezone.utc)
            mentions = [
                WSBMention(str(item["symbol"]).upper(), int(item["mentions"]), str(item["sentiment"]).lower())
                for item in state.get("mentions", [])
                if isinstance(item, dict) and _SYMBOL.fullmatch(str(item.get("symbol", "")).upper())
            ]
            return fetched_at, mentions
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None, []
