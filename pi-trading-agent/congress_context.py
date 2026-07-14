"""Read public congressional-trade aggregates as a research-only signal.

Kadoa's monitor aggregates public STOCK Act disclosures.  Those disclosures can
be filed up to 45 days after a transaction, so this module deliberately reports
them as slow-moving context and never creates, sizes, or vetoes an order.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.request import Request, urlopen


KADOA_TICKERS_URL = (
    "https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/"
    "main/public/data/tickers.json"
)


@dataclass
class CongressContext:
    """Congressional-trading context for the symbols currently being evaluated."""

    available: bool
    tracked_symbols: int = 0
    matched_symbols: int = 0
    highlights: list[str] = field(default_factory=list)
    explanation: str = "Congressional-trading context was not evaluated."


class CongressTradeAnalyzer:
    """Fetch and summarize Kadoa's public per-ticker aggregate dataset."""

    def __init__(
        self,
        url: str = KADOA_TICKERS_URL,
        timeout_seconds: float = 10.0,
        fetcher: Callable[[str, float], Any] | None = None,
    ):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._fetcher = fetcher or self._fetch_json

    @staticmethod
    def _fetch_json(url: str, timeout_seconds: float) -> Any:
        request = Request(url, headers={"User-Agent": "pi-trading-agent/1.0"})
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: fixed public HTTPS URL
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _normalise_symbol(symbol: Any) -> str:
        # Kadoa uses BRK.B whereas broker symbols often use BRK/B.
        return str(symbol).strip().upper().replace("/", ".")

    def analyze(self, symbols: list[str] | set[str] | tuple[str, ...]) -> CongressContext:
        watched = list(dict.fromkeys(self._normalise_symbol(symbol) for symbol in symbols))
        watched = [symbol for symbol in watched if symbol]
        try:
            records = self._fetcher(self.url, self.timeout_seconds)
            if not isinstance(records, list):
                raise ValueError("Kadoa ticker dataset was not a JSON array")
            by_ticker = {
                self._normalise_symbol(record.get("ticker")): record
                for record in records
                if isinstance(record, dict) and self._normalise_symbol(record.get("ticker"))
            }
            matches = [(symbol, by_ticker[symbol]) for symbol in watched if symbol in by_ticker]
            highlights = [self._format_highlight(symbol, record) for symbol, record in matches]
            return CongressContext(
                available=True,
                tracked_symbols=len(watched),
                matched_symbols=len(matches),
                highlights=highlights[:5],
                explanation=(
                    f"Matched {len(matches)}/{len(watched)} monitored symbols against "
                    "Kadoa's public aggregate STOCK Act disclosures. This is delayed "
                    "research context only, not a trading signal."
                ),
            )
        except Exception as exc:
            return CongressContext(
                available=False,
                tracked_symbols=len(watched),
                explanation=(
                    "Congressional-trading context unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    @staticmethod
    def _format_highlight(symbol: str, record: dict[str, Any]) -> str:
        purchases = int(record.get("purchases") or 0)
        sales = int(record.get("sales") or 0)
        filers = int(record.get("filer_count") or 0)
        trade_count = int(record.get("trade_count") or 0)
        balance = purchases - sales
        direction = "net purchases" if balance > 0 else "net sales" if balance < 0 else "balanced"
        return (
            f"{symbol}: {trade_count} disclosed trades by {filers} filers; "
            f"{purchases} purchases / {sales} sales ({direction})."
        )
