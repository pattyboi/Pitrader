"""Local, cross-checked reference mapping of tradable ticker to company name.

Alpaca's news API tags each article with the symbols it mentions, but a
mis-tagged or spurious symbol association would quietly corrupt per-symbol
news scoring. This module builds a small local mapping from two independent
public sources -- Alpaca's own asset directory and the SEC's public ticker
dataset -- and only treats a symbol as recognized once at least one of them
confirms it, refreshed on a multi-day interval rather than every iteration.

Like the other context modules, this never creates a trade, chooses a
symbol, or vetoes a decision on its own; it only decides whether an
already-collected news-to-symbol association is trustworthy enough to use.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

import duckdb
import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# Same per-mode host split as autonomous_universe.py: paper keys are
# rejected by the live host and vice versa.
ALPACA_ASSETS_URL_PAPER = "https://paper-api.alpaca.markets/v2/assets"
ALPACA_ASSETS_URL_LIVE = "https://api.alpaca.markets/v2/assets"

_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|"
    r"group|holdings?|class\s+[ab]|common\s+stock)\b\.?",
    re.IGNORECASE,
)
_NON_WORD = re.compile(r"[^a-z0-9\s]")


@dataclass
class SymbolRecord:
    ticker: str
    alpaca_name: str | None
    sec_name: str | None
    verified: bool


class SymbolReference:
    """Cross-checked local mapping of ticker to company name."""

    def __init__(
        self,
        database_path: Path,
        refresh_days: int,
        paper: bool = True,
        alpaca_fetcher: Callable[[str, dict], Any] | None = None,
        sec_fetcher: Callable[[str, float], Any] | None = None,
    ):
        self.database_path = database_path
        self.refresh_days = refresh_days
        self.assets_url = ALPACA_ASSETS_URL_PAPER if paper else ALPACA_ASSETS_URL_LIVE
        self._alpaca_fetcher = alpaca_fetcher or self._fetch_alpaca_asset
        self._sec_fetcher = sec_fetcher or self._fetch_sec_tickers

    @staticmethod
    def _fetch_alpaca_asset(url: str, headers: dict) -> Any:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _fetch_sec_tickers(url: str, timeout_seconds: float) -> Any:
        request = Request(
            url,
            headers={
                # SEC requires a descriptive User-Agent identifying the
                # requester; a generic or missing one can be blocked.
                "User-Agent": "pi-trading-agent/1.0 (personal research use)"
            },
        )
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: fixed public HTTPS URL
            return json.loads(response.read().decode("utf-8"))

    def refresh(self, symbols: list[str], api_key: str, secret_key: str) -> bool:
        """Refresh the local mapping for `symbols` if the interval has elapsed.

        Returns True when a refresh actually ran. Fails open: any source
        error for one symbol simply leaves that symbol unresolved, and a
        totally unreachable source leaves the previously cached mapping
        untouched.
        """
        symbols = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
        if not symbols:
            return False
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # The interval applies to already-known symbols, not to the whole
        # database. Autonomous discovery changes the candidate list every day;
        # newly seen symbols must be enriched immediately even when the last
        # full refresh happened recently.
        with self._connect() as conn:
            self._create_schema(conn)
            last_refreshed = conn.execute(
                "SELECT value FROM refresh_state WHERE name = 'last_refreshed'"
            ).fetchone()
            full_refresh_due = True
            if last_refreshed:
                try:
                    since = date.today() - date.fromisoformat(last_refreshed[0])
                    full_refresh_due = since >= timedelta(days=self.refresh_days)
                except ValueError:
                    full_refresh_due = True
            placeholders = ", ".join("?" for _ in symbols)
            known = {
                row[0]
                for row in conn.execute(
                    f"SELECT ticker FROM symbols WHERE ticker IN ({placeholders})", symbols
                ).fetchall()
            }

        refresh_symbols = symbols if full_refresh_due else [symbol for symbol in symbols if symbol not in known]
        if not refresh_symbols:
            return False

        # Do all network work without holding a DuckDB connection. The strategy
        # runs this method in a background thread; readers can continue using
        # the last complete snapshot while these bounded requests are pending.
        sec_source_succeeded = False
        try:
            sec_by_ticker = self._load_sec_mapping()
            sec_source_succeeded = True
        except Exception:
            sec_by_ticker = {}

        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        today = date.today().isoformat()
        records: list[tuple[str, Any, Any, bool, str]] = []
        alpaca_source_succeeded = False
        for symbol in refresh_symbols:
            alpaca_name = None
            try:
                asset = self._alpaca_fetcher(f"{self.assets_url}/{symbol}", headers)
                alpaca_source_succeeded = True
                if isinstance(asset, dict):
                    alpaca_name = asset.get("name")
            except Exception:
                alpaca_name = None
            sec_name = sec_by_ticker.get(symbol)
            if alpaca_name is None and sec_name is None:
                # Neither source recognizes this ticker; do not record a bare
                # entry that would let downstream code trust it.
                continue
            verified = (
                alpaca_name is not None
                and sec_name is not None
                and self._names_match(alpaca_name, sec_name)
            )
            records.append((symbol, alpaca_name, sec_name, verified, today))

        source_succeeded = sec_source_succeeded or alpaca_source_succeeded
        if not source_succeeded:
            return False

        with self._connect() as conn:
            self._create_schema(conn)
            for record in records:
                conn.execute(
                    """
                    INSERT INTO symbols (ticker, alpaca_name, sec_name, verified, refreshed_date)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (ticker) DO UPDATE SET
                        alpaca_name = excluded.alpaca_name,
                        sec_name = excluded.sec_name,
                        verified = excluded.verified,
                        refreshed_date = excluded.refreshed_date
                    """,
                    record,
                )
            if full_refresh_due:
                conn.execute(
                    """
                    INSERT INTO refresh_state VALUES ('last_refreshed', ?)
                    ON CONFLICT (name) DO UPDATE SET value = excluded.value
                    """,
                    (today,),
                )
            conn.commit()
        return True

    def _load_sec_mapping(self) -> dict[str, str]:
        raw = self._sec_fetcher(SEC_TICKERS_URL, 15.0)
        mapping: dict[str, str] = {}
        if isinstance(raw, dict):
            values: Any = raw.values()
        elif isinstance(raw, list):
            values = raw
        else:
            values = []
        for entry in values:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).strip().upper()
            title = str(entry.get("title", "")).strip()
            if ticker and title:
                mapping[ticker] = title
        return mapping

    def verified_symbols(self) -> set[str]:
        """Tickers recognized by at least one of the two sources.

        This is a sanity filter against a spurious or malformed tag, not a
        confidence tier: both cross-verified (`verified` true) and
        single-source tickers are returned. A symbol never seen by either
        source -- and therefore never inserted -- is excluded. An empty
        result (nothing refreshed yet, or both sources unavailable) means
        callers should fail open and trust Alpaca's raw tags unfiltered
        rather than treat "no local record" as "reject everything".
        """
        if not self.database_path.is_file():
            return set()
        try:
            with self._connect() as conn:
                self._create_schema(conn)
                rows = conn.execute("SELECT ticker FROM symbols").fetchall()
            return {row[0] for row in rows}
        except Exception:
            return set()

    def aliases_for_symbols(self, candidates: list[str] | set[str]) -> dict[str, tuple[str, ...]]:
        """Load normalized company-name aliases for all candidates in one query."""
        normalized = sorted(
            {str(symbol).strip().upper() for symbol in candidates if str(symbol).strip()}
        )
        if not normalized:
            return {}
        try:
            with self._connect() as conn:
                self._create_schema(conn)
                placeholders = ", ".join("?" for _ in normalized)
                rows = conn.execute(
                    f"SELECT ticker, alpaca_name, sec_name FROM symbols "
                    f"WHERE ticker IN ({placeholders})",
                    normalized,
                ).fetchall()
        except Exception:
            return {}
        aliases: dict[str, tuple[str, ...]] = {}
        for ticker, alpaca_name, sec_name in rows:
            names = tuple(
                dict.fromkeys(
                    normalized_name
                    for name in (alpaca_name, sec_name)
                    if name and (normalized_name := self._normalize_name(name))
                )
            )
            if names:
                aliases[ticker] = names
        return aliases

    def scan_text_for_symbols(self, text: str, candidates: list[str] | set[str]) -> set[str]:
        """Find company-name mentions of `candidates` in `text`.

        Bounded to the given candidates (the day's watchlist), never the
        whole market, to keep this cheap. Catches a mention Alpaca's own
        `symbols` tagging missed.
        """
        if not text.strip() or not candidates:
            return set()
        return self.scan_text_for_aliases(text, self.aliases_for_symbols(candidates))

    def scan_text_for_aliases(
        self, text: str, aliases: dict[str, tuple[str, ...]]
    ) -> set[str]:
        """Find alias mentions using a caller-preloaded mapping."""
        normalized_text = self._normalize_name(text)
        if not normalized_text:
            return set()
        found: set[str] = set()
        for ticker, names in aliases.items():
            for normalized_name in names:
                if normalized_name and normalized_name in normalized_text:
                    found.add(ticker)
                    break
        return found

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database_path))

    @staticmethod
    def _create_schema(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                ticker TEXT PRIMARY KEY,
                alpaca_name TEXT,
                sec_name TEXT,
                verified BOOLEAN NOT NULL,
                refreshed_date TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refresh_state (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        name = name.lower()
        name = _SUFFIXES.sub(" ", name)
        name = _NON_WORD.sub(" ", name)
        return " ".join(name.split())

    @classmethod
    def _names_match(cls, name_a: str, name_b: str) -> bool:
        normalized_a = cls._normalize_name(name_a)
        normalized_b = cls._normalize_name(name_b)
        if not normalized_a or not normalized_b:
            return False
        if normalized_a == normalized_b:
            return True
        tokens_a = set(normalized_a.split())
        tokens_b = set(normalized_b.split())
        # A shared distinctive token (longer than 3 characters, to skip
        # generic words) is enough: "Apple" in "apple inc" vs "apple" is a
        # confident match without requiring an exact string.
        return bool({token for token in tokens_a & tokens_b if len(token) > 3})
