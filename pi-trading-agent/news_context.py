"""Lightweight headline analysis for daily market context."""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

SEVERE_RISK_TERMS = (
    "bank failure",
    "cyberattack",
    "debt default",
    "sovereign default",
    "invasion",
    "market crash",
    "missile attack",
    "nuclear threat",
    "state of emergency",
    "terrorist attack",
    "war declared",
)

RISK_TERMS = (
    "bankruptcy",
    "conflict",
    "credit downgrade",
    "economic contraction",
    "embargo",
    "inflation surge",
    "layoffs",
    "rate hike",
    "recession",
    "sanctions",
    "supply disruption",
    "tariff",
)

OPPORTUNITY_TERMS = (
    "ceasefire",
    "economic recovery",
    "earnings beat",
    "peace agreement",
    "rate cut",
    "rescue package",
    "stimulus",
    "trade agreement",
)


@dataclass
class NewsContext:
    """A deterministic summary of recently published headlines."""

    available: bool
    score: int = 0
    article_count: int = 0
    risk_level: str = "unknown"
    headlines: list[str] = field(default_factory=list)
    explanation: str = "News context was not evaluated."


class WorldEventAnalyzer:
    """Fetch Alpaca news and score explicit terms without using an LLM."""

    def __init__(self, lookback_hours: int, max_articles: int, block_score: int):
        self.lookback_hours = lookback_hours
        self.max_articles = max_articles
        self.block_score = block_score

    @staticmethod
    def _contains(text: str, phrase: str) -> bool:
        """Match complete words so terms such as 'war' do not match 'award'."""
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    @classmethod
    def score_text(cls, text: str) -> tuple[int, list[str]]:
        """Return a transparent score and the terms responsible for it."""
        score = 0
        matched: list[str] = []
        for phrase in SEVERE_RISK_TERMS:
            if cls._contains(text, phrase):
                score -= 3
                matched.append(phrase)
        for phrase in RISK_TERMS:
            if cls._contains(text, phrase):
                score -= 1
                matched.append(phrase)
        for phrase in OPPORTUNITY_TERMS:
            if cls._contains(text, phrase):
                score += 1
                matched.append(phrase)
        return score, matched

    def analyze(self) -> NewsContext:
        """Fetch recent general market news and build a compact risk context."""
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        # alpaca-py does not read credentials from the environment on its own;
        # main.py exports these before the strategy starts.
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_API_SECRET", "")
        if not api_key or not secret_key:
            raise RuntimeError(
                "Alpaca credentials were not found in the environment; "
                "news retrieval requires them."
            )

        now = datetime.now(timezone.utc)
        request = NewsRequest(
            start=now - timedelta(hours=self.lookback_hours),
            end=now,
            limit=self.max_articles,
        )
        response = NewsClient(api_key=api_key, secret_key=secret_key).get_news(request)
        dataframe = getattr(response, "df", None)
        if dataframe is None or dataframe.empty:
            return NewsContext(
                available=True,
                risk_level="normal",
                explanation="Alpaca returned no recent news articles.",
            )

        scored_headlines: list[tuple[int, str]] = []
        total_score = 0
        article_count = 0
        for _, row in dataframe.iterrows():
            headline = str(self._row_value(row, "headline") or "").strip()
            summary = str(self._row_value(row, "summary") or "").strip()
            if not headline:
                continue
            article_count += 1
            article_score, matched = self.score_text(f"{headline} {summary}")
            total_score += article_score
            if article_score != 0:
                reason = ", ".join(matched)
                scored_headlines.append(
                    (article_score, f"[{article_score:+d}] {headline} (matched: {reason})")
                )

        if total_score <= self.block_score:
            risk_level = "high"
        elif total_score < 0:
            risk_level = "elevated"
        elif total_score > 0:
            risk_level = "constructive"
        else:
            risk_level = "normal"

        scored_headlines.sort(key=lambda item: abs(item[0]), reverse=True)
        top_headlines = [text for _, text in scored_headlines[:5]]
        return NewsContext(
            available=True,
            score=total_score,
            article_count=article_count,
            risk_level=risk_level,
            headlines=top_headlines,
            explanation=(
                f"Scored {article_count} recent articles using explicit keyword rules; "
                f"aggregate score {total_score}."
            ),
        )

    @staticmethod
    def _row_value(row: Any, key: str) -> Any:
        """Read a pandas row without assuming every news field is present."""
        try:
            return row.get(key)
        except (AttributeError, KeyError, TypeError):
            return None
