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
    # Per-symbol subset of `score`, built only from articles that mention
    # that symbol. A symbol present here with value 0 was covered by news
    # today with no matched risk/opportunity phrase (genuinely neutral); a
    # symbol absent entirely had no dedicated coverage at all, and callers
    # should fall back to the market-wide `score` for it.
    per_symbol_scores: dict[str, int] = field(default_factory=dict)
    article_count: int = 0
    risk_level: str = "unknown"
    headlines: list[str] = field(default_factory=list)
    # Raw articles for optional downstream analysis (e.g. the LLM layer).
    articles: list[dict] = field(default_factory=list)
    # Per-article breakdown (headline, summary, Alpaca's own symbol tags,
    # and that article's individual score) so a caller can re-attribute a
    # symbol mentioned by company name but missed by Alpaca's tagging --
    # see symbol_reference.py's scan_text_for_symbols.
    per_article: list[dict] = field(default_factory=list)
    explanation: str = "News context was not evaluated."


@dataclass
class ArticleScoring:
    """Pure scoring result for an already-fetched batch of articles."""

    total_score: int
    per_symbol_scores: dict[str, int]
    scored_headlines: list[str]
    article_count: int
    per_article: list[dict]


class WorldEventAnalyzer:
    """Fetch Alpaca news and score explicit terms without using an LLM."""

    # Duplicate-event dampening (score refinement only): repeated occurrences
    # of the *same* matched phrase across today's batch count for less, so
    # several wire-service copies of one story cannot inflate the score as
    # much as genuinely distinct events. 1st occurrence full weight, then
    # tapering; anything beyond the 3rd occurrence keeps the floor weight.
    _DUPLICATE_PHRASE_WEIGHTS = (1.0, 0.6, 0.3)
    # Recency decay floor (score refinement only): an article at the edge of
    # the lookback window still counts, just at reduced weight, rather than
    # being ignored outright.
    _RECENCY_WEIGHT_FLOOR = 0.4

    def __init__(
        self,
        lookback_hours: int,
        max_articles: int,
        block_score: int,
        refine_scoring: bool = False,
    ):
        self.lookback_hours = lookback_hours
        self.max_articles = max_articles
        self.block_score = block_score
        # Off by default: this changes the exact value of `score`, which
        # feeds NEWS_HIGH_RISK_SCORE and the adaptive model's training
        # target, so it is an explicit opt-in rather than a silent change to
        # an existing trade-blocking guard's behavior.
        self.refine_scoring = refine_scoring

    @staticmethod
    def _contains(text: str, phrase: str) -> bool:
        """Match complete words so terms such as 'war' do not match 'award'."""
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    @classmethod
    def _matched_terms(cls, text: str) -> list[tuple[str, int]]:
        """Return (phrase, delta) for every configured phrase found in text."""
        matches: list[tuple[str, int]] = []
        for phrase in SEVERE_RISK_TERMS:
            if cls._contains(text, phrase):
                matches.append((phrase, -3))
        for phrase in RISK_TERMS:
            if cls._contains(text, phrase):
                matches.append((phrase, -1))
        for phrase in OPPORTUNITY_TERMS:
            if cls._contains(text, phrase):
                matches.append((phrase, 1))
        return matches

    @classmethod
    def score_text(cls, text: str) -> tuple[int, list[str]]:
        """Return a transparent score and the terms responsible for it."""
        matches = cls._matched_terms(text)
        return sum(delta for _, delta in matches), [phrase for phrase, _ in matches]

    @classmethod
    def score_articles(
        cls,
        articles: list[dict[str, Any]],
        lookback_hours: int,
        refine: bool = False,
        now: datetime | None = None,
    ) -> ArticleScoring:
        """Score already-fetched articles; pure and independent of any network call.

        Each article dict needs "headline" and "summary"; "symbols"
        (list[str]) and "created_at" (datetime) are optional and only used
        when `refine` is true. When `refine` is false this reproduces the
        original unweighted scoring exactly, so NEWS_SCORE_REFINEMENT_ENABLED
        defaults to off without changing today's guard behavior.
        """
        now = now or datetime.now(timezone.utc)
        phrase_occurrences: dict[str, int] = {}
        total_score = 0.0
        per_symbol_scores: dict[str, float] = {}
        scored_headlines: list[tuple[float, str]] = []
        per_article: list[dict[str, Any]] = []
        article_count = 0
        for article in articles:
            headline = str(article.get("headline") or "").strip()
            summary = str(article.get("summary") or "").strip()
            if not headline:
                continue
            article_count += 1
            matches = cls._matched_terms(f"{headline} {summary}")

            recency_weight = 1.0
            if refine and matches:
                created_at = article.get("created_at")
                if isinstance(created_at, datetime) and lookback_hours > 0:
                    age_hours = (now - created_at).total_seconds() / 3600.0
                    age_hours = max(0.0, min(float(lookback_hours), age_hours))
                    recency_weight = 1.0 - (1.0 - cls._RECENCY_WEIGHT_FLOOR) * (
                        age_hours / lookback_hours
                    )

            article_score = 0.0
            matched_phrases: list[str] = []
            for phrase, delta in matches:
                duplicate_weight = 1.0
                if refine:
                    occurrence = phrase_occurrences.get(phrase, 0)
                    duplicate_weight = cls._DUPLICATE_PHRASE_WEIGHTS[
                        min(occurrence, len(cls._DUPLICATE_PHRASE_WEIGHTS) - 1)
                    ]
                    phrase_occurrences[phrase] = occurrence + 1
                article_score += delta * recency_weight * duplicate_weight
                matched_phrases.append(phrase)
            total_score += article_score

            # Recorded for every mentioned symbol, even a zero-scoring
            # article: that distinguishes "covered today, genuinely neutral"
            # (present with value 0) from "no dedicated coverage at all"
            # (absent), so callers know when to trust a symbol-specific 0
            # instead of falling back to the market-wide score.
            tagged_symbols = {str(s).strip().upper() for s in (article.get("symbols") or []) if str(s).strip()}
            for symbol in tagged_symbols:
                per_symbol_scores[symbol] = per_symbol_scores.get(symbol, 0.0) + article_score
            per_article.append(
                {
                    "headline": headline,
                    "summary": summary,
                    "symbols": sorted(tagged_symbols),
                    "score": round(article_score),
                }
            )

            rounded_article_score = round(article_score)
            if rounded_article_score != 0:
                reason = ", ".join(matched_phrases)
                scored_headlines.append(
                    (article_score, f"[{rounded_article_score:+d}] {headline} (matched: {reason})")
                )

        scored_headlines.sort(key=lambda item: abs(item[0]), reverse=True)
        return ArticleScoring(
            total_score=round(total_score),
            per_symbol_scores={symbol: round(value) for symbol, value in per_symbol_scores.items()},
            scored_headlines=[text for _, text in scored_headlines[:5]],
            article_count=article_count,
            per_article=per_article,
        )

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

        fetched_articles: list[dict[str, Any]] = []
        for _, row in dataframe.iterrows():
            headline = str(self._row_value(row, "headline") or "").strip()
            summary = str(self._row_value(row, "summary") or "").strip()
            if not headline:
                continue
            symbols_value = self._row_value(row, "symbols")
            created_at = self._row_value(row, "created_at")
            fetched_articles.append(
                {
                    "headline": headline,
                    "summary": summary,
                    "symbols": list(symbols_value) if isinstance(symbols_value, (list, tuple)) else [],
                    "created_at": created_at if isinstance(created_at, datetime) else None,
                }
            )

        scoring = self.score_articles(
            fetched_articles, self.lookback_hours, refine=self.refine_scoring, now=now
        )

        if scoring.total_score <= self.block_score:
            risk_level = "high"
        elif scoring.total_score < 0:
            risk_level = "elevated"
        elif scoring.total_score > 0:
            risk_level = "constructive"
        else:
            risk_level = "normal"

        return NewsContext(
            available=True,
            score=scoring.total_score,
            per_symbol_scores=scoring.per_symbol_scores,
            article_count=scoring.article_count,
            risk_level=risk_level,
            headlines=scoring.scored_headlines,
            articles=[
                {"headline": article["headline"], "summary": article["summary"]}
                for article in fetched_articles
            ],
            per_article=scoring.per_article,
            explanation=(
                f"Scored {scoring.article_count} recent articles using explicit keyword rules; "
                f"aggregate score {scoring.total_score}."
            ),
        )

    @staticmethod
    def _row_value(row: Any, key: str) -> Any:
        """Read a pandas row without assuming every news field is present."""
        try:
            return row.get(key)
        except (AttributeError, KeyError, TypeError):
            return None
