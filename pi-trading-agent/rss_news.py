"""Free, no-API-key RSS headline ingestion, merged into news_context.py's
article set. Deliberately free-only: no source here requires an API key or
paid tier (Yahoo Finance, MarketWatch, CNBC, etc. all publish plain RSS 2.0).

Produces article dicts in exactly the shape WorldEventAnalyzer.analyze()
already builds from Alpaca (`headline`, `summary`, `symbols`, `created_at`,
`url`), so everything downstream -- keyword scoring, the LLM assessment,
article_filter.py's full-article verdicts -- needs no RSS-specific handling.
An RSS article always has `symbols: []`; strategy.py's `_symbol_news_scores`
already extends coverage via a company-name text scan for exactly this case
(an article with no source-provided ticker tag), so RSS headlines are still
attributable to a watched symbol without any change there.
"""

import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

from safe_http import fetch_public_bytes

logger = logging.getLogger(__name__)

RSS_REQUEST_TIMEOUT_SECONDS = 15
RSS_FETCH_WORKERS = 4
RSS_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
# Some publishers reject requests with no User-Agent at all.
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; pi-trading-agent RSS reader)"}

_TAG_PATTERN = re.compile(r"<[^>]+>")


def _clean_text(raw: str | None) -> str:
    """Strip any literal HTML tags/entities a feed's <description> may carry."""
    if not raw:
        return ""
    return html.unescape(_TAG_PATTERN.sub("", raw)).strip()


def _parse_pub_date(raw: str | None) -> datetime | None:
    """RSS 2.0's <pubDate> is RFC 822; anything else is treated as unknown
    rather than raising -- a missing/odd date should not drop the article."""
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_feed(xml_text: str | bytes) -> list[dict[str, Any]]:
    """Pure parse of one RSS 2.0 document into article dicts; never raises
    (an unparseable feed yields no articles, exactly like an empty one)."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    articles: list[dict[str, Any]] = []
    for item in root.iter("item"):
        title = _clean_text(item.findtext("title"))
        if not title:
            continue
        description = _clean_text(item.findtext("description"))
        link = str(item.findtext("link") or "").strip()
        articles.append(
            {
                "headline": title,
                "summary": description,
                "symbols": [],
                "created_at": _parse_pub_date(item.findtext("pubDate")),
                "url": link,
            }
        )
    return articles


def fetch_articles(
    feed_urls: list[str],
    lookback_hours: int,
    max_articles: int,
    timeout: int = RSS_REQUEST_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch and merge every configured feed, newest first, capped at
    `max_articles` total. Each feed fails independently -- one broken or
    slow publisher never drops the others -- and the whole call never
    raises, matching every other news source in this pipeline.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0, lookback_hours))
    def fetch_one(feed_url: str) -> list[dict[str, Any]]:
        try:
            xml_text = fetch_public_bytes(
                feed_url,
                timeout=timeout,
                max_bytes=RSS_MAX_DOWNLOAD_BYTES,
                headers=_REQUEST_HEADERS,
            )
        except Exception as exc:
            logger.warning("RSS feed unavailable, skipping: %s (%s: %s)", feed_url, type(exc).__name__, exc)
            return []
        return _parse_feed(xml_text)

    if not feed_urls:
        return []
    with ThreadPoolExecutor(
        max_workers=min(RSS_FETCH_WORKERS, len(feed_urls)),
        thread_name_prefix="rss-feed",
    ) as executor:
        feed_articles = executor.map(fetch_one, feed_urls)

    seen_urls: set[str] = set()
    collected: list[dict[str, Any]] = []
    # executor.map preserves configured feed order, keeping duplicate handling
    # deterministic even though the network requests run concurrently.
    for articles in feed_articles:
        for article in articles:
            url = article["url"]
            if url and url in seen_urls:
                continue
            created_at = article["created_at"]
            if created_at is not None and created_at < cutoff:
                continue
            if url:
                seen_urls.add(url)
            collected.append(article)

    collected.sort(
        key=lambda article: article["created_at"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return collected[:max_articles]
