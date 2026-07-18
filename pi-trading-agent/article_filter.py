"""Pre-filter a financial news article and query the local Ollama model for
structured sentiment/risk context.

Standalone from the rest of the news stack (`news_context.py`, `llm_news.py`):
this fetches and extracts a full article body by URL, keeps only the
sentences most likely to matter for the watchlist, and asks a local model
(see MODEL) to structure that excerpt into a sentiment/risk verdict. Never raises;
any failure (fetch, extraction, model, parsing) is logged and yields None so
a caller can simply skip the article.
"""

import json
import logging
import re
from hashlib import sha256
from datetime import date
from pathlib import Path

import requests
import trafilatura

try:
    import tiktoken

    # cl100k_base is not this pipeline's actual tokenizer (MODEL's GGUF vocab
    # differs, and Ollama has no tokenize endpoint to query it -- see
    # llm_news.py's _TOKEN_ENCODER for the same reasoning), but a real BPE
    # encoder still segments dense financial text (tickers, "%", table-like
    # runs of digits) the way a subword tokenizer actually would, unlike a
    # linear words/chars heuristic. Fast enough (~0.3ms/article) to cost
    # nothing extra when available.
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent / ".article_cache.json"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL = "hf.co/unsloth/granite-4.0-micro-GGUF:Q4_K_M"
# Ollama defaults an API request to a ~4096-token context regardless of what
# the model natively supports unless a request explicitly asks for more
# (see llm_news.py's OLLAMA_NUM_CTX, which found this directly in this
# model's own server log); granite-4.0-micro trains at 131072, so this
# reclaims real headroom, though MAX_TOKENS below stays small on its own
# merits (an over-long excerpt isn't worth the model's attention either).
OLLAMA_NUM_CTX = 8192
# The JSON verdict has five short, fixed keys; this is generous headroom,
# not a target -- a real reply is typically well under this.
OLLAMA_NUM_PREDICT = 250
# MAX_TOKENS (800) worth of input plus OLLAMA_NUM_PREDICT worth of output is
# comfortably inside this even at this Pi's measured worst case (~1.6x
# token-estimate undercount, ~28 tokens/sec prompt-eval, ~4 tokens/sec
# generation -- see llm_news.py's REQUEST_TIMEOUT_SECONDS for the same
# measurement); generous margin is kept since this fetches an arbitrary
# third-party page, whose extraction time is not bounded by any of that.
REQUEST_TIMEOUT_SECONDS = 180

MIN_SENTENCE_WORDS = 8
TOP_SENTENCES = 15
TICKER_SCORE = 3
SIGNAL_WORD_SCORE = 1
# ~4 chars/token undercounted real tokenizer output for dense financial text
# (see llm_news.py's MAX_PROMPT_TOKENS_ESTIMATE); a words*1.3 estimate is used
# here instead, still an approximation, not an exact count.
TOKENS_PER_WORD = 1.3
MIN_TOKENS = 80
MAX_TOKENS = 800

SIGNAL_WORDS = {
    "earnings", "revenue", "guidance", "beat", "miss", "outlook", "forecast",
    "upgrade", "downgrade", "target", "initiated", "rating", "rate", "fed",
    "inflation", "gdp", "recession", "yield", "default", "bankruptcy",
    "lawsuit", "recall", "investigation", "tariff", "merger", "acquisition",
    "buyback", "dividend", "split", "ipo", "surge", "plunge", "rally",
    "selloff", "volatile", "risk",
}

_WORD_PATTERN = re.compile(r"[A-Za-z']+")


def _estimate_tokens(text: str) -> int:
    """tiktoken's real BPE count when available (see _TOKEN_ENCODER); a
    words*TOKENS_PER_WORD estimate otherwise. Never raises: arbitrary
    third-party article text can contain sequences tiktoken treats as
    special tokens."""
    if not text:
        return 0
    if _TOKEN_ENCODER is not None:
        try:
            return len(_TOKEN_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    return int(len(text.split()) * TOKENS_PER_WORD)


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in text.split(". ") if sentence.strip()]


def _score_sentence(sentence: str, tickers: set[str]) -> int:
    words = _WORD_PATTERN.findall(sentence)
    score = sum(TICKER_SCORE for word in words if word.upper() in tickers)
    score += sum(SIGNAL_WORD_SCORE for word in words if word.lower() in SIGNAL_WORDS)
    return score


def _select_top_sentences(sentences: list[str], tickers: set[str]) -> list[str]:
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: -_score_sentence(pair[1], tickers),
    )
    top_indices = sorted(index for index, _ in ranked[:TOP_SENTENCES])
    return [sentences[index] for index in top_indices]


def _truncate_to_token_limit(sentences: list[str], max_tokens: int) -> str:
    kept: list[str] = []
    for sentence in sentences:
        candidate = ". ".join(kept + [sentence])
        if kept and _estimate_tokens(candidate) > max_tokens:
            break
        kept.append(sentence)
    return ". ".join(kept)


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _cache_key(url: str, watchlist: list[str]) -> str:
    """Key model output by every input that can change its verdict."""
    symbols = sorted({str(symbol).strip().upper() for symbol in watchlist if str(symbol).strip()})
    digest = sha256("\0".join([MODEL, *symbols]).encode("utf-8")).hexdigest()[:16]
    return f"{date.today().isoformat()}:{digest}:{url}"


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def _build_prompt(filtered_text: str, watchlist: list[str]) -> str:
    return (
        "Analyze this financial news excerpt. Respond only in JSON, no other text.\n\n"
        f"Watchlist: {', '.join(watchlist)}\n\n"
        f"Article:\n{filtered_text}\n\n"
        '{"sentiment": "bullish|bearish|neutral", "confidence": 0.0-1.0, '
        '"affected_tickers": [], "key_risks": [], '
        '"catalyst_type": "earnings|macro|analyst|corporate|other"}'
    )


def _query_model(filtered_text: str, watchlist: list[str]) -> dict:
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": MODEL,
            "prompt": _build_prompt(filtered_text, watchlist),
            "stream": False,
            "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return json.loads(response.json()["response"])


def extract_financial_context(url: str, watchlist: list[str]) -> dict | None:
    """Fetch `url`, keep only its highest-signal sentences for `watchlist`,
    and return the model's structured sentiment/risk verdict, or None if the
    article is skipped (fetch/extraction failure, too low-signal, or any
    other problem). Cached per (url, calendar day)."""
    try:
        cache_key = _cache_key(url, watchlist)
        cache = _load_cache()
        if cache_key in cache:
            return cache[cache_key]

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded, include_comments=False, include_tables=False, fast=True
        )
        if not text or not text.strip():
            return None

        candidates = [
            sentence
            for sentence in _split_sentences(text)
            if len(sentence.split()) >= MIN_SENTENCE_WORDS
        ]
        if not candidates:
            return None

        tickers = {str(symbol).strip().upper() for symbol in watchlist if str(symbol).strip()}
        selected = _select_top_sentences(candidates, tickers)
        filtered_text = ". ".join(selected)
        if _estimate_tokens(filtered_text) < MIN_TOKENS:
            return None
        filtered_text = _truncate_to_token_limit(selected, MAX_TOKENS)

        result = _query_model(filtered_text, watchlist)

        cache[cache_key] = result
        _save_cache(cache)
        return result
    except Exception:
        logger.exception("extract_financial_context failed for %s", url)
        return None
