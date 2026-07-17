"""Optional LLM-based assessment of daily market news.

This layer reads the same Alpaca headlines the keyword scorer uses and asks
a language model for one structured risk assessment per trading day. It is
advisory by default, fails open like the rest of the news stack, and never
creates a buy signal on its own.

The only supported backend is a local Ollama server (its OpenAI-compatible
`/v1/chat/completions` endpoint) so this assessment never depends on an
outside service or leaves the box. `ollama.service` binds to loopback only;
`ollama-warmup.timer` loads the model ahead of market open so this call
isn't paying a cold-start cost during the trading iteration.
"""

import json
import re
from dataclasses import dataclass

OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

MAX_SUMMARY_CHARS = 400
# Defensive cap on how many symbols get listed in the prompt; the day's
# evaluation universe (watchlist + held + discovery candidates) is already
# bounded well under this in practice.
MAX_SYMBOLS_LISTED = 60
MIN_SCORE = -10
MAX_SCORE = 10
RISK_LEVELS = ("high", "elevated", "normal", "constructive")
# This assessment is optional and fails open. A local, CPU-bound small model
# is slower than a hosted API, so the budget is generous -- but still well
# below the daily trading iteration's own patience, and the warm-up timer
# means this is almost never paying for a cold model load.
REQUEST_TIMEOUT_SECONDS = 90
MAX_RESPONSE_TOKENS = 4096


@dataclass
class LLMNewsAssessment:
    """A structured, explainable market-risk assessment from the model."""

    available: bool
    score: int = 0
    risk_level: str = "unknown"
    reasoning: str = ""
    explanation: str = "LLM news assessment was not evaluated."


SYSTEM_PROMPT = (
    "You are a cautious market-risk analyst supporting a small automated "
    "trading agent. The agent rotates between broad US equity ETFs at most "
    "once per trading day; your assessment may only veto a trade, never "
    "create one.\n\n"
    "Given today's financial news headlines and summaries, assess aggregate "
    "downside risk for broad US equity markets over the next one to five "
    "trading sessions. Base the assessment only on the provided articles - "
    "do not assume events that are not mentioned. Multiple articles about "
    "the same event count as one event, not several.\n\n"
    "Score conservatively: reserve scores of -6 or below for genuinely "
    "severe, market-wide risk (major war escalation, systemic financial "
    "failure, market crash in progress). Ordinary negative news such as a "
    "single weak earnings report, routine rate speculation, or sector-level "
    "problems should stay above -4. When the news is mixed or unremarkable, "
    "score near 0.\n\n"
    "You may also be given the specific stocks/ETFs the agent is actually "
    "evaluating today (marking which are currently held) plus any "
    "symbol-specific news coverage separate from the general headlines. "
    "Your score still measures aggregate market risk, not any one symbol - "
    "but weigh concentrated bad news across several of today's symbols, or "
    "bad news specifically hitting a held position, more heavily than the "
    "same story would count in isolation, and mention the affected symbols "
    "by name in your reasoning when they drove the score."
)

JSON_FORMAT_INSTRUCTIONS = (
    "\n\nRespond with only a JSON object and no other text, using exactly "
    "these keys:\n"
    '{"score": <integer from -10 to 10>, '
    '"risk_level": "high" | "elevated" | "normal" | "constructive", '
    '"reasoning": "<two or three plain sentences citing the specific '
    'headlines that drove the score>"}'
)


class LLMNewsAnalyzer:
    """Score the day's headlines with one call to a local Ollama model."""

    def __init__(self, model: str, base_url: str = ""):
        self.model = str(model).strip()
        self.base_url = str(base_url).strip() or OLLAMA_DEFAULT_BASE_URL

    @staticmethod
    def _format_articles(articles: list[dict]) -> str:
        lines = []
        for index, article in enumerate(articles, start=1):
            headline = str(article.get("headline", "")).strip()
            summary = str(article.get("summary", "")).strip()[:MAX_SUMMARY_CHARS]
            if not headline:
                continue
            if summary:
                lines.append(f"{index}. {headline} - {summary}")
            else:
                lines.append(f"{index}. {headline}")
        return "\n".join(lines)

    @staticmethod
    def _format_symbol_context(
        symbols: list[str], held_symbols: set[str], symbol_scores: dict[str, int]
    ) -> str:
        """Describe today's evaluation universe so the model can reason about
        risk to the symbols this agent might actually trade, not just the
        market in the abstract."""
        unique_symbols = sorted(
            {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        )
        if not unique_symbols:
            return ""
        held_upper = {str(symbol).strip().upper() for symbol in held_symbols}
        listed = ", ".join(
            f"{symbol}*" if symbol in held_upper else symbol
            for symbol in unique_symbols[:MAX_SYMBOLS_LISTED]
        )
        lines = [f"Symbols under evaluation today (* = currently held): {listed}"]
        # Only nonzero scores are worth spending prompt tokens on; a zero
        # entry means "covered today, genuinely neutral" (see
        # NewsContext.per_symbol_scores), which isn't actionable here.
        covered = {
            symbol: score
            for symbol, score in symbol_scores.items()
            if symbol in unique_symbols and score != 0
        }
        if covered:
            coverage = ", ".join(f"{symbol}: {score:+d}" for symbol, score in sorted(covered.items()))
            lines.append(f"Symbol-specific news coverage: {coverage}")
        return "\n".join(lines)

    @staticmethod
    def _parse_assessment(text: str, article_count: int, model: str) -> LLMNewsAssessment:
        """Validate and repair a model reply into a usable assessment."""
        cleaned = text.strip()
        # Some models wrap JSON in a markdown code fence despite instructions.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        score = max(MIN_SCORE, min(MAX_SCORE, int(data["score"])))
        risk_level = str(data.get("risk_level", "")).strip().lower()
        if risk_level not in RISK_LEVELS:
            if score <= -6:
                risk_level = "high"
            elif score < 0:
                risk_level = "elevated"
            elif score > 0:
                risk_level = "constructive"
            else:
                risk_level = "normal"
        reasoning = str(data.get("reasoning", "")).strip()
        return LLMNewsAssessment(
            available=True,
            score=score,
            risk_level=risk_level,
            reasoning=reasoning,
            explanation=(
                f"{model} assessed {article_count} articles; "
                f"score {score:+d} ({risk_level})."
            ),
        )

    def assess(
        self,
        articles: list[dict],
        symbols: list[str] | None = None,
        held_symbols: set[str] | None = None,
        symbol_scores: dict[str, int] | None = None,
    ) -> LLMNewsAssessment:
        """Return one structured assessment; raise on any problem.

        `symbols`/`held_symbols`/`symbol_scores` are optional context about
        today's actual evaluation universe; omitting them reproduces the
        original market-headlines-only prompt exactly.
        """
        import requests

        article_text = self._format_articles(articles)
        if not article_text:
            return LLMNewsAssessment(
                available=False,
                explanation="No usable headlines were available to assess.",
            )
        user_text = (
            "Assess today's market news. Articles from the last 24 hours:\n\n"
            + article_text
        )
        symbol_context = self._format_symbol_context(
            symbols or [], held_symbols or set(), symbol_scores or {}
        )
        if symbol_context:
            user_text += "\n\n" + symbol_context
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model,
                "max_tokens": MAX_RESPONSE_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT + JSON_FORMAT_INSTRUCTIONS,
                    },
                    {"role": "user", "content": user_text},
                ],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        if not text:
            raise RuntimeError("The model returned an empty assessment")
        return self._parse_assessment(text, len(articles), self.model)
