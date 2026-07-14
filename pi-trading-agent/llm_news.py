"""Optional LLM-based assessment of daily market news.

This layer reads the same Alpaca headlines the keyword scorer uses and asks
a language model for one structured risk assessment per trading day. It is
advisory by default, fails open like the rest of the news stack, and never
creates a buy signal on its own.

Supported providers:

- "gemini" (default): Google's Gemini API via its OpenAI-compatible
  endpoint. Has a genuinely free, rate-limited tier.
- "openai_compatible": any other OpenAI-compatible endpoint (Groq,
  OpenRouter, a local server, ...) via LLM_NEWS_BASE_URL.
- "anthropic": the Claude API via the official anthropic SDK.
"""

import json
import os
import re
from dataclasses import dataclass

PROVIDERS = ("gemini", "openai_compatible", "anthropic")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

MAX_SUMMARY_CHARS = 400
MIN_SCORE = -10
MAX_SCORE = 10
RISK_LEVELS = ("high", "elevated", "normal", "constructive")
REQUEST_TIMEOUT_SECONDS = 60
# The reply is one short JSON object, but reasoning models (e.g. Gemini 2.5)
# spend hidden "thinking" tokens from this same cap; too small a cap yields an
# empty reply, which this layer treats as a failed (skipped) assessment.
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
    "score near 0."
)

# Used for providers without strict schema enforcement; the reply is still
# validated, clamped, and repaired in _parse_assessment below.
JSON_FORMAT_INSTRUCTIONS = (
    "\n\nRespond with only a JSON object and no other text, using exactly "
    "these keys:\n"
    '{"score": <integer from -10 to 10>, '
    '"risk_level": "high" | "elevated" | "normal" | "constructive", '
    '"reasoning": "<two or three plain sentences citing the specific '
    'headlines that drove the score>"}'
)

# Structured-outputs schema for the Anthropic provider. Numerical range
# limits are not supported by the API schema validator, so the score is
# clamped in code after parsing.
ANTHROPIC_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": (
                "Aggregate near-term risk score for broad US equity markets, "
                "from -10 (severe, market-wide danger) through 0 (neutral) "
                "to +10 (strongly constructive)."
            ),
        },
        "risk_level": {"type": "string", "enum": list(RISK_LEVELS)},
        "reasoning": {
            "type": "string",
            "description": (
                "Two or three plain sentences citing the specific headlines "
                "that drove the score."
            ),
        },
    },
    "required": ["score", "risk_level", "reasoning"],
    "additionalProperties": False,
}


class LLMNewsAnalyzer:
    """Score the day's headlines with one LLM API call."""

    def __init__(self, provider: str, model: str, base_url: str = ""):
        provider = str(provider).strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {provider}")
        self.provider = provider
        self.model = str(model).strip()
        self.base_url = str(base_url).strip()

    @staticmethod
    def _api_key() -> str:
        api_key = os.environ.get("LLM_NEWS_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "LLM_NEWS_API_KEY is not available in the environment; "
                "the LLM news assessment requires it."
            )
        return api_key

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

    def assess(self, articles: list[dict]) -> LLMNewsAssessment:
        """Return one structured assessment; raise on any API problem."""
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
        if self.provider == "anthropic":
            return self._assess_with_anthropic(user_text, len(articles))
        return self._assess_with_openai_compatible(user_text, len(articles))

    def _assess_with_openai_compatible(
        self, user_text: str, article_count: int
    ) -> LLMNewsAssessment:
        """Call Gemini or any other OpenAI-compatible chat endpoint."""
        import requests

        base_url = self.base_url or GEMINI_BASE_URL
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
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
        return self._parse_assessment(text, article_count, self.model)

    def _assess_with_anthropic(
        self, user_text: str, article_count: int
    ) -> LLMNewsAssessment:
        """Call the Claude API with a strict structured-outputs schema."""
        import anthropic

        # Fail fast on network problems; the price strategy must not wait
        # long on a news call. The SDK retries transient errors itself.
        client = anthropic.Anthropic(
            api_key=self._api_key(),
            timeout=float(REQUEST_TIMEOUT_SECONDS),
            max_retries=2,
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=MAX_RESPONSE_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={
                "format": {"type": "json_schema", "schema": ANTHROPIC_ASSESSMENT_SCHEMA}
            },
            messages=[{"role": "user", "content": user_text}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to assess this content")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("The assessment response was truncated")
        text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        return self._parse_assessment(text, article_count, self.model)
