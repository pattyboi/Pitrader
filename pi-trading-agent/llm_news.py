"""Optional Claude-based assessment of daily market news.

This layer reads the same Alpaca headlines the keyword scorer uses and asks
the Claude API for one structured risk assessment per trading day. It is
advisory by default, fails open like the rest of the news stack, and never
creates a buy signal on its own.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class LLMNewsAssessment:
    """A structured, explainable market-risk assessment from the model."""

    available: bool
    score: int = 0
    risk_level: str = "unknown"
    reasoning: str = ""
    explanation: str = "LLM news assessment was not evaluated."


# Structured-outputs schema. Numerical range limits are not supported by the
# API schema validator, so the score is clamped in code after parsing.
ASSESSMENT_SCHEMA = {
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
        "risk_level": {
            "type": "string",
            "enum": ["high", "elevated", "normal", "constructive"],
        },
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

MAX_SUMMARY_CHARS = 400
MIN_SCORE = -10
MAX_SCORE = 10


class LLMNewsAnalyzer:
    """Score the day's headlines with one Claude API call."""

    def __init__(self, model: str):
        self.model = model

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

    def assess(self, articles: list[dict]) -> LLMNewsAssessment:
        """Return one structured assessment; raise on any API problem."""
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not available in the environment; "
                "the LLM news assessment requires it."
            )

        article_text = self._format_articles(articles)
        if not article_text:
            return LLMNewsAssessment(
                available=False,
                explanation="No usable headlines were available to assess.",
            )

        # Fail fast on network problems; the price strategy must not wait
        # long on a news call. The SDK retries transient errors itself.
        client = anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=2)
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,  # deliberately small: the response is one short JSON object
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": ASSESSMENT_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Assess today's market news. Articles from the last "
                        "24 hours:\n\n" + article_text
                    ),
                }
            ],
        )

        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to assess this content")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("The assessment response was truncated")

        text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        data = json.loads(text)
        score = max(MIN_SCORE, min(MAX_SCORE, int(data["score"])))
        risk_level = str(data["risk_level"])
        reasoning = str(data["reasoning"]).strip()
        return LLMNewsAssessment(
            available=True,
            score=score,
            risk_level=risk_level,
            reasoning=reasoning,
            explanation=(
                f"Claude ({self.model}) assessed {len(articles)} articles; "
                f"score {score:+d} ({risk_level})."
            ),
        )
