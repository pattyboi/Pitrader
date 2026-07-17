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
# The narrative/exit/red-flag replies are meant to be short; a smaller cap
# keeps them fast and terse without needing the assessment's full budget.
NARRATIVE_MAX_TOKENS = 300
MAX_NARRATIVE_CHARS = 600


@dataclass
class LLMNewsAssessment:
    """A structured, explainable market-risk assessment from the model."""

    available: bool
    score: int = 0
    risk_level: str = "unknown"
    reasoning: str = ""
    explanation: str = "LLM news assessment was not evaluated."


@dataclass
class RedFlagCheck:
    """Whether a discovery candidate's own coverage suggests a severe,
    company-specific risk (fraud, delisting, imminent bankruptcy, major
    legal action) that the quantitative liquidity/price floor can't see."""

    available: bool
    flagged: bool = False
    reason: str = ""


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

DAY_SUMMARY_SYSTEM_PROMPT = (
    "You are writing a two-to-three sentence plain-English recap of one "
    "day's activity for a small automated trading agent, for a "
    "non-technical operator reading a daily email. You will be given the "
    "day's outcome, risk signals, and the actions actually taken. Summarize "
    "only what is given -- do not invent numbers, symbols, or reasoning not "
    "present in the input. Do not give investment advice or predictions. "
    "Plain text only, no markdown, no JSON."
)

EXIT_NARRATIVE_SYSTEM_PROMPT = (
    "A small automated trading agent just sold a position for a stated "
    "price-based reason (a take-profit, stop-loss, or holding-horizon "
    "rule) -- that decision is already final and this note cannot change "
    "it. In one plain sentence, note anything in the provided headlines "
    "about this specific company that plausibly relates to today's price "
    "move. If nothing in the coverage seems relevant, say so briefly. Base "
    "this only on the provided headlines. Plain text only, no markdown, no "
    "JSON."
)

RED_FLAG_SYSTEM_PROMPT = (
    "You are screening one company's recent headlines before a small "
    "automated trading agent is allowed to consider buying it today. Flag "
    'red_flag=true only for a severe, company-specific risk clearly stated '
    "in the headlines: fraud or accounting restatement, imminent "
    "bankruptcy, stock delisting, or a major regulatory/legal action "
    "directly threatening the company. Do NOT flag routine bad news such "
    "as a single earnings miss, an analyst downgrade, or sector-wide "
    "weakness -- those are normal and already handled elsewhere. Base this "
    "only on the provided headlines; do not assume anything not stated."
    "\n\nRespond with only a JSON object and no other text, using exactly "
    "these keys:\n"
    '{"red_flag": true | false, "reason": "<one short plain sentence>"}'
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
    def _clean_narrative(text: str) -> str:
        """Strip code fences/quoting a model might add and bound the length;
        these replies are read by a human, not parsed, so this is tolerant
        rather than strict like `_parse_assessment`/`_parse_red_flag`."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip("\"'").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:MAX_NARRATIVE_CHARS]

    @staticmethod
    def _parse_red_flag(text: str) -> RedFlagCheck:
        """Validate and repair a model reply into a usable red-flag check."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        flagged = bool(data.get("red_flag", False))
        reason = str(data.get("reason", "")).strip()
        return RedFlagCheck(available=True, flagged=flagged, reason=reason)

    def _chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        json_mode: bool,
        max_tokens: int = MAX_RESPONSE_TOKENS,
    ) -> str:
        """Shared request/response plumbing for every call this class makes."""
        import requests

        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        if not text:
            raise RuntimeError("The model returned an empty reply")
        return text

    def summarize_day(self, context_text: str) -> str:
        """A short plain-English recap of one iteration's report, for the
        daily email. Purely descriptive summarization of decisions already
        made -- never a new decision itself. Raises on any problem; callers
        should fail open to an empty string (the email just omits it)."""
        if not context_text.strip():
            return ""
        text = self._chat(
            DAY_SUMMARY_SYSTEM_PROMPT, context_text, json_mode=False, max_tokens=NARRATIVE_MAX_TOKENS
        )
        return self._clean_narrative(text)

    def explain_exit(self, symbol: str, price_reason: str, articles: list[dict]) -> str:
        """One plain sentence connecting a just-submitted exit to that
        symbol's own headlines, if it has dedicated coverage today. Purely
        descriptive -- the exit already fired on price alone before this is
        ever called. Raises on any problem; callers should fail open."""
        article_text = self._format_articles(articles)
        if not article_text:
            return ""
        user_text = (
            f"{symbol} was just sold by an automated trading agent. Reason "
            f"given: {price_reason}.\n\n"
            f"Today's news coverage specifically about {symbol}:\n\n{article_text}"
        )
        text = self._chat(
            EXIT_NARRATIVE_SYSTEM_PROMPT, user_text, json_mode=False, max_tokens=NARRATIVE_MAX_TOKENS
        )
        return self._clean_narrative(text)

    def check_red_flag(self, symbol: str, articles: list[dict]) -> RedFlagCheck:
        """Screen one discovery candidate's dedicated coverage for a severe,
        company-specific risk before it's allowed into today's tradeable
        universe. Raises on any problem; callers should fail open (treat as
        not flagged, exactly as before this feature)."""
        article_text = self._format_articles(articles)
        if not article_text:
            return RedFlagCheck(available=False)
        user_text = f"Recent headlines specifically about {symbol}:\n\n{article_text}"
        text = self._chat(
            RED_FLAG_SYSTEM_PROMPT, user_text, json_mode=True, max_tokens=NARRATIVE_MAX_TOKENS
        )
        return self._parse_red_flag(text)

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
        text = self._chat(SYSTEM_PROMPT + JSON_FORMAT_INSTRUCTIONS, user_text, json_mode=True)
        return self._parse_assessment(text, len(articles), self.model)
