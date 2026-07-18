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

try:
    import tiktoken

    # cl100k_base is not this pipeline's actual tokenizer (the local GGUF
    # models have their own vocabs, and Ollama exposes no tokenize endpoint
    # to query them -- see MAX_PROMPT_TOKENS_ESTIMATE), but a real BPE
    # encoder still segments dense financial text (tickers, "%", table-like
    # runs of digits) the way a subword tokenizer actually would, which a
    # linear words/chars heuristic cannot. Encoding is Rust-backed and fast
    # (~0.3ms for a full article on this Pi), so this is a strictly better
    # estimate at negligible extra cost when available.
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None

OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

MAX_SUMMARY_CHARS = 400
# Defensive cap on how many symbols get listed in the prompt; the day's
# evaluation universe (watchlist + held + discovery candidates) is already
# bounded well under this in practice.
MAX_SYMBOLS_LISTED = 60
# Ollama silently defaults an API request to a ~4096-token context regardless
# of what the underlying model actually supports (confirmed via its own
# "vram-based default context" log line) unless a request explicitly asks
# for more. granite-4.0-micro trains at 131072 tokens natively, so this
# reclaims real headroom over that implicit default -- it is not, by itself,
# a green light to fill the budgets below anywhere close to this number: see
# MAX_PROMPT_TOKENS_ESTIMATE/REQUEST_TIMEOUT_SECONDS for why this Pi's raw
# CPU throughput, not context capacity, is what actually bounds them.
OLLAMA_NUM_CTX = 8192
# Approximate input-token budget for one prompt's assembled text (articles +
# symbol context; the fixed system prompt is reserved separately, see
# assess()/explain_exit()/check_red_flag()). No tokenizer dependency here
# (Pi-constrained installs, and a real one -- e.g. tiktoken -- would tokenize
# with an OpenAI vocab that has nothing to do with the local GGUF models this
# actually calls, so it would just be a differently-wrong estimate dressed up
# as an exact one). A words*TOKENS_PER_WORD estimate is used instead of a
# chars/4 one (see article_filter.py's TOKENS_PER_WORD, calibrated the same
# way): measured against this Pi's actual Ollama server, chars/4 undercounted
# llama3.2:3b's real tokens by ~1.6x but overcounted granite-4.0-micro's by
# ~1.1x for the same content (different tokenizers). This budget is sized
# against the worse (undercounting) case so it stays safe if LLM_NEWS_MODEL
# is changed again. When the budget would be exceeded, articles are
# prioritized by |score| rather than hard-truncated in place.
TOKENS_PER_WORD = 1.3
MAX_PROMPT_TOKENS_ESTIMATE = 1800
MIN_SCORE = -10
MAX_SCORE = 10
RISK_LEVELS = ("high", "elevated", "normal", "constructive")
# This assessment is optional and fails open. A local, CPU-bound small model
# is slower than a hosted API -- prompt-eval on this Pi runs ~27-31
# tokens/sec regardless of which ~3-4B model is configured (measured
# directly against both llama3.2:3b and granite-4.0-micro; a bigger context
# *window* does not mean faster *throughput*), and generation is far slower
# still at ~4 tokens/sec. Sized so MAX_PROMPT_TOKENS_ESTIMATE's worst-case
# real token count (~1.6x, see above) plus the system prompt plus
# MAX_RESPONSE_TOKENS' worth of generation still finishes with margin to
# spare -- generous, but this is a once-a-day background call that should
# get a real answer rather than fail open on a hardware-imposed clock.
REQUEST_TIMEOUT_SECONDS = 240
# assess()'s reply is just a score plus "two or three plain sentences," so
# it doesn't need a large budget; sized against REQUEST_TIMEOUT_SECONDS at
# ~4 tokens/sec generation, see REQUEST_TIMEOUT_SECONDS's comment.
MAX_RESPONSE_TOKENS = 400
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
    def _estimate_tokens(text: str) -> int:
        """tiktoken's real BPE count when available (see _TOKEN_ENCODER for
        why this is a better proxy than a linear heuristic despite not
        matching the local model's actual vocab); a words*TOKENS_PER_WORD
        estimate otherwise. Approximate either way, see
        MAX_PROMPT_TOKENS_ESTIMATE. Never raises: arbitrary headline/summary
        text can contain sequences tiktoken treats as special tokens."""
        if not text:
            return 0
        if _TOKEN_ENCODER is not None:
            try:
                return len(_TOKEN_ENCODER.encode(text, disallowed_special=()))
            except Exception:
                pass
        return int(len(text.split()) * TOKENS_PER_WORD)

    @classmethod
    def _prioritize_articles(cls, articles: list[dict], budget_tokens: int) -> list[dict]:
        """Select the highest-signal prefix of `articles` (by |score|
        descending, ties keep original order) that fits an approximate token
        budget, then restore original order so the model still reads
        coverage chronologically. Missing "score" (e.g. explain_exit/
        check_red_flag callers) is treated as 0 -- no priority signal, falls
        back to original order. Never truncates an individual article's text
        (that's MAX_SUMMARY_CHARS's job, applied first here); this only
        decides which whole articles are included."""
        if budget_tokens <= 0 or not articles:
            return []
        ranked = sorted(
            enumerate(articles),
            key=lambda pair: (-abs(int(pair[1].get("score", 0) or 0)), pair[0]),
        )
        keep: set[int] = set()
        used = 0
        for index, article in ranked:
            headline = str(article.get("headline", "")).strip()
            if not headline:
                continue
            summary = str(article.get("summary", "")).strip()[:MAX_SUMMARY_CHARS]
            line = f"{headline} - {summary}" if summary else headline
            tokens = cls._estimate_tokens(line)
            if used + tokens > budget_tokens:
                continue
            keep.add(index)
            used += tokens
        return [articles[index] for index in sorted(keep)]

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
        # When the universe exceeds MAX_SYMBOLS_LISTED, prioritize held
        # positions and symbols with dedicated news coverage over a plain
        # alphabetical cutoff, so the ones that matter most survive the cap.
        priority = [symbol for symbol in unique_symbols if symbol in held_upper]
        priority += [
            symbol
            for symbol in unique_symbols
            if symbol not in held_upper and symbol_scores.get(symbol, 0) != 0
        ]
        seen = set(priority)
        priority += [symbol for symbol in unique_symbols if symbol not in seen]
        selected = sorted(priority[:MAX_SYMBOLS_LISTED])
        listed = ", ".join(f"{symbol}*" if symbol in held_upper else symbol for symbol in selected)
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
            "options": {"num_ctx": OLLAMA_NUM_CTX},
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
        reserved = self._estimate_tokens(EXIT_NARRATIVE_SYSTEM_PROMPT)
        article_budget = max(0, MAX_PROMPT_TOKENS_ESTIMATE - reserved)
        article_text = self._format_articles(
            self._prioritize_articles(articles, article_budget)
        )
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
        reserved = self._estimate_tokens(RED_FLAG_SYSTEM_PROMPT)
        article_budget = max(0, MAX_PROMPT_TOKENS_ESTIMATE - reserved)
        article_text = self._format_articles(
            self._prioritize_articles(articles, article_budget)
        )
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
        wrapper = "Assess today's market news. Articles from the last 24 hours:\n\n"
        symbol_context = self._format_symbol_context(
            symbols or [], held_symbols or set(), symbol_scores or {}
        )
        # Reserve room for the fixed system prompt, the fixed wrapper text,
        # and the (already bounded) symbol context, then give whatever's
        # left of the budget to the highest-signal articles rather than
        # dumping every headline in -- fails open to the unfiltered list on
        # any problem, since this feature must never break an assessment
        # that worked before. The system prompt rides on every call and was
        # previously left out of this reservation entirely, which is why a
        # 50-article day still overflowed the intended budget in practice.
        reserved = (
            self._estimate_tokens(SYSTEM_PROMPT + JSON_FORMAT_INSTRUCTIONS)
            + self._estimate_tokens(wrapper)
            + self._estimate_tokens(symbol_context)
        )
        article_budget = max(0, MAX_PROMPT_TOKENS_ESTIMATE - reserved)
        try:
            selected_articles = self._prioritize_articles(articles, article_budget)
        except Exception:
            selected_articles = articles
        article_text = self._format_articles(selected_articles)
        if not article_text:
            return LLMNewsAssessment(
                available=False,
                explanation="No usable headlines were available to assess.",
            )
        user_text = wrapper + article_text
        if symbol_context:
            user_text += "\n\n" + symbol_context
        text = self._chat(SYSTEM_PROMPT + JSON_FORMAT_INSTRUCTIONS, user_text, json_mode=True)
        return self._parse_assessment(text, len(selected_articles), self.model)
