"""Shared token-count estimate for the news/LLM pipeline.

Used by both `llm_news.py` (prompts sent to the local Ollama chat model) and
`article_filter.py` (prompts sent to the same model via its `/api/generate`
endpoint) to budget prompt text without a dependency on either model's real
tokenizer.
"""

try:
    import tiktoken

    # cl100k_base is not either caller's actual tokenizer (the local GGUF
    # models have their own vocabs, and Ollama exposes no tokenize endpoint
    # to query them), but a real BPE encoder still segments dense financial
    # text (tickers, "%", table-like runs of digits) the way a subword
    # tokenizer actually would, which a linear words/chars heuristic cannot.
    # Encoding is Rust-backed and fast (~0.3ms for a full article on this
    # Pi), so this is a strictly better estimate at negligible extra cost
    # when available.
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None

TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str, tokens_per_word: float = TOKENS_PER_WORD) -> int:
    """tiktoken's real BPE count when available; a words*tokens_per_word
    estimate otherwise. Approximate either way -- callers size their own
    budgets with that in mind. Never raises: arbitrary third-party or
    headline/summary text can contain sequences tiktoken treats as special
    tokens."""
    if not text:
        return 0
    if _TOKEN_ENCODER is not None:
        try:
            return len(_TOKEN_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    return int(len(text.split()) * tokens_per_word)
