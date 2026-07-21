"""Declarative mapping from validated JSON settings to strategy parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


EMAIL_CONFIG_KEYS = (
    "EMAIL_SMTP_HOST",
    "EMAIL_SMTP_PORT",
    "EMAIL_SMTP_USERNAME",
    "EMAIL_FROM_ADDRESS",
    "EMAIL_TO_ADDRESS",
    "EMAIL_USE_TLS",
)

NEWS_CONFIG_KEYS = (
    "NEWS_CONTEXT_ENABLED",
    "NEWS_LOOKBACK_HOURS",
    "NEWS_MAX_ARTICLES",
    "NEWS_HIGH_RISK_SCORE",
    "NEWS_SCORE_REFINEMENT_ENABLED",
)

LLM_CONFIG_KEYS = (
    "LLM_NEWS_ENABLED",
    "LLM_NEWS_MODEL",
    "LLM_NEWS_BASE_URL",
    "LLM_NEWS_FAIL_CLOSED_ON_UNAVAILABLE",
    "LLM_NEWS_BLOCK_SCORE",
)


def select_parameters(
    config: Mapping[str, Any],
    *key_groups: Iterable[str],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select validated config keys using their lowercase runtime names."""
    aliases = aliases or {}
    return {
        aliases.get(key, key.lower()): config[key]
        for group in key_groups
        for key in group
    }


def resolve_state_paths(
    base_dir: Path, filenames: Mapping[str, str]
) -> dict[str, str]:
    """Resolve all strategy-owned state files from one explicit manifest."""
    return {parameter: str(base_dir / filename) for parameter, filename in filenames.items()}
