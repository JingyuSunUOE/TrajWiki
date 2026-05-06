"""Lightweight context token estimation for offline ablation diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

TOKEN_ESTIMATOR_NAME = "whitespace_x1_3_v1"


def estimate_context_tokens(value: Any) -> int:
    """Estimate tokens with the same cheap policy used by existing diagnostics."""
    text = str(value or "")
    tokenish_count = len(text.split())
    return max(1, int(math.ceil(tokenish_count * 1.3))) if tokenish_count else 0


def estimate_context_tokens_many(values: Iterable[Any]) -> int:
    return sum(estimate_context_tokens(value) for value in values)


def token_breakdown(**sections: Any) -> dict[str, Any]:
    """Return a section-level token breakdown with a stable estimator label."""
    estimates = {name: estimate_context_tokens(value) for name, value in sections.items()}
    return {
        "token_estimator": TOKEN_ESTIMATOR_NAME,
        "sections": estimates,
        "total": sum(estimates.values()),
    }


def select_ranked_rows_with_budget(
    rows: Iterable[dict[str, Any]],
    *,
    budget_tokens: int,
    token_key: str = "estimated_tokens",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Keep ranked rows in order until either rank limit or budget is exhausted."""
    selected: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        if limit is not None and len(selected) >= limit:
            break
        item = dict(row)
        tokens = int(item.get(token_key) or item.get("context_token_estimate") or 0)
        if tokens <= 0:
            tokens = estimate_context_tokens(item)
        if used + tokens > budget_tokens:
            break
        item[token_key] = tokens
        selected.append(item)
        used += tokens
    return selected
