"""Text helpers used in prompt construction and retrieval."""

from __future__ import annotations

import re


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def keyword_overlap_score(query_keywords: set[str], candidate_keywords: set[str]) -> float:
    if not query_keywords or not candidate_keywords:
        return 0.0
    overlap = len(query_keywords & candidate_keywords)
    union = len(query_keywords | candidate_keywords)
    return overlap / union if union else 0.0


def extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-']+", text.lower())
    return {token for token in tokens if len(token) > 2}
