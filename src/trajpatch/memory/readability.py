"""Readability and grounding validators for retrieval-facing memory signals."""

from __future__ import annotations

import re
from collections.abc import Iterable

from trajpatch.utils.text import collapse_whitespace

_TRAILING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
_FRAGMENT_STARTERS = {
    "excited",
    "glad",
    "good",
    "here",
    "love",
    "loving",
    "now",
    "respect",
    "thanks",
    "wow",
    "yeah",
    "yes",
}
_FILLER_PHRASES = {
    "awesome",
    "excited for",
    "good",
    "here to a",
    "love and",
    "loving the",
    "now the",
    "respect for",
    "thanks",
    "wow",
    "wow.",
    "yeah",
    "yes",
}
_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")
_GENERIC_SINGLE_WORDS = {
    "heart",
    "time",
    "thing",
    "things",
    "stuff",
    "life",
    "family",
    "user",
}
_INTERNAL_SUMMARY_HEADING_RE = re.compile(
    r"(?im)^\s*#{1,6}\s*(?:"
    r"profile\s*/\s*stable\s*facts|"
    r"item\s+sets\s*/\s*named\s+entities|"
    r"relations\s*/\s*temporal\s+updates|"
    r"conflicts\s*/\s*uncertainty|"
    r"historical\s+evidence|"
    r"current\s+update|"
    r"summary"
    r")\s*$"
)
_INTERNAL_SUMMARY_INLINE_HEADING_RE = re.compile(
    r"(?i)#{1,6}\s*(?:"
    r"profile\s*/\s*stable\s*facts|"
    r"item\s+sets\s*/\s*named\s+entities|"
    r"relations\s*/\s*temporal\s+updates|"
    r"conflicts\s*/\s*uncertainty|"
    r"historical\s+evidence|"
    r"current\s+update|"
    r"summary"
    r")"
)
_INTERNAL_SUMMARY_FIELD_RE = re.compile(
    r"(?i)\b(?:"
    r"identity_summary|recent_update|historical_item_terms|source_surface_terms|"
    r"facet_values|entity_mentions|source_anchors|drift_cluster_keys|"
    r"display_items|display_counts|display_key_facts"
    r")\s*="
)
_INTERNAL_SUMMARY_TRAJECTORY_LABEL_RE = re.compile(
    r"(?i)\btrajectory\s+label:\s*[A-Za-z0-9][A-Za-z0-9_-]*(?:\s*[-;]\s*)?"
)
_INTERNAL_SUMMARY_CARD_RE = re.compile(r"(?i)\bCARD\s+epi-[A-Za-z0-9_-]+\b")
_INTERNAL_SUMMARY_PART_SPLIT_RE = re.compile(r"(?:\n+|\s+-\s+|;\s*)")
_INTERNAL_SUMMARY_DROP_VALUES = {
    "none",
    "none recorded",
    "not provided",
    "unknown",
    "n/a",
}


def normalized_surface(value: object) -> str:
    return collapse_whitespace(str(value or "").strip(" \t\r\n\"'“”‘’"))


def surface_key(value: object) -> str:
    return normalized_surface(value).strip(" .,:;!?").casefold()


def surface_supported(surface: object, texts: Iterable[object]) -> bool:
    needle = surface_key(surface)
    if not needle:
        return False
    for text in texts:
        haystack = surface_key(text)
        if needle and haystack and needle in haystack:
            return True
    return False


def is_fragment_like(value: object, *, allow_single_word: bool = True) -> bool:
    text = normalized_surface(value)
    if not text:
        return True
    key = text.casefold().strip(" .,:;!?")
    if key in _FILLER_PHRASES:
        return True
    tokens = _CONTENT_TOKEN_RE.findall(key)
    if not tokens:
        return True
    if any("'" in token and len(token) <= 4 for token in tokens):
        return True
    if tokens[-1] in _TRAILING_STOPWORDS:
        return True
    if tokens[0] in _FRAGMENT_STARTERS and len(tokens) <= 3:
        return True
    if len(tokens) == 1:
        if not allow_single_word:
            return True
        if tokens[0] in _GENERIC_SINGLE_WORDS:
            return True
    return False


def is_readable_claim_text(value: object) -> bool:
    text = normalized_surface(value)
    if not text or is_fragment_like(text, allow_single_word=False):
        return False
    return len(_CONTENT_TOKEN_RE.findall(text)) >= 3


def clean_readable_values(values: Iterable[object], *, allow_single_word: bool = True, limit: int | None = None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalized_surface(value)
        if not text or is_fragment_like(text, allow_single_word=allow_single_word):
            continue
        key = surface_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def clean_internal_memory_summary_text(
    value: object,
    *,
    max_parts: int | None = 12,
) -> str:
    """Remove debug-only memory summary scaffolding from routing-facing text.

    Per-trajectory summary exports intentionally keep their Markdown headings.
    Wiki routing text should not embed those internal headings because it is used
    as an embedding/rerank surface.
    """

    raw = str(value or "")
    if not raw.strip():
        return ""
    text = _INTERNAL_SUMMARY_HEADING_RE.sub("\n", raw)
    text = _INTERNAL_SUMMARY_INLINE_HEADING_RE.sub(" ", text)
    text = _INTERNAL_SUMMARY_CARD_RE.sub(" ", text)
    text = _INTERNAL_SUMMARY_TRAJECTORY_LABEL_RE.sub(" ", text)
    text = _INTERNAL_SUMMARY_FIELD_RE.sub(" ", text)
    text = re.sub(r"(?i)\bNone recorded\.?", " ", text)
    parts: list[str] = []
    seen: set[str] = set()
    for raw_part in _INTERNAL_SUMMARY_PART_SPLIT_RE.split(text):
        part = collapse_whitespace(raw_part.strip(" \t\r\n-:;,."))
        if not part:
            continue
        folded = part.casefold()
        if folded in _INTERNAL_SUMMARY_DROP_VALUES:
            continue
        if folded.startswith("trajectory label:"):
            continue
        if _INTERNAL_SUMMARY_INLINE_HEADING_RE.fullmatch(part):
            continue
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+){2,}", folded):
            continue
        key = surface_key(part)
        if not key or key in seen:
            continue
        seen.add(key)
        parts.append(part)
        if max_parts is not None and len(parts) >= max_parts:
            break
    return collapse_whitespace(" - ".join(parts))


def count_fragment_lines(markdown_section_text: str) -> int:
    count = 0
    for raw_line in markdown_section_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        value = re.sub(r"^\-\s+(?:\*\*[^*]+:\*\*\s*)?", "", line).strip()
        if is_fragment_like(value, allow_single_word=True):
            count += 1
    return count
