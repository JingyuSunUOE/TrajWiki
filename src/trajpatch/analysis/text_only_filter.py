"""Text-only visibility audit for LOCOMO evaluation rows.

The audit is intentionally conservative: it only excludes examples whose gold
answer appears to require visual perception or OCR beyond the raw message text
and the existing image caption text.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

FILTER_POLICY = "locomo_text_only_filter_v1"

_SHARED_IMAGE_RE = re.compile(r"\[shared image:\s*(?P<caption>[^\]]+)\]", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{2,120})[\"“”']")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_OCR_HINT_RE = re.compile(
    r"\b("
    r"book cover|cover of (?:a )?book|poster|flyer|flier|sign|signage|screenshot|screen|"
    r"menu|receipt|certificate|document|paper|letter|card|plaque|label|badge|ticket|"
    r"magazine cover|newspaper|logo text|text on|written on|words on"
    r")\b",
    re.IGNORECASE,
)
_VISUAL_COUNT_QUESTION_RE = re.compile(r"\b(how many|number of|count of)\b", re.IGNORECASE)
_VISUAL_ATTRIBUTE_RE = re.compile(
    r"\b(color|colour|symbol|flag|logo|pattern|shape|mark|emblem|sign)\b",
    re.IGNORECASE,
)
_VISUAL_OBJECT_RE = re.compile(
    r"\b(painting|drawing|sketch|mural|artwork|necklace|"
    r"bracelet|shirt|dress|dog|cat|turtle|tortoise|animal|flower|car|cup|bowl|object)\b",
    re.IGNORECASE,
)

_ARTICLES = {"a", "an", "the"}
_NUMBER_ALIASES = {
    "once": "one",
    "twice": "two",
    "thrice": "three",
}


def audit_text_only_visibility(
    row: Mapping[str, Any],
    evidence_text_by_ref: Mapping[str, str],
) -> dict[str, Any]:
    """Return text-only filter metadata for one evaluated LOCOMO row."""

    metadata = dict(row.get("metadata") or {})
    query_metadata = dict(metadata.get("query_metadata") or {})
    semantic_metadata = dict(metadata.get("semantic_metrics") or {})
    gold_refs = [
        str(ref)
        for ref in list(query_metadata.get("gold_evidence_refs") or query_metadata.get("gold_evidence_raw") or [])
        if str(ref).strip()
    ]
    evidence_texts = {ref: str(evidence_text_by_ref.get(ref) or "") for ref in gold_refs}
    image_refs = [ref for ref, text in evidence_texts.items() if _SHARED_IMAGE_RE.search(text)]
    available_text = " ".join(evidence_texts.values())
    gold_items = _gold_items_from_row(row, semantic_metadata)
    missing_items = [
        item for item in gold_items if item and not _item_supported_by_text(str(item), available_text)
    ]

    visual_type: str | None = None
    excluded = False
    reason: str | None = None
    if missing_items and image_refs:
        visual_type = _classify_visual_dependency(
            question=str(row.get("question") or ""),
            gold_items=missing_items,
            evidence_text=" ".join(evidence_texts.get(ref, "") for ref in image_refs),
        )
        if visual_type in {"ocr_text_on_image", "visual_object", "visual_count", "visual_attribute"}:
            excluded = True
            reason = visual_type
        else:
            reason = "ambiguous_needs_review"
    elif missing_items:
        reason = "gold_items_missing_from_text_without_visual_evidence"

    return {
        "filter_policy": FILTER_POLICY,
        "text_only_eligible": not excluded,
        "excluded_from_text_only": excluded,
        "visual_dependency_type": visual_type,
        "exclusion_reason": reason,
        "gold_items": gold_items,
        "gold_items_missing_from_text_input": missing_items,
        "gold_evidence_refs": gold_refs,
        "gold_evidence_image_refs": image_refs,
        "available_evidence_text_preview": {
            ref: _preview_text(text) for ref, text in evidence_texts.items() if text
        },
    }


def summarize_text_only_filter(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    filters = [dict(dict(row.get("metadata") or {}).get("evaluation_filter") or {}) for row in rows]
    filters = [item for item in filters if item]
    total = len(filters)
    excluded = [item for item in filters if bool(item.get("excluded_from_text_only"))]
    ambiguous = [
        item
        for item in filters
        if item.get("visual_dependency_type") == "ambiguous_needs_review"
        or item.get("exclusion_reason") == "ambiguous_needs_review"
    ]
    by_reason = Counter(str(item.get("exclusion_reason") or "included") for item in filters)
    by_type = Counter(str(item.get("visual_dependency_type") or "none") for item in filters)
    return {
        "policy": FILTER_POLICY,
        "total_queries": total,
        "included_count": total - len(excluded),
        "excluded_count": len(excluded),
        "ambiguous_count": len(ambiguous),
        "by_reason": dict(by_reason),
        "by_visual_dependency_type": dict(by_type),
        "excluded_query_task_ids": [
            str(row.get("query_task_id"))
            for row in rows
            if bool(dict(dict(row.get("metadata") or {}).get("evaluation_filter") or {}).get("excluded_from_text_only"))
        ],
    }


def compact_filter_for_details(result: Mapping[str, Any]) -> dict[str, Any]:
    """Drop potentially bulky evidence previews before inserting into details rows."""

    keys = [
        "filter_policy",
        "text_only_eligible",
        "excluded_from_text_only",
        "visual_dependency_type",
        "exclusion_reason",
        "gold_items_missing_from_text_input",
        "gold_evidence_image_refs",
    ]
    return {key: result.get(key) for key in keys if key in result}


def manifest_entry_for_row(row: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "filter_policy": FILTER_POLICY,
        "sample_id": row.get("sample_id"),
        "query_task_id": row.get("query_task_id"),
        "subset_key": row.get("subset_key"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "text_only_eligible": result.get("text_only_eligible"),
        "excluded_from_text_only": result.get("excluded_from_text_only"),
        "visual_dependency_type": result.get("visual_dependency_type"),
        "exclusion_reason": result.get("exclusion_reason"),
        "gold_items": list(result.get("gold_items") or []),
        "gold_items_missing_from_text_input": list(result.get("gold_items_missing_from_text_input") or []),
        "gold_evidence_refs": list(result.get("gold_evidence_refs") or []),
        "gold_evidence_image_refs": list(result.get("gold_evidence_image_refs") or []),
        "available_evidence_text_preview": dict(result.get("available_evidence_text_preview") or {}),
    }


def _gold_items_from_row(row: Mapping[str, Any], semantic_metadata: Mapping[str, Any]) -> list[str]:
    semantic_items = [
        str(item).strip()
        for item in list(semantic_metadata.get("f1_reference_items") or [])
        if str(item).strip()
    ]
    if semantic_items:
        return _dedupe_preserve(semantic_items)
    return _split_gold_answer(str(row.get("gold_answer") or ""))


def _split_gold_answer(gold_answer: str) -> list[str]:
    text = " ".join(str(gold_answer or "").split()).strip()
    if not text:
        return []
    quoted = [match.group(1).strip() for match in _QUOTED_RE.finditer(text) if match.group(1).strip()]
    if quoted:
        return _dedupe_preserve(quoted)
    if _looks_like_atomic_answer(text):
        return [text]
    parts = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", text)
    cleaned = [part.strip(" .") for part in parts if part.strip(" .")]
    return _dedupe_preserve(cleaned or [text])


def _looks_like_atomic_answer(text: str) -> bool:
    lower = text.casefold()
    if re.search(r"\b\d{4}\b|\b(?:january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\b", lower):
        return True
    if len(_TOKEN_RE.findall(lower)) <= 3:
        return True
    return False


def _item_supported_by_text(item: str, text: str) -> bool:
    normalized_item = _normalize_for_match(item)
    normalized_text = _normalize_for_match(text)
    if not normalized_item:
        return True
    if normalized_item in normalized_text:
        return True
    item_tokens = normalized_item.split()
    text_tokens = set(normalized_text.split())
    if len(item_tokens) == 1:
        return item_tokens[0] in text_tokens
    return all(token in text_tokens for token in item_tokens if len(token) > 2)


def _normalize_for_match(value: str) -> str:
    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(str(value).casefold()):
        token = _NUMBER_ALIASES.get(raw_token, raw_token)
        if token in _ARTICLES:
            continue
        tokens.append(_singularize(token))
    return " ".join(tokens)


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _classify_visual_dependency(*, question: str, gold_items: list[str], evidence_text: str) -> str:
    combined = " ".join([question, " ".join(gold_items), evidence_text])
    if _OCR_HINT_RE.search(combined):
        return "ocr_text_on_image"
    if _VISUAL_COUNT_QUESTION_RE.search(question) and _VISUAL_OBJECT_RE.search(evidence_text):
        return "visual_count"
    if _VISUAL_ATTRIBUTE_RE.search(combined):
        return "visual_attribute"
    if _VISUAL_OBJECT_RE.search(combined):
        return "visual_object"
    return "ambiguous_needs_review"


def _preview_text(text: str, *, limit: int = 500) -> str:
    collapsed = " ".join(str(text or "").split())
    return collapsed[:limit]


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output
