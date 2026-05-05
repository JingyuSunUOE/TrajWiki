"""Trajectory-level retrieval summary helpers."""

from __future__ import annotations

import re
from typing import Iterable

from trajpatch.memory.readability import clean_internal_memory_summary_text, clean_readable_values
from trajpatch.utils.text import collapse_whitespace, extract_keywords

SUMMARY_KEYWORD_BLACKLIST = {
    "card",
    "claim",
    "claims",
    "conflict",
    "conflicts",
    "entities",
    "entity",
    "evidence",
    "fact",
    "facts",
    "historical",
    "identity",
    "item",
    "items",
    "label",
    "memory",
    "named",
    "none",
    "profile",
    "recent",
    "recorded",
    "relation",
    "relations",
    "source",
    "stable",
    "summary",
    "surface",
    "temporal",
    "topic",
    "trajectory",
    "uncertain",
    "uncertainty",
    "update",
    "updates",
}

_SUMMARY_KEYWORD_METADATA_FIELDS = (
    "source_event_canonical_terms_v1",
    "source_event_object_terms_v1",
    "source_surface_raw_terms_v1",
    "source_surface_terms_v1",
    "exact_terms_v2",
    "exact_terms",
    "display_items",
    "display_key_facts",
    "facet_values",
)


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = collapse_whitespace(str(value))
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def build_deterministic_retrieval_summary(
    *,
    trajectory_label: str,
    active_claims: list[str],
    uncertain_claims: list[str],
    exact_terms: list[str],
    facet_values: list[str],
    entity_mentions: list[str],
    recent_snapshot_notes: list[str],
    display_items: list[str] | None = None,
    display_named_entities: list[str] | None = None,
    display_counts: list[str] | None = None,
    display_key_facts: list[str] | None = None,
) -> str:
    profile_lines = _dedupe_preserve(
        [
            f"Trajectory label: {trajectory_label}" if trajectory_label else "",
            *entity_mentions[:8],
            *active_claims[:10],
        ]
    )
    readable_items = clean_readable_values(
        [
            *(display_items or []),
            *(display_named_entities or []),
            *(display_counts or []),
            *exact_terms,
            *facet_values,
        ],
        allow_single_word=True,
        limit=16,
    )
    item_lines = _dedupe_preserve(readable_items)
    relation_lines = _dedupe_preserve([*(display_key_facts or [])[:8], *recent_snapshot_notes[:8]])
    conflict_lines = _dedupe_preserve(uncertain_claims[:8])

    if not profile_lines:
        profile_lines = ["None recorded."]
    if not item_lines:
        item_lines = ["None recorded."]
    if not relation_lines:
        relation_lines = ["None recorded."]
    if not conflict_lines:
        conflict_lines = ["None recorded."]

    return (
        "## Profile / Stable Facts\n"
        + "\n".join(f"- {line}" for line in profile_lines)
        + "\n\n## Item Sets / Named Entities\n"
        + "\n".join(f"- {line}" for line in item_lines)
        + "\n\n## Relations / Temporal Updates\n"
        + "\n".join(f"- {line}" for line in relation_lines)
        + "\n\n## Conflicts / Uncertainty\n"
        + "\n".join(f"- {line}" for line in conflict_lines)
    )


def build_summary_supporting_state(
    *,
    trajectory_label: str,
    active_claims: list[str],
    uncertain_claims: list[str],
    exact_terms: list[str],
    facet_values: list[str],
    entity_mentions: list[str],
    recent_snapshot_notes: list[str],
    display_items: list[str] | None = None,
    display_named_entities: list[str] | None = None,
    display_counts: list[str] | None = None,
    display_key_facts: list[str] | None = None,
) -> str:
    def _render_section(title: str, rows: list[str]) -> str:
        values = _dedupe_preserve(rows)
        if not values:
            values = ["None recorded."]
        return f"## {title}\n" + "\n".join(f"- {value}" for value in values)

    sections = [
        _render_section("Trajectory Label", [trajectory_label] if trajectory_label else []),
        _render_section("Readable Display Items", display_items or []),
        _render_section("Validated Named Entities", display_named_entities or []),
        _render_section("Validated Counts / Titles / Places", display_counts or []),
        _render_section("Active Claims", active_claims),
        _render_section("Facets", facet_values),
        _render_section("Exact Terms", exact_terms),
        _render_section("Entities", entity_mentions),
        _render_section("Display Key Facts", display_key_facts or []),
        _render_section("Recent Snapshot Notes", recent_snapshot_notes),
        _render_section("Uncertain Claims", uncertain_claims),
    ]
    return "\n\n".join(sections)


def _metadata_values(metadata: dict[str, object] | None, *field_names: str) -> list[str]:
    if not metadata:
        return []
    values: list[str] = []
    for field_name in field_names:
        raw = metadata.get(field_name) or []
        if isinstance(raw, dict):
            raw = raw.values()
        values.extend(str(value).strip() for value in list(raw) if str(value).strip())
    return _dedupe_preserve(values)


def is_internal_summary_keyword(value: object) -> bool:
    token = collapse_whitespace(str(value or "")).casefold().strip(" ,.;:!?")
    if not token:
        return True
    if token in SUMMARY_KEYWORD_BLACKLIST:
        return True
    return bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+){2,}", token))


def sanitize_summary_keyword_values(values: Iterable[object], *, limit: int = 32) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in sorted(extract_keywords(str(value or ""))):
            if is_internal_summary_keyword(token):
                continue
            if token in seen:
                continue
            seen.add(token)
            cleaned.append(token)
            if len(cleaned) >= limit:
                return cleaned
    return cleaned


def summary_keywords_v2(
    summary_text: str,
    metadata: dict[str, object] | None = None,
    *,
    limit: int = 32,
) -> list[str]:
    """Build retrieval keywords without leaking summary/debug scaffolding."""

    prioritized_values: list[object] = []
    for field_name in _SUMMARY_KEYWORD_METADATA_FIELDS:
        prioritized_values.extend(_metadata_values(metadata, field_name))
    cleaned_summary = clean_internal_memory_summary_text(summary_text, max_parts=None)
    prioritized_values.append(cleaned_summary)
    return sanitize_summary_keyword_values(prioritized_values, limit=limit)


def removed_internal_summary_keywords(summary_text: str) -> list[str]:
    raw_tokens = extract_keywords(summary_text)
    return sorted(token for token in raw_tokens if is_internal_summary_keyword(token))


def summary_keywords(summary_text: str) -> list[str]:
    return summary_keywords_v2(summary_text)


def fallback_summary_from_metadata(metadata: dict[str, object], *, trajectory_label: str = "") -> str:
    latest_semantic_text = collapse_whitespace(str(metadata.get("latest_semantic_text") or ""))
    exact_terms = [str(term).strip() for term in list(metadata.get("exact_terms") or []) if str(term).strip()]
    facet_values = [str(value).strip() for value in list(metadata.get("facet_values") or []) if str(value).strip()]
    entity_mentions = [str(value).strip() for value in list(metadata.get("entity_mentions") or []) if str(value).strip()]
    display_items = [str(value).strip() for value in list(metadata.get("display_items") or []) if str(value).strip()]
    display_named_entities = [
        str(value).strip() for value in list(metadata.get("display_named_entities") or []) if str(value).strip()
    ]
    display_counts = [str(value).strip() for value in list(metadata.get("display_counts") or []) if str(value).strip()]
    display_key_facts = [
        str(value).strip() for value in list(metadata.get("display_key_facts") or []) if str(value).strip()
    ]
    active_claims = [latest_semantic_text] if latest_semantic_text else []
    return build_deterministic_retrieval_summary(
        trajectory_label=trajectory_label,
        active_claims=active_claims,
        uncertain_claims=[],
        exact_terms=exact_terms,
        facet_values=facet_values,
        entity_mentions=entity_mentions,
        recent_snapshot_notes=[latest_semantic_text] if latest_semantic_text else [],
        display_items=display_items,
        display_named_entities=display_named_entities,
        display_counts=display_counts,
        display_key_facts=display_key_facts,
    )
