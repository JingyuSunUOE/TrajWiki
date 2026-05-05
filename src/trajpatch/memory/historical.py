"""Deterministic historical evidence cards for trajectory routing and wiki planning."""

from __future__ import annotations

from typing import Any, Iterable

from trajpatch.memory.readability import clean_internal_memory_summary_text, clean_readable_values
from trajpatch.memory.trajectory_summaries import sanitize_summary_keyword_values, summary_keywords_v2
from trajpatch.utils.text import collapse_whitespace, extract_keywords

GENERIC_TRAJECTORY_TERMS = {
    "activity",
    "activities",
    "area",
    "art",
    "card",
    "claim",
    "claims",
    "community",
    "conflict",
    "conflicts",
    "conversation",
    "entities",
    "entity",
    "evidence",
    "event",
    "events",
    "fact",
    "facts",
    "help",
    "historical",
    "home",
    "identity",
    "item",
    "items",
    "label",
    "memory",
    "named",
    "none",
    "people",
    "person",
    "place",
    "places",
    "plan",
    "plans",
    "profile",
    "project",
    "recent",
    "recorded",
    "relation",
    "relations",
    "source",
    "stable",
    "summary",
    "support",
    "surface",
    "temporal",
    "topic",
    "trajectory",
    "uncertain",
    "uncertainty",
    "update",
    "updates",
    "work",
    "working",
}


def dedupe_preserve(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = collapse_whitespace(str(value or ""))
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def metadata_list(metadata: dict[str, Any], *field_names: str) -> list[str]:
    values: list[str] = []
    for field_name in field_names:
        raw = metadata.get(field_name) or []
        if isinstance(raw, dict):
            raw = raw.values()
        values.extend(str(value).strip() for value in list(raw) if str(value).strip())
    return dedupe_preserve(values)


def is_generic_trajectory_term(value: Any) -> bool:
    text = collapse_whitespace(str(value or "")).strip(" ,.;:!?")
    if not text:
        return True
    folded = text.casefold()
    if folded in GENERIC_TRAJECTORY_TERMS:
        return True
    keywords = extract_keywords(text)
    if not keywords:
        return True
    return all(keyword in GENERIC_TRAJECTORY_TERMS for keyword in keywords)


def specific_terms(values: Iterable[Any], *, limit: int = 32) -> list[str]:
    readable = clean_readable_values(
        [
            collapse_whitespace(str(value or "")).strip(" ,.;:!?")
            for value in values
            if collapse_whitespace(str(value or ""))
        ],
        allow_single_word=True,
        limit=limit * 3,
    )
    return [
        value
        for value in dedupe_preserve(readable)
        if not is_generic_trajectory_term(value)
    ][:limit]


def sanitize_historical_item_terms(values: Iterable[Any], *, limit: int = 24) -> list[str]:
    return specific_terms(values, limit=limit)


def historical_item_terms_v2(
    *,
    metadata: dict[str, Any],
    active_claim_texts: list[str] | None = None,
    retrieval_summary_text: str = "",
    limit: int = 24,
) -> dict[str, list[str]]:
    source_surface_terms = metadata_list(metadata, "source_surface_terms_v1")
    source_surface_raw_terms = metadata_list(metadata, "source_surface_raw_terms_v1")
    source_event_object_terms = metadata_list(metadata, "source_event_object_terms_v1")
    source_event_canonical_terms = metadata_list(metadata, "source_event_canonical_terms_v1")
    source_temporal_relation_terms = metadata_list(metadata, "source_temporal_relation_terms_v1")
    source_event_values = dedupe_preserve([*source_event_canonical_terms, *source_event_object_terms])
    source_surface_values = dedupe_preserve([*source_event_values, *source_surface_raw_terms, *source_surface_terms])
    source_backed_terms = specific_terms(
        [
            *source_surface_values,
            *metadata_list(metadata, "display_items"),
            *metadata_list(metadata, "display_counts"),
            *metadata_list(metadata, "display_key_facts"),
            *source_surface_values,
            *metadata_list(metadata, "exact_terms_v2", "exact_terms"),
            *metadata_list(metadata, "facet_values"),
            *source_temporal_relation_terms,
            *specific_terms(active_claim_texts or [], limit=8),
        ],
        limit=limit,
    )
    fallback_budget = max(0, min(4, limit - len(source_backed_terms)))
    summary_fallback_terms: list[str] = []
    if fallback_budget:
        stored_keywords = metadata_list(metadata, "retrieval_summary_keywords_v2")
        if not stored_keywords:
            stored_keywords = sanitize_summary_keyword_values(
                metadata_list(metadata, "retrieval_summary_keywords", "latest_keywords"),
                limit=16,
            )
        if not stored_keywords and retrieval_summary_text:
            stored_keywords = summary_keywords_v2(retrieval_summary_text, metadata, limit=16)
        summary_fallback_terms = specific_terms(stored_keywords, limit=fallback_budget)
    return {
        "historical_item_terms": dedupe_preserve([*source_backed_terms, *summary_fallback_terms])[:limit],
        "source_backed_terms": source_backed_terms,
        "summary_fallback_terms": summary_fallback_terms,
    }


def term_key(value: Any) -> str:
    return collapse_whitespace(str(value or "")).casefold()


def term_keys(values: Iterable[Any]) -> set[str]:
    return {term_key(value) for value in values if term_key(value)}


def build_trajectory_historical_evidence_card(
    *,
    trajectory_id: str,
    trajectory_label: str,
    retrieval_summary_text: str,
    latest_semantic_text: str,
    metadata: dict[str, Any],
    active_claim_texts: list[str] | None = None,
    source_anchors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    source_surface_terms = metadata_list(metadata, "source_surface_terms_v1")
    source_surface_raw_terms = metadata_list(metadata, "source_surface_raw_terms_v1")
    source_event_object_terms = metadata_list(metadata, "source_event_object_terms_v1")
    source_event_canonical_terms = metadata_list(metadata, "source_event_canonical_terms_v1")
    source_temporal_relation_terms = metadata_list(metadata, "source_temporal_relation_terms_v1")
    source_event_values = dedupe_preserve([*source_event_canonical_terms, *source_event_object_terms])
    source_surface_values = dedupe_preserve([*source_event_values, *source_surface_raw_terms, *source_surface_terms])
    display_items = dedupe_preserve([*source_surface_values, *metadata_list(metadata, "display_items")])
    display_counts = metadata_list(metadata, "display_counts")
    display_key_facts = metadata_list(metadata, "display_key_facts")
    exact_terms = dedupe_preserve([*source_surface_values, *metadata_list(metadata, "exact_terms_v2", "exact_terms")])
    facet_values = metadata_list(metadata, "facet_values")
    entity_mentions = metadata_list(metadata, "entity_mentions")
    historical_terms = historical_item_terms_v2(
        metadata={
            **metadata,
            "source_surface_terms_v1": source_surface_terms,
            "source_surface_raw_terms_v1": source_surface_raw_terms,
            "source_event_object_terms_v1": source_event_object_terms,
            "source_event_canonical_terms_v1": source_event_canonical_terms,
            "source_temporal_relation_terms_v1": source_temporal_relation_terms,
            "display_items": display_items,
            "display_counts": display_counts,
            "display_key_facts": display_key_facts,
            "exact_terms_v2": exact_terms,
            "facet_values": facet_values,
        },
        active_claim_texts=active_claim_texts,
        retrieval_summary_text=retrieval_summary_text,
        limit=24,
    )
    historical_item_terms = historical_terms["historical_item_terms"]
    drift_cluster_keys = specific_terms(
        [
            *source_surface_values,
            *display_items,
            *display_counts,
            *exact_terms,
            *facet_values,
            *display_key_facts,
            *source_temporal_relation_terms,
        ],
        limit=24,
    )
    anchors = [
        {
            "source_ref": collapse_whitespace(str(anchor.get("source_ref") or "")),
            "text": collapse_whitespace(str(anchor.get("text") or ""))[:240],
        }
        for anchor in list(source_anchors or [])
        if collapse_whitespace(str(anchor.get("source_ref") or ""))
    ][:12]
    cleaned_summary = clean_internal_memory_summary_text(retrieval_summary_text, max_parts=3)
    identity_parts = dedupe_preserve(
        [
            trajectory_label,
            *entity_mentions[:6],
            cleaned_summary,
        ]
    )
    return {
        "trajectory_id": trajectory_id,
        "identity_summary": collapse_whitespace("; ".join(identity_parts[:4])),
        "recent_update": collapse_whitespace(latest_semantic_text)[:500],
        "historical_item_terms": historical_item_terms,
        "historical_item_terms_policy": "source_backed_terms_v2",
        "historical_item_terms_source_backed_v1": historical_terms["source_backed_terms"],
        "historical_item_terms_summary_fallback_v1": historical_terms["summary_fallback_terms"],
        "source_surface_terms": source_surface_values[:16],
        "facet_values": facet_values[:16],
        "entity_mentions": entity_mentions[:16],
        "source_anchors": anchors,
        "drift_cluster_keys": drift_cluster_keys,
        "display_items": display_items[:16],
        "display_counts": display_counts[:16],
        "display_key_facts": display_key_facts[:16],
    }


def render_trajectory_evidence_card(card: dict[str, Any]) -> str:
    anchors = [
        f"{anchor.get('source_ref')}: {anchor.get('text')}"
        for anchor in list(card.get("source_anchors") or [])[:6]
        if isinstance(anchor, dict)
    ]
    identity_summary = clean_internal_memory_summary_text(card.get("identity_summary") or "", max_parts=4)
    recent_update = clean_internal_memory_summary_text(card.get("recent_update") or "", max_parts=4)
    historical_terms = [
        clean_internal_memory_summary_text(value, max_parts=2)
        for value in list(card.get("historical_item_terms") or [])[:16]
    ]
    historical_terms = [value for value in historical_terms if value]
    source_surface_terms = [
        clean_internal_memory_summary_text(value, max_parts=2)
        for value in list(card.get("source_surface_terms") or [])[:16]
    ]
    source_surface_terms = [value for value in source_surface_terms if value]
    facet_values = [
        clean_internal_memory_summary_text(value, max_parts=2)
        for value in list(card.get("facet_values") or [])[:12]
    ]
    facet_values = [value for value in facet_values if value]
    return (
        f"CARD {card.get('trajectory_id')}\n"
        f"identity_summary={identity_summary or 'none'}\n"
        f"recent_update={recent_update or 'none'}\n"
        f"source_surface_terms={', '.join(source_surface_terms) or 'none'}\n"
        f"historical_item_terms={', '.join(historical_terms) or 'none'}\n"
        f"facet_values={', '.join(facet_values) or 'none'}\n"
        f"entity_mentions={', '.join(list(card.get('entity_mentions') or [])[:12]) or 'none'}\n"
        f"source_anchors={'; '.join(anchors) if anchors else 'none'}"
    )
