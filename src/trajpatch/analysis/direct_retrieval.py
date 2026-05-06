"""Reusable direct trajectory retrieval diagnostics for offline ablations."""

from __future__ import annotations

import re
from typing import Any, Iterable

from trajpatch.analysis.context_cost import estimate_context_tokens_many
from trajpatch.memory.facets import normalize_entity_key
from trajpatch.memory.historical import sanitize_historical_item_terms
from trajpatch.utils.text import collapse_whitespace, extract_keywords

DIRECT_TRAJECTORY_DIAGNOSTIC_TOP_N = 50
DIRECT_TRAJECTORY_TERM_LIMIT = 12
DIRECT_TRAJECTORY_STRING_LIMIT = 80

_ANSWER_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "for",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}

_DIRECT_TRAJECTORY_GENERIC_TERMS = {
    "about",
    "answer",
    "asked",
    "assistant",
    "context",
    "conversation",
    "details",
    "experience",
    "fact",
    "family",
    "feel",
    "friend",
    "good",
    "great",
    "help",
    "important",
    "information",
    "memory",
    "question",
    "said",
    "shared",
    "support",
    "talk",
    "thing",
    "things",
    "user",
}


def _direct_signal_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if value is None:
        return strings
    if isinstance(value, str):
        compact = collapse_whitespace(value)
        return [compact] if compact else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        for child in value.values():
            strings.extend(_direct_signal_strings(child))
        return strings
    if isinstance(value, (list, tuple, set)):
        for child in value:
            strings.extend(_direct_signal_strings(child))
        return strings
    return []


def direct_query_terms(
    *,
    question: str,
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
) -> set[str]:
    terms = {
        term
        for term in extract_keywords(question)
        if term not in _ANSWER_TOKEN_STOPWORDS and term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
    }
    for entity in query_entities:
        terms.update(
            term for term in extract_keywords(entity) if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        )
        entity_key = normalize_entity_key(entity)
        if entity_key:
            terms.add(entity_key)
    for value in list(query_facets.get("tags") or []) + list(query_facets.get("values") or []):
        terms.update(
            term
            for term in extract_keywords(str(value).replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        )
    item_family = str(query_shape.get("item_family") or "").strip()
    if item_family:
        terms.update(
            term
            for term in extract_keywords(item_family.replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        )
    return terms


def _truncate_direct_value(value: Any, limit: int = DIRECT_TRAJECTORY_STRING_LIMIT) -> str:
    text = collapse_whitespace(str(value or ""))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _compact_direct_terms(values: Iterable[Any], *, limit: int = DIRECT_TRAJECTORY_TERM_LIMIT) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _truncate_direct_value(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _direct_term_matches(query_terms: set[str], values: Iterable[Any]) -> list[str]:
    matched: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = collapse_whitespace(str(value or ""))
        if not text:
            continue
        value_terms = {
            term
            for term in extract_keywords(text)
            if term not in _ANSWER_TOKEN_STOPWORDS and term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if not (query_terms & value_terms):
            continue
        compact = _truncate_direct_value(text)
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        matched.append(compact)
        if len(matched) >= DIRECT_TRAJECTORY_TERM_LIMIT:
            break
    return matched


def direct_trajectory_signal_profile(
    *,
    trajectory_id: str,
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metadata = dict(trajectory_metadata.get(trajectory_id) or {})
    metadata_values: list[str] = []
    profile: dict[str, list[str]] = {
        "exact_terms": [],
        "source_surface_terms": [],
        "historical_terms": [],
        "display_items": [],
        "display_counts": [],
        "display_key_facts": [],
        "facet_values": [],
        "entity_mentions": [],
        "entity_keys": [],
        "drift_cluster_keys": [],
    }

    def add_profile_values(name: str, value: Any) -> None:
        strings = _direct_signal_strings(value)
        profile.setdefault(name, []).extend(strings)
        metadata_values.extend(strings)

    for key in [
        "label",
        "retrieval_summary_text",
        "trajectory_identity_summary_v1",
        "trajectory_recent_update_v1",
    ]:
        metadata_values.extend(_direct_signal_strings(metadata.get(key)))
    historical_values = sanitize_historical_item_terms(
        list(
            metadata.get("trajectory_historical_item_terms_v2")
            or metadata.get("trajectory_historical_item_terms_v1")
            or []
        ),
        limit=DIRECT_TRAJECTORY_TERM_LIMIT,
    )
    if historical_values:
        add_profile_values("historical_terms", historical_values)
    for key, profile_key in [
        ("historical_item_terms", "historical_terms"),
        ("wiki_historical_item_terms", "historical_terms"),
        ("exact_terms", "exact_terms"),
        ("exact_terms_v2", "exact_terms"),
        ("source_surface_terms_v1", "source_surface_terms"),
        ("source_surface_raw_terms_v1", "source_surface_terms"),
        ("display_items", "display_items"),
        ("display_counts", "display_counts"),
        ("display_key_facts", "display_key_facts"),
        ("facet_values", "facet_values"),
        ("entity_mentions", "entity_mentions"),
        ("entity_keys", "entity_keys"),
        ("drift_cluster_keys", "drift_cluster_keys"),
    ]:
        add_profile_values(profile_key, metadata.get(key))
    for card_key in ["trajectory_evidence_card_v1", "trajectory_historical_evidence_card_v1"]:
        card = metadata.get(card_key)
        metadata_values.extend(_direct_signal_strings(card))
        if isinstance(card, dict):
            for key, profile_key in [
                ("source_surface_terms", "source_surface_terms"),
                ("display_items", "display_items"),
                ("display_counts", "display_counts"),
                ("display_key_facts", "display_key_facts"),
                ("facet_values", "facet_values"),
                ("entity_mentions", "entity_mentions"),
                ("source_anchors", "historical_terms"),
            ]:
                profile.setdefault(profile_key, []).extend(_direct_signal_strings(card.get(key)))
            profile.setdefault("historical_terms", []).extend(
                _direct_signal_strings(
                    sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24)
                )
            )
    claim_values: list[str] = []
    for claim in claims_by_trajectory.get(trajectory_id, []):
        claim_values.append(str(claim.get("text") or ""))
        exact_terms = _direct_signal_strings(claim.get("exact_terms"))
        facets = _direct_signal_strings(claim.get("facets"))
        display_signals = _direct_signal_strings(claim.get("display_signals"))
        claim_values.extend(exact_terms)
        claim_values.extend(facets)
        claim_values.extend(display_signals)
        profile["exact_terms"].extend(exact_terms)
        profile["facet_values"].extend(facets)
        profile["display_items"].extend(display_signals)
    texts = [value for value in [*metadata_values, *claim_values] if collapse_whitespace(str(value or ""))]
    has_metadata_signals = any(collapse_whitespace(value) for value in metadata_values)
    return {
        "texts": texts,
        "has_metadata_signals": has_metadata_signals,
        **{key: _compact_direct_terms(values, limit=50) for key, values in profile.items()},
    }


def score_direct_trajectory_profile(
    *,
    trajectory_id: str,
    query_terms: set[str],
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    signal_profile = direct_trajectory_signal_profile(
        trajectory_id=trajectory_id,
        trajectory_metadata=trajectory_metadata,
        claims_by_trajectory=claims_by_trajectory,
    )
    texts = list(signal_profile["texts"])
    has_metadata_signals = bool(signal_profile["has_metadata_signals"])
    haystack = " ".join(texts).casefold()
    candidate_terms = {
        term
        for text in texts
        for term in extract_keywords(text)
        if term not in _ANSWER_TOKEN_STOPWORDS and term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
    }
    overlap = query_terms & candidate_terms
    lexical_score = float(len(overlap))
    if query_terms:
        lexical_score += 2.0 * (len(overlap) / len(query_terms))
    entity_overlap_score = 0.0
    matched_entity_terms: set[str] = set()
    for entity in query_entities:
        entity_key = normalize_entity_key(entity)
        entity_terms = {
            term for term in extract_keywords(entity) if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if entity_key and entity_key in candidate_terms:
            entity_overlap_score += 3.0
            matched_entity_terms.add(entity_key)
        elif entity_terms & candidate_terms:
            entity_overlap_score += 2.0
            matched_entity_terms.update(entity_terms & candidate_terms)
        elif entity and entity.casefold() in haystack:
            entity_overlap_score += 1.5
            matched_entity_terms.add(entity)
    facet_overlap_score = 0.0
    matched_facet_terms: set[str] = set()
    for facet in list(query_facets.get("tags") or []) + list(query_facets.get("values") or []):
        facet_terms = {
            term
            for term in extract_keywords(str(facet).replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if facet_terms and facet_terms <= candidate_terms:
            facet_overlap_score += 2.0
            matched_facet_terms.update(facet_terms)
        elif facet_terms & candidate_terms:
            facet_overlap_score += 1.0
            matched_facet_terms.update(facet_terms & candidate_terms)
    family_match_score = 0.0
    matched_family_terms: set[str] = set()
    item_family = str(query_shape.get("item_family") or "").strip()
    if item_family:
        family_terms = {
            term
            for term in extract_keywords(item_family.replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if family_terms & candidate_terms:
            family_match_score += 1.5
            matched_family_terms.update(family_terms & candidate_terms)
    temporal_score = 0.0
    if bool(query_shape.get("count_like")) and re.search(r"\b(?:count|number|times?|once|twice|\d+)\b", haystack):
        temporal_score += 0.75
    if bool(query_shape.get("list_like")) and any(
        key in haystack for key in ("display_items", "items", "list", "visited", "participated", "bought", "read")
    ):
        family_match_score += 0.5
    specific_values = (
        list(signal_profile.get("exact_terms") or [])
        + list(signal_profile.get("source_surface_terms") or [])
        + list(signal_profile.get("historical_terms") or [])
        + list(signal_profile.get("display_items") or [])
        + list(signal_profile.get("display_key_facts") or [])
    )
    query_object_overlap_terms = {
        term
        for value in specific_values
        for term in extract_keywords(value)
        if term in query_terms
    }
    query_object_overlap_score = (
        1.25 * (len(query_object_overlap_terms) / len(query_terms)) if query_terms else 0.0
    )
    generic_only_penalty = 0.0
    if overlap and not query_object_overlap_terms and not matched_entity_terms and not matched_facet_terms:
        generic_only_penalty = 0.25
    final_score = (
        lexical_score
        + entity_overlap_score
        + facet_overlap_score
        + family_match_score
        + temporal_score
        + query_object_overlap_score
        - generic_only_penalty
    )
    return {
        "trajectory_id": trajectory_id,
        "final_score": final_score,
        "lexical_score": lexical_score,
        "family_match_score": family_match_score,
        "query_object_overlap_score": query_object_overlap_score,
        "entity_overlap_score": entity_overlap_score,
        "facet_overlap_score": facet_overlap_score,
        "temporal_score": temporal_score,
        "generic_only_penalty": generic_only_penalty,
        "matched_query_terms": sorted(overlap),
        "matched_entity_terms": sorted(matched_entity_terms),
        "matched_facet_terms": sorted(matched_facet_terms),
        "matched_family_terms": sorted(matched_family_terms),
        "matched_exact_terms": _direct_term_matches(query_terms, signal_profile.get("exact_terms") or []),
        "matched_source_surface_terms": _direct_term_matches(
            query_terms, signal_profile.get("source_surface_terms") or []
        ),
        "matched_historical_terms": _direct_term_matches(query_terms, signal_profile.get("historical_terms") or []),
        "entity_keys": _compact_direct_terms(
            list(signal_profile.get("entity_keys") or []) + list(signal_profile.get("entity_mentions") or [])
        ),
        "facet_values": _compact_direct_terms(signal_profile.get("facet_values") or []),
        "item_family": str(query_shape.get("item_family") or ""),
        "exact_terms": _compact_direct_terms(signal_profile.get("exact_terms") or []),
        "source_surface_terms": _compact_direct_terms(signal_profile.get("source_surface_terms") or []),
        "historical_item_terms": _compact_direct_terms(signal_profile.get("historical_terms") or []),
        "has_metadata_signals": has_metadata_signals,
        "_texts": texts,
    }


def direct_compact_score_row(
    profile: dict[str, Any],
    *,
    rank: int,
    trajectory_refs: dict[str, set[str]] | None,
    trajectory_lengths: dict[str, int] | None,
) -> dict[str, Any]:
    trajectory_id = str(profile.get("trajectory_id") or "")
    refs = sorted((trajectory_refs or {}).get(trajectory_id, set()))
    texts = list(profile.get("_texts") or [])
    return {
        "rank": rank,
        "trajectory_id": trajectory_id,
        "item_id": trajectory_id,
        "item_type": "trajectory",
        "final_score": float(profile.get("final_score") or 0.0),
        "score": float(profile.get("final_score") or 0.0),
        "lexical_score": float(profile.get("lexical_score") or 0.0),
        "family_match_score": float(profile.get("family_match_score") or 0.0),
        "query_object_overlap_score": float(profile.get("query_object_overlap_score") or 0.0),
        "entity_overlap_score": float(profile.get("entity_overlap_score") or 0.0),
        "facet_overlap_score": float(profile.get("facet_overlap_score") or 0.0),
        "temporal_score": float(profile.get("temporal_score") or 0.0),
        "generic_only_penalty": float(profile.get("generic_only_penalty") or 0.0),
        "score_components": {
            "lexical": float(profile.get("lexical_score") or 0.0),
            "family": float(profile.get("family_match_score") or 0.0),
            "query_object_overlap": float(profile.get("query_object_overlap_score") or 0.0),
            "entity": float(profile.get("entity_overlap_score") or 0.0),
            "facet": float(profile.get("facet_overlap_score") or 0.0),
            "temporal": float(profile.get("temporal_score") or 0.0),
            "generic_only_penalty": float(profile.get("generic_only_penalty") or 0.0),
        },
        "matched_query_terms": list(profile.get("matched_query_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "matched_exact_terms": list(profile.get("matched_exact_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "matched_source_surface_terms": list(profile.get("matched_source_surface_terms") or [])[
            :DIRECT_TRAJECTORY_TERM_LIMIT
        ],
        "matched_historical_terms": list(profile.get("matched_historical_terms") or [])[
            :DIRECT_TRAJECTORY_TERM_LIMIT
        ],
        "entity_keys": list(profile.get("entity_keys") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "facet_values": list(profile.get("facet_values") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "item_family": profile.get("item_family") or "",
        "exact_terms": list(profile.get("exact_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "source_surface_terms": list(profile.get("source_surface_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "historical_item_terms": list(profile.get("historical_item_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "source_refs": refs,
        "linked_trajectory_ids": [trajectory_id],
        "snapshot_count_estimate": int((trajectory_lengths or {}).get(trajectory_id, 0) or 0),
        "source_ref_count_estimate": len(refs),
        "context_token_estimate": estimate_context_tokens_many(texts),
        "estimated_tokens": estimate_context_tokens_many(texts),
    }


def rank_direct_trajectories(
    *,
    sample_id: str,
    question: str,
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
    sample_to_trajectories: dict[str, set[str]],
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
    trajectory_refs: dict[str, set[str]] | None = None,
    trajectory_lengths: dict[str, int] | None = None,
    diagnostic_top_n: int = DIRECT_TRAJECTORY_DIAGNOSTIC_TOP_N,
) -> tuple[list[str], str, list[dict[str, Any]]]:
    candidates = sorted(str(trajectory_id) for trajectory_id in sample_to_trajectories.get(sample_id, set()))
    query_terms = direct_query_terms(
        question=question,
        query_entities=query_entities,
        query_facets=query_facets,
        query_shape=query_shape,
    )
    scored: list[tuple[float, str, dict[str, Any]]] = []
    metadata_signal_count = 0
    for trajectory_id in candidates:
        profile = score_direct_trajectory_profile(
            trajectory_id=trajectory_id,
            query_terms=query_terms,
            query_entities=query_entities,
            query_facets=query_facets,
            query_shape=query_shape,
            trajectory_metadata=trajectory_metadata,
            claims_by_trajectory=claims_by_trajectory,
        )
        metadata_signal_count += int(bool(profile.get("has_metadata_signals")))
        scored.append((float(profile.get("final_score") or 0.0), trajectory_id, profile))
    sorted_scored = sorted(scored, key=lambda item: (-item[0], item[1]))
    ranked = [trajectory_id for _, trajectory_id, _ in sorted_scored]
    compact_rows = [
        direct_compact_score_row(
            profile,
            rank=rank,
            trajectory_refs=trajectory_refs,
            trajectory_lengths=trajectory_lengths,
        )
        for rank, (_, _, profile) in enumerate(sorted_scored[: max(0, diagnostic_top_n)], start=1)
    ]
    scoring_mode = "metadata_lexical_v1" if metadata_signal_count else "fallback_text"
    return ranked, scoring_mode, compact_rows
