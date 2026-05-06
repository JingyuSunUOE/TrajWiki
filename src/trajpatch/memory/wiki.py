"""Sample-level wiki compilation for episodic trajectory navigation."""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from trajpatch.ids import slugify, wiki_page_id
from trajpatch.memory.historical import (
    build_trajectory_historical_evidence_card,
    render_trajectory_evidence_card,
    sanitize_historical_item_terms,
    specific_terms,
)
from trajpatch.memory.readability import clean_internal_memory_summary_text, clean_readable_values
from trajpatch.memory.trajectory_summaries import sanitize_summary_keyword_values, summary_keywords_v2
from trajpatch.prompts import load_prompt
from trajpatch.providers.base import EmbeddingProvider, LLMProvider
from trajpatch.storage.models import EmbeddingRecord, TrajectoryRecord, WikiPageRecord
from trajpatch.storage.repository import TrajWikiStore
from trajpatch.types import NormalizedMessage
from trajpatch.utils.text import collapse_whitespace, extract_keywords

_REQUIRED_WIKI_PAGE_HEADINGS = [
    "## Overview",
    "## Key Facts",
    "## Items / Counts",
    "## Linked Trajectories",
    "## Conflicts / Uncertainty",
]

_LINKED_TRAJECTORIES_HEADING = "## Linked Trajectories"
_WIKI_PLACEHOLDER_RE = re.compile(
    r"\b(?:not provided|none provided|no summary provided|unknown|n/a|not available|"
    r"no specific|no explicit|no items|no key facts|no information|none recorded|none)\b",
    flags=re.IGNORECASE,
)
_ROUTING_TEXT_INTERNAL_MARKERS = (
    "CARD ",
    "identity_summary=",
    "recent_update=",
    "source_surface_terms=",
    "historical_item_terms=",
    "source_anchors=",
    "Trajectory label:",
    "## Profile / Stable Facts",
    "## Item Sets / Named Entities",
    "## Relations / Temporal Updates",
    "None recorded",
)

_EVIDENCE_METADATA_FIELDS = {
    "exact_terms",
    "facet_values",
    "keywords",
    "display_items",
    "display_named_entities",
    "display_counts",
    "display_key_facts",
    "source_surface_terms_v1",
    "source_surface_raw_terms_v1",
    "wiki_source_event_object_terms_v1",
    "wiki_source_event_canonical_terms_v1",
    "wiki_source_temporal_relation_terms_v1",
    "wiki_historical_card_used",
    "wiki_historical_item_terms",
    "trajectory_drift_cluster_keys",
    "trajectory_drift_cluster_count_v1",
    "trajectory_source_anchors_v1",
    "trajectory_evidence_cards",
    "representative_trajectory_ids",
    "wiki_evidence_trajectory_count",
}


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = collapse_whitespace(str(value))
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _top_ranked_values(values: Iterable[str], *, limit: int) -> list[str]:
    counter: Counter[str] = Counter()
    first_seen: dict[str, str] = {}
    for value in values:
        text = collapse_whitespace(str(value))
        if not text:
            continue
        key = text.casefold()
        counter[key] += 1
        first_seen.setdefault(key, text)
    ranked = sorted(counter.items(), key=lambda item: (-item[1], first_seen[item[0]].casefold()))
    return [first_seen[key] for key, _ in ranked[:limit]]


@dataclass(slots=True)
class WikiPageSeed:
    seed_id: str
    page_type: str
    title: str
    slug: str
    trajectory_ids: list[str]
    entities: list[str]
    exact_terms: list[str]
    facet_values: list[str]
    keywords: list[str]
    representative_trajectory_ids: list[str]
    linked_slugs: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class WikiPageDraft:
    page_type: str
    title: str
    slug: str
    trajectory_ids: list[str]
    entities: list[str]
    linked_slugs: list[str]
    metadata: dict[str, Any]


class WikiCompiler:
    """Compile sample-level wiki pages from bounded trajectory seeds."""

    MAX_PLAN_SEEDS = 24
    MAX_PAGE_TRAJECTORIES = 6
    TARGET_PAGE_TRAJECTORIES = 4
    MIN_GROUPABLE_PAGE_TRAJECTORIES = 3
    MAX_REPRESENTATIVE_SUMMARIES = 4
    MAX_FIELD_VALUES = 8
    TOPIC_REDUNDANT_OVERLAP = 0.8
    BROAD_ENTITY_FACET_THRESHOLD = 24
    _GENERIC_PAGE_FAMILY_TERMS = {
        "about",
        "answer",
        "caroline",
        "conversation",
        "detail",
        "details",
        "discuss",
        "discussed",
        "evidence",
        "experience",
        "experiences",
        "family",
        "general",
        "great",
        "information",
        "item",
        "items",
        "melanie",
        "memory",
        "people",
        "person",
        "support",
        "thing",
        "things",
        "topic",
        "update",
    }
    _FACET_FAMILY_ALIASES = {
        "activity_location": "activities",
        "activities": "activities",
        "activity": "activities",
        "book": "books",
        "books": "books",
        "city": "places",
        "country": "places",
        "event_type": "events",
        "event": "events",
        "home_country": "places",
        "instrument": "instruments",
        "instruments": "instruments",
        "painted_object": "paintings",
        "painting": "paintings",
        "place": "places",
        "places": "places",
        "recipe": "recipes",
        "recipes": "recipes",
        "relationship_status": "relationship",
        "research_topic": "research",
        "screenplay": "writing",
        "script": "writing",
        "state": "places",
        "writing": "writing",
    }
    _RESCUE_UMBRELLA_FAMILY_TERMS = {
        "visual_art": {
            "art",
            "artwork",
            "canvas",
            "flower",
            "flowers",
            "horse",
            "image",
            "paint",
            "painted",
            "painting",
            "photo",
            "picture",
            "portrait",
            "sunrise",
            "sunset",
        },
        "research_and_planning": {
            "adoption",
            "agencies",
            "agency",
            "option",
            "options",
            "plan",
            "planning",
            "research",
            "researched",
            "study",
            "studied",
        },
        "books_and_reading": {
            "book",
            "books",
            "cover",
            "novel",
            "read",
            "reading",
            "story",
            "title",
        },
        "recipes_and_desserts": {
            "cake",
            "chocolate",
            "coconut",
            "cream",
            "dairy",
            "dessert",
            "desserts",
            "ice",
            "milk",
            "recipe",
            "recipes",
            "swirl",
            "vanilla",
        },
        "activities_and_hobbies": {
            "activity",
            "activities",
            "beach",
            "camping",
            "hike",
            "hiking",
            "hobby",
            "museum",
            "pottery",
            "running",
            "swimming",
            "trail",
        },
        "writing_and_feedback": {
            "contest",
            "feedback",
            "fiction",
            "rejected",
            "rejection",
            "screenplay",
            "screenplays",
            "script",
            "scripts",
            "story",
            "writing",
        },
        "pets": {
            "cat",
            "cats",
            "dog",
            "dogs",
            "pet",
            "pets",
            "tortoise",
            "tortoises",
            "turtle",
            "turtles",
        },
        "places_and_moves": {
            "area",
            "canada",
            "city",
            "country",
            "county",
            "mexico",
            "move",
            "moved",
            "place",
            "places",
            "sweden",
            "visited",
            "west",
        },
        "relationship_and_identity": {
            "breakup",
            "dating",
            "identity",
            "married",
            "parent",
            "relationship",
            "single",
            "status",
            "transgender",
        },
        "family_and_people": {
            "aunt",
            "children",
            "died",
            "family",
            "father",
            "kid",
            "kids",
            "mother",
            "passed",
            "uncle",
        },
    }
    _RESCUE_NOISE_FAMILY_TERMS = {
        "amazing",
        "exchange",
        "experience",
        "facts",
        "general",
        "great",
        "information",
        "memory",
        "support",
        "that",
        "thing",
        "topic",
    }
    _LOW_QUALITY_SINGLETON_TERMS = {
        *_GENERIC_PAGE_FAMILY_TERMS,
        *_RESCUE_NOISE_FAMILY_TERMS,
        "at",
        "cool",
        "evidence",
        "look",
        "nice",
        "one",
        "plus",
        "that",
        "this",
        "two",
        "wow",
        "yea",
        "yeah",
    }

    def __init__(
        self,
        store: TrajWikiStore,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        *,
        trace: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.trace = trace

    def _trace(self, message: str) -> None:
        if self.trace is not None:
            self.trace(message)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embedding_provider, "embed_documents"):
            return self.embedding_provider.embed_documents(texts)
        return self.embedding_provider.embed(texts)

    def _document_embedding_strategy(self) -> str:
        if hasattr(self.embedding_provider, "document_embedding_strategy"):
            return str(self.embedding_provider.document_embedding_strategy())
        return "shared_embed"

    @staticmethod
    def _trajectory_summary_text(trajectory: TrajectoryRecord) -> str:
        metadata = dict(trajectory.metadata_json or {})
        return collapse_whitespace(str(metadata.get("retrieval_summary_text") or ""))

    def _trajectory_routing_summary_text(self, trajectory: TrajectoryRecord) -> str:
        return clean_internal_memory_summary_text(self._trajectory_summary_text(trajectory))

    @staticmethod
    def _routing_text_internal_marker_hits(text: str) -> list[str]:
        folded = str(text or "").casefold()
        return [
            marker
            for marker in _ROUTING_TEXT_INTERNAL_MARKERS
            if marker.casefold() in folded
        ]

    @staticmethod
    def _clean_routing_value(value: object, *, max_parts: int = 2) -> str:
        return clean_internal_memory_summary_text(value, max_parts=max_parts)

    def _routing_card_values(
        self,
        card: dict[str, Any],
        field_name: str,
        *,
        limit: int,
        max_parts: int = 2,
    ) -> list[str]:
        values = [
            self._clean_routing_value(value, max_parts=max_parts)
            for value in list(card.get(field_name) or [])[:limit]
        ]
        return _dedupe_preserve(value for value in values if value)

    def _trajectory_routing_evidence_text(self, card: dict[str, Any]) -> str:
        trajectory_id = collapse_whitespace(str(card.get("trajectory_id") or "")).strip()
        identity_summary = self._clean_routing_value(card.get("identity_summary") or "", max_parts=3)
        recent_update = self._clean_routing_value(card.get("recent_update") or "", max_parts=3)
        source_surfaces = self._routing_card_values(card, "source_surface_terms", limit=10)
        historical_terms = self._routing_card_values(card, "historical_item_terms", limit=12)
        facet_values = self._routing_card_values(card, "facet_values", limit=8)
        entities = _dedupe_preserve(str(value) for value in list(card.get("entity_mentions") or [])[:8])
        source_refs: list[str] = []
        for anchor in list(card.get("source_anchors") or [])[:4]:
            if not isinstance(anchor, dict):
                continue
            source_ref = collapse_whitespace(str(anchor.get("source_ref") or "")).strip()
            preview = self._clean_routing_value(anchor.get("text") or "", max_parts=1)
            if len(preview) > 140:
                preview = preview[:137].rstrip() + "..."
            if source_ref and preview:
                source_refs.append(f"{source_ref}: {preview}")
            elif source_ref:
                source_refs.append(source_ref)

        lines: list[str] = []
        header = f"trajectory {trajectory_id}" if trajectory_id else "trajectory"
        summary_values = [value for value in [identity_summary, recent_update] if value]
        lines.append(f"{header}: {' - '.join(summary_values[:2])}" if summary_values else header)
        if source_surfaces:
            lines.append(f"surfaces: {', '.join(source_surfaces)}")
        if historical_terms:
            lines.append(f"historical terms: {', '.join(historical_terms)}")
        if facet_values:
            lines.append(f"facets: {', '.join(facet_values)}")
        if entities:
            lines.append(f"entities: {', '.join(entities)}")
        if source_refs:
            lines.append(f"sources: {'; '.join(source_refs)}")
        text = "\n".join(lines)
        return clean_internal_memory_summary_text(text, max_parts=None)

    @staticmethod
    def _trajectory_field_list(trajectory: TrajectoryRecord, field_name: str) -> list[str]:
        metadata = dict(trajectory.metadata_json or {})
        if field_name == "exact_terms":
            values = metadata.get("exact_terms_v2") or metadata.get("exact_terms") or []
        else:
            values = metadata.get(field_name) or []
        return [
            str(value).strip()
            for value in list(values)
            if str(value).strip()
        ]

    def _trajectory_evidence_card(self, trajectory: TrajectoryRecord) -> dict[str, Any]:
        metadata = dict(trajectory.metadata_json or {})
        card = metadata.get("trajectory_historical_evidence_card_v1")
        if isinstance(card, dict):
            return dict(card)
        return build_trajectory_historical_evidence_card(
            trajectory_id=trajectory.id,
            trajectory_label=str(trajectory.label or ""),
            retrieval_summary_text=self._trajectory_summary_text(trajectory),
            latest_semantic_text=str(metadata.get("latest_semantic_text") or ""),
            metadata=metadata,
        )

    def _trajectory_historical_terms(self, trajectory: TrajectoryRecord) -> list[str]:
        metadata = dict(trajectory.metadata_json or {})
        stored_v2 = sanitize_historical_item_terms(
            list(metadata.get("trajectory_historical_item_terms_v2") or []),
            limit=24,
        )
        if stored_v2:
            return stored_v2
        card = self._trajectory_evidence_card(trajectory)
        return sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24)

    def _trajectory_keyword_list(self, trajectory: TrajectoryRecord) -> list[str]:
        metadata = dict(trajectory.metadata_json or {})
        summary_keywords = sanitize_summary_keyword_values(
            list(metadata.get("retrieval_summary_keywords_v2") or []),
            limit=32,
        ) or sanitize_summary_keyword_values(
            list(metadata.get("retrieval_summary_keywords") or []),
            limit=32,
        )
        historical_terms = self._trajectory_historical_terms(trajectory)
        if summary_keywords:
            return _dedupe_preserve([*historical_terms[:6], *summary_keywords])
        latest_keywords = self._trajectory_field_list(trajectory, "latest_keywords")
        if latest_keywords:
            return _dedupe_preserve([*historical_terms[:6], *sanitize_summary_keyword_values(latest_keywords, limit=32)])
        return _dedupe_preserve(
            [
                *historical_terms[:6],
                *summary_keywords_v2(self._trajectory_routing_summary_text(trajectory), metadata),
            ]
        )

    @classmethod
    def _is_low_salience_noise_trajectory(cls, trajectory: TrajectoryRecord) -> bool:
        metadata = dict(trajectory.metadata_json or {})
        if not metadata.get("low_salience_memory_v1"):
            return False
        signal_fields = (
            "exact_terms",
            "exact_terms_v2",
            "facet_values",
            "entity_mentions",
            "display_items",
            "display_named_entities",
            "display_counts",
            "display_key_facts",
        )
        return not any(cls._trajectory_field_list(trajectory, field_name) for field_name in signal_fields)

    def _trajectory_rows_by_id(self, trajectories: list[TrajectoryRecord]) -> dict[str, TrajectoryRecord]:
        return {trajectory.id: trajectory for trajectory in trajectories}

    def _summary_embedding_rows_by_id(self, trajectories: list[TrajectoryRecord]) -> dict[str, EmbeddingRecord]:
        return self.store.fetch_embeddings_by_owner_ids([trajectory.id for trajectory in trajectories], "trajectory_summary")

    def _choose_representative_trajectory_ids(
        self,
        trajectory_rows: list[TrajectoryRecord],
        *,
        limit: int,
    ) -> list[str]:
        if len(trajectory_rows) <= limit:
            return [trajectory.id for trajectory in trajectory_rows]

        def richness(trajectory: TrajectoryRecord) -> tuple[int, int, int, str]:
            summary_text = self._trajectory_routing_summary_text(trajectory)
            return (
                len(self._trajectory_field_list(trajectory, "exact_terms"))
                + len(self._trajectory_field_list(trajectory, "facet_values"))
                + len(self._trajectory_historical_terms(trajectory)),
                len(self._trajectory_keyword_list(trajectory)),
                len(summary_text),
                trajectory.id,
            )

        ranked = sorted(trajectory_rows, key=richness, reverse=True)
        return [trajectory.id for trajectory in ranked[:limit]]

    def _trajectory_evidence_metadata(
        self,
        rows: list[TrajectoryRecord],
    ) -> dict[str, Any]:
        source_surface_raw_terms = _top_ranked_values(
            (
                term
                for row in rows
                for term in self._trajectory_field_list(row, "source_surface_raw_terms_v1")
            ),
            limit=self.MAX_FIELD_VALUES,
        )
        source_surface_terms = _top_ranked_values(
            (term for row in rows for term in self._trajectory_field_list(row, "source_surface_terms_v1")),
            limit=self.MAX_FIELD_VALUES,
        )
        source_event_object_terms = _top_ranked_values(
            (term for row in rows for term in self._trajectory_field_list(row, "source_event_object_terms_v1")),
            limit=self.MAX_FIELD_VALUES,
        )
        source_event_canonical_terms = _top_ranked_values(
            (term for row in rows for term in self._trajectory_field_list(row, "source_event_canonical_terms_v1")),
            limit=self.MAX_FIELD_VALUES,
        )
        source_temporal_relation_terms = _top_ranked_values(
            (term for row in rows for term in self._trajectory_field_list(row, "source_temporal_relation_terms_v1")),
            limit=self.MAX_FIELD_VALUES,
        )
        source_event_values = _dedupe_preserve([*source_event_canonical_terms, *source_event_object_terms])
        source_surface_values = _dedupe_preserve([*source_event_values, *source_surface_raw_terms, *source_surface_terms])
        exact_terms = _top_ranked_values(
            [
                *source_surface_values,
                *(term for row in rows for term in self._trajectory_field_list(row, "exact_terms")),
            ],
            limit=self.MAX_FIELD_VALUES,
        )
        facet_values = _top_ranked_values(
            (value for row in rows for value in self._trajectory_field_list(row, "facet_values")),
            limit=self.MAX_FIELD_VALUES,
        )
        keywords = _top_ranked_values(
            (keyword for row in rows for keyword in self._trajectory_keyword_list(row)),
            limit=self.MAX_FIELD_VALUES,
        )
        display_items = _top_ranked_values(
            [
                *source_surface_values,
                *(value for row in rows for value in self._trajectory_field_list(row, "display_items")),
            ],
            limit=self.MAX_FIELD_VALUES,
        )
        display_named_entities = _top_ranked_values(
            (value for row in rows for value in self._trajectory_field_list(row, "display_named_entities")),
            limit=self.MAX_FIELD_VALUES,
        )
        display_counts = _top_ranked_values(
            (value for row in rows for value in self._trajectory_field_list(row, "display_counts")),
            limit=self.MAX_FIELD_VALUES,
        )
        display_key_facts = _top_ranked_values(
            (value for row in rows for value in self._trajectory_field_list(row, "display_key_facts")),
            limit=self.MAX_FIELD_VALUES,
        )
        evidence_cards = [self._trajectory_evidence_card(row) for row in rows]
        historical_item_terms = _top_ranked_values(
            (
                value
                for card in evidence_cards
                for value in [
                    *list(card.get("source_surface_terms") or []),
                    *sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24),
                ]
            ),
            limit=max(self.MAX_FIELD_VALUES, 12),
        )
        drift_cluster_keys = _top_ranked_values(
            (
                value
                for card in evidence_cards
                for value in list(card.get("drift_cluster_keys") or [])
            ),
            limit=max(self.MAX_FIELD_VALUES, 12),
        )
        source_anchors = [
            anchor
            for card in evidence_cards
            for anchor in list(card.get("source_anchors") or [])
            if isinstance(anchor, dict)
        ][:12]
        representative_ids = self._choose_representative_trajectory_ids(
            rows,
            limit=min(self.MAX_REPRESENTATIVE_SUMMARIES, len(rows)),
        )
        representative_cards = [
            card
            for card in evidence_cards
            if str(card.get("trajectory_id") or "") in set(representative_ids)
        ][: self.MAX_REPRESENTATIVE_SUMMARIES]
        entity_mentions = _top_ranked_values(
            (value for row in rows for value in self._trajectory_field_list(row, "entity_mentions")),
            limit=self.MAX_FIELD_VALUES,
        )
        return {
            "exact_terms": exact_terms,
            "facet_values": facet_values,
            "keywords": _dedupe_preserve([*historical_item_terms[:6], *keywords]),
            "display_items": display_items,
            "display_named_entities": display_named_entities,
            "display_counts": display_counts,
            "display_key_facts": display_key_facts,
            "source_surface_terms_v1": source_surface_terms,
            "source_surface_raw_terms_v1": source_surface_raw_terms,
            "wiki_source_event_object_terms_v1": source_event_object_terms,
            "wiki_source_event_canonical_terms_v1": source_event_canonical_terms,
            "wiki_source_temporal_relation_terms_v1": source_temporal_relation_terms,
            "wiki_historical_card_used": bool(evidence_cards),
            "wiki_historical_item_terms": historical_item_terms,
            "trajectory_drift_cluster_keys": drift_cluster_keys,
            "trajectory_drift_cluster_count_v1": len(drift_cluster_keys),
            "trajectory_source_anchors_v1": source_anchors,
            "trajectory_evidence_cards": representative_cards,
            "representative_trajectory_ids": representative_ids,
            "entity_mentions": entity_mentions,
            "wiki_evidence_trajectory_count": len(rows),
        }

    @staticmethod
    def _metadata_has_evidence(metadata: dict[str, Any]) -> bool:
        evidence_fields = (
            "representative_trajectory_ids",
            "trajectory_evidence_cards",
            "exact_terms",
            "facet_values",
            "display_items",
            "display_named_entities",
            "display_counts",
            "display_key_facts",
            "source_surface_terms_v1",
            "source_surface_raw_terms_v1",
            "wiki_source_event_object_terms_v1",
            "wiki_source_event_canonical_terms_v1",
            "wiki_source_temporal_relation_terms_v1",
            "wiki_historical_item_terms",
        )
        return any(metadata.get(field_name) for field_name in evidence_fields)

    def _merge_evidence_metadata(
        self,
        metadata: dict[str, Any],
        rows: list[TrajectoryRecord],
        *,
        match_source: str,
        force_synthesized: bool = False,
    ) -> dict[str, Any]:
        merged = dict(metadata)
        evidence = self._trajectory_evidence_metadata(rows)
        synthesized = bool(force_synthesized or merged.get("wiki_evidence_metadata_synthesized"))
        for key, value in evidence.items():
            if key == "entity_mentions":
                continue
            if force_synthesized and key in _EVIDENCE_METADATA_FIELDS:
                merged[key] = value
                continue
            if not merged.get(key) and value:
                merged[key] = value
                synthesized = True
        merged.setdefault("wiki_seed_match_source", match_source)
        merged["wiki_evidence_trajectory_count"] = len(rows)
        merged["wiki_evidence_source_trajectory_ids"] = [row.id for row in rows]
        merged["wiki_evidence_metadata_synthesized"] = synthesized
        return merged

    def _build_seed(
        self,
        *,
        seed_id: str,
        page_type: str,
        title: str,
        slug: str,
        trajectory_ids: list[str],
        entities: list[str],
        trajectories_by_id: dict[str, TrajectoryRecord],
        linked_slugs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WikiPageSeed:
        rows = [trajectories_by_id[trajectory_id] for trajectory_id in trajectory_ids if trajectory_id in trajectories_by_id]
        evidence = self._trajectory_evidence_metadata(rows)
        seed_metadata = {
            **dict(metadata or {}),
            "display_items": evidence["display_items"],
            "display_named_entities": evidence["display_named_entities"],
            "display_counts": evidence["display_counts"],
            "display_key_facts": evidence["display_key_facts"],
            "source_surface_raw_terms_v1": evidence["source_surface_raw_terms_v1"],
            "source_surface_terms_v1": evidence["source_surface_terms_v1"],
            "wiki_source_event_object_terms_v1": evidence["wiki_source_event_object_terms_v1"],
            "wiki_source_event_canonical_terms_v1": evidence["wiki_source_event_canonical_terms_v1"],
            "wiki_source_temporal_relation_terms_v1": evidence["wiki_source_temporal_relation_terms_v1"],
            "wiki_historical_card_used": evidence["wiki_historical_card_used"],
            "wiki_historical_item_terms": evidence["wiki_historical_item_terms"],
            "trajectory_drift_cluster_keys": evidence["trajectory_drift_cluster_keys"],
            "trajectory_drift_cluster_count_v1": evidence["trajectory_drift_cluster_count_v1"],
            "trajectory_source_anchors_v1": evidence["trajectory_source_anchors_v1"],
            "trajectory_evidence_cards": evidence["trajectory_evidence_cards"],
            "wiki_evidence_trajectory_count": evidence["wiki_evidence_trajectory_count"],
            "wiki_evidence_source_trajectory_ids": list(dict.fromkeys(trajectory_ids)),
            "wiki_evidence_metadata_synthesized": False,
        }
        return WikiPageSeed(
            seed_id=seed_id,
            page_type=page_type,
            title=title,
            slug=slugify(slug),
            trajectory_ids=list(dict.fromkeys(trajectory_ids)),
            entities=_dedupe_preserve(entities),
            exact_terms=list(evidence["exact_terms"]),
            facet_values=list(evidence["facet_values"]),
            keywords=list(evidence["keywords"]),
            representative_trajectory_ids=list(evidence["representative_trajectory_ids"]),
            linked_slugs=list(dict.fromkeys(linked_slugs or [])),
            metadata=seed_metadata,
        )

    @staticmethod
    def _is_inventory_trajectory(metadata: dict[str, Any]) -> bool:
        exact_terms = [str(value).strip() for value in list(metadata.get("exact_terms") or []) if str(value).strip()]
        historical_terms = sanitize_historical_item_terms(
            list(
                metadata.get("trajectory_historical_item_terms_v2")
                or metadata.get("trajectory_historical_item_terms_v1")
                or []
            ),
            limit=24,
        )
        facet_tags = {str(value).strip().casefold() for value in list(metadata.get("facet_tags") or []) if str(value).strip()}
        display_items = [str(value).strip() for value in list(metadata.get("display_items") or []) if str(value).strip()]
        if len(_dedupe_preserve([*exact_terms, *historical_terms, *display_items])) >= 2:
            return True
        inventoryish = {
            "activities",
            "books",
            "recipes",
            "tournaments",
            "dogs",
            "instruments",
            "symbols",
            "places",
            "activity_location",
            "research_topic",
        }
        return bool(facet_tags & inventoryish)

    @classmethod
    def _normalize_page_family(cls, raw_family: str) -> str:
        family = slugify(raw_family).replace("-", "_")
        if not family:
            return "evidence"
        return cls._FACET_FAMILY_ALIASES.get(family, family)

    @classmethod
    def _family_from_terms(cls, values: Iterable[str]) -> str:
        keyword_counts: Counter[str] = Counter()
        first_seen: dict[str, str] = {}
        for value in values:
            for keyword in sorted(extract_keywords(str(value))):
                key = keyword.casefold()
                if key in cls._GENERIC_PAGE_FAMILY_TERMS or len(key) <= 2:
                    continue
                keyword_counts[key] += 1
                first_seen.setdefault(key, key)
        if not keyword_counts:
            return "evidence"
        ranked = sorted(keyword_counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
        return cls._normalize_page_family(ranked[0][0])

    @classmethod
    def _page_type_for_family(cls, family: str, *, has_display_values: bool, inventory_like: bool) -> str:
        if family in {
            "activities",
            "activities_and_hobbies",
            "books",
            "books_and_reading",
            "counts",
            "events",
            "family_and_people",
            "instruments",
            "paintings",
            "pets",
            "places",
            "places_and_moves",
            "recipes",
            "recipes_and_desserts",
            "research_and_planning",
            "visual_art",
            "writing",
            "writing_and_feedback",
        }:
            return "inventory"
        if has_display_values or inventory_like:
            return "inventory"
        return "topic"

    @classmethod
    def _rescue_umbrella_family_from_terms(cls, values: Iterable[object], *, fallback_family: str) -> str:
        keyword_counts: Counter[str] = Counter()
        for value in values:
            for keyword in extract_keywords(str(value or "")):
                key = keyword.casefold()
                if (
                    not key
                    or len(key) <= 2
                    or key in cls._GENERIC_PAGE_FAMILY_TERMS
                    or key in cls._RESCUE_NOISE_FAMILY_TERMS
                ):
                    continue
                keyword_counts[key] += 1
        if keyword_counts:
            family_scores: dict[str, int] = {}
            for family, terms in cls._RESCUE_UMBRELLA_FAMILY_TERMS.items():
                score = sum(keyword_counts.get(term, 0) for term in terms)
                if score:
                    family_scores[family] = score
            if family_scores:
                return max(family_scores.items(), key=lambda item: (item[1], item[0]))[0]
        normalized = cls._normalize_page_family(fallback_family)
        if normalized in cls._RESCUE_NOISE_FAMILY_TERMS or normalized == "evidence":
            return "general_events"
        return normalized

    @classmethod
    def _medium_granularity_chunks(cls, trajectory_ids: list[str]) -> list[list[str]]:
        ids = list(dict.fromkeys(trajectory_ids))
        if len(ids) <= cls.MAX_PAGE_TRAJECTORIES:
            return [ids] if ids else []
        chunks = [
            ids[index : index + cls.TARGET_PAGE_TRAJECTORIES]
            for index in range(0, len(ids), cls.TARGET_PAGE_TRAJECTORIES)
        ]
        if (
            len(chunks) >= 2
            and 0 < len(chunks[-1]) < cls.MIN_GROUPABLE_PAGE_TRAJECTORIES
            and len(chunks[-2]) + len(chunks[-1]) <= cls.MAX_PAGE_TRAJECTORIES
        ):
            chunks[-2].extend(chunks[-1])
            chunks.pop()
        return chunks

    @classmethod
    def _granularity_metadata(
        cls,
        trajectory_count: int,
        *,
        exception_reason: str | None = None,
        descriptor: object = "",
        family: object = "",
        group: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        singleton = trajectory_count == 1
        metadata: dict[str, Any] = {
            "wiki_page_granularity_target": "medium",
            "wiki_page_group_size": trajectory_count,
            "wiki_singleton_exception": singleton,
            **cls._singleton_policy_for_group(
                trajectory_count=trajectory_count,
                descriptor=descriptor,
                family=family,
                group=group,
            ),
        }
        if singleton:
            metadata["singleton_exception_reason"] = exception_reason or "no_compatible_groupable_neighbors"
        elif trajectory_count < cls.MIN_GROUPABLE_PAGE_TRAJECTORIES:
            metadata["wiki_below_target_exception"] = True
            metadata["singleton_exception_reason"] = exception_reason or "remaining_group_below_minimum"
        return metadata

    @classmethod
    def _descriptor_quality_score(cls, value: object) -> int:
        text = collapse_whitespace(str(value or ""))
        if not text:
            return 0
        tokens = [token.casefold() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", text)]
        if not tokens:
            return 0
        informative = [
            token
            for token in tokens
            if token not in cls._LOW_QUALITY_SINGLETON_TERMS
            and len(token) > 2
            and not token.isdigit()
        ]
        if not informative:
            return 0
        score = len(informative)
        if any(char.isupper() for char in text[1:]) or "'" in text or "’" in text or '"' in text:
            score += 1
        if len(informative) >= 2:
            score += 1
        return score

    @classmethod
    def _descriptor_is_low_quality(cls, value: object) -> bool:
        return cls._descriptor_quality_score(value) <= 0

    @classmethod
    def _group_descriptor_metadata(cls, group: dict[str, Any], *, fallback: str) -> dict[str, Any]:
        descriptors = clean_readable_values(
            [str(value) for value in list(group.get("descriptors") or []) if str(value).strip()],
            allow_single_word=True,
            limit=3,
        )
        specific_values = clean_readable_values(
            [str(value) for value in list(group.get("specific_values") or []) if str(value).strip()],
            allow_single_word=True,
            limit=8,
        )
        candidates = [*descriptors, *specific_values]
        for candidate in candidates:
            score = cls._descriptor_quality_score(candidate)
            if score > 0:
                return {
                    "descriptor": collapse_whitespace(candidate)[:80],
                    "wiki_descriptor_quality_score": score,
                    "wiki_descriptor_rewritten": bool(candidate not in descriptors[:1]),
                    "wiki_descriptor_rewrite_reason": (
                        "specific_signal_fallback" if candidate not in descriptors[:1] else None
                    ),
                }
        entity = next((str(value).strip() for value in list(group.get("entities") or []) if str(value).strip()), "")
        family_label = collapse_whitespace(str(fallback or "evidence").replace("_", " ").title())
        descriptor = collapse_whitespace(f"{entity} - {family_label}" if entity else family_label)[:80]
        return {
            "descriptor": descriptor,
            "wiki_descriptor_quality_score": cls._descriptor_quality_score(descriptor),
            "wiki_descriptor_rewritten": True,
            "wiki_descriptor_rewrite_reason": "low_quality_descriptor_fallback",
        }

    @classmethod
    def _group_readable_descriptor(cls, group: dict[str, Any], *, fallback: str) -> str:
        return str(cls._group_descriptor_metadata(group, fallback=fallback)["descriptor"])

    @classmethod
    def _singleton_policy_for_group(
        cls,
        *,
        trajectory_count: int,
        descriptor: object,
        family: object,
        group: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        descriptor_score = cls._descriptor_quality_score(descriptor)
        specific_score = 0
        if group:
            specific_score = max(
                [cls._descriptor_quality_score(value) for value in list(group.get("specific_values") or [])] or [0]
            )
        quality_score = max(descriptor_score, specific_score)
        if trajectory_count != 1:
            return {
                "wiki_singleton_policy": "not_singleton",
                "wiki_singleton_quality_score": quality_score,
                "wiki_singleton_allowed": True,
                "wiki_singleton_policy_reason": "not_singleton",
            }
        if quality_score > 0:
            return {
                "wiki_singleton_policy": "allowed_isolated_specific",
                "wiki_singleton_quality_score": quality_score,
                "wiki_singleton_allowed": True,
                "wiki_singleton_policy_reason": "specific_source_surface_or_descriptor",
            }
        return {
            "wiki_singleton_policy": "merge_required_low_quality",
            "wiki_singleton_quality_score": quality_score,
            "wiki_singleton_allowed": False,
            "wiki_singleton_policy_reason": "descriptor_and_family_are_low_quality",
        }

    def _entity_facet_group_profile(
        self,
        trajectory: TrajectoryRecord,
    ) -> tuple[str, str, str]:
        metadata = dict(trajectory.metadata_json or {})
        facet_values = self._trajectory_field_list(trajectory, "facet_values")
        if facet_values:
            cleaned = clean_readable_values([facet_values[0]], limit=1)
            descriptor = cleaned[0] if cleaned else str(facet_values[0])
            family = self._normalize_page_family(descriptor.split("=", 1)[0].strip().casefold() or "facet")
            page_type = self._page_type_for_family(
                family,
                has_display_values=bool(self._trajectory_field_list(trajectory, "display_items")),
                inventory_like=self._is_inventory_trajectory(metadata),
            )
            return page_type, family, descriptor
        card = self._trajectory_evidence_card(trajectory)
        display_values = _dedupe_preserve(
            [
                *self._trajectory_field_list(trajectory, "display_items"),
                *self._trajectory_field_list(trajectory, "display_counts"),
                *self._trajectory_field_list(trajectory, "display_key_facts"),
                *list(card.get("display_items") or []),
                *list(card.get("display_counts") or []),
            ]
        )
        historical_terms = specific_terms(
            [
                *display_values,
                *sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24),
                *list(card.get("drift_cluster_keys") or []),
                *self._trajectory_keyword_list(trajectory),
                trajectory.label,
            ],
            limit=8,
        )
        descriptor = historical_terms[0] if historical_terms else str(trajectory.label or trajectory.id)
        family = self._family_from_terms([*historical_terms, *display_values, trajectory.label])
        page_type = self._page_type_for_family(
            family,
            has_display_values=bool(display_values),
            inventory_like=self._is_inventory_trajectory(metadata),
        )
        return page_type, family, descriptor

    def _build_entity_facet_seeds(
        self,
        *,
        entity: str,
        trajectory_ids: list[str],
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> list[WikiPageSeed]:
        entity_key = slugify(entity) or "entity"
        grouped: dict[str, dict[str, Any]] = {}
        for trajectory_id in trajectory_ids:
            trajectory = trajectories_by_id.get(trajectory_id)
            if trajectory is None:
                continue
            page_type, family, descriptor = self._entity_facet_group_profile(trajectory)
            group_key = f"entity_facet:{entity_key}:{slugify(family) or 'facet'}"
            group = grouped.setdefault(
                group_key,
                {
                    "page_type": page_type,
                    "family": family,
                    "descriptors": [],
                    "specific_values": [],
                    "trajectory_ids": [],
                },
            )
            group["descriptors"] = [*list(group["descriptors"]), descriptor]
            group["specific_values"] = [*list(group["specific_values"]), descriptor]
            group["trajectory_ids"].append(trajectory_id)
        seeds: list[WikiPageSeed] = []
        for group_key, group in sorted(grouped.items()):
            page_type = str(group["page_type"])
            family = collapse_whitespace(str(group["family"]))
            descriptor_metadata = self._group_descriptor_metadata(group, fallback=family)
            readable_descriptor = str(descriptor_metadata["descriptor"])
            policy_descriptor = (
                ""
                if descriptor_metadata.get("wiki_descriptor_rewrite_reason") == "low_quality_descriptor_fallback"
                else readable_descriptor
            )
            chunks = self._medium_granularity_chunks(list(dict.fromkeys(group["trajectory_ids"])))
            for chunk_index, chunk_ids in enumerate(chunks, start=1):
                chunk_suffix = f" Part {chunk_index}" if len(chunks) > 1 else ""
                family_label = family.replace("_", " ").title()
                title = f"{entity} - {family_label}: {readable_descriptor}{chunk_suffix}"
                seed = self._build_seed(
                    seed_id=f"{group_key}::{chunk_index}",
                    page_type=page_type,
                    title=title,
                    slug=f"{entity_key}-{family}-{readable_descriptor}-{chunk_index if len(chunks) > 1 else ''}",
                    trajectory_ids=chunk_ids,
                    entities=[entity],
                    trajectories_by_id=trajectories_by_id,
                    metadata={
                        "seed_type": "entity_facet",
                        "routing_priority": "high",
                        "entity_facet_group_key": group_key,
                        "entity_facet_group_descriptor": readable_descriptor,
                        "entity_facet_source_entity": entity,
                        "entity_facet_trajectory_count": len(chunk_ids),
                        "entity_facet_split_from_broad_page": True,
                        **descriptor_metadata,
                        **self._granularity_metadata(
                            len(chunk_ids),
                            descriptor=policy_descriptor,
                            family=family,
                            group=group,
                        ),
                    },
                )
                seeds.append(seed)
        return seeds

    def _build_candidate_seeds(
        self,
        sample_id: str,
        trajectories: list[TrajectoryRecord],
    ) -> list[WikiPageSeed]:
        trajectories_by_id = self._trajectory_rows_by_id(trajectories)
        all_ids = [trajectory.id for trajectory in trajectories]
        entity_to_trajectory_ids: dict[str, list[str]] = defaultdict(list)
        for trajectory in trajectories:
            for entity in self._trajectory_field_list(trajectory, "entity_mentions"):
                entity_to_trajectory_ids[entity].append(trajectory.id)

        seeds: list[WikiPageSeed] = [
            self._build_seed(
                seed_id="index",
                page_type="index",
                title=f"{sample_id} wiki index",
                slug="index",
                trajectory_ids=all_ids,
                entities=sorted(entity_to_trajectory_ids),
                trajectories_by_id=trajectories_by_id,
                metadata={
                    "seed_type": "index",
                    "routing_priority": "low",
                },
            )
        ]
        raw_entity_count = 0
        raw_entity_facet_count = 0
        raw_inventory_count = 0
        raw_topic_count = 0

        for entity, trajectory_ids in sorted(entity_to_trajectory_ids.items()):
            unique_ids = list(dict.fromkeys(trajectory_ids))
            if len(unique_ids) < 2:
                continue
            raw_entity_count += 1
            broad_entity = len(unique_ids) > self.BROAD_ENTITY_FACET_THRESHOLD
            seeds.append(
                self._build_seed(
                    seed_id=f"entity::{slugify(entity)}",
                    page_type="entity",
                    title=f"{entity} overview",
                    slug=f"entity-{entity}",
                    trajectory_ids=unique_ids,
                    entities=[entity],
                    trajectories_by_id=trajectories_by_id,
                    metadata={
                        "seed_type": "entity",
                        "routing_priority": "profile" if broad_entity else "normal",
                        "broad_entity_profile": broad_entity,
                        "broad_entity_profile_trajectory_count": len(unique_ids),
                    },
                )
            )
            if broad_entity:
                facet_seeds = self._build_entity_facet_seeds(
                    entity=entity,
                    trajectory_ids=unique_ids,
                    trajectories_by_id=trajectories_by_id,
                )
                seeds.extend(facet_seeds)
                raw_entity_facet_count += len(facet_seeds)

        for entity, trajectory_ids in sorted(entity_to_trajectory_ids.items()):
            inventory_ids = [
                trajectory_id
                for trajectory_id in list(dict.fromkeys(trajectory_ids))
                if self._is_inventory_trajectory(dict(trajectories_by_id[trajectory_id].metadata_json or {}))
            ]
            if not inventory_ids:
                continue
            raw_inventory_count += 1
            seeds.append(
                self._build_seed(
                    seed_id=f"inventory::{slugify(entity)}",
                    page_type="inventory",
                    title=f"{entity} inventory",
                    slug=f"inventory-{entity}",
                    trajectory_ids=inventory_ids,
                    entities=[entity],
                    trajectories_by_id=trajectories_by_id,
                    metadata={"seed_type": "inventory", "routing_priority": "normal"},
                )
            )

        keyword_to_trajectory_ids: dict[str, list[str]] = defaultdict(list)
        keyword_to_entities: dict[str, list[str]] = defaultdict(list)
        for trajectory in trajectories:
            trajectory_entities = self._trajectory_field_list(trajectory, "entity_mentions")
            for keyword in self._trajectory_keyword_list(trajectory)[:3]:
                keyword_to_trajectory_ids[keyword].append(trajectory.id)
                keyword_to_entities[keyword].extend(trajectory_entities)
        for keyword, trajectory_ids in sorted(keyword_to_trajectory_ids.items()):
            unique_ids = list(dict.fromkeys(trajectory_ids))
            if len(unique_ids) < 2:
                continue
            raw_topic_count += 1
            seeds.append(
                self._build_seed(
                    seed_id=f"topic::{slugify(keyword)}",
                    page_type="topic",
                    title=f"{keyword} topic",
                    slug=f"topic-{keyword}",
                    trajectory_ids=unique_ids,
                    entities=_top_ranked_values(keyword_to_entities[keyword], limit=2),
                    trajectories_by_id=trajectories_by_id,
                    metadata={"seed_type": "topic", "routing_priority": "normal"},
                )
            )

        self._trace(
            f"sample={sample_id} wiki_seed_candidates trajectories={len(trajectories)} total={len(seeds)} "
            f"entity={raw_entity_count} entity_facet={raw_entity_facet_count} inventory={raw_inventory_count} topic={raw_topic_count}"
        )
        return seeds

    def _add_historical_backup_seeds(
        self,
        sample_id: str,
        seeds: list[WikiPageSeed],
        trajectories: list[TrajectoryRecord],
    ) -> list[WikiPageSeed]:
        trajectories_by_id = self._trajectory_rows_by_id(trajectories)
        non_index_covered = {
            trajectory_id
            for seed in seeds
            if seed.page_type != "index"
            for trajectory_id in seed.trajectory_ids
        }
        grouped: dict[str, dict[str, Any]] = {}
        for trajectory in trajectories:
            if trajectory.id in non_index_covered:
                continue
            card = self._trajectory_evidence_card(trajectory)
            historical_terms = specific_terms(
                [
                    *sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24),
                    *list(card.get("drift_cluster_keys") or []),
                    *list(card.get("display_items") or []),
                    *list(card.get("display_counts") or []),
                ],
                limit=8,
            )
            if not historical_terms:
                continue
            descriptor = historical_terms[0]
            entities = [
                str(value)
                for value in list(card.get("entity_mentions") or [])[:2]
                if str(value).strip()
            ]
            family = self._family_from_terms(historical_terms)
            page_type = self._page_type_for_family(
                family,
                has_display_values=bool(card.get("display_items") or card.get("display_counts")),
                inventory_like=len(historical_terms) >= 2,
            )
            entity_key = "-".join(slugify(entity) for entity in entities[:2] if slugify(entity)) or "general"
            group_key = f"historical:{page_type}:{entity_key}:{slugify(family) or 'evidence'}"
            group = grouped.setdefault(
                group_key,
                {
                    "page_type": page_type,
                    "family": family,
                    "descriptors": [],
                    "specific_values": [],
                    "entities": [],
                    "trajectory_ids": [],
                },
            )
            group["descriptors"] = [*list(group["descriptors"]), descriptor]
            group["specific_values"] = [*list(group["specific_values"]), *historical_terms]
            group["entities"] = _dedupe_preserve([*list(group["entities"]), *entities])
            group["trajectory_ids"] = [*list(group["trajectory_ids"]), trajectory.id]
        added: list[WikiPageSeed] = []
        for group_key, group in sorted(grouped.items()):
            page_type = str(group["page_type"])
            family = str(group["family"])
            descriptor_metadata = self._group_descriptor_metadata(group, fallback=family)
            readable_descriptor = str(descriptor_metadata["descriptor"])
            policy_descriptor = (
                ""
                if descriptor_metadata.get("wiki_descriptor_rewrite_reason") == "low_quality_descriptor_fallback"
                else readable_descriptor
            )
            trajectory_ids = [str(value) for value in list(group["trajectory_ids"])]
            chunks = self._medium_granularity_chunks(trajectory_ids)
            for chunk_index, chunk_ids in enumerate(chunks, start=1):
                chunk_suffix = f" Part {chunk_index}" if len(chunks) > 1 else ""
                added.append(
                    self._build_seed(
                        seed_id=f"{group_key}::{chunk_index}",
                        page_type=page_type,
                        title=f"{readable_descriptor} evidence{chunk_suffix}",
                        slug=f"historical-{family}-{readable_descriptor}-{chunk_index if len(chunks) > 1 else ''}",
                        trajectory_ids=chunk_ids,
                        entities=[str(value) for value in list(group["entities"])],
                        trajectories_by_id=trajectories_by_id,
                        metadata={
                            "seed_type": "historical_card_backup",
                            "routing_priority": "high",
                            "wiki_index_only_trajectory_rescued_count": len(chunk_ids),
                            "wiki_rescue_group_key": group_key,
                            **descriptor_metadata,
                            **self._granularity_metadata(
                                len(chunk_ids),
                                descriptor=policy_descriptor,
                                family=family,
                                group=group,
                            ),
                        },
                    )
                )
        if added:
            self._trace(f"sample={sample_id} wiki_historical_backup_seeds_added count={len(added)}")
        return [*seeds, *added]

    @staticmethod
    def _unique_slug(base_slug: str, used_slugs: set[str]) -> str:
        slug = slugify(base_slug) or "page"
        if slug not in used_slugs:
            used_slugs.add(slug)
            return slug
        suffix = 2
        while f"{slug}-{suffix}" in used_slugs:
            suffix += 1
        unique = f"{slug}-{suffix}"
        used_slugs.add(unique)
        return unique

    def _rescue_group_profile(self, trajectory: TrajectoryRecord) -> dict[str, Any]:
        card = self._trajectory_evidence_card(trajectory)
        entities = _dedupe_preserve(
            [
                *self._trajectory_field_list(trajectory, "entity_mentions"),
                *[
                    str(value)
                    for value in list(card.get("entity_mentions") or [])
                    if str(value).strip()
                ],
            ]
        )
        display_items = [
            str(value).strip()
            for value in [
                *self._trajectory_field_list(trajectory, "display_items"),
                *list(card.get("display_items") or []),
            ]
            if str(value).strip()
        ]
        display_counts = [
            str(value).strip()
            for value in [
                *self._trajectory_field_list(trajectory, "display_counts"),
                *list(card.get("display_counts") or []),
            ]
            if str(value).strip()
        ]
        terms = specific_terms(
            [
                *display_items,
                *display_counts,
                *list(card.get("drift_cluster_keys") or []),
                *sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24),
                *self._trajectory_field_list(trajectory, "facet_values"),
                *self._trajectory_keyword_list(trajectory),
                trajectory.label,
            ],
            limit=12,
        )
        descriptor = terms[0] if terms else str(trajectory.label or trajectory.id)
        source_surface_terms = [
            str(value).strip()
            for value in list(card.get("source_surface_terms") or [])
            if str(value).strip()
        ]
        historical_item_terms = sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24)
        raw_family = self._family_from_terms([*terms, trajectory.label])
        umbrella_family = self._rescue_umbrella_family_from_terms(
            [
                *terms,
                *display_items,
                *display_counts,
                *source_surface_terms,
                *historical_item_terms,
                *list(card.get("drift_cluster_keys") or []),
                trajectory.label,
            ],
            fallback_family=raw_family,
        )
        entity_key = "-".join(slugify(entity) for entity in entities[:2] if slugify(entity)) or "general"
        page_type = self._page_type_for_family(
            umbrella_family,
            has_display_values=bool(display_items or display_counts),
            inventory_like=len(terms) >= 2,
        )
        initial_group_key = f"{page_type}:{entity_key}:{raw_family}"
        merged_group_key = f"{page_type}:{entity_key}:{umbrella_family}"
        return {
            "page_type": page_type,
            "initial_group_key": initial_group_key,
            "merged_group_key": merged_group_key,
            "family": umbrella_family,
            "descriptor": descriptor,
            "specific_values": _dedupe_preserve(
                [
                    *source_surface_terms,
                    *display_items,
                    *display_counts,
                    *historical_item_terms,
                    *terms,
                ]
            ),
            "entities": entities,
            "entity_key": entity_key,
        }

    @classmethod
    def _merge_small_rescue_groups(
        cls,
        grouped: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        merged: dict[str, dict[str, object]] = {}
        small_by_entity: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
        for group_key, group in grouped.items():
            size = len(list(dict.fromkeys(list(group.get("trajectory_ids") or []))))
            if size >= cls.MIN_GROUPABLE_PAGE_TRAJECTORIES:
                group["wiki_rescue_merge_reason"] = "umbrella_family_grouping"
                group["wiki_rescue_merge_applied"] = bool(
                    len(set(str(value) for value in list(group.get("initial_group_keys") or []))) > 1
                    or group_key not in set(str(value) for value in list(group.get("initial_group_keys") or []))
                )
                merged[group_key] = group
                continue
            entity_key = str(group.get("entity_key") or "general")
            small_by_entity[entity_key].append((group_key, group))

        for entity_key, entries in sorted(small_by_entity.items()):
            if len(entries) == 1:
                group_key, group = entries[0]
                size = len(list(dict.fromkeys(list(group.get("trajectory_ids") or []))))
                group["wiki_rescue_merge_reason"] = (
                    "isolated_after_rescue_merge"
                    if size == 1
                    else "remaining_group_below_minimum"
                )
                group["wiki_rescue_merge_applied"] = False
                merged[group_key] = group
                continue
            combined: dict[str, object] = {
                "page_type": "inventory" if any(str(group.get("page_type")) == "inventory" for _, group in entries) else "topic",
                "family": "general_events",
                "entity_key": entity_key,
                "descriptors": [],
                "specific_values": [],
                "entities": [],
                "trajectory_ids": [],
                "initial_group_keys": [],
                "wiki_rescue_merge_applied": True,
                "wiki_rescue_merge_reason": "small_group_merged_by_entity",
            }
            page_families = [
                str(group.get("family") or "")
                for _, group in entries
                if str(group.get("family") or "").strip()
            ]
            if page_families:
                combined["family"] = page_families[0] if len(set(page_families)) == 1 else "mixed_evidence"
            for group_key, group in entries:
                combined["descriptors"] = [
                    *list(combined["descriptors"] or []),
                    *list(group.get("descriptors") or []),
                ]
                combined["specific_values"] = [
                    *list(combined["specific_values"] or []),
                    *list(group.get("specific_values") or []),
                ]
                combined["entities"] = _dedupe_preserve(
                    [*list(combined["entities"] or []), *list(group.get("entities") or [])]
                )
                combined["trajectory_ids"] = [
                    *list(combined["trajectory_ids"] or []),
                    *list(group.get("trajectory_ids") or []),
                ]
                combined["initial_group_keys"] = [
                    *list(combined["initial_group_keys"] or []),
                    *list(group.get("initial_group_keys") or [group_key]),
                ]
            merged_key = f"{combined['page_type']}:{entity_key}:{combined['family']}"
            merged[merged_key] = combined
        return cls._merge_low_quality_singleton_rescue_groups(merged)

    @classmethod
    def _merge_low_quality_singleton_rescue_groups(
        cls,
        grouped: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        kept: dict[str, dict[str, object]] = {}
        low_quality_by_family: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
        for group_key, group in grouped.items():
            trajectory_ids = list(dict.fromkeys(list(group.get("trajectory_ids") or [])))
            descriptor = next(
                (str(value) for value in list(group.get("descriptors") or []) if str(value).strip()),
                "",
            )
            policy = cls._singleton_policy_for_group(
                trajectory_count=len(trajectory_ids),
                descriptor=descriptor,
                family=group.get("family"),
                group=group,
            )
            if len(trajectory_ids) == 1 and policy["wiki_singleton_policy"] == "merge_required_low_quality":
                family_key = f"{group.get('page_type') or 'inventory'}:sample:{group.get('family') or 'general_events'}"
                low_quality_by_family[family_key].append((group_key, group))
                continue
            kept[group_key] = group
        for family_key, entries in sorted(low_quality_by_family.items()):
            if len(entries) == 1:
                group_key, group = entries[0]
                group["wiki_singleton_low_quality_merged"] = False
                kept[group_key] = group
                continue
            combined: dict[str, object] = {
                "page_type": "inventory" if any(str(group.get("page_type")) == "inventory" for _, group in entries) else "topic",
                "family": str(entries[0][1].get("family") or "general_events"),
                "entity_key": "sample",
                "descriptors": [],
                "specific_values": [],
                "entities": [],
                "trajectory_ids": [],
                "initial_group_keys": [],
                "wiki_rescue_merge_applied": True,
                "wiki_rescue_merge_reason": "sample_family_low_quality_merge",
                "wiki_singleton_low_quality_merged": True,
                "wiki_singleton_merged_from_group_keys": [group_key for group_key, _ in entries],
                "wiki_singleton_merge_stage": "sample_family_low_quality_merge",
            }
            for group_key, group in entries:
                combined["descriptors"] = [
                    *list(combined["descriptors"] or []),
                    *list(group.get("descriptors") or []),
                ]
                combined["specific_values"] = [
                    *list(combined["specific_values"] or []),
                    *list(group.get("specific_values") or []),
                ]
                combined["entities"] = _dedupe_preserve(
                    [*list(combined["entities"] or []), *list(group.get("entities") or [])]
                )
                combined["trajectory_ids"] = [
                    *list(combined["trajectory_ids"] or []),
                    *list(group.get("trajectory_ids") or []),
                ]
                combined["initial_group_keys"] = [
                    *list(combined["initial_group_keys"] or []),
                    *list(group.get("initial_group_keys") or [group_key]),
                ]
            kept[family_key] = combined
        return kept

    def _build_post_plan_rescue_drafts(
        self,
        sample_id: str,
        index_only_trajectory_ids: list[str],
        trajectories_by_id: dict[str, TrajectoryRecord],
        used_slugs: set[str],
    ) -> list[WikiPageDraft]:
        grouped: dict[str, dict[str, object]] = {}
        for trajectory_id in index_only_trajectory_ids:
            trajectory = trajectories_by_id.get(trajectory_id)
            if trajectory is None:
                continue
            profile = self._rescue_group_profile(trajectory)
            page_type = str(profile["page_type"])
            group_key = str(profile["merged_group_key"])
            descriptor = str(profile["descriptor"])
            entities = list(profile.get("entities") or [])
            group = grouped.setdefault(
                group_key,
                {
                    "page_type": page_type,
                    "family": profile.get("family"),
                    "entity_key": profile.get("entity_key") or "general",
                    "descriptors": [],
                    "specific_values": [],
                    "entities": [],
                    "trajectory_ids": [],
                    "initial_group_keys": [],
                },
            )
            group["trajectory_ids"] = [*list(group["trajectory_ids"]), trajectory_id]
            group["descriptors"] = [*list(group["descriptors"]), descriptor]
            group["specific_values"] = [
                *list(group.get("specific_values") or []),
                *list(profile.get("specific_values") or []),
            ]
            group["entities"] = _dedupe_preserve([*list(group["entities"]), *entities])
            group["initial_group_keys"] = _dedupe_preserve(
                [*list(group["initial_group_keys"]), str(profile["initial_group_key"])]
            )

        rescue_drafts: list[WikiPageDraft] = []
        grouped = self._merge_small_rescue_groups(grouped)
        for group_key, group in sorted(grouped.items()):
            page_type = str(group["page_type"])
            family = str(group.get("family") or group_key.rsplit(":", 1)[-1])
            descriptor_metadata = self._group_descriptor_metadata(group, fallback=family)
            descriptor = str(descriptor_metadata["descriptor"])
            policy_descriptor = (
                ""
                if descriptor_metadata.get("wiki_descriptor_rewrite_reason") == "low_quality_descriptor_fallback"
                else descriptor
            )
            trajectory_ids = [str(value) for value in list(group["trajectory_ids"])]
            entities = [str(value) for value in list(group["entities"])]
            chunks = self._medium_granularity_chunks(trajectory_ids)
            for chunk_index, chunk_ids in enumerate(chunks, start=1):
                chunk_suffix = f" part {chunk_index}" if len(chunks) > 1 else ""
                title = f"{descriptor} evidence{chunk_suffix}"
                base_slug = f"rescue-{group_key.replace(':', '-')}-{chunk_index if len(chunks) > 1 else ''}"
                exception_reason = (
                    "isolated_after_rescue_merge"
                    if len(chunk_ids) == 1
                    else ("remaining_group_below_minimum" if len(chunk_ids) < self.MIN_GROUPABLE_PAGE_TRAJECTORIES else None)
                )
                initial_group_keys = [str(value) for value in list(group.get("initial_group_keys") or [])]
                merge_reason = str(group.get("wiki_rescue_merge_reason") or "umbrella_family_grouping")
                rescue_metadata = {
                    "seed_type": "post_plan_rescue",
                    "routing_priority": "high",
                    "wiki_rescue_group_key": group_key,
                    "wiki_rescue_initial_group_key": (
                        initial_group_keys[0] if len(initial_group_keys) == 1 else initial_group_keys
                    ),
                    "wiki_rescue_merged_group_key": group_key,
                    "wiki_rescue_merge_applied": bool(group.get("wiki_rescue_merge_applied")),
                    "wiki_rescue_merge_reason": merge_reason,
                    "wiki_rescue_group_size_before_merge": max(
                        1,
                        len(initial_group_keys),
                    ),
                    "wiki_rescue_group_size_after_merge": len(trajectory_ids),
                    "wiki_rescue_reason": "post_plan_index_only_trajectory",
                    "wiki_singleton_low_quality_merged": bool(group.get("wiki_singleton_low_quality_merged")),
                    "wiki_singleton_merged_from_group_keys": list(group.get("wiki_singleton_merged_from_group_keys") or []),
                    "wiki_singleton_merge_stage": group.get("wiki_singleton_merge_stage"),
                    **descriptor_metadata,
                    **self._granularity_metadata(
                        len(chunk_ids),
                        exception_reason=exception_reason,
                        descriptor=policy_descriptor,
                        family=family,
                        group=group,
                    ),
                }
                seed = self._build_seed(
                    seed_id=f"post_plan_rescue::{group_key}::{chunk_index}",
                    page_type=page_type,
                    title=title,
                    slug=self._unique_slug(base_slug, used_slugs),
                    trajectory_ids=chunk_ids,
                    entities=entities,
                    trajectories_by_id=trajectories_by_id,
                    metadata=rescue_metadata,
                )
                draft = self._draft_from_seed(seed)
                draft.metadata.update(rescue_metadata)
                rescue_drafts.append(draft)
        if rescue_drafts:
            self._trace(
                f"sample={sample_id} wiki_post_plan_rescue_pages_built "
                f"pages={len(rescue_drafts)} trajectories={len(index_only_trajectory_ids)}"
            )
        return rescue_drafts

    def _split_overwide_non_index_drafts(
        self,
        sample_id: str,
        drafts: list[WikiPageDraft],
    ) -> tuple[list[WikiPageDraft], int]:
        overwide_count = sum(
            1
            for draft in drafts
            if draft.page_type != "index"
            and len(list(dict.fromkeys(draft.trajectory_ids))) > self.MAX_PAGE_TRAJECTORIES
        )
        if not overwide_count:
            return drafts, 0
        used_slugs = {draft.slug for draft in drafts}
        output: list[WikiPageDraft] = []
        for draft in drafts:
            trajectory_ids = list(dict.fromkeys(draft.trajectory_ids))
            if draft.page_type == "index" or len(trajectory_ids) <= self.MAX_PAGE_TRAJECTORIES:
                output.append(draft)
                continue
            chunks = self._medium_granularity_chunks(trajectory_ids)
            for part_index, chunk_ids in enumerate(chunks, start=1):
                title = f"{draft.title} Part {part_index}" if len(chunks) > 1 else draft.title
                slug = self._unique_slug(f"{draft.slug}-part-{part_index}", used_slugs)
                metadata = {
                    **dict(draft.metadata or {}),
                    "wiki_overwide_page_split": True,
                    "wiki_overwide_original_trajectory_count": len(trajectory_ids),
                    "wiki_overwide_split_parent_slug": draft.slug,
                    "wiki_overwide_split_part_index": part_index,
                    "wiki_overwide_split_part_count": len(chunks),
                    **self._granularity_metadata(
                        len(chunk_ids),
                        descriptor=title,
                        family=draft.page_type,
                        group={"descriptors": [title], "entities": list(draft.entities)},
                    ),
                }
                output.append(
                    WikiPageDraft(
                        page_type=draft.page_type,
                        title=title,
                        slug=slug,
                        trajectory_ids=chunk_ids,
                        entities=list(draft.entities),
                        linked_slugs=list(draft.linked_slugs),
                        metadata=metadata,
                    )
                )
        self._trace(
            f"sample={sample_id} wiki_overwide_non_index_split pages={overwide_count} "
            f"after_pages={len(output)}"
        )
        return output, overwide_count

    @staticmethod
    def _fragmentation_metadata_from_drafts(
        drafts: list[WikiPageDraft],
        *,
        overwide_non_index_page_count: int = 0,
    ) -> dict[str, Any]:
        non_index = [draft for draft in drafts if draft.page_type != "index"]
        page_count = len(non_index)
        trajectory_counts = [len(list(dict.fromkeys(draft.trajectory_ids))) for draft in non_index]
        singleton_count = sum(1 for count in trajectory_counts if count == 1)
        total_linked = sum(trajectory_counts)
        seed_type_counts: Counter[str] = Counter()
        seed_type_singletons: Counter[str] = Counter()
        for draft, trajectory_count in zip(non_index, trajectory_counts):
            seed_type = str((draft.metadata or {}).get("seed_type") or "unknown")
            seed_type_counts[seed_type] += 1
            if trajectory_count == 1:
                seed_type_singletons[seed_type] += 1
        metadata_rows = [dict(draft.metadata or {}) for draft in non_index]
        singleton_rate_by_seed_type = {
            seed_type: (
                seed_type_singletons[seed_type] / count
                if count
                else None
            )
            for seed_type, count in sorted(seed_type_counts.items())
        }
        return {
            "wiki_fragmentation_diagnostics_available": True,
            "wiki_non_index_page_count": page_count,
            "wiki_singleton_non_index_page_count": singleton_count,
            "wiki_singleton_non_index_page_rate": (singleton_count / page_count if page_count else None),
            "wiki_mean_trajectories_per_non_index_page": (total_linked / page_count if page_count else None),
            "wiki_singleton_rate_by_seed_type": singleton_rate_by_seed_type,
            "wiki_post_plan_rescue_singleton_count": seed_type_singletons.get("post_plan_rescue", 0),
            "wiki_entity_facet_singleton_count": seed_type_singletons.get("entity_facet", 0),
            "wiki_allowed_specific_singleton_count": sum(
                1 for metadata in metadata_rows
                if metadata.get("wiki_singleton_policy") == "allowed_isolated_specific"
            ),
            "wiki_low_quality_singleton_count": sum(
                1 for metadata in metadata_rows
                if metadata.get("wiki_singleton_policy") == "merge_required_low_quality"
            ),
            "wiki_low_quality_singleton_merged_count": sum(
                1 for metadata in metadata_rows
                if metadata.get("wiki_singleton_low_quality_merged") is True
            ),
            "wiki_overwide_non_index_page_count": overwide_non_index_page_count,
            "wiki_overwide_page_split_count": sum(
                1 for metadata in metadata_rows if metadata.get("wiki_overwide_page_split") is True
            ),
            "wiki_max_non_index_trajectory_count": max(trajectory_counts) if trajectory_counts else 0,
        }

    def _apply_non_index_coverage_audit(
        self,
        sample_id: str,
        drafts: list[WikiPageDraft],
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> list[WikiPageDraft]:
        all_trajectory_ids = list(trajectories_by_id)
        if not all_trajectory_ids:
            return drafts
        non_index_covered_before = {
            trajectory_id
            for draft in drafts
            if draft.page_type != "index"
            for trajectory_id in draft.trajectory_ids
            if trajectory_id in trajectories_by_id
        }
        index_only_before = [
            trajectory_id
            for trajectory_id in all_trajectory_ids
            if trajectory_id not in non_index_covered_before
        ]
        used_slugs = {draft.slug for draft in drafts}
        rescue_drafts = self._build_post_plan_rescue_drafts(
            sample_id,
            index_only_before,
            trajectories_by_id,
            used_slugs,
        )
        audited = [*drafts, *rescue_drafts]
        audited, overwide_before_split = self._split_overwide_non_index_drafts(sample_id, audited)
        non_index_covered_after = {
            trajectory_id
            for draft in audited
            if draft.page_type != "index"
            for trajectory_id in draft.trajectory_ids
            if trajectory_id in trajectories_by_id
        }
        index_only_after = [
            trajectory_id
            for trajectory_id in all_trajectory_ids
            if trajectory_id not in non_index_covered_after
        ]
        coverage_rate = len(non_index_covered_after) / len(all_trajectory_ids)
        rescue_trajectory_count = sum(len(draft.trajectory_ids) for draft in rescue_drafts)
        self._trace(
            f"sample={sample_id} wiki_non_index_coverage_audit trajectories={len(all_trajectory_ids)} "
            f"covered_before={len(non_index_covered_before)} index_only_before={len(index_only_before)} "
            f"rescue_pages={len(rescue_drafts)} rescue_trajectories={rescue_trajectory_count} "
            f"covered_after={len(non_index_covered_after)} coverage_rate={coverage_rate:.4f}"
        )
        if index_only_after:
            self._trace(
                f"sample={sample_id} wiki_non_index_coverage_incomplete "
                f"index_only={len(index_only_after)} ids={','.join(index_only_after[:12])}"
            )

        fragmentation_metadata = self._fragmentation_metadata_from_drafts(
            audited,
            overwide_non_index_page_count=overwide_before_split,
        )
        self._trace(
            f"sample={sample_id} wiki_fragmentation_diagnostics "
            f"non_index_pages={fragmentation_metadata['wiki_non_index_page_count']} "
            f"singleton_pages={fragmentation_metadata['wiki_singleton_non_index_page_count']} "
            f"singleton_rate={float(fragmentation_metadata['wiki_singleton_non_index_page_rate'] or 0.0):.4f} "
            f"mean_trajectories={float(fragmentation_metadata['wiki_mean_trajectories_per_non_index_page'] or 0.0):.2f}"
        )

        enriched: list[WikiPageDraft] = []
        for draft in audited:
            metadata = {
                **dict(draft.metadata or {}),
                "wiki_non_index_coverage_audit_used": True,
                "wiki_non_index_trajectory_coverage_rate": coverage_rate,
                "wiki_index_only_trajectory_count_before_rescue": len(index_only_before),
                "wiki_index_only_trajectory_count_after_rescue": len(index_only_after),
                "wiki_rescue_page_count": len(rescue_drafts),
                "wiki_rescue_trajectory_count": rescue_trajectory_count,
                "wiki_overwide_non_index_page_count": overwide_before_split,
                **fragmentation_metadata,
            }
            enriched.append(
                WikiPageDraft(
                    page_type=draft.page_type,
                    title=draft.title,
                    slug=draft.slug,
                    trajectory_ids=list(draft.trajectory_ids),
                    entities=list(draft.entities),
                    linked_slugs=list(draft.linked_slugs),
                    metadata=metadata,
                )
            )
        return enriched

    @staticmethod
    def _seed_overlap_ratio(topic_seed: WikiPageSeed, coverage_seed: WikiPageSeed) -> float:
        topic_ids = set(topic_seed.trajectory_ids)
        if not topic_ids:
            return 0.0
        return len(topic_ids & set(coverage_seed.trajectory_ids)) / len(topic_ids)

    def _suppress_redundant_topic_seeds(self, sample_id: str, seeds: list[WikiPageSeed]) -> list[WikiPageSeed]:
        coverage_seeds = [seed for seed in seeds if seed.page_type in {"entity", "inventory"}]
        kept: list[WikiPageSeed] = []
        suppressed = 0
        suppressed_slugs: list[str] = []
        for seed in seeds:
            if seed.page_type != "topic":
                kept.append(seed)
                continue
            if seed.metadata.get("seed_type") == "entity_facet":
                kept.append(seed)
                continue
            if any(
                self._seed_overlap_ratio(seed, coverage_seed) >= self.TOPIC_REDUNDANT_OVERLAP
                for coverage_seed in coverage_seeds
            ):
                suppressed += 1
                if len(suppressed_slugs) < 5:
                    suppressed_slugs.append(seed.slug)
                continue
            kept.append(seed)
        suppressed_preview = ",".join(suppressed_slugs) if suppressed_slugs else "none"
        self._trace(
            f"sample={sample_id} wiki_seed_topics_suppressed count={suppressed} sample_slugs={suppressed_preview}"
        )
        return kept

    @staticmethod
    def _chunked_ids(trajectory_ids: list[str], chunk_size: int) -> list[list[str]]:
        return [trajectory_ids[index : index + chunk_size] for index in range(0, len(trajectory_ids), chunk_size)]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def _split_seed_trajectory_ids(
        self,
        sample_id: str,
        seed: WikiPageSeed,
        embedding_rows: dict[str, EmbeddingRecord],
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> list[list[str]]:
        trajectory_ids = list(seed.trajectory_ids)
        if len(trajectory_ids) <= self.MAX_PAGE_TRAJECTORIES:
            return [trajectory_ids]
        shard_count = (len(trajectory_ids) + self.MAX_PAGE_TRAJECTORIES - 1) // self.MAX_PAGE_TRAJECTORIES
        available_ids = [trajectory_id for trajectory_id in trajectory_ids if trajectory_id in embedding_rows]
        if len(available_ids) < shard_count:
            self._trace(
                f"sample={sample_id} wiki_seed_split_fallback slug={seed.slug} "
                f"reason=insufficient_embeddings available_embeddings={len(available_ids)} required_shards={shard_count}"
            )
            return self._chunked_ids(trajectory_ids, self.MAX_PAGE_TRAJECTORIES)

        def _richness(trajectory_id: str) -> tuple[int, int, str]:
            trajectory = trajectories_by_id[trajectory_id]
            return (
                len(self._trajectory_field_list(trajectory, "exact_terms"))
                + len(self._trajectory_field_list(trajectory, "facet_values"))
                + len(self._trajectory_historical_terms(trajectory)),
                len(self._trajectory_keyword_list(trajectory)),
                trajectory_id,
            )

        centers: list[str] = [max(available_ids, key=_richness)]
        while len(centers) < shard_count:
            remaining = [trajectory_id for trajectory_id in available_ids if trajectory_id not in centers]
            if not remaining:
                break
            def min_distance(trajectory_id: str) -> float:
                vector = embedding_rows[trajectory_id].vector_json
                return min(
                    1.0 - self._cosine_similarity(vector, embedding_rows[center_id].vector_json)
                    for center_id in centers
                )

            centers.append(max(remaining, key=min_distance))

        clusters: list[list[str]] = [[center_id] for center_id in centers]
        remaining_ids = [trajectory_id for trajectory_id in trajectory_ids if trajectory_id not in centers]
        for trajectory_id in remaining_ids:
            if trajectory_id not in embedding_rows:
                target_index = min(range(len(clusters)), key=lambda index: (len(clusters[index]), index))
                clusters[target_index].append(trajectory_id)
                continue
            ranked_targets = sorted(
                range(len(clusters)),
                key=lambda index: (
                    max(
                        self._cosine_similarity(
                            embedding_rows[trajectory_id].vector_json,
                            embedding_rows[member_id].vector_json,
                        )
                        for member_id in clusters[index]
                        if member_id in embedding_rows
                    ),
                    -len(clusters[index]),
                    -index,
                ),
                reverse=True,
            )
            assigned = False
            for target_index in ranked_targets:
                if len(clusters[target_index]) >= self.MAX_PAGE_TRAJECTORIES:
                    continue
                clusters[target_index].append(trajectory_id)
                assigned = True
                break
            if not assigned:
                target_index = min(range(len(clusters)), key=lambda index: (len(clusters[index]), index))
                clusters[target_index].append(trajectory_id)

        flattened = [trajectory_id for cluster in clusters for trajectory_id in cluster]
        if set(flattened) != set(trajectory_ids):
            self._trace(
                f"sample={sample_id} wiki_seed_split_fallback slug={seed.slug} reason=cluster_membership_mismatch"
            )
            return self._chunked_ids(trajectory_ids, self.MAX_PAGE_TRAJECTORIES)
        ordered_clusters: list[list[str]] = []
        for cluster in clusters:
            ordered_clusters.append([trajectory_id for trajectory_id in trajectory_ids if trajectory_id in set(cluster)])
        return ordered_clusters

    def _cluster_descriptor(
        self,
        cluster_ids: list[str],
        trajectories_by_id: dict[str, TrajectoryRecord],
        base_entities: list[str],
    ) -> str:
        terms = _top_ranked_values(
            (
                value
                for trajectory_id in cluster_ids
                for value in (
                    self._trajectory_field_list(trajectories_by_id[trajectory_id], "exact_terms")
                    + self._trajectory_field_list(trajectories_by_id[trajectory_id], "facet_values")
                    + self._trajectory_historical_terms(trajectories_by_id[trajectory_id])
                    + self._trajectory_keyword_list(trajectories_by_id[trajectory_id])
                )
            ),
            limit=5,
        )
        excluded = {entity.casefold() for entity in base_entities}
        for term in terms:
            if term.casefold() not in excluded:
                return term
        return ""

    def _split_broad_seeds(
        self,
        sample_id: str,
        seeds: list[WikiPageSeed],
        trajectories: list[TrajectoryRecord],
    ) -> list[WikiPageSeed]:
        trajectories_by_id = self._trajectory_rows_by_id(trajectories)
        embedding_rows = self._summary_embedding_rows_by_id(trajectories)
        split_seeds: list[WikiPageSeed] = []
        for seed in seeds:
            if (
                seed.page_type == "index"
                or bool(seed.metadata.get("broad_entity_profile"))
                or len(seed.trajectory_ids) <= self.MAX_PAGE_TRAJECTORIES
            ):
                split_seeds.append(seed)
                continue
            clusters = self._split_seed_trajectory_ids(sample_id, seed, embedding_rows, trajectories_by_id)
            shard_count = len(clusters)
            cluster_sizes = ",".join(str(len(cluster)) for cluster in clusters)
            self._trace(
                f"sample={sample_id} wiki_seed_split slug={seed.slug} trajectories={len(seed.trajectory_ids)} "
                f"shards={shard_count} cluster_sizes={cluster_sizes}"
            )
            for shard_index, cluster_ids in enumerate(clusters, start=1):
                descriptor = self._cluster_descriptor(cluster_ids, trajectories_by_id, seed.entities)
                suffix = f"part-{shard_index}"
                slug = f"{seed.slug}-{slugify(descriptor)}-{suffix}" if descriptor else f"{seed.slug}-{suffix}"
                title_suffix = f" ({descriptor})" if descriptor else ""
                title = f"{seed.title}{title_suffix} Part {shard_index}"
                split_seeds.append(
                    self._build_seed(
                        seed_id=f"{seed.seed_id}::{suffix}",
                        page_type=seed.page_type,
                        title=title,
                        slug=slug,
                        trajectory_ids=cluster_ids,
                        entities=seed.entities,
                        trajectories_by_id=trajectories_by_id,
                        metadata={
                            **dict(seed.metadata),
                            "seed_type": seed.metadata.get("seed_type", seed.page_type),
                            "shard_index": shard_index,
                            "shard_count": shard_count,
                        },
                    )
                )
        return split_seeds

    def _cap_seeds(self, sample_id: str, seeds: list[WikiPageSeed]) -> list[WikiPageSeed]:
        if len(seeds) <= self.MAX_PLAN_SEEDS:
            return seeds
        priority = {"index": 0, "inventory": 1, "topic": 2, "entity": 3}
        index_seeds = [seed for seed in seeds if seed.page_type == "index"]
        non_index = [seed for seed in seeds if seed.page_type != "index"]
        non_index.sort(
            key=lambda seed: (
                0 if seed.metadata.get("routing_priority") == "high" else (2 if seed.metadata.get("routing_priority") == "profile" else 1),
                priority.get(seed.page_type, 9),
                -len(seed.trajectory_ids),
                -len(seed.exact_terms) - len(seed.facet_values) - len(seed.metadata.get("wiki_historical_item_terms") or []),
                seed.slug,
            )
        )
        capped = index_seeds[:1] + non_index[: max(self.MAX_PLAN_SEEDS - len(index_seeds[:1]), 0)]
        self._trace(f"sample={sample_id} wiki_seed_cap_applied before={len(seeds)} after={len(capped)}")
        return capped

    def _finalize_seed_links(self, seeds: list[WikiPageSeed]) -> list[WikiPageSeed]:
        entity_slugs_by_entity: dict[str, list[str]] = defaultdict(list)
        for seed in seeds:
            if seed.page_type != "entity":
                continue
            for entity in seed.entities:
                entity_slugs_by_entity[entity].append(seed.slug)
        finalized: list[WikiPageSeed] = []
        for seed in seeds:
            linked_slugs: list[str] = []
            if seed.page_type != "index":
                linked_slugs.append("index")
            if seed.page_type in {"inventory", "topic"}:
                for entity in seed.entities:
                    linked_slugs.extend(entity_slugs_by_entity.get(entity, []))
            finalized.append(
                WikiPageSeed(
                    seed_id=seed.seed_id,
                    page_type=seed.page_type,
                    title=seed.title,
                    slug=seed.slug,
                    trajectory_ids=list(seed.trajectory_ids),
                    entities=list(seed.entities),
                    exact_terms=list(seed.exact_terms),
                    facet_values=list(seed.facet_values),
                    keywords=list(seed.keywords),
                    representative_trajectory_ids=list(seed.representative_trajectory_ids),
                    linked_slugs=_dedupe_preserve(linked_slugs),
                    metadata=dict(seed.metadata),
                )
            )
        return finalized

    def _plan_seeds(self, sample_id: str, trajectories: list[TrajectoryRecord]) -> list[WikiPageSeed]:
        started_at = time.perf_counter()
        build_started = time.perf_counter()
        seeds = self._build_candidate_seeds(sample_id, trajectories)
        build_ms = (time.perf_counter() - build_started) * 1000.0
        suppress_started = time.perf_counter()
        seeds = self._suppress_redundant_topic_seeds(sample_id, seeds)
        suppress_ms = (time.perf_counter() - suppress_started) * 1000.0
        backup_started = time.perf_counter()
        seeds = self._add_historical_backup_seeds(sample_id, seeds, trajectories)
        backup_ms = (time.perf_counter() - backup_started) * 1000.0
        split_started = time.perf_counter()
        seeds = self._split_broad_seeds(sample_id, seeds, trajectories)
        split_ms = (time.perf_counter() - split_started) * 1000.0
        cap_started = time.perf_counter()
        seeds = self._cap_seeds(sample_id, seeds)
        cap_ms = (time.perf_counter() - cap_started) * 1000.0
        finalize_started = time.perf_counter()
        seeds = self._finalize_seed_links(seeds)
        finalize_ms = (time.perf_counter() - finalize_started) * 1000.0
        total_ms = (time.perf_counter() - started_at) * 1000.0
        self._trace(
            f"sample={sample_id} wiki_seed_plan_done total={len(seeds)} latency_ms={total_ms:.1f} "
            f"build_ms={build_ms:.1f} suppress_ms={suppress_ms:.1f} backup_ms={backup_ms:.1f} split_ms={split_ms:.1f} "
            f"cap_ms={cap_ms:.1f} finalize_ms={finalize_ms:.1f}"
        )
        self._trace(f"sample={sample_id} wiki_seed_final count={len(seeds)}")
        return seeds

    @staticmethod
    def _draft_from_seed(seed: WikiPageSeed) -> WikiPageDraft:
        return WikiPageDraft(
            page_type=seed.page_type,
            title=seed.title,
            slug=seed.slug,
            trajectory_ids=list(seed.trajectory_ids),
            entities=list(seed.entities),
            linked_slugs=list(seed.linked_slugs),
            metadata={
                **dict(seed.metadata),
                "seed_id": seed.seed_id,
                "seed_type": seed.metadata.get("seed_type", seed.page_type),
                "seed_trajectory_count": len(seed.trajectory_ids),
                "representative_trajectory_ids": list(seed.representative_trajectory_ids),
                "exact_terms": list(seed.exact_terms),
                "facet_values": list(seed.facet_values),
                "keywords": list(seed.keywords),
                "routing_priority": seed.metadata.get("routing_priority", "normal"),
                "shard_index": int(seed.metadata.get("shard_index", 1) or 1),
                "shard_count": int(seed.metadata.get("shard_count", 1) or 1),
            },
        )

    def _seed_manifest(self, seeds: list[WikiPageSeed], trajectories_by_id: dict[str, TrajectoryRecord]) -> str:
        sections: list[str] = []
        for seed in seeds:
            representative_sections = []
            for trajectory_id in seed.representative_trajectory_ids[: self.MAX_REPRESENTATIVE_SUMMARIES]:
                trajectory = trajectories_by_id.get(trajectory_id)
                if trajectory is None:
                    continue
                representative_sections.append(
                    f"- {trajectory_id}: {self._trajectory_routing_summary_text(trajectory) or 'None.'}"
                )
            evidence_card_sections = [
                render_trajectory_evidence_card(card)
                for card in list(seed.metadata.get("trajectory_evidence_cards") or [])[: self.MAX_REPRESENTATIVE_SUMMARIES]
                if isinstance(card, dict)
            ]
            sections.append(
                f"### {seed.seed_id}\n"
                f"page_type_hint={seed.page_type}\n"
                f"trajectory_count={len(seed.trajectory_ids)}\n"
                f"trajectory_ids={', '.join(seed.trajectory_ids) or 'none'}\n"
                f"entities={', '.join(seed.entities) or 'none'}\n"
                f"exact_terms={', '.join(seed.exact_terms) or 'none'}\n"
                f"facet_values={', '.join(seed.facet_values) or 'none'}\n"
                f"keywords={', '.join(seed.keywords) or 'none'}\n"
                f"historical_item_terms={', '.join(list(seed.metadata.get('wiki_historical_item_terms') or [])) or 'none'}\n"
                "representative_summaries:\n"
                + ("\n".join(representative_sections) if representative_sections else "- none")
                + "\ntrajectory_evidence_cards:\n"
                + ("\n".join(evidence_card_sections) if evidence_card_sections else "- none")
            )
        return "\n\n".join(sections)

    @staticmethod
    def _parse_plan(text: str) -> list[WikiPageDraft]:
        drafts: list[WikiPageDraft] = []
        pattern = re.compile(
            r"^- page_type=(?P<page_type>[^|]+)\| title=(?P<title>[^|]+)\| slug=(?P<slug>[^|]+)\| "
            r"trajectories=(?P<trajectories>[^|]+)\| entities=(?P<entities>[^|]+)\| links=(?P<links>.+)$",
            flags=re.MULTILINE,
        )
        for match in pattern.finditer(text):
            trajectories = [value.strip() for value in match.group("trajectories").split(",") if value.strip() and value.strip() != "none"]
            entities = [value.strip() for value in match.group("entities").split(",") if value.strip() and value.strip() != "none"]
            links = [value.strip() for value in match.group("links").split(",") if value.strip() and value.strip() != "none"]
            drafts.append(
                WikiPageDraft(
                    page_type=match.group("page_type").strip(),
                    title=match.group("title").strip(),
                    slug=slugify(match.group("slug").strip()),
                    trajectory_ids=trajectories,
                    entities=entities,
                    linked_slugs=links,
                    metadata={},
                )
            )
        return drafts

    @staticmethod
    def _has_valid_index(drafts: list[WikiPageDraft]) -> bool:
        return sum(1 for draft in drafts if draft.page_type == "index") == 1

    @staticmethod
    def _trajectory_overlap_score(draft: WikiPageDraft, seed: WikiPageSeed) -> tuple[float, int]:
        if draft.slug == seed.slug:
            return (2.0, len(seed.trajectory_ids))
        draft_ids = set(draft.trajectory_ids)
        seed_ids = set(seed.trajectory_ids)
        if not draft_ids or not seed_ids:
            return (0.0, 0)
        overlap = len(draft_ids & seed_ids)
        union = len(draft_ids | seed_ids)
        return (overlap / union if union else 0.0, overlap)

    @staticmethod
    def _best_overlapping_seed(
        draft: WikiPageDraft,
        seeds: list[WikiPageSeed],
    ) -> tuple[WikiPageSeed | None, str]:
        exact_slug_matches = [seed for seed in seeds if seed.slug == draft.slug]
        if exact_slug_matches:
            return exact_slug_matches[0], "slug_exact"

        same_type = [seed for seed in seeds if seed.page_type == draft.page_type]
        if same_type:
            matched = max(same_type, key=lambda seed: WikiCompiler._trajectory_overlap_score(draft, seed))
            if WikiCompiler._trajectory_overlap_score(draft, matched)[1] > 0:
                return matched, "same_type_overlap"

        if seeds:
            matched = max(seeds, key=lambda seed: WikiCompiler._trajectory_overlap_score(draft, seed))
            if WikiCompiler._trajectory_overlap_score(draft, matched)[1] > 0:
                return matched, "any_type_overlap"

        return None, "trajectory_synthesized"

    def _attach_seed_metadata(
        self,
        sample_id: str,
        drafts: list[WikiPageDraft],
        seeds: list[WikiPageSeed],
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> list[WikiPageDraft]:
        enriched: list[WikiPageDraft] = []
        for draft in drafts:
            matched_seed, match_source = self._best_overlapping_seed(draft, seeds)
            trajectory_ids = list(dict.fromkeys(draft.trajectory_ids or (matched_seed.trajectory_ids if matched_seed else [])))
            rows = [
                trajectories_by_id[trajectory_id]
                for trajectory_id in trajectory_ids
                if trajectory_id in trajectories_by_id
            ]
            evidence_entities = self._trajectory_evidence_metadata(rows).get("entity_mentions", []) if rows else []
            base_metadata: dict[str, Any] = {}
            if matched_seed is not None:
                base_metadata = {
                    **dict(matched_seed.metadata),
                    "seed_id": matched_seed.seed_id,
                    "seed_type": matched_seed.metadata.get("seed_type", matched_seed.page_type),
                    "seed_trajectory_count": len(matched_seed.trajectory_ids),
                    "representative_trajectory_ids": list(matched_seed.representative_trajectory_ids),
                    "exact_terms": list(matched_seed.exact_terms),
                    "facet_values": list(matched_seed.facet_values),
                    "keywords": list(matched_seed.keywords),
                    "routing_priority": matched_seed.metadata.get("routing_priority", "normal"),
                    "shard_index": int(matched_seed.metadata.get("shard_index", 1) or 1),
                    "shard_count": int(matched_seed.metadata.get("shard_count", 1) or 1),
                }
            else:
                base_metadata = {
                    "seed_id": f"planner::{draft.slug}",
                    "seed_type": "planner_derived",
                    "seed_trajectory_count": len(trajectory_ids),
                    "routing_priority": "normal",
                    "shard_index": 1,
                    "shard_count": 1,
                }
            metadata = self._merge_evidence_metadata(
                base_metadata,
                rows,
                match_source=match_source,
                force_synthesized=matched_seed is None or set(trajectory_ids) != set(matched_seed.trajectory_ids),
            )
            if metadata.get("wiki_evidence_metadata_synthesized"):
                self._trace(
                    f"sample={sample_id} wiki_page_metadata_synthesized slug={draft.slug} "
                    f"source={match_source} trajectories={len(rows)}"
                )
            linked_slugs = draft.linked_slugs or (list(matched_seed.linked_slugs) if matched_seed is not None else [])
            entities = draft.entities or (list(matched_seed.entities) if matched_seed is not None else list(evidence_entities))
            metadata = {
                **metadata,
                "wiki_seed_match_source": match_source,
            }
            enriched.append(
                WikiPageDraft(
                    page_type=draft.page_type,
                    title=draft.title,
                    slug=draft.slug,
                    trajectory_ids=trajectory_ids,
                    entities=_dedupe_preserve(entities),
                    linked_slugs=list(dict.fromkeys(linked_slugs)),
                    metadata=metadata,
                )
            )
        return enriched

    def _drop_empty_non_index_drafts(self, sample_id: str, drafts: list[WikiPageDraft]) -> list[WikiPageDraft]:
        kept: list[WikiPageDraft] = []
        dropped: list[str] = []
        for draft in drafts:
            if draft.page_type != "index" and not draft.trajectory_ids:
                dropped.append(draft.slug)
                continue
            kept.append(draft)
        if dropped:
            preview = ",".join(dropped[:8])
            self._trace(
                f"sample={sample_id} wiki_empty_non_index_pages_dropped count={len(dropped)} slugs={preview}"
            )
        return kept

    def _plan_pages(self, sample_id: str, seeds: list[WikiPageSeed], trajectories: list[TrajectoryRecord]) -> list[WikiPageDraft]:
        trajectories_by_id = self._trajectory_rows_by_id(trajectories)
        manifest = self._seed_manifest(seeds, trajectories_by_id)
        prompt = load_prompt("wiki_page_plan") + "\n\nCandidate seed manifest:\n" + manifest
        started_at = time.perf_counter()
        self._trace(
            f"sample={sample_id} wiki_plan_start seeds={len(seeds)} manifest_chars={len(manifest)}"
        )
        try:
            response = self.llm_provider.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata={"task": "wiki_page_plan", "sample_id": sample_id},
            )
            drafts = self._parse_plan(response.text)
            if drafts and self._has_valid_index(drafts):
                drafts = self._attach_seed_metadata(sample_id, drafts, seeds, trajectories_by_id)
                drafts = self._drop_empty_non_index_drafts(sample_id, drafts)
                drafts = self._apply_non_index_coverage_audit(sample_id, drafts, trajectories_by_id)
                self._trace(
                    f"sample={sample_id} wiki_plan_parse_ok pages={len(drafts)} response_chars={len(response.text)} "
                    f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
                )
                return drafts
            self._trace(
                f"sample={sample_id} wiki_plan_invalid reason=parse_empty response_chars={len(response.text)} "
                f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f} fallback=deterministic"
            )
        except Exception as exc:  # noqa: BLE001
            self._trace(
                f"sample={sample_id} wiki_plan_failed error={exc.__class__.__name__} "
                f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f} fallback=deterministic"
            )
        drafts = [self._draft_from_seed(seed) for seed in seeds]
        drafts = self._drop_empty_non_index_drafts(sample_id, drafts)
        drafts = self._apply_non_index_coverage_audit(sample_id, drafts, trajectories_by_id)
        self._trace(
            f"sample={sample_id} wiki_plan_fallback_done pages={len(drafts)} "
            f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
        )
        return drafts

    @staticmethod
    def _validate_page_markdown(markdown_text: str) -> list[str]:
        return [heading for heading in _REQUIRED_WIKI_PAGE_HEADINGS if heading not in markdown_text]

    @staticmethod
    def _draft_metadata_list(draft: WikiPageDraft, field_name: str) -> list[str]:
        return [str(value).strip() for value in list((draft.metadata or {}).get(field_name) or []) if str(value).strip()]

    def _ensure_draft_evidence_metadata(
        self,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
        *,
        match_source: str = "trajectory_fallback",
    ) -> WikiPageDraft:
        if not draft.trajectory_ids:
            return draft
        rows = [
            trajectories_by_id[trajectory_id]
            for trajectory_id in list(dict.fromkeys(draft.trajectory_ids))
            if trajectory_id in trajectories_by_id
        ]
        if not rows:
            return draft
        metadata_before = dict(draft.metadata or {})
        row_ids = [row.id for row in rows]
        source_ids = [
            str(value).strip()
            for value in list(metadata_before.get("wiki_evidence_source_trajectory_ids") or [])
            if str(value).strip()
        ]
        representative_ids = [
            str(value).strip()
            for value in list(metadata_before.get("representative_trajectory_ids") or [])
            if str(value).strip()
        ]
        force_synthesized = (
            not self._metadata_has_evidence(metadata_before)
            or bool(source_ids and set(source_ids) != set(row_ids))
            or bool(metadata_before.get("wiki_evidence_trajectory_count") not in {None, len(rows)})
            or bool(representative_ids and not set(representative_ids).issubset(set(row_ids)))
        )
        metadata = self._merge_evidence_metadata(
            metadata_before,
            rows,
            match_source=str(metadata_before.get("wiki_seed_match_source") or match_source),
            force_synthesized=force_synthesized,
        )
        if metadata == metadata_before:
            return draft
        entities = list(draft.entities) or list(self._trajectory_evidence_metadata(rows).get("entity_mentions") or [])
        return WikiPageDraft(
            page_type=draft.page_type,
            title=draft.title,
            slug=draft.slug,
            trajectory_ids=list(dict.fromkeys(draft.trajectory_ids)),
            entities=_dedupe_preserve(entities),
            linked_slugs=list(draft.linked_slugs),
            metadata=metadata,
        )

    @staticmethod
    def _is_placeholder_description(text: str) -> bool:
        stripped = collapse_whitespace(text).strip(" .:-")
        return not stripped or bool(_WIKI_PLACEHOLDER_RE.search(stripped))

    @classmethod
    def _section_has_substantive_bullet(cls, markdown_text: str, heading: str) -> bool:
        section = cls._extract_markdown_section(markdown_text, heading)
        for line in section.splitlines()[1:]:
            stripped = collapse_whitespace(line).strip()
            if not stripped.startswith("-"):
                continue
            value = stripped.lstrip("-").strip()
            if not cls._is_placeholder_description(value):
                return True
        return False

    @classmethod
    def _page_markdown_has_substantive_content(cls, markdown_text: str, draft: WikiPageDraft) -> bool:
        if draft.page_type == "index" or not draft.trajectory_ids:
            return True
        return cls._section_has_substantive_bullet(
            markdown_text,
            "## Key Facts",
        ) or cls._section_has_substantive_bullet(
            markdown_text,
            "## Items / Counts",
        )

    @classmethod
    def _short_linked_description(cls, text: str, *, limit: int = 180) -> str:
        collapsed = cls._sanitize_linked_description_text(text)
        if cls._is_placeholder_description(collapsed):
            return ""
        sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", collapsed)
        if sentence_match:
            collapsed = sentence_match.group(1)
        if len(collapsed) <= limit:
            return collapsed
        shortened = collapsed[:limit].rsplit(" ", 1)[0].strip(" ,;:")
        return f"{shortened}..." if shortened else ""

    @staticmethod
    def _sanitize_linked_description_text(text: str) -> str:
        collapsed = collapse_whitespace(text)
        collapsed = re.sub(r"#+\s*", "", collapsed)
        parts = [
            part.strip(" -:;")
            for part in re.split(r"\s+-\s+|[;\n\r]+", collapsed)
            if part.strip(" -:;")
        ]
        cleaned_parts: list[str] = []
        for part in parts:
            folded = part.casefold()
            if folded in {
                "profile / stable facts",
                "current update",
                "conflicts / uncertainty",
                "historical evidence",
                "summary",
            }:
                continue
            if folded.startswith(
                (
                    "trajectory label:",
                    "identity_summary=",
                    "recent_update=",
                    "historical_item_terms=",
                    "facet_values=",
                    "entity_mentions=",
                    "source_anchors=",
                    "card ",
                )
            ):
                continue
            # Drop internal slug-like labels while preserving natural facts.
            if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+){2,}", folded):
                continue
            cleaned_parts.append(part)
        return collapse_whitespace(" - ".join(cleaned_parts) if cleaned_parts else collapsed)

    @classmethod
    def _evidence_card_linked_description(cls, card: dict[str, Any]) -> str:
        candidates = [
            str(card.get("recent_update") or ""),
            "; ".join(str(value) for value in list(card.get("display_key_facts") or [])[:4] if str(value).strip()),
            ", ".join(
                str(value)
                for value in sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24)[:6]
                if str(value).strip()
            ),
            str(card.get("identity_summary") or ""),
        ]
        for candidate in candidates:
            description = cls._short_linked_description(candidate)
            if description:
                return description
        return ""

    def _linked_trajectory_descriptions(
        self,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> dict[str, str]:
        linked_ids = set(draft.trajectory_ids)
        descriptions: dict[str, str] = {}
        for card in list((draft.metadata or {}).get("trajectory_evidence_cards") or [])[: self.MAX_REPRESENTATIVE_SUMMARIES]:
            if not isinstance(card, dict):
                continue
            trajectory_id = str(card.get("trajectory_id") or "").strip()
            if not trajectory_id or trajectory_id not in linked_ids:
                continue
            description = self._evidence_card_linked_description(card)
            if description:
                descriptions[trajectory_id] = description
        for trajectory_id in self._draft_metadata_list(draft, "representative_trajectory_ids")[: self.MAX_REPRESENTATIVE_SUMMARIES]:
            if trajectory_id not in linked_ids:
                continue
            trajectory = trajectories_by_id.get(trajectory_id)
            if trajectory is None:
                continue
            if trajectory_id in descriptions:
                continue
            description = self._short_linked_description(self._trajectory_routing_summary_text(trajectory))
            if description:
                descriptions[trajectory_id] = description
        return descriptions

    def _linked_trajectory_section_stats(
        self,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> dict[str, int | bool]:
        descriptions = self._linked_trajectory_descriptions(draft, trajectories_by_id)
        described_count = sum(1 for trajectory_id in draft.trajectory_ids if trajectory_id in descriptions)
        undescribed_count = max(len(draft.trajectory_ids) - described_count, 0)
        return {
            "linked_trajectory_description_count": described_count,
            "linked_trajectory_undescribed_count": undescribed_count,
            "linked_trajectory_undescibed_count": undescribed_count,
            "linked_trajectory_section_rendered_deterministically": True,
        }

    @staticmethod
    def _chunk_ids(values: list[str], *, chunk_size: int = 12) -> list[list[str]]:
        return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]

    def _render_linked_trajectory_section(
        self,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> str:
        trajectory_ids = list(dict.fromkeys(draft.trajectory_ids))
        descriptions = self._linked_trajectory_descriptions(draft, trajectories_by_id)
        if not trajectory_ids:
            return f"{_LINKED_TRAJECTORIES_HEADING}\n- No linked trajectories."
        if draft.page_type == "index":
            lines = [
                f"- Total linked trajectories: {len(trajectory_ids)}",
            ]
            described_ids = [trajectory_id for trajectory_id in trajectory_ids if trajectory_id in descriptions]
            if described_ids:
                lines.append("- Representative trajectories:")
                lines.extend(f"  - {trajectory_id}: {descriptions[trajectory_id]}" for trajectory_id in described_ids)
            lines.append("- All linked trajectory ids:")
            lines.extend(f"  - {', '.join(chunk)}" for chunk in self._chunk_ids(trajectory_ids))
            return f"{_LINKED_TRAJECTORIES_HEADING}\n" + "\n".join(lines)
        lines = [
            f"- {trajectory_id}: {descriptions[trajectory_id]}" if trajectory_id in descriptions else f"- {trajectory_id}"
            for trajectory_id in trajectory_ids
        ]
        return f"{_LINKED_TRAJECTORIES_HEADING}\n" + "\n".join(lines)

    @staticmethod
    def _extract_markdown_section(markdown_text: str, heading: str) -> str:
        pattern = re.compile(rf"(?ms)^{re.escape(heading)}\n.*?(?=^## [^\n]+|\Z)")
        match = pattern.search(markdown_text)
        return match.group(0) if match else ""

    @classmethod
    def _linked_section_has_placeholder(cls, markdown_text: str) -> bool:
        section = cls._extract_markdown_section(markdown_text, _LINKED_TRAJECTORIES_HEADING)
        return bool(section and _WIKI_PLACEHOLDER_RE.search(section))

    def _rewrite_linked_trajectory_section(
        self,
        markdown_text: str,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> str:
        deterministic_section = self._render_linked_trajectory_section(draft, trajectories_by_id)
        pattern = re.compile(rf"(?ms)^{re.escape(_LINKED_TRAJECTORIES_HEADING)}\n.*?(?=^## [^\n]+|\Z)")
        match = pattern.search(markdown_text)
        if not match:
            return f"{markdown_text.rstrip()}\n\n{deterministic_section}"
        return (
            markdown_text[: match.start()].rstrip()
            + "\n\n"
            + deterministic_section
            + "\n\n"
            + markdown_text[match.end() :].lstrip()
        ).strip()

    def _representative_summary_sections(
        self,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> list[str]:
        sections: list[str] = []
        representative_ids = self._draft_metadata_list(draft, "representative_trajectory_ids")[: self.MAX_REPRESENTATIVE_SUMMARIES]
        if not representative_ids and draft.trajectory_ids:
            rows = [
                trajectories_by_id[trajectory_id]
                for trajectory_id in list(dict.fromkeys(draft.trajectory_ids))
                if trajectory_id in trajectories_by_id
            ]
            representative_ids = self._choose_representative_trajectory_ids(
                rows,
                limit=min(self.MAX_REPRESENTATIVE_SUMMARIES, len(rows)),
            )
        for trajectory_id in representative_ids:
            trajectory = trajectories_by_id.get(trajectory_id)
            if trajectory is None:
                continue
            sections.append(f"### {trajectory_id}\n{self._trajectory_routing_summary_text(trajectory) or 'None.'}")
        return sections

    def _trajectory_evidence_card_sections(
        self,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
        *,
        routing_facing: bool = False,
    ) -> list[str]:
        cards = [
            card
            for card in list((draft.metadata or {}).get("trajectory_evidence_cards") or [])
            if isinstance(card, dict)
        ]
        if not cards:
            representative_ids = self._draft_metadata_list(draft, "representative_trajectory_ids")[
                : self.MAX_REPRESENTATIVE_SUMMARIES
            ]
            if not representative_ids and draft.trajectory_ids:
                rows = [
                    trajectories_by_id[trajectory_id]
                    for trajectory_id in list(dict.fromkeys(draft.trajectory_ids))
                    if trajectory_id in trajectories_by_id
                ]
                representative_ids = self._choose_representative_trajectory_ids(
                    rows,
                    limit=min(self.MAX_REPRESENTATIVE_SUMMARIES, len(rows)),
                )
            cards = [
                self._trajectory_evidence_card(trajectories_by_id[trajectory_id])
                for trajectory_id in representative_ids
                if trajectory_id in trajectories_by_id
            ]
        if routing_facing:
            return [
                text
                for text in (
                    self._trajectory_routing_evidence_text(card)
                    for card in cards[: self.MAX_REPRESENTATIVE_SUMMARIES]
                )
                if text
            ]
        return [render_trajectory_evidence_card(card) for card in cards[: self.MAX_REPRESENTATIVE_SUMMARIES]]

    def _build_routing_text(
        self,
        draft: WikiPageDraft,
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> str:
        draft = self._ensure_draft_evidence_metadata(draft, trajectories_by_id)
        exact_terms = self._draft_metadata_list(draft, "exact_terms")
        facet_values = self._draft_metadata_list(draft, "facet_values")
        keywords = self._draft_metadata_list(draft, "keywords")
        display_items = self._draft_metadata_list(draft, "display_items")
        display_named_entities = self._draft_metadata_list(draft, "display_named_entities")
        display_counts = self._draft_metadata_list(draft, "display_counts")
        historical_terms = self._draft_metadata_list(draft, "wiki_historical_item_terms")
        representative_sections = self._representative_summary_sections(draft, trajectories_by_id)
        evidence_card_sections = self._trajectory_evidence_card_sections(
            draft,
            trajectories_by_id,
            routing_facing=True,
        )
        routing_priority = str((draft.metadata or {}).get("routing_priority") or "normal")
        lines = [
            f"page_type={draft.page_type}",
            f"title={draft.title}",
            f"routing_priority={routing_priority}",
            f"entities={', '.join(draft.entities) or 'none'}",
            f"exact_terms={', '.join(exact_terms) or 'none'}",
            f"facet_values={', '.join(facet_values) or 'none'}",
            f"display_items={', '.join(display_items) or 'none'}",
            f"display_named_entities={', '.join(display_named_entities) or 'none'}",
            f"display_counts={', '.join(display_counts) or 'none'}",
            f"historical_terms={', '.join(historical_terms) or 'none'}",
            f"keywords={', '.join(keywords) or 'none'}",
            f"trajectory_ids={', '.join(draft.trajectory_ids) or 'none'}",
            "representative_summaries:",
        ]
        if representative_sections:
            lines.extend(representative_sections)
        else:
            lines.append("- none")
        lines.append("trajectory_evidence_cards:")
        if evidence_card_sections:
            lines.extend(evidence_card_sections)
        else:
            lines.append("- none")
        return "\n".join(lines).strip()

    def _fallback_page_markdown(
        self,
        draft: WikiPageDraft,
        trajectory_rows: list[TrajectoryRecord],
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> str:
        draft = self._ensure_draft_evidence_metadata(draft, trajectories_by_id)
        summaries = [
            self._trajectory_routing_summary_text(trajectory)
            for trajectory in trajectory_rows
            if self._trajectory_routing_summary_text(trajectory)
        ]
        display_values = clean_readable_values(
            [
                *self._draft_metadata_list(draft, "display_items"),
                *self._draft_metadata_list(draft, "display_counts"),
                *self._draft_metadata_list(draft, "display_key_facts"),
                *self._draft_metadata_list(draft, "wiki_historical_item_terms"),
            ],
            allow_single_word=True,
            limit=20,
        )
        linked_section = self._render_linked_trajectory_section(draft, trajectories_by_id)
        fallback_items = display_values or [
            summary
            for summary in summaries[:6]
            if summary
        ] or list(dict.fromkeys(draft.trajectory_ids))[:6]
        items = "\n".join(f"- {value}" for value in fallback_items) or "- No displayable items."
        representative_sections = self._representative_summary_sections(draft, trajectories_by_id)
        evidence_card_sections = self._trajectory_evidence_card_sections(draft, trajectories_by_id)
        facts = "\n".join(f"- {section.splitlines()[-1]}" for section in representative_sections if section) or (
            "\n".join(f"- {summary}" for summary in summaries[:12]) if summaries else ""
        )
        if evidence_card_sections:
            facts = "\n".join(
                f"- {line}"
                for section in evidence_card_sections
                for line in section.splitlines()
                if line.startswith("historical_item_terms=") or line.startswith("source_anchors=")
            ) or facts
        if not facts:
            facts = "\n".join(f"- Linked trajectory: {trajectory_id}" for trajectory_id in list(dict.fromkeys(draft.trajectory_ids))[:12])
        if not facts:
            facts = "- No linked trajectory facts available."
        conflicts = "\n".join(
            f"- {line}"
            for line in _dedupe_preserve(
                [
                    str(value)
                    for trajectory in trajectory_rows
                    for value in list((trajectory.metadata_json or {}).get("conflict_lines") or [])
                ]
            )[:10]
        ) or "- None"
        return (
            "## Overview\n"
            f"- {draft.title} collects {len(trajectory_rows)} linked trajectories.\n\n"
            "## Key Facts\n"
            f"{facts}\n\n"
            "## Items / Counts\n"
            f"{items}\n\n"
            f"{linked_section}\n\n"
            "## Conflicts / Uncertainty\n"
            f"{conflicts}"
        )

    def _compile_page_markdown(
        self,
        sample_id: str,
        draft: WikiPageDraft,
        trajectory_rows: list[TrajectoryRecord],
        trajectories_by_id: dict[str, TrajectoryRecord],
    ) -> str:
        draft = self._ensure_draft_evidence_metadata(draft, trajectories_by_id)
        representative_sections = self._representative_summary_sections(draft, trajectories_by_id)
        evidence_card_sections = self._trajectory_evidence_card_sections(draft, trajectories_by_id)
        exact_terms = self._draft_metadata_list(draft, "exact_terms")
        facet_values = self._draft_metadata_list(draft, "facet_values")
        historical_terms = self._draft_metadata_list(draft, "wiki_historical_item_terms")
        display_items = self._draft_metadata_list(draft, "display_items")
        display_named_entities = self._draft_metadata_list(draft, "display_named_entities")
        display_counts = self._draft_metadata_list(draft, "display_counts")
        display_key_facts = self._draft_metadata_list(draft, "display_key_facts")
        omitted_summaries = max(len(draft.trajectory_ids) - len(representative_sections), 0)
        self._trace(
            f"sample={sample_id} wiki_page_compile_start slug={draft.slug} type={draft.page_type} "
            f"trajectories={len(trajectory_rows)} representatives={len(representative_sections)}"
        )
        prompt = (
            load_prompt("wiki_page_compile")
            + "\n\nPage type:\n"
            + draft.page_type
            + "\n\nPage title:\n"
            + draft.title
            + "\n\nLinked trajectory ids:\n"
            + (", ".join(draft.trajectory_ids) or "none")
            + "\n\nDominant entities:\n"
            + (", ".join(draft.entities) or "none")
            + "\n\nDominant exact terms:\n"
            + (", ".join(exact_terms) or "none")
            + "\n\nDominant facet values:\n"
            + (", ".join(facet_values) or "none")
            + "\n\nHistorical item/place/event/count terms:\n"
            + (", ".join(historical_terms) or "none")
            + "\n\nReadable display items:\n"
            + (", ".join(display_items) or "none")
            + "\n\nReadable named entities:\n"
            + (", ".join(display_named_entities) or "none")
            + "\n\nReadable counts:\n"
            + (", ".join(display_counts) or "none")
            + "\n\nReadable key facts:\n"
            + ("\n".join(f"- {fact}" for fact in display_key_facts) if display_key_facts else "- none")
            + "\n\nRepresentative trajectory summaries:\n"
            + ("\n\n".join(representative_sections) if representative_sections else "- none")
            + "\n\nTrajectory evidence cards:\n"
            + ("\n\n".join(evidence_card_sections) if evidence_card_sections else "- none")
        )
        self._trace(
            f"sample={sample_id} wiki_page_compile_context slug={draft.slug} linked_trajectories={len(draft.trajectory_ids)} "
            f"representatives={len(representative_sections)} evidence_cards={len(evidence_card_sections)} "
            f"omitted={omitted_summaries} prompt_chars={len(prompt)}"
        )
        started_at = time.perf_counter()
        try:
            response = self.llm_provider.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata={
                    "task": "wiki_page_compile",
                    "page_type": draft.page_type,
                    "trajectory_count": len(trajectory_rows),
                    "representative_summary_count": len(representative_sections),
                },
            )
            markdown_text = response.text.replace("\r\n", "\n").strip()
            if not collapse_whitespace(markdown_text):
                draft.metadata["wiki_placeholder_fallback_used"] = False
                draft.metadata["wiki_compile_fallback_reason"] = "empty_response"
                self._trace(
                    f"sample={sample_id} wiki_page_compile_invalid slug={draft.slug} reason=empty_response "
                    f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f} fallback=deterministic_markdown"
                )
                return self._fallback_page_markdown(draft, trajectory_rows, trajectories_by_id)
            missing_headings = self._validate_page_markdown(markdown_text)
            if missing_headings:
                draft.metadata["wiki_placeholder_fallback_used"] = False
                draft.metadata["wiki_compile_fallback_reason"] = "missing_headings"
                self._trace(
                    f"sample={sample_id} wiki_page_compile_invalid slug={draft.slug} "
                    f"reason=missing_headings missing={','.join(missing_headings)} "
                    f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f} fallback=deterministic_markdown"
                )
                return self._fallback_page_markdown(draft, trajectory_rows, trajectories_by_id)
            if not self._page_markdown_has_substantive_content(markdown_text, draft):
                draft.metadata["wiki_placeholder_fallback_used"] = True
                draft.metadata["wiki_compile_fallback_reason"] = "placeholder_content"
                self._trace(
                    f"sample={sample_id} wiki_page_compile_invalid slug={draft.slug} "
                    f"reason=placeholder_content latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f} "
                    "fallback=deterministic_markdown"
                )
                return self._fallback_page_markdown(draft, trajectory_rows, trajectories_by_id)
            placeholder_detected = self._linked_section_has_placeholder(markdown_text)
            markdown_text = self._rewrite_linked_trajectory_section(markdown_text, draft, trajectories_by_id)
            linked_stats = self._linked_trajectory_section_stats(draft, trajectories_by_id)
            self._trace(
                f"sample={sample_id} wiki_page_linked_section_rewritten slug={draft.slug} "
                f"linked={len(draft.trajectory_ids)} described={linked_stats['linked_trajectory_description_count']} "
                f"placeholder_detected={str(placeholder_detected).lower()}"
            )
            draft.metadata["wiki_placeholder_fallback_used"] = False
            draft.metadata.pop("wiki_compile_fallback_reason", None)
            self._trace(
                f"sample={sample_id} wiki_page_compile_llm_ok slug={draft.slug} chars={len(markdown_text)} "
                f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
            )
            return markdown_text
        except Exception as exc:  # noqa: BLE001
            self._trace(
                f"sample={sample_id} wiki_page_compile_failed slug={draft.slug} error={exc.__class__.__name__} "
                f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f} fallback=deterministic_markdown"
            )
        return self._fallback_page_markdown(draft, trajectory_rows, trajectories_by_id)

    def compile_sample(self, sample_id: str, dataset_name: str) -> list[WikiPageRecord]:
        raw_trajectories = self.store.list_trajectories(sample_id)
        trajectories = [
            trajectory
            for trajectory in raw_trajectories
            if not self._is_low_salience_noise_trajectory(trajectory)
        ]
        suppressed = len(raw_trajectories) - len(trajectories)
        if suppressed:
            self._trace(f"sample={sample_id} wiki_low_salience_trajectories_suppressed count={suppressed}")
        trajectories_by_id = self._trajectory_rows_by_id(trajectories)
        started_at = time.perf_counter()
        self._trace(f"sample={sample_id} wiki_compile_start trajectories={len(trajectories)}")
        seeds = self._plan_seeds(sample_id, trajectories)
        drafts = self._plan_pages(sample_id, seeds, trajectories)
        self._trace(f"sample={sample_id} wiki_plan_done pages={len(drafts)}")
        pages: list[WikiPageRecord] = []
        slug_to_id: dict[str, str] = {}
        for index, draft in enumerate(drafts, start=1):
            slug_to_id[draft.slug] = wiki_page_id(sample_id, draft.page_type, index)
        for index, draft in enumerate(drafts, start=1):
            page_id = wiki_page_id(sample_id, draft.page_type, index)
            draft = self._ensure_draft_evidence_metadata(draft, trajectories_by_id)
            trajectory_rows = [trajectories_by_id[trajectory_id] for trajectory_id in draft.trajectory_ids if trajectory_id in trajectories_by_id]
            page_started = time.perf_counter()
            markdown_text = self._compile_page_markdown(sample_id, draft, trajectory_rows, trajectories_by_id)
            markdown_ms = (time.perf_counter() - page_started) * 1000.0
            routing_started = time.perf_counter()
            routing_text = self._build_routing_text(draft, trajectories_by_id)
            raw_evidence_probe = "\n".join(
                self._trajectory_evidence_card_sections(draft, trajectories_by_id, routing_facing=False)
            )
            raw_marker_hits = self._routing_text_internal_marker_hits(raw_evidence_probe)
            routing_marker_hits = self._routing_text_internal_marker_hits(routing_text)
            removed_markers = sorted(set(raw_marker_hits) - set(routing_marker_hits))
            keywords = _top_ranked_values(
                [
                    *extract_keywords(draft.title),
                    *draft.entities,
                    *self._draft_metadata_list(draft, "exact_terms"),
                    *self._draft_metadata_list(draft, "facet_values"),
                    *self._draft_metadata_list(draft, "display_items"),
                    *self._draft_metadata_list(draft, "display_named_entities"),
                    *self._draft_metadata_list(draft, "display_counts"),
                    *self._draft_metadata_list(draft, "wiki_historical_item_terms"),
                    *self._draft_metadata_list(draft, "keywords"),
                ],
                limit=64,
            )
            routing_ms = (time.perf_counter() - routing_started) * 1000.0
            self._trace(
                f"sample={sample_id} wiki_routing_text_cleaned slug={draft.slug} "
                f"removed_markers={len(removed_markers)} chars_before={len(raw_evidence_probe)} "
                f"chars_after={len(routing_text)}"
            )
            if routing_marker_hits:
                self._trace(
                    f"sample={sample_id} wiki_routing_text_internal_marker_leak "
                    f"slug={draft.slug} markers={','.join(routing_marker_hits)}"
                )
            self._trace(
                f"sample={sample_id} wiki_routing_text_ready slug={draft.slug} chars={len(routing_text)} "
                f"keywords={len(keywords)} representatives={len(self._draft_metadata_list(draft, 'representative_trajectory_ids'))}"
            )
            self._trace(
                f"sample={sample_id} wiki_display_signals_ready slug={draft.slug} "
                f"items={len(self._draft_metadata_list(draft, 'display_items'))} "
                f"entities={len(self._draft_metadata_list(draft, 'display_named_entities'))} "
                f"counts={len(self._draft_metadata_list(draft, 'display_counts'))} "
                f"facts={len(self._draft_metadata_list(draft, 'display_key_facts'))}"
            )
            embed_started = time.perf_counter()
            vector = self._embed_documents([routing_text])[0]
            embedding_id = f"{page_id}-emb"
            self.store.save_embedding(
                embedding_id=embedding_id,
                owner_type="wiki_page",
                owner_id=page_id,
                model_name=self.embedding_provider.model_info().model_name,
                vector=vector,
                semantic_text=routing_text,
                metadata={"document_embedding_strategy": self._document_embedding_strategy()},
            )
            embed_ms = (time.perf_counter() - embed_started) * 1000.0
            page = WikiPageRecord(
                id=page_id,
                sample_id=sample_id,
                dataset_name=dataset_name,
                page_type=draft.page_type,
                title=draft.title,
                slug=draft.slug,
                markdown_text=markdown_text,
                keywords_json=keywords,
                trajectory_ids_json=list(dict.fromkeys(draft.trajectory_ids)),
                linked_page_ids_json=[slug_to_id[slug] for slug in draft.linked_slugs if slug in slug_to_id and slug_to_id[slug] != page_id],
                entity_names_json=list(dict.fromkeys(draft.entities)),
                embedding_id=embedding_id,
                metadata_json={
                    **dict(draft.metadata or {}),
                    "routing_text": routing_text,
                    "routing_text_internal_marker_count": len(routing_marker_hits),
                    "routing_text_cleaned": True,
                    "routing_text_cleaning_removed_markers": removed_markers,
                    "routing_text_internal_marker_leaks": routing_marker_hits,
                    "linked_page_slugs": list(dict.fromkeys(draft.linked_slugs)),
                    "page_type": draft.page_type,
                    "signal_source": "llm_validated_display_fields",
                    **self._linked_trajectory_section_stats(draft, trajectories_by_id),
                },
            )
            pages.append(page)
            self._trace(
                f"sample={sample_id} wiki_page_compile_done page_id={page.id} type={page.page_type} "
                f"trajectories={len(page.trajectory_ids_json)} markdown_ms={markdown_ms:.1f} "
                f"routing_ms={routing_ms:.1f} embed_ms={embed_ms:.1f} total_ms={(time.perf_counter() - page_started) * 1000.0:.1f}"
            )
        non_index_page_trajectory_ids = {
            trajectory_id
            for page in pages
            if page.page_type != "index"
            for trajectory_id in list(page.trajectory_ids_json or [])
        }
        index_only_after_compile = [
            trajectory_id
            for trajectory_id in trajectories_by_id
            if trajectory_id not in non_index_page_trajectory_ids
        ]
        if index_only_after_compile:
            self._trace(
                f"sample={sample_id} wiki_non_index_coverage_incomplete "
                f"stage=compile_done index_only={len(index_only_after_compile)} "
                f"ids={','.join(index_only_after_compile[:12])}"
            )
        self.store.replace_wiki_pages_for_sample(sample_id, pages)
        self.store.session.flush()
        self._trace(
            f"sample={sample_id} wiki_compile_done pages={len(pages)} latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
        )
        return pages
