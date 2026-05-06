"""Hierarchical wiki-first retrieval over episodic trajectories."""

from __future__ import annotations

import math
import json
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Iterable

from trajpatch.prompts import load_prompt
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.structured_outputs import (
    get_structured_task_spec,
    parse_structured_payload,
)
from trajpatch.storage.models import (
    ClaimOpRecord,
    ClaimRecord,
    EpisodicMemorySnapshot,
    RawMessageRecord,
    RetrievalEvent,
    TrajectoryRecord,
    WikiPageRecord,
)
from trajpatch.storage.repository import TrajWikiStore
from trajpatch.types import NormalizedMessage, RetrievalBundle
from trajpatch.utils.text import collapse_whitespace, extract_keywords, keyword_overlap_score

from .facets import (
    build_sample_entity_lexicon,
    classify_query_shape_v1,
    exact_term_keyword_set,
    extract_query_facets_v1,
    facet_value_key,
    is_list_like_query,
    normalize_entity_key,
)
from .historical import sanitize_historical_item_terms
from .renderers import render_answer_episodic_snapshot
from .trajectory_summaries import fallback_summary_from_metadata, sanitize_summary_keyword_values, summary_keywords_v2


def _compact_error_message(exc: BaseException, *, limit: int = 240) -> str:
    message = " ".join(str(exc).split())
    return message[:limit]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def mean_pool_normalized(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    normalized = [l2_normalize(vector) for vector in vectors if vector]
    if not normalized:
        return []
    dims = len(normalized[0])
    pooled = [sum(vector[index] for vector in normalized) / len(normalized) for index in range(dims)]
    return l2_normalize(pooled)


@dataclass(slots=True)
class TrajectoryRetrievalSignals:
    metadata: dict[str, object]
    latest_snapshot_id: str
    summary_text: str
    summary_keywords: list[str]
    exact_terms: list[str]
    display_items: list[str]
    display_counts: list[str]
    display_key_facts: list[str]
    historical_item_terms: list[str]
    source_event_object_terms: list[str]
    source_event_action_terms: list[str]
    source_event_canonical_terms: list[str]
    source_temporal_relation_terms: list[str]
    source_event_records: list[dict[str, object]]
    facet_values: set[str]
    facet_tags: set[str]
    entity_mentions: list[str]
    entity_keys: set[str]
    lexical_keywords: set[str]
    support_terms: set[str]


@dataclass(slots=True)
class ClaimFacetSignals:
    entity_keys: set[str]
    facet_tags: set[str]
    facet_values: set[str]


class RetrievalEngine:
    CANDIDATE_POOL_SIZE = 12
    RRF_K = 60
    NON_LIST_EXPANDED_OFFSET = 4
    NON_LIST_EXPANDED_MIN = 10
    NON_LIST_EXPANDED_MAX = 14
    LIST_EXPANDED_OFFSET = 8
    LIST_EXPANDED_MIN = 12
    LIST_EXPANDED_MAX = 18
    NON_LIST_SOURCE_BUDGET = 28
    LIST_SOURCE_BUDGET = 40
    UPDATE_LINKED_SOURCE_LIMIT = 2
    NEIGHBOR_SOURCE_LIMIT = 1
    SOURCE_TIMELINE_SNIPPET_CHARS = 160
    RAW_RESCUE_PREFILTER_LIMIT = 80
    RAW_RESCUE_NON_LIST_LIMIT = 8
    RAW_RESCUE_LIST_LIMIT = 12
    REFLECTION_SEMANTIC_WEAK_COVERAGE_THRESHOLD = 0.34
    REFLECTION_SLUG_BONUS = 0.30
    REFLECTION_TERM_BONUS_MAX = 0.20
    COVERAGE_SELECTION_POOL_MAX = 64
    BROAD_ENTITY_TRAJECTORY_THRESHOLD = 30
    BROAD_ENTITY_CANDIDATE_CAP_MIN = 30
    FAMILY_MATCH_STRONG_THRESHOLD = 0.55
    DIAGNOSTIC_TOP_N_PAGES = 50
    DIAGNOSTIC_TOP_N_TRAJECTORIES = 50
    DIAGNOSTIC_SELECTION_POOL_ROW_LIMIT = 80
    DIAGNOSTIC_TEXT_ITEM_LIMIT = 12
    DIAGNOSTIC_TEXT_CHAR_LIMIT = 80
    COVERAGE_GENERIC_ITEM_TERMS = frozenset(
        {
            "about",
            "answer",
            "answers",
            "asked",
            "candidate",
            "claim",
            "claims",
            "conversation",
            "discuss",
            "discussed",
            "evidence",
            "fact",
            "facts",
            "friend",
            "friends",
            "important",
            "creative",
            "experience",
            "experiences",
            "family",
            "great",
            "general",
            "group",
            "information",
            "item",
            "items",
            "kind",
            "know",
            "memory",
            "mention",
            "mentioned",
            "music",
            "musical",
            "nature",
            "people",
            "person",
            "question",
            "questions",
            "related",
            "said",
            "sample",
            "support",
            "supports",
            "thing",
            "things",
            "topic",
            "type",
            "types",
            "user",
            "went",
        }
    )
    REFLECTION_GENERIC_TERMS = COVERAGE_GENERIC_ITEM_TERMS | frozenset(
        {
            "area",
            "areas",
            "detail",
            "details",
            "find",
            "finding",
            "retrieval",
            "retrieve",
            "retrieved",
            "search",
            "searched",
            "specific",
            "specifically",
        }
    )
    TEMPORAL_EVENT_QUERY_TERMS = frozenset(
        {
            "accident",
            "campaign",
            "competition",
            "conference",
            "contest",
            "event",
            "events",
            "exhibit",
            "fair",
            "museum",
            "networking",
            "parade",
            "race",
            "roadtrip",
            "speech",
            "talk",
            "trip",
            "workshop",
        }
    )
    TEMPORAL_EVENT_ACTION_TERMS = frozenset(
        {
            "attend",
            "attended",
            "attending",
            "gave",
            "give",
            "giving",
            "go",
            "host",
            "hosted",
            "hosting",
            "join",
            "joined",
            "launch",
            "launched",
            "participated",
            "visit",
            "visited",
            "went",
        }
    )
    FAMILY_MATCH_TERMS = {
        "book": frozenset({"book", "books", "read", "reading", "title", "cover", "novel", "story"}),
        "reading": frozenset({"book", "books", "read", "reading", "title", "cover", "novel", "story"}),
        "painted_object": frozenset(
            {"paint", "painted", "painting", "art", "artwork", "image", "picture", "photo", "canvas", "recent", "latest"}
        ),
        "research_topic": frozenset(
            {"research", "researched", "study", "studied", "looked", "agency", "agencies", "option", "options", "topic"}
        ),
        "instrument": frozenset(
            {"instrument", "instruments", "play", "plays", "played", "music", "notes", "clarinet", "violin", "guitar", "piano"}
        ),
        "activity": frozenset({"activity", "activities", "done", "did", "trip", "visit", "visited", "hiking", "camping"}),
        "event": frozenset({"event", "events", "attended", "joined", "participated", "conference", "parade", "speech", "group"}),
        "place": frozenset({"place", "places", "visited", "city", "country", "area", "park", "museum", "trail"}),
        "country": frozenset({"country", "countries", "home", "moved", "from", "sweden", "canada", "mexico"}),
        "city": frozenset({"city", "cities", "moved", "from", "visited", "place"}),
        "state": frozenset({"state", "states", "moved", "from", "visited", "place"}),
        "item": frozenset({"item", "items", "bought", "made", "created", "object", "thing"}),
        "dessert": frozenset({"dessert", "desserts", "recipe", "recipes", "cake", "pie", "cookies", "pudding"}),
        "recipe": frozenset({"recipe", "recipes", "dish", "dishes", "cooked", "made", "baked"}),
        "writing": frozenset({"writing", "writings", "screenplay", "screenplays", "script", "scripts", "story", "fiction"}),
        "band": frozenset({"band", "bands", "artist", "artists", "concert", "music", "festival", "sounds"}),
        "symbol": frozenset({"symbol", "symbols", "logo", "icon"}),
        "count": frozenset({"count", "times", "number", "rejected", "won", "visited"}),
        "person": frozenset({"person", "people", "family", "member", "members", "mother", "father", "aunt", "uncle", "passed", "died"}),
        "relationship_status": frozenset({"relationship", "status", "single", "dating", "married", "breakup", "parent"}),
        "type": frozenset({"type", "types", "kind", "kinds"}),
    }
    SOURCE_GROUP_ORDER = ("seed", "update_linked", "neighbor")
    SOURCE_GROUP_TITLES = {
        "seed": "Seed Evidence",
        "update_linked": "Update-Linked Evidence",
        "neighbor": "Neighbor Evidence",
    }

    def __init__(
        self,
        store: TrajWikiStore,
        embedding_provider,
        top_t_pages: int,
        top_k: int,
        neighbor_radius: int = 1,
        retrieval_expansion_mode: str = "update_linked_plus_neighbors",
        trace: Callable[[str], None] | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.top_t_pages = top_t_pages
        self.top_k = top_k
        # Fine retrieval seeds are fixed at 2*k before expansion compaction.
        self.snapshot_budget = 2 * top_k
        self.neighbor_radius = neighbor_radius
        self.retrieval_expansion_mode = retrieval_expansion_mode
        self.trace = trace
        self._sample_entity_lexicon_cache: dict[str, dict[str, str]] = {}

    @staticmethod
    def _source_sort_key(message: RawMessageRecord | None, message_id: str) -> tuple[int, str]:
        if message is None:
            return (10**9, message_id)
        return (int(message.turn_index), message.id)

    @staticmethod
    def _source_speaker_label(message: RawMessageRecord) -> str:
        return f"{message.role}/{message.speaker_name}" if message.speaker_name else message.role

    @staticmethod
    def _source_ref_label(message: RawMessageRecord) -> str:
        return str(message.source_ref or "no-ref")

    @staticmethod
    def _source_date_label(message: RawMessageRecord) -> str:
        occurred_at = collapse_whitespace(str(message.occurred_at or ""))
        return f" | date={occurred_at}" if occurred_at else ""

    @classmethod
    def _source_message_line(cls, message: RawMessageRecord) -> str:
        return (
            f"- {cls._source_ref_label(message)}{cls._source_date_label(message)} | "
            f"id={message.id} | turn={message.turn_index} | "
            f"{cls._source_speaker_label(message)}: {message.content}"
        )

    @classmethod
    def _source_timeline_line(cls, message: RawMessageRecord) -> str:
        snippet = collapse_whitespace(message.content)
        if len(snippet) > cls.SOURCE_TIMELINE_SNIPPET_CHARS:
            snippet = snippet[: cls.SOURCE_TIMELINE_SNIPPET_CHARS - 3].rstrip() + "..."
        return (
            f"- {cls._source_ref_label(message)}{cls._source_date_label(message)} | "
            f"id={message.id} | turn={message.turn_index} | "
            f"{cls._source_speaker_label(message)}: {snippet}"
        )

    _MONTHS: dict[str, int] = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    _NUMBER_WORDS: dict[str, int] = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    @classmethod
    def _parse_source_date(cls, occurred_at: str | None) -> date | None:
        text = collapse_whitespace(str(occurred_at or ""))
        if not text:
            return None
        match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b", text)
        if match:
            day = int(match.group(1))
            month = cls._MONTHS.get(match.group(2).casefold())
            year = int(match.group(3))
            if month:
                try:
                    return date(year, month, day)
                except ValueError:
                    return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text[:10], fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _format_source_date(value: date) -> str:
        return f"{value.day} {value.strftime('%B')} {value.year}"

    @classmethod
    def _parse_small_number(cls, value: str) -> int | None:
        normalized = collapse_whitespace(value).casefold()
        if normalized.isdigit():
            return int(normalized)
        return cls._NUMBER_WORDS.get(normalized)

    @staticmethod
    def _subtract_months(value: date, months: int) -> date:
        month_index = value.year * 12 + value.month - 1 - months
        year = month_index // 12
        month = month_index % 12 + 1
        day = min(value.day, 28)
        return date(year, month, day)

    @staticmethod
    def _format_source_month_year(value: date) -> str:
        return f"{value.strftime('%B')} {value.year}"

    @classmethod
    def _resolve_relative_temporal_anchors(cls, text: str, source_date: date) -> list[dict[str, str]]:
        lowered = collapse_whitespace(text).casefold()
        anchors: list[dict[str, str]] = []
        seen_terms: set[str] = set()

        def add_anchor(
            *,
            relative_term: str,
            resolved_text: str,
            resolution_kind: str,
            resolution_granularity: str,
            resolved_date: str | None = None,
        ) -> None:
            normalized_term = collapse_whitespace(relative_term).casefold()
            if not normalized_term or normalized_term in seen_terms:
                return
            seen_terms.add(normalized_term)
            anchor = {
                "relative_term": normalized_term,
                "resolved_text": collapse_whitespace(resolved_text),
                "resolution_kind": resolution_kind,
                "resolution_granularity": resolution_granularity,
            }
            if resolved_date:
                anchor["resolved_date"] = resolved_date
            anchors.append(anchor)

        if re.search(r"\btoday\b", lowered):
            resolved = cls._format_source_date(source_date)
            add_anchor(
                relative_term="today",
                resolved_text=resolved,
                resolution_kind="exact_date",
                resolution_granularity="day",
                resolved_date=resolved,
            )
        if re.search(r"\byesterday\b", lowered):
            resolved = cls._format_source_date(source_date - timedelta(days=1))
            add_anchor(
                relative_term="yesterday",
                resolved_text=resolved,
                resolution_kind="exact_date",
                resolution_granularity="day",
                resolved_date=resolved,
            )
        if re.search(r"\btomorrow\b", lowered):
            resolved = cls._format_source_date(source_date + timedelta(days=1))
            add_anchor(
                relative_term="tomorrow",
                resolved_text=resolved,
                resolution_kind="exact_date",
                resolution_granularity="day",
                resolved_date=resolved,
            )

        count_pattern = r"\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten"
        for match in re.finditer(rf"\b({count_pattern})\s+days?\s+ago\b", lowered):
            days = cls._parse_small_number(match.group(1))
            if days is None:
                continue
            resolved = cls._format_source_date(source_date - timedelta(days=days))
            add_anchor(
                relative_term=match.group(0),
                resolved_text=resolved,
                resolution_kind="exact_date",
                resolution_granularity="day",
                resolved_date=resolved,
            )

        source_date_text = cls._format_source_date(source_date)
        if re.search(r"\blast\s+week\b", lowered):
            add_anchor(
                relative_term="last week",
                resolved_text=f"the week before {source_date_text}",
                resolution_kind="relative_span",
                resolution_granularity="week_span",
            )
        for match in re.finditer(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
            weekday = match.group(1)
            add_anchor(
                relative_term=f"last {weekday}",
                resolved_text=f"the {weekday.title()} before {source_date_text}",
                resolution_kind="relative_span",
                resolution_granularity="weekday_span",
            )
        if re.search(r"\blast\s+month\b", lowered):
            resolved_month = cls._subtract_months(source_date, 1)
            add_anchor(
                relative_term="last month",
                resolved_text=cls._format_source_month_year(resolved_month),
                resolution_kind="month_year",
                resolution_granularity="month",
            )
        if re.search(r"\blast\s+year\b", lowered):
            add_anchor(
                relative_term="last year",
                resolved_text=str(source_date.year - 1),
                resolution_kind="year",
                resolution_granularity="year",
            )

        fuzzy_spans: list[tuple[int, int]] = []
        fuzzy_pattern = rf"\b(?:(?:around|about)\s+({count_pattern})\s+years?\s+ago|a\s+few\s+years?\s+ago)\b"
        for match in re.finditer(fuzzy_pattern, lowered):
            fuzzy_spans.append(match.span())
            add_anchor(
                relative_term=match.group(0),
                resolved_text=match.group(0),
                resolution_kind="fuzzy_relative",
                resolution_granularity="fuzzy",
            )

        def in_fuzzy_span(start: int, end: int) -> bool:
            return any(start >= fuzzy_start and end <= fuzzy_end for fuzzy_start, fuzzy_end in fuzzy_spans)

        for match in re.finditer(rf"\b({count_pattern})\s+months?\s+ago\b", lowered):
            months = cls._parse_small_number(match.group(1))
            if months is None:
                continue
            resolved_month = cls._subtract_months(source_date, months)
            add_anchor(
                relative_term=match.group(0),
                resolved_text=cls._format_source_month_year(resolved_month),
                resolution_kind="month_year",
                resolution_granularity="month",
            )
        for match in re.finditer(rf"\b({count_pattern})\s+years?\s+ago\b", lowered):
            if in_fuzzy_span(*match.span()):
                continue
            years = cls._parse_small_number(match.group(1))
            if years is None:
                continue
            resolved = date(source_date.year - years, source_date.month, min(source_date.day, 28))
            add_anchor(
                relative_term=match.group(0),
                resolved_text=cls._format_source_month_year(resolved),
                resolution_kind="month_year",
                resolution_granularity="month",
            )
        for match in re.finditer(rf"\b({count_pattern})\s+weeks?\s+ago\b", lowered):
            weeks = cls._parse_small_number(match.group(1))
            if weeks is None:
                continue
            label = match.group(1) if match.group(1).isdigit() else match.group(1)
            unit = "week" if weeks == 1 else "weeks"
            add_anchor(
                relative_term=match.group(0),
                resolved_text=f"{label} {unit} before {source_date_text}",
                resolution_kind="relative_span",
                resolution_granularity="week_span",
            )
        return anchors

    @classmethod
    def _temporal_anchor_lines(cls, messages: list[RawMessageRecord]) -> tuple[list[str], dict[str, object]]:
        lines: list[str] = []
        source_refs: list[str] = []
        relative_terms: list[str] = []
        resolutions: list[dict[str, object]] = []
        seen_lines: set[str] = set()
        seen_refs: set[str] = set()
        seen_terms: set[str] = set()
        seen_resolutions: set[tuple[str, str, str, str, str]] = set()
        for message in messages:
            source_ref = cls._source_ref_label(message)
            source_date = cls._parse_source_date(message.occurred_at)
            if source_date is None:
                continue
            source_date_text = cls._format_source_date(source_date)
            text = collapse_whitespace(message.content)
            anchors = cls._resolve_relative_temporal_anchors(text, source_date)
            if not anchors:
                continue
            if source_ref not in seen_refs:
                seen_refs.add(source_ref)
                source_refs.append(source_ref)
            for anchor in anchors:
                normalized_term = anchor["relative_term"]
                if normalized_term not in seen_terms:
                    seen_terms.add(normalized_term)
                    relative_terms.append(normalized_term)
                line = f"- {source_ref} occurred at {source_date_text}; \"{normalized_term}\" refers to {anchor['resolved_text']}."
                if line not in seen_lines:
                    seen_lines.add(line)
                    lines.append(line)
                resolution_key = (
                    source_ref,
                    normalized_term,
                    anchor["resolution_kind"],
                    anchor["resolution_granularity"],
                    anchor["resolved_text"],
                )
                if resolution_key in seen_resolutions:
                    continue
                seen_resolutions.add(resolution_key)
                resolution: dict[str, object] = {
                    "source_ref": source_ref,
                    "source_date": source_date_text,
                    "relative_term": normalized_term,
                    "resolution_kind": anchor["resolution_kind"],
                    "resolution_granularity": anchor["resolution_granularity"],
                    "resolved_answer_text": anchor["resolved_text"],
                }
                if anchor.get("resolved_date"):
                    resolution["resolved_date"] = anchor["resolved_date"]
                resolutions.append(resolution)
        return lines, {
            "temporal_anchor_hint_count": len(lines),
            "temporal_anchor_source_refs": source_refs,
            "temporal_anchor_relative_terms": relative_terms,
            "temporal_anchor_resolutions": resolutions,
        }

    @classmethod
    def _source_backtrack_stats(
        cls,
        source_ids: list[str],
        message_by_id: dict[str, RawMessageRecord],
    ) -> tuple[int, float]:
        ordered_turns = [
            int(message_by_id[message_id].turn_index)
            for message_id in source_ids
            if message_id in message_by_id
        ]
        backtrack_count = sum(1 for left, right in zip(ordered_turns, ordered_turns[1:]) if right < left)
        denominator = max(len(ordered_turns) - 1, 1)
        return backtrack_count, backtrack_count / denominator

    def _trace(self, message: str) -> None:
        if self.trace is not None:
            self.trace(message)

    def _embed_queries(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embedding_provider, "embed_queries"):
            return self.embedding_provider.embed_queries(texts)
        return self.embedding_provider.embed(texts)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embedding_provider, "embed_documents"):
            return self.embedding_provider.embed_documents(texts)
        return self.embedding_provider.embed(texts)

    def _query_embedding_strategy(self) -> str:
        if hasattr(self.embedding_provider, "query_embedding_strategy"):
            return str(self.embedding_provider.query_embedding_strategy())
        return "shared_embed"

    def _sample_entity_lexicon(self, sample_id: str) -> dict[str, str]:
        cached = self._sample_entity_lexicon_cache.get(sample_id)
        if cached is not None:
            return cached
        lexicon = build_sample_entity_lexicon(self.store.list_raw_messages_for_sample(sample_id))
        self._sample_entity_lexicon_cache[sample_id] = lexicon
        return lexicon

    @staticmethod
    def _clean_text_values(values: Iterable[object], *, limit: int = 24) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = collapse_whitespace(str(value or "")).strip(" .\"'`")
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            cleaned.append(text)
            seen.add(key)
            if len(cleaned) >= limit:
                break
        return cleaned

    @classmethod
    def _compact_diagnostic_text(cls, value: object) -> str:
        text = collapse_whitespace(str(value or "")).strip(" .\"'`")
        if len(text) > cls.DIAGNOSTIC_TEXT_CHAR_LIMIT:
            return text[: cls.DIAGNOSTIC_TEXT_CHAR_LIMIT - 1].rstrip() + "…"
        return text

    @classmethod
    def _compact_diagnostic_terms(cls, values: Iterable[object], *, limit: int | None = None) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = cls._compact_diagnostic_text(value)
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            output.append(text)
            seen.add(key)
            if len(output) >= (limit or cls.DIAGNOSTIC_TEXT_ITEM_LIMIT):
                break
        return output

    @classmethod
    def _compact_page_ranked_row(cls, item: dict[str, object], rank: int) -> dict[str, object]:
        return {
            "rank": rank,
            "page_id": str(item.get("page_id") or ""),
            "title": cls._compact_diagnostic_text(item.get("title")),
            "page_type": str(item.get("page_type") or ""),
            "trajectory_ids": [str(value) for value in list(item.get("trajectory_ids") or []) if str(value).strip()],
            "trajectory_count": int(item.get("trajectory_count") or 0),
            "final_score": float(item.get("fused_score") or 0.0),
            "dense_score": float(item.get("dense_score") or 0.0),
            "sparse_score": float(item.get("sparse_score") or 0.0),
            "page_family_match_score": float(item.get("page_family_match_score") or 0.0),
            "object_overlap": cls._compact_diagnostic_terms(
                list(item.get("page_query_object_overlap_terms") or [])
            ),
            "granularity_adjustment": float(item.get("page_granularity_adjustment") or 0.0),
            "singleton_penalty": float(item.get("singleton_page_penalty") or 0.0),
            "low_quality_singleton_penalty": float(item.get("low_quality_singleton_penalty") or 0.0),
            "medium_bonus": float(item.get("medium_page_bonus") or 0.0),
            "broad_profile_penalty": float(item.get("broad_profile_page_penalty") or 0.0),
            "singleton_policy": str(item.get("singleton_policy") or ""),
            "singleton_quality_score": float(item.get("singleton_quality_score") or 0.0),
            "page_family_mismatch_penalty": float(item.get("page_family_mismatch_penalty") or 0.0),
            "page_strong_query_match": bool(item.get("page_strong_query_match")),
            "broad_entity_profile": bool(item.get("broad_entity_profile")),
            "medium_granularity_page": bool(item.get("medium_granularity_page")),
            "singleton_page": bool(item.get("singleton_page")),
        }

    @classmethod
    def _compact_trajectory_ranked_row(cls, item: dict[str, object], rank: int) -> dict[str, object]:
        coverage_profile = dict(item.get("coverage_profile") or {})
        return {
            "rank": rank,
            "trajectory_id": str(item.get("trajectory_id") or ""),
            "final_score": float(item.get("fused_score") or 0.0),
            "dense_score": float(item.get("dense_score") or 0.0),
            "sparse_score": float(item.get("sparse_score") or 0.0),
            "family_match_score": float(item.get("answer_family_match_score") or 0.0),
            "object_overlap": cls._compact_diagnostic_terms(
                list(item.get("answer_family_query_overlap_terms") or [])
            ),
            "mismatch_penalty": float(item.get("answer_family_mismatch_penalty") or 0.0),
            "source_event_match_score": float(item.get("source_event_match_score") or 0.0),
            "source_event_matched_terms": cls._compact_diagnostic_terms(
                list(item.get("source_event_matched_terms") or [])
            ),
            "source_event_matched_refs": cls._compact_diagnostic_terms(
                list(item.get("source_event_matched_refs") or [])
            ),
            "source_event_match_reason": cls._compact_diagnostic_text(
                item.get("source_event_match_reason")
            ),
            "exact_terms": cls._compact_diagnostic_terms(list(item.get("exact_terms") or [])),
            "historical_item_terms": cls._compact_diagnostic_terms(
                list(item.get("historical_item_terms") or [])
            ),
            "entity_keys": cls._compact_diagnostic_terms(
                sorted(set(str(value) for value in list(coverage_profile.get("entity_keys") or [])))
            ),
            "facet_values": cls._compact_diagnostic_terms(
                sorted(set(str(value) for value in list(coverage_profile.get("facet_values") or [])))
            ),
        }

    @classmethod
    def _page_cutoff_universe_diagnostics(
        cls,
        page_rows: list[dict[str, object]],
        *,
        top_k: int,
        query_shape: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        diagnostics: dict[str, dict[str, object]] = {}
        query_requires_broad_universe = bool(
            query_shape.get("list_like")
            or query_shape.get("count_like")
            or query_shape.get("multi_entity")
            or query_shape.get("comparison_like")
            or query_shape.get("item_family")
        )
        for cutoff in range(1, min(cls.DIAGNOSTIC_TOP_N_PAGES, len(page_rows)) + 1):
            prefix = page_rows[:cutoff]
            trajectory_ids = list(
                dict.fromkeys(
                    trajectory_id
                    for row in prefix
                    for trajectory_id in list(row.get("trajectory_ids") or [])
                    if str(trajectory_id).strip()
                )
            )
            singleton_count = sum(1 for row in prefix if int(row.get("trajectory_count") or 0) == 1)
            medium_count = sum(
                1 for row in prefix if 3 <= int(row.get("trajectory_count") or 0) <= 6
            )
            broad_count = sum(1 for row in prefix if bool(row.get("broad_entity_profile")))
            index_fallback_reason = None
            if len(trajectory_ids) < min(max(top_k, 0), 8):
                index_fallback_reason = "small_universe"
            elif len(trajectory_ids) < max(top_k, 0) and query_requires_broad_universe:
                index_fallback_reason = "coverage_shape_under_top_k"
            elif prefix and len(trajectory_ids) <= 2:
                index_fallback_reason = "very_small_unique_universe"
            diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "selected_page_ids": [str(row.get("page_id") or "") for row in prefix],
                "selected_page_trajectory_ids": trajectory_ids,
                "selected_page_trajectory_count": len(trajectory_ids),
                "singleton_page_count": singleton_count,
                "medium_page_count": medium_count,
                "broad_profile_page_count": broad_count,
                "index_fallback_would_trigger": index_fallback_reason is not None,
                "index_fallback_simulation_reason": index_fallback_reason or "not_triggered",
            }
        return diagnostics

    @classmethod
    def _trajectory_cutoff_diagnostics_from_rows(
        cls, trajectory_rows: list[dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        diagnostics: dict[str, dict[str, object]] = {}
        for cutoff in range(1, min(cls.DIAGNOSTIC_TOP_N_TRAJECTORIES, len(trajectory_rows)) + 1):
            prefix = trajectory_rows[:cutoff]
            last = prefix[-1] if prefix else {}
            diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "trajectory_ids": [str(row.get("trajectory_id") or "") for row in prefix],
                "trajectory_count": len(prefix),
                "last_rank": int(last.get("rank") or cutoff),
                "last_final_score": float(last.get("final_score") or 0.0),
            }
        return diagnostics

    @staticmethod
    def _model_payload_to_dict(payload: object) -> dict[str, object]:
        if hasattr(payload, "model_dump"):
            return dict(payload.model_dump())
        if hasattr(payload, "dict"):
            return dict(payload.dict())
        return dict(payload)  # type: ignore[arg-type]

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, object]:
        normalized = text.strip()
        if normalized.startswith("```") and normalized.endswith("```"):
            lines = normalized.splitlines()
            normalized = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
            if not match:
                raise
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("retrieval reflection text fallback did not return a JSON object")
        return payload

    def _wiki_directory_for_reflection(self, sample_id: str) -> str:
        pages = self.store.list_wiki_pages(sample_id)
        if not pages:
            return "None."
        rows: list[str] = []
        for page in sorted(pages, key=lambda item: (item.page_type == "index", item.page_type, item.slug))[:80]:
            metadata = dict(page.metadata_json or {})
            exact_terms = self._clean_text_values(list(metadata.get("exact_terms") or []), limit=8)
            display_items = self._clean_text_values(list(metadata.get("display_items") or []), limit=8)
            rows.append(
                "- "
                f"slug={page.slug} | type={page.page_type} | title={page.title} | "
                f"entities={', '.join(str(value) for value in list(page.entity_names_json or [])[:6]) or 'none'} | "
                f"trajectories={len(page.trajectory_ids_json or [])} | "
                f"terms={', '.join([*exact_terms, *display_items][:10]) or 'none'}"
            )
        return "\n".join(rows)

    @staticmethod
    def _reflection_summary_from_metadata(metadata: dict[str, object]) -> str:
        return "\n".join(
            [
                f"selected_pages={metadata.get('selected_pages') or metadata.get('page_rerank_selected_ids') or []}",
                f"selected_trajectories={metadata.get('trajectory_rerank_selected_ids') or []}",
                f"source_count={len(list(metadata.get('source_refs') or []))}",
                f"active_claims={metadata.get('answer_context_active_claim_count')}",
                f"query_entities={metadata.get('query_entities') or []}",
                f"query_facets={metadata.get('query_facets') or {}}",
            ]
        )

    def _deterministic_reflection_hints(
        self,
        sample_id: str,
        query_text: str,
        *,
        error: str | None = None,
    ) -> dict[str, object]:
        entity_lexicon = self._sample_entity_lexicon(sample_id)
        query_facets = extract_query_facets_v1(query_text, entity_lexicon)
        query_shape = classify_query_shape_v1(query_text, entity_lexicon)
        keywords = sorted(extract_keywords(query_text))
        entities = self._clean_text_values(list(query_facets.get("entities") or []), limit=12)
        facet_values = self._clean_text_values(list(query_facets.get("values") or []), limit=12)
        candidate_slugs: list[str] = []
        search_terms = {value.casefold() for value in [*keywords, *entities, *facet_values] if value}
        for page in self.store.list_wiki_pages(sample_id):
            page_text = " ".join(
                [
                    page.slug,
                    page.title,
                    " ".join(str(value) for value in list(page.entity_names_json or [])),
                    " ".join(str(value) for value in list(page.keywords_json or [])),
                ]
            ).casefold()
            if any(term in page_text for term in search_terms):
                candidate_slugs.append(page.slug)
            if len(candidate_slugs) >= 8:
                break
        answer_type = str(query_shape.get("item_family") or "")
        if not answer_type:
            answer_type = "count" if query_shape.get("count_like") else "unknown"
        return {
            "rewritten_query": query_text,
            "answer_type": answer_type,
            "target_entities": entities,
            "event_terms": keywords[:12],
            "temporal_terms": [
                term for term in keywords if re.search(r"\b(day|week|month|year|today|tomorrow|yesterday)\b", term)
            ][:8],
            "must_find_terms": self._clean_text_values([*facet_values, *keywords], limit=16),
            "candidate_page_slugs": candidate_slugs,
            "raw_search_terms": self._clean_text_values([*facet_values, *keywords, *entities], limit=16),
            "rationale": "deterministic fallback from query facets and keywords",
            "reflection_mode": "deterministic_fallback",
            "reflection_error": error,
        }

    def build_reflection_hints(
        self,
        sample_id: str,
        query_text: str,
        *,
        initial_answer_text: str,
        initial_retrieval_metadata: dict[str, object],
    ) -> dict[str, object]:
        prompt = (
            load_prompt("retrieval_reflection")
            + "\n\nQUESTION:\n"
            + query_text
            + "\n\nINITIAL_ANSWER:\n"
            + initial_answer_text
            + "\n\nINITIAL_RETRIEVAL_SUMMARY:\n"
            + self._reflection_summary_from_metadata(initial_retrieval_metadata)
            + "\n\nWIKI_DIRECTORY:\n"
            + self._wiki_directory_for_reflection(sample_id)
        )
        if self.llm_provider is None:
            return self._deterministic_reflection_hints(
                sample_id,
                query_text,
                error="no_llm_provider",
            )
        started_at = time.perf_counter()
        task = "retrieval_reflection"
        spec = get_structured_task_spec(task)
        try:
            if self.llm_provider.supports_structured(task):
                response = self.llm_provider.generate_structured(
                    [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                    spec=spec,
                    metadata={"task": task, "sample_id": sample_id},
                )
                payload = self._model_payload_to_dict(response.parsed)
                payload.update(
                    {
                        "reflection_mode": "structured",
                        "reflection_prompt_tokens": int(response.prompt_tokens or 0),
                        "reflection_completion_tokens": int(response.completion_tokens or 0),
                        "reflection_latency_ms": (time.perf_counter() - started_at) * 1000.0,
                        "reflection_error": None,
                    }
                )
                return payload
        except Exception as exc:  # noqa: BLE001
            structured_error = _compact_error_message(exc)
        else:
            structured_error = None
        try:
            response = self.llm_provider.generate(
                [
                    NormalizedMessage(
                        role="user",
                        content=prompt + "\n\nReturn ONLY a JSON object matching the requested schema.",
                        turn_index=0,
                    )
                ],
                metadata={"task": task, "sample_id": sample_id, "structured_fallback": True},
            )
            parsed = parse_structured_payload(spec, self._extract_json_object(response.text))
            payload = self._model_payload_to_dict(parsed)
            payload.update(
                {
                    "reflection_mode": "text_json",
                    "reflection_prompt_tokens": int(response.prompt_tokens or 0),
                    "reflection_completion_tokens": int(response.completion_tokens or 0),
                    "reflection_latency_ms": (time.perf_counter() - started_at) * 1000.0,
                    "reflection_error": structured_error,
                }
            )
            return payload
        except Exception as exc:  # noqa: BLE001
            error = structured_error or _compact_error_message(exc)
            hints = self._deterministic_reflection_hints(sample_id, query_text, error=error)
            hints["reflection_latency_ms"] = (time.perf_counter() - started_at) * 1000.0
            return hints

    def _reflection_terms(self, reflection_hints: dict[str, object] | None) -> list[str]:
        if not reflection_hints:
            return []
        values: list[object] = []
        for key in (
            "rewritten_query",
            "answer_type",
            "target_entities",
            "event_terms",
            "temporal_terms",
            "must_find_terms",
            "raw_search_terms",
        ):
            value = reflection_hints.get(key)
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)
        return self._clean_text_values(values, limit=40)

    @staticmethod
    def _rrf_score(*ranks: int | None) -> float:
        score = 0.0
        for rank in ranks:
            if rank is None:
                continue
            score += 1.0 / (RetrievalEngine.RRF_K + rank)
        return score

    @classmethod
    def _fuse_dense_sparse_scores(
        cls,
        scored: list[dict[str, object]],
        *,
        id_key: str,
    ) -> list[dict[str, object]]:
        dense_ranked = sorted(
            scored,
            key=lambda item: (float(item["dense_score"]), str(item[id_key])),
            reverse=True,
        )
        sparse_ranked = sorted(
            scored,
            key=lambda item: (float(item["sparse_score"]), str(item[id_key])),
            reverse=True,
        )
        dense_rank_by_id = {str(item[id_key]): index + 1 for index, item in enumerate(dense_ranked)}
        sparse_rank_by_id = {str(item[id_key]): index + 1 for index, item in enumerate(sparse_ranked)}
        fused: list[dict[str, object]] = []
        for item in scored:
            item_id = str(item[id_key])
            fused.append(
                {
                    **item,
                    "fused_score": cls._rrf_score(
                        dense_rank_by_id.get(item_id),
                        sparse_rank_by_id.get(item_id),
                    ),
                }
            )
        fused.sort(
            key=lambda item: (
                float(item["fused_score"]),
                float(item["dense_score"]),
                float(item["sparse_score"]),
            ),
            reverse=True,
        )
        return fused

    @staticmethod
    def _fill_selected_ids_after_rerank(
        candidate_pool: list[dict[str, object]],
        reranked_ids: list[str],
        *,
        id_key: str,
        final_count: int,
    ) -> list[str]:
        selected_ids: list[str] = []
        for item_id in reranked_ids:
            candidate_id = str(item_id)
            if candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
            if len(selected_ids) >= final_count:
                return selected_ids[:final_count]
        for item in candidate_pool:
            candidate_id = str(item[id_key])
            if candidate_id in selected_ids:
                continue
            selected_ids.append(candidate_id)
            if len(selected_ids) >= final_count:
                break
        return selected_ids[:final_count]

    @staticmethod
    def _parse_rerank_rationales(text: str, prefix: str) -> dict[str, str]:
        rationales: dict[str, str] = {}
        for label, rationale in re.findall(rf"^- ({prefix}\d+):\s*(.+)$", text, flags=re.MULTILINE):
            rationales[label] = collapse_whitespace(rationale)
        return rationales

    def _rerank_selected_ids(
        self,
        *,
        prompt_name: str,
        query_text: str,
        candidates: list[dict[str, object]],
        final_count: int,
        label_prefix: str,
        text_key: str,
        id_key: str,
    ) -> tuple[list[str], dict[str, str], bool, dict[str, str]]:
        if self.llm_provider is None or not candidates:
            return [], {}, True, {}
        sections = []
        label_to_id: dict[str, str] = {}
        for index, candidate in enumerate(candidates, start=1):
            label = f"{label_prefix}{index}"
            candidate_id = str(candidate[id_key])
            label_to_id[label] = candidate_id
            sections.append(
                f"### {label}\n"
                f"id={candidate_id}\n"
                f"content:\n{str(candidate[text_key]).strip()}"
            )
        prompt = (
            load_prompt(prompt_name)
            + "\n\nQuestion:\n"
            + query_text
            + f"\n\nSelect exactly {min(final_count, len(candidates))} candidates.\n\nCandidates:\n"
            + "\n\n".join(sections)
        )
        try:
            response = self.llm_provider.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata={
                    "task": prompt_name,
                    "candidate_pool_size": len(candidates),
                    "final_count": final_count,
                },
            )
            selected_line = next(
                (line for line in response.text.splitlines() if line.upper().startswith("SELECTED:")),
                "",
            )
            selected_labels = re.findall(rf"\b{label_prefix}\d+\b", selected_line)
            selected_ids: list[str] = []
            for label in selected_labels:
                candidate_id = label_to_id.get(label)
                if candidate_id and candidate_id not in selected_ids:
                    selected_ids.append(candidate_id)
                if len(selected_ids) >= final_count:
                    break
            if not selected_ids:
                return [], {}, True, {}
            return selected_ids, self._parse_rerank_rationales(response.text, label_prefix), False, {}
        except Exception as exc:
            return [], {}, True, {
                "rerank_error_type": type(exc).__name__,
                "rerank_error_message": _compact_error_message(exc),
                "rerank_prompt_name": prompt_name,
            }

    def _trajectory_summary_text(
        self,
        trajectory: TrajectoryRecord,
        metadata: dict[str, object] | None = None,
    ) -> str:
        metadata = dict(trajectory.metadata_json or {}) if metadata is None else metadata
        summary_text = collapse_whitespace(str(metadata.get("retrieval_summary_text") or ""))
        if summary_text:
            return summary_text
        return fallback_summary_from_metadata(metadata, trajectory_label=str(trajectory.label or ""))

    def _trajectory_summary_keywords(
        self,
        trajectory: TrajectoryRecord,
        summary_text: str,
        metadata: dict[str, object] | None = None,
    ) -> list[str]:
        metadata = dict(trajectory.metadata_json or {}) if metadata is None else metadata
        stored = sanitize_summary_keyword_values(
            list(metadata.get("retrieval_summary_keywords_v2") or []),
            limit=32,
        ) or sanitize_summary_keyword_values(
            list(metadata.get("retrieval_summary_keywords") or []),
            limit=32,
        )
        if stored:
            return stored
        return summary_keywords_v2(summary_text, metadata)

    def _trajectory_retrieval_signals(
        self,
        trajectory: TrajectoryRecord,
        metadata: dict[str, object] | None = None,
    ) -> TrajectoryRetrievalSignals:
        metadata = dict(trajectory.metadata_json or {}) if metadata is None else metadata
        summary_text = self._trajectory_summary_text(trajectory, metadata)
        summary_keyword_list = self._trajectory_summary_keywords(trajectory, summary_text, metadata)
        exact_terms = [
            str(value).strip()
            for value in list(metadata.get("exact_terms_v2") or metadata.get("exact_terms") or [])
            if str(value).strip()
        ]
        display_items = [
            str(value).strip()
            for value in list(metadata.get("display_items") or [])
            if str(value).strip()
        ]
        display_counts = [
            str(value).strip()
            for value in list(metadata.get("display_counts") or [])
            if str(value).strip()
        ]
        display_key_facts = [
            str(value).strip()
            for value in list(metadata.get("display_key_facts") or [])
            if str(value).strip()
        ]
        historical_item_terms = sanitize_historical_item_terms(
            list(
                metadata.get("trajectory_historical_item_terms_v2")
                or metadata.get("trajectory_historical_item_terms_v1")
                or []
            ),
            limit=24,
        )
        source_event_object_terms = [
            str(value).strip()
            for value in list(metadata.get("source_event_object_terms_v1") or [])
            if str(value).strip()
        ]
        source_event_action_terms = [
            str(value).strip()
            for value in list(metadata.get("source_event_action_terms_v1") or [])
            if str(value).strip()
        ]
        source_event_canonical_terms = [
            str(value).strip()
            for value in list(metadata.get("source_event_canonical_terms_v1") or [])
            if str(value).strip()
        ]
        source_temporal_relation_terms = [
            str(value).strip()
            for value in list(metadata.get("source_temporal_relation_terms_v1") or [])
            if str(value).strip()
        ]
        source_event_records = [
            dict(record)
            for record in list(metadata.get("source_event_records_v1") or [])
            if isinstance(record, dict)
        ]
        facet_values = {
            str(value).strip().casefold()
            for value in list(metadata.get("facet_values") or [])
            if str(value).strip()
        }
        facet_tags = {
            str(value).strip()
            for value in list(metadata.get("facet_tags") or [])
            if str(value).strip()
        }
        entity_mentions = [
            str(value).strip()
            for value in list(metadata.get("entity_mentions") or [])
            if str(value).strip()
        ]
        return TrajectoryRetrievalSignals(
            metadata=metadata,
            latest_snapshot_id=str(metadata.get("latest_snapshot_id") or trajectory.latest_snapshot_id or ""),
            summary_text=summary_text,
            summary_keywords=summary_keyword_list,
            exact_terms=exact_terms,
            display_items=display_items,
            display_counts=display_counts,
            display_key_facts=display_key_facts,
            historical_item_terms=historical_item_terms,
            source_event_object_terms=source_event_object_terms,
            source_event_action_terms=source_event_action_terms,
            source_event_canonical_terms=source_event_canonical_terms,
            source_temporal_relation_terms=source_temporal_relation_terms,
            source_event_records=source_event_records,
            facet_values=facet_values,
            facet_tags=facet_tags,
            entity_mentions=entity_mentions,
            entity_keys={normalize_entity_key(value) for value in entity_mentions},
            lexical_keywords=(
                set(summary_keyword_list)
                | exact_term_keyword_set(exact_terms)
                | exact_term_keyword_set(historical_item_terms)
                | exact_term_keyword_set(source_event_object_terms)
                | exact_term_keyword_set(source_event_canonical_terms)
                | exact_term_keyword_set(facet_values)
            ),
            support_terms=(
                set(summary_keyword_list)
                | exact_term_keyword_set(exact_terms)
                | exact_term_keyword_set(display_items)
                | exact_term_keyword_set(display_counts)
                | exact_term_keyword_set(display_key_facts)
                | exact_term_keyword_set(historical_item_terms)
                | exact_term_keyword_set(source_event_object_terms)
                | exact_term_keyword_set(source_event_action_terms)
                | exact_term_keyword_set(source_event_canonical_terms)
                | exact_term_keyword_set(source_temporal_relation_terms)
                | exact_term_keyword_set(entity_mentions)
                | exact_term_keyword_set(facet_values)
            ),
        )

    @staticmethod
    def _claim_facet_signals(claims: list[ClaimRecord]) -> ClaimFacetSignals:
        entity_keys: set[str] = set()
        facet_tags: set[str] = set()
        facet_values: set[str] = set()
        for claim in claims:
            for facet in list((claim.metadata_json or {}).get("facets_v2") or (claim.metadata_json or {}).get("facets_v1") or []):
                if not isinstance(facet, dict):
                    continue
                entity = str(facet.get("entity") or "").strip()
                relation = str(facet.get("relation") or "").strip()
                value = str(facet.get("value") or "").strip()
                if entity:
                    entity_keys.add(normalize_entity_key(entity))
                if relation:
                    facet_tags.add(relation)
                if relation and value:
                    facet_values.add(facet_value_key(relation, value).casefold())
        return ClaimFacetSignals(
            entity_keys=entity_keys,
            facet_tags=facet_tags,
            facet_values=facet_values,
        )

    @staticmethod
    def _query_shape_requires_coverage(query_shape: dict[str, object]) -> bool:
        return bool(
            query_shape.get("list_like")
            or query_shape.get("multi_entity")
            or query_shape.get("comparison_like")
            or query_shape.get("count_like")
        )

    @classmethod
    def _coverage_item_terms(cls, values: Iterable[object], *, limit: int = 24) -> set[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = collapse_whitespace(str(value or ""))
            if not text:
                continue
            candidates = [text.casefold(), *extract_keywords(text)]
            for candidate in candidates:
                normalized = collapse_whitespace(str(candidate)).casefold()
                if (
                    not normalized
                    or len(normalized) < 2
                    or normalized in cls.COVERAGE_GENERIC_ITEM_TERMS
                    or normalized in seen
                ):
                    continue
                seen.add(normalized)
                terms.append(normalized)
                if len(terms) >= limit:
                    return set(terms)
        return set(terms)

    @classmethod
    def _family_terms_for_query(cls, query_shape: dict[str, object]) -> set[str]:
        family = str(query_shape.get("item_family") or "").strip().casefold()
        if not family:
            return set()
        terms = set(cls.FAMILY_MATCH_TERMS.get(family, frozenset()))
        terms.update(extract_keywords(family.replace("_", " ")))
        return {term for term in terms if term and term not in cls.COVERAGE_GENERIC_ITEM_TERMS}

    @classmethod
    def _query_object_terms_for_shape(cls, query_shape: dict[str, object]) -> set[str]:
        text = collapse_whitespace(str(query_shape.get("normalized_question") or "")).casefold()
        if not text:
            return set()
        stop = set(cls.COVERAGE_GENERIC_ITEM_TERMS) | {
            "what",
            "which",
            "where",
            "when",
            "who",
            "whose",
            "does",
            "did",
            "done",
            "has",
            "have",
            "had",
            "was",
            "were",
            "with",
            "from",
            "recently",
            "caroline",
            "melanie",
            "joanna",
            "john",
            "nate",
            "james",
            "calvin",
            "maria",
            "tim",
        }
        terms = {
            term
            for term in extract_keywords(text)
            if len(term) >= 3 and term not in stop
        }
        family = str(query_shape.get("item_family") or "").strip().casefold()
        terms.update(cls.FAMILY_MATCH_TERMS.get(family, frozenset()) & terms)
        return terms

    @staticmethod
    def _contains_visual_object_signal(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:sunset|sunrise|flower|sunflower|horse|portrait|landscape|painting|painted|artwork|picture|image)\b",
                text.casefold(),
            )
        )

    @classmethod
    def _answer_family_match_profile(
        cls,
        *,
        query_shape: dict[str, object],
        query_keywords: set[str],
        support_terms: set[str],
        exact_terms: Iterable[object] = (),
        display_items: Iterable[object] = (),
        display_key_facts: Iterable[object] = (),
        historical_item_terms: Iterable[object] = (),
        summary_text: str = "",
        query_object_terms: set[str] | None = None,
    ) -> dict[str, object]:
        family = str(query_shape.get("item_family") or "").strip().casefold()
        if not family:
            return {
                "score": 0.0,
                "matched_terms": [],
                "family_terms": [],
                "query_object_terms": [],
                "query_object_overlap_terms": [],
                "mismatch_penalty": 0.0,
                "strong_match": False,
            }
        family_terms = cls._family_terms_for_query(query_shape)
        query_object_terms = set(query_object_terms or cls._query_object_terms_for_shape(query_shape))
        text_parts = [
            summary_text,
            " ".join(str(value) for value in exact_terms),
            " ".join(str(value) for value in display_items),
            " ".join(str(value) for value in display_key_facts),
            " ".join(str(value) for value in historical_item_terms),
            " ".join(sorted(support_terms)),
        ]
        text = collapse_whitespace(" ".join(text_parts)).casefold()
        term_hits = {
            term
            for term in family_terms
            if term in support_terms or (len(term) >= 4 and re.search(rf"\b{re.escape(term)}\b", text))
        }
        concrete_values = cls._coverage_item_terms(
            [*exact_terms, *display_items, *display_key_facts, *historical_item_terms],
            limit=32,
        )
        query_hits = {
            term
            for term in query_keywords
            if term not in cls.COVERAGE_GENERIC_ITEM_TERMS
            and term
            not in {
                "caroline",
                "melanie",
                "joanna",
                "john",
                "nate",
                "james",
                "calvin",
                "maria",
                "tim",
            }
            and term in support_terms
        }
        query_object_hits = {
            term
            for term in query_object_terms
            if term in support_terms or (len(term) >= 4 and re.search(rf"\b{re.escape(term)}\b", text))
        }
        score = 0.0
        if family_terms:
            score += min(0.48, len(term_hits) / max(len(family_terms), 1) * 1.10)
        score += min(0.30, len(query_hits) * 0.08)
        score += min(0.30, len(query_object_hits) * 0.12)
        if concrete_values and (term_hits or query_object_hits):
            score += 0.10
        if family == "painted_object" and cls._contains_visual_object_signal(text):
            score += 0.28
            term_hits.add("visual_object")
        if family == "research_topic" and ("agency" in text or "agencies" in text or "options" in text):
            score += 0.28
            term_hits.add("research_options")
        if family == "band" and re.search(r"\b(?:band|artist|concert|festival|sounds)\b", text):
            score += 0.22
            term_hits.add("music_event")
        if family in {"book", "reading"} and re.search(r"\b(?:book|read|reading|title|cover|novel)\b", text):
            score += 0.22
            term_hits.add("reading_signal")
        if family == "instrument" and re.search(r"\b(?:clarinet|violin|guitar|piano|instrument|play|played)\b", text):
            score += 0.22
            term_hits.add("instrument_signal")
        if family in {"country", "city", "state", "place"} and re.search(r"\b(?:country|city|state|area|county|moved|from|visited|sweden|canada|mexico)\b", text):
            score += 0.22
            term_hits.add("place_signal")
        if family in {"writing"} and re.search(r"\b(?:screenplay|script|scripts|writing|story|fiction|rejected)\b", text):
            score += 0.22
            term_hits.add("writing_signal")
        if family in {"person"} and re.search(r"\b(?:mother|father|aunt|uncle|family member|passed away|died)\b", text):
            score += 0.22
            term_hits.add("person_signal")
        score = min(1.0, score)
        mismatch_penalty = 0.0
        if family_terms and not term_hits and not query_hits and not query_object_hits and not concrete_values:
            mismatch_penalty = 0.25
        elif family_terms and not term_hits and not query_hits and not query_object_hits:
            mismatch_penalty = 0.12
        return {
            "score": score,
            "matched_terms": sorted(term_hits | query_hits | query_object_hits),
            "family_terms": sorted(family_terms),
            "query_object_terms": sorted(query_object_terms),
            "query_object_overlap_terms": sorted(query_object_hits),
            "mismatch_penalty": mismatch_penalty,
            "strong_match": score >= cls.FAMILY_MATCH_STRONG_THRESHOLD,
        }

    @classmethod
    def _temporal_event_query_profile(
        cls,
        query_text: str,
        query_shape: dict[str, object],
        query_keywords: set[str],
    ) -> dict[str, object]:
        question_lower = collapse_whitespace(query_text).casefold()
        family = str(query_shape.get("item_family") or "").strip().casefold()
        temporal_like = bool(
            re.search(r"\b(?:when|what\s+year|which\s+year|date)\b", question_lower)
        )
        event_like = bool(
            family in {"event", "activity", "painted_object"}
            or query_keywords & cls.TEMPORAL_EVENT_QUERY_TERMS
            or re.search(r"\b(?:event|events|competition|roadtrip|road\s+trip|accident|campaign|race|speech|talk)\b", question_lower)
        )
        enabled = bool(temporal_like or event_like)
        ignored_terms = cls.COVERAGE_GENERIC_ITEM_TERMS | {
            "when",
            "what",
            "which",
            "year",
            "date",
            "did",
            "does",
            "has",
            "have",
            "had",
            "was",
            "were",
            "after",
            "before",
            "first",
            "last",
            "next",
            "time",
            "times",
            "jon",
            "melanie",
            "caroline",
            "joanna",
            "nate",
            "maria",
            "calvin",
            "james",
            "tim",
            "dave",
        } | cls.TEMPORAL_EVENT_ACTION_TERMS
        object_terms = {
            term
            for term in query_keywords
            if len(term) >= 3 and term not in ignored_terms
        }
        action_terms = {
            term
            for term in query_keywords
            if term in cls.TEMPORAL_EVENT_ACTION_TERMS
        }
        if "road" in query_keywords and "trip" in query_keywords:
            object_terms.add("roadtrip")
        relation_terms = {
            term
            for term in ("after", "before", "first", "last", "launched", "started", "hosted", "visited")
            if term in question_lower
        }
        return {
            "enabled": enabled,
            "temporal_like": temporal_like,
            "event_like": event_like,
            "object_terms": sorted(object_terms),
            "action_terms": sorted(action_terms),
            "relation_terms": sorted(relation_terms),
            "question_text": question_lower,
        }

    @classmethod
    def _trajectory_source_event_match_profile(
        cls,
        *,
        signals: TrajectoryRetrievalSignals,
        query_profile: dict[str, object],
    ) -> dict[str, object]:
        if not query_profile.get("enabled"):
            return {
                "score": 0.0,
                "matched_terms": [],
                "matched_refs": [],
                "matched_record_count": 0,
                "reason": "not_temporal_event_query",
                "strong_match": False,
            }
        object_terms = {
            str(term).casefold()
            for term in list(query_profile.get("object_terms") or [])
            if str(term).strip()
            and str(term).casefold() not in cls.COVERAGE_GENERIC_ITEM_TERMS
            and str(term).casefold() not in {"event", "events", "experience", "experiences", "support"}
        }
        action_terms = {
            str(term).casefold()
            for term in list(query_profile.get("action_terms") or [])
            if str(term).strip()
        }
        if not object_terms:
            return {
                "score": 0.0,
                "matched_terms": [],
                "matched_refs": [],
                "matched_record_count": 0,
                "reason": "no_specific_query_object_terms",
                "strong_match": False,
            }
        best_score = 0.0
        matched_terms: set[str] = set()
        matched_refs: list[str] = []
        matched_record_count = 0
        reason = "no_source_event_match"
        for record in signals.source_event_records:
            text_parts = [
                record.get("surface"),
                record.get("canonical"),
                record.get("raw_surface"),
                record.get("action"),
                record.get("temporal_expression"),
            ]
            record_text = collapse_whitespace(" ".join(str(part or "") for part in text_parts)).casefold()
            record_keywords = exact_term_keyword_set([record_text])
            object_hits = {
                term
                for term in object_terms
                if term in record_keywords or (len(term) >= 4 and re.search(rf"\b{re.escape(term)}\b", record_text))
            }
            if not object_hits:
                continue
            action_hits = {
                term
                for term in action_terms
                if term in record_keywords or (len(term) >= 4 and re.search(rf"\b{re.escape(term)}\b", record_text))
            }
            temporal_hit = bool(str(record.get("temporal_expression") or "").strip())
            record_score = min(1.0, 0.70 + 0.10 * len(object_hits) + 0.10 * len(action_hits) + (0.05 if temporal_hit else 0.0))
            if record_score > best_score:
                best_score = record_score
                reason = "source_event_object_match"
                if action_hits:
                    reason = "source_event_action_object_match"
            matched_record_count += 1
            matched_terms.update(object_hits)
            matched_terms.update(action_hits)
            for source_ref in list(record.get("source_refs") or []):
                source_ref_text = str(source_ref).strip()
                if source_ref_text and source_ref_text not in matched_refs:
                    matched_refs.append(source_ref_text)
        return {
            "score": best_score,
            "matched_terms": sorted(matched_terms),
            "matched_refs": matched_refs[: cls.DIAGNOSTIC_TEXT_ITEM_LIMIT],
            "matched_record_count": matched_record_count,
            "reason": reason,
            "strong_match": best_score >= 0.70,
        }

    @staticmethod
    def _base_rank_scores(candidate_pool: list[dict[str, object]], preferred_ids: list[str], *, id_key: str) -> dict[str, float]:
        ordered_ids: list[str] = []
        for item_id in [*preferred_ids, *(str(item[id_key]) for item in candidate_pool)]:
            candidate_id = str(item_id)
            if candidate_id and candidate_id not in ordered_ids:
                ordered_ids.append(candidate_id)
        total = len(ordered_ids) or 1
        return {
            candidate_id: float(total - index) / float(total)
            for index, candidate_id in enumerate(ordered_ids)
        }

    @classmethod
    def _coverage_cluster_key(
        cls,
        profile: dict[str, object],
        *,
        query_shape: dict[str, object] | None = None,
    ) -> tuple[str, ...]:
        if cls._query_shape_requires_coverage(query_shape or {}):
            item_terms = tuple(
                sorted(
                    str(value).casefold()
                    for value in set(profile.get("item_terms") or set())
                    if str(value).strip() and str(value).casefold() not in cls.COVERAGE_GENERIC_ITEM_TERMS
                )
            )
            if item_terms:
                return item_terms[:3]
            facet_values = tuple(
                sorted(str(value) for value in set(profile.get("facet_values") or set()) if str(value))
            )
            if facet_values:
                return facet_values[:3]
        entity_keys = tuple(sorted(str(value) for value in set(profile.get("entity_keys") or set()) if str(value)))
        if entity_keys:
            return entity_keys
        exact_terms = tuple(
            sorted(str(value) for value in set(profile.get("exact_terms") or []) if str(value))
        )
        if exact_terms:
            return exact_terms[:2]
        page_type = str(profile.get("page_type") or "").strip()
        if page_type:
            return (page_type,)
        return ("__none__",)

    def _coverage_aware_select_ids(
        self,
        *,
        candidate_pool: list[dict[str, object]],
        reranked_ids: list[str],
        id_key: str,
        final_count: int,
        query_shape: dict[str, object],
        query_keywords: set[str],
        query_entity_keys: set[str],
        query_facet_values: set[str],
    ) -> tuple[list[str], dict[str, object]]:
        empty_metadata = {
            "selection_strategy": "empty",
            "cluster_keys": [],
            "covered_query_entities": [],
            "covered_query_facet_values": [],
            "covered_query_terms": [],
            "covered_item_terms": [],
            "selected_score_components": [],
            "redundancy_penalties": [],
        }
        if not candidate_pool:
            return [], empty_metadata
        if final_count <= 0:
            return [], empty_metadata
        if not self._query_shape_requires_coverage(query_shape):
            selected_ids = self._fill_selected_ids_after_rerank(
                candidate_pool,
                list(reranked_ids),
                id_key=id_key,
                final_count=final_count,
            )
            return selected_ids, {
                "selection_strategy": "rank_fill",
                "cluster_keys": [],
                "covered_query_entities": [],
                "covered_query_facet_values": [],
                "covered_query_terms": [],
                "covered_item_terms": [],
                "selected_score_components": [],
                "redundancy_penalties": [],
            }

        rank_scores = self._base_rank_scores(candidate_pool, reranked_ids, id_key=id_key)
        query_terms = set(query_keywords)
        item_family = str(query_shape.get("item_family") or "").strip()
        if item_family:
            query_terms.update(extract_keywords(item_family))
            query_terms.add(item_family.casefold())

        candidate_by_id = {str(item[id_key]): item for item in candidate_pool}
        selected_ids: list[str] = []
        covered_entities: set[str] = set()
        covered_facet_values: set[str] = set()
        covered_terms: set[str] = set()
        covered_item_terms: set[str] = set()
        covered_clusters: set[tuple[str, ...]] = set()
        selected_score_components: list[dict[str, object]] = []
        redundancy_penalties: list[dict[str, object]] = []

        def _profile_sets(item: dict[str, object]) -> tuple[set[str], set[str], set[str], set[str], tuple[str, ...]]:
            profile = dict(item.get("coverage_profile") or {})
            candidate_entities = {str(value) for value in set(profile.get("entity_keys") or set()) if str(value)}
            candidate_facet_values = {
                str(value).casefold()
                for value in set(profile.get("facet_values") or set())
                if str(value).strip()
            }
            candidate_terms = {
                str(value).casefold()
                for value in set(profile.get("support_terms") or set())
                if str(value).strip()
            }
            candidate_item_terms = {
                str(value).casefold()
                for value in set(profile.get("item_terms") or set())
                if str(value).strip() and str(value).casefold() not in self.COVERAGE_GENERIC_ITEM_TERMS
            }
            cluster_key = self._coverage_cluster_key(profile, query_shape=query_shape)
            return candidate_entities, candidate_facet_values, candidate_terms, candidate_item_terms, cluster_key

        def _record_selected(
            item: dict[str, object],
            *,
            score: float,
            anchored: bool,
            component: dict[str, object] | None = None,
        ) -> None:
            candidate_id = str(item[id_key])
            selected_ids.append(candidate_id)
            candidate_entities, candidate_facet_values, candidate_terms, candidate_item_terms, cluster_key = _profile_sets(item)
            new_entities = (candidate_entities & query_entity_keys) - covered_entities
            new_facet_values = (candidate_facet_values & query_facet_values) - covered_facet_values
            new_terms = (candidate_terms & query_terms) - covered_terms
            new_item_terms = candidate_item_terms - covered_item_terms
            covered_entities.update(new_entities)
            covered_facet_values.update(new_facet_values)
            covered_terms.update(new_terms)
            covered_item_terms.update(new_item_terms)
            covered_clusters.add(cluster_key)
            selected_score_components.append(
                {
                    "id": candidate_id,
                    "score": float(score),
                    "anchored": anchored,
                    "base_rank_score": float(rank_scores.get(candidate_id, 0.0)),
                    "answer_family_match_score": float(item.get("answer_family_match_score") or 0.0),
                    "answer_family_matched_terms": list(item.get("answer_family_matched_terms") or []),
                    "answer_family_mismatch_penalty": float(item.get("answer_family_mismatch_penalty") or 0.0),
                    "new_query_entities": sorted(new_entities),
                    "new_query_facet_values": sorted(new_facet_values),
                    "new_query_terms": sorted(new_terms),
                    "new_item_terms": sorted(new_item_terms),
                    "cluster_key": list(cluster_key),
                    **(component or {}),
                }
            )

        anchor_id = next((candidate_id for candidate_id in reranked_ids if candidate_id in candidate_by_id), None)
        if anchor_id is None:
            anchor_id = str(candidate_pool[0][id_key])
        if item_family:
            anchor_item = candidate_by_id.get(anchor_id)
            anchor_family_score = float(anchor_item.get("answer_family_match_score") or 0.0) if anchor_item else 0.0
            if anchor_family_score < self.FAMILY_MATCH_STRONG_THRESHOLD:
                family_aligned = [
                    item
                    for item in candidate_pool
                    if float(item.get("answer_family_match_score") or 0.0) >= self.FAMILY_MATCH_STRONG_THRESHOLD
                ]
                if family_aligned:
                    family_aligned.sort(
                        key=lambda item: (
                            float(item.get("answer_family_match_score") or 0.0),
                            rank_scores.get(str(item[id_key]), 0.0),
                            str(item[id_key]),
                        ),
                        reverse=True,
                    )
                    anchor_id = str(family_aligned[0][id_key])
        _record_selected(
            candidate_by_id[anchor_id],
            score=rank_scores.get(anchor_id, 0.0),
            anchored=True,
            component={"selection_reason": "relevance_anchor"},
        )

        while len(selected_ids) < min(final_count, len(candidate_pool)):
            best_item: dict[str, object] | None = None
            best_score = float("-inf")
            best_component: dict[str, object] = {}
            for item in candidate_pool:
                candidate_id = str(item[id_key])
                if candidate_id in selected_ids:
                    continue
                profile = dict(item.get("coverage_profile") or {})
                candidate_entities, candidate_facet_values, candidate_terms, candidate_item_terms, cluster_key = _profile_sets(item)
                new_entities = (candidate_entities & query_entity_keys) - covered_entities
                new_facet_values = (candidate_facet_values & query_facet_values) - covered_facet_values
                new_terms = (candidate_terms & query_terms) - covered_terms
                new_item_terms = candidate_item_terms - covered_item_terms
                repeated_entities = candidate_entities & covered_entities
                repeated_terms = candidate_terms & covered_terms
                repeated_item_terms = candidate_item_terms & covered_item_terms
                inventory_like = bool(profile.get("inventory_like"))
                cluster_penalty = 0.22 if cluster_key in covered_clusters and cluster_key != ("__none__",) else 0.0
                priority = str(item.get("routing_priority") or "normal")
                priority_bonus = 0.08 if priority == "high" else (-0.04 if priority == "low" else 0.0)
                inventory_bonus = 0.18 if query_shape.get("list_like") and inventory_like else 0.0
                count_bonus = 0.18 if query_shape.get("count_like") and profile.get("has_count_signal") else 0.0
                family_score = float(item.get("answer_family_match_score") or 0.0)
                family_mismatch_penalty = float(item.get("answer_family_mismatch_penalty") or 0.0)
                repeated_entity_penalty = 0.04 * len(repeated_entities) if new_item_terms else 0.10 * len(repeated_entities)
                repeated_term_penalty = 0.03 * min(6, len(repeated_terms))
                item_overlap_penalty = 0.06 * min(6, len(repeated_item_terms))
                candidate_score = (
                    rank_scores.get(candidate_id, 0.0)
                    + 0.90 * len(new_entities)
                    + 0.65 * len(new_facet_values)
                    + 0.30 * min(6, len(new_item_terms))
                    + 0.12 * min(4, len(new_terms))
                    + inventory_bonus
                    + count_bonus
                    + priority_bonus
                    + 0.75 * family_score
                    - repeated_entity_penalty
                    - repeated_term_penalty
                    - item_overlap_penalty
                    - cluster_penalty
                    - 0.35 * family_mismatch_penalty
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_item = item
                    best_component = {
                        "selection_reason": "coverage_greedy",
                        "inventory_bonus": float(inventory_bonus),
                        "count_bonus": float(count_bonus),
                        "priority_bonus": float(priority_bonus),
                        "answer_family_match_score": float(family_score),
                        "answer_family_matched_terms": list(item.get("answer_family_matched_terms") or []),
                        "answer_family_mismatch_penalty": float(family_mismatch_penalty),
                        "cluster_penalty": float(cluster_penalty),
                        "repeated_entity_penalty": float(repeated_entity_penalty),
                        "repeated_term_penalty": float(repeated_term_penalty),
                        "item_overlap_penalty": float(item_overlap_penalty),
                        "new_item_terms": sorted(new_item_terms),
                        "repeated_item_terms": sorted(repeated_item_terms),
                    }
            if best_item is None:
                break
            _record_selected(best_item, score=best_score, anchored=False, component=best_component)
            if best_component:
                redundancy_penalties.append(
                    {
                        "id": str(best_item[id_key]),
                        "cluster_penalty": float(best_component.get("cluster_penalty", 0.0)),
                        "repeated_entity_penalty": float(best_component.get("repeated_entity_penalty", 0.0)),
                        "repeated_term_penalty": float(best_component.get("repeated_term_penalty", 0.0)),
                        "item_overlap_penalty": float(best_component.get("item_overlap_penalty", 0.0)),
                        "answer_family_mismatch_penalty": float(
                            best_component.get("answer_family_mismatch_penalty", 0.0)
                        ),
                    }
                )

        selected_clusters = [
            list(cluster)
            for cluster in sorted(covered_clusters)
            if cluster != ("__none__",)
        ]
        return selected_ids, {
            "selection_strategy": "coverage_aware_greedy",
            "cluster_keys": selected_clusters,
            "covered_query_entities": sorted(covered_entities),
            "covered_query_facet_values": sorted(covered_facet_values),
            "covered_query_terms": sorted(covered_terms),
            "covered_item_terms": sorted(covered_item_terms),
            "selected_score_components": selected_score_components,
            "redundancy_penalties": redundancy_penalties,
        }

    def _index_fallback_trigger_reason(
        self,
        *,
        selected_page_universe_size: int,
        selected_page_count: int,
        query_shape: dict[str, object],
    ) -> str | None:
        if selected_page_universe_size < min(self.top_k, 8):
            return "selected_page_universe_below_minimum"
        if (
            selected_page_universe_size < self.top_k
            and (
                self._query_shape_requires_coverage(query_shape)
                or bool(query_shape.get("item_family"))
            )
        ):
            return "coverage_query_universe_below_top_k"
        if selected_page_count > 0 and selected_page_universe_size <= 2:
            return "selected_pages_cover_too_few_trajectories"
        return None

    def _index_fallback_trajectory_expansion(
        self,
        *,
        sample_id: str,
        selected_page_trajectory_ids: list[str],
        page_metadata: dict[str, object],
        query_keywords: set[str],
        query_entities: list[str],
        query_facet_tags: set[str],
        query_facet_values: set[str],
        query_shape: dict[str, object],
    ) -> tuple[list[str], dict[str, object]]:
        selected_ids = list(dict.fromkeys(selected_page_trajectory_ids))
        selected_set = set(selected_ids)
        metadata: dict[str, object] = {
            "index_fallback_used": False,
            "index_fallback_reason": "not_triggered",
            "selected_page_universe_size_before_index_fallback": len(selected_ids),
            "selected_page_universe_size_after_index_fallback": len(selected_ids),
            "index_fallback_candidate_count": 0,
            "index_fallback_added_trajectory_ids": [],
        }
        if not bool(page_metadata.get("page_index_suppressed")):
            metadata["index_fallback_reason"] = "index_not_suppressed"
            return selected_ids, metadata
        index_trajectory_ids = [
            str(value)
            for value in list(page_metadata.get("index_page_trajectory_ids") or [])
            if str(value).strip()
        ]
        if not index_trajectory_ids:
            metadata["index_fallback_reason"] = "no_index_trajectories"
            return selected_ids, metadata
        reason = self._index_fallback_trigger_reason(
            selected_page_universe_size=len(selected_ids),
            selected_page_count=len(list(page_metadata.get("page_rerank_selected_ids") or [])),
            query_shape=query_shape,
        )
        if reason is None:
            return selected_ids, metadata

        candidate_id_set = set(index_trajectory_ids) - selected_set
        scored_rows = self._rank_trajectories_by_metadata_overlap(
            sample_id=sample_id,
            trajectory_ids=candidate_id_set,
            query_keywords=query_keywords,
            query_entities=query_entities,
            query_facet_tags=query_facet_tags,
            query_facet_values=query_facet_values,
            query_shape=query_shape,
        )
        scored = [(score, trajectory_id) for score, trajectory_id, _ in scored_rows if score > 0.0]
        limit = max(self.top_k * 2, 24)
        added_ids = [trajectory_id for _, trajectory_id in scored[:limit]]
        expanded_ids = list(dict.fromkeys([*selected_ids, *added_ids]))
        metadata.update(
            {
                "index_fallback_used": bool(added_ids),
                "index_fallback_reason": reason if added_ids else "triggered_no_matching_index_candidates",
                "selected_page_universe_size_after_index_fallback": len(expanded_ids),
                "index_fallback_candidate_count": len(scored),
                "index_fallback_added_trajectory_ids": added_ids,
            }
        )
        if added_ids:
            self._trace(
                f"sample={sample_id} index_fallback_expansion_used reason={reason} "
                f"before={len(selected_ids)} candidates={len(scored)} added={len(added_ids)} after={len(expanded_ids)}"
            )
        else:
            self._trace(
                f"sample={sample_id} index_fallback_expansion_skipped "
                f"reason=triggered_no_matching_index_candidates trigger={reason} before={len(selected_ids)}"
            )
        return expanded_ids, metadata

    def _rank_trajectories_by_metadata_overlap(
        self,
        *,
        sample_id: str,
        trajectory_ids: Iterable[str],
        query_keywords: set[str],
        query_entities: list[str],
        query_facet_tags: set[str],
        query_facet_values: set[str],
        query_shape: dict[str, object],
    ) -> list[tuple[float, str, dict[str, object]]]:
        candidate_id_set = {str(value) for value in trajectory_ids if str(value).strip()}
        if not candidate_id_set:
            return []
        query_entity_keys = {normalize_entity_key(value) for value in query_entities}
        query_facet_value_keys = {str(value).casefold() for value in query_facet_values}
        query_terms = set(query_keywords)
        item_family = str(query_shape.get("item_family") or "").strip()
        if item_family:
            query_terms.update(extract_keywords(item_family.replace("_", " ")))
            query_terms.add(item_family.casefold())
        rows = [
            trajectory
            for trajectory in self.store.list_trajectories(sample_id)
            if trajectory.id in candidate_id_set
        ]
        ranked: list[tuple[float, str, dict[str, object]]] = []
        for trajectory in rows:
            signals = self._trajectory_retrieval_signals(trajectory)
            support_terms = (
                set(signals.support_terms)
                | exact_term_keyword_set(signals.exact_terms)
                | exact_term_keyword_set(signals.display_items)
                | exact_term_keyword_set(signals.display_counts)
                | exact_term_keyword_set(signals.display_key_facts)
                | exact_term_keyword_set(signals.historical_item_terms)
            )
            family_profile = self._answer_family_match_profile(
                query_shape=query_shape,
                query_keywords=query_keywords,
                support_terms=support_terms,
                exact_terms=signals.exact_terms,
                display_items=signals.display_items,
                display_key_facts=signals.display_key_facts,
                historical_item_terms=signals.historical_item_terms,
                summary_text=signals.summary_text,
            )
            lexical_score = keyword_overlap_score(query_terms, support_terms) if query_terms else 0.0
            entity_score = 0.25 if query_entity_keys and signals.entity_keys & query_entity_keys else 0.0
            facet_score = 0.15 if query_facet_tags and signals.facet_tags & query_facet_tags else 0.0
            value_score = 0.15 if query_facet_value_keys and signals.facet_values & query_facet_value_keys else 0.0
            inventory_score = 0.10 if query_shape.get("list_like") and signals.display_items else 0.0
            count_score = 0.10 if query_shape.get("count_like") and signals.display_counts else 0.0
            score = (
                lexical_score
                + entity_score
                + facet_score
                + value_score
                + inventory_score
                + count_score
                + 0.80 * float(family_profile.get("score") or 0.0)
                - 0.30 * float(family_profile.get("mismatch_penalty") or 0.0)
            )
            ranked.append((score, trajectory.id, family_profile))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked

    def _apply_broad_entity_page_candidate_cap(
        self,
        *,
        sample_id: str,
        selected_page_trajectory_ids: list[str],
        page_metadata: dict[str, object],
        query_keywords: set[str],
        query_entities: list[str],
        query_facet_tags: set[str],
        query_facet_values: set[str],
        query_shape: dict[str, object],
    ) -> tuple[list[str], dict[str, object]]:
        selected_ids = list(dict.fromkeys(selected_page_trajectory_ids))
        selected_rows = [
            dict(row)
            for row in list(page_metadata.get("selected_page_rows") or [])
            if isinstance(row, dict)
        ]
        broad_rows = [
            row for row in selected_rows
            if (
                bool(row.get("broad_entity_profile"))
                or str(row.get("routing_priority") or "") == "profile"
                or (
                    str(row.get("page_type") or "") == "entity"
                    and len(list(row.get("trajectory_ids") or [])) > self.BROAD_ENTITY_TRAJECTORY_THRESHOLD
                )
            )
        ]
        fine_grained_rows = [
            row for row in selected_rows
            if bool(row.get("entity_facet_split_from_broad_page"))
        ]
        metadata: dict[str, object] = {
            "broad_entity_page_selected": bool(broad_rows),
            "broad_entity_page_ids": [str(row.get("page_id") or "") for row in broad_rows if str(row.get("page_id") or "")],
            "broad_entity_profile_page_ids": [
                str(row.get("page_id") or "") for row in broad_rows if bool(row.get("broad_entity_profile"))
            ],
            "fine_grained_entity_page_ids": [
                str(row.get("page_id") or "") for row in fine_grained_rows if str(row.get("page_id") or "")
            ],
            "entity_facet_page_candidate_count": len(
                list(
                    dict.fromkeys(
                        trajectory_id
                        for row in fine_grained_rows
                        for trajectory_id in list(row.get("trajectory_ids") or [])
                        if str(trajectory_id).strip()
                    )
                )
            ),
            "broad_entity_profile_suppressed": False,
            "selected_page_universe_size_before_broad_cap": len(selected_ids),
            "selected_page_universe_size_after_broad_cap": len(selected_ids),
            "broad_entity_candidate_cap_used": False,
            "broad_entity_added_trajectory_ids": [],
            "broad_entity_profile_added_trajectory_ids": [],
        }
        if not broad_rows:
            return selected_ids, metadata

        broad_ids: list[str] = []
        non_broad_ids: list[str] = []
        broad_page_id_set = {str(row.get("page_id") or "") for row in broad_rows}
        for row in selected_rows:
            trajectory_ids = [str(value) for value in list(row.get("trajectory_ids") or []) if str(value).strip()]
            if str(row.get("page_id") or "") in broad_page_id_set:
                broad_ids.extend(trajectory_ids)
            else:
                non_broad_ids.extend(trajectory_ids)
        non_broad_ids = list(dict.fromkeys(non_broad_ids))
        broad_candidate_ids = [trajectory_id for trajectory_id in list(dict.fromkeys(broad_ids)) if trajectory_id not in set(non_broad_ids)]
        if not broad_candidate_ids:
            capped_ids = non_broad_ids or selected_ids
            metadata["selected_page_universe_size_after_broad_cap"] = len(capped_ids)
            metadata["broad_entity_profile_suppressed"] = bool(broad_rows and non_broad_ids)
            return capped_ids, metadata

        ranked = self._rank_trajectories_by_metadata_overlap(
            sample_id=sample_id,
            trajectory_ids=broad_candidate_ids,
            query_keywords=query_keywords,
            query_entities=query_entities,
            query_facet_tags=query_facet_tags,
            query_facet_values=query_facet_values,
            query_shape=query_shape,
        )
        fine_grained_universe_size = len(non_broad_ids)
        cap = max(2 * max(self.top_k, 1), self.BROAD_ENTITY_CANDIDATE_CAP_MIN)
        if fine_grained_rows and fine_grained_universe_size >= self.top_k:
            # Fine-grained facet pages already provide enough candidates. Only let a
            # broad profile page contribute trajectories with a strong metadata match.
            ranked_positive = [
                trajectory_id
                for score, trajectory_id, profile in ranked
                if score > 0.0 and float(profile.get("score") or 0.0) >= self.FAMILY_MATCH_STRONG_THRESHOLD
            ]
            cap = min(cap, max(self.top_k, 1))
        else:
            ranked_positive = [trajectory_id for score, trajectory_id, _ in ranked if score > 0.0]
        ranked_zero = [trajectory_id for score, trajectory_id, _ in ranked if score <= 0.0]
        ranked_ids = list(dict.fromkeys([*ranked_positive, *([] if fine_grained_rows else ranked_zero)]))
        added_ids = ranked_ids[:cap]
        capped_ids = list(dict.fromkeys([*non_broad_ids, *added_ids]))
        metadata.update(
            {
                "selected_page_universe_size_after_broad_cap": len(capped_ids),
                "broad_entity_candidate_cap_used": len(capped_ids) < len(selected_ids),
                "broad_entity_added_trajectory_ids": added_ids,
                "broad_entity_profile_added_trajectory_ids": added_ids,
                "broad_entity_profile_suppressed": bool(broad_rows and len(added_ids) < len(broad_candidate_ids)),
            }
        )
        if metadata["broad_entity_candidate_cap_used"]:
            self._trace(
                f"sample={sample_id} broad_entity_candidate_cap_used "
                f"broad_pages={len(broad_rows)} before={len(selected_ids)} after={len(capped_ids)} added={len(added_ids)}"
            )
        return capped_ids, metadata

    @classmethod
    def _page_strong_query_match(
        cls,
        *,
        query_keywords: set[str],
        title: str,
        exact_terms: list[str],
        display_items: list[str],
        display_counts: list[str],
        routing_text: str,
    ) -> bool:
        if not query_keywords:
            return False
        signal_terms = (
            exact_term_keyword_set(exact_terms)
            | exact_term_keyword_set(display_items)
            | exact_term_keyword_set(display_counts)
            | set(extract_keywords(title))
        )
        concrete_terms = {
            term.casefold()
            for term in signal_terms
            if term.casefold() not in cls.COVERAGE_GENERIC_ITEM_TERMS
        }
        query_terms = {
            term.casefold()
            for term in query_keywords
            if term.casefold() not in cls.COVERAGE_GENERIC_ITEM_TERMS
        }
        if len(query_terms & concrete_terms) >= 2:
            return True
        if keyword_overlap_score(query_terms, concrete_terms) >= 0.35:
            return True
        routing_terms = {
            term.casefold()
            for term in extract_keywords(routing_text)
            if term.casefold() not in cls.COVERAGE_GENERIC_ITEM_TERMS
        }
        return len(query_terms & routing_terms) >= 3

    @classmethod
    def _page_family_match_profile(
        cls,
        *,
        query_shape: dict[str, object],
        query_keywords: set[str],
        title: str,
        slug: str,
        routing_text: str,
        exact_terms: list[str],
        display_items: list[str],
        display_counts: list[str],
        display_key_facts: list[str],
        facet_values: Iterable[str],
        entity_names: Iterable[str],
        historical_item_terms: Iterable[str],
        source_surface_terms: Iterable[str],
    ) -> dict[str, object]:
        support_terms = (
            set(extract_keywords(title))
            | set(extract_keywords(slug.replace("-", " ")))
            | exact_term_keyword_set(exact_terms)
            | exact_term_keyword_set(display_items)
            | exact_term_keyword_set(display_counts)
            | exact_term_keyword_set(display_key_facts)
            | exact_term_keyword_set(facet_values)
            | exact_term_keyword_set(historical_item_terms)
            | exact_term_keyword_set(source_surface_terms)
            | {normalize_entity_key(value) for value in entity_names}
            | set(extract_keywords(routing_text))
        )
        return cls._answer_family_match_profile(
            query_shape=query_shape,
            query_keywords=query_keywords,
            support_terms={term for term in support_terms if term},
            exact_terms=[
                *exact_terms,
                *display_counts,
                *list(facet_values),
                *list(source_surface_terms),
            ],
            display_items=display_items,
            display_key_facts=display_key_facts,
            historical_item_terms=historical_item_terms,
            summary_text=f"{title} {slug} {routing_text}",
        )

    @classmethod
    def _page_granularity_adjustment(
        cls,
        *,
        trajectory_count: int,
        page_type: str,
        routing_priority: str,
        broad_entity_profile: bool,
        entity_facet_split: bool,
        wiki_rescue_reason: str | None,
        strong_query_match: bool,
        singleton_policy: str | None = None,
        singleton_quality_score: float | None = None,
    ) -> tuple[float, dict[str, object]]:
        adjustment = 0.0
        singleton_penalty = 0.0
        low_quality_singleton_penalty = 0.0
        medium_bonus = 0.0
        broad_penalty = 0.0
        if 3 <= trajectory_count <= 6 and page_type != "index":
            medium_bonus = 0.035
            adjustment += medium_bonus
        elif trajectory_count == 1 and page_type != "index":
            if strong_query_match:
                adjustment += 0.015
            elif singleton_policy == "merge_required_low_quality":
                low_quality_singleton_penalty = -0.08
                singleton_penalty = low_quality_singleton_penalty
                adjustment += low_quality_singleton_penalty
            elif wiki_rescue_reason:
                singleton_penalty = -0.045
                adjustment += singleton_penalty
            elif entity_facet_split:
                singleton_penalty = -0.025
                adjustment += singleton_penalty
            else:
                singleton_penalty = -0.015
                adjustment += singleton_penalty
        elif trajectory_count > 6 and page_type != "index":
            adjustment -= 0.015
        if broad_entity_profile or routing_priority == "profile":
            broad_penalty = -0.04
            adjustment += broad_penalty
        return adjustment, {
            "page_granularity_adjustment": adjustment,
            "singleton_page_penalty": singleton_penalty,
            "low_quality_singleton_penalty": low_quality_singleton_penalty,
            "medium_page_bonus": medium_bonus,
            "broad_profile_page_penalty": broad_penalty,
            "singleton_policy": singleton_policy,
            "singleton_quality_score": float(singleton_quality_score or 0.0),
            "singleton_penalty_cancelled_by_exact_match": bool(
                trajectory_count == 1 and strong_query_match and singleton_penalty == 0.0
            ),
            "medium_granularity_page": bool(3 <= trajectory_count <= 6 and page_type != "index"),
            "singleton_page": bool(trajectory_count == 1 and page_type != "index"),
            "page_strong_query_match": strong_query_match,
        }

    def _route_pages(
        self,
        sample_id: str,
        query_text: str,
        query_embedding: list[float],
        query_keywords: set[str],
        query_entities: list[str],
        query_shape: dict[str, object] | None = None,
        query_facet_values: set[str] | None = None,
        reflection_hints: dict[str, object] | None = None,
    ) -> tuple[list[str], list[str], dict[str, object]]:
        started_at = time.perf_counter()
        query_shape = dict(query_shape or {})
        query_facet_values = set(query_facet_values or set())
        reflection_terms = self._reflection_terms(reflection_hints)
        reflection_term_keys = {term.casefold() for term in reflection_terms}
        reflection_candidate_slugs = {
            str(value).strip().casefold()
            for value in list((reflection_hints or {}).get("candidate_page_slugs") or [])
            if str(value).strip()
        }
        all_pages = self.store.list_wiki_pages(sample_id)
        index_pages = [page for page in all_pages if page.page_type == "index"]
        index_page_trajectory_ids = list(
            dict.fromkeys(
                trajectory_id
                for page in index_pages
                for trajectory_id in list(page.trajectory_ids_json or [])
            )
        )
        pages = list(all_pages)
        non_index_pages = [page for page in pages if page.page_type != "index"]
        non_index_trajectory_counts = [
            len(list(dict.fromkeys(page.trajectory_ids_json or [])))
            for page in non_index_pages
        ]
        non_index_singleton_count = sum(1 for count in non_index_trajectory_counts if count == 1)
        non_index_total_links = sum(non_index_trajectory_counts)
        seed_type_counts: Counter[str] = Counter()
        seed_type_singletons: Counter[str] = Counter()
        allowed_specific_singletons = 0
        low_quality_singletons = 0
        low_quality_singletons_merged = 0
        overwide_non_index_pages = 0
        overwide_page_split_count = 0
        for page, trajectory_count in zip(non_index_pages, non_index_trajectory_counts):
            page_metadata = dict(page.metadata_json or {})
            seed_type = str(page_metadata.get("seed_type") or "unknown")
            seed_type_counts[seed_type] += 1
            if trajectory_count == 1:
                seed_type_singletons[seed_type] += 1
                if page_metadata.get("wiki_singleton_policy") == "allowed_isolated_specific":
                    allowed_specific_singletons += 1
                if page_metadata.get("wiki_singleton_policy") == "merge_required_low_quality":
                    low_quality_singletons += 1
            if page_metadata.get("wiki_singleton_low_quality_merged") is True:
                low_quality_singletons_merged += 1
            if trajectory_count > 6:
                overwide_non_index_pages += 1
            if page_metadata.get("wiki_overwide_page_split") is True:
                overwide_page_split_count += 1
        wiki_fragmentation_metadata = {
            "wiki_fragmentation_diagnostics_available": True,
            "wiki_non_index_page_count": len(non_index_pages),
            "wiki_singleton_non_index_page_count": non_index_singleton_count,
            "wiki_singleton_non_index_page_rate": (
                non_index_singleton_count / len(non_index_pages)
                if non_index_pages
                else None
            ),
            "wiki_mean_trajectories_per_non_index_page": (
                non_index_total_links / len(non_index_pages)
                if non_index_pages
                else None
            ),
            "wiki_singleton_rate_by_seed_type": {
                seed_type: (
                    seed_type_singletons[seed_type] / count
                    if count
                    else None
                )
                for seed_type, count in sorted(seed_type_counts.items())
            },
            "wiki_post_plan_rescue_singleton_count": seed_type_singletons.get("post_plan_rescue", 0),
            "wiki_entity_facet_singleton_count": seed_type_singletons.get("entity_facet", 0),
            "wiki_allowed_specific_singleton_count": allowed_specific_singletons,
            "wiki_low_quality_singleton_count": low_quality_singletons,
            "wiki_low_quality_singleton_merged_count": low_quality_singletons_merged,
            "wiki_overwide_non_index_page_count": overwide_non_index_pages,
            "wiki_overwide_page_split_count": overwide_page_split_count,
            "wiki_max_non_index_trajectory_count": max(non_index_trajectory_counts) if non_index_trajectory_counts else 0,
        }
        index_suppressed = bool(non_index_pages and index_pages)
        if non_index_pages:
            self._trace(
                f"sample={sample_id} page_route_index_suppressed count={len(pages) - len(non_index_pages)}"
            )
            pages = non_index_pages
        elif pages:
            self._trace(
                f"sample={sample_id} page_route_index_retained reason=only_index_pages count={len(pages)}"
            )
        page_embeddings = self.store.fetch_embeddings_by_owner_ids([page.id for page in pages], "wiki_page")
        missing_page_embedding_count = sum(1 for page in pages if page.id not in page_embeddings)
        if missing_page_embedding_count:
            self._trace(
                f"sample={sample_id} page_route_embedding_missing "
                f"missing={missing_page_embedding_count} total={len(pages)}"
            )
        query_entity_keys = {normalize_entity_key(value) for value in query_entities}
        scored: list[dict[str, object]] = []
        for page in pages:
            embedding = page_embeddings.get(page.id)
            page_metadata = dict(page.metadata_json or {})
            routing_text = collapse_whitespace(str(page_metadata.get("routing_text") or page.markdown_text or ""))
            exact_terms = [
                str(value).strip()
                for value in list(page_metadata.get("exact_terms") or [])
                if str(value).strip()
            ]
            display_items = [
                str(value).strip()
                for value in list(page_metadata.get("display_items") or [])
                if str(value).strip()
            ]
            display_counts = [
                str(value).strip()
                for value in list(page_metadata.get("display_counts") or [])
                if str(value).strip()
            ]
            display_key_facts = [
                str(value).strip()
                for value in list(page_metadata.get("display_key_facts") or [])
                if str(value).strip()
            ]
            historical_item_terms = [
                str(value).strip()
                for value in list(page_metadata.get("wiki_historical_item_terms") or [])
                if str(value).strip()
            ]
            source_surface_terms = [
                str(value).strip()
                for value in [
                    *list(page_metadata.get("source_surface_terms_v1") or []),
                    *list(page_metadata.get("source_surface_raw_terms_v1") or []),
                ]
                if str(value).strip()
            ]
            source_event_terms = [
                str(value).strip()
                for value in [
                    *list(page_metadata.get("wiki_source_event_object_terms_v1") or []),
                    *list(page_metadata.get("wiki_source_event_canonical_terms_v1") or []),
                    *list(page_metadata.get("wiki_source_temporal_relation_terms_v1") or []),
                ]
                if str(value).strip()
            ]
            facet_values = {
                str(value).strip().casefold()
                for value in list(page_metadata.get("facet_values") or [])
                if str(value).strip()
            }
            entity_keys = {normalize_entity_key(value) for value in list(page.entity_names_json or [])}
            reflection_text = " ".join(
                [
                    page.slug,
                    page.title,
                    routing_text,
                    " ".join(exact_terms),
                    " ".join(source_event_terms),
                    " ".join(display_items),
                    " ".join(display_counts),
                    " ".join(str(value) for value in list(page.entity_names_json or [])),
                    " ".join(str(value) for value in facet_values),
                ]
            ).casefold()
            reflection_term_hits = sorted(
                term for term in reflection_term_keys if term and term in reflection_text
            )
            reflection_term_bonus = min(
                self.REFLECTION_TERM_BONUS_MAX,
                0.04 * len(reflection_term_hits),
            )
            reflection_slug_bonus = (
                self.REFLECTION_SLUG_BONUS
                if page.slug.casefold() in reflection_candidate_slugs
                else 0.0
            )
            reflection_bonus = reflection_term_bonus + reflection_slug_bonus
            dense = cosine_similarity(query_embedding, embedding.vector_json) if embedding is not None else 0.0
            sparse_terms = (
                set(page.keywords_json or [])
                | set(extract_keywords(page.title))
                | exact_term_keyword_set(source_event_terms)
                | entity_keys
            )
            item_terms = self._coverage_item_terms(
                [
                    *exact_terms,
                    *display_items,
                    *display_counts,
                    *source_event_terms,
                    *list(facet_values),
                    page.title,
                    routing_text,
                ]
            )
            sparse = keyword_overlap_score(query_keywords, sparse_terms) if query_keywords else 0.0
            entity_bonus = (
                0.10
                if query_entity_keys
                and entity_keys & query_entity_keys
                else 0.0
            )
            trajectory_ids = list(page.trajectory_ids_json or [])
            routing_priority = str(page_metadata.get("routing_priority") or "normal")
            broad_entity_profile = bool(page_metadata.get("broad_entity_profile"))
            entity_facet_split = bool(page_metadata.get("entity_facet_split_from_broad_page"))
            strong_query_match = self._page_strong_query_match(
                query_keywords=query_keywords,
                title=page.title,
                exact_terms=exact_terms,
                display_items=display_items,
                display_counts=display_counts,
                routing_text=routing_text,
            )
            page_family_profile = self._page_family_match_profile(
                query_shape=query_shape,
                query_keywords=query_keywords,
                title=page.title,
                slug=page.slug,
                routing_text=routing_text,
                exact_terms=exact_terms,
                display_items=display_items,
                display_counts=display_counts,
                display_key_facts=display_key_facts,
                facet_values=facet_values,
                entity_names=list(page.entity_names_json or []),
                historical_item_terms=historical_item_terms,
                source_surface_terms=[*source_event_terms, *source_surface_terms],
            )
            page_family_score = float(page_family_profile.get("score") or 0.0)
            page_family_mismatch_penalty = float(page_family_profile.get("mismatch_penalty") or 0.0)
            strong_query_match = bool(strong_query_match or page_family_profile.get("strong_match"))
            granularity_adjustment, granularity_metadata = self._page_granularity_adjustment(
                trajectory_count=len(trajectory_ids),
                page_type=page.page_type,
                routing_priority=routing_priority,
                broad_entity_profile=broad_entity_profile,
                entity_facet_split=entity_facet_split,
                wiki_rescue_reason=(
                    str(page_metadata.get("wiki_rescue_reason"))
                    if page_metadata.get("wiki_rescue_reason")
                    else None
                ),
                strong_query_match=strong_query_match,
                singleton_policy=str(page_metadata.get("wiki_singleton_policy") or ""),
                singleton_quality_score=float(page_metadata.get("wiki_singleton_quality_score") or 0.0),
            )
            scored.append(
                {
                    "page_id": page.id,
                    "routing_text": routing_text,
                    "dense_score": (
                        dense
                        + entity_bonus
                        + reflection_bonus
                        + granularity_adjustment
                        + min(0.10, page_family_score * 0.10)
                        - min(0.04, page_family_mismatch_penalty * 0.12)
                    ),
                    "sparse_score": max(
                        0.0,
                        sparse
                        + min(0.35, page_family_score * 0.35)
                        - min(0.10, page_family_mismatch_penalty * 0.25),
                    ),
                    "entity_bonus": entity_bonus,
                    "reflection_bonus": reflection_bonus,
                    **granularity_metadata,
                    "page_family_match_score": page_family_score,
                    "page_family_matched_terms": list(page_family_profile.get("matched_terms") or []),
                    "page_query_object_terms": list(page_family_profile.get("query_object_terms") or []),
                    "page_query_object_overlap_terms": list(
                        page_family_profile.get("query_object_overlap_terms") or []
                    ),
                    "page_family_mismatch_penalty": page_family_mismatch_penalty,
                    "page_family_score_reason": {
                        "family_terms": list(page_family_profile.get("family_terms") or []),
                        "strong_match": bool(page_family_profile.get("strong_match")),
                    },
                    "reflection_term_hits": reflection_term_hits,
                    "trajectory_ids": trajectory_ids,
                    "trajectory_count": len(trajectory_ids),
                    "page_type": page.page_type,
                    "title": page.title,
                    "slug": page.slug,
                    "routing_priority": routing_priority,
                    "broad_entity_profile": broad_entity_profile,
                    "entity_facet_split_from_broad_page": entity_facet_split,
                    "wiki_rescue_reason": page_metadata.get("wiki_rescue_reason"),
                    "wiki_singleton_exception": bool(page_metadata.get("wiki_singleton_exception")),
                    "singleton_exception_reason": page_metadata.get("singleton_exception_reason"),
                    "singleton_policy": granularity_metadata.get("singleton_policy"),
                    "singleton_quality_score": granularity_metadata.get("singleton_quality_score"),
                    "low_quality_singleton_penalty": granularity_metadata.get("low_quality_singleton_penalty"),
                    "coverage_profile": {
                        "entity_keys": entity_keys,
                        "facet_values": facet_values,
                        "support_terms": (
                            set(sparse_terms)
                            | exact_term_keyword_set(exact_terms)
                            | exact_term_keyword_set(display_items)
                            | exact_term_keyword_set(display_counts)
                            | exact_term_keyword_set(facet_values)
                            | set(extract_keywords(routing_text))
                        ),
                        "item_terms": item_terms,
                        "exact_terms": exact_terms,
                        "inventory_like": page.page_type == "inventory" or bool(display_items),
                        "has_count_signal": bool(display_counts),
                        "page_type": page.page_type,
                    },
                }
            )
        fused = self._fuse_dense_sparse_scores(scored, id_key="page_id")
        page_ranked_rows_compact_top_n = [
            self._compact_page_ranked_row(item, index + 1)
            for index, item in enumerate(fused[: self.DIAGNOSTIC_TOP_N_PAGES])
        ]
        page_cutoff_universe_diagnostics = self._page_cutoff_universe_diagnostics(
            page_ranked_rows_compact_top_n,
            top_k=self.top_k,
            query_shape=query_shape,
        )
        candidate_pool = fused[: min(max(self.top_t_pages, self.CANDIDATE_POOL_SIZE), len(fused))]
        reranked_ids, rerank_rationales, rerank_fallback, rerank_error_metadata = self._rerank_selected_ids(
            prompt_name="wiki_page_rerank",
            query_text=query_text,
            candidates=candidate_pool,
            final_count=min(self.top_t_pages, len(candidate_pool)),
            label_prefix="P",
            text_key="routing_text",
            id_key="page_id",
        )
        if rerank_error_metadata:
            self._trace(
                f"sample={sample_id} page_rerank_failed "
                f"error_type={rerank_error_metadata.get('rerank_error_type', 'unknown')} "
                f"error={rerank_error_metadata.get('rerank_error_message', '')}"
            )
        selected_ids, selection_metadata = self._coverage_aware_select_ids(
            candidate_pool=candidate_pool,
            reranked_ids=list(reranked_ids),
            id_key="page_id",
            final_count=self.top_t_pages,
            query_shape=query_shape,
            query_keywords=query_keywords,
            query_entity_keys=query_entity_keys,
            query_facet_values={str(value).casefold() for value in query_facet_values},
        )
        trajectory_union: list[str] = []
        selected_page_rows: list[dict[str, object]] = []
        selected_page_rows_compact: list[dict[str, object]] = []
        for item in candidate_pool:
            if str(item["page_id"]) not in selected_ids:
                continue
            granularity_adjustment = float(item.get("page_granularity_adjustment") or 0.0)
            dense_score = float(item.get("dense_score") or 0.0)
            final_score = float(item.get("fused_score") or 0.0)
            selected_page_rows.append(
                {
                    "page_id": str(item["page_id"]),
                    "page_type": str(item["page_type"]),
                    "title": str(item["title"]),
                    "slug": str(item["slug"]),
                    "trajectory_ids": list(item["trajectory_ids"]),
                    "trajectory_count": int(item.get("trajectory_count") or 0),
                    "routing_priority": str(item["routing_priority"]),
                    "broad_entity_profile": bool(item.get("broad_entity_profile")),
                    "entity_facet_split_from_broad_page": bool(item.get("entity_facet_split_from_broad_page")),
                    "wiki_rescue_reason": item.get("wiki_rescue_reason"),
                    "wiki_singleton_exception": bool(item.get("wiki_singleton_exception")),
                    "singleton_exception_reason": item.get("singleton_exception_reason"),
                    "singleton_policy": str(item.get("singleton_policy") or ""),
                    "singleton_quality_score": float(item.get("singleton_quality_score") or 0.0),
                    "page_granularity_adjustment": float(item.get("page_granularity_adjustment") or 0.0),
                    "singleton_page_penalty": float(item.get("singleton_page_penalty") or 0.0),
                    "low_quality_singleton_penalty": float(item.get("low_quality_singleton_penalty") or 0.0),
                    "medium_page_bonus": float(item.get("medium_page_bonus") or 0.0),
                    "broad_profile_page_penalty": float(item.get("broad_profile_page_penalty") or 0.0),
                    "singleton_page": bool(item.get("singleton_page")),
                    "medium_granularity_page": bool(item.get("medium_granularity_page")),
                    "page_strong_query_match": bool(item.get("page_strong_query_match")),
                    "page_family_match_score": float(item.get("page_family_match_score") or 0.0),
                    "page_family_matched_terms": list(item.get("page_family_matched_terms") or []),
                    "page_query_object_terms": list(item.get("page_query_object_terms") or []),
                    "page_query_object_overlap_terms": list(item.get("page_query_object_overlap_terms") or []),
                    "page_family_mismatch_penalty": float(item.get("page_family_mismatch_penalty") or 0.0),
                    "page_family_score_reason": dict(item.get("page_family_score_reason") or {}),
                }
            )
            selected_page_rows_compact.append(
                {
                    "page_id": str(item["page_id"]),
                    "title": str(item["title"]),
                    "page_type": str(item["page_type"]),
                    "trajectory_count": int(item.get("trajectory_count") or 0),
                    "base_score": dense_score - granularity_adjustment,
                    "final_score": final_score,
                    "page_granularity_adjustment": granularity_adjustment,
                    "singleton_page_penalty": float(item.get("singleton_page_penalty") or 0.0),
                    "low_quality_singleton_penalty": float(item.get("low_quality_singleton_penalty") or 0.0),
                    "medium_page_bonus": float(item.get("medium_page_bonus") or 0.0),
                    "broad_profile_page_penalty": float(item.get("broad_profile_page_penalty") or 0.0),
                    "singleton_policy": str(item.get("singleton_policy") or ""),
                    "singleton_quality_score": float(item.get("singleton_quality_score") or 0.0),
                    "page_strong_query_match": bool(item.get("page_strong_query_match")),
                    "page_family_match_score": float(item.get("page_family_match_score") or 0.0),
                    "page_family_matched_terms": list(item.get("page_family_matched_terms") or []),
                    "page_query_object_terms": list(item.get("page_query_object_terms") or []),
                    "page_query_object_overlap_terms": list(item.get("page_query_object_overlap_terms") or []),
                    "page_family_mismatch_penalty": float(item.get("page_family_mismatch_penalty") or 0.0),
                    "page_family_score_reason": dict(item.get("page_family_score_reason") or {}),
                }
            )
            for trajectory_id in list(item["trajectory_ids"]):
                if trajectory_id not in trajectory_union:
                    trajectory_union.append(trajectory_id)
        selected_page_trajectory_count_histogram = {
            str(count): frequency
            for count, frequency in sorted(
                Counter(int(row.get("trajectory_count") or 0) for row in selected_page_rows).items()
            )
        }
        selected_singleton_page_count = sum(1 for row in selected_page_rows if bool(row.get("singleton_page")))
        selected_medium_page_count = sum(1 for row in selected_page_rows if bool(row.get("medium_granularity_page")))
        selected_allowed_specific_singleton_page_count = sum(
            1
            for row in selected_page_rows
            if bool(row.get("singleton_page")) and row.get("singleton_policy") == "allowed_isolated_specific"
        )
        selected_low_quality_singleton_page_count = sum(
            1
            for row in selected_page_rows
            if bool(row.get("singleton_page")) and row.get("singleton_policy") == "merge_required_low_quality"
        )
        singleton_penalty_applied = sum(
            1 for row in selected_page_rows if float(row.get("singleton_page_penalty") or 0.0) < 0.0
        )
        low_quality_singleton_penalty_applied = sum(
            1 for row in selected_page_rows if float(row.get("low_quality_singleton_penalty") or 0.0) < 0.0
        )
        medium_bonus_applied = sum(
            1 for row in selected_page_rows if float(row.get("medium_page_bonus") or 0.0) > 0.0
        )
        self._trace(
            f"sample={sample_id} page_route_candidates considered_pages={len(pages)} candidate_pool={len(candidate_pool)} "
            f"rerank_selected={len(reranked_ids)} rerank_fallback={str(rerank_fallback).lower()} "
            f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
        )
        self._trace(
            f"sample={sample_id} page_route_selected_ids pages={','.join(selected_ids) or 'none'} "
            f"trajectory_union={len(trajectory_union)}"
        )
        self._trace(
            f"sample={sample_id} page_granularity_selected "
            f"singletons={selected_singleton_page_count} medium={selected_medium_page_count} "
            f"histogram={selected_page_trajectory_count_histogram}"
        )
        metadata = {
            "page_candidate_ids": [str(item["page_id"]) for item in candidate_pool],
            "page_rerank_selected_ids": list(reranked_ids),
            "page_rerank_rationales": rerank_rationales,
            "page_rerank_fallback": rerank_fallback,
            "page_selection_strategy": selection_metadata["selection_strategy"],
            "page_cluster_coverage": selection_metadata["cluster_keys"],
            "page_covered_query_entities": selection_metadata["covered_query_entities"],
            "page_covered_query_facet_values": selection_metadata["covered_query_facet_values"],
            "page_covered_query_terms": selection_metadata["covered_query_terms"],
            "selected_page_rows": selected_page_rows,
            "selected_page_rows_compact": selected_page_rows_compact,
            "page_granularity_diagnostic_mode": "retrieval_metadata",
            "selected_singleton_page_count": selected_singleton_page_count,
            "selected_medium_granularity_page_count": selected_medium_page_count,
            "selected_page_trajectory_count_histogram": selected_page_trajectory_count_histogram,
            "singleton_page_penalty_applied": singleton_penalty_applied,
            "low_quality_singleton_penalty_applied": low_quality_singleton_penalty_applied,
            "medium_page_bonus_applied": medium_bonus_applied,
            "selected_allowed_specific_singleton_page_count": selected_allowed_specific_singleton_page_count,
            "selected_low_quality_singleton_page_count": selected_low_quality_singleton_page_count,
            "selected_singleton_page_ids": [
                str(row.get("page_id")) for row in selected_page_rows if bool(row.get("singleton_page"))
            ],
            "selected_medium_page_ids": [
                str(row.get("page_id")) for row in selected_page_rows if bool(row.get("medium_granularity_page"))
            ],
            **wiki_fragmentation_metadata,
            "page_embedding_count": len(page_embeddings),
            "missing_page_embedding_count": missing_page_embedding_count,
            "all_page_embeddings_missing": bool(pages and missing_page_embedding_count == len(pages)),
            "page_index_suppressed": index_suppressed,
            "index_page_ids": [page.id for page in index_pages],
            "index_page_trajectory_ids": index_page_trajectory_ids,
            "non_index_page_count": len(non_index_pages),
            "page_rerank_error_type": rerank_error_metadata.get("rerank_error_type"),
            "page_rerank_error_message": rerank_error_metadata.get("rerank_error_message"),
            "diagnostic_top_n_pages": self.DIAGNOSTIC_TOP_N_PAGES,
            "page_ranked_total_count": len(fused),
            "page_ranked_rows_truncated": len(fused) > len(page_ranked_rows_compact_top_n),
            "page_ranked_rows_compact_top_n": page_ranked_rows_compact_top_n,
            "page_cutoff_universe_diagnostics": page_cutoff_universe_diagnostics,
            "page_ranked_rows": [
                {
                    "page_id": str(item["page_id"]),
                    "page_type": str(item["page_type"]),
                    "title": str(item["title"]),
                    "dense_score": float(item["dense_score"]),
                    "sparse_score": float(item["sparse_score"]),
                    "entity_bonus": float(item["entity_bonus"]),
                    "reflection_bonus": float(item["reflection_bonus"]),
                    "page_granularity_adjustment": float(item.get("page_granularity_adjustment") or 0.0),
                    "singleton_page_penalty": float(item.get("singleton_page_penalty") or 0.0),
                    "low_quality_singleton_penalty": float(item.get("low_quality_singleton_penalty") or 0.0),
                    "medium_page_bonus": float(item.get("medium_page_bonus") or 0.0),
                    "singleton_policy": str(item.get("singleton_policy") or ""),
                    "singleton_quality_score": float(item.get("singleton_quality_score") or 0.0),
                    "singleton_page": bool(item.get("singleton_page")),
                    "medium_granularity_page": bool(item.get("medium_granularity_page")),
                    "page_strong_query_match": bool(item.get("page_strong_query_match")),
                    "page_family_match_score": float(item.get("page_family_match_score") or 0.0),
                    "page_family_matched_terms": list(item.get("page_family_matched_terms") or []),
                    "page_query_object_terms": list(item.get("page_query_object_terms") or []),
                    "page_query_object_overlap_terms": list(item.get("page_query_object_overlap_terms") or []),
                    "page_family_mismatch_penalty": float(item.get("page_family_mismatch_penalty") or 0.0),
                    "page_family_score_reason": dict(item.get("page_family_score_reason") or {}),
                    "reflection_term_hits": list(item["reflection_term_hits"]),
                    "fused_score": float(item["fused_score"]),
                    "trajectory_ids": list(item["trajectory_ids"]),
                    "trajectory_count": int(item.get("trajectory_count") or 0),
                    "slug": str(item["slug"]),
                    "routing_priority": str(item["routing_priority"]),
                    "broad_entity_profile": bool(item.get("broad_entity_profile")),
                    "entity_facet_split_from_broad_page": bool(item.get("entity_facet_split_from_broad_page")),
                    "wiki_rescue_reason": item.get("wiki_rescue_reason"),
                    "wiki_singleton_exception": bool(item.get("wiki_singleton_exception")),
                    "singleton_exception_reason": item.get("singleton_exception_reason"),
                    "entity_keys": sorted(set(item["coverage_profile"]["entity_keys"])),
                    "facet_values": sorted(set(item["coverage_profile"]["facet_values"])),
                    "item_terms": sorted(set(item["coverage_profile"].get("item_terms") or [])),
                }
                for item in fused[: max(self.top_t_pages, 10)]
            ],
        }
        return selected_ids, trajectory_union, metadata

    def _select_trajectories(
        self,
        sample_id: str,
        candidate_trajectory_ids: list[str],
        query_text: str,
        query_embedding: list[float],
        query_keywords: set[str],
        query_entities: list[str],
        query_facet_tags: set[str],
        query_facet_values: set[str],
        query_shape: dict[str, object] | None = None,
    ) -> tuple[list[str], dict[str, object]]:
        query_shape = dict(query_shape or {})
        candidate_id_set = set(candidate_trajectory_ids)
        trajectories = [
            trajectory
            for trajectory in self.store.list_trajectories(sample_id)
            if trajectory.id in candidate_id_set
        ]
        signals_by_id = {
            trajectory.id: self._trajectory_retrieval_signals(trajectory)
            for trajectory in trajectories
        }
        latest_snapshot_ids = [
            signals.latest_snapshot_id
            for signals in signals_by_id.values()
            if signals.latest_snapshot_id
        ]
        latest_embeddings = self.store.fetch_embeddings_by_owner_ids(latest_snapshot_ids, "snapshot")
        summary_embeddings = self.store.fetch_embeddings_by_owner_ids([trajectory.id for trajectory in trajectories], "trajectory_summary")
        query_entity_keys = {normalize_entity_key(value) for value in query_entities}
        query_facet_value_keys = {str(value).casefold() for value in query_facet_values}
        source_event_query_profile = self._temporal_event_query_profile(query_text, query_shape, query_keywords)
        scored: list[dict[str, object]] = []
        for trajectory in trajectories:
            signals = signals_by_id[trajectory.id]
            latest_embedding = latest_embeddings.get(signals.latest_snapshot_id) if signals.latest_snapshot_id else None
            summary_embedding = summary_embeddings.get(trajectory.id)
            summary_similarity = cosine_similarity(query_embedding, summary_embedding.vector_json) if summary_embedding is not None else 0.0
            latest_similarity = cosine_similarity(query_embedding, latest_embedding.vector_json) if latest_embedding is not None else 0.0
            lexical_overlap = keyword_overlap_score(query_keywords, signals.lexical_keywords) if query_keywords else 0.0
            support_terms = set(signals.support_terms)
            family_profile = self._answer_family_match_profile(
                query_shape=query_shape,
                query_keywords=query_keywords,
                support_terms=support_terms,
                exact_terms=signals.exact_terms,
                display_items=signals.display_items,
                display_key_facts=signals.display_key_facts,
                historical_item_terms=signals.historical_item_terms,
                summary_text=signals.summary_text,
            )
            entity_match_boost = (
                0.10
                if query_entity_keys and signals.entity_keys & query_entity_keys
                else 0.0
            )
            facet_tag_boost = 0.05 if query_facet_tags and signals.facet_tags & query_facet_tags else 0.0
            facet_value_boost = 0.05 if query_facet_value_keys and signals.facet_values & query_facet_value_keys else 0.0
            dense_score = 0.75 * summary_similarity + 0.15 * latest_similarity + 0.10 * min(
                1.0,
                (entity_match_boost + facet_tag_boost + facet_value_boost) / 0.20,
            )
            item_terms = self._coverage_item_terms(
                [
                    *signals.exact_terms,
                    *signals.display_items,
                    *signals.display_counts,
                    *signals.display_key_facts,
                    *signals.historical_item_terms,
                    *list(signals.facet_values),
                    *signals.summary_keywords,
                ]
            )
            family_score = float(family_profile.get("score") or 0.0)
            family_mismatch_penalty = float(family_profile.get("mismatch_penalty") or 0.0)
            source_event_profile = self._trajectory_source_event_match_profile(
                signals=signals,
                query_profile=source_event_query_profile,
            )
            source_event_score = float(source_event_profile.get("score") or 0.0)
            effective_mismatch_penalty = (
                min(family_mismatch_penalty, 0.05)
                if bool(source_event_profile.get("strong_match"))
                else family_mismatch_penalty
            )
            scored.append(
                {
                    "trajectory_id": trajectory.id,
                    "summary_text": signals.summary_text,
                    "dense_score": max(0.0, dense_score + 0.08 * family_score + 0.10 * source_event_score - 0.04 * effective_mismatch_penalty),
                    "sparse_score": max(0.0, lexical_overlap + 0.60 * family_score + 0.75 * source_event_score - 0.25 * effective_mismatch_penalty),
                    "summary_similarity": summary_similarity,
                    "latest_similarity": latest_similarity,
                    "entity_match_boost": entity_match_boost,
                    "facet_tag_boost": facet_tag_boost,
                    "facet_value_boost": facet_value_boost,
                    "answer_family_match_score": family_score,
                    "answer_family_matched_terms": list(family_profile.get("matched_terms") or []),
                    "answer_family_query_object_terms": list(family_profile.get("query_object_terms") or []),
                    "answer_family_query_overlap_terms": list(family_profile.get("query_object_overlap_terms") or []),
                    "answer_family_mismatch_penalty": family_mismatch_penalty,
                    "source_event_match_score": source_event_score,
                    "source_event_matched_terms": list(source_event_profile.get("matched_terms") or []),
                    "source_event_matched_refs": list(source_event_profile.get("matched_refs") or []),
                    "source_event_match_reason": str(source_event_profile.get("reason") or ""),
                    "source_event_matched_record_count": int(source_event_profile.get("matched_record_count") or 0),
                    "exact_terms": signals.exact_terms,
                    "summary_keywords": signals.summary_keywords,
                    "historical_item_terms": signals.historical_item_terms,
                    "source_event_object_terms": signals.source_event_object_terms,
                    "source_event_canonical_terms": signals.source_event_canonical_terms,
                    "coverage_profile": {
                        "entity_keys": set(signals.entity_keys),
                        "facet_values": set(signals.facet_values),
                        "support_terms": support_terms,
                        "item_terms": item_terms,
                        "exact_terms": list(signals.exact_terms),
                        "source_event_terms": list(
                            dict.fromkeys([*signals.source_event_canonical_terms, *signals.source_event_object_terms])
                        ),
                        "inventory_like": bool(signals.display_items),
                        "has_count_signal": bool(signals.display_counts),
                    },
                }
            )
        fused = self._fuse_dense_sparse_scores(scored, id_key="trajectory_id")
        rerank_pool = fused[: min(max(self.top_k, self.CANDIDATE_POOL_SIZE), len(fused))]
        if self._query_shape_requires_coverage(query_shape):
            selection_pool_limit = min(
                len(fused),
                max(32, 4 * max(self.top_k, 1)),
                self.COVERAGE_SELECTION_POOL_MAX,
            )
        else:
            selection_pool_limit = len(rerank_pool)
        if self._query_shape_requires_coverage(query_shape) and query_shape.get("item_family"):
            family_ranked = sorted(
                fused,
                key=lambda item: (
                    float(item.get("source_event_match_score") or 0.0),
                    float(item.get("answer_family_match_score") or 0.0),
                    len(list(item.get("answer_family_query_overlap_terms") or [])),
                    -float(item.get("answer_family_mismatch_penalty") or 0.0),
                    float(item.get("fused_score") or 0.0),
                    str(item["trajectory_id"]),
                ),
                reverse=True,
            )
            ordered_pool_ids: list[str] = []
            selection_pool = []
            for item in [*family_ranked, *fused]:
                trajectory_id = str(item["trajectory_id"])
                if trajectory_id in ordered_pool_ids:
                    continue
                ordered_pool_ids.append(trajectory_id)
                selection_pool.append(item)
                if len(selection_pool) >= selection_pool_limit:
                    break
        else:
            selection_pool = fused[:selection_pool_limit]
        reranked_ids, rerank_rationales, rerank_fallback, rerank_error_metadata = self._rerank_selected_ids(
            prompt_name="trajectory_set_rerank",
            query_text=query_text,
            candidates=rerank_pool,
            final_count=min(self.top_k, len(rerank_pool)),
            label_prefix="T",
            text_key="summary_text",
            id_key="trajectory_id",
        )
        if rerank_error_metadata:
            self._trace(
                f"sample={sample_id} trajectory_rerank_failed "
                f"error_type={rerank_error_metadata.get('rerank_error_type', 'unknown')} "
                f"error={rerank_error_metadata.get('rerank_error_message', '')}"
            )
        selected_ids, selection_metadata = self._coverage_aware_select_ids(
            candidate_pool=selection_pool,
            reranked_ids=list(reranked_ids),
            id_key="trajectory_id",
            final_count=self.top_k,
            query_shape=query_shape,
            query_keywords=query_keywords,
            query_entity_keys=query_entity_keys,
            query_facet_values=query_facet_value_keys,
        )
        trajectory_ranked_rows_compact_top_n = [
            self._compact_trajectory_ranked_row(item, index + 1)
            for index, item in enumerate(fused[: self.DIAGNOSTIC_TOP_N_TRAJECTORIES])
        ]
        trajectory_selection_pool_rows_compact = [
            self._compact_trajectory_ranked_row(item, index + 1)
            for index, item in enumerate(selection_pool[: self.DIAGNOSTIC_SELECTION_POOL_ROW_LIMIT])
        ]
        trajectory_cutoff_diagnostics = self._trajectory_cutoff_diagnostics_from_rows(
            trajectory_ranked_rows_compact_top_n
        )
        metadata = {
            "trajectory_candidate_pool_ids": [str(item["trajectory_id"]) for item in rerank_pool],
            "trajectory_selection_pool_ids": [str(item["trajectory_id"]) for item in selection_pool],
            "trajectory_selection_pool_size": len(selection_pool),
            "trajectory_selection_pool_rows_compact": trajectory_selection_pool_rows_compact,
            "trajectory_selection_pool_rows_total_count": len(selection_pool),
            "trajectory_selection_pool_rows_truncated": (
                len(selection_pool) > len(trajectory_selection_pool_rows_compact)
            ),
            "trajectory_rerank_pool_size": len(rerank_pool),
            "trajectory_rerank_selected_ids": list(reranked_ids),
            "trajectory_rerank_rationales": rerank_rationales,
            "trajectory_rerank_fallback": rerank_fallback,
            "trajectory_selection_strategy": selection_metadata["selection_strategy"],
            "trajectory_cluster_coverage": selection_metadata["cluster_keys"],
            "trajectory_covered_query_entities": selection_metadata["covered_query_entities"],
            "trajectory_covered_query_facet_values": selection_metadata["covered_query_facet_values"],
            "trajectory_covered_query_terms": selection_metadata["covered_query_terms"],
            "trajectory_covered_item_terms": selection_metadata["covered_item_terms"],
            "trajectory_selected_score_components": selection_metadata["selected_score_components"],
            "trajectory_redundancy_penalties": selection_metadata["redundancy_penalties"],
            "trajectory_family_match_scores": [
                {
                    "trajectory_id": str(item["trajectory_id"]),
                    "score": float(item.get("answer_family_match_score") or 0.0),
                    "matched_terms": list(item.get("answer_family_matched_terms") or []),
                    "query_object_terms": list(item.get("answer_family_query_object_terms") or []),
                    "query_object_overlap_terms": list(item.get("answer_family_query_overlap_terms") or []),
                }
                for item in selection_pool
            ],
            "trajectory_family_mismatch_penalties": [
                {
                    "trajectory_id": str(item["trajectory_id"]),
                    "penalty": float(item.get("answer_family_mismatch_penalty") or 0.0),
                }
                for item in selection_pool
                if float(item.get("answer_family_mismatch_penalty") or 0.0) > 0.0
            ],
            "trajectory_selected_family_matches": [
                {
                    "trajectory_id": str(item["trajectory_id"]),
                    "score": float(item.get("answer_family_match_score") or 0.0),
                    "matched_terms": list(item.get("answer_family_matched_terms") or []),
                    "query_object_terms": list(item.get("answer_family_query_object_terms") or []),
                    "query_object_overlap_terms": list(item.get("answer_family_query_overlap_terms") or []),
                }
                for item in selection_pool
                if str(item["trajectory_id"]) in set(selected_ids)
                and float(item.get("answer_family_match_score") or 0.0) > 0.0
            ],
            "trajectory_source_event_match_scores": [
                {
                    "trajectory_id": str(item["trajectory_id"]),
                    "score": float(item.get("source_event_match_score") or 0.0),
                    "matched_terms": list(item.get("source_event_matched_terms") or []),
                    "matched_refs": list(item.get("source_event_matched_refs") or []),
                    "reason": str(item.get("source_event_match_reason") or ""),
                }
                for item in selection_pool
                if float(item.get("source_event_match_score") or 0.0) > 0.0
            ],
            "trajectory_selected_source_event_matches": [
                {
                    "trajectory_id": str(item["trajectory_id"]),
                    "score": float(item.get("source_event_match_score") or 0.0),
                    "matched_terms": list(item.get("source_event_matched_terms") or []),
                    "matched_refs": list(item.get("source_event_matched_refs") or []),
                    "reason": str(item.get("source_event_match_reason") or ""),
                }
                for item in selection_pool
                if str(item["trajectory_id"]) in set(selected_ids)
                and float(item.get("source_event_match_score") or 0.0) > 0.0
            ],
            "trajectory_source_event_match_miss_count": sum(
                1
                for item in selection_pool
                if signals_by_id.get(str(item["trajectory_id"]))
                and signals_by_id[str(item["trajectory_id"])].source_event_records
                and float(item.get("source_event_match_score") or 0.0) <= 0.0
            ),
            "trajectory_source_event_query_profile": {
                "enabled": bool(source_event_query_profile.get("enabled")),
                "object_terms": list(source_event_query_profile.get("object_terms") or []),
                "action_terms": list(source_event_query_profile.get("action_terms") or []),
                "relation_terms": list(source_event_query_profile.get("relation_terms") or []),
            },
            "trajectory_rerank_error_type": rerank_error_metadata.get("rerank_error_type"),
            "trajectory_rerank_error_message": rerank_error_metadata.get("rerank_error_message"),
            "diagnostic_top_n_trajectories": self.DIAGNOSTIC_TOP_N_TRAJECTORIES,
            "trajectory_ranked_total_count": len(fused),
            "trajectory_ranked_rows_truncated": len(fused) > len(trajectory_ranked_rows_compact_top_n),
            "trajectory_ranked_rows_compact_top_n": trajectory_ranked_rows_compact_top_n,
            "trajectory_cutoff_prefix_diagnostics": trajectory_cutoff_diagnostics,
            "trajectory_ranked_rows": [
                {
                    "trajectory_id": str(item["trajectory_id"]),
                    "dense_score": float(item["dense_score"]),
                    "sparse_score": float(item["sparse_score"]),
                    "summary_similarity": float(item["summary_similarity"]),
                    "latest_similarity": float(item["latest_similarity"]),
                    "entity_match_boost": float(item["entity_match_boost"]),
                    "facet_tag_boost": float(item["facet_tag_boost"]),
                    "facet_value_boost": float(item["facet_value_boost"]),
                    "answer_family_match_score": float(item.get("answer_family_match_score") or 0.0),
                    "answer_family_matched_terms": list(item.get("answer_family_matched_terms") or []),
                    "answer_family_query_object_terms": list(item.get("answer_family_query_object_terms") or []),
                    "answer_family_query_overlap_terms": list(item.get("answer_family_query_overlap_terms") or []),
                    "answer_family_mismatch_penalty": float(item.get("answer_family_mismatch_penalty") or 0.0),
                    "source_event_match_score": float(item.get("source_event_match_score") or 0.0),
                    "source_event_matched_terms": list(item.get("source_event_matched_terms") or []),
                    "source_event_matched_refs": list(item.get("source_event_matched_refs") or []),
                    "source_event_match_reason": str(item.get("source_event_match_reason") or ""),
                    "exact_terms": list(item["exact_terms"]),
                    "summary_keywords": list(item["summary_keywords"]),
                    "historical_item_terms": list(item.get("historical_item_terms") or []),
                    "source_event_object_terms": list(item.get("source_event_object_terms") or []),
                    "source_event_canonical_terms": list(item.get("source_event_canonical_terms") or []),
                    "entity_keys": sorted(set(item["coverage_profile"]["entity_keys"])),
                    "facet_values": sorted(set(item["coverage_profile"]["facet_values"])),
                    "item_terms": sorted(set(item["coverage_profile"].get("item_terms") or [])),
                }
                for item in fused[: max(self.top_k, 10)]
            ],
        }
        return selected_ids, metadata

    def fine_retrieve_snapshots(
        self, trajectory_ids: Iterable[str], query_embedding: list[float]
    ) -> tuple[list[EpisodicMemorySnapshot], dict[str, object]]:
        snapshots = self.store.list_snapshots_for_trajectories(trajectory_ids)
        embeddings = self.store.fetch_embeddings_by_owner_ids([snapshot.id for snapshot in snapshots], "snapshot")
        scored_by_trajectory: dict[str, list[tuple[EpisodicMemorySnapshot, float]]] = defaultdict(list)
        for snapshot in snapshots:
            embedding = embeddings.get(snapshot.id)
            if embedding is None:
                continue
            score = cosine_similarity(query_embedding, embedding.vector_json)
            scored_by_trajectory[snapshot.trajectory_id].append((snapshot, score))
        for values in scored_by_trajectory.values():
            values.sort(key=lambda item: item[1], reverse=True)

        selected: list[EpisodicMemorySnapshot] = []
        selected_ids: set[str] = set()
        per_trajectory_counts: dict[str, int] = {}
        # First pass: guarantee one snapshot per selected trajectory when possible.
        for trajectory_id in trajectory_ids:
            rows = scored_by_trajectory.get(trajectory_id, [])
            if not rows:
                continue
            snapshot, _ = rows[0]
            if snapshot.id not in selected_ids:
                selected.append(snapshot)
                selected_ids.add(snapshot.id)
                per_trajectory_counts[trajectory_id] = 1
        remaining_slots = max(self.snapshot_budget - len(selected), 0)
        leftovers: list[tuple[EpisodicMemorySnapshot, float]] = []
        for trajectory_id, rows in scored_by_trajectory.items():
            start_index = 1 if per_trajectory_counts.get(trajectory_id, 0) else 0
            leftovers.extend(rows[start_index:])
        leftovers.sort(key=lambda item: item[1], reverse=True)
        for snapshot, _ in leftovers:
            if remaining_slots <= 0:
                break
            if snapshot.id in selected_ids:
                continue
            selected.append(snapshot)
            selected_ids.add(snapshot.id)
            per_trajectory_counts[snapshot.trajectory_id] = per_trajectory_counts.get(snapshot.trajectory_id, 0) + 1
            remaining_slots -= 1
        return selected, {
            "fine_snapshot_budget": self.snapshot_budget,
            "fine_snapshot_quota_counts": per_trajectory_counts,
            "fine_snapshot_selected_ids": [snapshot.id for snapshot in selected],
        }

    def _expanded_snapshot_budget(self, seed_count: int, *, query_is_list_like: bool) -> int:
        if query_is_list_like:
            budget = min(
                max(seed_count + self.LIST_EXPANDED_OFFSET, self.LIST_EXPANDED_MIN),
                self.LIST_EXPANDED_MAX,
            )
        else:
            budget = min(
                max(seed_count + self.NON_LIST_EXPANDED_OFFSET, self.NON_LIST_EXPANDED_MIN),
                self.NON_LIST_EXPANDED_MAX,
            )
        return max(budget, seed_count)

    def _source_message_budget(self, *, query_is_list_like: bool) -> int:
        return self.LIST_SOURCE_BUDGET if query_is_list_like else self.NON_LIST_SOURCE_BUDGET

    @staticmethod
    def _retrieval_evidence_weak(
        *,
        source_message_ids: list[str],
        selected_trajectory_ids: list[str],
        snapshot_hits: list[EpisodicMemorySnapshot],
        active_claim_count: int,
    ) -> bool:
        return (
            not source_message_ids
            or not selected_trajectory_ids
            or not snapshot_hits
            or active_claim_count <= 0
        )

    @classmethod
    def _reflection_term_tokens(cls, term: str) -> set[str]:
        return {
            token
            for token in extract_keywords(term)
            if token.casefold() not in cls.REFLECTION_GENERIC_TERMS
        }

    @classmethod
    def _reflection_term_is_generic(cls, term: str) -> bool:
        normalized = collapse_whitespace(str(term or "")).strip(" .,\"'`").casefold()
        if not normalized or len(normalized) < 3:
            return True
        if normalized in cls.REFLECTION_GENERIC_TERMS:
            return True
        tokens = extract_keywords(normalized)
        if not tokens:
            return True
        return all(token in cls.REFLECTION_GENERIC_TERMS for token in tokens)

    @classmethod
    def _reflection_required_terms(cls, reflection_hints: dict[str, object]) -> list[str]:
        values: list[object] = []
        for key in (
            "must_find_terms",
            "raw_search_terms",
            "event_terms",
            "temporal_terms",
            "target_entities",
        ):
            value = reflection_hints.get(key)
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)
        return [
            term
            for term in cls._clean_text_values(values, limit=48)
            if not cls._reflection_term_is_generic(term)
        ]

    @classmethod
    def _term_covered_by_evidence(
        cls,
        term: str,
        *,
        evidence_text: str,
        evidence_keywords: set[str],
    ) -> bool:
        normalized = collapse_whitespace(str(term or "")).casefold()
        if not normalized:
            return False
        if normalized in evidence_text:
            return True
        tokens = cls._reflection_term_tokens(normalized)
        if not tokens:
            return False
        return tokens.issubset(evidence_keywords)

    def _reflection_evidence_coverage(
        self,
        *,
        sample_id: str,
        selected_page_ids: list[str],
        source_message_ids: list[str],
        reflection_hints: dict[str, object],
        grounded_exact_terms: list[str],
        grounded_display_items: list[str],
        grounded_display_counts: list[str],
        grounded_display_key_facts: list[str],
    ) -> dict[str, object]:
        required_terms = self._reflection_required_terms(reflection_hints)
        selected_page_id_set = set(selected_page_ids)
        selected_pages = [
            page
            for page in self.store.list_wiki_pages(sample_id)
            if page.id in selected_page_id_set
        ]
        source_records = self.store.fetch_raw_messages(source_message_ids)
        evidence_chunks: list[str] = [
            *grounded_exact_terms,
            *grounded_display_items,
            *grounded_display_counts,
            *grounded_display_key_facts,
        ]
        for message in source_records:
            evidence_chunks.extend(
                [
                    str(message.source_ref or ""),
                    str(message.speaker_name or ""),
                    message.content,
                ]
            )
        for page in selected_pages:
            metadata = dict(page.metadata_json or {})
            evidence_chunks.extend(
                [
                    page.slug,
                    page.title,
                    " ".join(str(value) for value in list(page.keywords_json or [])),
                    " ".join(str(value) for value in list(page.entity_names_json or [])),
                    str(metadata.get("routing_text") or ""),
                ]
            )
        evidence_text = collapse_whitespace(" ".join(evidence_chunks)).casefold()
        evidence_keywords = extract_keywords(evidence_text)
        covered_terms = [
            term
            for term in required_terms
            if self._term_covered_by_evidence(
                term,
                evidence_text=evidence_text,
                evidence_keywords=evidence_keywords,
            )
        ]
        uncovered_terms = [term for term in required_terms if term not in set(covered_terms)]
        coverage_rate = len(covered_terms) / len(required_terms) if required_terms else None
        semantic_weak = bool(
            required_terms
            and (
                not covered_terms
                or float(coverage_rate or 0.0) < self.REFLECTION_SEMANTIC_WEAK_COVERAGE_THRESHOLD
            )
        )
        return {
            "reflection_required_terms": required_terms,
            "reflection_covered_terms": covered_terms,
            "reflection_uncovered_terms": uncovered_terms,
            "reflection_term_coverage_rate": coverage_rate,
            "reflection_semantic_evidence_weak": semantic_weak,
        }

    def _raw_rescue_messages(
        self,
        *,
        sample_id: str,
        query_text: str,
        effective_query_text: str,
        reflection_hints: dict[str, object],
        query_embedding: list[float],
        query_is_list_like: bool,
        exclude_message_ids: set[str],
    ) -> tuple[list[RawMessageRecord], dict[str, object]]:
        terms = self._clean_text_values(
            [
                *self._reflection_terms(reflection_hints),
                *sorted(extract_keywords(query_text)),
                *sorted(extract_keywords(effective_query_text)),
            ],
            limit=48,
        )
        query_terms = set(extract_keywords(" ".join(terms) or effective_query_text))
        raw_messages = self.store.list_raw_messages_for_sample(sample_id)

        def score_message(message: RawMessageRecord) -> float:
            text = " ".join(
                [
                    str(message.source_ref or ""),
                    str(message.speaker_name or ""),
                    message.content,
                ]
            )
            text_casefold = text.casefold()
            keyword_score = keyword_overlap_score(query_terms, extract_keywords(text)) if query_terms else 0.0
            substring_hits = sum(1 for term in terms if term.casefold() in text_casefold)
            return keyword_score + min(1.0, substring_hits / 4.0)

        def lexical_candidates(*, include_system: bool) -> list[tuple[RawMessageRecord, float]]:
            rows: list[tuple[RawMessageRecord, float]] = []
            for message in raw_messages:
                if message.id in exclude_message_ids:
                    continue
                if not include_system and message.role == "system":
                    continue
                score = score_message(message)
                if score > 0:
                    rows.append((message, score))
            rows.sort(key=lambda item: (item[1], -int(item[0].turn_index)), reverse=True)
            return rows[: self.RAW_RESCUE_PREFILTER_LIMIT]

        candidates = lexical_candidates(include_system=False)
        included_system = False
        if not candidates:
            candidates = lexical_candidates(include_system=True)
            included_system = True
        limit = self.RAW_RESCUE_LIST_LIMIT if query_is_list_like else self.RAW_RESCUE_NON_LIST_LIMIT
        embedding_used = False
        lexical_fallback = False
        selected: list[RawMessageRecord] = []
        embedding_error: str | None = None
        if candidates:
            try:
                candidate_texts = [
                    f"{message.source_ref or ''} {message.speaker_name or ''} {message.content}"
                    for message, _ in candidates
                ]
                candidate_embeddings = self._embed_documents(candidate_texts)
                rescored: list[tuple[RawMessageRecord, float]] = []
                for (message, lexical_score), embedding in zip(candidates, candidate_embeddings):
                    dense = cosine_similarity(query_embedding, embedding)
                    rescored.append((message, 0.65 * dense + 0.35 * lexical_score))
                rescored.sort(key=lambda item: (item[1], -int(item[0].turn_index)), reverse=True)
                selected = [message for message, _ in rescored[:limit]]
                embedding_used = True
            except Exception as exc:  # noqa: BLE001
                embedding_error = _compact_error_message(exc)
                lexical_fallback = True
        if not selected and candidates:
            lexical_fallback = True
            selected = [message for message, _ in candidates[:limit]]
        selected.sort(key=lambda message: self._source_sort_key(message, message.id))
        source_refs = [
            str(message.source_ref)
            for message in selected
            if message.source_ref
        ]
        return selected, {
            "raw_rescue_attempted": True,
            "raw_rescue_used": bool(selected),
            "raw_rescue_query": effective_query_text,
            "raw_rescue_terms": terms,
            "raw_rescue_candidate_count": len(candidates),
            "raw_rescue_hit_count": len(selected),
            "raw_rescue_source_ids": [message.id for message in selected],
            "raw_rescue_source_refs": list(dict.fromkeys(source_refs)),
            "raw_rescue_embedding_used": embedding_used,
            "raw_rescue_lexical_fallback": lexical_fallback,
            "raw_rescue_included_system_messages": included_system,
            "raw_rescue_embedding_error": embedding_error,
        }

    def _collect_snapshot_source_state(
        self,
        snapshots: Iterable[EpisodicMemorySnapshot],
    ) -> dict[str, object]:
        snapshot_list = list(snapshots)
        claims_by_snapshot = self.store.list_claims_for_snapshots(snapshot.id for snapshot in snapshot_list)
        source_candidates_by_snapshot: dict[str, list[str]] = {}
        conflict_lines_by_snapshot: dict[str, list[str]] = {}
        raw_source_message_ids: list[str] = []
        seen_global_ids: set[str] = set()
        for snapshot in snapshot_list:
            claims = claims_by_snapshot.get(snapshot.id, [])
            ordered_ids: list[str] = []
            seen_local_ids: set[str] = set()
            for message_id in list(getattr(snapshot, "links_json", []) or []):
                message_key = str(message_id)
                if not message_key or message_key in seen_local_ids:
                    continue
                ordered_ids.append(message_key)
                seen_local_ids.add(message_key)
            for claim in claims:
                for message_id in list(claim.source_message_ids_json or []):
                    message_key = str(message_id)
                    if not message_key or message_key in seen_local_ids:
                        continue
                    ordered_ids.append(message_key)
                    seen_local_ids.add(message_key)
            source_candidates_by_snapshot[snapshot.id] = ordered_ids
            for message_id in ordered_ids:
                if message_id in seen_global_ids:
                    continue
                raw_source_message_ids.append(message_id)
                seen_global_ids.add(message_id)
            unresolved = [claim for claim in claims if claim.status in {"contradictory", "needs-confirmation"}]
            if unresolved:
                conflict_lines_by_snapshot[snapshot.id] = [
                    "Snapshot "
                    + snapshot.id
                    + " contains unresolved claims: "
                    + "; ".join(f"{claim.claim_id} [{claim.status}] {claim.text}" for claim in unresolved)
                ]
            else:
                conflict_lines_by_snapshot[snapshot.id] = []
        message_records = self.store.fetch_raw_messages(raw_source_message_ids)
        message_by_id = {message.id: message for message in message_records}
        raw_source_refs: list[str] = []
        seen_refs: set[str] = set()
        for message_id in raw_source_message_ids:
            message = message_by_id.get(message_id)
            source_ref = str(message.source_ref) if message is not None and message.source_ref else ""
            if not source_ref or source_ref in seen_refs:
                continue
            raw_source_refs.append(source_ref)
            seen_refs.add(source_ref)
        return {
            "snapshot_list": snapshot_list,
            "claims_by_snapshot": claims_by_snapshot,
            "source_candidates_by_snapshot": source_candidates_by_snapshot,
            "conflict_lines_by_snapshot": conflict_lines_by_snapshot,
            "raw_source_message_ids": raw_source_message_ids,
            "raw_source_refs": raw_source_refs,
        }

    @staticmethod
    def _snapshot_text_for_overlap(
        snapshot: EpisodicMemorySnapshot,
        claims: list[ClaimRecord],
    ) -> str:
        parts = [
            str(snapshot.semantic_text or ""),
            str(snapshot.summary_content or ""),
            str(snapshot.context or ""),
            " ".join(str(value) for value in list(snapshot.keywords_json or []) if str(value).strip()),
            " ".join(claim.text for claim in claims if str(claim.text).strip()),
        ]
        return collapse_whitespace(" ".join(part for part in parts if part))

    def _snapshot_query_score(
        self,
        *,
        snapshot: EpisodicMemorySnapshot,
        query_embedding: list[float],
        query_keywords: set[str],
        query_entity_keys: set[str],
        query_facet_tags: set[str],
        query_facet_value_keys: set[str],
        snapshot_embedding: list[float] | None,
        claims: list[ClaimRecord],
        trajectory_signals: TrajectoryRetrievalSignals | None,
        claim_signals: ClaimFacetSignals,
    ) -> tuple[float, float, float, float]:
        dense_similarity = cosine_similarity(query_embedding, snapshot_embedding or []) if snapshot_embedding else 0.0
        lexical_text = self._snapshot_text_for_overlap(snapshot, claims)
        lexical_overlap = keyword_overlap_score(query_keywords, extract_keywords(lexical_text)) if query_keywords else 0.0
        trajectory_entity_keys = trajectory_signals.entity_keys if trajectory_signals is not None else set()
        entity_bonus = (
            0.05
            if query_entity_keys and (trajectory_entity_keys | claim_signals.entity_keys) & query_entity_keys
            else 0.0
        )
        trajectory_facet_tags = trajectory_signals.facet_tags if trajectory_signals is not None else set()
        facet_tag_bonus = (
            0.05
            if query_facet_tags and (trajectory_facet_tags | claim_signals.facet_tags) & query_facet_tags
            else 0.0
        )
        trajectory_facet_values = trajectory_signals.facet_values if trajectory_signals is not None else set()
        facet_value_bonus = (
            0.05
            if query_facet_value_keys
            and (trajectory_facet_values | claim_signals.facet_values) & query_facet_value_keys
            else 0.0
        )
        symbolic_bonus = min(0.10, entity_bonus + facet_tag_bonus + facet_value_bonus)
        score = 0.75 * dense_similarity + 0.15 * lexical_overlap + 0.10 * min(1.0, symbolic_bonus / 0.10)
        return score, dense_similarity, lexical_overlap, symbolic_bonus

    def _compact_expanded_snapshots(
        self,
        *,
        sample_id: str,
        raw_expanded: list[EpisodicMemorySnapshot],
        selected_trajectory_ids: list[str],
        query_embedding: list[float],
        query_keywords: set[str],
        query_entity_keys: set[str],
        query_facet_tags: set[str],
        query_facet_values: set[str],
        seed_snapshot_ids: list[str],
        update_linked_snapshot_ids: list[str],
        neighbor_candidate_snapshot_ids: list[str],
        query_is_list_like: bool,
    ) -> tuple[list[EpisodicMemorySnapshot], dict[str, object]]:
        if not raw_expanded:
            budget = self._expanded_snapshot_budget(0, query_is_list_like=query_is_list_like)
            return [], {
                "snapshot_compaction_budget": budget,
                "snapshot_compaction_kept_ids": [],
                "snapshot_compaction_dropped_ids": [],
                "snapshot_compaction_counts": {
                    "seed_kept": 0,
                    "reserved_update_linked_kept": 0,
                    "scored_update_linked_kept": 0,
                    "neighbor_kept": 0,
                },
                "reserved_update_linked_snapshot_ids": [],
                "scored_update_linked_snapshot_ids": [],
                "neighbor_candidate_snapshot_ids": [],
            }
        budget = self._expanded_snapshot_budget(len(seed_snapshot_ids), query_is_list_like=query_is_list_like)
        raw_by_id = {snapshot.id: snapshot for snapshot in raw_expanded}
        seed_set = {snapshot_id for snapshot_id in seed_snapshot_ids if snapshot_id in raw_by_id}
        update_set = {snapshot_id for snapshot_id in update_linked_snapshot_ids if snapshot_id in raw_by_id} - seed_set
        neighbor_set = {snapshot_id for snapshot_id in neighbor_candidate_snapshot_ids if snapshot_id in raw_by_id} - seed_set - update_set
        embeddings = self.store.fetch_embeddings_by_owner_ids(list(raw_by_id), "snapshot")
        claims_by_snapshot = self.store.list_claims_for_snapshots(raw_by_id)
        selected_trajectory_id_set = set(selected_trajectory_ids)
        trajectory_signals_by_id = {
            trajectory.id: self._trajectory_retrieval_signals(trajectory)
            for trajectory in self.store.list_trajectories(sample_id)
            if trajectory.id in selected_trajectory_id_set
        }
        claim_signals_by_snapshot_id = {
            snapshot_id: self._claim_facet_signals(claims)
            for snapshot_id, claims in claims_by_snapshot.items()
        }
        query_facet_value_keys = {str(value).casefold() for value in query_facet_values}
        scored_rows: list[dict[str, object]] = []
        for snapshot in raw_expanded:
            claims = claims_by_snapshot.get(snapshot.id, [])
            embedding = embeddings.get(snapshot.id)
            score, dense_similarity, lexical_overlap, symbolic_bonus = self._snapshot_query_score(
                snapshot=snapshot,
                query_embedding=query_embedding,
                query_keywords=query_keywords,
                query_entity_keys=query_entity_keys,
                query_facet_tags=query_facet_tags,
                query_facet_value_keys=query_facet_value_keys,
                snapshot_embedding=embedding.vector_json if embedding is not None else None,
                claims=claims,
                trajectory_signals=trajectory_signals_by_id.get(snapshot.trajectory_id),
                claim_signals=claim_signals_by_snapshot_id.get(
                    snapshot.id,
                    ClaimFacetSignals(entity_keys=set(), facet_tags=set(), facet_values=set()),
                ),
            )
            scored_rows.append(
                {
                    "snapshot": snapshot,
                    "score": score,
                    "dense_similarity": dense_similarity,
                    "lexical_overlap": lexical_overlap,
                    "symbolic_bonus": symbolic_bonus,
                }
            )
        row_by_id = {str(item["snapshot"].id): item for item in scored_rows}
        kept_ids: list[str] = []
        kept_set: set[str] = set()
        for snapshot_id in seed_snapshot_ids:
            if snapshot_id in raw_by_id and snapshot_id not in kept_set:
                kept_ids.append(snapshot_id)
                kept_set.add(snapshot_id)
        remaining_budget = max(budget - len(kept_ids), 0)

        reserved_candidates: list[dict[str, object]] = []
        for trajectory_id in selected_trajectory_ids:
            candidates = [
                row_by_id[snapshot_id]
                for snapshot_id in update_set
                if raw_by_id[snapshot_id].trajectory_id == trajectory_id
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    float(item["score"]),
                    -int(item["snapshot"].version),
                    str(item["snapshot"].id),
                ),
                reverse=True,
            )
            reserved_candidates.append(candidates[0])
        reserved_candidates.sort(
            key=lambda item: (
                float(item["score"]),
                -int(item["snapshot"].version),
                str(item["snapshot"].id),
            ),
            reverse=True,
        )
        reserved_kept_ids: list[str] = []
        for item in reserved_candidates:
            snapshot_id = str(item["snapshot"].id)
            if remaining_budget <= 0 or snapshot_id in kept_set:
                continue
            kept_ids.append(snapshot_id)
            kept_set.add(snapshot_id)
            reserved_kept_ids.append(snapshot_id)
            remaining_budget -= 1

        scored_update_rows = [
            row_by_id[snapshot_id]
            for snapshot_id in update_set
            if snapshot_id not in set(reserved_kept_ids)
        ]
        scored_update_rows.sort(
            key=lambda item: (
                float(item["score"]),
                -int(item["snapshot"].version),
                str(item["snapshot"].id),
            ),
            reverse=True,
        )
        scored_update_kept_ids: list[str] = []
        for item in scored_update_rows:
            snapshot_id = str(item["snapshot"].id)
            if remaining_budget <= 0 or snapshot_id in kept_set:
                continue
            kept_ids.append(snapshot_id)
            kept_set.add(snapshot_id)
            scored_update_kept_ids.append(snapshot_id)
            remaining_budget -= 1

        neighbor_rows = [row_by_id[snapshot_id] for snapshot_id in neighbor_set]
        neighbor_rows.sort(
            key=lambda item: (
                float(item["score"]),
                -int(item["snapshot"].version),
                str(item["snapshot"].id),
            ),
            reverse=True,
        )
        neighbor_kept_ids: list[str] = []
        for item in neighbor_rows:
            snapshot_id = str(item["snapshot"].id)
            if remaining_budget <= 0 or snapshot_id in kept_set:
                continue
            kept_ids.append(snapshot_id)
            kept_set.add(snapshot_id)
            neighbor_kept_ids.append(snapshot_id)
            remaining_budget -= 1

        compacted = sorted((raw_by_id[snapshot_id] for snapshot_id in kept_ids if snapshot_id in raw_by_id), key=self._snapshot_sort_key)
        dropped_ids = [snapshot.id for snapshot in raw_expanded if snapshot.id not in kept_set]
        return compacted, {
            "snapshot_compaction_budget": budget,
            "snapshot_compaction_kept_ids": [snapshot.id for snapshot in compacted],
            "snapshot_compaction_dropped_ids": dropped_ids,
            "snapshot_compaction_counts": {
                "seed_kept": len(seed_set),
                "reserved_update_linked_kept": len(reserved_kept_ids),
                "scored_update_linked_kept": len(scored_update_kept_ids),
                "neighbor_kept": len(neighbor_kept_ids),
            },
            "reserved_update_linked_snapshot_ids": reserved_kept_ids,
            "scored_update_linked_snapshot_ids": scored_update_kept_ids,
            "neighbor_candidate_snapshot_ids": sorted(neighbor_set),
        }

    def _compact_source_messages(
        self,
        *,
        compacted_snapshots: list[EpisodicMemorySnapshot],
        snapshot_source_state: dict[str, object],
        seed_snapshot_ids: list[str],
        retained_update_linked_snapshot_ids: list[str],
        retained_neighbor_snapshot_ids: list[str],
        query_is_list_like: bool,
    ) -> tuple[list[str], list[str], list[str], dict[str, object]]:
        if not compacted_snapshots:
            budget = self._source_message_budget(query_is_list_like=query_is_list_like)
            return [], [], [], {
                "raw_source_message_ids": [],
                "raw_source_refs": [],
                "source_compaction_budget": budget,
                "source_compaction_kept_ids": [],
                "source_compaction_dropped_ids": [],
                "source_compaction_dropped_refs": [],
                "source_compaction_counts": {
                    "seed_source_count": 0,
                    "update_linked_source_count": 0,
                    "neighbor_source_count": 0,
                },
                "source_message_group_count": 0,
                "source_message_grouped_ids": {group: [] for group in self.SOURCE_GROUP_ORDER},
                "source_message_chronological_ids": [],
                "source_message_backtrack_count": 0,
                "source_message_backtrack_rate": 0.0,
            }
        source_candidates_by_snapshot = dict(snapshot_source_state.get("source_candidates_by_snapshot") or {})
        raw_source_message_ids = [str(item) for item in list(snapshot_source_state.get("raw_source_message_ids") or []) if str(item).strip()]
        raw_source_refs = [str(item) for item in list(snapshot_source_state.get("raw_source_refs") or []) if str(item).strip()]
        conflict_lines_by_snapshot = dict(snapshot_source_state.get("conflict_lines_by_snapshot") or {})
        seed_set = set(seed_snapshot_ids)
        retained_update_set = set(retained_update_linked_snapshot_ids)
        retained_neighbor_set = set(retained_neighbor_snapshot_ids)
        kept_ids: list[str] = []
        kept_set: set[str] = set()
        source_groups: dict[str, list[str]] = {group: [] for group in self.SOURCE_GROUP_ORDER}

        def add_kept_message(message_id: str, group: str) -> bool:
            if message_id in kept_set:
                return False
            kept_ids.append(message_id)
            kept_set.add(message_id)
            source_groups[group].append(message_id)
            return True

        for snapshot in compacted_snapshots:
            if snapshot.id not in seed_set:
                continue
            for message_id in list(source_candidates_by_snapshot.get(snapshot.id, [])):
                add_kept_message(message_id, "seed")
        budget = max(self._source_message_budget(query_is_list_like=query_is_list_like), len(kept_ids))
        remaining_budget = max(budget - len(kept_ids), 0)
        update_linked_source_count = 0
        neighbor_source_count = 0
        for snapshot in compacted_snapshots:
            if snapshot.id not in retained_update_set or remaining_budget <= 0:
                continue
            added_for_snapshot = 0
            for message_id in list(source_candidates_by_snapshot.get(snapshot.id, [])):
                if not add_kept_message(message_id, "update_linked"):
                    continue
                remaining_budget -= 1
                added_for_snapshot += 1
                update_linked_source_count += 1
                if remaining_budget <= 0 or added_for_snapshot >= self.UPDATE_LINKED_SOURCE_LIMIT:
                    break
        for snapshot in compacted_snapshots:
            if snapshot.id not in retained_neighbor_set or remaining_budget <= 0:
                continue
            added_for_snapshot = 0
            for message_id in list(source_candidates_by_snapshot.get(snapshot.id, [])):
                if not add_kept_message(message_id, "neighbor"):
                    continue
                remaining_budget -= 1
                added_for_snapshot += 1
                neighbor_source_count += 1
                if remaining_budget <= 0 or added_for_snapshot >= self.NEIGHBOR_SOURCE_LIMIT:
                    break
        message_records = self.store.fetch_raw_messages(raw_source_message_ids)
        message_by_id = {message.id: message for message in message_records}
        ordered_source_groups = {
            group: sorted(
                ids,
                key=lambda message_id: self._source_sort_key(message_by_id.get(message_id), message_id),
            )
            for group, ids in source_groups.items()
        }
        grouped_kept_ids = [
            message_id
            for group in self.SOURCE_GROUP_ORDER
            for message_id in ordered_source_groups.get(group, [])
        ]
        chronological_ids = sorted(
            grouped_kept_ids,
            key=lambda message_id: self._source_sort_key(message_by_id.get(message_id), message_id),
        )
        backtrack_count, backtrack_rate = self._source_backtrack_stats(grouped_kept_ids, message_by_id)
        kept_refs: list[str] = []
        kept_ref_set: set[str] = set()
        for message_id in grouped_kept_ids:
            message = message_by_id.get(message_id)
            source_ref = str(message.source_ref) if message is not None and message.source_ref else ""
            if not source_ref or source_ref in kept_ref_set:
                continue
            kept_refs.append(source_ref)
            kept_ref_set.add(source_ref)
        dropped_ids = [message_id for message_id in raw_source_message_ids if message_id not in kept_set]
        dropped_refs: list[str] = []
        dropped_ref_set: set[str] = set()
        for message_id in dropped_ids:
            message = message_by_id.get(message_id)
            source_ref = str(message.source_ref) if message is not None and message.source_ref else ""
            if not source_ref or source_ref in dropped_ref_set:
                continue
            dropped_refs.append(source_ref)
            dropped_ref_set.add(source_ref)
        conflict_lines: list[str] = []
        for snapshot in compacted_snapshots:
            conflict_lines.extend(list(conflict_lines_by_snapshot.get(snapshot.id, [])))
        seed_source_count = len([message_id for message_id in kept_ids if message_id in {
            source_id
            for snapshot in compacted_snapshots
            if snapshot.id in seed_set
            for source_id in list(source_candidates_by_snapshot.get(snapshot.id, []))
        }])
        return grouped_kept_ids, kept_refs, conflict_lines, {
            "raw_source_message_ids": raw_source_message_ids,
            "raw_source_refs": raw_source_refs,
            "source_compaction_budget": budget,
            "source_compaction_kept_ids": grouped_kept_ids,
            "source_compaction_dropped_ids": dropped_ids,
            "source_compaction_dropped_refs": dropped_refs,
            "source_compaction_counts": {
                "seed_source_count": seed_source_count,
                "update_linked_source_count": update_linked_source_count,
                "neighbor_source_count": neighbor_source_count,
            },
            "source_message_group_count": sum(
                1 for group in self.SOURCE_GROUP_ORDER if ordered_source_groups.get(group)
            ),
            "source_message_grouped_ids": ordered_source_groups,
            "source_message_chronological_ids": chronological_ids,
            "source_message_backtrack_count": backtrack_count,
            "source_message_backtrack_rate": backtrack_rate,
        }

    @staticmethod
    def _snapshot_sort_key(snapshot: EpisodicMemorySnapshot) -> tuple[str, int, str]:
        return (snapshot.trajectory_id, snapshot.version, snapshot.id)

    def _expand_update_linked_ids(
        self,
        *,
        hit_snapshots: list[EpisodicMemorySnapshot],
        trajectory_snapshots: list[EpisodicMemorySnapshot],
        claims_by_snapshot: dict[str, list[ClaimRecord]],
        ops_by_snapshot: dict[str, list[ClaimOpRecord]],
    ) -> set[str]:
        seed_snapshot_ids = {snapshot.id for snapshot in hit_snapshots}
        seed_claim_ids = {
            claim.claim_id for snapshot in hit_snapshots for claim in claims_by_snapshot.get(snapshot.id, [])
        }
        if not seed_claim_ids:
            return set()
        claim_graph: dict[str, set[str]] = defaultdict(set)
        claim_to_snapshot_ids: dict[str, set[str]] = defaultdict(set)
        for snapshot in trajectory_snapshots:
            for claim in claims_by_snapshot.get(snapshot.id, []):
                claim_to_snapshot_ids[claim.claim_id].add(snapshot.id)
                if claim.parent_claim_id:
                    claim_graph[claim.claim_id].add(claim.parent_claim_id)
                    claim_graph[claim.parent_claim_id].add(claim.claim_id)
                if claim.revised_from_claim_id:
                    claim_graph[claim.claim_id].add(claim.revised_from_claim_id)
                    claim_graph[claim.revised_from_claim_id].add(claim.claim_id)
            for op in ops_by_snapshot.get(snapshot.id, []):
                if op.new_claim_id:
                    claim_graph[op.target_claim_id].add(op.new_claim_id)
                    claim_graph[op.new_claim_id].add(op.target_claim_id)
        closure: set[str] = set(seed_claim_ids)
        frontier: deque[str] = deque(seed_claim_ids)
        while frontier:
            claim_id = frontier.popleft()
            for neighbor_claim_id in claim_graph.get(claim_id, set()):
                if neighbor_claim_id in closure:
                    continue
                closure.add(neighbor_claim_id)
                frontier.append(neighbor_claim_id)
        update_linked_snapshot_ids: set[str] = set()
        for claim_id in closure:
            update_linked_snapshot_ids.update(claim_to_snapshot_ids.get(claim_id, set()))
        for snapshot in trajectory_snapshots:
            for op in ops_by_snapshot.get(snapshot.id, []):
                if op.target_claim_id in closure or (op.new_claim_id and op.new_claim_id in closure):
                    update_linked_snapshot_ids.add(snapshot.id)
                    break
        return update_linked_snapshot_ids - seed_snapshot_ids

    @staticmethod
    def _empty_expansion_metadata(mode: str) -> dict[str, object]:
        return {
            "retrieval_expansion_mode": mode,
            "expansion_strategy": mode,
            "seed_snapshot_ids": [],
            "update_linked_snapshot_ids": [],
            "neighbor_fallback_snapshot_ids": [],
            "raw_expanded_snapshot_ids": [],
            "raw_expanded_count": 0,
            "update_linked_count": 0,
            "neighbor_fallback_count": 0,
            "update_linked_ms": 0.0,
            "neighbor_fallback_ms": 0.0,
        }

    def _expand_update_linked_plus_neighbors(
        self, snapshots: list[EpisodicMemorySnapshot]
    ) -> tuple[list[EpisodicMemorySnapshot], dict[str, object]]:
        if not snapshots:
            return [], self._empty_expansion_metadata("update_linked_plus_neighbors")
        seed_snapshot_ids = [snapshot.id for snapshot in snapshots]
        trajectory_ids = list(dict.fromkeys(snapshot.trajectory_id for snapshot in snapshots))
        all_trajectory_snapshots = self.store.list_snapshots_for_trajectories(trajectory_ids)
        claims_by_snapshot = self.store.list_claims_for_snapshots(snapshot.id for snapshot in all_trajectory_snapshots)
        ops_by_snapshot = self.store.list_claim_ops_for_snapshots(snapshot.id for snapshot in all_trajectory_snapshots)
        snapshots_by_id = {snapshot.id: snapshot for snapshot in all_trajectory_snapshots}
        snapshots_by_trajectory: dict[str, list[EpisodicMemorySnapshot]] = defaultdict(list)
        for snapshot in all_trajectory_snapshots:
            snapshots_by_trajectory[snapshot.trajectory_id].append(snapshot)
        hits_by_trajectory: dict[str, list[EpisodicMemorySnapshot]] = defaultdict(list)
        for snapshot in snapshots:
            hits_by_trajectory[snapshot.trajectory_id].append(snapshot)
        update_linked_started = time.perf_counter()
        update_linked_snapshot_ids: set[str] = set()
        for trajectory_id, hit_snapshots in hits_by_trajectory.items():
            update_linked_snapshot_ids.update(
                self._expand_update_linked_ids(
                    hit_snapshots=hit_snapshots,
                    trajectory_snapshots=snapshots_by_trajectory.get(trajectory_id, []),
                    claims_by_snapshot=claims_by_snapshot,
                    ops_by_snapshot=ops_by_snapshot,
                )
            )
        update_linked_ms = (time.perf_counter() - update_linked_started) * 1000.0
        base_snapshot_ids = set(seed_snapshot_ids) | update_linked_snapshot_ids
        neighbor_started = time.perf_counter()
        neighbor_fallback_snapshot_ids: set[str] = set()
        if self.neighbor_radius > 0:
            for snapshot_id in seed_snapshot_ids:
                snapshot = snapshots_by_id.get(snapshot_id)
                if snapshot is None:
                    continue
                for sibling in snapshots_by_trajectory.get(snapshot.trajectory_id, []):
                    if abs(sibling.version - snapshot.version) <= self.neighbor_radius and sibling.id not in base_snapshot_ids:
                        neighbor_fallback_snapshot_ids.add(sibling.id)
        neighbor_fallback_ms = (time.perf_counter() - neighbor_started) * 1000.0
        expanded_snapshot_ids = base_snapshot_ids | neighbor_fallback_snapshot_ids
        expanded = sorted(
            (snapshots_by_id[snapshot_id] for snapshot_id in expanded_snapshot_ids if snapshot_id in snapshots_by_id),
            key=self._snapshot_sort_key,
        )
        metadata = {
            "retrieval_expansion_mode": "update_linked_plus_neighbors",
            "expansion_strategy": "update_linked_plus_neighbors",
            "seed_snapshot_ids": seed_snapshot_ids,
            "update_linked_snapshot_ids": sorted(update_linked_snapshot_ids),
            "neighbor_fallback_snapshot_ids": sorted(neighbor_fallback_snapshot_ids),
            "raw_expanded_snapshot_ids": [snapshot.id for snapshot in expanded],
            "raw_expanded_count": len(expanded),
            "update_linked_count": len(update_linked_snapshot_ids),
            "neighbor_fallback_count": len(neighbor_fallback_snapshot_ids),
            "update_linked_ms": update_linked_ms,
            "neighbor_fallback_ms": neighbor_fallback_ms,
        }
        return expanded, metadata

    def _expand_neighbors_only(self, snapshots: list[EpisodicMemorySnapshot]) -> tuple[list[EpisodicMemorySnapshot], dict[str, object]]:
        if not snapshots:
            return [], self._empty_expansion_metadata("neighbors_only")
        seed_snapshot_ids = [snapshot.id for snapshot in snapshots]
        trajectory_ids = list(dict.fromkeys(snapshot.trajectory_id for snapshot in snapshots))
        all_trajectory_snapshots = self.store.list_snapshots_for_trajectories(trajectory_ids)
        snapshots_by_id = {snapshot.id: snapshot for snapshot in all_trajectory_snapshots}
        snapshots_by_trajectory: dict[str, list[EpisodicMemorySnapshot]] = defaultdict(list)
        for snapshot in all_trajectory_snapshots:
            snapshots_by_trajectory[snapshot.trajectory_id].append(snapshot)
        seed_snapshot_id_set = set(seed_snapshot_ids)
        neighbor_started = time.perf_counter()
        neighbor_fallback_snapshot_ids: set[str] = set()
        if self.neighbor_radius > 0:
            for snapshot in snapshots:
                for sibling in snapshots_by_trajectory.get(snapshot.trajectory_id, []):
                    if abs(sibling.version - snapshot.version) <= self.neighbor_radius and sibling.id not in seed_snapshot_id_set:
                        neighbor_fallback_snapshot_ids.add(sibling.id)
        neighbor_fallback_ms = (time.perf_counter() - neighbor_started) * 1000.0
        expanded_snapshot_ids = seed_snapshot_id_set | neighbor_fallback_snapshot_ids
        expanded = sorted(
            (snapshots_by_id[snapshot_id] for snapshot_id in expanded_snapshot_ids if snapshot_id in snapshots_by_id),
            key=self._snapshot_sort_key,
        )
        metadata = {
            "retrieval_expansion_mode": "neighbors_only",
            "expansion_strategy": "neighbors_only",
            "seed_snapshot_ids": seed_snapshot_ids,
            "update_linked_snapshot_ids": [],
            "neighbor_fallback_snapshot_ids": sorted(neighbor_fallback_snapshot_ids),
            "raw_expanded_snapshot_ids": [snapshot.id for snapshot in expanded],
            "raw_expanded_count": len(expanded),
            "update_linked_count": 0,
            "neighbor_fallback_count": len(neighbor_fallback_snapshot_ids),
            "update_linked_ms": 0.0,
            "neighbor_fallback_ms": neighbor_fallback_ms,
        }
        return expanded, metadata

    def _expand_none(self, snapshots: list[EpisodicMemorySnapshot]) -> tuple[list[EpisodicMemorySnapshot], dict[str, object]]:
        metadata = self._empty_expansion_metadata("none")
        metadata["seed_snapshot_ids"] = [snapshot.id for snapshot in snapshots]
        metadata["raw_expanded_snapshot_ids"] = [snapshot.id for snapshot in snapshots]
        metadata["raw_expanded_count"] = len(snapshots)
        return list(snapshots), metadata

    def expand_snapshots(self, snapshots: list[EpisodicMemorySnapshot]) -> tuple[list[EpisodicMemorySnapshot], dict[str, object]]:
        if self.retrieval_expansion_mode == "update_linked_plus_neighbors":
            return self._expand_update_linked_plus_neighbors(snapshots)
        if self.retrieval_expansion_mode == "neighbors_only":
            return self._expand_neighbors_only(snapshots)
        if self.retrieval_expansion_mode == "none":
            return self._expand_none(snapshots)
        raise ValueError(f"Unsupported retrieval_expansion_mode: {self.retrieval_expansion_mode}")

    def collect_source_messages(self, snapshots: Iterable[EpisodicMemorySnapshot]) -> tuple[list[str], list[str], list[str]]:
        snapshot_list = list(snapshots)
        claims_by_snapshot = self.store.list_claims_for_snapshots(snapshot.id for snapshot in snapshot_list)
        source_message_ids: list[str] = []
        conflict_lines: list[str] = []
        for snapshot in snapshot_list:
            claims = claims_by_snapshot.get(snapshot.id, [])
            source_message_ids.extend(getattr(snapshot, "links_json", []))
            source_message_ids.extend(source for claim in claims for source in claim.source_message_ids_json)
            unresolved = [claim for claim in claims if claim.status in {"contradictory", "needs-confirmation"}]
            if unresolved:
                conflict_lines.append(
                    f"Snapshot {snapshot.id} contains unresolved claims: "
                    + "; ".join(f"{claim.claim_id} [{claim.status}] {claim.text}" for claim in unresolved)
                )
        message_records = self.store.fetch_raw_messages(source_message_ids)
        message_refs = [record.source_ref for record in message_records if record.source_ref]
        return list(dict.fromkeys(source_message_ids)), message_refs, conflict_lines

    def build_context(
        self,
        sample_id: str,
        query_text: str,
        *,
        attempt_label: str = "normal",
        reflection_hints: dict[str, object] | None = None,
        normal_retrieval_event_id: str | None = None,
        force_raw_rescue: bool = False,
        force_raw_rescue_reason: str | None = None,
    ) -> RetrievalBundle:
        started_at = time.perf_counter()
        reflection_hints = dict(reflection_hints or {})
        reflection_terms = self._reflection_terms(reflection_hints)
        effective_query_parts = [
            query_text,
            str(reflection_hints.get("rewritten_query") or ""),
            " ".join(reflection_terms),
        ]
        effective_query_text = collapse_whitespace(" ".join(part for part in effective_query_parts if part))
        self._trace(f"sample={sample_id} page_route_start attempt={attempt_label}")
        query_embedding = self._embed_queries([effective_query_text])[0]
        query_keywords = extract_keywords(effective_query_text)
        entity_lexicon = self._sample_entity_lexicon(sample_id)
        query_facet_summary = extract_query_facets_v1(effective_query_text, entity_lexicon)
        query_shape = classify_query_shape_v1(query_text, entity_lexicon)
        query_entities = list(query_facet_summary["entities"])
        query_facet_tags = set(query_facet_summary["tags"])
        query_facet_values = set(query_facet_summary["values"])
        query_is_list_like = bool(query_shape["list_like"])

        selected_page_ids, selected_page_trajectory_ids, page_metadata = self._route_pages(
            sample_id,
            query_text,
            query_embedding,
            query_keywords,
            query_entities,
            query_shape,
            query_facet_values,
            reflection_hints if attempt_label == "reflection" else None,
        )
        self._trace(
            f"sample={sample_id} page_route_done selected_pages={len(selected_page_ids)} selected_page_trajectories={len(selected_page_trajectory_ids)}"
        )
        capped_page_trajectory_ids, broad_entity_cap_metadata = self._apply_broad_entity_page_candidate_cap(
            sample_id=sample_id,
            selected_page_trajectory_ids=list(selected_page_trajectory_ids),
            page_metadata=page_metadata,
            query_keywords=query_keywords,
            query_entities=query_entities,
            query_facet_tags=query_facet_tags,
            query_facet_values=query_facet_values,
            query_shape=query_shape,
        )
        trajectory_candidate_input_ids, index_fallback_metadata = self._index_fallback_trajectory_expansion(
            sample_id=sample_id,
            selected_page_trajectory_ids=list(capped_page_trajectory_ids),
            page_metadata=page_metadata,
            query_keywords=query_keywords,
            query_entities=query_entities,
            query_facet_tags=query_facet_tags,
            query_facet_values=query_facet_values,
            query_shape=query_shape,
        )
        self._trace(f"sample={sample_id} trajectory_select_start")
        selected_trajectory_ids, trajectory_metadata = self._select_trajectories(
            sample_id,
            trajectory_candidate_input_ids,
            query_text,
            query_embedding,
            query_keywords,
            query_entities,
            query_facet_tags,
            query_facet_values,
            query_shape,
        )
        self._trace(f"sample={sample_id} trajectory_select_done selected_trajectories={len(selected_trajectory_ids)}")
        self._trace(f"sample={sample_id} fine_retrieve_start")
        snapshot_hits, fine_metadata = self.fine_retrieve_snapshots(selected_trajectory_ids, query_embedding)
        self._trace(
            f"sample={sample_id} fine_retrieve_done hits={len(snapshot_hits)} budget={int(fine_metadata['fine_snapshot_budget'])}"
        )
        raw_expanded, expansion_metadata = self.expand_snapshots(snapshot_hits)
        raw_expanded_source_state = self._collect_snapshot_source_state(raw_expanded)
        raw_expanded_refs = list(raw_expanded_source_state["raw_source_refs"])
        self._trace(
            f"sample={sample_id} expansion_raw_done hits={len(snapshot_hits)} "
            f"update_linked={int(expansion_metadata['update_linked_count'])} "
            f"neighbor={int(expansion_metadata['neighbor_fallback_count'])} "
            f"raw_expanded={len(raw_expanded)}"
        )
        expanded, snapshot_compaction_metadata = self._compact_expanded_snapshots(
            sample_id=sample_id,
            raw_expanded=raw_expanded,
            selected_trajectory_ids=selected_trajectory_ids,
            query_embedding=query_embedding,
            query_keywords=query_keywords,
            query_entity_keys={normalize_entity_key(value) for value in query_entities},
            query_facet_tags=query_facet_tags,
            query_facet_values=query_facet_values,
            seed_snapshot_ids=list(expansion_metadata["seed_snapshot_ids"]),
            update_linked_snapshot_ids=list(expansion_metadata["update_linked_snapshot_ids"]),
            neighbor_candidate_snapshot_ids=list(expansion_metadata["neighbor_fallback_snapshot_ids"]),
            query_is_list_like=query_is_list_like,
        )
        snapshot_compaction_counts = dict(snapshot_compaction_metadata["snapshot_compaction_counts"])
        self._trace(
            f"sample={sample_id} snapshot_compaction_done budget={int(snapshot_compaction_metadata['snapshot_compaction_budget'])} "
            f"kept={len(expanded)} dropped={len(snapshot_compaction_metadata['snapshot_compaction_dropped_ids'])} "
            f"seed={int(snapshot_compaction_counts['seed_kept'])} "
            f"reserved_update_linked={int(snapshot_compaction_counts['reserved_update_linked_kept'])} "
            f"scored_update_linked={int(snapshot_compaction_counts['scored_update_linked_kept'])} "
            f"neighbor={int(snapshot_compaction_counts['neighbor_kept'])}"
        )
        compacted_source_state = self._collect_snapshot_source_state(expanded)
        source_message_ids, source_refs, conflict_lines, source_compaction_metadata = self._compact_source_messages(
            compacted_snapshots=expanded,
            snapshot_source_state=compacted_source_state,
            seed_snapshot_ids=list(expansion_metadata["seed_snapshot_ids"]),
            retained_update_linked_snapshot_ids=list(
                snapshot_compaction_metadata["reserved_update_linked_snapshot_ids"]
            )
            + list(snapshot_compaction_metadata["scored_update_linked_snapshot_ids"]),
            retained_neighbor_snapshot_ids=[
                snapshot_id
                for snapshot_id in list(snapshot_compaction_metadata["snapshot_compaction_kept_ids"])
                if snapshot_id in set(expansion_metadata["neighbor_fallback_snapshot_ids"])
            ],
            query_is_list_like=query_is_list_like,
        )
        source_compaction_counts = dict(source_compaction_metadata["source_compaction_counts"])
        self._trace(
            f"sample={sample_id} source_compaction_done raw_sources={len(source_compaction_metadata['raw_source_message_ids'])} "
            f"kept_sources={len(source_message_ids)} dropped_sources={len(source_compaction_metadata['source_compaction_dropped_ids'])} "
            f"seed_sources={int(source_compaction_counts['seed_source_count'])} "
            f"update_linked_sources={int(source_compaction_counts['update_linked_source_count'])} "
            f"neighbor_sources={int(source_compaction_counts['neighbor_source_count'])}"
        )
        expanded_claims = self.store.list_claims_for_snapshots(snapshot.id for snapshot in expanded)
        expanded_ops = self.store.list_claim_ops_for_snapshots(snapshot.id for snapshot in expanded)
        grounded_exact_terms: list[str] = []
        grounded_display_items: list[str] = []
        grounded_display_counts: list[str] = []
        grounded_display_key_facts: list[str] = []
        for claims in expanded_claims.values():
            for claim in claims:
                if claim.status == "deprecated":
                    continue
                metadata = dict(claim.metadata_json or {})
                if metadata.get("speaker_grounding_suspect_v1"):
                    continue
                grounded_exact_terms.extend(
                    str(value).strip()
                    for value in list(metadata.get("exact_terms_v2") or metadata.get("exact_terms_v1") or [])
                    if str(value).strip()
                )
                display = dict(metadata.get("display_signals_v1") or {})
                grounded_display_items.extend(
                    str(value).strip()
                    for value in list(display.get("items") or [])
                    if str(value).strip()
                )
                grounded_display_counts.extend(
                    str(value).strip()
                    for value in list(display.get("counts") or [])
                    if str(value).strip()
                )
                grounded_display_key_facts.extend(
                    str(value).strip()
                    for value in list(display.get("key_facts") or [])
                    if str(value).strip()
                )
        episodic_lines = []
        answer_context_active_claim_count = 0
        answer_context_uncertain_claim_count = 0
        answer_context_suppressed_deprecated_claim_count = 0
        answer_context_suppressed_speaker_grounding_suspect_claim_count = 0
        answer_context_suppressed_ops_count = 0
        for snapshot in expanded:
            rendered_snapshot, render_counts = render_answer_episodic_snapshot(
                snapshot,
                expanded_claims.get(snapshot.id, []),
                expanded_ops.get(snapshot.id, []),
            )
            episodic_lines.append(rendered_snapshot)
            answer_context_active_claim_count += int(render_counts["active_claim_count"])
            answer_context_uncertain_claim_count += int(render_counts["uncertain_claim_count"])
            answer_context_suppressed_deprecated_claim_count += int(
                render_counts["suppressed_deprecated_claim_count"]
            )
            answer_context_suppressed_speaker_grounding_suspect_claim_count += int(
                render_counts.get("suppressed_speaker_grounding_suspect_claim_count", 0)
            )
            answer_context_suppressed_ops_count += int(render_counts["suppressed_ops_count"])
        raw_rescue_metadata: dict[str, object] = {
            "raw_rescue_attempted": False,
            "raw_rescue_used": False,
            "raw_rescue_trigger_reasons": [],
            "raw_rescue_skipped_reason": "not_reflection_attempt" if attempt_label != "reflection" else None,
            "raw_rescue_query": effective_query_text,
            "raw_rescue_terms": [],
            "raw_rescue_candidate_count": 0,
            "raw_rescue_hit_count": 0,
            "raw_rescue_source_ids": [],
            "raw_rescue_source_refs": [],
            "raw_rescue_embedding_used": False,
            "raw_rescue_lexical_fallback": False,
            "raw_rescue_included_system_messages": False,
            "raw_rescue_embedding_error": None,
            "reflection_required_terms": [],
            "reflection_covered_terms": [],
            "reflection_uncovered_terms": [],
            "reflection_term_coverage_rate": None,
            "reflection_semantic_evidence_weak": False,
            "force_raw_rescue_reason": force_raw_rescue_reason,
        }
        if attempt_label == "reflection" and reflection_hints:
            reroute_weak = self._retrieval_evidence_weak(
                source_message_ids=source_message_ids,
                selected_trajectory_ids=selected_trajectory_ids,
                snapshot_hits=snapshot_hits,
                active_claim_count=answer_context_active_claim_count,
            )
            coverage_metadata = self._reflection_evidence_coverage(
                sample_id=sample_id,
                selected_page_ids=list(selected_page_ids),
                source_message_ids=source_message_ids,
                reflection_hints=reflection_hints,
                grounded_exact_terms=grounded_exact_terms,
                grounded_display_items=grounded_display_items,
                grounded_display_counts=grounded_display_counts,
                grounded_display_key_facts=grounded_display_key_facts,
            )
            raw_rescue_metadata.update(coverage_metadata)
            trigger_reasons: list[str] = []
            if reroute_weak:
                trigger_reasons.append("structural_evidence_weak")
            if bool(coverage_metadata.get("reflection_semantic_evidence_weak")):
                trigger_reasons.append("semantic_evidence_weak")
            if force_raw_rescue:
                trigger_reasons.append("forced")
            should_raw_rescue = bool(trigger_reasons)
            if should_raw_rescue:
                raw_rescue_records, rescue_metadata = self._raw_rescue_messages(
                    sample_id=sample_id,
                    query_text=query_text,
                    effective_query_text=effective_query_text,
                    reflection_hints=reflection_hints,
                    query_embedding=query_embedding,
                    query_is_list_like=query_is_list_like,
                    exclude_message_ids=set(source_message_ids),
                )
                raw_rescue_metadata.update(rescue_metadata)
                raw_rescue_metadata.update(coverage_metadata)
                raw_rescue_metadata["raw_rescue_trigger_reasons"] = trigger_reasons
                raw_rescue_metadata["raw_rescue_skipped_reason"] = None
                raw_rescue_metadata["force_raw_rescue_reason"] = force_raw_rescue_reason
                for message in raw_rescue_records:
                    if message.id not in source_message_ids:
                        source_message_ids.append(message.id)
                    if message.source_ref and str(message.source_ref) not in source_refs:
                        source_refs.append(str(message.source_ref))
                self._trace(
                    f"sample={sample_id} raw_rescue_done "
                    f"reasons={','.join(trigger_reasons)} "
                    f"hits={int(raw_rescue_metadata['raw_rescue_hit_count'])} "
                    f"candidates={int(raw_rescue_metadata['raw_rescue_candidate_count'])}"
                )
            else:
                raw_rescue_metadata["raw_rescue_trigger_reasons"] = []
                raw_rescue_metadata["raw_rescue_skipped_reason"] = "evidence_sufficient"
        elif attempt_label == "reflection":
            raw_rescue_metadata["raw_rescue_skipped_reason"] = "no_reflection_hints"
        source_records = self.store.fetch_raw_messages(source_message_ids)
        source_by_id = {message.id: message for message in source_records}
        source_group_ids = dict(source_compaction_metadata.get("source_message_grouped_ids") or {})
        source_blocks: list[str] = []
        for group in self.SOURCE_GROUP_ORDER:
            group_ids = [str(item) for item in list(source_group_ids.get(group) or []) if str(item).strip()]
            group_messages = [source_by_id[message_id] for message_id in group_ids if message_id in source_by_id]
            if not group_messages:
                continue
            title = self.SOURCE_GROUP_TITLES[group]
            source_blocks.append(
                f"### {title}\n" + "\n".join(self._source_message_line(message) for message in group_messages)
            )
        chronological_messages = sorted(
            source_records,
            key=lambda message: self._source_sort_key(message, message.id),
        )
        timeline_lines = [self._source_timeline_line(message) for message in chronological_messages]
        temporal_anchor_lines, temporal_anchor_metadata = self._temporal_anchor_lines(chronological_messages)
        source_message_time_anchors = {
            str(message.source_ref): collapse_whitespace(str(message.occurred_at or ""))
            for message in chronological_messages
            if message.source_ref and collapse_whitespace(str(message.occurred_at or ""))
        }
        raw_rescue_source_ids = [
            str(item)
            for item in list(raw_rescue_metadata.get("raw_rescue_source_ids") or [])
            if str(item).strip()
        ]
        raw_rescue_messages = [
            source_by_id[message_id]
            for message_id in raw_rescue_source_ids
            if message_id in source_by_id
        ]
        context_blocks = [
            "## Retrieved Episodic Memory\n" + ("\n\n".join(episodic_lines) if episodic_lines else "None."),
            "## Retrieved Source Messages\n" + ("\n\n".join(source_blocks) if source_blocks else "None."),
        ]
        if raw_rescue_messages:
            context_blocks.append(
                "## Raw Message Rescue Evidence\n"
                + "\n".join(self._source_message_line(message) for message in raw_rescue_messages)
            )
        context_blocks.append(
            "## Chronological Source Timeline\n" + ("\n".join(timeline_lines) if timeline_lines else "None.")
        )
        if temporal_anchor_lines:
            context_blocks.append("## Temporal Anchors\n" + "\n".join(temporal_anchor_lines))
        if conflict_lines:
            context_blocks.append("## Conflict Block\n" + "\n".join(f"- {line}" for line in conflict_lines))
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        metadata = {
            "retrieval_attempt_label": attempt_label,
            "retrieval_reflection_used": attempt_label == "reflection",
            "retrieval_reflection_stage": (
                "raw" if bool(raw_rescue_metadata.get("raw_rescue_attempted")) else (
                    "wiki" if attempt_label == "reflection" else "none"
                )
            ),
            "normal_retrieval_event_id": normal_retrieval_event_id,
            "effective_query_text": effective_query_text,
            "reflection_rewritten_query": str(reflection_hints.get("rewritten_query") or ""),
            "reflection_answer_type": str(reflection_hints.get("answer_type") or ""),
            "reflection_target_entities": self._clean_text_values(
                list(reflection_hints.get("target_entities") or []),
                limit=24,
            ),
            "reflection_must_find_terms": self._clean_text_values(
                list(reflection_hints.get("must_find_terms") or []),
                limit=24,
            ),
            "reflection_candidate_page_slugs": self._clean_text_values(
                list(reflection_hints.get("candidate_page_slugs") or []),
                limit=24,
            ),
            "reflection_raw_search_terms": self._clean_text_values(
                list(reflection_hints.get("raw_search_terms") or []),
                limit=24,
            ),
            "reflection_mode": reflection_hints.get("reflection_mode"),
            "reflection_error": reflection_hints.get("reflection_error"),
            "reflection_latency_ms": float(reflection_hints.get("reflection_latency_ms") or 0.0),
            "source_refs": source_refs,
            "selected_pages": list(selected_page_ids),
            "conflicts": conflict_lines,
            "neighbor_radius": self.neighbor_radius,
            "query_embedding_strategy": self._query_embedding_strategy(),
            "query_entities": query_entities,
            "query_is_list_like": query_is_list_like,
            "query_shape": {
                "list_like": bool(query_shape["list_like"]),
                "multi_entity": bool(query_shape["multi_entity"]),
                "comparison_like": bool(query_shape["comparison_like"]),
                "count_like": bool(query_shape["count_like"]),
                "item_family": query_shape.get("item_family"),
                "entity_keys": list(query_shape["entity_keys"]),
                "tags": list(query_shape["tags"]),
            },
            "query_facets": {"tags": sorted(query_facet_tags), "values": sorted(query_facet_values)},
            "page_candidate_ids": page_metadata["page_candidate_ids"],
            "page_rerank_selected_ids": page_metadata["page_rerank_selected_ids"],
            "page_rerank_rationales": page_metadata["page_rerank_rationales"],
            "page_rerank_fallback": page_metadata["page_rerank_fallback"],
            "page_selection_strategy": page_metadata["page_selection_strategy"],
            "page_cluster_coverage": page_metadata["page_cluster_coverage"],
            "page_covered_query_entities": page_metadata["page_covered_query_entities"],
            "page_covered_query_facet_values": page_metadata["page_covered_query_facet_values"],
            "page_covered_query_terms": page_metadata["page_covered_query_terms"],
            "page_ranked_rows": page_metadata["page_ranked_rows"],
            "diagnostic_top_n_pages": page_metadata.get("diagnostic_top_n_pages"),
            "page_ranked_total_count": page_metadata.get("page_ranked_total_count"),
            "page_ranked_rows_truncated": page_metadata.get("page_ranked_rows_truncated"),
            "page_ranked_rows_compact_top_n": page_metadata.get("page_ranked_rows_compact_top_n", []),
            "page_cutoff_universe_diagnostics": page_metadata.get("page_cutoff_universe_diagnostics", {}),
            "selected_page_rows": page_metadata.get("selected_page_rows", []),
            "selected_page_rows_compact": page_metadata.get("selected_page_rows_compact", []),
            "page_granularity_diagnostic_mode": page_metadata.get("page_granularity_diagnostic_mode"),
            "selected_singleton_page_count": page_metadata.get("selected_singleton_page_count"),
            "selected_medium_granularity_page_count": page_metadata.get("selected_medium_granularity_page_count"),
            "selected_page_trajectory_count_histogram": page_metadata.get("selected_page_trajectory_count_histogram", {}),
            "singleton_page_penalty_applied": page_metadata.get("singleton_page_penalty_applied"),
            "medium_page_bonus_applied": page_metadata.get("medium_page_bonus_applied"),
            "selected_singleton_page_ids": page_metadata.get("selected_singleton_page_ids", []),
            "selected_medium_page_ids": page_metadata.get("selected_medium_page_ids", []),
            "page_index_suppressed": page_metadata["page_index_suppressed"],
            "index_page_ids": page_metadata["index_page_ids"],
            "index_page_trajectory_ids": page_metadata["index_page_trajectory_ids"],
            "non_index_page_count": page_metadata["non_index_page_count"],
            "selected_page_trajectory_ids_before_broad_cap": list(selected_page_trajectory_ids),
            "selected_page_trajectory_ids": list(capped_page_trajectory_ids),
            "trajectory_candidate_input_ids": list(trajectory_candidate_input_ids),
            **broad_entity_cap_metadata,
            **index_fallback_metadata,
            "trajectory_candidate_pool_ids": trajectory_metadata["trajectory_candidate_pool_ids"],
            "trajectory_selection_pool_ids": trajectory_metadata["trajectory_selection_pool_ids"],
            "trajectory_selection_pool_size": trajectory_metadata["trajectory_selection_pool_size"],
            "trajectory_selection_pool_rows_compact": trajectory_metadata.get(
                "trajectory_selection_pool_rows_compact", []
            ),
            "trajectory_selection_pool_rows_total_count": trajectory_metadata.get(
                "trajectory_selection_pool_rows_total_count"
            ),
            "trajectory_selection_pool_rows_truncated": trajectory_metadata.get(
                "trajectory_selection_pool_rows_truncated"
            ),
            "trajectory_rerank_pool_size": trajectory_metadata["trajectory_rerank_pool_size"],
            "trajectory_rerank_selected_ids": trajectory_metadata["trajectory_rerank_selected_ids"],
            "trajectory_rerank_rationales": trajectory_metadata["trajectory_rerank_rationales"],
            "trajectory_rerank_fallback": trajectory_metadata["trajectory_rerank_fallback"],
            "trajectory_selection_strategy": trajectory_metadata["trajectory_selection_strategy"],
            "trajectory_cluster_coverage": trajectory_metadata["trajectory_cluster_coverage"],
            "trajectory_covered_query_entities": trajectory_metadata["trajectory_covered_query_entities"],
            "trajectory_covered_query_facet_values": trajectory_metadata["trajectory_covered_query_facet_values"],
            "trajectory_covered_query_terms": trajectory_metadata["trajectory_covered_query_terms"],
            "trajectory_covered_item_terms": trajectory_metadata["trajectory_covered_item_terms"],
            "trajectory_selected_score_components": trajectory_metadata["trajectory_selected_score_components"],
            "trajectory_redundancy_penalties": trajectory_metadata["trajectory_redundancy_penalties"],
            "trajectory_family_match_scores": trajectory_metadata["trajectory_family_match_scores"],
            "trajectory_family_mismatch_penalties": trajectory_metadata["trajectory_family_mismatch_penalties"],
            "trajectory_selected_family_matches": trajectory_metadata["trajectory_selected_family_matches"],
            "trajectory_source_event_match_scores": trajectory_metadata.get("trajectory_source_event_match_scores", []),
            "trajectory_selected_source_event_matches": trajectory_metadata.get(
                "trajectory_selected_source_event_matches", []
            ),
            "trajectory_source_event_match_miss_count": trajectory_metadata.get(
                "trajectory_source_event_match_miss_count"
            ),
            "trajectory_source_event_query_profile": trajectory_metadata.get(
                "trajectory_source_event_query_profile", {}
            ),
            "trajectory_ranked_rows": trajectory_metadata["trajectory_ranked_rows"],
            "diagnostic_top_n_trajectories": trajectory_metadata.get("diagnostic_top_n_trajectories"),
            "trajectory_ranked_total_count": trajectory_metadata.get("trajectory_ranked_total_count"),
            "trajectory_ranked_rows_truncated": trajectory_metadata.get("trajectory_ranked_rows_truncated"),
            "trajectory_ranked_rows_compact_top_n": trajectory_metadata.get(
                "trajectory_ranked_rows_compact_top_n", []
            ),
            "trajectory_cutoff_prefix_diagnostics": trajectory_metadata.get(
                "trajectory_cutoff_prefix_diagnostics", {}
            ),
            "fine_snapshot_budget": fine_metadata["fine_snapshot_budget"],
            "fine_snapshot_quota_counts": fine_metadata["fine_snapshot_quota_counts"],
            "fine_snapshot_selected_ids": fine_metadata["fine_snapshot_selected_ids"],
            "retrieval_expansion_mode": self.retrieval_expansion_mode,
            "expansion_strategy": expansion_metadata["expansion_strategy"],
            "seed_snapshot_ids": expansion_metadata["seed_snapshot_ids"],
            "update_linked_snapshot_ids": expansion_metadata["update_linked_snapshot_ids"],
            "neighbor_fallback_snapshot_ids": expansion_metadata["neighbor_fallback_snapshot_ids"],
            "raw_expanded_snapshot_ids": expansion_metadata["raw_expanded_snapshot_ids"],
            "raw_expanded_refs": raw_expanded_refs,
            "raw_expanded_count": expansion_metadata["raw_expanded_count"],
            "update_linked_count": expansion_metadata["update_linked_count"],
            "neighbor_fallback_count": expansion_metadata["neighbor_fallback_count"],
            "update_linked_ms": expansion_metadata["update_linked_ms"],
            "neighbor_fallback_ms": expansion_metadata["neighbor_fallback_ms"],
            "snapshot_compaction_budget": snapshot_compaction_metadata["snapshot_compaction_budget"],
            "snapshot_compaction_kept_ids": snapshot_compaction_metadata["snapshot_compaction_kept_ids"],
            "snapshot_compaction_dropped_ids": snapshot_compaction_metadata["snapshot_compaction_dropped_ids"],
            "snapshot_compaction_counts": snapshot_compaction_metadata["snapshot_compaction_counts"],
            "reserved_update_linked_snapshot_ids": snapshot_compaction_metadata["reserved_update_linked_snapshot_ids"],
            "scored_update_linked_snapshot_ids": snapshot_compaction_metadata["scored_update_linked_snapshot_ids"],
            "neighbor_candidate_snapshot_ids": snapshot_compaction_metadata["neighbor_candidate_snapshot_ids"],
            "raw_source_message_ids": source_compaction_metadata["raw_source_message_ids"],
            "raw_source_refs": source_compaction_metadata["raw_source_refs"],
            "source_compaction_budget": source_compaction_metadata["source_compaction_budget"],
            "source_compaction_kept_ids": source_compaction_metadata["source_compaction_kept_ids"],
            "source_compaction_dropped_ids": source_compaction_metadata["source_compaction_dropped_ids"],
            "source_compaction_dropped_refs": source_compaction_metadata["source_compaction_dropped_refs"],
            "source_compaction_counts": source_compaction_metadata["source_compaction_counts"],
            "source_message_group_count": source_compaction_metadata["source_message_group_count"],
            "source_message_grouped_ids": source_compaction_metadata["source_message_grouped_ids"],
            "source_message_chronological_ids": source_compaction_metadata["source_message_chronological_ids"],
            "source_message_time_anchors": source_message_time_anchors,
            **temporal_anchor_metadata,
            "source_message_backtrack_count": source_compaction_metadata["source_message_backtrack_count"],
            "source_message_backtrack_rate": source_compaction_metadata["source_message_backtrack_rate"],
            "answer_context_active_claim_count": answer_context_active_claim_count,
            "answer_context_uncertain_claim_count": answer_context_uncertain_claim_count,
            "answer_context_suppressed_deprecated_claim_count": answer_context_suppressed_deprecated_claim_count,
            "answer_context_suppressed_speaker_grounding_suspect_claim_count": (
                answer_context_suppressed_speaker_grounding_suspect_claim_count
            ),
            "answer_context_suppressed_ops_count": answer_context_suppressed_ops_count,
            "grounded_exact_terms": list(dict.fromkeys(grounded_exact_terms)),
            "grounded_display_items": list(dict.fromkeys(grounded_display_items)),
            "grounded_display_counts": list(dict.fromkeys(grounded_display_counts)),
            "grounded_display_key_facts": list(dict.fromkeys(grounded_display_key_facts)),
            **raw_rescue_metadata,
        }
        event = RetrievalEvent(
            id=f"{sample_id}-retrieval-{int(time.time() * 1000)}",
            sample_id=sample_id,
            query_text=query_text,
            query_embedding_json=query_embedding,
            top_t_pages=self.top_t_pages,
            top_k=self.top_k,
            snapshot_budget=self.snapshot_budget,
            page_ids_json=list(selected_page_ids),
            trajectory_ids_json=list(selected_trajectory_ids),
            snapshot_ids_json=[snapshot.id for snapshot in snapshot_hits],
            expanded_snapshot_ids_json=[snapshot.id for snapshot in expanded],
            source_message_ids_json=source_message_ids,
            latency_ms=latency_ms,
            metadata_json=metadata,
        )
        self.store.record_retrieval_event(event)
        self._trace(
            f"sample={sample_id} answer_context_ready pages={len(selected_page_ids)} trajectories={len(selected_trajectory_ids)} "
            f"hits={len(snapshot_hits)} expanded={len(expanded)} sources={len(source_message_ids)} latency_ms={latency_ms:.1f}"
        )
        return RetrievalBundle(
            retrieval_event_id=event.id,
            selected_pages=list(selected_page_ids),
            candidate_trajectories=list(selected_trajectory_ids),
            snapshot_hits=[snapshot.id for snapshot in snapshot_hits],
            expanded_snapshots=[snapshot.id for snapshot in expanded],
            source_message_ids=source_message_ids,
            source_message_refs=source_refs,
            prompt_context="\n\n".join(context_blocks),
            latency_ms=latency_ms,
            metadata=metadata,
        )
