"""Deterministic retrieval-plan reuse and context rendering for rebuttal ablations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from trajpatch.analysis.context_cost import (
    TOKEN_ESTIMATOR_NAME,
    estimate_context_tokens,
)
from trajpatch.analysis.direct_retrieval import rank_direct_trajectories
from trajpatch.analysis.gold_labels import build_memory_index, extract_source_refs
from trajpatch.analysis.memory_index import source_message_ids_for_refs
from trajpatch.analysis.offline_ablation import _load_retrieval_events
from trajpatch.experiments.token_budget import TokenCounter, build_token_counter
from trajpatch.memory.facets import (
    build_sample_entity_lexicon,
    classify_query_shape_v1,
    extract_query_facets_v1,
)
from trajpatch.storage.models import RawMessageRecord
from trajpatch.utils.text import collapse_whitespace, extract_keywords


@dataclass(slots=True)
class QueryFeatures:
    sample_id: str
    query_task_id: str
    question: str
    keywords: list[str]
    entities: list[str]
    facet_tags: list[str]
    facet_values: list[str]
    query_shape: dict[str, Any]


@dataclass(slots=True)
class RetrievalPlan:
    sample_id: str
    query_task_id: str
    retrieval_event_id: str
    selected_page_ids: list[str]
    selected_trajectory_ids: list[str]
    selected_snapshot_ids: list[str]
    selected_source_message_ids: list[str]
    candidate_universe_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextRenderPolicy:
    variant: str
    routing_policy: str
    evidence_policy: str
    source_support_constraint: bool
    include_wiki_pages: bool = False
    include_trajectory_summaries: bool = False
    include_raw_sources: bool = True
    allow_partial_items: bool = True


@dataclass(slots=True)
class VariantContext:
    variant: str
    sample_id: str
    query_task_id: str
    policy: dict[str, Any]
    selected_page_ids: list[str]
    selected_trajectory_ids: list[str]
    selected_snapshot_ids: list[str]
    selected_source_message_ids: list[str]
    selected_source_refs: list[str]
    context_text: str
    estimated_context_tokens: int
    estimated_prompt_tokens: int
    max_prompt_tokens: int
    output_token_reserve: int
    budget_truncated: bool
    token_estimator: str = TOKEN_ESTIMATOR_NAME
    token_counter: str = TOKEN_ESTIMATOR_NAME
    token_counter_exact: bool = False
    token_safety_margin: int = 0
    budget_mode: str = "fixed_total"
    global_max_total_tokens: int | None = None
    reference_variant: str | None = None
    reference_prompt_tokens: int | None = None
    prompt_token_utilization: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def compact_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["context_sha256"] = hashlib.sha256(
            self.context_text.encode("utf-8")
        ).hexdigest()
        if not include_context:
            payload.pop("context_text", None)
        return payload


VARIANT_POLICIES: dict[str, ContextRenderPolicy] = {
    "full": ContextRenderPolicy(
        variant="full",
        routing_policy="wiki_routed",
        evidence_policy="full_expansion",
        source_support_constraint=True,
    ),
    "direct_trajectory": ContextRenderPolicy(
        variant="direct_trajectory",
        routing_policy="all_trajectories",
        evidence_policy="full_expansion",
        source_support_constraint=True,
        include_wiki_pages=False,
    ),
    "latest_snapshot": ContextRenderPolicy(
        variant="latest_snapshot",
        routing_policy="wiki_routed",
        evidence_policy="latest_per_trajectory",
        source_support_constraint=True,
    ),
    "hybrid_raw_rag": ContextRenderPolicy(
        variant="hybrid_raw_rag",
        routing_policy="hybrid_raw_messages",
        evidence_policy="raw_messages",
        source_support_constraint=True,
        include_wiki_pages=False,
        include_trajectory_summaries=False,
    ),
    "hybrid_raw_rag_matched": ContextRenderPolicy(
        variant="hybrid_raw_rag_matched",
        routing_policy="hybrid_raw_messages",
        evidence_policy="raw_messages",
        source_support_constraint=True,
        include_wiki_pages=False,
        include_trajectory_summaries=False,
    ),
    "wiki_summaries": ContextRenderPolicy(
        variant="wiki_summaries",
        routing_policy="wiki_routed",
        evidence_policy="wiki_summary_context",
        source_support_constraint=True,
        include_wiki_pages=True,
        include_trajectory_summaries=False,
        include_raw_sources=False,
    ),
    "no_claim_state": ContextRenderPolicy(
        variant="no_claim_state",
        routing_policy="wiki_routed",
        evidence_policy="flat_claim_state",
        source_support_constraint=True,
    ),
    "no_source_constraint": ContextRenderPolicy(
        variant="no_source_constraint",
        routing_policy="wiki_routed",
        evidence_policy="full_expansion",
        source_support_constraint=False,
    ),
    "full_context": ContextRenderPolicy(
        variant="full_context",
        routing_policy="all_raw_messages",
        evidence_policy="raw_messages",
        source_support_constraint=True,
        include_wiki_pages=False,
        include_trajectory_summaries=False,
        allow_partial_items=False,
    ),
    "full_context_matched": ContextRenderPolicy(
        variant="full_context_matched",
        routing_policy="all_raw_messages",
        evidence_policy="raw_messages",
        source_support_constraint=True,
        include_wiki_pages=False,
        include_trajectory_summaries=False,
        allow_partial_items=False,
    ),
    "naive_dense_rag": ContextRenderPolicy(
        variant="naive_dense_rag",
        routing_policy="naive_dense_chunks",
        evidence_policy="raw_chunks",
        source_support_constraint=True,
        include_wiki_pages=False,
        include_trajectory_summaries=False,
        include_raw_sources=False,
        allow_partial_items=False,
    ),
}


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


class VariantContextBuilder:
    """Build all counterfactual contexts from one immutable benchmark run."""

    def __init__(
        self,
        *,
        database_path: Path,
        cache_dir: Path,
        embedding_provider: Any | None = None,
        top_k: int = 15,
        max_prompt_tokens: int = 32_000,
        output_token_reserve: int = 512,
        token_counter: TokenCounter | None = None,
        token_safety_margin: int = 0,
        rag_chunk_size: int = 384,
        rag_chunk_overlap: int = 64,
        rag_top_k: int = 4,
    ) -> None:
        self.database_path = Path(database_path)
        self.cache_dir = Path(cache_dir)
        self.embedding_provider = embedding_provider
        self.top_k = max(1, int(top_k))
        self.max_prompt_tokens = max(512, int(max_prompt_tokens))
        self.output_token_reserve = max(0, int(output_token_reserve))
        self.token_counter = token_counter or TokenCounter(
            name=TOKEN_ESTIMATOR_NAME,
            exact=False,
        )
        self.token_safety_margin = max(0, int(token_safety_margin))
        self.rag_chunk_size = max(1, int(rag_chunk_size))
        self.rag_chunk_overlap = max(0, int(rag_chunk_overlap))
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("rag_chunk_overlap must be smaller than rag_chunk_size.")
        self.rag_top_k = max(1, int(rag_top_k))
        self._rag_chunk_counter: TokenCounter | None = None
        self._hybrid_ranking_cache: dict[
            tuple[str, str],
            tuple[list[str], dict[str, Any]],
        ] = {}
        self.memory_index = build_memory_index(self.database_path)
        self.retrieval_events = _load_retrieval_events(
            self.database_path,
            self.memory_index,
        )

    def compute_query_features(self, sample_row: dict[str, Any]) -> QueryFeatures:
        sample_id = str(sample_row.get("sample_id") or "")
        question = str(sample_row.get("question") or "")
        raw_messages = [
            SimpleNamespace(**message)
            for message in list(
                self.memory_index["sample_raw_messages"].get(sample_id, [])
            )
        ]
        lexicon = build_sample_entity_lexicon(
            cast(Iterable[RawMessageRecord], raw_messages)
        )
        facets = extract_query_facets_v1(question, lexicon)
        return QueryFeatures(
            sample_id=sample_id,
            query_task_id=str(sample_row.get("query_task_id") or ""),
            question=question,
            keywords=sorted(extract_keywords(question)),
            entities=_dedupe(facets.get("entities") or []),
            facet_tags=_dedupe(facets.get("tags") or []),
            facet_values=_dedupe(facets.get("values") or []),
            query_shape=dict(classify_query_shape_v1(question, lexicon)),
        )

    def build_retrieval_plan(self, sample_row: dict[str, Any]) -> RetrievalPlan:
        sample_id = str(sample_row.get("sample_id") or "")
        query_task_id = str(sample_row.get("query_task_id") or "")
        event_id = str(sample_row.get("retrieval_event_id") or "")
        event = dict(self.retrieval_events.get(event_id) or {})
        selected_trajectory_ids = _dedupe(event.get("trajectory_ids") or [])
        candidate_ids = _dedupe(
            dict(event.get("metadata") or {}).get("trajectory_candidate_input_ids")
            or selected_trajectory_ids
        )
        return RetrievalPlan(
            sample_id=sample_id,
            query_task_id=query_task_id,
            retrieval_event_id=event_id,
            selected_page_ids=_dedupe(event.get("page_ids") or []),
            selected_trajectory_ids=selected_trajectory_ids,
            selected_snapshot_ids=_dedupe(event.get("expanded_snapshot_ids") or []),
            selected_source_message_ids=_dedupe(event.get("source_message_ids") or []),
            candidate_universe_size=len(candidate_ids),
            metadata={
                "source_run_retrieval_event_id": event_id,
                "source_run_top_k": int(event.get("top_k") or self.top_k),
                "source_run_top_t_pages": int(event.get("top_t_pages") or 0),
            },
        )

    def render_variant(
        self,
        *,
        sample_row: dict[str, Any],
        features: QueryFeatures,
        plan: RetrievalPlan,
        variant: str,
        reference_prompt_tokens: int | None = None,
    ) -> VariantContext:
        if variant not in VARIANT_POLICIES:
            raise ValueError(f"Unsupported answer ablation variant: {variant}")
        policy = VARIANT_POLICIES[variant]
        matched_budget = variant in {
            "full_context_matched",
            "hybrid_raw_rag_matched",
        }
        if matched_budget and reference_prompt_tokens is None:
            raise ValueError(
                f"{variant} requires reference_prompt_tokens from the full variant."
            )
        if reference_prompt_tokens is not None and reference_prompt_tokens < 0:
            raise ValueError("reference_prompt_tokens must be non-negative.")
        page_ids = list(plan.selected_page_ids)
        trajectory_ids = list(plan.selected_trajectory_ids)
        snapshot_ids = list(plan.selected_snapshot_ids)
        source_message_ids = list(plan.selected_source_message_ids)
        ranking_metadata: dict[str, Any] = {}
        context_items_override: list[dict[str, Any]] | None = None

        if policy.routing_policy == "all_trajectories":
            ranked, scoring_mode, score_rows = rank_direct_trajectories(
                sample_id=features.sample_id,
                question=features.question,
                query_entities=features.entities,
                query_facets={
                    "tags": features.facet_tags,
                    "values": features.facet_values,
                },
                query_shape=features.query_shape,
                sample_to_trajectories=self.memory_index["sample_to_trajectories"],
                trajectory_metadata=self.memory_index["trajectory_metadata"],
                claims_by_trajectory=self.memory_index["claims_by_trajectory"],
                trajectory_refs=self.memory_index["trajectory_refs"],
                trajectory_lengths=self.memory_index["trajectory_lengths"],
                diagnostic_top_n=max(
                    self.top_k,
                    len(
                        self.memory_index["sample_to_trajectories"].get(
                            features.sample_id, set()
                        )
                    ),
                ),
            )
            trajectory_ids = ranked[: self.top_k]
            page_ids = []
            snapshot_ids = _dedupe(
                snapshot_id
                for trajectory_id in trajectory_ids
                for snapshot_id in self.memory_index["trajectory_to_snapshots"].get(
                    trajectory_id, []
                )
            )
            source_message_ids = _dedupe(
                message_id
                for snapshot_id in snapshot_ids
                for message_id in self.memory_index["snapshot_source_message_ids"].get(
                    snapshot_id, []
                )
            )
            ranking_metadata = {
                "direct_scoring_mode": scoring_mode,
                "direct_score_rows": score_rows[: self.top_k],
                "direct_candidate_universe_size": len(ranked),
            }
        elif policy.routing_policy == "hybrid_raw_messages":
            ranked_messages, ranking_metadata = self._rank_hybrid_raw_messages(
                features.sample_id,
                features.question,
            )
            page_ids = []
            trajectory_ids = []
            snapshot_ids = []
            source_message_ids = ranked_messages
        elif policy.routing_policy == "all_raw_messages":
            page_ids = []
            trajectory_ids = []
            snapshot_ids = []
            source_message_ids = [
                str(message.get("id") or "")
                for message in self.memory_index["sample_raw_messages"].get(
                    features.sample_id,
                    [],
                )
                if str(message.get("id") or "")
            ]
            ranking_metadata = {
                "full_context_message_count": len(source_message_ids),
                "full_context_truncation_policy": (
                    "latest_whole_messages_chronological_render_v1"
                ),
            }
        elif policy.routing_policy == "naive_dense_chunks":
            page_ids = []
            trajectory_ids = []
            snapshot_ids = []
            context_items_override, ranking_metadata = self._rank_naive_dense_chunks(
                features.sample_id,
                features.question,
            )
            source_message_ids = _dedupe(
                message_id
                for item in context_items_override
                for message_id in list(item.get("source_message_ids") or [])
            )

        if policy.evidence_policy == "latest_per_trajectory":
            snapshot_ids = _dedupe(
                snapshots[-1]
                for trajectory_id in trajectory_ids
                if (
                    snapshots := list(
                        self.memory_index["trajectory_to_snapshots"].get(
                            trajectory_id, []
                        )
                    )
                )
            )
            source_message_ids = _dedupe(
                message_id
                for snapshot_id in snapshot_ids
                for message_id in self.memory_index["snapshot_source_message_ids"].get(
                    snapshot_id, []
                )
            )

        context_items = context_items_override or self._context_items(
            policy=policy,
            page_ids=page_ids,
            trajectory_ids=trajectory_ids,
            snapshot_ids=snapshot_ids,
            source_message_ids=source_message_ids,
        )
        priority = {
            "snapshot": 0,
            "raw_source": 1,
            "rag_chunk": 1,
            "trajectory_summary": 2,
            "wiki_page": 3,
        }
        context_items.sort(key=lambda item: priority.get(str(item.get("item_type")), 9))
        budget_source_support_constraint = (
            True
            if policy.variant == "no_source_constraint"
            else policy.source_support_constraint
        )
        prompt_overhead = self.token_counter.count(
            self.answer_prompt(
                question=features.question,
                context_text="",
                source_support_constraint=budget_source_support_constraint,
            )
        )
        effective_max_total_tokens = self.max_prompt_tokens
        if matched_budget:
            assert reference_prompt_tokens is not None
            effective_max_total_tokens = min(
                self.max_prompt_tokens,
                reference_prompt_tokens
                + self.output_token_reserve
                + self.token_safety_margin,
            )
        context_budget = max(
            0,
            effective_max_total_tokens
            - self.output_token_reserve
            - self.token_safety_margin
            - prompt_overhead,
        )
        selected_items, budget_truncated = self._budget_items_for_policy(
            context_items,
            context_budget,
            policy=policy,
            allow_partial_items=policy.allow_partial_items,
        )
        context_text = "\n\n".join(
            str(item["text"]) for item in selected_items if item["text"]
        )
        matched_boundary_backoff_count = 0
        if matched_budget:
            assert reference_prompt_tokens is not None
            while selected_items:
                candidate_prompt = self.answer_prompt(
                    question=features.question,
                    context_text=context_text,
                    source_support_constraint=policy.source_support_constraint,
                )
                if self.token_counter.count(candidate_prompt) <= reference_prompt_tokens:
                    break
                if policy.routing_policy == "all_raw_messages":
                    selected_items.pop(0)
                else:
                    selected_items.pop()
                budget_truncated = True
                matched_boundary_backoff_count += 1
                context_text = "\n\n".join(
                    str(item["text"])
                    for item in selected_items
                    if item["text"]
                )
        visible_page_ids = _dedupe(
            item.get("page_id") for item in selected_items if item.get("page_id")
        )
        selected_trajectory_ids = _dedupe(
            item.get("trajectory_id")
            for item in selected_items
            if item.get("trajectory_id")
        )
        selected_snapshot_ids = _dedupe(
            item.get("snapshot_id")
            for item in selected_items
            if item.get("snapshot_id")
        )
        selected_source_message_ids = _dedupe(
            message_id
            for item in selected_items
            for message_id in (
                [item.get("source_message_id")]
                if item.get("source_message_id")
                else list(item.get("source_message_ids") or [])
            )
        )
        selected_source_refs = _dedupe(
            ref
            for item in selected_items
            for ref in list(item.get("source_refs") or [])
        )
        prompt = self.answer_prompt(
            question=features.question,
            context_text=context_text,
            source_support_constraint=policy.source_support_constraint,
        )
        counted_prompt_tokens = self.token_counter.count(prompt)
        prompt_token_utilization = (
            counted_prompt_tokens / reference_prompt_tokens
            if matched_budget and reference_prompt_tokens
            else None
        )
        return VariantContext(
            variant=variant,
            sample_id=features.sample_id,
            query_task_id=features.query_task_id,
            policy=asdict(policy),
            selected_page_ids=_dedupe(page_ids),
            selected_trajectory_ids=selected_trajectory_ids,
            selected_snapshot_ids=selected_snapshot_ids,
            selected_source_message_ids=selected_source_message_ids,
            selected_source_refs=selected_source_refs,
            context_text=context_text,
            estimated_context_tokens=self.token_counter.count(context_text),
            estimated_prompt_tokens=counted_prompt_tokens,
            max_prompt_tokens=effective_max_total_tokens,
            output_token_reserve=self.output_token_reserve,
            budget_truncated=budget_truncated,
            token_estimator=self.token_counter.name,
            token_counter=self.token_counter.name,
            token_counter_exact=self.token_counter.exact,
            token_safety_margin=self.token_safety_margin,
            budget_mode=(
                "full_prompt_matched_v1" if matched_budget else "fixed_total_v1"
            ),
            global_max_total_tokens=self.max_prompt_tokens,
            reference_variant="full" if matched_budget else None,
            reference_prompt_tokens=(
                reference_prompt_tokens if matched_budget else None
            ),
            prompt_token_utilization=prompt_token_utilization,
            metadata={
                **ranking_metadata,
                "legacy_estimated_context_tokens": estimate_context_tokens(
                    context_text
                ),
                "counter_context_tokens": self.token_counter.count(context_text),
                "counter_prompt_tokens": self.token_counter.count(prompt),
                "effective_max_total_tokens": effective_max_total_tokens,
                "global_max_total_tokens": self.max_prompt_tokens,
                "matched_budget_passed_before_call": (
                    counted_prompt_tokens <= reference_prompt_tokens
                    if matched_budget and reference_prompt_tokens is not None
                    else None
                ),
                "matched_boundary_backoff_count": (
                    matched_boundary_backoff_count if matched_budget else None
                ),
                "context_item_count_before_budget": len(context_items),
                "context_item_count_after_budget": len(selected_items),
                "budget_source_support_constraint": (
                    budget_source_support_constraint
                ),
                "visible_page_ids": visible_page_ids,
                "retrieval_selected_trajectory_ids": _dedupe(trajectory_ids),
                "retrieval_selected_snapshot_ids": _dedupe(snapshot_ids),
                "retrieval_selected_source_message_ids": _dedupe(source_message_ids),
            },
        )

    @staticmethod
    def answer_prompt(
        *,
        question: str,
        context_text: str,
        source_support_constraint: bool,
    ) -> str:
        grounding = (
            "Answer only with facts supported by the context. Do not add unsupported items. "
            "If the context is insufficient, say that the answer is not supported. "
            "When source references are visible, preserve them in a final 'Sources:' line."
            if source_support_constraint
            else "Answer the question using the context. Give a concise direct answer."
        )
        return (
            f"{grounding}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

    def _context_items(
        self,
        *,
        policy: ContextRenderPolicy,
        page_ids: list[str],
        trajectory_ids: list[str],
        snapshot_ids: list[str],
        source_message_ids: list[str],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if policy.include_wiki_pages:
            for page_id in page_ids:
                text = collapse_whitespace(
                    self.memory_index["page_texts"].get(page_id, "")
                )
                if text:
                    items.append(
                        {
                            "item_type": "wiki_page",
                            "page_id": page_id,
                            "linked_trajectory_ids": list(
                                self.memory_index["page_to_trajectory_ids"].get(
                                    page_id, []
                                )
                            ),
                            "text": f"[WIKI PAGE {page_id}]\n{text}",
                            "source_refs": [],
                        }
                    )
        if policy.include_trajectory_summaries:
            for trajectory_id in trajectory_ids:
                metadata = dict(
                    self.memory_index["trajectory_metadata"].get(trajectory_id, {})
                )
                summary = collapse_whitespace(
                    metadata.get("retrieval_summary_text")
                    or metadata.get("trajectory_identity_summary_v1")
                    or metadata.get("label")
                    or ""
                )
                if summary:
                    items.append(
                        {
                            "item_type": "trajectory_summary",
                            "trajectory_id": trajectory_id,
                            "text": f"[TRAJECTORY {trajectory_id}]\n{summary}",
                            "source_refs": sorted(
                                self.memory_index["trajectory_refs"].get(
                                    trajectory_id, set()
                                )
                            ),
                        }
                    )
        included_source_message_ids: set[str] = set()
        allowed_source_message_ids = set(source_message_ids)
        if policy.evidence_policy not in {"wiki_summary_context", "raw_messages"}:
            for snapshot_id in snapshot_ids:
                snapshot = dict(
                    self.memory_index["snapshot_records"].get(snapshot_id) or {}
                )
                if not snapshot:
                    continue
                claims = list(
                    self.memory_index["claims_by_snapshot"].get(snapshot_id, [])
                )
                rendered_claims: list[str] = []
                if policy.evidence_policy == "flat_claim_state":
                    rendered_claims = [
                        f"- {claim.get('claim_id')}: "
                        f"{collapse_whitespace(claim.get('text') or '')}"
                        for claim in claims
                        if collapse_whitespace(claim.get("text") or "")
                    ]
                    claim_heading = "Claims (lifecycle state hidden):"
                else:
                    active_claims = [
                        claim
                        for claim in claims
                        if str(claim.get("status") or "").lower() == "active"
                        and not bool(
                            dict(claim.get("metadata") or {}).get(
                                "speaker_grounding_suspect_v1"
                            )
                        )
                    ]
                    uncertain_claims = [
                        claim
                        for claim in claims
                        if str(claim.get("status") or "").lower()
                        in {"contradictory", "needs-confirmation"}
                    ]
                    rendered_claims = [
                        *[
                            f"- {claim.get('claim_id')}: "
                            f"{collapse_whitespace(claim.get('text') or '')}"
                            for claim in active_claims
                        ],
                        *(
                            ["Uncertainty / Conflict Claims:"]
                            if uncertain_claims
                            else []
                        ),
                        *[
                            f"- [{claim.get('status')}] {claim.get('claim_id')}: "
                            f"{collapse_whitespace(claim.get('text') or '')}"
                            for claim in uncertain_claims
                        ],
                    ]
                    claim_heading = "Active Claims:"
                linked_source_ids = [
                    message_id
                    for message_id in list(snapshot.get("source_message_ids") or [])
                    if message_id in allowed_source_message_ids
                ]
                included_source_message_ids.update(linked_source_ids)
                source_lines = [
                    self._raw_source_line(
                        self.memory_index["raw_messages_by_id"][message_id]
                    )
                    for message_id in linked_source_ids
                    if message_id in self.memory_index["raw_messages_by_id"]
                ]
                source_refs = _dedupe(
                    self.memory_index["raw_messages_by_id"][message_id].get(
                        "source_ref"
                    )
                    for message_id in linked_source_ids
                    if message_id in self.memory_index["raw_messages_by_id"]
                )
                lines = [
                    f"Episodic Snapshot {snapshot_id}",
                    f"Timestamp: {snapshot.get('timestamp') or ''}",
                    f"Summary: {snapshot.get('summary_content') or ''}",
                    f"Context: {snapshot.get('context') or ''}",
                    "Keywords: " + ", ".join(snapshot.get("keywords") or []),
                    "Flags: " + ", ".join(snapshot.get("status_flags") or []),
                    claim_heading,
                    *(rendered_claims or ["- none"]),
                ]
                if source_lines:
                    lines.extend(["Associated Source Messages:", *source_lines])
                items.append(
                    {
                        "item_type": "snapshot",
                        "trajectory_id": snapshot.get("trajectory_id"),
                        "snapshot_id": snapshot_id,
                        "source_message_ids": linked_source_ids,
                        "text": "\n".join(lines),
                        "source_refs": source_refs,
                    }
                )
        if policy.include_raw_sources:
            raw_messages = self.memory_index["raw_messages_by_id"]
            for message_id in source_message_ids:
                if message_id in included_source_message_ids:
                    continue
                message = raw_messages.get(message_id)
                if not message:
                    continue
                source_ref = str(message.get("source_ref") or "")
                items.append(
                    {
                        "item_type": "raw_source",
                        "source_message_id": message_id,
                        "text": self._raw_source_line(message),
                        "source_refs": [source_ref] if source_ref else [],
                    }
                )
        return items

    @staticmethod
    def _raw_source_line(message: dict[str, Any]) -> str:
        message_id = str(message.get("id") or "")
        source_ref = str(message.get("source_ref") or "")
        return (
            f"[SOURCE {source_ref or message_id} "
            f"speaker={message.get('speaker_name') or message.get('role') or 'unknown'} "
            f"time={message.get('occurred_at') or 'unknown'}]\n"
            f"{collapse_whitespace(message.get('content') or '')}"
        )

    def _budget_items(
        self,
        items: list[dict[str, Any]],
        token_budget: int,
        *,
        allow_partial_items: bool = True,
    ) -> tuple[list[dict[str, Any]], bool]:
        selected: list[dict[str, Any]] = []
        used = 0
        truncated = False
        for item in items:
            text = str(item.get("text") or "")
            prefix = "\n\n" if selected else ""
            tokens = self.token_counter.count(prefix + text)
            remaining = token_budget - used
            if tokens <= remaining:
                selected.append(dict(item))
                used += tokens
                continue
            truncated = True
            if allow_partial_items and remaining > 0:
                prefix_tokens = self.token_counter.count(prefix)
                shortened, _ = self.token_counter.truncate(
                    text,
                    max(0, remaining - prefix_tokens),
                )
                if shortened:
                    original_refs = {
                        str(ref)
                        for ref in list(item.get("source_refs") or [])
                        if str(ref).strip()
                    }
                    visible_refs = sorted(
                        set(extract_source_refs(shortened)) & original_refs
                    )
                    partial_item = {
                        **item,
                        "text": shortened,
                        "source_refs": visible_refs,
                        "truncated": True,
                    }
                    visible_ref_set = set(visible_refs)
                    if "source_message_ids" in item:
                        partial_item["source_message_ids"] = [
                            message_id
                            for message_id in list(
                                item.get("source_message_ids") or []
                            )
                            if str(
                                self.memory_index["raw_messages_by_id"]
                                .get(str(message_id), {})
                                .get("source_ref")
                                or ""
                            )
                            in visible_ref_set
                        ]
                    if item.get("source_message_id"):
                        message_id = str(item["source_message_id"])
                        message_ref = str(
                            self.memory_index["raw_messages_by_id"]
                            .get(message_id, {})
                            .get("source_ref")
                            or ""
                        )
                        partial_item["source_message_id"] = (
                            message_id if message_ref in visible_ref_set else None
                        )
                    selected.append(partial_item)
            break
        return selected, truncated or len(selected) < len(items)

    def _budget_items_for_policy(
        self,
        items: list[dict[str, Any]],
        token_budget: int,
        *,
        policy: ContextRenderPolicy,
        allow_partial_items: bool = True,
    ) -> tuple[list[dict[str, Any]], bool]:
        if policy.routing_policy != "all_raw_messages":
            return self._budget_items(
                items,
                token_budget,
                allow_partial_items=allow_partial_items,
            )

        # A budgeted full-context baseline follows standard left truncation:
        # retain the newest complete messages, then restore chronological order.
        selected, truncated = self._budget_items(
            list(reversed(items)),
            token_budget,
            allow_partial_items=False,
        )
        selected.reverse()
        return selected, truncated

    def _rag_token_counter(self) -> TokenCounter:
        if self._rag_chunk_counter is not None:
            return self._rag_chunk_counter
        if self.embedding_provider is None:
            self._rag_chunk_counter = TokenCounter(
                name=TOKEN_ESTIMATOR_NAME,
                exact=False,
            )
            return self._rag_chunk_counter
        model_name = str(self.embedding_provider.model_info().model_name or "")
        if model_name == "hash-embedding":
            self._rag_chunk_counter = TokenCounter(
                name=TOKEN_ESTIMATOR_NAME,
                exact=False,
            )
        else:
            self._rag_chunk_counter = build_token_counter(
                "hf",
                model_name=model_name,
                require_exact=self.token_counter.exact,
            )
        return self._rag_chunk_counter

    def _chunk_conversation(
        self,
        sample_id: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rendered_messages = [
            self._raw_source_line(message) for message in messages
        ]
        conversation = "\n\n".join(rendered_messages)
        message_spans: list[tuple[int, int, str, str]] = []
        cursor = 0
        for message, rendered in zip(messages, rendered_messages, strict=False):
            start = cursor
            end = start + len(rendered)
            message_spans.append(
                (
                    start,
                    end,
                    str(message.get("id") or ""),
                    str(message.get("source_ref") or ""),
                )
            )
            cursor = end + 2
        counter = self._rag_token_counter()
        chunks: list[dict[str, Any]] = []
        if counter.codec is not None:
            token_ids = counter.codec.encode(conversation)
            offsets: list[tuple[int, int]] | None = None
            encode_with_offsets = getattr(
                counter.codec,
                "encode_with_offsets",
                None,
            )
            if callable(encode_with_offsets):
                try:
                    offset_token_ids, candidate_offsets = encode_with_offsets(
                        conversation
                    )
                    if offset_token_ids == token_ids:
                        offsets = candidate_offsets
                except (
                    KeyError,
                    NotImplementedError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    offsets = None
            step = self.rag_chunk_size - self.rag_chunk_overlap
            for chunk_index, start in enumerate(range(0, len(token_ids), step)):
                token_window = token_ids[start : start + self.rag_chunk_size]
                text = counter.codec.decode(token_window).strip()
                source_message_ids: list[str] = []
                refs: list[str] = []
                if offsets is not None and token_window:
                    window_offsets = offsets[start : start + len(token_window)]
                    nonempty_offsets = [
                        (char_start, char_end)
                        for char_start, char_end in window_offsets
                        if char_end > char_start
                    ]
                    if nonempty_offsets:
                        char_start = min(item[0] for item in nonempty_offsets)
                        char_end = max(item[1] for item in nonempty_offsets)
                        source_message_ids = _dedupe(
                            message_id
                            for (
                                message_start,
                                message_end,
                                message_id,
                                _,
                            ) in message_spans
                            if message_id
                            and message_start < char_end
                            and message_end > char_start
                        )
                        refs = _dedupe(
                            source_ref
                            for (
                                message_start,
                                message_end,
                                _,
                                source_ref,
                            ) in message_spans
                            if source_ref
                            and message_start < char_end
                            and message_end > char_start
                        )
                if not refs:
                    refs = sorted(extract_source_refs(text))
                if not source_message_ids:
                    source_message_ids = self.source_message_ids_for_context_refs(
                        sample_id=sample_id,
                        refs=refs,
                    )
                chunks.append(
                    {
                        "contains_sensitive_text": True,
                        "chunk_id": f"{sample_id}__chunk_{chunk_index}",
                        "text": text,
                        "source_refs": refs,
                        "source_message_ids": source_message_ids,
                        "chunk_token_count": len(token_window),
                        "source_provenance_mode": (
                            "token_offset_overlap_v1"
                            if offsets is not None
                            else "visible_ref_fallback_v1"
                        ),
                    }
                )
                if start + self.rag_chunk_size >= len(token_ids):
                    break
            return chunks

        words = conversation.split()
        step = self.rag_chunk_size - self.rag_chunk_overlap
        for chunk_index, start in enumerate(range(0, len(words), step)):
            word_window = words[start : start + self.rag_chunk_size]
            text = " ".join(word_window)
            refs = sorted(extract_source_refs(text))
            chunks.append(
                {
                    "contains_sensitive_text": True,
                    "chunk_id": f"{sample_id}__chunk_{chunk_index}",
                    "text": text,
                    "source_refs": refs,
                    "source_message_ids": self.source_message_ids_for_context_refs(
                        sample_id=sample_id,
                        refs=refs,
                    ),
                    "chunk_token_count": len(word_window),
                    "source_provenance_mode": "visible_ref_fallback_v1",
                }
            )
            if start + self.rag_chunk_size >= len(words):
                break
        return chunks

    def _naive_rag_index(
        self,
        sample_id: str,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], np.ndarray, str]:
        if self.embedding_provider is None:
            raise ValueError("naive_dense_rag requires an embedding provider.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(self.embedding_provider.model_info().model_name or "")
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "messages": [
                        {
                            "id": message.get("id"),
                            "source_ref": message.get("source_ref"),
                            "speaker_name": message.get("speaker_name"),
                            "occurred_at": message.get("occurred_at"),
                            "content": message.get("content"),
                        }
                        for message in messages
                    ],
                    "embedding_model": model_name,
                    "chunk_size": self.rag_chunk_size,
                    "chunk_overlap": self.rag_chunk_overlap,
                    "normalize_embeddings": True,
                    "query_encoding": "document_embedding",
                    "chunk_tokenizer": self._rag_token_counter().name,
                    "chunk_provenance": "token_offset_overlap_v1",
                },
                ensure_ascii=True,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        vector_path = self.cache_dir / f"naive_rag_{fingerprint[:20]}.npz"
        chunk_path = self.cache_dir / f"naive_rag_{fingerprint[:20]}.json"
        if vector_path.exists() and chunk_path.exists():
            chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
            with np.load(vector_path, allow_pickle=False) as payload:
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
            if len(chunks) == len(vectors):
                return chunks, vectors, fingerprint

        chunks = self._chunk_conversation(sample_id, messages)
        vectors = np.asarray(
            self.embedding_provider.embed_documents(
                [str(chunk.get("text") or "") for chunk in chunks]
            ),
            dtype=np.float32,
        )
        if vectors.size:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0.0, 1.0, norms)
        chunk_tmp = chunk_path.with_suffix(".tmp")
        chunk_tmp.write_text(
            json.dumps(chunks, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        chunk_tmp.replace(chunk_path)
        vector_tmp = vector_path.with_suffix(".tmp.npz")
        np.savez_compressed(vector_tmp, vectors=vectors)
        vector_tmp.replace(vector_path)
        return chunks, vectors, fingerprint

    def _rank_naive_dense_chunks(
        self,
        sample_id: str,
        question: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages = list(self.memory_index["sample_raw_messages"].get(sample_id, []))
        if not messages:
            return [], {
                "naive_rag_candidate_universe_size": 0,
                "naive_rag_score_rows": [],
            }
        if self.embedding_provider is None:
            raise ValueError("naive_dense_rag requires an embedding provider.")
        chunks, vectors, fingerprint = self._naive_rag_index(sample_id, messages)
        query_vector = np.asarray(
            self.embedding_provider.embed_documents([question])[0],
            dtype=np.float32,
        )
        query_norm = float(np.linalg.norm(query_vector)) or 1.0
        query_vector = query_vector / query_norm
        scores = (
            vectors @ query_vector
            if vectors.size
            else np.zeros((len(chunks),), dtype=np.float32)
        )
        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(scores[index]), str(chunks[index]["chunk_id"])),
        )
        top_indices = ranked_indices[: self.rag_top_k]
        items = [
            {
                "item_type": "rag_chunk",
                "chunk_id": chunks[index]["chunk_id"],
                "source_refs": list(chunks[index].get("source_refs") or []),
                "source_message_ids": list(
                    chunks[index].get("source_message_ids") or []
                ),
                "text": (
                    f"[RAG CHUNK {chunks[index]['chunk_id']}]\n"
                    f"{chunks[index].get('text') or ''}"
                ),
                "dense_score": float(scores[index]),
            }
            for index in top_indices
        ]
        score_rows = [
            {
                "rank": rank,
                "chunk_id": chunks[index]["chunk_id"],
                "dense_score": float(scores[index]),
                "source_refs": list(chunks[index].get("source_refs") or []),
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]
        return items, {
            "naive_rag_candidate_universe_size": len(chunks),
            "naive_rag_score_rows": score_rows,
            "naive_rag_index_fingerprint": fingerprint,
            "naive_rag_chunk_size": self.rag_chunk_size,
            "naive_rag_chunk_overlap": self.rag_chunk_overlap,
            "naive_rag_top_k": self.rag_top_k,
            "naive_rag_query_encoding": "document_embedding",
            "naive_rag_chunk_tokenizer": self._rag_token_counter().name,
            "naive_rag_chunk_tokenizer_exact": self._rag_token_counter().exact,
        }

    def _rank_hybrid_raw_messages(
        self,
        sample_id: str,
        question: str,
    ) -> tuple[list[str], dict[str, Any]]:
        cache_key = (sample_id, question)
        cached = self._hybrid_ranking_cache.get(cache_key)
        if cached is not None:
            ranked, metadata = cached
            return list(ranked), copy.deepcopy(metadata)
        messages = list(self.memory_index["sample_raw_messages"].get(sample_id, []))
        if not messages:
            return [], {"hybrid_dense_available": False, "hybrid_score_rows": []}
        query_terms = extract_keywords(question)
        lexical_scores: dict[str, float] = {}
        for message in messages:
            message_terms = extract_keywords(
                " ".join(
                    str(message.get(key) or "")
                    for key in ("speaker_name", "source_ref", "content", "occurred_at")
                )
            )
            overlap = query_terms & message_terms
            lexical_scores[str(message["id"])] = float(len(overlap)) + (
                float(len(overlap)) / float(len(query_terms)) if query_terms else 0.0
            )
        lexical_ranked = sorted(
            lexical_scores,
            key=lambda message_id: (
                -lexical_scores[message_id],
                int(self.memory_index["raw_messages_by_id"][message_id]["turn_index"]),
            ),
        )
        dense_scores: dict[str, float] = {}
        if self.embedding_provider is not None:
            vectors = self._raw_message_embeddings(sample_id, messages)
            query_vector = np.asarray(
                self.embedding_provider.embed_queries([question])[0],
                dtype=np.float32,
            )
            query_norm = float(np.linalg.norm(query_vector)) or 1.0
            for message, vector in zip(messages, vectors, strict=False):
                vector_norm = float(np.linalg.norm(vector)) or 1.0
                dense_scores[str(message["id"])] = float(
                    np.dot(query_vector, vector) / (query_norm * vector_norm)
                )
        dense_ranked = sorted(
            dense_scores,
            key=lambda message_id: (-dense_scores[message_id], message_id),
        )
        lexical_rank = {
            message_id: rank for rank, message_id in enumerate(lexical_ranked, start=1)
        }
        dense_rank = {
            message_id: rank for rank, message_id in enumerate(dense_ranked, start=1)
        }
        fused_scores = {
            str(message["id"]): (
                1.0 / (60 + lexical_rank[str(message["id"])])
                + (
                    1.0 / (60 + dense_rank[str(message["id"])])
                    if str(message["id"]) in dense_rank
                    else 0.0
                )
            )
            for message in messages
        }
        ranked = sorted(
            fused_scores,
            key=lambda message_id: (-fused_scores[message_id], message_id),
        )
        score_rows = [
            {
                "rank": rank,
                "message_id": message_id,
                "source_ref": self.memory_index["raw_messages_by_id"][message_id].get(
                    "source_ref"
                ),
                "fused_score": fused_scores[message_id],
                "lexical_score": lexical_scores[message_id],
                "dense_score": dense_scores.get(message_id),
                "lexical_rank": lexical_rank[message_id],
                "dense_rank": dense_rank.get(message_id),
            }
            for rank, message_id in enumerate(ranked, start=1)
        ]
        metadata = {
            "hybrid_dense_available": bool(dense_scores),
            "hybrid_score_rows": score_rows,
            "hybrid_candidate_universe_size": len(ranked),
        }
        self._hybrid_ranking_cache[cache_key] = (
            list(ranked),
            copy.deepcopy(metadata),
        )
        return ranked, metadata

    def _raw_message_embeddings(
        self,
        sample_id: str,
        messages: list[dict[str, Any]],
    ) -> np.ndarray:
        assert self.embedding_provider is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        message_ids = [str(message["id"]) for message in messages]
        texts = [str(message.get("content") or "") for message in messages]
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "message_ids": message_ids,
                    "texts": texts,
                    "model": self.embedding_provider.model_info().model_name,
                },
                ensure_ascii=True,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"raw_embeddings_{fingerprint[:20]}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as payload:
                return np.asarray(payload["vectors"], dtype=np.float32)
        vectors = np.asarray(
            self.embedding_provider.embed_documents(texts),
            dtype=np.float32,
        )
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, vectors=vectors)
        tmp.replace(path)
        return vectors

    def source_message_ids_for_context_refs(
        self,
        *,
        sample_id: str,
        refs: Iterable[str],
    ) -> list[str]:
        return source_message_ids_for_refs(
            self.memory_index,
            sample_id=sample_id,
            source_refs=refs,
        )
