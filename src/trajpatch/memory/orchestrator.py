"""Agent memory extraction, trajectory tracking, and snapshot persistence."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from trajpatch.config import RunConfig
from trajpatch.exceptions import ParserValidationError, StructuredOutputError
from trajpatch.ids import claim_id, op_id, snapshot_id
from trajpatch.memory.extraction_recovery import (
    PartialMemoryDraft,
    build_fallback_episodic_memory,
    build_partial_memory_draft,
    build_section_repair_prompt,
    derive_exchange_timestamp,
    merge_partial_memory_drafts,
    normalize_memory_text,
    render_episodic_memory,
    render_partial_memory_draft,
)
from trajpatch.memory.facets import (
    assign_claim_metadata_v1,
    build_incoming_match_features,
    build_sample_entity_lexicon,
    build_trajectory_entity_facet_summary,
    exact_term_keyword_set,
    normalize_entity_key,
)
from trajpatch.memory.historical import (
    build_trajectory_historical_evidence_card,
    sanitize_historical_item_terms,
    specific_terms,
    term_keys,
)
from trajpatch.memory.llm_text_parsers import (
    parse_claim_signal_extraction,
    parse_claim_text_extraction,
    parse_claim_transition_decision,
    parse_episodic_memory,
    parse_match_decision,
)
from trajpatch.memory.preservation import (
    MustPreserveCandidate,
    audit_claim_preservation,
    extract_must_preserve_candidates,
    raw_records_from_normalized,
)
from trajpatch.memory.readability import (
    clean_readable_values,
    is_fragment_like,
    is_readable_claim_text,
    normalized_surface,
    surface_key,
    surface_supported,
)
from trajpatch.memory.retrieval import cosine_similarity
from trajpatch.memory.schemas import (
    ClaimSignalExactTerm,
    ClaimSignalExtractionResult,
    ClaimSignalFacet,
    ClaimOp,
    ClaimTextExtractionResult,
    ClaimTransitionDecision,
    EpisodicMemoryInput,
    MemoryClaim,
)
from trajpatch.memory.trajectory_summaries import (
    build_deterministic_retrieval_summary,
    build_summary_supporting_state,
    fallback_summary_from_metadata,
    removed_internal_summary_keywords,
    sanitize_summary_keyword_values,
    summary_keywords_v2,
)
from trajpatch.prompts import load_prompt
from trajpatch.providers.base import EmbeddingProvider, LLMProvider
from trajpatch.providers.structured_outputs import (
    episodic_input_from_structured,
    get_structured_task_spec,
    structured_prompt_name,
    structured_strategy_for_vendor,
    validate_claim_transition_judge_result,
    validate_trajectory_match_result,
)
from trajpatch.storage.models import ClaimOpRecord, ClaimRecord, EpisodicMemorySnapshot
from trajpatch.storage.repository import TrajWikiStore
from trajpatch.types import NormalizedMessage
from trajpatch.utils.json_utils import write_json
from trajpatch.utils.text import collapse_whitespace, extract_keywords, keyword_overlap_score


@dataclass(slots=True)
class ParsedMemory:
    memory_type: str
    semantic_text: str
    links: list[str]
    claims: list[MemoryClaim]
    raw: EpisodicMemoryInput
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionResult:
    parsed: EpisodicMemoryInput | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrecomputedGenerationAttempt:
    text: str
    prompt_text: str
    generation_latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    batch_size: int | None = None
    batch_index: int | None = None


@dataclass(slots=True)
class StructuredFirstPassAttempt:
    task: str
    vendor: str
    response: Any | None = None
    error: Exception | None = None
    strategy: str | None = None
    exchange_number: int | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class StructuredFinalizeResult:
    parsed_memory: ParsedMemory | None
    required_repair_or_fallback: bool = False


@dataclass(slots=True)
class NoMemoryForceRecallDecision:
    should_force: bool
    seed_source: str
    reason: str
    low_salience: bool
    salience_detected: bool


_ACK_ONLY_TOKENS = {
    "bye",
    "goodbye",
    "great",
    "later",
    "no",
    "noted",
    "ok",
    "okay",
    "see",
    "soon",
    "sounds",
    "sure",
    "talk",
    "thank",
    "thanks",
    "understood",
    "yes",
    "you",
}
_EPISODIC_SALIENCE_PATTERNS = (
    (
        "place_or_area_fact",
        re.compile(
            r"\b(?:old area|home\s*town|hometown|home country|neighbou?rhood|county|city|place|area|near|lives? in|moved to|moved from|returned from|came back from)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "incident_or_infrastructure_fact",
        re.compile(
            r"\b(?:flood|storm|fire|accident|earthquake|hit|damaged|ruined|power cut|infrastructure|roadways?|housing|homes?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "plan_goal_or_state_change",
        re.compile(
            r"\b(?:want(?:s)? to|goal(?:s)?|number one goal|focusing on|focus(?:ed)? on|improv(?:e|ing)|better|plan(?:s|ned|ning)?|working on|joined|donated|volunteered|moved|graduated|started|stopped)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "family_or_adoption_fact",
        re.compile(
            r"\b(?:adoption agencies|adoption agency|adopt(?:ion|ing)?|family|loving home|kids who need|children in need)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "media_title_or_recommendation_fact",
        re.compile(
            r"(?:[\"“”'][^\"“”']{2,80}[\"“”']|\b(?:book|books|novel|novels|series|movie|film|tv show|show|read|reading|watched|watching|recommend(?:ed|ing)?|favorites?|collection)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "list_item_or_inventory_fact",
        re.compile(
            r"\b(?:bought|buy|refurbished|equipment|headphones|mouse|desk|keyboard|hats?|turtles?|dogs?|breeds?|lab mixes?|chihuahua mixes?|snacks?|soda|candy|seltzer|chocolate|popcorn|fruit|veggies?|salad|energy balls?|candles?|oils?|essential oils?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "method_or_technique_fact",
        re.compile(
            r"\b(?:technique|techniques|strategy|strategies|method|methods|schedule|to-do list|todo list|pomodoro|eisenhower|matrix|bullet journal|time management)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "activity_or_practice_fact",
        re.compile(
            r"\b(?:yoga|aerial yoga|kundalini|martial arts?|kickboxing|taekwondo|board games?|wine tasting|shelter|homeless shelter|pet shelter|hobbies?|relax|calming|destress|de-stress|long drives?|fixing cars?|camp(?:ed|ing)?|hiking|beach|mountains?|forest)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "image_adjacent_fact",
        re.compile(
            r"\[(?:shared image|image):|shared image:|photo of|picture of|look at (?:this|these|that)|take a look",
            re.IGNORECASE,
        ),
    ),
    (
        "capitalized_place_like_term",
        re.compile(
            r"\b[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3}\s+"
            r"(?:County|City|River|Lake|Park|School|Center|Centre|Road|Street|Hospital|Shelter|Neighbourhood|Neighborhood)\b"
        ),
    ),
)


def _is_acknowledgement_only_text(text: str) -> bool:
    tokens = re.findall(r"[a-z]+", text.casefold())
    return bool(tokens) and len(tokens) <= 8 and all(token in _ACK_ONLY_TOKENS for token in tokens)


def detect_episodic_salience_v1(exchange_messages: list[NormalizedMessage]) -> tuple[bool, str]:
    """Return whether a no-memory first pass is likely a recall false negative."""

    text = collapse_whitespace(" ".join(message.content for message in exchange_messages if message.role != "system"))
    if not text or _is_acknowledgement_only_text(text):
        return False, "none"
    for reason, pattern in _EPISODIC_SALIENCE_PATTERNS:
        if pattern.search(text):
            return True, reason
    return False, "none"


def _base_episodic_seed_metadata(*, source: str, llm_has_memory: bool) -> dict[str, Any]:
    return {
        "episodic_seed_source_v1": source,
        "llm_has_memory_v1": llm_has_memory,
        "forced_episodic_seed_used_v1": False,
        "llm_no_memory_overridden_v1": False,
        "low_salience_memory_v1": False,
    }


CLAIM_SIGNAL_METADATA_KEYS = (
    "exact_terms_v2",
    "facets_v2",
    "display_signals_v1",
    "signal_extraction_source",
    "signal_extraction_failed_v1",
    "signal_extraction_discarded_v1",
)


@dataclass(slots=True)
class CanonicalClaimView:
    claim_id: str
    text: str
    normalized_text: str
    status: str
    source_message_ids: list[str]
    parent_claim_id: str | None = None
    revised_from_claim_id: str | None = None


@dataclass(slots=True)
class MatchedClaimPair:
    previous_claim_id: str
    current_local_claim_id: str
    normalized_text: str


@dataclass(slots=True)
class DerivedClaimState:
    next_claim_views: list[CanonicalClaimView]
    derived_ops: list[ClaimOp]
    matched_claim_pairs: list[MatchedClaimPair]
    new_claim_count: int
    status_updated_count: int
    model_ops_ignored_count: int
    unmatched_previous_count: int
    transition_judge_attempt_count: int = 0
    transition_judge_success_count: int = 0
    transition_judge_fallback_count: int = 0
    transition_revise_count: int = 0
    transition_add_count: int = 0
    reused_claim_ids: list[str] = field(default_factory=list)
    new_claim_ids: list[str] = field(default_factory=list)
    transition_debug: list[dict[str, Any]] = field(default_factory=list)


class MemoryOrchestrator:
    def __init__(
        self,
        config: RunConfig,
        store: TrajWikiStore,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        *,
        debug_dir: Path | None = None,
        trace: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.debug_dir = debug_dir
        self.trace = trace
        self.parse_attempts = 0
        self.parse_successes = 0
        self.parse_failures = 0
        self.repair_rounds = 0
        self.extraction_fallbacks = 0
        self.closed_on_fallback = 0
        self.trajectory_match_total_open = 0
        self.trajectory_match_prefiltered = 0
        self.trajectory_match_shortlisted = 0
        self.structured_attempts = 0
        self.structured_successes = 0
        self.structured_fallbacks = 0
        self.structured_attempts_by_task: dict[str, int] = defaultdict(int)
        self.structured_successes_by_task: dict[str, int] = defaultdict(int)
        self.structured_fallbacks_by_task: dict[str, int] = defaultdict(int)
        self.structured_attempts_by_vendor: dict[str, int] = defaultdict(int)
        self.structured_successes_by_vendor: dict[str, int] = defaultdict(int)
        self.structured_fallbacks_by_vendor: dict[str, int] = defaultdict(int)
        self.link_salvage_count = 0
        self.link_exchange_fallback_count = 0
        self.ops_parse_failure_count = 0
        self.ops_ignored_count = 0
        self.ops_synthesized_count = 0
        self.ops_model_supplied_count = 0
        self.ops_model_hint_count = 0
        self.claims_parse_failure_count = 0
        self.claims_required_repair_count = 0
        self.claim_text_exact_match_count = 0
        self.claim_status_updated_count = 0
        self.claim_new_add_count = 0
        self.claim_unmatched_previous_count = 0
        self.claim_transition_judge_attempt_count = 0
        self.claim_transition_judge_success_count = 0
        self.claim_transition_judge_fallback_count = 0
        self.claim_transition_revise_count = 0
        self.claim_transition_add_count = 0
        self.empty_repair_target_count = 0
        self.forced_memory_seed_count = 0
        self.low_salience_memory_count = 0
        self.llm_no_memory_forced_count = 0
        self.zero_claim_episodic_candidate_count = 0
        self.zero_claim_episodic_persisted_count = 0
        self.zero_claim_low_salience_skipped_count = 0
        self.debug_artifact_paths: dict[str, list[str]] = defaultdict(list)
        self._sample_entity_lexicon_cache: dict[str, dict[str, str]] = {}
        self._exchange_counters: dict[str, int] = defaultdict(int)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embedding_provider, "embed_documents"):
            return self.embedding_provider.embed_documents(texts)
        return self.embedding_provider.embed(texts)

    def _embed_queries(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embedding_provider, "embed_queries"):
            return self.embedding_provider.embed_queries(texts)
        return self.embedding_provider.embed(texts)

    def _document_embedding_strategy(self) -> str:
        if hasattr(self.embedding_provider, "document_embedding_strategy"):
            return str(self.embedding_provider.document_embedding_strategy())
        return "shared_embed"

    def _sample_entity_lexicon(self, sample_id: str) -> dict[str, str]:
        cached = self._sample_entity_lexicon_cache.get(sample_id)
        if cached is not None:
            return cached
        lexicon = build_sample_entity_lexicon(self.store.list_raw_messages_for_sample(sample_id))
        self._sample_entity_lexicon_cache[sample_id] = lexicon
        return lexicon

    @staticmethod
    def _status_flags_from_claims(claims: list[MemoryClaim]) -> list[str]:
        order = ("active", "deprecated", "contradictory", "needs-confirmation")
        present = {claim.status for claim in claims}
        return [status for status in order if status in present]

    def _build_claim_text_extract_prompt(
        self,
        *,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        structured: bool = False,
    ) -> str:
        return (
            load_prompt(
                "episodic_claim_text_extract_structured" if structured else "episodic_claim_text_extract"
            )
            + "\n\nConversation:\n"
            + self._render_conversation(exchange_messages)
            + "\n\nInitial summary_content:\n"
            + parsed.raw.summary_content
            + "\n\nInitial context:\n"
            + parsed.raw.context
            + "\n\nInitial keywords:\n"
            + ", ".join(parsed.raw.keywords)
            + "\n\nInitial claims:\n"
            + "- none; claims must be generated only from the raw conversation above."
        )

    def build_claim_text_first_pass_messages(
        self,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        *,
        structured: bool = False,
    ) -> list[NormalizedMessage]:
        return [
            NormalizedMessage(
                role="user",
                content=self._build_claim_text_extract_prompt(
                    parsed=parsed,
                    exchange_messages=exchange_messages,
                    structured=structured,
                ),
                turn_index=0,
            )
        ]

    def request_claim_text_structured_first_pass(
        self,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        *,
        exchange_number: int | None = None,
    ) -> StructuredFirstPassAttempt:
        task = "episodic_claim_text_extract"
        vendor = self._structured_vendor()
        try:
            response = self.llm_provider.generate_structured(
                self.build_claim_text_first_pass_messages(
                    parsed,
                    exchange_messages,
                    structured=True,
                ),
                spec=get_structured_task_spec(task),
                metadata={"task": task, "memory_type": "episodic"},
            )
            return StructuredFirstPassAttempt(
                task=task,
                vendor=str(response.metadata.get("structured_vendor") or vendor),
                response=response,
                strategy=str(response.metadata.get("structured_strategy") or structured_strategy_for_vendor(vendor)),
                exchange_number=exchange_number,
            )
        except Exception as exc:  # noqa: BLE001
            return self._structured_first_pass_failure(
                task,
                vendor,
                exc,
                exchange_number=exchange_number,
            )

    def _validate_claim_text_result(
        self,
        result: ClaimTextExtractionResult,
        exchange_messages: list[NormalizedMessage],
    ) -> tuple[list[MemoryClaim], dict[str, str], list[dict[str, object]]]:
        message_by_id = {
            str(message.raw_message_id): message
            for message in exchange_messages
            if message.raw_message_id
        }
        accepted: list[MemoryClaim] = []
        quotes_by_local_id: dict[str, str] = {}
        discarded: list[dict[str, object]] = []
        if not result.has_claims:
            return [], {}, []
        for item in result.claims:
            source_ids = [source_id for source_id in item.source_message_ids if source_id in message_by_id]
            source_texts = [message_by_id[source_id].content for source_id in source_ids]
            text = normalized_surface(item.text)
            quote = normalized_surface(item.supporting_quote)
            reason = ""
            if not source_ids:
                reason = "no_valid_source_message_ids"
            elif not is_readable_claim_text(text):
                reason = "unreadable_claim_text"
            elif quote and not surface_supported(quote, source_texts):
                reason = "supporting_quote_not_in_source"
            if reason:
                discarded.append(
                    {
                        "text": item.text,
                        "supporting_quote": item.supporting_quote,
                        "source_message_ids": list(item.source_message_ids),
                        "reason": reason,
                    }
                )
                continue
            local_id = f"tmp-c{len(accepted) + 1}"
            accepted.append(
                MemoryClaim(
                    claim_id=local_id,
                    status=item.status,
                    source_message_ids=source_ids,
                    text=text,
                )
            )
            if quote:
                quotes_by_local_id[local_id] = quote
        return accepted, quotes_by_local_id, discarded

    def _fallback_claims_from_preservation_candidates(
        self,
        sample_id: str,
        exchange_messages: list[NormalizedMessage],
    ) -> tuple[list[MemoryClaim], list[dict[str, object]]]:
        source_messages = raw_records_from_normalized(exchange_messages)
        candidates = [
            candidate
            for candidate in extract_must_preserve_candidates(
                source_messages,
                entity_lexicon=self._sample_entity_lexicon(sample_id),
            )
            if candidate.confidence == "high"
        ]
        claims: list[MemoryClaim] = []
        seen_surfaces: set[str] = set()
        for candidate in candidates:
            surface = collapse_whitespace(candidate.surface)
            surface_key_value = surface.casefold()
            if not surface or surface_key_value in seen_surfaces:
                continue
            seen_surfaces.add(surface_key_value)
            claims.append(
                MemoryClaim(
                    claim_id=f"tmp-c{len(claims) + 1}",
                    status="active",
                    source_message_ids=list(candidate.source_message_ids),
                    text=self._preservation_candidate_fallback_text(candidate, exchange_messages),
                )
            )
        return claims, [candidate.to_metadata() for candidate in candidates]

    @staticmethod
    def _preservation_candidate_fallback_text(
        candidate: MustPreserveCandidate,
        exchange_messages: list[NormalizedMessage],
    ) -> str:
        surface = collapse_whitespace(candidate.surface)
        speaker = next(
            (
                collapse_whitespace(message.speaker_name or "")
                for message in exchange_messages
                if collapse_whitespace(message.speaker_name or "")
            ),
            "",
        )
        subject = speaker or "The exchange"
        category = str(candidate.category or "").strip().casefold()
        temporal_expression = collapse_whitespace(candidate.temporal_expression or "")
        temporal_suffix = f" {temporal_expression}" if temporal_expression else ""
        if category == "activity":
            action = collapse_whitespace(candidate.event_action or "")
            if action:
                return f"{subject} {action} {surface}{temporal_suffix}."
            return f"{subject} mentioned {surface} as an activity{temporal_suffix}."
        if category == "book_title":
            return f"{subject} mentioned {surface} as a book title."
        if category == "recipe":
            return f"{subject} mentioned {surface} as a recipe or dish."
        if category == "instrument":
            return f"{subject} mentioned {surface} as an instrument."
        if category == "symbol":
            return f"{subject} mentioned {surface} as a symbol."
        if category == "place":
            return f"{subject} mentioned {surface} as a place."
        if category == "painted_object":
            return f"{subject} mentioned {surface} as something painted{temporal_suffix}."
        if category == "event_type":
            return f"{subject} mentioned {surface} as an event{temporal_suffix}."
        if category == "event_object":
            return f"{subject} mentioned {surface} as an event object{temporal_suffix}."
        if category == "count":
            return f"{subject} mentioned the count {surface}."
        if candidate.relation == "research_topic":
            return f"{subject} mentioned {surface} as a research topic."
        return f"{subject} mentioned {surface}{temporal_suffix}."

    def _apply_preservation_candidate_fallback(
        self,
        parsed: ParsedMemory,
        candidates: list[MustPreserveCandidate],
        exchange_messages: list[NormalizedMessage],
        *,
        metadata_field: str,
    ) -> ParsedMemory:
        existing_surfaces = {collapse_whitespace(claim.text).casefold() for claim in parsed.claims}
        added_claims: list[MemoryClaim] = []
        for candidate in candidates:
            fallback_text = self._preservation_candidate_fallback_text(candidate, exchange_messages)
            if fallback_text.casefold() in existing_surfaces:
                continue
            existing_surfaces.add(fallback_text.casefold())
            added_claims.append(
                MemoryClaim(
                    claim_id=f"tmp-c{len(parsed.claims) + len(added_claims) + 1}",
                    status="active",
                    source_message_ids=list(candidate.source_message_ids),
                    text=fallback_text,
                )
            )
        if not added_claims:
            return parsed
        parsed.claims = [*parsed.claims, *added_claims]
        parsed.raw.claims = [*parsed.raw.claims, *added_claims]
        parsed.metadata[metadata_field] = True
        parsed.metadata["claim_preservation_fallback_added_count_v1"] = int(
            parsed.metadata.get("claim_preservation_fallback_added_count_v1", 0)
        ) + len(added_claims)
        return parsed

    def _apply_claim_text_llm_stage(
        self,
        sample_id: str,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        *,
        structured_first_pass: StructuredFirstPassAttempt | None = None,
        first_attempt: PrecomputedGenerationAttempt | None = None,
    ) -> ParsedMemory:
        task = "episodic_claim_text_extract"
        valid_link_ids = self.store.list_raw_message_ids(sample_id)
        exchange_link_ids = [message.raw_message_id for message in exchange_messages if message.raw_message_id]
        self._trace(f"sample={sample_id} claim_text_llm_start")
        started_at = time.perf_counter()
        result: ClaimTextExtractionResult | None = None
        stage_metadata: dict[str, Any] = {
            "claim_text_llm_used_v1": False,
            "claim_text_llm_failed_v1": False,
        }
        try:
            if structured_first_pass is not None:
                self._record_structured_attempt(task, structured_first_pass.vendor)
                if structured_first_pass.response is None:
                    vendor = structured_first_pass.vendor
                    self._record_structured_fallback(task, vendor)
                    failure = structured_first_pass.error or StructuredOutputError(
                        "Structured claim text request failed before producing a response.",
                        vendor=vendor,
                        strategy="deterministic_fallback",
                    )
                    raise failure
                response = structured_first_pass.response
                vendor = str(response.metadata.get("structured_vendor") or structured_first_pass.vendor)
                self._record_structured_success(task, vendor)
                result = response.parsed
                stage_metadata.update(
                    {
                        "claim_text_llm_structured_v1": True,
                        "claim_text_llm_prompt_tokens_v1": response.prompt_tokens,
                        "claim_text_llm_completion_tokens_v1": response.completion_tokens,
                        "claim_text_llm_batched_v1": True,
                    }
                )
            elif first_attempt is not None:
                parser_diagnostics: list[dict[str, Any]] = []
                result = parse_claim_text_extraction(
                    first_attempt.text,
                    valid_link_ids,
                    exchange_link_ids=exchange_link_ids,
                    diagnostics=parser_diagnostics,
                )
                stage_metadata.update(
                    {
                        "claim_text_llm_structured_v1": False,
                        "claim_text_llm_parser_diagnostics_v1": parser_diagnostics,
                        "claim_text_llm_prompt_tokens_v1": first_attempt.prompt_tokens,
                        "claim_text_llm_completion_tokens_v1": first_attempt.completion_tokens,
                        "claim_text_llm_batched_v1": True,
                        "claim_text_llm_batch_size_v1": first_attempt.batch_size,
                        "claim_text_llm_batch_index_v1": first_attempt.batch_index,
                        "claim_text_llm_response_metadata_v1": dict(first_attempt.response_metadata),
                    }
                )
            elif self.llm_provider.supports_structured(task):
                vendor = self._structured_vendor()
                self._record_structured_attempt(task, vendor)
                response = self.llm_provider.generate_structured(
                    self.build_claim_text_first_pass_messages(
                        parsed,
                        exchange_messages,
                        structured=True,
                    ),
                    spec=get_structured_task_spec(task),
                    metadata={"task": task, "memory_type": "episodic"},
                )
                vendor = str(response.metadata.get("structured_vendor") or vendor)
                self._record_structured_success(task, vendor)
                result = response.parsed
                stage_metadata.update(
                    {
                        "claim_text_llm_structured_v1": True,
                        "claim_text_llm_prompt_tokens_v1": response.prompt_tokens,
                        "claim_text_llm_completion_tokens_v1": response.completion_tokens,
                    }
                )
            else:
                parser_diagnostics: list[dict[str, Any]] = []
                response = self.llm_provider.generate(
                    self.build_claim_text_first_pass_messages(parsed, exchange_messages),
                    metadata={"task": task, "memory_type": "episodic"},
                )
                result = parse_claim_text_extraction(
                    response.text,
                    valid_link_ids,
                    exchange_link_ids=exchange_link_ids,
                    diagnostics=parser_diagnostics,
                )
                stage_metadata.update(
                    {
                        "claim_text_llm_structured_v1": False,
                        "claim_text_llm_parser_diagnostics_v1": parser_diagnostics,
                        "claim_text_llm_prompt_tokens_v1": response.prompt_tokens,
                        "claim_text_llm_completion_tokens_v1": response.completion_tokens,
                    }
                )
            claims, quotes_by_local_id, discarded = self._validate_claim_text_result(result, exchange_messages)
            if not claims:
                raise ParserValidationError("claim text LLM produced no accepted claims", code="claim_text_empty")
            parsed.claims = claims
            parsed.raw.claims = claims
            parsed.raw.status_flags = self._status_flags_from_claims(claims)  # type: ignore[assignment]
            parsed.raw.raw_text = render_episodic_memory(parsed.raw)
            stage_metadata.update(
                {
                    "claim_text_llm_used_v1": True,
                    "claim_text_llm_failed_v1": False,
                    "claim_text_llm_discarded_v1": discarded,
                    "claim_supporting_quotes_by_local_id_v1": dict(quotes_by_local_id),
                    "claim_supporting_quotes_by_text_v1": {
                        self._normalize_claim_text(claim.text): quotes_by_local_id.get(claim.claim_id, "")
                        for claim in claims
                    },
                }
            )
            parsed.metadata.update(stage_metadata)
            self._trace(
                f"sample={sample_id} claim_text_llm_done claims={len(claims)} discarded={len(discarded)} "
                f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
            )
            return parsed
        except Exception as exc:  # noqa: BLE001
            vendor = structured_first_pass.vendor if structured_first_pass is not None else self._structured_vendor()
            strategy = (
                structured_first_pass.strategy
                if structured_first_pass is not None and structured_first_pass.strategy
                else (exc.strategy if isinstance(exc, StructuredOutputError) and exc.strategy else structured_strategy_for_vendor(vendor))
            )
            exchange_text = (
                f" exchange={structured_first_pass.exchange_number}"
                if structured_first_pass is not None and structured_first_pass.exchange_number is not None
                else ""
            )
            batch_text = " from_precomputed_batch=true" if structured_first_pass is not None else ""
            stage_metadata.update(
                {
                    "claim_text_llm_used_v1": False,
                    "claim_text_llm_failed_v1": True,
                    "claim_text_llm_error_v1": f"{exc.__class__.__name__}: {exc}",
                }
            )
            fallback_claims, fallback_candidates = self._fallback_claims_from_preservation_candidates(
                sample_id,
                exchange_messages,
            )
            if fallback_claims:
                parsed.claims = fallback_claims
                parsed.raw.claims = fallback_claims
                parsed.raw.status_flags = self._status_flags_from_claims(fallback_claims)  # type: ignore[assignment]
                parsed.raw.raw_text = render_episodic_memory(parsed.raw)
                stage_metadata.update(
                    {
                        "claim_text_fallback_source_v1": "preservation_candidates",
                        "claim_text_fallback_candidate_count_v1": len(fallback_candidates),
                        "claim_text_fallback_claim_count_v1": len(fallback_claims),
                        "claim_text_empty_after_fallback_v1": False,
                    }
                )
                self._trace(
                    f"sample={sample_id} claim_text_fallback_used candidates={len(fallback_candidates)} "
                    f"claims={len(fallback_claims)}"
                )
            else:
                parsed.claims = []
                parsed.raw.claims = []
                parsed.raw.status_flags = []  # type: ignore[assignment]
                parsed.raw.raw_text = render_episodic_memory(parsed.raw)
                stage_metadata.update(
                    {
                        "claim_text_fallback_source_v1": "none",
                        "claim_text_fallback_candidate_count_v1": len(fallback_candidates),
                        "claim_text_fallback_claim_count_v1": 0,
                        "claim_text_empty_after_fallback_v1": True,
                    }
                )
                self._trace(f"sample={sample_id} claim_text_empty_after_fallback")
            parsed.metadata.update(stage_metadata)
            self._trace(
                f"sample={sample_id}{exchange_text} claim_text_llm_failed error={exc.__class__.__name__} "
                f"vendor={vendor} strategy={strategy}{batch_text} "
                f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
            )
            return parsed

    def _build_claim_signal_extract_prompt(
        self,
        *,
        claims: list[ClaimRecord],
        source_by_id: dict[str, Any],
        structured: bool = False,
    ) -> str:
        blocks: list[str] = []
        for claim in claims:
            metadata = dict(claim.metadata_json or {})
            source_lines = []
            for source_id in list(claim.source_message_ids_json or []):
                message = source_by_id.get(source_id)
                if message is None:
                    continue
                speaker = getattr(message, "speaker_name", None) or getattr(message, "role", "")
                source_lines.append(f"- {source_id} ({speaker}): {message.content}")
            quote = str(metadata.get("claim_supporting_quote_v1") or "")
            blocks.append(
                f"### {claim.claim_id}\n"
                f"status={claim.status}\n"
                f"text={claim.text}\n"
                f"supporting_quote={quote or 'none'}\n"
                "source_messages:\n"
                + ("\n".join(source_lines) if source_lines else "- none")
            )
        return (
            load_prompt("claim_signal_extract_structured" if structured else "claim_signal_extract")
            + "\n\nClaims and source snippets:\n"
            + "\n\n".join(blocks)
        )

    def _build_claim_signal_extract_prompt_for_memory_claims(
        self,
        *,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        structured: bool = False,
    ) -> str:
        source_by_id = {
            str(message.raw_message_id): message
            for message in exchange_messages
            if message.raw_message_id
        }
        quote_by_local_id = dict(parsed.metadata.get("claim_supporting_quotes_by_local_id_v1") or {})
        quote_by_text = dict(parsed.metadata.get("claim_supporting_quotes_by_text_v1") or {})
        blocks: list[str] = []
        for claim in parsed.claims:
            source_lines = []
            for source_id in list(claim.source_message_ids or []):
                message = source_by_id.get(source_id)
                if message is None:
                    continue
                speaker = getattr(message, "speaker_name", None) or getattr(message, "role", "")
                source_lines.append(f"- {source_id} ({speaker}): {message.content}")
            quote = quote_by_local_id.get(claim.claim_id) or quote_by_text.get(
                self._normalize_claim_text(claim.text)
            ) or ""
            blocks.append(
                f"### {claim.claim_id}\n"
                f"status={claim.status}\n"
                f"text={claim.text}\n"
                f"supporting_quote={quote or 'none'}\n"
                "source_messages:\n"
                + ("\n".join(source_lines) if source_lines else "- none")
            )
        return (
            load_prompt("claim_signal_extract_structured" if structured else "claim_signal_extract")
            + "\n\nClaims and source snippets:\n"
            + "\n\n".join(blocks)
        )

    def build_claim_signal_first_pass_messages(
        self,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        *,
        structured: bool = False,
    ) -> list[NormalizedMessage]:
        return [
            NormalizedMessage(
                role="user",
                content=self._build_claim_signal_extract_prompt_for_memory_claims(
                    parsed=parsed,
                    exchange_messages=exchange_messages,
                    structured=structured,
                ),
                turn_index=0,
            )
        ]

    def request_claim_signal_structured_first_pass(
        self,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        *,
        exchange_number: int | None = None,
    ) -> StructuredFirstPassAttempt:
        task = "claim_signal_extract"
        vendor = self._structured_vendor()
        try:
            response = self.llm_provider.generate_structured(
                self.build_claim_signal_first_pass_messages(
                    parsed,
                    exchange_messages,
                    structured=True,
                ),
                spec=get_structured_task_spec(task),
                metadata={"task": task, "memory_type": "episodic"},
            )
            return StructuredFirstPassAttempt(
                task=task,
                vendor=str(response.metadata.get("structured_vendor") or vendor),
                response=response,
                strategy=str(response.metadata.get("structured_strategy") or structured_strategy_for_vendor(vendor)),
                exchange_number=exchange_number,
            )
        except Exception as exc:  # noqa: BLE001
            return self._structured_first_pass_failure(
                task,
                vendor,
                exc,
                exchange_number=exchange_number,
            )

    @staticmethod
    def _claim_support_texts(claim: ClaimRecord, source_by_id: dict[str, Any]) -> list[str]:
        metadata = dict(claim.metadata_json or {})
        texts = [str(claim.text or ""), str(metadata.get("claim_supporting_quote_v1") or "")]
        for source_id in list(claim.source_message_ids_json or []):
            message = source_by_id.get(source_id)
            if message is not None:
                texts.append(str(getattr(message, "content", "") or ""))
        return [text for text in texts if text.strip()]

    @staticmethod
    def _clear_baseline_signal_fields(metadata: dict[str, Any]) -> None:
        """v1 signals are a transient baseline; new persisted claims should route via v2."""
        metadata.pop("exact_terms_v1", None)
        metadata.pop("facets_v1", None)

    def _apply_deterministic_signal_fallback(
        self,
        claims: list[ClaimRecord],
        *,
        target_claim_ids: set[str] | None = None,
    ) -> None:
        for claim in claims:
            if target_claim_ids is not None and claim.claim_id not in target_claim_ids:
                continue
            metadata = dict(claim.metadata_json or {})
            exact_terms = clean_readable_values(metadata.get("exact_terms_v1") or [], allow_single_word=True)
            facets = []
            for facet in list(metadata.get("facets_v1") or []):
                if not isinstance(facet, dict):
                    continue
                value = str(facet.get("value") or "")
                if value and is_fragment_like(value, allow_single_word=True):
                    continue
                facets.append(dict(facet))
            metadata.update(
                {
                    "exact_terms_v2": exact_terms,
                    "facets_v2": facets,
                    "display_signals_v1": {
                        "items": exact_terms,
                        "named_entities": [],
                        "counts": [],
                        "key_facts": [claim.text] if is_readable_claim_text(claim.text) else [],
                    },
                    "signal_extraction_source": "deterministic_fallback",
                    "signal_extraction_failed_v1": True,
                    "signal_extraction_discarded_v1": [],
                }
            )
            self._clear_baseline_signal_fields(metadata)
            claim.metadata_json = metadata

    def _apply_claim_signal_result(
        self,
        result: ClaimSignalExtractionResult,
        claims: list[ClaimRecord],
        source_by_id: dict[str, Any],
        *,
        target_claim_ids: set[str] | None = None,
    ) -> tuple[int, list[dict[str, object]]]:
        target_claims = [
            claim for claim in claims if target_claim_ids is None or claim.claim_id in target_claim_ids
        ]
        claim_by_id = {claim.claim_id: claim for claim in target_claims}
        exact_terms_by_claim: dict[str, list[str]] = defaultdict(list)
        facets_by_claim: dict[str, list[dict[str, object]]] = defaultdict(list)
        display_by_claim: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: {"items": [], "named_entities": [], "counts": [], "key_facts": []}
        )
        discarded: list[dict[str, object]] = []

        def support_claim_for_value(value: str) -> ClaimRecord | None:
            for claim in target_claims:
                if surface_supported(value, self._claim_support_texts(claim, source_by_id)):
                    return claim
            return None

        for term in result.exact_terms:
            claim = claim_by_id.get(term.source_claim_id)
            surface = normalized_surface(term.surface)
            reason = ""
            if claim is None:
                reason = "unknown_source_claim_id"
            elif is_fragment_like(surface, allow_single_word=True):
                reason = "fragment_like_surface"
            elif not surface_supported(surface, self._claim_support_texts(claim, source_by_id)):
                reason = "surface_not_grounded"
            if reason:
                discarded.append(
                    {
                        "kind": "exact_term",
                        "surface": term.surface,
                        "source_claim_id": term.source_claim_id,
                        "reason": reason,
                    }
                )
                continue
            if surface_key(surface) not in {surface_key(value) for value in exact_terms_by_claim[claim.claim_id]}:
                exact_terms_by_claim[claim.claim_id].append(surface)

        for facet in result.facets:
            claim = claim_by_id.get(facet.source_claim_id)
            value = normalized_surface(facet.value)
            relation = normalized_surface(facet.relation)
            reason = ""
            if claim is None:
                reason = "unknown_source_claim_id"
            elif not relation or not value:
                reason = "missing_relation_or_value"
            elif is_fragment_like(value, allow_single_word=True):
                reason = "fragment_like_value"
            elif not surface_supported(value, self._claim_support_texts(claim, source_by_id)):
                reason = "value_not_grounded"
            if reason:
                discarded.append(
                    {
                        "kind": "facet",
                        "value": facet.value,
                        "relation": facet.relation,
                        "source_claim_id": facet.source_claim_id,
                        "reason": reason,
                    }
                )
                continue
            entity = normalized_surface(facet.entity or "")
            if entity and is_fragment_like(entity, allow_single_word=True):
                entity = ""
            value_span = normalized_surface(facet.value_span or value)
            if value_span and (
                is_fragment_like(value_span, allow_single_word=True)
                or not surface_supported(value_span, self._claim_support_texts(claim, source_by_id))
            ):
                value_span = value
            facets_by_claim[claim.claim_id].append(
                {
                    "entity": entity,
                    "relation": relation,
                    "value": value,
                    "value_span": value_span,
                    "source": ",".join(facet.source_message_ids or claim.source_message_ids_json),
                    "source_claim_id": claim.claim_id,
                }
            )

        display_sources = {
            "items": result.display_items,
            "named_entities": result.display_named_entities,
            "counts": result.display_counts,
        }
        for field_name, values in display_sources.items():
            for value in values:
                text = normalized_surface(value)
                claim = support_claim_for_value(text)
                if not text or claim is None or is_fragment_like(text, allow_single_word=True):
                    discarded.append(
                        {
                            "kind": f"display_{field_name}",
                            "surface": str(value),
                            "reason": "invalid_or_ungrounded",
                        }
                    )
                    continue
                display_by_claim[claim.claim_id][field_name].append(text)
        for value in result.display_key_facts:
            text = normalized_surface(value)
            claim = support_claim_for_value(text)
            if not text or not is_readable_claim_text(text):
                discarded.append({"kind": "display_key_fact", "surface": str(value), "reason": "unreadable"})
                continue
            if claim is None:
                claim = next((candidate for candidate in claims if surface_supported(candidate.text, [text])), None)
            if claim is None:
                discarded.append({"kind": "display_key_fact", "surface": str(value), "reason": "ungrounded"})
                continue
            display_by_claim[claim.claim_id]["key_facts"].append(text)

        kept_count = 0
        for claim in target_claims:
            metadata = dict(claim.metadata_json or {})
            exact_terms = clean_readable_values(exact_terms_by_claim.get(claim.claim_id, []), allow_single_word=True)
            facets = facets_by_claim.get(claim.claim_id, [])
            display = {
                field: clean_readable_values(values, allow_single_word=(field != "key_facts"))
                for field, values in display_by_claim.get(
                    claim.claim_id,
                    {"items": [], "named_entities": [], "counts": [], "key_facts": []},
                ).items()
            }
            metadata.update(
                {
                    "exact_terms_v2": exact_terms,
                    "facets_v2": facets,
                    "display_signals_v1": display,
                    "signal_extraction_source": "llm_validated",
                    "signal_extraction_failed_v1": False,
                    "signal_extraction_discarded_v1": [
                        item
                        for item in discarded
                        if str(item.get("source_claim_id") or claim.claim_id) == claim.claim_id
                    ],
                }
            )
            self._clear_baseline_signal_fields(metadata)
            kept_count += len(exact_terms) + len(facets) + sum(len(values) for values in display.values())
            claim.metadata_json = metadata
        return kept_count, discarded

    @staticmethod
    def _source_surface_record(candidate: MustPreserveCandidate) -> dict[str, object]:
        payload = candidate.to_metadata()
        payload["surface_key"] = surface_key(candidate.surface)
        return payload

    @staticmethod
    def _source_event_record(candidate: MustPreserveCandidate) -> dict[str, object] | None:
        category = str(candidate.category or "").strip()
        if category not in {"event_object", "event_type"}:
            return None
        surface = collapse_whitespace(candidate.surface)
        if not surface:
            return None
        canonical = collapse_whitespace(candidate.canonical or candidate.surface)
        payload: dict[str, object] = {
            "surface": surface,
            "canonical": canonical,
            "category": category,
            "action": collapse_whitespace(candidate.event_action or ""),
            "temporal_expression": collapse_whitespace(candidate.temporal_expression or ""),
            "source_refs": list(candidate.source_refs or []),
            "source_message_ids": list(candidate.source_message_ids or []),
            "rule": candidate.rule or "",
            "confidence": candidate.confidence,
            "raw_surface": collapse_whitespace(candidate.raw_surface or candidate.surface),
            "surface_key": surface_key(canonical or surface),
        }
        return payload

    def _merge_source_surface_terms_into_claims(
        self,
        claims: list[ClaimRecord],
        source_messages: list[Any],
        *,
        entity_lexicon: dict[str, str],
        target_claim_ids: set[str] | None = None,
    ) -> None:
        candidates = [
            candidate
            for candidate in extract_must_preserve_candidates(source_messages, entity_lexicon=entity_lexicon)
            if candidate.confidence == "high"
        ]
        if not candidates:
            return
        item_categories = {
            "activity",
            "book_title",
            "event_object",
            "event_type",
            "food",
            "instrument",
            "painted_object",
            "place",
            "preference_item",
            "recipe",
            "research_topic",
            "symbol",
            "test_type",
        }
        count_categories = {"count"}
        for claim in claims:
            if target_claim_ids is not None and claim.claim_id not in target_claim_ids:
                continue
            claim_sources = {str(value) for value in list(claim.source_message_ids_json or []) if str(value).strip()}
            if not claim_sources:
                continue
            matched = [
                candidate
                for candidate in candidates
                if claim_sources & {str(value) for value in list(candidate.source_message_ids or [])}
            ]
            if not matched:
                continue
            metadata = dict(claim.metadata_json or {})
            existing_records = [
                dict(record)
                for record in list(metadata.get("source_surface_records_v1") or [])
                if isinstance(record, dict)
            ]
            record_by_key = {
                str(record.get("surface_key") or surface_key(record.get("surface") or "")): record
                for record in existing_records
                if str(record.get("surface") or "").strip()
            }
            for candidate in matched:
                record_by_key[surface_key(candidate.surface)] = self._source_surface_record(candidate)
            source_records = list(record_by_key.values())
            event_records_by_key = {
                str(record.get("surface_key") or surface_key(record.get("canonical") or record.get("surface") or "")): dict(record)
                for record in list(metadata.get("source_event_records_v1") or [])
                if isinstance(record, dict) and str(record.get("surface") or "").strip()
            }
            for candidate in matched:
                event_record = self._source_event_record(candidate)
                if event_record is not None:
                    event_records_by_key[str(event_record["surface_key"])] = event_record
            event_records = list(event_records_by_key.values())
            event_object_terms = clean_readable_values(
                [str(record.get("surface") or "") for record in event_records],
                allow_single_word=True,
            )
            event_canonical_terms = clean_readable_values(
                [str(record.get("canonical") or "") for record in event_records],
                allow_single_word=True,
            )
            event_action_terms = clean_readable_values(
                [str(record.get("action") or "") for record in event_records],
                allow_single_word=True,
            )
            temporal_relation_terms = clean_readable_values(
                [str(record.get("temporal_expression") or "") for record in event_records],
                allow_single_word=False,
            )
            event_refs = list(
                dict.fromkeys(
                    str(source_ref)
                    for record in event_records
                    for source_ref in list(record.get("source_refs") or [])
                    if str(source_ref).strip()
                )
            )
            source_terms = clean_readable_values(
                [str(record.get("surface") or "") for record in source_records],
                allow_single_word=True,
            )
            raw_terms = clean_readable_values(
                [str(record.get("raw_surface") or "") for record in source_records],
                allow_single_word=True,
            )
            exact_terms = clean_readable_values(
                [
                    *list(metadata.get("exact_terms_v2") or metadata.get("exact_terms_v1") or []),
                    *raw_terms,
                    *source_terms,
                ],
                allow_single_word=True,
            )
            display = dict(metadata.get("display_signals_v1") or {})
            display_items = list(display.get("items") or [])
            display_counts = list(display.get("counts") or [])
            display_key_facts = list(display.get("key_facts") or [])
            for candidate in matched:
                category = str(candidate.category or "").casefold()
                display_surface = candidate.raw_surface or candidate.surface
                if category in count_categories:
                    display_counts.append(display_surface)
                elif category in item_categories:
                    display_items.append(display_surface)
                if not surface_supported(candidate.surface, [claim.text]):
                    display_key_facts.append(self._preservation_candidate_fallback_text(candidate, []))
            display["items"] = clean_readable_values(display_items, allow_single_word=True)
            display["counts"] = clean_readable_values(display_counts, allow_single_word=True)
            display["key_facts"] = clean_readable_values(display_key_facts, allow_single_word=False)
            metadata.update(
                {
                    "source_surface_terms_v1": source_terms,
                    "source_surface_raw_terms_v1": raw_terms,
                    "source_surface_records_v1": source_records,
                    "source_surface_categories_v1": list(
                        dict.fromkeys(
                            str(record.get("category") or "")
                            for record in source_records
                            if str(record.get("category") or "").strip()
                        )
                    ),
                    "source_surface_refs_v1": list(
                        dict.fromkeys(
                            str(source_ref)
                            for record in source_records
                            for source_ref in list(record.get("source_refs") or [])
                            if str(source_ref).strip()
                        )
                    ),
                    "source_event_records_v1": event_records,
                    "source_event_object_terms_v1": event_object_terms,
                    "source_event_action_terms_v1": event_action_terms,
                    "source_temporal_relation_terms_v1": temporal_relation_terms,
                    "source_event_canonical_terms_v1": event_canonical_terms,
                    "source_event_refs_v1": event_refs,
                    "exact_terms_v2": exact_terms,
                    "display_signals_v1": display,
                    "source_surface_merge_used_v1": True,
                }
            )
            self._clear_baseline_signal_fields(metadata)
            claim.metadata_json = metadata

    @staticmethod
    def _remap_claim_signal_result(
        result: ClaimSignalExtractionResult,
        claim_id_map: dict[str, str],
    ) -> ClaimSignalExtractionResult:
        return ClaimSignalExtractionResult(
            exact_terms=[
                ClaimSignalExactTerm(
                    surface=term.surface,
                    category=term.category,
                    source_claim_id=claim_id_map.get(term.source_claim_id, term.source_claim_id),
                    source_message_ids=list(term.source_message_ids),
                )
                for term in result.exact_terms
            ],
            facets=[
                ClaimSignalFacet(
                    relation=facet.relation,
                    value=facet.value,
                    entity=facet.entity,
                    value_span=facet.value_span,
                    source_claim_id=claim_id_map.get(facet.source_claim_id, facet.source_claim_id),
                    source_message_ids=list(facet.source_message_ids),
                )
                for facet in result.facets
            ],
            display_items=list(result.display_items),
            display_named_entities=list(result.display_named_entities),
            display_counts=list(result.display_counts),
            display_key_facts=list(result.display_key_facts),
        )

    def _apply_claim_signal_llm_stage(
        self,
        sample_id: str,
        claims: list[ClaimRecord],
        source_by_id: dict[str, Any],
        *,
        structured_first_pass: StructuredFirstPassAttempt | None = None,
        first_attempt: PrecomputedGenerationAttempt | None = None,
        claim_id_map: dict[str, str] | None = None,
        target_claim_ids: set[str] | None = None,
    ) -> None:
        task = "claim_signal_extract"
        self._trace(f"sample={sample_id} claim_signal_extract_start claims={len(claims)}")
        started_at = time.perf_counter()
        try:
            if structured_first_pass is not None:
                self._record_structured_attempt(task, structured_first_pass.vendor)
                if structured_first_pass.response is None:
                    vendor = structured_first_pass.vendor
                    self._record_structured_fallback(task, vendor)
                    failure = structured_first_pass.error or StructuredOutputError(
                        "Structured claim signal request failed before producing a response.",
                        vendor=vendor,
                        strategy="deterministic_fallback",
                    )
                    raise failure
                response = structured_first_pass.response
                vendor = str(response.metadata.get("structured_vendor") or structured_first_pass.vendor)
                self._record_structured_success(task, vendor)
                result = response.parsed
            elif first_attempt is not None:
                result = parse_claim_signal_extraction(first_attempt.text)
            elif self.llm_provider.supports_structured(task):
                vendor = self._structured_vendor()
                self._record_structured_attempt(task, vendor)
                response = self.llm_provider.generate_structured(
                    [
                        NormalizedMessage(
                            role="user",
                            content=self._build_claim_signal_extract_prompt(
                                claims=claims,
                                source_by_id=source_by_id,
                                structured=True,
                            ),
                            turn_index=0,
                        )
                    ],
                    spec=get_structured_task_spec(task),
                    metadata={"task": task, "memory_type": "episodic"},
                )
                vendor = str(response.metadata.get("structured_vendor") or vendor)
                self._record_structured_success(task, vendor)
                result = response.parsed
            else:
                response = self.llm_provider.generate(
                    [
                        NormalizedMessage(
                            role="user",
                            content=self._build_claim_signal_extract_prompt(
                                claims=claims,
                                source_by_id=source_by_id,
                            ),
                            turn_index=0,
                        )
                    ],
                    metadata={"task": task, "memory_type": "episodic"},
                )
                result = parse_claim_signal_extraction(response.text)
            if claim_id_map:
                result = self._remap_claim_signal_result(result, claim_id_map)
            kept_count, discarded = self._apply_claim_signal_result(
                result,
                claims,
                source_by_id,
                target_claim_ids=target_claim_ids,
            )
            self._trace(
                f"sample={sample_id} signal_validation_done kept={kept_count} discarded={len(discarded)}"
            )
            self._trace(
                f"sample={sample_id} claim_signal_extract_done latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
            )
        except Exception as exc:  # noqa: BLE001
            self._apply_deterministic_signal_fallback(claims, target_claim_ids=target_claim_ids)
            vendor = structured_first_pass.vendor if structured_first_pass is not None else self._structured_vendor()
            strategy = (
                structured_first_pass.strategy
                if structured_first_pass is not None and structured_first_pass.strategy
                else (exc.strategy if isinstance(exc, StructuredOutputError) and exc.strategy else structured_strategy_for_vendor(vendor))
            )
            exchange_text = (
                f" exchange={structured_first_pass.exchange_number}"
                if structured_first_pass is not None and structured_first_pass.exchange_number is not None
                else ""
            )
            batch_text = " from_precomputed_batch=true" if structured_first_pass is not None else ""
            self._trace(
                f"sample={sample_id}{exchange_text} claim_signal_extract_failed error={exc.__class__.__name__} "
                f"vendor={vendor} strategy={strategy}{batch_text} fallback=deterministic "
                f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
            )

    def _enrich_episodic_claim_facets(
        self,
        sample_id: str,
        claims: list[ClaimRecord],
        *,
        structured_first_pass: StructuredFirstPassAttempt | None = None,
        first_attempt: PrecomputedGenerationAttempt | None = None,
        claim_id_map: dict[str, str] | None = None,
        target_claim_ids: set[str] | None = None,
    ) -> None:
        if not claims:
            return
        source_ids = [
            source_id
            for claim in claims
            for source_id in claim.source_message_ids_json
        ]
        source_messages = self.store.fetch_raw_messages(source_ids)
        source_by_id = {message.id: message for message in source_messages}
        entity_lexicon = self._sample_entity_lexicon(sample_id)
        assign_claim_metadata_v1(claims, source_by_id, entity_lexicon)
        self._apply_claim_signal_llm_stage(
            sample_id,
            claims,
            source_by_id,
            structured_first_pass=structured_first_pass,
            first_attempt=first_attempt,
            claim_id_map=claim_id_map,
            target_claim_ids=target_claim_ids,
        )
        for claim in claims:
            metadata = dict(claim.metadata_json or {})
            if "exact_terms_v2" in metadata or "facets_v2" in metadata:
                self._clear_baseline_signal_fields(metadata)
                claim.metadata_json = metadata
        self._merge_source_surface_terms_into_claims(
            claims,
            source_messages,
            entity_lexicon=entity_lexicon,
            target_claim_ids=target_claim_ids,
        )

    def _update_trajectory_entity_facet_summary(self, sample_id: str, trajectory, claims: list[ClaimRecord]) -> None:
        entity_lexicon = self._sample_entity_lexicon(sample_id)
        source_ids = [
            source_id
            for claim in claims
            for source_id in claim.source_message_ids_json
        ]
        source_messages = self.store.fetch_raw_messages(source_ids)
        summary = build_trajectory_entity_facet_summary(
            claims,
            {message.id: message for message in source_messages},
            entity_lexicon,
        )
        preservation_misses: list[dict[str, object]] = []
        seen_miss_keys: set[tuple[str, str]] = set()
        for claim in claims:
            for miss in list((claim.metadata_json or {}).get("claim_preservation_misses_v1") or []):
                if not isinstance(miss, dict):
                    continue
                key = (str(miss.get("surface") or ""), str(miss.get("category") or ""))
                if key in seen_miss_keys:
                    continue
                seen_miss_keys.add(key)
                preservation_misses.append(dict(miss))
        trajectory.metadata_json = {
            **dict(trajectory.metadata_json or {}),
            **summary,
            "claim_preservation_misses_v1": preservation_misses,
        }

    def _build_trajectory_retrieval_summary_prompt(
        self,
        *,
        trajectory,
        active_claims: list[ClaimRecord],
        uncertain_claims: list[ClaimRecord],
        recent_snapshot_notes: list[str],
    ) -> str:
        metadata = dict(trajectory.metadata_json or {})
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
        supporting_state = build_summary_supporting_state(
            trajectory_label=str(trajectory.label or ""),
            active_claims=[claim.text for claim in active_claims],
            uncertain_claims=[claim.text for claim in uncertain_claims],
            exact_terms=exact_terms,
            facet_values=facet_values,
            entity_mentions=entity_mentions,
            recent_snapshot_notes=recent_snapshot_notes,
            display_items=display_items,
            display_named_entities=display_named_entities,
            display_counts=display_counts,
            display_key_facts=display_key_facts,
        )
        fallback_summary = build_deterministic_retrieval_summary(
            trajectory_label=str(trajectory.label or ""),
            active_claims=[claim.text for claim in active_claims],
            uncertain_claims=[claim.text for claim in uncertain_claims],
            exact_terms=exact_terms,
            facet_values=facet_values,
            entity_mentions=entity_mentions,
            recent_snapshot_notes=recent_snapshot_notes,
            display_items=display_items,
            display_named_entities=display_named_entities,
            display_counts=display_counts,
            display_key_facts=display_key_facts,
        )
        return (
            load_prompt("trajectory_retrieval_summary")
            + "\n\nTrajectory id:\n"
            + trajectory.id
            + "\n\nGrounded internal state:\n"
            + supporting_state
            + "\n\nDeterministic fallback draft:\n"
            + fallback_summary
        )

    @staticmethod
    def _summary_item_fragment_count(summary_text: str) -> int:
        lines = summary_text.splitlines()
        in_items = False
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                in_items = stripped == "## Item Sets / Named Entities"
                continue
            if not in_items or not stripped.startswith("- "):
                continue
            value = stripped[2:].strip()
            if is_fragment_like(value, allow_single_word=True):
                count += 1
        return count

    def _refresh_trajectory_retrieval_summary(self, trajectory_id: str, *, closed: bool = False) -> None:
        trajectory = self.store.get_trajectory(trajectory_id)
        if trajectory is None:
            return
        snapshots = self.store.list_snapshots(trajectory_id)
        if not snapshots:
            return
        latest_snapshot = snapshots[-1]
        latest_claims = self.store.list_claims_for_snapshot(latest_snapshot.id)
        active_claims = [claim for claim in latest_claims if claim.status == "active"]
        uncertain_claims = [
            claim for claim in latest_claims if claim.status in {"contradictory", "needs-confirmation"}
        ]
        note_snapshots = [
            snapshot
            for snapshot in snapshots
            if not bool((snapshot.metadata_json or {}).get("low_salience_memory_v1"))
        ] or snapshots
        recent_snapshot_notes = [
            f"v{snapshot.version}: {snapshot.summary_content}. {snapshot.context}"
            for snapshot in note_snapshots[-3:]
        ]
        fallback_summary = build_deterministic_retrieval_summary(
            trajectory_label=str(trajectory.label or ""),
            active_claims=[claim.text for claim in active_claims],
            uncertain_claims=[claim.text for claim in uncertain_claims],
            exact_terms=[
                str(term).strip()
                for term in list((trajectory.metadata_json or {}).get("exact_terms") or [])
                if str(term).strip()
            ],
            facet_values=[
                str(value).strip()
                for value in list((trajectory.metadata_json or {}).get("facet_values") or [])
                if str(value).strip()
            ],
            entity_mentions=[
                str(value).strip()
                for value in list((trajectory.metadata_json or {}).get("entity_mentions") or [])
                if str(value).strip()
            ],
            recent_snapshot_notes=recent_snapshot_notes,
            display_items=[
                str(value).strip()
                for value in list((trajectory.metadata_json or {}).get("display_items") or [])
                if str(value).strip()
            ],
            display_named_entities=[
                str(value).strip()
                for value in list((trajectory.metadata_json or {}).get("display_named_entities") or [])
                if str(value).strip()
            ],
            display_counts=[
                str(value).strip()
                for value in list((trajectory.metadata_json or {}).get("display_counts") or [])
                if str(value).strip()
            ],
            display_key_facts=[
                str(value).strip()
                for value in list((trajectory.metadata_json or {}).get("display_key_facts") or [])
                if str(value).strip()
            ],
        )
        summary_text = fallback_summary
        generation_metadata: dict[str, Any] = {
            "used_fallback": True,
            "closed_refresh": closed,
        }
        try:
            response = self.llm_provider.generate(
                [
                    NormalizedMessage(
                        role="user",
                        content=self._build_trajectory_retrieval_summary_prompt(
                            trajectory=trajectory,
                            active_claims=active_claims,
                            uncertain_claims=uncertain_claims,
                            recent_snapshot_notes=recent_snapshot_notes,
                        ),
                        turn_index=0,
                    )
                ],
                metadata={"task": "trajectory_retrieval_summary", "memory_type": "episodic"},
            )
            generated = response.text.strip()
            if generated and "## Profile / Stable Facts" in generated and "## Conflicts / Uncertainty" in generated:
                fragment_count = self._summary_item_fragment_count(generated)
                if fragment_count:
                    self._trace(
                        f"trajectory={trajectory_id} summary_readability_invalid fragments={fragment_count} fallback=cleaned"
                    )
                    generation_metadata = {
                        "used_fallback": True,
                        "closed_refresh": closed,
                        "invalid_generated_fragment_count": fragment_count,
                    }
                else:
                    summary_text = generated
                    generation_metadata = {
                        "used_fallback": False,
                        "closed_refresh": closed,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    }
        except Exception as exc:  # pragma: no cover - defensive fallback
            generation_metadata["error"] = type(exc).__name__
        summary_embedding = self.store.save_embedding(
            embedding_id=f"{trajectory_id}-summary",
            owner_type="trajectory_summary",
            owner_id=trajectory_id,
            model_name=self.embedding_provider.model_info().model_name,
            vector=self._embed_documents([summary_text])[0],
            semantic_text=summary_text,
            metadata={
                "memory_type": "episodic",
                "document_embedding_strategy": self._document_embedding_strategy(),
                **generation_metadata,
            },
        )
        metadata = dict(trajectory.metadata_json or {})
        exact_terms = [str(term).strip() for term in list(metadata.get("exact_terms") or []) if str(term).strip()]
        facet_tags = [str(value).strip() for value in list(metadata.get("facet_tags") or []) if str(value).strip()]
        facet_values = [str(value).strip() for value in list(metadata.get("facet_values") or []) if str(value).strip()]
        entity_mentions = [
            str(value).strip() for value in list(metadata.get("entity_mentions") or []) if str(value).strip()
        ]
        display_items = [str(value).strip() for value in list(metadata.get("display_items") or []) if str(value).strip()]
        display_named_entities = [
            str(value).strip() for value in list(metadata.get("display_named_entities") or []) if str(value).strip()
        ]
        display_counts = [str(value).strip() for value in list(metadata.get("display_counts") or []) if str(value).strip()]
        display_key_facts = [
            str(value).strip() for value in list(metadata.get("display_key_facts") or []) if str(value).strip()
        ]
        latest_embedding_id = str(latest_snapshot.embedding_ref or "")
        if not latest_embedding_id:
            latest_embedding = self.store.snapshot_embedding(latest_snapshot.id)
            if latest_embedding is not None:
                latest_embedding_id = latest_embedding.id
        active_source_ids = [
            source_id
            for claim in active_claims
            for source_id in list(claim.source_message_ids_json or [])
            if str(source_id).strip()
        ]
        source_anchors = [
            {
                "source_ref": str(message.source_ref or message.id),
                "text": collapse_whitespace(message.content),
            }
            for message in self.store.fetch_raw_messages(active_source_ids)
            if collapse_whitespace(str(message.content or ""))
        ]
        summary_metadata = {
            **metadata,
            "exact_terms": exact_terms,
            "facet_tags": facet_tags,
            "facet_values": facet_values,
            "entity_mentions": entity_mentions,
            "display_items": display_items,
            "display_named_entities": display_named_entities,
            "display_counts": display_counts,
            "display_key_facts": display_key_facts,
        }
        summary_keyword_list = summary_keywords_v2(summary_text, summary_metadata)
        removed_summary_keywords = removed_internal_summary_keywords(summary_text)
        evidence_card = build_trajectory_historical_evidence_card(
            trajectory_id=trajectory.id,
            trajectory_label=str(trajectory.label or ""),
            retrieval_summary_text=summary_text,
            latest_semantic_text=getattr(latest_snapshot, "semantic_text", ""),
            metadata={
                **summary_metadata,
                "exact_terms": exact_terms,
                "retrieval_summary_keywords": summary_keyword_list,
                "retrieval_summary_keywords_v2": summary_keyword_list,
            },
            active_claim_texts=[claim.text for claim in active_claims],
            source_anchors=source_anchors,
        )
        self._update_trajectory_match_state(
            trajectory,
            snapshot_id_value=latest_snapshot.id,
            snapshot_embedding_id=latest_embedding_id,
            semantic_text=getattr(latest_snapshot, "semantic_text", ""),
            metadata_updates={
                "retrieval_summary_text": summary_text,
                "retrieval_summary_keywords": summary_keyword_list,
                "retrieval_summary_keywords_v2": summary_keyword_list,
                "retrieval_summary_keyword_policy": "retrieval_summary_keywords_v2",
                "retrieval_summary_keywords_removed_internal_terms": removed_summary_keywords,
                "retrieval_summary_updated_at_version": latest_snapshot.version,
                "retrieval_summary_embedding_id": summary_embedding.id,
                "exact_terms": exact_terms,
                "facet_tags": facet_tags,
                "facet_values": facet_values,
                "entity_mentions": entity_mentions,
                "display_items": display_items,
                "display_named_entities": display_named_entities,
                "display_counts": display_counts,
                "display_key_facts": display_key_facts,
                "retrieval_summary_item_fragment_count": self._summary_item_fragment_count(summary_text),
                "trajectory_identity_summary_v1": evidence_card["identity_summary"],
                "trajectory_recent_update_v1": evidence_card["recent_update"],
                "trajectory_historical_item_terms_v1": list(evidence_card["historical_item_terms"]),
                "trajectory_historical_item_terms_v2": list(evidence_card["historical_item_terms"]),
                "trajectory_source_anchors_v1": list(evidence_card["source_anchors"]),
                "trajectory_drift_cluster_keys_v1": list(evidence_card["drift_cluster_keys"]),
                "trajectory_drift_cluster_count_v1": len(evidence_card["drift_cluster_keys"]),
                "trajectory_historical_evidence_card_v1": evidence_card,
            },
        )

    def _trace(self, message: str) -> None:
        if self.trace is not None:
            self.trace(message)

    def begin_exchange(
        self,
        sample_id: str,
        exchange_messages: list[NormalizedMessage],
    ) -> tuple[str, float]:
        self._exchange_counters[sample_id] += 1
        exchange_index = self._exchange_counters[sample_id]
        exchange_label = f"sample={sample_id} exchange={exchange_index}"
        exchange_started = time.perf_counter()
        self._trace(
            f"{exchange_label} start turns={[message.turn_index for message in exchange_messages]} "
            f"roles={[message.role for message in exchange_messages]}"
        )
        return exchange_label, exchange_started

    def complete_exchange(self, exchange_label: str, exchange_started: float) -> None:
        self._trace(
            f"{exchange_label} complete latency_ms={(time.perf_counter() - exchange_started) * 1000.0:.1f}"
        )

    def persist_memory(
        self,
        sample_id: str,
        dataset_name: str,
        parsed: ParsedMemory,
        *,
        exchange_messages: list[NormalizedMessage] | None = None,
    ) -> bool:
        if exchange_messages:
            parsed = self._apply_claim_text_llm_stage(sample_id, parsed, exchange_messages)
            parsed = self._apply_claim_preservation_audit(sample_id, parsed, exchange_messages)
        if self._apply_zero_claim_episodic_policy(sample_id, parsed):
            return False
        self._persist_memory(sample_id, dataset_name, parsed, exchange_messages=exchange_messages)
        return True

    def _apply_zero_claim_episodic_policy(self, sample_id: str, parsed: ParsedMemory) -> bool:
        """Return True when a zero-claim episodic seed should not be persisted."""

        if parsed.claims:
            return False
        self.zero_claim_episodic_candidate_count += 1
        source = str(parsed.metadata.get("episodic_seed_source_v1") or "unknown")
        reason = str(
            parsed.metadata.get("forced_episodic_seed_reason_v1")
            or parsed.metadata.get("llm_no_memory_reason_v1")
            or "none"
        )
        parsed.metadata.update(
            {
                "zero_claim_episodic_memory_v1": True,
                "zero_claim_episodic_source_v1": source,
                "zero_claim_episodic_reason_v1": reason,
            }
        )
        if bool(parsed.metadata.get("low_salience_memory_v1")):
            self.zero_claim_low_salience_skipped_count += 1
            parsed.metadata.update(
                {
                    "zero_claim_episodic_skipped_v1": True,
                    "zero_claim_episodic_skip_reason_v1": "zero_claim_low_salience",
                }
            )
            self._trace(
                f"sample={sample_id} zero_claim_low_salience_skip source={source} "
                f"reason={reason} claims=0"
            )
            return True
        self.zero_claim_episodic_persisted_count += 1
        parsed.metadata["zero_claim_episodic_persisted_v1"] = True
        self._trace(
            f"sample={sample_id} zero_claim_episodic_persisted low_salience=false "
            f"source={source} claims=0"
        )
        return False

    def _assert_claim_preservation_audited(
        self,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage] | None,
    ) -> None:
        if exchange_messages and parsed.claims and "claim_preservation_candidate_count_v1" not in parsed.metadata:
            raise ValueError("persist_audited_memory requires claim preservation audit metadata")

    def persist_audited_memory(
        self,
        sample_id: str,
        dataset_name: str,
        parsed: ParsedMemory,
        *,
        exchange_messages: list[NormalizedMessage] | None = None,
        claim_signal_structured_first_pass: StructuredFirstPassAttempt | None = None,
        claim_signal_first_attempt: PrecomputedGenerationAttempt | None = None,
    ) -> bool:
        self._assert_claim_preservation_audited(parsed, exchange_messages)
        if self._apply_zero_claim_episodic_policy(sample_id, parsed):
            return False
        self._persist_memory(
            sample_id,
            dataset_name,
            parsed,
            exchange_messages=exchange_messages,
            claim_signal_structured_first_pass=claim_signal_structured_first_pass,
            claim_signal_first_attempt=claim_signal_first_attempt,
        )
        return True

    def persist_preprocessed_memory(
        self,
        sample_id: str,
        dataset_name: str,
        parsed: ParsedMemory,
        *,
        exchange_messages: list[NormalizedMessage] | None = None,
        claim_signal_structured_first_pass: StructuredFirstPassAttempt | None = None,
        claim_signal_first_attempt: PrecomputedGenerationAttempt | None = None,
    ) -> None:
        self.persist_audited_memory(
            sample_id,
            dataset_name,
            parsed,
            exchange_messages=exchange_messages,
            claim_signal_structured_first_pass=claim_signal_structured_first_pass,
            claim_signal_first_attempt=claim_signal_first_attempt,
        )

    @staticmethod
    def _preservation_candidate_metadata(candidates: Iterable[MustPreserveCandidate]) -> list[dict[str, object]]:
        return [candidate.to_metadata() for candidate in candidates]

    @staticmethod
    def _preservation_candidates_from_weak_links(weak_links: Iterable[dict[str, object]]) -> list[MustPreserveCandidate]:
        candidates: list[MustPreserveCandidate] = []
        for weak_link in weak_links:
            payload = weak_link.get("candidate") if isinstance(weak_link, dict) else None
            if not isinstance(payload, dict):
                continue
            surface = str(payload.get("surface") or "").strip()
            category = str(payload.get("category") or "exact_term").strip()
            source_ids = [
                str(value)
                for value in list(payload.get("source_message_ids") or [])
                if str(value).strip()
            ]
            if not surface or not source_ids:
                continue
            candidates.append(
                MustPreserveCandidate(
                    surface=surface,
                    category=category,
                    source_message_ids=source_ids,
                    relation=(str(payload.get("relation")).strip() if payload.get("relation") else None),
                    confidence=str(payload.get("confidence") or "high"),
                    source_refs=[
                        str(value)
                        for value in list(payload.get("source_refs") or [])
                        if str(value).strip()
                    ],
                    rule=(str(payload.get("rule")).strip() if payload.get("rule") else None),
                    raw_surface=(
                        str(payload.get("raw_surface")).strip() if payload.get("raw_surface") else None
                    ),
                    canonical=(str(payload.get("canonical")).strip() if payload.get("canonical") else None),
                    event_action=(str(payload.get("action")).strip() if payload.get("action") else None),
                    temporal_expression=(
                        str(payload.get("temporal_expression")).strip()
                        if payload.get("temporal_expression")
                        else None
                    ),
                )
            )
        return candidates

    def _build_claim_preservation_repair_prompt(
        self,
        *,
        exchange_messages: list[NormalizedMessage],
        parsed: ParsedMemory,
        missing_candidates: list[MustPreserveCandidate],
    ) -> str:
        candidate_lines = []
        for index, candidate in enumerate(missing_candidates, start=1):
            relation = f" | relation={candidate.relation}" if candidate.relation else ""
            extras: list[str] = []
            if candidate.temporal_expression:
                extras.append(f"temporal_expression={candidate.temporal_expression}")
            if candidate.raw_surface:
                extras.append(f"raw_surface={candidate.raw_surface}")
            if candidate.source_refs:
                extras.append(f"source_refs={', '.join(candidate.source_refs)}")
            if candidate.rule:
                extras.append(f"rule={candidate.rule}")
            extra_text = f" | {' | '.join(extras)}" if extras else ""
            candidate_lines.append(
                f"- C{index}: surface={candidate.surface} | category={candidate.category}{relation} "
                f"| source_message_ids={', '.join(candidate.source_message_ids)}{extra_text}"
            )
        current_dsl = parsed.raw.raw_text or render_episodic_memory(parsed.raw)
        return (
            load_prompt("episodic_claim_preservation_repair")
            + "\n\nConversation:\n"
            + self._render_conversation(exchange_messages)
            + "\n\nCurrent episodic DSL:\n"
            + current_dsl
            + "\n\nMissing must-preserve candidates:\n"
            + ("\n".join(candidate_lines) if candidate_lines else "- none")
        )

    def _repair_claim_preservation(
        self,
        *,
        sample_id: str,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
        valid_link_ids: set[str],
        missing_candidates: list[MustPreserveCandidate],
    ) -> EpisodicMemoryInput | None:
        prompt = self._build_claim_preservation_repair_prompt(
            exchange_messages=exchange_messages,
            parsed=parsed,
            missing_candidates=missing_candidates,
        )
        response = self.llm_provider.generate(
            [NormalizedMessage(role="user", content=prompt, turn_index=0)],
            metadata={"task": "claim_preservation_repair", "memory_type": "episodic"},
        )
        normalized_text, _ = normalize_memory_text(response.text)
        exchange_link_ids = [message.raw_message_id for message in exchange_messages if message.raw_message_id]
        exchange_timestamp = derive_exchange_timestamp(exchange_messages)
        return parse_episodic_memory(
            normalized_text,
            valid_link_ids,
            exchange_link_ids=exchange_link_ids,
            exchange_timestamp=exchange_timestamp,
            parse_claims=True,
        )

    def _apply_claim_preservation_audit(
        self,
        sample_id: str,
        parsed: ParsedMemory,
        exchange_messages: list[NormalizedMessage],
    ) -> ParsedMemory:
        source_messages = raw_records_from_normalized(exchange_messages)
        entity_lexicon = self._sample_entity_lexicon(sample_id)
        candidates = extract_must_preserve_candidates(source_messages, entity_lexicon=entity_lexicon)
        initial_result = audit_claim_preservation(
            candidates=candidates,
            claims=parsed.claims,
            source_messages=source_messages,
            entity_lexicon=entity_lexicon,
        )
        self._trace(
            f"sample={sample_id} claim_preservation_audit candidates={len(candidates)} "
            f"missing={len(initial_result.missing_candidates)}"
        )
        metadata_updates: dict[str, Any] = {
            "claim_preservation_candidate_count_v1": len(candidates),
            "claim_preservation_missing_count_v1": len(initial_result.missing_candidates),
            "claim_preservation_misses_v1": self._preservation_candidate_metadata(initial_result.missing_candidates),
            "claim_preservation_repair_used_v1": False,
            "claim_preservation_repair_succeeded_v1": False,
            "claim_preservation_repair_failed_v1": False,
            "claim_preservation_weak_source_links_v1": list(initial_result.weak_source_links),
            "claim_preservation_diagnostics_v1": list(initial_result.diagnostics),
        }
        repair_candidates = [
            *initial_result.missing_candidates,
            *self._preservation_candidates_from_weak_links(initial_result.weak_source_links),
        ]
        if not repair_candidates:
            parsed.metadata.update(metadata_updates)
            return parsed
        repaired_raw: EpisodicMemoryInput | None = None
        repair_error: str | None = None
        try:
            repaired_raw = self._repair_claim_preservation(
                sample_id=sample_id,
                parsed=parsed,
                exchange_messages=exchange_messages,
                valid_link_ids=self.store.list_raw_message_ids(sample_id),
                missing_candidates=repair_candidates,
            )
        except Exception as exc:  # noqa: BLE001
            repair_error = str(exc)
        if repaired_raw is None:
            metadata_updates.update(
                {
                    "claim_preservation_repair_used_v1": True,
                    "claim_preservation_repair_failed_v1": True,
                    "claim_preservation_repair_error_v1": repair_error,
                    "claim_preservation_fallback_used_v1": True,
                }
            )
            parsed.metadata.update(metadata_updates)
            parsed = self._apply_preservation_candidate_fallback(
                parsed,
                repair_candidates,
                exchange_messages,
                metadata_field="claim_preservation_fallback_used_v1",
            )
            self._trace(
                f"sample={sample_id} claim_preservation_repair used=true remaining_missing={len(initial_result.missing_candidates)}"
            )
            return parsed
        repaired_result = audit_claim_preservation(
            candidates=candidates,
            claims=repaired_raw.claims,
            source_messages=source_messages,
            entity_lexicon=entity_lexicon,
        )
        repair_succeeded = not repaired_result.missing_candidates and not repaired_result.weak_source_links
        metadata_updates.update(
            {
                "claim_preservation_missing_count_v1": len(repaired_result.missing_candidates),
                "claim_preservation_misses_v1": self._preservation_candidate_metadata(repaired_result.missing_candidates),
                "claim_preservation_repair_used_v1": True,
                "claim_preservation_repair_succeeded_v1": repair_succeeded,
                "claim_preservation_repair_failed_v1": not repair_succeeded,
                "claim_preservation_fallback_used_v1": False,
                "claim_preservation_weak_source_links_v1": list(repaired_result.weak_source_links),
                "claim_preservation_diagnostics_v1": list(repaired_result.diagnostics),
                "claim_preservation_initial_misses_v1": self._preservation_candidate_metadata(
                    initial_result.missing_candidates
                ),
            }
        )
        parsed.metadata.update(metadata_updates)
        if repair_succeeded:
            parsed.raw = repaired_raw
            parsed.claims = repaired_raw.claims
            parsed.links = repaired_raw.links
            parsed.semantic_text = repaired_raw.semantic_text
            parsed.metadata["normalized_extraction_text"] = repaired_raw.raw_text
        else:
            parsed = self._apply_preservation_candidate_fallback(
                parsed,
                repaired_result.missing_candidates,
                exchange_messages,
                metadata_field="claim_preservation_fallback_used_v1",
            )
        self._trace(
            f"sample={sample_id} claim_preservation_repair used=true remaining_missing={len(repaired_result.missing_candidates)}"
        )
        return parsed

    def process_exchange(
        self,
        sample_id: str,
        dataset_name: str,
        exchange_messages: list[NormalizedMessage],
    ) -> None:
        if not exchange_messages:
            return
        exchange_label, exchange_started = self.begin_exchange(sample_id, exchange_messages)
        valid_link_ids = self.store.list_raw_message_ids(sample_id)
        episodic_started = time.perf_counter()
        episodic = self.extract_episodic(sample_id, exchange_messages, valid_link_ids)
        self._trace(
            f"{exchange_label} episodic_extract done has_memory={episodic is not None} "
            f"latency_ms={(time.perf_counter() - episodic_started) * 1000.0:.1f}"
        )
        if episodic is not None:
            persist_started = time.perf_counter()
            persisted = self.persist_memory(sample_id, dataset_name, episodic, exchange_messages=exchange_messages)
            if persisted:
                self._trace(
                    f"{exchange_label} episodic_persist done latency_ms={(time.perf_counter() - persist_started) * 1000.0:.1f}"
                )
            else:
                self._trace(
                    f"{exchange_label} episodic_persist skipped reason=zero_claim_low_salience "
                    f"latency_ms={(time.perf_counter() - persist_started) * 1000.0:.1f}"
                )
        self.complete_exchange(exchange_label, exchange_started)

    def finalize_trajectory(self, sample_id: str) -> None:
        for trajectory in self.store.list_trajectories(sample_id, open_only=True):
            self._close_trajectory(trajectory.id, close_reason="sample_finalize", refresh_summary=False)

    @staticmethod
    def _render_conversation(exchange_messages: list[NormalizedMessage]) -> str:
        return "\n".join(
            f"{message.raw_message_id or 'unknown'} [{message.role}]: {message.content}"
            for message in exchange_messages
        )

    def _structured_vendor(self) -> str:
        return str(self.llm_provider.model_info().metadata.get("vendor") or "unknown")

    @staticmethod
    def _one_line_error(exc: Exception, *, limit: int = 240) -> str:
        text = " ".join(str(exc).split())
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

    def _structured_first_pass_failure(
        self,
        task: str,
        vendor: str,
        exc: Exception,
        *,
        exchange_number: int | None = None,
    ) -> StructuredFirstPassAttempt:
        strategy = exc.strategy if isinstance(exc, StructuredOutputError) and exc.strategy else structured_strategy_for_vendor(vendor)
        return StructuredFirstPassAttempt(
            task=task,
            vendor=vendor,
            error=exc,
            strategy=strategy,
            exchange_number=exchange_number,
            error_type=exc.__class__.__name__,
            error_message=self._one_line_error(exc),
        )

    def _record_structured_attempt(self, task: str, vendor: str) -> None:
        self.structured_attempts += 1
        self.structured_attempts_by_task[task] += 1
        self.structured_attempts_by_vendor[vendor] += 1

    def _record_structured_success(self, task: str, vendor: str) -> None:
        self.structured_successes += 1
        self.structured_successes_by_task[task] += 1
        self.structured_successes_by_vendor[vendor] += 1

    def _record_structured_fallback(self, task: str, vendor: str) -> None:
        self.structured_fallbacks += 1
        self.structured_fallbacks_by_task[task] += 1
        self.structured_fallbacks_by_vendor[vendor] += 1

    @staticmethod
    def _parser_error_dict(exc: Exception | None) -> dict[str, Any]:
        if not isinstance(exc, ParserValidationError):
            return {}
        return exc.to_dict()

    def _record_parser_diagnostics(
        self,
        *,
        sample_id: str,
        task: str,
        attempt: int,
        diagnostics: list[dict[str, Any]],
    ) -> None:
        for diagnostic in diagnostics:
            kind = str(diagnostic.get("kind") or "link_salvage")
            if kind == "link_salvage":
                self.link_salvage_count += 1
                if diagnostic.get("exchange_link_fallback_used"):
                    self.link_exchange_fallback_count += 1
                self._trace(
                    f"sample={sample_id} task={task} attempt={attempt}/3 link_salvage "
                    f"field={diagnostic.get('field_path')} dropped={diagnostic.get('dropped_ids', [])} "
                    f"kept={diagnostic.get('kept_ids', [])} "
                    f"exchange_fallback={bool(diagnostic.get('exchange_link_fallback_used'))}"
                )
            elif kind == "ops_ignored":
                self.ops_ignored_count += 1
                self._trace(
                    f"sample={sample_id} task={task} attempt={attempt}/3 ops_parse_ignored "
                    f"code={diagnostic.get('code')} field={diagnostic.get('field')} "
                    f"reason={diagnostic.get('reason')}"
                )
            elif kind == "ops_defaulted":
                self._trace(
                    f"sample={sample_id} task={task} attempt={attempt}/3 ops_defaulted "
                    f"field={diagnostic.get('field')} value={diagnostic.get('value')}"
                )

    @staticmethod
    def _diagnostics_of_kind(
        diagnostics: list[dict[str, Any]], kind: str
    ) -> list[dict[str, Any]]:
        return [diagnostic for diagnostic in diagnostics if diagnostic.get("kind") == kind]

    @staticmethod
    def _normalize_claim_text(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    @staticmethod
    def _claim_status_priority(status: str) -> int:
        return {
            "active": 0,
            "deprecated": 1,
            "needs-confirmation": 2,
            "contradictory": 3,
        }.get(status, 0)

    @classmethod
    def _more_conservative_claim_status(cls, left: str, right: str) -> str:
        return left if cls._claim_status_priority(left) >= cls._claim_status_priority(right) else right

    @staticmethod
    def _dedupe_source_ids(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(str(value) for value in values if str(value)))

    def _dedupe_extracted_claims(
        self,
        claims: Iterable[MemoryClaim],
    ) -> tuple[list[MemoryClaim], list[dict[str, Any]]]:
        deduped_by_text: dict[str, MemoryClaim] = {}
        ordered_keys: list[str] = []
        duplicate_diagnostics: list[dict[str, Any]] = []
        for claim in claims:
            normalized_text = self._normalize_claim_text(claim.text)
            if normalized_text not in deduped_by_text:
                deduped_by_text[normalized_text] = claim.copy(
                    update={"source_message_ids": self._dedupe_source_ids(claim.source_message_ids)}
                )
                ordered_keys.append(normalized_text)
                continue

            previous = deduped_by_text[normalized_text]
            merged_sources = self._dedupe_source_ids(
                list(previous.source_message_ids) + list(claim.source_message_ids)
            )
            merged_status = self._more_conservative_claim_status(previous.status, claim.status)
            deduped_by_text[normalized_text] = previous.copy(
                update={"source_message_ids": merged_sources, "status": merged_status}
            )
            duplicate_diagnostics.append(
                {
                    "normalized_text": normalized_text,
                    "kept_claim_id": previous.claim_id,
                    "duplicate_claim_id": claim.claim_id,
                    "kept_status": previous.status,
                    "duplicate_status": claim.status,
                    "merged_status": merged_status,
                    "merged_source_message_ids": merged_sources,
                }
            )
        return [deduped_by_text[key] for key in ordered_keys], duplicate_diagnostics

    def _canonical_claim_view(self, claim: ClaimRecord) -> CanonicalClaimView:
        return CanonicalClaimView(
            claim_id=claim.claim_id,
            text=claim.text,
            normalized_text=self._normalize_claim_text(claim.text),
            status=claim.status,
            source_message_ids=list(claim.source_message_ids_json),
            parent_claim_id=claim.parent_claim_id,
            revised_from_claim_id=claim.revised_from_claim_id,
        )

    @staticmethod
    def _model_op_hint_dict(op: ClaimOp) -> dict[str, Any]:
        return {
            "op": op.op,
            "target_claim_id": op.target_claim_id,
            "new_claim_id": op.new_claim_id,
            "source_message_ids": list(op.source_message_ids),
            "rationale": op.rationale,
            "claim_text": op.claim_text,
            "metadata": dict(op.metadata),
        }

    @staticmethod
    def _claim_keyword_set(text: str) -> set[str]:
        return set(extract_keywords(text))

    @staticmethod
    def _candidate_label_map(prefix: str, resolved_ids: list[str]) -> dict[str, str]:
        return {f"{prefix}{index}": resolved_id for index, resolved_id in enumerate(resolved_ids, start=1)}

    @staticmethod
    def _selected_candidate_label_for_resolved_id(
        resolved_id: str | None,
        candidate_label_map: dict[str, str],
    ) -> str | None:
        if resolved_id is None:
            return None
        for label, candidate_id in candidate_label_map.items():
            if candidate_id == resolved_id:
                return label
        return None

    @staticmethod
    def _selected_candidate_label_from_text(text: str) -> str | None:
        match = re.search(r"^SELECTED_CANDIDATE:\s*(.+)$", text, flags=re.MULTILINE)
        if not match:
            return None
        selected = match.group(1).strip()
        if selected.lower() == "none":
            return None
        return selected

    def _resolve_text_trajectory_match_decision(
        self,
        *,
        sample_id: str,
        parsed: ParsedMemory,
        decision_text: str,
        candidate_label_map: dict[str, str],
        open_count: int,
        prefiltered_count: int,
        shortlist_count: int,
        embedding_latency_ms: float,
        match_started: float,
    ) -> str | None:
        selected_candidate_label = self._selected_candidate_label_from_text(decision_text)
        try:
            decision = parse_match_decision(decision_text, candidate_label_map)
        except ParserValidationError as exc:
            preview = " ".join(decision_text.split())
            if len(preview) > 240:
                preview = preview[:237] + "..."
            parsed.metadata["trajectory_match_selected_candidate_label"] = selected_candidate_label
            parsed.metadata["trajectory_match_selected_candidate_resolved_id"] = None
            parsed.metadata["trajectory_match_text_parse_failed"] = True
            parsed.metadata["trajectory_match_text_parse_error"] = self._one_line_error(exc)
            parsed.metadata["trajectory_match_text_parse_preview"] = preview
            parsed.metadata["trajectory_match_text_parse_fallback"] = "new"
            self._trace(
                f"sample={sample_id} match type=episodic open={open_count} "
                f"prefiltered={prefiltered_count} shortlist={shortlist_count} result=new "
                f"fallback=text_parse_failed error={self._one_line_error(exc, limit=120)} "
                f"embed_latency_ms={embedding_latency_ms:.1f} latency_ms={(time.perf_counter() - match_started) * 1000.0:.1f}"
            )
            return None
        resolved = decision.trajectory_id if decision.decision == "CONTINUE" else None
        parsed.metadata["trajectory_match_selected_candidate_label"] = selected_candidate_label
        parsed.metadata["trajectory_match_selected_candidate_resolved_id"] = resolved
        self._trace(
            f"sample={sample_id} match type=episodic open={open_count} "
            f"prefiltered={prefiltered_count} shortlist={shortlist_count} result={resolved or 'new'} "
            f"embed_latency_ms={embedding_latency_ms:.1f} latency_ms={(time.perf_counter() - match_started) * 1000.0:.1f}"
        )
        return resolved

    def _shortlist_transition_candidates(
        self,
        current_claim: MemoryClaim,
        available_previous: list[tuple[int, CanonicalClaimView]],
    ) -> list[tuple[int, CanonicalClaimView, int]]:
        current_keywords = self._claim_keyword_set(current_claim.text)
        if not current_keywords:
            return []
        ranked: list[tuple[int, CanonicalClaimView, int]] = []
        for index, previous in available_previous:
            overlap = len(current_keywords & self._claim_keyword_set(previous.text))
            if overlap > 0:
                ranked.append((index, previous, overlap))
        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked[:3]

    def _build_trajectory_match_prompt(
        self,
        *,
        memory_type: str,
        semantic_text: str,
        candidates: list[tuple[str, str, str]],
    ) -> str:
        candidate_lines = "\n\n".join(
            f"### {label}\n"
            f"Trajectory summary:\n{summary}\n\n"
            f"Latest update note:\n{latest_update}"
            for label, summary, latest_update in candidates
        )
        return (
            load_prompt("trajectory_match")
            + "\n\nMemory type:\n"
            + memory_type
            + "\n\nNew memory:\n"
            + semantic_text
            + "\n\nCandidates:\n"
            + candidate_lines
        )

    def _build_trajectory_match_structured_prompt(
        self,
        *,
        memory_type: str,
        semantic_text: str,
        candidates: list[tuple[str, str, str]],
    ) -> str:
        candidate_lines = "\n\n".join(
            f"### {label}\n"
            f"Trajectory summary:\n{summary}\n\n"
            f"Latest update note:\n{latest_update}"
            for label, summary, latest_update in candidates
        )
        return (
            load_prompt("trajectory_match_structured")
            + "\n\nMemory type:\n"
            + memory_type
            + "\n\nNew memory:\n"
            + semantic_text
            + "\n\nCandidates:\n"
            + candidate_lines
        )

    def _build_claim_transition_prompt_with_template(
        self,
        prompt_name: str,
        current_claim: MemoryClaim,
        candidates: list[tuple[str, CanonicalClaimView]],
        exchange_text: str,
    ) -> str:
        candidate_lines = "\n".join(
            f"- label={label} | status={candidate.status} | text={candidate.text}"
            for label, candidate in candidates
        )
        return (
            load_prompt(prompt_name)
            + "\n\nCurrent claim:\n"
            + f"status={current_claim.status} | text={current_claim.text}"
            + "\n\nCandidate previous claims:\n"
            + candidate_lines
            + "\n\nCurrent exchange:\n"
            + exchange_text
        )

    def _build_claim_transition_prompt(
        self,
        current_claim: MemoryClaim,
        candidates: list[tuple[str, CanonicalClaimView]],
        exchange_text: str,
    ) -> str:
        return self._build_claim_transition_prompt_with_template(
            "claim_transition_judge",
            current_claim,
            candidates,
            exchange_text,
        )

    def _build_claim_transition_structured_prompt(
        self,
        current_claim: MemoryClaim,
        candidates: list[tuple[str, CanonicalClaimView]],
        exchange_text: str,
    ) -> str:
        return self._build_claim_transition_prompt_with_template(
            structured_prompt_name("claim_transition_judge"),
            current_claim,
            candidates,
            exchange_text,
        )

    def _adjudicate_claim_transition(
        self,
        *,
        sample_id: str,
        current_claim: MemoryClaim,
        candidates: list[CanonicalClaimView],
        exchange_text: str,
    ) -> tuple[ClaimTransitionDecision, bool, str | None, float, dict[str, Any], str | None]:
        task = "claim_transition_judge"
        candidate_label_map = self._candidate_label_map(
            "P",
            [candidate.claim_id for candidate in candidates],
        )
        if len(candidate_label_map) != len(candidates):
            raise ValueError("candidate label map length mismatch")
        labeled_candidates = [
            (label, candidate)
            for label, candidate in zip(candidate_label_map.keys(), candidates, strict=True)
        ]
        prompt = self._build_claim_transition_prompt(current_claim, labeled_candidates, exchange_text)
        started_at = time.perf_counter()
        metadata = {"task": task, "memory_type": "episodic"}
        if self.llm_provider.supports_structured(task):
            vendor = self._structured_vendor()
            self._record_structured_attempt(task, vendor)
            try:
                response = self.llm_provider.generate_structured(
                    [
                        NormalizedMessage(
                            role="user",
                            content=self._build_claim_transition_structured_prompt(
                                current_claim,
                                labeled_candidates,
                                exchange_text,
                            ),
                            turn_index=0,
                        )
                    ],
                    spec=get_structured_task_spec(task),
                    metadata=metadata,
                )
                vendor = str(response.metadata.get("structured_vendor") or vendor)
                self._record_structured_success(task, vendor)
                decision = validate_claim_transition_judge_result(response.parsed, candidate_label_map)
                return (
                    decision,
                    True,
                    None,
                    (time.perf_counter() - started_at) * 1000.0,
                    dict(response.metadata),
                    response.parsed.selected_candidate,
                )
            except Exception as exc:  # noqa: BLE001
                self._record_structured_fallback(task, vendor)
                return (
                    ClaimTransitionDecision(
                        decision="ADD",
                        previous_claim_id=None,
                        rationale="Structured claim transition adjudication failed; falling back to ADD.",
                    ),
                    False,
                    str(exc),
                    (time.perf_counter() - started_at) * 1000.0,
                    self._structured_failure_metadata(task, exc, vendor=vendor),
                    None,
                )
        try:
            response = self.llm_provider.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata=metadata,
            )
            selected_candidate_label = self._selected_candidate_label_from_text(response.text)
            decision = parse_claim_transition_decision(response.text, candidate_label_map)
            return (
                decision,
                True,
                None,
                (time.perf_counter() - started_at) * 1000.0,
                dict(response.metadata),
                selected_candidate_label,
            )
        except Exception as exc:  # noqa: BLE001
            return (
                ClaimTransitionDecision(
                    decision="ADD",
                    previous_claim_id=None,
                    rationale="Claim transition adjudication failed; falling back to ADD.",
                ),
                False,
                str(exc),
                (time.perf_counter() - started_at) * 1000.0,
                {},
                None,
            )

    def _derive_episodic_state(
        self,
        sample_id: str,
        trajectory_id_value: str,
        parsed: ParsedMemory,
        previous_claims: list[ClaimRecord],
        exchange_text: str,
    ) -> DerivedClaimState:
        previous_views = [self._canonical_claim_view(claim) for claim in previous_claims]
        extracted_claims, duplicate_extracted_claims = self._dedupe_extracted_claims(parsed.claims)
        parsed.claims = extracted_claims
        parsed.raw.claims = extracted_claims
        parsed.metadata["duplicate_extracted_claim_count"] = len(duplicate_extracted_claims)
        parsed.metadata["duplicate_extracted_claims"] = duplicate_extracted_claims
        previous_by_text: dict[str, list[int]] = defaultdict(list)
        for index, previous in enumerate(previous_views):
            previous_by_text[previous.normalized_text].append(index)

        matched_updates: dict[int, CanonicalClaimView] = {}
        matched_pairs: list[MatchedClaimPair] = []
        reused_claim_ids: list[str] = []
        unmatched_current: list[MemoryClaim] = []
        derived_ops: list[ClaimOp] = []
        status_updated_count = 0

        for current in extracted_claims:
            normalized_text = self._normalize_claim_text(current.text)
            candidate_indices = previous_by_text.get(normalized_text, [])
            if not candidate_indices:
                unmatched_current.append(current)
                continue
            previous_index = candidate_indices.pop(0)
            previous = previous_views[previous_index]
            merged_sources = self._dedupe_source_ids(previous.source_message_ids + list(current.source_message_ids))
            if previous.status != current.status:
                status_updated_count += 1
                op_type = "DEPRECATE" if current.status == "deprecated" else "REVISE"
                derived_ops.append(
                    ClaimOp(
                        op=op_type,
                        target_claim_id=previous.claim_id,
                        new_claim_id=None if op_type == "DEPRECATE" else previous.claim_id,
                        source_message_ids=list(current.source_message_ids),
                        rationale=(
                            f"Status changed from {previous.status} to {current.status} "
                            "for the same claim text."
                        ),
                        claim_text=current.text,
                        metadata={
                            "status_transition": True,
                            "previous_status": previous.status,
                            "new_status": current.status,
                            "same_claim_id": True,
                        },
                    )
                )
            matched_updates[previous_index] = CanonicalClaimView(
                claim_id=previous.claim_id,
                text=previous.text,
                normalized_text=previous.normalized_text,
                status=current.status,
                source_message_ids=merged_sources,
                parent_claim_id=previous.parent_claim_id,
                revised_from_claim_id=previous.revised_from_claim_id,
            )
            matched_pairs.append(
                MatchedClaimPair(
                    previous_claim_id=previous.claim_id,
                    current_local_claim_id=current.claim_id,
                    normalized_text=normalized_text,
                )
            )
            reused_claim_ids.append(previous.claim_id)

        next_claim_counter = max(
            self.store.next_claim_ordinal(trajectory_id_value),
            self._next_suffix(previous_claims, r"-c(\d+)$"),
        )
        revised_previous_updates: dict[int, CanonicalClaimView] = {}
        available_previous: list[tuple[int, CanonicalClaimView]] = [
            (index, previous_views[index])
            for index in range(len(previous_views))
            if index not in matched_updates
        ]
        new_claim_views: list[CanonicalClaimView] = []
        new_claim_ids: list[str] = []
        transition_debug: list[dict[str, Any]] = []
        transition_judge_attempt_count = 0
        transition_judge_success_count = 0
        transition_judge_fallback_count = 0
        transition_revise_count = 0
        transition_add_count = 0

        for current in unmatched_current:
            candidates_with_overlap = self._shortlist_transition_candidates(current, available_previous)
            candidate_views = [candidate for _, candidate, _ in candidates_with_overlap]
            debug_entry: dict[str, Any] = {
                "current_local_claim_id": current.claim_id,
                "current_text": current.text,
                "current_status": current.status,
                "current_source_message_ids": list(current.source_message_ids),
                "candidate_previous_claims": [
                    {
                        "claim_id": candidate.claim_id,
                        "text": candidate.text,
                        "status": candidate.status,
                        "token_overlap": overlap,
                    }
                    for _, candidate, overlap in candidates_with_overlap
                ],
            }
            transition_decision = ClaimTransitionDecision(
                decision="ADD",
                previous_claim_id=None,
                rationale="No plausible previous claim candidate was found.",
            )
            decision_success = False
            fallback_reason: str | None = None
            decision_latency_ms = 0.0
            decision_metadata: dict[str, Any] = {}

            if candidate_views:
                transition_judge_attempt_count += 1
                candidate_label_map = self._candidate_label_map(
                    "P",
                    [candidate.claim_id for candidate in candidate_views],
                )
                self._trace(
                    f"sample={sample_id} claim_transition_start current={current.claim_id} candidates={len(candidate_views)}"
                )
                (
                    transition_decision,
                    decision_success,
                    fallback_reason,
                    decision_latency_ms,
                    decision_metadata,
                    selected_candidate_label,
                ) = self._adjudicate_claim_transition(
                    sample_id=sample_id,
                    current_claim=current,
                    candidates=candidate_views,
                    exchange_text=exchange_text,
                )
                if decision_success:
                    transition_judge_success_count += 1
                    self._trace(
                        f"sample={sample_id} claim_transition_done decision={transition_decision.decision} "
                        f"previous={transition_decision.previous_claim_id or 'none'} latency_ms={decision_latency_ms:.1f}"
                    )
                else:
                    transition_judge_fallback_count += 1
                    self._trace(
                        f"sample={sample_id} claim_transition_fallback current={current.claim_id} "
                        f"reason={fallback_reason or 'unknown'}"
                    )
            else:
                debug_entry["decision_reason"] = "no_candidates"

            assigned_claim_id = claim_id(trajectory_id_value, next_claim_counter)
            next_claim_counter += 1

            if transition_decision.decision == "REVISE" and transition_decision.previous_claim_id:
                previous_match = next(
                    (
                        (index, previous)
                        for index, previous in available_previous
                        if previous.claim_id == transition_decision.previous_claim_id
                    ),
                    None,
                )
                if previous_match is None:
                    transition_decision = ClaimTransitionDecision(
                        decision="ADD",
                        previous_claim_id=None,
                        rationale="The shortlisted previous claim was unavailable; fell back to ADD.",
                    )
                    transition_judge_fallback_count += 1
                    debug_entry["fallback_reason"] = "previous_claim_unavailable"
                else:
                    previous_index, previous = previous_match
                    revised_previous_updates[previous_index] = CanonicalClaimView(
                        claim_id=previous.claim_id,
                        text=previous.text,
                        normalized_text=previous.normalized_text,
                        status="deprecated",
                        source_message_ids=self._dedupe_source_ids(
                            list(previous.source_message_ids) + list(current.source_message_ids)
                        ),
                        parent_claim_id=previous.parent_claim_id,
                        revised_from_claim_id=previous.revised_from_claim_id,
                    )
                    available_previous = [
                        (index, candidate)
                        for index, candidate in available_previous
                        if index != previous_index
                    ]
                    derived_ops.append(
                        ClaimOp(
                            op="REVISE",
                            target_claim_id=previous.claim_id,
                            new_claim_id=assigned_claim_id,
                            source_message_ids=list(current.source_message_ids),
                            rationale=transition_decision.rationale,
                            claim_text=current.text,
                        )
                    )
                    transition_revise_count += 1
                    debug_entry["applied_decision"] = "REVISE"

            if transition_decision.decision != "REVISE":
                derived_ops.append(
                    ClaimOp(
                        op="ADD",
                        target_claim_id=assigned_claim_id,
                        new_claim_id=assigned_claim_id,
                        source_message_ids=list(current.source_message_ids),
                        rationale=(
                            "Initial claim introduced in a new trajectory."
                            if not previous_claims
                            else "New claim appended to an existing episodic trajectory."
                        ),
                        claim_text=current.text,
                    )
                )
                transition_add_count += 1
                debug_entry["applied_decision"] = "ADD"

            new_claim_views.append(
                CanonicalClaimView(
                    claim_id=assigned_claim_id,
                    text=current.text,
                    normalized_text=self._normalize_claim_text(current.text),
                    status=current.status,
                    source_message_ids=list(current.source_message_ids),
                    parent_claim_id=transition_decision.previous_claim_id
                    if transition_decision.decision == "REVISE"
                    else None,
                    revised_from_claim_id=transition_decision.previous_claim_id
                    if transition_decision.decision == "REVISE"
                    else None,
                )
            )
            new_claim_ids.append(assigned_claim_id)
            debug_entry.update(
                {
                    "candidate_label_map": (
                        [
                            {"label": label, "resolved_id": resolved_id}
                            for label, resolved_id in candidate_label_map.items()
                        ]
                        if candidate_views
                        else []
                    ),
                    "decision": transition_decision.decision,
                    "previous_claim_id": transition_decision.previous_claim_id,
                    "selected_candidate_label": selected_candidate_label if candidate_views else None,
                    "selected_candidate_resolved_id": transition_decision.previous_claim_id,
                    "rationale": transition_decision.rationale,
                    "assigned_claim_id": assigned_claim_id,
                    "judge_invoked": bool(candidate_views),
                    "judge_success": decision_success if candidate_views else False,
                    "judge_latency_ms": decision_latency_ms,
                    "judge_fallback_reason": fallback_reason,
                    "judge_metadata": decision_metadata,
                }
            )
            transition_debug.append(debug_entry)

        next_claim_views: list[CanonicalClaimView] = []
        unmatched_previous_count = 0
        for index, previous in enumerate(previous_views):
            updated = matched_updates.get(index)
            if updated is not None:
                next_claim_views.append(updated)
                continue
            revised = revised_previous_updates.get(index)
            if revised is not None:
                next_claim_views.append(revised)
                continue
            next_claim_views.append(previous)
            unmatched_previous_count += 1
        next_claim_views.extend(new_claim_views)

        return DerivedClaimState(
            next_claim_views=next_claim_views,
            derived_ops=derived_ops,
            matched_claim_pairs=matched_pairs,
            new_claim_count=transition_add_count,
            status_updated_count=status_updated_count,
            model_ops_ignored_count=len(getattr(parsed.raw, "ops", []) or []),
            unmatched_previous_count=unmatched_previous_count,
            transition_judge_attempt_count=transition_judge_attempt_count,
            transition_judge_success_count=transition_judge_success_count,
            transition_judge_fallback_count=transition_judge_fallback_count,
            transition_revise_count=transition_revise_count,
            transition_add_count=transition_add_count,
            reused_claim_ids=reused_claim_ids,
            new_claim_ids=new_claim_ids,
            transition_debug=transition_debug,
        )

    @staticmethod
    def _safe_structured_raw(raw: Any) -> Any:
        if raw is None or isinstance(raw, (str, int, float, bool, list, dict)):
            return raw
        if hasattr(raw, "model_dump"):
            try:
                return raw.model_dump()
            except Exception:  # noqa: BLE001
                return repr(raw)
        if hasattr(raw, "dict"):
            try:
                return raw.dict()
            except Exception:  # noqa: BLE001
                return repr(raw)
        return repr(raw)

    def _structured_failure_metadata(
        self,
        task: str,
        exc: Exception,
        *,
        vendor: str,
    ) -> dict[str, Any]:
        strategy = "text_dsl_fallback"
        refusal: str | None = None
        raw: Any | None = None
        if isinstance(exc, StructuredOutputError):
            strategy = exc.strategy or strategy
            refusal = exc.refusal
            raw = exc.raw
        return {
            "structured_requested": True,
            "structured_task": task,
            "structured_vendor": vendor,
            "structured_strategy": strategy,
            "structured_success": False,
            "structured_fallback_used": True,
            "structured_fallback_reason": str(exc),
            "structured_refusal": refusal,
            "structured_failure_raw": self._safe_structured_raw(raw),
        }

    def _build_forced_episodic_seed(
        self,
        exchange_messages: list[NormalizedMessage],
        *,
        metadata: dict[str, Any],
        reason: str,
        override_field: str,
        seed_source: str,
        low_salience: bool,
    ) -> ParsedMemory:
        raw = build_fallback_episodic_memory(None, exchange_messages)
        self.forced_memory_seed_count += 1
        self.llm_no_memory_forced_count += 1
        if low_salience:
            self.low_salience_memory_count += 1
        seed_metadata = {
            **dict(metadata),
            **_base_episodic_seed_metadata(source=seed_source, llm_has_memory=False),
            "forced_episodic_seed_used_v1": True,
            "forced_episodic_seed_reason_v1": reason,
            "llm_no_memory_overridden_v1": True,
            "low_salience_memory_v1": low_salience,
            override_field: True,
        }
        return ParsedMemory(
            memory_type="episodic",
            semantic_text=raw.semantic_text,
            links=raw.links,
            claims=raw.claims,
            raw=raw,
            metadata=seed_metadata,
        )

    def _no_memory_force_recall_decision(
        self,
        exchange_messages: list[NormalizedMessage],
    ) -> NoMemoryForceRecallDecision:
        has_salience, reason = detect_episodic_salience_v1(exchange_messages)
        if self.config.dataset != "locomo":
            return NoMemoryForceRecallDecision(
                should_force=False,
                seed_source="none",
                reason="force_recall_disabled_for_dataset",
                low_salience=False,
                salience_detected=has_salience,
            )
        if has_salience:
            return NoMemoryForceRecallDecision(
                should_force=True,
                seed_source="forced_salient_no_memory",
                reason=reason,
                low_salience=False,
                salience_detected=True,
            )
        if self.config.dataset == "locomo":
            low_reason = "acknowledgement_or_low_salience_no_memory"
            return NoMemoryForceRecallDecision(
                should_force=True,
                seed_source="forced_low_salience_no_memory",
                reason=low_reason,
                low_salience=True,
                salience_detected=False,
            )
        return NoMemoryForceRecallDecision(
            should_force=False,
            seed_source="none",
            reason="none",
            low_salience=False,
            salience_detected=False,
        )

    def _build_structured_request_messages(
        self,
        task: str,
        exchange_messages: list[NormalizedMessage],
    ) -> list[NormalizedMessage]:
        return [
            NormalizedMessage(
                role="user",
                content=load_prompt(structured_prompt_name(task))
                + "\n\nConversation:\n"
                + self._render_conversation(exchange_messages),
                turn_index=0,
            )
        ]

    def request_episodic_structured_first_pass(
        self,
        exchange_messages: list[NormalizedMessage],
    ) -> StructuredFirstPassAttempt:
        task = "episodic_extract"
        vendor = self._structured_vendor()
        try:
            response = self.llm_provider.generate_structured(
                self._build_structured_request_messages(task, exchange_messages),
                spec=get_structured_task_spec(task),
                metadata={"task": task},
            )
            return StructuredFirstPassAttempt(
                task=task,
                vendor=str(response.metadata.get("structured_vendor") or vendor),
                response=response,
            )
        except Exception as exc:  # noqa: BLE001
            return StructuredFirstPassAttempt(task=task, vendor=vendor, error=exc)

    def finalize_episodic_structured_first_pass(
        self,
        sample_id: str,
        exchange_messages: list[NormalizedMessage],
        valid_link_ids: set[str],
        first_pass: StructuredFirstPassAttempt,
        *,
        batched_first_pass_metadata: dict[str, Any] | None = None,
    ) -> StructuredFinalizeResult:
        task = "episodic_extract"
        exchange_link_ids = [message.raw_message_id for message in exchange_messages if message.raw_message_id]
        exchange_timestamp = derive_exchange_timestamp(exchange_messages)
        self._record_structured_attempt(task, first_pass.vendor)
        if first_pass.response is not None:
            vendor = str(first_pass.response.metadata.get("structured_vendor") or first_pass.vendor)
            try:
                self._record_structured_success(task, vendor)
                metadata = {**dict(first_pass.response.metadata), **dict(batched_first_pass_metadata or {})}
                structured = episodic_input_from_structured(
                    first_pass.response.parsed,
                    valid_link_ids,
                    exchange_link_ids=exchange_link_ids,
                    exchange_timestamp=exchange_timestamp,
                )
                if structured is None:
                    decision = self._no_memory_force_recall_decision(exchange_messages)
                    metadata.update(
                        {
                            "llm_has_memory_v1": False,
                            "llm_no_memory_reason_v1": getattr(first_pass.response.parsed, "reason", None),
                            "episodic_salience_detected_v1": decision.salience_detected,
                        }
                    )
                    if decision.should_force:
                        self._trace(
                            f"sample={sample_id} task={task} structured_no_memory_overridden "
                            f"source={decision.seed_source} reason={decision.reason}"
                        )
                        return StructuredFinalizeResult(
                            parsed_memory=self._build_forced_episodic_seed(
                                exchange_messages,
                                metadata=metadata,
                                reason=decision.reason,
                                override_field="structured_no_memory_overridden_v1",
                                seed_source=decision.seed_source,
                                low_salience=decision.low_salience,
                            )
                        )
                    return StructuredFinalizeResult(parsed_memory=None)
                return StructuredFinalizeResult(
                    parsed_memory=ParsedMemory(
                        memory_type="episodic",
                        semantic_text=structured.semantic_text,
                        links=structured.links,
                        claims=structured.claims,
                        raw=structured,
                        metadata={
                            **metadata,
                            **_base_episodic_seed_metadata(source="llm_structured", llm_has_memory=True),
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._record_structured_fallback(task, vendor)
                structured_failure = self._structured_failure_metadata(task, exc, vendor=vendor)
        else:
            vendor = first_pass.vendor
            self._record_structured_fallback(task, vendor)
            failure = first_pass.error or StructuredOutputError(
                "Structured first-pass request failed before producing a response.",
                vendor=vendor,
                strategy="text_dsl_fallback",
            )
            structured_failure = self._structured_failure_metadata(task, failure, vendor=vendor)
        parser_diagnostics: list[dict[str, Any]] = []

        def episodic_parser(text: str):
            nonlocal parser_diagnostics
            parser_diagnostics = []
            return parse_episodic_memory(
                text,
                valid_link_ids,
                exchange_link_ids=exchange_link_ids,
                exchange_timestamp=exchange_timestamp,
                diagnostics=parser_diagnostics,
            )

        response = self._generate_with_repair(
            load_prompt("episodic_extract"),
            exchange_messages,
            sample_id=sample_id,
            memory_type="episodic",
            parser=episodic_parser,
            parser_diagnostics_getter=lambda: list(parser_diagnostics),
            metadata={"task": task, **structured_failure},
            structured_failure=structured_failure,
            batched_first_pass_metadata=batched_first_pass_metadata,
        )
        if response.parsed is None:
            decision = self._no_memory_force_recall_decision(exchange_messages)
            if decision.should_force:
                self._trace(
                    f"sample={sample_id} task={task} text_no_memory_overridden "
                    f"source={decision.seed_source} reason={decision.reason}"
                )
                return StructuredFinalizeResult(
                    parsed_memory=self._build_forced_episodic_seed(
                        exchange_messages,
                        metadata={
                            **response.metadata,
                            "episodic_salience_detected_v1": decision.salience_detected,
                        },
                        reason=decision.reason,
                        override_field="text_no_memory_overridden_v1",
                        seed_source=decision.seed_source,
                        low_salience=decision.low_salience,
                    ),
                    required_repair_or_fallback=True,
                )
            return StructuredFinalizeResult(parsed_memory=None, required_repair_or_fallback=True)
        return StructuredFinalizeResult(
            parsed_memory=ParsedMemory(
                memory_type="episodic",
                semantic_text=response.parsed.semantic_text,
                links=response.parsed.links,
                claims=response.parsed.claims,
                raw=response.parsed,
                metadata={
                    **response.metadata,
                    **_base_episodic_seed_metadata(source="llm_text_fallback", llm_has_memory=True),
                },
            ),
            required_repair_or_fallback=True,
        )

    def extract_episodic(
        self,
        sample_id: str,
        exchange_messages: list[NormalizedMessage],
        valid_link_ids: set[str],
        *,
        first_attempt: PrecomputedGenerationAttempt | None = None,
    ) -> ParsedMemory | None:
        task = "episodic_extract"
        exchange_link_ids = [message.raw_message_id for message in exchange_messages if message.raw_message_id]
        exchange_timestamp = derive_exchange_timestamp(exchange_messages)
        parser_diagnostics: list[dict[str, Any]] = []

        def episodic_parser(text: str):
            nonlocal parser_diagnostics
            parser_diagnostics = []
            return parse_episodic_memory(
                text,
                valid_link_ids,
                exchange_link_ids=exchange_link_ids,
                exchange_timestamp=exchange_timestamp,
                diagnostics=parser_diagnostics,
            )

        if self.llm_provider.supports_structured(task):
            request = self.request_episodic_structured_first_pass(exchange_messages)
            finalized = self.finalize_episodic_structured_first_pass(
                sample_id,
                exchange_messages,
                valid_link_ids,
                request,
            )
            return finalized.parsed_memory
        template = load_prompt("episodic_extract")
        response = self._generate_with_repair(
            template,
            exchange_messages,
            sample_id=sample_id,
            memory_type="episodic",
            parser=episodic_parser,
            parser_diagnostics_getter=lambda: list(parser_diagnostics),
            metadata={"task": task},
            initial_attempt=first_attempt,
        )
        if response.parsed is None:
            decision = self._no_memory_force_recall_decision(exchange_messages)
            if decision.should_force:
                self._trace(
                    f"sample={sample_id} task={task} text_no_memory_overridden "
                    f"source={decision.seed_source} reason={decision.reason}"
                )
                return self._build_forced_episodic_seed(
                    exchange_messages,
                    metadata={
                        **response.metadata,
                        "episodic_salience_detected_v1": decision.salience_detected,
                    },
                    reason=decision.reason,
                    override_field="text_no_memory_overridden_v1",
                    seed_source=decision.seed_source,
                    low_salience=decision.low_salience,
                )
            return None
        return ParsedMemory(
            memory_type="episodic",
            semantic_text=response.parsed.semantic_text,
            links=response.parsed.links,
            claims=response.parsed.claims,
            raw=response.parsed,
            metadata={
                **response.metadata,
                **_base_episodic_seed_metadata(source="llm_text", llm_has_memory=True),
            },
        )

    def _load_trajectory_match_state(self, trajectory) -> dict[str, Any] | None:
        metadata = dict(trajectory.metadata_json or {})
        latest_snapshot_id = str(metadata.get("latest_snapshot_id") or trajectory.latest_snapshot_id or "")
        latest_embedding_id = str(metadata.get("latest_snapshot_embedding_id") or "")
        latest_semantic_text = str(metadata.get("latest_semantic_text") or "")
        latest_keywords = [str(value).strip() for value in list(metadata.get("latest_keywords") or []) if str(value).strip()]
        if latest_snapshot_id and (not latest_semantic_text or not latest_keywords or not latest_embedding_id):
            snapshot = self.store.latest_snapshot(trajectory.id)
            if snapshot is not None:
                latest_semantic_text = latest_semantic_text or getattr(snapshot, "semantic_text", "")
                latest_keywords = latest_keywords or list(extract_keywords(getattr(snapshot, "semantic_text", "")))
                if not latest_embedding_id:
                    embedding = self.store.snapshot_embedding(snapshot.id)
                    if embedding is not None:
                        latest_embedding_id = embedding.id
        summary_text = str(metadata.get("retrieval_summary_text") or "").strip()
        if not summary_text:
            summary_text = fallback_summary_from_metadata(metadata, trajectory_label=str(trajectory.label or ""))
        summary_keyword_list = sanitize_summary_keyword_values(
            list(metadata.get("retrieval_summary_keywords_v2") or []),
            limit=32,
        ) or sanitize_summary_keyword_values(
            list(metadata.get("retrieval_summary_keywords") or []),
            limit=32,
        ) or summary_keywords_v2(summary_text, metadata)
        summary_embedding_id = str(metadata.get("retrieval_summary_embedding_id") or "")
        if not summary_embedding_id:
            summary_embedding = self.store.fetch_embedding(trajectory.id, "trajectory_summary")
            if summary_embedding is not None:
                summary_embedding_id = summary_embedding.id
        metadata.update(
            {
                "latest_snapshot_id": latest_snapshot_id or None,
                "latest_snapshot_embedding_id": latest_embedding_id or None,
                "latest_semantic_text": latest_semantic_text,
                "latest_keywords": latest_keywords,
                "retrieval_summary_text": summary_text,
                "retrieval_summary_keywords": summary_keyword_list,
                "retrieval_summary_keywords_v2": summary_keyword_list,
                "retrieval_summary_keyword_policy": metadata.get(
                    "retrieval_summary_keyword_policy",
                    "retrieval_summary_keywords_v2",
                ),
                "retrieval_summary_embedding_id": summary_embedding_id or None,
            }
        )
        trajectory.metadata_json = metadata
        if not summary_text and not latest_semantic_text:
            return None
        return {
            "trajectory": trajectory,
            "latest_snapshot_id": latest_snapshot_id,
            "latest_embedding_id": latest_embedding_id,
            "latest_semantic_text": latest_semantic_text,
            "latest_keywords": latest_keywords,
            "retrieval_summary_text": summary_text,
            "retrieval_summary_keywords": summary_keyword_list,
            "retrieval_summary_keywords_v2": summary_keyword_list,
            "retrieval_summary_embedding_id": summary_embedding_id,
            "exact_terms": [str(value).strip() for value in list(metadata.get("exact_terms") or []) if str(value).strip()],
            "facet_tags": [str(value).strip() for value in list(metadata.get("facet_tags") or []) if str(value).strip()],
            "facet_values": [str(value).strip() for value in list(metadata.get("facet_values") or []) if str(value).strip()],
            "entity_mentions": [str(value).strip() for value in list(metadata.get("entity_mentions") or []) if str(value).strip()],
            "historical_item_terms": sanitize_historical_item_terms(
                list(
                    metadata.get("trajectory_historical_item_terms_v2")
                    or metadata.get("trajectory_historical_item_terms_v1")
                    or []
                ),
                limit=24,
            ),
            "drift_cluster_keys": [
                str(value).strip()
                for value in list(metadata.get("trajectory_drift_cluster_keys_v1") or [])
                if str(value).strip()
            ],
            "trajectory_identity_summary": str(metadata.get("trajectory_identity_summary_v1") or ""),
            "trajectory_recent_update": str(metadata.get("trajectory_recent_update_v1") or ""),
        }

    def _update_trajectory_match_state(
        self,
        trajectory,
        *,
        snapshot_id_value: str,
        snapshot_embedding_id: str,
        semantic_text: str,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        trajectory.metadata_json = {
            **dict(trajectory.metadata_json or {}),
            "latest_snapshot_id": snapshot_id_value,
            "latest_snapshot_embedding_id": snapshot_embedding_id or None,
            "latest_semantic_text": semantic_text,
            "latest_keywords": list(extract_keywords(semantic_text)),
            **dict(metadata_updates or {}),
        }

    @staticmethod
    def _keyword_overlap_count(left: list[str], right: list[str]) -> int:
        return len(set(left) & set(right))

    def match_trajectory(self, sample_id: str, parsed: ParsedMemory) -> str | None:
        match_started = time.perf_counter()
        open_trajectories = self.store.list_trajectories(sample_id, open_only=True)
        if not open_trajectories:
            self._trace(f"sample={sample_id} match type=episodic open=0 result=new")
            return None
        source_messages = self.store.fetch_raw_messages(parsed.links)
        incoming_features = build_incoming_match_features(
            parsed.semantic_text,
            [claim.text for claim in parsed.claims],
            source_messages,
            self._sample_entity_lexicon(sample_id),
        )
        query_keywords = set(incoming_features["keywords"]) | exact_term_keyword_set(incoming_features["exact_terms"])
        query_keywords |= exact_term_keyword_set(incoming_features["facet_values"])
        query_keywords |= exact_term_keyword_set(incoming_features["entities"])
        incoming_entity_keys = {normalize_entity_key(value) for value in incoming_features["entities"]}
        incoming_facet_tags = set(incoming_features["facet_tags"])
        incoming_facet_values = set(incoming_features["facet_values"])
        incoming_specific_keys = term_keys(
            specific_terms(
                [
                    *incoming_features["exact_terms"],
                    *incoming_features["facet_values"],
                    *incoming_features["entities"],
                    *incoming_features["keywords"],
                ],
                limit=40,
            )
        )
        self.trajectory_match_total_open += len(open_trajectories)
        candidates: list[tuple[Any, dict[str, Any], int]] = []
        for trajectory in open_trajectories:
            state = self._load_trajectory_match_state(trajectory)
            if state is None:
                continue
            trajectory_keywords = set(state["retrieval_summary_keywords"]) | set(state["latest_keywords"])
            trajectory_keywords |= exact_term_keyword_set(state["exact_terms"])
            trajectory_keywords |= exact_term_keyword_set(state["facet_values"])
            trajectory_keywords |= exact_term_keyword_set(state["entity_mentions"])
            trajectory_keywords |= exact_term_keyword_set(state["historical_item_terms"])
            trajectory_keywords |= exact_term_keyword_set(state["drift_cluster_keys"])
            overlap_count = self._keyword_overlap_count(list(query_keywords), list(trajectory_keywords))
            if incoming_entity_keys and {
                normalize_entity_key(value) for value in state["entity_mentions"]
            } & incoming_entity_keys:
                overlap_count += 2
            if incoming_facet_values and set(state["facet_values"]) & incoming_facet_values:
                overlap_count += 1
            elif incoming_facet_tags and set(state["facet_tags"]) & incoming_facet_tags:
                overlap_count += 1
            if overlap_count > 0:
                candidates.append((trajectory, state, overlap_count))
        if not candidates:
            candidates = []
            for trajectory in open_trajectories:
                state = self._load_trajectory_match_state(trajectory)
                if state is not None:
                    candidates.append((trajectory, state, 0))
        elif len(candidates) > 32:
            candidates.sort(key=lambda item: item[2], reverse=True)
            candidates = candidates[:32]
        self.trajectory_match_prefiltered += len(candidates)
        if not candidates:
            self._trace(
                f"sample={sample_id} match type=episodic open={len(open_trajectories)} "
                f"prefiltered=0 result=new latency_ms={(time.perf_counter() - match_started) * 1000.0:.1f}"
            )
            return None
        embedding_started = time.perf_counter()
        query_embedding = self._embed_queries([parsed.semantic_text])[0]
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000.0
        latest_embeddings = self.store.fetch_embeddings_by_owner_ids(
            [state["latest_snapshot_id"] for _, state, _ in candidates if state["latest_snapshot_id"]],
            "snapshot",
        )
        summary_embeddings = self.store.fetch_embeddings_by_owner_ids(
            [trajectory.id for trajectory, _, _ in candidates],
            "trajectory_summary",
        )
        scored: list[tuple[str, float]] = []
        scoring_debug: list[dict[str, Any]] = []
        for trajectory, state, _ in candidates:
            latest_embedding = latest_embeddings.get(state["latest_snapshot_id"]) if state["latest_snapshot_id"] else None
            summary_embedding = summary_embeddings.get(trajectory.id)
            summary_similarity = (
                cosine_similarity(query_embedding, summary_embedding.vector_json)
                if summary_embedding is not None
                else 0.0
            )
            latest_similarity = (
                cosine_similarity(query_embedding, latest_embedding.vector_json)
                if latest_embedding is not None
                else 0.0
            )
            trajectory_keywords = set(state["retrieval_summary_keywords"]) | set(state["latest_keywords"])
            trajectory_keywords |= exact_term_keyword_set(state["exact_terms"])
            trajectory_keywords |= exact_term_keyword_set(state["facet_values"])
            trajectory_keywords |= exact_term_keyword_set(state["entity_mentions"])
            trajectory_keywords |= exact_term_keyword_set(state["historical_item_terms"])
            trajectory_keywords |= exact_term_keyword_set(state["drift_cluster_keys"])
            lexical = keyword_overlap_score(query_keywords, trajectory_keywords)
            entity_bonus = (
                0.08
                if incoming_entity_keys and {normalize_entity_key(value) for value in state["entity_mentions"]} & incoming_entity_keys
                else 0.0
            )
            facet_value_bonus = 0.06 if incoming_facet_values and set(state["facet_values"]) & incoming_facet_values else 0.0
            facet_tag_bonus = 0.03 if incoming_facet_tags and set(state["facet_tags"]) & incoming_facet_tags else 0.0
            trajectory_specific_keys = term_keys(
                specific_terms(
                    [
                        *state["exact_terms"],
                        *state["facet_values"],
                        *state["entity_mentions"],
                        *state["historical_item_terms"],
                        *state["drift_cluster_keys"],
                        *state["retrieval_summary_keywords"],
                    ],
                    limit=64,
                )
            )
            continuity_overlap = sorted(incoming_specific_keys & trajectory_specific_keys)
            continuity_bonus = min(0.14, 0.04 * len(continuity_overlap))
            shared_entity_only = bool(
                incoming_entity_keys
                and {normalize_entity_key(value) for value in state["entity_mentions"]} & incoming_entity_keys
                and not continuity_overlap
            )
            mismatch_penalty = 0.0
            if shared_entity_only and incoming_specific_keys and trajectory_specific_keys:
                mismatch_penalty = 0.10
            elif incoming_specific_keys and trajectory_specific_keys and not continuity_overlap and lexical > 0:
                mismatch_penalty = 0.04
            score = 0.6 * summary_similarity + 0.2 * latest_similarity + 0.2 * lexical
            score += entity_bonus + facet_value_bonus + facet_tag_bonus + continuity_bonus - mismatch_penalty
            scored.append((trajectory.id, score))
            scoring_debug.append(
                {
                    "trajectory_id": trajectory.id,
                    "score": score,
                    "summary_similarity": summary_similarity,
                    "latest_similarity": latest_similarity,
                    "lexical": lexical,
                    "entity_bonus": entity_bonus,
                    "facet_value_bonus": facet_value_bonus,
                    "facet_tag_bonus": facet_tag_bonus,
                    "continuity_bonus": continuity_bonus,
                    "mismatch_penalty": mismatch_penalty,
                    "continuity_overlap": continuity_overlap[:12],
                }
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        scoring_debug.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        shortlist_entries = [
            {
                "label": f"T{index}",
                "trajectory_id": trajectory_id,
                "summary": str(
                    next(
                        state["retrieval_summary_text"]
                        for trajectory, state, _ in candidates
                        if trajectory.id == trajectory_id
                    )
                ),
                "latest_update": str(
                    next(
                        state["latest_semantic_text"] or "None recorded."
                        for trajectory, state, _ in candidates
                        if trajectory.id == trajectory_id
                    )
                ),
            }
            for index, (trajectory_id, _) in enumerate(scored[:3], start=1)
        ]
        shortlist = [entry["trajectory_id"] for entry in shortlist_entries]
        candidate_label_map = {
            entry["label"]: entry["trajectory_id"] for entry in shortlist_entries
        }
        self.trajectory_match_shortlisted += len(shortlist)
        if not shortlist:
            self._trace(
                f"sample={sample_id} match type=episodic open={len(open_trajectories)} "
                f"prefiltered={len(candidates)} shortlist=0 result=new "
                f"embed_latency_ms={embedding_latency_ms:.1f} latency_ms={(time.perf_counter() - match_started) * 1000.0:.1f}"
            )
            return None
        parsed.metadata["trajectory_match_scored_candidates_v1"] = scoring_debug[:8]
        if self.llm_provider.model_info().provider_kind == "mock" and not getattr(
            self.llm_provider, "has_callback", False
        ):
            threshold = self.config.episodic_match_threshold
            best_id, best_score = scored[0]
            decision = best_id if best_score >= threshold else None
            self._trace(
                f"sample={sample_id} match type=episodic open={len(open_trajectories)} "
                f"prefiltered={len(candidates)} shortlist={len(shortlist)} result={decision or 'new'} "
                f"embed_latency_ms={embedding_latency_ms:.1f} latency_ms={(time.perf_counter() - match_started) * 1000.0:.1f}"
            )
            return decision
        parsed.metadata["trajectory_match_candidate_label_map"] = [
            {"label": entry["label"], "resolved_id": entry["trajectory_id"]}
            for entry in shortlist_entries
        ]
        prompt = self._build_trajectory_match_prompt(
            memory_type="episodic",
            semantic_text=parsed.semantic_text,
            candidates=[
                (entry["label"], entry["summary"], entry["latest_update"])
                for entry in shortlist_entries
            ],
        )
        if self.llm_provider.supports_structured("trajectory_match"):
            vendor = self._structured_vendor()
            self._record_structured_attempt("trajectory_match", vendor)
            try:
                response = self.llm_provider.generate_structured(
                    [
                        NormalizedMessage(
                            role="user",
                            content=self._build_trajectory_match_structured_prompt(
                                memory_type="episodic",
                                semantic_text=parsed.semantic_text,
                                candidates=[
                                    (entry["label"], entry["summary"], entry["latest_update"])
                                    for entry in shortlist_entries
                                ],
                            ),
                            turn_index=0,
                        )
                    ],
                    spec=get_structured_task_spec("trajectory_match"),
                    metadata={"task": "trajectory_match", "memory_type": "episodic"},
                )
                vendor = str(response.metadata.get("structured_vendor") or vendor)
                decision = validate_trajectory_match_result(response.parsed, candidate_label_map)
                self._record_structured_success("trajectory_match", vendor)
                parsed.metadata["trajectory_match_selected_candidate_label"] = (
                    response.parsed.selected_candidate
                )
                parsed.metadata["trajectory_match_selected_candidate_resolved_id"] = decision
                self._trace(
                    f"sample={sample_id} match type={parsed.memory_type} open={len(open_trajectories)} "
                    f"prefiltered={len(candidates)} shortlist={len(shortlist)} result={decision or 'new'} "
                    f"embed_latency_ms={embedding_latency_ms:.1f} latency_ms={(time.perf_counter() - match_started) * 1000.0:.1f}"
                )
                return decision
            except Exception as exc:  # noqa: BLE001
                self._record_structured_fallback("trajectory_match", vendor)
                structured_failure = self._structured_failure_metadata("trajectory_match", exc, vendor=vendor)
                decision_text = self.llm_provider.generate(
                    [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                    metadata={
                        "task": "trajectory_match",
                        "memory_type": "episodic",
                        **structured_failure,
                    },
                ).text
                return self._resolve_text_trajectory_match_decision(
                    sample_id=sample_id,
                    parsed=parsed,
                    decision_text=decision_text,
                    candidate_label_map=candidate_label_map,
                    open_count=len(open_trajectories),
                    prefiltered_count=len(candidates),
                    shortlist_count=len(shortlist),
                    embedding_latency_ms=embedding_latency_ms,
                    match_started=match_started,
                )
        decision_text = self.llm_provider.generate(
            [NormalizedMessage(role="user", content=prompt, turn_index=0)],
            metadata={"task": "trajectory_match", "memory_type": "episodic"},
        ).text
        return self._resolve_text_trajectory_match_decision(
            sample_id=sample_id,
            parsed=parsed,
            decision_text=decision_text,
            candidate_label_map=candidate_label_map,
            open_count=len(open_trajectories),
            prefiltered_count=len(candidates),
            shortlist_count=len(shortlist),
            embedding_latency_ms=embedding_latency_ms,
            match_started=match_started,
        )

    def apply_claim_ops(
        self,
        sample_id: str,
        trajectory_id_value: str,
        parsed: ParsedMemory,
        previous_claims: list[ClaimRecord],
        exchange_text: str,
    ) -> tuple[list[ClaimRecord], list[ClaimOpRecord]]:
        derived = self._derive_episodic_state(
            sample_id,
            trajectory_id_value,
            parsed,
            previous_claims,
            exchange_text,
        )
        self.claim_text_exact_match_count += len(derived.matched_claim_pairs)
        self.claim_status_updated_count += derived.status_updated_count
        self.claim_new_add_count += derived.new_claim_count
        self.claim_unmatched_previous_count += derived.unmatched_previous_count
        self.ops_model_hint_count += derived.model_ops_ignored_count
        self.ops_model_supplied_count += derived.model_ops_ignored_count
        self.ops_synthesized_count += len(derived.derived_ops)
        self.claim_transition_judge_attempt_count += derived.transition_judge_attempt_count
        self.claim_transition_judge_success_count += derived.transition_judge_success_count
        self.claim_transition_judge_fallback_count += derived.transition_judge_fallback_count
        self.claim_transition_revise_count += derived.transition_revise_count
        self.claim_transition_add_count += derived.transition_add_count

        model_op_hints = [self._model_op_hint_dict(op) for op in list(parsed.raw.ops)]
        local_to_final_claim_ids = {
            pair.current_local_claim_id: pair.previous_claim_id
            for pair in derived.matched_claim_pairs
        }
        for debug_entry in derived.transition_debug:
            current_local_claim_id = str(debug_entry.get("current_local_claim_id") or "")
            assigned_claim_id = str(debug_entry.get("assigned_claim_id") or "")
            if current_local_claim_id and assigned_claim_id:
                local_to_final_claim_ids[current_local_claim_id] = assigned_claim_id
        parsed.metadata.update(
            {
                "ops_model_hint_count": derived.model_ops_ignored_count,
                "ops_model_supplied_count": derived.model_ops_ignored_count,
                "ops_model_ignored_for_state": derived.model_ops_ignored_count > 0,
                "model_ops_present": bool(derived.model_ops_ignored_count),
                "model_ops_valid_count": derived.model_ops_ignored_count,
                "model_ops_ignored_for_state": True if derived.model_ops_ignored_count else False,
                "ops_synthesized_count": len(derived.derived_ops),
                "ops_synthesis_strategy": "deterministic_claim_diff",
                "claim_text_exact_match_count": len(derived.matched_claim_pairs),
                "claim_status_updated_count": derived.status_updated_count,
                "claim_new_add_count": derived.new_claim_count,
                "claim_unmatched_previous_count": derived.unmatched_previous_count,
                "claim_transition_judge_attempt_count": derived.transition_judge_attempt_count,
                "claim_transition_judge_success_count": derived.transition_judge_success_count,
                "claim_transition_judge_fallback_count": derived.transition_judge_fallback_count,
                "claim_transition_revise_count": derived.transition_revise_count,
                "claim_transition_add_count": derived.transition_add_count,
                "matched_claim_pairs": [
                    {
                        "previous_claim_id": pair.previous_claim_id,
                        "current_local_claim_id": pair.current_local_claim_id,
                        "normalized_text": pair.normalized_text,
                    }
                    for pair in derived.matched_claim_pairs
                ],
                "reused_claim_ids": list(derived.reused_claim_ids),
                "newly_allocated_claim_ids": list(derived.new_claim_ids),
                "local_to_final_claim_ids_v1": dict(local_to_final_claim_ids),
                "previous_claims_summary": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "status": claim.status,
                        "source_message_ids": list(claim.source_message_ids_json),
                    }
                    for claim in previous_claims
                ],
                "current_extracted_claims_summary": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "status": claim.status,
                        "source_message_ids": list(claim.source_message_ids),
                    }
                    for claim in parsed.claims
                ],
                "system_derived_ops": [self._model_op_hint_dict(op) for op in derived.derived_ops],
                "ignored_model_ops": model_op_hints,
                "model_ops_ignored_for_state_count": derived.model_ops_ignored_count,
                "claim_transition_debug": list(derived.transition_debug),
            }
        )
        quote_by_text = dict(parsed.metadata.get("claim_supporting_quotes_by_text_v1") or {})
        previous_by_claim_id = {claim.claim_id: claim for claim in previous_claims}
        claims = []
        for claim_view in derived.next_claim_views:
            claim_metadata: dict[str, Any] = {}
            previous_claim = previous_by_claim_id.get(claim_view.claim_id)
            if previous_claim is not None:
                previous_metadata = dict(previous_claim.metadata_json or {})
                for key in CLAIM_SIGNAL_METADATA_KEYS:
                    if key in previous_metadata:
                        claim_metadata[key] = previous_metadata[key]
            quote = quote_by_text.get(claim_view.normalized_text) or quote_by_text.get(
                self._normalize_claim_text(claim_view.text)
            )
            if quote:
                claim_metadata["claim_supporting_quote_v1"] = quote
            claims.append(
                ClaimRecord(
                    id=f"{trajectory_id_value}-{claim_view.claim_id}",
                    snapshot_id="PENDING",
                    trajectory_id=trajectory_id_value,
                    claim_id=claim_view.claim_id,
                    text=claim_view.text,
                    status=claim_view.status,
                    source_message_ids_json=list(claim_view.source_message_ids),
                    parent_claim_id=claim_view.parent_claim_id,
                    revised_from_claim_id=claim_view.revised_from_claim_id,
                    metadata_json=claim_metadata,
                )
            )
        next_op_counter = self.store.next_op_ordinal(trajectory_id_value)
        op_records = [
            ClaimOpRecord(
                id=op_id(trajectory_id_value, next_op_counter + index),
                snapshot_id="PENDING",
                trajectory_id=trajectory_id_value,
                op_type=op.op,
                target_claim_id=op.target_claim_id,
                new_claim_id=op.new_claim_id,
                source_message_ids_json=list(op.source_message_ids),
                rationale=op.rationale,
                metadata_json={
                    "system_derived": True,
                    "strategy": "deterministic_claim_diff",
                    **dict(op.metadata),
                },
            )
            for index, op in enumerate(derived.derived_ops)
        ]
        return claims, op_records

    @staticmethod
    def _next_suffix(rows: Iterable[object], pattern: str) -> int:
        highest = 0
        for row in rows:
            row_id = getattr(row, "claim_id", None) or getattr(row, "id", "")
            match = re.search(pattern, str(row_id))
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _persist_memory(
        self,
        sample_id: str,
        dataset_name: str,
        parsed: ParsedMemory,
        *,
        exchange_messages: list[NormalizedMessage] | None = None,
        claim_signal_structured_first_pass: StructuredFirstPassAttempt | None = None,
        claim_signal_first_attempt: PrecomputedGenerationAttempt | None = None,
    ) -> None:
        persist_started = time.perf_counter()
        match_started = time.perf_counter()
        matched_trajectory_id = self.match_trajectory(sample_id, parsed)
        match_latency_ms = (time.perf_counter() - match_started) * 1000.0
        if matched_trajectory_id is None:
            trajectory = self.store.create_trajectory(
                sample_id=sample_id,
                dataset_name=dataset_name,
                label=self._trajectory_label(parsed),
                strict_matching=False,
                max_length=self.config.m,
                metadata={"created_from": parsed.semantic_text[:120], **dict(parsed.metadata)},
            )
            previous_claims: list[ClaimRecord] = []
            version = 1
            trajectory_action = "new"
        else:
            trajectory = self.store.get_trajectory(matched_trajectory_id)
            if trajectory is None:
                return
            previous_claims = self.store.latest_claims(trajectory.id)
            latest_snapshot = self.store.latest_snapshot(trajectory.id)
            version = 1 if latest_snapshot is None else latest_snapshot.version + 1
            trajectory_action = "continue"
        exchange_text = self._render_conversation(exchange_messages or [])
        claims, ops = self.apply_claim_ops(
            sample_id,
            trajectory.id,
            parsed,
            previous_claims,
            exchange_text,
        )
        trajectory.metadata_json = {**dict(trajectory.metadata_json or {}), **dict(parsed.metadata)}
        self._trace(
            f"sample={sample_id} claim_diff matched={int(parsed.metadata.get('claim_text_exact_match_count', 0))} "
            f"new={int(parsed.metadata.get('claim_new_add_count', 0))} "
            f"carry_forward={int(parsed.metadata.get('claim_unmatched_previous_count', 0))}"
        )
        if parsed.metadata.get("ops_model_hint_count"):
            self._trace(
                f"sample={sample_id} model_ops_ignored_for_state count={int(parsed.metadata.get('ops_model_hint_count', 0))}"
            )
        if parsed.metadata.get("claim_transition_judge_attempt_count"):
            self._trace(
                f"sample={sample_id} claim_transition_summary attempts={int(parsed.metadata.get('claim_transition_judge_attempt_count', 0))} "
                f"successes={int(parsed.metadata.get('claim_transition_judge_success_count', 0))} "
                f"fallbacks={int(parsed.metadata.get('claim_transition_judge_fallback_count', 0))} "
                f"revises={int(parsed.metadata.get('claim_transition_revise_count', 0))} "
                f"adds={int(parsed.metadata.get('claim_transition_add_count', 0))}"
            )
        if parsed.metadata.get("ops_synthesized_count"):
            self._trace(
                f"sample={sample_id} ops_synthesized strategy={parsed.metadata.get('ops_synthesis_strategy', 'add_only')} "
                f"count={int(parsed.metadata.get('ops_synthesized_count', 0))}"
            )
        snapshot_id_value = snapshot_id(trajectory.id, version)
        for claim in claims:
            claim.snapshot_id = snapshot_id_value
            claim.id = f"{snapshot_id_value}:{claim.claim_id}"
            claim.metadata_json = {**dict(claim.metadata_json or {}), **dict(parsed.metadata)}
        for op in ops:
            op.snapshot_id = snapshot_id_value
            op.metadata_json = {**dict(op.metadata_json or {}), **dict(parsed.metadata)}
        claim_id_map = {
            str(key): str(value)
            for key, value in dict(parsed.metadata.get("local_to_final_claim_ids_v1") or {}).items()
            if str(key).strip() and str(value).strip()
        }
        claim_signal_target_ids = set(claim_id_map.values()) if claim_id_map and (
            claim_signal_structured_first_pass is not None or claim_signal_first_attempt is not None
        ) else None
        self._enrich_episodic_claim_facets(
            sample_id,
            claims,
            structured_first_pass=claim_signal_structured_first_pass,
            first_attempt=claim_signal_first_attempt,
            claim_id_map=claim_id_map,
            target_claim_ids=claim_signal_target_ids,
        )
        self._update_trajectory_entity_facet_summary(sample_id, trajectory, claims)
        snapshot = EpisodicMemorySnapshot(
            id=snapshot_id_value,
            trajectory_id=trajectory.id,
            version=version,
            timestamp=parsed.raw.timestamp,
            links_json=parsed.raw.links,
            summary_content=parsed.raw.summary_content,
            context=parsed.raw.context,
            keywords_json=parsed.raw.keywords,
            status_flags_json=parsed.raw.status_flags,
            embedding_ref=None,
            semantic_text=parsed.semantic_text,
            raw_text=parsed.raw.raw_text,
            metadata_json=dict(parsed.metadata),
        )
        self.store.save_episodic_snapshot(snapshot)
        self.store.replace_claims_for_snapshot(claims)
        self.store.add_claim_ops(ops)
        embedding_started = time.perf_counter()
        vector = self._embed_documents([parsed.semantic_text])[0]
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000.0
        embedding = self.store.save_embedding(
            embedding_id=f"{snapshot_id_value}-emb",
            owner_type="snapshot",
            owner_id=snapshot_id_value,
            model_name=self.embedding_provider.model_info().model_name,
            vector=vector,
            semantic_text=parsed.semantic_text,
            metadata={
                "memory_type": "episodic",
                "document_embedding_strategy": self._document_embedding_strategy(),
            },
        )
        snapshot.embedding_ref = embedding.id
        self.store.session.flush()
        summary_started = time.perf_counter()
        self._refresh_trajectory_retrieval_summary(trajectory.id, closed=False)
        summary_latency_ms = (time.perf_counter() - summary_started) * 1000.0
        if parsed.metadata.get("force_close_after_persist"):
            self.closed_on_fallback += 1
            self._close_trajectory(
                trajectory.id,
                close_reason=str(parsed.metadata.get("close_reason") or "extraction_fallback_close"),
            )
        if trajectory.max_length and version >= trajectory.max_length:
            self._close_trajectory(trajectory.id, close_reason="max_length", refresh_summary=False)
        self._trace(
            f"sample={sample_id} persist type=episodic trajectory={trajectory.id} action={trajectory_action} "
            f"version={version} claims={len(claims)} ops={len(ops)} "
            f"match_latency_ms={match_latency_ms:.1f} embed_latency_ms={embedding_latency_ms:.1f} "
            f"summary_latency_ms={summary_latency_ms:.1f} "
            f"total_latency_ms={(time.perf_counter() - persist_started) * 1000.0:.1f}"
        )

    def _close_trajectory(self, trajectory_id: str, close_reason: str, *, refresh_summary: bool = True) -> None:
        trajectory = self.store.get_trajectory(trajectory_id)
        if trajectory is None:
            return
        trajectory.metadata_json = {**dict(trajectory.metadata_json or {}), "close_reason": close_reason}
        trajectory.is_open = False
        trajectory.closed_at = trajectory.closed_at or datetime.utcnow()
        if refresh_summary:
            self._refresh_trajectory_retrieval_summary(trajectory_id, closed=True)

    def _trajectory_label(self, parsed: ParsedMemory) -> str:
        raw = parsed.raw
        label_source = f"{raw.summary_content} {' '.join(raw.keywords)}"
        keywords = list(extract_keywords(label_source))
        return "-".join(keywords[:5]) if keywords else "episodic-trajectory"

    def _record_debug_artifact(
        self,
        sample_id: str,
        exchange_messages: list[NormalizedMessage],
        task: str,
        payload: dict[str, Any],
    ) -> str | None:
        if self.debug_dir is None:
            return None
        exchange_index = exchange_messages[-1].turn_index if exchange_messages else 0
        path = self.debug_dir / sample_id / f"t{exchange_index:04d}_{task}.json"
        write_json(path, payload)
        self.debug_artifact_paths[sample_id].append(str(path))
        return str(path)

    def _generate_with_repair(
        self,
        template: str,
        exchange_messages: list[NormalizedMessage],
        *,
        sample_id: str,
        memory_type: str,
        parser: Callable[[str], object | None],
        parser_diagnostics_getter: Callable[[], list[dict[str, Any]]] | None = None,
        metadata: dict,
        structured_failure: dict[str, Any] | None = None,
        initial_attempt: PrecomputedGenerationAttempt | None = None,
        batched_first_pass_metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        conversation = self._render_conversation(exchange_messages)
        task = str(metadata.get("task") or f"{memory_type}_extract")
        latest_error: Exception | None = None
        accumulated_draft: PartialMemoryDraft | None = None
        attempt_payloads: list[dict[str, Any]] = []
        last_parser_diagnostics: list[dict[str, Any]] = []
        repair_after_batched_attempt = bool((batched_first_pass_metadata or {}).get("batched_attempt"))
        batch_metadata = {
            "batched_attempt": initial_attempt is not None,
            "batch_size": initial_attempt.batch_size if initial_attempt is not None else None,
            "batch_index": initial_attempt.batch_index if initial_attempt is not None else None,
            "batch_latency_ms": (
                initial_attempt.response_metadata.get("batch_wall_time_ms")
                if initial_attempt is not None
                else None
            ),
            "batch_mode": (
                initial_attempt.response_metadata.get("batch_mode")
                if initial_attempt is not None
                else None
            ),
        }
        if batched_first_pass_metadata:
            batch_metadata.update(dict(batched_first_pass_metadata))
            batch_metadata["batched_attempt"] = bool(batch_metadata.get("batched_attempt"))
        messages = [
            NormalizedMessage(
                role="user",
                content=template + "\n\nConversation:\n" + conversation,
                turn_index=0,
            )
        ]
        for attempt in range(3):
            self.parse_attempts += 1
            using_precomputed_attempt = attempt == 0 and initial_attempt is not None
            current_prompt = (
                initial_attempt.prompt_text
                if using_precomputed_attempt and initial_attempt is not None
                else messages[-1].content
            )
            self._trace(
                f"sample={sample_id} task={task} attempt={attempt + 1}/3 llm_generate_start"
                + (" batched=True" if using_precomputed_attempt else "")
            )
            if using_precomputed_attempt and initial_attempt is not None:
                raw_output = initial_attempt.text
                generation_latency_ms = initial_attempt.generation_latency_ms
                prompt_tokens = initial_attempt.prompt_tokens
                completion_tokens = initial_attempt.completion_tokens
                response_metadata = dict(initial_attempt.response_metadata)
            else:
                generation_started = time.perf_counter()
                attempt_metadata = dict(metadata)
                if attempt > 0:
                    attempt_metadata["repair_round"] = attempt
                else:
                    attempt_metadata.pop("repair_round", None)
                response = self.llm_provider.generate(messages, metadata=attempt_metadata)
                raw_output = response.text
                generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0
                prompt_tokens = response.prompt_tokens
                completion_tokens = response.completion_tokens
                response_metadata = dict(response.metadata)
            self._trace(
                f"sample={sample_id} task={task} attempt={attempt + 1}/3 llm_generate_done "
                f"latency_ms={generation_latency_ms:.1f} chars={len(raw_output)}"
            )
            if raw_output.strip() == "NO_MEMORY":
                self._trace(f"sample={sample_id} task={task} attempt={attempt + 1}/3 result=NO_MEMORY")
                return ExtractionResult(
                    parsed=None,
                    metadata={
                        "extraction_task": task,
                        "extraction_repair_count": len(attempt_payloads),
                        "extraction_fallback_used": False,
                        "extraction_repair_failed": False,
                        "repair_after_batched_attempt": repair_after_batched_attempt,
                        "llm_has_memory_v1": False,
                        "llm_no_memory_reason_v1": "NO_MEMORY",
                        **batch_metadata,
                        "memory_debug_artifact_paths": [],
                        **dict(structured_failure or {}),
                    },
                )
            normalized_text, explanation_text = normalize_memory_text(raw_output)
            incoming_draft = build_partial_memory_draft(memory_type, raw_output)
            accumulated_draft = merge_partial_memory_drafts(accumulated_draft, incoming_draft)
            candidate_text = render_partial_memory_draft(memory_type, accumulated_draft) or normalized_text
            try:
                parsed = parser(candidate_text)
                parse_diagnostics = list(parser_diagnostics_getter() or []) if parser_diagnostics_getter else []
                last_parser_diagnostics = list(parse_diagnostics)
                link_diagnostics = self._diagnostics_of_kind(parse_diagnostics, "link_salvage")
                ignored_ops_diagnostics = self._diagnostics_of_kind(parse_diagnostics, "ops_ignored")
                if parse_diagnostics:
                    self._record_parser_diagnostics(
                        sample_id=sample_id,
                        task=task,
                        attempt=attempt + 1,
                        diagnostics=parse_diagnostics,
                    )
                self.parse_successes += 1
                self._trace(
                    f"sample={sample_id} task={task} attempt={attempt + 1}/3 parse_success"
                )
                result_metadata = {
                    "extraction_task": task,
                    "extraction_repair_count": len(attempt_payloads),
                    "extraction_fallback_used": False,
                    "extraction_repair_failed": False,
                    "repair_after_batched_attempt": repair_after_batched_attempt,
                    **batch_metadata,
                    "normalized_extraction_text": candidate_text,
                    "parser_diagnostics": parse_diagnostics,
                    "link_salvage_diagnostics": link_diagnostics,
                    "link_salvage_count": len(link_diagnostics),
                    "exchange_link_fallback_used": any(
                        diagnostic.get("exchange_link_fallback_used") for diagnostic in link_diagnostics
                    ),
                    "ops_ignored_diagnostics": ignored_ops_diagnostics,
                    "ops_ignored_count": len(ignored_ops_diagnostics),
                    "memory_debug_artifact_paths": [],
                    **dict(structured_failure or {}),
                }
                if attempt_payloads:
                    debug_path = self._record_debug_artifact(
                        sample_id,
                        exchange_messages,
                        task,
                        {
                            "task": task,
                            "sample_id": sample_id,
                            "memory_type": memory_type,
                            "conversation": conversation,
                            "attempts": attempt_payloads,
                            "final_outcome": "repaired_success",
                            "final_candidate_text": candidate_text,
                            "parser_diagnostics": parse_diagnostics,
                            "fallback_used": False,
                            "structured_failure": structured_failure,
                        },
                    )
                    if debug_path is not None:
                        result_metadata["memory_debug_artifact_paths"] = [debug_path]
                return ExtractionResult(parsed=parsed, metadata=result_metadata)
            except Exception as exc:  # noqa: BLE001
                latest_error = exc
                self.parse_failures += 1
                parse_diagnostics = list(parser_diagnostics_getter() or []) if parser_diagnostics_getter else []
                last_parser_diagnostics = list(parse_diagnostics)
                if parse_diagnostics:
                    self._record_parser_diagnostics(
                        sample_id=sample_id,
                        task=task,
                        attempt=attempt + 1,
                        diagnostics=parse_diagnostics,
                    )
                error_payload = self._parser_error_dict(exc)
                if isinstance(exc, ParserValidationError):
                    if exc.section == "OPS":
                        self.ops_parse_failure_count += 1
                    elif exc.section == "CLAIMS":
                        self.claims_parse_failure_count += 1
                self._trace(
                    f"sample={sample_id} task={task} attempt={attempt + 1}/3 parse_failed error={exc}"
                    + (
                        f" code={error_payload.get('code')} section={error_payload.get('section')} field={error_payload.get('field')}"
                        if error_payload
                        else ""
                    )
                )
                accumulated_draft = build_partial_memory_draft(
                    memory_type,
                    candidate_text or raw_output,
                    validation_error=exc,
                )
                if "CLAIMS" in accumulated_draft.repair_targets:
                    self.claims_required_repair_count += 1
                if not accumulated_draft.repair_targets:
                    self.empty_repair_target_count += 1
                if using_precomputed_attempt:
                    repair_after_batched_attempt = True
                attempt_payloads.append(
                    {
                        "attempt_index": attempt,
                        "prompt": current_prompt,
                        "raw_output": raw_output,
                        "generation_latency_ms": generation_latency_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "batched_attempt": using_precomputed_attempt,
                        "batch_size": batch_metadata["batch_size"],
                        "batch_index": batch_metadata["batch_index"],
                        "batch_latency_ms": batch_metadata["batch_latency_ms"],
                        "response_metadata": response_metadata,
                        "normalized_text": normalized_text,
                        "explanation_text": explanation_text,
                        "candidate_text": candidate_text,
                        "draft": accumulated_draft.to_debug_dict(),
                        "parser_diagnostics": parse_diagnostics,
                        "parser_error": error_payload,
                        "validation_error": str(exc),
                    }
                )
                if attempt == 2:
                    break
                self.repair_rounds += 1
                repair_prompt = build_section_repair_prompt(
                    template=load_prompt("repair"),
                    memory_type=memory_type,
                    conversation=conversation,
                    draft=accumulated_draft,
                    validation_error=str(exc),
                )
                self._trace(
                    f"sample={sample_id} task={task} attempt={attempt + 1}/3 repair_targets={accumulated_draft.repair_targets}"
                )
                messages = [NormalizedMessage(role="user", content=repair_prompt, turn_index=0)]
        fallback_memory = build_fallback_episodic_memory(accumulated_draft, exchange_messages)
        self.extraction_fallbacks += 1
        self._trace(f"sample={sample_id} task={task} fallback_memory_used")
        fallback_payload = (
            fallback_memory.model_dump()
            if hasattr(fallback_memory, "model_dump")
            else fallback_memory.dict()
        )
        debug_path = self._record_debug_artifact(
            sample_id,
            exchange_messages,
            task,
            {
                "task": task,
                "sample_id": sample_id,
                "memory_type": memory_type,
                "conversation": conversation,
                "attempts": attempt_payloads,
                "parser_diagnostics": last_parser_diagnostics,
                "final_outcome": "fallback_memory",
                "fallback_used": True,
                "fallback_memory": fallback_payload,
                "fallback_raw_text": fallback_memory.raw_text,
                "latest_error": str(latest_error) if latest_error is not None else None,
                "structured_failure": structured_failure,
            },
        )
        return ExtractionResult(
            parsed=fallback_memory,
            metadata={
                "extraction_task": task,
                "extraction_repair_count": len(attempt_payloads),
                "extraction_fallback_used": True,
                "extraction_repair_failed": True,
                "repair_after_batched_attempt": repair_after_batched_attempt,
                **batch_metadata,
                "fallback_reason": str(latest_error) if latest_error is not None else "repair_exhausted",
                "force_close_after_persist": True,
                "close_reason": "extraction_fallback_close",
                "parser_diagnostics": last_parser_diagnostics,
                "link_salvage_diagnostics": self._diagnostics_of_kind(last_parser_diagnostics, "link_salvage"),
                "link_salvage_count": len(self._diagnostics_of_kind(last_parser_diagnostics, "link_salvage")),
                "exchange_link_fallback_used": any(
                    diagnostic.get("exchange_link_fallback_used")
                    for diagnostic in self._diagnostics_of_kind(last_parser_diagnostics, "link_salvage")
                ),
                "ops_ignored_diagnostics": self._diagnostics_of_kind(last_parser_diagnostics, "ops_ignored"),
                "ops_ignored_count": len(self._diagnostics_of_kind(last_parser_diagnostics, "ops_ignored")),
                "memory_debug_artifact_paths": [debug_path] if debug_path is not None else [],
                **dict(structured_failure or {}),
            },
        )
