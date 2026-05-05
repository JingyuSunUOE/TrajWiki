"""Structured-output task schemas and conversion helpers for remote providers."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from trajpatch.exceptions import ParserValidationError
from trajpatch.memory.extraction_recovery import render_episodic_memory
from trajpatch.memory.llm_text_parsers import parse_episodic_memory
from trajpatch.memory.schemas import (
    ClaimSignalExtractionResult,
    ClaimTextExtractionResult,
    ClaimTransitionDecision,
    EpisodicMemoryInput,
)
from trajpatch.types import StructuredTaskSpec

STRUCTURED_TASKS = {
    "episodic_extract",
    "episodic_claim_text_extract",
    "claim_signal_extract",
    "trajectory_match",
    "claim_transition_judge",
    "locomo_judge",
    "medmt_judge",
    "semantic_metric_schema",
    "semantic_metric_extract",
    "retrieval_reflection",
    "answer_evidence_synthesis",
    "answer_type_verification",
    "answer_count_validation",
    "answer_repair_arbitration",
}

CLAIM_STATUS_VALUES = ["active", "deprecated", "contradictory", "needs-confirmation"]
JUDGE_VERDICT_VALUES = ["CORRECT", "PARTIAL", "INCORRECT"]
MATCH_DECISION_VALUES = ["CONTINUE", "NEW"]
CLAIM_TRANSITION_DECISION_VALUES = ["REVISE", "ADD"]
SEMANTIC_SLOT_VALUE_TYPES = ["string", "number", "date", "boolean", "entity", "list", "unknown"]


def _model_validate(model_cls, payload: Any):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _model_json_schema(model_cls) -> dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def infer_remote_vendor(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if normalized.startswith("gemini-"):
        return "gemini"
    if normalized.startswith("claude-"):
        return "anthropic"
    return "unknown"


def structured_strategy_for_vendor(vendor: str) -> str:
    return {
        "openai": "openai_json_schema",
        "gemini": "gemini_json_schema",
        "anthropic": "anthropic_tool_schema",
    }.get(vendor, "text_dsl_fallback")


def structured_prompt_name(task: str) -> str:
    return {
        "episodic_extract": "episodic_extract_structured",
        "episodic_claim_text_extract": "episodic_claim_text_extract_structured",
        "claim_signal_extract": "claim_signal_extract_structured",
        "trajectory_match": "trajectory_match_structured",
        "claim_transition_judge": "claim_transition_judge_structured",
        "locomo_judge": "locomo_judge_structured",
        "medmt_judge": "medmt_judge_structured",
        "answer_count_validation": "locomo_answer_count_validation",
        "answer_type_verification": "locomo_answer_type_verification",
        "answer_repair_arbitration": "locomo_answer_repair_arbitration",
    }[task]


def _add_additional_properties_false(schema: Any) -> Any:
    if isinstance(schema, dict):
        if schema.get("type") == "object" or schema.get("properties"):
            schema.setdefault("additionalProperties", False)
        for value in schema.values():
            _add_additional_properties_false(value)
    elif isinstance(schema, list):
        for item in schema:
            _add_additional_properties_false(item)
    return schema


def _add_property_ordering(schema: Any) -> Any:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            schema["propertyOrdering"] = list(properties.keys())
            for value in properties.values():
                _add_property_ordering(value)
        for key in ("items",):
            if key in schema:
                _add_property_ordering(schema[key])
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            if key in schema and isinstance(schema[key], list):
                for item in schema[key]:
                    _add_property_ordering(item)
    elif isinstance(schema, list):
        for item in schema:
            _add_property_ordering(item)
    return schema


def _remove_property_ordering(schema: Any) -> Any:
    if isinstance(schema, dict):
        schema.pop("propertyOrdering", None)
        for value in schema.values():
            _remove_property_ordering(value)
    elif isinstance(schema, list):
        for item in schema:
            _remove_property_ordering(item)
    return schema


def vendor_schema(spec: StructuredTaskSpec, vendor: str) -> dict[str, Any]:
    schema = copy.deepcopy(spec.json_schema)
    schema = _add_additional_properties_false(schema)
    if vendor == "gemini":
        return _add_property_ordering(schema)
    return _remove_property_ordering(schema)


def structured_schema_diagnostics(schema: dict[str, Any], *, vendor: str) -> dict[str, int | bool]:
    missing_required = 0
    missing_additional_properties = 0
    property_ordering_count = 0

    def walk(value: Any) -> None:
        nonlocal missing_required, missing_additional_properties, property_ordering_count
        if isinstance(value, dict):
            if "propertyOrdering" in value:
                property_ordering_count += 1
            properties = value.get("properties")
            if isinstance(properties, dict):
                required = set(value.get("required", []))
                missing_required += sum(1 for key in properties if key not in required)
                if value.get("additionalProperties") is not False:
                    missing_additional_properties += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    unsupported_keywords = 0 if vendor == "gemini" else property_ordering_count
    return {
        "ok": missing_required == 0
        and missing_additional_properties == 0
        and unsupported_keywords == 0,
        "missing_required": missing_required,
        "missing_additional_properties": missing_additional_properties,
        "property_ordering": property_ordering_count,
        "unsupported_keywords": unsupported_keywords,
    }


def _claim_text_extraction_schema() -> dict[str, Any]:
    claim_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": CLAIM_STATUS_VALUES},
            "source_message_ids": {"type": "array", "items": {"type": "string"}},
            "supporting_quote": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["status", "source_message_ids", "supporting_quote", "text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "has_claims": {"type": "boolean"},
            "claims": {"type": "array", "items": claim_schema},
            "reason": {"type": ["string", "null"]},
        },
        "required": ["has_claims", "claims", "reason"],
        "additionalProperties": False,
    }


def _claim_signal_extraction_schema() -> dict[str, Any]:
    exact_term_schema = {
        "type": "object",
        "properties": {
            "surface": {"type": "string"},
            "category": {"type": "string"},
            "source_claim_id": {"type": "string"},
            "source_message_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["surface", "category", "source_claim_id", "source_message_ids"],
        "additionalProperties": False,
    }
    facet_schema = {
        "type": "object",
        "properties": {
            "relation": {"type": "string"},
            "value": {"type": "string"},
            "entity": {"type": ["string", "null"]},
            "value_span": {"type": ["string", "null"]},
            "source_claim_id": {"type": "string"},
            "source_message_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "relation",
            "value",
            "entity",
            "value_span",
            "source_claim_id",
            "source_message_ids",
        ],
        "additionalProperties": False,
    }
    string_array_schema = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "exact_terms": {"type": "array", "items": exact_term_schema},
            "facets": {"type": "array", "items": facet_schema},
            "display_items": string_array_schema,
            "display_named_entities": string_array_schema,
            "display_counts": string_array_schema,
            "display_key_facts": string_array_schema,
        },
        "required": [
            "exact_terms",
            "facets",
            "display_items",
            "display_named_entities",
            "display_counts",
            "display_key_facts",
        ],
        "additionalProperties": False,
    }


def _dedupe_preserve_order(values: list[str] | None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values or []:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _system_timestamp(payload_timestamp: str | None, exchange_timestamp: str | None) -> str:
    if exchange_timestamp:
        return str(exchange_timestamp)
    if payload_timestamp:
        return str(payload_timestamp)
    return datetime.utcnow().isoformat()


def _system_links(payload_links: list[str] | None, exchange_link_ids: list[str] | None) -> list[str]:
    if exchange_link_ids:
        return _dedupe_preserve_order(exchange_link_ids)
    return _dedupe_preserve_order(payload_links)


class EpisodicMemoryPayload(BaseModel):
    summary_content: str = Field(description="One short sentence summarizing the event in this exchange.")
    context: str = Field(description="One or two sentences of supporting background for the same event.")
    keywords: list[str] = Field(description="Short retrieval keywords or short phrases, not full sentences.")


class EpisodicExtractionResult(BaseModel):
    has_memory: bool
    memory: EpisodicMemoryPayload | None = None
    reason: str | None = None


class TrajectoryMatchResult(BaseModel):
    decision: Literal["CONTINUE", "NEW"]
    selected_candidate: str | None = Field(
        default=None,
        description="Selected trajectory candidate label such as T1, or null when decision is NEW.",
    )
    rationale: str = Field(
        description="One short sentence explaining why this continues the same evolving subject or why no candidate fits."
    )


class ClaimTransitionJudgeResult(BaseModel):
    decision: Literal["REVISE", "ADD"]
    selected_candidate: str | None = Field(
        default=None,
        description="Selected previous-claim candidate label such as P1, or null when decision is ADD.",
    )
    rationale: str = Field(
        description="One short sentence explaining what old claim is replaced or why the claim is only an addition."
    )


class JudgeVerdictResult(BaseModel):
    verdict: Literal["CORRECT", "PARTIAL", "INCORRECT"]


class SemanticSlotSpec(BaseModel):
    slot_id: str = Field(description="Stable lowercase snake_case slot id.")
    description: str = Field(description="What fact or item family this slot captures.")
    value_type: Literal["string", "number", "date", "boolean", "entity", "list", "unknown"] = Field(
        description="Expected canonical value type."
    )
    required: bool = Field(description="Whether this slot is required by the reference answer.")


class SemanticSlotSchemaResult(BaseModel):
    question_type: str = Field(
        description="Short question type label, e.g. list, count, scalar, temporal, yes_no."
    )
    slots: list[SemanticSlotSpec] = Field(
        description="Ordered slots needed to grade the reference answer."
    )


class SemanticExtractedSlot(BaseModel):
    slot_id: str = Field(description="Slot id from the provided schema.")
    canonical_values: list[str] = Field(
        description="Canonical answer values extracted from the provided answer only."
    )


class SemanticSlotExtractionResult(BaseModel):
    slots: list[SemanticExtractedSlot] = Field(
        description="Extracted canonical values by schema slot."
    )


class RetrievalReflectionResult(BaseModel):
    rewritten_query: str = Field(description="Search-oriented rewrite of the original question.")
    answer_type: str = Field(description="Expected answer type such as place, person, count, date, list, or unknown.")
    target_entities: list[str] = Field(description="Entities that should guide wiki and raw-message search.")
    event_terms: list[str] = Field(description="Event or action terms relevant to the query.")
    temporal_terms: list[str] = Field(description="Time expressions relevant to the query.")
    must_find_terms: list[str] = Field(description="Terms or phrases that evidence should contain or imply.")
    candidate_page_slugs: list[str] = Field(description="Wiki page slugs likely to contain evidence.")
    raw_search_terms: list[str] = Field(description="Terms to use when rescuing evidence from raw messages.")
    rationale: str = Field(description="One short explanation of the retrieval intent.")


class AnswerSynthesisFact(BaseModel):
    fact_text: str = Field(description="One grounded fact used to answer the question.")
    source_refs: list[str] = Field(description="Source refs supporting this fact, such as D8:4.")


class AnswerSynthesisEvent(BaseModel):
    event_id: str = Field(description="Stable local id for this event, such as E1.")
    event_text: str = Field(description="Short grounded description of the counted or excluded event.")
    source_refs: list[str] = Field(description="Source refs supporting this event decision.")
    reason: str = Field(description="Why this event was counted or excluded.")


class AnswerEvidenceSynthesisResult(BaseModel):
    can_answer: bool = Field(description="Whether the retrieved evidence supports answering the question.")
    answer_type: str = Field(description="Short answer type such as count, list, status, place, date, or fact.")
    final_answer: str = Field(
        description="Final concise natural-language answer text, or empty when can_answer=false. Count answers must not be bare numbers."
    )
    supporting_facts: list[AnswerSynthesisFact] = Field(description="Grounded facts used for the answer.")
    supporting_source_refs: list[str] = Field(description="Deduplicated source refs supporting final_answer.")
    counted_events: list[AnswerSynthesisEvent] = Field(
        description="Distinct completed events counted for count questions."
    )
    excluded_events: list[AnswerSynthesisEvent] = Field(
        description="Candidate events excluded as future plans, duplicates, uncertain, or irrelevant."
    )
    uncertainties: list[str] = Field(description="Evidence gaps or ambiguities that affect the answer.")
    abstain_reason: str | None = Field(
        default=None,
        description="Brief reason why the evidence cannot answer the question, or null when can_answer=true.",
    )


class AnswerTypeVerificationResult(BaseModel):
    expected_answer_type: str = Field(description="Question-implied answer type.")
    observed_answer_type: str = Field(description="Answer-text-implied type.")
    type_match: bool = Field(description="Whether the answer text matches the expected type.")
    issue: str = Field(description="Short issue label, or 'none' when type_match is true.")
    repair_instruction: str = Field(description="Concise instruction for repairing the answer when needed.")


class AnswerRepairArbitrationResult(BaseModel):
    decision: Literal["keep_initial", "use_repair", "safe_abstain"] = Field(
        description="Which candidate answer should be used after comparing initial and repaired answers."
    )
    repair_violation: Literal[
        "none",
        "wrong_type",
        "lost_required_value",
        "less_specific",
        "unsupported_extra",
        "abstain_despite_support",
        "evidence_mismatch",
    ] = Field(description="Main reason the repaired answer should be rejected, or none.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in the arbitration decision.")
    reason: str = Field(description="One short source-grounded reason for the decision.")


class AnswerCountValidatedEvent(BaseModel):
    event_id: str = Field(description="Candidate event id copied from the validator input.")
    decision: Literal["COUNT", "EXCLUDE", "UNCERTAIN"] = Field(description="Whether this candidate should count.")
    source_refs: list[str] = Field(description="Source refs used for this event decision.")
    reason: str = Field(description="Short source-grounded reason for the decision.")


class AnswerCountValidationResult(BaseModel):
    count_scope: Literal[
        "completed_events",
        "planned_events",
        "states",
        "possessions",
        "mentions",
        "unknown",
    ] = Field(description="The event/count scope implied by the question.")
    validated_events: list[AnswerCountValidatedEvent] = Field(
        description="Validation decisions for supplied candidate events only."
    )
    final_count: int | None = Field(default=None, description="Final count if supported by supplied candidates.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in the validation.")
    validator_notes: str = Field(description="Brief note about uncertainty or important scope decisions.")


def _episodic_memory_payload_schema() -> dict[str, Any]:
    return {
        "type": ["object", "null"],
        "properties": {
            "summary_content": {
                "type": "string",
                "description": "One short sentence summarizing the event in this exchange.",
            },
            "context": {
                "type": "string",
                "description": "One or two sentences of supporting background for the same event.",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short retrieval keywords or short phrases, not full sentences.",
            },
        },
        "required": [
            "summary_content",
            "context",
            "keywords",
        ],
        "additionalProperties": False,
    }


def _episodic_extraction_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "has_memory": {"type": "boolean"},
            "memory": _episodic_memory_payload_schema(),
            "reason": {"type": ["string", "null"]},
        },
        "required": ["has_memory", "memory", "reason"],
        "additionalProperties": False,
    }


def _trajectory_match_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": MATCH_DECISION_VALUES},
            "selected_candidate": {
                "type": ["string", "null"],
                "description": "Selected trajectory candidate label such as T1, or null when decision is NEW.",
            },
            "rationale": {
                "type": "string",
                "description": "One short sentence explaining why this continues the same evolving subject or why no candidate fits.",
            },
        },
        "required": ["decision", "selected_candidate", "rationale"],
        "additionalProperties": False,
    }


def _judge_verdict_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": JUDGE_VERDICT_VALUES},
        },
        "required": ["verdict"],
        "additionalProperties": False,
    }


def _semantic_slot_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question_type": {"type": "string"},
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot_id": {"type": "string"},
                        "description": {"type": "string"},
                        "value_type": {"type": "string", "enum": SEMANTIC_SLOT_VALUE_TYPES},
                        "required": {"type": "boolean"},
                    },
                    "required": ["slot_id", "description", "value_type", "required"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["question_type", "slots"],
        "additionalProperties": False,
    }


def _semantic_slot_extraction_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot_id": {"type": "string"},
                        "canonical_values": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["slot_id", "canonical_values"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["slots"],
        "additionalProperties": False,
    }


def _retrieval_reflection_schema() -> dict[str, Any]:
    array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "rewritten_query": {"type": "string"},
            "answer_type": {"type": "string"},
            "target_entities": array,
            "event_terms": array,
            "temporal_terms": array,
            "must_find_terms": array,
            "candidate_page_slugs": array,
            "raw_search_terms": array,
            "rationale": {"type": "string"},
        },
        "required": [
            "rewritten_query",
            "answer_type",
            "target_entities",
            "event_terms",
            "temporal_terms",
            "must_find_terms",
            "candidate_page_slugs",
            "raw_search_terms",
            "rationale",
        ],
        "additionalProperties": False,
    }


def _answer_synthesis_fact_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fact_text": {"type": "string"},
            "source_refs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["fact_text", "source_refs"],
        "additionalProperties": False,
    }


def _answer_synthesis_event_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "event_id": {"type": "string"},
            "event_text": {"type": "string"},
            "source_refs": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["event_id", "event_text", "source_refs", "reason"],
        "additionalProperties": False,
    }


def _answer_evidence_synthesis_schema() -> dict[str, Any]:
    fact_schema = _answer_synthesis_fact_schema()
    event_schema = _answer_synthesis_event_schema()
    return {
        "type": "object",
        "properties": {
            "can_answer": {"type": "boolean"},
            "answer_type": {"type": "string"},
            "final_answer": {
                "type": "string",
                "description": "Final concise natural-language answer text; count answers must not be bare numbers.",
            },
            "supporting_facts": {"type": "array", "items": fact_schema},
            "supporting_source_refs": {"type": "array", "items": {"type": "string"}},
            "counted_events": {"type": "array", "items": event_schema},
            "excluded_events": {"type": "array", "items": event_schema},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "abstain_reason": {"type": ["string", "null"]},
        },
        "required": [
            "can_answer",
            "answer_type",
            "final_answer",
            "supporting_facts",
            "supporting_source_refs",
            "counted_events",
            "excluded_events",
            "uncertainties",
            "abstain_reason",
        ],
        "additionalProperties": False,
    }


def _answer_count_validation_schema() -> dict[str, Any]:
    event_schema = {
        "type": "object",
        "properties": {
            "event_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["COUNT", "EXCLUDE", "UNCERTAIN"]},
            "source_refs": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["event_id", "decision", "source_refs", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "count_scope": {
                "type": "string",
                "enum": ["completed_events", "planned_events", "states", "possessions", "mentions", "unknown"],
            },
            "validated_events": {"type": "array", "items": event_schema},
            "final_count": {"type": ["integer", "null"]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "validator_notes": {"type": "string"},
        },
        "required": ["count_scope", "validated_events", "final_count", "confidence", "validator_notes"],
        "additionalProperties": False,
    }


def _answer_type_verification_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "expected_answer_type": {"type": "string"},
            "observed_answer_type": {"type": "string"},
            "type_match": {"type": "boolean"},
            "issue": {"type": "string"},
            "repair_instruction": {"type": "string"},
        },
        "required": [
            "expected_answer_type",
            "observed_answer_type",
            "type_match",
            "issue",
            "repair_instruction",
        ],
        "additionalProperties": False,
    }


def _answer_repair_arbitration_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["keep_initial", "use_repair", "safe_abstain"]},
            "repair_violation": {
                "type": "string",
                "enum": [
                    "none",
                    "wrong_type",
                    "lost_required_value",
                    "less_specific",
                    "unsupported_extra",
                    "abstain_despite_support",
                    "evidence_mismatch",
                ],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"},
        },
        "required": ["decision", "repair_violation", "confidence", "reason"],
        "additionalProperties": False,
    }


def _claim_transition_judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": CLAIM_TRANSITION_DECISION_VALUES},
            "selected_candidate": {
                "type": ["string", "null"],
                "description": "Selected previous-claim candidate label such as P1, or null when decision is ADD.",
            },
            "rationale": {
                "type": "string",
                "description": "One short sentence explaining what old claim is replaced or why the claim is only an addition.",
            },
        },
        "required": ["decision", "selected_candidate", "rationale"],
        "additionalProperties": False,
    }


STRUCTURED_SPECS: dict[str, StructuredTaskSpec] = {
    "episodic_extract": StructuredTaskSpec(
        task="episodic_extract",
        schema_name="trajpatch_episodic_extract",
        tool_name="trajpatch_episodic_extract",
        description="Decide whether an exchange should become episodic memory and extract a compact memory seed.",
        json_schema=_episodic_extraction_schema(),
        response_model=EpisodicExtractionResult,
    ),
    "episodic_claim_text_extract": StructuredTaskSpec(
        task="episodic_claim_text_extract",
        schema_name="trajpatch_claim_text_extract",
        tool_name="trajpatch_claim_text_extract",
        description="Rewrite raw exchange details into readable source-grounded atomic claim texts.",
        json_schema=_claim_text_extraction_schema(),
        response_model=ClaimTextExtractionResult,
    ),
    "claim_signal_extract": StructuredTaskSpec(
        task="claim_signal_extract",
        schema_name="trajpatch_claim_signal_extract",
        tool_name="trajpatch_claim_signal_extract",
        description="Extract validated retrieval and display signals from readable claims.",
        json_schema=_claim_signal_extraction_schema(),
        response_model=ClaimSignalExtractionResult,
    ),
    "trajectory_match": StructuredTaskSpec(
        task="trajectory_match",
        schema_name="trajpatch_trajectory_match",
        tool_name="trajpatch_trajectory_match",
        description="Decide whether a new memory continues an existing trajectory or starts a new one.",
        json_schema=_trajectory_match_schema(),
        response_model=TrajectoryMatchResult,
    ),
    "claim_transition_judge": StructuredTaskSpec(
        task="claim_transition_judge",
        schema_name="trajpatch_claim_transition_judge",
        tool_name="trajpatch_claim_transition_judge",
        description="Decide whether the current claim clearly revises a shortlisted previous claim or should be added as new.",
        json_schema=_claim_transition_judge_schema(),
        response_model=ClaimTransitionJudgeResult,
    ),
    "locomo_judge": StructuredTaskSpec(
        task="locomo_judge",
        schema_name="trajpatch_locomo_judge",
        tool_name="trajpatch_locomo_judge",
        description="Judge whether the candidate answer passes the LOCOMO rubric.",
        json_schema=_judge_verdict_schema(),
        response_model=JudgeVerdictResult,
    ),
    "medmt_judge": StructuredTaskSpec(
        task="medmt_judge",
        schema_name="trajpatch_medmt_judge",
        tool_name="trajpatch_medmt_judge",
        description="Judge whether the candidate answer passes the MedMT rubric.",
        json_schema=_judge_verdict_schema(),
        response_model=JudgeVerdictResult,
    ),
    "semantic_metric_schema": StructuredTaskSpec(
        task="semantic_metric_schema",
        schema_name="trajpatch_semantic_metric_schema",
        tool_name="trajpatch_semantic_metric_schema",
        description="Build a compact slot schema for semantic answer metric extraction.",
        json_schema=_semantic_slot_schema(),
        response_model=SemanticSlotSchemaResult,
    ),
    "semantic_metric_extract": StructuredTaskSpec(
        task="semantic_metric_extract",
        schema_name="trajpatch_semantic_metric_extract",
        tool_name="trajpatch_semantic_metric_extract",
        description="Extract canonical values from one answer according to a slot schema.",
        json_schema=_semantic_slot_extraction_schema(),
        response_model=SemanticSlotExtractionResult,
    ),
    "retrieval_reflection": StructuredTaskSpec(
        task="retrieval_reflection",
        schema_name="trajpatch_retrieval_reflection",
        tool_name="trajpatch_retrieval_reflection",
        description="Analyze a failed or weak retrieval and produce query/routing hints.",
        json_schema=_retrieval_reflection_schema(),
        response_model=RetrievalReflectionResult,
    ),
    "answer_evidence_synthesis": StructuredTaskSpec(
        task="answer_evidence_synthesis",
        schema_name="trajpatch_answer_evidence_synthesis",
        tool_name="trajpatch_answer_evidence_synthesis",
        description="Synthesize grounded evidence and a final LOCOMO answer from retrieved context.",
        json_schema=_answer_evidence_synthesis_schema(),
        response_model=AnswerEvidenceSynthesisResult,
    ),
    "answer_type_verification": StructuredTaskSpec(
        task="answer_type_verification",
        schema_name="trajpatch_answer_type_verification",
        tool_name="trajpatch_answer_type_verification",
        description="Verify whether a free-form LOCOMO answer matches the question-implied answer type.",
        json_schema=_answer_type_verification_schema(),
        response_model=AnswerTypeVerificationResult,
    ),
    "answer_repair_arbitration": StructuredTaskSpec(
        task="answer_repair_arbitration",
        schema_name="trajpatch_answer_repair_arbitration",
        tool_name="trajpatch_answer_repair_arbitration",
        description="Choose between initial and repaired LOCOMO answers when repair may have degraded the answer.",
        json_schema=_answer_repair_arbitration_schema(),
        response_model=AnswerRepairArbitrationResult,
    ),
    "answer_count_validation": StructuredTaskSpec(
        task="answer_count_validation",
        schema_name="trajpatch_answer_count_validation",
        tool_name="trajpatch_answer_count_validation",
        description="Arbitrate high-risk LOCOMO count event candidates using only retrieved source lines.",
        json_schema=_answer_count_validation_schema(),
        response_model=AnswerCountValidationResult,
    ),
}


def get_structured_task_spec(task: str) -> StructuredTaskSpec:
    if task not in STRUCTURED_SPECS:
        raise KeyError(f"No structured task spec registered for task: {task}")
    return STRUCTURED_SPECS[task]


def parse_structured_payload(spec: StructuredTaskSpec, payload: Any):
    return _model_validate(spec.response_model, payload)


def _ensure_has_memory(result, task: str) -> None:
    if result.has_memory and result.memory is None:
        raise ParserValidationError(f"{task} returned has_memory=true but memory=null")
    if not result.has_memory and result.memory is not None:
        raise ParserValidationError(f"{task} returned has_memory=false but memory was populated")


def episodic_input_from_structured(
    result: EpisodicExtractionResult,
    valid_link_ids: set[str] | None = None,
    *,
    exchange_link_ids: list[str] | None = None,
    exchange_timestamp: str | None = None,
) -> EpisodicMemoryInput | None:
    _ensure_has_memory(result, "episodic_extract")
    if not result.has_memory or result.memory is None:
        return None
    payload = result.memory
    snapshot = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp=_system_timestamp(None, exchange_timestamp),
        summary_content=payload.summary_content,
        context=payload.context,
        keywords=payload.keywords,
        links=_system_links(None, exchange_link_ids),
        status_flags=[],
        claims=[],
        ops=[],
        raw_text="",
    )
    snapshot.raw_text = render_episodic_memory(snapshot)
    return snapshot


def validate_trajectory_match_result(
    result: TrajectoryMatchResult,
    allowed_candidate_labels: dict[str, str],
) -> str | None:
    selected_candidate = result.selected_candidate
    if result.decision == "CONTINUE":
        if not selected_candidate:
            raise ParserValidationError(
                "Structured trajectory_match returned CONTINUE without selected_candidate"
            )
        if selected_candidate not in allowed_candidate_labels:
            raise ParserValidationError(
                f"Structured trajectory_match returned non-shortlisted selected_candidate: {selected_candidate}"
            )
        return allowed_candidate_labels[selected_candidate]
    if selected_candidate is not None:
        raise ParserValidationError(
            "Structured trajectory_match returned NEW with a non-null selected_candidate"
        )
    return None


def validate_claim_transition_judge_result(
    result: ClaimTransitionJudgeResult,
    allowed_candidate_labels: dict[str, str],
) -> ClaimTransitionDecision:
    previous_claim_id = result.selected_candidate
    if result.decision == "REVISE":
        if not previous_claim_id:
            raise ParserValidationError(
                "Structured claim_transition_judge returned REVISE without selected_candidate"
            )
        if previous_claim_id not in allowed_candidate_labels:
            raise ParserValidationError(
                f"Structured claim_transition_judge returned non-shortlisted selected_candidate: {previous_claim_id}"
            )
        previous_claim_id = allowed_candidate_labels[previous_claim_id]
    else:
        if previous_claim_id is not None:
            raise ParserValidationError(
                "Structured claim_transition_judge returned ADD with a non-null selected_candidate"
            )
        previous_claim_id = None
    return ClaimTransitionDecision(
        decision=result.decision,
        previous_claim_id=previous_claim_id,
        rationale=result.rationale,
    )


def validate_judge_verdict_result(result: JudgeVerdictResult) -> JudgeVerdictResult:
    return result
