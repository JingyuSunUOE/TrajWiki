"""Shared runtime types used across the TrajWiki codebase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class NormalizedMessage:
    role: MessageRole
    content: str
    turn_index: int
    source_ref: str | None = None
    speaker_name: str | None = None
    occurred_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_message_id: str | None = None


@dataclass(slots=True)
class DatasetSample:
    sample_id: str
    dataset_name: str
    payload: dict[str, Any]
    subset_key: str | None = None
    scene_tag: str | None = None
    excluded: bool = False
    exclusion_reason: str | None = None


@dataclass(slots=True)
class QueryTask:
    query_task_id: str
    sample_id: str
    question: str
    gold_answer: str | None = None
    gold_evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelInfo:
    provider_kind: str
    model_name: str
    parameter_billions: float | None = None
    is_remote: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    text: str
    raw: Any | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredTaskSpec:
    task: str
    schema_name: str
    tool_name: str
    description: str
    json_schema: dict[str, Any]
    response_model: Any


@dataclass(slots=True)
class StructuredLLMResponse:
    parsed: Any
    raw: Any | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DevicePlan:
    device_mode: str
    accelerator: str
    tensor_parallel_size: int = 1
    device_map: str | None = None
    cpu_offload: bool = False
    visible_devices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalBundle:
    retrieval_event_id: str
    selected_pages: list[str]
    candidate_trajectories: list[str]
    snapshot_hits: list[str]
    expanded_snapshots: list[str]
    source_message_ids: list[str]
    source_message_refs: list[str]
    prompt_context: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunReport:
    dataset: str
    processed_samples: int
    processed_queries: int
    metrics: dict[str, float]
    details: dict[str, Any]


@dataclass(slots=True)
class AnswerResult:
    sample_id: str
    dataset_name: str
    subset_key: str
    scene_tag: str | None
    query_task_id: str
    question: str
    gold_answer: str | None
    rubric: str | None
    answer_prompt: str
    answer_text: str
    answer_record_id: str
    retrieval_event_id: str
    retrieval_source_refs: list[str]
    retrieval_source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JudgeResult:
    verdict: str
    prompt: str
    score: float | None = None
    rationale: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
