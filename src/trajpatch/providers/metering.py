"""Provider wrappers for latency and token metering."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from itertools import count
from threading import Lock
from typing import Any

from trajpatch.types import LLMResponse, ModelInfo, NormalizedMessage, StructuredLLMResponse, StructuredTaskSpec

from .base import LLMProvider


def _compact_error_message(exc: BaseException, *, limit: int = 240) -> str:
    message = " ".join(str(exc).split())
    return message[:limit]


def _exception_metadata(exc: BaseException) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error_message": _compact_error_message(exc),
    }
    for attr in ("status_code", "code", "request_id"):
        value = getattr(exc, attr, None)
        if value is not None:
            metadata[attr] = str(value)
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None and "status_code" not in metadata:
            metadata["status_code"] = str(status_code)
        request_id = getattr(response, "request_id", None) or getattr(response, "headers", {}).get("x-request-id")
        if request_id is not None and "request_id" not in metadata:
            metadata["request_id"] = str(request_id)
    return metadata


@dataclass(slots=True)
class MeteredCall:
    role: str
    task: str
    prompt_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    provider_call_id: str
    provider_call_kind: str
    logical_item_count: int
    metadata: dict[str, Any]


def phase_for_task(task: str) -> str:
    normalized = str(task or "unknown").strip().lower()
    if not normalized or normalized == "unknown":
        return "unknown"
    if "repair" in normalized:
        return "repair"
    if normalized.startswith("semantic_metric"):
        return "semantic_metrics"
    if any(
        marker in normalized
        for marker in (
            "episodic",
            "claim",
            "trajectory_match",
            "trajectory_retrieval_summary",
            "trajectory_summary",
            "retrieval_summary",
            "wiki_page_plan",
            "wiki_page_compile",
            "wiki_seed",
        )
    ):
        return "memory_build"
    if normalized in {"locomo_judge", "medmt_judge", "benchmark_judge", "judge"}:
        return "judge"
    if normalized.startswith("answer_") or normalized in {
        "locomo_answer_generation",
        "medmt_answer_generation",
    }:
        return "answer"
    if normalized in {
        "retrieval_reflection",
            "wiki_page_rerank",
            "trajectory_set_rerank",
    }:
        return "retrieval"
    return "unknown"


def _fallback_categories(metadata: dict[str, Any]) -> set[str]:
    categories: set[str] = set()
    structured_requested = metadata.get("structured_requested")
    if metadata.get("structured_supported") is False and structured_requested is not False:
        categories.add("structured_unsupported")
    if metadata.get("structured_fallback_used") or metadata.get("fallback_used"):
        fallback_mode = (
            metadata.get("fallback_mode")
            or metadata.get("structured_fallback_category")
            or metadata.get("fallback_reason")
            or "fallback"
        )
        categories.add(str(fallback_mode))
    mode = str(metadata.get("mode") or metadata.get("answer_synthesis_mode") or "")
    if mode == "text_json":
        categories.add("text_json_fallback")
    elif mode == "deterministic_fallback":
        categories.add("deterministic_fallback")
    strategy = str(metadata.get("structured_strategy") or "")
    if "dsl" in strategy:
        categories.add("text_dsl_fallback")
    return categories


def _is_repair_record(task: str, metadata: dict[str, Any]) -> bool:
    normalized = str(task or "").lower()
    return bool(
        "repair" in normalized
        or metadata.get("repair_requested")
        or metadata.get("answer_repair_raw_text")
        or metadata.get("answer_repair_discarded")
    )


def _empty_group_summary() -> dict[str, Any]:
    return {
        "provider_call_count": 0,
        "logical_call_count": 0,
        "batch_call_count": 0,
        "batch_item_count": 0,
        "avg_batch_size": 0.0,
        "max_batch_size": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "error_count": 0,
        "fallback_count": 0,
        "repair_count": 0,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "task_counts": {},
        "fallback_counts": {},
        "repair_counts": {},
    }


def summarize_llm_calls(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize metered records without retaining prompts."""

    provider_ids: set[str] = set()
    batch_provider_ids: set[str] = set()
    batch_sizes: dict[str, int] = {}
    task_counter: Counter[str] = Counter()
    fallback_counter: Counter[str] = Counter()
    repair_counter: Counter[str] = Counter()
    by_role: dict[str, dict[str, Any]] = {}
    by_phase: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = {}
    overall = _empty_group_summary()

    def ensure(mapping: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        if key not in mapping:
            mapping[key] = _empty_group_summary()
        return mapping[key]

    def add_to_group(summary: dict[str, Any], record: dict[str, Any], metadata: dict[str, Any]) -> None:
        task = str(record.get("task") or metadata.get("task") or "unknown")
        provider_call_id = str(metadata.get("provider_call_id") or record.get("provider_call_id") or "")
        provider_call_kind = str(
            metadata.get("provider_call_kind") or record.get("provider_call_kind") or "generate"
        )
        batch_size = int(metadata.get("batch_size") or record.get("batch_size") or 1)
        is_error = bool(metadata.get("error_type") or metadata.get("structured_failure"))
        summary["logical_call_count"] += batch_size if is_error and provider_call_kind == "generate_batch" else 1
        summary["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
        summary["completion_tokens"] += int(record.get("completion_tokens") or 0)
        summary["total_tokens"] += int(record.get("prompt_tokens") or 0) + int(
            record.get("completion_tokens") or 0
        )
        summary["latency_ms"] += float(record.get("latency_ms") or 0.0)
        if is_error:
            summary["error_count"] += 1
        if metadata.get("cache_hit"):
            summary["cache_hit_count"] += 1
        if metadata.get("cache_miss"):
            summary["cache_miss_count"] += 1
        task_counts = Counter(dict(summary.get("task_counts", {})))
        task_counts[task] += 1
        summary["task_counts"] = dict(task_counts)
        fallback_categories = _fallback_categories(metadata)
        if fallback_categories:
            summary["fallback_count"] += 1
            fallback_counts = Counter(dict(summary.get("fallback_counts", {})))
            fallback_counts.update(fallback_categories)
            summary["fallback_counts"] = dict(fallback_counts)
        if _is_repair_record(task, metadata):
            summary["repair_count"] += 1
            repair_counts = Counter(dict(summary.get("repair_counts", {})))
            repair_counts[task] += 1
            summary["repair_counts"] = dict(repair_counts)
        if provider_call_id:
            seen_key = f"_provider_seen::{provider_call_id}"
            if not summary.get(seen_key):
                summary[seen_key] = True
                summary["provider_call_count"] += 1
                if provider_call_kind == "generate_batch" or batch_size > 1:
                    summary["batch_call_count"] += 1
                    summary["batch_item_count"] += batch_size
                    summary["max_batch_size"] = max(int(summary["max_batch_size"]), batch_size)

    for index, record in enumerate(records):
        metadata = dict(record.get("metadata") or {})
        task = str(record.get("task") or metadata.get("task") or "unknown")
        role = str(record.get("role") or metadata.get("role") or "unknown")
        phase = str(metadata.get("phase") or phase_for_task(task))
        provider_call_id = str(
            metadata.get("provider_call_id") or record.get("provider_call_id") or f"legacy-{index}"
        )
        provider_call_kind = str(
            metadata.get("provider_call_kind") or record.get("provider_call_kind") or "generate"
        )
        batch_size = int(metadata.get("batch_size") or record.get("batch_size") or 1)
        provider_ids.add(provider_call_id)
        if provider_call_kind == "generate_batch" or batch_size > 1:
            batch_provider_ids.add(provider_call_id)
            batch_sizes[provider_call_id] = batch_size
        task_counter[task] += 1
        fallback_counter.update(_fallback_categories(metadata))
        if _is_repair_record(task, metadata):
            repair_counter[task] += 1
        add_to_group(overall, record, metadata)
        add_to_group(ensure(by_role, role), record, metadata)
        add_to_group(ensure(by_phase, phase), record, metadata)
        add_to_group(ensure(by_task, task), record, metadata)

    def finalize(summary: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            key: value
            for key, value in summary.items()
            if not str(key).startswith("_provider_seen::")
        }
        batch_call_count = int(cleaned.get("batch_call_count") or 0)
        batch_item_count = int(cleaned.get("batch_item_count") or 0)
        cleaned["avg_batch_size"] = (
            float(batch_item_count) / batch_call_count if batch_call_count else 0.0
        )
        return cleaned

    overall = finalize(overall)
    overall["provider_call_count"] = len(provider_ids)
    overall["batch_call_count"] = len(batch_provider_ids)
    overall["batch_item_count"] = sum(batch_sizes.values())
    overall["avg_batch_size"] = (
        float(overall["batch_item_count"]) / overall["batch_call_count"]
        if overall["batch_call_count"]
        else 0.0
    )
    overall["max_batch_size"] = max(batch_sizes.values(), default=0)
    overall["task_counts"] = dict(task_counter)
    overall["fallback_counts"] = dict(fallback_counter)
    overall["repair_counts"] = dict(repair_counter)
    return {
        "overall": overall,
        "by_role": {key: finalize(value) for key, value in sorted(by_role.items())},
        "by_phase": {key: finalize(value) for key, value in sorted(by_phase.items())},
        "by_task": {key: finalize(value) for key, value in sorted(by_task.items())},
        "fallbacks": {
            "fallback_count": sum(fallback_counter.values()),
            "fallback_counts": dict(fallback_counter),
        },
        "repairs": {
            "repair_count": sum(repair_counter.values()),
            "repair_counts": dict(repair_counter),
        },
        "cache": {
            "cache_hit_count": int(overall.get("cache_hit_count") or 0),
            "cache_miss_count": int(overall.get("cache_miss_count") or 0),
        },
    }


class MeteredLLMProvider(LLMProvider):
    def __init__(self, provider: LLMProvider, *, role: str) -> None:
        self.provider = provider
        self.role = role
        self.calls: list[MeteredCall] = []
        self._lock = Lock()
        self._call_counter = count(1)

    def _next_provider_call_id(self, kind: str) -> str:
        with self._lock:
            return f"{self.role}-{kind}-{next(self._call_counter):06d}"

    def generate(
        self,
        messages: list[NormalizedMessage],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started_at = time.perf_counter()
        provider_call_id = self._next_provider_call_id("generate")
        prompt_text = "\n".join(message.content for message in messages)
        try:
            response = self.provider.generate(messages, system_prompt=system_prompt, metadata=metadata)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            call_metadata = {
                "role": self.role,
                "task": str((metadata or {}).get("task", "unknown")),
                "provider_model": self.provider.model_info().model_name,
                **dict(metadata or {}),
                "provider_call_id": provider_call_id,
                "provider_call_kind": "generate",
                "logical_item_count": 1,
                "batch_size": 1,
                "batch_index": 0,
                **_exception_metadata(exc),
            }
            with self._lock:
                self.calls.append(
                    MeteredCall(
                        role=self.role,
                        task=call_metadata["task"],
                        prompt_text=prompt_text,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        provider_call_id=provider_call_id,
                        provider_call_kind="generate",
                        logical_item_count=1,
                        metadata=call_metadata,
                    )
                )
            raise
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        prompt_tokens = int(response.prompt_tokens or 0)
        completion_tokens = int(response.completion_tokens or 0)
        call_metadata = {
            "role": self.role,
            "task": str((metadata or {}).get("task", "unknown")),
            "provider_model": self.provider.model_info().model_name,
            **dict(metadata or {}),
            **dict(response.metadata or {}),
            "provider_call_id": provider_call_id,
            "provider_call_kind": "generate",
            "logical_item_count": 1,
            "batch_size": 1,
            "batch_index": 0,
        }
        with self._lock:
            self.calls.append(
                MeteredCall(
                    role=self.role,
                    task=call_metadata["task"],
                    prompt_text=prompt_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    provider_call_id=provider_call_id,
                    provider_call_kind="generate",
                    logical_item_count=1,
                    metadata=call_metadata,
                )
            )
        response.metadata = {
            **dict(metadata or {}),
            **dict(response.metadata or {}),
            "latency_ms": latency_ms,
            "meter_role": self.role,
            "provider_call_id": provider_call_id,
            "provider_call_kind": "generate",
            "logical_item_count": 1,
        }
        return response

    def generate_batch(
        self,
        batch_messages: list[list[NormalizedMessage]],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LLMResponse]:
        started_at = time.perf_counter()
        provider_call_id = self._next_provider_call_id("generate_batch")
        batch_size = len(batch_messages)
        prompt_text = "\n\n--- batch item ---\n\n".join(
            "\n".join(message.content for message in messages) for messages in batch_messages
        )
        try:
            responses = self.provider.generate_batch(
                batch_messages,
                system_prompt=system_prompt,
                metadata=metadata,
            )
            if len(responses) != batch_size:
                raise ValueError(
                    f"generate_batch returned {len(responses)} responses for {batch_size} requests."
                )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            call_metadata = {
                "role": self.role,
                "task": str((metadata or {}).get("task", "unknown")),
                "provider_model": self.provider.model_info().model_name,
                **dict(metadata or {}),
                "provider_call_id": provider_call_id,
                "provider_call_kind": "generate_batch",
                "logical_item_count": batch_size,
                "batched": True,
                "batch_size": batch_size,
                "batch_index": None,
                **_exception_metadata(exc),
            }
            with self._lock:
                self.calls.append(
                    MeteredCall(
                        role=self.role,
                        task=call_metadata["task"],
                        prompt_text=prompt_text,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        provider_call_id=provider_call_id,
                        provider_call_kind="generate_batch",
                        logical_item_count=batch_size,
                        metadata=call_metadata,
                    )
                )
            raise
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        per_item_latency_ms = latency_ms / batch_size if batch_size else 0.0
        for index, (messages, response) in enumerate(zip(batch_messages, responses, strict=False)):
            prompt_text = "\n".join(message.content for message in messages)
            prompt_tokens = int(response.prompt_tokens or 0)
            completion_tokens = int(response.completion_tokens or 0)
            call_metadata = {
                "role": self.role,
                "task": str((metadata or {}).get("task", "unknown")),
                "provider_model": self.provider.model_info().model_name,
                **dict(metadata or {}),
                **dict(response.metadata or {}),
                "provider_call_id": provider_call_id,
                "provider_call_kind": "generate_batch",
                "logical_item_count": batch_size,
                "batched": True,
                "batch_size": batch_size,
                "batch_index": index,
            }
            with self._lock:
                self.calls.append(
                    MeteredCall(
                        role=self.role,
                        task=call_metadata["task"],
                        prompt_text=prompt_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=per_item_latency_ms,
                        provider_call_id=provider_call_id,
                        provider_call_kind="generate_batch",
                        logical_item_count=batch_size,
                        metadata=call_metadata,
                    )
                )
            response.metadata = {
                **dict(metadata or {}),
                **dict(response.metadata or {}),
                "latency_ms": per_item_latency_ms,
                "batch_wall_time_ms": latency_ms,
                "meter_role": self.role,
                "batched": True,
                "batch_size": batch_size,
                "batch_index": index,
                "provider_call_id": provider_call_id,
                "provider_call_kind": "generate_batch",
                "logical_item_count": batch_size,
            }
        return responses

    def supports_structured(self, task: str) -> bool:
        return self.provider.supports_structured(task)

    def generate_structured(
        self,
        messages: list[NormalizedMessage],
        *,
        spec: StructuredTaskSpec,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StructuredLLMResponse:
        started_at = time.perf_counter()
        provider_call_id = self._next_provider_call_id("generate_structured")
        prompt_text = "\n".join(message.content for message in messages)
        try:
            response = self.provider.generate_structured(
                messages,
                spec=spec,
                system_prompt=system_prompt,
                metadata=metadata,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            call_metadata = {
                "role": self.role,
                "task": str((metadata or {}).get("task", spec.task)),
                "provider_model": self.provider.model_info().model_name,
                **dict(metadata or {}),
                "provider_call_id": provider_call_id,
                "provider_call_kind": "generate_structured",
                "logical_item_count": 1,
                "batch_size": 1,
                "batch_index": 0,
                "structured_requested": True,
                "structured_supported": True,
                "structured_success": False,
                "structured_failure": True,
                **_exception_metadata(exc),
            }
            with self._lock:
                self.calls.append(
                    MeteredCall(
                        role=self.role,
                        task=call_metadata["task"],
                        prompt_text=prompt_text,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        provider_call_id=provider_call_id,
                        provider_call_kind="generate_structured",
                        logical_item_count=1,
                        metadata=call_metadata,
                    )
                )
            raise
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        prompt_tokens = int(response.prompt_tokens or 0)
        completion_tokens = int(response.completion_tokens or 0)
        call_metadata = {
            "role": self.role,
            "task": str((metadata or {}).get("task", spec.task)),
            "provider_model": self.provider.model_info().model_name,
            **dict(metadata or {}),
            **dict(response.metadata or {}),
            "provider_call_id": provider_call_id,
            "provider_call_kind": "generate_structured",
            "logical_item_count": 1,
            "batch_size": 1,
            "batch_index": 0,
            "structured_requested": True,
            "structured_supported": True,
            "structured_success": True,
            "structured_failure": False,
        }
        with self._lock:
            self.calls.append(
                MeteredCall(
                    role=self.role,
                    task=call_metadata["task"],
                    prompt_text=prompt_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    provider_call_id=provider_call_id,
                    provider_call_kind="generate_structured",
                    logical_item_count=1,
                    metadata=call_metadata,
                )
            )
        response.metadata = {
            **dict(metadata or {}),
            **dict(response.metadata or {}),
            "latency_ms": latency_ms,
            "meter_role": self.role,
            "provider_call_id": provider_call_id,
            "provider_call_kind": "generate_structured",
            "logical_item_count": 1,
        }
        return response

    def model_info(self) -> ModelInfo:
        info = self.provider.model_info()
        info.metadata = {**info.metadata, "meter_role": self.role}
        return info

    def __getattr__(self, name: str):
        return getattr(self.provider, name)

    def snapshot(self) -> int:
        with self._lock:
            return len(self.calls)

    def diff(self, start_index: int, end_index: int | None = None) -> dict[str, Any]:
        with self._lock:
            rows = list(self.calls[start_index:end_index])
        prompt_tokens = sum(row.prompt_tokens for row in rows)
        completion_tokens = sum(row.completion_tokens for row in rows)
        return {
            "call_count": len(rows),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": sum(row.latency_ms for row in rows),
            "tasks": [row.task for row in rows],
            "records": [
                {
                    "role": row.role,
                    "task": row.task,
                    "prompt_text": row.prompt_text,
                    "prompt_tokens": row.prompt_tokens,
                    "completion_tokens": row.completion_tokens,
                    "latency_ms": row.latency_ms,
                    "provider_call_id": row.provider_call_id,
                    "provider_call_kind": row.provider_call_kind,
                    "logical_item_count": row.logical_item_count,
                    "metadata": row.metadata,
                }
                for row in rows
            ],
            **summarize_llm_calls(
                [
                    {
                        "role": row.role,
                        "task": row.task,
                        "prompt_tokens": row.prompt_tokens,
                        "completion_tokens": row.completion_tokens,
                        "latency_ms": row.latency_ms,
                        "provider_call_id": row.provider_call_id,
                        "provider_call_kind": row.provider_call_kind,
                        "logical_item_count": row.logical_item_count,
                        "metadata": row.metadata,
                    }
                    for row in rows
                ]
            )["overall"],
        }

    def sanitized_records(self, *, limit: int = 20, failures_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self.calls)
        if failures_only:
            rows = [row for row in rows if row.metadata.get("error_type") or row.metadata.get("structured_failure")]
        rows = rows[-limit:]
        allowed_metadata_keys = {
            "role",
            "task",
            "provider_model",
            "provider_call_id",
            "provider_call_kind",
            "logical_item_count",
            "batch_size",
            "batch_index",
            "batched",
            "query_task_id",
            "sample_id",
            "memory_type",
            "structured_requested",
            "structured_supported",
            "structured_success",
            "structured_failure",
            "fallback_used",
            "fallback_mode",
            "fallback_reason",
            "error_type",
            "error_message",
            "status_code",
            "code",
            "request_id",
        }
        sanitized: list[dict[str, Any]] = []
        for row in rows:
            metadata = {
                key: value
                for key, value in dict(row.metadata or {}).items()
                if key in allowed_metadata_keys and value is not None
            }
            sanitized.append(
                {
                    "role": row.role,
                    "task": row.task,
                    "prompt_tokens": row.prompt_tokens,
                    "completion_tokens": row.completion_tokens,
                    "latency_ms": row.latency_ms,
                    "provider_call_id": row.provider_call_id,
                    "provider_call_kind": row.provider_call_kind,
                    "logical_item_count": row.logical_item_count,
                    "metadata": metadata,
                }
            )
        return sanitized
