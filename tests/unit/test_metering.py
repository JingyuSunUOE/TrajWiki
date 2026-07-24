from __future__ import annotations

import pytest

from trajpatch.diagnostics.fallback_repair import (
    build_fallback_repair_events_for_row,
    summarize_events_for_query,
    summarize_fallback_repair_events,
)
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.metering import MeteredLLMProvider, phase_for_task, summarize_llm_calls
from trajpatch.types import LLMResponse, ModelInfo, NormalizedMessage


class _BatchProvider(LLMProvider):
    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        return LLMResponse(text="ok", prompt_tokens=2, completion_tokens=1)

    def generate_batch(self, batch_messages, *, system_prompt=None, metadata=None):
        return [
            LLMResponse(text=f"ok-{index}", prompt_tokens=3, completion_tokens=1)
            for index, _messages in enumerate(batch_messages)
        ]

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="test", model_name="batch-provider")


class _FailingProvider(_BatchProvider):
    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        raise TimeoutError("remote request timed out")

    def generate_batch(self, batch_messages, *, system_prompt=None, metadata=None):
        raise ConnectionError("connection reset")


def test_metered_generate_counts_one_provider_and_logical_call() -> None:
    provider = MeteredLLMProvider(_BatchProvider(), role="backbone")

    provider.generate(
        [NormalizedMessage(role="user", content="hello", turn_index=0)],
        metadata={"task": "answer_evidence_synthesis"},
    )
    usage = provider.diff(0)

    assert usage["provider_call_count"] == 1
    assert usage["logical_call_count"] == 1
    assert usage["batch_call_count"] == 0
    assert usage["records"][0]["metadata"]["provider_call_kind"] == "generate"


def test_metered_generate_records_provider_failure() -> None:
    provider = MeteredLLMProvider(_FailingProvider(), role="backbone")

    with pytest.raises(TimeoutError):
        provider.generate(
            [NormalizedMessage(role="user", content="hello", turn_index=0)],
            metadata={"task": "answer_generation", "query_task_id": "q1"},
        )
    usage = provider.diff(0)
    failures = provider.sanitized_records(failures_only=True)

    assert usage["provider_call_count"] == 1
    assert usage["error_count"] == 1
    assert failures[0]["metadata"]["error_type"] == "TimeoutError"
    assert failures[0]["metadata"]["query_task_id"] == "q1"
    assert "prompt_text" not in failures[0]


def test_metered_generate_batch_shares_provider_call_id() -> None:
    provider = MeteredLLMProvider(
        _BatchProvider(),
        role="backbone",
        run_id="run-1",
        worker_id=3,
    )

    provider.generate_batch(
        [
            [NormalizedMessage(role="user", content="a", turn_index=0)],
            [NormalizedMessage(role="user", content="b", turn_index=0)],
            [NormalizedMessage(role="user", content="c", turn_index=0)],
            [NormalizedMessage(role="user", content="d", turn_index=0)],
        ],
        metadata={"task": "episodic_extract"},
    )
    usage = provider.diff(0)
    provider_call_ids = {record["metadata"]["provider_call_id"] for record in usage["records"]}
    provider_call_uids = {record["metadata"]["provider_call_uid"] for record in usage["records"]}
    call_item_uids = {record["metadata"]["call_item_uid"] for record in usage["records"]}

    assert usage["provider_call_count"] == 1
    assert usage["logical_call_count"] == 4
    assert usage["batch_call_count"] == 1
    assert usage["batch_item_count"] == 4
    assert usage["avg_batch_size"] == 4
    assert len(provider_call_ids) == 1
    assert provider_call_uids == {"run-1/3/backbone/backbone-generate_batch-000001"}
    assert len(call_item_uids) == 4


def test_metered_call_uids_do_not_collide_across_workers() -> None:
    first = MeteredLLMProvider(
        _BatchProvider(),
        role="backbone",
        run_id="run-1",
        worker_id=0,
    )
    second = MeteredLLMProvider(
        _BatchProvider(),
        role="backbone",
        run_id="run-1",
        worker_id=1,
    )
    message = [NormalizedMessage(role="user", content="hello", turn_index=0)]

    first.generate(message, metadata={"task": "answer_generation"})
    second.generate(message, metadata={"task": "answer_generation"})

    first_uid = first.diff(0)["records"][0]["metadata"]["provider_call_uid"]
    second_uid = second.diff(0)["records"][0]["metadata"]["provider_call_uid"]
    assert first_uid != second_uid


def test_metered_generate_batch_records_one_provider_failure_for_batch() -> None:
    provider = MeteredLLMProvider(_FailingProvider(), role="backbone")

    with pytest.raises(ConnectionError):
        provider.generate_batch(
            [
                [NormalizedMessage(role="user", content="a", turn_index=0)],
                [NormalizedMessage(role="user", content="b", turn_index=0)],
                [NormalizedMessage(role="user", content="c", turn_index=0)],
            ],
            metadata={"task": "episodic_extract"},
        )
    usage = provider.diff(0)

    assert usage["provider_call_count"] == 1
    assert usage["logical_call_count"] == 3
    assert usage["batch_call_count"] == 1
    assert usage["batch_item_count"] == 3
    assert usage["error_count"] == 1
    assert usage["records"][0]["metadata"]["logical_item_count"] == 3


def test_summarize_llm_calls_counts_fallback_and_repair() -> None:
    records = [
        {
            "role": "judge",
            "task": "locomo_judge",
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "latency_ms": 20.0,
            "provider_call_id": "judge-1",
            "provider_call_kind": "generate",
            "metadata": {
                "structured_requested": True,
                "structured_supported": False,
                "fallback_used": True,
                "fallback_mode": "text_json",
            },
        },
        {
            "role": "backbone",
            "task": "locomo_answer_repair",
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "latency_ms": 10.0,
            "provider_call_id": "backbone-1",
            "provider_call_kind": "generate",
            "metadata": {},
        },
    ]

    summary = summarize_llm_calls(records)

    assert summary["overall"]["provider_call_count"] == 2
    assert summary["fallbacks"]["fallback_counts"]["structured_unsupported"] == 1
    assert summary["fallbacks"]["fallback_counts"]["text_json"] == 1
    assert summary["repairs"]["repair_counts"]["locomo_answer_repair"] == 1


def test_summarize_llm_calls_does_not_count_text_only_local_judge_as_fallback() -> None:
    records = [
        {
            "role": "judge",
            "task": "locomo_judge",
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "latency_ms": 20.0,
            "provider_call_id": "judge-1",
            "provider_call_kind": "generate",
            "metadata": {
                "structured_requested": False,
                "structured_supported": False,
                "judge_mode": "text_only",
            },
        },
    ]

    summary = summarize_llm_calls(records)

    assert summary["overall"]["provider_call_count"] == 1
    assert summary["overall"]["fallback_count"] == 0
    assert summary["fallbacks"]["fallback_counts"] == {}


def test_phase_for_task_classifies_trajectory_retrieval_summary_as_memory_build() -> None:
    assert phase_for_task("trajectory_retrieval_summary") == "memory_build"


def test_phase_for_task_keeps_memory_internal_judge_out_of_evaluation_judge() -> None:
    assert phase_for_task("claim_transition_judge") == "memory_build"
    assert phase_for_task("locomo_judge") == "judge"
    assert phase_for_task("benchmark_judge") == "judge"


def test_fallback_repair_events_capture_non_structured_and_repair_risk() -> None:
    row = {
        "sample_id": "conv-1",
        "query_task_id": "conv-1_qa_0",
        "judge_verdict": "incorrect",
        "judge_score": 0.0,
        "tokens": {"total_tokens": 30},
        "metadata": {
            "llm_fallback_counts": {"structured_unsupported": 1, "deterministic_fallback": 1},
            "llm_repair_counts": {"locomo_answer_repair": 1},
            "llm_usage": {"provider_call_count": 3, "total_tokens": 30},
            "llm_usage_by_task": {
                "benchmark_judge": {
                    "provider_call_count": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "latency_ms": 20,
                    "fallback_count": 1,
                },
                "locomo_answer_repair": {
                    "provider_call_count": 1,
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "latency_ms": 10,
                    "repair_count": 1,
                },
            },
            "judge_metadata": {
                "task": "benchmark_judge",
                "structured_requested": True,
                "structured_supported": False,
            },
            "answer_metadata": {
                "answer_repair_attempted": True,
                "answer_repair_used": False,
                "answer_repair_discarded": True,
                "answer_repair_discard_reason": "json_parse_failed",
                "answer_repair_prompt_tokens": 5,
                "answer_repair_completion_tokens": 2,
                "answer_repair_latency_ms": 10,
            },
        },
    }

    events = build_fallback_repair_events_for_row(
        row,
        run_meta={"backbone_model": "mock-backbone", "judge_model": "mock-judge"},
    )
    query_summary = summarize_events_for_query(events)
    run_summary = summarize_fallback_repair_events(events, [row])

    assert len(events) == 3
    assert {event["event_kind"] for event in events} == {"fallback", "repair"}
    assert any(event["trigger"] == "structured_unsupported" for event in events)
    assert any(event["action"] == "deterministic_fallback" for event in events)
    assert any(event["outcome"] == "discarded" for event in events)
    assert query_summary["quality_flags"]["has_discarded_repair"] is True
    assert query_summary["quality_flags"]["quality_risk_event_count"] == 2
    assert query_summary["quality_flags"]["quality_risk_weighted_event_count"] == 2
    assert run_summary["overall"]["fallback_event_count"] == 2
    assert run_summary["overall"]["repair_discarded_count"] == 1
    assert run_summary["overall"]["quality_risk_event_count"] == 2
    assert run_summary["overall"]["quality_risk_weighted_event_count"] == 2
    assert run_summary["paper_ready"]["mean_judge_score_with_fallback"] == 0.0


def test_fallback_repair_quality_risk_count_is_not_weighted() -> None:
    events = [
        {
            "event_kind": "fallback",
            "phase": "answer",
            "task": "answer_evidence_synthesis",
            "provider_role": "backbone",
            "model": "mock",
            "action": "deterministic_fallback",
            "outcome": "success",
            "event_weight": 0.25,
            "provider_call_delta": 0,
            "prompt_tokens_delta": 0,
            "completion_tokens_delta": 0,
            "latency_ms_delta": 0,
            "sample_id": "conv-1",
            "query_task_id": "q1",
        },
        {
            "event_kind": "repair",
            "phase": "answer",
            "task": "locomo_answer_repair",
            "provider_role": "backbone",
            "model": "mock",
            "action": "discard_repair",
            "outcome": "discarded",
            "event_weight": 0.5,
            "provider_call_delta": 1,
            "prompt_tokens_delta": 4,
            "completion_tokens_delta": 2,
            "latency_ms_delta": 10,
            "sample_id": "conv-1",
            "query_task_id": "q1",
        },
    ]

    query_summary = summarize_events_for_query(events)
    run_summary = summarize_fallback_repair_events(events, [{"sample_id": "conv-1", "query_task_id": "q1"}])

    assert query_summary["quality_flags"]["quality_risk_event_count"] == 2
    assert query_summary["quality_flags"]["quality_risk_weighted_event_count"] == 0.75
    assert run_summary["overall"]["quality_risk_event_count"] == 2
    assert run_summary["overall"]["quality_risk_weighted_event_count"] == 0.75
