from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajpatch.experiments.answer_ablation import (
    _answer_is_abstention,
    _answer_stage_usage,
    _completed_job,
    _provider_prompt_within_total_budget,
    _provider_records_from_jobs,
    import_baseline_answers,
)
from trajpatch.experiments.token_budget import TokenCounter, build_token_counter
from trajpatch.experiments.variant_contexts import (
    VARIANT_POLICIES,
    VariantContextBuilder,
)


class _CharacterCodec:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def encode_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        return self.encode(text), [
            (index, index + 1)
            for index in range(len(text))
        ]


def test_exact_counter_counts_and_truncates_without_estimator_drift() -> None:
    counter = TokenCounter(name="character-test", exact=True, codec=_CharacterCodec())

    assert counter.count("abcd") == 4
    assert counter.truncate("abcdef", 4) == ("abcd", True)
    assert counter.truncate("abc", 4) == ("abc", False)
    assert counter.truncate("abc", 0) == ("", True)


def test_requiring_exact_counter_rejects_estimator() -> None:
    with pytest.raises(RuntimeError, match="exact token counter"):
        build_token_counter(
            "estimate",
            model_name="mock-model",
            require_exact=True,
        )


def test_answer_abstention_detection_does_not_treat_safe_abstention_as_hallucination() -> None:
    assert _answer_is_abstention("") is True
    assert _answer_is_abstention("The answer is not supported by the context.") is True
    assert _answer_is_abstention("Alice prefers tea.") is False


def test_naive_rag_chunks_keep_message_level_source_provenance() -> None:
    builder = object.__new__(VariantContextBuilder)
    builder.rag_chunk_size = 64
    builder.rag_chunk_overlap = 8
    builder._rag_chunk_counter = TokenCounter(
        name="character-test",
        exact=True,
        codec=_CharacterCodec(),
    )
    builder.memory_index = {}
    messages = [
        {
            "id": "message-1",
            "source_ref": "D1:1",
            "speaker_name": "Alice",
            "occurred_at": "2025-01-01",
            "content": "Alice prefers tea in the morning.",
        },
        {
            "id": "message-2",
            "source_ref": "D1:2",
            "speaker_name": "Bob",
            "occurred_at": "2025-01-02",
            "content": "Bob prefers coffee in the evening.",
        },
    ]

    chunks = builder._chunk_conversation("conv-1", messages)

    assert chunks
    assert all(
        row["source_provenance_mode"] == "token_offset_overlap_v1"
        for row in chunks
    )
    assert all(row["contains_sensitive_text"] is True for row in chunks)
    assert {source_id for row in chunks for source_id in row["source_message_ids"]} == {
        "message-1",
        "message-2",
    }
    assert {source_ref for row in chunks for source_ref in row["source_refs"]} == {
        "D1:1",
        "D1:2",
    }


def test_full_context_budgeting_never_splits_a_raw_message() -> None:
    builder = object.__new__(VariantContextBuilder)
    builder.token_counter = TokenCounter(
        name="character-test",
        exact=True,
        codec=_CharacterCodec(),
    )
    items = [
        {"item_type": "raw_source", "text": "first"},
        {"item_type": "raw_source", "text": "second"},
    ]

    selected, truncated = builder._budget_items(
        items,
        token_budget=8,
        allow_partial_items=False,
    )

    assert [row["text"] for row in selected] == ["first"]
    assert truncated is True


def test_full_context_budgeting_keeps_latest_messages_then_restores_order() -> None:
    builder = object.__new__(VariantContextBuilder)
    builder.token_counter = TokenCounter(
        name="character-test",
        exact=True,
        codec=_CharacterCodec(),
    )
    items = [
        {"item_type": "raw_source", "text": "oldest"},
        {"item_type": "raw_source", "text": "middle"},
        {"item_type": "raw_source", "text": "latest"},
    ]

    selected, truncated = builder._budget_items_for_policy(
        items,
        token_budget=15,
        policy=VARIANT_POLICIES["full_context"],
        allow_partial_items=False,
    )

    assert [row["text"] for row in selected] == ["middle", "latest"]
    assert truncated is True


def test_partial_context_item_keeps_only_source_refs_visible_after_truncation() -> None:
    builder = object.__new__(VariantContextBuilder)
    builder.token_counter = TokenCounter(
        name="character-test",
        exact=True,
        codec=_CharacterCodec(),
    )
    builder.memory_index = {
        "raw_messages_by_id": {
            "message-1": {"source_ref": "D1:1"},
            "message-2": {"source_ref": "D1:2"},
        }
    }
    text = (
        "Snapshot summary and claims.\n"
        "[SOURCE D1:1 speaker=Alice]\nVisible evidence.\n"
        "[SOURCE D1:2 speaker=Bob]\nEvidence beyond the budget."
    )
    first_source_end = text.index("[SOURCE D1:2")
    items = [
        {
            "item_type": "snapshot",
            "text": text,
            "source_refs": ["D1:1", "D1:2"],
            "source_message_ids": ["message-1", "message-2"],
        }
    ]

    selected, truncated = builder._budget_items(
        items,
        token_budget=first_source_end,
        allow_partial_items=True,
    )

    assert truncated is True
    assert selected[0]["source_refs"] == ["D1:1"]
    assert selected[0]["source_message_ids"] == ["message-1"]
    assert "D1:2" not in selected[0]["text"]


def test_provider_usage_reconciliation_does_not_double_count_margin() -> None:
    assert (
        _provider_prompt_within_total_budget(
            actual_prompt_tokens=31_488,
            max_output_tokens=512,
            max_total_tokens=32_000,
        )
        is True
    )
    assert (
        _provider_prompt_within_total_budget(
            actual_prompt_tokens=31_489,
            max_output_tokens=512,
            max_total_tokens=32_000,
        )
        is False
    )
    assert (
        _provider_prompt_within_total_budget(
            actual_prompt_tokens=None,
            max_output_tokens=512,
            max_total_tokens=32_000,
        )
        is None
    )


def test_partial_stage_usage_preserves_missing_values() -> None:
    assert _answer_stage_usage(
        {"answer_stage_initial_prompt_tokens": 12},
        stage="initial",
    ) == (12, None)


def test_resume_accepts_only_complete_schema_and_hash_valid_jobs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "answer_generation_job_v1",
                "status": "complete",
                "experiment_config_hash": "config-a",
                "variant": "full",
                "query_task_id": "query-1",
                "prompt_sha256": "prompt-a",
            }
        ),
        encoding="utf-8",
    )

    assert (
        _completed_job(
            path,
            expected_schema="answer_generation_job_v1",
            expected_config_hash="config-a",
            expected_variant="full",
            expected_query_task_id="query-1",
            expected_prompt_sha256="prompt-a",
        )
        is not None
    )
    assert (
        _completed_job(
            path,
            expected_schema="answer_generation_job_v1",
            expected_config_hash="config-b",
        )
        is None
    )
    assert (
        _completed_job(
            path,
            expected_schema="answer_generation_job_v1",
            expected_prompt_sha256="prompt-b",
        )
        is None
    )


def test_interrupted_error_job_preserves_provider_call_accounting(
    tmp_path: Path,
) -> None:
    job_path = (
        tmp_path
        / "jobs"
        / "generation"
        / "full"
        / "query-1.json"
    )
    job_path.parent.mkdir(parents=True)
    job_path.write_text(
        json.dumps(
            {
                "schema_version": "answer_generation_job_v1",
                "status": "error",
                "variant": "full",
                "sample_id": "sample-1",
                "query_task_id": "query-1",
                "provider_call_id": "backbone-generate-000001",
                "provider_call_uid": "run/worker/backbone/call-1",
                "call_item_uid": "run/worker/backbone/call-1/0",
                "logical_call_item_uid": "config/generation/full/query-1",
                "prompt_tokens": None,
                "completion_tokens": None,
                "latency_ms": 12.5,
                "error_type": "RuntimeError",
            }
        ),
        encoding="utf-8",
    )

    rows = _provider_records_from_jobs(tmp_path)

    assert len(rows) == 1
    assert rows[0]["call_item_uid"] == "run/worker/backbone/call-1/0"
    assert rows[0]["logical_call_item_uid"] == "config/generation/full/query-1"
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["prompt_tokens"] is None


def test_baseline_import_supports_qa_uid_and_nested_usage(tmp_path: Path) -> None:
    source = tmp_path / "mem0.jsonl"
    source.write_text(
        json.dumps(
            {
                "qa_uid": "conv-1_qa_0",
                "sample_id": "conv-1",
                "question": "Question?",
                "gold_answer": "Gold",
                "prediction": "Answer",
                "model_name": "gpt-4o-mini",
                "response": {
                    "usage": {
                        "input_tokens": 123,
                        "output_tokens": 7,
                        "runtime_seconds": 1.25,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"

    manifest = import_baseline_answers(
        source,
        output_path=output,
        method="mem0_saved",
    )
    row = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["methods"] == ["mem0_saved"]
    assert row["query_task_id"] == "conv-1_qa_0"
    assert row["method"] == "mem0_saved"
    assert row["prompt_tokens"] == 123
    assert row["completion_tokens"] == 7
    assert row["latency_ms"] == 1250.0
    assert row["source_file_sha256"] == manifest["source_sha256"]
