from __future__ import annotations

import json
from pathlib import Path

from trajpatch.analysis.cost_benefit import (
    _augment_cost_rows_with_dollars,
    _cost_phase_summary,
    _preferred_analysis_artifact,
    build_cost_query_rows,
    build_cost_reconciliation,
    build_memory_scaling_rows,
    compact_cost_call_row,
)
from trajpatch.analysis.cost_model import (
    break_even_queries,
    cost_phase_for_task,
    deployment_scope_for_task,
    estimate_dollar_cost,
    reusable_scope_for_task,
)
from trajpatch.storage.database import create_schema
from trajpatch.storage.models import (
    ClaimOpRecord,
    ClaimRecord,
    EpisodicMemorySnapshot,
    RawMessageRecord,
    RetrievalEvent,
    TrajectoryRecord,
    WikiPageRecord,
)


def _toy_cost_db(path: Path) -> Path:
    session_factory = create_schema(path)
    session = session_factory()
    try:
        session.add(
            RawMessageRecord(
                id="msg-1",
                sample_id="conv-1",
                dataset_name="locomo",
                turn_index=1,
                role="user",
                speaker_name="Alice",
                content="Alice prefers tea.",
                source_ref="D1:1",
                occurred_at="2024-01-01",
                metadata_json={},
            )
        )
        session.add(
            TrajectoryRecord(
                id="traj-1",
                sample_id="conv-1",
                dataset_name="locomo",
                label="Alice tea preference",
                latest_snapshot_id="snap-1",
                snapshot_count=1,
                metadata_json={},
            )
        )
        session.add(
            EpisodicMemorySnapshot(
                id="snap-1",
                trajectory_id="traj-1",
                version=1,
                timestamp="2024-01-01",
                links_json=["msg-1"],
                summary_content="Alice prefers tea.",
                context="Preference memory.",
                keywords_json=["alice", "tea"],
                status_flags_json=[],
                semantic_text="Alice prefers tea.",
                raw_text="Alice prefers tea.",
                metadata_json={},
            )
        )
        session.add(
            ClaimRecord(
                id="claim-1",
                snapshot_id="snap-1",
                trajectory_id="traj-1",
                claim_id="c1",
                text="Alice prefers tea.",
                status="active",
                source_message_ids_json=["msg-1"],
                metadata_json={},
            )
        )
        session.add(
            ClaimOpRecord(
                id="op-1",
                snapshot_id="snap-1",
                trajectory_id="traj-1",
                op_type="ADD",
                target_claim_id="c1",
                rationale="new memory",
                source_message_ids_json=["msg-1"],
                metadata_json={},
            )
        )
        session.add(
            WikiPageRecord(
                id="page-1",
                sample_id="conv-1",
                dataset_name="locomo",
                page_type="topic",
                title="Alice",
                slug="alice",
                markdown_text="Alice prefers tea.",
                keywords_json=["alice", "tea"],
                trajectory_ids_json=["traj-1"],
                linked_page_ids_json=[],
                entity_names_json=["Alice"],
                metadata_json={},
            )
        )
        session.add(
            RetrievalEvent(
                id="ret-1",
                sample_id="conv-1",
                query_text="What does Alice prefer?",
                query_embedding_json=[],
                top_t_pages=1,
                top_k=1,
                snapshot_budget=2,
                page_ids_json=["page-1"],
                trajectory_ids_json=["traj-1"],
                snapshot_ids_json=["snap-1"],
                expanded_snapshot_ids_json=["snap-1"],
                source_message_ids_json=["msg-1"],
                latency_ms=1.0,
                metadata_json={
                    "trajectory_candidate_input_ids": ["traj-1"],
                    "answer_context_token_breakdown_v1": {"total": 12},
                },
            )
        )
        session.commit()
    finally:
        session.close()
    return path


def test_cost_phase_scope_mapping() -> None:
    assert cost_phase_for_task("episodic_extract") == "memory_build"
    assert cost_phase_for_task("wiki_page_rerank") == "query_time"
    assert cost_phase_for_task("answer_generation") == "answer_generation"
    assert cost_phase_for_task("locomo_judge") == "evaluation_only"
    assert cost_phase_for_task("semantic_metric_extract") == "evaluation_only"
    assert cost_phase_for_task("answer_generation_repair") == "repair_validation"
    assert deployment_scope_for_task("locomo_judge") == "benchmark_only"
    assert reusable_scope_for_task("episodic_extract") == "upfront_reusable"
    assert reusable_scope_for_task("wiki_page_rerank") == "per_query"


def test_compact_cost_call_preserves_missing_provider_usage() -> None:
    row = compact_cost_call_row(
        {
            "role": "backbone",
            "task": "answer_generation",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "metadata": {
                "provider_prompt_usage_available": False,
                "provider_completion_usage_available": False,
            },
        }
    )

    assert row["prompt_tokens"] is None
    assert row["completion_tokens"] is None
    assert row["total_tokens"] is None
    assert row["provider_usage_available"] is False


def test_dollar_estimator_and_break_even() -> None:
    missing = estimate_dollar_cost(1000, 500, "gpt-4o-mini", None)
    assert missing["total_cost"] is None
    assert missing["price_available"] is False

    priced = estimate_dollar_cost(
        1_000_000,
        2_000_000,
        "gpt-4o-mini",
        {
            "currency": "USD",
            "models": {
                "gpt-4o-mini": {
                    "input_per_1m_tokens": 0.1,
                    "output_per_1m_tokens": 0.2,
                }
            },
        },
    )
    assert priced["total_cost"] == 0.5
    missing_usage = estimate_dollar_cost(
        None,
        500,
        "gpt-4o-mini",
        {
            "currency": "USD",
            "models": {
                "gpt-4o-mini": {
                    "input_per_1m_tokens": 0.1,
                    "output_per_1m_tokens": 0.2,
                }
            },
        },
    )
    assert missing_usage["total_cost"] is None
    assert missing_usage["price_available"] is True
    assert missing_usage["usage_available"] is False

    assert break_even_queries(100, 25)["break_even_queries"] == 4
    no_saving = break_even_queries(100, 0)
    assert no_saving["break_even_queries"] is None
    assert no_saving["reason"] == "no_query_cost_saving"


def test_cost_summary_does_not_report_partial_dollar_total() -> None:
    price_config = {
        "currency": "USD",
        "models": {
            "gpt-4o-mini": {
                "input_per_1m_tokens": 0.1,
                "output_per_1m_tokens": 0.2,
            }
        },
    }
    rows = _augment_cost_rows_with_dollars(
        [
            {
                "cost_phase": "query_time",
                "deployment_scope": "deployment",
                "reusable_scope": "per_query",
                "model": "gpt-4o-mini",
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "provider_usage_available": True,
                "latency_ms": 2.0,
            },
            {
                "cost_phase": "query_time",
                "deployment_scope": "deployment",
                "reusable_scope": "per_query",
                "model": "gpt-4o-mini",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "provider_usage_available": False,
                "latency_ms": 3.0,
            },
        ],
        price_config,
    )
    summary = _cost_phase_summary(rows)

    assert rows[1]["dollar_cost"] is None
    assert summary[0]["provider_usage_missing_rows"] == 1
    assert summary[0]["dollar_cost"] is None
    assert summary[0]["dollar_total_complete"] is False


def test_memory_scaling_and_query_cost_rows(tmp_path: Path) -> None:
    database_path = _toy_cost_db(tmp_path / "toy.sqlite")
    memory_rows = build_memory_scaling_rows(database_path)

    assert memory_rows[0]["sample_id"] == "conv-1"
    assert memory_rows[0]["raw_message_count"] == 1
    assert memory_rows[0]["trajectory_count"] == 1
    assert memory_rows[0]["snapshot_count"] == 1
    assert memory_rows[0]["claim_count"] == 1
    assert memory_rows[0]["claim_op_count"] == 1
    assert memory_rows[0]["wiki_page_count"] == 1

    sample_rows = [
        {
            "sample_id": "conv-1",
            "query_task_id": "conv-1_qa_0",
            "question": "What does Alice prefer?",
            "answer_prompt": "Question: What does Alice prefer?\nContext: Alice prefers tea.",
            "retrieval_event_id": "ret-1",
            "retrieval_source_refs": ["D1:1"],
            "metrics": {"F1": 1.0, "judge_acc": 1.0},
            "tokens": {"backbone_prompt_tokens": 20, "backbone_completion_tokens": 3},
            "metadata": {"answer_prompt_tokens": 20, "answer_completion_tokens": 3},
        }
    ]
    query_rows = build_cost_query_rows(sample_rows, database_path)

    assert query_rows[0]["trajwiki_final_context_tokens"] == 12
    assert query_rows[0]["wiki_routed_candidate_universe"] == 1
    assert query_rows[0]["direct_candidate_universe"] == 1
    assert query_rows[0]["answer_prompt_tokens"] == 20
    assert query_rows[0]["unsupported_answer_proxy"] is False
    assert json.loads(json.dumps(query_rows[0]))["schema_version"] == "cost_query_v2"


def test_cost_reconciliation_detects_duplicate_and_missing_uids() -> None:
    raw_records = [
        {"prompt_tokens": 10, "completion_tokens": 2},
        {"prompt_tokens": 4, "completion_tokens": 1},
    ]
    compact_rows = [
        {
            "provider_call_uid": "run/0/backbone/call-1",
            "call_item_uid": "run/0/backbone/call-1/0",
            "worker_id": "0",
            "role": "backbone",
            "task": "answer_generation",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "latency_ms": 5.0,
            "sample_id": "conv-1",
            "query_task_id": "q1",
        },
        {
            "provider_call_uid": "run/1/backbone/call-1",
            "call_item_uid": "run/1/backbone/call-1/0",
            "worker_id": "1",
            "role": "backbone",
            "task": "answer_generation",
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5,
            "latency_ms": 2.0,
            "sample_id": "conv-2",
            "query_task_id": "q2",
        },
    ]

    summary, rows = build_cost_reconciliation(raw_records, compact_rows)

    assert summary["reconciled"] is True
    assert summary["provider_call_count"] == 2
    assert summary["duplicate_call_item_uid_count"] == 0
    assert len(rows) == 2


def test_cost_analysis_prefers_rebuilt_v2_artifacts(tmp_path: Path) -> None:
    primary = tmp_path / "analysis" / "offline_ablation_rows.jsonl"
    primary.parent.mkdir(parents=True)
    primary.write_text("{}\n", encoding="utf-8")

    assert (
        _preferred_analysis_artifact(tmp_path, "offline_ablation_rows.jsonl") == primary
    )

    versioned = tmp_path / "analysis_v2" / "offline_ablation_rows.jsonl"
    versioned.parent.mkdir(parents=True)
    versioned.write_text("{}\n", encoding="utf-8")

    assert (
        _preferred_analysis_artifact(tmp_path, "offline_ablation_rows.jsonl")
        == versioned
    )
