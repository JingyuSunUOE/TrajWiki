from __future__ import annotations

import json
from pathlib import Path

from trajpatch.analysis.cost_benefit import build_cost_query_rows, build_memory_scaling_rows
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

    assert break_even_queries(100, 25)["break_even_queries"] == 4
    no_saving = break_even_queries(100, 0)
    assert no_saving["break_even_queries"] is None
    assert no_saving["reason"] == "no_query_cost_saving"


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
    assert json.loads(json.dumps(query_rows[0]))["schema_version"] == "cost_query_v1"
