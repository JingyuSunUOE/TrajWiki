from __future__ import annotations

import json
from pathlib import Path

from trajpatch.analysis.context_cost import estimate_context_tokens, select_ranked_rows_with_budget
from trajpatch.analysis.direct_retrieval import rank_direct_trajectories
from trajpatch.analysis.failure_attribution import _rank_direct_trajectories
from trajpatch.analysis.gold_labels import build_gold_label_rows, build_memory_index
from trajpatch.analysis.offline_ablation import (
    _full_rows,
    _snapshot_depth_rows,
    build_offline_ablation_rows,
)
from trajpatch.storage.database import create_schema
from trajpatch.storage.models import (
    ClaimRecord,
    EpisodicMemorySnapshot,
    RawMessageRecord,
    TrajectoryRecord,
    WikiPageRecord,
)


def _toy_memory_db(path: Path) -> Path:
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
                metadata_json={
                    "retrieval_summary_text": "Alice prefers tea.",
                    "exact_terms_v2": ["tea", "Alice"],
                    "entity_keys": ["alice"],
                },
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
                metadata_json={"exact_terms_v2": ["tea"], "facets_v2": ["preference"]},
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
        session.commit()
    finally:
        session.close()
    return path


def _sample_row() -> dict:
    return {
        "sample_id": "conv-1",
        "query_task_id": "conv-1_qa_0",
        "question": "What does Alice prefer?",
        "gold_answer": "Tea",
        "retrieval_event_id": "ret-1",
        "metrics": {"F1": 1.0, "judge_acc": 1.0},
        "metadata": {"query_metadata": {"gold_evidence_refs": ["D1:1"]}},
    }


def test_gold_label_resolver_maps_refs_to_memory_objects(tmp_path: Path) -> None:
    memory_index = build_memory_index(_toy_memory_db(tmp_path / "toy.sqlite"))
    rows = build_gold_label_rows([_sample_row()], memory_index)

    assert rows == [
            {
                "schema_version": "gold_labels_v2",
                "contains_sensitive_text": True,
                "sample_id": "conv-1",
            "query_task_id": "conv-1_qa_0",
            "question": "What does Alice prefer?",
            "gold_answer": "Tea",
            "gold_source_refs": ["D1:1"],
            "gold_source_message_ids": ["msg-1"],
            "gold_trajectory_ids": ["traj-1"],
            "gold_page_ids": ["page-1"],
            "gold_snapshot_ids": ["snap-1"],
            "gold_ref_count": 1,
            "gold_trajectory_count": 1,
        }
    ]


def test_direct_trajectory_ranking_helper_matches_failure_attribution_wrapper(tmp_path: Path) -> None:
    memory_index = build_memory_index(_toy_memory_db(tmp_path / "toy.sqlite"))
    kwargs = {
        "sample_id": "conv-1",
        "question": "What does Alice prefer?",
        "query_entities": ["Alice"],
        "query_facets": {"tags": ["preference"], "values": ["tea"]},
        "query_shape": {"item_family": "preference", "list_like": False, "count_like": False},
        "sample_to_trajectories": memory_index["sample_to_trajectories"],
        "trajectory_metadata": memory_index["trajectory_metadata"],
        "claims_by_trajectory": memory_index["claims_by_trajectory"],
        "trajectory_refs": memory_index["trajectory_refs"],
        "trajectory_lengths": memory_index["trajectory_lengths"],
    }

    shared = rank_direct_trajectories(**kwargs)
    wrapped = _rank_direct_trajectories(**kwargs)

    assert wrapped == shared
    assert shared[0] == ["traj-1"]
    assert shared[2][0]["source_refs"] == ["D1:1"]


def test_token_estimator_and_budget_selector_are_stable() -> None:
    assert estimate_context_tokens("") == 0
    assert estimate_context_tokens("one two three") == 4

    rows = [
        {"rank": 1, "item_id": "a", "estimated_tokens": 3},
        {"rank": 2, "item_id": "b", "estimated_tokens": 4},
        {"rank": 3, "item_id": "c", "estimated_tokens": 1},
    ]
    selected = select_ranked_rows_with_budget(rows, budget_tokens=7)
    assert [row["item_id"] for row in selected] == ["a", "b"]
    assert sum(row["estimated_tokens"] for row in selected) <= 7

    too_small = select_ranked_rows_with_budget(rows, budget_tokens=2)
    assert too_small == []


def test_offline_variants_cover_toy_gold_evidence(tmp_path: Path) -> None:
    memory_index = build_memory_index(_toy_memory_db(tmp_path / "toy.sqlite"))
    sample_rows = [_sample_row()]
    gold_rows = build_gold_label_rows(sample_rows, memory_index)
    retrieval_events = {
        "ret-1": {
            "id": "ret-1",
            "page_ids": ["page-1"],
            "trajectory_ids": ["traj-1"],
            "snapshot_ids": ["snap-1"],
            "expanded_snapshot_ids": ["snap-1"],
            "source_message_ids": ["msg-1"],
            "source_refs": ["D1:1"],
            "metadata": {
                "ablation_page_ranked_rows_v1": [
                    {
                        "rank": 1,
                        "item_id": "page-1",
                        "item_type": "wiki_page",
                        "page_id": "page-1",
                        "linked_trajectory_ids": ["traj-1"],
                        "source_refs": ["D1:1"],
                        "estimated_tokens": 5,
                    }
                ]
            },
        }
    }

    rows = build_offline_ablation_rows(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        retrieval_events=retrieval_events,
        memory_index=memory_index,
        variants=["full", "no_wiki_direct", "wiki_only", "flat_raw", "snapshot_m1"],
        budgets=[200],
        rank_cutoffs=[1],
    )
    by_variant = {row["variant"]: row for row in rows}

    assert set(by_variant) == {"full", "no_wiki_direct", "wiki_only", "flat_raw", "snapshot_m1"}
    assert all(
        row["gold_ref_coverage"] == 1.0
        for variant, row in by_variant.items()
        if variant != "wiki_only"
    )
    assert by_variant["wiki_only"]["gold_ref_coverage"] is None
    assert by_variant["wiki_only"]["wiki_linked_gold_trajectory_coverage"] == 1.0
    assert by_variant["full"]["observed_answer_available"] is True
    assert by_variant["wiki_only"]["observed_answer_available"] is False


def test_gold_label_artifact_rows_are_json_serializable(tmp_path: Path) -> None:
    memory_index = build_memory_index(_toy_memory_db(tmp_path / "toy.sqlite"))
    rows = build_gold_label_rows([_sample_row()], memory_index)

    assert json.loads(json.dumps(rows[0]))["gold_source_refs"] == ["D1:1"]


def test_snapshot_depth_cutoff_is_applied_to_trajectories_before_expansion() -> None:
    memory_index = {
        "trajectory_to_snapshots": {
            "traj-1": ["s1", "s2"],
            "traj-2": ["s3", "s4"],
        },
        "snapshot_refs": {
            "s1": {"D1:1"},
            "s2": {"D1:2"},
            "s3": {"D1:3"},
            "s4": {"D1:4"},
        },
        "snapshot_texts": {snapshot_id: snapshot_id for snapshot_id in ["s1", "s2", "s3", "s4"]},
    }
    rows, _ = _snapshot_depth_rows(
        {"trajectory_ids": ["traj-1", "traj-2"]},
        memory_index,
        depth=2,
    )

    first_trajectory_rows = [row for row in rows if row["trajectory_rank"] <= 1]
    assert [row["item_id"] for row in first_trajectory_rows] == ["s1", "s2"]


def test_full_variant_coverage_uses_only_final_context_sources() -> None:
    rows, _ = _full_rows(
        {
            "trajectory_ids": ["traj-1"],
            "expanded_snapshot_ids": ["snap-1"],
            "source_message_ids": ["msg-kept"],
            "metadata": {"trajectory_candidate_input_ids": ["traj-1"]},
        },
        {
            "trajectory_to_snapshots": {"traj-1": ["snap-1"]},
            "snapshot_source_message_ids": {
                "snap-1": ["msg-kept", "msg-compacted"]
            },
            "snapshot_texts": {"snap-1": "snapshot summary"},
            "raw_messages_by_id": {
                "msg-kept": {"source_ref": "D1:1", "content": "kept evidence"},
                "msg-compacted": {
                    "source_ref": "D1:2",
                    "content": "compacted evidence",
                },
            },
        },
    )

    assert rows[0]["source_message_ids"] == ["msg-kept"]
    assert rows[0]["source_refs"] == ["D1:1"]
