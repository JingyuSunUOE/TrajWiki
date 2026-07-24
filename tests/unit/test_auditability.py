from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from trajpatch.analysis.auditability import (
    _conflict_obsolete_rows,
    build_answer_context_claim_rows,
    build_answer_support_rows,
    build_auditability_rows,
    build_claim_lifecycle_rows,
    load_audit_labels,
)


def _toy_memory_index() -> dict:
    return {
        "raw_message_ids_by_ref": {"D1:1": {"m1"}, "D1:2": {"m2"}},
        "raw_messages_by_id": {
            "m1": {"id": "m1", "sample_id": "sample-1", "source_ref": "D1:1", "content": "Alice loves tea."},
            "m2": {"id": "m2", "sample_id": "sample-1", "source_ref": "D1:2", "content": "Bob likes coffee."},
        },
        "sample_to_trajectories": {"sample-1": {"traj-1"}},
        "trajectory_refs": {"traj-1": {"D1:1"}},
        "sample_raw_messages": {
            "sample-1": [
                {"id": "m1", "source_ref": "D1:1"},
                {"id": "m2", "source_ref": "D1:2"},
            ]
        },
        "page_to_trajectory_ids": {"page-1": ["traj-1"]},
        "claims_by_snapshot": {
            "snap-1": [
                {
                    "id": "claim-rec-1",
                    "claim_id": "claim-1",
                    "snapshot_id": "snap-1",
                    "trajectory_id": "traj-1",
                    "text": "Alice loves tea.",
                    "status": "active",
                    "source_refs": {"D1:1"},
                    "source_message_ids": ["m1"],
                    "metadata": {},
                },
                {
                    "id": "claim-rec-2",
                    "claim_id": "claim-2",
                    "snapshot_id": "snap-1",
                    "trajectory_id": "traj-1",
                    "text": "Alice hates tea.",
                    "status": "deprecated",
                    "source_refs": {"D1:2"},
                    "source_message_ids": ["m2"],
                    "metadata": {},
                },
            ]
        },
    }


def test_answer_support_rows_parse_structured_items_and_invalid_refs() -> None:
    sample_rows = [
        {
            "sample_id": "sample-1",
            "query_task_id": "query-1",
            "answer_text": "Tea",
            "metadata": {
                "answer_metadata": {
                    "answer_synthesis_supporting_refs": ["D1:1"],
                    "answer_supported_list_item_refs": {"Tea": ["D1:1"]},
                    "bridge_finalization_source_refs": ["D1:2"],
                    "invalid_supporting_refs": ["D9:9"],
                }
            },
        }
    ]
    rows = build_answer_support_rows(sample_rows, _toy_memory_index())

    final = next(row for row in rows if row["item_kind"] == "final_answer")
    assert final["support_status"] == "invalid_source_ref"
    assert final["contains_sensitive_text"] is True
    assert final["source_refs"] == ["D1:1", "D1:2"]
    assert final["invalid_supporting_refs"] == ["D9:9"]
    assert any(row["item_kind"] == "supported_list_item" and row["source_message_ids"] == ["m1"] for row in rows)


def test_context_claim_rows_mark_deprecated_claims_suppressed() -> None:
    sample_rows = [{"sample_id": "sample-1", "query_task_id": "query-1", "retrieval_event_id": "ret-1"}]
    retrieval_events = {"ret-1": {"id": "ret-1", "expanded_snapshot_ids": ["snap-1"]}}
    rows = build_answer_context_claim_rows(sample_rows, _toy_memory_index(), retrieval_events)

    deprecated = next(row for row in rows if row["claim_id"] == "claim-2")
    assert deprecated["contains_sensitive_text"] is True
    assert deprecated["kept_in_answer_context"] is False
    assert deprecated["suppressed_reason"] == "deprecated"
    active = next(row for row in rows if row["claim_id"] == "claim-1")
    assert active["kept_in_answer_context"] is True


def test_auditability_rows_flag_no_gold_overlap_as_unsupported() -> None:
    sample_rows = [
        {
            "sample_id": "sample-1",
            "query_task_id": "query-1",
            "retrieval_event_id": "ret-1",
            "answer_text": "Coffee",
            "judge_verdict": "incorrect",
            "metrics": {"F1": 0.0, "judge_acc": 0.0},
            "retrieval_source_refs": ["D1:1"],
            "metadata": {"answer_metadata": {"answer_synthesis_supporting_refs": ["D1:2"]}},
        }
    ]
    gold_rows = [
        {
            "sample_id": "sample-1",
            "query_task_id": "query-1",
            "gold_source_refs": ["D1:1"],
            "gold_page_ids": ["page-1"],
            "gold_trajectory_ids": ["traj-1"],
        }
    ]
    retrieval_events = {
        "ret-1": {
            "id": "ret-1",
            "page_ids": ["page-1"],
            "trajectory_ids": ["traj-1"],
            "source_refs": ["D1:1"],
            "expanded_refs": ["D1:1"],
            "metadata": {},
        }
    }
    memory_index = _toy_memory_index()
    support_rows = build_answer_support_rows(sample_rows, memory_index)
    claim_rows = build_answer_context_claim_rows(
        sample_rows,
        memory_index,
        {"ret-1": {"id": "ret-1", "expanded_snapshot_ids": ["snap-1"], "metadata": {}}},
    )

    rows = build_auditability_rows(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        memory_index=memory_index,
        retrieval_events=retrieval_events,
        answer_support_rows=support_rows,
        answer_context_claim_rows=claim_rows,
    )
    assert rows[0]["source_supported_proxy"] is False
    assert rows[0]["unsupported_answer_risk"] is True
    assert rows[0]["unsupported_reason"] == "support_refs_no_gold_overlap"
    assert rows[0]["stored_gold_ref_coverage"] == 1.0
    assert rows[0]["trajectory_linked_gold_ref_coverage"] == 1.0
    assert rows[0]["page_routed_gold_ref_coverage"] == 1.0
    assert rows[0]["trajectory_selected_gold_ref_coverage"] == 1.0
    assert rows[0]["snapshot_expanded_gold_ref_coverage"] == 1.0


def test_claim_lifecycle_rows_export_add_revise_deprecate(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE trajectories (id TEXT PRIMARY KEY, sample_id TEXT);
        CREATE TABLE claim_ops (
            id TEXT PRIMARY KEY,
            snapshot_id TEXT,
            trajectory_id TEXT,
            op_type TEXT,
            target_claim_id TEXT,
            new_claim_id TEXT,
            source_message_ids_json TEXT,
            rationale TEXT,
            metadata_json TEXT
        );
        """
    )
    connection.execute("INSERT INTO trajectories VALUES (?, ?)", ("traj-1", "sample-1"))
    for op_type in ["ADD", "REVISE", "DEPRECATE"]:
        connection.execute(
            "INSERT INTO claim_ops VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"op-{op_type}",
                "snap-1",
                "traj-1",
                op_type,
                "claim-old",
                "claim-new",
                '["m1"]',
                "because evidence changed",
                "{}",
            ),
        )
    connection.commit()
    connection.close()

    rows = build_claim_lifecycle_rows(db_path, _toy_memory_index())
    assert {row["op_type"] for row in rows} == {"ADD", "REVISE", "DEPRECATE"}
    assert all(row["contains_sensitive_text"] is True for row in rows)
    assert rows[0]["source_refs"] == ["D1:1"]


def test_load_audit_labels_accepts_partial_csv(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "query_task_id", "true_source_supported", "human_audit_seconds"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-1",
                "query_task_id": "query-1",
                "true_source_supported": "yes",
                "human_audit_seconds": "12.5",
            }
        )

    labels = load_audit_labels(path)
    label = labels[("sample-1", "query-1")]
    assert label["true_source_supported"] is True
    assert label["human_audit_seconds"] == 12.5


def test_conflict_and_obsolete_labels_use_matching_metric_semantics() -> None:
    audit_rows = [
        {
            "sample_id": "sample-1",
            "query_task_id": "query-1",
            "conflict_claim_visible": True,
            "obsolete_answer_risk_proxy": False,
        },
        {
            "sample_id": "sample-1",
            "query_task_id": "query-2",
            "conflict_claim_visible": False,
            "obsolete_answer_risk_proxy": True,
        },
    ]
    labels = {
        ("sample-1", "query-1"): {
            "true_conflict_required": True,
            "true_conflict_handled": True,
            "true_obsolete_required": False,
        },
        ("sample-1", "query-2"): {
            "true_conflict_required": False,
            "true_obsolete_required": True,
            "true_obsolete_handled": False,
        },
    }

    row = _conflict_obsolete_rows(
        audit_rows,
        labels,
        ["trajwiki_observed"],
    )[0]

    assert row["conflict_accuracy_if_labeled"] == 1.0
    assert row["conflict_handled_rate_if_labeled"] == 1.0
    assert row["obsolete_accuracy_if_labeled"] == 1.0
    assert row["obsolete_handled_rate_if_labeled"] == 0.0
