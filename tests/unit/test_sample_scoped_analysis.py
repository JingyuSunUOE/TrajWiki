from __future__ import annotations

import json
from pathlib import Path

from trajpatch.analysis.auditability import (
    build_answer_support_rows,
    build_audit_packet_rows,
)
from trajpatch.analysis.gold_labels import (
    build_gold_label_rows,
    load_or_build_gold_labels,
)
from trajpatch.analysis.memory_index import (
    build_sample_scoped_ref_indexes,
    versioned_analysis_path,
)


def _memory_index() -> dict:
    raw_messages_by_id = {
        "conv-a-m1": {
            "id": "conv-a-m1",
            "sample_id": "conv-a",
            "source_ref": "D1:1",
            "content": "Alice evidence",
        },
        "conv-b-m1": {
            "id": "conv-b-m1",
            "sample_id": "conv-b",
            "source_ref": "D1:1",
            "content": "Bob evidence",
        },
    }
    trajectory_refs = {"traj-a": {"D1:1"}, "traj-b": {"D1:1"}}
    trajectory_to_sample = {"traj-a": "conv-a", "traj-b": "conv-b"}
    return {
        "raw_messages_by_id": raw_messages_by_id,
        "raw_message_ids_by_ref": {"D1:1": {"conv-a-m1", "conv-b-m1"}},
        "sample_to_trajectories": {"conv-a": {"traj-a"}, "conv-b": {"traj-b"}},
        "trajectory_refs": trajectory_refs,
        "sample_to_pages": {"conv-a": {"page-a"}, "conv-b": {"page-b"}},
        "page_to_trajectory_ids": {"page-a": ["traj-a"], "page-b": ["traj-b"]},
        "page_types": {"page-a": "entity", "page-b": "entity"},
        "snapshot_refs": {"snap-a": {"D1:1"}, "snap-b": {"D1:1"}},
        "snapshot_to_trajectory": {"snap-a": "traj-a", "snap-b": "traj-b"},
        "trajectory_to_sample": trajectory_to_sample,
        **build_sample_scoped_ref_indexes(
            raw_messages_by_id=raw_messages_by_id,
            trajectory_refs=trajectory_refs,
            trajectory_to_sample=trajectory_to_sample,
        ),
    }


def test_gold_labels_do_not_expand_duplicate_refs_across_samples() -> None:
    rows = build_gold_label_rows(
        [
            {
                "sample_id": "conv-a",
                "query_task_id": "qa-a",
                "question": "What happened?",
                "gold_answer": "Alice evidence",
                "metadata": {"query_metadata": {"gold_evidence_refs": ["D1:1"]}},
            }
        ],
        _memory_index(),
    )

    assert rows[0]["schema_version"] == "gold_labels_v2"
    assert rows[0]["gold_source_message_ids"] == ["conv-a-m1"]
    assert rows[0]["gold_trajectory_ids"] == ["traj-a"]
    assert rows[0]["gold_page_ids"] == ["page-a"]
    assert rows[0]["gold_snapshot_ids"] == ["snap-a"]


def test_audit_rows_do_not_include_another_samples_duplicate_ref() -> None:
    sample_rows = [
        {
            "sample_id": "conv-a",
            "query_task_id": "qa-a",
            "question": "What happened?",
            "answer_text": "Alice evidence",
            "gold_answer": "Alice evidence",
            "retrieval_source_refs": ["D1:1"],
            "metadata": {
                "answer_metadata": {
                    "answer_synthesis_supporting_refs": ["D1:1"],
                }
            },
        }
    ]
    memory_index = _memory_index()
    support_rows = build_answer_support_rows(sample_rows, memory_index)
    packets = build_audit_packet_rows(
        sample_rows=sample_rows,
        memory_index=memory_index,
        retrieval_events={},
        answer_support_rows=support_rows,
        answer_context_claim_rows=[],
        packet_save_mode="compact",
    )

    assert support_rows[0]["source_message_ids"] == ["conv-a-m1"]
    assert packets[0]["schema_version"] == "audit_packet_v2"
    assert packets[0]["audit_source_count"] == 1
    assert packets[0]["source_message_previews"][0]["message_id"] == "conv-a-m1"


def test_invalid_v2_gold_cache_is_replaced_without_duplicate_rows(
    tmp_path: Path,
) -> None:
    sample_rows = [
        {
            "sample_id": "conv-a",
            "query_task_id": "qa-a",
            "question": "What happened?",
            "gold_answer": "Alice evidence",
            "metadata": {"query_metadata": {"gold_evidence_refs": ["D1:1"]}},
        }
    ]
    cache_path = tmp_path / "analysis_v2" / "gold_labels.jsonl"
    cache_path.parent.mkdir(parents=True)
    duplicate = {
        "schema_version": "gold_labels_v2",
        "sample_id": "conv-a",
        "query_task_id": "qa-a",
    }
    cache_path.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    primary_path = tmp_path / "analysis" / "gold_labels.jsonl"
    primary_path.parent.mkdir(parents=True)
    primary_path.write_text(
        json.dumps(build_gold_label_rows(sample_rows, _memory_index())[0]) + "\n",
        encoding="utf-8",
    )

    rows = load_or_build_gold_labels(tmp_path, sample_rows, _memory_index())

    assert len(rows) == 1
    persisted = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(persisted) == 1
    assert persisted[0]["gold_source_message_ids"] == ["conv-a-m1"]


def test_v2_gold_cache_without_sensitive_marker_is_rebuilt(tmp_path: Path) -> None:
    sample_rows = [
        {
            "sample_id": "conv-a",
            "query_task_id": "qa-a",
            "question": "What happened?",
            "gold_answer": "Alice evidence",
            "metadata": {"query_metadata": {"gold_evidence_refs": ["D1:1"]}},
        }
    ]
    cache_path = tmp_path / "analysis_v2" / "gold_labels.jsonl"
    cache_path.parent.mkdir(parents=True)
    stale_row = build_gold_label_rows(sample_rows, _memory_index())[0]
    stale_row.pop("contains_sensitive_text")
    cache_path.write_text(json.dumps(stale_row) + "\n", encoding="utf-8")

    rows = load_or_build_gold_labels(tmp_path, sample_rows, _memory_index())

    assert rows[0]["contains_sensitive_text"] is True
    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert persisted["contains_sensitive_text"] is True


def test_existing_v2_artifact_remains_the_rebuild_target(tmp_path: Path) -> None:
    primary = tmp_path / "analysis" / "gold_labels.jsonl"
    primary.parent.mkdir(parents=True)
    primary.write_text(
        json.dumps({"schema_version": "gold_labels_v2"}) + "\n",
        encoding="utf-8",
    )
    versioned = tmp_path / "analysis_v2" / "gold_labels.jsonl"
    versioned.parent.mkdir(parents=True)
    versioned.write_text("invalid\n", encoding="utf-8")

    assert (
        versioned_analysis_path(
            tmp_path,
            filename="gold_labels.jsonl",
            accepted_schema_versions={"gold_labels_v2"},
        )
        == versioned
    )
