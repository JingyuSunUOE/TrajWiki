from __future__ import annotations

import pytest

from trajpatch.analysis.trajectory_drift import (
    build_trajectory_drift_diagnostics,
    build_trajectory_drift_rows_from_embeddings,
    compact_query_drift_row,
    trajectory_drift_fields_for_gold_trajectories,
)


def _embedding(vector: list[float], *, model: str = "unit-embedding") -> dict[str, object]:
    norm = sum(value * value for value in vector) ** 0.5
    return {"vector": vector, "norm": norm, "model_name": model}


def test_trajectory_drift_singleton_bucket() -> None:
    rows = build_trajectory_drift_rows_from_embeddings(
        trajectory_to_sample={"t1": "sample-1"},
        trajectory_snapshot_ids_ordered={"t1": ["s1"]},
        snapshot_versions={"s1": 1},
        snapshot_embeddings={"s1": _embedding([1.0, 0.0])},
        summary_embeddings={"t1": _embedding([1.0, 0.0])},
    )

    row = rows[0]
    assert row["drift_bucket"] == "singleton"
    assert row["head_tail_cosine"] == pytest.approx(1.0)
    assert row["head_tail_distance"] == pytest.approx(0.0)
    assert row["summary_head_cosine"] == pytest.approx(1.0)
    assert row["embedding_available"] is True
    assert "vector" not in row


def test_trajectory_drift_head_tail_adjacent_and_summary_metrics() -> None:
    rows = build_trajectory_drift_rows_from_embeddings(
        trajectory_to_sample={"t1": "sample-1"},
        trajectory_snapshot_ids_ordered={"t1": ["s1", "s2", "s3"]},
        snapshot_versions={"s1": 1, "s2": 2, "s3": 3},
        snapshot_embeddings={
            "s1": _embedding([1.0, 0.0]),
            "s2": _embedding([0.8, 0.6]),
            "s3": _embedding([0.0, 1.0]),
        },
        summary_embeddings={"t1": _embedding([0.0, 1.0])},
    )

    row = rows[0]
    assert row["drift_bucket"] == "possible_drift"
    assert row["head_snapshot_id"] == "s1"
    assert row["tail_snapshot_id"] == "s3"
    assert row["head_tail_cosine"] == pytest.approx(0.0)
    assert row["adjacent_mean_cosine"] == pytest.approx(0.7)
    assert row["adjacent_min_cosine"] == pytest.approx(0.6)
    assert row["adjacent_max_distance"] == pytest.approx(0.4)
    assert row["summary_head_cosine"] == pytest.approx(0.0)
    assert row["summary_tail_cosine"] == pytest.approx(1.0)
    assert row["summary_centroid_cosine"] is not None


def test_trajectory_drift_missing_snapshot_embedding_does_not_raise() -> None:
    rows = build_trajectory_drift_rows_from_embeddings(
        trajectory_to_sample={"t1": "sample-1"},
        trajectory_snapshot_ids_ordered={"t1": ["s1", "s2"]},
        snapshot_versions={"s1": 1, "s2": 2},
        snapshot_embeddings={"s1": _embedding([1.0, 0.0])},
        summary_embeddings={},
    )

    row = rows[0]
    assert row["drift_bucket"] == "missing_embedding"
    assert row["embedding_available"] is False
    assert row["head_tail_cosine"] is None
    assert row["low_similarity_update_pair_count"] == 0


def test_trajectory_drift_low_similarity_pairs_are_compact() -> None:
    rows = build_trajectory_drift_rows_from_embeddings(
        trajectory_to_sample={"t1": "sample-1"},
        trajectory_snapshot_ids_ordered={"t1": ["s1", "s2", "s3"]},
        snapshot_versions={"s1": 1, "s2": 2, "s3": 3},
        snapshot_embeddings={
            "s1": _embedding([1.0, 0.0]),
            "s2": _embedding([0.0, 1.0]),
            "s3": _embedding([0.0, 0.9]),
        },
        summary_embeddings={},
    )

    row = rows[0]
    assert row["low_similarity_update_pair_count"] == 1
    assert row["low_similarity_update_pairs"] == [
        {
            "from_snapshot_id": "s1",
            "to_snapshot_id": "s2",
            "from_version": 1,
            "to_version": 2,
            "cosine": pytest.approx(0.0),
        }
    ]
    assert "raw_text" not in row["low_similarity_update_pairs"][0]


def test_trajectory_drift_gold_query_fields_and_diagnostics() -> None:
    rows = build_trajectory_drift_rows_from_embeddings(
        trajectory_to_sample={"t1": "sample-1", "t2": "sample-1"},
        trajectory_snapshot_ids_ordered={"t1": ["s1", "s2"], "t2": ["s3", "s4"]},
        snapshot_versions={"s1": 1, "s2": 2, "s3": 1, "s4": 2},
        snapshot_embeddings={
            "s1": _embedding([1.0, 0.0]),
            "s2": _embedding([0.8, 0.6]),
            "s3": _embedding([1.0, 0.0]),
            "s4": _embedding([0.0, 1.0]),
        },
        summary_embeddings={"t1": _embedding([0.8, 0.6]), "t2": _embedding([0.0, 1.0])},
    )
    by_id = {row["trajectory_id"]: row for row in rows}

    fields = trajectory_drift_fields_for_gold_trajectories(["t1", "t2"], by_id)
    assert fields["gold_trajectory_head_tail_cosine_min"] == pytest.approx(0.0)
    assert fields["gold_trajectory_possible_drift_count"] == 1
    assert fields["trajectory_drift_risk_observed"] is True

    query_row = {
        "sample_id": "sample-1",
        "query_task_id": "q1",
        "question": "Question?",
        "judge_verdict": "incorrect",
        "reason": "coarse_retrieval_miss",
        "gold_trajectory_ids": ["t1", "t2"],
        **fields,
    }
    diagnostics = build_trajectory_drift_diagnostics(
        trajectory_drift_rows=rows,
        query_rows=[query_row],
        failed_rows=[query_row],
    )
    assert diagnostics["trajectory_count"] == 2
    assert diagnostics["drift_bucket_counts"]["possible_drift"] == 1
    assert diagnostics["drift_by_query_verdict"]["incorrect"]["trajectory_drift_risk_count"] == 1
    assert diagnostics["drift_by_failure_reason"]["coarse_retrieval_miss"]["query_count"] == 1

    compact = compact_query_drift_row(query_row)
    assert compact["trajectory_drift_risk_observed"] is True
    assert "question" in compact
    assert "vector" not in compact

