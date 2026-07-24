from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trajpatch.analysis.query_scope import (
    filter_query_rows,
    load_query_scope,
    scoped_analysis_dir,
    validate_scope_against_rows,
)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "answer_ablation_sampling_v1",
                "sample_size_selected": 2,
                "selected_queries": [
                    {
                        "sample_id": "conv-1",
                        "query_task_id": "conv-1_qa_0",
                        "stratum": "ordinary",
                    },
                    {
                        "sample_id": "conv-2",
                        "query_task_id": "conv-2_qa_1",
                        "stratum": "update_sensitive",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_query_scope_validates_filters_and_isolates_output(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "answer_ablation_test"
    experiment_dir.mkdir()
    scope = load_query_scope(_manifest(experiment_dir / "sampling_manifest.json"))
    assert scope is not None
    rows = [
        {"sample_id": "conv-1", "query_task_id": "conv-1_qa_0"},
        {"sample_id": "conv-2", "query_task_id": "conv-2_qa_1"},
        {"sample_id": "conv-3", "query_task_id": "conv-3_qa_0"},
    ]
    validate_scope_against_rows(scope, rows)
    assert len(filter_query_rows(rows, scope)) == 2
    assert scoped_analysis_dir(tmp_path, scope) == (
        experiment_dir / "offline_analysis"
    )
    assert scope.metadata()["scoped_query_count"] == 2


def test_query_scope_rejects_duplicate_or_misaligned_queries(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path / "sampling_manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["selected_queries"].append(dict(payload["selected_queries"][0]))
    payload["sample_size_selected"] = 3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate query_task_id"):
        load_query_scope(manifest_path)

    scope = load_query_scope(_manifest(manifest_path))
    assert scope is not None
    with pytest.raises(ValueError, match="alignment mismatch"):
        validate_scope_against_rows(
            scope,
            [
                {"sample_id": "wrong", "query_task_id": "conv-1_qa_0"},
                {"sample_id": "conv-2", "query_task_id": "conv-2_qa_1"},
            ],
        )


def test_query_scope_rejects_a_different_source_run_or_details_hash(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "run-a"
    other_run = tmp_path / "run-b"
    source_run.mkdir()
    other_run.mkdir()
    source_details = source_run / "details.json"
    source_details.write_text('{"run":"a"}', encoding="utf-8")
    (other_run / "details.json").write_text('{"run":"b"}', encoding="utf-8")
    manifest_path = _manifest(tmp_path / "sampling_manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_run_dir"] = str(source_run)
    payload["source_details_sha256"] = hashlib.sha256(
        source_details.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    scope = load_query_scope(manifest_path)
    assert scope is not None
    rows = [
        {"sample_id": "conv-1", "query_task_id": "conv-1_qa_0"},
        {"sample_id": "conv-2", "query_task_id": "conv-2_qa_1"},
    ]

    validate_scope_against_rows(scope, rows, run_dir=source_run)
    with pytest.raises(ValueError, match="different source run"):
        validate_scope_against_rows(scope, rows, run_dir=other_run)

    source_details.write_text('{"run":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="details hash"):
        validate_scope_against_rows(scope, rows, run_dir=source_run)
