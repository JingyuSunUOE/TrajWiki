from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trajpatch.experiments.bundle import (
    package_rebuttal_bundle,
    validate_rebuttal_bundle,
)


def test_rebuttal_bundle_is_portable_checksummed_and_excludes_env(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname='toy'\nversion='0.0.1'\n",
        encoding="utf-8",
    )
    (repository / "README.md").write_text("toy\n", encoding="utf-8")
    (repository / "src").mkdir()
    (repository / "src" / "toy.py").write_text("VALUE = 1\n", encoding="utf-8")

    source_run = tmp_path / "source_run"
    source_run.mkdir()
    (source_run / "details.json").write_text("{}\n", encoding="utf-8")
    (source_run / "summary.json").write_text("{}\n", encoding="utf-8")
    database = source_run / "trajpatch.sqlite"
    database.write_bytes(b"SQLite format 3\x00toy")

    experiment = source_run / "rebuttal_experiments" / "answer_ablation_toy"
    experiment.mkdir(parents=True)
    manifest = {
        "status": "complete",
        "config_hash": "toy-hash",
        "source_run_dir": str(source_run),
        "source_database_path": str(database),
        "source_details_sha256": hashlib.sha256(
            (source_run / "details.json").read_bytes()
        ).hexdigest(),
        "source_database_sha256": hashlib.sha256(
            database.read_bytes()
        ).hexdigest(),
        "external_baseline_attachments": [],
    }
    (experiment / "experiment_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (experiment / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "sampling_status": "post_hoc_nested_extension",
                "selected_queries": [],
            }
        ),
        encoding="utf-8",
    )
    for name in [
        "integrity_report.json",
        "validation_report.json",
        "provider_call_rows.jsonl",
        "variant_context_rows.jsonl",
    ]:
        (experiment / name).write_text("{}\n", encoding="utf-8")

    workflow = tmp_path / "workflow"
    workflow.mkdir()
    (workflow / "application.log").write_text("complete\n", encoding="utf-8")
    (workflow / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    bundle_path = tmp_path / "verification.tar.zst"

    packaged = package_rebuttal_bundle(
        experiment,
        output_path=bundle_path,
        workflow_logs_path=workflow,
        repository_path=repository,
    )
    validation = validate_rebuttal_bundle(bundle_path)

    assert packaged["bundle_path"] == str(bundle_path)
    assert bundle_path.stat().st_mode & 0o777 == 0o600
    assert validation["error_count"] == 0
    assert validation["contains_sensitive_text"] is True
    assert packaged["bundle_sha256"] == validation["bundle_sha256"]

    original_details = (source_run / "details.json").read_bytes()
    (source_run / "details.json").write_text(
        '{"mutated": true}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_details_sha256"):
        package_rebuttal_bundle(
            experiment,
            output_path=tmp_path / "mutated-source.tar.zst",
            repository_path=repository,
        )
    (source_run / "details.json").write_bytes(original_details)

    (experiment / "validation_report.json").write_text(
        json.dumps({"error_count": 1, "errors": ["corrupt"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"validation_report\.json"):
        package_rebuttal_bundle(
            experiment,
            output_path=tmp_path / "invalid.tar.zst",
            repository_path=repository,
        )

    (experiment / "validation_report.json").write_text(
        json.dumps({"error_count": 0, "errors": []}) + "\n",
        encoding="utf-8",
    )
    (experiment / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "sampling_profile": "rebuttal_200_v1",
                "sampling_status": "post_hoc_nested_extension",
                "selected_queries": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="required artifacts"):
        package_rebuttal_bundle(
            experiment,
            output_path=tmp_path / "incomplete-200.tar.zst",
            repository_path=repository,
        )
