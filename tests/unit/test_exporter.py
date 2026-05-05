from __future__ import annotations

import json

from trajpatch.pipeline.exporter import ArtifactExporter
from trajpatch.storage.models import ClaimOpRecord, ClaimRecord, EpisodicMemorySnapshot, TrajectoryRecord


def test_export_sample_trajectories_writes_compact_manifest_and_full_json(run_config, store) -> None:
    trajectory = TrajectoryRecord(
        id="epi-sample-001",
        sample_id="sample",
        dataset_name="locomo",
        label="caroline-status",
        is_open=True,
        snapshot_count=1,
        latest_snapshot_id="epi-sample-001-v001",
        metadata_json={
            "retrieval_summary_text": "Caroline is single.",
            "trajectory_historical_item_terms_v1": ["single"],
            "facet_tags": ["relationship_status"],
            "facet_values": ["relationship_status=single"],
            "entity_mentions": ["Caroline"],
        },
    )
    snapshot = EpisodicMemorySnapshot(
        id="epi-sample-001-v001",
        trajectory_id=trajectory.id,
        version=1,
        timestamp="2026-01-01T00:00:00",
        links_json=[],
        summary_content="Caroline is single.",
        context="relationship status",
        keywords_json=["single"],
        status_flags_json=[],
        semantic_text="Caroline is single.",
        raw_text="Caroline is single.",
        metadata_json={},
    )
    claim = ClaimRecord(
        id="claim-row-1",
        snapshot_id=snapshot.id,
        trajectory_id=trajectory.id,
        claim_id="c1",
        text="Caroline is single.",
        status="active",
        source_message_ids_json=["sample-m0001"],
        metadata_json={},
    )
    op = ClaimOpRecord(
        id="op-row-1",
        snapshot_id=snapshot.id,
        trajectory_id=trajectory.id,
        op_type="ADD",
        target_claim_id="c1",
        rationale="initial",
        source_message_ids_json=["sample-m0001"],
        metadata_json={},
    )
    store.session.add_all([trajectory, snapshot, claim, op])
    store.session.commit()

    exporter = ArtifactExporter(run_config.output_dir, store, run_dir=run_config.output_dir)
    stats = exporter.export_sample_trajectories("sample")

    full_json = run_config.output_dir / "memories" / "sample" / "epi-sample-001.json"
    manifest = run_config.output_dir / "memories" / "sample" / "trajectories.jsonl"
    assert full_json.exists()
    assert manifest.exists()
    assert stats["trajectory_count"] == 1

    full_payload = json.loads(full_json.read_text())
    assert full_payload["snapshots"][0]["claims"][0]["text"] == "Caroline is single."

    manifest_row = json.loads(manifest.read_text().strip())
    assert manifest_row["schema_version"] == "trajectory_export_manifest_v1"
    assert manifest_row["trajectory_id"] == "epi-sample-001"
    assert manifest_row["json_path"] == "epi-sample-001.json"
    assert manifest_row["summary_path"] == "epi-sample-001.summary.md"
    assert "snapshots" not in manifest_row
    assert "claims" not in manifest_row
    assert "operations" not in manifest_row
