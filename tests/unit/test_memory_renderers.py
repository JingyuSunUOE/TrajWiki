from __future__ import annotations

from trajpatch.memory.renderers import render_answer_episodic_snapshot, render_episodic_snapshot
from trajpatch.storage.models import ClaimOpRecord, ClaimRecord, EpisodicMemorySnapshot


def _snapshot() -> EpisodicMemorySnapshot:
    return EpisodicMemorySnapshot(
        id="epi-sample-001-v001",
        trajectory_id="epi-sample-001",
        version=1,
        timestamp="2026-04-23T00:00:00Z",
        links_json=["sample-m0001"],
        summary_content="Alice shared her tea preference.",
        context="Alice talked about drinks.",
        keywords_json=["alice", "tea"],
        status_flags_json=["active"],
        embedding_ref=None,
        semantic_text="Alice likes tea.",
        raw_text="Alice likes tea.",
        metadata_json={},
    )


def _claim(claim_id: str, text: str, status: str) -> ClaimRecord:
    return ClaimRecord(
        id=f"epi-sample-001-v001:{claim_id}",
        snapshot_id="epi-sample-001-v001",
        trajectory_id="epi-sample-001",
        claim_id=claim_id,
        text=text,
        status=status,
        source_message_ids_json=["sample-m0001"],
        parent_claim_id=None,
        revised_from_claim_id=None,
        metadata_json={},
    )


def _op() -> ClaimOpRecord:
    return ClaimOpRecord(
        id="epi-sample-001-v001-op001",
        snapshot_id="epi-sample-001-v001",
        trajectory_id="epi-sample-001",
        op_type="REVISE",
        target_claim_id="c-old",
        new_claim_id="c-active",
        source_message_ids_json=["sample-m0001"],
        rationale="Internal transition log.",
        metadata_json={},
    )


def test_answer_renderer_filters_deprecated_claims_and_ops() -> None:
    claims = [
        _claim("c-active", "Alice likes tea.", "active"),
        _claim("c-deprecated", "Alice likes coffee.", "deprecated"),
        _claim("c-uncertain", "Alice might like matcha.", "needs-confirmation"),
    ]

    rendered, counts = render_answer_episodic_snapshot(_snapshot(), claims, [_op()])

    assert "Timestamp: 2026-04-23T00:00:00Z" in rendered
    assert "Alice likes tea." in rendered
    assert "Alice likes coffee." not in rendered
    assert "Operations:" not in rendered
    assert "Internal transition log" not in rendered
    assert "Uncertainty / Conflict Claims:" in rendered
    assert "[needs-confirmation] c-uncertain: Alice might like matcha." in rendered
    assert counts == {
        "active_claim_count": 1,
        "uncertain_claim_count": 1,
        "suppressed_deprecated_claim_count": 1,
        "suppressed_speaker_grounding_suspect_claim_count": 0,
        "suppressed_ops_count": 1,
    }


def test_full_renderer_still_includes_all_claims_and_ops_for_debug() -> None:
    claims = [
        _claim("c-active", "Alice likes tea.", "active"),
        _claim("c-deprecated", "Alice likes coffee.", "deprecated"),
    ]

    rendered = render_episodic_snapshot(_snapshot(), claims, [_op()])

    assert "[active] c-active: Alice likes tea." in rendered
    assert "[deprecated] c-deprecated: Alice likes coffee." in rendered
    assert "Operations:" in rendered
    assert "Internal transition log." in rendered
