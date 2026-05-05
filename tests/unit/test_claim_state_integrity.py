from __future__ import annotations

import pytest

from trajpatch.memory.orchestrator import CanonicalClaimView, MemoryOrchestrator, ParsedMemory
from trajpatch.memory.schemas import EpisodicMemoryInput, MemoryClaim
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.storage.models import ClaimRecord


def _orchestrator(run_config, store) -> MemoryOrchestrator:
    return MemoryOrchestrator(
        run_config,
        store,
        MockLLMProvider(),
        HashEmbeddingProvider(),
    )


def _claim_record(
    *,
    trajectory_id: str = "traj-sample-001",
    claim_id: str = "traj-sample-001-c001",
    text: str = "Caroline researched adoption agencies.",
    status: str = "active",
    sources: list[str] | None = None,
) -> ClaimRecord:
    return ClaimRecord(
        id=f"snapshot-001:{claim_id}",
        snapshot_id="snapshot-001",
        trajectory_id=trajectory_id,
        claim_id=claim_id,
        text=text,
        status=status,
        source_message_ids_json=sources or ["sample-m0001"],
        parent_claim_id=None,
        revised_from_claim_id=None,
        metadata_json={},
    )


def _parsed(claims: list[MemoryClaim]) -> ParsedMemory:
    raw = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp="2026-04-20T10:00:00Z",
        summary_content="Summary.",
        context="Context.",
        keywords=["memory"],
        links=[],
        status_flags=["active"],
        claims=claims,
        raw_text="raw",
    )
    return ParsedMemory(
        memory_type="episodic",
        semantic_text=raw.semantic_text,
        links=[],
        claims=claims,
        raw=raw,
    )


def test_duplicate_extracted_claims_are_deduped_and_sources_merged(run_config, store) -> None:
    orchestrator = _orchestrator(run_config, store)
    previous = _claim_record(status="active", sources=["sample-m0001"])
    parsed = _parsed(
        [
            MemoryClaim(
                claim_id="local-1",
                status="active",
                source_message_ids=["sample-m0002"],
                text="Caroline researched adoption agencies.",
            ),
            MemoryClaim(
                claim_id="local-2",
                status="contradictory",
                source_message_ids=["sample-m0003"],
                text="Caroline   researched adoption agencies.",
            ),
        ]
    )

    claims, ops = orchestrator.apply_claim_ops(
        "sample",
        previous.trajectory_id,
        parsed,
        [previous],
        "Caroline researched adoption agencies.",
    )

    assert [claim.claim_id for claim in claims] == [previous.claim_id]
    assert claims[0].status == "contradictory"
    assert claims[0].source_message_ids_json == ["sample-m0001", "sample-m0002", "sample-m0003"]
    assert parsed.metadata["duplicate_extracted_claim_count"] == 1
    assert parsed.metadata["claim_new_add_count"] == 0
    assert len(ops) == 1
    assert ops[0].op_type == "REVISE"
    assert ops[0].new_claim_id == previous.claim_id
    assert ops[0].metadata_json["status_transition"] is True


def test_exact_text_active_to_deprecated_generates_deprecate_op(run_config, store) -> None:
    orchestrator = _orchestrator(run_config, store)
    previous = _claim_record(status="active")
    parsed = _parsed(
        [
            MemoryClaim(
                claim_id="local-1",
                status="deprecated",
                source_message_ids=["sample-m0002"],
                text=previous.text,
            )
        ]
    )

    claims, ops = orchestrator.apply_claim_ops(
        "sample",
        previous.trajectory_id,
        parsed,
        [previous],
        previous.text,
    )

    assert claims[0].claim_id == previous.claim_id
    assert claims[0].status == "deprecated"
    assert len(ops) == 1
    assert ops[0].op_type == "DEPRECATE"
    assert ops[0].target_claim_id == previous.claim_id
    assert ops[0].new_claim_id is None
    assert ops[0].metadata_json == {
        "system_derived": True,
        "strategy": "deterministic_claim_diff",
        "status_transition": True,
        "previous_status": "active",
        "new_status": "deprecated",
        "same_claim_id": True,
    }
    assert "Status changed from active to deprecated" in ops[0].rationale
    assert parsed.metadata["ops_synthesized_count"] == 1
    assert parsed.metadata["claim_transition_revise_count"] == 0


def test_exact_text_status_change_generates_same_id_revise_op(run_config, store) -> None:
    orchestrator = _orchestrator(run_config, store)
    previous = _claim_record(status="active")
    parsed = _parsed(
        [
            MemoryClaim(
                claim_id="local-1",
                status="needs-confirmation",
                source_message_ids=["sample-m0002"],
                text=previous.text,
            )
        ]
    )

    claims, ops = orchestrator.apply_claim_ops(
        "sample",
        previous.trajectory_id,
        parsed,
        [previous],
        previous.text,
    )

    assert claims[0].claim_id == previous.claim_id
    assert claims[0].status == "needs-confirmation"
    assert len(ops) == 1
    assert ops[0].op_type == "REVISE"
    assert ops[0].target_claim_id == previous.claim_id
    assert ops[0].new_claim_id == previous.claim_id
    assert ops[0].metadata_json["status_transition"] is True
    assert ops[0].metadata_json["same_claim_id"] is True
    assert parsed.metadata["claim_transition_revise_count"] == 0


def test_candidate_label_map_length_mismatch_raises(run_config, store, monkeypatch) -> None:
    orchestrator = _orchestrator(run_config, store)
    monkeypatch.setattr(orchestrator, "_candidate_label_map", lambda _prefix, ids: {"P1": ids[0]})

    candidates = [
        CanonicalClaimView(
            claim_id="traj-c001",
            text="Claim one.",
            normalized_text="claim one.",
            status="active",
            source_message_ids=[],
        ),
        CanonicalClaimView(
            claim_id="traj-c002",
            text="Claim two.",
            normalized_text="claim two.",
            status="active",
            source_message_ids=[],
        ),
    ]

    with pytest.raises(ValueError, match="candidate label map length mismatch"):
        orchestrator._adjudicate_claim_transition(
            sample_id="sample",
            current_claim=MemoryClaim(
                claim_id="local-1",
                status="active",
                source_message_ids=[],
                text="New claim.",
            ),
            candidates=candidates,
            exchange_text="New claim.",
        )


def test_next_claim_ordinal_uses_max_suffix_not_count(store) -> None:
    store.session.add_all(
        [
            _claim_record(claim_id="traj-sample-001-c001"),
            _claim_record(claim_id="traj-sample-001-c003", text="Different claim."),
            _claim_record(claim_id="nonstandard", text="Ignored id."),
        ]
    )
    store.session.flush()

    assert store.next_claim_ordinal("traj-sample-001") == 4


def test_list_claims_for_snapshots_accepts_generator(store) -> None:
    store.session.add_all(
        [
            _claim_record(claim_id="traj-sample-001-c001", sources=["sample-m0001"]),
            _claim_record(
                claim_id="traj-sample-001-c002",
                text="Second claim.",
                sources=["sample-m0002"],
            ),
        ]
    )
    store.session.flush()

    grouped = store.list_claims_for_snapshots(snapshot_id for snapshot_id in ["snapshot-001"])

    assert [claim.claim_id for claim in grouped["snapshot-001"]] == [
        "traj-sample-001-c001",
        "traj-sample-001-c002",
    ]
