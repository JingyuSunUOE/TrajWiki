"""Rendering helpers for trajectory inspection and answer-time context building."""

from __future__ import annotations

from trajpatch.storage.models import ClaimOpRecord, ClaimRecord, EpisodicMemorySnapshot


def render_episodic_snapshot(
    snapshot: EpisodicMemorySnapshot, claims: list[ClaimRecord], ops: list[ClaimOpRecord]
) -> str:
    lines = [
        f"Episodic Snapshot {snapshot.id}",
        f"Summary: {snapshot.summary_content}",
        f"Context: {snapshot.context}",
        "Keywords: " + ", ".join(snapshot.keywords_json),
        "Flags: " + ", ".join(snapshot.status_flags_json),
        "Claims:",
    ]
    for claim in claims:
        lines.append(f"- [{claim.status}] {claim.claim_id}: {claim.text}")
    if ops:
        lines.append("Operations:")
        for op in ops:
            lines.append(
                f"- {op.op_type} target={op.target_claim_id} new={op.new_claim_id or 'none'} reason={op.rationale}"
            )
    return "\n".join(lines)


def render_answer_episodic_snapshot(
    snapshot: EpisodicMemorySnapshot, claims: list[ClaimRecord], ops: list[ClaimOpRecord]
) -> tuple[str, dict[str, int]]:
    """Render answer-time memory without stale claims or internal state logs."""

    active_claims = [
        claim
        for claim in claims
        if claim.status == "active" and not (claim.metadata_json or {}).get("speaker_grounding_suspect_v1")
    ]
    speaker_grounding_suspect_claims = [
        claim
        for claim in claims
        if claim.status == "active" and (claim.metadata_json or {}).get("speaker_grounding_suspect_v1")
    ]
    uncertain_claims = [
        claim for claim in claims if claim.status in {"contradictory", "needs-confirmation"}
    ]
    deprecated_claims = [claim for claim in claims if claim.status == "deprecated"]
    lines = [
        f"Episodic Snapshot {snapshot.id}",
        f"Timestamp: {snapshot.timestamp}",
        f"Summary: {snapshot.summary_content}",
        f"Context: {snapshot.context}",
        "Keywords: " + ", ".join(snapshot.keywords_json),
        "Flags: " + ", ".join(snapshot.status_flags_json),
        "Active Claims:",
    ]
    if active_claims:
        for claim in active_claims:
            lines.append(f"- {claim.claim_id}: {claim.text}")
    else:
        lines.append("- none")
    if uncertain_claims:
        lines.append("Uncertainty / Conflict Claims:")
        for claim in uncertain_claims:
            lines.append(f"- [{claim.status}] {claim.claim_id}: {claim.text}")
    return "\n".join(lines), {
        "active_claim_count": len(active_claims),
        "uncertain_claim_count": len(uncertain_claims),
        "suppressed_deprecated_claim_count": len(deprecated_claims),
        "suppressed_speaker_grounding_suspect_claim_count": len(speaker_grounding_suspect_claims),
        "suppressed_ops_count": len(ops),
    }
