from __future__ import annotations

from trajpatch.exceptions import ParserValidationError
from trajpatch.memory.extraction_recovery import (
    PartialMemoryDraft,
    build_fallback_episodic_memory,
    build_partial_memory_draft,
    merge_partial_memory_drafts,
    normalize_memory_text,
)
from trajpatch.types import NormalizedMessage


def test_normalize_memory_text_prefers_markers_over_explanation():
    normalized, explanation = normalize_memory_text(
        "Here is the repaired memory.\n"
        "BEGIN_MEMORY_DSL\n"
        "SUMMARY_CONTENT: Smoking update.\n"
        "CONTEXT: The user mentioned smoking.\n"
        "KEYWORDS: smoking\n"
        "END_MEMORY_DSL"
    )

    assert normalized.startswith("SUMMARY_CONTENT: Smoking update.")
    assert "Here is the repaired memory." in explanation


def test_normalize_memory_text_extracts_relevant_code_block():
    normalized, explanation = normalize_memory_text(
        "Short note first.\n```text\nSUMMARY_CONTENT: Smoking update.\nCONTEXT: The user mentioned smoking.\nKEYWORDS: smoking\n```\nIgnored tail."
    )

    assert normalized.startswith("SUMMARY_CONTENT: Smoking update.")
    assert "Short note first." in explanation


def test_merge_partial_drafts_keeps_completed_sections_when_repair_fills_missing_items():
    initial = build_partial_memory_draft(
        "episodic",
        "\n".join(
            [
                "SUMMARY_CONTENT: Smoking update.",
                "CONTEXT: The user mentioned smoking.",
            ]
        ),
    )
    repair = build_partial_memory_draft(
        "episodic",
        "\n".join(
            [
                "KEYWORDS: smoking",
            ]
        ),
    )
    initial.repair_targets = ["KEYWORDS"]
    merged = merge_partial_memory_drafts(initial, repair)

    assert merged.top_fields["SUMMARY_CONTENT"] == "Smoking update."
    assert merged.top_fields["KEYWORDS"] == "smoking"


def test_build_fallback_episodic_memory_marks_uncertainty_and_links_sources():
    draft = PartialMemoryDraft(
        memory_type="episodic",
        explanation_text="The model explained that the user may have changed their smoking habit.",
    )
    exchange = [
        NormalizedMessage(role="user", content="I smoke 5 cigarettes.", turn_index=0, raw_message_id="sample-m0001"),
        NormalizedMessage(role="assistant", content="Understood.", turn_index=1, raw_message_id="sample-m0002"),
    ]

    fallback = build_fallback_episodic_memory(draft, exchange)

    assert fallback.status_flags == []
    assert fallback.claims == []
    assert fallback.ops == []
    assert fallback.links == ["sample-m0001", "sample-m0002"]


def test_build_partial_memory_draft_maps_structured_errors_to_precise_repair_targets():
    base_text = "\n".join(
        [
            "SUMMARY_CONTENT: Smoking update.",
            "CONTEXT: The user mentioned smoking.",
            "KEYWORDS: smoking",
            "",
            "[CLAIMS]",
            "- claim_id=tmp-c1 | status=active | source_message_ids=sample-m0001 | text=User discussed smoking.",
            "",
            "[OPS]",
            "- op=ADD | target_claim_id=tmp-c1 | new_claim_id=tmp-c1 | source_message_ids=sample-m0001 | rationale=Initial fact. | claim_text=User discussed smoking.",
        ]
    )

    ops_draft = build_partial_memory_draft(
        "episodic",
        base_text,
        validation_error=ParserValidationError(
            "Missing required key 'rationale' in OPS record.",
            code="missing_record_key",
            section="OPS",
            field="rationale",
        ),
    )
    claims_draft = build_partial_memory_draft(
        "episodic",
        base_text,
        validation_error=ParserValidationError(
            "Invalid claim status: resolved",
            code="invalid_claim_status",
            section="CLAIMS",
            field="status",
        ),
    )
    links_draft = build_partial_memory_draft(
        "episodic",
        base_text,
        validation_error=ParserValidationError(
            "Unknown raw message ids in links: ['conv-26']",
            code="invalid_links",
            section="top",
            field="LINKS",
        ),
    )

    assert ops_draft.repair_targets == []
    assert claims_draft.repair_targets == ["CLAIMS"]
    assert links_draft.repair_targets == ["LINKS"]


def test_build_partial_memory_draft_does_not_require_memory_type():
    episodic_draft = build_partial_memory_draft(
        "episodic",
        "\n".join(
            [
                "SUMMARY_CONTENT: Smoking update.",
                "CONTEXT: The user mentioned smoking.",
                "KEYWORDS: smoking",
            ]
        ),
    )

    assert "MEMORY_TYPE" not in episodic_draft.missing_top_fields
    assert "MEMORY_TYPE" not in episodic_draft.repair_targets
