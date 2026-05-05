from __future__ import annotations

from trajpatch.memory.llm_text_parsers import parse_episodic_memory


def test_parse_episodic_seed_without_claims() -> None:
    parsed = parse_episodic_memory(
        "SUMMARY_CONTENT: Caroline researched adoption agencies.\n"
        "CONTEXT: Caroline discussed adoption research.\n"
        "KEYWORDS: Caroline, adoption, agencies",
        {"sample-m0000"},
        exchange_link_ids=["sample-m0000"],
        exchange_timestamp="2026-04-22T10:00:00Z",
    )

    assert parsed is not None
    assert parsed.claims == []
    assert parsed.status_flags == []
    assert parsed.links == ["sample-m0000"]
    assert parsed.timestamp == "2026-04-22T10:00:00Z"


def test_parse_episodic_seed_ignores_legacy_claims_by_default() -> None:
    diagnostics: list[dict[str, object]] = []

    parsed = parse_episodic_memory(
        "SUMMARY_CONTENT: Caroline researched adoption agencies.\n"
        "CONTEXT: Caroline discussed adoption research.\n"
        "KEYWORDS: Caroline, adoption, agencies\n\n"
        "[CLAIMS]\n"
        "- status=active | source_message_ids=sample-m0000 | text=Caroline researched adoption agencies.",
        {"sample-m0000"},
        exchange_link_ids=["sample-m0000"],
        diagnostics=diagnostics,
    )

    assert parsed is not None
    assert parsed.claims == []
    assert any(item["kind"] == "legacy_claims_ignored" for item in diagnostics)


def test_parse_episodic_memory_can_still_parse_claims_for_preservation_repair() -> None:
    parsed = parse_episodic_memory(
        "SUMMARY_CONTENT: Caroline researched adoption agencies.\n"
        "CONTEXT: Caroline discussed adoption research.\n"
        "KEYWORDS: Caroline, adoption, agencies\n\n"
        "[CLAIMS]\n"
        "- status=active | source_message_ids=sample-m0000 | text=Caroline researched adoption agencies.",
        {"sample-m0000"},
        exchange_link_ids=["sample-m0000"],
        parse_claims=True,
    )

    assert parsed is not None
    assert len(parsed.claims) == 1
    assert parsed.claims[0].text == "Caroline researched adoption agencies."


def test_parse_episodic_no_memory_unchanged() -> None:
    assert parse_episodic_memory("NO_MEMORY") is None
