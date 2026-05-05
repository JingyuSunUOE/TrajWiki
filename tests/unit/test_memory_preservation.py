from __future__ import annotations

from trajpatch.memory.orchestrator import MemoryOrchestrator, ParsedMemory
from trajpatch.memory.historical import build_trajectory_historical_evidence_card, historical_item_terms_v2
from trajpatch.memory.preservation import (
    MustPreserveCandidate,
    audit_claim_preservation,
    extract_must_preserve_candidates,
    raw_records_from_normalized,
)
from trajpatch.memory.trajectory_summaries import summary_keywords_v2
from trajpatch.memory.schemas import (
    ClaimSignalExactTerm,
    ClaimSignalExtractionResult,
    EpisodicMemoryInput,
    MemoryClaim,
)
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.storage.models import ClaimRecord
from trajpatch.types import NormalizedMessage


def _message(raw_id: str, content: str, *, speaker: str = "Caroline") -> NormalizedMessage:
    return NormalizedMessage(
        role="user",
        content=content,
        turn_index=0,
        speaker_name=speaker,
        raw_message_id=raw_id,
    )


def test_extract_must_preserve_candidates_captures_research_lists_and_events() -> None:
    records = raw_records_from_normalized(
        [
            _message("sample-m0001", "I researched adoption agencies."),
            _message("sample-m0002", "My activities include pottery, camping, painting, and swimming.", speaker="Melanie"),
            _message("sample-m0003", "I joined a mentoring program and gave a school speech."),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.relation) for candidate in candidates}

    assert ("adoption agencies", "research_topic", "research_topic") in rows
    assert ("pottery", "activity", None) in rows
    assert ("camping", "activity", None) in rows
    assert ("painting", "activity", None) in rows
    assert ("swimming", "activity", None) in rows
    assert ("mentoring program", "event_type", "event_type") in rows
    assert ("school speech", "event_type", "event_type") in rows


def test_extract_must_preserve_candidates_canonicalizes_school_event_talk_alias() -> None:
    records = raw_records_from_normalized(
        [
            _message(
                "sample-m0001",
                "I wanted to tell you about my school event. I talked about my transgender journey and encouraged students.",
            ),
            _message("sample-m0002", "I felt powerful giving my talk."),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.rule) for candidate in candidates}
    by_surface = {candidate.surface.casefold(): candidate for candidate in candidates}

    assert ("school event", "event_type", "school_event_pattern") in rows
    assert ("school speech", "event_type", "school_speech_event_alias_pattern") in rows
    assert ("school talk", "event_type", "school_talk_event_alias_pattern") in rows
    assert "sample-m0001" in by_surface["school speech"].source_message_ids
    assert "sample-m0002" in by_surface["school speech"].source_message_ids


def test_extract_must_preserve_candidates_canonicalizes_mentorship_program() -> None:
    records = raw_records_from_normalized(
        [
            _message(
                "sample-m0001",
                "I joined a mentorship program for LGBTQ youth.",
            ),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.rule) for candidate in candidates}

    assert ("mentorship program", "event_type", "event_phrase_pattern") in rows
    assert ("mentoring program", "event_type", "mentoring_program_canonical_alias") in rows


def test_extract_must_preserve_candidates_captures_source_surface_objects() -> None:
    records = raw_records_from_normalized(
        [
            _message("sample-m0001", "We painted a sunset last weekend.", speaker="Melanie"),
            _message("sample-m0002", "My favorite snack is ginger snaps."),
            _message("sample-m0003", "She researched adoption agencies."),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.rule) for candidate in candidates}

    assert ("sunset", "painted_object", "painted_object_action_pattern") in rows
    assert ("ginger snaps", "food", "favorite_food_pattern") in rows
    assert ("adoption agencies", "research_topic", "research_topic_pattern") in rows


def test_extract_must_preserve_candidates_captures_preference_surfaces() -> None:
    records = raw_records_from_normalized(
        [
            _message("sample-m0001", "They were stoked for the dinosaur exhibit!", speaker="Melanie"),
            _message("sample-m0002", "The kids love learning about animals.", speaker="Melanie"),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.rule) for candidate in candidates}

    assert ("dinosaur exhibit", "preference_item", "preference_excitement_pattern") in rows
    assert ("learning about animals", "preference_item", "preference_verb_pattern") in rows


def test_extract_must_preserve_candidates_captures_temporal_event_records() -> None:
    records = raw_records_from_normalized(
        [
            _message("sample-m0001", "I'm also hosting a dance competition next month.", speaker="Jon"),
            _message("sample-m0002", "Yesterday I chose to go to networking events for my store.", speaker="Jon"),
            _message("sample-m0003", "We took a roadtrip this past weekend.", speaker="Melanie"),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.rule) for candidate in candidates}
    by_event_surface = {
        candidate.surface.casefold(): candidate
        for candidate in candidates
        if candidate.category == "event_object"
    }

    assert ("dance competition", "event_object", "event_object_action_pattern") in rows
    assert by_event_surface["dance competition"].event_action.casefold() == "hosting"
    assert by_event_surface["dance competition"].temporal_expression.casefold() == "next month"
    assert ("networking events", "event_object", "networking_event_action_pattern") in rows
    assert by_event_surface["networking events"].temporal_expression.casefold() == "yesterday"
    assert ("roadtrip", "event_object", "roadtrip_temporal_pattern") in rows
    assert by_event_surface["roadtrip"].temporal_expression.casefold() == "this past weekend"


def test_extract_must_preserve_candidates_canonicalizes_car_accident_caption() -> None:
    records = raw_records_from_normalized(
        [
            _message(
                "sample-m0001",
                "[shared image: a damaged car on a flatbed truck after an accident]",
                speaker="Maria",
            ),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.canonical) for candidate in candidates}

    assert ("car accident", "event_object", "car accident") in rows


def test_extract_must_preserve_candidates_captures_destress_activity_surfaces() -> None:
    records = raw_records_from_normalized(
        [
            _message(
                "sample-m0001",
                "I just signed up for a pottery class yesterday. It's like therapy for me, letting me express myself and get creative.",
                speaker="Melanie",
            ),
            _message(
                "sample-m0002",
                "I've been running farther to de-stress, which has been great for my headspace.",
                speaker="Melanie",
            ),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    rows = {(candidate.surface.casefold(), candidate.category, candidate.rule) for candidate in candidates}

    assert ("pottery class", "activity", "registered_activity_pattern") in rows
    assert ("pottery", "activity", "registered_activity_base_pattern") in rows
    assert ("running", "activity", "activity_purpose_pattern") in rows


def test_extract_must_preserve_candidates_preserves_title_and_test_raw_surfaces() -> None:
    records = raw_records_from_normalized(
        [
            _message("sample-m0001", "I read The Hobbit last month."),
            _message("sample-m0002", "John retook the military aptitude test with great results."),
        ]
    )

    candidates = extract_must_preserve_candidates(records)
    records_by_category = {(candidate.category, candidate.surface): candidate for candidate in candidates}

    book = records_by_category[("book_title", "The Hobbit")]
    assert book.raw_surface == "The Hobbit"
    test = records_by_category[("test_type", "the military aptitude test")]
    assert test.raw_surface == "the military aptitude test"


def test_extract_must_preserve_candidates_captures_counts() -> None:
    records = raw_records_from_normalized([_message("sample-m0001", "I have 3 dogs and 2 cats.")])

    candidates = extract_must_preserve_candidates(records)

    rows = {(candidate.surface.casefold(), candidate.category) for candidate in candidates}
    assert ("3 dogs", "count") in rows
    assert any(surface.startswith("2") and category == "count" for surface, category in rows)


def test_extract_must_preserve_candidates_filters_generic_fillers() -> None:
    records = raw_records_from_normalized([_message("sample-m0001", "Yeah, thanks. Friday was fine.")])

    candidates = extract_must_preserve_candidates(records)

    assert candidates == []


def test_audit_claim_preservation_accepts_claim_text_exact_terms_and_facets() -> None:
    source = raw_records_from_normalized([_message("sample-m0001", "I researched adoption agencies.")])
    candidate = MustPreserveCandidate(
        surface="adoption agencies",
        category="research_topic",
        relation="research_topic",
        source_message_ids=["sample-m0001"],
    )
    claims = [
        MemoryClaim(
            claim_id="tmp-c1",
            status="active",
            source_message_ids=["sample-m0001"],
            text="Caroline researched adoption agencies.",
        )
    ]

    result = audit_claim_preservation(candidates=[candidate], claims=claims, source_messages=source)

    assert result.covered
    assert result.missing_candidates == []
    assert result.weak_source_links == []


def test_audit_claim_preservation_requires_each_list_item() -> None:
    source = raw_records_from_normalized(
        [_message("sample-m0001", "My activities include pottery, camping, painting, and swimming.", speaker="Melanie")]
    )
    candidates = [
        MustPreserveCandidate(surface="pottery", category="list_item", source_message_ids=["sample-m0001"]),
        MustPreserveCandidate(surface="camping", category="list_item", source_message_ids=["sample-m0001"]),
    ]
    claims = [
        MemoryClaim(
            claim_id="tmp-c1",
            status="active",
            source_message_ids=["sample-m0001"],
            text="Melanie enjoys several activities.",
        )
    ]

    result = audit_claim_preservation(candidates=candidates, claims=claims, source_messages=source)

    assert not result.covered
    assert {candidate.surface for candidate in result.missing_candidates} == {"pottery", "camping"}


def test_audit_claim_preservation_reports_generalized_painted_object() -> None:
    source = raw_records_from_normalized([_message("sample-m0001", "I painted a sunset last weekend.")])
    candidates = extract_must_preserve_candidates(source)
    claims = [
        MemoryClaim(
            claim_id="tmp-c1",
            status="active",
            source_message_ids=["sample-m0001"],
            text="Caroline completed a nature-inspired artwork last weekend.",
        )
    ]

    result = audit_claim_preservation(candidates=candidates, claims=claims, source_messages=source)

    assert not result.covered
    assert any(candidate.surface == "sunset" for candidate in result.missing_candidates)


def test_audit_claim_preservation_reports_weak_source_link() -> None:
    source = raw_records_from_normalized(
        [
            _message("sample-m0001", "I researched adoption agencies."),
            _message("sample-m0002", "I discussed family planning."),
        ]
    )
    candidate = MustPreserveCandidate(
        surface="adoption agencies",
        category="research_topic",
        relation="research_topic",
        source_message_ids=["sample-m0001"],
    )
    claims = [
        MemoryClaim(
            claim_id="tmp-c1",
            status="active",
            source_message_ids=["sample-m0002"],
            text="Caroline researched adoption agencies.",
        )
    ]

    result = audit_claim_preservation(candidates=[candidate], claims=claims, source_messages=source)

    assert not result.covered
    assert result.missing_candidates == []
    assert result.weak_source_links


def test_orchestrator_repairs_missing_list_item_claims_before_persist(run_config, store) -> None:
    raw_message = _message(
        "sample-m0000",
        "My activities include pottery, camping, painting, and swimming.",
        speaker="Melanie",
    )
    store.add_raw_message("sample", "locomo", raw_message)

    def callback(messages, system_prompt, metadata):
        if (metadata or {}).get("task") == "claim_preservation_repair":
            return (
                "BEGIN_MEMORY_DSL\n"
                "SUMMARY_CONTENT: Melanie listed her activities.\n"
                "CONTEXT: Melanie described activities she participates in.\n"
                "KEYWORDS: Melanie, activities, pottery, camping, painting, swimming\n\n"
                "[CLAIMS]\n"
                "- status=active | source_message_ids=sample-m0000 | text=Melanie participates in pottery.\n"
                "- status=active | source_message_ids=sample-m0000 | text=Melanie participates in camping.\n"
                "- status=active | source_message_ids=sample-m0000 | text=Melanie participates in painting.\n"
                "- status=active | source_message_ids=sample-m0000 | text=Melanie participates in swimming.\n"
                "END_MEMORY_DSL"
            )
        return MockLLMProvider()._default_response(messages, metadata or {})

    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(callback=callback), HashEmbeddingProvider())
    raw = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp="2026-04-21T10:00:00Z",
        summary_content="Melanie listed activities.",
        context="Melanie discussed hobbies.",
        keywords=["Melanie", "activities"],
        links=["sample-m0000"],
        status_flags=["active"],
        claims=[
            MemoryClaim(
                claim_id="tmp-c1",
                status="active",
                source_message_ids=["sample-m0000"],
                text="Melanie enjoys several activities.",
            )
        ],
        ops=[],
        raw_text="raw",
    )
    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text=raw.semantic_text,
        links=list(raw.links),
        claims=list(raw.claims),
        raw=raw,
        metadata={},
    )

    orchestrator.persist_memory("sample", "locomo", parsed, exchange_messages=[raw_message])

    trajectory = store.list_trajectories("sample")[0]
    claims = store.latest_claims(trajectory.id)
    claim_texts = [claim.text for claim in claims]
    assert "Melanie participates in pottery." in claim_texts
    assert "Melanie participates in swimming." in claim_texts
    assert claims[0].metadata_json["claim_text_llm_used_v1"] is True
    assert claims[0].metadata_json["claim_preservation_repair_used_v1"] is False
    assert trajectory.metadata_json["claim_preservation_misses_v1"] == []


def test_orchestrator_uses_item_level_preservation_fallback_when_repair_fails(run_config, store) -> None:
    raw_message = _message(
        "sample-m0000",
        "My activities include pottery, camping, painting, and swimming.",
        speaker="Melanie",
    )
    store.add_raw_message("sample", "locomo", raw_message)

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "episodic_claim_text_extract":
            return (
                "HAS_CLAIMS: true\n"
                "REASON: broad umbrella claim only\n\n"
                "[CLAIMS]\n"
                "- status=active | source_message_ids=sample-m0000 | supporting_quote=My activities include pottery, camping, painting, and swimming. | text=Melanie enjoys several activities."
            )
        if (metadata or {}).get("task") == "claim_preservation_repair":
            return "NOT A VALID MEMORY DSL"
        return MockLLMProvider()._default_response(messages, metadata or {})

    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(callback=callback), HashEmbeddingProvider())
    raw = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp="2026-04-21T10:00:00Z",
        summary_content="Melanie listed her activities.",
        context="Melanie described activities she participates in.",
        keywords=["Melanie", "activities", "pottery", "camping", "painting", "swimming"],
        links=["sample-m0000"],
        status_flags=["active"],
        claims=[
            MemoryClaim(
                claim_id="tmp-c1",
                status="active",
                source_message_ids=["sample-m0000"],
                text="Melanie enjoys several activities.",
            )
        ],
        ops=[],
        raw_text="raw",
    )
    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text=raw.semantic_text,
        links=list(raw.links),
        claims=list(raw.claims),
        raw=raw,
        metadata={},
    )

    orchestrator.persist_memory("sample", "locomo", parsed, exchange_messages=[raw_message])

    trajectory = store.list_trajectories("sample")[0]
    claims = store.latest_claims(trajectory.id)
    claim_texts = [claim.text for claim in claims]
    assert "Melanie mentioned pottery as an activity." in claim_texts
    assert "Melanie mentioned swimming as an activity." in claim_texts
    assert any(claim.metadata_json["claim_preservation_fallback_used_v1"] is True for claim in claims)


def test_persist_audited_memory_requires_preservation_metadata(run_config, store) -> None:
    raw_message = _message("sample-m0000", "I researched adoption agencies.")
    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text="Caroline researched adoption agencies.",
        links=["sample-m0000"],
        claims=[
            MemoryClaim(
                claim_id="tmp-c1",
                status="active",
                source_message_ids=["sample-m0000"],
                text="Caroline researched adoption agencies.",
            )
        ],
        raw=EpisodicMemoryInput(
            memory_type="episodic",
            timestamp="2026-04-21T10:00:00Z",
            summary_content="Caroline researched adoption agencies.",
            context="Caroline discussed adoption research.",
            keywords=["Caroline", "adoption"],
            links=["sample-m0000"],
            status_flags=["active"],
            claims=[],
            ops=[],
            raw_text="raw",
        ),
        metadata={},
    )
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(), HashEmbeddingProvider())

    try:
        orchestrator.persist_audited_memory("sample", "locomo", parsed, exchange_messages=[raw_message])
    except ValueError as exc:
        assert "requires claim preservation audit metadata" in str(exc)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("persist_audited_memory accepted unaudited parsed claims")


def test_claim_text_failure_falls_back_to_preservation_candidates(run_config, store) -> None:
    raw_message = _message("sample-m0000", "I researched adoption agencies.")
    store.add_raw_message("sample", "locomo", raw_message)

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "episodic_claim_text_extract":
            return "NO_MEMORY"
        return MockLLMProvider()._default_response(messages, metadata or {})

    raw = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp="2026-04-21T10:00:00Z",
        summary_content="Caroline researched adoption agencies.",
        context="Caroline discussed adoption research.",
        keywords=["Caroline", "adoption"],
        links=["sample-m0000"],
        status_flags=[],
        claims=[],
        ops=[],
        raw_text="raw",
    )
    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text=raw.semantic_text,
        links=list(raw.links),
        claims=[],
        raw=raw,
        metadata={},
    )
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(callback=callback), HashEmbeddingProvider())

    orchestrator.persist_memory("sample", "locomo", parsed, exchange_messages=[raw_message])

    trajectory = store.list_trajectories("sample")[0]
    claims = store.latest_claims(trajectory.id)
    assert any("adoption agencies" in claim.text for claim in claims)
    assert claims[0].metadata_json["claim_text_llm_failed_v1"] is True
    assert claims[0].metadata_json["claim_text_fallback_source_v1"] == "preservation_candidates"
    assert claims[0].metadata_json["claim_text_empty_after_fallback_v1"] is False


def test_claim_text_failure_without_candidates_allows_empty_claim_snapshot(run_config, store) -> None:
    raw_message = _message("sample-m0000", "Thanks for the help.")
    store.add_raw_message("sample", "locomo", raw_message)

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "episodic_claim_text_extract":
            return "NO_MEMORY"
        return MockLLMProvider()._default_response(messages, metadata or {})

    raw = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp="2026-04-21T10:00:00Z",
        summary_content="A brief acknowledgement was exchanged.",
        context="The exchange has little factual content.",
        keywords=["acknowledgement"],
        links=["sample-m0000"],
        status_flags=[],
        claims=[],
        ops=[],
        raw_text="raw",
    )
    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text=raw.semantic_text,
        links=list(raw.links),
        claims=[],
        raw=raw,
        metadata={},
    )
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(callback=callback), HashEmbeddingProvider())

    persisted = orchestrator.persist_memory("sample", "locomo", parsed, exchange_messages=[raw_message])

    assert persisted is True
    trajectory = store.list_trajectories("sample")[0]
    assert store.latest_claims(trajectory.id) == []
    latest_snapshot = store.latest_snapshot(trajectory.id)
    assert latest_snapshot is not None
    assert latest_snapshot.metadata_json["claim_text_llm_failed_v1"] is True
    assert latest_snapshot.metadata_json["claim_text_empty_after_fallback_v1"] is True
    assert latest_snapshot.metadata_json["zero_claim_episodic_memory_v1"] is True
    assert latest_snapshot.metadata_json["zero_claim_episodic_persisted_v1"] is True
    assert orchestrator.zero_claim_episodic_candidate_count == 1
    assert orchestrator.zero_claim_episodic_persisted_count == 1
    assert orchestrator.zero_claim_low_salience_skipped_count == 0


def test_low_salience_claim_text_failure_without_candidates_skips_persist(run_config, store) -> None:
    raw_message = _message("sample-m0000", "Thanks for the help.")
    store.add_raw_message("sample", "locomo", raw_message)

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "episodic_claim_text_extract":
            return "NO_MEMORY"
        return MockLLMProvider()._default_response(messages, metadata or {})

    raw = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp="2026-04-21T10:00:00Z",
        summary_content="A brief acknowledgement was exchanged.",
        context="The exchange has little factual content.",
        keywords=["acknowledgement"],
        links=["sample-m0000"],
        status_flags=[],
        claims=[],
        ops=[],
        raw_text="raw",
    )
    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text=raw.semantic_text,
        links=list(raw.links),
        claims=[],
        raw=raw,
        metadata={
            "low_salience_memory_v1": True,
            "episodic_seed_source_v1": "forced_low_salience_no_memory",
            "forced_episodic_seed_reason_v1": "acknowledgement_or_low_salience_no_memory",
        },
    )
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(callback=callback), HashEmbeddingProvider())

    persisted = orchestrator.persist_memory("sample", "locomo", parsed, exchange_messages=[raw_message])

    assert persisted is False
    assert store.list_trajectories("sample") == []
    assert orchestrator.zero_claim_episodic_candidate_count == 1
    assert orchestrator.zero_claim_episodic_persisted_count == 0
    assert orchestrator.zero_claim_low_salience_skipped_count == 1


def test_claim_signal_validation_discards_fragment_terms(run_config, store) -> None:
    raw_message = _message(
        "sample-m0000",
        "Caroline researched adoption agencies and read Charlotte Web.",
        speaker="Caroline",
    )
    store.add_raw_message("sample", "locomo", raw_message)
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(), HashEmbeddingProvider())
    trajectory = store.create_trajectory(
        sample_id="sample",
        dataset_name="locomo",
        label="adoption",
        strict_matching=False,
        max_length=5,
        metadata={},
    )
    claim = ClaimRecord(
        id="claim-1",
        snapshot_id="snapshot-1",
        trajectory_id=trajectory.id,
        claim_id="epi-sample-001-c001",
        text="Caroline researched adoption agencies and read Charlotte Web.",
        status="active",
        source_message_ids_json=["sample-m0000"],
        parent_claim_id=None,
        revised_from_claim_id=None,
        metadata_json={
            "exact_terms_v1": ["adoption agencies"],
            "exact_terms_discarded_v1": [{"surface": "Here to a", "reason": "fragment"}],
            "facets_v1": [],
            "facet_discarded_v1": [],
        },
    )
    result = ClaimSignalExtractionResult(
        exact_terms=[
            ClaimSignalExactTerm(
                surface="adoption agencies",
                category="research_topic",
                source_claim_id=claim.claim_id,
                source_message_ids=["sample-m0000"],
            ),
            ClaimSignalExactTerm(
                surface="Here to a",
                category="other",
                source_claim_id=claim.claim_id,
                source_message_ids=["sample-m0000"],
            ),
            ClaimSignalExactTerm(
                surface="Charlotte's Web",
                category="title",
                source_claim_id=claim.claim_id,
                source_message_ids=["sample-m0000"],
            ),
            ClaimSignalExactTerm(
                surface="Charlotte Web",
                category="title",
                source_claim_id=claim.claim_id,
                source_message_ids=["sample-m0000"],
            ),
        ],
        display_items=["adoption agencies", "Here to a", "Charlotte Web"],
        display_key_facts=["Caroline researched adoption agencies and read Charlotte Web."],
    )

    kept, discarded = orchestrator._apply_claim_signal_result(  # noqa: SLF001
        result,
        [claim],
        {"sample-m0000": raw_message},
    )

    metadata = claim.metadata_json
    assert kept >= 3
    assert metadata["exact_terms_v2"] == ["adoption agencies", "Charlotte Web"]
    assert metadata["display_signals_v1"]["items"] == ["adoption agencies", "Charlotte Web"]
    assert "exact_terms_v1" not in metadata
    assert "facets_v1" not in metadata
    assert metadata["exact_terms_discarded_v1"] == [{"surface": "Here to a", "reason": "fragment"}]
    assert any(item["surface"] == "Here to a" for item in discarded)
    assert any(item["surface"] == "Charlotte's Web" for item in discarded)


def test_deterministic_signal_fallback_clears_baseline_retrieval_fields(run_config, store) -> None:
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(), HashEmbeddingProvider())
    claim = ClaimRecord(
        id="claim-1",
        snapshot_id="snapshot-1",
        trajectory_id="trajectory-1",
        claim_id="claim-1",
        text="Caroline researched adoption agencies.",
        status="active",
        source_message_ids_json=[],
        parent_claim_id=None,
        revised_from_claim_id=None,
        metadata_json={
            "exact_terms_v1": ["adoption agencies"],
            "exact_terms_discarded_v1": [{"surface": "Thanks", "reason": "filler"}],
            "facets_v1": [
                {
                    "relation": "research_topic",
                    "value": "adoption agencies",
                    "entity": "Caroline",
                    "source_claim_id": "claim-1",
                }
            ],
            "facet_discarded_v1": [],
        },
    )

    orchestrator._apply_deterministic_signal_fallback([claim])  # noqa: SLF001

    metadata = claim.metadata_json
    assert metadata["exact_terms_v2"] == ["adoption agencies"]
    assert metadata["facets_v2"][0]["relation"] == "research_topic"
    assert "exact_terms_v1" not in metadata
    assert "facets_v1" not in metadata
    assert metadata["exact_terms_discarded_v1"] == [{"surface": "Thanks", "reason": "filler"}]


def test_source_surface_merge_preserves_terms_when_llm_signals_are_empty(run_config, store) -> None:
    raw_message = _message("sample-m0000", "Melanie painted a sunset last weekend.", speaker="Melanie")
    store.add_raw_message("sample", "locomo", raw_message)
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(), HashEmbeddingProvider())
    claim = ClaimRecord(
        id="claim-1",
        snapshot_id="snapshot-1",
        trajectory_id="trajectory-1",
        claim_id="claim-1",
        text="Melanie completed a nature-inspired artwork last weekend.",
        status="active",
        source_message_ids_json=["sample-m0000"],
        parent_claim_id=None,
        revised_from_claim_id=None,
        metadata_json={"exact_terms_v2": [], "display_signals_v1": {"items": [], "counts": [], "key_facts": []}},
    )

    orchestrator._merge_source_surface_terms_into_claims(  # noqa: SLF001
        [claim],
        raw_records_from_normalized([raw_message]),
        entity_lexicon={},
    )

    metadata = claim.metadata_json
    assert "sunset" in metadata["source_surface_terms_v1"]
    assert "sunset" in metadata["exact_terms_v2"]
    assert "sunset" in metadata["display_signals_v1"]["items"]


def test_source_surface_merge_preserves_raw_title_terms(run_config, store) -> None:
    raw_message = _message("sample-m0000", "John retook the military aptitude test.", speaker="John")
    store.add_raw_message("sample", "locomo", raw_message)
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(), HashEmbeddingProvider())
    claim = ClaimRecord(
        id="claim-1",
        snapshot_id="snapshot-1",
        trajectory_id="trajectory-1",
        claim_id="claim-1",
        text="John took an assessment multiple times.",
        status="active",
        source_message_ids_json=["sample-m0000"],
        parent_claim_id=None,
        revised_from_claim_id=None,
        metadata_json={"exact_terms_v2": [], "display_signals_v1": {"items": [], "counts": [], "key_facts": []}},
    )

    orchestrator._merge_source_surface_terms_into_claims(  # noqa: SLF001
        [claim],
        raw_records_from_normalized([raw_message]),
        entity_lexicon={},
    )

    metadata = claim.metadata_json
    assert "the military aptitude test" in metadata["source_surface_raw_terms_v1"]
    assert "the military aptitude test" in metadata["exact_terms_v2"]
    assert "the military aptitude test" in metadata["display_signals_v1"]["items"]


def test_historical_evidence_card_prioritizes_source_surface_terms() -> None:
    card = build_trajectory_historical_evidence_card(
        trajectory_id="epi-sample-001",
        trajectory_label="painting",
        retrieval_summary_text="Melanie made nature-inspired artwork.",
        latest_semantic_text="Melanie discussed art.",
        metadata={
            "source_surface_terms_v1": ["sunset"],
            "exact_terms": [],
            "display_items": [],
            "display_key_facts": ["Melanie made nature-inspired artwork."],
        },
        active_claim_texts=["Melanie made nature-inspired artwork."],
        source_anchors=[{"source_ref": "D8:6", "text": "Melanie painted a sunset last weekend."}],
    )

    assert "sunset" in card["source_surface_terms"]
    assert "sunset" in card["historical_item_terms"]
    assert "sunset" in card["display_items"]
    assert card["historical_item_terms_policy"] == "source_backed_terms_v2"


def test_summary_keywords_v2_removes_internal_summary_markers() -> None:
    summary = """
    ## Profile / Stable Facts
    - Trajectory label: Melanie pottery
    - None recorded.

    ## Item Sets / Named Entities
    - Pottery class
    - sunset

    ## Relations / Temporal Updates
    - Melanie painted a sunset.
    """
    keywords = summary_keywords_v2(
        summary,
        {
            "source_surface_terms_v1": ["sunset"],
            "exact_terms_v2": ["pottery class"],
        },
    )

    assert "sunset" in keywords
    assert "pottery" in keywords
    assert "profile" not in keywords
    assert "stable" not in keywords
    assert "facts" not in keywords
    assert "none" not in keywords
    assert "recorded" not in keywords
    assert "label" not in keywords


def test_historical_item_terms_v2_does_not_inherit_noisy_keywords() -> None:
    terms = historical_item_terms_v2(
        metadata={
            "retrieval_summary_keywords": ["profile", "facts", "none", "recorded", "adoption"],
            "source_surface_terms_v1": ["adoption agencies"],
            "exact_terms_v2": ["adoption agencies"],
        },
        active_claim_texts=["Caroline researched adoption agencies."],
        retrieval_summary_text="## Profile / Stable Facts\n- None recorded.\n- adoption agencies",
    )

    assert "adoption agencies" in terms["historical_item_terms"]
    assert "profile" not in terms["historical_item_terms"]
    assert "facts" not in terms["historical_item_terms"]
    assert "None recorded" not in terms["historical_item_terms"]
