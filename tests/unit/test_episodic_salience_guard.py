from __future__ import annotations

from trajpatch.memory.orchestrator import (
    MemoryOrchestrator,
    StructuredFirstPassAttempt,
    detect_episodic_salience_v1,
)
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.providers.structured_outputs import EpisodicExtractionResult
from trajpatch.types import NormalizedMessage, StructuredLLMResponse


def _message(
    raw_id: str,
    content: str,
    *,
    role: str = "assistant",
    turn_index: int = 0,
    speaker: str = "John",
) -> NormalizedMessage:
    return NormalizedMessage(
        role=role,  # type: ignore[arg-type]
        content=content,
        turn_index=turn_index,
        speaker_name=speaker,
        raw_message_id=raw_id,
    )


def _no_memory_first_pass() -> StructuredFirstPassAttempt:
    return StructuredFirstPassAttempt(
        task="episodic_extract",
        vendor="openai",
        response=StructuredLLMResponse(
            parsed=EpisodicExtractionResult(has_memory=False, memory=None, reason="No memory."),
            metadata={
                "structured_vendor": "openai",
                "structured_strategy": "openai_json_schema",
                "structured_success": True,
                "structured_fallback_used": False,
            },
        ),
    )


def _locomo_config(run_config):
    return run_config.copy(update={"dataset": "locomo"})


def test_detect_episodic_salience_catches_old_area_and_place_fact() -> None:
    messages = [
        _message(
            "conv-41-m0305",
            "Sure, Maria! I want to work on improving my old area, West County, too.",
        )
    ]

    has_salience, reason = detect_episodic_salience_v1(messages)

    assert has_salience
    assert reason in {"place_or_area_fact", "plan_goal_or_state_change", "capitalized_place_like_term"}


def test_detect_episodic_salience_catches_flood_fact() -> None:
    messages = [_message("conv-41-m0479", "My old area was hit by a nasty flood last week.")]

    has_salience, reason = detect_episodic_salience_v1(messages)

    assert has_salience
    assert reason in {"place_or_area_fact", "incident_or_infrastructure_fact"}


def test_detect_episodic_salience_catches_short_item_facts() -> None:
    cases = [
        ("conv-26-m0025", "Researching adoption agencies — it's been a dream to have a family.", "family_or_adoption_fact"),
        ("conv-47-m0507", "I chose headphones from Sennheiser and bought a mouse from Logitech.", "list_item_or_inventory_fact"),
        ("conv-48-m0220", "The Eisenhower Matrix sorts tasks into four boxes.", "method_or_technique_fact"),
        ("conv-43-m0601", "I'm excited to watch a new show called \"The Wheel of Time\".", "media_title_or_recommendation_fact"),
        ("conv-41-m0387", "I'm trying kundalini yoga to get stronger.", "activity_or_practice_fact"),
        ("conv-44-m0105", "The hats don't bother them; they put them on for fun. [shared image: a photo of a dog wearing a party hat]", "list_item_or_inventory_fact"),
    ]

    for raw_id, content, expected_reason in cases:
        has_salience, reason = detect_episodic_salience_v1([_message(raw_id, content)])
        assert has_salience, content
        assert reason == expected_reason


def test_detect_episodic_salience_ignores_pure_acknowledgement() -> None:
    messages = [_message("sample-m0001", "Thanks! Talk soon.")]

    assert detect_episodic_salience_v1(messages) == (False, "none")


def test_structured_no_memory_is_overridden_for_salient_exchange(run_config, store) -> None:
    orchestrator = MemoryOrchestrator(_locomo_config(run_config), store, MockLLMProvider(), HashEmbeddingProvider())
    messages = [
        _message(
            "conv-41-m0305",
            "Sure, Maria! I want to work on improving my old area, West County, too.",
        )
    ]

    result = orchestrator.finalize_episodic_structured_first_pass(
        "conv-41",
        messages,
        {"conv-41-m0305"},
        _no_memory_first_pass(),
    )

    assert result.parsed_memory is not None
    assert result.parsed_memory.links == ["conv-41-m0305"]
    assert result.parsed_memory.metadata["forced_episodic_seed_used_v1"] is True
    assert result.parsed_memory.metadata["structured_no_memory_overridden_v1"] is True
    assert result.parsed_memory.metadata["llm_no_memory_overridden_v1"] is True
    assert result.parsed_memory.metadata["episodic_seed_source_v1"] == "forced_salient_no_memory"
    assert result.parsed_memory.metadata["llm_has_memory_v1"] is False
    assert result.parsed_memory.metadata["low_salience_memory_v1"] is False
    assert "force_close_after_persist" not in result.parsed_memory.metadata


def test_locomo_structured_no_memory_forces_low_salience_acknowledgement(run_config, store) -> None:
    orchestrator = MemoryOrchestrator(_locomo_config(run_config), store, MockLLMProvider(), HashEmbeddingProvider())
    messages = [_message("sample-m0001", "Thanks! Talk soon.")]

    result = orchestrator.finalize_episodic_structured_first_pass(
        "sample",
        messages,
        {"sample-m0001"},
        _no_memory_first_pass(),
    )

    assert result.parsed_memory is not None
    assert result.parsed_memory.links == ["sample-m0001"]
    assert result.parsed_memory.metadata["forced_episodic_seed_used_v1"] is True
    assert result.parsed_memory.metadata["structured_no_memory_overridden_v1"] is True
    assert result.parsed_memory.metadata["episodic_seed_source_v1"] == "forced_low_salience_no_memory"
    assert result.parsed_memory.metadata["low_salience_memory_v1"] is True
    assert orchestrator.forced_memory_seed_count == 1
    assert orchestrator.low_salience_memory_count == 1
    assert orchestrator.llm_no_memory_forced_count == 1


def test_low_salience_text_no_memory_with_empty_claims_skips_persist(run_config, store) -> None:
    message = _message("sample-m0001", "Thanks! Talk soon.")
    store.add_raw_message("sample", "locomo", message)

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task in {"episodic_extract", "episodic_claim_text_extract"}:
            return "NO_MEMORY"
        return MockLLMProvider()._default_response(messages, metadata or {})

    orchestrator = MemoryOrchestrator(
        _locomo_config(run_config),
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
    )

    orchestrator.process_exchange("sample", "locomo", [message])

    assert store.list_trajectories("sample") == []
    assert orchestrator.forced_memory_seed_count == 1
    assert orchestrator.low_salience_memory_count == 1
    assert orchestrator.llm_no_memory_forced_count == 1
    assert orchestrator.zero_claim_episodic_candidate_count == 1
    assert orchestrator.zero_claim_episodic_persisted_count == 0
    assert orchestrator.zero_claim_low_salience_skipped_count == 1


def test_medmt_structured_no_memory_can_still_skip_acknowledgement(run_config, store) -> None:
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(), HashEmbeddingProvider())
    messages = [_message("sample-m0001", "Thanks! Talk soon.")]

    result = orchestrator.finalize_episodic_structured_first_pass(
        "sample",
        messages,
        {"sample-m0001"},
        _no_memory_first_pass(),
    )

    assert result.parsed_memory is None


def test_medmt_structured_no_memory_skips_salient_exchange(run_config, store) -> None:
    orchestrator = MemoryOrchestrator(run_config, store, MockLLMProvider(), HashEmbeddingProvider())
    messages = [_message("sample-m0002", "My old area was hit by a nasty flood last week.")]

    result = orchestrator.finalize_episodic_structured_first_pass(
        "sample",
        messages,
        {"sample-m0002"},
        _no_memory_first_pass(),
    )

    assert result.parsed_memory is None
    assert orchestrator.forced_memory_seed_count == 0
    assert orchestrator.low_salience_memory_count == 0
    assert orchestrator.llm_no_memory_forced_count == 0


def test_medmt_text_no_memory_skips_salient_exchange(run_config, store) -> None:
    message = _message("sample-m0003", "My old area was hit by a nasty flood last week.")
    store.add_raw_message("sample", "medmt", message)

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "episodic_extract":
            return "NO_MEMORY"
        return MockLLMProvider()._default_response(messages, metadata or {})

    orchestrator = MemoryOrchestrator(
        run_config,
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
    )
    orchestrator.process_exchange("sample", "medmt", [message])

    assert store.list_trajectories("sample") == []
    assert orchestrator.forced_memory_seed_count == 0
    assert orchestrator.low_salience_memory_count == 0
    assert orchestrator.llm_no_memory_forced_count == 0


def test_text_no_memory_override_continues_to_claim_stage(run_config, store) -> None:
    user_message = _message(
        "conv-41-m0304",
        "Let's work together to make a real difference.",
        role="user",
        turn_index=304,
        speaker="Maria",
    )
    assistant_message = _message(
        "conv-41-m0305",
        "Sure, Maria! I want to work on improving my old area, West County, too.",
        role="assistant",
        turn_index=305,
    )
    store.add_raw_message("conv-41", "locomo", user_message)
    store.add_raw_message("conv-41", "locomo", assistant_message)

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "episodic_extract":
            return "NO_MEMORY"
        if task == "episodic_claim_text_extract":
            return (
                "HAS_CLAIMS: true\n"
                "REASON: old area fact\n\n"
                "[CLAIMS]\n"
                "- status=active | source_message_ids=conv-41-m0305 | "
                "supporting_quote=I want to work on improving my old area, West County, too. | "
                "text=John wants to improve his old area, West County."
            )
        return MockLLMProvider()._default_response(messages, metadata or {})

    orchestrator = MemoryOrchestrator(
        _locomo_config(run_config),
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
    )
    orchestrator.process_exchange("conv-41", "locomo", [user_message, assistant_message])

    trajectory = store.list_trajectories("conv-41")[0]
    claims = store.latest_claims(trajectory.id)

    assert any("West County" in claim.text for claim in claims)
    assert any("conv-41-m0305" in list(claim.source_message_ids_json) for claim in claims)
