from __future__ import annotations

from trajpatch.prompts import (
    ANSWER_GENERATION_PROMPT,
    CLAIM_SIGNAL_EXTRACT_PROMPT,
    CLAIM_SIGNAL_EXTRACT_STRUCTURED_PROMPT,
    CLAIM_TRANSITION_JUDGE_PROMPT,
    CLAIM_TRANSITION_JUDGE_STRUCTURED_PROMPT,
    EPISODIC_CLAIM_TEXT_EXTRACT_PROMPT,
    EPISODIC_CLAIM_TEXT_EXTRACT_STRUCTURED_PROMPT,
    EPISODIC_CLAIM_PRESERVATION_REPAIR_PROMPT,
    EPISODIC_EXTRACT_PROMPT,
    EPISODIC_EXTRACT_STRUCTURED_PROMPT,
    LOCOMO_ANSWER_COUNT_VALIDATION_PROMPT,
    LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT,
    LOCOMO_ANSWER_FREEFORM_PROMPT,
    LOCOMO_ANSWER_GENERATION_PROMPT,
    LOCOMO_ANSWER_REPAIR_ARBITRATION_PROMPT,
    LOCOMO_ANSWER_REPAIR_PROMPT,
    LOCOMO_ANSWER_TYPE_VERIFICATION_PROMPT,
    LOCOMO_JUDGE_PROMPT,
    LOCOMO_JUDGE_STRUCTURED_PROMPT,
    MEDMT_ANSWER_GENERATION_PROMPT,
    MEDMT_JUDGE_PROMPT,
    MEDMT_JUDGE_STRUCTURED_PROMPT,
    PROMPT_NAME_TO_VARIABLE,
    PROMPT_REGISTRY,
    RETRIEVAL_REFLECTION_PROMPT,
    SEMANTIC_METRIC_EXTRACT_PROMPT,
    SEMANTIC_METRIC_SCHEMA_PROMPT,
    STRUCTURE_REPAIR_PROMPT,
    TRAJECTORY_SET_RERANK_PROMPT,
    TRAJECTORY_MATCH_PROMPT,
    TRAJECTORY_MATCH_STRUCTURED_PROMPT,
    TRAJECTORY_RETRIEVAL_SUMMARY_PROMPT,
    WIKI_PAGE_COMPILE_PROMPT,
    WIKI_PAGE_PLAN_PROMPT,
    WIKI_PAGE_RERANK_PROMPT,
    list_prompts,
    load_prompt,
)


def test_prompt_registry_exposes_all_named_prompts():
    expected = {
        "episodic_extract",
        "episodic_claim_text_extract",
        "episodic_claim_text_extract_structured",
        "claim_signal_extract",
        "claim_signal_extract_structured",
        "trajectory_match",
        "claim_transition_judge",
        "episodic_extract_structured",
        "trajectory_match_structured",
        "claim_transition_judge_structured",
        "repair",
        "episodic_claim_preservation_repair",
        "answer_generation",
        "locomo_answer_generation",
        "locomo_answer_evidence_synthesis",
        "locomo_answer_freeform",
        "locomo_answer_type_verification",
        "locomo_answer_count_validation",
        "locomo_answer_repair",
        "locomo_answer_repair_arbitration",
        "medmt_answer_generation",
        "trajectory_retrieval_summary",
        "wiki_page_plan",
        "wiki_page_compile",
        "wiki_page_rerank",
        "trajectory_set_rerank",
        "locomo_judge",
        "medmt_judge",
        "locomo_judge_structured",
        "medmt_judge_structured",
        "semantic_metric_schema",
        "semantic_metric_extract",
        "retrieval_reflection",
    }
    assert set(list_prompts()) == expected
    assert set(PROMPT_REGISTRY) == expected
    for name in expected:
        assert load_prompt(name).strip()


def test_prompt_registry_points_to_individual_constants():
    expected_map = {
        "episodic_extract": EPISODIC_EXTRACT_PROMPT,
        "episodic_claim_text_extract": EPISODIC_CLAIM_TEXT_EXTRACT_PROMPT,
        "episodic_claim_text_extract_structured": EPISODIC_CLAIM_TEXT_EXTRACT_STRUCTURED_PROMPT,
        "claim_signal_extract": CLAIM_SIGNAL_EXTRACT_PROMPT,
        "claim_signal_extract_structured": CLAIM_SIGNAL_EXTRACT_STRUCTURED_PROMPT,
        "trajectory_match": TRAJECTORY_MATCH_PROMPT,
        "claim_transition_judge": CLAIM_TRANSITION_JUDGE_PROMPT,
        "episodic_extract_structured": EPISODIC_EXTRACT_STRUCTURED_PROMPT,
        "trajectory_match_structured": TRAJECTORY_MATCH_STRUCTURED_PROMPT,
        "claim_transition_judge_structured": CLAIM_TRANSITION_JUDGE_STRUCTURED_PROMPT,
        "repair": STRUCTURE_REPAIR_PROMPT,
        "episodic_claim_preservation_repair": EPISODIC_CLAIM_PRESERVATION_REPAIR_PROMPT,
        "answer_generation": ANSWER_GENERATION_PROMPT,
        "locomo_answer_generation": LOCOMO_ANSWER_GENERATION_PROMPT,
        "locomo_answer_evidence_synthesis": LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT,
        "locomo_answer_freeform": LOCOMO_ANSWER_FREEFORM_PROMPT,
        "locomo_answer_type_verification": LOCOMO_ANSWER_TYPE_VERIFICATION_PROMPT,
        "locomo_answer_count_validation": LOCOMO_ANSWER_COUNT_VALIDATION_PROMPT,
        "locomo_answer_repair": LOCOMO_ANSWER_REPAIR_PROMPT,
        "locomo_answer_repair_arbitration": LOCOMO_ANSWER_REPAIR_ARBITRATION_PROMPT,
        "medmt_answer_generation": MEDMT_ANSWER_GENERATION_PROMPT,
        "trajectory_retrieval_summary": TRAJECTORY_RETRIEVAL_SUMMARY_PROMPT,
        "wiki_page_plan": WIKI_PAGE_PLAN_PROMPT,
        "wiki_page_compile": WIKI_PAGE_COMPILE_PROMPT,
        "wiki_page_rerank": WIKI_PAGE_RERANK_PROMPT,
        "trajectory_set_rerank": TRAJECTORY_SET_RERANK_PROMPT,
        "locomo_judge": LOCOMO_JUDGE_PROMPT,
        "medmt_judge": MEDMT_JUDGE_PROMPT,
        "locomo_judge_structured": LOCOMO_JUDGE_STRUCTURED_PROMPT,
        "medmt_judge_structured": MEDMT_JUDGE_STRUCTURED_PROMPT,
        "semantic_metric_schema": SEMANTIC_METRIC_SCHEMA_PROMPT,
        "semantic_metric_extract": SEMANTIC_METRIC_EXTRACT_PROMPT,
        "retrieval_reflection": RETRIEVAL_REFLECTION_PROMPT,
    }
    assert PROMPT_NAME_TO_VARIABLE == {
        "episodic_extract": "EPISODIC_EXTRACT_PROMPT",
        "episodic_claim_text_extract": "EPISODIC_CLAIM_TEXT_EXTRACT_PROMPT",
        "episodic_claim_text_extract_structured": "EPISODIC_CLAIM_TEXT_EXTRACT_STRUCTURED_PROMPT",
        "claim_signal_extract": "CLAIM_SIGNAL_EXTRACT_PROMPT",
        "claim_signal_extract_structured": "CLAIM_SIGNAL_EXTRACT_STRUCTURED_PROMPT",
        "trajectory_match": "TRAJECTORY_MATCH_PROMPT",
        "claim_transition_judge": "CLAIM_TRANSITION_JUDGE_PROMPT",
        "episodic_extract_structured": "EPISODIC_EXTRACT_STRUCTURED_PROMPT",
        "trajectory_match_structured": "TRAJECTORY_MATCH_STRUCTURED_PROMPT",
        "claim_transition_judge_structured": "CLAIM_TRANSITION_JUDGE_STRUCTURED_PROMPT",
        "repair": "STRUCTURE_REPAIR_PROMPT",
        "episodic_claim_preservation_repair": "EPISODIC_CLAIM_PRESERVATION_REPAIR_PROMPT",
        "answer_generation": "ANSWER_GENERATION_PROMPT",
        "locomo_answer_generation": "LOCOMO_ANSWER_GENERATION_PROMPT",
        "locomo_answer_evidence_synthesis": "LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT",
        "locomo_answer_freeform": "LOCOMO_ANSWER_FREEFORM_PROMPT",
        "locomo_answer_type_verification": "LOCOMO_ANSWER_TYPE_VERIFICATION_PROMPT",
        "locomo_answer_count_validation": "LOCOMO_ANSWER_COUNT_VALIDATION_PROMPT",
        "locomo_answer_repair": "LOCOMO_ANSWER_REPAIR_PROMPT",
        "locomo_answer_repair_arbitration": "LOCOMO_ANSWER_REPAIR_ARBITRATION_PROMPT",
        "medmt_answer_generation": "MEDMT_ANSWER_GENERATION_PROMPT",
        "trajectory_retrieval_summary": "TRAJECTORY_RETRIEVAL_SUMMARY_PROMPT",
        "wiki_page_plan": "WIKI_PAGE_PLAN_PROMPT",
        "wiki_page_compile": "WIKI_PAGE_COMPILE_PROMPT",
        "wiki_page_rerank": "WIKI_PAGE_RERANK_PROMPT",
        "trajectory_set_rerank": "TRAJECTORY_SET_RERANK_PROMPT",
        "locomo_judge": "LOCOMO_JUDGE_PROMPT",
        "medmt_judge": "MEDMT_JUDGE_PROMPT",
        "locomo_judge_structured": "LOCOMO_JUDGE_STRUCTURED_PROMPT",
        "medmt_judge_structured": "MEDMT_JUDGE_STRUCTURED_PROMPT",
        "semantic_metric_schema": "SEMANTIC_METRIC_SCHEMA_PROMPT",
        "semantic_metric_extract": "SEMANTIC_METRIC_EXTRACT_PROMPT",
        "retrieval_reflection": "RETRIEVAL_REFLECTION_PROMPT",
    }
    for name, prompt_value in expected_map.items():
        assert PROMPT_REGISTRY[name] == prompt_value
        assert load_prompt(name) == prompt_value


def test_core_prompts_keep_expected_task_headers():
    assert ANSWER_GENERATION_PROMPT.startswith("TASK=ANSWER_GENERATION")
    assert EPISODIC_CLAIM_TEXT_EXTRACT_PROMPT.startswith("TASK=EPISODIC_CLAIM_TEXT_EXTRACT")
    assert CLAIM_SIGNAL_EXTRACT_PROMPT.startswith("TASK=CLAIM_SIGNAL_EXTRACT")
    assert LOCOMO_ANSWER_GENERATION_PROMPT.startswith("TASK=LOCOMO_ANSWER_GENERATION")
    assert LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT.startswith("TASK=LOCOMO_ANSWER_EVIDENCE_SYNTHESIS")
    assert LOCOMO_ANSWER_FREEFORM_PROMPT.startswith("TASK=LOCOMO_ANSWER_FREEFORM")
    assert LOCOMO_ANSWER_TYPE_VERIFICATION_PROMPT.startswith("TASK=LOCOMO_ANSWER_TYPE_VERIFICATION")
    assert LOCOMO_ANSWER_REPAIR_PROMPT.startswith("TASK=LOCOMO_ANSWER_REPAIR")
    assert LOCOMO_ANSWER_REPAIR_ARBITRATION_PROMPT.startswith("TASK=LOCOMO_ANSWER_REPAIR_ARBITRATION")
    assert MEDMT_ANSWER_GENERATION_PROMPT.startswith("TASK=MEDMT_ANSWER_GENERATION")
    assert TRAJECTORY_RETRIEVAL_SUMMARY_PROMPT.startswith("TASK=TRAJECTORY_RETRIEVAL_SUMMARY")
    assert WIKI_PAGE_PLAN_PROMPT.startswith("TASK=WIKI_PAGE_PLAN")
    assert WIKI_PAGE_COMPILE_PROMPT.startswith("TASK=WIKI_PAGE_COMPILE")
    assert WIKI_PAGE_RERANK_PROMPT.startswith("TASK=WIKI_PAGE_RERANK")
    assert TRAJECTORY_SET_RERANK_PROMPT.startswith("TASK=TRAJECTORY_SET_RERANK")
    assert LOCOMO_JUDGE_PROMPT.startswith("TASK=LOCOMO_JUDGE")
    assert MEDMT_JUDGE_PROMPT.startswith("TASK=MEDMT_JUDGE")
    assert SEMANTIC_METRIC_SCHEMA_PROMPT.startswith("TASK=SEMANTIC_METRIC_SCHEMA")
    assert SEMANTIC_METRIC_EXTRACT_PROMPT.startswith("TASK=SEMANTIC_METRIC_EXTRACT")
    assert RETRIEVAL_REFLECTION_PROMPT.startswith("TASK=RETRIEVAL_REFLECTION")
    assert EPISODIC_EXTRACT_STRUCTURED_PROMPT.startswith("TASK=EPISODIC_EXTRACTION_STRUCTURED")
    assert TRAJECTORY_MATCH_STRUCTURED_PROMPT.startswith("TASK=TRAJECTORY_MATCH_STRUCTURED")
    assert CLAIM_TRANSITION_JUDGE_PROMPT.startswith("TASK=CLAIM_TRANSITION_JUDGE")
    assert CLAIM_TRANSITION_JUDGE_STRUCTURED_PROMPT.startswith("TASK=CLAIM_TRANSITION_JUDGE_STRUCTURED")
    assert EPISODIC_CLAIM_PRESERVATION_REPAIR_PROMPT.startswith("TASK=EPISODIC_CLAIM_PRESERVATION_REPAIR")
    assert LOCOMO_JUDGE_STRUCTURED_PROMPT.startswith("TASK=LOCOMO_JUDGE_STRUCTURED")
    assert MEDMT_JUDGE_STRUCTURED_PROMPT.startswith("TASK=MEDMT_JUDGE_STRUCTURED")


def test_locomo_answer_prompt_pushes_exact_and_complete_answers():
    assert "include all in-scope supported items" in LOCOMO_ANSWER_GENERATION_PROMPT
    assert "cover each entity explicitly" in LOCOMO_ANSWER_GENERATION_PROMPT
    assert "exact count only when completed distinct events are supported" in LOCOMO_ANSWER_GENERATION_PROMPT
    assert "unsupported extras" in LOCOMO_ANSWER_GENERATION_PROMPT
    assert "broad paraphrases" in LOCOMO_ANSWER_GENERATION_PROMPT
    assert "conversational filler" in LOCOMO_ANSWER_GENERATION_PROMPT
    assert "duplicate event mentions" in LOCOMO_ANSWER_GENERATION_PROMPT
    assert "distinct completed events" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "future plans" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "natural language, not a bare number" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "retrieved-evidence lower bound" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "source line date=... fields" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "## Temporal Anchors" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "yesterday" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "Date/time answers" in LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT
    assert "source-derived" in LOCOMO_ANSWER_COUNT_VALIDATION_PROMPT
    assert "pronoun plus image caption" in LOCOMO_ANSWER_COUNT_VALIDATION_PROMPT
    assert "plain natural-language answer text only" in LOCOMO_ANSWER_REPAIR_PROMPT
    assert "Do not return JSON" in LOCOMO_ANSWER_REPAIR_PROMPT


def test_wiki_and_trajectory_prompts_include_historical_drift_rules():
    assert "historical evidence cards" in WIKI_PAGE_PLAN_PROMPT
    assert "historical evidence cards" in WIKI_PAGE_COMPILE_PROMPT
    assert "older but query-relevant facts" in TRAJECTORY_RETRIEVAL_SUMMARY_PROMPT
    assert "sunset" in WIKI_PAGE_COMPILE_PROMPT
    assert "sunset" in TRAJECTORY_RETRIEVAL_SUMMARY_PROMPT
    assert "Same person, broad topic, or generic words" in TRAJECTORY_MATCH_PROMPT
    assert "same evolving event, project, item, place, count, or fact thread" in TRAJECTORY_MATCH_PROMPT
    assert "Same person, broad topic, or generic words" in TRAJECTORY_MATCH_STRUCTURED_PROMPT


def test_episodic_prompts_make_claims_the_only_prompt_target():
    assert "MEMORY_TYPE:" not in EPISODIC_EXTRACT_PROMPT
    assert "[CLAIMS]" not in EPISODIC_EXTRACT_PROMPT
    assert "[OPS]" not in EPISODIC_EXTRACT_PROMPT
    assert "TIMESTAMP:" not in EPISODIC_EXTRACT_PROMPT
    assert "LINKS:" not in EPISODIC_EXTRACT_PROMPT
    assert "STATUS_FLAGS:" not in EPISODIC_EXTRACT_PROMPT
    assert "claim_id=" not in EPISODIC_EXTRACT_PROMPT
    assert "Do not output claims" in EPISODIC_EXTRACT_PROMPT
    assert "Claims are generated by a separate dedicated claim-text extraction stage." in EPISODIC_EXTRACT_PROMPT
    assert "Memory type is determined by the task and must not be emitted by the model." in EPISODIC_EXTRACT_PROMPT
    assert "Do not return NO_MEMORY when a concrete fact appears after an acknowledgement" in EPISODIC_EXTRACT_PROMPT
    assert "NO_MEMORY is only a low-value label for diagnostics" in EPISODIC_EXTRACT_PROMPT
    assert "Always create a memory seed for named places" in EPISODIC_EXTRACT_PROMPT
    assert "Return only: summary_content, context, and keywords." in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    assert "nature-inspired artwork" in EPISODIC_CLAIM_TEXT_EXTRACT_PROMPT
    assert "nature-inspired artwork" in EPISODIC_CLAIM_TEXT_EXTRACT_STRUCTURED_PROMPT
    assert "Do not include claims, claim statuses, source_message_ids" in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    assert "summary_content must be a single event sentence" in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    assert "keywords must be short keywords or short phrases" in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    assert (
        "Do not set has_memory=false when a concrete fact appears after an acknowledgement"
        in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    )
    assert "has_memory=false is only a low-value diagnostic label" in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    assert "Always create a memory seed for named places" in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    assert "Claims are generated by a separate dedicated claim-text extraction stage." in EPISODIC_EXTRACT_STRUCTURED_PROMPT
    assert "supporting_quote" in EPISODIC_CLAIM_TEXT_EXTRACT_PROMPT
    assert "item-level claim for every stated item" in EPISODIC_CLAIM_TEXT_EXTRACT_PROMPT


def test_prompts_do_not_contain_previous_output_regression():
    for name, prompt in PROMPT_REGISTRY.items():
        assert "previous_output" not in prompt, name


def test_structured_prompts_use_structured_output_channel_phrase():
    structured_prompt_names = [
        "episodic_claim_text_extract_structured",
        "claim_signal_extract_structured",
        "episodic_extract_structured",
        "trajectory_match_structured",
        "claim_transition_judge_structured",
        "locomo_judge_structured",
        "medmt_judge_structured",
        "semantic_metric_schema",
        "semantic_metric_extract",
        "retrieval_reflection",
    ]
    for name in structured_prompt_names:
        assert "structured output channel" in PROMPT_REGISTRY[name]


def test_judge_prompts_share_semantic_bodies_across_text_and_structured_variants():
    for shared_snippet in [
        "Allow semantic equivalence, near-synonyms, ordering/format variants",
        "Accept date variants for the same time point",
        "unanchored relative date such as \"last Tuesday\" is PARTIAL",
        "Harmless same-category extras are allowed",
        "Counts are strict",
        "unqualified wrong count is INCORRECT",
        "retrieved-evidence lower bound",
        "Missing a required list item is PARTIAL",
    ]:
        assert shared_snippet in LOCOMO_JUDGE_PROMPT
        assert shared_snippet in LOCOMO_JUDGE_STRUCTURED_PROMPT

    for shared_snippet in [
        "using only the supplied dialogue and rubric",
        "Generally helpful medical advice is not enough unless it satisfies the rubric",
        "Mark generic, ambiguous, or key-step-missing answers INCORRECT.",
        "correct earlier person, condition, symptom, medication, time, or detail",
        "follow the governing system/task constraint",
        "explicitly identify the contradiction and request confirmation or clarification",
    ]:
        assert shared_snippet in MEDMT_JUDGE_PROMPT
        assert shared_snippet in MEDMT_JUDGE_STRUCTURED_PROMPT

    assert "Respond with ONLY ONE WORD:" in LOCOMO_JUDGE_PROMPT
    assert "Respond with ONLY ONE WORD:" in MEDMT_JUDGE_PROMPT
    assert "Respond with ONLY ONE WORD:" not in LOCOMO_JUDGE_STRUCTURED_PROMPT
    assert "Respond with ONLY ONE WORD:" not in MEDMT_JUDGE_STRUCTURED_PROMPT
    assert "PARTIAL" in LOCOMO_JUDGE_PROMPT
    assert "PARTIAL" in LOCOMO_JUDGE_STRUCTURED_PROMPT
    assert "PARTIAL" in MEDMT_JUDGE_PROMPT
    assert "PARTIAL" in MEDMT_JUDGE_STRUCTURED_PROMPT


def test_matching_prompts_use_candidate_labels_instead_of_real_ids():
    assert "TRAJECTORY_ID:" not in TRAJECTORY_MATCH_PROMPT
    assert "PREVIOUS_CLAIM_ID:" not in CLAIM_TRANSITION_JUDGE_PROMPT
    assert "SELECTED_CANDIDATE:" in TRAJECTORY_MATCH_PROMPT
    assert "SELECTED_CANDIDATE:" in CLAIM_TRANSITION_JUDGE_PROMPT
    assert "one short sentence" in TRAJECTORY_MATCH_PROMPT
    assert "one short sentence" in CLAIM_TRANSITION_JUDGE_PROMPT
    assert "Do not say candidates are merely related or similar." in TRAJECTORY_MATCH_PROMPT
    assert "Do not say the claims are merely related or similar." in CLAIM_TRANSITION_JUDGE_PROMPT
    assert "selected_candidate" in TRAJECTORY_MATCH_STRUCTURED_PROMPT
    assert "selected_candidate" in CLAIM_TRANSITION_JUDGE_STRUCTURED_PROMPT
