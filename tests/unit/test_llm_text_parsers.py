from __future__ import annotations

from trajpatch.memory.llm_text_parsers import (
    parse_claim_transition_decision,
    parse_episodic_memory,
    parse_judge_verdict,
    parse_match_decision,
)


def test_match_decision_ignores_qwen_thinking_block() -> None:
    parsed = parse_match_decision(
        "<think>\nI should compare the candidates first.\n</think>\n"
        "DECISION: CONTINUE\n"
        "SELECTED_CANDIDATE: T1\n"
        "RATIONALE: The memory continues the same adoption-planning thread.",
        {"T1": "traj-1"},
    )

    assert parsed.decision == "CONTINUE"
    assert parsed.trajectory_id == "traj-1"


def test_match_decision_ignores_unclosed_qwen_thinking_preamble() -> None:
    parsed = parse_match_decision(
        "<think>\nThe first candidate is relevant.\n"
        "Reason: T2 has the same topic.\n"
        "DECISION: CONTINUE\n"
        "SELECTED_CANDIDATE: T2\n"
        "RATIONALE: The new claim updates the same trajectory.",
        {"T2": "traj-2"},
    )

    assert parsed.trajectory_id == "traj-2"


def test_other_dsl_parsers_ignore_qwen_thinking_block() -> None:
    transition = parse_claim_transition_decision(
        "<think>\nThis is a revision.\n</think>\n"
        "DECISION: REVISE\n"
        "SELECTED_CANDIDATE: P1\n"
        "RATIONALE: The new claim refines the earlier one.",
        {"P1": "claim-1"},
    )
    episodic = parse_episodic_memory(
        "<think>\nSummarize only stable facts.\n</think>\n"
        "SUMMARY_CONTENT: Caroline researched adoption agencies.\n"
        "CONTEXT: Caroline discussed adoption research.\n"
        "KEYWORDS: Caroline, adoption, agencies",
        {"sample-m0000"},
        exchange_link_ids=["sample-m0000"],
    )
    verdict = parse_judge_verdict(
        "<think>\nThe answer is partly right.\n</think>\nVERDICT: PARTIAL\nRATIONALE: Missing one detail."
    )

    assert transition.previous_claim_id == "claim-1"
    assert episodic is not None
    assert episodic.summary_content == "Caroline researched adoption agencies."
    assert verdict.verdict == "PARTIAL"
