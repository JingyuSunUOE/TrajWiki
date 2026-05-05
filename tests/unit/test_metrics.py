from __future__ import annotations

import pytest

from trajpatch.utils.metrics import bleu1


def test_bleu1_is_case_insensitive_for_identical_content() -> None:
    assert bleu1("SWEDEN", "sweden") == 1.0


def test_bleu1_does_not_apply_brevity_penalty() -> None:
    assert bleu1("sweden", "sweden home country") == 1.0


def test_bleu1_penalizes_extra_candidate_tokens() -> None:
    assert bleu1("sweden extra", "sweden") == 0.5


def test_bleu1_count_mismatch_keeps_shared_slot_tokens() -> None:
    assert bleu1("tournament_count: 5", "tournament_count: 7") == pytest.approx(2 / 3)
