from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trajpatch.config import RunConfig
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.pipeline.semantic_metrics import SemanticMetricEvaluator, SemanticMetricResult
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.structured_outputs import parse_structured_payload
from trajpatch.types import AnswerResult, JudgeResult, LLMResponse, ModelInfo, StructuredLLMResponse


class _SemanticMetricProvider(LLMProvider):
    def __init__(self, payloads_by_task: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.payloads_by_task = {
            task: list(payloads)
            for task, payloads in (payloads_by_task or {}).items()
        }
        self.structured_calls: list[str] = []
        self.text_calls = 0

    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        self.text_calls += 1
        raise RuntimeError("text fallback should not be used in this test")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="semantic-test-model", is_remote=True)

    def supports_structured(self, task: str) -> bool:
        return True

    def generate_structured(
        self,
        messages,
        *,
        spec,
        system_prompt=None,
        metadata=None,
    ) -> StructuredLLMResponse:
        task = str((metadata or {}).get("task") or spec.task)
        self.structured_calls.append(task)
        payloads = self.payloads_by_task.get(task) or []
        if not payloads:
            raise RuntimeError(f"no payload configured for {task}")
        parsed = parse_structured_payload(spec, payloads.pop(0))
        return StructuredLLMResponse(
            parsed=parsed,
            prompt_tokens=11,
            completion_tokens=3,
            metadata=dict(metadata or {}),
        )


class _BadTextSemanticMetricProvider(LLMProvider):
    def __init__(self) -> None:
        self.text_calls = 0

    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        self.text_calls += 1
        return LLMResponse(
            text="not json",
            prompt_tokens=7,
            completion_tokens=2,
            metadata=dict(metadata or {}),
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="bad-text-model", is_remote=True)

    def supports_structured(self, task: str) -> bool:
        return False


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        dataset="locomo",
        dataset_path=tmp_path,
        output_dir=tmp_path / "output",
        memory_cache_dir=tmp_path / "cache",
        provider_kind="mock",
        embedding_model="hash-embedding",
    )


def _schema(slot_id: str = "items") -> dict[str, Any]:
    return {
        "question_type": "list",
        "slots": [
            {
                "slot_id": slot_id,
                "description": "Required answer items.",
                "value_type": "list",
                "required": True,
            }
        ],
    }


def _extraction(slot_id: str, values: list[str]) -> dict[str, Any]:
    return {"slots": [{"slot_id": slot_id, "canonical_values": values}]}


def test_semantic_metrics_compute_canonical_set_f1_and_no_bp_bleu_for_partial_list(
    tmp_path: Path,
) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("activities")],
            "semantic_metric_extract": [
                _extraction("activities", ["pride parade", "school speech", "support group"]),
                _extraction("activities", ["pride parade"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="What activities did Sam mention?",
        reference_answer="Pride parade, school speech, support group",
        candidate_answer="Pride parade",
        query_task_id="q-list-partial",
    )

    assert result.f1 == pytest.approx(0.5)
    assert result.bleu_1 == 1.0
    assert result.prompt_tokens == 33
    assert result.completion_tokens == 9
    assert result.metadata["mode"] == "structured"
    assert result.metadata["f1_policy"] == "semantic_canonical_set_f1_v2"
    assert result.metadata["bleu_policy"] == "semantic_canonical_bleu1_no_bp_v1"
    assert result.metadata["f1_precision"] == 1.0
    assert result.metadata["f1_recall"] == pytest.approx(1 / 3)
    assert result.metadata["f1_reference_items"] == [
        "activities: pride parade",
        "activities: school speech",
        "activities: support group",
    ]
    assert result.metadata["f1_candidate_items"] == ["activities: pride parade"]
    assert result.metadata["f1_overlap_items"] == ["activities: pride parade"]
    assert result.metadata["f1_soft_overlap_items"] == ["activities: pride parade"]
    assert "semantic_slot_f1" not in result.metadata
    assert "semantic_canonical_bleu1" not in result.metadata
    assert result.metadata["cache_hits"] == {
        "schema": False,
        "reference_extract": False,
        "candidate_extract": False,
    }


def test_semantic_metrics_canonical_set_f1_penalizes_overgenerated_extra_values(
    tmp_path: Path,
) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("books")],
            "semantic_metric_extract": [
                _extraction("books", ["dune", "the hobbit", "parable of the sower"]),
                _extraction("books", ["dune", "the hobbit", "parable of the sower", "foundation"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="Which books were mentioned?",
        reference_answer="Dune, The Hobbit, Parable of the Sower",
        candidate_answer="Dune, The Hobbit, Parable of the Sower, Foundation",
        query_task_id="q-overgenerated",
    )

    assert result.f1 == pytest.approx(6 / 7)
    assert result.f1 < 1.0
    assert result.bleu_1 == pytest.approx(8 / 9)


def test_semantic_metrics_reward_canonical_paraphrase_equivalence(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("place")],
            "semantic_metric_extract": [
                _extraction("place", ["new york city"]),
                _extraction("place", ["new york city"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="Where did Alex move?",
        reference_answer="NYC",
        candidate_answer="New York City",
        query_task_id="q-paraphrase",
    )

    assert result.f1 == 1.0
    assert result.bleu_1 == 1.0


def test_semantic_metrics_canonical_set_f1_handles_empty_extractions(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("items"), _schema("items")],
            "semantic_metric_extract": [
                _extraction("items", []),
                _extraction("items", []),
                _extraction("items", ["tea"]),
                _extraction("items", []),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    empty_result = evaluator.evaluate_locomo(
        question="What was mentioned?",
        reference_answer="",
        candidate_answer="",
        query_task_id="q-empty-both",
    )
    one_sided_result = evaluator.evaluate_locomo(
        question="What was mentioned?",
        reference_answer="Tea",
        candidate_answer="",
        query_task_id="q-empty-candidate",
    )

    assert empty_result.f1 == 1.0
    assert empty_result.metadata["f1_precision"] == 1.0
    assert empty_result.metadata["f1_recall"] == 1.0
    assert one_sided_result.f1 == 0.0


def test_semantic_metrics_canonical_set_f1_reduces_long_answer_penalty(
    tmp_path: Path,
) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("events")],
            "semantic_metric_extract": [
                _extraction("events", ["pride parade", "school speech", "support group"]),
                _extraction(
                    "events",
                    [
                        "support group",
                        "conference",
                        "school speech",
                        "pride parade",
                        "art show",
                    ],
                ),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="What LGBTQ+ events has Caroline participated in?",
        reference_answer="Pride parade, school speech, support group",
        candidate_answer=(
            "1. Caroline attended an LGBTQ support group yesterday.\n"
            "2. Caroline attended an LGBTQ conference two days ago.\n"
            "3. Caroline discussed her transgender journey at a school event last week.\n"
            "4. Caroline attended an LGBTQ pride parade last week.\n"
            "5. Caroline is organizing an LGBTQ art show next month."
        ),
        query_task_id="q-long-answer",
    )

    assert result.f1 == pytest.approx(0.75)
    assert result.metadata["f1_precision"] == pytest.approx(3 / 5)
    assert result.metadata["f1_recall"] == 1.0
    assert result.metadata["f1_overlap_items"] == [
        "events: pride parade",
        "events: school speech",
        "events: support group",
    ]


def test_semantic_metrics_soft_matches_benign_modifiers(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("events")],
            "semantic_metric_extract": [
                _extraction("events", ["pride parade", "school speech", "support group"]),
                _extraction("events", ["LGBTQ support group", "LGBTQ pride parade", "conference"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="What LGBTQ+ events has Caroline participated in?",
        reference_answer="Pride parade, school speech, support group",
        candidate_answer="LGBTQ support group, LGBTQ pride parade, conference",
        query_task_id="q-soft-modifiers",
    )

    assert result.f1 == pytest.approx(2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)))
    assert result.metadata["f1_exact_overlap_items"] == []
    assert result.metadata["f1_soft_overlap_items"] == [
        "events: pride parade",
        "events: support group",
    ]
    assert result.metadata["f1_unmatched_reference_items"] == ["events: school speech"]
    assert {pair["reason"] for pair in result.metadata["f1_soft_match_pairs"]} == {
        "modifier_containment"
    }


def test_semantic_metrics_soft_matches_aliases_and_safe_containment(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("place")],
            "semantic_metric_extract": [
                _extraction("place", ["New York"]),
                _extraction("place", ["NYC"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="Where did Alex move?",
        reference_answer="New York",
        candidate_answer="NYC",
        query_task_id="q-alias",
    )

    assert result.f1 == 1.0
    assert result.metadata["f1_soft_match_pairs"][0]["reason"] == "alias"


def test_semantic_metrics_soft_matches_event_aliases(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("events")],
            "semantic_metric_extract": [
                _extraction("events", ["mentoring program", "school speech"]),
                _extraction(
                    "events",
                    [
                        "mentorship program for LGBTQ youth",
                        "shared her journey at a school event and encouraged students",
                    ],
                ),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="What events has Caroline participated in to help children?",
        reference_answer="Mentoring program, school speech",
        candidate_answer=(
            "Caroline joined a mentorship program for LGBTQ youth and shared her journey at a school event."
        ),
        query_task_id="q-event-alias",
    )

    assert result.f1 == 1.0
    assert result.metadata["f1_unmatched_reference_items"] == []
    assert {pair["reason"] for pair in result.metadata["f1_soft_match_pairs"]} == {"phrase_alias"}


def test_semantic_metrics_does_not_match_bare_school_event_to_speech(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("events")],
            "semantic_metric_extract": [
                _extraction("events", ["school speech"]),
                _extraction("events", ["school event"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="What events did Caroline attend?",
        reference_answer="school speech",
        candidate_answer="school event",
        query_task_id="q-bare-school-event",
    )

    assert result.f1 == 0.0
    assert result.metadata["f1_soft_match_pairs"] == []


def test_semantic_metrics_soft_matches_surface_aliases(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("items")],
            "semantic_metric_extract": [
                _extraction("items", ["Kingkiller Chronicles", "gaming convention", "streetfighter"]),
                _extraction("items", ["Kingkiller Chronicle", "game convention", "street fighter"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="Which items were mentioned?",
        reference_answer="Kingkiller Chronicles, gaming convention, streetfighter",
        candidate_answer="Kingkiller Chronicle, game convention, street fighter",
        query_task_id="q-surface-alias",
    )

    assert result.f1 == 1.0
    assert result.metadata["f1_unmatched_reference_items"] == []


def test_semantic_metrics_contrastive_modifiers_prevent_overmatching(tmp_path: Path) -> None:
    provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("cars")],
            "semantic_metric_extract": [
                _extraction("cars", ["old Prius", "new Prius"]),
                _extraction("cars", ["Prius"]),
            ],
        }
    )
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="Which cars were mentioned?",
        reference_answer="old Prius, new Prius",
        candidate_answer="Prius",
        query_task_id="q-contrastive",
    )

    assert result.f1 == 0.0
    assert result.metadata["f1_soft_match_pairs"] == []
    assert result.metadata["f1_unmatched_reference_items"] == [
        "cars: new prius",
        "cars: old prius",
    ]


def _answer_result_for_override(question: str, answer_text: str, gold_answer: str) -> AnswerResult:
    return AnswerResult(
        dataset_name="locomo",
        sample_id="conv-test",
        subset_key="multi_hop",
        scene_tag=None,
        query_task_id="q-override",
        question=question,
        gold_answer=gold_answer,
        rubric=None,
        answer_prompt="prompt",
        answer_text=answer_text,
        answer_record_id="answer-1",
        retrieval_event_id="retrieval-1",
        retrieval_source_refs=[],
        retrieval_source_ids=[],
        metadata={},
    )


def test_locomo_judge_semantic_override_promotes_full_coverage_date_answer() -> None:
    runner = PipelineRunner.__new__(PipelineRunner)
    answer_result = _answer_result_for_override(
        "When did Melanie go on a hike after the roadtrip?",
        "The hike happened on 19 October 2023, the day after their road trip.",
        "19 October 2023",
    )
    judge_result = JudgeResult(
        verdict="partial",
        prompt="judge prompt",
        score=0.5,
        rationale="Too much explanation.",
        metadata={},
    )
    semantic_result = SemanticMetricResult(
        f1=1.0,
        bleu_1=1.0,
        metadata={
            "f1_reference_items": ["date: 19 october 2023"],
            "f1_exact_overlap_items": ["date: 19 october 2023"],
            "f1_unmatched_reference_items": [],
        },
    )

    overridden, judge_acc = runner._apply_locomo_judge_semantic_override(
        answer_result,
        judge_result,
        0.5,
        semantic_result,
    )

    assert overridden.verdict == "correct"
    assert overridden.score == 1.0
    assert judge_acc == 1.0
    assert overridden.metadata["raw_judge_verdict"] == "partial"
    assert overridden.metadata["judge_semantic_override_used"] is True


def test_locomo_judge_semantic_override_does_not_promote_wrong_count() -> None:
    runner = PipelineRunner.__new__(PipelineRunner)
    answer_result = _answer_result_for_override(
        "How many times has Joanna's scripts been rejected?",
        "One.",
        "Twice",
    )
    judge_result = JudgeResult(verdict="incorrect", prompt="judge prompt", score=0.0, metadata={})
    semantic_result = SemanticMetricResult(
        f1=1.0,
        bleu_1=1.0,
        metadata={
            "f1_reference_items": ["count: twice"],
            "f1_exact_overlap_items": [],
            "f1_unmatched_reference_items": [],
        },
    )

    unchanged, judge_acc = runner._apply_locomo_judge_semantic_override(
        answer_result,
        judge_result,
        0.0,
        semantic_result,
    )

    assert unchanged is judge_result
    assert judge_acc == 0.0


def test_semantic_metrics_cache_hits_do_not_add_tokens(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("items")],
            "semantic_metric_extract": [
                _extraction("items", ["tea"]),
                _extraction("items", ["tea"]),
            ],
        }
    )
    first = SemanticMetricEvaluator(first_provider, config).evaluate_locomo(
        question="What does Alice prefer?",
        reference_answer="Tea",
        candidate_answer="Tea",
        query_task_id="q-cache",
    )

    second_provider = _SemanticMetricProvider({})
    second = SemanticMetricEvaluator(second_provider, config).evaluate_locomo(
        question="What does Alice prefer?",
        reference_answer="Tea",
        candidate_answer="Tea",
        query_task_id="q-cache",
    )

    assert first.prompt_tokens == 33
    assert first.metadata["cache_miss_count"] == 3
    assert first.metadata["llm_call_count"] == 3
    assert second.prompt_tokens == 0
    assert second.completion_tokens == 0
    assert second.metadata["mode"] == "cache"
    assert all(second.metadata["cache_hits"].values())
    assert second.metadata["cache_hit_count"] == 3
    assert second.metadata["llm_call_count"] == 0
    assert second_provider.structured_calls == []


def test_rebuild_memory_cache_does_not_bypass_semantic_metric_cache(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("items")],
            "semantic_metric_extract": [
                _extraction("items", ["tea"]),
                _extraction("items", ["tea"]),
            ],
        }
    )
    SemanticMetricEvaluator(first_provider, config).evaluate_locomo(
        question="What does Alice prefer?",
        reference_answer="Tea",
        candidate_answer="Tea",
        query_task_id="q-cache-memory-rebuild",
    )

    cache_reusing_config = config.copy(update={"rebuild_memory_cache": True})
    second_provider = _SemanticMetricProvider({})
    second = SemanticMetricEvaluator(second_provider, cache_reusing_config).evaluate_locomo(
        question="What does Alice prefer?",
        reference_answer="Tea",
        candidate_answer="Tea",
        query_task_id="q-cache-memory-rebuild",
    )

    assert second.metadata["mode"] == "cache"
    assert second.metadata["cache_hit_count"] == 3
    assert second.metadata["llm_call_count"] == 0
    assert second_provider.structured_calls == []


def test_rebuild_semantic_metric_cache_bypasses_semantic_cache(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("items")],
            "semantic_metric_extract": [
                _extraction("items", ["tea"]),
                _extraction("items", ["tea"]),
            ],
        }
    )
    SemanticMetricEvaluator(first_provider, config).evaluate_locomo(
        question="What does Alice prefer?",
        reference_answer="Tea",
        candidate_answer="Tea",
        query_task_id="q-cache-semantic-rebuild",
    )

    rebuilding_config = config.copy(update={"rebuild_semantic_metric_cache": True})
    second_provider = _SemanticMetricProvider(
        {
            "semantic_metric_schema": [_schema("items")],
            "semantic_metric_extract": [
                _extraction("items", ["tea"]),
                _extraction("items", ["coffee"]),
            ],
        }
    )
    second = SemanticMetricEvaluator(second_provider, rebuilding_config).evaluate_locomo(
        question="What does Alice prefer?",
        reference_answer="Tea",
        candidate_answer="Tea",
        query_task_id="q-cache-semantic-rebuild",
    )

    assert second.metadata["mode"] == "structured"
    assert second.metadata["cache_miss_count"] == 3
    assert second.metadata["llm_call_count"] == 3
    assert second_provider.structured_calls == [
        "semantic_metric_schema",
        "semantic_metric_extract",
        "semantic_metric_extract",
    ]
    assert second.f1 == 0.0


def test_semantic_metric_failure_uses_deterministic_fallback(tmp_path: Path) -> None:
    evaluator = SemanticMetricEvaluator(None, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="What activities were mentioned?",
        reference_answer="Pride parade, school speech",
        candidate_answer="Pride parade",
        query_task_id="q-fallback",
    )

    assert result.f1 == pytest.approx(2 / 3)
    assert result.bleu_1 == 1.0
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.metadata["mode"] == "deterministic_fallback"
    assert result.metadata["cache_miss_count"] == 3
    assert result.metadata["llm_call_count"] == 0
    assert result.metadata["error"]


def test_semantic_metric_text_fallback_parse_failure_counts_llm_calls(tmp_path: Path) -> None:
    provider = _BadTextSemanticMetricProvider()
    evaluator = SemanticMetricEvaluator(provider, _config(tmp_path))

    result = evaluator.evaluate_locomo(
        question="What activities were mentioned?",
        reference_answer="Pride parade, school speech",
        candidate_answer="Pride parade",
        query_task_id="q-bad-text-fallback",
    )

    assert result.metadata["mode"] == "deterministic_fallback"
    assert result.metadata["cache_miss_count"] == 3
    assert result.metadata["llm_call_count"] == 3
    assert result.prompt_tokens == 21
    assert result.completion_tokens == 6
    assert provider.text_calls == 3
    assert "text fallback failed" in result.metadata["error"]
