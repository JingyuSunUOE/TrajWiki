from __future__ import annotations

import json

from trajpatch.exceptions import StructuredOutputError
from trajpatch.memory.llm_text_parsers import parse_judge_verdict
from trajpatch.pipeline.answering import AnswerGenerator, BenchmarkJudge
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.prompts import (
    ANSWER_GENERATION_PROMPT,
    LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT,
    LOCOMO_ANSWER_FREEFORM_PROMPT,
    LOCOMO_ANSWER_GENERATION_PROMPT,
    LOCOMO_ANSWER_REPAIR_ARBITRATION_PROMPT,
    LOCOMO_ANSWER_REPAIR_PROMPT,
    LOCOMO_JUDGE_PROMPT,
    LOCOMO_JUDGE_STRUCTURED_PROMPT,
    MEDMT_ANSWER_GENERATION_PROMPT,
    MEDMT_JUDGE_PROMPT,
)
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.mock import MockLLMProvider
from trajpatch.providers.structured_outputs import get_structured_task_spec, parse_structured_payload
from trajpatch.types import LLMResponse, ModelInfo, QueryTask, RetrievalBundle, StructuredLLMResponse


class _StructuredJudgeProvider(LLMProvider):
    def __init__(
        self,
        *,
        structured_payload: dict | None = None,
        structured_exception: Exception | None = None,
        text_response: str = "CORRECT",
        text_exception: Exception | None = None,
        supports_structured: bool = True,
    ) -> None:
        self.structured_payload = structured_payload
        self.structured_exception = structured_exception
        self.text_response = text_response
        self.text_exception = text_exception
        self._supports_structured = supports_structured
        self.generate_calls = 0
        self.generate_structured_calls = 0

    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        self.generate_calls += 1
        if self.text_exception is not None:
            raise self.text_exception
        return LLMResponse(text=self.text_response, prompt_tokens=5, completion_tokens=1, metadata=dict(metadata or {}))

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_kind="remote",
            model_name="fake-judge",
            is_remote=True,
            metadata={"vendor": "openai"},
        )

    def supports_structured(self, task: str) -> bool:
        return self._supports_structured

    def generate_structured(self, messages, *, spec, system_prompt=None, metadata=None) -> StructuredLLMResponse:
        self.generate_structured_calls += 1
        if self.structured_exception is not None:
            raise self.structured_exception
        parsed = parse_structured_payload(spec, self.structured_payload or {"verdict": "CORRECT"})
        return StructuredLLMResponse(
            parsed=parsed,
            prompt_tokens=7,
            completion_tokens=1,
            metadata={
                **dict(metadata or {}),
                "structured_vendor": "openai",
                "structured_strategy": "openai_json_schema",
                "structured_success": True,
            },
        )


class _AnswerRepairProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.prompts: list[str] = []

    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        self.calls.append(dict(metadata or {}))
        self.prompts.append(messages[-1].content if messages else "")
        text = self._responses.pop(0)
        return LLMResponse(text=text, prompt_tokens=5, completion_tokens=1, metadata=dict(metadata or {}))

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="mock", model_name="fake-answer", is_remote=False)

    def supports_structured(self, task: str) -> bool:
        return False

    def generate_structured(self, messages, *, spec, system_prompt=None, metadata=None) -> StructuredLLMResponse:
        raise NotImplementedError


class _StructuredAnswerProvider(LLMProvider):
    def __init__(
        self,
        *,
        structured_payload: dict | None = None,
        structured_payloads: list[dict] | None = None,
        count_validation_payload: dict | None = None,
        count_validation_payloads: list[dict] | None = None,
        answer_type_verification_payload: dict | None = None,
        answer_type_verification_payloads: list[dict] | None = None,
        answer_repair_arbitration_payload: dict | None = None,
        answer_repair_arbitration_payloads: list[dict] | None = None,
        structured_exception: Exception | None = None,
        text_response: str = "",
        text_responses: list[str] | None = None,
        supports_structured: bool = True,
    ) -> None:
        self.structured_payload = structured_payload
        self.structured_payloads = list(structured_payloads or [])
        self.count_validation_payload = count_validation_payload
        self.count_validation_payloads = list(count_validation_payloads or [])
        self.answer_type_verification_payload = answer_type_verification_payload
        self.answer_type_verification_payloads = list(answer_type_verification_payloads or [])
        self.answer_repair_arbitration_payload = answer_repair_arbitration_payload
        self.answer_repair_arbitration_payloads = list(answer_repair_arbitration_payloads or [])
        self.structured_exception = structured_exception
        self.text_response = text_response
        self.text_responses = list(text_responses or [])
        self._supports_structured = supports_structured
        self.generate_calls = 0
        self.generate_structured_calls = 0
        self.prompts: list[str] = []

    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        self.generate_calls += 1
        self.prompts.append(messages[-1].content if messages else "")
        text = self.text_responses.pop(0) if self.text_responses else self.text_response
        return LLMResponse(text=text, prompt_tokens=11, completion_tokens=3, metadata=dict(metadata or {}))

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="fake-answer", is_remote=True, metadata={"vendor": "openai"})

    def supports_structured(self, task: str) -> bool:
        if not self._supports_structured:
            return False
        if task == "answer_evidence_synthesis":
            return True
        if task == "answer_count_validation":
            return bool(self.count_validation_payloads or self.count_validation_payload)
        if task == "answer_type_verification":
            return bool(self.answer_type_verification_payloads or self.answer_type_verification_payload)
        if task == "answer_repair_arbitration":
            return bool(self.answer_repair_arbitration_payloads or self.answer_repair_arbitration_payload)
        return False

    def generate_structured(self, messages, *, spec, system_prompt=None, metadata=None) -> StructuredLLMResponse:
        self.generate_structured_calls += 1
        self.prompts.append(messages[-1].content if messages else "")
        if self.structured_exception is not None:
            raise self.structured_exception
        if spec.task == "answer_count_validation":
            payload = (
                self.count_validation_payloads.pop(0)
                if self.count_validation_payloads
                else self.count_validation_payload
            )
            if payload is None:
                raise NotImplementedError("count validation payload not configured")
        elif spec.task == "answer_type_verification":
            payload = (
                self.answer_type_verification_payloads.pop(0)
                if self.answer_type_verification_payloads
                else self.answer_type_verification_payload
            )
            if payload is None:
                raise NotImplementedError("answer type verification payload not configured")
        elif spec.task == "answer_repair_arbitration":
            payload = (
                self.answer_repair_arbitration_payloads.pop(0)
                if self.answer_repair_arbitration_payloads
                else self.answer_repair_arbitration_payload
            )
            if payload is None:
                raise NotImplementedError("answer repair arbitration payload not configured")
        else:
            payload = (
                self.structured_payloads.pop(0)
                if self.structured_payloads
                else self.structured_payload or _answer_synthesis_payload("Single")
            )
        parsed = parse_structured_payload(spec, payload)
        return StructuredLLMResponse(parsed=parsed, prompt_tokens=17, completion_tokens=5, metadata=dict(metadata or {}))


class _FailingGenerateAnswerProvider(_StructuredAnswerProvider):
    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        self.generate_calls += 1
        self.prompts.append(messages[-1].content if messages else "")
        raise RuntimeError("provider unavailable")


class _FailingRepairAfterInitialProvider(_AnswerRepairProvider):
    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        if self.calls:
            self.calls.append(dict(metadata or {}))
            self.prompts.append(messages[-1].content if messages else "")
            raise RuntimeError("provider unavailable")
        return super().generate(messages, system_prompt=system_prompt, metadata=metadata)


def _answer_synthesis_payload(final_answer: str, *, answer_type: str = "status") -> dict:
    return {
        "can_answer": True,
        "answer_type": answer_type,
        "final_answer": final_answer,
        "supporting_facts": [{"fact_text": "Caroline is a single parent.", "source_refs": ["D2:14"]}],
        "supporting_source_refs": ["D2:14"],
        "counted_events": [],
        "excluded_events": [],
        "uncertainties": [],
        "abstain_reason": None,
    }


def _retrieval_bundle() -> RetrievalBundle:
    return RetrievalBundle(
        retrieval_event_id="retrieval-1",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=[],
        prompt_context="Retrieved memory goes here.",
        latency_ms=1.0,
    )


def _locomo_retrieval_bundle_with_shape() -> RetrievalBundle:
    return RetrievalBundle(
        retrieval_event_id="retrieval-1",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=[],
        prompt_context="Retrieved memory goes here.",
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "book",
                "tags": ["list_like"],
            },
            "grounded_display_items": ["The Hobbit"],
            "grounded_exact_terms": ["The Hobbit"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": ["Tim read The Hobbit."],
        },
    )


def _locomo_preference_retrieval_bundle() -> RetrievalBundle:
    return RetrievalBundle(
        retrieval_event_id="retrieval-preference",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=[],
        prompt_context=(
            "[D6:6] Melanie: They were stoked for the dinosaur exhibit! "
            "They love learning about animals.\n"
            "[D8:4] Melanie: The kids love exploring nature."
        ),
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "preference",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": ["dinosaur exhibit", "learning about animals", "nature"],
            "grounded_source_surface_raw_terms": ["the dinosaur exhibit", "learning about animals", "nature"],
            "grounded_exact_terms": ["dinosaurs", "dinosaur exhibit", "nature"],
            "grounded_display_items": ["learning about animals", "nature"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": [
                "The kids were stoked for the dinosaur exhibit.",
                "The kids love exploring nature.",
            ],
            "wiki_historical_item_terms": ["dinosaur exhibit", "nature"],
        },
    )


def _locomo_count_retrieval_bundle() -> RetrievalBundle:
    return RetrievalBundle(
        retrieval_event_id="retrieval-count",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=[],
        prompt_context="Retrieved count memory goes here.",
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": False,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": True,
                "item_family": None,
                "tags": ["count_like"],
            },
            "grounded_display_items": [],
            "grounded_exact_terms": [],
            "grounded_display_counts": ["2"],
            "grounded_display_key_facts": ["Joanna found new hiking trails twice."],
        },
    )


def _locomo_count_retrieval_bundle_with_refs() -> RetrievalBundle:
    bundle = _locomo_count_retrieval_bundle()
    bundle.source_message_refs.extend(["D8:4", "D11:3", "D11:8", "D14:1", "D24:12", "D27:12"])
    bundle.prompt_context = (
        "[D8:4] Joanna found a new hiking trail.\n"
        "[D11:3] Joanna found another new hiking trail.\n"
        "[D11:8] Nate: Your hikes sound like a blast.\n"
        "[D14:1] Joanna's script was rejected.\n"
        "[D24:12] Joanna said another script was rejected.\n"
        "[D27:12] Joanna assumes a few scripts will be rejected."
    )
    return bundle


def test_answer_generator_uses_medmt_specific_prompt_and_metadata():
    generator = AnswerGenerator(MockLLMProvider())
    query_task = QueryTask(
        query_task_id="medmt-q1",
        sample_id="sample-a",
        question="What do you know about my smoking habits?",
        metadata={
            "answer_context": {
                "dataset": "medmt",
                "category": "Instruction Clarification",
                "subtype": "Information Contradiction",
                "rubric": "Explicitly identify the contradiction and ask for confirmation.",
                "scene_tag": "HC",
                "subset_key": "information_contradiction",
            }
        },
    )

    prompt = generator.build_prompt(query_task, _retrieval_bundle())

    assert prompt.startswith(MEDMT_ANSWER_GENERATION_PROMPT)
    assert "CATEGORY:\nInstruction Clarification" in prompt
    assert "SUBTYPE:\nInformation Contradiction" in prompt
    assert "SCENE_TAG:\nHC" in prompt
    assert "SUBSET_KEY:\ninformation_contradiction" in prompt
    assert "RUBRIC:\nExplicitly identify the contradiction and ask for confirmation." in prompt
    assert "RETRIEVED_CONTEXT:\nRetrieved memory goes here." in prompt
    assert "FINAL_USER_TURN:\nWhat do you know about my smoking habits?" in prompt


def test_answer_generator_returns_failure_response_when_provider_generate_fails():
    provider = _FailingGenerateAnswerProvider(
        structured_exception=StructuredOutputError("structured unavailable"),
        supports_structured=True,
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-49_qa_fail",
        sample_id="conv-49",
        question="What happened?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] ..."},
    )

    prompt, response = generator.generate(query_task, _retrieval_bundle())

    assert prompt
    assert response.text.startswith("The retrieved context does not support an answer")
    assert response.metadata["answer_generation_failed"] is True
    assert response.metadata["answer_generation_error_type"] == "RuntimeError"
    assert response.metadata["answer_postcheck_skip_reason"] == "answer_generation_failed"
    assert provider.generate_calls == 1


def test_answer_generator_discards_failed_repair_and_keeps_initial_answer():
    provider = _FailingRepairAfterInitialProvider(["The Hobbit, Extra Book"])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-book-repair",
        sample_id="conv-book",
        question="What books did Tim read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D2:14] ..."},
    )

    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text == "The Hobbit, Extra Book"
    assert response.metadata["answer_postcheck_used"] is True
    assert response.metadata["answer_repair_attempted"] is True
    assert response.metadata["answer_repair_used"] is False
    assert response.metadata["answer_repair_discard_reason"] == "provider_exception"
    assert response.metadata["answer_repair_error_type"] == "RuntimeError"


def test_answer_generator_uses_locomo_specific_prompt_and_records_prompt_name():
    generator = AnswerGenerator(MockLLMProvider())
    query_task = QueryTask(
        query_task_id="locomo-q1",
        sample_id="conv-1",
        question="What does Alice prefer?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] Alice: I love tea."},
    )

    prompt = generator.build_prompt(query_task, _locomo_retrieval_bundle_with_shape())
    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert prompt.startswith(LOCOMO_ANSWER_GENERATION_PROMPT)
    assert "QUERY_SHAPE_RULES" in prompt
    assert "Question:\nWhat does Alice prefer?" in prompt
    assert response.metadata["answer_prompt_name"] == "locomo_answer_freeform"
    assert response.metadata["answer_prompt_stage"] == "initial"
    assert response.metadata["answer_synthesis_mode"] == "freeform_v2"


def test_answer_generator_falls_back_to_legacy_structured_synthesis_when_freeform_empty():
    provider = _StructuredAnswerProvider(structured_payload=_answer_synthesis_payload("Single"))
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-status",
        sample_id="conv-26",
        question="What is Caroline's relationship status?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D2:14] Caroline is a single parent."},
    )

    prompt, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert prompt.startswith(LOCOMO_ANSWER_EVIDENCE_SYNTHESIS_PROMPT)
    assert response.text == "Single"
    assert provider.generate_structured_calls == 1
    assert provider.generate_calls == 1
    assert response.metadata["answer_prompt_name"] == "locomo_answer_evidence_synthesis"
    assert response.metadata["answer_prompt_stage"] == "synthesis"
    assert response.metadata["answer_synthesis_used"] is True
    assert response.metadata["answer_synthesis_mode"] == "structured"
    assert response.metadata["answer_synthesis_answer_type"] == "status"
    assert response.metadata["answer_synthesis_supporting_refs"] == ["D2:14"]


def test_answer_generator_uses_freeform_locomo_answer_for_plain_text_provider():
    provider = _AnswerRepairProvider(["Answer: Single.\nRationale: D2:14 says Caroline is a single parent."])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-freeform",
        sample_id="conv-26",
        question="What is Caroline's relationship status?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D2:14] Caroline is a single parent."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-freeform",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D2:14"],
        prompt_context="[D2:14] Caroline is a single parent.",
        latency_ms=1.0,
    )

    prompt, response = generator.generate(query_task, bundle)

    assert prompt.startswith(LOCOMO_ANSWER_FREEFORM_PROMPT)
    assert response.text == "Single."
    assert response.metadata["answer_synthesis_mode"] == "freeform_v2"
    assert response.metadata["answer_freeform_used"] is True
    assert response.metadata["answer_freeform_parse_format"] == "answer_rationale"
    assert response.metadata["answer_type_verification_used"] is False
    assert provider.calls[0]["task"] == "answer_freeform_generation"


def test_answer_generator_does_not_treat_when_context_question_as_date_type():
    provider = _AnswerRepairProvider(
        ["Answer: Dave and his father worked on cars in the garage.\nRationale: The retrieved source says they did garage projects."]
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-when-context",
        sample_id="conv-1",
        question="When Dave was a child, what did he and his father do in the garage?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] garage projects"},
    )

    _, response = generator.generate(query_task, _retrieval_bundle())

    assert response.text == "Dave and his father worked on cars in the garage."
    assert response.metadata["answer_expected_type"] != "date"
    assert response.metadata["answer_type_match"] is True


def test_answer_generator_repairs_freeform_type_mismatch_with_lightweight_verifier():
    provider = _StructuredAnswerProvider(
        text_responses=["Answer: one hike\nRationale: The source mentions one hike.", "19 October 2023"],
        answer_type_verification_payload={
            "expected_answer_type": "date",
            "observed_answer_type": "count",
            "type_match": False,
            "issue": "date_question_count_style_answer",
            "repair_instruction": "Return the concrete date from retrieved evidence.",
        },
        supports_structured=True,
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_76-freeform",
        sample_id="conv-26",
        question="When did Melanie go on a hike after the roadtrip?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D18:17] yesterday"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-date-freeform",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D18:17"],
        prompt_context=(
            "[D18:17] date=20 October 2023 | Melanie: We did it yesterday after the road trip.\n"
            '## Temporal Anchors\n- D18:17 occurred at 20 October 2023; "yesterday" refers to 19 October 2023.'
        ),
        latency_ms=1.0,
    )

    prompt, response = generator.generate(query_task, bundle)

    assert response.text == "19 October 2023"
    assert prompt.startswith(LOCOMO_ANSWER_REPAIR_PROMPT)
    assert response.metadata["answer_type_verification_used"] is True
    assert response.metadata["answer_type_match"] is False
    assert response.metadata["answer_postcheck_issue"] == "answer_type_mismatch"
    assert provider.generate_structured_calls == 1


def test_locomo_answer_synthesis_prompt_uses_temporal_anchors_without_gold_leakage():
    provider = _StructuredAnswerProvider(
        structured_payload={
            "can_answer": True,
            "answer_type": "date",
            "final_answer": "19 October 2023",
            "supporting_facts": [
                {"fact_text": "D18:17 says the hike happened yesterday after the road trip.", "source_refs": ["D18:17"]}
            ],
            "supporting_source_refs": ["D18:17"],
            "counted_events": [],
            "excluded_events": [],
            "uncertainties": [],
            "abstain_reason": None,
        }
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_76",
        sample_id="conv-26",
        question="When did Melanie go on a hike after the roadtrip?",
        metadata={"category_name": "multi_hop"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-time",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=["conv-26-m0396"],
        source_message_refs=["D18:17"],
        prompt_context=(
            "## Retrieved Source Messages\n"
            "- D18:17 | date=6:55 pm on 20 October, 2023 | id=conv-26-m0396 | turn=396 | "
            "assistant/Melanie: Yup, we just did it yesterday after the road trip.\n\n"
            "## Temporal Anchors\n"
            '- D18:17 occurred at 20 October 2023; "yesterday" refers to 19 October 2023.'
        ),
        latency_ms=1.0,
    )

    prompt, response = generator.generate(query_task, bundle)

    assert response.text == "19 October 2023"
    assert "date=6:55 pm on 20 October, 2023" in prompt
    assert "## Temporal Anchors" in prompt
    assert "19 October 2023" in prompt
    assert "evidence_only_conversation" not in prompt


def test_time_like_family_validation_accepts_source_date_and_temporal_anchor():
    assert AnswerGenerator._source_text_matches_query_family(
        source_text=(
            '- D18:17 | date=6:55 pm on 20 October, 2023 | assistant/Melanie: '
            "Yup, we just did it yesterday.\n"
            '## Temporal Anchors\n- D18:17 occurred at 20 October 2023; "yesterday" refers to 19 October 2023.'
        ),
        question="When did Melanie go on a hike after the roadtrip?",
        query_shape={"count_like": False, "item_family": None},
    )


def test_answer_generator_parses_freeform_json_answer_response():
    provider = _StructuredAnswerProvider(
        supports_structured=False,
        text_response='{"can_answer":true,"answer_type":"status","final_answer":"Single",'
        '"supporting_facts":[{"fact_text":"Caroline is a single parent.","source_refs":["D2:14"]}],'
        '"supporting_source_refs":["D2:14"],"counted_events":[],"excluded_events":[],'
        '"uncertainties":[],"abstain_reason":null}',
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-status-json",
        sample_id="conv-26",
        question="What is Caroline's relationship status?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D2:14] Caroline is a single parent."},
    )

    prompt, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert prompt.startswith(LOCOMO_ANSWER_FREEFORM_PROMPT)
    assert response.text == "Single"
    assert provider.generate_structured_calls == 0
    assert provider.generate_calls == 1
    assert response.metadata["answer_synthesis_mode"] == "freeform_v2"
    assert response.metadata["answer_freeform_parse_format"] == "json_answer"


def test_answer_generator_normalizes_incomplete_freeform_json_answer_response():
    provider = _StructuredAnswerProvider(
        supports_structured=False,
        text_response=json.dumps(
            {
                "can_answer": True,
                "final_answer": "Single",
                "supporting_source_refs": ["D2:14"],
                "counted_events": ["Caroline is single."],
                "excluded_events": [],
                "abstain_reason": "",
            }
        ),
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-status-json-incomplete",
        sample_id="conv-26",
        question="What is Caroline's relationship status?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D2:14] Caroline is a single parent."},
    )

    prompt, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert prompt.startswith(LOCOMO_ANSWER_FREEFORM_PROMPT)
    assert response.text == "Single"
    assert response.metadata["answer_synthesis_mode"] == "freeform_v2"
    assert response.metadata["answer_freeform_parse_format"] == "json_answer"
    assert response.metadata["answer_synthesis_answer_type"] in {"value", "status"}


def test_answer_synthesis_abstention_uses_reflection_trigger_phrase():
    payload = _answer_synthesis_payload("", answer_type="unknown")
    payload["can_answer"] = False
    payload["supporting_facts"] = []
    payload["supporting_source_refs"] = []
    payload["abstain_reason"] = "no relationship status evidence is present"
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-abstain",
        sample_id="conv-26",
        question="What is Caroline's relationship status?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": ""},
    )

    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text.startswith("The retrieved context does not support an answer")
    assert PipelineRunner._answer_is_context_abstention(response.text)
    assert response.metadata["answer_postcheck_skipped"] is True
    assert response.metadata["answer_postcheck_skip_reason"] == "synthesis_can_answer_false"
    assert response.metadata["answer_repair_used"] is False
    assert provider.generate_calls == 1


def test_answer_generator_synthesis_counts_distinct_completed_events():
    payload = _answer_synthesis_payload("twice", answer_type="count")
    payload["supporting_facts"] = [
        {"fact_text": "Joanna found a new hiking trail once.", "source_refs": ["D8:4"]},
        {"fact_text": "Joanna found more new hiking trails later.", "source_refs": ["D11:3"]},
    ]
    payload["supporting_source_refs"] = ["D8:4", "D11:3"]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "first new hiking trail", "source_refs": ["D8:4"], "reason": "completed distinct event"},
        {"event_id": "E2", "event_text": "more new hiking trails", "source_refs": ["D11:3"], "reason": "completed distinct event"},
    ]
    payload["excluded_events"] = [
        {"event_id": "X1", "event_text": "future trail plan", "source_refs": ["D14:19"], "reason": "future plan"},
        {"event_id": "X2", "event_text": "same hike spot detail", "source_refs": ["D11:5"], "reason": "duplicate detail"},
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_19",
        sample_id="conv-42",
        question="How many times has Joanna found new hiking trails?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )

    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text == "twice"
    assert response.metadata["answer_synthesis_answer_type"] == "count"
    assert response.metadata["answer_count_naturalized"] is False
    assert len(response.metadata["answer_synthesis_counted_events"]) == 2
    assert len(response.metadata["answer_synthesis_excluded_events"]) == 2


def test_answer_generator_naturalizes_bare_numeric_count_answer():
    payload = _answer_synthesis_payload("2", answer_type="count")
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "first rejection", "source_refs": ["D1:1"], "reason": "completed"},
        {"event_id": "E2", "event_text": "second rejection", "source_refs": ["D2:1"], "reason": "completed"},
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_49",
        sample_id="conv-42",
        question="How many times has Joanna's scripts been rejected?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] ..."},
    )

    _, response = generator.generate(query_task, _locomo_count_retrieval_bundle())

    assert response.text == "Twice."
    assert response.metadata["answer_count_naturalized"] is True
    assert response.metadata["answer_count_naturalized_from"] == "2"
    assert response.metadata["answer_count_naturalized_lower_bound"] is False


def test_answer_generator_naturalizes_evidence_limited_count_lower_bound():
    payload = _answer_synthesis_payload("1", answer_type="count")
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "one rejection", "source_refs": ["D1:1"], "reason": "completed"}
    ]
    payload["uncertainties"] = ["The retrieved evidence may be missing later rejection events."]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_49",
        sample_id="conv-42",
        question="How many times has Joanna's scripts been rejected?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] ..."},
    )

    _, response = generator.generate(query_task, _locomo_count_retrieval_bundle())

    assert response.text == "The retrieved evidence confirms one rejection."
    assert response.metadata["answer_count_naturalized"] is True
    assert response.metadata["answer_count_naturalized_from"] == "1"
    assert response.metadata["answer_count_naturalized_lower_bound"] is True
    assert response.metadata["answer_count_lower_bound_reasons"] == ["uncertainties"]


def test_answer_generator_marks_invalid_count_candidate_as_lower_bound():
    payload = _answer_synthesis_payload("2", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4", "D99:1"]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "Joanna found a new hiking trail.", "source_refs": ["D8:4"], "reason": "completed"},
        {"event_id": "E2", "event_text": "Joanna found another new hiking trail.", "source_refs": ["D99:1"], "reason": "completed"},
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_19",
        sample_id="conv-42",
        question="How many times has Joanna found new hiking trails?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )

    _, response = generator.generate(query_task, _locomo_count_retrieval_bundle_with_refs())

    assert response.text == "The retrieved evidence confirms one time."
    assert response.metadata["answer_count_naturalized_lower_bound"] is True
    assert "invalid_supporting_refs" in response.metadata["answer_count_lower_bound_reasons"]
    assert "count_validation_excluded_plausible_candidates" in response.metadata["answer_count_lower_bound_reasons"]
    assert response.metadata["answer_count_lower_bound_excluded_event_count"] == 1


def test_answer_generator_marks_uncertain_count_candidate_as_lower_bound():
    payload = _answer_synthesis_payload("2", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4", "D11:3"]
    payload["supporting_facts"] = [
        {"fact_text": "Joanna found a new hiking trail.", "source_refs": ["D8:4"]},
        {"fact_text": "Joanna described an uncertain second trail candidate.", "source_refs": ["D11:3"]},
    ]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "Joanna found a new hiking trail.", "source_refs": ["D8:4"], "reason": "completed"},
        {
            "event_id": "E2",
            "event_text": "Joanna might have found another new hiking trail.",
            "source_refs": ["D11:3"],
            "reason": "assumption",
        },
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_19",
        sample_id="conv-42",
        question="How many times has Joanna found new hiking trails?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )

    _, response = generator.generate(query_task, _locomo_count_retrieval_bundle_with_refs())

    assert response.text == "The retrieved evidence confirms one time."
    assert response.metadata["answer_count_naturalized_lower_bound"] is True
    assert "count_validation_excluded_plausible_candidates" in response.metadata["answer_count_lower_bound_reasons"]


def test_answer_generator_excludes_reaction_and_assumption_count_events():
    payload = _answer_synthesis_payload("3", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4", "D11:3", "D11:8"]
    payload["supporting_facts"] = [
        {"fact_text": "Joanna found a new hiking trail.", "source_refs": ["D8:4"]},
        {"fact_text": "Joanna found another new hiking trail.", "source_refs": ["D11:3"]},
        {"fact_text": "Nate reacted to Joanna's hikes.", "source_refs": ["D11:8"]},
    ]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "Joanna found a new hiking trail.", "source_refs": ["D8:4"], "reason": "completed"},
        {"event_id": "E2", "event_text": "Joanna found another new hiking trail.", "source_refs": ["D11:3"], "reason": "completed"},
        {"event_id": "E3", "event_text": "Your hikes sound like a blast.", "source_refs": ["D11:8"], "reason": "sounds like a blast"},
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_19",
        sample_id="conv-42",
        question="How many times has Joanna found new hiking trails?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )

    _, response = generator.generate(query_task, _locomo_count_retrieval_bundle_with_refs())

    assert response.text == "Twice."
    assert response.metadata["answer_count_naturalized_lower_bound"] is False
    assert len(response.metadata["answer_synthesis_counted_events"]) == 2
    assert len(response.metadata["count_validation_excluded_events"]) == 1


def test_answer_generator_keeps_completed_event_with_positive_adjective():
    payload = _answer_synthesis_payload("1", answer_type="exact count")
    payload["supporting_source_refs"] = ["D8:4"]
    payload["supporting_facts"] = [
        {"fact_text": "Joanna found an awesome hiking trail.", "source_refs": ["D8:4"]}
    ]
    payload["counted_events"] = [
        {
            "event_id": "E1",
            "event_text": "Joanna found an awesome hiking trail.",
            "source_refs": ["D8:4"],
            "reason": "completed distinct event",
        }
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_19",
        sample_id="conv-42",
        question="How many times has Joanna found new hiking trails?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )

    _, response = generator.generate(query_task, _locomo_count_retrieval_bundle_with_refs())

    assert response.text == "Once."
    assert len(response.metadata["answer_synthesis_counted_events"]) == 1
    assert response.metadata["count_validation_positive_event_signal"]
    assert response.metadata["count_validation_rejection_signal"] == []


def test_answer_generator_marks_invalid_supporting_source_refs():
    payload = _answer_synthesis_payload("Five times.", answer_type="count")
    payload["supporting_source_refs"] = ["D14:12"]
    payload["supporting_facts"] = [{"fact_text": "miscopied rejection ref", "source_refs": ["D14:12"]}]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "miscopied rejection ref", "source_refs": ["D14:12"], "reason": "completed"},
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_49",
        sample_id="conv-42",
        question="How many times has Joanna's scripts been rejected?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D14:1] ..."},
    )

    _, response = generator.generate(query_task, _locomo_count_retrieval_bundle_with_refs())

    assert response.text.startswith("The retrieved context does not support")
    assert response.metadata["invalid_supporting_refs"] == ["D14:12"]
    assert response.metadata["answer_synthesis_can_answer"] is False


def test_answer_generator_rejects_count_answer_for_time_question():
    payload = _answer_synthesis_payload("Once.", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4"]
    payload["supporting_facts"] = [{"fact_text": "Melanie went to the beach once.", "source_refs": ["D8:4"]}]
    retry_payload = _answer_synthesis_payload("19 October 2023", answer_type="date")
    retry_payload["supporting_source_refs"] = ["D8:4"]
    retry_payload["supporting_facts"] = [{"fact_text": "Melanie went hiking yesterday.", "source_refs": ["D8:4"]}]
    provider = _StructuredAnswerProvider(structured_payloads=[payload, retry_payload])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa-time",
        sample_id="conv-26",
        question="When did Melanie go to the beach?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-time",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:4"],
        prompt_context="D8:4 | Melanie: I went to the beach last Tuesday.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "19 October 2023"
    assert provider.generate_structured_calls == 2
    assert "TYPE_CONSTRAINT_RETRY" in provider.prompts[2]
    assert response.metadata["answer_synthesis_question_type_mismatch"] is True
    assert response.metadata["answer_synthesis_typed_retry_used"] is True
    assert response.metadata["answer_synthesis_typed_retry_success"] is True
    assert response.metadata["answer_synthesis_typed_retry_expected_type"] == "date"
    assert response.metadata["answer_synthesis_initial_answer_type"] == "count"


def test_answer_generator_rejects_count_style_sentence_for_date_recovery_before_retry():
    payload = _answer_synthesis_payload(
        "The retrieved evidence confirms one hike after the road trip.",
        answer_type="count",
    )
    payload["supporting_source_refs"] = ["D18:17"]
    payload["supporting_facts"] = [
        {"fact_text": "Melanie went hiking yesterday after the road trip.", "source_refs": ["D18:17"]}
    ]
    payload["counted_events"] = [
        {
            "event_id": "E1",
            "event_text": "Melanie went hiking after the road trip.",
            "source_refs": ["D18:17"],
            "reason": "completed hike",
        }
    ]
    retry_payload = _answer_synthesis_payload("19 October 2023", answer_type="date")
    retry_payload["supporting_source_refs"] = ["D18:17"]
    retry_payload["supporting_facts"] = [
        {"fact_text": "D18:17 occurred on 20 October 2023; yesterday was 19 October 2023.", "source_refs": ["D18:17"]}
    ]
    provider = _StructuredAnswerProvider(structured_payloads=[payload, retry_payload])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_76",
        sample_id="conv-26",
        question="When did Melanie go on a hike after the roadtrip?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D18:17] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-hike-date",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D18:17"],
        prompt_context=(
            'D18:17 | date=6:55 pm on 20 October, 2023 | Melanie: The hike after the road trip happened yesterday!\n'
            '## Temporal Anchors\n- D18:17 occurred at 20 October 2023; "yesterday" refers to 19 October 2023.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "19 October 2023"
    assert provider.generate_structured_calls == 2
    assert response.metadata["answer_synthesis_type_mismatch_recovered"] is False
    assert response.metadata["answer_synthesis_type_recovery_rejected_reason"] == "date_question_count_style_answer"
    assert response.metadata["answer_synthesis_typed_retry_used"] is True
    assert response.metadata["answer_synthesis_typed_retry_success"] is True


def test_answer_generator_temporal_alignment_replaces_unaligned_date_with_anchor_date():
    payload = _answer_synthesis_payload("27 June 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D4:8", "D18:17"]
    payload["supporting_facts"] = [
        {"fact_text": "Melanie went hiking on 27 June 2023.", "source_refs": ["D4:8"]},
        {"fact_text": "D18:17 says the hike after the road trip happened yesterday.", "source_refs": ["D18:17"]},
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_76",
        sample_id="conv-26",
        question="When did Melanie go on a hike after the roadtrip?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D18:17] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-hike-date",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D4:8", "D18:17"],
        prompt_context=(
            "D4:8 | date=10:37 am on 27 June, 2023 | Melanie: We went hiking during the camping trip.\n"
            "D18:17 | date=6:55 pm on 20 October, 2023 | Melanie: The hike after the road trip happened yesterday!\n"
            '## Temporal Anchors\n- D18:17 occurred at 20 October 2023; "yesterday" refers to 19 October 2023.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "19 October 2023"
    assert response.metadata["answer_temporal_alignment_checked"] is True
    assert response.metadata["answer_temporal_alignment_valid"] is True
    assert response.metadata["answer_temporal_repair_used"] is True
    assert response.metadata["answer_temporal_selected_source_ref"] == "D18:17"


def test_answer_generator_temporal_alignment_ranks_query_relevant_museum_source_first():
    payload = _answer_synthesis_payload("24 August 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D6:4", "D14:4"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_20",
        sample_id="conv-26",
        question="When did Melanie go to the museum?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D6:4] museum"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-museum-date",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D6:4", "D14:4"],
        prompt_context=(
            "[D6:4] Melanie: Yesterday I took the kids to the museum.\n"
            "[D14:4] Melanie: Yesterday I signed up for a pottery class.\n"
            '## Temporal Anchors\n- D6:4 occurred at 6 July 2023; "yesterday" refers to 5 July 2023.\n'
            '- D14:4 occurred at 25 August 2023; "yesterday" refers to 24 August 2023.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "5 July 2023"
    assert response.metadata["answer_temporal_selected_source_ref"] == "D6:4"
    assert response.metadata["answer_temporal_selected_confidence"] == "high"
    assert "museum" in response.metadata["answer_temporal_candidate_match_terms"]


def test_answer_generator_temporal_alignment_does_not_select_bare_relative_source():
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-school-speech-date",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D3:1", "D19:2"],
        prompt_context=(
            "[D3:1] Caroline: Yesterday I gave my school speech and encouraged students.\n"
            "[D19:2] Caroline: Yesterday was emotional.\n"
            '## Temporal Anchors\n- D3:1 occurred at 9 June 2023; "yesterday" refers to 8 June 2023.\n'
            '- D19:2 occurred at 22 October 2023; "yesterday" refers to 21 October 2023.'
        ),
        latency_ms=1.0,
    )

    diagnostics = AnswerGenerator._temporal_answer_alignment_diagnostics(
        answer_text="21 October 2023",
        retrieval_bundle=bundle,
        question="When did Caroline give a speech at a school?",
    )

    assert diagnostics["answer_temporal_selected_source_ref"] == "D3:1"
    assert diagnostics["answer_temporal_selected_date"] == "8 June 2023"
    low_rows = [
        row for row in diagnostics["answer_temporal_candidate_dates"]
        if row["source_ref"] == "D19:2"
    ]
    assert low_rows
    assert low_rows[0]["confidence"] == "low"


def test_answer_generator_repairs_last_week_school_speech_to_relative_span():
    payload = _answer_synthesis_payload("9 June 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D3:1"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_8",
        sample_id="conv-26",
        question="When did Caroline give a speech at a school?",
        metadata={"category_name": "temporal", "evidence_only_conversation": "[D3:1] school event"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-school-last-week",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D3:1"],
        prompt_context=(
            "[D3:1] Caroline: I went to a school event last week and talked about my transgender journey.\n"
            '## Temporal Anchors\n- D3:1 occurred at 9 June 2023; "last week" refers to the week before 9 June 2023.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "the week before 9 June 2023"
    assert response.metadata["answer_temporal_alignment_valid"] is True
    assert response.metadata["answer_temporal_repair_used"] is True
    assert response.metadata["answer_temporal_selected_source_ref"] == "D3:1"
    assert response.metadata["answer_temporal_selected_date"] is None
    assert response.metadata["answer_temporal_selected_answer_text"] == "the week before 9 June 2023"
    assert response.metadata["answer_temporal_selected_resolution_kind"] == "relative_span"
    assert response.metadata["answer_temporal_selected_resolution_granularity"] == "week_span"
    assert response.metadata["answer_temporal_selected_relative_term"] == "last week"


def test_answer_generator_repairs_last_week_picnic_to_relative_span():
    payload = _answer_synthesis_payload("6 July 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D6:11"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_21",
        sample_id="conv-26",
        question="When did Caroline have a picnic?",
        metadata={"category_name": "temporal", "evidence_only_conversation": "[D6:11] picnic"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-picnic-last-week",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D6:11"],
        prompt_context=(
            "[D6:11] Caroline: We even had a picnic last week with friends and family.\n"
            '## Temporal Anchors\n- D6:11 occurred at 6 July 2023; "last week" refers to approximately the week before 6 July 2023.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "the week before 6 July 2023"
    assert response.metadata["answer_temporal_alignment_valid"] is True
    assert response.metadata["answer_temporal_repair_used"] is True
    assert response.metadata["answer_temporal_selected_answer_text"] == "the week before 6 July 2023"
    assert response.metadata["answer_temporal_selected_resolution_kind"] == "relative_span"
    assert response.metadata["answer_temporal_selected_resolution_granularity"] == "week_span"


def test_answer_generator_repairs_one_year_ago_to_month_year_target():
    payload = _answer_synthesis_payload("8 October 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D12:2"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-49_qa_58",
        sample_id="conv-49",
        question="When did Evan start lifting weights?",
        metadata={"category_name": "temporal", "evidence_only_conversation": "[D12:2] lifting weights"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-lifting-one-year-ago",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D12:2"],
        prompt_context=(
            "[D12:2] Evan: I started lifting weights one year ago and it has been a journey.\n"
            '## Temporal Anchors\n- D12:2 occurred at 8 October 2023; "one year ago" refers to October 2022.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "October 2022"
    assert response.metadata["answer_temporal_alignment_valid"] is True
    assert response.metadata["answer_temporal_repair_used"] is True
    assert response.metadata["answer_temporal_selected_source_ref"] == "D12:2"
    assert response.metadata["answer_temporal_selected_date"] is None
    assert response.metadata["answer_temporal_selected_answer_text"] == "October 2022"
    assert response.metadata["answer_temporal_selected_resolution_kind"] == "month_year"
    assert response.metadata["answer_temporal_selected_resolution_granularity"] == "month"
    assert response.metadata["answer_temporal_selected_relative_term"] == "one year ago"


def test_answer_generator_repairs_last_friday_to_weekday_span_target():
    payload = _answer_synthesis_payload("15 July 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D8:9"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_28",
        sample_id="conv-26",
        question="When did Caroline go to the adoption meeting?",
        metadata={"category_name": "temporal", "evidence_only_conversation": "[D8:9] adoption meeting"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-adoption-last-friday",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:9"],
        prompt_context=(
            "[D8:9] Caroline: Last Friday I went to a council meeting for adoption.\n"
            '## Temporal Anchors\n- D8:9 occurred at 15 July 2023; "last friday" refers to the Friday before 15 July 2023.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "the Friday before 15 July 2023"
    assert response.metadata["answer_temporal_alignment_valid"] is True
    assert response.metadata["answer_temporal_repair_used"] is True
    assert response.metadata["answer_temporal_selected_answer_text"] == "the Friday before 15 July 2023"
    assert response.metadata["answer_temporal_selected_resolution_kind"] == "relative_span"
    assert response.metadata["answer_temporal_selected_resolution_granularity"] == "weekday_span"


def test_answer_generator_repairs_fuzzy_relative_without_source_date_collapse():
    payload = _answer_synthesis_payload("8 February 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D5:15"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-30_qa_11",
        sample_id="conv-30",
        question="When did Gina get her tattoo?",
        metadata={"category_name": "temporal", "evidence_only_conversation": "[D5:15] tattoo"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-tattoo-fuzzy",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D5:15"],
        prompt_context=(
            "[D5:15] Gina: Got the tattoo a few years ago, it stands for freedom.\n"
            '## Temporal Anchors\n- D5:15 occurred at 8 February 2023; "a few years ago" refers to a few years ago.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "a few years ago"
    assert response.metadata["answer_temporal_alignment_valid"] is True
    assert response.metadata["answer_temporal_repair_used"] is True
    assert response.metadata["answer_temporal_selected_answer_text"] == "a few years ago"
    assert response.metadata["answer_temporal_selected_resolution_kind"] == "fuzzy_relative"
    assert response.metadata["answer_temporal_selected_resolution_granularity"] == "fuzzy"


def test_answer_generator_temporal_alignment_safe_abstains_when_only_unrelated_dates_exist():
    payload = _answer_synthesis_payload("21 October 2023", answer_type="date")
    payload["supporting_source_refs"] = ["D19:2"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_school",
        sample_id="conv-26",
        question="When did Caroline give a speech at a school?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D19:2] yesterday"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-unrelated-date",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D19:2"],
        prompt_context=(
            "[D19:2] Caroline: Yesterday was emotional.\n"
            '## Temporal Anchors\n- D19:2 occurred at 22 October 2023; "yesterday" refers to 21 October 2023.'
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "The retrieved context does not support an answer to this question."
    assert response.metadata["answer_temporal_no_query_relevant_candidate"] is True
    assert response.metadata["answer_temporal_repair_action"] == "safe_abstain_no_aligned_candidate"


def test_answer_generator_skips_list_coverage_repair_for_date_questions():
    provider = _AnswerRepairProvider(["Melanie signed up for a pottery class."])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_pottery_date",
        sample_id="conv-26",
        question="When did Melanie sign up for a pottery class?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D5:4] pottery"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-date-list-skip",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D5:4"],
        prompt_context="[D5:4] Melanie: I signed up for a pottery class.",
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "activity",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": ["pottery class"],
            "grounded_exact_terms": ["pottery class"],
            "grounded_display_items": ["pottery class"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": ["Melanie signed up for a pottery class."],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.metadata["answer_list_coverage_skipped"] is True
    assert response.metadata["answer_list_coverage_skip_reason"] == "date_time_question"
    assert response.metadata["answer_list_repair_blocked_by_expected_type"] is True
    assert response.metadata["answer_list_coverage_repair_used"] is False
    assert response.metadata["answer_postcheck_issue"] is None


def test_answer_generator_rejects_count_answer_for_non_count_question():
    payload = _answer_synthesis_payload("Two times.", answer_type="event_count")
    payload["supporting_source_refs"] = ["D8:4"]
    retry_payload = _answer_synthesis_payload("Sweden", answer_type="place")
    retry_payload["supporting_source_refs"] = ["D8:4"]
    retry_payload["supporting_facts"] = [{"fact_text": "Caroline moved from Sweden.", "source_refs": ["D8:4"]}]
    provider = _StructuredAnswerProvider(structured_payloads=[payload, retry_payload])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_where",
        sample_id="conv-26",
        question="Where did Caroline move from four years ago?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-where",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:4"],
        prompt_context="[D8:4] Caroline: I moved from Sweden four years ago.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Sweden"
    assert provider.generate_structured_calls == 2
    assert response.metadata["answer_synthesis_question_type_mismatch"] is True
    assert response.metadata["answer_synthesis_typed_retry_used"] is True
    assert response.metadata["answer_synthesis_typed_retry_success"] is True


def test_answer_generator_rejects_count_style_text_for_non_count_question():
    payload = _answer_synthesis_payload("Two items.", answer_type="list")
    payload["supporting_source_refs"] = ["D8:4"]
    retry_payload = _answer_synthesis_payload("a book and a lamp", answer_type="list")
    retry_payload["supporting_source_refs"] = ["D8:4"]
    retry_payload["supporting_facts"] = [{"fact_text": "Melanie bought a book and a lamp.", "source_refs": ["D8:4"]}]
    provider = _StructuredAnswerProvider(structured_payloads=[payload, retry_payload])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_items",
        sample_id="conv-26",
        question="What items did Melanie buy?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-items",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:4"],
        prompt_context="[D8:4] Melanie: I bought a book and a lamp.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "a book and a lamp"
    assert provider.generate_structured_calls == 2
    assert response.metadata["answer_synthesis_question_type_mismatch"] is True
    assert response.metadata["answer_synthesis_count_style_text_mismatch"] is True
    assert response.metadata["answer_synthesis_typed_retry_used"] is True
    assert response.metadata["answer_synthesis_typed_retry_success"] is True


def test_answer_generator_recovers_mislabeled_count_type_when_final_answer_is_non_count_value():
    payload = _answer_synthesis_payload("Jolene has not tried surfing yet.", answer_type="count")
    payload["supporting_source_refs"] = ["D25:4"]
    payload["supporting_facts"] = [
        {"fact_text": "Jolene has not tried surfing yet.", "source_refs": ["D25:4"]}
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-48_qa_80",
        sample_id="conv-48",
        question="Has Jolene tried surfing?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D25:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-surfing",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D25:4"],
        prompt_context="D25:4 | Jolene: I have not tried surfing yet.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Jolene has not tried surfing yet."
    assert provider.generate_structured_calls == 1
    assert provider.generate_calls == 1
    assert response.metadata["answer_synthesis_question_type_mismatch"] is True
    assert response.metadata["answer_synthesis_type_mismatch_recovered"] is True
    assert response.metadata["answer_synthesis_recovered_from_answer_type"] == "count"
    assert response.metadata["answer_synthesis_recovered_answer_type"] == "boolean"
    assert response.metadata["answer_synthesis_typed_retry_used"] is False


def test_answer_generator_recovers_mislabeled_count_type_for_year_answer_to_time_question():
    payload = _answer_synthesis_payload("2023", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4"]
    payload["supporting_facts"] = [{"fact_text": "The event happened in 2023.", "source_refs": ["D8:4"]}]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-year",
        sample_id="conv-1",
        question="What year did Melanie go on the hike?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-year",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:4"],
        prompt_context="D8:4 | date=2023 | Melanie: I went on the hike that year.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "2023"
    assert provider.generate_structured_calls == 1
    assert provider.generate_calls == 1
    assert response.metadata["answer_synthesis_type_mismatch_recovered"] is True
    assert response.metadata["answer_synthesis_recovered_answer_type"] == "date"


def test_answer_generator_abstains_when_type_constrained_retry_still_mismatches():
    payload = _answer_synthesis_payload("Once.", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4"]
    retry_payload = _answer_synthesis_payload("Two times.", answer_type="count")
    retry_payload["supporting_source_refs"] = ["D8:4"]
    provider = _StructuredAnswerProvider(structured_payloads=[payload, retry_payload])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa-time",
        sample_id="conv-26",
        question="When did Melanie go to the beach?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-time",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:4"],
        prompt_context="D8:4 | Melanie: I went to the beach last Tuesday.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support")
    assert "synthesized a count answer" not in response.text
    assert provider.generate_structured_calls == 2
    assert response.metadata["answer_synthesis_typed_retry_used"] is True
    assert response.metadata["answer_synthesis_typed_retry_success"] is False
    assert response.metadata["answer_synthesis_repair_reason"] == "question_type_mismatch_time_question_count_answer"
    assert response.metadata["answer_synthesis_safe_abstain_used"] is True
    assert response.metadata["answer_synthesis_internal_abstain_reason_suppressed"] is True


def test_answer_generator_abstains_when_date_retry_text_is_count_style_sentence():
    payload = _answer_synthesis_payload("Once.", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4"]
    retry_payload = _answer_synthesis_payload("The retrieved evidence confirms one hike.", answer_type="date")
    retry_payload["supporting_source_refs"] = ["D8:4"]
    provider = _StructuredAnswerProvider(structured_payloads=[payload, retry_payload])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa-time",
        sample_id="conv-26",
        question="When did Melanie go to the beach?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-time",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:4"],
        prompt_context="D8:4 | Melanie: I went to the beach last Tuesday.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support an answer to this question")
    assert response.metadata["answer_synthesis_typed_retry_used"] is True
    assert response.metadata["answer_synthesis_typed_retry_success"] is False
    assert response.metadata["answer_synthesis_expected_type_text_valid"] is False
    assert response.metadata["answer_synthesis_expected_type_text_rejection_reason"] == "date_question_count_style_answer"


def test_answer_generator_tolerates_incomplete_text_json_for_failed_typed_retry():
    payload = _answer_synthesis_payload("Once.", answer_type="count")
    payload["supporting_source_refs"] = ["D8:4"]
    retry_payload = _answer_synthesis_payload("Two times.", answer_type="count")
    retry_payload["supporting_source_refs"] = ["D8:4"]
    provider = _StructuredAnswerProvider(
        structured_payloads=[payload, retry_payload],
        text_responses=["", '{"can_answer": false, "abstain_reason": "not enough date evidence"}'],
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_51",
        sample_id="conv-42",
        question="When did Nate get Tilly for Joanna?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-date",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:4"],
        prompt_context="D8:4 | Nate: I got Tilly today.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support an answer to this question")
    assert provider.generate_structured_calls == 2
    assert provider.generate_calls == 2
    assert response.metadata["answer_synthesis_typed_retry_used"] is True
    assert response.metadata["answer_synthesis_typed_retry_success"] is False
    assert response.metadata["answer_synthesis_typed_retry_text_json_normalized"] is True
    assert "answer_type" in response.metadata["answer_synthesis_typed_retry_text_json_missing_fields"]
    assert response.metadata["answer_synthesis_typed_retry_final_policy"] == "standard_abstain"
    assert response.metadata["answer_synthesis_safe_abstain_used"] is True
    assert "synthesized a count answer" not in response.text


def test_answer_generator_rejects_family_mismatched_supporting_ref():
    payload = _answer_synthesis_payload("Two children", answer_type="count")
    payload["supporting_source_refs"] = ["D8:9"]
    payload["supporting_facts"] = [{"fact_text": "Caroline mentioned two children.", "source_refs": ["D8:9"]}]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_75",
        sample_id="conv-26",
        question="How many pets did Caroline want to adopt?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:9] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-pets",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:9"],
        prompt_context="D8:9 | Caroline: I want to adopt two children someday.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support")
    assert response.metadata["answer_synthesis_invalid_family_refs"] == ["D8:9"]
    assert response.metadata["answer_synthesis_can_answer"] is False


def test_answer_generator_rejects_family_mismatched_count_event_ref():
    payload = _answer_synthesis_payload("Two times.", answer_type="count")
    payload["supporting_source_refs"] = []
    payload["supporting_facts"] = []
    payload["counted_events"] = [
        {
            "event_id": "E1",
            "event_text": "Caroline mentioned two children.",
            "source_refs": ["D8:9"],
            "reason": "completed",
        }
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_75",
        sample_id="conv-26",
        question="How many pets did Caroline want to adopt?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:9] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-pets-count",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:9"],
        prompt_context="D8:9 | Caroline: I want to adopt two children someday.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support")
    assert response.metadata["answer_synthesis_invalid_family_refs"] == ["D8:9"]
    assert response.metadata["count_validation_excluded_events"][0]["source_refs"] == []
    assert response.metadata["answer_synthesis_can_answer"] is False


def test_answer_generator_accepts_turtle_walk_image_caption_aliases():
    payload = _answer_synthesis_payload("2", answer_type="count")
    payload["supporting_source_refs"] = ["D5:4", "D25:15"]
    payload["supporting_facts"] = [
        {"fact_text": "Nate walked the turtles.", "source_refs": ["D5:4"]},
        {"fact_text": "Nate took his turtles out for a walk.", "source_refs": ["D25:15"]},
    ]
    payload["counted_events"] = [
        {
            "event_id": "E1",
            "event_text": "Nate was walking them with two tortoises shown in the image.",
            "source_refs": ["D5:4"],
            "reason": "completed walk event",
        },
        {
            "event_id": "E2",
            "event_text": "Nate took his turtles out for a walk.",
            "source_refs": ["D25:15"],
            "reason": "completed walk event",
        },
    ]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_53",
        sample_id="conv-42",
        question="How many times has Nate taken his turtles on a walk?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D5:4] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-turtles",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D5:4", "D25:15"],
        prompt_context=(
            "[D5:4] Nate: walking them always reminds me of home. "
            "[shared image: a photography of two tortoises laying on the ground in a jungle]\n"
            "[D25:15] Nate: I was bored today, so I just took my turtles out for a walk."
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Twice."
    assert response.metadata["answer_synthesis_can_answer"] is True
    assert response.metadata["answer_synthesis_invalid_family_refs"] == []
    assert len(response.metadata["answer_synthesis_counted_events"]) == 2
    assert response.metadata["source_family_validation_alias_hits"]
    assert response.metadata["count_validation_ref_acceptance_reasons"]


def test_answer_generator_adds_source_derived_count_candidate_before_llm_validation():
    payload = _answer_synthesis_payload("1", answer_type="count")
    payload["supporting_source_refs"] = ["D25:15"]
    payload["supporting_facts"] = [
        {"fact_text": "Nate took his turtles out for a walk.", "source_refs": ["D25:15"]},
    ]
    payload["counted_events"] = [
        {
            "event_id": "E1",
            "event_text": "Nate took his turtles out for a walk.",
            "source_refs": ["D25:15"],
            "reason": "completed walk event",
        }
    ]
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-turtles",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D5:4", "D25:15"],
        prompt_context=(
            "[D5:4] Nate: walking them always reminds me to enjoy the small stuff! "
            "[shared image: a photography of two tortoises laying on the ground in a jungle]\n"
            "[D25:15] Nate: I was bored today, so I just took my turtles out for a walk."
        ),
        latency_ms=1.0,
    )
    derived = AnswerGenerator._source_derived_count_candidates(
        question="How many times has Nate taken his turtles on a walk?",
        retrieval_bundle=bundle,
        existing_refs={"D25:15"},
    )
    assert [candidate["valid_source_refs"] for candidate in derived] == [["D5:4"]]
    source_derived_id = derived[0]["event_id"]
    validator_payload = {
        "count_scope": "completed_events",
        "validated_events": [
            {"event_id": "E1", "decision": "COUNT", "source_refs": ["D25:15"], "reason": "Explicit turtle walk."},
            {"event_id": source_derived_id, "decision": "COUNT", "source_refs": ["D5:4"], "reason": "Walking them refers to the tortoises in the image."},
        ],
        "final_count": 2,
        "confidence": "high",
        "validator_notes": "Both source-backed walk events count.",
    }
    provider = _StructuredAnswerProvider(structured_payload=payload, count_validation_payload=validator_payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_53",
        sample_id="conv-42",
        question="How many times has Nate taken his turtles on a walk?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D5:4] ..."},
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Twice."
    assert response.metadata["count_validation_llm_used"] is True
    assert response.metadata["count_validation_llm_success"] is True
    assert response.metadata["count_validation_source_derived_candidate_count"] == 1
    assert response.metadata["count_validation_source_derived_candidate_refs"] == ["D5:4"]
    assert response.metadata["count_validation_llm_changed_count"] == 1
    assert {event["source_refs"][0] for event in response.metadata["answer_synthesis_counted_events"]} == {"D5:4", "D25:15"}


def test_answer_generator_rejects_passive_turtle_observation_as_source_derived_count_candidate():
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-turtles-passive",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D24:9"],
        prompt_context=(
            "[D24:9] Nate: I'm not sure I'll ever understand why watching my turtles slowly walk around "
            "makes me so happy."
        ),
        latency_ms=1.0,
    )

    scan = AnswerGenerator._source_derived_count_candidate_scan(
        question="How many times has Nate taken his turtles on a walk?",
        retrieval_bundle=bundle,
        existing_refs=set(),
    )

    assert scan["candidates"] == []
    assert scan["passive_rejected_refs"] == ["D24:9"]


def test_answer_generator_llm_count_validator_counts_planned_events_when_question_asks_plans():
    payload = _answer_synthesis_payload("3", answer_type="count")
    payload["supporting_source_refs"] = ["D11:6", "D12:4", "D23:26"]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "Audrey should join Andrew for a hike.", "source_refs": ["D11:6"], "reason": "planned hike"},
        {"event_id": "E2", "event_text": "They should plan a trip for both of them.", "source_refs": ["D12:4"], "reason": "planned hike"},
        {"event_id": "E3", "event_text": "Audrey needs to go on a hike with them.", "source_refs": ["D23:26"], "reason": "planned hike"},
    ]
    validator_payload = {
        "count_scope": "planned_events",
        "validated_events": [
            {"event_id": "E1", "decision": "COUNT", "source_refs": ["D11:6"], "reason": "Question asks planned hikes."},
            {"event_id": "E2", "decision": "COUNT", "source_refs": ["D12:4"], "reason": "Question asks planned hikes."},
            {"event_id": "E3", "decision": "COUNT", "source_refs": ["D23:26"], "reason": "Question asks planned hikes."},
        ],
        "final_count": 3,
        "confidence": "high",
        "validator_notes": "All three are planned hike mentions.",
    }
    provider = _StructuredAnswerProvider(structured_payload=payload, count_validation_payload=validator_payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-44_qa_17",
        sample_id="conv-44",
        question="How many times did Audrey and Andrew plan to hike together?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D11:6] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-planned-hikes",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D11:6", "D12:4", "D23:26"],
        prompt_context=(
            "[D11:6] Audrey: I should join you for a hike and bring my dogs.\n"
            "[D12:4] Audrey: We should plan a trip for both of us and our pups!\n"
            "[D23:26] Audrey: I need to go on a hike with them."
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Three times."
    assert response.metadata["count_validation_llm_used"] is True
    assert response.metadata["count_validation_llm_success"] is True
    assert response.metadata["count_validation_llm_scope"] == "planned_events"
    assert response.metadata["count_validation_llm_changed_count"] == 3


def test_answer_generator_llm_count_validator_can_keep_future_plan_excluded_for_completed_question():
    payload = _answer_synthesis_payload("1", answer_type="count")
    payload["supporting_source_refs"] = ["D11:6"]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "Audrey should join Andrew for a hike.", "source_refs": ["D11:6"], "reason": "planned hike"}
    ]
    validator_payload = {
        "count_scope": "completed_events",
        "validated_events": [
            {"event_id": "E1", "decision": "EXCLUDE", "source_refs": ["D11:6"], "reason": "Only a future plan, not completed."}
        ],
        "final_count": 0,
        "confidence": "high",
        "validator_notes": "No completed hikes are supported.",
    }
    provider = _StructuredAnswerProvider(structured_payload=payload, count_validation_payload=validator_payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-44_qa_completed_hike",
        sample_id="conv-44",
        question="How many times did Audrey and Andrew hike together?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D11:6] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-completed-hike",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D11:6"],
        prompt_context="[D11:6] Audrey: I should join you for a hike and bring my dogs.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support")
    assert response.metadata["count_validation_llm_used"] is True
    assert response.metadata["count_validation_llm_success"] is True
    assert response.metadata["count_validation_llm_scope"] == "completed_events"


def test_answer_generator_rejects_llm_count_validator_new_source_ref():
    payload = _answer_synthesis_payload("1", answer_type="count")
    payload["supporting_source_refs"] = ["D8:9"]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "Caroline mentioned two children.", "source_refs": ["D8:9"], "reason": "completed"}
    ]
    validator_payload = {
        "count_scope": "possessions",
        "validated_events": [
            {"event_id": "E1", "decision": "COUNT", "source_refs": ["D99:1"], "reason": "Invalid new source ref."}
        ],
        "final_count": 1,
        "confidence": "high",
        "validator_notes": "This should be rejected by safety checks.",
    }
    provider = _StructuredAnswerProvider(structured_payload=payload, count_validation_payload=validator_payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_pets",
        sample_id="conv-26",
        question="How many pets did Caroline want to adopt?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D8:9] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-pets-count",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D8:9"],
        prompt_context="D8:9 | Caroline: I want to adopt two children someday.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support")
    assert response.metadata["count_validation_llm_used"] is True
    assert response.metadata["count_validation_llm_success"] is False
    assert "new_source_ref" in response.metadata["count_validation_llm_error"]


def test_answer_generator_ignores_low_confidence_llm_count_validator():
    payload = _answer_synthesis_payload("1", answer_type="count")
    payload["supporting_source_refs"] = ["D11:6"]
    payload["counted_events"] = [
        {"event_id": "E1", "event_text": "Audrey should join Andrew for a hike.", "source_refs": ["D11:6"], "reason": "planned hike"}
    ]
    validator_payload = {
        "count_scope": "planned_events",
        "validated_events": [
            {"event_id": "E1", "decision": "COUNT", "source_refs": ["D11:6"], "reason": "Maybe planned."}
        ],
        "final_count": 1,
        "confidence": "low",
        "validator_notes": "Low confidence.",
    }
    provider = _StructuredAnswerProvider(structured_payload=payload, count_validation_payload=validator_payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-44_qa_low_conf",
        sample_id="conv-44",
        question="How many times did Audrey and Andrew plan to hike together?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D11:6] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-low-conf",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D11:6"],
        prompt_context="[D11:6] Audrey: I should join you for a hike and bring my dogs.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text.startswith("The retrieved context does not support")
    assert response.metadata["count_validation_llm_used"] is True
    assert response.metadata["count_validation_llm_success"] is False
    assert response.metadata["count_validation_llm_error"] == "low_confidence"


def test_answer_generator_applies_alias_bridge_fact():
    payload = _answer_synthesis_payload("John's old area", answer_type="place")
    payload["supporting_source_refs"] = ["D23:1"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-41_qa_42",
        sample_id="conv-41",
        question="What area was hit by a flood?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D23:1] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-bridge",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D14:21", "D23:1"],
        prompt_context="[D14:21] John: my old area, West County\n[D23:1] John: my old area was hit by a flood.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "West County"
    assert response.metadata["bridge_facts_used"]
    assert response.metadata["bridge_finalization_used"] is True
    assert response.metadata["bridge_finalization_target"] == "West County"
    assert response.metadata["bridge_finalization_action"] == "applied"


def test_answer_generator_does_not_extract_bridge_target_from_renderer_flags():
    payload = _answer_synthesis_payload("John's old area", answer_type="place")
    payload["supporting_source_refs"] = ["D23:1"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-41_qa_42",
        sample_id="conv-41",
        question="What area was hit by a flood?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D23:1] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-bridge-flags",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D14:21", "D23:1"],
        prompt_context=(
            "[D14:21] John: my old area, West County\n"
            "Flags: active\n"
            "[D23:1] John: my old area was hit by a flood.\n"
            "Flags: active"
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "West County"
    assert all(fact.get("target") != "Flags" for fact in response.metadata["bridge_facts_used"])


def test_answer_generator_ignores_conflicting_bridge_targets():
    payload = _answer_synthesis_payload("John's old area", answer_type="place")
    payload["supporting_source_refs"] = ["D23:1"]
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-41_qa_42",
        sample_id="conv-41",
        question="What area was hit by a flood?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D23:1] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-bridge-conflict",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D14:21", "D18:2", "D23:1"],
        prompt_context=(
            "[D14:21] John: my old area, West County\n"
            "[D18:2] John: my old area is East County\n"
            "[D23:1] John: my old area was hit by a flood."
        ),
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "John's old area"
    assert response.metadata["bridge_facts_used"] == []
    assert response.metadata["bridge_facts_conflicted"]
    assert response.metadata["bridge_finalization_conflicted"] is True


def test_answer_generator_repairs_legacy_bridge_alias_answer():
    provider = _AnswerRepairProvider(["The old area was hit by a flood.", "West County"])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-41_qa_42",
        sample_id="conv-41",
        question="What area was hit by a flood?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D23:1] ..."},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-bridge-repair",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D14:21", "D23:1"],
        prompt_context="[D14:21] John: my old area, West County\n[D23:1] John: my old area was hit by a flood.",
        latency_ms=1.0,
    )

    prompt, response = generator.generate(query_task, bundle)

    assert response.text == "West County"
    assert response.metadata["bridge_finalization_used"] is True
    assert response.metadata["bridge_finalization_action"] in {"applied", "deterministic_replace", "finalized"}
    assert response.metadata["answer_bridge_repair_used"] is False


def test_context_abstention_detection_covers_common_phrasings():
    assert PipelineRunner._answer_is_context_abstention(
        "The retrieved context does not provide information about Caroline's relationship status."
    )
    assert PipelineRunner._answer_is_context_abstention("The context does not mention a relationship status.")
    assert PipelineRunner._answer_is_context_abstention("That detail is not specified in the retrieved context.")
    assert not PipelineRunner._answer_is_context_abstention("Caroline is single.")


def test_answer_generator_keeps_generic_prompt_for_non_locomo_non_medmt():
    generator = AnswerGenerator(MockLLMProvider())
    query_task = QueryTask(
        query_task_id="generic-q1",
        sample_id="sample-generic",
        question="What does Alice prefer?",
        metadata={},
    )

    prompt = generator.build_prompt(query_task, _retrieval_bundle())

    assert prompt.startswith(ANSWER_GENERATION_PROMPT)
    assert "Question:\nWhat does Alice prefer?" in prompt


def test_answer_generator_repairs_unsupported_locomo_list_extras():
    provider = _AnswerRepairProvider(["The Hobbit, Dune", "The Hobbit"])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-repair",
        sample_id="conv-1",
        question="What books has Tim read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] Tim read The Hobbit."},
    )

    prompt, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text == "The Hobbit"
    assert "PREVIOUS_ANSWER:\nThe Hobbit, Dune" in prompt
    assert "REPAIR_INSTRUCTION:" in prompt
    assert prompt.startswith(LOCOMO_ANSWER_REPAIR_PROMPT)
    assert "TASK=LOCOMO_ANSWER_EVIDENCE_SYNTHESIS" not in prompt
    assert "Return structured data only" not in prompt
    assert "JSON object matching" not in prompt
    assert "Do not return JSON" in prompt
    assert prompt == provider.prompts[1]
    assert response.metadata["answer_prompt_stage"] == "repair"
    assert response.metadata["answer_postcheck_used"] is True
    assert response.metadata["answer_postcheck_issue"] == "unsupported_extra_items"
    assert response.metadata["answer_repair_used"] is True
    assert response.metadata["answer_repair_format"] == "plain_text"
    assert response.metadata["answer_repair_discarded"] is False
    assert provider.calls[0]["task"] == "answer_freeform_generation"
    assert provider.calls[1]["task"] == "answer_generation_repair"
    assert response.metadata["answer_initial_text"] == "The Hobbit, Dune"
    assert response.metadata["answer_initial_prompt_tokens"] == 5
    assert response.metadata["answer_initial_completion_tokens"] == 1


def test_answer_repair_arbitration_schema_and_prompt_registered():
    spec = get_structured_task_spec("answer_repair_arbitration")

    assert spec.json_schema["properties"]["decision"]["enum"] == [
        "keep_initial",
        "use_repair",
        "safe_abstain",
    ]
    assert "locomo_answer_repair_arbitration" in LOCOMO_ANSWER_REPAIR_ARBITRATION_PROMPT.casefold()


def test_repair_arbitration_trigger_detects_absolute_date_drop():
    query_task = QueryTask(
        query_task_id="conv-26_qa_58",
        sample_id="conv-26",
        question="When did Melanie make a plate in pottery class?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D14:4] plate"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-date",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D14:4"],
        prompt_context=(
            "[D14:4] date=25 August 2023; Melanie: Yeah, I made it in pottery class yesterday."
        ),
        latency_ms=1.0,
    )

    metadata = AnswerGenerator._repair_arbitration_trigger(
        query_task=query_task,
        retrieval_bundle=bundle,
        initial_answer_text="Melanie made a plate in pottery class on 24 August 2023.",
        repaired_answer_text="Melanie made a plate in pottery class yesterday.",
        issue="missing_supported_list_items",
        repair_validation_metadata={},
    )

    assert metadata["answer_repair_arbitration_triggered"] is True
    assert metadata["answer_repair_arbitration_trigger_reason"] == "absolute_date_dropped"


def test_answer_generator_keeps_initial_when_high_risk_repair_drops_year_without_arbitration():
    provider = _AnswerRepairProvider(
        [
            "Answer: 2020 items.\nRationale: Audrey had the dogs for 3 years in 2023.",
            "I'm unable to determine the year Audrey adopted the first three of her dogs.",
        ]
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-44_qa_0",
        sample_id="conv-44",
        question="Which year did Audrey adopt the first three of her dogs?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:7] 3 years"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-year",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D1:7"],
        prompt_context="[D1:7] date=27 March 2023; Audrey: I've had them for 3 years.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "2020 items."
    assert response.metadata["answer_repair_arbitration_triggered"] is True
    assert response.metadata["answer_repair_arbitration_used"] is False
    assert response.metadata["answer_repair_arbitration_action"] == "keep_initial"
    assert response.metadata["answer_repair_discard_reason"] == "arbitration_keep_initial"
    assert response.metadata["answer_repair_used"] is False


def test_answer_generator_accepts_arbitration_use_repair_decision():
    provider = _StructuredAnswerProvider(
        text_responses=[
            "Answer: 2020 items.\nRationale: Audrey had the dogs for 3 years in 2023.",
            "I'm unable to determine the year Audrey adopted the first three of her dogs.",
        ],
        answer_repair_arbitration_payload={
            "decision": "use_repair",
            "repair_violation": "none",
            "confidence": "high",
            "reason": "The repaired answer is safer than the malformed initial answer.",
        },
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-44_qa_0",
        sample_id="conv-44",
        question="Which year did Audrey adopt the first three of her dogs?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:7] 3 years"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-year",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D1:7"],
        prompt_context="[D1:7] date=27 March 2023; Audrey: I've had them for 3 years.",
        latency_ms=1.0,
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "I'm unable to determine the year Audrey adopted the first three of her dogs."
    assert response.metadata["answer_repair_arbitration_used"] is True
    assert response.metadata["answer_repair_arbitration_decision"] == "use_repair"
    assert response.metadata["answer_repair_used"] is True
    assert provider.generate_structured_calls == 1


def test_answer_generator_discards_repair_that_drops_supported_activity_item():
    initial = "Melanie uses several activities to destress: running, playing violin, engaging in pottery, and family time in nature."
    bad_repair = "Melanie uses several activities to destress, including running, playing her violin, and family time in nature."
    provider = _AnswerRepairProvider([initial, bad_repair])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_24",
        sample_id="conv-26",
        question="What does Melanie do to destress?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D5:4] pottery [D7:22] running"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-destress",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D5:4", "D7:22"],
        prompt_context=(
            "[D5:4] Melanie: I just signed up for a pottery class yesterday. "
            "It's like therapy for me, letting me express myself and get creative.\n"
            "[D7:22] Melanie: I've been running farther to de-stress, which has been great for my headspace."
        ),
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "activity",
                "tags": ["list_like"],
            },
            "grounded_exact_terms": ["pottery class", "pottery", "running"],
            "grounded_display_items": ["pottery class", "running"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": [
                "Melanie signed up for a pottery class as therapy.",
                "Melanie runs to de-stress.",
            ],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == initial
    repair_prompt = provider.prompts[-1]
    assert "MUST_KEEP_SUPPORTED_ITEMS:" in repair_prompt
    assert "pottery" in repair_prompt
    assert response.metadata["answer_repair_used"] is False
    assert response.metadata["answer_repair_discarded"] is True
    assert response.metadata["answer_repair_discard_reason"] == "dropped_supported_required_items"
    assert "pottery" in response.metadata["answer_repair_dropped_supported_items"]
    assert response.metadata["answer_repair_preserved_initial_answer"] is True
    assert "running" in response.metadata["answer_required_item_candidates"]
    assert any("pottery" in item for item in response.metadata["answer_required_item_candidates"])
    assert "reading" not in response.metadata["answer_missing_supported_list_items"]
    assert "playing violin" not in response.metadata["answer_missing_supported_list_items"]


def test_answer_generator_repairs_overgeneric_preference_items():
    provider = _AnswerRepairProvider(["animals and nature", "dinosaurs and nature"])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_19",
        sample_id="conv-26",
        question="What do Melanie's kids like?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D6:6] dinosaur exhibit"},
    )

    prompt, response = generator.generate(query_task, _locomo_preference_retrieval_bundle())

    assert response.text == "dinosaurs and nature"
    assert "source_surface_terms=dinosaur exhibit" in prompt
    assert "Replace broad umbrella words" in prompt
    assert response.metadata["answer_postcheck_issue"] == "overgeneric_item"
    assert response.metadata["answer_overgeneric_item_detected"] is True
    assert response.metadata["answer_specific_item_repair_used"] is True
    assert response.metadata["answer_specific_replacement_candidates"]


def test_answer_generator_repairs_scope_mismatched_preference_extras():
    provider = _AnswerRepairProvider(
        ["dinosaur exhibit, crafting with clay, nature", "dinosaur exhibit and nature"]
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_19",
        sample_id="conv-26",
        question="What do Melanie's kids like?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D6:6] dinosaur exhibit"},
    )

    _, response = generator.generate(query_task, _locomo_preference_retrieval_bundle())

    assert response.text == "dinosaur exhibit and nature"
    assert response.metadata["answer_postcheck_issue"] == "scope_mismatched_extra_item"
    assert response.metadata["answer_scope_mismatched_extra_items"] == ["crafting with clay"]
    assert response.metadata["answer_specific_item_repair_used"] is True


def test_answer_generator_keeps_specific_preference_answer_without_repair():
    provider = _AnswerRepairProvider(["dinosaurs and nature"])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_19",
        sample_id="conv-26",
        question="What do Melanie's kids like?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D6:6] dinosaur exhibit"},
    )

    _, response = generator.generate(query_task, _locomo_preference_retrieval_bundle())

    assert response.text == "dinosaurs and nature"
    assert response.metadata["answer_postcheck_used"] is False
    assert response.metadata["answer_postcheck_issue"] is None
    assert response.metadata["answer_overgeneric_item_detected"] is False


def test_answer_generator_repairs_missing_supported_list_items():
    provider = _AnswerRepairProvider(
        ["John attended a violin concert.", "John attended a live music event and a violin concert."]
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-41_qa_36",
        sample_id="conv-41",
        question="What music events has John attended?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] live music event"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-list-missing",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D1:1", "D2:3"],
        prompt_context="[D1:1] John attended a live music event.\n[D2:3] John attended a violin concert.",
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "event",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": ["live music event", "violin concert"],
            "grounded_exact_terms": ["live music event", "violin concert"],
            "grounded_display_items": ["live music event", "violin concert"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": [
                "John attended a live music event.",
                "John attended a violin concert.",
            ],
        },
    )

    prompt, response = generator.generate(query_task, bundle)

    assert response.text == "John attended a live music event and a violin concert."
    assert "SUPPORTED_LIST_ITEMS:" in prompt
    assert "live music event" in prompt
    assert response.metadata["answer_postcheck_issue"] == "missing_supported_list_items"
    assert response.metadata["answer_missing_supported_list_items"] == ["live music event"]
    assert response.metadata["answer_list_coverage_repair_used"] is True


def test_answer_generator_does_not_list_repair_single_value_kind_question():
    provider = _AnswerRepairProvider(["Jolene was working on an electrical engineering project at the beginning of January 2023."])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-48_qa_0",
        sample_id="conv-48",
        question="What kind of project was Jolene working on in the beginning of January 2023?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:2] electrical engineering project"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-single-value-type",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D1:2"],
        prompt_context="[D1:2] Jolene finished an electrical engineering project last week.",
        latency_ms=1.0,
        metadata={
            "grounded_source_surface_terms": [
                "Jolene finished an electrical engineering project last week",
                "programming",
                "sustainable water purifier",
            ],
            "grounded_exact_terms": ["electrical engineering project", "programming", "sustainable water purifier"],
            "grounded_display_items": ["electrical engineering project", "sustainable water purifier"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": ["Jolene finished an electrical engineering project last week."],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Jolene was working on an electrical engineering project at the beginning of January 2023."
    assert response.metadata["answer_postcheck_issue"] is None
    assert response.metadata["answer_list_coverage_skipped"] is True
    assert response.metadata["answer_list_coverage_skip_reason"] == "value_question"
    assert response.metadata["answer_list_coverage_repair_used"] is False
    assert len(provider.calls) == 1


def test_answer_generator_does_not_list_repair_duration_count_question():
    provider = _AnswerRepairProvider(["After about 11 weeks, Tim reconnected with the fellow Harry Potter fan from California."])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-43_qa_16",
        sample_id="conv-43",
        question="After how many weeks did Tim reconnect with the fellow Harry Potter fan from California?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D4:7] three weeks"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-duration-count",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D4:7"],
        prompt_context="[D4:7] Tim reconnected with a fellow Harry Potter fan from California after three weeks.",
        latency_ms=1.0,
        metadata={
            "grounded_source_surface_terms": [
                "Harry Potter fan project",
                "discussing collaborations",
                "characters, spells",
                "London",
            ],
            "grounded_exact_terms": ["Harry Potter fan project", "discussing collaborations", "London"],
            "grounded_display_items": ["Harry Potter fan project", "London"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": ["Tim reconnected after three weeks."],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "After about 11 weeks, Tim reconnected with the fellow Harry Potter fan from California."
    assert response.metadata["answer_postcheck_issue"] is None
    assert response.metadata["answer_list_coverage_skipped"] is True
    assert response.metadata["answer_list_coverage_skip_reason"] == "duration_count_question"
    assert response.metadata["answer_missing_supported_list_items"] == []
    assert len(provider.calls) == 1


def test_answer_generator_does_not_list_repair_single_value_activity_question():
    provider = _AnswerRepairProvider(["Sam takes up painting classes in October 2023."])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-49_qa_60",
        sample_id="conv-49",
        question="Which new activity does Sam take up in October 2023?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D9:4] kayaking"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-single-activity",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D9:4"],
        prompt_context="[D9:4] Sam started kayaking in October 2023.",
        latency_ms=1.0,
        metadata={
            "grounded_source_surface_terms": ["hiking", "trying painting", "paint set", "watercolors"],
            "grounded_exact_terms": ["hiking", "painting", "watercolors"],
            "grounded_display_items": ["hiking", "painting classes", "watercolors"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": ["Sam was thinking about trying painting."],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Sam takes up painting classes in October 2023."
    assert response.metadata["answer_postcheck_issue"] is None
    assert response.metadata["answer_list_coverage_skipped"] is True
    assert response.metadata["answer_list_coverage_skip_reason"] == "value_question"
    assert response.metadata["answer_missing_supported_list_items"] == []
    assert len(provider.calls) == 1


def test_answer_generator_discards_list_repair_when_missing_items_do_not_improve():
    provider = _AnswerRepairProvider(["John attended a violin concert.", "John attended a violin concert."])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-41_qa_36",
        sample_id="conv-41",
        question="What music events has John attended?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] live music event"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-list-missing-no-improvement",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D1:1", "D2:3"],
        prompt_context="[D1:1] John attended a live music event.\n[D2:3] John attended a violin concert.",
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "event",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": ["live music event", "violin concert"],
            "grounded_exact_terms": ["live music event", "violin concert"],
            "grounded_display_items": ["live music event", "violin concert"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": [
                "John attended a live music event.",
                "John attended a violin concert.",
            ],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "John attended a violin concert."
    assert response.metadata["answer_postcheck_issue"] == "missing_supported_list_items"
    assert response.metadata["answer_repair_used"] is False
    assert response.metadata["answer_repair_discarded"] is True
    assert response.metadata["answer_repair_discard_reason"] == "list_coverage_not_improved"
    assert response.metadata["answer_repair_preserved_initial_answer"] is True
    assert response.metadata["answer_list_coverage_repair_used"] is False
    assert response.metadata["answer_list_coverage_repair_success"] is False
    assert response.metadata["answer_repair_missing_required_items_after_repair"] == ["live music event"]


def test_answer_generator_repairs_event_list_scope_extras_and_keeps_required_items():
    initial = (
        "Caroline attended a council meeting for adoption, joined a mentorship program for LGBTQ youth, "
        "applied to adoption agencies, and shared her journey at a school event to encourage students."
    )
    repaired = "Caroline joined a mentoring program and gave a school speech."
    provider = _AnswerRepairProvider([initial, repaired])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_34",
        sample_id="conv-26",
        question="What events has Caroline participated in to help children?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D3:1] school event [D4:2] mentorship"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-event-list-repair",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D3:1", "D4:2", "D8:1", "D9:1"],
        prompt_context=(
            "[D3:1] Caroline: I wanted to tell you about my school event. "
            "I shared my journey and encouraged students.\n"
            "[D4:2] Caroline: I joined a mentorship program for LGBTQ youth.\n"
            "[D8:1] Caroline: The council meeting about adoption was emotional.\n"
            "[D9:1] Caroline: I applied to adoption agencies."
        ),
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "event",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": [
                "school event",
                "school speech",
                "mentorship program",
                "mentoring program",
            ],
            "grounded_exact_terms": ["school speech", "mentoring program"],
            "grounded_display_items": ["school speech", "mentoring program"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": [
                "Caroline shared her journey at a school event and encouraged students.",
                "Caroline joined a mentorship program for LGBTQ youth.",
            ],
        },
    )

    prompt, response = generator.generate(query_task, bundle)

    assert response.text == repaired
    assert "REMOVE_SCOPE_MISMATCHED_ITEMS:" in prompt
    assert "council meeting" in prompt
    assert "adoption agencies" in prompt
    assert "mentoring program" in prompt
    assert "school speech" in prompt
    assert response.metadata["answer_postcheck_issue"] == "scope_mismatched_extra_item"
    assert response.metadata["answer_scope_mismatched_extra_items"] == [
        "council meeting",
        "adoption agencies",
    ]
    assert response.metadata["event_canonical_alias_items"] == ["school speech", "mentoring program"]
    assert set(response.metadata["answer_supported_required_items"]) == {"mentoring program", "school speech"}
    assert response.metadata["answer_repair_removed_scope_mismatched_items"] == [
        "council meeting",
        "adoption agencies",
    ]


def test_answer_generator_repairs_abstain_with_supported_list_items():
    provider = _AnswerRepairProvider(
        ["The retrieved context does not support an answer to this question.", "Charlotte's Web"]
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_23",
        sample_id="conv-26",
        question="What books has Melanie read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D6:10] Charlotte's Web"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-list-abstain",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D6:10"],
        prompt_context="[D6:10] Melanie read Charlotte's Web with the kids.",
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "book",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": ["Charlotte's Web"],
            "grounded_exact_terms": ["Charlotte's Web"],
            "grounded_display_items": ["Charlotte's Web"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": ["Melanie read Charlotte's Web with the kids."],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Charlotte's Web"
    assert response.metadata["answer_postcheck_issue"] == "abstain_despite_supported_list_items"
    assert response.metadata["answer_abstain_despite_supported_items"] is True
    assert response.metadata["answer_list_coverage_repair_used"] is True


def test_answer_generator_book_scope_rejects_non_book_surfaces_from_missing_items():
    provider = _AnswerRepairProvider(
        ["The retrieved context does not support an answer to this question.", "Charlotte's Web"]
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-26_qa_23",
        sample_id="conv-26",
        question="What books has Melanie read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D6:10] Charlotte's Web"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-book-scope",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D6:10", "D4:2", "D3:1"],
        prompt_context=(
            "[D6:10] Melanie read Charlotte's Web with the kids.\n"
            "[D4:2] Melanie joined an LGBTQ support group.\n"
            "[D3:1] Caroline spoke at a school event."
        ),
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "book",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": ["Charlotte's Web", "LGBTQ support group", "school event"],
            "grounded_exact_terms": ["Charlotte's Web", "LGBTQ support group", "school event"],
            "grounded_display_items": ["Charlotte's Web", "LGBTQ support group", "school event"],
            "grounded_display_counts": [],
            "grounded_display_key_facts": [
                "Melanie read Charlotte's Web with the kids.",
                "Melanie joined an LGBTQ support group.",
                "Caroline spoke at a school event.",
            ],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "Charlotte's Web"
    assert response.metadata["answer_missing_supported_list_items"] == ["Charlotte's Web"]
    assert "LGBTQ support group" not in response.metadata["answer_missing_supported_list_items"]
    assert "school event" not in response.metadata["answer_missing_supported_list_items"]
    assert "LGBTQ support group" in response.metadata["answer_scope_rejected_items"]


def test_answer_generator_does_not_repair_generic_only_list_surface():
    provider = _AnswerRepairProvider(["John attended a concert."])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="generic-list",
        sample_id="conv-1",
        question="What events has John attended?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] event"},
    )
    bundle = RetrievalBundle(
        retrieval_event_id="retrieval-generic-list",
        selected_pages=[],
        candidate_trajectories=[],
        snapshot_hits=[],
        expanded_snapshots=[],
        source_message_ids=[],
        source_message_refs=["D1:1"],
        prompt_context="[D1:1] John attended an event.",
        latency_ms=1.0,
        metadata={
            "query_shape": {
                "list_like": True,
                "multi_entity": False,
                "comparison_like": False,
                "count_like": False,
                "item_family": "event",
                "tags": ["list_like"],
            },
            "grounded_source_surface_terms": ["event"],
            "grounded_exact_terms": ["event"],
            "grounded_display_items": ["event"],
            "grounded_display_counts": [],
        },
    )

    _, response = generator.generate(query_task, bundle)

    assert response.text == "John attended a concert."
    assert response.metadata["answer_postcheck_used"] is False


def test_answer_generator_extracts_final_answer_from_json_repair():
    provider = _AnswerRepairProvider(
        ["The Hobbit, Dune", '{"can_answer":true,"final_answer":"The Hobbit"}']
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-repair-json",
        sample_id="conv-1",
        question="What books has Tim read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] Tim read The Hobbit."},
    )

    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text == "The Hobbit"
    assert response.metadata["answer_repair_format"] == "json"
    assert response.metadata["answer_repair_json_extracted"] is True
    assert response.metadata["answer_repair_discarded"] is False


def test_answer_generator_extracts_final_answer_from_fenced_json_repair():
    provider = _AnswerRepairProvider(
        ["The Hobbit, Dune", '```json\n{"can_answer":true,"final_answer":"The Hobbit"}\n```']
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-repair-fenced-json",
        sample_id="conv-1",
        question="What books has Tim read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] Tim read The Hobbit."},
    )

    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text == "The Hobbit"
    assert response.metadata["answer_repair_format"] == "fenced_json"
    assert response.metadata["answer_repair_json_extracted"] is True
    assert response.metadata["answer_repair_discarded"] is False


def test_answer_generator_converts_can_answer_false_json_repair_to_abstention():
    provider = _AnswerRepairProvider(
        [
            "The Hobbit, Dune",
            '{"can_answer":false,"final_answer":"","abstain_reason":"no supported book evidence remains"}',
        ]
    )
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-repair-json-abstain",
        sample_id="conv-1",
        question="What books has Tim read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] Tim read The Hobbit."},
    )

    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text == "The Hobbit, Dune"
    assert response.metadata["answer_repair_json_extracted"] is True
    assert response.metadata["answer_repair_discarded"] is True
    assert response.metadata["answer_repair_discard_reason"] == "arbitration_keep_initial"
    assert response.metadata["answer_repair_arbitration_triggered"] is True
    assert response.metadata["answer_repair_arbitration_action"] == "keep_initial"


def test_answer_generator_discards_malformed_json_repair_and_keeps_initial_answer_prompt():
    provider = _AnswerRepairProvider(["The Hobbit, Dune", '{"can_answer":true,'])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-repair-bad-json",
        sample_id="conv-1",
        question="What books has Tim read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] Tim read The Hobbit."},
    )

    prompt, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text == "The Hobbit, Dune"
    assert prompt == provider.prompts[0]
    assert response.metadata["answer_prompt_stage"] == "initial"
    assert response.metadata["answer_repair_attempted"] is True
    assert response.metadata["answer_repair_used"] is False
    assert response.metadata["answer_repair_discarded"] is True
    assert response.metadata["answer_repair_discard_reason"] == "json_parse_failed"


def test_answer_generator_skips_postcheck_for_synthesis_abstention_with_list_surfaces():
    payload = _answer_synthesis_payload("", answer_type="unknown")
    payload["can_answer"] = False
    payload["supporting_facts"] = []
    payload["supporting_source_refs"] = []
    payload["abstain_reason"] = "book evidence is missing, and no supported title answers the question"
    provider = _StructuredAnswerProvider(structured_payload=payload)
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="locomo-q-abstain-list",
        sample_id="conv-1",
        question="What books has Tim read?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": ""},
    )

    _, response = generator.generate(query_task, _locomo_retrieval_bundle_with_shape())

    assert response.text.startswith("The retrieved context does not support an answer")
    assert response.metadata["answer_postcheck_skipped"] is True
    assert response.metadata["answer_postcheck_skip_reason"] == "synthesis_can_answer_false"
    assert response.metadata["answer_postcheck_used"] is False
    assert response.metadata["answer_repair_used"] is False
    assert provider.generate_calls == 1


def test_answer_generator_discards_bad_count_repair_without_overwriting_initial_answer():
    provider = _AnswerRepairProvider(["3", '{"can_answer":true,'])
    generator = AnswerGenerator(provider)
    query_task = QueryTask(
        query_task_id="conv-42_qa_19",
        sample_id="conv-42",
        question="How many times has Joanna found new hiking trails?",
        metadata={"category_name": "multi_hop", "evidence_only_conversation": "[D1:1] Joanna found two trails."},
    )

    prompt, response = generator.generate(query_task, _locomo_count_retrieval_bundle())

    assert response.text == "Three times."
    assert prompt == provider.prompts[0]
    assert response.metadata["answer_count_naturalized"] is True
    assert response.metadata["answer_repair_used"] is False


def test_benchmark_judge_uses_full_medmt_context():
    judge = BenchmarkJudge(MockLLMProvider(model_name="mock-judge"))
    query_task = QueryTask(
        query_task_id="medmt-q1",
        sample_id="sample-a",
        question="What do you know about my smoking habits?",
        metadata={
            "test_point": "Explicitly identify the contradiction and ask for confirmation.",
            "judge_context": {
                "dataset": "medmt",
                "category": "Instruction Clarification",
                "subtype": "Information Contradiction",
                "full_dialogue": "[TURN 1][SYSTEM] Keep answers short.\n[TURN 2][USER] I smoke 5 cigarettes a day.",
                "final_user_turn": "What do you know about my smoking habits?",
                "rubric": "Explicitly identify the contradiction and ask for confirmation.",
            },
        },
    )

    prompt = judge.build_prompt("medmt", query_task, "You said both that you smoke and that you never smoke. Can you confirm which is correct?")

    assert prompt.startswith(MEDMT_JUDGE_PROMPT)
    assert "CATEGORY:\nInstruction Clarification" in prompt
    assert "SUBTYPE:\nInformation Contradiction" in prompt
    assert "FULL_DIALOGUE:\n[TURN 1][SYSTEM] Keep answers short." in prompt
    assert "FINAL_USER_TURN:\nWhat do you know about my smoking habits?" in prompt
    assert "RUBRIC:\nExplicitly identify the contradiction and ask for confirmation." in prompt
    assert "CANDIDATE_ANSWER:\nYou said both that you smoke and that you never smoke." in prompt


def test_locomo_judge_prompt_documents_lenient_core_and_time_rules():
    for prompt in [LOCOMO_JUDGE_PROMPT, LOCOMO_JUDGE_STRUCTURED_PROMPT]:
        assert "Allow semantic equivalence" in prompt
        assert "Harmless same-category extras are allowed" in prompt
        assert "all required gold facts/items are covered" in prompt
        assert "Missing a required list item is PARTIAL" in prompt
        assert "unqualified wrong count is INCORRECT" in prompt
        assert "retrieved-evidence lower bound" in prompt
        assert "May 7th" in prompt
        assert "7 May" in prompt
        assert "last Tuesday" in prompt
        assert "LGBTQ support group" in prompt
        assert "LGBTQ pride parade" in prompt
        assert "shared her journey at a school event" in prompt
        assert "mentorship program for LGBTQ youth" in prompt
        assert "omits \"school speech\"" in prompt
        assert "count, time, or place" in prompt


def test_benchmark_judge_structured_success_does_not_fallback():
    provider = _StructuredJudgeProvider(structured_payload={"verdict": "CORRECT"})
    judge = BenchmarkJudge(provider)
    query_task = QueryTask(
        query_task_id="locomo-q1",
        sample_id="conv-1",
        question="What does Alice prefer?",
        gold_answer="Tea",
        metadata={},
    )

    result = judge.judge("locomo", query_task, "Tea")

    assert result is not None
    assert result.verdict == "correct"
    assert result.score == 1.0
    assert result.rationale is None
    assert result.metadata["judge_mode"] == "structured"
    assert result.metadata["structured_success"] is True
    assert result.metadata["structured_fallback_used"] is False
    assert result.metadata["judge_score_policy"] == "partial_credit_v1"
    assert provider.generate_structured_calls == 1
    assert provider.generate_calls == 0


def test_benchmark_judge_structured_partial_scores_half_credit():
    provider = _StructuredJudgeProvider(structured_payload={"verdict": "PARTIAL"})
    judge = BenchmarkJudge(provider)
    query_task = QueryTask(
        query_task_id="locomo-q1-partial",
        sample_id="conv-1",
        question="What does Alice prefer?",
        gold_answer="Tea and coffee",
        metadata={},
    )

    result = judge.judge("locomo", query_task, "Tea")

    assert result is not None
    assert result.verdict == "partial"
    assert result.score == 0.5
    assert result.metadata["judge_score_policy"] == "partial_credit_v1"


def test_benchmark_judge_structured_failure_falls_back_to_text():
    provider = _StructuredJudgeProvider(
        structured_exception=StructuredOutputError(
            "invalid schema rejected by provider",
            vendor="openai",
            strategy="openai_json_schema",
        ),
        text_response="CORRECT.",
    )
    judge = BenchmarkJudge(provider)
    query_task = QueryTask(
        query_task_id="locomo-q2",
        sample_id="conv-2",
        question="What does Alice prefer?",
        gold_answer="Tea",
        metadata={},
    )

    result = judge.judge("locomo", query_task, "Tea")

    assert result is not None
    assert result.verdict == "correct"
    assert result.metadata["judge_mode"] == "text_fallback"
    assert result.metadata["structured_requested"] is True
    assert result.metadata["structured_success"] is False
    assert result.metadata["structured_fallback_used"] is True
    assert result.metadata["structured_fallback_category"] == "structured_schema_error"
    assert provider.generate_structured_calls == 1
    assert provider.generate_calls == 1


def test_benchmark_judge_text_fallback_preserves_rationale_when_present():
    provider = _StructuredJudgeProvider(
        structured_exception=StructuredOutputError(
            "invalid schema rejected by provider",
            vendor="openai",
            strategy="openai_json_schema",
        ),
        text_response="VERDICT: PARTIAL\nRATIONALE: Missing one required item.",
    )
    judge = BenchmarkJudge(provider)
    query_task = QueryTask(
        query_task_id="locomo-q2-rationale",
        sample_id="conv-2",
        question="What does Alice prefer?",
        gold_answer="Tea and coffee",
        metadata={},
    )

    result = judge.judge("locomo", query_task, "Tea")

    assert result is not None
    assert result.verdict == "partial"
    assert result.score == 0.5
    assert result.rationale == "Missing one required item."
    assert result.metadata["judge_mode"] == "text_fallback"
    assert result.metadata["structured_fallback_used"] is True


def test_benchmark_judge_returns_judge_error_when_text_fallback_is_unparseable():
    provider = _StructuredJudgeProvider(
        structured_exception=StructuredOutputError(
            "invalid schema rejected by provider",
            vendor="openai",
            strategy="openai_json_schema",
        ),
        text_response="I think the candidate is probably correct overall.",
    )
    judge = BenchmarkJudge(provider)
    query_task = QueryTask(
        query_task_id="locomo-q3",
        sample_id="conv-3",
        question="What does Alice prefer?",
        gold_answer="Tea",
        metadata={},
    )

    result = judge.judge("locomo", query_task, "Tea")

    assert result is not None
    assert result.verdict == "judge_error"
    assert result.metadata["judge_mode"] == "text_fallback"
    assert result.metadata["judge_execution_failed"] is True
    assert result.metadata["structured_fallback_category"] == "text_parser_error"
    assert result.metadata["judge_error_type"] == "ParserValidationError"


def test_parse_judge_verdict_accepts_light_format_variants():
    assert parse_judge_verdict("CORRECT.").verdict == "CORRECT"
    assert parse_judge_verdict("`CORRECT`").verdict == "CORRECT"
    assert parse_judge_verdict("```text\nINCORRECT\n```").verdict == "INCORRECT"
    assert parse_judge_verdict("PARTIAL.").verdict == "PARTIAL"
    assert parse_judge_verdict("`PARTIAL`").verdict == "PARTIAL"
    assert parse_judge_verdict("```text\nPARTIAL\n```").verdict == "PARTIAL"
    assert parse_judge_verdict("VERDICT: PARTIAL").verdict == "PARTIAL"
    assert parse_judge_verdict("PASS").verdict == "CORRECT"
    assert parse_judge_verdict("FAIL").verdict == "INCORRECT"
