from __future__ import annotations

import json

import pytest

from trajpatch.providers.litellm_provider import LiteLLMProvider
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.metering import MeteredLLMProvider
from trajpatch.providers.structured_outputs import (
    get_structured_task_spec,
    parse_structured_payload,
    structured_schema_diagnostics,
    vendor_schema,
)
from trajpatch.types import LLMResponse, ModelInfo, NormalizedMessage, StructuredLLMResponse


class _FailingStructuredProvider(LLMProvider):
    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        return LLMResponse(text="ok")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="failing-structured", is_remote=True)

    def supports_structured(self, task: str) -> bool:
        return True

    def generate_structured(self, messages, *, spec, system_prompt=None, metadata=None) -> StructuredLLMResponse:
        raise RuntimeError("schema rejected by provider")


class _SuccessfulStructuredProvider(LLMProvider):
    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        return LLMResponse(text="ok")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="successful-structured", is_remote=True)

    def supports_structured(self, task: str) -> bool:
        return True

    def generate_structured(self, messages, *, spec, system_prompt=None, metadata=None) -> StructuredLLMResponse:
        parsed = parse_structured_payload(
            spec,
            {"verdict": "CORRECT"} if spec.task.endswith("_judge") else {},
        )
        return StructuredLLMResponse(
            parsed=parsed,
            prompt_tokens=3,
            completion_tokens=1,
            metadata={"structured_vendor": "openai"},
        )


def _assert_strict_schema(schema: dict, *, vendor: str) -> None:
    diagnostics = structured_schema_diagnostics(schema, vendor=vendor)
    assert diagnostics["missing_required"] == 0
    assert diagnostics["missing_additional_properties"] == 0
    assert diagnostics["unsupported_keywords"] == 0


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_claim_stage_openai_schemas_are_strict_and_do_not_leak_gemini_keywords():
    for task in ("episodic_claim_text_extract", "claim_signal_extract"):
        schema = vendor_schema(get_structured_task_spec(task), "openai")
        _assert_strict_schema(schema, vendor="openai")
        assert not _contains_key(schema, "propertyOrdering")


def test_claim_stage_gemini_schemas_keep_property_ordering():
    for task in ("episodic_claim_text_extract", "claim_signal_extract"):
        schema = vendor_schema(get_structured_task_spec(task), "gemini")
        _assert_strict_schema(schema, vendor="gemini")
        assert _contains_key(schema, "propertyOrdering")


def test_claim_stage_structured_payloads_parse_nullable_fields_and_empty_arrays():
    claim_text = parse_structured_payload(
        get_structured_task_spec("episodic_claim_text_extract"),
        {
            "has_claims": True,
            "claims": [
                {
                    "status": "active",
                    "source_message_ids": ["sample-m0001"],
                    "supporting_quote": "I play clarinet.",
                    "text": "The user plays clarinet.",
                }
            ],
            "reason": None,
        },
    )
    assert claim_text.has_claims is True
    assert claim_text.reason is None

    claim_signal = parse_structured_payload(
        get_structured_task_spec("claim_signal_extract"),
        {
            "exact_terms": [],
            "facets": [
                {
                    "relation": "event_type",
                    "value": "school speech",
                    "entity": None,
                    "value_span": None,
                    "source_claim_id": "tmp-c1",
                    "source_message_ids": [],
                }
            ],
            "display_items": [],
            "display_named_entities": [],
            "display_counts": [],
            "display_key_facts": [],
        },
    )
    assert claim_signal.facets[0].entity is None
    assert claim_signal.facets[0].value_span is None


def test_openai_structured_requests_json_schema(monkeypatch):
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "has_memory": True,
                                    "memory": {
                                        "summary_content": "User discussed smoking habits.",
                                        "context": "The exchange updates the user's smoking status.",
                                        "keywords": ["smoking", "habit"],
                                    },
                                    "reason": None,
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self):
            self.chat = FakeChat()

    provider = LiteLLMProvider("gpt-4o-mini")
    monkeypatch.setattr(provider, "_get_openai_client", lambda: FakeOpenAIClient())

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="extract", turn_index=0)],
        spec=get_structured_task_spec("episodic_extract"),
        metadata={"task": "episodic_extract"},
    )

    assert response.parsed.has_memory is True
    assert not hasattr(response.parsed.memory, "claims")
    assert response.metadata["structured_vendor"] == "openai"
    assert response.metadata["structured_strategy"] == "openai_json_schema"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "trajpatch_episodic_extract"


def test_openai_claim_text_structured_request_uses_strict_schema(monkeypatch):
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            schema = kwargs["response_format"]["json_schema"]["schema"]
            _assert_strict_schema(schema, vendor="openai")
            assert not _contains_key(schema, "propertyOrdering")
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "has_claims": True,
                                    "claims": [
                                        {
                                            "status": "active",
                                            "source_message_ids": ["sample-m0001"],
                                            "supporting_quote": "I play clarinet.",
                                            "text": "The user plays clarinet.",
                                        }
                                    ],
                                    "reason": None,
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self):
            self.chat = FakeChat()

    provider = LiteLLMProvider("gpt-4o-mini")
    monkeypatch.setattr(provider, "_get_openai_client", lambda: FakeOpenAIClient())

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="extract claims", turn_index=0)],
        spec=get_structured_task_spec("episodic_claim_text_extract"),
        metadata={"task": "episodic_claim_text_extract"},
    )

    assert response.parsed.has_claims is True
    assert response.parsed.claims[0].text == "The user plays clarinet."
    assert calls[0]["response_format"]["json_schema"]["name"] == "trajpatch_claim_text_extract"


def test_openai_claim_signal_structured_request_uses_strict_schema(monkeypatch):
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            schema = kwargs["response_format"]["json_schema"]["schema"]
            _assert_strict_schema(schema, vendor="openai")
            assert not _contains_key(schema, "propertyOrdering")
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "exact_terms": [
                                        {
                                            "surface": "clarinet",
                                            "category": "instrument",
                                            "source_claim_id": "tmp-c1",
                                            "source_message_ids": ["sample-m0001"],
                                        }
                                    ],
                                    "facets": [
                                        {
                                            "relation": "event_type",
                                            "value": "school speech",
                                            "entity": None,
                                            "value_span": None,
                                            "source_claim_id": "tmp-c1",
                                            "source_message_ids": [],
                                        }
                                    ],
                                    "display_items": ["clarinet"],
                                    "display_named_entities": [],
                                    "display_counts": [],
                                    "display_key_facts": ["The user plays clarinet."],
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self):
            self.chat = FakeChat()

    provider = LiteLLMProvider("gpt-4o-mini")
    monkeypatch.setattr(provider, "_get_openai_client", lambda: FakeOpenAIClient())

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="extract signals", turn_index=0)],
        spec=get_structured_task_spec("claim_signal_extract"),
        metadata={"task": "claim_signal_extract"},
    )

    assert response.parsed.exact_terms[0].surface == "clarinet"
    assert response.parsed.facets[0].entity is None
    assert calls[0]["response_format"]["json_schema"]["name"] == "trajpatch_claim_signal_extract"


def test_gemini_structured_requests_json_schema(monkeypatch):
    calls: list[dict] = []

    class FakeUsage:
        prompt_token_count = 13
        candidates_token_count = 5

    class FakeGeminiResponse:
        text = json.dumps({"decision": "NEW", "selected_candidate": None, "rationale": "Different topic."})
        usage_metadata = FakeUsage()

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return FakeGeminiResponse()

    class FakeGeminiClient:
        def __init__(self):
            self.models = FakeModels()

    provider = LiteLLMProvider("gemini-2.0-flash")
    monkeypatch.setattr(provider, "_get_gemini_client", lambda: FakeGeminiClient())

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="match", turn_index=0)],
        spec=get_structured_task_spec("trajectory_match"),
        metadata={"task": "trajectory_match"},
    )

    assert response.parsed.decision == "NEW"
    assert response.metadata["structured_vendor"] == "gemini"
    assert response.metadata["structured_strategy"] == "gemini_json_schema"
    assert calls[0]["config"]["response_mime_type"] == "application/json"
    assert calls[0]["config"]["response_json_schema"]["propertyOrdering"]


def test_anthropic_structured_uses_tool_schema(monkeypatch):
    calls: list[dict] = []

    class FakeAnthropicResponse:
        usage = type("Usage", (), {"input_tokens": 9, "output_tokens": 3})()
        content = [
            {
                "type": "tool_use",
                "name": "trajpatch_locomo_judge",
                "input": {"verdict": "CORRECT"},
            }
        ]

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return FakeAnthropicResponse()

    class FakeAnthropicClient:
        def __init__(self):
            self.messages = FakeMessages()

    provider = LiteLLMProvider("claude-3-5-sonnet")
    monkeypatch.setattr(provider, "_get_anthropic_client", lambda: FakeAnthropicClient())

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="judge", turn_index=0)],
        spec=get_structured_task_spec("locomo_judge"),
        metadata={"task": "locomo_judge"},
    )

    assert response.parsed.verdict == "CORRECT"
    assert response.metadata["structured_vendor"] == "anthropic"
    assert response.metadata["structured_strategy"] == "anthropic_tool_schema"
    assert calls[0]["tool_choice"] == {"type": "tool", "name": "trajpatch_locomo_judge"}
    assert calls[0]["tools"][0]["input_schema"]["type"] == "object"


def test_unknown_remote_model_skips_structured():
    provider = LiteLLMProvider("mistral-large")
    assert provider.supports_structured("episodic_extract") is False


def test_metered_structured_failure_is_recorded_before_reraising():
    provider = MeteredLLMProvider(_FailingStructuredProvider(), role="backbone")
    spec = get_structured_task_spec("locomo_judge")

    with pytest.raises(RuntimeError, match="schema rejected"):
        provider.generate_structured(
            [NormalizedMessage(role="user", content="judge", turn_index=0)],
            spec=spec,
            metadata={"task": "locomo_judge"},
        )

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call.task == "locomo_judge"
    assert call.prompt_tokens == 0
    assert call.completion_tokens == 0
    assert call.metadata["structured_success"] is False
    assert call.metadata["structured_failure"] is True
    assert call.metadata["error_type"] == "RuntimeError"
    assert "schema rejected" in call.metadata["error_message"]


def test_metered_structured_success_marks_success_metadata():
    provider = MeteredLLMProvider(_SuccessfulStructuredProvider(), role="judge")

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="judge", turn_index=0)],
        spec=get_structured_task_spec("locomo_judge"),
        metadata={"task": "locomo_judge"},
    )

    assert response.parsed.verdict == "CORRECT"
    assert len(provider.calls) == 1
    assert provider.calls[0].prompt_tokens == 3
    assert provider.calls[0].completion_tokens == 1
    assert provider.calls[0].metadata["structured_success"] is True
    assert provider.calls[0].metadata["structured_failure"] is False
