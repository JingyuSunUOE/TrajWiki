from __future__ import annotations

import json

import pytest

from trajpatch.config import RunConfig
from trajpatch.exceptions import ProviderConfigurationError, StructuredOutputError
from trajpatch.providers.factory import build_llm_provider
from trajpatch.providers.openai_compatible_provider import OpenAICompatibleProvider
from trajpatch.providers.structured_outputs import STRUCTURED_TASKS, get_structured_task_spec
from trajpatch.types import NormalizedMessage


class _FakeCompletions:
    def __init__(self, responses=None, failures=None) -> None:
        self.calls: list[dict] = []
        self.responses = list(responses or [])
        self.failures = list(failures or [])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        if self.responses:
            return self.responses.pop(0)
        return {
            "choices": [{"message": {"content": "plain answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def _episodic_response(summary: str = "User discussed memory."):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "has_memory": True,
                            "memory": {
                                "summary_content": summary,
                                "context": "The exchange contains a durable memory.",
                                "keywords": ["memory"],
                            },
                            "reason": None,
                        }
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def test_openai_compatible_supports_all_registered_structured_tasks():
    provider = OpenAICompatibleProvider("qwen3-8b", client=_FakeClient(_FakeCompletions()))

    assert all(provider.supports_structured(task) for task in STRUCTURED_TASKS)


def test_openai_compatible_generate_calls_chat_completions_and_records_usage():
    completions = _FakeCompletions()
    provider = OpenAICompatibleProvider(
        "qwen3-8b",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        client=_FakeClient(completions),
    )

    response = provider.generate([NormalizedMessage(role="user", content="hello", turn_index=0)])

    assert response.text == "plain answer"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2
    assert completions.calls[0]["model"] == "qwen3-8b"
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert completions.calls[0]["temperature"] == 0
    assert response.metadata["structured_vendor"] == "openai-compatible"
    assert response.metadata["openai_compatible_base_url"] == "http://localhost:8000/v1"


def test_openai_compatible_generate_forwards_experiment_generation_controls():
    completions = _FakeCompletions()
    provider = OpenAICompatibleProvider(
        "qwen3-8b",
        client=_FakeClient(completions),
    )

    response = provider.generate(
        [NormalizedMessage(role="user", content="hello", turn_index=0)],
        metadata={
            "generation_temperature": 0.25,
            "generation_seed": 7,
            "generation_max_tokens": 512,
        },
    )

    assert completions.calls[0]["temperature"] == 0.25
    assert completions.calls[0]["seed"] == 7
    assert completions.calls[0]["max_tokens"] == 512
    assert response.metadata["generation_temperature_requested"] == 0.25
    assert response.metadata["generation_seed_requested"] == 7
    assert response.metadata["generation_max_tokens_requested"] == 512


def test_openai_compatible_vllm_guided_json_request_parses_payload():
    completions = _FakeCompletions(responses=[_episodic_response()])
    provider = OpenAICompatibleProvider("qwen3-8b", client=_FakeClient(completions))

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="extract", turn_index=0)],
        spec=get_structured_task_spec("episodic_extract"),
    )

    assert response.parsed.has_memory is True
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7
    assert completions.calls[0]["extra_body"]["guided_json"]["type"] == "object"
    assert completions.calls[0]["temperature"] == 0
    assert response.metadata["structured_strategy"] == "vllm_guided_json"
    assert response.metadata["structured_success"] is True


def test_openai_compatible_vllm_falls_back_to_structured_outputs_when_guided_json_unsupported():
    completions = _FakeCompletions(
        responses=[_episodic_response("Fallback worked.")],
        failures=[RuntimeError("400 unsupported guided_json parameter"), None],
    )
    provider = OpenAICompatibleProvider("qwen3-8b", client=_FakeClient(completions))

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="extract", turn_index=0)],
        spec=get_structured_task_spec("episodic_extract"),
    )

    assert len(completions.calls) == 2
    assert "guided_json" in completions.calls[0]["extra_body"]
    assert "structured_outputs" in completions.calls[1]["extra_body"]
    assert response.metadata["structured_strategy"] == "vllm_structured_outputs_json"
    assert response.metadata["structured_fallback_used"] is True


def test_openai_compatible_vllm_raises_structured_error_when_guided_modes_fail():
    completions = _FakeCompletions(
        failures=[
            RuntimeError("400 unsupported guided_json parameter"),
            RuntimeError("400 unsupported structured_outputs parameter"),
        ],
    )
    provider = OpenAICompatibleProvider("qwen3-8b", client=_FakeClient(completions))

    with pytest.raises(StructuredOutputError) as exc_info:
        provider.generate_structured(
            [NormalizedMessage(role="user", content="extract", turn_index=0)],
            spec=get_structured_task_spec("episodic_extract"),
        )

    assert exc_info.value.vendor == "openai-compatible"
    assert exc_info.value.strategy == "vllm_structured_outputs_json"


def test_openai_compatible_openai_json_schema_mode_uses_response_format():
    completions = _FakeCompletions(responses=[_episodic_response()])
    provider = OpenAICompatibleProvider(
        "schema-model",
        structured_mode="openai_json_schema",
        client=_FakeClient(completions),
    )

    response = provider.generate_structured(
        [NormalizedMessage(role="user", content="extract", turn_index=0)],
        spec=get_structured_task_spec("episodic_extract"),
    )

    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    assert completions.calls[0]["temperature"] == 0
    assert completions.calls[0]["response_format"]["json_schema"]["name"] == "trajpatch_episodic_extract"
    assert response.metadata["structured_strategy"] == "openai_json_schema"


def test_openai_compatible_structured_non_json_error_includes_preview():
    completions = _FakeCompletions(
        responses=[
            {
                "choices": [{"message": {"content": "<think>not json</think>"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        ]
    )
    provider = OpenAICompatibleProvider("qwen3-8b", client=_FakeClient(completions))

    with pytest.raises(StructuredOutputError) as exc_info:
        provider.generate_structured(
            [NormalizedMessage(role="user", content="extract", turn_index=0)],
            spec=get_structured_task_spec("episodic_extract"),
        )

    assert "non-JSON content" in str(exc_info.value)
    assert "<think>not json" in str(exc_info.value)


def test_openai_compatible_text_json_mode_does_not_claim_native_structured():
    provider = OpenAICompatibleProvider(
        "qwen3-8b",
        structured_mode="text_json",
        client=_FakeClient(_FakeCompletions()),
    )

    assert provider.supports_structured("episodic_extract") is False
    with pytest.raises(StructuredOutputError):
        provider.generate_structured(
            [NormalizedMessage(role="user", content="extract", turn_index=0)],
            spec=get_structured_task_spec("episodic_extract"),
        )


def test_openai_compatible_factory_builds_provider_without_local_device_restriction(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="openai-compatible",
        judge_provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        judge_model="qwen3-8b",
        embedding_model="hash-embedding",
        conv_workers=4,
        openai_compatible_base_url="http://localhost:8000/v1",
        openai_compatible_api_key="EMPTY",
    )

    provider = build_llm_provider(config, role="backbone", metered=False)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model_info().provider_kind == "openai-compatible"
    assert provider.model_info().is_remote is True


def test_factory_rejects_unknown_provider_kind(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        provider_kind="mock",
        embedding_model="hash-embedding",
    )

    with pytest.raises(ProviderConfigurationError):
        build_llm_provider(config, role="backbone", provider_kind="unsupported", metered=False)  # type: ignore[arg-type]
