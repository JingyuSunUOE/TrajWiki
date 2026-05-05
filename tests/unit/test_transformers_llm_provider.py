from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from trajpatch.providers.transformers_provider import TransformersLLMProvider
from trajpatch.types import DevicePlan, NormalizedMessage


class FakeTensor:
    def __init__(self, data) -> None:
        self.data = data

    @property
    def ndim(self) -> int:
        if not self.data or not isinstance(self.data[0], list):
            return 1
        return 2

    @property
    def shape(self) -> tuple[int, ...]:
        if self.ndim == 1:
            return (len(self.data),)
        return (len(self.data), len(self.data[0]))

    def unsqueeze(self, axis: int):
        assert axis == 0
        return FakeTensor([self.data])

    def to(self, _device):
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, column = item
            if isinstance(column, slice):
                return FakeTensor(self.data[row][column])
            return self.data[row][column]
        result = self.data[item]
        if isinstance(result, list):
            return FakeTensor(result)
        return result


class FakeTorchModule:
    @staticmethod
    def tensor(data):
        return FakeTensor(data)

    @staticmethod
    def ones_like(tensor):
        if tensor.ndim == 1:
            return FakeTensor([1 for _ in tensor.data])
        return FakeTensor([[1 for _ in row] for row in tensor.data])

    @staticmethod
    def device(name: str):
        return name


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ fake }}"
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.last_messages = None
        self.last_prompt = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=False):
        self.last_messages = messages
        assert add_generation_prompt is True
        assert tokenize is False
        assert enable_thinking is False
        return "<chat-template-prompt>"

    def __call__(self, prompt, return_tensors):
        self.last_prompt = prompt
        assert prompt == "<chat-template-prompt>"
        assert return_tensors == "pt"
        return {
            "input_ids": FakeTorchModule.tensor([[11, 12, 13]]),
            "attention_mask": FakeTorchModule.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "NO_MEMORY"


class FakeFallbackTokenizer:
    def __init__(self) -> None:
        self.chat_template = None
        self.eos_token_id = 7
        self.pad_token_id = None
        self.last_prompt = None

    def __call__(self, prompt, return_tensors):
        self.last_prompt = prompt
        assert return_tensors == "pt"
        return {
            "input_ids": FakeTorchModule.tensor([[21, 22, 23, 24]]),
            "attention_mask": FakeTorchModule.tensor([[1, 1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "NO_MEMORY"


class FakeBatchTokenizer:
    def __init__(self) -> None:
        self.chat_template = None
        self.eos_token_id = 5
        self.pad_token_id = 0
        self.padding_side = "right"
        self.last_prompts = None

    def __call__(self, prompt, return_tensors, padding=False):
        assert return_tensors == "pt"
        if isinstance(prompt, list):
            assert padding is True
            self.last_prompts = list(prompt)
            return {
                "input_ids": FakeTorchModule.tensor([[11, 12, 13], [21, 22, 0]]),
                "attention_mask": FakeTorchModule.tensor([[1, 1, 1], [1, 1, 0]]),
            }
        self.last_prompts = [prompt]
        return {
            "input_ids": FakeTorchModule.tensor([[11, 12, 13]]),
            "attention_mask": FakeTorchModule.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        values = getattr(token_ids, "data", token_ids)
        return " ".join(str(value) for value in values)


class FakeModel:
    def __init__(self, generated_ids) -> None:
        self.generated_ids = generated_ids
        self.device = "cpu"
        self.config = SimpleNamespace(eos_token_id=9, pad_token_id=None)
        self.generation_config = SimpleNamespace(
            eos_token_id=19,
            pad_token_id=None,
            max_length=20,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
        )
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return self.generated_ids


def _provider_with_fakes(tokenizer, model) -> TransformersLLMProvider:
    provider = TransformersLLMProvider("fake-local-model")
    provider._tokenizer = tokenizer
    provider._model = model
    return provider


def _install_fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorchModule)


def test_local_provider_uses_chat_template_when_available(monkeypatch):
    _install_fake_torch(monkeypatch)
    tokenizer = FakeChatTokenizer()
    model = FakeModel(FakeTorchModule.tensor([[11, 12, 13, 31, 32]]))
    provider = _provider_with_fakes(tokenizer, model)

    response = provider.generate(
        [NormalizedMessage(role="user", content="Question?", turn_index=0)],
        system_prompt="Be structured.",
    )

    assert tokenizer.last_messages == [
        {"role": "system", "content": "Be structured."},
        {"role": "user", "content": "Question?"},
    ]
    assert tokenizer.last_prompt == "<chat-template-prompt>"
    assert response.text == "NO_MEMORY"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2
    assert response.metadata["used_chat_template"] is True
    assert response.metadata["chat_template_fallback"] is False
    assert response.metadata["generation_budget_tokens"] == 256
    assert model.last_generate_kwargs["do_sample"] is False
    assert model.last_generate_kwargs["max_new_tokens"] == 256
    assert model.last_generate_kwargs["pad_token_id"] == 0
    assert model.last_generate_kwargs["eos_token_id"] == 19
    assert "max_length" not in model.last_generate_kwargs
    assert "temperature" not in model.last_generate_kwargs
    assert "top_p" not in model.last_generate_kwargs
    assert "top_k" not in model.last_generate_kwargs


def test_local_provider_falls_back_to_manual_prompt_without_chat_template(monkeypatch):
    _install_fake_torch(monkeypatch)
    tokenizer = FakeFallbackTokenizer()
    model = FakeModel(FakeTorchModule.tensor([[21, 22, 23, 24, 25]]))
    provider = _provider_with_fakes(tokenizer, model)

    with pytest.warns(RuntimeWarning, match="falling back to manual prompt serialization"):
        response = provider.generate(
            [NormalizedMessage(role="user", content="TASK=EPISODIC_EXTRACTION", turn_index=0)],
            system_prompt="Return strict DSL.",
        )

    assert "System: Return strict DSL." in tokenizer.last_prompt
    assert "User: TASK=EPISODIC_EXTRACTION" in tokenizer.last_prompt
    assert tokenizer.last_prompt.endswith("Assistant:")
    assert response.text == "NO_MEMORY"
    assert response.prompt_tokens == 4
    assert response.completion_tokens == 1
    assert response.metadata["used_chat_template"] is False
    assert response.metadata["chat_template_fallback"] is True


def test_local_provider_generate_batch_preserves_order_and_prompt_lengths(monkeypatch):
    _install_fake_torch(monkeypatch)
    tokenizer = FakeBatchTokenizer()
    model = FakeModel(FakeTorchModule.tensor([[11, 12, 13, 31, 32], [21, 22, 0, 41, 42]]))
    provider = _provider_with_fakes(tokenizer, model)

    responses = provider.generate_batch(
        [
            [NormalizedMessage(role="user", content="First prompt", turn_index=0)],
            [NormalizedMessage(role="user", content="Second prompt", turn_index=0)],
        ],
        metadata={"task": "episodic_extract"},
    )

    assert tokenizer.last_prompts is not None
    assert len(tokenizer.last_prompts) == 2
    assert len(responses) == 2
    assert responses[0].text == "31 32"
    assert responses[0].prompt_tokens == 3
    assert responses[0].completion_tokens == 2
    assert responses[0].metadata["batched"] is True
    assert responses[0].metadata["batch_size"] == 2
    assert responses[0].metadata["batch_index"] == 0
    assert responses[0].metadata["generation_budget_tokens"] == 192
    assert "serialized_prompt" not in responses[0].metadata
    assert responses[1].text == "41 42"
    assert responses[1].prompt_tokens == 2
    assert responses[1].completion_tokens == 2
    assert responses[1].metadata["batch_index"] == 1
    assert "serialized_prompt" not in responses[1].metadata
    assert model.last_generate_kwargs["max_new_tokens"] == 192


def test_local_provider_forces_left_padding_on_tokenizer_init(monkeypatch):
    class EnsureTokenizer:
        def __init__(self) -> None:
            self.chat_template = None
            self.eos_token_id = 3
            self.pad_token_id = 0
            self.padding_side = "right"

        def __call__(self, prompt, return_tensors):
            assert return_tensors == "pt"
            return {
                "input_ids": FakeTorchModule.tensor([[11, 12, 13]]),
                "attention_mask": FakeTorchModule.tensor([[1, 1, 1]]),
            }

        def decode(self, token_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return "NO_MEMORY"

    tokenizer = EnsureTokenizer()
    model = FakeModel(FakeTorchModule.tensor([[11, 12, 13, 31]]))

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return model

    _install_fake_torch(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
        ),
    )

    provider = TransformersLLMProvider(
        "fake-local-model",
        device_plan=DevicePlan(device_mode="single", accelerator="cpu"),
    )
    provider.generate(
        [NormalizedMessage(role="user", content="Prompt", turn_index=0)],
        metadata={"task": "episodic_extract"},
    )

    assert tokenizer.padding_side == "left"


def test_generation_kwargs_avoid_conflicting_length_and_sampling_fields():
    tokenizer = FakeChatTokenizer()
    model = FakeModel(FakeTorchModule.tensor([[1, 2, 3, 4]]))
    provider = _provider_with_fakes(tokenizer, model)
    provider._sanitize_model_generation_defaults()

    generation_kwargs = provider._build_generation_kwargs()

    assert generation_kwargs == {
        "do_sample": False,
        "max_new_tokens": 256,
        "pad_token_id": 0,
        "eos_token_id": 19,
    }
    assert model.generation_config.temperature == 1.0
    assert model.generation_config.top_p == 1.0
    assert model.generation_config.top_k == 50
    assert model.generation_config.max_length is None


def test_manual_prompt_preserves_structured_extraction_instructions():
    prompt = TransformersLLMProvider._build_manual_prompt(
        [
            NormalizedMessage(
                role="user",
                content="TASK=EPISODIC_EXTRACTION\nConversation:\nUser: I smoke 5 cigarettes a day.",
                turn_index=0,
            )
        ],
        system_prompt="Return the exact memory DSL.",
    )

    assert "System: Return the exact memory DSL." in prompt
    assert "TASK=EPISODIC_EXTRACTION" in prompt
    assert "Conversation:\nUser: I smoke 5 cigarettes a day." in prompt
    assert prompt.endswith("Assistant:")


@pytest.mark.parametrize(
    ("metadata", "expected_budget"),
    [
        ({"task": "answer_generation"}, 512),
        ({"task": "answer_freeform_generation"}, 512),
        ({"task": "answer_evidence_synthesis"}, 512),
        ({"task": "answer_type_verification"}, 128),
        ({"task": "answer_count_validation"}, 384),
        ({"task": "answer_generation_repair"}, 256),
        ({"task": "episodic_extract"}, 192),
        ({"task": "trajectory_match"}, 64),
        ({"task": "claim_transition_judge"}, 64),
        ({"task": "locomo_judge"}, 96),
        ({"task": "medmt_judge"}, 96),
        ({"task": "semantic_metric_schema"}, 256),
        ({"task": "semantic_metric_extract"}, 256),
        ({"task": "unknown_task"}, 256),
        ({}, 256),
    ],
)
def test_generation_budget_varies_by_task(metadata, expected_budget):
    tokenizer = FakeChatTokenizer()
    model = FakeModel(FakeTorchModule.tensor([[1, 2, 3, 4]]))
    provider = _provider_with_fakes(tokenizer, model)
    provider._sanitize_model_generation_defaults()

    generation_kwargs = provider._build_generation_kwargs(metadata)

    assert generation_kwargs["max_new_tokens"] == expected_budget


@pytest.mark.parametrize(
    ("metadata", "expected_budget"),
    [
        ({"task": "episodic_extract", "repair_round": 1}, 96),
        ({"task": "episodic_extract", "repair_round": 2}, 96),
        ({"task": "trajectory_match", "repair_round": 1}, 64),
    ],
)
def test_generation_budget_uses_repair_caps_for_extraction_repairs(metadata, expected_budget):
    tokenizer = FakeChatTokenizer()
    model = FakeModel(FakeTorchModule.tensor([[1, 2, 3, 4]]))
    provider = _provider_with_fakes(tokenizer, model)
    provider._sanitize_model_generation_defaults()

    generation_kwargs = provider._build_generation_kwargs(metadata)

    assert generation_kwargs["max_new_tokens"] == expected_budget
