"""Local Hugging Face Transformers providers for generation and embeddings."""

from __future__ import annotations

import os
import warnings
from threading import Lock
from typing import Any

import numpy as np

from trajpatch.types import (
    DevicePlan,
    LLMResponse,
    ModelInfo,
    NormalizedMessage,
    StructuredLLMResponse,
    StructuredTaskSpec,
)
from trajpatch.utils.env import load_runtime_env

from .base import EmbeddingProvider, LLMProvider
from .devices import build_device_plan, build_embedding_device_plan, infer_parameter_size_b, resolve_hf_model_name


class TransformersLLMProvider(LLMProvider):
    _TASK_MAX_NEW_TOKENS: dict[str, int] = {
        "answer_generation": 512,
        "answer_freeform_generation": 512,
        "answer_evidence_synthesis": 512,
        "answer_type_verification": 128,
        "answer_count_validation": 384,
        "answer_generation_repair": 256,
        "episodic_extract": 192,
        "episodic_claim_text_extract": 192,
        "claim_signal_extract": 256,
        "trajectory_match": 64,
        "claim_transition_judge": 64,
        "trajectory_retrieval_summary": 256,
        "wiki_page_plan": 256,
        "wiki_page_compile": 384,
        "wiki_page_rerank": 128,
        "trajectory_set_rerank": 128,
        "locomo_judge": 96,
        "medmt_judge": 96,
        "semantic_metric_schema": 256,
        "semantic_metric_extract": 256,
    }
    _REPAIR_MAX_NEW_TOKENS: dict[str, int] = {
        "episodic_extract": 96,
    }
    _DEFAULT_MAX_NEW_TOKENS = 256

    def __init__(self, model_name: str, device_mode: str = "auto", device_plan: DevicePlan | None = None) -> None:
        load_runtime_env(override=True)
        self.model_name = model_name
        self.device_plan = device_plan or build_device_plan(model_name, device_mode)
        self._model = None
        self._tokenizer = None
        self._chat_template_fallback_warned = False

    def _ensure_model_and_tokenizer(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_runtime_env(override=True)
        hf_token = os.getenv("HF_TOKEN")
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        # Decoder-only batch generation expects left padding; right padding can
        # change outputs and trigger Hugging Face correctness warnings.
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        self._tokenizer = tokenizer
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            model_kwargs["token"] = hf_token
        if self.device_plan.device_map:
            model_kwargs["device_map"] = self.device_plan.device_map
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if not self.device_plan.device_map:
            if self.device_plan.accelerator.startswith("cuda"):
                model = model.to(self.device_plan.accelerator)
        self._model = model
        self._sanitize_model_generation_defaults()

    @staticmethod
    def _build_chat_messages(
        messages: list[NormalizedMessage], system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        chat_messages: list[dict[str, str]] = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            chat_messages.append({"role": message.role, "content": message.content})
        return chat_messages

    @staticmethod
    def _build_manual_prompt(messages: list[NormalizedMessage], system_prompt: str | None = None) -> str:
        prompt_lines: list[str] = []
        if system_prompt:
            prompt_lines.append(f"System: {system_prompt}")
        for message in messages:
            prompt_lines.append(f"{message.role.title()}: {message.content}")
        prompt_lines.append("Assistant:")
        return "\n".join(prompt_lines)

    def _warn_chat_template_fallback(self, reason: str) -> None:
        if self._chat_template_fallback_warned:
            return
        warnings.warn(
            "Tokenizer chat template unavailable for local model "
            f"{self.model_name}; falling back to manual prompt serialization. Reason: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        self._chat_template_fallback_warned = True

    def _serialize_prompt(
        self, messages: list[NormalizedMessage], system_prompt: str | None = None
    ) -> dict[str, Any]:
        assert self._tokenizer is not None
        tokenizer = self._tokenizer
        chat_messages = self._build_chat_messages(messages, system_prompt)
        if getattr(tokenizer, "chat_template", None):
            try:
                prompt = self._build_chat_template_text(tokenizer, chat_messages)
                return {
                    "used_chat_template": True,
                    "chat_template_fallback": False,
                    "serialized_prompt": prompt,
                }
            except Exception as exc:  # noqa: BLE001
                self._warn_chat_template_fallback(self._chat_template_exception_reason(exc))
        else:
            self._warn_chat_template_fallback("tokenizer has no chat_template")
        return {
            "used_chat_template": False,
            "chat_template_fallback": True,
            "serialized_prompt": self._build_manual_prompt(messages, system_prompt),
        }

    def _resolve_input_device(self):
        assert self._model is not None
        import torch

        model_device = getattr(self._model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device
        hf_device_map = getattr(self._model, "hf_device_map", None) or {}
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str) and mapped_device not in {"disk", "cpu", "meta"}:
                return torch.device(mapped_device)
        if self.device_plan.accelerator.startswith("cuda"):
            return torch.device(self.device_plan.accelerator)
        return torch.device("cpu")

    @staticmethod
    def _chat_template_exception_reason(exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return f"{type(exc).__name__}: {message}"
        return type(exc).__name__

    @staticmethod
    def _build_chat_template_text(tokenizer, chat_messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        try:
            return tokenizer.apply_chat_template(chat_messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(chat_messages, **kwargs)

    def _sanitize_model_generation_defaults(self) -> None:
        assert self._model is not None
        assert self._tokenizer is not None

        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is None:
            return

        default_like_values = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "typical_p": 1.0,
            "min_p": None,
            "penalty_alpha": None,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "max_length": None,
        }
        for field_name, default_value in default_like_values.items():
            if hasattr(generation_config, field_name):
                setattr(generation_config, field_name, default_value)

        if getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = self._first_defined(
                getattr(self._tokenizer, "pad_token_id", None),
                getattr(self._tokenizer, "eos_token_id", None),
                getattr(self._model.config, "pad_token_id", None),
                getattr(self._model.config, "eos_token_id", None),
            )
        if getattr(generation_config, "eos_token_id", None) is None:
            generation_config.eos_token_id = self._first_defined(
                getattr(self._tokenizer, "eos_token_id", None),
                getattr(self._model.config, "eos_token_id", None),
            )

    def _build_generation_inputs(
        self, messages: list[NormalizedMessage], system_prompt: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self._tokenizer is not None
        import torch

        tokenizer = self._tokenizer
        input_device = self._resolve_input_device()
        prompt_metadata = self._serialize_prompt(messages, system_prompt)
        prompt = str(prompt_metadata["serialized_prompt"])
        rendered = tokenizer(prompt, return_tensors="pt")
        input_ids = rendered["input_ids"]
        attention_mask = rendered.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return (
            {
                "input_ids": input_ids.to(input_device),
                "attention_mask": attention_mask.to(input_device),
            },
            {
                **prompt_metadata,
                "prompt_token_count": int(input_ids.shape[-1]),
            },
        )

    @staticmethod
    def _sequence_length(mask_row) -> int:
        if hasattr(mask_row, "sum"):
            try:
                total = mask_row.sum()
                if hasattr(total, "item"):
                    return int(total.item())
                return int(total)
            except Exception:  # noqa: BLE001
                pass
        values = getattr(mask_row, "data", mask_row)
        if not isinstance(values, list):
            values = list(values)
        return int(sum(int(value) for value in values))

    def _build_batch_generation_inputs(
        self,
        batch_messages: list[list[NormalizedMessage]],
        system_prompt: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert self._tokenizer is not None
        import torch

        tokenizer = self._tokenizer
        input_device = self._resolve_input_device()
        prompt_metadata = [self._serialize_prompt(messages, system_prompt) for messages in batch_messages]
        prompts = [str(item["serialized_prompt"]) for item in prompt_metadata]
        rendered = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = rendered["input_ids"]
        attention_mask = rendered.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        for index, item in enumerate(prompt_metadata):
            item["prompt_token_count"] = self._sequence_length(attention_mask[index])
        return (
            {
                "input_ids": input_ids.to(input_device),
                "attention_mask": attention_mask.to(input_device),
            },
            prompt_metadata,
        )

    @staticmethod
    def _first_defined(*values):
        for value in values:
            if value is not None:
                return value
        return None

    @classmethod
    def _resolve_generation_budget(cls, metadata: dict[str, Any] | None = None) -> int:
        metadata = metadata or {}
        explicit_budget = metadata.get("generation_max_tokens")
        if explicit_budget is not None:
            return max(1, int(explicit_budget))
        task = str(metadata.get("task") or "").strip()
        repair_round = metadata.get("repair_round")
        if repair_round is not None and task in cls._REPAIR_MAX_NEW_TOKENS:
            return cls._REPAIR_MAX_NEW_TOKENS[task]
        return cls._TASK_MAX_NEW_TOKENS.get(task, cls._DEFAULT_MAX_NEW_TOKENS)

    def _build_generation_kwargs(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._model is not None
        assert self._tokenizer is not None

        inherited = getattr(self._model, "generation_config", None)
        eos_token_id = self._first_defined(
            getattr(inherited, "eos_token_id", None),
            getattr(self._tokenizer, "eos_token_id", None),
            getattr(self._model.config, "eos_token_id", None),
        )
        pad_token_id = self._first_defined(
            getattr(inherited, "pad_token_id", None),
            getattr(self._tokenizer, "pad_token_id", None),
            eos_token_id,
            getattr(self._model.config, "pad_token_id", None),
            getattr(self._model.config, "eos_token_id", None),
        )

        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": self._resolve_generation_budget(metadata),
        }
        requested_temperature = (metadata or {}).get("generation_temperature")
        if requested_temperature is not None and float(requested_temperature) > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = float(requested_temperature)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        return generation_kwargs

    @staticmethod
    def _decode_generated_tokens(tokenizer, generated_ids, prompt_length: int, row_index: int = 0) -> tuple[str, int]:
        generated_only = generated_ids[row_index, prompt_length:]
        completion_tokens = int(generated_only.shape[-1]) if generated_only.ndim > 0 else 0
        if completion_tokens == 0:
            return "", 0
        return tokenizer.decode(generated_only, skip_special_tokens=True).strip(), completion_tokens

    def generate(
        self,
        messages: list[NormalizedMessage],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._ensure_model_and_tokenizer()
        assert self._model is not None
        assert self._tokenizer is not None
        generation_inputs, prompt_metadata = self._build_generation_inputs(messages, system_prompt)
        generation_kwargs = self._build_generation_kwargs(metadata)
        generation_seed = (metadata or {}).get("generation_seed")
        if generation_seed is not None:
            import torch

            torch.manual_seed(int(generation_seed))
        generation_budget_tokens = int(generation_kwargs["max_new_tokens"])
        generated_ids = self._model.generate(
            input_ids=generation_inputs["input_ids"],
            attention_mask=generation_inputs["attention_mask"],
            **generation_kwargs,
        )
        prompt_tokens = int(prompt_metadata["prompt_token_count"])
        answer, completion_tokens = self._decode_generated_tokens(
            self._tokenizer, generated_ids, prompt_tokens
        )
        return LLMResponse(
            text=answer,
            raw={
                "device_plan": self.device_plan.metadata,
                "used_chat_template": prompt_metadata["used_chat_template"],
                "chat_template_fallback": prompt_metadata["chat_template_fallback"],
                "generation_kwargs": generation_kwargs,
                "generation_budget_tokens": generation_budget_tokens,
                "generation_temperature_requested": (metadata or {}).get(
                    "generation_temperature"
                ),
                "generation_seed_requested": generation_seed,
                "generation_max_tokens_requested": (metadata or {}).get(
                    "generation_max_tokens"
                ),
            },
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                "estimated_usage": True,
                "device_plan": self.device_plan.metadata,
                "used_chat_template": prompt_metadata["used_chat_template"],
                "chat_template_fallback": prompt_metadata["chat_template_fallback"],
                "generation_kwargs": generation_kwargs,
                "generation_budget_tokens": generation_budget_tokens,
                "generation_temperature_requested": (metadata or {}).get(
                    "generation_temperature"
                ),
                "generation_seed_requested": generation_seed,
                "generation_max_tokens_requested": (metadata or {}).get(
                    "generation_max_tokens"
                ),
            },
        )

    def generate_batch(
        self,
        batch_messages: list[list[NormalizedMessage]],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LLMResponse]:
        if not batch_messages:
            return []
        self._ensure_model_and_tokenizer()
        assert self._model is not None
        assert self._tokenizer is not None
        generation_inputs, prompt_metadata = self._build_batch_generation_inputs(batch_messages, system_prompt)
        generation_kwargs = self._build_generation_kwargs(metadata)
        generation_seed = (metadata or {}).get("generation_seed")
        if generation_seed is not None:
            import torch

            torch.manual_seed(int(generation_seed))
        generation_budget_tokens = int(generation_kwargs["max_new_tokens"])
        generated_ids = self._model.generate(
            input_ids=generation_inputs["input_ids"],
            attention_mask=generation_inputs["attention_mask"],
            **generation_kwargs,
        )
        batch_size = len(batch_messages)
        padded_prompt_tokens = int(generation_inputs["input_ids"].shape[-1])
        responses: list[LLMResponse] = []
        for index, item in enumerate(prompt_metadata):
            prompt_tokens = int(item["prompt_token_count"])
            answer, completion_tokens = self._decode_generated_tokens(
                self._tokenizer,
                generated_ids,
                padded_prompt_tokens,
                row_index=index,
            )
            responses.append(
                LLMResponse(
                    text=answer,
                    raw={
                        "device_plan": self.device_plan.metadata,
                        "used_chat_template": item["used_chat_template"],
                        "chat_template_fallback": item["chat_template_fallback"],
                        "generation_kwargs": generation_kwargs,
                        "generation_budget_tokens": generation_budget_tokens,
                        "generation_temperature_requested": (metadata or {}).get(
                            "generation_temperature"
                        ),
                        "generation_seed_requested": generation_seed,
                        "generation_max_tokens_requested": (metadata or {}).get(
                            "generation_max_tokens"
                        ),
                        "batched": True,
                        "batch_size": batch_size,
                        "batch_index": index,
                    },
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    metadata={
                        "estimated_usage": True,
                        "device_plan": self.device_plan.metadata,
                        "used_chat_template": item["used_chat_template"],
                        "chat_template_fallback": item["chat_template_fallback"],
                        "generation_kwargs": generation_kwargs,
                        "generation_budget_tokens": generation_budget_tokens,
                        "generation_temperature_requested": (metadata or {}).get(
                            "generation_temperature"
                        ),
                        "generation_seed_requested": generation_seed,
                        "generation_max_tokens_requested": (metadata or {}).get(
                            "generation_max_tokens"
                        ),
                        "batched": True,
                        "batch_size": batch_size,
                        "batch_index": index,
                    },
                )
            )
        return responses

    def supports_structured(self, task: str) -> bool:
        return False

    def generate_structured(
        self,
        messages: list[NormalizedMessage],
        *,
        spec: StructuredTaskSpec,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StructuredLLMResponse:
        raise NotImplementedError("Local TransformersLLMProvider does not implement structured outputs.")

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_kind="local",
            model_name=self.model_name,
            parameter_billions=infer_parameter_size_b(self.model_name),
            is_remote=False,
            metadata={"device_plan": self.device_plan.metadata},
        )


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    _MODEL_LOAD_LOCK = Lock()
    _MODEL_CACHE: dict[tuple[str, str, str, str], Any] = {}
    _MODEL_RUNTIME_LOCKS: dict[tuple[str, str, str, str], Lock] = {}
    _QWEN_QUERY_FALLBACK_PROMPT = "Instruct: Retrieve memories that directly answer the user's question.\nQuery: "
    _DTYPE_MISMATCH_MARKERS = (
        "expected mat1 and mat2 to have the same dtype",
        "mat1 and mat2 must have the same dtype",
        "c10::BFloat16",
        "bfloat16",
    )

    def __init__(
        self, model_name: str, device_mode: str = "auto", device_plan: DevicePlan | None = None
    ) -> None:
        load_runtime_env(override=True)
        self.model_name = model_name
        self.resolved_model_name = resolve_hf_model_name(model_name)
        self.device_plan = device_plan or build_embedding_device_plan(model_name, device_mode)
        self._model = None
        self._model_cache_key_value: tuple[str, str, str, str] | None = None
        self._model_runtime_lock = Lock()
        self._document_strategy = "plain_encode"
        self._query_strategy = "plain_encode"
        self._autocast_dtype_retry_count = 0
        self._float32_dtype_retry_count = 0

    def _preferred_cuda_torch_dtype(self) -> tuple[object | None, str]:
        if not str(self.device_plan.accelerator).startswith("cuda"):
            return None, "default"
        try:
            import torch
        except ModuleNotFoundError:
            return None, "default"
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16, "bfloat16"
        except Exception:  # noqa: BLE001
            pass
        return torch.float16, "float16"

    def _sentence_transformer_kwargs(self, hf_token: str | None) -> tuple[dict[str, Any], str]:
        sentence_transformer_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device": self.device_plan.accelerator,
        }
        if hf_token:
            sentence_transformer_kwargs["token"] = hf_token
        torch_dtype, dtype_name = self._preferred_cuda_torch_dtype()
        if torch_dtype is not None:
            sentence_transformer_kwargs["model_kwargs"] = {"torch_dtype": torch_dtype}
        self.device_plan.metadata["requested_torch_dtype"] = dtype_name
        return sentence_transformer_kwargs, dtype_name

    @staticmethod
    def _is_sentence_transformer_kwargs_unsupported(exc: TypeError) -> bool:
        message = str(exc)
        markers = ("model_kwargs", "torch_dtype", "unexpected keyword argument")
        return any(marker in message for marker in markers)

    def _ensure_model(self):
        if self._model is None:
            # Hugging Face / SentenceTransformer large-model loading uses shared
            # cache files and meta tensors internally. Loading several copies in
            # parallel worker threads can leave partially materialized meta
            # tensors and fail when SentenceTransformer moves the module to CUDA.
            # Workers in this runner are threads in one process, so identical
            # embedding models on the same device can safely share one materialized
            # module; encode calls are serialized by the per-model runtime lock.
            with self._MODEL_LOAD_LOCK:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    load_runtime_env(override=True)
                    hf_token = os.getenv("HF_TOKEN")
                    sentence_transformer_kwargs, dtype_name = self._sentence_transformer_kwargs(
                        hf_token
                    )
                    cache_key = (
                        self.resolved_model_name,
                        str(self.device_plan.accelerator),
                        dtype_name,
                        hf_token or "",
                    )
                    cached_model = self._MODEL_CACHE.get(cache_key)
                    if cached_model is None:
                        try:
                            cached_model = SentenceTransformer(
                                self.resolved_model_name, **sentence_transformer_kwargs
                            )
                        except TypeError as exc:
                            if (
                                "model_kwargs" not in sentence_transformer_kwargs
                                or not self._is_sentence_transformer_kwargs_unsupported(exc)
                            ):
                                raise
                            sentence_transformer_kwargs = {
                                key: value
                                for key, value in sentence_transformer_kwargs.items()
                                if key != "model_kwargs"
                            }
                            self.device_plan.metadata["requested_torch_dtype_fallback"] = "default"
                            cached_model = SentenceTransformer(
                                self.resolved_model_name, **sentence_transformer_kwargs
                            )
                        self._MODEL_CACHE[cache_key] = cached_model
                    self._model = cached_model
                    self._model_cache_key_value = cache_key
                    self._model_runtime_lock = self._MODEL_RUNTIME_LOCKS.setdefault(
                        cache_key, Lock()
                    )
        return self._model

    def _is_qwen_embedding(self) -> bool:
        return "qwen3-embedding" in self.resolved_model_name.lower()

    def _to_vectors(self, values) -> list[list[float]]:
        return [np.asarray(vector, dtype=float).tolist() for vector in values]

    @classmethod
    def _is_dtype_mismatch_error(cls, exc: RuntimeError) -> bool:
        message = str(exc)
        return "mat1" in message and any(marker in message for marker in cls._DTYPE_MISMATCH_MARKERS)

    def _coerce_model_to_float32(self) -> None:
        try:
            import torch

            float32_dtype = torch.float32
        except ModuleNotFoundError:
            # Unit tests can exercise this recovery path without installing
            # torch. Real SentenceTransformer usage requires torch and will use
            # the dtype object above.
            float32_dtype = "float32"
        model = self._ensure_model()
        if hasattr(model, "to"):
            try:
                model.to(dtype=float32_dtype)
            except TypeError:
                model.to(float32_dtype)
        elif hasattr(model, "float"):
            model.float()
        else:
            raise RuntimeError(
                "Embedding model raised a dtype mismatch but does not support float32 coercion."
            )
        self._float32_dtype_retry_count += 1

    @staticmethod
    def _first_parameter_device_and_dtype(model) -> tuple[object | None, object | None]:
        parameters = getattr(model, "parameters", None)
        if parameters is None:
            return None, None
        try:
            parameter = next(parameters())
        except Exception:  # noqa: BLE001
            return None, None
        return getattr(parameter, "device", None), getattr(parameter, "dtype", None)

    def _try_encode_with_autocast(
        self,
        texts: list[str],
        *,
        strategy: str,
        **kwargs,
    ) -> tuple[Any, str] | None:
        try:
            import torch
        except ModuleNotFoundError:
            return None

        model = self._ensure_model()
        device, dtype = self._first_parameter_device_and_dtype(model)
        if dtype not in {torch.bfloat16, torch.float16}:
            return None
        device_type = getattr(device, "type", None) or str(device).split(":", 1)[0]
        if device_type not in {"cuda", "cpu"}:
            return None
        try:
            with torch.autocast(device_type=device_type, dtype=dtype):
                vectors = model.encode(texts, normalize_embeddings=True, **kwargs)
        except RuntimeError:
            return None
        self._autocast_dtype_retry_count += 1
        dtype_name = "bf16" if dtype is torch.bfloat16 else "fp16"
        return vectors, f"{strategy}_{dtype_name}_autocast_retry"

    def _encode_with_dtype_recovery(
        self,
        texts: list[str],
        *,
        strategy: str,
        **kwargs,
    ) -> tuple[Any, str]:
        model = self._ensure_model()
        with self._model_runtime_lock:
            try:
                return model.encode(texts, normalize_embeddings=True, **kwargs), strategy
            except RuntimeError as exc:
                if not self._is_dtype_mismatch_error(exc):
                    raise
                autocast_result = self._try_encode_with_autocast(texts, strategy=strategy, **kwargs)
                if autocast_result is not None:
                    return autocast_result
                self._coerce_model_to_float32()
                return model.encode(
                    texts, normalize_embeddings=True, **kwargs
                ), f"{strategy}_float32_retry"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors, strategy = self._encode_with_dtype_recovery(texts, strategy="plain_encode")
        self._document_strategy = strategy
        return self._to_vectors(vectors)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if not self._is_qwen_embedding():
            vectors, strategy = self._encode_with_dtype_recovery(texts, strategy="plain_encode")
            self._query_strategy = strategy
            return self._to_vectors(vectors)
        try:
            vectors, strategy = self._encode_with_dtype_recovery(
                texts,
                strategy="qwen_prompt_name_query",
                prompt_name="query",
            )
            self._query_strategy = strategy
            return self._to_vectors(vectors)
        except (TypeError, ValueError):
            prefixed = [f"{self._QWEN_QUERY_FALLBACK_PROMPT}{text}" for text in texts]
            vectors, strategy = self._encode_with_dtype_recovery(
                prefixed,
                strategy="qwen_prefixed_query_fallback",
            )
            self._query_strategy = strategy
            return self._to_vectors(vectors)

    def document_embedding_strategy(self) -> str:
        return self._document_strategy

    def query_embedding_strategy(self) -> str:
        return self._query_strategy

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_kind="local",
            model_name=self.resolved_model_name,
            parameter_billions=infer_parameter_size_b(self.resolved_model_name),
            is_remote=False,
            metadata={
                "device_plan": self.device_plan.metadata,
                "shared_model_cache": True,
                "autocast_dtype_retry_count": self._autocast_dtype_retry_count,
                "float32_dtype_retry_count": self._float32_dtype_retry_count,
            },
        )
