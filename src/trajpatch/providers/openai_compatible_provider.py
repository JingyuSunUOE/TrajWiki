"""OpenAI-compatible provider for local/private structured chat services."""

from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any

from trajpatch.exceptions import StructuredOutputError
from trajpatch.types import (
    LLMResponse,
    ModelInfo,
    NormalizedMessage,
    StructuredLLMResponse,
    StructuredTaskSpec,
)
from trajpatch.utils.env import load_runtime_env

from .base import LLMProvider
from .structured_outputs import STRUCTURED_TASKS, parse_structured_payload, vendor_schema


def _get_value(obj: Any, *keys: str, default: Any = None) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


class OpenAICompatibleProvider(LLMProvider):
    """Provider for vLLM or private OpenAI-compatible chat completion servers."""

    _STRUCTURED_MODES = {"vllm", "openai_json_schema", "text_json"}

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        structured_mode: str = "vllm",
        client: Any | None = None,
    ) -> None:
        load_runtime_env(override=True)
        normalized_mode = str(structured_mode or "vllm").strip().lower().replace("-", "_")
        if normalized_mode not in self._STRUCTURED_MODES:
            raise ValueError(
                "structured_mode must be one of: vllm, openai_json_schema, text_json."
            )
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL") or "http://localhost:8000/v1"
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY") or "EMPTY"
        self.structured_mode = normalized_mode
        self._client = client
        self._client_init_lock = Lock()

    @staticmethod
    def _format_messages(
        messages: list[NormalizedMessage], system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        formatted: list[dict[str, str]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        formatted.extend({"role": message.role, "content": message.content} for message in messages)
        return formatted

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                else:
                    item_type = getattr(item, "type", None)
                    if item_type == "text":
                        parts.append(str(getattr(item, "text", "") or ""))
            return "".join(parts)
        return str(content or "")

    @staticmethod
    def _usage_tokens(response: Any) -> tuple[int | None, int | None]:
        usage = _get_value(response, "usage", default={}) or {}
        return _get_value(usage, "prompt_tokens"), _get_value(usage, "completion_tokens")

    @staticmethod
    def _is_guided_parameter_unsupported(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "structured_outputs",
                "guided_json",
                "extra_body",
                "unknown field",
                "unrecognized",
                "unsupported",
                "not supported",
                "bad request",
                "400",
            )
        )

    def _get_client(self):
        if self._client is None:
            with self._client_init_lock:
                if self._client is None:
                    from openai import OpenAI

                    self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "structured_vendor": "openai-compatible",
            "openai_compatible_base_url": self.base_url,
            "openai_compatible_structured_mode": self.structured_mode,
            "structured_backend": "vllm" if self.structured_mode == "vllm" else self.structured_mode,
        }

    def generate(
        self,
        messages: list[NormalizedMessage],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        request_metadata = dict(metadata or {})
        generation_temperature = request_metadata.get("generation_temperature")
        generation_seed = request_metadata.get("generation_seed")
        generation_max_tokens = request_metadata.get("generation_max_tokens")
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._format_messages(messages, system_prompt),
            "temperature": (
                float(generation_temperature)
                if generation_temperature is not None
                else 0
            ),
        }
        if generation_seed is not None:
            request_kwargs["seed"] = int(generation_seed)
        if generation_max_tokens is not None:
            request_kwargs["max_tokens"] = int(generation_max_tokens)
        response = client.chat.completions.create(
            **request_kwargs,
        )
        choice = _get_value(response, "choices", default=[{}])[0]
        content = _get_value(choice, "message", "content", default="")
        prompt_tokens, completion_tokens = self._usage_tokens(response)
        return LLMResponse(
            text=self._extract_text_content(content),
            raw=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                **self._base_metadata(),
                "requested_model": self.model_name,
                "resolved_model": _get_value(response, "model") or self.model_name,
                "provider_request_id": _get_value(response, "id"),
                "system_fingerprint": _get_value(response, "system_fingerprint"),
                "finish_reason": _get_value(choice, "finish_reason"),
                "generation_temperature_requested": generation_temperature,
                "generation_seed_requested": generation_seed,
                "generation_max_tokens_requested": generation_max_tokens,
            },
        )

    def supports_structured(self, task: str) -> bool:
        return task in STRUCTURED_TASKS and self.structured_mode != "text_json"

    def generate_structured(
        self,
        messages: list[NormalizedMessage],
        *,
        spec: StructuredTaskSpec,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StructuredLLMResponse:
        if not self.supports_structured(spec.task):
            raise StructuredOutputError(
                f"Structured outputs are not enabled for {self.model_name}",
                vendor="openai-compatible",
                strategy="text_json",
            )
        try:
            if self.structured_mode == "openai_json_schema":
                return self._generate_openai_json_schema(messages, spec, system_prompt)
            return self._generate_vllm_guided_json(messages, spec, system_prompt)
        except StructuredOutputError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StructuredOutputError(
                f"OpenAI-compatible structured output request failed for {self.model_name}: {exc}",
                vendor="openai-compatible",
                strategy=self.structured_mode,
            ) from exc

    def _generate_openai_json_schema(
        self,
        messages: list[NormalizedMessage],
        spec: StructuredTaskSpec,
        system_prompt: str | None,
    ) -> StructuredLLMResponse:
        schema = vendor_schema(spec, "openai")
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=self._format_messages(messages, system_prompt),
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": spec.schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        return self._parse_structured_response(
            response,
            spec,
            strategy="openai_json_schema",
            fallback_used=False,
            fallback_reason=None,
        )

    def _generate_vllm_guided_json(
        self,
        messages: list[NormalizedMessage],
        spec: StructuredTaskSpec,
        system_prompt: str | None,
    ) -> StructuredLLMResponse:
        schema = vendor_schema(spec, "openai")
        formatted_messages = self._format_messages(messages, system_prompt)
        # vLLM 0.10.x is the pinned CUDA-12 runtime target for this project and
        # its documented OpenAI-compatible guided JSON API is `guided_json`.
        # Newer vLLM releases prefer `structured_outputs`. Try the stable 0.10.x
        # path first, then fall forward to the newer parameter when needed.
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=0,
                extra_body={"guided_json": schema},
            )
            return self._parse_structured_response(
                response,
                spec,
                strategy="vllm_guided_json",
                fallback_used=False,
                fallback_reason=None,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._is_guided_parameter_unsupported(exc):
                raise StructuredOutputError(
                    f"vLLM guided JSON request failed for {spec.task}: {exc}",
                    vendor="openai-compatible",
                    strategy="vllm_guided_json",
                ) from exc
            first_error = str(exc)
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=0,
                extra_body={"structured_outputs": {"json": schema}},
            )
            return self._parse_structured_response(
                response,
                spec,
                strategy="vllm_structured_outputs_json",
                fallback_used=True,
                fallback_reason=f"guided_json_unsupported: {first_error[:240]}",
            )
        except Exception as exc:  # noqa: BLE001
            raise StructuredOutputError(
                f"vLLM structured_outputs request failed for {spec.task}: {exc}",
                vendor="openai-compatible",
                strategy="vllm_structured_outputs_json",
            ) from exc

    def _parse_structured_response(
        self,
        response: Any,
        spec: StructuredTaskSpec,
        *,
        strategy: str,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> StructuredLLMResponse:
        choice = _get_value(response, "choices", default=[None])[0]
        refusal = _get_value(choice, "message", "refusal")
        if refusal:
            raise StructuredOutputError(
                f"OpenAI-compatible structured output refusal for {spec.task}",
                vendor="openai-compatible",
                strategy=strategy,
                refusal=str(refusal),
                raw=response,
            )
        content = self._extract_text_content(_get_value(choice, "message", "content", default=""))
        if not content:
            raise StructuredOutputError(
                f"OpenAI-compatible structured output returned empty content for {spec.task}",
                vendor="openai-compatible",
                strategy=strategy,
                raw=response,
            )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            preview = content[:240].replace("\n", "\\n")
            raise StructuredOutputError(
                f"OpenAI-compatible structured output returned non-JSON content for "
                f"{spec.task}: {exc}; preview={preview!r}",
                vendor="openai-compatible",
                strategy=strategy,
                raw=response,
            ) from exc
        parsed = parse_structured_payload(spec, payload)
        prompt_tokens, completion_tokens = self._usage_tokens(response)
        return StructuredLLMResponse(
            parsed=parsed,
            raw=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                **self._base_metadata(),
                "structured_requested": True,
                "structured_task": spec.task,
                "structured_strategy": strategy,
                "structured_schema_name": spec.schema_name,
                "structured_success": True,
                "structured_fallback_used": fallback_used,
                "structured_fallback_reason": fallback_reason,
                "structured_refusal": None,
            },
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_kind="openai-compatible",
            model_name=self.model_name,
            is_remote=True,
            metadata={
                "vendor": "openai-compatible",
                "base_url": self.base_url,
                "structured_mode": self.structured_mode,
                "structured_backend": "vllm"
                if self.structured_mode == "vllm"
                else self.structured_mode,
            },
        )
