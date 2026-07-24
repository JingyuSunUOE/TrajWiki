"""Remote provider backed by LiteLLM for text and official SDKs for structured tasks."""

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
from .structured_outputs import (
    STRUCTURED_TASKS,
    infer_remote_vendor,
    parse_structured_payload,
    structured_strategy_for_vendor,
    vendor_schema,
)


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


class LiteLLMProvider(LLMProvider):
    def __init__(self, model_name: str) -> None:
        load_runtime_env(override=True)
        self.model_name = model_name
        self.vendor = infer_remote_vendor(model_name)
        self._openai_client = None
        self._gemini_client = None
        self._anthropic_client = None
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
    def _flatten_messages(messages: list[NormalizedMessage], system_prompt: str | None = None) -> str:
        prompt_lines: list[str] = []
        if system_prompt:
            prompt_lines.append(f"SYSTEM:\n{system_prompt}")
        for message in messages:
            prompt_lines.append(f"{message.role.upper()}:\n{message.content}")
        return "\n\n".join(prompt_lines)

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
    def _build_structured_metadata(
        *,
        task: str,
        vendor: str,
        strategy: str,
        schema_name: str,
        success: bool,
        fallback_used: bool,
        fallback_reason: str | None = None,
        refusal: str | None = None,
    ) -> dict[str, Any]:
        return {
            "structured_requested": True,
            "structured_task": task,
            "structured_vendor": vendor,
            "structured_strategy": strategy,
            "structured_schema_name": schema_name,
            "structured_success": success,
            "structured_fallback_used": fallback_used,
            "structured_fallback_reason": fallback_reason,
            "structured_refusal": refusal,
        }

    def _get_openai_client(self):
        if self._openai_client is None:
            with self._client_init_lock:
                if self._openai_client is None:
                    from openai import OpenAI

                    api_key = os.getenv("OPENAI_API_KEY")
                    self._openai_client = OpenAI(api_key=api_key) if api_key else OpenAI()
        return self._openai_client

    def _get_gemini_client(self):
        if self._gemini_client is None:
            with self._client_init_lock:
                if self._gemini_client is None:
                    from google import genai

                    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                    self._gemini_client = genai.Client(api_key=api_key) if api_key else genai.Client()
        return self._gemini_client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            with self._client_init_lock:
                if self._anthropic_client is None:
                    from anthropic import Anthropic

                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    self._anthropic_client = Anthropic(api_key=api_key) if api_key else Anthropic()
        return self._anthropic_client

    def generate(
        self,
        messages: list[NormalizedMessage],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from litellm import completion

        formatted = self._format_messages(messages, system_prompt)
        request_metadata = dict(metadata or {})
        generation_temperature = request_metadata.get("generation_temperature")
        generation_seed = request_metadata.get("generation_seed")
        generation_max_tokens = request_metadata.get("generation_max_tokens")
        request_kwargs: dict[str, Any] = {}
        if generation_temperature is not None:
            request_kwargs["temperature"] = float(generation_temperature)
        if generation_seed is not None:
            request_kwargs["seed"] = int(generation_seed)
            request_kwargs["drop_params"] = True
        if generation_max_tokens is not None:
            request_kwargs["max_tokens"] = int(generation_max_tokens)
        response = completion(model=self.model_name, messages=formatted, **request_kwargs)
        choice = _get_value(response, "choices", default=[{}])[0]
        content = _get_value(choice, "message", "content", default="")
        usage = _get_value(response, "usage", default={}) or {}
        return LLMResponse(
            text=self._extract_text_content(content),
            raw=response,
            prompt_tokens=_get_value(usage, "prompt_tokens"),
            completion_tokens=_get_value(usage, "completion_tokens"),
            metadata={
                "structured_vendor": self.vendor,
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
        return task in STRUCTURED_TASKS and self.vendor in {"openai", "gemini", "anthropic"}

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
                f"Structured outputs are not supported for model {self.model_name}",
                vendor=self.vendor,
                strategy="text_dsl_fallback",
            )

        strategy = structured_strategy_for_vendor(self.vendor)
        try:
            if self.vendor == "openai":
                return self._generate_structured_openai(messages, spec, strategy, system_prompt)
            if self.vendor == "gemini":
                return self._generate_structured_gemini(messages, spec, strategy, system_prompt)
            if self.vendor == "anthropic":
                return self._generate_structured_anthropic(messages, spec, strategy, system_prompt)
        except StructuredOutputError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StructuredOutputError(
                f"Structured output request failed for {self.model_name}: {exc}",
                vendor=self.vendor,
                strategy=strategy,
            ) from exc
        raise StructuredOutputError(
            f"No structured strategy is implemented for vendor {self.vendor}",
            vendor=self.vendor,
            strategy="text_dsl_fallback",
        )

    def _generate_structured_openai(
        self,
        messages: list[NormalizedMessage],
        spec: StructuredTaskSpec,
        strategy: str,
        system_prompt: str | None,
    ) -> StructuredLLMResponse:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=self._format_messages(messages, system_prompt),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": spec.schema_name,
                    "strict": True,
                    "schema": vendor_schema(spec, "openai"),
                },
            },
        )
        message = _get_value(response, "choices", default=[None])[0]
        refusal = _get_value(message, "message", "refusal")
        if refusal:
            raise StructuredOutputError(
                f"OpenAI structured output refusal for {spec.task}",
                vendor="openai",
                strategy=strategy,
                refusal=str(refusal),
                raw=response,
            )
        content = self._extract_text_content(_get_value(message, "message", "content", default=""))
        if not content:
            raise StructuredOutputError(
                f"OpenAI structured output returned empty content for {spec.task}",
                vendor="openai",
                strategy=strategy,
                raw=response,
            )
        payload = json.loads(content)
        parsed = parse_structured_payload(spec, payload)
        usage = _get_value(response, "usage", default={}) or {}
        return StructuredLLMResponse(
            parsed=parsed,
            raw=response,
            prompt_tokens=_get_value(usage, "prompt_tokens"),
            completion_tokens=_get_value(usage, "completion_tokens"),
            metadata=self._build_structured_metadata(
                task=spec.task,
                vendor="openai",
                strategy=strategy,
                schema_name=spec.schema_name,
                success=True,
                fallback_used=False,
                refusal=None,
            ),
        )

    def _generate_structured_gemini(
        self,
        messages: list[NormalizedMessage],
        spec: StructuredTaskSpec,
        strategy: str,
        system_prompt: str | None,
    ) -> StructuredLLMResponse:
        client = self._get_gemini_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=self._flatten_messages(messages, system_prompt),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": vendor_schema(spec, "gemini"),
            },
        )
        text = self._extract_text_content(getattr(response, "text", "") or "")
        if not text:
            raise StructuredOutputError(
                f"Gemini structured output returned empty content for {spec.task}",
                vendor="gemini",
                strategy=strategy,
                raw=response,
            )
        payload = json.loads(text)
        parsed = parse_structured_payload(spec, payload)
        usage = getattr(response, "usage_metadata", None)
        return StructuredLLMResponse(
            parsed=parsed,
            raw=response,
            prompt_tokens=_get_value(usage, "prompt_token_count"),
            completion_tokens=_get_value(usage, "candidates_token_count"),
            metadata=self._build_structured_metadata(
                task=spec.task,
                vendor="gemini",
                strategy=strategy,
                schema_name=spec.schema_name,
                success=True,
                fallback_used=False,
                refusal=None,
            ),
        )

    def _generate_structured_anthropic(
        self,
        messages: list[NormalizedMessage],
        spec: StructuredTaskSpec,
        strategy: str,
        system_prompt: str | None,
    ) -> StructuredLLMResponse:
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=system_prompt or "",
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
                if message.role in {"user", "assistant"}
            ],
            tools=[
                {
                    "name": spec.tool_name,
                    "description": spec.description,
                    "input_schema": vendor_schema(spec, "anthropic"),
                }
            ],
            tool_choice={"type": "tool", "name": spec.tool_name},
        )
        tool_input = None
        text_blocks: list[str] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None) or _get_value(block, "type")
            if block_type == "tool_use" and (
                getattr(block, "name", None) or _get_value(block, "name")
            ) == spec.tool_name:
                tool_input = getattr(block, "input", None) or _get_value(block, "input")
            elif block_type == "text":
                text_blocks.append(str(getattr(block, "text", None) or _get_value(block, "text") or ""))
        if tool_input is None:
            raise StructuredOutputError(
                f"Anthropic structured output did not emit tool_use for {spec.task}",
                vendor="anthropic",
                strategy=strategy,
                raw={
                    "response": response,
                    "text_blocks": text_blocks,
                },
            )
        parsed = parse_structured_payload(spec, tool_input)
        usage = getattr(response, "usage", None)
        return StructuredLLMResponse(
            parsed=parsed,
            raw=response,
            prompt_tokens=_get_value(usage, "input_tokens"),
            completion_tokens=_get_value(usage, "output_tokens"),
            metadata=self._build_structured_metadata(
                task=spec.task,
                vendor="anthropic",
                strategy=strategy,
                schema_name=spec.schema_name,
                success=True,
                fallback_used=False,
                refusal=None,
            ),
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_kind="remote",
            model_name=self.model_name,
            is_remote=True,
            metadata={"vendor": self.vendor},
        )
