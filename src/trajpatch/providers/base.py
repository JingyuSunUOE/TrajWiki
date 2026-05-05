"""Provider interfaces for language models and embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trajpatch.types import LLMResponse, ModelInfo, NormalizedMessage, StructuredLLMResponse, StructuredTaskSpec


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[NormalizedMessage],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def model_info(self) -> ModelInfo:
        raise NotImplementedError

    def generate_batch(
        self,
        batch_messages: list[list[NormalizedMessage]],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LLMResponse]:
        return [
            self.generate(messages, system_prompt=system_prompt, metadata=metadata)
            for messages in batch_messages
        ]

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
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def document_embedding_strategy(self) -> str:
        return "shared_embed"

    def query_embedding_strategy(self) -> str:
        return "shared_embed"

    @abstractmethod
    def model_info(self) -> ModelInfo:
        raise NotImplementedError
