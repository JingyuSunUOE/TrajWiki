"""Deterministic mock providers used for testing and dry runs."""

from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from typing import Any, Callable

from trajpatch.types import LLMResponse, ModelInfo, NormalizedMessage, StructuredLLMResponse, StructuredTaskSpec

from .base import EmbeddingProvider, LLMProvider


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        scripted_responses: list[str] | None = None,
        callback: Callable[[list[NormalizedMessage], str | None, dict[str, Any] | None], str] | None = None,
        model_name: str = "mock-backbone",
    ) -> None:
        self._responses = deque(scripted_responses or [])
        self._callback = callback
        self._model_name = model_name
        self.has_callback = callback is not None

    def generate(
        self,
        messages: list[NormalizedMessage],
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._callback is not None:
            text = self._callback(messages, system_prompt, metadata)
        elif self._responses:
            text = self._responses.popleft()
        else:
            text = self._default_response(messages, metadata or {})
        prompt_tokens = sum(len(message.content.split()) for message in messages)
        completion_tokens = len(text.split())
        return LLMResponse(
            text=text,
            raw={"mock": True},
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={"estimated_usage": True},
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="mock", model_name=self._model_name, is_remote=False)

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
        raise NotImplementedError("MockLLMProvider does not implement structured outputs.")

    def _default_response(
        self, messages: list[NormalizedMessage], metadata: dict[str, Any] | None = None
    ) -> str:
        metadata = metadata or {}
        task = metadata.get("task")
        prompt = messages[-1].content if messages else ""
        if task == "episodic_extract":
            exchange_lines = [
                line
                for line in prompt.splitlines()
                if "]:" in line or ("[user]:" in line or "[assistant]:" in line)
            ]
            summary = " ".join(line.split("]:", 1)[-1].strip() for line in exchange_lines[-2:])[:180]
            return (
                f"SUMMARY_CONTENT: {summary or 'Conversation exchange.'}\n"
                f"CONTEXT: {summary or 'Conversation exchange.'}\n"
                "KEYWORDS: conversation, exchange"
            )
        if task == "episodic_claim_text_extract":
            raw_ids = re.findall(r"([A-Za-z0-9\-]+-m\d{4})", prompt)
            exchange_lines = [
                line
                for line in prompt.splitlines()
                if "]:" in line or ("[user]:" in line or "[assistant]:" in line)
            ]
            text = " ".join(line.split("]:", 1)[-1].strip() for line in exchange_lines[-2:])[:180]
            links = ", ".join(raw_ids[-2:]) if raw_ids else "none"
            inventory = re.search(r"\binclude[s]?\s+([^.!?]+)", prompt, flags=re.IGNORECASE)
            if inventory:
                subject = "Melanie" if "Melanie" in prompt else "The user"
                items = [
                    item.strip(" .,:;!?")
                    for item in re.split(r"\s*(?:,|\band\b)\s*", inventory.group(1))
                    if item.strip(" .,:;!?")
                ]
                claim_lines = "\n".join(
                    f"- status=active | source_message_ids={links} | supporting_quote={inventory.group(0)} | text={subject} participates in {item}."
                    for item in items
                )
                return f"HAS_CLAIMS: true\nREASON: mock inventory extraction\n\n[CLAIMS]\n{claim_lines}"
            return (
                "HAS_CLAIMS: true\n"
                "REASON: mock claim text extraction\n\n"
                "[CLAIMS]\n"
                f"- status=active | source_message_ids={links} | supporting_quote={text or 'Conversation exchange.'} | text={text or 'Conversation exchange.'}"
            )
        if task == "claim_signal_extract":
            return (
                "[EXACT_TERMS]\n"
                "- none\n\n"
                "[FACETS]\n"
                "- none\n\n"
                "[DISPLAY_ITEMS]\n"
                "- none\n\n"
                "[DISPLAY_NAMED_ENTITIES]\n"
                "- none\n\n"
                "[DISPLAY_COUNTS]\n"
                "- none\n\n"
                "[DISPLAY_KEY_FACTS]\n"
                "- none"
            )
        if task == "trajectory_match":
            return "DECISION: NEW\nSELECTED_CANDIDATE: none\nRATIONALE: No candidate clearly fits."
        if task == "claim_transition_judge":
            return "DECISION: ADD\nSELECTED_CANDIDATE: none\nRATIONALE: The claim is treated as a new addition."
        if task == "trajectory_retrieval_summary":
            return (
                "## Profile / Stable Facts\n"
                "- Mock retrieval summary.\n\n"
                "## Item Sets / Named Entities\n"
                "- None recorded.\n\n"
                "## Relations / Temporal Updates\n"
                "- None recorded.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None recorded."
            )
        if task == "wiki_page_plan":
            return (
                "## Pages\n"
                "- page_type=index | title=Index | slug=index | trajectories=none | entities=none | links=none\n"
                "- page_type=topic | title=General Topic | slug=general-topic | trajectories=none | entities=none | links=index"
            )
        if task == "wiki_page_compile":
            return (
                "## Overview\n"
                "- Mock wiki page.\n\n"
                "## Key Facts\n"
                "- Included for routing.\n\n"
                "## Items / Counts\n"
                "- None.\n\n"
                "## Linked Trajectories\n"
                "- None.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None."
            )
        if task in {"wiki_page_rerank", "trajectory_set_rerank"}:
            final_count = int(metadata.get("final_count", 1) or 1)
            labels = re.findall(r"^### ([A-Z]\d+)$", prompt, flags=re.MULTILINE)
            selected = labels[:final_count]
            rationale_lines = "\n".join(f"- {label}: Mock rerank keeps the current order." for label in selected)
            return f"SELECTED: {', '.join(selected)}\nRATIONALES:\n{rationale_lines}"
        if task == "retrieval_reflection":
            return (
                '{"rewritten_query":"mock retrieval retry","answer_type":"unknown",'
                '"target_entities":[],"event_terms":[],"temporal_terms":[],'
                '"must_find_terms":[],"candidate_page_slugs":[],"raw_search_terms":[],'
                '"rationale":"Mock reflection retry."}'
            )
        if task in {"answer_generation", "answer_generation_repair"}:
            question_match = re.search(r"Question:\n(.+)", prompt, re.DOTALL)
            question = question_match.group(1).strip() if question_match else "the query"
            return f"Mock answer based on retrieved memory for: {question}"
        if task in {"judge", "locomo_judge", "medmt_judge"}:
            return "CORRECT"
        return "NO_MEMORY"


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "hash-embedding", dimensions: int = 32) -> None:
        self._model_name = model_name
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = []
            for index in range(self._dimensions):
                byte = digest[index % len(digest)]
                raw.append((byte / 255.0) - 0.5)
            norm = math.sqrt(sum(value * value for value in raw)) or 1.0
            vectors.append([value / norm for value in raw])
        return vectors

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="mock", model_name=self._model_name, is_remote=False)
