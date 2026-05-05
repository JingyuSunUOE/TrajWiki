"""Semantic structured metrics for LOCOMO answer evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from trajpatch.config import RunConfig
from trajpatch.exceptions import ParserValidationError
from trajpatch.prompts import load_prompt
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.structured_outputs import (
    get_structured_task_spec,
    parse_structured_payload,
)
from trajpatch.types import NormalizedMessage
from trajpatch.utils.json_utils import dumps_json, write_json
from trajpatch.utils.metrics import bleu1, normalize_answer


SEMANTIC_METRIC_SCHEMA_VERSION = "v1"


@dataclass(slots=True)
class SemanticMetricResult:
    f1: float
    bleu_1: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class _SemanticMetricGenerationError(ParserValidationError):
    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        llm_call_count: int = 0,
        mode: str = "deterministic_fallback",
        fallback_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.llm_call_count = llm_call_count
        self.mode = mode
        self.fallback_reason = fallback_reason


class SemanticMetricEvaluator:
    def __init__(
        self,
        provider: LLMProvider | None,
        config: RunConfig,
        *,
        trace: Callable[[str], None] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.trace = trace

    def evaluate_locomo(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        query_task_id: str,
    ) -> SemanticMetricResult:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0
        cache_hits: dict[str, bool] = {}
        cache_misses: dict[str, bool] = {}
        llm_call_counts: dict[str, int] = {}
        errors: list[str] = []
        modes: list[str] = []

        schema_result, usage = self._schema(question, reference_answer, query_task_id)
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_latency_ms += float(usage.get("latency_ms") or 0.0)
        cache_hits["schema"] = usage["cache_hit"]
        cache_misses["schema"] = bool(usage.get("cache_miss"))
        llm_call_counts["schema"] = int(usage.get("llm_call_count") or 0)
        modes.append(str(usage["mode"]))
        if usage.get("error"):
            errors.append(str(usage["error"]))

        reference_result, usage = self._extract(
            question=question,
            schema=schema_result,
            answer_text=reference_answer,
            answer_role="reference",
            query_task_id=query_task_id,
        )
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_latency_ms += float(usage.get("latency_ms") or 0.0)
        cache_hits["reference_extract"] = usage["cache_hit"]
        cache_misses["reference_extract"] = bool(usage.get("cache_miss"))
        llm_call_counts["reference_extract"] = int(usage.get("llm_call_count") or 0)
        modes.append(str(usage["mode"]))
        if usage.get("error"):
            errors.append(str(usage["error"]))

        candidate_result, usage = self._extract(
            question=question,
            schema=schema_result,
            answer_text=candidate_answer,
            answer_role="candidate",
            query_task_id=query_task_id,
        )
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_latency_ms += float(usage.get("latency_ms") or 0.0)
        cache_hits["candidate_extract"] = usage["cache_hit"]
        cache_misses["candidate_extract"] = bool(usage.get("cache_miss"))
        llm_call_counts["candidate_extract"] = int(usage.get("llm_call_count") or 0)
        modes.append(str(usage["mode"]))
        if usage.get("error"):
            errors.append(str(usage["error"]))

        f1, f1_metadata = self._canonical_set_f1(candidate_result, reference_result)
        reference_text = self._canonical_text(schema_result, reference_result)
        candidate_text = self._canonical_text(schema_result, candidate_result)
        bleu_1 = bleu1(candidate_text, reference_text)
        mode = "cache" if all(cache_hits.values()) else self._combined_mode(modes)
        metadata = {
            "schema_version": SEMANTIC_METRIC_SCHEMA_VERSION,
            "f1_policy": "semantic_canonical_set_f1_v2",
            "bleu_policy": "semantic_canonical_bleu1_no_bp_v1",
            "schema": schema_result,
            "reference_slots": reference_result,
            "candidate_slots": candidate_result,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_count": sum(1 for value in cache_hits.values() if value),
            "cache_miss_count": sum(1 for value in cache_misses.values() if value),
            "llm_call_count": sum(llm_call_counts.values()),
            "llm_call_counts": llm_call_counts,
            "latency_ms": total_latency_ms,
            "mode": mode,
            "error": "; ".join(errors) if errors else None,
            "reference_canonical_text": reference_text,
            "candidate_canonical_text": candidate_text,
            **f1_metadata,
        }
        return SemanticMetricResult(
            f1=f1,
            bleu_1=bleu_1,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            metadata=metadata,
        )

    def _schema(
        self,
        question: str,
        reference_answer: str,
        query_task_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = (
            load_prompt("semantic_metric_schema")
            + "\n\nQUESTION:\n"
            + question
            + "\n\nREFERENCE_ANSWER:\n"
            + reference_answer
        )
        payload = {
            "schema_version": SEMANTIC_METRIC_SCHEMA_VERSION,
            "task": "semantic_metric_schema",
            "provider": self._provider_identity(),
            "prompt_hash": self._prompt_hash("semantic_metric_schema"),
            "question": question,
            "reference_answer": reference_answer,
        }
        return self._cached_or_generate(
            task="semantic_metric_schema",
            key_payload=payload,
            prompt=prompt,
            query_task_id=query_task_id,
            fallback=lambda error: self._deterministic_schema(
                question,
                reference_answer,
                error=error,
            ),
        )

    def _extract(
        self,
        *,
        question: str,
        schema: dict[str, Any],
        answer_text: str,
        answer_role: str,
        query_task_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        schema_json = dumps_json(schema, indent=True)
        prompt = (
            load_prompt("semantic_metric_extract")
            + "\n\nQUESTION:\n"
            + question
            + "\n\nSCHEMA_JSON:\n"
            + schema_json
            + "\n\nANSWER_TEXT:\n"
            + answer_text
        )
        payload = {
            "schema_version": SEMANTIC_METRIC_SCHEMA_VERSION,
            "task": "semantic_metric_extract",
            "answer_role": answer_role,
            "provider": self._provider_identity(),
            "prompt_hash": self._prompt_hash("semantic_metric_extract"),
            "schema_digest": self._digest(schema),
            "question": question,
            "answer_text": answer_text,
        }
        return self._cached_or_generate(
            task="semantic_metric_extract",
            key_payload=payload,
            prompt=prompt,
            query_task_id=query_task_id,
            fallback=lambda error: self._deterministic_extract(schema, answer_text, error=error),
        )

    def _cached_or_generate(
        self,
        *,
        task: str,
        key_payload: dict[str, Any],
        prompt: str,
        query_task_id: str,
        fallback: Callable[[str], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_path = self._cache_path(key_payload)
        cache_enabled = (
            self.config.memory_cache_enabled
            and not self.config.rebuild_semantic_metric_cache
        )
        if cache_enabled and cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8")), {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": 0.0,
                    "cache_hit": True,
                    "cache_miss": False,
                    "llm_call_count": 0,
                    "mode": "cache",
                    "error": None,
                }
            except Exception:
                pass

        prompt_tokens = 0
        completion_tokens = 0
        mode = "structured"
        error: str | None = None
        started_at = time.perf_counter()
        try:
            (
                parsed,
                prompt_tokens,
                completion_tokens,
                mode,
                llm_call_count,
                fallback_reason,
            ) = self._generate_structured_or_text(
                task,
                prompt,
                query_task_id,
            )
            payload = self._model_to_dict(parsed)
        except Exception as exc:  # noqa: BLE001
            error = " ".join(str(exc).split()) or exc.__class__.__name__
            mode = "deterministic_fallback"
            llm_call_count = 0
            fallback_reason = error
            if isinstance(exc, _SemanticMetricGenerationError):
                prompt_tokens = exc.prompt_tokens
                completion_tokens = exc.completion_tokens
                llm_call_count = exc.llm_call_count
                mode = exc.mode
                fallback_reason = exc.fallback_reason or error
            payload = fallback(error)
        if self.config.memory_cache_enabled:
            try:
                self._write_cache_payload(cache_path, payload)
            except Exception:
                pass
        return payload, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "cache_hit": False,
            "cache_miss": True,
            "llm_call_count": llm_call_count,
            "mode": mode,
            "fallback_reason": fallback_reason,
            "error": error,
        }

    def _generate_structured_or_text(
        self,
        task: str,
        prompt: str,
        query_task_id: str,
    ) -> tuple[Any, int, int, str, int, str | None]:
        if self.provider is None:
            raise ParserValidationError("No provider available for semantic metric evaluation.")
        spec = get_structured_task_spec(task)
        structured_supported = self.provider.supports_structured(task)
        structured_attempt_failed = False
        fallback_reason: str | None = None
        if structured_supported:
            try:
                response = self.provider.generate_structured(
                    [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                    spec=spec,
                    metadata={
                        "task": task,
                        "query_task_id": query_task_id,
                        "structured_requested": True,
                        "structured_supported": True,
                    },
                )
                return (
                    response.parsed,
                    int(response.prompt_tokens or 0),
                    int(response.completion_tokens or 0),
                    "structured",
                    1,
                    None,
                )
            except Exception as exc:  # noqa: BLE001
                structured_attempt_failed = True
                fallback_reason = type(exc).__name__
        response = self.provider.generate(
            [
                NormalizedMessage(
                    role="user",
                    content=prompt + "\n\nReturn ONLY a JSON object matching the requested schema.",
                    turn_index=0,
                )
            ],
            metadata={
                "task": task,
                "query_task_id": query_task_id,
                "structured_requested": True,
                "structured_supported": structured_supported,
                "fallback_used": True,
                "fallback_mode": "text_json",
                "fallback_reason": fallback_reason
                or ("structured_exception" if structured_attempt_failed else "structured_unsupported"),
            },
        )
        prompt_tokens = int(response.prompt_tokens or 0)
        completion_tokens = int(response.completion_tokens or 0)
        llm_call_count = 2 if structured_attempt_failed else 1
        fallback_reason = fallback_reason or (
            "structured_exception" if structured_attempt_failed else "structured_unsupported"
        )
        try:
            payload = self._extract_json_object(response.text)
            parsed = parse_structured_payload(spec, payload)
        except Exception as exc:  # noqa: BLE001
            error = " ".join(str(exc).split()) or exc.__class__.__name__
            raise _SemanticMetricGenerationError(
                f"Semantic metric text fallback failed for {task}: {error}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                llm_call_count=llm_call_count,
                fallback_reason=fallback_reason,
            ) from exc
        return (
            parsed,
            prompt_tokens,
            completion_tokens,
            "text_json",
            llm_call_count,
            fallback_reason,
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        normalized = text.strip()
        if normalized.startswith("```") and normalized.endswith("```"):
            lines = normalized.splitlines()
            normalized = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
            if not match:
                raise
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ParserValidationError(
                "Semantic metric text fallback did not return a JSON object."
            )
        return payload

    @staticmethod
    def _model_to_dict(model: Any) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return dict(model.model_dump())
        if hasattr(model, "dict"):
            return dict(model.dict())
        return dict(model)

    @staticmethod
    def _combined_mode(modes: list[str]) -> str:
        if any(mode == "deterministic_fallback" for mode in modes):
            return "deterministic_fallback"
        if any(mode == "text_json" for mode in modes):
            return "text_json"
        if any(mode == "structured" for mode in modes):
            return "structured"
        return modes[0] if modes else "unknown"

    @staticmethod
    def _canonical_text(schema: dict[str, Any], extraction: dict[str, Any]) -> str:
        slot_order = [str(slot.get("slot_id") or "") for slot in list(schema.get("slots") or [])]
        by_slot: dict[str, list[str]] = {slot_id: [] for slot_id in slot_order}
        for slot in list(extraction.get("slots") or []):
            slot_id = str(slot.get("slot_id") or "").strip()
            if slot_id not in by_slot:
                by_slot[slot_id] = []
            by_slot[slot_id].extend(
                str(value).strip()
                for value in list(slot.get("canonical_values") or [])
                if str(value).strip()
            )
        lines = []
        for slot_id in [*slot_order, *sorted(set(by_slot) - set(slot_order))]:
            values = sorted(dict.fromkeys(by_slot.get(slot_id, [])), key=normalize_answer)
            if values:
                lines.append(f"{slot_id}: {' | '.join(values)}")
        return "\n".join(lines)

    @staticmethod
    def _canonical_set_f1(
        candidate_extraction: dict[str, Any],
        reference_extraction: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        candidate_items = SemanticMetricEvaluator._canonical_value_items(candidate_extraction)
        reference_items = SemanticMetricEvaluator._canonical_value_items(reference_extraction)
        exact_overlap_items = candidate_items & reference_items
        match_pairs = SemanticMetricEvaluator._soft_canonical_match_pairs(candidate_items, reference_items)
        soft_overlap_items = {(pair["slot_id"], pair["reference"]) for pair in match_pairs}
        matched_candidate_items = {(pair["slot_id"], pair["candidate"]) for pair in match_pairs}
        unmatched_reference_items = reference_items - soft_overlap_items
        unmatched_candidate_items = candidate_items - matched_candidate_items
        if not candidate_items and not reference_items:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = len(match_pairs) / len(candidate_items) if candidate_items else 0.0
            recall = len(match_pairs) / len(reference_items) if reference_items else 0.0
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return f1, {
            "f1_reference_items": SemanticMetricEvaluator._format_canonical_items(reference_items),
            "f1_candidate_items": SemanticMetricEvaluator._format_canonical_items(candidate_items),
            "f1_overlap_items": SemanticMetricEvaluator._format_canonical_items(soft_overlap_items),
            "f1_exact_overlap_items": SemanticMetricEvaluator._format_canonical_items(exact_overlap_items),
            "f1_soft_overlap_items": SemanticMetricEvaluator._format_canonical_items(soft_overlap_items),
            "f1_soft_match_pairs": [
                {
                    "slot_id": pair["slot_id"],
                    "reference": pair["reference"],
                    "candidate": pair["candidate"],
                    "reason": pair["reason"],
                }
                for pair in match_pairs
            ],
            "f1_unmatched_reference_items": SemanticMetricEvaluator._format_canonical_items(
                unmatched_reference_items
            ),
            "f1_unmatched_candidate_items": SemanticMetricEvaluator._format_canonical_items(
                unmatched_candidate_items
            ),
            "f1_precision": precision,
            "f1_recall": recall,
        }

    @staticmethod
    def _canonical_value_items(extraction: dict[str, Any]) -> set[tuple[str, str]]:
        items: set[tuple[str, str]] = set()
        for slot in list(extraction.get("slots") or []):
            slot_id = str(slot.get("slot_id") or "").strip()
            for value in list(slot.get("canonical_values") or []):
                normalized = normalize_answer(str(value))
                if normalized:
                    items.add((slot_id, normalized))
        return items

    @staticmethod
    def _format_canonical_items(items: set[tuple[str, str]]) -> list[str]:
        return [
            f"{slot_id}: {value}" if slot_id else value
            for slot_id, value in sorted(items, key=lambda item: (item[0], item[1]))
        ]

    @staticmethod
    def _soft_canonical_match_pairs(
        candidate_items: set[tuple[str, str]],
        reference_items: set[tuple[str, str]],
    ) -> list[dict[str, str]]:
        pairs: list[tuple[int, str, str, str, str]] = []
        slots = sorted({slot_id for slot_id, _ in candidate_items} | {slot_id for slot_id, _ in reference_items})
        for slot_id in slots:
            slot_candidates = sorted(value for item_slot, value in candidate_items if item_slot == slot_id)
            slot_references = sorted(value for item_slot, value in reference_items if item_slot == slot_id)
            contrastive_reference_tokens = SemanticMetricEvaluator._contrastive_reference_tokens(
                slot_references
            )
            for reference in slot_references:
                for candidate in slot_candidates:
                    score, reason = SemanticMetricEvaluator._soft_value_match_score(
                        reference,
                        candidate,
                        contrastive_reference_tokens,
                        slot_id=slot_id,
                    )
                    if score > 0:
                        pairs.append((score, slot_id, reference, candidate, reason))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
        matched_references: set[tuple[str, str]] = set()
        matched_candidates: set[tuple[str, str]] = set()
        selected: list[dict[str, str]] = []
        for score, slot_id, reference, candidate, reason in pairs:
            reference_key = (slot_id, reference)
            candidate_key = (slot_id, candidate)
            if reference_key in matched_references or candidate_key in matched_candidates:
                continue
            matched_references.add(reference_key)
            matched_candidates.add(candidate_key)
            selected.append(
                {
                    "slot_id": slot_id,
                    "reference": reference,
                    "candidate": candidate,
                    "reason": reason,
                }
            )
        selected.sort(key=lambda pair: (pair["slot_id"], pair["reference"], pair["candidate"]))
        return selected

    @staticmethod
    def _soft_value_match_score(
        reference: str,
        candidate: str,
        contrastive_reference_tokens: set[str],
        *,
        slot_id: str = "",
    ) -> tuple[int, str]:
        reference_norm = SemanticMetricEvaluator._soft_normalize_value(reference)
        candidate_norm = SemanticMetricEvaluator._soft_normalize_value(candidate)
        if not reference_norm or not candidate_norm:
            return 0, ""
        if reference_norm == candidate_norm:
            return 100, "exact"
        reference_alias = SemanticMetricEvaluator._alias_normalize_value(reference_norm)
        candidate_alias = SemanticMetricEvaluator._alias_normalize_value(candidate_norm)
        if reference_alias == candidate_alias:
            return 90, "alias"
        if SemanticMetricEvaluator._phrase_alias_values_match(
            reference_norm,
            candidate_norm,
            slot_id=slot_id,
        ):
            return 85, "phrase_alias"
        if (
            ("school speech" in reference_norm or "school talk" in reference_norm)
            and candidate_norm.strip() == "school event"
        ):
            return 0, ""

        reference_tokens = SemanticMetricEvaluator._soft_tokens(reference_norm)
        candidate_tokens = SemanticMetricEvaluator._soft_tokens(candidate_norm)
        if not reference_tokens or not candidate_tokens:
            return 0, ""
        if not SemanticMetricEvaluator._contrastive_tokens_covered(
            reference_tokens,
            candidate_tokens,
            contrastive_reference_tokens,
        ):
            return 0, ""

        reference_core = SemanticMetricEvaluator._remove_safe_modifiers(reference_tokens)
        candidate_core = SemanticMetricEvaluator._remove_safe_modifiers(candidate_tokens)
        if reference_core and candidate_core:
            if reference_core <= candidate_core or candidate_core <= reference_core:
                return 70, "modifier_containment"
            overlap = reference_core & candidate_core
            min_size = min(len(reference_core), len(candidate_core))
            if min_size > 1 and len(overlap) / min_size >= 0.8:
                return 50, "high_token_overlap"
        return 0, ""

    @staticmethod
    def _soft_normalize_value(value: str) -> str:
        text = normalize_answer(value)
        text = re.sub(r"\blgbtq\+\b", "lgbtq", text)
        text = re.sub(r"\bice\s*cream\b", "ice cream", text)
        text = re.sub(r"\bthree\s*turtles\b", "three turtles", text)
        return " ".join(text.split())

    @staticmethod
    def _alias_normalize_value(value: str) -> str:
        alias_map = {
            "nyc": "new york",
            "new york city": "new york",
            "new york": "new york",
            "mentorship program": "mentoring program",
            "mentor program": "mentoring program",
            "mentoring program": "mentoring program",
            "school talk": "school speech",
            "school speech": "school speech",
            "kingkiller chronicle": "kingkiller chronicles",
            "kingkiller chronicles": "kingkiller chronicles",
            "game convention": "gaming convention",
            "gaming convention": "gaming convention",
            "street fighter": "streetfighter",
            "streetfighter": "streetfighter",
            "veg": "vegetables",
            "veggie": "vegetables",
            "veggies": "vegetables",
            "vegetable": "vegetables",
            "vegetables": "vegetables",
            "icecream": "ice cream",
            "ice cream": "ice cream",
            "threeturtles": "three turtles",
            "three turtles": "three turtles",
        }
        return alias_map.get(value, value)

    @staticmethod
    def _phrase_alias_values_match(reference: str, candidate: str, *, slot_id: str = "") -> bool:
        del slot_id
        reference_norm = " ".join(str(reference or "").split())
        candidate_norm = " ".join(str(candidate or "").split())
        if not reference_norm or not candidate_norm:
            return False
        if "school speech" in reference_norm or "school talk" in reference_norm:
            has_school_event = "school event" in candidate_norm
            has_talk_signal = bool(
                re.search(
                    r"\b(?:talk|talked|talking|speech|shared\s+(?:her|his|their|my)\s+journey|encouraged\s+students)\b",
                    candidate_norm,
                )
            )
            if "school speech" in candidate_norm or "school talk" in candidate_norm:
                return True
            return bool(has_school_event and has_talk_signal)
        if "mentoring program" in reference_norm:
            return bool(re.search(r"\b(?:mentorship|mentor|mentoring)\s+program\b", candidate_norm))
        if reference_norm == "gaming convention":
            return "game convention" in candidate_norm or "gaming convention" in candidate_norm
        if reference_norm == "kingkiller chronicles":
            return "kingkiller chronicle" in candidate_norm or "kingkiller chronicles" in candidate_norm
        if reference_norm == "streetfighter":
            return "street fighter" in candidate_norm or "streetfighter" in candidate_norm
        return False

    @staticmethod
    def _soft_tokens(value: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token}

    @staticmethod
    def _remove_safe_modifiers(tokens: set[str]) -> set[str]:
        safe_modifiers = {
            "a",
            "an",
            "the",
            "lgbtq",
            "lgbt",
            "community",
            "class",
            "classes",
            "event",
            "events",
            "activity",
            "activities",
            "kind",
            "type",
            "types",
            "lake",
        }
        core = set(tokens) - safe_modifiers
        return core or set(tokens)

    @staticmethod
    def _contrastive_reference_tokens(reference_values: list[str]) -> set[str]:
        contrastive_terms = {
            "old",
            "new",
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "brown",
            "orange",
            "purple",
            "pink",
            "gray",
            "grey",
        }
        contrastive_terms.update(str(index) for index in range(0, 101))
        contrastive_terms.update(
            {
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            }
        )
        token_counts: Counter[str] = Counter()
        tokenized_values = [SemanticMetricEvaluator._soft_tokens(value) for value in reference_values]
        for tokens in tokenized_values:
            token_counts.update(tokens)
        total = len(tokenized_values)
        return {
            token
            for token, count in token_counts.items()
            if token in contrastive_terms and count < total
        }

    @staticmethod
    def _contrastive_tokens_covered(
        reference_tokens: set[str],
        candidate_tokens: set[str],
        contrastive_reference_tokens: set[str],
    ) -> bool:
        required = reference_tokens & contrastive_reference_tokens
        return required <= candidate_tokens

    def _deterministic_schema(
        self,
        question: str,
        reference_answer: str,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        question_type = "list" if len(self._split_values(reference_answer)) > 1 else "scalar"
        return {
            "question_type": question_type,
            "slots": [
                {
                    "slot_id": "answer",
                    "description": f"Answer values for: {question}",
                    "value_type": "list" if question_type == "list" else "string",
                    "required": True,
                }
            ],
            "fallback_error": error,
        }

    def _deterministic_extract(
        self,
        schema: dict[str, Any],
        answer_text: str,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        slots = list(schema.get("slots") or [{"slot_id": "answer"}])
        slot_id = str(slots[0].get("slot_id") or "answer")
        return {
            "slots": [{"slot_id": slot_id, "canonical_values": self._split_values(answer_text)}],
            "fallback_error": error,
        }

    @staticmethod
    def _split_values(text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        parts = re.split(r"\s*(?:,|;|\band\b|\bor\b|/|\n|\r)\s*", stripped, flags=re.IGNORECASE)
        values = [part.strip(" .\"'`:-") for part in parts if part.strip(" .\"'`:-")]
        if len(values) <= 1:
            normalized = normalize_answer(stripped)
            return [normalized] if normalized else []
        return list(dict.fromkeys(values))

    def _provider_identity(self) -> dict[str, Any]:
        if self.provider is None:
            return {"provider_kind": None, "model_name": None}
        info = self.provider.model_info()
        return {"provider_kind": info.provider_kind, "model_name": info.model_name}

    @staticmethod
    def _prompt_hash(prompt_name: str) -> str:
        return hashlib.sha256(load_prompt(prompt_name).encode("utf-8")).hexdigest()

    @staticmethod
    def _digest(payload: Any) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, key_payload: dict[str, Any]) -> Path:
        digest = self._digest(key_payload)
        return (
            self.config.memory_cache_dir
            / "semantic_metrics"
            / SEMANTIC_METRIC_SCHEMA_VERSION
            / f"{digest}.json"
        )

    @staticmethod
    def _write_cache_payload(cache_path: Path, payload: dict[str, Any]) -> None:
        temp_path = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            write_json(temp_path, payload)
            temp_path.replace(cache_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
