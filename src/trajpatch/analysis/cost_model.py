"""Cost classification and pricing helpers for offline cost-benefit analysis."""

from __future__ import annotations

import math
from typing import Any

from trajpatch.providers.metering import phase_for_task

CostPhase = str
DeploymentScope = str
ReusableScope = str


def _normalized_task(task: str) -> str:
    return str(task or "unknown").strip().lower()


def _is_fallback_or_repair(task: str, metadata: dict[str, Any]) -> bool:
    normalized = _normalized_task(task)
    return bool(
        "repair" in normalized
        or "validation" in normalized
        or metadata.get("repair_requested")
        or metadata.get("fallback_used")
        or metadata.get("structured_fallback_used")
        or metadata.get("answer_repair_raw_text")
        or metadata.get("answer_repair_discarded")
    )


def _is_memory_build_task(task: str) -> bool:
    normalized = _normalized_task(task)
    return bool(
        any(
            marker in normalized
            for marker in (
                "episodic",
                "claim",
                "trajectory_match",
                "trajectory_retrieval_summary",
                "trajectory_summary",
                "retrieval_summary",
                "wiki_page_plan",
                "wiki_page_compile",
                "wiki_seed",
            )
        )
    )


def cost_phase_for_task(task: str, metadata: dict[str, Any] | None = None) -> CostPhase:
    metadata = dict(metadata or {})
    normalized = _normalized_task(task)
    legacy_phase = phase_for_task(normalized)
    if _is_fallback_or_repair(normalized, metadata):
        return "repair_validation"
    if legacy_phase == "memory_build" or _is_memory_build_task(normalized):
        return "memory_build"
    if legacy_phase in {"retrieval"} or normalized in {
        "retrieval_reflection",
        "wiki_page_rerank",
        "trajectory_set_rerank",
    }:
        return "query_time"
    if legacy_phase == "answer" or normalized in {
        "answer_generation",
        "answer_evidence_synthesis",
        "answer_type_verification",
    }:
        return "answer_generation"
    if legacy_phase in {"judge", "semantic_metrics"} or normalized.startswith("semantic_metric"):
        return "evaluation_only"
    return "unknown"


def deployment_scope_for_task(task: str, metadata: dict[str, Any] | None = None) -> DeploymentScope:
    phase = cost_phase_for_task(task, metadata)
    if phase == "evaluation_only":
        return "benchmark_only"
    if dict(metadata or {}).get("benchmark_only") or dict(metadata or {}).get("evaluation_only"):
        return "benchmark_only"
    return "deployment"


def reusable_scope_for_task(task: str, metadata: dict[str, Any] | None = None) -> ReusableScope:
    metadata = dict(metadata or {})
    phase = cost_phase_for_task(task, metadata)
    if deployment_scope_for_task(task, metadata) == "benchmark_only":
        return "benchmark_only"
    if phase == "memory_build" or _is_memory_build_task(task) or metadata.get("memory_type"):
        return "upfront_reusable"
    if phase in {"query_time", "answer_generation", "repair_validation"}:
        return "per_query"
    return "per_query"


def lookup_model_price(model: str, price_config: dict[str, Any] | None) -> dict[str, Any] | None:
    models = dict((price_config or {}).get("models") or {})
    if not models:
        return None
    model_key = str(model or "").strip()
    if model_key in models:
        return dict(models[model_key] or {})
    normalized = model_key.casefold()
    for key, value in models.items():
        if str(key).casefold() == normalized:
            return dict(value or {})
    return None


def estimate_dollar_cost(
    prompt_tokens: int | float | None,
    completion_tokens: int | float | None,
    model: str,
    price_config: dict[str, Any] | None,
) -> dict[str, Any]:
    price = lookup_model_price(model, price_config)
    currency = str((price_config or {}).get("currency") or "USD")
    if not price:
        return {
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
            "currency": currency,
            "price_available": False,
            "usage_available": prompt_tokens is not None and completion_tokens is not None,
        }
    input_price = price.get("input_per_1m_tokens")
    output_price = price.get("output_per_1m_tokens")
    if input_price is None or output_price is None:
        return {
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
            "currency": currency,
            "price_available": False,
            "usage_available": prompt_tokens is not None and completion_tokens is not None,
        }
    if prompt_tokens is None or completion_tokens is None:
        return {
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
            "currency": currency,
            "price_available": True,
            "usage_available": False,
        }
    input_cost = float(prompt_tokens) * float(input_price) / 1_000_000.0
    output_cost = float(completion_tokens) * float(output_price) / 1_000_000.0
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "currency": currency,
        "price_available": True,
        "usage_available": True,
    }


def break_even_queries(upfront_delta: float, per_query_saving: float) -> dict[str, Any]:
    if per_query_saving <= 0:
        return {
            "break_even_queries": None,
            "reason": "no_query_cost_saving",
        }
    return {
        "break_even_queries": max(
            0,
            math.ceil(float(upfront_delta) / float(per_query_saving)),
        ),
        "reason": "positive_query_cost_saving",
    }
