"""Fallback and repair diagnostics derived from compact run metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from trajpatch.providers.metering import phase_for_task


QUALITY_RISK_ACTIONS = {
    "deterministic_fallback",
    "discard_repair",
    "schema_validation_failed",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator or 1.0)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _is_quality_risk_event(event: dict[str, Any]) -> bool:
    return (
        str(event.get("action")) in QUALITY_RISK_ACTIONS
        or str(event.get("outcome")) in {"discarded", "failed"}
    )


def _model_for_phase(phase: str, run_meta: dict[str, Any] | None) -> str:
    run_meta = run_meta or {}
    if phase in {"judge", "semantic_metrics"}:
        return str(run_meta.get("judge_model") or "unknown")
    return str(run_meta.get("backbone_model") or "unknown")


def _role_for_phase(phase: str) -> str:
    return "judge" if phase in {"judge", "semantic_metrics"} else "backbone"


def _trigger_for_fallback(category: str) -> str:
    normalized = str(category or "").lower()
    if "unsupported" in normalized:
        return "structured_unsupported"
    if "parse" in normalized and "json" in normalized:
        return "text_json_parse_failed"
    if "dsl" in normalized:
        return "dsl_parse_failed"
    if "deterministic" in normalized:
        return "deterministic_fallback"
    if "schema" in normalized:
        return "schema_validation_failed"
    if "provider" in normalized or "error" in normalized:
        return "provider_error"
    return "structured_parse_failed"


def _action_for_fallback(category: str) -> str:
    normalized = str(category or "").lower()
    if "deterministic" in normalized:
        return "deterministic_fallback"
    if "dsl" in normalized:
        return "text_dsl_fallback"
    if "json" in normalized or "unsupported" in normalized or "structured" in normalized:
        return "text_json_fallback"
    return "text_json_fallback"


def _infer_fallback_task(category: str, row: dict[str, Any]) -> str:
    metadata = dict(row.get("metadata") or {})
    answer_metadata = dict(metadata.get("answer_metadata") or {})
    judge_metadata = dict(metadata.get("judge_metadata") or {})
    semantic_metadata = dict(metadata.get("semantic_metrics") or {})
    normalized = str(category or "").lower()
    if str(answer_metadata.get("answer_synthesis_mode") or "").lower() in {
        "text_json",
        "deterministic_fallback",
    }:
        return "answer_evidence_synthesis"
    if str(semantic_metadata.get("mode") or "").lower() in {"text_json", "deterministic_fallback"}:
        return "semantic_metric_extract"
    if judge_metadata.get("structured_fallback_used") or judge_metadata.get("structured_supported") is False:
        return str(judge_metadata.get("structured_task") or judge_metadata.get("task") or "benchmark_judge")
    if "deterministic" in normalized:
        return "unknown_deterministic_fallback"
    return "unknown_fallback"


def _task_usage(row: dict[str, Any], task: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    return dict(dict(metadata.get("llm_usage_by_task") or {}).get(task) or {})


def _event_cost_from_usage(usage: dict[str, Any], weight: float) -> dict[str, float]:
    fallback_count = _as_float(usage.get("fallback_count") or usage.get("repair_count"), 0.0)
    divisor = fallback_count if fallback_count > 0 else max(weight, 1.0)
    return {
        "provider_call_delta": _as_float(usage.get("provider_call_count")) * weight / divisor,
        "prompt_tokens_delta": _as_float(usage.get("prompt_tokens")) * weight / divisor,
        "completion_tokens_delta": _as_float(usage.get("completion_tokens")) * weight / divisor,
        "latency_ms_delta": _as_float(usage.get("latency_ms")) * weight / divisor,
    }


def _base_event(
    *,
    row: dict[str, Any],
    run_meta: dict[str, Any] | None,
    event_index: int,
    event_kind: str,
    task: str,
    trigger: str,
    action: str,
    outcome: str,
    event_weight: float,
    cost: dict[str, float] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    structured_supported: Any = None,
    structured_requested: Any = None,
) -> dict[str, Any]:
    phase = phase_for_task(task)
    cost = cost or {}
    event = {
        "schema_version": "fallback_repair_event_v1",
        "event_id": f"{row.get('query_task_id')}-fr-{event_index:03d}",
        "sample_id": row.get("sample_id"),
        "query_task_id": row.get("query_task_id"),
        "phase": phase,
        "task": task,
        "provider_role": _role_for_phase(phase),
        "model": _model_for_phase(phase, run_meta),
        "event_kind": event_kind,
        "trigger": trigger,
        "action": action,
        "outcome": outcome,
        "event_weight": float(event_weight),
        "provider_call_delta": float(cost.get("provider_call_delta") or 0.0),
        "prompt_tokens_delta": float(cost.get("prompt_tokens_delta") or 0.0),
        "completion_tokens_delta": float(cost.get("completion_tokens_delta") or 0.0),
        "latency_ms_delta": float(cost.get("latency_ms_delta") or 0.0),
        "error_type": error_type,
        "error_message": " ".join(str(error_message or "").split())[:240] if error_message else None,
        "structured_supported": structured_supported,
        "structured_requested": structured_requested,
    }
    return event


def build_fallback_repair_events_for_row(
    row: dict[str, Any],
    *,
    run_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build compact fallback/repair/cache events for one evaluated query row."""

    metadata = dict(row.get("metadata") or {})
    answer_metadata = dict(metadata.get("answer_metadata") or {})
    judge_metadata = dict(metadata.get("judge_metadata") or {})
    semantic_metadata = dict(metadata.get("semantic_metrics") or {})
    fallback_counts = dict(metadata.get("llm_fallback_counts") or {})
    repair_counts = dict(metadata.get("llm_repair_counts") or {})
    events: list[dict[str, Any]] = []
    event_index = 1

    for category, raw_weight in sorted(fallback_counts.items()):
        weight = _as_float(raw_weight)
        if weight <= 0:
            continue
        task = _infer_fallback_task(str(category), row)
        usage = _task_usage(row, task)
        trigger = _trigger_for_fallback(str(category))
        action = _action_for_fallback(str(category))
        error_type = None
        error_message = None
        structured_supported = None
        structured_requested = None
        if phase_for_task(task) == "judge":
            error_type = judge_metadata.get("judge_error_type")
            error_message = judge_metadata.get("judge_error_message")
            structured_supported = judge_metadata.get("structured_supported")
            structured_requested = judge_metadata.get("structured_requested")
        events.append(
            _base_event(
                row=row,
                run_meta=run_meta,
                event_index=event_index,
                event_kind="fallback",
                task=task,
                trigger=trigger,
                action=action,
                outcome="success",
                event_weight=weight,
                cost=_event_cost_from_usage(usage, weight),
                error_type=error_type,
                error_message=error_message,
                structured_supported=structured_supported,
                structured_requested=structured_requested,
            )
        )
        event_index += 1

    if answer_metadata.get("answer_repair_attempted") and not any(repair_counts.values()):
        repair_counts["locomo_answer_repair"] = 1
    if answer_metadata.get("count_validation_llm_used") and "answer_count_validation" not in repair_counts:
        repair_counts["answer_count_validation"] = 1
    if answer_metadata.get("answer_type_verification_used") and "answer_type_verification" not in repair_counts:
        repair_counts["answer_type_verification"] = 1
    for task, raw_weight in sorted(repair_counts.items()):
        weight = _as_float(raw_weight)
        if weight <= 0:
            continue
        usage = _task_usage(row, str(task))
        cost = _event_cost_from_usage(usage, weight)
        if str(task) == "answer_count_validation":
            outcome = "success" if answer_metadata.get("count_validation_llm_success") else "failed"
            events.append(
                _base_event(
                    row=row,
                    run_meta=run_meta,
                    event_index=event_index,
                    event_kind="repair",
                    task=str(task),
                    trigger="count_validation_uncertain",
                    action="llm_count_validation",
                    outcome=outcome,
                    event_weight=weight,
                    cost=cost,
                    error_message=answer_metadata.get("count_validation_llm_error"),
                )
            )
            event_index += 1
            continue
        if str(task) == "answer_type_verification":
            outcome = "success" if answer_metadata.get("answer_type_verification_success") is not False else "failed"
            events.append(
                _base_event(
                    row=row,
                    run_meta=run_meta,
                    event_index=event_index,
                    event_kind="repair",
                    task=str(task),
                    trigger="answer_type_mismatch",
                    action="answer_type_verification",
                    outcome=outcome,
                    event_weight=weight,
                    cost=cost,
                    error_message=answer_metadata.get("answer_type_verification_error"),
                )
            )
            event_index += 1
            continue
        if str(task) == "locomo_answer_repair":
            cost = {
                "provider_call_delta": 1.0 if answer_metadata.get("answer_repair_attempted") else 0.0,
                "prompt_tokens_delta": _as_float(answer_metadata.get("answer_repair_prompt_tokens")),
                "completion_tokens_delta": _as_float(answer_metadata.get("answer_repair_completion_tokens")),
                "latency_ms_delta": _as_float(answer_metadata.get("answer_repair_latency_ms")),
            }
        discarded = bool(answer_metadata.get("answer_repair_discarded"))
        used = bool(answer_metadata.get("answer_repair_used"))
        outcome = "discarded" if discarded else "success" if used else "failed"
        action = "discard_repair" if discarded else "use_repair" if used else "llm_repair"
        trigger = (
            "invalid_answer_format"
            if answer_metadata.get("answer_repair_json_extracted") or discarded
            else "postcheck_issue"
        )
        events.append(
            _base_event(
                row=row,
                run_meta=run_meta,
                event_index=event_index,
                event_kind="repair",
                task=str(task),
                trigger=trigger,
                action=action,
                outcome=outcome,
                event_weight=weight,
                cost=cost,
                error_type=answer_metadata.get("answer_repair_error_type"),
                error_message=answer_metadata.get("answer_repair_error_message")
                or answer_metadata.get("answer_repair_discard_reason"),
            )
        )
        event_index += 1

    cache_hits = dict(semantic_metadata.get("cache_hits") or {})
    cache_misses = dict(semantic_metadata.get("cache_misses") or {})
    for step, hit in sorted(cache_hits.items()):
        if not hit:
            continue
        events.append(
            _base_event(
                row=row,
                run_meta=run_meta,
                event_index=event_index,
                event_kind="cache",
                task=f"semantic_metric_{step}",
                trigger="cache_hit",
                action="cache_hit",
                outcome="success",
                event_weight=1.0,
            )
        )
        event_index += 1
    for step, miss in sorted(cache_misses.items()):
        if not miss:
            continue
        events.append(
            _base_event(
                row=row,
                run_meta=run_meta,
                event_index=event_index,
                event_kind="cache",
                task=f"semantic_metric_{step}",
                trigger="cache_miss",
                action="cache_miss",
                outcome="success",
                event_weight=1.0,
            )
        )
        event_index += 1

    return events


def summarize_events_for_query(events: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts = Counter(str(event.get("event_kind") or "unknown") for event in events)
    action_counts = Counter(str(event.get("action") or "unknown") for event in events)
    trigger_counts = Counter(str(event.get("trigger") or "unknown") for event in events)
    extra_cost = {
        "provider_call_delta": sum(_as_float(event.get("provider_call_delta")) for event in events),
        "prompt_tokens_delta": sum(_as_float(event.get("prompt_tokens_delta")) for event in events),
        "completion_tokens_delta": sum(_as_float(event.get("completion_tokens_delta")) for event in events),
        "total_tokens_delta": sum(
            _as_float(event.get("prompt_tokens_delta")) + _as_float(event.get("completion_tokens_delta"))
            for event in events
        ),
        "latency_ms_delta": sum(_as_float(event.get("latency_ms_delta")) for event in events),
    }
    quality_risk_count = sum(1 for event in events if _is_quality_risk_event(event))
    quality_risk_weighted_count = sum(
        _as_float(event.get("event_weight"), 1.0)
        for event in events
        if _is_quality_risk_event(event)
    )
    return {
        "event_count": len(events),
        "event_counts": dict(event_counts),
        "action_counts": dict(action_counts),
        "trigger_counts": dict(trigger_counts),
        "extra_cost": extra_cost,
        "quality_flags": {
            "has_fallback": bool(event_counts.get("fallback")),
            "has_repair": bool(event_counts.get("repair")),
            "has_discarded_repair": any(str(event.get("outcome")) == "discarded" for event in events),
            "has_deterministic_fallback": bool(action_counts.get("deterministic_fallback")),
            "quality_risk_event_count": quality_risk_count,
            "quality_risk_weighted_event_count": quality_risk_weighted_count,
        },
        "event_ids": [str(event.get("event_id")) for event in events],
    }


def summarize_fallback_repair_events(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def group_summary(group_events: list[dict[str, Any]]) -> dict[str, Any]:
        fallback_weight = sum(_as_float(event.get("event_weight"), 1.0) for event in group_events if event.get("event_kind") == "fallback")
        repair_weight = sum(_as_float(event.get("event_weight"), 1.0) for event in group_events if event.get("event_kind") == "repair")
        repair_discarded = sum(
            _as_float(event.get("event_weight"), 1.0)
            for event in group_events
            if event.get("event_kind") == "repair" and event.get("outcome") == "discarded"
        )
        return {
            "event_count": len(group_events),
            "weighted_event_count": sum(_as_float(event.get("event_weight"), 1.0) for event in group_events),
            "fallback_event_count": fallback_weight,
            "repair_event_count": repair_weight,
            "cache_event_count": sum(_as_float(event.get("event_weight"), 1.0) for event in group_events if event.get("event_kind") == "cache"),
            "extra_provider_call_count": sum(_as_float(event.get("provider_call_delta")) for event in group_events),
            "extra_prompt_tokens": sum(_as_float(event.get("prompt_tokens_delta")) for event in group_events),
            "extra_completion_tokens": sum(_as_float(event.get("completion_tokens_delta")) for event in group_events),
            "extra_total_tokens": sum(
                _as_float(event.get("prompt_tokens_delta")) + _as_float(event.get("completion_tokens_delta"))
                for event in group_events
            ),
            "extra_latency_ms": sum(_as_float(event.get("latency_ms_delta")) for event in group_events),
            "repair_discarded_count": repair_discarded,
            "repair_discard_rate": _safe_rate(repair_discarded, repair_weight),
            "quality_risk_event_count": sum(1 for event in group_events if _is_quality_risk_event(event)),
            "quality_risk_weighted_event_count": sum(
                _as_float(event.get("event_weight"), 1.0)
                for event in group_events
                if _is_quality_risk_event(event)
            ),
        }

    by_phase_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_role_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_phase_events[str(event.get("phase") or "unknown")].append(event)
        by_task_events[str(event.get("task") or "unknown")].append(event)
        by_role_events[str(event.get("provider_role") or "unknown")].append(event)
        by_model_events[str(event.get("model") or "unknown")].append(event)

    chains = Counter()
    for query_key in sorted({(event.get("sample_id"), event.get("query_task_id")) for event in events}):
        query_events = [
            event
            for event in events
            if (event.get("sample_id"), event.get("query_task_id")) == query_key
            and event.get("event_kind") in {"fallback", "repair"}
        ]
        if not query_events:
            continue
        chain = "->".join(str(event.get("action") or event.get("trigger") or "event") for event in query_events)
        chains[chain] += 1

    event_query_ids = {(event.get("sample_id"), event.get("query_task_id")) for event in events}
    fallback_query_ids = {
        (event.get("sample_id"), event.get("query_task_id"))
        for event in events
        if event.get("event_kind") == "fallback"
    }
    repair_query_ids = {
        (event.get("sample_id"), event.get("query_task_id"))
        for event in events
        if event.get("event_kind") == "repair"
    }
    with_fallback_scores: list[float] = []
    without_fallback_scores: list[float] = []
    with_repair_scores: list[float] = []
    without_repair_scores: list[float] = []
    verdict_with_fallback = Counter()
    verdict_without_fallback = Counter()
    verdict_with_repair = Counter()
    verdict_without_repair = Counter()
    for row in rows:
        key = (row.get("sample_id"), row.get("query_task_id"))
        verdict = str(row.get("judge_verdict") or "unknown").lower()
        score = row.get("judge_score")
        if score is None:
            score = dict(row.get("metrics") or {}).get("judge_acc")
        score_value = None if score is None else _as_float(score)
        if key in fallback_query_ids:
            verdict_with_fallback[verdict] += 1
            if score_value is not None:
                with_fallback_scores.append(score_value)
        else:
            verdict_without_fallback[verdict] += 1
            if score_value is not None:
                without_fallback_scores.append(score_value)
        if key in repair_query_ids:
            verdict_with_repair[verdict] += 1
            if score_value is not None:
                with_repair_scores.append(score_value)
        else:
            verdict_without_repair[verdict] += 1
            if score_value is not None:
                without_repair_scores.append(score_value)

    overall = group_summary(events)
    total_provider_calls = sum(
        _as_float(dict(dict(row.get("metadata") or {}).get("llm_usage") or {}).get("provider_call_count"))
        for row in rows
    )
    total_tokens = sum(_as_float(dict(row.get("tokens") or {}).get("total_tokens")) for row in rows)
    paper_ready = {
        "structured_unsupported_rate_by_task": {
            task: _safe_rate(
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("trigger") == "structured_unsupported"),
                len(task_events),
            )
            for task, task_events in sorted(by_task_events.items())
        },
        "fallback_success_rate_by_task": {
            task: _safe_rate(
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("event_kind") == "fallback" and event.get("outcome") == "success"),
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("event_kind") == "fallback"),
            )
            for task, task_events in sorted(by_task_events.items())
        },
        "repair_attempt_rate_by_task": {
            task: _safe_rate(
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("event_kind") == "repair"),
                overall["weighted_event_count"],
            )
            for task, task_events in sorted(by_task_events.items())
        },
        "repair_discard_rate_by_task": {
            task: _safe_rate(
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("event_kind") == "repair" and event.get("outcome") == "discarded"),
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("event_kind") == "repair"),
            )
            for task, task_events in sorted(by_task_events.items())
        },
        "deterministic_fallback_rate_by_task": {
            task: _safe_rate(
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("action") == "deterministic_fallback"),
                sum(_as_float(event.get("event_weight"), 1.0) for event in task_events if event.get("event_kind") == "fallback"),
            )
            for task, task_events in sorted(by_task_events.items())
        },
        "extra_llm_call_rate": _safe_rate(overall["extra_provider_call_count"], total_provider_calls),
        "extra_token_overhead_rate": _safe_rate(overall["extra_total_tokens"], total_tokens),
        "mean_judge_score_with_fallback": _safe_mean(with_fallback_scores),
        "mean_judge_score_without_fallback": _safe_mean(without_fallback_scores),
        "incorrect_rate_with_repair": _safe_rate(verdict_with_repair["incorrect"], sum(verdict_with_repair.values())),
        "incorrect_rate_without_repair": _safe_rate(verdict_without_repair["incorrect"], sum(verdict_without_repair.values())),
    }

    return {
        "diagnostic_mode": "events_v1",
        "overall": overall,
        "by_phase": {key: group_summary(value) for key, value in sorted(by_phase_events.items())},
        "by_task": {key: group_summary(value) for key, value in sorted(by_task_events.items())},
        "by_provider_role": {key: group_summary(value) for key, value in sorted(by_role_events.items())},
        "by_model": {key: group_summary(value) for key, value in sorted(by_model_events.items())},
        "fallback_chains": dict(chains.most_common(20)),
        "quality_correlation": {
            "query_count": len(rows),
            "query_with_any_event_count": len(event_query_ids),
            "fallback_query_count": len(fallback_query_ids),
            "repair_query_count": len(repair_query_ids),
            "mean_judge_score_with_fallback": _safe_mean(with_fallback_scores),
            "mean_judge_score_without_fallback": _safe_mean(without_fallback_scores),
            "mean_judge_score_with_repair": _safe_mean(with_repair_scores),
            "mean_judge_score_without_repair": _safe_mean(without_repair_scores),
            "verdict_counts_with_fallback": dict(verdict_with_fallback),
            "verdict_counts_without_fallback": dict(verdict_without_fallback),
            "verdict_counts_with_repair": dict(verdict_with_repair),
            "verdict_counts_without_repair": dict(verdict_without_repair),
        },
        "provider_cost_overhead": {
            "extra_provider_call_count": overall["extra_provider_call_count"],
            "extra_total_tokens": overall["extra_total_tokens"],
            "extra_latency_ms": overall["extra_latency_ms"],
        },
        "quality_risk_events": {
            "count": overall["quality_risk_event_count"],
            "weighted_count": overall["quality_risk_weighted_event_count"],
            "deterministic_fallback_count": sum(
                _as_float(event.get("event_weight"), 1.0)
                for event in events
                if event.get("action") == "deterministic_fallback"
            ),
            "discarded_repair_count": sum(
                _as_float(event.get("event_weight"), 1.0)
                for event in events
                if event.get("outcome") == "discarded"
            ),
        },
        "paper_ready": paper_ready,
    }


def legacy_fallback_repair_diagnostics(
    rows: list[dict[str, Any]],
    llm_call_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_counts = Counter()
    repair_counts = Counter()
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        fallback_counts.update(dict(metadata.get("llm_fallback_counts") or {}))
        repair_counts.update(dict(metadata.get("llm_repair_counts") or {}))
    llm_call_diagnostics = llm_call_diagnostics or {}
    overall_llm = dict(llm_call_diagnostics.get("overall") or {})
    return {
        "diagnostic_mode": "legacy_counters",
        "overall": {
            "event_count": sum(fallback_counts.values()) + sum(repair_counts.values()),
            "weighted_event_count": sum(fallback_counts.values()) + sum(repair_counts.values()),
            "fallback_event_count": sum(fallback_counts.values()),
            "repair_event_count": sum(repair_counts.values()),
            "cache_event_count": 0,
            "extra_provider_call_count": overall_llm.get("fallback_count", 0),
            "extra_prompt_tokens": 0,
            "extra_completion_tokens": 0,
            "extra_total_tokens": 0,
            "extra_latency_ms": 0,
            "repair_discarded_count": 0,
            "repair_discard_rate": 0.0,
            "quality_risk_event_count": 0,
            "quality_risk_weighted_event_count": 0.0,
        },
        "by_phase": {},
        "by_task": {},
        "by_provider_role": {},
        "by_model": {},
        "fallback_chains": {},
        "quality_correlation": {},
        "provider_cost_overhead": {},
        "quality_risk_events": {},
        "paper_ready": {
            "extra_llm_call_rate": 0.0,
            "extra_token_overhead_rate": 0.0,
        },
        "legacy_fallback_counts": dict(fallback_counts),
        "legacy_repair_counts": dict(repair_counts),
    }
