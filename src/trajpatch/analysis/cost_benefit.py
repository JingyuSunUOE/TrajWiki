"""Offline cost-benefit and scalability analysis for completed benchmark runs."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from trajpatch.analysis.context_cost import TOKEN_ESTIMATOR_NAME, estimate_context_tokens
from trajpatch.analysis.cost_model import (
    break_even_queries,
    cost_phase_for_task,
    deployment_scope_for_task,
    estimate_dollar_cost,
    reusable_scope_for_task,
)
from trajpatch.analysis.gold_labels import build_memory_index, load_details_rows, load_json_field
from trajpatch.utils.json_utils import append_jsonl, write_json

DEFAULT_BASELINES = [
    "trajwiki_observed",
    "full_context_proxy",
    "no_wiki_direct",
    "flat_raw",
    "wiki_only",
]
DEFAULT_FUTURE_QUERY_COUNTS = [1, 2, 5, 10, 20, 50, 100]


def parse_int_list(value: str | Iterable[int] | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        output = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        output = [int(item) for item in value]
    return sorted({item for item in output if item > 0}) or list(default)


def parse_str_list(value: str | Iterable[str] | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        output = [part.strip() for part in value.split(",") if part.strip()]
    else:
        output = [str(item).strip() for item in value if str(item).strip()]
    return output or list(default)


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def load_price_config(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Price config not found: {resolved}")
    return dict(json.loads(resolved.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl_replace(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    if rows:
        append_jsonl(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact_cost_call_row(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    task = str(record.get("task") or metadata.get("task") or "unknown")
    model = str(metadata.get("provider_model") or metadata.get("model") or "")
    prompt_tokens = int(record.get("prompt_tokens") or metadata.get("prompt_tokens") or 0)
    completion_tokens = int(record.get("completion_tokens") or metadata.get("completion_tokens") or 0)
    return {
        "schema_version": "cost_call_v1",
        "provider_call_id": str(metadata.get("provider_call_id") or record.get("provider_call_id") or ""),
        "sample_id": metadata.get("sample_id"),
        "query_task_id": metadata.get("query_task_id"),
        "role": str(record.get("role") or metadata.get("role") or "unknown"),
        "task": task,
        "cost_phase": cost_phase_for_task(task, metadata),
        "deployment_scope": deployment_scope_for_task(task, metadata),
        "reusable_scope": reusable_scope_for_task(task, metadata),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": float(record.get("latency_ms") or metadata.get("latency_ms") or 0.0),
        "is_repair": cost_phase_for_task(task, metadata) == "repair_validation",
        "is_fallback": bool(metadata.get("fallback_used") or metadata.get("structured_fallback_used")),
        "structured_requested": bool(metadata.get("structured_requested")),
        "structured_fallback_used": bool(metadata.get("structured_fallback_used")),
        "batch_size": int(metadata.get("batch_size") or record.get("batch_size") or 1),
    }


def build_compact_cost_call_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_cost_call_row(record) for record in records]


def _load_retrieval_events(database_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        output: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            """
            SELECT id, page_ids_json, trajectory_ids_json, snapshot_ids_json,
                   expanded_snapshot_ids_json, source_message_ids_json, metadata_json
            FROM retrieval_events
            """
        ):
            output[str(row["id"])] = {
                "page_ids": load_json_field(row["page_ids_json"], []),
                "trajectory_ids": load_json_field(row["trajectory_ids_json"], []),
                "snapshot_ids": load_json_field(row["snapshot_ids_json"], []),
                "expanded_snapshot_ids": load_json_field(row["expanded_snapshot_ids_json"], []),
                "source_message_ids": load_json_field(row["source_message_ids_json"], []),
                "metadata": dict(load_json_field(row["metadata_json"], {})),
            }
        return output
    finally:
        connection.close()


def build_cost_query_rows(
    sample_rows: list[dict[str, Any]],
    database_path: Path,
) -> list[dict[str, Any]]:
    memory_index = build_memory_index(database_path)
    retrieval_events = _load_retrieval_events(database_path)
    raw_tokens_by_sample = {
        sample_id: sum(estimate_context_tokens(message.get("content", "")) for message in messages)
        for sample_id, messages in memory_index["sample_raw_messages"].items()
    }
    rows: list[dict[str, Any]] = []
    for row in sample_rows:
        sample_id = str(row.get("sample_id") or "")
        query_task_id = str(row.get("query_task_id") or "")
        event = retrieval_events.get(str(row.get("retrieval_event_id") or ""), {})
        event_metadata = dict(event.get("metadata") or {})
        retrieval_diag = dict(dict(row.get("metadata") or {}).get("retrieval_compact_diagnostics") or {})
        context_breakdown = dict(event_metadata.get("answer_context_token_breakdown_v1") or {})
        context_total = context_breakdown.get("total")
        answer_prompt = str(row.get("answer_prompt") or "")
        raw_memory_tokens = int(raw_tokens_by_sample.get(sample_id, 0))
        full_context_prompt_tokens = raw_memory_tokens + estimate_context_tokens(row.get("question") or "")
        trajectory_candidate_ids = [
            str(item)
            for item in list(event_metadata.get("trajectory_candidate_input_ids") or [])
            if str(item).strip()
        ]
        if not trajectory_candidate_ids:
            trajectory_candidate_ids = [
                str(item)
                for item in list(event.get("trajectory_ids") or [])
                if str(item).strip()
            ]
        direct_candidate_universe = len(memory_index["sample_to_trajectories"].get(sample_id, set()))
        metrics = dict(row.get("metrics") or {})
        source_refs = [str(item) for item in list(row.get("retrieval_source_refs") or []) if str(item).strip()]
        rows.append(
            {
                "schema_version": "cost_query_v1",
                "sample_id": sample_id,
                "query_task_id": query_task_id,
                "raw_memory_token_estimate": raw_memory_tokens,
                "full_context_prompt_token_estimate": full_context_prompt_tokens,
                "trajwiki_final_context_tokens": int(context_total or estimate_context_tokens(answer_prompt)),
                "wiki_routed_candidate_universe": len(trajectory_candidate_ids),
                "direct_candidate_universe": direct_candidate_universe,
                "selected_page_count": len(list(event.get("page_ids") or [])),
                "selected_trajectory_count": len(list(event.get("trajectory_ids") or [])),
                "selected_snapshot_count": len(list(event.get("expanded_snapshot_ids") or [])),
                "selected_source_count": len(list(event.get("source_message_ids") or [])),
                "answer_prompt_tokens": int(
                    dict(row.get("metadata") or {}).get("answer_prompt_tokens")
                    or dict(row.get("tokens") or {}).get("backbone_prompt_tokens")
                    or 0
                ),
                "answer_completion_tokens": int(
                    dict(row.get("metadata") or {}).get("answer_completion_tokens")
                    or dict(row.get("tokens") or {}).get("backbone_completion_tokens")
                    or 0
                ),
                "observed_f1": metrics.get("F1"),
                "observed_judge_acc": metrics.get("judge_acc"),
                "unsupported_answer_proxy": (
                    not source_refs
                    or bool(retrieval_diag.get("reflection_semantic_evidence_weak"))
                    or bool(row.get("initial_retrieval_bundle_weak"))
                ),
                "token_estimator": TOKEN_ESTIMATOR_NAME,
            }
        )
    return rows


def _reconstruct_cost_call_rows_from_details(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_row in sample_rows:
        metadata = dict(sample_row.get("metadata") or {})
        for task, usage in dict(metadata.get("llm_usage_by_task") or {}).items():
            task = str(task)
            usage = dict(usage or {})
            rows.append(
                compact_cost_call_row(
                    {
                        "role": "judge" if cost_phase_for_task(task, {}) == "evaluation_only" else "backbone",
                        "task": task,
                        "prompt_tokens": int(float(usage.get("prompt_tokens") or 0)),
                        "completion_tokens": int(float(usage.get("completion_tokens") or 0)),
                        "latency_ms": float(usage.get("latency_ms") or 0.0),
                        "metadata": {
                            "provider_call_id": f"{sample_row.get('query_task_id')}::{task}",
                            "sample_id": sample_row.get("sample_id"),
                            "query_task_id": sample_row.get("query_task_id"),
                            "task": task,
                            "batch_size": int(float(usage.get("batch_item_count") or 1)),
                        },
                    }
                )
            )
    return rows


def build_memory_scaling_rows(database_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        sample_ids = {
            str(row["sample_id"])
            for row in connection.execute("SELECT DISTINCT sample_id FROM raw_messages")
        }
        rows: list[dict[str, Any]] = []
        for sample_id in sorted(sample_ids):
            raw_messages = connection.execute(
                "SELECT content FROM raw_messages WHERE sample_id = ? ORDER BY turn_index",
                (sample_id,),
            ).fetchall()
            trajectory_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM trajectories WHERE sample_id = ?",
                    (sample_id,),
                ).fetchone()[0]
                or 0
            )
            snapshot_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM episodic_snapshots s
                    JOIN trajectories t ON s.trajectory_id = t.id
                    WHERE t.sample_id = ?
                    """,
                    (sample_id,),
                ).fetchone()[0]
                or 0
            )
            claim_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM claims c
                    JOIN trajectories t ON c.trajectory_id = t.id
                    WHERE t.sample_id = ?
                    """,
                    (sample_id,),
                ).fetchone()[0]
                or 0
            )
            claim_op_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM claim_ops o
                    JOIN trajectories t ON o.trajectory_id = t.id
                    WHERE t.sample_id = ?
                    """,
                    (sample_id,),
                ).fetchone()[0]
                or 0
            )
            wiki_page_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM wiki_pages WHERE sample_id = ?",
                    (sample_id,),
                ).fetchone()[0]
                or 0
            )
            non_index_wiki_page_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM wiki_pages WHERE sample_id = ? AND page_type != 'index'",
                    (sample_id,),
                ).fetchone()[0]
                or 0
            )
            rows.append(
                {
                    "schema_version": "memory_scaling_v1",
                    "sample_id": sample_id,
                    "raw_message_count": len(raw_messages),
                    "raw_memory_token_estimate": sum(
                        estimate_context_tokens(row["content"] or "") for row in raw_messages
                    ),
                    "trajectory_count": trajectory_count,
                    "snapshot_count": snapshot_count,
                    "claim_count": claim_count,
                    "claim_op_count": claim_op_count,
                    "wiki_page_count": wiki_page_count,
                    "non_index_wiki_page_count": non_index_wiki_page_count,
                    "avg_trajectory_length": (
                        float(snapshot_count) / float(trajectory_count) if trajectory_count else 0.0
                    ),
                }
            )
        return rows
    finally:
        connection.close()


def build_candidate_scaling_rows(
    cost_query_rows: list[dict[str, Any]],
    memory_scaling_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    memory_by_sample = {str(row.get("sample_id")): row for row in memory_scaling_rows}
    rows: list[dict[str, Any]] = []
    for query_row in cost_query_rows:
        sample_id = str(query_row.get("sample_id") or "")
        memory = memory_by_sample.get(sample_id, {})
        rows.append(
            {
                "sample_id": sample_id,
                "query_task_id": query_row.get("query_task_id"),
                "raw_memory_token_estimate": memory.get("raw_memory_token_estimate"),
                "raw_message_count": memory.get("raw_message_count"),
                "trajectory_count": memory.get("trajectory_count"),
                "snapshot_count": memory.get("snapshot_count"),
                "claim_count": memory.get("claim_count"),
                "wiki_page_count": memory.get("wiki_page_count"),
                "wiki_routed_candidate_universe": query_row.get("wiki_routed_candidate_universe"),
                "direct_candidate_universe": query_row.get("direct_candidate_universe"),
                "selected_page_count": query_row.get("selected_page_count"),
                "selected_trajectory_count": query_row.get("selected_trajectory_count"),
                "selected_snapshot_count": query_row.get("selected_snapshot_count"),
                "selected_source_count": query_row.get("selected_source_count"),
            }
        )
    return rows


def _augment_cost_rows_with_dollars(
    rows: list[dict[str, Any]],
    price_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        priced = estimate_dollar_cost(
            int(row.get("prompt_tokens") or 0),
            int(row.get("completion_tokens") or 0),
            str(row.get("model") or ""),
            price_config,
        )
        output.append(
            {
                **row,
                "input_cost": priced["input_cost"],
                "output_cost": priced["output_cost"],
                "dollar_cost": priced["total_cost"],
                "currency": priced["currency"],
                "price_available": priced["price_available"],
            }
        )
    return output


def _cost_phase_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("cost_phase") or "unknown"),
                str(row.get("deployment_scope") or "deployment"),
                str(row.get("reusable_scope") or "per_query"),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (phase, deployment_scope, reusable_scope), group in sorted(grouped.items()):
        dollar_values = [row.get("dollar_cost") for row in group if row.get("dollar_cost") is not None]
        output.append(
            {
                "cost_phase": phase,
                "deployment_scope": deployment_scope,
                "reusable_scope": reusable_scope,
                "provider_call_rows": len(group),
                "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in group),
                "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in group),
                "total_tokens": sum(int(row.get("total_tokens") or 0) for row in group),
                "latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in group),
                "dollar_cost": sum(float(value) for value in dollar_values) if dollar_values else None,
            }
        )
    return output


def _total_for_scope(
    rows: list[dict[str, Any]],
    *,
    deployment_scope: str | None = None,
    reusable_scope: str | None = None,
) -> dict[str, Any]:
    selected = rows
    if deployment_scope is not None:
        selected = [row for row in selected if row.get("deployment_scope") == deployment_scope]
    if reusable_scope is not None:
        selected = [row for row in selected if row.get("reusable_scope") == reusable_scope]
    dollar_values = [row.get("dollar_cost") for row in selected if row.get("dollar_cost") is not None]
    return {
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in selected),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in selected),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in selected),
        "latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in selected),
        "dollar_cost": sum(float(value) for value in dollar_values) if dollar_values else None,
    }


def _offline_proxy_rows(
    rows: list[dict[str, Any]],
    *,
    variant: str,
) -> list[dict[str, Any]]:
    variant_rows = [row for row in rows if row.get("variant") == variant]
    if not variant_rows:
        return []
    cutoff_rows = [row for row in variant_rows if int(row.get("rank_cutoff") or 0) == 15]
    if cutoff_rows:
        variant_rows = cutoff_rows
    max_budget = max(int(row.get("budget_tokens") or 0) for row in variant_rows)
    return [row for row in variant_rows if int(row.get("budget_tokens") or 0) == max_budget]


def _quality_table_rows(
    *,
    baselines: list[str],
    sample_rows: list[dict[str, Any]],
    cost_query_rows: list[dict[str, Any]],
    cost_call_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
    query_count: int,
) -> list[dict[str, Any]]:
    deployment_total = _total_for_scope(cost_call_rows, deployment_scope="deployment")
    benchmark_total = _total_for_scope(cost_call_rows, deployment_scope="benchmark_only")
    upfront = _total_for_scope(
        cost_call_rows,
        deployment_scope="deployment",
        reusable_scope="upfront_reusable",
    )
    per_query = _total_for_scope(
        cost_call_rows,
        deployment_scope="deployment",
        reusable_scope="per_query",
    )
    observed_f1 = _safe_mean(dict(row.get("metrics") or {}).get("F1") for row in sample_rows)
    observed_judge = _safe_mean(dict(row.get("metrics") or {}).get("judge_acc") for row in sample_rows)
    unsupported_rate = _safe_rate(
        sum(1 for row in cost_query_rows if row.get("unsupported_answer_proxy") is True),
        len(cost_query_rows),
    )
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        if baseline == "trajwiki_observed":
            output.append(
                {
                    "method": baseline,
                    "result_type": "observed_answer_level",
                    "query_count": query_count,
                    "observed_f1": observed_f1,
                    "observed_judge_acc": observed_judge,
                    "unsupported_answer_proxy_rate": unsupported_rate,
                    "upfront_total_tokens": upfront["total_tokens"],
                    "per_query_total_tokens": (
                        float(per_query["total_tokens"]) / query_count if query_count else None
                    ),
                    "deployment_total_tokens": deployment_total["total_tokens"],
                    "benchmark_only_total_tokens": benchmark_total["total_tokens"],
                    "deployment_dollar_cost": deployment_total["dollar_cost"],
                    "benchmark_only_dollar_cost": benchmark_total["dollar_cost"],
                    "mean_context_tokens": _safe_mean(
                        row.get("trajwiki_final_context_tokens") for row in cost_query_rows
                    ),
                }
            )
        elif baseline == "full_context_proxy":
            completion_mean = _safe_mean(row.get("answer_completion_tokens") for row in cost_query_rows) or 0.0
            context_mean = _safe_mean(row.get("full_context_prompt_token_estimate") for row in cost_query_rows)
            output.append(
                {
                    "method": baseline,
                    "result_type": "offline_cost_proxy",
                    "query_count": query_count,
                    "observed_f1": None,
                    "observed_judge_acc": None,
                    "unsupported_answer_proxy_rate": None,
                    "upfront_total_tokens": 0,
                    "per_query_total_tokens": (context_mean or 0.0) + completion_mean,
                    "deployment_total_tokens": ((context_mean or 0.0) + completion_mean) * query_count,
                    "benchmark_only_total_tokens": 0,
                    "deployment_dollar_cost": None,
                    "benchmark_only_dollar_cost": None,
                    "mean_context_tokens": context_mean,
                }
            )
        else:
            proxy_rows = _offline_proxy_rows(offline_rows, variant=baseline)
            context_mean = _safe_mean(row.get("estimated_context_tokens") for row in proxy_rows)
            completion_mean = _safe_mean(row.get("answer_completion_tokens") for row in cost_query_rows) or 0.0
            output.append(
                {
                    "method": baseline,
                    "result_type": "offline_context_proxy",
                    "query_count": len(proxy_rows),
                    "observed_f1": None,
                    "observed_judge_acc": None,
                    "unsupported_answer_proxy_rate": _safe_rate(
                        sum(1 for row in proxy_rows if row.get("unsupported_evidence_risk") is True),
                        len(proxy_rows),
                    ),
                    "mean_gold_ref_coverage_proxy": _safe_mean(
                        row.get("gold_ref_coverage") for row in proxy_rows
                    ),
                    "mean_gold_trajectory_recall_proxy": _safe_mean(
                        row.get("gold_trajectory_recall") for row in proxy_rows
                    ),
                    "upfront_total_tokens": 0,
                    "per_query_total_tokens": (context_mean or 0.0) + completion_mean,
                    "deployment_total_tokens": ((context_mean or 0.0) + completion_mean) * len(proxy_rows),
                    "benchmark_only_total_tokens": 0,
                    "deployment_dollar_cost": None,
                    "benchmark_only_dollar_cost": None,
                    "mean_context_tokens": context_mean,
                }
            )
    return output


def _break_even_rows(quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trajwiki = next((row for row in quality_rows if row.get("method") == "trajwiki_observed"), None)
    if not trajwiki:
        return []
    output: list[dict[str, Any]] = []
    tw_upfront = float(trajwiki.get("upfront_total_tokens") or 0.0)
    tw_per_query = float(trajwiki.get("per_query_total_tokens") or 0.0)
    for baseline in quality_rows:
        if baseline.get("method") == "trajwiki_observed":
            continue
        baseline_upfront = float(baseline.get("upfront_total_tokens") or 0.0)
        baseline_per_query = float(baseline.get("per_query_total_tokens") or 0.0)
        token_break_even = break_even_queries(
            tw_upfront - baseline_upfront,
            baseline_per_query - tw_per_query,
        )
        output.append(
            {
                "baseline": baseline.get("method"),
                "cost_metric": "tokens",
                "trajwiki_upfront": tw_upfront,
                "baseline_upfront": baseline_upfront,
                "trajwiki_per_query": tw_per_query,
                "baseline_per_query": baseline_per_query,
                "per_query_saving": baseline_per_query - tw_per_query,
                **token_break_even,
            }
        )
    return output


def _amortized_curve_rows(
    quality_rows: list[dict[str, Any]],
    future_query_counts: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in quality_rows:
        upfront = float(row.get("upfront_total_tokens") or 0.0)
        per_query = float(row.get("per_query_total_tokens") or 0.0)
        for query_count in future_query_counts:
            output.append(
                {
                    "method": row.get("method"),
                    "future_query_count": query_count,
                    "amortized_tokens_per_query": (upfront + per_query * query_count) / query_count,
                    "upfront_total_tokens": upfront,
                    "per_query_total_tokens": per_query,
                }
            )
    return output


def analyze_cost_benefit(
    run_path: Path | str,
    *,
    baselines: str | Iterable[str] | None = None,
    price_config_path: Path | str | None = None,
    future_query_counts: str | Iterable[int] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_path).expanduser().resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    run_meta, sample_rows, database_path = load_details_rows(run_dir)
    if str(run_meta.get("dataset")).lower() != "locomo":
        raise ValueError("Cost-benefit analysis currently supports LOCOMO runs only.")
    selected_baselines = parse_str_list(baselines, DEFAULT_BASELINES)
    selected_future_counts = parse_int_list(future_query_counts, DEFAULT_FUTURE_QUERY_COUNTS)
    price_config = load_price_config(price_config_path)
    analysis_dir = run_dir / "analysis"

    cost_call_rows = _read_jsonl(analysis_dir / "cost_call_rows.jsonl")
    if not cost_call_rows:
        cost_call_rows = _reconstruct_cost_call_rows_from_details(sample_rows)
    cost_call_rows = _augment_cost_rows_with_dollars(cost_call_rows, price_config)
    cost_query_rows = _read_jsonl(analysis_dir / "cost_query_rows.jsonl")
    if not cost_query_rows:
        cost_query_rows = build_cost_query_rows(sample_rows, database_path)
    offline_rows = _read_jsonl(analysis_dir / "offline_ablation_rows.jsonl")
    memory_scaling = build_memory_scaling_rows(database_path)
    candidate_scaling = build_candidate_scaling_rows(cost_query_rows, memory_scaling)
    phase_summary = _cost_phase_summary(cost_call_rows)
    quality_rows = _quality_table_rows(
        baselines=selected_baselines,
        sample_rows=sample_rows,
        cost_query_rows=cost_query_rows,
        cost_call_rows=cost_call_rows,
        offline_rows=offline_rows,
        query_count=len(sample_rows),
    )
    break_even = _break_even_rows(quality_rows)
    amortized_curve = _amortized_curve_rows(quality_rows, selected_future_counts)

    _write_csv(
        analysis_dir / "cost_phase_summary.csv",
        phase_summary,
        [
            "cost_phase",
            "deployment_scope",
            "reusable_scope",
            "provider_call_rows",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_ms",
            "dollar_cost",
        ],
    )
    _write_csv(
        analysis_dir / "cost_quality_table.csv",
        quality_rows,
        [
            "method",
            "result_type",
            "query_count",
            "observed_f1",
            "observed_judge_acc",
            "unsupported_answer_proxy_rate",
            "mean_gold_ref_coverage_proxy",
            "mean_gold_trajectory_recall_proxy",
            "upfront_total_tokens",
            "per_query_total_tokens",
            "deployment_total_tokens",
            "benchmark_only_total_tokens",
            "deployment_dollar_cost",
            "benchmark_only_dollar_cost",
            "mean_context_tokens",
        ],
    )
    _write_csv(
        analysis_dir / "amortization_break_even.csv",
        break_even,
        [
            "baseline",
            "cost_metric",
            "trajwiki_upfront",
            "baseline_upfront",
            "trajwiki_per_query",
            "baseline_per_query",
            "per_query_saving",
            "break_even_queries",
            "reason",
        ],
    )
    _write_csv(
        analysis_dir / "amortized_cost_curve.csv",
        amortized_curve,
        [
            "method",
            "future_query_count",
            "amortized_tokens_per_query",
            "upfront_total_tokens",
            "per_query_total_tokens",
        ],
    )
    _write_csv(
        analysis_dir / "memory_scaling.csv",
        memory_scaling,
        [
            "sample_id",
            "raw_message_count",
            "raw_memory_token_estimate",
            "trajectory_count",
            "snapshot_count",
            "claim_count",
            "claim_op_count",
            "wiki_page_count",
            "non_index_wiki_page_count",
            "avg_trajectory_length",
        ],
    )
    _write_csv(
        analysis_dir / "candidate_scaling.csv",
        candidate_scaling,
        [
            "sample_id",
            "query_task_id",
            "raw_memory_token_estimate",
            "raw_message_count",
            "trajectory_count",
            "snapshot_count",
            "claim_count",
            "wiki_page_count",
            "wiki_routed_candidate_universe",
            "direct_candidate_universe",
            "selected_page_count",
            "selected_trajectory_count",
            "selected_snapshot_count",
            "selected_source_count",
        ],
    )
    write_json(
        analysis_dir / "cost_benefit_summary.json",
        {
            "schema_version": "cost_benefit_summary_v1",
            "diagnostic_mode": "offline_cost_benefit_scalability",
            "run_id": run_meta.get("run_id"),
            "baselines": selected_baselines,
            "future_query_counts": selected_future_counts,
            "token_estimator": TOKEN_ESTIMATOR_NAME,
            "price_config_path": str(price_config_path) if price_config_path else None,
            "price_config_date": (price_config or {}).get("price_config_date"),
            "paths": {
                "cost_phase_summary": str(analysis_dir / "cost_phase_summary.csv"),
                "cost_quality_table": str(analysis_dir / "cost_quality_table.csv"),
                "amortization_break_even": str(analysis_dir / "amortization_break_even.csv"),
                "amortized_cost_curve": str(analysis_dir / "amortized_cost_curve.csv"),
                "memory_scaling": str(analysis_dir / "memory_scaling.csv"),
                "candidate_scaling": str(analysis_dir / "candidate_scaling.csv"),
                "cost_call_rows": str(analysis_dir / "cost_call_rows.jsonl"),
                "cost_query_rows": str(analysis_dir / "cost_query_rows.jsonl"),
            },
        },
    )
    return {
        "schema_version": "cost_benefit_report_v1",
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "query_count": len(sample_rows),
        "summary_path": str(analysis_dir / "cost_benefit_summary.json"),
        "quality_table_path": str(analysis_dir / "cost_quality_table.csv"),
        "phase_summary_path": str(analysis_dir / "cost_phase_summary.csv"),
        "break_even_path": str(analysis_dir / "amortization_break_even.csv"),
    }


def write_cost_diagnostic_artifacts(
    *,
    run_dir: Path,
    sample_rows: list[dict[str, Any]],
    database_path: Path,
    call_records: list[dict[str, Any]],
    save_compact_calls: bool,
) -> dict[str, str]:
    analysis_dir = run_dir / "analysis"
    cost_query_rows = build_cost_query_rows(sample_rows, database_path)
    _write_jsonl_replace(analysis_dir / "cost_query_rows.jsonl", cost_query_rows)
    if save_compact_calls:
        cost_call_rows = build_compact_cost_call_rows(call_records)
        _write_jsonl_replace(analysis_dir / "cost_call_rows.jsonl", cost_call_rows)
    return {
        "cost_query_rows": str(analysis_dir / "cost_query_rows.jsonl"),
        "cost_call_rows": str(analysis_dir / "cost_call_rows.jsonl"),
    }
