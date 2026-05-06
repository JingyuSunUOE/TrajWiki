"""Offline counterfactual retrieval/context ablation reports."""

from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Iterable

from trajpatch.analysis.context_cost import (
    TOKEN_ESTIMATOR_NAME,
    estimate_context_tokens,
    select_ranked_rows_with_budget,
)
from trajpatch.analysis.direct_retrieval import rank_direct_trajectories
from trajpatch.analysis.gold_labels import (
    build_memory_index,
    extract_source_refs,
    load_details_rows,
    load_json_field,
    load_or_build_gold_labels,
)
from trajpatch.memory.facets import (
    build_sample_entity_lexicon,
    classify_query_shape_v1,
    extract_query_facets_v1,
)
from trajpatch.utils.json_utils import append_jsonl, write_json
from trajpatch.utils.text import extract_keywords

DEFAULT_VARIANTS = [
    "full",
    "no_wiki_direct",
    "wiki_only",
    "flat_raw",
    "snapshot_m1",
    "snapshot_m2",
    "source_supported_only",
]
DEFAULT_BUDGETS = [4000, 8000, 16000, 32000]
DEFAULT_CUTOFFS = [5, 10, 15, 20, 30, 50]


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


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return (float(numerator) / float(denominator)) if denominator else None


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _load_retrieval_events(database_path: Path, memory_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_messages_by_id = memory_index["raw_messages_by_id"]
    snapshot_refs = memory_index["snapshot_refs"]
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        events: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            """
            SELECT id, query_text, top_t_pages, top_k, page_ids_json, trajectory_ids_json,
                   snapshot_ids_json, expanded_snapshot_ids_json, source_message_ids_json, metadata_json
            FROM retrieval_events
            """
        ):
            metadata = dict(load_json_field(row["metadata_json"], {}))
            source_ids = [
                str(item) for item in load_json_field(row["source_message_ids_json"], []) if str(item).strip()
            ]
            source_refs = [
                str(raw_messages_by_id[message_id]["source_ref"])
                for message_id in source_ids
                if message_id in raw_messages_by_id and raw_messages_by_id[message_id]["source_ref"]
            ]
            if metadata.get("source_refs"):
                source_refs = [str(item) for item in list(metadata.get("source_refs") or []) if str(item).strip()]
            expanded_snapshot_ids = [
                str(item) for item in load_json_field(row["expanded_snapshot_ids_json"], []) if str(item).strip()
            ]
            events[str(row["id"])] = {
                "id": str(row["id"]),
                "query_text": str(row["query_text"] or ""),
                "top_t_pages": int(row["top_t_pages"] or 0),
                "top_k": int(row["top_k"] or 0),
                "page_ids": [str(item) for item in load_json_field(row["page_ids_json"], []) if str(item).strip()],
                "trajectory_ids": [
                    str(item) for item in load_json_field(row["trajectory_ids_json"], []) if str(item).strip()
                ],
                "snapshot_ids": [
                    str(item) for item in load_json_field(row["snapshot_ids_json"], []) if str(item).strip()
                ],
                "expanded_snapshot_ids": expanded_snapshot_ids,
                "source_message_ids": source_ids,
                "source_refs": list(dict.fromkeys(source_refs)),
                "expanded_refs": sorted(
                    set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in expanded_snapshot_ids))
                ),
                "metadata": metadata,
            }
        return events
    finally:
        connection.close()


def _query_diagnostics(
    row: dict[str, Any],
    event: dict[str, Any],
    memory_index: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    metadata = dict(event.get("metadata") or {})
    entities = [str(item) for item in list(metadata.get("query_entities") or []) if str(item).strip()]
    facets = dict(metadata.get("query_facets") or {})
    query_shape = dict(metadata.get("query_shape") or {})
    if entities or facets or query_shape:
        return (
            entities,
            {
                "tags": [str(item) for item in list(facets.get("tags") or []) if str(item).strip()],
                "values": [str(item) for item in list(facets.get("values") or []) if str(item).strip()],
            },
            query_shape,
        )
    sample_id = str(row.get("sample_id") or "")
    raw_messages = [
        SimpleNamespace(**message)
        for message in list(memory_index["sample_raw_messages"].get(sample_id, []))
    ]
    lexicon = build_sample_entity_lexicon(raw_messages)
    question = str(row.get("question") or "")
    query_facets = extract_query_facets_v1(question, lexicon)
    return (
        list(query_facets.get("entities") or []),
        {
            "tags": list(query_facets.get("tags") or []),
            "values": list(query_facets.get("values") or []),
        },
        dict(classify_query_shape_v1(question, lexicon)),
    )


def _coverage(
    *,
    gold: dict[str, Any],
    selected_source_refs: Iterable[str],
    selected_trajectory_ids: Iterable[str],
) -> dict[str, Any]:
    gold_refs = set(str(item) for item in list(gold.get("gold_source_refs") or []) if str(item).strip())
    gold_trajectory_ids = set(
        str(item) for item in list(gold.get("gold_trajectory_ids") or []) if str(item).strip()
    )
    source_ref_set = set(str(item) for item in selected_source_refs if str(item).strip())
    trajectory_set = set(str(item) for item in selected_trajectory_ids if str(item).strip())
    covered_refs = sorted(gold_refs & source_ref_set)
    covered_trajectories = sorted(gold_trajectory_ids & trajectory_set)
    return {
        "gold_ref_coverage": _safe_rate(len(covered_refs), len(gold_refs)),
        "gold_ref_coverage_count": len(covered_refs),
        "all_gold_refs": bool(gold_refs) and gold_refs <= source_ref_set,
        "gold_trajectory_recall": _safe_rate(len(covered_trajectories), len(gold_trajectory_ids)),
        "gold_trajectory_count_in_selection": len(covered_trajectories),
        "all_gold_trajectories": bool(gold_trajectory_ids) and gold_trajectory_ids <= trajectory_set,
        "covered_gold_source_refs": covered_refs,
        "covered_gold_trajectory_ids": covered_trajectories,
    }


def _source_rows_for_messages(message_ids: Iterable[str], memory_index: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages_by_id = memory_index["raw_messages_by_id"]
    rows: list[dict[str, Any]] = []
    for rank, message_id in enumerate(dict.fromkeys(str(item) for item in message_ids if str(item).strip()), start=1):
        message = raw_messages_by_id.get(message_id)
        if not message:
            continue
        rows.append(
            {
                "rank": rank,
                "item_id": message_id,
                "item_type": "source_message",
                "source_refs": [message["source_ref"]] if message.get("source_ref") else [],
                "estimated_tokens": estimate_context_tokens(message.get("content", "")),
                "text": message.get("content", ""),
            }
        )
    return rows


def _full_rows(event: dict[str, Any], memory_index: dict[str, Any]) -> list[dict[str, Any]]:
    return _source_rows_for_messages(event.get("source_message_ids") or [], memory_index)


def _direct_rows(
    row: dict[str, Any],
    event: dict[str, Any],
    memory_index: dict[str, Any],
    *,
    rank_limit: int,
) -> tuple[list[dict[str, Any]], int]:
    query_entities, query_facets, query_shape = _query_diagnostics(row, event, memory_index)
    ranked_ids, _, compact_rows = rank_direct_trajectories(
        sample_id=str(row.get("sample_id") or ""),
        question=str(row.get("question") or ""),
        query_entities=query_entities,
        query_facets=query_facets,
        query_shape=query_shape,
        sample_to_trajectories=memory_index["sample_to_trajectories"],
        trajectory_metadata=memory_index["trajectory_metadata"],
        claims_by_trajectory=memory_index["claims_by_trajectory"],
        trajectory_refs=memory_index["trajectory_refs"],
        trajectory_lengths=memory_index["trajectory_lengths"],
        diagnostic_top_n=rank_limit,
    )
    return compact_rows, len(ranked_ids)


def _page_rank_rows(event: dict[str, Any], memory_index: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    metadata = dict(event.get("metadata") or {})
    rows = list(
        metadata.get("ablation_page_ranked_rows_v1")
        or metadata.get("page_ranked_rows_compact_top_n")
        or metadata.get("page_ranked_rows")
        or []
    )
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        page_id = str(row.get("page_id") or row.get("item_id") or "")
        if not page_id:
            continue
        fallback_trajectory_ids = memory_index["page_to_trajectory_ids"].get(page_id, [])
        trajectory_ids = [
            str(item)
            for item in list(
                row.get("trajectory_ids")
                or row.get("linked_trajectory_ids")
                or fallback_trajectory_ids
            )
            if str(item).strip()
        ]
        text = memory_index["page_texts"].get(page_id, "")
        source_refs = [
            str(item) for item in list(row.get("source_refs") or []) if str(item).strip()
        ] or extract_source_refs(text)
        normalized.append(
            {
                **dict(row),
                "rank": int(row.get("rank") or index),
                "item_id": page_id,
                "item_type": "wiki_page",
                "page_id": page_id,
                "linked_trajectory_ids": list(dict.fromkeys(trajectory_ids)),
                "trajectory_ids": list(dict.fromkeys(trajectory_ids)),
                "source_refs": sorted(set(source_refs)),
                "estimated_tokens": int(row.get("estimated_tokens") or estimate_context_tokens(text)),
            }
        )
    total = int(metadata.get("page_ranked_total_count") or len(normalized))
    return normalized, total


def _flat_raw_rows(
    row: dict[str, Any],
    memory_index: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    sample_id = str(row.get("sample_id") or "")
    question_terms = extract_keywords(str(row.get("question") or ""))
    scored: list[tuple[float, dict[str, Any]]] = []
    for message in list(memory_index["sample_raw_messages"].get(sample_id, [])):
        message_terms = extract_keywords(
            " ".join(
                str(message.get(key) or "")
                for key in ["speaker_name", "source_ref", "content", "occurred_at"]
            )
        )
        overlap = question_terms & message_terms
        score = float(len(overlap))
        if question_terms:
            score += len(overlap) / len(question_terms)
        if score <= 0:
            score = 0.001
        source_ref = str(message.get("source_ref") or "")
        trajectory_ids = sorted(memory_index["source_ref_to_trajectories"].get(source_ref, set()))
        scored.append(
            (
                score,
                {
                    "item_id": str(message["id"]),
                    "item_type": "raw_message",
                    "source_refs": [source_ref] if source_ref else [],
                    "linked_trajectory_ids": trajectory_ids,
                    "estimated_tokens": estimate_context_tokens(message.get("content", "")),
                    "score": score,
                    "final_score": score,
                },
            )
        )
    scored.sort(
        key=lambda item: (
            -item[0],
            int(memory_index["raw_messages_by_id"][item[1]["item_id"]]["turn_index"]),
        )
    )
    rows = [{**payload, "rank": rank} for rank, (_, payload) in enumerate(scored, start=1)]
    return rows, len(rows)


def _snapshot_depth_rows(
    event: dict[str, Any],
    memory_index: dict[str, Any],
    depth: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    rank = 1
    selected_trajectories = [str(item) for item in list(event.get("trajectory_ids") or []) if str(item).strip()]
    for trajectory_id in selected_trajectories:
        snapshot_ids = list(memory_index["trajectory_to_snapshots"].get(trajectory_id, []))[-depth:]
        for snapshot_id in snapshot_ids:
            rows.append(
                {
                    "rank": rank,
                    "item_id": snapshot_id,
                    "item_type": "snapshot",
                    "selected_snapshot_ids": [snapshot_id],
                    "linked_trajectory_ids": [trajectory_id],
                    "source_refs": sorted(memory_index["snapshot_refs"].get(snapshot_id, set())),
                    "estimated_tokens": estimate_context_tokens(
                        memory_index["snapshot_texts"].get(snapshot_id, "")
                    ),
                }
            )
            rank += 1
    return rows, len(rows)


def _source_supported_claim_rows(
    event: dict[str, Any],
    memory_index: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    rank = 1
    for snapshot_id in list(event.get("expanded_snapshot_ids") or []):
        trajectory_id = memory_index["snapshot_to_trajectory"].get(snapshot_id, "")
        for claim in list(memory_index["claims_by_snapshot"].get(snapshot_id, [])):
            source_refs = sorted(set(claim.get("source_refs") or []))
            if not source_refs:
                continue
            rows.append(
                {
                    "rank": rank,
                    "item_id": f"{snapshot_id}:claim:{rank}",
                    "item_type": "source_supported_claim",
                    "selected_snapshot_ids": [snapshot_id],
                    "linked_trajectory_ids": [trajectory_id] if trajectory_id else [],
                    "source_refs": source_refs,
                    "estimated_tokens": estimate_context_tokens(claim.get("text", "")),
                }
            )
            rank += 1
    return rows, len(rows)


def _rows_for_variant(
    *,
    variant: str,
    row: dict[str, Any],
    event: dict[str, Any],
    memory_index: dict[str, Any],
    rank_limit: int,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    if variant == "full":
        rows = _full_rows(event, memory_index)
        return rows, len(rows), {"counterfactual_observability": "observed_full_run"}
    if variant == "no_wiki_direct":
        rows, total = _direct_rows(row, event, memory_index, rank_limit=rank_limit)
        return rows, total, {"counterfactual_observability": "metadata_direct_retrieval"}
    if variant == "wiki_only":
        rows, total = _page_rank_rows(event, memory_index)
        return rows, total, {"counterfactual_observability": "saved_wiki_rank_proxy"}
    if variant == "flat_raw":
        rows, total = _flat_raw_rows(row, memory_index)
        return rows, total, {"counterfactual_observability": "lexical_raw_message_proxy"}
    if variant.startswith("snapshot_m"):
        try:
            depth = max(1, int(variant.removeprefix("snapshot_m")))
        except ValueError:
            depth = 1
        rows, total = _snapshot_depth_rows(event, memory_index, depth)
        return rows, total, {
            "counterfactual_observability": "selected_trajectory_snapshot_depth_proxy",
            "snapshot_depth": depth,
        }
    if variant == "source_supported_only":
        rows, total = _source_supported_claim_rows(event, memory_index)
        return rows, total, {"counterfactual_observability": "not_counterfactually_observable"}
    return [], 0, {"counterfactual_observability": "unknown_variant"}


def _selected_payload(selected_rows: list[dict[str, Any]], memory_index: dict[str, Any]) -> dict[str, Any]:
    source_refs = sorted(
        {
            str(ref)
            for row in selected_rows
            for ref in list(row.get("source_refs") or [])
            if str(ref).strip()
        }
    )
    trajectory_ids = sorted(
        {
            str(tid)
            for row in selected_rows
            for tid in list(row.get("linked_trajectory_ids") or row.get("trajectory_ids") or [])
            if str(tid).strip()
        }
    )
    snapshot_ids = sorted(
        {
            str(snapshot_id)
            for row in selected_rows
            for snapshot_id in list(row.get("selected_snapshot_ids") or row.get("snapshot_ids") or [])
            if str(snapshot_id).strip()
        }
    )
    if not snapshot_ids and trajectory_ids:
        snapshot_ids = sorted(
            {
                snapshot_id
                for trajectory_id in trajectory_ids
                for snapshot_id in list(memory_index["trajectory_to_snapshots"].get(trajectory_id, []))
            }
        )
    return {
        "selected_item_ids": [
            str(row.get("item_id") or row.get("trajectory_id") or row.get("page_id") or "")
            for row in selected_rows
        ],
        "selected_trajectory_ids": trajectory_ids,
        "selected_snapshot_ids": snapshot_ids,
        "selected_source_refs": source_refs,
        "estimated_context_tokens": sum(
            int(row.get("estimated_tokens") or row.get("context_token_estimate") or 0)
            for row in selected_rows
        ),
    }


def build_offline_ablation_rows(
    *,
    sample_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    retrieval_events: dict[str, dict[str, Any]],
    memory_index: dict[str, Any],
    variants: list[str],
    budgets: list[int],
    rank_cutoffs: list[int],
) -> list[dict[str, Any]]:
    gold_by_query = {str(row.get("query_task_id")): row for row in gold_rows}
    max_rank_limit = max(rank_cutoffs or [50])
    output: list[dict[str, Any]] = []
    for sample_row in sample_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        event = retrieval_events.get(str(sample_row.get("retrieval_event_id") or ""), {})
        gold = gold_by_query.get(query_task_id, {})
        for variant in variants:
            base_rows, candidate_universe_size, variant_meta = _rows_for_variant(
                variant=variant,
                row=sample_row,
                event=event,
                memory_index=memory_index,
                rank_limit=max_rank_limit,
            )
            for rank_cutoff in rank_cutoffs:
                cutoff_rows = base_rows[:rank_cutoff]
                for budget in budgets:
                    selected_rows = select_ranked_rows_with_budget(cutoff_rows, budget_tokens=budget)
                    selected = _selected_payload(selected_rows, memory_index)
                    coverage = _coverage(
                        gold=gold,
                        selected_source_refs=selected["selected_source_refs"],
                        selected_trajectory_ids=selected["selected_trajectory_ids"],
                    )
                    metrics = dict(sample_row.get("metrics") or {})
                    observed = variant == "full"
                    output.append(
                        {
                            "schema_version": "offline_ablation_row_v1",
                            "variant": variant,
                            "sample_id": sample_row.get("sample_id"),
                            "query_task_id": query_task_id,
                            "question": sample_row.get("question"),
                            "candidate_universe_size": candidate_universe_size,
                            "rank_cutoff": rank_cutoff,
                            "budget_tokens": budget,
                            **selected,
                            **coverage,
                            "unsupported_evidence_risk": coverage["all_gold_refs"] is not True,
                            "answerability_proxy": coverage["gold_ref_coverage"],
                            "observed_answer_judge_acc": metrics.get("judge_acc") if observed else None,
                            "observed_answer_f1": metrics.get("F1") if observed else None,
                            "observed_answer_available": observed,
                            "token_estimator": TOKEN_ESTIMATOR_NAME,
                            **variant_meta,
                        }
                    )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("rank_cutoff") or 0) == 15:
            grouped[(str(row.get("variant")), int(row.get("budget_tokens") or 0))].append(row)
    if not grouped:
        for row in rows:
            grouped[(str(row.get("variant")), int(row.get("budget_tokens") or 0))].append(row)
    output: list[dict[str, Any]] = []
    for (variant, budget), group in sorted(grouped.items()):
        output.append(
            {
                "variant": variant,
                "budget_tokens": budget,
                "mean_gold_ref_coverage": _safe_mean(row.get("gold_ref_coverage") for row in group),
                "mean_gold_trajectory_recall@15": _safe_mean(row.get("gold_trajectory_recall") for row in group),
                "all_gold_refs_rate": _safe_rate(
                    sum(1 for row in group if row.get("all_gold_refs") is True),
                    len(group),
                ),
                "all_gold_trajectories@15_rate": _safe_rate(
                    sum(1 for row in group if row.get("all_gold_trajectories") is True),
                    len(group),
                ),
                "mean_candidate_universe_size": _safe_mean(row.get("candidate_universe_size") for row in group),
                "mean_estimated_context_tokens": _safe_mean(row.get("estimated_context_tokens") for row in group),
                "unsupported_evidence_risk_rate": _safe_rate(
                    sum(1 for row in group if row.get("unsupported_evidence_risk") is True),
                    len(group),
                ),
                "observed_judge_acc_if_available": _safe_mean(
                    row.get("observed_answer_judge_acc") for row in group
                ),
                "observed_f1_if_available": _safe_mean(row.get("observed_answer_f1") for row in group),
                "query_count": len(group),
            }
        )
    return output


def _cost_recall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("variant")),
                int(row.get("budget_tokens") or 0),
                int(row.get("rank_cutoff") or 0),
            )
        ].append(row)
    return [
        {
            "variant": variant,
            "budget_tokens": budget,
            "rank_cutoff": cutoff,
            "mean_gold_ref_coverage": _safe_mean(row.get("gold_ref_coverage") for row in group),
            "mean_gold_trajectory_recall": _safe_mean(row.get("gold_trajectory_recall") for row in group),
            "mean_estimated_context_tokens": _safe_mean(row.get("estimated_context_tokens") for row in group),
            "query_count": len(group),
        }
        for (variant, budget, cutoff), group in sorted(grouped.items())
    ]


def _evidence_funnel_rows(
    *,
    sample_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    retrieval_events: dict[str, dict[str, Any]],
    memory_index: dict[str, Any],
) -> list[dict[str, Any]]:
    gold_by_query = {str(row.get("query_task_id")): row for row in gold_rows}
    stages = [
        "memory_stored",
        "page_routed",
        "trajectory_selected",
        "snapshot_selected",
        "final_source_grounded",
    ]
    hits = {stage: 0 for stage in stages}
    total = 0
    for row in sample_rows:
        query_id = str(row.get("query_task_id") or "")
        gold = gold_by_query.get(query_id, {})
        gold_refs = set(gold.get("gold_source_refs") or [])
        if not gold_refs:
            continue
        total += 1
        event = retrieval_events.get(str(row.get("retrieval_event_id") or ""), {})
        sample_id = str(row.get("sample_id") or "")
        sample_refs = set().union(
            *(
                memory_index["trajectory_refs"].get(tid, set())
                for tid in memory_index["sample_to_trajectories"].get(sample_id, set())
            )
        )
        if gold_refs & sample_refs:
            hits["memory_stored"] += 1
        page_trajectory_ids = {
            tid
            for page_id in list(event.get("page_ids") or [])
            for tid in memory_index["page_to_trajectory_ids"].get(page_id, [])
        }
        if set(gold.get("gold_trajectory_ids") or []) & page_trajectory_ids:
            hits["page_routed"] += 1
        if set(gold.get("gold_trajectory_ids") or []) & set(event.get("trajectory_ids") or []):
            hits["trajectory_selected"] += 1
        if set(gold.get("gold_snapshot_ids") or []) & set(event.get("expanded_snapshot_ids") or []):
            hits["snapshot_selected"] += 1
        if gold_refs & set(event.get("source_refs") or []):
            hits["final_source_grounded"] += 1
    return [
        {
            "stage": stage,
            "query_count": total,
            "hit_count": hits[stage],
            "hit_rate": _safe_rate(hits[stage], total),
        }
        for stage in stages
    ]


def _variant_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if int(row.get("rank_cutoff") or 0) == 15:
            key = f"{row.get('query_task_id')}::{row.get('budget_tokens')}"
            by_query[key][str(row.get("variant"))] = row
    examples: list[dict[str, Any]] = []
    for variants in by_query.values():
        full = variants.get("full", {})
        direct = variants.get("no_wiki_direct", {})
        wiki = variants.get("wiki_only", {})
        flat = variants.get("flat_raw", {})
        for label, candidate in [
            (
                "direct_beats_full",
                direct
                if (direct.get("gold_ref_coverage") or 0) > (full.get("gold_ref_coverage") or 0)
                else {},
            ),
            (
                "full_beats_direct",
                full
                if (full.get("gold_ref_coverage") or 0) > (direct.get("gold_ref_coverage") or 0)
                else {},
            ),
            (
                "wiki_only_loses_evidence",
                wiki
                if (wiki.get("gold_ref_coverage") or 0) < (full.get("gold_ref_coverage") or 0)
                else {},
            ),
            ("flat_raw_high_cost_hit", flat if flat.get("all_gold_refs") is True else {}),
        ]:
            if candidate and len([item for item in examples if item.get("example_type") == label]) < 10:
                examples.append({"example_type": label, **candidate})
    return examples[:40]


def analyze_offline_ablation(
    run_path: Path | str,
    *,
    variants: str | Iterable[str] | None = None,
    budgets: str | Iterable[int] | None = None,
    rank_cutoffs: str | Iterable[int] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_path).expanduser().resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    run_meta, sample_rows, database_path = load_details_rows(run_dir)
    if str(run_meta.get("dataset")).lower() != "locomo":
        raise ValueError("Offline ablation currently supports LOCOMO runs only.")
    selected_variants = parse_str_list(variants, DEFAULT_VARIANTS)
    selected_budgets = parse_int_list(budgets, DEFAULT_BUDGETS)
    selected_cutoffs = parse_int_list(rank_cutoffs, DEFAULT_CUTOFFS)
    memory_index = build_memory_index(database_path)
    gold_rows = load_or_build_gold_labels(run_dir, sample_rows, memory_index)
    retrieval_events = _load_retrieval_events(database_path, memory_index)
    rows = build_offline_ablation_rows(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        retrieval_events=retrieval_events,
        memory_index=memory_index,
        variants=selected_variants,
        budgets=selected_budgets,
        rank_cutoffs=selected_cutoffs,
    )
    summary_table = _summary_rows(rows)
    cost_recall = _cost_recall_rows(rows)
    evidence_funnel = _evidence_funnel_rows(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        retrieval_events=retrieval_events,
        memory_index=memory_index,
    )
    examples = _variant_examples(rows)

    analysis_dir = run_dir / "analysis"
    for path in [
        analysis_dir / "offline_ablation_rows.jsonl",
        analysis_dir / "variant_examples.jsonl",
    ]:
        if path.exists():
            path.unlink()
    append_jsonl(analysis_dir / "offline_ablation_rows.jsonl", rows)
    append_jsonl(analysis_dir / "variant_examples.jsonl", examples)
    write_json(
        analysis_dir / "offline_ablation_summary.json",
        {
            "schema_version": "offline_ablation_summary_v1",
            "diagnostic_mode": "offline_counterfactual_retrieval_context_ablation",
            "run_id": run_meta.get("run_id"),
            "variants": selected_variants,
            "budgets": selected_budgets,
            "rank_cutoffs": selected_cutoffs,
            "token_estimator": TOKEN_ESTIMATOR_NAME,
            "table": summary_table,
            "paths": {
                "rows": str(analysis_dir / "offline_ablation_rows.jsonl"),
                "table": str(analysis_dir / "offline_ablation_table.csv"),
                "cost_recall_curve": str(analysis_dir / "cost_recall_curve.csv"),
                "evidence_funnel": str(analysis_dir / "evidence_funnel.csv"),
                "variant_examples": str(analysis_dir / "variant_examples.jsonl"),
                "gold_labels": str(analysis_dir / "gold_labels.jsonl"),
            },
        },
    )
    _write_csv(
        analysis_dir / "offline_ablation_table.csv",
        summary_table,
        [
            "variant",
            "budget_tokens",
            "mean_gold_ref_coverage",
            "mean_gold_trajectory_recall@15",
            "all_gold_refs_rate",
            "all_gold_trajectories@15_rate",
            "mean_candidate_universe_size",
            "mean_estimated_context_tokens",
            "unsupported_evidence_risk_rate",
            "observed_judge_acc_if_available",
            "observed_f1_if_available",
            "query_count",
        ],
    )
    _write_csv(
        analysis_dir / "cost_recall_curve.csv",
        cost_recall,
        [
            "variant",
            "budget_tokens",
            "rank_cutoff",
            "mean_gold_ref_coverage",
            "mean_gold_trajectory_recall",
            "mean_estimated_context_tokens",
            "query_count",
        ],
    )
    _write_csv(
        analysis_dir / "evidence_funnel.csv",
        evidence_funnel,
        ["stage", "query_count", "hit_count", "hit_rate"],
    )
    return {
        "schema_version": "offline_ablation_report_v1",
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "query_count": len(sample_rows),
        "row_count": len(rows),
        "summary_path": str(analysis_dir / "offline_ablation_summary.json"),
        "table_path": str(analysis_dir / "offline_ablation_table.csv"),
        "rows_path": str(analysis_dir / "offline_ablation_rows.jsonl"),
    }
