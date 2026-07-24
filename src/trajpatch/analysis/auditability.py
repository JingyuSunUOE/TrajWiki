"""Offline auditability and interpretability diagnostics for LOCOMO runs."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from trajpatch.analysis.context_cost import TOKEN_ESTIMATOR_NAME, estimate_context_tokens
from trajpatch.analysis.gold_labels import (
    build_memory_index,
    extract_source_refs,
    load_details_rows,
    load_json_field,
    load_or_build_gold_labels,
)
from trajpatch.analysis.memory_index import (
    source_message_ids_for_refs,
    versioned_analysis_path,
)
from trajpatch.analysis.query_scope import (
    filter_query_rows,
    load_query_scope,
    scoped_analysis_dir,
    validate_scope_against_rows,
)
from trajpatch.utils.json_utils import append_jsonl, write_json

DEFAULT_BASELINES = [
    "trajwiki_observed",
    "full_context_proxy",
    "no_wiki_direct",
    "flat_raw",
    "wiki_only",
]

CONFLICT_STATUSES = {"contradictory", "needs-confirmation", "needs_confirmation", "conflicted"}
DEPRECATED_STATUSES = {"deprecated", "obsolete", "superseded"}
UNSUPPORTED_POSTCHECK_ISSUES = {
    "unsupported_extra_items",
    "unsupported_count",
    "invalid_supporting_refs",
    "scope_mismatched_extra_item",
    "overgeneric_item",
}


def parse_str_list(value: str | Iterable[str] | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        output = [part.strip() for part in value.split(",") if part.strip()]
    else:
        output = [str(item).strip() for item in value if str(item).strip()]
    return output or list(default)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _gold_ref_coverage(
    gold_refs: Iterable[str],
    candidate_refs: set[str],
) -> float | None:
    gold_ref_set = set(gold_refs)
    return (
        len(gold_ref_set & candidate_refs) / len(gold_ref_set)
        if gold_ref_set
        else None
    )


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _safe_median(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return median(clean) if clean else None


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


def _dedupe_str(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _compact_text(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(row.get("metadata"))


def _answer_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_row_metadata(row).get("answer_metadata"))


def _collect_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            refs.extend(_collect_refs(nested))
        return _dedupe_str(refs)
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            refs.extend(_collect_refs(nested))
        return _dedupe_str(refs)
    return extract_source_refs(value)


def _refs_from_event_like(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key in ("source_refs", "valid_source_refs", "supporting_source_refs", "source_ref", "new_source_ref"):
            refs = _collect_refs(value.get(key))
            if refs:
                return refs
    return _collect_refs(value)


def support_refs_from_answer_metadata(answer_metadata: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in [
        "answer_synthesis_supporting_refs",
        "bridge_finalization_source_refs",
        "count_validation_source_derived_candidate_refs",
        "raw_rescue_source_refs",
        "answer_temporal_selected_source_ref",
        "valid_supporting_refs",
        "supporting_source_refs",
    ]:
        refs.extend(_collect_refs(answer_metadata.get(key)))

    payload = _as_dict(answer_metadata.get("answer_synthesis_payload"))
    refs.extend(_collect_refs(payload.get("supporting_source_refs")))
    for item_refs in [
        answer_metadata.get("answer_supported_list_item_refs"),
        answer_metadata.get("answer_supported_required_item_refs"),
    ]:
        refs.extend(_collect_refs(item_refs))
    for key in [
        "answer_synthesis_counted_events",
        "count_validation_llm_decisions",
        "count_validation_source_derived_candidate_events",
        "bridge_facts_used",
    ]:
        for event in _as_list(answer_metadata.get(key)):
            refs.extend(_refs_from_event_like(event))
    return _dedupe_str(refs)


def invalid_refs_from_answer_metadata(answer_metadata: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ["invalid_supporting_refs", "answer_synthesis_invalid_family_refs"]:
        refs.extend(_collect_refs(answer_metadata.get(key)))
    return _dedupe_str(refs)


def _source_message_ids_for_refs(
    refs: Iterable[str],
    memory_index: dict[str, Any],
    *,
    sample_id: str,
) -> list[str]:
    return source_message_ids_for_refs(
        memory_index,
        sample_id=sample_id,
        source_refs=refs,
    )


def _is_abstention(answer_text: Any) -> bool:
    text = " ".join(str(answer_text or "").casefold().split())
    if not text:
        return True
    abstain_markers = [
        "not enough",
        "cannot answer",
        "can't answer",
        "do not have enough",
        "don't have enough",
        "insufficient",
        "unknown",
        "not available",
        "no retrieved evidence",
    ]
    return any(marker in text for marker in abstain_markers)


def _answer_issue(answer_metadata: dict[str, Any]) -> str | None:
    issue = answer_metadata.get("answer_postcheck_issue")
    return str(issue).strip() if issue is not None and str(issue).strip() else None


def build_answer_support_rows(
    sample_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_row in sample_rows:
        sample_id = str(sample_row.get("sample_id") or "")
        query_task_id = str(sample_row.get("query_task_id") or "")
        answer_metadata = _answer_metadata(sample_row)
        invalid_refs = invalid_refs_from_answer_metadata(answer_metadata)
        support_refs = support_refs_from_answer_metadata(answer_metadata)
        issue = _answer_issue(answer_metadata)
        final_status = "supported"
        if invalid_refs:
            final_status = "invalid_source_ref"
        elif not support_refs:
            final_status = "no_visible_source"
        elif issue in UNSUPPORTED_POSTCHECK_ISSUES:
            final_status = "unsupported_postcheck"
        rows.append(
            {
                "schema_version": "answer_support_item_v1",
                "contains_sensitive_text": True,
                "sample_id": sample_id,
                "query_task_id": query_task_id,
                "item_id": f"{query_task_id}:final_answer",
                "item_kind": "final_answer",
                "item_text_preview": _compact_text(sample_row.get("answer_text")),
                "support_origin": "answer_metadata",
                "source_refs": support_refs,
                "source_message_ids": _source_message_ids_for_refs(
                    support_refs,
                    memory_index,
                    sample_id=sample_id,
                ),
                "invalid_supporting_refs": invalid_refs,
                "support_status": final_status,
                "answer_postcheck_issue": issue,
            }
        )

        for key, kind in [
            ("answer_supported_list_item_refs", "supported_list_item"),
            ("answer_supported_required_item_refs", "supported_required_item"),
        ]:
            for item_text, refs_value in _as_dict(answer_metadata.get(key)).items():
                refs = _collect_refs(refs_value)
                rows.append(
                    {
                        "schema_version": "answer_support_item_v1",
                        "contains_sensitive_text": True,
                        "sample_id": sample_id,
                        "query_task_id": query_task_id,
                        "item_id": f"{query_task_id}:{kind}:{_compact_text(item_text, limit=48)}",
                        "item_kind": kind,
                        "item_text_preview": _compact_text(item_text),
                        "support_origin": key,
                        "source_refs": refs,
                        "source_message_ids": _source_message_ids_for_refs(
                            refs,
                            memory_index,
                            sample_id=sample_id,
                        ),
                        "invalid_supporting_refs": [],
                        "support_status": "supported" if refs else "no_visible_source",
                        "answer_postcheck_issue": issue,
                    }
                )

        bridge_refs = _collect_refs(answer_metadata.get("bridge_finalization_source_refs"))
        if bridge_refs or answer_metadata.get("bridge_finalization_used"):
            rows.append(
                {
                    "schema_version": "answer_support_item_v1",
                    "contains_sensitive_text": True,
                    "sample_id": sample_id,
                    "query_task_id": query_task_id,
                    "item_id": f"{query_task_id}:bridge_finalization",
                    "item_kind": "bridge_finalization",
                    "item_text_preview": _compact_text(answer_metadata.get("bridge_finalization_target")),
                    "support_origin": "bridge_finalization_source_refs",
                    "source_refs": bridge_refs,
                    "source_message_ids": _source_message_ids_for_refs(
                        bridge_refs,
                        memory_index,
                        sample_id=sample_id,
                    ),
                    "invalid_supporting_refs": [],
                    "support_status": "supported" if bridge_refs else "no_visible_source",
                    "answer_postcheck_issue": issue,
                }
            )

        temporal_ref = _collect_refs(answer_metadata.get("answer_temporal_selected_source_ref"))
        if temporal_ref:
            rows.append(
                {
                    "schema_version": "answer_support_item_v1",
                    "contains_sensitive_text": True,
                    "sample_id": sample_id,
                    "query_task_id": query_task_id,
                    "item_id": f"{query_task_id}:temporal_alignment",
                    "item_kind": "temporal_alignment",
                    "item_text_preview": _compact_text(
                        answer_metadata.get("answer_temporal_selected_answer_text")
                        or answer_metadata.get("answer_temporal_selected_date")
                    ),
                    "support_origin": "answer_temporal_selected_source_ref",
                    "source_refs": temporal_ref,
                    "source_message_ids": _source_message_ids_for_refs(
                        temporal_ref,
                        memory_index,
                        sample_id=sample_id,
                    ),
                    "invalid_supporting_refs": [],
                    "support_status": "supported",
                    "answer_postcheck_issue": issue,
                }
            )

        for index, event in enumerate(_as_list(answer_metadata.get("answer_synthesis_counted_events")), start=1):
            refs = _refs_from_event_like(event)
            text = event.get("event") if isinstance(event, dict) else event
            rows.append(
                {
                    "schema_version": "answer_support_item_v1",
                    "contains_sensitive_text": True,
                    "sample_id": sample_id,
                    "query_task_id": query_task_id,
                    "item_id": f"{query_task_id}:counted_event:{index}",
                    "item_kind": "counted_event",
                    "item_text_preview": _compact_text(text),
                    "support_origin": "answer_synthesis_counted_events",
                    "source_refs": refs,
                    "source_message_ids": _source_message_ids_for_refs(
                        refs,
                        memory_index,
                        sample_id=sample_id,
                    ),
                    "invalid_supporting_refs": [],
                    "support_status": "supported" if refs else "no_visible_source",
                    "answer_postcheck_issue": issue,
                }
            )
    return rows


def _load_retrieval_events(database_path: Path, memory_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_messages_by_id = memory_index.get("raw_messages_by_id", {})
    snapshot_refs = memory_index.get("snapshot_refs", {})
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        events: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            """
            SELECT id, sample_id, page_ids_json, trajectory_ids_json, snapshot_ids_json,
                   expanded_snapshot_ids_json, source_message_ids_json, metadata_json
            FROM retrieval_events
            """
        ):
            source_ids = [
                str(item) for item in load_json_field(row["source_message_ids_json"], []) if str(item).strip()
            ]
            source_refs = [
                str(raw_messages_by_id[message_id]["source_ref"])
                for message_id in source_ids
                if message_id in raw_messages_by_id and raw_messages_by_id[message_id].get("source_ref")
            ]
            expanded_snapshot_ids = [
                str(item) for item in load_json_field(row["expanded_snapshot_ids_json"], []) if str(item).strip()
            ]
            metadata = dict(load_json_field(row["metadata_json"], {}))
            if metadata.get("source_refs"):
                source_refs = [str(item) for item in list(metadata.get("source_refs") or []) if str(item).strip()]
            events[str(row["id"])] = {
                "id": str(row["id"]),
                "sample_id": str(row["sample_id"] or ""),
                "page_ids": [str(item) for item in load_json_field(row["page_ids_json"], []) if str(item).strip()],
                "trajectory_ids": [
                    str(item) for item in load_json_field(row["trajectory_ids_json"], []) if str(item).strip()
                ],
                "snapshot_ids": [
                    str(item) for item in load_json_field(row["snapshot_ids_json"], []) if str(item).strip()
                ],
                "expanded_snapshot_ids": expanded_snapshot_ids,
                "source_message_ids": source_ids,
                "source_refs": _dedupe_str(source_refs),
                "expanded_refs": sorted(
                    set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in expanded_snapshot_ids))
                ),
                "metadata": metadata,
            }
        return events
    finally:
        connection.close()


def build_answer_context_claim_rows(
    sample_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
    retrieval_events: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    claims_by_snapshot = memory_index.get("claims_by_snapshot", {})
    for sample_row in sample_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        event = retrieval_events.get(str(sample_row.get("retrieval_event_id") or ""), {})
        for snapshot_id in list(event.get("expanded_snapshot_ids") or []):
            for claim in list(claims_by_snapshot.get(str(snapshot_id), [])):
                status = str(claim.get("status") or "").strip().lower()
                metadata = _as_dict(claim.get("metadata"))
                suppressed_reason = None
                if status in DEPRECATED_STATUSES:
                    suppressed_reason = "deprecated"
                elif metadata.get("speaker_grounding_suspect"):
                    suppressed_reason = "speaker_grounding_suspect"
                kept = suppressed_reason is None
                claim_id = str(claim.get("claim_id") or claim.get("id") or "")
                rows.append(
                    {
                        "schema_version": "answer_context_claim_v1",
                        "contains_sensitive_text": True,
                        "sample_id": str(sample_row.get("sample_id") or ""),
                        "query_task_id": query_task_id,
                        "retrieval_event_id": event.get("id"),
                        "claim_id": claim_id,
                        "claim_record_id": claim.get("id"),
                        "status": status,
                        "trajectory_id": claim.get("trajectory_id"),
                        "snapshot_id": snapshot_id,
                        "source_refs": sorted(claim.get("source_refs") or []),
                        "source_message_ids": list(claim.get("source_message_ids") or []),
                        "kept_in_answer_context": kept,
                        "suppressed_reason": suppressed_reason,
                        "claim_text_preview": _compact_text(claim.get("text"), limit=220),
                        "facets": list(claim.get("facets") or []),
                    }
                )
    return rows


def build_claim_lifecycle_rows(database_path: Path, memory_index: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages_by_id = memory_index.get("raw_messages_by_id", {})
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows: list[dict[str, Any]] = []
        for row in connection.execute(
            """
            SELECT o.id, o.snapshot_id, o.trajectory_id, o.op_type, o.target_claim_id,
                   o.new_claim_id, o.source_message_ids_json, o.rationale, o.metadata_json,
                   t.sample_id
            FROM claim_ops o
            JOIN trajectories t ON o.trajectory_id = t.id
            """
        ):
            source_ids = [
                str(item) for item in load_json_field(row["source_message_ids_json"], []) if str(item).strip()
            ]
            source_refs = [
                str(raw_messages_by_id[message_id]["source_ref"])
                for message_id in source_ids
                if message_id in raw_messages_by_id and raw_messages_by_id[message_id].get("source_ref")
            ]
            metadata = dict(load_json_field(row["metadata_json"], {}))
            rows.append(
                {
                    "schema_version": "claim_lifecycle_v1",
                    "contains_sensitive_text": True,
                    "sample_id": str(row["sample_id"] or ""),
                    "snapshot_id": str(row["snapshot_id"] or ""),
                    "trajectory_id": str(row["trajectory_id"] or ""),
                    "op_id": str(row["id"] or ""),
                    "op_type": str(row["op_type"] or "").upper(),
                    "target_claim_id": str(row["target_claim_id"] or ""),
                    "new_claim_id": str(row["new_claim_id"] or "") if row["new_claim_id"] else None,
                    "source_message_ids": source_ids,
                    "source_refs": _dedupe_str(source_refs),
                    "rationale_preview": _compact_text(row["rationale"], limit=220),
                    "claim_transition_debug": metadata.get("claim_transition_debug"),
                    "system_derived_ops": metadata.get("system_derived_ops"),
                    "ignored_model_ops": metadata.get("ignored_model_ops"),
                    "metadata": metadata,
                }
            )
        return rows
    finally:
        connection.close()


def _gold_by_query(gold_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("query_task_id") or ""): row for row in gold_rows}


def _support_rows_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_task_id") or "")].append(row)
    return grouped


def _context_claims_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_task_id") or "")].append(row)
    return grouped


def _sample_memory_refs(sample_id: str, memory_index: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for trajectory_id in memory_index.get("sample_to_trajectories", {}).get(sample_id, set()):
        refs.update(memory_index.get("trajectory_refs", {}).get(trajectory_id, set()))
    return refs


def _failure_stage(
    sample_row: dict[str, Any],
    gold: dict[str, Any],
    event: dict[str, Any],
    *,
    source_supported_proxy: bool,
    unsupported_answer_risk: bool,
    memory_index: dict[str, Any],
) -> str:
    verdict = str(sample_row.get("judge_verdict") or "").lower()
    if verdict == "correct" and source_supported_proxy:
        return "correct"
    gold_refs = set(str(item) for item in list(gold.get("gold_source_refs") or []) if str(item).strip())
    if not gold_refs:
        return "unknown_no_gold_refs"
    if not (gold_refs & _sample_memory_refs(str(sample_row.get("sample_id") or ""), memory_index)):
        return "memory_absent"
    gold_pages = set(str(item) for item in list(gold.get("gold_page_ids") or []) if str(item).strip())
    routed_pages = set(str(item) for item in list(event.get("page_ids") or []) if str(item).strip())
    if gold_pages and not (gold_pages & routed_pages):
        return "page_routing_miss"
    gold_trajectories = set(
        str(item) for item in list(gold.get("gold_trajectory_ids") or []) if str(item).strip()
    )
    selected_trajectories = set(str(item) for item in list(event.get("trajectory_ids") or []) if str(item).strip())
    if gold_trajectories and not (gold_trajectories & selected_trajectories):
        return "trajectory_selection_miss"
    expanded_refs = set(str(item) for item in list(event.get("expanded_refs") or []) if str(item).strip())
    selected_refs = set(str(item) for item in list(event.get("source_refs") or []) if str(item).strip())
    raw_expanded_refs = set(
        _collect_refs(_as_dict(event.get("metadata")).get("ablation_snapshot_candidate_rows_v1"))
    )
    if raw_expanded_refs and gold_refs & raw_expanded_refs and not gold_refs & expanded_refs:
        return "snapshot_compaction_miss"
    if gold_refs & expanded_refs and not gold_refs & selected_refs:
        return "source_compaction_miss"
    if gold_refs & selected_refs and unsupported_answer_risk:
        return "unsupported_overgeneration"
    if gold_refs & selected_refs:
        return "answer_synthesis_error"
    return "grounding_miss"


def build_auditability_rows(
    *,
    sample_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
    retrieval_events: dict[str, dict[str, Any]],
    answer_support_rows: list[dict[str, Any]],
    answer_context_claim_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_by_query = _gold_by_query(gold_rows)
    support_by_query = _support_rows_by_query(answer_support_rows)
    claims_by_query = _context_claims_by_query(answer_context_claim_rows)
    rows: list[dict[str, Any]] = []
    for sample_row in sample_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        answer_metadata = _answer_metadata(sample_row)
        gold = gold_by_query.get(query_task_id, {})
        event = retrieval_events.get(str(sample_row.get("retrieval_event_id") or ""), {})
        support_rows = support_by_query.get(query_task_id, [])
        claim_rows = claims_by_query.get(query_task_id, [])
        support_refs = _dedupe_str(
            ref
            for support_row in support_rows
            for ref in list(support_row.get("source_refs") or [])
            if support_row.get("support_status") != "no_visible_source"
        )
        invalid_refs = _dedupe_str(
            ref for support_row in support_rows for ref in list(support_row.get("invalid_supporting_refs") or [])
        )
        gold_refs = _dedupe_str(gold.get("gold_source_refs") or [])
        retrieved_refs = _dedupe_str(sample_row.get("retrieval_source_refs") or event.get("source_refs") or [])
        support_gold_overlap = sorted(set(support_refs) & set(gold_refs))
        retrieved_gold_overlap = sorted(set(retrieved_refs) & set(gold_refs))
        sample_id = str(sample_row.get("sample_id") or "")
        sample_raw_refs = {
            str(message.get("source_ref") or "")
            for message in memory_index.get("sample_raw_messages", {}).get(
                sample_id,
                [],
            )
            if str(message.get("source_ref") or "").strip()
        }
        sample_trajectory_ids = set(
            memory_index.get("sample_to_trajectories", {}).get(sample_id, set())
        )
        trajectory_linked_refs = set().union(
            *(
                memory_index.get("trajectory_refs", {}).get(trajectory_id, set())
                for trajectory_id in sample_trajectory_ids
            )
        )
        routed_trajectory_ids = {
            str(trajectory_id)
            for page_id in list(event.get("page_ids") or [])
            for trajectory_id in memory_index.get("page_to_trajectory_ids", {}).get(
                str(page_id),
                [],
            )
        }
        page_routed_refs = set().union(
            *(
                memory_index.get("trajectory_refs", {}).get(
                    trajectory_id,
                    set(),
                )
                for trajectory_id in routed_trajectory_ids
            )
        )
        selected_trajectory_refs = set().union(
            *(
                memory_index.get("trajectory_refs", {}).get(
                    str(trajectory_id),
                    set(),
                )
                for trajectory_id in list(event.get("trajectory_ids") or [])
            )
        )
        expanded_refs = {
            str(ref)
            for ref in list(event.get("expanded_refs") or [])
            if str(ref).strip()
        }

        answer_text = str(sample_row.get("answer_text") or "")
        answer_abstained = _is_abstention(answer_text)
        issue = _answer_issue(answer_metadata)
        observed_answer_available = bool(answer_text.strip()) and str(sample_row.get("judge_verdict") or "") != "judge_error"
        no_valid_support = observed_answer_available and not answer_abstained and not support_refs
        no_gold_overlap = bool(gold_refs) and bool(support_refs) and not support_gold_overlap
        unsupported_answer_risk = bool(
            issue in UNSUPPORTED_POSTCHECK_ISSUES
            or invalid_refs
            or no_valid_support
            or no_gold_overlap
        )
        unsupported_reason = None
        if invalid_refs:
            unsupported_reason = "invalid_supporting_refs"
        elif issue in UNSUPPORTED_POSTCHECK_ISSUES:
            unsupported_reason = str(issue)
        elif no_valid_support:
            unsupported_reason = "non_abstain_no_valid_support_refs"
        elif no_gold_overlap:
            unsupported_reason = "support_refs_no_gold_overlap"
        source_supported_proxy = bool(
            observed_answer_available
            and not answer_abstained
            and support_refs
            and not invalid_refs
            and issue not in UNSUPPORTED_POSTCHECK_ISSUES
            and (not gold_refs or bool(support_gold_overlap))
        )
        conflict_claim_visible = any(
            row.get("kept_in_answer_context") and str(row.get("status") or "").lower() in CONFLICT_STATUSES
            for row in claim_rows
        )
        event_metadata = _as_dict(event.get("metadata"))
        conflict_block_visible = bool(
            event_metadata.get("conflicts")
            or event_metadata.get("conflict_blocks")
            or answer_metadata.get("bridge_facts_conflicted")
        )
        deprecated_claims = [
            row for row in claim_rows if str(row.get("status") or "").lower() in DEPRECATED_STATUSES
        ]
        deprecated_suppressed = [
            row for row in deprecated_claims if str(row.get("suppressed_reason") or "") == "deprecated"
        ]
        deprecated_refs = {
            str(ref)
            for row in deprecated_claims
            for ref in list(row.get("source_refs") or [])
            if str(ref).strip()
        }
        active_refs = {
            str(ref)
            for row in claim_rows
            if row.get("kept_in_answer_context")
            for ref in list(row.get("source_refs") or [])
            if str(ref).strip()
        }
        deprecated_only_support = any(ref in deprecated_refs and ref not in active_refs for ref in support_refs)
        deprecated_surface_hit = _deprecated_surface_hit(answer_text, deprecated_claims)
        deprecated_leakage = bool(deprecated_only_support or deprecated_surface_hit)
        obsolete_risk = bool(deprecated_leakage)
        metrics = _as_dict(sample_row.get("metrics"))
        stage = _failure_stage(
            sample_row,
            gold,
            event,
            source_supported_proxy=source_supported_proxy,
            unsupported_answer_risk=unsupported_answer_risk,
            memory_index=memory_index,
        )
        audit_token_estimate = estimate_context_tokens(
            "\n".join(
                [
                    str(sample_row.get("question") or ""),
                    answer_text,
                    str(sample_row.get("gold_answer") or ""),
                    "\n".join(str(row.get("item_text_preview") or "") for row in support_rows),
                    "\n".join(str(row.get("claim_text_preview") or "") for row in claim_rows if row.get("kept_in_answer_context")),
                ]
            )
        )
        rows.append(
            {
                "schema_version": "auditability_row_v2",
                "sample_id": sample_id,
                "query_task_id": query_task_id,
                "judge_verdict": sample_row.get("judge_verdict"),
                "observed_answer_available": observed_answer_available,
                "answer_abstained": answer_abstained,
                "answer_support_ref_count": len(support_refs),
                "answer_support_refs": support_refs,
                "invalid_supporting_refs": invalid_refs,
                "gold_source_refs": gold_refs,
                "support_gold_ref_overlap_count": len(support_gold_overlap),
                "support_gold_ref_coverage": _safe_rate(len(support_gold_overlap), len(gold_refs)),
                "retrieved_gold_ref_coverage": _safe_rate(len(retrieved_gold_overlap), len(gold_refs)),
                "stored_gold_ref_coverage": _gold_ref_coverage(
                    gold_refs,
                    sample_raw_refs,
                ),
                "trajectory_linked_gold_ref_coverage": _gold_ref_coverage(
                    gold_refs,
                    trajectory_linked_refs,
                ),
                "page_routed_gold_ref_coverage": _gold_ref_coverage(
                    gold_refs,
                    page_routed_refs,
                ),
                "trajectory_selected_gold_ref_coverage": _gold_ref_coverage(
                    gold_refs,
                    selected_trajectory_refs,
                ),
                "snapshot_expanded_gold_ref_coverage": _gold_ref_coverage(
                    gold_refs,
                    expanded_refs,
                ),
                "source_supported_proxy": source_supported_proxy,
                "unsupported_answer_risk": unsupported_answer_risk,
                "unsupported_reason": unsupported_reason,
                "answer_postcheck_issue": issue,
                "audit_packet_item_count": len(support_rows) + len(claim_rows),
                "audit_packet_source_count": len(set(support_refs) | set(retrieved_refs)),
                "audit_packet_claim_count": len(claim_rows),
                "audit_packet_estimated_tokens": audit_token_estimate,
                "failure_localization_stage": stage,
                "conflict_claim_visible": conflict_claim_visible,
                "conflict_block_visible": conflict_block_visible,
                "bridge_finalization_conflicted": bool(answer_metadata.get("bridge_finalization_conflicted")),
                "bridge_finalization_action": answer_metadata.get("bridge_finalization_action"),
                "deprecated_claim_count": len(deprecated_claims),
                "deprecated_claim_suppressed_count": len(deprecated_suppressed),
                "deprecated_source_ref_leakage_proxy": deprecated_leakage,
                "obsolete_answer_risk_proxy": obsolete_risk,
                "observed_f1": metrics.get("F1"),
                "observed_judge_acc": metrics.get("judge_acc"),
                "token_estimator": TOKEN_ESTIMATOR_NAME,
            }
        )
    return rows


def _error_propagation_funnel_rows(
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_fields = [
        ("source_stored", "stored_gold_ref_coverage"),
        ("linked_to_trajectory", "trajectory_linked_gold_ref_coverage"),
        ("reachable_from_selected_pages", "page_routed_gold_ref_coverage"),
        ("trajectory_selected", "trajectory_selected_gold_ref_coverage"),
        ("snapshot_expanded", "snapshot_expanded_gold_ref_coverage"),
        ("source_in_final_context", "retrieved_gold_ref_coverage"),
        ("used_in_final_support", "support_gold_ref_coverage"),
    ]
    eligible = [row for row in audit_rows if list(row.get("gold_source_refs") or [])]

    def observed_correct(row: dict[str, Any]) -> bool:
        verdict = str(row.get("judge_verdict") or "").strip().lower()
        if verdict:
            return verdict == "correct"
        return float(row.get("observed_judge_acc") or 0.0) >= 1.0

    output: list[dict[str, Any]] = []
    for stage, field in stage_fields:
        observable = [
            row for row in eligible if row.get(field) is not None
        ]
        any_hit = [row for row in observable if float(row.get(field) or 0.0) > 0.0]
        all_hit = [row for row in observable if float(row.get(field) or 0.0) >= 1.0]
        output.append(
            {
                "stage": stage,
                "coverage_field": field,
                "eligible_query_count": len(eligible),
                "observable_query_count": len(observable),
                "mean_gold_ref_coverage": _safe_mean(
                    row.get(field) for row in observable
                ),
                "any_gold_ref_hit_count": len(any_hit),
                "any_gold_ref_hit_rate": _safe_rate(len(any_hit), len(observable)),
                "all_gold_refs_hit_count": len(all_hit),
                "all_gold_refs_hit_rate": _safe_rate(len(all_hit), len(observable)),
                "observed_judge_correct_rate_given_any_hit": _safe_rate(
                    sum(1 for row in any_hit if observed_correct(row)),
                    len(any_hit),
                ),
                "observed_judge_correct_rate_given_all_hit": _safe_rate(
                    sum(1 for row in all_hit if observed_correct(row)),
                    len(all_hit),
                ),
            }
        )
    return output


def _deprecated_surface_hit(answer_text: str, deprecated_claims: list[dict[str, Any]]) -> bool:
    normalized_answer = " ".join(str(answer_text or "").casefold().split())
    if not normalized_answer:
        return False
    for claim in deprecated_claims:
        preview = str(claim.get("claim_text_preview") or "").strip()
        if len(preview) < 12:
            continue
        surface = " ".join(preview.casefold().split())
        if len(surface) >= 12 and surface in normalized_answer:
            return True
    return False


def build_audit_packet_rows(
    *,
    sample_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
    retrieval_events: dict[str, dict[str, Any]],
    answer_support_rows: list[dict[str, Any]],
    answer_context_claim_rows: list[dict[str, Any]],
    packet_save_mode: str = "summary",
) -> list[dict[str, Any]]:
    support_by_query = _support_rows_by_query(answer_support_rows)
    claims_by_query = _context_claims_by_query(answer_context_claim_rows)
    raw_messages_by_id = memory_index.get("raw_messages_by_id", {})
    rows: list[dict[str, Any]] = []
    include_previews = str(packet_save_mode or "summary") == "compact"
    for sample_row in sample_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        event = retrieval_events.get(str(sample_row.get("retrieval_event_id") or ""), {})
        support_rows = support_by_query.get(query_task_id, [])
        claim_rows = claims_by_query.get(query_task_id, [])
        support_refs = _dedupe_str(
            ref for support_row in support_rows for ref in list(support_row.get("source_refs") or [])
        )
        retrieved_refs = _dedupe_str(sample_row.get("retrieval_source_refs") or event.get("source_refs") or [])
        audit_refs = support_refs or retrieved_refs
        sample_id = str(sample_row.get("sample_id") or "")
        source_message_ids = source_message_ids_for_refs(
            memory_index,
            sample_id=sample_id,
            source_refs=audit_refs,
        )
        source_texts = [
            str(raw_messages_by_id[message_id].get("content") or "")
            for message_id in source_message_ids
            if message_id in raw_messages_by_id
        ]
        kept_claims = [row for row in claim_rows if row.get("kept_in_answer_context")]
        token_estimate = estimate_context_tokens(
            "\n".join(
                [
                    str(sample_row.get("question") or ""),
                    str(sample_row.get("answer_text") or ""),
                    str(sample_row.get("gold_answer") or ""),
                    "\n".join(source_texts),
                    "\n".join(str(row.get("claim_text_preview") or "") for row in kept_claims),
                ]
            )
        )
        row = {
            "schema_version": "audit_packet_v2",
            "contains_sensitive_text": True,
            "sample_id": sample_id,
            "query_task_id": query_task_id,
            "question": sample_row.get("question"),
            "answer_preview": _compact_text(sample_row.get("answer_text"), limit=260),
            "gold_answer_preview": _compact_text(sample_row.get("gold_answer"), limit=260),
            "retrieved_refs": retrieved_refs,
            "support_refs": support_refs,
            "claim_ids": _dedupe_str(row.get("claim_id") for row in kept_claims),
            "conflict_claim_count": sum(
                1 for row in kept_claims if str(row.get("status") or "").lower() in CONFLICT_STATUSES
            ),
            "deprecated_claim_count": sum(
                1 for row in claim_rows if str(row.get("status") or "").lower() in DEPRECATED_STATUSES
            ),
            "estimated_audit_tokens": token_estimate,
            "audit_item_count": len(source_message_ids) + len(kept_claims),
            "audit_source_count": len(source_message_ids),
            "audit_claim_count": len(kept_claims),
            "token_estimator": TOKEN_ESTIMATOR_NAME,
        }
        if include_previews:
            row["source_message_previews"] = [
                {
                    "message_id": message_id,
                    "source_ref": raw_messages_by_id.get(message_id, {}).get("source_ref"),
                    "text_preview": _compact_text(raw_messages_by_id.get(message_id, {}).get("content"), limit=220),
                }
                for message_id in source_message_ids
                if message_id in raw_messages_by_id
            ]
            row["claim_previews"] = [
                {
                    "claim_id": claim.get("claim_id"),
                    "status": claim.get("status"),
                    "source_refs": claim.get("source_refs"),
                    "text_preview": claim.get("claim_text_preview"),
                }
                for claim in kept_claims
            ]
        rows.append(row)
    return rows


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "supported", "correct"}:
        return True
    if text in {"false", "0", "no", "n", "unsupported", "incorrect"}:
        return False
    return None


def load_audit_labels(path: Path | str | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Audit labels not found: {resolved}")
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    bool_fields = {
        "true_source_supported",
        "true_conflict_required",
        "true_conflict_handled",
        "true_obsolete_required",
        "true_obsolete_handled",
        "human_audit_supported_decision",
        "human_audit_correct",
    }
    if resolved.suffix.lower() == ".csv":
        with resolved.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
    else:
        source_rows = _read_jsonl(resolved)
    for row in source_rows:
        label = dict(row)
        for field in bool_fields:
            if field in label:
                label[field] = _parse_bool(label.get(field))
        if label.get("human_audit_seconds") not in {None, ""}:
            label["human_audit_seconds"] = float(label["human_audit_seconds"])
        key = (str(label.get("sample_id") or ""), str(label.get("query_task_id") or ""))
        labels[key] = label
    return labels


def _source_support_table_rows(
    audit_rows: list[dict[str, Any]],
    baselines: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        if baseline != "trajwiki_observed":
            output.append(
                {
                    "method": baseline,
                    "result_type": "audit_packet_context_proxy",
                    "query_count": 0,
                    "source_supported_answer_rate_proxy": None,
                    "mean_support_gold_ref_coverage": None,
                    "mean_retrieved_gold_ref_coverage": None,
                    "invalid_supporting_ref_rate": None,
                    "observed_judge_acc_if_available": None,
                    "observed_f1_if_available": None,
                }
            )
            continue
        output.append(
            {
                "method": baseline,
                "result_type": "observed_answer_proxy",
                "query_count": len(audit_rows),
                "source_supported_answer_rate_proxy": _safe_rate(
                    sum(1 for row in audit_rows if row.get("source_supported_proxy") is True),
                    len(audit_rows),
                ),
                "mean_support_gold_ref_coverage": _safe_mean(
                    row.get("support_gold_ref_coverage") for row in audit_rows
                ),
                "mean_retrieved_gold_ref_coverage": _safe_mean(
                    row.get("retrieved_gold_ref_coverage") for row in audit_rows
                ),
                "invalid_supporting_ref_rate": _safe_rate(
                    sum(1 for row in audit_rows if row.get("invalid_supporting_refs")),
                    len(audit_rows),
                ),
                "observed_judge_acc_if_available": _safe_mean(
                    row.get("observed_judge_acc") for row in audit_rows
                ),
                "observed_f1_if_available": _safe_mean(row.get("observed_f1") for row in audit_rows),
            }
        )
    return output


def _unsupported_table_rows(audit_rows: list[dict[str, Any]], baselines: list[str]) -> list[dict[str, Any]]:
    reason_counts = Counter(str(row.get("unsupported_reason") or "") for row in audit_rows)
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        if baseline != "trajwiki_observed":
            output.append(
                {
                    "method": baseline,
                    "query_count": 0,
                    "unsupported_answer_risk_rate": None,
                    "unsupported_extra_items_count": None,
                    "unsupported_count_count": None,
                    "no_support_refs_count": None,
                    "invalid_supporting_ref_count": None,
                    "no_gold_overlap_count": None,
                }
            )
            continue
        output.append(
            {
                "method": baseline,
                "query_count": len(audit_rows),
                "unsupported_answer_risk_rate": _safe_rate(
                    sum(1 for row in audit_rows if row.get("unsupported_answer_risk") is True),
                    len(audit_rows),
                ),
                "unsupported_extra_items_count": reason_counts["unsupported_extra_items"],
                "unsupported_count_count": reason_counts["unsupported_count"],
                "no_support_refs_count": reason_counts["non_abstain_no_valid_support_refs"],
                "invalid_supporting_ref_count": reason_counts["invalid_supporting_refs"],
                "no_gold_overlap_count": reason_counts["support_refs_no_gold_overlap"],
            }
        )
    return output


def _binary_metric_rows(
    predictions: dict[tuple[str, str], bool],
    labels: dict[tuple[str, str], dict[str, Any]],
    label_field: str,
) -> dict[str, Any]:
    comparable = [
        (prediction, labels[key].get(label_field))
        for key, prediction in predictions.items()
        if key in labels and labels[key].get(label_field) is not None
    ]
    if not comparable:
        return {"accuracy_if_labeled": None, "precision_if_labeled": None, "recall_if_labeled": None, "f1_if_labeled": None}
    tp = sum(1 for prediction, truth in comparable if prediction is True and truth is True)
    fp = sum(1 for prediction, truth in comparable if prediction is True and truth is False)
    fn = sum(1 for prediction, truth in comparable if prediction is False and truth is True)
    correct = sum(1 for prediction, truth in comparable if bool(prediction) == bool(truth))
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "accuracy_if_labeled": _safe_rate(correct, len(comparable)),
        "precision_if_labeled": precision,
        "recall_if_labeled": recall,
        "f1_if_labeled": f1,
    }


def _failure_localization_rows(
    audit_rows: list[dict[str, Any]],
    labels: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_counts = Counter(str(row.get("failure_localization_stage") or "unknown") for row in audit_rows)
    labeled_pairs = [
        (
            str(row.get("failure_localization_stage") or "unknown"),
            str(labels.get((str(row.get("sample_id") or ""), str(row.get("query_task_id") or "")), {}).get("true_failure_stage") or ""),
        )
        for row in audit_rows
        if labels.get((str(row.get("sample_id") or ""), str(row.get("query_task_id") or "")), {}).get("true_failure_stage")
    ]
    overall_accuracy = _safe_rate(
        sum(1 for predicted, truth in labeled_pairs if predicted == truth),
        len(labeled_pairs),
    )
    rows: list[dict[str, Any]] = []
    for stage, count in sorted(stage_counts.items()):
        if labeled_pairs:
            tp = sum(1 for predicted, truth in labeled_pairs if predicted == stage and truth == stage)
            fp = sum(1 for predicted, truth in labeled_pairs if predicted == stage and truth != stage)
            fn = sum(1 for predicted, truth in labeled_pairs if predicted != stage and truth == stage)
            precision = _safe_rate(tp, tp + fp)
            recall = _safe_rate(tp, tp + fn)
            f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
        else:
            precision = recall = f1 = None
        rows.append(
            {
                "stage": stage,
                "query_count": count,
                "query_rate": _safe_rate(count, len(audit_rows)),
                "precision_if_labeled": precision,
                "recall_if_labeled": recall,
                "f1_if_labeled": f1,
                "accuracy_if_labeled": overall_accuracy,
            }
        )
    return rows


def _conflict_obsolete_rows(
    audit_rows: list[dict[str, Any]],
    labels: dict[tuple[str, str], dict[str, Any]],
    baselines: list[str],
) -> list[dict[str, Any]]:
    conflict_predictions = {
        (str(row.get("sample_id") or ""), str(row.get("query_task_id") or "")): bool(
            row.get("conflict_claim_visible") or row.get("conflict_block_visible")
        )
        for row in audit_rows
    }
    obsolete_risk_predictions = {
        (str(row.get("sample_id") or ""), str(row.get("query_task_id") or "")): bool(
            row.get("obsolete_answer_risk_proxy")
        )
        for row in audit_rows
    }
    conflict_detection_metrics = _binary_metric_rows(
        conflict_predictions,
        labels,
        "true_conflict_required",
    )
    obsolete_safe_predictions = {
        key: not risk
        for key, risk in obsolete_risk_predictions.items()
        if labels.get(key, {}).get("true_obsolete_required") is True
    }
    obsolete_proxy_metrics = _binary_metric_rows(
        obsolete_safe_predictions,
        labels,
        "true_obsolete_handled",
    )
    conflict_handled_labels = [
        bool(label.get("true_conflict_handled"))
        for label in labels.values()
        if label.get("true_conflict_required") is True
        and label.get("true_conflict_handled") is not None
    ]
    obsolete_handled_labels = [
        bool(label.get("true_obsolete_handled"))
        for label in labels.values()
        if label.get("true_obsolete_required") is True
        and label.get("true_obsolete_handled") is not None
    ]
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        if baseline != "trajwiki_observed":
            output.append(
                {
                    "method": baseline,
                    "query_count": 0,
                    "conflict_exposed_rate": None,
                    "bridge_conflicted_count": None,
                    "deprecated_suppression_rate": None,
                    "deprecated_leakage_proxy_rate": None,
                    "obsolete_answer_risk_proxy_rate": None,
                    "conflict_accuracy_if_labeled": None,
                    "obsolete_accuracy_if_labeled": None,
                    "conflict_detection_f1_if_labeled": None,
                    "conflict_handled_rate_if_labeled": None,
                    "obsolete_handled_rate_if_labeled": None,
                    "obsolete_proxy_f1_if_labeled": None,
                }
            )
            continue
        deprecated_total = sum(int(row.get("deprecated_claim_count") or 0) for row in audit_rows)
        deprecated_suppressed = sum(int(row.get("deprecated_claim_suppressed_count") or 0) for row in audit_rows)
        output.append(
            {
                "method": baseline,
                "query_count": len(audit_rows),
                "conflict_exposed_rate": _safe_rate(
                    sum(1 for row in audit_rows if row.get("conflict_claim_visible") or row.get("conflict_block_visible")),
                    len(audit_rows),
                ),
                "bridge_conflicted_count": sum(1 for row in audit_rows if row.get("bridge_finalization_conflicted")),
                "deprecated_suppression_rate": _safe_rate(deprecated_suppressed, deprecated_total),
                "deprecated_leakage_proxy_rate": _safe_rate(
                    sum(1 for row in audit_rows if row.get("deprecated_source_ref_leakage_proxy") is True),
                    len(audit_rows),
                ),
                "obsolete_answer_risk_proxy_rate": _safe_rate(
                    sum(1 for row in audit_rows if row.get("obsolete_answer_risk_proxy") is True),
                    len(audit_rows),
                ),
                "conflict_accuracy_if_labeled": conflict_detection_metrics.get(
                    "accuracy_if_labeled"
                ),
                "obsolete_accuracy_if_labeled": obsolete_proxy_metrics.get(
                    "accuracy_if_labeled"
                ),
                "conflict_detection_f1_if_labeled": conflict_detection_metrics.get(
                    "f1_if_labeled"
                ),
                "conflict_handled_rate_if_labeled": (
                    _safe_mean(conflict_handled_labels)
                    if conflict_handled_labels
                    else None
                ),
                "obsolete_handled_rate_if_labeled": (
                    _safe_mean(obsolete_handled_labels)
                    if obsolete_handled_labels
                    else None
                ),
                "obsolete_proxy_f1_if_labeled": obsolete_proxy_metrics.get(
                    "f1_if_labeled"
                ),
            }
        )
    return output


def _load_offline_proxy_rows(analysis_dir: Path, variant: str) -> list[dict[str, Any]]:
    rows = [row for row in _read_jsonl(analysis_dir / "offline_ablation_rows.jsonl") if row.get("variant") == variant]
    if not rows:
        return []
    cutoff_rows = [row for row in rows if int(row.get("rank_cutoff") or 0) == 15]
    if cutoff_rows:
        rows = cutoff_rows
    max_budget = max(int(row.get("budget_tokens") or 0) for row in rows)
    return [row for row in rows if int(row.get("budget_tokens") or 0) == max_budget]


def _audit_packet_cost_rows(
    *,
    baselines: list[str],
    audit_packet_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
    analysis_dir: Path,
    labels: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        if baseline == "trajwiki_observed":
            rows = audit_packet_rows
            seconds = [
                labels.get((str(row.get("sample_id") or ""), str(row.get("query_task_id") or "")), {}).get(
                    "human_audit_seconds"
                )
                for row in rows
            ]
            human_correct = [
                labels.get((str(row.get("sample_id") or ""), str(row.get("query_task_id") or "")), {}).get(
                    "human_audit_correct"
                )
                for row in rows
                if labels.get((str(row.get("sample_id") or ""), str(row.get("query_task_id") or "")), {}).get(
                    "human_audit_correct"
                )
                is not None
            ]
            output.append(
                {
                    "method": baseline,
                    "availability": "observed",
                    "query_count": len(rows),
                    "mean_audit_source_count": _safe_mean(row.get("audit_source_count") for row in rows),
                    "median_audit_source_count": _safe_median(row.get("audit_source_count") for row in rows),
                    "mean_audit_claim_count": _safe_mean(row.get("audit_claim_count") for row in rows),
                    "mean_audit_packet_tokens": _safe_mean(row.get("estimated_audit_tokens") for row in rows),
                    "median_audit_packet_tokens": _safe_median(row.get("estimated_audit_tokens") for row in rows),
                    "human_audit_seconds_mean_if_available": _safe_mean(seconds),
                    "human_audit_error_rate_if_available": (
                        _safe_rate(sum(1 for value in human_correct if value is False), len(human_correct))
                        if human_correct
                        else None
                    ),
                }
            )
        elif baseline == "full_context_proxy":
            source_counts: list[int] = []
            token_counts: list[int] = []
            for row in sample_rows:
                messages = list(memory_index.get("sample_raw_messages", {}).get(str(row.get("sample_id") or ""), []))
                source_counts.append(sum(1 for message in messages if message.get("source_ref")))
                token_counts.append(sum(estimate_context_tokens(message.get("content", "")) for message in messages))
            output.append(
                {
                    "method": baseline,
                    "availability": "context_proxy",
                    "query_count": len(sample_rows),
                    "mean_audit_source_count": _safe_mean(source_counts),
                    "median_audit_source_count": _safe_median(source_counts),
                    "mean_audit_claim_count": 0.0,
                    "mean_audit_packet_tokens": _safe_mean(token_counts),
                    "median_audit_packet_tokens": _safe_median(token_counts),
                    "human_audit_seconds_mean_if_available": None,
                    "human_audit_error_rate_if_available": None,
                }
            )
        else:
            proxy_rows = _load_offline_proxy_rows(analysis_dir, baseline)
            output.append(
                {
                    "method": baseline,
                    "availability": "context_proxy" if proxy_rows else "not_available",
                    "query_count": len(proxy_rows),
                    "mean_audit_source_count": _safe_mean(
                        len(list(row.get("selected_source_refs") or [])) for row in proxy_rows
                    ),
                    "median_audit_source_count": _safe_median(
                        len(list(row.get("selected_source_refs") or [])) for row in proxy_rows
                    ),
                    "mean_audit_claim_count": None,
                    "mean_audit_packet_tokens": _safe_mean(row.get("estimated_context_tokens") for row in proxy_rows),
                    "median_audit_packet_tokens": _safe_median(row.get("estimated_context_tokens") for row in proxy_rows),
                    "human_audit_seconds_mean_if_available": None,
                    "human_audit_error_rate_if_available": None,
                }
            )
    return output


def _audit_examples(audit_rows: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in audit_rows:
        reason = None
        if row.get("unsupported_answer_risk"):
            reason = "unsupported_answer_risk"
        elif row.get("deprecated_source_ref_leakage_proxy"):
            reason = "deprecated_leakage_proxy"
        elif row.get("conflict_claim_visible") or row.get("conflict_block_visible"):
            reason = "conflict_visible"
        elif row.get("support_gold_ref_coverage") is not None and float(row.get("support_gold_ref_coverage") or 0.0) < 1.0:
            reason = "low_support_gold_overlap"
        if reason is None:
            continue
        examples.append(
            {
                "schema_version": "audit_example_v1",
                "example_type": reason,
                "sample_id": row.get("sample_id"),
                "query_task_id": row.get("query_task_id"),
                "failure_localization_stage": row.get("failure_localization_stage"),
                "support_gold_ref_coverage": row.get("support_gold_ref_coverage"),
                "answer_support_refs": row.get("answer_support_refs"),
                "gold_source_refs": row.get("gold_source_refs"),
                "unsupported_reason": row.get("unsupported_reason"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _write_analysis_tables(
    *,
    analysis_dir: Path,
    baselines: list[str],
    audit_rows: list[dict[str, Any]],
    audit_packet_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
    labels: dict[tuple[str, str], dict[str, Any]],
    run_meta: dict[str, Any],
    sampling_scope: dict[str, Any] | None = None,
) -> dict[str, str]:
    source_support_rows = _source_support_table_rows(audit_rows, baselines)
    unsupported_rows = _unsupported_table_rows(audit_rows, baselines)
    failure_rows = _failure_localization_rows(audit_rows, labels)
    conflict_rows = _conflict_obsolete_rows(audit_rows, labels, baselines)
    packet_cost_rows = _audit_packet_cost_rows(
        baselines=baselines,
        audit_packet_rows=audit_packet_rows,
        sample_rows=sample_rows,
        memory_index=memory_index,
        analysis_dir=analysis_dir,
        labels=labels,
    )
    funnel_rows = _error_propagation_funnel_rows(audit_rows)
    examples = _audit_examples(audit_rows)

    _write_csv(
        analysis_dir / "source_support_table.csv",
        source_support_rows,
        [
            "method",
            "result_type",
            "query_count",
            "source_supported_answer_rate_proxy",
            "mean_support_gold_ref_coverage",
            "mean_retrieved_gold_ref_coverage",
            "invalid_supporting_ref_rate",
            "observed_judge_acc_if_available",
            "observed_f1_if_available",
        ],
    )
    _write_csv(
        analysis_dir / "unsupported_answer_table.csv",
        unsupported_rows,
        [
            "method",
            "query_count",
            "unsupported_answer_risk_rate",
            "unsupported_extra_items_count",
            "unsupported_count_count",
            "no_support_refs_count",
            "invalid_supporting_ref_count",
            "no_gold_overlap_count",
        ],
    )
    _write_csv(
        analysis_dir / "failure_localization_table.csv",
        failure_rows,
        [
            "stage",
            "query_count",
            "query_rate",
            "precision_if_labeled",
            "recall_if_labeled",
            "f1_if_labeled",
            "accuracy_if_labeled",
        ],
    )
    _write_csv(
        analysis_dir / "conflict_obsolete_table.csv",
        conflict_rows,
        [
            "method",
            "query_count",
            "conflict_exposed_rate",
            "bridge_conflicted_count",
            "deprecated_suppression_rate",
            "deprecated_leakage_proxy_rate",
            "obsolete_answer_risk_proxy_rate",
            "conflict_accuracy_if_labeled",
            "obsolete_accuracy_if_labeled",
            "conflict_detection_f1_if_labeled",
            "conflict_handled_rate_if_labeled",
            "obsolete_handled_rate_if_labeled",
            "obsolete_proxy_f1_if_labeled",
        ],
    )
    _write_csv(
        analysis_dir / "audit_packet_cost.csv",
        packet_cost_rows,
        [
            "method",
            "availability",
            "query_count",
            "mean_audit_source_count",
            "median_audit_source_count",
            "mean_audit_claim_count",
            "mean_audit_packet_tokens",
            "median_audit_packet_tokens",
            "human_audit_seconds_mean_if_available",
            "human_audit_error_rate_if_available",
        ],
    )
    _write_csv(
        analysis_dir / "error_propagation_funnel.csv",
        funnel_rows,
        [
            "stage",
            "coverage_field",
            "eligible_query_count",
            "observable_query_count",
            "mean_gold_ref_coverage",
            "any_gold_ref_hit_count",
            "any_gold_ref_hit_rate",
            "all_gold_refs_hit_count",
            "all_gold_refs_hit_rate",
            "observed_judge_correct_rate_given_any_hit",
            "observed_judge_correct_rate_given_all_hit",
        ],
    )
    _write_jsonl_replace(analysis_dir / "audit_examples.jsonl", examples)
    write_json(
        analysis_dir / "auditability_summary.json",
        {
            "schema_version": "auditability_summary_v2",
            "diagnostic_mode": "offline_auditability_interpretability",
            "run_id": run_meta.get("run_id"),
            "baselines": baselines,
            "query_count": len(audit_rows),
            "label_count": len(labels),
            "sampling_scope": sampling_scope,
            "token_estimator": TOKEN_ESTIMATOR_NAME,
            "source_supported_answer_rate_proxy": _safe_rate(
                sum(1 for row in audit_rows if row.get("source_supported_proxy") is True),
                len(audit_rows),
            ),
            "unsupported_answer_risk_rate": _safe_rate(
                sum(1 for row in audit_rows if row.get("unsupported_answer_risk") is True),
                len(audit_rows),
            ),
            "labeled_metric_definitions": {
                "conflict_accuracy_if_labeled": (
                    "Accuracy of conflict-visible diagnostics against "
                    "true_conflict_required."
                ),
                "conflict_handled_rate_if_labeled": (
                    "Human-labeled true_conflict_handled rate among rows where "
                    "true_conflict_required is true."
                ),
                "obsolete_accuracy_if_labeled": (
                    "Accuracy of the inverse obsolete-risk proxy against "
                    "true_obsolete_handled, restricted to required cases."
                ),
                "obsolete_handled_rate_if_labeled": (
                    "Human-labeled true_obsolete_handled rate among rows where "
                    "true_obsolete_required is true."
                ),
            },
            "paths": {
                "source_support_table": str(analysis_dir / "source_support_table.csv"),
                "unsupported_answer_table": str(analysis_dir / "unsupported_answer_table.csv"),
                "failure_localization_table": str(analysis_dir / "failure_localization_table.csv"),
                "conflict_obsolete_table": str(analysis_dir / "conflict_obsolete_table.csv"),
                "audit_packet_cost": str(analysis_dir / "audit_packet_cost.csv"),
                "error_propagation_funnel": str(
                    analysis_dir / "error_propagation_funnel.csv"
                ),
                "audit_examples": str(analysis_dir / "audit_examples.jsonl"),
                "answer_support_rows": str(analysis_dir / "answer_support_rows.jsonl"),
                "answer_context_claim_rows": str(analysis_dir / "answer_context_claim_rows.jsonl"),
                "claim_lifecycle_rows": str(analysis_dir / "claim_lifecycle_rows.jsonl"),
                "audit_packet_rows": str(analysis_dir / "audit_packet_rows.jsonl"),
                "auditability_rows": str(analysis_dir / "auditability_rows.jsonl"),
            },
        },
    )
    return {
        "source_support_table_path": str(analysis_dir / "source_support_table.csv"),
        "unsupported_answer_table_path": str(analysis_dir / "unsupported_answer_table.csv"),
        "failure_localization_table_path": str(analysis_dir / "failure_localization_table.csv"),
        "conflict_obsolete_table_path": str(analysis_dir / "conflict_obsolete_table.csv"),
        "audit_packet_cost_path": str(analysis_dir / "audit_packet_cost.csv"),
        "error_propagation_funnel_path": str(
            analysis_dir / "error_propagation_funnel.csv"
        ),
        "audit_examples_path": str(analysis_dir / "audit_examples.jsonl"),
        "summary_path": str(analysis_dir / "auditability_summary.json"),
    }


def _build_runtime_artifacts(
    *,
    sample_rows: list[dict[str, Any]],
    database_path: Path,
    run_dir: Path,
    packet_save_mode: str,
) -> dict[str, Any]:
    memory_index = build_memory_index(database_path)
    retrieval_events = _load_retrieval_events(database_path, memory_index)
    answer_support_rows = build_answer_support_rows(sample_rows, memory_index)
    answer_context_claim_rows = build_answer_context_claim_rows(sample_rows, memory_index, retrieval_events)
    claim_lifecycle_rows = build_claim_lifecycle_rows(database_path, memory_index)
    audit_packet_rows = build_audit_packet_rows(
        sample_rows=sample_rows,
        memory_index=memory_index,
        retrieval_events=retrieval_events,
        answer_support_rows=answer_support_rows,
        answer_context_claim_rows=answer_context_claim_rows,
        packet_save_mode=packet_save_mode,
    )
    analysis_dir = versioned_analysis_path(
        run_dir,
        filename="audit_packet_rows.jsonl",
        accepted_schema_versions={"audit_packet_v2"},
    ).parent
    _write_jsonl_replace(analysis_dir / "answer_support_rows.jsonl", answer_support_rows)
    _write_jsonl_replace(analysis_dir / "answer_context_claim_rows.jsonl", answer_context_claim_rows)
    _write_jsonl_replace(analysis_dir / "claim_lifecycle_rows.jsonl", claim_lifecycle_rows)
    _write_jsonl_replace(analysis_dir / "audit_packet_rows.jsonl", audit_packet_rows)
    return {
        "memory_index": memory_index,
        "retrieval_events": retrieval_events,
        "answer_support_rows": answer_support_rows,
        "answer_context_claim_rows": answer_context_claim_rows,
        "claim_lifecycle_rows": claim_lifecycle_rows,
        "audit_packet_rows": audit_packet_rows,
    }


def write_auditability_artifacts(
    *,
    run_dir: Path,
    sample_rows: list[dict[str, Any]],
    database_path: Path,
    packet_save_mode: str = "summary",
) -> dict[str, str]:
    artifacts = _build_runtime_artifacts(
        sample_rows=sample_rows,
        database_path=database_path,
        run_dir=run_dir,
        packet_save_mode=packet_save_mode,
    )
    analysis_dir = versioned_analysis_path(
        run_dir,
        filename="audit_packet_rows.jsonl",
        accepted_schema_versions={"audit_packet_v2"},
    ).parent
    write_json(
        analysis_dir / "auditability_summary.json",
        {
            "schema_version": "auditability_summary_v2",
            "diagnostic_mode": "runtime_auditability_artifacts",
            "query_count": len(sample_rows),
            "answer_support_row_count": len(artifacts["answer_support_rows"]),
            "answer_context_claim_row_count": len(artifacts["answer_context_claim_rows"]),
            "claim_lifecycle_row_count": len(artifacts["claim_lifecycle_rows"]),
            "audit_packet_row_count": len(artifacts["audit_packet_rows"]),
            "packet_save_mode": packet_save_mode,
            "token_estimator": TOKEN_ESTIMATOR_NAME,
            "paths": {
                "answer_support_rows": str(analysis_dir / "answer_support_rows.jsonl"),
                "answer_context_claim_rows": str(analysis_dir / "answer_context_claim_rows.jsonl"),
                "claim_lifecycle_rows": str(analysis_dir / "claim_lifecycle_rows.jsonl"),
                "audit_packet_rows": str(analysis_dir / "audit_packet_rows.jsonl"),
                "auditability_summary": str(analysis_dir / "auditability_summary.json"),
            },
        },
    )
    return {
        "answer_support_rows": str(analysis_dir / "answer_support_rows.jsonl"),
        "answer_context_claim_rows": str(analysis_dir / "answer_context_claim_rows.jsonl"),
        "claim_lifecycle_rows": str(analysis_dir / "claim_lifecycle_rows.jsonl"),
        "audit_packet_rows": str(analysis_dir / "audit_packet_rows.jsonl"),
        "auditability_summary": str(analysis_dir / "auditability_summary.json"),
    }


def analyze_auditability(
    run_path: Path | str,
    *,
    baselines: str | Iterable[str] | None = None,
    audit_labels_path: Path | str | None = None,
    sampling_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_path).expanduser().resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    run_meta, sample_rows, database_path = load_details_rows(run_dir)
    if str(run_meta.get("dataset")).lower() != "locomo":
        raise ValueError("Auditability analysis currently supports LOCOMO runs only.")
    all_sample_rows = sample_rows
    scope = load_query_scope(sampling_manifest_path)
    if scope is not None:
        validate_scope_against_rows(
            scope,
            all_sample_rows,
            run_dir=run_dir,
        )
    selected_baselines = parse_str_list(baselines, DEFAULT_BASELINES)
    primary_dir = run_dir / "analysis"
    v2_dir = run_dir / "analysis_v2"
    runtime_analysis_dir = (
        v2_dir if (v2_dir / "audit_packet_rows.jsonl").exists() else primary_dir
    )
    memory_index = build_memory_index(database_path)
    retrieval_events = _load_retrieval_events(database_path, memory_index)

    answer_support_rows = _read_jsonl(runtime_analysis_dir / "answer_support_rows.jsonl")
    answer_context_claim_rows = _read_jsonl(
        runtime_analysis_dir / "answer_context_claim_rows.jsonl"
    )
    claim_lifecycle_rows = _read_jsonl(
        runtime_analysis_dir / "claim_lifecycle_rows.jsonl"
    )
    audit_packet_rows = _read_jsonl(runtime_analysis_dir / "audit_packet_rows.jsonl")
    packet_versions = {str(row.get("schema_version") or "") for row in audit_packet_rows}
    expected_query_ids = {
        str(row.get("query_task_id") or "")
        for row in all_sample_rows
        if str(row.get("query_task_id") or "")
    }

    def complete_query_rows(rows: list[dict[str, Any]]) -> bool:
        query_ids = [
            str(row.get("query_task_id") or "")
            for row in rows
            if str(row.get("query_task_id") or "")
        ]
        return len(query_ids) == len(set(query_ids)) and set(query_ids) == expected_query_ids

    if (
        not complete_query_rows(answer_support_rows)
        or not complete_query_rows(audit_packet_rows)
        or packet_versions != {"audit_packet_v2"}
    ):
        runtime = _build_runtime_artifacts(
            sample_rows=all_sample_rows,
            database_path=database_path,
            run_dir=run_dir,
            packet_save_mode="summary",
        )
        runtime_analysis_dir = (
            v2_dir if (v2_dir / "audit_packet_rows.jsonl").exists() else primary_dir
        )
        memory_index = runtime["memory_index"]
        retrieval_events = runtime["retrieval_events"]
        answer_support_rows = runtime["answer_support_rows"]
        answer_context_claim_rows = runtime["answer_context_claim_rows"]
        claim_lifecycle_rows = runtime["claim_lifecycle_rows"]
        audit_packet_rows = runtime["audit_packet_rows"]
    elif not answer_context_claim_rows:
        answer_context_claim_rows = build_answer_context_claim_rows(
            all_sample_rows,
            memory_index,
            retrieval_events,
        )
        _write_jsonl_replace(
            runtime_analysis_dir / "answer_context_claim_rows.jsonl",
            answer_context_claim_rows,
        )
    if not claim_lifecycle_rows:
        claim_lifecycle_rows = build_claim_lifecycle_rows(database_path, memory_index)
        _write_jsonl_replace(
            runtime_analysis_dir / "claim_lifecycle_rows.jsonl",
            claim_lifecycle_rows,
        )

    gold_rows = load_or_build_gold_labels(run_dir, all_sample_rows, memory_index)
    audit_rows = build_auditability_rows(
        sample_rows=all_sample_rows,
        gold_rows=gold_rows,
        memory_index=memory_index,
        retrieval_events=retrieval_events,
        answer_support_rows=answer_support_rows,
        answer_context_claim_rows=answer_context_claim_rows,
    )
    analysis_dir = (
        scoped_analysis_dir(run_dir, scope)
        if scope is not None
        else runtime_analysis_dir
    )
    if scope is not None:
        sample_rows = filter_query_rows(all_sample_rows, scope)
        audit_rows = filter_query_rows(audit_rows, scope)
        answer_support_rows = filter_query_rows(answer_support_rows, scope)
        answer_context_claim_rows = filter_query_rows(
            answer_context_claim_rows,
            scope,
        )
        audit_packet_rows = filter_query_rows(audit_packet_rows, scope)
        claim_lifecycle_rows = [
            row
            for row in claim_lifecycle_rows
            if str(row.get("sample_id") or "") in scope.sample_ids
        ]
        _write_jsonl_replace(
            analysis_dir / "answer_support_rows.jsonl",
            answer_support_rows,
        )
        _write_jsonl_replace(
            analysis_dir / "answer_context_claim_rows.jsonl",
            answer_context_claim_rows,
        )
        _write_jsonl_replace(
            analysis_dir / "claim_lifecycle_rows.jsonl",
            claim_lifecycle_rows,
        )
        _write_jsonl_replace(
            analysis_dir / "audit_packet_rows.jsonl",
            audit_packet_rows,
        )
    _write_jsonl_replace(analysis_dir / "auditability_rows.jsonl", audit_rows)
    labels = load_audit_labels(audit_labels_path)
    if scope is not None:
        labels = {
            key: value
            for key, value in labels.items()
            if key[1] in scope.query_ids
        }
    paths = _write_analysis_tables(
        analysis_dir=analysis_dir,
        baselines=selected_baselines,
        audit_rows=audit_rows,
        audit_packet_rows=audit_packet_rows,
        sample_rows=sample_rows,
        memory_index=memory_index,
        labels=labels,
        run_meta=run_meta,
        sampling_scope=scope.metadata() if scope is not None else None,
    )
    return {
        "schema_version": "auditability_report_v2",
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "query_count": len(sample_rows),
        "sampling_scope": scope.metadata() if scope is not None else None,
        "baselines": selected_baselines,
        "audit_labels_path": str(audit_labels_path) if audit_labels_path else None,
        "answer_support_rows_path": str(analysis_dir / "answer_support_rows.jsonl"),
        "answer_context_claim_rows_path": str(analysis_dir / "answer_context_claim_rows.jsonl"),
        "claim_lifecycle_rows_path": str(analysis_dir / "claim_lifecycle_rows.jsonl"),
        "audit_packet_rows_path": str(analysis_dir / "audit_packet_rows.jsonl"),
        **paths,
    }
