"""Blinded human-audit packet preparation and analysis."""

from __future__ import annotations

import csv
import gzip
import json
import random
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from typing import Any

from rich.console import Console
from rich.panel import Panel

from trajpatch.analysis.auditability import (
    analyze_auditability,
    load_audit_labels,
    write_auditability_artifacts,
)
from trajpatch.analysis.gold_labels import (
    build_memory_index,
    extract_source_refs,
    load_details_rows,
)
from trajpatch.analysis.memory_index import source_message_ids_for_refs
from trajpatch.utils.json_utils import write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, sort_keys=True, default=str)
            )
            handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _category(row: dict[str, Any]) -> str:
    stage = str(row.get("failure_localization_stage") or "")
    revision_or_conflict_case = bool(
        row.get("deprecated_source_ref_leakage_proxy")
        or row.get("obsolete_answer_risk_proxy")
        or row.get("bridge_finalization_conflicted")
        or (
            stage == "correct"
            and (row.get("conflict_claim_visible") or row.get("conflict_block_visible"))
        )
    )
    if revision_or_conflict_case:
        return "revision_conflict"
    if stage in {
        "memory_absent",
        "page_routing_miss",
        "trajectory_selection_miss",
        "snapshot_compaction_miss",
        "source_compaction_miss",
    }:
        return "routing_selection"
    if row.get("answer_experiment_observed"):
        if (
            row.get("answer_experiment_judge_score") == 1.0
            and row.get("answer_experiment_source_supported_proxy") is True
        ):
            return "correct_supported"
        if (
            row.get("answer_experiment_unsupported_answer_risk") is True
            or (
                row.get("answer_experiment_judge_score") is not None
                and float(row["answer_experiment_judge_score"]) < 1.0
            )
        ):
            return "answer_unsupported"
    if stage in {
        "grounding_miss",
        "answer_synthesis_error",
        "unsupported_overgeneration",
    } or row.get("unsupported_answer_risk"):
        return "answer_unsupported"
    if stage == "correct" and row.get("source_supported_proxy"):
        return "correct_supported"
    return "other"


def _packet_text(packet: dict[str, Any]) -> str:
    lines: list[str] = []
    for source in list(packet.get("source_message_previews") or []):
        lines.append(
            f"[{source.get('source_ref') or source.get('message_id')}]\n"
            f"{source.get('text_preview') or ''}"
        )
    for claim in list(packet.get("claim_previews") or []):
        source_refs = ",".join(claim.get("source_refs") or [])
        lines.append(
            f"[Claim status={claim.get('status')} refs={source_refs}]\n"
            f"{claim.get('text_preview') or ''}"
        )
    return "\n\n".join(lines)


def _experiment_packet_text(
    *,
    answer_text: str,
    context: dict[str, Any],
    memory_index: dict[str, Any],
) -> str:
    selected_refs = {
        str(ref)
        for ref in list(context.get("selected_source_refs") or [])
        if str(ref).strip()
    }
    answer_refs = set(extract_source_refs(answer_text)) & selected_refs
    packet_refs = answer_refs or selected_refs
    sample_id = str(context.get("sample_id") or "")
    source_ids = source_message_ids_for_refs(
        memory_index,
        sample_id=sample_id,
        source_refs=sorted(packet_refs),
    )
    pages = ", ".join(context.get("selected_page_ids") or []) or "none"
    trajectories = (
        ", ".join(context.get("selected_trajectory_ids") or []) or "none"
    )
    snapshots = ", ".join(context.get("selected_snapshot_ids") or []) or "none"
    cited_refs = ", ".join(sorted(answer_refs)) or "none"
    lines = [
        "Retrieved provenance chain:",
        f"Pages: {pages}",
        f"Trajectories: {trajectories}",
        f"Snapshots: {snapshots}",
        f"Answer-cited refs: {cited_refs}",
    ]
    for source_id in source_ids:
        source = memory_index["raw_messages_by_id"].get(source_id, {})
        source_ref = str(source.get("source_ref") or source_id)
        lines.append(
            f"[SOURCE {source_ref} speaker="
            f"{source.get('speaker_name') or source.get('role') or 'unknown'} "
            f"time={source.get('occurred_at') or 'unknown'}]\n"
            f"{source.get('content') or ''}"
        )
    for snapshot_id in dict.fromkeys(
        str(value)
        for value in list(context.get("selected_snapshot_ids") or [])
        if str(value).strip()
    ):
        for claim in memory_index["claims_by_snapshot"].get(snapshot_id, []):
            claim_refs = {
                str(
                    memory_index["raw_messages_by_id"]
                    .get(source_id, {})
                    .get("source_ref")
                    or ""
                )
                for source_id in list(claim.get("source_message_ids") or [])
            }
            claim_refs.discard("")
            if packet_refs and not (claim_refs & packet_refs):
                continue
            lines.append(
                f"[CLAIM {claim.get('claim_id')} status={claim.get('status')} "
                f"refs={','.join(sorted(claim_refs))}]\n"
                f"{claim.get('text') or ''}"
            )
    return "\n\n".join(lines)


def prepare_audit_study(
    run_path: Path | str,
    *,
    case_count: int = 24,
    seed: int = 7,
    answer_experiment_path: Path | str | None = None,
) -> dict[str, Any]:
    if case_count <= 0:
        raise ValueError("case_count must be positive.")
    run_dir = Path(run_path).expanduser().resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    _, sample_rows, database_path = load_details_rows(run_dir)
    answer_experiment: Path | None = None
    experiment_query_ids: set[str] | None = None
    experiment_contexts: dict[tuple[str, str], dict[str, Any]] = {}
    experiment_answers: dict[str, dict[str, Any]] = {}
    experiment_full_rows: dict[str, dict[str, Any]] = {}
    experiment_gold_rows: dict[str, dict[str, Any]] = {}
    if answer_experiment_path is not None:
        answer_experiment = Path(answer_experiment_path).expanduser().resolve()
        experiment_manifest = json.loads(
            (answer_experiment / "experiment_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        source_run = Path(
            str(experiment_manifest.get("source_run_dir") or "")
        ).resolve()
        if source_run != run_dir:
            raise ValueError(
                "The answer experiment does not use the requested benchmark run."
            )
        if experiment_manifest.get("status") != "complete":
            raise ValueError(
                "The answer experiment must have status='complete' before audit "
                "study preparation."
            )
        sampling = json.loads(
            (answer_experiment / "sampling_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        experiment_query_ids = {
            str(row.get("query_task_id") or "")
            for row in list(sampling.get("selected_queries") or [])
            if str(row.get("query_task_id") or "")
        }
        experiment_contexts = {
            (
                str(row.get("variant") or ""),
                str(row.get("query_task_id") or ""),
            ): row
            for row in _read_jsonl(
                answer_experiment / "variant_context_rows.jsonl"
            )
        }
        generation_dir = answer_experiment / "jobs" / "generation" / "full"
        for path in sorted(generation_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "complete":
                experiment_answers[
                    str(payload.get("query_task_id") or "")
                ] = payload
        experiment_full_rows = {
            str(row.get("query_task_id") or ""): row
            for row in _read_jsonl(
                answer_experiment / "answer_ablation_rows.jsonl"
            )
            if str(row.get("variant") or "") == "full"
        }
        experiment_gold_rows = {
            str(row.get("query_task_id") or ""): row
            for row in _read_jsonl(
                answer_experiment / "gold_label_rows.jsonl"
            )
            if str(row.get("query_task_id") or "")
        }
        missing_answers = sorted(experiment_query_ids - set(experiment_answers))
        if missing_answers:
            raise ValueError(
                "The answer experiment lacks completed regenerated Full TrajWiki "
                f"answers for {len(missing_answers)} queries; "
                f"examples={missing_answers[:5]}."
            )
        missing_analysis_rows = sorted(
            experiment_query_ids - set(experiment_full_rows)
        )
        if missing_analysis_rows:
            raise ValueError(
                "The answer experiment lacks analyzed Full TrajWiki rows for "
                f"{len(missing_analysis_rows)} queries; "
                f"examples={missing_analysis_rows[:5]}."
            )
        if set(experiment_gold_rows) != experiment_query_ids:
            raise ValueError(
                "The answer experiment normalized gold-label rows do not match "
                "its sampled query set."
            )
    write_auditability_artifacts(
        run_dir=run_dir,
        sample_rows=sample_rows,
        database_path=database_path,
        packet_save_mode="compact",
    )
    analyze_auditability(run_dir)
    analysis_dir = (
        run_dir / "analysis_v2"
        if (run_dir / "analysis_v2" / "audit_packet_rows.jsonl").exists()
        else run_dir / "analysis"
    )
    audit_rows = _read_jsonl(analysis_dir / "auditability_rows.jsonl")
    packets = _read_jsonl(analysis_dir / "audit_packet_rows.jsonl")
    packet_by_query = {str(row.get("query_task_id")): row for row in packets}
    sample_by_query = {str(row.get("query_task_id")): row for row in sample_rows}
    memory_index = build_memory_index(database_path)
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_selection_rows: list[dict[str, Any]] = []
    for row in audit_rows:
        if (
            experiment_query_ids is not None
            and str(row.get("query_task_id") or "") not in experiment_query_ids
        ):
            continue
        selection_row = dict(row)
        query_task_id = str(row.get("query_task_id") or "")
        if answer_experiment is not None:
            full_row = experiment_full_rows[query_task_id]
            selection_row.update(
                {
                    "answer_experiment_observed": True,
                    "answer_experiment_judge_score": full_row.get(
                        "independent_judge_score"
                    ),
                    "answer_experiment_source_supported_proxy": full_row.get(
                        "source_supported_proxy"
                    ),
                    "answer_experiment_unsupported_answer_risk": full_row.get(
                        "unsupported_answer_risk"
                    ),
                }
            )
        eligible_selection_rows.append(selection_row)
        grouped[_category(selection_row)].append(selection_row)
    for values in grouped.values():
        values.sort(
            key=lambda row: (str(row.get("sample_id")), str(row.get("query_task_id")))
        )
        rng.shuffle(values)
    category_order = [
        "correct_supported",
        "routing_selection",
        "answer_unsupported",
        "revision_conflict",
    ]
    per_category = max(1, case_count // len(category_order))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category in category_order:
        for row in grouped[category][:per_category]:
            selected.append(row)
            selected_ids.add(str(row.get("query_task_id")))
    if len(selected) < case_count:
        remainder = [
            row
            for row in eligible_selection_rows
            if str(row.get("query_task_id")) not in selected_ids
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: case_count - len(selected)])
    selected = selected[:case_count]

    experiment_suffix = (
        f"_{answer_experiment.name}" if answer_experiment is not None else ""
    )
    study_dir = (
        run_dir
        / "rebuttal_experiments"
        / f"audit_study_seed{seed}_{len(selected)}{experiment_suffix}"
    )
    study_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for case_index, audit_row in enumerate(selected):
        query_task_id = str(audit_row.get("query_task_id") or "")
        sample_row = sample_by_query[query_task_id]
        sample_id = str(sample_row.get("sample_id") or "")
        traj_packet = packet_by_query.get(query_task_id, {})
        candidate_answer = sample_row.get("answer_text")
        packet_text = _packet_text(traj_packet)
        if answer_experiment is not None:
            candidate_answer = experiment_answers[query_task_id].get("answer_text")
            context = dict(experiment_contexts.get(("full", query_task_id)) or {})
            if not context:
                raise ValueError(
                    "The answer experiment lacks a Full TrajWiki context for "
                    f"{query_task_id}."
                )
            context_path = context.get("context_path")
            if context_path and Path(str(context_path)).exists():
                with gzip.open(
                    Path(str(context_path)),
                    "rt",
                    encoding="utf-8",
                ) as handle:
                    context = {**context, **json.load(handle)}
            packet_text = _experiment_packet_text(
                answer_text=str(candidate_answer or ""),
                context=context,
                memory_index=memory_index,
            )
        full_sources = [
            {
                "source_ref": message.get("source_ref"),
                "speaker": message.get("speaker_name") or message.get("role"),
                "occurred_at": message.get("occurred_at"),
                "text": message.get("content"),
            }
            for message in memory_index["sample_raw_messages"].get(sample_id, [])
        ]
        condition_payloads = {
            "trajwiki_packet": packet_text,
            "full_dialogue": "\n\n".join(
                f"[{source.get('source_ref')} speaker={source.get('speaker')} "
                f"time={source.get('occurred_at')}]\n{source.get('text')}"
                for source in full_sources
            ),
        }
        gold_source_refs = (
            list(
                experiment_gold_rows[query_task_id].get(
                    "gold_source_refs"
                )
                or []
            )
            if answer_experiment is not None
            else list(
                extract_source_refs(
                    dict(
                        dict(sample_row.get("metadata") or {}).get(
                            "query_metadata"
                        )
                        or {}
                    )
                )
            )
        )
        case_id = f"case-{case_index + 1:02d}"
        condition_aliases = list(condition_payloads)
        rng.shuffle(condition_aliases)
        aliases = {
            condition_aliases[0]: "Packet A",
            condition_aliases[1]: "Packet B",
        }
        for annotator_slot in [1, 2]:
            first_condition = (
                "trajwiki_packet"
                if (case_index + annotator_slot) % 2 == 0
                else "full_dialogue"
            )
            order = [
                first_condition,
                next(
                    condition
                    for condition in condition_payloads
                    if condition != first_condition
                ),
            ]
            for order_index, condition in enumerate(order, start=1):
                task_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{study_dir}:{case_id}:{annotator_slot}:{condition}",
                    )
                )
                tasks.append(
                    {
                        "schema_version": "human_audit_task_v1",
                        "task_id": task_id,
                        "case_id": case_id,
                        "annotator_slot": annotator_slot,
                        "order_index": order_index,
                        "packet_label": aliases[condition],
                        "question": sample_row.get("question"),
                        "candidate_answer": candidate_answer,
                        "evidence_packet": condition_payloads[condition],
                        "allowed_failure_stages": [
                            "memory_absent",
                            "page_routing_miss",
                            "trajectory_selection_miss",
                            "snapshot_compaction_miss",
                            "source_compaction_miss",
                            "grounding_miss",
                            "answer_synthesis_error",
                            "unsupported_overgeneration",
                            "correct",
                            "uncertain",
                        ],
                        "contains_sensitive_text": True,
                    }
                )
                key_rows.append(
                    {
                        "task_id": task_id,
                        "contains_sensitive_text": True,
                        "case_id": case_id,
                        "sample_id": sample_id,
                        "query_task_id": query_task_id,
                        "annotator_slot": annotator_slot,
                        "order_index": order_index,
                        "condition": condition,
                        "selection_category": _category(audit_row),
                        "gold_answer": sample_row.get("gold_answer"),
                        "gold_source_refs": sorted(gold_source_refs),
                    }
                )
    ordered_tasks: list[dict[str, Any]] = []
    for annotator_slot in [1, 2]:
        presentation_index = 0
        for order_index in [1, 2]:
            block = [
                row
                for row in tasks
                if int(row["annotator_slot"]) == annotator_slot
                and int(row["order_index"]) == order_index
            ]
            rng.shuffle(block)
            for row in block:
                presentation_index += 1
                row["presentation_index"] = presentation_index
                ordered_tasks.append(row)
    tasks = ordered_tasks
    presentation_by_task = {
        str(row["task_id"]): int(row["presentation_index"]) for row in tasks
    }
    for row in key_rows:
        row["presentation_index"] = presentation_by_task[str(row["task_id"])]
    primary_condition_counts = Counter(
        (
            int(row["annotator_slot"]),
            str(row["condition"]),
        )
        for row in key_rows
        if int(row["order_index"]) == 1
    )
    _write_jsonl(study_dir / "audit_tasks.jsonl", tasks)
    _write_jsonl(study_dir / "audit_study_key.jsonl", key_rows)
    label_template_path = study_dir / "audit_labels_template.csv"
    with label_template_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "contains_sensitive_text",
            "sample_id",
            "query_task_id",
            "question",
            "candidate_answer",
            "gold_answer",
            "gold_source_refs",
            "true_source_supported",
            "true_failure_stage",
            "true_supporting_source_refs",
            "true_conflict_required",
            "true_conflict_handled",
            "true_obsolete_required",
            "true_obsolete_handled",
            "adjudication_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        seen_queries: set[str] = set()
        for row in key_rows:
            query_task_id = str(row.get("query_task_id") or "")
            if query_task_id in seen_queries:
                continue
            seen_queries.add(query_task_id)
            writer.writerow(
                {
                    "contains_sensitive_text": True,
                    "sample_id": row.get("sample_id"),
                    "query_task_id": query_task_id,
                    "question": sample_by_query[query_task_id].get("question"),
                    "candidate_answer": (
                        experiment_answers[query_task_id].get("answer_text")
                        if answer_experiment is not None
                        else sample_by_query[query_task_id].get("answer_text")
                    ),
                    "gold_answer": row.get("gold_answer"),
                    "gold_source_refs": ",".join(
                        row.get("gold_source_refs") or []
                    ),
                    "true_source_supported": "",
                    "true_failure_stage": "",
                    "true_supporting_source_refs": "",
                    "true_conflict_required": "",
                    "true_conflict_handled": "",
                    "true_obsolete_required": "",
                    "true_obsolete_handled": "",
                    "adjudication_notes": "",
                }
            )
    manifest = {
        "schema_version": "human_audit_study_v1",
        "run_dir": str(run_dir),
        "answer_experiment_dir": (
            str(answer_experiment) if answer_experiment is not None else None
        ),
        "case_count": len(selected),
        "task_count": len(tasks),
        "primary_first_exposure_task_count": len(selected) * 2,
        "selection_category_counts": dict(
            sorted(Counter(_category(row) for row in selected).items())
        ),
        "selection_category_available_counts": {
            category: len(grouped[category])
            for category in category_order
        },
        "selection_category_target_per_category": per_category,
        "selection_category_shortages": {
            category: max(0, per_category - len(grouped[category]))
            for category in category_order
            if len(grouped[category]) < per_category
        },
        "primary_condition_counts": {
            f"annotator_{annotator_slot}:{condition}": count
            for (annotator_slot, condition), count in sorted(
                primary_condition_counts.items()
            )
        },
        "annotator_count": 2,
        "seed": seed,
        "contains_sensitive_text": True,
        "blinding": {
            "gold_answer_excluded": True,
            "method_name_excluded_from_tasks": True,
            "judge_verdict_excluded": True,
            "predicted_failure_stage_excluded": True,
            "proxy_labels_excluded": True,
        },
        "tasks_path": str(study_dir / "audit_tasks.jsonl"),
        "key_path": str(study_dir / "audit_study_key.jsonl"),
        "audit_labels_template_path": str(label_template_path),
    }
    write_json(study_dir / "audit_study_manifest.json", manifest)
    return {**manifest, "study_dir": str(study_dir)}


def conduct_audit_study(
    tasks_path: Path | str,
    *,
    annotator_slot: int,
    annotator_id: str,
    output_path: Path | str,
    max_exposure: int = 1,
    console: Console | None = None,
    input_fn: Callable[[str], str] = input,
) -> dict[str, Any]:
    if annotator_slot not in {1, 2}:
        raise ValueError("annotator_slot must be 1 or 2.")
    if not str(annotator_id).strip():
        raise ValueError("annotator_id must be non-empty.")
    if max_exposure not in {1, 2}:
        raise ValueError("max_exposure must be 1 (primary) or 2 (including crossover).")
    console = console or Console()
    tasks = [
        row
        for row in _read_jsonl(Path(tasks_path))
        if int(row.get("annotator_slot") or 0) == int(annotator_slot)
        and int(row.get("order_index") or 0) <= max_exposure
    ]
    tasks.sort(key=lambda row: int(row.get("presentation_index") or 0))
    results: list[dict[str, Any]] = _read_jsonl(Path(output_path))
    conflicting_results = [
        row
        for row in results
        if int(row.get("annotator_slot") or 0) != annotator_slot
        or str(row.get("annotator_id") or "") != str(annotator_id)
    ]
    if conflicting_results:
        raise ValueError(
            "The output file already contains results for another annotator "
            "or annotator slot."
        )
    completed = {str(row.get("task_id")) for row in results}
    for task in tasks:
        if str(task.get("task_id")) in completed:
            continue
        console.print(
            Panel(
                f"Question:\n{task.get('question')}\n\n"
                f"Candidate answer:\n{task.get('candidate_answer')}\n\n"
                f"{task.get('packet_label')}:\n{task.get('evidence_packet')}",
                title=f"Audit {task.get('case_id')}",
            )
        )
        started = time.perf_counter()
        supported = ""
        while supported not in {"yes", "no", "uncertain"}:
            supported = (
                input_fn("Source-supported? [yes/no/uncertain]: ").strip().lower()
            )
        source_refs = input_fn(
            "Supporting or refuting source refs (comma-separated, blank if none): "
        ).strip()
        allowed_stages = {
            str(value) for value in list(task.get("allowed_failure_stages") or [])
        }
        stage = ""
        while stage not in allowed_stages:
            stage = input_fn("Failure stage (or correct/uncertain): ").strip().lower()
        confidence = ""
        while not confidence.isdigit() or int(confidence) not in range(1, 6):
            confidence = input_fn("Confidence [1-5]: ").strip()
        notes = input_fn("Optional notes: ").strip()
        elapsed = time.perf_counter() - started
        result = {
            "schema_version": "human_audit_result_v1",
            "contains_sensitive_text": True,
            "task_id": task.get("task_id"),
            "case_id": task.get("case_id"),
            "annotator_slot": annotator_slot,
            "annotator_id": annotator_id,
            "source_supported_decision": supported,
            "localized_source_refs": extract_source_refs(source_refs),
            "failure_stage_decision": stage,
            "confidence": int(confidence),
            "notes": notes,
            "audit_seconds": elapsed,
        }
        results.append(result)
        _write_jsonl(Path(output_path), results)
    return {
        "annotator_id": annotator_id,
        "annotator_slot": annotator_slot,
        "max_exposure": max_exposure,
        "target_task_count": len(tasks),
        "completed_task_count": len(
            [
                row
                for row in results
                if str(row.get("task_id") or "")
                in {str(task.get("task_id") or "") for task in tasks}
            ]
        ),
        "output_path": str(output_path),
    }


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=False)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def analyze_audit_study(
    study_path: Path | str,
    *,
    audit_labels_path: Path | str | None = None,
) -> dict[str, Any]:
    study_dir = Path(study_path).expanduser().resolve()
    key_rows = _read_jsonl(study_dir / "audit_study_key.jsonl")
    key_task_ids = [str(row.get("task_id") or "") for row in key_rows]
    duplicate_key_ids = [
        task_id
        for task_id, count in Counter(key_task_ids).items()
        if task_id and count > 1
    ]
    if duplicate_key_ids:
        raise ValueError(
            f"Duplicate task IDs in audit study key: {duplicate_key_ids[:5]}"
        )
    key_by_task = {
        str(row.get("task_id")): row for row in key_rows if row.get("task_id")
    }
    results_by_task: dict[str, dict[str, Any]] = {}
    for path in sorted(study_dir.glob("audit_results*.jsonl")):
        for row in _read_jsonl(path):
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise ValueError(f"Audit result in {path} is missing task_id.")
            if task_id in results_by_task:
                raise ValueError(
                    f"Duplicate audit result for task_id={task_id!r} across result files."
                )
            key = key_by_task.get(task_id)
            if key is None:
                raise ValueError(
                    f"Audit result references unknown task_id={task_id!r}."
                )
            if int(row.get("annotator_slot") or 0) != int(
                key.get("annotator_slot") or 0
            ):
                raise ValueError(
                    f"Audit result annotator_slot does not match task_id={task_id!r}."
                )
            if not str(row.get("annotator_id") or "").strip():
                raise ValueError(
                    f"Audit result is missing annotator_id for task_id={task_id!r}."
                )
            results_by_task[task_id] = row
    annotators_by_slot: dict[int, set[str]] = defaultdict(set)
    for row in results_by_task.values():
        annotators_by_slot[int(row.get("annotator_slot") or 0)].add(
            str(row.get("annotator_id") or "")
        )
    mixed_slots = {
        slot: sorted(annotators)
        for slot, annotators in annotators_by_slot.items()
        if len(annotators) > 1
    }
    if mixed_slots:
        raise ValueError(
            "Each annotator slot must use exactly one annotator_id; "
            f"mixed slots={mixed_slots}."
        )
    cross_slot_overlap = (
        annotators_by_slot.get(1, set())
        & annotators_by_slot.get(2, set())
    )
    if cross_slot_overlap:
        raise ValueError(
            "The two annotator slots must use distinct annotator_id values; "
            f"overlap={sorted(cross_slot_overlap)}."
        )
    enriched = [
        {**row, **key_by_task.get(str(row.get("task_id")), {})}
        for row in results_by_task.values()
        if str(row.get("task_id")) in key_by_task
    ]
    labels = load_audit_labels(audit_labels_path)
    summary_rows: list[dict[str, Any]] = []
    for condition in ["trajwiki_packet", "full_dialogue"]:
        all_exposures = [row for row in enriched if row.get("condition") == condition]
        group = [row for row in all_exposures if int(row.get("order_index") or 0) == 1]
        labeled = [
            (
                row,
                labels.get(
                    (str(row.get("sample_id")), str(row.get("query_task_id"))), {}
                ),
            )
            for row in group
        ]
        support_correct = [
            (
                str(row.get("source_supported_decision"))
                == (
                    "yes"
                    if label.get("true_source_supported") is True
                    else "no"
                )
            )
            for row, label in labeled
            if label.get("true_source_supported") is not None
            and str(row.get("source_supported_decision"))
            in {"yes", "no", "uncertain"}
        ]
        stage_correct = [
            str(row.get("failure_stage_decision"))
            == str(label.get("true_failure_stage"))
            for row, label in labeled
            if label.get("true_failure_stage")
        ]
        localization_scores: list[tuple[float, float, bool]] = []
        for row, label in labeled:
            true_refs = set(
                extract_source_refs(
                    label.get("true_supporting_source_refs")
                    or label.get("gold_source_refs")
                )
            )
            if not true_refs:
                continue
            predicted_refs = set(
                extract_source_refs(row.get("localized_source_refs") or [])
            )
            overlap = true_refs & predicted_refs
            precision = len(overlap) / len(predicted_refs) if predicted_refs else 0.0
            recall = len(overlap) / len(true_refs)
            localization_scores.append((precision, recall, bool(overlap)))
        summary_rows.append(
            {
                "condition": condition,
                "audit_count": len(group),
                "all_exposure_audit_count": len(all_exposures),
                "mean_audit_seconds": (
                    mean(float(row.get("audit_seconds") or 0.0) for row in group)
                    if group
                    else None
                ),
                "median_audit_seconds": (
                    median(float(row.get("audit_seconds") or 0.0) for row in group)
                    if group
                    else None
                ),
                "supported_yes_rate": (
                    mean(
                        float(
                            str(row.get("source_supported_decision") or "")
                            == "yes"
                        )
                        for row in group
                    )
                    if group
                    else None
                ),
                "supported_no_rate": (
                    mean(
                        float(
                            str(row.get("source_supported_decision") or "")
                            == "no"
                        )
                        for row in group
                    )
                    if group
                    else None
                ),
                "uncertain_rate": (
                    mean(
                        float(
                            str(row.get("source_supported_decision") or "")
                            == "uncertain"
                        )
                        for row in group
                    )
                    if group
                    else None
                ),
                "mean_confidence": (
                    mean(float(row.get("confidence") or 0.0) for row in group)
                    if group
                    else None
                ),
                "source_support_accuracy": (
                    mean(float(value) for value in support_correct)
                    if support_correct
                    else None
                ),
                "source_support_accuracy_count": len(support_correct),
                "failure_stage_accuracy": (
                    mean(float(value) for value in stage_correct)
                    if stage_correct
                    else None
                ),
                "failure_stage_accuracy_count": len(stage_correct),
                "source_localization_any_overlap_accuracy": (
                    mean(float(item[2]) for item in localization_scores)
                    if localization_scores
                    else None
                ),
                "source_localization_precision": (
                    mean(item[0] for item in localization_scores)
                    if localization_scores
                    else None
                ),
                "source_localization_recall": (
                    mean(item[1] for item in localization_scores)
                    if localization_scores
                    else None
                ),
                "source_localization_count": len(localization_scores),
                "accuracy_label_available": bool(
                    support_correct or stage_correct or localization_scores
                ),
            }
        )
    by_condition_case_annotator: dict[tuple[str, str, int], dict[str, Any]] = {
        (
            str(row.get("condition")),
            str(row.get("case_id")),
            int(row.get("annotator_slot") or 0),
        ): row
        for row in enriched
    }
    support_kappas: dict[str, float | None] = {}
    stage_kappas: dict[str, float | None] = {}
    disagreements: list[dict[str, Any]] = []
    for condition in ["trajwiki_packet", "full_dialogue"]:
        left: list[str] = []
        right: list[str] = []
        left_stages: list[str] = []
        right_stages: list[str] = []
        case_ids = sorted(
            {
                case_id
                for candidate_condition, case_id, _ in by_condition_case_annotator
                if candidate_condition == condition
            }
        )
        for case_id in case_ids:
            first = by_condition_case_annotator.get((condition, case_id, 1))
            second = by_condition_case_annotator.get((condition, case_id, 2))
            if not first or not second:
                continue
            first_decision = str(first.get("source_supported_decision") or "")
            second_decision = str(second.get("source_supported_decision") or "")
            left.append(first_decision)
            right.append(second_decision)
            first_stage = str(first.get("failure_stage_decision") or "")
            second_stage = str(second.get("failure_stage_decision") or "")
            left_stages.append(first_stage)
            right_stages.append(second_stage)
            if first_decision != second_decision or first_stage != second_stage:
                disagreements.append(
                    {
                        "case_id": case_id,
                        "condition": condition,
                        "annotator_1_decision": first_decision,
                        "annotator_2_decision": second_decision,
                        "annotator_1_stage": first_stage,
                        "annotator_2_stage": second_stage,
                        "requires_third_adjudication": True,
                    }
                )
        support_kappas[condition] = _cohen_kappa(left, right)
        stage_kappas[condition] = _cohen_kappa(left_stages, right_stages)
    _write_jsonl(study_dir / "audit_disagreements.jsonl", disagreements)
    csv_path = study_dir / "human_audit_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "audit_count",
                "all_exposure_audit_count",
                "mean_audit_seconds",
                "median_audit_seconds",
                "supported_yes_rate",
                "supported_no_rate",
                "uncertain_rate",
                "mean_confidence",
                "source_support_accuracy",
                "source_support_accuracy_count",
                "failure_stage_accuracy",
                "failure_stage_accuracy_count",
                "source_localization_any_overlap_accuracy",
                "source_localization_precision",
                "source_localization_recall",
                "source_localization_count",
                "accuracy_label_available",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    report = {
        "schema_version": "human_audit_report_v1",
        "study_dir": str(study_dir),
        "result_count": len(enriched),
        "primary_first_exposure_result_count": sum(
            int(row.get("audit_count") or 0) for row in summary_rows
        ),
        "expected_primary_first_exposure_result_count": sum(
            1 for row in key_rows if int(row.get("order_index") or 0) == 1
        ),
        "primary_first_exposure_complete": {
            str(row.get("task_id") or "")
            for row in key_rows
            if int(row.get("order_index") or 0) == 1
        }.issubset(results_by_task),
        "crossover_result_count": sum(
            1
            for row in enriched
            if int(row.get("order_index") or 0) == 2
        ),
        "condition_summary": summary_rows,
        "source_support_cohen_kappa_exploratory": support_kappas,
        "failure_stage_cohen_kappa_exploratory": stage_kappas,
        "primary_analysis_uses_first_exposure_only": True,
        "disagreement_count": len(disagreements),
        "audit_labels_path": str(audit_labels_path) if audit_labels_path else None,
        "accuracy_metrics_reported": any(
            bool(row.get("accuracy_label_available")) for row in summary_rows
        ),
        "summary_path": str(csv_path),
        "disagreements_path": str(study_dir / "audit_disagreements.jsonl"),
    }
    write_json(study_dir / "human_audit_summary.json", report)
    return report
