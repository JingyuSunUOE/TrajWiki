"""Gold evidence label resolution for LOCOMO offline ablations."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from trajpatch.utils.json_utils import append_jsonl

SOURCE_REF_RE = re.compile(r"\bD\d+:\d+\b")


def load_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def extract_source_refs(value: Any) -> list[str]:
    values = list(value) if isinstance(value, (list, tuple, set)) else [value]
    refs: list[str] = []
    seen: set[str] = set()
    for item in values:
        for match in SOURCE_REF_RE.finditer(str(item or "")):
            ref = match.group(0)
            if ref in seen:
                continue
            seen.add(ref)
            refs.append(ref)
    return refs


def gold_refs_from_query_metadata(query_metadata: dict[str, Any]) -> list[str]:
    for key in [
        "evidence_only_conversation",
        "gold_evidence_refs",
        "gold_evidence_raw",
        "gold_evidence",
    ]:
        refs = extract_source_refs(query_metadata.get(key))
        if refs:
            return refs
    return []


def load_details_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    details_path = run_dir / "details.json"
    details = json.loads(details_path.read_text(encoding="utf-8"))
    run_meta = dict(details.get("run_meta") or {})
    database_path = Path(run_meta.get("database_path") or run_dir / "trajpatch.sqlite")
    return run_meta, list(details.get("samples") or []), database_path


def build_memory_index(database_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        raw_messages_by_id: dict[str, dict[str, Any]] = {}
        raw_message_ids_by_ref: dict[str, set[str]] = defaultdict(set)
        sample_raw_messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in connection.execute(
            """
            SELECT id, sample_id, turn_index, role, speaker_name, content, source_ref, occurred_at
            FROM raw_messages
            """
        ):
            record = {
                "id": str(row["id"]),
                "sample_id": str(row["sample_id"]),
                "turn_index": int(row["turn_index"] or 0),
                "role": str(row["role"] or ""),
                "speaker_name": row["speaker_name"],
                "content": str(row["content"] or ""),
                "source_ref": str(row["source_ref"]) if row["source_ref"] else "",
                "occurred_at": row["occurred_at"],
            }
            raw_messages_by_id[record["id"]] = record
            sample_raw_messages[record["sample_id"]].append(record)
            if record["source_ref"]:
                raw_message_ids_by_ref[record["source_ref"]].add(record["id"])

        trajectory_to_sample: dict[str, str] = {}
        sample_to_trajectories: dict[str, set[str]] = defaultdict(set)
        trajectory_metadata: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            "SELECT id, sample_id, label, latest_snapshot_id, metadata_json FROM trajectories"
        ):
            trajectory_id = str(row["id"])
            sample_id = str(row["sample_id"])
            metadata = dict(load_json_field(row["metadata_json"], {}))
            metadata.setdefault("label", str(row["label"] or ""))
            if row["latest_snapshot_id"]:
                metadata.setdefault("latest_snapshot_id", str(row["latest_snapshot_id"]))
            trajectory_to_sample[trajectory_id] = sample_id
            sample_to_trajectories[sample_id].add(trajectory_id)
            trajectory_metadata[trajectory_id] = metadata

        claims_by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        claims_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in connection.execute(
            """
            SELECT id, snapshot_id, trajectory_id, claim_id, text, status, source_message_ids_json,
                   metadata_json
            FROM claims
            """
        ):
            source_message_ids = [
                str(item) for item in load_json_field(row["source_message_ids_json"], []) if str(item).strip()
            ]
            metadata = dict(load_json_field(row["metadata_json"], {}))
            source_refs = {
                str(raw_messages_by_id[message_id]["source_ref"])
                for message_id in source_message_ids
                if message_id in raw_messages_by_id and raw_messages_by_id[message_id]["source_ref"]
            }
            claim = {
                "id": str(row["id"]),
                "claim_id": str(row["claim_id"] or ""),
                "snapshot_id": str(row["snapshot_id"]),
                "trajectory_id": str(row["trajectory_id"]),
                "text": str(row["text"] or ""),
                "status": str(row["status"] or ""),
                "source_message_ids": source_message_ids,
                "source_refs": source_refs,
                "facets": list(metadata.get("facets_v2") or metadata.get("facets_v1") or []),
                "exact_terms": list(metadata.get("exact_terms_v2") or metadata.get("exact_terms_v1") or []),
                "display_signals": dict(metadata.get("display_signals_v1") or {}),
                "metadata": metadata,
            }
            claims_by_snapshot[claim["snapshot_id"]].append(claim)
            claims_by_trajectory[claim["trajectory_id"]].append(claim)

        snapshot_to_trajectory: dict[str, str] = {}
        trajectory_to_snapshots: dict[str, list[str]] = defaultdict(list)
        trajectory_snapshot_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
        snapshot_refs: dict[str, set[str]] = {}
        snapshot_source_message_ids: dict[str, list[str]] = {}
        snapshot_texts: dict[str, str] = {}
        snapshot_versions: dict[str, int] = {}
        snapshot_metadata: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            """
            SELECT id, trajectory_id, version, links_json, summary_content, context, semantic_text,
                   raw_text, metadata_json
            FROM episodic_snapshots
            """
        ):
            snapshot_id = str(row["id"])
            trajectory_id = str(row["trajectory_id"])
            version = int(row["version"] or 0)
            link_ids = [str(item) for item in load_json_field(row["links_json"], []) if str(item).strip()]
            source_ids = list(dict.fromkeys(
                link_ids
                + [
                    message_id
                    for claim in claims_by_snapshot.get(snapshot_id, [])
                    for message_id in list(claim.get("source_message_ids") or [])
                ]
            ))
            refs = {
                str(raw_messages_by_id[message_id]["source_ref"])
                for message_id in source_ids
                if message_id in raw_messages_by_id and raw_messages_by_id[message_id]["source_ref"]
            }
            snapshot_to_trajectory[snapshot_id] = trajectory_id
            trajectory_snapshot_rows[trajectory_id].append((version, snapshot_id))
            snapshot_versions[snapshot_id] = version
            snapshot_refs[snapshot_id] = refs
            snapshot_source_message_ids[snapshot_id] = source_ids
            snapshot_metadata[snapshot_id] = dict(load_json_field(row["metadata_json"], {}))
            snapshot_texts[snapshot_id] = "\n".join(
                str(value or "")
                for value in [
                    row["summary_content"],
                    row["context"],
                    row["semantic_text"],
                    row["raw_text"],
                    *[claim.get("text", "") for claim in claims_by_snapshot.get(snapshot_id, [])],
                ]
                if str(value or "").strip()
            )

        for trajectory_id, rows in trajectory_snapshot_rows.items():
            trajectory_to_snapshots[trajectory_id] = [
                snapshot_id for _, snapshot_id in sorted(rows, key=lambda item: (item[0], item[1]))
            ]
        trajectory_lengths = {
            trajectory_id: len(snapshot_ids)
            for trajectory_id, snapshot_ids in trajectory_to_snapshots.items()
        }
        trajectory_refs: dict[str, set[str]] = defaultdict(set)
        for trajectory_id, snapshot_ids in trajectory_to_snapshots.items():
            for snapshot_id in snapshot_ids:
                trajectory_refs[trajectory_id].update(snapshot_refs.get(snapshot_id, set()))

        page_to_trajectory_ids: dict[str, list[str]] = {}
        page_to_sample: dict[str, str] = {}
        page_types: dict[str, str] = {}
        page_texts: dict[str, str] = {}
        page_titles: dict[str, str] = {}
        sample_to_pages: dict[str, set[str]] = defaultdict(set)
        for row in connection.execute(
            """
            SELECT id, sample_id, page_type, title, markdown_text, trajectory_ids_json
            FROM wiki_pages
            """
        ):
            page_id = str(row["id"])
            sample_id = str(row["sample_id"])
            page_to_sample[page_id] = sample_id
            page_types[page_id] = str(row["page_type"] or "")
            page_titles[page_id] = str(row["title"] or "")
            page_texts[page_id] = str(row["markdown_text"] or "")
            page_to_trajectory_ids[page_id] = [
                str(item) for item in load_json_field(row["trajectory_ids_json"], []) if str(item).strip()
            ]
            sample_to_pages[sample_id].add(page_id)

        source_ref_to_trajectories: dict[str, set[str]] = defaultdict(set)
        for trajectory_id, refs in trajectory_refs.items():
            for ref in refs:
                source_ref_to_trajectories[ref].add(trajectory_id)

        return {
            "raw_messages_by_id": raw_messages_by_id,
            "raw_message_ids_by_ref": raw_message_ids_by_ref,
            "sample_raw_messages": sample_raw_messages,
            "trajectory_to_sample": trajectory_to_sample,
            "sample_to_trajectories": sample_to_trajectories,
            "trajectory_metadata": trajectory_metadata,
            "trajectory_refs": trajectory_refs,
            "trajectory_lengths": trajectory_lengths,
            "trajectory_to_snapshots": trajectory_to_snapshots,
            "claims_by_snapshot": claims_by_snapshot,
            "claims_by_trajectory": claims_by_trajectory,
            "snapshot_to_trajectory": snapshot_to_trajectory,
            "snapshot_refs": snapshot_refs,
            "snapshot_source_message_ids": snapshot_source_message_ids,
            "snapshot_texts": snapshot_texts,
            "snapshot_versions": snapshot_versions,
            "snapshot_metadata": snapshot_metadata,
            "page_to_trajectory_ids": page_to_trajectory_ids,
            "page_to_sample": page_to_sample,
            "page_types": page_types,
            "page_texts": page_texts,
            "page_titles": page_titles,
            "sample_to_pages": sample_to_pages,
            "source_ref_to_trajectories": source_ref_to_trajectories,
        }
    finally:
        connection.close()


def build_gold_label_rows(
    sample_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_message_ids_by_ref = memory_index["raw_message_ids_by_ref"]
    sample_to_trajectories = memory_index["sample_to_trajectories"]
    trajectory_refs = memory_index["trajectory_refs"]
    sample_to_pages = memory_index["sample_to_pages"]
    page_to_trajectory_ids = memory_index["page_to_trajectory_ids"]
    page_types = memory_index["page_types"]
    snapshot_refs = memory_index["snapshot_refs"]
    snapshot_to_trajectory = memory_index["snapshot_to_trajectory"]
    trajectory_to_sample = memory_index["trajectory_to_sample"]

    rows: list[dict[str, Any]] = []
    for row in sample_rows:
        sample_id = str(row.get("sample_id") or "")
        query_metadata = dict(dict(row.get("metadata") or {}).get("query_metadata") or {})
        gold_refs = gold_refs_from_query_metadata(query_metadata)
        gold_ref_set = set(gold_refs)
        gold_source_message_ids = sorted(
            {
                message_id
                for ref in gold_refs
                for message_id in raw_message_ids_by_ref.get(ref, set())
            }
        )
        gold_trajectory_ids = sorted(
            trajectory_id
            for trajectory_id in sample_to_trajectories.get(sample_id, set())
            if trajectory_refs.get(trajectory_id, set()) & gold_ref_set
        )
        gold_trajectory_set = set(gold_trajectory_ids)
        gold_page_ids = sorted(
            page_id
            for page_id in sample_to_pages.get(sample_id, set())
            if page_types.get(page_id) != "index"
            and gold_trajectory_set & set(page_to_trajectory_ids.get(page_id, []))
        )
        gold_snapshot_ids = sorted(
            snapshot_id
            for snapshot_id, refs in snapshot_refs.items()
            if refs & gold_ref_set
            and trajectory_to_sample.get(snapshot_to_trajectory.get(snapshot_id, "")) == sample_id
        )
        rows.append(
            {
                "schema_version": "gold_labels_v1",
                "sample_id": sample_id,
                "query_task_id": str(row.get("query_task_id") or ""),
                "question": str(row.get("question") or ""),
                "gold_answer": row.get("gold_answer"),
                "gold_source_refs": gold_refs,
                "gold_source_message_ids": gold_source_message_ids,
                "gold_trajectory_ids": gold_trajectory_ids,
                "gold_page_ids": gold_page_ids,
                "gold_snapshot_ids": gold_snapshot_ids,
                "gold_ref_count": len(gold_refs),
                "gold_trajectory_count": len(gold_trajectory_ids),
            }
        )
    return rows


def write_gold_labels_artifact(
    run_dir: Path,
    sample_rows: list[dict[str, Any]],
    *,
    database_path: Path | None = None,
) -> Path:
    if database_path is None:
        _, _, resolved_database_path = (
            load_details_rows(run_dir)
            if (run_dir / "details.json").exists()
            else ({}, [], run_dir / "trajpatch.sqlite")
        )
        database_path = resolved_database_path
    memory_index = build_memory_index(database_path)
    rows = build_gold_label_rows(sample_rows, memory_index)
    path = run_dir / "analysis" / "gold_labels.jsonl"
    if path.exists():
        path.unlink()
    append_jsonl(path, rows)
    return path


def load_or_build_gold_labels(
    run_dir: Path,
    sample_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
) -> list[dict[str, Any]]:
    path = run_dir / "analysis" / "gold_labels.jsonl"
    if path.exists():
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    rows = build_gold_label_rows(sample_rows, memory_index)
    append_jsonl(path, rows)
    return rows
