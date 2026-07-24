"""Sample-scoped helpers for analysis-time memory indexes.

LOCOMO source references such as ``D1:1`` are only unique within a dialogue
sample.  Analysis code must therefore resolve references with the owning
``sample_id`` instead of treating them as run-global identifiers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def build_sample_scoped_ref_indexes(
    *,
    raw_messages_by_id: dict[str, dict[str, Any]],
    trajectory_refs: dict[str, set[str]],
    trajectory_to_sample: dict[str, str],
) -> dict[str, Any]:
    """Build reference indexes whose keys include the owning sample."""

    sample_raw_message_ids_by_ref: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for message_id, message in raw_messages_by_id.items():
        sample_id = str(message.get("sample_id") or "")
        source_ref = str(message.get("source_ref") or "")
        if sample_id and source_ref:
            sample_raw_message_ids_by_ref[sample_id][source_ref].add(str(message_id))

    sample_source_ref_to_trajectories: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for trajectory_id, refs in trajectory_refs.items():
        sample_id = str(trajectory_to_sample.get(str(trajectory_id)) or "")
        if not sample_id:
            continue
        for source_ref in refs:
            if source_ref:
                sample_source_ref_to_trajectories[sample_id][str(source_ref)].add(
                    str(trajectory_id)
                )

    return {
        "sample_raw_message_ids_by_ref": {
            sample_id: {source_ref: set(ids) for source_ref, ids in by_ref.items()}
            for sample_id, by_ref in sample_raw_message_ids_by_ref.items()
        },
        "sample_source_ref_to_trajectories": {
            sample_id: {
                source_ref: set(trajectory_ids)
                for source_ref, trajectory_ids in by_ref.items()
            }
            for sample_id, by_ref in sample_source_ref_to_trajectories.items()
        },
    }


def source_message_ids_for_refs(
    memory_index: dict[str, Any],
    *,
    sample_id: str,
    source_refs: Iterable[str],
) -> list[str]:
    """Resolve refs inside one sample, with a filtered legacy fallback."""

    sample_id = str(sample_id or "")
    scoped = dict(memory_index.get("sample_raw_message_ids_by_ref") or {}).get(
        sample_id
    )
    refs = {str(ref) for ref in source_refs if str(ref).strip()}
    if scoped is not None:
        return sorted(
            {
                str(message_id)
                for source_ref in refs
                for message_id in dict(scoped).get(source_ref, set())
                if str(message_id).strip()
            }
        )

    raw_messages_by_id = dict(memory_index.get("raw_messages_by_id") or {})
    legacy = dict(memory_index.get("raw_message_ids_by_ref") or {})
    return sorted(
        {
            str(message_id)
            for source_ref in refs
            for message_id in legacy.get(source_ref, set())
            if str(message_id).strip()
            and str(raw_messages_by_id.get(str(message_id), {}).get("sample_id") or "")
            == sample_id
        }
    )


def trajectory_ids_for_source_ref(
    memory_index: dict[str, Any],
    *,
    sample_id: str,
    source_ref: str,
) -> list[str]:
    """Resolve source-linked trajectories inside one sample."""

    sample_id = str(sample_id or "")
    source_ref = str(source_ref or "")
    scoped = dict(memory_index.get("sample_source_ref_to_trajectories") or {}).get(
        sample_id
    )
    if scoped is not None:
        return sorted(str(item) for item in dict(scoped).get(source_ref, set()))

    trajectory_to_sample = dict(memory_index.get("trajectory_to_sample") or {})
    legacy = dict(memory_index.get("source_ref_to_trajectories") or {})
    return sorted(
        str(trajectory_id)
        for trajectory_id in legacy.get(source_ref, set())
        if str(trajectory_to_sample.get(str(trajectory_id)) or "") == sample_id
    )


def versioned_analysis_path(
    run_dir: Path,
    *,
    filename: str,
    accepted_schema_versions: set[str],
) -> Path:
    """Choose a non-destructive v2 path when an incompatible v1 file exists."""

    primary = run_dir / "analysis" / filename
    versioned = run_dir / "analysis_v2" / filename
    if versioned.exists():
        return versioned
    if not primary.exists():
        return primary

    first_schema = ""
    try:
        for line in primary.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            import json

            first_schema = str(json.loads(line).get("schema_version") or "")
            break
    except (OSError, ValueError, TypeError):
        first_schema = ""
    if first_schema in accepted_schema_versions:
        return primary
    return versioned
