"""Validated query scopes for sample-filtered offline analyses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryScope:
    """A query subset loaded from an answer-ablation sampling manifest."""

    manifest_path: Path
    manifest_sha256: str
    query_ids: frozenset[str]
    sample_ids: frozenset[str]
    selected_queries: tuple[dict[str, str], ...]
    source_run_dir: Path | None = None
    source_details_sha256: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "sampling_manifest_path": str(self.manifest_path),
            "sampling_manifest_sha256": self.manifest_sha256,
            "sampling_schema_version": "query_scope_v1",
            "scoped_query_count": len(self.query_ids),
            "scoped_sample_count": len(self.sample_ids),
            "source_run_dir": (
                str(self.source_run_dir)
                if self.source_run_dir is not None
                else None
            ),
            "source_details_sha256": self.source_details_sha256,
        }


def _resolve_manifest_path(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "sampling_manifest.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Sampling manifest not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Sampling manifest is not a file: {resolved}")
    return resolved


def load_query_scope(path: Path | str | None) -> QueryScope | None:
    """Load and validate an answer-ablation sampling manifest."""

    if path is None:
        return None
    manifest_path = _resolve_manifest_path(path)
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid sampling manifest JSON: {manifest_path}") from exc
    selected = payload.get("selected_queries")
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            "Sampling manifest must contain a non-empty selected_queries list."
        )

    query_ids: set[str] = set()
    sample_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(selected):
        if not isinstance(row, dict):
            raise ValueError(
                f"selected_queries[{index}] must be a JSON object."
            )
        query_task_id = str(row.get("query_task_id") or "").strip()
        sample_id = str(row.get("sample_id") or "").strip()
        if not query_task_id or not sample_id:
            raise ValueError(
                f"selected_queries[{index}] must include sample_id and query_task_id."
            )
        if query_task_id in query_ids:
            raise ValueError(
                f"Duplicate query_task_id in sampling manifest: {query_task_id}"
            )
        query_ids.add(query_task_id)
        sample_ids.add(sample_id)
        normalized.append(
            {
                "sample_id": sample_id,
                "query_task_id": query_task_id,
                "stratum": str(row.get("stratum") or "").strip(),
            }
        )
    declared_size = payload.get("sample_size_selected")
    if declared_size is not None and int(declared_size) != len(normalized):
        raise ValueError(
            "Sampling manifest sample_size_selected does not match "
            f"selected_queries: {declared_size} != {len(normalized)}."
        )
    source_run_dir: Path | None = None
    source_details_sha256 = str(
        payload.get("source_details_sha256") or ""
    ).strip() or None
    source_run_value = str(payload.get("source_run_dir") or "").strip()
    experiment_manifest_path = manifest_path.parent / "experiment_manifest.json"
    if experiment_manifest_path.exists():
        try:
            experiment_manifest = json.loads(
                experiment_manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid sibling experiment manifest: {experiment_manifest_path}"
            ) from exc
        source_run_value = source_run_value or str(
            experiment_manifest.get("source_run_dir") or ""
        ).strip()
        source_details_sha256 = source_details_sha256 or str(
            experiment_manifest.get("source_details_sha256")
            or dict(experiment_manifest.get("config") or {}).get(
                "source_details_sha256"
            )
            or ""
        ).strip() or None
    if source_run_value:
        source_run_dir = Path(source_run_value).expanduser().resolve()
    return QueryScope(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        query_ids=frozenset(query_ids),
        sample_ids=frozenset(sample_ids),
        selected_queries=tuple(normalized),
        source_run_dir=source_run_dir,
        source_details_sha256=source_details_sha256,
    )


def validate_scope_against_rows(
    scope: QueryScope,
    rows: Iterable[dict[str, Any]],
    *,
    run_dir: Path | str | None = None,
) -> None:
    """Require exact sample/query alignment with the source run."""

    run_pairs: dict[str, str] = {}
    duplicate_ids: set[str] = set()
    for row in rows:
        query_task_id = str(row.get("query_task_id") or "").strip()
        sample_id = str(row.get("sample_id") or "").strip()
        if not query_task_id:
            continue
        if query_task_id in run_pairs:
            duplicate_ids.add(query_task_id)
        run_pairs[query_task_id] = sample_id
    if duplicate_ids:
        raise ValueError(
            "Source run contains duplicate query_task_id values: "
            + ", ".join(sorted(duplicate_ids)[:10])
        )
    missing = sorted(scope.query_ids - set(run_pairs))
    if missing:
        raise ValueError(
            "Sampling manifest queries are missing from the source run: "
            + ", ".join(missing[:10])
        )
    mismatched = sorted(
        row["query_task_id"]
        for row in scope.selected_queries
        if run_pairs.get(row["query_task_id"]) != row["sample_id"]
    )
    if mismatched:
        raise ValueError(
            "Sampling manifest sample/query alignment mismatch: "
            + ", ".join(mismatched[:10])
        )
    if run_dir is None:
        return
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if (
        scope.source_run_dir is not None
        and resolved_run_dir != scope.source_run_dir
    ):
        raise ValueError(
            "Sampling manifest belongs to a different source run: "
            f"{scope.source_run_dir} != {resolved_run_dir}."
        )
    if scope.source_details_sha256:
        details_path = resolved_run_dir / "details.json"
        if not details_path.exists():
            raise FileNotFoundError(details_path)
        actual_sha256 = hashlib.sha256(details_path.read_bytes()).hexdigest()
        if actual_sha256 != scope.source_details_sha256:
            raise ValueError(
                "Sampling manifest source details hash does not match the "
                "requested run."
            )


def filter_query_rows(
    rows: Iterable[dict[str, Any]],
    scope: QueryScope | None,
) -> list[dict[str, Any]]:
    if scope is None:
        return list(rows)
    return [
        row
        for row in rows
        if str(row.get("query_task_id") or "").strip() in scope.query_ids
    ]


def scoped_analysis_dir(run_dir: Path, scope: QueryScope | None) -> Path:
    """Keep subset reports beside their sampling manifest, not in full-run analysis."""

    if scope is None:
        return run_dir / "analysis"
    path = scope.manifest_path.parent / "offline_analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path
