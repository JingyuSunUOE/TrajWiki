"""Offline semantic drift diagnostics for memory trajectories.

The helpers in this module only read persisted embedding vectors and trajectory
ordering metadata. They intentionally do not call an embedding provider or LLM.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

LOW_ADJACENT_COSINE_THRESHOLD = 0.50
LOW_DRIFT_EXAMPLE_LIMIT = 10
DRIFT_SCHEMA_VERSION = "trajectory_semantic_drift_v1"
LENGTH_BUCKET_ORDER = ("1", "2-3", "4-6", "7-10", "11-15", "16+")


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _safe_median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _safe_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((float(percentile) / 100.0) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _numeric_stats(values: Iterable[Any]) -> dict[str, Any]:
    numeric_values: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numeric_values.append(number)
    return {
        "count": len(numeric_values),
        "mean": _safe_mean(numeric_values),
        "median": _safe_median(numeric_values),
        "p10": _safe_percentile(numeric_values, 10),
        "p25": _safe_percentile(numeric_values, 25),
        "p75": _safe_percentile(numeric_values, 75),
        "p90": _safe_percentile(numeric_values, 90),
        "min": min(numeric_values) if numeric_values else None,
        "max": max(numeric_values) if numeric_values else None,
    }


def _parse_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    vector: list[float] = []
    for item in parsed:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        vector.append(number)
    return vector or None


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _embedding_record(vector: Sequence[float], *, model_name: str = "test-model") -> dict[str, Any]:
    return {
        "vector": [float(value) for value in vector],
        "norm": _norm(vector),
        "model_name": model_name,
    }


def cosine_from_embeddings(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> float | None:
    if not left or not right:
        return None
    left_vector = left.get("vector")
    right_vector = right.get("vector")
    if not isinstance(left_vector, list) or not isinstance(right_vector, list):
        return None
    if len(left_vector) != len(right_vector) or not left_vector:
        return None
    try:
        left_norm = float(left.get("norm") or _norm(left_vector))
        right_norm = float(right.get("norm") or _norm(right_vector))
    except (TypeError, ValueError):
        return None
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left_vector, right_vector, strict=True))
    cosine = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, float(cosine)))


def _centroid_embedding(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    vectors = [record.get("vector") for record in records if isinstance(record.get("vector"), list)]
    if not vectors:
        return None
    dimension = len(vectors[0])
    if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
        return None
    centroid = [sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(dimension)]
    return _embedding_record(centroid, model_name=str(records[0].get("model_name") or "unknown"))


def _drift_bucket(snapshot_count: int, head_tail_cosine: float | None, embedding_available: bool) -> str:
    if snapshot_count == 1 and embedding_available:
        return "singleton"
    if not embedding_available or head_tail_cosine is None:
        return "missing_embedding"
    if head_tail_cosine >= 0.70:
        return "stable"
    if head_tail_cosine >= 0.55:
        return "moderate_span"
    if head_tail_cosine >= 0.40:
        return "high_span"
    return "possible_drift"


def length_bucket(snapshot_count: int) -> str:
    if snapshot_count <= 1:
        return "1"
    if snapshot_count <= 3:
        return "2-3"
    if snapshot_count <= 6:
        return "4-6"
    if snapshot_count <= 10:
        return "7-10"
    if snapshot_count <= 15:
        return "11-15"
    return "16+"


def _embedding_model_name(records: Sequence[Mapping[str, Any]]) -> str | None:
    for record in records:
        model_name = str(record.get("model_name") or "").strip()
        if model_name:
            return model_name
    return None


def build_trajectory_drift_rows_from_embeddings(
    *,
    trajectory_to_sample: Mapping[str, str],
    trajectory_snapshot_ids_ordered: Mapping[str, Sequence[str]],
    snapshot_versions: Mapping[str, int],
    snapshot_embeddings: Mapping[str, Mapping[str, Any]],
    summary_embeddings: Mapping[str, Mapping[str, Any]],
    low_adjacent_cosine_threshold: float = LOW_ADJACENT_COSINE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Build compact drift rows from preloaded embeddings.

    The returned rows never include embedding vectors or source text.
    """

    rows: list[dict[str, Any]] = []
    for trajectory_id in sorted(trajectory_to_sample):
        snapshot_ids = [str(snapshot_id) for snapshot_id in trajectory_snapshot_ids_ordered.get(trajectory_id, [])]
        snapshot_count = len(snapshot_ids)
        sample_id = str(trajectory_to_sample.get(trajectory_id) or "")
        snapshot_records = [snapshot_embeddings.get(snapshot_id) for snapshot_id in snapshot_ids]
        complete_snapshot_embeddings = bool(snapshot_ids) and all(record is not None for record in snapshot_records)
        head_snapshot_id = snapshot_ids[0] if snapshot_ids else None
        tail_snapshot_id = snapshot_ids[-1] if snapshot_ids else None
        head_record = snapshot_embeddings.get(head_snapshot_id or "")
        tail_record = snapshot_embeddings.get(tail_snapshot_id or "")
        head_tail_cosine = cosine_from_embeddings(head_record, tail_record) if complete_snapshot_embeddings else None

        adjacent_cosines: list[float] = []
        low_pairs: list[dict[str, Any]] = []
        if complete_snapshot_embeddings and len(snapshot_ids) >= 2:
            for left_id, right_id in zip(snapshot_ids, snapshot_ids[1:], strict=False):
                cosine = cosine_from_embeddings(snapshot_embeddings.get(left_id), snapshot_embeddings.get(right_id))
                if cosine is None:
                    complete_snapshot_embeddings = False
                    adjacent_cosines = []
                    low_pairs = []
                    head_tail_cosine = None
                    break
                adjacent_cosines.append(cosine)
                if cosine < low_adjacent_cosine_threshold:
                    low_pairs.append(
                        {
                            "from_snapshot_id": left_id,
                            "to_snapshot_id": right_id,
                            "from_version": snapshot_versions.get(left_id),
                            "to_version": snapshot_versions.get(right_id),
                            "cosine": cosine,
                        }
                    )

        summary_record = summary_embeddings.get(trajectory_id)
        centroid_record = (
            _centroid_embedding([record for record in snapshot_records if record is not None])
            if complete_snapshot_embeddings
            else None
        )
        summary_head_cosine = cosine_from_embeddings(summary_record, head_record) if complete_snapshot_embeddings else None
        summary_tail_cosine = cosine_from_embeddings(summary_record, tail_record) if complete_snapshot_embeddings else None
        summary_centroid_cosine = cosine_from_embeddings(summary_record, centroid_record) if centroid_record else None
        embedding_available = bool(complete_snapshot_embeddings)
        bucket = _drift_bucket(snapshot_count, head_tail_cosine, embedding_available)
        model_name = _embedding_model_name(
            [record for record in [head_record, tail_record, summary_record] if record is not None]
        )
        rows.append(
            {
                "schema_version": DRIFT_SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "sample_id": sample_id,
                "snapshot_count": snapshot_count,
                "embedding_model": model_name,
                "embedding_available": embedding_available,
                "head_snapshot_id": head_snapshot_id,
                "tail_snapshot_id": tail_snapshot_id,
                "head_tail_cosine": head_tail_cosine,
                "head_tail_distance": (1.0 - head_tail_cosine) if head_tail_cosine is not None else None,
                "adjacent_mean_cosine": _safe_mean(adjacent_cosines),
                "adjacent_min_cosine": min(adjacent_cosines) if adjacent_cosines else None,
                "adjacent_min_cosine_stats": _numeric_stats(adjacent_cosines),
                "adjacent_max_distance": (1.0 - min(adjacent_cosines)) if adjacent_cosines else None,
                "summary_head_cosine": summary_head_cosine,
                "summary_tail_cosine": summary_tail_cosine,
                "summary_centroid_cosine": summary_centroid_cosine,
                "low_similarity_update_pair_count": len(low_pairs),
                "low_similarity_update_pairs": low_pairs,
                "drift_bucket": bucket,
                "length_bucket": length_bucket(snapshot_count),
            }
        )
    return rows


def _load_embedding_maps(database_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    snapshot_embeddings: dict[str, dict[str, Any]] = {}
    summary_embeddings: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        for row in connection.execute(
            """
            SELECT owner_type, owner_id, model_name, vector_json, norm
            FROM embeddings
            WHERE owner_type IN ('snapshot', 'trajectory_summary')
            """
        ):
            vector = _parse_vector(row["vector_json"])
            if vector is None:
                continue
            try:
                norm = float(row["norm"] or _norm(vector))
            except (TypeError, ValueError):
                norm = _norm(vector)
            record = {
                "vector": vector,
                "norm": norm,
                "model_name": str(row["model_name"] or ""),
            }
            owner_id = str(row["owner_id"])
            owner_type = str(row["owner_type"])
            if owner_type == "snapshot":
                snapshot_embeddings[owner_id] = record
            elif owner_type == "trajectory_summary":
                summary_embeddings[owner_id] = record
    finally:
        connection.close()
    return snapshot_embeddings, summary_embeddings


def build_trajectory_drift_rows(
    *,
    database_path: Path,
    trajectory_to_sample: Mapping[str, str],
    trajectory_snapshot_ids_ordered: Mapping[str, Sequence[str]],
    snapshot_versions: Mapping[str, int],
) -> list[dict[str, Any]]:
    snapshot_embeddings, summary_embeddings = _load_embedding_maps(database_path)
    return build_trajectory_drift_rows_from_embeddings(
        trajectory_to_sample=trajectory_to_sample,
        trajectory_snapshot_ids_ordered=trajectory_snapshot_ids_ordered,
        snapshot_versions=snapshot_versions,
        snapshot_embeddings=snapshot_embeddings,
        summary_embeddings=summary_embeddings,
    )


def trajectory_drift_fields_for_gold_trajectories(
    gold_trajectory_ids: Sequence[str],
    drift_rows_by_trajectory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        drift_rows_by_trajectory[str(trajectory_id)]
        for trajectory_id in gold_trajectory_ids
        if str(trajectory_id) in drift_rows_by_trajectory
    ]
    head_tail_values = [
        float(row["head_tail_cosine"])
        for row in rows
        if row.get("head_tail_cosine") is not None
    ]
    adjacent_min_values = [
        float(row["adjacent_min_cosine"])
        for row in rows
        if row.get("adjacent_min_cosine") is not None
    ]
    summary_tail_values = [
        float(row["summary_tail_cosine"])
        for row in rows
        if row.get("summary_tail_cosine") is not None
    ]
    buckets = Counter(str(row.get("drift_bucket") or "unknown") for row in rows)
    possible_count = int(buckets.get("possible_drift", 0))
    high_span_count = int(buckets.get("high_span", 0))
    low_pair_count = sum(int(row.get("low_similarity_update_pair_count") or 0) for row in rows)
    return {
        "gold_trajectory_head_tail_cosine_min": min(head_tail_values) if head_tail_values else None,
        "gold_trajectory_head_tail_cosine_mean": _safe_mean(head_tail_values),
        "gold_trajectory_adjacent_min_cosine_min": min(adjacent_min_values) if adjacent_min_values else None,
        "gold_trajectory_summary_tail_cosine_min": min(summary_tail_values) if summary_tail_values else None,
        "gold_trajectory_possible_drift_count": possible_count,
        "gold_trajectory_high_span_count": high_span_count,
        "gold_trajectory_drift_buckets": dict(sorted(buckets.items())),
        "gold_trajectory_low_similarity_update_pair_count": low_pair_count,
        "trajectory_drift_risk_observed": bool(possible_count or high_span_count),
    }


def _aggregate_drift_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trajectory_count = len(rows)
    embedding_available_count = sum(1 for row in rows if bool(row.get("embedding_available")))
    missing_embedding_count = sum(1 for row in rows if row.get("drift_bucket") == "missing_embedding")
    singleton_count = sum(1 for row in rows if row.get("drift_bucket") == "singleton")
    bucket_counts = Counter(str(row.get("drift_bucket") or "unknown") for row in rows)
    possible_or_high_count = int(bucket_counts.get("possible_drift", 0) + bucket_counts.get("high_span", 0))
    return {
        "trajectory_count": trajectory_count,
        "embedding_available_count": embedding_available_count,
        "missing_embedding_count": missing_embedding_count,
        "missing_embedding_rate": _safe_rate(missing_embedding_count, trajectory_count),
        "singleton_trajectory_count": singleton_count,
        "singleton_trajectory_rate": _safe_rate(singleton_count, trajectory_count),
        "non_singleton_trajectory_count": max(0, trajectory_count - singleton_count),
        "head_tail_cosine_stats": _numeric_stats(row.get("head_tail_cosine") for row in rows),
        "adjacent_mean_cosine_stats": _numeric_stats(row.get("adjacent_mean_cosine") for row in rows),
        "adjacent_min_cosine_stats": _numeric_stats(row.get("adjacent_min_cosine") for row in rows),
        "summary_head_cosine_stats": _numeric_stats(row.get("summary_head_cosine") for row in rows),
        "summary_tail_cosine_stats": _numeric_stats(row.get("summary_tail_cosine") for row in rows),
        "summary_centroid_cosine_stats": _numeric_stats(row.get("summary_centroid_cosine") for row in rows),
        "drift_bucket_counts": dict(sorted(bucket_counts.items())),
        "possible_or_high_span_count": possible_or_high_count,
        "possible_or_high_span_rate": _safe_rate(possible_or_high_count, trajectory_count),
    }


def _aggregate_query_drift(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    risk_count = sum(1 for row in rows if bool(row.get("trajectory_drift_risk_observed")))
    bucket_counts: Counter[str] = Counter()
    for row in rows:
        bucket_counts.update(dict(row.get("gold_trajectory_drift_buckets") or {}))
    return {
        "query_count": len(rows),
        "trajectory_drift_risk_count": risk_count,
        "trajectory_drift_risk_rate": _safe_rate(risk_count, len(rows)),
        "gold_head_tail_min_stats": _numeric_stats(
            row.get("gold_trajectory_head_tail_cosine_min") for row in rows
        ),
        "gold_head_tail_mean_stats": _numeric_stats(
            row.get("gold_trajectory_head_tail_cosine_mean") for row in rows
        ),
        "gold_adjacent_min_stats": _numeric_stats(
            row.get("gold_trajectory_adjacent_min_cosine_min") for row in rows
        ),
        "gold_summary_tail_min_stats": _numeric_stats(
            row.get("gold_trajectory_summary_tail_cosine_min") for row in rows
        ),
        "gold_possible_drift_count": sum(
            int(row.get("gold_trajectory_possible_drift_count") or 0) for row in rows
        ),
        "gold_high_span_count": sum(int(row.get("gold_trajectory_high_span_count") or 0) for row in rows),
        "gold_low_similarity_update_pair_count": sum(
            int(row.get("gold_trajectory_low_similarity_update_pair_count") or 0) for row in rows
        ),
        "gold_drift_bucket_counts": dict(sorted(bucket_counts.items())),
    }


def _group_query_drift(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key) or "unknown").lower()
        grouped[value].append(row)
    return {value: _aggregate_query_drift(group_rows) for value, group_rows in sorted(grouped.items())}


def _drift_by_length_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {bucket: [] for bucket in LENGTH_BUCKET_ORDER}
    for row in rows:
        grouped.setdefault(str(row.get("length_bucket") or "unknown"), []).append(row)
    output: dict[str, Any] = {}
    for bucket, bucket_rows in grouped.items():
        if not bucket_rows:
            continue
        aggregate = _aggregate_drift_rows(bucket_rows)
        output[bucket] = {
            "trajectory_count": aggregate["trajectory_count"],
            "embedding_available_count": aggregate["embedding_available_count"],
            "missing_embedding_count": aggregate["missing_embedding_count"],
            "head_tail_cosine_stats": aggregate["head_tail_cosine_stats"],
            "adjacent_mean_cosine_stats": aggregate["adjacent_mean_cosine_stats"],
            "adjacent_min_cosine_stats": aggregate["adjacent_min_cosine_stats"],
            "summary_tail_cosine_stats": aggregate["summary_tail_cosine_stats"],
            "drift_bucket_counts": aggregate["drift_bucket_counts"],
            "possible_or_high_span_count": aggregate["possible_or_high_span_count"],
            "possible_or_high_span_rate": aggregate["possible_or_high_span_rate"],
        }
    return output


def _low_drift_examples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("head_tail_cosine") is not None
        and str(row.get("drift_bucket") or "") not in {"singleton", "missing_embedding"}
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (
            float(row.get("head_tail_cosine") or 1.0),
            str(row.get("trajectory_id") or ""),
        ),
    )
    examples: list[dict[str, Any]] = []
    for row in ordered[:LOW_DRIFT_EXAMPLE_LIMIT]:
        examples.append(
            {
                "trajectory_id": row.get("trajectory_id"),
                "sample_id": row.get("sample_id"),
                "snapshot_count": row.get("snapshot_count"),
                "head_tail_cosine": row.get("head_tail_cosine"),
                "adjacent_min_cosine": row.get("adjacent_min_cosine"),
                "summary_tail_cosine": row.get("summary_tail_cosine"),
                "drift_bucket": row.get("drift_bucket"),
                "low_similarity_update_pair_count": row.get("low_similarity_update_pair_count"),
                "low_similarity_update_pairs": row.get("low_similarity_update_pairs"),
            }
        )
    return examples


def build_trajectory_drift_diagnostics(
    *,
    trajectory_drift_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    failed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_rows = list(trajectory_drift_rows)
    diagnostics = {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "low_adjacent_cosine_threshold": LOW_ADJACENT_COSINE_THRESHOLD,
        **_aggregate_drift_rows(all_rows),
        "drift_by_length_bucket": _drift_by_length_bucket(all_rows),
        "drift_by_query_verdict": _group_query_drift(query_rows, "judge_verdict"),
        "drift_by_failure_reason": _group_query_drift(failed_rows, "reason"),
        "gold_trajectory_drift_stats": {
            "all_queries": _aggregate_query_drift(query_rows),
            "failed_queries": _aggregate_query_drift(failed_rows),
        },
        "low_drift_trajectory_examples": _low_drift_examples(all_rows),
    }
    return diagnostics


def compact_query_drift_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "sample_id": row.get("sample_id"),
        "query_task_id": row.get("query_task_id"),
        "question": row.get("question"),
        "judge_verdict": row.get("judge_verdict"),
        "reason": row.get("reason"),
        "gold_trajectory_ids": row.get("gold_trajectory_ids"),
        "gold_trajectory_head_tail_cosine_min": row.get("gold_trajectory_head_tail_cosine_min"),
        "gold_trajectory_head_tail_cosine_mean": row.get("gold_trajectory_head_tail_cosine_mean"),
        "gold_trajectory_adjacent_min_cosine_min": row.get("gold_trajectory_adjacent_min_cosine_min"),
        "gold_trajectory_summary_tail_cosine_min": row.get("gold_trajectory_summary_tail_cosine_min"),
        "gold_trajectory_possible_drift_count": row.get("gold_trajectory_possible_drift_count"),
        "gold_trajectory_high_span_count": row.get("gold_trajectory_high_span_count"),
        "gold_trajectory_drift_buckets": row.get("gold_trajectory_drift_buckets"),
        "gold_trajectory_low_similarity_update_pair_count": row.get(
            "gold_trajectory_low_similarity_update_pair_count"
        ),
        "trajectory_drift_risk_observed": row.get("trajectory_drift_risk_observed"),
    }

