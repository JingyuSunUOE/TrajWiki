"""Resumable answer-level ablations over a completed LOCOMO TrajWiki run."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

from trajpatch.analysis.auditability import (
    invalid_refs_from_answer_metadata,
    support_refs_from_answer_metadata,
)
from trajpatch.analysis.context_cost import TOKEN_ESTIMATOR_NAME
from trajpatch.analysis.gold_labels import (
    extract_source_refs,
    load_details_rows,
    load_or_build_gold_labels,
)
from trajpatch.config import ProviderKind, RunConfig
from trajpatch.exceptions import ProviderConfigurationError
from trajpatch.experiments.progress import ExperimentProgress
from trajpatch.experiments.statistics import (
    complete_prevalence_weighted_metric,
    dialogue_cluster_paired_bootstrap_rows,
    judge_accuracy,
    judge_score,
    paired_bootstrap_rows,
    prevalence_weighted_paired_bootstrap_rows,
    stratum_summary_rows,
)
from trajpatch.experiments.token_budget import build_token_counter
from trajpatch.experiments.variant_contexts import (
    VARIANT_POLICIES,
    VariantContextBuilder,
)
from trajpatch.providers.factory import build_embedding_provider, build_llm_provider
from trajpatch.providers.metering import MeteredLLMProvider
from trajpatch.providers.structured_outputs import infer_remote_vendor
from trajpatch.types import NormalizedMessage
from trajpatch.utils.env import load_runtime_env
from trajpatch.utils.metrics import bleu1, normalize_answer

DEFAULT_VARIANTS = [
    "full",
    "direct_trajectory",
    "latest_snapshot",
    "hybrid_raw_rag",
    "wiki_summaries",
    "no_claim_state",
    "no_source_constraint",
    "full_context",
    "naive_dense_rag",
    "full_context_matched",
    "hybrid_raw_rag_matched",
]
DERIVED_VARIANTS = [
    "no_answer_validation_or_repair",
    "no_retrieval_retry",
]
OBSERVED_DIAGNOSTIC_VARIANTS = ["observed_full_pipeline"]
INDEPENDENT_JUDGE_PROMPT_VERSION = "independent_answer_judge_v1"
ANSWER_PROMPT_VERSION = "answer_ablation_neutral_v1"
VARIANT_DISPLAY_NAMES = {
    "full": "Full TrajWiki",
    "direct_trajectory": "No Wiki Routing",
    "latest_snapshot": "Latest Snapshot Only",
    "hybrid_raw_rag": "Budgeted Flat Raw Memory",
    "hybrid_raw_rag_matched": "Flat Raw Memory (TrajWiki-Matched)",
    "wiki_summaries": "Wiki Summaries Only",
    "no_claim_state": "Lifecycle State Hidden",
    "no_source_constraint": "No Source-Support Constraint",
    "full_context": "Full Context (32K)",
    "full_context_matched": "Full Context (TrajWiki-Matched)",
    "naive_dense_rag": "Naive Dense RAG",
    "observed_full_pipeline": "Observed Full Pipeline",
    "no_answer_validation_or_repair": "Observed Before Validation/Repair",
    "no_retrieval_retry": "Observed Before Retrieval Retry",
    "mem0_saved": "Mem0 (Saved External)",
}
SAMPLING_PROFILES: dict[str, dict[str, Any]] = {
    "rebuttal_60_v1": {
        "sample_size": 60,
        "quotas": {
            "strict_deep_history": 9,
            "update_sensitive": 26,
            "ordinary": 25,
        },
        "expected_population": {
            "strict_deep_history": 9,
            "update_sensitive": 188,
            "ordinary": 85,
        },
        "sampling_status": "pilot",
    },
    "rebuttal_200_v1": {
        "sample_size": 200,
        "quotas": {
            "strict_deep_history": 9,
            "update_sensitive": 132,
            "ordinary": 59,
        },
        "expected_population": {
            "strict_deep_history": 9,
            "update_sensitive": 188,
            "ordinary": 85,
        },
        "sampling_status": "post_hoc_nested_extension",
    },
}


def _comparison_group(variant: str, baseline_methods: set[str]) -> str:
    if variant == "full":
        return "controlled_reference"
    if variant in OBSERVED_DIAGNOSTIC_VARIANTS:
        return "observed_pipeline"
    if variant in DERIVED_VARIANTS:
        return "observed_pipeline_stage_diagnostic"
    if variant in baseline_methods:
        return "historical_external_baseline"
    if variant in {"full_context", "hybrid_raw_rag"}:
        return "32k_upper_bound"
    if variant in {
        "full_context_matched",
        "hybrid_raw_rag_matched",
    }:
        return "token_matched_baseline"
    if variant == "naive_dense_rag":
        return "strong_rag_baseline"
    return "component_ablation"


def _parse_list(value: str | Iterable[str] | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = [str(item).strip() for item in value if str(item).strip()]
    return list(dict.fromkeys(values)) or list(default)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-") or "item"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            payload, handle, indent=2, ensure_ascii=True, sort_keys=True, default=str
        )
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, default=str)
    temp_path.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, sort_keys=True, default=str)
            )
            handle.write("\n")
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _package_versions() -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for package in [
        "trajwiki",
        "pydantic",
        "litellm",
        "openai",
        "anthropic",
        "numpy",
        "torch",
        "transformers",
        "sentence-transformers",
        "tiktoken",
    ]:
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = None
    return output


def _source_tree_sha256(repository: Path) -> str:
    hasher = hashlib.sha256()
    candidates = sorted((repository / "src" / "trajpatch").rglob("*.py"))
    candidates.append(repository / "pyproject.toml")
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _git_state(run_dir: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    source_tree_sha256 = _source_tree_sha256(repository)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": os.getenv("TRAJWIKI_GIT_COMMIT") or None,
            "dirty": (
                os.getenv("TRAJWIKI_GIT_DIRTY") == "1"
                if os.getenv("TRAJWIKI_GIT_DIRTY") is not None
                else None
            ),
            "dirty_hash": None,
            "source_tree_sha256": source_tree_sha256,
            "controller_source_hash": (
                os.getenv("TRAJWIKI_SOURCE_HASH") or None
            ),
            "image_ref": os.getenv("TRAJWIKI_IMAGE_REF") or None,
            "workflow_id": os.getenv("TRAJWIKI_WORKFLOW_ID") or None,
        }
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_hash": _sha256_bytes(status.encode("utf-8")) if status else None,
        "source_tree_sha256": source_tree_sha256,
        "controller_source_hash": os.getenv("TRAJWIKI_SOURCE_HASH") or None,
        "source_run_dir": str(run_dir),
    }


def _stratum_for_gold(
    gold: dict[str, Any],
    memory_index: dict[str, Any],
) -> str:
    sample_id = str(gold.get("sample_id") or "")
    gold_refs = {
        str(item)
        for item in list(gold.get("gold_source_refs") or [])
        if str(item).strip()
    }
    gold_source_message_ids = {
        str(item)
        for item in list(gold.get("gold_source_message_ids") or [])
        if str(item).strip()
    }

    latest_refs_by_sample = memory_index.get("_rebuttal_latest_refs_by_sample")
    if latest_refs_by_sample is None:
        latest_refs_by_sample = defaultdict(set)
        for trajectory_id, snapshot_ids in memory_index[
            "trajectory_to_snapshots"
        ].items():
            if not snapshot_ids:
                continue
            owner = str(memory_index["trajectory_to_sample"].get(trajectory_id) or "")
            latest_refs_by_sample[owner].update(
                memory_index["snapshot_refs"].get(snapshot_ids[-1], set())
            )
        memory_index["_rebuttal_latest_refs_by_sample"] = latest_refs_by_sample

    # These are the cases for which latest-snapshot evidence is structurally
    # unable to expose every gold source, independent of a query-time ranker.
    if gold_refs - set(latest_refs_by_sample.get(sample_id, set())):
        return "strict_deep_history"

    deprecated_source_ids = memory_index.get("_rebuttal_deprecated_source_message_ids")
    if deprecated_source_ids is None:
        deprecated_source_ids = {
            str(message_id)
            for claims in memory_index["claims_by_snapshot"].values()
            for claim in claims
            if str(claim.get("status") or "").lower() == "deprecated"
            for message_id in list(claim.get("source_message_ids") or [])
            if str(message_id).strip()
        }
        memory_index["_rebuttal_deprecated_source_message_ids"] = deprecated_source_ids

    if gold_source_message_ids & set(deprecated_source_ids):
        return "update_sensitive"
    return "ordinary"


def _sample_queries(
    *,
    sample_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    memory_index: dict[str, Any],
    sample_size: int,
    seed: int,
    sampling_profile: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_query = {str(row.get("query_task_id")): row for row in gold_rows}
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        query_task_id = str(row.get("query_task_id") or "")
        enriched = dict(row)
        enriched["rebuttal_stratum"] = _stratum_for_gold(
            gold_by_query.get(query_task_id, {}),
            memory_index,
        )
        strata[enriched["rebuttal_stratum"]].append(enriched)
    for values in strata.values():
        values.sort(
            key=lambda row: (str(row.get("sample_id")), str(row.get("query_task_id")))
        )

    target = min(max(1, int(sample_size)), len(sample_rows))
    requested_profile = str(sampling_profile or "auto").strip().lower()
    if requested_profile not in {"auto", *SAMPLING_PROFILES}:
        raise ValueError(
            "sampling_profile must be auto, rebuttal_60_v1, or rebuttal_200_v1."
        )
    resolved_profile = requested_profile
    if resolved_profile == "auto":
        resolved_profile = {
            60: "rebuttal_60_v1",
            200: "rebuttal_200_v1",
        }.get(target, "adaptive_v1")
    profile = SAMPLING_PROFILES.get(resolved_profile)
    if profile is not None:
        if target != int(profile["sample_size"]):
            raise ValueError(
                f"sampling_profile={resolved_profile} requires "
                f"sample_size={profile['sample_size']}, received {sample_size}."
            )
        quotas = {
            str(key): int(value)
            for key, value in dict(profile["quotas"]).items()
        }
        population_counts_for_validation = {
            stratum: len(strata[stratum])
            for stratum in [
                "strict_deep_history",
                "update_sensitive",
                "ordinary",
            ]
        }
        expected_population = dict(profile.get("expected_population") or {})
        if (
            requested_profile != "auto"
            and
            expected_population
            and population_counts_for_validation != expected_population
        ):
            raise ValueError(
                f"sampling_profile={resolved_profile} expects population "
                f"{expected_population}, observed {population_counts_for_validation}."
            )
        shortages = {
            stratum: quotas[stratum] - len(strata[stratum])
            for stratum in quotas
            if len(strata[stratum]) < quotas[stratum]
        }
        if shortages:
            raise ValueError(
                f"The fixed {resolved_profile} sample cannot satisfy its stratum "
                f"quotas; shortages={shortages}."
            )
    else:
        strict_target = min(
            len(strata["strict_deep_history"]), max(1, round(target * 0.15))
        )
        update_target = min(len(strata["update_sensitive"]), round(target * 0.43))
        quotas = {
            "strict_deep_history": strict_target,
            "update_sensitive": update_target,
            "ordinary": max(0, target - strict_target - update_target),
        }
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for stratum in ["strict_deep_history", "update_sensitive", "ordinary"]:
        candidates = list(strata[stratum])
        rng.shuffle(candidates)
        for row in candidates[: min(quotas[stratum], len(candidates))]:
            selected.append(row)
            selected_ids.add(str(row.get("query_task_id")))
    if len(selected) < target:
        remainder = [
            row
            for row in sample_rows
            if str(row.get("query_task_id")) not in selected_ids
        ]
        rng.shuffle(remainder)
        for row in remainder[: target - len(selected)]:
            enriched = dict(row)
            query_task_id = str(row.get("query_task_id") or "")
            enriched["rebuttal_stratum"] = _stratum_for_gold(
                gold_by_query.get(query_task_id, {}),
                memory_index,
            )
            selected.append(enriched)
    selected.sort(
        key=lambda row: (str(row.get("sample_id")), str(row.get("query_task_id")))
    )
    population_counts = Counter(
        _stratum_for_gold(
            gold_by_query.get(str(row.get("query_task_id")), {}), memory_index
        )
        for row in sample_rows
    )
    population_total = sum(population_counts.values()) or 1
    selected_counts = dict(
        Counter(str(row.get("rebuttal_stratum")) for row in selected)
    )
    return selected, {
        "schema_version": "answer_ablation_sampling_v2",
        "sample_size_requested": sample_size,
        "sample_size_selected": len(selected),
        "sample_seed": seed,
        "sampling_profile_requested": requested_profile,
        "sampling_profile": resolved_profile,
        "sampling_status": (
            str(profile.get("sampling_status"))
            if profile is not None
            else "exploratory_adaptive_sample"
        ),
        "quota_policy": selected_counts,
        "requested_quota_policy": quotas,
        "stratum_definitions": {
            "strict_deep_history": (
                "At least one gold source reference is absent from every latest "
                "snapshot in the owning dialogue sample."
            ),
            "update_sensitive": (
                "Not strict-deep, and at least one gold source message supports a "
                "claim state recorded as deprecated after revision."
            ),
            "ordinary": "Neither strict-deep nor update-sensitive.",
        },
        "selected_counts_by_stratum": selected_counts,
        "population_counts_by_stratum": dict(population_counts),
        "population_prevalence": {
            stratum: count / population_total
            for stratum, count in population_counts.items()
        },
        "selected_queries": [
            {
                "sample_id": row.get("sample_id"),
                "query_task_id": row.get("query_task_id"),
                "stratum": row.get("rebuttal_stratum"),
            }
            for row in selected
        ],
    }


def _provider_records(provider: MeteredLLMProvider) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for call in provider.calls_snapshot():
        metadata = dict(call.metadata or {})
        prompt_usage_available = metadata.get(
            "provider_prompt_usage_available"
        ) is not False
        completion_usage_available = metadata.get(
            "provider_completion_usage_available"
        ) is not False
        prompt_tokens = call.prompt_tokens if prompt_usage_available else None
        completion_tokens = (
            call.completion_tokens if completion_usage_available else None
        )
        records.append(
            {
                "schema_version": "experiment_provider_call_v1",
                "role": call.role,
                "task": call.task,
                "provider_call_id": call.provider_call_id,
                "provider_call_uid": metadata.get("provider_call_uid"),
                "call_item_uid": metadata.get("call_item_uid"),
                "logical_call_item_uid": metadata.get("logical_call_item_uid"),
                "sample_id": metadata.get("sample_id"),
                "query_task_id": metadata.get("query_task_id"),
                "variant": metadata.get("variant"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    prompt_tokens + completion_tokens
                    if prompt_tokens is not None and completion_tokens is not None
                    else None
                ),
                "provider_prompt_usage_available": prompt_usage_available,
                "provider_completion_usage_available": (
                    completion_usage_available
                ),
                "provider_usage_available": (
                    prompt_usage_available and completion_usage_available
                ),
                "latency_ms": call.latency_ms,
                "model": metadata.get("provider_model"),
                "requested_model": metadata.get("requested_model"),
                "resolved_model": metadata.get("resolved_model"),
                "provider_request_id": metadata.get("provider_request_id"),
                "system_fingerprint": metadata.get("system_fingerprint"),
                "finish_reason": metadata.get("finish_reason"),
                "call_origin": "new",
                "generation_temperature_requested": metadata.get(
                    "generation_temperature_requested"
                ),
                "generation_seed_requested": metadata.get("generation_seed_requested"),
                "generation_max_tokens_requested": metadata.get(
                    "generation_max_tokens_requested"
                ),
                "error_type": metadata.get("error_type"),
            }
        )
    return records


def _provider_records_from_jobs(experiment_dir: Path) -> list[dict[str, Any]]:
    """Recover compact call records from atomic jobs after an interrupted run."""

    records: list[dict[str, Any]] = []
    for stage, role, task in [
        ("generation", "backbone", "answer_generation"),
        ("judge", "independent_judge", "locomo_judge"),
    ]:
        for path in sorted((experiment_dir / "jobs" / stage).glob("*/*.json")):
            try:
                payload = _read_json(path)
            except (OSError, ValueError):
                continue
            if (
                payload.get("status") not in {"complete", "error"}
                or not payload.get("call_item_uid")
            ):
                continue
            response_metadata = dict(payload.get("response_metadata") or {})
            prompt_tokens = (
                int(payload["prompt_tokens"])
                if payload.get("prompt_tokens") is not None
                else None
            )
            completion_tokens = (
                int(payload["completion_tokens"])
                if payload.get("completion_tokens") is not None
                else None
            )
            records.append(
                {
                    "schema_version": "experiment_provider_call_v1",
                    "role": role,
                    "task": task,
                    "provider_call_id": payload.get("provider_call_id"),
                    "provider_call_uid": payload.get("provider_call_uid"),
                    "call_item_uid": payload.get("call_item_uid"),
                    "logical_call_item_uid": payload.get("logical_call_item_uid"),
                    "sample_id": payload.get("sample_id"),
                    "query_task_id": payload.get("query_task_id"),
                    "variant": payload.get("variant"),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": (
                        prompt_tokens + completion_tokens
                        if prompt_tokens is not None
                        and completion_tokens is not None
                        else None
                    ),
                    "provider_prompt_usage_available": (
                        prompt_tokens is not None
                    ),
                    "provider_completion_usage_available": (
                        completion_tokens is not None
                    ),
                    "provider_usage_available": (
                        prompt_tokens is not None
                        and completion_tokens is not None
                    ),
                    "latency_ms": (
                        float(payload["latency_ms"])
                        if payload.get("latency_ms") is not None
                        else None
                    ),
                    "model": response_metadata.get("resolved_model")
                    or response_metadata.get("requested_model"),
                    "requested_model": response_metadata.get("requested_model"),
                    "resolved_model": response_metadata.get("resolved_model"),
                    "provider_request_id": response_metadata.get("provider_request_id"),
                    "system_fingerprint": response_metadata.get("system_fingerprint"),
                    "finish_reason": response_metadata.get("finish_reason"),
                    "call_origin": payload.get("call_origin") or (
                        "parent_experiment"
                        if payload.get("generation_source")
                        == "verified_parent_experiment"
                        or payload.get("judgment_source")
                        == "verified_parent_experiment"
                        else "new"
                    ),
                    "error_type": payload.get("error_type"),
                    "recovered_from_job": True,
                }
            )
    return records


def _provider_record_for_logical_uid(
    provider: MeteredLLMProvider,
    logical_call_item_uid: str,
) -> dict[str, Any]:
    for row in reversed(_provider_records(provider)):
        if str(row.get("logical_call_item_uid") or "") == logical_call_item_uid:
            return row
    return {}


def _provider_attempt_count(rows: Iterable[dict[str, Any]]) -> int:
    keys: set[str] = set()
    anonymous = 0
    for row in rows:
        key = str(row.get("provider_call_uid") or "")
        if not key:
            key = str(row.get("call_item_uid") or "")
        if key:
            keys.add(key)
        else:
            anonymous += 1
    return len(keys) + anonymous


def _job_path(
    experiment_dir: Path, stage: str, variant: str, query_task_id: str
) -> Path:
    return (
        experiment_dir
        / "jobs"
        / _slug(stage)
        / _slug(variant)
        / f"{_slug(query_task_id)}.json"
    )


def _completed_job(
    path: Path,
    *,
    expected_schema: str | None = None,
    expected_config_hash: str | None = None,
    expected_variant: str | None = None,
    expected_query_task_id: str | None = None,
    expected_prompt_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except (OSError, ValueError):
        return None
    if payload.get("status") != "complete":
        return None
    if expected_schema and payload.get("schema_version") != expected_schema:
        return None
    if (
        expected_config_hash
        and payload.get("experiment_config_hash") != expected_config_hash
    ):
        return None
    if expected_variant and payload.get("variant") != expected_variant:
        return None
    if (
        expected_query_task_id
        and payload.get("query_task_id") != expected_query_task_id
    ):
        return None
    if (
        expected_prompt_sha256
        and payload.get("prompt_sha256") != expected_prompt_sha256
    ):
        return None
    return payload


def _parent_config_value(config: dict[str, Any], key: str) -> Any:
    if key == "max_total_tokens":
        return config.get("max_total_tokens") or config.get("max_prompt_tokens")
    return config.get(key)


def _reuse_parent_experiment(
    *,
    parent_experiment_path: Path,
    experiment_dir: Path,
    child_config: dict[str, Any],
    child_config_hash: str,
    child_sampling: dict[str, Any],
    selected_variants: list[str],
    baseline_methods: list[str],
    source_rows: dict[str, dict[str, Any]],
    expected_generation_prompt_sha256: Any,
    policy: str,
) -> dict[str, Any]:
    """Verify and attach reusable generation and judgment jobs atomically."""

    parent_dir = parent_experiment_path.expanduser().resolve()
    if parent_dir == experiment_dir.resolve():
        raise ValueError("reuse_experiment must refer to a different experiment.")
    manifest_path = parent_dir / "experiment_manifest.json"
    sampling_path = parent_dir / "sampling_manifest.json"
    if not manifest_path.exists() or not sampling_path.exists():
        message = (
            "Parent experiment must contain experiment_manifest.json and "
            "sampling_manifest.json."
        )
        if policy == "require":
            raise ValueError(message)
        return {
            "status": "unavailable",
            "errors": [message],
            "generation_job_count": 0,
            "judgment_job_count": 0,
            "provider_call_row_count": 0,
        }

    parent_manifest = _read_json(manifest_path)
    parent_sampling = _read_json(sampling_path)
    parent_config = dict(parent_manifest.get("config") or {})
    errors: list[str] = []
    if str(parent_manifest.get("status") or "") != "complete":
        errors.append(
            f"parent experiment status is {parent_manifest.get('status')!r}, not complete"
        )
    compatibility_keys = [
        "source_run_id",
        "source_database_sha256",
        "source_details_sha256",
        "max_total_tokens",
        "max_output_tokens",
        "token_counter_requested",
        "require_exact_token_counter",
        "token_safety_margin",
        "rag_chunk_size",
        "rag_chunk_overlap",
        "rag_top_k",
        "backbone_provider_kind",
        "backbone_model",
        "independent_judge_provider_kind",
        "independent_judge_model",
        "generation_temperature",
        "generation_seed",
        "full_answer_policy",
        "answer_prompt_version",
        "judge_prompt_version",
    ]
    config_mismatches = [
        key
        for key in compatibility_keys
        if _parent_config_value(parent_config, key)
        != _parent_config_value(child_config, key)
    ]
    if config_mismatches:
        errors.append(
            "parent experiment protocol mismatch for fields "
            f"{config_mismatches}"
        )

    child_queries = {
        str(row.get("query_task_id") or ""): str(row.get("stratum") or "")
        for row in list(child_sampling.get("selected_queries") or [])
    }
    parent_queries = {
        str(row.get("query_task_id") or ""): str(row.get("stratum") or "")
        for row in list(parent_sampling.get("selected_queries") or [])
    }
    if child_sampling.get("sampling_profile") == "rebuttal_200_v1":
        expected_parent_counts = {
            "strict_deep_history": 9,
            "update_sensitive": 26,
            "ordinary": 25,
        }
        if len(parent_queries) != 60:
            errors.append(
                "rebuttal_200_v1 requires a 60-query parent experiment"
            )
        if dict(parent_sampling.get("selected_counts_by_stratum") or {}) != (
            expected_parent_counts
        ):
            errors.append(
                "rebuttal_200_v1 parent stratum counts must be 9/26/25"
            )
    missing_queries = sorted(set(parent_queries) - set(child_queries))
    stratum_mismatches = sorted(
        query_task_id
        for query_task_id in set(parent_queries) & set(child_queries)
        if parent_queries[query_task_id] != child_queries[query_task_id]
    )
    if missing_queries:
        errors.append(
            f"{len(missing_queries)} parent queries are absent from the child sample"
        )
    if stratum_mismatches:
        errors.append(
            "parent/child stratum mismatch for queries "
            f"{stratum_mismatches[:5]}"
        )

    parent_variants = list(parent_config.get("variants") or [])
    reusable_generation_variants = [
        variant for variant in selected_variants if variant in parent_variants
    ]
    parent_baselines = set(
        parent_manifest.get("baseline_methods")
        or parent_config.get("baseline_methods")
        or []
    )
    reusable_judgment_variants = list(
        dict.fromkeys(
            [
                *reusable_generation_variants,
                *OBSERVED_DIAGNOSTIC_VARIANTS,
                *[
                    method
                    for method in baseline_methods
                    if method in parent_baselines
                ],
            ]
        )
    )
    parent_config_hash = str(parent_manifest.get("config_hash") or "")
    generation_copies: list[tuple[Path, dict[str, Any]]] = []
    generation_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    judgment_copies: list[tuple[Path, dict[str, Any]]] = []

    if not errors:
        for query_task_id in sorted(parent_queries):
            for variant in reusable_generation_variants:
                expected_prompt_hash = expected_generation_prompt_sha256(
                    variant,
                    query_task_id,
                )
                parent_path = _job_path(
                    parent_dir,
                    "generation",
                    variant,
                    query_task_id,
                )
                parent_job = _completed_job(
                    parent_path,
                    expected_schema="answer_generation_job_v1",
                    expected_config_hash=parent_config_hash,
                    expected_variant=variant,
                    expected_query_task_id=query_task_id,
                    expected_prompt_sha256=expected_prompt_hash,
                )
                if parent_job is None:
                    errors.append(
                        "parent generation job failed verification: "
                        f"{variant}/{query_task_id}"
                    )
                    continue
                copied = {
                    **parent_job,
                    "experiment_config_hash": child_config_hash,
                    "generation_source": "verified_parent_experiment",
                    "parent_generation_source": parent_job.get(
                        "generation_source"
                    ),
                    "call_origin": "parent_experiment",
                    "parent_experiment_config_hash": parent_config_hash,
                    "parent_job_sha256": _sha256_file(parent_path),
                    "parent_provider_call_uid": parent_job.get(
                        "provider_call_uid"
                    ),
                    "parent_call_item_uid": parent_job.get("call_item_uid"),
                }
                generation_payloads[(variant, query_task_id)] = copied
                generation_copies.append(
                    (
                        _job_path(
                            experiment_dir,
                            "generation",
                            variant,
                            query_task_id,
                        ),
                        copied,
                    )
                )

        for query_task_id in sorted(parent_queries):
            sample_row = source_rows.get(query_task_id)
            if sample_row is None:
                continue
            for variant in reusable_judgment_variants:
                generation = generation_payloads.get((variant, query_task_id))
                if generation is None:
                    generation = _completed_job(
                        _job_path(
                            experiment_dir,
                            "generation",
                            variant,
                            query_task_id,
                        ),
                        expected_schema="answer_generation_job_v1",
                        expected_config_hash=child_config_hash,
                        expected_variant=variant,
                        expected_query_task_id=query_task_id,
                    )
                if generation is None:
                    errors.append(
                        "child answer needed for parent judgment verification is "
                        f"missing: {variant}/{query_task_id}"
                    )
                    continue
                expected_prompt_hash = _sha256_bytes(
                    _judge_prompt(
                        str(sample_row.get("question") or ""),
                        str(sample_row.get("gold_answer") or ""),
                        str(generation.get("answer_text") or ""),
                    ).encode("utf-8")
                )
                parent_path = _job_path(
                    parent_dir,
                    "judge",
                    variant,
                    query_task_id,
                )
                parent_job = _completed_job(
                    parent_path,
                    expected_schema="independent_judgment_v1",
                    expected_config_hash=parent_config_hash,
                    expected_variant=variant,
                    expected_query_task_id=query_task_id,
                    expected_prompt_sha256=expected_prompt_hash,
                )
                if parent_job is None:
                    errors.append(
                        "parent judgment job failed verification: "
                        f"{variant}/{query_task_id}"
                    )
                    continue
                copied = {
                    **parent_job,
                    "experiment_config_hash": child_config_hash,
                    "judgment_source": "verified_parent_experiment",
                    "call_origin": "parent_experiment",
                    "parent_experiment_config_hash": parent_config_hash,
                    "parent_job_sha256": _sha256_file(parent_path),
                    "parent_provider_call_uid": parent_job.get(
                        "provider_call_uid"
                    ),
                    "parent_call_item_uid": parent_job.get("call_item_uid"),
                }
                judgment_copies.append(
                    (
                        _job_path(
                            experiment_dir,
                            "judge",
                            variant,
                            query_task_id,
                        ),
                        copied,
                    )
                )

    provider_rows: list[dict[str, Any]] = []
    expected_provider_jobs = [
        (
            str(payload.get("call_item_uid") or ""),
            "backbone",
            str(payload.get("variant") or ""),
            str(payload.get("query_task_id") or ""),
        )
        for _, payload in generation_copies
    ] + [
        (
            str(payload.get("call_item_uid") or ""),
            "independent_judge",
            str(payload.get("variant") or ""),
            str(payload.get("query_task_id") or ""),
        )
        for _, payload in judgment_copies
    ]
    expected_call_item_counts = Counter(
        call_item_uid
        for call_item_uid, _, _, _ in expected_provider_jobs
        if call_item_uid
    )
    expected_call_item_uids = set(expected_call_item_counts)
    if any(not call_item_uid for call_item_uid, _, _, _ in expected_provider_jobs):
        errors.append("one or more verified parent jobs lack call_item_uid")
    duplicated_job_call_items = sorted(
        call_item_uid
        for call_item_uid, count in expected_call_item_counts.items()
        if count > 1
    )
    if duplicated_job_call_items:
        errors.append(
            f"{len(duplicated_job_call_items)} verified parent jobs share a "
            "call_item_uid"
        )
    expected_provider_job_by_uid = {
        call_item_uid: (role, variant, query_task_id)
        for call_item_uid, role, variant, query_task_id in expected_provider_jobs
        if call_item_uid
    }
    reusable_query_ids = set(parent_queries)
    for row in (
        _read_jsonl(parent_dir / "provider_call_rows.jsonl")
        if not errors
        else []
    ):
        if row.get("superseded_external_attachment") is True:
            continue
        query_task_id = str(row.get("query_task_id") or "")
        variant = str(row.get("variant") or "")
        role = str(row.get("role") or "")
        call_item_uid = str(row.get("call_item_uid") or "")
        if query_task_id not in reusable_query_ids:
            continue
        if call_item_uid not in expected_call_item_uids:
            continue
        if expected_provider_job_by_uid[call_item_uid] != (
            role,
            variant,
            query_task_id,
        ):
            continue
        if role == "backbone" and variant not in reusable_generation_variants:
            continue
        if (
            role == "independent_judge"
            and variant not in reusable_judgment_variants
        ):
            continue
        if role not in {"backbone", "independent_judge"}:
            continue
        provider_rows.append(
            {
                **row,
                "call_origin": "parent_experiment",
                "parent_experiment_config_hash": parent_config_hash,
            }
        )

    provider_call_item_counts = Counter(
        str(row.get("call_item_uid") or "") for row in provider_rows
    )
    missing_provider_call_items = sorted(
        expected_call_item_uids - set(provider_call_item_counts)
    )
    duplicate_provider_call_items = sorted(
        call_item_uid
        for call_item_uid, count in provider_call_item_counts.items()
        if call_item_uid and count > 1
    )
    if missing_provider_call_items:
        errors.append(
            f"{len(missing_provider_call_items)} verified parent jobs lack a "
            "matching provider ledger row"
        )
    if duplicate_provider_call_items:
        errors.append(
            f"{len(duplicate_provider_call_items)} parent provider ledger call "
            "items are duplicated"
        )

    if errors and policy == "require":
        raise ValueError(
            "Parent experiment reuse verification failed before provider calls: "
            + "; ".join(errors[:10])
        )

    if errors and policy == "best-effort":
        generation_copies = []
        judgment_copies = []
        provider_rows = []

    for path, payload in generation_copies:
        _atomic_json(path, payload)
    for path, payload in judgment_copies:
        _atomic_json(path, payload)

    report = {
        "schema_version": "parent_experiment_reuse_v1",
        "status": "verified" if not errors else "not_reused",
        "policy": policy,
        "parent_experiment_path": str(parent_dir),
        "parent_manifest_sha256": _sha256_file(manifest_path),
        "parent_config_hash": parent_config_hash,
        "parent_query_count": len(parent_queries),
        "parent_query_overlap": len(set(parent_queries) & set(child_queries)),
        "reusable_generation_variants": reusable_generation_variants,
        "reusable_judgment_variants": reusable_judgment_variants,
        "generation_job_count": len(generation_copies),
        "judgment_job_count": len(judgment_copies),
        "provider_call_row_count": len(provider_rows),
        "errors": errors,
    }
    _atomic_json(experiment_dir / "parent_experiment_reuse.json", report)
    if provider_rows:
        existing_rows = _read_jsonl(experiment_dir / "provider_call_rows.jsonl")
        merged: dict[str, dict[str, Any]] = {}
        for row in [*existing_rows, *provider_rows]:
            key = str(row.get("call_item_uid") or "")
            if not key:
                key = (
                    f"{row.get('role')}:{row.get('variant')}:"
                    f"{row.get('query_task_id')}:{row.get('provider_call_uid')}"
                )
            merged[key] = row
        _write_jsonl(
            experiment_dir / "provider_call_rows.jsonl",
            list(merged.values()),
        )
    return report


def _token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _mean_present(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    return mean(values) if values else None


def _judge_prompt(question: str, gold_answer: str, candidate_answer: str) -> str:
    return (
        "You are an independent evaluator. Judge only whether the candidate answer correctly "
        "answers the question relative to the reference answer. Do not infer which system "
        "produced it. Use CORRECT when all required facts are present without material errors, "
        "PARTIAL when some required facts are missing but there is meaningful correct content, "
        "and INCORRECT otherwise.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{gold_answer}\n\n"
        f"CANDIDATE ANSWER:\n{candidate_answer}\n\n"
        "Return exactly two lines:\nVERDICT: CORRECT|PARTIAL|INCORRECT\nRATIONALE: <brief reason>"
    )


def _parse_judge_verdict(text: str) -> str:
    value = str(text or "")
    match = re.search(
        r"(?im)^\s*VERDICT\s*:\s*(incorrect|partial|correct)\b",
        value,
    )
    if match:
        return match.group(1).lower()
    exact = re.fullmatch(
        r"\s*(?:`{1,3})?(incorrect|partial|correct)(?:`{1,3})?[.!]?\s*",
        value,
        flags=re.IGNORECASE,
    )
    return exact.group(1).lower() if exact else "judge_error"


def _answer_stage_usage(
    answer_metadata: dict[str, Any],
    *,
    stage: str,
) -> tuple[int | None, int | None]:
    prompt_value = answer_metadata.get(f"answer_stage_{stage}_prompt_tokens")
    completion_value = answer_metadata.get(f"answer_stage_{stage}_completion_tokens")
    if prompt_value is not None or completion_value is not None:
        return (
            int(prompt_value) if prompt_value is not None else None,
            int(completion_value) if completion_value is not None else None,
        )
    if stage == "initial":
        prompt_value = answer_metadata.get("answer_initial_prompt_tokens")
        completion_value = answer_metadata.get("answer_initial_completion_tokens")
    elif stage == "pre_reflection":
        prompt_value = answer_metadata.get("initial_answer_prompt_tokens")
        completion_value = answer_metadata.get("initial_answer_completion_tokens")
    return (
        int(prompt_value) if prompt_value is not None else None,
        int(completion_value) if completion_value is not None else None,
    )


def _provider_prompt_within_total_budget(
    *,
    actual_prompt_tokens: int | None,
    max_output_tokens: int,
    max_total_tokens: int,
) -> bool | None:
    """Reconcile provider prompt usage after the pre-call envelope margin."""

    if actual_prompt_tokens is None:
        return None
    return int(actual_prompt_tokens) + int(max_output_tokens) <= int(
        max_total_tokens
    )


def _derived_answers(sample_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    answer_metadata = dict(
        dict(sample_row.get("metadata") or {}).get("answer_metadata") or {}
    )
    final_answer = str(sample_row.get("answer_text") or "")
    initial_prompt_tokens, initial_completion_tokens = _answer_stage_usage(
        answer_metadata,
        stage="initial",
    )
    pre_prompt_tokens, pre_completion_tokens = _answer_stage_usage(
        answer_metadata,
        stage="pre_reflection",
    )
    fallback_refs = support_refs_from_answer_metadata(answer_metadata)
    fallback_invalid_refs = invalid_refs_from_answer_metadata(answer_metadata)
    initial_observable = "answer_stage_initial_text" in answer_metadata
    pre_reflection_observable = "answer_stage_pre_reflection_text" in answer_metadata
    return {
        "no_answer_validation_or_repair": {
            "answer_text": str(
                answer_metadata.get("answer_stage_initial_text")
                if initial_observable
                else answer_metadata.get("answer_initial_text") or final_answer
            ),
            "support_refs": list(
                answer_metadata.get("answer_stage_initial_supporting_refs") or []
                if initial_observable
                else fallback_refs
            ),
            "invalid_support_refs": list(
                answer_metadata.get("answer_stage_initial_invalid_supporting_refs")
                or []
                if initial_observable
                else fallback_invalid_refs
            ),
            "prompt_tokens": initial_prompt_tokens,
            "completion_tokens": initial_completion_tokens,
            "stage_observable": initial_observable,
            "stage_source_field": (
                "answer_stage_initial_text"
                if initial_observable
                else "legacy_fallback_not_formal"
            ),
        },
        "no_retrieval_retry": {
            "answer_text": str(
                answer_metadata.get("answer_stage_pre_reflection_text")
                if pre_reflection_observable
                else answer_metadata.get("initial_answer_text") or final_answer
            ),
            "support_refs": list(
                answer_metadata.get("answer_stage_pre_reflection_supporting_refs") or []
                if pre_reflection_observable
                else fallback_refs
            ),
            "invalid_support_refs": list(
                answer_metadata.get(
                    "answer_stage_pre_reflection_invalid_supporting_refs"
                )
                or []
                if pre_reflection_observable
                else fallback_invalid_refs
            ),
            "prompt_tokens": pre_prompt_tokens,
            "completion_tokens": pre_completion_tokens,
            "stage_observable": pre_reflection_observable,
            "stage_source_field": (
                "answer_stage_pre_reflection_text"
                if pre_reflection_observable
                else "legacy_fallback_not_formal"
            ),
        },
    }


def _scoreable_answer_text(answer_text: str) -> str:
    """Remove provenance-only citation lines from answer-quality metrics."""

    return re.sub(
        r"(?im)^\s*(?:sources?|supporting\s+sources?)\s*:\s*.*$",
        "",
        str(answer_text or ""),
    ).strip()


def _answer_is_abstention(answer_text: str) -> bool:
    text = " ".join(str(answer_text or "").casefold().split())
    if not text:
        return True
    return any(
        marker in text
        for marker in [
            "not enough",
            "not supported",
            "cannot answer",
            "can't answer",
            "do not have enough",
            "don't have enough",
            "insufficient",
            "unknown",
            "not available",
            "no retrieved evidence",
        ]
    )


def _require_remote_credentials(provider_kind: str, model: str, *, role: str) -> None:
    if provider_kind != "remote":
        return
    required_keys = {
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(infer_remote_vendor(model), ())
    if required_keys and not any(os.getenv(key) for key in required_keys):
        required_key_label = " or ".join(required_keys)
        legacy_note = (
            " The legacy Claud_API_Key spelling is intentionally not accepted."
            if required_keys == ("ANTHROPIC_API_KEY",)
            else ""
        )
        raise ProviderConfigurationError(
            f"{role} model {model!r} requires {required_key_label}.{legacy_note}"
        )


def _effective_provider_concurrency(provider_kind: str, requested: int) -> int:
    """Serialize bare Transformers providers, which share model and RNG state."""

    return 1 if provider_kind == "local" else max(1, int(requested))


def _experiment_config_hash(payload: dict[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode(
            "utf-8"
        )
    )


def run_answer_ablation(
    run_path: Path | str,
    *,
    variants: str | Iterable[str] | None = None,
    baseline_answers_path: Path | str | None = None,
    sample_size: int = 60,
    sampling_profile: str = "auto",
    sample_seed: int = 7,
    max_total_tokens: int = 32_000,
    max_prompt_tokens: int | None = None,
    max_output_tokens: int = 512,
    token_counter: str = "auto",
    require_exact_token_counter: bool = False,
    token_safety_margin: int = 128,
    rag_chunk_size: int = 384,
    rag_chunk_overlap: int = 64,
    rag_top_k: int = 4,
    backbone_provider_kind: str = "remote",
    backbone_model: str = "gpt-4o-mini",
    independent_judge_provider_kind: str = "remote",
    independent_judge_model: str = "claude-sonnet-4-6",
    generation_temperature: float = 0.0,
    generation_seed: int = 7,
    generation_max_concurrency: int = 6,
    judge_max_concurrency: int = 6,
    context_save_mode: str = "full",
    max_provider_calls: int = 1500,
    reuse_experiment_path: Path | str | None = None,
    reuse_policy: str = "off",
    progress: bool = False,
    progress_interval_seconds: int = 30,
    report_path: Path | str | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    load_runtime_env(override=True)
    run_dir = Path(run_path).expanduser().resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    run_meta, sample_rows, database_path = load_details_rows(run_dir)
    if str(run_meta.get("dataset") or "").lower() != "locomo":
        raise ValueError(
            "Answer-level rebuttal ablations currently support LOCOMO runs only."
        )
    source_query_ids = [
        str(row.get("query_task_id") or "").strip()
        for row in sample_rows
    ]
    missing_source_query_ids = sum(
        1 for query_task_id in source_query_ids if not query_task_id
    )
    duplicate_source_query_ids = [
        query_task_id
        for query_task_id, count in Counter(source_query_ids).items()
        if query_task_id and count > 1
    ]
    if missing_source_query_ids or duplicate_source_query_ids:
        raise ValueError(
            "Source run query IDs are not unique and complete: "
            f"missing={missing_source_query_ids}, "
            f"duplicates={duplicate_source_query_ids[:5]}."
        )
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size > len(sample_rows):
        raise ValueError(
            f"sample_size={sample_size} exceeds the {len(sample_rows)} "
            "available source-run queries."
        )
    normalized_reuse_policy = str(reuse_policy or "off").strip().lower()
    if normalized_reuse_policy not in {"require", "best-effort", "off"}:
        raise ValueError("reuse_policy must be require, best-effort, or off.")
    if normalized_reuse_policy == "require" and reuse_experiment_path is None:
        raise ValueError(
            "reuse_policy=require requires --reuse-experiment."
        )
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive.")
    resolved_max_total_tokens = (
        int(max_prompt_tokens)
        if max_prompt_tokens is not None
        else int(max_total_tokens)
    )
    if resolved_max_total_tokens <= 0 or max_output_tokens <= 0:
        raise ValueError("max_total_tokens and max_output_tokens must be positive.")
    if max_output_tokens + token_safety_margin >= resolved_max_total_tokens:
        raise ValueError(
            "max_output_tokens plus token_safety_margin must be smaller than "
            "max_total_tokens."
        )
    if rag_chunk_size <= 0 or rag_top_k <= 0:
        raise ValueError("rag_chunk_size and rag_top_k must be positive.")
    if rag_chunk_overlap < 0 or rag_chunk_overlap >= rag_chunk_size:
        raise ValueError(
            "rag_chunk_overlap must be non-negative and smaller than rag_chunk_size."
        )
    if generation_max_concurrency <= 0 or judge_max_concurrency <= 0:
        raise ValueError("generation and judge concurrency must be positive.")
    if max_provider_calls <= 0:
        raise ValueError("max_provider_calls must be positive.")
    if context_save_mode not in {"full", "compact"}:
        raise ValueError("context_save_mode must be 'full' or 'compact'.")
    supported_provider_kinds = {
        "mock",
        "remote",
        "openai-compatible",
        "local",
    }
    for role, provider_kind in [
        ("backbone", backbone_provider_kind),
        ("independent judge", independent_judge_provider_kind),
    ]:
        if provider_kind not in supported_provider_kinds:
            raise ValueError(
                f"Unsupported {role} provider kind: {provider_kind!r}."
            )
    selected_variants = _parse_list(variants, DEFAULT_VARIANTS)
    unknown = sorted(set(selected_variants) - set(VARIANT_POLICIES))
    if unknown:
        raise ValueError(f"Unsupported answer ablation variants: {unknown}")
    matched_variants = {
        "full_context_matched",
        "hybrid_raw_rag_matched",
    }
    if matched_variants & set(selected_variants) and "full" not in selected_variants:
        raise ValueError(
            "Prompt-token-matched baselines require the full variant as their "
            "per-query reference."
        )
    imported_baseline_rows: list[dict[str, Any]] = []
    baseline_path: Path | None = None
    if baseline_answers_path is not None:
        baseline_path = Path(baseline_answers_path).expanduser().resolve()
        if not baseline_path.exists():
            raise FileNotFoundError(baseline_path)
        imported_baseline_rows = _read_jsonl(baseline_path)
        if not imported_baseline_rows:
            raise ValueError("The imported baseline answer file is empty.")
        invalid_rows = [
            index
            for index, row in enumerate(imported_baseline_rows, start=1)
            if not str(row.get("method") or "").strip()
            or not str(row.get("query_task_id") or "").strip()
        ]
        if invalid_rows:
            raise ValueError(
                "Imported baseline rows must include method and query_task_id; "
                f"invalid row indexes={invalid_rows[:5]}."
            )
    baseline_methods = sorted(
        {
            str(row.get("method") or "").strip()
            for row in imported_baseline_rows
            if str(row.get("method") or "").strip()
        }
    )
    reserved_methods = (
        set(selected_variants)
        | set(DERIVED_VARIANTS)
        | set(OBSERVED_DIAGNOSTIC_VARIANTS)
    )
    collisions = sorted(reserved_methods & set(baseline_methods))
    if collisions:
        raise ValueError(
            f"Imported baseline method names collide with ablation variants: {collisions}"
        )
    effective_generation_concurrency = _effective_provider_concurrency(
        backbone_provider_kind,
        generation_max_concurrency,
    )
    effective_judge_concurrency = _effective_provider_concurrency(
        independent_judge_provider_kind,
        judge_max_concurrency,
    )
    execution_state = _git_state(run_dir)
    package_versions = _package_versions()

    config_payload = {
        "source_run_id": run_meta.get("run_id"),
        "source_database_sha256": _sha256_file(database_path),
        "source_details_sha256": _sha256_file(run_dir / "details.json"),
        "variants": selected_variants,
        "sample_size": sample_size,
        "sampling_profile": sampling_profile,
        "sample_seed": sample_seed,
        "max_total_tokens": resolved_max_total_tokens,
        "max_prompt_tokens": resolved_max_total_tokens,
        "max_output_tokens": max_output_tokens,
        "token_counter_requested": token_counter,
        "require_exact_token_counter": require_exact_token_counter,
        "token_safety_margin": token_safety_margin,
        "rag_chunk_size": rag_chunk_size,
        "rag_chunk_overlap": rag_chunk_overlap,
        "rag_top_k": rag_top_k,
        "backbone_provider_kind": backbone_provider_kind,
        "backbone_model": backbone_model,
        "independent_judge_provider_kind": independent_judge_provider_kind,
        "independent_judge_model": independent_judge_model,
        "generation_temperature": generation_temperature,
        "generation_seed": generation_seed,
        "full_answer_policy": "rerun_with_shared_neutral_prompt_v1",
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "judge_prompt_version": INDEPENDENT_JUDGE_PROMPT_VERSION,
        "source_tree_sha256": execution_state["source_tree_sha256"],
        "package_versions": package_versions,
    }
    config_hash = _experiment_config_hash(config_payload)
    experiment_dir = (
        run_dir / "rebuttal_experiments" / f"answer_ablation_{config_hash[:16]}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    progress_reporter = ExperimentProgress(
        experiment_dir / "progress.json",
        enabled=progress,
        interval_seconds=progress_interval_seconds,
    )

    resolved_report_path = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else None
    )

    def finalize_report(payload: dict[str, Any]) -> dict[str, Any]:
        if resolved_report_path is not None:
            _atomic_json(resolved_report_path, payload)
        progress_reporter.close()
        return payload

    def completed_experiment_job(
        stage: str,
        variant: str,
        query_task_id: str,
        *,
        expected_prompt_sha256: str | None = None,
    ) -> dict[str, Any] | None:
        schema = (
            "answer_generation_job_v1"
            if stage == "generation"
            else "independent_judgment_v1"
        )
        return _completed_job(
            _job_path(experiment_dir, stage, variant, query_task_id),
            expected_schema=schema,
            expected_config_hash=config_hash,
            expected_variant=variant,
            expected_query_task_id=query_task_id,
            expected_prompt_sha256=expected_prompt_sha256,
        )

    manifest_path = experiment_dir / "experiment_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        existing_manifest = _read_json(manifest_path)
        if existing_manifest.get("config_hash") != config_hash:
            raise ValueError(
                "Existing experiment directory has a mismatched config hash."
            )
        if not resume:
            raise ValueError("Experiment already exists; pass --resume to continue it.")
    provided_baseline_methods = list(baseline_methods)
    existing_baseline_methods = list(
        existing_manifest.get("baseline_methods")
        or dict(existing_manifest.get("config") or {}).get("baseline_methods")
        or []
    )
    baseline_methods = sorted(
        set(existing_baseline_methods) | set(provided_baseline_methods)
    )
    baseline_attachment: dict[str, Any] | None = None
    replaced_external_methods: set[str] = set()
    retained_baseline_attachments = list(
        existing_manifest.get("external_baseline_attachments") or []
    )
    if baseline_path is not None:
        baseline_attachment = {
            "path": str(baseline_path),
            "sha256": _sha256_file(baseline_path),
            "methods": sorted(provided_baseline_methods),
        }
        canonical_attachments: list[dict[str, Any]] = []
        for attachment in retained_baseline_attachments:
            overlapping_methods = set(attachment.get("methods") or []) & set(
                provided_baseline_methods
            )
            if overlapping_methods and str(attachment.get("sha256") or "") != str(
                baseline_attachment["sha256"]
            ):
                replaced_external_methods.update(overlapping_methods)
            remaining_methods = sorted(
                set(attachment.get("methods") or [])
                - set(provided_baseline_methods)
            )
            if remaining_methods:
                canonical_attachments.append(
                    {**attachment, "methods": remaining_methods}
                )
        canonical_attachments.append(baseline_attachment)
        retained_baseline_attachments = canonical_attachments

    provider_config = RunConfig(
        dataset="locomo",
        dataset_subset=str(run_meta.get("dataset_scope_key") or "multi_hop"),
        # Providers do not read the original dataset here; all experiment inputs
        # come from the immutable run details and SQLite snapshot.
        dataset_path=run_dir,
        output_dir=experiment_dir,
        index_database_path=experiment_dir / "unused_index.sqlite",
        provider_kind=backbone_provider_kind,
        backbone_provider_kind=backbone_provider_kind,
        judge_provider_kind=independent_judge_provider_kind,
        backbone_model=backbone_model,
        judge_model=independent_judge_model,
        embedding_model=str(run_meta.get("embedding_model") or "hash-embedding"),
        device_mode=str(run_meta.get("device_mode") or "auto"),
        conv_workers=1,
        cuda_preflight_mode="off",
    )
    invocation_id = uuid.uuid4().hex[:12]
    try:
        answer_token_counter = build_token_counter(
            token_counter,
            model_name=backbone_model,
            require_exact=require_exact_token_counter,
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    embedding_provider = (
        build_embedding_provider(provider_config)
        if {
            "hybrid_raw_rag",
            "hybrid_raw_rag_matched",
            "naive_dense_rag",
        }
        & set(selected_variants)
        else None
    )
    context_builder = VariantContextBuilder(
        database_path=database_path,
        cache_dir=experiment_dir / "cache",
        embedding_provider=embedding_provider,
        top_k=int(run_meta.get("k") or 15),
        max_prompt_tokens=resolved_max_total_tokens,
        output_token_reserve=max_output_tokens,
        token_counter=answer_token_counter,
        token_safety_margin=token_safety_margin,
        rag_chunk_size=rag_chunk_size,
        rag_chunk_overlap=rag_chunk_overlap,
        rag_top_k=rag_top_k,
    )
    gold_rows = load_or_build_gold_labels(
        run_dir,
        sample_rows,
        context_builder.memory_index,
    )
    selected_rows, sampling_manifest = _sample_queries(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        memory_index=context_builder.memory_index,
        sample_size=sample_size,
        seed=sample_seed,
        sampling_profile=sampling_profile,
    )
    sampling_manifest = {
        **sampling_manifest,
        "source_run_dir": str(run_dir),
        "source_run_id": run_meta.get("run_id"),
        "source_details_sha256": config_payload["source_details_sha256"],
    }
    selected_query_ids = {
        str(row.get("query_task_id") or "")
        for row in selected_rows
    }
    selected_gold_rows = [
        {**row, "contains_sensitive_text": True}
        for row in gold_rows
        if str(row.get("query_task_id") or "") in selected_query_ids
    ]
    if len(selected_gold_rows) != len(selected_query_ids):
        raise ValueError(
            "Normalized gold labels do not cover every selected answer-ablation query."
        )
    _atomic_json(experiment_dir / "sampling_manifest.json", sampling_manifest)
    _write_jsonl(experiment_dir / "gold_label_rows.jsonl", selected_gold_rows)
    source_rows = {str(row.get("query_task_id")): row for row in selected_rows}
    imported_by_method_query = {
        (str(row.get("method") or ""), str(row.get("query_task_id") or "")): row
        for row in imported_baseline_rows
    }
    if len(imported_by_method_query) != len(imported_baseline_rows):
        raise ValueError(
            "Imported baseline answers contain duplicate (method, query_task_id) rows."
        )
    for method in provided_baseline_methods:
        missing_queries = sorted(
            query_task_id
            for query_task_id in source_rows
            if (method, query_task_id) not in imported_by_method_query
        )
        if missing_queries:
            raise ValueError(
                f"Imported baseline {method!r} is missing {len(missing_queries)} selected "
                f"queries; examples={missing_queries[:5]}"
            )
        for query_task_id, sample_row in source_rows.items():
            imported = imported_by_method_query[(method, query_task_id)]
            imported_sample_id = str(imported.get("sample_id") or "")
            expected_sample_id = str(sample_row.get("sample_id") or "")
            if not imported_sample_id:
                raise ValueError(
                    f"Imported baseline {method!r} is missing sample_id for "
                    f"{query_task_id}."
                )
            if imported_sample_id != expected_sample_id:
                raise ValueError(
                    f"Imported baseline {method!r} maps {query_task_id} to "
                    f"{imported_sample_id!r}, expected {expected_sample_id!r}."
                )
            expected_question_hash = _sha256_bytes(
                str(sample_row.get("question") or "").encode("utf-8")
            )
            expected_gold_hash = _sha256_bytes(
                str(sample_row.get("gold_answer") or "").encode("utf-8")
            )
            if not imported.get("question_hash"):
                raise ValueError(
                    f"Imported baseline {method!r} is missing a verified question hash "
                    f"for {query_task_id}."
                )
            if str(imported.get("question_hash")) != expected_question_hash:
                raise ValueError(
                    f"Imported baseline {method!r} has a question hash mismatch for "
                    f"{query_task_id}."
                )
            if not imported.get("gold_answer_hash"):
                raise ValueError(
                    f"Imported baseline {method!r} is missing a verified gold-answer "
                    f"hash for {query_task_id}."
                )
            if str(imported.get("gold_answer_hash")) != expected_gold_hash:
                raise ValueError(
                    f"Imported baseline {method!r} has a gold-answer hash mismatch for "
                    f"{query_task_id}."
                )

    manifest = {
        "schema_version": "answer_ablation_experiment_v1",
        "config_hash": config_hash,
        "config": config_payload,
        "source_run_dir": str(run_dir),
        "source_database_path": str(database_path),
        "source_database_sha256": config_payload["source_database_sha256"],
        "source_details_sha256": config_payload["source_details_sha256"],
        "git": execution_state,
        "package_versions": package_versions,
        "contains_sensitive_text": True,
        "context_save_mode": context_save_mode,
        "baseline_methods": baseline_methods,
        "external_baseline_attachments": retained_baseline_attachments,
        "resume_enabled": resume,
        "reuse_policy": normalized_reuse_policy,
        "reuse_experiment_path": (
            str(Path(reuse_experiment_path).expanduser().resolve())
            if reuse_experiment_path is not None
            else None
        ),
        "max_provider_calls": max_provider_calls,
        "generation_max_concurrency_requested": max(1, int(generation_max_concurrency)),
        "generation_max_concurrency_effective": effective_generation_concurrency,
        "judge_max_concurrency_requested": max(1, int(judge_max_concurrency)),
        "judge_max_concurrency_effective": effective_judge_concurrency,
        "invocations": [
            *list(existing_manifest.get("invocations") or []),
            {
                "invocation_id": invocation_id,
                "status": "running",
                "generation_call_count": 0,
                "judge_call_count": 0,
            },
        ],
        "status": "running",
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        experiment_dir / "external_baseline_manifest.json",
        {
            "schema_version": "external_baseline_attachments_v1",
            "core_config_hash": config_hash,
            "attachments": manifest["external_baseline_attachments"],
            "methods": baseline_methods,
        },
    )

    retrieval_plan_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    token_budget_rows: list[dict[str, Any]] = []
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    progress_reporter.start_stage(
        "context_preparation",
        total=len(selected_rows) * len(selected_variants),
    )
    for sample_row in selected_rows:
        features = context_builder.compute_query_features(sample_row)
        plan = context_builder.build_retrieval_plan(sample_row)
        retrieval_plan_rows.append(
            {
                "schema_version": "retrieval_plan_v1",
                "contains_sensitive_text": True,
                **asdict(features),
                **asdict(plan),
            }
        )
        ordered_variants = [
            *(
                ["full"]
                if "full" in selected_variants
                else []
            ),
            *[variant for variant in selected_variants if variant != "full"],
        ]
        full_reference_prompt_tokens: int | None = None
        for variant in ordered_variants:
            try:
                context = context_builder.render_variant(
                    sample_row=sample_row,
                    features=features,
                    plan=plan,
                    variant=variant,
                    reference_prompt_tokens=(
                        full_reference_prompt_tokens
                        if variant in matched_variants
                        else None
                    ),
                )
            except Exception as exc:
                progress_reporter.fail(exc)
                raise
            if variant == "full":
                full_reference_prompt_tokens = context.estimated_prompt_tokens
            if (
                context.reference_prompt_tokens is not None
                and context.estimated_prompt_tokens
                > context.reference_prompt_tokens
            ):
                error = (
                    f"{variant}/{features.query_task_id} exceeds its locally "
                    "counted Full TrajWiki prompt budget: "
                    f"{context.estimated_prompt_tokens}>"
                    f"{context.reference_prompt_tokens}."
                )
                progress_reporter.fail(error)
                raise ValueError(error)
            context_path = (
                experiment_dir
                / "contexts"
                / _slug(variant)
                / f"{_slug(features.query_task_id)}.json.gz"
            )
            payload = {
                **context.compact_dict(include_context=True),
                "contains_sensitive_text": True,
            }
            if context_save_mode == "full":
                _atomic_json_gz(context_path, payload)
            contexts[(variant, features.query_task_id)] = payload
            context_rows.append(
                {
                    "schema_version": "variant_context_v1",
                    "contains_sensitive_text": True,
                    **context.compact_dict(include_context=False),
                    "context_path": str(context_path)
                    if context_save_mode == "full"
                    else None,
                    "stratum": sample_row.get("rebuttal_stratum"),
                }
            )
            token_budget_rows.append(
                {
                    "variant": variant,
                    "sample_id": context.sample_id,
                    "query_task_id": context.query_task_id,
                    "token_counter": context.token_counter,
                    "token_counter_exact": context.token_counter_exact,
                    "max_total_tokens": context.max_prompt_tokens,
                    "global_max_total_tokens": context.global_max_total_tokens,
                    "output_token_reserve": context.output_token_reserve,
                    "token_safety_margin": context.token_safety_margin,
                    "budget_mode": context.budget_mode,
                    "reference_variant": context.reference_variant,
                    "reference_prompt_tokens": context.reference_prompt_tokens,
                    "context_tokens": context.estimated_context_tokens,
                    "prompt_tokens_before_call": context.estimated_prompt_tokens,
                    "prompt_token_utilization": context.prompt_token_utilization,
                    "actual_provider_prompt_tokens": None,
                    "actual_reference_prompt_tokens": None,
                    "actual_prompt_token_utilization": None,
                    "budget_truncated": context.budget_truncated,
                    "budget_passed_before_call": (
                        context.estimated_prompt_tokens
                        + context.output_token_reserve
                        + context.token_safety_margin
                        <= context.max_prompt_tokens
                    ),
                    "matched_budget_passed_before_call": (
                        context.estimated_prompt_tokens
                        <= context.reference_prompt_tokens
                        if context.reference_prompt_tokens is not None
                        else None
                    ),
                    "budget_passed_after_call": None,
                    "matched_budget_passed_after_call": None,
                    "matched_budget_low_utilization": (
                        context.prompt_token_utilization < 0.90
                        if context.prompt_token_utilization is not None
                        else None
                    ),
                    "matched_budget_low_utilization_reason": (
                        "whole_item_boundary_or_candidate_exhaustion"
                        if context.prompt_token_utilization is not None
                        and context.prompt_token_utilization < 0.90
                        else None
                    ),
                    "matched_boundary_backoff_count": context.metadata.get(
                        "matched_boundary_backoff_count"
                    ),
                }
            )
            progress_reporter.advance()
    progress_reporter.finish_stage()
    _write_jsonl(experiment_dir / "retrieval_plan_rows.jsonl", retrieval_plan_rows)
    _write_jsonl(experiment_dir / "variant_context_rows.jsonl", context_rows)
    _write_csv(
        experiment_dir / "token_budget_audit.csv",
        token_budget_rows,
        [
            "variant",
            "sample_id",
            "query_task_id",
            "token_counter",
            "token_counter_exact",
            "max_total_tokens",
            "global_max_total_tokens",
            "output_token_reserve",
            "token_safety_margin",
            "budget_mode",
            "reference_variant",
            "reference_prompt_tokens",
            "context_tokens",
            "prompt_tokens_before_call",
            "prompt_token_utilization",
            "actual_provider_prompt_tokens",
            "actual_reference_prompt_tokens",
            "actual_prompt_token_utilization",
            "budget_truncated",
            "budget_passed_before_call",
            "matched_budget_passed_before_call",
            "budget_passed_after_call",
            "matched_budget_passed_after_call",
            "matched_budget_low_utilization",
            "matched_budget_low_utilization_reason",
            "matched_boundary_backoff_count",
        ],
    )

    def expected_generation_prompt_sha256(
        variant: str,
        query_task_id: str,
    ) -> str:
        sample_row = source_rows[query_task_id]
        context = contexts[(variant, query_task_id)]
        policy = VARIANT_POLICIES[variant]
        prompt = context_builder.answer_prompt(
            question=str(sample_row.get("question") or ""),
            context_text=str(context.get("context_text") or ""),
            source_support_constraint=policy.source_support_constraint,
        )
        return _sha256_bytes(prompt.encode("utf-8"))

    generation_tasks: list[tuple[str, str]] = []
    unavailable_stage_rows: list[dict[str, Any]] = []
    for sample_row in selected_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        answer_metadata = dict(
            dict(sample_row.get("metadata") or {}).get("answer_metadata") or {}
        )
        full_stage_prompt_tokens, full_stage_completion_tokens = _answer_stage_usage(
            answer_metadata,
            stage="post_validation",
        )
        if full_stage_prompt_tokens is None:
            initial_tokens = answer_metadata.get("answer_initial_prompt_tokens")
            repair_tokens = answer_metadata.get("answer_repair_prompt_tokens")
            full_stage_prompt_tokens = (
                int(initial_tokens or 0) + int(repair_tokens or 0)
                if initial_tokens is not None or repair_tokens is not None
                else None
            )
        if full_stage_completion_tokens is None:
            initial_tokens = answer_metadata.get("answer_initial_completion_tokens")
            repair_tokens = answer_metadata.get("answer_repair_completion_tokens")
            full_stage_completion_tokens = (
                int(initial_tokens or 0) + int(repair_tokens or 0)
                if initial_tokens is not None or repair_tokens is not None
                else None
            )
        observed_full_path = _job_path(
            experiment_dir,
            "generation",
            "observed_full_pipeline",
            query_task_id,
        )
        if (
            completed_experiment_job(
                "generation",
                "observed_full_pipeline",
                query_task_id,
            )
            is None
        ):
            _atomic_json(
                observed_full_path,
                {
                    "schema_version": "answer_generation_job_v1",
                    "contains_sensitive_text": True,
                    "status": "complete",
                    "variant": "observed_full_pipeline",
                    "sample_id": sample_row.get("sample_id"),
                    "query_task_id": query_task_id,
                    "experiment_config_hash": config_hash,
                    "answer_text": sample_row.get("answer_text"),
                    "generation_source": "observed_full_run",
                    "stage_observable": True,
                    "stage_source_field": "answer_text",
                    "support_refs": support_refs_from_answer_metadata(answer_metadata),
                    "invalid_support_refs": invalid_refs_from_answer_metadata(
                        answer_metadata
                    ),
                    "prompt_tokens": full_stage_prompt_tokens,
                    "completion_tokens": full_stage_completion_tokens,
                },
            )
        for variant, answer_payload in _derived_answers(sample_row).items():
            if answer_payload.get("stage_observable") is not True:
                unavailable_stage_rows.append(
                    {
                        "schema_version": "unavailable_answer_stage_v1",
                        "variant": variant,
                        "sample_id": sample_row.get("sample_id"),
                        "query_task_id": query_task_id,
                        "stage_source_field": answer_payload.get("stage_source_field"),
                        "reason": "required_saved_answer_stage_is_absent",
                        "judge_scheduled": False,
                    }
                )
                continue
            path = _job_path(experiment_dir, "generation", variant, query_task_id)
            if (
                completed_experiment_job(
                    "generation",
                    variant,
                    query_task_id,
                )
                is None
            ):
                _atomic_json(
                    path,
                    {
                        "schema_version": "answer_generation_job_v1",
                        "contains_sensitive_text": True,
                        "status": "complete",
                        "variant": variant,
                        "sample_id": sample_row.get("sample_id"),
                        "query_task_id": query_task_id,
                        "experiment_config_hash": config_hash,
                        **answer_payload,
                        "generation_source": "derived_from_saved_answer_stage",
                    },
                )
        for method in provided_baseline_methods:
            imported = imported_by_method_query[(method, query_task_id)]
            path = _job_path(experiment_dir, "generation", method, query_task_id)
            existing_imported = completed_experiment_job(
                "generation",
                method,
                query_task_id,
            )
            imported_source_sha = str(imported.get("source_file_sha256") or "")
            authoritative_source_sha = imported_source_sha or (
                str(baseline_attachment.get("sha256") or "")
                if baseline_attachment
                else ""
            )
            replacement_required = existing_imported is None or any(
                [
                    str(existing_imported.get("external_baseline_sha256") or "")
                    != authoritative_source_sha,
                    str(existing_imported.get("answer_text") or "")
                    != str(imported.get("answer_text") or ""),
                    existing_imported.get("prompt_tokens")
                    != imported.get("prompt_tokens"),
                    existing_imported.get("completion_tokens")
                    != imported.get("completion_tokens"),
                    existing_imported.get("latency_ms")
                    != imported.get("latency_ms"),
                    existing_imported.get("model") != imported.get("model"),
                    existing_imported.get("protocol") != imported.get("protocol"),
                ]
            )
            if replacement_required:
                _atomic_json(
                    path,
                    {
                        "schema_version": "answer_generation_job_v1",
                        "contains_sensitive_text": True,
                        "status": "complete",
                        "variant": method,
                        "sample_id": sample_row.get("sample_id"),
                        "query_task_id": query_task_id,
                        "experiment_config_hash": config_hash,
                        "answer_text": imported.get("answer_text"),
                        "generation_source": "imported_baseline",
                        "prompt_tokens": imported.get("prompt_tokens"),
                        "completion_tokens": imported.get("completion_tokens"),
                        "latency_ms": imported.get("latency_ms"),
                        "model": imported.get("model"),
                        "protocol": imported.get("protocol"),
                        "external_baseline_sha256": authoritative_source_sha,
                    },
                )
                if existing_imported is not None:
                    replaced_external_methods.add(method)
                    judge_path = _job_path(
                        experiment_dir,
                        "judge",
                        method,
                        query_task_id,
                    )
                    judge_path.unlink(missing_ok=True)
        for variant in selected_variants:
            if (
                completed_experiment_job(
                    "generation",
                    variant,
                    query_task_id,
                    expected_prompt_sha256=expected_generation_prompt_sha256(
                        variant,
                        query_task_id,
                    ),
                )
                is None
            ):
                generation_tasks.append((variant, query_task_id))
    _write_jsonl(
        experiment_dir / "unavailable_stage_rows.jsonl",
        unavailable_stage_rows,
    )
    if (
        reuse_experiment_path is not None
        and normalized_reuse_policy != "off"
    ):
        parent_reuse_report = _reuse_parent_experiment(
            parent_experiment_path=Path(reuse_experiment_path),
            experiment_dir=experiment_dir,
            child_config=config_payload,
            child_config_hash=config_hash,
            child_sampling=sampling_manifest,
            selected_variants=selected_variants,
            baseline_methods=baseline_methods,
            source_rows=source_rows,
            expected_generation_prompt_sha256=(
                expected_generation_prompt_sha256
            ),
            policy=normalized_reuse_policy,
        )
    elif (experiment_dir / "parent_experiment_reuse.json").exists():
        parent_reuse_report = _read_json(
            experiment_dir / "parent_experiment_reuse.json"
        )
    else:
        parent_reuse_report = {
            "schema_version": "parent_experiment_reuse_v1",
            "status": "disabled",
            "policy": normalized_reuse_policy,
            "generation_job_count": 0,
            "judgment_job_count": 0,
            "provider_call_row_count": 0,
            "errors": [],
        }
        _atomic_json(
            experiment_dir / "parent_experiment_reuse.json",
            parent_reuse_report,
        )
    sampling_manifest = {
        **sampling_manifest,
        "parent_query_overlap": int(
            parent_reuse_report.get("parent_query_overlap") or 0
        ),
        "parent_experiment_config_hash": parent_reuse_report.get(
            "parent_config_hash"
        ),
    }
    _atomic_json(experiment_dir / "sampling_manifest.json", sampling_manifest)
    generation_tasks = [
        (variant, query_task_id)
        for variant, query_task_id in generation_tasks
        if completed_experiment_job(
            "generation",
            variant,
            query_task_id,
            expected_prompt_sha256=expected_generation_prompt_sha256(
                variant,
                query_task_id,
            ),
        )
        is None
    ]
    manifest["parent_experiment_reuse"] = parent_reuse_report
    _atomic_json(manifest_path, manifest)
    if replaced_external_methods:
        previous_provider_rows = _read_jsonl(
            experiment_dir / "provider_call_rows.jsonl"
        )
        for row in previous_provider_rows:
            if (
                str(row.get("variant") or "") in replaced_external_methods
                and str(row.get("role") or "") == "independent_judge"
            ):
                row["superseded_external_attachment"] = True
                row["superseded_by_sha256"] = (
                    baseline_attachment.get("sha256")
                    if baseline_attachment
                    else None
                )
        _write_jsonl(
            experiment_dir / "provider_call_rows.jsonl",
            previous_provider_rows,
        )

    all_variants = list(
        dict.fromkeys(
            [
                *selected_variants,
                *OBSERVED_DIAGNOSTIC_VARIANTS,
                *[
                    variant
                    for variant in DERIVED_VARIANTS
                    if any(
                        completed_experiment_job(
                            "generation",
                            variant,
                            str(row.get("query_task_id") or ""),
                        )
                        is not None
                        for row in selected_rows
                    )
                ],
                *baseline_methods,
            ]
        )
    )
    generation_task_keys = set(generation_tasks)
    planned_judge_call_count = 0
    for sample_row in selected_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        for variant in all_variants:
            generation_ready = completed_experiment_job(
                "generation",
                variant,
                query_task_id,
            )
            generation_pending = (variant, query_task_id) in generation_task_keys
            if generation_ready is None and not generation_pending:
                continue
            if generation_pending:
                planned_judge_call_count += 1
                continue
            assert generation_ready is not None
            expected_judge_prompt_hash = _sha256_bytes(
                _judge_prompt(
                    str(sample_row.get("question") or ""),
                    str(sample_row.get("gold_answer") or ""),
                    str(generation_ready.get("answer_text") or ""),
                ).encode("utf-8")
            )
            if (
                completed_experiment_job(
                    "judge",
                    variant,
                    query_task_id,
                    expected_prompt_sha256=expected_judge_prompt_hash,
                )
                is None
            ):
                planned_judge_call_count += 1
    planned_provider_call_count = len(generation_tasks) + planned_judge_call_count
    provider_rows_path = experiment_dir / "provider_call_rows.jsonl"
    existing_provider_records = _read_jsonl(provider_rows_path)
    existing_call_item_uids = {
        str(row.get("call_item_uid") or "")
        for row in existing_provider_records
        if str(row.get("call_item_uid") or "")
    }
    recovered_before_execution = _provider_records_from_jobs(experiment_dir)
    for recovered in recovered_before_execution:
        call_item_uid = str(recovered.get("call_item_uid") or "")
        if call_item_uid and call_item_uid not in existing_call_item_uids:
            existing_provider_records.append(recovered)
            existing_call_item_uids.add(call_item_uid)
    if recovered_before_execution:
        _write_jsonl(provider_rows_path, existing_provider_records)
    existing_provider_call_count = _provider_attempt_count(
        existing_provider_records
    )
    existing_new_provider_call_count = _provider_attempt_count(
        row
        for row in existing_provider_records
        if row.get("call_origin") == "new"
    )
    projected_provider_call_count_total = (
        existing_provider_call_count + planned_provider_call_count
    )
    manifest["planned_generation_call_count"] = len(generation_tasks)
    manifest["planned_judge_call_count"] = planned_judge_call_count
    manifest["planned_provider_call_count"] = planned_provider_call_count
    call_plan = {
        "schema_version": "answer_ablation_call_plan_v1",
        "dry_run": dry_run,
        "selected_query_count": len(selected_rows),
        "generated_variant_count": len(selected_variants),
        "external_baseline_methods": baseline_methods,
        "generation_call_count": len(generation_tasks),
        "judge_call_count": planned_judge_call_count,
        "provider_call_count": planned_provider_call_count,
        "reused_generation_job_count": int(
            parent_reuse_report.get("generation_job_count") or 0
        ),
        "reused_judgment_job_count": int(
            parent_reuse_report.get("judgment_job_count") or 0
        ),
        "reused_provider_call_row_count": int(
            parent_reuse_report.get("provider_call_row_count") or 0
        ),
        "existing_provider_call_count": existing_provider_call_count,
        "existing_new_provider_call_count": existing_new_provider_call_count,
        "projected_provider_call_count_total": (
            projected_provider_call_count_total
        ),
        "max_provider_calls": max_provider_calls,
        "unavailable_stage_row_count": len(unavailable_stage_rows),
        "expected_variant_context_row_count": (
            len(selected_rows) * len(selected_variants)
        ),
        "expected_answer_row_count": (
            len(selected_rows)
            * (
                len(selected_variants)
                + len(OBSERVED_DIAGNOSTIC_VARIANTS)
                + len(baseline_methods)
            )
        ),
        "expected_successful_logical_provider_work": (
            len(selected_rows)
            * (
                len(selected_variants)
                + len(selected_variants)
                + len(OBSERVED_DIAGNOSTIC_VARIANTS)
                + len(baseline_methods)
            )
        ),
        "token_counter": answer_token_counter.name,
        "token_counter_exact": answer_token_counter.exact,
    }
    if (
        sampling_manifest.get("sampling_profile") == "rebuttal_200_v1"
        and normalized_reuse_policy == "require"
        and existing_new_provider_call_count == 0
    ):
        expected_counts = {
            "reused_generation_job_count": 540,
            "reused_judgment_job_count": 660,
            "reused_provider_call_row_count": 1200,
            "existing_provider_call_count": 1200,
            "existing_new_provider_call_count": 0,
            "generation_call_count": 1660,
            "judge_call_count": 1940,
            "provider_call_count": 3600,
            "expected_variant_context_row_count": 2200,
            "expected_answer_row_count": 2600,
            "unavailable_stage_row_count": 400,
            "expected_successful_logical_provider_work": 4800,
        }
        mismatches = {
            key: {
                "expected": expected,
                "observed": call_plan.get(key),
            }
            for key, expected in expected_counts.items()
            if call_plan.get(key) != expected
        }
        if mismatches:
            progress_reporter.fail(
                f"rebuttal_200_v1 call plan mismatch: {mismatches}"
            )
            raise ValueError(
                "rebuttal_200_v1 did not produce the required first-run call "
                f"plan: {mismatches}. No provider calls were made."
            )
    _atomic_json(experiment_dir / "call_plan.json", call_plan)
    if (
        planned_provider_call_count > 0
        and projected_provider_call_count_total > max_provider_calls
    ):
        manifest["status"] = "blocked_by_call_cap"
        manifest["invocations"][-1]["status"] = "blocked_by_call_cap"
        _atomic_json(manifest_path, manifest)
        raise ValueError(
            f"Experiment projects {projected_provider_call_count_total} total provider "
            f"calls ({existing_provider_call_count} existing + "
            f"{len(generation_tasks)} generation + {planned_judge_call_count} judge), "
            f"exceeding --max-provider-calls={max_provider_calls}. No provider calls "
            "were made."
        )
    _atomic_json(manifest_path, manifest)
    if dry_run:
        progress_reporter.start_stage("dry_run_validation", total=1)
        progress_reporter.advance()
        progress_reporter.finish_stage()
        manifest["status"] = "dry_run_complete"
        manifest["invocations"][-1].update(
            {
                "status": "dry_run_complete",
                "generation_call_count": 0,
                "judge_call_count": 0,
            }
        )
        _atomic_json(manifest_path, manifest)
        return finalize_report({
            **call_plan,
            "experiment_dir": str(experiment_dir),
            "manifest_path": str(manifest_path),
            "sampling_manifest": str(
                experiment_dir / "sampling_manifest.json"
            ),
            "token_budget_audit": str(
                experiment_dir / "token_budget_audit.csv"
            ),
        })

    if planned_provider_call_count == 0:
        provider_rows = _read_jsonl(
            experiment_dir / "provider_call_rows.jsonl"
        )
        known_call_item_uids = {
            str(row.get("call_item_uid") or "")
            for row in provider_rows
            if row.get("call_item_uid")
        }
        for recovered in _provider_records_from_jobs(experiment_dir):
            call_item_uid = str(recovered.get("call_item_uid") or "")
            if call_item_uid and call_item_uid not in known_call_item_uids:
                provider_rows.append(recovered)
                known_call_item_uids.add(call_item_uid)
        _write_jsonl(experiment_dir / "provider_call_rows.jsonl", provider_rows)
        progress_reporter.start_stage("analysis", total=1)
        report = analyze_answer_ablation(experiment_dir)
        progress_reporter.advance()
        progress_reporter.finish_stage()
        manifest["status"] = (
            "complete"
            if int(report.get("integrity_error_count") or 0) == 0
            else "incomplete"
        )
        manifest["generated_provider_call_count"] = 0
        manifest["judge_provider_call_count"] = 0
        manifest["provider_call_count_total"] = _provider_attempt_count(
            provider_rows
        )
        manifest["invocations"][-1].update(
            {
                "status": manifest["status"],
                "generation_call_count": 0,
                "judge_call_count": 0,
            }
        )
        manifest["artifacts"] = report
        _atomic_json(manifest_path, manifest)
        return finalize_report(report)

    _require_remote_credentials(backbone_provider_kind, backbone_model, role="backbone")
    _require_remote_credentials(
        independent_judge_provider_kind,
        independent_judge_model,
        role="independent judge",
    )
    backbone = build_llm_provider(
        provider_config,
        role="backbone",
        model_name=backbone_model,
        provider_kind=cast(ProviderKind, backbone_provider_kind),
    )
    independent_judge = build_llm_provider(
        provider_config,
        role="independent_judge",
        model_name=independent_judge_model,
        provider_kind=cast(ProviderKind, independent_judge_provider_kind),
    )
    assert isinstance(backbone, MeteredLLMProvider)
    assert isinstance(independent_judge, MeteredLLMProvider)
    backbone.set_call_namespace(
        run_id=config_hash[:16],
        worker_id=f"generation-{invocation_id}",
    )
    independent_judge.set_call_namespace(
        run_id=config_hash[:16],
        worker_id=f"judge-{invocation_id}",
    )

    rng = random.Random(sample_seed)
    rng.shuffle(generation_tasks)

    def generate_one(variant: str, query_task_id: str) -> bool:
        sample_row = source_rows[query_task_id]
        context = contexts[(variant, query_task_id)]
        policy = VARIANT_POLICIES[variant]
        logical_call_item_uid = (
            f"{config_hash}/generation/{variant}/{query_task_id}"
        )
        prompt = context_builder.answer_prompt(
            question=str(sample_row.get("question") or ""),
            context_text=str(context.get("context_text") or ""),
            source_support_constraint=policy.source_support_constraint,
        )
        path = _job_path(experiment_dir, "generation", variant, query_task_id)
        try:
            counted_prompt_tokens = answer_token_counter.count(prompt)
            if (
                counted_prompt_tokens
                + max_output_tokens
                + token_safety_margin
                > resolved_max_total_tokens
            ):
                raise ValueError(
                    "Pre-call token budget exceeded after context rendering: "
                    f"{counted_prompt_tokens}+{max_output_tokens}+"
                    f"{token_safety_margin}>{resolved_max_total_tokens}."
                )
            response = backbone.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata={
                    "task": "answer_generation",
                    "experiment_stage": "answer_ablation_generation",
                    "sample_id": sample_row.get("sample_id"),
                    "query_task_id": query_task_id,
                    "variant": variant,
                    "logical_call_item_uid": logical_call_item_uid,
                    "generation_temperature": generation_temperature,
                    "generation_seed": generation_seed,
                    "generation_max_tokens": max_output_tokens,
                },
            )
            provider_prompt_usage_available = response.prompt_tokens is not None
            provider_completion_usage_available = (
                response.completion_tokens is not None
            )
            payload = {
                "schema_version": "answer_generation_job_v1",
                "contains_sensitive_text": True,
                "status": "complete",
                "variant": variant,
                "sample_id": sample_row.get("sample_id"),
                "query_task_id": query_task_id,
                "answer_text": response.text,
                "generation_source": "counterfactual_rerun",
                "experiment_config_hash": config_hash,
                "support_refs": sorted(extract_source_refs(response.text)),
                "invalid_support_refs": sorted(
                    set(extract_source_refs(response.text))
                    - set(context.get("selected_source_refs") or [])
                ),
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                "counted_prompt_tokens": counted_prompt_tokens,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "provider_prompt_usage_available": (
                    provider_prompt_usage_available
                ),
                "provider_completion_usage_available": (
                    provider_completion_usage_available
                ),
                "provider_usage_available": (
                    provider_prompt_usage_available
                    and provider_completion_usage_available
                ),
                "max_total_tokens": resolved_max_total_tokens,
                "output_token_reserve": max_output_tokens,
                "token_safety_margin": token_safety_margin,
                "token_counter": answer_token_counter.name,
                "token_counter_exact": answer_token_counter.exact,
                "budget_mode": context.get("budget_mode"),
                "reference_variant": context.get("reference_variant"),
                "reference_prompt_tokens": context.get(
                    "reference_prompt_tokens"
                ),
                "matched_budget_passed_before_call": (
                    counted_prompt_tokens
                    <= int(context["reference_prompt_tokens"])
                    if context.get("reference_prompt_tokens") is not None
                    else None
                ),
                "budget_passed_after_call": (
                    _provider_prompt_within_total_budget(
                        actual_prompt_tokens=response.prompt_tokens,
                        max_output_tokens=max_output_tokens,
                        max_total_tokens=resolved_max_total_tokens,
                    )
                ),
                "latency_ms": response.metadata.get("latency_ms"),
                "provider_call_id": response.metadata.get("provider_call_id"),
                "provider_call_uid": response.metadata.get("provider_call_uid"),
                "call_item_uid": response.metadata.get("call_item_uid"),
                "logical_call_item_uid": response.metadata.get(
                    "logical_call_item_uid"
                ),
                "response_metadata": {
                    key: response.metadata.get(key)
                    for key in [
                        "requested_model",
                        "resolved_model",
                        "provider_request_id",
                        "system_fingerprint",
                        "finish_reason",
                    ]
                },
            }
        except Exception as exc:
            failed_call = _provider_record_for_logical_uid(
                backbone,
                logical_call_item_uid,
            )
            payload = {
                "schema_version": "answer_generation_job_v1",
                "contains_sensitive_text": True,
                "status": "error",
                "variant": variant,
                "sample_id": sample_row.get("sample_id"),
                "query_task_id": query_task_id,
                "experiment_config_hash": config_hash,
                "error_type": exc.__class__.__name__,
                "error_message": " ".join(str(exc).split())[:500],
                "prompt_tokens": failed_call.get("prompt_tokens"),
                "completion_tokens": failed_call.get("completion_tokens"),
                "latency_ms": failed_call.get("latency_ms"),
                "provider_call_id": failed_call.get("provider_call_id"),
                "provider_call_uid": failed_call.get("provider_call_uid"),
                "call_item_uid": failed_call.get("call_item_uid"),
                "logical_call_item_uid": logical_call_item_uid,
                "response_metadata": {
                    key: failed_call.get(key)
                    for key in [
                        "requested_model",
                        "resolved_model",
                        "provider_request_id",
                        "system_fingerprint",
                        "finish_reason",
                    ]
                },
            }
        _atomic_json(path, payload)
        return payload.get("status") == "complete"

    progress_reporter.start_stage(
        "generation",
        total=len(generation_tasks),
        reused=int(parent_reuse_report.get("generation_job_count") or 0),
    )
    with ThreadPoolExecutor(max_workers=effective_generation_concurrency) as executor:
        futures = [
            executor.submit(generate_one, variant, query_task_id)
            for variant, query_task_id in generation_tasks
        ]
        for future in as_completed(futures):
            succeeded = future.result()
            progress_reporter.advance(
                succeeded=succeeded,
                error=None if succeeded else "generation job failed",
            )
    progress_reporter.finish_stage(
        status=(
            "complete"
            if all(
                completed_experiment_job(
                    "generation",
                    variant,
                    query_task_id,
                )
                is not None
                for variant, query_task_id in generation_tasks
            )
            else "complete_with_errors"
        )
    )

    judge_tasks: list[tuple[str, str]] = []
    for sample_row in selected_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        variant_order = list(all_variants)
        random.Random(f"{sample_seed}:{query_task_id}").shuffle(variant_order)
        for variant in variant_order:
            generation_job = completed_experiment_job(
                "generation",
                variant,
                query_task_id,
            )
            if generation_job is None:
                continue
            expected_judge_prompt_hash = _sha256_bytes(
                _judge_prompt(
                    str(sample_row.get("question") or ""),
                    str(sample_row.get("gold_answer") or ""),
                    str(generation_job.get("answer_text") or ""),
                ).encode("utf-8")
            )
            if (
                completed_experiment_job(
                    "judge",
                    variant,
                    query_task_id,
                    expected_prompt_sha256=expected_judge_prompt_hash,
                )
                is None
            ):
                judge_tasks.append((variant, query_task_id))
    already_used = len(backbone.calls)
    projected_after_generation = (
        existing_provider_call_count + already_used + len(judge_tasks)
    )
    if projected_after_generation > max_provider_calls:
        raise ValueError(
            f"Experiment projects {projected_after_generation} total provider "
            "calls after generation, exceeding "
            f"--max-provider-calls={max_provider_calls}."
        )

    def judge_one(variant: str, query_task_id: str) -> bool:
        sample_row = source_rows[query_task_id]
        generation_job = _read_json(
            _job_path(experiment_dir, "generation", variant, query_task_id)
        )
        logical_call_item_uid = f"{config_hash}/judge/{variant}/{query_task_id}"
        prompt = _judge_prompt(
            str(sample_row.get("question") or ""),
            str(sample_row.get("gold_answer") or ""),
            str(generation_job.get("answer_text") or ""),
        )
        path = _job_path(experiment_dir, "judge", variant, query_task_id)
        try:
            response = independent_judge.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata={
                    "task": "locomo_judge",
                    "experiment_stage": "independent_answer_judge",
                    "sample_id": sample_row.get("sample_id"),
                    "query_task_id": query_task_id,
                    "variant": variant,
                    "logical_call_item_uid": logical_call_item_uid,
                    "generation_temperature": 0.0,
                    "generation_seed": generation_seed,
                    "generation_max_tokens": 128,
                },
            )
            verdict = _parse_judge_verdict(response.text)
            if verdict == "judge_error":
                raise ValueError(
                    "Independent judge did not return an explicit VERDICT line."
                )
            provider_prompt_usage_available = response.prompt_tokens is not None
            provider_completion_usage_available = (
                response.completion_tokens is not None
            )
            payload = {
                "schema_version": "independent_judgment_v1",
                "contains_sensitive_text": True,
                "status": "complete",
                "variant": variant,
                "sample_id": sample_row.get("sample_id"),
                "query_task_id": query_task_id,
                "experiment_config_hash": config_hash,
                "verdict": verdict,
                "rationale": response.text[:1000],
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                "prompt_is_blinded": True,
                "prompt_contains_variant_name": False,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "provider_prompt_usage_available": (
                    provider_prompt_usage_available
                ),
                "provider_completion_usage_available": (
                    provider_completion_usage_available
                ),
                "provider_usage_available": (
                    provider_prompt_usage_available
                    and provider_completion_usage_available
                ),
                "latency_ms": response.metadata.get("latency_ms"),
                "provider_call_id": response.metadata.get("provider_call_id"),
                "provider_call_uid": response.metadata.get("provider_call_uid"),
                "call_item_uid": response.metadata.get("call_item_uid"),
                "logical_call_item_uid": response.metadata.get(
                    "logical_call_item_uid"
                ),
                "response_metadata": {
                    key: response.metadata.get(key)
                    for key in [
                        "requested_model",
                        "resolved_model",
                        "provider_request_id",
                        "system_fingerprint",
                        "finish_reason",
                    ]
                },
            }
        except Exception as exc:
            failed_call = _provider_record_for_logical_uid(
                independent_judge,
                logical_call_item_uid,
            )
            payload = {
                "schema_version": "independent_judgment_v1",
                "contains_sensitive_text": True,
                "status": "error",
                "variant": variant,
                "sample_id": sample_row.get("sample_id"),
                "query_task_id": query_task_id,
                "experiment_config_hash": config_hash,
                "error_type": exc.__class__.__name__,
                "error_message": " ".join(str(exc).split())[:500],
                "prompt_tokens": failed_call.get("prompt_tokens"),
                "completion_tokens": failed_call.get("completion_tokens"),
                "latency_ms": failed_call.get("latency_ms"),
                "provider_call_id": failed_call.get("provider_call_id"),
                "provider_call_uid": failed_call.get("provider_call_uid"),
                "call_item_uid": failed_call.get("call_item_uid"),
                "logical_call_item_uid": logical_call_item_uid,
                "response_metadata": {
                    key: failed_call.get(key)
                    for key in [
                        "requested_model",
                        "resolved_model",
                        "provider_request_id",
                        "system_fingerprint",
                        "finish_reason",
                    ]
                },
            }
        _atomic_json(path, payload)
        return payload.get("status") == "complete"

    progress_reporter.start_stage(
        "judge",
        total=len(judge_tasks),
        reused=int(parent_reuse_report.get("judgment_job_count") or 0),
    )
    with ThreadPoolExecutor(max_workers=effective_judge_concurrency) as executor:
        futures = [
            executor.submit(judge_one, variant, query_task_id)
            for variant, query_task_id in judge_tasks
        ]
        for future in as_completed(futures):
            succeeded = future.result()
            progress_reporter.advance(
                succeeded=succeeded,
                error=None if succeeded else "judge job failed",
            )
    progress_reporter.finish_stage(
        status=(
            "complete"
            if all(
                completed_experiment_job(
                    "judge",
                    variant,
                    query_task_id,
                )
                is not None
                for variant, query_task_id in judge_tasks
            )
            else "complete_with_errors"
        )
    )

    provider_rows = [
        *_read_jsonl(experiment_dir / "provider_call_rows.jsonl"),
        *_provider_records(backbone),
        *_provider_records(independent_judge),
    ]
    known_call_item_uids = {
        str(row.get("call_item_uid") or "")
        for row in provider_rows
        if row.get("call_item_uid")
    }
    for recovered in _provider_records_from_jobs(experiment_dir):
        call_item_uid = str(recovered.get("call_item_uid") or "")
        if call_item_uid and call_item_uid not in known_call_item_uids:
            provider_rows.append(recovered)
            known_call_item_uids.add(call_item_uid)
    _write_jsonl(experiment_dir / "provider_call_rows.jsonl", provider_rows)
    progress_reporter.start_stage("analysis", total=1)
    report = analyze_answer_ablation(experiment_dir)
    progress_reporter.advance()
    progress_reporter.finish_stage()
    manifest["status"] = (
        "complete"
        if int(report.get("integrity_error_count") or 0) == 0
        else "incomplete"
    )
    manifest["generated_provider_call_count"] = len(backbone.calls)
    manifest["judge_provider_call_count"] = len(independent_judge.calls)
    manifest["provider_call_count_total"] = _provider_attempt_count(provider_rows)
    manifest["invocations"][-1].update(
        {
            "status": manifest["status"],
            "generation_call_count": len(backbone.calls),
            "judge_call_count": len(independent_judge.calls),
        }
    )
    manifest["artifacts"] = report
    _atomic_json(manifest_path, manifest)
    return finalize_report(report)


def analyze_answer_ablation(experiment_path: Path | str) -> dict[str, Any]:
    experiment_dir = Path(experiment_path).expanduser().resolve()
    manifest = _read_json(experiment_dir / "experiment_manifest.json")
    manifest_baseline_methods = list(
        manifest.get("baseline_methods")
        or dict(manifest.get("config") or {}).get("baseline_methods")
        or []
    )
    run_dir = Path(str(manifest["source_run_dir"]))
    _, sample_rows, _ = load_details_rows(run_dir)
    source_by_query = {str(row.get("query_task_id")): row for row in sample_rows}
    sampling = _read_json(experiment_dir / "sampling_manifest.json")
    selected = {
        str(row["query_task_id"]): row for row in sampling.get("selected_queries", [])
    }
    gold_label_rows = _read_jsonl(experiment_dir / "gold_label_rows.jsonl")
    gold_by_query = {
        str(row.get("query_task_id") or ""): row
        for row in gold_label_rows
        if str(row.get("query_task_id") or "")
    }
    if gold_label_rows and set(gold_by_query) != set(selected):
        raise ValueError(
            "gold_label_rows.jsonl does not match the sampled query set."
        )
    context_rows = _read_jsonl(experiment_dir / "variant_context_rows.jsonl")
    context_by_variant_query = {
        (str(row.get("variant")), str(row.get("query_task_id"))): row
        for row in context_rows
    }
    rows: list[dict[str, Any]] = []
    judgment_rows: list[dict[str, Any]] = []
    answer_stage_rows: list[dict[str, Any]] = []
    available_derived_variants = [
        variant
        for variant in DERIVED_VARIANTS
        if any(
            _completed_job(
                _job_path(experiment_dir, "generation", variant, query_task_id),
                expected_schema="answer_generation_job_v1",
                expected_config_hash=str(manifest.get("config_hash") or ""),
                expected_variant=variant,
                expected_query_task_id=query_task_id,
            )
            is not None
            for query_task_id in selected
        )
    ]
    variants = list(
        dict.fromkeys(
            [
                *manifest["config"]["variants"],
                *OBSERVED_DIAGNOSTIC_VARIANTS,
                *available_derived_variants,
                *manifest_baseline_methods,
            ]
        )
    )
    for query_task_id, selection in sorted(selected.items()):
        sample_row = source_by_query[query_task_id]
        gold_answer = str(sample_row.get("gold_answer") or "")
        gold_refs = {
            str(ref)
            for ref in list(
                gold_by_query.get(query_task_id, {}).get("gold_source_refs")
                or []
            )
            if str(ref).strip()
        }
        if not gold_refs and not gold_label_rows:
            gold_refs = set(
                extract_source_refs(
                    dict(
                        dict(sample_row.get("metadata") or {}).get(
                            "query_metadata"
                        )
                        or {}
                    )
                )
            )
        for variant in variants:
            generation_path = _job_path(
                experiment_dir, "generation", variant, query_task_id
            )
            judge_path = _job_path(experiment_dir, "judge", variant, query_task_id)
            generation = _completed_job(
                generation_path,
                expected_schema="answer_generation_job_v1",
                expected_config_hash=str(manifest.get("config_hash") or ""),
                expected_variant=variant,
                expected_query_task_id=query_task_id,
            )
            judgment = _completed_job(
                judge_path,
                expected_schema="independent_judgment_v1",
                expected_config_hash=str(manifest.get("config_hash") or ""),
                expected_variant=variant,
                expected_query_task_id=query_task_id,
            )
            if generation is None:
                continue
            context = context_by_variant_query.get((variant, query_task_id))
            generation_source = str(generation.get("generation_source") or "")
            if context is None and generation_source in {
                "observed_full_run",
                "derived_from_saved_answer_stage",
            }:
                context = context_by_variant_query.get(("full", query_task_id), {})
            elif context is None:
                context = {}
            selected_refs = set(
                str(ref) for ref in list(context.get("selected_source_refs") or [])
            )
            answer_text = str(generation.get("answer_text") or "")
            expected_judge_prompt_sha256 = _sha256_bytes(
                _judge_prompt(
                    str(sample_row.get("question") or ""),
                    gold_answer,
                    answer_text,
                ).encode("utf-8")
            )
            judge_prompt_hash_valid = (
                str(judgment.get("prompt_sha256") or "") == expected_judge_prompt_sha256
                if judgment
                else None
            )
            scoreable_answer_text = _scoreable_answer_text(answer_text)
            answer_refs = set(
                str(ref)
                for ref in list(generation.get("support_refs") or [])
                if str(ref).strip()
            ) or set(extract_source_refs(answer_text))
            if generation_source in {
                "observed_full_run",
                "derived_from_saved_answer_stage",
            }:
                selected_refs_for_support = set(
                    str(ref)
                    for ref in list(sample_row.get("retrieval_source_refs") or [])
                    if str(ref).strip()
                )
            else:
                selected_refs_for_support = selected_refs
            invalid_refs = sorted(
                set(
                    str(ref)
                    for ref in list(generation.get("invalid_support_refs") or [])
                )
                | (answer_refs - selected_refs_for_support)
            )
            answer_abstained = _answer_is_abstention(answer_text)
            evidence_ref_observable = (
                variant != "wiki_summaries" and generation_source != "imported_baseline"
            )
            support_observable = evidence_ref_observable
            source_supported_proxy = (
                bool(
                    support_observable
                    and not answer_abstained
                    and selected_refs_for_support
                    and answer_refs
                    and not invalid_refs
                    and (not gold_refs or bool(answer_refs & gold_refs))
                )
                if support_observable
                else None
            )
            source_validation_observable = support_observable
            would_pass_source_validation = (
                bool(
                    source_supported_proxy
                    or (answer_abstained and not invalid_refs)
                )
                if source_validation_observable
                else None
            )
            independent_judge_score = (
                judge_score(judgment.get("verdict")) if judgment else None
            )
            post_source_validation_judge_score = (
                (independent_judge_score if would_pass_source_validation else 0.0)
                if source_validation_observable and independent_judge_score is not None
                else None
            )
            evidence_refs_for_metrics = selected_refs_for_support
            row = {
                "schema_version": "answer_ablation_row_v1",
                "contains_sensitive_text": True,
                "variant": variant,
                "variant_display_name": VARIANT_DISPLAY_NAMES.get(
                    variant,
                    variant.replace("_", " ").title(),
                ),
                "comparison_group": _comparison_group(
                    variant,
                    set(manifest_baseline_methods),
                ),
                "sample_id": sample_row.get("sample_id"),
                "query_task_id": query_task_id,
                "stratum": selection.get("stratum"),
                "question": sample_row.get("question"),
                "gold_answer": sample_row.get("gold_answer"),
                "answer_text": answer_text,
                "generation_source": generation_source,
                "stage_observable": generation.get("stage_observable"),
                "stage_source_field": generation.get("stage_source_field"),
                "independent_judge_verdict": judgment.get("verdict")
                if judgment
                else None,
                "independent_judge_score": independent_judge_score,
                "independent_judge_prompt_hash_valid": judge_prompt_hash_valid,
                "token_f1": _token_f1(scoreable_answer_text, gold_answer),
                "bleu1": bleu1(scoreable_answer_text, gold_answer),
                "answer_abstained": answer_abstained,
                "evidence_ref_observable": evidence_ref_observable,
                "selected_source_ref_count": (
                    len(evidence_refs_for_metrics)
                    if evidence_ref_observable
                    else None
                ),
                "gold_source_ref_count": len(gold_refs),
                "gold_ref_coverage": (
                    len(evidence_refs_for_metrics & gold_refs) / len(gold_refs)
                    if evidence_ref_observable and gold_refs
                    else None
                ),
                "answer_support_refs": sorted(answer_refs),
                "invalid_answer_support_refs": invalid_refs,
                "source_supported_proxy": source_supported_proxy,
                "unsupported_answer_risk": (
                    bool(
                        invalid_refs
                        or (
                            not answer_abstained
                            and not source_supported_proxy
                        )
                    )
                    if support_observable
                    else None
                ),
                "source_validation_observable": source_validation_observable,
                "would_pass_source_validation": would_pass_source_validation,
                "post_source_validation_judge_score": (
                    post_source_validation_judge_score
                ),
                "estimated_context_tokens": context.get("estimated_context_tokens"),
                "estimated_prompt_tokens": context.get("estimated_prompt_tokens"),
                "prompt_tokens": generation.get("prompt_tokens"),
                "completion_tokens": generation.get("completion_tokens"),
                "generation_latency_ms": generation.get("latency_ms"),
                "budget_truncated": context.get("budget_truncated"),
                "budget_mode": context.get("budget_mode"),
                "global_max_total_tokens": context.get(
                    "global_max_total_tokens"
                ),
                "reference_variant": context.get("reference_variant"),
                "reference_prompt_tokens": context.get(
                    "reference_prompt_tokens"
                ),
                "prompt_token_utilization": context.get(
                    "prompt_token_utilization"
                ),
                "token_estimator": context.get("token_estimator")
                or TOKEN_ESTIMATOR_NAME,
                "token_counter": context.get("token_counter"),
                "token_counter_exact": context.get("token_counter_exact"),
                "token_safety_margin": context.get("token_safety_margin"),
                "budget_passed_after_call": generation.get(
                    "budget_passed_after_call"
                ),
            }
            rows.append(row)
            answer_stage_rows.append(
                {
                    "schema_version": "answer_stage_v1",
                    "contains_sensitive_text": True,
                    "variant": variant,
                    "sample_id": sample_row.get("sample_id"),
                    "query_task_id": query_task_id,
                    "generation_source": generation.get("generation_source"),
                    "stage_observable": generation.get("stage_observable"),
                    "stage_source_field": generation.get("stage_source_field"),
                    "answer_text": answer_text,
                    "provider_call_uid": generation.get("provider_call_uid"),
                    "call_item_uid": generation.get("call_item_uid"),
                    "support_refs": list(generation.get("support_refs") or []),
                    "invalid_support_refs": list(
                        generation.get("invalid_support_refs") or []
                    ),
                    "prompt_tokens": generation.get("prompt_tokens"),
                    "completion_tokens": generation.get("completion_tokens"),
                }
            )
            if judgment:
                judgment_rows.append(
                    {
                        **judgment,
                        "prompt_hash_matches_blinded_template": (
                            judge_prompt_hash_valid
                        ),
                    }
                )

    actual_full_prompt_tokens = {
        str(row.get("query_task_id") or ""): int(row["prompt_tokens"])
        for row in rows
        if row.get("variant") == "full" and row.get("prompt_tokens") is not None
    }
    for row in rows:
        if row.get("budget_mode") != "full_prompt_matched_v1":
            row["actual_reference_prompt_tokens"] = None
            row["actual_prompt_token_utilization"] = None
            row["matched_budget_passed_after_call"] = None
            continue
        query_task_id = str(row.get("query_task_id") or "")
        reference_tokens = actual_full_prompt_tokens.get(query_task_id)
        candidate_tokens = (
            int(row["prompt_tokens"])
            if row.get("prompt_tokens") is not None
            else None
        )
        row["actual_reference_prompt_tokens"] = reference_tokens
        row["actual_prompt_token_utilization"] = (
            candidate_tokens / reference_tokens
            if candidate_tokens is not None and reference_tokens
            else None
        )
        row["matched_budget_passed_after_call"] = (
            candidate_tokens <= reference_tokens
            if candidate_tokens is not None and reference_tokens is not None
            else None
        )

    _write_jsonl(experiment_dir / "answer_ablation_rows.jsonl", rows)
    _write_jsonl(experiment_dir / "answer_stage_rows.jsonl", answer_stage_rows)
    _write_jsonl(experiment_dir / "independent_judgment_rows.jsonl", judgment_rows)
    summary_rows: list[dict[str, Any]] = []
    for variant in sorted({str(row.get("variant")) for row in rows}):
        group = [row for row in rows if row.get("variant") == variant]
        judge_values = [
            float(row["independent_judge_score"])
            for row in group
            if row.get("independent_judge_score") is not None
        ]
        judge_accuracy_values = [
            score
            for row in group
            if (
                score := judge_accuracy(
                    row.get("independent_judge_verdict")
                )
            )
            is not None
        ]
        judge_partial_values = [
            float(
                str(row.get("independent_judge_verdict") or "") == "partial"
            )
            for row in group
            if row.get("independent_judge_verdict") is not None
        ]
        support_values = [
            bool(row["source_supported_proxy"])
            for row in group
            if row.get("source_supported_proxy") is not None
        ]
        risk_values = [
            bool(row["unsupported_answer_risk"])
            for row in group
            if row.get("unsupported_answer_risk") is not None
        ]
        source_validation_values = [
            bool(row["would_pass_source_validation"])
            for row in group
            if row.get("would_pass_source_validation") is not None
        ]
        post_validation_judge_values = [
            float(row["post_source_validation_judge_score"])
            for row in group
            if row.get("post_source_validation_judge_score") is not None
        ]
        stage_observability_values = [
            bool(row["stage_observable"])
            for row in group
            if row.get("stage_observable") is not None
        ]
        summary_rows.append(
            {
                "variant": variant,
                "variant_display_name": VARIANT_DISPLAY_NAMES.get(
                    variant,
                    variant.replace("_", " ").title(),
                ),
                "comparison_group": _comparison_group(
                    variant,
                    set(manifest_baseline_methods),
                ),
                "query_count": len(group),
                "mean_independent_judge_score": mean(judge_values)
                if judge_values
                else None,
                "independent_judge_accuracy": (
                    mean(judge_accuracy_values)
                    if judge_accuracy_values
                    else None
                ),
                "independent_judge_partial_rate": (
                    mean(judge_partial_values)
                    if judge_partial_values
                    else None
                ),
                "mean_token_f1": mean(float(row["token_f1"]) for row in group),
                "mean_bleu1": mean(float(row["bleu1"]) for row in group),
                "abstention_rate": mean(
                    float(bool(row.get("answer_abstained"))) for row in group
                ),
                "mean_gold_ref_coverage": mean(
                    float(row["gold_ref_coverage"])
                    for row in group
                    if row.get("gold_ref_coverage") is not None
                )
                if any(row.get("gold_ref_coverage") is not None for row in group)
                else None,
                "source_supported_proxy_rate": (
                    mean(float(value) for value in support_values)
                    if support_values
                    else None
                ),
                "unsupported_answer_risk_rate": (
                    mean(float(value) for value in risk_values) if risk_values else None
                ),
                "source_validation_pass_rate": (
                    mean(float(value) for value in source_validation_values)
                    if source_validation_values
                    else None
                ),
                "mean_post_source_validation_judge_score": (
                    mean(post_validation_judge_values)
                    if post_validation_judge_values
                    else None
                ),
                "stage_observable_rate": (
                    mean(float(value) for value in stage_observability_values)
                    if stage_observability_values
                    else None
                ),
                "mean_estimated_prompt_tokens": _mean_present(
                    group,
                    "estimated_prompt_tokens",
                ),
                "mean_observed_generation_tokens": (
                    mean(
                        float(row["prompt_tokens"])
                        + float(row["completion_tokens"])
                        for row in group
                        if row.get("prompt_tokens") is not None
                        and row.get("completion_tokens") is not None
                    )
                    if any(
                        row.get("prompt_tokens") is not None
                        and row.get("completion_tokens") is not None
                        for row in group
                    )
                    else None
                ),
            }
        )
    _write_csv(
        experiment_dir / "answer_ablation_table.csv",
        summary_rows,
        [
            "variant",
            "variant_display_name",
            "comparison_group",
            "query_count",
            "mean_independent_judge_score",
            "independent_judge_accuracy",
            "independent_judge_partial_rate",
            "mean_token_f1",
            "mean_bleu1",
            "abstention_rate",
            "mean_gold_ref_coverage",
            "source_supported_proxy_rate",
            "unsupported_answer_risk_rate",
            "source_validation_pass_rate",
            "mean_post_source_validation_judge_score",
            "stage_observable_rate",
            "mean_estimated_prompt_tokens",
            "mean_observed_generation_tokens",
        ],
    )
    primary_comparison_variants = set(manifest["config"]["variants"])
    paired = paired_bootstrap_rows(
        [
            row
            for row in rows
            if str(row.get("variant") or "") in primary_comparison_variants
        ],
        reference_variant="full",
        iterations=10_000,
        seed=int(manifest["config"].get("sample_seed") or 7),
    )
    _write_csv(
        experiment_dir / "paired_statistics.csv",
        paired,
        [
            "reference_variant",
            "variant",
            "metric",
            "metric_direction",
            "paired_query_count",
            "reference_mean",
            "variant_mean",
            "mean_delta_variant_minus_reference",
            "ci95_low",
            "ci95_high",
            "win_count",
            "tie_count",
            "loss_count",
            "bootstrap_iterations",
            "bootstrap_seed",
        ],
    )
    prevalence_weighted_paired = prevalence_weighted_paired_bootstrap_rows(
        [
            row
            for row in rows
            if str(row.get("variant") or "") in primary_comparison_variants
        ],
        prevalence={
            str(key): float(value)
            for key, value in dict(
                sampling.get("population_prevalence") or {}
            ).items()
        },
        reference_variant="full",
        iterations=10_000,
        seed=int(manifest["config"].get("sample_seed") or 7),
    )
    _write_csv(
        experiment_dir / "prevalence_weighted_paired_statistics.csv",
        prevalence_weighted_paired,
        [
            "reference_variant",
            "variant",
            "metric",
            "metric_direction",
            "estimator",
            "paired_query_count",
            "prevalence_coverage",
            "reference_mean",
            "variant_mean",
            "mean_delta_variant_minus_reference",
            "ci95_low",
            "ci95_high",
            "win_count",
            "tie_count",
            "loss_count",
            "bootstrap_iterations",
            "bootstrap_seed",
        ],
    )
    cluster_paired = dialogue_cluster_paired_bootstrap_rows(
        [
            row
            for row in rows
            if str(row.get("variant") or "") in primary_comparison_variants
        ],
        reference_variant="full",
        iterations=10_000,
        seed=int(manifest["config"].get("sample_seed") or 7),
    )
    _write_csv(
        experiment_dir / "dialogue_cluster_paired_statistics.csv",
        cluster_paired,
        [
            "reference_variant",
            "variant",
            "metric",
            "metric_direction",
            "estimator",
            "paired_query_count",
            "dialogue_cluster_count",
            "reference_mean",
            "variant_mean",
            "mean_delta_variant_minus_reference",
            "ci95_low",
            "ci95_high",
            "win_count",
            "tie_count",
            "loss_count",
            "bootstrap_iterations",
            "bootstrap_seed",
        ],
    )
    external_paired = paired_bootstrap_rows(
        [
            row
            for row in rows
            if str(row.get("variant") or "")
            in {"full", *set(manifest_baseline_methods)}
        ],
        reference_variant="full",
        iterations=10_000,
        seed=int(manifest["config"].get("sample_seed") or 7),
    )
    _write_csv(
        experiment_dir / "external_baseline_paired_statistics.csv",
        external_paired,
        [
            "reference_variant",
            "variant",
            "metric",
            "metric_direction",
            "paired_query_count",
            "reference_mean",
            "variant_mean",
            "mean_delta_variant_minus_reference",
            "ci95_low",
            "ci95_high",
            "win_count",
            "tie_count",
            "loss_count",
            "bootstrap_iterations",
            "bootstrap_seed",
        ],
    )
    stage_paired = paired_bootstrap_rows(
        [
            row
            for row in rows
            if str(row.get("variant") or "") == "observed_full_pipeline"
            or (
                str(row.get("variant") or "") in set(DERIVED_VARIANTS)
                and row.get("stage_observable") is True
            )
        ],
        reference_variant="observed_full_pipeline",
        iterations=10_000,
        seed=int(manifest["config"].get("sample_seed") or 7),
    )
    _write_csv(
        experiment_dir / "stage_diagnostic_statistics.csv",
        stage_paired,
        [
            "reference_variant",
            "variant",
            "metric",
            "metric_direction",
            "paired_query_count",
            "reference_mean",
            "variant_mean",
            "mean_delta_variant_minus_reference",
            "ci95_low",
            "ci95_high",
            "win_count",
            "tie_count",
            "loss_count",
            "bootstrap_iterations",
            "bootstrap_seed",
        ],
    )
    stratum_rows = stratum_summary_rows(
        rows,
        prevalence=dict(sampling.get("population_prevalence") or {}),
    )
    prevalence = dict(sampling.get("population_prevalence") or {})
    selected_counts_by_stratum = {
        str(stratum): int(count)
        for stratum, count in dict(
            sampling.get("selected_counts_by_stratum") or {}
        ).items()
    }

    for variant in sorted({str(row.get("variant")) for row in rows}):
        variant_strata = [
            row for row in stratum_rows if row.get("variant") == variant
        ]
        weighted_judge, judge_prevalence_coverage = (
            complete_prevalence_weighted_metric(
                variant_strata,
                field="mean_independent_judge_score",
                count_field="independent_judge_score_count",
                prevalence=prevalence,
                selected_counts_by_stratum=selected_counts_by_stratum,
            )
        )
        weighted_judge_accuracy, judge_accuracy_prevalence_coverage = (
            complete_prevalence_weighted_metric(
                variant_strata,
                field="independent_judge_accuracy",
                count_field="independent_judge_accuracy_count",
                prevalence=prevalence,
                selected_counts_by_stratum=selected_counts_by_stratum,
            )
        )
        weighted_f1, f1_prevalence_coverage = complete_prevalence_weighted_metric(
            variant_strata,
            field="mean_token_f1",
            count_field="token_f1_count",
            prevalence=prevalence,
            selected_counts_by_stratum=selected_counts_by_stratum,
        )
        weighted_bleu, bleu_prevalence_coverage = complete_prevalence_weighted_metric(
            variant_strata,
            field="mean_bleu1",
            count_field="bleu1_count",
            prevalence=prevalence,
            selected_counts_by_stratum=selected_counts_by_stratum,
        )
        weighted_abstention, abstention_prevalence_coverage = (
            complete_prevalence_weighted_metric(
                variant_strata,
                field="abstention_rate",
                count_field="abstention_count",
                prevalence=prevalence,
                selected_counts_by_stratum=selected_counts_by_stratum,
            )
        )
        weighted_source_support, source_support_prevalence_coverage = (
            complete_prevalence_weighted_metric(
                variant_strata,
                field="source_supported_proxy_rate",
                count_field="source_supported_proxy_count",
                prevalence=prevalence,
                selected_counts_by_stratum=selected_counts_by_stratum,
            )
        )
        weighted_unsupported_risk, unsupported_risk_prevalence_coverage = (
            complete_prevalence_weighted_metric(
                variant_strata,
                field="unsupported_answer_risk_rate",
                count_field="unsupported_answer_risk_count",
                prevalence=prevalence,
                selected_counts_by_stratum=selected_counts_by_stratum,
            )
        )
        weighted_source_validation, source_validation_prevalence_coverage = (
            complete_prevalence_weighted_metric(
                variant_strata,
                field="source_validation_pass_rate",
                count_field="source_validation_count",
                prevalence=prevalence,
                selected_counts_by_stratum=selected_counts_by_stratum,
            )
        )
        weighted_post_validation_judge, post_validation_prevalence_coverage = (
            complete_prevalence_weighted_metric(
                variant_strata,
                field="mean_post_source_validation_judge_score",
                count_field="post_source_validation_judge_count",
                prevalence=prevalence,
                selected_counts_by_stratum=selected_counts_by_stratum,
            )
        )
        stratum_rows.append(
            {
                "variant": variant,
                "stratum": "prevalence_weighted_overall",
                "query_count": sum(
                    int(row.get("query_count") or 0) for row in variant_strata
                ),
                "prevalence_weight": 1.0,
                "mean_independent_judge_score": weighted_judge,
                "independent_judge_accuracy": weighted_judge_accuracy,
                "mean_token_f1": weighted_f1,
                "mean_bleu1": weighted_bleu,
                "abstention_rate": weighted_abstention,
                "source_supported_proxy_rate": weighted_source_support,
                "unsupported_answer_risk_rate": weighted_unsupported_risk,
                "source_validation_pass_rate": weighted_source_validation,
                "mean_post_source_validation_judge_score": (
                    weighted_post_validation_judge
                ),
                "mean_estimated_prompt_tokens": None,
                "independent_judge_score_count": sum(
                    int(row.get("independent_judge_score_count") or 0)
                    for row in variant_strata
                ),
                "independent_judge_accuracy_count": sum(
                    int(row.get("independent_judge_accuracy_count") or 0)
                    for row in variant_strata
                ),
                "token_f1_count": sum(
                    int(row.get("token_f1_count") or 0)
                    for row in variant_strata
                ),
                "bleu1_count": sum(
                    int(row.get("bleu1_count") or 0)
                    for row in variant_strata
                ),
                "abstention_count": sum(
                    int(row.get("abstention_count") or 0)
                    for row in variant_strata
                ),
                "source_supported_proxy_count": sum(
                    int(row.get("source_supported_proxy_count") or 0)
                    for row in variant_strata
                ),
                "unsupported_answer_risk_count": sum(
                    int(row.get("unsupported_answer_risk_count") or 0)
                    for row in variant_strata
                ),
                "source_validation_count": sum(
                    int(row.get("source_validation_count") or 0)
                    for row in variant_strata
                ),
                "post_source_validation_judge_count": sum(
                    int(row.get("post_source_validation_judge_count") or 0)
                    for row in variant_strata
                ),
                "estimated_prompt_tokens_count": sum(
                    int(row.get("estimated_prompt_tokens_count") or 0)
                    for row in variant_strata
                ),
                "judge_prevalence_coverage": judge_prevalence_coverage,
                "judge_accuracy_prevalence_coverage": (
                    judge_accuracy_prevalence_coverage
                ),
                "f1_prevalence_coverage": f1_prevalence_coverage,
                "bleu_prevalence_coverage": bleu_prevalence_coverage,
                "abstention_prevalence_coverage": abstention_prevalence_coverage,
                "source_support_prevalence_coverage": (
                    source_support_prevalence_coverage
                ),
                "unsupported_risk_prevalence_coverage": (
                    unsupported_risk_prevalence_coverage
                ),
                "source_validation_prevalence_coverage": (
                    source_validation_prevalence_coverage
                ),
                "post_validation_prevalence_coverage": (
                    post_validation_prevalence_coverage
                ),
            }
        )
    _write_csv(
        experiment_dir / "stratum_summary.csv",
        stratum_rows,
        [
            "variant",
            "stratum",
            "query_count",
            "prevalence_weight",
            "mean_independent_judge_score",
            "independent_judge_score_count",
            "independent_judge_accuracy",
            "independent_judge_accuracy_count",
            "mean_token_f1",
            "token_f1_count",
            "mean_bleu1",
            "bleu1_count",
            "abstention_rate",
            "abstention_count",
            "source_supported_proxy_rate",
            "source_supported_proxy_count",
            "unsupported_answer_risk_rate",
            "unsupported_answer_risk_count",
            "source_validation_pass_rate",
            "source_validation_count",
            "mean_post_source_validation_judge_score",
            "post_source_validation_judge_count",
            "mean_estimated_prompt_tokens",
            "estimated_prompt_tokens_count",
            "judge_prevalence_coverage",
            "judge_accuracy_prevalence_coverage",
            "f1_prevalence_coverage",
            "bleu_prevalence_coverage",
            "abstention_prevalence_coverage",
            "source_support_prevalence_coverage",
            "unsupported_risk_prevalence_coverage",
            "source_validation_prevalence_coverage",
            "post_validation_prevalence_coverage",
        ],
    )
    answer_by_variant_query = {
        (str(row.get("variant") or ""), str(row.get("query_task_id") or "")): row
        for row in rows
    }
    token_budget_rows: list[dict[str, Any]] = []
    for context in context_rows:
        key = (
            str(context.get("variant") or ""),
            str(context.get("query_task_id") or ""),
        )
        answer = answer_by_variant_query.get(key, {})
        actual_prompt_tokens = answer.get("prompt_tokens")
        max_total_tokens = int(
            context.get("max_prompt_tokens")
            or manifest["config"].get("max_total_tokens")
            or manifest["config"].get("max_prompt_tokens")
            or 0
        )
        output_reserve = int(context.get("output_token_reserve") or 0)
        safety_margin = int(
            context.get("token_safety_margin")
            or manifest["config"].get("token_safety_margin")
            or 0
        )
        reference_prompt_tokens = context.get("reference_prompt_tokens")
        actual_reference_prompt_tokens = actual_full_prompt_tokens.get(key[1])
        actual_prompt_token_utilization = (
            float(actual_prompt_tokens) / actual_reference_prompt_tokens
            if actual_prompt_tokens is not None
            and actual_reference_prompt_tokens
            and reference_prompt_tokens is not None
            else None
        )
        matched_budget_passed_after_call = (
            int(actual_prompt_tokens) <= actual_reference_prompt_tokens
            if actual_prompt_tokens is not None
            and actual_reference_prompt_tokens is not None
            and reference_prompt_tokens is not None
            else None
        )
        prompt_tokens_before_call = context.get("estimated_prompt_tokens")
        local_prompt_utilization = (
            float(context["prompt_token_utilization"])
            if context.get("prompt_token_utilization") is not None
            else None
        )
        token_budget_rows.append(
            {
                "variant": key[0],
                "sample_id": context.get("sample_id"),
                "query_task_id": key[1],
                "token_counter": context.get("token_counter")
                or context.get("token_estimator"),
                "token_counter_exact": context.get("token_counter_exact"),
                "max_total_tokens": max_total_tokens,
                "global_max_total_tokens": context.get(
                    "global_max_total_tokens"
                )
                or manifest["config"].get("max_total_tokens")
                or manifest["config"].get("max_prompt_tokens"),
                "output_token_reserve": output_reserve,
                "token_safety_margin": safety_margin,
                "budget_mode": context.get("budget_mode"),
                "reference_variant": context.get("reference_variant"),
                "reference_prompt_tokens": reference_prompt_tokens,
                "context_tokens": context.get("estimated_context_tokens"),
                "prompt_tokens_before_call": prompt_tokens_before_call,
                "prompt_token_utilization": context.get(
                    "prompt_token_utilization"
                ),
                "actual_provider_prompt_tokens": actual_prompt_tokens,
                "actual_reference_prompt_tokens": (
                    actual_reference_prompt_tokens
                    if reference_prompt_tokens is not None
                    else None
                ),
                "actual_prompt_token_utilization": (
                    actual_prompt_token_utilization
                ),
                "counted_actual_prompt_delta": (
                    int(actual_prompt_tokens) - int(prompt_tokens_before_call)
                    if actual_prompt_tokens is not None
                    and prompt_tokens_before_call is not None
                    else None
                ),
                "budget_truncated": context.get("budget_truncated"),
                "budget_passed_before_call": (
                    int(context.get("estimated_prompt_tokens") or 0)
                    + output_reserve
                    + safety_margin
                    <= max_total_tokens
                    if max_total_tokens
                    else None
                ),
                "matched_budget_passed_before_call": (
                    int(prompt_tokens_before_call)
                    <= int(reference_prompt_tokens)
                    if prompt_tokens_before_call is not None
                    and reference_prompt_tokens is not None
                    else None
                ),
                "budget_passed_after_call": (
                    _provider_prompt_within_total_budget(
                        actual_prompt_tokens=actual_prompt_tokens,
                        max_output_tokens=output_reserve,
                        max_total_tokens=max_total_tokens,
                    )
                    if max_total_tokens
                    else None
                ),
                "matched_budget_passed_after_call": (
                    matched_budget_passed_after_call
                ),
                "post_provider_match_status": (
                    "pass"
                    if matched_budget_passed_after_call is True
                    else (
                        "fail"
                        if matched_budget_passed_after_call is False
                        else (
                            "not_available"
                            if reference_prompt_tokens is not None
                            else "not_applicable"
                        )
                    )
                ),
                "matched_budget_low_utilization": (
                    local_prompt_utilization < 0.90
                    if local_prompt_utilization is not None
                    else None
                ),
                "matched_budget_low_utilization_reason": (
                    "whole_item_boundary_or_candidate_exhaustion"
                    if local_prompt_utilization is not None
                    and local_prompt_utilization < 0.90
                    else None
                ),
                "matched_boundary_backoff_count": dict(
                    context.get("metadata") or {}
                ).get("matched_boundary_backoff_count"),
            }
        )
    _write_csv(
        experiment_dir / "token_budget_audit.csv",
        token_budget_rows,
        [
            "variant",
            "sample_id",
            "query_task_id",
            "token_counter",
            "token_counter_exact",
            "max_total_tokens",
            "global_max_total_tokens",
            "output_token_reserve",
            "token_safety_margin",
            "budget_mode",
            "reference_variant",
            "reference_prompt_tokens",
            "context_tokens",
            "prompt_tokens_before_call",
            "prompt_token_utilization",
            "actual_provider_prompt_tokens",
            "actual_reference_prompt_tokens",
            "actual_prompt_token_utilization",
            "counted_actual_prompt_delta",
            "budget_truncated",
            "budget_passed_before_call",
            "matched_budget_passed_before_call",
            "budget_passed_after_call",
            "matched_budget_passed_after_call",
            "post_provider_match_status",
            "matched_budget_low_utilization",
            "matched_budget_low_utilization_reason",
            "matched_boundary_backoff_count",
        ],
    )
    matched_compliance_rows: list[dict[str, Any]] = []
    for variant in [
        "full_context_matched",
        "hybrid_raw_rag_matched",
    ]:
        group = [
            row for row in token_budget_rows if row.get("variant") == variant
        ]
        if not group:
            continue
        local_utilization = [
            float(row["prompt_token_utilization"])
            for row in group
            if row.get("prompt_token_utilization") is not None
        ]
        actual_utilization = [
            float(row["actual_prompt_token_utilization"])
            for row in group
            if row.get("actual_prompt_token_utilization") is not None
        ]
        matched_compliance_rows.append(
            {
                "variant": variant,
                "variant_display_name": VARIANT_DISPLAY_NAMES[variant],
                "query_count": len(group),
                "local_match_pass_count": sum(
                    row.get("matched_budget_passed_before_call") is True
                    for row in group
                ),
                "local_match_fail_count": sum(
                    row.get("matched_budget_passed_before_call") is False
                    for row in group
                ),
                "provider_match_pass_count": sum(
                    row.get("matched_budget_passed_after_call") is True
                    for row in group
                ),
                "provider_match_fail_count": sum(
                    row.get("matched_budget_passed_after_call") is False
                    for row in group
                ),
                "provider_match_unavailable_count": sum(
                    row.get("matched_budget_passed_after_call") is None
                    for row in group
                ),
                "low_utilization_count": sum(
                    row.get("matched_budget_low_utilization") is True
                    for row in group
                ),
                "mean_local_utilization": (
                    mean(local_utilization) if local_utilization else None
                ),
                "mean_provider_utilization": (
                    mean(actual_utilization) if actual_utilization else None
                ),
            }
        )
    _write_csv(
        experiment_dir / "matched_budget_compliance.csv",
        matched_compliance_rows,
        [
            "variant",
            "variant_display_name",
            "query_count",
            "local_match_pass_count",
            "local_match_fail_count",
            "provider_match_pass_count",
            "provider_match_fail_count",
            "provider_match_unavailable_count",
            "low_utilization_count",
            "mean_local_utilization",
            "mean_provider_utilization",
        ],
    )
    provider_rows = _read_jsonl(experiment_dir / "provider_call_rows.jsonl")
    active_provider_rows = [
        row
        for row in provider_rows
        if row.get("superseded_external_attachment") is not True
    ]
    completion_truncation_rows: list[dict[str, Any]] = []
    for variant, role in sorted(
        {
            (
                str(row.get("variant") or "unknown"),
                str(row.get("role") or "unknown"),
            )
            for row in active_provider_rows
        }
    ):
        scoped = [
            row
            for row in active_provider_rows
            if str(row.get("variant") or "unknown") == variant
            and str(row.get("role") or "unknown") == role
        ]
        known_finish = [
            row for row in scoped if row.get("finish_reason") is not None
        ]
        completion_truncation_rows.append(
            {
                "variant": variant,
                "role": role,
                "provider_call_count": len(scoped),
                "known_finish_reason_count": len(known_finish),
                "finish_reason_length_count": sum(
                    str(row.get("finish_reason") or "").lower() == "length"
                    for row in known_finish
                ),
                "finish_reason_length_rate": (
                    sum(
                        str(row.get("finish_reason") or "").lower() == "length"
                        for row in known_finish
                    )
                    / len(known_finish)
                    if known_finish
                    else None
                ),
            }
        )
    _write_csv(
        experiment_dir / "completion_truncation_summary.csv",
        completion_truncation_rows,
        [
            "variant",
            "role",
            "provider_call_count",
            "known_finish_reason_count",
            "finish_reason_length_count",
            "finish_reason_length_rate",
        ],
    )
    fingerprint_counts = Counter(
        (
            str(row.get("role") or "unknown"),
            str(row.get("model") or row.get("resolved_model") or "unknown"),
            str(row.get("system_fingerprint") or "unavailable"),
            str(row.get("call_origin") or "unknown"),
        )
        for row in active_provider_rows
    )
    provider_fingerprint_rows = [
        {
            "role": role,
            "model": model,
            "system_fingerprint": fingerprint,
            "call_origin": call_origin,
            "provider_call_count": count,
        }
        for (role, model, fingerprint, call_origin), count in sorted(
            fingerprint_counts.items()
        )
    ]
    _write_csv(
        experiment_dir / "provider_fingerprint_summary.csv",
        provider_fingerprint_rows,
        [
            "role",
            "model",
            "system_fingerprint",
            "call_origin",
            "provider_call_count",
        ],
    )
    provider_origin_rows: list[dict[str, Any]] = []
    for call_origin, role in sorted(
        {
            (
                str(row.get("call_origin") or "unknown"),
                str(row.get("role") or "unknown"),
            )
            for row in active_provider_rows
        }
    ):
        scoped = [
            row
            for row in active_provider_rows
            if str(row.get("call_origin") or "unknown") == call_origin
            and str(row.get("role") or "unknown") == role
        ]
        complete_usage = [
            row
            for row in scoped
            if row.get("prompt_tokens") is not None
            and row.get("completion_tokens") is not None
        ]
        latency_rows = [
            row for row in scoped if row.get("latency_ms") is not None
        ]
        provider_origin_rows.append(
            {
                "call_origin": call_origin,
                "role": role,
                "provider_call_count": len(scoped),
                "prompt_tokens": (
                    sum(int(row["prompt_tokens"]) for row in scoped)
                    if len(complete_usage) == len(scoped)
                    else None
                ),
                "completion_tokens": (
                    sum(int(row["completion_tokens"]) for row in scoped)
                    if len(complete_usage) == len(scoped)
                    else None
                ),
                "total_tokens": (
                    sum(
                        int(row["prompt_tokens"])
                        + int(row["completion_tokens"])
                        for row in scoped
                    )
                    if len(complete_usage) == len(scoped)
                    else None
                ),
                "usage_missing_count": len(scoped) - len(complete_usage),
                "latency_ms": (
                    sum(float(row["latency_ms"]) for row in latency_rows)
                    if len(latency_rows) == len(scoped)
                    else None
                ),
                "known_latency_ms": sum(
                    float(row["latency_ms"]) for row in latency_rows
                ),
                "latency_missing_count": len(scoped) - len(latency_rows),
            }
        )
    _write_csv(
        experiment_dir / "provider_call_origin_summary.csv",
        provider_origin_rows,
        [
            "call_origin",
            "role",
            "provider_call_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "usage_missing_count",
            "latency_ms",
            "known_latency_ms",
            "latency_missing_count",
        ],
    )
    variant_cost_rows: list[dict[str, Any]] = []
    for variant in sorted(
        {str(row.get("variant") or "") for row in active_provider_rows}
    ):
        variant_group = [
            row
            for row in active_provider_rows
            if str(row.get("variant") or "") == variant
        ]
        roles = sorted({str(row.get("role") or "unknown") for row in variant_group})
        for role in [*roles, "all_provider_roles"]:
            group = (
                variant_group
                if role == "all_provider_roles"
                else [row for row in variant_group if str(row.get("role")) == role]
            )
            prompt_usage_rows = [
                row for row in group if row.get("prompt_tokens") is not None
            ]
            completion_usage_rows = [
                row for row in group if row.get("completion_tokens") is not None
            ]
            complete_usage_rows = [
                row
                for row in group
                if row.get("prompt_tokens") is not None
                and row.get("completion_tokens") is not None
            ]
            latency_rows = [
                row for row in group if row.get("latency_ms") is not None
            ]
            variant_cost_rows.append(
                {
                    "variant": variant,
                    "variant_display_name": VARIANT_DISPLAY_NAMES.get(
                        variant,
                        variant.replace("_", " ").title(),
                    ),
                    "role": role,
                    "deployment_scope": (
                        "benchmark_only"
                        if role == "independent_judge"
                        else (
                            "mixed"
                            if role == "all_provider_roles"
                            else "deployment"
                        )
                    ),
                    "cost_source": "observed_provider_calls",
                    "provider_call_count": len(
                        {
                            str(row.get("provider_call_uid"))
                            for row in group
                            if row.get("provider_call_uid")
                        }
                    ),
                    "prompt_tokens": (
                        sum(int(row["prompt_tokens"]) for row in prompt_usage_rows)
                        if group and len(prompt_usage_rows) == len(group)
                        else None
                    ),
                    "completion_tokens": (
                        sum(
                            int(row["completion_tokens"])
                            for row in completion_usage_rows
                        )
                        if group and len(completion_usage_rows) == len(group)
                        else None
                    ),
                    "total_tokens": (
                        sum(
                            int(row["prompt_tokens"])
                            + int(row["completion_tokens"])
                            for row in complete_usage_rows
                        )
                        if group and len(complete_usage_rows) == len(group)
                        else None
                    ),
                    "known_prompt_tokens": sum(
                        int(row["prompt_tokens"]) for row in prompt_usage_rows
                    ),
                    "known_completion_tokens": sum(
                        int(row["completion_tokens"])
                        for row in completion_usage_rows
                    ),
                    "known_total_tokens": sum(
                        int(row["prompt_tokens"])
                        + int(row["completion_tokens"])
                        for row in complete_usage_rows
                    ),
                    "latency_ms": (
                        sum(float(row["latency_ms"]) for row in latency_rows)
                        if group and len(latency_rows) == len(group)
                        else None
                    ),
                    "known_latency_ms": sum(
                        float(row["latency_ms"]) for row in latency_rows
                    ),
                    "latency_missing_count": len(group) - len(latency_rows),
                    "usage_missing_count": len(group)
                    - len(complete_usage_rows),
                    "usage_complete": bool(group)
                    and len(complete_usage_rows) == len(group),
                }
            )
    for variant in manifest_baseline_methods:
        imported_rows = [
            row
            for row in rows
            if str(row.get("variant") or "") == variant
            and str(row.get("generation_source") or "") == "imported_baseline"
        ]
        if not imported_rows:
            continue
        imported_prompt_usage_rows = [
            row for row in imported_rows if row.get("prompt_tokens") is not None
        ]
        imported_completion_usage_rows = [
            row
            for row in imported_rows
            if row.get("completion_tokens") is not None
        ]
        rows_with_usage = [
            row
            for row in imported_rows
            if row.get("prompt_tokens") is not None
            and row.get("completion_tokens") is not None
        ]
        imported_latency_rows = [
            row
            for row in imported_rows
            if row.get("generation_latency_ms") is not None
        ]
        variant_cost_rows.append(
            {
                "variant": variant,
                "variant_display_name": VARIANT_DISPLAY_NAMES.get(
                    variant,
                    variant.replace("_", " ").title(),
                ),
                "role": "historical_generation",
                "deployment_scope": "historical_external",
                "cost_source": "imported_baseline_usage",
                "provider_call_count": len(imported_rows),
                "prompt_tokens": (
                    sum(
                        int(row["prompt_tokens"])
                        for row in imported_prompt_usage_rows
                    )
                    if imported_rows
                    and len(imported_prompt_usage_rows) == len(imported_rows)
                    else None
                ),
                "completion_tokens": (
                    sum(
                        int(row["completion_tokens"])
                        for row in imported_completion_usage_rows
                    )
                    if imported_rows
                    and len(imported_completion_usage_rows)
                    == len(imported_rows)
                    else None
                ),
                "total_tokens": (
                    sum(
                        int(row["prompt_tokens"])
                        + int(row["completion_tokens"])
                        for row in rows_with_usage
                    )
                    if imported_rows
                    and len(rows_with_usage) == len(imported_rows)
                    else None
                ),
                "known_prompt_tokens": sum(
                    int(row["prompt_tokens"])
                    for row in imported_prompt_usage_rows
                ),
                "known_completion_tokens": sum(
                    int(row["completion_tokens"])
                    for row in imported_completion_usage_rows
                ),
                "known_total_tokens": sum(
                    int(row["prompt_tokens"])
                    + int(row["completion_tokens"])
                    for row in rows_with_usage
                ),
                "latency_ms": (
                    sum(
                        float(row["generation_latency_ms"])
                        for row in imported_latency_rows
                    )
                    if len(imported_latency_rows) == len(imported_rows)
                    else None
                ),
                "known_latency_ms": sum(
                    float(row["generation_latency_ms"])
                    for row in imported_latency_rows
                ),
                "latency_missing_count": (
                    len(imported_rows) - len(imported_latency_rows)
                ),
                "usage_missing_count": len(imported_rows) - len(rows_with_usage),
                "usage_complete": bool(imported_rows)
                and len(rows_with_usage) == len(imported_rows),
            }
        )
    _write_csv(
        experiment_dir / "variant_cost_summary.csv",
        variant_cost_rows,
        [
            "variant",
            "variant_display_name",
            "role",
            "deployment_scope",
            "cost_source",
            "provider_call_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "known_prompt_tokens",
            "known_completion_tokens",
            "known_total_tokens",
            "latency_ms",
            "known_latency_ms",
            "latency_missing_count",
            "usage_missing_count",
            "usage_complete",
        ],
    )
    integrity_errors: list[str] = []
    integrity_warnings: list[str] = []
    if any(row.get("prompt_contains_variant_name") for row in judgment_rows):
        integrity_errors.append("independent judge prompt leaked a variant name")
    invalid_judge_prompt_hashes = [
        str(row.get("query_task_id") or "")
        for row in judgment_rows
        if row.get("prompt_hash_matches_blinded_template") is not True
    ]
    if invalid_judge_prompt_hashes:
        integrity_errors.append(
            "independent judge prompt hash mismatch for queries: "
            f"{invalid_judge_prompt_hashes[:5]}"
        )
    call_item_uids = [
        str(row.get("call_item_uid") or "")
        for row in provider_rows
        if row.get("call_item_uid")
    ]
    missing_call_item_uid_count = sum(
        1 for row in provider_rows if not str(row.get("call_item_uid") or "")
    )
    duplicate_call_items = [
        uid for uid, count in Counter(call_item_uids).items() if count > 1
    ]
    if missing_call_item_uid_count:
        integrity_errors.append(
            f"{missing_call_item_uid_count} provider call rows lack call_item_uid"
        )
    if duplicate_call_items:
        integrity_errors.append(
            f"duplicate call_item_uid values: {duplicate_call_items[:5]}"
        )
    logical_call_uids = [
        str(row.get("logical_call_item_uid") or "")
        for row in active_provider_rows
        if row.get("logical_call_item_uid")
    ]
    duplicate_logical_calls = [
        uid for uid, count in Counter(logical_call_uids).items() if count > 1
    ]
    if duplicate_logical_calls:
        integrity_warnings.append(
            "one or more logical jobs required multiple provider attempts: "
            f"{duplicate_logical_calls[:5]}"
        )
    configured_max_total_tokens = int(
        manifest["config"].get("max_total_tokens")
        or manifest["config"].get("max_prompt_tokens")
        or 0
    )
    configured_safety_margin = int(
        manifest["config"].get("token_safety_margin") or 0
    )
    if any(
        int(row.get("estimated_prompt_tokens") or 0)
        + int(manifest["config"].get("max_output_tokens") or 0)
        + int(row.get("token_safety_margin") or configured_safety_margin)
        > configured_max_total_tokens
        for row in context_rows
    ):
        integrity_errors.append(
            "one or more contexts exceed the configured total token budget"
        )
    if any(row.get("budget_passed_after_call") is False for row in rows):
        integrity_errors.append(
            "one or more provider-reported prompts exceed the configured total "
            "token budget"
        )
    matched_local_failures = [
        f"{row.get('variant')}/{row.get('query_task_id')}"
        for row in token_budget_rows
        if row.get("matched_budget_passed_before_call") is False
    ]
    if matched_local_failures:
        integrity_errors.append(
            "one or more matched baselines exceed the locally counted Full "
            f"TrajWiki prompt: {matched_local_failures[:5]}"
        )
    matched_provider_failures = [
        f"{row.get('variant')}/{row.get('query_task_id')}"
        for row in token_budget_rows
        if row.get("matched_budget_passed_after_call") is False
    ]
    if matched_provider_failures:
        integrity_errors.append(
            "one or more matched baselines exceed the provider-reported Full "
            f"TrajWiki prompt: {matched_provider_failures[:5]}"
        )
    matched_provider_unavailable = [
        f"{row.get('variant')}/{row.get('query_task_id')}"
        for row in token_budget_rows
        if row.get("reference_prompt_tokens") is not None
        and row.get("matched_budget_passed_after_call") is None
    ]
    if (
        sampling.get("sampling_profile") == "rebuttal_200_v1"
        and matched_provider_unavailable
    ):
        integrity_errors.append(
            "provider prompt usage is unavailable for one or more matched "
            f"comparisons: {matched_provider_unavailable[:5]}"
        )
    low_utilization_count = sum(
        row.get("matched_budget_low_utilization") is True
        for row in token_budget_rows
    )
    if low_utilization_count:
        integrity_warnings.append(
            f"{low_utilization_count} matched contexts use less than 90% of "
            "their Full TrajWiki prompt budget because only whole evidence "
            "items are retained or candidates are exhausted"
        )
    finish_reason_length_count = sum(
        str(row.get("finish_reason") or "").lower() == "length"
        for row in active_provider_rows
    )
    if finish_reason_length_count:
        integrity_warnings.append(
            f"{finish_reason_length_count} provider calls ended with "
            "finish_reason=length; they were retained and not automatically retried"
        )
    missing_actual_prompt_usage = [
        f"{row.get('variant')}/{row.get('query_task_id')}"
        for row in rows
        if row.get("generation_source") == "counterfactual_rerun"
        and row.get("prompt_tokens") is None
    ]
    if missing_actual_prompt_usage:
        integrity_errors.append(
            "provider prompt-token usage is unavailable for regenerated answers: "
            f"{missing_actual_prompt_usage[:5]}"
        )
    if manifest["config"].get("require_exact_token_counter") and any(
        row.get("token_counter_exact") is not True for row in context_rows
    ):
        integrity_errors.append(
            "formal run required an exact token counter but one or more contexts "
            "used an estimator"
        )
    expected_context_row_count = len(selected) * len(
        manifest["config"]["variants"]
    )
    if len(context_rows) != expected_context_row_count:
        integrity_errors.append(
            f"context row count is {len(context_rows)}, expected "
            f"{expected_context_row_count}"
        )
    if sampling.get("sampling_profile") == "rebuttal_200_v1":
        expected_selected_counts = {
            "strict_deep_history": 9,
            "update_sensitive": 132,
            "ordinary": 59,
        }
        if dict(sampling.get("selected_counts_by_stratum") or {}) != (
            expected_selected_counts
        ):
            integrity_errors.append(
                "rebuttal_200_v1 selected stratum counts are not 9/132/59"
            )
        if sampling.get("sampling_status") != "post_hoc_nested_extension":
            integrity_errors.append(
                "rebuttal_200_v1 is not marked as a post-hoc nested extension"
            )
    core_expected_variants = list(
        dict.fromkeys(
            [
                *manifest["config"]["variants"],
                *OBSERVED_DIAGNOSTIC_VARIANTS,
                *manifest_baseline_methods,
            ]
        )
    )
    expected_answer_keys = {
        (variant, query_task_id)
        for variant in core_expected_variants
        for query_task_id in selected
    }
    expected_answer_keys.update(
        (variant, query_task_id)
        for variant in DERIVED_VARIANTS
        for query_task_id in selected
        if _completed_job(
            _job_path(
                experiment_dir,
                "generation",
                variant,
                query_task_id,
            ),
            expected_schema="answer_generation_job_v1",
            expected_config_hash=str(manifest.get("config_hash") or ""),
            expected_variant=variant,
            expected_query_task_id=query_task_id,
        )
        is not None
    )
    expected_answer_row_count = len(expected_answer_keys)
    if len(rows) != expected_answer_row_count:
        integrity_errors.append(
            f"answer row count is {len(rows)}, expected {expected_answer_row_count}"
        )
    if len(judgment_rows) != expected_answer_row_count:
        integrity_errors.append(
            f"independent judgment row count is {len(judgment_rows)}, "
            f"expected {expected_answer_row_count}"
        )
    unobservable_stage_rows = [
        row
        for row in rows
        if str(row.get("variant") or "") in set(DERIVED_VARIANTS)
        and row.get("stage_observable") is not True
    ]
    if unobservable_stage_rows:
        integrity_warnings.append(
            f"{len(unobservable_stage_rows)} legacy answer-stage rows are excluded "
            "from formal stage statistics because exact stage captures are unavailable"
        )
    integrity = {
        "schema_version": "answer_ablation_integrity_v1",
        "error_count": len(integrity_errors),
        "errors": integrity_errors,
        "warning_count": len(integrity_warnings),
        "warnings": integrity_warnings,
        "missing_call_item_uid_count": missing_call_item_uid_count,
        "duplicate_call_item_uid_count": len(duplicate_call_items),
        "independent_judge_prompt_blinded": not any(
            row.get("prompt_contains_variant_name") for row in judgment_rows
        )
        and not invalid_judge_prompt_hashes,
        "query_count": len(selected),
        "expected_variant_context_row_count": expected_context_row_count,
        "variant_context_row_count": len(context_rows),
        "expected_answer_row_count": expected_answer_row_count,
        "answer_row_count": len(rows),
        "independent_judgment_row_count": len(judgment_rows),
        "provider_call_row_count": len(provider_rows),
        "active_provider_call_row_count": len(active_provider_rows),
        "superseded_provider_call_row_count": (
            len(provider_rows) - len(active_provider_rows)
        ),
        "parent_provider_call_row_count": sum(
            row.get("call_origin") == "parent_experiment"
            for row in active_provider_rows
        ),
        "new_provider_call_row_count": sum(
            row.get("call_origin") == "new"
            for row in active_provider_rows
        ),
        "matched_budget_local_failure_count": len(matched_local_failures),
        "matched_budget_provider_failure_count": len(
            matched_provider_failures
        ),
        "matched_budget_provider_unavailable_count": len(
            matched_provider_unavailable
        ),
        "matched_budget_low_utilization_count": low_utilization_count,
        "finish_reason_length_count": finish_reason_length_count,
    }
    _atomic_json(experiment_dir / "integrity_report.json", integrity)
    examples = sorted(
        rows,
        key=lambda row: (
            float(row.get("token_f1") or 0.0),
            str(row.get("variant")),
            str(row.get("query_task_id")),
        ),
    )[:40]
    _write_jsonl(experiment_dir / "variant_examples.jsonl", examples)
    rows_by_variant_query = {
        (str(row.get("variant") or ""), str(row.get("query_task_id") or "")): row
        for row in rows
    }
    balanced_examples: list[dict[str, Any]] = []
    comparator_variants = [
        variant
        for variant in [
            "full_context",
            "full_context_matched",
            "hybrid_raw_rag",
            "hybrid_raw_rag_matched",
            "naive_dense_rag",
            *manifest_baseline_methods,
        ]
        if any(str(row.get("variant") or "") == variant for row in rows)
    ]
    for comparator in comparator_variants:
        comparisons: list[dict[str, Any]] = []
        for query_task_id in selected:
            full_row = rows_by_variant_query.get(("full", query_task_id))
            other_row = rows_by_variant_query.get((comparator, query_task_id))
            if full_row is None or other_row is None:
                continue
            full_score = full_row.get("independent_judge_score")
            other_score = other_row.get("independent_judge_score")
            if full_score is None or other_score is None:
                continue
            delta = float(other_score) - float(full_score)
            comparisons.append(
                {
                    "schema_version": "balanced_failure_example_v1",
                    "contains_sensitive_text": True,
                    "comparator_variant": comparator,
                    "comparator_display_name": VARIANT_DISPLAY_NAMES.get(
                        comparator,
                        comparator.replace("_", " ").title(),
                    ),
                    "direction": (
                        "comparator_better"
                        if delta > 0
                        else ("trajwiki_better" if delta < 0 else "judge_tie")
                    ),
                    "judge_score_delta_comparator_minus_trajwiki": delta,
                    "sample_id": full_row.get("sample_id"),
                    "query_task_id": query_task_id,
                    "stratum": full_row.get("stratum"),
                    "question": full_row.get("question"),
                    "gold_answer": full_row.get("gold_answer"),
                    "trajwiki_answer": full_row.get("answer_text"),
                    "comparator_answer": other_row.get("answer_text"),
                    "trajwiki_judge_verdict": full_row.get(
                        "independent_judge_verdict"
                    ),
                    "comparator_judge_verdict": other_row.get(
                        "independent_judge_verdict"
                    ),
                    "trajwiki_gold_ref_coverage": full_row.get(
                        "gold_ref_coverage"
                    ),
                    "comparator_gold_ref_coverage": other_row.get(
                        "gold_ref_coverage"
                    ),
                }
            )
        for direction in ["comparator_better", "trajwiki_better", "judge_tie"]:
            candidates = [
                row for row in comparisons if row.get("direction") == direction
            ]
            candidates.sort(
                key=lambda row: (
                    -abs(
                        float(
                            row.get(
                                "judge_score_delta_comparator_minus_trajwiki"
                            )
                            or 0.0
                        )
                    ),
                    str(row.get("query_task_id") or ""),
                )
            )
            balanced_examples.extend(candidates[:5])
    _write_jsonl(
        experiment_dir / "balanced_failure_examples.jsonl",
        balanced_examples,
    )
    return {
        "schema_version": "answer_ablation_report_v1",
        "experiment_dir": str(experiment_dir),
        "query_count": len(selected),
        "variant_count": len({str(row.get("variant")) for row in rows}),
        "integrity_error_count": len(integrity_errors),
        "answer_ablation_rows": str(experiment_dir / "answer_ablation_rows.jsonl"),
        "answer_ablation_table": str(experiment_dir / "answer_ablation_table.csv"),
        "paired_statistics": str(experiment_dir / "paired_statistics.csv"),
        "prevalence_weighted_paired_statistics": str(
            experiment_dir / "prevalence_weighted_paired_statistics.csv"
        ),
        "dialogue_cluster_paired_statistics": str(
            experiment_dir / "dialogue_cluster_paired_statistics.csv"
        ),
        "external_baseline_paired_statistics": str(
            experiment_dir / "external_baseline_paired_statistics.csv"
        ),
        "stage_diagnostic_statistics": str(
            experiment_dir / "stage_diagnostic_statistics.csv"
        ),
        "unavailable_stage_rows": str(
            experiment_dir / "unavailable_stage_rows.jsonl"
        ),
        "stratum_summary": str(experiment_dir / "stratum_summary.csv"),
        "variant_cost_summary": str(experiment_dir / "variant_cost_summary.csv"),
        "token_budget_audit": str(experiment_dir / "token_budget_audit.csv"),
        "matched_budget_compliance": str(
            experiment_dir / "matched_budget_compliance.csv"
        ),
        "completion_truncation_summary": str(
            experiment_dir / "completion_truncation_summary.csv"
        ),
        "provider_fingerprint_summary": str(
            experiment_dir / "provider_fingerprint_summary.csv"
        ),
        "provider_call_origin_summary": str(
            experiment_dir / "provider_call_origin_summary.csv"
        ),
        "balanced_failure_examples": str(
            experiment_dir / "balanced_failure_examples.jsonl"
        ),
        "integrity_report": str(experiment_dir / "integrity_report.json"),
    }


def import_baseline_answers(
    source_path: Path | str,
    *,
    output_path: Path | str | None = None,
    method: str | None = None,
    run_path: Path | str | None = None,
) -> dict[str, Any]:
    """Normalize external per-query baseline predictions for later blinded judging."""

    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = list(payload if isinstance(payload, list) else payload.get("rows") or [])
    else:
        rows = _read_jsonl(source)
    source_sha256 = _sha256_file(source)
    normalized: list[dict[str, Any]] = []

    def first_present(*values: Any) -> Any:
        return next((value for value in values if value is not None), None)

    def mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or "")
        query_task_id = str(
            row.get("query_task_id")
            or row.get("question_id")
            or row.get("qa_uid")
            or ""
        )
        row_method = str(
            method
            or row.get("method")
            or row.get("variant")
            or row.get("baseline")
            or ""
        )
        answer_text = str(
            row.get("answer_text") or row.get("prediction") or row.get("answer") or ""
        )
        if not query_task_id or not row_method:
            raise ValueError(
                f"Baseline row {index} must include query_task_id and method/variant."
            )
        response = mapping(row.get("response"))
        metadata = mapping(row.get("metadata"))
        usage_candidates = [
            mapping(row.get("usage")),
            mapping(row.get("model_usage")),
            mapping(response.get("usage")),
            mapping(metadata.get("usage")),
            mapping(metadata.get("model_usage")),
        ]

        def usage_value(
            *keys: str,
            _usage_candidates: tuple[dict[str, Any], ...] = tuple(
                usage_candidates
            ),
        ) -> Any:
            return first_present(
                *(
                    usage.get(key)
                    for usage in _usage_candidates
                    for key in keys
                )
            )

        prompt_tokens = first_present(
            row.get("prompt_tokens"),
            row.get("input_tokens"),
            row.get("model_prompt_tokens"),
            usage_value("prompt_tokens", "input_tokens"),
        )
        completion_tokens = first_present(
            row.get("completion_tokens"),
            row.get("output_tokens"),
            row.get("model_completion_tokens"),
            usage_value("completion_tokens", "output_tokens"),
        )
        latency_ms = first_present(
            row.get("latency_ms"),
            usage_value("latency_ms"),
            (
                float(row["model_runtime_seconds"]) * 1000.0
                if row.get("model_runtime_seconds") is not None
                else None
            ),
            (
                float(usage_value("runtime_seconds")) * 1000.0
                if usage_value("runtime_seconds") is not None
                else None
            ),
        )
        question = str(row.get("question") or "")
        gold_answer = str(
            row.get("gold_answer") or row.get("reference_answer") or ""
        )
        normalized.append(
            {
                "schema_version": "imported_baseline_answer_v1",
                "contains_sensitive_text": True,
                "sample_id": sample_id,
                "query_task_id": query_task_id,
                "method": row_method,
                "answer_text": answer_text,
                "question_hash": (
                    _sha256_bytes(question.encode("utf-8")) if question else None
                ),
                "gold_answer_hash": (
                    _sha256_bytes(gold_answer.encode("utf-8"))
                    if gold_answer
                    else None
                ),
                "prompt_tokens": (
                    int(prompt_tokens) if prompt_tokens is not None else None
                ),
                "completion_tokens": (
                    int(completion_tokens)
                    if completion_tokens is not None
                    else None
                ),
                "latency_ms": (
                    float(latency_ms) if latency_ms is not None else None
                ),
                "model": row.get("model")
                or row.get("model_name")
                or row.get("backbone_model"),
                "protocol": row.get("protocol") or "historical_external",
                "source_file_sha256": source_sha256,
                "source_row_index": index,
            }
        )
    if not normalized:
        raise ValueError("Baseline prediction source contains no rows.")
    keys = [
        (str(row["method"]), str(row["query_task_id"])) for row in normalized
    ]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Baseline predictions contain duplicate method/query rows: {duplicates[:5]}"
        )

    alignment_report: dict[str, Any] | None = None
    if run_path is not None:
        run_dir = Path(run_path).expanduser().resolve()
        if run_dir.is_file():
            run_dir = run_dir.parent
        _, run_rows, _ = load_details_rows(run_dir)
        run_query_ids = [
            str(row.get("query_task_id") or "").strip()
            for row in run_rows
        ]
        duplicate_run_query_ids = [
            query_task_id
            for query_task_id, count in Counter(run_query_ids).items()
            if query_task_id and count > 1
        ]
        if any(not query_task_id for query_task_id in run_query_ids):
            raise ValueError("Source run contains a missing query_task_id.")
        if duplicate_run_query_ids:
            raise ValueError(
                "Source run contains duplicate query_task_id values: "
                f"{duplicate_run_query_ids[:5]}."
            )
        expected = {
            str(row.get("query_task_id") or ""): row for row in run_rows
        }
        normalized_ids = {str(row["query_task_id"]) for row in normalized}
        missing = sorted(set(expected) - normalized_ids)
        extra = sorted(normalized_ids - set(expected))
        method_alignment: dict[str, dict[str, Any]] = {}
        for method_name in sorted(
            {str(row.get("method") or "") for row in normalized}
        ):
            method_ids = {
                str(row["query_task_id"])
                for row in normalized
                if str(row.get("method") or "") == method_name
            }
            method_missing = sorted(set(expected) - method_ids)
            method_extra = sorted(method_ids - set(expected))
            method_alignment[method_name] = {
                "query_count": len(method_ids),
                "missing_query_count": len(method_missing),
                "extra_query_count": len(method_extra),
                "missing_queries": method_missing[:20],
                "extra_queries": method_extra[:20],
                "aligned": not method_missing and not method_extra,
            }
        mismatches: list[str] = []
        for row in normalized:
            query_task_id = str(row["query_task_id"])
            expected_row = expected.get(query_task_id)
            if expected_row is None:
                continue
            expected_sample_id = str(expected_row.get("sample_id") or "")
            if not row.get("sample_id"):
                mismatches.append(f"{query_task_id}:sample_id_missing")
            elif str(row["sample_id"]) != expected_sample_id:
                mismatches.append(f"{query_task_id}:sample_id")
            expected_question_hash = _sha256_bytes(
                str(expected_row.get("question") or "").encode("utf-8")
            )
            expected_gold_hash = _sha256_bytes(
                str(expected_row.get("gold_answer") or "").encode("utf-8")
            )
            if row.get("question_hash") is None:
                mismatches.append(f"{query_task_id}:question_missing")
            elif row.get("question_hash") != expected_question_hash:
                mismatches.append(f"{query_task_id}:question")
            if row.get("gold_answer_hash") is None:
                mismatches.append(f"{query_task_id}:gold_answer_missing")
            elif row.get("gold_answer_hash") != expected_gold_hash:
                mismatches.append(f"{query_task_id}:gold_answer")
        alignment_report = {
            "schema_version": "baseline_alignment_v1",
            "run_dir": str(run_dir),
            "expected_query_count": len(expected),
            "baseline_query_count": len(normalized_ids),
            "missing_query_count": len(missing),
            "extra_query_count": len(extra),
            "mismatch_count": len(mismatches),
            "missing_queries": missing[:20],
            "extra_queries": extra[:20],
            "mismatches": mismatches[:20],
            "methods": method_alignment,
            "aligned": (
                not missing
                and not extra
                and not mismatches
                and all(
                    bool(report.get("aligned"))
                    for report in method_alignment.values()
                )
            ),
        }
        if not alignment_report["aligned"]:
            incomplete_methods = sorted(
                method_name
                for method_name, report in method_alignment.items()
                if not report.get("aligned")
            )
            raise ValueError(
                "Baseline predictions do not align with the source run: "
                f"missing={len(missing)}, extra={len(extra)}, "
                f"mismatches={len(mismatches)}, "
                f"incomplete_methods={incomplete_methods}."
            )
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}.normalized.jsonl")
    )
    _write_jsonl(target, normalized)
    manifest = {
        "schema_version": "imported_baseline_manifest_v1",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "output_path": str(target),
        "row_count": len(normalized),
        "methods": sorted({row["method"] for row in normalized}),
        "run_path": str(Path(run_path).expanduser().resolve())
        if run_path is not None
        else None,
        "alignment": alignment_report,
    }
    _atomic_json(target.with_suffix(target.suffix + ".manifest.json"), manifest)
    if alignment_report is not None:
        _atomic_json(
            target.with_suffix(target.suffix + ".alignment.json"),
            alignment_report,
        )
    return manifest
