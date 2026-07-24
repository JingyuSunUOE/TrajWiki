"""Artifact integrity checks for benchmark runs and rebuttal experiments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from trajpatch.analysis.gold_labels import build_memory_index, load_details_rows
from trajpatch.analysis.memory_index import source_message_ids_for_refs
from trajpatch.experiments.progress import ExperimentProgress
from trajpatch.utils.json_utils import write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_experiment(path: Path) -> dict[str, Any]:
    progress = ExperimentProgress(path / "progress.json", enabled=False)
    progress.start_stage("validation", total=1)
    required = [
        "experiment_manifest.json",
        "sampling_manifest.json",
        "gold_label_rows.jsonl",
        "retrieval_plan_rows.jsonl",
        "variant_context_rows.jsonl",
        "answer_ablation_rows.jsonl",
        "answer_stage_rows.jsonl",
        "independent_judgment_rows.jsonl",
        "provider_call_rows.jsonl",
        "integrity_report.json",
        "token_budget_audit.csv",
        "unavailable_stage_rows.jsonl",
        "external_baseline_manifest.json",
    ]
    errors = [
        f"missing required artifact: {name}"
        for name in required
        if not (path / name).exists()
    ]
    warnings: list[str] = []
    manifest = (
        json.loads((path / "experiment_manifest.json").read_text(encoding="utf-8"))
        if (path / "experiment_manifest.json").exists()
        else {}
    )
    sampling = (
        json.loads((path / "sampling_manifest.json").read_text(encoding="utf-8"))
        if (path / "sampling_manifest.json").exists()
        else {}
    )
    integrity = (
        json.loads((path / "integrity_report.json").read_text(encoding="utf-8"))
        if (path / "integrity_report.json").exists()
        else {}
    )
    configured_sampling_profile = str(
        dict(manifest.get("config") or {}).get("sampling_profile") or ""
    )
    sampling_profile = (
        str(sampling.get("sampling_profile") or "")
        if configured_sampling_profile in {"", "auto"}
        else configured_sampling_profile
    )
    if sampling_profile == "rebuttal_200_v1":
        required_200 = [
            "parent_experiment_reuse.json",
            "progress.json",
            "prevalence_weighted_paired_statistics.csv",
            "dialogue_cluster_paired_statistics.csv",
            "matched_budget_compliance.csv",
            "completion_truncation_summary.csv",
            "provider_fingerprint_summary.csv",
            "provider_call_origin_summary.csv",
        ]
        errors.extend(
            f"missing required 200-query artifact: {name}"
            for name in required_200
            if not (path / name).exists()
        )
    if manifest and manifest.get("status") != "complete":
        errors.append(
            f"experiment manifest status is {manifest.get('status')!r}, expected 'complete'"
        )
    for error in list(integrity.get("errors") or []):
        errors.append(f"experiment integrity report: {error}")
    for warning in list(integrity.get("warnings") or []):
        warnings.append(f"experiment integrity report: {warning}")

    selected_rows = list(sampling.get("selected_queries") or [])
    selected_query_ids = [str(row.get("query_task_id") or "") for row in selected_rows]
    duplicate_selected = [
        query_task_id
        for query_task_id, count in Counter(selected_query_ids).items()
        if query_task_id and count > 1
    ]
    if duplicate_selected:
        errors.append(f"duplicate sampled query ids: {duplicate_selected[:5]}")
    selected_query_set = {
        query_task_id for query_task_id in selected_query_ids if query_task_id
    }
    configured_sample_size = int(
        dict(manifest.get("config") or {}).get("sample_size") or 0
    )
    if configured_sample_size and len(selected_query_set) != configured_sample_size:
        errors.append(
            f"sampling manifest has {len(selected_query_set)} unique queries, "
            f"expected {configured_sample_size}"
        )
    if sampling_profile == "rebuttal_200_v1":
        if dict(sampling.get("selected_counts_by_stratum") or {}) != {
            "strict_deep_history": 9,
            "update_sensitive": 132,
            "ordinary": 59,
        }:
            errors.append(
                "rebuttal_200_v1 selected stratum counts are not 9/132/59"
            )
        if sampling.get("sampling_status") != "post_hoc_nested_extension":
            errors.append(
                "rebuttal_200_v1 lacks its post_hoc_nested_extension marker"
            )
        if int(sampling.get("parent_query_overlap") or 0) != 60:
            errors.append(
                "rebuttal_200_v1 does not record all 60 parent queries as reused"
            )
        parent_reuse_path = path / "parent_experiment_reuse.json"
        if parent_reuse_path.exists():
            parent_reuse = json.loads(
                parent_reuse_path.read_text(encoding="utf-8")
            )
            for key, expected in [
                ("generation_job_count", 540),
                ("judgment_job_count", 660),
                ("provider_call_row_count", 1200),
                ("parent_query_overlap", 60),
            ]:
                if int(parent_reuse.get(key) or 0) != expected:
                    errors.append(
                        f"parent reuse {key} is {parent_reuse.get(key)!r}, "
                        f"expected {expected}"
                    )
            if parent_reuse.get("status") != "verified":
                errors.append("parent experiment reuse status is not verified")
    if sampling.get("source_run_dir") and str(
        Path(str(sampling["source_run_dir"])).expanduser().resolve()
    ) != str(Path(str(manifest.get("source_run_dir") or "")).expanduser().resolve()):
        errors.append(
            "sampling manifest source_run_dir does not match the experiment manifest"
        )
    expected_source_details_sha = str(
        manifest.get("source_details_sha256")
        or dict(manifest.get("config") or {}).get("source_details_sha256")
        or ""
    )
    if (
        expected_source_details_sha
        and str(sampling.get("source_details_sha256") or "")
        != expected_source_details_sha
    ):
        errors.append(
            "sampling manifest source details hash does not match the experiment manifest"
        )
    source_run_value = str(manifest.get("source_run_dir") or "").strip()
    if not source_run_value:
        errors.append("experiment manifest is missing source_run_dir")
    else:
        source_run_dir = Path(source_run_value).expanduser().resolve()
        source_details_path = source_run_dir / "details.json"
        if not source_details_path.exists():
            errors.append(f"source details artifact is missing: {source_details_path}")
        elif (
            expected_source_details_sha
            and _sha256_file(source_details_path) != expected_source_details_sha
        ):
            errors.append("source details artifact no longer matches its recorded hash")
    source_database_value = str(
        manifest.get("source_database_path") or ""
    ).strip()
    expected_database_sha = str(
        manifest.get("source_database_sha256")
        or dict(manifest.get("config") or {}).get("source_database_sha256")
        or ""
    )
    if not source_database_value:
        errors.append("experiment manifest is missing source_database_path")
    else:
        source_database_path = Path(source_database_value).expanduser().resolve()
        if not source_database_path.exists():
            errors.append(
                f"source database artifact is missing: {source_database_path}"
            )
        elif (
            expected_database_sha
            and _sha256_file(source_database_path) != expected_database_sha
        ):
            errors.append("source database artifact no longer matches its recorded hash")

    external_manifest_path = path / "external_baseline_manifest.json"
    external_manifest = (
        json.loads(external_manifest_path.read_text(encoding="utf-8"))
        if external_manifest_path.exists()
        else {}
    )
    manifest_attachments = list(
        manifest.get("external_baseline_attachments") or []
    )
    if external_manifest:
        if str(external_manifest.get("core_config_hash") or "") != str(
            manifest.get("config_hash") or ""
        ):
            errors.append("external baseline manifest has a mismatched core config hash")
        if list(external_manifest.get("attachments") or []) != manifest_attachments:
            errors.append(
                "external baseline attachments differ between experiment manifests"
            )
    attachment_methods: list[str] = []
    for attachment in manifest_attachments:
        attachment_path_value = str(attachment.get("path") or "").strip()
        attachment_sha = str(attachment.get("sha256") or "").strip()
        methods = [
            str(value)
            for value in list(attachment.get("methods") or [])
            if str(value).strip()
        ]
        attachment_methods.extend(methods)
        if not attachment_path_value:
            errors.append("external baseline attachment is missing its source path")
            continue
        attachment_path = Path(attachment_path_value).expanduser().resolve()
        if not attachment_path.exists():
            errors.append(
                f"external baseline attachment is missing: {attachment_path}"
            )
        elif not attachment_sha or _sha256_file(attachment_path) != attachment_sha:
            errors.append(
                "external baseline attachment no longer matches its recorded hash: "
                f"{attachment_path}"
            )
    duplicate_attachment_methods = [
        method
        for method, count in Counter(attachment_methods).items()
        if count > 1
    ]
    if duplicate_attachment_methods:
        errors.append(
            "external baseline methods occur in multiple attachments: "
            f"{duplicate_attachment_methods[:5]}"
        )
    gold_label_rows = _read_jsonl(path / "gold_label_rows.jsonl")
    gold_label_query_ids = [
        str(row.get("query_task_id") or "")
        for row in gold_label_rows
    ]
    if len(gold_label_query_ids) != len(set(gold_label_query_ids)):
        errors.append("gold-label rows contain duplicate query ids")
    if set(gold_label_query_ids) != selected_query_set:
        missing = sorted(selected_query_set - set(gold_label_query_ids))
        unexpected = sorted(set(gold_label_query_ids) - selected_query_set)
        errors.append(
            "gold-label query coverage mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if any(
        row.get("schema_version") != "gold_labels_v2"
        for row in gold_label_rows
    ):
        errors.append("one or more experiment gold-label rows use an invalid schema")
    if any(
        row.get("contains_sensitive_text") is not True
        for row in gold_label_rows
    ):
        errors.append(
            "one or more experiment gold-label rows lack the sensitive-text marker"
        )

    available_derived = [
        variant
        for variant in [
            "no_answer_validation_or_repair",
            "no_retrieval_retry",
        ]
        if any(
            (path / "jobs" / "generation" / variant).glob("*.json")
        )
    ]
    baseline_methods = list(
        manifest.get("baseline_methods")
        or dict(manifest.get("config") or {}).get("baseline_methods")
        or []
    )
    if set(attachment_methods) != set(baseline_methods):
        errors.append(
            "external baseline attachment methods do not match baseline_methods"
        )
    config_hash = str(manifest.get("config_hash") or "")
    invalid_job_examples: list[str] = []
    expected_provider_jobs: dict[str, tuple[str, str, str]] = {}
    duplicate_provider_job_uids: list[str] = []
    for stage, expected_schema in [
        ("generation", "answer_generation_job_v1"),
        ("judge", "independent_judgment_v1"),
    ]:
        for job_path in sorted((path / "jobs" / stage).glob("*/*.json")):
            try:
                payload = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                invalid_job_examples.append(
                    f"{job_path.relative_to(path)}:unreadable"
                )
                continue
            if payload.get("status") != "complete":
                continue
            if (
                payload.get("schema_version") != expected_schema
                or payload.get("experiment_config_hash") != config_hash
            ):
                invalid_job_examples.append(
                    f"{job_path.relative_to(path)}:schema_or_hash_mismatch"
                )
                continue
            provider_backed = stage == "judge" or (
                payload.get("generation_source")
                in {"counterfactual_rerun", "verified_parent_experiment"}
            )
            if not provider_backed:
                continue
            call_item_uid = str(payload.get("call_item_uid") or "")
            if not call_item_uid:
                invalid_job_examples.append(
                    f"{job_path.relative_to(path)}:missing_call_item_uid"
                )
                continue
            expected_provider_job = (
                "independent_judge" if stage == "judge" else "backbone",
                str(payload.get("variant") or ""),
                str(payload.get("query_task_id") or ""),
            )
            if call_item_uid in expected_provider_jobs:
                duplicate_provider_job_uids.append(call_item_uid)
            else:
                expected_provider_jobs[call_item_uid] = expected_provider_job
    if invalid_job_examples:
        errors.append(
            "completed job schema/hash validation failed: "
            f"{invalid_job_examples[:5]}"
        )
    if duplicate_provider_job_uids:
        errors.append(
            "multiple completed jobs share provider call_item_uid values: "
            f"{duplicate_provider_job_uids[:5]}"
        )
    core_variants = list(
        dict.fromkeys(
            [
                *list(
                    dict(manifest.get("config") or {}).get("variants") or []
                ),
                "observed_full_pipeline",
                *baseline_methods,
            ]
        )
    )
    expected_answer_keys = {
        (variant, query_task_id)
        for variant in core_variants
        for query_task_id in selected_query_set
    }
    for variant in available_derived:
        for job_path in sorted(
            (path / "jobs" / "generation" / variant).glob("*.json")
        ):
            try:
                payload = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            query_task_id = str(payload.get("query_task_id") or "")
            if (
                payload.get("status") == "complete"
                and payload.get("schema_version") == "answer_generation_job_v1"
                and payload.get("experiment_config_hash") == config_hash
                and payload.get("variant") == variant
                and query_task_id in selected_query_set
            ):
                expected_answer_keys.add((variant, query_task_id))
    answer_rows = _read_jsonl(path / "answer_ablation_rows.jsonl")
    judgment_rows = _read_jsonl(path / "independent_judgment_rows.jsonl")
    answer_keys = [
        (str(row.get("variant") or ""), str(row.get("query_task_id") or ""))
        for row in answer_rows
    ]
    judgment_keys = [
        (str(row.get("variant") or ""), str(row.get("query_task_id") or ""))
        for row in judgment_rows
    ]
    for label, keys in [
        ("answer", answer_keys),
        ("independent judgment", judgment_keys),
    ]:
        duplicate_keys = [key for key, count in Counter(keys).items() if count > 1]
        missing_keys = sorted(expected_answer_keys - set(keys))
        unexpected_keys = sorted(set(keys) - expected_answer_keys)
        if duplicate_keys:
            errors.append(f"duplicate {label} rows: {duplicate_keys[:5]}")
        if missing_keys:
            errors.append(f"missing {label} rows: {missing_keys[:5]}")
        if unexpected_keys:
            errors.append(f"unexpected {label} rows: {unexpected_keys[:5]}")
    if any(row.get("prompt_contains_variant_name") for row in judgment_rows):
        errors.append("one or more independent judge prompts leaked a variant name")
    if any(
        row.get("prompt_hash_matches_blinded_template") is not True
        for row in judgment_rows
    ):
        errors.append(
            "one or more independent judge prompt hashes do not match the blinded template"
        )
    invalid_verdicts = [
        f"{row.get('variant')}/{row.get('query_task_id')}"
        for row in judgment_rows
        if str(row.get("verdict") or "") not in {"correct", "partial", "incorrect"}
    ]
    if invalid_verdicts:
        errors.append(f"invalid independent judge verdicts: {invalid_verdicts[:5]}")

    provider_rows = _read_jsonl(path / "provider_call_rows.jsonl")
    item_uids = [
        str(row.get("call_item_uid") or "")
        for row in provider_rows
        if row.get("call_item_uid")
    ]
    missing_item_uid_count = sum(
        1 for row in provider_rows if not str(row.get("call_item_uid") or "")
    )
    if missing_item_uid_count:
        errors.append(f"{missing_item_uid_count} provider call rows lack call_item_uid")
    duplicates = [uid for uid, count in Counter(item_uids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate call_item_uid values: {duplicates[:5]}")
    provider_by_item_uid = {
        str(row.get("call_item_uid") or ""): row
        for row in provider_rows
        if str(row.get("call_item_uid") or "")
    }
    missing_provider_job_rows = sorted(
        set(expected_provider_jobs) - set(provider_by_item_uid)
    )
    if missing_provider_job_rows:
        errors.append(
            f"{len(missing_provider_job_rows)} completed provider-backed jobs "
            "lack a matching provider ledger row"
        )
    provider_job_scope_mismatches = [
        call_item_uid
        for call_item_uid, (role, variant, query_task_id) in (
            expected_provider_jobs.items()
        )
        if call_item_uid in provider_by_item_uid
        and (
            str(provider_by_item_uid[call_item_uid].get("role") or ""),
            str(provider_by_item_uid[call_item_uid].get("variant") or ""),
            str(provider_by_item_uid[call_item_uid].get("query_task_id") or ""),
        )
        != (role, variant, query_task_id)
    ]
    if provider_job_scope_mismatches:
        errors.append(
            "provider ledger role/variant/query mismatch for completed jobs: "
            f"{provider_job_scope_mismatches[:5]}"
        )
    max_provider_calls = int(manifest.get("max_provider_calls") or 0)
    provider_attempt_keys = {
        str(row.get("provider_call_uid") or row.get("call_item_uid") or "")
        for row in provider_rows
        if row.get("provider_call_uid") or row.get("call_item_uid")
    }
    provider_attempt_count = len(provider_attempt_keys) + sum(
        1
        for row in provider_rows
        if not row.get("provider_call_uid") and not row.get("call_item_uid")
    )
    if max_provider_calls and provider_attempt_count > max_provider_calls:
        errors.append(
            f"provider call count {provider_attempt_count} exceeds the recorded "
            f"maximum {max_provider_calls}"
        )
    retrieval_rows = _read_jsonl(path / "retrieval_plan_rows.jsonl")
    retrieval_query_ids = [
        str(row.get("query_task_id") or "") for row in retrieval_rows
    ]
    if set(retrieval_query_ids) != selected_query_set:
        missing = sorted(selected_query_set - set(retrieval_query_ids))
        unexpected = sorted(set(retrieval_query_ids) - selected_query_set)
        errors.append(
            "retrieval-plan query coverage mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if len(retrieval_query_ids) != len(set(retrieval_query_ids)):
        errors.append("retrieval-plan rows contain duplicate query ids")

    context_rows = _read_jsonl(path / "variant_context_rows.jsonl")
    context_keys = [
        (str(row.get("variant") or ""), str(row.get("query_task_id") or ""))
        for row in context_rows
    ]
    expected_context_keys: set[tuple[str, str]] = {
        (str(variant), query_task_id)
        for variant in list(dict(manifest.get("config") or {}).get("variants") or [])
        for query_task_id in selected_query_set
    }
    if len(context_keys) != len(set(context_keys)):
        errors.append("variant-context rows contain duplicate variant/query keys")
    if set(context_keys) != expected_context_keys:
        missing_context_keys = sorted(expected_context_keys - set(context_keys))
        unexpected_context_keys = sorted(set(context_keys) - expected_context_keys)
        errors.append(
            "variant-context coverage mismatch: "
            f"missing={missing_context_keys[:5]}, "
            f"unexpected={unexpected_context_keys[:5]}"
        )
    sample_by_query = {
        str(row.get("query_task_id") or ""): str(row.get("sample_id") or "")
        for row in selected_rows
    }
    mismatch_examples: list[str] = []
    for row in [*context_rows, *answer_rows, *judgment_rows]:
        query_task_id = str(row.get("query_task_id") or "")
        sample_id = str(row.get("sample_id") or "")
        expected_sample_id = sample_by_query.get(query_task_id)
        if expected_sample_id is not None and sample_id != expected_sample_id:
            mismatch_examples.append(f"{sample_id}/{query_task_id}")
            if len(mismatch_examples) == 5:
                break
    if mismatch_examples:
        errors.append(f"experiment sample/query mismatches: {mismatch_examples}")

    provider_scope_mismatches: list[str] = []
    for row in provider_rows:
        query_task_id = str(row.get("query_task_id") or "")
        if not query_task_id:
            continue
        expected_sample_id = sample_by_query.get(query_task_id)
        sample_id = str(row.get("sample_id") or "")
        if expected_sample_id is None or sample_id != expected_sample_id:
            provider_scope_mismatches.append(
                f"{sample_id}/{query_task_id}"
            )
            if len(provider_scope_mismatches) == 5:
                break
    if provider_scope_mismatches:
        errors.append(
            "provider call sample/query mismatches: "
            f"{provider_scope_mismatches}"
        )

    for row in context_rows:
        max_tokens = int(
            row.get("max_prompt_tokens")
            or dict(manifest.get("config") or {}).get("max_total_tokens")
            or 0
        )
        reserve = int(row.get("output_token_reserve") or 0)
        safety_margin = int(
            row.get("token_safety_margin")
            or dict(manifest.get("config") or {}).get("token_safety_margin")
            or 0
        )
        estimated = int(row.get("estimated_prompt_tokens") or 0)
        if max_tokens and estimated + reserve + safety_margin > max_tokens:
            errors.append(
                f"context budget exceeded for {row.get('variant')}/{row.get('query_task_id')}"
            )
        if (
            dict(manifest.get("config") or {}).get("require_exact_token_counter")
            and row.get("token_counter_exact") is not True
        ):
            errors.append(
                f"non-exact token counter for {row.get('variant')}/"
                f"{row.get('query_task_id')}"
            )
    if any(row.get("budget_passed_after_call") is False for row in answer_rows):
        errors.append("one or more provider-reported prompts exceeded the token budget")
    if any(
        row.get("matched_budget_passed_after_call") is False
        for row in answer_rows
    ):
        errors.append(
            "one or more matched baselines exceeded the provider-reported "
            "Full TrajWiki prompt"
        )
    matched_context_rows = [
        row
        for row in context_rows
        if row.get("budget_mode") == "full_prompt_matched_v1"
    ]
    if any(
        int(row.get("estimated_prompt_tokens") or 0)
        > int(row.get("reference_prompt_tokens") or -1)
        for row in matched_context_rows
    ):
        errors.append(
            "one or more matched contexts exceed the locally counted Full "
            "TrajWiki prompt"
        )
    if sampling_profile == "rebuttal_200_v1":
        unavailable_matched_usage = [
            f"{row.get('variant')}/{row.get('query_task_id')}"
            for row in answer_rows
            if row.get("reference_prompt_tokens") is not None
            and row.get("matched_budget_passed_after_call") is None
        ]
        if unavailable_matched_usage:
            errors.append(
                "provider prompt usage is unavailable for matched comparisons: "
                f"{unavailable_matched_usage[:5]}"
            )
    missing_actual_prompt_usage = [
        f"{row.get('variant')}/{row.get('query_task_id')}"
        for row in answer_rows
        if row.get("generation_source") == "counterfactual_rerun"
        and row.get("prompt_tokens") is None
    ]
    if missing_actual_prompt_usage:
        errors.append(
            "provider prompt-token usage is unavailable for regenerated answers: "
            f"{missing_actual_prompt_usage[:5]}"
        )
    unavailable_stage_rows = _read_jsonl(path / "unavailable_stage_rows.jsonl")
    if sampling_profile == "rebuttal_200_v1":
        if len(expected_provider_jobs) != 4800:
            errors.append(
                "rebuttal_200_v1 has "
                f"{len(expected_provider_jobs)} provider-backed completed jobs, "
                "expected 4800"
            )
        if len(context_rows) != 2200:
            errors.append(
                f"rebuttal_200_v1 has {len(context_rows)} context rows, expected 2200"
            )
        if len(answer_rows) != 2600:
            errors.append(
                f"rebuttal_200_v1 has {len(answer_rows)} answer rows, expected 2600"
            )
        if len(judgment_rows) != 2600:
            errors.append(
                "rebuttal_200_v1 has "
                f"{len(judgment_rows)} judgment rows, expected 2600"
            )
        if len(unavailable_stage_rows) != 400:
            errors.append(
                "rebuttal_200_v1 has "
                f"{len(unavailable_stage_rows)} unavailable-stage rows, expected 400"
            )
    report = {
        "schema_version": "artifact_integrity_v1",
        "artifact_type": "answer_ablation_experiment",
        "path": str(path),
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
        "provider_call_row_count": len(provider_rows),
        "provider_attempt_count": provider_attempt_count,
        "provider_backed_completed_job_count": len(expected_provider_jobs),
        "missing_provider_job_row_count": len(missing_provider_job_rows),
        "provider_job_scope_mismatch_count": len(
            provider_job_scope_mismatches
        ),
        "missing_call_item_uid_count": missing_item_uid_count,
        "selected_query_count": len(selected_query_set),
        "gold_label_row_count": len(gold_label_rows),
        "expected_answer_row_count": len(expected_answer_keys),
        "answer_row_count": len(answer_rows),
        "independent_judgment_row_count": len(judgment_rows),
        "variant_context_row_count": len(context_rows),
    }
    write_json(path / "validation_report.json", report)
    progress.advance(
        succeeded=not errors,
        error=errors[0] if errors else None,
    )
    progress.finish_stage(
        status="complete" if not errors else "complete_with_errors"
    )
    return report


def _validate_run(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    _, sample_rows, database_path = load_details_rows(path)
    memory_index = build_memory_index(database_path)
    query_samples = {
        str(row.get("query_task_id") or ""): str(row.get("sample_id") or "")
        for row in sample_rows
    }
    checked_rows = 0
    for filename, refs_field, ids_field, current_schema in [
        (
            "gold_labels.jsonl",
            "gold_source_refs",
            "gold_source_message_ids",
            "gold_labels_v2",
        ),
        (
            "audit_packet_rows.jsonl",
            "support_refs",
            "source_message_ids",
            "audit_packet_v2",
        ),
    ]:
        v2_path = path / "analysis_v2" / filename
        artifact_path = v2_path if v2_path.exists() else path / "analysis" / filename
        rows = _read_jsonl(artifact_path)
        schemas = {str(row.get("schema_version") or "") for row in rows}
        if rows and current_schema not in schemas:
            warnings.append(
                f"{artifact_path.relative_to(path)} uses legacy schemas {sorted(schemas)}; "
                "rerun the corresponding offline analysis to create sample-scoped v2 rows."
            )
        foreign_examples: list[str] = []
        foreign_count = 0
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            refs = list(row.get(refs_field) or [])
            recorded_ids = list(row.get(ids_field) or [])
            if recorded_ids:
                expected_ids = source_message_ids_for_refs(
                    memory_index,
                    sample_id=sample_id,
                    source_refs=refs,
                )
                foreign_ids = sorted(set(recorded_ids) - set(expected_ids))
                if foreign_ids:
                    foreign_count += 1
                    if len(foreign_examples) < 5:
                        foreign_examples.append(f"{sample_id}:{foreign_ids[:3]}")
            checked_rows += 1
        if foreign_count:
            errors.append(
                f"{artifact_path.relative_to(path)} has {foreign_count} rows with "
                f"foreign source ids; examples={foreign_examples}"
            )

    cost_rows = _read_jsonl(path / "analysis" / "cost_call_rows.jsonl")
    cost_schemas = {str(row.get("schema_version") or "") for row in cost_rows}
    if cost_rows and "cost_call_v2" not in cost_schemas:
        warnings.append(
            f"analysis/cost_call_rows.jsonl uses legacy schemas {sorted(cost_schemas)}; "
            "sample/query attribution and call uniqueness may be unobservable."
        )
    item_uids = [
        str(row.get("call_item_uid") or "")
        for row in cost_rows
        if row.get("call_item_uid")
    ]
    duplicates = [uid for uid, count in Counter(item_uids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate cost call_item_uid values: {duplicates[:5]}")
    if cost_rows and not item_uids:
        warnings.append(
            "analysis/cost_call_rows.jsonl has no call_item_uid values; uniqueness "
            "cannot be verified for this legacy run."
        )
    missing_sample_count = 0
    mismatch_examples: list[str] = []
    for row in cost_rows:
        query_task_id = str(row.get("query_task_id") or "")
        sample_id = str(row.get("sample_id") or "")
        if query_task_id and not sample_id:
            missing_sample_count += 1
        elif (
            query_task_id
            and query_task_id in query_samples
            and query_samples[query_task_id] != sample_id
            and len(mismatch_examples) < 5
        ):
            mismatch_examples.append(f"{sample_id}/{query_task_id}")
    if missing_sample_count:
        warnings.append(
            f"{missing_sample_count} cost rows have a query_task_id but no sample_id; "
            "legacy rows are retained but excluded from strict attribution checks."
        )
    if mismatch_examples:
        errors.append(f"cost row sample/query mismatches: {mismatch_examples}")
    report = {
        "schema_version": "artifact_integrity_v1",
        "artifact_type": "benchmark_run",
        "path": str(path),
        "database_path": str(database_path),
        "query_count": len(sample_rows),
        "checked_provenance_row_count": checked_rows,
        "cost_call_row_count": len(cost_rows),
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    output_dir = path / "analysis_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "integrity_report.json", report)
    return report


def validate_run_artifacts(path: Path | str) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if (target / "experiment_manifest.json").exists():
        return _validate_experiment(target)
    if (target / "details.json").exists():
        return _validate_run(target)
    raise FileNotFoundError(
        f"{target} is neither a benchmark run nor an answer-ablation experiment."
    )
