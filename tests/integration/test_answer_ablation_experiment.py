from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
from collections import Counter
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from trajpatch.analysis.auditability import analyze_auditability
from trajpatch.analysis.cost_benefit import analyze_cost_benefit
from trajpatch.analysis.failure_attribution import analyze_locomo_run_failures
from trajpatch.analysis.offline_ablation import analyze_offline_ablation
from trajpatch.config import RunConfig
from trajpatch.experiments.answer_ablation import (
    import_baseline_answers,
    run_answer_ablation,
)
from trajpatch.experiments.audit_study import (
    analyze_audit_study,
    conduct_audit_study,
    prepare_audit_study,
)
from trajpatch.experiments.validation import validate_run_artifacts
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider

PILOT_VARIANTS = (
    "full,direct_trajectory,latest_snapshot,hybrid_raw_rag,wiki_summaries,"
    "no_claim_state,no_source_constraint,full_context,naive_dense_rag"
)
EXTENDED_VARIANTS = (
    PILOT_VARIANTS
    + ",full_context_matched,hybrid_raw_rag_matched"
)


def _locomo_file(path: Path) -> Path:
    rows = []
    for index in range(3):
        rows.append(
            {
                "sample_id": f"conv-ablation-{index}",
                "qa_uid": f"conv-ablation-{index}_qa_0",
                "category": 1,
                "category_name": "multi_hop",
                "question": f"What does Person {index} prefer?",
                "answer": f"Tea {index}",
                "evidence": ["D1:1"],
                "full_conversation": (
                    "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                    f"[D1:1] Person {index}: I prefer Tea {index}.\n"
                    f"[D1:2] Friend {index}: Noted.\n"
                ),
            }
        )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_answer_ablation_is_resumable_and_blinded(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    config = RunConfig(
        dataset="locomo",
        dataset_path=_locomo_file(tmp_path / "locomo.jsonl"),
        output_dir=output_dir,
        index_database_path=output_dir / "trajpatch_index.sqlite",
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        memory_cache_dir=tmp_path / "cache",
        m=2,
        t_pages=2,
        k=1,
        max_samples=3,
        ablation_diagnostics=True,
        retrieval_rank_save_mode="full",
        cost_diagnostics=True,
        cost_call_save_mode="compact",
        auditability_diagnostics=True,
        audit_packet_save_mode="compact",
    )
    benchmark = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()
    run_dir = Path(benchmark.details["run_meta"]["run_dir"])
    with pytest.raises(ValueError, match="context_save_mode"):
        run_answer_ablation(
            run_dir,
            sample_size=3,
            backbone_provider_kind="mock",
            backbone_model="mock-backbone",
            independent_judge_provider_kind="mock",
            independent_judge_model="mock-independent-judge",
            context_save_mode="invalid",
        )
    with pytest.raises(ValueError, match="No provider calls were made"):
        run_answer_ablation(
            run_dir,
            variants="full",
            sample_size=3,
            sample_seed=13,
            backbone_provider_kind="mock",
            backbone_model="mock-backbone",
            independent_judge_provider_kind="mock",
            independent_judge_model="mock-independent-judge",
            max_provider_calls=1,
        )
    blocked_manifests = list(
        (run_dir / "rebuttal_experiments").glob(
            "answer_ablation_*/experiment_manifest.json"
        )
    )
    blocked_manifest = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in blocked_manifests
        if json.loads(path.read_text(encoding="utf-8"))["config"]["sample_seed"] == 13
    )
    assert blocked_manifest["status"] == "blocked_by_call_cap"
    assert blocked_manifest["planned_provider_call_count"] > 1
    raw_baselines = tmp_path / "baseline_answers.jsonl"
    raw_baselines.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"conv-ablation-{index}",
                    "query_task_id": f"conv-ablation-{index}_qa_0",
                    "question": f"What does Person {index} prefer?",
                    "gold_answer": f"Tea {index}",
                    "method": "saved_baseline",
                    "answer_text": f"Tea {index}",
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    normalized_baselines = tmp_path / "baseline_answers.normalized.jsonl"
    imported = import_baseline_answers(
        raw_baselines,
        output_path=normalized_baselines,
        run_path=run_dir,
    )
    assert imported["alignment"]["aligned"] is True
    incomplete_multi_method = tmp_path / "incomplete_multi_method.jsonl"
    incomplete_multi_method.write_text(
        "\n".join(
            [
                *[
                    json.dumps(
                        {
                            "sample_id": f"conv-ablation-{index}",
                            "query_task_id": f"conv-ablation-{index}_qa_0",
                            "question": f"What does Person {index} prefer?",
                            "gold_answer": f"Tea {index}",
                            "method": "complete_method",
                            "answer_text": f"Tea {index}",
                        }
                    )
                    for index in range(3)
                ],
                *[
                    json.dumps(
                        {
                            "sample_id": f"conv-ablation-{index}",
                            "query_task_id": f"conv-ablation-{index}_qa_0",
                            "question": f"What does Person {index} prefer?",
                            "gold_answer": f"Tea {index}",
                            "method": "incomplete_method",
                            "answer_text": f"Tea {index}",
                        }
                    )
                    for index in range(2)
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete_methods"):
        import_baseline_answers(
            incomplete_multi_method,
            output_path=tmp_path / "must_not_import.jsonl",
            run_path=run_dir,
        )
    missing_provenance = tmp_path / "missing_provenance.jsonl"
    missing_provenance.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"conv-ablation-{index}",
                    "query_task_id": f"conv-ablation-{index}_qa_0",
                    "method": "unverified_method",
                    "answer_text": f"Tea {index}",
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatches=6"):
        import_baseline_answers(
            missing_provenance,
            output_path=tmp_path / "must_not_import_unverified.jsonl",
            run_path=run_dir,
        )

    dry_run = run_answer_ablation(
        run_dir,
        variants=PILOT_VARIANTS,
        sample_size=3,
        backbone_provider_kind="mock",
        backbone_model="mock-backbone",
        independent_judge_provider_kind="mock",
        independent_judge_model="mock-independent-judge",
        generation_max_concurrency=2,
        judge_max_concurrency=2,
        max_provider_calls=200,
        dry_run=True,
    )
    assert dry_run["provider_call_count"] == 63
    assert dry_run["generation_call_count"] == 27
    assert dry_run["judge_call_count"] == 36

    first = run_answer_ablation(
        run_dir,
        variants=PILOT_VARIANTS,
        baseline_answers_path=normalized_baselines,
        sample_size=3,
        backbone_provider_kind="mock",
        backbone_model="mock-backbone",
        independent_judge_provider_kind="mock",
        independent_judge_model="mock-independent-judge",
        generation_max_concurrency=2,
        judge_max_concurrency=2,
        max_provider_calls=200,
    )
    experiment_dir = Path(first["experiment_dir"])
    provider_path = experiment_dir / "provider_call_rows.jsonl"
    first_provider_rows = provider_path.read_text(encoding="utf-8")
    integrity = json.loads(
        (experiment_dir / "integrity_report.json").read_text(encoding="utf-8")
    )

    assert first["query_count"] == 3
    assert first["variant_count"] == 13
    assert integrity["error_count"] == 0
    assert integrity["independent_judge_prompt_blinded"] is True
    assert (experiment_dir / "answer_ablation_table.csv").exists()
    assert (experiment_dir / "paired_statistics.csv").exists()
    assert (experiment_dir / "stage_diagnostic_statistics.csv").exists()
    assert (experiment_dir / "stratum_summary.csv").exists()
    assert (experiment_dir / "token_budget_audit.csv").exists()
    assert (experiment_dir / "gold_label_rows.jsonl").exists()
    assert (experiment_dir / "external_baseline_manifest.json").exists()
    assert (experiment_dir / "balanced_failure_examples.jsonl").exists()
    gold_label_rows = [
        json.loads(line)
        for line in (experiment_dir / "gold_label_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert all(row["contains_sensitive_text"] is True for row in gold_label_rows)
    with (experiment_dir / "variant_cost_summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        cost_rows = list(csv.DictReader(handle))
    historical_cost = next(
        row
        for row in cost_rows
        if row["variant"] == "saved_baseline"
        and row["role"] == "historical_generation"
    )
    assert historical_cost["prompt_tokens"] == ""
    assert historical_cost["completion_tokens"] == ""
    assert historical_cost["total_tokens"] == ""
    assert historical_cost["latency_ms"] == ""
    assert historical_cost["latency_missing_count"] == "3"
    assert historical_cost["usage_complete"] == "False"
    answer_rows = [
        json.loads(line)
        for line in (experiment_dir / "answer_ablation_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    baseline_rows = [
        row for row in answer_rows if row.get("variant") == "saved_baseline"
    ]
    wiki_summary_rows = [
        row for row in answer_rows if row.get("variant") == "wiki_summaries"
    ]
    stage_rows = [
        row
        for row in answer_rows
        if row.get("variant")
        in {"no_answer_validation_or_repair", "no_retrieval_retry"}
    ]
    assert len(baseline_rows) == 3
    assert all(row.get("source_supported_proxy") is None for row in baseline_rows)
    assert all(row.get("gold_ref_coverage") is None for row in baseline_rows)
    assert all(
        row.get("source_validation_observable") is False
        and row.get("would_pass_source_validation") is None
        for row in wiki_summary_rows
    )
    assert all(row.get("gold_ref_coverage") is None for row in wiki_summary_rows)
    assert stage_rows and all(row.get("stage_observable") is True for row in stage_rows)
    assert all(
        row.get("generation_source") == "counterfactual_rerun"
        for row in answer_rows
        if row.get("variant") == "full"
    )
    assert all(
        row.get("generation_source") == "observed_full_run"
        for row in answer_rows
        if row.get("variant") == "observed_full_pipeline"
    )
    extension_dry_run = run_answer_ablation(
        run_dir,
        variants=EXTENDED_VARIANTS,
        baseline_answers_path=normalized_baselines,
        sample_size=3,
        backbone_provider_kind="mock",
        backbone_model="mock-backbone",
        independent_judge_provider_kind="mock",
        independent_judge_model="mock-independent-judge",
        generation_max_concurrency=2,
        judge_max_concurrency=2,
        max_provider_calls=200,
        reuse_experiment_path=experiment_dir,
        reuse_policy="require",
        dry_run=True,
    )
    assert extension_dry_run["reused_generation_job_count"] == 27
    assert extension_dry_run["reused_judgment_job_count"] == 33
    assert extension_dry_run["generation_call_count"] == 6
    assert extension_dry_run["judge_call_count"] == 12
    extension = run_answer_ablation(
        run_dir,
        variants=EXTENDED_VARIANTS,
        baseline_answers_path=normalized_baselines,
        sample_size=3,
        backbone_provider_kind="mock",
        backbone_model="mock-backbone",
        independent_judge_provider_kind="mock",
        independent_judge_model="mock-independent-judge",
        generation_max_concurrency=2,
        judge_max_concurrency=2,
        max_provider_calls=200,
        reuse_experiment_path=experiment_dir,
        reuse_policy="require",
    )
    extension_dir = Path(extension["experiment_dir"])
    extension_contexts = [
        json.loads(line)
        for line in (extension_dir / "variant_context_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    extension_by_key = {
        (str(row["variant"]), str(row["query_task_id"])): row
        for row in extension_contexts
    }
    for query_task_id in {
        str(row["query_task_id"]) for row in extension_contexts
    }:
        full = extension_by_key[("full", query_task_id)]
        for matched_variant in [
            "full_context_matched",
            "hybrid_raw_rag_matched",
        ]:
            matched = extension_by_key[(matched_variant, query_task_id)]
            assert (
                int(matched["estimated_prompt_tokens"])
                <= int(full["estimated_prompt_tokens"])
            )
    provider_origins = Counter(
        str(json.loads(line).get("call_origin") or "")
        for line in (extension_dir / "provider_call_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    assert provider_origins["parent_experiment"] == 60
    assert provider_origins["new"] == 18
    assert validate_run_artifacts(extension_dir)["error_count"] == 0
    extension_provider_path = extension_dir / "provider_call_rows.jsonl"
    extension_provider_text = extension_provider_path.read_text(encoding="utf-8")
    extension_provider_lines = [
        line for line in extension_provider_text.splitlines() if line.strip()
    ]
    extension_provider_path.write_text(
        "\n".join(extension_provider_lines[1:]) + "\n",
        encoding="utf-8",
    )
    corrupted_validation = validate_run_artifacts(extension_dir)
    assert corrupted_validation["missing_provider_job_row_count"] == 1
    assert corrupted_validation["error_count"] > 0
    extension_provider_path.write_text(
        extension_provider_text,
        encoding="utf-8",
    )
    assert validate_run_artifacts(extension_dir)["error_count"] == 0
    with pytest.raises(ValueError, match="reuse verification failed"):
        run_answer_ablation(
            run_dir,
            variants=EXTENDED_VARIANTS,
            baseline_answers_path=normalized_baselines,
            sample_size=3,
            backbone_provider_kind="mock",
            backbone_model="mock-backbone",
            independent_judge_provider_kind="mock",
            independent_judge_model="mock-independent-judge",
            generation_seed=8,
            max_provider_calls=200,
            reuse_experiment_path=experiment_dir,
            reuse_policy="require",
            dry_run=True,
        )
    corrupt_parent_dir = tmp_path / "corrupt_parent_experiment"
    shutil.copytree(experiment_dir, corrupt_parent_dir)
    corrupt_provider_path = corrupt_parent_dir / "provider_call_rows.jsonl"
    corrupt_provider_rows = [
        json.loads(line)
        for line in corrupt_provider_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corrupt_provider_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in corrupt_provider_rows[1:]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provider ledger"):
        run_answer_ablation(
            run_dir,
            variants=EXTENDED_VARIANTS,
            baseline_answers_path=normalized_baselines,
            sample_size=3,
            sample_seed=19,
            backbone_provider_kind="mock",
            backbone_model="mock-backbone",
            independent_judge_provider_kind="mock",
            independent_judge_model="mock-independent-judge",
            max_provider_calls=200,
            reuse_experiment_path=corrupt_parent_dir,
            reuse_policy="require",
            dry_run=True,
        )
    sampling_manifest_path = experiment_dir / "sampling_manifest.json"
    scoped_ablation = analyze_offline_ablation(
        run_dir,
        variants="full,no_wiki_direct",
        budgets="4000",
        rank_cutoffs="5",
        sampling_manifest_path=sampling_manifest_path,
    )
    scoped_analysis_dir = experiment_dir / "offline_analysis"
    assert scoped_ablation["query_count"] == 3
    assert Path(scoped_ablation["analysis_dir"]) == scoped_analysis_dir
    scoped_cost = analyze_cost_benefit(
        run_dir,
        baselines="trajwiki_observed,no_wiki_direct",
        future_query_counts="1,2",
        sampling_manifest_path=sampling_manifest_path,
    )
    scoped_auditability = analyze_auditability(
        run_dir,
        baselines="trajwiki_observed,no_wiki_direct",
        sampling_manifest_path=sampling_manifest_path,
    )
    scoped_failures = analyze_locomo_run_failures(
        run_dir,
        top_examples_per_bucket=1,
        sampling_manifest_path=sampling_manifest_path,
    )
    assert scoped_cost["query_count"] == 3
    assert scoped_auditability["query_count"] == 3
    assert scoped_failures["totals"]["total_queries"] == 3
    assert scoped_failures["analysis_scope"]["mode"] == "sampling_manifest"
    assert (
        json.loads(
            (scoped_analysis_dir / "cost_benefit_summary.json").read_text(
                encoding="utf-8"
            )
        )["sampling_scope"]["scoped_query_count"]
        == 3
    )
    assert (
        json.loads(
            (scoped_analysis_dir / "auditability_summary.json").read_text(
                encoding="utf-8"
            )
        )["sampling_scope"]["scoped_query_count"]
        == 3
    )
    context_rows = [
        json.loads(line)
        for line in (experiment_dir / "variant_context_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    context_by_variant = {
        str(row["variant"]): row
        for row in context_rows
        if row.get("query_task_id") == "conv-ablation-0_qa_0"
    }
    assert all(row["contains_sensitive_text"] is True for row in context_rows)
    assert context_by_variant["direct_trajectory"]["selected_page_ids"] == []
    assert context_by_variant["wiki_summaries"]["selected_trajectory_ids"] == []
    assert context_by_variant["wiki_summaries"]["selected_snapshot_ids"] == []
    assert context_by_variant["wiki_summaries"]["selected_source_refs"] == []
    assert (
        context_by_variant["wiki_summaries"]["policy"][
            "source_support_constraint"
        ]
        is True
    )
    assert context_by_variant["full_context"]["selected_page_ids"] == []
    assert context_by_variant["full_context"]["budget_truncated"] is False
    assert (
        context_by_variant["naive_dense_rag"]["metadata"][
            "naive_rag_candidate_universe_size"
        ]
        >= 1
    )
    with gzip.open(
        context_by_variant["full"]["context_path"], "rt", encoding="utf-8"
    ) as handle:
        full_context = json.load(handle)["context_text"]
    with gzip.open(
        context_by_variant["no_source_constraint"]["context_path"],
        "rt",
        encoding="utf-8",
    ) as handle:
        no_source_constraint_context = json.load(handle)["context_text"]
    with gzip.open(
        context_by_variant["wiki_summaries"]["context_path"],
        "rt",
        encoding="utf-8",
    ) as handle:
        wiki_payload = json.load(handle)
        wiki_context = wiki_payload["context_text"]
    assert wiki_payload["contains_sensitive_text"] is True
    with gzip.open(
        context_by_variant["no_claim_state"]["context_path"],
        "rt",
        encoding="utf-8",
    ) as handle:
        no_claim_state_context = json.load(handle)["context_text"]
    assert "[WIKI PAGE" not in full_context
    assert no_source_constraint_context == full_context
    assert (
        context_by_variant["no_source_constraint"]["metadata"][
            "budget_source_support_constraint"
        ]
        is True
    )
    assert "[WIKI PAGE" in wiki_context
    assert "Claims (lifecycle state hidden):" in no_claim_state_context
    visible_full_refs = set(re.findall(r"\[SOURCE\s+([^\s\]]+)", full_context))
    assert set(context_by_variant["full"]["selected_source_refs"]) == visible_full_refs

    second = run_answer_ablation(
        run_dir,
        variants=PILOT_VARIANTS,
        baseline_answers_path=normalized_baselines,
        sample_size=3,
        backbone_provider_kind="mock",
        backbone_model="mock-backbone",
        independent_judge_provider_kind="mock",
        independent_judge_model="mock-independent-judge",
        generation_max_concurrency=2,
        judge_max_concurrency=2,
        max_provider_calls=200,
        resume=True,
    )
    assert second["experiment_dir"] == first["experiment_dir"]
    assert provider_path.read_text(encoding="utf-8") == first_provider_rows
    assert validate_run_artifacts(experiment_dir)["error_count"] == 0
    original_call_uids = {
        row["call_item_uid"]
        for row in (
            json.loads(line)
            for line in first_provider_rows.splitlines()
            if line.strip()
        )
    }
    provider_path.unlink()
    third = run_answer_ablation(
        run_dir,
        variants=PILOT_VARIANTS,
        baseline_answers_path=normalized_baselines,
        sample_size=3,
        backbone_provider_kind="mock",
        backbone_model="mock-backbone",
        independent_judge_provider_kind="mock",
        independent_judge_model="mock-independent-judge",
        generation_max_concurrency=2,
        judge_max_concurrency=2,
        max_provider_calls=200,
        resume=True,
    )
    recovered_rows = [
        json.loads(line)
        for line in provider_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["call_item_uid"] for row in recovered_rows} == original_call_uids
    assert validate_run_artifacts(Path(third["experiment_dir"]))["error_count"] == 0
    manifest = json.loads(
        (experiment_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    invocation_ids = [row["invocation_id"] for row in manifest["invocations"]]
    assert len(invocation_ids) == 4
    assert len(invocation_ids) == len(set(invocation_ids))

    full_job_path = (
        experiment_dir
        / "jobs"
        / "generation"
        / "full"
        / "conv-ablation-0_qa_0.json"
    )
    full_job_before_replacement = full_job_path.read_bytes()
    replacement_source = tmp_path / "replacement_baseline.jsonl"
    replacement_source.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"conv-ablation-{index}",
                    "query_task_id": f"conv-ablation-{index}_qa_0",
                    "question": f"What does Person {index} prefer?",
                    "gold_answer": f"Tea {index}",
                    "method": "saved_baseline",
                    "answer_text": f"Replacement Tea {index}",
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    replacement_normalized = tmp_path / "replacement_baseline.normalized.jsonl"
    import_baseline_answers(
        replacement_source,
        output_path=replacement_normalized,
        run_path=run_dir,
    )
    replaced = run_answer_ablation(
        run_dir,
        variants=PILOT_VARIANTS,
        baseline_answers_path=replacement_normalized,
        sample_size=3,
        backbone_provider_kind="mock",
        backbone_model="mock-backbone",
        independent_judge_provider_kind="mock",
        independent_judge_model="mock-independent-judge",
        generation_max_concurrency=2,
        judge_max_concurrency=2,
        max_provider_calls=200,
        resume=True,
    )
    assert replaced["experiment_dir"] == str(experiment_dir)
    assert full_job_path.read_bytes() == full_job_before_replacement
    replacement_manifest = json.loads(
        (experiment_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert replacement_manifest["generated_provider_call_count"] == 0
    assert replacement_manifest["judge_provider_call_count"] == 3
    replacement_rows = [
        json.loads(line)
        for line in (experiment_dir / "answer_ablation_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all(
        str(row["answer_text"]).startswith("Replacement Tea")
        for row in replacement_rows
        if row.get("variant") == "saved_baseline"
    )
    replacement_provider_rows = [
        json.loads(line)
        for line in provider_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert (
        sum(
            1
            for row in replacement_provider_rows
            if row.get("variant") == "saved_baseline"
            and row.get("superseded_external_attachment") is True
        )
        == 3
    )
    assert validate_run_artifacts(experiment_dir)["error_count"] == 0
    replacement_bytes = replacement_normalized.read_bytes()
    replacement_normalized.write_bytes(replacement_bytes + b"\n")
    tampered_baseline_report = validate_run_artifacts(experiment_dir)
    assert any(
        "external baseline attachment no longer matches" in error
        for error in tampered_baseline_report["errors"]
    )
    replacement_normalized.write_bytes(replacement_bytes)
    source_details_path = run_dir / "details.json"
    source_details_bytes = source_details_path.read_bytes()
    source_details_path.write_bytes(source_details_bytes + b"\n")
    tampered_source_report = validate_run_artifacts(experiment_dir)
    assert any(
        "source details artifact no longer matches" in error
        for error in tampered_source_report["errors"]
    )
    source_details_path.write_bytes(source_details_bytes)
    assert validate_run_artifacts(experiment_dir)["error_count"] == 0
    provider_bytes = provider_path.read_bytes()
    provider_rows_for_tamper = [
        json.loads(line)
        for line in provider_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    provider_rows_for_tamper[0]["sample_id"] = "foreign-sample"
    provider_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True)
            for row in provider_rows_for_tamper
        )
        + "\n",
        encoding="utf-8",
    )
    tampered_provider_scope_report = validate_run_artifacts(experiment_dir)
    assert any(
        "provider call sample/query mismatches" in error
        for error in tampered_provider_scope_report["errors"]
    )
    provider_path.write_bytes(provider_bytes)
    assert validate_run_artifacts(experiment_dir)["error_count"] == 0

    audit = prepare_audit_study(
        run_dir,
        case_count=3,
        seed=7,
        answer_experiment_path=experiment_dir,
    )
    task_rows = [
        json.loads(line)
        for line in Path(audit["tasks_path"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert task_rows
    assert all("gold_answer" not in row for row in task_rows)
    assert all("condition" not in row for row in task_rows)
    assert all("judge_verdict" not in row for row in task_rows)
    assert all("failure_localization_stage" not in row for row in task_rows)
    assert all(row["contains_sensitive_text"] is True for row in task_rows)
    assert audit["primary_first_exposure_task_count"] == 6
    assert Path(audit["audit_labels_template_path"]).exists()
    with Path(audit["audit_labels_template_path"]).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        adjudication_rows = list(csv.DictReader(handle))
    assert adjudication_rows
    assert all(row["question"] for row in adjudication_rows)
    assert all(row["candidate_answer"] for row in adjudication_rows)
    assert audit["answer_experiment_dir"] == str(experiment_dir)
    assert sum(audit["selection_category_counts"].values()) == 3
    for annotator_slot in [1, 2]:
        condition_counts = [
            count
            for key, count in audit["primary_condition_counts"].items()
            if key.startswith(f"annotator_{annotator_slot}:")
        ]
        assert sum(condition_counts) == 3
        assert max(condition_counts) - min(condition_counts) <= 1

    responses = iter(
        value for _ in range(3) for value in ["yes", "D1:1", "correct", "5", ""]
    )
    with pytest.raises(ValueError, match="annotator_slot"):
        conduct_audit_study(
            audit["tasks_path"],
            annotator_slot=3,
            annotator_id="annotator-a",
            output_path=Path(audit["study_dir"]) / "invalid.jsonl",
        )
    audit_result = conduct_audit_study(
        audit["tasks_path"],
        annotator_slot=1,
        annotator_id="annotator-a",
        output_path=Path(audit["study_dir"]) / "audit_results_1.jsonl",
        console=Console(file=StringIO(), force_terminal=False),
        input_fn=lambda _: next(responses),
    )
    assert audit_result["max_exposure"] == 1
    assert audit_result["target_task_count"] == 3
    assert audit_result["completed_task_count"] == 3
    with pytest.raises(ValueError, match="another annotator"):
        conduct_audit_study(
            audit["tasks_path"],
            annotator_slot=2,
            annotator_id="annotator-b",
            output_path=Path(audit_result["output_path"]),
        )

    audit_summary = analyze_audit_study(audit["study_dir"])
    assert audit_summary["accuracy_metrics_reported"] is False
    assert audit_summary["primary_first_exposure_result_count"] == 3
    assert audit_summary["expected_primary_first_exposure_result_count"] == 6
    assert audit_summary["primary_first_exposure_complete"] is False

    first_result_path = Path(audit_result["output_path"])
    same_annotator_path = (
        Path(audit["study_dir"]) / "audit_results_same_annotator.jsonl"
    )
    slot_two_task = next(
        row for row in task_rows if row["annotator_slot"] == 2
    )
    same_annotator_path.write_text(
        json.dumps(
            {
                "task_id": slot_two_task["task_id"],
                "annotator_slot": 2,
                "annotator_id": "annotator-a",
                "source_supported_decision": "uncertain",
                "failure_stage_decision": "uncertain",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="distinct annotator_id"):
        analyze_audit_study(audit["study_dir"])
    same_annotator_path.unlink()

    duplicate_path = Path(audit["study_dir"]) / "audit_results_duplicate.jsonl"
    duplicate_path.write_text(
        first_result_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate audit result"):
        analyze_audit_study(audit["study_dir"])
    duplicate_path.unlink()
    unknown_path = Path(audit["study_dir"]) / "audit_results_unknown.jsonl"
    unknown_path.write_text(
        json.dumps(
            {
                "task_id": "unknown-task",
                "annotator_slot": 1,
                "source_supported_decision": "uncertain",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown task_id"):
        analyze_audit_study(audit["study_dir"])
