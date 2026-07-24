from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from rich.console import Console

from trajpatch.analysis import analyze_auditability, analyze_offline_ablation
from trajpatch.analysis.failure_attribution import analyze_locomo_run_failures
from trajpatch.analysis.cost_benefit import analyze_cost_benefit
from trajpatch.config import RunConfig
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider


def _locomo_file(path: Path) -> Path:
    row = {
        "sample_id": "conv-audit",
        "qa_uid": "conv-audit_qa_0",
        "category": 1,
        "category_name": "multi_hop",
        "question": "What does Alice prefer?",
        "answer": "Tea",
        "evidence": ["D1:1"],
        "full_conversation": (
            "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
            "[D1:1] Alice: I love tea.\n"
            "[D1:2] Bob: Nice.\n"
        ),
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _multi_locomo_file(path: Path, *, sample_count: int = 5) -> Path:
    rows = []
    for index in range(sample_count):
        rows.append(
            {
                "sample_id": f"conv-audit-parallel-{index}",
                "qa_uid": f"conv-audit-parallel-{index}_qa_0",
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
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _medmt_root(root: Path) -> Path:
    root.mkdir()
    payload = [
        {
            "id": f"med-audit-{index}",
            "messages": [
                {"role": "system", "content": "Keep answers short."},
                {"role": "user", "content": f"History item {index}."},
                {"role": "assistant", "content": "Acknowledged."},
                {"role": "user", "content": f"What do you remember about item {index}?"},
            ],
            "test_point": f"Mention item {index}.",
            "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
            "meta": {"sence_type": {"type": "Consultation"}},
        }
        for index in range(3)
    ]
    (root / "long_context_memory_and_understanding.json").write_text(json.dumps(payload), encoding="utf-8")
    for file_name in [
        "resistance_to_contextual_interference.json",
        "information_contradiction.json",
    ]:
        (root / file_name).write_text("[]", encoding="utf-8")
    return root


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_auditability_artifacts_are_generated_from_mock_locomo_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    config = RunConfig(
        dataset="locomo",
        dataset_path=_locomo_file(tmp_path / "locomo.jsonl"),
        output_dir=output_dir,
        database_path=None,
        index_database_path=output_dir / "trajpatch_index.sqlite",
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        memory_cache_dir=tmp_path / "cache",
        m=2,
        t_pages=2,
        k=1,
        max_samples=1,
        ablation_diagnostics=True,
        retrieval_rank_save_mode="full",
        cost_diagnostics=True,
        cost_call_save_mode="compact",
        auditability_diagnostics=True,
        audit_packet_save_mode="compact",
    )
    report = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()
    run_dir = Path(report.details["run_meta"]["run_dir"])

    answer_support_path = run_dir / "analysis" / "answer_support_rows.jsonl"
    context_claim_path = run_dir / "analysis" / "answer_context_claim_rows.jsonl"
    lifecycle_path = run_dir / "analysis" / "claim_lifecycle_rows.jsonl"
    packet_path = run_dir / "analysis" / "audit_packet_rows.jsonl"
    assert answer_support_path.exists()
    assert context_claim_path.exists()
    assert lifecycle_path.exists()
    assert packet_path.exists()
    details = json.loads((run_dir / "details.json").read_text(encoding="utf-8"))
    assert "answer_support_rows" in details["paths"]
    support_rows = [
        json.loads(line) for line in answer_support_path.read_text(encoding="utf-8").splitlines() if line
    ]
    packet_rows = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines() if line]
    assert support_rows
    assert packet_rows[0]["schema_version"] == "audit_packet_v2"

    analyze_offline_ablation(
        run_dir,
        variants="full,no_wiki_direct,flat_raw,wiki_only",
        budgets="4000",
        rank_cutoffs="15",
    )
    generated = analyze_auditability(
        run_dir,
        baselines="trajwiki_observed,full_context_proxy,no_wiki_direct",
    )
    for key in [
        "summary_path",
        "source_support_table_path",
        "unsupported_answer_table_path",
        "failure_localization_table_path",
        "conflict_obsolete_table_path",
        "audit_packet_cost_path",
        "error_propagation_funnel_path",
    ]:
        assert Path(generated[key]).exists()

    with Path(generated["source_support_table_path"]).open(encoding="utf-8", newline="") as handle:
        source_support_rows = list(csv.DictReader(handle))
    assert {row["method"] for row in source_support_rows} >= {
        "trajwiki_observed",
        "full_context_proxy",
        "no_wiki_direct",
    }
    assert (run_dir / "analysis" / "audit_examples.jsonl").exists()


def test_parallel_locomo_diagnostics_do_not_duplicate_or_conflict(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    stream = io.StringIO()
    config = RunConfig(
        dataset="locomo",
        dataset_path=_multi_locomo_file(tmp_path / "parallel_locomo.jsonl", sample_count=5),
        output_dir=output_dir,
        database_path=None,
        index_database_path=output_dir / "trajpatch_index.sqlite",
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        memory_cache_dir=tmp_path / "cache",
        m=2,
        t_pages=2,
        k=1,
        max_samples=5,
        conv_workers=5,
        memory_extract_batch_size="auto",
        judge_max_concurrency=6,
        ablation_diagnostics=True,
        retrieval_rank_save_mode="full",
        cost_diagnostics=True,
        cost_call_save_mode="compact",
        auditability_diagnostics=True,
        audit_packet_save_mode="compact",
        verbose=True,
    )
    report = PipelineRunner(
        config,
        console=Console(file=stream, force_terminal=False, color_system=None, width=240),
    ).run()
    run_dir = Path(report.details["run_meta"]["run_dir"])

    assert report.details["run_meta"]["worker_mode"] == "sharded"
    assert len(report.details["run_meta"]["worker_shards"]) == 5
    assert "sample_sharding_done" in stream.getvalue()
    assert "shard_merge_done" in stream.getvalue()

    expected_queries = 5
    gold_rows = _jsonl_rows(run_dir / "analysis" / "gold_labels.jsonl")
    cost_query_rows = _jsonl_rows(run_dir / "analysis" / "cost_query_rows.jsonl")
    audit_packet_rows = _jsonl_rows(run_dir / "analysis" / "audit_packet_rows.jsonl")
    support_rows = _jsonl_rows(run_dir / "analysis" / "answer_support_rows.jsonl")
    assert len(gold_rows) == expected_queries
    assert len(cost_query_rows) == expected_queries
    assert len(audit_packet_rows) == expected_queries
    assert len({row["query_task_id"] for row in gold_rows}) == expected_queries
    assert len({row["query_task_id"] for row in cost_query_rows}) == expected_queries
    assert len({row["query_task_id"] for row in audit_packet_rows}) == expected_queries
    assert {row["query_task_id"] for row in support_rows} == {row["query_task_id"] for row in gold_rows}
    assert all(row["contains_sensitive_text"] is True for row in gold_rows)
    assert all(row["contains_sensitive_text"] is True for row in audit_packet_rows)
    assert all(row["contains_sensitive_text"] is True for row in support_rows)

    cost_call_rows = _jsonl_rows(run_dir / "analysis" / "cost_call_rows.jsonl")
    call_item_uids = [
        row["call_item_uid"]
        for row in cost_call_rows
        if str(row.get("call_item_uid") or "").strip()
    ]
    assert call_item_uids
    assert len(call_item_uids) == len(set(call_item_uids))
    expected_sample_by_query = {
        str(row["query_task_id"]): str(row["sample_id"]) for row in gold_rows
    }
    attributed_query_rows = [
        row for row in cost_call_rows if str(row.get("query_task_id") or "").strip()
    ]
    assert attributed_query_rows
    assert all(
        str(row.get("sample_id") or "")
        == expected_sample_by_query[str(row["query_task_id"])]
        for row in attributed_query_rows
    )
    reconciliation = json.loads(
        (run_dir / "analysis" / "cost_reconciliation.json").read_text(
            encoding="utf-8"
        )
    )
    assert reconciliation["reconciled"] is True

    analyze_offline_ablation(
        run_dir,
        variants="full,no_wiki_direct,flat_raw,wiki_only",
        budgets="4000",
        rank_cutoffs="15",
    )
    cost_report = analyze_cost_benefit(
        run_dir,
        baselines="trajwiki_observed,full_context_proxy,no_wiki_direct",
        future_query_counts="1,2",
    )
    packet_path = run_dir / "analysis" / "audit_packet_rows.jsonl"
    packet_path.write_text(
        json.dumps(audit_packet_rows[0]) + "\n",
        encoding="utf-8",
    )
    audit_report = analyze_auditability(
        run_dir,
        baselines="trajwiki_observed,full_context_proxy,no_wiki_direct",
    )
    failure_report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=1)

    assert Path(cost_report["quality_table_path"]).exists()
    assert Path(audit_report["source_support_table_path"]).exists()
    assert Path(audit_report["error_propagation_funnel_path"]).exists()
    assert len(_jsonl_rows(packet_path)) == expected_queries
    assert failure_report["totals"]["total_queries"] == expected_queries


def test_medmt_run_ignores_locomo_only_diagnostics_with_sharded_workers(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    config = RunConfig(
        dataset="medmt",
        dataset_path=_medmt_root(tmp_path / "medmt"),
        output_dir=output_dir,
        database_path=None,
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
        conv_workers=3,
        memory_extract_batch_size="auto",
        judge_max_concurrency=6,
        ablation_diagnostics=True,
        retrieval_rank_save_mode="full",
        cost_diagnostics=True,
        cost_call_save_mode="compact",
        auditability_diagnostics=True,
        audit_packet_save_mode="compact",
        verbose=True,
    )
    report = PipelineRunner(config).run()
    run_dir = Path(report.details["run_meta"]["run_dir"])

    assert report.details["run_meta"]["dataset"] == "medmt"
    assert report.details["run_meta"]["worker_mode"] == "sharded"
    details = json.loads((run_dir / "details.json").read_text(encoding="utf-8"))
    assert len(details["samples"]) == 3
    assert not (run_dir / "analysis" / "gold_labels.jsonl").exists()
    assert not (run_dir / "analysis" / "cost_query_rows.jsonl").exists()
    assert not (run_dir / "analysis" / "answer_support_rows.jsonl").exists()
