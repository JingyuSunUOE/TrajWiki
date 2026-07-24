from __future__ import annotations

import csv
import json
from pathlib import Path

from trajpatch.analysis import analyze_cost_benefit, analyze_offline_ablation
from trajpatch.config import RunConfig
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider


def _locomo_file(path: Path) -> Path:
    row = {
        "sample_id": "conv-cost",
        "qa_uid": "conv-cost_qa_0",
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


def test_cost_benefit_artifacts_are_generated_from_mock_locomo_run(tmp_path: Path) -> None:
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
    )
    report = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()
    run_dir = Path(report.details["run_meta"]["run_dir"])

    cost_call_path = run_dir / "analysis" / "cost_call_rows.jsonl"
    cost_query_path = run_dir / "analysis" / "cost_query_rows.jsonl"
    assert cost_call_path.exists()
    assert cost_query_path.exists()
    details = json.loads((run_dir / "details.json").read_text(encoding="utf-8"))
    assert "cost_query_rows" in details["paths"]
    call_rows = [json.loads(line) for line in cost_call_path.read_text(encoding="utf-8").splitlines() if line]
    query_rows = [
        json.loads(line) for line in cost_query_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert call_rows
    assert query_rows[0]["schema_version"] == "cost_query_v2"
    assert (run_dir / "analysis" / "cost_reconciliation.json").exists()

    analyze_offline_ablation(
        run_dir,
        variants="full,no_wiki_direct,flat_raw,wiki_only",
        budgets="4000",
        rank_cutoffs="15",
    )
    generated = analyze_cost_benefit(
        run_dir,
        baselines="trajwiki_observed,full_context_proxy,no_wiki_direct",
        future_query_counts="1,2,5",
    )
    for key in ["summary_path", "quality_table_path", "phase_summary_path", "break_even_path"]:
        assert Path(generated[key]).exists()

    with Path(generated["quality_table_path"]).open(encoding="utf-8", newline="") as handle:
        quality_rows = list(csv.DictReader(handle))
    assert {row["method"] for row in quality_rows} >= {
        "trajwiki_observed",
        "full_context_proxy",
        "no_wiki_direct",
    }
    assert (run_dir / "analysis" / "memory_scaling.csv").exists()
    assert (run_dir / "analysis" / "candidate_scaling.csv").exists()
