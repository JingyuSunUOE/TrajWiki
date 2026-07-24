from __future__ import annotations

import csv
import json
from pathlib import Path

from trajpatch.analysis import analyze_offline_ablation
from trajpatch.config import RunConfig
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider


def _locomo_file(path: Path) -> Path:
    row = {
        "sample_id": "conv-offline",
        "qa_uid": "conv-offline_qa_0",
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


def test_offline_ablation_artifacts_are_generated_from_mock_locomo_run(tmp_path: Path) -> None:
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
    )
    report = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()
    run_dir = Path(report.details["run_meta"]["run_dir"])

    gold_path = run_dir / "analysis" / "gold_labels.jsonl"
    assert gold_path.exists()
    gold_rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line]
    assert gold_rows[0]["schema_version"] == "gold_labels_v2"

    details = json.loads((run_dir / "details.json").read_text(encoding="utf-8"))
    retrieval_diagnostics = details["samples"][0]["metadata"]["retrieval_compact_diagnostics"]
    assert "ablation_page_ranked_rows_v1" in retrieval_diagnostics
    assert "ablation_trajectory_ranked_rows_v1" in retrieval_diagnostics
    assert "answer_context_token_breakdown_v1" in retrieval_diagnostics

    generated = analyze_offline_ablation(
        run_dir,
        variants="full,no_wiki_direct,wiki_only,flat_raw,snapshot_m1,source_supported_only",
        budgets="4000",
        rank_cutoffs="1,15",
    )
    assert Path(generated["summary_path"]).exists()
    assert Path(generated["table_path"]).exists()
    assert Path(generated["rows_path"]).exists()

    with Path(generated["table_path"]).open(encoding="utf-8", newline="") as handle:
        table_rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in table_rows} >= {
        "full",
        "no_wiki_direct",
        "wiki_only",
        "flat_raw",
        "snapshot_m1",
        "source_supported_only",
    }
