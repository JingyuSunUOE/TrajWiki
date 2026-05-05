from __future__ import annotations

import json
from pathlib import Path

from trajpatch.config import RunConfig
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.reporting import SQLiteReportReader
from trajpatch.types import LLMResponse, ModelInfo


class _UnparseableJudgeProvider(LLMProvider):
    def generate(self, messages, *, system_prompt=None, metadata=None) -> LLMResponse:
        return LLMResponse(text="I think this answer is probably correct.", metadata=dict(metadata or {}))

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="unparseable-judge", is_remote=True)


def _sample_medmt_root(root: Path) -> Path:
    root.mkdir()
    payload = [
        {
            "id": "sample-a",
            "messages": [
                {"role": "system", "content": "Keep answers short."},
                {"role": "user", "content": "I smoke 5 cigarettes a day."},
                {"role": "assistant", "content": "Thanks, noted."},
                {"role": "user", "content": "What do you know about my smoking habits?"},
            ],
            "test_point": "Mention the smoking habit.",
            "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
            "meta": {"sence_type": {"type": "Consultation"}},
        }
    ]
    (root / "long_context_memory_and_understanding.json").write_text(json.dumps(payload), encoding="utf-8")
    for file_name in [
        "resistance_to_contextual_interference.json",
        "information_contradiction.json",
    ]:
        (root / file_name).write_text("[]", encoding="utf-8")
    return root


def _medmt_root(root: Path) -> Path:
    root.mkdir()
    payloads = {
        "long_context_memory_and_understanding.json": {
            "id": "med-long",
            "messages": [
                {"role": "system", "content": "Keep answers short."},
                {"role": "user", "content": "History long."},
                {"role": "assistant", "content": "Acknowledged."},
                {"role": "user", "content": "Final long question?"},
            ],
            "test_point": "Mention long-context memory.",
            "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
            "meta": {"sence_type": {"type": "Consultation"}},
        },
        "resistance_to_contextual_interference.json": {
            "id": "med-resist",
            "messages": [
                {"role": "system", "content": "Stay grounded."},
                {"role": "user", "content": "History resist."},
                {"role": "assistant", "content": "Acknowledged."},
                {"role": "user", "content": "Final resistance question?"},
            ],
            "test_point": "Mention interference resistance.",
            "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
            "meta": {"sence_type": {"type": "Consultation"}},
        },
        "information_contradiction.json": {
            "id": "med-contradict",
            "messages": [
                {"role": "system", "content": "Call out contradictions."},
                {"role": "user", "content": "History contradiction."},
                {"role": "assistant", "content": "Acknowledged."},
                {"role": "user", "content": "Final contradiction question?"},
            ],
            "test_point": "Mention contradiction.",
            "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
            "meta": {"sence_type": {"type": "Consultation"}},
        },
    }
    for file_name, payload in payloads.items():
        (root / file_name).write_text(json.dumps([payload]), encoding="utf-8")
    return root


def _locomo_benchmark_file(path: Path) -> Path:
    rows = [
        {
            "sample_id": "conv-1",
            "qa_uid": "conv-1_qa_0",
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
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _locomo_root(root: Path) -> Path:
    root.mkdir()
    payloads = {
        "category_1_multi_hop.jsonl": {
            "sample_id": "conv-multi",
            "qa_uid": "conv-multi_qa_0",
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
        },
        "category_2_temporal.jsonl": {
            "sample_id": "conv-temporal",
            "qa_uid": "conv-temporal_qa_0",
            "category": 2,
            "category_name": "temporal",
            "question": "When is the appointment?",
            "answer": "Tomorrow",
            "evidence": ["D1:1"],
            "full_conversation": (
                "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                "[D1:1] Alice: The appointment is tomorrow.\n"
                "[D1:2] Bob: Noted.\n"
            ),
        },
        "category_3_open_domain.jsonl": {
            "sample_id": "conv-open",
            "qa_uid": "conv-open_qa_0",
            "category": 3,
            "category_name": "open_domain",
            "question": "What topic came up?",
            "answer": "Travel",
            "evidence": ["D1:1"],
            "full_conversation": (
                "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                "[D1:1] Alice: I want to travel soon.\n"
                "[D1:2] Bob: Sounds good.\n"
            ),
        },
        "category_4_single_hop.jsonl": {
            "sample_id": "conv-single",
            "qa_uid": "conv-single_qa_0",
            "category": 4,
            "category_name": "single_hop",
            "question": "What beverage is mentioned?",
            "answer": "Tea",
            "evidence": ["D1:1"],
            "full_conversation": (
                "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                "[D1:1] Alice: Tea is my favorite.\n"
                "[D1:2] Bob: Nice.\n"
            ),
        },
    }
    for file_name, payload in payloads.items():
        (root / file_name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return root


def test_default_run_sqlite_isolated_and_indexed(tmp_path: Path):
    output_dir = tmp_path / "old_output"
    index_db = output_dir / "trajpatch_index.sqlite"

    medmt_config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        memory_cache_dir=tmp_path / "cache" / "medmt",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=2,
    )
    medmt_report = PipelineRunner(
        medmt_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()

    medmt_db = Path(medmt_report.details["run_meta"]["database_path"])
    assert medmt_db.exists()
    assert medmt_db.parent.name == medmt_report.details["run_meta"]["run_id"]
    assert medmt_db.parent.parent.name == "all"
    assert medmt_db.parent.parent.parent.name == "medmt"
    assert medmt_report.details["run_meta"]["dataset_scope_key"] == "all"
    assert medmt_report.details["run_meta"]["neighbor_radius"] == 2
    assert medmt_report.details["run_meta"]["retrieval_expansion_mode"] == "update_linked_plus_neighbors"
    assert "_nr2_remupdate-linked-plus-neighbors" in medmt_report.details["run_meta"]["run_id"]
    medmt_details = json.loads(Path(medmt_report.details["paths"]["details"]).read_text(encoding="utf-8"))
    backbone_usage = medmt_details["samples"][0]["metadata"]["backbone_usage"]
    assert "records" not in backbone_usage
    assert backbone_usage["records_suppressed"] is True
    assert backbone_usage["record_count"] > 0
    assert "answer_generation" in backbone_usage["task_counts"]
    assert backbone_usage["provider_call_count"] > 0
    assert backbone_usage["logical_call_count"] >= backbone_usage["provider_call_count"]
    assert "by_task" in backbone_usage
    assert "records" not in medmt_details["samples"][0]["metadata"]["llm_usage"]
    assert "fallback_repair_summary" in medmt_details["samples"][0]["metadata"]
    medmt_summary = json.loads(Path(medmt_report.details["paths"]["summary"]).read_text(encoding="utf-8"))
    assert medmt_summary["llm_call_diagnostics"]["overall"]["provider_call_count"] > 0
    assert "answer_generation" in medmt_summary["llm_call_diagnostics"]["by_task"]
    assert "fallback_repair_diagnostics" in medmt_summary
    assert Path(medmt_summary["paths"]["fallback_repair_events"]).exists()

    locomo_config = RunConfig(
        dataset="locomo",
        dataset_subset="multi_hop",
        dataset_path=_locomo_root(tmp_path / "locomo"),
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        memory_cache_dir=tmp_path / "cache" / "locomo",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=0,
    )
    locomo_report = PipelineRunner(
        locomo_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()
    locomo_metrics = locomo_report.details["aggregates"]["overall"]["metrics"]
    assert "F1" in locomo_metrics
    assert "BLEU-1" in locomo_metrics
    assert "semantic_slot_f1" not in locomo_metrics
    assert "semantic_canonical_bleu1" not in locomo_metrics

    locomo_db = Path(locomo_report.details["run_meta"]["database_path"])
    assert locomo_db.exists()
    assert locomo_db.parent.name == locomo_report.details["run_meta"]["run_id"]
    assert locomo_db.parent.parent.name == "multi_hop"
    assert locomo_db.parent.parent.parent.name == "locomo"
    assert locomo_report.details["run_meta"]["dataset_scope_key"] == "multi_hop"
    assert medmt_db != locomo_db
    assert index_db.exists()

    reader = SQLiteReportReader()
    medmt_rows = reader.report_index(index_database_path=index_db, dataset="medmt")
    locomo_rows = reader.report_index(index_database_path=index_db, dataset="locomo")
    assert len(medmt_rows) == 1
    assert len(locomo_rows) == 1
    assert medmt_rows[0]["run_database_path"] == str(medmt_db)
    assert locomo_rows[0]["run_database_path"] == str(locomo_db)
    assert "F1" in locomo_rows[0]["metrics"]
    assert "BLEU-1" in locomo_rows[0]["metrics"]
    assert "semantic_slot_f1" not in locomo_rows[0]["metrics"]
    assert "semantic_canonical_bleu1" not in locomo_rows[0]["metrics"]
    assert medmt_rows[0]["neighbor_radius"] == 2
    assert locomo_rows[0]["neighbor_radius"] == 0
    assert locomo_rows[0]["dataset_scope_key"] == "multi_hop"
    assert medmt_rows[0]["retrieval_expansion_mode"] == "update_linked_plus_neighbors"
    locomo_details = json.loads(Path(locomo_report.details["paths"]["details"]).read_text(encoding="utf-8"))
    locomo_row_metrics = locomo_details["samples"][0]["metrics"]
    assert "F1" in locomo_row_metrics
    assert "BLEU-1" in locomo_row_metrics
    assert "semantic_slot_f1" not in locomo_row_metrics
    assert "semantic_canonical_bleu1" not in locomo_row_metrics
    assert "llm_usage" in locomo_details["samples"][0]["metadata"]
    assert "semantic_metric_extract" in locomo_details["samples"][0]["metadata"]["llm_usage_by_task"]
    assert locomo_details["samples"][0]["metadata"]["semantic_metrics"]["cache_miss_count"] == 3
    assert locomo_details["samples"][0]["metadata"]["llm_usage"]["cache_miss_count"] == 3
    assert "fallback_repair_summary" in locomo_details["samples"][0]["metadata"]
    assert locomo_details["samples"][0]["metadata"]["evaluation_filter"]["text_only_eligible"] is True
    assert locomo_details["samples"][0]["metadata"]["evaluation_filter"]["excluded_from_text_only"] is False
    retrieval_compact = locomo_details["samples"][0]["metadata"]["retrieval_compact_diagnostics"]
    assert retrieval_compact["diagnostic_top_n_pages"] == 50
    assert retrieval_compact["diagnostic_top_n_trajectories"] == 50
    assert "page_ranked_rows_compact_top_n" in retrieval_compact
    assert "trajectory_ranked_rows_compact_top_n" in retrieval_compact
    locomo_summary = json.loads(Path(locomo_report.details["paths"]["summary"]).read_text(encoding="utf-8"))
    assert "semantic_metrics" in locomo_summary["llm_call_diagnostics"]["by_phase"]
    assert locomo_summary["llm_call_diagnostics"]["by_phase"]["semantic_metrics"]["cache_miss_count"] == 3
    assert locomo_summary["compact_retrieval_diagnostics"]["query_count"] == 1
    assert locomo_summary["compact_retrieval_diagnostics"]["saved_page_rank_limit_max"] == 50
    assert locomo_summary["evaluation_filters"]["text_only"]["included_count"] == 1
    assert locomo_summary["evaluation_filters"]["text_only"]["excluded_count"] == 0
    assert locomo_summary["aggregates"]["text_only_filtered"]["overall"]["count"] == 1
    assert Path(locomo_summary["paths"]["text_only_filter_manifest"]).exists()
    assert Path(locomo_summary["paths"]["text_only_filtered_summary"]).exists()
    assert "fallback_repair_diagnostics" in locomo_summary
    assert Path(locomo_summary["paths"]["fallback_repair_events"]).exists()


def test_run_id_includes_readable_config_and_unique_suffix(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "old_output",
        database_path=None,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=2,
    )
    runner = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    )

    first = runner._build_run_id()
    second = runner._build_run_id()

    assert first != second
    assert "_medmt_all_mock-backbone_mock-judge_hash-embedding" in first
    assert "_m2_tp2_k1_nr2_remupdate-linked-plus-neighbors" in first
    assert "_r" in first


def test_report_reader_reads_index_and_single_run(tmp_path: Path):
    output_dir = tmp_path / "old_output"
    index_db = output_dir / "trajpatch_index.sqlite"
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=2,
    )
    report = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()
    run_db = Path(report.details["run_meta"]["database_path"])

    reader = SQLiteReportReader()
    index_rows = reader.report_index(index_database_path=index_db, dataset="medmt")
    filtered_rows = reader.report_index(index_database_path=index_db, dataset="medmt", neighbor_radius=2)
    single_rows = reader.report_single_run(database_path=run_db)

    assert len(index_rows) == 1
    assert len(filtered_rows) == 1
    assert len(single_rows) == 1
    assert index_rows[0]["run_id"] == report.details["run_meta"]["run_id"]
    assert single_rows[0]["run_id"] == report.details["run_meta"]["run_id"]
    assert index_rows[0]["metrics"]["judge_acc"] == single_rows[0]["metrics"]["judge_acc"]
    assert index_rows[0]["neighbor_radius"] == 2
    assert single_rows[0]["neighbor_radius"] == 2
    assert index_rows[0]["dataset_scope_key"] == "all"
    assert single_rows[0]["dataset_scope_key"] == "all"
    assert index_rows[0]["retrieval_expansion_mode"] == "update_linked_plus_neighbors"
    assert single_rows[0]["retrieval_expansion_mode"] == "update_linked_plus_neighbors"


def test_runner_excludes_judge_error_from_judge_acc_and_reports_counts(tmp_path: Path):
    output_dir = tmp_path / "old_output"
    index_db = output_dir / "trajpatch_index.sqlite"
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="broken-judge",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=2,
    )

    report = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_UnparseableJudgeProvider(),
    ).run()

    assert report.metrics["judge_acc"] is None
    assert report.details["aggregates"]["overall"]["metrics"]["judge_acc"] is None
    assert report.details["judge"]["judge_evaluable_count"] == 0
    assert report.details["judge"]["judge_execution_failed_count"] == 1
    details_payload = json.loads(Path(report.details["paths"]["details"]).read_text(encoding="utf-8"))
    assert details_payload["samples"][0]["judge_verdict"] == "judge_error"
    assert details_payload["samples"][0]["metrics"]["judge_acc"] is None


def test_locomo_subset_runs_are_scoped_in_output_and_reports(tmp_path: Path):
    output_dir = tmp_path / "old_output"
    index_db = output_dir / "trajpatch_index.sqlite"
    dataset_root = _locomo_root(tmp_path / "locomo")

    subset_config = RunConfig(
        dataset="locomo",
        dataset_subset="multi_hop",
        dataset_path=dataset_root,
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=0,
    )
    subset_report = PipelineRunner(
        subset_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()

    full_config = RunConfig(
        dataset="locomo",
        dataset_subset="all",
        dataset_path=dataset_root,
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=0,
    )
    full_report = PipelineRunner(
        full_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()

    subset_run_meta = subset_report.details["run_meta"]
    full_run_meta = full_report.details["run_meta"]
    subset_rows = subset_report.details["aggregates"]["rows"]
    full_rows = full_report.details["aggregates"]["rows"]

    assert subset_run_meta["dataset_scope_key"] == "multi_hop"
    assert Path(subset_run_meta["database_path"]).parent.parent.name == "multi_hop"
    assert "multi-hop" in subset_run_meta["run_id"]
    assert [row["subset"] for row in subset_rows] == ["multi_hop"]
    assert subset_report.details["aggregates"]["overall"]["count"] == 1

    assert full_run_meta["dataset_scope_key"] == "all"
    assert Path(full_run_meta["database_path"]).parent.parent.name == "all"
    assert [row["subset"] for row in full_rows] == ["multi_hop", "temporal", "open_domain", "single_hop"]

    reader = SQLiteReportReader()
    all_rows = reader.report_index(index_database_path=index_db, dataset="locomo")
    multi_hop_rows = reader.report_index(index_database_path=index_db, dataset="locomo", subset="multi_hop")
    full_scope_rows = reader.report_index(index_database_path=index_db, dataset="locomo", subset="all")

    assert len(all_rows) == 2
    assert {row["dataset_scope_key"] for row in all_rows} == {"all", "multi_hop"}
    assert [row["run_id"] for row in multi_hop_rows] == [subset_run_meta["run_id"]]
    assert [row["run_id"] for row in full_scope_rows] == [full_run_meta["run_id"]]


def test_medmt_subset_runs_are_scoped_in_output_and_reports(tmp_path: Path):
    output_dir = tmp_path / "old_output"
    index_db = output_dir / "trajpatch_index.sqlite"
    dataset_root = _medmt_root(tmp_path / "medmt")

    subset_config = RunConfig(
        dataset="medmt",
        dataset_subset="information_contradiction",
        dataset_path=dataset_root,
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=1,
    )
    subset_report = PipelineRunner(
        subset_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()

    full_config = RunConfig(
        dataset="medmt",
        dataset_subset="all",
        dataset_path=dataset_root,
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        neighbor_radius=1,
    )
    full_report = PipelineRunner(
        full_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()

    subset_run_meta = subset_report.details["run_meta"]
    full_run_meta = full_report.details["run_meta"]
    subset_rows = subset_report.details["aggregates"]["rows"]
    full_rows = full_report.details["aggregates"]["rows"]

    assert subset_run_meta["dataset_scope_key"] == "information_contradiction"
    assert Path(subset_run_meta["database_path"]).parent.parent.name == "information_contradiction"
    assert "information-contradiction" in subset_run_meta["run_id"]
    assert {row["subset"] for row in subset_rows} == {"information_contradiction"}
    assert len(subset_rows) == 3
    assert subset_report.details["aggregates"]["overall"]["count"] == 1

    assert full_run_meta["dataset_scope_key"] == "all"
    assert Path(full_run_meta["database_path"]).parent.parent.name == "all"
    assert {row["subset"] for row in full_rows} == {
        "long_context_memory_and_understanding",
        "resistance_to_contextual_interference",
        "information_contradiction",
    }
    assert len(full_rows) == 9

    reader = SQLiteReportReader()
    all_rows = reader.report_index(index_database_path=index_db, dataset="medmt")
    subset_rows_report = reader.report_index(
        index_database_path=index_db,
        dataset="medmt",
        subset="information_contradiction",
    )
    full_scope_rows = reader.report_index(index_database_path=index_db, dataset="medmt", subset="all")

    assert len(all_rows) == 2
    assert {row["dataset_scope_key"] for row in all_rows} == {"all", "information_contradiction"}
    assert [row["run_id"] for row in subset_rows_report] == [subset_run_meta["run_id"]]
    assert [row["run_id"] for row in full_scope_rows] == [full_run_meta["run_id"]]


def test_report_reader_filters_by_retrieval_expansion_mode(tmp_path: Path):
    output_dir = tmp_path / "old_output"
    index_db = output_dir / "trajpatch_index.sqlite"
    dataset_root = _sample_medmt_root(tmp_path / "medmt")

    default_config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_root,
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        retrieval_expansion_mode="update_linked_plus_neighbors",
    )
    default_report = PipelineRunner(
        default_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()

    none_config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_root,
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        retrieval_expansion_mode="none",
    )
    none_report = PipelineRunner(
        none_config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    ).run()

    reader = SQLiteReportReader()
    all_rows = reader.report_index(index_database_path=index_db, dataset="medmt")
    default_rows = reader.report_index(
        index_database_path=index_db,
        dataset="medmt",
        retrieval_expansion_mode="update_linked_plus_neighbors",
    )
    none_rows = reader.report_index(
        index_database_path=index_db,
        dataset="medmt",
        retrieval_expansion_mode="none",
    )

    assert len(all_rows) == 2
    assert {row["retrieval_expansion_mode"] for row in all_rows} == {
        "update_linked_plus_neighbors",
        "none",
    }
    assert [row["run_id"] for row in default_rows] == [default_report.details["run_meta"]["run_id"]]
    assert [row["run_id"] for row in none_rows] == [none_report.details["run_meta"]["run_id"]]
