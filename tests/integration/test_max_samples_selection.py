from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trajpatch.config import RunConfig
from trajpatch.memory.orchestrator import MemoryOrchestrator
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.providers.transformers_provider import SentenceTransformerEmbeddingProvider
from trajpatch.storage.models import AnswerRecord, RetrievalEvent, TrajectoryRecord, WikiPageRecord


def _write_locomo_row(path: Path, *, sample_id: str, qa_uid: str, category: int, category_name: str, question: str) -> None:
    path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "qa_uid": qa_uid,
                "category": category,
                "category_name": category_name,
                "question": question,
                "answer": "Tea",
                "evidence": ["D1:1"],
                "full_conversation": (
                    "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                    "[D1:1] Alice: I love tea.\n"
                    "[D1:2] Bob: Noted.\n"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _locomo_grouped_root(root: Path) -> Path:
    root.mkdir()
    _write_locomo_row(
        root / "category_1_multi_hop.jsonl",
        sample_id="conv-shared",
        qa_uid="conv-shared_qa_0",
        category=1,
        category_name="multi_hop",
        question="What beverage is mentioned in multi-hop?",
    )
    _write_locomo_row(
        root / "category_2_temporal.jsonl",
        sample_id="conv-shared",
        qa_uid="conv-shared_qa_1",
        category=2,
        category_name="temporal",
        question="When was the beverage mentioned?",
    )
    _write_locomo_row(
        root / "category_3_open_domain.jsonl",
        sample_id="conv-other",
        qa_uid="conv-other_qa_0",
        category=3,
        category_name="open_domain",
        question="What topic came up?",
    )
    _write_locomo_row(
        root / "category_4_single_hop.jsonl",
        sample_id="conv-third",
        qa_uid="conv-third_qa_0",
        category=4,
        category_name="single_hop",
        question="What beverage is mentioned in single hop?",
    )
    return root


def _medmt_all_root(root: Path) -> Path:
    root.mkdir()
    payloads = {
        "long_context_memory_and_understanding.json": [
            {
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
            }
        ],
        "resistance_to_contextual_interference.json": [
            {
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
            }
        ],
        "information_contradiction.json": [
            {
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
            }
        ],
    }
    for file_name, payload in payloads.items():
        (root / file_name).write_text(json.dumps(payload), encoding="utf-8")
    return root


def _medmt_excluded_first_root(root: Path) -> Path:
    root.mkdir()
    (root / "long_context_memory_and_understanding.json").write_text(
        json.dumps(
            [
                {
                    "id": "med-excluded",
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "History excluded."},
                        {"role": "assistant", "content": "Acknowledged."},
                        {"role": "user", "content": "Final excluded question?"},
                    ],
                    "test_point": "Excluded due to unsupported scene.",
                    "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
                    "meta": {"sence_type": {"type": "Nursing"}},
                },
                {
                    "id": "med-active",
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "History active."},
                        {"role": "assistant", "content": "Acknowledged."},
                        {"role": "user", "content": "Final active question?"},
                    ],
                    "test_point": "Should still be selected.",
                    "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
                    "meta": {"sence_type": {"type": "Consultation"}},
                },
            ]
        ),
        encoding="utf-8",
    )
    for file_name in [
        "resistance_to_contextual_interference.json",
        "information_contradiction.json",
    ]:
        (root / file_name).write_text("[]", encoding="utf-8")
    return root


def _locomo_two_exchange_root(root: Path) -> Path:
    root.mkdir()
    (root / "category_1_multi_hop.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "conv-two",
                "qa_uid": "conv-two_qa_0",
                "category": 1,
                "category_name": "multi_hop",
                "question": "What drinks are mentioned?",
                "answer": "Tea and coffee",
                "evidence": ["D1:1", "D1:3"],
                "full_conversation": (
                    "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                    "[D1:1] Alice: I love tea.\n"
                    "[D1:2] Bob: Noted.\n"
                    "[D1:3] Alice: I also drink coffee.\n"
                    "[D1:4] Bob: Noted again.\n"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for file_name in [
        "category_2_temporal.jsonl",
        "category_3_open_domain.jsonl",
        "category_4_single_hop.jsonl",
    ]:
        (root / file_name).write_text("", encoding="utf-8")
    return root


def _counting_mock_callback(counter: Counter[str]):
    delegate = MockLLMProvider()

    def _callback(messages, system_prompt, metadata):
        task = str((metadata or {}).get("task") or "unknown")
        counter[task] += 1
        if task == "episodic_extract":
            return (
                "SUMMARY_CONTENT: Alice loves tea.\n"
                "CONTEXT: Beverage preference.\n"
                "KEYWORDS: tea, beverage"
            )
        return delegate._default_response(messages, metadata or {})

    return _callback


def test_locomo_max_samples_groups_queries_by_conversation_and_builds_memory_once(tmp_path: Path):
    dataset_path = _locomo_grouped_root(tmp_path / "locomo")
    counter: Counter[str] = Counter()
    config = RunConfig(
        dataset="locomo",
        dataset_subset="all",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        max_samples=1,
    )

    runner = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(callback=_counting_mock_callback(counter)),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    )
    report = runner.run()

    assert report.processed_samples == 1
    assert report.processed_queries == 2
    assert report.details["run_meta"]["logical_sample_count"] == 1
    assert report.details["run_meta"]["selected_row_count"] == 2
    assert report.details["run_meta"]["selected_query_count"] == 2
    assert report.details["run_meta"]["selection_strategy"] == "locomo-logical-sample"
    assert report.details["run_meta"]["selected_counts_by_subset"] == {"multi_hop": 1, "temporal": 1}
    subset_counts = {
        row["subset"]: row["count"]
        for row in report.details["aggregates"]["rows"]
        if row["count"] > 0
    }
    assert subset_counts == {"multi_hop": 1, "temporal": 1}
    assert counter["episodic_extract"] == 1
    assert counter["wiki_page_plan"] == 1
    assert counter["answer_freeform_generation"] == 2
    assert counter["answer_generation"] == 0
    assert counter["retrieval_reflection"] == 0
    assert runner.session.query(AnswerRecord).count() == 2


def test_medmt_max_samples_all_uses_round_robin_across_subsets(tmp_path: Path):
    dataset_path = _medmt_all_root(tmp_path / "medmt")
    config = RunConfig(
        dataset="medmt",
        dataset_subset="all",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        max_samples=2,
    )

    runner = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    )
    report = runner.run()

    assert report.processed_samples == 2
    assert report.processed_queries == 2
    assert report.details["run_meta"]["selection_strategy"] == "medmt-round-robin-all"
    assert report.details["run_meta"]["selected_counts_by_subset"] == {
        "long_context_memory_and_understanding": 1,
        "resistance_to_contextual_interference": 1,
    }
    subset_counts = {
        row["subset"]: row["count"]
        for row in report.details["aggregates"]["rows"]
        if row["count"] > 0
    }
    assert set(subset_counts) == {
        "long_context_memory_and_understanding",
        "resistance_to_contextual_interference",
    }
    assert runner.session.query(AnswerRecord).count() == 2


def test_excluded_samples_do_not_consume_max_samples_budget(tmp_path: Path):
    dataset_path = _medmt_excluded_first_root(tmp_path / "medmt")
    config = RunConfig(
        dataset="medmt",
        dataset_subset="long_context_memory_and_understanding",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        max_samples=1,
    )

    runner = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(model_name="mock-judge"),
    )
    report = runner.run()

    assert report.processed_samples == 1
    assert report.processed_queries == 1
    assert report.details["run_meta"]["selected_row_count"] == 1
    assert report.details["run_meta"]["selected_query_count"] == 1
    assert report.details["exclusions"]["excluded_count"] == 1
    assert report.details["exclusions"]["sample_ids"] == ["med-excluded"]
    assert runner.session.query(AnswerRecord).count() == 1
    assert runner.session.query(AnswerRecord).first().sample_id == "med-active"


def test_conv_workers_shards_logical_samples_and_merges_final_database(monkeypatch, tmp_path: Path):
    dataset_path = _locomo_grouped_root(tmp_path / "locomo")
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch_root))
    config = RunConfig(
        dataset="locomo",
        dataset_subset="all",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        max_samples=2,
        conv_workers=2,
    )

    runner = PipelineRunner(config)
    report = runner.run()

    assert report.processed_samples == 2
    assert report.processed_queries == 3
    assert report.details["run_meta"]["conv_workers"] == 2
    assert report.details["run_meta"]["worker_mode"] == "sharded"
    assert len(report.details["run_meta"]["worker_shards"]) == 2
    run_meta = report.details["run_meta"]
    worker_db_root = Path(run_meta["worker_database_root"])
    assert run_meta["worker_database_local_scratch_used"] is True
    assert worker_db_root.is_relative_to(scratch_root)
    assert len(run_meta["worker_database_paths"]) == 2
    assert run_meta["worker_database_cleanup"]["attempted"] is True
    assert run_meta["worker_database_cleanup"]["kept_paths"] == []
    assert runner.session.query(AnswerRecord).count() == 3
    assert runner.session.query(RetrievalEvent).count() == 3
    assert runner.session.query(TrajectoryRecord).count() >= 2
    assert runner.session.query(WikiPageRecord).count() >= 2
    for entry in run_meta["worker_database_paths"]:
        database_path = Path(entry["database_path"])
        assert database_path.is_relative_to(worker_db_root)
        assert not database_path.exists()
    assert (Path(report.details["paths"]["run_dir"]) / "memories" / "conv-shared").exists()
    assert (Path(report.details["paths"]["run_dir"]) / "memories" / "conv-other").exists()


def test_conv_workers_worker_db_falls_back_to_run_dir_when_tmp_unwritable(monkeypatch, tmp_path: Path):
    dataset_path = _locomo_grouped_root(tmp_path / "locomo")
    blocked_tmp = tmp_path / "blocked-tmp"
    blocked_tmp.write_text("not-a-directory", encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(blocked_tmp))
    monkeypatch.setattr("trajpatch.pipeline.runner.tempfile.gettempdir", lambda: str(blocked_tmp))
    config = RunConfig(
        dataset="locomo",
        dataset_subset="all",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        max_samples=2,
        conv_workers=2,
    )

    report = PipelineRunner(config).run()
    run_meta = report.details["run_meta"]
    fallback_root = Path(report.details["paths"]["run_dir"]) / "shards"

    assert run_meta["worker_database_local_scratch_used"] is False
    assert Path(run_meta["worker_database_root"]) == fallback_root
    assert run_meta["worker_database_warnings"]
    assert run_meta["worker_database_cleanup"]["attempted"] is True
    assert len(run_meta["worker_database_cleanup"]["kept_paths"]) == 2
    for entry in run_meta["worker_database_paths"]:
        assert Path(entry["database_path"]).parent == fallback_root
        assert Path(entry["database_path"]).exists()


def test_memory_replay_commits_per_window_and_separates_finalize_and_wiki(monkeypatch, tmp_path: Path):
    dataset_path = _locomo_two_exchange_root(tmp_path / "locomo-two")
    commit_stages: list[tuple[str, str | None]] = []
    original_commit_session = PipelineRunner._commit_session

    def _recording_commit(self, *, stage: str, sample_id: str | None = None) -> None:
        commit_stages.append((stage, sample_id))
        return original_commit_session(self, stage=stage, sample_id=sample_id)

    monkeypatch.setattr(PipelineRunner, "_commit_session", _recording_commit)
    config = RunConfig(
        dataset="locomo",
        dataset_subset="multi_hop",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=4,
        t_pages=2,
        k=1,
        max_samples=1,
        memory_extract_batch_size=1,
    )

    PipelineRunner(config).run()

    assert commit_stages.count(("memory_window_persist", "conv-two")) == 2
    assert ("memory_finalize", "conv-two") in commit_stages
    assert ("wiki_compile", "conv-two") in commit_stages


def test_max_length_close_does_not_refresh_summary_with_closed_true(monkeypatch, tmp_path: Path):
    dataset_path = _locomo_grouped_root(tmp_path / "locomo")
    refresh_flags: list[bool] = []
    original_refresh = MemoryOrchestrator._refresh_trajectory_retrieval_summary

    def _record_refresh(self, trajectory_id: str, *, closed: bool) -> None:
        refresh_flags.append(closed)
        return original_refresh(self, trajectory_id, closed=closed)

    monkeypatch.setattr(MemoryOrchestrator, "_refresh_trajectory_retrieval_summary", _record_refresh)
    config = RunConfig(
        dataset="locomo",
        dataset_subset="all",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=1,
        t_pages=2,
        k=1,
        max_samples=1,
    )

    PipelineRunner(config).run()

    assert False in refresh_flags
    assert True not in refresh_flags


def test_conv_workers_preallocate_embedding_devices(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "trajpatch.providers.devices.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "H100-0",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 76 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            },
            {
                "index": 1,
                "name": "H100-1",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 76 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            },
        ],
    )

    def fake_embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(SentenceTransformerEmbeddingProvider, "embed_documents", fake_embed)
    monkeypatch.setattr(SentenceTransformerEmbeddingProvider, "embed_queries", fake_embed)

    dataset_path = _locomo_grouped_root(tmp_path / "locomo")
    config = RunConfig(
        dataset="locomo",
        dataset_subset="all",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        provider_kind="mock",
        memory_cache_enabled=False,
        backbone_model="mock-backbone",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        max_samples=2,
        conv_workers=2,
    )

    report = PipelineRunner(config).run()
    run_meta = report.details["run_meta"]

    assert run_meta["worker_device_allocation"]["enabled"] is True
    assert [row["accelerator"] for row in run_meta["worker_device_allocation"]["assignments"]] == [
        "cuda:0",
        "cuda:1",
    ]
    assert [row["embedding_accelerator"] for row in run_meta["worker_shards"]] == ["cuda:0", "cuda:1"]
    actual = run_meta["worker_device_allocation"]["worker_actual_device_allocations"]
    assert actual["00"]["embedding"]["accelerator"] == "cuda:0"
    assert actual["01"]["embedding"]["accelerator"] == "cuda:1"
    assert report.details["worker_device_allocation"]["assignments"] == run_meta["worker_device_allocation"]["assignments"]
