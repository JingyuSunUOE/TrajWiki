from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trajpatch.config import RunConfig
from trajpatch.exceptions import ParserValidationError
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider


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
    (root / "long_context_memory_and_understanding.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    for file_name in [
        "resistance_to_contextual_interference.json",
        "information_contradiction.json",
    ]:
        (root / file_name).write_text("[]", encoding="utf-8")
    return root


def test_index_schema_guard_raises_on_missing_required_run_registry_columns(tmp_path: Path):
    index_db = tmp_path / "trajpatch_index.sqlite"
    connection = sqlite3.connect(index_db)
    try:
        connection.execute(
            """
            CREATE TABLE run_registry (
                run_id TEXT PRIMARY KEY,
                dataset TEXT,
                run_dir TEXT,
                run_database_path TEXT,
                backbone_model TEXT,
                backbone_provider_kind TEXT,
                judge_model TEXT,
                judge_provider_kind TEXT,
                embedding_model TEXT,
                m INTEGER,
                k INTEGER,
                r INTEGER,
                started_at TEXT,
                completed_at TEXT,
                processed_samples INTEGER,
                processed_queries INTEGER,
                excluded_count INTEGER,
                total_runtime_s REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                metadata_json TEXT,
                created_at TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "old_output",
        database_path=None,
        index_database_path=index_db,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
    )

    with pytest.raises(
        ParserValidationError,
        match=(
            "missing required columns .*dataset_scope_key.*neighbor_radius.*retrieval_expansion_mode"
            "|missing required columns .*dataset_scope_key.*retrieval_expansion_mode.*neighbor_radius"
            "|missing required columns .*neighbor_radius.*dataset_scope_key.*retrieval_expansion_mode"
            "|missing required columns .*neighbor_radius.*retrieval_expansion_mode.*dataset_scope_key"
            "|missing required columns .*retrieval_expansion_mode.*dataset_scope_key.*neighbor_radius"
            "|missing required columns .*retrieval_expansion_mode.*neighbor_radius.*dataset_scope_key"
        ),
    ):
        PipelineRunner(
            config,
            llm_provider=MockLLMProvider(),
            embedding_provider=HashEmbeddingProvider(),
            judge_provider=MockLLMProvider(model_name="mock-judge"),
        ).run()
