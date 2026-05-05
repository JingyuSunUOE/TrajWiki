from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from trajpatch.config import RunConfig
from trajpatch.pipeline.runner import PipelineRunner, PipelineShard


def _runner(tmp_path) -> PipelineRunner:
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "output",
        provider_kind="mock",
        backbone_provider_kind="mock",
        judge_provider_kind="mock",
    )
    return PipelineRunner(config, console=Console(file=StringIO()))


def test_run_failed_json_and_events_are_written_without_prompt_text(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner.run_started_at = "2026-01-01T00:00:00"
    runner._prepare_run_database()
    try:
        runner._emit_event(
            "answer_generation_start",
            stage="answer_generation",
            sample_id="conv-49",
            query_task_id="conv-49_qa_1",
        )
        runner._write_run_failed(RuntimeError("remote provider disconnected"), stage="answer_generation")

        assert runner.run_dir is not None
        failure_path = runner.run_dir / "run_failed.json"
        events_path = runner.run_dir / "status" / "events.jsonl"
        assert failure_path.exists()
        assert events_path.exists()

        payload = json.loads(failure_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "run_failure_v1"
        assert payload["stage"] == "answer_generation"
        assert payload["sample_id"] == "conv-49"
        assert payload["query_task_id"] == "conv-49_qa_1"
        assert payload["error_type"] == "RuntimeError"
        assert "prompt_text" not in json.dumps(payload)
    finally:
        if runner.session is not None:
            runner.session.close()


def test_failed_worker_shard_is_copied_to_run_dir(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner.run_started_at = "2026-01-01T00:00:00"
    runner._prepare_run_database()
    try:
        source = tmp_path / "scratch" / "worker-00.sqlite"
        source.parent.mkdir(parents=True)
        source.write_text("sqlite bytes", encoding="utf-8")
        path_wal = source.with_name(source.name + "-wal")
        path_wal.write_text("wal bytes", encoding="utf-8")
        shard = PipelineShard(worker_id=0, groups=[], query_count=0, database_path=source)

        runner._preserve_failed_worker_shard(shard, RuntimeError("worker failed"))

        assert runner.run_dir is not None
        copied = runner.run_dir / "failed_shards" / "worker-00" / "worker-00.sqlite"
        copied_wal = runner.run_dir / "failed_shards" / "worker-00" / "worker-00.sqlite-wal"
        assert copied.read_text(encoding="utf-8") == "sqlite bytes"
        assert copied_wal.read_text(encoding="utf-8") == "wal bytes"
        assert str(copied) in runner.failed_shard_paths
    finally:
        if runner.session is not None:
            runner.session.close()
