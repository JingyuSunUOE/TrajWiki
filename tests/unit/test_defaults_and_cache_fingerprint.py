from __future__ import annotations

from pathlib import Path

import pytest

from trajpatch.cache.fingerprints import (
    MEMORY_CACHE_CODE_PATHS,
    MEMORY_CACHE_SCHEMA_VERSION,
    MEMORY_PROMPT_NAMES,
    build_memory_fingerprint,
    prompt_hashes,
    source_hashes,
)
from trajpatch.config import RunConfig

try:
    from typer.testing import CliRunner
except ModuleNotFoundError:  # pragma: no cover - depends on test environment extras
    CliRunner = None
    app = None
else:
    from trajpatch.cli import app


def test_run_config_defaults_use_output_dir(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    config = RunConfig(dataset="locomo", dataset_path=dataset_path)

    assert config.output_dir == Path("output").expanduser().resolve()
    assert config.index_database_path == Path("output/trajpatch_index.sqlite").expanduser().resolve()
    assert config.rebuild_semantic_metric_cache is False


@pytest.mark.skipif(CliRunner is None, reason="typer is not installed in the current test environment")
def test_cli_help_defaults_do_not_reference_previous_output() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "previous_output" not in result.stdout
    assert "semantic metric" in result.stdout


def test_memory_cache_fingerprint_covers_current_memory_prompts() -> None:
    expected = {
        "episodic_claim_text_extract",
        "episodic_claim_text_extract_structured",
        "claim_signal_extract",
        "claim_signal_extract_structured",
        "episodic_claim_preservation_repair",
    }

    assert MEMORY_CACHE_SCHEMA_VERSION == "v23"
    assert expected <= set(MEMORY_PROMPT_NAMES)
    assert expected <= set(prompt_hashes())
    assert "memory/orchestrator.py" in MEMORY_CACHE_CODE_PATHS
    assert "pipeline/runner.py" in MEMORY_CACHE_CODE_PATHS
    assert "providers/openai_compatible_provider.py" in MEMORY_CACHE_CODE_PATHS
    assert "providers/structured_outputs.py" in MEMORY_CACHE_CODE_PATHS
    assert "memory/orchestrator.py" in source_hashes()
    assert "pipeline/runner.py" in source_hashes()
    assert "providers/openai_compatible_provider.py" in source_hashes()


class _DummyAdapter:
    adapter_version = "test-v1"


def test_memory_cache_fingerprint_ignores_retrieval_and_batch_width(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    base = RunConfig(
        dataset="locomo",
        dataset_path=dataset_path,
        output_dir=tmp_path / "output",
        database_path=tmp_path / "output" / "run.sqlite",
        memory_extract_batch_size="auto",
        t_pages=5,
        m=4,
    )

    base_fingerprint, _ = build_memory_fingerprint(base, _DummyAdapter())
    same_memory_fingerprint, _ = build_memory_fingerprint(
        base.copy(update={"memory_extract_batch_size": 2, "t_pages": 10}),
        _DummyAdapter(),
    )
    changed_memory_fingerprint, _ = build_memory_fingerprint(
        base.copy(update={"m": 7}),
        _DummyAdapter(),
    )

    assert same_memory_fingerprint == base_fingerprint
    assert changed_memory_fingerprint != base_fingerprint


def test_memory_cache_fingerprint_tracks_openai_compatible_structured_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    base = RunConfig(
        dataset="locomo",
        dataset_path=dataset_path,
        output_dir=tmp_path / "output",
        database_path=tmp_path / "output" / "run.sqlite",
        provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        openai_compatible_base_url="http://localhost:8000/v1",
        openai_compatible_structured_mode="vllm",
        m=4,
    )

    base_fingerprint, _ = build_memory_fingerprint(base, _DummyAdapter())
    changed_mode_fingerprint, _ = build_memory_fingerprint(
        base.copy(update={"openai_compatible_structured_mode": "openai_json_schema"}),
        _DummyAdapter(),
    )

    assert changed_mode_fingerprint != base_fingerprint


def test_memory_cache_fingerprint_tracks_source_hash_changes(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    config = RunConfig(
        dataset="locomo",
        dataset_path=dataset_path,
        output_dir=tmp_path / "output",
        database_path=tmp_path / "output" / "run.sqlite",
        m=4,
    )

    monkeypatch.setattr("trajpatch.cache.fingerprints.source_hashes", lambda: {"memory/orchestrator.py": "a"})
    first_fingerprint, first_payload = build_memory_fingerprint(config, _DummyAdapter())
    monkeypatch.setattr("trajpatch.cache.fingerprints.source_hashes", lambda: {"memory/orchestrator.py": "b"})
    second_fingerprint, second_payload = build_memory_fingerprint(config, _DummyAdapter())

    assert first_payload["source_hashes"] == {"memory/orchestrator.py": "a"}
    assert second_payload["source_hashes"] == {"memory/orchestrator.py": "b"}
    assert first_fingerprint != second_fingerprint
