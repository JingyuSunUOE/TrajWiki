from __future__ import annotations

from pathlib import Path

import pytest

from trajpatch.config import RunConfig
from trajpatch.storage.database import create_schema
from trajpatch.storage.repository import TrajPatchStore


@pytest.fixture()
def run_config(tmp_path: Path) -> RunConfig:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    return RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "trajpatch.sqlite",
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        m=2,
        t_pages=2,
        k=2,
    )


@pytest.fixture()
def store(run_config: RunConfig) -> TrajPatchStore:
    session_factory = create_schema(run_config.database_path)
    session = session_factory()
    return TrajPatchStore(session)
