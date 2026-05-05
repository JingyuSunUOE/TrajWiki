from __future__ import annotations

import pytest

from trajpatch.config import RunConfig
from trajpatch.pipeline import sweep as sweep_module


def test_run_grid_propagates_retrieval_expansion_mode(tmp_path, monkeypatch):
    observed_modes: list[str] = []

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run(self):
            observed_modes.append(self.config.retrieval_expansion_mode)
            return type(
                "Report",
                (),
                {
                    "metrics": {"judge_acc": 1.0},
                    "details": {
                        "run_meta": {
                            "retrieval_expansion_mode": self.config.retrieval_expansion_mode
                        }
                    },
                },
            )()

    monkeypatch.setattr(sweep_module, "PipelineRunner", FakeRunner)
    base = RunConfig(
        dataset="medmt",
        dataset_path=tmp_path,
        output_dir=tmp_path / "old_output",
        database_path=tmp_path / "run.sqlite",
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
    )

    reports = sweep_module.run_grid(
        base,
        m_values=[4],
        k_values=[2],
        r_values=[1],
        neighbor_radius_values=[0],
        retrieval_expansion_mode_values=["update_linked_plus_neighbors", "none"],
    )

    assert observed_modes == ["update_linked_plus_neighbors", "none"]
    assert [report["retrieval_expansion_mode"] for report in reports] == observed_modes


def test_run_grid_revalidates_invalid_retrieval_expansion_mode(tmp_path):
    base = RunConfig(
        dataset="medmt",
        dataset_path=tmp_path,
        output_dir=tmp_path / "old_output",
        database_path=tmp_path / "run.sqlite",
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
    )

    with pytest.raises(Exception):
        sweep_module.run_grid(
            base,
            m_values=[4],
            k_values=[2],
            r_values=[1],
            neighbor_radius_values=[0],
            retrieval_expansion_mode_values=["bad-mode"],
        )
