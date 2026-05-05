from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_plot_module():
    module_path = Path(__file__).resolve().parents[2] / "plot" / "plot_trajectory_drift_analysis.py"
    spec = importlib.util.spec_from_file_location("plot_trajectory_drift_analysis", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot_drift = _load_plot_module()


def _stats(mean: float, *, count: int = 3) -> dict[str, float | int]:
    return {
        "count": count,
        "mean": mean,
        "median": mean,
        "p10": mean - 0.1,
        "p25": mean - 0.05,
        "p75": mean + 0.05,
        "p90": mean + 0.1,
    }


def _diagnostics() -> dict[str, object]:
    return {
        "trajectory_count": 5,
        "non_singleton_trajectory_count": 4,
        "embedding_available_count": 5,
        "missing_embedding_count": 0,
        "singleton_trajectory_rate": 0.2,
        "possible_or_high_span_rate": 0.25,
        "head_tail_cosine_stats": _stats(0.72, count=5),
        "adjacent_mean_cosine_stats": _stats(0.68, count=4),
        "summary_head_cosine_stats": _stats(0.80, count=5),
        "summary_tail_cosine_stats": _stats(0.78, count=5),
        "summary_centroid_cosine_stats": _stats(0.83, count=5),
        "drift_by_length_bucket": {
            "1": {
                "trajectory_count": 1,
                "head_tail_cosine_stats": _stats(1.0, count=1),
                "adjacent_mean_cosine_stats": {"count": 0, "mean": None},
                "possible_or_high_span_rate": 0.0,
            },
            "11-15": {
                "trajectory_count": 1,
                "head_tail_cosine_stats": _stats(0.52, count=1),
                "adjacent_mean_cosine_stats": _stats(0.66, count=1),
                "adjacent_min_cosine_stats": _stats(0.50, count=1),
                "summary_tail_cosine_stats": _stats(0.70, count=1),
                "possible_or_high_span_rate": 1.0,
            },
            "2-3": {
                "trajectory_count": 2,
                "head_tail_cosine_stats": _stats(0.72, count=2),
                "adjacent_mean_cosine_stats": _stats(0.70, count=2),
                "adjacent_min_cosine_stats": _stats(0.64, count=2),
                "summary_tail_cosine_stats": _stats(0.76, count=2),
                "possible_or_high_span_rate": 0.0,
            },
            "4-6": {
                "trajectory_count": 1,
                "head_tail_cosine_stats": _stats(0.61, count=1),
                "adjacent_mean_cosine_stats": _stats(0.67, count=1),
                "adjacent_min_cosine_stats": _stats(0.57, count=1),
                "summary_tail_cosine_stats": _stats(0.73, count=1),
                "possible_or_high_span_rate": 0.2,
            },
        },
    }


def test_non_singleton_distribution_rows_are_compact() -> None:
    rows = [
        {
            "trajectory_id": "t1",
            "sample_id": "s1",
            "snapshot_count": 1,
            "embedding_available": True,
            "head_tail_cosine": 1.0,
            "vector": [1, 2, 3],
        },
        {
            "trajectory_id": "t2",
            "sample_id": "s1",
            "snapshot_count": 3,
            "embedding_available": True,
            "length_bucket": "2-3",
            "drift_bucket": "stable",
            "head_tail_cosine": 0.75,
            "adjacent_mean_cosine": 0.70,
            "summary_tail_cosine": 0.80,
            "raw_text": "do not preserve",
            "vector": [1, 2, 3],
        },
    ]

    compact = plot_drift.non_singleton_distribution_rows(rows)

    assert len(compact) == 1
    assert compact[0]["trajectory_id"] == "t2"
    assert compact[0]["head_tail_cosine"] == pytest.approx(0.75)
    assert "raw_text" not in compact[0]
    assert "vector" not in compact[0]


def test_length_bucket_rows_have_stable_order_and_values() -> None:
    rows = plot_drift.length_bucket_rows(_diagnostics())

    assert [row["length_bucket"] for row in rows] == ["2-3", "4-6", "11-15"]
    assert rows[0]["head_tail_cosine_mean"] == pytest.approx(0.72)
    assert rows[0]["adjacent_mean_cosine_mean"] == pytest.approx(0.70)
    assert rows[-1]["possible_or_high_span_rate"] == pytest.approx(1.0)


def test_resolve_run_path_selects_latest_completed_run(tmp_path: Path) -> None:
    old_run = tmp_path / "20260101_old"
    new_run = tmp_path / "20260102_new"
    old_run.mkdir()
    new_run.mkdir()
    for run_dir in [old_run, new_run]:
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "details.json").write_text("{}", encoding="utf-8")

    assert plot_drift.resolve_run_path(tmp_path) == new_run.resolve()


def test_plot_script_writes_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"run_meta": {"run_id": "unit-run"}}),
        encoding="utf-8",
    )
    (run_dir / "details.json").write_text(json.dumps({"samples": []}), encoding="utf-8")
    (analysis_dir / "trajectory_drift_diagnostics.json").write_text(
        json.dumps(_diagnostics()),
        encoding="utf-8",
    )
    drift_rows = [
        {
            "trajectory_id": "t1",
            "sample_id": "s1",
            "snapshot_count": 2,
            "embedding_available": True,
            "length_bucket": "2-3",
            "drift_bucket": "stable",
            "head_tail_cosine": 0.72,
            "adjacent_mean_cosine": 0.70,
            "summary_tail_cosine": 0.76,
        },
        {
            "trajectory_id": "t2",
            "sample_id": "s1",
            "snapshot_count": 12,
            "embedding_available": True,
            "length_bucket": "11-15",
            "drift_bucket": "high_span",
            "head_tail_cosine": 0.52,
            "adjacent_mean_cosine": 0.66,
            "summary_tail_cosine": 0.70,
        },
    ]
    (analysis_dir / "trajectory_drift_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in drift_rows) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "plots"
    summary = plot_drift.run(run_dir, output_dir)

    for name in [
        "trajectory_head_tail_distribution.pdf",
        "trajectory_head_tail_distribution.png",
        "trajectory_drift_by_length.pdf",
        "trajectory_drift_by_length.png",
        "trajectory_head_tail_distribution.csv",
        "trajectory_drift_by_length.csv",
        "trajectory_drift_plot_summary.json",
    ]:
        assert (output_dir / name).exists()
    assert summary["run_id"] == "unit-run"
    assert summary["non_singleton_trajectory_count"] == 4
