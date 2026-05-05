from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_plot_module():
    module_path = Path(__file__).resolve().parents[2] / "plot" / "plot_hyperparameter_analysis.py"
    spec = importlib.util.spec_from_file_location("plot_hyperparameter_analysis", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot_hparams = _load_plot_module()


def _detail_row(
    query_id: str,
    *,
    page_rows: list[dict[str, object]],
    trajectory_prefixes: dict[str, list[str]],
    max_snapshot_rank: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "query_task_id": query_id,
        "sample_id": "sample-1",
        "metadata": {
            "retrieval_compact_diagnostics": {
                "page_ranked_rows_compact_top_n": page_rows,
                "page_ranked_total_count": len(page_rows),
                "trajectory_cutoff_prefix_diagnostics": {
                    str(cutoff): {"trajectory_ids": ids, "cutoff": cutoff}
                    for cutoff, ids in trajectory_prefixes.items()
                },
            },
            "query_metadata": {"gold_evidence_refs": []},
        },
    }
    if max_snapshot_rank is not None:
        row["max_gold_snapshot_rank_required"] = max_snapshot_rank
    return row


def test_page_and_trajectory_cutoff_aggregation() -> None:
    rows = [
        _detail_row(
            "q1",
            page_rows=[
                {"trajectory_ids": ["t1"]},
                {"trajectory_ids": ["x1"]},
                {"trajectory_ids": ["t2"]},
            ],
            trajectory_prefixes={1: ["t1"], 2: ["t1", "x1"], 3: ["t1", "t2", "x1"]},
        ),
        _detail_row(
            "q2",
            page_rows=[
                {"trajectory_ids": ["x2"]},
                {"trajectory_ids": ["t3"]},
                {"trajectory_ids": ["x3"]},
            ],
            trajectory_prefixes={1: ["x2"], 2: ["x2", "t3"], 3: ["x2", "t3", "x3"]},
        ),
    ]
    direct_by_query = {
        "q1": {"gold_trajectory_ids": ["t1", "t2"]},
        "q2": {"gold_trajectory_ids": ["t3"]},
    }

    page_rows = plot_hparams.compute_page_cutoff_rows(rows, direct_by_query, cutoffs=[1, 2, 3])
    assert page_rows[0]["mean_gold_trajectory_recall"] == pytest.approx(0.25)
    assert page_rows[1]["mean_gold_trajectory_recall"] == pytest.approx(0.75)
    assert page_rows[2]["mean_gold_trajectory_recall"] == pytest.approx(1.0)
    assert page_rows[2]["all_gold_trajectory_coverage_rate"] == pytest.approx(1.0)

    trajectory_rows = plot_hparams.compute_trajectory_cutoff_rows(
        rows,
        direct_by_query,
        cutoffs=[1, 2, 3],
    )
    assert trajectory_rows[0]["mean_gold_trajectory_recall"] == pytest.approx(0.25)
    assert trajectory_rows[1]["mean_gold_trajectory_recall"] == pytest.approx(0.75)
    assert trajectory_rows[2]["all_gold_trajectory_coverage_rate"] == pytest.approx(1.0)


def test_snapshot_cdf_statistics() -> None:
    cdf_rows, summary = plot_hparams.compute_snapshot_cdf_rows([1, 3, 5, 10], m_value=5)

    assert summary["median"] == 5
    assert summary["p90"] == 10
    assert summary["p95"] == 10
    assert summary["max"] == 10
    assert summary["coverage_at_m"] == pytest.approx(0.75)
    assert cdf_rows[0]["required_snapshot_rank"] == 1
    assert cdf_rows[0]["query_coverage"] == pytest.approx(0.25)
    assert cdf_rows[-1]["query_coverage"] == pytest.approx(1.0)


def test_snapshot_marginal_rows() -> None:
    rows = [
        {
            "required_snapshot_rank": 1,
            "covered_query_count": 2,
            "query_count": 6,
            "query_coverage": 2 / 6,
        },
        {
            "required_snapshot_rank": 2,
            "covered_query_count": 5,
            "query_count": 6,
            "query_coverage": 5 / 6,
        },
        {
            "required_snapshot_rank": 3,
            "covered_query_count": 6,
            "query_count": 6,
            "query_coverage": 1.0,
        },
    ]

    marginal = plot_hparams.compute_snapshot_marginal_rows(rows)
    summary = plot_hparams.compute_snapshot_marginal_summary(
        marginal,
        {"coverage_at_m": 1.0},
    )

    assert [row["newly_covered_query_count"] for row in marginal] == [2, 3, 1]
    assert marginal[1]["marginal_coverage_gain"] == pytest.approx(0.5)
    assert summary["peak_marginal_rank"] == 2
    assert summary["peak_marginal_query_count"] == 3
    assert summary["coverage_at_5"] == pytest.approx(1.0)
    assert summary["coverage_at_10"] == pytest.approx(1.0)


def test_resolve_run_path_selects_latest_completed_run(tmp_path: Path) -> None:
    old_run = tmp_path / "20260101_old"
    new_run = tmp_path / "20260102_new"
    old_run.mkdir()
    new_run.mkdir()
    for run_dir in [old_run, new_run]:
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "details.json").write_text("{}", encoding="utf-8")

    resolved = plot_hparams.resolve_run_path(tmp_path)
    assert resolved == new_run.resolve()


def test_plot_script_writes_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"run_meta": {"m": 5, "run_id": "unit-run"}}),
        encoding="utf-8",
    )
    details = {
        "samples": [
            _detail_row(
                "q1",
                page_rows=[
                    {"trajectory_ids": ["t1"]},
                    {"trajectory_ids": ["x1"]},
                    {"trajectory_ids": ["t2"]},
                    {"trajectory_ids": ["x2"]},
                    {"trajectory_ids": ["x3"]},
                ],
                trajectory_prefixes={
                    5: ["t1", "x1", "t2", "x2", "x3"],
                    10: ["t1", "x1", "t2", "x2", "x3"],
                    15: ["t1", "x1", "t2", "x2", "x3"],
                    20: ["t1", "x1", "t2", "x2", "x3"],
                    30: ["t1", "x1", "t2", "x2", "x3"],
                    50: ["t1", "x1", "t2", "x2", "x3"],
                },
                max_snapshot_rank=4,
            )
        ]
    }
    (run_dir / "details.json").write_text(json.dumps(details), encoding="utf-8")
    (analysis_dir / "direct_retrieval_rows.jsonl").write_text(
        json.dumps({"query_task_id": "q1", "sample_id": "sample-1", "gold_trajectory_ids": ["t1", "t2"]})
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "plots"
    summary = plot_hparams.run(run_dir, output_dir)

    for name in [
        "hyperparam_page_cutoff.pdf",
        "hyperparam_page_cutoff.png",
        "hyperparam_trajectory_k.pdf",
        "hyperparam_trajectory_k.png",
        "hyperparam_m_snapshot_cdf.pdf",
        "hyperparam_m_snapshot_cdf.png",
        "hyperparam_m_marginal_utility.pdf",
        "hyperparam_m_marginal_utility.png",
        "hyperparam_page_cutoff.csv",
        "hyperparam_trajectory_k.csv",
        "hyperparam_m_snapshot_cdf.csv",
        "hyperparam_m_marginal_utility.csv",
        "hyperparam_analysis_summary.json",
    ]:
        assert (output_dir / name).exists()
    assert summary["snapshot_cdf_summary"]["coverage_at_m"] == pytest.approx(1.0)
    assert summary["snapshot_marginal_summary"]["peak_marginal_rank"] == 4
    assert "snapshot_marginal_pdf" in summary["outputs"]
    assert "snapshot_marginal_png" in summary["outputs"]
