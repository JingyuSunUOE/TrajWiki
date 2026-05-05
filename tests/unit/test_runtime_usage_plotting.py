from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_plot_module():
    module_path = Path(__file__).resolve().parents[2] / "plot" / "plot_runtime_usage_analysis.py"
    spec = importlib.util.spec_from_file_location("plot_runtime_usage_analysis", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot_runtime = _load_plot_module()


def _summary_by_task(tasks: dict[str, dict[str, float]]) -> dict[str, object]:
    return {
        "run_meta": {"run_id": "unit-run"},
        "costs": {"total_runtime_s": 123.0},
        "llm_call_diagnostics": {"by_task": tasks},
    }


def test_task_runtime_rows_sorting_and_share() -> None:
    summary = _summary_by_task(
        {
            "small_task": {"latency_ms": 1_000, "provider_call_count": 1, "total_tokens": 10},
            "large_task": {"latency_ms": 10_000, "provider_call_count": 2, "total_tokens": 20},
            "medium_task": {"latency_ms": 4_000, "provider_call_count": 3, "total_tokens": 30},
        }
    )

    rows = plot_runtime.task_runtime_rows(summary)

    assert [row["task"] for row in rows] == ["large_task", "medium_task", "small_task"]
    assert rows[0]["runtime_share"] == pytest.approx(10_000 / 15_000)
    assert rows[1]["latency_seconds"] == pytest.approx(4.0)
    assert rows[1]["latency_minutes"] == pytest.approx(4_000 / 60_000)
    assert rows[1]["latency_hours"] == pytest.approx(4_000 / 3_600_000)


def test_known_task_ids_get_human_readable_display_names() -> None:
    summary = _summary_by_task(
        {
            "trajectory_retrieval_summary": {"latency_ms": 100},
            "answer_evidence_synthesis": {"latency_ms": 50},
        }
    )

    rows = plot_runtime.task_runtime_rows(summary)
    by_task = {row["task"]: row for row in rows}

    assert by_task["trajectory_retrieval_summary"]["display_task"] == "Trajectory summaries"
    assert by_task["answer_evidence_synthesis"]["display_task"] == "Answer synthesis"


def test_grouped_pie_rows_adds_other_bucket() -> None:
    summary = _summary_by_task(
        {
            "task_a": {"latency_ms": 100},
            "task_b": {"latency_ms": 80},
            "task_c": {"latency_ms": 20},
            "task_d": {"latency_ms": 10},
        }
    )
    rows = plot_runtime.task_runtime_rows(summary)

    pie_rows = plot_runtime.grouped_pie_rows(rows, top_n=2)

    assert [row["task"] for row in pie_rows] == ["task_a", "task_b", "Other"]
    other = pie_rows[-1]
    assert other["is_grouped_other"] is True
    assert other["latency_ms"] == pytest.approx(30)
    assert other["runtime_share"] == pytest.approx(30 / 210)
    assert other["grouped_tasks"] == ["task_c", "task_d"]


def test_format_duration_auto_and_explicit_units() -> None:
    assert plot_runtime.format_duration(12.345, unit="auto") == "12.3s"
    assert plot_runtime.format_duration(125, unit="auto") == "2.1m"
    assert plot_runtime.format_duration(7200, unit="auto") == "2.00h"
    assert plot_runtime.format_duration(7200, unit="minutes") == "120.0m"
    assert plot_runtime.format_duration(7200, unit="seconds") == "7200.0s"


def test_resolve_run_path_selects_latest_completed_run(tmp_path: Path) -> None:
    old_run = tmp_path / "20260101_old"
    new_run = tmp_path / "20260102_new"
    old_run.mkdir()
    new_run.mkdir()
    for run_dir in [old_run, new_run]:
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "details.json").write_text("{}", encoding="utf-8")

    resolved = plot_runtime.resolve_run_path(tmp_path)
    assert resolved == new_run.resolve()


def test_plot_script_writes_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "details.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            _summary_by_task(
                {
                    "task_a": {
                        "latency_ms": 110_000,
                        "provider_call_count": 2,
                        "total_tokens": 110,
                    },
                    "task_b": {
                        "latency_ms": 25_000,
                        "provider_call_count": 1,
                        "total_tokens": 25,
                    },
                    "task_c": {
                        "latency_ms": 10_000,
                        "provider_call_count": 1,
                        "total_tokens": 10,
                    },
                }
            )
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "plots"
    summary = plot_runtime.run(run_dir, output_dir, top_n=2)

    for name in [
        "runtime_usage_by_task_pie.pdf",
        "runtime_usage_by_task_pie.png",
        "runtime_usage_by_task.csv",
        "runtime_usage_analysis_summary.json",
    ]:
        assert (output_dir / name).exists()
    assert summary["total_provider_latency_ms"] == pytest.approx(145_000)
    assert summary["wall_clock_runtime_seconds"] == pytest.approx(123.0)
    assert summary["other"]["task"] == "Other"
    assert summary["other"]["display_task"] == "Other tasks"
    assert summary["other"]["grouped_tasks"] == ["task_c"]

    csv_text = (output_dir / "runtime_usage_by_task.csv").read_text(encoding="utf-8")
    assert "prompt" not in csv_text.lower()
    assert "raw context" not in csv_text.lower()
    assert "embedding" not in csv_text.lower()
