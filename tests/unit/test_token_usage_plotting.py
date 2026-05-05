from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_plot_module():
    module_path = Path(__file__).resolve().parents[2] / "plot" / "plot_token_usage_analysis.py"
    spec = importlib.util.spec_from_file_location("plot_token_usage_analysis", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot_tokens = _load_plot_module()


def _summary_by_task(tasks: dict[str, dict[str, float]]) -> dict[str, object]:
    return {
        "run_meta": {"run_id": "unit-run"},
        "llm_call_diagnostics": {"by_task": tasks},
    }


def test_task_token_rows_sorting_and_share() -> None:
    summary = _summary_by_task(
        {
            "small_task": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "large_task": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
            "medium_task": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        }
    )

    rows = plot_tokens.task_token_rows(summary)

    assert [row["task"] for row in rows] == ["large_task", "medium_task", "small_task"]
    assert rows[0]["token_share"] == pytest.approx(100 / 155)
    assert rows[1]["token_share"] == pytest.approx(40 / 155)
    assert rows[2]["token_share"] == pytest.approx(15 / 155)
    assert all(row["is_grouped_other"] is False for row in rows)


def test_known_task_ids_get_human_readable_display_names() -> None:
    summary = _summary_by_task(
        {
            "wiki_page_rerank": {"total_tokens": 100},
            "answer_evidence_synthesis": {"total_tokens": 50},
        }
    )

    rows = plot_tokens.task_token_rows(summary)
    by_task = {row["task"]: row for row in rows}

    assert by_task["wiki_page_rerank"]["display_task"] == "Wiki page reranking"
    assert by_task["answer_evidence_synthesis"]["display_task"] == "Answer synthesis"


def test_total_tokens_falls_back_to_prompt_plus_completion() -> None:
    summary = _summary_by_task(
        {
            "task_without_total": {"prompt_tokens": 12, "completion_tokens": 8},
            "task_with_total": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 30},
        }
    )

    rows = plot_tokens.task_token_rows(summary)
    by_task = {row["task"]: row for row in rows}

    assert by_task["task_without_total"]["total_tokens"] == pytest.approx(20)
    assert by_task["task_without_total"]["metric_tokens"] == pytest.approx(20)
    assert by_task["task_with_total"]["metric_tokens"] == pytest.approx(30)


def test_grouped_pie_rows_adds_other_bucket() -> None:
    summary = _summary_by_task(
        {
            "task_a": {"total_tokens": 100},
            "task_b": {"total_tokens": 80},
            "task_c": {"total_tokens": 20},
            "task_d": {"total_tokens": 10},
        }
    )
    rows = plot_tokens.task_token_rows(summary)

    pie_rows = plot_tokens.grouped_pie_rows(rows, top_n=2)

    assert [row["task"] for row in pie_rows] == ["task_a", "task_b", "Other"]
    other = pie_rows[-1]
    assert other["is_grouped_other"] is True
    assert other["metric_tokens"] == pytest.approx(30)
    assert other["token_share"] == pytest.approx(30 / 210)
    assert other["grouped_tasks"] == ["task_c", "task_d"]


def test_prompt_metric_uses_prompt_token_share() -> None:
    summary = _summary_by_task(
        {
            "prompt_heavy": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
            "completion_heavy": {"prompt_tokens": 10, "completion_tokens": 90, "total_tokens": 100},
        }
    )

    rows = plot_tokens.task_token_rows(summary, metric="prompt_tokens")

    assert [row["task"] for row in rows] == ["prompt_heavy", "completion_heavy"]
    assert rows[0]["metric_tokens"] == pytest.approx(90)
    assert rows[0]["token_share"] == pytest.approx(0.9)


def test_resolve_run_path_selects_latest_completed_run(tmp_path: Path) -> None:
    old_run = tmp_path / "20260101_old"
    new_run = tmp_path / "20260102_new"
    old_run.mkdir()
    new_run.mkdir()
    for run_dir in [old_run, new_run]:
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "details.json").write_text("{}", encoding="utf-8")

    resolved = plot_tokens.resolve_run_path(tmp_path)
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
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                        "provider_call_count": 2,
                        "latency_ms": 50,
                    },
                    "task_b": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                        "provider_call_count": 1,
                        "latency_ms": 20,
                    },
                    "task_c": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                        "provider_call_count": 1,
                        "latency_ms": 10,
                    },
                }
            )
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "plots"
    summary = plot_tokens.run(run_dir, output_dir, top_n=2)

    for name in [
        "token_usage_by_task_pie.pdf",
        "token_usage_by_task_pie.png",
        "token_usage_by_task.csv",
        "token_usage_analysis_summary.json",
    ]:
        assert (output_dir / name).exists()
    assert summary["total_metric_tokens"] == pytest.approx(145)
    assert summary["other"]["task"] == "Other"
    assert summary["other"]["display_task"] == "Other tasks"
    assert summary["other"]["grouped_tasks"] == ["task_c"]

    csv_text = (output_dir / "token_usage_by_task.csv").read_text(encoding="utf-8")
    assert "prompt" not in csv_text.lower().replace("prompt_tokens", "")
    assert "raw context" not in csv_text.lower()
    assert "embedding" not in csv_text.lower()
