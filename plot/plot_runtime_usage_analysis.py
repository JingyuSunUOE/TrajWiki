from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trajpatch-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trajpatch-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_UNITS = {"auto", "seconds", "minutes", "hours"}
SOURCE_MODES = {"auto", "cost-phase", "summary-task"}

PHASE_DISPLAY_NAMES = {
    "answer_generation": "Answer generation",
    "evaluation_only": "Benchmark evaluation",
    "memory_build": "Memory construction",
    "query_time": "Query-time retrieval",
    "repair_validation": "Repair/validation",
    "unknown": "Unknown",
}

TASK_DISPLAY_NAMES = {
    "answer_count_validation": "Count validation",
    "answer_evidence_synthesis": "Answer synthesis",
    "answer_generation_repair": "Answer repair",
    "claim_preservation_repair": "Claim preservation repair",
    "claim_signal_extract": "Claim signal extraction",
    "claim_transition_judge": "Claim transition judging",
    "episodic_claim_text_extract": "Claim text extraction",
    "episodic_extract": "Episodic extraction",
    "locomo_judge": "LOCOMO judging",
    "retrieval_reflection": "Retrieval reflection",
    "semantic_metric_extract": "Semantic metric extraction",
    "semantic_metric_schema": "Semantic metric schema",
    "trajectory_match": "Trajectory matching",
    "trajectory_retrieval_summary": "Trajectory summaries",
    "trajectory_set_rerank": "Trajectory reranking",
    "wiki_page_compile": "Wiki page compilation",
    "wiki_page_plan": "Wiki page planning",
    "wiki_page_rerank": "Wiki page reranking",
}


def resolve_run_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "summary.json").exists() and (path / "details.json").exists():
        return path
    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "summary.json").exists() and (child / "details.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not locate a run with summary.json/details.json under {path}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def display_task_name(task: str) -> str:
    if task == "Other":
        return "Other tasks"
    return TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").capitalize())


def display_phase_name(phase: str) -> str:
    if phase == "Other":
        return "Other phases"
    return PHASE_DISPLAY_NAMES.get(phase, phase.replace("_", " ").capitalize())


def phase_runtime_rows(run_path: Path) -> list[dict[str, Any]]:
    csv_rows = load_csv_rows(run_path / "analysis" / "cost_phase_summary.csv")
    if not csv_rows:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for row in csv_rows:
        phase = str(row.get("cost_phase") or "unknown")
        bucket = grouped.setdefault(
            phase,
            {
                "task": phase,
                "display_task": display_phase_name(phase),
                "latency_ms": 0.0,
                "latency_seconds": 0.0,
                "latency_minutes": 0.0,
                "latency_hours": 0.0,
                "provider_call_count": 0.0,
                "total_tokens": 0.0,
                "runtime_share": 0.0,
                "is_grouped_other": False,
                "source": "cost_phase_summary",
            },
        )
        bucket["latency_ms"] += _number(float(row.get("latency_ms") or 0.0))
        bucket["provider_call_count"] += _number(float(row.get("provider_call_rows") or 0.0))
        bucket["total_tokens"] += _number(float(row.get("total_tokens") or 0.0))

    raw_rows = list(grouped.values())
    total_latency_ms = sum(row["latency_ms"] for row in raw_rows)
    for row in raw_rows:
        row["latency_seconds"] = row["latency_ms"] / 1000
        row["latency_minutes"] = row["latency_ms"] / 60_000
        row["latency_hours"] = row["latency_ms"] / 3_600_000
        row["runtime_share"] = row["latency_ms"] / total_latency_ms if total_latency_ms else 0.0
    return sorted(raw_rows, key=lambda row: (-float(row["latency_ms"]), str(row["task"])))


def _time_unit_for_seconds(seconds: float, requested: str) -> str:
    if requested != "auto":
        return requested
    if seconds >= 3600:
        return "hours"
    if seconds >= 60:
        return "minutes"
    return "seconds"


def _time_value(seconds: float, unit: str) -> float:
    if unit == "hours":
        return seconds / 3600
    if unit == "minutes":
        return seconds / 60
    return seconds


def format_duration(seconds: float, *, unit: str = "auto") -> str:
    resolved_unit = _time_unit_for_seconds(seconds, unit)
    value = _time_value(seconds, resolved_unit)
    if resolved_unit == "hours":
        return f"{value:.2f}h"
    if resolved_unit == "minutes":
        return f"{value:.1f}m"
    return f"{value:.1f}s"


def task_runtime_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_task = summary.get("llm_call_diagnostics", {}).get("by_task")
    if not isinstance(by_task, dict):
        return []

    raw_rows: list[dict[str, Any]] = []
    for task, stats in by_task.items():
        if not isinstance(stats, dict):
            continue
        latency_ms = _number(stats.get("latency_ms"))
        raw_rows.append(
            {
                "task": str(task),
                "display_task": display_task_name(str(task)),
                "latency_ms": latency_ms,
                "latency_seconds": latency_ms / 1000,
                "latency_minutes": latency_ms / 60_000,
                "latency_hours": latency_ms / 3_600_000,
                "provider_call_count": _number(stats.get("provider_call_count")),
                "total_tokens": _number(stats.get("total_tokens")),
                "runtime_share": 0.0,
                "is_grouped_other": False,
                "source": "summary_llm_call_diagnostics",
            }
        )

    total_latency_ms = sum(row["latency_ms"] for row in raw_rows)
    for row in raw_rows:
        row["runtime_share"] = row["latency_ms"] / total_latency_ms if total_latency_ms else 0.0
    return sorted(raw_rows, key=lambda row: (-float(row["latency_ms"]), str(row["task"])))


def grouped_pie_rows(rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    top_rows = [dict(row) for row in rows[:top_n]]
    other_rows = rows[top_n:]
    if not other_rows:
        return top_rows

    total_latency_ms = sum(float(row["latency_ms"]) for row in rows)
    other_latency_ms = sum(float(row["latency_ms"]) for row in other_rows)
    other_calls = sum(float(row["provider_call_count"]) for row in other_rows)
    other_tokens = sum(float(row["total_tokens"]) for row in other_rows)
    top_rows.append(
        {
            "task": "Other",
            "display_task": display_task_name("Other"),
            "latency_ms": other_latency_ms,
            "latency_seconds": other_latency_ms / 1000,
            "latency_minutes": other_latency_ms / 60_000,
            "latency_hours": other_latency_ms / 3_600_000,
            "provider_call_count": other_calls,
            "total_tokens": other_tokens,
            "runtime_share": other_latency_ms / total_latency_ms if total_latency_ms else 0.0,
            "is_grouped_other": True,
            "grouped_task_count": len(other_rows),
            "grouped_tasks": [row["task"] for row in other_rows],
        }
    )
    return top_rows


def plot_runtime_pie(
    rows: list[dict[str, Any]],
    *,
    output_base: Path,
    time_unit: str,
    title: str,
) -> None:
    if not rows:
        return
    values = [float(row["latency_ms"]) for row in rows]
    total_ms = sum(values)
    if total_ms <= 0:
        return
    total_seconds = total_ms / 1000
    labels = [
        f"{row['display_task']}: {format_duration(float(row['latency_seconds']), unit=time_unit)} "
        f"({float(row['runtime_share']) * 100:.1f}%)"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    colors = list(plt.get_cmap("tab20").colors)
    wedges, _texts = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=colors[: len(values)],
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.text(
        0,
        0.06,
        format_duration(total_seconds, unit=time_unit),
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(0, -0.12, "provider latency", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.legend(
        wedges,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=8,
        columnspacing=1.8,
        handlelength=1.1,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    run_path: Path,
    output_dir: Path,
    *,
    top_n: int = 12,
    time_unit: str = "auto",
    title: str = "Runtime by Task",
    source: str = "auto",
) -> dict[str, Any]:
    if time_unit not in TIME_UNITS:
        raise ValueError(f"Unsupported time unit: {time_unit}")
    if source not in SOURCE_MODES:
        raise ValueError(f"Unsupported source: {source}")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    run_path = resolve_run_path(run_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(run_path / "summary.json")
    selected_source = source
    rows: list[dict[str, Any]] = []
    if source in {"auto", "cost-phase"}:
        rows = phase_runtime_rows(run_path)
        if rows:
            selected_source = "cost-phase"
        elif source == "cost-phase":
            raise FileNotFoundError(f"Could not read cost phase summary under {run_path / 'analysis'}")
    if not rows:
        rows = task_runtime_rows(summary)
        selected_source = "summary-task"
    pie_rows = grouped_pie_rows(rows, top_n=top_n)
    output_stem = "runtime_usage_by_cost_phase" if selected_source == "cost-phase" else "runtime_usage_by_task"

    write_csv(
        output_dir / f"{output_stem}.csv",
        rows,
        [
            "task",
            "display_task",
            "latency_ms",
            "latency_seconds",
            "latency_minutes",
            "latency_hours",
            "provider_call_count",
            "total_tokens",
            "runtime_share",
            "is_grouped_other",
            "source",
        ],
    )
    plot_runtime_pie(
        pie_rows,
        output_base=output_dir / f"{output_stem}_pie",
        time_unit=time_unit,
        title=title,
    )

    other_row = next((row for row in pie_rows if row.get("is_grouped_other")), None)
    total_provider_latency_ms = sum(float(row["latency_ms"]) for row in rows)
    wall_clock_runtime_s = _number(summary.get("costs", {}).get("total_runtime_s"))
    output_summary = {
        "schema_version": "runtime_usage_analysis_plot_v1",
        "run_path": str(run_path),
        "run_id": summary.get("run_meta", {}).get("run_id") or run_path.name,
        "top_n": top_n,
        "time_unit": time_unit,
        "source": selected_source,
        "task_count": len(rows),
        "total_provider_latency_ms": total_provider_latency_ms,
        "total_provider_latency_seconds": total_provider_latency_ms / 1000,
        "wall_clock_runtime_seconds": wall_clock_runtime_s,
        "top_tasks": [
            {
                "task": row["task"],
                "display_task": row["display_task"],
                "latency_ms": row["latency_ms"],
                "runtime_share": row["runtime_share"],
            }
            for row in rows[:top_n]
        ],
        "other": other_row,
        "outputs": {
            "runtime_usage_pie_pdf": str((output_dir / f"{output_stem}_pie.pdf").resolve()),
            "runtime_usage_pie_png": str((output_dir / f"{output_stem}_pie.png").resolve()),
            "runtime_usage_csv": str((output_dir / f"{output_stem}.csv").resolve()),
        },
    }
    (output_dir / "runtime_usage_analysis_summary.json").write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot offline runtime diagnostics by LLM task.")
    parser.add_argument("--run-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("plot"), type=Path)
    parser.add_argument("--top-n", default=12, type=int)
    parser.add_argument("--time-unit", default="auto", choices=sorted(TIME_UNITS))
    parser.add_argument("--source", default="auto", choices=sorted(SOURCE_MODES))
    parser.add_argument("--title", default="Runtime by Task")
    args = parser.parse_args(argv)
    summary = run(
        args.run_path,
        args.output_dir,
        top_n=args.top_n,
        time_unit=args.time_unit,
        title=args.title,
        source=args.source,
    )
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
