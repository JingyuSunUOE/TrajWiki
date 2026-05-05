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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


VALID_METRICS = {"total_tokens", "prompt_tokens", "completion_tokens"}

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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _token_value(stats: dict[str, Any], metric: str) -> float:
    prompt_tokens = _number(stats.get("prompt_tokens"))
    completion_tokens = _number(stats.get("completion_tokens"))
    if metric == "prompt_tokens":
        return prompt_tokens
    if metric == "completion_tokens":
        return completion_tokens
    total_tokens = _number(stats.get("total_tokens"))
    return total_tokens if total_tokens else prompt_tokens + completion_tokens


def display_task_name(task: str) -> str:
    if task == "Other":
        return "Other tasks"
    return TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").capitalize())


def task_token_rows(summary: dict[str, Any], *, metric: str = "total_tokens") -> list[dict[str, Any]]:
    if metric not in VALID_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    by_task = summary.get("llm_call_diagnostics", {}).get("by_task")
    if not isinstance(by_task, dict):
        return []

    raw_rows: list[dict[str, Any]] = []
    for task, stats in by_task.items():
        if not isinstance(stats, dict):
            continue
        prompt_tokens = _number(stats.get("prompt_tokens"))
        completion_tokens = _number(stats.get("completion_tokens"))
        total_tokens = _number(stats.get("total_tokens")) or prompt_tokens + completion_tokens
        metric_tokens = _token_value(stats, metric)
        raw_rows.append(
            {
                "task": str(task),
                "display_task": display_task_name(str(task)),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "provider_call_count": _number(stats.get("provider_call_count")),
                "latency_ms": _number(stats.get("latency_ms")),
                "metric_tokens": metric_tokens,
                "token_share": 0.0,
                "is_grouped_other": False,
            }
        )

    total_metric_tokens = sum(row["metric_tokens"] for row in raw_rows)
    for row in raw_rows:
        row["token_share"] = row["metric_tokens"] / total_metric_tokens if total_metric_tokens else 0.0
    return sorted(raw_rows, key=lambda row: (-float(row["metric_tokens"]), str(row["task"])))


def grouped_pie_rows(rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    top_rows = [dict(row) for row in rows[:top_n]]
    other_rows = rows[top_n:]
    if not other_rows:
        return top_rows

    total_metric_tokens = sum(float(row["metric_tokens"]) for row in rows)
    other_metric_tokens = sum(float(row["metric_tokens"]) for row in other_rows)
    other_prompt_tokens = sum(float(row["prompt_tokens"]) for row in other_rows)
    other_completion_tokens = sum(float(row["completion_tokens"]) for row in other_rows)
    other_total_tokens = sum(float(row["total_tokens"]) for row in other_rows)
    other_calls = sum(float(row["provider_call_count"]) for row in other_rows)
    other_latency = sum(float(row["latency_ms"]) for row in other_rows)
    top_rows.append(
        {
            "task": "Other",
            "display_task": display_task_name("Other"),
            "prompt_tokens": other_prompt_tokens,
            "completion_tokens": other_completion_tokens,
            "total_tokens": other_total_tokens,
            "provider_call_count": other_calls,
            "latency_ms": other_latency,
            "metric_tokens": other_metric_tokens,
            "token_share": other_metric_tokens / total_metric_tokens if total_metric_tokens else 0.0,
            "is_grouped_other": True,
            "grouped_task_count": len(other_rows),
            "grouped_tasks": [row["task"] for row in other_rows],
        }
    )
    return top_rows


def _format_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def plot_token_pie(
    rows: list[dict[str, Any]],
    *,
    output_base: Path,
    metric: str,
    title: str,
) -> None:
    if not rows:
        return
    values = [float(row["metric_tokens"]) for row in rows]
    total = sum(values)
    if total <= 0:
        return
    labels = [
        f"{row['display_task']}: {_format_tokens(float(row['metric_tokens']))} "
        f"({float(row['token_share']) * 100:.1f}%)"
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
        _format_tokens(total),
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(0, -0.12, metric.replace("_", " "), ha="center", va="center", fontsize=9)
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
    fig.tight_layout(rect=(0.0, 0.25, 1.0, 1.0))
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    run_path: Path,
    output_dir: Path,
    *,
    top_n: int = 12,
    metric: str = "total_tokens",
    title: str = "Token Usage by Task",
) -> dict[str, Any]:
    if metric not in VALID_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    run_path = resolve_run_path(run_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(run_path / "summary.json")
    rows = task_token_rows(summary, metric=metric)
    pie_rows = grouped_pie_rows(rows, top_n=top_n)

    write_csv(
        output_dir / "token_usage_by_task.csv",
        rows,
        [
            "task",
            "display_task",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "provider_call_count",
            "latency_ms",
            "metric_tokens",
            "token_share",
            "is_grouped_other",
        ],
    )
    plot_token_pie(
        pie_rows,
        output_base=output_dir / "token_usage_by_task_pie",
        metric=metric,
        title=title,
    )

    other_row = next((row for row in pie_rows if row.get("is_grouped_other")), None)
    output_summary = {
        "schema_version": "token_usage_analysis_plot_v1",
        "run_path": str(run_path),
        "run_id": summary.get("run_meta", {}).get("run_id") or run_path.name,
        "metric": metric,
        "top_n": top_n,
        "task_count": len(rows),
        "total_metric_tokens": sum(float(row["metric_tokens"]) for row in rows),
        "top_tasks": [
            {
                "task": row["task"],
                "display_task": row["display_task"],
                "metric_tokens": row["metric_tokens"],
                "token_share": row["token_share"],
            }
            for row in rows[:top_n]
        ],
        "other": other_row,
        "outputs": {
            "token_usage_pie_pdf": str((output_dir / "token_usage_by_task_pie.pdf").resolve()),
            "token_usage_pie_png": str((output_dir / "token_usage_by_task_pie.png").resolve()),
            "token_usage_csv": str((output_dir / "token_usage_by_task.csv").resolve()),
        },
    }
    (output_dir / "token_usage_analysis_summary.json").write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot offline token usage diagnostics by LLM task.")
    parser.add_argument("--run-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("plot"), type=Path)
    parser.add_argument("--top-n", default=12, type=int)
    parser.add_argument("--metric", default="total_tokens", choices=sorted(VALID_METRICS))
    parser.add_argument("--title", default="Token Usage by Task")
    args = parser.parse_args(argv)
    summary = run(
        args.run_path,
        args.output_dir,
        top_n=args.top_n,
        metric=args.metric,
        title=args.title,
    )
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
