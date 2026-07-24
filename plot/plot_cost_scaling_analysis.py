from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trajpatch-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trajpatch-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float:
    if value in {None, ""}:
        return 0.0
    return float(value)


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_column(rows: list[dict[str, Any]], column: str) -> dict[str, float]:
    values = [number(row.get(column)) for row in rows]
    return {
        f"{column}_mean": safe_mean(values),
        f"{column}_min": min(values) if values else 0.0,
        f"{column}_max": max(values) if values else 0.0,
    }


def candidate_rows_by_sample(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id") or "")].append(row)

    output: list[dict[str, Any]] = []
    for sample_id, sample_rows in grouped.items():
        first = sample_rows[0]
        direct_values = [number(row.get("direct_candidate_universe")) for row in sample_rows]
        routed_values = [number(row.get("wiki_routed_candidate_universe")) for row in sample_rows]
        output.append(
            {
                "sample_id": sample_id,
                "raw_message_count": number(first.get("raw_message_count")),
                "raw_memory_token_estimate": number(first.get("raw_memory_token_estimate")),
                "trajectory_count": number(first.get("trajectory_count")),
                "snapshot_count": number(first.get("snapshot_count")),
                "claim_count": number(first.get("claim_count")),
                "wiki_page_count": number(first.get("wiki_page_count")),
                "direct_candidate_universe_mean": safe_mean(direct_values),
                "wiki_routed_candidate_universe_mean": safe_mean(routed_values),
            }
        )
    return sorted(output, key=lambda row: (row["trajectory_count"], row["sample_id"]))


def plot_memory_scaling(rows: list[dict[str, str]], output_base: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: number(row.get("raw_message_count")))
    x_values = list(range(1, len(sorted_rows) + 1))
    series = [
        ("raw_message_count", "Raw messages"),
        ("trajectory_count", "Trajectories"),
        ("snapshot_count", "Snapshots"),
        ("claim_count", "Claims"),
        ("wiki_page_count", "Wiki pages"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = plt.get_cmap("tab10").colors
    for index, (column, label) in enumerate(series):
        ax.plot(
            x_values,
            [number(row.get(column)) for row in sorted_rows],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=label,
            color=colors[index % len(colors)],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Dialogue sample, sorted by raw message count")
    ax.set_ylabel("Memory object count (log scale)")
    ax.set_title("Memory Size Across Dialogue Samples")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_scaling(rows: list[dict[str, Any]], output_base: Path) -> None:
    x_values = [row["trajectory_count"] for row in rows]
    direct_values = [row["direct_candidate_universe_mean"] for row in rows]
    routed_values = [row["wiki_routed_candidate_universe_mean"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(x_values, direct_values, marker="o", linewidth=2.0, label="Direct trajectory universe")
    ax.plot(x_values, routed_values, marker="s", linewidth=2.0, label="Wiki-routed universe")
    ax.set_xlabel("Trajectories stored in dialogue memory")
    ax.set_ylabel("Candidate trajectories per query")
    ax.set_title("Candidate Universe Scaling")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(run_path: Path, output_dir: Path) -> dict[str, Any]:
    run_path = resolve_run_path(run_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = run_path / "analysis"
    memory_rows = read_csv_rows(analysis_dir / "memory_scaling.csv")
    candidate_query_rows = read_csv_rows(analysis_dir / "candidate_scaling.csv")
    candidate_sample_rows = candidate_rows_by_sample(candidate_query_rows)

    write_csv(
        output_dir / "candidate_scaling_by_sample.csv",
        candidate_sample_rows,
        [
            "sample_id",
            "raw_message_count",
            "raw_memory_token_estimate",
            "trajectory_count",
            "snapshot_count",
            "claim_count",
            "wiki_page_count",
            "direct_candidate_universe_mean",
            "wiki_routed_candidate_universe_mean",
        ],
    )

    summary: dict[str, Any] = {
        "schema_version": "cost_scaling_analysis_plot_v1",
        "run_path": str(run_path),
        "sample_count": len(memory_rows),
        "query_count": len(candidate_query_rows),
        "memory": {},
        "candidate": {},
        "outputs": {
            "memory_scaling_pdf": str((output_dir / "memory_scaling_over_samples.pdf").resolve()),
            "memory_scaling_png": str((output_dir / "memory_scaling_over_samples.png").resolve()),
            "candidate_scaling_pdf": str((output_dir / "candidate_universe_scaling.pdf").resolve()),
            "candidate_scaling_png": str((output_dir / "candidate_universe_scaling.png").resolve()),
            "candidate_scaling_by_sample_csv": str((output_dir / "candidate_scaling_by_sample.csv").resolve()),
        },
    }
    for column in [
        "raw_message_count",
        "raw_memory_token_estimate",
        "trajectory_count",
        "snapshot_count",
        "claim_count",
        "claim_op_count",
        "wiki_page_count",
        "non_index_wiki_page_count",
        "avg_trajectory_length",
    ]:
        summary["memory"].update(summarize_column(memory_rows, column))
    for column in [
        "wiki_routed_candidate_universe",
        "direct_candidate_universe",
        "selected_page_count",
        "selected_trajectory_count",
        "selected_snapshot_count",
        "selected_source_count",
    ]:
        summary["candidate"].update(summarize_column(candidate_query_rows, column))
    direct_mean = summary["candidate"].get("direct_candidate_universe_mean", 0.0)
    routed_mean = summary["candidate"].get("wiki_routed_candidate_universe_mean", 0.0)
    summary["candidate"]["direct_to_wiki_candidate_ratio"] = (
        direct_mean / routed_mean if routed_mean else None
    )

    plot_memory_scaling(memory_rows, output_dir / "memory_scaling_over_samples")
    plot_candidate_scaling(candidate_sample_rows, output_dir / "candidate_universe_scaling")
    (output_dir / "cost_scaling_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot TrajWiki memory and candidate scaling diagnostics.")
    parser.add_argument("--run-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("plot/generated_cost_analysis"), type=Path)
    args = parser.parse_args(argv)
    summary = run(args.run_path, args.output_dir)
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
