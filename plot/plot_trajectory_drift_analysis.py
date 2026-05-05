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


LENGTH_BUCKET_ORDER = ["2-3", "4-6", "7-10", "11-15"]


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def non_singleton_distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        snapshot_count = int(row.get("snapshot_count") or 0)
        if snapshot_count <= 1 or row.get("embedding_available") is False:
            continue
        head_tail = _number(row.get("head_tail_cosine"))
        if head_tail is None:
            continue
        output.append(
            {
                "trajectory_id": row.get("trajectory_id"),
                "sample_id": row.get("sample_id"),
                "snapshot_count": snapshot_count,
                "length_bucket": row.get("length_bucket"),
                "drift_bucket": row.get("drift_bucket"),
                "head_tail_cosine": head_tail,
                "adjacent_mean_cosine": _number(row.get("adjacent_mean_cosine")),
                "summary_tail_cosine": _number(row.get("summary_tail_cosine")),
            }
        )
    return output


def _stat_mean(bucket: dict[str, Any], key: str) -> float | None:
    stats = bucket.get(key)
    if not isinstance(stats, dict):
        return None
    return _number(stats.get("mean"))


def length_bucket_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    by_length = diagnostics.get("drift_by_length_bucket")
    if not isinstance(by_length, dict):
        return []
    rows: list[dict[str, Any]] = []
    for bucket_name in LENGTH_BUCKET_ORDER:
        bucket = by_length.get(bucket_name)
        if not isinstance(bucket, dict):
            continue
        rows.append(
            {
                "length_bucket": bucket_name,
                "trajectory_count": bucket.get("trajectory_count", 0),
                "head_tail_cosine_mean": _stat_mean(bucket, "head_tail_cosine_stats"),
                "adjacent_mean_cosine_mean": _stat_mean(bucket, "adjacent_mean_cosine_stats"),
                "adjacent_min_cosine_mean": _stat_mean(bucket, "adjacent_min_cosine_stats"),
                "summary_tail_cosine_mean": _stat_mean(bucket, "summary_tail_cosine_stats"),
                "possible_or_high_span_rate": bucket.get("possible_or_high_span_rate"),
            }
        )
    return rows


def plot_head_tail_distribution(rows: list[dict[str, Any]], output_base: Path) -> None:
    values = [float(row["head_tail_cosine"]) for row in rows if row.get("head_tail_cosine") is not None]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.hist(values, bins=24, range=(0.0, 1.0), color="#4C78A8", alpha=0.82, edgecolor="white")
    for threshold, label, color in [
        (0.70, "stable", "#2F7D32"),
        (0.55, "moderate", "#C47F00"),
        (0.40, "possible drift", "#A23B3B"),
    ]:
        ax.axvline(threshold, linestyle="--", linewidth=1.2, color=color, label=label)
    ax.set_title("Head-Tail Semantic Similarity")
    ax.set_xlabel("Head-tail cosine similarity")
    ax.set_ylabel("Trajectory count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_drift_by_length(rows: list[dict[str, Any]], output_base: Path) -> None:
    if not rows:
        return
    x = [row["length_bucket"] for row in rows]
    head_tail = [row.get("head_tail_cosine_mean") for row in rows]
    adjacent = [row.get("adjacent_mean_cosine_mean") for row in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(x, head_tail, marker="o", linewidth=2.0, label="Head-tail cosine")
    ax.plot(x, adjacent, marker="s", linewidth=2.0, label="Adjacent update cosine")
    ax.set_title("Semantic Drift by Trajectory Length")
    ax.set_xlabel("Trajectory length bucket")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def run(run_path: Path, output_dir: Path) -> dict[str, Any]:
    run_path = resolve_run_path(run_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(run_path / "summary.json")
    diagnostics = load_json(run_path / "analysis" / "trajectory_drift_diagnostics.json")
    drift_rows = load_jsonl(run_path / "analysis" / "trajectory_drift_rows.jsonl")

    distribution_rows = non_singleton_distribution_rows(drift_rows)
    by_length_rows = length_bucket_rows(diagnostics)

    write_csv(
        output_dir / "trajectory_head_tail_distribution.csv",
        distribution_rows,
        [
            "trajectory_id",
            "sample_id",
            "snapshot_count",
            "length_bucket",
            "drift_bucket",
            "head_tail_cosine",
            "adjacent_mean_cosine",
            "summary_tail_cosine",
        ],
    )
    write_csv(
        output_dir / "trajectory_drift_by_length.csv",
        by_length_rows,
        [
            "length_bucket",
            "trajectory_count",
            "head_tail_cosine_mean",
            "adjacent_mean_cosine_mean",
            "adjacent_min_cosine_mean",
            "summary_tail_cosine_mean",
            "possible_or_high_span_rate",
        ],
    )
    plot_head_tail_distribution(distribution_rows, output_dir / "trajectory_head_tail_distribution")
    plot_drift_by_length(by_length_rows, output_dir / "trajectory_drift_by_length")

    output_summary = {
        "schema_version": "trajectory_drift_plot_v1",
        "run_path": str(run_path),
        "run_id": summary.get("run_meta", {}).get("run_id") or run_path.name,
        "trajectory_count": diagnostics.get("trajectory_count"),
        "non_singleton_trajectory_count": diagnostics.get("non_singleton_trajectory_count"),
        "embedding_available_count": diagnostics.get("embedding_available_count"),
        "missing_embedding_count": diagnostics.get("missing_embedding_count"),
        "singleton_trajectory_rate": diagnostics.get("singleton_trajectory_rate"),
        "possible_or_high_span_rate": diagnostics.get("possible_or_high_span_rate"),
        "head_tail_cosine_stats": diagnostics.get("head_tail_cosine_stats"),
        "adjacent_mean_cosine_stats": diagnostics.get("adjacent_mean_cosine_stats"),
        "outputs": {
            "head_tail_distribution_pdf": str((output_dir / "trajectory_head_tail_distribution.pdf").resolve()),
            "head_tail_distribution_png": str((output_dir / "trajectory_head_tail_distribution.png").resolve()),
            "drift_by_length_pdf": str((output_dir / "trajectory_drift_by_length.pdf").resolve()),
            "drift_by_length_png": str((output_dir / "trajectory_drift_by_length.png").resolve()),
        },
    }
    (output_dir / "trajectory_drift_plot_summary.json").write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot offline trajectory semantic drift diagnostics.")
    parser.add_argument("--run-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("plot"), type=Path)
    args = parser.parse_args(argv)
    summary = run(args.run_path, args.output_dir)
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
