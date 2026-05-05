from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trajpatch-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trajpatch-cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_CUTOFFS = [5, 10, 15, 20, 30, 50]


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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile_nearest(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def direct_rows_by_query(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("query_task_id")): row for row in rows if row.get("query_task_id")}


def details_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    rows = details.get("samples")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def gold_trajectory_ids(
    row: dict[str, Any],
    direct_by_query: dict[str, dict[str, Any]],
    raw_ref_to_message_id: dict[tuple[str, str], str] | None = None,
    snapshots_by_trajectory: dict[str, list[tuple[str, int, set[str]]]] | None = None,
) -> set[str]:
    query_id = str(row.get("query_task_id") or "")
    direct = direct_by_query.get(query_id) or {}
    direct_ids = {str(value) for value in direct.get("gold_trajectory_ids") or [] if value}
    if direct_ids:
        return direct_ids
    if not raw_ref_to_message_id or not snapshots_by_trajectory:
        return set()
    sample_id = str(row.get("sample_id") or direct.get("sample_id") or "")
    gold_message_ids = {
        raw_ref_to_message_id[(sample_id, source_ref)]
        for source_ref in _query_gold_refs(row)
        if (sample_id, source_ref) in raw_ref_to_message_id
    }
    if not gold_message_ids:
        return set()
    trajectory_ids: set[str] = set()
    for trajectory_id, snapshots in snapshots_by_trajectory.items():
        if any(links & gold_message_ids for _snapshot_id, _version, links in snapshots):
            trajectory_ids.add(trajectory_id)
    return trajectory_ids


def retrieval_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get("retrieval_compact_diagnostics")
    return value if isinstance(value, dict) else {}


def compute_page_cutoff_rows(
    rows: list[dict[str, Any]],
    direct_by_query: dict[str, dict[str, Any]],
    *,
    raw_ref_to_message_id: dict[tuple[str, str], str] | None = None,
    snapshots_by_trajectory: dict[str, list[tuple[str, int, set[str]]]] | None = None,
    cutoffs: list[int] = DEFAULT_CUTOFFS,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    query_count = sum(
        1
        for row in rows
        if gold_trajectory_ids(row, direct_by_query, raw_ref_to_message_id, snapshots_by_trajectory)
    )
    for cutoff in cutoffs:
        recalls: list[float] = []
        all_gold: list[float] = []
        universe_sizes: list[float] = []
        not_observed = 0
        for row in rows:
            gold = gold_trajectory_ids(row, direct_by_query, raw_ref_to_message_id, snapshots_by_trajectory)
            if not gold:
                continue
            metadata = retrieval_metadata(row)
            page_rows = metadata.get("page_ranked_rows_compact_top_n") or []
            total_count = int(metadata.get("page_ranked_total_count") or len(page_rows))
            required_saved_count = min(cutoff, total_count)
            if len(page_rows) < required_saved_count:
                not_observed += 1
                continue
            universe: set[str] = set()
            for page_row in page_rows[:required_saved_count]:
                universe.update(str(value) for value in page_row.get("trajectory_ids") or [] if value)
            recalls.append(len(gold & universe) / len(gold))
            all_gold.append(float(gold <= universe))
            universe_sizes.append(float(len(universe)))
        output.append(
            {
                "cutoff": cutoff,
                "query_count": query_count,
                "observed_query_count": len(recalls),
                "not_observed_query_count": not_observed,
                "mean_gold_trajectory_recall": mean(recalls),
                "all_gold_trajectory_coverage_rate": mean(all_gold),
                "mean_candidate_universe_size": mean(universe_sizes),
            }
        )
    return output


def compute_trajectory_cutoff_rows(
    rows: list[dict[str, Any]],
    direct_by_query: dict[str, dict[str, Any]],
    *,
    raw_ref_to_message_id: dict[tuple[str, str], str] | None = None,
    snapshots_by_trajectory: dict[str, list[tuple[str, int, set[str]]]] | None = None,
    cutoffs: list[int] = DEFAULT_CUTOFFS,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    query_count = sum(
        1
        for row in rows
        if gold_trajectory_ids(row, direct_by_query, raw_ref_to_message_id, snapshots_by_trajectory)
    )
    for cutoff in cutoffs:
        recalls: list[float] = []
        all_gold: list[float] = []
        prefix_sizes: list[float] = []
        not_observed = 0
        for row in rows:
            gold = gold_trajectory_ids(row, direct_by_query, raw_ref_to_message_id, snapshots_by_trajectory)
            if not gold:
                continue
            prefix = retrieval_metadata(row).get("trajectory_cutoff_prefix_diagnostics") or {}
            cutoff_row = prefix.get(str(cutoff))
            if not isinstance(cutoff_row, dict):
                not_observed += 1
                continue
            ids = {str(value) for value in cutoff_row.get("trajectory_ids") or [] if value}
            recalls.append(len(gold & ids) / len(gold))
            all_gold.append(float(gold <= ids))
            prefix_sizes.append(float(len(ids)))
        output.append(
            {
                "cutoff": cutoff,
                "query_count": query_count,
                "observed_query_count": len(recalls),
                "not_observed_query_count": not_observed,
                "mean_gold_trajectory_recall": mean(recalls),
                "all_gold_trajectory_coverage_rate": mean(all_gold),
                "mean_prefix_size": mean(prefix_sizes),
            }
        )
    return output


def _first_int_value(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return int(value)
    return None


def _query_gold_refs(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return set()
    query_metadata = metadata.get("query_metadata")
    if not isinstance(query_metadata, dict):
        return set()
    return {str(value) for value in query_metadata.get("gold_evidence_refs") or [] if value}


def _load_sqlite_snapshot_indexes(run_path: Path) -> tuple[
    dict[tuple[str, str], str],
    dict[str, list[tuple[str, int, set[str]]]],
    dict[str, int],
]:
    database_path = run_path / "trajpatch.sqlite"
    if not database_path.exists():
        return {}, {}, {}
    conn = sqlite3.connect(database_path)
    try:
        raw_ref_to_message_id: dict[tuple[str, str], str] = {}
        for message_id, sample_id, source_ref in conn.execute(
            "select id, sample_id, source_ref from raw_messages"
        ):
            if sample_id and source_ref:
                raw_ref_to_message_id[(str(sample_id), str(source_ref))] = str(message_id)

        snapshots_by_trajectory: dict[str, list[tuple[str, int, set[str]]]] = {}
        for snapshot_id, trajectory_id, version, links_json in conn.execute(
            "select id, trajectory_id, version, links_json from episodic_snapshots"
        ):
            try:
                links = set(json.loads(links_json or "[]"))
            except json.JSONDecodeError:
                links = set()
            snapshots_by_trajectory.setdefault(str(trajectory_id), []).append(
                (str(snapshot_id), int(version), {str(link) for link in links})
            )
        for values in snapshots_by_trajectory.values():
            values.sort(key=lambda item: item[1])

        trajectory_lengths = {
            str(trajectory_id): int(snapshot_count)
            for trajectory_id, snapshot_count in conn.execute("select id, snapshot_count from trajectories")
        }
    finally:
        conn.close()
    return raw_ref_to_message_id, snapshots_by_trajectory, trajectory_lengths


def compute_required_snapshot_ranks(
    rows: list[dict[str, Any]],
    direct_by_query: dict[str, dict[str, Any]],
    run_path: Path,
) -> tuple[list[int], dict[str, int]]:
    raw_ref_to_message_id, snapshots_by_trajectory, trajectory_lengths = _load_sqlite_snapshot_indexes(
        run_path
    )
    ranks: list[int] = []
    method_counts: dict[str, int] = {
        "details_or_direct_field": 0,
        "sqlite_gold_ref_links": 0,
        "trajectory_length_fallback": 0,
        "missing": 0,
    }
    for row in rows:
        query_id = str(row.get("query_task_id") or "")
        direct = direct_by_query.get(query_id) or {}
        field_value = _first_int_value(
            row.get("max_gold_snapshot_rank_required"),
            direct.get("max_gold_snapshot_rank_required"),
        )
        if field_value is not None:
            ranks.append(max(1, field_value))
            method_counts["details_or_direct_field"] += 1
            continue

        gold_trajectories = gold_trajectory_ids(
            row,
            direct_by_query,
            raw_ref_to_message_id,
            snapshots_by_trajectory,
        )
        if not gold_trajectories:
            method_counts["missing"] += 1
            continue
        sample_id = str(row.get("sample_id") or direct.get("sample_id") or "")
        gold_message_ids = {
            raw_ref_to_message_id[(sample_id, source_ref)]
            for source_ref in _query_gold_refs(row)
            if (sample_id, source_ref) in raw_ref_to_message_id
        }
        matched_versions: list[int] = []
        if gold_message_ids:
            for trajectory_id in gold_trajectories:
                for _snapshot_id, version, links in snapshots_by_trajectory.get(trajectory_id, []):
                    if links & gold_message_ids:
                        matched_versions.append(version)
        if matched_versions:
            ranks.append(max(matched_versions))
            method_counts["sqlite_gold_ref_links"] += 1
            continue

        lengths = [
            trajectory_lengths[trajectory_id]
            for trajectory_id in gold_trajectories
            if trajectory_id in trajectory_lengths
        ]
        if lengths:
            ranks.append(max(lengths))
            method_counts["trajectory_length_fallback"] += 1
        else:
            method_counts["missing"] += 1
    return ranks, method_counts


def compute_snapshot_cdf_rows(
    ranks: list[int],
    *,
    m_value: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not ranks:
        return [], {
            "query_count": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "m": m_value,
            "coverage_at_m": None,
        }
    ordered = sorted(max(1, int(value)) for value in ranks)
    max_rank = ordered[-1]
    rows: list[dict[str, Any]] = []
    covered = 0
    index = 0
    for rank in range(1, max_rank + 1):
        while index < len(ordered) and ordered[index] <= rank:
            covered += 1
            index += 1
        rows.append(
            {
                "required_snapshot_rank": rank,
                "covered_query_count": covered,
                "query_count": len(ordered),
                "query_coverage": covered / len(ordered),
            }
        )
    coverage_at_m = None
    if m_value is not None:
        coverage_at_m = sum(1 for value in ordered if value <= m_value) / len(ordered)
    summary = {
        "query_count": len(ordered),
        "median": percentile_nearest(ordered, 0.50),
        "p90": percentile_nearest(ordered, 0.90),
        "p95": percentile_nearest(ordered, 0.95),
        "max": max_rank,
        "m": m_value,
        "coverage_at_m": coverage_at_m,
    }
    return rows, summary


def compute_snapshot_marginal_rows(cdf_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_covered = 0
    for row in cdf_rows:
        covered = int(row.get("covered_query_count") or 0)
        query_count = int(row.get("query_count") or 0)
        newly_covered = max(0, covered - previous_covered)
        previous_covered = covered
        rows.append(
            {
                "required_snapshot_rank": int(row["required_snapshot_rank"]),
                "newly_covered_query_count": newly_covered,
                "query_count": query_count,
                "marginal_coverage_gain": (newly_covered / query_count) if query_count else None,
                "cumulative_covered_query_count": covered,
                "cumulative_query_coverage": row.get("query_coverage"),
            }
        )
    return rows


def compute_snapshot_marginal_summary(
    marginal_rows: list[dict[str, Any]],
    cdf_summary: dict[str, Any],
) -> dict[str, Any]:
    if not marginal_rows:
        return {
            "query_count": 0,
            "max_rank": None,
            "peak_marginal_rank": None,
            "peak_marginal_query_count": None,
            "coverage_at_5": None,
            "coverage_at_10": None,
            "coverage_at_m": cdf_summary.get("coverage_at_m"),
        }
    peak = max(
        marginal_rows,
        key=lambda row: (
            int(row.get("newly_covered_query_count") or 0),
            -int(row.get("required_snapshot_rank") or 0),
        ),
    )

    def coverage_at(rank: int) -> float | None:
        candidates = [
            row
            for row in marginal_rows
            if int(row.get("required_snapshot_rank") or 0) <= rank
        ]
        if not candidates:
            return None
        return float(candidates[-1]["cumulative_query_coverage"])

    return {
        "query_count": int(marginal_rows[0].get("query_count") or 0),
        "max_rank": int(marginal_rows[-1]["required_snapshot_rank"]),
        "peak_marginal_rank": int(peak["required_snapshot_rank"]),
        "peak_marginal_query_count": int(peak["newly_covered_query_count"]),
        "coverage_at_5": coverage_at(5),
        "coverage_at_10": coverage_at(10),
        "coverage_at_m": cdf_summary.get("coverage_at_m"),
    }


def _none_to_nan(values: list[Any]) -> list[float]:
    return [float("nan") if value is None else float(value) for value in values]


def plot_cutoff_lines(
    rows: list[dict[str, Any]],
    *,
    output_base: Path,
    title: str,
    xlabel: str,
) -> None:
    x = [int(row["cutoff"]) for row in rows]
    recall = _none_to_nan([row.get("mean_gold_trajectory_recall") for row in rows])
    all_gold = _none_to_nan([row.get("all_gold_trajectory_coverage_rate") for row in rows])

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(x, recall, marker="o", linewidth=2.0, label="Mean gold trajectory recall")
    ax.plot(x, all_gold, marker="s", linewidth=2.0, label="All-gold trajectory coverage")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_snapshot_cdf(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_base: Path,
) -> None:
    if not rows:
        return
    x = [int(row["required_snapshot_rank"]) for row in rows]
    y = [float(row["query_coverage"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.step(x, y, where="post", linewidth=2.1, label="Gold evidence coverage")
    m_value = summary.get("m")
    if isinstance(m_value, int):
        ax.axvline(m_value, color="#b23a48", linestyle="--", linewidth=1.5)
    for label, color in [("median", "#555555"), ("p90", "#777777"), ("p95", "#999999")]:
        value = summary.get(label)
        if isinstance(value, int):
            ax.axvline(value, color=color, linestyle=":", linewidth=1.2)
            ax.text(value, 0.08, label, rotation=90, va="bottom", ha="right", fontsize=8, color=color)
    ax.set_title("Gold Snapshot Rank CDF")
    ax.set_xlabel("Required snapshot rank")
    ax.set_ylabel("Query coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(1, max(x))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_snapshot_marginal_utility(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_base: Path,
) -> None:
    if not rows:
        return
    x = [int(row["required_snapshot_rank"]) for row in rows]
    y = [int(row["newly_covered_query_count"]) for row in rows]
    m_value = summary.get("m")
    colors = ["#b23a48" if rank == m_value else "#2f6f9f" for rank in x]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(x, y, color=colors, width=0.72)
    ax.set_title("Marginal Utility of Snapshot Budget m")
    ax.set_xlabel("Required snapshot rank")
    ax.set_ylabel("Newly covered queries")
    ax.set_xticks(x)
    ax.set_xlim(min(x) - 0.7, max(x) + 0.7)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def run(run_path: Path, output_dir: Path) -> dict[str, Any]:
    run_path = resolve_run_path(run_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(run_path / "summary.json")
    details = load_json(run_path / "details.json")
    rows = details_rows(details)
    direct = load_jsonl(run_path / "analysis" / "direct_retrieval_rows.jsonl")
    direct_by_query = direct_rows_by_query(direct)
    m_value = summary.get("run_meta", {}).get("m")
    m_int = int(m_value) if isinstance(m_value, int) else None
    raw_ref_to_message_id, snapshots_by_trajectory, _trajectory_lengths = _load_sqlite_snapshot_indexes(
        run_path
    )

    page_rows = compute_page_cutoff_rows(
        rows,
        direct_by_query,
        raw_ref_to_message_id=raw_ref_to_message_id,
        snapshots_by_trajectory=snapshots_by_trajectory,
    )
    trajectory_rows = compute_trajectory_cutoff_rows(
        rows,
        direct_by_query,
        raw_ref_to_message_id=raw_ref_to_message_id,
        snapshots_by_trajectory=snapshots_by_trajectory,
    )
    snapshot_ranks, rank_method_counts = compute_required_snapshot_ranks(rows, direct_by_query, run_path)
    cdf_rows, cdf_summary = compute_snapshot_cdf_rows(snapshot_ranks, m_value=m_int)
    cdf_summary["rank_source_method_counts"] = rank_method_counts
    marginal_rows = compute_snapshot_marginal_rows(cdf_rows)
    marginal_summary = compute_snapshot_marginal_summary(marginal_rows, cdf_summary)

    write_csv(
        output_dir / "hyperparam_page_cutoff.csv",
        page_rows,
        [
            "cutoff",
            "query_count",
            "observed_query_count",
            "not_observed_query_count",
            "mean_gold_trajectory_recall",
            "all_gold_trajectory_coverage_rate",
            "mean_candidate_universe_size",
        ],
    )
    write_csv(
        output_dir / "hyperparam_trajectory_k.csv",
        trajectory_rows,
        [
            "cutoff",
            "query_count",
            "observed_query_count",
            "not_observed_query_count",
            "mean_gold_trajectory_recall",
            "all_gold_trajectory_coverage_rate",
            "mean_prefix_size",
        ],
    )
    write_csv(
        output_dir / "hyperparam_m_snapshot_cdf.csv",
        cdf_rows,
        [
            "required_snapshot_rank",
            "covered_query_count",
            "query_count",
            "query_coverage",
        ],
    )
    write_csv(
        output_dir / "hyperparam_m_marginal_utility.csv",
        marginal_rows,
        [
            "required_snapshot_rank",
            "newly_covered_query_count",
            "query_count",
            "marginal_coverage_gain",
            "cumulative_covered_query_count",
            "cumulative_query_coverage",
        ],
    )

    plot_cutoff_lines(
        page_rows,
        output_base=output_dir / "hyperparam_page_cutoff",
        title="Page Routing Cutoff",
        xlabel="t-pages",
    )
    plot_cutoff_lines(
        trajectory_rows,
        output_base=output_dir / "hyperparam_trajectory_k",
        title="Trajectory Top-k Cutoff",
        xlabel="k trajectories",
    )
    plot_snapshot_cdf(cdf_rows, cdf_summary, output_base=output_dir / "hyperparam_m_snapshot_cdf")
    plot_snapshot_marginal_utility(
        marginal_rows,
        {**marginal_summary, "m": m_int},
        output_base=output_dir / "hyperparam_m_marginal_utility",
    )

    output_summary = {
        "schema_version": "hyperparameter_analysis_plot_v1",
        "run_path": str(run_path),
        "run_id": summary.get("run_meta", {}).get("run_id") or run_path.name,
        "m": m_int,
        "page_cutoff_rows": page_rows,
        "trajectory_cutoff_rows": trajectory_rows,
        "snapshot_cdf_summary": cdf_summary,
        "snapshot_marginal_summary": marginal_summary,
        "outputs": {
            "page_cutoff_pdf": str((output_dir / "hyperparam_page_cutoff.pdf").resolve()),
            "page_cutoff_png": str((output_dir / "hyperparam_page_cutoff.png").resolve()),
            "trajectory_k_pdf": str((output_dir / "hyperparam_trajectory_k.pdf").resolve()),
            "trajectory_k_png": str((output_dir / "hyperparam_trajectory_k.png").resolve()),
            "snapshot_cdf_pdf": str((output_dir / "hyperparam_m_snapshot_cdf.pdf").resolve()),
            "snapshot_cdf_png": str((output_dir / "hyperparam_m_snapshot_cdf.png").resolve()),
            "snapshot_marginal_pdf": str(
                (output_dir / "hyperparam_m_marginal_utility.pdf").resolve()
            ),
            "snapshot_marginal_png": str(
                (output_dir / "hyperparam_m_marginal_utility.png").resolve()
            ),
        },
    }
    (output_dir / "hyperparam_analysis_summary.json").write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot offline TrajWiki hyperparameter diagnostics.")
    parser.add_argument("--run-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("plot"), type=Path)
    args = parser.parse_args(argv)
    summary = run(args.run_path, args.output_dir)
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
