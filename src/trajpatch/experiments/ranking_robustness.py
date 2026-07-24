"""Offline sensitivity analysis for saved deterministic retrieval scores."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean
from typing import Any

from trajpatch.analysis.gold_labels import (
    build_memory_index,
    load_details_rows,
    load_or_build_gold_labels,
)
from trajpatch.analysis.offline_ablation import _load_retrieval_events
from trajpatch.analysis.query_scope import (
    filter_query_rows,
    load_query_scope,
    scoped_analysis_dir,
    validate_scope_against_rows,
)
from trajpatch.utils.json_utils import write_json

PERTURBED_WEIGHTS = [
    "summary_similarity_weight",
    "latest_similarity_weight",
    "metadata_signal_weight",
    "dense_family_weight",
    "dense_source_event_weight",
    "sparse_family_weight",
    "sparse_source_event_weight",
]
PAGE_PERTURBED_WEIGHTS = [
    "page_entity_bonus_weight",
    "page_reflection_bonus_weight",
    "page_granularity_weight",
    "page_family_weight",
    "page_mismatch_weight",
]
RRF_K = 60


def _parse_cutoffs(
    value: str | list[int] | tuple[int, ...],
    *,
    default: list[int],
) -> list[int]:
    if isinstance(value, str):
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        parsed = [int(item) for item in value]
    values = sorted({item for item in parsed if item > 0})
    return values or list(default)


def _rank_id(row: dict[str, Any]) -> str:
    return str(row.get("trajectory_id") or row.get("item_id") or "")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _trajectory_score_terms(row: dict[str, Any]) -> dict[str, float]:
    components = dict(row.get("score_components") or {})
    summary = float(components.get("summary_similarity") or 0.0)
    latest = float(components.get("latest_similarity") or 0.0)
    metadata_signal = min(
        1.0,
        (
            float(components.get("entity_match_boost") or 0.0)
            + float(components.get("facet_tag_boost") or 0.0)
            + float(components.get("facet_value_boost") or 0.0)
        )
        / 0.20,
    )
    family = float(components.get("answer_family_match_score") or 0.0)
    source_event = float(components.get("source_event_match_score") or 0.0)
    dense_score = float(components.get("dense_score") or 0.0)
    sparse_score = float(components.get("sparse_score") or 0.0)
    known_dense = (
        0.75 * summary
        + 0.15 * latest
        + 0.10 * metadata_signal
        + 0.08 * family
        + 0.10 * source_event
    )
    known_sparse = 0.60 * family + 0.75 * source_event
    return {
        "summary": summary,
        "latest": latest,
        "metadata_signal": metadata_signal,
        "family": family,
        "source_event": source_event,
        # Residuals preserve lexical evidence, mismatch penalties, and clipping,
        # whose internal terms are not fully recoverable from compact v1 rows.
        "dense_residual": dense_score - known_dense,
        "sparse_residual": sparse_score - known_sparse,
    }


def _rerank_with_multipliers(
    ranked_rows: list[dict[str, Any]],
    multipliers: dict[str, float],
) -> list[dict[str, Any]]:
    rescored: list[dict[str, Any]] = []
    for row in ranked_rows:
        terms = _trajectory_score_terms(row)
        dense_score = max(
            0.0,
            terms["dense_residual"]
            + 0.75 * multipliers["summary_similarity_weight"] * terms["summary"]
            + 0.15 * multipliers["latest_similarity_weight"] * terms["latest"]
            + 0.10 * multipliers["metadata_signal_weight"] * terms["metadata_signal"]
            + 0.08 * multipliers["dense_family_weight"] * terms["family"]
            + 0.10 * multipliers["dense_source_event_weight"] * terms["source_event"],
        )
        sparse_score = max(
            0.0,
            terms["sparse_residual"]
            + 0.60 * multipliers["sparse_family_weight"] * terms["family"]
            + 0.75 * multipliers["sparse_source_event_weight"] * terms["source_event"],
        )
        rescored.append(
            {
                **row,
                "_perturbed_dense_score": dense_score,
                "_perturbed_sparse_score": sparse_score,
            }
        )
    dense_ranked = sorted(
        rescored,
        key=lambda row: (float(row["_perturbed_dense_score"]), _rank_id(row)),
        reverse=True,
    )
    sparse_ranked = sorted(
        rescored,
        key=lambda row: (float(row["_perturbed_sparse_score"]), _rank_id(row)),
        reverse=True,
    )
    dense_rank = {_rank_id(row): rank for rank, row in enumerate(dense_ranked, start=1)}
    sparse_rank = {
        _rank_id(row): rank for rank, row in enumerate(sparse_ranked, start=1)
    }
    for row in rescored:
        item_id = _rank_id(row)
        row["_perturbed_fused_score"] = 1.0 / (RRF_K + dense_rank[item_id]) + 1.0 / (
            RRF_K + sparse_rank[item_id]
        )
    return sorted(
        rescored,
        key=lambda row: (
            float(row["_perturbed_fused_score"]),
            float(row["_perturbed_dense_score"]),
            float(row["_perturbed_sparse_score"]),
        ),
        reverse=True,
    )


def _rerank_pages_with_multipliers(
    ranked_rows: list[dict[str, Any]],
    multipliers: dict[str, float],
) -> list[dict[str, Any]]:
    rescored: list[dict[str, Any]] = []
    for row in ranked_rows:
        components = dict(row.get("score_components") or {})
        dense_score = float(components.get("dense_score") or 0.0)
        sparse_score = float(components.get("sparse_score") or 0.0)
        entity_bonus = float(components.get("entity_bonus") or 0.0)
        reflection_bonus = float(components.get("reflection_bonus") or 0.0)
        granularity = float(
            components.get("page_granularity_adjustment") or 0.0
        )
        family = float(components.get("page_family_match_score") or 0.0)
        mismatch = float(
            components.get("page_family_mismatch_penalty") or 0.0
        )
        family_dense = min(0.10, family * 0.10)
        family_sparse = min(0.35, family * 0.35)
        mismatch_dense = min(0.04, mismatch * 0.12)
        mismatch_sparse = min(0.10, mismatch * 0.25)
        dense_residual = (
            dense_score
            - entity_bonus
            - reflection_bonus
            - granularity
            - family_dense
            + mismatch_dense
        )
        sparse_residual = (
            sparse_score
            - family_sparse
            + mismatch_sparse
        )
        rescored.append(
            {
                **row,
                "_perturbed_dense_score": (
                    dense_residual
                    + multipliers["page_entity_bonus_weight"] * entity_bonus
                    + multipliers["page_reflection_bonus_weight"]
                    * reflection_bonus
                    + multipliers["page_granularity_weight"] * granularity
                    + multipliers["page_family_weight"] * family_dense
                    - multipliers["page_mismatch_weight"] * mismatch_dense
                ),
                "_perturbed_sparse_score": max(
                    0.0,
                    sparse_residual
                    + multipliers["page_family_weight"] * family_sparse
                    - multipliers["page_mismatch_weight"] * mismatch_sparse,
                ),
            }
        )
    dense_ranked = sorted(
        rescored,
        key=lambda row: (
            float(row["_perturbed_dense_score"]),
            str(row.get("page_id") or row.get("item_id") or ""),
        ),
        reverse=True,
    )
    sparse_ranked = sorted(
        rescored,
        key=lambda row: (
            float(row["_perturbed_sparse_score"]),
            str(row.get("page_id") or row.get("item_id") or ""),
        ),
        reverse=True,
    )
    dense_rank = {
        str(row.get("page_id") or row.get("item_id") or ""): rank
        for rank, row in enumerate(dense_ranked, start=1)
    }
    sparse_rank = {
        str(row.get("page_id") or row.get("item_id") or ""): rank
        for rank, row in enumerate(sparse_ranked, start=1)
    }
    for row in rescored:
        item_id = str(row.get("page_id") or row.get("item_id") or "")
        row["_perturbed_fused_score"] = (
            1.0 / (RRF_K + dense_rank[item_id])
            + 1.0 / (RRF_K + sparse_rank[item_id])
        )
    return sorted(
        rescored,
        key=lambda row: (
            float(row["_perturbed_fused_score"]),
            float(row["_perturbed_dense_score"]),
            float(row["_perturbed_sparse_score"]),
        ),
        reverse=True,
    )


def analyze_ranking_robustness(
    run_path: Path | str,
    *,
    relative_perturbation: float = 0.20,
    random_draws: int = 100,
    seed: int = 7,
    cutoff: int = 15,
    page_cutoffs: str | list[int] | tuple[int, ...] = "5,10,15",
    trajectory_cutoffs: str | list[int] | tuple[int, ...] = "5,10,15,20,30",
    sampling_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    if relative_perturbation < 0:
        raise ValueError("relative_perturbation must be non-negative.")
    if random_draws <= 0:
        raise ValueError("random_draws must be positive.")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    run_dir = Path(run_path).expanduser().resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    _, sample_rows, database_path = load_details_rows(run_dir)
    scope = load_query_scope(sampling_manifest_path)
    if scope is not None:
        validate_scope_against_rows(scope, sample_rows, run_dir=run_dir)
        sample_rows = filter_query_rows(sample_rows, scope)
    memory_index = build_memory_index(database_path)
    gold_rows = load_or_build_gold_labels(run_dir, sample_rows, memory_index)
    gold_by_query = {str(row.get("query_task_id")): row for row in gold_rows}
    events = _load_retrieval_events(database_path, memory_index)
    rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []
    incomplete_rankings: list[str] = []
    incomplete_page_rankings: list[str] = []
    resolved_page_cutoffs = _parse_cutoffs(
        page_cutoffs,
        default=[5, 10, 15],
    )
    resolved_trajectory_cutoffs = _parse_cutoffs(
        trajectory_cutoffs,
        default=[5, 10, 15, 20, 30],
    )
    for sample_row in sample_rows:
        query_task_id = str(sample_row.get("query_task_id") or "")
        event = events.get(str(sample_row.get("retrieval_event_id") or ""), {})
        event_metadata = dict(event.get("metadata") or {})
        ranked_rows = list(
            event_metadata.get("ablation_trajectory_ranked_rows_v1") or []
        )
        ranked_rows = [row for row in ranked_rows if _rank_id(row)]
        ranked_total = event_metadata.get("trajectory_ranked_total_count")
        full_ranking_verified = bool(
            ranked_rows
            and not event_metadata.get("trajectory_ranked_rows_truncated")
            and (
                (
                    ranked_total is not None
                    and len(ranked_rows) >= int(ranked_total or 0)
                )
                or event_metadata.get("retrieval_rank_save_mode") == "full"
            )
        )
        if not full_ranking_verified:
            incomplete_rankings.append(query_task_id)
            continue
        baseline_ids = [_rank_id(row) for row in ranked_rows[:cutoff]]
        gold = gold_by_query.get(query_task_id, {})
        gold_trajectories = set(gold.get("gold_trajectory_ids") or [])
        gold_refs = set(gold.get("gold_source_refs") or [])
        page_rows = list(event_metadata.get("ablation_page_ranked_rows_v1") or [])
        page_rows = [
            row
            for row in page_rows
            if str(row.get("page_id") or row.get("item_id") or "")
        ]
        page_ranked_total = event_metadata.get("page_ranked_total_count")
        full_page_ranking_verified = bool(
            page_rows
            and not event_metadata.get("page_ranked_rows_truncated")
            and (
                (
                    page_ranked_total is not None
                    and len(page_rows) >= int(page_ranked_total or 0)
                )
                or event_metadata.get("retrieval_rank_save_mode") == "full"
            )
        )
        if not full_page_ranking_verified:
            incomplete_page_rankings.append(query_task_id)
            continue
        observed_page_cutoff = int(
            event.get("top_t_pages")
            or event_metadata.get("source_run_top_t_pages")
            or 0
        )
        effective_page_cutoff = min(
            observed_page_cutoff or max(resolved_page_cutoffs),
            len(page_rows),
        )
        baseline_page_ids = {
            str(row.get("page_id") or row.get("item_id") or "")
            for row in page_rows[:effective_page_cutoff]
        }
        reconstructed_pages = _rerank_pages_with_multipliers(
            page_rows,
            {name: 1.0 for name in PAGE_PERTURBED_WEIGHTS},
        )
        page_reconstruction_jaccard = _jaccard(
            baseline_page_ids,
            {
                str(row.get("page_id") or row.get("item_id") or "")
                for row in reconstructed_pages[:effective_page_cutoff]
            },
        )
        for page_cutoff in resolved_page_cutoffs:
            page_replay_observable = bool(
                page_rows
                and observed_page_cutoff
                and page_cutoff <= observed_page_cutoff
            )
            candidate_ids = {
                str(trajectory_id)
                for page_row in page_rows[:page_cutoff]
                for trajectory_id in list(
                    page_row.get("linked_trajectory_ids")
                    or page_row.get("trajectory_ids")
                    or []
                )
                if str(trajectory_id)
            }
            filtered_ranked = [
                row for row in ranked_rows if _rank_id(row) in candidate_ids
            ]
            for trajectory_cutoff in resolved_trajectory_cutoffs:
                selected_ids = {
                    _rank_id(row)
                    for row in filtered_ranked[:trajectory_cutoff]
                }
                selected_refs = set().union(
                    *(
                        memory_index["trajectory_refs"].get(
                            trajectory_id,
                            set(),
                        )
                        for trajectory_id in selected_ids
                    )
                )
                baseline_at_k = {
                    _rank_id(row) for row in ranked_rows[:trajectory_cutoff]
                }
                cutoff_rows.append(
                    {
                        "sample_id": sample_row.get("sample_id"),
                        "query_task_id": query_task_id,
                        "page_cutoff": page_cutoff,
                        "trajectory_cutoff": trajectory_cutoff,
                        "observed_page_cutoff": observed_page_cutoff,
                        "replay_observable": page_replay_observable,
                        "candidate_universe_size": (
                            len(candidate_ids)
                            if page_replay_observable
                            else None
                        ),
                        "gold_trajectory_recall": (
                            (
                                len(selected_ids & gold_trajectories)
                                / len(gold_trajectories)
                                if gold_trajectories
                                else 1.0
                            )
                            if page_replay_observable
                            else None
                        ),
                        "gold_ref_coverage": (
                            (
                                len(selected_refs & gold_refs) / len(gold_refs)
                                if gold_refs
                                else 1.0
                            )
                            if page_replay_observable
                            else None
                        ),
                        "top_k_jaccard_vs_observed_candidate_ranking": (
                            _jaccard(baseline_at_k, selected_ids)
                            if page_replay_observable
                            else None
                        ),
                        "fixed_saved_page_scores": True,
                        "fixed_saved_trajectory_scores": True,
                        "llm_rerank_not_reexecuted": True,
                    }
                )
        draw_recalls: list[float] = []
        draw_ref_coverages: list[float] = []
        draw_jaccards: list[float] = []
        page_draw_recalls: list[float] = []
        page_draw_ref_coverages: list[float] = []
        page_draw_jaccards: list[float] = []
        page_draw_candidate_sizes: list[float] = []
        rng = random.Random(f"{seed}:{query_task_id}")
        page_rng = random.Random(f"{seed}:{query_task_id}:page")
        reconstructed = _rerank_with_multipliers(
            ranked_rows,
            {name: 1.0 for name in PERTURBED_WEIGHTS},
        )
        reconstruction_jaccard = _jaccard(
            set(baseline_ids),
            {_rank_id(row) for row in reconstructed[:cutoff]},
        )
        for _ in range(max(1, int(random_draws))):
            multipliers = {
                name: 1.0
                + rng.uniform(-abs(relative_perturbation), abs(relative_perturbation))
                for name in PERTURBED_WEIGHTS
            }
            perturbed = _rerank_with_multipliers(
                ranked_rows,
                multipliers,
            )
            selected = {_rank_id(row) for row in perturbed[:cutoff]}
            selected_refs = set().union(
                *(
                    memory_index["trajectory_refs"].get(trajectory_id, set())
                    for trajectory_id in selected
                )
            )
            draw_recalls.append(
                len(selected & gold_trajectories) / len(gold_trajectories)
                if gold_trajectories
                else 1.0
            )
            draw_ref_coverages.append(
                len(selected_refs & gold_refs) / len(gold_refs) if gold_refs else 1.0
            )
            draw_jaccards.append(_jaccard(set(baseline_ids), selected))
            page_multipliers = {
                name: 1.0
                + page_rng.uniform(
                    -abs(relative_perturbation),
                    abs(relative_perturbation),
                )
                for name in PAGE_PERTURBED_WEIGHTS
            }
            perturbed_pages = _rerank_pages_with_multipliers(
                page_rows,
                page_multipliers,
            )
            selected_page_ids = {
                str(row.get("page_id") or row.get("item_id") or "")
                for row in perturbed_pages[:effective_page_cutoff]
            }
            selected_page_rows = [
                row
                for row in perturbed_pages[:effective_page_cutoff]
                if str(row.get("page_id") or row.get("item_id") or "")
                in selected_page_ids
            ]
            page_candidate_ids = {
                str(trajectory_id)
                for page_row in selected_page_rows
                for trajectory_id in list(
                    page_row.get("linked_trajectory_ids")
                    or page_row.get("trajectory_ids")
                    or []
                )
                if str(trajectory_id)
            }
            page_candidate_refs = set().union(
                *(
                    memory_index["trajectory_refs"].get(
                        trajectory_id,
                        set(),
                    )
                    for trajectory_id in page_candidate_ids
                )
            )
            page_draw_recalls.append(
                len(page_candidate_ids & gold_trajectories)
                / len(gold_trajectories)
                if gold_trajectories
                else 1.0
            )
            page_draw_ref_coverages.append(
                len(page_candidate_refs & gold_refs) / len(gold_refs)
                if gold_refs
                else 1.0
            )
            page_draw_jaccards.append(
                _jaccard(baseline_page_ids, selected_page_ids)
            )
            page_draw_candidate_sizes.append(float(len(page_candidate_ids)))
        rows.append(
            {
                "sample_id": sample_row.get("sample_id"),
                "query_task_id": query_task_id,
                "cutoff": cutoff,
                "perturbed_weight_count": len(PERTURBED_WEIGHTS),
                "random_draws": random_draws,
                "relative_perturbation": relative_perturbation,
                "baseline_reconstruction_jaccard": reconstruction_jaccard,
                "page_baseline_reconstruction_jaccard": (
                    page_reconstruction_jaccard
                ),
                "mean_gold_trajectory_recall": mean(draw_recalls),
                "mean_gold_ref_coverage": mean(draw_ref_coverages),
                "mean_top_k_jaccard": mean(draw_jaccards),
                "min_top_k_jaccard": min(draw_jaccards),
                "mean_page_candidate_gold_trajectory_recall": mean(
                    page_draw_recalls
                ),
                "mean_page_candidate_gold_ref_coverage": mean(
                    page_draw_ref_coverages
                ),
                "mean_page_top_t_jaccard": mean(page_draw_jaccards),
                "min_page_top_t_jaccard": min(page_draw_jaccards),
                "mean_page_candidate_universe_size": mean(
                    page_draw_candidate_sizes
                ),
                "llm_rerank_held_fixed": True,
            }
        )
    if incomplete_rankings:
        raise ValueError(
            "Ranking robustness requires an untruncated full trajectory ranking for "
            f"every query; missing or incomplete={len(incomplete_rankings)}, "
            f"examples={incomplete_rankings[:5]}. Run the benchmark with "
            "--ablation-diagnostics --retrieval-rank-save-mode full."
        )
    if incomplete_page_rankings:
        raise ValueError(
            "Ranking robustness requires an untruncated full page ranking for "
            f"every query; missing or incomplete={len(incomplete_page_rankings)}, "
            f"examples={incomplete_page_rankings[:5]}. Run the benchmark with "
            "--ablation-diagnostics --retrieval-rank-save-mode full."
        )
    reconstruction_mismatches = [
        row
        for row in rows
        if float(row.get("baseline_reconstruction_jaccard") or 0.0) < 1.0
    ]
    if reconstruction_mismatches:
        examples = [
            str(row.get("query_task_id") or "") for row in reconstruction_mismatches[:5]
        ]
        raise ValueError(
            "Saved compact score rows do not exactly reconstruct the baseline Top-K "
            f"for {len(reconstruction_mismatches)} queries; examples={examples}. "
            "Weight-perturbation results would be invalid."
        )
    page_reconstruction_mismatches = [
        row
        for row in rows
        if float(row.get("page_baseline_reconstruction_jaccard") or 0.0) < 1.0
    ]
    if page_reconstruction_mismatches:
        examples = [
            str(row.get("query_task_id") or "")
            for row in page_reconstruction_mismatches[:5]
        ]
        raise ValueError(
            "Saved compact page-score rows do not exactly reconstruct the "
            f"baseline Top-t for {len(page_reconstruction_mismatches)} queries; "
            f"examples={examples}. Page-weight perturbation results would be invalid."
        )
    analysis_dir = (
        scoped_analysis_dir(run_dir, scope)
        if scope is not None
        else run_dir / "analysis_v2"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "ranking_robustness.csv"
    fieldnames = [
        "sample_id",
        "query_task_id",
        "cutoff",
        "perturbed_weight_count",
        "random_draws",
        "relative_perturbation",
        "baseline_reconstruction_jaccard",
        "page_baseline_reconstruction_jaccard",
        "mean_gold_trajectory_recall",
        "mean_gold_ref_coverage",
        "mean_top_k_jaccard",
        "min_top_k_jaccard",
        "mean_page_candidate_gold_trajectory_recall",
        "mean_page_candidate_gold_ref_coverage",
        "mean_page_top_t_jaccard",
        "min_page_top_t_jaccard",
        "mean_page_candidate_universe_size",
        "llm_rerank_held_fixed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cutoff_path = analysis_dir / "ranking_cutoff_replay.csv"
    cutoff_fieldnames = [
        "sample_id",
        "query_task_id",
        "page_cutoff",
        "trajectory_cutoff",
        "observed_page_cutoff",
        "replay_observable",
        "candidate_universe_size",
        "gold_trajectory_recall",
        "gold_ref_coverage",
        "top_k_jaccard_vs_observed_candidate_ranking",
        "fixed_saved_page_scores",
        "fixed_saved_trajectory_scores",
        "llm_rerank_not_reexecuted",
    ]
    with cutoff_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cutoff_fieldnames)
        writer.writeheader()
        writer.writerows(cutoff_rows)
    summary = {
        "schema_version": "ranking_robustness_v1",
        "diagnostic_scope": "deterministic_score_components_only",
        "llm_rerank_held_fixed": True,
        "perturbed_weights": PERTURBED_WEIGHTS,
        "perturbed_trajectory_weights": PERTURBED_WEIGHTS,
        "perturbed_page_weights": PAGE_PERTURBED_WEIGHTS,
        "fixed_terms": [
            "lexical_and_mismatch_residual",
            "rrf_k",
            "llm_trajectory_set_rerank",
        ],
        "query_count": len(rows),
        "relative_perturbation": relative_perturbation,
        "random_draws": random_draws,
        "seed": seed,
        "cutoff": cutoff,
        "page_cutoffs": resolved_page_cutoffs,
        "trajectory_cutoffs": resolved_trajectory_cutoffs,
        "cutoff_replay_scope": (
            "t_pages_at_or_below_observed_cutoff_with_saved_scores"
        ),
        "trajectory_max_length_m_replay_supported": False,
        "trajectory_max_length_m_note": (
            "Changing m alters trajectory construction and cannot be replayed from "
            "the fixed database."
        ),
        "mean_gold_trajectory_recall": (
            mean(float(row["mean_gold_trajectory_recall"]) for row in rows)
            if rows
            else None
        ),
        "mean_gold_ref_coverage": (
            mean(float(row["mean_gold_ref_coverage"]) for row in rows) if rows else None
        ),
        "mean_top_k_jaccard": (
            mean(float(row["mean_top_k_jaccard"]) for row in rows) if rows else None
        ),
        "mean_page_candidate_gold_trajectory_recall": (
            mean(
                float(row["mean_page_candidate_gold_trajectory_recall"])
                for row in rows
            )
            if rows
            else None
        ),
        "mean_page_candidate_gold_ref_coverage": (
            mean(
                float(row["mean_page_candidate_gold_ref_coverage"])
                for row in rows
            )
            if rows
            else None
        ),
        "mean_page_top_t_jaccard": (
            mean(float(row["mean_page_top_t_jaccard"]) for row in rows)
            if rows
            else None
        ),
        "mean_baseline_reconstruction_jaccard": (
            mean(float(row["baseline_reconstruction_jaccard"]) for row in rows)
            if rows
            else None
        ),
        "baseline_reconstruction_exact": True,
        "baseline_reconstruction_mismatch_count": 0,
        "page_baseline_reconstruction_exact": True,
        "page_baseline_reconstruction_mismatch_count": 0,
        "full_rankings_verified": True,
        "query_scope": scope.metadata() if scope is not None else None,
        "rows_path": str(csv_path),
        "cutoff_rows_path": str(cutoff_path),
    }
    write_json(analysis_dir / "ranking_robustness_summary.json", summary)
    return summary
