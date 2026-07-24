"""Deterministic paired statistics for rebuttal experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from statistics import mean
from typing import Any


def judge_score(verdict: Any) -> float | None:
    normalized = str(verdict or "").strip().lower()
    if normalized == "correct":
        return 1.0
    if normalized == "partial":
        return 0.5
    if normalized == "incorrect":
        return 0.0
    return None


def judge_accuracy(verdict: Any) -> float | None:
    normalized = str(verdict or "").strip().lower()
    if normalized == "correct":
        return 1.0
    if normalized in {"partial", "incorrect"}:
        return 0.0
    return None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_specs() -> dict[
    str,
    tuple[Callable[[dict[str, Any]], float | None], str],
]:
    return {
        "independent_judge_score": (
            lambda row: judge_score(row.get("independent_judge_verdict")),
            "higher_is_better",
        ),
        "independent_judge_accuracy": (
            lambda row: judge_accuracy(row.get("independent_judge_verdict")),
            "higher_is_better",
        ),
        "token_f1": (
            lambda row: (
                float(row["token_f1"])
                if row.get("token_f1") is not None
                else None
            ),
            "higher_is_better",
        ),
        "bleu1": (
            lambda row: (
                float(row["bleu1"]) if row.get("bleu1") is not None else None
            ),
            "higher_is_better",
        ),
        "abstention_rate": (
            lambda row: float(bool(row.get("answer_abstained"))),
            "lower_is_better",
        ),
        "gold_ref_coverage": (
            lambda row: (
                float(row["gold_ref_coverage"])
                if row.get("gold_ref_coverage") is not None
                else None
            ),
            "higher_is_better",
        ),
        "source_supported_proxy": (
            lambda row: (
                float(bool(row["source_supported_proxy"]))
                if row.get("source_supported_proxy") is not None
                else None
            ),
            "higher_is_better",
        ),
        "unsupported_answer_risk": (
            lambda row: (
                float(bool(row["unsupported_answer_risk"]))
                if row.get("unsupported_answer_risk") is not None
                else None
            ),
            "lower_is_better",
        ),
        "post_source_validation_judge_score": (
            lambda row: (
                float(row["post_source_validation_judge_score"])
                if row.get("post_source_validation_judge_score") is not None
                else None
            ),
            "higher_is_better",
        ),
    }


def _comparison_pairs(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str,
    variant: str,
    getter: Callable[[dict[str, Any]], float | None],
) -> list[dict[str, Any]]:
    by_variant_query = {
        (str(row.get("variant")), str(row.get("query_task_id"))): row
        for row in rows
    }
    query_ids = sorted(
        {
            query_task_id
            for candidate_variant, query_task_id in by_variant_query
            if candidate_variant == variant
            and (reference_variant, query_task_id) in by_variant_query
        }
    )
    output: list[dict[str, Any]] = []
    for query_task_id in query_ids:
        reference_row = by_variant_query[(reference_variant, query_task_id)]
        candidate_row = by_variant_query[(variant, query_task_id)]
        reference = getter(reference_row)
        candidate = getter(candidate_row)
        if reference is None or candidate is None:
            continue
        output.append(
            {
                "query_task_id": query_task_id,
                "sample_id": str(
                    candidate_row.get("sample_id")
                    or reference_row.get("sample_id")
                    or ""
                ),
                "stratum": str(
                    candidate_row.get("stratum")
                    or reference_row.get("stratum")
                    or "unknown"
                ),
                "reference": reference,
                "candidate": candidate,
            }
        )
    return output


def paired_bootstrap_rows(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str = "full",
    iterations: int = 10_000,
    seed: int = 7,
) -> list[dict[str, Any]]:
    by_variant_query = {
        (str(row.get("variant")), str(row.get("query_task_id"))): row for row in rows
    }
    variants = sorted(
        {
            str(row.get("variant"))
            for row in rows
            if str(row.get("variant")) != reference_variant
        }
    )
    output: list[dict[str, Any]] = []
    metric_specs = _metric_specs()
    for variant in variants:
        query_ids = sorted(
            {
                query_task_id
                for (candidate_variant, query_task_id) in by_variant_query
                if candidate_variant == variant
                and (reference_variant, query_task_id) in by_variant_query
            }
        )
        for metric_name, (getter, metric_direction) in metric_specs.items():
            pairs: list[tuple[float, float]] = []
            for query_task_id in query_ids:
                reference = getter(by_variant_query[(reference_variant, query_task_id)])
                candidate = getter(by_variant_query[(variant, query_task_id)])
                if reference is not None and candidate is not None:
                    pairs.append((reference, candidate))
            if not pairs:
                output.append(
                    {
                        "reference_variant": reference_variant,
                        "variant": variant,
                        "metric": metric_name,
                        "metric_direction": metric_direction,
                        "paired_query_count": 0,
                        "reference_mean": None,
                        "variant_mean": None,
                        "mean_delta_variant_minus_reference": None,
                        "ci95_low": None,
                        "ci95_high": None,
                        "win_count": 0,
                        "tie_count": 0,
                        "loss_count": 0,
                        "bootstrap_iterations": iterations,
                        "bootstrap_seed": seed,
                    }
                )
                continue
            rng = random.Random(f"{seed}:{variant}:{metric_name}")
            deltas: list[float] = []
            for _ in range(max(1, int(iterations))):
                sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
                deltas.append(
                    mean(candidate - reference for reference, candidate in sample)
                )
            output.append(
                {
                    "reference_variant": reference_variant,
                    "variant": variant,
                    "metric": metric_name,
                    "metric_direction": metric_direction,
                    "paired_query_count": len(pairs),
                    "reference_mean": mean(reference for reference, _ in pairs),
                    "variant_mean": mean(candidate for _, candidate in pairs),
                    "mean_delta_variant_minus_reference": mean(
                        candidate - reference for reference, candidate in pairs
                    ),
                    "ci95_low": _percentile(deltas, 0.025),
                    "ci95_high": _percentile(deltas, 0.975),
                    "win_count": sum(
                        1
                        for reference, candidate in pairs
                        if (
                            candidate > reference
                            if metric_direction == "higher_is_better"
                            else candidate < reference
                        )
                    ),
                    "tie_count": sum(
                        1 for reference, candidate in pairs if candidate == reference
                    ),
                    "loss_count": sum(
                        1
                        for reference, candidate in pairs
                        if (
                            candidate < reference
                            if metric_direction == "higher_is_better"
                            else candidate > reference
                        )
                    ),
                    "bootstrap_iterations": iterations,
                    "bootstrap_seed": seed,
                }
            )
    return output


def prevalence_weighted_paired_bootstrap_rows(
    rows: list[dict[str, Any]],
    *,
    prevalence: dict[str, float],
    reference_variant: str = "full",
    iterations: int = 10_000,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Bootstrap within strata, then aggregate using population prevalence."""

    variants = sorted(
        {
            str(row.get("variant"))
            for row in rows
            if str(row.get("variant")) != reference_variant
        }
    )
    output: list[dict[str, Any]] = []
    expected_strata = {
        str(stratum): float(weight)
        for stratum, weight in prevalence.items()
        if float(weight) > 0
    }
    for variant in variants:
        for metric_name, (getter, metric_direction) in _metric_specs().items():
            pairs = _comparison_pairs(
                rows,
                reference_variant=reference_variant,
                variant=variant,
                getter=getter,
            )
            by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pair in pairs:
                by_stratum[str(pair["stratum"])].append(pair)
            covered = sum(
                weight
                for stratum, weight in expected_strata.items()
                if by_stratum.get(stratum)
            )
            complete = abs(covered - sum(expected_strata.values())) <= 1e-9
            base = {
                "reference_variant": reference_variant,
                "variant": variant,
                "metric": metric_name,
                "metric_direction": metric_direction,
                "estimator": "population_prevalence_weighted",
                "paired_query_count": len(pairs),
                "prevalence_coverage": covered,
                "bootstrap_iterations": iterations,
                "bootstrap_seed": seed,
            }
            if not pairs or not complete:
                output.append(
                    {
                        **base,
                        "reference_mean": None,
                        "variant_mean": None,
                        "mean_delta_variant_minus_reference": None,
                        "ci95_low": None,
                        "ci95_high": None,
                        "win_count": 0,
                        "tie_count": 0,
                        "loss_count": 0,
                    }
                )
                continue
            reference_mean = sum(
                expected_strata[stratum]
                * mean(float(pair["reference"]) for pair in by_stratum[stratum])
                for stratum in expected_strata
            )
            variant_mean = sum(
                expected_strata[stratum]
                * mean(float(pair["candidate"]) for pair in by_stratum[stratum])
                for stratum in expected_strata
            )
            rng = random.Random(
                f"{seed}:weighted:{variant}:{metric_name}"
            )
            deltas: list[float] = []
            for _ in range(max(1, int(iterations))):
                delta = 0.0
                for stratum, weight in expected_strata.items():
                    stratum_pairs = by_stratum[stratum]
                    sampled = [
                        stratum_pairs[rng.randrange(len(stratum_pairs))]
                        for _ in range(len(stratum_pairs))
                    ]
                    delta += weight * mean(
                        float(pair["candidate"]) - float(pair["reference"])
                        for pair in sampled
                    )
                deltas.append(delta)
            output.append(
                {
                    **base,
                    "reference_mean": reference_mean,
                    "variant_mean": variant_mean,
                    "mean_delta_variant_minus_reference": (
                        variant_mean - reference_mean
                    ),
                    "ci95_low": _percentile(deltas, 0.025),
                    "ci95_high": _percentile(deltas, 0.975),
                    "win_count": sum(
                        1
                        for pair in pairs
                        if (
                            float(pair["candidate"]) > float(pair["reference"])
                            if metric_direction == "higher_is_better"
                            else float(pair["candidate"]) < float(pair["reference"])
                        )
                    ),
                    "tie_count": sum(
                        1
                        for pair in pairs
                        if float(pair["candidate"]) == float(pair["reference"])
                    ),
                    "loss_count": sum(
                        1
                        for pair in pairs
                        if (
                            float(pair["candidate"]) < float(pair["reference"])
                            if metric_direction == "higher_is_better"
                            else float(pair["candidate"]) > float(pair["reference"])
                        )
                    ),
                }
            )
    return output


def dialogue_cluster_paired_bootstrap_rows(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str = "full",
    iterations: int = 10_000,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Resample dialogue IDs and retain all paired queries in each draw."""

    variants = sorted(
        {
            str(row.get("variant"))
            for row in rows
            if str(row.get("variant")) != reference_variant
        }
    )
    output: list[dict[str, Any]] = []
    for variant in variants:
        for metric_name, (getter, metric_direction) in _metric_specs().items():
            pairs = _comparison_pairs(
                rows,
                reference_variant=reference_variant,
                variant=variant,
                getter=getter,
            )
            by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pair in pairs:
                by_sample[str(pair["sample_id"])].append(pair)
            sample_ids = sorted(by_sample)
            base = {
                "reference_variant": reference_variant,
                "variant": variant,
                "metric": metric_name,
                "metric_direction": metric_direction,
                "estimator": "dialogue_cluster_bootstrap",
                "paired_query_count": len(pairs),
                "dialogue_cluster_count": len(sample_ids),
                "bootstrap_iterations": iterations,
                "bootstrap_seed": seed,
            }
            if not pairs or not sample_ids:
                output.append(
                    {
                        **base,
                        "reference_mean": None,
                        "variant_mean": None,
                        "mean_delta_variant_minus_reference": None,
                        "ci95_low": None,
                        "ci95_high": None,
                        "win_count": 0,
                        "tie_count": 0,
                        "loss_count": 0,
                    }
                )
                continue
            rng = random.Random(f"{seed}:cluster:{variant}:{metric_name}")
            deltas: list[float] = []
            for _ in range(max(1, int(iterations))):
                sampled_pairs = [
                    pair
                    for _ in range(len(sample_ids))
                    for pair in by_sample[sample_ids[rng.randrange(len(sample_ids))]]
                ]
                deltas.append(
                    mean(
                        float(pair["candidate"]) - float(pair["reference"])
                        for pair in sampled_pairs
                    )
                )
            output.append(
                {
                    **base,
                    "reference_mean": mean(
                        float(pair["reference"]) for pair in pairs
                    ),
                    "variant_mean": mean(
                        float(pair["candidate"]) for pair in pairs
                    ),
                    "mean_delta_variant_minus_reference": mean(
                        float(pair["candidate"]) - float(pair["reference"])
                        for pair in pairs
                    ),
                    "ci95_low": _percentile(deltas, 0.025),
                    "ci95_high": _percentile(deltas, 0.975),
                    "win_count": sum(
                        1
                        for pair in pairs
                        if (
                            float(pair["candidate"]) > float(pair["reference"])
                            if metric_direction == "higher_is_better"
                            else float(pair["candidate"]) < float(pair["reference"])
                        )
                    ),
                    "tie_count": sum(
                        1
                        for pair in pairs
                        if float(pair["candidate"]) == float(pair["reference"])
                    ),
                    "loss_count": sum(
                        1
                        for pair in pairs
                        if (
                            float(pair["candidate"]) < float(pair["reference"])
                            if metric_direction == "higher_is_better"
                            else float(pair["candidate"]) > float(pair["reference"])
                        )
                    ),
                }
            )
    return output


def stratum_summary_rows(
    rows: list[dict[str, Any]],
    *,
    prevalence: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("variant")), str(row.get("stratum") or "unknown"))].append(
            row
        )
    output: list[dict[str, Any]] = []
    for (variant, stratum), group in sorted(grouped.items()):
        judge_values = [
            score
            for row in group
            if (score := judge_score(row.get("independent_judge_verdict"))) is not None
        ]
        judge_accuracy_values = [
            score
            for row in group
            if (
                score := judge_accuracy(row.get("independent_judge_verdict"))
            )
            is not None
        ]
        f1_values = [
            float(row["token_f1"]) for row in group if row.get("token_f1") is not None
        ]
        bleu_values = [
            float(row["bleu1"]) for row in group if row.get("bleu1") is not None
        ]
        source_validation_values = [
            float(bool(row.get("would_pass_source_validation")))
            for row in group
            if row.get("would_pass_source_validation") is not None
        ]
        post_validation_judge_values = [
            float(row["post_source_validation_judge_score"])
            for row in group
            if row.get("post_source_validation_judge_score") is not None
        ]
        unsupported_risk_values = [
            float(bool(row.get("unsupported_answer_risk")))
            for row in group
            if row.get("unsupported_answer_risk") is not None
        ]
        source_support_values = [
            float(bool(row.get("source_supported_proxy")))
            for row in group
            if row.get("source_supported_proxy") is not None
        ]
        prompt_token_values = [
            float(row["estimated_prompt_tokens"])
            for row in group
            if row.get("estimated_prompt_tokens") is not None
        ]
        output.append(
            {
                "variant": variant,
                "stratum": stratum,
                "query_count": len(group),
                "prevalence_weight": (prevalence or {}).get(stratum),
                "mean_independent_judge_score": mean(judge_values)
                if judge_values
                else None,
                "independent_judge_score_count": len(judge_values),
                "independent_judge_accuracy": (
                    mean(judge_accuracy_values)
                    if judge_accuracy_values
                    else None
                ),
                "independent_judge_accuracy_count": len(
                    judge_accuracy_values
                ),
                "mean_token_f1": mean(f1_values) if f1_values else None,
                "token_f1_count": len(f1_values),
                "mean_bleu1": mean(bleu_values) if bleu_values else None,
                "bleu1_count": len(bleu_values),
                "abstention_rate": mean(
                    float(bool(row.get("answer_abstained"))) for row in group
                ),
                "abstention_count": len(group),
                "source_supported_proxy_rate": (
                    mean(source_support_values)
                    if source_support_values
                    else None
                ),
                "source_supported_proxy_count": len(source_support_values),
                "unsupported_answer_risk_rate": (
                    mean(unsupported_risk_values)
                    if unsupported_risk_values
                    else None
                ),
                "unsupported_answer_risk_count": len(unsupported_risk_values),
                "source_validation_pass_rate": (
                    mean(source_validation_values) if source_validation_values else None
                ),
                "source_validation_count": len(source_validation_values),
                "mean_post_source_validation_judge_score": (
                    mean(post_validation_judge_values)
                    if post_validation_judge_values
                    else None
                ),
                "post_source_validation_judge_count": len(
                    post_validation_judge_values
                ),
                "mean_estimated_prompt_tokens": mean(
                    prompt_token_values
                )
                if prompt_token_values
                else None,
                "estimated_prompt_tokens_count": len(prompt_token_values),
            }
        )
    return output


def complete_prevalence_weighted_metric(
    variant_strata: list[dict[str, Any]],
    *,
    field: str,
    count_field: str,
    prevalence: dict[str, float],
    selected_counts_by_stratum: dict[str, int],
) -> tuple[float | None, float]:
    """Weight only metrics observed for every sampled query in every stratum."""

    eligible = [
        row
        for row in variant_strata
        if row.get(field) is not None
        and str(row.get("stratum")) in prevalence
        and int(row.get(count_field) or 0)
        == int(selected_counts_by_stratum.get(str(row.get("stratum")), 0))
    ]
    covered_prevalence = sum(
        float(prevalence.get(str(row.get("stratum")), 0.0))
        for row in eligible
    )
    if abs(covered_prevalence - 1.0) > 1e-9:
        return None, covered_prevalence
    return (
        sum(
            float(row[field])
            * float(prevalence.get(str(row.get("stratum")), 0.0))
            for row in eligible
        ),
        covered_prevalence,
    )
