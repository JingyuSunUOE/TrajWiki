from __future__ import annotations

import pytest

from trajpatch.exceptions import ProviderConfigurationError
from trajpatch.experiments.answer_ablation import (
    _derived_answers,
    _effective_provider_concurrency,
    _parse_judge_verdict,
    _require_remote_credentials,
    _sample_queries,
    _source_tree_sha256,
    _stratum_for_gold,
)
from trajpatch.experiments.ranking_robustness import (
    PAGE_PERTURBED_WEIGHTS,
    PERTURBED_WEIGHTS,
    _rerank_pages_with_multipliers,
    _rerank_with_multipliers,
)
from trajpatch.experiments.statistics import (
    complete_prevalence_weighted_metric,
    dialogue_cluster_paired_bootstrap_rows,
    judge_accuracy,
    paired_bootstrap_rows,
    prevalence_weighted_paired_bootstrap_rows,
    stratum_summary_rows,
)


def test_paired_bootstrap_is_deterministic() -> None:
    rows = [
        {
            "variant": "full",
            "query_task_id": "q1",
            "independent_judge_verdict": "correct",
            "token_f1": 1.0,
            "gold_ref_coverage": 1.0,
            "source_supported_proxy": True,
            "unsupported_answer_risk": False,
        },
        {
            "variant": "candidate",
            "query_task_id": "q1",
            "independent_judge_verdict": "incorrect",
            "token_f1": 0.0,
            "gold_ref_coverage": 0.0,
            "source_supported_proxy": False,
            "unsupported_answer_risk": True,
        },
        {
            "variant": "full",
            "query_task_id": "q2",
            "independent_judge_verdict": "partial",
            "token_f1": 0.5,
            "gold_ref_coverage": 0.5,
            "source_supported_proxy": True,
            "unsupported_answer_risk": False,
        },
        {
            "variant": "candidate",
            "query_task_id": "q2",
            "independent_judge_verdict": "partial",
            "token_f1": 0.5,
            "gold_ref_coverage": 0.5,
            "source_supported_proxy": True,
            "unsupported_answer_risk": False,
        },
    ]

    first = paired_bootstrap_rows(rows, iterations=100, seed=7)
    second = paired_bootstrap_rows(rows, iterations=100, seed=7)

    assert first == second
    judge_row = next(row for row in first if row["metric"] == "independent_judge_score")
    assert judge_row["win_count"] == 0
    assert judge_row["tie_count"] == 1
    assert judge_row["loss_count"] == 1
    risk_row = next(
        row for row in first if row["metric"] == "unsupported_answer_risk"
    )
    assert risk_row["metric_direction"] == "lower_is_better"
    assert risk_row["loss_count"] == 1
    assert risk_row["tie_count"] == 1
    strict_accuracy_row = next(
        row
        for row in first
        if row["metric"] == "independent_judge_accuracy"
    )
    assert strict_accuracy_row["reference_mean"] == pytest.approx(0.5)
    assert strict_accuracy_row["variant_mean"] == pytest.approx(0.0)


def test_weighted_and_cluster_bootstraps_are_deterministic() -> None:
    rows: list[dict] = []
    for query, sample, stratum, full_score, other_score in [
        ("q1", "s1", "strict", "correct", "incorrect"),
        ("q2", "s1", "strict", "correct", "partial"),
        ("q3", "s2", "ordinary", "incorrect", "correct"),
        ("q4", "s3", "ordinary", "partial", "partial"),
    ]:
        for variant, verdict in [
            ("full", full_score),
            ("other", other_score),
        ]:
            rows.append(
                {
                    "variant": variant,
                    "query_task_id": query,
                    "sample_id": sample,
                    "stratum": stratum,
                    "independent_judge_verdict": verdict,
                    "token_f1": 0.5,
                    "bleu1": 0.5,
                    "answer_abstained": False,
                }
            )
    weighted_first = prevalence_weighted_paired_bootstrap_rows(
        rows,
        prevalence={"strict": 0.25, "ordinary": 0.75},
        iterations=100,
        seed=7,
    )
    weighted_second = prevalence_weighted_paired_bootstrap_rows(
        rows,
        prevalence={"strict": 0.25, "ordinary": 0.75},
        iterations=100,
        seed=7,
    )
    cluster_first = dialogue_cluster_paired_bootstrap_rows(
        rows,
        iterations=100,
        seed=7,
    )
    cluster_second = dialogue_cluster_paired_bootstrap_rows(
        rows,
        iterations=100,
        seed=7,
    )
    assert weighted_first == weighted_second
    assert cluster_first == cluster_second
    weighted_judge = next(
        row
        for row in weighted_first
        if row["metric"] == "independent_judge_score"
    )
    cluster_judge = next(
        row
        for row in cluster_first
        if row["metric"] == "independent_judge_score"
    )
    assert weighted_judge["prevalence_coverage"] == pytest.approx(1.0)
    assert cluster_judge["dialogue_cluster_count"] == 3


def test_judge_accuracy_treats_partial_as_not_strictly_correct() -> None:
    assert judge_accuracy("correct") == 1.0
    assert judge_accuracy("partial") == 0.0
    assert judge_accuracy("incorrect") == 0.0
    assert judge_accuracy("judge_error") is None


def test_stratum_summary_excludes_unobservable_support_values() -> None:
    summary = stratum_summary_rows(
        [
            {
                "variant": "full",
                "stratum": "ordinary",
                "answer_abstained": False,
                "source_supported_proxy": True,
            },
            {
                "variant": "full",
                "stratum": "ordinary",
                "answer_abstained": False,
                "source_supported_proxy": None,
            },
        ]
    )

    assert summary[0]["source_supported_proxy_rate"] == 1.0
    assert summary[0]["source_supported_proxy_count"] == 1


def test_prevalence_weighting_rejects_partial_within_stratum_observation() -> None:
    rows = [
        {
            "stratum": "deep",
            "metric": 0.5,
            "metric_count": 2,
        },
        {
            "stratum": "ordinary",
            "metric": 1.0,
            "metric_count": 1,
        },
    ]
    value, coverage = complete_prevalence_weighted_metric(
        rows,
        field="metric",
        count_field="metric_count",
        prevalence={"deep": 0.4, "ordinary": 0.6},
        selected_counts_by_stratum={"deep": 2, "ordinary": 2},
    )

    assert value is None
    assert coverage == pytest.approx(0.4)

    rows[1]["metric_count"] = 2
    value, coverage = complete_prevalence_weighted_metric(
        rows,
        field="metric",
        count_field="metric_count",
        prevalence={"deep": 0.4, "ordinary": 0.6},
        selected_counts_by_stratum={"deep": 2, "ordinary": 2},
    )

    assert value == pytest.approx(0.8)
    assert coverage == pytest.approx(1.0)


def test_independent_judge_requires_explicit_verdict_field() -> None:
    assert _parse_judge_verdict("VERDICT: CORRECT\nRATIONALE: supported") == "correct"
    assert _parse_judge_verdict("CORRECT") == "correct"
    assert _parse_judge_verdict("The answer is not correct.") == "judge_error"


def test_answer_stage_ablation_marks_legacy_fallback_as_unobservable() -> None:
    legacy = _derived_answers(
        {
            "answer_text": "final",
            "metadata": {
                "answer_metadata": {
                    "answer_initial_text": "legacy initial",
                    "initial_answer_text": "legacy pre-retry",
                }
            },
        }
    )
    assert legacy["no_answer_validation_or_repair"]["stage_observable"] is False
    assert legacy["no_retrieval_retry"]["stage_observable"] is False

    captured = _derived_answers(
        {
            "answer_text": "final",
            "metadata": {
                "answer_metadata": {
                    "answer_stage_initial_text": "",
                    "answer_stage_initial_supporting_refs": [],
                    "answer_stage_pre_reflection_text": "pre-retry",
                    "answer_stage_pre_reflection_supporting_refs": [],
                }
            },
        }
    )
    assert captured["no_answer_validation_or_repair"]["stage_observable"] is True
    assert captured["no_answer_validation_or_repair"]["answer_text"] == ""
    assert captured["no_retrieval_retry"]["stage_observable"] is True


def test_remote_rebuttal_credentials_use_standard_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("Claud_API_Key", "legacy-typo")
    with pytest.raises(ProviderConfigurationError, match="ANTHROPIC_API_KEY"):
        _require_remote_credentials(
            "remote",
            "claude-sonnet-4-6",
            role="independent judge",
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _require_remote_credentials(
        "remote",
        "claude-sonnet-4-6",
        role="independent judge",
    )


def test_bare_local_rebuttal_provider_is_serialized() -> None:
    assert _effective_provider_concurrency("local", 6) == 1
    assert _effective_provider_concurrency("remote", 6) == 6
    assert _effective_provider_concurrency("openai-compatible", 6) == 6


def test_source_tree_hash_covers_untracked_experiment_code(tmp_path) -> None:
    source = tmp_path / "src" / "trajpatch"
    source.mkdir(parents=True)
    module = source / "experiment.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    first = _source_tree_sha256(tmp_path)
    module.write_text("VALUE = 2\n", encoding="utf-8")

    assert _source_tree_sha256(tmp_path) != first


def test_rebuttal_strata_distinguish_deep_update_and_ordinary() -> None:
    memory_index = {
        "trajectory_to_snapshots": {
            "t-deep": ["s-deep-old", "s-deep-latest"],
            "t-ordinary": ["s-ordinary"],
        },
        "trajectory_to_sample": {
            "t-deep": "sample-1",
            "t-ordinary": "sample-1",
        },
        "snapshot_refs": {
            "s-deep-old": {"D1:1"},
            "s-deep-latest": {"D2:1"},
            "s-ordinary": {"D3:1"},
        },
        "claims_by_snapshot": {
            "s-deep-old": [
                {
                    "status": "deprecated",
                    "source_message_ids": ["message-update"],
                }
            ],
            "s-deep-latest": [],
            "s-ordinary": [],
        },
    }

    assert (
        _stratum_for_gold(
            {
                "sample_id": "sample-1",
                "gold_source_refs": ["D1:1"],
                "gold_source_message_ids": ["message-deep"],
            },
            memory_index,
        )
        == "strict_deep_history"
    )
    assert (
        _stratum_for_gold(
            {
                "sample_id": "sample-1",
                "gold_source_refs": ["D2:1"],
                "gold_source_message_ids": ["message-update"],
            },
            memory_index,
        )
        == "update_sensitive"
    )
    assert (
        _stratum_for_gold(
            {
                "sample_id": "sample-1",
                "gold_source_refs": ["D3:1"],
                "gold_source_message_ids": ["message-ordinary"],
            },
            memory_index,
        )
        == "ordinary"
    )


def test_fixed_sixty_query_sample_preserves_preregistered_quotas() -> None:
    sample_rows: list[dict] = []
    gold_rows: list[dict] = []
    trajectory_to_snapshots: dict[str, list[str]] = {}
    trajectory_to_sample: dict[str, str] = {}
    snapshot_refs: dict[str, set[str]] = {}
    claims_by_snapshot: dict[str, list[dict]] = {}
    for index in range(60):
        sample_id = f"sample-{index:02d}"
        query_task_id = f"{sample_id}_qa_0"
        trajectory_id = f"trajectory-{index:02d}"
        latest_snapshot_id = f"snapshot-{index:02d}-latest"
        gold_ref = f"D{index + 1}:1"
        gold_message_id = f"message-{index:02d}"
        sample_rows.append(
            {
                "sample_id": sample_id,
                "query_task_id": query_task_id,
            }
        )
        gold_rows.append(
            {
                "sample_id": sample_id,
                "query_task_id": query_task_id,
                "gold_source_refs": [gold_ref],
                "gold_source_message_ids": [gold_message_id],
            }
        )
        trajectory_to_sample[trajectory_id] = sample_id
        claims_by_snapshot[latest_snapshot_id] = []
        if index < 9:
            old_snapshot_id = f"snapshot-{index:02d}-old"
            trajectory_to_snapshots[trajectory_id] = [
                old_snapshot_id,
                latest_snapshot_id,
            ]
            snapshot_refs[old_snapshot_id] = {gold_ref}
            snapshot_refs[latest_snapshot_id] = {f"D{index + 1}:2"}
            claims_by_snapshot[old_snapshot_id] = []
        else:
            trajectory_to_snapshots[trajectory_id] = [latest_snapshot_id]
            snapshot_refs[latest_snapshot_id] = {gold_ref}
            if index < 35:
                claims_by_snapshot[latest_snapshot_id] = [
                    {
                        "status": "deprecated",
                        "source_message_ids": [gold_message_id],
                    }
                ]
    memory_index = {
        "trajectory_to_snapshots": trajectory_to_snapshots,
        "trajectory_to_sample": trajectory_to_sample,
        "snapshot_refs": snapshot_refs,
        "claims_by_snapshot": claims_by_snapshot,
    }

    selected, manifest = _sample_queries(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        memory_index=memory_index,
        sample_size=60,
        seed=7,
    )

    assert len(selected) == 60
    assert manifest["selected_counts_by_stratum"] == {
        "strict_deep_history": 9,
        "update_sensitive": 26,
        "ordinary": 25,
    }


def test_rebuttal_200_sampling_is_exact_and_contains_pilot() -> None:
    sample_rows: list[dict] = []
    gold_rows: list[dict] = []
    trajectory_to_sample: dict[str, str] = {}
    trajectory_to_snapshots: dict[str, list[str]] = {}
    snapshot_refs: dict[str, set[str]] = {}
    claims_by_snapshot: dict[str, list[dict]] = {}
    for index in range(282):
        sample_id = f"sample-{index:03d}"
        query_task_id = f"query-{index:03d}"
        trajectory_id = f"trajectory-{index:03d}"
        latest_snapshot_id = f"snapshot-{index:03d}-latest"
        gold_ref = f"D{index + 1}:1"
        gold_message_id = f"message-{index:03d}"
        sample_rows.append(
            {
                "sample_id": sample_id,
                "query_task_id": query_task_id,
            }
        )
        gold_rows.append(
            {
                "sample_id": sample_id,
                "query_task_id": query_task_id,
                "gold_source_refs": [gold_ref],
                "gold_source_message_ids": [gold_message_id],
            }
        )
        trajectory_to_sample[trajectory_id] = sample_id
        if index < 9:
            old_snapshot_id = f"snapshot-{index:03d}-old"
            trajectory_to_snapshots[trajectory_id] = [
                old_snapshot_id,
                latest_snapshot_id,
            ]
            snapshot_refs[old_snapshot_id] = {gold_ref}
            snapshot_refs[latest_snapshot_id] = {f"D{index + 1}:2"}
            claims_by_snapshot[old_snapshot_id] = []
            claims_by_snapshot[latest_snapshot_id] = []
        else:
            trajectory_to_snapshots[trajectory_id] = [latest_snapshot_id]
            snapshot_refs[latest_snapshot_id] = {gold_ref}
            claims_by_snapshot[latest_snapshot_id] = (
                [
                    {
                        "status": "deprecated",
                        "source_message_ids": [gold_message_id],
                    }
                ]
                if index < 197
                else []
            )
    memory_index = {
        "trajectory_to_snapshots": trajectory_to_snapshots,
        "trajectory_to_sample": trajectory_to_sample,
        "snapshot_refs": snapshot_refs,
        "claims_by_snapshot": claims_by_snapshot,
    }

    pilot, pilot_manifest = _sample_queries(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        memory_index=memory_index,
        sample_size=60,
        seed=7,
        sampling_profile="rebuttal_60_v1",
    )
    extension, extension_manifest = _sample_queries(
        sample_rows=sample_rows,
        gold_rows=gold_rows,
        memory_index=memory_index,
        sample_size=200,
        seed=7,
        sampling_profile="rebuttal_200_v1",
    )

    pilot_ids = {str(row["query_task_id"]) for row in pilot}
    extension_ids = {str(row["query_task_id"]) for row in extension}
    assert len(pilot_ids) == 60
    assert len(extension_ids) == 200
    assert pilot_ids <= extension_ids
    assert pilot_manifest["selected_counts_by_stratum"] == {
        "strict_deep_history": 9,
        "update_sensitive": 26,
        "ordinary": 25,
    }
    assert extension_manifest["selected_counts_by_stratum"] == {
        "strict_deep_history": 9,
        "update_sensitive": 132,
        "ordinary": 59,
    }
    assert extension_manifest["quota_policy"] == (
        extension_manifest["selected_counts_by_stratum"]
    )
    assert (
        extension_manifest["sampling_status"]
        == "post_hoc_nested_extension"
    )


def test_ranking_robustness_reconstructs_saved_rrf_order() -> None:
    ranked_rows = [
        {
            "trajectory_id": "t-a",
            "score_components": {
                "dense_score": 0.8,
                "sparse_score": 0.2,
                "summary_similarity": 0.8,
                "latest_similarity": 0.4,
                "entity_match_boost": 0.1,
                "facet_tag_boost": 0.0,
                "facet_value_boost": 0.0,
                "answer_family_match_score": 0.5,
                "source_event_match_score": 0.0,
            },
        },
        {
            "trajectory_id": "t-b",
            "score_components": {
                "dense_score": 0.4,
                "sparse_score": 0.9,
                "summary_similarity": 0.3,
                "latest_similarity": 0.2,
                "entity_match_boost": 0.0,
                "facet_tag_boost": 0.0,
                "facet_value_boost": 0.0,
                "answer_family_match_score": 0.2,
                "source_event_match_score": 0.1,
            },
        },
    ]

    reconstructed = _rerank_with_multipliers(
        ranked_rows,
        {name: 1.0 for name in PERTURBED_WEIGHTS},
    )

    assert [row["trajectory_id"] for row in reconstructed] == ["t-a", "t-b"]


def test_page_ranking_robustness_reconstructs_saved_rrf_order() -> None:
    ranked_rows = [
        {
            "page_id": "page-a",
            "score_components": {
                "dense_score": 0.8,
                "sparse_score": 0.5,
                "entity_bonus": 0.1,
                "reflection_bonus": 0.04,
                "page_granularity_adjustment": 0.035,
                "page_family_match_score": 0.8,
                "page_family_mismatch_penalty": 0.0,
            },
        },
        {
            "page_id": "page-b",
            "score_components": {
                "dense_score": 0.3,
                "sparse_score": 0.2,
                "entity_bonus": 0.0,
                "reflection_bonus": 0.0,
                "page_granularity_adjustment": -0.02,
                "page_family_match_score": 0.2,
                "page_family_mismatch_penalty": 0.1,
            },
        },
    ]

    reconstructed = _rerank_pages_with_multipliers(
        ranked_rows,
        {name: 1.0 for name in PAGE_PERTURBED_WEIGHTS},
    )

    assert [row["page_id"] for row in reconstructed] == ["page-a", "page-b"]
