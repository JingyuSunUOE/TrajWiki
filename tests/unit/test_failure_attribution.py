from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from trajpatch.analysis import (
    analyze_locomo_run_failures,
    diff_locomo_failure_reports,
    load_incomplete_run_diagnostics,
    print_incomplete_run_diagnostics,
    print_locomo_failure_diff,
    print_locomo_failure_report,
)
from trajpatch.analysis.failure_attribution import (
    _build_direct_vs_routed_retrieval_diagnostics,
    _build_offline_parameter_diagnostics,
    _classify_grounded_answer_failure,
    _compute_direct_trajectory_ablation_fields,
    _compute_offline_parameter_fields,
    _evaluation_filter_fields_from_row,
    _family_ranking_fields_from_metadata,
    _page_family_fields_from_retrieval_metadata,
    _page_granularity_fields_from_page_ids,
    _page_granularity_fields_from_retrieval_metadata,
    _routing_text_marker_fields_for_pages,
    _temporal_grounding_fields,
    _wiki_fragmentation_fields_from_page_tables,
)
from trajpatch.memory.facets import facet_value_key
from trajpatch.storage.database import create_schema
from trajpatch.storage.models import (
    ClaimRecord,
    EmbeddingRecord,
    EpisodicMemorySnapshot,
    RetrievalEvent,
    WikiPageRecord,
)
from trajpatch.storage.repository import TrajPatchStore
from trajpatch.types import NormalizedMessage

try:
    from typer.testing import CliRunner
except ModuleNotFoundError:  # pragma: no cover - depends on test environment extras
    CliRunner = None
    app = None
    runner = None
else:
    from trajpatch.cli import app

    runner = CliRunner()


def test_incomplete_run_diagnostics_loads_and_prints_run_failed(tmp_path: Path) -> None:
    run_dir = tmp_path / "scope" / "run-1"
    (run_dir / "status").mkdir(parents=True)
    (run_dir / "trajpatch.sqlite").write_text("", encoding="utf-8")
    (run_dir / "run_failed.json").write_text(
        json.dumps(
            {
                "schema_version": "run_failure_v1",
                "run_id": "run-1",
                "stage": "answer_generation",
                "worker_id": 1,
                "sample_id": "conv-49",
                "query_task_id": "conv-49_qa_1",
                "error_type": "TimeoutError",
                "error_message": "remote request timed out",
                "database_table_counts": {"tables": {"answers": 0}},
                "failed_shard_paths": [str(run_dir / "failed_shards" / "worker-01.sqlite")],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status" / "events.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00",
                "event_type": "answer_generation_start",
                "stage": "answer_generation",
                "sample_id": "conv-49",
                "query_task_id": "conv-49_qa_1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = load_incomplete_run_diagnostics(tmp_path / "scope")
    assert report is not None
    assert report["report_type"] == "incomplete_run"
    assert report["run_failed"]["sample_id"] == "conv-49"

    buffer = io.StringIO()
    print_incomplete_run_diagnostics(report, console=Console(file=buffer, force_terminal=False))
    rendered = buffer.getvalue()
    assert "Incomplete Run Diagnostics" in rendered
    assert "TimeoutError" in rendered


def test_page_granularity_fields_prefer_compact_rows_and_mark_missing_metadata() -> None:
    fields = _page_granularity_fields_from_retrieval_metadata(
        {
            "selected_page_rows_compact": [
                {"page_id": "wiki-medium", "trajectory_count": 4, "medium_page_bonus": 0.035},
                {
                    "page_id": "wiki-singleton",
                    "trajectory_count": 1,
                    "singleton_page_penalty": -0.045,
                    "low_quality_singleton_penalty": -0.08,
                    "singleton_policy": "merge_required_low_quality",
                },
            ],
            "singleton_page_penalty_applied": 1,
            "low_quality_singleton_penalty_applied": 1,
            "medium_page_bonus_applied": 1,
            "selected_singleton_page_ids": ["wiki-singleton"],
            "selected_medium_page_ids": ["wiki-medium"],
        }
    )

    assert fields["page_granularity_metadata_available"] is True
    assert fields["selected_singleton_page_count"] == 1
    assert fields["selected_medium_granularity_page_count"] == 1
    assert fields["selected_low_quality_singleton_page_count"] == 1
    assert fields["low_quality_singleton_penalty_applied"] == 1
    assert fields["selected_page_trajectory_count_histogram"] == {"1": 1, "4": 1}
    assert fields["selected_medium_page_ids"] == ["wiki-medium"]

    missing = _page_granularity_fields_from_retrieval_metadata({})

    assert missing["page_granularity_metadata_available"] is False
    assert missing["selected_singleton_page_count"] is None
    assert missing["selected_medium_granularity_page_count"] is None


def test_evaluation_filter_fields_from_row_are_report_ready() -> None:
    fields = _evaluation_filter_fields_from_row(
        {
            "metadata": {
                "evaluation_filter": {
                    "text_only_eligible": False,
                    "excluded_from_text_only": True,
                    "visual_dependency_type": "ocr_text_on_image",
                    "exclusion_reason": "ocr_text_on_image",
                    "gold_items_missing_from_text_input": ["Nothing is Impossible"],
                    "gold_evidence_image_refs": ["D7:8"],
                }
            }
        }
    )

    assert fields["text_only_filter_available"] is True
    assert fields["text_only_eligible"] is False
    assert fields["excluded_from_text_only"] is True
    assert fields["visual_dependency_type"] == "ocr_text_on_image"
    assert fields["text_only_missing_gold_items"] == ["Nothing is Impossible"]


def test_page_granularity_fields_can_fallback_to_sqlite_page_ids() -> None:
    fields = _page_granularity_fields_from_page_ids(
        ["wiki-a", "wiki-b", "wiki-c"],
        {
            "wiki-a": ["traj-a"],
            "wiki-b": ["traj-b", "traj-c", "traj-d"],
            "wiki-c": ["traj-e", "traj-f", "traj-g", "traj-h"],
        },
    )

    assert fields["page_granularity_metadata_available"] is True
    assert fields["page_granularity_metadata_source"] == "sqlite_wiki_pages_fallback"
    assert fields["selected_singleton_page_count"] == 1
    assert fields["selected_medium_granularity_page_count"] == 2
    assert fields["selected_page_trajectory_count_histogram"] == {"1": 1, "3": 1, "4": 1}


def test_wiki_fragmentation_fields_can_fallback_to_sqlite_pages() -> None:
    fields = _wiki_fragmentation_fields_from_page_tables(
        "conv-x",
        {"conv-x": {"wiki-index", "wiki-a", "wiki-b", "wiki-c"}},
        {
            "wiki-index": ["traj-a", "traj-b", "traj-c"],
            "wiki-a": ["traj-a"],
            "wiki-b": ["traj-b", "traj-c", "traj-d"],
            "wiki-c": ["traj-e"],
        },
        {
            "wiki-index": {"seed_type": "index"},
            "wiki-a": {"seed_type": "post_plan_rescue", "wiki_singleton_policy": "allowed_isolated_specific"},
            "wiki-b": {"seed_type": "entity_facet"},
            "wiki-c": {
                "seed_type": "post_plan_rescue",
                "wiki_singleton_policy": "merge_required_low_quality",
                "wiki_singleton_low_quality_merged": True,
            },
        },
        {"wiki-index": "index", "wiki-a": "inventory", "wiki-b": "inventory", "wiki-c": "topic"},
    )

    assert fields["wiki_fragmentation_metadata_source"] == "sqlite_wiki_pages_fallback"
    assert fields["wiki_non_index_page_count"] == 3
    assert fields["wiki_singleton_non_index_page_count"] == 2
    assert fields["wiki_post_plan_rescue_singleton_count"] == 2
    assert fields["wiki_entity_facet_singleton_count"] == 0
    assert fields["wiki_allowed_specific_singleton_count"] == 1
    assert fields["wiki_low_quality_singleton_count"] == 1
    assert fields["wiki_low_quality_singleton_merged_count"] == 1
    assert fields["wiki_overwide_non_index_page_count"] == 0


def test_page_family_fields_from_retrieval_metadata() -> None:
    fields = _page_family_fields_from_retrieval_metadata(
        {
            "selected_page_rows_compact": [
                {
                    "page_id": "wiki-research",
                    "trajectory_count": 4,
                    "page_family_match_score": 0.78,
                    "page_query_object_terms": ["adoption", "research"],
                    "page_query_object_overlap_terms": ["adoption"],
                    "page_family_mismatch_penalty": 0.0,
                }
            ]
        }
    )

    assert fields["page_family_routing_available"] is True
    assert fields["page_family_match_score_max"] == 0.78
    assert fields["page_family_query_object_overlap_terms"] == ["adoption"]


def test_source_event_fields_from_retrieval_metadata() -> None:
    fields = _family_ranking_fields_from_metadata(
        retrieval_metadata={
            "trajectory_source_event_match_scores": [
                {
                    "trajectory_id": "t_gold",
                    "score": 0.9,
                    "matched_terms": ["dance", "competition"],
                    "matched_refs": ["D8:13"],
                    "reason": "source_event_action_object_match",
                }
            ],
            "trajectory_selected_source_event_matches": [
                {
                    "trajectory_id": "t_gold",
                    "score": 0.9,
                    "matched_terms": ["dance", "competition"],
                    "matched_refs": ["D8:13"],
                    "reason": "source_event_action_object_match",
                }
            ],
        },
        gold_trajectory_ids=["t_gold"],
        top_k_trajectory_ids=["t_gold"],
        selection_pool_trajectory_ids=["t_gold"],
    )

    assert fields["gold_source_event_matched_in_selection_pool_count"] == 1
    assert fields["gold_source_event_matched_in_top_k_count"] == 1
    assert fields["trajectory_selected_source_event_match_count"] == 1
    assert fields["trajectory_source_event_matched_terms"] == ["competition", "dance"]
    assert fields["trajectory_source_event_matched_refs"] == ["D8:13"]


def test_grounded_failure_treats_date_answer_as_atomic_value() -> None:
    reason = _classify_grounded_answer_failure(
        gold_answer="29 January, 2023",
        answer_text="10 May 2023",
        query_shape={},
        answer_metadata={"answer_expected_type": "date"},
    )

    assert reason == "answer_selection_error_after_grounding"


def test_grounded_failure_treats_legacy_temporal_answer_as_atomic_value() -> None:
    reason = _classify_grounded_answer_failure(
        gold_answer="May, 2023",
        answer_text="The retrieved evidence describes dance competitions but does not give a date.",
        query_shape={},
        answer_metadata={},
    )

    assert reason == "answer_selection_error_after_grounding"


def test_grounded_failure_still_marks_true_list_omission_incomplete() -> None:
    reason = _classify_grounded_answer_failure(
        gold_answer="running, pottery",
        answer_text="running",
        query_shape={"list_like": True},
        answer_metadata={"answer_expected_type": "list"},
    )

    assert reason == "answer_incomplete_after_grounding"


def test_offline_parameter_fields_use_compact_ranked_rows() -> None:
    retrieval_metadata = {
        "diagnostic_top_n_pages": 50,
        "diagnostic_top_n_trajectories": 50,
        "page_ranked_total_count": 6,
        "trajectory_ranked_total_count": 6,
        "page_ranked_rows_truncated": False,
        "trajectory_ranked_rows_truncated": False,
        "page_ranked_rows_compact_top_n": [
            {"rank": 1, "page_id": "wiki-noise-1", "trajectory_ids": ["t_noise_1"], "trajectory_count": 1},
            {"rank": 2, "page_id": "wiki-noise-2", "trajectory_ids": ["t_noise_2"], "trajectory_count": 1},
            {"rank": 3, "page_id": "wiki-gold-a", "trajectory_ids": ["t_gold_a"], "trajectory_count": 1},
            {"rank": 4, "page_id": "wiki-noise-3", "trajectory_ids": ["t_noise_3"], "trajectory_count": 1},
            {"rank": 5, "page_id": "wiki-gold-b", "trajectory_ids": ["t_gold_b"], "trajectory_count": 1},
            {"rank": 6, "page_id": "wiki-noise-4", "trajectory_ids": ["t_noise_4"], "trajectory_count": 1},
        ],
        "trajectory_ranked_rows_compact_top_n": [
            {"rank": 1, "trajectory_id": "t_noise_1"},
            {"rank": 2, "trajectory_id": "t_noise_2"},
            {"rank": 3, "trajectory_id": "t_gold_a"},
            {"rank": 4, "trajectory_id": "t_noise_3"},
            {"rank": 5, "trajectory_id": "t_gold_b"},
            {"rank": 6, "trajectory_id": "t_noise_4"},
        ],
    }

    fields = _compute_offline_parameter_fields(
        retrieval_metadata=retrieval_metadata,
        gold_refs={"D1:1", "D1:2"},
        gold_page_ids=["wiki-gold-a", "wiki-gold-b"],
        gold_trajectory_ids=["t_gold_a", "t_gold_b"],
        top_t_page_ids=["wiki-noise-1"],
        top_k_trajectory_ids=["t_noise_1"],
        page_to_trajectory_ids={
            "wiki-noise-1": ["t_noise_1"],
            "wiki-noise-2": ["t_noise_2"],
            "wiki-gold-a": ["t_gold_a"],
            "wiki-noise-3": ["t_noise_3"],
            "wiki-gold-b": ["t_gold_b"],
            "wiki-noise-4": ["t_noise_4"],
        },
        trajectory_refs={
            "t_gold_a": {"D1:1"},
            "t_gold_b": {"D1:2"},
            "t_noise_1": {"D9:1"},
            "t_noise_2": {"D9:2"},
            "t_noise_3": {"D9:3"},
            "t_noise_4": {"D9:4"},
        },
    )

    assert fields["offline_page_diagnostic_mode"] == "compact_ranked_top_n"
    assert fields["offline_trajectory_diagnostic_mode"] == "compact_ranked_top_n"
    assert fields["offline_min_t_saved_to_cover_all_gold_pages"] == 5
    assert fields["offline_min_t_saved_to_cover_all_gold_trajectories"] == 5
    assert fields["offline_min_k_saved_to_cover_all_gold_trajectories"] == 5
    assert fields["offline_min_k_saved_to_cover_all_gold_refs"] == 5
    assert fields["offline_page_cutoff_diagnostics"]["5"]["page_universe_can_cover_all_gold_refs"] is True
    assert fields["offline_page_cutoff_diagnostics"]["10"]["not_observed_within_saved_cutoff"] is True
    assert fields["offline_trajectory_cutoff_diagnostics"]["5"]["top_k_can_cover_all_gold_refs"] is True
    assert fields["offline_trajectory_cutoff_diagnostics"]["10"]["not_observed_within_saved_cutoff"] is True


def test_offline_parameter_diagnostics_aggregate_compact_and_legacy_modes() -> None:
    compact = {
        "offline_page_diagnostic_mode": "compact_ranked_top_n",
        "offline_trajectory_diagnostic_mode": "compact_ranked_top_n",
        "offline_saved_page_rank_limit": 50,
        "offline_saved_trajectory_rank_limit": 50,
        "offline_page_cutoff_diagnostics": {
            "5": {
                "observed_within_saved_cutoff": True,
                "gold_page_count": 2,
                "all_gold_pages_in_cutoff": True,
                "gold_page_recall": 1.0,
                "gold_trajectory_count": 2,
                "all_gold_trajectories_in_selected_pages_at_cutoff": True,
                "page_universe_can_cover_all_gold_refs": True,
                "page_universe_gold_ref_coverage_rate": 1.0,
                "selected_page_trajectory_count": 5,
            }
        },
        "offline_trajectory_cutoff_diagnostics": {
            "5": {
                "observed_within_saved_cutoff": True,
                "gold_trajectory_count": 2,
                "all_gold_trajectories_in_cutoff": True,
                "gold_trajectory_recall": 1.0,
                "top_k_can_cover_all_gold_refs": True,
                "top_k_gold_ref_coverage_rate": 1.0,
            }
        },
    }
    legacy = {
        **compact,
        "offline_page_diagnostic_mode": "legacy_selected_only",
        "offline_trajectory_diagnostic_mode": "legacy_selected_only",
    }

    diagnostics = _build_offline_parameter_diagnostics([compact, legacy], [compact])

    assert diagnostics["diagnostic_mode"] == "compact_ranked_top_n"
    assert diagnostics["page_diagnostic_mode_counts"]["compact_ranked_top_n"] == 1
    assert diagnostics["page_diagnostic_mode_counts"]["legacy_selected_only"] == 1
    assert diagnostics["page_cutoffs"]["5"]["all_queries"]["all_gold_pages_rate"] == pytest.approx(1.0)
    assert diagnostics["trajectory_cutoffs"]["5"]["failed_queries"]["top_k_can_cover_all_gold_refs_rate"] == pytest.approx(1.0)


def test_direct_trajectory_ablation_emits_compact_score_rows_without_raw_text() -> None:
    fields = _compute_direct_trajectory_ablation_fields(
        sample_id="sample-direct",
        question="What did Caroline research?",
        gold_refs={"D1:1"},
        gold_trajectory_ids=["t_gold"],
        routed_top_k_trajectory_ids=["t_noise"],
        selected_page_trajectory_ids=["t_noise"],
        query_entities=["Caroline"],
        query_facets={"tags": [], "values": ["research_topic"]},
        query_shape={"list_like": True, "item_family": "research_topic"},
        top_k=1,
        sample_to_trajectories={"sample-direct": {"t_gold", "t_noise"}},
        trajectory_metadata={
            "t_gold": {
                "retrieval_summary_text": "Caroline researched adoption agencies.",
                "source_surface_terms_v1": ["adoption agencies"],
                "trajectory_historical_item_terms_v1": ["research", "adoption agencies"],
                "entity_mentions": ["Caroline"],
            },
            "t_noise": {
                "retrieval_summary_text": "Caroline went camping.",
                "entity_mentions": ["Caroline"],
            },
        },
        claims_by_trajectory={
            "t_gold": [{"text": "Caroline researched adoption agencies.", "exact_terms": ["adoption agencies"]}],
            "t_noise": [{"text": "Caroline went camping."}],
        },
        trajectory_refs={"t_gold": {"D1:1"}, "t_noise": {"D1:2"}},
        trajectory_lengths={"t_gold": 2, "t_noise": 1},
    )

    assert fields["direct_trajectory_top_k_ids"] == ["t_gold"]
    assert fields["direct_page_routing_bottleneck_suspected"] is True
    assert fields["routed_vs_direct_top_k_overlap_count"] == 0
    assert fields["direct_top_k_not_in_selected_page_universe_count"] == 1
    assert fields["direct_top_k_estimated_context_token_count"] > 0
    compact_row = fields["direct_trajectory_ranked_rows_compact_top_n"][0]
    assert compact_row["trajectory_id"] == "t_gold"
    assert "adoption agencies" in compact_row["source_surface_terms"]
    forbidden_keys = {"raw_source_text", "routing_text", "markdown_text", "prompt_context", "answer_context"}
    assert not (forbidden_keys & set(compact_row))
    assert fields["direct_trajectory_cutoff_diagnostics"]["5"]["not_observed_within_saved_cutoff"] is True


def test_direct_vs_routed_diagnostics_aggregate_cutoffs_and_overlap() -> None:
    row = {
        "query_shape": {"list_like": True, "item_family": "book"},
        "gold_trajectory_count": 2,
        "gold_trajectory_ids": ["t_gold_a", "t_gold_b"],
        "gold_refs": ["D1:1", "D1:2"],
        "gold_trajectory_recall_at_k": 0.5,
        "selected_page_universe_size": 3,
        "direct_gold_trajectory_count_in_top_k": 2,
        "direct_gold_trajectory_recall_at_k": 1.0,
        "direct_top_k_can_cover_all_gold_refs": True,
        "direct_top_k_gold_ref_coverage_rate": 1.0,
        "direct_vs_page_routed_recall_delta": 0.5,
        "direct_page_routing_bottleneck_suspected": True,
        "direct_trajectory_candidate_universe_size": 30,
        "direct_candidate_universe_to_routed_universe_ratio": 10.0,
        "routed_vs_direct_top_k_overlap_rate": 0.2,
        "direct_top_k_not_in_selected_page_universe_count": 2,
        "direct_top_k_estimated_context_token_count": 420,
        "direct_trajectory_cutoff_diagnostics": {
            "5": {
                "observed_within_saved_cutoff": True,
                "gold_trajectory_count": 2,
                "gold_trajectory_count_in_cutoff": 2,
                "gold_trajectory_recall": 1.0,
                "all_gold_trajectories_in_cutoff": True,
                "top_k_can_cover_all_gold_refs": True,
                "top_k_gold_ref_coverage_rate": 1.0,
                "estimated_context_token_count": 420,
                "estimated_source_ref_count": 4,
                "estimated_snapshot_count": 3,
            }
        },
    }

    diagnostics = _build_direct_vs_routed_retrieval_diagnostics([row], [row])

    assert diagnostics["diagnostic_mode"] == "metadata_direct_retrieval_ablation"
    assert diagnostics["all_queries"]["mean_direct_gold_trajectory_recall_at_k"] == pytest.approx(1.0)
    assert diagnostics["all_queries"]["mean_routed_vs_direct_top_k_overlap_rate"] == pytest.approx(0.2)
    assert diagnostics["direct_cutoffs"]["5"]["all_queries"]["all_gold_trajectories_rate"] == pytest.approx(1.0)
    assert diagnostics["direct_cutoffs"]["5"]["all_queries"]["mean_estimated_context_token_count"] == pytest.approx(420)
    assert "item_family:book" in diagnostics["by_query_shape"]


def test_routing_text_marker_fields_for_pages_reports_cleaning_status() -> None:
    fields = _routing_text_marker_fields_for_pages(
        ["wiki-clean", "wiki-leaky"],
        {
            "wiki-clean": {"routing_text_cleaned": True, "routing_text_internal_marker_count": 0},
            "wiki-leaky": {"routing_text_cleaned": True, "routing_text_internal_marker_count": 2},
        },
    )

    assert fields["routing_text_cleaned"] is True
    assert fields["routing_text_internal_marker_count"] == 2
    assert fields["routing_text_internal_marker_page_ids"] == ["wiki-leaky"]


def test_temporal_grounding_fields_detect_missing_prompt_anchor() -> None:
    raw_messages_by_id = {
        "conv-26-m0396": {
            "id": "conv-26-m0396",
            "source_ref": "D18:17",
            "content": "Yup, we just did it yesterday after the road trip.",
            "occurred_at": "6:55 pm on 20 October, 2023",
        }
    }
    raw_message_ids_by_ref = {"D18:17": {"conv-26-m0396"}}

    missing = _temporal_grounding_fields(
        question="When did Melanie go on a hike after the roadtrip?",
        gold_refs={"D18:17"},
        grounded_refs={"D18:17"},
        retrieval_metadata={},
        raw_messages_by_id=raw_messages_by_id,
        raw_message_ids_by_ref=raw_message_ids_by_ref,
    )
    visible = _temporal_grounding_fields(
        question="When did Melanie go on a hike after the roadtrip?",
        gold_refs={"D18:17"},
        grounded_refs={"D18:17"},
        retrieval_metadata={
            "source_message_time_anchors": {"D18:17": "6:55 pm on 20 October, 2023"},
            "temporal_anchor_hint_count": 1,
        },
        raw_messages_by_id=raw_messages_by_id,
        raw_message_ids_by_ref=raw_message_ids_by_ref,
    )

    assert missing["temporal_anchor_available_in_raw"] is True
    assert missing["temporal_anchor_visible_in_prompt"] is False
    assert missing["temporal_anchor_missing_in_answer_context"] is True
    assert missing["relative_time_terms_present"] == ["yesterday"]
    assert visible["temporal_anchor_visible_in_prompt"] is True
    assert visible["temporal_anchor_missing_in_answer_context"] is False


def _add_raw_messages(
    store: TrajPatchStore,
    sample_id: str,
    raw_messages: list[dict[str, str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for turn_index, spec in enumerate(raw_messages, start=1):
        message = NormalizedMessage(
            role=spec.get("role", "user"),
            content=spec["content"],
            turn_index=turn_index,
            speaker_name=spec.get("speaker_name"),
            source_ref=spec["ref"],
        )
        record = store.add_raw_message(sample_id, "locomo", message)
        mapping[spec["ref"]] = record.id
    return mapping


def _add_episodic_snapshot(
    store: TrajPatchStore,
    *,
    sample_id: str,
    label: str,
    ordinal: int,
    refs: list[str],
    raw_ids_by_ref: dict[str, str],
    claim_text: str,
    claim_metadata: dict | None = None,
    snapshot_metadata: dict | None = None,
    trajectory_metadata: dict | None = None,
) -> tuple[str, str]:
    trajectory = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label=label,
        strict_matching=False,
        max_length=4,
        metadata=trajectory_metadata or {},
    )
    snapshot = EpisodicMemorySnapshot(
        id=f"{trajectory.id}-v001",
        trajectory_id=trajectory.id,
        version=1,
        timestamp=f"2026-04-18T00:00:{ordinal:02d}Z",
        links_json=[raw_ids_by_ref[ref] for ref in refs],
        summary_content=label,
        context=label,
        keywords_json=[label],
        status_flags_json=["active"],
        embedding_ref=None,
        semantic_text=label,
        raw_text=label,
        metadata_json=snapshot_metadata or {},
    )
    store.save_episodic_snapshot(snapshot)
    store.replace_claims_for_snapshot(
        [
            ClaimRecord(
                id=f"{snapshot.id}:claim",
                snapshot_id=snapshot.id,
                trajectory_id=trajectory.id,
                claim_id=f"{trajectory.id}-c001",
                text=claim_text,
                status="active",
                source_message_ids_json=[raw_ids_by_ref[ref] for ref in refs],
                parent_claim_id=None,
                revised_from_claim_id=None,
                metadata_json=claim_metadata or {},
            )
        ]
    )
    return trajectory.id, snapshot.id


def _add_extra_snapshot(
    store: TrajPatchStore,
    *,
    trajectory_id: str,
    ordinal: int,
    version: int,
    refs: list[str],
    raw_ids_by_ref: dict[str, str],
    label: str,
    claim_text: str | None = None,
    claim_metadata: dict | None = None,
    snapshot_metadata: dict | None = None,
    snapshot_suffix: str | None = None,
) -> str:
    snapshot_id = (
        f"{trajectory_id}-{snapshot_suffix}"
        if snapshot_suffix
        else f"{trajectory_id}-v{version:03d}"
    )
    snapshot = EpisodicMemorySnapshot(
        id=snapshot_id,
        trajectory_id=trajectory_id,
        version=version,
        timestamp=f"2026-04-18T00:00:{ordinal + version:02d}Z",
        links_json=[raw_ids_by_ref[ref] for ref in refs],
        summary_content=label,
        context=label,
        keywords_json=[label],
        status_flags_json=["active"],
        embedding_ref=None,
        semantic_text=label,
        raw_text=label,
        metadata_json=snapshot_metadata or {},
    )
    store.save_episodic_snapshot(snapshot)
    if claim_text:
        store.replace_claims_for_snapshot(
            [
                ClaimRecord(
                    id=f"{snapshot.id}:claim",
                    snapshot_id=snapshot.id,
                    trajectory_id=trajectory_id,
                    claim_id=f"{trajectory_id}-c{version:03d}",
                    text=claim_text,
                    status="active",
                    source_message_ids_json=[raw_ids_by_ref[ref] for ref in refs],
                    parent_claim_id=None,
                    revised_from_claim_id=None,
                    metadata_json=claim_metadata or {},
                )
            ]
        )
    return snapshot.id


def _evidence_text(refs: list[str] | None) -> str:
    if not refs:
        return "No explicit evidence refs here."
    return "\n".join(f"[{ref}] Evidence for {ref}." for ref in refs)


def _build_run_from_specs(
    run_dir: Path,
    samples_spec: list[dict[str, object]],
    *,
    dataset: str = "locomo",
    m: int = 4,
    summary_payload: dict[str, object] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path = run_dir / "trajpatch.sqlite"
    session_factory = create_schema(database_path)
    session = session_factory()
    store = TrajPatchStore(session)

    detail_rows: list[dict[str, object]] = []
    for ordinal, spec in enumerate(samples_spec, start=1):
        sample_id = str(spec["sample_id"])
        raw_ids_by_ref = _add_raw_messages(store, sample_id, list(spec["raw_messages"]))
        trajectory_ids: dict[str, str] = {}
        snapshot_ids: dict[str, str] = {}
        trajectory_specs = dict(spec["trajectories"])
        for label, trajectory_spec in dict(spec["trajectories"]).items():
            trajectory_id, snapshot_id = _add_episodic_snapshot(
                store,
                sample_id=sample_id,
                label=label,
                ordinal=ordinal,
                refs=list(trajectory_spec["refs"]),
                raw_ids_by_ref=raw_ids_by_ref,
                claim_text=str(trajectory_spec["claim_text"]),
                claim_metadata=dict(trajectory_spec.get("claim_metadata") or {}),
                snapshot_metadata=dict(trajectory_spec.get("snapshot_metadata") or {}),
                trajectory_metadata=dict(trajectory_spec.get("trajectory_metadata") or {}),
            )
            trajectory_ids[label] = trajectory_id
            snapshot_ids[label] = snapshot_id
            for extra_snapshot in list(trajectory_spec.get("extra_snapshots") or []):
                extra_snapshot_spec = dict(extra_snapshot)
                version = int(extra_snapshot_spec.get("version", 1))
                extra_label = str(extra_snapshot_spec.get("label", f"{label}_v{version:03d}"))
                snapshot_ids[extra_label] = _add_extra_snapshot(
                    store,
                    trajectory_id=trajectory_id,
                    ordinal=ordinal,
                    version=version,
                    refs=list(extra_snapshot_spec.get("refs") or []),
                    raw_ids_by_ref=raw_ids_by_ref,
                    label=extra_label,
                    claim_text=(
                        str(extra_snapshot_spec["claim_text"])
                        if extra_snapshot_spec.get("claim_text") is not None
                        else None
                    ),
                    claim_metadata=dict(extra_snapshot_spec.get("claim_metadata") or {}),
                    snapshot_metadata=dict(extra_snapshot_spec.get("snapshot_metadata") or {}),
                    snapshot_suffix=(
                        str(extra_snapshot_spec["snapshot_suffix"])
                        if extra_snapshot_spec.get("snapshot_suffix") is not None
                        else None
                    ),
                )

        page_ids: list[str] = []
        page_ids_by_label: dict[str, str] = {}
        for index, (label, trajectory_id) in enumerate(trajectory_ids.items(), start=1):
            page_id = f"{sample_id}-page-{index:03d}"
            page_spec = dict(trajectory_specs[label])
            session.add(
                WikiPageRecord(
                    id=page_id,
                    sample_id=sample_id,
                    dataset_name="locomo",
                    page_type=str(page_spec.get("page_type", "topic")),
                    title=str(page_spec.get("page_title", f"{label} page")),
                    slug=f"{sample_id}-{label}",
                    markdown_text=str(page_spec.get("page_markdown_text", f"{label} page")),
                    keywords_json=list(page_spec.get("page_keywords", [label])),
                    trajectory_ids_json=[trajectory_id],
                    linked_page_ids_json=[],
                    entity_names_json=list(page_spec.get("page_entities", [])),
                    embedding_id=None,
                    metadata_json={},
                )
            )
            page_ids.append(page_id)
            page_ids_by_label[label] = page_id

        retrieval_event_id = f"{sample_id}-retrieval"
        retrieval_metadata = dict(spec.get("retrieval_metadata") or {})
        coarse_ranked = []
        for item in list(retrieval_metadata.get("coarse_ranked_trajectories") or []):
            converted = dict(item)
            trajectory_label = str(converted.pop("trajectory_label"))
            converted["trajectory_id"] = trajectory_ids[trajectory_label]
            latest_label = converted.pop("latest_snapshot_label", None)
            converted["latest_snapshot_id"] = snapshot_ids[str(latest_label)] if latest_label else None
            coarse_ranked.append(converted)
        if coarse_ranked:
            retrieval_metadata["coarse_ranked_trajectories"] = coarse_ranked
            retrieval_metadata.setdefault(
                "coarse_top_k_selected_ids",
                [trajectory_ids[label] for label in list(spec["candidate"])],
            )

        entity_linked_labels = list(retrieval_metadata.get("entity_linked_trajectory_labels") or [])
        if entity_linked_labels:
            retrieval_metadata["entity_linked_trajectory_ids"] = [
                trajectory_ids[label] for label in entity_linked_labels
            ]
            retrieval_metadata["entity_linked_snapshot_ids"] = [
                snapshot_ids[label] for label in entity_linked_labels
            ]
            retrieval_metadata.pop("entity_linked_trajectory_labels", None)
        selected_page_trajectory_labels = list(retrieval_metadata.get("selected_page_trajectory_labels") or [])
        if selected_page_trajectory_labels:
            retrieval_metadata["selected_page_trajectory_ids"] = [
                trajectory_ids[label] for label in selected_page_trajectory_labels
            ]
            retrieval_metadata.pop("selected_page_trajectory_labels", None)
        selection_pool_labels = list(retrieval_metadata.get("trajectory_selection_pool_labels") or [])
        if selection_pool_labels:
            retrieval_metadata["trajectory_selection_pool_ids"] = [
                trajectory_ids[label] for label in selection_pool_labels
            ]
            retrieval_metadata.pop("trajectory_selection_pool_labels", None)
        raw_expanded_snapshot_labels = list(retrieval_metadata.get("raw_expanded_snapshot_labels") or [])
        if raw_expanded_snapshot_labels:
            retrieval_metadata["raw_expanded_snapshot_ids"] = [
                snapshot_ids[label] for label in raw_expanded_snapshot_labels
            ]
            retrieval_metadata.pop("raw_expanded_snapshot_labels", None)

        grounded_source_ids = [
            raw_ids_by_ref[ref]
            for ref in list(spec.get("grounded_refs") or [])
            if ref in raw_ids_by_ref
        ]
        page_selection_labels = list(spec.get("page_candidate") or list(trajectory_ids))
        store.record_retrieval_event(
            RetrievalEvent(
                id=retrieval_event_id,
                sample_id=sample_id,
                query_text=str(spec["question"]),
                query_embedding_json=[0.0],
                top_t_pages=int(spec.get("top_t_pages", max(1, len(page_selection_labels)))),
                top_k=int(spec.get("top_k", 3)),
                snapshot_budget=int(spec.get("snapshot_budget", 6)),
                page_ids_json=[page_ids_by_label[label] for label in page_selection_labels],
                trajectory_ids_json=[trajectory_ids[label] for label in list(spec["candidate"])],
                snapshot_ids_json=[snapshot_ids[label] for label in list(spec["hits"])],
                expanded_snapshot_ids_json=[snapshot_ids[label] for label in list(spec["expanded"])],
                source_message_ids_json=grounded_source_ids,
                latency_ms=1.0,
                metadata_json=retrieval_metadata,
            )
        )

        query_metadata = {
            "category": 1,
            "category_name": "multi_hop",
            "evidence_only_conversation": _evidence_text(list(spec.get("gold_refs") or [])),
        }
        query_metadata.update(dict(spec.get("query_metadata_extra") or {}))
        detail_rows.append(
            {
                "sample_id": sample_id,
                "query_task_id": str(spec.get("query_task_id", f"{sample_id}_qa_0")),
                "question": str(spec["question"]),
                "gold_answer": spec.get("gold_answer"),
                "answer_text": spec.get("answer_text", f"answer for {sample_id}"),
                "judge_verdict": spec.get("judge_verdict", "incorrect"),
                "judge_rationale": spec.get("judge_rationale", f"judge rationale for {sample_id}"),
                "retrieval_event_id": retrieval_event_id,
                "answer_prompt": "prompt",
                "metadata": {
                    "query_metadata": query_metadata,
                    "answer_metadata": dict(spec.get("answer_metadata") or {}),
                    "judge_metadata": dict(spec.get("judge_metadata") or {}),
                },
            }
        )

    session.commit()
    session.close()

    details_payload = {
        "run_meta": {
            "run_id": run_dir.name,
            "dataset": dataset,
            "dataset_scope_key": "multi_hop",
            "m": m,
        },
        "samples": detail_rows,
    }
    (run_dir / "details.json").write_text(json.dumps(details_payload), encoding="utf-8")
    if summary_payload is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    return run_dir


def _new_style_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-memory",
            "question": "What did Andrew do?",
            "gold_answer": "hiked",
            "gold_refs": ["D0:1"],
            "raw_messages": [
                {"ref": "D0:1", "content": "Andrew hiked today.", "speaker_name": "Andrew"},
                {"ref": "D0:2", "content": "Andrew painted today.", "speaker_name": "Andrew"},
            ],
            "trajectories": {
                "t_bad": {
                    "refs": ["D0:2"],
                    "claim_text": "Andrew painted today.",
                }
            },
            "candidate": ["t_bad"],
            "hits": ["t_bad"],
            "expanded": ["t_bad"],
            "grounded_refs": ["D0:2"],
            "retrieval_metadata": {"source_refs": ["D0:2"]},
        },
        {
            "sample_id": "sample-coarse",
            "question": "What did Caroline research?",
            "gold_answer": "adoption agencies",
            "gold_refs": ["D1:1"],
            "raw_messages": [
                {
                    "ref": "D1:1",
                    "content": "Caroline is researching adoption agencies.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D1:2",
                    "content": "Caroline is planning a nature outing.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_good": {
                    "refs": ["D1:1"],
                    "claim_text": "Caroline researches adoption agencies.",
                    "claim_metadata": {
                        "facets_v1": [
                            {
                                "entity": "Caroline",
                                "relation": "research_topic",
                                "value": "adoption agencies",
                                "value_span": "adoption agencies",
                                "facet_type": "topic",
                                "source": "claim_text",
                                "confidence": 0.96,
                            }
                        ]
                    },
                    "trajectory_metadata": {
                        "entity_mentions": ["Caroline"],
                        "facet_tags": ["research_topic"],
                        "facet_values": [facet_value_key("research_topic", "adoption agencies")],
                    },
                },
                "t_bad": {
                    "refs": ["D1:2"],
                    "claim_text": "Caroline plans a nature outing.",
                    "trajectory_metadata": {
                        "entity_mentions": ["Caroline"],
                        "facet_tags": [],
                        "facet_values": [],
                    },
                },
                "t_other": {
                    "refs": ["D1:2"],
                    "claim_text": "Caroline is thinking about school.",
                },
                "t_other_2": {
                    "refs": ["D1:2"],
                    "claim_text": "Caroline received praise from friends.",
                },
            },
            "candidate": ["t_bad"],
            "hits": ["t_bad"],
            "expanded": ["t_bad"],
            "grounded_refs": ["D1:2"],
            "retrieval_metadata": {
                "source_refs": ["D1:2"],
                "query_entities": ["Caroline"],
                "query_facets": {"tags": ["research_topic"], "values": []},
                "coarse_ranked_trajectories": [
                    {
                        "trajectory_label": "t_bad",
                        "latest_snapshot_label": "t_bad",
                        "total_score": 0.82,
                        "trajectory_cluster_similarity": 0.72,
                        "latest_snapshot_similarity": 0.64,
                        "lexical_overlap": 0.05,
                        "entity_match_boost": 0.10,
                        "facet_tag_boost": 0.00,
                        "facet_value_boost": 0.00,
                        "entity_mentions": ["Caroline"],
                        "facet_tags": [],
                        "facet_values": [],
                    },
                    {
                        "trajectory_label": "t_other",
                        "latest_snapshot_label": "t_other",
                        "total_score": 0.76,
                        "trajectory_cluster_similarity": 0.66,
                        "latest_snapshot_similarity": 0.61,
                        "lexical_overlap": 0.04,
                        "entity_match_boost": 0.10,
                        "facet_tag_boost": 0.00,
                        "facet_value_boost": 0.00,
                        "entity_mentions": ["Caroline"],
                        "facet_tags": [],
                        "facet_values": [],
                    },
                    {
                        "trajectory_label": "t_other_2",
                        "latest_snapshot_label": "t_other_2",
                        "total_score": 0.70,
                        "trajectory_cluster_similarity": 0.62,
                        "latest_snapshot_similarity": 0.55,
                        "lexical_overlap": 0.00,
                        "entity_match_boost": 0.10,
                        "facet_tag_boost": 0.00,
                        "facet_value_boost": 0.00,
                        "entity_mentions": ["Caroline"],
                        "facet_tags": [],
                        "facet_values": [],
                    },
                    {
                        "trajectory_label": "t_good",
                        "latest_snapshot_label": "t_good",
                        "total_score": 0.68,
                        "trajectory_cluster_similarity": 0.55,
                        "latest_snapshot_similarity": 0.60,
                        "lexical_overlap": 0.10,
                        "entity_match_boost": 0.10,
                        "facet_tag_boost": 0.05,
                        "facet_value_boost": 0.00,
                        "entity_mentions": ["Caroline"],
                        "facet_tags": ["research_topic"],
                        "facet_values": [facet_value_key("research_topic", "adoption agencies")],
                    },
                ],
            },
        },
        {
            "sample_id": "sample-answer",
            "question": "What books has Tim read?",
            "gold_answer": "The Hobbit",
            "gold_refs": ["D2:1"],
            "raw_messages": [
                {"ref": "D2:1", "content": "Tim read The Hobbit.", "speaker_name": "Tim"},
                {"ref": "D2:2", "content": "Tim likes fantasy novels.", "speaker_name": "Tim"},
            ],
            "trajectories": {
                "t_bad": {
                    "refs": ["D2:2"],
                    "claim_text": "Tim likes fantasy novels.",
                },
                "t_good": {
                    "refs": ["D2:1"],
                    "claim_text": "Tim read The Hobbit.",
                },
            },
            "candidate": ["t_bad", "t_good"],
            "hits": ["t_bad", "t_good"],
            "expanded": ["t_bad", "t_good"],
            "grounded_refs": ["D2:1"],
            "retrieval_metadata": {"source_refs": ["D2:1"]},
        },
        {
            "sample_id": "sample-facet",
            "question": "What is Caroline's relationship status?",
            "gold_answer": "Single",
            "gold_refs": ["D3:1"],
            "raw_messages": [
                {
                    "ref": "D3:1",
                    "content": "It will be tough as a single parent.",
                    "speaker_name": "Caroline",
                }
            ],
            "trajectories": {
                "t_good": {
                    "refs": ["D3:1"],
                    "claim_text": "Caroline plans to create a family.",
                    "trajectory_metadata": {
                        "entity_mentions": ["Caroline"],
                        "facet_tags": [],
                        "facet_values": [],
                    },
                }
            },
            "candidate": ["t_good"],
            "hits": ["t_good"],
            "expanded": ["t_good"],
            "grounded_refs": ["D3:1"],
            "retrieval_metadata": {"source_refs": ["D3:1"]},
        },
        {
            "sample_id": "sample-fragmented",
            "question": "Where did Caroline move from four years ago?",
            "gold_answer": "Sweden",
            "gold_refs": ["D4:1", "D4:2"],
            "raw_messages": [
                {
                    "ref": "D4:1",
                    "content": "Caroline moved from her home country four years ago.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D4:2",
                    "content": "My home country, Sweden, is where my grandmother lives.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_move": {
                    "refs": ["D4:1"],
                    "claim_text": "Caroline moved from her home country four years ago.",
                    "trajectory_metadata": {
                        "entity_mentions": ["Caroline"],
                        "facet_tags": [],
                        "facet_values": [],
                    },
                },
                "t_home": {
                    "refs": ["D4:2"],
                    "claim_text": "Caroline's home country is Sweden.",
                    "claim_metadata": {
                        "facets_v1": [
                            {
                                "entity": "Caroline",
                                "relation": "home_country",
                                "value": "Sweden",
                                "value_span": "home country, Sweden",
                                "facet_type": "origin",
                                "source": "claim_text",
                                "confidence": 0.97,
                            }
                        ]
                    },
                    "trajectory_metadata": {
                        "entity_mentions": ["Caroline"],
                        "facet_tags": ["home_country"],
                        "facet_values": [facet_value_key("home_country", "Sweden")],
                    },
                },
            },
            "candidate": ["t_move"],
            "hits": ["t_move"],
            "expanded": ["t_move", "t_home"],
            "grounded_refs": [],
            "retrieval_metadata": {
                "source_refs": [],
                "query_entities": ["Caroline"],
                "query_facets": {"tags": ["home_country"], "values": []},
                "entity_linked_trajectory_labels": ["t_home"],
            },
        },
        {
            "sample_id": "sample-pass",
            "question": "Where does Maria live?",
            "gold_answer": "London",
            "gold_refs": ["D5:1"],
            "raw_messages": [
                {"ref": "D5:1", "content": "Maria lives in London.", "speaker_name": "Maria"}
            ],
            "trajectories": {
                "t_good": {
                    "refs": ["D5:1"],
                    "claim_text": "Maria lives in London.",
                }
            },
            "candidate": ["t_good"],
            "hits": ["t_good"],
            "expanded": ["t_good"],
            "grounded_refs": ["D5:1"],
            "judge_verdict": "correct",
            "retrieval_metadata": {"source_refs": ["D5:1"]},
        },
    ]


def _judge_diagnostic_samples() -> list[dict[str, object]]:
    samples = _new_style_samples()
    for sample in samples:
        if sample["sample_id"] == "sample-memory":
            sample["judge_metadata"] = {
                "judge_mode": "text_fallback",
                "structured_requested": True,
                "structured_success": False,
                "structured_fallback_used": True,
                "structured_fallback_reason": "invalid schema rejected by provider",
                "structured_fallback_category": "structured_schema_error",
                "judge_execution_failed": False,
            }
        elif sample["sample_id"] == "sample-coarse":
            sample["judge_verdict"] = "correct"
            sample["judge_metadata"] = {
                "judge_mode": "structured",
                "structured_requested": True,
                "structured_success": True,
                "structured_fallback_used": False,
                "structured_fallback_reason": None,
                "structured_fallback_category": None,
                "judge_execution_failed": False,
            }
        elif sample["sample_id"] == "sample-answer":
            sample["judge_verdict"] = "judge_error"
            sample["judge_rationale"] = "Judge execution failed: ParserValidationError"
            sample["judge_metadata"] = {
                "judge_mode": "text_fallback",
                "structured_requested": True,
                "structured_success": False,
                "structured_fallback_used": True,
                "structured_fallback_reason": "invalid schema rejected by provider",
                "structured_fallback_category": "text_parser_error",
                "judge_execution_failed": True,
                "judge_error_type": "ParserValidationError",
                "judge_error_message": "Missing or invalid judge verdict.",
            }
    return samples


def _old_style_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-old-style",
            "question": "What is Caroline's relationship status?",
            "gold_answer": "Single",
            "gold_refs": ["D9:1"],
            "raw_messages": [
                {
                    "ref": "D9:1",
                    "content": "It will be tough as a single parent.",
                    "speaker_name": "Caroline",
                }
            ],
            "trajectories": {
                "t_good": {
                    "refs": ["D9:1"],
                    "claim_text": "Caroline plans to create a family.",
                }
            },
            "candidate": ["t_good"],
            "hits": ["t_good"],
            "expanded": ["t_good"],
            "grounded_refs": ["D9:1"],
            "retrieval_metadata": {"source_refs": ["D9:1"]},
        }
    ]


def _multi_hop_coverage_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-single-covered",
            "question": "What is Caroline's relationship status?",
            "gold_answer": "Single",
            "gold_refs": ["D10:1"],
            "raw_messages": [
                {
                    "ref": "D10:1",
                    "content": "It will be tough as a single parent.",
                    "speaker_name": "Caroline",
                }
            ],
            "trajectories": {
                "t_single": {
                    "refs": ["D10:1"],
                    "claim_text": "Caroline is single.",
                }
            },
            "candidate": ["t_single"],
            "hits": ["t_single"],
            "expanded": ["t_single"],
            "grounded_refs": ["D10:1"],
            "retrieval_metadata": {"source_refs": ["D10:1"]},
        },
        {
            "sample_id": "sample-partial-covered",
            "question": "Where did Caroline move from four years ago?",
            "gold_answer": "Sweden",
            "gold_refs": ["D11:1", "D11:2"],
            "raw_messages": [
                {
                    "ref": "D11:1",
                    "content": "Caroline moved from her home country four years ago.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D11:2",
                    "content": "My home country, Sweden, is where my grandmother lives.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_move": {
                    "refs": ["D11:1"],
                    "claim_text": "Caroline moved from her home country four years ago.",
                },
                "t_home": {
                    "refs": ["D11:2"],
                    "claim_text": "Caroline's home country is Sweden.",
                },
            },
            "candidate": ["t_move"],
            "hits": ["t_move"],
            "expanded": ["t_move"],
            "grounded_refs": [],
            "retrieval_metadata": {
                "source_refs": [],
                "trajectory_selection_pool_labels": ["t_move", "t_home"],
                "trajectory_selection_strategy": "coverage_aware_greedy",
            },
        },
        {
            "sample_id": "sample-full-covered",
            "question": "Where did Caroline move from four years ago?",
            "gold_answer": "Sweden",
            "gold_refs": ["D12:1", "D12:2"],
            "raw_messages": [
                {
                    "ref": "D12:1",
                    "content": "Caroline moved from her home country four years ago.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D12:2",
                    "content": "My home country, Sweden, is where my grandmother lives.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_move": {
                    "refs": ["D12:1"],
                    "claim_text": "Caroline moved from her home country four years ago.",
                },
                "t_home": {
                    "refs": ["D12:2"],
                    "claim_text": "Caroline's home country is Sweden.",
                },
            },
            "candidate": ["t_move", "t_home"],
            "hits": ["t_move", "t_home"],
            "expanded": ["t_move", "t_home"],
            "grounded_refs": ["D12:1", "D12:2"],
            "retrieval_metadata": {
                "source_refs": ["D12:1", "D12:2"],
                "trajectory_selection_pool_labels": ["t_move", "t_home"],
                "trajectory_selection_strategy": "coverage_aware_greedy",
            },
        },
    ]


def _preservation_and_stage_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-trajectory-storage",
            "question": "What is Caroline's relationship status?",
            "gold_answer": "Single",
            "gold_refs": ["D20:1"],
            "raw_messages": [
                {
                    "ref": "D20:1",
                    "content": "It will be tough as a single parent.",
                    "speaker_name": "Caroline",
                }
            ],
            "trajectories": {
                "t_single": {
                    "refs": ["D20:1"],
                    "claim_text": "Caroline plans for a family someday.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline plans for a family someday."},
                    "page_markdown_text": "Caroline plans for a family someday.",
                }
            },
            "candidate": ["t_single"],
            "hits": ["t_single"],
            "expanded": ["t_single"],
            "grounded_refs": ["D20:1"],
            "retrieval_metadata": {"source_refs": ["D20:1"]},
        },
        {
            "sample_id": "sample-summary-compression",
            "question": "What book did Tim read?",
            "gold_answer": "The Hobbit",
            "gold_refs": ["D21:1"],
            "raw_messages": [
                {"ref": "D21:1", "content": "Tim read The Hobbit.", "speaker_name": "Tim"}
            ],
            "trajectories": {
                "t_book": {
                    "refs": ["D21:1"],
                    "claim_text": "Tim read The Hobbit.",
                    "trajectory_metadata": {
                        "retrieval_summary_text": "Tim likes fantasy novels.",
                        "trajectory_historical_evidence_card_v1": {
                            "trajectory_id": "t_book",
                            "identity_summary": "Tim reading evidence",
                            "recent_update": "Tim discussed fantasy novels.",
                            "historical_item_terms": ["The Hobbit"],
                            "facet_values": [],
                            "entity_mentions": ["Tim"],
                            "source_anchors": [{"source_ref": "D21:1", "text": "Tim read The Hobbit."}],
                            "drift_cluster_keys": ["The Hobbit"],
                            "display_items": ["The Hobbit"],
                            "display_counts": [],
                            "display_key_facts": ["Tim read The Hobbit."],
                        },
                    },
                    "page_markdown_text": "Tim read The Hobbit.",
                }
            },
            "candidate": ["t_book"],
            "hits": ["t_book"],
            "expanded": ["t_book"],
            "grounded_refs": ["D21:1"],
            "retrieval_metadata": {"source_refs": ["D21:1"]},
        },
        {
            "sample_id": "sample-wiki-compression",
            "question": "Where did Caroline move from?",
            "gold_answer": "Sweden",
            "gold_refs": ["D22:1"],
            "raw_messages": [
                {
                    "ref": "D22:1",
                    "content": "Caroline moved from Sweden.",
                    "speaker_name": "Caroline",
                }
            ],
            "trajectories": {
                "t_move": {
                    "refs": ["D22:1"],
                    "claim_text": "Caroline moved from Sweden.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline moved from Sweden."},
                    "page_markdown_text": "Caroline moved from her home country.",
                }
            },
            "candidate": ["t_move"],
            "hits": ["t_move"],
            "expanded": ["t_move"],
            "grounded_refs": ["D22:1"],
            "retrieval_metadata": {"source_refs": ["D22:1"]},
        },
        {
            "sample_id": "sample-page-routing-fail",
            "question": "What did Caroline research?",
            "gold_answer": "adoption agencies",
            "gold_refs": ["D23:1"],
            "raw_messages": [
                {
                    "ref": "D23:1",
                    "content": "Caroline researched adoption agencies.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D23:2",
                    "content": "Caroline went hiking.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_gold": {
                    "refs": ["D23:1"],
                    "claim_text": "Caroline researched adoption agencies.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline researched adoption agencies."},
                    "page_markdown_text": "Caroline researched adoption agencies.",
                },
                "t_other": {
                    "refs": ["D23:2"],
                    "claim_text": "Caroline went hiking.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline went hiking."},
                    "page_markdown_text": "Caroline went hiking.",
                },
            },
            "page_candidate": ["t_other"],
            "top_t_pages": 1,
            "candidate": ["t_other"],
            "hits": ["t_other"],
            "expanded": ["t_other"],
            "grounded_refs": [],
            "retrieval_metadata": {"source_refs": []},
        },
        {
            "sample_id": "sample-selected-pages-fail",
            "question": "Where did Caroline move from four years ago?",
            "gold_answer": "Sweden",
            "gold_refs": ["D24:1", "D24:2"],
            "raw_messages": [
                {
                    "ref": "D24:1",
                    "content": "Caroline moved from her home country four years ago.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D24:2",
                    "content": "Caroline's home country is Sweden.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_move": {
                    "refs": ["D24:1"],
                    "claim_text": "Caroline moved from her home country four years ago.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline moved from her home country four years ago."},
                    "page_markdown_text": "Caroline moved from her home country four years ago.",
                },
                "t_home": {
                    "refs": ["D24:2"],
                    "claim_text": "Caroline's home country is Sweden.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline's home country is Sweden."},
                    "page_markdown_text": "Caroline's home country is Sweden.",
                },
            },
            "page_candidate": ["t_move", "t_home"],
            "top_t_pages": 2,
            "candidate": ["t_move"],
            "hits": ["t_move"],
            "expanded": ["t_move"],
            "grounded_refs": [],
            "retrieval_metadata": {
                "source_refs": [],
                "selected_page_trajectory_labels": ["t_move"],
            },
        },
        {
            "sample_id": "sample-trajectory-topk-fail",
            "question": "Where did Caroline move from four years ago?",
            "gold_answer": "Sweden",
            "gold_refs": ["D25:1", "D25:2"],
            "raw_messages": [
                {
                    "ref": "D25:1",
                    "content": "Caroline moved from her home country four years ago.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D25:2",
                    "content": "Caroline's home country is Sweden.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_move": {
                    "refs": ["D25:1"],
                    "claim_text": "Caroline moved from her home country four years ago.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline moved from her home country four years ago."},
                    "page_markdown_text": "Caroline moved from her home country four years ago.",
                },
                "t_home": {
                    "refs": ["D25:2"],
                    "claim_text": "Caroline's home country is Sweden.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline's home country is Sweden."},
                    "page_markdown_text": "Caroline's home country is Sweden.",
                },
            },
            "page_candidate": ["t_move", "t_home"],
            "top_t_pages": 2,
            "candidate": ["t_move"],
            "hits": ["t_move"],
            "expanded": ["t_move"],
            "grounded_refs": [],
            "retrieval_metadata": {
                "source_refs": [],
                "selected_page_trajectory_labels": ["t_move", "t_home"],
            },
        },
        {
            "sample_id": "sample-strict-answer-error",
            "question": "What book did Tim read?",
            "gold_answer": "The Hobbit",
            "gold_refs": ["D26:1"],
            "raw_messages": [
                {"ref": "D26:1", "content": "Tim read The Hobbit.", "speaker_name": "Tim"}
            ],
            "trajectories": {
                "t_book": {
                    "refs": ["D26:1"],
                    "claim_text": "Tim read The Hobbit.",
                    "trajectory_metadata": {"retrieval_summary_text": "Tim read The Hobbit."},
                    "page_markdown_text": "Tim read The Hobbit.",
                }
            },
            "page_candidate": ["t_book"],
            "top_t_pages": 1,
            "candidate": ["t_book"],
            "hits": ["t_book"],
            "expanded": ["t_book"],
            "grounded_refs": ["D26:1"],
            "retrieval_metadata": {
                "source_refs": ["D26:1"],
                "selected_page_trajectory_labels": ["t_book"],
            },
        },
    ]


def _compaction_failure_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-snapshot-compaction-miss",
            "question": "Where did Caroline move from four years ago?",
            "gold_answer": "Sweden",
            "gold_refs": ["D30:1", "D30:2"],
            "raw_messages": [
                {
                    "ref": "D30:1",
                    "content": "Caroline moved from her home country four years ago.",
                    "speaker_name": "Caroline",
                },
                {
                    "ref": "D30:2",
                    "content": "Caroline's home country is Sweden.",
                    "speaker_name": "Caroline",
                },
            ],
            "trajectories": {
                "t_move": {
                    "refs": ["D30:1"],
                    "claim_text": "Caroline moved from her home country four years ago.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline moved from her home country four years ago."},
                    "page_markdown_text": "Caroline moved from her home country four years ago.",
                },
                "t_home": {
                    "refs": ["D30:2"],
                    "claim_text": "Caroline's home country is Sweden.",
                    "trajectory_metadata": {"retrieval_summary_text": "Caroline's home country is Sweden."},
                    "page_markdown_text": "Caroline's home country is Sweden.",
                },
            },
            "page_candidate": ["t_move", "t_home"],
            "top_t_pages": 2,
            "candidate": ["t_move", "t_home"],
            "hits": ["t_move"],
            "expanded": ["t_move"],
            "grounded_refs": [],
            "retrieval_metadata": {
                "source_refs": [],
                "selected_page_trajectory_labels": ["t_move", "t_home"],
                "raw_expanded_snapshot_labels": ["t_move", "t_home"],
            },
        },
        {
            "sample_id": "sample-source-compaction-miss",
            "question": "What book did Tim read?",
            "gold_answer": "The Hobbit",
            "gold_refs": ["D31:1"],
            "raw_messages": [
                {"ref": "D31:1", "content": "Tim read The Hobbit.", "speaker_name": "Tim"}
            ],
            "trajectories": {
                "t_book": {
                    "refs": ["D31:1"],
                    "claim_text": "Tim read The Hobbit.",
                    "trajectory_metadata": {"retrieval_summary_text": "Tim read The Hobbit."},
                    "page_markdown_text": "Tim read The Hobbit.",
                }
            },
            "page_candidate": ["t_book"],
            "top_t_pages": 1,
            "candidate": ["t_book"],
            "hits": ["t_book"],
            "expanded": ["t_book"],
            "grounded_refs": [],
            "retrieval_metadata": {
                "source_refs": [],
                "selected_page_trajectory_labels": ["t_book"],
                "raw_expanded_snapshot_labels": ["t_book"],
            },
        },
    ]


def _cutoff_diagnostic_samples() -> list[dict[str, object]]:
    trajectories: dict[str, dict[str, object]] = {}
    raw_messages: list[dict[str, str]] = []
    for index in range(1, 11):
        label = f"t_noise_{index}"
        ref = f"D40:{index + 2}"
        raw_messages.append({"ref": ref, "content": f"Noise fact {index}.", "speaker_name": "Speaker"})
        trajectories[label] = {
            "refs": [ref],
            "claim_text": f"Noise fact {index}.",
            "trajectory_metadata": {"retrieval_summary_text": f"Noise fact {index}."},
            "page_markdown_text": f"Noise fact {index}.",
        }
    raw_messages.extend(
        [
            {"ref": "D40:1", "content": "Caroline moved from Sweden.", "speaker_name": "Caroline"},
            {"ref": "D40:2", "content": "Caroline chose counseling.", "speaker_name": "Caroline"},
        ]
    )
    trajectories["t_gold_a"] = {
        "refs": ["D40:1"],
        "claim_text": "Caroline moved from Sweden.",
        "trajectory_metadata": {"retrieval_summary_text": "Caroline moved from Sweden."},
        "page_markdown_text": "Caroline moved from Sweden.",
    }
    trajectories["t_gold_b"] = {
        "refs": ["D40:2"],
        "claim_text": "Caroline chose counseling.",
        "trajectory_metadata": {"retrieval_summary_text": "Caroline chose counseling."},
        "page_markdown_text": "Caroline chose counseling.",
    }
    return [
        {
            "sample_id": "sample-cutoff",
            "question": "Where did Caroline move from and what path did she choose?",
            "gold_answer": "Sweden and counseling",
            "gold_refs": ["D40:1", "D40:2"],
            "raw_messages": raw_messages,
            "trajectories": trajectories,
            "page_candidate": [
                "t_noise_1",
                "t_noise_2",
                "t_noise_3",
                "t_gold_a",
                "t_noise_4",
                "t_gold_b",
                "t_noise_5",
                "t_noise_6",
                "t_noise_7",
                "t_noise_8",
            ],
            "top_t_pages": 10,
            "candidate": [
                "t_noise_1",
                "t_noise_2",
                "t_gold_a",
                "t_noise_3",
                "t_gold_b",
                "t_noise_4",
                "t_noise_5",
                "t_noise_6",
                "t_noise_7",
                "t_noise_8",
            ],
            "top_k": 10,
            "hits": ["t_gold_a"],
            "expanded": ["t_gold_a", "t_gold_b"],
            "grounded_refs": [],
            "retrieval_metadata": {
                "source_refs": [],
                "selected_page_trajectory_labels": [
                    "t_noise_1",
                    "t_noise_2",
                    "t_noise_3",
                    "t_gold_a",
                    "t_noise_4",
                    "t_gold_b",
                ],
            },
        }
    ]


def _trajectory_length_overprovisioned_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-trajectory-length-over",
            "question": "What city did Caroline finally move to?",
            "gold_answer": "London",
            "gold_refs": ["D50:5"],
            "raw_messages": [
                {"ref": "D50:1", "content": "Caroline started planning a move.", "speaker_name": "Caroline"},
                {"ref": "D50:2", "content": "Caroline compared several cities.", "speaker_name": "Caroline"},
                {"ref": "D50:3", "content": "Caroline considered Manchester.", "speaker_name": "Caroline"},
                {"ref": "D50:4", "content": "Caroline shortlisted London.", "speaker_name": "Caroline"},
                {"ref": "D50:5", "content": "Caroline finally moved to London.", "speaker_name": "Caroline"},
            ],
            "trajectories": {
                "t_story": {
                    "refs": ["D50:1"],
                    "claim_text": "Caroline started planning a move.",
                    "extra_snapshots": [
                        {
                            "label": "t_story_v002",
                            "version": 2,
                            "refs": ["D50:2"],
                            "snapshot_suffix": "zzz",
                        },
                        {
                            "label": "t_story_v003",
                            "version": 3,
                            "refs": ["D50:3"],
                            "snapshot_suffix": "mmm",
                        },
                        {
                            "label": "t_story_v004",
                            "version": 4,
                            "refs": ["D50:4"],
                            "snapshot_suffix": "bbb",
                        },
                        {
                            "label": "t_story_v005",
                            "version": 5,
                            "refs": ["D50:5"],
                            "snapshot_suffix": "aaa",
                        },
                    ],
                    "trajectory_metadata": {
                        "retrieval_summary_text": "Caroline finally moved to London."
                    },
                    "page_markdown_text": "Caroline finally moved to London.",
                }
            },
            "candidate": ["t_story"],
            "hits": ["t_story"],
            "expanded": ["t_story"],
            "grounded_refs": ["D50:5"],
            "retrieval_metadata": {"source_refs": ["D50:5"]},
        }
    ]


def _trajectory_length_pressure_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-trajectory-length-pressure",
            "question": "What place did Tim finally choose?",
            "gold_answer": "Oxford",
            "gold_refs": ["D51:3"],
            "raw_messages": [
                {"ref": "D51:1", "content": "Tim began a university search.", "speaker_name": "Tim"},
                {"ref": "D51:2", "content": "Tim compared Cambridge and Oxford.", "speaker_name": "Tim"},
                {"ref": "D51:3", "content": "Tim finally chose Oxford.", "speaker_name": "Tim"},
            ],
            "trajectories": {
                "t_story": {
                    "refs": ["D51:1"],
                    "claim_text": "Tim began a university search.",
                    "extra_snapshots": [
                        {"label": "t_story_v002", "version": 2, "refs": ["D51:2"]},
                        {"label": "t_story_v003", "version": 3, "refs": ["D51:3"]},
                    ],
                    "trajectory_metadata": {"retrieval_summary_text": "Tim finally chose Oxford."},
                    "page_markdown_text": "Tim finally chose Oxford.",
                }
            },
            "candidate": ["t_story"],
            "hits": ["t_story"],
            "expanded": ["t_story"],
            "grounded_refs": ["D51:3"],
            "retrieval_metadata": {"source_refs": ["D51:3"]},
        }
    ]


def test_analyze_locomo_run_failures_reports_rank_facet_and_fragmentation_diagnostics(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _new_style_samples())

    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=2)

    assert report["run_meta"]["dataset"] == "locomo"
    assert report["totals"]["total_queries"] == 6
    assert report["totals"]["failed_queries"] == 5
    assert report["reason_counts"]["memory_absent"] == 1
    assert report["reason_counts"]["coarse_retrieval_miss"] == 1
    assert report["reason_counts"]["grounding_miss"] == 1
    assert (
        report["reason_counts"]["answer_selection_error_after_grounding"]
        + report["reason_counts"]["answer_incomplete_after_grounding"]
        + report["reason_counts"]["answer_overgenerated_after_grounding"]
    ) == 2
    coarse_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-coarse")
    assert coarse_row["gold_trajectory_rank"] == 4
    assert "facet_present_for_gold_ref" in coarse_row["diagnostic_flags"]
    assert coarse_row["gold_claim_facets"][0]["relation"] == "research_topic"
    answer_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-answer")
    assert answer_row["gold_snapshot_rank"] == 2
    assert answer_row["query_shape"]["list_like"] is True
    facet_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-facet")
    assert "exact_facet_missing" in facet_row["diagnostic_flags"]
    assert "gold_ref_present_but_no_supported_facet" in facet_row["diagnostic_flags"]
    assert "gold_ref_present_but_claim_generalized" in facet_row["diagnostic_flags"]
    assert "gold_item_present_in_claims_rate_over_failed" in report["memory_preservation_diagnostics"]
    fragmented_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-fragmented")
    assert fragmented_row["gold_refs_split_across_trajectories"] is True
    assert fragmented_row["min_gold_covering_trajectory_count"] == 2
    assert fragmented_row["entity_linked_added_relevant_snapshot"] is True
    assert fragmented_row["gold_trajectory_count_in_top_k"] == 1
    assert fragmented_row["gold_trajectory_recall_at_k"] == pytest.approx(0.5)
    assert fragmented_row["top_k_can_cover_all_gold_refs"] is False
    assert fragmented_row["top_k_redundant_entity_cluster_ratio"] is not None
    assert report["rank_diagnostics"]["coarse_rank_observed_count"] == 1
    assert report["facet_diagnostics"]["exact_facet_missing_rate"] == pytest.approx(1 / 5)
    assert report["fragmentation_diagnostics"]["split_across_trajectories_rate"] == pytest.approx(1 / 5)
    assert "wiki_fragmentation_diagnostics" in report
    assert "selected_singleton_page_slot_rate" in report["wiki_fragmentation_diagnostics"]
    assert report["multi_hop_top_k_diagnostics"]["split_case_count"] == 1
    assert report["multi_hop_top_k_diagnostics"]["mean_gold_trajectory_recall_at_k_over_split_cases"] == pytest.approx(0.5)
    assert report["multi_hop_top_k_diagnostics"]["top_k_can_cover_all_gold_refs_rate_over_split_cases"] == pytest.approx(0.0)
    assert "split_case_redundancy_diagnostics" in report


def test_analyze_locomo_run_failures_uses_fallback_gold_evidence_metadata(tmp_path: Path):
    samples = _new_style_samples()
    for sample in samples:
        if sample["sample_id"] == "sample-memory":
            sample["query_metadata_extra"] = {
                "evidence_only_conversation": "",
                "gold_evidence_refs": ["D0:1"],
            }
    run_dir = _build_run_from_specs(tmp_path / "run", samples)

    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=1)

    memory_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-memory")
    assert memory_row["reason"] == "memory_absent"
    assert memory_row["gold_refs"] == ["D0:1"]
    assert report["reason_counts"]["unknown_no_gold_refs"] == 0


def test_analyze_locomo_run_failures_uses_raw_combined_gold_evidence_metadata(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        [
            {
                "sample_id": "conv-26",
                "query_task_id": "conv-26_qa_37",
                "question": "What did Melanie paint recently?",
                "gold_answer": "sunset",
                "gold_refs": ["D8:6", "D9:17"],
                "raw_messages": [
                    {
                        "ref": "D8:6",
                        "content": "Melanie painted a sunset last weekend.",
                        "speaker_name": "Melanie",
                    },
                    {
                        "ref": "D9:17",
                        "content": "Melanie and her kids finished another painting like the last one.",
                        "speaker_name": "Melanie",
                    },
                    {
                        "ref": "D14:30",
                        "content": "Melanie painted a sunflower recently.",
                        "speaker_name": "Melanie",
                    },
                ],
                "trajectories": {
                    "t_sunset_last": {
                        "refs": ["D8:6"],
                        "claim_text": "Melanie painted a sunset last weekend.",
                    },
                    "t_sunset_bridge": {
                        "refs": ["D9:17"],
                        "claim_text": "Melanie and her kids finished another painting like the last one.",
                    },
                    "t_sunflower": {
                        "refs": ["D14:30"],
                        "claim_text": "Melanie painted a sunflower recently.",
                    },
                },
                "candidate": ["t_sunflower"],
                "hits": ["t_sunflower"],
                "expanded": ["t_sunflower"],
                "grounded_refs": ["D14:30"],
                "answer_text": "Melanie painted a sunflower recently.",
                "retrieval_metadata": {"source_refs": ["D14:30"]},
                "query_metadata_extra": {
                    "evidence_only_conversation": "",
                    "gold_evidence_raw": ["D8:6; D9:17"],
                },
            }
        ],
    )

    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=1)

    row = report["failed_rows"][0]
    assert row["query_task_id"] == "conv-26_qa_37"
    assert row["reason"] == "coarse_retrieval_miss"
    assert row["gold_refs"] == ["D8:6", "D9:17"]
    assert row["gold_trajectory_count"] == 2
    assert row["gold_trajectory_count_in_top_k"] == 0
    assert report["reason_counts"]["unknown_no_gold_refs"] == 0


def test_analyze_locomo_run_failures_only_uses_unknown_when_no_gold_ref_source_exists(tmp_path: Path):
    samples = [_new_style_samples()[0]]
    samples[0]["gold_refs"] = []
    samples[0]["query_metadata_extra"] = {
        "evidence_only_conversation": "",
        "gold_evidence_refs": [],
        "gold_evidence_raw": [],
    }
    run_dir = _build_run_from_specs(tmp_path / "run", samples)

    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=1)

    assert report["reason_counts"]["unknown_no_gold_refs"] == 1
    assert report["failed_rows"][0]["reason"] == "unknown_no_gold_refs"


def test_analyze_locomo_run_failures_reports_force_recall_gold_refs(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        [
            {
                "sample_id": "sample-forced",
                "question": "What area was hit by a flood?",
                "gold_answer": "West County",
                "gold_refs": ["D60:1"],
                "raw_messages": [
                    {
                        "ref": "D60:1",
                        "content": "My old area, West County, was hit by a nasty flood last week.",
                        "speaker_name": "John",
                    }
                ],
                "trajectories": {
                    "t_forced": {
                        "refs": ["D60:1"],
                        "claim_text": "John's old area, West County, was hit by a flood.",
                        "snapshot_metadata": {
                            "forced_episodic_seed_used_v1": True,
                            "llm_no_memory_overridden_v1": True,
                            "forced_episodic_seed_reason_v1": "incident_or_infrastructure_fact",
                        },
                    }
                },
                "candidate": ["t_forced"],
                "hits": ["t_forced"],
                "expanded": ["t_forced"],
                "grounded_refs": [],
                "retrieval_metadata": {"source_refs": []},
            }
        ],
    )

    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=2)

    diagnostics = report["memory_force_recall_diagnostics"]
    assert diagnostics["forced_memory_seed_count"] == 1
    assert diagnostics["llm_no_memory_forced_count"] == 1
    assert diagnostics["failed_queries_with_gold_refs_in_forced_memory_count"] == 1
    row = report["failed_rows"][0]
    assert row["gold_ref_forced_memory_hit"] is True
    assert row["gold_refs_in_forced_memory"] == ["D60:1"]
    assert row["reason"] != "memory_absent"


def test_analyze_locomo_run_failures_reports_zero_claim_memory_diagnostics(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        [
            {
                "sample_id": "sample-zero-claim",
                "question": "What did Andrew do?",
                "gold_answer": "hiked",
                "gold_refs": ["D0:1"],
                "raw_messages": [
                    {"ref": "D0:1", "content": "Andrew hiked today.", "speaker_name": "Andrew"},
                ],
                "trajectories": {
                    "t_main": {
                        "refs": ["D0:1"],
                        "claim_text": "Andrew hiked today.",
                        "extra_snapshots": [
                            {
                                "version": 2,
                                "refs": ["D0:1"],
                                "label": "empty update",
                                "snapshot_metadata": {"zero_claim_episodic_memory_v1": True},
                            }
                        ],
                    }
                },
                "candidate": ["t_main"],
                "hits": ["t_main"],
                "expanded": ["t_main"],
                "grounded_refs": ["D0:1"],
            }
        ],
        summary_payload={
            "memory": {
                "zero_claim_episodic_candidate_count": 3,
                "zero_claim_episodic_persisted_count": 1,
                "zero_claim_low_salience_skipped_count": 2,
            }
        },
    )

    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=1)
    diagnostics = report["memory_force_recall_diagnostics"]

    assert diagnostics["zero_claim_episodic_candidate_count"] == 3
    assert diagnostics["zero_claim_episodic_persisted_count"] == 1
    assert diagnostics["zero_claim_low_salience_skipped_count"] == 2

    buffer = io.StringIO()
    print_locomo_failure_report(report, console=Console(file=buffer, force_terminal=False))
    output = buffer.getvalue()
    assert "zero_claim_episodic_candidate_count" in output
    assert "zero_claim_low_salience_skipped_count" in output


def test_analyze_locomo_run_failures_gracefully_handles_old_style_metadata(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _old_style_samples())

    report = analyze_locomo_run_failures(run_dir)

    row = report["failed_rows"][0]
    assert row["gold_trajectory_rank"] is None
    assert row["gold_trajectory_count_in_top_k"] == 1
    assert row["top_k_can_cover_all_gold_refs"] is True
    assert row["query_facets"]["tags"] == ["relationship_status"]
    assert "exact_facet_missing" in row["diagnostic_flags"]


def test_analyze_locomo_run_failures_reports_multi_hop_top_k_coverage(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _multi_hop_coverage_samples())

    report = analyze_locomo_run_failures(run_dir)

    single_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-single-covered")
    assert single_row["gold_trajectory_count_in_top_k"] == 1
    assert single_row["gold_trajectory_recall_at_k"] == pytest.approx(1.0)
    assert single_row["top_k_can_cover_all_gold_refs"] is True
    assert single_row["trajectory_selection_pool_available"] is False
    assert single_row["trajectory_selection_pool_size"] is None
    assert single_row["gold_trajectory_count_in_selection_pool"] is None
    assert single_row["gold_trajectory_recall_in_selection_pool"] is None

    partial_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-partial-covered")
    assert partial_row["gold_trajectory_count_in_top_k"] == 1
    assert partial_row["trajectory_selection_pool_available"] is True
    assert partial_row["gold_trajectory_count_in_selection_pool"] == 2
    assert partial_row["gold_trajectory_recall_at_k"] == pytest.approx(0.5)
    assert partial_row["gold_trajectory_recall_in_selection_pool"] == pytest.approx(1.0)
    assert partial_row["selection_pool_can_cover_all_gold_trajectories"] is True
    assert partial_row["trajectory_selection_strategy"] == "coverage_aware_greedy"
    assert partial_row["top_k_can_cover_all_gold_refs"] is False
    assert partial_row["top_k_covering_trajectory_count"] is None

    full_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-full-covered")
    assert full_row["gold_trajectory_count_in_top_k"] == 2
    assert full_row["gold_trajectory_recall_at_k"] == pytest.approx(1.0)
    assert full_row["top_k_can_cover_all_gold_refs"] is True
    assert full_row["top_k_covering_trajectory_count"] == 2

    diagnostics = report["multi_hop_top_k_diagnostics"]
    assert diagnostics["split_case_count"] == 2
    assert diagnostics["selection_pool_available_failed_gold_query_count"] == 2
    assert diagnostics["selection_pool_available_split_case_count"] == 2
    assert diagnostics["selection_pool_can_cover_all_gold_trajectories_rate_over_split_cases"] == pytest.approx(1.0)
    assert diagnostics["gold_trajectory_lost_before_selection_pool_count"] == 0
    assert diagnostics["gold_trajectory_lost_during_final_top_k_count"] == 1
    assert diagnostics["all_gold_trajectories_in_top_k_rate_over_split_cases"] == pytest.approx(0.5)
    assert diagnostics["top_k_can_cover_all_gold_refs_rate_over_split_cases"] == pytest.approx(0.5)
    assert diagnostics["zero_gold_trajectory_in_top_k_rate_over_split_cases"] == pytest.approx(0.0)
    assert diagnostics["one_gold_trajectory_in_top_k_rate_over_split_cases"] == pytest.approx(0.5)
    assert diagnostics["two_or_more_gold_trajectories_in_top_k_rate_over_split_cases"] == pytest.approx(0.5)


def test_analyze_locomo_run_failures_recomputes_stale_query_shape_metadata(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        [
            {
                "sample_id": "sample-stale-shape",
                "question": "How many times has Melanie gone to the beach in 2023?",
                "gold_answer": "2",
                "gold_refs": ["D20:1"],
                "raw_messages": [
                    {
                        "ref": "D20:1",
                        "content": "Melanie went to the beach twice in 2023.",
                        "speaker_name": "Melanie",
                    }
                ],
                "trajectories": {
                    "t_beach": {
                        "refs": ["D20:1"],
                        "claim_text": "Melanie went to the beach twice in 2023.",
                    }
                },
                "candidate": ["t_beach"],
                "hits": ["t_beach"],
                "expanded": ["t_beach"],
                "grounded_refs": [],
                "retrieval_metadata": {
                    "source_refs": [],
                    "query_shape": {
                        "list_like": False,
                        "multi_entity": False,
                        "comparison_like": False,
                        "count_like": False,
                        "item_family": None,
                        "tags": [],
                    },
                },
            }
        ],
    )

    report = analyze_locomo_run_failures(run_dir)
    row = report["failed_rows"][0]

    assert row["query_shape"]["count_like"] is True
    assert row["query_shape"]["item_family"] == "count"
    assert row["query_shape"]["query_shape_source"] == "derived_current_v1"
    assert row["query_shape"]["query_shape_metadata_mismatch"] is True


def test_analyze_locomo_run_failures_reports_preservation_and_stage_diagnostics(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _preservation_and_stage_samples())

    report = analyze_locomo_run_failures(run_dir)

    storage_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-trajectory-storage")
    assert storage_row["gold_answer_fully_present_in_claims"] is False
    assert storage_row["trajectory_storage_miss"] is True

    summary_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-summary-compression")
    assert summary_row["gold_answer_fully_present_in_claims"] is True
    assert summary_row["gold_answer_fully_present_in_trajectory_summaries"] is False
    assert summary_row["summary_compression_miss"] is True
    assert summary_row["gold_item_present_in_historical_card"] is True
    assert summary_row["gold_item_present_in_latest_summary"] is False
    assert summary_row["gold_item_present_in_wiki_after_historical_card"] is True

    wiki_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-wiki-compression")
    assert wiki_row["gold_answer_fully_present_in_trajectory_summaries"] is True
    assert wiki_row["gold_answer_fully_present_in_wiki_pages"] is False
    assert wiki_row["wiki_compilation_compression_miss"] is True

    page_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-page-routing-fail")
    assert page_row["page_routing_failed"] is True
    assert page_row["gold_pages_in_top_t"] == []

    selected_pages_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-selected-pages-fail")
    assert selected_pages_row["selected_pages_failed_to_cover_all_gold_trajectories"] is True
    assert selected_pages_row["all_gold_trajectories_in_selected_pages"] is False

    trajectory_topk_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-trajectory-topk-fail")
    assert trajectory_topk_row["trajectory_retrieval_failed_after_page_routing"] is True
    assert trajectory_topk_row["all_gold_trajectories_in_selected_pages"] is True
    assert trajectory_topk_row["all_gold_trajectories_in_final_top_k_after_page_routing"] is False

    strict_row = next(row for row in report["failed_rows"] if row["sample_id"] == "sample-strict-answer-error")
    assert strict_row["all_gold_refs_grounded"] is True
    assert strict_row["strict_retrieval_success_but_answer_wrong"] is True

    preservation = report["memory_preservation_diagnostics"]
    assert preservation["trajectory_storage_miss_rate_over_failed"] == pytest.approx(1 / 7)
    assert preservation["summary_compression_miss_rate_over_failed"] == pytest.approx(1 / 7)
    assert preservation["wiki_compilation_compression_miss_rate_over_failed"] == pytest.approx(1 / 7)
    assert "gold_item_present_in_historical_card_rate_over_failed" in preservation
    assert "trajectory_drift_suspected_rate_over_failed" in preservation

    stages = report["stage_diagnostics"]
    assert stages["page_routing_failed_rate_over_failed"] == pytest.approx(1 / 7)
    assert stages["selected_pages_failed_to_cover_all_gold_trajectories_rate_over_failed"] == pytest.approx(2 / 7)
    assert stages["trajectory_retrieval_failed_after_page_routing_rate_over_failed"] == pytest.approx(3 / 7)
    assert stages["strict_retrieval_success_but_answer_wrong_rate_over_failed"] == pytest.approx(3 / 7)


def test_analyze_locomo_run_failures_reports_compaction_diagnostics(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _compaction_failure_samples())

    report = analyze_locomo_run_failures(run_dir)

    snapshot_row = next(
        row for row in report["failed_rows"] if row["sample_id"] == "sample-snapshot-compaction-miss"
    )
    assert snapshot_row["pre_compaction_expansion_could_cover_all_gold_refs"] is True
    assert snapshot_row["snapshot_compaction_miss"] is True
    assert snapshot_row["gold_refs_lost_by_snapshot_compaction"] == ["D30:2"]
    assert snapshot_row["strict_retrieval_success_but_answer_wrong"] is False

    source_row = next(
        row for row in report["failed_rows"] if row["sample_id"] == "sample-source-compaction-miss"
    )
    assert source_row["pre_compaction_expansion_could_cover_all_gold_refs"] is True
    assert source_row["snapshot_compaction_miss"] is False
    assert source_row["source_compaction_miss"] is True
    assert source_row["gold_refs_lost_by_source_compaction"] == ["D31:1"]
    assert source_row["strict_retrieval_success_but_answer_wrong"] is False

    diagnostics = report["compaction_diagnostics"]
    assert diagnostics["pre_compaction_expansion_cover_rate_over_failed"] == pytest.approx(1.0)
    assert diagnostics["snapshot_compaction_miss_rate_over_failed"] == pytest.approx(0.5)
    assert diagnostics["source_compaction_miss_rate_over_failed"] == pytest.approx(0.5)


def test_analyze_locomo_run_failures_reports_page_and_trajectory_cutoff_diagnostics(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _cutoff_diagnostic_samples())

    report = analyze_locomo_run_failures(run_dir)

    row = report["query_outcomes"][0]
    assert row["min_t_pages_to_cover_all_gold_pages"] == 6
    assert row["min_t_pages_to_cover_all_gold_trajectories_via_pages"] == 6
    assert row["min_k_to_cover_all_gold_trajectories"] == 5
    assert row["min_k_to_cover_all_gold_refs"] == 5
    assert row["page_cutoff_diagnostics"]["5"]["all_gold_pages_in_cutoff"] is False
    assert row["page_cutoff_diagnostics"]["6"]["all_gold_pages_in_cutoff"] is True
    assert row["trajectory_cutoff_diagnostics"]["4"]["all_gold_trajectories_in_cutoff"] is False
    assert row["trajectory_cutoff_diagnostics"]["5"]["all_gold_trajectories_in_cutoff"] is True
    assert row["trajectory_cutoff_diagnostics"]["4"]["top_k_can_cover_all_gold_refs"] is False
    assert row["trajectory_cutoff_diagnostics"]["5"]["top_k_can_cover_all_gold_refs"] is True

    failed_row = report["failed_rows"][0]
    assert failed_row["min_t_pages_to_cover_all_gold_pages"] == 6
    assert failed_row["min_k_to_cover_all_gold_refs"] == 5

    page_cutoffs = report["cutoff_diagnostics"]["page_cutoffs"]
    assert page_cutoffs["5"]["all_queries"]["all_gold_pages_rate"] == pytest.approx(0.0)
    assert page_cutoffs["6"]["all_queries"]["all_gold_pages_rate"] == pytest.approx(1.0)
    assert page_cutoffs["5"]["all_queries"]["selected_pages_cover_all_gold_trajectories_rate"] == pytest.approx(0.0)
    assert page_cutoffs["6"]["all_queries"]["selected_pages_cover_all_gold_trajectories_rate"] == pytest.approx(1.0)

    trajectory_cutoffs = report["cutoff_diagnostics"]["trajectory_cutoffs"]
    assert trajectory_cutoffs["4"]["all_queries"]["all_gold_trajectories_rate"] == pytest.approx(0.0)
    assert trajectory_cutoffs["4"]["all_queries"]["mean_gold_trajectory_recall"] == pytest.approx(0.5)
    assert trajectory_cutoffs["5"]["all_queries"]["all_gold_trajectories_rate"] == pytest.approx(1.0)
    assert trajectory_cutoffs["4"]["all_queries"]["top_k_can_cover_all_gold_refs_rate"] == pytest.approx(0.0)
    assert trajectory_cutoffs["5"]["all_queries"]["top_k_can_cover_all_gold_refs_rate"] == pytest.approx(1.0)
    assert report["cutoff_diagnostics"]["suggested_cutoffs"]["page_95"] == 6
    assert report["cutoff_diagnostics"]["suggested_cutoffs"]["trajectory_95"] == 5
    assert report["offline_parameter_diagnostics"]["diagnostic_mode"] == "legacy_selected_only"
    assert row["offline_page_diagnostic_mode"] == "legacy_selected_only"
    assert row["offline_trajectory_diagnostic_mode"] == "legacy_selected_only"
    assert "5" in row["offline_page_cutoff_diagnostics"]


def test_analyze_locomo_run_failures_reports_direct_trajectory_ablation_page_bottleneck(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        [
            {
                "sample_id": "sample-direct-page",
                "question": "What did Caroline research?",
                "gold_answer": "adoption agencies",
                "gold_refs": ["D1:1"],
                "raw_messages": [
                    {"ref": "D1:1", "content": "Caroline researched adoption agencies.", "speaker_name": "Caroline"},
                    {"ref": "D1:2", "content": "Caroline went camping.", "speaker_name": "Caroline"},
                ],
                "trajectories": {
                    "t_gold": {
                        "refs": ["D1:1"],
                        "claim_text": "Caroline researched adoption agencies.",
                        "trajectory_metadata": {
                            "retrieval_summary_text": "Caroline researched adoption agencies.",
                            "trajectory_historical_item_terms_v1": ["adoption agencies", "research"],
                            "entity_mentions": ["Caroline"],
                        },
                    },
                    "t_distract": {
                        "refs": ["D1:2"],
                        "claim_text": "Caroline went camping.",
                        "trajectory_metadata": {
                            "retrieval_summary_text": "Caroline went camping.",
                            "entity_mentions": ["Caroline"],
                        },
                    },
                },
                "page_candidate": ["t_distract"],
                "candidate": ["t_distract"],
                "hits": ["t_distract"],
                "expanded": ["t_distract"],
                "grounded_refs": [],
                "top_k": 1,
            }
        ],
    )

    report = analyze_locomo_run_failures(run_dir)

    row = report["failed_rows"][0]
    assert row["gold_trajectory_recall_at_k"] == pytest.approx(0.0)
    assert row["direct_gold_trajectory_recall_at_k"] == pytest.approx(1.0)
    assert row["direct_page_routing_bottleneck_suspected"] is True
    assert row["direct_trajectory_bottleneck"] == "page_routing"
    assert report["direct_trajectory_ablation_diagnostics"]["page_routing_bottleneck_suspected_count"] == 1
    assert "item_family:research_topic" in report["direct_trajectory_ablation_diagnostics"]["by_query_shape"]
    assert report["direct_vs_routed_retrieval_diagnostics"]["diagnostic_mode"] == "metadata_direct_retrieval_ablation"
    assert (run_dir / "analysis" / "direct_retrieval_ablation.json").exists()
    assert (run_dir / "analysis" / "direct_retrieval_rows.jsonl").exists()


def test_analyze_locomo_run_failures_direct_ablation_distinguishes_ranking_and_signal_gap(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        [
            {
                "sample_id": "sample-direct-ranking",
                "question": "What did Caroline research?",
                "gold_answer": "adoption agencies",
                "gold_refs": ["D2:1"],
                "raw_messages": [
                    {"ref": "D2:1", "content": "Caroline researched adoption agencies.", "speaker_name": "Caroline"},
                    {"ref": "D2:2", "content": "Caroline went camping.", "speaker_name": "Caroline"},
                ],
                "trajectories": {
                    "t_gold": {
                        "refs": ["D2:1"],
                        "claim_text": "Caroline researched adoption agencies.",
                        "trajectory_metadata": {
                            "retrieval_summary_text": "Caroline researched adoption agencies.",
                            "trajectory_historical_item_terms_v1": ["adoption agencies", "research"],
                            "entity_mentions": ["Caroline"],
                        },
                    },
                    "t_distract": {
                        "refs": ["D2:2"],
                        "claim_text": "Caroline went camping.",
                        "trajectory_metadata": {"retrieval_summary_text": "Caroline went camping."},
                    },
                },
                "page_candidate": ["t_distract", "t_gold"],
                "candidate": ["t_distract"],
                "hits": ["t_distract"],
                "expanded": ["t_distract"],
                "grounded_refs": [],
                "top_k": 1,
            },
            {
                "sample_id": "sample-direct-gap",
                "question": "What did Melanie paint recently?",
                "gold_answer": "sunset",
                "gold_refs": ["D3:1"],
                "raw_messages": [
                    {"ref": "D3:1", "content": "Melanie made something yesterday.", "speaker_name": "Melanie"},
                    {"ref": "D3:2", "content": "Melanie painted a sunflower recently.", "speaker_name": "Melanie"},
                ],
                "trajectories": {
                    "t_gold": {
                        "refs": ["D3:1"],
                        "claim_text": "Melanie made something yesterday.",
                        "trajectory_metadata": {"retrieval_summary_text": "Melanie made something yesterday."},
                    },
                    "t_distract": {
                        "refs": ["D3:2"],
                        "claim_text": "Melanie painted a sunflower recently.",
                        "trajectory_metadata": {
                            "retrieval_summary_text": "Melanie painted a sunflower recently.",
                            "trajectory_historical_item_terms_v1": ["painted", "sunflower"],
                        },
                    },
                },
                "page_candidate": ["t_distract", "t_gold"],
                "candidate": ["t_distract"],
                "hits": ["t_distract"],
                "expanded": ["t_distract"],
                "grounded_refs": [],
                "top_k": 1,
            },
        ],
    )

    report = analyze_locomo_run_failures(run_dir)
    rows = {row["sample_id"]: row for row in report["failed_rows"]}

    ranking_row = rows["sample-direct-ranking"]
    assert ranking_row["direct_gold_trajectory_recall_at_k"] == pytest.approx(1.0)
    assert ranking_row["direct_page_routing_bottleneck_suspected"] is False
    assert ranking_row["direct_trajectory_bottleneck"] == "trajectory_ranking"

    gap_row = rows["sample-direct-gap"]
    assert gap_row["direct_gold_trajectory_recall_at_k"] == pytest.approx(0.0)
    assert gap_row["direct_page_routing_bottleneck_suspected"] is False
    assert gap_row["direct_trajectory_bottleneck"] == "metadata_signal_gap"


def test_analyze_locomo_run_failures_reports_trajectory_length_diagnostics(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        _trajectory_length_overprovisioned_samples(),
        m=20,
    )

    report = analyze_locomo_run_failures(run_dir)

    row = report["query_outcomes"][0]
    assert row["gold_trajectory_length_max"] == 5
    assert row["gold_trajectory_lengths"]
    assert row["gold_trajectory_at_m_limit_count"] == 0
    assert row["max_gold_snapshot_rank_required"] == 5
    assert row["max_gold_snapshot_version_required"] == 5
    assert set(row["gold_snapshot_ranks"].values()) == {5}
    assert set(row["gold_snapshot_versions"].values()) == {5}

    diagnostics = report["trajectory_length_diagnostics"]
    assert diagnostics["configured_m"] == 20
    assert diagnostics["all_trajectories"]["p95"] == pytest.approx(5.0)
    assert diagnostics["all_trajectories"]["at_m_limit_rate"] == pytest.approx(0.0)
    assert diagnostics["all_queries_gold_trajectories"]["gold_trajectory_length_stats"]["p95"] == pytest.approx(5.0)
    assert diagnostics["all_queries_gold_snapshot_requirements"]["max_gold_snapshot_rank_required_stats"]["p95"] == pytest.approx(5.0)
    assert diagnostics["all_queries_gold_snapshot_requirements"]["max_gold_snapshot_version_required_stats"]["p95"] == pytest.approx(5.0)
    assert diagnostics["recommendations"]["m_overprovisioned_possible"] is True
    assert diagnostics["recommendations"]["m_pressure_detected"] is False
    assert diagnostics["recommendations"]["candidate_lower_m_at_95"] == 5
    assert diagnostics["recommendations"]["interpretation"] == "m_overprovisioned_possible"

    failed_row = report["failed_rows"][0]
    assert failed_row["gold_trajectory_length_max"] == 5
    assert failed_row["max_gold_snapshot_rank_required"] == 5


def test_analyze_locomo_run_failures_detects_trajectory_length_pressure(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run",
        _trajectory_length_pressure_samples(),
        m=3,
    )

    report = analyze_locomo_run_failures(run_dir)

    row = report["query_outcomes"][0]
    assert row["gold_trajectory_length_max"] == 3
    assert row["gold_trajectory_at_m_limit_count"] == 1
    assert row["gold_trajectory_at_m_limit_rate"] == pytest.approx(1.0)
    assert row["max_gold_snapshot_rank_required"] == 3

    diagnostics = report["trajectory_length_diagnostics"]
    assert diagnostics["all_trajectories"]["at_m_limit_rate"] == pytest.approx(1.0)
    assert diagnostics["all_queries_gold_trajectories"]["gold_trajectory_at_m_limit_rate"] == pytest.approx(1.0)
    assert diagnostics["recommendations"]["m_pressure_detected"] is True
    assert diagnostics["recommendations"]["m_overprovisioned_possible"] is False
    assert diagnostics["recommendations"]["interpretation"] == "m_pressure"


def test_analyze_locomo_run_failures_reports_trajectory_drift_diagnostics(tmp_path: Path):
    run_dir = _build_run_from_specs(
        tmp_path / "run-drift",
        [
            {
                "sample_id": "sample-drift",
                "query_task_id": "sample-drift_qa_1",
                "question": "What changed?",
                "gold_answer": "the update",
                "gold_refs": ["D0:1", "D0:2"],
                "raw_messages": [
                    {"ref": "D0:1", "content": "First memory.", "speaker_name": "A"},
                    {"ref": "D0:2", "content": "Later update.", "speaker_name": "A"},
                ],
                "trajectories": {
                    "t_gold": {
                        "refs": ["D0:1"],
                        "claim_text": "First memory.",
                        "extra_snapshots": [
                            {
                                "version": 2,
                                "refs": ["D0:2"],
                                "claim_text": "Later update.",
                            }
                        ],
                    }
                },
                "candidate": ["t_gold"],
                "hits": ["t_gold"],
                "expanded": ["t_gold"],
                "grounded_refs": ["D0:1", "D0:2"],
                "retrieval_metadata": {"source_refs": ["D0:1", "D0:2"]},
                "judge_verdict": "incorrect",
            }
        ],
    )
    session_factory = create_schema(run_dir / "trajpatch.sqlite")
    session = session_factory()
    trajectory_id = "epi-sample-drift-001"
    for owner_id, owner_type, vector in [
        (f"{trajectory_id}-v001", "snapshot", [1.0, 0.0]),
        (f"{trajectory_id}-v002", "snapshot", [0.0, 1.0]),
        (trajectory_id, "trajectory_summary", [0.0, 1.0]),
    ]:
        session.add(
            EmbeddingRecord(
                id=f"{owner_id}-{owner_type}-embedding",
                owner_type=owner_type,
                owner_id=owner_id,
                model_name="unit-embedding",
                vector_json=vector,
                semantic_text="semantic text",
                norm=sum(value * value for value in vector) ** 0.5,
                metadata_json={},
            )
        )
    session.commit()
    session.close()

    report = analyze_locomo_run_failures(run_dir)

    diagnostics = report["trajectory_drift_diagnostics"]
    assert diagnostics["trajectory_count"] == 1
    assert diagnostics["drift_bucket_counts"]["possible_drift"] == 1
    row = report["query_outcomes"][0]
    assert row["gold_trajectory_head_tail_cosine_min"] == pytest.approx(0.0)
    assert row["trajectory_drift_risk_observed"] is True
    failed_row = report["failed_rows"][0]
    assert failed_row["gold_trajectory_possible_drift_count"] == 1
    assert "trajectory_drift_artifacts" in report
    drift_rows_path = Path(report["trajectory_drift_artifacts"]["rows"])
    assert drift_rows_path.exists()
    drift_rows_text = drift_rows_path.read_text(encoding="utf-8")
    assert "vector_json" not in drift_rows_text
    assert "semantic text" not in drift_rows_text


def test_analyze_locomo_run_failures_handles_all_correct_rows_and_cuda_preflight(tmp_path: Path) -> None:
    run_dir = _build_run_from_specs(
        tmp_path / "run-correct",
        [
            {
                "sample_id": "sample-correct",
                "query_task_id": "sample-correct_qa_1",
                "question": "What did Alice adopt?",
                "gold_answer": "a dog",
                "answer_text": "a dog",
                "judge_verdict": "correct",
                "gold_refs": ["D1:1"],
                "raw_messages": [
                    {"ref": "D1:1", "content": "Alice adopted a dog.", "speaker_name": "Alice"},
                ],
                "trajectories": {
                    "t_good": {
                        "refs": ["D1:1"],
                        "claim_text": "Alice adopted a dog.",
                    }
                },
                "candidate": ["t_good"],
                "hits": ["t_good"],
                "expanded": ["t_good"],
                "grounded_refs": ["D1:1"],
            }
        ],
        summary_payload={
            "cuda_preflight": {
                "mode": "warn",
                "enabled": True,
                "risk": "low",
                "warnings": [],
                "errors": [],
                "assignments": [
                    {
                        "role": "embedding_main",
                        "device": "cuda:0",
                    }
                ],
            }
        },
    )

    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=1)

    assert report["totals"]["failed_queries"] == 0
    assert len(report["query_outcomes"]) == 1
    assert report["cuda_preflight_diagnostics"]["risk"] == "low"

    buffer = io.StringIO()
    print_locomo_failure_report(report, console=Console(file=buffer, force_terminal=False))
    rendered = buffer.getvalue()
    assert "CUDA Preflight" in rendered
    assert "low" in rendered


def test_analyze_locomo_run_failures_resolves_latest_run_from_scope_dir(tmp_path: Path):
    scope_dir = tmp_path / "scope"
    older = _build_run_from_specs(scope_dir / "20260418_010000_old", _new_style_samples())
    newer = _build_run_from_specs(scope_dir / "20260418_020000_new", _new_style_samples())

    report = analyze_locomo_run_failures(scope_dir)

    assert report["run_meta"]["resolved_run_dir"] == str(newer)
    assert report["run_meta"]["run_id"] == newer.name
    assert older != newer


def test_analyze_locomo_run_failures_requires_complete_run_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Could not locate a completed run"):
        analyze_locomo_run_failures(tmp_path / "missing")


def test_analyze_locomo_run_failures_rejects_non_locomo_runs(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "medmt-run", _new_style_samples(), dataset="medmt")
    with pytest.raises(ValueError, match="supports LOCOMO runs only"):
        analyze_locomo_run_failures(run_dir)


def test_analyze_locomo_run_failures_includes_judge_diagnostics_and_excludes_judge_error_from_failed_rows(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _judge_diagnostic_samples())

    report = analyze_locomo_run_failures(run_dir)

    diagnostics = report["judge_diagnostics"]
    assert diagnostics["judge_evaluable_count"] == 5
    assert diagnostics["judge_execution_failed_count"] == 1
    assert diagnostics["structured_requested_rate_over_all"] == pytest.approx(0.5)
    assert diagnostics["structured_success_rate_over_all"] == pytest.approx(1 / 6)
    assert diagnostics["text_fallback_rate_over_all"] == pytest.approx(1 / 3)
    assert diagnostics["incorrect_queries_judged_via_text_fallback_count"] == 1
    assert diagnostics["structured_fallback_category_counts"]["structured_schema_error"] == 1
    assert diagnostics["structured_fallback_category_counts"]["text_parser_error"] == 1

    failed_query_ids = {row["query_task_id"] for row in report["failed_rows"]}
    assert "sample-answer_qa_0" not in failed_query_ids
    query_outcomes = {row["query_task_id"]: row for row in report["query_outcomes"]}
    assert query_outcomes["sample-answer_qa_0"]["reason"] == "judge_error"
    assert query_outcomes["sample-answer_qa_0"]["judge_execution_failed"] is True
    assert query_outcomes["sample-memory_qa_0"]["judge_mode"] == "text_fallback"


def test_analyze_locomo_run_failures_includes_answer_synthesis_diagnostics(tmp_path: Path):
    samples = _judge_diagnostic_samples()
    samples[0]["answer_metadata"] = {
        "answer_synthesis_used": True,
        "answer_synthesis_mode": "structured",
        "answer_synthesis_can_answer": True,
        "answer_synthesis_answer_type": "count",
        "answer_synthesis_supporting_refs": ["D1:1"],
        "answer_synthesis_counted_events": [
            {"event_id": "E1", "event_text": "first event", "source_refs": ["D1:1"], "reason": "completed"}
        ],
        "answer_synthesis_excluded_events": [
            {"event_id": "X1", "event_text": "future plan", "source_refs": ["D1:2"], "reason": "future plan"}
        ],
        "answer_synthesis_expected_type_text_valid": False,
        "answer_synthesis_expected_type_text_rejection_reason": "date_question_count_style_answer",
        "answer_freeform_used": True,
        "answer_type_verification_used": True,
        "answer_type_verification_success": True,
        "answer_type_match": False,
        "count_validation_source_derived_candidate_events": [
            {"event_id": "source-derived:D1:3:abc", "source_refs": ["D1:3"]}
        ],
        "count_validation_source_derived_candidate_count": 1,
        "count_validation_source_derived_candidate_refs": ["D1:3"],
        "count_validation_source_derived_trigger_terms": ["turtle", "walk"],
        "count_validation_llm_decisions": [
            {"event_id": "source-derived:D1:3:abc", "decision": "COUNT", "source_refs": ["D1:3"]}
        ],
        "answer_repair_arbitration_triggered": True,
        "answer_repair_arbitration_used": True,
        "answer_repair_arbitration_success": True,
        "answer_repair_arbitration_decision": "keep_initial",
        "answer_repair_arbitration_violation": "lost_required_value",
        "answer_repair_arbitration_confidence": "high",
        "answer_repair_arbitration_action": "keep_initial",
        "answer_repair_arbitration_kept_initial": True,
    }
    samples[1]["answer_metadata"] = {
        "answer_synthesis_used": False,
        "answer_synthesis_mode": "legacy_fallback",
        "answer_synthesis_error": "text_json failed",
    }
    run_dir = _build_run_from_specs(tmp_path / "run", samples)

    report = analyze_locomo_run_failures(run_dir)

    diagnostics = report["answer_synthesis_diagnostics"]
    assert diagnostics["used_count"] == 1
    assert diagnostics["structured_count"] == 1
    assert diagnostics["legacy_fallback_count"] == 1
    assert diagnostics["freeform_used_count"] == 1
    assert diagnostics["answer_type_verification_used_count"] == 1
    assert diagnostics["answer_type_verification_success_count"] == 1
    assert diagnostics["answer_type_mismatch_count"] == 1
    assert diagnostics["can_answer_count"] == 1
    assert diagnostics["excluded_event_reason_counts"]["future plan"] == 1
    assert diagnostics["expected_type_text_rejected_count"] == 1
    assert diagnostics["source_derived_count_candidate_count"] == 1
    assert diagnostics["source_derived_count_candidate_accepted_count"] == 1
    assert diagnostics["repair_arbitration_triggered_count"] == 1
    assert diagnostics["repair_arbitration_used_count"] == 1
    assert diagnostics["repair_arbitration_keep_initial_count"] == 1
    first_outcome = report["query_outcomes"][0]
    assert first_outcome["answer_synthesis_used"] is True
    assert first_outcome["answer_synthesis_counted_event_count"] == 1
    assert first_outcome["answer_synthesis_excluded_event_count"] == 1
    assert first_outcome["answer_freeform_used"] is True
    assert first_outcome["answer_type_verification_used"] is True
    assert first_outcome["count_validation_source_derived_candidate_refs"] == ["D1:3"]
    assert first_outcome["answer_repair_arbitration_action"] == "keep_initial"


def test_analyze_locomo_run_failures_marks_judge_leniency_candidates(tmp_path: Path):
    samples = _new_style_samples()
    samples[0].update(
        {
            "question": "What LGBTQ+ events has Caroline participated in?",
            "gold_answer": "Pride parade, school speech, support group",
            "answer_text": "Pride parade, school speech, support group, LGBTQ conference",
            "judge_verdict": "partial",
        }
    )
    samples[0]["answer_metadata"] = {
        "answer_synthesis_used": True,
        "answer_synthesis_mode": "structured",
        "answer_synthesis_can_answer": True,
        "answer_synthesis_answer_type": "list",
    }
    samples[1].update(
        {
            "question": "How many times has Joanna's scripts been rejected?",
            "gold_answer": "Twice",
            "answer_text": "The retrieved evidence confirms one rejection.",
            "judge_verdict": "partial",
        }
    )
    samples[1]["answer_metadata"] = {
        "answer_synthesis_used": True,
        "answer_synthesis_mode": "structured",
        "answer_synthesis_can_answer": True,
        "answer_synthesis_answer_type": "count",
        "answer_count_lower_bound_reasons": ["count_validation_excluded_plausible_candidates"],
        "answer_count_naturalized_lower_bound_reason": "count_validation_excluded_plausible_candidates",
        "answer_count_lower_bound_excluded_event_count": 1,
    }
    run_dir = _build_run_from_specs(tmp_path / "run", samples)

    report = analyze_locomo_run_failures(run_dir)
    outcomes = {row["query_task_id"]: row for row in report["query_outcomes"]}
    list_row = outcomes["sample-memory_qa_0"]
    count_row = outcomes["sample-coarse_qa_0"]

    assert list_row["judge_leniency_candidate"] is True
    assert list_row["gold_all_items_covered_by_answer"] is True
    assert list_row["answer_has_extra_items"] is True
    assert list_row["count_answer_is_evidence_limited_lower_bound"] is False
    assert count_row["judge_leniency_candidate"] is True
    assert count_row["count_answer_is_evidence_limited_lower_bound"] is True
    assert count_row["answer_count_lower_bound_reasons"] == [
        "count_validation_excluded_plausible_candidates"
    ]
    assert count_row["answer_count_lower_bound_excluded_event_count"] == 1


def test_analyze_locomo_run_failures_includes_raw_rescue_attempt_diagnostics(tmp_path: Path):
    samples = _judge_diagnostic_samples()
    samples[0]["answer_metadata"] = {
        "retrieval_reflection_used": True,
        "retrieval_reflection_retry_used": True,
        "retrieval_reflection_stage": "raw",
        "raw_rescue_attempted": True,
        "raw_rescue_used": True,
        "raw_rescue_hit_count": 1,
        "raw_rescue_trigger_reasons": ["semantic_evidence_weak"],
        "reflection_required_terms": ["flood", "West County"],
        "reflection_covered_terms": ["West County"],
        "reflection_uncovered_terms": ["flood"],
        "reflection_term_coverage_rate": 0.5,
        "reflection_semantic_evidence_weak": True,
        "post_reflection_raw_rescue_used": True,
        "post_reflection_raw_rescue_reason": "post_reflection_abstain",
        "reflection_answer_changed": True,
    }
    samples[1]["answer_metadata"] = {
        "retrieval_reflection_used": True,
        "retrieval_reflection_retry_used": True,
        "retrieval_reflection_stage": "wiki",
        "raw_rescue_attempted": False,
        "raw_rescue_skipped_reason": "evidence_sufficient",
        "reflection_answer_changed": False,
    }
    run_dir = _build_run_from_specs(tmp_path / "run", samples)

    report = analyze_locomo_run_failures(run_dir)

    diagnostics = report["retrieval_reflection_diagnostics"]
    assert diagnostics["retry_triggered_count"] == 2
    assert diagnostics["raw_rescue_attempted_count"] == 1
    assert diagnostics["raw_rescue_skipped_count"] == 1
    assert diagnostics["semantic_weak_trigger_count"] == 1
    assert diagnostics["post_reflection_raw_rescue_count"] == 1
    assert diagnostics["raw_rescue_hit_rate_over_attempted"] == pytest.approx(1.0)
    first_outcome = report["query_outcomes"][0]
    assert first_outcome["raw_rescue_attempted"] is True
    assert first_outcome["raw_rescue_trigger_reasons"] == ["semantic_evidence_weak"]
    assert first_outcome["reflection_uncovered_terms"] == ["flood"]
    assert first_outcome["post_reflection_raw_rescue_reason"] == "post_reflection_abstain"


def test_diff_locomo_failure_reports_includes_judge_diagnostics_delta(tmp_path: Path):
    before_run = _build_run_from_specs(tmp_path / "before", _judge_diagnostic_samples())
    after_samples = _judge_diagnostic_samples()
    for sample in after_samples:
        if sample["sample_id"] == "sample-answer":
            sample["judge_verdict"] = "correct"
            sample["judge_metadata"] = {
                "judge_mode": "structured",
                "structured_requested": True,
                "structured_success": True,
                "structured_fallback_used": False,
                "structured_fallback_reason": None,
                "structured_fallback_category": None,
                "judge_execution_failed": False,
            }
    after_run = _build_run_from_specs(tmp_path / "after", after_samples)

    diff_report = diff_locomo_failure_reports(before_run, after_run, top_examples_per_bucket=1)

    judge_delta = diff_report["judge_diagnostics_delta"]
    assert judge_delta["judge_execution_failed_count"] == -1.0
    assert judge_delta["structured_success_rate_over_all"] == pytest.approx(1 / 6)
    assert judge_delta["text_fallback_rate_over_all"] == pytest.approx(-1 / 6)
    assert "structured_schema_error" in judge_delta["structured_fallback_category_counts"]
    assert "text_parser_error" in judge_delta["structured_fallback_category_counts"]


def test_diff_locomo_failure_reports_summarizes_reason_transitions(tmp_path: Path):
    before_run = _build_run_from_specs(tmp_path / "before", _new_style_samples())
    after_samples = _new_style_samples()
    for sample in after_samples:
        if sample["sample_id"] == "sample-coarse":
            sample["judge_verdict"] = "correct"
    after_run = _build_run_from_specs(tmp_path / "after", after_samples)

    diff_report = diff_locomo_failure_reports(before_run, after_run, top_examples_per_bucket=2)

    assert diff_report["totals"]["failed_queries_delta"] == -1
    assert "cutoff_diagnostics_delta" in diff_report
    transitions = {
        (row["before_reason"], row["after_reason"]): row["count"]
        for row in diff_report["reason_transition_matrix"]
    }
    assert transitions[("coarse_retrieval_miss", "correct")] == 1
    assert diff_report["outcome_counts_delta"]["correct"] == 1
    assert diff_report["improved_queries"][0]["sample_id"] == "sample-coarse"


def test_diff_locomo_failure_reports_summarizes_multi_hop_coverage_transitions(tmp_path: Path):
    before_run = _build_run_from_specs(tmp_path / "before", _multi_hop_coverage_samples())
    after_samples = _multi_hop_coverage_samples()
    for sample in after_samples:
        if sample["sample_id"] == "sample-partial-covered":
            sample["candidate"] = ["t_move", "t_home"]
            sample["hits"] = ["t_move", "t_home"]
            sample["expanded"] = ["t_move", "t_home"]
            sample["grounded_refs"] = ["D11:1", "D11:2"]
            sample["retrieval_metadata"] = {"source_refs": ["D11:1", "D11:2"]}
    after_run = _build_run_from_specs(tmp_path / "after", after_samples)

    diff_report = diff_locomo_failure_reports(before_run, after_run, top_examples_per_bucket=2)

    assert diff_report["multi_hop_top_k_diagnostics_delta"]["top_k_can_cover_all_gold_refs_rate_over_split_cases"] == pytest.approx(0.5)
    coverability_transitions = {
        (row["before"], row["after"]): row["count"]
        for row in diff_report["coverage_transition_counts"]["coverability"]
    }
    assert coverability_transitions[("not_coverable_in_top_k", "coverable_in_top_k")] == 1
    assert diff_report["improved_multi_hop_queries"][0]["sample_id"] == "sample-partial-covered"


def test_diff_locomo_failure_reports_includes_preservation_stage_and_strict_answer_deltas(tmp_path: Path):
    before_run = _build_run_from_specs(tmp_path / "before", _preservation_and_stage_samples())
    after_samples = _preservation_and_stage_samples()
    for sample in after_samples:
        if sample["sample_id"] == "sample-summary-compression":
            sample["trajectories"]["t_book"]["trajectory_metadata"]["retrieval_summary_text"] = "Tim read The Hobbit."
        if sample["sample_id"] == "sample-page-routing-fail":
            sample["page_candidate"] = ["t_gold"]
            sample["top_t_pages"] = 1
            sample["candidate"] = ["t_gold"]
            sample["hits"] = ["t_gold"]
            sample["expanded"] = ["t_gold"]
            sample["grounded_refs"] = ["D23:1"]
            sample["retrieval_metadata"] = {"source_refs": ["D23:1"]}
        if sample["sample_id"] == "sample-strict-answer-error":
            sample["judge_verdict"] = "correct"
    after_run = _build_run_from_specs(tmp_path / "after", after_samples)

    diff_report = diff_locomo_failure_reports(before_run, after_run, top_examples_per_bucket=3)

    assert "memory_preservation_diagnostics_delta" in diff_report
    assert "stage_diagnostics_delta" in diff_report
    assert diff_report["memory_preservation_diagnostics_delta"]["summary_compression_miss_rate_over_failed"] is not None
    assert diff_report["stage_diagnostics_delta"]["page_routing_failed_rate_over_failed"] is not None
    strict_transitions = {
        (row["before"], row["after"]): row["count"]
        for row in diff_report["strict_answer_error_transition_counts"]
    }
    assert strict_transitions[("strict_answer_error", "not_strict_answer_error")] == 1


def test_diff_locomo_failure_reports_includes_trajectory_length_delta(tmp_path: Path):
    before_run = _build_run_from_specs(
        tmp_path / "before",
        _trajectory_length_overprovisioned_samples(),
        m=20,
    )
    after_run = _build_run_from_specs(
        tmp_path / "after",
        _trajectory_length_overprovisioned_samples(),
        m=8,
    )

    diff_report = diff_locomo_failure_reports(before_run, after_run, top_examples_per_bucket=1)

    delta = diff_report["trajectory_length_diagnostics_delta"]
    assert delta["all_trajectory_p95"]["before"] == pytest.approx(5.0)
    assert delta["all_trajectory_p95"]["after"] == pytest.approx(5.0)
    assert delta["gold_required_rank_p95"]["before"] == pytest.approx(5.0)
    assert delta["candidate_lower_m_at_95"]["before"] == 5
    assert delta["candidate_lower_m_at_95"]["after"] == 5
    assert delta["interpretation"]["before"] == "m_overprovisioned_possible"
    assert delta["interpretation"]["after"] == "m_looks_reasonable"


def test_print_locomo_failure_report_smoke(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _judge_diagnostic_samples())
    report = analyze_locomo_run_failures(run_dir, top_examples_per_bucket=1)
    for examples in report["examples_by_reason"].values():
        if examples:
            examples[0]["fallback_repair_summary"] = {"event_count": 1}
            examples[0]["fallback_repair_quality_flags"] = {
                "quality_risk_event_count": 1,
                "has_deterministic_fallback": True,
                "has_discarded_repair": False,
            }
            break
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    print_locomo_failure_report(report, console=console, show_ranks=True, show_facets=True)

    output = buffer.getvalue()
    assert "LOCOMO Failure Attribution" in output
    assert "Judge Diagnostics" in output
    assert "Answer Synthesis Diagnostics" in output
    assert "Fallback / Repair Risk Diagnostics" in output
    assert "Fallback/repair:" in output
    assert "Rank Diagnostics Over Failed Queries" in output
    assert "Facet / Fragmentation Diagnostics Over Failed Queries" in output
    assert "Wiki Fragmentation Diagnostics" in output
    assert "Memory Preservation Diagnostics Over Failed Queries" in output
    assert "Routing / Retrieval Stage Diagnostics Over Failed Queries" in output
    assert "Multi-Hop Top-k Coverage Over Failed Queries" in output
    assert "Page Cutoff Diagnostics" in output
    assert "Trajectory Cutoff Diagnostics" in output
    assert "Trajectory Length Diagnostics" in output
    assert "Judge:" in output
    assert "Ranks:" in output
    assert "Retrieval coverage:" in output
    assert "Selection pool:" in output
    assert "Page routing:" in output
    assert "Page granularity:" in output
    assert "Memory preservation:" in output
    assert "Historical evidence:" in output
    assert "Wiki fragmentation:" in output
    assert "Trajectory length:" in output
    assert "Answer synthesis:" in output
    assert "Judge leniency:" in output
    assert "Gold snapshot requirement:" in output
    assert "Gold claim facets:" in output
    assert "Temporal grounding:" not in output
    assert "judge_rationale=None" not in output
    assert "coarse:None" not in output


def test_print_locomo_failure_diff_smoke(tmp_path: Path):
    before_run = _build_run_from_specs(tmp_path / "before", _judge_diagnostic_samples())
    after_samples = _judge_diagnostic_samples()
    for sample in after_samples:
        if sample["sample_id"] == "sample-coarse":
            sample["judge_verdict"] = "correct"
    after_run = _build_run_from_specs(tmp_path / "after", after_samples)
    diff_report = diff_locomo_failure_reports(before_run, after_run, top_examples_per_bucket=1)
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    print_locomo_failure_diff(diff_report, console=console)

    output = buffer.getvalue()
    assert "LOCOMO Failure Attribution Diff" in output
    assert "Reason Transition Matrix" in output
    assert "Memory Preservation Delta" in output
    assert "Routing / Retrieval Stage Delta" in output
    assert "Strict Answer Error Transitions" in output
    assert "Multi-Hop Coverage Delta" in output
    assert "Judge Diagnostics Delta" in output
    assert "Cutoff Diagnostics Delta" in output
    assert "Trajectory Length Delta" in output


@pytest.mark.skipif(runner is None, reason="typer is not installed in the current test environment")
def test_analyze_failures_cli_prints_rich_report(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _judge_diagnostic_samples())

    result = runner.invoke(
        app,
        [
            "analyze-failures",
            "--run-path",
            str(run_dir),
            "--top-examples-per-bucket",
            "1",
            "--show-ranks",
            "--show-facets",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "LOCOMO Failure Attribution" in result.stdout
    assert "Judge Diagnostics" in result.stdout
    assert "Rank Diagnostics Over Failed Queries" in result.stdout
    assert "Judge:" in result.stdout
    assert "Diagnostic flags:" in result.stdout
    assert "judge_rationale=None" not in result.stdout


@pytest.mark.skipif(runner is None, reason="typer is not installed in the current test environment")
def test_analyze_failures_cli_prints_json_report(tmp_path: Path):
    run_dir = _build_run_from_specs(tmp_path / "run", _judge_diagnostic_samples())

    result = runner.invoke(
        app,
        [
            "analyze-failures",
            "--run-path",
            str(run_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["run_meta"]["dataset"] == "locomo"
    assert payload["reason_counts"]["memory_absent"] == 1
    assert "judge_diagnostics" in payload
    assert payload["judge_diagnostics"]["judge_execution_failed_count"] == 1
    assert payload["multi_hop_top_k_diagnostics"]["split_case_count"] == 1
    assert "memory_preservation_diagnostics" in payload
    assert "stage_diagnostics" in payload
    assert "cutoff_diagnostics" in payload
    assert "trajectory_length_diagnostics" in payload
    assert "fallback_repair_diagnostics" in payload
    assert payload["fallback_repair_diagnostics"]["diagnostic_mode"] == "legacy_counters"
    assert "memory_force_recall_diagnostics" in payload
    assert "forced_memory_seed_count" in payload["memory_force_recall_diagnostics"]


@pytest.mark.skipif(runner is None, reason="typer is not installed in the current test environment")
def test_analyze_failures_diff_cli_prints_json_report(tmp_path: Path):
    before_run = _build_run_from_specs(tmp_path / "before", _new_style_samples())
    after_samples = _new_style_samples()
    for sample in after_samples:
        if sample["sample_id"] == "sample-coarse":
            sample["judge_verdict"] = "correct"
    after_run = _build_run_from_specs(tmp_path / "after", after_samples)

    result = runner.invoke(
        app,
        [
            "analyze-failures-diff",
            "--before",
            str(before_run),
            "--after",
            str(after_run),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["totals"]["failed_queries_delta"] == -1
    transitions = {
        (row["before_reason"], row["after_reason"]): row["count"]
        for row in payload["reason_transition_matrix"]
    }
    assert transitions[("coarse_retrieval_miss", "correct")] == 1
    assert "multi_hop_top_k_diagnostics_delta" in payload
    assert "memory_preservation_diagnostics_delta" in payload
    assert "stage_diagnostics_delta" in payload
    assert "cutoff_diagnostics_delta" in payload
    assert "trajectory_length_diagnostics_delta" in payload
    assert "fallback_repair_diagnostics_delta" in payload
