from __future__ import annotations

from trajpatch.memory.retrieval import RetrievalEngine
from trajpatch.memory.wiki import WikiCompiler
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.providers.structured_outputs import get_structured_task_spec, parse_structured_payload
from trajpatch.storage.models import ClaimRecord, EpisodicMemorySnapshot, RawMessageRecord, WikiPageRecord
from trajpatch.types import ModelInfo, NormalizedMessage
from trajpatch.utils.text import extract_keywords


class _CompactionEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(model_name="compaction-embedding")

    def _vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        if "neighbor" in lowered:
            return [0.0, 1.0]
        if "adoption" in lowered or "seed" in lowered or "update" in lowered:
            return [1.0, 0.0]
        return [-1.0, 0.0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="mock", model_name="compaction-embedding", is_remote=False)


class _FailingDocumentEmbeddingProvider(_CompactionEmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")


def test_fuse_dense_sparse_scores_preserves_rrf_tie_breaks() -> None:
    scored = [
        {"item_id": "a", "dense_score": 1.0, "sparse_score": 1.0},
        {"item_id": "b", "dense_score": 1.0, "sparse_score": 1.0},
        {"item_id": "c", "dense_score": 0.0, "sparse_score": 0.0},
    ]

    fused = RetrievalEngine._fuse_dense_sparse_scores(scored, id_key="item_id")

    assert [item["item_id"] for item in fused] == ["b", "a", "c"]
    assert all("fused_score" in item for item in fused)


def test_fill_selected_ids_after_rerank_backfills_in_candidate_order() -> None:
    candidate_pool = [
        {"item_id": "a"},
        {"item_id": "b"},
        {"item_id": "c"},
    ]

    selected = RetrievalEngine._fill_selected_ids_after_rerank(
        candidate_pool,
        ["c"],
        id_key="item_id",
        final_count=2,
    )

    assert selected == ["c", "a"]


def test_source_message_lines_include_time_anchors() -> None:
    message = RawMessageRecord(
        id="conv-26-m0396",
        sample_id="conv-26",
        dataset_name="locomo",
        turn_index=396,
        role="assistant",
        speaker_name="Melanie",
        content="Thanks, Caroline! Yup, we just did it yesterday!",
        source_ref="D18:17",
        occurred_at="6:55 pm on 20 October, 2023",
        metadata_json={},
    )
    no_date = RawMessageRecord(
        id="conv-26-m0397",
        sample_id="conv-26",
        dataset_name="locomo",
        turn_index=397,
        role="user",
        speaker_name="Caroline",
        content="Glad you had fun.",
        source_ref="D18:18",
        occurred_at=None,
        metadata_json={},
    )

    line = RetrievalEngine._source_message_line(message)
    timeline_line = RetrievalEngine._source_timeline_line(message)
    no_date_line = RetrievalEngine._source_message_line(no_date)

    assert "D18:17 | date=6:55 pm on 20 October, 2023 | id=conv-26-m0396" in line
    assert "D18:17 | date=6:55 pm on 20 October, 2023 | id=conv-26-m0396" in timeline_line
    assert "date=" not in no_date_line


def test_page_granularity_prior_rewards_medium_pages_and_penalizes_weak_singletons() -> None:
    medium_adjustment, medium_metadata = RetrievalEngine._page_granularity_adjustment(
        trajectory_count=4,
        page_type="inventory",
        routing_priority="normal",
        broad_entity_profile=False,
        entity_facet_split=False,
        wiki_rescue_reason=None,
        strong_query_match=False,
    )
    singleton_adjustment, singleton_metadata = RetrievalEngine._page_granularity_adjustment(
        trajectory_count=1,
        page_type="inventory",
        routing_priority="high",
        broad_entity_profile=False,
        entity_facet_split=True,
        wiki_rescue_reason="post_plan_index_only_trajectory",
        strong_query_match=False,
    )
    exact_singleton_adjustment, exact_singleton_metadata = RetrievalEngine._page_granularity_adjustment(
        trajectory_count=1,
        page_type="inventory",
        routing_priority="high",
        broad_entity_profile=False,
        entity_facet_split=True,
        wiki_rescue_reason="post_plan_index_only_trajectory",
        strong_query_match=True,
    )
    low_quality_singleton_adjustment, low_quality_singleton_metadata = RetrievalEngine._page_granularity_adjustment(
        trajectory_count=1,
        page_type="inventory",
        routing_priority="high",
        broad_entity_profile=False,
        entity_facet_split=False,
        wiki_rescue_reason=None,
        strong_query_match=False,
        singleton_policy="merge_required_low_quality",
        singleton_quality_score=0.0,
    )

    assert medium_adjustment > 0
    assert medium_metadata["medium_granularity_page"] is True
    assert singleton_adjustment < 0
    assert singleton_metadata["singleton_page_penalty"] < 0
    assert low_quality_singleton_adjustment < singleton_adjustment
    assert low_quality_singleton_metadata["low_quality_singleton_penalty"] < 0
    assert low_quality_singleton_metadata["singleton_policy"] == "merge_required_low_quality"
    assert exact_singleton_adjustment > singleton_adjustment
    assert exact_singleton_metadata["singleton_penalty_cancelled_by_exact_match"] is True


def test_temporal_anchor_lines_resolve_relative_dates() -> None:
    message = RawMessageRecord(
        id="conv-26-m0396",
        sample_id="conv-26",
        dataset_name="locomo",
        turn_index=396,
        role="assistant",
        speaker_name="Melanie",
        content="Yup, we just did it yesterday after the road trip.",
        source_ref="D18:17",
        occurred_at="6:55 pm on 20 October, 2023",
        metadata_json={},
    )
    today_message = RawMessageRecord(
        id="conv-42-m0245",
        sample_id="conv-42",
        dataset_name="locomo",
        turn_index=245,
        role="assistant",
        speaker_name="Nate",
        content="I got this new pup for you today.",
        source_ref="D13:9",
        occurred_at="3:00 pm on 25 May, 2022",
        metadata_json={},
    )

    lines, metadata = RetrievalEngine._temporal_anchor_lines([message, today_message])

    assert any('D18:17 occurred at 20 October 2023; "yesterday" refers to 19 October 2023.' in line for line in lines)
    assert any('D13:9 occurred at 25 May 2022; "today" refers to 25 May 2022.' in line for line in lines)
    assert metadata["temporal_anchor_hint_count"] == 2
    assert metadata["temporal_anchor_source_refs"] == ["D18:17", "D13:9"]
    assert metadata["temporal_anchor_relative_terms"] == ["yesterday", "today"]


def test_coverage_aware_selection_prefers_complementary_trajectory_clusters(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=5, top_k=2)
    candidate_pool = [
        {
            "trajectory_id": "t-alice-1",
            "routing_priority": "normal",
            "coverage_profile": {
                "entity_keys": {"alice"},
                "facet_values": set(),
                "support_terms": {"alice", "book", "hobbit"},
                "exact_terms": ["The Hobbit"],
                "inventory_like": True,
                "has_count_signal": False,
            },
        },
        {
            "trajectory_id": "t-alice-2",
            "routing_priority": "normal",
            "coverage_profile": {
                "entity_keys": {"alice"},
                "facet_values": set(),
                "support_terms": {"alice", "book", "dune"},
                "exact_terms": ["Dune"],
                "inventory_like": True,
                "has_count_signal": False,
            },
        },
        {
            "trajectory_id": "t-bob-1",
            "routing_priority": "normal",
            "coverage_profile": {
                "entity_keys": {"bob"},
                "facet_values": set(),
                "support_terms": {"bob", "book", "foundation"},
                "exact_terms": ["Foundation"],
                "inventory_like": True,
                "has_count_signal": False,
            },
        },
    ]

    selected, metadata = engine._coverage_aware_select_ids(
        candidate_pool=candidate_pool,
        reranked_ids=["t-alice-1", "t-alice-2", "t-bob-1"],
        id_key="trajectory_id",
        final_count=2,
        query_shape={
            "list_like": True,
            "multi_entity": True,
            "comparison_like": True,
            "count_like": False,
            "item_family": "book",
        },
        query_keywords={"alice", "bob", "books"},
        query_entity_keys={"alice", "bob"},
        query_facet_values=set(),
    )

    assert selected == ["t-alice-1", "t-bob-1"]
    assert metadata["selection_strategy"] == "coverage_aware_greedy"


def test_wiki_seed_planning_uses_historical_evidence_card_for_index_only_trajectory(store) -> None:
    compiler = WikiCompiler(store, MockLLMProvider(), _CompactionEmbeddingProvider())
    trajectory = store.create_trajectory(
        sample_id="conv-26",
        dataset_name="locomo",
        label="melanie-art",
        strict_matching=False,
        max_length=20,
        metadata={
            "retrieval_summary_text": "Melanie discussed pottery and an art injury.",
            "latest_semantic_text": "Melanie hurt her hand while making pottery.",
            "entity_mentions": ["Melanie"],
            "trajectory_historical_evidence_card_v1": {
                "trajectory_id": "pending",
                "identity_summary": "Melanie art evidence",
                "recent_update": "Melanie hurt her hand while making pottery.",
                "historical_item_terms": ["sunset", "painting"],
                "facet_values": ["art_style=sunset"],
                "entity_mentions": ["Melanie"],
                "source_anchors": [{"source_ref": "D8:6", "text": "Melanie painted a sunset recently."}],
                "drift_cluster_keys": ["sunset", "pottery"],
                "display_items": ["sunset"],
                "display_counts": [],
                "display_key_facts": ["Melanie painted a sunset recently."],
            },
        },
    )
    card = dict(trajectory.metadata_json["trajectory_historical_evidence_card_v1"])
    card["trajectory_id"] = trajectory.id
    trajectory.metadata_json = {**trajectory.metadata_json, "trajectory_historical_evidence_card_v1": card}

    seeds = compiler._plan_seeds("conv-26", [trajectory])
    non_index = [seed for seed in seeds if seed.page_type != "index"]

    assert non_index
    assert non_index[0].metadata["seed_type"] == "historical_card_backup"
    assert "sunset" in non_index[0].metadata["wiki_historical_item_terms"]
    manifest = compiler._seed_manifest(seeds, {trajectory.id: trajectory})
    assert "trajectory_evidence_cards" in manifest
    assert "sunset" in manifest


def test_coverage_aware_selection_prefers_new_item_terms_for_same_entity(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=5, top_k=2)
    candidate_pool = [
        {
            "trajectory_id": "t-melanie-clarinet",
            "routing_priority": "normal",
            "coverage_profile": {
                "entity_keys": {"melanie"},
                "facet_values": set(),
                "support_terms": {"melanie", "instrument", "clarinet"},
                "item_terms": {"clarinet"},
                "exact_terms": ["clarinet"],
                "inventory_like": True,
                "has_count_signal": False,
            },
        },
        {
            "trajectory_id": "t-melanie-music-generic",
            "routing_priority": "normal",
            "coverage_profile": {
                "entity_keys": {"melanie"},
                "facet_values": set(),
                "support_terms": {"melanie", "instrument", "clarinet", "music"},
                "item_terms": {"clarinet", "music"},
                "exact_terms": ["clarinet"],
                "inventory_like": True,
                "has_count_signal": False,
            },
        },
        {
            "trajectory_id": "t-melanie-violin",
            "routing_priority": "normal",
            "coverage_profile": {
                "entity_keys": {"melanie"},
                "facet_values": set(),
                "support_terms": {"melanie", "instrument", "violin"},
                "item_terms": {"violin"},
                "exact_terms": ["violin"],
                "inventory_like": True,
                "has_count_signal": False,
            },
        },
    ]

    selected, metadata = engine._coverage_aware_select_ids(
        candidate_pool=candidate_pool,
        reranked_ids=["t-melanie-clarinet", "t-melanie-music-generic", "t-melanie-violin"],
        id_key="trajectory_id",
        final_count=2,
        query_shape={
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "instrument",
        },
        query_keywords={"melanie", "instruments"},
        query_entity_keys={"melanie"},
        query_facet_values=set(),
    )

    assert selected == ["t-melanie-clarinet", "t-melanie-violin"]
    assert "violin" in metadata["covered_item_terms"]
    assert metadata["selected_score_components"][0]["anchored"] is True


def test_coverage_aware_selection_keeps_rank_fill_for_simple_queries(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=5, top_k=2)
    candidate_pool = [
        {"trajectory_id": "t1", "coverage_profile": {"entity_keys": {"alice"}, "facet_values": set(), "support_terms": {"alice"}}},
        {"trajectory_id": "t2", "coverage_profile": {"entity_keys": {"alice"}, "facet_values": set(), "support_terms": {"travel"}}},
        {"trajectory_id": "t3", "coverage_profile": {"entity_keys": {"alice"}, "facet_values": set(), "support_terms": {"trip"}}},
    ]

    selected, metadata = engine._coverage_aware_select_ids(
        candidate_pool=candidate_pool,
        reranked_ids=["t2"],
        id_key="trajectory_id",
        final_count=2,
        query_shape={
            "list_like": False,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": None,
        },
        query_keywords={"trip"},
        query_entity_keys={"alice"},
        query_facet_values=set(),
    )

    assert selected == ["t2", "t1"]
    assert metadata["selection_strategy"] == "rank_fill"


def test_snapshot_budget_is_explicitly_two_times_top_k(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=5, top_k=5)

    assert engine.snapshot_budget == 10


def test_trajectory_retrieval_signals_use_stored_summary_and_fallback(store) -> None:
    provider = _CompactionEmbeddingProvider()
    engine = RetrievalEngine(store, provider, top_t_pages=2, top_k=2)
    trajectory = store.create_trajectory(
        sample_id="sample-signals",
        dataset_name="locomo",
        label="adoption",
        strict_matching=False,
        max_length=6,
        metadata={
            "latest_snapshot_id": "sample-signals-t001-v001",
            "retrieval_summary_text": "Caroline researched adoption agencies.",
            "retrieval_summary_keywords": ["caroline", "adoption"],
            "exact_terms": ["Becoming Nicole"],
            "trajectory_historical_item_terms_v1": ["adoption agencies"],
            "facet_tags": ["research_topic"],
            "facet_values": ["research_topic=adoption agencies"],
            "entity_mentions": ["Caroline"],
        },
    )
    fallback_trajectory = store.create_trajectory(
        sample_id="sample-signals",
        dataset_name="locomo",
        label="fallback label",
        strict_matching=False,
        max_length=6,
        metadata={"latest_semantic_text": "Fallback adoption note."},
    )

    signals = engine._trajectory_retrieval_signals(trajectory)
    fallback_signals = engine._trajectory_retrieval_signals(fallback_trajectory)

    assert signals.latest_snapshot_id == "sample-signals-t001-v001"
    assert signals.summary_text == "Caroline researched adoption agencies."
    assert signals.summary_keywords == ["caroline", "adoption"]
    assert signals.exact_terms == ["Becoming Nicole"]
    assert signals.historical_item_terms == ["adoption agencies"]
    assert signals.facet_tags == {"research_topic"}
    assert signals.facet_values == {"research_topic=adoption agencies"}
    assert signals.entity_mentions == ["Caroline"]
    assert signals.entity_keys == {"caroline"}
    assert {"becoming", "nicole", "adoption"} <= signals.lexical_keywords
    assert "agencies" in signals.support_terms
    assert "Fallback adoption note." in fallback_signals.summary_text
    assert "fallback" in fallback_signals.lexical_keywords


def test_trajectory_retrieval_signals_sanitize_legacy_summary_keywords(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=2, top_k=2)
    trajectory = store.create_trajectory(
        sample_id="sample-legacy-keywords",
        dataset_name="locomo",
        label="adoption",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "## Profile / Stable Facts\n- Caroline researched adoption agencies.",
            "retrieval_summary_keywords": ["profile", "facts", "none", "recorded", "adoption"],
            "trajectory_historical_item_terms_v1": ["None recorded", "adoption agencies"],
        },
    )

    signals = engine._trajectory_retrieval_signals(trajectory)

    assert signals.summary_keywords == ["adoption"]
    assert signals.historical_item_terms == ["adoption agencies"]
    assert "profile" not in signals.lexical_keywords
    assert "recorded" not in signals.lexical_keywords


def test_source_event_metadata_boosts_temporal_event_trajectory_ranking(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-source-event-ranking"
    gold = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="generic business update",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Jon discussed online store planning with fashion bloggers.",
            "retrieval_summary_keywords": ["jon", "store", "planning"],
            "entity_mentions": ["Jon"],
            "source_event_records_v1": [
                {
                    "surface": "dance competition",
                    "canonical": "dance competition",
                    "category": "event_object",
                    "action": "hosting",
                    "temporal_expression": "next month",
                    "source_refs": ["D8:13"],
                    "source_message_ids": ["sample-source-event-ranking-m0013"],
                    "rule": "event_object_action_pattern",
                    "confidence": "high",
                }
            ],
            "source_event_object_terms_v1": ["dance competition"],
            "source_event_action_terms_v1": ["hosting"],
            "source_temporal_relation_terms_v1": ["next month"],
            "source_event_canonical_terms_v1": ["dance competition"],
        },
    )
    distractor = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="store support",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Jon discussed store planning and general event support.",
            "retrieval_summary_keywords": ["jon", "store", "event"],
            "entity_mentions": ["Jon"],
        },
    )
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=1)

    selected, metadata = engine._select_trajectories(
        sample_id=sample_id,
        candidate_trajectory_ids=[distractor.id, gold.id],
        query_text="When did Jon host a dance competition?",
        query_embedding=provider.embed_queries(["When did Jon host a dance competition?"])[0],
        query_keywords=extract_keywords("When did Jon host a dance competition?"),
        query_entities=["Jon"],
        query_facet_tags=set(),
        query_facet_values=set(),
        query_shape={
            "list_like": False,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "event",
        },
    )

    assert selected == [gold.id]
    compact_by_id = {
        row["trajectory_id"]: row for row in metadata["trajectory_ranked_rows_compact_top_n"]
    }
    assert compact_by_id[gold.id]["source_event_match_score"] > 0
    assert "dance" in compact_by_id[gold.id]["source_event_matched_terms"]
    assert "D8:13" in compact_by_id[gold.id]["source_event_matched_refs"]
    assert metadata["trajectory_selected_source_event_matches"][0]["trajectory_id"] == gold.id


def test_generic_event_terms_do_not_trigger_source_event_strong_match(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=1, top_k=1)
    trajectory = store.create_trajectory(
        sample_id="sample-source-event-generic",
        dataset_name="locomo",
        label="generic event",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Jon had a general experience.",
            "source_event_records_v1": [
                {
                    "surface": "event",
                    "canonical": "event",
                    "category": "event_object",
                    "source_refs": ["D1:1"],
                    "source_message_ids": ["m1"],
                    "confidence": "high",
                }
            ],
            "source_event_object_terms_v1": ["event"],
            "source_event_canonical_terms_v1": ["event"],
        },
    )

    signals = engine._trajectory_retrieval_signals(trajectory)
    profile = engine._temporal_event_query_profile(
        "When did Jon attend the event?",
        {"item_family": "event"},
        extract_keywords("When did Jon attend the event?"),
    )
    match = engine._trajectory_source_event_match_profile(signals=signals, query_profile=profile)

    assert match["score"] == 0.0
    assert match["reason"] in {"no_specific_query_object_terms", "no_source_event_match"}


def test_broad_entity_page_candidate_cap_preserves_precise_pages_and_caps_entity_noise(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=5, top_k=5)
    sample_id = "sample-broad-cap"
    precise = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="adoption agencies",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Caroline researched adoption agencies.",
            "trajectory_historical_item_terms_v1": ["adoption agencies"],
            "entity_mentions": ["Caroline"],
        },
    )
    broad_ids: list[str] = []
    for index in range(36):
        trajectory = store.create_trajectory(
            sample_id=sample_id,
            dataset_name="locomo",
            label=f"entity-noise-{index}",
            strict_matching=False,
            max_length=6,
            metadata={
                "retrieval_summary_text": (
                    "Caroline researched adoption agency options."
                    if index == 7
                    else f"Caroline and Melanie discussed unrelated family support item {index}."
                ),
                "trajectory_historical_item_terms_v1": ["adoption agencies"] if index == 7 else ["family support"],
                "entity_mentions": ["Caroline"],
            },
        )
        broad_ids.append(trajectory.id)

    capped_ids, metadata = engine._apply_broad_entity_page_candidate_cap(
        sample_id=sample_id,
        selected_page_trajectory_ids=[precise.id, *broad_ids],
        page_metadata={
            "selected_page_rows": [
                {
                    "page_id": "inventory-adoption",
                    "page_type": "inventory",
                    "trajectory_ids": [precise.id],
                },
                {
                    "page_id": "entity-caroline",
                    "page_type": "entity",
                    "trajectory_ids": broad_ids,
                },
            ]
        },
        query_keywords={"caroline", "research", "adoption"},
        query_entities=["Caroline"],
        query_facet_tags=set(),
        query_facet_values=set(),
        query_shape={
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "research_topic",
        },
    )

    assert precise.id in capped_ids
    assert metadata["broad_entity_candidate_cap_used"] is True
    assert metadata["selected_page_universe_size_before_broad_cap"] == 37
    assert metadata["selected_page_universe_size_after_broad_cap"] <= 31
    assert broad_ids[7] in metadata["broad_entity_added_trajectory_ids"][:5]


def test_broad_entity_profile_is_suppressed_when_facet_pages_are_sufficient(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=5, top_k=5)
    sample_id = "sample-broad-profile-suppression"
    fine_ids: list[str] = []
    for index in range(5):
        trajectory = store.create_trajectory(
            sample_id=sample_id,
            dataset_name="locomo",
            label=f"fine-adoption-{index}",
            strict_matching=False,
            max_length=6,
            metadata={
                "retrieval_summary_text": f"Caroline researched adoption agencies detail {index}.",
                "trajectory_historical_item_terms_v1": ["adoption agencies"],
                "entity_mentions": ["Caroline"],
                "facet_values": ["research_topic=adoption agencies"],
                "facet_tags": ["research_topic"],
            },
        )
        fine_ids.append(trajectory.id)
    broad_ids: list[str] = []
    for index in range(36):
        trajectory = store.create_trajectory(
            sample_id=sample_id,
            dataset_name="locomo",
            label=f"broad-profile-{index}",
            strict_matching=False,
            max_length=6,
            metadata={
                "retrieval_summary_text": (
                    "Caroline researched adoption agencies and agency options."
                    if index == 3
                    else f"Caroline discussed unrelated family support topic {index}."
                ),
                "trajectory_historical_item_terms_v1": ["adoption agencies"] if index == 3 else ["family support"],
                "entity_mentions": ["Caroline"],
            },
        )
        broad_ids.append(trajectory.id)

    capped_ids, metadata = engine._apply_broad_entity_page_candidate_cap(
        sample_id=sample_id,
        selected_page_trajectory_ids=[*fine_ids, *broad_ids],
        page_metadata={
            "selected_page_rows": [
                {
                    "page_id": "topic-caroline-adoption",
                    "page_type": "topic",
                    "trajectory_ids": fine_ids,
                    "entity_facet_split_from_broad_page": True,
                },
                {
                    "page_id": "entity-caroline",
                    "page_type": "entity",
                    "trajectory_ids": broad_ids,
                    "broad_entity_profile": True,
                    "routing_priority": "profile",
                },
            ]
        },
        query_keywords={"caroline", "research", "adoption"},
        query_entities=["Caroline"],
        query_facet_tags={"research_topic"},
        query_facet_values={"research_topic=adoption agencies"},
        query_shape={
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "research_topic",
            "normalized_question": "What did Caroline research?",
        },
    )

    assert set(fine_ids) <= set(capped_ids)
    assert metadata["broad_entity_profile_suppressed"] is True
    assert metadata["fine_grained_entity_page_ids"] == ["topic-caroline-adoption"]
    assert metadata["entity_facet_page_candidate_count"] == 5
    assert broad_ids[3] in metadata["broad_entity_profile_added_trajectory_ids"]
    assert len(metadata["broad_entity_profile_added_trajectory_ids"]) < len(broad_ids)


def test_family_aligned_candidate_can_replace_generic_anchor(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=5, top_k=2)
    candidate_pool = [
        {
            "trajectory_id": "generic-family",
            "answer_family_match_score": 0.0,
            "answer_family_mismatch_penalty": 0.25,
            "coverage_profile": {
                "entity_keys": {"caroline"},
                "facet_values": set(),
                "support_terms": {"caroline", "family", "support"},
                "item_terms": set(),
                "exact_terms": [],
                "inventory_like": False,
                "has_count_signal": False,
            },
        },
        {
            "trajectory_id": "research-agencies",
            "answer_family_match_score": 0.82,
            "answer_family_matched_terms": ["research_options"],
            "answer_family_mismatch_penalty": 0.0,
            "coverage_profile": {
                "entity_keys": {"caroline"},
                "facet_values": set(),
                "support_terms": {"caroline", "research", "adoption", "agencies"},
                "item_terms": {"adoption", "agencies"},
                "exact_terms": ["adoption agencies"],
                "inventory_like": True,
                "has_count_signal": False,
            },
        },
    ]

    selected, metadata = engine._coverage_aware_select_ids(
        candidate_pool=candidate_pool,
        reranked_ids=["generic-family", "research-agencies"],
        id_key="trajectory_id",
        final_count=1,
        query_shape={
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "research_topic",
        },
        query_keywords={"caroline", "research", "adoption"},
        query_entity_keys={"caroline"},
        query_facet_values=set(),
    )

    assert selected == ["research-agencies"]
    assert metadata["selected_score_components"][0]["answer_family_match_score"] == 0.82


def test_answer_family_profile_prioritizes_specific_painted_object_over_generic_art() -> None:
    query_shape = {
        "list_like": True,
        "multi_entity": False,
        "comparison_like": False,
        "count_like": False,
        "item_family": "painted_object",
        "normalized_question": "What did Melanie paint recently?",
    }

    specific = RetrievalEngine._answer_family_match_profile(
        query_shape=query_shape,
        query_keywords={"melanie", "paint", "recently"},
        support_terms={"melanie", "painted", "sunset", "painting"},
        exact_terms=["sunset"],
        display_items=["sunset painting"],
        historical_item_terms=["sunset"],
        summary_text="Melanie recently painted a sunset.",
    )
    generic = RetrievalEngine._answer_family_match_profile(
        query_shape=query_shape,
        query_keywords={"melanie", "paint", "recently"},
        support_terms={"melanie", "family", "support", "creative"},
        summary_text="Melanie shared a creative family support experience.",
    )

    assert specific["score"] >= RetrievalEngine.FAMILY_MATCH_STRONG_THRESHOLD
    assert specific["score"] > generic["score"]
    assert generic["mismatch_penalty"] > 0


def test_answer_family_profile_matches_research_topic_and_person_events() -> None:
    research_shape = {
        "list_like": True,
        "multi_entity": False,
        "comparison_like": False,
        "count_like": False,
        "item_family": "research_topic",
        "normalized_question": "What did Caroline research?",
    }
    person_shape = {
        "list_like": True,
        "multi_entity": False,
        "comparison_like": False,
        "count_like": False,
        "item_family": "person",
        "normalized_question": "Which family member passed away?",
    }

    research = RetrievalEngine._answer_family_match_profile(
        query_shape=research_shape,
        query_keywords={"caroline", "research"},
        support_terms={"caroline", "researched", "adoption", "agencies", "options"},
        exact_terms=["adoption agencies"],
        historical_item_terms=["adoption agencies"],
        summary_text="Caroline researched adoption agencies and options.",
    )
    person = RetrievalEngine._answer_family_match_profile(
        query_shape=person_shape,
        query_keywords={"family", "member", "passed", "away"},
        support_terms={"mother", "passed", "away", "family"},
        exact_terms=["mother"],
        historical_item_terms=["mother passed away"],
        summary_text="Her mother passed away.",
    )

    assert research["score"] >= RetrievalEngine.FAMILY_MATCH_STRONG_THRESHOLD
    assert "research_options" in research["matched_terms"]
    assert person["score"] >= RetrievalEngine.FAMILY_MATCH_STRONG_THRESHOLD
    assert "person_signal" in person["matched_terms"]


def test_claim_facet_signals_deduplicate_and_ignore_empty_values() -> None:
    claims = [
        ClaimRecord(
            id="snapshot:claim-1",
            snapshot_id="snapshot",
            trajectory_id="trajectory",
            claim_id="claim-1",
            text="Caroline researched adoption agencies.",
            status="active",
            source_message_ids_json=[],
            parent_claim_id=None,
            revised_from_claim_id=None,
            metadata_json={
                "facets_v1": [
                    {
                        "entity": "Caroline",
                        "relation": "research_topic",
                        "value": "adoption agencies",
                    },
                    {
                        "entity": "Caroline",
                        "relation": "research_topic",
                        "value": "adoption agencies",
                    },
                    {"entity": "", "relation": "", "value": ""},
                ]
            },
        )
    ]

    signals = RetrievalEngine._claim_facet_signals(claims)

    assert signals.entity_keys == {"caroline"}
    assert signals.facet_tags == {"research_topic"}
    assert signals.facet_values == {"research_topic=adoption agencies"}


def test_retrieval_reflection_structured_payload_parses() -> None:
    spec = get_structured_task_spec("retrieval_reflection")

    parsed = parse_structured_payload(
        spec,
        {
            "rewritten_query": "flood area county",
            "answer_type": "place",
            "target_entities": ["West County"],
            "event_terms": ["flood"],
            "temporal_terms": [],
            "must_find_terms": ["flood", "West County"],
            "candidate_page_slugs": ["topic-flood"],
            "raw_search_terms": ["flood", "county"],
            "rationale": "Find raw or wiki evidence about a flood-hit area.",
        },
    )

    assert parsed.rewritten_query == "flood area county"
    assert parsed.candidate_page_slugs == ["topic-flood"]


def test_page_routing_uses_reflection_candidate_slug_bias(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-reflection-page-bias"
    target = WikiPageRecord(
        id="wiki-target",
        sample_id=sample_id,
        dataset_name="locomo",
        page_type="topic",
        title="Flood Recovery",
        slug="topic-flood",
        markdown_text="West County flood recovery.",
        keywords_json=[],
        trajectory_ids_json=["traj-flood"],
        linked_page_ids_json=[],
        entity_names_json=[],
        embedding_id=None,
        metadata_json={"routing_text": "West County flood recovery."},
    )
    distractor = WikiPageRecord(
        id="wiki-distractor",
        sample_id=sample_id,
        dataset_name="locomo",
        page_type="topic",
        title="Gardening",
        slug="topic-garden",
        markdown_text="Garden notes.",
        keywords_json=[],
        trajectory_ids_json=["traj-garden"],
        linked_page_ids_json=[],
        entity_names_json=[],
        embedding_id=None,
        metadata_json={"routing_text": "Garden notes."},
    )
    store.save_wiki_page(distractor)
    store.save_wiki_page(target)
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=1)

    selected_pages, trajectory_union, metadata = engine._route_pages(
        sample_id,
        "What area was hit by a flood?",
        provider.embed_queries(["flood"])[0],
        set(),
        [],
        {"list_like": False, "multi_entity": False, "comparison_like": False, "count_like": False},
        set(),
        {
            "candidate_page_slugs": ["topic-flood"],
            "must_find_terms": ["flood", "West County"],
        },
    )

    assert selected_pages == ["wiki-target"]
    assert trajectory_union == ["traj-flood"]
    ranked = {row["page_id"]: row for row in metadata["page_ranked_rows"]}
    assert ranked["wiki-target"]["reflection_bonus"] > 0


def test_page_routing_persists_compact_granularity_metadata(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-page-granularity-metadata"
    pages = [
        WikiPageRecord(
            id="wiki-medium",
            sample_id=sample_id,
            dataset_name="locomo",
            page_type="inventory",
            title="Adoption Research",
            slug="adoption-research",
            markdown_text="adoption agencies",
            keywords_json=["adoption"],
            trajectory_ids_json=["traj-a", "traj-b", "traj-c", "traj-d"],
            linked_page_ids_json=[],
            entity_names_json=["Caroline"],
            embedding_id=None,
            metadata_json={"routing_text": "adoption agencies research"},
        ),
        WikiPageRecord(
            id="wiki-singleton",
            sample_id=sample_id,
            dataset_name="locomo",
            page_type="inventory",
            title="Weak Singleton",
            slug="weak-singleton",
            markdown_text="family support",
            keywords_json=["family"],
            trajectory_ids_json=["traj-single"],
            linked_page_ids_json=[],
            entity_names_json=["Caroline"],
            embedding_id=None,
            metadata_json={
                "routing_text": "family support",
                "wiki_rescue_reason": "post_plan_index_only_trajectory",
                "wiki_singleton_policy": "merge_required_low_quality",
                "wiki_singleton_quality_score": 0,
            },
        ),
        WikiPageRecord(
            id="wiki-broad-profile",
            sample_id=sample_id,
            dataset_name="locomo",
            page_type="entity",
            title="Caroline Profile",
            slug="caroline-profile",
            markdown_text="Caroline profile",
            keywords_json=["caroline"],
            trajectory_ids_json=[f"traj-profile-{index}" for index in range(8)],
            linked_page_ids_json=[],
            entity_names_json=["Caroline"],
            embedding_id=None,
            metadata_json={
                "routing_text": "Caroline profile overview",
                "broad_entity_profile": True,
                "routing_priority": "profile",
            },
        ),
    ]
    for page in pages:
        store.save_wiki_page(page)
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=3, top_k=5)

    _, _, metadata = engine._route_pages(
        sample_id,
        "What did Caroline research?",
        provider.embed_queries(["adoption research"])[0],
        {"adoption", "research"},
        ["Caroline"],
        {
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "research_topic",
            "normalized_question": "What did Caroline research?",
        },
        set(),
        None,
    )

    assert metadata["selected_singleton_page_count"] is not None
    assert metadata["selected_medium_granularity_page_count"] >= 1
    assert metadata["selected_page_trajectory_count_histogram"]
    assert metadata["selected_page_rows_compact"]
    compact = {row["page_id"]: row for row in metadata["selected_page_rows_compact"]}
    assert compact["wiki-medium"]["medium_page_bonus"] > 0
    assert compact["wiki-medium"]["page_family_match_score"] > 0
    assert "research" in compact["wiki-medium"]["page_query_object_overlap_terms"]
    assert "routing_text" not in compact["wiki-medium"]
    assert "trajectory_ids" not in compact["wiki-medium"]
    assert any(row["singleton_page_penalty"] < 0 for row in compact.values())
    assert any(row["low_quality_singleton_penalty"] < 0 for row in compact.values())
    assert any(row["singleton_policy"] == "merge_required_low_quality" for row in compact.values())
    assert metadata["low_quality_singleton_penalty_applied"] >= 1
    assert metadata["page_granularity_diagnostic_mode"] == "retrieval_metadata"
    assert metadata["page_ranked_rows_compact_top_n"]
    ranked_compact = metadata["page_ranked_rows_compact_top_n"][0]
    assert "routing_text" not in ranked_compact
    assert "markdown_text" not in ranked_compact
    assert "prompt_context" not in ranked_compact
    assert "trajectory_ids" in ranked_compact
    assert "page_family_match_score" in ranked_compact
    assert "medium_bonus" in ranked_compact


def test_page_routing_persists_top_n_compact_rankings_and_cutoffs(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-page-compact-top-n"
    for index in range(60):
        store.save_wiki_page(
            WikiPageRecord(
                id=f"wiki-{index:02d}",
                sample_id=sample_id,
                dataset_name="locomo",
                page_type="topic",
                title=f"Research Page {index}",
                slug=f"research-page-{index}",
                markdown_text=f"Research notes {index}",
                keywords_json=["research", str(index)],
                trajectory_ids_json=[f"traj-{index:02d}"],
                linked_page_ids_json=[],
                entity_names_json=["Caroline"],
                embedding_id=None,
                metadata_json={"routing_text": f"Caroline research adoption agency {index}"},
            )
        )
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=3, top_k=10)

    _, _, metadata = engine._route_pages(
        sample_id,
        "What did Caroline research?",
        provider.embed_queries(["research adoption"])[0],
        {"research", "adoption"},
        ["Caroline"],
        {
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "research_topic",
            "normalized_question": "What did Caroline research?",
        },
        set(),
        None,
    )

    rows = metadata["page_ranked_rows_compact_top_n"]
    assert metadata["diagnostic_top_n_pages"] == 50
    assert metadata["page_ranked_total_count"] == 60
    assert metadata["page_ranked_rows_truncated"] is True
    assert len(rows) == 50
    assert rows[0]["rank"] == 1
    assert all("routing_text" not in row for row in rows)
    assert all("markdown_text" not in row for row in rows)
    assert metadata["page_cutoff_universe_diagnostics"]["5"]["selected_page_trajectory_count"] == 5
    assert metadata["page_cutoff_universe_diagnostics"]["50"]["selected_page_trajectory_count"] == 50


def test_page_routing_uses_wiki_source_event_terms(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-page-source-event"
    roadtrip = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="roadtrip",
        strict_matching=False,
        max_length=6,
        metadata={"retrieval_summary_text": "Melanie discussed family logistics."},
    )
    generic = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="generic",
        strict_matching=False,
        max_length=6,
        metadata={"retrieval_summary_text": "Melanie discussed family logistics."},
    )
    store.save_wiki_page(
        WikiPageRecord(
            id="wiki-roadtrip",
            sample_id=sample_id,
            dataset_name="locomo",
            page_type="inventory",
            title="Melanie Family Evidence",
            slug="wiki-roadtrip",
            markdown_text="Melanie family evidence.",
            keywords_json=["melanie", "family"],
            trajectory_ids_json=[roadtrip.id],
            linked_page_ids_json=[],
            entity_names_json=["Melanie"],
            embedding_id=None,
            metadata_json={
                "wiki_source_event_object_terms_v1": ["roadtrip"],
                "wiki_source_event_canonical_terms_v1": ["roadtrip"],
                "wiki_source_temporal_relation_terms_v1": ["this past weekend"],
                "wiki_historical_item_terms": ["roadtrip"],
                "exact_terms": ["roadtrip"],
            },
        )
    )
    store.save_wiki_page(
        WikiPageRecord(
            id="wiki-generic",
            sample_id=sample_id,
            dataset_name="locomo",
            page_type="inventory",
            title="Melanie General Evidence",
            slug="wiki-generic",
            markdown_text="Melanie family evidence.",
            keywords_json=["melanie", "family"],
            trajectory_ids_json=[generic.id],
            linked_page_ids_json=[],
            entity_names_json=["Melanie"],
            embedding_id=None,
            metadata_json={},
        )
    )
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=1)

    page_ids, trajectory_ids, metadata = engine._route_pages(
        sample_id,
        "When did Melanie's family go on a roadtrip?",
        provider.embed_queries(["When did Melanie's family go on a roadtrip?"])[0],
        extract_keywords("When did Melanie's family go on a roadtrip?"),
            ["Melanie"],
        {
            "item_family": "event",
            "list_like": False,
            "count_like": False,
            "normalized_question": "When did Melanie's family go on a roadtrip?",
        },
        set(),
        None,
    )

    assert page_ids == ["wiki-roadtrip"]
    assert trajectory_ids == [roadtrip.id]
    compact = metadata["selected_page_rows_compact"][0]
    assert compact["page_family_match_score"] > 0
    assert "roadtrip" in compact["page_query_object_overlap_terms"]


def test_trajectory_selection_persists_top_n_compact_rankings_and_pool_rows(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-trajectory-compact-top-n"
    trajectory_ids: list[str] = []
    for index in range(70):
        trajectory = store.create_trajectory(
            sample_id=sample_id,
            dataset_name="locomo",
            label=f"research-{index}",
            strict_matching=False,
            max_length=6,
            metadata={
                "retrieval_summary_text": f"Caroline researched adoption agency {index}.",
                "retrieval_summary_keywords": ["caroline", "research", "adoption", str(index)],
                "exact_terms_v2": [f"adoption agency {index}", "research"],
                "trajectory_historical_item_terms_v1": [f"agency {index}", "adoption"],
                "entity_mentions": ["Caroline"],
                "facet_values": ["adoption agencies"],
            },
        )
        trajectory_ids.append(trajectory.id)
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=3, top_k=10)

    _, metadata = engine._select_trajectories(
        sample_id,
        trajectory_ids,
        "What did Caroline research?",
        provider.embed_queries(["research adoption"])[0],
        {"research", "adoption"},
        ["Caroline"],
        set(),
        {"adoption agencies"},
        {
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "research_topic",
            "normalized_question": "What did Caroline research?",
        },
    )

    ranked = metadata["trajectory_ranked_rows_compact_top_n"]
    pool_rows = metadata["trajectory_selection_pool_rows_compact"]
    assert metadata["diagnostic_top_n_trajectories"] == 50
    assert metadata["trajectory_ranked_total_count"] == 70
    assert metadata["trajectory_ranked_rows_truncated"] is True
    assert len(ranked) == 50
    assert metadata["trajectory_selection_pool_rows_total_count"] == 40
    assert metadata["trajectory_selection_pool_rows_truncated"] is False
    assert pool_rows
    assert all("summary_text" not in row for row in ranked)
    assert all(len(row["exact_terms"]) <= 12 for row in ranked)
    assert metadata["trajectory_cutoff_prefix_diagnostics"]["50"]["trajectory_count"] == 50


def test_page_family_exact_match_can_cancel_singleton_penalty(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-page-family-singleton"
    pages = [
        WikiPageRecord(
            id="wiki-sunset",
            sample_id=sample_id,
            dataset_name="locomo",
            page_type="inventory",
            title="Melanie Painting",
            slug="melanie-painting",
            markdown_text="sunset painting",
            keywords_json=["painting"],
            trajectory_ids_json=["traj-sunset"],
            linked_page_ids_json=[],
            entity_names_json=["Melanie"],
            embedding_id=None,
            metadata_json={
                "routing_text": "Melanie painted a sunset with vivid colors.",
                "exact_terms": ["sunset"],
                "display_items": ["sunset"],
                "wiki_rescue_reason": "post_plan_index_only_trajectory",
            },
        ),
        WikiPageRecord(
            id="wiki-generic",
            sample_id=sample_id,
            dataset_name="locomo",
            page_type="inventory",
            title="Creative Nature",
            slug="creative-nature",
            markdown_text="creative nature support",
            keywords_json=["creative", "nature"],
            trajectory_ids_json=["traj-generic", "traj-generic-2", "traj-generic-3"],
            linked_page_ids_json=[],
            entity_names_json=["Melanie"],
            embedding_id=None,
            metadata_json={"routing_text": "creative nature support experience"},
        ),
    ]
    for page in pages:
        store.save_wiki_page(page)
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=5)

    selected, _, metadata = engine._route_pages(
        sample_id,
        "What did Melanie paint recently?",
        provider.embed_queries(["painting"])[0],
        {"paint", "painted", "sunset", "recently"},
        ["Melanie"],
        {
            "list_like": False,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "painted_object",
            "normalized_question": "What did Melanie paint recently?",
        },
        set(),
        None,
    )

    assert selected == ["wiki-sunset"]
    ranked = {row["page_id"]: row for row in metadata["page_ranked_rows"]}
    assert ranked["wiki-sunset"]["page_family_match_score"] > ranked["wiki-generic"]["page_family_match_score"]
    assert ranked["wiki-sunset"]["singleton_page_penalty"] == 0


def _add_raw_message(store, *, sample_id: str, ref: str, content: str, turn_index: int) -> str:
    record = store.add_raw_message(
        sample_id,
        "locomo",
        NormalizedMessage(
            role="user",
            content=content,
            turn_index=turn_index,
            speaker_name="Caroline",
            source_ref=ref,
        ),
    )
    return record.id


def test_raw_rescue_searches_only_current_sample_and_keeps_matching_turn(store) -> None:
    provider = _CompactionEmbeddingProvider()
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=1)
    target_id = _add_raw_message(
        store,
        sample_id="conv-41",
        ref="D1:7",
        content="West County was hit by a flood after the storm.",
        turn_index=7,
    )
    _add_raw_message(
        store,
        sample_id="conv-other",
        ref="D9:9",
        content="East County was hit by a flood.",
        turn_index=9,
    )
    store.session.flush()

    hits, metadata = engine._raw_rescue_messages(
        sample_id="conv-41",
        query_text="What area was hit by a flood?",
        effective_query_text="flood area county West County",
        reflection_hints={
            "must_find_terms": ["flood", "West County"],
            "raw_search_terms": ["flood", "county"],
        },
        query_embedding=provider.embed_queries(["flood area county West County"])[0],
        query_is_list_like=False,
        exclude_message_ids=set(),
    )

    assert [message.id for message in hits] == [target_id]
    assert metadata["raw_rescue_used"] is True
    assert metadata["raw_rescue_hit_count"] == 1
    assert metadata["raw_rescue_source_refs"] == ["D1:7"]


def test_raw_rescue_falls_back_to_lexical_when_embedding_fails(store) -> None:
    provider = _FailingDocumentEmbeddingProvider()
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=1)
    _add_raw_message(
        store,
        sample_id="conv-lexical",
        ref="D2:4",
        content="West County was hit by a flood.",
        turn_index=4,
    )
    store.session.flush()

    hits, metadata = engine._raw_rescue_messages(
        sample_id="conv-lexical",
        query_text="What area was hit by a flood?",
        effective_query_text="flood area West County",
        reflection_hints={"must_find_terms": ["flood", "West County"]},
        query_embedding=provider.embed_queries(["flood area West County"])[0],
        query_is_list_like=False,
        exclude_message_ids=set(),
    )

    assert [message.source_ref for message in hits] == ["D2:4"]
    assert metadata["raw_rescue_embedding_used"] is False
    assert metadata["raw_rescue_lexical_fallback"] is True
    assert metadata["raw_rescue_embedding_error"] == "embedding unavailable"


def test_index_fallback_expands_too_small_selected_page_universe(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-index-fallback"
    selected = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="selected",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Jon has a generic business note.",
            "entity_mentions": ["Jon"],
            "exact_terms": ["business note"],
            "latest_keywords": ["jon", "business"],
        },
    )
    fair = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="fair",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Jon showcased his studio at a fair.",
            "entity_mentions": ["Jon"],
            "exact_terms": ["showcased studio at a fair"],
            "display_items": ["studio fair"],
            "facet_tags": ["event"],
            "facet_values": ["event=fair"],
            "latest_keywords": ["jon", "fair", "event"],
        },
    )
    networking = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="networking",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Jon attended networking events to promote his business.",
            "entity_mentions": ["Jon"],
            "exact_terms": ["attended networking events"],
            "display_items": ["networking events"],
            "facet_tags": ["event"],
            "facet_values": ["event=networking events"],
            "latest_keywords": ["jon", "networking", "events"],
        },
    )
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=5)

    expanded, metadata = engine._index_fallback_trajectory_expansion(
        sample_id=sample_id,
        selected_page_trajectory_ids=[selected.id],
        page_metadata={
            "page_index_suppressed": True,
            "index_page_trajectory_ids": [selected.id, fair.id, networking.id],
            "page_rerank_selected_ids": ["wiki-small"],
        },
        query_keywords={"jon", "events", "business", "promote"},
        query_entities=["Jon"],
        query_facet_tags={"event"},
        query_facet_values=set(),
        query_shape={
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "event",
        },
    )

    assert metadata["index_fallback_used"] is True
    assert metadata["index_fallback_reason"] == "selected_page_universe_below_minimum"
    assert selected.id in expanded
    assert fair.id in expanded
    assert networking.id in expanded
    assert metadata["selected_page_universe_size_before_index_fallback"] == 1


def test_index_fallback_does_not_expand_sufficient_universe(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-index-no-fallback"
    selected_ids = []
    for index in range(10):
        trajectory = store.create_trajectory(
            sample_id=sample_id,
            dataset_name="locomo",
            label=f"selected-{index}",
            strict_matching=False,
            max_length=6,
            metadata={
                "retrieval_summary_text": f"Jon event {index}.",
                "entity_mentions": ["Jon"],
                "exact_terms": [f"event {index}"],
                "latest_keywords": ["jon", "event"],
            },
        )
        selected_ids.append(trajectory.id)
    extra = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="extra",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Jon attended a fair.",
            "entity_mentions": ["Jon"],
            "exact_terms": ["fair"],
            "latest_keywords": ["jon", "fair"],
        },
    )
    store.session.flush()
    engine = RetrievalEngine(store, provider, top_t_pages=5, top_k=5)

    expanded, metadata = engine._index_fallback_trajectory_expansion(
        sample_id=sample_id,
        selected_page_trajectory_ids=selected_ids,
        page_metadata={
            "page_index_suppressed": True,
            "index_page_trajectory_ids": [*selected_ids, extra.id],
            "page_rerank_selected_ids": ["wiki-rich"],
        },
        query_keywords={"jon", "events"},
        query_entities=["Jon"],
        query_facet_tags={"event"},
        query_facet_values=set(),
        query_shape={
            "list_like": True,
            "multi_entity": False,
            "comparison_like": False,
            "count_like": False,
            "item_family": "event",
        },
    )

    assert expanded == selected_ids
    assert metadata["index_fallback_used"] is False
    assert metadata["index_fallback_reason"] == "not_triggered"


def _add_retrieval_page(
    store,
    provider: _CompactionEmbeddingProvider,
    *,
    sample_id: str,
    trajectory_id: str,
    page_text: str,
    page_id: str = "wiki-evidence",
) -> None:
    page = WikiPageRecord(
        id=page_id,
        sample_id=sample_id,
        dataset_name="locomo",
        page_type="topic",
        title=page_text,
        slug=page_id,
        markdown_text=page_text,
        keywords_json=list(extract_keywords(page_text)),
        trajectory_ids_json=[trajectory_id],
        linked_page_ids_json=[],
        entity_names_json=[],
        embedding_id=f"{page_id}-emb",
        metadata_json={"routing_text": page_text},
    )
    store.save_wiki_page(page)
    store.save_embedding(
        embedding_id=f"{page_id}-emb",
        owner_type="wiki_page",
        owner_id=page.id,
        model_name=provider.model_info().model_name,
        vector=provider.embed_documents([page_text])[0],
        semantic_text=page_text,
        metadata={"document_embedding_strategy": provider.document_embedding_strategy()},
    )
    store.save_embedding(
        embedding_id=f"{trajectory_id}-summary-emb",
        owner_type="trajectory_summary",
        owner_id=trajectory_id,
        model_name=provider.model_info().model_name,
        vector=provider.embed_documents([page_text])[0],
        semantic_text=page_text,
        metadata={"document_embedding_strategy": provider.document_embedding_strategy()},
    )


def test_reflection_semantic_weakness_triggers_raw_rescue_with_non_empty_retrieval(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-raw-semantic-weak"
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=1, neighbor_radius=0)
    trajectory = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="community",
        strict_matching=False,
        max_length=6,
        metadata={"retrieval_summary_text": "Caroline discussed adoption agencies."},
    )
    linked_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D1:1",
        content="Caroline discussed adoption agencies.",
        turn_index=1,
    )
    rescue_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D1:7",
        content="West County was hit by a flood after the storm.",
        turn_index=7,
    )
    _add_snapshot(
        store,
        provider,
        trajectory_id=trajectory.id,
        version=1,
        semantic_text="Caroline discussed adoption agencies.",
        source_message_ids=[linked_id],
        claim_id=f"{trajectory.id}-seed",
    )
    _add_retrieval_page(
        store,
        provider,
        sample_id=sample_id,
        trajectory_id=trajectory.id,
        page_text="Caroline adoption agency notes.",
    )
    store.session.flush()

    bundle = engine.build_context(
        sample_id,
        "What area was hit by a flood?",
        attempt_label="reflection",
        reflection_hints={
            "must_find_terms": ["area", "hit", "flood", "flooding"],
            "raw_search_terms": ["flood", "West County"],
        },
    )

    assert bundle.metadata["raw_rescue_attempted"] is True
    assert "semantic_evidence_weak" in bundle.metadata["raw_rescue_trigger_reasons"]
    assert bundle.metadata["reflection_semantic_evidence_weak"] is True
    assert "flood" in {term.casefold() for term in bundle.metadata["reflection_uncovered_terms"]}
    assert rescue_id in bundle.source_message_ids
    assert "D1:7" in bundle.source_message_refs


def test_reflection_semantic_coverage_skips_raw_rescue_when_evidence_contains_terms(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-raw-semantic-covered"
    engine = RetrievalEngine(store, provider, top_t_pages=1, top_k=1, neighbor_radius=0)
    trajectory = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="flood",
        strict_matching=False,
        max_length=6,
        metadata={"retrieval_summary_text": "West County flood evidence."},
    )
    linked_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D2:1",
        content="West County was hit by a flood after the storm.",
        turn_index=1,
    )
    _add_snapshot(
        store,
        provider,
        trajectory_id=trajectory.id,
        version=1,
        semantic_text="West County was hit by a flood after the storm.",
        source_message_ids=[linked_id],
        claim_id=f"{trajectory.id}-seed",
    )
    _add_retrieval_page(
        store,
        provider,
        sample_id=sample_id,
        trajectory_id=trajectory.id,
        page_text="West County flood evidence.",
    )
    store.session.flush()

    bundle = engine.build_context(
        sample_id,
        "What area was hit by a flood?",
        attempt_label="reflection",
        reflection_hints={
            "must_find_terms": ["area", "hit", "flood", "flooding"],
            "raw_search_terms": ["flood", "West County"],
        },
    )

    assert bundle.metadata["raw_rescue_attempted"] is False
    assert bundle.metadata["raw_rescue_skipped_reason"] == "evidence_sufficient"
    assert bundle.metadata["reflection_semantic_evidence_weak"] is False
    assert "flood" in {term.casefold() for term in bundle.metadata["reflection_covered_terms"]}


def test_generic_reflection_terms_do_not_trigger_semantic_raw_rescue(store) -> None:
    engine = RetrievalEngine(store, _CompactionEmbeddingProvider(), top_t_pages=1, top_k=1)

    assert engine._reflection_required_terms(
        {"must_find_terms": ["area", "information"], "raw_search_terms": ["retrieve", "specific details"]}
    ) == []


def _add_snapshot(
    store,
    provider: _CompactionEmbeddingProvider,
    *,
    trajectory_id: str,
    version: int,
    semantic_text: str,
    source_message_ids: list[str],
    claim_id: str,
    revised_from_claim_id: str | None = None,
) -> EpisodicMemorySnapshot:
    snapshot = EpisodicMemorySnapshot(
        id=f"{trajectory_id}-v{version:03d}",
        trajectory_id=trajectory_id,
        version=version,
        timestamp=f"2026-04-20T10:00:{version:02d}Z",
        links_json=list(source_message_ids),
        summary_content=semantic_text,
        context=semantic_text,
        keywords_json=semantic_text.casefold().split(),
        status_flags_json=["active"],
        embedding_ref=f"{trajectory_id}-v{version:03d}-emb",
        semantic_text=semantic_text,
        raw_text=semantic_text,
        metadata_json={},
    )
    store.save_episodic_snapshot(snapshot)
    store.replace_claims_for_snapshot(
        [
            ClaimRecord(
                id=f"{snapshot.id}:claim",
                snapshot_id=snapshot.id,
                trajectory_id=trajectory_id,
                claim_id=claim_id,
                text=semantic_text,
                status="active",
                source_message_ids_json=list(source_message_ids),
                parent_claim_id=None,
                revised_from_claim_id=revised_from_claim_id,
                metadata_json={
                    "facets_v1": [
                        {
                            "entity": "Caroline",
                            "relation": "research_topic",
                            "value": "adoption agencies",
                            "value_span": "adoption agencies",
                        }
                    ]
                },
            )
        ]
    )
    store.save_embedding(
        embedding_id=snapshot.embedding_ref,
        owner_type="snapshot",
        owner_id=snapshot.id,
        model_name=provider.model_info().model_name,
        vector=provider.embed_documents([semantic_text])[0],
        semantic_text=semantic_text,
        metadata={"document_embedding_strategy": provider.document_embedding_strategy()},
    )
    return snapshot


def test_source_compaction_groups_sources_and_sorts_within_group(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-source-group-order"
    engine = RetrievalEngine(store, provider, top_t_pages=2, top_k=2, neighbor_radius=1)
    trajectory = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="research",
        strict_matching=False,
        max_length=6,
        metadata={},
    )
    later_message_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D1:2",
        content="Caroline added a later adoption note.",
        turn_index=20,
    )
    earlier_message_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D1:1",
        content="Caroline started the adoption note.",
        turn_index=10,
    )
    seed_snapshot = _add_snapshot(
        store,
        provider,
        trajectory_id=trajectory.id,
        version=1,
        semantic_text="seed adoption note",
        source_message_ids=[later_message_id, earlier_message_id],
        claim_id=f"{trajectory.id}-seed",
    )
    store.session.flush()

    source_state = engine._collect_snapshot_source_state([seed_snapshot])
    source_ids, source_refs, _, source_meta = engine._compact_source_messages(
        compacted_snapshots=[seed_snapshot],
        snapshot_source_state=source_state,
        seed_snapshot_ids=[seed_snapshot.id],
        retained_update_linked_snapshot_ids=[],
        retained_neighbor_snapshot_ids=[],
        query_is_list_like=False,
    )

    assert source_ids == [earlier_message_id, later_message_id]
    assert source_refs == ["D1:1", "D1:2"]
    assert source_meta["source_message_group_count"] == 1
    assert source_meta["source_message_grouped_ids"] == {
        "seed": [earlier_message_id, later_message_id],
        "update_linked": [],
        "neighbor": [],
    }
    assert source_meta["source_message_chronological_ids"] == [earlier_message_id, later_message_id]
    assert source_meta["source_message_backtrack_count"] == 0
    assert source_meta["source_message_backtrack_rate"] == 0.0


def test_build_context_renders_grouped_sources_and_chronological_timeline(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-build-context-source-order"
    trajectory = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="research",
        strict_matching=False,
        max_length=6,
        metadata={
            "retrieval_summary_text": "Caroline researched adoption agencies.",
            "retrieval_summary_keywords": ["caroline", "adoption", "agencies"],
            "entity_mentions": ["Caroline"],
        },
    )
    later_message_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D1:2",
        content="Caroline later compared adoption agencies.",
        turn_index=20,
    )
    earlier_message_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D1:1",
        content="Caroline first researched adoption agencies.",
        turn_index=10,
    )
    _add_snapshot(
        store,
        provider,
        trajectory_id=trajectory.id,
        version=1,
        semantic_text="Caroline researched adoption agencies.",
        source_message_ids=[later_message_id, earlier_message_id],
        claim_id=f"{trajectory.id}-seed",
    )
    page = WikiPageRecord(
        id="wiki-source-order",
        sample_id=sample_id,
        dataset_name="locomo",
        page_type="topic",
        title="Adoption Research",
        slug="adoption-research",
        markdown_text="Caroline researched adoption agencies.",
        keywords_json=["caroline", "adoption", "agencies"],
        trajectory_ids_json=[trajectory.id],
        linked_page_ids_json=[],
        entity_names_json=["Caroline"],
        embedding_id="wiki-source-order-emb",
        metadata_json={"routing_text": "Caroline researched adoption agencies."},
    )
    store.save_wiki_page(page)
    store.save_embedding(
        embedding_id="wiki-source-order-emb",
        owner_type="wiki_page",
        owner_id=page.id,
        model_name=provider.model_info().model_name,
        vector=provider.embed_documents(["Caroline researched adoption agencies."])[0],
        semantic_text="Caroline researched adoption agencies.",
        metadata={"document_embedding_strategy": provider.document_embedding_strategy()},
    )
    store.save_embedding(
        embedding_id=f"{trajectory.id}-summary-emb",
        owner_type="trajectory_summary",
        owner_id=trajectory.id,
        model_name=provider.model_info().model_name,
        vector=provider.embed_documents(["Caroline researched adoption agencies."])[0],
        semantic_text="Caroline researched adoption agencies.",
        metadata={"document_embedding_strategy": provider.document_embedding_strategy()},
    )
    store.session.flush()

    engine = RetrievalEngine(
        store,
        provider,
        top_t_pages=1,
        top_k=1,
        neighbor_radius=0,
    )

    bundle = engine.build_context(sample_id, "What did Caroline research about adoption?")

    assert "## Retrieved Source Messages" in bundle.prompt_context
    assert "### Seed Evidence" in bundle.prompt_context
    assert "## Chronological Source Timeline" in bundle.prompt_context
    grouped_earlier = bundle.prompt_context.index("D1:1 | id=")
    grouped_later = bundle.prompt_context.index("D1:2 | id=")
    assert grouped_earlier < grouped_later
    assert "turn=10" in bundle.prompt_context
    assert "turn=20" in bundle.prompt_context
    assert bundle.metadata["source_message_grouped_ids"]["seed"] == [earlier_message_id, later_message_id]
    assert bundle.metadata["source_message_chronological_ids"] == [earlier_message_id, later_message_id]
    assert bundle.metadata["source_message_backtrack_count"] == 0
    assert bundle.metadata["answer_context_active_claim_count"] == 1
    assert bundle.metadata["answer_context_uncertain_claim_count"] == 0
    assert bundle.metadata["answer_context_suppressed_deprecated_claim_count"] == 0
    assert bundle.metadata["answer_context_suppressed_ops_count"] == 0
    assert bundle.metadata["trajectory_candidate_pool_ids"] == [trajectory.id]
    assert bundle.metadata["trajectory_selection_pool_ids"] == [trajectory.id]
    assert bundle.metadata["trajectory_selection_pool_size"] == 1
    assert bundle.metadata["trajectory_rerank_pool_size"] == 1
    assert "adoption" in bundle.metadata["trajectory_covered_item_terms"]
    assert isinstance(bundle.metadata["trajectory_selected_score_components"], list)
    assert isinstance(bundle.metadata["trajectory_redundancy_penalties"], list)


def test_update_linked_expansion_only_adds_neighbors_around_seed(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-expand-boundary"
    engine = RetrievalEngine(store, provider, top_t_pages=2, top_k=2, neighbor_radius=1)
    trajectory = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label="research",
        strict_matching=False,
        max_length=6,
        metadata={},
    )
    seed_message_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D0:1",
        content="Caroline researched adoption agencies.",
        turn_index=1,
    )
    update_message_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D0:2",
        content="Caroline updated her adoption agency research notes.",
        turn_index=2,
    )
    neighbor_message_id = _add_raw_message(
        store,
        sample_id=sample_id,
        ref="D0:3",
        content="Caroline mentioned a neighbor detail.",
        turn_index=3,
    )

    seed_snapshot = _add_snapshot(
        store,
        provider,
        trajectory_id=trajectory.id,
        version=1,
        semantic_text="seed adoption note",
        source_message_ids=[seed_message_id],
        claim_id=f"{trajectory.id}-c001",
    )
    update_snapshot = _add_snapshot(
        store,
        provider,
        trajectory_id=trajectory.id,
        version=3,
        semantic_text="update adoption note",
        source_message_ids=[update_message_id],
        claim_id=f"{trajectory.id}-c003",
        revised_from_claim_id=f"{trajectory.id}-c001",
    )
    neighbor_snapshot = _add_snapshot(
        store,
        provider,
        trajectory_id=trajectory.id,
        version=4,
        semantic_text="neighbor only note",
        source_message_ids=[neighbor_message_id],
        claim_id=f"{trajectory.id}-c004",
    )
    store.session.flush()

    expanded, metadata = engine._expand_update_linked_plus_neighbors([seed_snapshot])

    assert [snapshot.id for snapshot in expanded] == [seed_snapshot.id, update_snapshot.id]
    assert metadata["update_linked_snapshot_ids"] == [update_snapshot.id]
    assert metadata["neighbor_fallback_snapshot_ids"] == []
    assert neighbor_snapshot.id not in metadata["raw_expanded_snapshot_ids"]


def test_snapshot_and_source_compaction_preserve_seed_and_cap_non_list_budget(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-compaction-non-list"
    engine = RetrievalEngine(store, provider, top_t_pages=5, top_k=5, neighbor_radius=1)
    selected_trajectory_ids: list[str] = []
    seed_snapshots: list[EpisodicMemorySnapshot] = []
    update_snapshots: list[EpisodicMemorySnapshot] = []
    neighbor_snapshots: list[EpisodicMemorySnapshot] = []
    turn_index = 1

    for index in range(5):
        trajectory = store.create_trajectory(
            sample_id=sample_id,
            dataset_name="locomo",
            label=f"traj-{index}",
            strict_matching=False,
            max_length=6,
            metadata={
                "entity_mentions": ["Caroline"],
                "facet_tags": ["research_topic"],
                "facet_values": ["research_topic=adoption agencies"],
            },
        )
        selected_trajectory_ids.append(trajectory.id)
        seed_refs = []
        update_refs = []
        neighbor_refs = []
        for offset in range(3):
            seed_refs.append(
                _add_raw_message(
                    store,
                    sample_id=sample_id,
                    ref=f"D{index}:{offset + 1}",
                    content=f"Seed {index} adoption message {offset}.",
                    turn_index=turn_index,
                )
            )
            turn_index += 1
            update_refs.append(
                _add_raw_message(
                    store,
                    sample_id=sample_id,
                    ref=f"D{index}:{offset + 11}",
                    content=f"Update {index} adoption message {offset}.",
                    turn_index=turn_index,
                )
            )
            turn_index += 1
        for offset in range(2):
            neighbor_refs.append(
                _add_raw_message(
                    store,
                    sample_id=sample_id,
                    ref=f"D{index}:{offset + 21}",
                    content=f"Neighbor {index} detail {offset}.",
                    turn_index=turn_index,
                )
            )
            turn_index += 1
        seed_snapshots.append(
            _add_snapshot(
                store,
                provider,
                trajectory_id=trajectory.id,
                version=1,
                semantic_text=f"seed adoption summary {index}",
                source_message_ids=seed_refs,
                claim_id=f"{trajectory.id}-seed",
            )
        )
        update_snapshots.append(
            _add_snapshot(
                store,
                provider,
                trajectory_id=trajectory.id,
                version=2,
                semantic_text=f"update adoption summary {index}",
                source_message_ids=update_refs,
                claim_id=f"{trajectory.id}-update",
                revised_from_claim_id=f"{trajectory.id}-seed",
            )
        )
        neighbor_snapshots.append(
            _add_snapshot(
                store,
                provider,
                trajectory_id=trajectory.id,
                version=3,
                semantic_text=f"neighbor summary {index}",
                source_message_ids=neighbor_refs,
                claim_id=f"{trajectory.id}-neighbor",
            )
        )
    store.session.flush()

    raw_expanded = seed_snapshots + update_snapshots + neighbor_snapshots
    compacted, snapshot_meta = engine._compact_expanded_snapshots(
        sample_id=sample_id,
        raw_expanded=raw_expanded,
        selected_trajectory_ids=selected_trajectory_ids,
        query_embedding=provider.embed_queries(["What did Caroline research?"])[0],
        query_keywords={"caroline", "research", "adoption"},
        query_entity_keys={"caroline"},
        query_facet_tags={"research_topic"},
        query_facet_values=set(),
        seed_snapshot_ids=[snapshot.id for snapshot in seed_snapshots],
        update_linked_snapshot_ids=[snapshot.id for snapshot in update_snapshots],
        neighbor_candidate_snapshot_ids=[snapshot.id for snapshot in neighbor_snapshots],
        query_is_list_like=False,
    )

    kept_ids = {snapshot.id for snapshot in compacted}
    assert len(compacted) == 10
    assert {snapshot.id for snapshot in seed_snapshots} <= kept_ids
    assert snapshot_meta["snapshot_compaction_counts"]["reserved_update_linked_kept"] == 5
    assert snapshot_meta["snapshot_compaction_counts"]["neighbor_kept"] == 0

    source_state = engine._collect_snapshot_source_state(compacted)
    source_ids, _, _, source_meta = engine._compact_source_messages(
        compacted_snapshots=compacted,
        snapshot_source_state=source_state,
        seed_snapshot_ids=[snapshot.id for snapshot in seed_snapshots],
        retained_update_linked_snapshot_ids=[
            snapshot.id for snapshot in update_snapshots if snapshot.id in kept_ids
        ],
        retained_neighbor_snapshot_ids=[],
        query_is_list_like=False,
    )

    assert len(source_ids) <= 28
    assert source_meta["source_compaction_counts"]["seed_source_count"] == 15
    assert source_meta["source_compaction_counts"]["update_linked_source_count"] == 10
    assert source_meta["source_compaction_counts"]["neighbor_source_count"] == 0


def test_snapshot_compaction_uses_wider_budget_for_list_like_queries(store) -> None:
    provider = _CompactionEmbeddingProvider()
    sample_id = "sample-compaction-list"
    engine = RetrievalEngine(store, provider, top_t_pages=5, top_k=5, neighbor_radius=1)
    selected_trajectory_ids: list[str] = []
    seed_snapshots: list[EpisodicMemorySnapshot] = []
    update_snapshots: list[EpisodicMemorySnapshot] = []
    neighbor_snapshots: list[EpisodicMemorySnapshot] = []
    turn_index = 1

    for index in range(5):
        trajectory = store.create_trajectory(
            sample_id=sample_id,
            dataset_name="locomo",
            label=f"traj-{index}",
            strict_matching=False,
            max_length=6,
            metadata={},
        )
        selected_trajectory_ids.append(trajectory.id)
        seed_message = _add_raw_message(
            store,
            sample_id=sample_id,
            ref=f"L{index}:1",
            content=f"Seed list item {index}",
            turn_index=turn_index,
        )
        turn_index += 1
        update_message = _add_raw_message(
            store,
            sample_id=sample_id,
            ref=f"L{index}:2",
            content=f"Update list item {index}",
            turn_index=turn_index,
        )
        turn_index += 1
        neighbor_message = _add_raw_message(
            store,
            sample_id=sample_id,
            ref=f"L{index}:3",
            content=f"Neighbor list item {index}",
            turn_index=turn_index,
        )
        turn_index += 1
        seed_snapshots.append(
            _add_snapshot(
                store,
                provider,
                trajectory_id=trajectory.id,
                version=1,
                semantic_text=f"seed adoption list {index}",
                source_message_ids=[seed_message],
                claim_id=f"{trajectory.id}-seed",
            )
        )
        update_snapshots.append(
            _add_snapshot(
                store,
                provider,
                trajectory_id=trajectory.id,
                version=2,
                semantic_text=f"update adoption list {index}",
                source_message_ids=[update_message],
                claim_id=f"{trajectory.id}-update",
                revised_from_claim_id=f"{trajectory.id}-seed",
            )
        )
        neighbor_snapshots.append(
            _add_snapshot(
                store,
                provider,
                trajectory_id=trajectory.id,
                version=3,
                semantic_text=f"neighbor list {index}",
                source_message_ids=[neighbor_message],
                claim_id=f"{trajectory.id}-neighbor",
            )
        )
    store.session.flush()

    compacted, snapshot_meta = engine._compact_expanded_snapshots(
        sample_id=sample_id,
        raw_expanded=seed_snapshots + update_snapshots + neighbor_snapshots,
        selected_trajectory_ids=selected_trajectory_ids,
        query_embedding=provider.embed_queries(["What books did Caroline mention?"])[0],
        query_keywords={"books", "caroline"},
        query_entity_keys={"caroline"},
        query_facet_tags=set(),
        query_facet_values=set(),
        seed_snapshot_ids=[snapshot.id for snapshot in seed_snapshots],
        update_linked_snapshot_ids=[snapshot.id for snapshot in update_snapshots],
        neighbor_candidate_snapshot_ids=[snapshot.id for snapshot in neighbor_snapshots],
        query_is_list_like=True,
    )

    assert len(compacted) == 13
    assert snapshot_meta["snapshot_compaction_budget"] == 13
    assert snapshot_meta["snapshot_compaction_counts"]["reserved_update_linked_kept"] == 5
    assert snapshot_meta["snapshot_compaction_counts"]["neighbor_kept"] == 3
