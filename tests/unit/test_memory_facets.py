from __future__ import annotations

from trajpatch.memory.facets import (
    assign_claim_metadata_v1,
    build_sample_entity_lexicon,
    build_trajectory_entity_facet_summary,
    classify_query_shape_v1,
    clean_entity_mentions_v1,
    clean_facet_records_v1,
    extract_claim_facets_v1,
    extract_entities_from_text,
    extract_exact_terms_v1,
    extract_query_facets_v1,
    is_list_like_query,
)
from trajpatch.storage.models import ClaimRecord, RawMessageRecord


def _raw_message(message_id: str, speaker_name: str, content: str) -> RawMessageRecord:
    return RawMessageRecord(
        id=message_id,
        sample_id="sample",
        dataset_name="locomo",
        turn_index=int(message_id.rsplit("m", 1)[-1]),
        role="user",
        speaker_name=speaker_name,
        content=content,
        source_ref=None,
        occurred_at=None,
        metadata_json={},
    )


def test_extract_entities_from_text_supports_possessives_and_filters_noise():
    messages = [
        _raw_message("sample-m0001", "Caroline", "Caroline is planning a trip."),
        _raw_message("sample-m0002", "Maria", "Maria wants to help."),
        _raw_message("sample-m0003", "Guide", "Did you know Caroline's mentor is helpful?"),
    ]
    lexicon = build_sample_entity_lexicon(messages)

    entities = extract_entities_from_text("What is Caroline's relationship status and what did Maria do?", lexicon)

    assert entities == ["Caroline", "Maria"]
    assert "Did" not in lexicon.values()


def test_clean_entity_mentions_v1_filters_fillers_fragments_and_weekdays():
    cleaned = clean_entity_mentions_v1(
        ["Caroline", "Maria", "Thanks", "Yeah", "I've", "Friday"],
        speaker_entities=["Caroline", "Maria"],
    )

    assert cleaned == ["Caroline", "Maria"]


def test_extract_claim_facets_v1_prefers_supporting_speaker_for_generic_user_claims():
    messages = [
        _raw_message("sample-m0001", "Caroline", "It'll be tough as a single parent."),
    ]
    entity_lexicon = build_sample_entity_lexicon(messages)

    facets = extract_claim_facets_v1(
        "The user plans to create a family for kids who need one.",
        messages,
        entity_lexicon,
    )

    assert facets[0]["entity"] == "Caroline"
    assert facets[0]["relation"] == "relationship_status"
    assert facets[0]["value"] == "single"


def test_assign_claim_metadata_v1_rebinds_research_facet_to_adoption_claim():
    source_messages = {
        "sample-m0001": _raw_message(
            "sample-m0001",
            "Caroline",
            "I am researching adoption agencies because I want to become a mom.",
        )
    }
    claims = [
        ClaimRecord(
            id="c1",
            snapshot_id="snap-1",
            trajectory_id="traj-1",
            claim_id="c1",
            text="The user dreams of having a family.",
            status="active",
            source_message_ids_json=["sample-m0001"],
            parent_claim_id=None,
            revised_from_claim_id=None,
            metadata_json={},
        ),
        ClaimRecord(
            id="c2",
            snapshot_id="snap-1",
            trajectory_id="traj-1",
            claim_id="c2",
            text="The user is exploring adoption agencies.",
            status="active",
            source_message_ids_json=["sample-m0001"],
            parent_claim_id=None,
            revised_from_claim_id=None,
            metadata_json={},
        ),
    ]

    assign_claim_metadata_v1(claims, source_messages, build_sample_entity_lexicon(source_messages.values()))

    family_facets = list(claims[0].metadata_json.get("facets_v1") or [])
    adoption_facets = list(claims[1].metadata_json.get("facets_v1") or [])
    assert family_facets == []
    assert adoption_facets[0]["relation"] == "research_topic"
    assert adoption_facets[0]["value"] == "adoption agencies"


def test_extract_exact_terms_v1_captures_titles_instruments_symbols_and_places():
    source_messages = [
        _raw_message(
            "sample-m0001",
            "Melanie",
            "I play clarinet and violin, I loved Becoming Nicole, and I waved a rainbow flag in San Francisco.",
        ),
    ]

    exact_terms = extract_exact_terms_v1(
        "Melanie shared favorite instruments and symbols.",
        source_messages,
    )

    assert "clarinet" in exact_terms
    assert "violin" in exact_terms
    assert "Becoming Nicole" in exact_terms
    assert "rainbow flag" in exact_terms
    assert "San Francisco" in exact_terms


def test_extract_exact_terms_v1_filters_fillers_pronouns_and_weak_title_noise():
    source_messages = [
        _raw_message(
            "sample-m0001",
            "Melanie",
            "Thanks, I've got Friday plans in San Francisco, I play clarinet, and I loved \"Becoming Nicole\" beside a rainbow flag.",
        ),
    ]

    exact_terms = extract_exact_terms_v1(
        "Melanie shared favorite instruments and books.",
        source_messages,
    )

    assert "Becoming Nicole" in exact_terms
    assert "clarinet" in exact_terms
    assert "rainbow flag" in exact_terms
    assert "San Francisco" in exact_terms
    assert "Thanks" not in exact_terms
    assert "I've" not in exact_terms
    assert "Friday" not in exact_terms


def test_clean_facet_records_v1_clears_invalid_entities_and_drops_invalid_values():
    facets = clean_facet_records_v1(
        [
            {
                "entity": "Thanks",
                "relation": "home_country",
                "value": "Sweden",
                "value_span": "home country Sweden",
                "facet_type": "origin",
                "source": "claim_text",
                "confidence": 0.97,
            },
            {
                "entity": "Caroline",
                "relation": "relationship_status",
                "value": "single",
                "value_span": "single parent",
                "facet_type": "identity_status",
                "source": "claim_text",
                "confidence": 0.98,
            },
            {
                "entity": "Caroline",
                "relation": "research_topic",
                "value": "Really",
                "value_span": "Really",
                "facet_type": "topic",
                "source": "claim_text",
                "confidence": 0.5,
            },
        ],
        speaker_entities=["Caroline"],
    )

    assert len(facets) == 2
    assert facets[0]["relation"] == "home_country"
    assert facets[0]["value"] == "Sweden"
    assert facets[0]["entity"] is None
    assert facets[1]["relation"] == "relationship_status"
    assert facets[1]["entity"] == "Caroline"


def test_build_trajectory_entity_facet_summary_uses_active_claims_and_exact_terms():
    source_messages = {
        "sample-m0001": _raw_message("sample-m0001", "Caroline", "I am researching adoption agencies."),
        "sample-m0002": _raw_message("sample-m0002", "Caroline", "I play the violin."),
    }
    claims = [
        ClaimRecord(
            id="c1",
            snapshot_id="snap-1",
            trajectory_id="traj-1",
            claim_id="c1",
            text="The user is exploring adoption agencies.",
            status="active",
            source_message_ids_json=["sample-m0001"],
            parent_claim_id=None,
            revised_from_claim_id=None,
            metadata_json={
                "facets_v1": [
                    {
                        "entity": "Caroline",
                        "relation": "research_topic",
                        "value": "adoption agencies",
                        "value_span": "adoption agencies",
                        "facet_type": "topic",
                        "source": "source_message:sample-m0001",
                        "confidence": 0.96,
                    }
                ],
                "exact_terms_v1": ["Becoming Nicole"],
            },
        ),
        ClaimRecord(
            id="c2",
            snapshot_id="snap-1",
            trajectory_id="traj-1",
            claim_id="c2",
            text="Caroline previously lived in Sweden.",
            status="deprecated",
            source_message_ids_json=["sample-m0002"],
            parent_claim_id=None,
            revised_from_claim_id=None,
            metadata_json={
                "facets_v1": [
                    {
                        "entity": "Caroline",
                        "relation": "home_country",
                        "value": "Sweden",
                        "value_span": "Sweden",
                        "facet_type": "origin",
                        "source": "source_message:sample-m0002",
                        "confidence": 0.97,
                    }
                ],
                "exact_terms_v1": ["Sweden"],
            },
        ),
    ]

    summary = build_trajectory_entity_facet_summary(
        claims,
        source_messages,
        build_sample_entity_lexicon(source_messages.values()),
    )

    assert summary["entity_mentions"] == ["Caroline"]
    assert summary["facet_tags"] == ["research_topic"]
    assert summary["facet_values"] == ["research_topic=adoption agencies"]
    assert summary["exact_terms"] == ["Becoming Nicole"]


def test_assign_claim_metadata_and_trajectory_summary_only_keep_cleaned_retrieval_fields():
    source_messages = {
        "sample-m0001": _raw_message(
            "sample-m0001",
            "Caroline",
            "Thanks, I've been a single parent since moving from my home country, Sweden. On Friday I played clarinet in San Francisco.",
        ),
    }
    claims = [
        ClaimRecord(
            id="c1",
            snapshot_id="snap-1",
            trajectory_id="traj-1",
            claim_id="c1",
            text="The user is a single parent who moved from a home country.",
            status="active",
            source_message_ids_json=["sample-m0001"],
            parent_claim_id=None,
            revised_from_claim_id=None,
            metadata_json={},
        )
    ]

    assign_claim_metadata_v1(claims, source_messages, build_sample_entity_lexicon(source_messages.values()))
    metadata = claims[0].metadata_json

    assert set(metadata["exact_terms_v1"]) == {"San Francisco", "Sweden", "clarinet"}
    assert metadata["exact_terms_discarded_v1"]
    assert "Friday" not in metadata["exact_terms_v1"]
    assert metadata["facets_v1"][0]["relation"] == "relationship_status"
    assert metadata["facets_v1"][1]["relation"] == "home_country"

    summary = build_trajectory_entity_facet_summary(
        claims,
        source_messages,
        build_sample_entity_lexicon(source_messages.values()),
    )

    assert summary["entity_mentions"] == ["Caroline"]
    assert "Friday" not in summary["exact_terms"]
    assert set(summary["exact_terms"]) == {"San Francisco", "Sweden", "clarinet"}
    assert summary["exact_terms_discarded_v1"]


def test_assign_claim_metadata_flags_first_person_wrong_speaker_claim():
    source_messages = {
        "sample-m0001": _raw_message(
            "sample-m0001",
            "Melanie",
            "My son got into an accident last week.",
        ),
    }
    claims = [
        ClaimRecord(
            id="c1",
            snapshot_id="snap-1",
            trajectory_id="traj-1",
            claim_id="c1",
            text="Caroline's son got into an accident last week.",
            status="active",
            source_message_ids_json=["sample-m0001"],
            parent_claim_id=None,
            revised_from_claim_id=None,
            metadata_json={},
        )
    ]

    assign_claim_metadata_v1(claims, source_messages, build_sample_entity_lexicon(source_messages.values()))
    summary = build_trajectory_entity_facet_summary(
        claims,
        source_messages,
        build_sample_entity_lexicon(source_messages.values()),
    )

    assert claims[0].metadata_json["speaker_grounding_suspect_v1"] is True
    assert "first_person_possessive_mismatched_subject" in claims[0].metadata_json[
        "speaker_grounding_suspect_reasons_v1"
    ]
    assert summary["display_key_facts"] == []


def test_extract_query_facets_v1_derives_entities_tags_and_location_values():
    lexicon = build_sample_entity_lexicon(
        [
            _raw_message("sample-m0001", "Caroline", "I moved from my home country."),
            _raw_message("sample-m0002", "Dave", "I returned from San Francisco."),
        ]
    )

    query_facets = extract_query_facets_v1("What was Dave doing in San Francisco?", lexicon)

    assert query_facets["entities"] == ["Dave"]
    assert "activity_location" in query_facets["tags"]
    assert "activity_location=san francisco" in query_facets["values"]


def test_is_list_like_query_detects_inventory_questions():
    assert is_list_like_query("What instruments does Melanie play?")
    assert is_list_like_query("Which cities did they both visit?")
    assert is_list_like_query("Which countries has James visited?")
    assert is_list_like_query("What items has Melanie bought?")
    assert is_list_like_query("What places or events has Calvin visited in Tokyo?")
    assert is_list_like_query("What desserts has Maria made?")
    assert is_list_like_query("What shelters did Nora mention?")
    assert is_list_like_query("What kind of writing does Tim do?")
    assert not is_list_like_query("What kind of project was Jolene working on?")
    assert is_list_like_query("What do Melanie's kids like?")
    assert is_list_like_query("What are Dave's dreams?")
    assert is_list_like_query("Who or which organizations have been the beneficiaries?")
    assert not is_list_like_query("Where did Caroline move from four years ago?")


def test_classify_query_shape_v1_detects_list_comparison_and_count_patterns():
    lexicon = build_sample_entity_lexicon(
        [
            _raw_message("sample-m0001", "Alice", "I read The Hobbit."),
            _raw_message("sample-m0002", "Bob", "I read Dune."),
        ]
    )

    comparison_shape = classify_query_shape_v1("Which books have both Alice and Bob read?", lexicon)
    count_shape = classify_query_shape_v1("How many books has Alice read?", lexicon)
    countries_shape = classify_query_shape_v1("Which countries has James visited?", lexicon)
    places_events_shape = classify_query_shape_v1("What places or events has Calvin visited in Tokyo?", lexicon)
    dessert_shape = classify_query_shape_v1("What desserts has Maria made?", lexicon)
    writing_shape = classify_query_shape_v1("What writings has Tim shared?", lexicon)
    painted_shape = classify_query_shape_v1("What did Melanie paint recently?", lexicon)
    research_shape = classify_query_shape_v1("What did Caroline research?", lexicon)
    band_shape = classify_query_shape_v1("What musical artists or bands has Melanie seen?", lexicon)
    screenplay_shape = classify_query_shape_v1("Which of Joanna's screenplay scripts was rejected?", lexicon)
    preference_shape = classify_query_shape_v1("What do Melanie's kids like?", lexicon)
    pet_names_shape = classify_query_shape_v1("What are Melanie's pets' names?", lexicon)
    dream_shape = classify_query_shape_v1("What are Dave's dreams?", lexicon)
    organization_shape = classify_query_shape_v1("Who or which organizations have been the beneficiaries?", lexicon)
    single_fact_shape = classify_query_shape_v1("Where did Caroline move from four years ago?", lexicon)
    kind_project_shape = classify_query_shape_v1(
        "What kind of project was Jolene working on in the beginning of January 2023?",
        lexicon,
    )
    duration_count_shape = classify_query_shape_v1(
        "How many weeks passed between Maria adopting Coco and Shadow?",
        lexicon,
    )

    assert comparison_shape["list_like"] is True
    assert comparison_shape["multi_entity"] is True
    assert comparison_shape["comparison_like"] is True
    assert comparison_shape["item_family"] == "book"
    assert set(comparison_shape["entities"]) == {"Alice", "Bob"}
    assert count_shape["count_like"] is True
    assert count_shape["item_family"] == "book"
    assert countries_shape["list_like"] is True
    assert countries_shape["item_family"] == "country"
    assert places_events_shape["list_like"] is True
    assert places_events_shape["item_family"] == "place"
    assert dessert_shape["list_like"] is True
    assert dessert_shape["item_family"] == "dessert"
    assert writing_shape["list_like"] is True
    assert writing_shape["item_family"] == "writing"
    assert painted_shape["list_like"] is True
    assert painted_shape["item_family"] == "painted_object"
    assert research_shape["list_like"] is True
    assert research_shape["item_family"] == "research_topic"
    assert band_shape["list_like"] is True
    assert band_shape["item_family"] == "band"
    assert screenplay_shape["list_like"] is True
    assert screenplay_shape["item_family"] == "writing"
    assert preference_shape["list_like"] is True
    assert preference_shape["item_family"] == "preference"
    assert pet_names_shape["list_like"] is True
    assert pet_names_shape["item_family"] == "pet"
    assert dream_shape["list_like"] is True
    assert dream_shape["item_family"] == "dream"
    assert organization_shape["list_like"] is True
    assert organization_shape["item_family"] == "organization"
    assert single_fact_shape["list_like"] is False
    assert kind_project_shape["list_like"] is False
    assert kind_project_shape["item_family"] == "type"
    assert duration_count_shape["count_like"] is True
    assert duration_count_shape["duration_count_like"] is True
    assert "duration_count" in duration_count_shape["tags"]
