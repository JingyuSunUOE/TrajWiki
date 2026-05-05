from __future__ import annotations

from trajpatch.memory.retrieval import RetrievalEngine
from trajpatch.memory.wiki import WikiCompiler, WikiPageDraft
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.storage.models import WikiPageRecord
from trajpatch.storage.repository import TrajPatchStore


class _KeywordPageEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(model_name="keyword-page-embedding")

    def _vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        if "adoption" in lowered:
            return [1.0, 0.0]
        if "music" in lowered:
            return [0.0, 1.0]
        return [-1.0, 0.0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def query_embedding_strategy(self) -> str:
        return "keyword-query"

    def document_embedding_strategy(self) -> str:
        return "keyword-document"


def _add_trajectory(
    store: TrajPatchStore,
    *,
    sample_id: str,
    dataset_name: str = "locomo",
    label: str = "background",
    metadata: dict | None = None,
) -> None:
    store.create_trajectory(
        sample_id=sample_id,
        dataset_name=dataset_name,
        label=label,
        strict_matching=False,
        max_length=6,
        metadata=metadata
        or {
            "retrieval_summary_text": "Caroline moved from Sweden and keeps in touch with family there.",
            "entity_mentions": ["Caroline"],
            "exact_terms": ["Sweden"],
            "facet_values": ["home_country=sweden"],
            "facet_tags": ["home_country"],
            "latest_keywords": ["moved", "sweden", "family"],
        },
    )
    store.session.flush()


def test_wiki_compiler_splits_broad_entity_into_facet_pages(store: TrajPatchStore) -> None:
    sample_id = "sample-broad-entity-facets"
    groups = [
        ("relationship_status=single", "single status"),
        ("research_topic=adoption agencies", "adoption agencies"),
        ("event_type=lgbtq events", "LGBTQ events"),
        ("painted_object=sunset painting", "sunset painting"),
        ("count=script rejections", "script rejections"),
    ]
    for group_index, (facet_value, term) in enumerate(groups):
        for item_index in range(6):
            _add_trajectory(
                store,
                sample_id=sample_id,
                label=f"{term}-{item_index}",
                metadata={
                    "retrieval_summary_text": f"Caroline discussed {term} detail {item_index}.",
                    "entity_mentions": ["Caroline"],
                    "exact_terms": [term],
                    "facet_values": [facet_value],
                    "facet_tags": [facet_value.split("=", 1)[0]],
                    "display_items": [term] if group_index in {1, 2, 3} else [],
                    "display_counts": ["2"] if group_index == 4 else [],
                    "trajectory_historical_item_terms_v1": [term],
                    "latest_keywords": ["caroline", *term.split()],
                },
            )

    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    seeds = compiler._plan_seeds(sample_id, store.list_trajectories(sample_id))

    profile_seeds = [seed for seed in seeds if seed.metadata.get("broad_entity_profile")]
    facet_seeds = [seed for seed in seeds if seed.metadata.get("seed_type") == "entity_facet"]

    assert len(profile_seeds) == 1
    assert profile_seeds[0].metadata["routing_priority"] == "profile"
    assert len(profile_seeds[0].trajectory_ids) == 30
    assert facet_seeds
    assert all(seed.metadata["entity_facet_split_from_broad_page"] is True for seed in facet_seeds)
    assert all(len(seed.trajectory_ids) <= compiler.MAX_PAGE_TRAJECTORIES for seed in facet_seeds)
    assert all(len(seed.trajectory_ids) >= compiler.MIN_GROUPABLE_PAGE_TRAJECTORIES for seed in facet_seeds)
    assert any(seed.metadata["entity_facet_source_entity"] == "Caroline" for seed in facet_seeds)
    assert any("adoption agencies" in seed.title.casefold() for seed in facet_seeds)


def test_wiki_compiler_splits_broad_entity_without_facet_values(store: TrajPatchStore) -> None:
    sample_id = "sample-broad-entity-no-facets"
    for index in range(25):
        _add_trajectory(
            store,
            sample_id=sample_id,
            label=f"pottery-bowl-{index}",
            metadata={
                "retrieval_summary_text": f"Caroline worked on pottery bowl detail {index}.",
                "entity_mentions": ["Caroline"],
                "exact_terms": [f"pottery bowl detail {index}"],
                "display_items": [f"pottery bowl detail {index}"],
                "trajectory_historical_item_terms_v1": [f"pottery bowl detail {index}"],
                "latest_keywords": ["caroline", "pottery", "bowl"],
            },
        )

    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    seeds = compiler._plan_seeds(sample_id, store.list_trajectories(sample_id))

    facet_seeds = [seed for seed in seeds if seed.metadata.get("seed_type") == "entity_facet"]

    assert facet_seeds
    assert all(seed.metadata["entity_facet_split_from_broad_page"] is True for seed in facet_seeds)
    assert all(3 <= len(seed.trajectory_ids) <= compiler.MAX_PAGE_TRAJECTORIES for seed in facet_seeds)
    assert any("pottery bowl" in seed.title.casefold() for seed in facet_seeds)


def test_wiki_compiler_traces_invalid_plan_and_invalid_markdown(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-invalid"
    _add_trajectory(store, sample_id=sample_id)
    traces: list[str] = []

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "wiki_page_plan":
            return "## Pages\n- malformed line without required fields"
        if task == "wiki_page_compile":
            return "# Freeform page without the required headings"
        raise AssertionError(f"Unexpected task: {task}")

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
        trace=traces.append,
    )

    pages = compiler.compile_sample(sample_id, "locomo")

    assert pages
    assert any(
        "wiki_plan_invalid reason=parse_empty" in line and "fallback=deterministic" in line
        for line in traces
    )
    assert any("wiki_seed_plan_done total=" in line and "latency_ms=" in line for line in traces)
    assert any("wiki_plan_fallback_done pages=" in line for line in traces)
    assert any("wiki_page_compile_start slug=index" in line for line in traces)
    assert any("wiki_page_compile_context slug=index" in line and "prompt_chars=" in line for line in traces)
    assert any(
        "wiki_page_compile_invalid slug=index reason=missing_headings" in line
        and "fallback=deterministic_markdown" in line
        for line in traces
    )
    assert any("wiki_compile_done pages=" in line and "latency_ms=" in line for line in traces)
    assert pages[0].markdown_text.startswith("## Overview")
    assert "## Linked Trajectories" in pages[0].markdown_text


def test_wiki_compiler_traces_generation_failures(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-failure"
    _add_trajectory(store, sample_id=sample_id)
    traces: list[str] = []

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "wiki_page_plan":
            raise RuntimeError("plan failed")
        if task == "wiki_page_compile":
            raise RuntimeError("compile failed")
        raise AssertionError(f"Unexpected task: {task}")

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
        trace=traces.append,
    )

    pages = compiler.compile_sample(sample_id, "locomo")

    assert pages
    assert any(
        "wiki_plan_failed error=RuntimeError" in line and "fallback=deterministic" in line
        for line in traces
    )
    assert any(
        "wiki_page_compile_failed slug=index error=RuntimeError" in line
        and "fallback=deterministic_markdown" in line
        for line in traces
    )
    assert any("wiki_page_compile_done page_id=" in line for line in traces)


def test_wiki_compiler_drops_empty_non_index_planner_pages(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-empty-planner-page"
    _add_trajectory(store, sample_id=sample_id, label="pets")
    trajectory_id = store.list_trajectories(sample_id)[0].id
    traces: list[str] = []

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "wiki_page_plan":
            return (
                "## Pages\n"
                f"- page_type=index | title=Index | slug=index | trajectories={trajectory_id} | entities=Caroline | links=none\n"
                "- page_type=inventory | title=Pets | slug=empty-pets | trajectories=none | entities=Caroline | links=index"
            )
        if task == "wiki_page_compile":
            return (
                "## Overview\n"
                "- Concrete overview.\n\n"
                "## Key Facts\n"
                "- Caroline moved from Sweden.\n\n"
                "## Items / Counts\n"
                "- Sweden.\n\n"
                "## Linked Trajectories\n"
                "- Linked.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None."
            )
        raise AssertionError(f"Unexpected task: {task}")

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
        trace=traces.append,
    )

    pages = compiler.compile_sample(sample_id, "locomo")

    assert all(page.slug != "empty-pets" for page in pages)
    assert any(page.page_type != "index" and trajectory_id in page.trajectory_ids_json for page in pages)
    assert any("wiki_empty_non_index_pages_dropped count=1 slugs=empty-pets" in line for line in traces)


def test_wiki_compiler_rewrites_placeholder_linked_section_and_metadata(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-placeholder"
    for index in range(10):
        _add_trajectory(
            store,
            sample_id=sample_id,
            label=f"topic-{index}",
            metadata={
                "retrieval_summary_text": f"Caroline discussed concrete memory topic {index}.",
                "retrieval_summary_keywords": ["caroline", "memory", f"topic{index}"],
                "entity_mentions": ["Caroline"],
                "exact_terms": [f"memory topic {index}"],
                "facet_values": [f"topic=memory topic {index}"],
                "latest_keywords": ["memory", f"topic{index}"],
            },
        )
    trajectory_ids = [trajectory.id for trajectory in store.list_trajectories(sample_id)]
    traces: list[str] = []

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "wiki_page_plan":
            return "## Pages\n- malformed line without required fields"
        if task == "wiki_page_compile":
            return (
                "## Overview\n"
                "- Page.\n\n"
                "## Key Facts\n"
                "- Facts.\n\n"
                "## Items / Counts\n"
                "- Items.\n\n"
                "## Linked Trajectories\n"
                f"- {trajectory_ids[0]}: Not provided.\n"
                "- fabricated-linked-row: Unknown.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None."
            )
        raise AssertionError(f"Unexpected task: {task}")

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
        trace=traces.append,
    )
    pages = compiler.compile_sample(sample_id, "locomo")
    index_page = next(page for page in pages if page.page_type == "index")

    assert "Not provided" not in index_page.markdown_text
    assert "fabricated-linked-row" not in index_page.markdown_text
    assert "Total linked trajectories: 10" in index_page.markdown_text
    for trajectory_id in trajectory_ids:
        assert trajectory_id in index_page.markdown_text
    assert index_page.metadata_json["linked_trajectory_section_rendered_deterministically"] is True
    assert index_page.metadata_json["linked_trajectory_description_count"] <= 4
    assert index_page.metadata_json["linked_trajectory_undescibed_count"] >= 6
    assert any(
        "wiki_page_linked_section_rewritten slug=index" in line
        and "linked=10" in line
        and "placeholder_detected=true" in line
        for line in traces
    )


def test_wiki_compiler_sanitizes_internal_summary_format_in_linked_section(
    store: TrajPatchStore,
) -> None:
    sample_id = "sample-wiki-linked-sanitize"
    _add_trajectory(
        store,
        sample_id=sample_id,
        label="home-children-caroline-kids-family",
        metadata={
            "retrieval_summary_text": (
                "## Profile / Stable Facts\n"
                "- Trajectory label: home-children-caroline-kids-family\n"
                "- Caroline\n"
                "- Melanie\n"
                "- Taking in kids in need."
            ),
            "entity_mentions": ["Caroline", "Melanie"],
            "exact_terms": ["Taking in kids in need"],
            "display_key_facts": ["Taking in kids in need."],
            "latest_keywords": ["kids", "family"],
            "trajectory_historical_evidence_card_v1": {
                "trajectory_id": "epi-sanitize",
                "identity_summary": "## Profile / Stable Facts\n- Trajectory label: debug-label\n- Taking in kids in need.",
                "recent_update": "recent_update=Caroline is taking in kids in need.",
                "source_surface_terms": ["Taking in kids in need"],
                "historical_item_terms": ["None recorded", "Taking in kids in need"],
                "facet_values": ["identity_summary=debug"],
                "entity_mentions": ["Caroline", "Melanie"],
                "source_anchors": [
                    {
                        "source_ref": "D1:1",
                        "text": "Trajectory label: debug-label - Taking in kids in need.",
                    }
                ],
            },
        },
    )
    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    trajectory = store.list_trajectories(sample_id)[0]
    seed = compiler._build_seed(
        seed_id="index",
        page_type="index",
        title="Index",
        slug="index",
        trajectory_ids=[trajectory.id],
        entities=["Caroline", "Melanie"],
        trajectories_by_id={trajectory.id: trajectory},
    )
    draft = compiler._draft_from_seed(seed)

    linked_section = compiler._render_linked_trajectory_section(draft, {trajectory.id: trajectory})

    assert "None recorded" not in seed.metadata["wiki_historical_item_terms"]
    assert "Taking in kids in need" in seed.metadata["wiki_historical_item_terms"]
    assert "Taking in kids in need" in linked_section
    assert "## Profile / Stable Facts" not in linked_section
    assert "Trajectory label:" not in linked_section


def test_wiki_routing_text_sanitizes_internal_summary_format(
    store: TrajPatchStore,
) -> None:
    sample_id = "sample-wiki-routing-sanitize"
    _add_trajectory(
        store,
        sample_id=sample_id,
        label="home-children-caroline-kids-family",
        metadata={
            "retrieval_summary_text": (
                "## Profile / Stable Facts\n"
                "- Trajectory label: home-children-caroline-kids-family\n"
                "- Caroline\n"
                "- Melanie\n"
                "- Taking in kids in need.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None recorded."
            ),
            "entity_mentions": ["Caroline", "Melanie"],
            "exact_terms": ["Taking in kids in need"],
            "display_key_facts": ["Taking in kids in need."],
            "latest_keywords": ["kids", "family"],
        },
    )
    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    trajectory = store.list_trajectories(sample_id)[0]
    seed = compiler._build_seed(
        seed_id="topic",
        page_type="topic",
        title="Family support",
        slug="family-support",
        trajectory_ids=[trajectory.id],
        entities=["Caroline", "Melanie"],
        trajectories_by_id={trajectory.id: trajectory},
    )
    draft = compiler._draft_from_seed(seed)

    routing_text = compiler._build_routing_text(draft, {trajectory.id: trajectory})
    fallback_markdown = compiler._fallback_page_markdown(draft, [trajectory], {trajectory.id: trajectory})

    assert "Taking in kids in need" in routing_text
    assert "Taking in kids in need" in fallback_markdown
    human_card_sections = compiler._trajectory_evidence_card_sections(draft, {trajectory.id: trajectory})
    assert any("CARD" in section for section in human_card_sections)
    for forbidden in [
        "CARD ",
        "identity_summary=",
        "recent_update=",
        "source_anchors=",
        "## Profile / Stable Facts",
        "Trajectory label:",
        "None recorded",
    ]:
        assert forbidden not in routing_text
    for forbidden in ["## Profile / Stable Facts", "Trajectory label:", "None recorded"]:
        assert forbidden not in fallback_markdown


def test_wiki_post_plan_audit_rescues_planner_dropped_trajectories(
    store: TrajPatchStore,
) -> None:
    sample_id = "sample-wiki-post-plan-rescue"
    for index in range(8):
        _add_trajectory(
            store,
            sample_id=sample_id,
            label=f"jon-business-event-{index}",
            metadata={
                "retrieval_summary_text": f"Jon business event evidence {index}.",
                "retrieval_summary_keywords": ["jon", "business", f"event{index}"],
                "entity_mentions": ["Jon"],
                "exact_terms": [f"networking event {index}"],
                "display_items": [f"networking event {index}"],
                "display_key_facts": [f"Jon attended networking event {index}."],
                "facet_values": [f"event=networking event {index}"],
                "latest_keywords": ["jon", "event", f"event{index}"],
            },
        )
    trajectory_ids = [trajectory.id for trajectory in store.list_trajectories(sample_id)]
    traces: list[str] = []

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "wiki_page_plan":
            return (
                "## Pages\n"
                f"- page_type=index | title=Index | slug=index | trajectories={','.join(trajectory_ids)} | entities=Jon | links=none\n"
                f"- page_type=inventory | title=Only one event | slug=only-one-event | trajectories={trajectory_ids[0]} | entities=Jon | links=index"
            )
        if task == "wiki_page_compile":
            return (
                "## Overview\n"
                "- Page.\n\n"
                "## Key Facts\n"
                "- No specific key facts.\n\n"
                "## Items / Counts\n"
                "- No explicit items.\n\n"
                "## Linked Trajectories\n"
                "- Not provided.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None."
            )
        raise AssertionError(f"Unexpected task: {task}")

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
        trace=traces.append,
    )

    pages = compiler.compile_sample(sample_id, "locomo")
    non_index_trajectory_ids = {
        trajectory_id
        for page in pages
        if page.page_type != "index"
        for trajectory_id in page.trajectory_ids_json
    }
    rescue_pages = [
        page
        for page in pages
        if page.metadata_json.get("wiki_rescue_reason") == "post_plan_index_only_trajectory"
    ]

    assert set(trajectory_ids) <= non_index_trajectory_ids
    assert rescue_pages
    assert all(page.metadata_json["routing_priority"] == "high" for page in rescue_pages)
    assert all(3 <= len(page.trajectory_ids_json) <= compiler.MAX_PAGE_TRAJECTORIES for page in rescue_pages)
    assert all(page.metadata_json["wiki_singleton_exception"] is False for page in rescue_pages)
    assert all("No specific key facts" not in page.markdown_text for page in rescue_pages)
    assert all("Not provided" not in page.markdown_text for page in rescue_pages)
    assert rescue_pages[0].metadata_json["wiki_singleton_non_index_page_rate"] < 0.5
    assert any("wiki_non_index_coverage_audit" in line and "rescue_pages=" in line for line in traces)
    assert any("wiki_fragmentation_diagnostics" in line for line in traces)
    assert not any("wiki_non_index_coverage_incomplete" in line for line in traces)


def test_post_plan_rescue_merges_noisy_singletons_into_medium_pages(store: TrajPatchStore) -> None:
    sample_id = "sample-rescue-umbrella-merge"
    terms = [
        ("Caroline", "painted sunset", "sunset painting"),
        ("Caroline", "painted horse", "horse artwork"),
        ("Caroline", "research adoption agencies", "adoption agencies"),
        ("Caroline", "looked into adoption options", "adoption options"),
        ("Caroline", "read The Hobbit", "The Hobbit"),
        ("Caroline", "read Charlotte's Web", "Charlotte's Web"),
        ("Caroline", "that exchange was amazing", "that exchange"),
        ("Caroline", "support facts were great", "support facts"),
    ]
    for index, (entity, summary, exact) in enumerate(terms):
        _add_trajectory(
            store,
            sample_id=sample_id,
            label=f"{exact}-{index}",
            metadata={
                "retrieval_summary_text": f"{entity} discussed {summary}.",
                "entity_mentions": [entity],
                "exact_terms": [exact],
                "display_items": [exact],
                "trajectory_historical_item_terms_v1": [exact],
                "latest_keywords": [entity, *exact.split()],
            },
        )
    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    trajectories = store.list_trajectories(sample_id)
    drafts = compiler._build_post_plan_rescue_drafts(
        sample_id,
        [trajectory.id for trajectory in trajectories],
        compiler._trajectory_rows_by_id(trajectories),
        set(),
    )

    assert drafts
    assert len(drafts) < len(terms)
    assert all(len(draft.trajectory_ids) <= compiler.MAX_PAGE_TRAJECTORIES for draft in drafts)
    assert any(len(draft.trajectory_ids) >= compiler.MIN_GROUPABLE_PAGE_TRAJECTORIES for draft in drafts)
    assert any(draft.metadata.get("wiki_rescue_merge_applied") is True for draft in drafts)
    assert all(
        draft.metadata.get("singleton_exception_reason") == "isolated_after_rescue_merge"
        for draft in drafts
        if len(draft.trajectory_ids) == 1
    )


def test_singleton_policy_allows_specific_pages_and_rejects_low_quality_descriptors() -> None:
    specific = WikiCompiler._granularity_metadata(
        1,
        descriptor="Bach evidence",
        family="music",
        group={"specific_values": ["Bach"]},
    )
    low_quality = WikiCompiler._granularity_metadata(
        1,
        descriptor="Plus evidence",
        family="plus",
        group={"specific_values": ["plus"]},
    )
    rewritten = WikiCompiler._group_descriptor_metadata(
        {
            "descriptors": ["plus"],
            "specific_values": ["Charlotte's Web"],
            "entities": ["Melanie"],
        },
        fallback="books_and_reading",
    )

    assert specific["wiki_singleton_policy"] == "allowed_isolated_specific"
    assert specific["wiki_singleton_allowed"] is True
    assert low_quality["wiki_singleton_policy"] == "merge_required_low_quality"
    assert low_quality["wiki_singleton_allowed"] is False
    assert rewritten["descriptor"] == "Charlotte's Web"
    assert rewritten["wiki_descriptor_rewritten"] is True


def test_wiki_compiler_splits_overwide_non_index_drafts(store: TrajPatchStore) -> None:
    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    draft = WikiPageDraft(
        page_type="entity",
        title="Caroline Profile",
        slug="caroline-profile",
        trajectory_ids=[f"traj-{index}" for index in range(10)],
        entities=["Caroline"],
        linked_slugs=[],
        metadata={"seed_type": "entity", "routing_priority": "profile"},
    )

    split_drafts, overwide_count = compiler._split_overwide_non_index_drafts("sample-overwide", [draft])

    assert overwide_count == 1
    assert len(split_drafts) > 1
    assert all(len(split.trajectory_ids) <= compiler.MAX_PAGE_TRAJECTORIES for split in split_drafts)
    assert all(3 <= len(split.trajectory_ids) <= compiler.MAX_PAGE_TRAJECTORIES for split in split_drafts)
    assert all(split.metadata["wiki_overwide_page_split"] is True for split in split_drafts)
    assert split_drafts[0].metadata["wiki_overwide_original_trajectory_count"] == 10


def test_wiki_compiler_synthesizes_evidence_for_planner_derived_page(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-planner-derived"
    for label, term, summary in [
        (
            "counseling",
            "LGBTQ+ counseling workshop",
            "Caroline attended an LGBTQ+ counseling workshop about supporting trans people.",
        ),
        (
            "dinosaur",
            "dinosaur exhibit",
            "Melanie's kids were excited for the dinosaur exhibit and animal learning.",
        ),
        (
            "pottery",
            "pottery bowl",
            "Melanie shared a colorful pottery bowl and felt proud of the project.",
        ),
    ]:
        _add_trajectory(
            store,
            sample_id=sample_id,
            label=label,
            metadata={
                "retrieval_summary_text": summary,
                "retrieval_summary_keywords": term.lower().split(),
                "entity_mentions": ["Caroline", "Melanie"],
                "exact_terms": [term],
                "facet_values": [f"topic={term.lower()}"],
                "display_items": [summary],
                "latest_keywords": term.lower().split(),
            },
        )
    trajectory_ids = [trajectory.id for trajectory in store.list_trajectories(sample_id)]
    traces: list[str] = []
    compile_prompts: dict[str, str] = {}

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "wiki_page_plan":
            return (
                "## Pages\n"
                f"- page_type=index | title=Index | slug=index | trajectories={','.join(trajectory_ids)} | entities=Caroline,Melanie | links=none\n"
                f"- page_type=topic | title=Celebrating Life Moments | slug=celebrating-life-moments | trajectories={','.join(trajectory_ids)} | entities=Caroline,Melanie | links=index"
            )
        if task == "wiki_page_compile":
            prompt = messages[-1].content
            if "Page title:\nCelebrating Life Moments" in prompt:
                compile_prompts["topic"] = prompt
                return (
                    "## Overview\n"
                    "This page covers celebrating life moments.\n\n"
                    "## Key Facts\n"
                    "- No specific key facts available.\n\n"
                    "## Items / Counts\n"
                    "- No explicit items or counts provided.\n\n"
                    "## Linked Trajectories\n"
                    "- Linked.\n\n"
                    "## Conflicts / Uncertainty\n"
                    "- None."
                )
            return (
                "## Overview\n"
                "- Index.\n\n"
                "## Key Facts\n"
                "- Concrete overview.\n\n"
                "## Items / Counts\n"
                "- Concrete items.\n\n"
                "## Linked Trajectories\n"
                "- Linked.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None."
            )
        raise AssertionError(f"Unexpected task: {task}")

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
        trace=traces.append,
    )
    pages = compiler.compile_sample(sample_id, "locomo")
    topic_page = next(page for page in pages if page.slug == "celebrating-life-moments")

    assert "Representative trajectory summaries:\n- none" not in compile_prompts["topic"]
    assert "Trajectory evidence cards:\n- none" not in compile_prompts["topic"]
    assert "LGBTQ+ counseling workshop" in topic_page.metadata_json["routing_text"]
    assert "dinosaur exhibit" in topic_page.metadata_json["routing_text"]
    assert "pottery bowl" in topic_page.metadata_json["routing_text"]
    assert "No specific key facts" not in topic_page.markdown_text
    assert "No explicit items" not in topic_page.markdown_text
    assert "LGBTQ+ counseling workshop" in topic_page.markdown_text
    assert topic_page.metadata_json["wiki_seed_match_source"] in {"same_type_overlap", "any_type_overlap"}
    assert topic_page.metadata_json["wiki_evidence_trajectory_count"] == 3
    assert any(
        "wiki_page_compile_invalid slug=celebrating-life-moments reason=placeholder_content" in line
        for line in traces
    )


def test_wiki_compiler_overwrites_seed_evidence_when_planner_trajectory_ids_differ(
    store: TrajPatchStore,
) -> None:
    sample_id = "sample-wiki-stale-seed"
    _add_trajectory(
        store,
        sample_id=sample_id,
        label="adoption",
        metadata={
            "retrieval_summary_text": "Caroline researched an adoption agency.",
            "entity_mentions": ["Caroline"],
            "exact_terms": ["adoption agency"],
            "facet_values": ["topic=adoption"],
            "display_items": ["Caroline researched an adoption agency."],
            "latest_keywords": ["adoption"],
        },
    )
    _add_trajectory(
        store,
        sample_id=sample_id,
        label="pottery",
        metadata={
            "retrieval_summary_text": "Melanie shared a colorful pottery bowl.",
            "entity_mentions": ["Melanie"],
            "exact_terms": ["pottery bowl"],
            "facet_values": ["item=pottery bowl"],
            "display_items": ["Melanie shared a colorful pottery bowl."],
            "latest_keywords": ["pottery", "bowl"],
        },
    )
    trajectories = store.list_trajectories(sample_id)
    trajectories_by_id = {trajectory.id: trajectory for trajectory in trajectories}
    adoption, pottery = trajectories
    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    stale_seed = compiler._build_seed(
        seed_id="topic-art",
        page_type="topic",
        title="Art",
        slug="art",
        trajectory_ids=[adoption.id],
        entities=["Caroline"],
        trajectories_by_id=trajectories_by_id,
    )
    planner_draft = compiler._draft_from_seed(stale_seed)
    planner_draft = type(planner_draft)(
        page_type=planner_draft.page_type,
        title=planner_draft.title,
        slug=planner_draft.slug,
        trajectory_ids=[pottery.id],
        entities=["Melanie"],
        linked_slugs=planner_draft.linked_slugs,
        metadata={},
    )

    enriched = compiler._attach_seed_metadata(
        sample_id,
        [planner_draft],
        [stale_seed],
        trajectories_by_id,
    )[0]

    assert enriched.metadata["wiki_seed_match_source"] == "slug_exact"
    assert enriched.metadata["wiki_evidence_metadata_synthesized"] is True
    assert enriched.metadata["wiki_evidence_source_trajectory_ids"] == [pottery.id]
    assert "pottery bowl" in enriched.metadata["exact_terms"]
    assert "adoption agency" not in enriched.metadata["exact_terms"]
    assert enriched.metadata["representative_trajectory_ids"] == [pottery.id]


def test_wiki_compiler_synthesizes_metadata_when_no_seed_matches(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-no-seed"
    _add_trajectory(
        store,
        sample_id=sample_id,
        label="pottery",
        metadata={
            "retrieval_summary_text": "Melanie shared a colorful pottery bowl.",
            "retrieval_summary_keywords": ["melanie", "pottery", "bowl"],
            "entity_mentions": ["Melanie"],
            "exact_terms": ["pottery bowl"],
            "facet_values": ["item=pottery bowl"],
            "display_items": ["Melanie shared a colorful pottery bowl."],
            "latest_keywords": ["pottery", "bowl"],
        },
    )
    trajectory = store.list_trajectories(sample_id)[0]
    compiler = WikiCompiler(store, MockLLMProvider(), HashEmbeddingProvider())
    draft = compiler._attach_seed_metadata(
        sample_id,
        [
            compiler._draft_from_seed(
                compiler._build_seed(
                    seed_id="temporary",
                    page_type="topic",
                    title="Temporary",
                    slug="temporary",
                    trajectory_ids=[trajectory.id],
                    entities=[],
                    trajectories_by_id={trajectory.id: trajectory},
                )
            )
        ],
        [],
        {trajectory.id: trajectory},
    )[0]

    assert draft.metadata["wiki_seed_match_source"] == "trajectory_synthesized"
    assert draft.metadata["wiki_evidence_metadata_synthesized"] is True
    assert draft.metadata["wiki_evidence_trajectory_count"] == 1
    assert "pottery bowl" in draft.metadata["exact_terms"]
    assert draft.metadata["representative_trajectory_ids"] == [trajectory.id]


def test_wiki_compiler_splits_broad_entity_seed_and_caps_representatives(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-broad"
    seen_representative_counts: list[int] = []
    traces: list[str] = []

    for index in range(7):
        _add_trajectory(
            store,
            sample_id=sample_id,
            label=f"background-{index}",
            metadata={
                "retrieval_summary_text": f"Caroline discussed adoption agencies topic {index}.",
                "retrieval_summary_keywords": ["caroline", "adoption", f"topic{index}"],
                "entity_mentions": ["Caroline"],
                "exact_terms": [f"adoption topic {index}"],
                "facet_values": [f"research_topic=adoption topic {index}"],
                "facet_tags": ["research_topic"],
                "latest_keywords": ["adoption", "topic"],
            },
        )

    def callback(messages, system_prompt, metadata):
        task = (metadata or {}).get("task")
        if task == "wiki_page_plan":
            return "## Pages\n- malformed line without required fields"
        if task == "wiki_page_compile":
            prompt = messages[-1].content
            seen_representative_counts.append(prompt.count("### "))
            return (
                "## Overview\n"
                "- Compact page.\n\n"
                "## Key Facts\n"
                "- Facts.\n\n"
                "## Items / Counts\n"
                "- Items.\n\n"
                "## Linked Trajectories\n"
                "- Linked.\n\n"
                "## Conflicts / Uncertainty\n"
                "- None."
            )
        raise AssertionError(f"Unexpected task: {task}")

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=callback),
        HashEmbeddingProvider(),
        trace=traces.append,
    )
    pages = compiler.compile_sample(sample_id, "locomo")

    entity_pages = [page for page in pages if page.page_type == "entity"]
    assert len(entity_pages) == 2
    assert all(len(page.trajectory_ids_json) <= 6 for page in entity_pages)
    assert {page.metadata_json["shard_count"] for page in entity_pages} == {2}
    assert sorted(page.metadata_json["shard_index"] for page in entity_pages) == [1, 2]
    assert all("routing_text" in page.metadata_json for page in pages)
    assert all(count <= 4 for count in seen_representative_counts)
    assert any("wiki_seed_split slug=entity-caroline" in line and "cluster_sizes=" in line for line in traces)
    assert any("wiki_page_compile_context slug=entity-caroline-" in line and "omitted=" in line for line in traces)
    assert any("wiki_page_compile_done page_id=" in line and "embed_ms=" in line for line in traces)


def test_wiki_compiler_suppresses_redundant_topic_seed(store: TrajPatchStore) -> None:
    sample_id = "sample-wiki-topic-suppression"
    for index in range(2):
        _add_trajectory(
            store,
            sample_id=sample_id,
            label=f"origin-{index}",
            metadata={
                "retrieval_summary_text": f"Caroline talked about Sweden and moving abroad {index}.",
                "retrieval_summary_keywords": ["caroline", "origin", "sweden"],
                "entity_mentions": ["Caroline"],
                "exact_terms": ["Sweden"],
                "facet_values": ["home_country=sweden"],
                "facet_tags": ["home_country"],
                "latest_keywords": ["origin", "sweden", "move"],
            },
        )

    compiler = WikiCompiler(
        store,
        MockLLMProvider(callback=lambda messages, system_prompt, metadata: "## Pages\n- malformed line without required fields" if (metadata or {}).get("task") == "wiki_page_plan" else (
            "## Overview\n- Page.\n\n## Key Facts\n- Facts.\n\n## Items / Counts\n- Items.\n\n## Linked Trajectories\n- Linked.\n\n## Conflicts / Uncertainty\n- None."
        )),
        HashEmbeddingProvider(),
    )
    pages = compiler.compile_sample(sample_id, "locomo")

    assert any(page.page_type == "entity" for page in pages)
    assert not any(page.page_type == "topic" and "origin" in page.slug for page in pages)


def test_route_pages_excludes_index_when_non_index_pages_exist(store: TrajPatchStore) -> None:
    sample_id = "sample-page-routing"
    provider = _KeywordPageEmbeddingProvider()

    index_page = WikiPageRecord(
        id="wiki-index",
        sample_id=sample_id,
        dataset_name="locomo",
        page_type="index",
        title="Index",
        slug="index",
        markdown_text="Index markdown",
        keywords_json=["adoption", "generic"],
        trajectory_ids_json=["traj-index"],
        linked_page_ids_json=[],
        entity_names_json=["Caroline"],
        embedding_id="wiki-index-emb",
        metadata_json={"routing_text": "generic index adoption page", "routing_priority": "low"},
    )
    topic_page = WikiPageRecord(
        id="wiki-topic",
        sample_id=sample_id,
        dataset_name="locomo",
        page_type="topic",
        title="Adoption Topic",
        slug="topic-adoption",
        markdown_text="Topic markdown",
        keywords_json=["adoption", "agencies"],
        trajectory_ids_json=["traj-topic"],
        linked_page_ids_json=[],
        entity_names_json=["Caroline"],
        embedding_id="wiki-topic-emb",
        metadata_json={"routing_text": "adoption agencies research topic", "routing_priority": "normal"},
    )
    store.save_wiki_page(index_page)
    store.save_wiki_page(topic_page)
    store.save_embedding(
        embedding_id="wiki-index-emb",
        owner_type="wiki_page",
        owner_id="wiki-index",
        model_name=provider.model_info().model_name,
        vector=provider.embed_documents(["generic index adoption page"])[0],
        semantic_text="generic index adoption page",
        metadata={"document_embedding_strategy": provider.document_embedding_strategy()},
    )
    store.save_embedding(
        embedding_id="wiki-topic-emb",
        owner_type="wiki_page",
        owner_id="wiki-topic",
        model_name=provider.model_info().model_name,
        vector=provider.embed_documents(["adoption agencies research topic"])[0],
        semantic_text="adoption agencies research topic",
        metadata={"document_embedding_strategy": provider.document_embedding_strategy()},
    )
    store.session.flush()

    traces: list[str] = []
    retrieval = RetrievalEngine(
        store,
        provider,
        top_t_pages=2,
        top_k=2,
        llm_provider=MockLLMProvider(
            callback=lambda messages, system_prompt, metadata: "SELECTED: P1\nRATIONALES:\n- P1: Keep focused page."
            if (metadata or {}).get("task") == "wiki_page_rerank"
            else MockLLMProvider()._default_response(messages, metadata or {})
        ),
        trace=traces.append,
    )

    selected_ids, trajectory_ids, metadata = retrieval._route_pages(
        sample_id,
        "What did Caroline research about adoption?",
        provider.embed_queries(["What did Caroline research about adoption?"])[0],
        {"adoption", "research"},
        ["Caroline"],
    )

    assert selected_ids == ["wiki-topic"]
    assert trajectory_ids == ["traj-topic"]
    assert metadata["page_candidate_ids"] == ["wiki-topic"]
    assert any("page_route_index_suppressed count=1" in line for line in traces)
    assert any("page_route_candidates considered_pages=1 candidate_pool=1" in line for line in traces)
    assert any("page_route_selected_ids pages=wiki-topic trajectory_union=1" in line for line in traces)


def test_route_pages_records_rerank_error_and_missing_embedding(store: TrajPatchStore) -> None:
    sample_id = "sample-page-routing-error"
    provider = _KeywordPageEmbeddingProvider()
    page = WikiPageRecord(
        id="wiki-topic-missing-embedding",
        sample_id=sample_id,
        dataset_name="locomo",
        page_type="topic",
        title="Adoption Topic",
        slug="topic-adoption",
        markdown_text="Topic markdown",
        keywords_json=["adoption", "agencies"],
        trajectory_ids_json=["traj-topic"],
        linked_page_ids_json=[],
        entity_names_json=["Caroline"],
        embedding_id="missing-embedding",
        metadata_json={"routing_text": "adoption agencies research topic"},
    )
    store.save_wiki_page(page)
    store.session.flush()

    def fail_rerank(messages, system_prompt, metadata):
        if (metadata or {}).get("task") == "wiki_page_rerank":
            raise RuntimeError("rerank provider unavailable")
        return MockLLMProvider()._default_response(messages, metadata or {})

    traces: list[str] = []
    retrieval = RetrievalEngine(
        store,
        provider,
        top_t_pages=1,
        top_k=1,
        llm_provider=MockLLMProvider(callback=fail_rerank),
        trace=traces.append,
    )

    selected_ids, trajectory_ids, metadata = retrieval._route_pages(
        sample_id,
        "What did Caroline research about adoption?",
        provider.embed_queries(["What did Caroline research about adoption?"])[0],
        {"adoption", "research"},
        ["Caroline"],
    )

    assert selected_ids == ["wiki-topic-missing-embedding"]
    assert trajectory_ids == ["traj-topic"]
    assert metadata["missing_page_embedding_count"] == 1
    assert metadata["all_page_embeddings_missing"] is True
    assert metadata["page_rerank_fallback"] is True
    assert metadata["page_rerank_error_type"] == "RuntimeError"
    assert "rerank provider unavailable" in metadata["page_rerank_error_message"]
    assert any("page_route_embedding_missing missing=1 total=1" in line for line in traces)
    assert any("page_rerank_failed error_type=RuntimeError" in line for line in traces)


def test_select_trajectories_records_rerank_error(store: TrajPatchStore) -> None:
    sample_id = "sample-trajectory-rerank-error"
    _add_trajectory(
        store,
        sample_id=sample_id,
        metadata={
            "retrieval_summary_text": "Caroline researched adoption agencies.",
            "retrieval_summary_keywords": ["caroline", "adoption", "agencies"],
            "entity_mentions": ["Caroline"],
            "exact_terms": ["adoption agencies"],
            "facet_values": ["research_topic=adoption agencies"],
            "facet_tags": ["research_topic"],
        },
    )
    trajectory = store.list_trajectories(sample_id)[0]

    def fail_rerank(messages, system_prompt, metadata):
        if (metadata or {}).get("task") == "trajectory_set_rerank":
            raise RuntimeError("trajectory rerank unavailable")
        return MockLLMProvider()._default_response(messages, metadata or {})

    traces: list[str] = []
    retrieval = RetrievalEngine(
        store,
        _KeywordPageEmbeddingProvider(),
        top_t_pages=1,
        top_k=1,
        llm_provider=MockLLMProvider(callback=fail_rerank),
        trace=traces.append,
    )

    selected_ids, metadata = retrieval._select_trajectories(
        sample_id,
        [trajectory.id],
        "What did Caroline research?",
        [1.0, 0.0],
        {"adoption", "research"},
        ["Caroline"],
        {"research_topic"},
        {"research_topic=adoption agencies"},
    )

    assert selected_ids == [trajectory.id]
    assert metadata["trajectory_rerank_fallback"] is True
    assert metadata["trajectory_rerank_error_type"] == "RuntimeError"
    assert "trajectory rerank unavailable" in metadata["trajectory_rerank_error_message"]
    assert any("trajectory_rerank_failed error_type=RuntimeError" in line for line in traces)
