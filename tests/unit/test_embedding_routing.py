from __future__ import annotations

from trajpatch.memory.orchestrator import MemoryOrchestrator, ParsedMemory
from trajpatch.memory.schemas import EpisodicMemoryInput, MemoryClaim
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.providers.structured_outputs import TrajectoryMatchResult
from trajpatch.providers.transformers_provider import SentenceTransformerEmbeddingProvider
from trajpatch.storage.models import ClaimRecord, EpisodicMemorySnapshot
from trajpatch.types import ModelInfo, NormalizedMessage, StructuredLLMResponse
from trajpatch.utils.text import extract_keywords


class _FakeSentenceTransformerModel:
    def __init__(self, *, fail_on_prompt_name: bool = False) -> None:
        self.fail_on_prompt_name = fail_on_prompt_name
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts, **kwargs):
        normalized = list(texts)
        self.calls.append((normalized, dict(kwargs)))
        if self.fail_on_prompt_name and kwargs.get("prompt_name") == "query":
            raise TypeError("prompt_name unsupported")
        return [[1.0, 0.0] for _ in normalized]


class _DtypeMismatchSentenceTransformerModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def encode(self, texts, **kwargs):
        normalized = list(texts)
        self.calls.append((normalized, dict(kwargs)))
        if len(self.calls) == 1:
            raise RuntimeError("expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16")
        return [[1.0, 0.0] for _ in normalized]

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class TrackingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(model_name="tracking-embedding")
        self.query_calls: list[list[str]] = []
        self.document_calls: list[list[str]] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_calls.append(list(texts))
        return super().embed(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return super().embed(texts)

    def query_embedding_strategy(self) -> str:
        return "tracking-query"

    def document_embedding_strategy(self) -> str:
        return "tracking-document"

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="mock", model_name="tracking-embedding", is_remote=False)


class KeywordEmbeddingProvider(TrackingEmbeddingProvider):
    def _vector_for_text(self, text: str) -> list[float]:
        lowered = text.casefold()
        if any(token in lowered for token in ("adoption", "agency", "family plan")):
            return [1.0, 0.0]
        if any(token in lowered for token in ("necklace", "jewelry", "gemstone")):
            return [0.0, 1.0]
        if any(token in lowered for token in ("violin", "clarinet", "orchestra")):
            return [-1.0, 0.0]
        return [0.0, -1.0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_calls.append(list(texts))
        return [self._vector_for_text(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector_for_text(text) for text in texts]


class StructuredTrajectoryMatchProvider(MockLLMProvider):
    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="structured-trajectory-match", is_remote=True)

    def supports_structured(self, task: str) -> bool:
        return task == "trajectory_match"

    def generate_structured(
        self,
        messages,
        *,
        spec,
        system_prompt=None,
        metadata=None,
    ) -> StructuredLLMResponse:
        assert spec.task == "trajectory_match"
        prompt = messages[-1].content if messages else ""
        return StructuredLLMResponse(
            parsed=TrajectoryMatchResult(
                decision="CONTINUE",
                selected_candidate="T1",
                rationale="The trajectory summary matches the same adoption-planning thread.",
            ),
            prompt_tokens=len(prompt.split()),
            completion_tokens=1,
            metadata={"structured_vendor": "openai", "structured_success": True},
        )


def _seed_open_trajectory(
    store,
    embedding_provider: KeywordEmbeddingProvider,
    *,
    sample_id: str,
    label: str,
    latest_text: str,
    summary_text: str,
    exact_terms: list[str] | None = None,
    facet_tags: list[str] | None = None,
    facet_values: list[str] | None = None,
    entity_mentions: list[str] | None = None,
):
    trajectory = store.create_trajectory(
        sample_id=sample_id,
        dataset_name="locomo",
        label=label,
        strict_matching=False,
        max_length=6,
        metadata={},
    )
    snapshot = EpisodicMemorySnapshot(
        id=f"{trajectory.id}-v001",
        trajectory_id=trajectory.id,
        version=1,
        timestamp="2026-04-20T10:00:00Z",
        links_json=[],
        summary_content=latest_text,
        context=latest_text,
        keywords_json=list(extract_keywords(latest_text)),
        status_flags_json=["active"],
        embedding_ref=None,
        semantic_text=latest_text,
        raw_text=latest_text,
    )
    store.save_episodic_snapshot(snapshot)
    latest_embedding = store.save_embedding(
        embedding_id=f"{snapshot.id}-emb",
        owner_type="snapshot",
        owner_id=snapshot.id,
        model_name=embedding_provider.model_info().model_name,
        vector=embedding_provider.embed_documents([latest_text])[0],
        semantic_text=latest_text,
        metadata={"document_embedding_strategy": embedding_provider.document_embedding_strategy()},
    )
    snapshot.embedding_ref = latest_embedding.id
    store.replace_claims_for_snapshot(
        [
            ClaimRecord(
                id=f"{trajectory.id}-c001",
                claim_id=f"{trajectory.id}-c001",
                snapshot_id=snapshot.id,
                trajectory_id=trajectory.id,
                status="active",
                text=latest_text,
                source_message_ids_json=[],
                metadata_json={"exact_terms_v1": list(exact_terms or []), "facets_v1": []},
                parent_claim_id=None,
                revised_from_claim_id=None,
            )
        ]
    )
    summary_embedding = store.save_embedding(
        embedding_id=f"{trajectory.id}-summary",
        owner_type="trajectory_summary",
        owner_id=trajectory.id,
        model_name=embedding_provider.model_info().model_name,
        vector=embedding_provider.embed_documents([summary_text])[0],
        semantic_text=summary_text,
        metadata={"document_embedding_strategy": embedding_provider.document_embedding_strategy()},
    )
    trajectory.metadata_json = {
        "latest_snapshot_id": snapshot.id,
        "latest_snapshot_embedding_id": latest_embedding.id,
        "latest_semantic_text": latest_text,
        "latest_keywords": list(extract_keywords(latest_text)),
        "retrieval_summary_text": summary_text,
        "retrieval_summary_keywords": list(extract_keywords(summary_text)),
        "retrieval_summary_embedding_id": summary_embedding.id,
        "exact_terms": list(exact_terms or []),
        "facet_tags": list(facet_tags or []),
        "facet_values": list(facet_values or []),
        "entity_mentions": list(entity_mentions or []),
    }
    store.session.flush()
    return trajectory


def test_sentence_transformer_qwen_query_path_uses_prompt_name_when_available():
    provider = SentenceTransformerEmbeddingProvider("Qwen/Qwen3-Embedding-0.6B")
    fake_model = _FakeSentenceTransformerModel()
    provider._model = fake_model

    vectors = provider.embed_queries(["What did Caroline research?"])

    assert vectors == [[1.0, 0.0]]
    assert fake_model.calls[0][1]["prompt_name"] == "query"
    assert provider.query_embedding_strategy() == "qwen_prompt_name_query"


def test_sentence_transformer_qwen_query_path_falls_back_to_prefixed_prompt():
    provider = SentenceTransformerEmbeddingProvider("Qwen/Qwen3-Embedding-0.6B")
    fake_model = _FakeSentenceTransformerModel(fail_on_prompt_name=True)
    provider._model = fake_model

    vectors = provider.embed_queries(["What did Caroline research?"])

    assert vectors == [[1.0, 0.0]]
    assert fake_model.calls[0][1]["prompt_name"] == "query"
    assert fake_model.calls[1][0] == [
        "Instruct: Retrieve memories that directly answer the user's question.\nQuery: What did Caroline research?"
    ]
    assert provider.query_embedding_strategy() == "qwen_prefixed_query_fallback"


def test_sentence_transformer_document_encode_retries_float32_on_dtype_mismatch():
    provider = SentenceTransformerEmbeddingProvider("Qwen/Qwen3-Embedding-0.6B")
    fake_model = _DtypeMismatchSentenceTransformerModel()
    provider._model = fake_model

    vectors = provider.embed_documents(["Caroline researched adoption agencies."])

    assert vectors == [[1.0, 0.0]]
    assert len(fake_model.calls) == 2
    assert str(fake_model.to_calls[0][1]["dtype"]) in {"torch.float32", "float32"}
    assert provider.document_embedding_strategy() == "plain_encode_float32_retry"
    assert provider.model_info().metadata["float32_dtype_retry_count"] == 1


def test_sentence_transformer_qwen_query_retries_float32_on_dtype_mismatch():
    provider = SentenceTransformerEmbeddingProvider("Qwen/Qwen3-Embedding-0.6B")
    fake_model = _DtypeMismatchSentenceTransformerModel()
    provider._model = fake_model

    vectors = provider.embed_queries(["What did Caroline research?"])

    assert vectors == [[1.0, 0.0]]
    assert fake_model.calls[0][1]["prompt_name"] == "query"
    assert str(fake_model.to_calls[0][1]["dtype"]) in {"torch.float32", "float32"}
    assert provider.query_embedding_strategy() == "qwen_prompt_name_query_float32_retry"


def test_orchestrator_routes_query_and_document_embeddings_and_persists_facet_metadata(run_config, store):
    embedding_provider = TrackingEmbeddingProvider()
    provider = MockLLMProvider(
        callback=lambda messages, system_prompt, metadata: (
            "DECISION: CONTINUE\n"
            "SELECTED_CANDIDATE: T1\n"
            "RATIONALE: The trajectory summary matches the same adoption-planning thread."
            if (metadata or {}).get("task") == "trajectory_match"
            else (
                "[EXACT_TERMS]\n"
                "- surface=adoption agencies | category=research_topic | source_claim_id=epi-sample-001-c001 | source_message_ids=sample-m0000\n\n"
                "[FACETS]\n"
                "- relation=research_topic | value=adoption agencies | entity=Caroline | value_span=adoption agencies | source_claim_id=epi-sample-001-c001 | source_message_ids=sample-m0000\n\n"
                "[DISPLAY_ITEMS]\n"
                "- adoption agencies\n\n"
                "[DISPLAY_NAMED_ENTITIES]\n"
                "- Caroline\n\n"
                "[DISPLAY_COUNTS]\n"
                "- none\n\n"
                "[DISPLAY_KEY_FACTS]\n"
                "- Caroline is researching adoption agencies."
            )
            if (metadata or {}).get("task") == "claim_signal_extract"
            else MockLLMProvider()._default_response(messages, metadata or {})
        )
    )
    orchestrator = MemoryOrchestrator(run_config, store, provider, embedding_provider)
    for turn_index, (speaker_name, content) in enumerate(
        [
            ("Caroline", "I am researching adoption agencies."),
            ("Caroline", "I hope it works out."),
        ]
    ):
        store.add_raw_message(
            "sample",
            "locomo",
            NormalizedMessage(
                role="user" if turn_index == 0 else "assistant",
                content=content,
                turn_index=turn_index,
                speaker_name=speaker_name,
                raw_message_id=f"sample-m{turn_index:04d}",
            ),
        )

    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text="Caroline is researching adoption agencies.",
        links=["sample-m0000", "sample-m0001"],
        claims=[
            MemoryClaim(
                claim_id="tmp-c1",
                status="active",
                source_message_ids=["sample-m0000"],
                text="Caroline is researching adoption agencies.",
            )
        ],
        raw=EpisodicMemoryInput(
            memory_type="episodic",
            timestamp="2026-04-18T10:00:00Z",
            summary_content="Caroline is researching adoption agencies.",
            context="Caroline discussed adoption planning.",
            keywords=["caroline", "adoption"],
            links=["sample-m0000", "sample-m0001"],
            status_flags=["active"],
            claims=[
                MemoryClaim(
                    claim_id="tmp-c1",
                    status="active",
                    source_message_ids=["sample-m0000"],
                    text="Caroline is researching adoption agencies.",
                )
            ],
            ops=[],
            raw_text="raw",
        ),
        metadata={},
    )

    orchestrator.persist_memory("sample", "locomo", parsed)

    assert embedding_provider.document_calls[0] == ["Caroline is researching adoption agencies."]
    assert len(embedding_provider.document_calls) >= 2
    trajectory = store.list_trajectories("sample")[0]
    claims = store.latest_claims(trajectory.id)
    assert claims[0].metadata_json["facets_v2"][0]["relation"] == "research_topic"
    assert "facets_v1" not in claims[0].metadata_json
    assert "exact_terms_v1" not in claims[0].metadata_json
    assert trajectory.metadata_json["entity_mentions"] == ["Caroline"]
    assert trajectory.metadata_json["facet_tags"] == ["research_topic"]
    assert trajectory.metadata_json["facet_values"] == ["research_topic=adoption agencies"]
    assert "adoption agencies" in trajectory.metadata_json["trajectory_historical_item_terms_v1"]
    assert trajectory.metadata_json["trajectory_drift_cluster_count_v1"] >= 1
    assert trajectory.metadata_json["trajectory_historical_evidence_card_v1"]["historical_item_terms"]
    assert "## Profile / Stable Facts" in trajectory.metadata_json["retrieval_summary_text"]
    assert trajectory.metadata_json["retrieval_summary_embedding_id"] == f"{trajectory.id}-summary"
    snapshot_embedding = store.snapshot_embedding(trajectory.latest_snapshot_id)
    assert snapshot_embedding is not None
    assert snapshot_embedding.metadata_json["document_embedding_strategy"] == "tracking-document"
    summary_embedding = store.fetch_embedding(trajectory.id, "trajectory_summary")
    assert summary_embedding is not None
    assert summary_embedding.metadata_json["document_embedding_strategy"] == "tracking-document"

    matched = orchestrator.match_trajectory("sample", parsed)

    assert matched == trajectory.id
    assert embedding_provider.query_calls == [["Caroline is researching adoption agencies."]]
    assert parsed.metadata["trajectory_match_scored_candidates_v1"][0]["continuity_bonus"] > 0
    assert trajectory.metadata_json["latest_snapshot_id"] == trajectory.latest_snapshot_id
    assert trajectory.metadata_json["latest_snapshot_embedding_id"] == snapshot_embedding.id


def test_match_trajectory_prefers_summary_when_latest_snapshot_is_misleading(run_config, store):
    embedding_provider = KeywordEmbeddingProvider()
    provider = MockLLMProvider(
        callback=lambda messages, system_prompt, metadata: (
            "DECISION: CONTINUE\n"
            "SELECTED_CANDIDATE: T1\n"
            "RATIONALE: The trajectory summary matches the same adoption-planning thread."
            if (metadata or {}).get("task") == "trajectory_match"
            else MockLLMProvider()._default_response(messages, metadata or {})
        )
    )
    orchestrator = MemoryOrchestrator(run_config, store, provider, embedding_provider)
    for turn_index, content in enumerate(
        [
            "Caroline is researching adoption agencies.",
            "Caroline mentioned her grandmother's necklace.",
        ]
    ):
        store.add_raw_message(
            "sample",
            "locomo",
            NormalizedMessage(
                role="user",
                content=content,
                turn_index=turn_index,
                speaker_name="Caroline",
                raw_message_id=f"sample-m{turn_index:04d}",
            ),
        )

    target = _seed_open_trajectory(
        store,
        embedding_provider,
        sample_id="sample",
        label="adoption-thread",
        latest_text="Caroline talked about a necklace from her grandmother.",
        summary_text="Caroline is researching adoption agencies and family planning.",
        exact_terms=["adoption agencies"],
        facet_tags=["research_topic"],
        facet_values=["research_topic=adoption agencies"],
        entity_mentions=["Caroline"],
    )
    distractor = _seed_open_trajectory(
        store,
        embedding_provider,
        sample_id="sample",
        label="music-thread",
        latest_text="Caroline is researching adoption agencies right now.",
        summary_text="Caroline practices violin and clarinet for orchestra rehearsals.",
        exact_terms=["violin", "clarinet"],
        entity_mentions=["Caroline"],
    )

    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text="Caroline is researching adoption agencies.",
        links=["sample-m0000", "sample-m0001"],
        claims=[
            MemoryClaim(
                claim_id="tmp-c1",
                status="active",
                source_message_ids=["sample-m0000"],
                text="Caroline is researching adoption agencies.",
            )
        ],
        raw=EpisodicMemoryInput(
            memory_type="episodic",
            timestamp="2026-04-20T11:00:00Z",
            summary_content="Caroline is researching adoption agencies.",
            context="Caroline talked about adoption planning.",
            keywords=["caroline", "adoption"],
            links=["sample-m0000", "sample-m0001"],
            status_flags=["active"],
            claims=[
                MemoryClaim(
                    claim_id="tmp-c1",
                    status="active",
                    source_message_ids=["sample-m0000"],
                    text="Caroline is researching adoption agencies.",
                )
            ],
            ops=[],
            raw_text="raw",
        ),
        metadata={},
    )

    matched = orchestrator.match_trajectory("sample", parsed)

    assert matched == target.id
    assert parsed.metadata["trajectory_match_candidate_label_map"][0]["resolved_id"] == target.id
    assert parsed.metadata["trajectory_match_candidate_label_map"][1]["resolved_id"] == distractor.id


def test_match_trajectory_malformed_text_decision_falls_back_to_new(run_config, store):
    embedding_provider = KeywordEmbeddingProvider()
    provider = MockLLMProvider(
        callback=lambda messages, system_prompt, metadata: (
            "<think>\nThe model reasoned but did not emit the required DSL."
            if (metadata or {}).get("task") == "trajectory_match"
            else MockLLMProvider()._default_response(messages, metadata or {})
        )
    )
    orchestrator = MemoryOrchestrator(run_config, store, provider, embedding_provider)
    store.add_raw_message(
        "sample",
        "locomo",
        NormalizedMessage(
            role="user",
            content="Caroline is researching adoption agencies.",
            turn_index=0,
            speaker_name="Caroline",
            raw_message_id="sample-m0000",
        ),
    )
    _seed_open_trajectory(
        store,
        embedding_provider,
        sample_id="sample",
        label="adoption-thread",
        latest_text="Caroline previously researched adoption agencies.",
        summary_text="Caroline is researching adoption agencies and family planning.",
        exact_terms=["adoption agencies"],
        entity_mentions=["Caroline"],
    )
    parsed = ParsedMemory(
        memory_type="episodic",
        semantic_text="Caroline is researching adoption agencies.",
        links=["sample-m0000"],
        claims=[
            MemoryClaim(
                claim_id="tmp-c1",
                status="active",
                source_message_ids=["sample-m0000"],
                text="Caroline is researching adoption agencies.",
            )
        ],
        raw=EpisodicMemoryInput(
            memory_type="episodic",
            timestamp="2026-04-20T11:00:00Z",
            summary_content="Caroline is researching adoption agencies.",
            context="Caroline talked about adoption planning.",
            keywords=["caroline", "adoption"],
            links=["sample-m0000"],
            status_flags=["active"],
            claims=[
                MemoryClaim(
                    claim_id="tmp-c1",
                    status="active",
                    source_message_ids=["sample-m0000"],
                    text="Caroline is researching adoption agencies.",
                )
            ],
            ops=[],
            raw_text="raw",
        ),
        metadata={},
    )

    matched = orchestrator.match_trajectory("sample", parsed)

    assert matched is None
    assert parsed.metadata["trajectory_match_selected_candidate_resolved_id"] is None
    assert parsed.metadata["trajectory_match_text_parse_failed"] is True
    assert parsed.metadata["trajectory_match_text_parse_fallback"] == "new"


def test_match_trajectory_structured_and_text_paths_agree_on_summary_primary_selection(run_config, store):
    embedding_provider = KeywordEmbeddingProvider()
    for turn_index, content in enumerate(
        [
            "Caroline is researching adoption agencies.",
            "Caroline mentioned her grandmother's necklace.",
        ]
    ):
        store.add_raw_message(
            "sample",
            "locomo",
            NormalizedMessage(
                role="user",
                content=content,
                turn_index=turn_index,
                speaker_name="Caroline",
                raw_message_id=f"sample-m{turn_index:04d}",
            ),
        )
    target = _seed_open_trajectory(
        store,
        embedding_provider,
        sample_id="sample",
        label="adoption-thread",
        latest_text="Caroline talked about a necklace from her grandmother.",
        summary_text="Caroline is researching adoption agencies and family planning.",
        exact_terms=["adoption agencies"],
        facet_tags=["research_topic"],
        facet_values=["research_topic=adoption agencies"],
        entity_mentions=["Caroline"],
    )
    _seed_open_trajectory(
        store,
        embedding_provider,
        sample_id="sample",
        label="music-thread",
        latest_text="Caroline is researching adoption agencies right now.",
        summary_text="Caroline practices violin and clarinet for orchestra rehearsals.",
        exact_terms=["violin", "clarinet"],
        entity_mentions=["Caroline"],
    )

    parsed_text = ParsedMemory(
        memory_type="episodic",
        semantic_text="Caroline is researching adoption agencies.",
        links=["sample-m0000", "sample-m0001"],
        claims=[
            MemoryClaim(
                claim_id="tmp-c1",
                status="active",
                source_message_ids=["sample-m0000"],
                text="Caroline is researching adoption agencies.",
            )
        ],
        raw=EpisodicMemoryInput(
            memory_type="episodic",
            timestamp="2026-04-20T11:00:00Z",
            summary_content="Caroline is researching adoption agencies.",
            context="Caroline talked about adoption planning.",
            keywords=["caroline", "adoption"],
            links=["sample-m0000", "sample-m0001"],
            status_flags=["active"],
            claims=[
                MemoryClaim(
                    claim_id="tmp-c1",
                    status="active",
                    source_message_ids=["sample-m0000"],
                    text="Caroline is researching adoption agencies.",
                )
            ],
            ops=[],
            raw_text="raw",
        ),
        metadata={},
    )
    parsed_structured = ParsedMemory(
        memory_type="episodic",
        semantic_text=parsed_text.semantic_text,
        links=list(parsed_text.links),
        claims=list(parsed_text.claims),
        raw=parsed_text.raw,
        metadata={},
    )

    text_provider = MockLLMProvider(
        callback=lambda messages, system_prompt, metadata: (
            "DECISION: CONTINUE\n"
            "SELECTED_CANDIDATE: T1\n"
            "RATIONALE: The trajectory summary matches the same adoption-planning thread."
            if (metadata or {}).get("task") == "trajectory_match"
            else MockLLMProvider()._default_response(messages, metadata or {})
        )
    )
    structured_provider = StructuredTrajectoryMatchProvider()

    text_orchestrator = MemoryOrchestrator(run_config, store, text_provider, embedding_provider)
    structured_orchestrator = MemoryOrchestrator(run_config, store, structured_provider, embedding_provider)

    matched_text = text_orchestrator.match_trajectory("sample", parsed_text)
    matched_structured = structured_orchestrator.match_trajectory("sample", parsed_structured)

    assert matched_text == target.id
    assert matched_structured == target.id
    assert parsed_text.metadata["trajectory_match_candidate_label_map"] == parsed_structured.metadata["trajectory_match_candidate_label_map"]
