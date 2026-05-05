"""SQLAlchemy ORM models for benchmark state, episodic memory, wiki pages, and reporting indexes."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class IndexBase(DeclarativeBase):
    """Declarative base for cross-run reporting index tables."""


class RawMessageRecord(Base):
    __tablename__ = "raw_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    sample_id: Mapped[str] = mapped_column(String, index=True)
    dataset_name: Mapped[str] = mapped_column(String, index=True)
    turn_index: Mapped[int] = mapped_column(Integer, index=True)
    role: Mapped[str] = mapped_column(String)
    speaker_name: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    source_ref: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    occurred_at: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrajectoryRecord(Base):
    __tablename__ = "trajectories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    sample_id: Mapped[str] = mapped_column(String, index=True)
    dataset_name: Mapped[str] = mapped_column(String, index=True)
    label: Mapped[str] = mapped_column(String)
    strict_matching: Mapped[bool] = mapped_column(Boolean, default=False)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    snapshot_count: Mapped[int] = mapped_column(Integer, default=0)
    max_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_snapshot_id: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EpisodicMemorySnapshot(Base):
    __tablename__ = "episodic_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    trajectory_id: Mapped[str] = mapped_column(ForeignKey("trajectories.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, index=True)
    timestamp: Mapped[str] = mapped_column(String)
    links_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary_content: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(Text)
    keywords_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    status_flags_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    embedding_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    semantic_text: Mapped[str] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ClaimRecord(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String, index=True)
    trajectory_id: Mapped[str] = mapped_column(String, index=True)
    claim_id: Mapped[str] = mapped_column(String, index=True)
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, index=True)
    source_message_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    parent_claim_id: Mapped[str | None] = mapped_column(String, nullable=True)
    revised_from_claim_id: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ClaimOpRecord(Base):
    __tablename__ = "claim_ops"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String, index=True)
    trajectory_id: Mapped[str] = mapped_column(String, index=True)
    op_type: Mapped[str] = mapped_column(String, index=True)
    target_claim_id: Mapped[str] = mapped_column(String, index=True)
    new_claim_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source_message_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    rationale: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WikiPageRecord(Base):
    __tablename__ = "wiki_pages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    sample_id: Mapped[str] = mapped_column(String, index=True)
    dataset_name: Mapped[str] = mapped_column(String, index=True)
    page_type: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    slug: Mapped[str] = mapped_column(String, index=True)
    markdown_text: Mapped[str] = mapped_column(Text)
    keywords_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    trajectory_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    linked_page_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    entity_names_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    embedding_id: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EmbeddingRecord(Base):
    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_type: Mapped[str] = mapped_column(String, index=True)
    owner_id: Mapped[str] = mapped_column(String, index=True)
    model_name: Mapped[str] = mapped_column(String)
    vector_json: Mapped[list[float]] = mapped_column(JSON)
    semantic_text: Mapped[str] = mapped_column(Text)
    norm: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RetrievalEvent(Base):
    __tablename__ = "retrieval_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    sample_id: Mapped[str] = mapped_column(String, index=True)
    query_text: Mapped[str] = mapped_column(Text)
    query_embedding_json: Mapped[list[float]] = mapped_column(JSON)
    top_t_pages: Mapped[int] = mapped_column(Integer, default=0)
    top_k: Mapped[int] = mapped_column(Integer)
    snapshot_budget: Mapped[int] = mapped_column(Integer, default=0)
    page_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    trajectory_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    snapshot_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    expanded_snapshot_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_message_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnswerRecord(Base):
    __tablename__ = "answers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    sample_id: Mapped[str] = mapped_column(String, index=True)
    query_task_id: Mapped[str] = mapped_column(String, index=True)
    retrieval_event_id: Mapped[str] = mapped_column(String, index=True)
    model_name: Mapped[str] = mapped_column(String)
    answer_text: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvaluationRecord(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    sample_id: Mapped[str] = mapped_column(String, index=True)
    query_task_id: Mapped[str] = mapped_column(String, index=True)
    answer_record_id: Mapped[str] = mapped_column(String, index=True)
    dataset_name: Mapped[str] = mapped_column(String)
    metric_name: Mapped[str] = mapped_column(String, index=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RunMetaRecord(Base):
    __tablename__ = "run_meta"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset: Mapped[str] = mapped_column(String, index=True)
    dataset_scope_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    run_dir: Mapped[str] = mapped_column(String)
    run_database_path: Mapped[str] = mapped_column(String)
    backbone_model: Mapped[str] = mapped_column(String, index=True)
    backbone_provider_kind: Mapped[str] = mapped_column(String, index=True)
    judge_model: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    judge_provider_kind: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    embedding_model: Mapped[str] = mapped_column(String, index=True)
    m: Mapped[int] = mapped_column(Integer)
    t_pages: Mapped[int] = mapped_column(Integer)
    k: Mapped[int] = mapped_column(Integer)
    neighbor_radius: Mapped[int] = mapped_column(Integer, default=1)
    retrieval_expansion_mode: Mapped[str] = mapped_column(String, default="update_linked_plus_neighbors")
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    processed_samples: Mapped[int] = mapped_column(Integer, default=0)
    processed_queries: Mapped[int] = mapped_column(Integer, default=0)
    excluded_count: Mapped[int] = mapped_column(Integer, default=0)
    total_runtime_s: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AggregateMetricRecord(Base):
    __tablename__ = "aggregate_metrics"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    dataset: Mapped[str] = mapped_column(String, index=True)
    group_level: Mapped[str] = mapped_column(String, index=True)
    subset_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    scene_tag: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    metric_name: Mapped[str] = mapped_column(String, index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    runtime_ms: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IndexedRunRecord(IndexBase):
    __tablename__ = "run_registry"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset: Mapped[str] = mapped_column(String, index=True)
    dataset_scope_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    run_dir: Mapped[str] = mapped_column(String)
    run_database_path: Mapped[str] = mapped_column(String)
    backbone_model: Mapped[str] = mapped_column(String, index=True)
    backbone_provider_kind: Mapped[str] = mapped_column(String, index=True)
    judge_model: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    judge_provider_kind: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    embedding_model: Mapped[str] = mapped_column(String, index=True)
    m: Mapped[int] = mapped_column(Integer)
    t_pages: Mapped[int] = mapped_column(Integer)
    k: Mapped[int] = mapped_column(Integer)
    neighbor_radius: Mapped[int] = mapped_column(Integer, default=1)
    retrieval_expansion_mode: Mapped[str] = mapped_column(String, default="update_linked_plus_neighbors")
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    processed_samples: Mapped[int] = mapped_column(Integer, default=0)
    processed_queries: Mapped[int] = mapped_column(Integer, default=0)
    excluded_count: Mapped[int] = mapped_column(Integer, default=0)
    total_runtime_s: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IndexedAggregateMetricRecord(IndexBase):
    __tablename__ = "aggregate_metric_registry"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    dataset: Mapped[str] = mapped_column(String, index=True)
    group_level: Mapped[str] = mapped_column(String, index=True)
    subset_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    scene_tag: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    metric_name: Mapped[str] = mapped_column(String, index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    runtime_ms: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
