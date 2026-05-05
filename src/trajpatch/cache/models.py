"""Typed cache payloads and manifests."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CacheManifestEntry(BaseModel):
    sample_id: str
    sample_ids: list[str] = Field(default_factory=list)
    sample_fingerprint: str
    history_fingerprint: str
    bundle_path: str
    created_at: str
    last_accessed_at: str | None = None
    hit_count: int = 0


class CacheManifest(BaseModel):
    schema_version: str
    build_fingerprint: str
    dataset_name: str
    adapter_version: str
    provider_kind: str
    backbone_model: str
    embedding_model: str
    created_at: str
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    sample_index: dict[str, CacheManifestEntry] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)


class MemoryCacheBundle(BaseModel):
    sample_meta: dict[str, object]
    trajectories: list[dict[str, object]] = Field(default_factory=list)
    episodic_snapshots: list[dict[str, object]] = Field(default_factory=list)
    wiki_pages: list[dict[str, object]] = Field(default_factory=list)
    claims: list[dict[str, object]] = Field(default_factory=list)
    claim_ops: list[dict[str, object]] = Field(default_factory=list)
    embeddings: list[dict[str, object]] = Field(default_factory=list)
    memory_stats: dict[str, object] = Field(default_factory=dict)
