from __future__ import annotations

import json

import pytest

from trajpatch.cache.manager import MemoryCacheManager
from trajpatch.cache.models import MemoryCacheBundle
from trajpatch.ids import snapshot_id, trajectory_id, wiki_page_id
from trajpatch.types import DatasetSample


def test_memory_cache_remap_rewrites_wiki_embeddings_and_metadata_refs() -> None:
    old_sample_id = "sample-a"
    new_sample_id = "sample-b"
    old_trajectory_id = trajectory_id(old_sample_id, 1)
    old_snapshot_id = snapshot_id(old_trajectory_id, 1)
    old_claim_id = f"{old_trajectory_id}-c001"
    old_page_id = wiki_page_id(old_sample_id, "entity", 2)
    old_linked_page_id = wiki_page_id(old_sample_id, "topic", 3)
    old_snapshot_embedding_id = f"{old_snapshot_id}-emb"
    old_summary_embedding_id = f"{old_trajectory_id}-summary"
    old_page_embedding_id = f"{old_page_id}-emb"

    bundle = MemoryCacheBundle(
        sample_meta={
            "sample_id": old_sample_id,
            "dataset_name": "medmt",
            "history_fingerprint": "same-history",
            "build_fingerprint": "same-build",
        },
        trajectories=[
            {
                "id": old_trajectory_id,
                "sample_id": old_sample_id,
                "dataset_name": "medmt",
                "label": "profile",
                "strict_matching": False,
                "is_open": False,
                "snapshot_count": 1,
                "max_length": 6,
                "latest_snapshot_id": old_snapshot_id,
                "metadata_json": {
                    "latest_snapshot_id": old_snapshot_id,
                    "latest_snapshot_embedding_id": old_snapshot_embedding_id,
                    "retrieval_summary_embedding_id": old_summary_embedding_id,
                    "nested": {
                        "snapshot": old_snapshot_id,
                        "source": f"{old_sample_id}-m0001",
                    },
                },
            }
        ],
        episodic_snapshots=[
            {
                "id": old_snapshot_id,
                "trajectory_id": old_trajectory_id,
                "version": 1,
                "timestamp": "2026-04-06T12:00:00Z",
                "links_json": [f"{old_sample_id}-m0001"],
                "summary_content": f"The user reports a smoking habit in {old_snapshot_id}.",
                "context": f"Health disclosure from {old_sample_id}-m0001.",
                "keywords_json": ["smoking"],
                "status_flags_json": ["active"],
                "embedding_ref": old_snapshot_embedding_id,
                "semantic_text": f"User smokes 5 cigarettes a day. Source {old_sample_id}-m0001.",
                "raw_text": f"raw source {old_sample_id}-m0001",
                "metadata_json": {"linked_source": f"{old_sample_id}-m0001"},
            }
        ],
        wiki_pages=[
            {
                "id": old_page_id,
                "sample_id": old_sample_id,
                "dataset_name": "medmt",
                "page_type": "entity",
                "title": "User",
                "slug": "user",
                "markdown_text": (
                    "## Overview\n"
                    f"User smoking history linked to {old_trajectory_id}, {old_snapshot_id}, "
                    f"and {old_sample_id}-m0001."
                ),
                "keywords_json": ["user", "smoking"],
                "trajectory_ids_json": [old_trajectory_id],
                "linked_page_ids_json": [old_linked_page_id],
                "entity_names_json": ["User"],
                "embedding_id": old_page_embedding_id,
                "metadata_json": {
                    "representative_trajectory_ids": [old_trajectory_id],
                    "linked_page_ids": [old_linked_page_id],
                    "snapshot_id": old_snapshot_id,
                    "embedding_id": old_page_embedding_id,
                    "source_message_id": f"{old_sample_id}-m0001",
                    "routing_text": (
                        f"Linked trajectory {old_trajectory_id}; snapshot {old_snapshot_id}; "
                        f"page {old_linked_page_id}; source {old_sample_id}-m0001."
                    ),
                },
            },
            {
                "id": old_linked_page_id,
                "sample_id": old_sample_id,
                "dataset_name": "medmt",
                "page_type": "topic",
                "title": "Smoking",
                "slug": "smoking",
                "markdown_text": f"## Overview\nSmoking facts from {old_trajectory_id}.",
                "keywords_json": ["smoking"],
                "trajectory_ids_json": [old_trajectory_id],
                "linked_page_ids_json": [old_page_id],
                "entity_names_json": ["User"],
                "embedding_id": f"{old_linked_page_id}-emb",
                "metadata_json": {},
            },
        ],
        claims=[
            {
                "id": f"{old_claim_id}-row",
                "snapshot_id": old_snapshot_id,
                "trajectory_id": old_trajectory_id,
                "claim_id": old_claim_id,
                "text": "User smokes 5 cigarettes a day.",
                "status": "active",
                "source_message_ids_json": [f"{old_sample_id}-m0001"],
                "parent_claim_id": None,
                "revised_from_claim_id": None,
                "metadata_json": {"source": f"{old_sample_id}-m0001"},
            }
        ],
        claim_ops=[
            {
                "id": f"{old_trajectory_id}-op001",
                "snapshot_id": old_snapshot_id,
                "trajectory_id": old_trajectory_id,
                "op_type": "ADD",
                "target_claim_id": old_claim_id,
                "new_claim_id": old_claim_id,
                "source_message_ids_json": [f"{old_sample_id}-m0001"],
                "rationale": "Initial extraction.",
                "metadata_json": {"claim": old_claim_id},
            }
        ],
        embeddings=[
            {
                "id": old_snapshot_embedding_id,
                "owner_type": "snapshot",
                "owner_id": old_snapshot_id,
                "model_name": "hash",
                "vector_json": [1.0, 0.0],
                "semantic_text": f"User smokes 5 cigarettes a day. Source {old_sample_id}-m0001.",
                "norm": 1.0,
                "metadata_json": {},
            },
            {
                "id": old_summary_embedding_id,
                "owner_type": "trajectory_summary",
                "owner_id": old_trajectory_id,
                "model_name": "hash",
                "vector_json": [0.0, 1.0],
                "semantic_text": f"Smoking summary for {old_trajectory_id}.",
                "norm": 1.0,
                "metadata_json": {},
            },
            {
                "id": old_page_embedding_id,
                "owner_type": "wiki_page",
                "owner_id": old_page_id,
                "model_name": "hash",
                "vector_json": [0.5, 0.5],
                "semantic_text": (
                    f"User smoking page for {old_page_id}; trajectory {old_trajectory_id}; "
                    f"source {old_sample_id}-m0001."
                ),
                "norm": 1.0,
                "metadata_json": {},
            },
            {
                "id": f"{old_linked_page_id}-emb",
                "owner_type": "wiki_page",
                "owner_id": old_linked_page_id,
                "model_name": "hash",
                "vector_json": [0.25, 0.75],
                "semantic_text": f"Smoking topic page linked to {old_page_id}.",
                "norm": 1.0,
                "metadata_json": {},
            },
        ],
    )
    sample = DatasetSample(sample_id=new_sample_id, dataset_name="medmt", payload={})
    manager = object.__new__(MemoryCacheManager)

    remapped = manager._remap_bundle_for_sample(bundle, sample)

    new_trajectory_id = trajectory_id(new_sample_id, 1)
    new_snapshot_id = snapshot_id(new_trajectory_id, 1)
    new_page_id = wiki_page_id(new_sample_id, "entity", 2)
    new_linked_page_id = wiki_page_id(new_sample_id, "topic", 3)
    trajectory = remapped.trajectories[0]
    assert trajectory["id"] == new_trajectory_id
    assert trajectory["latest_snapshot_id"] == new_snapshot_id
    assert trajectory["metadata_json"]["latest_snapshot_id"] == new_snapshot_id
    assert trajectory["metadata_json"]["latest_snapshot_embedding_id"] == f"{new_snapshot_id}-emb"
    assert trajectory["metadata_json"]["retrieval_summary_embedding_id"] == f"{new_trajectory_id}-summary"
    assert trajectory["metadata_json"]["nested"]["source"] == f"{new_sample_id}-m0001"

    page = next(row for row in remapped.wiki_pages if row["page_type"] == "entity")
    assert page["id"] == new_page_id
    assert page["trajectory_ids_json"] == [new_trajectory_id]
    assert page["linked_page_ids_json"] == [new_linked_page_id]
    assert page["embedding_id"] == f"{new_page_id}-emb"
    assert page["metadata_json"]["representative_trajectory_ids"] == [new_trajectory_id]
    assert page["metadata_json"]["linked_page_ids"] == [new_linked_page_id]
    assert page["metadata_json"]["snapshot_id"] == new_snapshot_id
    assert page["metadata_json"]["embedding_id"] == f"{new_page_id}-emb"
    assert page["metadata_json"]["source_message_id"] == f"{new_sample_id}-m0001"
    assert new_trajectory_id in page["metadata_json"]["routing_text"]
    assert new_snapshot_id in page["metadata_json"]["routing_text"]
    assert new_linked_page_id in page["metadata_json"]["routing_text"]
    assert f"{new_sample_id}-m0001" in page["metadata_json"]["routing_text"]
    assert old_sample_id not in page["metadata_json"]["routing_text"]
    assert old_trajectory_id not in page["metadata_json"]["routing_text"]
    assert new_trajectory_id in page["markdown_text"]
    assert new_snapshot_id in page["markdown_text"]
    assert f"{new_sample_id}-m0001" in page["markdown_text"]
    assert old_sample_id not in page["markdown_text"]

    wiki_embeddings = [row for row in remapped.embeddings if row["owner_type"] == "wiki_page"]
    assert {row["owner_id"] for row in wiki_embeddings} == {new_page_id, new_linked_page_id}
    assert {row["id"] for row in wiki_embeddings} == {f"{new_page_id}-emb", f"{new_linked_page_id}-emb"}
    assert all(old_sample_id not in row["semantic_text"] for row in wiki_embeddings)
    assert any(new_trajectory_id in row["semantic_text"] for row in wiki_embeddings)
    assert any(new_page_id in row["semantic_text"] for row in wiki_embeddings)

    critical_payload = {
        "trajectories": remapped.trajectories,
        "snapshots": remapped.episodic_snapshots,
        "wiki_pages": remapped.wiki_pages,
        "claims": remapped.claims,
        "claim_ops": remapped.claim_ops,
        "embeddings": remapped.embeddings,
    }
    assert old_sample_id not in json.dumps(critical_payload, sort_keys=True)


def test_memory_cache_remap_rejects_unknown_embedding_owner_type() -> None:
    bundle = MemoryCacheBundle(
        sample_meta={
            "sample_id": "sample-a",
            "dataset_name": "medmt",
            "history_fingerprint": "same-history",
            "build_fingerprint": "same-build",
        },
        embeddings=[
            {
                "id": "unknown-emb",
                "owner_type": "future_owner",
                "owner_id": "future-owner-id",
                "model_name": "hash",
                "vector_json": [1.0],
                "semantic_text": "future",
                "norm": 1.0,
                "metadata_json": {},
            }
        ],
    )
    manager = object.__new__(MemoryCacheManager)
    sample = DatasetSample(sample_id="sample-b", dataset_name="medmt", payload={})

    with pytest.raises(ValueError, match="Unsupported cached embedding owner_type"):
        manager._remap_bundle_for_sample(bundle, sample)
