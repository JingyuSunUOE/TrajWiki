"""Deterministic fingerprints for memory-build cache reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trajpatch.prompts.manager import load_prompt

MEMORY_CACHE_SCHEMA_VERSION = "v23"
MEMORY_PROMPT_NAMES = [
    "episodic_extract",
    "episodic_claim_text_extract",
    "episodic_claim_text_extract_structured",
    "claim_signal_extract",
    "claim_signal_extract_structured",
    "trajectory_match",
    "claim_transition_judge",
    "episodic_extract_structured",
    "trajectory_match_structured",
    "claim_transition_judge_structured",
    "repair",
    "episodic_claim_preservation_repair",
    "trajectory_retrieval_summary",
    "wiki_page_plan",
    "wiki_page_compile",
    "wiki_page_rerank",
    "trajectory_set_rerank",
]
MEMORY_CACHE_CODE_PATHS = [
    "ids.py",
    "datasets/locomo.py",
    "datasets/medmt.py",
    "pipeline/runner.py",
    "memory/orchestrator.py",
    "memory/llm_text_parsers.py",
    "memory/schemas.py",
    "memory/preservation.py",
    "memory/facets.py",
    "memory/wiki.py",
    "memory/trajectory_summaries.py",
    "memory/extraction_recovery.py",
    "cache/manager.py",
    "providers/openai_compatible_provider.py",
    "providers/structured_outputs.py",
    "storage/repository.py",
    "storage/models.py",
]


def stable_digest(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prompt_hashes() -> dict[str, str]:
    return {
        prompt_name: stable_digest(load_prompt(prompt_name))
        for prompt_name in MEMORY_PROMPT_NAMES
    }


def source_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    hashes: dict[str, str] = {}
    for relative_path in MEMORY_CACHE_CODE_PATHS:
        path = package_root / relative_path
        if not path.exists():
            hashes[relative_path] = "missing"
            continue
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def build_fingerprint_payload(config, adapter) -> dict[str, Any]:
    openai_compatible_memory_fields = {}
    if getattr(config, "provider_kind", None) == "openai-compatible":
        openai_compatible_memory_fields = {
            "openai_compatible_base_url": getattr(config, "openai_compatible_base_url", None),
            "openai_compatible_structured_mode": getattr(
                config, "openai_compatible_structured_mode", None
            ),
        }
    return {
        "schema_version": MEMORY_CACHE_SCHEMA_VERSION,
        "dataset_name": config.dataset,
        "adapter_version": getattr(adapter, "adapter_version", "v1"),
        "provider_kind": config.provider_kind,
        "backbone_model": config.backbone_model,
        **openai_compatible_memory_fields,
        "embedding_model": config.embedding_model,
        "m": config.m,
        "episodic_match_threshold": config.episodic_match_threshold,
        "prompt_hashes": prompt_hashes(),
        "source_hashes": source_hashes(),
    }


def build_memory_fingerprint(config, adapter) -> tuple[str, dict[str, Any]]:
    payload = build_fingerprint_payload(config, adapter)
    return stable_digest(payload), payload


def build_sample_history_fingerprint(sample, adapter) -> tuple[str, Any]:
    history_payload = adapter.history_fingerprint_payload(sample)
    return stable_digest(history_payload), history_payload
