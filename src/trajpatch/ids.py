"""Helpers for generating stable episodic trajectory, snapshot, claim, op, and wiki ids."""

from __future__ import annotations

import re


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "item"


def trajectory_id(sample_id: str, ordinal: int) -> str:
    return f"epi-{slugify(sample_id)}-{ordinal:03d}"


def snapshot_id(trajectory_prefix: str, version: int) -> str:
    return f"{trajectory_prefix}-v{version:03d}"


def claim_id(trajectory_prefix: str, ordinal: int) -> str:
    return f"{trajectory_prefix}-c{ordinal:03d}"


def op_id(trajectory_prefix: str, ordinal: int) -> str:
    return f"{trajectory_prefix}-op{ordinal:03d}"


def wiki_page_id(sample_id: str, page_type: str, ordinal: int) -> str:
    return f"wiki-{slugify(sample_id)}-{slugify(page_type)}-{ordinal:03d}"
