"""Helpers for detecting multimodal/image-bearing benchmark samples."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


IMAGE_MARKER_RE = re.compile(
    r"\[(?:shared\s+)?image\s*:[^\]]+\]|<image>|image_url|input_image|data:image/",
    re.IGNORECASE,
)
IMAGE_TYPE_VALUES = {
    "image",
    "image_url",
    "input_image",
    "input_image_url",
    "image_path",
    "photo",
    "picture",
}
IMAGE_KEYS = {
    "image",
    "images",
    "image_url",
    "image_urls",
    "image_path",
    "image_paths",
    "photo",
    "photos",
    "picture",
    "pictures",
    "attachment",
    "attachments",
}


def contains_image_content(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, str):
        return bool(IMAGE_MARKER_RE.search(payload))
    if isinstance(payload, Mapping):
        payload_type = payload.get("type")
        if isinstance(payload_type, str) and payload_type.lower() in IMAGE_TYPE_VALUES:
            return True
        if any(key in payload for key in IMAGE_KEYS):
            return True
        return any(contains_image_content(value) for value in payload.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        return any(contains_image_content(item) for item in payload)
    return False


def contains_structured_image_content(payload: Any) -> bool:
    """Detect image-bearing structured payloads while ignoring plain text markers."""

    if payload is None:
        return False
    if isinstance(payload, str):
        return False
    if isinstance(payload, Mapping):
        payload_type = payload.get("type")
        if isinstance(payload_type, str) and payload_type.lower() in IMAGE_TYPE_VALUES:
            return True
        if any(key in payload for key in IMAGE_KEYS):
            return True
        return any(contains_structured_image_content(value) for value in payload.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        return any(contains_structured_image_content(item) for item in payload)
    return False
