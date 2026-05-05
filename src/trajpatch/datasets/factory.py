"""Factory for dataset adapters."""

from __future__ import annotations

from trajpatch.exceptions import DatasetFormatError

from .base import DatasetAdapter
from .locomo import LocomoAdapter
from .medmt import MedMTAdapter


def build_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    if dataset_name == "locomo":
        return LocomoAdapter()
    if dataset_name == "medmt":
        return MedMTAdapter()
    raise DatasetFormatError(f"Unsupported dataset: {dataset_name}")
