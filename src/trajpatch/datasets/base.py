"""Base dataset adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from trajpatch.types import DatasetSample, NormalizedMessage, QueryTask


class DatasetAdapter(ABC):
    """Abstract adapter for converting a dataset into replay and evaluation tasks."""

    dataset_name: str
    adapter_version: str = "v1"

    @abstractmethod
    def load_samples(
        self,
        dataset_path: Path,
        max_samples: int | None = None,
        dataset_subset: str | None = None,
    ) -> list[DatasetSample]:
        raise NotImplementedError

    @abstractmethod
    def iterate_turns(self, sample: DatasetSample) -> list[NormalizedMessage]:
        raise NotImplementedError

    @abstractmethod
    def build_query_tasks(self, sample: DatasetSample) -> list[QueryTask]:
        raise NotImplementedError

    @abstractmethod
    def history_fingerprint_payload(self, sample: DatasetSample):
        raise NotImplementedError

    @abstractmethod
    def subset_key(self, sample: DatasetSample) -> str:
        raise NotImplementedError

    @abstractmethod
    def scene_tag(self, sample: DatasetSample) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def score_answer(
        self,
        sample: DatasetSample,
        query_task: QueryTask,
        answer_text: str,
        retrieved_source_refs: list[str],
        judge_result: dict | None = None,
    ) -> list[dict]:
        raise NotImplementedError
