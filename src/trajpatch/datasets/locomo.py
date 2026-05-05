"""Adapter for the LOCOMO long-conversation QA benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path

from trajpatch.exceptions import DatasetFormatError
from trajpatch.types import DatasetSample, NormalizedMessage, QueryTask
from trajpatch.utils.metrics import evidence_recall, exact_match
from trajpatch.utils.multimodal import contains_structured_image_content

from .base import DatasetAdapter

SESSION_RE = re.compile(r"^===== SESSION (?P<session>\d+) \| DATE: (?P<date>.+?) =====$")
MESSAGE_RE = re.compile(r"^\[(?P<source_ref>D\d+:\d+)\] (?P<speaker>[^:]+): (?P<content>.+)$")
SOURCE_REF_RE = re.compile(r"\bD\d+:\d+\b")


def _normalize_evidence_refs(raw_evidence: object) -> list[str]:
    values: list[object]
    if raw_evidence is None:
        values = []
    elif isinstance(raw_evidence, (list, tuple, set)):
        values = list(raw_evidence)
    else:
        values = [raw_evidence]

    refs: list[str] = []
    seen: set[str] = set()
    for value in values:
        for match in SOURCE_REF_RE.finditer(str(value or "")):
            ref = match.group(0)
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def _build_evidence_only_conversation(
    conversation: str | None,
    evidence_refs: list[str],
) -> tuple[str, list[str]]:
    if not evidence_refs:
        return "", []

    requested_refs = set(evidence_refs)
    found_refs: set[str] = set()
    evidence_lines: list[str] = []
    for raw_line in str(conversation or "").splitlines():
        line = raw_line.strip()
        message_match = MESSAGE_RE.match(line)
        if not message_match:
            continue
        source_ref = message_match.group("source_ref")
        if source_ref in requested_refs:
            found_refs.add(source_ref)
            evidence_lines.append(line)

    missing_refs = [ref for ref in evidence_refs if ref not in found_refs]
    return "\n".join(evidence_lines), missing_refs


class LocomoAdapter(DatasetAdapter):
    dataset_name = "locomo"
    adapter_version = "v3"
    all_subset_key = "all"
    subset_file_by_key = {
        "multi_hop": "category_1_multi_hop.jsonl",
        "temporal": "category_2_temporal.jsonl",
        "open_domain": "category_3_open_domain.jsonl",
        "single_hop": "category_4_single_hop.jsonl",
    }
    supported_subset_files = [
        "category_1_multi_hop.jsonl",
        "category_2_temporal.jsonl",
        "category_3_open_domain.jsonl",
        "category_4_single_hop.jsonl",
    ]
    subset_key_by_file = {value: key for key, value in subset_file_by_key.items()}

    @classmethod
    def supported_subset_keys(cls) -> list[str]:
        return [cls.all_subset_key, *cls.subset_file_by_key.keys()]

    @classmethod
    def derive_subset_key_from_path(cls, dataset_path: Path) -> str | None:
        if dataset_path.is_dir():
            return None
        return cls.subset_key_by_file.get(dataset_path.name)

    @classmethod
    def resolve_subset_scope(cls, dataset_path: Path, dataset_subset: str | None = None) -> str:
        if dataset_subset is not None and dataset_subset not in cls.supported_subset_keys():
            raise DatasetFormatError(
                f"Unsupported LOCOMO subset '{dataset_subset}'. Expected one of {cls.supported_subset_keys()}."
            )
        if dataset_path.is_dir():
            return dataset_subset or cls.all_subset_key
        derived = cls.derive_subset_key_from_path(dataset_path)
        if derived is None:
            if dataset_subset is None:
                return cls.all_subset_key
            raise DatasetFormatError(
                f"LOCOMO subset '{dataset_subset}' requires a dataset directory or a canonical subset file name; "
                f"'{dataset_path.name}' cannot be matched to a supported subset."
            )
        requested = dataset_subset or derived
        if requested == cls.all_subset_key:
            raise DatasetFormatError(
                "LOCOMO subset 'all' requires a dataset directory; it cannot be used with a single subset file."
            )
        if requested != derived:
            raise DatasetFormatError(
                f"LOCOMO subset '{requested}' does not match file-derived subset '{derived}' for {dataset_path.name}."
            )
        return derived

    @classmethod
    def resolve_subset_paths(cls, dataset_path: Path, dataset_subset: str | None = None) -> list[Path]:
        scope_key = cls.resolve_subset_scope(dataset_path, dataset_subset)
        if dataset_path.is_dir():
            if scope_key == cls.all_subset_key:
                return [dataset_path / file_name for file_name in cls.supported_subset_files]
            return [dataset_path / cls.subset_file_by_key[scope_key]]
        return [dataset_path]

    def load_samples(
        self,
        dataset_path: Path,
        max_samples: int | None = None,
        dataset_subset: str | None = None,
    ) -> list[DatasetSample]:
        rows: list[DatasetSample] = []
        paths = self.resolve_subset_paths(dataset_path, dataset_subset)
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    if int(payload.get("category", 0) or 0) == 5:
                        continue
                    subset_key = str(payload.get("category_name") or "unknown")
                    # LOCOMO conversations may mention shared images in plain text. Treat those
                    # as normal dialogue and only exclude genuinely structured image payloads.
                    contains_image = contains_structured_image_content(payload)
                    rows.append(
                        DatasetSample(
                            sample_id=payload["sample_id"],
                            dataset_name=self.dataset_name,
                            payload={**payload, "subset_key": subset_key, "source_file": path.name},
                            subset_key=subset_key,
                            excluded=contains_image,
                            exclusion_reason="contains_image" if contains_image else None,
                        )
                    )
        return rows

    def iterate_turns(self, sample: DatasetSample) -> list[NormalizedMessage]:
        conversation = sample.payload.get("full_conversation")
        if not conversation:
            raise DatasetFormatError(f"LOCOMO sample {sample.sample_id} has no full_conversation")

        normalized: list[NormalizedMessage] = []
        current_date: str | None = None
        speaker_to_role: dict[str, str] = {}

        for raw_line in conversation.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            session_match = SESSION_RE.match(line)
            if session_match:
                current_date = session_match.group("date")
                continue
            message_match = MESSAGE_RE.match(line)
            if not message_match:
                continue
            speaker = message_match.group("speaker")
            if speaker not in speaker_to_role:
                speaker_to_role[speaker] = "user" if not speaker_to_role else "assistant"
            normalized.append(
                NormalizedMessage(
                    role=speaker_to_role[speaker],  # type: ignore[arg-type]
                    content=message_match.group("content").strip(),
                    turn_index=len(normalized),
                    source_ref=message_match.group("source_ref"),
                    speaker_name=speaker,
                    occurred_at=current_date,
                    metadata={"session_date": current_date},
                )
            )
        return normalized

    def build_query_tasks(self, sample: DatasetSample) -> list[QueryTask]:
        payload = sample.payload
        gold_evidence_raw = payload.get("evidence", [])
        gold_evidence_refs = _normalize_evidence_refs(gold_evidence_raw)
        raw_evidence_only_conversation = str(payload.get("evidence_only_conversation") or "").strip()
        if raw_evidence_only_conversation:
            evidence_only_conversation = raw_evidence_only_conversation
            evidence_only_conversation_generated = False
            evidence_missing_refs: list[str] = []
        else:
            evidence_only_conversation, evidence_missing_refs = _build_evidence_only_conversation(
                payload.get("full_conversation"),
                gold_evidence_refs,
            )
            evidence_only_conversation_generated = bool(gold_evidence_refs)
        return [
            QueryTask(
                query_task_id=payload["qa_uid"],
                sample_id=sample.sample_id,
                question=payload["question"],
                gold_answer=payload.get("answer"),
                gold_evidence=gold_evidence_refs,
                metadata={
                    "category": payload.get("category"),
                    "category_name": payload.get("category_name"),
                    "evidence_only_conversation": evidence_only_conversation,
                    "gold_evidence_refs": gold_evidence_refs,
                    "gold_evidence_raw": gold_evidence_raw,
                    "evidence_only_conversation_generated": evidence_only_conversation_generated,
                    "evidence_missing_refs": evidence_missing_refs,
                },
            )
        ]

    def history_fingerprint_payload(self, sample: DatasetSample) -> str:
        conversation = sample.payload.get("full_conversation")
        if not conversation:
            raise DatasetFormatError(f"LOCOMO sample {sample.sample_id} has no full_conversation")
        return str(conversation)

    def subset_key(self, sample: DatasetSample) -> str:
        return str(sample.subset_key or sample.payload.get("subset_key") or sample.payload.get("category_name") or "unknown")

    def scene_tag(self, sample: DatasetSample) -> str | None:
        return None

    def score_answer(
        self,
        sample: DatasetSample,
        query_task: QueryTask,
        answer_text: str,
        retrieved_source_refs: list[str],
        judge_result: dict | None = None,
    ) -> list[dict]:
        gold = query_task.gold_answer or ""
        return [
            {"metric_name": "exact_match", "metric_value": exact_match(answer_text, gold)},
            {
                "metric_name": "evidence_recall",
                "metric_value": evidence_recall(retrieved_source_refs, query_task.gold_evidence),
                "details": {
                    "retrieved_source_refs": retrieved_source_refs,
                    "gold_evidence": query_task.gold_evidence,
                },
            },
        ]
