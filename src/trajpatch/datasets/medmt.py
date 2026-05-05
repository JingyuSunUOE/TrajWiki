"""Adapter for the MedMT JSON benchmark files."""

from __future__ import annotations

import json
from pathlib import Path

from trajpatch.exceptions import DatasetFormatError
from trajpatch.types import DatasetSample, NormalizedMessage, QueryTask
from trajpatch.utils.multimodal import contains_structured_image_content

from .base import DatasetAdapter


class MedMTAdapter(DatasetAdapter):
    dataset_name = "medmt"
    adapter_version = "v3"
    all_subset_key = "all"
    subset_file_by_key = {
        "long_context_memory_and_understanding": "long_context_memory_and_understanding.json",
        "resistance_to_contextual_interference": "resistance_to_contextual_interference.json",
        "information_contradiction": "information_contradiction.json",
    }
    selected_files = list(subset_file_by_key.values())
    subset_key_by_file = {value: key for key, value in subset_file_by_key.items()}
    scene_mapping = {
        "Consultation": "HC",
        "Diagnosis": "PSRO",
        "Rehabilitation": "PTRM",
    }

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
                f"Unsupported MedMT subset '{dataset_subset}'. Expected one of {cls.supported_subset_keys()}."
            )
        if dataset_path.is_dir():
            return dataset_subset or cls.all_subset_key
        derived = cls.derive_subset_key_from_path(dataset_path)
        if derived is None:
            if dataset_subset is None:
                return cls.all_subset_key
            raise DatasetFormatError(
                f"MedMT subset '{dataset_subset}' requires a dataset directory or a canonical subset file name; "
                f"'{dataset_path.name}' cannot be matched to a supported subset."
            )
        requested = dataset_subset or derived
        if requested == cls.all_subset_key:
            raise DatasetFormatError(
                "MedMT subset 'all' requires a dataset directory; it cannot be used with a single subset file."
            )
        if requested != derived:
            raise DatasetFormatError(
                f"MedMT subset '{requested}' does not match file-derived subset '{derived}' for {dataset_path.name}."
            )
        return derived

    @classmethod
    def resolve_subset_paths(cls, dataset_path: Path, dataset_subset: str | None = None) -> list[Path]:
        scope_key = cls.resolve_subset_scope(dataset_path, dataset_subset)
        if dataset_path.is_dir():
            if scope_key == cls.all_subset_key:
                return [dataset_path / file_name for file_name in cls.selected_files]
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
            payload = json.loads(path.read_text(encoding="utf-8"))
            for item in payload:
                scene_type = ((item.get("meta") or {}).get("sence_type") or {}).get("type")
                scene_tag = self.scene_mapping.get(str(scene_type)) if scene_type else None
                # MedMT may include textual image mentions inside message strings. Treat those
                # as normal text and only exclude genuinely structured image payloads.
                contains_image = contains_structured_image_content(item.get("messages"))
                if contains_image:
                    excluded = True
                    exclusion_reason = "contains_image"
                elif scene_tag is None:
                    excluded = True
                    exclusion_reason = "unsupported_scene_tag"
                else:
                    excluded = False
                    exclusion_reason = None
                rows.append(
                    DatasetSample(
                        sample_id=item["id"],
                        dataset_name=self.dataset_name,
                        payload={**item, "subset_key": path.stem, "scene_tag": scene_tag, "source_file": path.name},
                        subset_key=path.stem,
                        scene_tag=scene_tag,
                        excluded=excluded,
                        exclusion_reason=exclusion_reason,
                    )
                )
        return rows

    def iterate_turns(self, sample: DatasetSample) -> list[NormalizedMessage]:
        messages = sample.payload.get("messages")
        if not messages:
            raise DatasetFormatError(f"MedMT sample {sample.sample_id} has no messages")
        history = messages[:-1]
        normalized: list[NormalizedMessage] = []
        for index, message in enumerate(history):
            normalized.append(
                NormalizedMessage(
                    role=message["role"],
                    content=message["content"].strip(),
                    turn_index=index,
                    metadata={},
                )
            )
        return normalized

    def build_query_tasks(self, sample: DatasetSample) -> list[QueryTask]:
        messages = sample.payload.get("messages")
        if not messages:
            raise DatasetFormatError(f"MedMT sample {sample.sample_id} has no messages")
        query_message = messages[-1]
        if query_message["role"] != "user":
            raise DatasetFormatError(
                f"MedMT sample {sample.sample_id} last message must be user, got {query_message['role']}"
            )
        category, subtype = self._instruction_following_metadata(sample)
        final_user_turn = self._stringify_content(query_message["content"])
        full_dialogue = self._serialize_full_dialogue(messages)
        answer_context = {
            "dataset": self.dataset_name,
            "category": category,
            "subtype": subtype,
            "rubric": sample.payload.get("test_point"),
            "scene_tag": sample.scene_tag,
            "subset_key": sample.subset_key,
        }
        judge_context = {
            "dataset": self.dataset_name,
            "category": category,
            "subtype": subtype,
            "full_dialogue": full_dialogue,
            "final_user_turn": final_user_turn,
            "rubric": sample.payload.get("test_point"),
            "subset_key": sample.subset_key,
            "scene_tag": sample.scene_tag,
        }
        return [
            QueryTask(
                query_task_id=f"{sample.sample_id}-final-query",
                sample_id=sample.sample_id,
                question=final_user_turn,
                metadata={
                    "test_point": sample.payload.get("test_point"),
                    "evaluated_info": sample.payload.get("evaluated_info", {}),
                    "meta": sample.payload.get("meta", {}),
                    "marked_info": sample.payload.get("marked_info", {}),
                    "answer_context": answer_context,
                    "judge_context": judge_context,
                },
            )
        ]

    def history_fingerprint_payload(self, sample: DatasetSample) -> list[dict]:
        messages = sample.payload.get("messages")
        if not messages:
            raise DatasetFormatError(f"MedMT sample {sample.sample_id} has no messages")
        return list(messages[:-1])

    def subset_key(self, sample: DatasetSample) -> str:
        return str(sample.subset_key or sample.payload.get("subset_key") or "unknown")

    def scene_tag(self, sample: DatasetSample) -> str | None:
        return sample.scene_tag or sample.payload.get("scene_tag")

    def _instruction_following_metadata(self, sample: DatasetSample) -> tuple[str, str]:
        meta = sample.payload.get("meta", {})
        instruction_block = (
            meta.get("insturct_following_type")
            or meta.get("instruction_following_type")
            or {}
        )
        category = str(instruction_block.get("type") or sample.subset_key or "Unknown")
        subtype = str(instruction_block.get("sub_type") or "Unknown")
        return category, subtype

    def _serialize_full_dialogue(self, messages: list[dict]) -> str:
        lines: list[str] = []
        for index, message in enumerate(messages, start=1):
            role = str(message.get("role", "unknown")).upper()
            content = self._stringify_content(message.get("content"))
            lines.append(f"[TURN {index}][{role}] {content}")
        return "\n".join(lines)

    def _stringify_content(self, content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        return json.dumps(content, ensure_ascii=False, sort_keys=True)

    def score_answer(
        self,
        sample: DatasetSample,
        query_task: QueryTask,
        answer_text: str,
        retrieved_source_refs: list[str],
        judge_result: dict | None = None,
    ) -> list[dict]:
        if judge_result is None:
            judge_result = {"verdict": "unknown", "score": 0.0, "rationale": "Judge not configured."}
        evaluated_info = query_task.metadata.get("evaluated_info", {})
        baseline_votes: dict[str, str] = {}
        for model_name, info in evaluated_info.items():
            if isinstance(info, dict):
                verdict = info.get("verify_result") or info.get("label")
                if verdict is not None:
                    baseline_votes[model_name] = str(verdict)
        yes_votes = sum(1 for verdict in baseline_votes.values() if verdict.lower() in {"yes", "pass", "correct"})
        no_votes = sum(1 for verdict in baseline_votes.values() if verdict.lower() in {"no", "fail", "incorrect"})
        baseline_majority = "correct" if yes_votes >= no_votes else "incorrect"
        return [
            {
                "metric_name": "rubric_pass",
                "metric_value": float(judge_result.get("verdict", "").lower() == "correct"),
                "verdict": judge_result.get("verdict"),
                "rationale": judge_result.get("rationale"),
                "details": {
                    "judge_score": judge_result.get("score"),
                    "test_point": query_task.metadata.get("test_point"),
                    "baseline_votes": baseline_votes,
                    "baseline_majority": baseline_majority,
                    "retrieved_source_refs": retrieved_source_refs,
                },
            }
        ]
