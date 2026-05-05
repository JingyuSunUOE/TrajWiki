from __future__ import annotations

import json

import pytest

from trajpatch.config import RunConfig
from trajpatch.exceptions import DatasetFormatError
from trajpatch.datasets.locomo import LocomoAdapter
from trajpatch.datasets.medmt import MedMTAdapter


def test_locomo_adapter_parses_conversation_and_keeps_textual_shared_image_markers(tmp_path):
    dataset_path = tmp_path / "locomo.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "conv-1",
                        "qa_uid": "conv-1_qa_0",
                        "category": 1,
                        "category_name": "multi_hop",
                        "question": "What does Alice prefer?",
                        "answer": "Tea",
                        "evidence": ["D1:1"],
                        "full_conversation": (
                            "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                            "[D1:1] Alice: I love tea.\n"
                            "[D1:2] Bob: Nice, I prefer coffee.\n"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "conv-image",
                        "qa_uid": "conv-image_qa_0",
                        "category": 3,
                        "category_name": "open_domain",
                        "question": "What happened at the party?",
                        "answer": "People celebrated.",
                        "evidence": ["D3:1"],
                        "full_conversation": (
                            "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                            "[D3:1] Alice: Look at this! [shared image: a photography of a party]\n"
                            "[D3:2] Bob: Nice.\n"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "conv-2",
                        "qa_uid": "conv-2_qa_0",
                        "category": 5,
                        "category_name": "adversarial",
                        "question": "Ignore me?",
                        "answer": "Yes",
                        "evidence": ["D2:1"],
                        "full_conversation": (
                            "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                            "[D2:1] Alice: Ignore this row.\n"
                        ),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = LocomoAdapter()
    samples = adapter.load_samples(dataset_path)
    active_samples = [sample for sample in samples if not sample.excluded]
    image_sample = next(sample for sample in samples if sample.sample_id == "conv-image")
    sample = active_samples[0]
    turns = adapter.iterate_turns(sample)
    task = adapter.build_query_tasks(sample)[0]
    history = adapter.history_fingerprint_payload(sample)

    assert len(samples) == 2
    assert len(active_samples) == 2
    assert sample.subset_key == "multi_hop"
    assert image_sample.excluded is False
    assert image_sample.exclusion_reason is None
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
    assert task.gold_evidence == ["D1:1"]
    assert "[D1:1] Alice: I love tea." in task.metadata["evidence_only_conversation"]
    assert task.metadata["gold_evidence_refs"] == ["D1:1"]
    assert task.metadata["gold_evidence_raw"] == ["D1:1"]
    assert task.metadata["evidence_only_conversation_generated"] is True
    assert task.metadata["evidence_missing_refs"] == []
    assert "Alice: I love tea." in history


def test_locomo_adapter_normalizes_combined_evidence_and_synthesizes_evidence_text(tmp_path):
    dataset_path = tmp_path / "locomo.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "conv-combined",
                        "qa_uid": "conv-combined_qa_0",
                        "category": 1,
                        "category_name": "multi_hop",
                        "question": "What did Melanie paint recently?",
                        "answer": "sunset",
                        "evidence": ["D8:6; D9:17"],
                        "evidence_only_conversation": "",
                        "full_conversation": (
                            "===== SESSION 8 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                            "[D8:5] Caroline: What are you painting?\n"
                            "[D8:6] Melanie: We painted a sunset last weekend.\n"
                            "===== SESSION 9 | DATE: 2:00 pm on 2 Jan, 2024 =====\n"
                            "[D9:17] Melanie: My kids and I finished another painting like our last one.\n"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "conv-spaced",
                        "qa_uid": "conv-spaced_qa_0",
                        "category": 3,
                        "category_name": "open_domain",
                        "question": "Which refs matter?",
                        "answer": "all",
                        "evidence": ["D9:1 D4:4 D4:6"],
                        "full_conversation": (
                            "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                            "[D4:4] Alice: Four four.\n"
                            "[D4:6] Alice: Four six.\n"
                            "[D9:1] Alice: Nine one.\n"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "conv-existing",
                        "qa_uid": "conv-existing_qa_0",
                        "category": 4,
                        "category_name": "single_hop",
                        "question": "What does Alice like?",
                        "answer": "tea",
                        "evidence": ["D1:1"],
                        "evidence_only_conversation": "[D1:1] Existing evidence text.",
                        "full_conversation": (
                            "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                            "[D1:1] Alice: I like tea.\n"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "conv-missing",
                        "qa_uid": "conv-missing_qa_0",
                        "category": 4,
                        "category_name": "single_hop",
                        "question": "What is missing?",
                        "answer": "missing",
                        "evidence": ["D1:1 D99:9"],
                        "full_conversation": (
                            "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                            "[D1:1] Alice: Present evidence.\n"
                        ),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = LocomoAdapter()
    tasks = {
        sample.sample_id: adapter.build_query_tasks(sample)[0]
        for sample in adapter.load_samples(dataset_path)
    }

    combined = tasks["conv-combined"]
    assert combined.gold_evidence == ["D8:6", "D9:17"]
    assert "[D8:6] Melanie: We painted a sunset last weekend." in combined.metadata["evidence_only_conversation"]
    assert (
        "[D9:17] Melanie: My kids and I finished another painting like our last one."
        in combined.metadata["evidence_only_conversation"]
    )
    assert combined.metadata["evidence_only_conversation_generated"] is True
    assert combined.metadata["evidence_missing_refs"] == []

    spaced = tasks["conv-spaced"]
    assert spaced.gold_evidence == ["D9:1", "D4:4", "D4:6"]
    assert spaced.metadata["evidence_only_conversation"].splitlines() == [
        "[D4:4] Alice: Four four.",
        "[D4:6] Alice: Four six.",
        "[D9:1] Alice: Nine one.",
    ]

    existing = tasks["conv-existing"]
    assert existing.gold_evidence == ["D1:1"]
    assert existing.metadata["evidence_only_conversation"] == "[D1:1] Existing evidence text."
    assert existing.metadata["evidence_only_conversation_generated"] is False
    assert existing.metadata["evidence_missing_refs"] == []

    missing = tasks["conv-missing"]
    assert missing.gold_evidence == ["D1:1", "D99:9"]
    assert missing.metadata["evidence_only_conversation"] == "[D1:1] Alice: Present evidence."
    assert missing.metadata["evidence_missing_refs"] == ["D99:9"]


def test_medmt_adapter_builds_history_query_and_scene_tags(tmp_path):
    dataset_root = tmp_path / "medmt"
    dataset_root.mkdir()
    (dataset_root / "long_context_memory_and_understanding.json").write_text(
        json.dumps(
            [
                {
                    "id": "med-1",
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "First question"},
                        {"role": "assistant", "content": "First answer"},
                        {"role": "user", "content": "Final question?"},
                    ],
                    "test_point": "Be concise.",
                    "evaluated_info": {"model-x": {"verify_result": "Yes"}},
                    "meta": {
                        "sence_type": {"type": "Consultation"},
                        "insturct_following_type": {
                            "type": "Long-Context Memory and Understanding",
                            "sub_type": "Detailed Information Comprehension",
                        },
                    },
                },
                {
                    "id": "med-2",
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "Another question"},
                        {"role": "assistant", "content": "Another answer"},
                        {"role": "user", "content": "Final question?"},
                    ],
                    "test_point": "Stay grounded.",
                    "meta": {"sence_type": {"type": "Nursing"}},
                },
                {
                    "id": "med-3",
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": [{"type": "text", "text": "Look at this"}, {"type": "image_url", "image_url": "https://example.com/x.png"}]},
                        {"role": "assistant", "content": "Another answer"},
                        {"role": "user", "content": "Final question?"},
                    ],
                    "test_point": "Use the text only.",
                    "meta": {"sence_type": {"type": "Consultation"}},
                },
                {
                    "id": "med-4",
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "I shared a photo earlier. [shared image: a chest x-ray photo]"},
                        {"role": "assistant", "content": "Understood."},
                        {"role": "user", "content": "Summarize what I said."},
                    ],
                    "test_point": "Use the conversation text.",
                    "meta": {"sence_type": {"type": "Consultation"}},
                },
            ]
        ),
        encoding="utf-8",
    )
    for file_name in [
        "resistance_to_contextual_interference.json",
        "information_contradiction.json",
    ]:
        (dataset_root / file_name).write_text("[]", encoding="utf-8")

    adapter = MedMTAdapter()
    samples = adapter.load_samples(dataset_root)
    sample = samples[0]
    nursing_sample = samples[1]
    image_sample = samples[2]
    textual_image_sample = samples[3]
    turns = adapter.iterate_turns(sample)
    task = adapter.build_query_tasks(sample)[0]
    history = adapter.history_fingerprint_payload(sample)

    assert [turn.role for turn in turns] == ["system", "user", "assistant"]
    assert task.question == "Final question?"
    assert task.metadata["test_point"] == "Be concise."
    assert len(history) == 3
    assert history[-1]["role"] == "assistant"
    assert sample.scene_tag == "HC"
    assert task.metadata["answer_context"] == {
        "dataset": "medmt",
        "category": "Long-Context Memory and Understanding",
        "subtype": "Detailed Information Comprehension",
        "rubric": "Be concise.",
        "scene_tag": "HC",
        "subset_key": "long_context_memory_and_understanding",
    }
    assert task.metadata["judge_context"]["dataset"] == "medmt"
    assert task.metadata["judge_context"]["category"] == "Long-Context Memory and Understanding"
    assert task.metadata["judge_context"]["subtype"] == "Detailed Information Comprehension"
    assert task.metadata["judge_context"]["final_user_turn"] == "Final question?"
    assert task.metadata["judge_context"]["rubric"] == "Be concise."
    assert "[TURN 1][SYSTEM] System prompt." in task.metadata["judge_context"]["full_dialogue"]
    assert "[TURN 4][USER] Final question?" in task.metadata["judge_context"]["full_dialogue"]
    assert nursing_sample.excluded is True
    assert nursing_sample.scene_tag is None
    assert nursing_sample.exclusion_reason == "unsupported_scene_tag"
    assert image_sample.excluded is True
    assert image_sample.exclusion_reason == "contains_image"
    assert textual_image_sample.excluded is False
    assert textual_image_sample.scene_tag == "HC"
    assert textual_image_sample.exclusion_reason is None


def _write_locomo_subset(path, *, sample_id: str, category_name: str, question: str) -> None:
    path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "qa_uid": f"{sample_id}_qa_0",
                "category_name": category_name,
                "question": question,
                "answer": "Answer",
                "evidence": ["D1:1"],
                "full_conversation": (
                    "===== SESSION 1 | DATE: 1:00 pm on 1 Jan, 2024 =====\n"
                    "[D1:1] Alice: First fact.\n"
                    "[D1:2] Bob: Second fact.\n"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_locomo_adapter_subset_scope_resolves_directories_and_files(tmp_path):
    dataset_root = tmp_path / "locomo"
    dataset_root.mkdir()
    _write_locomo_subset(dataset_root / "category_1_multi_hop.jsonl", sample_id="conv-multi", category_name="multi_hop", question="Q1")
    _write_locomo_subset(dataset_root / "category_2_temporal.jsonl", sample_id="conv-temporal", category_name="temporal", question="Q2")
    _write_locomo_subset(dataset_root / "category_3_open_domain.jsonl", sample_id="conv-open", category_name="open_domain", question="Q3")
    _write_locomo_subset(dataset_root / "category_4_single_hop.jsonl", sample_id="conv-single", category_name="single_hop", question="Q4")

    adapter = LocomoAdapter()
    legacy_file = tmp_path / "all_qa.jsonl"
    _write_locomo_subset(legacy_file, sample_id="conv-legacy", category_name="multi_hop", question="Q0")

    all_samples = adapter.load_samples(dataset_root, dataset_subset="all")
    multi_hop_samples = adapter.load_samples(dataset_root, dataset_subset="multi_hop")
    derived_file_samples = adapter.load_samples(dataset_root / "category_2_temporal.jsonl")
    legacy_file_samples = adapter.load_samples(legacy_file)

    assert [sample.sample_id for sample in all_samples] == [
        "conv-multi",
        "conv-temporal",
        "conv-open",
        "conv-single",
    ]
    assert [sample.sample_id for sample in multi_hop_samples] == ["conv-multi"]
    assert [sample.sample_id for sample in derived_file_samples] == ["conv-temporal"]
    assert [sample.sample_id for sample in legacy_file_samples] == ["conv-legacy"]
    assert adapter.resolve_subset_scope(dataset_root, "all") == "all"
    assert adapter.resolve_subset_scope(dataset_root / "category_2_temporal.jsonl") == "temporal"
    assert adapter.resolve_subset_scope(legacy_file) == "all"


def test_locomo_adapter_rejects_invalid_subset_file_combinations(tmp_path):
    dataset_root = tmp_path / "locomo"
    dataset_root.mkdir()
    subset_path = dataset_root / "category_1_multi_hop.jsonl"
    _write_locomo_subset(subset_path, sample_id="conv-multi", category_name="multi_hop", question="Q1")

    adapter = LocomoAdapter()

    with pytest.raises(DatasetFormatError, match="does not match file-derived subset 'multi_hop'"):
        adapter.load_samples(subset_path, dataset_subset="temporal")
    with pytest.raises(DatasetFormatError, match="subset 'all' requires a dataset directory"):
        adapter.load_samples(subset_path, dataset_subset="all")


def test_run_config_validates_locomo_subset_usage(tmp_path):
    dataset_root = tmp_path / "locomo"
    dataset_root.mkdir()
    _write_locomo_subset(dataset_root / "category_1_multi_hop.jsonl", sample_id="conv-multi", category_name="multi_hop", question="Q1")
    _write_locomo_subset(dataset_root / "category_2_temporal.jsonl", sample_id="conv-temporal", category_name="temporal", question="Q2")
    _write_locomo_subset(dataset_root / "category_3_open_domain.jsonl", sample_id="conv-open", category_name="open_domain", question="Q3")
    _write_locomo_subset(dataset_root / "category_4_single_hop.jsonl", sample_id="conv-single", category_name="single_hop", question="Q4")

    config = RunConfig(dataset="locomo", dataset_subset="multi_hop", dataset_path=dataset_root, database_path=tmp_path / "run.sqlite")
    assert config.dataset_subset == "multi_hop"

    with pytest.raises(ValueError, match="dataset_subset for LOCOMO must be one of"):
        RunConfig(dataset="locomo", dataset_subset="bad", dataset_path=dataset_root, database_path=tmp_path / "bad.sqlite")


def _write_medmt_subset(path, *, sample_id: str, test_point: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": sample_id,
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "History turn"},
                        {"role": "assistant", "content": "Assistant turn"},
                        {"role": "user", "content": "Final question?"},
                    ],
                    "test_point": test_point,
                    "meta": {"sence_type": {"type": "Consultation"}},
                }
            ]
        ),
        encoding="utf-8",
    )


def test_medmt_adapter_subset_scope_resolves_directories_and_files(tmp_path):
    dataset_root = tmp_path / "medmt"
    dataset_root.mkdir()
    _write_medmt_subset(dataset_root / "long_context_memory_and_understanding.json", sample_id="med-long", test_point="TP1")
    _write_medmt_subset(dataset_root / "resistance_to_contextual_interference.json", sample_id="med-resist", test_point="TP2")
    _write_medmt_subset(dataset_root / "information_contradiction.json", sample_id="med-contradict", test_point="TP3")

    adapter = MedMTAdapter()

    all_samples = adapter.load_samples(dataset_root, dataset_subset="all")
    subset_samples = adapter.load_samples(
        dataset_root,
        dataset_subset="resistance_to_contextual_interference",
    )
    derived_file_samples = adapter.load_samples(dataset_root / "information_contradiction.json")

    assert [sample.sample_id for sample in all_samples] == [
        "med-long",
        "med-resist",
        "med-contradict",
    ]
    assert [sample.sample_id for sample in subset_samples] == ["med-resist"]
    assert [sample.sample_id for sample in derived_file_samples] == ["med-contradict"]
    assert adapter.resolve_subset_scope(dataset_root, "all") == "all"
    assert adapter.resolve_subset_scope(dataset_root / "information_contradiction.json") == "information_contradiction"


def test_medmt_adapter_rejects_invalid_subset_file_combinations(tmp_path):
    dataset_root = tmp_path / "medmt"
    dataset_root.mkdir()
    subset_path = dataset_root / "long_context_memory_and_understanding.json"
    _write_medmt_subset(subset_path, sample_id="med-long", test_point="TP1")

    adapter = MedMTAdapter()

    with pytest.raises(
        DatasetFormatError,
        match="does not match file-derived subset 'long_context_memory_and_understanding'",
    ):
        adapter.load_samples(subset_path, dataset_subset="information_contradiction")
    with pytest.raises(DatasetFormatError, match="subset 'all' requires a dataset directory"):
        adapter.load_samples(subset_path, dataset_subset="all")


def test_run_config_validates_medmt_subset_usage(tmp_path):
    dataset_root = tmp_path / "medmt"
    dataset_root.mkdir()
    _write_medmt_subset(dataset_root / "long_context_memory_and_understanding.json", sample_id="med-long", test_point="TP1")
    _write_medmt_subset(dataset_root / "resistance_to_contextual_interference.json", sample_id="med-resist", test_point="TP2")
    _write_medmt_subset(dataset_root / "information_contradiction.json", sample_id="med-contradict", test_point="TP3")

    config = RunConfig(
        dataset="medmt",
        dataset_subset="information_contradiction",
        dataset_path=dataset_root,
        database_path=tmp_path / "med.sqlite",
    )
    assert config.dataset_subset == "information_contradiction"

    with pytest.raises(ValueError, match="dataset_subset for MEDMT must be one of"):
        RunConfig(
            dataset="medmt",
            dataset_subset="multi_hop",
            dataset_path=dataset_root,
            database_path=tmp_path / "bad-med.sqlite",
        )
