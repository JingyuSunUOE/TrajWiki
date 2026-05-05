from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select

from trajpatch.config import RunConfig
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.mock import HashEmbeddingProvider, MockLLMProvider
from trajpatch.storage.models import AnswerRecord, EmbeddingRecord, EvaluationRecord, TrajectoryRecord, WikiPageRecord


def _sample_medmt_root(root: Path, *, sample_id: str = "sample-a") -> Path:
    root.mkdir()
    payload = [
        {
            "id": sample_id,
            "messages": [
                {"role": "system", "content": "Keep answers short."},
                {"role": "user", "content": "I smoke 5 cigarettes a day."},
                {"role": "assistant", "content": "Thanks, noted."},
                {"role": "user", "content": "Actually I never smoke."},
                {"role": "assistant", "content": "That contradicts the earlier message."},
                {"role": "user", "content": "What do you know about my smoking habits?"},
            ],
            "test_point": "Mention uncertainty when the context contains contradiction.",
            "evaluated_info": {"baseline-a": {"verify_result": "Yes"}},
            "meta": {
                "sence_type": {"type": "Consultation"},
                "insturct_following_type": {
                    "type": "Instruction Clarification",
                    "sub_type": "Information Contradiction",
                },
            },
        }
    ]
    (root / "long_context_memory_and_understanding.json").write_text(json.dumps(payload), encoding="utf-8")
    for file_name in [
        "resistance_to_contextual_interference.json",
        "information_contradiction.json",
    ]:
        (root / file_name).write_text("[]", encoding="utf-8")
    return root


def _scripted_callback(messages, system_prompt, metadata):
    prompt = messages[-1].content
    task = metadata.get("task")
    raw_ids = re.findall(r"(sample-a-m\d{4})", prompt)
    if task == "episodic_extract":
        if "I smoke 5 cigarettes a day." in prompt:
            return (
                "SUMMARY_CONTENT: The user reports a smoking habit.\n"
                "CONTEXT: Health behavior disclosure.\n"
                "KEYWORDS: smoking, cigarettes"
            )
        if "Actually I never smoke." in prompt:
            return (
                "SUMMARY_CONTENT: The user denies smoking.\n"
                "CONTEXT: Contradictory update to prior health behavior.\n"
                "KEYWORDS: smoking, contradiction"
            )
        return "NO_MEMORY"
    if task == "episodic_claim_text_extract":
        if "I smoke 5 cigarettes a day." in prompt:
            return (
                "HAS_CLAIMS: true\n"
                "REASON: extracted smoking claim\n\n"
                "[CLAIMS]\n"
                f"- status=active | source_message_ids={raw_ids[-2]} | supporting_quote=I smoke 5 cigarettes a day. | text=User smokes 5 cigarettes a day."
            )
        if "Actually I never smoke." in prompt:
            return (
                "HAS_CLAIMS: true\n"
                "REASON: extracted smoking denial\n\n"
                "[CLAIMS]\n"
                f"- status=contradictory | source_message_ids={raw_ids[-2]} | supporting_quote=Actually I never smoke. | text=User says they never smoke."
            )
        return "NO_MEMORY"
    if task == "trajectory_match":
        return "DECISION: CONTINUE\nSELECTED_CANDIDATE: T1\nRATIONALE: This continues the same smoking trajectory."
    if task == "answer_generation":
        return "The memory is contradictory: one message says you smoke 5 cigarettes a day, and a later message says you never smoke. I would confirm which is current."
    if task == "medmt_judge":
        return "CORRECT"
    return "NO_MEMORY"


def test_memory_cache_hits_on_second_run_and_resets_db(tmp_path: Path):
    dataset_path = _sample_medmt_root(tmp_path / "medmt")
    cache_dir = tmp_path / ".trajpatch_cache"
    counter: Counter[str] = Counter()

    def callback(messages, system_prompt, metadata):
        counter[str((metadata or {}).get("task"))] += 1
        return _scripted_callback(messages, system_prompt, metadata or {})

    config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=True,
        memory_cache_dir=cache_dir,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
    )

    runner_first = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(callback=callback),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(callback=callback, model_name="mock-judge"),
    )
    report_first = runner_first.run()
    first_counts = dict(counter)

    assert report_first.details["cache"]["cache_hits"] == 0.0
    assert report_first.details["cache"]["cache_misses"] == 1.0
    assert report_first.details["cache"]["cache_writes"] == 1.0

    runner_second = PipelineRunner(
        config,
        llm_provider=MockLLMProvider(callback=callback),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(callback=callback, model_name="mock-judge"),
    )
    report_second = runner_second.run()

    assert report_second.details["cache"]["cache_hits"] == 1.0
    assert report_second.details["cache"]["cache_writes"] == 0.0
    assert counter["episodic_extract"] == first_counts["episodic_extract"]
    assert counter["trajectory_match"] == first_counts["trajectory_match"]
    assert counter["answer_generation"] == first_counts["answer_generation"] + 1
    assert counter["medmt_judge"] == first_counts["medmt_judge"] + 1
    assert runner_second.session.scalar(select(func.count()).select_from(TrajectoryRecord)) == 1
    assert runner_second.session.scalar(select(func.count()).select_from(AnswerRecord)) == 1
    assert runner_second.session.scalar(select(func.count()).select_from(EvaluationRecord)) == 1


def test_memory_cache_ignores_retrieval_width_and_judge_model_but_tracks_m(tmp_path: Path):
    dataset_path = _sample_medmt_root(tmp_path / "medmt")
    cache_dir = tmp_path / ".trajpatch_cache"
    base = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=True,
        memory_cache_dir=cache_dir,
        provider_kind="mock",
        backbone_model="mock-backbone",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
    )

    PipelineRunner(
        base,
        llm_provider=MockLLMProvider(callback=_scripted_callback),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(callback=_scripted_callback, model_name="mock-judge"),
    ).run()

    report_same_memory = PipelineRunner(
        base.copy(update={"k": 3, "t_pages": 7, "neighbor_radius": 2, "judge_model": "mock-judge-v2"}),
        llm_provider=MockLLMProvider(callback=_scripted_callback),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(callback=_scripted_callback, model_name="mock-judge-v2"),
    ).run()
    assert report_same_memory.details["cache"]["cache_hits"] == 1.0
    assert report_same_memory.details["cache"]["cache_misses"] == 0.0

    report_changed_m = PipelineRunner(
        base.copy(update={"m": 5}),
        llm_provider=MockLLMProvider(callback=_scripted_callback),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(callback=_scripted_callback, model_name="mock-judge"),
    ).run()
    assert report_changed_m.details["cache"]["cache_misses"] == 1.0


def test_memory_cache_remaps_wiki_pages_for_same_history_different_sample_id(tmp_path: Path):
    first_dataset_path = _sample_medmt_root(tmp_path / "medmt-a", sample_id="sample-a")
    second_dataset_path = _sample_medmt_root(tmp_path / "medmt-b", sample_id="sample-b")
    cache_dir = tmp_path / ".trajpatch_cache"
    base_kwargs = {
        "dataset": "medmt",
        "output_dir": tmp_path / "artifacts",
        "database_path": tmp_path / "artifacts" / "run.sqlite",
        "memory_cache_enabled": True,
        "memory_cache_dir": cache_dir,
        "provider_kind": "mock",
        "backbone_model": "mock-backbone",
        "embedding_model": "huggingface/Qwen3-Embedding-8B",
        "judge_model": "mock-judge",
        "m": 2,
        "t_pages": 2,
        "k": 1,
    }
    first_config = RunConfig(dataset_path=first_dataset_path, **base_kwargs)
    PipelineRunner(
        first_config,
        llm_provider=MockLLMProvider(callback=_scripted_callback),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(callback=_scripted_callback, model_name="mock-judge"),
    ).run()

    second_config = RunConfig(dataset_path=second_dataset_path, **base_kwargs)
    runner_second = PipelineRunner(
        second_config,
        llm_provider=MockLLMProvider(callback=_scripted_callback),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=MockLLMProvider(callback=_scripted_callback, model_name="mock-judge"),
    )
    report_second = runner_second.run()

    assert report_second.details["cache"]["cache_hits"] == 1.0
    pages = list(
        runner_second.session.execute(
            select(WikiPageRecord).where(WikiPageRecord.sample_id == "sample-b")
        )
        .scalars()
        .all()
    )
    assert pages
    assert all(page.id.startswith("wiki-sample-b-") for page in pages)
    assert all(page.embedding_id == f"{page.id}-emb" for page in pages if page.embedding_id)
    assert all("sample-a" not in json.dumps(page.trajectory_ids_json + page.linked_page_ids_json) for page in pages)
    assert all("sample-a" not in json.dumps(page.metadata_json or {}, sort_keys=True) for page in pages)
    assert all("sample-a" not in (page.markdown_text or "") for page in pages)

    wiki_embeddings = list(
        runner_second.session.execute(
            select(EmbeddingRecord).where(EmbeddingRecord.owner_type == "wiki_page")
        )
        .scalars()
        .all()
    )
    assert wiki_embeddings
    assert {embedding.owner_id for embedding in wiki_embeddings} == {page.id for page in pages}
    assert all(embedding.id == f"{embedding.owner_id}-emb" for embedding in wiki_embeddings)
    assert all("sample-a" not in (embedding.semantic_text or "") for embedding in wiki_embeddings)
    assert all("sample-a" not in trajectory.id for trajectory in runner_second.store.list_trajectories("sample-b"))
