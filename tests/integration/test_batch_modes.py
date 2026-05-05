from __future__ import annotations

import json
import io
import re
import sqlite3
from collections import Counter
from pathlib import Path

from rich.console import Console

from trajpatch.config import RunConfig
from trajpatch.exceptions import ProviderConfigurationError, StructuredOutputError
from trajpatch.memory.schemas import (
    ClaimSignalExactTerm,
    ClaimSignalExtractionResult,
    ClaimSignalFacet,
    ClaimTextExtractionClaim,
    ClaimTextExtractionResult,
)
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.mock import HashEmbeddingProvider
from trajpatch.providers.structured_outputs import (
    EpisodicExtractionResult,
    EpisodicMemoryPayload,
)
from trajpatch.storage.models import ClaimRecord
from trajpatch.types import LLMResponse, ModelInfo, NormalizedMessage, StructuredLLMResponse, StructuredTaskSpec
from trajpatch.utils.json_utils import write_json


def _sample_medmt_root(root: Path) -> Path:
    root.mkdir()
    payload = [
        {
            "id": "sample-a",
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


def _text_task_response(task: str, prompt: str) -> str:
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
            raw_ids = re.findall(r"(sample-a-m\d{4})", prompt)
            return (
                "HAS_CLAIMS: true\n"
                "REASON: extracted smoking claim\n\n"
                "[CLAIMS]\n"
                f"- status=active | source_message_ids={raw_ids[-2]} | supporting_quote=I smoke 5 cigarettes a day. | text=User smokes 5 cigarettes a day."
            )
        if "Actually I never smoke." in prompt:
            raw_ids = re.findall(r"(sample-a-m\d{4})", prompt)
            return (
                "HAS_CLAIMS: true\n"
                "REASON: extracted smoking denial\n\n"
                "[CLAIMS]\n"
                f"- status=contradictory | source_message_ids={raw_ids[-2]} | supporting_quote=Actually I never smoke. | text=User says they never smoke."
            )
        return "HAS_CLAIMS: false\nREASON: no claim"
    if task == "claim_signal_extract":
        claim_ids = re.findall(r"^### ([^\n]+)$", prompt, flags=re.MULTILINE)
        claim_id = claim_ids[0] if claim_ids else "tmp-c1"
        return (
            "[EXACT_TERMS]\n"
            f"- surface=smoke | category=health_behavior | source_claim_id={claim_id} | source_message_ids=none\n\n"
            "[FACETS]\n"
            f"- relation=health_behavior | value=smoke | entity=User | value_span=smoke | source_claim_id={claim_id} | source_message_ids=none\n\n"
            "[DISPLAY_ITEMS]\n"
            "- smoke\n\n"
            "[DISPLAY_NAMED_ENTITIES]\n"
            "- none\n\n"
            "[DISPLAY_COUNTS]\n"
            "- none\n\n"
            "[DISPLAY_KEY_FACTS]\n"
            "- User smokes 5 cigarettes a day."
        )
    if task == "trajectory_match":
        return "DECISION: CONTINUE\nSELECTED_CANDIDATE: T1\nRATIONALE: This continues the same smoking trajectory."
    if task == "claim_transition_judge":
        return "DECISION: ADD\nSELECTED_CANDIDATE: none\nRATIONALE: The claim is treated as a new addition."
    if task == "trajectory_retrieval_summary":
        return (
            "## Profile / Stable Facts\n"
            "- Smoking history is contradictory.\n\n"
            "## Item Sets / Named Entities\n"
            "- None recorded.\n\n"
            "## Relations / Temporal Updates\n"
            "- A later exchange denies earlier smoking history.\n\n"
            "## Conflicts / Uncertainty\n"
            "- Smoking status is contradictory."
        )
    if task == "wiki_page_plan":
        return (
            "## Pages\n"
            "- page_type=index | title=Index | slug=index | trajectories=epi-sample-a-001 | entities=none | links=none\n"
            "- page_type=topic | title=Smoking Topic | slug=smoking-topic | trajectories=epi-sample-a-001 | entities=none | links=index"
        )
    if task == "wiki_page_compile":
        return (
            "## Overview\n"
            "- Smoking-related wiki page.\n\n"
            "## Key Facts\n"
            "- Earlier smoking history is contradicted by a later denial.\n\n"
            "## Items / Counts\n"
            "- None.\n\n"
            "## Linked Trajectories\n"
            "- epi-sample-a-001\n\n"
            "## Conflicts / Uncertainty\n"
            "- Smoking status is contradictory."
        )
    if task in {"wiki_page_rerank", "trajectory_set_rerank"}:
        final_count = int(re.search(r"Select exactly (\d+) candidates", prompt).group(1))
        labels = re.findall(r"^### ([A-Z]\d+)$", prompt, flags=re.MULTILINE)
        selected = labels[:final_count]
        rationale_lines = "\n".join(f"- {label}: Keep current order." for label in selected)
        return f"SELECTED: {', '.join(selected)}\nRATIONALES:\n{rationale_lines}"
    if task == "answer_generation":
        return "The memory is contradictory: one message says you smoke 5 cigarettes a day, and a later message says you never smoke."
    if task == "medmt_judge":
        return "CORRECT"
    return "NO_MEMORY"


class _LocalBatchProvider(LLMProvider):
    def __init__(self) -> None:
        self.batch_calls: list[int] = []

    def generate(self, messages, *, system_prompt=None, metadata=None):
        task = str((metadata or {}).get("task") or "")
        prompt = messages[-1].content if messages else ""
        text = _text_task_response(task, prompt)
        return LLMResponse(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            metadata={"estimated_usage": True},
        )

    def generate_batch(self, batch_messages, *, system_prompt=None, metadata=None):
        self.batch_calls.append(len(batch_messages))
        responses = []
        for index, messages in enumerate(batch_messages):
            response = self.generate(messages, system_prompt=system_prompt, metadata=metadata)
            response.metadata = {
                **dict(response.metadata),
                "batched": True,
                "batch_size": len(batch_messages),
                "batch_index": index,
                "batch_wall_time_ms": 1.0,
                "serialized_prompt": messages[-1].content if messages else "",
                "prompt_text": messages[-1].content if messages else "",
                "messages": [message.content for message in messages],
                "source_text": "unsafe source text should not persist",
            }
            responses.append(response)
        return responses

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="local", model_name="local-batch-test", is_remote=False)


def _metadata_payloads(database_path: Path) -> list[dict]:
    payloads: list[dict] = []
    with sqlite3.connect(database_path) as connection:
        for table in ("trajectories", "episodic_snapshots", "claims"):
            for (raw_metadata,) in connection.execute(f"SELECT metadata_json FROM {table}"):
                if not raw_metadata:
                    continue
                payloads.append(json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata)
    return payloads


def _assert_no_unsafe_metadata_keys(value) -> None:
    unsafe_keys = {
        "serialized_prompt",
        "prompt",
        "prompt_text",
        "messages",
        "raw",
        "retrieved_context",
        "source_text",
        "markdown_text",
    }
    if isinstance(value, dict):
        assert not (set(value) & unsafe_keys)
        for nested in value.values():
            _assert_no_unsafe_metadata_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_unsafe_metadata_keys(nested)


class _RemoteStructuredProvider(_LocalBatchProvider):
    def __init__(self) -> None:
        super().__init__()
        self.structured_calls: list[str] = []

    def supports_structured(self, task: str) -> bool:
        return task in {"episodic_extract", "episodic_claim_text_extract", "claim_signal_extract"}

    def generate_structured(
        self,
        messages: list[NormalizedMessage],
        *,
        spec: StructuredTaskSpec,
        system_prompt: str | None = None,
        metadata: dict | None = None,
    ) -> StructuredLLMResponse:
        self.structured_calls.append(spec.task)
        prompt = messages[-1].content if messages else ""
        raw_ids = re.findall(r"(sample-a-m\d{4})", prompt)
        if spec.task == "episodic_claim_text_extract":
            if "I smoke 5 cigarettes a day." in prompt:
                parsed = ClaimTextExtractionResult(
                    has_claims=True,
                    claims=[
                        ClaimTextExtractionClaim(
                            status="active",
                            source_message_ids=[raw_ids[-2]],
                            supporting_quote="I smoke 5 cigarettes a day.",
                            text="User smokes 5 cigarettes a day.",
                        )
                    ],
                )
            elif "Actually I never smoke." in prompt:
                parsed = ClaimTextExtractionResult(
                    has_claims=True,
                    claims=[
                        ClaimTextExtractionClaim(
                            status="contradictory",
                            source_message_ids=[raw_ids[-2]],
                            supporting_quote="Actually I never smoke.",
                            text="User says they never smoke.",
                        )
                    ],
                )
            else:
                parsed = ClaimTextExtractionResult(has_claims=False, claims=[], reason="No claim.")
        elif spec.task == "claim_signal_extract":
            claim_ids = re.findall(r"^### ([^\n]+)$", prompt, flags=re.MULTILINE)
            parsed = ClaimSignalExtractionResult(
                exact_terms=[
                    ClaimSignalExactTerm(
                        surface="smoke",
                        category="health_behavior",
                        source_claim_id=claim_ids[0] if claim_ids else "tmp-c1",
                        source_message_ids=[],
                    )
                ],
                facets=[
                    ClaimSignalFacet(
                        relation="health_behavior",
                        value="smoke",
                        entity="User",
                        value_span="smoke",
                        source_claim_id=claim_ids[0] if claim_ids else "tmp-c1",
                        source_message_ids=[],
                    )
                ],
                display_items=["smoke"],
                display_key_facts=["User smokes 5 cigarettes a day."],
            )
        elif "I smoke 5 cigarettes a day." in prompt:
            parsed = EpisodicExtractionResult(
                has_memory=True,
                memory=EpisodicMemoryPayload(
                    summary_content="The user reports a smoking habit.",
                    context="Health behavior disclosure.",
                    keywords=["smoking", "cigarettes"],
                ),
                reason=None,
            )
        elif "Actually I never smoke." in prompt:
            parsed = EpisodicExtractionResult(
                has_memory=True,
                memory=EpisodicMemoryPayload(
                    summary_content="The user denies smoking.",
                    context="Contradictory update to prior health behavior.",
                    keywords=["smoking", "contradiction"],
                ),
                reason=None,
            )
        else:
            parsed = EpisodicExtractionResult(has_memory=False, memory=None, reason="No memory.")
        return StructuredLLMResponse(
            parsed=parsed,
            prompt_tokens=len(prompt.split()),
            completion_tokens=1,
            metadata={
                "structured_vendor": "openai",
                "structured_strategy": "openai_json_schema",
                "structured_success": True,
                "structured_fallback_used": False,
            },
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider_kind="remote", model_name="remote-structured-test", is_remote=True, metadata={"vendor": "openai"})


class _OpenAICompatibleStructuredProvider(_RemoteStructuredProvider):
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_kind="openai-compatible",
            model_name="qwen3-8b",
            is_remote=True,
            metadata={
                "vendor": "openai-compatible",
                "base_url": "http://localhost:8000/v1",
                "structured_mode": "vllm",
            },
        )


class _RemoteStructuredSignalFailureProvider(_RemoteStructuredProvider):
    def generate_structured(
        self,
        messages: list[NormalizedMessage],
        *,
        spec: StructuredTaskSpec,
        system_prompt: str | None = None,
        metadata: dict | None = None,
    ) -> StructuredLLMResponse:
        if spec.task == "claim_signal_extract":
            self.structured_calls.append(spec.task)
            raise StructuredOutputError(
                "invalid schema for claim_signal_extract",
                vendor="openai",
                strategy="openai_json_schema",
            )
        return super().generate_structured(
            messages,
            spec=spec,
            system_prompt=system_prompt,
            metadata=metadata,
        )


def test_memory_extract_batch_size_runs_local_text_batch(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="local",
        backbone_model="local-batch-test",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        memory_extract_batch_size=2,
    )
    provider = _LocalBatchProvider()
    runner = PipelineRunner(
        config,
        llm_provider=provider,
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
    )
    runner.run()

    assert runner._episodic_batch_mode() == "local_text_batch"
    assert provider.batch_calls == [2, 2, 2]
    assert runner.episodic_batch_call_count == 1
    assert runner.episodic_batch_item_count == 2
    assert runner.episodic_batch_effective_size_final == 2
    assert runner.memory_stage_batch_stats["episodic_extract"] == {
        "calls": 1,
        "items": 2,
        "fallbacks": 0,
        "structured_failures": 0,
    }
    assert runner.memory_stage_batch_stats["claim_text"] == {
        "calls": 1,
        "items": 2,
        "fallbacks": 0,
        "structured_failures": 0,
    }
    assert runner.memory_stage_batch_stats["claim_signal"] == {
        "calls": 1,
        "items": 2,
        "fallbacks": 0,
        "structured_failures": 0,
    }
    payloads = _metadata_payloads(config.database_path)
    for payload in payloads:
        _assert_no_unsafe_metadata_keys(payload)
    response_metadata_entries = [
        payload["claim_text_llm_response_metadata_v1"]
        for payload in payloads
        if "claim_text_llm_response_metadata_v1" in payload
    ]
    assert response_metadata_entries
    assert all(entry["batch_mode"] == "local_text_batch" for entry in response_metadata_entries)
    assert {entry["batch_index"] for entry in response_metadata_entries} == {0, 1}
    assert all(entry["batch_size"] == 2 for entry in response_metadata_entries)


def test_memory_extract_batch_size_runs_remote_structured_concurrent(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="remote",
        backbone_model="gpt-4o-mini",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        memory_extract_batch_size=2,
    )
    provider = _RemoteStructuredProvider()
    runner = PipelineRunner(
        config,
        llm_provider=provider,
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
    )
    runner.run()

    assert runner._episodic_batch_mode() == "structured_concurrent"
    assert Counter(provider.structured_calls) == Counter(
        {
            "episodic_extract": 2,
            "episodic_claim_text_extract": 2,
            "claim_signal_extract": 2,
        }
    )
    assert runner.episodic_batch_call_count == 1
    assert runner.episodic_batch_item_count == 2
    assert runner.episodic_batch_effective_size_final == 2
    assert runner.memory_stage_batch_stats["episodic_extract"] == {
        "calls": 1,
        "items": 2,
        "fallbacks": 0,
        "structured_failures": 0,
    }
    assert runner.memory_stage_batch_stats["claim_text"] == {
        "calls": 1,
        "items": 2,
        "fallbacks": 0,
        "structured_failures": 0,
    }
    assert runner.memory_stage_batch_stats["claim_signal"] == {
        "calls": 1,
        "items": 2,
        "fallbacks": 0,
        "structured_failures": 0,
    }

    claims = runner.store.session.query(ClaimRecord).all()
    assert claims
    for claim in claims:
        metadata = dict(claim.metadata_json or {})
        assert "exact_terms_v1" not in metadata
        assert "facets_v1" not in metadata
        if metadata.get("facets_v2"):
            assert all(facet["source_claim_id"].startswith("epi-sample-a-") for facet in metadata["facets_v2"])
            assert not any(facet["source_claim_id"].startswith("tmp-") for facet in metadata["facets_v2"])
    trajectories = runner.store.list_trajectories("sample-a")
    latest_snapshot = runner.store.latest_snapshot(trajectories[0].id)
    latest_claims = runner.store.list_claims_for_snapshot(latest_snapshot.id)
    carried_claims = [claim for claim in latest_claims if claim.claim_id.endswith("c001")]
    assert carried_claims
    assert (carried_claims[0].metadata_json or {}).get("exact_terms_v2")
    assert "exact_terms_v1" not in (carried_claims[0].metadata_json or {})
    assert "facets_v1" not in (carried_claims[0].metadata_json or {})


def test_openai_compatible_structured_provider_uses_remote_like_concurrent_memory_mode(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        memory_extract_batch_size=2,
    )
    provider = _OpenAICompatibleStructuredProvider()
    runner = PipelineRunner(
        config,
        llm_provider=provider,
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
    )
    runner.run()

    assert runner._episodic_batch_mode() == "structured_concurrent"
    assert Counter(provider.structured_calls) == Counter(
        {
            "episodic_extract": 2,
            "episodic_claim_text_extract": 2,
            "claim_signal_extract": 2,
        }
    )
    assert runner.memory_stage_batch_stats["episodic_extract"]["calls"] == 1
    assert runner.episodic_batch_effective_size_final == 2


def test_openai_compatible_vllm_autostart_metadata_is_recorded(monkeypatch, tmp_path: Path):
    instances = []

    class _FakeManagedVLLMServer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.status_dir = kwargs["status_dir"]
            instances.append(self)

        def start(self):
            self.started = True
            self.status_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.status_dir / "vllm_server.json", self.metadata())

        def stop(self):
            self.stopped = True
            write_json(self.status_dir / "vllm_server.json", self.metadata())

        def metadata(self):
            return {
                "vllm_autostart": True,
                "vllm_reused_existing_server": False,
                "vllm_started_by_runner": True,
                "vllm_model": self.kwargs["model"],
                "vllm_served_model_name": self.kwargs["served_model_name"],
                "vllm_base_url": self.kwargs["base_url"],
                "vllm_pid": 4321,
                "vllm_startup_latency_ms": 12.5,
                "vllm_keep_alive": self.kwargs["keep_alive"],
            }

    monkeypatch.setattr("trajpatch.pipeline.runner.ManagedVLLMServer", _FakeManagedVLLMServer)
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        memory_extract_batch_size=2,
        vllm_autostart=True,
        vllm_model="Qwen/Qwen3-8B",
        vllm_served_model_name="qwen3-8b",
    )
    runner = PipelineRunner(
        config,
        llm_provider=_OpenAICompatibleStructuredProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
    )
    report = runner.run()

    assert len(instances) == 1
    assert instances[0].started is True
    assert instances[0].stopped is True
    run_meta = report.details["run_meta"]
    assert run_meta["vllm_autostart"] is True
    assert run_meta["vllm_started_by_runner"] is True
    assert run_meta["vllm_model"] == "Qwen/Qwen3-8B"
    assert run_meta["vllm_served_model_name"] == "qwen3-8b"
    assert (runner.run_dir / "status" / "vllm_server.json").exists()
    assert (runner.run_dir / "status" / "cuda_preflight.json").exists()
    assert run_meta["cuda_preflight_mode"] == "warn"
    assert "cuda_preflight_risk" in run_meta


def test_strict_cuda_preflight_failure_writes_run_failed(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "trajpatch.providers.cuda_preflight.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "GPU-0",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 76 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            }
        ],
    )
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        vllm_autostart=True,
        cuda_preflight_mode="strict",
    )
    runner = PipelineRunner(
        config,
        llm_provider=_OpenAICompatibleStructuredProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
    )

    try:
        runner.run()
    except ProviderConfigurationError:
        pass
    else:
        raise AssertionError("strict CUDA preflight should fail before benchmark execution")

    assert runner.run_dir is not None
    failed_path = runner.run_dir / "run_failed.json"
    assert failed_path.exists()
    failed_payload = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failed_payload["stage"] == "cuda_preflight"
    assert failed_payload["cuda_preflight"]["risk"] == "fail"
    assert (runner.run_dir / "status" / "cuda_preflight.json").exists()


def test_remote_structured_claim_signal_failure_is_counted_and_logged(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="remote",
        backbone_model="gpt-4o-mini",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        memory_extract_batch_size=2,
        verbose=True,
    )
    provider = _RemoteStructuredSignalFailureProvider()
    stream = io.StringIO()
    runner = PipelineRunner(
        config,
        llm_provider=provider,
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
        console=Console(file=stream, force_terminal=False, color_system=None, width=240),
    )

    runner.run()

    assert runner.memory_stage_batch_stats["claim_signal"]["structured_failures"] == 2
    output = stream.getvalue()
    assert "structured_schema_static_check task=claim_signal_extract vendor=openai ok=true" in output
    assert "structured_failure=true error_type=StructuredOutputError" in output
    assert "invalid schema for claim_signal_extract" in output
    assert "claim_signal_batch_done mode=structured_concurrent request_successes=0 request_failures=2" in output


def test_memory_extract_batch_size_remote_structured_auto_defaults_to_8(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="remote",
        backbone_model="gpt-4o-mini",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        memory_extract_batch_size="auto",
    )
    runner = PipelineRunner(
        config,
        llm_provider=_RemoteStructuredProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
    )

    assert runner._resolve_memory_extract_batch_size() == 8


def test_remote_structured_auto_batch_is_worker_aware_and_judge_auto_is_conservative(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="remote",
        judge_provider_kind="remote",
        backbone_model="gpt-4o-mini",
        embedding_model="hash-embedding",
        judge_model="gpt-4o-mini",
        memory_extract_batch_size="auto",
        judge_max_concurrency="auto",
        conv_workers=2,
    )
    runner = PipelineRunner(
        config,
        llm_provider=_RemoteStructuredProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_RemoteStructuredProvider(),
    )

    assert runner._resolve_memory_extract_batch_size() == 4
    assert runner._resolve_judge_parallelism() == 2

    explicit_config = config.copy(update={"memory_extract_batch_size": 3, "judge_max_concurrency": 4})
    explicit_runner = PipelineRunner(
        explicit_config,
        llm_provider=_RemoteStructuredProvider(),
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_RemoteStructuredProvider(),
    )
    assert explicit_runner._resolve_memory_extract_batch_size() == 3
    assert explicit_runner._resolve_judge_parallelism() == 4


def test_memory_extract_batch_size_one_uses_serial_memory_stages(tmp_path: Path):
    config = RunConfig(
        dataset="medmt",
        dataset_path=_sample_medmt_root(tmp_path / "medmt"),
        output_dir=tmp_path / "artifacts",
        database_path=tmp_path / "artifacts" / "run.sqlite",
        memory_cache_enabled=False,
        provider_kind="local",
        backbone_model="local-batch-test",
        embedding_model="hash-embedding",
        judge_model="mock-judge",
        m=2,
        t_pages=2,
        k=1,
        memory_extract_batch_size=1,
    )
    provider = _LocalBatchProvider()
    runner = PipelineRunner(
        config,
        llm_provider=provider,
        embedding_provider=HashEmbeddingProvider(),
        judge_provider=_LocalBatchProvider(),
    )
    runner.run()

    assert provider.batch_calls == []
    assert runner.episodic_batch_call_count == 0
    assert runner.episodic_batch_item_count == 0
    assert runner.memory_stage_batch_stats == {}
