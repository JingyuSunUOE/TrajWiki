"""Top-level offline benchmark runner."""

from __future__ import annotations

import os
import json
import re
import shutil
import sqlite3
import tempfile
import time
import traceback
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock, Thread
from typing import Any, Iterable

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sqlalchemy import delete, func, inspect, select

from trajpatch.cache import MemoryCacheManager
from trajpatch.config import RunConfig
from trajpatch.datasets import build_dataset_adapter
from trajpatch.datasets.locomo import LocomoAdapter
from trajpatch.datasets.medmt import MedMTAdapter
from trajpatch.diagnostics.fallback_repair import (
    build_fallback_repair_events_for_row,
    summarize_events_for_query,
    summarize_fallback_repair_events,
)
from trajpatch.analysis.text_only_filter import (
    audit_text_only_visibility,
    compact_filter_for_details,
    manifest_entry_for_row,
    summarize_text_only_filter,
)
from trajpatch.analysis.auditability import write_auditability_artifacts
from trajpatch.analysis.cost_benefit import write_cost_diagnostic_artifacts
from trajpatch.analysis.gold_labels import write_gold_labels_artifact
from trajpatch.exceptions import ParserValidationError, ProviderConfigurationError
from trajpatch.memory import MemoryOrchestrator, RetrievalEngine
from trajpatch.memory.wiki import WikiCompiler
from trajpatch.memory.orchestrator import PrecomputedGenerationAttempt, StructuredFirstPassAttempt
from trajpatch.prompts import load_prompt
from trajpatch.providers import MeteredLLMProvider, build_provider_bundle
from trajpatch.providers.metering import phase_for_task, summarize_llm_calls
from trajpatch.providers.devices import build_worker_embedding_device_plans, infer_parameter_size_b
from trajpatch.providers.cuda_preflight import CUDAPreflightReport, run_cuda_preflight
from trajpatch.providers.vllm_server import ManagedVLLMServer
from trajpatch.providers.structured_outputs import (
    get_structured_task_spec,
    structured_schema_diagnostics,
    vendor_schema,
)
from trajpatch.storage.database import create_index_schema, create_schema
from trajpatch.storage.models import (
    AggregateMetricRecord,
    AnswerRecord,
    Base,
    ClaimOpRecord,
    ClaimRecord,
    EpisodicMemorySnapshot,
    EvaluationRecord,
    IndexedAggregateMetricRecord,
    IndexedRunRecord,
    RawMessageRecord,
    RetrievalEvent,
    RunMetaRecord,
    TrajectoryRecord,
)
from trajpatch.storage.repository import TrajWikiStore
from trajpatch.types import (
    AnswerResult,
    DatasetSample,
    DevicePlan,
    JudgeResult,
    LLMResponse,
    NormalizedMessage,
    QueryTask,
    RetrievalBundle,
    RunReport,
)
from trajpatch.utils.json_utils import append_jsonl, write_json
from trajpatch.utils.metrics import bleu1

from .answering import AnswerGenerator, BenchmarkJudge
from .exporter import ArtifactExporter
from .semantic_metrics import SemanticMetricEvaluator, SemanticMetricResult

LOCOMO_SUBSET_ORDER = ["multi_hop", "temporal", "open_domain", "single_hop"]
MEDMT_SUBSET_ORDER = [
    "long_context_memory_and_understanding",
    "resistance_to_contextual_interference",
    "information_contradiction",
]
MEDMT_SCENE_ORDER = ["HC", "PSRO", "PTRM"]
MEDMT_SUBSET_LABELS = {
    "long_context_memory_and_understanding": "Long-Context Memory and Understanding",
    "resistance_to_contextual_interference": "Resistance to Contextual Interference",
    "information_contradiction": "Information Contradiction",
}
REMOTE_LIKE_PROVIDER_KINDS = {"remote", "openai-compatible"}


@dataclass(slots=True)
class LogicalSampleGroup:
    group_id: str
    memory_sample: DatasetSample
    query_rows: list[DatasetSample]
    excluded_rows: list[DatasetSample]


@dataclass(slots=True)
class PipelineShard:
    worker_id: int
    groups: list[LogicalSampleGroup]
    query_count: int
    embedding_device_plan: DevicePlan | None = None
    database_path: Path | None = None
    database_root: Path | None = None
    local_scratch_used: bool = False


@dataclass(slots=True)
class PipelineShardResult:
    worker_id: int
    group_ids: list[str]
    query_count: int
    answer_results_by_group: dict[str, list[AnswerResult]]
    database_path: Path
    runtime_ms: float
    cache_stats: dict[str, float]
    memory_stage_batch_stats: dict[str, dict[str, int]]
    episodic_batch_call_count: int
    episodic_batch_item_count: int
    episodic_batch_repair_after_batch_count: int
    episodic_batch_oom_backoff_count: int
    episodic_batch_effective_size_max: int
    episodic_batch_effective_size_final: int
    orchestrator_counters: dict[str, Any]
    structured_call_metadata: list[dict[str, Any]]
    provider_call_records: list[dict[str, Any]]
    device_allocation: dict[str, Any]


class SemaphoreLimitedLLMProvider:
    """Limit concurrent remote backbone calls across threaded shard workers."""

    def __init__(self, provider, semaphore: BoundedSemaphore) -> None:
        self.provider = provider
        self.semaphore = semaphore

    def generate(self, *args, **kwargs):
        with self.semaphore:
            return self.provider.generate(*args, **kwargs)

    def generate_batch(self, *args, **kwargs):
        with self.semaphore:
            return self.provider.generate_batch(*args, **kwargs)

    def generate_structured(self, *args, **kwargs):
        with self.semaphore:
            return self.provider.generate_structured(*args, **kwargs)

    def supports_structured(self, task: str) -> bool:
        return self.provider.supports_structured(task)

    def model_info(self):
        return self.provider.model_info()

    def snapshot(self) -> int:
        return self.provider.snapshot() if hasattr(self.provider, "snapshot") else 0

    def diff(self, start_index: int, end_index: int | None = None) -> dict[str, Any]:
        if hasattr(self.provider, "diff"):
            return self.provider.diff(start_index, end_index)
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0, "records": []}

    def __getattr__(self, name: str):
        return getattr(self.provider, name)


ORCHESTRATOR_COUNTER_FIELDS = [
    "parse_attempts",
    "parse_successes",
    "parse_failures",
    "repair_rounds",
    "extraction_fallbacks",
    "closed_on_fallback",
    "trajectory_match_total_open",
    "trajectory_match_prefiltered",
    "trajectory_match_shortlisted",
    "link_salvage_count",
    "link_exchange_fallback_count",
    "ops_parse_failure_count",
    "ops_ignored_count",
    "ops_synthesized_count",
    "ops_model_hint_count",
    "ops_model_supplied_count",
    "claims_parse_failure_count",
    "claims_required_repair_count",
    "claim_text_exact_match_count",
    "claim_status_updated_count",
    "claim_new_add_count",
    "claim_unmatched_previous_count",
    "claim_transition_judge_attempt_count",
    "claim_transition_judge_success_count",
    "claim_transition_judge_fallback_count",
    "claim_transition_revise_count",
    "claim_transition_add_count",
    "empty_repair_target_count",
    "forced_memory_seed_count",
    "low_salience_memory_count",
    "llm_no_memory_forced_count",
    "zero_claim_episodic_candidate_count",
    "zero_claim_episodic_persisted_count",
    "zero_claim_low_salience_skipped_count",
    "structured_attempts",
    "structured_successes",
    "structured_fallbacks",
]

ORCHESTRATOR_COUNTER_MAP_FIELDS = [
    "structured_attempts_by_task",
    "structured_successes_by_task",
    "structured_fallbacks_by_task",
    "structured_attempts_by_vendor",
    "structured_successes_by_vendor",
    "structured_fallbacks_by_vendor",
]

SHARD_MERGE_TABLES = [
    "raw_messages",
    "trajectories",
    "episodic_snapshots",
    "wiki_pages",
    "claims",
    "claim_ops",
    "embeddings",
    "retrieval_events",
    "answers",
]

EVENT_LOG_LOCK = Lock()
HEARTBEAT_INTERVAL_S = 45.0


class PipelineRunner:
    def __init__(
        self,
        config: RunConfig,
        *,
        llm_provider=None,
        embedding_provider=None,
        judge_provider=None,
        console: Console | None = None,
        device_plan_overrides: dict[str, DevicePlan | None] | None = None,
    ):
        self.config = config
        self.console = console or Console()
        self.adapter = build_dataset_adapter(config.dataset)
        self.device_allocation: dict[str, Any] = {}
        self.cuda_preflight_report: CUDAPreflightReport = run_cuda_preflight(
            config,
            raise_on_strict=False,
        )
        preflight_overrides = dict(self.cuda_preflight_report.device_plan_overrides)
        preflight_overrides.update(device_plan_overrides or {})
        if llm_provider is None or embedding_provider is None or judge_provider is None:
            built_llm, built_judge, built_embedding, self.device_allocation = build_provider_bundle(
                config,
                device_plan_overrides=preflight_overrides,
            )
        else:
            built_llm, built_judge, built_embedding = llm_provider, judge_provider, embedding_provider
        self.llm_provider = self._ensure_metered(llm_provider or built_llm, role="backbone")
        self.embedding_provider = embedding_provider or built_embedding
        self.judge_provider = self._ensure_metered(judge_provider or built_judge, role="judge")
        if self.judge_provider is None:
            raise ProviderConfigurationError("Benchmark runner requires a judge model/provider.")
        self.cache = MemoryCacheManager(config, self.adapter)
        self.session_factory = None
        self.session = None
        self.store = None
        self.orchestrator = None
        self.wiki = None
        self.retrieval = None
        self.answering = AnswerGenerator(self.llm_provider, trace=self._trace if self.config.verbose else None)
        self.judge = BenchmarkJudge(self.judge_provider, trace=self._trace if self.config.verbose else None)
        self.semantic_metrics = SemanticMetricEvaluator(
            self.judge_provider,
            self.config,
            trace=self._trace if self.config.verbose else None,
        )
        self.exporter = None
        self.cache_stats = self._empty_cache_stats()
        self.run_started_at: str | None = None
        self.run_dir: Path | None = None
        self.run_id: str | None = None
        self.dataset_scope_key, self.dataset_scope = self._resolve_dataset_scope()
        self.excluded_samples: list[dict[str, Any]] = []
        self.selected_logical_sample_count = 0
        self.selected_row_count = 0
        self.selected_query_count = 0
        self.selection_strategy = "none"
        self.selected_counts_by_subset: dict[str, int] = {}
        self.judge_parallelism = self._resolve_judge_parallelism()
        self.judge_total_wall_time_ms = 0.0
        self.judge_queue_latency_ms_total = 0.0
        self.judge_queue_events = 0
        self.worker_id: int | None = None
        self.worker_shards: list[dict[str, Any]] = []
        self.worker_device_allocation: dict[str, Any] = {}
        self.worker_structured_call_metadata: list[dict[str, Any]] = []
        self.worker_provider_call_records: list[dict[str, Any]] = []
        self.worker_database_root: str | None = None
        self.worker_database_local_scratch_used = False
        self.worker_database_paths: list[dict[str, Any]] = []
        self.worker_database_cleanup: dict[str, Any] = {"attempted": False, "cleaned_paths": [], "kept_paths": []}
        self.worker_database_warnings: list[str] = []
        self.failed_shard_paths: list[str] = []
        self.failed_shard_copy_errors: list[str] = []
        self.backbone_inflight_semaphore: BoundedSemaphore | None = None
        self.memory_extract_batch_size_cap: int | None = None
        self.episodic_batch_call_count = 0
        self.episodic_batch_item_count = 0
        self.episodic_batch_repair_after_batch_count = 0
        self.episodic_batch_oom_backoff_count = 0
        self.episodic_batch_effective_size_max = 0
        self.episodic_batch_effective_size_final = 1
        self.memory_stage_batch_stats: dict[str, dict[str, int]] = {}
        self._structured_schema_checks_logged: set[tuple[str, str]] = set()
        self._event_counter = 0
        self._recent_lifecycle_events: list[dict[str, Any]] = []
        self._current_stage = "initialized"
        self._current_sample_id: str | None = None
        self._current_query_task_id: str | None = None
        self._current_stage_started_at = time.perf_counter()
        self._current_query_started_at = time.perf_counter()
        self._heartbeat_stop: Event | None = None
        self._heartbeat_thread: Thread | None = None
        self._heartbeat_started_at = time.perf_counter()
        self._last_heartbeat_event_count = 0
        self.fallback_repair_events: list[dict[str, Any]] = []
        self.text_only_filter_manifest: list[dict[str, Any]] = []
        self.text_only_filter_summary: dict[str, Any] = {}
        self.managed_vllm_server: ManagedVLLMServer | None = None
        self.vllm_server_metadata: dict[str, Any] = {}

    def _trace(self, message: str) -> None:
        if not self.config.verbose:
            return
        if self.worker_id is not None:
            message = f"worker={self.worker_id:02d} {message}"
        self.console.print(f"[dim][trace][/dim] {message}")

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, BaseException):
            return f"{value.__class__.__name__}: {PipelineRunner._compact_error_text(value, limit=500)}"
        if isinstance(value, dict):
            return {str(key): PipelineRunner._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [PipelineRunner._json_safe(item) for item in value]
        return str(value)

    def _status_dir(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "status"

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        self._event_counter += 1
        event = {
            "schema_version": "run_event_v1",
            "event_index": self._event_counter,
            "event_id": f"{self.worker_id if self.worker_id is not None else 'main'}-{self._event_counter:06d}",
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": self.worker_id,
            "stage": payload.get("stage") or event_type,
            **payload,
        }
        event = self._json_safe(event)
        now = time.perf_counter()
        previous_stage = self._current_stage
        previous_sample = self._current_sample_id
        previous_query = self._current_query_task_id
        self._current_stage = str(event.get("stage") or event_type)
        if event.get("sample_id") is not None:
            next_sample = str(event.get("sample_id"))
            self._current_sample_id = next_sample
            if next_sample != previous_sample and event.get("query_task_id") is None:
                self._current_query_task_id = None
        if event.get("query_task_id") is not None:
            self._current_query_task_id = str(event.get("query_task_id"))
        self._set_meter_call_context(
            sample_id=self._current_sample_id,
            query_task_id=self._current_query_task_id,
        )
        self._recent_lifecycle_events.append(event)
        self._recent_lifecycle_events = self._recent_lifecycle_events[-50:]
        if self._current_stage != previous_stage:
            self._current_stage_started_at = now
        status_dir = self._status_dir()
        if status_dir is not None:
            try:
                with EVENT_LOG_LOCK:
                    append_jsonl(status_dir / "events.jsonl", [event])
            except Exception as exc:  # noqa: BLE001
                self._trace(f"event_log_write_failed event={event_type} error={exc.__class__.__name__}: {exc}")
        if self._current_query_task_id != previous_query:
            self._current_query_started_at = now

    def _cuda_preflight_payload(self) -> dict[str, Any]:
        return self._json_safe(self.cuda_preflight_report.to_dict())

    def _write_cuda_preflight_report(self) -> None:
        if self.run_dir is None or not bool(self.config.cuda_preflight_report):
            return
        try:
            write_json(self.run_dir / "status" / "cuda_preflight.json", self._cuda_preflight_payload())
        except Exception as exc:  # noqa: BLE001
            self._trace(
                f"cuda_preflight_report_write_failed error={exc.__class__.__name__}: {exc}"
            )

    def _emit_cuda_preflight_events(self) -> None:
        report = self.cuda_preflight_report
        self._emit_event(
            "cuda_preflight_start",
            stage="cuda_preflight",
            mode=report.mode,
            enabled=report.enabled,
            visible_device_count=len(report.inventory),
        )
        for reservation in report.reservations:
            self._emit_event(
                "cuda_preflight_reserve",
                stage="cuda_preflight",
                **reservation,
            )
        for assignment in report.assignments:
            self._emit_event(
                "cuda_preflight_assign",
                stage="cuda_preflight",
                **assignment,
            )
        for warning in report.warnings:
            self._emit_event(
                "cuda_preflight_warning",
                stage="cuda_preflight",
                warning=warning,
            )
            self._trace(f"cuda_preflight_warning warning={warning}")
        if report.errors:
            self._emit_event(
                "cuda_preflight_failed",
                stage="cuda_preflight",
                errors=list(report.errors),
                risk=report.risk,
            )
            for error in report.errors:
                self._trace(f"cuda_preflight_error error={error}")
        self._emit_event(
            "cuda_preflight_done",
            stage="cuda_preflight",
            enabled=report.enabled,
            risk=report.risk,
            warning_count=len(report.warnings),
            error_count=len(report.errors),
        )
        self._trace(
            f"cuda_preflight_done enabled={str(report.enabled).lower()} risk={report.risk} "
            f"warnings={len(report.warnings)} errors={len(report.errors)}"
        )

    def _fail_if_cuda_preflight_strict(self) -> None:
        report = self.cuda_preflight_report
        if self.config.cuda_preflight_mode != "strict":
            return
        if not report.errors and report.risk != "fail":
            return
        message = "CUDA preflight failed: " + "; ".join(report.errors or report.warnings)
        self._write_run_failed(
            ProviderConfigurationError(message),
            stage="cuda_preflight",
            extra={"cuda_preflight": self._cuda_preflight_payload()},
        )
        raise ProviderConfigurationError(message)

    def _provider_call_count(self) -> int:
        total = 0
        for provider in self._metered_providers():
            try:
                total += int(provider.diff(0).get("provider_call_count", 0))
            except Exception:
                total += len(getattr(provider, "calls", []))
        return total

    def _heartbeat_loop(self) -> None:
        assert self._heartbeat_stop is not None
        while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL_S):
            now = time.perf_counter()
            run_elapsed_s = now - self._heartbeat_started_at
            stage_elapsed_s = now - self._current_stage_started_at
            query_elapsed = (
                f"{now - self._current_query_started_at:.1f}"
                if self._current_query_task_id is not None
                else "-"
            )
            event_count = self._event_counter
            self._trace(
                "heartbeat "
                f"stage={self._current_stage} sample={self._current_sample_id or '-'} "
                f"query={self._current_query_task_id or '-'} run_elapsed_s={run_elapsed_s:.1f} "
                f"stage_elapsed_s={stage_elapsed_s:.1f} query_elapsed_s={query_elapsed} "
                f"provider_calls={self._provider_call_count()} events={event_count} "
                f"new_events={event_count - self._last_heartbeat_event_count}"
            )
            self._last_heartbeat_event_count = event_count

    def _start_heartbeat(self) -> None:
        if not self.config.verbose or self._heartbeat_thread is not None:
            return
        self._heartbeat_started_at = time.perf_counter()
        self._heartbeat_stop = Event()
        self._heartbeat_thread = Thread(target=self._heartbeat_loop, name="trajpatch-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    @staticmethod
    def _filesystem_snapshot(path: Path | None) -> dict[str, Any]:
        target = path if path is not None else Path.cwd()
        if target.exists() and target.is_file():
            target = target.parent
        try:
            usage = shutil.disk_usage(target)
            free_gb = usage.free / float(1024**3)
            total_gb = usage.total / float(1024**3)
            return {
                "path": str(target),
                "free_bytes": int(usage.free),
                "total_bytes": int(usage.total),
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "path": str(target),
                "free_bytes": None,
                "total_bytes": None,
                "free_gb": None,
                "total_gb": None,
                "error": f"{exc.__class__.__name__}: {exc}",
            }

    @staticmethod
    def _write_probe(path: Path | None) -> dict[str, Any]:
        target = path if path is not None else Path.cwd()
        if target.exists() and target.is_file():
            target = target.parent
        probe_path = target / f".trajpatch-write-probe-{uuid.uuid4().hex[:8]}"
        snapshot = PipelineRunner._filesystem_snapshot(target)
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe_path.write_text("ok", encoding="utf-8")
            probe_path.unlink()
            return {"path": str(target), "ok": True, "filesystem": snapshot}
        except Exception as exc:  # noqa: BLE001
            try:
                if probe_path.exists():
                    probe_path.unlink()
            except Exception:
                pass
            return {
                "path": str(target),
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error_message": PipelineRunner._compact_error_text(exc, limit=300),
                "filesystem": snapshot,
            }

    def _emit_filesystem_probes(self, *, include_worker_root: bool = False) -> None:
        probes = {
            "run_dir": self._write_probe(self.run_dir),
            "run_db_parent": self._write_probe(Path(self.config.database_path).parent if self.config.database_path else None),
            "cache_dir": self._write_probe(self.config.memory_cache_dir),
        }
        if include_worker_root and self.worker_database_root:
            probes["worker_database_root"] = self._write_probe(Path(self.worker_database_root))
        self._emit_event("filesystem_probe", stage="filesystem_probe", probes=probes)

    def _metered_providers(self) -> list[MeteredLLMProvider]:
        providers: list[MeteredLLMProvider] = []
        for provider in (self.llm_provider, self.judge_provider):
            if isinstance(provider, SemaphoreLimitedLLMProvider):
                provider = provider.provider
            if isinstance(provider, MeteredLLMProvider):
                providers.append(provider)
        return providers

    def _recent_provider_failure_records(self, *, limit: int = 20) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for provider in self._metered_providers():
            if hasattr(provider, "sanitized_records"):
                records.extend(provider.sanitized_records(limit=limit, failures_only=True))
        return records[-limit:]

    @staticmethod
    def _sqlite_table_counts(database_path: Path | None) -> dict[str, Any]:
        if database_path is None:
            return {"available": False, "reason": "database_path_missing"}
        if not database_path.exists():
            return {"available": False, "path": str(database_path), "reason": "database_missing"}
        counts: dict[str, Any] = {"available": True, "path": str(database_path), "tables": {}}
        try:
            with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
                table_names = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall()
                ]
                for table_name in table_names:
                    try:
                        counts["tables"][table_name] = int(
                            connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                        )
                    except Exception as exc:  # noqa: BLE001
                        counts["tables"][table_name] = f"{exc.__class__.__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            counts.update(
                {
                    "available": False,
                    "error_type": exc.__class__.__name__,
                    "error_message": PipelineRunner._compact_error_text(exc, limit=300),
                }
            )
        return counts

    def _load_recent_event_log(self, *, limit: int = 50) -> list[dict[str, Any]]:
        status_dir = self._status_dir()
        if status_dir is None:
            return list(self._recent_lifecycle_events[-limit:])
        path = status_dir / "events.jsonl"
        if not path.exists():
            return list(self._recent_lifecycle_events[-limit:])
        events: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
                if not line.strip():
                    continue
                events.append(json.loads(line))
        except Exception:
            return list(self._recent_lifecycle_events[-limit:])
        return events

    def _commit_session(self, *, stage: str, sample_id: str | None = None) -> None:
        assert self.session is not None
        database_path = Path(self.config.database_path) if self.config.database_path is not None else None
        snapshot = self._filesystem_snapshot(database_path)
        try:
            self.session.commit()
        except Exception as exc:  # noqa: BLE001
            try:
                self.session.rollback()
            except Exception:
                pass
            worker_label = f"{self.worker_id:02d}" if self.worker_id is not None else "main"
            free_gb = snapshot.get("free_gb")
            free_text = f"{free_gb}" if free_gb is not None else "unknown"
            self._trace(
                f"db_commit_failed stage={stage} sample={sample_id or '-'} worker={worker_label} "
                f"database={database_path or '-'} free_gb={free_text}"
            )
            self._emit_event(
                "db_commit_failed",
                stage=stage,
                sample_id=sample_id,
                worker_id=self.worker_id,
                database_path=str(database_path) if database_path is not None else None,
                free_gb=free_gb,
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=500),
            )
            raise RuntimeError(
                f"DB commit failed stage={stage} sample={sample_id or '-'} worker={worker_label} "
                f"database={database_path or '-'} worker_db_root={self.worker_database_root or '-'} "
                f"free_gb={free_text}: {exc}"
            ) from exc

    def _resolve_worker_database_root(self) -> tuple[Path, bool, list[str]]:
        assert self.run_id is not None
        assert self.run_dir is not None
        warnings: list[str] = []
        candidates: list[Path] = []
        env_tmp = os.environ.get("TMPDIR")
        if env_tmp:
            candidates.append(Path(env_tmp).expanduser())
        system_tmp = Path(tempfile.gettempdir()).expanduser()
        if all(candidate != system_tmp for candidate in candidates):
            candidates.append(system_tmp)
        for base in candidates:
            root = base / "trajpatch-shards" / self.run_id
            try:
                root.mkdir(parents=True, exist_ok=True)
                probe = root / ".write-test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                return root, True, warnings
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{root}: {exc.__class__.__name__}: {exc}")
        fallback = self.run_dir / "shards"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, False, warnings

    @staticmethod
    def _cleanup_worker_database_files(database_path: Path) -> list[str]:
        removed: list[str] = []
        for candidate in (database_path, Path(f"{database_path}-shm"), Path(f"{database_path}-wal")):
            if not candidate.exists():
                continue
            candidate.unlink()
            removed.append(str(candidate))
        return removed

    def _cleanup_worker_databases(self, shard_results: list[PipelineShardResult]) -> None:
        cleanup: dict[str, Any] = {"attempted": False, "cleaned_paths": [], "kept_paths": []}
        if not shard_results:
            self.worker_database_cleanup = cleanup
            return
        cleanup["attempted"] = True
        if not self.worker_database_local_scratch_used:
            cleanup["kept_paths"] = [str(result.database_path) for result in shard_results]
            self.worker_database_cleanup = cleanup
            return
        cleaned_paths: list[str] = []
        kept_paths: list[str] = []
        for result in shard_results:
            try:
                cleaned_paths.extend(self._cleanup_worker_database_files(result.database_path))
            except Exception as exc:  # noqa: BLE001
                kept_paths.append(str(result.database_path))
                self.worker_database_warnings.append(
                    f"cleanup_failed {result.database_path}: {exc.__class__.__name__}: {exc}"
                )
        cleanup["cleaned_paths"] = cleaned_paths
        cleanup["kept_paths"] = kept_paths
        self.worker_database_cleanup = cleanup
        self._trace(
            f"worker_db_cleanup attempted=true cleaned={len(cleaned_paths)} kept={len(kept_paths)}"
        )

    @staticmethod
    def _copy_sqlite_sidecars(source_path: Path, target_dir: Path) -> tuple[list[str], list[str]]:
        copied: list[str] = []
        errors: list[str] = []
        target_dir.mkdir(parents=True, exist_ok=True)
        for candidate in (source_path, Path(f"{source_path}-wal"), Path(f"{source_path}-shm")):
            if not candidate.exists():
                continue
            target = target_dir / candidate.name
            try:
                shutil.copy2(candidate, target)
                copied.append(str(target))
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{candidate}: {exc.__class__.__name__}: {PipelineRunner._compact_error_text(exc, limit=300)}"
                )
        return copied, errors

    def _preserve_failed_worker_shard(self, shard: PipelineShard, exc: BaseException) -> None:
        if self.run_dir is None or shard.database_path is None:
            return
        target_dir = self.run_dir / "failed_shards" / f"worker-{shard.worker_id:02d}"
        copied, errors = self._copy_sqlite_sidecars(shard.database_path, target_dir)
        if copied:
            self.failed_shard_paths.extend(copied)
            self._emit_event(
                "failed_shard_preserved",
                stage="failed_shard_preserved",
                worker_id=shard.worker_id,
                database_path=str(shard.database_path),
                copied_paths=copied,
            )
        else:
            self._emit_event(
                "failed_shard_missing",
                stage="failed_shard_missing",
                worker_id=shard.worker_id,
                database_path=str(shard.database_path),
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=300),
            )
        if errors:
            self.failed_shard_copy_errors.extend(errors)
            self._emit_event(
                "failed_shard_copy_error",
                stage="failed_shard_copy_error",
                worker_id=shard.worker_id,
                errors=errors,
            )

    def _write_run_failed(
        self,
        exc: BaseException,
        *,
        stage: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.run_dir is None:
            return
        failure_stage = stage or self._current_stage or "unknown"
        payload = {
            "schema_version": "run_failure_v1",
            "run_id": self.run_id,
            "stage": failure_stage,
            "worker_id": self.worker_id,
            "sample_id": self._current_sample_id,
            "query_task_id": self._current_query_task_id,
            "error_type": exc.__class__.__name__,
            "error_message": self._compact_error_text(exc, limit=1000),
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[-20:],
            "database_path": str(self.config.database_path) if self.config.database_path is not None else None,
            "index_database_path": str(self.config.index_database_path),
            "worker_database_root": self.worker_database_root,
            "worker_database_paths": list(self.worker_database_paths),
            "worker_database_local_scratch_used": self.worker_database_local_scratch_used,
            "worker_database_cleanup": dict(self.worker_database_cleanup),
            "worker_database_warnings": list(self.worker_database_warnings),
            "failed_shard_paths": list(self.failed_shard_paths),
            "failed_shard_copy_errors": list(self.failed_shard_copy_errors),
            "run_dir": str(self.run_dir),
            "output_dir": str(self.config.output_dir),
            "filesystem": {
                "run_dir": self._filesystem_snapshot(self.run_dir),
                "database": self._filesystem_snapshot(self.config.database_path),
                "worker_database_root": self._filesystem_snapshot(Path(self.worker_database_root))
                if self.worker_database_root
                else None,
                "cache_dir": self._filesystem_snapshot(self.config.memory_cache_dir),
            },
            "database_table_counts": self._sqlite_table_counts(self.config.database_path),
            "failed_shard_table_counts": [
                self._sqlite_table_counts(Path(path))
                for path in self.failed_shard_paths
                if str(path).endswith(".sqlite")
            ],
            "recent_events": self._load_recent_event_log(limit=50),
            "recent_provider_failures": self._recent_provider_failure_records(limit=20),
            "vllm_server": dict(self.vllm_server_metadata),
            "cuda_preflight": self._cuda_preflight_payload(),
        }
        if extra:
            payload.update(extra)
        try:
            write_json(self.run_dir / "run_failed.json", self._json_safe(payload))
        except Exception as write_exc:  # noqa: BLE001
            self._trace(
                f"run_failed_write_failed stage={failure_stage} "
                f"error={write_exc.__class__.__name__}: {write_exc}"
            )

    def _start_managed_vllm_if_needed(self) -> None:
        if not self.config.vllm_autostart:
            self.vllm_server_metadata = {
                "vllm_autostart": False,
                "vllm_reused_existing_server": False,
                "vllm_started_by_runner": False,
                "vllm_model": self.config.vllm_model,
                "vllm_served_model_name": self.config.vllm_served_model_name,
                "vllm_base_url": self.config.openai_compatible_base_url,
                "vllm_pid": None,
                "vllm_startup_latency_ms": None,
                "vllm_keep_alive": bool(self.config.vllm_keep_alive),
            }
            return
        assert self.run_dir is not None
        model = self.config.vllm_model or self.config.backbone_model
        served_model_name = self.config.vllm_served_model_name or self.config.backbone_model
        base_url = self.config.openai_compatible_base_url or f"http://127.0.0.1:{self.config.vllm_port}/v1"

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            self._emit_event(event_type, stage=event_type, **payload)

        self.managed_vllm_server = ManagedVLLMServer(
            base_url=base_url,
            autostart=True,
            model=model,
            served_model_name=served_model_name,
            host=self.config.vllm_host,
            port=self.config.vllm_port,
            cuda_visible_devices=self.config.vllm_cuda_visible_devices,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            dtype=self.config.vllm_dtype,
            extra_args=self.config.vllm_extra_args,
            startup_timeout_s=self.config.vllm_startup_timeout_s,
            keep_alive=self.config.vllm_keep_alive,
            status_dir=self.run_dir / "status",
            preflight_reservation={
                "risk": self.cuda_preflight_report.risk,
                "reserved_device_indices": sorted(self.cuda_preflight_report.reserved_device_indices),
                "reservations": list(self.cuda_preflight_report.reservations),
            },
            trace=self._trace if self.config.verbose else None,
            event_callback=emit,
        )
        try:
            self.managed_vllm_server.start()
        except Exception:
            self.vllm_server_metadata = self.managed_vllm_server.metadata()
            raise
        self.vllm_server_metadata = self.managed_vllm_server.metadata()
        if self.config.openai_compatible_base_url != base_url:
            self.config.openai_compatible_base_url = base_url

    def _stop_managed_vllm(self) -> None:
        if self.managed_vllm_server is None:
            return
        try:
            self.managed_vllm_server.stop()
        finally:
            self.vllm_server_metadata = self.managed_vllm_server.metadata()

    def run(self) -> RunReport:
        wallclock_started = time.perf_counter()
        self.run_started_at = datetime.utcnow().isoformat()
        try:
            self._prepare_run_database()
            self._start_heartbeat()
            self._emit_event(
                "run_start",
                stage="run_start",
                run_id=self.run_id,
                dataset=self.config.dataset,
                dataset_subset=self.config.dataset_subset,
                conv_workers=self.config.conv_workers,
                database_path=str(self.config.database_path),
            )
            self._write_cuda_preflight_report()
            self._emit_cuda_preflight_events()
            self._fail_if_cuda_preflight_strict()
            self._start_managed_vllm_if_needed()
            self._emit_filesystem_probes()
            assert self.exporter is not None

            samples = self.adapter.load_samples(
                self.config.dataset_path,
                self.config.max_samples,
                self.config.dataset_subset,
            )
            selected_groups, excluded_rows, selection_meta = self._select_logical_sample_groups(samples)
            self.excluded_samples = self._collect_exclusions(excluded_rows)
            self.selected_logical_sample_count = int(selection_meta["logical_sample_count"])
            self.selected_row_count = int(selection_meta["selected_row_count"])
            self.selected_query_count = int(selection_meta["selected_query_count"])
            self.selection_strategy = str(selection_meta["selection_strategy"])
            self.selected_counts_by_subset = dict(selection_meta["selected_counts_by_subset"])
            self._emit_event(
                "selection_done",
                stage="selection",
                logical_samples=self.selected_logical_sample_count,
                selected_rows=self.selected_row_count,
                selected_queries=self.selected_query_count,
                selected_counts_by_subset=self.selected_counts_by_subset,
            )
            self._trace(
                f"selection requested_max_samples={self.config.max_samples} "
                f"logical_samples={self.selected_logical_sample_count} selected_rows={self.selected_row_count} "
                f"selected_queries={self.selected_query_count} per_subset={self.selected_counts_by_subset} "
                f"strategy={self.selection_strategy}"
            )

            self.console.print(
                f"[bold]Benchmark[/bold] dataset={self.config.dataset} "
                f"logical_samples={self.selected_logical_sample_count} selected_rows={self.selected_row_count} "
                f"queries={self.selected_query_count} excluded={len(self.excluded_samples)} "
                f"conv_workers={self.config.conv_workers}"
            )
            self._warn_if_requested_backbone_concurrency_is_high()
            answer_results = self._generate_answers(selected_groups)
            self._emit_event(
                "evaluation_start",
                stage="evaluation",
                answers=len(answer_results),
                judge_parallelism=self.judge_parallelism,
            )
            evaluated_rows = self._evaluate_answers(answer_results)
            self._emit_event("evaluation_done", stage="evaluation", rows=len(evaluated_rows))
            self._finalize_fallback_repair_diagnostics(evaluated_rows)
            details = self._build_details_payload(answer_results, evaluated_rows)
            summary = self._build_summary_payload(
                answer_results=answer_results,
                evaluated_rows=evaluated_rows,
                wallclock_ms=(time.perf_counter() - wallclock_started) * 1000.0,
            )
            self._emit_event("artifact_write_start", stage="artifact_write")
            self._write_fallback_repair_events()
            self._write_text_only_filter_artifacts(summary)
            self._write_gold_label_artifacts(evaluated_rows)
            self._write_cost_diagnostic_artifacts(evaluated_rows)
            self._write_auditability_artifacts(evaluated_rows)
            self._persist_run_reporting(summary)
            self._persist_run_index(summary)
            self.exporter.export_benchmark_details(details)
            self.exporter.export_benchmark_summary(summary)
            self._emit_event(
                "artifact_write_done",
                stage="artifact_write",
                details_path=str(self.run_dir / "details.json") if self.run_dir else None,
                summary_path=str(self.run_dir / "summary.json") if self.run_dir else None,
            )
            self._trace("final_report_written")
            self.console.print(f"Artifacts written to {self.run_dir}")

            overall_metrics = dict(summary["aggregates"]["overall"].get("metrics", {}))
            self._emit_event(
                "run_done",
                stage="completed",
                processed_samples=self.selected_logical_sample_count,
                processed_queries=len(answer_results),
            )
            return RunReport(
                dataset=self.config.dataset,
                processed_samples=self.selected_logical_sample_count,
                processed_queries=len(answer_results),
                metrics=overall_metrics,
                details=summary,
            )
        except Exception as exc:
            self._emit_event(
                "run_failed",
                stage=self._current_stage or "run_failed",
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=500),
            )
            self._write_run_failed(exc, stage=self._current_stage or "run_failed")
            raise
        finally:
            self._stop_managed_vllm()
            self._stop_heartbeat()

    def _prepare_run_database(self) -> None:
        if self.session is not None:
            self.session.close()
        self.dataset_scope_key, self.dataset_scope = self._resolve_dataset_scope()
        self.run_id = self._build_run_id()
        self._set_meter_call_namespace(run_id=self.run_id, worker_id="main")
        self.run_dir = self.config.output_dir / self.config.dataset / self.dataset_scope_key / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.config.database_path is None:
            self.config.database_path = self.run_dir / "trajpatch.sqlite"
        if self.config.reset_run_db:
            for candidate in (
                self.config.database_path,
                Path(f"{self.config.database_path}-shm"),
                Path(f"{self.config.database_path}-wal"),
            ):
                if candidate.exists():
                    candidate.unlink()
        self.session_factory = create_schema(self.config.database_path)
        self.session = self.session_factory()
        self.store = TrajWikiStore(self.session)
        self.orchestrator = MemoryOrchestrator(
            self.config,
            self.store,
            self.llm_provider,
            self.embedding_provider,
            debug_dir=self.run_dir / "debug",
            trace=self._trace if self.config.verbose else None,
        )
        self.wiki = WikiCompiler(
            self.store,
            self.llm_provider,
            self.embedding_provider,
            trace=self._trace if self.config.verbose else None,
        )
        self.retrieval = RetrievalEngine(
            self.store,
            self.embedding_provider,
            self.config.t_pages,
            self.config.k,
            self.config.neighbor_radius,
            retrieval_expansion_mode=self.config.retrieval_expansion_mode,
            trace=self._trace if self.config.verbose else None,
            llm_provider=self.llm_provider,
            ablation_diagnostics_enabled=self.config.ablation_diagnostics,
            retrieval_rank_save_mode=self.config.retrieval_rank_save_mode,
            retrieval_rank_save_limit=self.config.retrieval_rank_save_limit,
        )
        self.exporter = ArtifactExporter(self.config.output_dir, self.store, run_dir=self.run_dir)
        self.cache_stats = self._empty_cache_stats()
        self.excluded_samples = []
        self.selected_logical_sample_count = 0
        self.selected_row_count = 0
        self.selected_query_count = 0
        self.selection_strategy = "none"
        self.selected_counts_by_subset = {}
        self.judge_parallelism = self._resolve_judge_parallelism()
        self.judge_total_wall_time_ms = 0.0
        self.judge_queue_latency_ms_total = 0.0
        self.judge_queue_events = 0
        self.worker_shards = []
        self.worker_device_allocation = {}
        self.worker_structured_call_metadata = []
        self.worker_provider_call_records = []
        self.worker_database_root = None
        self.worker_database_local_scratch_used = False
        self.worker_database_paths = []
        self.worker_database_cleanup = {"attempted": False, "cleaned_paths": [], "kept_paths": []}
        self.worker_database_warnings = []
        self.failed_shard_paths = []
        self.failed_shard_copy_errors = []
        self.backbone_inflight_semaphore = None
        self.memory_extract_batch_size_cap = None
        self.episodic_batch_call_count = 0
        self.episodic_batch_item_count = 0
        self.episodic_batch_repair_after_batch_count = 0
        self.episodic_batch_oom_backoff_count = 0
        self.episodic_batch_effective_size_max = 0
        self.episodic_batch_effective_size_final = 1
        self.memory_stage_batch_stats = {}
        self._structured_schema_checks_logged = set()
        self._event_counter = 0
        self._recent_lifecycle_events = []
        self._current_stage = "prepared"
        self._current_sample_id = None
        self._current_query_task_id = None
        self.fallback_repair_events = []
        self.text_only_filter_manifest = []
        self.text_only_filter_summary = {}
        self.managed_vllm_server = None
        self.vllm_server_metadata = {}

    def _prepare_worker_database(
        self,
        *,
        worker_id: int,
        run_id: str,
        run_dir: Path,
        database_path: Path,
        database_root: Path,
        local_scratch_used: bool,
    ) -> None:
        if self.session is not None:
            self.session.close()
        self.worker_id = worker_id
        self.dataset_scope_key, self.dataset_scope = self._resolve_dataset_scope()
        self.run_id = run_id
        self._set_meter_call_namespace(run_id=run_id, worker_id=worker_id)
        self.run_dir = run_dir
        self.config.database_path = database_path
        self.worker_database_root = str(database_root)
        self.worker_database_local_scratch_used = local_scratch_used
        self.failed_shard_paths = []
        self.failed_shard_copy_errors = []
        for candidate in (
            database_path,
            Path(f"{database_path}-shm"),
            Path(f"{database_path}-wal"),
        ):
            if candidate.exists():
                candidate.unlink()
        self.session_factory = create_schema(database_path, profile="worker_shard")
        self.session = self.session_factory()
        self.store = TrajWikiStore(self.session)
        self.orchestrator = MemoryOrchestrator(
            self.config,
            self.store,
            self.llm_provider,
            self.embedding_provider,
            debug_dir=run_dir / "debug" / f"worker-{worker_id:02d}",
            trace=self._trace if self.config.verbose else None,
        )
        self.wiki = WikiCompiler(
            self.store,
            self.llm_provider,
            self.embedding_provider,
            trace=self._trace if self.config.verbose else None,
        )
        self.retrieval = RetrievalEngine(
            self.store,
            self.embedding_provider,
            self.config.t_pages,
            self.config.k,
            self.config.neighbor_radius,
            retrieval_expansion_mode=self.config.retrieval_expansion_mode,
            trace=self._trace if self.config.verbose else None,
            llm_provider=self.llm_provider,
            ablation_diagnostics_enabled=self.config.ablation_diagnostics,
            retrieval_rank_save_mode=self.config.retrieval_rank_save_mode,
            retrieval_rank_save_limit=self.config.retrieval_rank_save_limit,
        )
        self.exporter = ArtifactExporter(self.config.output_dir, self.store, run_dir=run_dir)
        self.cache_stats = self._empty_cache_stats()
        self.judge_parallelism = self._resolve_judge_parallelism()
        self.judge_total_wall_time_ms = 0.0
        self.judge_queue_latency_ms_total = 0.0
        self.judge_queue_events = 0
        self.memory_extract_batch_size_cap = None
        self.episodic_batch_call_count = 0
        self.episodic_batch_item_count = 0
        self.episodic_batch_repair_after_batch_count = 0
        self.episodic_batch_oom_backoff_count = 0
        self.episodic_batch_effective_size_max = 0
        self.episodic_batch_effective_size_final = 1
        self.memory_stage_batch_stats = {}
        self._structured_schema_checks_logged = set()
        self._event_counter = 0
        self._recent_lifecycle_events = []
        self._current_stage = "worker_prepared"
        self._current_sample_id = None
        self._current_query_task_id = None
        self._start_heartbeat()
        snapshot = self._filesystem_snapshot(database_root)
        self._emit_event(
            "worker_start",
            stage="worker_start",
            worker_id=worker_id,
            database_path=str(database_path),
            database_root=str(database_root),
            local_scratch_used=local_scratch_used,
        )
        self._emit_filesystem_probes(include_worker_root=True)
        self._trace(
            f"worker_db_root_selected root={database_root} local_scratch={str(local_scratch_used).lower()} "
            f"free_gb={snapshot.get('free_gb') if snapshot.get('free_gb') is not None else 'unknown'}"
        )

    def _generate_answers(self, groups: list[LogicalSampleGroup]) -> list[AnswerResult]:
        results: list[AnswerResult] = []
        if not groups:
            return results
        if self.config.conv_workers > 1:
            return self._generate_answers_sharded(groups)
        self.console.print("[bold]Phase 1[/bold] Building memory, compiling wiki, and generating answers")
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task("memory", total=len(groups))
            for group in groups:
                progress.update(task_id, description=f"memory [{group.memory_sample.sample_id}]")
                results.extend(self._run_sample_group(group))
                assert self.exporter is not None
                sample_id = group.memory_sample.sample_id
                self._emit_event("sample_artifact_write_start", stage="artifact_write", sample_id=sample_id)
                try:
                    trajectory_export = self.exporter.export_sample_trajectories(sample_id)
                    self.exporter.export_sample_wiki(sample_id)
                    self._emit_event(
                        "sample_artifact_write_done",
                        stage="artifact_write",
                        sample_id=sample_id,
                        trajectory_export=trajectory_export,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._emit_event(
                        "sample_artifact_write_failed",
                        stage="artifact_write",
                        sample_id=sample_id,
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                    raise
                progress.advance(task_id)
        return results

    def _generate_answers_sharded(self, groups: list[LogicalSampleGroup]) -> list[AnswerResult]:
        assert self.run_id is not None
        assert self.run_dir is not None
        assert self.config.database_path is not None
        worker_count = min(max(int(self.config.conv_workers), 1), len(groups))
        shards = self._build_worker_shards(groups, worker_count)
        worker_database_root, local_scratch_used, database_warnings = self._resolve_worker_database_root()
        self.worker_database_root = str(worker_database_root)
        self.worker_database_local_scratch_used = local_scratch_used
        self.worker_database_warnings = list(database_warnings)
        worker_db_snapshot = self._filesystem_snapshot(worker_database_root)
        self._trace(
            f"worker_db_root_selected root={worker_database_root} local_scratch={str(local_scratch_used).lower()} "
            f"free_gb={worker_db_snapshot.get('free_gb') if worker_db_snapshot.get('free_gb') is not None else 'unknown'}"
        )
        self._emit_event(
            "worker_db_root_selected",
            stage="worker_db_root_selected",
            worker_database_root=str(worker_database_root),
            local_scratch_used=local_scratch_used,
            filesystem=worker_db_snapshot,
            warnings=database_warnings,
        )
        self._emit_filesystem_probes(include_worker_root=True)
        embedding_device_plans, worker_device_allocation = build_worker_embedding_device_plans(
            self.config,
            worker_count,
            excluded_device_indices=set(self.cuda_preflight_report.reserved_device_indices),
            preflight_report=self.cuda_preflight_report.to_dict(),
        )
        self.worker_device_allocation = worker_device_allocation
        for shard in shards:
            if shard.worker_id < len(embedding_device_plans):
                shard.embedding_device_plan = embedding_device_plans[shard.worker_id]
            shard.database_root = worker_database_root
            shard.local_scratch_used = local_scratch_used
            shard.database_path = worker_database_root / f"worker-{shard.worker_id:02d}.sqlite"
        self.worker_database_paths = [
            {
                "worker_id": shard.worker_id,
                "database_path": str(shard.database_path) if shard.database_path is not None else None,
                "database_root": str(shard.database_root) if shard.database_root is not None else None,
                "local_scratch_used": shard.local_scratch_used,
            }
            for shard in shards
        ]
        self.worker_shards = [
            {
                "worker_id": shard.worker_id,
                "logical_samples": len(shard.groups),
                "queries": shard.query_count,
                "sample_ids": [group.group_id for group in shard.groups],
                "database_path": str(shard.database_path) if shard.database_path is not None else None,
                "local_scratch_used": shard.local_scratch_used,
                "embedding_accelerator": (
                    shard.embedding_device_plan.accelerator if shard.embedding_device_plan is not None else None
                ),
                "embedding_selected_device_index": (
                    shard.embedding_device_plan.metadata.get("selected_device_index")
                    if shard.embedding_device_plan is not None
                    else None
                ),
            }
            for shard in shards
        ]
        self._trace(f"sample_sharding_done workers={worker_count} shards={self.worker_shards}")
        self._emit_event(
            "sample_sharding_done",
            stage="sample_sharding",
            worker_count=worker_count,
            shards=self.worker_shards,
        )
        devices = ",".join(
            f"worker{item['worker_id']:02d}:{item.get('accelerator')}"
            for item in self.worker_device_allocation.get("assignments", [])
        )
        self._trace(
            f"worker_embedding_assignment_done workers={worker_count} devices={devices or 'none'} "
            f"model_sharing={str(bool(self.worker_device_allocation.get('model_sharing_enabled'))).lower()} "
            f"warnings={len(self.worker_device_allocation.get('warnings', []))}"
        )
        self.console.print(
            "[bold]Phase 1[/bold] Building memory, compiling wiki, and generating answers "
            f"(workers={worker_count})"
        )
        self.backbone_inflight_semaphore = self._build_backbone_inflight_semaphore(worker_count)

        shard_results: list[PipelineShardResult] = []
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task(f"memory workers={worker_count}", total=len(groups))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(self._run_worker_shard, shard): shard
                    for shard in shards
                    if shard.groups
                }
                try:
                    for future in as_completed(future_map):
                        shard = future_map[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            for pending in future_map:
                                pending.cancel()
                            sample_ids = ",".join(group.group_id for group in shard.groups)
                            self._emit_event(
                                "worker_failed",
                                stage="worker_failed",
                                worker_id=shard.worker_id,
                                sample_ids=[group.group_id for group in shard.groups],
                                database_path=str(shard.database_path) if shard.database_path is not None else None,
                                error_type=exc.__class__.__name__,
                                error_message=self._compact_error_text(exc, limit=500),
                            )
                            self._preserve_failed_worker_shard(shard, exc)
                            raise RuntimeError(
                                f"Worker {shard.worker_id:02d} failed for samples [{sample_ids}]: {exc}"
                            ) from exc
                        shard_results.append(result)
                        progress.update(task_id, description=f"memory worker={result.worker_id:02d}")
                        progress.advance(task_id, advance=len(shard.groups))
                        self._trace(
                            f"worker_done worker={result.worker_id:02d} logical_samples={len(result.group_ids)} "
                            f"queries={result.query_count} runtime_ms={result.runtime_ms:.1f} "
                            f"cache_hits={int(result.cache_stats.get('cache_hits', 0.0))}"
                        )
                        self._emit_event(
                            "worker_done",
                            stage="worker_done",
                            worker_id=result.worker_id,
                            sample_ids=result.group_ids,
                            query_count=result.query_count,
                            runtime_ms=result.runtime_ms,
                            database_path=str(result.database_path),
                        )
                finally:
                    self.backbone_inflight_semaphore = None

        self._trace(f"shard_merge_start workers={len(shard_results)}")
        self._emit_event("shard_merge_start", stage="shard_merge", workers=len(shard_results))
        try:
            self._merge_shard_databases(shard_results)
        except Exception as exc:  # noqa: BLE001
            self._emit_event(
                "shard_merge_failed",
                stage="shard_merge",
                workers=len(shard_results),
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=500),
            )
            raise
        self._cleanup_worker_databases(shard_results)
        self._apply_shard_result_stats(shard_results)
        self._trace(f"shard_merge_done workers={len(shard_results)}")
        self._emit_event("shard_merge_done", stage="shard_merge", workers=len(shard_results))
        answers_by_group: dict[str, list[AnswerResult]] = {}
        for result in shard_results:
            answers_by_group.update(result.answer_results_by_group)
        ordered_results: list[AnswerResult] = []
        for group in groups:
            ordered_results.extend(answers_by_group.get(group.group_id, []))
        return ordered_results

    def _build_worker_shards(self, groups: list[LogicalSampleGroup], worker_count: int) -> list[PipelineShard]:
        shards = [PipelineShard(worker_id=index, groups=[], query_count=0) for index in range(worker_count)]
        indexed_groups = [
            (
                index,
                group,
                sum(len(self.adapter.build_query_tasks(row)) for row in group.query_rows),
            )
            for index, group in enumerate(groups)
        ]
        for _, group, query_count in sorted(indexed_groups, key=lambda item: (-item[2], item[0])):
            target = min(shards, key=lambda shard: (shard.query_count, shard.worker_id))
            target.groups.append(group)
            target.query_count += query_count
        for shard in shards:
            order_index = {group.group_id: index for index, group in enumerate(groups)}
            shard.groups.sort(key=lambda group: order_index[group.group_id])
        return shards

    def _run_worker_shard(self, shard: PipelineShard) -> PipelineShardResult:
        assert self.run_id is not None
        assert self.run_dir is not None
        assert shard.database_path is not None
        assert shard.database_root is not None
        worker_config = self._copy_config_for_worker(shard.worker_id, shard.database_path)
        device_plan_overrides = (
            {"embedding": shard.embedding_device_plan}
            if shard.embedding_device_plan is not None
            else None
        )
        worker = PipelineRunner(worker_config, console=self.console, device_plan_overrides=device_plan_overrides)
        if self.backbone_inflight_semaphore is not None:
            worker.llm_provider = SemaphoreLimitedLLMProvider(worker.llm_provider, self.backbone_inflight_semaphore)
            worker.answering = AnswerGenerator(worker.llm_provider, trace=worker._trace if worker.config.verbose else None)
        worker._prepare_worker_database(
            worker_id=shard.worker_id,
            run_id=self.run_id,
            run_dir=self.run_dir,
            database_path=worker_config.database_path,
            database_root=shard.database_root,
            local_scratch_used=shard.local_scratch_used,
        )
        started = time.perf_counter()
        answer_results_by_group: dict[str, list[AnswerResult]] = {}
        try:
            for group in shard.groups:
                worker._emit_event(
                    "worker_sample_start",
                    stage="sample_start",
                    worker_id=shard.worker_id,
                    sample_id=group.group_id,
                    query_rows=len(group.query_rows),
                )
                answer_results_by_group[group.group_id] = worker._run_sample_group(group)
                assert worker.exporter is not None
                sample_id = group.memory_sample.sample_id
                worker._emit_event(
                    "sample_artifact_write_start",
                    stage="artifact_write",
                    worker_id=shard.worker_id,
                    sample_id=sample_id,
                )
                try:
                    trajectory_export = worker.exporter.export_sample_trajectories(sample_id)
                    worker.exporter.export_sample_wiki(sample_id)
                    worker._emit_event(
                        "sample_artifact_write_done",
                        stage="artifact_write",
                        worker_id=shard.worker_id,
                        sample_id=sample_id,
                        trajectory_export=trajectory_export,
                    )
                except Exception as exc:  # noqa: BLE001
                    worker._emit_event(
                        "sample_artifact_write_failed",
                        stage="artifact_write",
                        worker_id=shard.worker_id,
                        sample_id=sample_id,
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                    raise
                worker._emit_event(
                    "worker_sample_done",
                    stage="sample_done",
                    worker_id=shard.worker_id,
                    sample_id=group.group_id,
                    answers=len(answer_results_by_group[group.group_id]),
                )
            if worker.session is not None:
                worker._commit_session(stage="worker_finalize")
            worker._emit_event(
                "worker_finalize_done",
                stage="worker_finalize",
                worker_id=shard.worker_id,
                database_path=str(worker_config.database_path),
            )
            worker._sync_cache_lock_stats()
            structured_records = worker._structured_call_metadata()
            provider_call_records = worker._provider_call_records()
            orchestrator_counters = worker._orchestrator_counter_snapshot()
            runtime_ms = (time.perf_counter() - started) * 1000.0
            return PipelineShardResult(
                worker_id=shard.worker_id,
                group_ids=[group.group_id for group in shard.groups],
                query_count=shard.query_count,
                answer_results_by_group=answer_results_by_group,
                database_path=worker_config.database_path,
                runtime_ms=runtime_ms,
                cache_stats=dict(worker.cache_stats),
                memory_stage_batch_stats=worker._copy_memory_stage_batch_stats(worker.memory_stage_batch_stats),
                episodic_batch_call_count=worker.episodic_batch_call_count,
                episodic_batch_item_count=worker.episodic_batch_item_count,
                episodic_batch_repair_after_batch_count=worker.episodic_batch_repair_after_batch_count,
                episodic_batch_oom_backoff_count=worker.episodic_batch_oom_backoff_count,
                episodic_batch_effective_size_max=worker.episodic_batch_effective_size_max,
                episodic_batch_effective_size_final=worker.episodic_batch_effective_size_final,
                orchestrator_counters=orchestrator_counters,
                structured_call_metadata=structured_records,
                provider_call_records=provider_call_records,
                device_allocation=dict(worker.device_allocation),
            )
        except Exception as exc:
            worker._emit_event(
                "worker_failed",
                stage=worker._current_stage or "worker_failed",
                worker_id=shard.worker_id,
                sample_ids=[group.group_id for group in shard.groups],
                database_path=str(worker_config.database_path),
                error_type=exc.__class__.__name__,
                error_message=worker._compact_error_text(exc, limit=500),
                recent_provider_failures=worker._recent_provider_failure_records(limit=10),
            )
            try:
                write_json(
                    self.run_dir / "status" / f"worker-{shard.worker_id:02d}-failed.json",
                    worker._json_safe(
                        {
                            "schema_version": "worker_failure_v1",
                            "worker_id": shard.worker_id,
                            "sample_ids": [group.group_id for group in shard.groups],
                            "database_path": str(worker_config.database_path),
                            "error_type": exc.__class__.__name__,
                            "error_message": worker._compact_error_text(exc, limit=1000),
                            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[-20:],
                            "database_table_counts": worker._sqlite_table_counts(worker_config.database_path),
                            "recent_events": worker._load_recent_event_log(limit=50),
                            "recent_provider_failures": worker._recent_provider_failure_records(limit=20),
                        }
                    ),
                )
            except Exception as write_exc:  # noqa: BLE001
                worker._trace(
                    f"worker_failure_write_failed worker={shard.worker_id:02d} "
                    f"error={write_exc.__class__.__name__}: {write_exc}"
                )
            raise
        finally:
            if worker.session is not None:
                worker.session.close()
            worker._stop_heartbeat()

    def _copy_config_for_worker(self, worker_id: int, database_path: Path) -> RunConfig:
        updates = {
            "database_path": database_path,
            "reset_run_db": True,
            "index_database_path": self.config.index_database_path,
        }
        if hasattr(self.config, "model_copy"):
            return self.config.model_copy(update=updates)
        return self.config.copy(update=updates)

    def _build_backbone_inflight_semaphore(self, worker_count: int) -> BoundedSemaphore | None:
        if worker_count <= 1:
            return None
        info = self.llm_provider.model_info()
        if info.provider_kind not in REMOTE_LIKE_PROVIDER_KINDS:
            return None
        limit = min(8, max(1, self._estimated_backbone_inflight_limit()))
        return BoundedSemaphore(limit)

    def _estimated_backbone_inflight_limit(self) -> int:
        return max(1, int(self.config.conv_workers) * self._resolve_memory_extract_batch_size())

    def _warn_if_requested_backbone_concurrency_is_high(self) -> None:
        if self.config.conv_workers <= 1:
            return
        info = self.llm_provider.model_info()
        if info.provider_kind not in REMOTE_LIKE_PROVIDER_KINDS:
            return
        estimated = self._estimated_backbone_inflight_limit()
        if estimated > 16:
            self.console.print(
                "[yellow]Warning:[/yellow] requested backbone concurrency is high "
                f"(conv_workers * memory_extract_batch_size = {estimated}). "
                "This may trigger remote provider rate limits."
            )

    def _merge_shard_databases(self, shard_results: list[PipelineShardResult]) -> None:
        assert self.config.database_path is not None
        if self.session is not None:
            self._commit_session(stage="shard_merge_prepare")
        main_path = self.config.database_path
        table_columns = {
            table_name: [column.name for column in Base.metadata.tables[table_name].columns]
            for table_name in SHARD_MERGE_TABLES
        }
        with sqlite3.connect(str(main_path)) as connection:
            for result in sorted(shard_results, key=lambda item: item.worker_id):
                if not result.database_path.exists():
                    raise RuntimeError(f"Shard database does not exist: {result.database_path}")
                alias = f"shard_{result.worker_id:02d}"
                connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(result.database_path),))
                try:
                    for table_name in SHARD_MERGE_TABLES:
                        columns = table_columns[table_name]
                        quoted_columns = ", ".join(f'"{column}"' for column in columns)
                        try:
                            connection.execute(
                                f'INSERT INTO "{table_name}" ({quoted_columns}) '
                                f'SELECT {quoted_columns} FROM {alias}."{table_name}"'
                            )
                        except sqlite3.IntegrityError as exc:
                            raise RuntimeError(
                                f"Shard merge conflict worker={result.worker_id:02d} table={table_name}: {exc}"
                            ) from exc
                    connection.commit()
                finally:
                    connection.execute(f"DETACH DATABASE {alias}")
        if self.session is not None:
            self.session.expire_all()

    def _apply_shard_result_stats(self, shard_results: list[PipelineShardResult]) -> None:
        self.cache_stats = self._empty_cache_stats()
        self.memory_stage_batch_stats = {}
        self.worker_structured_call_metadata = []
        self.worker_provider_call_records = []
        self.episodic_batch_call_count = 0
        self.episodic_batch_item_count = 0
        self.episodic_batch_repair_after_batch_count = 0
        self.episodic_batch_oom_backoff_count = 0
        self.episodic_batch_effective_size_max = 0
        self.episodic_batch_effective_size_final = 1
        worker_actual_allocations: dict[str, Any] = {}
        for result in shard_results:
            worker_actual_allocations[f"{result.worker_id:02d}"] = dict(result.device_allocation)
            for key, value in result.cache_stats.items():
                self.cache_stats[key] = float(self.cache_stats.get(key, 0.0)) + float(value)
            self.episodic_batch_call_count += result.episodic_batch_call_count
            self.episodic_batch_item_count += result.episodic_batch_item_count
            self.episodic_batch_repair_after_batch_count += result.episodic_batch_repair_after_batch_count
            self.episodic_batch_oom_backoff_count += result.episodic_batch_oom_backoff_count
            self.episodic_batch_effective_size_max = max(
                self.episodic_batch_effective_size_max,
                result.episodic_batch_effective_size_max,
            )
            self.episodic_batch_effective_size_final = max(
                self.episodic_batch_effective_size_final,
                result.episodic_batch_effective_size_final,
            )
            for stage, stats in result.memory_stage_batch_stats.items():
                target = self.memory_stage_batch_stats.setdefault(
                    stage,
                    {"calls": 0, "items": 0, "fallbacks": 0, "structured_failures": 0},
                )
                for key, value in stats.items():
                    target[key] = int(target.get(key, 0)) + int(value)
            self.worker_structured_call_metadata.extend(result.structured_call_metadata)
            self.worker_provider_call_records.extend(result.provider_call_records)
        if self.orchestrator is not None:
            for field_name in ORCHESTRATOR_COUNTER_FIELDS:
                setattr(
                    self.orchestrator,
                    field_name,
                    sum(int(result.orchestrator_counters.get(field_name, 0)) for result in shard_results),
                )
            for field_name in ORCHESTRATOR_COUNTER_MAP_FIELDS:
                merged: Counter[str] = Counter()
                for result in shard_results:
                    merged.update(dict(result.orchestrator_counters.get(field_name, {})))
                setattr(self.orchestrator, field_name, Counter(merged))
            debug_paths: dict[str, list[str]] = {}
            for result in shard_results:
                for sample_id, paths in dict(result.orchestrator_counters.get("debug_artifact_paths", {})).items():
                    debug_paths.setdefault(str(sample_id), []).extend(list(paths))
            self.orchestrator.debug_artifact_paths = debug_paths
        if worker_actual_allocations:
            self.worker_device_allocation = {
                **self.worker_device_allocation,
                "worker_actual_device_allocations": worker_actual_allocations,
            }

    def _orchestrator_counter_snapshot(self) -> dict[str, Any]:
        assert self.orchestrator is not None
        snapshot: dict[str, Any] = {
            field_name: getattr(self.orchestrator, field_name, 0)
            for field_name in ORCHESTRATOR_COUNTER_FIELDS
        }
        for field_name in ORCHESTRATOR_COUNTER_MAP_FIELDS:
            snapshot[field_name] = dict(getattr(self.orchestrator, field_name, {}))
        snapshot["debug_artifact_paths"] = {
            sample_id: list(paths)
            for sample_id, paths in getattr(self.orchestrator, "debug_artifact_paths", {}).items()
        }
        return snapshot

    def _structured_call_metadata(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for provider in (self.llm_provider, self.judge_provider):
            if isinstance(provider, SemaphoreLimitedLLMProvider):
                provider = provider.provider
            if isinstance(provider, MeteredLLMProvider):
                records.extend(row.metadata for row in provider.calls if row.metadata.get("structured_requested"))
        return records

    def _provider_call_records(self) -> list[dict[str, Any]]:
        """Return all metered calls without prompt text."""

        records: list[dict[str, Any]] = []
        for provider in (self.llm_provider, self.judge_provider):
            if isinstance(provider, SemaphoreLimitedLLMProvider):
                provider = provider.provider
            if not isinstance(provider, MeteredLLMProvider):
                continue
            for row in provider.calls:
                metadata = dict(row.metadata or {})
                records.append(
                    {
                        "role": row.role,
                        "task": row.task,
                        "prompt_tokens": row.prompt_tokens,
                        "completion_tokens": row.completion_tokens,
                        "latency_ms": row.latency_ms,
                        "provider_call_id": row.provider_call_id,
                        "provider_call_uid": metadata.get("provider_call_uid"),
                        "call_item_uid": metadata.get("call_item_uid"),
                        "provider_call_kind": row.provider_call_kind,
                        "logical_item_count": row.logical_item_count,
                        "metadata": metadata,
                    }
                )
        return records

    @staticmethod
    def _answer_is_context_abstention(answer_text: str) -> bool:
        normalized = " ".join(answer_text.casefold().split())
        regex_patterns = [
            r"\bretrieved context (?:does not|doesn't) support\b",
            r"\bnot supported by the retrieved context\b",
            r"\bnot enough information\b",
            r"\bdo(?:es)? not have enough information\b",
            r"\bdo(?:es)? not provide enough information\b",
            r"\bdo(?:es)? not provide (?:any )?information\b",
            r"\bdo(?:es)? not mention\b",
            r"\bdo(?:es)? not specify\b",
            r"\bnot specified in (?:the )?(?:retrieved )?context\b",
            r"\bnot provided in (?:the )?(?:retrieved )?context\b",
            r"\bno information (?:about|regarding|on)\b",
            r"\bcannot determine\b",
            r"\bcan't determine\b",
            r"\bunknown from context\b",
        ]
        return any(re.search(pattern, normalized) for pattern in regex_patterns)

    @staticmethod
    def _retrieval_bundle_is_weak(bundle: RetrievalBundle) -> bool:
        metadata = dict(bundle.metadata or {})
        return (
            not bundle.source_message_ids
            or not bundle.candidate_trajectories
            or not bundle.snapshot_hits
            or int(metadata.get("answer_context_active_claim_count") or 0) <= 0
        )

    def _should_retry_locomo_reflection(
        self,
        query_sample: DatasetSample,
        retrieval_bundle: RetrievalBundle,
        answer_text: str,
    ) -> bool:
        if query_sample.dataset_name != "locomo":
            return False
        return self._answer_is_context_abstention(answer_text) or self._retrieval_bundle_is_weak(
            retrieval_bundle
        )

    def _reflection_retry_metadata_default(self) -> dict[str, Any]:
        return {
            "retrieval_reflection_retry_used": False,
            "retrieval_reflection_used": False,
            "retrieval_reflection_stage": "none",
            "initial_answer_abstained": False,
            "initial_retrieval_bundle_weak": False,
            "initial_answer_generation_failed": False,
            "reflection_rewritten_query": "",
            "reflection_must_find_terms": [],
            "reflection_candidate_page_slugs": [],
            "reflection_reroute_event_id": None,
            "post_reflection_raw_rescue_used": False,
            "post_reflection_raw_rescue_reason": None,
            "post_reflection_raw_rescue_event_id": None,
            "final_reflection_retrieval_event_id": None,
            "raw_rescue_attempted": False,
            "raw_rescue_trigger_reasons": [],
            "raw_rescue_skipped_reason": None,
            "reflection_required_terms": [],
            "reflection_covered_terms": [],
            "reflection_uncovered_terms": [],
            "reflection_term_coverage_rate": None,
            "reflection_semantic_evidence_weak": False,
            "raw_rescue_used": False,
            "raw_rescue_hit_count": 0,
            "raw_rescue_source_refs": [],
            "raw_rescue_compensated_memory_gap": False,
            "reflection_answer_changed": False,
        }

    @staticmethod
    def _answer_synthesis_can_answer_false(response: LLMResponse) -> bool:
        return dict(response.metadata or {}).get("answer_synthesis_can_answer") is False

    def _maybe_retry_with_retrieval_reflection(
        self,
        *,
        sample_id: str,
        query_sample: DatasetSample,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        answer_prompt: str,
        answer_response: LLMResponse,
    ) -> tuple[RetrievalBundle, str, LLMResponse, dict[str, Any]]:
        metadata = self._reflection_retry_metadata_default()
        initial_answer_abstained = self._answer_is_context_abstention(answer_response.text)
        initial_retrieval_bundle_weak = self._retrieval_bundle_is_weak(retrieval_bundle)
        initial_answer_generation_failed = bool(dict(answer_response.metadata or {}).get("answer_generation_failed"))
        should_retry = (
            query_sample.dataset_name == "locomo"
            and (initial_answer_abstained or initial_retrieval_bundle_weak)
            and not initial_answer_generation_failed
        )
        metadata["initial_answer_generation_failed"] = initial_answer_generation_failed
        if self.retrieval is None or self.answering is None or not should_retry:
            return retrieval_bundle, answer_prompt, answer_response, metadata

        self._trace(
            f"sample={sample_id} query_task_id={query_task.query_task_id} "
            "retrieval_reflection_start"
        )
        reflection_hints = self.retrieval.build_reflection_hints(
            sample_id,
            query_task.question,
            initial_answer_text=answer_response.text,
            initial_retrieval_metadata=dict(retrieval_bundle.metadata or {}),
        )
        reflection_hints = {
            **dict(reflection_hints),
            "initial_answer_abstained": initial_answer_abstained,
            "initial_retrieval_bundle_weak": initial_retrieval_bundle_weak,
        }
        retry_bundle = self.retrieval.build_context(
            sample_id,
            query_task.question,
            attempt_label="reflection",
            reflection_hints=reflection_hints,
            normal_retrieval_event_id=retrieval_bundle.retrieval_event_id,
        )
        retry_prompt, retry_response = self.answering.generate(query_task, retry_bundle)
        retry_metadata = dict(retry_bundle.metadata or {})
        final_bundle = retry_bundle
        final_prompt = retry_prompt
        final_response = retry_response
        final_retry_metadata = retry_metadata
        post_reflection_raw_rescue_reason: str | None = None
        if not bool(retry_metadata.get("raw_rescue_attempted")):
            if self._answer_is_context_abstention(retry_response.text):
                post_reflection_raw_rescue_reason = "post_reflection_abstain"
            elif self._answer_synthesis_can_answer_false(retry_response):
                post_reflection_raw_rescue_reason = "post_reflection_synthesis_can_answer_false"
        if post_reflection_raw_rescue_reason is not None:
            forced_bundle = self.retrieval.build_context(
                sample_id,
                query_task.question,
                attempt_label="reflection",
                reflection_hints=reflection_hints,
                normal_retrieval_event_id=retrieval_bundle.retrieval_event_id,
                force_raw_rescue=True,
                force_raw_rescue_reason=post_reflection_raw_rescue_reason,
            )
            forced_prompt, forced_response = self.answering.generate(query_task, forced_bundle)
            final_bundle = forced_bundle
            final_prompt = forced_prompt
            final_response = forced_response
            final_retry_metadata = dict(forced_bundle.metadata or {})
        answer_changed = answer_response.text.strip() != final_response.text.strip()
        metadata.update(
            {
                "retrieval_reflection_retry_used": True,
                "retrieval_reflection_used": True,
                "retrieval_reflection_stage": final_retry_metadata.get("retrieval_reflection_stage") or "wiki",
                "initial_answer_abstained": initial_answer_abstained,
                "initial_retrieval_bundle_weak": initial_retrieval_bundle_weak,
                "reflection_rewritten_query": final_retry_metadata.get("reflection_rewritten_query") or "",
                "reflection_must_find_terms": list(final_retry_metadata.get("reflection_must_find_terms") or []),
                "reflection_candidate_page_slugs": list(
                    final_retry_metadata.get("reflection_candidate_page_slugs") or []
                ),
                "reflection_mode": final_retry_metadata.get("reflection_mode"),
                "reflection_error": final_retry_metadata.get("reflection_error"),
                "reflection_latency_ms": float(final_retry_metadata.get("reflection_latency_ms") or 0.0),
                "normal_retrieval_event_id": retrieval_bundle.retrieval_event_id,
                "initial_answer_text": answer_response.text,
                "initial_answer_prompt": answer_prompt,
                "initial_retrieval_event_id": retrieval_bundle.retrieval_event_id,
                "initial_answer_prompt_tokens": int(answer_response.prompt_tokens or 0),
                "initial_answer_completion_tokens": int(answer_response.completion_tokens or 0),
                "initial_answer_latency_ms": float(answer_response.metadata.get("latency_ms", 0.0)),
                "initial_retrieval_latency_ms": float(retrieval_bundle.latency_ms),
                "reflection_reroute_event_id": retry_bundle.retrieval_event_id,
                "reflection_reroute_latency_ms": float(retry_bundle.latency_ms),
                "reflection_retrieval_event_id": final_bundle.retrieval_event_id,
                "reflection_retrieval_latency_ms": float(final_bundle.latency_ms),
                "post_reflection_raw_rescue_used": post_reflection_raw_rescue_reason is not None,
                "post_reflection_raw_rescue_reason": post_reflection_raw_rescue_reason,
                "post_reflection_raw_rescue_event_id": (
                    final_bundle.retrieval_event_id if post_reflection_raw_rescue_reason is not None else None
                ),
                "final_reflection_retrieval_event_id": final_bundle.retrieval_event_id,
                "raw_rescue_attempted": bool(final_retry_metadata.get("raw_rescue_attempted")),
                "raw_rescue_trigger_reasons": list(final_retry_metadata.get("raw_rescue_trigger_reasons") or []),
                "raw_rescue_skipped_reason": final_retry_metadata.get("raw_rescue_skipped_reason"),
                "reflection_required_terms": list(final_retry_metadata.get("reflection_required_terms") or []),
                "reflection_covered_terms": list(final_retry_metadata.get("reflection_covered_terms") or []),
                "reflection_uncovered_terms": list(final_retry_metadata.get("reflection_uncovered_terms") or []),
                "reflection_term_coverage_rate": final_retry_metadata.get("reflection_term_coverage_rate"),
                "reflection_semantic_evidence_weak": bool(
                    final_retry_metadata.get("reflection_semantic_evidence_weak")
                ),
                "raw_rescue_used": bool(final_retry_metadata.get("raw_rescue_used")),
                "raw_rescue_hit_count": int(final_retry_metadata.get("raw_rescue_hit_count") or 0),
                "raw_rescue_source_refs": list(final_retry_metadata.get("raw_rescue_source_refs") or []),
                "raw_rescue_source_ids": list(final_retry_metadata.get("raw_rescue_source_ids") or []),
                "raw_rescue_compensated_memory_gap": bool(
                    final_retry_metadata.get("raw_rescue_used")
                    and final_retry_metadata.get("raw_rescue_hit_count")
                    and int(final_retry_metadata.get("answer_context_active_claim_count") or 0) <= 0
                ),
                "reflection_answer_changed": answer_changed,
            }
        )
        final_response.metadata.update(metadata)
        self._trace(
            f"sample={sample_id} query_task_id={query_task.query_task_id} "
            f"retrieval_reflection_done stage={metadata['retrieval_reflection_stage']} "
            f"raw_hits={metadata['raw_rescue_hit_count']} answer_changed={str(answer_changed).lower()}"
        )
        return final_bundle, final_prompt, final_response, metadata

    def _run_sample_group(self, group: LogicalSampleGroup) -> list[AnswerResult]:
        assert self.store is not None
        assert self.session is not None
        assert self.retrieval is not None

        sample = group.memory_sample
        sample_started = time.perf_counter()
        backbone_start = self._meter_snapshot(self.llm_provider)
        group_subsets = list(dict.fromkeys(str(row.subset_key or "unknown") for row in group.query_rows))
        self._emit_event(
            "sample_start",
            stage="sample_start",
            sample_id=sample.sample_id,
            subsets=group_subsets,
            query_rows=len(group.query_rows),
        )
        self._trace(
            f"sample={sample.sample_id} start subset={','.join(group_subsets)} "
            f"scene={sample.scene_tag or '-'} query_rows={len(group.query_rows)}"
        )
        try:
            raw_started = time.perf_counter()
            messages = self._prepare_raw_messages(sample)
            raw_latency_ms = (time.perf_counter() - raw_started) * 1000.0
            self._trace(f"sample={sample.sample_id} raw_messages count={len(messages)} latency_ms={raw_latency_ms:.1f}")
            memory_started = time.perf_counter()
            self._emit_event("memory_build_start", stage="memory_build", sample_id=sample.sample_id)
            memory_info = self._build_or_restore_sample_memory(sample, messages)
            memory_stage_latency_ms = (time.perf_counter() - memory_started) * 1000.0
            self._emit_event(
                "memory_build_done",
                stage="memory_build",
                sample_id=sample.sample_id,
                cache_hit=bool(memory_info["cache_hit"]),
                latency_ms=memory_stage_latency_ms,
            )
            self._trace(
                f"sample={sample.sample_id} memory_build cache_hit={memory_info['cache_hit']} "
                f"latency_ms={memory_stage_latency_ms:.1f}"
            )
        except Exception as exc:
            self._emit_event(
                "sample_failed",
                stage=self._current_stage or "sample_failed",
                sample_id=sample.sample_id,
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=500),
            )
            raise
        query_entries: list[tuple[DatasetSample, QueryTask]] = []
        for query_row in group.query_rows:
            for query_task in self.adapter.build_query_tasks(query_row):
                query_entries.append((query_row, query_task))
        if not query_entries:
            self._trace(f"sample={sample.sample_id} no_query_tasks")
            return []

        partial_results: list[tuple[AnswerResult, dict[str, Any]]] = []
        per_query_share = max(len(query_entries), 1)
        for index, (query_sample, query_task) in enumerate(query_entries):
            self._emit_event(
                "query_start",
                stage="query",
                sample_id=sample.sample_id,
                query_task_id=query_task.query_task_id,
                query_index=index + 1,
                query_count=len(query_entries),
            )
            self._trace(
                f"sample={sample.sample_id} query={index + 1}/{len(query_entries)} "
                f"query_task_id={query_task.query_task_id} page_route_start"
            )
            try:
                retrieval_bundle = self.retrieval.build_context(sample.sample_id, query_task.question)
                self._emit_event(
                    "retrieval_done",
                    stage="retrieval",
                    sample_id=sample.sample_id,
                    query_task_id=query_task.query_task_id,
                    latency_ms=retrieval_bundle.latency_ms,
                    pages=len(retrieval_bundle.selected_pages),
                    trajectories=len(retrieval_bundle.candidate_trajectories),
                    snapshots=len(retrieval_bundle.snapshot_hits),
                    expanded=len(retrieval_bundle.expanded_snapshots),
                    sources=len(retrieval_bundle.source_message_ids),
                    retrieval_event_id=retrieval_bundle.retrieval_event_id,
                )
                self._trace(
                    f"sample={sample.sample_id} query_task_id={query_task.query_task_id} retrieval_done "
                    f"latency_ms={retrieval_bundle.latency_ms:.1f} "
                    f"pages={len(retrieval_bundle.selected_pages)} trajectories={len(retrieval_bundle.candidate_trajectories)} "
                    f"snapshots={len(retrieval_bundle.snapshot_hits)} expanded={len(retrieval_bundle.expanded_snapshots)} "
                    f"sources={len(retrieval_bundle.source_message_ids)}"
                )
                self._emit_event(
                    "answer_generation_start",
                    stage="answer_generation",
                    sample_id=sample.sample_id,
                    query_task_id=query_task.query_task_id,
                )
                self._trace(f"sample={sample.sample_id} query_task_id={query_task.query_task_id} answer_start")
                answer_prompt, answer_response = self.answering.generate(query_task, retrieval_bundle)
                self._emit_event(
                    "answer_generation_done",
                    stage="answer_generation",
                    sample_id=sample.sample_id,
                    query_task_id=query_task.query_task_id,
                    latency_ms=float(answer_response.metadata.get("latency_ms", 0.0)),
                    prompt_tokens=int(answer_response.prompt_tokens or 0),
                    completion_tokens=int(answer_response.completion_tokens or 0),
                    answer_generation_failed=bool(answer_response.metadata.get("answer_generation_failed")),
                )
                self._trace(
                    f"sample={sample.sample_id} query_task_id={query_task.query_task_id} answer_done "
                    f"latency_ms={float(answer_response.metadata.get('latency_ms', 0.0)):.1f} "
                    f"prompt_tokens={int(answer_response.prompt_tokens or 0)} "
                    f"completion_tokens={int(answer_response.completion_tokens or 0)}"
                )
            except Exception as exc:
                self._emit_event(
                    "sample_failed",
                    stage=self._current_stage or "sample_failed",
                    sample_id=sample.sample_id,
                    query_task_id=query_task.query_task_id,
                    error_type=exc.__class__.__name__,
                    error_message=self._compact_error_text(exc, limit=500),
                )
                raise
            pre_reflection_answer_text = str(answer_response.text or "")
            pre_reflection_event_id = retrieval_bundle.retrieval_event_id
            pre_reflection_answer_metadata = dict(answer_response.metadata or {})
            first_attempt_initial_stage_metadata = {
                key: value
                for key, value in pre_reflection_answer_metadata.items()
                if str(key).startswith("answer_stage_initial_")
            }
            pre_reflection_supporting_refs = list(
                pre_reflection_answer_metadata.get("answer_synthesis_supporting_refs")
                or dict(
                    pre_reflection_answer_metadata.get("answer_synthesis_payload") or {}
                ).get("supporting_source_refs")
                or []
            )
            pre_reflection_invalid_supporting_refs = list(
                pre_reflection_answer_metadata.get("invalid_supporting_refs") or []
            )
            pre_reflection_prompt_tokens = int(
                pre_reflection_answer_metadata.get(
                    "answer_stage_post_validation_prompt_tokens"
                )
                or answer_response.prompt_tokens
                or 0
            )
            pre_reflection_completion_tokens = int(
                pre_reflection_answer_metadata.get(
                    "answer_stage_post_validation_completion_tokens"
                )
                or answer_response.completion_tokens
                or 0
            )
            retrieval_bundle, answer_prompt, answer_response, reflection_retry_metadata = (
                self._maybe_retry_with_retrieval_reflection(
                    sample_id=sample.sample_id,
                    query_sample=query_sample,
                    query_task=query_task,
                    retrieval_bundle=retrieval_bundle,
                    answer_prompt=answer_prompt,
                    answer_response=answer_response,
                )
            )
            answer_metadata = {
                **dict(answer_response.metadata),
                **reflection_retry_metadata,
                **first_attempt_initial_stage_metadata,
                "answer_stage_pre_reflection_text": pre_reflection_answer_text,
                "answer_stage_pre_reflection_retrieval_event_id": pre_reflection_event_id,
                "answer_stage_pre_reflection_supporting_refs": pre_reflection_supporting_refs,
                "answer_stage_pre_reflection_invalid_supporting_refs": (
                    pre_reflection_invalid_supporting_refs
                ),
                "answer_stage_pre_reflection_prompt_tokens": pre_reflection_prompt_tokens,
                "answer_stage_pre_reflection_completion_tokens": (
                    pre_reflection_completion_tokens
                ),
                "answer_stage_final_text": str(answer_response.text or ""),
                "answer_stage_reflection_changed": (
                    pre_reflection_answer_text.strip() != str(answer_response.text or "").strip()
                ),
            }
            retrieval_compact_diagnostics = self._retrieval_compact_diagnostics_for_details(
                dict(retrieval_bundle.metadata or {})
            )
            answer_record = AnswerRecord(
                id=f"{query_task.query_task_id}-answer",
                sample_id=sample.sample_id,
                query_task_id=query_task.query_task_id,
                retrieval_event_id=retrieval_bundle.retrieval_event_id,
                model_name=self.llm_provider.model_info().model_name,
                answer_text=answer_response.text,
                prompt_tokens=answer_response.prompt_tokens,
                completion_tokens=answer_response.completion_tokens,
                metadata_json={
                    "subset_key": query_sample.subset_key,
                    "scene_tag": query_sample.scene_tag,
                    "answer_prompt": answer_prompt,
                    "answer_prompt_name": answer_response.metadata.get("answer_prompt_name"),
                    "retrieval_source_refs": retrieval_bundle.source_message_refs,
                    "retrieval_source_ids": retrieval_bundle.source_message_ids,
                    "answer_metadata": answer_metadata,
                    "retrieval_reflection": reflection_retry_metadata,
                },
            )
            self.store.record_answer(answer_record)
            partial_results.append(
                (
                    AnswerResult(
                        sample_id=query_sample.sample_id,
                        dataset_name=query_sample.dataset_name,
                        subset_key=str(query_sample.subset_key or self.adapter.subset_key(query_sample)),
                        scene_tag=query_sample.scene_tag or self.adapter.scene_tag(query_sample),
                        query_task_id=query_task.query_task_id,
                        question=query_task.question,
                        gold_answer=query_task.gold_answer,
                        rubric=str(query_task.metadata.get("test_point")) if query_task.metadata.get("test_point") else None,
                        answer_prompt=answer_prompt,
                        answer_text=answer_response.text,
                        answer_record_id=answer_record.id,
                        retrieval_event_id=retrieval_bundle.retrieval_event_id,
                        retrieval_source_refs=list(retrieval_bundle.source_message_refs),
                        retrieval_source_ids=list(retrieval_bundle.source_message_ids),
                        metadata={
                            "source_file": query_sample.payload.get("source_file"),
                            "query_index": index,
                            "query_count": len(query_entries),
                            "retrieval_latency_ms": retrieval_bundle.latency_ms,
                            "answer_latency_ms": float(answer_response.metadata.get("latency_ms", 0.0)),
                            "answer_prompt_tokens": int(answer_response.prompt_tokens or 0),
                            "answer_completion_tokens": int(answer_response.completion_tokens or 0),
                            "answer_prompt_name": answer_response.metadata.get("answer_prompt_name"),
                            "answer_metadata": answer_metadata,
                            "retrieval_compact_diagnostics": retrieval_compact_diagnostics,
                            **reflection_retry_metadata,
                            "cache_hit": memory_info["cache_hit"],
                            "memory_build_latency_ms": float(memory_info["memory_build_latency_ms"]),
                            "cache_load_latency_ms": float(memory_info["cache_load_latency_ms"]),
                            "extraction_repair_count": int(memory_info.get("extraction_repair_count", 0)),
                            "extraction_fallback_count": int(memory_info.get("extraction_fallback_count", 0)),
                            "closed_on_fallback_count": int(memory_info.get("closed_on_fallback_count", 0)),
                            "structured_attempt_count": int(memory_info.get("structured_attempt_count", 0)),
                            "structured_success_count": int(memory_info.get("structured_success_count", 0)),
                            "structured_fallback_count": int(memory_info.get("structured_fallback_count", 0)),
                            "structured_attempts_by_task": dict(memory_info.get("structured_attempts_by_task", {})),
                            "structured_successes_by_task": dict(memory_info.get("structured_successes_by_task", {})),
                            "structured_fallbacks_by_task": dict(memory_info.get("structured_fallbacks_by_task", {})),
                            "structured_attempts_by_vendor": dict(memory_info.get("structured_attempts_by_vendor", {})),
                            "structured_successes_by_vendor": dict(memory_info.get("structured_successes_by_vendor", {})),
                            "structured_fallbacks_by_vendor": dict(memory_info.get("structured_fallbacks_by_vendor", {})),
                            "trajectory_match_total_open": int(memory_info.get("trajectory_match_total_open", 0)),
                            "trajectory_match_prefiltered_count": int(memory_info.get("trajectory_match_prefiltered_count", 0)),
                            "trajectory_match_shortlist_count": int(memory_info.get("trajectory_match_shortlist_count", 0)),
                            "wiki_compile_latency_ms": float(memory_info.get("wiki_compile_latency_ms", 0.0)),
                            "link_salvage_count": int(memory_info.get("link_salvage_count", 0)),
                            "link_exchange_fallback_count": int(memory_info.get("link_exchange_fallback_count", 0)),
                            "ops_parse_failure_count": int(memory_info.get("ops_parse_failure_count", 0)),
                            "ops_ignored_count": int(memory_info.get("ops_ignored_count", 0)),
                            "ops_synthesized_count": int(memory_info.get("ops_synthesized_count", 0)),
                            "ops_model_hint_count": int(memory_info.get("ops_model_hint_count", 0)),
                            "ops_model_supplied_count": int(memory_info.get("ops_model_supplied_count", 0)),
                            "claims_parse_failure_count": int(memory_info.get("claims_parse_failure_count", 0)),
                            "claims_required_repair_count": int(memory_info.get("claims_required_repair_count", 0)),
                            "claim_text_exact_match_count": int(memory_info.get("claim_text_exact_match_count", 0)),
                            "claim_status_updated_count": int(memory_info.get("claim_status_updated_count", 0)),
                            "claim_new_add_count": int(memory_info.get("claim_new_add_count", 0)),
                            "claim_unmatched_previous_count": int(memory_info.get("claim_unmatched_previous_count", 0)),
                            "claim_transition_judge_attempt_count": int(memory_info.get("claim_transition_judge_attempt_count", 0)),
                            "claim_transition_judge_success_count": int(memory_info.get("claim_transition_judge_success_count", 0)),
                            "claim_transition_judge_fallback_count": int(memory_info.get("claim_transition_judge_fallback_count", 0)),
                            "claim_transition_revise_count": int(memory_info.get("claim_transition_revise_count", 0)),
                            "claim_transition_add_count": int(memory_info.get("claim_transition_add_count", 0)),
                            "episodic_batch_call_count": int(memory_info.get("episodic_batch_call_count", 0)),
                            "episodic_batch_item_count": int(memory_info.get("episodic_batch_item_count", 0)),
                            "episodic_batch_repair_after_batch_count": int(
                                memory_info.get("episodic_batch_repair_after_batch_count", 0)
                            ),
                            "episodic_batch_oom_backoff_count": int(
                                memory_info.get("episodic_batch_oom_backoff_count", 0)
                            ),
                            "episodic_batch_effective_size_max": int(
                                memory_info.get("episodic_batch_effective_size_max", 1)
                            ),
                            "episodic_batch_effective_size_final": int(
                                memory_info.get("episodic_batch_effective_size_final", 1)
                            ),
                            "memory_stage_batch_stats": dict(memory_info.get("memory_stage_batch_stats", {})),
                            "empty_repair_target_count": int(memory_info.get("empty_repair_target_count", 0)),
                            "assistant_only_exchange_count": int(
                                memory_info.get("assistant_only_exchange_count", 0)
                            ),
                            "forced_memory_seed_count": int(memory_info.get("forced_memory_seed_count", 0)),
                            "low_salience_memory_count": int(memory_info.get("low_salience_memory_count", 0)),
                            "llm_no_memory_forced_count": int(memory_info.get("llm_no_memory_forced_count", 0)),
                            "zero_claim_episodic_candidate_count": int(
                                memory_info.get("zero_claim_episodic_candidate_count", 0)
                            ),
                            "zero_claim_episodic_persisted_count": int(
                                memory_info.get("zero_claim_episodic_persisted_count", 0)
                            ),
                            "zero_claim_low_salience_skipped_count": int(
                                memory_info.get("zero_claim_low_salience_skipped_count", 0)
                            ),
                            "memory_debug_artifact_paths": list(memory_info.get("memory_debug_artifact_paths", [])),
                        },
                    ),
                    query_task.metadata,
                )
            )
        self._commit_session(stage="sample_answers", sample_id=sample.sample_id)
        backbone_usage = self._meter_diff(self.llm_provider, backbone_start)
        sample_runtime_ms = (time.perf_counter() - sample_started) * 1000.0
        self._trace(
            f"sample={sample.sample_id} complete queries={len(query_entries)} "
            f"runtime_ms={sample_runtime_ms:.1f} backbone_calls={int(backbone_usage.get('call_count', 0))} "
            f"backbone_tokens={int(backbone_usage.get('total_tokens', 0))}"
        )
        self._emit_event(
            "sample_done",
            stage="sample_done",
            sample_id=sample.sample_id,
            queries=len(query_entries),
            runtime_ms=sample_runtime_ms,
            backbone_calls=int(backbone_usage.get("call_count", 0)),
            backbone_tokens=int(backbone_usage.get("total_tokens", 0)),
        )

        finalized_results: list[AnswerResult] = []
        for answer_result, query_metadata in partial_results:
            answer_result.metadata.update(
                {
                    "query_metadata": dict(query_metadata),
                    "sample_runtime_ms": sample_runtime_ms / per_query_share,
                    "backbone_usage": self._split_usage(backbone_usage, per_query_share),
                }
            )
            finalized_results.append(answer_result)
        return finalized_results

    def _select_logical_sample_groups(
        self, samples: list[DatasetSample]
    ) -> tuple[list[LogicalSampleGroup], list[DatasetSample], dict[str, Any]]:
        if isinstance(self.adapter, LocomoAdapter):
            groups, excluded_rows = self._select_locomo_logical_sample_groups(samples)
        elif isinstance(self.adapter, MedMTAdapter):
            groups, excluded_rows = self._select_medmt_logical_sample_groups(samples)
        else:
            groups = []
            excluded_rows = [sample for sample in samples if sample.excluded]
            active_samples = [sample for sample in samples if not sample.excluded]
            budget = self.config.max_samples
            if budget is not None:
                active_samples = active_samples[:budget]
            for sample in active_samples:
                groups.append(
                    LogicalSampleGroup(
                        group_id=sample.sample_id,
                        memory_sample=sample,
                        query_rows=[sample],
                        excluded_rows=[],
                    )
                )
        selected_counts_by_subset = Counter(
            str(row.subset_key or self.adapter.subset_key(row) or "unknown")
            for group in groups
            for row in group.query_rows
        )
        selected_query_count = sum(
            len(self.adapter.build_query_tasks(row))
            for group in groups
            for row in group.query_rows
        )
        selection_meta = {
            "logical_sample_count": len(groups),
            "selected_row_count": sum(len(group.query_rows) for group in groups),
            "selected_query_count": selected_query_count,
            "selection_strategy": self._selection_strategy_name(),
            "selected_counts_by_subset": dict(selected_counts_by_subset),
        }
        return groups, excluded_rows, selection_meta

    def _selection_strategy_name(self) -> str:
        if isinstance(self.adapter, LocomoAdapter):
            return "locomo-logical-sample"
        if isinstance(self.adapter, MedMTAdapter):
            if self.dataset_scope_key == self.adapter.all_subset_key and self.config.max_samples is not None:
                return "medmt-round-robin-all"
            return "medmt-sequential"
        return "sequential"

    def _select_locomo_logical_sample_groups(
        self, samples: list[DatasetSample]
    ) -> tuple[list[LogicalSampleGroup], list[DatasetSample]]:
        grouped_rows: dict[str, list[DatasetSample]] = {}
        for sample in samples:
            grouped_rows.setdefault(sample.sample_id, []).append(sample)

        selected_groups: list[LogicalSampleGroup] = []
        excluded_rows: list[DatasetSample] = []
        budget = self.config.max_samples

        for sample_id, rows in grouped_rows.items():
            active_rows = [row for row in rows if not row.excluded]
            excluded_rows.extend(row for row in rows if row.excluded)
            if not active_rows:
                continue
            if budget is not None and len(selected_groups) >= budget:
                break
            selected_groups.append(
                LogicalSampleGroup(
                    group_id=sample_id,
                    memory_sample=active_rows[0],
                    query_rows=active_rows,
                    excluded_rows=[row for row in rows if row.excluded],
                )
            )
        return selected_groups, excluded_rows

    def _select_medmt_logical_sample_groups(
        self, samples: list[DatasetSample]
    ) -> tuple[list[LogicalSampleGroup], list[DatasetSample]]:
        if self.dataset_scope_key == self.adapter.all_subset_key and self.config.max_samples is not None:
            return self._select_medmt_round_robin_groups(samples)
        return self._select_medmt_sequential_groups(samples)

    def _select_medmt_round_robin_groups(
        self, samples: list[DatasetSample]
    ) -> tuple[list[LogicalSampleGroup], list[DatasetSample]]:
        by_subset: dict[str, list[DatasetSample]] = {subset_key: [] for subset_key in MEDMT_SUBSET_ORDER}
        excluded_rows: list[DatasetSample] = []
        for sample in samples:
            subset_key = str(sample.subset_key or self.adapter.subset_key(sample) or "unknown")
            if sample.excluded:
                excluded_rows.append(sample)
                continue
            by_subset.setdefault(subset_key, []).append(sample)

        selected_groups: list[LogicalSampleGroup] = []
        budget = self.config.max_samples or 0
        while len(selected_groups) < budget:
            made_progress = False
            for subset_key in MEDMT_SUBSET_ORDER:
                bucket = by_subset.get(subset_key, [])
                if not bucket:
                    continue
                sample = bucket.pop(0)
                selected_groups.append(
                    LogicalSampleGroup(
                        group_id=sample.sample_id,
                        memory_sample=sample,
                        query_rows=[sample],
                        excluded_rows=[],
                    )
                )
                made_progress = True
                if len(selected_groups) >= budget:
                    break
            if not made_progress:
                break
        return selected_groups, excluded_rows

    def _select_medmt_sequential_groups(
        self, samples: list[DatasetSample]
    ) -> tuple[list[LogicalSampleGroup], list[DatasetSample]]:
        selected_groups: list[LogicalSampleGroup] = []
        excluded_rows: list[DatasetSample] = []
        budget = self.config.max_samples
        for sample in samples:
            if sample.excluded:
                excluded_rows.append(sample)
                continue
            if budget is not None and len(selected_groups) >= budget:
                break
            selected_groups.append(
                LogicalSampleGroup(
                    group_id=sample.sample_id,
                    memory_sample=sample,
                    query_rows=[sample],
                    excluded_rows=[],
                )
            )
        return selected_groups, excluded_rows

    def _prepare_raw_messages(self, sample: DatasetSample) -> list[NormalizedMessage]:
        assert self.store is not None
        messages = self.adapter.iterate_turns(sample)
        for message in messages:
            record = self.store.add_raw_message(sample.sample_id, sample.dataset_name, message)
            message.raw_message_id = record.id
        return messages

    def _resolve_memory_extract_batch_size(self) -> int:
        info = self.llm_provider.model_info()
        configured = self.config.memory_extract_batch_size
        if info.provider_kind == "local":
            if configured == "auto":
                parameter_billions = info.parameter_billions
                if parameter_billions is None:
                    parameter_billions = infer_parameter_size_b(info.model_name)
                if parameter_billions is None:
                    resolved = 1
                elif parameter_billions >= 30:
                    resolved = 2
                elif parameter_billions >= 13:
                    resolved = 4
                else:
                    resolved = 8
            else:
                resolved = int(configured)
        elif (
            info.provider_kind in REMOTE_LIKE_PROVIDER_KINDS
            and self.llm_provider.supports_structured("episodic_extract")
        ):
            if configured == "auto":
                resolved = min(8, max(1, 8 // max(int(self.config.conv_workers), 1)))
            else:
                resolved = int(configured)
        else:
            resolved = 1
        if self.memory_extract_batch_size_cap is not None:
            resolved = min(resolved, self.memory_extract_batch_size_cap)
        resolved = max(int(resolved), 1)
        self.episodic_batch_effective_size_final = resolved
        self.episodic_batch_effective_size_max = max(self.episodic_batch_effective_size_max, resolved)
        return resolved

    def _episodic_batch_mode(self) -> str:
        return self._memory_stage_batch_mode("episodic_extract")

    def _memory_stage_batch_mode(self, task: str) -> str:
        info = self.llm_provider.model_info()
        if info.provider_kind == "local":
            return "local_text_batch"
        if info.provider_kind in REMOTE_LIKE_PROVIDER_KINDS and self.llm_provider.supports_structured(task):
            return "structured_concurrent"
        return "serial"

    def _bump_memory_stage_batch_stat(self, stage: str, key: str, amount: int = 1) -> None:
        stats = self.memory_stage_batch_stats.setdefault(
            stage,
            {"calls": 0, "items": 0, "fallbacks": 0, "structured_failures": 0},
        )
        stats[key] = int(stats.get(key, 0)) + amount

    def _structured_vendor(self) -> str:
        return str(self.llm_provider.model_info().metadata.get("vendor") or "unknown")

    @staticmethod
    def _compact_error_text(error: object, *, limit: int = 180) -> str:
        if error is None:
            return "none"
        text = " ".join(str(error).split())
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text or error.__class__.__name__

    @staticmethod
    def _structured_error_type(outcome: StructuredFirstPassAttempt) -> str:
        return outcome.error_type or (outcome.error.__class__.__name__ if outcome.error is not None else "none")

    @staticmethod
    def _structured_error_message(outcome: StructuredFirstPassAttempt) -> str:
        return outcome.error_message or PipelineRunner._compact_error_text(outcome.error)

    def _trace_structured_schema_check(self, task: str) -> None:
        if not self.config.verbose:
            return
        vendor = self._structured_vendor()
        key = (task, vendor)
        if key in self._structured_schema_checks_logged:
            return
        self._structured_schema_checks_logged.add(key)
        diagnostics = structured_schema_diagnostics(
            vendor_schema(get_structured_task_spec(task), vendor),
            vendor=vendor,
        )
        self._trace(
            f"structured_schema_static_check task={task} vendor={vendor} "
            f"ok={str(bool(diagnostics['ok'])).lower()} "
            f"missing_required={int(diagnostics['missing_required'])} "
            f"missing_additional_properties={int(diagnostics['missing_additional_properties'])} "
            f"unsupported_keywords={int(diagnostics['unsupported_keywords'])}"
        )

    @staticmethod
    def _memory_stage_batch_stats_diff(
        current: dict[str, dict[str, int]],
        before: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int]]:
        stage_names = set(current) | set(before)
        diff: dict[str, dict[str, int]] = {}
        for stage in sorted(stage_names):
            keys = set(current.get(stage, {})) | set(before.get(stage, {}))
            stage_diff = {
                key: int(current.get(stage, {}).get(key, 0)) - int(before.get(stage, {}).get(key, 0))
                for key in keys
            }
            if any(value for value in stage_diff.values()):
                diff[stage] = stage_diff
        return diff

    @staticmethod
    def _copy_memory_stage_batch_stats(value: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        return {stage: dict(stats) for stage, stats in value.items()}

    @staticmethod
    def _compact_precomputed_metadata_value(value: Any, *, depth: int = 0) -> Any:
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
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:500]
        if depth >= 4:
            return str(value)[:200]
        if isinstance(value, dict):
            compact: dict[str, Any] = {}
            for key, nested in value.items():
                key_text = str(key)
                if key_text in unsafe_keys:
                    continue
                compact[key_text] = PipelineRunner._compact_precomputed_metadata_value(
                    nested,
                    depth=depth + 1,
                )
            return compact
        if isinstance(value, (list, tuple)):
            return [
                PipelineRunner._compact_precomputed_metadata_value(item, depth=depth + 1)
                for item in list(value)[:50]
            ]
        return str(value)[:200]

    @classmethod
    def _compact_precomputed_response_metadata(
        cls,
        response_metadata: dict[str, Any] | None,
        *,
        batch_mode: str,
    ) -> dict[str, Any]:
        allowed_keys = {
            "estimated_usage",
            "device_plan",
            "used_chat_template",
            "chat_template_fallback",
            "generation_budget_tokens",
            "generation_kwargs",
            "batched",
            "batch_size",
            "batch_index",
            "latency_ms",
            "batch_wall_time_ms",
            "meter_role",
            "provider_call_kind",
        }
        metadata = dict(response_metadata or {})
        compact = {
            key: cls._compact_precomputed_metadata_value(metadata[key])
            for key in allowed_keys
            if key in metadata
        }
        compact["batch_mode"] = batch_mode
        return compact

    @staticmethod
    def _ordered_exchanges(messages: list[NormalizedMessage]) -> list[list[NormalizedMessage]]:
        exchanges: list[list[NormalizedMessage]] = []
        system_messages: list[NormalizedMessage] = []
        current_user: NormalizedMessage | None = None

        def session_key(message: NormalizedMessage) -> str | None:
            value = message.metadata.get("session_date") if isinstance(message.metadata, dict) else None
            return str(value or message.occurred_at or "").strip() or None

        for message in messages:
            if message.role == "system":
                system_messages.append(message)
                continue
            if message.role == "user":
                current_user = message
                continue
            if message.role == "assistant":
                if current_user is not None and session_key(current_user) == session_key(message):
                    exchanges.append([*system_messages, current_user, message])
                    current_user = None
                else:
                    current_user = None
                    exchanges.append([*system_messages, message])
        return exchanges

    @staticmethod
    def _assistant_only_exchange_count(exchanges: list[list[NormalizedMessage]]) -> int:
        count = 0
        for exchange in exchanges:
            non_system_roles = [message.role for message in exchange if message.role != "system"]
            if non_system_roles == ["assistant"]:
                count += 1
        return count

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        message = str(exc).lower()
        return "cuda out of memory" in message or "cuda error: out of memory" in message or "out of memory" in message

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _build_episodic_batch_messages(
        self,
        exchanges: list[list[NormalizedMessage]],
    ) -> list[list[NormalizedMessage]]:
        assert self.orchestrator is not None
        prompt = load_prompt("episodic_extract")
        batch_messages: list[list[NormalizedMessage]] = []
        for exchange_messages in exchanges:
            batch_messages.append(
                [
                    NormalizedMessage(
                        role="user",
                        content=prompt + "\n\nConversation:\n" + self.orchestrator._render_conversation(exchange_messages),
                        turn_index=0,
                    )
                ]
            )
        return batch_messages

    def _build_claim_text_batch_messages(
        self,
        states: list[dict[str, Any]],
    ) -> list[list[NormalizedMessage]]:
        assert self.orchestrator is not None
        return [
            self.orchestrator.build_claim_text_first_pass_messages(
                state["episodic"],
                list(state["exchange_messages"]),
            )
            for state in states
        ]

    def _build_claim_signal_batch_messages(
        self,
        states: list[dict[str, Any]],
    ) -> list[list[NormalizedMessage]]:
        assert self.orchestrator is not None
        return [
            self.orchestrator.build_claim_signal_first_pass_messages(
                state["episodic"],
                list(state["exchange_messages"]),
            )
            for state in states
        ]

    @staticmethod
    def _exchange_number(exchange_label: str) -> int:
        return int(exchange_label.rsplit("exchange=", 1)[-1])

    def _run_structured_concurrent_episodic_window(
        self,
        sample: DatasetSample,
        current_states: list[dict[str, Any]],
        *,
        valid_link_ids: set[str],
    ) -> tuple[int, int, float]:
        assert self.orchestrator is not None
        self._trace_structured_schema_check("episodic_extract")
        batch_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(current_states)) as executor:
            outcomes = list(
                executor.map(
                    self.orchestrator.request_episodic_structured_first_pass,
                    [list(state["exchange_messages"]) for state in current_states],
                )
            )
        batch_latency_ms = (time.perf_counter() - batch_started) * 1000.0
        self.episodic_batch_call_count += 1
        self.episodic_batch_item_count += len(current_states)
        self._bump_memory_stage_batch_stat("episodic_extract", "calls", 1)
        self._bump_memory_stage_batch_stat("episodic_extract", "items", len(current_states))
        success_count = 0
        parse_fail_count = 0
        for index, (state, outcome) in enumerate(zip(current_states, outcomes, strict=False)):
            exchange_label = str(state["exchange_label"])
            exchange_messages = list(state["exchange_messages"])
            if outcome.response is not None:
                self._trace(
                    f"sample={sample.sample_id} episodic_batch_item exchange={self._exchange_number(exchange_label)} "
                    f"mode=structured_concurrent attempt=1 prompt_tokens={int(outcome.response.prompt_tokens or 0)} "
                    f"completion_tokens={int(outcome.response.completion_tokens or 0)}"
                )
            else:
                self._bump_memory_stage_batch_stat("episodic_extract", "structured_failures", 1)
                self._trace(
                    f"sample={sample.sample_id} episodic_batch_item exchange={self._exchange_number(exchange_label)} "
                    f"mode=structured_concurrent attempt=1 structured_failure=true "
                    f"error_type={self._structured_error_type(outcome)} "
                    f"error={self._structured_error_message(outcome)}"
                )
            episodic_started = time.perf_counter()
            finalized = self.orchestrator.finalize_episodic_structured_first_pass(
                sample.sample_id,
                exchange_messages,
                valid_link_ids,
                outcome,
                batched_first_pass_metadata={
                    "batched_attempt": True,
                    "batch_size": len(current_states),
                    "batch_index": index,
                    "batch_latency_ms": batch_latency_ms,
                    "batch_mode": "structured_concurrent",
                },
            )
            episodic = finalized.parsed_memory
            state["episodic"] = episodic
            self._trace(
                f"{exchange_label} episodic_extract done has_memory={episodic is not None} "
                f"latency_ms={(time.perf_counter() - episodic_started) * 1000.0:.1f}"
            )
            if finalized.required_repair_or_fallback:
                self.episodic_batch_repair_after_batch_count += 1
                parse_fail_count += 1
            else:
                success_count += 1
        return success_count, parse_fail_count, batch_latency_ms

    def _generate_local_memory_stage_batch(
        self,
        *,
        sample_id: str,
        stage_name: str,
        task: str,
        batch_messages: list[list[NormalizedMessage]],
    ) -> list[Any] | None:
        try:
            responses = self.llm_provider.generate_batch(
                batch_messages,
                metadata={"task": task, "memory_type": "episodic"},
            )
            self._bump_memory_stage_batch_stat(stage_name, "calls", 1)
            self._bump_memory_stage_batch_stat(stage_name, "items", len(batch_messages))
            return responses
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc) and len(batch_messages) > 1:
                new_batch_size = max(len(batch_messages) // 2, 1)
                self.episodic_batch_oom_backoff_count += 1
                self.memory_extract_batch_size_cap = (
                    new_batch_size
                    if self.memory_extract_batch_size_cap is None
                    else min(self.memory_extract_batch_size_cap, new_batch_size)
                )
                self._clear_cuda_cache()
                self._trace(
                    f"sample={sample_id} {stage_name}_batch_backoff from={len(batch_messages)} "
                    f"to={self._resolve_memory_extract_batch_size()} reason=oom falling_back=serial"
                )
                return None
            raise

    def _run_claim_text_stage_for_window(self, sample: DatasetSample, memory_states: list[dict[str, Any]]) -> None:
        if not memory_states:
            return
        assert self.orchestrator is not None
        task = "episodic_claim_text_extract"
        stage_name = "claim_text"
        batch_mode = self._memory_stage_batch_mode(task)
        exchange_numbers = [self._exchange_number(str(state["exchange_label"])) for state in memory_states]
        if batch_mode == "structured_concurrent":
            self._trace_structured_schema_check(task)
            self._trace(
                f"sample={sample.sample_id} claim_text_batch_start mode={batch_mode} "
                f"batch_size={len(memory_states)} exchanges={exchange_numbers}"
            )
            started_at = time.perf_counter()
            try:
                with ThreadPoolExecutor(max_workers=len(memory_states)) as executor:
                    outcomes = list(
                        executor.map(
                            lambda state: self.orchestrator.request_claim_text_structured_first_pass(
                                state["episodic"],
                                list(state["exchange_messages"]),
                                exchange_number=self._exchange_number(str(state["exchange_label"])),
                            ),
                            memory_states,
                        )
                    )
                self._bump_memory_stage_batch_stat(stage_name, "calls", 1)
                self._bump_memory_stage_batch_stat(stage_name, "items", len(memory_states))
                request_failures = sum(1 for outcome in outcomes if outcome.error is not None)
                request_successes = len(outcomes) - request_failures
                if request_failures:
                    self._bump_memory_stage_batch_stat(stage_name, "structured_failures", request_failures)
                for state, outcome in zip(memory_states, outcomes, strict=False):
                    response = outcome.response
                    self._trace(
                        f"sample={sample.sample_id} claim_text_batch_item exchange={self._exchange_number(str(state['exchange_label']))} "
                        f"mode={batch_mode} prompt_tokens={int(getattr(response, 'prompt_tokens', 0) or 0)} "
                        f"completion_tokens={int(getattr(response, 'completion_tokens', 0) or 0)} "
                        f"structured_failure={str(outcome.error is not None).lower()} "
                        f"error_type={self._structured_error_type(outcome)} "
                        f"error={self._structured_error_message(outcome)}"
                    )
                    state["episodic"] = self.orchestrator._apply_claim_text_llm_stage(
                        sample.sample_id,
                        state["episodic"],
                        list(state["exchange_messages"]),
                        structured_first_pass=outcome,
                    )
                self._trace(
                    f"sample={sample.sample_id} claim_text_batch_done mode={batch_mode} "
                    f"request_successes={request_successes} request_failures={request_failures} "
                    f"fallbacks={request_failures} latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._bump_memory_stage_batch_stat(stage_name, "fallbacks", len(memory_states))
                self._trace(
                    f"sample={sample.sample_id} claim_text_batch_dispatch_failed mode={batch_mode} "
                    f"batch_size={len(memory_states)} reason={exc} falling_back=serial"
                )
        elif batch_mode == "local_text_batch":
            self._trace(
                f"sample={sample.sample_id} claim_text_batch_start mode={batch_mode} "
                f"batch_size={len(memory_states)} exchanges={exchange_numbers}"
            )
            batch_messages = self._build_claim_text_batch_messages(memory_states)
            responses = self._generate_local_memory_stage_batch(
                sample_id=sample.sample_id,
                stage_name=stage_name,
                task=task,
                batch_messages=batch_messages,
            )
            if responses is not None:
                started_at = time.perf_counter()
                for index, (state, response) in enumerate(zip(memory_states, responses, strict=False)):
                    self._trace(
                        f"sample={sample.sample_id} claim_text_batch_item exchange={self._exchange_number(str(state['exchange_label']))} "
                        f"mode={batch_mode} prompt_tokens={int(response.prompt_tokens or 0)} "
                        f"completion_tokens={int(response.completion_tokens or 0)}"
                    )
                    first_attempt = PrecomputedGenerationAttempt(
                        text=response.text,
                        prompt_text=batch_messages[index][-1].content,
                        generation_latency_ms=float(response.metadata.get("latency_ms", 0.0)),
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        response_metadata=self._compact_precomputed_response_metadata(
                            dict(response.metadata),
                            batch_mode=batch_mode,
                        ),
                        batch_size=int(response.metadata.get("batch_size", len(memory_states))),
                        batch_index=int(response.metadata.get("batch_index", index)),
                    )
                    state["episodic"] = self.orchestrator._apply_claim_text_llm_stage(
                        sample.sample_id,
                        state["episodic"],
                        list(state["exchange_messages"]),
                        first_attempt=first_attempt,
                    )
                self._trace(
                    f"sample={sample.sample_id} claim_text_batch_done mode={batch_mode} "
                    f"latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
                )
                return
            self._bump_memory_stage_batch_stat(stage_name, "fallbacks", len(memory_states))

        for state in memory_states:
            state["episodic"] = self.orchestrator._apply_claim_text_llm_stage(
                sample.sample_id,
                state["episodic"],
                list(state["exchange_messages"]),
            )

    def _run_claim_signal_stage_for_window(self, sample: DatasetSample, memory_states: list[dict[str, Any]]) -> None:
        signal_states = [state for state in memory_states if getattr(state.get("episodic"), "claims", None)]
        if not signal_states:
            return
        assert self.orchestrator is not None
        task = "claim_signal_extract"
        stage_name = "claim_signal"
        batch_mode = self._memory_stage_batch_mode(task)
        exchange_numbers = [self._exchange_number(str(state["exchange_label"])) for state in signal_states]
        if batch_mode == "structured_concurrent":
            self._trace_structured_schema_check(task)
            self._trace(
                f"sample={sample.sample_id} claim_signal_batch_start mode={batch_mode} "
                f"batch_size={len(signal_states)} exchanges={exchange_numbers}"
            )
            started_at = time.perf_counter()
            try:
                with ThreadPoolExecutor(max_workers=len(signal_states)) as executor:
                    outcomes = list(
                        executor.map(
                            lambda state: self.orchestrator.request_claim_signal_structured_first_pass(
                                state["episodic"],
                                list(state["exchange_messages"]),
                                exchange_number=self._exchange_number(str(state["exchange_label"])),
                            ),
                            signal_states,
                        )
                    )
                self._bump_memory_stage_batch_stat(stage_name, "calls", 1)
                self._bump_memory_stage_batch_stat(stage_name, "items", len(signal_states))
                request_failures = sum(1 for outcome in outcomes if outcome.error is not None)
                request_successes = len(outcomes) - request_failures
                if request_failures:
                    self._bump_memory_stage_batch_stat(stage_name, "structured_failures", request_failures)
                for state, outcome in zip(signal_states, outcomes, strict=False):
                    response = outcome.response
                    self._trace(
                        f"sample={sample.sample_id} claim_signal_batch_item exchange={self._exchange_number(str(state['exchange_label']))} "
                        f"mode={batch_mode} prompt_tokens={int(getattr(response, 'prompt_tokens', 0) or 0)} "
                        f"completion_tokens={int(getattr(response, 'completion_tokens', 0) or 0)} "
                        f"structured_failure={str(outcome.error is not None).lower()} "
                        f"error_type={self._structured_error_type(outcome)} "
                        f"error={self._structured_error_message(outcome)}"
                    )
                    state["claim_signal_structured_first_pass"] = outcome
                self._trace(
                    f"sample={sample.sample_id} claim_signal_batch_done mode={batch_mode} "
                    f"request_successes={request_successes} request_failures={request_failures} "
                    f"fallbacks={request_failures} latency_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._bump_memory_stage_batch_stat(stage_name, "fallbacks", len(signal_states))
                self._trace(
                    f"sample={sample.sample_id} claim_signal_batch_dispatch_failed mode={batch_mode} "
                    f"batch_size={len(signal_states)} reason={exc} falling_back=serial"
                )
                return
        if batch_mode == "local_text_batch":
            self._trace(
                f"sample={sample.sample_id} claim_signal_batch_start mode={batch_mode} "
                f"batch_size={len(signal_states)} exchanges={exchange_numbers}"
            )
            batch_messages = self._build_claim_signal_batch_messages(signal_states)
            responses = self._generate_local_memory_stage_batch(
                sample_id=sample.sample_id,
                stage_name=stage_name,
                task=task,
                batch_messages=batch_messages,
            )
            if responses is not None:
                for index, (state, response) in enumerate(zip(signal_states, responses, strict=False)):
                    self._trace(
                        f"sample={sample.sample_id} claim_signal_batch_item exchange={self._exchange_number(str(state['exchange_label']))} "
                        f"mode={batch_mode} prompt_tokens={int(response.prompt_tokens or 0)} "
                        f"completion_tokens={int(response.completion_tokens or 0)}"
                    )
                    state["claim_signal_first_attempt"] = PrecomputedGenerationAttempt(
                        text=response.text,
                        prompt_text=batch_messages[index][-1].content,
                        generation_latency_ms=float(response.metadata.get("latency_ms", 0.0)),
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        response_metadata=self._compact_precomputed_response_metadata(
                            dict(response.metadata),
                            batch_mode=batch_mode,
                        ),
                        batch_size=int(response.metadata.get("batch_size", len(signal_states))),
                        batch_index=int(response.metadata.get("batch_index", index)),
                    )
                self._trace(
                    f"sample={sample.sample_id} claim_signal_batch_done mode={batch_mode}"
                )
                return
            self._bump_memory_stage_batch_stat(stage_name, "fallbacks", len(signal_states))

    def _run_batched_episodic_replay(
        self,
        sample: DatasetSample,
        exchanges: list[list[NormalizedMessage]],
        *,
        valid_link_ids: set[str],
    ) -> None:
        assert self.orchestrator is not None
        effective_batch_size = self._resolve_memory_extract_batch_size()
        batch_mode = self._episodic_batch_mode()
        if effective_batch_size <= 1 or batch_mode == "serial":
            if self._memory_stage_batch_mode("episodic_extract") == "structured_concurrent":
                for task_name in (
                    "episodic_extract",
                    "episodic_claim_text_extract",
                    "claim_signal_extract",
                ):
                    if self.llm_provider.supports_structured(task_name):
                        self._trace_structured_schema_check(task_name)
            for exchange_messages in exchanges:
                self.orchestrator.process_exchange(sample.sample_id, sample.dataset_name, exchange_messages)
                self._commit_session(stage="memory_window_persist", sample_id=sample.sample_id)
            return

        position = 0
        while position < len(exchanges):
            window_size = min(effective_batch_size, len(exchanges) - position)
            window = exchanges[position : position + window_size]
            states: list[dict[str, Any]] = []
            for exchange_messages in window:
                exchange_label, exchange_started = self.orchestrator.begin_exchange(sample.sample_id, exchange_messages)
                states.append(
                    {
                        "exchange_messages": exchange_messages,
                        "exchange_label": exchange_label,
                        "exchange_started": exchange_started,
                    }
                )

            pending_index = 0
            while pending_index < len(states):
                current_batch_size = min(effective_batch_size, len(states) - pending_index)
                current_states = states[pending_index : pending_index + current_batch_size]
                current_exchanges = [state["exchange_messages"] for state in current_states]
                exchange_numbers = [self._exchange_number(str(state["exchange_label"])) for state in current_states]
                if batch_mode == "structured_concurrent":
                    try:
                        self._trace(
                            f"sample={sample.sample_id} episodic_batch_start mode={batch_mode} "
                            f"batch_size={len(current_states)} exchanges={exchange_numbers}"
                        )
                        success_count, parse_fail_count, batch_latency_ms = self._run_structured_concurrent_episodic_window(
                            sample,
                            current_states,
                            valid_link_ids=valid_link_ids,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._trace(
                            f"sample={sample.sample_id} episodic_batch_dispatch_failed mode={batch_mode} "
                            f"batch_size={len(current_states)} reason={exc} falling_back=serial"
                        )
                        success_count = 0
                        parse_fail_count = 0
                        batch_latency_ms = 0.0
                        for state in current_states:
                            exchange_label = str(state["exchange_label"])
                            exchange_messages = list(state["exchange_messages"])
                            episodic_started = time.perf_counter()
                            episodic = self.orchestrator.extract_episodic(
                                sample.sample_id,
                                exchange_messages,
                                valid_link_ids,
                            )
                            self._trace(
                                f"{exchange_label} episodic_extract done has_memory={episodic is not None} "
                                f"latency_ms={(time.perf_counter() - episodic_started) * 1000.0:.1f}"
                            )
                            state["episodic"] = episodic
                            success_count += 1
                else:
                    batch_messages = self._build_episodic_batch_messages(current_exchanges)
                    while True:
                        self._trace(
                            f"sample={sample.sample_id} episodic_batch_start mode={batch_mode} "
                            f"batch_size={len(current_states)} exchanges={exchange_numbers}"
                        )
                        try:
                            responses = self.llm_provider.generate_batch(
                                batch_messages,
                                metadata={"task": "episodic_extract"},
                            )
                            break
                        except Exception as exc:  # noqa: BLE001
                            if not self._is_cuda_oom(exc) or len(current_states) <= 1:
                                raise
                            new_batch_size = max(len(current_states) // 2, 1)
                            self.episodic_batch_oom_backoff_count += 1
                            self.memory_extract_batch_size_cap = (
                                new_batch_size
                                if self.memory_extract_batch_size_cap is None
                                else min(self.memory_extract_batch_size_cap, new_batch_size)
                            )
                            effective_batch_size = self._resolve_memory_extract_batch_size()
                            self._clear_cuda_cache()
                            self._trace(
                                f"sample={sample.sample_id} episodic_batch_backoff from={len(current_states)} "
                                f"to={effective_batch_size} reason=oom"
                            )
                            current_batch_size = min(effective_batch_size, len(states) - pending_index)
                            current_states = states[pending_index : pending_index + current_batch_size]
                            current_exchanges = [state["exchange_messages"] for state in current_states]
                            exchange_numbers = [self._exchange_number(str(state["exchange_label"])) for state in current_states]
                            batch_messages = self._build_episodic_batch_messages(current_exchanges)

                    self.episodic_batch_call_count += 1
                    self.episodic_batch_item_count += len(current_states)
                    self._bump_memory_stage_batch_stat("episodic_extract", "calls", 1)
                    self._bump_memory_stage_batch_stat("episodic_extract", "items", len(current_states))
                    batch_latency_ms = float(
                        responses[0].metadata.get(
                            "batch_wall_time_ms",
                            responses[0].metadata.get("latency_ms", 0.0),
                        )
                    ) if responses else 0.0
                    success_count = 0
                    parse_fail_count = 0
                    for index, (state, response) in enumerate(zip(current_states, responses, strict=False)):
                        exchange_label = str(state["exchange_label"])
                        exchange_messages = list(state["exchange_messages"])
                        self._trace(
                            f"sample={sample.sample_id} episodic_batch_item exchange={self._exchange_number(exchange_label)} "
                            f"mode={batch_mode} attempt=1 prompt_tokens={int(response.prompt_tokens or 0)} "
                            f"completion_tokens={int(response.completion_tokens or 0)}"
                        )
                        episodic_started = time.perf_counter()
                        first_attempt = PrecomputedGenerationAttempt(
                            text=response.text,
                            prompt_text=batch_messages[index][-1].content,
                            generation_latency_ms=float(response.metadata.get("latency_ms", 0.0)),
                            prompt_tokens=response.prompt_tokens,
                            completion_tokens=response.completion_tokens,
                            response_metadata=self._compact_precomputed_response_metadata(
                                dict(response.metadata),
                                batch_mode=batch_mode,
                            ),
                            batch_size=int(response.metadata.get("batch_size", len(current_states))),
                            batch_index=int(response.metadata.get("batch_index", index)),
                        )
                        episodic = self.orchestrator.extract_episodic(
                            sample.sample_id,
                            exchange_messages,
                            valid_link_ids,
                            first_attempt=first_attempt,
                        )
                        self._trace(
                            f"{exchange_label} episodic_extract done has_memory={episodic is not None} "
                            f"latency_ms={(time.perf_counter() - episodic_started) * 1000.0:.1f}"
                        )
                        state["episodic"] = episodic
                        if episodic is not None and episodic.metadata.get("repair_after_batched_attempt"):
                            self.episodic_batch_repair_after_batch_count += 1
                            parse_fail_count += 1
                        else:
                            success_count += 1
                memory_states = [state for state in current_states if state.get("episodic") is not None]
                self._run_claim_text_stage_for_window(sample, memory_states)
                for state in memory_states:
                    state["episodic"] = self.orchestrator._apply_claim_preservation_audit(
                        sample.sample_id,
                        state["episodic"],
                        list(state["exchange_messages"]),
                    )
                self._run_claim_signal_stage_for_window(sample, memory_states)
                for state in current_states:
                    exchange_label = str(state["exchange_label"])
                    exchange_started = float(state["exchange_started"])
                    episodic = state.get("episodic")
                    if episodic is not None:
                        persist_started = time.perf_counter()
                        persisted = self.orchestrator.persist_audited_memory(
                            sample.sample_id,
                            sample.dataset_name,
                            episodic,
                            exchange_messages=list(state["exchange_messages"]),
                            claim_signal_structured_first_pass=state.get("claim_signal_structured_first_pass"),
                            claim_signal_first_attempt=state.get("claim_signal_first_attempt"),
                        )
                        if persisted:
                            self._trace(
                                f"{exchange_label} episodic_persist done latency_ms={(time.perf_counter() - persist_started) * 1000.0:.1f}"
                            )
                        else:
                            self._trace(
                                f"{exchange_label} episodic_persist skipped reason=zero_claim_low_salience "
                                f"latency_ms={(time.perf_counter() - persist_started) * 1000.0:.1f}"
                            )
                    self.orchestrator.complete_exchange(exchange_label, exchange_started)
                self._commit_session(stage="memory_window_persist", sample_id=sample.sample_id)
                self._trace(
                    f"sample={sample.sample_id} episodic_batch_done mode={batch_mode} latency_ms={batch_latency_ms:.1f} "
                    f"success_count={success_count} parse_fail_count={parse_fail_count}"
                )
                pending_index += len(current_states)
            position += window_size

    def _build_or_restore_sample_memory(self, sample: DatasetSample, messages: list[NormalizedMessage]) -> dict[str, Any]:
        assert self.store is not None
        assert self.session is not None
        assert self.orchestrator is not None
        sample_fingerprint, _ = self.cache.sample_fingerprint(sample)
        exchanges = self._ordered_exchanges(messages)
        exchange_count = len(exchanges)
        assistant_only_exchange_count = self._assistant_only_exchange_count(exchanges)
        if self.config.memory_cache_enabled and not self.config.rebuild_memory_cache:
            load_started_at = time.perf_counter()
            bundle, sample_fingerprint = self.cache.load_sample_cache(sample)
            if bundle is not None:
                self.cache.hydrate_sample_cache(self.store, bundle, sample)
                self._commit_session(stage="memory_cache_hydrate", sample_id=sample.sample_id)
                load_latency_ms = (time.perf_counter() - load_started_at) * 1000.0
                self.cache_stats["cache_hits"] += 1.0
                self.cache_stats["cache_load_latency_ms"] += load_latency_ms
                self.cache_stats["memory_build_time_saved_ms"] += float(
                    bundle.memory_stats.get("build_latency_ms", 0.0) or 0.0
                )
                self._trace(
                    f"sample={sample.sample_id} memory_cache hit load_latency_ms={load_latency_ms:.1f} "
                    f"saved_build_latency_ms={float(bundle.memory_stats.get('build_latency_ms', 0.0) or 0.0):.1f}"
                )
                self.orchestrator.zero_claim_episodic_candidate_count += int(
                    bundle.memory_stats.get("zero_claim_episodic_candidate_count", 0) or 0
                )
                self.orchestrator.zero_claim_episodic_persisted_count += int(
                    bundle.memory_stats.get("zero_claim_episodic_persisted_count", 0) or 0
                )
                self.orchestrator.zero_claim_low_salience_skipped_count += int(
                    bundle.memory_stats.get("zero_claim_low_salience_skipped_count", 0) or 0
                )
                return {
                    "cache_hit": True,
                    "memory_build_latency_ms": 0.0,
                    "cache_load_latency_ms": load_latency_ms,
                    "saved_build_latency_ms": float(bundle.memory_stats.get("build_latency_ms", 0.0) or 0.0),
                    "extraction_repair_count": int(bundle.memory_stats.get("repair_rounds", 0) or 0),
                    "extraction_fallback_count": int(bundle.memory_stats.get("fallback_count", 0) or 0),
                    "closed_on_fallback_count": int(bundle.memory_stats.get("closed_on_fallback_count", 0) or 0),
                    "trajectory_match_total_open": int(bundle.memory_stats.get("trajectory_match_total_open", 0) or 0),
                    "trajectory_match_prefiltered_count": int(bundle.memory_stats.get("trajectory_match_prefiltered_count", 0) or 0),
                    "trajectory_match_shortlist_count": int(bundle.memory_stats.get("trajectory_match_shortlist_count", 0) or 0),
                    "wiki_compile_latency_ms": float(bundle.memory_stats.get("wiki_compile_latency_ms", 0.0) or 0.0),
                    "link_salvage_count": int(bundle.memory_stats.get("link_salvage_count", 0) or 0),
                    "link_exchange_fallback_count": int(bundle.memory_stats.get("link_exchange_fallback_count", 0) or 0),
                    "ops_parse_failure_count": int(bundle.memory_stats.get("ops_parse_failure_count", 0) or 0),
                    "ops_ignored_count": int(bundle.memory_stats.get("ops_ignored_count", 0) or 0),
                    "ops_synthesized_count": int(bundle.memory_stats.get("ops_synthesized_count", 0) or 0),
                    "ops_model_hint_count": int(bundle.memory_stats.get("ops_model_hint_count", 0) or 0),
                    "ops_model_supplied_count": int(bundle.memory_stats.get("ops_model_supplied_count", 0) or 0),
                    "claims_parse_failure_count": int(bundle.memory_stats.get("claims_parse_failure_count", 0) or 0),
                    "claims_required_repair_count": int(bundle.memory_stats.get("claims_required_repair_count", 0) or 0),
                    "claim_text_exact_match_count": int(bundle.memory_stats.get("claim_text_exact_match_count", 0) or 0),
                    "claim_status_updated_count": int(bundle.memory_stats.get("claim_status_updated_count", 0) or 0),
                    "claim_new_add_count": int(bundle.memory_stats.get("claim_new_add_count", 0) or 0),
                    "claim_unmatched_previous_count": int(bundle.memory_stats.get("claim_unmatched_previous_count", 0) or 0),
                    "claim_transition_judge_attempt_count": int(bundle.memory_stats.get("claim_transition_judge_attempt_count", 0) or 0),
                    "claim_transition_judge_success_count": int(bundle.memory_stats.get("claim_transition_judge_success_count", 0) or 0),
                    "claim_transition_judge_fallback_count": int(bundle.memory_stats.get("claim_transition_judge_fallback_count", 0) or 0),
                    "claim_transition_revise_count": int(bundle.memory_stats.get("claim_transition_revise_count", 0) or 0),
                    "claim_transition_add_count": int(bundle.memory_stats.get("claim_transition_add_count", 0) or 0),
                    "episodic_batch_call_count": int(bundle.memory_stats.get("episodic_batch_call_count", 0) or 0),
                    "episodic_batch_item_count": int(bundle.memory_stats.get("episodic_batch_item_count", 0) or 0),
                    "episodic_batch_repair_after_batch_count": int(
                        bundle.memory_stats.get("episodic_batch_repair_after_batch_count", 0) or 0
                    ),
                    "episodic_batch_oom_backoff_count": int(
                        bundle.memory_stats.get("episodic_batch_oom_backoff_count", 0) or 0
                    ),
                    "episodic_batch_effective_size_max": int(
                        bundle.memory_stats.get("episodic_batch_effective_size_max", 1) or 1
                    ),
                    "episodic_batch_effective_size_final": int(
                        bundle.memory_stats.get("episodic_batch_effective_size_final", 1) or 1
                    ),
                    "memory_stage_batch_stats": dict(bundle.memory_stats.get("memory_stage_batch_stats", {})),
                    "empty_repair_target_count": int(bundle.memory_stats.get("empty_repair_target_count", 0) or 0),
                    "assistant_only_exchange_count": int(
                        bundle.memory_stats.get("assistant_only_exchange_count", assistant_only_exchange_count) or 0
                    ),
                    "forced_memory_seed_count": int(bundle.memory_stats.get("forced_memory_seed_count", 0) or 0),
                    "low_salience_memory_count": int(bundle.memory_stats.get("low_salience_memory_count", 0) or 0),
                    "llm_no_memory_forced_count": int(bundle.memory_stats.get("llm_no_memory_forced_count", 0) or 0),
                    "zero_claim_episodic_candidate_count": int(
                        bundle.memory_stats.get("zero_claim_episodic_candidate_count", 0) or 0
                    ),
                    "zero_claim_episodic_persisted_count": int(
                        bundle.memory_stats.get("zero_claim_episodic_persisted_count", 0) or 0
                    ),
                    "zero_claim_low_salience_skipped_count": int(
                        bundle.memory_stats.get("zero_claim_low_salience_skipped_count", 0) or 0
                    ),
                    "structured_attempt_count": int(bundle.memory_stats.get("structured_attempts", 0) or 0),
                    "structured_success_count": int(bundle.memory_stats.get("structured_successes", 0) or 0),
                    "structured_fallback_count": int(bundle.memory_stats.get("structured_fallbacks", 0) or 0),
                    "structured_attempts_by_task": dict(bundle.memory_stats.get("structured_attempts_by_task", {})),
                    "structured_successes_by_task": dict(bundle.memory_stats.get("structured_successes_by_task", {})),
                    "structured_fallbacks_by_task": dict(bundle.memory_stats.get("structured_fallbacks_by_task", {})),
                    "structured_attempts_by_vendor": dict(bundle.memory_stats.get("structured_attempts_by_vendor", {})),
                    "structured_successes_by_vendor": dict(bundle.memory_stats.get("structured_successes_by_vendor", {})),
                    "structured_fallbacks_by_vendor": dict(bundle.memory_stats.get("structured_fallbacks_by_vendor", {})),
                    "memory_debug_artifact_paths": [],
                }
            self.cache_stats["cache_misses"] += 1.0
        elif self.config.memory_cache_enabled:
            self.cache_stats["cache_misses"] += 1.0

        valid_link_ids = set(self.store.list_raw_message_ids(sample.sample_id))
        parse_attempts_before = self.orchestrator.parse_attempts
        parse_failures_before = self.orchestrator.parse_failures
        repair_rounds_before = self.orchestrator.repair_rounds
        fallback_before = self.orchestrator.extraction_fallbacks
        closed_on_fallback_before = self.orchestrator.closed_on_fallback
        trajectory_match_total_open_before = self.orchestrator.trajectory_match_total_open
        trajectory_match_prefiltered_before = self.orchestrator.trajectory_match_prefiltered
        trajectory_match_shortlisted_before = self.orchestrator.trajectory_match_shortlisted
        link_salvage_before = self.orchestrator.link_salvage_count
        link_exchange_fallback_before = self.orchestrator.link_exchange_fallback_count
        ops_parse_failures_before = self.orchestrator.ops_parse_failure_count
        ops_ignored_before = self.orchestrator.ops_ignored_count
        ops_synthesized_before = self.orchestrator.ops_synthesized_count
        ops_model_hint_before = self.orchestrator.ops_model_hint_count
        ops_model_supplied_before = self.orchestrator.ops_model_supplied_count
        claims_parse_failures_before = self.orchestrator.claims_parse_failure_count
        claims_required_repair_before = self.orchestrator.claims_required_repair_count
        claim_text_matches_before = self.orchestrator.claim_text_exact_match_count
        claim_status_updates_before = self.orchestrator.claim_status_updated_count
        claim_new_add_before = self.orchestrator.claim_new_add_count
        claim_unmatched_previous_before = self.orchestrator.claim_unmatched_previous_count
        claim_transition_attempts_before = self.orchestrator.claim_transition_judge_attempt_count
        claim_transition_successes_before = self.orchestrator.claim_transition_judge_success_count
        claim_transition_fallbacks_before = self.orchestrator.claim_transition_judge_fallback_count
        claim_transition_revises_before = self.orchestrator.claim_transition_revise_count
        claim_transition_adds_before = self.orchestrator.claim_transition_add_count
        empty_repair_target_before = self.orchestrator.empty_repair_target_count
        structured_attempts_before = self.orchestrator.structured_attempts
        structured_successes_before = self.orchestrator.structured_successes
        structured_fallbacks_before = self.orchestrator.structured_fallbacks
        structured_attempts_by_task_before = dict(self.orchestrator.structured_attempts_by_task)
        structured_successes_by_task_before = dict(self.orchestrator.structured_successes_by_task)
        structured_fallbacks_by_task_before = dict(self.orchestrator.structured_fallbacks_by_task)
        structured_attempts_by_vendor_before = dict(self.orchestrator.structured_attempts_by_vendor)
        structured_successes_by_vendor_before = dict(self.orchestrator.structured_successes_by_vendor)
        structured_fallbacks_by_vendor_before = dict(self.orchestrator.structured_fallbacks_by_vendor)
        debug_paths_before = len(self.orchestrator.debug_artifact_paths.get(sample.sample_id, []))
        episodic_batch_call_count_before = self.episodic_batch_call_count
        episodic_batch_item_count_before = self.episodic_batch_item_count
        episodic_batch_repair_after_batch_before = self.episodic_batch_repair_after_batch_count
        episodic_batch_oom_backoff_before = self.episodic_batch_oom_backoff_count
        memory_stage_batch_stats_before = self._copy_memory_stage_batch_stats(self.memory_stage_batch_stats)
        forced_memory_seed_before = self.orchestrator.forced_memory_seed_count
        low_salience_memory_before = self.orchestrator.low_salience_memory_count
        llm_no_memory_forced_before = self.orchestrator.llm_no_memory_forced_count
        zero_claim_candidate_before = self.orchestrator.zero_claim_episodic_candidate_count
        zero_claim_persisted_before = self.orchestrator.zero_claim_episodic_persisted_count
        zero_claim_low_salience_skipped_before = self.orchestrator.zero_claim_low_salience_skipped_count
        sample_effective_batch_size_max = self._resolve_memory_extract_batch_size()
        build_started_at = time.perf_counter()
        self._trace(
            f"sample={sample.sample_id} memory_replay start messages={len(messages)} "
            f"exchanges={exchange_count} assistant_only={assistant_only_exchange_count}"
        )
        self._run_batched_episodic_replay(
            sample,
            exchanges,
            valid_link_ids=valid_link_ids,
        )
        self.orchestrator.finalize_trajectory(sample.sample_id)
        self._commit_session(stage="memory_finalize", sample_id=sample.sample_id)
        wiki_started_at = time.perf_counter()
        assert self.wiki is not None
        self.wiki.compile_sample(sample.sample_id, sample.dataset_name)
        self._commit_session(stage="wiki_compile", sample_id=sample.sample_id)
        wiki_latency_ms = (time.perf_counter() - wiki_started_at) * 1000.0
        build_latency_ms = (time.perf_counter() - build_started_at) * 1000.0
        self._trace(
            f"sample={sample.sample_id} memory_replay done exchanges={exchange_count} "
            f"latency_ms={build_latency_ms:.1f} wiki_latency_ms={wiki_latency_ms:.1f}"
        )
        if self.config.memory_cache_enabled:
            bundle = self.store.export_sample_memory_bundle(sample.sample_id)
            bundle.sample_meta.update(
                {
                    "sample_id": sample.sample_id,
                    "dataset_name": sample.dataset_name,
                    "history_fingerprint": sample_fingerprint,
                    "build_fingerprint": self.cache.build_fingerprint,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
            bundle.memory_stats.update(
                {
                    "build_latency_ms": build_latency_ms,
                    "parse_attempts": self.orchestrator.parse_attempts - parse_attempts_before,
                    "parse_failures": self.orchestrator.parse_failures - parse_failures_before,
                    "repair_rounds": self.orchestrator.repair_rounds - repair_rounds_before,
                    "fallback_count": self.orchestrator.extraction_fallbacks - fallback_before,
                    "closed_on_fallback_count": self.orchestrator.closed_on_fallback - closed_on_fallback_before,
                    "trajectory_match_total_open": self.orchestrator.trajectory_match_total_open - trajectory_match_total_open_before,
                    "trajectory_match_prefiltered_count": self.orchestrator.trajectory_match_prefiltered - trajectory_match_prefiltered_before,
                    "trajectory_match_shortlist_count": self.orchestrator.trajectory_match_shortlisted - trajectory_match_shortlisted_before,
                    "wiki_compile_latency_ms": wiki_latency_ms,
                    "link_salvage_count": self.orchestrator.link_salvage_count - link_salvage_before,
                    "link_exchange_fallback_count": self.orchestrator.link_exchange_fallback_count - link_exchange_fallback_before,
                    "ops_parse_failure_count": self.orchestrator.ops_parse_failure_count - ops_parse_failures_before,
                    "ops_ignored_count": self.orchestrator.ops_ignored_count - ops_ignored_before,
                    "ops_synthesized_count": self.orchestrator.ops_synthesized_count - ops_synthesized_before,
                    "ops_model_hint_count": self.orchestrator.ops_model_hint_count - ops_model_hint_before,
                    "ops_model_supplied_count": self.orchestrator.ops_model_supplied_count - ops_model_supplied_before,
                    "claims_parse_failure_count": self.orchestrator.claims_parse_failure_count - claims_parse_failures_before,
                    "claims_required_repair_count": self.orchestrator.claims_required_repair_count - claims_required_repair_before,
                    "claim_text_exact_match_count": self.orchestrator.claim_text_exact_match_count - claim_text_matches_before,
                    "claim_status_updated_count": self.orchestrator.claim_status_updated_count - claim_status_updates_before,
                    "claim_new_add_count": self.orchestrator.claim_new_add_count - claim_new_add_before,
                    "claim_unmatched_previous_count": self.orchestrator.claim_unmatched_previous_count - claim_unmatched_previous_before,
                    "claim_transition_judge_attempt_count": self.orchestrator.claim_transition_judge_attempt_count - claim_transition_attempts_before,
                    "claim_transition_judge_success_count": self.orchestrator.claim_transition_judge_success_count - claim_transition_successes_before,
                    "claim_transition_judge_fallback_count": self.orchestrator.claim_transition_judge_fallback_count - claim_transition_fallbacks_before,
                    "claim_transition_revise_count": self.orchestrator.claim_transition_revise_count - claim_transition_revises_before,
                    "claim_transition_add_count": self.orchestrator.claim_transition_add_count - claim_transition_adds_before,
                    "episodic_batch_call_count": self.episodic_batch_call_count - episodic_batch_call_count_before,
                    "episodic_batch_item_count": self.episodic_batch_item_count - episodic_batch_item_count_before,
                    "episodic_batch_repair_after_batch_count": (
                        self.episodic_batch_repair_after_batch_count - episodic_batch_repair_after_batch_before
                    ),
                    "episodic_batch_oom_backoff_count": (
                        self.episodic_batch_oom_backoff_count - episodic_batch_oom_backoff_before
                    ),
                    "episodic_batch_effective_size_max": sample_effective_batch_size_max,
                    "episodic_batch_effective_size_final": self.episodic_batch_effective_size_final,
                    "memory_stage_batch_stats": self._memory_stage_batch_stats_diff(
                        self.memory_stage_batch_stats,
                        memory_stage_batch_stats_before,
                    ),
                    "empty_repair_target_count": self.orchestrator.empty_repair_target_count - empty_repair_target_before,
                    "assistant_only_exchange_count": assistant_only_exchange_count,
                    "forced_memory_seed_count": self.orchestrator.forced_memory_seed_count - forced_memory_seed_before,
                    "low_salience_memory_count": self.orchestrator.low_salience_memory_count - low_salience_memory_before,
                    "llm_no_memory_forced_count": self.orchestrator.llm_no_memory_forced_count - llm_no_memory_forced_before,
                    "zero_claim_episodic_candidate_count": (
                        self.orchestrator.zero_claim_episodic_candidate_count - zero_claim_candidate_before
                    ),
                    "zero_claim_episodic_persisted_count": (
                        self.orchestrator.zero_claim_episodic_persisted_count - zero_claim_persisted_before
                    ),
                    "zero_claim_low_salience_skipped_count": (
                        self.orchestrator.zero_claim_low_salience_skipped_count
                        - zero_claim_low_salience_skipped_before
                    ),
                    "structured_attempts": self.orchestrator.structured_attempts - structured_attempts_before,
                    "structured_successes": self.orchestrator.structured_successes - structured_successes_before,
                    "structured_fallbacks": self.orchestrator.structured_fallbacks - structured_fallbacks_before,
                    "structured_attempts_by_task": {
                        key: value - structured_attempts_by_task_before.get(key, 0)
                        for key, value in self.orchestrator.structured_attempts_by_task.items()
                        if value - structured_attempts_by_task_before.get(key, 0)
                    },
                    "structured_successes_by_task": {
                        key: value - structured_successes_by_task_before.get(key, 0)
                        for key, value in self.orchestrator.structured_successes_by_task.items()
                        if value - structured_successes_by_task_before.get(key, 0)
                    },
                    "structured_fallbacks_by_task": {
                        key: value - structured_fallbacks_by_task_before.get(key, 0)
                        for key, value in self.orchestrator.structured_fallbacks_by_task.items()
                        if value - structured_fallbacks_by_task_before.get(key, 0)
                    },
                    "structured_attempts_by_vendor": {
                        key: value - structured_attempts_by_vendor_before.get(key, 0)
                        for key, value in self.orchestrator.structured_attempts_by_vendor.items()
                        if value - structured_attempts_by_vendor_before.get(key, 0)
                    },
                    "structured_successes_by_vendor": {
                        key: value - structured_successes_by_vendor_before.get(key, 0)
                        for key, value in self.orchestrator.structured_successes_by_vendor.items()
                        if value - structured_successes_by_vendor_before.get(key, 0)
                    },
                    "structured_fallbacks_by_vendor": {
                        key: value - structured_fallbacks_by_vendor_before.get(key, 0)
                        for key, value in self.orchestrator.structured_fallbacks_by_vendor.items()
                        if value - structured_fallbacks_by_vendor_before.get(key, 0)
                    },
                }
            )
            _, cache_written = self.cache.save_sample_cache(sample, bundle, sample_fingerprint)
            if cache_written:
                self.cache_stats["cache_writes"] += 1.0
        return {
            "cache_hit": False,
            "memory_build_latency_ms": build_latency_ms,
            "cache_load_latency_ms": 0.0,
            "saved_build_latency_ms": 0.0,
            "extraction_repair_count": self.orchestrator.repair_rounds - repair_rounds_before,
            "extraction_fallback_count": self.orchestrator.extraction_fallbacks - fallback_before,
            "closed_on_fallback_count": self.orchestrator.closed_on_fallback - closed_on_fallback_before,
            "trajectory_match_total_open": self.orchestrator.trajectory_match_total_open - trajectory_match_total_open_before,
            "trajectory_match_prefiltered_count": self.orchestrator.trajectory_match_prefiltered - trajectory_match_prefiltered_before,
            "trajectory_match_shortlist_count": self.orchestrator.trajectory_match_shortlisted - trajectory_match_shortlisted_before,
            "wiki_compile_latency_ms": wiki_latency_ms,
            "link_salvage_count": self.orchestrator.link_salvage_count - link_salvage_before,
            "link_exchange_fallback_count": self.orchestrator.link_exchange_fallback_count - link_exchange_fallback_before,
            "ops_parse_failure_count": self.orchestrator.ops_parse_failure_count - ops_parse_failures_before,
            "ops_ignored_count": self.orchestrator.ops_ignored_count - ops_ignored_before,
            "ops_synthesized_count": self.orchestrator.ops_synthesized_count - ops_synthesized_before,
            "ops_model_hint_count": self.orchestrator.ops_model_hint_count - ops_model_hint_before,
            "ops_model_supplied_count": self.orchestrator.ops_model_supplied_count - ops_model_supplied_before,
            "claims_parse_failure_count": self.orchestrator.claims_parse_failure_count - claims_parse_failures_before,
            "claims_required_repair_count": self.orchestrator.claims_required_repair_count - claims_required_repair_before,
            "claim_text_exact_match_count": self.orchestrator.claim_text_exact_match_count - claim_text_matches_before,
            "claim_status_updated_count": self.orchestrator.claim_status_updated_count - claim_status_updates_before,
            "claim_new_add_count": self.orchestrator.claim_new_add_count - claim_new_add_before,
            "claim_unmatched_previous_count": self.orchestrator.claim_unmatched_previous_count - claim_unmatched_previous_before,
            "claim_transition_judge_attempt_count": self.orchestrator.claim_transition_judge_attempt_count - claim_transition_attempts_before,
            "claim_transition_judge_success_count": self.orchestrator.claim_transition_judge_success_count - claim_transition_successes_before,
            "claim_transition_judge_fallback_count": self.orchestrator.claim_transition_judge_fallback_count - claim_transition_fallbacks_before,
            "claim_transition_revise_count": self.orchestrator.claim_transition_revise_count - claim_transition_revises_before,
            "claim_transition_add_count": self.orchestrator.claim_transition_add_count - claim_transition_adds_before,
            "episodic_batch_call_count": self.episodic_batch_call_count - episodic_batch_call_count_before,
            "episodic_batch_item_count": self.episodic_batch_item_count - episodic_batch_item_count_before,
            "episodic_batch_repair_after_batch_count": (
                self.episodic_batch_repair_after_batch_count - episodic_batch_repair_after_batch_before
            ),
            "episodic_batch_oom_backoff_count": (
                self.episodic_batch_oom_backoff_count - episodic_batch_oom_backoff_before
            ),
            "episodic_batch_effective_size_max": sample_effective_batch_size_max,
            "episodic_batch_effective_size_final": self.episodic_batch_effective_size_final,
            "memory_stage_batch_stats": self._memory_stage_batch_stats_diff(
                self.memory_stage_batch_stats,
                memory_stage_batch_stats_before,
            ),
            "empty_repair_target_count": self.orchestrator.empty_repair_target_count - empty_repair_target_before,
            "assistant_only_exchange_count": assistant_only_exchange_count,
            "forced_memory_seed_count": self.orchestrator.forced_memory_seed_count - forced_memory_seed_before,
            "low_salience_memory_count": self.orchestrator.low_salience_memory_count - low_salience_memory_before,
            "llm_no_memory_forced_count": self.orchestrator.llm_no_memory_forced_count - llm_no_memory_forced_before,
            "zero_claim_episodic_candidate_count": (
                self.orchestrator.zero_claim_episodic_candidate_count - zero_claim_candidate_before
            ),
            "zero_claim_episodic_persisted_count": (
                self.orchestrator.zero_claim_episodic_persisted_count - zero_claim_persisted_before
            ),
            "zero_claim_low_salience_skipped_count": (
                self.orchestrator.zero_claim_low_salience_skipped_count
                - zero_claim_low_salience_skipped_before
            ),
            "structured_attempt_count": self.orchestrator.structured_attempts - structured_attempts_before,
            "structured_success_count": self.orchestrator.structured_successes - structured_successes_before,
            "structured_fallback_count": self.orchestrator.structured_fallbacks - structured_fallbacks_before,
            "structured_attempts_by_task": {
                key: value - structured_attempts_by_task_before.get(key, 0)
                for key, value in self.orchestrator.structured_attempts_by_task.items()
                if value - structured_attempts_by_task_before.get(key, 0)
            },
            "structured_successes_by_task": {
                key: value - structured_successes_by_task_before.get(key, 0)
                for key, value in self.orchestrator.structured_successes_by_task.items()
                if value - structured_successes_by_task_before.get(key, 0)
            },
            "structured_fallbacks_by_task": {
                key: value - structured_fallbacks_by_task_before.get(key, 0)
                for key, value in self.orchestrator.structured_fallbacks_by_task.items()
                if value - structured_fallbacks_by_task_before.get(key, 0)
            },
            "structured_attempts_by_vendor": {
                key: value - structured_attempts_by_vendor_before.get(key, 0)
                for key, value in self.orchestrator.structured_attempts_by_vendor.items()
                if value - structured_attempts_by_vendor_before.get(key, 0)
            },
            "structured_successes_by_vendor": {
                key: value - structured_successes_by_vendor_before.get(key, 0)
                for key, value in self.orchestrator.structured_successes_by_vendor.items()
                if value - structured_successes_by_vendor_before.get(key, 0)
            },
            "structured_fallbacks_by_vendor": {
                key: value - structured_fallbacks_by_vendor_before.get(key, 0)
                for key, value in self.orchestrator.structured_fallbacks_by_vendor.items()
                if value - structured_fallbacks_by_vendor_before.get(key, 0)
            },
            "memory_debug_artifact_paths": list(
                self.orchestrator.debug_artifact_paths.get(sample.sample_id, [])
            )[debug_paths_before:],
        }

    def _evaluate_answers(self, answer_results: list[AnswerResult]) -> list[dict[str, Any]]:
        assert self.store is not None
        assert self.session is not None
        rows: list[dict[str, Any]] = []
        if not answer_results:
            return rows

        self.console.print("[bold]Phase 2[/bold] Computing metrics and running benchmark judge")
        judge_results: dict[int, tuple[JudgeResult, float]] = {}
        semantic_results: dict[int, SemanticMetricResult | None] = {}
        semantic_indexes = [
            index
            for index, answer_result in enumerate(answer_results)
            if answer_result.dataset_name == "locomo"
        ]
        text_only_audit_count = len(semantic_indexes) if self.config.dataset == "locomo" else 0
        total_evaluation_steps = len(answer_results) + len(semantic_indexes) + text_only_audit_count
        judge_started = time.perf_counter()
        self._trace(
            f"judge_batch start answers={len(answer_results)} parallelism={self.judge_parallelism}"
        )
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task("Judging answers", total=total_evaluation_steps)
            with ThreadPoolExecutor(max_workers=self.judge_parallelism) as executor:
                future_map = {}
                for index, answer_result in enumerate(answer_results):
                    submitted_at = time.perf_counter()
                    future = executor.submit(self._judge_single_answer, answer_result, submitted_at)
                    future_map[future] = (index, answer_result)
                for future in as_completed(future_map):
                    index, answer_result = future_map[future]
                    progress.update(task_id, description=f"Judging [{answer_result.sample_id}]")
                    try:
                        judge_results[index] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        judge_results[index] = (
                            JudgeResult(
                                verdict="judge_error",
                                rationale=f"Judge execution failed: {exc}",
                                prompt=self.judge.build_prompt(
                                    answer_result.dataset_name,
                                    QueryTask(
                                        query_task_id=answer_result.query_task_id,
                                        sample_id=answer_result.sample_id,
                                        question=answer_result.question,
                                        gold_answer=answer_result.gold_answer,
                                        metadata=dict(answer_result.metadata.get("query_metadata", {})),
                                    ),
                                    answer_result.answer_text,
                                ),
                                metadata={
                                    "judge_mode": "judge_error",
                                    "structured_requested": False,
                                    "structured_success": False,
                                    "structured_fallback_used": False,
                                    "structured_fallback_reason": None,
                                    "structured_fallback_category": "judge_execution_error",
                                    "structured_refusal": None,
                                    "judge_execution_failed": True,
                                    "judge_error_type": exc.__class__.__name__,
                                    "judge_error_message": " ".join(str(exc).split()) or exc.__class__.__name__,
                                },
                            ),
                            0.0,
                        )
                    progress.advance(task_id)
            self.judge_total_wall_time_ms = (time.perf_counter() - judge_started) * 1000.0
            self._trace(
                f"judge_batch done answers={len(answer_results)} wall_time_ms={self.judge_total_wall_time_ms:.1f}"
            )
            if semantic_indexes:
                semantic_started = time.perf_counter()
                semantic_done = 0
                semantic_cache_hits = 0
                semantic_cache_misses = 0
                semantic_llm_calls = 0
                progress_interval = max(1, min(25, max(len(semantic_indexes) // 4, 1)))
                self._trace(
                    f"semantic_metrics_batch start answers={len(semantic_indexes)} "
                    f"parallelism={self.judge_parallelism} "
                    f"cache_rebuild={str(self.config.rebuild_semantic_metric_cache).lower()}"
                )
                with ThreadPoolExecutor(max_workers=self.judge_parallelism) as executor:
                    future_map = {
                        executor.submit(self._compute_semantic_metrics, answer_results[index]): (
                            index,
                            answer_results[index],
                        )
                        for index in semantic_indexes
                    }
                    for future in as_completed(future_map):
                        index, answer_result = future_map[future]
                        progress.update(
                            task_id,
                            description=f"Scoring semantic metrics [{answer_result.sample_id}]",
                        )
                        try:
                            semantic_result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            semantic_result = self._semantic_metrics_error_result(answer_result, exc)
                        semantic_results[index] = semantic_result
                        semantic_done += 1
                        metadata = dict(semantic_result.metadata or {})
                        semantic_cache_hits += int(metadata.get("cache_hit_count") or 0)
                        semantic_cache_misses += int(metadata.get("cache_miss_count") or 0)
                        semantic_llm_calls += int(metadata.get("llm_call_count") or 0)
                        if semantic_done == len(semantic_indexes) or semantic_done % progress_interval == 0:
                            self._trace(
                                f"semantic_metrics_progress done={semantic_done}/{len(semantic_indexes)} "
                                f"cache_hits={semantic_cache_hits} cache_misses={semantic_cache_misses} "
                                f"llm_calls={semantic_llm_calls}"
                            )
                        progress.advance(task_id)
                semantic_wall_time_ms = (time.perf_counter() - semantic_started) * 1000.0
                self._trace(
                    f"semantic_metrics_batch done answers={len(semantic_indexes)} "
                    f"wall_time_ms={semantic_wall_time_ms:.1f} cache_hits={semantic_cache_hits} "
                    f"cache_misses={semantic_cache_misses} llm_calls={semantic_llm_calls}"
                )
            for index, answer_result in enumerate(answer_results):
                judge_result, judge_queue_latency_ms = judge_results[index]
                rows.append(
                    self._evaluate_single_answer(
                        answer_result,
                        judge_result,
                        judge_queue_latency_ms,
                        semantic_result=semantic_results.get(index),
                    )
                )
            self._apply_text_only_filter_audit(rows, progress=progress, task_id=task_id)
        self._commit_session(stage="evaluation_persist")
        self._print_tables(rows)
        return rows

    def _apply_text_only_filter_audit(
        self,
        rows: list[dict[str, Any]],
        *,
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> None:
        if self.config.dataset != "locomo":
            return
        assert self.session is not None
        locomo_rows = [row for row in rows if str(row.get("dataset_name") or "") == "locomo"]
        if not locomo_rows:
            self.text_only_filter_manifest = []
            self.text_only_filter_summary = summarize_text_only_filter([])
            return

        refs_by_sample: dict[str, set[str]] = defaultdict(set)
        for row in locomo_rows:
            metadata = dict(row.get("metadata") or {})
            query_metadata = dict(metadata.get("query_metadata") or {})
            for ref in list(query_metadata.get("gold_evidence_refs") or query_metadata.get("gold_evidence_raw") or []):
                if str(ref).strip():
                    refs_by_sample[str(row.get("sample_id"))].add(str(ref))

        sample_ids = sorted(refs_by_sample)
        all_refs = sorted({ref for refs in refs_by_sample.values() for ref in refs})
        evidence_text_by_sample_ref: dict[tuple[str, str], str] = {}
        if sample_ids and all_refs:
            message_rows = self.session.execute(
                select(RawMessageRecord.sample_id, RawMessageRecord.source_ref, RawMessageRecord.content).where(
                    RawMessageRecord.sample_id.in_(sample_ids),
                    RawMessageRecord.source_ref.in_(all_refs),
                )
            ).all()
            for sample_id, source_ref, content in message_rows:
                if source_ref is None:
                    continue
                evidence_text_by_sample_ref[(str(sample_id), str(source_ref))] = str(content or "")

        started_at = time.perf_counter()
        interval = max(1, min(25, max(len(locomo_rows) // 4, 1)))
        included_count = 0
        excluded_count = 0
        ambiguous_count = 0
        manifest: list[dict[str, Any]] = []
        self._trace(
            f"text_only_filter_start rows={len(locomo_rows)} policy=locomo_text_only_filter_v1"
        )
        for index, row in enumerate(locomo_rows, start=1):
            sample_id = str(row.get("sample_id"))
            metadata = dict(row.get("metadata") or {})
            query_metadata = dict(metadata.get("query_metadata") or {})
            gold_refs = [
                str(ref)
                for ref in list(query_metadata.get("gold_evidence_refs") or query_metadata.get("gold_evidence_raw") or [])
                if str(ref).strip()
            ]
            evidence_text_by_ref = {
                ref: evidence_text_by_sample_ref.get((sample_id, ref), "")
                for ref in gold_refs
            }
            result = audit_text_only_visibility(row, evidence_text_by_ref)
            compact = compact_filter_for_details(result)
            metadata["evaluation_filter"] = compact
            row["metadata"] = metadata
            manifest.append(manifest_entry_for_row(row, result))

            if bool(compact.get("excluded_from_text_only")):
                excluded_count += 1
            elif compact.get("visual_dependency_type") == "ambiguous_needs_review" or compact.get("exclusion_reason") == "ambiguous_needs_review":
                included_count += 1
                ambiguous_count += 1
            else:
                included_count += 1

            if progress is not None and task_id is not None:
                progress.update(task_id, description=f"Auditing text-only visibility [{sample_id}]")
                progress.advance(task_id)
            if index == len(locomo_rows) or index % interval == 0:
                self._trace(
                    f"text_only_filter_progress done={index}/{len(locomo_rows)} "
                    f"included={included_count} excluded={excluded_count} ambiguous={ambiguous_count}"
                )

        self.text_only_filter_manifest = manifest
        self.text_only_filter_summary = summarize_text_only_filter(locomo_rows)
        wall_time_ms = (time.perf_counter() - started_at) * 1000.0
        self._trace(
            f"text_only_filter_done rows={len(locomo_rows)} included={self.text_only_filter_summary.get('included_count')} "
            f"excluded={self.text_only_filter_summary.get('excluded_count')} "
            f"ambiguous={self.text_only_filter_summary.get('ambiguous_count')} wall_time_ms={wall_time_ms:.1f}"
        )

    def _judge_single_answer(self, answer_result: AnswerResult, submitted_at: float) -> tuple[JudgeResult, float]:
        self._set_meter_call_context(
            sample_id=answer_result.sample_id,
            query_task_id=answer_result.query_task_id,
        )
        query_metadata = dict(answer_result.metadata.get("query_metadata", {}))
        if answer_result.rubric and not query_metadata.get("test_point"):
            query_metadata["test_point"] = answer_result.rubric
        query_task = QueryTask(
            query_task_id=answer_result.query_task_id,
            sample_id=answer_result.sample_id,
            question=answer_result.question,
            gold_answer=answer_result.gold_answer,
            metadata=query_metadata,
        )
        started_at = time.perf_counter()
        queue_latency_ms = (started_at - submitted_at) * 1000.0
        self._trace(
            f"judge_start sample={answer_result.sample_id} query_task_id={answer_result.query_task_id} "
            f"queue_latency_ms={queue_latency_ms:.1f}"
        )
        judge_result = self.judge.judge(answer_result.dataset_name, query_task, answer_result.answer_text)
        if judge_result is None:
            raise ProviderConfigurationError("Judge provider returned no result.")
        self._trace(
            f"judge_done sample={answer_result.sample_id} query_task_id={answer_result.query_task_id} "
            f"latency_ms={judge_result.latency_ms:.1f} verdict={judge_result.verdict}"
        )
        return judge_result, queue_latency_ms

    def _compute_semantic_metrics(self, answer_result: AnswerResult) -> SemanticMetricResult:
        self._set_meter_call_context(
            sample_id=answer_result.sample_id,
            query_task_id=answer_result.query_task_id,
        )
        return self.semantic_metrics.evaluate_locomo(
            question=answer_result.question,
            reference_answer=answer_result.gold_answer or "",
            candidate_answer=answer_result.answer_text,
            query_task_id=answer_result.query_task_id,
        )

    def _semantic_metrics_error_result(
        self,
        answer_result: AnswerResult,
        exc: BaseException,
    ) -> SemanticMetricResult:
        error = " ".join(str(exc).split()) or exc.__class__.__name__
        schema = self.semantic_metrics._deterministic_schema(
            answer_result.question,
            answer_result.gold_answer or "",
            error=error,
        )
        reference_slots = self.semantic_metrics._deterministic_extract(
            schema,
            answer_result.gold_answer or "",
            error=error,
        )
        candidate_slots = self.semantic_metrics._deterministic_extract(
            schema,
            answer_result.answer_text,
            error=error,
        )
        f1, f1_metadata = SemanticMetricEvaluator._canonical_set_f1(candidate_slots, reference_slots)
        reference_text = SemanticMetricEvaluator._canonical_text(schema, reference_slots)
        candidate_text = SemanticMetricEvaluator._canonical_text(schema, candidate_slots)
        return SemanticMetricResult(
            f1=f1,
            bleu_1=bleu1(candidate_text, reference_text),
            prompt_tokens=0,
            completion_tokens=0,
            metadata={
                "schema_version": "v1",
                "f1_policy": "semantic_canonical_set_f1_v2",
                "bleu_policy": "semantic_canonical_bleu1_no_bp_v1",
                "schema": schema,
                "reference_slots": reference_slots,
                "candidate_slots": candidate_slots,
                "cache_hits": {
                    "schema": False,
                    "reference_extract": False,
                    "candidate_extract": False,
                },
                "cache_misses": {
                    "schema": False,
                    "reference_extract": False,
                    "candidate_extract": False,
                },
                "cache_hit_count": 0,
                "cache_miss_count": 0,
                "llm_call_count": 0,
                "llm_call_counts": {
                    "schema": 0,
                    "reference_extract": 0,
                    "candidate_extract": 0,
                },
                "mode": "deterministic_fallback",
                "error": f"semantic_metrics_worker_error: {error}",
                "reference_canonical_text": reference_text,
                "candidate_canonical_text": candidate_text,
                **f1_metadata,
            },
        )

    def _evaluate_single_answer(
        self,
        answer_result: AnswerResult,
        judge_result: JudgeResult,
        judge_queue_latency_ms: float,
        *,
        semantic_result: SemanticMetricResult | None = None,
    ) -> dict[str, Any]:
        assert self.store is not None
        self.judge_queue_latency_ms_total += judge_queue_latency_ms
        self.judge_queue_events += 1
        query_metadata = dict(answer_result.metadata.get("query_metadata", {}))
        if answer_result.rubric and not query_metadata.get("test_point"):
            query_metadata["test_point"] = answer_result.rubric

        judge_verdict = str(judge_result.verdict).lower()
        judge_acc: float | None
        if judge_verdict in {"correct", "partial", "incorrect"}:
            judge_acc = (
                float(judge_result.score)
                if judge_result.score is not None
                else {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[judge_verdict]
            )
        else:
            judge_acc = None

        if answer_result.dataset_name == "locomo":
            if semantic_result is None:
                semantic_result = self._compute_semantic_metrics(answer_result)
            lexical_metrics = {
                "F1": semantic_result.f1,
                "BLEU-1": semantic_result.bleu_1,
            }
            judge_result, judge_acc = self._apply_locomo_judge_semantic_override(
                answer_result,
                judge_result,
                judge_acc,
                semantic_result,
            )
        else:
            lexical_metrics = {}
        metrics = {
            **lexical_metrics,
            "judge_acc": judge_acc,
        }

        usage = answer_result.metadata.get("backbone_usage", {})
        token_payload = {
            "backbone_prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "backbone_completion_tokens": int(usage.get("completion_tokens", 0)),
            "judge_prompt_tokens": int(judge_result.prompt_tokens or 0),
            "judge_completion_tokens": int(judge_result.completion_tokens or 0),
            "semantic_prompt_tokens": int(semantic_result.prompt_tokens if semantic_result is not None else 0),
            "semantic_completion_tokens": int(
                semantic_result.completion_tokens if semantic_result is not None else 0
            ),
        }
        token_payload["prompt_tokens"] = (
            token_payload["backbone_prompt_tokens"]
            + token_payload["judge_prompt_tokens"]
            + token_payload["semantic_prompt_tokens"]
        )
        token_payload["completion_tokens"] = (
            token_payload["backbone_completion_tokens"]
            + token_payload["judge_completion_tokens"]
            + token_payload["semantic_completion_tokens"]
        )
        token_payload["total_tokens"] = token_payload["prompt_tokens"] + token_payload["completion_tokens"]
        llm_usage, llm_usage_by_task, llm_fallback_counts, llm_repair_counts = self._build_query_llm_usage(
            usage,
            judge_result,
            semantic_result,
        )

        latency_payload = {
            "sample_runtime_ms": float(answer_result.metadata.get("sample_runtime_ms", 0.0)),
            "memory_build_latency_ms": float(answer_result.metadata.get("memory_build_latency_ms", 0.0)),
            "cache_load_latency_ms": float(answer_result.metadata.get("cache_load_latency_ms", 0.0)),
            "retrieval_latency_ms": float(answer_result.metadata.get("retrieval_latency_ms", 0.0)),
            "answer_latency_ms": float(answer_result.metadata.get("answer_latency_ms", 0.0)),
            "judge_latency_ms": float(judge_result.latency_ms),
            "judge_queue_latency_ms": judge_queue_latency_ms,
        }
        latency_payload["runtime_ms"] = latency_payload["sample_runtime_ms"] + latency_payload["judge_latency_ms"]

        metric_rows = self._build_evaluation_rows(answer_result, metrics, judge_result)
        self.store.record_evaluations(metric_rows)

        return {
            "sample_id": answer_result.sample_id,
            "dataset_name": answer_result.dataset_name,
            "subset_key": answer_result.subset_key,
            "scene_tag": answer_result.scene_tag,
            "query_task_id": answer_result.query_task_id,
            "question": answer_result.question,
            "gold_answer": answer_result.gold_answer,
            "rubric": answer_result.rubric,
            "answer_prompt": answer_result.answer_prompt,
            "answer_text": answer_result.answer_text,
            "judge_prompt": judge_result.prompt,
            "judge_verdict": judge_result.verdict,
            "judge_score": judge_result.score,
            "judge_rationale": judge_result.rationale,
            "answer_record_id": answer_result.answer_record_id,
            "retrieval_event_id": answer_result.retrieval_event_id,
            "retrieval_source_refs": answer_result.retrieval_source_refs,
            "retrieval_source_ids": answer_result.retrieval_source_ids,
            "retrieval_reflection_retry_used": bool(
                answer_result.metadata.get("retrieval_reflection_retry_used")
            ),
            "retrieval_reflection_used": bool(answer_result.metadata.get("retrieval_reflection_used")),
            "retrieval_reflection_stage": str(answer_result.metadata.get("retrieval_reflection_stage") or "none"),
            "initial_answer_abstained": bool(answer_result.metadata.get("initial_answer_abstained")),
            "initial_retrieval_bundle_weak": bool(answer_result.metadata.get("initial_retrieval_bundle_weak")),
            "reflection_rewritten_query": answer_result.metadata.get("reflection_rewritten_query") or "",
            "reflection_must_find_terms": list(answer_result.metadata.get("reflection_must_find_terms") or []),
            "reflection_candidate_page_slugs": list(
                answer_result.metadata.get("reflection_candidate_page_slugs") or []
            ),
            "reflection_reroute_event_id": answer_result.metadata.get("reflection_reroute_event_id"),
            "post_reflection_raw_rescue_used": bool(
                answer_result.metadata.get("post_reflection_raw_rescue_used")
            ),
            "post_reflection_raw_rescue_reason": answer_result.metadata.get("post_reflection_raw_rescue_reason"),
            "post_reflection_raw_rescue_event_id": answer_result.metadata.get(
                "post_reflection_raw_rescue_event_id"
            ),
            "final_reflection_retrieval_event_id": answer_result.metadata.get("final_reflection_retrieval_event_id"),
            "raw_rescue_attempted": bool(answer_result.metadata.get("raw_rescue_attempted")),
            "raw_rescue_trigger_reasons": list(answer_result.metadata.get("raw_rescue_trigger_reasons") or []),
            "raw_rescue_skipped_reason": answer_result.metadata.get("raw_rescue_skipped_reason"),
            "reflection_required_terms": list(answer_result.metadata.get("reflection_required_terms") or []),
            "reflection_covered_terms": list(answer_result.metadata.get("reflection_covered_terms") or []),
            "reflection_uncovered_terms": list(answer_result.metadata.get("reflection_uncovered_terms") or []),
            "reflection_term_coverage_rate": answer_result.metadata.get("reflection_term_coverage_rate"),
            "reflection_semantic_evidence_weak": bool(
                answer_result.metadata.get("reflection_semantic_evidence_weak")
            ),
            "raw_rescue_used": bool(answer_result.metadata.get("raw_rescue_used")),
            "raw_rescue_hit_count": int(answer_result.metadata.get("raw_rescue_hit_count") or 0),
            "raw_rescue_source_refs": list(answer_result.metadata.get("raw_rescue_source_refs") or []),
            "raw_rescue_compensated_memory_gap": bool(
                answer_result.metadata.get("raw_rescue_compensated_memory_gap")
            ),
            "reflection_answer_changed": bool(answer_result.metadata.get("reflection_answer_changed")),
            "metrics": metrics,
            "tokens": token_payload,
            "latency_ms": latency_payload,
            "cache_hit": bool(answer_result.metadata.get("cache_hit")),
            "metadata": {
                "source_file": answer_result.metadata.get("source_file"),
                "query_metadata": query_metadata,
                "answer_metadata": answer_result.metadata.get("answer_metadata", {}),
                "backbone_usage": usage,
                "llm_usage": llm_usage,
                "llm_usage_by_task": llm_usage_by_task,
                "llm_fallback_counts": llm_fallback_counts,
                "llm_repair_counts": llm_repair_counts,
                "judge_metadata": judge_result.metadata,
                "semantic_metrics": semantic_result.metadata if semantic_result is not None else {},
                "retrieval_compact_diagnostics": dict(
                    answer_result.metadata.get("retrieval_compact_diagnostics") or {}
                ),
                "memory_build": {
                    "extraction_repair_count": int(answer_result.metadata.get("extraction_repair_count", 0)),
                    "extraction_fallback_count": int(answer_result.metadata.get("extraction_fallback_count", 0)),
                    "closed_on_fallback_count": int(answer_result.metadata.get("closed_on_fallback_count", 0)),
                    "structured_attempt_count": int(answer_result.metadata.get("structured_attempt_count", 0)),
                    "structured_success_count": int(answer_result.metadata.get("structured_success_count", 0)),
                    "structured_fallback_count": int(answer_result.metadata.get("structured_fallback_count", 0)),
                    "structured_attempts_by_task": dict(answer_result.metadata.get("structured_attempts_by_task", {})),
                    "structured_successes_by_task": dict(answer_result.metadata.get("structured_successes_by_task", {})),
                    "structured_fallbacks_by_task": dict(answer_result.metadata.get("structured_fallbacks_by_task", {})),
                    "structured_attempts_by_vendor": dict(answer_result.metadata.get("structured_attempts_by_vendor", {})),
                    "structured_successes_by_vendor": dict(answer_result.metadata.get("structured_successes_by_vendor", {})),
                    "structured_fallbacks_by_vendor": dict(answer_result.metadata.get("structured_fallbacks_by_vendor", {})),
                    "trajectory_match_total_open": int(answer_result.metadata.get("trajectory_match_total_open", 0)),
                    "trajectory_match_prefiltered_count": int(answer_result.metadata.get("trajectory_match_prefiltered_count", 0)),
                    "trajectory_match_shortlist_count": int(answer_result.metadata.get("trajectory_match_shortlist_count", 0)),
                    "wiki_compile_latency_ms": float(answer_result.metadata.get("wiki_compile_latency_ms", 0.0)),
                    "episodic_batch_call_count": int(answer_result.metadata.get("episodic_batch_call_count", 0)),
                    "episodic_batch_item_count": int(answer_result.metadata.get("episodic_batch_item_count", 0)),
                    "episodic_batch_repair_after_batch_count": int(
                        answer_result.metadata.get("episodic_batch_repair_after_batch_count", 0)
                    ),
                    "episodic_batch_oom_backoff_count": int(
                        answer_result.metadata.get("episodic_batch_oom_backoff_count", 0)
                    ),
                    "episodic_batch_effective_size_max": int(
                        answer_result.metadata.get("episodic_batch_effective_size_max", 1)
                    ),
                    "episodic_batch_effective_size_final": int(
                        answer_result.metadata.get("episodic_batch_effective_size_final", 1)
                    ),
                    "memory_stage_batch_stats": dict(answer_result.metadata.get("memory_stage_batch_stats", {})),
                    "assistant_only_exchange_count": int(
                        answer_result.metadata.get("assistant_only_exchange_count", 0)
                    ),
                    "forced_memory_seed_count": int(answer_result.metadata.get("forced_memory_seed_count", 0)),
                    "low_salience_memory_count": int(answer_result.metadata.get("low_salience_memory_count", 0)),
                    "llm_no_memory_forced_count": int(answer_result.metadata.get("llm_no_memory_forced_count", 0)),
                    "zero_claim_episodic_candidate_count": int(
                        answer_result.metadata.get("zero_claim_episodic_candidate_count", 0)
                    ),
                    "zero_claim_episodic_persisted_count": int(
                        answer_result.metadata.get("zero_claim_episodic_persisted_count", 0)
                    ),
                    "zero_claim_low_salience_skipped_count": int(
                        answer_result.metadata.get("zero_claim_low_salience_skipped_count", 0)
                    ),
                    "memory_debug_artifact_paths": list(answer_result.metadata.get("memory_debug_artifact_paths", [])),
                },
                "judge_parallelism": self.judge_parallelism,
            },
        }

    def _apply_locomo_judge_semantic_override(
        self,
        answer_result: AnswerResult,
        judge_result: JudgeResult,
        judge_acc: float | None,
        semantic_result: SemanticMetricResult,
    ) -> tuple[JudgeResult, float | None]:
        verdict = str(judge_result.verdict or "").lower()
        if verdict == "correct":
            return judge_result, judge_acc
        semantic_metadata = dict(semantic_result.metadata or {})
        if float(semantic_result.f1) < 0.999999:
            return judge_result, judge_acc
        unmatched_reference = list(semantic_metadata.get("f1_unmatched_reference_items") or [])
        if unmatched_reference:
            return judge_result, judge_acc
        if self._answer_text_is_abstention(answer_result.answer_text):
            return judge_result, judge_acc
        if self._locomo_question_is_count_like(answer_result.question):
            reference_items = list(semantic_metadata.get("f1_reference_items") or [])
            exact_overlap_items = set(str(item) for item in list(semantic_metadata.get("f1_exact_overlap_items") or []))
            if not reference_items or any(str(item) not in exact_overlap_items for item in reference_items):
                return judge_result, judge_acc

        metadata = dict(judge_result.metadata or {})
        metadata.update(
            {
                "raw_judge_verdict": judge_result.verdict,
                "raw_judge_score": judge_result.score,
                "raw_judge_rationale": judge_result.rationale,
                "judge_semantic_override_used": True,
                "judge_semantic_override_reason": "semantic_f1_full_coverage",
            }
        )
        overridden = JudgeResult(
            verdict="correct",
            prompt=judge_result.prompt,
            score=1.0,
            rationale=judge_result.rationale,
            prompt_tokens=judge_result.prompt_tokens,
            completion_tokens=judge_result.completion_tokens,
            latency_ms=judge_result.latency_ms,
            metadata=metadata,
        )
        return overridden, 1.0

    @staticmethod
    def _answer_text_is_abstention(answer_text: str) -> bool:
        text = " ".join(str(answer_text or "").lower().split())
        if not text:
            return True
        abstention_patterns = [
            r"\bdoes not support\b",
            r"\bdoes not provide\b",
            r"\bdoes not mention\b",
            r"\bnot enough information\b",
            r"\binsufficient information\b",
            r"\bcannot answer\b",
            r"\bcan't answer\b",
            r"\bnot specified\b",
            r"\bnot available\b",
        ]
        return any(re.search(pattern, text) for pattern in abstention_patterns)

    @staticmethod
    def _locomo_question_is_count_like(question: str) -> bool:
        text = " ".join(str(question or "").lower().split())
        return bool(
            re.search(
                r"\b(how many|how much|how often|number of|count of|times has|times did|times was|times were)\b",
                text,
            )
        )

    def _build_evaluation_rows(
        self,
        answer_result: AnswerResult,
        metrics: dict[str, float | None],
        judge_result: JudgeResult,
    ) -> list[EvaluationRecord]:
        rows: list[EvaluationRecord] = []
        for index, (metric_name, metric_value) in enumerate(metrics.items()):
            is_judge_metric = metric_name == "judge_acc"
            rows.append(
                EvaluationRecord(
                    id=f"{answer_result.query_task_id}-{index:02d}-{metric_name}",
                    sample_id=answer_result.sample_id,
                    query_task_id=answer_result.query_task_id,
                    answer_record_id=answer_result.answer_record_id,
                    dataset_name=answer_result.dataset_name,
                    metric_name=metric_name,
                    metric_value=None if metric_value is None else float(metric_value),
                    verdict=judge_result.verdict if is_judge_metric else None,
                    rationale=judge_result.rationale if is_judge_metric else None,
                    details_json={
                        "judge_score": judge_result.score if is_judge_metric else None,
                        "judge_prompt": judge_result.prompt if is_judge_metric else None,
                        "retrieval_source_refs": answer_result.retrieval_source_refs,
                        "rubric": answer_result.rubric,
                        "gold_answer": answer_result.gold_answer,
                    },
                )
            )
        return rows

    def _build_details_payload(self, answer_results: list[AnswerResult], evaluated_rows: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.run_dir is not None
        return {
            "run_meta": self._run_meta(
                processed_samples=self.selected_logical_sample_count,
                processed_queries=len(answer_results),
            ),
            "samples": evaluated_rows,
            "paths": {
                "run_dir": str(self.run_dir),
                "database": str(self.config.database_path),
                "details": str(self.run_dir / "details.json"),
                "summary": str(self.run_dir / "summary.json"),
                "fallback_repair_events": str(self.run_dir / "fallback_repair_events.jsonl"),
                "gold_labels": str(self.run_dir / "analysis" / "gold_labels.jsonl"),
                "offline_ablation_rows": str(self.run_dir / "analysis" / "offline_ablation_rows.jsonl"),
                "offline_ablation_summary": str(self.run_dir / "analysis" / "offline_ablation_summary.json"),
                "offline_ablation_table": str(self.run_dir / "analysis" / "offline_ablation_table.csv"),
                "evidence_funnel": str(self.run_dir / "analysis" / "evidence_funnel.csv"),
                "cost_recall_curve": str(self.run_dir / "analysis" / "cost_recall_curve.csv"),
                "variant_examples": str(self.run_dir / "analysis" / "variant_examples.jsonl"),
                "cost_call_rows": str(self.run_dir / "analysis" / "cost_call_rows.jsonl"),
                "cost_query_rows": str(self.run_dir / "analysis" / "cost_query_rows.jsonl"),
                "cost_reconciliation": str(
                    self.run_dir / "analysis" / "cost_reconciliation.json"
                ),
                "cost_reconciliation_csv": str(
                    self.run_dir / "analysis" / "cost_reconciliation.csv"
                ),
                "cost_phase_summary": str(self.run_dir / "analysis" / "cost_phase_summary.csv"),
                "cost_quality_table": str(self.run_dir / "analysis" / "cost_quality_table.csv"),
                "amortization_break_even": str(self.run_dir / "analysis" / "amortization_break_even.csv"),
                "amortized_cost_curve": str(self.run_dir / "analysis" / "amortized_cost_curve.csv"),
                "memory_scaling": str(self.run_dir / "analysis" / "memory_scaling.csv"),
                "candidate_scaling": str(self.run_dir / "analysis" / "candidate_scaling.csv"),
                "cost_benefit_summary": str(self.run_dir / "analysis" / "cost_benefit_summary.json"),
                "answer_support_rows": str(self.run_dir / "analysis" / "answer_support_rows.jsonl"),
                "answer_context_claim_rows": str(
                    self.run_dir / "analysis" / "answer_context_claim_rows.jsonl"
                ),
                "claim_lifecycle_rows": str(self.run_dir / "analysis" / "claim_lifecycle_rows.jsonl"),
                "audit_packet_rows": str(self.run_dir / "analysis" / "audit_packet_rows.jsonl"),
                "auditability_rows": str(self.run_dir / "analysis" / "auditability_rows.jsonl"),
                "auditability_summary": str(self.run_dir / "analysis" / "auditability_summary.json"),
                "source_support_table": str(self.run_dir / "analysis" / "source_support_table.csv"),
                "unsupported_answer_table": str(self.run_dir / "analysis" / "unsupported_answer_table.csv"),
                "failure_localization_table": str(
                    self.run_dir / "analysis" / "failure_localization_table.csv"
                ),
                "conflict_obsolete_table": str(self.run_dir / "analysis" / "conflict_obsolete_table.csv"),
                "audit_packet_cost": str(self.run_dir / "analysis" / "audit_packet_cost.csv"),
                "audit_examples": str(self.run_dir / "analysis" / "audit_examples.jsonl"),
                "text_only_filter_manifest": str(self.run_dir / "analysis" / "text_only_filter_manifest.json"),
                "text_only_filtered_summary": str(self.run_dir / "analysis" / "text_only_filtered_summary.json"),
            },
        }

    def _finalize_fallback_repair_diagnostics(self, evaluated_rows: list[dict[str, Any]]) -> None:
        run_meta = self._run_meta(
            processed_samples=self.selected_logical_sample_count,
            processed_queries=len(evaluated_rows),
        )
        events: list[dict[str, Any]] = []
        for row in evaluated_rows:
            row_events = build_fallback_repair_events_for_row(row, run_meta=run_meta)
            events.extend(row_events)
            metadata = dict(row.get("metadata") or {})
            query_summary = summarize_events_for_query(row_events)
            metadata["fallback_repair_summary"] = {
                "event_count": query_summary["event_count"],
                "event_ids": query_summary["event_ids"],
                "diagnostic_mode": "events_v1",
            }
            metadata["fallback_repair_event_counts"] = query_summary["event_counts"]
            metadata["fallback_repair_extra_cost"] = query_summary["extra_cost"]
            metadata["fallback_repair_quality_flags"] = query_summary["quality_flags"]
            row["metadata"] = metadata
        self.fallback_repair_events = events

    def _write_fallback_repair_events(self) -> None:
        if self.run_dir is None:
            return
        path = self.run_dir / "fallback_repair_events.jsonl"
        try:
            if path.exists():
                path.unlink()
            if self.fallback_repair_events:
                append_jsonl(path, [self._json_safe(event) for event in self.fallback_repair_events])
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self._emit_event(
                "fallback_repair_events_write_failed",
                stage="artifact_write",
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=300),
            )

    def _write_text_only_filter_artifacts(self, summary_payload: dict[str, Any]) -> None:
        if self.run_dir is None or self.config.dataset != "locomo":
            return
        analysis_dir = self.run_dir / "analysis"
        try:
            write_json(
                analysis_dir / "text_only_filter_manifest.json",
                {
                    "policy": "locomo_text_only_filter_v1",
                    "summary": dict(summary_payload.get("evaluation_filters", {}).get("text_only", {})),
                    "rows": list(self.text_only_filter_manifest),
                },
            )
            write_json(
                analysis_dir / "text_only_filtered_summary.json",
                {
                    "policy": "locomo_text_only_filter_v1",
                    "evaluation_filter": dict(summary_payload.get("evaluation_filters", {}).get("text_only", {})),
                    "aggregates": dict(summary_payload.get("aggregates", {}).get("text_only_filtered", {})),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_event(
                "text_only_filter_artifact_write_failed",
                stage="artifact_write",
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=300),
            )

    def _write_gold_label_artifacts(self, evaluated_rows: list[dict[str, Any]]) -> None:
        if self.run_dir is None or self.config.dataset != "locomo" or not self.config.ablation_diagnostics:
            return
        try:
            path = write_gold_labels_artifact(
                self.run_dir,
                evaluated_rows,
                database_path=self.config.database_path,
            )
            self._emit_event(
                "gold_labels_artifact_written",
                stage="artifact_write",
                path=str(path),
                row_count=len(evaluated_rows),
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_event(
                "gold_labels_artifact_write_failed",
                stage="artifact_write",
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=300),
            )

    def _cost_call_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for index, record in enumerate(
            [*self._provider_call_records(), *self.worker_provider_call_records]
        ):
            metadata = dict(record.get("metadata") or {})
            key = str(
                record.get("call_item_uid")
                or metadata.get("call_item_uid")
                or (
                    f"legacy/{index}/"
                    f"{record.get('role') or metadata.get('role') or ''}/"
                    f"{record.get('provider_call_id') or metadata.get('provider_call_id') or ''}/"
                    f"{record.get('provider_call_uid') or metadata.get('provider_call_uid') or ''}/"
                    f"{metadata.get('batch_index')}"
                )
            )
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            records.append(dict(record))
        return records

    def _write_cost_diagnostic_artifacts(self, evaluated_rows: list[dict[str, Any]]) -> None:
        if self.run_dir is None or self.config.dataset != "locomo" or not self.config.cost_diagnostics:
            return
        try:
            paths = write_cost_diagnostic_artifacts(
                run_dir=self.run_dir,
                sample_rows=evaluated_rows,
                database_path=self.config.database_path,
                call_records=self._cost_call_records(),
                save_compact_calls=self.config.cost_call_save_mode == "compact",
            )
            self._emit_event(
                "cost_diagnostic_artifacts_written",
                stage="artifact_write",
                **paths,
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_event(
                "cost_diagnostic_artifacts_write_failed",
                stage="artifact_write",
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=300),
            )

    def _write_auditability_artifacts(self, evaluated_rows: list[dict[str, Any]]) -> None:
        if self.run_dir is None or self.config.dataset != "locomo" or not self.config.auditability_diagnostics:
            return
        try:
            paths = write_auditability_artifacts(
                run_dir=self.run_dir,
                sample_rows=evaluated_rows,
                database_path=self.config.database_path,
                packet_save_mode=self.config.audit_packet_save_mode,
            )
            self._emit_event(
                "auditability_artifacts_written",
                stage="artifact_write",
                **paths,
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_event(
                "auditability_artifacts_write_failed",
                stage="artifact_write",
                error_type=exc.__class__.__name__,
                error_message=self._compact_error_text(exc, limit=300),
            )

    def _build_summary_payload(
        self,
        *,
        answer_results: list[AnswerResult],
        evaluated_rows: list[dict[str, Any]],
        wallclock_ms: float,
    ) -> dict[str, Any]:
        assert self.run_dir is not None
        self._sync_cache_lock_stats()
        aggregates = self._aggregate_results(evaluated_rows)
        evaluation_filters: dict[str, Any] = {}
        if self.config.dataset == "locomo":
            text_only_rows = self._text_only_filtered_rows(evaluated_rows)
            aggregates["text_only_filtered"] = self._aggregate_locomo(text_only_rows)
            evaluation_filters["text_only"] = self._text_only_filter_summary_payload(evaluated_rows)
        overall_costs = self._aggregate_costs(evaluated_rows)
        summary = {
            "run_meta": self._run_meta(
                processed_samples=self.selected_logical_sample_count,
                processed_queries=len(answer_results),
            ),
            "aggregates": aggregates,
            "costs": {
                **overall_costs,
                "total_runtime_ms": wallclock_ms,
                "total_runtime_s": wallclock_ms / 1000.0,
            },
            "memory": self._memory_summary(),
            "structured": self._structured_summary(),
            "judge": self._judge_summary(evaluated_rows),
            "evaluation_filters": evaluation_filters,
            "llm_call_diagnostics": self._llm_call_diagnostics(evaluated_rows),
            "compact_retrieval_diagnostics": self._compact_retrieval_diagnostics_summary(evaluated_rows),
            "fallback_repair_diagnostics": summarize_fallback_repair_events(
                self.fallback_repair_events,
                evaluated_rows,
            ),
            "device_allocation": self.device_allocation,
            "worker_device_allocation": dict(self.worker_device_allocation),
            "cuda_preflight": self._cuda_preflight_payload(),
            "exclusions": self._build_exclusions_summary(),
            "cache": {
                **self.cache_stats,
                "cache_hit_rate": (
                    self.cache_stats["cache_hits"]
                    / max(self.cache_stats["cache_hits"] + self.cache_stats["cache_misses"], 1.0)
                ),
            },
            "paths": {
                "run_dir": str(self.run_dir),
                "database": str(self.config.database_path),
                "details": str(self.run_dir / "details.json"),
                "summary": str(self.run_dir / "summary.json"),
                "fallback_repair_events": str(self.run_dir / "fallback_repair_events.jsonl"),
                "gold_labels": str(self.run_dir / "analysis" / "gold_labels.jsonl"),
                "offline_ablation_rows": str(self.run_dir / "analysis" / "offline_ablation_rows.jsonl"),
                "offline_ablation_summary": str(self.run_dir / "analysis" / "offline_ablation_summary.json"),
                "offline_ablation_table": str(self.run_dir / "analysis" / "offline_ablation_table.csv"),
                "evidence_funnel": str(self.run_dir / "analysis" / "evidence_funnel.csv"),
                "cost_recall_curve": str(self.run_dir / "analysis" / "cost_recall_curve.csv"),
                "variant_examples": str(self.run_dir / "analysis" / "variant_examples.jsonl"),
                "cost_call_rows": str(self.run_dir / "analysis" / "cost_call_rows.jsonl"),
                "cost_query_rows": str(self.run_dir / "analysis" / "cost_query_rows.jsonl"),
                "cost_reconciliation": str(
                    self.run_dir / "analysis" / "cost_reconciliation.json"
                ),
                "cost_reconciliation_csv": str(
                    self.run_dir / "analysis" / "cost_reconciliation.csv"
                ),
                "cost_phase_summary": str(self.run_dir / "analysis" / "cost_phase_summary.csv"),
                "cost_quality_table": str(self.run_dir / "analysis" / "cost_quality_table.csv"),
                "amortization_break_even": str(self.run_dir / "analysis" / "amortization_break_even.csv"),
                "amortized_cost_curve": str(self.run_dir / "analysis" / "amortized_cost_curve.csv"),
                "memory_scaling": str(self.run_dir / "analysis" / "memory_scaling.csv"),
                "candidate_scaling": str(self.run_dir / "analysis" / "candidate_scaling.csv"),
                "cost_benefit_summary": str(self.run_dir / "analysis" / "cost_benefit_summary.json"),
                "answer_support_rows": str(self.run_dir / "analysis" / "answer_support_rows.jsonl"),
                "answer_context_claim_rows": str(
                    self.run_dir / "analysis" / "answer_context_claim_rows.jsonl"
                ),
                "claim_lifecycle_rows": str(self.run_dir / "analysis" / "claim_lifecycle_rows.jsonl"),
                "audit_packet_rows": str(self.run_dir / "analysis" / "audit_packet_rows.jsonl"),
                "auditability_rows": str(self.run_dir / "analysis" / "auditability_rows.jsonl"),
                "auditability_summary": str(self.run_dir / "analysis" / "auditability_summary.json"),
                "source_support_table": str(self.run_dir / "analysis" / "source_support_table.csv"),
                "unsupported_answer_table": str(self.run_dir / "analysis" / "unsupported_answer_table.csv"),
                "failure_localization_table": str(
                    self.run_dir / "analysis" / "failure_localization_table.csv"
                ),
                "conflict_obsolete_table": str(self.run_dir / "analysis" / "conflict_obsolete_table.csv"),
                "audit_packet_cost": str(self.run_dir / "analysis" / "audit_packet_cost.csv"),
                "audit_examples": str(self.run_dir / "analysis" / "audit_examples.jsonl"),
                "text_only_filter_manifest": str(self.run_dir / "analysis" / "text_only_filter_manifest.json"),
                "text_only_filtered_summary": str(self.run_dir / "analysis" / "text_only_filtered_summary.json"),
            },
        }
        return summary

    @staticmethod
    def _text_only_filtered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if not bool(dict(dict(row.get("metadata") or {}).get("evaluation_filter") or {}).get("excluded_from_text_only"))
        ]

    def _text_only_filter_summary_payload(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_text_only_filter(rows)
        by_subset: dict[str, Any] = {}
        for subset_key in self.dataset_scope:
            subset_rows = [row for row in rows if str(row.get("subset_key")) == str(subset_key)]
            by_subset[str(subset_key)] = summarize_text_only_filter(subset_rows)
        summary["by_subset"] = by_subset
        return summary

    @staticmethod
    def _retrieval_compact_diagnostics_for_details(metadata: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "retrieval_rank_save_mode",
            "retrieval_rank_save_limit",
            "diagnostic_top_n_pages",
            "diagnostic_top_n_trajectories",
            "page_ranked_total_count",
            "page_ranked_rows_truncated",
            "page_ranked_rows_compact_top_n",
            "ablation_page_ranked_rows_v1",
            "page_cutoff_universe_diagnostics",
            "page_granularity_diagnostic_mode",
            "selected_page_rows_compact",
            "selected_singleton_page_count",
            "selected_medium_granularity_page_count",
            "selected_page_trajectory_count_histogram",
            "singleton_page_penalty_applied",
            "medium_page_bonus_applied",
            "trajectory_ranked_total_count",
            "trajectory_ranked_rows_truncated",
            "trajectory_ranked_rows_compact_top_n",
            "ablation_trajectory_ranked_rows_v1",
            "trajectory_selection_pool_rows_compact",
            "trajectory_selection_pool_rows_total_count",
            "trajectory_selection_pool_rows_truncated",
            "trajectory_cutoff_prefix_diagnostics",
            "ablation_snapshot_candidate_rows_v1",
            "ablation_source_candidate_rows_v1",
            "answer_context_token_breakdown_v1",
        ]
        return {key: metadata.get(key) for key in keys if key in metadata}

    @staticmethod
    def _compact_retrieval_diagnostics_summary(evaluated_rows: list[dict[str, Any]]) -> dict[str, Any]:
        diagnostics = [
            dict(dict(row.get("metadata") or {}).get("retrieval_compact_diagnostics") or {})
            for row in evaluated_rows
        ]
        diagnostics = [row for row in diagnostics if row]
        if not diagnostics:
            return {
                "query_count": 0,
                "diagnostic_mode": "missing",
            }
        page_limits = [
            int(row.get("diagnostic_top_n_pages") or 0)
            for row in diagnostics
            if row.get("diagnostic_top_n_pages") is not None
        ]
        trajectory_limits = [
            int(row.get("diagnostic_top_n_trajectories") or 0)
            for row in diagnostics
            if row.get("diagnostic_top_n_trajectories") is not None
        ]
        page_total_counts = [
            int(row.get("page_ranked_total_count") or 0)
            for row in diagnostics
            if row.get("page_ranked_total_count") is not None
        ]
        trajectory_total_counts = [
            int(row.get("trajectory_ranked_total_count") or 0)
            for row in diagnostics
            if row.get("trajectory_ranked_total_count") is not None
        ]
        return {
            "query_count": len(diagnostics),
            "diagnostic_mode": "compact_retrieval_metadata",
            "saved_page_rank_limit_max": max(page_limits) if page_limits else None,
            "saved_trajectory_rank_limit_max": max(trajectory_limits) if trajectory_limits else None,
            "page_ranked_rows_truncated_count": sum(
                1 for row in diagnostics if bool(row.get("page_ranked_rows_truncated"))
            ),
            "trajectory_ranked_rows_truncated_count": sum(
                1 for row in diagnostics if bool(row.get("trajectory_ranked_rows_truncated"))
            ),
            "mean_page_ranked_total_count": (
                sum(page_total_counts) / len(page_total_counts) if page_total_counts else None
            ),
            "mean_trajectory_ranked_total_count": (
                sum(trajectory_total_counts) / len(trajectory_total_counts)
                if trajectory_total_counts
                else None
            ),
        }

    def _run_meta(self, *, processed_samples: int, processed_queries: int) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.config.dataset,
            "dataset_path": str(self.config.dataset_path),
            "output_dir": str(self.config.output_dir),
            "run_dir": str(self.run_dir) if self.run_dir is not None else None,
            "database_path": str(self.config.database_path),
            "index_database_path": str(self.config.index_database_path),
            "started_at": self.run_started_at,
            "completed_at": datetime.utcnow().isoformat(),
            "processed_samples": processed_samples,
            "processed_queries": processed_queries,
            "logical_sample_count": self.selected_logical_sample_count,
            "selected_row_count": self.selected_row_count,
            "selected_query_count": self.selected_query_count,
            "selection_strategy": self.selection_strategy,
            "selected_counts_by_subset": dict(self.selected_counts_by_subset),
            "backbone_provider_kind": self.config.backbone_provider_kind,
            "judge_provider_kind": self.config.judge_provider_kind,
            "backbone_model": self.config.backbone_model,
            "judge_model": self.config.judge_model,
            "embedding_model": self.config.embedding_model,
            "device_mode": self.config.device_mode,
            "openai_compatible_base_url": self.config.openai_compatible_base_url,
            "openai_compatible_structured_mode": self.config.openai_compatible_structured_mode,
            "openai_compatible_api_key_configured": bool(self.config.openai_compatible_api_key),
            "vllm_autostart": bool(self.config.vllm_autostart),
            "vllm_reused_existing_server": bool(
                self.vllm_server_metadata.get("vllm_reused_existing_server", False)
            ),
            "vllm_started_by_runner": bool(
                self.vllm_server_metadata.get("vllm_started_by_runner", False)
            ),
            "vllm_model": self.vllm_server_metadata.get("vllm_model") or self.config.vllm_model,
            "vllm_served_model_name": self.vllm_server_metadata.get("vllm_served_model_name")
            or self.config.vllm_served_model_name,
            "vllm_base_url": self.vllm_server_metadata.get("vllm_base_url")
            or self.config.openai_compatible_base_url,
            "vllm_pid": self.vllm_server_metadata.get("vllm_pid"),
            "vllm_startup_latency_ms": self.vllm_server_metadata.get("vllm_startup_latency_ms"),
            "vllm_keep_alive": bool(self.config.vllm_keep_alive),
            "cuda_preflight_mode": self.config.cuda_preflight_mode,
            "cuda_preflight_risk": self.cuda_preflight_report.risk,
            "cuda_preflight_warnings": list(self.cuda_preflight_report.warnings),
            "cuda_preflight_errors": list(self.cuda_preflight_report.errors),
            "cuda_preflight_assignments": list(self.cuda_preflight_report.assignments),
            "memory_extract_batch_size": self.config.memory_extract_batch_size,
            "effective_memory_extract_batch_size": self._resolve_memory_extract_batch_size(),
            "memory_stage_batch_stats": self._copy_memory_stage_batch_stats(self.memory_stage_batch_stats),
            "m": self.config.m,
            "t_pages": self.config.t_pages,
            "k": self.config.k,
            "neighbor_radius": self.config.neighbor_radius,
            "retrieval_expansion_mode": self.config.retrieval_expansion_mode,
            "ablation_diagnostics": bool(self.config.ablation_diagnostics),
            "retrieval_rank_save_mode": self.config.retrieval_rank_save_mode,
            "retrieval_rank_save_limit": self.config.retrieval_rank_save_limit,
            "offline_context_budgets": self.config.offline_context_budgets,
            "offline_rank_cutoffs": self.config.offline_rank_cutoffs,
            "cost_diagnostics": bool(self.config.cost_diagnostics),
            "cost_call_save_mode": self.config.cost_call_save_mode,
            "cost_price_config": str(self.config.cost_price_config) if self.config.cost_price_config else None,
            "future_query_counts": self.config.future_query_counts,
            "auditability_diagnostics": bool(self.config.auditability_diagnostics),
            "audit_packet_save_mode": self.config.audit_packet_save_mode,
            "max_samples": self.config.max_samples,
            "conv_workers": self.config.conv_workers,
            "worker_mode": "sharded" if self.config.conv_workers > 1 else "serial",
            "worker_shards": list(self.worker_shards),
            "worker_database_root": self.worker_database_root,
            "worker_database_local_scratch_used": self.worker_database_local_scratch_used,
            "worker_database_paths": list(self.worker_database_paths),
            "worker_database_cleanup": dict(self.worker_database_cleanup),
            "worker_database_warnings": list(self.worker_database_warnings),
            "worker_device_allocation": dict(self.worker_device_allocation),
            "estimated_backbone_inflight_limit": self._estimated_backbone_inflight_limit(),
            "judge_max_concurrency": self.config.judge_max_concurrency,
            "judge_parallelism": self.judge_parallelism,
            "dataset_scope_key": self.dataset_scope_key,
            "dataset_scope": self._dataset_scope(),
        }

    def _aggregate_results(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self.config.dataset == "locomo":
            return self._aggregate_locomo(rows)
        return self._aggregate_medmt(rows)

    def _aggregate_locomo(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["subset_key"])].append(row)
        subset_rows = [
            self._aggregate_locomo_group(subset_key, grouped.get(subset_key, []))
            for subset_key in self.dataset_scope
        ]
        overall = self._aggregate_locomo_group("overall", rows)
        return {"rows": subset_rows, "overall": overall}

    def _aggregate_locomo_group(self, subset_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        costs = self._aggregate_costs(rows)
        judge_counts = self._judge_counts(rows)
        metrics = {
            "F1": self._mean_metric(rows, "F1"),
            "BLEU-1": self._mean_metric(rows, "BLEU-1"),
            "judge_acc": self._mean_metric(rows, "judge_acc"),
        }
        return {
            "subset": subset_key,
            "count": len(rows),
            "metrics": metrics,
            "runtime_s": costs["runtime_ms"] / 1000.0,
            "prompt_tokens": costs["prompt_tokens"],
            "completion_tokens": costs["completion_tokens"],
            "total_tokens": costs["total_tokens"],
            "judge_evaluable_count": judge_counts["judge_evaluable_count"],
            "judge_execution_failed_count": judge_counts["judge_execution_failed_count"],
        }

    def _aggregate_medmt(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["subset_key"]), str(row["scene_tag"]))].append(row)
        aggregate_rows = []
        for subset_key in self.dataset_scope:
            for scene_tag in MEDMT_SCENE_ORDER:
                group_rows = grouped.get((subset_key, scene_tag), [])
                judge_counts = self._judge_counts(group_rows)
                aggregate_rows.append(
                    {
                        "subset": subset_key,
                        "scene_tag": scene_tag,
                        "count": len(group_rows),
                        "metrics": {"judge_acc": self._mean_metric(group_rows, "judge_acc")},
                        "costs": self._aggregate_costs(group_rows),
                        "judge_evaluable_count": judge_counts["judge_evaluable_count"],
                        "judge_execution_failed_count": judge_counts["judge_execution_failed_count"],
                    }
                )
        overall_counts = self._judge_counts(rows)
        overall = {
            "count": len(rows),
            "metrics": {"judge_acc": self._mean_metric(rows, "judge_acc")},
            "costs": self._aggregate_costs(rows),
            "judge_evaluable_count": overall_counts["judge_evaluable_count"],
            "judge_execution_failed_count": overall_counts["judge_execution_failed_count"],
        }
        return {"rows": aggregate_rows, "overall": overall}

    def _aggregate_costs(self, rows: Iterable[dict[str, Any]]) -> dict[str, float]:
        prompt_tokens = 0
        completion_tokens = 0
        runtime_ms = 0.0
        for row in rows:
            prompt_tokens += int(row.get("tokens", {}).get("prompt_tokens", 0))
            completion_tokens += int(row.get("tokens", {}).get("completion_tokens", 0))
            runtime_ms += float(row.get("latency_ms", {}).get("runtime_ms", 0.0))
        return {
            "runtime_ms": runtime_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _memory_summary(self) -> dict[str, Any]:
        assert self.session is not None
        assert self.orchestrator is not None
        avg_trajectory_length = self.session.scalar(
            select(func.avg(TrajectoryRecord.snapshot_count)).where(TrajectoryRecord.snapshot_count > 0)
        )
        claim_op_rows = self.session.execute(
            select(ClaimOpRecord.op_type, func.count()).group_by(ClaimOpRecord.op_type)
        ).all()
        conflict_snapshots = self.session.scalar(
            select(func.count(func.distinct(ClaimRecord.snapshot_id))).where(
                ClaimRecord.status.in_(["contradictory", "needs-confirmation"])
            )
        )
        total_snapshots = self.session.scalar(
            select(func.count(func.distinct(ClaimRecord.snapshot_id))).where(
                ClaimRecord.snapshot_id.is_not(None)
            )
        )
        snapshot_metadata_rows = self.session.execute(select(EpisodicMemorySnapshot.metadata_json)).scalars().all()
        forced_memory_seed_count = sum(
            1 for metadata in snapshot_metadata_rows if dict(metadata or {}).get("forced_episodic_seed_used_v1")
        )
        low_salience_memory_count = sum(
            1 for metadata in snapshot_metadata_rows if dict(metadata or {}).get("low_salience_memory_v1")
        )
        llm_no_memory_forced_count = sum(
            1 for metadata in snapshot_metadata_rows if dict(metadata or {}).get("llm_no_memory_overridden_v1")
        )
        zero_claim_snapshot_count = sum(
            1 for metadata in snapshot_metadata_rows if dict(metadata or {}).get("zero_claim_episodic_memory_v1")
        )
        zero_claim_persisted_count = max(
            int(self.orchestrator.zero_claim_episodic_persisted_count),
            int(zero_claim_snapshot_count),
        )
        zero_claim_skipped_count = int(self.orchestrator.zero_claim_low_salience_skipped_count)
        zero_claim_candidate_count = max(
            int(self.orchestrator.zero_claim_episodic_candidate_count),
            zero_claim_persisted_count + zero_claim_skipped_count,
        )
        avg_retrieval_latency = self.session.scalar(select(func.avg(RetrievalEvent.latency_ms)))
        return {
            "parse_success_rate": (
                self.orchestrator.parse_successes / self.orchestrator.parse_attempts
                if self.orchestrator.parse_attempts
                else 0.0
            ),
            "parse_attempts": self.orchestrator.parse_attempts,
            "parse_failures": self.orchestrator.parse_failures,
            "repair_rounds": self.orchestrator.repair_rounds,
            "extraction_fallbacks": self.orchestrator.extraction_fallbacks,
            "closed_on_fallback": self.orchestrator.closed_on_fallback,
            "trajectory_match_total_open": self.orchestrator.trajectory_match_total_open,
            "trajectory_match_prefiltered_count": self.orchestrator.trajectory_match_prefiltered,
            "trajectory_match_shortlist_count": self.orchestrator.trajectory_match_shortlisted,
            "link_salvage_count": self.orchestrator.link_salvage_count,
            "link_exchange_fallback_count": self.orchestrator.link_exchange_fallback_count,
            "ops_parse_failure_count": self.orchestrator.ops_parse_failure_count,
            "ops_ignored_count": self.orchestrator.ops_ignored_count,
            "ops_synthesized_count": self.orchestrator.ops_synthesized_count,
            "ops_model_hint_count": self.orchestrator.ops_model_hint_count,
            "ops_model_supplied_count": self.orchestrator.ops_model_supplied_count,
            "claims_parse_failure_count": self.orchestrator.claims_parse_failure_count,
            "claims_required_repair_count": self.orchestrator.claims_required_repair_count,
            "claim_text_exact_match_count": self.orchestrator.claim_text_exact_match_count,
            "claim_status_updated_count": self.orchestrator.claim_status_updated_count,
            "claim_new_add_count": self.orchestrator.claim_new_add_count,
            "claim_unmatched_previous_count": self.orchestrator.claim_unmatched_previous_count,
            "claim_transition_judge_attempt_count": self.orchestrator.claim_transition_judge_attempt_count,
            "claim_transition_judge_success_count": self.orchestrator.claim_transition_judge_success_count,
            "claim_transition_judge_fallback_count": self.orchestrator.claim_transition_judge_fallback_count,
            "claim_transition_revise_count": self.orchestrator.claim_transition_revise_count,
            "claim_transition_add_count": self.orchestrator.claim_transition_add_count,
            "episodic_batch_call_count": self.episodic_batch_call_count,
            "episodic_batch_item_count": self.episodic_batch_item_count,
            "episodic_batch_repair_after_batch_count": self.episodic_batch_repair_after_batch_count,
            "episodic_batch_oom_backoff_count": self.episodic_batch_oom_backoff_count,
            "episodic_batch_effective_size_max": self.episodic_batch_effective_size_max,
            "episodic_batch_effective_size_final": self.episodic_batch_effective_size_final,
            "memory_stage_batch_stats": self._copy_memory_stage_batch_stats(self.memory_stage_batch_stats),
            "empty_repair_target_count": self.orchestrator.empty_repair_target_count,
            "forced_memory_seed_count": forced_memory_seed_count,
            "low_salience_memory_count": low_salience_memory_count,
            "llm_no_memory_forced_count": llm_no_memory_forced_count,
            "zero_claim_episodic_candidate_count": zero_claim_candidate_count,
            "zero_claim_episodic_persisted_count": zero_claim_persisted_count,
            "zero_claim_low_salience_skipped_count": zero_claim_skipped_count,
            "structured_attempts": self.orchestrator.structured_attempts,
            "structured_successes": self.orchestrator.structured_successes,
            "structured_fallbacks": self.orchestrator.structured_fallbacks,
            "structured_attempts_by_task": dict(self.orchestrator.structured_attempts_by_task),
            "structured_successes_by_task": dict(self.orchestrator.structured_successes_by_task),
            "structured_fallbacks_by_task": dict(self.orchestrator.structured_fallbacks_by_task),
            "structured_attempts_by_vendor": dict(self.orchestrator.structured_attempts_by_vendor),
            "structured_successes_by_vendor": dict(self.orchestrator.structured_successes_by_vendor),
            "structured_fallbacks_by_vendor": dict(self.orchestrator.structured_fallbacks_by_vendor),
            "debug_artifact_count": sum(len(paths) for paths in self.orchestrator.debug_artifact_paths.values()),
            "average_trajectory_length": float(avg_trajectory_length or 0.0),
            "claim_op_distribution": {op_type: count for op_type, count in claim_op_rows},
            "conflict_rate": (float(conflict_snapshots or 0) / float(total_snapshots or 1)),
            "average_retrieval_latency_ms": float(avg_retrieval_latency or 0.0),
        }

    def _structured_summary(self) -> dict[str, Any]:
        providers = [self.llm_provider, self.judge_provider]
        records: list[dict[str, Any]] = []
        records.extend(self.worker_structured_call_metadata)
        for provider in providers:
            if isinstance(provider, MeteredLLMProvider):
                records.extend(row.metadata for row in provider.calls if row.metadata.get("structured_requested"))
        attempts = len(records)
        successes = sum(1 for row in records if row.get("structured_success"))
        fallbacks = sum(1 for row in records if row.get("structured_fallback_used"))
        attempts_by_task = Counter(str(row.get("structured_task") or "unknown") for row in records)
        attempts_by_vendor = Counter(str(row.get("structured_vendor") or "unknown") for row in records)
        attempts_by_strategy = Counter(str(row.get("structured_strategy") or "unknown") for row in records)
        return {
            "attempts": attempts,
            "successes": successes,
            "fallbacks": fallbacks,
            "attempts_by_task": dict(attempts_by_task),
            "attempts_by_vendor": dict(attempts_by_vendor),
            "attempts_by_strategy": dict(attempts_by_strategy),
        }

    def _llm_call_diagnostics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        overall = self._compact_usage_entry()
        by_task: dict[str, dict[str, Any]] = {}
        by_phase: dict[str, dict[str, Any]] = {}
        by_role: dict[str, dict[str, Any]] = {
            "backbone": self._compact_usage_entry(),
            "judge": self._compact_usage_entry(),
        }
        fallback_counts: Counter[str] = Counter()
        repair_counts: Counter[str] = Counter()
        cache_counts: Counter[str] = Counter()

        for row in rows:
            metadata = dict(row.get("metadata") or {})
            llm_usage = dict(metadata.get("llm_usage") or {})
            if not llm_usage:
                continue
            self._merge_usage_entry(overall, llm_usage)
            fallback_counts.update(dict(metadata.get("llm_fallback_counts") or {}))
            repair_counts.update(dict(metadata.get("llm_repair_counts") or {}))
            cache_counts["cache_hit_count"] += int(float(llm_usage.get("cache_hit_count", 0) or 0))
            cache_counts["cache_miss_count"] += int(float(llm_usage.get("cache_miss_count", 0) or 0))

            task_entries = dict(metadata.get("llm_usage_by_task") or {})
            for task, task_usage in task_entries.items():
                task = str(task)
                entry = dict(task_usage or {})
                by_task[task] = by_task.get(task, self._compact_usage_entry())
                self._merge_usage_entry(by_task[task], entry)
                phase = phase_for_task(task)
                by_phase[phase] = by_phase.get(phase, self._compact_usage_entry())
                self._merge_usage_entry(by_phase[phase], entry)
                role = "judge" if phase == "judge" else "backbone"
                if phase == "semantic_metrics":
                    role = "judge"
                by_role[role] = by_role.get(role, self._compact_usage_entry())
                self._merge_usage_entry(by_role[role], entry)

        overall["fallback_count"] = sum(fallback_counts.values())
        overall["repair_count"] = sum(repair_counts.values())
        overall["cache_hit_count"] = cache_counts["cache_hit_count"]
        overall["cache_miss_count"] = cache_counts["cache_miss_count"]
        return {
            "overall": overall,
            "by_role": by_role,
            "by_phase": dict(sorted(by_phase.items())),
            "by_task": dict(
                sorted(
                    by_task.items(),
                    key=lambda item: float(item[1].get("total_tokens", 0) or 0),
                    reverse=True,
                )
            ),
            "fallbacks": {
                "fallback_count": sum(fallback_counts.values()),
                "fallback_counts": dict(fallback_counts),
            },
            "repairs": {
                "repair_count": sum(repair_counts.values()),
                "repair_counts": dict(repair_counts),
            },
            "cache": dict(cache_counts),
        }

    def _judge_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        average_queue_latency_ms = (
            self.judge_queue_latency_ms_total / self.judge_queue_events if self.judge_queue_events else 0.0
        )
        judge_counts = self._judge_counts(rows)
        return {
            "parallelism": self.judge_parallelism,
            "total_wall_time_ms": self.judge_total_wall_time_ms,
            "average_queue_latency_ms": average_queue_latency_ms,
            "judge_evaluable_count": judge_counts["judge_evaluable_count"],
            "judge_execution_failed_count": judge_counts["judge_execution_failed_count"],
        }

    def _print_tables(self, rows: list[dict[str, Any]]) -> None:
        if self.config.dataset == "locomo":
            self._print_locomo_table(rows)
        else:
            self._print_medmt_table(rows)

    def _print_locomo_table(self, rows: list[dict[str, Any]]) -> None:
        aggregates = self._aggregate_locomo(rows)
        table = Table(title="LOCOMO Benchmark Summary")
        table.add_column("Subset")
        table.add_column("Count", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("BLEU-1", justify="right")
        table.add_column("judge_acc", justify="right")
        table.add_column("judge_eval", justify="right")
        table.add_column("judge_errors", justify="right")
        table.add_column("runtime_s", justify="right")
        table.add_column("prompt_tokens", justify="right")
        table.add_column("completion_tokens", justify="right")
        table.add_column("total_tokens", justify="right")
        for row in aggregates["rows"]:
            metrics = row["metrics"]
            table.add_row(
                row["subset"],
                str(row["count"]),
                self._render_metric_value(metrics["F1"]),
                self._render_metric_value(metrics["BLEU-1"]),
                self._render_metric_value(metrics["judge_acc"]),
                str(int(row.get("judge_evaluable_count", 0))),
                str(int(row.get("judge_execution_failed_count", 0))),
                f"{row['runtime_s']:.2f}",
                str(row["prompt_tokens"]),
                str(row["completion_tokens"]),
                str(row["total_tokens"]),
            )
        overall = aggregates["overall"]
        metrics = overall["metrics"]
        table.add_row(
            "Overall",
            str(overall["count"]),
            self._render_metric_value(metrics["F1"]),
            self._render_metric_value(metrics["BLEU-1"]),
            self._render_metric_value(metrics["judge_acc"]),
            str(int(overall.get("judge_evaluable_count", 0))),
            str(int(overall.get("judge_execution_failed_count", 0))),
            f"{overall['runtime_s']:.2f}",
            str(overall["prompt_tokens"]),
            str(overall["completion_tokens"]),
            str(overall["total_tokens"]),
        )
        self.console.print(table)
        self._print_locomo_text_only_filtered_table(rows)

    def _print_locomo_text_only_filtered_table(self, rows: list[dict[str, Any]]) -> None:
        if not any(dict(dict(row.get("metadata") or {}).get("evaluation_filter") or {}) for row in rows):
            return
        included_rows = self._text_only_filtered_rows(rows)
        aggregates = self._aggregate_locomo(included_rows)
        filter_summary = self._text_only_filter_summary_payload(rows)
        by_subset = dict(filter_summary.get("by_subset") or {})
        table = Table(title="LOCOMO Benchmark Summary (Text-Only Filtered, Visual/OCR Excluded)")
        table.add_column("Subset")
        table.add_column("Count", justify="right")
        table.add_column("Excluded", justify="right")
        table.add_column("Ambiguous", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("BLEU-1", justify="right")
        table.add_column("judge_acc", justify="right")
        table.add_column("judge_eval", justify="right")
        table.add_column("judge_errors", justify="right")
        table.add_column("runtime_s", justify="right")
        for row in aggregates["rows"]:
            subset = str(row["subset"])
            metrics = row["metrics"]
            subset_filter = dict(by_subset.get(subset) or {})
            table.add_row(
                subset,
                str(row["count"]),
                str(int(subset_filter.get("excluded_count") or 0)),
                str(int(subset_filter.get("ambiguous_count") or 0)),
                self._render_metric_value(metrics["F1"]),
                self._render_metric_value(metrics["BLEU-1"]),
                self._render_metric_value(metrics["judge_acc"]),
                str(int(row.get("judge_evaluable_count", 0))),
                str(int(row.get("judge_execution_failed_count", 0))),
                f"{row['runtime_s']:.2f}",
            )
        overall = aggregates["overall"]
        metrics = overall["metrics"]
        table.add_row(
            "Overall",
            str(overall["count"]),
            str(int(filter_summary.get("excluded_count") or 0)),
            str(int(filter_summary.get("ambiguous_count") or 0)),
            self._render_metric_value(metrics["F1"]),
            self._render_metric_value(metrics["BLEU-1"]),
            self._render_metric_value(metrics["judge_acc"]),
            str(int(overall.get("judge_evaluable_count", 0))),
            str(int(overall.get("judge_execution_failed_count", 0))),
            f"{overall['runtime_s']:.2f}",
        )
        self.console.print(table)
        self._trace(
            "text_only_filtered_metrics "
            f"count={overall['count']} excluded={int(filter_summary.get('excluded_count') or 0)} "
            f"ambiguous={int(filter_summary.get('ambiguous_count') or 0)} "
            f"F1={self._render_metric_value(metrics['F1'])} "
            f"BLEU-1={self._render_metric_value(metrics['BLEU-1'])} "
            f"judge_acc={self._render_metric_value(metrics['judge_acc'])}"
        )

    def _print_medmt_table(self, rows: list[dict[str, Any]]) -> None:
        aggregates = self._aggregate_medmt(rows)
        table = Table(title="MedMT Benchmark Summary")
        table.add_column("Subset")
        table.add_column("Scene")
        table.add_column("Count", justify="right")
        table.add_column("judge_acc", justify="right")
        table.add_column("judge_eval", justify="right")
        table.add_column("judge_errors", justify="right")
        for row in aggregates["rows"]:
            table.add_row(
                MEDMT_SUBSET_LABELS.get(str(row["subset"]), str(row["subset"]).replace("_", " ").title()),
                str(row["scene_tag"]),
                str(row["count"]),
                self._render_metric_value(row["metrics"]["judge_acc"]),
                str(int(row.get("judge_evaluable_count", 0))),
                str(int(row.get("judge_execution_failed_count", 0))),
            )
        overall = aggregates["overall"]
        table.add_row(
            "Overall",
            "-",
            str(overall["count"]),
            self._render_metric_value(overall["metrics"]["judge_acc"]),
            str(int(overall.get("judge_evaluable_count", 0))),
            str(int(overall.get("judge_execution_failed_count", 0))),
        )
        self.console.print(table)
        excluded_count = len(self.excluded_samples)
        if excluded_count:
            self.console.print(f"Excluded MedMT samples: {excluded_count}")

    def _collect_exclusions(self, samples: list[DatasetSample]) -> list[dict[str, Any]]:
        excluded: list[dict[str, Any]] = []
        for sample in samples:
            if sample.excluded:
                excluded.append(
                    {
                        "sample_id": sample.sample_id,
                        "subset_key": sample.subset_key,
                        "scene_tag": sample.scene_tag,
                        "reason": sample.exclusion_reason or "excluded",
                        "source_file": sample.payload.get("source_file"),
                    }
                )
        return excluded

    def _build_exclusions_summary(self) -> dict[str, Any]:
        by_subset = Counter(str(item.get("subset_key") or "unknown") for item in self.excluded_samples)
        by_reason = Counter(str(item.get("reason") or "unknown") for item in self.excluded_samples)
        skipped_image_count = int(by_reason.get("contains_image", 0))
        return {
            "excluded_count": len(self.excluded_samples),
            "skipped_count": skipped_image_count,
            "skipped_image_count": skipped_image_count,
            "sample_ids": [item["sample_id"] for item in self.excluded_samples],
            "by_subset": dict(by_subset),
            "by_reason": dict(by_reason),
        }

    def _persist_run_reporting(self, summary_payload: dict[str, Any]) -> None:
        assert self.store is not None
        self._ensure_reporting_schema_compatibility()
        run_meta_payload = self._run_meta_record_payload(summary_payload)
        aggregate_payloads = self._aggregate_metric_payloads(summary_payload)
        self.store.record_run_meta(RunMetaRecord(**run_meta_payload))
        self.store.replace_aggregate_metrics(
            str(run_meta_payload["run_id"]),
            [AggregateMetricRecord(**payload) for payload in aggregate_payloads],
        )
        self._commit_session(stage="run_reporting")

    def _persist_run_index(self, summary_payload: dict[str, Any]) -> None:
        session_factory = create_index_schema(self.config.index_database_path)
        session = session_factory()
        try:
            self._ensure_index_schema_compatibility(session)
            run_meta_payload = self._run_meta_record_payload(summary_payload)
            aggregate_payloads = self._aggregate_metric_payloads(summary_payload)
            session.merge(IndexedRunRecord(**run_meta_payload))
            session.execute(
                delete(IndexedAggregateMetricRecord).where(
                    IndexedAggregateMetricRecord.run_id == str(run_meta_payload["run_id"])
                )
            )
            for payload in aggregate_payloads:
                session.add(IndexedAggregateMetricRecord(**payload))
            session.commit()
        finally:
            session.close()

    def _run_meta_record_payload(self, summary_payload: dict[str, Any]) -> dict[str, Any]:
        run_meta = dict(summary_payload["run_meta"])
        costs = dict(summary_payload.get("costs", {}))
        exclusions = dict(summary_payload.get("exclusions", {}))
        return {
            "run_id": str(run_meta["run_id"]),
            "dataset": str(run_meta["dataset"]),
            "dataset_scope_key": str(run_meta.get("dataset_scope_key") or "all"),
            "run_dir": str(run_meta.get("run_dir") or self.run_dir),
            "run_database_path": str(run_meta["database_path"]),
            "backbone_model": str(run_meta["backbone_model"]),
            "backbone_provider_kind": str(run_meta["backbone_provider_kind"]),
            "judge_model": run_meta.get("judge_model"),
            "judge_provider_kind": run_meta.get("judge_provider_kind"),
            "embedding_model": str(run_meta["embedding_model"]),
            "m": int(run_meta["m"]),
            "t_pages": int(run_meta.get("t_pages", self.config.t_pages)),
            "k": int(run_meta["k"]),
            "neighbor_radius": int(run_meta.get("neighbor_radius", 1)),
            "retrieval_expansion_mode": str(
                run_meta.get("retrieval_expansion_mode") or "update_linked_plus_neighbors"
            ),
            "started_at": run_meta.get("started_at"),
            "completed_at": run_meta.get("completed_at"),
            "processed_samples": int(run_meta.get("processed_samples", 0)),
            "processed_queries": int(run_meta.get("processed_queries", 0)),
            "excluded_count": int(exclusions.get("excluded_count", 0)),
            "total_runtime_s": float(costs.get("total_runtime_s", 0.0)),
            "prompt_tokens": int(costs.get("prompt_tokens", 0)),
            "completion_tokens": int(costs.get("completion_tokens", 0)),
            "total_tokens": int(costs.get("total_tokens", 0)),
            "metadata_json": {
                "dataset_path": run_meta.get("dataset_path"),
                "output_dir": run_meta.get("output_dir"),
                "index_database_path": run_meta.get("index_database_path"),
                "device_mode": run_meta.get("device_mode"),
                "openai_compatible_base_url": run_meta.get("openai_compatible_base_url"),
                "openai_compatible_structured_mode": run_meta.get(
                    "openai_compatible_structured_mode"
                ),
                "openai_compatible_api_key_configured": run_meta.get(
                    "openai_compatible_api_key_configured"
                ),
                "vllm_autostart": run_meta.get("vllm_autostart"),
                "vllm_reused_existing_server": run_meta.get("vllm_reused_existing_server"),
                "vllm_started_by_runner": run_meta.get("vllm_started_by_runner"),
                "vllm_model": run_meta.get("vllm_model"),
                "vllm_served_model_name": run_meta.get("vllm_served_model_name"),
                "vllm_base_url": run_meta.get("vllm_base_url"),
                "vllm_pid": run_meta.get("vllm_pid"),
                "vllm_startup_latency_ms": run_meta.get("vllm_startup_latency_ms"),
                "vllm_keep_alive": run_meta.get("vllm_keep_alive"),
                "cuda_preflight_mode": run_meta.get("cuda_preflight_mode"),
                "cuda_preflight_risk": run_meta.get("cuda_preflight_risk"),
                "cuda_preflight_warnings": run_meta.get("cuda_preflight_warnings", []),
                "cuda_preflight_errors": run_meta.get("cuda_preflight_errors", []),
                "cuda_preflight_assignments": run_meta.get("cuda_preflight_assignments", []),
                "max_samples": run_meta.get("max_samples"),
                "conv_workers": run_meta.get("conv_workers"),
                "worker_mode": run_meta.get("worker_mode"),
                "worker_shards": run_meta.get("worker_shards", []),
                "worker_database_root": run_meta.get("worker_database_root"),
                "worker_database_local_scratch_used": run_meta.get("worker_database_local_scratch_used"),
                "worker_database_paths": run_meta.get("worker_database_paths", []),
                "worker_database_cleanup": run_meta.get("worker_database_cleanup", {}),
                "worker_database_warnings": run_meta.get("worker_database_warnings", []),
                "effective_memory_extract_batch_size": run_meta.get("effective_memory_extract_batch_size"),
                "estimated_backbone_inflight_limit": run_meta.get("estimated_backbone_inflight_limit"),
                "judge_parallelism": run_meta.get("judge_parallelism"),
                "logical_sample_count": run_meta.get("logical_sample_count"),
                "selected_row_count": run_meta.get("selected_row_count"),
                "selected_query_count": run_meta.get("selected_query_count"),
                "selection_strategy": run_meta.get("selection_strategy"),
                "selected_counts_by_subset": run_meta.get("selected_counts_by_subset", {}),
                "retrieval_expansion_mode": run_meta.get("retrieval_expansion_mode"),
                "dataset_scope_key": run_meta.get("dataset_scope_key"),
                "dataset_scope": run_meta.get("dataset_scope", []),
                "device_allocation": summary_payload.get("device_allocation", {}),
            },
        }

    def _aggregate_metric_payloads(self, summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = str(summary_payload["run_meta"]["run_id"])
        dataset = str(summary_payload["run_meta"]["dataset"])
        payloads: list[dict[str, Any]] = []

        def add_metric_rows(
            *,
            group_level: str,
            metrics: dict[str, float],
            sample_count: int,
            runtime_ms: float,
            prompt_tokens: int,
            completion_tokens: int,
            total_tokens: int,
            subset_key: str | None = None,
            scene_tag: str | None = None,
        ) -> None:
            row_key = self._slugify(
                "|".join(
                    [
                        run_id,
                        group_level,
                        subset_key or "all",
                        scene_tag or "none",
                    ]
                )
            )
            for metric_name, metric_value in metrics.items():
                if metric_value is None:
                    continue
                payloads.append(
                    {
                        "id": f"{run_id}-{row_key}-{metric_name}",
                        "run_id": run_id,
                        "dataset": dataset,
                        "group_level": group_level,
                        "subset_key": subset_key,
                        "scene_tag": scene_tag,
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                        "sample_count": int(sample_count),
                        "runtime_ms": float(runtime_ms),
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": int(completion_tokens),
                        "total_tokens": int(total_tokens),
                    }
                )

        aggregates = dict(summary_payload["aggregates"])
        overall = dict(aggregates["overall"])
        if dataset == "locomo":
            for row in aggregates.get("rows", []):
                add_metric_rows(
                    group_level="subset",
                    subset_key=str(row["subset"]),
                    metrics=dict(row["metrics"]),
                    sample_count=int(row["count"]),
                    runtime_ms=float(row["runtime_s"]) * 1000.0,
                    prompt_tokens=int(row["prompt_tokens"]),
                    completion_tokens=int(row["completion_tokens"]),
                    total_tokens=int(row["total_tokens"]),
                )
            add_metric_rows(
                group_level="overall",
                metrics=dict(overall["metrics"]),
                sample_count=int(overall["count"]),
                runtime_ms=float(summary_payload["costs"]["total_runtime_ms"]),
                prompt_tokens=int(summary_payload["costs"]["prompt_tokens"]),
                completion_tokens=int(summary_payload["costs"]["completion_tokens"]),
                total_tokens=int(summary_payload["costs"]["total_tokens"]),
            )
            return payloads

        for row in aggregates.get("rows", []):
            costs = dict(row.get("costs", {}))
            add_metric_rows(
                group_level="scene",
                subset_key=str(row["subset"]),
                scene_tag=str(row["scene_tag"]),
                metrics=dict(row["metrics"]),
                sample_count=int(row["count"]),
                runtime_ms=float(costs.get("runtime_ms", 0.0)),
                prompt_tokens=int(costs.get("prompt_tokens", 0)),
                completion_tokens=int(costs.get("completion_tokens", 0)),
                total_tokens=int(costs.get("total_tokens", 0)),
            )
        add_metric_rows(
            group_level="overall",
            metrics=dict(overall["metrics"]),
            sample_count=int(overall["count"]),
            runtime_ms=float(summary_payload["costs"]["total_runtime_ms"]),
            prompt_tokens=int(summary_payload["costs"]["prompt_tokens"]),
            completion_tokens=int(summary_payload["costs"]["completion_tokens"]),
            total_tokens=int(summary_payload["costs"]["total_tokens"]),
        )
        return payloads

    def _build_run_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = uuid.uuid4().hex[:6]
        dataset = self._slugify(self.config.dataset)
        dataset_scope = self._slugify(self.dataset_scope_key)
        backbone = self._slugify(self.config.backbone_model)
        judge = self._slugify(self.config.judge_model or "no-judge")
        embedding = self._slugify(self.config.embedding_model)
        return (
            f"{timestamp}_r{random_suffix}_{dataset}_{dataset_scope}_{backbone}_{judge}_{embedding}"
            f"_m{self.config.m}_tp{self.config.t_pages}_k{self.config.k}_nr{self.config.neighbor_radius}"
            f"_rem{self._slugify(self.config.retrieval_expansion_mode)}"
        )

    def _dataset_scope(self) -> list[str]:
        return list(self.dataset_scope)

    def _resolve_dataset_scope(self) -> tuple[str, list[str]]:
        if self.config.dataset == "locomo":
            assert isinstance(self.adapter, LocomoAdapter)
            scope_key = self.adapter.resolve_subset_scope(self.config.dataset_path, self.config.dataset_subset)
            if scope_key == self.adapter.all_subset_key:
                return scope_key, list(LOCOMO_SUBSET_ORDER)
            return scope_key, [scope_key]
        assert isinstance(self.adapter, MedMTAdapter)
        scope_key = self.adapter.resolve_subset_scope(self.config.dataset_path, self.config.dataset_subset)
        if scope_key == self.adapter.all_subset_key:
            return scope_key, list(MEDMT_SUBSET_ORDER)
        return scope_key, [scope_key]

    @staticmethod
    def _slugify(value: str) -> str:
        compact = value.split("/")[-1]
        compact = re.sub(r"[^A-Za-z0-9]+", "-", compact).strip("-").lower()
        return compact or "model"

    @staticmethod
    def _ensure_metered(provider, *, role: str):
        if provider is None:
            return None
        if isinstance(provider, MeteredLLMProvider):
            return provider
        return MeteredLLMProvider(provider, role=role)

    def _set_meter_call_namespace(
        self,
        *,
        run_id: str,
        worker_id: str | int,
    ) -> None:
        for provider in (self.llm_provider, self.judge_provider):
            if isinstance(provider, SemaphoreLimitedLLMProvider):
                provider = provider.provider
            if isinstance(provider, MeteredLLMProvider):
                provider.set_call_namespace(run_id=run_id, worker_id=worker_id)

    def _set_meter_call_context(
        self,
        *,
        sample_id: str | None,
        query_task_id: str | None,
    ) -> None:
        for provider in (self.llm_provider, self.judge_provider):
            if isinstance(provider, SemaphoreLimitedLLMProvider):
                provider = provider.provider
            if isinstance(provider, MeteredLLMProvider):
                provider.set_call_context(
                    sample_id=sample_id,
                    query_task_id=query_task_id,
                )

    @staticmethod
    def _empty_cache_stats() -> dict[str, float]:
        return {
            "cache_hits": 0.0,
            "cache_misses": 0.0,
            "cache_writes": 0.0,
            "cache_load_latency_ms": 0.0,
            "memory_build_time_saved_ms": 0.0,
            "cache_lock_stale_removed": 0.0,
            "cache_lock_acquire_failed": 0.0,
            "cache_write_skipped_due_to_lock": 0.0,
        }

    def _sync_cache_lock_stats(self) -> None:
        for key, value in getattr(self.cache, "lock_stats", {}).items():
            self.cache_stats[key] = max(float(self.cache_stats.get(key, 0.0)), float(value))

    def _ensure_reporting_schema_compatibility(self) -> None:
        assert self.session is not None
        inspector = inspect(self.session.bind)
        self._assert_required_columns(
            inspector,
            table_name="run_meta",
            required_columns={"neighbor_radius", "dataset_scope_key", "retrieval_expansion_mode"},
            database_path=self.config.database_path,
        )

    def _ensure_index_schema_compatibility(self, session) -> None:
        inspector = inspect(session.bind)
        self._assert_required_columns(
            inspector,
            table_name="run_registry",
            required_columns={"neighbor_radius", "dataset_scope_key", "retrieval_expansion_mode"},
            database_path=self.config.index_database_path,
        )

    @staticmethod
    def _assert_required_columns(inspector, *, table_name: str, required_columns: set[str], database_path: Path | None) -> None:
        existing_tables = set(inspector.get_table_names())
        if table_name not in existing_tables:
            return
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        missing = sorted(required_columns - columns)
        if missing:
            path_text = str(database_path) if database_path is not None else "<unknown>"
            raise ParserValidationError(
                f"SQLite schema at {path_text} is missing required columns {missing} in table '{table_name}'. "
                "Delete the old SQLite file and rerun to rebuild the schema."
            )

    @staticmethod
    def _meter_snapshot(provider) -> int:
        return provider.snapshot() if provider is not None and hasattr(provider, "snapshot") else 0

    @staticmethod
    def _meter_diff(provider, start_index: int) -> dict[str, Any]:
        if provider is not None and hasattr(provider, "diff"):
            return provider.diff(start_index)
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0, "records": []}

    @staticmethod
    def _scale_llm_summary(summary: dict[str, Any], share: int) -> dict[str, Any]:
        share = max(share, 1)
        scaled: dict[str, Any] = {}
        numeric_keys = {
            "provider_call_count",
            "logical_call_count",
            "batch_call_count",
            "batch_item_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_ms",
            "error_count",
            "fallback_count",
            "repair_count",
            "cache_hit_count",
            "cache_miss_count",
        }
        for key, value in summary.items():
            if isinstance(value, dict):
                if all(isinstance(nested, (int, float)) for nested in value.values()):
                    scaled[key] = {nested_key: nested_value / share for nested_key, nested_value in value.items()}
                else:
                    scaled[key] = PipelineRunner._scale_llm_summary(value, share)
            elif key in numeric_keys and isinstance(value, (int, float)):
                scaled[key] = value / share
            else:
                scaled[key] = value
        return scaled

    def _resolve_judge_parallelism(self) -> int:
        value = self.config.judge_max_concurrency
        if isinstance(value, int):
            return max(value, 1)
        if str(value) != "auto":
            return max(int(value), 1)
        if (
            self.judge_provider is not None
            and self.judge_provider.model_info().provider_kind in REMOTE_LIKE_PROVIDER_KINDS
        ):
            return 2
        return 1

    @staticmethod
    def _split_usage(usage: dict[str, Any], share: int) -> dict[str, Any]:
        share = max(share, 1)
        tasks = list(usage.get("tasks", []))
        records = list(usage.get("records", []))
        call_summary = summarize_llm_calls(records) if records else {}
        scaled_summary = PipelineRunner._scale_llm_summary(call_summary, share) if call_summary else {}
        overall = dict(scaled_summary.get("overall", {}))
        return {
            "call_count": int(usage.get("call_count", 0)) / share,
            "provider_call_count": float(overall.get("provider_call_count", usage.get("provider_call_count", 0)))
            if overall
            else int(usage.get("provider_call_count", 0)) / share,
            "logical_call_count": float(overall.get("logical_call_count", usage.get("logical_call_count", 0)))
            if overall
            else int(usage.get("logical_call_count", usage.get("call_count", 0))) / share,
            "batch_call_count": float(overall.get("batch_call_count", usage.get("batch_call_count", 0)))
            if overall
            else int(usage.get("batch_call_count", 0)) / share,
            "batch_item_count": float(overall.get("batch_item_count", usage.get("batch_item_count", 0)))
            if overall
            else int(usage.get("batch_item_count", 0)) / share,
            "avg_batch_size": float(overall.get("avg_batch_size", usage.get("avg_batch_size", 0.0)))
            if overall
            else float(usage.get("avg_batch_size", 0.0)),
            "max_batch_size": int(overall.get("max_batch_size", usage.get("max_batch_size", 0)))
            if overall
            else int(usage.get("max_batch_size", 0)),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)) // share,
            "completion_tokens": int(usage.get("completion_tokens", 0)) // share,
            "total_tokens": int(usage.get("total_tokens", 0)) // share,
            "latency_ms": float(usage.get("latency_ms", 0.0)) / share,
            "tasks": tasks,
            "task_counts": dict(Counter(str(task) for task in tasks)),
            "by_phase": dict(scaled_summary.get("by_phase", {})),
            "by_task": dict(scaled_summary.get("by_task", {})),
            "fallback_counts": dict((scaled_summary.get("fallbacks") or {}).get("fallback_counts", {})),
            "repair_counts": dict((scaled_summary.get("repairs") or {}).get("repair_counts", {})),
            "cache_counts": dict(scaled_summary.get("cache", {})),
            "record_count": len(records),
            "records_suppressed": True,
        }

    @staticmethod
    def _compact_usage_entry(
        *,
        provider_call_count: float = 0.0,
        logical_call_count: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        fallback_count: float = 0.0,
        repair_count: float = 0.0,
        error_count: float = 0.0,
        cache_hit_count: float = 0.0,
        cache_miss_count: float = 0.0,
    ) -> dict[str, Any]:
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        return {
            "provider_call_count": provider_call_count,
            "logical_call_count": logical_call_count,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": total_tokens,
            "latency_ms": float(latency_ms),
            "fallback_count": fallback_count,
            "repair_count": repair_count,
            "error_count": error_count,
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
        }

    @staticmethod
    def _merge_usage_entry(target: dict[str, Any], entry: dict[str, Any]) -> None:
        for key in (
            "provider_call_count",
            "logical_call_count",
            "batch_call_count",
            "batch_item_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_ms",
            "fallback_count",
            "repair_count",
            "error_count",
            "cache_hit_count",
            "cache_miss_count",
        ):
            target[key] = float(target.get(key, 0.0)) + float(entry.get(key, 0.0))

    @staticmethod
    def _judge_llm_usage_entry(judge_result: JudgeResult) -> tuple[str, dict[str, Any], dict[str, int]]:
        metadata = dict(judge_result.metadata or {})
        task = str(metadata.get("structured_task") or metadata.get("task") or "benchmark_judge")
        execution_failed = bool(metadata.get("judge_execution_failed"))
        structured_fallback_used = bool(metadata.get("structured_fallback_used"))
        provider_call_count = 0.0 if execution_failed and not (judge_result.prompt_tokens or judge_result.completion_tokens) else 1.0
        logical_call_count = provider_call_count
        if structured_fallback_used:
            provider_call_count += 1.0
            logical_call_count += 1.0
        fallback_counts: dict[str, int] = {}
        if metadata.get("structured_supported") is False:
            fallback_counts["structured_unsupported"] = 1
        if structured_fallback_used:
            category = str(metadata.get("structured_fallback_category") or "structured_fallback")
            fallback_counts[category] = fallback_counts.get(category, 0) + 1
        return (
            task,
            PipelineRunner._compact_usage_entry(
                provider_call_count=provider_call_count,
                logical_call_count=logical_call_count,
                prompt_tokens=int(judge_result.prompt_tokens or 0),
                completion_tokens=int(judge_result.completion_tokens or 0),
                latency_ms=float(judge_result.latency_ms or 0.0),
                fallback_count=sum(fallback_counts.values()),
                error_count=1.0 if execution_failed else 0.0,
            ),
            fallback_counts,
        )

    @staticmethod
    def _semantic_llm_usage_entries(semantic_result) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, int]]:
        if semantic_result is None:
            return {}, {}, {}
        metadata = dict(semantic_result.metadata or {})
        cache_hits = dict(metadata.get("cache_hits") or {})
        cache_misses = dict(metadata.get("cache_misses") or {})
        call_counts = dict(metadata.get("llm_call_counts") or {})
        prompt_tokens = int(semantic_result.prompt_tokens or 0)
        completion_tokens = int(semantic_result.completion_tokens or 0)
        latency_ms = float(metadata.get("latency_ms") or 0.0)
        total_calls = max(int(metadata.get("llm_call_count") or 0), 1 if prompt_tokens or completion_tokens else 0)
        per_call_prompt = prompt_tokens // max(total_calls, 1)
        per_call_completion = completion_tokens // max(total_calls, 1)
        per_call_latency = latency_ms / max(total_calls, 1)
        entries: dict[str, dict[str, Any]] = {}
        task_map = {
            "schema": "semantic_metric_schema",
            "reference_extract": "semantic_metric_extract",
            "candidate_extract": "semantic_metric_extract",
        }
        for step, task in task_map.items():
            calls = int(call_counts.get(step) or (0 if cache_hits.get(step) else 1 if cache_misses.get(step) else 0))
            entry = PipelineRunner._compact_usage_entry(
                provider_call_count=float(calls),
                logical_call_count=float(calls),
                prompt_tokens=per_call_prompt * calls,
                completion_tokens=per_call_completion * calls,
                latency_ms=per_call_latency * calls,
                cache_hit_count=1.0 if cache_hits.get(step) else 0.0,
                cache_miss_count=1.0 if cache_misses.get(step) else 0.0,
            )
            if task in entries:
                PipelineRunner._merge_usage_entry(entries[task], entry)
            else:
                entries[task] = entry
        cache_counts = {
            "cache_hit_count": int(metadata.get("cache_hit_count") or sum(1 for value in cache_hits.values() if value)),
            "cache_miss_count": int(metadata.get("cache_miss_count") or sum(1 for value in cache_misses.values() if value)),
        }
        fallback_counts: dict[str, int] = {}
        mode = str(metadata.get("mode") or "")
        if mode == "text_json":
            fallback_counts["text_json_fallback"] = 1
        elif mode == "deterministic_fallback":
            fallback_counts["deterministic_fallback"] = 1
        return entries, fallback_counts, cache_counts

    @staticmethod
    def _build_query_llm_usage(
        backbone_usage: dict[str, Any],
        judge_result: JudgeResult,
        semantic_result,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], dict[str, int]]:
        overall = PipelineRunner._compact_usage_entry(
            provider_call_count=float(backbone_usage.get("provider_call_count", backbone_usage.get("call_count", 0)) or 0),
            logical_call_count=float(backbone_usage.get("logical_call_count", backbone_usage.get("call_count", 0)) or 0),
            prompt_tokens=int(backbone_usage.get("prompt_tokens", 0)),
            completion_tokens=int(backbone_usage.get("completion_tokens", 0)),
            latency_ms=float(backbone_usage.get("latency_ms", 0.0)),
            fallback_count=sum(float(value) for value in dict(backbone_usage.get("fallback_counts", {})).values()),
            repair_count=sum(float(value) for value in dict(backbone_usage.get("repair_counts", {})).values()),
            cache_hit_count=float(dict(backbone_usage.get("cache_counts", {})).get("cache_hit_count", 0) or 0),
            cache_miss_count=float(dict(backbone_usage.get("cache_counts", {})).get("cache_miss_count", 0) or 0),
        )
        by_task: dict[str, dict[str, Any]] = {
            str(task): dict(summary)
            for task, summary in dict(backbone_usage.get("by_task", {})).items()
        }
        fallback_counts = Counter(dict(backbone_usage.get("fallback_counts", {})))
        repair_counts = Counter(dict(backbone_usage.get("repair_counts", {})))

        judge_task, judge_entry, judge_fallbacks = PipelineRunner._judge_llm_usage_entry(judge_result)
        PipelineRunner._merge_usage_entry(overall, judge_entry)
        by_task[judge_task] = by_task.get(judge_task, PipelineRunner._compact_usage_entry())
        PipelineRunner._merge_usage_entry(by_task[judge_task], judge_entry)
        fallback_counts.update(judge_fallbacks)

        semantic_entries, semantic_fallbacks, _semantic_cache_counts = PipelineRunner._semantic_llm_usage_entries(
            semantic_result
        )
        for task, entry in semantic_entries.items():
            PipelineRunner._merge_usage_entry(overall, entry)
            by_task[task] = by_task.get(task, PipelineRunner._compact_usage_entry())
            PipelineRunner._merge_usage_entry(by_task[task], entry)
        fallback_counts.update(semantic_fallbacks)
        return overall, by_task, dict(fallback_counts), dict(repair_counts)

    @staticmethod
    def _mean_metric(rows: list[dict[str, Any]], metric_name: str) -> float | None:
        values = [
            float(value)
            for row in rows
            for value in [row.get("metrics", {}).get(metric_name)]
            if value is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _judge_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        judge_evaluable_count = 0
        judge_execution_failed_count = 0
        for row in rows:
            verdict = str(row.get("judge_verdict", "")).lower()
            if verdict in {"correct", "partial", "incorrect"}:
                judge_evaluable_count += 1
            elif verdict == "judge_error":
                judge_execution_failed_count += 1
        return {
            "judge_evaluable_count": judge_evaluable_count,
            "judge_execution_failed_count": judge_execution_failed_count,
        }

    @staticmethod
    def _render_metric_value(value: Any) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4f}"
