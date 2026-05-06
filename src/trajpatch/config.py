"""Configuration models for TrajWiki runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from trajpatch.datasets.locomo import LocomoAdapter
from trajpatch.datasets.medmt import MedMTAdapter

try:
    from pydantic import model_validator
except ImportError:  # pydantic v1
    model_validator = None
    from pydantic import root_validator


ProviderKind = Literal["mock", "remote", "local", "openai-compatible"]
DeviceMode = Literal["auto", "cpu", "single", "multi"]
CudaPreflightMode = Literal["off", "warn", "strict"]
JudgeConcurrency = Literal["auto"] | int
MemoryExtractBatchSize = Literal["auto"] | int
RetrievalExpansionMode = Literal["update_linked_plus_neighbors", "neighbors_only", "none"]
OpenAICompatibleStructuredMode = Literal["vllm", "openai_json_schema", "text_json"]
RetrievalRankSaveMode = Literal["top-n", "full"]
CostCallSaveMode = Literal["summary", "compact"]
AuditPacketSaveMode = Literal["summary", "compact"]


DATASET_SUBSET_OPTIONS = {
    "locomo": LocomoAdapter.supported_subset_keys,
    "medmt": MedMTAdapter.supported_subset_keys,
}


def _normalize_judge_max_concurrency(value: JudgeConcurrency | str) -> JudgeConcurrency:
    if value == "auto":
        return "auto"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "auto":
            return "auto"
        if stripped.isdigit() and int(stripped) >= 1:
            return int(stripped)
        raise ValueError("judge_max_concurrency must be 'auto' or an integer >= 1.")
    if isinstance(value, int) and value >= 1:
        return value
    raise ValueError("judge_max_concurrency must be 'auto' or an integer >= 1.")


def _normalize_memory_extract_batch_size(
    value: MemoryExtractBatchSize | str,
) -> MemoryExtractBatchSize:
    if value == "auto":
        return "auto"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "auto":
            return "auto"
        if stripped.isdigit() and int(stripped) >= 1:
            return int(stripped)
        raise ValueError("memory_extract_batch_size must be 'auto' or an integer >= 1.")
    if isinstance(value, int) and value >= 1:
        return value
    raise ValueError("memory_extract_batch_size must be 'auto' or an integer >= 1.")


def _normalize_openai_compatible_structured_mode(value: str) -> OpenAICompatibleStructuredMode:
    normalized = str(value or "vllm").strip().lower().replace("-", "_")
    if normalized in {"vllm", "openai_json_schema", "text_json"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(
        "openai_compatible_structured_mode must be one of: vllm, openai_json_schema, text_json."
    )


def _normalize_cuda_preflight_mode(value: str) -> CudaPreflightMode:
    normalized = str(value or "warn").strip().lower()
    if normalized in {"off", "warn", "strict"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("cuda_preflight_mode must be one of: off, warn, strict.")


def _normalize_retrieval_rank_save_mode(value: str) -> RetrievalRankSaveMode:
    normalized = str(value or "top-n").strip().lower().replace("_", "-")
    if normalized in {"top-n", "full"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("retrieval_rank_save_mode must be one of: top-n, full.")


def _normalize_cost_call_save_mode(value: str) -> CostCallSaveMode:
    normalized = str(value or "summary").strip().lower().replace("_", "-")
    if normalized in {"summary", "compact"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("cost_call_save_mode must be one of: summary, compact.")


def _normalize_audit_packet_save_mode(value: str) -> AuditPacketSaveMode:
    normalized = str(value or "summary").strip().lower().replace("_", "-")
    if normalized in {"summary", "compact"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("audit_packet_save_mode must be one of: summary, compact.")


class RunConfig(BaseModel):
    """Top-level runtime configuration."""

    dataset: Literal["locomo", "medmt"]
    dataset_subset: str | None = None
    dataset_path: Path
    output_dir: Path = Path("output")
    database_path: Path | None = None
    index_database_path: Path | None = None
    memory_cache_enabled: bool = True
    memory_cache_dir: Path = Path(".trajpatch_cache")
    rebuild_memory_cache: bool = False
    rebuild_semantic_metric_cache: bool = False
    reset_run_db: bool = True
    provider_kind: ProviderKind = "mock"
    backbone_provider_kind: ProviderKind | None = None
    judge_provider_kind: ProviderKind | None = None
    backbone_model: str = "mock-backbone"
    embedding_model: str = "huggingface/Qwen3-Embedding-8B"
    judge_model: str | None = "mock-judge"
    device_mode: DeviceMode = "auto"
    random_seed: int = 7
    m: int = Field(default=6, ge=1)
    t_pages: int = Field(default=5, ge=1)
    k: int = Field(default=3, ge=1)
    neighbor_radius: int = Field(default=1, ge=0)
    retrieval_expansion_mode: RetrievalExpansionMode = "update_linked_plus_neighbors"
    max_samples: int | None = Field(default=None, ge=1)
    conv_workers: int = Field(default=1, ge=1)
    judge_max_concurrency: JudgeConcurrency = "auto"
    memory_extract_batch_size: MemoryExtractBatchSize = "auto"
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_structured_mode: OpenAICompatibleStructuredMode = "vllm"
    vllm_autostart: bool = False
    vllm_model: str | None = None
    vllm_served_model_name: str | None = None
    vllm_host: str = "127.0.0.1"
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    vllm_cuda_visible_devices: str | None = None
    vllm_tensor_parallel_size: int = Field(default=1, ge=1)
    vllm_gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    vllm_dtype: str | None = None
    vllm_extra_args: str | None = None
    vllm_startup_timeout_s: int = Field(default=600, ge=1)
    vllm_keep_alive: bool = False
    cuda_preflight_mode: CudaPreflightMode = "warn"
    cuda_preflight_reserve_gb: float = Field(default=2.0, ge=0.0)
    cuda_preflight_report: bool = True
    ablation_diagnostics: bool = False
    retrieval_rank_save_mode: RetrievalRankSaveMode = "top-n"
    retrieval_rank_save_limit: int = Field(default=100, ge=1)
    offline_context_budgets: str = "4000,8000,16000,32000"
    offline_rank_cutoffs: str = "5,10,15,20,30,50"
    cost_diagnostics: bool = False
    cost_call_save_mode: CostCallSaveMode = "summary"
    cost_price_config: Path | None = None
    future_query_counts: str = "1,2,5,10,20,50,100"
    auditability_diagnostics: bool = False
    audit_packet_save_mode: AuditPacketSaveMode = "summary"
    episodic_match_threshold: float = 0.72
    export_jsonl: bool = True
    verbose: bool = False

    if model_validator is not None:

        @model_validator(mode="after")
        def normalize_paths(self) -> "RunConfig":
            if self.backbone_provider_kind is None:
                self.backbone_provider_kind = self.provider_kind
            if self.judge_provider_kind is None:
                self.judge_provider_kind = self.backbone_provider_kind
            self.provider_kind = self.backbone_provider_kind
            if self.conv_workers > 1 and self.backbone_provider_kind == "local":
                raise ValueError("conv_workers > 1 is not supported with local backbone providers.")
            if self.dataset_subset is not None:
                self.dataset_subset = self.dataset_subset.strip().lower()
                if not self.dataset_subset:
                    self.dataset_subset = None
            supported_subset_keys = DATASET_SUBSET_OPTIONS[self.dataset]()
            if self.dataset_subset is not None and self.dataset_subset not in supported_subset_keys:
                raise ValueError(
                    f"dataset_subset for {self.dataset.upper()} must be one of {supported_subset_keys}."
                )
            self.dataset_path = self.dataset_path.expanduser().resolve()
            self.output_dir = self.output_dir.expanduser().resolve()
            self.judge_max_concurrency = _normalize_judge_max_concurrency(self.judge_max_concurrency)
            self.memory_extract_batch_size = _normalize_memory_extract_batch_size(
                self.memory_extract_batch_size
            )
            uses_openai_compatible = (
                self.backbone_provider_kind == "openai-compatible"
                or self.judge_provider_kind == "openai-compatible"
            )
            if self.vllm_autostart and not uses_openai_compatible:
                raise ValueError("vllm_autostart requires an openai-compatible backbone or judge provider.")
            if uses_openai_compatible and self.openai_compatible_base_url is None:
                env_base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
                if env_base_url:
                    self.openai_compatible_base_url = env_base_url
                elif self.vllm_autostart:
                    self.openai_compatible_base_url = f"http://127.0.0.1:{self.vllm_port}/v1"
                else:
                    self.openai_compatible_base_url = "http://localhost:8000/v1"
            if uses_openai_compatible and self.openai_compatible_api_key is None:
                self.openai_compatible_api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY") or "EMPTY"
            if self.vllm_model is not None:
                self.vllm_model = self.vllm_model.strip() or None
            if self.vllm_served_model_name is not None:
                self.vllm_served_model_name = self.vllm_served_model_name.strip() or None
            self.vllm_host = str(self.vllm_host or "127.0.0.1").strip() or "127.0.0.1"
            self.openai_compatible_structured_mode = _normalize_openai_compatible_structured_mode(
                self.openai_compatible_structured_mode
            )
            self.cuda_preflight_mode = _normalize_cuda_preflight_mode(self.cuda_preflight_mode)
            self.retrieval_rank_save_mode = _normalize_retrieval_rank_save_mode(
                self.retrieval_rank_save_mode
            )
            self.cost_call_save_mode = _normalize_cost_call_save_mode(self.cost_call_save_mode)
            self.audit_packet_save_mode = _normalize_audit_packet_save_mode(
                self.audit_packet_save_mode
            )
            if self.database_path is not None:
                self.database_path = self.database_path.expanduser().resolve()
            if self.cost_price_config is not None:
                self.cost_price_config = self.cost_price_config.expanduser().resolve()
            if self.index_database_path is None:
                self.index_database_path = self.output_dir / "trajpatch_index.sqlite"
            self.index_database_path = self.index_database_path.expanduser().resolve()
            self.memory_cache_dir = self.memory_cache_dir.expanduser().resolve()
            return self

    else:

        @root_validator(skip_on_failure=True)
        def normalize_paths(cls, values: dict) -> dict:
            if values.get("backbone_provider_kind") is None:
                values["backbone_provider_kind"] = values.get("provider_kind", "mock")
            if values.get("judge_provider_kind") is None:
                values["judge_provider_kind"] = values["backbone_provider_kind"]
            values["provider_kind"] = values["backbone_provider_kind"]
            if int(values.get("conv_workers", 1) or 1) > 1 and values["backbone_provider_kind"] == "local":
                raise ValueError("conv_workers > 1 is not supported with local backbone providers.")
            dataset_subset = values.get("dataset_subset")
            if dataset_subset is not None:
                dataset_subset = str(dataset_subset).strip().lower()
                values["dataset_subset"] = dataset_subset or None
            dataset = values.get("dataset")
            supported_subset_keys = DATASET_SUBSET_OPTIONS[str(dataset)]()
            if values.get("dataset_subset") is not None and values["dataset_subset"] not in supported_subset_keys:
                raise ValueError(
                    f"dataset_subset for {str(dataset).upper()} must be one of {supported_subset_keys}."
                )
            values["dataset_path"] = values["dataset_path"].expanduser().resolve()
            values["output_dir"] = values["output_dir"].expanduser().resolve()
            values["judge_max_concurrency"] = _normalize_judge_max_concurrency(
                values.get("judge_max_concurrency", "auto")
            )
            values["memory_extract_batch_size"] = _normalize_memory_extract_batch_size(
                values.get("memory_extract_batch_size", "auto")
            )
            uses_openai_compatible = (
                values.get("backbone_provider_kind") == "openai-compatible"
                or values.get("judge_provider_kind") == "openai-compatible"
            )
            if values.get("vllm_autostart") and not uses_openai_compatible:
                raise ValueError("vllm_autostart requires an openai-compatible backbone or judge provider.")
            if uses_openai_compatible and values.get("openai_compatible_base_url") is None:
                env_base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
                if env_base_url:
                    values["openai_compatible_base_url"] = env_base_url
                elif values.get("vllm_autostart"):
                    values["openai_compatible_base_url"] = (
                        f"http://127.0.0.1:{int(values.get('vllm_port', 8000))}/v1"
                    )
                else:
                    values["openai_compatible_base_url"] = "http://localhost:8000/v1"
            if uses_openai_compatible and values.get("openai_compatible_api_key") is None:
                values["openai_compatible_api_key"] = os.getenv("OPENAI_COMPATIBLE_API_KEY") or "EMPTY"
            if values.get("vllm_model") is not None:
                values["vllm_model"] = str(values["vllm_model"]).strip() or None
            if values.get("vllm_served_model_name") is not None:
                values["vllm_served_model_name"] = str(values["vllm_served_model_name"]).strip() or None
            values["vllm_host"] = str(values.get("vllm_host") or "127.0.0.1").strip() or "127.0.0.1"
            values["openai_compatible_structured_mode"] = _normalize_openai_compatible_structured_mode(
                values.get("openai_compatible_structured_mode", "vllm")
            )
            values["cuda_preflight_mode"] = _normalize_cuda_preflight_mode(
                values.get("cuda_preflight_mode", "warn")
            )
            values["retrieval_rank_save_mode"] = _normalize_retrieval_rank_save_mode(
                values.get("retrieval_rank_save_mode", "top-n")
            )
            values["cost_call_save_mode"] = _normalize_cost_call_save_mode(
                values.get("cost_call_save_mode", "summary")
            )
            values["audit_packet_save_mode"] = _normalize_audit_packet_save_mode(
                values.get("audit_packet_save_mode", "summary")
            )
            if values.get("database_path") is not None:
                values["database_path"] = values["database_path"].expanduser().resolve()
            if values.get("cost_price_config") is not None:
                values["cost_price_config"] = values["cost_price_config"].expanduser().resolve()
            if values.get("index_database_path") is None:
                values["index_database_path"] = values["output_dir"] / "trajpatch_index.sqlite"
            values["index_database_path"] = values["index_database_path"].expanduser().resolve()
            values["memory_cache_dir"] = values["memory_cache_dir"].expanduser().resolve()
            return values
