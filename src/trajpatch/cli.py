"""Typer CLI entrypoint for TrajWiki."""

from __future__ import annotations

from pathlib import Path

try:
    import typer
except Exception:  # noqa: BLE001
    class _FallbackTyper:
        def __init__(self, *args, **kwargs):
            self.commands = []

        def command(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def __call__(self, *args, **kwargs):
            raise SystemExit(
                "Typer is not available in the current Python environment. "
                "Install project dependencies first, e.g. `python -m pip install -e .`."
            )

    class _FallbackModule:
        Typer = _FallbackTyper
        BadParameter = ValueError

        @staticmethod
        def Option(default=..., *args, **kwargs):
            return default

        @staticmethod
        def Argument(default=..., *args, **kwargs):
            return default

    typer = _FallbackModule()

from rich.console import Console

from trajpatch.analysis import (
    analyze_auditability,
    analyze_cost_benefit,
    analyze_locomo_run_failures,
    analyze_offline_ablation,
    diff_locomo_failure_reports,
    load_incomplete_run_diagnostics,
    print_incomplete_run_diagnostics,
    print_locomo_failure_diff,
    print_locomo_failure_report,
)
from trajpatch.config import RunConfig
from trajpatch.exceptions import ProviderConfigurationError
from trajpatch.experiments import (
    analyze_answer_ablation,
    analyze_audit_study,
    analyze_ranking_robustness,
    conduct_audit_study,
    import_baseline_answers,
    package_rebuttal_bundle,
    prepare_audit_study,
    run_answer_ablation,
    validate_rebuttal_bundle,
    validate_run_artifacts,
)
from trajpatch.pipeline.exporter import ArtifactExporter
from trajpatch.pipeline.inspector import Inspector
from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.pipeline.sweep import run_grid
from trajpatch.reporting import SQLiteReportReader
from trajpatch.storage.database import create_schema
from trajpatch.storage.repository import TrajWikiStore
from trajpatch.utils.json_utils import write_json

app = typer.Typer(no_args_is_help=True)
console = Console()


def _run_benchmark(
    *,
    dataset: str,
    dataset_subset: str | None,
    dataset_path: Path,
    output_dir: Path,
    database_path: Path | None,
    index_database_path: Path,
    memory_cache_enabled: bool,
    memory_cache_dir: Path,
    rebuild_memory_cache: bool,
    rebuild_semantic_metric_cache: bool,
    reset_run_db: bool,
    provider_kind: str | None,
    backbone_provider_kind: str,
    judge_provider_kind: str | None,
    backbone_model: str,
    embedding_model: str,
    judge_model: str,
    device_mode: str,
    m: int,
    t_pages: int,
    k: int,
    neighbor_radius: int,
    retrieval_expansion_mode: str,
    max_samples: int | None,
    conv_workers: int,
    judge_max_concurrency: str,
    memory_extract_batch_size: str,
    openai_compatible_base_url: str | None,
    openai_compatible_api_key: str | None,
    openai_compatible_structured_mode: str,
    vllm_autostart: bool,
    vllm_model: str | None,
    vllm_served_model_name: str | None,
    vllm_host: str,
    vllm_port: int,
    vllm_cuda_visible_devices: str | None,
    vllm_tensor_parallel_size: int,
    vllm_gpu_memory_utilization: float | None,
    vllm_dtype: str | None,
    vllm_extra_args: str | None,
    vllm_startup_timeout_s: int,
    vllm_keep_alive: bool,
    cuda_preflight_mode: str,
    cuda_preflight_reserve_gb: float,
    cuda_preflight_report: bool,
    ablation_diagnostics: bool,
    retrieval_rank_save_mode: str,
    retrieval_rank_save_limit: int,
    offline_context_budgets: str,
    offline_rank_cutoffs: str,
    cost_diagnostics: bool,
    cost_call_save_mode: str,
    cost_price_config: Path | None,
    future_query_counts: str,
    auditability_diagnostics: bool,
    audit_packet_save_mode: str,
    verbose: bool,
) -> None:
    chosen_backbone_provider = provider_kind or backbone_provider_kind
    chosen_judge_provider = judge_provider_kind or chosen_backbone_provider
    config = RunConfig(
        dataset=dataset,
        dataset_subset=dataset_subset,
        dataset_path=dataset_path,
        output_dir=output_dir,
        database_path=database_path,
        index_database_path=index_database_path,
        memory_cache_enabled=memory_cache_enabled,
        memory_cache_dir=memory_cache_dir,
        rebuild_memory_cache=rebuild_memory_cache,
        rebuild_semantic_metric_cache=rebuild_semantic_metric_cache,
        reset_run_db=reset_run_db,
        provider_kind=chosen_backbone_provider,
        backbone_provider_kind=chosen_backbone_provider,
        judge_provider_kind=chosen_judge_provider,
        backbone_model=backbone_model,
        embedding_model=embedding_model,
        judge_model=judge_model,
        device_mode=device_mode,
        m=m,
        t_pages=t_pages,
        k=k,
        neighbor_radius=neighbor_radius,
        retrieval_expansion_mode=retrieval_expansion_mode,
        max_samples=max_samples,
        conv_workers=conv_workers,
        judge_max_concurrency=judge_max_concurrency,
        memory_extract_batch_size=memory_extract_batch_size,
        openai_compatible_base_url=openai_compatible_base_url,
        openai_compatible_api_key=openai_compatible_api_key,
        openai_compatible_structured_mode=openai_compatible_structured_mode,
        vllm_autostart=vllm_autostart,
        vllm_model=vllm_model,
        vllm_served_model_name=vllm_served_model_name,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        vllm_cuda_visible_devices=vllm_cuda_visible_devices,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_dtype=vllm_dtype,
        vllm_extra_args=vllm_extra_args,
        vllm_startup_timeout_s=vllm_startup_timeout_s,
        vllm_keep_alive=vllm_keep_alive,
        cuda_preflight_mode=cuda_preflight_mode,
        cuda_preflight_reserve_gb=cuda_preflight_reserve_gb,
        cuda_preflight_report=cuda_preflight_report,
        ablation_diagnostics=ablation_diagnostics,
        retrieval_rank_save_mode=retrieval_rank_save_mode,
        retrieval_rank_save_limit=retrieval_rank_save_limit,
        offline_context_budgets=offline_context_budgets,
        offline_rank_cutoffs=offline_rank_cutoffs,
        cost_diagnostics=cost_diagnostics,
        cost_call_save_mode=cost_call_save_mode,
        cost_price_config=cost_price_config,
        future_query_counts=future_query_counts,
        auditability_diagnostics=auditability_diagnostics,
        audit_packet_save_mode=audit_packet_save_mode,
        verbose=verbose,
    )
    report = PipelineRunner(config, console=console).run()
    summary_path = report.details["paths"]["summary"]
    console.print(f"Summary written to {summary_path}")


@app.command()
def run(
    dataset: str = typer.Option(..., help="Dataset name: locomo or medmt."),
    subset: str | None = typer.Option(
        None,
        "--subset",
        help=(
            "Optional dataset subset. LOCOMO: all, multi_hop, temporal, open_domain, single_hop. "
            "MedMT: all, long_context_memory_and_understanding, "
            "resistance_to_contextual_interference, information_contradiction."
        ),
    ),
    dataset_path: Path = typer.Option(..., exists=True, help="Path to a benchmark file or dataset root."),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for benchmark artifacts."),
    database_path: Path | None = typer.Option(
        None, help="Optional override for the per-run SQLite path."
    ),
    index_database_path: Path = typer.Option(
        Path("output/trajpatch_index.sqlite"), help="Global SQLite index for benchmark summaries."
    ),
    memory_cache_enabled: bool = typer.Option(
        True, "--memory-cache/--no-memory-cache", help="Enable reusable memory-build cache."
    ),
    memory_cache_dir: Path = typer.Option(
        Path(".trajpatch_cache"), help="Shared cache directory for reusable memory bundles."
    ),
    rebuild_memory_cache: bool = typer.Option(
        False, help="Ignore existing memory cache and rebuild cache entries."
    ),
    rebuild_semantic_metric_cache: bool = typer.Option(
        False,
        "--rebuild-semantic-metric-cache/--no-rebuild-semantic-metric-cache",
        help="Ignore existing semantic metric cache and regenerate F1/BLEU-1 extraction entries.",
    ),
    reset_run_db: bool = typer.Option(
        True, "--reset-run-db/--no-reset-run-db", help="Recreate the run SQLite database before execution."
    ),
    provider_kind: str | None = typer.Option(
        None, help="Deprecated alias for --backbone-provider-kind."
    ),
    backbone_provider_kind: str = typer.Option(
        "mock", help="mock, remote, local, or openai-compatible for the answering backbone."
    ),
    judge_provider_kind: str | None = typer.Option(
        None,
        help="mock, remote, local, or openai-compatible for the benchmark judge. Defaults to backbone provider kind.",
    ),
    backbone_model: str = typer.Option("mock-backbone", help="Backbone LLM name."),
    embedding_model: str = typer.Option(
        "huggingface/Qwen3-Embedding-8B", help="Embedding model name."
    ),
    judge_model: str = typer.Option("mock-judge", help="Judge model name."),
    device_mode: str = typer.Option("auto", help="auto, cpu, single, or multi."),
    m: int = typer.Option(6, help="Maximum episodic trajectory length."),
    t_pages: int = typer.Option(5, help="Top-T wiki pages."),
    k: int = typer.Option(3, help="Top-k candidate trajectories."),
    neighbor_radius: int = typer.Option(1, help="Trajectory neighbor snapshot expansion radius."),
    retrieval_expansion_mode: str = typer.Option(
        "update_linked_plus_neighbors",
        help=(
            "Episodic retrieval expansion mode: update_linked_plus_neighbors, "
            "neighbors_only, or none."
        ),
    ),
    max_samples: int | None = typer.Option(None, help="Optional sample limit for debugging."),
    conv_workers: int = typer.Option(
        1,
        help="Logical sample workers. LOCOMO uses conversation workers; MedMT uses sample workers.",
    ),
    judge_max_concurrency: str = typer.Option("auto", help="Judge concurrency: auto or integer >= 1."),
    memory_extract_batch_size: str = typer.Option(
        "auto",
        help="Memory extraction stage batch/concurrency: auto or integer >= 1.",
    ),
    openai_compatible_base_url: str | None = typer.Option(
        None,
        "--openai-compatible-base-url",
        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1. Defaults to OPENAI_COMPATIBLE_BASE_URL.",
    ),
    openai_compatible_api_key: str | None = typer.Option(
        None,
        "--openai-compatible-api-key",
        help="OpenAI-compatible API key. vLLM commonly accepts EMPTY. Defaults to OPENAI_COMPATIBLE_API_KEY.",
    ),
    openai_compatible_structured_mode: str = typer.Option(
        "vllm",
        "--openai-compatible-structured-mode",
        help="Structured output mode for openai-compatible providers: vllm, openai_json_schema, or text_json.",
    ),
    vllm_autostart: bool = typer.Option(
        False,
        "--vllm-autostart/--no-vllm-autostart",
        help="Start a local vLLM OpenAI-compatible server before benchmark execution.",
    ),
    vllm_model: str | None = typer.Option(None, "--vllm-model", help="Model passed to `vllm serve`."),
    vllm_served_model_name: str | None = typer.Option(
        None,
        "--vllm-served-model-name",
        help="Served model name exposed by vLLM. Defaults to the backbone model name.",
    ),
    vllm_host: str = typer.Option("127.0.0.1", "--vllm-host", help="Host passed to `vllm serve`."),
    vllm_port: int = typer.Option(8000, "--vllm-port", help="Port passed to `vllm serve`."),
    vllm_cuda_visible_devices: str | None = typer.Option(
        None,
        "--vllm-cuda-visible-devices",
        help="CUDA_VISIBLE_DEVICES value for the vLLM subprocess only.",
    ),
    vllm_tensor_parallel_size: int = typer.Option(
        1,
        "--vllm-tensor-parallel-size",
        help="Optional vLLM tensor parallel size.",
    ),
    vllm_gpu_memory_utilization: float | None = typer.Option(
        None,
        "--vllm-gpu-memory-utilization",
        help="Optional vLLM GPU memory utilization value.",
    ),
    vllm_dtype: str | None = typer.Option(None, "--vllm-dtype", help="Optional vLLM dtype."),
    vllm_extra_args: str | None = typer.Option(
        None,
        "--vllm-extra-args",
        help="Additional arguments appended to `vllm serve`, parsed with shell-like splitting.",
    ),
    vllm_startup_timeout_s: int = typer.Option(
        600,
        "--vllm-startup-timeout-s",
        help="Seconds to wait for the vLLM /v1/models endpoint.",
    ),
    vllm_keep_alive: bool = typer.Option(
        False,
        "--vllm-keep-alive/--no-vllm-keep-alive",
        help="Keep an autostarted vLLM process alive after the benchmark exits.",
    ),
    cuda_preflight_mode: str = typer.Option(
        "warn",
        "--cuda-preflight-mode",
        help="CUDA preflight behavior: off, warn, or strict.",
    ),
    cuda_preflight_reserve_gb: float = typer.Option(
        2.0,
        "--cuda-preflight-reserve-gb",
        help="Per-GPU safety reserve in GiB for CUDA preflight planning.",
    ),
    cuda_preflight_report: bool = typer.Option(
        True,
        "--cuda-preflight-report/--no-cuda-preflight-report",
        help="Write status/cuda_preflight.json.",
    ),
    ablation_diagnostics: bool = typer.Option(
        False,
        "--ablation-diagnostics/--no-ablation-diagnostics",
        help="Write extra retrieval/gold-label diagnostics for offline ablation analysis.",
    ),
    retrieval_rank_save_mode: str = typer.Option(
        "top-n",
        "--retrieval-rank-save-mode",
        help="Retrieval diagnostic row saving: top-n or full.",
    ),
    retrieval_rank_save_limit: int = typer.Option(
        100,
        "--retrieval-rank-save-limit",
        help="Top-N page/trajectory ranked rows to save when retrieval rank save mode is top-n.",
    ),
    offline_context_budgets: str = typer.Option(
        "4000,8000,16000,32000",
        "--offline-context-budgets",
        help="Comma-separated context budgets recorded in run metadata for offline ablations.",
    ),
    offline_rank_cutoffs: str = typer.Option(
        "5,10,15,20,30,50",
        "--offline-rank-cutoffs",
        help="Comma-separated rank cutoffs recorded in run metadata for offline ablations.",
    ),
    cost_diagnostics: bool = typer.Option(
        False,
        "--cost-diagnostics/--no-cost-diagnostics",
        help="Write extra cost-benefit diagnostics for offline analysis.",
    ),
    cost_call_save_mode: str = typer.Option(
        "summary",
        "--cost-call-save-mode",
        help="Cost call row saving: summary or compact.",
    ),
    cost_price_config: Path | None = typer.Option(
        None,
        "--cost-price-config",
        help="Optional LLM pricing JSON used by offline cost-benefit analysis.",
    ),
    future_query_counts: str = typer.Option(
        "1,2,5,10,20,50,100",
        "--future-query-counts",
        help="Comma-separated future query counts recorded in run metadata.",
    ),
    auditability_diagnostics: bool = typer.Option(
        False,
        "--auditability-diagnostics/--no-auditability-diagnostics",
        help="Write extra provenance/auditability diagnostics for offline analysis.",
    ),
    audit_packet_save_mode: str = typer.Option(
        "summary",
        "--audit-packet-save-mode",
        help="Audit packet row saving: summary or compact.",
    ),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Print step-level timing logs."),
) -> None:
    _run_benchmark(
        dataset=dataset,
        dataset_subset=subset,
        dataset_path=dataset_path,
        output_dir=output_dir,
        database_path=database_path,
        index_database_path=index_database_path,
        memory_cache_enabled=memory_cache_enabled,
        memory_cache_dir=memory_cache_dir,
        rebuild_memory_cache=rebuild_memory_cache,
        rebuild_semantic_metric_cache=rebuild_semantic_metric_cache,
        reset_run_db=reset_run_db,
        provider_kind=provider_kind,
        backbone_provider_kind=backbone_provider_kind,
        judge_provider_kind=judge_provider_kind,
        backbone_model=backbone_model,
        embedding_model=embedding_model,
        judge_model=judge_model,
        device_mode=device_mode,
        m=m,
        t_pages=t_pages,
        k=k,
        neighbor_radius=neighbor_radius,
        retrieval_expansion_mode=retrieval_expansion_mode,
        max_samples=max_samples,
        conv_workers=conv_workers,
        judge_max_concurrency=judge_max_concurrency,
        memory_extract_batch_size=memory_extract_batch_size,
        openai_compatible_base_url=openai_compatible_base_url,
        openai_compatible_api_key=openai_compatible_api_key,
        openai_compatible_structured_mode=openai_compatible_structured_mode,
        vllm_autostart=vllm_autostart,
        vllm_model=vllm_model,
        vllm_served_model_name=vllm_served_model_name,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        vllm_cuda_visible_devices=vllm_cuda_visible_devices,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_dtype=vllm_dtype,
        vllm_extra_args=vllm_extra_args,
        vllm_startup_timeout_s=vllm_startup_timeout_s,
        vllm_keep_alive=vllm_keep_alive,
        cuda_preflight_mode=cuda_preflight_mode,
        cuda_preflight_reserve_gb=cuda_preflight_reserve_gb,
        cuda_preflight_report=cuda_preflight_report,
        ablation_diagnostics=ablation_diagnostics,
        retrieval_rank_save_mode=retrieval_rank_save_mode,
        retrieval_rank_save_limit=retrieval_rank_save_limit,
        offline_context_budgets=offline_context_budgets,
        offline_rank_cutoffs=offline_rank_cutoffs,
        cost_diagnostics=cost_diagnostics,
        cost_call_save_mode=cost_call_save_mode,
        cost_price_config=cost_price_config,
        future_query_counts=future_query_counts,
        auditability_diagnostics=auditability_diagnostics,
        audit_packet_save_mode=audit_packet_save_mode,
        verbose=verbose,
    )


@app.command(name="benchmark-locomo")
def benchmark_locomo(
    subset: str | None = typer.Option(
        None,
        "--subset",
        help="Optional LOCOMO subset: all, multi_hop, temporal, open_domain, or single_hop.",
    ),
    dataset_path: Path = typer.Option(..., exists=True, help="Path to LOCOMO all_qa.jsonl or dataset root."),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for benchmark artifacts."),
    database_path: Path | None = typer.Option(None, help="Optional override for the per-run SQLite path."),
    index_database_path: Path = typer.Option(
        Path("output/trajpatch_index.sqlite"), help="Global SQLite index for benchmark summaries."
    ),
    memory_cache_enabled: bool = typer.Option(True, "--memory-cache/--no-memory-cache"),
    memory_cache_dir: Path = typer.Option(Path(".trajpatch_cache")),
    rebuild_memory_cache: bool = typer.Option(False),
    rebuild_semantic_metric_cache: bool = typer.Option(
        False,
        "--rebuild-semantic-metric-cache/--no-rebuild-semantic-metric-cache",
        help="Ignore existing semantic metric cache and regenerate F1/BLEU-1 extraction entries.",
    ),
    reset_run_db: bool = typer.Option(True, "--reset-run-db/--no-reset-run-db"),
    provider_kind: str | None = typer.Option(None),
    backbone_provider_kind: str = typer.Option("mock"),
    judge_provider_kind: str | None = typer.Option(None),
    backbone_model: str = typer.Option("mock-backbone"),
    embedding_model: str = typer.Option("huggingface/Qwen3-Embedding-8B"),
    judge_model: str = typer.Option("mock-judge"),
    device_mode: str = typer.Option("auto"),
    m: int = typer.Option(6),
    t_pages: int = typer.Option(5),
    k: int = typer.Option(3),
    neighbor_radius: int = typer.Option(1),
    retrieval_expansion_mode: str = typer.Option("update_linked_plus_neighbors"),
    max_samples: int | None = typer.Option(None),
    conv_workers: int = typer.Option(1),
    judge_max_concurrency: str = typer.Option("auto"),
    memory_extract_batch_size: str = typer.Option("auto"),
    openai_compatible_base_url: str | None = typer.Option(
        None,
        "--openai-compatible-base-url",
        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1.",
    ),
    openai_compatible_api_key: str | None = typer.Option(
        None,
        "--openai-compatible-api-key",
        help="OpenAI-compatible API key. vLLM commonly accepts EMPTY.",
    ),
    openai_compatible_structured_mode: str = typer.Option(
        "vllm",
        "--openai-compatible-structured-mode",
        help="Structured output mode: vllm, openai_json_schema, or text_json.",
    ),
    vllm_autostart: bool = typer.Option(False, "--vllm-autostart/--no-vllm-autostart"),
    vllm_model: str | None = typer.Option(None, "--vllm-model"),
    vllm_served_model_name: str | None = typer.Option(None, "--vllm-served-model-name"),
    vllm_host: str = typer.Option("127.0.0.1", "--vllm-host"),
    vllm_port: int = typer.Option(8000, "--vllm-port"),
    vllm_cuda_visible_devices: str | None = typer.Option(None, "--vllm-cuda-visible-devices"),
    vllm_tensor_parallel_size: int = typer.Option(1, "--vllm-tensor-parallel-size"),
    vllm_gpu_memory_utilization: float | None = typer.Option(None, "--vllm-gpu-memory-utilization"),
    vllm_dtype: str | None = typer.Option(None, "--vllm-dtype"),
    vllm_extra_args: str | None = typer.Option(None, "--vllm-extra-args"),
    vllm_startup_timeout_s: int = typer.Option(600, "--vllm-startup-timeout-s"),
    vllm_keep_alive: bool = typer.Option(False, "--vllm-keep-alive/--no-vllm-keep-alive"),
    cuda_preflight_mode: str = typer.Option("warn", "--cuda-preflight-mode"),
    cuda_preflight_reserve_gb: float = typer.Option(2.0, "--cuda-preflight-reserve-gb"),
    cuda_preflight_report: bool = typer.Option(
        True, "--cuda-preflight-report/--no-cuda-preflight-report"
    ),
    ablation_diagnostics: bool = typer.Option(False, "--ablation-diagnostics/--no-ablation-diagnostics"),
    retrieval_rank_save_mode: str = typer.Option("top-n", "--retrieval-rank-save-mode"),
    retrieval_rank_save_limit: int = typer.Option(100, "--retrieval-rank-save-limit"),
    offline_context_budgets: str = typer.Option("4000,8000,16000,32000", "--offline-context-budgets"),
    offline_rank_cutoffs: str = typer.Option("5,10,15,20,30,50", "--offline-rank-cutoffs"),
    cost_diagnostics: bool = typer.Option(False, "--cost-diagnostics/--no-cost-diagnostics"),
    cost_call_save_mode: str = typer.Option("summary", "--cost-call-save-mode"),
    cost_price_config: Path | None = typer.Option(None, "--cost-price-config"),
    future_query_counts: str = typer.Option("1,2,5,10,20,50,100", "--future-query-counts"),
    auditability_diagnostics: bool = typer.Option(False, "--auditability-diagnostics/--no-auditability-diagnostics"),
    audit_packet_save_mode: str = typer.Option("summary", "--audit-packet-save-mode"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose"),
) -> None:
    _run_benchmark(
        dataset="locomo",
        dataset_subset=subset,
        dataset_path=dataset_path,
        output_dir=output_dir,
        database_path=database_path,
        index_database_path=index_database_path,
        memory_cache_enabled=memory_cache_enabled,
        memory_cache_dir=memory_cache_dir,
        rebuild_memory_cache=rebuild_memory_cache,
        rebuild_semantic_metric_cache=rebuild_semantic_metric_cache,
        reset_run_db=reset_run_db,
        provider_kind=provider_kind,
        backbone_provider_kind=backbone_provider_kind,
        judge_provider_kind=judge_provider_kind,
        backbone_model=backbone_model,
        embedding_model=embedding_model,
        judge_model=judge_model,
        device_mode=device_mode,
        m=m,
        t_pages=t_pages,
        k=k,
        neighbor_radius=neighbor_radius,
        retrieval_expansion_mode=retrieval_expansion_mode,
        max_samples=max_samples,
        conv_workers=conv_workers,
        judge_max_concurrency=judge_max_concurrency,
        memory_extract_batch_size=memory_extract_batch_size,
        openai_compatible_base_url=openai_compatible_base_url,
        openai_compatible_api_key=openai_compatible_api_key,
        openai_compatible_structured_mode=openai_compatible_structured_mode,
        vllm_autostart=vllm_autostart,
        vllm_model=vllm_model,
        vllm_served_model_name=vllm_served_model_name,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        vllm_cuda_visible_devices=vllm_cuda_visible_devices,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_dtype=vllm_dtype,
        vllm_extra_args=vllm_extra_args,
        vllm_startup_timeout_s=vllm_startup_timeout_s,
        vllm_keep_alive=vllm_keep_alive,
        cuda_preflight_mode=cuda_preflight_mode,
        cuda_preflight_reserve_gb=cuda_preflight_reserve_gb,
        cuda_preflight_report=cuda_preflight_report,
        ablation_diagnostics=ablation_diagnostics,
        retrieval_rank_save_mode=retrieval_rank_save_mode,
        retrieval_rank_save_limit=retrieval_rank_save_limit,
        offline_context_budgets=offline_context_budgets,
        offline_rank_cutoffs=offline_rank_cutoffs,
        cost_diagnostics=cost_diagnostics,
        cost_call_save_mode=cost_call_save_mode,
        cost_price_config=cost_price_config,
        future_query_counts=future_query_counts,
        auditability_diagnostics=auditability_diagnostics,
        audit_packet_save_mode=audit_packet_save_mode,
        verbose=verbose,
    )


@app.command(name="benchmark-medmt")
def benchmark_medmt(
    subset: str | None = typer.Option(
        None,
        "--subset",
        help=(
            "Optional MedMT subset: all, long_context_memory_and_understanding, "
            "resistance_to_contextual_interference, or information_contradiction."
        ),
    ),
    dataset_path: Path = typer.Option(..., exists=True, help="Path to MedMT dataset root or one selected file."),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for benchmark artifacts."),
    database_path: Path | None = typer.Option(None, help="Optional override for the per-run SQLite path."),
    index_database_path: Path = typer.Option(
        Path("output/trajpatch_index.sqlite"), help="Global SQLite index for benchmark summaries."
    ),
    memory_cache_enabled: bool = typer.Option(True, "--memory-cache/--no-memory-cache"),
    memory_cache_dir: Path = typer.Option(Path(".trajpatch_cache")),
    rebuild_memory_cache: bool = typer.Option(False),
    rebuild_semantic_metric_cache: bool = typer.Option(
        False,
        "--rebuild-semantic-metric-cache/--no-rebuild-semantic-metric-cache",
        help="Ignore existing semantic metric cache and regenerate F1/BLEU-1 extraction entries.",
    ),
    reset_run_db: bool = typer.Option(True, "--reset-run-db/--no-reset-run-db"),
    provider_kind: str | None = typer.Option(None),
    backbone_provider_kind: str = typer.Option("mock"),
    judge_provider_kind: str | None = typer.Option(None),
    backbone_model: str = typer.Option("mock-backbone"),
    embedding_model: str = typer.Option("huggingface/Qwen3-Embedding-8B"),
    judge_model: str = typer.Option("mock-judge"),
    device_mode: str = typer.Option("auto"),
    m: int = typer.Option(6),
    t_pages: int = typer.Option(5),
    k: int = typer.Option(3),
    neighbor_radius: int = typer.Option(1),
    retrieval_expansion_mode: str = typer.Option("update_linked_plus_neighbors"),
    max_samples: int | None = typer.Option(None),
    conv_workers: int = typer.Option(1),
    judge_max_concurrency: str = typer.Option("auto"),
    memory_extract_batch_size: str = typer.Option("auto"),
    openai_compatible_base_url: str | None = typer.Option(
        None,
        "--openai-compatible-base-url",
        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1.",
    ),
    openai_compatible_api_key: str | None = typer.Option(
        None,
        "--openai-compatible-api-key",
        help="OpenAI-compatible API key. vLLM commonly accepts EMPTY.",
    ),
    openai_compatible_structured_mode: str = typer.Option(
        "vllm",
        "--openai-compatible-structured-mode",
        help="Structured output mode: vllm, openai_json_schema, or text_json.",
    ),
    vllm_autostart: bool = typer.Option(False, "--vllm-autostart/--no-vllm-autostart"),
    vllm_model: str | None = typer.Option(None, "--vllm-model"),
    vllm_served_model_name: str | None = typer.Option(None, "--vllm-served-model-name"),
    vllm_host: str = typer.Option("127.0.0.1", "--vllm-host"),
    vllm_port: int = typer.Option(8000, "--vllm-port"),
    vllm_cuda_visible_devices: str | None = typer.Option(None, "--vllm-cuda-visible-devices"),
    vllm_tensor_parallel_size: int = typer.Option(1, "--vllm-tensor-parallel-size"),
    vllm_gpu_memory_utilization: float | None = typer.Option(None, "--vllm-gpu-memory-utilization"),
    vllm_dtype: str | None = typer.Option(None, "--vllm-dtype"),
    vllm_extra_args: str | None = typer.Option(None, "--vllm-extra-args"),
    vllm_startup_timeout_s: int = typer.Option(600, "--vllm-startup-timeout-s"),
    vllm_keep_alive: bool = typer.Option(False, "--vllm-keep-alive/--no-vllm-keep-alive"),
    cuda_preflight_mode: str = typer.Option("warn", "--cuda-preflight-mode"),
    cuda_preflight_reserve_gb: float = typer.Option(2.0, "--cuda-preflight-reserve-gb"),
    cuda_preflight_report: bool = typer.Option(
        True, "--cuda-preflight-report/--no-cuda-preflight-report"
    ),
    ablation_diagnostics: bool = typer.Option(False, "--ablation-diagnostics/--no-ablation-diagnostics"),
    retrieval_rank_save_mode: str = typer.Option("top-n", "--retrieval-rank-save-mode"),
    retrieval_rank_save_limit: int = typer.Option(100, "--retrieval-rank-save-limit"),
    offline_context_budgets: str = typer.Option("4000,8000,16000,32000", "--offline-context-budgets"),
    offline_rank_cutoffs: str = typer.Option("5,10,15,20,30,50", "--offline-rank-cutoffs"),
    cost_diagnostics: bool = typer.Option(False, "--cost-diagnostics/--no-cost-diagnostics"),
    cost_call_save_mode: str = typer.Option("summary", "--cost-call-save-mode"),
    cost_price_config: Path | None = typer.Option(None, "--cost-price-config"),
    future_query_counts: str = typer.Option("1,2,5,10,20,50,100", "--future-query-counts"),
    auditability_diagnostics: bool = typer.Option(False, "--auditability-diagnostics/--no-auditability-diagnostics"),
    audit_packet_save_mode: str = typer.Option("summary", "--audit-packet-save-mode"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose"),
) -> None:
    _run_benchmark(
        dataset="medmt",
        dataset_subset=subset,
        dataset_path=dataset_path,
        output_dir=output_dir,
        database_path=database_path,
        index_database_path=index_database_path,
        memory_cache_enabled=memory_cache_enabled,
        memory_cache_dir=memory_cache_dir,
        rebuild_memory_cache=rebuild_memory_cache,
        rebuild_semantic_metric_cache=rebuild_semantic_metric_cache,
        reset_run_db=reset_run_db,
        provider_kind=provider_kind,
        backbone_provider_kind=backbone_provider_kind,
        judge_provider_kind=judge_provider_kind,
        backbone_model=backbone_model,
        embedding_model=embedding_model,
        judge_model=judge_model,
        device_mode=device_mode,
        m=m,
        t_pages=t_pages,
        k=k,
        neighbor_radius=neighbor_radius,
        retrieval_expansion_mode=retrieval_expansion_mode,
        max_samples=max_samples,
        conv_workers=conv_workers,
        judge_max_concurrency=judge_max_concurrency,
        memory_extract_batch_size=memory_extract_batch_size,
        openai_compatible_base_url=openai_compatible_base_url,
        openai_compatible_api_key=openai_compatible_api_key,
        openai_compatible_structured_mode=openai_compatible_structured_mode,
        vllm_autostart=vllm_autostart,
        vllm_model=vllm_model,
        vllm_served_model_name=vllm_served_model_name,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        vllm_cuda_visible_devices=vllm_cuda_visible_devices,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_dtype=vllm_dtype,
        vllm_extra_args=vllm_extra_args,
        vllm_startup_timeout_s=vllm_startup_timeout_s,
        vllm_keep_alive=vllm_keep_alive,
        cuda_preflight_mode=cuda_preflight_mode,
        cuda_preflight_reserve_gb=cuda_preflight_reserve_gb,
        cuda_preflight_report=cuda_preflight_report,
        ablation_diagnostics=ablation_diagnostics,
        retrieval_rank_save_mode=retrieval_rank_save_mode,
        retrieval_rank_save_limit=retrieval_rank_save_limit,
        offline_context_budgets=offline_context_budgets,
        offline_rank_cutoffs=offline_rank_cutoffs,
        cost_diagnostics=cost_diagnostics,
        cost_call_save_mode=cost_call_save_mode,
        cost_price_config=cost_price_config,
        future_query_counts=future_query_counts,
        auditability_diagnostics=auditability_diagnostics,
        audit_packet_save_mode=audit_packet_save_mode,
        verbose=verbose,
    )


@app.command()
def sweep(
    dataset: str = typer.Option(...),
    dataset_path: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("output/sweeps")),
    index_database_path: Path = typer.Option(Path("output/trajpatch_index.sqlite")),
    provider_kind: str | None = typer.Option(None),
    backbone_provider_kind: str = typer.Option("mock"),
    judge_provider_kind: str | None = typer.Option(None),
    backbone_model: str = typer.Option("mock-backbone"),
    embedding_model: str = typer.Option("huggingface/Qwen3-Embedding-8B"),
    judge_model: str = typer.Option("mock-judge"),
    openai_compatible_base_url: str | None = typer.Option(None, "--openai-compatible-base-url"),
    openai_compatible_api_key: str | None = typer.Option(None, "--openai-compatible-api-key"),
    openai_compatible_structured_mode: str = typer.Option("vllm", "--openai-compatible-structured-mode"),
    vllm_autostart: bool = typer.Option(False, "--vllm-autostart/--no-vllm-autostart"),
    vllm_model: str | None = typer.Option(None, "--vllm-model"),
    vllm_served_model_name: str | None = typer.Option(None, "--vllm-served-model-name"),
    vllm_host: str = typer.Option("127.0.0.1", "--vllm-host"),
    vllm_port: int = typer.Option(8000, "--vllm-port"),
    vllm_cuda_visible_devices: str | None = typer.Option(None, "--vllm-cuda-visible-devices"),
    vllm_tensor_parallel_size: int = typer.Option(1, "--vllm-tensor-parallel-size"),
    vllm_gpu_memory_utilization: float | None = typer.Option(None, "--vllm-gpu-memory-utilization"),
    vllm_dtype: str | None = typer.Option(None, "--vllm-dtype"),
    vllm_extra_args: str | None = typer.Option(None, "--vllm-extra-args"),
    vllm_startup_timeout_s: int = typer.Option(600, "--vllm-startup-timeout-s"),
    vllm_keep_alive: bool = typer.Option(False, "--vllm-keep-alive/--no-vllm-keep-alive"),
    cuda_preflight_mode: str = typer.Option("warn", "--cuda-preflight-mode"),
    cuda_preflight_reserve_gb: float = typer.Option(2.0, "--cuda-preflight-reserve-gb"),
    cuda_preflight_report: bool = typer.Option(
        True, "--cuda-preflight-report/--no-cuda-preflight-report"
    ),
    m_values: str = typer.Option("4,6"),
    k_values: str = typer.Option("2,3"),
    t_page_values: str = typer.Option("3,5"),
    neighbor_radius_values: str = typer.Option("0,1"),
    retrieval_expansion_mode_values: str = typer.Option("update_linked_plus_neighbors"),
) -> None:
    chosen_backbone_provider = provider_kind or backbone_provider_kind
    chosen_judge_provider = judge_provider_kind or chosen_backbone_provider
    base_config = RunConfig(
        dataset=dataset,
        dataset_path=dataset_path,
        output_dir=output_dir,
        database_path=None,
        index_database_path=index_database_path,
        provider_kind=chosen_backbone_provider,
        backbone_provider_kind=chosen_backbone_provider,
        judge_provider_kind=chosen_judge_provider,
        backbone_model=backbone_model,
        embedding_model=embedding_model,
        judge_model=judge_model,
        openai_compatible_base_url=openai_compatible_base_url,
        openai_compatible_api_key=openai_compatible_api_key,
        openai_compatible_structured_mode=openai_compatible_structured_mode,
        vllm_autostart=vllm_autostart,
        vllm_model=vllm_model,
        vllm_served_model_name=vllm_served_model_name,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        vllm_cuda_visible_devices=vllm_cuda_visible_devices,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_dtype=vllm_dtype,
        vllm_extra_args=vllm_extra_args,
        vllm_startup_timeout_s=vllm_startup_timeout_s,
        vllm_keep_alive=vllm_keep_alive,
        cuda_preflight_mode=cuda_preflight_mode,
        cuda_preflight_reserve_gb=cuda_preflight_reserve_gb,
        cuda_preflight_report=cuda_preflight_report,
        retrieval_expansion_mode="update_linked_plus_neighbors",
    )
    reports = run_grid(
        base_config,
        m_values=[int(value) for value in m_values.split(",") if value],
        k_values=[int(value) for value in k_values.split(",") if value],
        t_page_values=[int(value) for value in t_page_values.split(",") if value],
        neighbor_radius_values=[int(value) for value in neighbor_radius_values.split(",") if value],
        retrieval_expansion_mode_values=[
            value.strip() for value in retrieval_expansion_mode_values.split(",") if value.strip()
        ],
    )
    write_json(output_dir / "sweep_summary.json", reports)
    console.print_json(data={"runs": reports})


@app.command()
def report(
    database_path: Path | None = typer.Option(None, help="Optional single-run SQLite path."),
    index_database_path: Path = typer.Option(
        Path("output/trajpatch_index.sqlite"), help="Cross-run SQLite index path."
    ),
    dataset: str | None = typer.Option(None),
    subset: str | None = typer.Option(
        None,
        "--subset",
        help=(
            "Optional run-level subset scope filter. LOCOMO: all, multi_hop, temporal, open_domain, single_hop. "
            "MedMT: all, long_context_memory_and_understanding, "
            "resistance_to_contextual_interference, information_contradiction."
        ),
    ),
    backbone_model: str | None = typer.Option(None),
    judge_model: str | None = typer.Option(None),
    embedding_model: str | None = typer.Option(None),
    backbone_provider_kind: str | None = typer.Option(None),
    judge_provider_kind: str | None = typer.Option(None),
    m: int | None = typer.Option(None),
    k: int | None = typer.Option(None),
    t_pages: int | None = typer.Option(None),
    neighbor_radius: int | None = typer.Option(None),
    retrieval_expansion_mode: str | None = typer.Option(None),
    run_id: str | None = typer.Option(None),
    limit: int | None = typer.Option(None),
    sort_by: str = typer.Option("completed_at_desc"),
) -> None:
    reader = SQLiteReportReader(console=console)
    if database_path is not None:
        rows = reader.report_single_run(database_path=database_path, sort_by=sort_by)
    else:
        rows = reader.report_index(
            index_database_path=index_database_path,
            dataset=dataset,
            subset=subset,
            backbone_model=backbone_model,
            judge_model=judge_model,
            embedding_model=embedding_model,
            backbone_provider_kind=backbone_provider_kind,
            judge_provider_kind=judge_provider_kind,
            m=m,
            k=k,
            t_pages=t_pages,
            neighbor_radius=neighbor_radius,
            retrieval_expansion_mode=retrieval_expansion_mode,
            run_id=run_id,
            limit=limit,
            sort_by=sort_by,
        )
    reader.print_tables(rows)


@app.command()
def inspect(
    database_path: Path = typer.Option(..., exists=True, help="Path to a single run SQLite database."),
    trajectory_id: str | None = typer.Option(None),
    snapshot_id: str | None = typer.Option(None),
) -> None:
    session_factory = create_schema(database_path)
    session = session_factory()
    inspector = Inspector(TrajWikiStore(session))
    if trajectory_id:
        console.print_json(data=inspector.trajectory(trajectory_id))
        return
    if snapshot_id:
        console.print_json(data=inspector.snapshot(snapshot_id))
        return
    raise typer.BadParameter("Pass either --trajectory-id or --snapshot-id.")


@app.command()
def analyze_failures(
    run_path: Path = typer.Option(
        ...,
        exists=True,
        help="Path to a completed LOCOMO run directory or a subset scope directory containing run subdirectories.",
    ),
    top_examples_per_bucket: int = typer.Option(
        5,
        min=1,
        help="Maximum number of failed examples to print for each failure reason bucket.",
    ),
    sampling_manifest: Path | None = typer.Option(
        None,
        "--sampling-manifest",
        exists=True,
        help="Optional answer-ablation sampling manifest used to restrict query-level diagnostics.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the raw attribution report as JSON instead of Rich tables.",
    ),
    show_ranks: bool = typer.Option(
        False,
        help="Include coarse/fine rank details under printed example rows.",
    ),
    show_facets: bool = typer.Option(
        False,
        help="Include facet and diagnostic flag details under printed example rows.",
    ),
) -> None:
    try:
        report = analyze_locomo_run_failures(
            run_path,
            top_examples_per_bucket=top_examples_per_bucket,
            sampling_manifest_path=sampling_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        incomplete_report = load_incomplete_run_diagnostics(run_path)
        if incomplete_report is not None:
            if json_output:
                console.print_json(data=incomplete_report)
                return
            print_incomplete_run_diagnostics(incomplete_report, console=console)
            return
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        console.print_json(data=report)
        return
    print_locomo_failure_report(
        report,
        console=console,
        show_ranks=show_ranks,
        show_facets=show_facets,
    )


@app.command(name="analyze-offline-ablation")
def analyze_offline_ablation_command(
    run_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to a completed LOCOMO run directory.",
    ),
    variants: str = typer.Option(
        "full,no_wiki_direct,wiki_only,flat_raw,snapshot_m1,snapshot_m2,source_supported_only",
        help="Comma-separated offline ablation variants.",
    ),
    budgets: str = typer.Option(
        "4000,8000,16000,32000",
        help="Comma-separated context token budgets.",
    ),
    rank_cutoffs: str = typer.Option(
        "5,10,15,20,30,50",
        help="Comma-separated retrieval rank cutoffs.",
    ),
    sampling_manifest: Path | None = typer.Option(
        None,
        "--sampling-manifest",
        exists=True,
        help="Optional answer-ablation sampling manifest used to restrict this report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print generated artifact paths and summary metadata as JSON.",
    ),
) -> None:
    try:
        report = analyze_offline_ablation(
            run_path,
            variants=variants,
            budgets=budgets,
            rank_cutoffs=rank_cutoffs,
            sampling_manifest_path=sampling_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Offline ablation artifacts written to {report['analysis_dir']}")
    console.print(f"Summary: {report['summary_path']}")
    console.print(f"Table: {report['table_path']}")


@app.command(name="analyze-cost-benefit")
def analyze_cost_benefit_command(
    run_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to a completed LOCOMO run directory.",
    ),
    baselines: str = typer.Option(
        "trajwiki_observed,full_context_proxy,no_wiki_direct,flat_raw,wiki_only",
        help="Comma-separated observed/proxy methods to include.",
    ),
    price_config: Path | None = typer.Option(
        None,
        "--price-config",
        help="Optional LLM pricing JSON used to estimate dollar cost.",
    ),
    future_query_counts: str = typer.Option(
        "1,2,5,10,20,50,100",
        "--future-query-counts",
        help="Comma-separated future query counts for amortization curves.",
    ),
    sampling_manifest: Path | None = typer.Option(
        None,
        "--sampling-manifest",
        exists=True,
        help="Optional answer-ablation sampling manifest used to restrict this report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print generated artifact paths and summary metadata as JSON.",
    ),
) -> None:
    try:
        report = analyze_cost_benefit(
            run_path,
            baselines=baselines,
            price_config_path=price_config,
            future_query_counts=future_query_counts,
            sampling_manifest_path=sampling_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Cost-benefit artifacts written to {report['analysis_dir']}")
    console.print(f"Summary: {report['summary_path']}")
    console.print(f"Quality table: {report['quality_table_path']}")
    console.print(f"Break-even table: {report['break_even_path']}")


@app.command(name="analyze-auditability")
def analyze_auditability_command(
    run_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to a completed LOCOMO run directory.",
    ),
    baselines: str = typer.Option(
        "trajwiki_observed,full_context_proxy,no_wiki_direct,flat_raw,wiki_only",
        help="Comma-separated observed/proxy methods to include.",
    ),
    audit_labels: Path | None = typer.Option(
        None,
        "--audit-labels",
        help="Optional CSV/JSONL human audit labels.",
    ),
    sampling_manifest: Path | None = typer.Option(
        None,
        "--sampling-manifest",
        exists=True,
        help="Optional answer-ablation sampling manifest used to restrict this report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print generated artifact paths and summary metadata as JSON.",
    ),
) -> None:
    try:
        report = analyze_auditability(
            run_path,
            baselines=baselines,
            audit_labels_path=audit_labels,
            sampling_manifest_path=sampling_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Auditability artifacts written to {report['analysis_dir']}")
    console.print(f"Summary: {report['summary_path']}")
    console.print(f"Source support table: {report['source_support_table_path']}")
    console.print(f"Failure localization table: {report['failure_localization_table_path']}")
    console.print(f"Error propagation funnel: {report['error_propagation_funnel_path']}")


@app.command(name="run-answer-ablation")
def run_answer_ablation_command(
    run_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Completed LOCOMO run used as the immutable memory source.",
    ),
    variants: str = typer.Option(
        "full,direct_trajectory,latest_snapshot,hybrid_raw_rag,wiki_summaries,no_claim_state,no_source_constraint,full_context,naive_dense_rag,full_context_matched,hybrid_raw_rag_matched",
        help="Comma-separated answer-level ablation variants.",
    ),
    baseline_answers: Path | None = typer.Option(
        None,
        "--baseline-answers",
        exists=True,
        help="Optional normalized JSONL from import-baseline-answers for the same blind judge.",
    ),
    sample_size: int = typer.Option(60, min=1),
    sampling_profile: str = typer.Option(
        "auto",
        help="auto, rebuttal_60_v1, or rebuttal_200_v1.",
    ),
    sample_seed: int = typer.Option(7),
    max_total_tokens: int = typer.Option(
        32000,
        "--max-total-tokens",
        "--max-prompt-tokens",
        min=512,
        help="Total prompt-plus-output budget; --max-prompt-tokens is a legacy alias.",
    ),
    max_output_tokens: int = typer.Option(512, min=1),
    token_counter: str = typer.Option(
        "auto",
        help="Token counter: auto, tiktoken, hf, or estimate.",
    ),
    require_exact_token_counter: bool = typer.Option(
        False,
        "--require-exact-token-counter/--allow-estimated-token-counter",
    ),
    token_safety_margin: int = typer.Option(128, min=0),
    rag_chunk_size: int = typer.Option(384, min=1),
    rag_chunk_overlap: int = typer.Option(64, min=0),
    rag_top_k: int = typer.Option(4, min=1),
    backbone_provider_kind: str = typer.Option("remote"),
    backbone_model: str = typer.Option("gpt-4o-mini"),
    independent_judge_provider_kind: str = typer.Option("remote"),
    independent_judge_model: str = typer.Option("claude-sonnet-4-6"),
    generation_temperature: float = typer.Option(0.0),
    generation_seed: int = typer.Option(7),
    generation_max_concurrency: int = typer.Option(6, min=1),
    judge_max_concurrency: int = typer.Option(6, min=1),
    context_save_mode: str = typer.Option("full", help="full or compact."),
    max_provider_calls: int = typer.Option(1500, min=1),
    reuse_experiment: Path | None = typer.Option(
        None,
        "--reuse-experiment",
        exists=True,
        help="Verified parent answer-ablation experiment to reuse.",
    ),
    reuse_policy: str = typer.Option(
        "off",
        help="Parent reuse policy: require, best-effort, or off.",
    ),
    progress: bool = typer.Option(
        False,
        "--progress/--no-progress",
        help="Print periodic stage progress to stderr and update progress.json.",
    ),
    progress_interval_seconds: int = typer.Option(30, min=1),
    report_path: Path | None = typer.Option(
        None,
        "--report-path",
        help="Atomically write the final machine-readable report to this path.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build and validate contexts/call plan without LLM provider calls.",
    ),
) -> None:
    try:
        report = run_answer_ablation(
            run_path,
            variants=variants,
            baseline_answers_path=baseline_answers,
            sample_size=sample_size,
            sampling_profile=sampling_profile,
            sample_seed=sample_seed,
            max_total_tokens=max_total_tokens,
            max_output_tokens=max_output_tokens,
            token_counter=token_counter,
            require_exact_token_counter=require_exact_token_counter,
            token_safety_margin=token_safety_margin,
            rag_chunk_size=rag_chunk_size,
            rag_chunk_overlap=rag_chunk_overlap,
            rag_top_k=rag_top_k,
            backbone_provider_kind=backbone_provider_kind,
            backbone_model=backbone_model,
            independent_judge_provider_kind=independent_judge_provider_kind,
            independent_judge_model=independent_judge_model,
            generation_temperature=generation_temperature,
            generation_seed=generation_seed,
            generation_max_concurrency=generation_max_concurrency,
            judge_max_concurrency=judge_max_concurrency,
            context_save_mode=context_save_mode,
            max_provider_calls=max_provider_calls,
            reuse_experiment_path=reuse_experiment,
            reuse_policy=reuse_policy,
            progress=progress,
            progress_interval_seconds=progress_interval_seconds,
            report_path=report_path,
            resume=resume,
            dry_run=dry_run,
        )
    except (FileNotFoundError, ProviderConfigurationError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if report_path is not None:
        console.print(f"Report written atomically to {report_path.resolve()}")
        console.print(f"Experiment: {report['experiment_dir']}")
    else:
        console.print_json(data=report)


@app.command(name="analyze-answer-ablation")
def analyze_answer_ablation_command(
    experiment_path: Path = typer.Argument(..., exists=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = analyze_answer_ablation(experiment_path)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Answer ablation table: {report['answer_ablation_table']}")
    console.print(f"Paired statistics: {report['paired_statistics']}")
    console.print(
        f"Observed-stage diagnostics: {report['stage_diagnostic_statistics']}"
    )
    console.print(f"Integrity report: {report['integrity_report']}")


@app.command(name="analyze-ranking-robustness")
def analyze_ranking_robustness_command(
    run_path: Path = typer.Argument(..., exists=True),
    relative_perturbation: float = typer.Option(0.20, min=0.0),
    random_draws: int = typer.Option(100, min=1),
    seed: int = typer.Option(7),
    cutoff: int = typer.Option(15, min=1),
    page_cutoffs: str = typer.Option("5,10,15"),
    trajectory_cutoffs: str = typer.Option("5,10,15,20,30"),
    sampling_manifest: Path | None = typer.Option(
        None,
        "--sampling-manifest",
        exists=True,
        help="Restrict the report to an answer-ablation sampling manifest.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = analyze_ranking_robustness(
            run_path,
            relative_perturbation=relative_perturbation,
            random_draws=random_draws,
            seed=seed,
            cutoff=cutoff,
            page_cutoffs=page_cutoffs,
            trajectory_cutoffs=trajectory_cutoffs,
            sampling_manifest_path=sampling_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Ranking robustness rows: {report['rows_path']}")
    console.print(f"Cutoff replay rows: {report['cutoff_rows_path']}")


@app.command(name="validate-run-artifacts")
def validate_run_artifacts_command(
    path: Path = typer.Argument(..., exists=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = validate_run_artifacts(path)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
        if report["error_count"]:
            raise typer.Exit(code=1)
        return
    status = "passed" if not report["error_count"] else "failed"
    console.print(
        f"Artifact validation {status}: {report['error_count']} error(s)."
    )
    for error in report["errors"]:
        console.print(f"- {error}")
    if report["error_count"]:
        raise typer.Exit(code=1)


@app.command(name="package-rebuttal-bundle")
def package_rebuttal_bundle_command(
    experiment_path: Path = typer.Argument(..., exists=True),
    output_path: Path | None = typer.Option(None, "--output-path"),
    workflow_logs: Path | None = typer.Option(
        None,
        "--workflow-logs",
        exists=True,
        help="Optional private workflow reports and Kubernetes logs.",
    ),
    repository_path: Path | None = typer.Option(
        None,
        "--repository-path",
        exists=True,
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = package_rebuttal_bundle(
            experiment_path,
            output_path=output_path,
            workflow_logs_path=workflow_logs,
            repository_path=repository_path,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Bundle: {report['bundle_path']}")
    console.print(f"SHA256: {report['bundle_sha256']}")


@app.command(name="validate-rebuttal-bundle")
def validate_rebuttal_bundle_command(
    bundle_path: Path = typer.Argument(..., exists=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = validate_rebuttal_bundle(bundle_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
    else:
        status = "passed" if not report["error_count"] else "failed"
        console.print(
            f"Rebuttal bundle validation {status}: "
            f"{report['error_count']} error(s)."
        )
        for error in report["errors"]:
            console.print(f"- {error}")
    if report["error_count"]:
        raise typer.Exit(code=1)


@app.command(name="prepare-audit-study")
def prepare_audit_study_command(
    run_path: Path = typer.Argument(..., exists=True),
    case_count: int = typer.Option(24, min=1),
    seed: int = typer.Option(7),
    answer_experiment: Path | None = typer.Option(
        None,
        "--answer-experiment",
        exists=True,
        help="Use the regenerated Full TrajWiki answers from this experiment.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = prepare_audit_study(
            run_path,
            case_count=case_count,
            seed=seed,
            answer_experiment_path=answer_experiment,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Audit tasks: {report['tasks_path']}")
    console.print(f"Blinding key (do not distribute to annotators): {report['key_path']}")
    console.print(
        "Adjudication template (keep private): "
        f"{report['audit_labels_template_path']}"
    )
@app.command(name="conduct-audit-study")
def conduct_audit_study_command(
    tasks_path: Path = typer.Argument(..., exists=True),
    annotator_slot: int = typer.Option(..., min=1, max=2),
    annotator_id: str = typer.Option(...),
    output_path: Path = typer.Option(...),
    max_exposure: int = typer.Option(
        1,
        "--max-exposure",
        min=1,
        max=2,
        help=(
            "Run first-exposure primary tasks only (1), or also the optional "
            "second-exposure crossover block (2)."
        ),
    ),
) -> None:
    try:
        report = conduct_audit_study(
            tasks_path,
            annotator_slot=annotator_slot,
            annotator_id=annotator_id,
            output_path=output_path,
            max_exposure=max_exposure,
            console=console,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=report)


@app.command(name="analyze-audit-study")
def analyze_audit_study_command(
    study_path: Path = typer.Argument(..., exists=True),
    audit_labels: Path | None = typer.Option(None, "--audit-labels"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = analyze_audit_study(
            study_path,
            audit_labels_path=audit_labels,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Human audit summary: {report['summary_path']}")
    console.print(f"Disagreements: {report['disagreements_path']}")


@app.command(name="import-baseline-answers")
def import_baseline_answers_command(
    source_path: Path = typer.Argument(..., exists=True),
    output_path: Path | None = typer.Option(None, "--output-path"),
    method: str | None = typer.Option(
        None,
        "--method",
        help="Override the method name for every imported row.",
    ),
    run_path: Path | None = typer.Option(
        None,
        "--run-path",
        exists=True,
        help="Validate all prediction ids, samples, questions, and gold answers.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = import_baseline_answers(
            source_path,
            output_path=output_path,
            method=method,
            run_path=run_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Normalized baseline answers: {report['output_path']}")


@app.command()
def analyze_failures_diff(
    before: Path = typer.Option(
        ...,
        exists=True,
        help="Path to a baseline LOCOMO run directory or a subset scope directory containing run subdirectories.",
    ),
    after: Path = typer.Option(
        ...,
        exists=True,
        help="Path to a comparison LOCOMO run directory or a subset scope directory containing run subdirectories.",
    ),
    top_examples_per_bucket: int = typer.Option(
        5,
        min=1,
        help="Maximum number of query transitions to print for each diff bucket.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the raw diff report as JSON instead of Rich tables.",
    ),
) -> None:
    try:
        diff_report = diff_locomo_failure_reports(
            before,
            after,
            top_examples_per_bucket=top_examples_per_bucket,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        console.print_json(data=diff_report)
        return
    print_locomo_failure_diff(diff_report, console=console)


@app.command()
def export(
    database_path: Path = typer.Option(..., exists=True, help="Path to a single run SQLite database."),
    output_dir: Path = typer.Option(Path("output/export")),
    sample_id: str | None = typer.Option(None),
) -> None:
    session_factory = create_schema(database_path)
    session = session_factory()
    store = TrajWikiStore(session)
    exporter = ArtifactExporter(output_dir, store)
    if sample_id is not None:
        exporter.export_sample_trajectories(sample_id)
        console.print(f"Exported trajectories for {sample_id} to {output_dir}")
        return
    for current_sample_id in store.list_sample_ids():
        exporter.export_sample_trajectories(current_sample_id)
    console.print(f"Exported trajectories to {output_dir}")


if __name__ == "__main__":
    app()
