from __future__ import annotations

from trajpatch.config import RunConfig
from trajpatch.providers.devices import (
    build_role_device_plans,
    build_worker_embedding_device_plans,
    nvidia_smi_inventory_from_csv,
    visible_cuda_physical_to_logical_map,
)
from trajpatch.providers.cuda_preflight import run_cuda_preflight
from trajpatch.providers.factory import build_provider_bundle
from trajpatch.types import DevicePlan


def test_role_device_plans_split_backbone_judge_and_embedding_across_two_gpus(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "trajpatch.providers.devices.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "GPU-0",
                "total_bytes": 24 * 1024**3,
                "free_bytes": 20 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            },
            {
                "index": 1,
                "name": "GPU-1",
                "total_bytes": 24 * 1024**3,
                "free_bytes": 18 * 1024**3,
                "used_bytes": 6 * 1024**3,
                "source": "test",
            },
        ],
    )
    config = RunConfig(
        dataset="medmt",
        dataset_path=tmp_path / "dataset.json",
        output_dir=tmp_path / "old_output",
        database_path=tmp_path / "old_output.sqlite",
        backbone_provider_kind="local",
        judge_provider_kind="local",
        backbone_model="Qwen/Qwen2.5-7B-Instruct",
        judge_model="Qwen/Qwen2.5-3B-Instruct",
        embedding_model="huggingface/Qwen3-Embedding-8B",
    )

    plans = build_role_device_plans(config)

    assert plans["backbone"].accelerator == "cuda:0"
    assert plans["judge"].accelerator == "cuda:1"
    assert plans["embedding"].accelerator in {"cuda:0", "cuda:1"}


def test_openai_compatible_provider_is_treated_as_remote_for_device_planning(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "trajpatch.providers.devices.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "GPU-0",
                "total_bytes": 24 * 1024**3,
                "free_bytes": 20 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            }
        ],
    )
    config = RunConfig(
        dataset="medmt",
        dataset_path=tmp_path / "dataset.json",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="openai-compatible",
        judge_provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        judge_model="qwen3-8b",
        embedding_model="huggingface/Qwen3-Embedding-8B",
    )

    plans = build_role_device_plans(config)

    assert plans["backbone"].accelerator == "remote"
    assert plans["judge"].accelerator == "remote"
    assert plans["embedding"].accelerator == "cuda:0"


def test_cuda_preflight_off_returns_disabled_report(monkeypatch, tmp_path):
    monkeypatch.setattr("trajpatch.providers.cuda_preflight.detect_cuda_inventory", lambda: [])
    config = RunConfig(
        dataset="medmt",
        dataset_path=tmp_path / "dataset.json",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="remote",
        judge_provider_kind="remote",
        backbone_model="gpt-4o-mini",
        judge_model="gpt-4o-mini",
        cuda_preflight_mode="off",
    )

    report = run_cuda_preflight(config)

    assert report.enabled is False
    assert report.risk == "off"
    assert report.device_plan_overrides == {}


def test_cuda_preflight_reserves_vllm_gpu_and_assigns_embedding_elsewhere(monkeypatch, tmp_path):
    inventory = [
        {
            "index": 0,
            "name": "GPU-0",
            "total_bytes": 80 * 1024**3,
            "free_bytes": 76 * 1024**3,
            "used_bytes": 4 * 1024**3,
            "source": "test",
        },
        {
            "index": 1,
            "name": "GPU-1",
            "total_bytes": 80 * 1024**3,
            "free_bytes": 76 * 1024**3,
            "used_bytes": 4 * 1024**3,
            "source": "test",
        },
    ]
    monkeypatch.setattr("trajpatch.providers.cuda_preflight.detect_cuda_inventory", lambda: inventory)
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="openai-compatible",
        judge_provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        judge_model="qwen3-8b",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        vllm_autostart=True,
        vllm_cuda_visible_devices="0",
        conv_workers=2,
    )

    report = run_cuda_preflight(config)

    assert report.reserved_device_indices == {0}
    assert report.reservations[0]["role"] == "vllm_server"
    assert report.device_plan_overrides["embedding"].accelerator == "cuda:1"
    worker_assignments = [row for row in report.assignments if row["role"].startswith("embedding_worker_")]
    assert worker_assignments
    assert all(row["device_index"] == 1 for row in worker_assignments)


def test_cuda_preflight_strict_rejects_vllm_autostart_without_device_isolation(tmp_path):
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="openai-compatible",
        judge_provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        judge_model="qwen3-8b",
        vllm_autostart=True,
        cuda_preflight_mode="strict",
    )

    try:
        run_cuda_preflight(
            config,
            inventory=[
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
    except Exception as exc:  # noqa: BLE001
        assert "CUDA preflight failed" in str(exc)
    else:
        raise AssertionError("strict CUDA preflight should reject unspecified vLLM devices")


def test_cuda_preflight_warn_allows_vllm_autostart_without_device_isolation(tmp_path):
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="openai-compatible",
        judge_provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        judge_model="qwen3-8b",
        vllm_autostart=True,
        cuda_preflight_mode="warn",
    )

    report = run_cuda_preflight(
        config,
        inventory=[
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

    assert report.errors == []
    assert report.warnings
    assert report.risk in {"medium", "high"}


def test_cuda_preflight_local_roles_are_spread_across_gpus(monkeypatch, tmp_path):
    inventory = [
        {
            "index": 0,
            "name": "GPU-0",
            "total_bytes": 80 * 1024**3,
            "free_bytes": 76 * 1024**3,
            "used_bytes": 4 * 1024**3,
            "source": "test",
        },
        {
            "index": 1,
            "name": "GPU-1",
            "total_bytes": 80 * 1024**3,
            "free_bytes": 74 * 1024**3,
            "used_bytes": 6 * 1024**3,
            "source": "test",
        },
        {
            "index": 2,
            "name": "GPU-2",
            "total_bytes": 80 * 1024**3,
            "free_bytes": 72 * 1024**3,
            "used_bytes": 8 * 1024**3,
            "source": "test",
        },
    ]
    monkeypatch.setattr("trajpatch.providers.cuda_preflight.detect_cuda_inventory", lambda: inventory)
    config = RunConfig(
        dataset="medmt",
        dataset_path=tmp_path / "dataset.json",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="local",
        judge_provider_kind="local",
        backbone_model="Qwen/Qwen3-8B",
        judge_model="Qwen/Qwen3-8B",
        embedding_model="huggingface/Qwen3-Embedding-8B",
    )

    report = run_cuda_preflight(config)

    accelerators = {
        role: plan.accelerator
        for role, plan in report.device_plan_overrides.items()
        if plan is not None and role in {"backbone", "judge", "embedding"}
    }
    assert len(set(accelerators.values())) == 3


def test_worker_embedding_device_plans_balance_four_workers_across_two_equal_gpus(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "trajpatch.providers.devices.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "H100-0",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 76 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            },
            {
                "index": 1,
                "name": "H100-1",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 76 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            },
        ],
    )
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="remote",
        judge_provider_kind="remote",
        backbone_model="gpt-4o-mini",
        judge_model="gpt-4o-mini",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        conv_workers=4,
    )

    plans, metadata = build_worker_embedding_device_plans(config, 4)

    assert [plan.accelerator for plan in plans if plan is not None] == [
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
    ]
    assert metadata["enabled"] is True
    assert metadata["model_sharing_enabled"] is True
    assert metadata["serialized_encode_per_shared_model"] is True
    assert metadata["per_device"]["0"]["worker_ids"] == [0, 2]
    assert metadata["per_device"]["1"]["worker_ids"] == [1, 3]
    assert metadata["per_device"]["0"]["model_instance_count"] == 1
    assert metadata["per_device"]["1"]["model_instance_count"] == 1
    cuda0_metadata = metadata["per_device"]["0"]
    assert cuda0_metadata["uncached_estimated_float32_bytes"] > cuda0_metadata[
        "estimated_float32_bytes"
    ]
    assert metadata["warnings"] == []


def test_worker_embedding_device_plans_give_larger_gpu_more_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "trajpatch.providers.devices.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "GPU-0",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 76 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            },
            {
                "index": 1,
                "name": "GPU-1",
                "total_bytes": 40 * 1024**3,
                "free_bytes": 38 * 1024**3,
                "used_bytes": 2 * 1024**3,
                "source": "test",
            },
        ],
    )
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "artifacts",
        backbone_provider_kind="remote",
        judge_provider_kind="remote",
        backbone_model="gpt-4o-mini",
        judge_model="gpt-4o-mini",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        conv_workers=3,
    )

    plans, metadata = build_worker_embedding_device_plans(config, 3)

    accelerators = [plan.accelerator for plan in plans if plan is not None]
    assert accelerators.count("cuda:0") > accelerators.count("cuda:1")
    assert metadata["per_device"]["0"]["worker_count"] == 2
    assert metadata["per_device"]["1"]["worker_count"] == 1
    assert metadata["per_device"]["0"]["model_instance_count"] == 1
    assert metadata["per_device"]["1"]["model_instance_count"] == 1


def test_worker_embedding_device_plans_skip_hash_embedding_and_use_cpu_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr("trajpatch.providers.devices.detect_cuda_inventory", lambda: [])
    hash_config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "hash-artifacts",
        provider_kind="mock",
        embedding_model="hash-embedding",
        conv_workers=2,
    )
    cpu_config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "cpu-artifacts",
        provider_kind="mock",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        device_mode="cpu",
        conv_workers=2,
    )

    hash_plans, hash_metadata = build_worker_embedding_device_plans(hash_config, 2)
    cpu_plans, cpu_metadata = build_worker_embedding_device_plans(cpu_config, 2)

    assert hash_plans == [None, None]
    assert hash_metadata["enabled"] is False
    assert hash_metadata["reason"] == "hash_embedding"
    assert [plan.accelerator for plan in cpu_plans if plan is not None] == ["cpu", "cpu"]
    assert cpu_metadata["enabled"] is False
    assert cpu_metadata["reason"] == "device_mode_cpu"


def test_worker_embedding_device_plans_warn_when_estimate_exceeds_free_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "trajpatch.providers.devices.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "Tiny-GPU",
                "total_bytes": 16 * 1024**3,
                "free_bytes": 8 * 1024**3,
                "used_bytes": 8 * 1024**3,
                "source": "test",
            }
        ],
    )
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "artifacts",
        provider_kind="mock",
        embedding_model="huggingface/Qwen3-Embedding-8B",
        conv_workers=2,
    )

    _, metadata = build_worker_embedding_device_plans(config, 2)

    assert metadata["enabled"] is True
    assert metadata["warnings"]


def test_build_provider_bundle_respects_embedding_device_plan_override(tmp_path):
    config = RunConfig(
        dataset="locomo",
        dataset_path=tmp_path / "locomo",
        output_dir=tmp_path / "artifacts",
        provider_kind="mock",
        embedding_model="huggingface/Qwen3-Embedding-8B",
    )
    override = DevicePlan(
        device_mode="single",
        accelerator="cuda:7",
        visible_devices=[7],
        metadata={"role": "embedding", "selected_device_index": 7},
    )

    _, _, embedding, allocation = build_provider_bundle(
        config,
        device_plan_overrides={"embedding": override},
    )

    assert embedding.device_plan.accelerator == "cuda:7"
    assert allocation["embedding"]["accelerator"] == "cuda:7"
    assert allocation["embedding"]["metadata"]["selected_device_index"] == 7


def test_nvidia_smi_inventory_maps_visible_physical_ids_to_logical_ids(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    csv = "2, H100-physical-2, 80000, 1000, 79000\n3, H100-physical-3, 80000, 2000, 78000\n"

    inventory = nvidia_smi_inventory_from_csv(csv, visible_cuda_physical_to_logical_map())

    assert [row["index"] for row in inventory] == [0, 1]
    assert [row["physical_index"] for row in inventory] == [2, 3]
    assert inventory[0]["source"] == "nvidia-smi"


def test_nvidia_smi_inventory_uses_physical_ids_without_visible_filter(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    csv = "2, H100-physical-2, 80000, 1000, 79000\n"

    inventory = nvidia_smi_inventory_from_csv(csv, visible_cuda_physical_to_logical_map())

    assert inventory[0]["index"] == 2
    assert inventory[0]["physical_index"] == 2


def test_nvidia_smi_inventory_skips_non_numeric_visible_tokens(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-GPU-abc/1/0")
    csv = "0, MIG-parent, 80000, 1000, 79000\n"

    inventory = nvidia_smi_inventory_from_csv(csv, visible_cuda_physical_to_logical_map())

    assert inventory == []
