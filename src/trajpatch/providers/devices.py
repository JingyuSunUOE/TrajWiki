"""Heuristics for assigning local models and embeddings to available devices."""

from __future__ import annotations

import os
import re
import subprocess
from copy import deepcopy

from trajpatch.types import DevicePlan


def infer_parameter_size_b(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    if match:
        return float(match.group(1))
    return None


def detect_visible_cuda_devices() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return [index for index, token in enumerate(visible.split(",")) if token.strip() != ""]
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def visible_cuda_physical_to_logical_map() -> dict[int, int] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return None
    mapping: dict[int, int] = {}
    for logical_index, token in enumerate(token.strip() for token in visible.split(",")):
        if not token:
            continue
        if not token.isdigit():
            return {}
        mapping[int(token)] = logical_index
    return mapping


def nvidia_smi_inventory_from_csv(stdout: str, physical_to_logical: dict[int, int] | None = None) -> list[dict]:
    inventory = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        physical_index = int(parts[0])
        if physical_to_logical is None:
            logical_index = physical_index
        elif physical_index in physical_to_logical:
            logical_index = physical_to_logical[physical_index]
        else:
            continue
        total_mb = int(parts[2])
        used_mb = int(parts[3])
        free_mb = int(parts[4])
        inventory.append(
            {
                "index": logical_index,
                "physical_index": physical_index,
                "name": parts[1],
                "total_bytes": total_mb * 1024 * 1024,
                "free_bytes": free_mb * 1024 * 1024,
                "used_bytes": used_mb * 1024 * 1024,
                "source": "nvidia-smi",
            }
        )
    return inventory


def detect_cuda_inventory() -> list[dict]:
    visible_devices = detect_visible_cuda_devices()
    if not visible_devices:
        return []
    try:
        import torch

        inventory = []
        for index in visible_devices:
            properties = torch.cuda.get_device_properties(index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                used_bytes = total_bytes - free_bytes
            except Exception:
                total_bytes = int(properties.total_memory)
                used_bytes = int(torch.cuda.memory_allocated(index))
                free_bytes = total_bytes - used_bytes
            inventory.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_bytes": int(total_bytes),
                    "free_bytes": int(free_bytes),
                    "used_bytes": int(used_bytes),
                    "source": "torch",
                }
            )
        return inventory
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return nvidia_smi_inventory_from_csv(result.stdout, visible_cuda_physical_to_logical_map())
    except Exception:
        return []


def build_device_plan(model_name: str, device_mode: str) -> DevicePlan:
    parameter_billions = infer_parameter_size_b(model_name)
    visible_devices = detect_visible_cuda_devices()
    if device_mode == "cpu" or not visible_devices:
        return DevicePlan(
            device_mode="cpu",
            accelerator="cpu",
            visible_devices=visible_devices,
            metadata={"parameter_billions": parameter_billions},
        )

    if device_mode == "single":
        return DevicePlan(
            device_mode="single",
            accelerator=f"cuda:{visible_devices[0]}",
            visible_devices=visible_devices[:1],
            metadata={"parameter_billions": parameter_billions},
        )

    if device_mode == "multi" or ((parameter_billions or 0) > 13 and len(visible_devices) > 1):
        return DevicePlan(
            device_mode="multi",
            accelerator="cuda",
            tensor_parallel_size=len(visible_devices),
            device_map="auto",
            visible_devices=visible_devices,
            metadata={"parameter_billions": parameter_billions},
        )

    cpu_offload = bool(parameter_billions and parameter_billions > 30 and len(visible_devices) == 1)
    return DevicePlan(
        device_mode="auto",
        accelerator=f"cuda:{visible_devices[0]}",
        device_map="auto" if cpu_offload else None,
        cpu_offload=cpu_offload,
        visible_devices=visible_devices[:1],
        metadata={"parameter_billions": parameter_billions},
    )


def resolve_hf_model_name(model_name: str) -> str:
    aliases = {
        "huggingface/Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    }
    return aliases.get(model_name, model_name)


def estimate_model_bytes(model_name: str, *, bytes_per_param: int = 2, overhead: float = 1.2) -> int | None:
    parameter_billions = infer_parameter_size_b(model_name)
    if parameter_billions is None:
        return None
    return int(parameter_billions * 1_000_000_000 * bytes_per_param * overhead)


def build_embedding_device_plan(model_name: str, device_mode: str) -> DevicePlan:
    resolved_model_name = resolve_hf_model_name(model_name)
    inventory = detect_cuda_inventory()
    estimated_required_bytes = estimate_model_bytes(resolved_model_name)
    metadata = {
        "requested_model_name": model_name,
        "resolved_model_name": resolved_model_name,
        "estimated_required_bytes": estimated_required_bytes,
        "cuda_inventory": inventory,
    }
    if device_mode == "cpu" or not inventory:
        return DevicePlan(
            device_mode="cpu",
            accelerator="cpu",
            visible_devices=[],
            metadata=metadata,
        )

    ranked = sorted(inventory, key=lambda item: item["free_bytes"], reverse=True)
    suitable = [
        item for item in ranked if estimated_required_bytes is None or item["free_bytes"] >= estimated_required_bytes
    ]
    chosen = suitable[0] if suitable else ranked[0]
    metadata.update(
        {
            "selected_device_index": chosen["index"],
            "selected_device_name": chosen["name"],
            "selected_device_free_bytes": chosen["free_bytes"],
            "selected_device_total_bytes": chosen["total_bytes"],
            "meets_estimated_requirement": (
                True if estimated_required_bytes is None else chosen["free_bytes"] >= estimated_required_bytes
            ),
        }
    )
    return DevicePlan(
        device_mode="single",
        accelerator=f"cuda:{chosen['index']}",
        visible_devices=[chosen["index"]],
        metadata=metadata,
    )


def build_worker_embedding_device_plans(
    config,
    worker_count: int,
    *,
    excluded_device_indices: set[int] | None = None,
    preflight_report: dict | None = None,
) -> tuple[list[DevicePlan | None], dict]:
    """Pre-assign local embedding devices for sharded workers.

    Threaded workers cannot safely use per-thread CUDA_VISIBLE_DEVICES. This
    helper produces explicit SentenceTransformer device plans up front so each
    worker uses a deterministic cuda:N target.
    """

    worker_count = max(int(worker_count), 0)
    resolved_model_name = resolve_hf_model_name(config.embedding_model)
    estimated_required_bytes = estimate_model_bytes(resolved_model_name, bytes_per_param=2, overhead=1.2)
    estimated_float32_required_bytes = estimate_model_bytes(
        resolved_model_name, bytes_per_param=4, overhead=1.2
    )
    excluded_device_indices = set(excluded_device_indices or set())
    metadata = {
        "enabled": False,
        "worker_count": worker_count,
        "requested_model_name": config.embedding_model,
        "resolved_model_name": resolved_model_name,
        "estimated_required_bytes": estimated_required_bytes,
        "estimated_float32_required_bytes": estimated_float32_required_bytes,
        "model_sharing_enabled": True,
        "serialized_encode_per_shared_model": True,
        "excluded_device_indices": sorted(excluded_device_indices),
        "cuda_preflight": dict(preflight_report or {}),
        "assignments": [],
        "per_device": {},
        "warnings": [],
    }
    if worker_count <= 1:
        metadata["reason"] = "single_worker"
        return [None for _ in range(worker_count)], metadata
    if config.embedding_model == "hash-embedding":
        metadata["reason"] = "hash_embedding"
        return [None for _ in range(worker_count)], metadata

    inventory = deepcopy(detect_cuda_inventory())
    metadata["cuda_inventory"] = inventory
    if config.device_mode == "cpu" or not inventory:
        reason = "device_mode_cpu" if config.device_mode == "cpu" else "no_cuda_inventory"
        metadata["reason"] = reason
        plans = [
            DevicePlan(
                device_mode="cpu",
                accelerator="cpu",
                visible_devices=[],
                metadata={
                    "role": "embedding",
                    "worker_id": worker_id,
                    "worker_count": worker_count,
                    "worker_preallocated": True,
                    "worker_preallocation_reason": reason,
                    "requested_model_name": config.embedding_model,
                    "resolved_model_name": resolved_model_name,
                    "estimated_required_bytes": estimated_required_bytes,
                    "estimated_float32_required_bytes": estimated_float32_required_bytes,
                    "model_sharing_enabled": False,
                    "cuda_inventory": inventory,
                },
            )
            for worker_id in range(worker_count)
        ]
        metadata["assignments"] = [
            {"worker_id": worker_id, "accelerator": "cpu", "selected_device_index": None}
            for worker_id in range(worker_count)
        ]
        return plans, metadata

    candidate_inventory = [
        item for item in inventory if int(item.get("index", -1)) not in excluded_device_indices
    ]
    if not candidate_inventory and inventory:
        candidate_inventory = inventory
        metadata["warnings"].append(
            "all CUDA devices are excluded by preflight reservations; worker embeddings may share reserved devices"
        )
    devices = sorted(
        candidate_inventory,
        key=lambda item: (-int(item.get("free_bytes", 0)), int(item.get("index", 0))),
    )
    load_unit = int(estimated_required_bytes or 1)
    float32_unit = int(estimated_float32_required_bytes or load_unit)
    projected_load_by_device = {int(item["index"]): 0 for item in devices}
    worker_ids_by_device: dict[int, list[int]] = {int(item["index"]): [] for item in devices}
    plans: list[DevicePlan | None] = []

    def projected_ratio(item: dict) -> tuple[float, int]:
        index = int(item["index"])
        denominator = max(int(item.get("free_bytes", 0)), 1)
        return ((projected_load_by_device[index] + load_unit) / denominator, index)

    for worker_id in range(worker_count):
        chosen = min(devices, key=projected_ratio)
        device_index = int(chosen["index"])
        projected_load_by_device[device_index] += load_unit
        worker_ids_by_device[device_index].append(worker_id)
        plans.append(
            DevicePlan(
                device_mode="single",
                accelerator=f"cuda:{device_index}",
                visible_devices=[device_index],
                metadata={
                    "role": "embedding",
                    "worker_id": worker_id,
                    "worker_count": worker_count,
                    "worker_preallocated": True,
                    "requested_model_name": config.embedding_model,
                    "resolved_model_name": resolved_model_name,
                    "estimated_required_bytes": estimated_required_bytes,
                    "estimated_float32_required_bytes": estimated_float32_required_bytes,
                    "model_sharing_enabled": True,
                    "serialized_encode_per_shared_model": True,
                    "selected_device_index": device_index,
                    "selected_device_name": chosen.get("name"),
                    "selected_device_free_bytes": int(chosen.get("free_bytes", 0)),
                    "selected_device_total_bytes": int(chosen.get("total_bytes", 0)),
                    "cuda_inventory": inventory,
                },
            )
        )

    per_device: dict[str, dict] = {}
    warnings_list: list[str] = list(metadata.get("warnings", []))
    for item in devices:
        device_index = int(item["index"])
        worker_ids = worker_ids_by_device[device_index]
        model_instance_count = 1 if worker_ids else 0
        estimated_bf16 = load_unit * model_instance_count if estimated_required_bytes is not None else None
        estimated_float32 = (
            float32_unit * model_instance_count
            if estimated_float32_required_bytes is not None
            else None
        )
        uncached_estimated_bf16 = (
            load_unit * len(worker_ids) if estimated_required_bytes is not None else None
        )
        uncached_estimated_float32 = (
            float32_unit * len(worker_ids)
            if estimated_float32_required_bytes is not None
            else None
        )
        free_bytes = int(item.get("free_bytes", 0))
        total_bytes = int(item.get("total_bytes", 0))
        bf16_ratio = (estimated_bf16 / free_bytes) if estimated_bf16 is not None and free_bytes > 0 else None
        float32_ratio = (
            (estimated_float32 / free_bytes) if estimated_float32 is not None and free_bytes > 0 else None
        )
        if bf16_ratio is not None and bf16_ratio > 0.9:
            warnings_list.append(
                f"embedding bf16 estimate for cuda:{device_index} uses {bf16_ratio:.2f} of free memory"
            )
        if float32_ratio is not None and float32_ratio > 0.9:
            warnings_list.append(
                f"embedding float32 worst-case estimate for cuda:{device_index} uses "
                f"{float32_ratio:.2f} of free memory"
            )
        per_device[str(device_index)] = {
            "device_index": device_index,
            "device_name": item.get("name"),
            "worker_ids": worker_ids,
            "worker_count": len(worker_ids),
            "model_instance_count": model_instance_count,
            "model_sharing_enabled": True,
            "serialized_encode_per_shared_model": True,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "estimated_bf16_bytes": estimated_bf16,
            "estimated_float32_bytes": estimated_float32,
            "uncached_estimated_bf16_bytes": uncached_estimated_bf16,
            "uncached_estimated_float32_bytes": uncached_estimated_float32,
            "estimated_bf16_free_ratio": bf16_ratio,
            "estimated_float32_free_ratio": float32_ratio,
        }

    for plan in plans:
        if plan is None:
            continue
        device_index = int(plan.visible_devices[0])
        plan.metadata["assigned_worker_count_for_device"] = len(worker_ids_by_device[device_index])
        plan.metadata["assigned_worker_ids_for_device"] = list(worker_ids_by_device[device_index])

    metadata.update(
        {
            "enabled": True,
            "reason": "worker_preallocated",
            "model_sharing_enabled": True,
            "serialized_encode_per_shared_model": True,
            "assignments": [
                {
                    "worker_id": worker_id,
                    "accelerator": plan.accelerator if plan is not None else None,
                    "selected_device_index": (
                        plan.metadata.get("selected_device_index") if plan is not None else None
                    ),
                }
                for worker_id, plan in enumerate(plans)
            ],
            "per_device": per_device,
            "warnings": warnings_list,
        }
    )
    return plans, metadata


def _rank_inventory(
    inventory: list[dict], estimated_required_bytes: int | None, excluded: set[int] | None = None
) -> list[dict]:
    excluded = excluded or set()
    filtered = [item for item in inventory if item["index"] not in excluded]
    filtered.sort(
        key=lambda item: (
            0 if estimated_required_bytes is None or item["free_bytes"] >= estimated_required_bytes else 1,
            -item["free_bytes"],
        )
    )
    return filtered


def _single_gpu_plan(
    *,
    role: str,
    model_name: str,
    inventory: list[dict],
    estimated_required_bytes: int | None,
    excluded: set[int] | None = None,
) -> DevicePlan:
    ranked = _rank_inventory(inventory, estimated_required_bytes, excluded)
    if not ranked:
        return DevicePlan(
            device_mode="cpu",
            accelerator="cpu",
            visible_devices=[],
            metadata={
                "role": role,
                "requested_model_name": model_name,
                "estimated_required_bytes": estimated_required_bytes,
                "cuda_inventory": inventory,
            },
        )
    chosen = ranked[0]
    return DevicePlan(
        device_mode="single",
        accelerator=f"cuda:{chosen['index']}",
        visible_devices=[chosen["index"]],
        metadata={
            "role": role,
            "requested_model_name": model_name,
            "estimated_required_bytes": estimated_required_bytes,
            "selected_device_index": chosen["index"],
            "selected_device_name": chosen["name"],
            "selected_device_free_bytes": chosen["free_bytes"],
            "selected_device_total_bytes": chosen["total_bytes"],
            "cuda_inventory": inventory,
            "shared_with": [],
        },
    )


def build_role_device_plans(config) -> dict[str, DevicePlan]:
    inventory = deepcopy(detect_cuda_inventory())
    local_backbone = config.backbone_provider_kind == "local"
    local_judge = config.judge_provider_kind == "local" and bool(config.judge_model)
    embedding_local = config.embedding_model != "hash-embedding"
    used_primary: set[int] = set()

    if config.device_mode == "cpu" or not inventory:
        return {
            "backbone": DevicePlan(
                device_mode="cpu" if local_backbone else "remote",
                accelerator="cpu" if local_backbone else "remote",
                visible_devices=[],
                metadata={"role": "backbone", "cuda_inventory": inventory},
            ),
            "judge": DevicePlan(
                device_mode="cpu" if local_judge else "remote",
                accelerator="cpu" if local_judge else "remote",
                visible_devices=[],
                metadata={"role": "judge", "cuda_inventory": inventory},
            ),
            "embedding": DevicePlan(
                device_mode="cpu",
                accelerator="cpu",
                visible_devices=[],
                metadata={"role": "embedding", "cuda_inventory": inventory},
            ),
        }

    if local_backbone:
        backbone_plan = _single_gpu_plan(
            role="backbone",
            model_name=config.backbone_model,
            inventory=inventory,
            estimated_required_bytes=estimate_model_bytes(config.backbone_model),
        )
        used_primary.update(backbone_plan.visible_devices)
    else:
        backbone_plan = DevicePlan(
            device_mode="remote",
            accelerator="remote",
            visible_devices=[],
            metadata={"role": "backbone", "cuda_inventory": inventory},
        )

    if local_judge:
        judge_excluded = used_primary if len(inventory) > 1 else set()
        judge_plan = _single_gpu_plan(
            role="judge",
            model_name=config.judge_model,
            inventory=inventory,
            estimated_required_bytes=estimate_model_bytes(config.judge_model),
            excluded=judge_excluded,
        )
        used_primary.update(judge_plan.visible_devices)
    else:
        judge_plan = DevicePlan(
            device_mode="remote" if config.judge_model else "disabled",
            accelerator="remote" if config.judge_model else "none",
            visible_devices=[],
            metadata={"role": "judge", "cuda_inventory": inventory},
        )

    if not embedding_local:
        embedding_plan = DevicePlan(
            device_mode="cpu",
            accelerator="cpu",
            visible_devices=[],
            metadata={"role": "embedding", "cuda_inventory": inventory},
        )
    else:
        remaining = [item for item in inventory if item["index"] not in used_primary]
        if remaining:
            embedding_plan = _single_gpu_plan(
                role="embedding",
                model_name=config.embedding_model,
                inventory=inventory,
                estimated_required_bytes=estimate_model_bytes(resolve_hf_model_name(config.embedding_model)),
                excluded=used_primary,
            )
        elif local_judge and judge_plan.visible_devices:
            embedding_plan = DevicePlan(
                device_mode="single",
                accelerator=judge_plan.accelerator,
                visible_devices=list(judge_plan.visible_devices),
                metadata={
                    **judge_plan.metadata,
                    "role": "embedding",
                    "requested_model_name": config.embedding_model,
                    "shared_with": ["judge"],
                },
            )
        elif local_backbone and backbone_plan.visible_devices:
            embedding_plan = DevicePlan(
                device_mode="single",
                accelerator=backbone_plan.accelerator,
                visible_devices=list(backbone_plan.visible_devices),
                metadata={
                    **backbone_plan.metadata,
                    "role": "embedding",
                    "requested_model_name": config.embedding_model,
                    "shared_with": ["backbone"],
                },
            )
        else:
            embedding_plan = build_embedding_device_plan(config.embedding_model, config.device_mode)

    if local_backbone and local_judge and backbone_plan.visible_devices == judge_plan.visible_devices:
        judge_plan.metadata["shared_with"] = ["backbone"]
        backbone_plan.metadata["shared_with"] = ["judge"]
    return {
        "backbone": backbone_plan,
        "judge": judge_plan,
        "embedding": embedding_plan,
    }
