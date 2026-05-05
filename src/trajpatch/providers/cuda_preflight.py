"""CUDA memory preflight planning before local model loads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trajpatch.exceptions import ProviderConfigurationError
from trajpatch.types import DevicePlan

from .devices import detect_cuda_inventory, estimate_model_bytes, resolve_hf_model_name

GIB = 1024**3
DEFAULT_VLLM_MEMORY_UTILIZATION = 0.90
REMOTE_LIKE_PROVIDER_KINDS = {"remote", "openai-compatible"}


@dataclass(slots=True)
class CUDAPreflightReport:
    enabled: bool
    mode: str
    risk: str
    inventory: list[dict[str, Any]] = field(default_factory=list)
    reservations: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suggested_config_changes: list[str] = field(default_factory=list)
    device_plan_overrides: dict[str, DevicePlan | None] = field(default_factory=dict)
    reserved_device_indices: set[int] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "risk": self.risk,
            "inventory": list(self.inventory),
            "reservations": list(self.reservations),
            "assignments": list(self.assignments),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "suggested_config_changes": list(self.suggested_config_changes),
            "reserved_device_indices": sorted(self.reserved_device_indices),
        }


def parse_cuda_visible_devices(value: str | None) -> list[int]:
    if value is None or not str(value).strip():
        return []
    result: list[int] = []
    for token in str(value).split(","):
        stripped = token.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            result.append(int(stripped))
    return result


def _safe_model_bytes(model_name: str | None, *, bytes_per_param: int = 2) -> int | None:
    if not model_name:
        return None
    return estimate_model_bytes(model_name, bytes_per_param=bytes_per_param, overhead=1.2)


def _risk_for_ratio(ratio: float | None, *, has_error: bool = False, has_warning: bool = False) -> str:
    if has_error:
        return "fail"
    if ratio is None:
        return "medium" if has_warning else "low"
    if ratio > 1.0:
        return "fail"
    if ratio >= 0.85:
        return "high"
    if ratio >= 0.60:
        return "medium"
    return "low"


def _merge_risk(left: str, right: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "fail": 3}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _make_remote_plan(role: str, inventory: list[dict[str, Any]]) -> DevicePlan:
    return DevicePlan(
        device_mode="remote",
        accelerator="remote",
        visible_devices=[],
        metadata={"role": role, "cuda_inventory": inventory, "cuda_preflight": True},
    )


def _make_cpu_plan(role: str, model_name: str | None, inventory: list[dict[str, Any]]) -> DevicePlan:
    return DevicePlan(
        device_mode="cpu",
        accelerator="cpu",
        visible_devices=[],
        metadata={
            "role": role,
            "requested_model_name": model_name,
            "cuda_inventory": inventory,
            "cuda_preflight": True,
        },
    )


def _assignment_to_plan(assignment: dict[str, Any], inventory: list[dict[str, Any]]) -> DevicePlan:
    device_index = int(assignment["device_index"])
    return DevicePlan(
        device_mode="single",
        accelerator=f"cuda:{device_index}",
        visible_devices=[device_index],
        metadata={
            "role": assignment["role"],
            "requested_model_name": assignment.get("model_name"),
            "estimated_required_bytes": assignment.get("estimated_required_bytes"),
            "selected_device_index": device_index,
            "selected_device_name": assignment.get("device_name"),
            "selected_device_free_bytes": assignment.get("device_free_bytes"),
            "selected_device_total_bytes": assignment.get("device_total_bytes"),
            "projected_used_bytes": assignment.get("projected_used_bytes"),
            "projected_available_bytes": assignment.get("projected_available_bytes"),
            "projected_free_ratio": assignment.get("projected_free_ratio"),
            "risk": assignment.get("risk"),
            "cuda_inventory": inventory,
            "cuda_preflight": True,
            "shared_with": list(assignment.get("shared_with") or []),
        },
    )


class _Planner:
    def __init__(self, config, *, inventory: list[dict[str, Any]], reserve_gb: float) -> None:
        self.config = config
        self.inventory = [dict(item) for item in inventory]
        self.reserve_bytes = int(float(reserve_gb or 0.0) * GIB)
        self.reserved_by_device = {int(item["index"]): 0 for item in self.inventory}
        self.assigned_bytes_by_device = {int(item["index"]): 0 for item in self.inventory}
        self.assigned_by_device: dict[int, list[str]] = {int(item["index"]): [] for item in self.inventory}
        self.assignments: list[dict[str, Any]] = []
        self.reservations: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.suggestions: list[str] = []
        self.risk = "low"

    def _device_by_index(self, index: int) -> dict[str, Any] | None:
        for item in self.inventory:
            if int(item["index"]) == int(index):
                return item
        return None

    def reserve_vllm(self) -> set[int]:
        if not bool(getattr(self.config, "vllm_autostart", False)):
            return set()
        requested = parse_cuda_visible_devices(getattr(self.config, "vllm_cuda_visible_devices", None))
        if not requested:
            message = (
                "vLLM autostart is enabled but --vllm-cuda-visible-devices is not set; "
                "the server may see all GPUs and conflict with embedding/local models."
            )
            if getattr(self.config, "cuda_preflight_mode", "warn") == "strict":
                self.errors.append(message)
            else:
                self.warnings.append(message)
            self.suggestions.append("Set --vllm-cuda-visible-devices to isolate vLLM from embedding workers.")
            return set()

        utilization = (
            float(getattr(self.config, "vllm_gpu_memory_utilization", 0.0) or 0.0)
            or DEFAULT_VLLM_MEMORY_UTILIZATION
        )
        reserved: set[int] = set()
        for device_index in requested:
            device = self._device_by_index(device_index)
            if device is None:
                message = f"vLLM requested cuda:{device_index}, but it is not visible in CUDA inventory."
                if getattr(self.config, "cuda_preflight_mode", "warn") == "strict":
                    self.errors.append(message)
                else:
                    self.warnings.append(message)
                continue
            total_bytes = int(device.get("total_bytes", 0))
            reservation_bytes = int(total_bytes * utilization)
            self.reserved_by_device[device_index] = max(
                self.reserved_by_device.get(device_index, 0),
                reservation_bytes,
            )
            reserved.add(device_index)
            row = {
                "role": "vllm_server",
                "device_index": device_index,
                "device_name": device.get("name"),
                "reservation_bytes": reservation_bytes,
                "reservation_ratio_of_total": utilization,
                "model_name": getattr(self.config, "vllm_model", None)
                or getattr(self.config, "backbone_model", None),
                "source": "vllm_autostart",
            }
            self.reservations.append(row)
        return reserved

    def assign_role(
        self,
        *,
        role: str,
        model_name: str,
        estimated_required_bytes: int | None,
        prefer_avoid_reserved: bool = True,
    ) -> dict[str, Any] | None:
        if not self.inventory:
            return None
        ranked: list[tuple[float, int, dict[str, Any], int, int]] = []
        for device in self.inventory:
            index = int(device["index"])
            free_bytes = int(device.get("free_bytes", 0))
            reserved_bytes = int(self.reserved_by_device.get(index, 0))
            already_assigned_bytes = int(self.assigned_bytes_by_device.get(index, 0))
            available_bytes = max(free_bytes - reserved_bytes - self.reserve_bytes, 1)
            projected_bytes = int(estimated_required_bytes or 0)
            ratio = (
                (already_assigned_bytes + projected_bytes) / available_bytes
                if estimated_required_bytes is not None
                else 0.5
            )
            reserved_penalty = 1.0 if prefer_avoid_reserved and reserved_bytes > 0 else 0.0
            ranked.append((reserved_penalty, ratio, device, available_bytes, projected_bytes))
        ranked.sort(key=lambda row: (row[0], row[1], -int(row[2].get("free_bytes", 0)), int(row[2]["index"])))
        _, ratio, chosen, available_bytes, projected_bytes = ranked[0]
        device_index = int(chosen["index"])
        self.assigned_by_device[device_index].append(role)
        self.assigned_bytes_by_device[device_index] = (
            int(self.assigned_bytes_by_device.get(device_index, 0)) + projected_bytes
        )
        existing_roles = list(self.assigned_by_device[device_index])
        risk = _risk_for_ratio(ratio if estimated_required_bytes is not None else None)
        if risk == "fail":
            self.errors.append(
                f"{role} estimated memory exceeds available CUDA budget on cuda:{device_index}."
            )
        elif risk == "high":
            self.warnings.append(
                f"{role} estimated memory uses {ratio:.2f} of available CUDA budget on cuda:{device_index}."
            )
        if self.reserved_by_device.get(device_index, 0) > 0:
            self.warnings.append(f"{role} is assigned to cuda:{device_index}, which is reserved for vLLM.")
            self.suggestions.append(
                "Use --vllm-cuda-visible-devices, reduce conv-workers, or move embeddings to CPU/another GPU."
            )
        assignment = {
            "role": role,
            "model_name": model_name,
            "device_index": device_index,
            "device_name": chosen.get("name"),
            "device_free_bytes": int(chosen.get("free_bytes", 0)),
            "device_total_bytes": int(chosen.get("total_bytes", 0)),
            "estimated_required_bytes": estimated_required_bytes,
            "projected_available_bytes": available_bytes,
            "projected_used_bytes": projected_bytes,
            "projected_free_ratio": ratio if estimated_required_bytes is not None else None,
            "risk": risk,
            "shared_with": [item for item in existing_roles if item != role],
        }
        self.assignments.append(assignment)
        self.risk = _merge_risk(self.risk, risk)
        return assignment


def run_cuda_preflight(
    config,
    *,
    inventory: list[dict[str, Any]] | None = None,
    raise_on_strict: bool = True,
) -> CUDAPreflightReport:
    mode = str(getattr(config, "cuda_preflight_mode", "warn") or "warn")
    if mode == "off":
        return CUDAPreflightReport(enabled=False, mode=mode, risk="off")

    cuda_inventory = [dict(item) for item in (detect_cuda_inventory() if inventory is None else inventory)]
    planner = _Planner(
        config,
        inventory=cuda_inventory,
        reserve_gb=float(getattr(config, "cuda_preflight_reserve_gb", 2.0) or 0.0),
    )
    if not cuda_inventory or getattr(config, "device_mode", "auto") == "cpu":
        reason = "no_cuda_inventory" if not cuda_inventory else "device_mode_cpu"
        return CUDAPreflightReport(
            enabled=True,
            mode=mode,
            risk="low",
            inventory=cuda_inventory,
            warnings=[],
            errors=[],
            suggested_config_changes=[reason],
            assignments=[],
            reservations=[],
            device_plan_overrides={},
            reserved_device_indices=set(),
        )

    reserved = planner.reserve_vllm()
    overrides: dict[str, DevicePlan | None] = {}

    if getattr(config, "backbone_provider_kind", None) == "local":
        assignment = planner.assign_role(
            role="local_backbone",
            model_name=getattr(config, "backbone_model", ""),
            estimated_required_bytes=_safe_model_bytes(getattr(config, "backbone_model", None)),
        )
        if assignment is not None:
            overrides["backbone"] = _assignment_to_plan(assignment, cuda_inventory)
    elif getattr(config, "backbone_provider_kind", None) in REMOTE_LIKE_PROVIDER_KINDS:
        overrides["backbone"] = _make_remote_plan("backbone", cuda_inventory)

    if getattr(config, "judge_provider_kind", None) == "local" and getattr(config, "judge_model", None):
        assignment = planner.assign_role(
            role="local_judge",
            model_name=getattr(config, "judge_model", ""),
            estimated_required_bytes=_safe_model_bytes(getattr(config, "judge_model", None)),
        )
        if assignment is not None:
            overrides["judge"] = _assignment_to_plan(assignment, cuda_inventory)
    elif getattr(config, "judge_provider_kind", None) in REMOTE_LIKE_PROVIDER_KINDS:
        overrides["judge"] = _make_remote_plan("judge", cuda_inventory)

    if getattr(config, "embedding_model", None) == "hash-embedding":
        overrides["embedding"] = _make_cpu_plan("embedding", "hash-embedding", cuda_inventory)
    else:
        assignment = planner.assign_role(
            role="embedding_main",
            model_name=getattr(config, "embedding_model", ""),
            estimated_required_bytes=_safe_model_bytes(
                resolve_hf_model_name(getattr(config, "embedding_model", "")),
            ),
        )
        if assignment is not None:
            overrides["embedding"] = _assignment_to_plan(assignment, cuda_inventory)

    worker_count = max(int(getattr(config, "conv_workers", 1) or 1), 1)
    if worker_count > 1 and getattr(config, "embedding_model", None) != "hash-embedding":
        estimated_embedding = _safe_model_bytes(resolve_hf_model_name(getattr(config, "embedding_model", "")))
        for worker_id in range(worker_count):
            planner.assign_role(
                role=f"embedding_worker_{worker_id:02d}",
                model_name=getattr(config, "embedding_model", ""),
                estimated_required_bytes=estimated_embedding,
            )

    try:
        batch_size = (
            1
            if getattr(config, "memory_extract_batch_size", "auto") == "auto"
            else int(getattr(config, "memory_extract_batch_size", 1))
        )
    except Exception:
        batch_size = 1
    try:
        judge_concurrency = (
            2
            if getattr(config, "judge_max_concurrency", "auto") == "auto"
            and getattr(config, "judge_provider_kind", None) in REMOTE_LIKE_PROVIDER_KINDS
            else int(getattr(config, "judge_max_concurrency", 1))
            if getattr(config, "judge_max_concurrency", "auto") != "auto"
            else 1
        )
    except Exception:
        judge_concurrency = 1
    estimated_inflight = worker_count * max(batch_size, 1) + max(judge_concurrency, 1)
    if getattr(config, "backbone_provider_kind", None) == "openai-compatible" and estimated_inflight > 16:
        planner.warnings.append(
            f"Estimated openai-compatible in-flight pressure is {estimated_inflight}; vLLM queue/GPU memory may be stressed."
        )
        planner.suggestions.append("Reduce conv-workers, memory-extract-batch-size, or judge-max-concurrency.")

    risk = planner.risk
    if planner.errors:
        risk = "fail"
    elif planner.warnings:
        risk = _merge_risk(risk, "medium")

    report = CUDAPreflightReport(
        enabled=True,
        mode=mode,
        risk=risk,
        inventory=cuda_inventory,
        reservations=planner.reservations,
        assignments=planner.assignments,
        warnings=list(dict.fromkeys(planner.warnings)),
        errors=list(dict.fromkeys(planner.errors)),
        suggested_config_changes=list(dict.fromkeys(planner.suggestions)),
        device_plan_overrides=overrides,
        reserved_device_indices=reserved,
    )
    if raise_on_strict and mode == "strict" and (report.errors or report.risk == "fail"):
        raise ProviderConfigurationError(
            "CUDA preflight failed: " + "; ".join(report.errors or report.warnings)
        )
    return report
