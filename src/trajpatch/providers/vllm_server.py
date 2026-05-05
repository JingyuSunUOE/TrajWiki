"""Optional lifecycle manager for a local vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from trajpatch.exceptions import ProviderConfigurationError
from trajpatch.utils.json_utils import write_json


EventCallback = Callable[[str, dict[str, Any]], None]
TraceCallback = Callable[[str], None]


@dataclass(slots=True)
class ManagedVLLMServer:
    """Start vLLM only when explicitly requested and only stop what we started."""

    base_url: str
    autostart: bool
    model: str
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8000
    cuda_visible_devices: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    extra_args: str | None = None
    startup_timeout_s: int = 600
    keep_alive: bool = False
    status_dir: Path | None = None
    preflight_reservation: dict | None = None
    trace: TraceCallback | None = None
    event_callback: EventCallback | None = None
    popen_factory: Any = subprocess.Popen
    poll_interval_s: float = 1.0
    request_timeout_s: float = 2.0

    process: subprocess.Popen | None = field(default=None, init=False)
    reused_existing_server: bool = field(default=False, init=False)
    started_by_runner: bool = field(default=False, init=False)
    ready: bool = field(default=False, init=False)
    startup_latency_ms: float | None = field(default=None, init=False)
    failure_reason: str | None = field(default=None, init=False)
    failure_log_tail: str | None = field(default=None, init=False)
    log_path: Path | None = field(default=None, init=False)
    status_path: Path | None = field(default=None, init=False)
    _log_handle: Any | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "ManagedVLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"

    @property
    def client_host(self) -> str:
        return "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host

    @property
    def effective_base_url(self) -> str:
        return f"http://{self.client_host}:{self.port}/v1"

    def start(self) -> None:
        started_at = time.perf_counter()
        self._prepare_paths()
        self._event("vllm_autostart_check", base_url=self.base_url, models_url=self.models_url)
        self._trace(f"vllm_autostart_check base_url={self.base_url} models_url={self.models_url}")
        existing = self._fetch_models()
        if existing.available:
            self._validate_served_model(existing.model_ids, source="existing")
            self.reused_existing_server = True
            self.ready = True
            self.startup_latency_ms = (time.perf_counter() - started_at) * 1000.0
            self._event(
                "vllm_autostart_reuse_existing",
                base_url=self.base_url,
                served_model_name=self.served_model_name,
                model_ids=existing.model_ids,
                startup_latency_ms=self.startup_latency_ms,
            )
            self._trace(
                "vllm_autostart_reuse_existing "
                f"base_url={self.base_url} served_model_name={self.served_model_name}"
            )
            self._write_status()
            return
        if not self.autostart:
            self.failure_reason = existing.error or "vLLM server is not reachable and autostart is disabled."
            self._write_status()
            raise ProviderConfigurationError(self.failure_reason)
        self._launch_process()
        deadline = time.perf_counter() + float(self.startup_timeout_s)
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self.failure_log_tail = self._read_log_tail()
                self.failure_reason = self._failure_with_log_tail(
                    f"vLLM server exited before readiness check completed with code {self.process.returncode}."
                )
                self._event("vllm_autostart_failed", error_message=self.failure_reason)
                self._trace(f"vllm_autostart_failed error={self.failure_reason}")
                self._write_status()
                raise ProviderConfigurationError(self.failure_reason)
            models = self._fetch_models()
            if models.available:
                self._validate_served_model(models.model_ids, source="started")
                self.ready = True
                self.startup_latency_ms = (time.perf_counter() - started_at) * 1000.0
                self._event(
                    "vllm_autostart_ready",
                    base_url=self.base_url,
                    served_model_name=self.served_model_name,
                    pid=self.process.pid if self.process is not None else None,
                    model_ids=models.model_ids,
                    startup_latency_ms=self.startup_latency_ms,
                )
                self._trace(
                    "vllm_autostart_ready "
                    f"base_url={self.base_url} served_model_name={self.served_model_name} "
                    f"latency_ms={self.startup_latency_ms:.1f}"
                )
                self._write_status()
                return
            time.sleep(max(float(self.poll_interval_s), 0.05))
        self.failure_reason = (
            f"Timed out after {self.startup_timeout_s}s waiting for vLLM server at {self.models_url}."
        )
        self.failure_log_tail = self._read_log_tail()
        self.failure_reason = self._failure_with_log_tail(self.failure_reason)
        self._event("vllm_autostart_failed", error_message=self.failure_reason)
        self._trace(f"vllm_autostart_failed error={self.failure_reason}")
        self._write_status()
        raise ProviderConfigurationError(self.failure_reason)

    def stop(self) -> None:
        if self.process is None:
            self._close_log_handle()
            return
        if self.keep_alive:
            self._event(
                "vllm_autostart_stop",
                action="kept_alive",
                pid=self.process.pid,
                base_url=self.base_url,
            )
            self._trace(f"vllm_autostart_stop action=kept_alive pid={self.process.pid}")
            self._write_status()
            self._close_log_handle()
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
                action = "terminated"
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
                action = "killed"
        else:
            action = "already_exited"
        self._event(
            "vllm_autostart_stop",
            action=action,
            pid=self.process.pid,
            returncode=self.process.returncode,
            base_url=self.base_url,
        )
        self._trace(f"vllm_autostart_stop action={action} pid={self.process.pid}")
        self._write_status()
        self._close_log_handle()

    def metadata(self) -> dict[str, Any]:
        return {
            "vllm_autostart": bool(self.autostart),
            "vllm_reused_existing_server": bool(self.reused_existing_server),
            "vllm_started_by_runner": bool(self.started_by_runner),
            "vllm_model": self.model,
            "vllm_served_model_name": self.served_model_name,
            "vllm_base_url": self.base_url,
            "vllm_pid": self.process.pid if self.process is not None else None,
            "vllm_startup_latency_ms": self.startup_latency_ms,
            "vllm_keep_alive": bool(self.keep_alive),
            "cuda_preflight_reservation": dict(self.preflight_reservation or {}),
            "vllm_ready": bool(self.ready),
            "vllm_failure_reason": self.failure_reason,
            "vllm_failure_log_tail": self.failure_log_tail,
            "vllm_log_path": str(self.log_path) if self.log_path is not None else None,
            "vllm_status_path": str(self.status_path) if self.status_path is not None else None,
        }

    def _prepare_paths(self) -> None:
        if self.status_dir is None:
            return
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.status_dir / "vllm_server.log"
        self.status_path = self.status_dir / "vllm_server.json"

    def _launch_process(self) -> None:
        command = [
            "vllm",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(int(self.port)),
            "--served-model-name",
            self.served_model_name,
        ]
        if self.tensor_parallel_size and int(self.tensor_parallel_size) > 1:
            command.extend(["--tensor-parallel-size", str(int(self.tensor_parallel_size))])
        if self.gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization", str(float(self.gpu_memory_utilization))])
        if self.dtype:
            command.extend(["--dtype", str(self.dtype)])
        if self.extra_args:
            command.extend(shlex.split(self.extra_args))
        env = os.environ.copy()
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        stdout = subprocess.DEVNULL
        if self.log_path is not None:
            self._log_handle = self.log_path.open("ab")
            stdout = self._log_handle
        self.process = self.popen_factory(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self.started_by_runner = True
        self._event(
            "vllm_autostart_start",
            model=self.model,
            served_model_name=self.served_model_name,
            host=self.host,
            port=self.port,
            base_url=self.base_url,
            cuda_visible_devices=self.cuda_visible_devices,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            pid=self.process.pid,
            log_path=str(self.log_path) if self.log_path is not None else None,
        )
        self._trace(
            "vllm_autostart_start "
            f"model={self.model} served_model_name={self.served_model_name} "
            f"host={self.host} port={self.port} pid={self.process.pid}"
        )

    def _fetch_models(self) -> "_ModelsCheck":
        request = Request(self.models_url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=float(self.request_timeout_s)) as response:
                status = int(getattr(response, "status", 0) or 0)
                if status != 200:
                    return _ModelsCheck(False, [], f"models endpoint returned HTTP {status}")
                payload = json.loads(response.read().decode("utf-8"))
            return _ModelsCheck(True, self._model_ids_from_payload(payload), None)
        except HTTPError as exc:
            return _ModelsCheck(False, [], f"models endpoint returned HTTP {exc.code}")
        except URLError as exc:
            return _ModelsCheck(False, [], str(exc.reason))
        except Exception as exc:  # noqa: BLE001
            return _ModelsCheck(False, [], f"{exc.__class__.__name__}: {exc}")

    @staticmethod
    def _model_ids_from_payload(payload: Any) -> list[str]:
        data = payload.get("data", []) if isinstance(payload, dict) else []
        ids: list[str] = []
        for item in data:
            if isinstance(item, dict) and item.get("id") is not None:
                ids.append(str(item["id"]))
            elif getattr(item, "id", None) is not None:
                ids.append(str(getattr(item, "id")))
        return ids

    def _validate_served_model(self, model_ids: list[str], *, source: str) -> None:
        if self.served_model_name and self.served_model_name not in model_ids:
            self.failure_reason = (
                f"vLLM {source} server is reachable at {self.base_url}, but served model "
                f"{self.served_model_name!r} was not found in /models: {model_ids}."
            )
            self._event(
                "vllm_autostart_failed",
                error_message=self.failure_reason,
                model_ids=model_ids,
                source=source,
            )
            self._trace(f"vllm_autostart_failed error={self.failure_reason}")
            self._write_status()
            raise ProviderConfigurationError(self.failure_reason)

    def _write_status(self) -> None:
        if self.status_path is None:
            return
        write_json(self.status_path, self.metadata())

    def _read_log_tail(self, *, byte_limit: int = 4000) -> str | None:
        if self.log_path is None or not self.log_path.exists():
            return None
        try:
            if self._log_handle is not None:
                self._log_handle.flush()
            with self.log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - int(byte_limit)))
                data = handle.read()
        except OSError:
            return None
        text = data.decode("utf-8", errors="replace").strip()
        return text or None

    def _failure_with_log_tail(self, reason: str) -> str:
        if not self.failure_log_tail:
            return reason
        return (
            f"{reason} See {self.log_path} for full logs. "
            f"Recent vLLM log tail: {self.failure_log_tail[-1200:]}"
        )

    def _event(self, event_type: str, **payload: Any) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, payload)

    def _trace(self, message: str) -> None:
        if self.trace is not None:
            self.trace(message)

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None


@dataclass(slots=True)
class _ModelsCheck:
    available: bool
    model_ids: list[str]
    error: str | None
