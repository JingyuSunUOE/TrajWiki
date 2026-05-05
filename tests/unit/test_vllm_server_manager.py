from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from trajpatch.config import RunConfig
from trajpatch.exceptions import ProviderConfigurationError
from trajpatch.providers.vllm_server import ManagedVLLMServer


class _Response:
    status = 200

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FakeProcess:
    pid = 4321

    def __init__(self, returncode=None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _models_payload(*model_ids: str) -> dict:
    return {"data": [{"id": model_id} for model_id in model_ids]}


def test_run_config_rejects_vllm_autostart_without_openai_compatible_provider(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="openai-compatible"):
        RunConfig(
            dataset="medmt",
            dataset_path=dataset_path,
            provider_kind="remote",
            backbone_model="gpt-4o-mini",
            vllm_autostart=True,
        )


def test_run_config_autostart_defaults_base_url_to_loopback_port(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        provider_kind="openai-compatible",
        backbone_model="qwen3-8b",
        vllm_autostart=True,
        vllm_port=8123,
    )

    assert config.openai_compatible_base_url == "http://127.0.0.1:8123/v1"
    assert config.openai_compatible_api_key == "EMPTY"


def test_managed_vllm_uses_loopback_client_url_when_binding_all_interfaces():
    manager = ManagedVLLMServer(
        base_url="http://127.0.0.1:8000/v1",
        autostart=True,
        model="Qwen/Qwen3-8B",
        served_model_name="qwen3-8b",
        host="0.0.0.0",
        port=8000,
    )

    assert manager.client_host == "127.0.0.1"
    assert manager.effective_base_url == "http://127.0.0.1:8000/v1"


def test_managed_vllm_reuses_existing_server_without_starting_process(monkeypatch, tmp_path):
    popen_calls = []
    monkeypatch.setattr(
        "trajpatch.providers.vllm_server.urlopen",
        lambda request, timeout: _Response(_models_payload("qwen3-8b")),
    )

    manager = ManagedVLLMServer(
        base_url="http://127.0.0.1:8000/v1",
        autostart=True,
        model="Qwen/Qwen3-8B",
        served_model_name="qwen3-8b",
        status_dir=tmp_path,
        popen_factory=lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )
    manager.start()
    manager.stop()

    assert manager.reused_existing_server is True
    assert manager.started_by_runner is False
    assert popen_calls == []
    assert (tmp_path / "vllm_server.json").exists()


def test_managed_vllm_launches_process_with_expected_command_and_env(monkeypatch, tmp_path):
    responses = [
        URLError("connection refused"),
        _Response(_models_payload("qwen3-8b")),
    ]
    popen_calls = []
    process = _FakeProcess()

    def fake_urlopen(request, timeout):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    monkeypatch.setattr("trajpatch.providers.vllm_server.urlopen", fake_urlopen)
    manager = ManagedVLLMServer(
        base_url="http://127.0.0.1:8000/v1",
        autostart=True,
        model="Qwen/Qwen3-8B",
        served_model_name="qwen3-8b",
        host="0.0.0.0",
        port=8000,
        cuda_visible_devices="0",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        extra_args="--max-model-len 8192",
        status_dir=tmp_path,
        popen_factory=fake_popen,
        poll_interval_s=0.01,
    )

    manager.start()
    manager.stop()

    command, kwargs = popen_calls[0]
    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert "--host" in command and "0.0.0.0" in command
    assert "--port" in command and "8000" in command
    assert "--served-model-name" in command and "qwen3-8b" in command
    assert "--tensor-parallel-size" in command and "2" in command
    assert "--gpu-memory-utilization" in command and "0.8" in command
    assert "--dtype" in command and "bfloat16" in command
    assert "--max-model-len" in command and "8192" in command
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert manager.started_by_runner is True
    assert process.terminated is True


def test_managed_vllm_wrong_served_model_raises_without_closing_existing_service(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "trajpatch.providers.vllm_server.urlopen",
        lambda request, timeout: _Response(_models_payload("other-model")),
    )
    manager = ManagedVLLMServer(
        base_url="http://127.0.0.1:8000/v1",
        autostart=True,
        model="Qwen/Qwen3-8B",
        served_model_name="qwen3-8b",
        status_dir=tmp_path,
        popen_factory=lambda *args, **kwargs: pytest.fail("should not start process"),
    )

    with pytest.raises(ProviderConfigurationError, match="qwen3-8b"):
        manager.start()

    assert manager.started_by_runner is False
    assert (tmp_path / "vllm_server.json").exists()


def test_managed_vllm_keep_alive_does_not_terminate_started_process(monkeypatch, tmp_path):
    responses = [
        URLError("connection refused"),
        _Response(_models_payload("qwen3-8b")),
    ]
    process = _FakeProcess()

    def fake_urlopen(request, timeout):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("trajpatch.providers.vllm_server.urlopen", fake_urlopen)
    manager = ManagedVLLMServer(
        base_url="http://127.0.0.1:8000/v1",
        autostart=True,
        model="Qwen/Qwen3-8B",
        served_model_name="qwen3-8b",
        keep_alive=True,
        status_dir=tmp_path,
        popen_factory=lambda *args, **kwargs: process,
        poll_interval_s=0.01,
    )

    manager.start()
    manager.stop()

    assert process.terminated is False
    assert process.killed is False


def test_managed_vllm_early_exit_error_includes_log_tail(monkeypatch, tmp_path):
    monkeypatch.setattr("trajpatch.providers.vllm_server.urlopen", lambda request, timeout: (_ for _ in ()).throw(URLError("connection refused")))
    process = _FakeProcess(returncode=1)

    def fake_popen(command, **kwargs):
        kwargs["stdout"].write(b"RuntimeError: CUDA out of memory while loading model\n")
        kwargs["stdout"].flush()
        return process

    manager = ManagedVLLMServer(
        base_url="http://127.0.0.1:8000/v1",
        autostart=True,
        model="Qwen/Qwen3-8B",
        served_model_name="qwen3-8b",
        status_dir=tmp_path,
        popen_factory=fake_popen,
        poll_interval_s=0.01,
    )

    with pytest.raises(ProviderConfigurationError, match="CUDA out of memory"):
        manager.start()

    payload = json.loads((tmp_path / "vllm_server.json").read_text(encoding="utf-8"))
    assert "CUDA out of memory" in payload["vllm_failure_log_tail"]
