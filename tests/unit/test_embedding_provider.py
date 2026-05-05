from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

from trajpatch.config import RunConfig
from trajpatch.providers.factory import build_embedding_provider
from trajpatch.providers.transformers_provider import SentenceTransformerEmbeddingProvider
from trajpatch.types import DevicePlan


def _clear_embedding_model_cache() -> None:
    with SentenceTransformerEmbeddingProvider._MODEL_LOAD_LOCK:
        SentenceTransformerEmbeddingProvider._MODEL_CACHE.clear()
        SentenceTransformerEmbeddingProvider._MODEL_RUNTIME_LOCKS.clear()


def test_build_embedding_provider_uses_sentence_transformer_for_remote_local_and_openai_compatible(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    remote_config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "remote-artifacts",
        database_path=tmp_path / "remote.sqlite",
        provider_kind="remote",
        embedding_model="huggingface/Qwen3-Embedding-8B",
    )
    local_config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "local-artifacts",
        database_path=tmp_path / "local.sqlite",
        provider_kind="local",
        embedding_model="huggingface/Qwen3-Embedding-8B",
    )
    openai_compatible_config = RunConfig(
        dataset="medmt",
        dataset_path=dataset_path,
        output_dir=tmp_path / "openai-compatible-artifacts",
        database_path=tmp_path / "openai-compatible.sqlite",
        provider_kind="openai-compatible",
        embedding_model="huggingface/Qwen3-Embedding-8B",
    )

    remote_provider = build_embedding_provider(remote_config)
    local_provider = build_embedding_provider(local_config)
    openai_compatible_provider = build_embedding_provider(openai_compatible_config)

    assert isinstance(remote_provider, SentenceTransformerEmbeddingProvider)
    assert isinstance(local_provider, SentenceTransformerEmbeddingProvider)
    assert isinstance(openai_compatible_provider, SentenceTransformerEmbeddingProvider)
    assert remote_provider.resolved_model_name == "Qwen/Qwen3-Embedding-8B"
    assert local_provider.resolved_model_name == "Qwen/Qwen3-Embedding-8B"
    assert openai_compatible_provider.resolved_model_name == "Qwen/Qwen3-Embedding-8B"


def test_embedding_provider_picks_gpu_with_most_free_memory(monkeypatch):
    monkeypatch.setattr(
        "trajpatch.providers.devices.detect_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "GPU-0",
                "total_bytes": 16 * 1024**3,
                "free_bytes": 8 * 1024**3,
                "used_bytes": 8 * 1024**3,
                "source": "test",
            },
            {
                "index": 1,
                "name": "GPU-1",
                "total_bytes": 24 * 1024**3,
                "free_bytes": 20 * 1024**3,
                "used_bytes": 4 * 1024**3,
                "source": "test",
            },
        ],
    )

    provider = SentenceTransformerEmbeddingProvider("huggingface/Qwen3-Embedding-8B")

    assert provider.device_plan.accelerator == "cuda:1"
    assert provider.device_plan.metadata["selected_device_index"] == 1
    assert provider.device_plan.metadata["requested_model_name"] == "huggingface/Qwen3-Embedding-8B"
    assert provider.device_plan.metadata["resolved_model_name"] == "Qwen/Qwen3-Embedding-8B"


def test_embedding_provider_serializes_sentence_transformer_loads(monkeypatch):
    _clear_embedding_model_cache()
    active_loads = 0
    load_count = 0
    max_active_loads = 0
    lock = threading.Lock()

    class FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            nonlocal active_loads, load_count, max_active_loads
            with lock:
                active_loads += 1
                load_count += 1
                max_active_loads = max(max_active_loads, active_loads)
            time.sleep(0.01)
            with lock:
                active_loads -= 1

        def encode(self, texts, normalize_embeddings=True, **_kwargs):
            assert normalize_embeddings is True
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    providers = [
        SentenceTransformerEmbeddingProvider(
            "Qwen/Qwen3-Embedding-8B",
            device_plan=DevicePlan(device_mode="single", accelerator="cpu"),
        )
        for _ in range(4)
    ]
    threads = [threading.Thread(target=provider._ensure_model) for provider in providers]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert load_count == 1
    assert max_active_loads == 1
    assert len({id(provider._model) for provider in providers}) == 1
    _clear_embedding_model_cache()


def test_embedding_provider_requests_bfloat16_for_cuda_load(monkeypatch):
    _clear_embedding_model_cache()
    captured_kwargs = []

    class FakeSentenceTransformer:
        def __init__(self, *_args, **kwargs):
            captured_kwargs.append(kwargs)

        def encode(self, texts, normalize_embeddings=True, **_kwargs):
            assert normalize_embeddings is True
            return [[1.0, 0.0] for _ in texts]

    fake_torch = SimpleNamespace(
        bfloat16="torch.bfloat16",
        float16="torch.float16",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    provider = SentenceTransformerEmbeddingProvider(
        "Qwen/Qwen3-Embedding-8B",
        device_plan=DevicePlan(
            device_mode="single",
            accelerator="cuda:0",
            visible_devices=[0],
            metadata={},
        ),
    )

    provider._ensure_model()

    assert captured_kwargs[0]["model_kwargs"]["torch_dtype"] == "torch.bfloat16"
    assert provider.device_plan.metadata["requested_torch_dtype"] == "bfloat16"
    _clear_embedding_model_cache()
