"""Factories for runtime LLM and embedding providers."""

from __future__ import annotations

from trajpatch.config import ProviderKind, RunConfig
from trajpatch.exceptions import ProviderConfigurationError
from trajpatch.types import DevicePlan
from trajpatch.utils.env import load_runtime_env

from .base import EmbeddingProvider, LLMProvider
from .devices import build_role_device_plans
from .litellm_provider import LiteLLMProvider
from .mock import HashEmbeddingProvider, MockLLMProvider
from .metering import MeteredLLMProvider
from .openai_compatible_provider import OpenAICompatibleProvider
from .transformers_provider import SentenceTransformerEmbeddingProvider, TransformersLLMProvider


def build_llm_provider(
    config: RunConfig,
    *,
    role: str = "backbone",
    model_name: str | None = None,
    provider_kind: ProviderKind | None = None,
    device_plan=None,
    metered: bool = True,
) -> LLMProvider:
    load_runtime_env(override=True)
    chosen_model = model_name or (config.backbone_model if role == "backbone" else config.judge_model)
    chosen_provider_kind = provider_kind or (
        config.backbone_provider_kind if role == "backbone" else config.judge_provider_kind
    )
    if chosen_model is None:
        raise ProviderConfigurationError(f"Missing model name for role: {role}")
    if chosen_provider_kind == "mock":
        provider: LLMProvider = MockLLMProvider(model_name=chosen_model)
    elif chosen_provider_kind == "remote":
        provider = LiteLLMProvider(chosen_model)
    elif chosen_provider_kind == "openai-compatible":
        provider = OpenAICompatibleProvider(
            chosen_model,
            base_url=config.openai_compatible_base_url,
            api_key=config.openai_compatible_api_key,
            structured_mode=config.openai_compatible_structured_mode,
        )
    elif chosen_provider_kind == "local":
        provider = TransformersLLMProvider(chosen_model, device_mode=config.device_mode, device_plan=device_plan)
    else:
        raise ProviderConfigurationError(f"Unsupported provider kind: {chosen_provider_kind}")
    return MeteredLLMProvider(provider, role=role) if metered else provider


def build_embedding_provider(config: RunConfig, *, device_plan=None) -> EmbeddingProvider:
    load_runtime_env(override=True)
    if config.embedding_model == "hash-embedding":
        return HashEmbeddingProvider(model_name=config.embedding_model)
    if config.backbone_provider_kind in {"local", "remote", "mock", "openai-compatible"}:
        return SentenceTransformerEmbeddingProvider(
            config.embedding_model, device_mode=config.device_mode, device_plan=device_plan
        )
    raise ProviderConfigurationError(
        f"Cannot build embedding provider for {config.backbone_provider_kind}/{config.embedding_model}"
    )


def build_provider_bundle(
    config: RunConfig,
    *,
    device_plan_overrides: dict[str, DevicePlan | None] | None = None,
) -> tuple[LLMProvider, LLMProvider, EmbeddingProvider, dict]:
    device_plans = build_role_device_plans(config)
    for role, plan in (device_plan_overrides or {}).items():
        if plan is not None:
            device_plans[role] = plan
    backbone = build_llm_provider(
        config,
        role="backbone",
        provider_kind=config.backbone_provider_kind,
        model_name=config.backbone_model,
        device_plan=device_plans["backbone"],
    )
    judge = build_llm_provider(
        config,
        role="judge",
        provider_kind=config.judge_provider_kind,
        model_name=config.judge_model,
        device_plan=device_plans["judge"],
    )
    embedding = build_embedding_provider(config, device_plan=device_plans["embedding"])
    return backbone, judge, embedding, {
        role: {
            "device_mode": plan.device_mode,
            "accelerator": plan.accelerator,
            "visible_devices": plan.visible_devices,
            "metadata": plan.metadata,
        }
        for role, plan in device_plans.items()
    }
