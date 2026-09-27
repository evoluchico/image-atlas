"""LLM abstraction and provider adapters."""

from __future__ import annotations

from ..config.schema import ModelConfig
from .base import (
    GenerationConfig,
    LLMClient,
    LLMResponse,
    Message,
    PermanentLLMError,
    TransientLLMError,
    generate_with_retry,
)


def create_client(cfg: ModelConfig) -> LLMClient:
    """Instantiate the adapter for one model configuration."""
    common = dict(
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        timeout_s=cfg.timeout_s,
        supports_temperature=cfg.supports_temperature,
        extra_params=cfg.extra_params,
    )
    if cfg.provider == "mock":
        from .mock_client import MockLLMClient

        return MockLLMClient(model=cfg.model, **cfg.extra_params)
    if cfg.provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(cfg.model, **common)
    if cfg.provider == "local":
        from .local_client import LocalClient

        return LocalClient(cfg.model, **common)
    if cfg.provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(cfg.model, **common)
    raise ValueError(f"unknown provider {cfg.provider!r}")


__all__ = [
    "GenerationConfig",
    "LLMClient",
    "LLMResponse",
    "Message",
    "PermanentLLMError",
    "TransientLLMError",
    "create_client",
    "generate_with_retry",
]
