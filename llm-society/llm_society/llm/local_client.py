"""Adapter for local OpenAI-compatible servers (vLLM, llama.cpp server, Ollama, LM Studio)."""

from __future__ import annotations

from .openai_client import OpenAIClient


class LocalClient(OpenAIClient):
    provider = "local"
    default_base_url = "http://localhost:8000/v1"
    default_api_key_env = "LOCAL_LLM_API_KEY"
    max_tokens_field = "max_tokens"
    api_key_required = False
