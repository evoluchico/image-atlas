"""OpenAI Chat Completions adapter (also the base for OpenAI-compatible servers)."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import GenerationConfig, LLMResponse, Message, PermanentLLMError, TransientLLMError

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class OpenAIClient:
    provider = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_api_key_env = "OPENAI_API_KEY"
    # Newer OpenAI models use max_completion_tokens; OpenAI-compatible servers use max_tokens.
    max_tokens_field = "max_completion_tokens"
    api_key_required = True

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
        supports_temperature: bool | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.api_key_env = api_key_env or self.default_api_key_env
        self.api_key = os.environ.get(self.api_key_env)
        if self.api_key_required and not self.api_key:
            raise PermanentLLMError(f"environment variable {self.api_key_env} is not set")
        self.timeout_s = timeout_s
        self.supports_temperature = True if supports_temperature is None else supports_temperature
        self.extra_params = dict(extra_params or {})
        self._client: httpx.AsyncClient | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "supports_temperature": self.supports_temperature,
            "extra_params": self.extra_params,
        }

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout_s)
        return self._client

    async def generate(
        self, system_prompt: str, messages: list[Message], generation_config: GenerationConfig
    ) -> LLMResponse:
        notes: list[str] = []
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}]
            + [{"role": m.role, "content": m.content} for m in messages],
            self.max_tokens_field: generation_config.max_output_tokens,
        }
        if generation_config.temperature is not None:
            if self.supports_temperature:
                body["temperature"] = generation_config.temperature
            else:
                notes.append("temperature_not_sent")
        if generation_config.top_p is not None and self.supports_temperature:
            body["top_p"] = generation_config.top_p
        if generation_config.seed is not None:
            body["seed"] = generation_config.seed
        body.update(self.extra_params)
        body.update(generation_config.extra)

        t0 = time.monotonic()
        try:
            r = await self._http().post("/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientLLMError(f"{type(exc).__name__}: {exc}") from exc
        latency = time.monotonic() - t0
        if r.status_code in TRANSIENT_STATUS:
            raise TransientLLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise PermanentLLMError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            provider=self.provider,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            latency_s=latency,
            request=body,
            raw=data,
            notes=notes,
        )
