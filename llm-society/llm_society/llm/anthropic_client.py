"""Anthropic Messages API adapter (official `anthropic` SDK, optional dependency).

Notes relevant to experimental control:

* Several current Claude models reject sampling parameters (`temperature`,
  `top_p`) with a 400. For those models the adapter omits them and records
  the note `temperature_not_sent`, so it is visible in the raw logs that the
  configured temperature (including the probe temperature of 0) was *not*
  applied. Override detection with `model.supports_temperature`.
* Models that think by default spend output tokens on thinking, so
  `max_output_tokens` must leave room for it. Use `model.extra_params`
  (e.g. `{"output_config": {"effort": "low"}}`) to control effort.
* A `refusal` stop reason is returned as data (empty text, finish_reason
  "refusal"), never silently regenerated.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import GenerationConfig, LLMResponse, Message, PermanentLLMError, TransientLLMError

# Model-id prefixes that reject sampling parameters.
NO_SAMPLING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
)


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
        supports_temperature: bool | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise PermanentLLMError("the anthropic provider needs `pip install llm-society[anthropic]`") from exc
        self._anthropic = anthropic
        self.model = model
        kwargs: dict[str, Any] = {"timeout": timeout_s, "max_retries": 0}  # retries handled by the engine
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise PermanentLLMError(f"environment variable {api_key_env} is not set")
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        if supports_temperature is None:
            supports_temperature = not model.startswith(NO_SAMPLING_PREFIXES)
        self.supports_temperature = supports_temperature
        self.extra_params = dict(extra_params or {})

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "supports_temperature": self.supports_temperature,
            "extra_params": self.extra_params,
        }

    async def generate(
        self, system_prompt: str, messages: list[Message], generation_config: GenerationConfig
    ) -> LLMResponse:
        a = self._anthropic
        notes: list[str] = []
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": generation_config.max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if generation_config.temperature is not None:
            if self.supports_temperature:
                body["temperature"] = generation_config.temperature
            else:
                notes.append("temperature_not_sent")
        if generation_config.top_p is not None and self.supports_temperature:
            body["top_p"] = generation_config.top_p
        if generation_config.seed is not None:
            notes.append("seed_not_supported")
        body.update(self.extra_params)
        body.update(generation_config.extra)

        t0 = time.monotonic()
        try:
            resp = await self._client.messages.create(**body)
        except (a.RateLimitError, a.InternalServerError, a.APITimeoutError, a.APIConnectionError) as exc:
            raise TransientLLMError(f"{type(exc).__name__}: {exc}") from exc
        except a.APIStatusError as exc:
            if exc.status_code in (408, 409, 429, 529) or exc.status_code >= 500:
                raise TransientLLMError(f"HTTP {exc.status_code}: {exc.message}") from exc
            raise PermanentLLMError(f"HTTP {exc.status_code}: {exc.message}") from exc
        latency = time.monotonic() - t0

        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=resp.model,
            provider=self.provider,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            finish_reason=resp.stop_reason,
            latency_s=latency,
            request=body,
            raw=resp.model_dump(mode="json"),
            notes=notes,
        )
