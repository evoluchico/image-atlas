"""Provider-independent LLM interface.

The simulation engine only ever talks to `LLMClient`. Provider adapters live
in sibling modules and translate to/from their wire formats.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class GenerationConfig:
    temperature: float | None
    max_output_tokens: int
    seed: int | None = None
    top_p: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    # Request body as sent (credentials never included) and raw provider response.
    request: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    # Adapter notes, e.g. "temperature_not_sent" when a model rejects sampling params.
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMResponse":
        return cls(**d)


class TransientLLMError(Exception):
    """Timeouts, rate limits, overloaded / 5xx: safe to retry."""


class PermanentLLMError(Exception):
    """Bad request, auth failure, unknown model: retrying will not help."""


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        generation_config: GenerationConfig,
    ) -> LLMResponse: ...

    def describe(self) -> dict[str, Any]: ...


async def generate_with_retry(
    client: LLMClient,
    system_prompt: str,
    messages: list[Message],
    generation_config: GenerationConfig,
    *,
    max_retries: int = 6,
    backoff_base_s: float = 2.0,
    backoff_max_s: float = 60.0,
    on_retry=None,
) -> LLMResponse:
    """Call `client.generate`, retrying transient failures with exponential backoff.

    Backoff jitter uses a private, unseeded RNG: it only affects wall-clock
    timing, never experimental randomness.
    """
    jitter = random.Random()
    attempt = 0
    while True:
        t0 = time.monotonic()
        try:
            resp = await client.generate(system_prompt, messages, generation_config)
            if resp.latency_s is None:
                resp.latency_s = time.monotonic() - t0
            return resp
        except TransientLLMError as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = min(backoff_max_s, backoff_base_s * 2 ** (attempt - 1)) * (0.5 + jitter.random())
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            await asyncio.sleep(delay)
