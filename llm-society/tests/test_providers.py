"""Provider adapters, exercised without network access."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from llm_society.llm.base import GenerationConfig, Message, PermanentLLMError, TransientLLMError, generate_with_retry


def test_openai_compatible_adapter(monkeypatch):
    from llm_society.llm.local_client import LocalClient

    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        if seen.setdefault("n", 0) == 0:
            seen["n"] = 1
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={
            "model": "llama-x", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2}})

    client = LocalClient("llama-x", extra_params={"top_k": 5})
    client._client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(handler))
    resp = asyncio.run(generate_with_retry(client, "sys", [Message("user", "hi")], GenerationConfig(0.7, 50),
                                           backoff_base_s=0.001))
    assert resp.text == "hello" and resp.prompt_tokens == 11
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["temperature"] == 0.7 and body["max_tokens"] == 50 and body["top_k"] == 5


def test_openai_adapter_errors(monkeypatch):
    from llm_society.llm.openai_client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    client = OpenAIClient("gpt-x")
    client._client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")))
    with pytest.raises(PermanentLLMError):
        asyncio.run(client.generate("s", [Message("user", "u")], GenerationConfig(0.0, 5)))
    client._client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(lambda r: httpx.Response(503, text="busy")))
    with pytest.raises(TransientLLMError):
        asyncio.run(client.generate("s", [Message("user", "u")], GenerationConfig(0.0, 5)))
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(PermanentLLMError):
        OpenAIClient("gpt-x")


def test_anthropic_adapter_request_shape():
    pytest.importorskip("anthropic")
    from llm_society.llm.anthropic_client import AnthropicClient

    client = AnthropicClient("claude-opus-5", extra_params={"output_config": {"effort": "low"}})
    captured = {}

    async def fake_create(**body):
        captured.update(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="A reply.")],
            model="claude-opus-5", stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=30, output_tokens=12),
            model_dump=lambda mode="json": {"id": "msg_x"},
        )

    client._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    resp = asyncio.run(client.generate("system", [Message("user", "u")], GenerationConfig(0.7, 400)))
    assert resp.text == "A reply." and resp.completion_tokens == 12
    assert "temperature" not in captured and "temperature_not_sent" in resp.notes
    assert captured["system"] == "system" and captured["max_tokens"] == 400
    assert captured["output_config"] == {"effort": "low"}
