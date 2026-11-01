"""Tests without real API keys: the SDKs are replaced with fake objects."""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from pydantic import ValidationError

from clients import LLMClientError
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Provider, Settings

REQ = httpx.Request("POST", "https://api.example.com/v1")


def manager(provider: Provider, **config) -> AsyncLLMManager:
    return AsyncLLMManager(Settings(
        config=ModelConfig(provider=provider, model="test-model", **config),
        openai_api_key="sk-test", anthropic_api_key="sk-ant-test", gemini_api_key="g-test",
    ))


def run(coro):
    return asyncio.run(coro)


async def collect(gen):
    return [x async for x in gen]


# ---------------------------------------------------------------- schemas
def test_temperature_out_of_range():
    with pytest.raises(ValidationError):
        ModelConfig(provider=Provider.OPENAI, model="m", temperature=2.5)


def test_anthropic_rejects_temperature_above_1():
    with pytest.raises(ValidationError):
        ModelConfig(provider=Provider.ANTHROPIC, model="m", temperature=1.5)


def test_max_tokens_must_be_positive():
    with pytest.raises(ValidationError):
        ModelConfig(provider=Provider.OPENAI, model="m", max_tokens=0)


def test_empty_message_is_invalid():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


def test_missing_key_for_provider():
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(config=ModelConfig(provider=Provider.ANTHROPIC, model="m"), openai_api_key="sk-x")


def test_keys_are_not_printed():
    assert "sk-test" not in repr(manager(Provider.OPENAI).settings)


# ---------------------------------------------------------------- OpenAI / Gemini
def fake_openai(response=None, chunks=None, error=None):
    async def create(**kwargs):
        if error:
            raise error
        if kwargs.get("stream"):
            async def gen():
                for c in chunks:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))])
            return gen()
        return response
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_openai_generate_normalizes_response():
    m = manager(Provider.OPENAI)
    m.client.client = fake_openai(response=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Entropy measures disorder."), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
    ))
    r = run(m.generate("What is entropy?"))
    assert r.ok and r.content == "Entropy measures disorder."
    assert r.usage.total == 19 and r.finish_reason == "stop"


def test_openai_stream_yields_chunks():
    m = manager(Provider.GEMINI)   # Gemini reuses the OpenAI client
    m.client.client = fake_openai(chunks=["Entropy ", "is ", None, "..."])
    assert run(collect(m.stream("hi"))) == ["Entropy ", "is ", "..."]


@pytest.mark.parametrize("error, error_type, retryable", [
    (openai.RateLimitError("429", response=httpx.Response(429, request=REQ), body=None), "rate_limit", True),
    (openai.AuthenticationError("401", response=httpx.Response(401, request=REQ), body=None), "auth", False),
    (openai.APITimeoutError(request=REQ), "timeout", True),
    (openai.APIConnectionError(request=REQ), "network", True),
    (openai.NotFoundError("404", response=httpx.Response(404, request=REQ), body=None), "bad_request", False),
])
def test_openai_controlled_errors(error, error_type, retryable):
    m = manager(Provider.OPENAI)
    m.client.client = fake_openai(error=error)
    r = run(m.generate("hi"))            # does not raise: returns the structured error
    assert not r.ok and r.error.type == error_type and r.error.retryable is retryable


def test_stream_error_raises_llm_client_error():
    m = manager(Provider.OPENAI)
    m.client.client = fake_openai(error=openai.RateLimitError("429", response=httpx.Response(429, request=REQ), body=None))
    with pytest.raises(LLMClientError) as info:
        run(collect(m.stream("hi")))
    assert info.value.error.type == "rate_limit"


# ---------------------------------------------------------------- Anthropic
def fake_anthropic(response=None, texts=None, error=None, capture=None):
    async def create(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        if error:
            raise error
        return response

    class Stream:
        async def __aenter__(self):
            async def gen():
                for t in texts:
                    yield t
            return SimpleNamespace(text_stream=gen())

        async def __aexit__(self, *a):
            return False

    return SimpleNamespace(messages=SimpleNamespace(create=create, stream=lambda **kw: Stream()))


def test_anthropic_separates_system_and_joins_blocks():
    m = manager(Provider.ANTHROPIC, temperature=0.5)
    sent = {}
    m.client.client = fake_anthropic(capture=sent, response=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Entropy "), SimpleNamespace(type="text", text="measures disorder.")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=6), stop_reason="end_turn",
    ))
    r = run(m.generate("What is entropy?", system="Be brief."))
    assert r.content == "Entropy measures disorder."
    assert sent["system"] == "Be brief."
    assert all(msg["role"] != "system" for msg in sent["messages"])


def test_anthropic_stream():
    m = manager(Provider.ANTHROPIC)
    m.client.client = fake_anthropic(texts=["Hello", " world"])
    assert run(collect(m.stream("hi"))) == ["Hello", " world"]


def test_anthropic_controlled_rate_limit():
    m = manager(Provider.ANTHROPIC)
    m.client.client = fake_anthropic(error=anthropic.RateLimitError("429", response=httpx.Response(429, request=REQ), body=None))
    r = run(m.generate("hi"))
    assert r.error.type == "rate_limit" and r.error.retryable


# ---------------------------------------------------------------- interchangeability
@pytest.mark.parametrize("provider, class_name", [
    (Provider.OPENAI, "OpenAIClient"), (Provider.ANTHROPIC, "AnthropicClient"), (Provider.GEMINI, "GeminiClient"),
])
def test_manager_picks_client_from_config(provider, class_name):
    assert type(manager(provider).client).__name__ == class_name


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g-x")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    m = AsyncLLMManager.from_env()
    assert m.settings.config.provider == Provider.GEMINI and m.settings.config.temperature == 0.2
