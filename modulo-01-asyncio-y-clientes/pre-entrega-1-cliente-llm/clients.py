"""Async clients per provider, all sharing the same interface (BaseLLMClient)."""

import inspect
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, List

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, TokenUsage

logger = logging.getLogger("llm_client")


class LLMClientError(Exception):
    """The only exception stream() can raise: it carries the already classified error (LLMError)."""

    def __init__(self, error: LLMError):
        super().__init__(f"[{error.type}] {error.message}")
        self.error = error


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------
class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key

    @abstractmethod
    async def _generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Actual SDK call. May raise the SDK's own exceptions."""

    @abstractmethod
    def _stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        """Actual SDK generator. May raise the SDK's own exceptions."""

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError:
        """Translate an SDK exception into a domain LLMError."""

    # --- Public methods: they never let an SDK exception escape ---
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        start = time.perf_counter()
        try:
            response = await self._generate(messages)
        except Exception as exc:
            error = self._classify_error(exc)
            logger.error("generate failed (%s): %s", error.type, error.message)
            response = ModelResponse(provider=self.config.provider, model=self.config.model, error=error)
        response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return response

    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        """Async generator of text chunks, yielded as they arrive from the API."""
        try:
            async for chunk in self._stream(messages):
                if chunk:
                    yield chunk
        except Exception as exc:
            error = self._classify_error(exc)
            logger.error("stream failed (%s): %s", error.type, error.message)
            raise LLMClientError(error) from exc


# ---------------------------------------------------------------------------
# OpenAI (and any OpenAI-compatible API)
# ---------------------------------------------------------------------------
class OpenAIClient(BaseLLMClient):
    base_url: str | None = None   # None = api.openai.com

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config, api_key)
        # AsyncOpenAI: NON-blocking client. Its retries with backoff cover 429, 5xx and network failures.
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=self.base_url,
            timeout=config.timeout, max_retries=config.max_retries,
        )

    def _request_params(self, messages: List[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def _generate(self, messages: List[ChatMessage]) -> ModelResponse:
        r = await self.client.chat.completions.create(**self._request_params(messages))
        usage = TokenUsage(input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens) if r.usage else None
        return ModelResponse(
            provider=self.config.provider, model=self.config.model,
            content=r.choices[0].message.content or "",
            usage=usage, finish_reason=r.choices[0].finish_reason,
        )

    async def _stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        response = await self.client.chat.completions.create(**self._request_params(messages), stream=True)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _classify_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        # Order matters: APITimeoutError subclasses APIConnectionError, and status errors subclass APIStatusError
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return LLMError(type="auth", message="Invalid API key or missing permissions", retryable=False, status_code=status)
        if isinstance(exc, openai.RateLimitError):
            return LLMError(type="rate_limit", message="Rate limit or quota exceeded", retryable=True, status_code=status)
        if isinstance(exc, openai.APITimeoutError):
            return LLMError(type="timeout", message=f"No response within {self.config.timeout}s", retryable=True)
        if isinstance(exc, openai.APIConnectionError):
            return LLMError(type="network", message="Could not connect to the API", retryable=True)
        if isinstance(exc, openai.BadRequestError | openai.NotFoundError | openai.UnprocessableEntityError):
            return LLMError(type="bad_request", message=_short_message(exc), retryable=False, status_code=status)
        if isinstance(exc, openai.InternalServerError):
            return LLMError(type="server", message="Provider server error", retryable=True, status_code=status)
        return LLMError(type="unknown", message=_short_message(exc), retryable=False, status_code=status)


class GeminiClient(OpenAIClient):
    """Gemini exposes an OpenAI-compatible API: only the base URL changes."""
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config, api_key)
        self.client = AsyncAnthropic(api_key=api_key, timeout=config.timeout, max_retries=config.max_retries)
        # Newer SDK versions (1.x) no longer accept `temperature`: send it only if it exists
        self._supports_temperature = "temperature" in inspect.signature(self.client.messages.create).parameters
        if not self._supports_temperature:
            logger.info("The installed Anthropic SDK does not accept 'temperature': the parameter is skipped")

    def _request_params(self, messages: List[ChatMessage]) -> dict:
        # Difference from OpenAI: "system" goes separately, not inside the message list
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        params = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
            "max_tokens": self.config.max_tokens,   # required by Anthropic
        }
        if self._supports_temperature:
            params["temperature"] = self.config.temperature
        if system:
            params["system"] = system
        return params

    async def _generate(self, messages: List[ChatMessage]) -> ModelResponse:
        r = await self.client.messages.create(**self._request_params(messages))
        # Difference from OpenAI: content is a LIST of blocks; the text blocks are joined
        text = "".join(b.text for b in r.content if b.type == "text")
        return ModelResponse(
            provider=self.config.provider, model=self.config.model, content=text,
            usage=TokenUsage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens),
            finish_reason=r.stop_reason,
        )

    async def _stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._request_params(messages)) as s:
            async for text in s.text_stream:
                yield text

    def _classify_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            return LLMError(type="auth", message="Invalid API key or missing permissions", retryable=False, status_code=status)
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError(type="rate_limit", message="Rate limit or quota exceeded", retryable=True, status_code=status)
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMError(type="timeout", message=f"No response within {self.config.timeout}s", retryable=True)
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(type="network", message="Could not connect to the API", retryable=True)
        if isinstance(exc, anthropic.BadRequestError | anthropic.NotFoundError | anthropic.UnprocessableEntityError):
            return LLMError(type="bad_request", message=_short_message(exc), retryable=False, status_code=status)
        if isinstance(exc, anthropic.InternalServerError):
            return LLMError(type="server", message="Provider server error", retryable=True, status_code=status)
        return LLMError(type="unknown", message=_short_message(exc), retryable=False, status_code=status)


def _short_message(exc: Exception) -> str:
    """Short message with no sensitive data."""
    text = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
    return text[:300]
