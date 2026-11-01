"""AsyncLLMManager: picks the client from the configuration and exposes a single API."""

import os
from typing import AsyncIterator, List, Optional, Union

from clients import AnthropicClient, BaseLLMClient, GeminiClient, OpenAIClient
from schemas import DEFAULT_MODELS, ChatMessage, ModelConfig, ModelResponse, Provider, Role, Settings

# Provider -> class registry. Adding a new provider = one line here.
CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}

PromptInput = Union[str, List[ChatMessage]]


class AsyncLLMManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        client_class = CLIENTS[settings.config.provider]
        self.client: BaseLLMClient = client_class(settings.config, settings.api_key())

    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> "AsyncLLMManager":
        """Build the configuration from environment variables (.env). Pydantic validates everything before creating the client."""
        selected = Provider(provider or os.getenv("LLM_PROVIDER", "openai").lower())
        config = ModelConfig(
            provider=selected,
            model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[selected],
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 1024)),
            timeout=float(os.getenv("LLM_TIMEOUT", 30)),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", 2)),
        )
        return cls(Settings(
            config=config,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
        ))

    @staticmethod
    def _to_messages(prompt: PromptInput, system: Optional[str] = None) -> List[ChatMessage]:
        messages = [ChatMessage(role=Role.USER, content=prompt)] if isinstance(prompt, str) else list(prompt)
        if system:
            messages.insert(0, ChatMessage(role=Role.SYSTEM, content=system))
        return messages

    async def generate(self, prompt: PromptInput, system: Optional[str] = None) -> ModelResponse:
        """Full response. Never raises provider errors: they come in ModelResponse.error."""
        return await self.client.generate(self._to_messages(prompt, system))

    async def stream(self, prompt: PromptInput, system: Optional[str] = None) -> AsyncIterator[str]:
        """Text chunks as they arrive. If the API fails, raises LLMClientError (already classified)."""
        async for chunk in self.client.stream(self._to_messages(prompt, system)):
            yield chunk

    def __repr__(self) -> str:
        c = self.settings.config
        return f"AsyncLLMManager(provider={c.provider.value}, model={c.model}, temperature={c.temperature}, max_tokens={c.max_tokens})"
