"""Pydantic schemas: input messages, model configuration and the unified response."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"   # extra: free tier, through the OpenAI-compatible API


# Default model per provider (override with LLM_MODEL in .env)
DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
    Provider.GEMINI: "gemini-3.8-flash",
}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """One conversation message. Replaces nested dicts like {"role": ..., "content": ...}."""
    model_config = ConfigDict(frozen=True, use_enum_values=True)

    role: Role
    content: str = Field(min_length=1, description="Message text (must not be empty)")


class ModelConfig(BaseModel):
    """Model parameters, validated before calling the API."""
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=32_000)
    timeout: float = Field(default=30.0, gt=0, description="Max seconds per call")
    max_retries: int = Field(default=2, ge=0, le=5, description="Automatic SDK retries (429, 5xx, network)")

    @model_validator(mode="after")
    def temperature_per_provider(self):
        # Anthropic accepts temperature between 0 and 1; OpenAI and Gemini, between 0 and 2
        if self.provider == Provider.ANTHROPIC and self.temperature > 1.0:
            raise ValueError("Anthropic accepts temperature between 0 and 1")
        return self


class Settings(BaseModel):
    """Full configuration: which provider to use, its parameters and the API keys (hidden in logs)."""
    config: ModelConfig
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    gemini_api_key: Optional[SecretStr] = None

    def _key_for(self, provider: Provider) -> Optional[SecretStr]:
        return {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.GEMINI: self.gemini_api_key,
        }[provider]

    @model_validator(mode="after")
    def key_for_selected_provider(self):
        key = self._key_for(self.config.provider)
        if key is None or not key.get_secret_value().strip():
            env_var = f"{self.config.provider.value.upper()}_API_KEY"
            raise ValueError(f"Missing {env_var} to use provider '{self.config.provider.value}'")
        return self

    def api_key(self) -> str:
        return self._key_for(self.config.provider).get_secret_value()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
ErrorType = Literal["auth", "rate_limit", "timeout", "network", "bad_request", "server", "unknown"]


class LLMError(BaseModel):
    """Controlled error: the program gets this instead of an exception that crashes it."""
    type: ErrorType
    message: str
    retryable: bool = Field(description="True if retrying later makes sense (429, network, 5xx)")
    status_code: Optional[int] = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Response with the same shape regardless of the provider."""
    provider: Provider
    model: str
    content: str = ""
    usage: Optional[TokenUsage] = None
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[LLMError] = None

    @property
    def ok(self) -> bool:
        return self.error is None
