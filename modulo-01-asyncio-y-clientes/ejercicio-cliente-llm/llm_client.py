import asyncio
import os
from abc import ABC, abstractmethod
from enum import Enum

import anthropic
import openai
from pydantic import BaseModel, SecretStr, ValidationError, field_validator, model_validator
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMConfig(BaseModel):
    """
    API keys stored as SecretStr (hidden when printed).
    Pydantic checks that keys are not blank and that at least one is present.
    """
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def blank_to_none(cls, value):
        # An empty or whitespace-only key counts as "no key"
        text = value.get_secret_value() if isinstance(value, SecretStr) else value
        if text is None or not str(text).strip():
            return None
        return value

    @model_validator(mode="after")
    def at_least_one_key(self):
        if self.openai_api_key is None and self.anthropic_api_key is None:
            raise ValueError("Set at least OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return self

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, prompt: str) -> str:
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,  # required by Anthropic
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

class LLMFactory:
    @staticmethod
    def create_client(provider: Provider, config: LLMConfig) -> BaseLLMClient:
        if provider == Provider.OPENAI:
            if config.openai_api_key is None:
                raise ValueError("Config has no OPENAI_API_KEY")
            return OpenAIClient(config.openai_api_key.get_secret_value())

        if provider == Provider.ANTHROPIC:
            if config.anthropic_api_key is None:
                raise ValueError("Config has no ANTHROPIC_API_KEY")
            return AnthropicClient(config.anthropic_api_key.get_secret_value())

        raise ValueError(f"Unsupported provider: {provider}")

async def main():
    try:
        config = LLMConfig(
            openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
            anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
        )
    except ValidationError as e:
        print(f"Invalid configuration: {e.errors()[0]['msg']}")
        return
    print(config)  # keys are shown as '**********'

    for provider in Provider:
        try:
            client = LLMFactory.create_client(provider, config)
            answer = await client.chat("¿Qué es la entropía? Respondé en una oración.")
            print(f"[{provider.value}] {answer}")
        except ValueError as e:
            print(f"[{provider.value}] CONFIG: {e}")
        except (openai.AuthenticationError, anthropic.AuthenticationError):
            print(f"[{provider.value}] CREDENTIALS: the API key is invalid")
        except (openai.RateLimitError, anthropic.RateLimitError):
            print(f"[{provider.value}] LIMIT: quota or rate limit exceeded")
        except (openai.NotFoundError, anthropic.NotFoundError):
            print(f"[{provider.value}] MODEL: the model does not exist or you have no access")
        except (openai.APIConnectionError, anthropic.APIConnectionError):
            print(f"[{provider.value}] NETWORK: could not connect to the API")
        except (openai.APIStatusError, anthropic.APIStatusError) as e:
            print(f"[{provider.value}] API: error {e.status_code}")

if __name__ == "__main__":
    asyncio.run(main())
