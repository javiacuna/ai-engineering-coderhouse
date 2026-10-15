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
    API Keys guardadas como SecretStr (se ocultan al imprimir).
    Pydantic valida que las keys no estén vacías y que haya al menos una.
    """
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def vacia_a_none(cls, valor):
        # Una key vacía o con solo espacios cuenta como "no hay key"
        texto = valor.get_secret_value() if isinstance(valor, SecretStr) else valor
        if texto is None or not str(texto).strip():
            return None
        return valor

    @model_validator(mode="after")
    def al_menos_una_key(self):
        if self.openai_api_key is None and self.anthropic_api_key is None:
            raise ValueError("Definí al menos OPENAI_API_KEY o ANTHROPIC_API_KEY")
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
            max_tokens=1024,  # en Anthropic es obligatorio
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

class LLMFactory:
    @staticmethod
    def create_client(provider: Provider, config: LLMConfig) -> BaseLLMClient:
        if provider == Provider.OPENAI:
            if config.openai_api_key is None:
                raise ValueError("La config no tiene OPENAI_API_KEY")
            return OpenAIClient(config.openai_api_key.get_secret_value())

        if provider == Provider.ANTHROPIC:
            if config.anthropic_api_key is None:
                raise ValueError("La config no tiene ANTHROPIC_API_KEY")
            return AnthropicClient(config.anthropic_api_key.get_secret_value())

        raise ValueError(f"Proveedor no soportado: {provider}")

async def main():
    try:
        config = LLMConfig(
            openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
            anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
        )
    except ValidationError as e:
        print(f"Configuración inválida: {e.errors()[0]['msg']}")
        return
    print(config)  # las keys se ven como '**********'

    for provider in Provider:
        try:
            client = LLMFactory.create_client(provider, config)
            respuesta = await client.chat("¿Qué es la entropía? Respondé en una oración.")
            print(f"[{provider.value}] {respuesta}")
        except ValueError as e:
            print(f"[{provider.value}] CONFIG: {e}")
        except (openai.AuthenticationError, anthropic.AuthenticationError):
            print(f"[{provider.value}] CREDENCIALES: la API key es inválida")
        except (openai.RateLimitError, anthropic.RateLimitError):
            print(f"[{provider.value}] LÍMITE: superaste la cuota o el rate limit")
        except (openai.NotFoundError, anthropic.NotFoundError):
            print(f"[{provider.value}] MODELO: el modelo no existe o no tenés acceso")
        except (openai.APIConnectionError, anthropic.APIConnectionError):
            print(f"[{provider.value}] RED: no se pudo conectar con la API")
        except (openai.APIStatusError, anthropic.APIStatusError) as e:
            print(f"[{provider.value}] API: error {e.status_code}")

if __name__ == "__main__":
    asyncio.run(main())
