# Pre-entrega 1 · Unified Async LLM Client

Cliente **asíncrono** y **unificado** para modelos de lenguaje: la misma interfaz sirve para **OpenAI**, **Anthropic** y **Gemini**, y el proveedor se elige con una variable de configuración.

| Requisito | Dónde está |
|---|---|
| Intercambiabilidad | `BaseLLMClient` (clase abstracta) + `AsyncLLMManager`, que elige el cliente según `LLM_PROVIDER` |
| Asincronía | `AsyncOpenAI` / `AsyncAnthropic` con `await`; nada bloquea el event loop |
| Streaming | `stream()` es un generador asíncrono: `async for` sobre el stream del SDK + `yield` de cada fragmento |
| Validación | `schemas.py` (Pydantic): `ChatMessage`, `ModelConfig` (temperatura 0–2, `max_tokens` > 0), `Settings`, `ModelResponse` |
| Errores controlados | Auth, rate limit, timeout, red y errores del servidor se devuelven como `LLMError`; nunca rompen el programa |

## Estructura

```
pre-entrega-1-cliente-llm/
├── schemas.py        # Modelos Pydantic de entrada, configuración y salida
├── clients.py        # BaseLLMClient + OpenAIClient, AnthropicClient, GeminiClient
├── manager.py        # AsyncLLMManager: elige el cliente desde la configuración
├── main.py           # Script de validación: modo normal + streaming
├── tests/            # 21 tests con los SDKs simulados (no necesitan API keys)
├── requirements.txt
└── .env.example
```

## Instalación

Requiere **Python 3.12+**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # y completá la key del proveedor que vayas a usar
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (default `openai`) | `openai`, `anthropic` o `gemini` |
| `OPENAI_API_KEY` | Si el proveedor es `openai` | Key de [platform.openai.com](https://platform.openai.com) |
| `ANTHROPIC_API_KEY` | Si el proveedor es `anthropic` | Key de [platform.claude.com](https://platform.claude.com) |
| `GEMINI_API_KEY` o `GOOGLE_API_KEY` | Si el proveedor es `gemini` | Key gratuita de [aistudio.google.com](https://aistudio.google.com) |
| `LLM_MODEL` | No | Modelo a usar; por defecto `gpt-4o-mini`, `claude-haiku-4-5` o `gemini-3.8-flash` |
| `LLM_TEMPERATURE` | No (0.7) | Entre 0 y 2 (Anthropic: 0 a 1). Pydantic rechaza valores fuera de rango |
| `LLM_MAX_TOKENS` | No (1024) | Máximo de tokens de la respuesta |
| `LLM_TIMEOUT` | No (30) | Segundos máximos por llamada |
| `LLM_MAX_RETRIES` | No (2) | Reintentos automáticos con backoff ante 429, 5xx o fallas de red |

## Uso

```bash
python main.py                      # usa LLM_PROVIDER
python main.py --provider gemini    # fuerza un proveedor
```

El script pregunta *"¿Qué es la entropía?"* dos veces: en **modo normal** (respuesta completa con latencia y tokens) y en **modo streaming** (el texto se imprime a medida que llega).

Desde código:

```python
from manager import AsyncLLMManager

manager = AsyncLLMManager.from_env()                 # o AsyncLLMManager(Settings(...))

respuesta = await manager.generate("¿Qué es la entropía?")
if respuesta.ok:
    print(respuesta.content, respuesta.usage.total)
else:
    print(respuesta.error.type, respuesta.error.retryable)

async for fragmento in manager.stream("¿Qué es la entropía?"):
    print(fragmento, end="", flush=True)
```

## Tests

```bash
pytest -q
```

Los tests reemplazan los SDKs por objetos falsos, así que corren sin keys ni internet. Cubren las validaciones de Pydantic, la normalización de respuestas de cada proveedor, el streaming y la clasificación de errores (401, 404, 429, timeout y red).

## Decisiones de diseño

- **Una respuesta con la misma forma para todos.** Cada SDK devuelve el texto distinto (OpenAI: `choices[0].message.content`; Anthropic: una lista de bloques en `content`). Los clientes lo normalizan a `ModelResponse`.
- **Errores como datos.** `generate()` nunca lanza errores del proveedor: devuelve `ModelResponse(error=LLMError(...))` con `type` y `retryable`. En `stream()`, como un generador no puede "devolver" un error a mitad de camino, se lanza una única excepción propia, `LLMClientError`, con el `LLMError` adentro.
- **Reintentos.** Se usan los reintentos con backoff exponencial que ya traen los SDKs oficiales (`max_retries`), que cubren 429, 5xx y fallas de red. Los errores que no tiene sentido reintentar (key inválida, modelo inexistente) se marcan `retryable=False`.
- **Secretos.** Las API keys son `SecretStr`: no aparecen en `print`, logs ni tracebacks.
- **Gemini sin código nuevo.** Gemini ofrece una [API compatible con OpenAI](https://ai.google.dev/gemini-api/docs/openai), así que `GeminiClient` hereda de `OpenAIClient` y solo cambia la URL base. Se incluyó porque tiene plan gratuito y permite probar el proyecto sin costo.
- **Anthropic y `temperature`.** Las versiones nuevas del SDK de Anthropic (1.x) ya no aceptan `temperature`. El cliente detecta si el parámetro existe y lo omite si no está disponible.
