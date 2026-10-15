# Módulo 1 — Asincronía y abstracción de clientes LLM

## Ejercicio: Orquestador concurrente de modelos

📁 [`ejercicio-orquestador/`](ejercicio-orquestador/)

Simula un backend que consulta varios modelos de IA de forma concurrente:

- **Parte A:** tres corrutinas (`gpt_4_call`, `claude_3_call`, `local_llama_call`) con distinta latencia, lanzadas a la vez con `asyncio.gather`.
- **Resiliencia:** `asyncio.timeout` de 2 s por llamada; si una tarda más, se captura `TimeoutError` y el programa sigue.
- **Parte B:** 10 llamadas limitadas con `asyncio.Semaphore(2)`, así que solo corren 2 al mismo tiempo.
- No usa `time.sleep` (bloqueante), solo `await asyncio.sleep`.

No necesita API keys ni dependencias externas:

```bash
python ejercicio-orquestador/orquestador.py
```

## Pre-entrega: Cliente LLM unificado

📁 [`pre-entrega-cliente-llm/`](pre-entrega-cliente-llm/)

Capa de abstracción para hablar con OpenAI y Anthropic con una interfaz común:

- `BaseLLMClient` (clase abstracta) con `async def chat(prompt) -> str`.
- `OpenAIClient` y `AnthropicClient` usando los SDKs asíncronos (`AsyncOpenAI`, `AsyncAnthropic`).
- `LLMFactory` para elegir el proveedor en tiempo de ejecución.
- `LLMConfig` con Pydantic: las keys son `SecretStr` (no aparecen en logs) y se validan con `field_validator` / `model_validator`.
- Manejo específico de errores: credenciales, rate limit, modelo inexistente y red.

Variables de entorno necesarias (al menos una):

| Variable | Proveedor |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |

```bash
export OPENAI_API_KEY="..."        # y/o ANTHROPIC_API_KEY
python pre-entrega-cliente-llm/cliente_llm.py
```
