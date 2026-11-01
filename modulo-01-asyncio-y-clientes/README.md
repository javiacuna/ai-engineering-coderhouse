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
python ejercicio-orquestador/orchestrator.py
```

## Ejercicio: Cliente LLM con Factory (esqueleto de clase)

📁 [`ejercicio-cliente-llm/`](ejercicio-cliente-llm/)

Completa el esqueleto visto en clase: `LLMConfig` con `SecretStr` y validadores de Pydantic, `BaseLLMClient` abstracta, `OpenAIClient` y `AnthropicClient`, un `LLMFactory` y manejo de errores específico por tipo (credenciales, rate limit, modelo, red).

```bash
python ejercicio-cliente-llm/llm_client.py
```

## Pre-entrega 1: Unified Async LLM Client

📁 [`pre-entrega-1-cliente-llm/`](pre-entrega-1-cliente-llm/) · ver su [README](pre-entrega-1-cliente-llm/README.md)

Cliente asíncrono con interfaz común para OpenAI, Anthropic y Gemini: esquemas Pydantic (`schemas.py`), `AsyncLLMManager` que elige el proveedor por configuración, streaming con generadores asíncronos (`yield`), errores controlados (auth, rate limit, timeout, red) y 21 tests sin API keys.

```bash
cd pre-entrega-1-cliente-llm
python main.py --provider gemini
pytest -q
```
