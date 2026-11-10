# Módulo 7 · Observabilidad y trazado

## Ejercicio: Instrumentación y análisis de trazas con Arize Phoenix

📁 [`ejercicio-phoenix-trazas/`](ejercicio-phoenix-trazas/)

Instrumentación del orquestador multi-agente de la pre-entrega 6 con Arize Phoenix y OpenInference: los agentes, las llamadas al LLM, las herramientas y las fuentes de datos. Genera tráfico con 5 consultas complejas y 2 fallas provocadas, y analiza las trazas: latencia externa vs. local, p50/p95, tokens y costo por agente, groundedness de las respuestas y propagación de errores.

## Ejercicio: Escalabilidad asíncrona con FastAPI y Redis

📁 [`ejercicio-fastapi-redis/`](ejercicio-fastapi-redis/)

API con FastAPI que no corre el agente dentro del request: guarda el estado de la tarea en Redis, la encola y devuelve un `job_id` al instante. Un worker la toma con `BLPOP` y registra cada cambio de estado (`pending → processing → completed / failed`), y el cliente consulta `/status/{job_id}` por polling.
