# Módulo 7 · Observabilidad y trazado

## Ejercicio: Instrumentación y análisis de trazas con Arize Phoenix

📁 [`ejercicio-phoenix-trazas/`](ejercicio-phoenix-trazas/)

Instrumentación del orquestador multi-agente de la pre-entrega 6 con Arize Phoenix y OpenInference: los agentes, las llamadas al LLM, las herramientas y las fuentes de datos. Genera tráfico con 5 consultas complejas y 2 fallas provocadas, y analiza las trazas: latencia externa vs. local, p50/p95, tokens y costo por agente, groundedness de las respuestas y propagación de errores.

## Ejercicio: Escalabilidad asíncrona con FastAPI y Redis

📁 [`ejercicio-fastapi-redis/`](ejercicio-fastapi-redis/)

API con FastAPI que no corre el agente dentro del request: guarda el estado de la tarea en Redis, la encola y devuelve un `job_id` al instante. Un worker la toma con `BLPOP` y registra cada cambio de estado (`pending → processing → completed / failed`), y el cliente consulta `/status/{job_id}` por polling.

## Pre-entrega 7: API de producción y monitoreo activo

📁 [`pre-entrega-7-api-produccion/`](pre-entrega-7-api-produccion/)

API con FastAPI que expone el orquestador multi-agente del módulo 6 en modo producción. El API encola cada tarea en Redis y responde con un `job_id`; un worker corre el grafo con los checkpoints de LangGraph en Redis. Las tareas críticas (por costo estimado o por enviar un email) se pausan con `interrupt()` hasta que alguien las aprueba en `POST /tasks/{id}/approve`. Cada ejecución queda trazada en Arize Phoenix, con costo, tokens y latencia, y una prueba de carga con 5 peticiones concurrentes lee de Phoenix el costo por ejecución y la latencia p95.
