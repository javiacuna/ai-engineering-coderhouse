# Ejercicio: Escalabilidad asíncrona con FastAPI y Redis

Una API que **no corre el agente dentro del request**: registra la tarea en Redis, la encola y devuelve un `job_id` en milisegundos. Un worker la procesa aparte, y el cliente consulta el estado por **polling**.

```
POST /process ──► Redis: status:<job_id> = pending ──► cola tasks_queue (RPUSH)
                                                             │
                              worker (BLPOP) ◄───────────────┘
                                 │  processing → completed / failed
                                 ▼
GET /status/<job_id> ◄── lee status:<job_id>
```

| Archivo | Qué hace |
|---|---|
| `main.py` | La API (`/process`, `/status/{job_id}`), sus modelos Pydantic de entrada y salida, y el worker |
| `tests/test_jobs.py` | 9 tests contra un Redis en memoria (`fakeredis`): no hace falta un servidor de Redis |
| `pytest.ini` | Activa el modo async de pytest |

## Cómo ejecutarlo

**1. Instalar** (en el `.venv` del repo):

```bash
pip install fastapi uvicorn redis fakeredis pytest-asyncio httpx
```

**2. Levantar Redis.** En Mac, con Homebrew:

```bash
brew install redis
brew services start redis      # queda corriendo en localhost:6379
redis-cli ping                 # tiene que responder PONG
```

Si tenés Docker: `docker run -d -p 6379:6379 redis`.

**3. Levantar la API** (el worker arranca con ella):

```bash
cd modulo-07-observabilidad/ejercicio-fastapi-redis
uvicorn main:app --reload
```

**4. Probar** desde otra terminal:

```bash
curl -X POST localhost:8000/process -H "Content-Type: application/json" -d '{"query": "hola"}'
curl localhost:8000/status/<job_id>
```

## Salida real

```
POST:      {"job_id":"d110e442-…","status":"pending"}
t=0,5 s:   {"job_id":"d110e442-…","status":"processing","query":"hola","result":null,"error":null,
            "created_at":"…17:54:39","started_at":"…17:54:39", …}
t=5,5 s:   {"job_id":"d110e442-…","status":"completed","query":"hola",
            "result":"Resultado para 'hola': procesado por el agente en 5 s.","error":null,
            "started_at":"…17:54:39","finished_at":"…17:54:44", …}
inventado: {"detail":"No existe la tarea no-existe"}   [404]
```

El POST responde con **202 Accepted**: la tarea se aceptó, pero todavía no terminó. Como el worker está libre, la toma en el mismo milisegundo, así que la primera consulta ya muestra `processing`. Si mandás varios POST seguidos, los que esperan en la cola se ven en `pending`.

## Decisiones de diseño

| Decisión | Por qué |
|---|---|
| Modelos de respuesta `TaskResponse` (POST) y `JobStatus` (GET) con `response_model` | La salida también queda validada, no solo la entrada: `status` solo puede ser `pending`, `processing`, `completed` o `failed`. Los dos modelos aparecen documentados en el esquema OpenAPI, en <http://localhost:8000/docs> |
| Se guarda `pending` **antes** del `RPUSH` | Si el worker fuera más rápido que la escritura, un cliente que consulta enseguida recibiría 404 por una tarea que sí existe (el "error común" de la unidad) |
| `RPUSH` para encolar y `BLPOP` para sacar | Cola FIFO: las tareas se procesan en el orden en que llegaron |
| `BLPOP` con `timeout=0` | El worker se bloquea hasta que llega trabajo: no consulta Redis en un bucle ni gasta CPU mientras espera |
| Estado `failed` con el error | Si el agente falla, el cliente lo ve en `/status` y el worker sigue con la próxima tarea |
| Errores de Redis → **503** en la API y reintento en el worker | Si Redis se cae, la API avisa que el servicio no está disponible (no un 500 genérico) y el worker no muere: reintenta cada 2 s |
| `created_at`, `started_at`, `finished_at` | Permiten medir cuánto esperó cada tarea en la cola y cuánto tardó en procesarse: la misma separación de latencias del ejercicio de Phoenix |
| Vencimiento de 24 h (`ex=JOB_TTL_SECONDS`) | Los registros de tareas viejas no se acumulan para siempre en Redis |
| `process_query` aparte | Es el único lugar que cambia para correr el grafo real: `await graph.ainvoke(...)` en vez de `asyncio.sleep` |

`PROCESSING_SECONDS` (5 por defecto) y `REDIS_URL` se pueden cambiar con variables de entorno.

**Escalar:** como la cola vive en Redis, se pueden correr varios workers en procesos separados que consuman de la misma cola. `BLPOP` entrega cada `job_id` a un solo worker, así que ninguna tarea se procesa dos veces.

## Tests

```bash
python -m pytest -q
```

Usan `fakeredis`, un Redis en memoria, y verifican:
- que el POST responde enseguida con `pending` y deja el estado guardado antes de encolar;
- el 404 para un `job_id` inventado y el 422 para una consulta vacía;
- el recorrido `pending → processing → completed`;
- que las tareas se procesan en orden;
- que una tarea que falla queda en `failed` y el worker sigue con la siguiente;
- el 503 cuando Redis no está disponible;
- que el POST devuelve solo los campos del contrato (`job_id` y `status`) y que el esquema OpenAPI documenta `TaskResponse` y `JobStatus`.
