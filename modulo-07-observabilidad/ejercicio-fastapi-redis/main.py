"""Asynchronous processing with FastAPI + Redis: the API does not run the agent inside the request.

Flow:  POST /process  →  Redis (status key + queue)  →  worker (runs the job)  →  GET /status/{job_id} (polling)

- The API answers in milliseconds with a job_id: it only writes the initial status and enqueues the job.
- The worker takes job_ids from the queue with BLPOP (blocks until there is work, without busy waiting)
  and stores each state change in Redis: pending → processing → completed (or failed).
- The client polls /status/{job_id} until the status is completed or failed.

Run (with Redis on localhost:6379):
    uvicorn main:app --reload
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from redis import asyncio as aioredis
from redis.exceptions import RedisError

# Configuración de Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_NAME = "tasks_queue"
STATUS_PREFIX = "status:"

# How long the worker "works" on each job (simulates a LangGraph run) and how long a job is kept in Redis
PROCESSING_SECONDS = float(os.getenv("PROCESSING_SECONDS", "5"))
JOB_TTL_SECONDS = 24 * 60 * 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("jobs")

app = FastAPI(title="Agent jobs API")
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)


JobState = Literal["pending", "processing", "completed", "failed"]


class TaskRequest(BaseModel):
    query: str = Field(min_length=1, description="Request for the agent")


class TaskResponse(BaseModel):
    """Answer of POST /process: the ticket the client uses to poll the job."""
    job_id: str = Field(description="Unique id of the job (uuid4)")
    status: JobState = Field(description="Always 'pending' when the job is created")


class JobStatus(BaseModel):
    """Answer of GET /status/{job_id}: the job's record as stored in Redis."""
    job_id: str
    status: JobState
    query: str
    result: str | None = Field(default=None, description="Agent's answer, once the status is 'completed'")
    error: str | None = Field(default=None, description="Error message, if the status is 'failed'")
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None


def status_key(job_id: str) -> str:
    return f"{STATUS_PREFIX}{job_id}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def save_status(job_id: str, **fields: Any) -> dict[str, Any]:
    """Merges `fields` into the job's JSON record in Redis and returns the new record."""
    raw = await redis_client.get(status_key(job_id))
    record = json.loads(raw) if raw else {"job_id": job_id}
    record.update(fields, updated_at=now())
    await redis_client.set(status_key(job_id), json.dumps(record, ensure_ascii=False), ex=JOB_TTL_SECONDS)
    return record


@app.post("/process", status_code=202, response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """Registers the job and enqueues it. Answers at once with the job_id: the work happens in the worker.

    1. Generates a unique job_id (uuid4).
    2. Saves the initial status ("pending") in Redis BEFORE enqueuing: if the worker were faster than this
       write, a client polling right away would get a 404 for a job that does exist.
    3. Pushes the job_id to the queue (RPUSH: the worker takes from the other end with BLPOP → FIFO).
    4. Returns a TaskResponse {"job_id", "status": "pending"} with 202 Accepted (accepted, not finished).
       `response_model` validates that output and documents it in the OpenAPI schema (/docs).
    """
    job_id = str(uuid.uuid4())
    try:
        await save_status(job_id, status="pending", query=request.query, result=None,
                          error=None, created_at=now())
        await redis_client.rpush(QUEUE_NAME, job_id)
    except RedisError as e:
        logger.error("could not enqueue job %s: %s", job_id, e)
        raise HTTPException(status_code=503, detail="La cola de tareas no está disponible. Intentá más tarde.")
    logger.info("job %s enqueued", job_id)
    return TaskResponse(job_id=job_id, status="pending")


@app.get("/status/{job_id}", response_model=JobStatus, responses={404: {"description": "The job does not exist"}})
async def get_status(job_id: str):
    """Returns the job's record as a JobStatus (status, query, result, error and timestamps). 404 if it does not exist.
    Validating the record with the model means a corrupt entry in Redis fails here, not in the client."""
    try:
        raw = await redis_client.get(status_key(job_id))
    except RedisError as e:
        logger.error("could not read job %s: %s", job_id, e)
        raise HTTPException(status_code=503, detail="No se pudo consultar el estado. Intentá más tarde.")
    if raw is None:
        raise HTTPException(status_code=404, detail=f"No existe la tarea {job_id}")
    return JobStatus.model_validate_json(raw)


async def process_query(query: str) -> str:
    """Simulates the LangGraph run. In the real system this would be `await graph.ainvoke(...)`."""
    await asyncio.sleep(PROCESSING_SECONDS)
    return f"Resultado para '{query}': procesado por el agente en {PROCESSING_SECONDS:g} s."


async def handle_job(job_id: str) -> None:
    """Runs one job and records each state change. A failing job is marked 'failed' and does not stop the worker."""
    raw = await redis_client.get(status_key(job_id))
    if raw is None:   # expired or deleted while it was waiting in the queue
        logger.warning("job %s has no status record; skipping it", job_id)
        return
    job = json.loads(raw)
    await save_status(job_id, status="processing", started_at=now())
    logger.info("job %s processing", job_id)
    try:
        result = await process_query(job["query"])
    except Exception as e:   # the agent failed: the error is stored for the client, the worker goes on
        logger.exception("job %s failed", job_id)
        await save_status(job_id, status="failed", error=f"{type(e).__name__}: {e}", finished_at=now())
        return
    await save_status(job_id, status="completed", result=result, finished_at=now())
    logger.info("job %s completed", job_id)


async def task_worker():
    """Infinite loop: takes a job_id from the queue with BLPOP and processes it.

    BLPOP with timeout=0 blocks until a job arrives, so an idle worker uses no CPU and does not poll Redis.
    If Redis drops, the worker waits and retries instead of dying: the API would keep enqueuing jobs that
    nobody processes.
    """
    print("Worker iniciado...")
    while True:
        try:
            _, job_id = await redis_client.blpop(QUEUE_NAME, timeout=0)
            await handle_job(job_id)
        except asyncio.CancelledError:   # the API is shutting down
            logger.info("worker stopped")
            raise
        except RedisError as e:
            logger.error("Redis error in the worker: %s; retrying in 2 s", e)
            await asyncio.sleep(2)
        except Exception:   # e.g. a corrupt record: log it and keep serving the queue
            logger.exception("unexpected error in the worker")


# El worker arranca junto con la API, en el mismo event loop que usa uvicorn
@app.on_event("startup")
async def start_worker():
    asyncio.create_task(task_worker())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
