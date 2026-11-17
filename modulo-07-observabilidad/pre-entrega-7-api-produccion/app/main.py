"""Production API for the multi-agent system: async endpoints, Redis, HITL and Phoenix traces.

    POST /tasks                 enqueue a task → 202 + job_id, in milliseconds (the agents run in a worker)
    GET  /tasks/{job_id}        status and result (polling)
    GET  /tasks                 latest jobs
    POST /tasks/{job_id}/approve   human decision for a job in WAITING_APPROVAL (hitl.py)
    GET  /outbox                reports "sent by email" (the simulated side effect)
    GET  /health                Redis, workers and queue

No endpoint runs the agents: they only read/write Redis with the async client, so the event loop is never
blocked and the API answers while the workers are busy.

Run:  uvicorn app.main:app --reload       (Redis on REDIS_URL, Phoenix on PHOENIX_COLLECTOR_ENDPOINT)
"""

import inspect
import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, status
from redis import asyncio as aioredis
from redis.exceptions import RedisError

from app.checkpointer import open_checkpointer
from app.config import load_settings
from app.graph import OUTBOX_KEY, build_task_graph
from app.hitl import router as hitl_router
from app.jobs import JobStore
from app.observability import setup_tracing, shutdown_tracing
from app.schemas import JobView, TaskCreated, TaskRequest
from app.worker import Worker

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: tracing → Redis → checkpointer → graph → workers. Shutdown in reverse order."""
    settings = load_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    setup_tracing(settings)   # before building the graph, so every component is traced
    async with AsyncExitStack() as stack:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        stack.push_async_callback(redis.aclose)
        checkpointer = await open_checkpointer(settings.checkpointer, settings.redis_url, redis, stack,
                                               settings.checkpoint_ttl_seconds)
        graph = build_task_graph(checkpointer, redis, settings)
        app.state.settings, app.state.redis, app.state.graph = settings, redis, graph
        app.state.jobs = JobStore(redis, settings.job_ttl_seconds)
        app.state.worker = None
        if settings.run_worker_in_api:
            app.state.worker = Worker(app.state.jobs, graph, settings)
            app.state.worker.start()
        logger.info("API ready (worker in process: %s)", settings.run_worker_in_api)
        try:
            yield
        finally:
            if app.state.worker:
                await app.state.worker.stop()
    shutdown_tracing()


# FastAPI ≥ 0.16x traces every HTTP request by itself when there is a global OpenTelemetry provider. Those spans
# (one per poll of GET /tasks/{id}) would be separate ~5 ms traces in Phoenix and drag down the p50/p95 of the
# project, which should measure agent EXECUTIONS. They are turned off; the API latency is measured by the load test.
_options = {"telemetry": {"tracing": False}} if "telemetry" in inspect.signature(FastAPI.__init__).parameters else {}
app = FastAPI(title="API del sistema multi-agente", version="1.0.0", lifespan=lifespan,
              description="Pre-entrega 7: orquestador multi-agente con cola en Redis, aprobación humana y trazas en Phoenix.",
              **_options)
app.include_router(hitl_router)


def redis_unavailable(error: RedisError) -> HTTPException:
    logger.error("Redis error: %s", error)
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                         detail="Redis no está disponible. Intentá de nuevo en unos segundos.")


@app.post("/tasks", response_model=TaskCreated, status_code=status.HTTP_202_ACCEPTED, tags=["tasks"])
async def create_task(body: TaskRequest, request: Request) -> TaskCreated:
    """Registers the job (PENDING) and enqueues it. Answers at once: the work happens in a worker."""
    try:
        job = await request.app.state.jobs.create(body.task, body.deliver_to)
    except RedisError as e:
        raise redis_unavailable(e)
    return TaskCreated(job_id=job.job_id, status=job.status, status_url=f"/tasks/{job.job_id}")


@app.get("/tasks/{job_id}", response_model=JobView, tags=["tasks"],
         responses={404: {"description": "The job does not exist (or expired)"}})
async def get_task(job_id: str, request: Request) -> JobView:
    try:
        job = await request.app.state.jobs.get(job_id)
    except RedisError as e:
        raise redis_unavailable(e)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No existe el job {job_id}")
    return job


@app.get("/tasks", response_model=list[JobView], tags=["tasks"])
async def list_tasks(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[JobView]:
    try:
        return await request.app.state.jobs.list(limit)
    except RedisError as e:
        raise redis_unavailable(e)


@app.get("/outbox", tags=["human-in-the-loop"])
async def outbox(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    """Reports sent by email (simulated). They only show up after a human approved the delivery."""
    try:
        return [json.loads(m) for m in await request.app.state.redis.lrange(OUTBOX_KEY, -limit, -1)]
    except RedisError as e:
        raise redis_unavailable(e)


@app.get("/health", tags=["ops"])
async def health(request: Request) -> dict:
    state = request.app.state
    try:
        await state.redis.ping()
        queued = await state.jobs.queue_length()
    except RedisError as e:
        raise redis_unavailable(e)
    return {"status": "ok", "redis": "ok", "checkpointer": state.settings.checkpointer, "queued_jobs": queued,
            "worker_loops": state.worker.size if state.worker else 0,
            "tracing": state.settings.phoenix_project if state.settings.phoenix_enabled else "disabled"}
