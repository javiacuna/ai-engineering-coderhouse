"""Worker: takes jobs from the Redis queue, runs (or resumes) the graph and records every status change.

    PENDING → RUNNING → DONE
                      → WAITING_APPROVAL → (POST /approve) → PENDING → RUNNING → DONE / REJECTED
                      → FAILED   (any exception or timeout: the client never polls forever)

It runs in two ways:
- inside the API (RUN_WORKER_IN_API=true, the default): WORKER_CONCURRENCY coroutines in the same event loop
  as FastAPI. Everything they do is async (Redis, Gemini, the graph), so the API keeps answering.
- as a separate process: `python -m app.worker` (with RUN_WORKER_IN_API=false in the API). Several worker
  processes can share the queue: BLPOP hands each message to only one of them.
"""

import asyncio
import logging
import signal
import time
import traceback
from contextlib import AsyncExitStack
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from opentelemetry.trace import Status, StatusCode
from redis import asyncio as aioredis
from redis.exceptions import RedisError

from app.config import Settings, load_settings
from app.jobs import InvalidTransition, JobStore
from app.observability import job_span, setup_tracing, shutdown_tracing, trace_id_of
from app.schemas import ApprovalRequest, JobStatus, JobView, ProductSummary, TaskResult, now_iso

logger = logging.getLogger("app.worker")


class Worker:
    def __init__(self, jobs: JobStore, graph: CompiledStateGraph, settings: Settings):
        self.jobs = jobs
        self.graph = graph
        self.settings = settings
        self._stopping = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self, concurrency: int | None = None) -> None:
        n = concurrency or self.settings.worker_concurrency
        self._tasks = [asyncio.create_task(self._loop(i), name=f"worker-{i}") for i in range(n)]
        logger.info("%d worker loops started", n)

    async def stop(self, grace_seconds: float = 10) -> None:
        """Stops taking messages and gives the running jobs `grace_seconds` to finish; the rest are cancelled
        (and marked FAILED, so no client keeps polling a job that will never end)."""
        self._stopping.set()
        if not self._tasks:
            return
        _, pending = await asyncio.wait(self._tasks, timeout=grace_seconds)
        for task in pending:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    @property
    def size(self) -> int:
        return sum(not t.done() for t in self._tasks)

    async def _loop(self, number: int) -> None:
        while not self._stopping.is_set():
            try:
                message = await self.jobs.dequeue(timeout=1)   # timeout: check _stopping every second
                if message:
                    await self.process(message)
            except RedisError as e:   # Redis down: wait and retry, don't let the loop die
                logger.error("worker %d: Redis error %s; retrying in 2 s", number, e)
                await asyncio.sleep(2)
            except Exception:         # a bug while handling one message must not kill the loop
                logger.exception("worker %d: unexpected error", number)

    # ------------------------------------------------------------------ one message
    async def process(self, message: dict[str, Any]) -> JobView | None:
        job_id, action = message["job_id"], message.get("action", "start")
        try:
            job = await self.jobs.transition(job_id, {JobStatus.PENDING}, JobStatus.RUNNING, error=None)
        except InvalidTransition as e:   # expired, deleted or already taken: nothing to do
            logger.warning("skipping message %s: %s", message, e)
            return None
        if job.started_at is None:       # first run (a resume keeps the original start)
            job = await self.jobs.transition(job_id, None, started_at=now_iso())

        config = {"configurable": {"thread_id": job_id}}
        graph_input: Any = ({"task": job.task, "deliver_to": job.deliver_to} if action == "start"
                            else Command(resume=message["decision"]))
        started = time.perf_counter()
        trace_id = None
        try:
            with job_span(job_id, action, job.task) as span:
                trace_id = trace_id_of(span)
                try:
                    # The timeout turns a hung LLM call into FAILED instead of a job RUNNING forever
                    async with asyncio.timeout(self.settings.job_timeout_seconds):
                        await self.graph.ainvoke(graph_input, config)
                    snapshot = await self.graph.aget_state(config)
                except BaseException as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)[:200]))
                    raise
                span.set_attribute("output.value", self._summary(snapshot))
                span.set_status(Status(StatusCode.OK))
        except asyncio.CancelledError:
            await self._fail(job_id, "El worker se detuvo mientras corría el job", started, trace_id)
            raise
        except TimeoutError:
            return await self._fail(job_id, f"TimeoutError: el job superó {self.settings.job_timeout_seconds:.0f} s",
                                    started, trace_id)
        except Exception as e:   # the agent failed in the background: FAILED with the error, never silent
            logger.error("job %s failed:\n%s", job_id, traceback.format_exc())
            return await self._fail(job_id, f"{type(e).__name__}: {str(e)[:500]}", started, trace_id)
        return await self._finish(job_id, snapshot, started, trace_id)

    async def _finish(self, job_id: str, snapshot, started: float, trace_id: str | None) -> JobView:
        values = snapshot.values
        common = await self._common(job_id, started, trace_id, values)
        if snapshot.interrupts:   # paused by a HITL node: the request for the reviewer is the interrupt value
            request = ApprovalRequest.model_validate(snapshot.interrupts[0].value)
            logger.info("job %s waiting for approval (%s)", job_id, request.stage)
            return await self.jobs.transition(job_id, {JobStatus.RUNNING}, JobStatus.WAITING_APPROVAL,
                                              approval_request=request, **common)
        status = JobStatus.REJECTED if values.get("outcome") == "rejected" else JobStatus.DONE
        result = None
        if status == JobStatus.DONE:
            result = TaskResult(final_answer=values.get("final_answer"), flow=values.get("flow", []),
                                products=[ProductSummary(**p) for p in values.get("products", [])],
                                delivered_to=values.get("delivered_to"))
        logger.info("job %s %s", job_id, status.value)
        return await self.jobs.transition(job_id, {JobStatus.RUNNING}, status, result=result,
                                          finished_at=now_iso(), **common)

    async def _fail(self, job_id: str, error: str, started: float, trace_id: str | None) -> JobView | None:
        try:
            common = await self._common(job_id, started, trace_id, {})
            return await self.jobs.transition(job_id, None, JobStatus.FAILED, error=error, finished_at=now_iso(),
                                              approval_request=None, **common)
        except Exception:   # even if Redis is down, log it: the job will look stuck in RUNNING
            logger.exception("could not mark job %s as FAILED", job_id)
            return None

    async def _common(self, job_id: str, started: float, trace_id: str | None, values: dict[str, Any]) -> dict[str, Any]:
        job = await self.jobs.get(job_id)
        fields: dict[str, Any] = {"run_seconds": round(job.run_seconds + time.perf_counter() - started, 3),
                                  "trace_ids": job.trace_ids + ([trace_id] if trace_id else [])}
        if values.get("risk"):
            fields["risk"] = values["risk"]
        if values.get("approvals"):
            fields["approvals"] = values["approvals"]
        return fields

    @staticmethod
    def _summary(snapshot) -> str:
        if snapshot.interrupts:
            return f"WAITING_APPROVAL ({snapshot.interrupts[0].value.get('stage')})"
        return snapshot.values.get("final_answer") or snapshot.values.get("outcome", "")


# --------------------------------------------------------------------------- standalone process
async def main() -> None:
    """`python -m app.worker`: a worker process separate from the API (start the API with RUN_WORKER_IN_API=false)."""
    from app.checkpointer import open_checkpointer
    from app.graph import build_task_graph

    settings = load_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    setup_tracing(settings)
    async with AsyncExitStack() as stack:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        stack.push_async_callback(redis.aclose)
        checkpointer = await open_checkpointer(settings.checkpointer, settings.redis_url, redis, stack,
                                               settings.checkpoint_ttl_seconds)
        worker = Worker(JobStore(redis, settings.job_ttl_seconds), build_task_graph(checkpointer, redis, settings), settings)
        worker.start()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        logger.info("worker process ready (Ctrl+C to stop)")
        await stop.wait()
        await worker.stop()
    shutdown_tracing()


if __name__ == "__main__":
    asyncio.run(main())
