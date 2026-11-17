"""Job store in Redis: one record per job, the work queue and safe status transitions.

    job:{job_id}     STRING  the JobView as JSON (status, task, result, error, approvals...)
    jobs:queue       LIST    messages for the workers: {"job_id", "action": "start" | "resume", "decision"?}
    jobs:index       ZSET    job ids by creation time (for GET /tasks)

Status changes go through `transition`, a compare-and-set (WATCH/MULTI): it only changes the status if the
current one is the expected one. That avoids races such as two reviewers approving the same job at the same
time, or a worker starting a job that was already taken.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from redis import asyncio as aioredis
from redis.exceptions import WatchError

from app.schemas import JobStatus, JobView, now_iso

logger = logging.getLogger("app.jobs")

QUEUE_KEY = "jobs:queue"
INDEX_KEY = "jobs:index"


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


class InvalidTransition(Exception):
    """The job is not in the status the operation needs (for example, approving a job that is not waiting)."""

    def __init__(self, job_id: str, current: JobStatus | None, expected: set[JobStatus]):
        self.current = current
        wanted = ", ".join(sorted(s.value for s in expected))
        super().__init__(f"El job {job_id} está en {current.value if current else 'ningún estado (no existe)'}; "
                         f"se esperaba {wanted}")


class JobStore:
    def __init__(self, client: aioredis.Redis, ttl_seconds: int):
        self.client = client
        self.ttl_seconds = ttl_seconds

    async def create(self, task: str, deliver_to: str | None) -> JobView:
        """Saves the job as PENDING and enqueues it, in ONE transaction: the record always exists before any
        worker can take the job (otherwise an early poll could get a 404 for a job that does exist)."""
        now = now_iso()
        job = JobView(job_id=str(uuid.uuid4()), status=JobStatus.PENDING, task=task, deliver_to=deliver_to,
                      created_at=now, updated_at=now)
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.set(job_key(job.job_id), job.model_dump_json(), ex=self.ttl_seconds)
            pipe.zadd(INDEX_KEY, {job.job_id: _epoch(now)})
            pipe.rpush(QUEUE_KEY, json.dumps({"job_id": job.job_id, "action": "start"}))
            await pipe.execute()
        return job

    async def get(self, job_id: str) -> JobView | None:
        raw = await self.client.get(job_key(job_id))
        return JobView.model_validate_json(raw) if raw else None

    async def list(self, limit: int = 20) -> list[JobView]:
        ids = await self.client.zrevrange(INDEX_KEY, 0, limit - 1)
        raws = await self.client.mget([job_key(i) for i in ids]) if ids else []
        return [JobView.model_validate_json(r) for r in raws if r]

    async def transition(self, job_id: str, expected: set[JobStatus] | None, new_status: JobStatus | None = None,
                         **fields: Any) -> JobView:
        """Updates the job atomically. With `expected`, only if its current status is one of those."""
        key = job_key(job_id)
        while True:
            async with self.client.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    current = JobView.model_validate_json(raw) if raw else None
                    if current is None or (expected is not None and current.status not in expected):
                        await pipe.unwatch()
                        raise InvalidTransition(job_id, current.status if current else None, expected or set())
                    data = current.model_dump()
                    data.update(fields, updated_at=now_iso())
                    if new_status is not None:
                        data["status"] = new_status
                    updated = JobView.model_validate(data)
                    pipe.multi()
                    pipe.set(key, updated.model_dump_json(), ex=self.ttl_seconds)
                    await pipe.execute()
                    return updated
                except WatchError:   # someone else changed the job in between: read it again and retry
                    continue

    async def approve(self, job_id: str, decision: dict[str, Any]) -> JobView:
        """WAITING_APPROVAL → PENDING and enqueue the resume. Raises InvalidTransition if it was not waiting."""
        job = await self.transition(job_id, {JobStatus.WAITING_APPROVAL}, JobStatus.PENDING, approval_request=None)
        await self.client.rpush(QUEUE_KEY, json.dumps({"job_id": job_id, "action": "resume", "decision": decision}))
        return job

    async def dequeue(self, timeout: float = 1.0) -> dict[str, Any] | None:
        """Next message of the queue. BLPOP blocks until there is one (or `timeout` passes): no busy polling."""
        item = await self.client.blpop([QUEUE_KEY], timeout=timeout)
        return json.loads(item[1]) if item else None

    async def queue_length(self) -> int:
        return await self.client.llen(QUEUE_KEY)


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()
