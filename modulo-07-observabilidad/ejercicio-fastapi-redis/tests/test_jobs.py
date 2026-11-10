"""Tests of the API and the worker against an in-memory Redis (fakeredis): no Redis server needed."""

import asyncio
import sys
from pathlib import Path

import fakeredis
import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def redis(monkeypatch):
    fake = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(main, "redis_client", fake)
    monkeypatch.setattr(main, "PROCESSING_SECONDS", 0.2)
    yield fake
    await fake.aclose()


@pytest.fixture
async def client(redis):
    # ASGITransport does not fire the startup event: each test starts the worker itself when it needs it
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def worker(redis):
    task = asyncio.create_task(main.task_worker())
    yield task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def wait_for_status(client, job_id, wanted, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        body = (await client.get(f"/status/{job_id}")).json()
        if body["status"] == wanted:
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached '{wanted}' (last: {body['status']})")


async def test_create_task_returns_at_once_and_enqueues(client, redis):
    response = await client.post("/process", json={"query": "hola"})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    # the status record exists BEFORE the worker takes the job, and the job_id is in the queue
    assert await redis.lrange(main.QUEUE_NAME, 0, -1) == [body["job_id"]]
    status = (await client.get(f"/status/{body['job_id']}")).json()
    assert status["status"] == "pending" and status["query"] == "hola" and status["result"] is None


async def test_unknown_job_is_404(client):
    response = await client.get("/status/no-existe")
    assert response.status_code == 404


async def test_empty_query_is_rejected(client):
    response = await client.post("/process", json={"query": ""})
    assert response.status_code == 422


async def test_worker_goes_pending_processing_completed(client, worker):
    job_id = (await client.post("/process", json={"query": "hola"})).json()["job_id"]
    await wait_for_status(client, job_id, "processing")
    done = await wait_for_status(client, job_id, "completed")
    assert "hola" in done["result"]
    assert done["error"] is None and done["finished_at"] >= done["started_at"]


async def test_jobs_are_processed_in_order(client, worker, monkeypatch):
    monkeypatch.setattr(main, "PROCESSING_SECONDS", 0.05)
    ids = [(await client.post("/process", json={"query": f"q{n}"})).json()["job_id"] for n in range(3)]
    finished = [await wait_for_status(client, job_id, "completed") for job_id in ids]
    assert [f["finished_at"] for f in finished] == sorted(f["finished_at"] for f in finished)


async def test_failed_job_is_marked_and_worker_keeps_going(client, worker, monkeypatch):
    real = main.process_query

    async def flaky(query: str) -> str:
        if query == "rompe":
            raise RuntimeError("el agente falló")
        return await real(query)

    monkeypatch.setattr(main, "process_query", flaky)
    bad = (await client.post("/process", json={"query": "rompe"})).json()["job_id"]
    good = (await client.post("/process", json={"query": "sigue"})).json()["job_id"]
    failed = await wait_for_status(client, bad, "failed")
    assert failed["error"] == "RuntimeError: el agente falló"
    await wait_for_status(client, good, "completed")


async def test_redis_down_returns_503(client, monkeypatch):
    async def down(*args, **kwargs):
        raise RedisConnectionError("Redis caído")

    monkeypatch.setattr(main.redis_client, "get", down)
    assert (await client.post("/process", json={"query": "hola"})).status_code == 503
    assert (await client.get("/status/abc")).status_code == 503


async def test_openapi_documents_the_response_models(client):
    schema = (await client.get("/openapi.json")).json()
    post = schema["paths"]["/process"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]
    get = schema["paths"]["/status/{job_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert post["$ref"].endswith("/TaskResponse") and get["$ref"].endswith("/JobStatus")
    states = schema["components"]["schemas"]["TaskResponse"]["properties"]["status"]["enum"]
    assert states == ["pending", "processing", "completed", "failed"]


async def test_post_response_has_only_the_contract_fields(client):
    body = (await client.post("/process", json={"query": "hola"})).json()
    assert set(body) == {"job_id", "status"}
