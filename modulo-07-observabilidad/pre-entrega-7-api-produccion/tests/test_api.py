"""The whole system without external services: API + worker + graph + HITL, with fakeredis and the offline
model (LLM_PROVIDER=fake). Every test runs the real code paths: Redis queue, checkpoints, interrupts, statuses."""

import asyncio
import json
import time

import fakeredis
import httpx
import pytest
from langgraph.graph import END, START, StateGraph

from app import main
from app.checkpointer import AsyncRedisCheckpointer
from app.config import Settings
from app.graph import OUTBOX_KEY, build_task_graph
from app.hitl import assess_risk
from app.jobs import QUEUE_KEY, JobStore
from app.schemas import JobStatus
from app.worker import Worker

ONE_PRODUCT = "¿Qué opinan los clientes de los SoundMax Pro y cuánto cuestan?"
TWO_PRODUCTS = "Compará SoundMax Pro y AudioZen X: sentimiento y precio promedio"
THREE_PRODUCTS = "Armá un ranking de SoundMax Pro, AudioZen X y BassBoom 3 según el sentimiento"


@pytest.fixture(autouse=True)
def offline_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("FAKE_LLM_LATENCY_SECONDS", "0")
    import app.orchestrator.data.sources as sources
    monkeypatch.setattr(sources, "LATENCY_SECONDS", 0)


def settings(**overrides) -> Settings:
    return Settings(**{"phoenix_enabled": False, "worker_concurrency": 5, "job_timeout_seconds": 30, **overrides})


class System:
    """What the API lifespan builds, on a fakeredis: jobs, checkpointer, graph and worker."""

    def __init__(self, redis, cfg: Settings, orchestrator=None):
        self.redis, self.settings = redis, cfg
        self.jobs = JobStore(redis, cfg.job_ttl_seconds)
        self.graph = build_task_graph(AsyncRedisCheckpointer(redis), redis, cfg, orchestrator)
        self.worker = Worker(self.jobs, self.graph, cfg)
        main.app.state.settings, main.app.state.redis, main.app.state.graph = cfg, redis, self.graph
        main.app.state.jobs, main.app.state.worker = self.jobs, self.worker

    async def drain(self) -> None:
        """Processes every queued message (what the worker loops do), one at a time."""
        while message := await self.jobs.dequeue(timeout=0.01):
            await self.worker.process(message)


@pytest.fixture
async def redis():
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
async def system(redis):
    return System(redis, settings())


@pytest.fixture
async def client(system):
    transport = httpx.ASGITransport(app=main.app)   # no lifespan: the System fixture plays its part
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def post(client, task, **extra) -> str:
    response = await client.post("/tasks", json={"task": task, **extra})
    assert response.status_code == 202
    return response.json()["job_id"]


async def status_of(client, job_id) -> dict:
    return (await client.get(f"/tasks/{job_id}")).json()


# ------------------------------------------------------------------ async API + Redis
async def test_post_enqueues_and_answers_without_running_anything(client, redis):
    started = time.perf_counter()
    response = await client.post("/tasks", json={"task": TWO_PRODUCTS})
    assert response.status_code == 202 and time.perf_counter() - started < 0.5
    body = response.json()
    assert body["status"] == "PENDING" and body["status_url"] == f"/tasks/{body['job_id']}"
    # the record exists BEFORE a worker takes it, and the job is in the queue
    assert (await status_of(client, body["job_id"]))["status"] == "PENDING"
    assert json.loads((await redis.lrange(QUEUE_KEY, 0, -1))[0]) == {"job_id": body["job_id"], "action": "start"}


async def test_worker_runs_the_multi_agent_system(client, system):
    job_id = await post(client, TWO_PRODUCTS)
    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "DONE" and job["error"] is None
    assert job["result"]["flow"] == ["researcher", "analyst", "synthesizer", "FINISH"]
    assert {p["product"] for p in job["result"]["products"]} == {"SoundMax Pro", "AudioZen X"}
    assert "SoundMax Pro" in job["result"]["final_answer"] and job["run_seconds"] > 0
    assert job["risk"]["critical"] is False and job["started_at"] <= job["finished_at"]


async def test_unknown_job_and_invalid_bodies(client):
    assert (await client.get("/tasks/no-existe")).status_code == 404
    assert (await client.post("/tasks", json={"task": "hey"})).status_code == 422                 # too short
    assert (await client.post("/tasks", json={"task": TWO_PRODUCTS, "deliver_to": "no-es-un-mail"})).status_code == 422
    assert (await client.post("/tasks/no-existe/approve", json={"approved": True, "reviewer": "x"})).status_code == 404


async def test_health_and_list(client, system):
    await post(client, ONE_PRODUCT)
    health = (await client.get("/health")).json()
    assert health["redis"] == "ok" and health["queued_jobs"] == 1
    assert len((await client.get("/tasks")).json()) == 1


# ------------------------------------------------------------------ FAILED, never silent
def exploding_orchestrator():
    def boom(state):
        raise RuntimeError("el proveedor del LLM devolvió 500")

    builder = StateGraph(dict)
    builder.add_node("boom", boom)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    return builder.compile()


async def test_an_exception_in_the_background_marks_the_job_failed(client, redis):
    system = System(redis, settings(), orchestrator=exploding_orchestrator())
    job_id = await post(client, ONE_PRODUCT)
    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "FAILED" and job["error"] == "RuntimeError: el proveedor del LLM devolvió 500"
    assert job["finished_at"] is not None


async def test_a_hung_job_times_out_as_failed(client, redis):
    async def hang(state):
        await asyncio.sleep(10)
        return {}

    builder = StateGraph(dict)
    builder.add_node("hang", hang)
    builder.add_edge(START, "hang")
    builder.add_edge("hang", END)
    system = System(redis, settings(job_timeout_seconds=0.2), orchestrator=builder.compile())
    job_id = await post(client, ONE_PRODUCT)
    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "FAILED" and job["error"].startswith("TimeoutError")


# ------------------------------------------------------------------ human-in-the-loop
def test_risk_policy_is_deterministic():
    cfg = settings()
    assert not assess_risk(ONE_PRODUCT, None, cfg).critical
    assert not assess_risk(TWO_PRODUCTS, None, cfg).critical
    risk = assess_risk(THREE_PRODUCTS, "a@b.com", cfg)
    assert risk.critical and risk.estimated_cost_usd == 0.0073 and risk.needs_delivery_approval
    assert "umbral" in risk.reasons[0]


async def test_critical_task_waits_and_runs_only_after_approval(client, system, redis):
    job_id = await post(client, THREE_PRODUCTS)
    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "WAITING_APPROVAL" and job["result"] is None
    assert job["approval_request"]["stage"] == "execution" and job["approval_request"]["estimated_cost_usd"] == 0.0073

    decision = {"approved": True, "reviewer": "javier", "comment": "costo aceptable"}
    response = await client.post(f"/tasks/{job_id}/approve", json=decision)
    assert response.status_code == 202 and response.json()["status"] == "PENDING"
    second = await client.post(f"/tasks/{job_id}/approve", json=decision)
    assert second.status_code == 409                       # it is not waiting anymore: no double approval

    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "DONE" and len(job["result"]["products"]) == 3
    assert [(a["stage"], a["approved"], a["reviewer"]) for a in job["approvals"]] == [("execution", True, "javier")]
    assert job["approval_request"] is None


async def test_rejected_task_is_never_run(client, system):
    job_id = await post(client, THREE_PRODUCTS)
    await system.drain()
    await client.post(f"/tasks/{job_id}/approve", json={"approved": False, "reviewer": "javier", "comment": "caro"})
    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "REJECTED" and job["result"] is None
    snapshot = await system.graph.aget_state({"configurable": {"thread_id": job_id}})
    assert "final_answer" not in snapshot.values           # the orchestrator never ran: no tokens spent


async def test_report_is_sent_only_after_approving_the_draft(client, system, redis):
    job_id = await post(client, ONE_PRODUCT, deliver_to="cliente@ejemplo.com")
    await system.drain()
    job = await status_of(client, job_id)
    request = job["approval_request"]
    assert job["status"] == "WAITING_APPROVAL" and request["stage"] == "delivery"
    assert "SoundMax Pro" in request["draft"] and request["deliver_to"] == "cliente@ejemplo.com"
    assert await redis.llen(OUTBOX_KEY) == 0               # nothing sent while waiting

    await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "reviewer": "javier"})
    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "DONE" and job["result"]["delivered_to"] == "cliente@ejemplo.com"
    sent = (await client.get("/outbox")).json()
    assert len(sent) == 1 and sent[0]["job_id"] == job_id and sent[0]["body"] == request["draft"]


async def test_rejected_delivery_keeps_the_answer_but_sends_nothing(client, system, redis):
    job_id = await post(client, ONE_PRODUCT, deliver_to="cliente@ejemplo.com")
    await system.drain()
    await client.post(f"/tasks/{job_id}/approve", json={"approved": False, "reviewer": "javier"})
    await system.drain()
    job = await status_of(client, job_id)
    assert job["status"] == "DONE" and job["result"]["delivered_to"] is None and job["result"]["final_answer"]
    assert await redis.llen(OUTBOX_KEY) == 0


async def test_approval_works_after_a_restart(client, redis):
    first = System(redis, settings())
    job_id = await post(client, THREE_PRODUCTS)
    await first.drain()
    assert (await status_of(client, job_id))["status"] == "WAITING_APPROVAL"

    # "Restart": everything in memory is rebuilt; only Redis remains (job record + checkpoints)
    restarted = System(redis, settings())
    await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "reviewer": "javier"})
    await restarted.drain()
    assert (await status_of(client, job_id))["status"] == "DONE"


# ------------------------------------------------------------------ concurrency
async def test_five_concurrent_jobs_run_in_parallel(client, redis, monkeypatch):
    monkeypatch.setenv("FAKE_LLM_LATENCY_SECONDS", "0.05")
    system = System(redis, settings())
    one = await post(client, ONE_PRODUCT)                   # one job alone, as a reference
    started = time.perf_counter()
    await system.worker.process(await system.jobs.dequeue())
    single = time.perf_counter() - started
    assert (await status_of(client, one))["status"] == "DONE"

    # fakeredis implements the blocking BLPOP in a way that serializes the waiting loops (a real Redis does not:
    # see results/load_test.md). In this test the workers take messages with a non-blocking LPOP instead.
    async def dequeue(timeout: float = 1.0):
        item = await redis.lpop(QUEUE_KEY)
        if item is None:
            await asyncio.sleep(0.01)
        return json.loads(item) if item else None

    monkeypatch.setattr(system.jobs, "dequeue", dequeue)
    ids = await asyncio.gather(*(post(client, ONE_PRODUCT) for _ in range(5)))   # 5 POST at the same time
    started = time.perf_counter()
    system.worker.start(5)
    try:
        while time.perf_counter() - started < 10:
            statuses = [(await status_of(client, i))["status"] for i in ids]
            if all(s == "DONE" for s in statuses):
                break
            await asyncio.sleep(0.02)
    finally:
        await system.worker.stop()
    elapsed = time.perf_counter() - started
    assert all(s == "DONE" for s in statuses)
    assert elapsed < single * 2, (elapsed, single)          # sequentially they would take ~5x one job


def test_llm_calls_share_one_rate_limiter(monkeypatch):
    import app.orchestrator.agents.llm as llm
    monkeypatch.setattr(llm, "_rate_limiter", None)
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "12")
    supervisor, worker = llm.chat_models("supervisor")[0], llm.chat_models("worker")[0]
    assert supervisor.rate_limiter is not None and supervisor.rate_limiter is worker.rate_limiter
    monkeypatch.setattr(llm, "_rate_limiter", None)
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "0")
    assert llm.chat_models("worker")[0].rate_limiter is None


def test_rate_limit_defaults_to_the_gemini_free_tier(monkeypatch):
    import app.orchestrator.agents.llm as llm
    monkeypatch.delenv("LLM_REQUESTS_PER_MINUTE", raising=False)
    monkeypatch.setattr(llm, "_rate_limiter", None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm.shared_rate_limiter() is not None              # free tier: on by default
    monkeypatch.setattr(llm, "_rate_limiter", None)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert llm.shared_rate_limiter() is None                  # offline model: no limit
