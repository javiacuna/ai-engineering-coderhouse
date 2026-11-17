"""Load test: 5 concurrent requests against the API, then cost and latency from Phoenix.

What it does:
1. Sends the 5 POST /tasks AT THE SAME TIME (asyncio.gather) and measures how long each POST took: since the
   API only enqueues, they should take milliseconds even with the workers busy.
2. Polls GET /tasks/{id} for each job until it is DONE, REJECTED or FAILED. With --approve it also approves
   the jobs that stop in WAITING_APPROVAL (none of the 5 default tasks is critical).
3. Asks Phoenix (GraphQL API) for what IT computed: the cost and latency of each trace, and the p50/p95 of
   trace latency for the test's time window. Those are the numbers to compare with the dashboard captures.
4. Saves everything in results/load_test.json and results/load_test.md.

Usage (API and Phoenix running):
    python scripts/load_test.py
    python scripts/load_test.py --approve --n 5 --api http://localhost:8000 --phoenix http://localhost:6006
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))   # PHOENIX_PROJECT_NAME and the endpoint, if the .env changes them

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FINAL = {"DONE", "REJECTED", "FAILED"}

TASKS = [
    "Compará las opiniones de los clientes sobre los SoundMax Pro y los AudioZen X: ¿cuál tiene mejor sentimiento "
    "y cuál es el precio promedio de cada uno?",
    "¿Cuáles son las principales quejas de los clientes sobre los AudioZen X y en qué tienda conviene comprarlos?",
    "Tengo $120.000 de presupuesto. ¿Me alcanza para el BassBoom 3? ¿Qué dicen las reseñas sobre su calidad?",
    "Para usar en el gimnasio, ¿qué me conviene más entre SoundMax Pro y AudioZen X? Tené en cuenta comodidad y precio.",
    "¿Qué opinan los clientes de los SoundMax Pro y en qué tienda están más baratos?",
]


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between the closest ranks (same method as numpy and Phoenix)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


async def submit(client: httpx.AsyncClient, task: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.post("/tasks", json={"task": task})
    response.raise_for_status()
    return {**response.json(), "task": task, "submit_ms": round((time.perf_counter() - started) * 1000, 1)}


async def follow(client: httpx.AsyncClient, job: dict[str, Any], approve: bool, timeout: float, poll: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = (await client.get(f"/tasks/{job['job_id']}")).json()
        if view["status"] in FINAL:
            return view
        if view["status"] == "WAITING_APPROVAL" and approve:
            await client.post(f"/tasks/{job['job_id']}/approve",
                              json={"approved": True, "reviewer": "load-test", "comment": "aprobado por el script"})
        await asyncio.sleep(poll)
    return {**view, "status": "TIMEOUT"}


async def phoenix_metrics(base_url: str, project: str, trace_ids: list[str], start: datetime, end: datetime) -> dict:
    """Cost and latency per trace, and the p50/p95 of the window, as Phoenix computes them."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        async def gql(query: str, variables: dict | None = None) -> dict:
            response = await client.post("/graphql", json={"query": query, "variables": variables or {}})
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                raise RuntimeError(body["errors"][0]["message"])
            return body["data"]

        projects = await gql("{ projects { edges { node { id name } } } }")
        project_id = next((e["node"]["id"] for e in projects["projects"]["edges"] if e["node"]["name"] == project), None)
        if project_id is None:
            raise RuntimeError(f"Phoenix no tiene el proyecto {project!r}")
        window = {"start": (start - timedelta(seconds=5)).isoformat(), "end": (end + timedelta(seconds=30)).isoformat()}
        quantiles = await gql(
            """query($id: ID!, $range: TimeRange) { node(id: $id) { ... on Project {
                 p50: latencyMsQuantile(probability: 0.5, timeRange: $range)
                 p95: latencyMsQuantile(probability: 0.95, timeRange: $range) } } }""",
            {"id": project_id, "range": window})
        traces = {}
        for trace_id in trace_ids:
            data = await gql(
                """query($id: ID!, $trace: ID!) { node(id: $id) { ... on Project { trace(traceId: $trace) {
                     latencyMs costSummary { prompt { tokens cost } completion { tokens cost } total { tokens cost } }
                   } } } }""", {"id": project_id, "trace": trace_id})
            traces[trace_id] = data["node"]["trace"]
        return {"project": project, "p50_ms": quantiles["node"]["p50"], "p95_ms": quantiles["node"]["p95"],
                "traces": traces}


def markdown(report: dict) -> str:
    lines = [f"# Prueba de carga · {report['n']} peticiones concurrentes · {report['started_at']}", "",
             f"POST /tasks: {report['submit_ms_max']} ms la más lenta (el API solo encola). "
             f"Estados finales: {report['statuses']}.", "",
             "| Job | Estado | Flujo | Espera en cola | Ejecución | Total | Tokens (in/out) | Costo (Phoenix) |",
             "|---|---|---|---|---|---|---|---|"]
    for job in report["jobs"]:
        ph = job.get("phoenix") or {}
        cost = ph.get("cost_usd")
        lines.append(f"| `{job['job_id'][:8]}` | {job['status']} | {' → '.join(job['flow']) or '—'} | "
                     f"{job['queue_seconds']} s | {job['run_seconds']} s | {job['total_seconds']} s | "
                     f"{ph.get('prompt_tokens', '—')}/{ph.get('completion_tokens', '—')} | "
                     f"{'US$ %.5f' % cost if cost is not None else '—'} |")
    c, ph = report["client"], report.get("phoenix")
    lines += ["", "## Latencia", "",
              "| Medición | p50 | p95 |", "|---|---|---|",
              f"| Total del job, medido por la API (cola + ejecución) | {c['total_p50']} s | {c['total_p95']} s |",
              f"| Ejecución del grafo (run_seconds) | {c['run_p50']} s | {c['run_p95']} s |"]
    if ph:
        p50 = ph["p50_ms"] / 1000 if ph["p50_ms"] is not None else None
        p95 = ph["p95_ms"] / 1000 if ph["p95_ms"] is not None else None
        lines.append(f"| Trazas en Phoenix (latencia de cada ejecución) | {p50 and round(p50, 2)} s | {p95 and round(p95, 2)} s |")
        lines += ["", f"Costo total de la corrida según Phoenix: US$ {report['phoenix_total_cost_usd']:.5f} · "
                      f"promedio por ejecución: US$ {report['phoenix_avg_cost_usd']:.5f}"]
    return "\n".join(lines) + "\n"


async def main() -> int:
    parser = argparse.ArgumentParser(description="5 concurrent requests + cost and p95 from Phoenix")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--phoenix", default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    parser.add_argument("--project", default=os.getenv("PHOENIX_PROJECT_NAME", "pe7-api-agente"))
    parser.add_argument("--n", type=int, default=5, help="number of concurrent requests")
    parser.add_argument("--approve", action="store_true", help="approve the jobs that wait for a human")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args()

    tasks = [TASKS[i % len(TASKS)] for i in range(args.n)]
    started = datetime.now(timezone.utc)
    async with httpx.AsyncClient(base_url=args.api, timeout=30) as client:
        submitted = await asyncio.gather(*(submit(client, t) for t in tasks))   # the 5 POST at the same time
        print(f"{len(submitted)} jobs encolados; POST más lento: {max(j['submit_ms'] for j in submitted)} ms")
        views = await asyncio.gather(*(follow(client, j, args.approve, args.timeout, args.poll) for j in submitted))
    finished = datetime.now(timezone.utc)

    jobs = []
    for sub, view in zip(submitted, views):
        jobs.append({"job_id": sub["job_id"], "task": sub["task"], "submit_ms": sub["submit_ms"], "status": view["status"],
                     "flow": (view.get("result") or {}).get("flow", []), "error": view.get("error"),
                     "queue_seconds": round(seconds_between(view["created_at"], view.get("started_at")) or 0, 2),
                     "run_seconds": round(view.get("run_seconds", 0), 2),
                     "total_seconds": round(seconds_between(view["created_at"], view.get("finished_at")) or 0, 2),
                     "trace_ids": view.get("trace_ids", [])})
    totals, runs = [j["total_seconds"] for j in jobs], [j["run_seconds"] for j in jobs]
    statuses: dict[str, int] = {}
    for j in jobs:
        statuses[j["status"]] = statuses.get(j["status"], 0) + 1
    report: dict[str, Any] = {
        "n": args.n, "started_at": started.isoformat(timespec="seconds"), "statuses": statuses, "jobs": jobs,
        "submit_ms_max": max(j["submit_ms"] for j in jobs),
        "client": {k: round(v, 2) if v is not None else None for k, v in {
            "total_p50": percentile(totals, 0.5), "total_p95": percentile(totals, 0.95),
            "run_p50": percentile(runs, 0.5), "run_p95": percentile(runs, 0.95)}.items()},
    }

    await asyncio.sleep(5)   # the exporter sends spans in batches: give it a moment before asking Phoenix
    try:
        ph = await phoenix_metrics(args.phoenix, args.project, [t for j in jobs for t in j["trace_ids"]], started, finished)
        report["phoenix"] = {"project": ph["project"], "p50_ms": ph["p50_ms"], "p95_ms": ph["p95_ms"]}
        total_cost = 0.0
        for j in jobs:
            cost = prompt = completion = 0.0
            for trace_id in j["trace_ids"]:
                summary = (ph["traces"].get(trace_id) or {}).get("costSummary") or {}
                cost += (summary.get("total") or {}).get("cost") or 0
                prompt += (summary.get("prompt") or {}).get("tokens") or 0
                completion += (summary.get("completion") or {}).get("tokens") or 0
            j["phoenix"] = {"cost_usd": round(cost, 6), "prompt_tokens": int(prompt), "completion_tokens": int(completion)}
            total_cost += cost
        report["phoenix_total_cost_usd"] = round(total_cost, 6)
        report["phoenix_avg_cost_usd"] = round(total_cost / len(jobs), 6)
    except Exception as e:   # Phoenix down or tracing disabled: the latency part of the report still works
        print(f"No pude leer las métricas de Phoenix ({e}); el informe queda solo con los tiempos de la API.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "load_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text = markdown(report)
    (RESULTS_DIR / "load_test.md").write_text(text, encoding="utf-8")
    print("\n" + text + "\nGuardado en results/load_test.md y results/load_test.json")
    return 0 if statuses.get("FAILED", 0) == 0 and "TIMEOUT" not in statuses else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
