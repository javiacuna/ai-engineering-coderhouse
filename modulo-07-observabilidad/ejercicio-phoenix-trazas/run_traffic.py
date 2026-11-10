"""Generates traffic for the Phoenix analysis: 5 complex requests + 2 provoked failures.

Each request runs the full Module 6 orchestrator (supervisor -> researcher -> analyst -> synthesizer),
inside a root span with the scenario as metadata and the prompt version as a tag.

Usage (with Phoenix running: python -m phoenix.server.main serve):
    python run_traffic.py                # new Phoenix project per session (no mixed metrics)
    python run_traffic.py --pause 60     # longer pause if Gemini answers 429 (free-tier rate limit)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openinference.instrumentation import using_attributes
from opentelemetry import trace

from tracing import DEFAULT_ENDPOINT, PROMPT_TEMPLATE_VERSION, retriever_span, setup_tracing

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SCENARIOS = [
    {"id": "q1-comparacion", "type": "compleja",
     "task": "Compará las opiniones de los clientes sobre los auriculares SoundMax Pro y AudioZen X: ¿cuál tiene "
             "mejor sentimiento en las reseñas, cuál es el precio promedio de cada uno y cuál me recomendarías?"},
    {"id": "q2-ranking", "type": "compleja",
     "task": "Armá un ranking de SoundMax Pro, AudioZen X y BassBoom 3 según el sentimiento de las reseñas y "
             "decime cuál tiene la mejor relación precio-valoración."},
    {"id": "q3-quejas", "type": "compleja",
     "task": "¿Cuáles son las principales quejas de los clientes sobre los AudioZen X y en qué tienda conviene comprarlos?"},
    {"id": "q4-presupuesto", "type": "compleja",
     "task": "Tengo $120.000 de presupuesto. ¿Me alcanza para el BassBoom 3? ¿Qué dicen las reseñas sobre su calidad?"},
    {"id": "q5-uso", "type": "compleja",
     "task": "Para usar en el gimnasio, ¿qué me conviene más entre SoundMax Pro y AudioZen X? Tené en cuenta la "
             "comodidad, la durabilidad y el precio."},
    # Provoked failures
    {"id": "f1-fuera-de-alcance", "type": "falla: fuera de alcance",
     "task": "¿Qué opinan los clientes del iPhone 17 y cuánto cuesta en las tiendas?"},
    {"id": "f2-fuente-caida", "type": "falla: herramienta desconectada", "break_reviews": True,
     "task": "¿Qué opinan los clientes de los SoundMax Pro y cuánto cuestan?"},
]


def phoenix_is_up(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(endpoint, timeout=3):
            return True
    except Exception:
        return False


async def run_scenario(app, scenario: dict, session_id: str) -> dict:
    from data import sources
    from graph import run_task

    original_fetch = sources.fetch_reviews
    if scenario.get("break_reviews"):
        async def unavailable(product: str):
            raise ConnectionError("fuente de reseñas no disponible (simulación de caída)")
        # Still wrapped in its RETRIEVER span: the trace shows the failure where it happens (the data source)
        sources.fetch_reviews = retriever_span(trace.get_tracer("data_sources"), "db.reviews.search", unavailable)

    tracer = trace.get_tracer("traffic")
    try:
        # Metadata and tags travel to EVERY span of this request (agents, LLMs, tools, retrievers)
        with using_attributes(session_id=session_id, tags=[scenario["type"]],
                              metadata={"scenario": scenario["id"]},
                              prompt_template_version=PROMPT_TEMPLATE_VERSION):
            with tracer.start_as_current_span(f"consulta:{scenario['id']}") as root:
                root.set_attribute("openinference.span.kind", "CHAIN")
                root.set_attribute("input.value", scenario["task"])
                trace_id = format(root.get_span_context().trace_id, "032x")
                state = await run_task(app, scenario["task"], verbose=False)
                root.set_attribute("output.value", state.get("final_answer") or "")
    finally:
        sources.fetch_reviews = original_fetch

    return {
        "scenario": scenario["id"], "type": scenario["type"], "task": scenario["task"], "trace_id": trace_id,
        "elapsed_seconds": state.get("elapsed_seconds"), "flow": [d.next_agent for d in state["decisions"]],
        "errors": [c.error for c in state["contributions"] if c.error],
        "final_answer": state.get("final_answer"),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate traced traffic for Phoenix")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--project", default=None, help="Phoenix project (default: a new one per session)")
    parser.add_argument("--pause", type=float, default=30, help="seconds between requests (Gemini free-tier rate limits)")
    parser.add_argument("--only", nargs="*", help="run only these scenario ids")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

    if not phoenix_is_up(args.endpoint):
        print(f"Phoenix no responde en {args.endpoint}. Levantalo en otra terminal con:\n"
              "    python -m phoenix.server.main serve")
        return 1

    # A NEW project per session, so the p95 of this session is not mixed with older runs
    project = args.project or f"orquestador-m7-{datetime.now():%Y%m%d-%H%M}"
    tracer_provider = setup_tracing(project, args.endpoint)
    from graph import build_graph   # imported after instrumenting, so every component is traced

    app = build_graph()
    scenarios = [s for s in SCENARIOS if not args.only or s["id"] in args.only]
    results = []
    for n, scenario in enumerate(scenarios, start=1):
        print(f"[{n}/{len(scenarios)}] {scenario['id']} ({scenario['type']})")
        started = time.perf_counter()
        result = await run_scenario(app, scenario, session_id=project)
        results.append(result)
        print(f"    {time.perf_counter() - started:.1f} s | flujo: {' → '.join(result['flow'])} | "
              f"errores: {len(result['errors'])}")
        if n < len(scenarios):
            await asyncio.sleep(args.pause)

    tracer_provider.force_flush()   # make sure every span reached Phoenix before exiting
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "traffic_run.json").write_text(
        json.dumps({"project": project, "prompt_template_version": PROMPT_TEMPLATE_VERSION, "runs": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nProyecto de Phoenix: {project}  →  {args.endpoint}")
    print(f"Ahora: python analyze_traces.py --project {project}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
