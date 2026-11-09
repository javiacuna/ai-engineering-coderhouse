"""Runs the multi-agent orchestrator on a request and saves the delegation trace.

Usage:
    python main.py                          # default request (two products: research + analysis + synthesis)
    python main.py "tu pedido"
    python main.py --mermaid                # prints the graph as Mermaid (for the README)
"""

import argparse
import asyncio
import logging
import sys
import warnings

from dotenv import load_dotenv

from graph import BASE_DIR, build_graph, run_task, save_trace

DEFAULT_TASK = ("Compará las opiniones de los clientes sobre los auriculares SoundMax Pro y AudioZen X: "
                "¿cuál tiene mejor sentimiento en las reseñas, cuál es el precio promedio de cada uno "
                "y cuál me recomendarías?")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent research and analysis orchestrator")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid and exit")
    parser.add_argument("--trace", default="traces/last_run.json", help="where to save the trace")
    args = parser.parse_args()

    load_dotenv()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

    app = build_graph()
    if args.mermaid:
        print(app.get_graph().draw_mermaid())
        return 0

    state = await run_task(app, args.task)
    save_trace(state, BASE_DIR / args.trace)
    print(f"\nFlujo: {' → '.join(['supervisor → ' + d.next_agent for d in state['decisions']])}")
    print(f"Traza guardada en {args.trace}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
