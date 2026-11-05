"""Demo of the ReAct agent: multi-step reasoning, error recovery, clarification and memory per thread.

Writes the ReAct trace to traces/react_trace.json and traces/react_trace.log.

Usage:
    python main.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from agent import BASE_DIR, DB_PATH, RECURSION_LIMIT, TurnTrace, build_graph, build_model, run_turn, sqlite_checkpointer

TRACE_DIR = BASE_DIR / "traces"

# (thread_id, question). Turns of the same thread share memory through the SQLite checkpointer.
SCENARIOS: list[tuple[str, str]] = [
    # Session 1: the example of the assignment + "¿y el último?" + a comparison that needs several tool calls
    ("sesion-1", "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"),
    ("sesion-1", "¿Y el último?"),
    ("sesion-1", "¿Quién gastó más en total: ese cliente o Diego Pereyra?"),
    # Session 2: ambiguous name -> the agent asks; the answer is understood thanks to the memory
    ("sesion-2", "¿Cuánto gastó García en total?"),
    ("sesion-2", "El de Rosario"),
    # Session 3: the tool returns an error -> the agent corrects itself with another tool and retries
    ("sesion-3", "Mostrame los pedidos del cliente 999, es Sofía García"),
]


def setup_logging() -> None:
    TRACE_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(TRACE_DIR / "react_trace.log", mode="w", encoding="utf-8")
    console = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
    agent_logger = logging.getLogger("agent")
    agent_logger.setLevel(logging.INFO)
    agent_logger.handlers = [file_handler, console]
    agent_logger.propagate = False


async def main() -> int:
    load_dotenv()
    setup_logging()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash") if os.getenv("LLM_PROVIDER", "gemini") == "gemini" \
        else os.getenv("LLM_PROVIDER")
    traces: list[TurnTrace] = []

    async with sqlite_checkpointer() as checkpointer:
        # The demo starts its sessions from zero so the trace is reproducible
        for thread_id in dict.fromkeys(t for t, _ in SCENARIOS):
            await checkpointer.adelete_thread(thread_id)
        app = build_graph(build_model(), checkpointer)

        current = None
        for thread_id, question in SCENARIOS:
            if thread_id != current:
                print(f"\n{'=' * 20} {thread_id} {'=' * 20}")
                current = thread_id
            traces.append(await run_turn(app, question, thread_id))

    multi_step = [t for t in traces if t.tool_calls >= 2]
    print(f"\nTurnos: {len(traces)} | llamadas a herramientas: {sum(t.tool_calls for t in traces)} | "
          f"turnos multi-paso (≥2 herramientas): {len(multi_step)} | errores: {sum(t.error is not None for t in traces)}")

    (TRACE_DIR / "react_trace.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "recursion_limit": RECURSION_LIMIT,
        "checkpointer": f"AsyncSqliteSaver ({DB_PATH.relative_to(BASE_DIR)})",
        "turns": [t.model_dump() for t in traces],
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Traza guardada en traces/react_trace.json y traces/react_trace.log")
    print("Memoria persistida en SQLite: probá  python ask.py --thread sesion-1 \"¿Cuál era el total del primer cliente?\"")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
