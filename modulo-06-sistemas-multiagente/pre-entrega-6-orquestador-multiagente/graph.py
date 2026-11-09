"""Orchestrator graph: hierarchical topology with a supervisor and three specialists.

    START -> supervisor --(researcher)--> researcher --+
                 ^      --(analyst)----> analyst ------+--> back to supervisor
                 |      --(synthesizer)-> synthesizer -+
                 +--------------------------------------+
             supervisor --(FINISH)--> END
"""

import functools
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import make_analyst_node
from agents.llm import chat_models
from agents.research_agent import make_research_node
from agents.synthesis_agent import make_synthesis_node
from state import OrchestratorState
from supervisor import MAX_STEPS, make_supervisor_node, route_from_supervisor

logger = logging.getLogger("orchestrator")

BASE_DIR = Path(__file__).resolve().parent
GRAPH_RECURSION_LIMIT = 2 * MAX_STEPS + 4     # each step = supervisor + specialist; a last safety net

# Which state keys each node may write ("messages" is the shared conversation, appended by everyone)
OWNERSHIP = {
    "supervisor": {"next_agent", "instruction", "step", "decisions"},
    "researcher": {"research", "contributions"},
    "analyst": {"analysis", "contributions"},
    "synthesizer": {"final_answer", "contributions"},
}


RUBRIC_LABELS = {"research_complete": "investigación", "analysis_complete": "análisis",
                 "analysis_consistent": "consistencia", "final_answer_ready": "respuesta"}


class StateOwnershipError(Exception):
    """A node wrote a part of the state owned by another agent."""


def guarded(name: str, node: Callable[[OrchestratorState], Awaitable[dict[str, Any]]]):
    """Wraps a node so it can only write its own keys: agents can't overwrite each other's work."""
    allowed = OWNERSHIP[name] | {"messages"}

    @functools.wraps(node)
    async def wrapper(state: OrchestratorState) -> dict[str, Any]:
        update = await node(state)
        foreign = set(update) - allowed
        if foreign:
            raise StateOwnershipError(f"'{name}' wrote {sorted(foreign)}; it owns {sorted(OWNERSHIP[name])}")
        return update

    return wrapper


def build_graph(supervisor_models: list[Any] | None = None, worker_models: list[Any] | None = None) -> CompiledStateGraph:
    supervisor_models = supervisor_models or chat_models("supervisor")
    worker_models = worker_models or chat_models("worker")

    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", guarded("supervisor", make_supervisor_node(supervisor_models)))
    builder.add_node("researcher", guarded("researcher", make_research_node(worker_models)))
    builder.add_node("analyst", guarded("analyst", make_analyst_node(worker_models)))
    builder.add_node("synthesizer", guarded("synthesizer", make_synthesis_node(worker_models)))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_from_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "synthesizer": "synthesizer", "FINISH": END},
    )
    for specialist in ("researcher", "analyst", "synthesizer"):
        builder.add_edge(specialist, "supervisor")   # every result goes back to the supervisor for validation
    return builder.compile()


def initial_state(task: str) -> dict[str, Any]:
    return {"messages": [HumanMessage(content=task)], "task": task, "next_agent": "researcher", "instruction": "",
            "step": 0, "research": None, "analysis": None, "final_answer": None, "contributions": [], "decisions": []}


async def run_task(app: CompiledStateGraph, task: str, verbose: bool = True) -> dict[str, Any]:
    """Runs a task streaming the delegation flow. Returns the final state plus the elapsed time."""
    state: dict[str, Any] = initial_state(task)
    started = time.perf_counter()
    if verbose:
        print(f"👤 Usuario: {task}\n")
    try:
        async for update in app.astream(state, {"recursion_limit": GRAPH_RECURSION_LIMIT}, stream_mode="updates"):
            for node, output in update.items():
                for key, value in output.items():
                    state[key] = state[key] + value if key in ("messages", "contributions", "decisions") else value
                if not verbose:
                    continue
                if node == "supervisor":
                    d = output["decisions"][-1]
                    rubric = " ".join(f"{RUBRIC_LABELS.get(k, k)}{'✓' if v else '✗'}" for k, v in d.rubric.items())
                    flag = " ⚠ regla del sistema" if d.overridden else ""
                    print(f"🧭 supervisor (paso {d.step}) [{rubric}] → {d.next_agent}{flag}\n   motivo: {d.reason}")
                    if d.instruction and d.next_agent != "FINISH":
                        print(f"   instrucción: {d.instruction[:220]}")
                else:
                    c = output["contributions"][-1]
                    icon = {"researcher": "🔎", "analyst": "📊", "synthesizer": "✍️"}[node]
                    print(f"{icon} {node} (ronda {c.round})" + (f" | herramientas: {', '.join(c.tools_used)}" if c.tools_used else ""))
                    if c.error:
                        print(f"   ERROR: {c.error}")
                    elif node != "synthesizer":
                        print(f"   {c.summary[:300]}")
                print()
    except GraphRecursionError:
        logger.error("graph recursion limit reached")
        state["final_answer"] = state.get("final_answer") or "Se alcanzó el límite de pasos sin una respuesta final."
    state["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    if verbose:
        print("=" * 70 + f"\n✅ Respuesta final ({state['elapsed_seconds']} s):\n\n{state['final_answer']}")
    return state


def save_trace(state: dict[str, Any], path: Path) -> None:
    """Saves who contributed what, the supervisor's decisions and the final artifacts as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda v: v.model_dump() if hasattr(v, "model_dump") else v  # noqa: E731
    trace = {
        "task": state["task"],
        "elapsed_seconds": state.get("elapsed_seconds"),
        "flow": [d.next_agent for d in state["decisions"]],
        "decisions": [dump(d) for d in state["decisions"]],
        "contributions": [dump(c) for c in state["contributions"]],
        "research": dump(state.get("research")),
        "analysis": dump(state.get("analysis")),
        "final_answer": state.get("final_answer"),
    }
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
