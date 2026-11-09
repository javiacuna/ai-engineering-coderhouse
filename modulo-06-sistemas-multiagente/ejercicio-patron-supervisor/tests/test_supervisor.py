"""Tests of the supervisor graph with a scripted router (no API key needed)."""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supervisor import build_graph, run  # noqa: E402


class ScriptedRouter:
    """Stands in for llm.with_structured_output(Route): returns the scripted decisions in order."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.received = []

    async def ainvoke(self, messages):
        self.received.append(messages)
        return self.decisions.pop(0)


class RuleRouter:
    """Follows the prompt's rules by reading the history, like a well-behaved LLM would."""

    async def ainvoke(self, messages):
        names = {getattr(m, "name", None) for m in messages}
        if "Escritor" in names:
            return {"next_agent": "FINALIZAR", "reason": "informe listo"}
        if "Analista" in names:
            return {"next_agent": "Escritor", "reason": "hay datos"}
        return {"next_agent": "Analista", "reason": "faltan datos"}


def route(name):
    return {"next_agent": name, "reason": "test"}


def test_expected_sequence():
    steps = asyncio.run(run(build_graph(RuleRouter()), "Necesito un reporte sobre ventas."))
    assert steps == ["supervisor", "Analista", "supervisor", "Escritor", "supervisor", "fin"]


def test_supervisor_sees_the_whole_history_and_its_prompt():
    router = ScriptedRouter(route("Analista"), route("Escritor"), route("FINALIZAR"))
    asyncio.run(run(build_graph(router), "Necesito un reporte sobre ventas."))
    assert all(isinstance(call[0], SystemMessage) for call in router.received)
    last_call = router.received[-1]
    assert [getattr(m, "name", None) for m in last_call[1:]] == [None, "Analista", "Escritor"]


def test_final_state_has_both_contributions():
    graph = build_graph(RuleRouter())
    state = asyncio.run(graph.ainvoke({"messages": [("user", "Necesito un reporte sobre ventas.")]}))
    assert [m.name for m in state["messages"][1:]] == ["Analista", "Escritor"]
    assert state["next"] == "FINALIZAR"


def test_unrelated_request_finishes_without_specialists():
    steps = asyncio.run(run(build_graph(ScriptedRouter(route("FINALIZAR"))), "Hola"))
    assert steps == ["supervisor", "fin"]


def test_invalid_decision_finishes_instead_of_crashing():
    steps = asyncio.run(run(build_graph(ScriptedRouter({"next_agent": "Contador"})), "Necesito un reporte"))
    assert steps == ["supervisor", "fin"]


def test_loop_is_stopped_by_the_recursion_limit():
    looping = ScriptedRouter(*[route("Analista")] * 50)          # never says FINALIZAR
    steps = asyncio.run(run(build_graph(looping), "Necesito un reporte"))
    assert steps[-1] == "LIMITE"
    assert len(steps) <= 11
