"""Tests of the tools, the ReAct cycle, error recovery, SQLite persistence and the limits.
A scripted model replaces the LLM, so they need no API key."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
from agent import build_graph, run_turn, sqlite_checkpointer, trim_history  # noqa: E402
from tools import get_customer_orders, get_order_detail, search_customers  # noqa: E402


class ScriptedModel:
    """Returns the scripted AIMessages in order and records what it received on each call."""

    def __init__(self, *responses: AIMessage):
        self.responses = list(responses)
        self.received: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.received.append(messages)
        return self.responses.pop(0)


def call(tool_name: str, call_id: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])


def answer(text: str) -> AIMessage:
    return AIMessage(content=text)


def run(coro):
    return asyncio.run(coro)


async def turns(db_path: Path, model: ScriptedModel, *questions: tuple[str, str], recursion_limit: int = 10):
    async with sqlite_checkpointer(db_path) as saver:
        app = build_graph(model, saver)
        return [await run_turn(app, q, thread, recursion_limit) for thread, q in questions]


# ---------------------------------------------------------------- tools
def test_search_customers_handles_accents_and_ambiguity():
    one = run(search_customers.ainvoke({"name": "lucia fernandez"}))
    assert one["coincidencias"] == 1 and one["clientes"][0]["id"] == 102
    many = run(search_customers.ainvoke({"name": "García"}))
    assert many["coincidencias"] == 2 and "aviso" in many
    none = run(search_customers.ainvoke({"name": "Zapata"}))
    assert "error" in none and "sugerencia" in none


def test_customer_orders_matches_the_assignment_example():
    result = run(get_customer_orders.ainvoke({"customer_id": 102}))
    assert result["pedidos"] == 3 and result["total"] == 14500
    assert result["detalle"][-1]["pedido_id"] == 5012                      # last = most recent


def test_cancelled_orders_do_not_count_in_the_total():
    result = run(get_customer_orders.ainvoke({"customer_id": 101}))
    assert result["pedidos"] == 2 and result["pedidos_cancelados"] == 1 and result["total"] == 98000


def test_unknown_ids_return_an_error_with_a_suggestion():
    assert "search_customers" in run(get_customer_orders.ainvoke({"customer_id": 999}))["sugerencia"]
    assert "error" in run(get_order_detail.ainvoke({"order_id": 1}))


def test_tool_descriptions_tell_the_llm_when_to_use_them():
    assert "ID" in search_customers.description and "varios" in search_customers.description.lower()
    assert "search_customers" in get_customer_orders.description
    assert "get_customer_orders" in get_order_detail.description


# ---------------------------------------------------------------- ReAct cycle
def test_multi_step_reasoning_calls_tools_twice(tmp_path):
    model = ScriptedModel(
        call("search_customers", "c1", name="Lucía Fernández"),
        call("get_customer_orders", "c2", customer_id=102),
        answer("Lucía Fernández tuvo 3 pedidos por $14.500."),
    )
    [trace] = run(turns(tmp_path / "db.sqlite", model, ("t1", "¿Cuánto gastó Lucía Fernández?")))
    assert [s.type for s in trace.steps] == ["accion", "observacion", "accion", "observacion", "respuesta"]
    assert trace.tool_calls == 2
    assert trace.steps[3].result["total"] == 14500                          # the real tool result
    assert trace.answer.endswith("$14.500.")
    assert isinstance(model.received[0][0], SystemMessage)                  # system prompt on every call
    assert isinstance(model.received[2][-1], ToolMessage)                   # observation went back to the agent


def test_tool_error_leads_to_a_second_attempt(tmp_path):
    model = ScriptedModel(
        call("get_customer_orders", "c1", customer_id=999),
        call("search_customers", "c2", name="Sofía García"),
        call("get_customer_orders", "c3", customer_id=103),
        answer("Sofía García tiene 1 pedido por $39.500."),
    )
    [trace] = run(turns(tmp_path / "db.sqlite", model, ("t1", "Pedidos del cliente 999, es Sofía García")))
    observations = [s.result for s in trace.steps if s.type == "observacion"]
    assert "error" in observations[0]                                       # first attempt failed
    assert observations[-1]["total"] == 39500                               # retry succeeded
    assert trace.tool_calls == 3


def test_invalid_arguments_come_back_as_an_observation(tmp_path):
    model = ScriptedModel(call("get_order_detail", "c1", order_id="abc"), answer("No pude leer ese pedido."))
    [trace] = run(turns(tmp_path / "db.sqlite", model, ("t1", "Detalle del pedido abc")))
    assert trace.steps[1].type == "observacion" and "error" in str(trace.steps[1].result).lower()
    assert trace.error is None                                              # the graph did not crash


# ---------------------------------------------------------------- persistence
def test_same_thread_remembers_and_other_thread_does_not(tmp_path):
    model = ScriptedModel(
        call("get_customer_orders", "c1", customer_id=102), answer("3 pedidos, $14.500."),
        call("get_order_detail", "c2", order_id=5012), answer("El último es el 5012."),
        answer("¿El último de qué?"),
    )
    first, second, other = run(turns(tmp_path / "db.sqlite", model,
                                     ("t1", "¿Cuántos pedidos tuvo el cliente 102?"),
                                     ("t1", "¿Y el último?"),
                                     ("t2", "¿Y el último?")))
    assert first.messages_in_thread == 4 and second.messages_in_thread == 8
    sent_on_second_turn = model.received[2]
    assert any("cliente 102" in str(m.content) for m in sent_on_second_turn)   # turn 1 was in the context
    assert other.messages_in_thread == 2
    assert not any("cliente 102" in str(m.content) for m in model.received[4])


def test_memory_survives_a_restart(tmp_path):
    db_path = tmp_path / "db.sqlite"
    run(turns(db_path, ScriptedModel(answer("Hola, ¿en qué te ayudo?")), ("t1", "Hola, soy Javier")))
    # A NEW graph and a NEW checkpointer on the same file: like running the script again
    model = ScriptedModel(answer("Te llamás Javier."))
    [trace] = run(turns(db_path, model, ("t1", "¿Cómo me llamo?")))
    assert trace.messages_in_thread == 4
    assert any("soy Javier" in str(m.content) for m in model.received[0])


# ---------------------------------------------------------------- limits
def test_recursion_limit_stops_an_endless_loop(tmp_path):
    model = ScriptedModel(*[call("get_customer_orders", f"c{i}", customer_id=102) for i in range(20)])
    [trace] = run(turns(tmp_path / "db.sqlite", model, ("t1", "loop"), recursion_limit=5))
    assert trace.error is not None and "límite" in trace.error
    assert trace.tool_calls <= 3                                            # stopped early, not 20 calls


def test_trim_history_keeps_recent_messages_and_valid_pairs(monkeypatch):
    monkeypatch.setattr(agent, "MAX_CONTEXT_TOKENS", 120)
    history: list[BaseMessage] = []
    for i in range(30):
        history += [HumanMessage(content=f"pregunta {i} " + "texto " * 10),
                    AIMessage(content="", tool_calls=[{"name": "get_order_detail", "args": {"order_id": 5012}, "id": f"x{i}"}]),
                    ToolMessage(content=json.dumps({"ok": i}), tool_call_id=f"x{i}"),
                    AIMessage(content=f"respuesta {i}")]
    history.append(HumanMessage(content="nueva pregunta"))   # the model is always called after a user message
    trimmed = trim_history(history)
    assert len(trimmed) < len(history)
    assert isinstance(trimmed[0], HumanMessage)                             # never starts with an orphan tool result
    assert trimmed[-1] is history[-1]                                       # the most recent message is kept


def test_new_database_can_delete_threads_before_any_write(tmp_path):
    async def scenario():
        async with sqlite_checkpointer(tmp_path / "new.sqlite") as saver:
            await saver.adelete_thread("never-used")    # fails with "no such table" without setup()
    run(scenario())
