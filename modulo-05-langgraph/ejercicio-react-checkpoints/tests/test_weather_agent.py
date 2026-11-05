"""Tests of the ReAct cycle and the checkpoint memory, with a fake model (no API key needed)."""

import asyncio
import sys
from itertools import count
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weather_agent import build_graph, get_weather, tools_used  # noqa: E402

CITIES = ("Madrid", "Londres", "Córdoba")


class FakeWeatherModel(GenericFakeChatModel):
    """Decides like a ReAct LLM, but with rules:
    - last message is a tool result -> final answer using it;
    - the user names a city and the conversation (whole history) is about the weather -> call get_weather;
    - otherwise -> it does not know what the user means."""

    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls += 1
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(content=f"El clima es: {last.content}")
        city = next((c for c in CITIES if c.lower() in last.content.lower()), None)
        about_weather = any("tiempo" in m.content or "clima" in m.content
                            for m in messages if isinstance(m, HumanMessage))
        if city and about_weather:
            return AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"location": city},
                                                      "id": f"call_{next(_ids)}"}])
        return AIMessage(content="No entiendo a qué te referís, ¿qué querés saber de esa ciudad?")


_ids = count(1)


def ask(app, question, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    return asyncio.run(app.ainvoke({"messages": [HumanMessage(content=question)]}, config))


@pytest.fixture
def app():
    return build_graph(FakeWeatherModel(messages=iter([])))


def test_tool_contract():
    assert get_weather.name == "get_weather"
    assert "clima" in get_weather.description
    assert get_weather.invoke({"location": "madrid "}) == "Soleado, 22°C"
    assert "No hay datos" in get_weather.invoke({"location": "Tokio"})


def test_react_cycle_agent_tools_agent_end(app):
    result = ask(app, "¿Qué tiempo hace en Madrid?", "t1")
    kinds = [type(m).__name__ for m in result["messages"]]
    # Reason (tool call) -> Act (ToolMessage) -> Observe + final answer -> END
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert result["messages"][2].tool_call_id == result["messages"][1].tool_calls[0]["id"]   # mapped back
    assert result["messages"][-1].content == "El clima es: Soleado, 22°C"
    assert not result["messages"][-1].tool_calls                                             # ended by the LLM


def test_same_thread_remembers_the_context(app):
    ask(app, "¿Qué tiempo hace en Madrid?", "t1")
    result = ask(app, "¿Y en Londres?", "t1")
    assert len(result["messages"]) == 8                      # 4 from turn 1 + 4 from turn 2: concatenated
    assert result["messages"][-1].content == "El clima es: Nublado con lloviznas, 14°C"
    assert tools_used(result["messages"][4:]) == [
        "get_weather({'location': 'Londres'}) -> Nublado con lloviznas, 14°C"]


def test_new_thread_has_no_memory(app):
    ask(app, "¿Qué tiempo hace en Madrid?", "t1")
    result = ask(app, "¿Y en Londres?", "t2")
    assert len(result["messages"]) == 2                      # only this turn: nothing from t1
    assert "No entiendo" in result["messages"][-1].content
    assert tools_used(result["messages"]) == []


def test_state_survives_in_the_checkpointer(app):
    ask(app, "¿Qué tiempo hace en Madrid?", "t1")
    snapshot = asyncio.run(app.aget_state({"configurable": {"thread_id": "t1"}}))
    assert len(snapshot.values["messages"]) == 4
    assert snapshot.next == ()                               # finished: nothing pending
    history = asyncio.run(_history(app, "t1"))
    assert len(history) >= 4                                 # one checkpoint per step: time travel is possible


async def _history(app, thread_id):
    return [s async for s in app.aget_state_history({"configurable": {"thread_id": thread_id}})]
