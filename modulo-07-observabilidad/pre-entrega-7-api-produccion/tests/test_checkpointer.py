"""AsyncRedisCheckpointer against an in-memory Redis (fakeredis): state, interrupts, subgraphs and restarts."""

import operator
from typing import Annotated, TypedDict

import fakeredis
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.checkpointer import AsyncRedisCheckpointer


class State(TypedDict):
    steps: Annotated[list[str], operator.add]
    approved: bool


def build(checkpointer):
    inner = StateGraph(State)
    inner.add_node("inner_work", lambda s: {"steps": ["inner"]})
    inner.add_edge(START, "inner_work")
    inner.add_edge("inner_work", END)
    sub = inner.compile()   # no checkpointer: it inherits the parent's (namespaced checkpoints)

    def ask(state: State) -> dict:
        decision = interrupt({"question": "¿Aprobás?"})
        return {"approved": decision == "si", "steps": ["ask"]}

    outer = StateGraph(State)
    outer.add_node("first", lambda s: {"steps": ["first"]})
    outer.add_node("sub", sub)
    outer.add_node("ask", ask)
    outer.add_node("last", lambda s: {"steps": ["last"]})
    outer.add_edge(START, "first")
    outer.add_edge("first", "sub")
    outer.add_edge("sub", "ask")
    outer.add_edge("ask", "last")
    outer.add_edge("last", END)
    return outer.compile(checkpointer=checkpointer)


@pytest.fixture
async def redis():
    client = fakeredis.FakeAsyncRedis()
    yield client
    await client.aclose()


async def test_interrupt_survives_a_restart(redis):
    config = {"configurable": {"thread_id": "job-1"}}
    graph = build(AsyncRedisCheckpointer(redis))
    await graph.ainvoke({"steps": [], "approved": False}, config)
    paused = await graph.aget_state(config)
    assert paused.next == ("ask",) and paused.interrupts[0].value == {"question": "¿Aprobás?"}

    # "Restart": a brand-new checkpointer and graph, same Redis → the thread resumes where it stopped
    restarted = build(AsyncRedisCheckpointer(redis))
    result = await restarted.ainvoke(Command(resume="si"), config)
    # (the subgraph returns its whole state, so "first" is added twice: same as with InMemorySaver)
    assert result["steps"] == ["first", "first", "inner", "ask", "last"] and result["approved"] is True
    assert not (await restarted.aget_state(config)).next


async def test_history_namespaces_and_delete(redis):
    saver = AsyncRedisCheckpointer(redis)
    graph = build(saver)
    config = {"configurable": {"thread_id": "job-2"}}
    await graph.ainvoke({"steps": [], "approved": False}, config)
    history = [c async for c in saver.alist(config)]
    assert len({c.config["configurable"]["checkpoint_ns"] for c in history}) == 2   # main graph + subgraph
    main = {"configurable": {"thread_id": "job-2", "checkpoint_ns": ""}}
    ids = [c.config["configurable"]["checkpoint_id"] async for c in saver.alist(main)]
    assert ids == sorted(ids, reverse=True) and len(ids) >= 4          # newest first
    assert [c.checkpoint["id"] async for c in saver.alist(main, limit=2)] == ids[:2]
    before = {"configurable": {"thread_id": "job-2", "checkpoint_id": ids[1]}}
    assert [c.checkpoint["id"] async for c in saver.alist(main, before=before)] == ids[2:]
    assert await redis.keys("lg:*")
    await saver.adelete_thread("job-2")
    assert await redis.keys("lg:*") == []


async def test_threads_are_isolated_and_ttl_is_applied(redis):
    saver = AsyncRedisCheckpointer(redis, ttl_seconds=3600)
    graph = build(saver)
    a, b = {"configurable": {"thread_id": "a"}}, {"configurable": {"thread_id": "b"}}
    await graph.ainvoke({"steps": [], "approved": False}, a)
    await graph.ainvoke({"steps": [], "approved": False}, b)
    await graph.ainvoke(Command(resume="no"), a)
    assert (await graph.aget_state(a)).values["approved"] is False
    assert (await graph.aget_state(b)).next == ("ask",)                # b still waiting
    ttls = [await redis.ttl(k) for k in await redis.keys("lg:*")]
    assert ttls and all(0 < t <= 3600 for t in ttls)
