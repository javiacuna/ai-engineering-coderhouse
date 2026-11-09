"""Tests of the shared state: validation, write isolation, reviewer decisions and the async flow."""

import asyncio
import sys
import time
from pathlib import Path

import pytest
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared_state  # noqa: E402
from shared_state import (  # noqa: E402
    AgentState, ResearchArtifact, ReviewArtifact, StateOwnershipError, app, owns, researcher_node, reviewer_node,
    run_example,
)


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    monkeypatch.setattr(shared_state, "RESEARCH_DELAY_SECONDS", 0)


def research(score: float) -> ResearchArtifact:
    return ResearchArtifact(topic="t", findings=["un hallazgo"], confidence_score=score)


# ---------------------------------------------------------------- contracts
@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_confidence_score_must_be_between_0_and_1(score):
    with pytest.raises(ValidationError):
        research(score)


def test_findings_cannot_be_empty():
    with pytest.raises(ValidationError):
        ResearchArtifact(topic="t", findings=[], confidence_score=0.9)
    with pytest.raises(ValidationError):
        ResearchArtifact(topic="t", findings=["   "], confidence_score=0.9)


# ---------------------------------------------------------------- end to end
def test_high_confidence_is_approved(capsys):
    final = asyncio.run(run_example(0.92))
    assert final["researcher_output"].confidence_score == 0.92
    assert final["reviewer_output"].is_approved and final["errors"] == []
    out = capsys.readouterr().out
    assert out.index("Nodo 'investigador' finalizado.") < out.index("--- Revisando resultados ---")


def test_low_confidence_is_rejected_with_feedback():
    final = asyncio.run(run_example(0.6))
    assert not final["reviewer_output"].is_approved
    assert "0.60" in final["reviewer_output"].feedback


def test_threshold_is_strict():
    final = asyncio.run(run_example(0.8))                     # "> 0.8": exactly 0.8 is not enough
    assert not final["reviewer_output"].is_approved


def test_invalid_score_is_handled_without_crashing():
    final = asyncio.run(run_example(1.5))
    assert final["researcher_output"] is None                 # the invalid artifact never entered the state
    assert len(final["errors"]) == 1 and "confidence_score" in final["errors"][0]
    assert not final["reviewer_output"].is_approved
    assert "No hay resultados" in final["reviewer_output"].feedback


# ---------------------------------------------------------------- write isolation
def test_each_node_returns_only_its_own_key():
    state = {"topic": "t", "researcher_output": None, "reviewer_output": None, "errors": []}
    config = {"configurable": {"simulated_confidence": 0.9}}
    assert set(asyncio.run(researcher_node(state, config))) == {"researcher_output"}
    state["researcher_output"] = research(0.9)
    assert set(asyncio.run(reviewer_node(state, config))) == {"reviewer_output"}
    assert researcher_node.owned_keys == ("researcher_output",) and reviewer_node.owned_keys == ("reviewer_output",)


def test_writing_another_agents_key_is_rejected():
    @owns("reviewer_output")
    async def rogue_reviewer(state, config):
        return {"reviewer_output": ReviewArtifact(is_approved=True, feedback="ok"),
                "researcher_output": research(1.0)}             # overwriting the researcher's work

    workflow = StateGraph(AgentState)
    workflow.add_node("revisor", rogue_reviewer)
    workflow.set_entry_point("revisor")
    workflow.add_edge("revisor", END)
    with pytest.raises(StateOwnershipError, match="researcher_output"):
        asyncio.run(workflow.compile().ainvoke({"topic": "t", "researcher_output": None,
                                                "reviewer_output": None, "errors": []}))


def test_errors_are_appended_not_overwritten():
    state = {"topic": "t", "researcher_output": None, "reviewer_output": None, "errors": ["error previo"]}
    final = asyncio.run(app.ainvoke(state, {"configurable": {"simulated_confidence": 1.5}}))
    assert final["errors"][0] == "error previo" and len(final["errors"]) == 2


# ---------------------------------------------------------------- async
def test_independent_runs_overlap_in_time(monkeypatch):
    monkeypatch.setattr(shared_state, "RESEARCH_DELAY_SECONDS", 0.3)

    async def three_runs():
        state = {"topic": "t", "researcher_output": None, "reviewer_output": None, "errors": []}
        await asyncio.gather(*(app.ainvoke(dict(state), {"configurable": {"simulated_confidence": 0.9}})
                               for _ in range(3)))

    start = time.perf_counter()
    asyncio.run(three_runs())
    assert time.perf_counter() - start < 0.8                  # ~0.3 s total, not 3 x 0.3 s
