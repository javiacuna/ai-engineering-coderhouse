"""Tests of the orchestrator with scripted models (no API key needed)."""

import asyncio
import itertools
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.sources as sources  # noqa: E402
from agents.analyst_agent import make_analyst_tools, review_polarity  # noqa: E402
from agents.research_agent import merge_findings, search_reviews  # noqa: E402
from graph import StateOwnershipError, build_graph, guarded, initial_state, run_task  # noqa: E402
from state import ProductFindings, ResearchFindings, Review  # noqa: E402
from supervisor import MAX_ROUNDS, MAX_STEPS, RouteDecision, evaluate_rubric  # noqa: E402

TASK = "Compará SoundMax Pro y AudioZen X: sentimiento, precio promedio y recomendación."


@pytest.fixture(autouse=True)
def no_latency(monkeypatch):
    monkeypatch.setattr(sources, "LATENCY_SECONDS", 0)


# ---------------------------------------------------------------- scripted models
class ScriptedWorker(GenericFakeChatModel):
    """Worker model: returns the scripted AIMessages in order (tool calls or final texts)."""

    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedSupervisor:
    """Stands in for model.with_structured_output(RouteDecision)."""

    def __init__(self, *choices, fail=False):
        self.choices = itertools.chain(choices, itertools.repeat(choices[-1] if choices else None))
        self.fail = fail
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.fail:
            raise ConnectionError("503 overloaded")
        name, instruction = next(self.choices)
        return RouteDecision(next_agent=name, instruction=instruction, reason="test")


def calls(*pairs):
    return AIMessage(content="", tool_calls=[{"name": n, "args": {"product": p}, "id": f"{n}-{p}-{i}"}
                                             for i, (n, p) in enumerate(pairs)])


def text(t):
    return AIMessage(content=t)


RESEARCH_BOTH = [calls(("search_reviews", "SoundMax Pro"), ("search_prices", "SoundMax Pro"),
                       ("search_reviews", "AudioZen X"), ("search_prices", "AudioZen X")),
                 text("Encontré 6 reseñas y 3 precios de cada producto.")]
ANALYSIS_BOTH = [calls(("analyze_sentiment", "SoundMax Pro"), ("compute_statistics", "SoundMax Pro"),
                       ("analyze_sentiment", "AudioZen X"), ("compute_statistics", "AudioZen X")),
                 text("SoundMax Pro tiene mejor percepción; AudioZen X es más barato.")]
SYNTHESIS = [text("Respuesta final: conviene SoundMax Pro.")]


def run(supervisor, worker_messages, task=TASK):
    app = build_graph([supervisor], [ScriptedWorker(messages=iter(worker_messages))])
    return asyncio.run(run_task(app, task, verbose=False))


# ---------------------------------------------------------------- end-to-end flows
def test_happy_path_researcher_analyst_synthesizer():
    supervisor = ScriptedSupervisor(("researcher", "Buscá ambos"), ("analyst", "Analizá ambos"),
                                    ("synthesizer", "Respondé"), ("FINISH", ""))
    state = run(supervisor, RESEARCH_BOTH + ANALYSIS_BOTH + SYNTHESIS)
    assert [d.next_agent for d in state["decisions"]] == ["researcher", "analyst", "synthesizer", "FINISH"]
    assert [c.agent for c in state["contributions"]] == ["researcher", "analyst", "synthesizer"]
    assert state["final_answer"] == "Respuesta final: conviene SoundMax Pro."
    assert all(state["decisions"][-1].rubric.values())                       # validated before END
    soundmax = state["analysis"].get("SoundMax Pro")
    assert soundmax.avg_rating == 4.17 and soundmax.avg_price == 182000 and soundmax.review_count == 6


def test_incomplete_research_is_refined_before_analysis():
    # Round 1 only covers SoundMax Pro. The LLM wants to go to the analyst, the rubric sends it back.
    supervisor = ScriptedSupervisor(("researcher", "Buscá"), ("analyst", "Analizá"), ("analyst", "Analizá"),
                                    ("synthesizer", "Respondé"), ("FINISH", ""))
    worker = [calls(("search_reviews", "SoundMax Pro"), ("search_prices", "SoundMax Pro")), text("Solo SoundMax."),
              calls(("search_reviews", "AudioZen X"), ("search_prices", "AudioZen X")), text("Ahora AudioZen."),
              *ANALYSIS_BOTH, *SYNTHESIS]
    state = run(supervisor, worker)
    flow = [d.next_agent for d in state["decisions"]]
    assert flow[:3] == ["researcher", "researcher", "analyst"]
    second = state["decisions"][1]
    assert second.overridden and "AudioZen X" in second.instruction           # tells WHAT to refine
    assert [c.round for c in state["contributions"] if c.agent == "researcher"] == [1, 2]
    assert {p.product for p in state["research"].products if p.found} == {"SoundMax Pro", "AudioZen X"}  # merged


def test_finish_without_final_answer_is_not_allowed():
    supervisor = ScriptedSupervisor(("researcher", "Buscá"), ("FINISH", ""))
    state = run(supervisor, RESEARCH_BOTH + ANALYSIS_BOTH + SYNTHESIS)
    assert state["decisions"][1].overridden and state["decisions"][1].next_agent == "analyst"
    assert state["final_answer"] and state["decisions"][-1].next_agent == "FINISH"


def test_infinite_supervisor_is_stopped():
    # The LLM always asks for more research and the researcher never finds anything useful
    supervisor = ScriptedSupervisor(("researcher", "Buscá otra vez"))
    worker = [text("No encontré nada.")] * 50
    state = run(supervisor, worker)
    assert sum(c.agent == "researcher" for c in state["contributions"]) == MAX_ROUNDS   # refinement budget
    assert state["final_answer"] and len(state["decisions"]) <= MAX_STEPS
    assert state["decisions"][-1].next_agent == "FINISH"


def test_supervisor_llm_failure_falls_back_to_the_rubric_plan():
    state = run(ScriptedSupervisor(fail=True), RESEARCH_BOTH + ANALYSIS_BOTH + SYNTHESIS)
    assert [d.next_agent for d in state["decisions"]] == ["researcher", "analyst", "synthesizer", "FINISH"]
    assert "plan determinístico" in state["decisions"][0].reason


def test_specialist_failure_does_not_stop_the_graph():
    class Broken(ScriptedWorker):
        async def _agenerate(self, *args, **kwargs):
            raise ConnectionError("503")

    app = build_graph([ScriptedSupervisor(("researcher", "Buscá"))], [Broken(messages=iter([]))])
    state = asyncio.run(run_task(app, TASK, verbose=False))
    assert any(c.error for c in state["contributions"])
    assert state["final_answer"]                                              # still answers (with limitations)


def test_unknown_product_is_reported_not_invented():
    supervisor = ScriptedSupervisor(("researcher", "Buscá"), ("synthesizer", "Respondé"), ("FINISH", ""))
    worker = [calls(("search_reviews", "Zumbador 9000")), text("No existe."), text("No hay datos de Zumbador 9000.")]
    state = run(supervisor, worker, task="¿Qué opinan del Zumbador 9000?")
    assert state["research"].products[0].found is False
    assert state["analysis"] is None and "No hay datos" in state["final_answer"]


# ---------------------------------------------------------------- state and rubric
def test_nodes_cannot_write_other_agents_keys():
    async def rogue(state):
        return {"analysis": None, "research": None}

    with pytest.raises(StateOwnershipError, match="research"):
        asyncio.run(guarded("analyst", rogue)(initial_state(TASK)))


def test_rubric_detects_missing_products_and_inconsistency():
    state = initial_state(TASK)
    rubric, problems = evaluate_rubric(state)
    assert not any(rubric.values()) and "no hay investigación" in problems[0]
    reviews = [Review(**r) for r in sources.PRODUCTS["SoundMax Pro"]["reviews"]]
    state["research"] = ResearchFindings(products=[ProductFindings(product="SoundMax Pro", found=True, reviews=reviews)],
                                         summary="")
    rubric, problems = evaluate_rubric(state)
    assert not rubric["research_complete"]
    assert any("AudioZen X" in p for p in problems) and any("precios de SoundMax Pro" in p for p in problems)


# ---------------------------------------------------------------- tools
def test_search_tool_resolves_aliases_and_suggests():
    found = asyncio.run(search_reviews.ainvoke({"product": "soundmax"}))
    assert found["product"] == "SoundMax Pro" and found["review_count"] == 6
    missing = asyncio.run(search_reviews.ainvoke({"product": "AudioZen Ultra"}))
    assert missing["found"] is False and missing["suggestions"] == ["AudioZen X"]


def test_merge_keeps_previous_rounds():
    first = merge_findings(None, [("search_reviews", {"product": "SoundMax Pro"},
                                   asyncio.run(search_reviews.ainvoke({"product": "SoundMax Pro"})))], "r1")
    second = merge_findings(first, [("search_reviews", {"product": "AudioZen X"},
                                     asyncio.run(search_reviews.ainvoke({"product": "AudioZen X"})))], "r2")
    assert {p.product for p in second.products} == {"SoundMax Pro", "AudioZen X"} and second.summary == "r2"


def test_sentiment_handles_negation_and_modifiers():
    assert review_polarity("Excelente sonido, muy cómodos") == 1
    assert review_polarity("Se rompió al mes. Mala calidad.") == -1
    assert review_polarity("No están mal") > 0                               # negation flips "mal"
    assert review_polarity("Buen sonido, aunque un poco caros") == 0          # "un poco" is a modifier


def test_analyst_tools_only_see_the_researched_products():
    reviews = [Review(**r) for r in sources.PRODUCTS["AudioZen X"]["reviews"]]
    tools = {t.name: t for t in make_analyst_tools(
        ResearchFindings(products=[ProductFindings(product="AudioZen X", found=True, reviews=reviews)], summary=""))}
    stats = tools["compute_statistics"].invoke({"product": "AudioZen X"})
    assert stats["avg_rating"] == 2.83 and stats["review_count"] == 6
    assert "error" in tools["analyze_sentiment"].invoke({"product": "SoundMax Pro"})   # not researched


# ---------------------------------------------------------------- rate limits
def test_rate_limit_waits_and_retries_the_same_model(monkeypatch):
    import agents.llm as llm
    monkeypatch.setattr(llm, "RATE_LIMIT_WAIT_SECONDS", 0.01)
    attempts = []

    async def call(model):
        attempts.append(model)
        if len(attempts) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: You exceeded your current quota")
        return "ok"

    class Model:
        model = "gemini-flash-lite-latest"

    same = Model()
    assert asyncio.run(llm.with_fallback([same, same], call, "test")) == "ok"
    assert len(attempts) == 2                                   # waited and retried once, no duplicate fallback


def test_duplicate_fallback_is_not_retried_on_other_errors():
    import agents.llm as llm
    attempts = []

    async def call(model):
        attempts.append(model)
        raise ConnectionError("down")

    class Model:
        model = "same-model"

    with pytest.raises(ConnectionError):
        asyncio.run(llm.with_fallback([Model(), Model()], call, "test"))
    assert len(attempts) == 1
