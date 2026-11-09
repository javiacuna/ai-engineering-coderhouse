"""Supervisor: intelligent router, quality control and flow controller.

Each turn it:
1. Evaluates the RUBRIC (computed in code, so it can't be "hallucinated" as complete).
2. Asks its LLM which agent works next and with what instruction (structured output).
3. Applies GUARDS: a refinement budget per agent, a max number of steps, and no END without a final answer
   that passed the rubric. If the LLM fails, a deterministic plan takes over.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.llm import with_fallback
from data.sources import products_mentioned
from state import OrchestratorState, Route, SupervisorDecision, rounds_of

logger = logging.getLogger("orchestrator.supervisor")

MAX_STEPS = 8          # supervisor turns per task: the stop condition against the "infinite supervisor"
MAX_ROUNDS = 2         # times each specialist may work (1st attempt + 1 refinement)
MIN_REVIEWS = 3        # sufficiency criterion: fewer reviews are not enough to conclude


class RouteDecision(BaseModel):
    """Structured output of the supervisor LLM."""

    next_agent: Route = Field(description="'researcher', 'analyst', 'synthesizer' o 'FINISH'")
    instruction: str = Field(description="Tarea concreta para ese agente, con los productos y datos que tiene que cubrir")
    reason: str = Field(description="Por qué elegiste ese paso, en una oración")


SUPERVISOR_PROMPT = f"""\
Sos el SUPERVISOR de un equipo de análisis de productos. Coordinás; nunca investigás, calculás ni
redactás la respuesta final vos mismo.

Equipo:
- researcher: busca reseñas y precios en fuentes externas.
- analyst: analiza esos datos (sentimiento, puntajes, precios). Necesita que la investigación esté hecha.
- synthesizer: redacta la respuesta final al usuario con los resultados validados.

Rúbrica de calidad (la calcula el sistema y te la paso en cada turno):
- research_complete: cada producto pedido fue encontrado, con al menos {MIN_REVIEWS} reseñas y al menos 1 precio.
- analysis_complete: cada producto encontrado tiene sentimiento y estadísticas.
- analysis_consistent: la cantidad de reseñas analizadas coincide con la investigada.
- final_answer_ready: el synthesizer ya entregó la respuesta.

Cómo decidir:
1. Si falta investigación -> researcher, indicando QUÉ productos buscar.
2. Si la investigación está completa pero falta o falla el análisis -> analyst, indicando qué productos analizar.
3. Si un ítem falla después de que un agente trabajó, pedile que REFINE indicando exactamente qué falta.
4. Si los tres primeros ítems se cumplen -> synthesizer, recordándole qué preguntó el usuario.
5. Si final_answer_ready es true -> FINISH.
Cada especialista puede trabajar como máximo {MAX_ROUNDS} veces. Si un producto no existe en las fuentes,
no insistas: pasá al análisis de lo que sí hay y que el synthesizer lo aclare."""


# ---------------------------------------------------------------------------
# Rubric (code, not LLM)
# ---------------------------------------------------------------------------
def evaluate_rubric(state: OrchestratorState) -> tuple[dict[str, bool], list[str]]:
    """Returns the rubric and the list of concrete problems (used as refinement instructions)."""
    research, analysis = state.get("research"), state.get("analysis")
    requested = products_mentioned(state["task"])
    problems: list[str] = []

    if research is None:
        problems.append(f"no hay investigación (productos pedidos: {', '.join(requested) or 'los que mencione el usuario'})")
    else:
        for product in requested:
            found = research.get(product)
            if found is None or not found.found:
                problems.append(f"falta investigar {product}")
            elif len(found.reviews) < MIN_REVIEWS:
                problems.append(f"{product} tiene {len(found.reviews)} reseñas (mínimo {MIN_REVIEWS})")
            elif not found.prices:
                problems.append(f"faltan precios de {product}")
    research_complete = research is not None and not problems

    analysis_problems: list[str] = []
    found_products = [p for p in research.products if p.found] if research else []
    if found_products:
        if analysis is None:
            analysis_problems.append("no hay análisis")
        else:
            for findings in found_products:
                result = analysis.get(findings.product)
                if result is None or result.sentiment_score is None or result.avg_rating is None:
                    analysis_problems.append(f"falta sentimiento o estadísticas de {findings.product}")
    analysis_complete = bool(found_products) and analysis is not None and not analysis_problems

    consistency_problems: list[str] = []
    if analysis is not None and research is not None:
        for result in analysis.products:
            findings = research.get(result.product)
            if findings and result.review_count != len(findings.reviews):
                consistency_problems.append(
                    f"{result.product}: se analizaron {result.review_count} reseñas pero se investigaron {len(findings.reviews)}")
    analysis_consistent = analysis_complete and not consistency_problems

    rubric = {
        "research_complete": research_complete,
        "analysis_complete": analysis_complete,
        "analysis_consistent": analysis_consistent,
        "final_answer_ready": bool(state.get("final_answer")),
    }
    return rubric, problems + analysis_problems + consistency_problems


def deterministic_plan(state: OrchestratorState, rubric: dict[str, bool], problems: list[str]) -> RouteDecision:
    """The route the rubric implies. Used when the LLM fails, and as the target of the guards."""
    detail = "; ".join(problems) or "sin problemas"
    if rubric["final_answer_ready"]:
        return RouteDecision(next_agent="FINISH", instruction="", reason="la respuesta final está lista")
    if not rubric["research_complete"] and rounds_of(state, "researcher") < MAX_ROUNDS:
        return RouteDecision(next_agent="researcher", reason="la investigación está incompleta",
                             instruction=f"Buscá reseñas y precios. Pendiente: {detail}. Pedido del usuario: {state['task']}")
    if not (rubric["analysis_complete"] and rubric["analysis_consistent"]) and rounds_of(state, "analyst") < MAX_ROUNDS \
            and state.get("research") and any(p.found for p in state["research"].products):
        return RouteDecision(next_agent="analyst", reason="el análisis está incompleto o es inconsistente",
                             instruction=f"Analizá sentimiento y estadísticas de cada producto con datos. Pendiente: {detail}")
    return RouteDecision(next_agent="synthesizer", reason="la rúbrica se cumple o se agotaron las revisiones",
                         instruction=f"Respondé el pedido del usuario. Limitaciones a aclarar: {detail}")


def apply_guards(choice: RouteDecision, state: OrchestratorState, rubric: dict[str, bool],
                 problems: list[str]) -> tuple[RouteDecision, bool]:
    """Corrects the LLM's choice when it breaks a hard rule. Returns (decision, overridden)."""
    plan = deterministic_plan(state, rubric, problems)
    step = state.get("step", 0) + 1

    if rubric["final_answer_ready"]:
        return plan, choice.next_agent != "FINISH"                          # done: always finish
    if step >= MAX_STEPS:                                                   # stop condition
        forced = RouteDecision(next_agent="synthesizer", reason=f"límite de {MAX_STEPS} pasos alcanzado",
                               instruction=f"Respondé con lo disponible. Limitaciones: {'; '.join(problems) or 'ninguna'}")
        return forced, True
    if choice.next_agent == "FINISH":                                       # no END without a final answer
        return plan, True
    if choice.next_agent in ("researcher", "analyst") and rounds_of(state, choice.next_agent) >= MAX_ROUNDS:
        return plan, True                                                   # refinement budget exhausted
    if choice.next_agent in ("analyst", "synthesizer") and not rubric["research_complete"] and plan.next_agent == "researcher":
        return plan, True                                                   # nothing to analyze/summarize yet
    if choice.next_agent == "synthesizer" and plan.next_agent == "analyst":
        return plan, True                                                   # analysis missing or inconsistent
    return choice, False


def contributions_digest(state: OrchestratorState) -> str:
    lines = []
    for c in state.get("contributions", []):
        status = f"ERROR: {c.error}" if c.error else c.summary[:300].replace("\n", " ")
        tools = f" | herramientas: {', '.join(c.tools_used)}" if c.tools_used else ""
        lines.append(f"- {c.agent} (ronda {c.round}){tools} | {status}")
    return "\n".join(lines) or "- (nadie trabajó todavía)"


def make_supervisor_node(models: list[Any]):
    async def supervisor_node(state: OrchestratorState) -> dict[str, Any]:
        step = state.get("step", 0) + 1
        rubric, problems = evaluate_rubric(state)
        # Context isolation: the supervisor gets the task, the rubric and a digest of the contributions,
        # not the raw data nor the agents' tool transcripts
        context = (
            f"PEDIDO DEL USUARIO:\n{state['task']}\n\n"
            f"PASO {step} de {MAX_STEPS}.\n\nRÚBRICA:\n"
            + "\n".join(f"- {k}: {'✓' if v else '✗'}" for k, v in rubric.items())
            + f"\n\nPROBLEMAS DETECTADOS:\n{'; '.join(problems) or 'ninguno'}\n\n"
            f"RONDAS USADAS: researcher {rounds_of(state, 'researcher')}/{MAX_ROUNDS}, "
            f"analyst {rounds_of(state, 'analyst')}/{MAX_ROUNDS}\n\n"
            f"CONTRIBUCIONES:\n{contributions_digest(state)}"
        )
        messages = [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=context)]

        try:
            choice = await with_fallback(models, lambda m: m.with_structured_output(RouteDecision).ainvoke(messages),
                                         "supervisor")
            if isinstance(choice, dict):
                choice = RouteDecision(**choice)
        except Exception as e:   # the supervisor never leaves the graph stuck: the rubric decides
            logger.error("supervisor LLM failed (%s): using the deterministic plan", type(e).__name__)
            choice = deterministic_plan(state, rubric, problems)
            choice.reason = f"(plan determinístico, el LLM falló) {choice.reason}"

        decision, overridden = apply_guards(choice, state, rubric, problems)
        if overridden:
            decision.reason = f"[regla del sistema] {decision.reason} (el LLM había elegido {choice.next_agent})"
        record = SupervisorDecision(step=step, next_agent=decision.next_agent, instruction=decision.instruction,
                                    reason=decision.reason, rubric=rubric, overridden=overridden)
        logger.info("step %d -> %s | %s", step, decision.next_agent, decision.reason)
        return {
            "next_agent": decision.next_agent, "instruction": decision.instruction, "step": step,
            "decisions": [record],
            "messages": [AIMessage(content=f"→ {decision.next_agent}: {decision.reason}", name="supervisor")],
        }

    return supervisor_node


def route_from_supervisor(state: OrchestratorState) -> Route:
    """Conditional edge: the node name chosen by the supervisor (Literal return)."""
    return state["next_agent"]
