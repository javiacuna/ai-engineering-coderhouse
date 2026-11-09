"""Shared state between two async agents, with Pydantic-validated artifacts and write isolation.

    START -> investigador -> revisor -> END

- Each agent owns one "plot" of the state and returns ONLY that key: the researcher writes
  researcher_output, the reviewer writes reviewer_output. A guard rejects any write outside it.
- Artifacts are validated with Pydantic BEFORE entering the state: an invalid score (e.g. 1.5) is
  caught inside the node and recorded in `errors`, instead of crashing the graph or leaving it in limbo.
"""

import asyncio
import functools
import operator
from typing import Annotated, Any, Awaitable, Callable, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

APPROVAL_THRESHOLD = 0.8   # the reviewer approves research with confidence_score above this value
RESEARCH_DELAY_SECONDS = 1.0


# TODO: Define ResearchArtifact y ReviewArtifact usando Pydantic
# Asegúrate de incluir validaciones (ej. score entre 0 y 1)
class ResearchArtifact(BaseModel):
    """Output contract of the researcher."""

    topic: str = Field(min_length=1)
    findings: List[str] = Field(min_length=1, description="At least one finding")
    confidence_score: float = Field(ge=0, le=1, description="How reliable the findings are, from 0 to 1")

    @field_validator("findings")
    @classmethod
    def no_empty_findings(cls, values: List[str]) -> List[str]:
        cleaned = [v.strip() for v in values if v.strip()]
        if not cleaned:
            raise ValueError("findings cannot be empty strings")
        return cleaned


class ReviewArtifact(BaseModel):
    """Output contract of the reviewer."""

    is_approved: bool
    feedback: str = Field(min_length=1)


class AgentState(TypedDict):
    topic: str                                          # input, written only by the caller
    # TODO: Añadir campos para researcher_output y reviewer_output
    researcher_output: Optional[ResearchArtifact]      # owned by "investigador"
    reviewer_output: Optional[ReviewArtifact]          # owned by "revisor"
    # Shared error log. Reducer operator.add concatenates, so any node can APPEND its error without
    # overwriting the others' (the safe alternative to a global list with .append())
    errors: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Write isolation: each node may only return the keys it owns
# ---------------------------------------------------------------------------
NodeFunction = Callable[..., Awaitable[dict[str, Any]]]


class StateOwnershipError(Exception):
    """A node tried to write a part of the state that belongs to another agent."""


def owns(*keys: str) -> Callable[[NodeFunction], NodeFunction]:
    """Declares which state keys a node may write. "errors" is shared (append-only) and always allowed."""
    allowed = set(keys) | {"errors"}

    def decorator(node: NodeFunction) -> NodeFunction:
        @functools.wraps(node)
        async def guarded(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            update = await node(state, config)
            foreign = set(update) - allowed
            if foreign:
                raise StateOwnershipError(f"node '{node.__name__}' wrote {sorted(foreign)}, it only owns {sorted(keys)}")
            return update

        guarded.owned_keys = tuple(keys)
        return guarded

    return decorator


# TODO: Implementar el nodo de investigación asíncrono
@owns("researcher_output")
async def researcher_node(state: AgentState, config: RunnableConfig):
    # 1. Crear un ResearchArtifact
    # 2. Retornar la actualización del estado para 'researcher_output'
    print(f"--- Investigando: {state['topic']} ---")
    await asyncio.sleep(RESEARCH_DELAY_SECONDS)   # simulated research (an LLM or a web search): the event loop stays free meanwhile

    # The simulated confidence comes from the run config (not from the state): it is a test knob, not data
    confidence = config.get("configurable", {}).get("simulated_confidence", 0.92)
    try:
        artifact = ResearchArtifact(
            topic=state["topic"],
            findings=[
                "Las topologías jerárquicas centralizan el control en un supervisor.",
                "Cada agente debe escribir solo su parte del estado para evitar condiciones de carrera.",
                "asyncio permite que los agentes esperen al LLM sin bloquear al orquestador.",
            ],
            confidence_score=confidence,
        )
    except ValidationError as e:
        # The invalid artifact never enters the state: the error is recorded and the flow continues
        problem = e.errors()[0]
        message = f"investigador: artefacto inválido ({problem['loc'][0]}: {problem['msg']})"
        print(f"    ValidationError manejado → {message}")
        return {"researcher_output": None, "errors": [message]}
    return {"researcher_output": artifact}


# TODO: Implementar el nodo de revisión asíncrono
@owns("reviewer_output")
async def reviewer_node(state: AgentState, config: RunnableConfig):
    # 1. Leer researcher_output del estado
    # 2. Validar y crear un ReviewArtifact
    # 3. Retornar la actualización para 'reviewer_output'
    print("--- Revisando resultados ---")
    research = state.get("researcher_output")
    if research is None:
        review = ReviewArtifact(is_approved=False,
                                feedback="No hay resultados de investigación válidos para revisar.")
    elif research.confidence_score > APPROVAL_THRESHOLD:
        review = ReviewArtifact(is_approved=True,
                                feedback=f"Aprobado: {len(research.findings)} hallazgos con confianza "
                                         f"{research.confidence_score:.2f}.")
    else:
        review = ReviewArtifact(is_approved=False,
                                feedback=f"Rechazado: la confianza {research.confidence_score:.2f} no supera "
                                         f"{APPROVAL_THRESHOLD}. Se necesitan más fuentes que respalden los hallazgos.")
    return {"reviewer_output": review}


# Configuración del Grafo
workflow = StateGraph(AgentState)

# TODO: Añadir nodos y definir el flujo (entry point, edges)
workflow.add_node("investigador", researcher_node)
workflow.add_node("revisor", reviewer_node)
workflow.set_entry_point("investigador")
# The reviewer runs only after the researcher finished writing: the edge is the dependency guarantee
workflow.add_edge("investigador", "revisor")
workflow.add_edge("revisor", END)

app = workflow.compile()


async def run_example(simulated_confidence: float = 0.92) -> AgentState:
    initial_state = {"topic": "Sistemas Multi-Agente", "researcher_output": None, "reviewer_output": None, "errors": []}
    # TODO: Ejecutar el grafo usando app.astream
    config = {"configurable": {"simulated_confidence": simulated_confidence}}
    final_state: dict[str, Any] = dict(initial_state)
    async for update in app.astream(initial_state, config, stream_mode="updates"):
        for node, output in update.items():
            print(f"Nodo '{node}' finalizado.")
            for key, value in output.items():
                final_state[key] = final_state[key] + value if key == "errors" else value

    review = final_state["reviewer_output"]
    print(f"Revisión: {'APROBADA' if review.is_approved else 'RECHAZADA'} — {review.feedback}")
    if final_state["errors"]:
        print(f"Errores registrados: {final_state['errors']}")
    return final_state


async def main() -> None:
    for confidence, label in [(0.92, "confianza alta"), (0.6, "confianza baja"), (1.5, "score inválido")]:
        print(f"\n===== Escenario: {label} (confidence_score={confidence}) =====")
        await run_example(confidence)


if __name__ == "__main__":
    asyncio.run(main())
