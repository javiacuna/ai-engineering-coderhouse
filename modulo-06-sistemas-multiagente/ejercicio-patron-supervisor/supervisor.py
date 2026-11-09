"""Supervisor pattern with LangGraph: a supervisor LLM decides, at every step, which specialist works next.

    START -> supervisor --(Analista)--> Analista --+
                 ^      --(Escritor)--> Escritor --+--> back to supervisor
                 +--------------------------------+
             supervisor --(FINALIZAR)--> END

The supervisor coordinates and the specialists execute: the supervisor never analyzes or writes.

The course template uses OpenAI; this version uses Gemini (free tier) by default.
Set LLM_PROVIDER=openai and OPENAI_API_KEY to use gpt-4o-mini instead.
"""

import asyncio
import logging
import os
import re
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("supervisor")

AgentName = Literal["Analista", "Escritor", "FINALIZAR"]
RECURSION_LIMIT = 10          # max graph steps: caps cost if the supervisor ever loops
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 2


# TODO: Define el estado global del agente (AgentState)
# Debe incluir una lista de mensajes y una variable 'next' para la delegación.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # shared history; add_messages appends
    next: str                                                # the supervisor's last routing decision
    reason: str                                              # why it decided it (for the trace)


class Route(TypedDict):
    """Routing decision of the supervisor."""

    next_agent: Annotated[AgentName, ..., "Siguiente paso: 'Analista', 'Escritor' o 'FINALIZAR'"]
    reason: Annotated[str, ..., "Motivo breve de la decisión, en una oración"]


SUPERVISOR_PROMPT = """\
Sos el supervisor de un equipo que prepara informes. NO analizás datos ni escribís el informe:
solo decidís quién trabaja a continuación.

Especialistas:
- Analista: procesa los datos y devuelve los resultados numéricos.
- Escritor: redacta el informe a partir de los resultados del Analista.

Reglas, en este orden (mirá los mensajes con nombre "Analista" o "Escritor" en el historial):
1. Pedido fuera de alcance: si el usuario no pide un informe ni un análisis (un saludo, una pregunta
   general) -> "FINALIZAR", sin llamar a nadie.
2. Pedido ambiguo: si pide un informe pero no dice SOBRE QUÉ tema o datos (por ejemplo "hacé un reporte",
   "necesito un informe"), o el tema no es medible (ventas, clientes, inventario, costos, etc.) -> "FINALIZAR"
   sin llamar a nadie, y en "reason" escribí la pregunta que hay que hacerle al usuario para aclararlo.
   Ante la duda entre analizar o aclarar, aclarar: es más barato preguntar que analizar lo que no se pidió.
3. Si el Analista todavía no respondió -> "Analista".
4. Si el Analista ya respondió y el Escritor todavía no -> "Escritor".
5. Si el Escritor ya entregó el informe -> "FINALIZAR". No vuelvas a llamar a nadie.
"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_router() -> Runnable:
    """Supervisor LLM with structured output: it can only answer a valid Route."""
    if os.getenv("LLM_PROVIDER", "gemini").lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=LLM_TIMEOUT,
                          max_retries=LLM_MAX_RETRIES).with_structured_output(Route)

    from langchain_google_genai import ChatGoogleGenerativeAI

    def gemini(name: str) -> Runnable:
        return ChatGoogleGenerativeAI(model=name, temperature=0, timeout=LLM_TIMEOUT,
                                      max_retries=LLM_MAX_RETRIES).with_structured_output(Route)

    router = gemini(os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    fallback = os.getenv("GEMINI_FALLBACK_MODEL")
    return router.with_fallbacks([gemini(fallback)]) if fallback else router


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
# TODO: Implementa el nodo Supervisor
# Debe usar el LLM para decidir si el siguiente paso es "Analista", "Escritor" o "FINALIZAR".
def make_supervisor_node(router: Runnable):
    """The router is injected (not stored in the state), so the tests can use a fake one."""

    async def supervisor_node(state: AgentState) -> dict[str, Any]:
        decision: Route = await router.ainvoke([SystemMessage(content=SUPERVISOR_PROMPT), *state["messages"]])
        next_agent = decision.get("next_agent") if isinstance(decision, dict) else None
        if next_agent not in ("Analista", "Escritor", "FINALIZAR"):
            # Defensive: an invalid answer ends the run instead of routing to a node that does not exist
            logger.warning("Invalid supervisor decision %r: finishing", decision)
            next_agent = "FINALIZAR"
        reason = decision.get("reason", "") if isinstance(decision, dict) else ""
        logger.info("supervisor -> %s (%s)", next_agent, reason)
        update: dict[str, Any] = {"next": next_agent, "reason": reason}
        specialists_worked = any(getattr(m, "name", None) in ("Analista", "Escritor") for m in state["messages"])
        if next_agent == "FINALIZAR" and not specialists_worked:
            # Ambiguous or out-of-scope request: answer with the question / reason instead of ending silently
            update["messages"] = [AIMessage(content=reason or "¿Sobre qué tema necesitás el informe?", name="supervisor")]
        return update

    return supervisor_node


# Nodos especialistas
# Simulated data per topic: a real Analista would query a database or call an analytics tool
METRICS = {
    "ventas": {"variacion": "+20%", "detalle": "ingresos de $12,4 M contra $10,3 M del período anterior", "lectura": "crecimiento"},
    "clientes": {"variacion": "+8%", "detalle": "1.340 clientes activos y una retención del 91%", "lectura": "crecimiento"},
    "inventario": {"variacion": "-12%", "detalle": "rotación de 34 días, 6 productos con quiebre de stock", "lectura": "alerta"},
    "costos": {"variacion": "+5%", "detalle": "costo logístico por pedido de $2.150", "lectura": "alerta"},
}


def user_request(state: AgentState) -> str:
    """The user's request: the last message that does not come from an agent of the team."""
    return next((m.content for m in reversed(state["messages"])
                 if getattr(m, "name", None) not in ("Analista", "Escritor", "supervisor")), "")


def detect_topic(request: str) -> str | None:
    """Topic of the request, matching WHOLE words (singular or plural): a substring check would read
    "inventario" as "venta"."""
    words = set(re.findall(r"\w+", request.lower()))
    return next((topic for topic in METRICS if topic in words or topic.rstrip("s") in words), None)


def analyst_node(state: AgentState):
    """Processes the data of the topic the user asked about (read from the request, not hard-coded)."""
    request = user_request(state)
    topic = detect_topic(request)
    if topic is None:
        content = f"No tengo datos para el pedido «{request}». Temas disponibles: {', '.join(METRICS)}."
    else:
        m = METRICS[topic]
        content = f"Datos procesados ({topic}): {m['variacion']} respecto del período anterior; {m['detalle']}."
    return {"messages": [HumanMessage(content=content, name="Analista")]}


def writer_node(state: AgentState):
    """Writes the report from the Analista's results (the last message with that name)."""
    analysis = next((m.content for m in reversed(state["messages"]) if getattr(m, "name", None) == "Analista"), "")
    topic = detect_topic(user_request(state))
    if topic is None:
        content = f"No se pudo armar el informe: {analysis}"
    else:
        reading = "Crecimiento positivo" if METRICS[topic]["lectura"] == "crecimiento" else "Requiere atención"
        content = f"El informe de {topic} está listo: {reading}. {analysis.split(': ', 1)[-1]}"
    return {"messages": [HumanMessage(content=content, name="Escritor")]}


# TODO: Configura el Grafo
# 1. Crea el StateGraph
# 2. Añade los nodos: 'supervisor', 'Analista', 'Escritor'
# 3. Añade aristas: Analista -> supervisor, Escritor -> supervisor
# 4. Añade la arista condicional desde 'supervisor' basada en el estado['next']
def build_graph(router: Runnable) -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", make_supervisor_node(router))
    builder.add_node("Analista", analyst_node)
    builder.add_node("Escritor", writer_node)

    builder.set_entry_point("supervisor")
    # Every specialist reports back to the supervisor: it is the only one that decides what comes next
    builder.add_edge("Analista", "supervisor")
    builder.add_edge("Escritor", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {"Analista": "Analista", "Escritor": "Escritor", "FINALIZAR": END},
    )
    return builder.compile()


async def run(graph: CompiledStateGraph, request: str) -> list[str]:
    """Runs the graph with astream and returns the sequence of executed steps."""
    steps: list[str] = []
    print(f"Usuario: {request}\n")
    try:
        async for update in graph.astream({"messages": [HumanMessage(content=request)]},
                                          {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
            for node, output in update.items():
                if node == "supervisor":
                    steps.append("supervisor")
                    print(f"[supervisor] decide: {output['next']} ({output.get('reason', '')})")
                    if output.get("messages"):   # it finished without delegating: answer to the user
                        print(f"[supervisor → usuario] {output['messages'][-1].content}")
                else:
                    steps.append(node)
                    print(f"[{node}] {output['messages'][-1].content}")
    except GraphRecursionError:
        print(f"\nCorte de seguridad: se alcanzó el límite de {RECURSION_LIMIT} pasos (el supervisor entró en un bucle).")
        steps.append("LIMITE")
        return steps
    steps.append("fin")
    print(f"\nSecuencia: {' → '.join(steps)}")
    return steps


# Compilar y ejecutar
async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    graph = build_graph(build_router())
    requests = [
        "Necesito un reporte sobre ventas.",        # the assignment's request
        "Armame un informe de inventario.",          # another topic: the specialists adapt to it
        "Hacé un reporte.",                           # ambiguous: the supervisor asks instead of guessing
    ]
    for request in requests:
        await run(graph, request)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
