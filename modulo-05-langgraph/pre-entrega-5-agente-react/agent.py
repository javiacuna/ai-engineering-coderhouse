"""ReAct agent with LangGraph and SQLite persistence.

    START -> agent --(tools_condition: the LLM asked for tools)--> tools --+
               ^                                                         |
               +------------------- observation -------------------------+
             agent --(no tool calls: final answer)--> END

- The LLM decides by itself when and which tool to call (no manual if/else routing).
- Tool errors come back as observations, so the agent can retry with other arguments or ask the user.
- AsyncSqliteSaver stores a checkpoint after every node, per thread_id, in a SQLite file:
  the conversation survives a restart of the script.
"""

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from tools import TOOLS

logger = logging.getLogger("agent")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "checkpoints.sqlite"

RECURSION_LIMIT = 10          # max graph steps per turn: caps cost if the model keeps calling tools
MAX_CONTEXT_TOKENS = 4000     # history sent to the LLM is trimmed to this budget ("dirty state")
LLM_TIMEOUT = 60              # seconds per LLM call
LLM_MAX_RETRIES = 2           # the SDK default is 6 retries without timeout

SYSTEM_PROMPT = """\
Sos un asistente de atención al cliente de una tienda de tecnología. Respondés en español, de forma breve.
Tenés herramientas para consultar clientes y pedidos en la base de datos.

Reglas:
1. Nunca inventes datos: todo número, nombre, fecha o estado tiene que salir de una herramienta o de la conversación.
2. Podés llamar a varias herramientas, una después de otra, hasta tener todo lo necesario para responder.
3. Si una herramienta devuelve "error", leé la "sugerencia": reintentá con otros argumentos o con otra herramienta.
   Si no podés resolverlo solo, pedile al usuario el dato que falta.
4. Si una búsqueda devuelve varios clientes posibles, preguntá cuál es antes de seguir.
5. Usá el contexto de la conversación: si el usuario dice "¿y el último?", se refiere a lo que se venía hablando.
6. Mostrá los montos en pesos con separador de miles, por ejemplo $14.500."""


# ---------------------------------------------------------------------------
# 1. State
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    """Inherits `messages: Annotated[list[AnyMessage], add_messages]`.

    add_messages is the reducer: each node returns only its NEW messages and they are appended to the
    history (and a message with an existing id replaces it instead of duplicating it). The state is
    never mutated in place: nodes return updates and LangGraph merges them.
    """


# ---------------------------------------------------------------------------
# 2. Model (with the tools bound)
# ---------------------------------------------------------------------------
def build_model() -> Runnable:
    """LLM with bind_tools(TOOLS): the tools' names, descriptions and schemas are sent on every call.

    LLM_PROVIDER: gemini (default, free tier), openai or anthropic.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0,
                          timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES).bind_tools(TOOLS)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"), max_tokens=1024,
                             timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES).bind_tools(TOOLS)

    from langchain_google_genai import ChatGoogleGenerativeAI

    def gemini(name: str) -> Runnable:
        return ChatGoogleGenerativeAI(model=name, temperature=0, timeout=LLM_TIMEOUT,
                                      max_retries=LLM_MAX_RETRIES).bind_tools(TOOLS)

    model = gemini(os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    fallback = os.getenv("GEMINI_FALLBACK_MODEL")
    return model.with_fallbacks([gemini(fallback)]) if fallback else model


# ---------------------------------------------------------------------------
# 3. Graph
# ---------------------------------------------------------------------------
def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keeps the most recent part of the history that fits in MAX_CONTEXT_TOKENS.

    The checkpoint keeps the FULL history; only what is sent to the LLM is trimmed. start_on="human"
    makes the trimmed history start at a user message, so a tool result is never separated from the
    tool call that produced it (the APIs reject that).
    """
    return trim_messages(messages, max_tokens=MAX_CONTEXT_TOKENS, strategy="last",
                         token_counter=count_tokens_approximately, start_on="human",
                         end_on=("human", "tool"), include_system=False)


def build_graph(model: Runnable | BaseChatModel, checkpointer: Any = None) -> CompiledStateGraph:
    """The model and the checkpointer are passed in: infrastructure objects do not belong to the state."""

    async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        history = trim_history(state["messages"])
        response = await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *history])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    # handle_tool_errors=True: an unexpected exception or invalid arguments become a ToolMessage with
    # the error text, so the agent can correct itself instead of the graph crashing
    workflow.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=checkpointer)


@asynccontextmanager
async def sqlite_checkpointer(path: Path = DB_PATH) -> AsyncIterator[AsyncSqliteSaver]:
    """AsyncSqliteSaver: the async version of SqliteSaver (the graph runs with ainvoke/astream)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        await saver.setup()   # creates the tables on a new file (otherwise they appear only on the first write)
        yield saver


# ---------------------------------------------------------------------------
# 4. Running a turn and recording its ReAct trace
# ---------------------------------------------------------------------------
class TraceStep(BaseModel):
    type: Literal["accion", "observacion", "respuesta"]
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: Any = None
    text: str | None = None


class TurnTrace(BaseModel):
    thread_id: str
    question: str
    steps: list[TraceStep]
    answer: str
    tool_calls: int
    messages_in_thread: int
    duration_seconds: float
    error: str | None = None


def parse_tool_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def steps_from_update(node: str, update: dict[str, Any]) -> list[TraceStep]:
    """Converts one graph update (what a node returned) into trace steps."""
    steps = []
    for message in (update or {}).get("messages", []):
        if isinstance(message, AIMessage) and message.tool_calls:
            steps += [TraceStep(type="accion", tool=c["name"], args=c["args"]) for c in message.tool_calls]
        elif isinstance(message, AIMessage):
            steps.append(TraceStep(type="respuesta", text=message.text))
        elif isinstance(message, ToolMessage):
            steps.append(TraceStep(type="observacion", tool=message.name, result=parse_tool_content(message.content)))
    return steps


async def run_turn(app: CompiledStateGraph, question: str, thread_id: str,
                   recursion_limit: int = RECURSION_LIMIT) -> TurnTrace:
    """Sends a user message to a thread, streaming the ReAct steps as they happen."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    steps: list[TraceStep] = []
    error: str | None = None
    started = time.perf_counter()
    logger.info("[%s] usuario: %s", thread_id, question)
    try:
        # stream_mode="updates": one item per executed node, with the update that node returned
        async for chunk in app.astream({"messages": [HumanMessage(content=question)]}, config, stream_mode="updates"):
            for node, update in chunk.items():
                for step in steps_from_update(node, update):
                    steps.append(step)
                    if step.type == "accion":
                        logger.info("[%s]   → acción: %s(%s)", thread_id, step.tool, step.args)
                    elif step.type == "observacion":
                        logger.info("[%s]   ← observación de %s: %s", thread_id, step.tool,
                                    json.dumps(step.result, ensure_ascii=False, default=str)[:300])
    except GraphRecursionError:
        error = f"Se alcanzó el límite de {recursion_limit} pasos sin una respuesta final."
        logger.warning("[%s] %s", thread_id, error)

    state = await app.aget_state(config)
    answer = next((s.text for s in reversed(steps) if s.type == "respuesta"), None) or error or ""
    logger.info("[%s] agente: %s", thread_id, answer)
    return TurnTrace(thread_id=thread_id, question=question, steps=steps, answer=answer,
                     tool_calls=sum(s.type == "accion" for s in steps),
                     messages_in_thread=len(state.values.get("messages", [])),
                     duration_seconds=round(time.perf_counter() - started, 2), error=error)
