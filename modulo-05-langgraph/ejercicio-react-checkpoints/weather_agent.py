"""ReAct weather agent with LangGraph and checkpoint persistence.

    START -> agent --(the LLM asked for a tool)--> tools --+
               ^                                           |
               +-------------------------------------------+   (observation goes back to the agent)
             agent --(no tool calls: final answer)--> END

The checkpointer saves the state after every node, per thread_id: a second message on the same
thread sees the whole previous conversation; a different thread_id starts from zero.

The course template uses OpenAI; this version uses Gemini (free tier) by default.
Set LLM_PROVIDER=openai and OPENAI_API_KEY to use gpt-4o-mini instead.
"""

import asyncio
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# Max graph steps per invocation: a model that keeps calling tools can never loop forever
RECURSION_LIMIT = 10

# Each LLM call waits at most LLM_TIMEOUT seconds and is retried LLM_MAX_RETRIES times
# (the SDK default is 6 retries and no timeout: a slow or overloaded model could stall for minutes
# before the fallback model gets a chance)
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 2


# 1. Define el Estado (State) del agente
class AgentState(TypedDict):
    # TODO: Usa add_messages para que el historial sea acumulativo
    messages: Annotated[list[BaseMessage], add_messages]


# 2. Herramientas
class WeatherInput(BaseModel):
    location: str = Field(..., min_length=2, description="Ciudad a consultar, por ejemplo 'Madrid' o 'Londres'.")


# Simulated weather data (a real tool would call a weather API)
WEATHER = {
    "madrid": "Soleado, 22°C",
    "londres": "Nublado con lloviznas, 14°C",
    "london": "Nublado con lloviznas, 14°C",
    "córdoba": "Despejado, 27°C",
}


@tool(args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Consulta el clima actual (condición y temperatura) de una ciudad.
    Usala siempre que el usuario pregunte por el tiempo, la temperatura o si va a llover en un lugar.
    Devuelve un texto como "Soleado, 22°C", o un aviso si no hay datos para esa ciudad."""
    return WEATHER.get(location.strip().lower(), f"No hay datos del clima para {location}.")


tools = [get_weather]
# TODO: Crea el ToolNode con la lista de herramientas
# ToolNode runs every tool call of the last AIMessage and returns one ToolMessage per call
# (with the same tool_call_id): that is how the results are mapped back to the agent.
# If a tool raises, it returns the error as a ToolMessage so the LLM can react, instead of crashing.
tool_node = ToolNode(tools)


# 3. LLM con Herramientas vinculadas
# TODO: Instancia ChatOpenAI y usa .bind_tools(tools)
def build_model() -> BaseChatModel:
    """Chat model with the tools bound (their names, descriptions and schemas are sent to the LLM)."""
    if os.getenv("LLM_PROVIDER", "gemini").lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=LLM_TIMEOUT,
                          max_retries=LLM_MAX_RETRIES).bind_tools(tools)

    from langchain_google_genai import ChatGoogleGenerativeAI
    primary = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), temperature=0,
                                     timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    model = primary.bind_tools(tools)
    fallback_name = os.getenv("GEMINI_FALLBACK_MODEL")
    if fallback_name:   # e.g. when the main model answers 503 "high demand"
        fallback = ChatGoogleGenerativeAI(model=fallback_name, temperature=0, timeout=LLM_TIMEOUT,
                                          max_retries=LLM_MAX_RETRIES).bind_tools(tools)
        model = model.with_fallbacks([fallback])
    return model


# 4. Nodos + 5. Orquestación del Grafo
def build_graph(model, checkpointer=None):
    """Builds the ReAct graph. The model is passed in (not stored in the state): infrastructure objects
    do not belong to the state, and the tests can inject a fake model."""

    async def call_model(state: AgentState):
        # TODO: Invoca al modelo con los mensajes actuales
        # Retorna un diccionario con la nueva clave "messages"
        response = await model.ainvoke(state["messages"])   # the WHOLE history: that is the memory
        return {"messages": [response]}                      # add_messages appends it

    workflow = StateGraph(AgentState)

    # TODO: Añade el nodo 'agent' y el nodo 'tools'
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # TODO: Define la lógica de aristas:
    # - START -> agent
    # - agent -> conditional_edges (usa tools_condition)
    # - tools -> agent
    workflow.add_edge(START, "agent")
    # tools_condition: "tools" if the last AIMessage has tool_calls, END otherwise.
    # So the graph ends only when the LLM decides it needs no more tools.
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")   # the observation goes back to the agent: the ReAct cycle

    # 6. Persistencia
    # TODO: Instancia MemorySaver()
    # TODO: Compila el grafo pasando el checkpointer
    memory = checkpointer or MemorySaver()   # snapshot of the state after every node, per thread_id
    return workflow.compile(checkpointer=memory)


def tools_used(messages: list[BaseMessage]) -> list[str]:
    """Tool calls made in a turn, as 'name(args) -> result'."""
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    return [f"{call['name']}({call['args']}) -> {results.get(call['id'], '?')}"
            for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]


async def ask(app, question: str, thread_id: str) -> AIMessage:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    before = len((await app.aget_state(config)).values.get("messages", []))
    print(f"[thread={thread_id}] usuario: {question}  (consultando al modelo...)", flush=True)
    result = await app.ainvoke({"messages": [HumanMessage(content=question)]}, config)
    turn = result["messages"][before:]          # only the messages of this turn
    for call in tools_used(turn):
        print(f"   herramienta: {call}")
    print(f"   agente: {result['messages'][-1].text}")
    print(f"   mensajes guardados en el thread: {len(result['messages'])}")
    return result["messages"][-1]


async def main():
    load_dotenv()
    app = build_graph(build_model())

    # Identificador de sesión para persistencia
    thread_id = "dev_test_01"

    # Primera interacción
    print("--- Paso 1 ---")
    await ask(app, "¿Qué tiempo hace en Madrid?", thread_id)

    # Segunda interacción (reanudando desde checkpoint)
    print("\n--- Paso 2 ---")
    await ask(app, "¿Y en Londres?", thread_id)

    # Memory check that does not depend on the model guessing: only the conversation history
    # can tell which city was asked first
    memory_question = "¿Cuál fue la primera ciudad por la que te pregunté?"
    print("\n--- Paso 3 (mismo thread_id: recuerda) ---")
    await ask(app, memory_question, thread_id)

    print("\n--- Paso 4 (otro thread_id: sin memoria) ---")
    await ask(app, memory_question, "otro_thread_02")

if __name__ == "__main__":
    asyncio.run(main())
