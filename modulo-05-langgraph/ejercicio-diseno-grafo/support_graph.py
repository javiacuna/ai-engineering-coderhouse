"""Multi-level technical support assistant designed as a LangGraph state graph.

    START -> triage --(simple)--> search_knowledge -> evaluator --(valid)----------> END
                    \\                     ^             |
                     \\                    +--(retry)----+ (attempts < 3)
                      \\                                 |
                       +--(complex)--> human_support <--+ (attempts >= 3)
                                       [interrupt: waits for a person]
                                            |
                                           END

- The state keeps an audit trail (failed_attempts) so the human agent knows what was tried and why it failed.
- human_support is human-in-the-loop: the graph PAUSES there (interrupt), its state is saved by a
  checkpointer, and it resumes when a person answers (Command(resume=...)).

Runs without API keys: triage, search and evaluation are simulated with simple rules so the
mechanics of the graph (state updates, reducers, conditional edges, cycle, interrupt) can be seen.
"""

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

MAX_ATTEMPTS = 3   # safety brake: the search -> evaluator cycle can never loop forever


# ---------------------------------------------------------------------------
# 1. State: the working memory that flows through the graph
# ---------------------------------------------------------------------------
class SupportState(TypedDict):
    # Reducer add_messages: nodes return only their NEW messages and they are appended,
    # instead of overwriting the conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    category: str                     # set by triage: "account", "billing", "complex"...
    attempts: int                     # search attempts made so far
    solution: Optional[str]           # last solution proposed by search_knowledge
    is_solution_valid: bool           # evaluator's verdict on that solution
    # Audit trail: why each attempt failed. Reducer operator.add concatenates lists, so every
    # node appends its entry instead of replacing the previous ones (same idea as add_messages)
    failed_attempts: Annotated[List[str], operator.add]
    is_escalated: bool                # True once the case is handed to a human
    resolved_by: Optional[str]        # "agent" or "human"


# ---------------------------------------------------------------------------
# 2. Nodes: small functions with one responsibility each; they return only what they change
# ---------------------------------------------------------------------------
COMPLEX_KEYWORDS = ("caída", "producción", "datos perdidos", "seguridad")

# Simulated knowledge base: category -> answers, from the least to the most specific.
# The first search attempt finds the generic one; the evaluator rejects it and a new search refines it.
KNOWLEDGE_BASE = {
    "account": ["Revisá que el usuario esté bien escrito.",
                "Usá 'Olvidé mi contraseña' en la pantalla de login para recibir un enlace de recuperación."],
    "billing": [],   # no articles: the search never succeeds and the case must be escalated
}


def triage(state: SupportState) -> dict:
    """Classifies the user's question. A critical case is recorded in the audit trail."""
    question = state["messages"][-1].content.lower()
    if any(k in question for k in COMPLEX_KEYWORDS):
        print("[triage] category=complex")
        return {"category": "complex",
                "failed_attempts": ["triage: caso crítico (posible caída o impacto en producción), sin búsqueda"]}
    category = "account" if ("contraseña" in question or "sesión" in question) else "billing"
    print(f"[triage] category={category}")
    return {"category": category}


def search_knowledge(state: SupportState) -> dict:
    """Searches the knowledge base (simulated RAG). Each call counts as one attempt."""
    attempt = state["attempts"] + 1
    articles = KNOWLEDGE_BASE.get(state["category"], [])
    solution = articles[min(attempt, len(articles)) - 1] if articles else None
    print(f"[search] attempt {attempt} -> {solution or 'no result'}")
    return {"attempts": attempt, "solution": solution}


def evaluator(state: SupportState) -> dict:
    """Checks whether the solution answers the question and records WHY it fails.
    It only writes its verdict: the routing decision belongs to the conditional edge."""
    solution = state["solution"]
    if not solution:
        reason = "la base de conocimiento no tiene artículos para esta categoría"
    elif "enlace" not in solution and "'" not in solution:   # simulated criterion: a concrete action
        reason = f"la solución es genérica, no indica una acción concreta: «{solution}»"
    else:
        print("[evaluator] valid=True")
        return {"is_solution_valid": True, "resolved_by": "agent", "messages": [AIMessage(content=solution)]}
    print(f"[evaluator] valid=False ({reason})")
    return {"is_solution_valid": False,
            "failed_attempts": [f"intento {state['attempts']} ({state['category']}): {reason}"]}


def human_support(state: SupportState) -> dict:
    """Hands the case to a human agent and PAUSES the graph until that person answers.

    interrupt() saves the state through the checkpointer and returns control to the caller with the
    payload below. When the graph is resumed with Command(resume=answer), this node runs again
    FROM THE START and interrupt() returns the answer: code before interrupt() must have no side
    effects (no emails, no tickets), because it runs twice.
    """
    answer = interrupt({
        "question": state["messages"][0].content,
        "category": state["category"],
        "attempts": state["attempts"],
        "failed_attempts": state["failed_attempts"],   # the context the human needs
    })
    print(f"[human] answered: {answer}")
    return {"is_escalated": True, "resolved_by": "human", "messages": [AIMessage(content=answer)]}


# ---------------------------------------------------------------------------
# 3. Routing functions (the logic of the conditional edges)
# ---------------------------------------------------------------------------
def route_after_triage(state: SupportState) -> Literal["search", "escalate"]:
    return "escalate" if state["category"] == "complex" else "search"


def route_after_evaluation(state: SupportState) -> Literal["end", "escalate", "retry"]:
    if state["is_solution_valid"]:
        return "end"                       # good solution -> END
    if state["attempts"] >= MAX_ATTEMPTS:
        return "escalate"                  # failed 3 times -> Human_Support
    return "retry"                         # try another search -> back to Search_Knowledge (CYCLE)


# ---------------------------------------------------------------------------
# 4. Graph
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    workflow = StateGraph(SupportState)
    workflow.add_node("triage", triage)
    workflow.add_node("search_knowledge", search_knowledge)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("human_support", human_support)

    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", route_after_triage,
                                   {"search": "search_knowledge", "escalate": "human_support"})
    workflow.add_edge("search_knowledge", "evaluator")
    workflow.add_conditional_edges("evaluator", route_after_evaluation,
                                   {"end": END, "escalate": "human_support", "retry": "search_knowledge"})
    workflow.add_edge("human_support", END)
    # The checkpointer saves the state after every step, per thread_id: required by interrupt().
    # InMemorySaver (formerly MemorySaver) lives in RAM; production uses a persistent one (SQLite, Postgres).
    return workflow.compile(checkpointer=checkpointer or InMemorySaver())


def initial_state(question: str) -> SupportState:
    return {"messages": [HumanMessage(content=question)], "category": "", "attempts": 0, "solution": None,
            "is_solution_valid": False, "failed_attempts": [], "is_escalated": False, "resolved_by": None}


# Simulated answers of the human agent, one per escalated case
HUMAN_ANSWERS = {
    "billing": "Revisé tu cuenta: hubo un doble cobro. Ya iniciamos la devolución, la vas a ver en 5 días hábiles.",
    "complex": "Confirmamos el incidente y el equipo de guardia ya lo está resolviendo. Te avisamos al terminar.",
}

if __name__ == "__main__":
    app = build_graph()
    scenarios = [
        "No puedo iniciar sesión, creo que me olvidé la contraseña",   # solved on the 2nd attempt
        "Me cobraron dos veces la suscripción",                        # no articles -> escalated after 3
        "Hay una caída en producción y los clientes no pueden pagar",  # complex -> escalated by triage
    ]
    for n, question in enumerate(scenarios, start=1):
        print(f"\n=== {question}")
        config = {"configurable": {"thread_id": f"ticket-{n}"}}   # one conversation = one thread
        result = app.invoke(initial_state(question), config)

        if "__interrupt__" in result:
            # The graph is paused in human_support: show the person the context it needs
            handoff = result["__interrupt__"][0].value
            print(f"⏸  paused at {app.get_state(config).next} — handoff for the human agent:")
            for entry in handoff["failed_attempts"]:
                print(f"     - {entry}")
            # ... minutes or hours later, the person answers and the SAME thread resumes ...
            result = app.invoke(Command(resume=HUMAN_ANSWERS[handoff["category"]]), config)

        print(f"→ resolved_by={result['resolved_by']} | attempts={result['attempts']} | "
              f"is_escalated={result['is_escalated']} | messages={len(result['messages'])}")
        print(f"  answer: {result['messages'][-1].content}")

    # The compiled graph can draw itself: this is the source of diagram.mmd
    print("\n" + app.get_graph().draw_mermaid())
