"""Multi-level technical support assistant designed as a LangGraph state graph.

    START -> triage --(simple)--> search_knowledge -> evaluator --(valid)----------> END
                    \\                     ^             |
                     \\                    +--(retry)----+ (attempts < 3)
                      \\                                 |
                       +--(complex)--> human_support <--+ (attempts >= 3)
                                            |
                                           END

Runs without API keys: triage, search and evaluation are simulated with simple rules so the
mechanics of the graph (state updates, reducer, conditional edges, cycle) can be seen.
"""

from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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
    is_escalated: bool                # True once the case is handed to a human


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
    """Classifies the user's question."""
    question = state["messages"][-1].content.lower()
    if any(k in question for k in COMPLEX_KEYWORDS):
        category = "complex"
    elif "contraseña" in question or "sesión" in question:
        category = "account"
    else:
        category = "billing"
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
    """Checks whether the solution answers the question. It only writes its verdict:
    the routing decision belongs to the conditional edge."""
    solution = state["solution"]
    # Simulated criterion: a useful answer gives a concrete action (here, it mentions a link or a button)
    valid = bool(solution) and ("enlace" in solution or "'" in solution)
    print(f"[evaluator] valid={valid}")
    update = {"is_solution_valid": valid}
    if valid:
        update["messages"] = [AIMessage(content=solution)]
    return update


def human_support(state: SupportState) -> dict:
    """Hands the case to a human agent."""
    print(f"[human] escalated after {state['attempts']} attempt(s), category={state['category']}")
    return {"is_escalated": True,
            "messages": [AIMessage(content="Derivamos tu caso a una persona del equipo de soporte.")]}


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
def build_graph():
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
    return workflow.compile()


def initial_state(question: str) -> SupportState:
    return {"messages": [HumanMessage(content=question)], "category": "", "attempts": 0,
            "solution": None, "is_solution_valid": False, "is_escalated": False}


if __name__ == "__main__":
    app = build_graph()
    scenarios = [
        "No puedo iniciar sesión, creo que me olvidé la contraseña",   # solved on the 2nd attempt
        "Me cobraron dos veces la suscripción",                        # no articles -> escalated after 3
        "Hay una caída en producción y los clientes no pueden pagar",  # complex -> escalated by triage
    ]
    for question in scenarios:
        print(f"\n=== {question}")
        final = app.invoke(initial_state(question))
        print(f"→ attempts={final['attempts']} | is_escalated={final['is_escalated']} | "
              f"messages={len(final['messages'])} | answer: {final['messages'][-1].content}")

    # The compiled graph can draw itself: this is the source of diagram.mmd
    print("\n" + app.get_graph().draw_mermaid())
