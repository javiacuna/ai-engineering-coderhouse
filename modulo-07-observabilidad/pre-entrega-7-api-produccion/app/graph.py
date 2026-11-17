"""Task graph: the module 6 orchestrator wrapped with risk control, human approval and delivery.

    START → assess_risk ─┬─(critical)──→ approve_execution ──(approved)──┐
                         │                       └──(rejected)──→ END     │
                         └─(not critical)───────────────────────→ run_orchestrator
                                                                          │
                       END ←──(no deliver_to)─────────────────────────────┤
                       END ←──(rejected)── approve_delivery ←─(deliver_to)┘
                                                └──(approved)──→ send_report → END

- run_orchestrator runs the module 6 multi-agent graph (supervisor + researcher + analyst + synthesizer) as a
  SUBGRAPH: it inherits the Redis checkpointer, so every step of every agent is checkpointed too.
- The graph is compiled ONCE and shared by every job: each job is a different thread_id (= job_id).
"""

import json
import logging
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from redis import asyncio as aioredis

from app.config import Settings
from app.hitl import assess_risk, make_approval_node, route_after_delivery_approval, route_after_execution_approval
from app.orchestrator.graph import GRAPH_RECURSION_LIMIT, initial_state
from app.orchestrator.graph import build_graph as build_orchestrator

logger = logging.getLogger("app.graph")

OUTBOX_KEY = "outbox"   # the simulated email server: sent reports are appended here


class TaskState(TypedDict, total=False):
    task: str
    deliver_to: Optional[str]
    risk: dict[str, Any]
    approvals: Annotated[list[dict[str, Any]], operator.add]   # every human decision, in order (audit trail)
    final_answer: Optional[str]
    flow: list[str]
    products: list[dict[str, Any]]
    delivered_to: Optional[str]
    outcome: Literal["completed", "rejected", "not_delivered"]


def build_task_graph(checkpointer: BaseCheckpointSaver, redis: aioredis.Redis, settings: Settings,
                     orchestrator: CompiledStateGraph | None = None) -> CompiledStateGraph:
    orchestrator = orchestrator or build_orchestrator()

    async def assess(state: TaskState) -> dict[str, Any]:
        """Deterministic risk policy (hitl.py): decides if the task needs approval before running."""
        return {"risk": assess_risk(state["task"], state.get("deliver_to"), settings).model_dump()}

    async def run_orchestrator(state: TaskState) -> dict[str, Any]:
        """Runs the module 6 multi-agent system and keeps only plain, serializable results."""
        try:
            final = await orchestrator.ainvoke(initial_state(state["task"]), {"recursion_limit": GRAPH_RECURSION_LIMIT})
        except GraphRecursionError:
            logger.error("orchestrator recursion limit reached")
            return {"final_answer": "Se alcanzó el límite de pasos sin una respuesta final.", "flow": [],
                    "products": [], "outcome": "completed"}
        analysis = final.get("analysis")
        products = [{"product": p.product, "avg_rating": p.avg_rating, "sentiment_score": p.sentiment_score,
                     "avg_price": p.avg_price, "best_store": p.best_store} for p in analysis.products] if analysis else []
        return {"final_answer": final.get("final_answer"), "flow": [d.next_agent for d in final["decisions"]],
                "products": products, "outcome": "completed"}

    async def send_report(state: TaskState, config: RunnableConfig) -> dict[str, Any]:
        """The side effect (simulated email): only reachable after a human approved the draft."""
        message = {"job_id": config["configurable"]["thread_id"], "to": state["deliver_to"],
                   "subject": "Informe de productos", "body": state.get("final_answer"),
                   "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        await redis.rpush(OUTBOX_KEY, json.dumps(message, ensure_ascii=False))
        logger.info("report sent to %s", state["deliver_to"])
        return {"delivered_to": state["deliver_to"], "outcome": "completed"}

    def route_after_risk(state: TaskState) -> str:
        return "approve_execution" if state["risk"]["critical"] else "run_orchestrator"

    def route_after_orchestrator(state: TaskState) -> str:
        return "approve_delivery" if state.get("deliver_to") else END

    builder = StateGraph(TaskState)
    builder.add_node("assess_risk", assess)
    builder.add_node("approve_execution", make_approval_node("execution"))
    builder.add_node("run_orchestrator", run_orchestrator)
    builder.add_node("approve_delivery", make_approval_node("delivery"))
    builder.add_node("send_report", send_report)

    builder.add_edge(START, "assess_risk")
    builder.add_conditional_edges("assess_risk", route_after_risk, ["approve_execution", "run_orchestrator"])
    builder.add_conditional_edges("approve_execution", route_after_execution_approval, ["run_orchestrator", END])
    builder.add_conditional_edges("run_orchestrator", route_after_orchestrator, ["approve_delivery", END])
    builder.add_conditional_edges("approve_delivery", route_after_delivery_approval, ["send_report", END])
    builder.add_edge("send_report", END)
    return builder.compile(checkpointer=checkpointer)
