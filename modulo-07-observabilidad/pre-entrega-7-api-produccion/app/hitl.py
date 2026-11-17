"""Human-in-the-loop: which tasks are critical, the approval nodes (interrupt) and the approval endpoint.

Two moments need a human, and each one is a node of the graph that pauses it with interrupt():

1. "execution" — before running a task whose estimated cost reaches HITL_COST_THRESHOLD_USD
   (business rule by cost: e.g. comparing 3 or more products). Rejected → nothing is run, no tokens spent.
2. "delivery" — before SENDING the report by email (`deliver_to`): a side effect that cannot be undone.
   The reviewer sees the draft. Rejected → the answer stays in the job, but nothing is sent.

How the pause works: interrupt(request) stops the graph and the checkpointer saves its state in Redis under
the job's thread_id. The worker marks the job WAITING_APPROVAL and is free for other jobs (nobody waits in
memory). POST /tasks/{id}/approve enqueues a resume; a worker runs Command(resume=decision) and interrupt()
returns that decision inside the node. It works even if the API was restarted in between.

Rule of thumb from the unit: on resume the node runs AGAIN from its first line, so nothing with side
effects goes before interrupt(). The email is sent in the next node (send_report), never here.
"""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from langgraph.graph import END
from langgraph.types import interrupt
from redis.exceptions import RedisError

from app.config import Settings
from app.jobs import InvalidTransition
from app.orchestrator.data.sources import products_mentioned
from app.schemas import ApprovalDecision, ApprovalRecord, ApprovalRequest, JobView, RiskAssessment

Stage = Literal["execution", "delivery"]


# --------------------------------------------------------------------------- policy
def assess_risk(task: str, deliver_to: str | None, settings: Settings) -> RiskAssessment:
    """Deterministic business rules (not an LLM): the same task always gets the same answer.

    The estimate comes from the module 7 traces: each requested product adds a research and an analysis
    round. 1 product ≈ 9 LLM calls / US$ 0.004, 3 products ≈ 12 calls / US$ 0.0073.
    """
    products = products_mentioned(task)
    estimated_cost = round(settings.cost_base_usd + settings.cost_per_product_usd * len(products), 4)
    reasons = []
    if estimated_cost >= settings.hitl_cost_threshold_usd:
        reasons.append(f"costo estimado US$ {estimated_cost:.4f} ≥ umbral US$ {settings.hitl_cost_threshold_usd:.4f} "
                       f"({len(products)} productos: {', '.join(products)})")
    return RiskAssessment(critical=bool(reasons), reasons=reasons, products=products,
                          estimated_llm_calls=6 + 2 * len(products), estimated_cost_usd=estimated_cost,
                          needs_delivery_approval=deliver_to is not None)


def approval_request(stage: Stage, state: dict[str, Any]) -> ApprovalRequest:
    risk = state.get("risk") or {}
    if stage == "execution":
        return ApprovalRequest(stage=stage, question="¿Aprobás ejecutar esta tarea? Es crítica por su costo estimado.",
                               reasons=risk.get("reasons", []), estimated_cost_usd=risk.get("estimated_cost_usd"))
    return ApprovalRequest(stage=stage, question=f"¿Aprobás enviar este informe a {state['deliver_to']}?",
                           reasons=["enviar un email a un cliente es una acción con efectos externos que no se puede deshacer"],
                           draft=state.get("final_answer"), deliver_to=state["deliver_to"])


# --------------------------------------------------------------------------- graph nodes
def make_approval_node(stage: Stage):
    """Node that pauses the graph until a human decides. Everything before interrupt() runs again on resume,
    so it only builds the request (no side effects)."""

    async def approval_node(state: dict[str, Any]) -> dict[str, Any]:
        request = approval_request(stage, state)
        raw_decision = interrupt(request.model_dump())     # ← the graph stops here; its state is in Redis
        decision = ApprovalDecision.model_validate(raw_decision)
        record = ApprovalRecord(stage=stage, **decision.model_dump())
        update: dict[str, Any] = {"approvals": [record.model_dump()]}
        if not decision.approved:
            update["outcome"] = "rejected" if stage == "execution" else "not_delivered"
        return update

    approval_node.__name__ = f"approve_{stage}"
    return approval_node


def last_decision_approved(state: dict[str, Any]) -> bool:
    approvals = state.get("approvals") or []
    return bool(approvals) and approvals[-1]["approved"]


def route_after_execution_approval(state: dict[str, Any]) -> str:
    return "run_orchestrator" if last_decision_approved(state) else END


def route_after_delivery_approval(state: dict[str, Any]) -> str:
    return "send_report" if last_decision_approved(state) else END


# --------------------------------------------------------------------------- endpoint
router = APIRouter(tags=["human-in-the-loop"])


@router.post("/tasks/{job_id}/approve", response_model=JobView, status_code=status.HTTP_202_ACCEPTED,
             responses={404: {"description": "The job does not exist"},
                        409: {"description": "The job is not waiting for approval"}})
async def approve_task(job_id: str, decision: ApprovalDecision, request: Request) -> JobView:
    """Approves or rejects a job in WAITING_APPROVAL. Does not run anything: it enqueues the resume and answers
    at once (202). The decision is stored in the job's `approvals` for auditing."""
    jobs = request.app.state.jobs
    try:
        return await jobs.approve(job_id, decision.model_dump())
    except InvalidTransition as e:
        code = status.HTTP_404_NOT_FOUND if e.current is None else status.HTTP_409_CONFLICT
        raise HTTPException(status_code=code, detail=str(e))
    except RedisError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis no está disponible")
