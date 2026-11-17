"""API contracts (Pydantic): what clients send and receive. They are also the OpenAPI schema in /docs."""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobStatus(str, Enum):
    PENDING = "PENDING"                    # queued, waiting for a worker
    RUNNING = "RUNNING"                    # a worker is running the graph
    WAITING_APPROVAL = "WAITING_APPROVAL"  # paused by the HITL node, state saved in Redis
    DONE = "DONE"                          # finished (approved or not critical)
    REJECTED = "REJECTED"                  # the reviewer rejected the execution: nothing was run
    FAILED = "FAILED"                      # an exception or a timeout: the error is in `error`


FINAL_STATUSES = {JobStatus.DONE, JobStatus.REJECTED, JobStatus.FAILED}


class TaskRequest(BaseModel):
    task: str = Field(min_length=5, max_length=2000, description="Request for the multi-agent system",
                      examples=["Compará SoundMax Pro y AudioZen X: sentimiento, precio promedio y recomendación"])
    deliver_to: Optional[str] = Field(
        default=None, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="Email to send the report to. Sending is a side effect: it always needs human approval")


class TaskCreated(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str = Field(description="Poll this URL until the status is DONE, REJECTED or FAILED")


class RiskAssessment(BaseModel):
    """Output of the risk node: decides whether the task needs approval before running."""

    critical: bool
    reasons: list[str] = []
    products: list[str] = []
    estimated_llm_calls: int
    estimated_cost_usd: float
    needs_delivery_approval: bool = False


class ApprovalRequest(BaseModel):
    """What the reviewer sees while the job is in WAITING_APPROVAL (the value passed to interrupt())."""

    stage: Literal["execution", "delivery"]
    question: str
    reasons: list[str] = []
    estimated_cost_usd: Optional[float] = None
    draft: Optional[str] = Field(default=None, description="Report to be sent (delivery stage)")
    deliver_to: Optional[str] = None


class ApprovalDecision(BaseModel):
    """Body of POST /tasks/{job_id}/approve. It is the value interrupt() returns when the graph resumes."""

    approved: bool
    reviewer: str = Field(min_length=1, max_length=100, description="Who decides (kept for auditing)")
    comment: Optional[str] = Field(default=None, max_length=1000)


class ApprovalRecord(ApprovalDecision):
    stage: Literal["execution", "delivery"]
    decided_at: str = Field(default_factory=now_iso)


class ProductSummary(BaseModel):
    product: str
    avg_rating: Optional[float] = None
    sentiment_score: Optional[float] = None
    avg_price: Optional[float] = None
    best_store: Optional[str] = None


class TaskResult(BaseModel):
    final_answer: Optional[str] = None
    flow: list[str] = Field(default=[], description="Agents chosen by the supervisor, in order")
    products: list[ProductSummary] = []
    delivered_to: Optional[str] = None


class JobView(BaseModel):
    """Body of GET /tasks/{job_id}: the job record stored in Redis."""

    job_id: str
    status: JobStatus
    task: str
    deliver_to: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    run_seconds: float = Field(default=0, description="Time spent running the graph (all attempts, no waits)")
    risk: Optional[RiskAssessment] = None
    approval_request: Optional[ApprovalRequest] = None
    approvals: list[ApprovalRecord] = []
    result: Optional[TaskResult] = None
    error: Optional[str] = None
    trace_ids: list[str] = Field(default=[], description="Phoenix trace of each run (start and every resume)")
