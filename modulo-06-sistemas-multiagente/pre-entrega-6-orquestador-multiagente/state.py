"""Shared state of the orchestrator: what each agent contributed, validated with Pydantic.

Ownership (each node writes ONLY its keys; the guard in graph.py enforces it):
    supervisor   -> next_agent, instruction, step, decisions
    researcher   -> research, contributions
    analyst      -> analysis, contributions
    synthesizer  -> final_answer, contributions
Every node may also append to `messages` (the visible conversation).
"""

import operator
from datetime import datetime
from typing import Annotated, Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

AgentName = Literal["researcher", "analyst", "synthesizer"]
Route = Literal["researcher", "analyst", "synthesizer", "FINISH"]


# ---------------------------------------------------------------------------
# Researcher artifact
# ---------------------------------------------------------------------------
class Review(BaseModel):
    rating: int = Field(ge=1, le=5)
    text: str = Field(min_length=1)
    source: str
    date: str


class PriceQuote(BaseModel):
    store: str
    price: int = Field(gt=0)


class ProductFindings(BaseModel):
    product: str
    found: bool
    category: Optional[str] = None
    reviews: list[Review] = []
    prices: list[PriceQuote] = []
    suggestions: list[str] = []      # alternatives when the product was not found


class ResearchFindings(BaseModel):
    products: list[ProductFindings]
    summary: str                     # the researcher's own summary (LLM text)

    def get(self, product: str) -> Optional[ProductFindings]:
        return next((p for p in self.products if p.product == product), None)


# ---------------------------------------------------------------------------
# Analyst artifact
# ---------------------------------------------------------------------------
class ProductAnalysis(BaseModel):
    product: str
    review_count: int = Field(ge=0)
    avg_rating: Optional[float] = Field(default=None, ge=1, le=5)
    sentiment_score: Optional[float] = Field(default=None, ge=-1, le=1)   # -1 very negative, +1 very positive
    positive_pct: Optional[float] = Field(default=None, ge=0, le=100)
    negative_pct: Optional[float] = Field(default=None, ge=0, le=100)
    avg_price: Optional[float] = Field(default=None, gt=0)
    min_price: Optional[int] = Field(default=None, gt=0)
    best_store: Optional[str] = None
    pros: list[str] = []
    cons: list[str] = []


class AnalysisResult(BaseModel):
    products: list[ProductAnalysis]
    summary: str                     # the analyst's interpretation (LLM text)

    def get(self, product: str) -> Optional[ProductAnalysis]:
        return next((p for p in self.products if p.product == product), None)


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------
class Contribution(BaseModel):
    """What an agent contributed in one turn: the answer to "who said what"."""

    agent: AgentName
    round: int = Field(ge=1)                 # 1st, 2nd... time this agent worked (refinements)
    instruction: str                         # what the supervisor asked
    tools_used: list[str] = []               # e.g. "search_reviews(SoundMax Pro)"
    summary: str
    error: Optional[str] = None
    at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class SupervisorDecision(BaseModel):
    step: int
    next_agent: Route
    instruction: str
    reason: str
    rubric: dict[str, bool]                  # rubric state when the decision was made
    overridden: bool = False                 # True when a guard replaced the LLM's choice


class OrchestratorState(MessagesState):
    task: str                                                        # the user's request
    next_agent: Route                                                # supervisor's routing decision
    instruction: str                                                 # task for the chosen specialist
    step: int                                                        # supervisor turns so far
    research: Optional[ResearchFindings]
    analysis: Optional[AnalysisResult]
    final_answer: Optional[str]
    # Append-only logs (reducer operator.add): nodes add entries, never overwrite each other's
    contributions: Annotated[list[Contribution], operator.add]
    decisions: Annotated[list[SupervisorDecision], operator.add]


def rounds_of(state: OrchestratorState, agent: AgentName) -> int:
    return sum(c.agent == agent for c in state.get("contributions", []))
