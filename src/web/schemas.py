from typing import Literal

from pydantic import BaseModel, Field


class Question(BaseModel):
    message: str
    session_id: str | None = None


class Answer(BaseModel):
    message: str
    session_id: str


class AgentTurnRequest(BaseModel):
    message: str
    session_id: str | None = None


class PlanSummaryItem(BaseModel):
    tool: str


class AgentResponse(BaseModel):
    answer: str
    session_id: str
    trace_id: str
    plan_summary: list[PlanSummaryItem] = Field(default_factory=list)
    latency_ms: int = 0
    fallback_used: str | None = None
    trace: dict | None = None


class ReplayRequest(BaseModel):
    trace_ids: list[str] = Field(default_factory=list)


class RecommendationItemView(BaseModel):
    sku_id: int
    spu_id: int
    sku_name: str
    spu_name: str
    brand: str
    price: float
    reason: str
    default_img: str = ""
    storage_options: list[str] = Field(default_factory=list)


class DialogueStateView(BaseModel):
    domain: str
    intent: str
    filled_slots: dict = Field(default_factory=dict)
    pending_slots: list[str] = Field(default_factory=list)
    suggested_budget_min: int | None = None


class DialogueTurnRequest(BaseModel):
    message: str
    session_id: str | None = None


class DialogueTurnResponse(BaseModel):
    session_id: str
    message: str
    mode: Literal["dialogue", "qa_fallback"]
    action: Literal["ask_slot", "recommend", "compare", "fallback_qa", "reset"]
    state: DialogueStateView
    recommendations: list[RecommendationItemView] = Field(default_factory=list)
