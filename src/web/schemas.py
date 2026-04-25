from typing import Literal

from pydantic import BaseModel, Field


class Question(BaseModel):
    message: str
    session_id: str | None = None


class Answer(BaseModel):
    message: str
    session_id: str


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
