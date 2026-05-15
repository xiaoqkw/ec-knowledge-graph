from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EntityRef(BaseModel):
    param_name: str
    entity: str
    label: str
    matched: bool = True
    score_gap: float | None = None


class ToolPlan(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_name: str
    input_payload: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    output_payload: dict[str, Any] = Field(default_factory=dict)
    failure_mode: str | None = None
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    request_id: str
    session_id: str
    user_query: str
    intent: str
    plan: list[ToolPlan] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    failure_stage: str | None = None
    failure_tags: list[str] = Field(default_factory=list)
    quality_signals: list[str] = Field(default_factory=list)
    quality_signal_rules: dict[str, str] = Field(default_factory=dict)
    fallback_used: str | None = None
    latency_breakdown: dict[str, int] = Field(default_factory=dict)
    final_answer: str = ""
    total_latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunResult(BaseModel):
    answer: str
    session_id: str
    trace: ExecutionTrace


class ValidationResult(BaseModel):
    ok: bool
    stage: str | None = None
    error: str | None = None
    failure_mode: str | None = None
    normalized_cypher: str = ""
    explain_checked: bool = False
