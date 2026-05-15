from __future__ import annotations

from pydantic import BaseModel, Field

from agent.guard import CypherGuard
from agent.tools.base import ToolBase


class GraphQueryInput(BaseModel):
    cypher_query: str
    params: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int = 2000


class GraphQueryOutput(BaseModel):
    rows: list[dict] = Field(default_factory=list)
    row_count: int = 0
    non_empty_result: bool = False
    normalized_cypher: str
    explain_checked: bool = False


class GraphQueryTool(ToolBase):
    name = "graph_query_tool"
    description = "Execute a read-only Cypher query."
    input_model = GraphQueryInput
    output_model = GraphQueryOutput
    failure_modes = ("unsafe_query_blocked", "query_timeout", "query_validation_failed", "tool_error")
    latency_budget_ms = 2000

    def __init__(self, resources, *, enable_explain: bool = True):
        self.resources = resources
        self.guard = CypherGuard(graph_client_factory=resources.get_writer, enable_explain=enable_explain)

    def run(self, payload: GraphQueryInput) -> GraphQueryOutput:
        writer = self.resources.get_writer()
        validation = self.guard.validate(payload.cypher_query, payload.params)
        if not validation.ok:
            exc = RuntimeError(validation.error or "Cypher validation failed.")
            exc.failure_mode = validation.failure_mode or "tool_error"
            exc.validation_stage = validation.stage
            raise exc

        try:
            rows = writer.query(
                validation.normalized_cypher,
                timeout=payload.timeout_ms / 1000 if payload.timeout_ms else None,
                **payload.params,
            )
        except Exception as exc:
            wrapped = RuntimeError(str(exc))
            wrapped.failure_mode = "query_timeout" if "timeout" in str(exc).lower() else "tool_error"
            raise wrapped from exc
        return GraphQueryOutput(
            rows=rows,
            row_count=len(rows),
            non_empty_result=bool(rows),
            normalized_cypher=validation.normalized_cypher,
            explain_checked=validation.explain_checked,
        )
