from __future__ import annotations

from pydantic import BaseModel, Field

from agent.tools.base import ToolBase


class ProductCompareInput(BaseModel):
    spu_ids: list[int] = Field(default_factory=list)
    use_case: str | None = None


class ProductCompareOutput(BaseModel):
    comparison: str


class ProductCompareTool(ToolBase):
    name = "product_compare_tool"
    description = "Compare two product groups."
    input_model = ProductCompareInput
    output_model = ProductCompareOutput
    failure_modes = ("tool_error",)
    latency_budget_ms = 800

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, payload: ProductCompareInput) -> ProductCompareOutput:
        return ProductCompareOutput(
            comparison=self.retriever.compare(payload.spu_ids, payload.use_case),
        )
