from __future__ import annotations

from pydantic import BaseModel, Field

from agent.tools.base import ToolBase
from dialogue.types import RecommendationItem


class ProductSearchInput(BaseModel):
    slots: dict = Field(default_factory=dict)
    limit: int = 3


class ProductSearchOutput(BaseModel):
    recommendations: list[dict] = Field(default_factory=list)


class ProductSearchTool(ToolBase):
    name = "product_search_tool"
    description = "Search products for dialogue recommendations."
    input_model = ProductSearchInput
    output_model = ProductSearchOutput
    failure_modes = ("tool_error",)
    latency_budget_ms = 800

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, payload: ProductSearchInput) -> ProductSearchOutput:
        items = self.retriever.search(payload.slots, limit=payload.limit)
        return ProductSearchOutput(recommendations=[self._to_dict(item) for item in items])

    @staticmethod
    def _to_dict(item: RecommendationItem) -> dict:
        return item.to_dict()
