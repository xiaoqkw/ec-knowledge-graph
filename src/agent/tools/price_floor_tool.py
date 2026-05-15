from __future__ import annotations

from pydantic import BaseModel

from agent.tools.base import ToolBase


class PriceFloorInput(BaseModel):
    brand: str | None = None


class PriceFloorOutput(BaseModel):
    min_price: int | None = None


class PriceFloorTool(ToolBase):
    name = "price_floor_tool"
    description = "Get current minimum sale price for the brand."
    input_model = PriceFloorInput
    output_model = PriceFloorOutput
    failure_modes = ("tool_error",)
    latency_budget_ms = 500

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, payload: PriceFloorInput) -> PriceFloorOutput:
        return PriceFloorOutput(min_price=self.retriever.get_min_price(payload.brand))
