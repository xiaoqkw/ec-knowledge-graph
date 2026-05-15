from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolBase(ABC):
    name: str = ""
    description: str = ""
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    failure_modes: tuple[str, ...] = ()
    latency_budget_ms: int = 1000

    @property
    def input_schema(self) -> dict[str, Any]:
        if hasattr(self.input_model, "model_json_schema"):
            return self.input_model.model_json_schema()
        return self.input_model.schema()

    @property
    def output_schema(self) -> dict[str, Any]:
        if hasattr(self.output_model, "model_json_schema"):
            return self.output_model.model_json_schema()
        return self.output_model.schema()

    @abstractmethod
    def run(self, payload: BaseModel) -> BaseModel:
        raise NotImplementedError
