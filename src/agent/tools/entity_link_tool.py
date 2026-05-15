from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import ToolBase
from agent.types import EntityRef
from configuration.config import ENTITY_INDEX_CONFIG


class EntityLinkInput(BaseModel):
    entities: list[EntityRef] = Field(default_factory=list)
    mode: Literal["exact_match", "fulltext", "hybrid"] = "hybrid"
    top_k: int = 3


class EntityCandidate(BaseModel):
    name: str
    score: float | None = None


class EntityLinkOutput(BaseModel):
    aligned_entities: list[EntityRef] = Field(default_factory=list)
    candidates_by_param: dict[str, list[EntityCandidate]] = Field(default_factory=dict)
    used_mode: str


class EntityLinkTool(ToolBase):
    name = "entity_link_tool"
    description = "Align entities to graph node names."
    input_model = EntityLinkInput
    output_model = EntityLinkOutput
    failure_modes = ("unsupported_label", "tool_error")
    latency_budget_ms = 1500

    def __init__(self, resources):
        self.resources = resources

    def run(self, payload: EntityLinkInput) -> EntityLinkOutput:
        graph = self.resources.get_graph()
        aligned_entities = []
        candidates_by_param: dict[str, list[EntityCandidate]] = {}

        for item in payload.entities:
            if item.label not in ENTITY_INDEX_CONFIG:
                aligned_entities.append(item.copy(update={"matched": False}))
                candidates_by_param[item.param_name] = []
                continue

            candidates = self._search_entities(
                graph=graph,
                label=item.label,
                query=item.entity,
                mode=payload.mode,
                k=payload.top_k,
            )
            candidates_by_param[item.param_name] = candidates
            if not candidates:
                aligned_entities.append(item.copy(update={"matched": False}))
                continue

            score_gap = None
            if len(candidates) >= 2 and candidates[0].score is not None and candidates[1].score is not None:
                score_gap = float(candidates[0].score - candidates[1].score)
            aligned_entities.append(
                item.copy(
                    update={
                        "entity": candidates[0].name,
                        "matched": True,
                        "score_gap": score_gap,
                    }
                )
            )

        return EntityLinkOutput(
            aligned_entities=aligned_entities,
            candidates_by_param=candidates_by_param,
            used_mode=payload.mode,
        )

    def _search_entities(self, *, graph, label: str, query: str, mode: str, k: int) -> list[EntityCandidate]:
        if mode == "exact_match":
            rows = graph.query(
                f"""
                MATCH (n:{label} {{name: $query}})
                RETURN n.name AS name
                """,
                params={"query": query},
            )
            return self._dedupe([EntityCandidate(name=row["name"]) for row in rows if row.get("name")])

        if mode == "fulltext":
            index_info = ENTITY_INDEX_CONFIG[label]
            rows = graph.query(
                """
                CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
                RETURN node.name AS name, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                params={
                    "index_name": index_info["fulltext_index"],
                    "query": query,
                    "limit": k,
                },
            )
            return self._dedupe(
                [
                    EntityCandidate(name=row["name"], score=float(row["score"]) if row.get("score") is not None else None)
                    for row in rows
                    if row.get("name")
                ]
            )

        vector = self.resources.get_vector(label)
        if hasattr(vector, "similarity_search_with_score"):
            results = vector.similarity_search_with_score(query, k=k)
            return self._dedupe(
                [
                    EntityCandidate(name=item.page_content, score=float(score))
                    for item, score in results
                    if getattr(item, "page_content", "")
                ]
            )
        results = vector.similarity_search(query, k=k)
        return self._dedupe(
            [EntityCandidate(name=item.page_content, score=None) for item in results if getattr(item, "page_content", "")]
        )

    @staticmethod
    def _dedupe(candidates: list[EntityCandidate]) -> list[EntityCandidate]:
        names = []
        seen = set()
        for candidate in candidates:
            if not candidate.name or candidate.name in seen:
                continue
            seen.add(candidate.name)
            names.append(candidate)
        return names
