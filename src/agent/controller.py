from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from agent.runtime import AgentRuntime, dump_model, parse_model, repair_json
from agent.tools import AnswerTool, EntityLinkTool, GraphQueryTool, PriceFloorTool, ProductCompareTool, ProductSearchTool
from agent.types import AgentRunResult, EntityRef, EvidenceItem, ExecutionTrace, ToolCallRecord, ToolPlan
from configuration.config import ENTITY_INDEX_CONFIG
from web.memory import CachedAlignedEntity, FailureMemory, QASession


TEMPLATE_CYPHERS = {
    "brand_products": """
        MATCH (tm:Trademark {name: $param_0})<-[:Belong]-(spu:SPU)
        RETURN DISTINCT spu.name AS name
        ORDER BY name
        LIMIT 20
    """,
    "category_brands": """
        MATCH (spu:SPU)-[:Belong]->(:Category3 {name: $param_0})
        MATCH (spu)-[:Belong]->(tm:Trademark)
        RETURN DISTINCT tm.name AS name
        ORDER BY name
        LIMIT 20
    """,
    "product_skus": """
        MATCH (sku:SKU)-[:Belong]->(:SPU {name: $param_0})
        RETURN DISTINCT sku.name AS name
        ORDER BY name
        LIMIT 20
    """,
}
SUPPORTED_KGQA_TOOLS = {"entity_link_tool", "graph_query_tool", "answer_tool"}
SCALAR_EVIDENCE_TYPES = (str, int, float, bool)
TEXT_EVIDENCE_LIMIT = 2
GRAPH_EVIDENCE_LIMIT = 3
TEXT_EVIDENCE_TIMEOUT_SECONDS = 0.5
RETURN_ALIAS_PATTERN = re.compile(
    r"(?i)^\s*(?:distinct\s+)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?\s*$"
)
MATCH_LABEL_PATTERN = re.compile(r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)")
CAMEL_TO_SNAKE_PATTERN_1 = re.compile(r"(.)([A-Z][a-z]+)")
CAMEL_TO_SNAKE_PATTERN_2 = re.compile(r"([a-z0-9])([A-Z])")
ENTITY_ID_LOOKUP_PATTERN = re.compile(
    r"(?i)\b(?:id|elementid)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*=\s*\$([A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass
class GraphEvidenceCandidate:
    row_index: int
    field_name: str
    value: str
    node_label: str | None
    entity_id: str | None
    resolved_var: str | None
    resolved_property: str | None
    lookup_id: Any = None


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _answer_references_result(answer: str, query_result: list[dict]) -> bool:
    lowered = answer.lower()
    for row in query_result:
        for value in row.values():
            if isinstance(value, str) and value and value.lower() in lowered:
                return True
    return False


PLANNER_FAILURE_TAGS = {
    "parse_failure",
    "plan_schema_invalid",
    "entity_missing",
    "unsafe_query_blocked",
    "query_timeout",
    "query_empty",
}


class SchemaMappingCache:
    def __init__(self):
        self._cache: dict[str, str] = {}

    def get(self, label: str, raw_entity: str) -> str | None:
        return self._cache.get(AgentController._normalize_cache_key(label, raw_entity))

    def set(self, label: str, raw_entity: str, canonical_name: str) -> None:
        self._cache[AgentController._normalize_cache_key(label, raw_entity)] = canonical_name


class EntityAlignmentCache:
    def __init__(self, *, match_strategy: str = "exact"):
        self.match_strategy = match_strategy
        self._cache: dict[str, CachedAlignedEntity] = {}

    def lookup(self, label: str, raw_entity: str) -> CachedAlignedEntity | None:
        if self.match_strategy != "exact":
            raise ValueError(f"Unsupported entity cache match strategy: {self.match_strategy}")
        return self._cache.get(AgentController._normalize_cache_key(label, raw_entity))

    def upsert(self, item: CachedAlignedEntity) -> None:
        self._cache[AgentController._normalize_cache_key(item.label, item.raw_entity)] = item


class AgentController:
    def __init__(self, runtime: AgentRuntime, *, retriever=None, enable_explain: bool = True):
        self.runtime = runtime
        self.retriever = retriever
        self.schema_mapping_cache = SchemaMappingCache()
        self.runtime.register_tool(EntityLinkTool(runtime.resources))
        self.runtime.register_tool(GraphQueryTool(runtime.resources, enable_explain=enable_explain))
        self.runtime.register_tool(AnswerTool(runtime.resources))
        if retriever is not None:
            self.runtime.register_tool(ProductSearchTool(retriever))
            self.runtime.register_tool(ProductCompareTool(retriever))
            self.runtime.register_tool(PriceFloorTool(retriever))

    def run_kgqa(
        self,
        question: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        align_entities: bool = True,
        session: QASession | None = None,
    ) -> AgentRunResult:
        trace = self.runtime.build_trace(session_id=session_id, user_query=question, intent="kgqa")
        planner = KGQAPlanner(self.runtime.resources)
        memory_hint = self._build_memory_hint(session, trace)

        try:
            plan_result = planner.plan(question, history=history, memory_hint=memory_hint)
        except Exception as exc:
            trace.metadata["planner_exception"] = str(exc)
            trace.fallback_used = "template_cypher"
            plan_result = self._template_plan(question)
            self._set_failure(trace, "plan_schema_invalid", "plan")

        trace.metadata["planner_raw_output"] = plan_result["raw_content"]
        trace.metadata["planner_repaired_output"] = plan_result["repaired_content"]
        trace.metadata["parsed_payload"] = dump_model(plan_result["payload"]) if plan_result["payload"] is not None else None
        trace.metadata["planner_parse_success_raw"] = plan_result["parse_success_raw"]
        trace.metadata["planner_parse_success_repaired"] = plan_result["parse_success_repaired"]
        trace.metadata["planner_invalid_tool_names"] = list(plan_result.get("invalid_tool_names", []))

        if not plan_result["parse_success_raw"]:
            self._set_failure(trace, "parse_failure", "parse")
        if not plan_result["plan_valid"]:
            self._set_failure(trace, "plan_schema_invalid", "plan")

        payload = plan_result["payload"] if isinstance(plan_result["payload"], dict) else {}
        cypher_query = str(payload.get("cypher_query", "")).strip()
        raw_entities = [parse_model(EntityRef, item) for item in payload.get("entities_to_align", [])]
        if self._uses_entity_id_lookup(cypher_query, raw_entities):
            trace.metadata["planner_entity_id_lookup_rejected"] = True
            self._set_failure(trace, "plan_schema_invalid", "plan")
            trace.fallback_used = "template_cypher"
            fallback_plan = self._template_plan(question)
            payload = fallback_plan["payload"] if isinstance(fallback_plan["payload"], dict) else {}
            cypher_query = str(payload.get("cypher_query", "")).strip()
            raw_entities = [parse_model(EntityRef, item) for item in payload.get("entities_to_align", [])]
        planner_plan = [parse_model(ToolPlan, item) for item in payload.get("tool_plan", [])]
        trace.plan = planner_plan
        if not plan_result["plan_valid"] or not trace.plan:
            trace.plan = self._default_tool_plan(align_entities=align_entities)
            if trace.fallback_used is None:
                trace.fallback_used = "default_tool_plan"
        elif trace.fallback_used == "template_cypher":
            trace.plan = planner_plan

        plan_by_name = {item.tool_name: item.arguments for item in trace.plan}

        aligned_entities = raw_entities
        candidates_by_param: dict[str, list[dict]] = {}
        entity_link_args = plan_by_name.get("entity_link_tool", {})
        should_align = align_entities and "entity_link_tool" in plan_by_name
        if raw_entities and should_align:
            cache_lookup = self._resolve_entity_cache_hits(session, raw_entities)
            aligned_entities = cache_lookup["aligned_entities"]
            candidates_by_param = cache_lookup["candidates_by_param"]
            missing_entities = cache_lookup["missing_entities"]
            trace.metadata["task_memory_hit_count"] = cache_lookup["hit_count"]
            trace.metadata["task_memory_miss_count"] = len(missing_entities)
            trace.metadata["task_memory_hit_keys"] = cache_lookup["hit_keys"]
            trace.metadata["task_memory_skipped_entity_link"] = bool(raw_entities and not missing_entities)

            if missing_entities:
                entity_record = self.runtime.invoke_tool(
                    trace,
                    "entity_link_tool",
                    {
                        "entities": [dump_model(item) for item in missing_entities],
                        "mode": self._resolve_entity_link_mode(entity_link_args, trace),
                        "top_k": self._coerce_int_argument(
                            entity_link_args.get("top_k"),
                            default=3,
                            minimum=1,
                            maximum=10,
                            trace=trace,
                            argument_name="entity_link_tool.top_k",
                        ),
                    },
                )
                if entity_record.ok:
                    missing_aligned = [parse_model(EntityRef, item) for item in entity_record.output_payload["aligned_entities"]]
                    missing_candidates = entity_record.output_payload.get("candidates_by_param", {})
                    candidates_by_param.update(missing_candidates)
                    aligned_entities = self._merge_aligned_entities(raw_entities, cache_lookup["cached_entities"], missing_aligned)
                    self._mark_entity_signal(trace, raw_entities, aligned_entities, candidates_by_param)
                else:
                    self._set_failure(trace, "entity_missing", "entity_link")
            else:
                trace.tool_calls.append(
                    ToolCallRecord(
                        tool_name="entity_link_tool",
                        input_payload={"entities": [dump_model(item) for item in raw_entities]},
                        ok=True,
                        output_payload={
                            "aligned_entities": [dump_model(item) for item in aligned_entities],
                            "candidates_by_param": candidates_by_param,
                            "used_mode": "session_cache",
                        },
                        cache_hit=True,
                        latency_ms=0,
                    )
                )
                self._mark_entity_signal(trace, raw_entities, aligned_entities, candidates_by_param)
        else:
            trace.metadata["task_memory_hit_count"] = 0
            trace.metadata["task_memory_miss_count"] = 0
            trace.metadata["task_memory_hit_keys"] = []
            trace.metadata["task_memory_skipped_entity_link"] = False

        executed_params = {
            item.param_name: item.entity
            for item in aligned_entities
            if item.param_name and item.entity is not None
        }

        rows = []
        query_args = plan_by_name.get("graph_query_tool", {})
        if cypher_query and "graph_query_tool" in plan_by_name:
            query_record = self.runtime.invoke_tool(
                trace,
                "graph_query_tool",
                {
                    "cypher_query": cypher_query,
                    "params": executed_params,
                    "timeout_ms": self._coerce_int_argument(
                        query_args.get("timeout_ms"),
                        default=2000,
                        minimum=100,
                        maximum=10000,
                        trace=trace,
                        argument_name="graph_query_tool.timeout_ms",
                    ),
                },
            )
            if query_record.ok:
                rows = query_record.output_payload["rows"]
                if not rows:
                    self._set_failure(trace, "query_empty", "query")
            else:
                if query_record.failure_mode == "unsafe_query_blocked":
                    self._set_failure(trace, "unsafe_query_blocked", "safety")
                elif query_record.failure_mode == "query_validation_failed":
                    self._set_failure(trace, "plan_schema_invalid", "query")
                elif query_record.failure_mode == "query_timeout":
                    self._set_failure(trace, "query_timeout", "query")
                else:
                    self._set_failure(trace, "query_timeout", "query")
        else:
            self._set_failure(trace, "plan_schema_invalid", "plan")

        if "answer_tool" in plan_by_name:
            evidence = self._build_evidence(cypher_query=cypher_query, rows=rows)
            trace.evidence = evidence
            answer_record = self.runtime.invoke_tool(
                trace,
                "answer_tool",
                {
                    "question": question,
                    "query_result": rows,
                    "history": history or [],
                    "evidence": [dump_model(item) for item in evidence],
                },
            )
            if answer_record.ok:
                answer = answer_record.output_payload.get("answer", "")
                self._mark_answer_signal(trace, answer, rows, bool(answer_record.output_payload.get("llm_used")))
            else:
                answer = "当前图谱中没有找到相关信息。"
        else:
            trace.evidence = self._build_evidence(cypher_query=cypher_query, rows=rows)
            answer = json.dumps(rows, ensure_ascii=False) if rows else "当前图谱中没有找到相关信息。"

        trace.metadata["cypher_query"] = cypher_query
        trace.metadata["entities_to_align"] = [dump_model(item) for item in raw_entities]
        trace.metadata["aligned_entities"] = [dump_model(item) for item in aligned_entities]
        trace.metadata["executed_params"] = executed_params
        trace.metadata["query_result"] = rows
        trace.metadata["confirmed_entity_cache_updates"] = self._build_confirmed_entity_cache_updates(
            session=session,
            raw_entities=raw_entities,
            aligned_entities=aligned_entities,
            executed_params=executed_params,
            cypher_query=cypher_query,
            rows=rows,
            trace=trace,
            entity_link_mode=entity_link_args.get("mode"),
        )
        trace.metadata["session_failure_memory"] = self._build_session_failure_memory(trace, cypher_query, executed_params)
        return self.runtime.finalize_trace(trace, answer=answer)

    @staticmethod
    def _default_tool_plan(*, align_entities: bool) -> list[ToolPlan]:
        plan = []
        if align_entities:
            plan.append(ToolPlan(tool_name="entity_link_tool", arguments={"mode": "hybrid", "top_k": 3}))
        plan.append(ToolPlan(tool_name="graph_query_tool", arguments={"timeout_ms": 2000}))
        plan.append(ToolPlan(tool_name="answer_tool", arguments={}))
        return plan

    def _template_plan(self, question: str) -> dict:
        normalized = question.lower()
        if any(keyword in normalized for keyword in ("哪些产品", "有什么产品", "有哪些产")):
            payload = {
                "tool_plan": [
                    {"tool_name": "entity_link_tool", "arguments": {"mode": "fulltext", "top_k": 3}},
                    {"tool_name": "graph_query_tool", "arguments": {"timeout_ms": 2000}},
                    {"tool_name": "answer_tool", "arguments": {}},
                ],
                "cypher_query": TEMPLATE_CYPHERS["brand_products"].strip(),
                "entities_to_align": [{"param_name": "param_0", "entity": self._extract_subject(question), "label": "Trademark"}],
            }
            return self._planned_payload(payload)
        if any(keyword in normalized for keyword in ("哪些品牌", "什么品牌")):
            payload = {
                "tool_plan": [
                    {"tool_name": "entity_link_tool", "arguments": {"mode": "fulltext", "top_k": 3}},
                    {"tool_name": "graph_query_tool", "arguments": {"timeout_ms": 2000}},
                    {"tool_name": "answer_tool", "arguments": {}},
                ],
                "cypher_query": TEMPLATE_CYPHERS["category_brands"].strip(),
                "entities_to_align": [{"param_name": "param_0", "entity": self._extract_subject(question), "label": "Category3"}],
            }
            return self._planned_payload(payload)
        if "sku" in normalized:
            payload = {
                "tool_plan": [
                    {"tool_name": "entity_link_tool", "arguments": {"mode": "hybrid", "top_k": 3}},
                    {"tool_name": "graph_query_tool", "arguments": {"timeout_ms": 2000}},
                    {"tool_name": "answer_tool", "arguments": {}},
                ],
                "cypher_query": TEMPLATE_CYPHERS["product_skus"].strip(),
                "entities_to_align": [{"param_name": "param_0", "entity": self._extract_subject(question), "label": "SPU"}],
            }
            return self._planned_payload(payload)
        return self._planned_payload(
            {
                "tool_plan": [{"tool_name": "answer_tool", "arguments": {}}],
                "cypher_query": "",
                "entities_to_align": [],
            },
            plan_valid=False,
        )

    @staticmethod
    def _extract_subject(question: str) -> str:
        text = question.strip().rstrip("？?。")
        for separator in ("都有哪些产品", "有哪些产品", "有什么产品", "哪些品牌", "什么品牌", "有哪些sku", "哪些sku", "哪些 SKU", "哪些sku"):
            if separator in text:
                return text.split(separator, 1)[0].strip() or text
        return text

    @staticmethod
    def _planned_payload(payload: dict, *, plan_valid: bool = True) -> dict:
        return {
            "raw_content": "",
            "repaired_content": "",
            "parse_success_raw": True,
            "parse_success_repaired": True,
            "payload": payload,
            "parse_error": None,
            "plan_valid": plan_valid,
        }

    @staticmethod
    def _resolve_entity_link_mode(arguments: dict, trace: ExecutionTrace) -> str:
        mode = arguments.get("mode", "hybrid")
        if mode in {"exact_match", "fulltext", "hybrid"}:
            return mode
        AgentController._mark_argument_fallback(
            trace,
            argument_name="entity_link_tool.mode",
            raw_value=mode,
            fallback_value="hybrid",
        )
        return "hybrid"

    @staticmethod
    def _coerce_int_argument(
        raw_value,
        *,
        default: int,
        minimum: int,
        maximum: int,
        trace: ExecutionTrace,
        argument_name: str,
    ) -> int:
        if raw_value is None:
            return default
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            AgentController._mark_argument_fallback(
                trace,
                argument_name=argument_name,
                raw_value=raw_value,
                fallback_value=default,
            )
            return default

        if value < minimum or value > maximum:
            AgentController._mark_argument_fallback(
                trace,
                argument_name=argument_name,
                raw_value=raw_value,
                fallback_value=max(minimum, min(value, maximum)),
            )
            value = max(minimum, min(value, maximum))
        return value

    @staticmethod
    def _mark_argument_fallback(
        trace: ExecutionTrace,
        *,
        argument_name: str,
        raw_value,
        fallback_value,
    ) -> None:
        AgentController._set_failure(trace, "plan_schema_invalid", "plan")
        if trace.fallback_used is None:
            trace.fallback_used = "default_tool_arguments"
        fallbacks = trace.metadata.setdefault("plan_argument_fallbacks", [])
        fallbacks.append(
            {
                "argument_name": argument_name,
                "raw_value": raw_value,
                "fallback_value": fallback_value,
            }
        )

    @staticmethod
    def _set_failure(trace: ExecutionTrace, tag: str, stage: str) -> None:
        _append_unique(trace.failure_tags, tag)
        if trace.failure_stage is None:
            trace.failure_stage = stage

    def _mark_entity_signal(
        self,
        trace: ExecutionTrace,
        raw_entities: list[EntityRef],
        aligned_entities: list[EntityRef],
        candidates_by_param: dict[str, list[dict]],
    ) -> None:
        for raw_item, aligned_item in zip(raw_entities, aligned_entities):
            reason = self._entity_signal_reason(raw_item, aligned_item, candidates_by_param.get(raw_item.param_name, []))
            if reason is None:
                continue
            _append_unique(trace.quality_signals, "entity_misaligned")
            trace.quality_signal_rules["entity_misaligned"] = reason
            self._set_failure(trace, "entity_misaligned", "entity_link")
            return

    def _mark_answer_signal(self, trace: ExecutionTrace, answer: str, query_result: list[dict], llm_used: bool) -> None:
        weak, reason = self._is_weak_answer(answer, query_result, llm_used)
        if not weak or reason is None:
            return
        _append_unique(trace.quality_signals, "answer_weak")
        trace.quality_signal_rules["answer_weak"] = reason
        self._set_failure(trace, "answer_weak", "answer")

    @staticmethod
    def _entity_signal_reason(raw_item: EntityRef, aligned_item: EntityRef, candidates: list[dict]) -> str | None:
        if raw_item.label not in ENTITY_INDEX_CONFIG:
            return "unsupported_label"
        if raw_item.entity == aligned_item.entity and not candidates:
            return "no_candidate_passthrough"
        if len(candidates) >= 2:
            first = candidates[0].get("score")
            second = candidates[1].get("score")
            if first is not None and second is not None and float(first) - float(second) < 0.1:
                return "score_gap_lt_0.1"
        return None

    @staticmethod
    def _is_weak_answer(answer: str, query_result: list[dict], llm_used: bool) -> tuple[bool, str | None]:
        weak_patterns = ("当前图谱中没有找到", "无法回答", "请稍后再试")
        if not llm_used and query_result:
            return True, "json_dump_fallback"
        if len(answer.strip()) < 10:
            return True, "answer_too_short"
        if any(pattern in answer for pattern in weak_patterns):
            return True, "weak_template_pattern"
        if query_result and not _answer_references_result(answer, query_result):
            return True, "answer_not_grounded"
        return False, None

    @staticmethod
    def _normalize_cache_key(label: str, raw_entity: str) -> str:
        return f"{label}::{' '.join(str(raw_entity or '').strip().lower().split())}"

    def _build_memory_hint(self, session: QASession | None, trace: ExecutionTrace) -> str | None:
        if session is None:
            trace.metadata["task_memory_recent_failures_used"] = 0
            return None

        entity_lines = []
        for item in list(session.task_memory.entity_cache.values())[-3:]:
            entity_lines.append(f"{item.label}:{item.raw_entity}->{item.aligned_entity}")

        failures = [
            item
            for item in session.task_memory.recent_failures
            if item.primary_failure_tag in PLANNER_FAILURE_TAGS
        ][-2:]
        trace.metadata["task_memory_recent_failures_used"] = len(failures)
        failure_lines = [
            f"{item.primary_failure_tag}@{item.failure_stage or 'unknown'}:{item.cypher_excerpt}"
            for item in failures
        ]
        parts = []
        if entity_lines:
            parts.append("已确认实体对齐:\n" + "\n".join(entity_lines))
        if failure_lines:
            parts.append("近期失败摘要:\n" + "\n".join(failure_lines))
        return "\n\n".join(parts) if parts else None

    def _resolve_entity_cache_hits(self, session: QASession | None, raw_entities: list[EntityRef]) -> dict:
        if session is None:
            return {
                "aligned_entities": raw_entities,
                "cached_entities": {},
                "candidates_by_param": {},
                "missing_entities": raw_entities,
                "hit_count": 0,
                "hit_keys": [],
            }

        cached_entities: dict[str, EntityRef] = {}
        candidates_by_param: dict[str, list[dict]] = {}
        missing_entities: list[EntityRef] = []
        hit_keys: list[str] = []

        for item in raw_entities:
            key = self._normalize_cache_key(item.label, item.entity)
            cached = session.task_memory.entity_cache.lookup(item.label, item.entity)
            if cached is not None:
                cached_entities[item.param_name] = (
                    item.model_copy(
                        update={
                            "entity": cached.aligned_entity,
                            "matched": cached.matched,
                            "score_gap": cached.score_gap,
                        }
                    )
                    if hasattr(item, "model_copy")
                    else item.copy(
                        update={
                            "entity": cached.aligned_entity,
                            "matched": cached.matched,
                            "score_gap": cached.score_gap,
                        }
                    )
                )
                candidates_by_param[item.param_name] = [{"name": cached.aligned_entity, "score": None}]
                hit_keys.append(key)
                continue
            mapped = self.schema_mapping_cache.get(item.label, item.entity)
            if mapped is not None:
                cached_entities[item.param_name] = (
                    item.model_copy(
                        update={
                            "entity": mapped,
                            "matched": True,
                        }
                    )
                    if hasattr(item, "model_copy")
                    else item.copy(
                        update={
                            "entity": mapped,
                            "matched": True,
                        }
                    )
                )
                candidates_by_param[item.param_name] = [{"name": mapped, "score": None}]
                hit_keys.append(key)
                continue
            missing_entities.append(item)

        aligned_entities = []
        for item in raw_entities:
            aligned_entities.append(cached_entities.get(item.param_name, item))
        return {
            "aligned_entities": aligned_entities,
            "cached_entities": cached_entities,
            "candidates_by_param": candidates_by_param,
            "missing_entities": missing_entities,
            "hit_count": len(hit_keys),
            "hit_keys": hit_keys,
        }

    @staticmethod
    def _merge_aligned_entities(
        raw_entities: list[EntityRef],
        cached_entities: dict[str, EntityRef],
        missing_aligned: list[EntityRef],
    ) -> list[EntityRef]:
        aligned_by_param = {item.param_name: item for item in missing_aligned}
        merged = []
        for raw_item in raw_entities:
            merged.append(cached_entities.get(raw_item.param_name, aligned_by_param.get(raw_item.param_name, raw_item)))
        return merged

    def _build_confirmed_entity_cache_updates(
        self,
        *,
        session: QASession | None,
        raw_entities: list[EntityRef],
        aligned_entities: list[EntityRef],
        executed_params: dict[str, str],
        cypher_query: str,
        rows: list[dict],
        trace: ExecutionTrace,
        entity_link_mode,
    ) -> list[dict]:
        if session is None or not rows or len(raw_entities) != 1:
            return []
        if "entity_misaligned" in trace.quality_signals:
            return []

        raw_item = raw_entities[0]
        aligned_item = aligned_entities[0]
        if not aligned_item.matched:
            return []
        if raw_item.param_name not in executed_params:
            return []
        if f"${raw_item.param_name}" not in cypher_query:
            return []

        update = CachedAlignedEntity(
            label=raw_item.label,
            raw_entity=raw_item.entity,
            aligned_entity=aligned_item.entity,
            param_name=raw_item.param_name,
            matched=aligned_item.matched,
            score_gap=aligned_item.score_gap,
            source_mode=str(entity_link_mode or "hybrid"),
            updated_at=time.time(),
        )
        if raw_item.label in {"Trademark", "Category3"}:
            self.schema_mapping_cache.set(raw_item.label, raw_item.entity, aligned_item.entity)
        return [
            dump_model(
                update
            )
        ]

    @staticmethod
    def _build_session_failure_memory(trace: ExecutionTrace, cypher_query: str, executed_params: dict) -> dict | None:
        primary = next((tag for tag in trace.failure_tags if tag in PLANNER_FAILURE_TAGS), None)
        if primary is None:
            return None
        limited_params = {}
        for index, key in enumerate(executed_params):
            if index >= 3:
                break
            limited_params[key] = executed_params[key]
        return dump_model(
            FailureMemory(
                trace_id=trace.request_id,
                failure_stage=trace.failure_stage,
                primary_failure_tag=primary,
                cypher_excerpt=str(cypher_query or "")[:200],
                executed_params_excerpt=limited_params,
            )
        )

    def _build_evidence(self, *, cypher_query: str, rows: list[dict]) -> list[EvidenceItem]:
        if not rows:
            return []

        graph_candidates = self._build_graph_evidence_candidates(cypher_query=cypher_query, rows=rows)
        graph_evidence = [
            EvidenceItem(
                kind="graph",
                source="graph",
                node_label=item.node_label,
                field_name=item.field_name,
                value=item.value,
                entity_id=item.entity_id,
                text_excerpt=None,
            )
            for item in graph_candidates
        ]
        text_evidence = self._build_text_evidence(graph_candidates)
        return graph_evidence + text_evidence

    def _build_graph_evidence_candidates(self, *, cypher_query: str, rows: list[dict]) -> list[GraphEvidenceCandidate]:
        alias_map = self._parse_return_alias_map(cypher_query)
        variable_labels = self._parse_variable_labels(cypher_query)
        candidates: list[GraphEvidenceCandidate] = []
        seen = set()

        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue

            row_id_fields = {
                str(key): value
                for key, value in row.items()
                if self._is_id_field(key) and self._is_scalar_value(value)
            }

            for key, raw_value in row.items():
                raw_field_name = str(key)
                if self._is_id_field(raw_field_name):
                    continue
                if not self._is_scalar_value(raw_value):
                    continue
                value = str(raw_value)
                if not value.strip():
                    continue

                resolved = alias_map.get(raw_field_name)
                node_label = None
                resolved_var = None
                resolved_property = None
                if resolved is not None:
                    resolved_var = resolved["var"]
                    resolved_property = resolved["prop"]
                    node_label = variable_labels.get(resolved_var)
                field_name = self._normalize_field_name(raw_field_name)
                entity_id, lookup_id = self._resolve_entity_identity(
                    row=row,
                    raw_field_name=raw_field_name,
                    node_label=node_label,
                    resolved_property=resolved_property,
                    row_id_fields=row_id_fields,
                )
                candidate = GraphEvidenceCandidate(
                    row_index=row_index,
                    field_name=field_name,
                    value=value,
                    node_label=node_label,
                    entity_id=entity_id,
                    resolved_var=resolved_var,
                    resolved_property=resolved_property,
                    lookup_id=lookup_id,
                )
                dedupe_key = (candidate.field_name, candidate.value)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                candidates.append(candidate)
                if len(candidates) >= GRAPH_EVIDENCE_LIMIT:
                    return candidates
        return candidates

    def _build_text_evidence(self, graph_candidates: list[GraphEvidenceCandidate]) -> list[EvidenceItem]:
        supplement_candidates = []
        seen_entities = set()
        for item in graph_candidates:
            if item.node_label not in {"SPU", "SKU"}:
                continue
            if item.lookup_id is None and item.resolved_property != "name":
                continue
            entity_key = item.entity_id or f"{item.node_label}:{item.value}"
            if entity_key in seen_entities:
                continue
            if self._has_graph_description_for_same_var(item, graph_candidates):
                continue
            seen_entities.add(entity_key)
            supplement_candidates.append(item)
            if len(supplement_candidates) >= TEXT_EVIDENCE_LIMIT:
                break

        if not supplement_candidates:
            return []

        try:
            descriptions = self._lookup_spu_descriptions(supplement_candidates)
        except Exception:
            return []

        evidence = []
        seen_text_keys = set()
        for item in supplement_candidates:
            row = descriptions.get(item.entity_id or f"{item.node_label}:{item.value}")
            if row is None:
                continue
            entity_id = row.get("entity_id")
            dedupe_key = entity_id or f"{item.node_label}:{item.value}"
            if dedupe_key in seen_text_keys:
                continue
            seen_text_keys.add(dedupe_key)
            evidence.append(
                EvidenceItem(
                    kind="text",
                    source="spu_description",
                    node_label="SPU",
                    field_name="description",
                    value=str(row.get("spu_name", "")),
                    entity_id=str(entity_id) if entity_id is not None else None,
                    text_excerpt=str(row.get("description", ""))[:160] or None,
                )
            )
        return evidence

    def _lookup_spu_descriptions(self, candidates: list[GraphEvidenceCandidate]) -> dict[str, dict]:
        writer = self.runtime.resources.get_writer()
        results: dict[str, dict] = {}

        spu_ids = [item.lookup_id for item in candidates if item.node_label == "SPU" and item.lookup_id is not None]
        spu_names = [item.value for item in candidates if item.node_label == "SPU" and item.lookup_id is None]
        sku_ids = [item.lookup_id for item in candidates if item.node_label == "SKU" and item.lookup_id is not None]
        sku_names = [item.value for item in candidates if item.node_label == "SKU" and item.lookup_id is None]

        if spu_ids:
            rows = writer.query(
                """
                MATCH (spu:SPU)
                WHERE spu.id IN $ids
                  AND coalesce(spu.description, '') <> ''
                RETURN spu.id AS spu_id, spu.name AS spu_name, spu.description AS description
                """,
                timeout=TEXT_EVIDENCE_TIMEOUT_SECONDS,
                ids=spu_ids,
            )
            for row in rows:
                key = str(row.get("spu_id"))
                results[key] = {
                    "entity_id": key,
                    "spu_name": str(row.get("spu_name", "")),
                    "description": str(row.get("description", "")),
                }
        if spu_names:
            rows = writer.query(
                """
                MATCH (spu:SPU)
                WHERE spu.name IN $names
                  AND coalesce(spu.description, '') <> ''
                RETURN spu.id AS spu_id, spu.name AS spu_name, spu.description AS description
                """,
                timeout=TEXT_EVIDENCE_TIMEOUT_SECONDS,
                names=spu_names,
            )
            grouped = self._group_rows_by_name(rows, "spu_name")
            for name, items in grouped.items():
                if len(items) != 1:
                    continue
                row = items[0]
                results[f"SPU:{name}"] = {
                    "entity_id": str(row.get("spu_id")) if row.get("spu_id") is not None else None,
                    "spu_name": str(row.get("spu_name", "")),
                    "description": str(row.get("description", "")),
                }
        if sku_ids:
            rows = writer.query(
                """
                MATCH (sku:SKU)-[:Belong]->(spu:SPU)
                WHERE sku.id IN $ids
                  AND coalesce(spu.description, '') <> ''
                RETURN sku.id AS sku_id, spu.id AS spu_id, spu.name AS spu_name, spu.description AS description
                """,
                timeout=TEXT_EVIDENCE_TIMEOUT_SECONDS,
                ids=sku_ids,
            )
            for row in rows:
                key = str(row.get("sku_id"))
                results[key] = {
                    "entity_id": str(row.get("spu_id")) if row.get("spu_id") is not None else None,
                    "spu_name": str(row.get("spu_name", "")),
                    "description": str(row.get("description", "")),
                }
        if sku_names:
            rows = writer.query(
                """
                MATCH (sku:SKU)-[:Belong]->(spu:SPU)
                WHERE sku.name IN $names
                  AND coalesce(spu.description, '') <> ''
                RETURN sku.name AS sku_name, spu.id AS spu_id, spu.name AS spu_name, spu.description AS description
                """,
                timeout=TEXT_EVIDENCE_TIMEOUT_SECONDS,
                names=sku_names,
            )
            grouped = self._group_rows_by_name(rows, "sku_name")
            for name, items in grouped.items():
                if len(items) != 1:
                    continue
                row = items[0]
                results[f"SKU:{name}"] = {
                    "entity_id": str(row.get("spu_id")) if row.get("spu_id") is not None else None,
                    "spu_name": str(row.get("spu_name", "")),
                    "description": str(row.get("description", "")),
                }
        return results

    @staticmethod
    def _group_rows_by_name(rows: list[dict], field_name: str) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            key = str(row.get(field_name, ""))
            if not key:
                continue
            grouped.setdefault(key, []).append(row)
        return grouped

    @staticmethod
    def _has_graph_description_for_same_var(
        candidate: GraphEvidenceCandidate,
        graph_candidates: list[GraphEvidenceCandidate],
    ) -> bool:
        if not candidate.resolved_var:
            return False
        for item in graph_candidates:
            if item.row_index != candidate.row_index:
                continue
            if item.resolved_var != candidate.resolved_var:
                continue
            if item.resolved_property != "description":
                continue
            return True
        return False

    @staticmethod
    def _resolve_entity_identity(
        *,
        row: dict[str, Any],
        raw_field_name: str,
        node_label: str | None,
        resolved_property: str | None,
        row_id_fields: dict[str, Any],
    ) -> tuple[str | None, Any]:
        if node_label not in {"SPU", "SKU"}:
            return None, None
        if not row_id_fields and resolved_property != "name" and AgentController._paired_name_field(raw_field_name) is None:
            return None, None

        preferred_keys = []
        lowered = raw_field_name.lower()
        if node_label == "SKU":
            if lowered.startswith("sku"):
                preferred_keys.append("sku_id")
            preferred_keys.extend(["sku_id"])
        else:
            if lowered.startswith("product"):
                preferred_keys.append("product_id")
            if lowered.startswith("spu"):
                preferred_keys.append("spu_id")
            preferred_keys.extend(["product_id", "spu_id"])
        for key in preferred_keys:
            if key in row_id_fields:
                return str(row_id_fields[key]), row_id_fields[key]
        return None, None

    @staticmethod
    def _paired_name_field(field_name: str) -> str | None:
        lowered = field_name.lower()
        if lowered == "product_name":
            return field_name
        if lowered == "spu_name":
            return field_name
        if lowered == "sku_name":
            return field_name
        if lowered == "name":
            return field_name
        return None

    @staticmethod
    def _is_id_field(field_name: Any) -> bool:
        text = str(field_name or "")
        lowered = text.lower()
        return lowered == "id" or lowered.endswith("_id")

    @staticmethod
    def _is_scalar_value(value: Any) -> bool:
        return isinstance(value, SCALAR_EVIDENCE_TYPES)

    @staticmethod
    def _normalize_field_name(field_name: str) -> str:
        text = str(field_name or "").strip()
        if not text:
            return text
        text = text.replace(".", "_").replace("-", "_").replace(" ", "_")
        text = CAMEL_TO_SNAKE_PATTERN_1.sub(r"\1_\2", text)
        text = CAMEL_TO_SNAKE_PATTERN_2.sub(r"\1_\2", text)
        text = re.sub(r"_+", "_", text)
        return text.lower().strip("_")

    @staticmethod
    def _uses_entity_id_lookup(cypher_query: str, raw_entities: list[EntityRef]) -> bool:
        if not cypher_query or not raw_entities:
            return False
        by_param = {
            str(item.param_name or "").strip(): str(item.label or "").strip()
            for item in raw_entities
            if str(item.param_name or "").strip() and str(item.label or "").strip()
        }
        if not by_param:
            return False
        variable_labels = AgentController._parse_variable_labels(cypher_query)
        for matched in ENTITY_ID_LOOKUP_PATTERN.finditer(cypher_query):
            variable_name = matched.group(1)
            param_name = matched.group(2)
            variable_label = variable_labels.get(variable_name)
            if variable_label is None:
                continue
            if by_param.get(param_name) == variable_label:
                return True
        return False

    @staticmethod
    def _parse_variable_labels(cypher_query: str) -> dict[str, str]:
        labels = {}
        for variable, label in MATCH_LABEL_PATTERN.findall(cypher_query or ""):
            labels.setdefault(variable, label)
        return labels

    @staticmethod
    def _parse_return_alias_map(cypher_query: str) -> dict[str, dict[str, str | None]]:
        lowered = cypher_query.lower()
        marker = " return "
        index = lowered.find(marker)
        if index < 0:
            return {}
        clause = cypher_query[index + len(marker) :]
        items = []
        current = []
        depth = 0
        for char in clause:
            if char in "([{" :
                depth += 1
            elif char in ")]}":
                depth = max(depth - 1, 0)
            if char == "," and depth == 0:
                items.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            items.append("".join(current).strip())

        mapping = {}
        for item in items:
            matched = RETURN_ALIAS_PATTERN.match(item)
            if matched is None:
                continue
            variable, prop, alias = matched.groups()
            output_name = alias or f"{variable}.{prop}"
            mapping[str(output_name)] = {
                "var": variable,
                "prop": prop,
                "label": None,
            }
        return mapping


class KGQAPlanner:
    def __init__(self, resources):
        self.resources = resources

    def plan(self, question: str, *, history: list[dict[str, str]] | None = None, memory_hint: str | None = None) -> dict:
        llm = self.resources.get_llm()
        if llm is None:
            raise RuntimeError("LLM is not enabled for the current runtime.")
        if self.resources.json_parser is None:
            raise ImportError("JSON parser dependencies are not installed in the current environment.")

        prompt = """
你是一个专业的 Neo4j Cypher 查询规划器。请根据用户当前问题、最近对话历史和知识图谱 schema，输出严格 JSON。

最近对话历史：
{history_text}

任务记忆提示（可能为空）：
{memory_hint}

用户问题：{question}

图谱 schema：
{schema_info}

文本标签语义约束：
- `AttributeTag`：表示可结构化、可检索、可作为筛选条件的属性值。优先对应材质、功效、风格、颜色、工艺、功能特征，例如“纯棉”“控油”“去屑”“复古”“白色”“黑色钛金属”“无硅油”。不要把商品主干名词/品类词、低信息量营销词、纯规格型号词标成 `AttributeTag`。连续属性如果可分别筛选，应拆开理解。
- `PeopleTag`：表示明确的适用对象或人群，例如“儿童”“男士”“女士”“婴幼儿”“女童”“男宝宝”。不要把容量、型号、属性词标成 `PeopleTag`。
- `SpecTag`：表示规格、数量、容量、型号、版本号、数值范围或组合表达，例如“2K”“150ml”“12GB+256GB”“A3294”“5G”“110码”“iPhone 16 Pro”。数字/字母/符号/单位要整体保留。不要把纯属性词标成 `SpecTag`。

约束：
- 对需要实体对齐的参数，禁止生成 `id(node) = $param` 或 `elementId(node) = $param` 这类按图内部 ID 匹配的条件；默认使用 `node.name = $param` 或等价的业务属性匹配。
- 查询商品、品牌、类目等节点的展示名称时，优先使用稳定字段 `name`。尽量不要使用 `title` 属性；只有在 schema 或问题明确要求且确认该标签稳定存在 `title` 时才使用。
- `tool_plan` 保持格式中的结构，根据问题合理填写 arguments：
  - `entity_link_tool`：用于把用户原始实体对齐到图谱标准节点名。可选参数：
    - `mode`：`fulltext | hybrid`
    - `top_k`：返回候选数量，整数，范围 `1-10`
  - `graph_query_tool`：用于执行参数化 Cypher 查询。可选参数：
    - `timeout_ms`：查询超时毫秒数，整数，范围 `100-10000`
  - `answer_tool`：用于根据问题、查询结果和证据生成最终中文回答。`arguments` 保持空对象即可。
- `entities_to_align` 是“0 到多个”待对齐实体列表。

输出格式：
{{
  "tool_plan": [
    {{"tool_name": "entity_link_tool", "arguments": {{"mode": "hybrid", "top_k": 3}}}},
    {{"tool_name": "graph_query_tool", "arguments": {{"timeout_ms": 2000}}}},
    {{"tool_name": "answer_tool", "arguments": {{}}}}
  ],
  "cypher_query": "参数化 Cypher 查询",
  "entities_to_align": [
    {{
      "param_name": "param_0",
      "entity": "苹果",
      "label": "Trademark"
    }},
    {{
      "param_name": "param_1",
      "entity": "手机",
      "label": "Category3"
    }}
  ]
}}
"""
        rendered = prompt.format(
            question=question,
            history_text=self._format_history(history),
            memory_hint=memory_hint or "无",
            schema_info=self.resources.get_schema(),
        )
        result = llm.invoke(rendered)
        raw_content = getattr(result, "content", str(result))
        parse_result = self._parse_output(raw_content)
        payload = parse_result["payload"] if isinstance(parse_result["payload"], dict) else {}
        normalized_entities = self._normalize_entities(payload.get("entities_to_align"))
        normalized_tool_plan, invalid_tool_names = self._normalize_tool_plan(payload.get("tool_plan"))
        parse_result["payload"] = (
            {
                "tool_plan": normalized_tool_plan,
                "cypher_query": str(payload.get("cypher_query", "")).strip(),
                "entities_to_align": normalized_entities,
            }
            if payload
            else None
        )
        parse_result["plan_valid"] = bool(
            parse_result["payload"]
            and isinstance(parse_result["payload"]["tool_plan"], list)
            and not invalid_tool_names
            and parse_result["payload"]["cypher_query"]
            and isinstance(parse_result["payload"]["entities_to_align"], list)
        )
        parse_result["invalid_tool_names"] = invalid_tool_names
        return parse_result

    def _parse_output(self, raw_content: str) -> dict:
        parse_success_raw = False
        parse_success_repaired = False
        payload = None
        parse_error = None
        repaired_content = raw_content

        try:
            payload = self.resources.json_parser.invoke(raw_content)
            parse_success_raw = True
            parse_success_repaired = True
        except Exception as raw_exc:
            parse_error = str(raw_exc)
            repaired_content = repair_json(raw_content, ensure_ascii=False) if repair_json is not None else raw_content
            try:
                payload = self.resources.json_parser.invoke(repaired_content)
                parse_success_repaired = True
            except Exception as repaired_exc:
                parse_error = str(repaired_exc)

        if payload is not None and not isinstance(payload, dict):
            parse_error = f"Parsed JSON payload must be an object, got {type(payload).__name__}."
            payload = None
            parse_success_raw = False
            parse_success_repaired = False

        return {
            "raw_content": raw_content,
            "repaired_content": repaired_content,
            "parse_success_raw": parse_success_raw,
            "parse_success_repaired": parse_success_repaired,
            "payload": payload,
            "parse_error": parse_error,
        }

    @staticmethod
    def _normalize_entities(raw_entities) -> list[dict]:
        if not isinstance(raw_entities, list):
            return []
        normalized = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            param_name = str(item.get("param_name", "")).strip()
            entity = str(item.get("entity", "")).strip()
            label = str(item.get("label", "")).strip()
            if not param_name or not entity or not label:
                continue
            normalized.append({"param_name": param_name, "entity": entity, "label": label})
        return normalized

    @staticmethod
    def _normalize_tool_plan(raw_tool_plan) -> tuple[list[dict], list[str]]:
        if not isinstance(raw_tool_plan, list):
            return [], []
        normalized = []
        invalid_tool_names = []
        for item in raw_tool_plan:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", "")).strip()
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if not tool_name:
                continue
            if tool_name not in SUPPORTED_KGQA_TOOLS:
                invalid_tool_names.append(tool_name)
                continue
            normalized.append({"tool_name": tool_name, "arguments": arguments})
        return normalized, invalid_tool_names

    @staticmethod
    def _format_history(history: list[dict[str, str]] | None) -> str:
        if not history:
            return "无"
        lines = []
        for turn in history:
            user_message = turn.get("user", "").strip()
            assistant_message = turn.get("assistant", "").strip()
            if user_message:
                lines.append(f"用户: {user_message}")
            if assistant_message:
                lines.append(f"助手: {assistant_message}")
        return "\n".join(lines) if lines else "无"
