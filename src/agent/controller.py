from __future__ import annotations

import json

from agent.runtime import AgentRuntime, dump_model, parse_model, repair_json
from agent.tools import AnswerTool, EntityLinkTool, GraphQueryTool, PriceFloorTool, ProductCompareTool, ProductSearchTool
from agent.types import AgentRunResult, EntityRef, ExecutionTrace, ToolPlan
from configuration.config import ENTITY_INDEX_CONFIG


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


class AgentController:
    def __init__(self, runtime: AgentRuntime, *, retriever=None, enable_explain: bool = True):
        self.runtime = runtime
        self.retriever = retriever
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
    ) -> AgentRunResult:
        trace = self.runtime.build_trace(session_id=session_id, user_query=question, intent="kgqa")
        planner = KGQAPlanner(self.runtime.resources)

        try:
            plan_result = planner.plan(question, history=history)
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
        planner_plan = [parse_model(ToolPlan, item) for item in payload.get("tool_plan", [])]
        trace.plan = planner_plan
        if not plan_result["plan_valid"] or not trace.plan:
            trace.plan = self._default_tool_plan(align_entities=align_entities)
            if trace.fallback_used is None:
                trace.fallback_used = "default_tool_plan"

        plan_by_name = {item.tool_name: item.arguments for item in trace.plan}

        aligned_entities = raw_entities
        candidates_by_param: dict[str, list[dict]] = {}
        entity_link_args = plan_by_name.get("entity_link_tool", {})
        should_align = align_entities and "entity_link_tool" in plan_by_name
        if raw_entities and should_align:
            entity_record = self.runtime.invoke_tool(
                trace,
                "entity_link_tool",
                {
                    "entities": [dump_model(item) for item in raw_entities],
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
                aligned_entities = [parse_model(EntityRef, item) for item in entity_record.output_payload["aligned_entities"]]
                candidates_by_param = entity_record.output_payload.get("candidates_by_param", {})
                self._mark_entity_signal(trace, raw_entities, aligned_entities, candidates_by_param)
            else:
                self._set_failure(trace, "entity_missing", "entity_link")

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
            answer_record = self.runtime.invoke_tool(
                trace,
                "answer_tool",
                {
                    "question": question,
                    "query_result": rows,
                    "history": history or [],
                },
            )
            if answer_record.ok:
                answer = answer_record.output_payload.get("answer", "")
                self._mark_answer_signal(trace, answer, rows, bool(answer_record.output_payload.get("llm_used")))
            else:
                answer = "当前图谱中没有找到相关信息。"
        else:
            answer = json.dumps(rows, ensure_ascii=False) if rows else "当前图谱中没有找到相关信息。"

        trace.metadata["cypher_query"] = cypher_query
        trace.metadata["entities_to_align"] = [dump_model(item) for item in raw_entities]
        trace.metadata["aligned_entities"] = [dump_model(item) for item in aligned_entities]
        trace.metadata["executed_params"] = executed_params
        trace.metadata["query_result"] = rows
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


class KGQAPlanner:
    def __init__(self, resources):
        self.resources = resources

    def plan(self, question: str, *, history: list[dict[str, str]] | None = None) -> dict:
        llm = self.resources.get_llm()
        if llm is None:
            raise RuntimeError("LLM is not enabled for the current runtime.")
        if self.resources.json_parser is None:
            raise ImportError("JSON parser dependencies are not installed in the current environment.")

        prompt = """
你是一个专业的 Neo4j Cypher 查询规划器。请根据用户当前问题、最近对话历史和知识图谱 schema，输出严格 JSON。

最近对话历史：
{history_text}

用户问题：{question}

图谱 schema：
{schema_info}

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
      "entity": "原始实体名称",
      "label": "图谱节点标签"
    }}
  ]
}}
"""
        rendered = prompt.format(
            question=question,
            history_text=self._format_history(history),
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
