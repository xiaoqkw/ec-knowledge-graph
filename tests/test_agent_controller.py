import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.controller import AgentController
from agent.types import EvidenceItem


class StubResources:
    def __init__(self):
        self.json_parser = None
        self.str_parser = None


class StubRuntime:
    def __init__(self):
        self.resources = StubResources()
        self.tools = {}
        self.calls = []

    def register_tool(self, tool):
        self.tools[tool.name] = tool

    def build_trace(self, *, session_id, user_query, intent):
        from agent.types import ExecutionTrace

        return ExecutionTrace(request_id="trace-1", session_id=session_id or "session-1", user_query=user_query, intent=intent)

    def invoke_tool(self, trace, tool_name, payload):
        self.calls.append((tool_name, payload))
        if tool_name == "entity_link_tool":
            return type(
                "Record",
                (),
                {
                    "ok": True,
                    "failure_mode": None,
                    "output_payload": {
                        "aligned_entities": payload["entities"],
                        "candidates_by_param": {"param_0": [{"name": "Apple", "score": 1.0}]},
                    },
                },
            )()
        if tool_name == "graph_query_tool":
            return type(
                "Record",
                (),
                {
                    "ok": False,
                    "failure_mode": "query_validation_failed",
                    "output_payload": {},
                },
            )()
        return type(
            "Record",
            (),
            {
                "ok": True,
                "failure_mode": None,
                "output_payload": {"answer": "answer", "llm_used": True},
            },
        )()

    def finalize_trace(self, trace, *, answer):
        return type("Result", (), {"answer": answer, "session_id": trace.session_id, "trace": trace})()


class StubPlanner:
    def __init__(self, should_raise=False, tool_plan=None):
        self.should_raise = should_raise
        self.tool_plan = tool_plan or [
            {"tool_name": "entity_link_tool", "arguments": {}},
            {"tool_name": "graph_query_tool", "arguments": {}},
            {"tool_name": "answer_tool", "arguments": {}},
        ]

    def plan(self, question, *, history=None, memory_hint=None):
        if self.should_raise:
            raise RuntimeError("planner unavailable")
        return {
            "raw_content": "{}",
            "repaired_content": "{}",
            "parse_success_raw": True,
            "parse_success_repaired": True,
            "payload": {
                "tool_plan": self.tool_plan,
                "cypher_query": "MATCH (n) RETURN n",
                "entities_to_align": [
                    {"param_name": "param_0", "entity": "Apple", "label": "Trademark"},
                ],
            },
            "parse_error": None,
            "plan_valid": True,
        }


class PromptCaptureLLM:
    def __init__(self):
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return type("Result", (), {"content": "{}"})()


class StubJsonParser:
    @staticmethod
    def invoke(_content):
        return {
            "tool_plan": [
                {"tool_name": "entity_link_tool", "arguments": {"mode": "hybrid", "top_k": 3}},
                {"tool_name": "graph_query_tool", "arguments": {"timeout_ms": 2000}},
                {"tool_name": "answer_tool", "arguments": {}},
            ],
            "cypher_query": "MATCH (n:AttributeTag) WHERE n.name = $param_0 RETURN n.name AS name",
            "entities_to_align": [
                {"param_name": "param_0", "entity": "不卡顿", "label": "AttributeTag"},
            ],
        }


class AgentControllerTestCase(unittest.TestCase):
    def test_kgqa_planner_prompt_includes_tag_semantics(self):
        import agent.controller as controller_module

        llm = PromptCaptureLLM()

        class PlannerResources:
            json_parser = StubJsonParser()

            @staticmethod
            def get_llm():
                return llm

            @staticmethod
            def get_schema():
                return "(:SPU)-[:Have]->(:AttributeTag)"

        planner = controller_module.KGQAPlanner(PlannerResources())
        planner.plan("帮我找一个不卡顿的手机")

        self.assertIn("AttributeTag", llm.prompt)
        self.assertIn("PeopleTag", llm.prompt)
        self.assertIn("SpecTag", llm.prompt)
        self.assertIn("不要把纯属性词标成 `SpecTag`", llm.prompt)
        self.assertIn("连续属性如果可分别筛选，应拆开理解", llm.prompt)
        self.assertIn("entities_to_align` 是“0 到多个”待对齐实体列表", llm.prompt)
        self.assertIn('"param_name": "param_1"', llm.prompt)
        self.assertIn('"label": "Category3"', llm.prompt)
        self.assertIn("`top_k`：返回候选数量，整数，范围 `1-10`", llm.prompt)
        self.assertIn("`timeout_ms`：查询超时毫秒数，整数，范围 `100-10000`", llm.prompt)

    def test_field_name_normalizes_to_snake_case(self):
        self.assertEqual(AgentController._normalize_field_name("productName"), "product_name")
        self.assertEqual(AgentController._normalize_field_name("BrandName"), "brand_name")
        self.assertEqual(AgentController._normalize_field_name("spu.name"), "spu_name")

    def test_non_name_spu_field_can_bind_lookup_id_when_explicit_id_exists(self):
        entity_id, lookup_id = AgentController._resolve_entity_identity(
            row={"price": 1999, "spu_id": 12},
            raw_field_name="price",
            node_label="SPU",
            resolved_property="price",
            row_id_fields={"spu_id": 12},
        )
        self.assertEqual(entity_id, "12")
        self.assertEqual(lookup_id, 12)

    def test_non_name_field_does_not_bind_generic_id(self):
        entity_id, lookup_id = AgentController._resolve_entity_identity(
            row={"price": 1999, "id": 88},
            raw_field_name="price",
            node_label="SPU",
            resolved_property="price",
            row_id_fields={"id": 88},
        )
        self.assertIsNone(entity_id)
        self.assertIsNone(lookup_id)

    def test_sku_field_does_not_bind_product_id_fallback(self):
        entity_id, lookup_id = AgentController._resolve_entity_identity(
            row={"price": 1999, "product_id": 12},
            raw_field_name="price",
            node_label="SKU",
            resolved_property="price",
            row_id_fields={"product_id": 12},
        )
        self.assertIsNone(entity_id)
        self.assertIsNone(lookup_id)

    def test_sku_field_prefers_sku_id_over_spu_id(self):
        entity_id, lookup_id = AgentController._resolve_entity_identity(
            row={"price": 1999, "spu_id": 12, "sku_id": 34},
            raw_field_name="price",
            node_label="SKU",
            resolved_property="price",
            row_id_fields={"spu_id": 12, "sku_id": 34},
        )
        self.assertEqual(entity_id, "34")
        self.assertEqual(lookup_id, 34)

    def test_entity_id_lookup_guard_does_not_reject_mixed_query_on_other_label(self):
        cypher_query = (
            "MATCH (tm:Trademark {name: $param_0}) "
            "MATCH (spu:SPU) WHERE id(spu) = $param_1 "
            "RETURN tm.name AS brand_name"
        )
        raw_entities = [
            type("Entity", (), {"param_name": "param_0", "label": "Trademark"})(),
            type("Entity", (), {"param_name": "param_1", "label": "Category3"})(),
        ]
        self.assertFalse(AgentController._uses_entity_id_lookup(cypher_query, raw_entities))

    def test_evidence_item_rejects_invalid_kind_and_source(self):
        with self.assertRaises(Exception):
            EvidenceItem(
                kind="bad",
                source="graph",
                field_name="name",
                value="x",
            )
        with self.assertRaises(Exception):
            EvidenceItem(
                kind="graph",
                source="bad",
                field_name="name",
                value="x",
            )

    def test_planner_exception_sets_fallback(self):
        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None

        import agent.controller as controller_module

        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: StubPlanner(should_raise=True)
        try:
            result = AgentController.run_kgqa(controller, "question")
        finally:
            controller_module.KGQAPlanner = original

        self.assertEqual(result.trace.fallback_used, "template_cypher")
        self.assertIn("plan_schema_invalid", result.trace.failure_tags)

    def test_query_validation_failure_is_not_unsafe(self):
        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None

        import agent.controller as controller_module

        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: StubPlanner()
        try:
            result = AgentController.run_kgqa(controller, "question")
        finally:
            controller_module.KGQAPlanner = original

        self.assertNotIn("unsafe_query_blocked", result.trace.failure_tags)
        self.assertIn("plan_schema_invalid", result.trace.failure_tags)
        self.assertEqual(result.trace.failure_stage, "query")
        self.assertEqual(result.trace.plan[0].tool_name, "entity_link_tool")

    def test_tool_plan_arguments_drive_execution_payload(self):
        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None
        controller.schema_mapping_cache = None

        import agent.controller as controller_module

        planner = StubPlanner(
            tool_plan=[
                {"tool_name": "entity_link_tool", "arguments": {"mode": "fulltext", "top_k": 5}},
                {"tool_name": "graph_query_tool", "arguments": {"timeout_ms": 3456}},
                {"tool_name": "answer_tool", "arguments": {}},
            ]
        )
        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: planner
        try:
            AgentController.run_kgqa(controller, "question")
        finally:
            controller_module.KGQAPlanner = original

        self.assertEqual(runtime.calls[0][1]["mode"], "fulltext")
        self.assertEqual(runtime.calls[0][1]["top_k"], 5)
        self.assertEqual(runtime.calls[1][1]["timeout_ms"], 3456)
        self.assertIn("evidence", runtime.calls[2][1])

    def test_invalid_tool_arguments_fallback_to_defaults(self):
        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None

        import agent.controller as controller_module

        planner = StubPlanner(
            tool_plan=[
                {"tool_name": "entity_link_tool", "arguments": {"mode": "bad-mode", "top_k": "three"}},
                {"tool_name": "graph_query_tool", "arguments": {"timeout_ms": {}}},
                {"tool_name": "answer_tool", "arguments": {}},
            ]
        )
        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: planner
        try:
            result = AgentController.run_kgqa(controller, "question")
        finally:
            controller_module.KGQAPlanner = original

        self.assertEqual(runtime.calls[0][1]["mode"], "hybrid")
        self.assertEqual(runtime.calls[0][1]["top_k"], 3)
        self.assertEqual(runtime.calls[1][1]["timeout_ms"], 2000)
        self.assertEqual(result.trace.fallback_used, "default_tool_arguments")
        self.assertIn("plan_schema_invalid", result.trace.failure_tags)

    def test_entity_id_lookup_plan_is_rejected_and_falls_back_to_template(self):
        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None
        controller.schema_mapping_cache = None

        import agent.controller as controller_module

        class IdLookupPlanner:
            def plan(self, question, *, history=None, memory_hint=None):
                return {
                    "raw_content": "{}",
                    "repaired_content": "{}",
                    "parse_success_raw": True,
                    "parse_success_repaired": True,
                    "payload": {
                        "tool_plan": [
                            {"tool_name": "entity_link_tool", "arguments": {"mode": "hybrid", "top_k": 3}},
                            {"tool_name": "graph_query_tool", "arguments": {"timeout_ms": 2000}},
                            {"tool_name": "answer_tool", "arguments": {}},
                        ],
                        "cypher_query": "MATCH (t:Trademark) WHERE id(t) = $param_0 RETURN t.name AS name",
                        "entities_to_align": [
                            {"param_name": "param_0", "entity": "苹果", "label": "Trademark"},
                        ],
                    },
                    "parse_error": None,
                    "plan_valid": True,
                }

        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: IdLookupPlanner()
        try:
            result = AgentController.run_kgqa(controller, "苹果都有哪些产品？")
        finally:
            controller_module.KGQAPlanner = original

        self.assertEqual(result.trace.fallback_used, "template_cypher")
        self.assertTrue(result.trace.metadata["planner_entity_id_lookup_rejected"])
        self.assertIn("plan_schema_invalid", result.trace.failure_tags)
        self.assertIn("name: $param_0", result.trace.metadata["cypher_query"])

    def test_answer_tool_is_not_executed_when_missing_from_plan(self):
        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None

        import agent.controller as controller_module

        planner = StubPlanner(
            tool_plan=[
                {"tool_name": "entity_link_tool", "arguments": {}},
                {"tool_name": "graph_query_tool", "arguments": {}},
            ]
        )
        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: planner
        try:
            AgentController.run_kgqa(controller, "question")
        finally:
            controller_module.KGQAPlanner = original

        self.assertEqual([call[0] for call in runtime.calls], ["entity_link_tool", "graph_query_tool"])

    def test_unknown_tool_name_invalidates_plan_and_falls_back(self):
        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None

        import agent.controller as controller_module

        class InvalidToolPlanner:
            def plan(self, question, *, history=None, memory_hint=None):
                return {
                    "raw_content": "{}",
                    "repaired_content": "{}",
                    "parse_success_raw": True,
                    "parse_success_repaired": True,
                    "payload": {
                        "tool_plan": [
                            {"tool_name": "graph_query_tool", "arguments": {}},
                        ],
                        "cypher_query": "MATCH (n) RETURN n",
                        "entities_to_align": [
                            {"param_name": "param_0", "entity": "Apple", "label": "Trademark"},
                        ],
                    },
                    "parse_error": None,
                    "plan_valid": False,
                    "invalid_tool_names": ["unknown_tool"],
                }

        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: InvalidToolPlanner()
        try:
            result = AgentController.run_kgqa(controller, "question", align_entities=True)
        finally:
            controller_module.KGQAPlanner = original

        self.assertIn("plan_schema_invalid", result.trace.failure_tags)
        self.assertEqual(result.trace.fallback_used, "default_tool_plan")
        self.assertEqual(result.trace.metadata["planner_invalid_tool_names"], ["unknown_tool"])
        self.assertEqual(result.trace.plan[0].tool_name, "entity_link_tool")

    def test_session_cache_hit_skips_entity_link_tool_and_marks_cache_hit(self):
        from web.memory import CachedAlignedEntity, QASession

        import agent.controller as controller_module

        runtime = StubRuntime()
        controller = object.__new__(AgentController)
        controller.runtime = runtime
        controller.retriever = None
        controller.schema_mapping_cache = controller_module.SchemaMappingCache()

        session = QASession(session_id="session-1")
        session.task_memory.entity_cache.upsert(
            CachedAlignedEntity(
                label="Trademark",
                raw_entity="Apple",
                aligned_entity="Apple Inc.",
                param_name="param_0",
                matched=True,
            )
        )
        planner = StubPlanner()
        original = controller_module.KGQAPlanner
        controller_module.KGQAPlanner = lambda resources: planner
        try:
            result = AgentController.run_kgqa(controller, "question", session=session)
        finally:
            controller_module.KGQAPlanner = original

        self.assertEqual([call[0] for call in runtime.calls], ["graph_query_tool", "answer_tool"])
        self.assertTrue(result.trace.metadata["task_memory_skipped_entity_link"])
        self.assertTrue(result.trace.tool_calls[0].cache_hit)
        self.assertEqual(result.trace.metadata["task_memory_hit_keys"], ["Trademark::apple"])


if __name__ == "__main__":
    unittest.main()
