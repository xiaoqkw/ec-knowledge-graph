import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.controller import AgentController


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


class AgentControllerTestCase(unittest.TestCase):
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
