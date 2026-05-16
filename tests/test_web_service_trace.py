import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web.service import ChatService


class ChatServiceTraceTestCase(unittest.TestCase):
    def build_service(self) -> ChatService:
        service = object.__new__(ChatService)
        service.json_parser = None
        service.str_parser = None
        service.llm = None
        service.graph = None
        service.neo4j_vectors = {}
        return service

    def test_parse_cypher_output_tracks_missing_query(self):
        service = self.build_service()
        payload = {"entities_to_align": []}
        service.json_parser = type("Parser", (), {"invoke": staticmethod(lambda _: payload)})()
        parsed = service._parse_cypher_output("{}")
        self.assertTrue(parsed["parse_success_raw"])
        self.assertTrue(parsed["parse_success_repaired"])
        self.assertFalse(parsed["cypher_query_present"])

    def test_parse_cypher_output_rejects_non_object_json(self):
        service = self.build_service()
        service.json_parser = type("Parser", (), {"invoke": staticmethod(lambda _: [])})()
        parsed = service._parse_cypher_output("[]")
        self.assertFalse(parsed["parse_success_raw"])
        self.assertFalse(parsed["parse_success_repaired"])
        self.assertFalse(parsed["cypher_query_present"])
        self.assertIsNone(parsed["payload"])
        self.assertIn("object", parsed["parse_error"])

    def test_trace_chat_uses_raw_entities_for_ablation_params(self):
        service = self.build_service()
        service._build_cypher_prompt = lambda question, history=None: "prompt"
        service._invoke_cypher_llm = lambda prompt: '{"cypher_query":"MATCH (n) RETURN n","entities_to_align":[{"param_name":"param_0","entity":"iphone 16 pro","label":"SPU"}]}'
        service._parse_cypher_output = lambda raw: {
            "raw_content": raw,
            "repaired_content": raw,
            "parse_success_raw": True,
            "parse_success_repaired": True,
            "cypher_query_present": True,
            "payload": {
                "cypher_query": "MATCH (n) RETURN n",
                "entities_to_align": [
                    {"param_name": "param_0", "entity": "iphone 16 pro", "label": "SPU"}
                ],
            },
            "parse_error": None,
        }
        service._entity_align = lambda entities, **kwargs: [
            {"param_name": "param_0", "entity": "Apple iPhone 16 Pro", "label": "SPU"}
        ]
        service._execute_cypher = lambda cypher, params: [{"name": params["param_0"]}]
        service._generate_answer = lambda question, query_result, history=None: "answer"
        service._is_unsafe_cypher = lambda cypher: False

        trace = service.trace_chat("question", align_entities=False)
        self.assertEqual(trace["aligned_entities"][0]["entity"], "iphone 16 pro")
        self.assertEqual(trace["executed_params"], {"param_0": "iphone 16 pro"})

    def test_chat_keeps_string_return(self):
        service = self.build_service()
        service.trace_chat = lambda question, history=None, align_entities=True, session=None, enable_session_memory=False: {
            "cypher_query": "MATCH (n) RETURN n",
            "entities_to_align": [],
            "aligned_entities": [],
            "executed_params": {},
            "query_result": [],
            "answer": "final answer",
        }
        self.assertEqual(service.chat("test"), "final answer")

    def test_trace_chat_marks_execution_unsuccessful_when_query_not_executed(self):
        service = self.build_service()
        service.controller = object()
        service.run_agent = lambda question, history=None, align_entities=True, session=None, enable_session_memory=False: type(
            "Result",
            (),
            {
                "answer": "answer",
                "trace": type(
                    "Trace",
                    (),
                    {
                        "metadata": {
                            "planner_raw_output": "{}",
                            "planner_repaired_output": "{}",
                            "parsed_payload": {"cypher_query": "MATCH (n) RETURN n"},
                            "planner_parse_success_raw": True,
                            "planner_parse_success_repaired": True,
                            "cypher_query": "MATCH (n) RETURN n",
                            "entities_to_align": [],
                            "aligned_entities": [],
                            "executed_params": {},
                            "query_result": [],
                        },
                        "failure_tags": ["plan_schema_invalid"],
                        "tool_calls": [
                            type(
                                "Call",
                                (),
                                {
                                    "tool_name": "graph_query_tool",
                                    "ok": False,
                                    "metadata": {"error": "schema invalid"},
                                },
                            )()
                        ],
                        "request_id": "trace-1",
                    },
                )(),
            },
        )()
        trace = service.trace_chat("question")
        self.assertFalse(trace["execution_success"])

    def test_build_session_commit_payload_contains_trace_and_memory_fields(self):
        service = self.build_service()
        result = type(
            "Result",
            (),
            {
                "answer": "answer",
                "trace": type(
                    "Trace",
                    (),
                    {
                        "request_id": "trace-1",
                        "metadata": {
                            "confirmed_entity_cache_updates": [{"label": "Trademark", "raw_entity": "Apple"}],
                            "session_failure_memory": {"trace_id": "trace-1", "primary_failure_tag": "query_empty"},
                        },
                    },
                )(),
            },
        )()
        payload = service.build_session_commit_payload(result, user_message="question")
        self.assertEqual(payload["trace_id"], "trace-1")
        self.assertEqual(payload["entity_cache_updates"][0]["label"], "Trademark")
        self.assertEqual(payload["recent_failure"]["primary_failure_tag"], "query_empty")


if __name__ == "__main__":
    unittest.main()
