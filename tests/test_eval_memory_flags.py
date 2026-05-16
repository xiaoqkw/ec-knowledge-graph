import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval import entity_linking_eval, kgqa_eval
from web.memory import InMemoryQASessionStore


class EvalMemoryFlagsTestCase(unittest.TestCase):
    def test_kgqa_run_llm_baseline_passes_memory_flag_when_memory_off(self):
        service = type(
            "StubService",
            (),
            {
                "trace_chat": staticmethod(
                    lambda *args, **kwargs: {
                        "raw_cypher_output": "",
                        "repaired_cypher_output": "",
                        "parse_success_raw": True,
                        "parse_success_repaired": True,
                        "cypher_query_present": True,
                        "cypher_query": "MATCH (n) RETURN n",
                        "entities_to_align": [],
                        "aligned_entities": [],
                        "executed_params": {},
                        "unsafe_cypher": False,
                        "execution_success": True,
                        "execution_error": None,
                        "non_empty_result": True,
                        "query_result": [{"name": "Apple"}],
                        "answer": "Apple",
                    }
                )
            },
        )()
        with patch.object(service, "trace_chat", wraps=service.trace_chat) as trace_chat:
            kgqa_eval.run_llm_baseline(
                service,
                {"question": "q", "gold_cypher_type": "brand_products", "answer_keywords": [], "required_entities": []},
                "full",
                enable_session_memory=False,
            )
            self.assertFalse(trace_chat.call_args.kwargs["enable_session_memory"])

    def test_kgqa_run_llm_baseline_memory_on_commits_session(self):
        trace = type(
            "Trace",
            (),
            {
                "request_id": "trace-1",
                "failure_tags": [],
                "tool_calls": [],
                "metadata": {
                    "planner_raw_output": "",
                    "planner_repaired_output": "",
                    "planner_parse_success_raw": True,
                    "planner_parse_success_repaired": True,
                    "parsed_payload": {},
                    "cypher_query": "MATCH (n) RETURN n",
                    "entities_to_align": [],
                    "aligned_entities": [],
                    "executed_params": {},
                    "query_result": [{"name": "Apple"}],
                    "confirmed_entity_cache_updates": [],
                    "session_failure_memory": None,
                },
            },
        )()
        result = type("Result", (), {"answer": "Apple", "trace": trace})()
        service = type(
            "StubService",
            (),
            {
                "run_agent": staticmethod(lambda *args, **kwargs: result),
                "build_session_commit_payload": staticmethod(
                    lambda result, *, user_message: {
                        "user_message": user_message,
                        "assistant_message": result.answer,
                        "trace_id": result.trace.request_id,
                        "entity_cache_updates": [],
                        "recent_failure": None,
                    }
                ),
            },
        )()
        store = InMemoryQASessionStore()
        session = store.get_or_create("eval-session")
        record = kgqa_eval.run_llm_baseline(
            service,
            {"question": "q", "gold_cypher_type": "brand_products", "answer_keywords": [], "required_entities": []},
            "full",
            enable_session_memory=True,
            session_store=store,
            session=session,
        )
        self.assertEqual(session.last_trace_id, "trace-1")
        self.assertEqual(session.history[-1]["user"], "q")
        self.assertEqual(session.last_trace_id, "trace-1")

    def test_entity_linking_eval_rejects_session_memory_flag(self):
        test_argv = ["entity_linking_eval.py", "--enable-session-memory"]
        with patch.object(sys, "argv", test_argv):
            with self.assertRaises(ValueError):
                entity_linking_eval.main()

    def test_kgqa_eval_all_uses_separate_sessions_per_baseline(self):
        created_session_ids = []

        class StubStore:
            def get_or_create(self, session_id):
                created_session_ids.append(session_id)
                return type("Session", (), {"session_id": session_id, "history": []})()

        class StubService:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                return None

        with patch.object(kgqa_eval, "load_dataset", return_value=[]), patch.object(
            kgqa_eval,
            "ChatService",
            return_value=StubService(),
        ), patch.object(
            kgqa_eval,
            "InMemoryQASessionStore",
            return_value=StubStore(),
        ):
            test_argv = ["kgqa_eval.py", "--baseline", "all", "--enable-session-memory"]
            with patch.object(sys, "argv", test_argv):
                kgqa_eval.main()
        self.assertEqual(
            created_session_ids,
            ["kgqa_eval_session_template", "kgqa_eval_session_full", "kgqa_eval_session_ablation"],
        )


if __name__ == "__main__":
    unittest.main()
