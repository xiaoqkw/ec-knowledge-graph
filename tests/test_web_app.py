import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from fastapi.testclient import TestClient
    import web.app
    from web.memory import InMemoryQASessionStore
except ImportError:
    TestClient = None
    web = None


@unittest.skipIf(TestClient is None, "fastapi is not installed in the current environment")
class WebAppTestCase(unittest.TestCase):
    def test_get_qa_service_uses_shared_agent_controller(self):
        stub_controller = object()
        stub_service = object()
        with patch.object(web.app, "qa_service", None), patch.object(
            web.app,
            "qa_service_error",
            None,
        ), patch.object(
            web.app,
            "get_agent_controller",
            return_value=stub_controller,
        ), patch.object(
            web.app,
            "ChatService",
            return_value=stub_service,
        ) as chat_service_cls:
            service = web.app.get_qa_service()
            self.assertIs(service, stub_service)
            chat_service_cls.assert_called_once_with(controller=stub_controller)

    def test_chat_keeps_session_history(self):
        class StubQAService:
            def __init__(self):
                self.calls = []

            def run_agent(self, message, history=None, session_id=None, session=None, enable_session_memory=False, align_entities=True):
                captured_history = [dict(item) for item in (history or [])]
                self.calls.append((message, captured_history))
                trace = type(
                    "Trace",
                    (),
                    {
                        "request_id": f"trace-{len(self.calls)}",
                        "plan": [],
                        "tool_calls": [],
                        "total_latency_ms": 12,
                        "fallback_used": None,
                        "metadata": {
                            "confirmed_entity_cache_updates": [],
                            "session_failure_memory": None,
                        },
                    },
                )()
                return type(
                    "Result",
                    (),
                    {
                        "answer": f"answer:{message}",
                        "session_id": session_id,
                        "trace": trace,
                    },
                )()

            def build_session_commit_payload(self, result, *, user_message):
                return {
                    "user_message": user_message,
                    "assistant_message": result.answer,
                    "trace_id": result.trace.request_id,
                    "entity_cache_updates": [],
                    "recent_failure": None,
                }

        qa_stub = StubQAService()
        with patch.object(web.app, "qa_service", qa_stub), patch.object(
            web.app,
            "qa_service_error",
            None,
        ), patch.object(
            web.app,
            "qa_session_store",
            InMemoryQASessionStore(),
        ):
            client = TestClient(web.app.app)

            first = client.post("/api/chat", json={"message": "Apple 都有哪些产品"})
            self.assertEqual(first.status_code, 200)
            first_payload = first.json()
            self.assertTrue(first_payload["session_id"])

            second = client.post(
                "/api/chat",
                json={
                    "message": "那它有哪些手机",
                    "session_id": first_payload["session_id"],
                },
            )
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["session_id"], first_payload["session_id"])
            self.assertEqual(qa_stub.calls[0][1], [])
            self.assertEqual(
                qa_stub.calls[1][1],
                [{"user": "Apple 都有哪些产品", "assistant": "answer:Apple 都有哪些产品"}],
            )

    def test_chat_returns_503_when_qa_not_enabled(self):
        with patch.object(web.app, "qa_service", None), patch.object(
            web.app,
            "qa_service_error",
            "disabled for test",
        ):
            client = TestClient(web.app.app)
            response = client.post("/api/chat", json={"message": "Apple 都有哪些产品"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("disabled", response.json()["detail"])

    def test_close_services_clears_error_flags(self):
        with patch.object(web.app, "qa_service_error", "qa failed"), patch.object(
            web.app,
            "dialogue_service_error",
            "dialogue failed",
        ), patch.object(
            web.app,
            "qa_service",
            None,
        ), patch.object(
            web.app,
            "dialogue_service",
            None,
        ), patch.object(
            web.app,
            "agent_runtime",
            None,
        ), patch.object(
            web.app,
            "shared_retriever",
            None,
        ), patch.object(
            web.app,
            "agent_controller",
            None,
        ):
            web.app.close_services()
            self.assertIsNone(web.app.qa_service_error)
            self.assertIsNone(web.app.dialogue_service_error)

    def test_get_qa_service_can_retry_after_shutdown_clears_error(self):
        stub_controller = object()
        stub_service = object()
        closable = type("Closable", (), {"close": lambda self: None})()
        with patch.object(web.app, "qa_service", None), patch.object(
            web.app,
            "qa_service_error",
            "old failure",
        ), patch.object(
            web.app,
            "dialogue_service_error",
            None,
        ), patch.object(
            web.app,
            "agent_runtime",
            None,
        ), patch.object(
            web.app,
            "shared_retriever",
            None,
        ), patch.object(
            web.app,
            "agent_controller",
            None,
        ), patch.object(
            web.app,
            "dialogue_service",
            closable,
        ):
            web.app.close_services()
        with patch.object(web.app, "get_agent_controller", return_value=stub_controller), patch.object(
            web.app,
            "ChatService",
            return_value=stub_service,
        ):
            service = web.app.get_qa_service()
            self.assertIs(service, stub_service)

    def test_get_dialogue_service_can_retry_after_shutdown_clears_error(self):
        stub_controller = object()
        stub_dialogue_service = object()
        stub_retriever = object()
        with patch.object(web.app, "qa_service_error", None), patch.object(
            web.app,
            "dialogue_service_error",
            "old failure",
        ), patch.object(
            web.app,
            "dialogue_service",
            None,
        ), patch.object(
            web.app,
            "agent_runtime",
            None,
        ), patch.object(
            web.app,
            "shared_retriever",
            None,
        ), patch.object(
            web.app,
            "agent_controller",
            None,
        ), patch.object(
            web.app,
            "qa_service",
            None,
        ):
            web.app.close_services()
        with patch.object(web.app, "get_agent_controller", return_value=stub_controller), patch.object(
            web.app,
            "shared_retriever",
            stub_retriever,
        ), patch.object(
            web.app,
            "DialogueService",
            return_value=stub_dialogue_service,
        ):
            service = web.app.get_dialogue_service()
            self.assertIs(service, stub_dialogue_service)

    def test_dialogue_response_keeps_suggested_budget_min(self):
        stub_dialogue_service = type(
            "StubDialogueService",
            (),
            {
                "chat": staticmethod(
                    lambda *args, **kwargs: {
                        "session_id": "session-1",
                        "message": "需要放宽预算",
                        "mode": "dialogue",
                        "action": "ask_slot",
                        "state": {
                            "domain": "phone_guide",
                            "intent": "inform",
                            "filled_slots": {"brand": "苹果", "budget_max": 4000},
                            "pending_slots": [],
                            "suggested_budget_min": 8197,
                        },
                        "recommendations": [],
                    }
                )
            },
        )()
        with patch.object(
            web.app,
            "get_dialogue_service",
            return_value=stub_dialogue_service,
        ):
            client = TestClient(web.app.app)
            response = client.post(
                "/api/dialogue/chat",
                json={"message": "帮我筛一下吧", "session_id": "session-1"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["state"]["suggested_budget_min"], 8197)

    def test_agent_chat_returns_trace_and_debug_payload(self):
        class StubQAService:
            def run_agent(self, message, history=None, session_id=None, session=None, enable_session_memory=False, align_entities=True):
                trace = type(
                    "Trace",
                    (),
                    {
                        "request_id": "trace-1",
                        "plan": [type("ToolPlan", (), {"tool_name": "entity_link_tool"})()],
                        "tool_calls": [type("ToolCall", (), {"tool_name": "entity_link_tool"})()],
                        "total_latency_ms": 12,
                        "fallback_used": None,
                        "dict": lambda self: {"request_id": "trace-1", "tool_calls": []},
                    },
                )()
                return type(
                    "Result",
                    (),
                    {
                        "answer": "agent answer",
                        "session_id": session_id,
                        "trace": trace,
                    },
                )()

            def build_session_commit_payload(self, result, *, user_message):
                return {
                    "user_message": user_message,
                    "assistant_message": result.answer,
                    "trace_id": result.trace.request_id,
                    "entity_cache_updates": [],
                    "recent_failure": None,
                }

        qa_stub = StubQAService()
        with patch.object(web.app, "qa_service", qa_stub), patch.object(
            web.app,
            "qa_service_error",
            None,
        ), patch.object(
            web.app,
            "qa_session_store",
            InMemoryQASessionStore(),
        ):
            client = TestClient(web.app.app)
            response = client.post("/api/agent/chat?debug=true", json={"message": "test"})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["answer"], "agent answer")
            self.assertEqual(payload["trace_id"], "trace-1")
            self.assertEqual(payload["plan_summary"][0]["tool"], "entity_link_tool")
            self.assertEqual(payload["trace"]["request_id"], "trace-1")


if __name__ == "__main__":
    unittest.main()
