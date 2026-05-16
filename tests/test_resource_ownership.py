import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dialogue.service import DialogueService
import web.app
from web.service import ChatService


class ResourceOwnershipTestCase(unittest.TestCase):
    def test_chat_service_does_not_close_injected_runtime(self):
        runtime = type(
            "Runtime",
            (),
            {
                "close": lambda self: setattr(self, "closed", True),
                "closed": False,
                "resources": type("Resources", (), {"get_llm": staticmethod(lambda: None), "json_parser": None, "str_parser": None})(),
            },
        )()
        controller = type("Controller", (), {"runtime": runtime})()
        service = ChatService(controller=controller)
        service.close()
        self.assertFalse(runtime.closed)

    def test_dialogue_service_does_not_close_injected_runtime_or_retriever(self):
        runtime = type(
            "Runtime",
            (),
            {
                "close": lambda self: setattr(self, "closed", True),
                "closed": False,
                "resources": type("Resources", (), {"get_llm": staticmethod(lambda: None), "json_parser": None, "str_parser": None})(),
            },
        )()
        controller = type("Controller", (), {"runtime": runtime})()
        retriever = type(
            "Retriever",
            (),
            {
                "closed": False,
                "close": lambda self: setattr(self, "closed", True),
                "load_brand_vocabulary": lambda self: [],
            },
        )()
        nlu = type("NLU", (), {"parse": staticmethod(lambda *args, **kwargs: type("Result", (), {"intent": "reset", "slots": {}})())})()
        service = DialogueService(retriever=retriever, nlu=nlu, llm_enabled=False, agent_controller=controller)
        service.close()
        self.assertFalse(runtime.closed)
        self.assertFalse(retriever.closed)

    def test_app_shutdown_closes_shared_runtime_and_retriever_once(self):
        class StubRuntime:
            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        class StubRetriever:
            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        runtime = StubRuntime()
        retriever = StubRetriever()
        dialogue_service = type("DialogueService", (), {"close": lambda self: None})()
        qa_service = type("QAService", (), {"close": lambda self: None})()

        with patch.object(web.app, "agent_runtime", runtime), patch.object(
            web.app,
            "shared_retriever",
            retriever,
        ), patch.object(
            web.app,
            "agent_controller",
            object(),
        ), patch.object(
            web.app,
            "dialogue_service",
            dialogue_service,
        ), patch.object(
            web.app,
            "qa_service",
            qa_service,
        ):
            web.app.close_services()
            self.assertEqual(runtime.close_count, 1)
            self.assertEqual(retriever.close_count, 1)
            self.assertIsNone(web.app.agent_runtime)
            self.assertIsNone(web.app.shared_retriever)
            self.assertIsNone(web.app.agent_controller)
            self.assertIsNone(web.app.dialogue_service)
            self.assertIsNone(web.app.qa_service)


if __name__ == "__main__":
    unittest.main()
