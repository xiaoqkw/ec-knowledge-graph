import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval import dialogue_smoke
from eval.dialogue_smoke import load_dataset, write_summary


class DialogueSmokeTestCase(unittest.TestCase):
    def test_load_dataset_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "smoke.jsonl"
            path.write_text(json.dumps({"case_id": "c1"}) + "\n", encoding="utf-8")
            dataset = load_dataset(path)
            self.assertEqual(dataset[0]["case_id"], "c1")

    def test_write_summary_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "out.json"
            write_summary(path, {"cases": []})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["cases"], [])

    def test_smoke_qa_handler_reuses_fixed_session_id(self):
        runtime = type(
            "Runtime",
            (),
            {
                "list_traces": staticmethod(lambda session_id: []),
                "close": staticmethod(lambda: None),
            },
        )()
        retriever = type("Retriever", (), {"close": staticmethod(lambda: None)})()
        controller = object()

        observed_dialogue_session_ids = []

        def chat_stub(message, session_id=None, qa_handler=None):
            observed_dialogue_session_ids.append(session_id)
            if qa_handler is not None:
                qa_handler(message)
            return {"session_id": session_id, "action": "fallback_qa", "recommendations": []}

        dialogue_service = type(
            "DialogueService",
            (),
            {
                "chat": staticmethod(chat_stub),
                "close": staticmethod(lambda: None),
            },
        )()

        observed_qa_session_ids = []

        class StubQAService:
            def run_agent(self, message, history=None, session_id=None, session=None, enable_session_memory=False):
                observed_qa_session_ids.append(session_id)
                trace = type("Trace", (), {"request_id": "trace-1"})()
                return type("Result", (), {"answer": "qa", "trace": trace})()

            def build_session_commit_payload(self, result, *, user_message):
                return {
                    "user_message": user_message,
                    "assistant_message": result.answer,
                    "trace_id": result.trace.request_id,
                    "entity_cache_updates": [],
                    "recent_failure": None,
                }

            def close(self):
                return None

        cases = [{"case_id": "case_1", "turns": ["Apple 都有哪些产品"], "expected_action": "fallback_qa"}]

        with patch.object(dialogue_smoke, "DEEPSEEK_API_KEY", "x"), patch.object(
            dialogue_smoke,
            "PhoneGuideRetriever",
            return_value=retriever,
        ), patch.object(
            dialogue_smoke,
            "AgentRuntime",
            return_value=runtime,
        ), patch.object(
            dialogue_smoke,
            "AgentController",
            return_value=controller,
        ), patch.object(
            dialogue_smoke,
            "DialogueService",
            return_value=dialogue_service,
        ), patch.object(
            dialogue_smoke,
            "ChatService",
            return_value=StubQAService(),
        ), patch.object(
            dialogue_smoke,
            "load_dataset",
            return_value=cases,
        ), patch.object(
            sys,
            "argv",
            ["dialogue_smoke.py"],
        ):
            dialogue_smoke.main()
        self.assertEqual(observed_dialogue_session_ids, ["smoke_case_1"])
        self.assertEqual(observed_qa_session_ids, ["smoke_case_1"])


if __name__ == "__main__":
    unittest.main()
