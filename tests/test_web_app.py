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
except ImportError:
    TestClient = None
    web = None


@unittest.skipIf(TestClient is None, "fastapi is not installed in the current environment")
class WebAppTestCase(unittest.TestCase):
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

    def test_dialogue_response_keeps_suggested_budget_min(self):
        with patch.object(
            web.app.dialogue_service,
            "chat",
            return_value={
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
            },
        ):
            client = TestClient(web.app.app)
            response = client.post(
                "/api/dialogue/chat",
                json={"message": "帮我筛一下吧", "session_id": "session-1"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["state"]["suggested_budget_min"], 8197)


if __name__ == "__main__":
    unittest.main()
