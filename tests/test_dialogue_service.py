import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dialogue.service import DialogueService
from dialogue.state import InMemorySessionStore
from dialogue.types import RecommendationItem


class StubNLU:
    def __init__(self):
        self.outputs = []

    def queue(self, intent, slots=None):
        self.outputs.append({"intent": intent, "slots": slots or {}})

    def parse(self, message, *, brand_vocabulary, state_has_context):
        payload = self.outputs.pop(0)
        return type("NLUResult", (), payload)


class StubRetriever:
    def __init__(self):
        self.calls = []
        self.results = {}
        self.compare_text = "comparison"
        self.min_price_value = 8197

    def load_brand_vocabulary(self):
        return ["苹果", "红米", "华为", "OPPO", "VIVO", "小米"]

    def load_storage_vocabulary(self):
        return ["128G", "256G"]

    def search(self, slots, limit=3):
        normalized = tuple(sorted((key, value) for key, value in slots.items() if value is not None))
        self.calls.append(dict(slots))
        return self.results.get(normalized, [])

    def compare(self, spu_ids, use_case):
        return f"comparison:{use_case}"

    def get_min_price(self, brand=None):
        return self.min_price_value

    def close(self):
        return None


def build_item(sku_id, spu_id, name, price):
    return RecommendationItem(
        sku_id=sku_id,
        spu_id=spu_id,
        sku_name=name,
        spu_name=name,
        brand="苹果" if "iPhone" in name else "红米",
        price=price,
        reason="test",
        storage_options=["128G", "256G"],
        source_text=name,
    )


class DialogueServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.nlu = StubNLU()
        self.retriever = StubRetriever()
        self.service = DialogueService(
            store=InMemorySessionStore(),
            nlu=self.nlu,
            retriever=self.retriever,
            llm_enabled=False,
        )

    def test_ask_required_slots_in_order(self):
        self.nlu.queue("recommend", {})
        response = self.service.chat("想买手机")
        self.assertEqual(response["action"], "ask_slot")
        self.assertIn("预算", response["message"])

        self.nlu.queue("inform", {"budget_max": 5000})
        response = self.service.chat("5000 以内", response["session_id"])
        self.assertEqual(response["action"], "ask_slot")
        self.assertIn("更看重哪一方面", response["message"])

    def test_relax_storage_when_no_exact_match(self):
        relaxed_key = (
            ("brand", "苹果"),
            ("budget_max", 8000),
            ("use_case", "拍照"),
        )
        self.retriever.results[relaxed_key] = [
            build_item(42, 19, "iPhone 16 Pro", 6961),
        ]

        self.nlu.queue(
            "recommend",
            {"budget_max": 8000, "brand": "苹果", "storage": "256G", "use_case": "拍照"},
        )
        response = self.service.chat("苹果 256G 拍照")
        self.assertEqual(response["action"], "recommend")
        self.assertIn("放宽", response["message"])
        self.assertEqual(response["recommendations"][0]["sku_id"], 42)
        self.assertEqual(self.retriever.calls[0]["storage"], "256G")
        self.assertIsNone(self.retriever.calls[1]["storage"])

    def test_compare_uses_last_spu_candidates_and_new_slots(self):
        recommend_key = (
            ("budget_max", 8000),
            ("use_case", "拍照"),
        )
        self.retriever.results[recommend_key] = [
            build_item(42, 19, "iPhone 16 Pro", 6961),
            build_item(41, 18, "Mate 40 Pro", 1999),
        ]

        self.nlu.queue("recommend", {"budget_max": 8000, "use_case": "拍照"})
        first = self.service.chat("预算 8000，拍照")
        self.assertEqual(first["action"], "recommend")

        self.nlu.queue("compare", {"brand": "苹果"})
        second = self.service.chat("把前两个比一下，我更想要苹果", first["session_id"])
        self.assertEqual(second["action"], "compare")
        self.assertEqual(second["message"], "comparison:拍照")
        self.assertEqual(
            self.service.store.snapshot(first["session_id"]).slots["brand"],
            "苹果",
        )

    def test_fallback_qa_returns_qa_fallback_mode(self):
        self.nlu.queue("fallback_qa", {})
        response = self.service.chat("Apple 都有哪些产品", qa_handler=lambda _: "qa result")
        self.assertEqual(response["action"], "fallback_qa")
        self.assertEqual(response["mode"], "qa_fallback")
        self.assertEqual(response["message"], "qa result")

    def test_accept_budget_suggestion_on_follow_up(self):
        session = self.service.store.get_or_create(None)
        session.slots.update({"brand": "苹果", "budget_max": 4000, "use_case": "拍照"})
        session.awaiting_budget_confirmation = True
        session.suggested_budget_min = 8197
        self.service.store.save(session)

        expected_key = (
            ("brand", "苹果"),
            ("budget_max", 8197),
            ("use_case", "拍照"),
        )
        self.retriever.results[expected_key] = [
            build_item(8, 3, "iPhone 12", 8197),
        ]

        self.nlu.queue("inform", {})
        response = self.service.chat("帮我筛一下吧", session.session_id)
        self.assertEqual(response["action"], "recommend")
        self.assertEqual(response["recommendations"][0]["sku_id"], 8)
        self.assertFalse(self.service.store.snapshot(session.session_id).awaiting_budget_confirmation)

    def test_reject_budget_suggestion_on_negative_reply(self):
        session = self.service.store.get_or_create(None)
        session.slots.update({"brand": "苹果", "budget_max": 4000, "use_case": "拍照"})
        session.awaiting_budget_confirmation = True
        session.suggested_budget_min = 8197
        self.service.store.save(session)

        self.nlu.queue("inform", {})
        response = self.service.chat("不可以，太贵了", session.session_id)
        self.assertEqual(response["action"], "ask_slot")
        self.assertEqual(self.retriever.calls[-1]["budget_max"], 4000)
        snapshot = self.service.store.snapshot(session.session_id)
        self.assertEqual(snapshot.slots["budget_max"], 4000)
        self.assertTrue(snapshot.awaiting_budget_confirmation)

    def test_accept_budget_suggestion_keeps_new_slots(self):
        session = self.service.store.get_or_create(None)
        session.slots.update({"brand": "苹果", "budget_max": 4000, "use_case": "拍照"})
        session.awaiting_budget_confirmation = True
        session.suggested_budget_min = 8197
        self.service.store.save(session)

        expected_key = (
            ("brand", "华为"),
            ("budget_max", 8197),
            ("use_case", "拍照"),
        )
        self.nlu.queue("inform", {"brand": "华为"})
        response = self.service.chat("帮我筛一下吧，我更想要华为", session.session_id)
        self.assertEqual(response["action"], "ask_slot")
        self.assertEqual(self.retriever.calls[-1]["brand"], "华为")
        self.assertEqual(self.retriever.calls[-1]["budget_max"], 8197)
        snapshot = self.service.store.snapshot(session.session_id)
        self.assertEqual(snapshot.slots["brand"], "华为")
        self.assertEqual(snapshot.slots["budget_max"], 8197)
        self.assertFalse(snapshot.awaiting_budget_confirmation)

    def test_compare_takes_precedence_over_budget_confirmation(self):
        session = self.service.store.get_or_create(None)
        session.slots.update({"brand": "苹果", "budget_max": 4000, "use_case": "拍照"})
        session.awaiting_budget_confirmation = True
        session.suggested_budget_min = 8197
        session.last_recommendation_spu_ids = [19, 18]
        self.service.store.save(session)

        self.nlu.queue("compare", {})
        response = self.service.chat("可以，把前两个比一下", session.session_id)
        self.assertEqual(response["action"], "compare")
        snapshot = self.service.store.snapshot(session.session_id)
        self.assertEqual(snapshot.slots["budget_max"], 4000)
        self.assertTrue(snapshot.awaiting_budget_confirmation)

    def test_new_budget_overrides_suggested_budget(self):
        session = self.service.store.get_or_create(None)
        session.slots.update({"brand": "苹果", "budget_max": 4000, "use_case": "拍照"})
        session.awaiting_budget_confirmation = True
        session.suggested_budget_min = 8197
        self.service.store.save(session)

        expected_key = (
            ("brand", "华为"),
            ("budget_max", 9000),
            ("use_case", "拍照"),
        )
        self.retriever.results[expected_key] = [
            build_item(41, 18, "Mate 40 Pro", 8999),
        ]

        self.nlu.queue("inform", {"brand": "华为", "budget_max": 9000})
        response = self.service.chat("可以，那就9000吧，我更想要华为", session.session_id)
        self.assertEqual(response["action"], "recommend")
        self.assertEqual(response["recommendations"][0]["sku_id"], 41)
        self.assertEqual(self.retriever.calls[-1]["budget_max"], 9000)
        snapshot = self.service.store.snapshot(session.session_id)
        self.assertEqual(snapshot.slots["brand"], "华为")
        self.assertEqual(snapshot.slots["budget_max"], 9000)
        self.assertTrue(snapshot.awaiting_budget_confirmation is False)


if __name__ == "__main__":
    unittest.main()
