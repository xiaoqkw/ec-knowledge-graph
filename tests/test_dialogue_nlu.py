import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dialogue.nlu import DialogueNLU


class DialogueNLUTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.nlu = DialogueNLU(llm_enabled=False)
        self.brands = ["苹果", "红米", "华为", "OPPO", "VIVO", "小米"]

    def test_extract_budget_brand_storage_and_use_case(self):
        result = self.nlu.parse(
            "想买手机，预算 8000 以内，想要苹果，256G，更看重拍照",
            brand_vocabulary=self.brands,
            state_has_context=False,
        )
        self.assertEqual(result.intent, "recommend")
        self.assertEqual(result.slots["budget_max"], 8000)
        self.assertEqual(result.slots["brand"], "苹果")
        self.assertEqual(result.slots["storage"], "256G")
        self.assertEqual(result.slots["use_case"], "拍照")

    def test_extract_budget_from_first_turn_phone_context(self):
        result = self.nlu.parse(
            "想买手机，4k",
            brand_vocabulary=self.brands,
            state_has_context=False,
        )
        self.assertEqual(result.slots["budget_max"], 4000)

        result = self.nlu.parse(
            "想买手机，4000",
            brand_vocabulary=self.brands,
            state_has_context=False,
        )
        self.assertEqual(result.slots["budget_max"], 4000)

    def test_detect_reset(self):
        result = self.nlu.parse(
            "重新开始",
            brand_vocabulary=self.brands,
            state_has_context=True,
        )
        self.assertEqual(result.intent, "reset")

    def test_extract_budget_from_context_only_amount(self):
        result = self.nlu.parse(
            "4000",
            brand_vocabulary=self.brands,
            state_has_context=True,
        )
        self.assertEqual(result.slots["budget_max"], 4000)

        result = self.nlu.parse(
            "4k",
            brand_vocabulary=self.brands,
            state_has_context=True,
        )
        self.assertEqual(result.slots["budget_max"], 4000)

    def test_compare_keeps_new_slots(self):
        result = self.nlu.parse(
            "把前两个比一下，我更看重拍照",
            brand_vocabulary=self.brands,
            state_has_context=True,
        )
        self.assertEqual(result.intent, "compare")
        self.assertEqual(result.slots["use_case"], "拍照")

        result = self.nlu.parse(
            "比较一下，更想要苹果",
            brand_vocabulary=self.brands,
            state_has_context=True,
        )
        self.assertEqual(result.intent, "compare")
        self.assertEqual(result.slots["brand"], "苹果")

    def test_storage_phrase_does_not_override_existing_budget(self):
        result = self.nlu.parse(
            "苹果 256G",
            brand_vocabulary=self.brands,
            state_has_context=True,
        )
        self.assertEqual(result.slots["brand"], "苹果")
        self.assertEqual(result.slots["storage"], "256G")
        self.assertNotIn("budget_max", result.slots)

    def test_fallback_for_non_dialogue_query(self):
        result = self.nlu.parse(
            "Apple 都有哪些产品",
            brand_vocabulary=self.brands,
            state_has_context=False,
        )
        self.assertEqual(result.intent, "fallback_qa")

    def test_fallback_qa_is_confident_without_llm_override(self):
        class GuardedNLU(DialogueNLU):
            def _parse_with_llm(self, message, brand_vocabulary):
                raise AssertionError("LLM fallback should not be called for confident fallback_qa")

        nlu = GuardedNLU(llm_enabled=False)
        result = nlu.parse(
            "苹果都有哪些产品",
            brand_vocabulary=self.brands,
            state_has_context=False,
        )
        self.assertEqual(result.intent, "fallback_qa")


if __name__ == "__main__":
    unittest.main()
