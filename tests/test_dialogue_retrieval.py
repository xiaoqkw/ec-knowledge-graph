import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dialogue.retrieval import PhoneGuideRetriever
from dialogue.types import RecommendationItem


class DialogueRetrievalTestCase(unittest.TestCase):
    def test_collect_storage_options_merges_all_sku_variants(self):
        rows = [
            {"storage_values": ["128G"]},
            {"storage_values": ["256G"]},
            {"storage_values": ["128G", "512G"]},
        ]
        self.assertEqual(
            PhoneGuideRetriever._collect_storage_options(rows),
            ["128G", "256G", "512G"],
        )

    def test_score_item_does_not_depend_on_reason_text(self):
        retriever = object.__new__(PhoneGuideRetriever)
        first = RecommendationItem(
            sku_id=1,
            spu_id=1,
            sku_name="拍照旗舰手机",
            spu_name="拍照旗舰手机",
            brand="苹果",
            price=6000,
            reason="拍照相关描述更强",
            source_text="拍照旗舰手机 强悍影像系统",
        )
        second = RecommendationItem(
            sku_id=1,
            spu_id=1,
            sku_name="拍照旗舰手机",
            spu_name="拍照旗舰手机",
            brand="苹果",
            price=6000,
            reason="完全不同的说明文案",
            source_text="拍照旗舰手机 强悍影像系统",
        )
        slots = {"use_case": "拍照", "budget_max": 8000, "storage": None}
        self.assertEqual(
            retriever._score_item(first, slots),
            retriever._score_item(second, slots),
        )

    def test_score_item_uses_description_text(self):
        retriever = object.__new__(PhoneGuideRetriever)
        weak = RecommendationItem(
            sku_id=1,
            spu_id=1,
            sku_name="手机A",
            spu_name="手机A",
            brand="苹果",
            price=5000,
            reason="test",
            source_text="手机A 普通描述",
        )
        strong = RecommendationItem(
            sku_id=2,
            spu_id=2,
            sku_name="手机B",
            spu_name="手机B",
            brand="苹果",
            price=5000,
            reason="test",
            source_text="手机B 影像 自拍 镜头 表现突出",
        )
        slots = {"use_case": "拍照", "budget_max": 8000, "storage": None}
        self.assertGreater(
            retriever._score_item(strong, slots),
            retriever._score_item(weak, slots),
        )


if __name__ == "__main__":
    unittest.main()
