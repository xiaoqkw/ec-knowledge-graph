import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.sku_reranker.reranker import RuleBasedSKUReranker


class SKURerankerTestCase(unittest.TestCase):
    def setUp(self):
        self.reranker = RuleBasedSKUReranker()

    def test_rerank_model_match_dominates_color_match(self):
        query = "iphone a2404 黑色"
        candidates = [
            "iphone a2405 黑色",
            "iphone a2404 白色",
        ]
        ranked = self.reranker.rerank(query, candidates)
        self.assertEqual(ranked[0], "iphone a2404 白色")

    def test_capacity_conflict_penalty_pushes_wrong_candidate_down(self):
        query = "iphone 256gb 黑色"
        candidates = [
            "iphone 128gb 黑色",
            "iphone 256gb 白色",
        ]
        ranked = self.reranker.rerank(query, candidates)
        self.assertEqual(ranked[0], "iphone 256gb 白色")

    def test_color_match_breaks_tie_with_same_model_and_capacity(self):
        query = "redmi 10x 128gb 冰雾白"
        candidates = [
            "redmi 10x 128gb 明月灰",
            "redmi 10x 128gb 冰雾白",
        ]
        ranked = self.reranker.rerank(query, candidates)
        self.assertEqual(ranked[0], "redmi 10x 128gb 冰雾白")

    def test_conflict_tokens_detect_nearby_variant_conflicts(self):
        scored = self.reranker.score_candidates(
            "i-softto z02",
            ["i-softto z03", "i-softto z02"],
        )
        self.assertGreater(scored[0].score, scored[1].score)
        self.assertEqual(scored[1].features.conflict_tokens, 1)

    def test_capacity_match_requires_model_or_brand_context(self):
        scored = self.reranker.score_candidates(
            "redmi10x 4g 4gb+128gb",
            [
                "保税！小蓝管蓝色特润~依泉唇膏柔润护唇膏4g依泉唇膏保湿不粘",
                "Redmi 10X 4G Helio G85 128GB 4GB+128GB 冰雾白",
            ],
        )
        self.assertEqual(scored[0].candidate, "Redmi 10X 4G Helio G85 128GB 4GB+128GB 冰雾白")
        self.assertFalse(scored[1].features.capacity_match)

    def test_brand_match_requires_meaningful_overlap(self):
        weak = self.reranker.score_candidate("redmi10x 4g 4gb+128gb", "保税！小蓝管蓝色特润~依泉唇膏柔润护唇膏4g依泉唇膏保湿不粘")
        strong = self.reranker.score_candidate("十月稻田 长粒香 5kg", "十月稻田 辽河长粒香 东北大米 5kg")
        self.assertFalse(weak.features.brand_match)
        self.assertTrue(strong.features.brand_match)

    def test_capacity_dimensions_do_not_cross_match(self):
        weak = self.reranker.score_candidate("十月稻田 长粒香 5kg", "TCL 65Q10 65英寸 QLED原色量子点电视 3+32GB 平板电视")
        strong = self.reranker.score_candidate("十月稻田 长粒香 5kg", "十月稻田 辽河长粒香 东北大米 5kg")
        self.assertFalse(weak.features.capacity_match)
        self.assertTrue(strong.features.capacity_match)

    def test_subject_token_gate_penalizes_unrelated_candidate(self):
        weak = self.reranker.score_candidate("tcl65q10 65 qled 3+32gb", "【满3件包邮】苏菲超熟睡裤型夜用卫生巾安心裤M码/L码 2片 *3包")
        strong = self.reranker.score_candidate("tcl65q10 65 qled 3+32gb", "TCL 65Q10 65英寸 QLED原色量子点电视 3+32GB 平板电视")
        self.assertGreaterEqual(weak.features.conflict_tokens, 1)
        self.assertLess(weak.score, strong.score)


if __name__ == "__main__":
    unittest.main()
