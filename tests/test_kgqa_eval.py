import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.kgqa_eval import answer_keyword_hit, compute_metrics, coverage_flags, has_answer_keywords, write_summary


class KGQAEvalTestCase(unittest.TestCase):
    def test_answer_keyword_hit_uses_case_insensitive_substring(self):
        self.assertTrue(answer_keyword_hit("推荐 Apple iPhone 16 Pro", ["iphone"]))
        self.assertTrue(answer_keyword_hit("推荐 Apple iPhone 16 Pro", ["IPHONE 16"]))
        self.assertFalse(answer_keyword_hit("推荐华为 Mate 40 Pro", ["iPhone"]))

    def test_coverage_flags_distinguish_any_and_all(self):
        any_hit, all_hit = coverage_flags(["苹果", "手机"], {"param_0": "苹果"})
        self.assertTrue(any_hit)
        self.assertFalse(all_hit)

    def test_has_answer_keywords(self):
        self.assertTrue(has_answer_keywords(["iphone"]))
        self.assertFalse(has_answer_keywords([]))
        self.assertFalse(has_answer_keywords(["", "   "]))

    def test_compute_metrics(self):
        records = [
            {
                "must_execute": True,
                "has_answer_keywords": True,
                "parse_success_raw": True,
                "parse_success_repaired": True,
                "cypher_query_present": True,
                "execution_success": True,
                "non_empty_result": True,
                "answer_keyword_hit": False,
                "unsafe_cypher": False,
                "entity_any_coverage_hit": True,
                "entity_all_coverage_hit": False,
            },
            {
                "must_execute": False,
                "has_answer_keywords": False,
                "parse_success_raw": False,
                "parse_success_repaired": True,
                "cypher_query_present": False,
                "execution_success": False,
                "non_empty_result": False,
                "answer_keyword_hit": True,
                "unsafe_cypher": True,
                "entity_any_coverage_hit": False,
                "entity_all_coverage_hit": False,
            },
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["must_execute_total"], 1)
        self.assertEqual(metrics["answer_keyword_total"], 1)
        self.assertAlmostEqual(metrics["raw_json_parse_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["repaired_json_parse_success_rate"], 1.0)
        self.assertAlmostEqual(metrics["cypher_query_present_rate"], 0.5)
        self.assertAlmostEqual(metrics["cypher_execution_success_rate"], 1.0)
        self.assertAlmostEqual(metrics["non_empty_result_rate"], 1.0)
        self.assertAlmostEqual(metrics["answer_keyword_hit_rate"], 0.0)
        self.assertAlmostEqual(metrics["unsafe_cypher_rate"], 0.5)
        self.assertAlmostEqual(metrics["entity_any_coverage_rate"], 0.5)
        self.assertAlmostEqual(metrics["entity_all_coverage_rate"], 0.0)

    def test_write_summary(self):
        results = {
            "full": {
                "total": 2,
                "must_execute_total": 1,
                "answer_keyword_total": 1,
                "raw_json_parse_success_rate": 0.5,
                "repaired_json_parse_success_rate": 1.0,
                "cypher_query_present_rate": 0.5,
                "cypher_execution_success_rate": 1.0,
                "non_empty_result_rate": 1.0,
                "answer_keyword_hit_rate": 0.5,
                "unsafe_cypher_rate": 0.0,
                "entity_any_coverage_rate": 0.5,
                "entity_all_coverage_rate": 0.0,
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            write_summary(summary_path, results)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["full"]["must_execute_total"], 1)


if __name__ == "__main__":
    unittest.main()
