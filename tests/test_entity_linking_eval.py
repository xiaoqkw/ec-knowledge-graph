import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.entity_linking_eval import compute_metrics, write_summary


class EntityLinkingEvalTestCase(unittest.TestCase):
    def test_compute_metrics(self):
        records = [
            {
                "query": "苹果",
                "label": "Trademark",
                "gold": "苹果",
                "candidates": ["苹果", "华为"],
                "top1_hit": True,
                "topk_hit": True,
            },
            {
                "query": "iphone 16 pro",
                "label": "SPU",
                "gold": "Apple iPhone 16 Pro",
                "candidates": ["Apple iPhone 12", "Apple iPhone 16 Pro"],
                "top1_hit": False,
                "topk_hit": True,
            },
        ]
        metrics = compute_metrics(records, top_k=5)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["top_k"], 5)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["topk_recall"], 1.0)
        self.assertAlmostEqual(metrics["by_label_accuracy"]["Trademark"], 1.0)
        self.assertAlmostEqual(metrics["by_label_accuracy"]["SPU"], 0.0)

    def test_write_summary(self):
        results = {
            "hybrid": {
                "metrics": {
                    "total": 2,
                    "top1_accuracy": 0.5,
                    "top_k": 3,
                    "topk_recall": 1.0,
                    "by_label_accuracy": {"SPU": 0.0},
                },
                "records": [],
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            write_summary(summary_path, results)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["hybrid"]["total"], 2)
        self.assertEqual(payload["hybrid"]["top_k"], 3)


if __name__ == "__main__":
    unittest.main()
