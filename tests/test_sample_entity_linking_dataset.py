import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.sample_entity_linking_dataset import sample_by_label


class SampleEntityLinkingDatasetTestCase(unittest.TestCase):
    def test_sample_by_label_returns_balanced_rows(self):
        dataset = []
        for label in ("SKU", "SPU", "Trademark"):
            for index in range(5):
                dataset.append({"label": label, "query": f"{label}-{index}", "gold": f"{label}-gold-{index}"})
        sampled = sample_by_label(dataset, per_label=2, seed=42)
        counts = {}
        for item in sampled:
            counts[item["label"]] = counts.get(item["label"], 0) + 1
        self.assertEqual(counts, {"SKU": 2, "SPU": 2, "Trademark": 2})

    def test_sample_by_label_raises_when_label_is_too_small(self):
        dataset = [
            {"label": "SKU", "query": "a", "gold": "g1"},
            {"label": "SPU", "query": "b", "gold": "g2"},
        ]
        with self.assertRaises(ValueError):
            sample_by_label(dataset, per_label=2, seed=42)


if __name__ == "__main__":
    unittest.main()
