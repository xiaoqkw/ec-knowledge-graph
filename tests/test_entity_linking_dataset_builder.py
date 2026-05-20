import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval import build_entity_linking_dataset as builder
from agent.tools.entity_link_tool import build_safe_fulltext_query


class StubSearchBackend:
    def __init__(self, pools):
        self.pools = pools
        self.closed = False
        self.hybrid_error = None

    def close(self) -> None:
        self.closed = True

    def search(self, label: str, query: str, *, mode: str, top_k: int) -> list[str]:
        gold = self.pools[label][0]["name"]
        if mode == "exact_match":
            return [gold] if builder.normalize_query(query) == builder.normalize_query(gold) else []
        return [gold]

    def search_bundle(self, label: str, query: str) -> dict[str, list[str]]:
        return {
            "exact_match": self.search(label, query, mode="exact_match", top_k=1),
            "fulltext": self.search(label, query, mode="fulltext", top_k=20),
            "hybrid": self.search(label, query, mode="hybrid", top_k=20),
        }


class EntityLinkingDatasetBuilderTestCase(unittest.TestCase):
    def test_build_safe_fulltext_query_escapes_lucene_special_chars(self):
        escaped = build_safe_fulltext_query("蛋白粉/200克 (草莓味)")
        self.assertIn("\\/", escaped)
        self.assertIn("\\(", escaped)
        self.assertIn("\\)", escaped)

    def test_resolve_local_embedding_model_name_prefers_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_home = Path(tmp_dir)
            snapshot = fake_home / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-base-zh-v1.5" / "snapshots" / "snap_1"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "pytorch_model.bin").write_text("x", encoding="utf-8")
            with patch.object(builder.Path, "home", return_value=fake_home):
                resolved = builder.resolve_local_embedding_model_name("BAAI/bge-base-zh-v1.5")
        self.assertEqual(resolved, str(snapshot))

    def test_scale_targets_from_total_preserves_ratio(self):
        targets = builder.scale_targets_from_total(1001, builder.DEFAULT_RATIO)
        self.assertEqual(sum(targets.values()), 1001)
        self.assertEqual(targets["SKU"], 308)
        self.assertEqual(targets["Trademark"], 231)
        self.assertEqual(targets["SPU"], 231)
        self.assertEqual(targets["Category3"], 231)

    def test_build_local_variants_includes_compact_and_reordered_forms(self):
        variants = builder.build_local_variants(
            "SKU",
            "Apple iPhone 12 (A2404) 128GB 黑色 支持移动联通电信5G 双卡双待手机",
            "apple iphone 12 a2404 128gb 黑色",
            {},
        )
        queries = [item["query"] for item in variants]
        self.assertIn("appleiphone12a2404128gb黑色", [builder.normalize_query(item) for item in queries])
        self.assertTrue(any("a2404" in query.lower() and "128gb" in query.lower() for query in queries))

    def test_generate_dataset_preserves_v1_and_adds_new_variants(self):
        old_records = [
            {"query": "caremille", "label": "Trademark", "gold": "CAREMiLLE"},
            {"query": "iphone 12", "label": "SPU", "gold": "Apple iPhone 12"},
            {"query": "手机 耳机", "label": "Category3", "gold": "手机耳机"},
            {
                "query": "apple iphone 12 a2404 128gb 黑色",
                "label": "SKU",
                "gold": "Apple iPhone 12 (A2404) 128GB 黑色 支持移动联通电信5G 双卡双待手机",
            },
        ]
        pools = {
            "Trademark": [{"entity_id": 1, "name": "CAREMiLLE"}],
            "SPU": [{"entity_id": 2, "name": "Apple iPhone 12"}],
            "Category3": [{"entity_id": 3, "name": "手机耳机"}],
            "SKU": [{"entity_id": 4, "name": "Apple iPhone 12 (A2404) 128GB 黑色 支持移动联通电信5G 双卡双待手机"}],
        }
        stub_searcher = StubSearchBackend(pools)

        with (
            patch.object(builder, "load_dataset", return_value=old_records),
            patch.object(builder, "load_entity_pools", return_value=pools),
            patch.object(builder, "load_seed_queries", return_value={
                "Trademark": [{"query": "caremille", "gold": "CAREMiLLE", "source": "entity_linking_v1"}],
                "SPU": [{"query": "iphone 12", "gold": "Apple iPhone 12", "source": "entity_linking_v1"}],
                "Category3": [{"query": "手机 耳机", "gold": "手机耳机", "source": "entity_linking_v1"}],
                "SKU": [{"query": "apple iphone 12 a2404 128gb 黑色", "gold": "Apple iPhone 12 (A2404) 128GB 黑色 支持移动联通电信5G 双卡双待手机", "source": "entity_linking_v1"}],
            }),
            patch.object(builder, "SearchBackend", return_value=stub_searcher),
            patch.object(builder, "generate_llm_variants", return_value=[]),
        ):
            records, review_records, summary = builder.generate_dataset(
                dataset_path=ROOT_DIR / "data" / "eval" / "entity_linking.jsonl",
                target_by_label={"SKU": 2, "Trademark": 2, "SPU": 2, "Category3": 2},
                llm_enabled=False,
                max_llm_items_per_seed=1,
                llm_workers=1,
                validation_workers=1,
                limit_by_label=None,
            )

        self.assertTrue(stub_searcher.closed)
        self.assertGreaterEqual(len(records), 8)
        self.assertEqual(len(review_records), 0)
        self.assertEqual(summary["counts"]["SKU"], 2)
        self.assertEqual(summary["counts"]["Trademark"], 2)
        self.assertEqual(summary["counts"]["SPU"], 2)
        self.assertEqual(summary["counts"]["Category3"], 2)
        self.assertTrue(any(item.get("metadata", {}).get("preserved_from_v1") for item in records))
        self.assertTrue(any(item["variant_type"] == "heuristic" for item in records))


if __name__ == "__main__":
    unittest.main()
