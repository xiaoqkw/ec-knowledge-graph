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

from eval.entity_linking_eval import (
    ANALYSIS_DIR,
    build_summary_payload,
    cleanup_auxiliary_outputs,
    cleanup_stale_baseline_outputs,
    compute_metrics,
    evaluate_hybrid_rerank_baseline,
    get_candidates_for_rerank,
    write_markdown_report,
    write_summary,
)


class StubService:
    def __init__(self, results: dict[tuple[str, str], list[str]]):
        self.results = results
        self.calls = []

    def _search_entities(self, label: str, query: str, mode: str, k: int) -> list[str]:
        self.calls.append((label, query, mode, k))
        return list(self.results.get((query, mode), []))


class EntityLinkingEvalTestCase(unittest.TestCase):
    def test_compute_metrics_includes_sku_subset(self):
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
                "label": "SKU",
                "gold": "Apple iPhone 16 Pro 256GB 黑色",
                "candidates": ["Apple iPhone 16 Pro 128GB 黑色", "Apple iPhone 16 Pro 256GB 黑色"],
                "top1_hit": False,
                "topk_hit": True,
            },
        ]
        metrics = compute_metrics(records, top_k=5)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["top_k"], 5)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["topk_recall"], 1.0)
        self.assertEqual(metrics["sku_subset_total"], 1)
        self.assertAlmostEqual(metrics["sku_subset_top1_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["sku_subset_topk_recall"], 1.0)

    def test_get_candidates_for_rerank_uses_fulltext_and_hybrid_union(self):
        service = StubService(
            {
                ("q", "fulltext"): ["a", "b", "c"],
                ("q", "hybrid"): ["b", "d"],
            }
        )
        payload = get_candidates_for_rerank(service, "q", label="SKU")
        self.assertEqual(payload["fulltext_candidates"], ["a", "b", "c"])
        self.assertEqual(payload["hybrid_candidates"], ["b", "d"])
        self.assertEqual(payload["union_candidates"], ["a", "b", "c", "d"])
        self.assertEqual(
            service.calls,
            [("SKU", "q", "fulltext", 20), ("SKU", "q", "hybrid", 20)],
        )

    def test_evaluate_hybrid_rerank_reuses_hybrid_cache_for_all_run(self):
        dataset = [
            {"query": "sku q", "label": "SKU", "gold": "Phone 256GB 黑色"},
            {"query": "spu q", "label": "SPU", "gold": "SPU Gold"},
        ]
        service = StubService(
            {
                ("sku q", "fulltext"): ["Phone 128GB 黑色", "Phone 256GB 黑色"],
                ("sku q", "hybrid"): ["Phone 128GB 黑色", "Phone 256GB 黑色"],
            }
        )
        hybrid_cache = [
            {
                "query": "sku q",
                "label": "SKU",
                "gold": "Phone 256GB 黑色",
                "candidates": ["Phone 128GB 黑色", "Phone 256GB 黑色"],
                "top1_hit": False,
                "topk_hit": True,
            },
            {
                "query": "spu q",
                "label": "SPU",
                "gold": "SPU Gold",
                "candidates": ["SPU Gold"],
                "top1_hit": True,
                "topk_hit": True,
            },
        ]
        metrics, records = evaluate_hybrid_rerank_baseline(service, dataset, top_k=2, hybrid_reference_cache=hybrid_cache)
        self.assertEqual(len(service.calls), 2)
        self.assertEqual(records[1]["candidates"], ["SPU Gold"])
        self.assertEqual(records[0]["hybrid_top1"], "Phone 128GB 黑色")
        self.assertEqual(metrics["baseline_notes"]["non_sku_behavior"], "reuse_hybrid")

    def test_write_summary_keeps_executed_baselines_and_auxiliary_artifacts(self):
        payload = {
            "executed_baselines": ["hybrid_rerank"],
            "auxiliary_artifacts": {
                "markdown_generated": False,
                "reranker_analysis_generated": True,
                "reranker_reference_mode": "internal_hybrid_reference",
            },
            "hybrid_rerank": {
                "metrics": {
                    "total": 2,
                    "top1_accuracy": 0.5,
                    "top_k": 3,
                    "topk_recall": 1.0,
                    "by_label_accuracy": {"SKU": 0.5},
                    "sku_subset_total": 1,
                    "sku_subset_top1_accuracy": 1.0,
                    "sku_subset_topk_recall": 1.0,
                    "baseline_notes": {"scope": "sku_only"},
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            write_summary(summary_path, payload)
            parsed = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["executed_baselines"], ["hybrid_rerank"])
        self.assertTrue(parsed["auxiliary_artifacts"]["reranker_analysis_generated"])
        self.assertEqual(parsed["hybrid_rerank"]["metrics"]["sku_subset_total"], 1)

    def test_build_summary_payload_uses_actual_auxiliary_artifact_state(self):
        results = {
            "hybrid_rerank": {
                "metrics": {
                    "total": 1,
                    "top1_accuracy": 1.0,
                    "top_k": 3,
                    "topk_recall": 1.0,
                    "by_label_accuracy": {"SKU": 1.0},
                    "sku_subset_total": 1,
                    "sku_subset_top1_accuracy": 1.0,
                    "sku_subset_topk_recall": 1.0,
                    "baseline_notes": {"scope": "sku_only"},
                }
            }
        }
        payload = build_summary_payload(
            baselines=["hybrid_rerank"],
            results=results,
            auxiliary_artifacts={
                "markdown_generated": False,
                "reranker_analysis_generated": True,
                "reranker_reference_mode": "internal_hybrid_reference",
            },
        )
        self.assertFalse(payload["auxiliary_artifacts"]["markdown_generated"])
        self.assertTrue(payload["auxiliary_artifacts"]["reranker_analysis_generated"])

    def test_write_markdown_report_only_shows_main_three_rows(self):
        payload = {
            "fulltext": {
                "metrics": {
                    "sku_subset_total": 20,
                    "sku_subset_top1_accuracy": 0.7,
                    "sku_subset_topk_recall": 0.8,
                    "baseline_notes": {
                        "retrieval_view": "neo4j_fulltext_index_query",
                        "ranking_signal": "fulltext_index_score_desc",
                    },
                }
            },
            "hybrid": {
                "metrics": {
                    "sku_subset_total": 20,
                    "sku_subset_top1_accuracy": 0.65,
                    "sku_subset_topk_recall": 0.9,
                    "baseline_notes": {
                        "retrieval_view": "neo4jvector_hybrid_search",
                        "ranking_signal": "returned_order_from_similarity_search",
                    },
                }
            },
            "hybrid_rerank": {
                "metrics": {
                    "sku_subset_total": 20,
                    "sku_subset_top1_accuracy": 0.75,
                    "sku_subset_topk_recall": 0.95,
                    "baseline_notes": {
                        "sku_candidate_pool": "fulltext_top20_plus_hybrid_top20_union_dedup_preserve_order",
                        "sku_ranking_method": "rule_reranker_weighted_sum",
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "summary.md"
            write_markdown_report(path, payload)
            content = path.read_text(encoding="utf-8")
        self.assertIn("| fulltext |", content)
        self.assertIn("| hybrid |", content)
        self.assertIn("| hybrid_rerank |", content)
        self.assertNotIn("| exact_match |", content)

    def test_cleanup_auxiliary_outputs_only_removes_fixed_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            eval_dir = root / "eval"
            analysis_dir = eval_dir / "sku_reranker"
            analysis_dir.mkdir(parents=True)
            markdown = eval_dir / "entity_linking_summary_memory_off.md"
            markdown.write_text("x", encoding="utf-8")
            for filename in ("rerank_helped.jsonl", "rerank_hurt.jsonl", "rerank_still_wrong.jsonl", "ablation.jsonl"):
                (analysis_dir / filename).write_text("x", encoding="utf-8")
            with patch("eval.entity_linking_eval.LOG_DIR", root), patch("eval.entity_linking_eval.ANALYSIS_DIR", analysis_dir):
                cleanup_auxiliary_outputs(memory_suffix="memory_off")
            self.assertFalse(markdown.exists())
            self.assertFalse((analysis_dir / "rerank_helped.jsonl").exists())
            self.assertFalse((analysis_dir / "rerank_hurt.jsonl").exists())
            self.assertFalse((analysis_dir / "rerank_still_wrong.jsonl").exists())
            self.assertTrue((analysis_dir / "ablation.jsonl").exists())

    def test_cleanup_stale_baseline_outputs_only_removes_unexecuted_baselines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            for baseline in ("exact_match", "fulltext", "hybrid", "hybrid_rerank"):
                (eval_dir / f"entity_linking_{baseline}_memory_off.jsonl").write_text("x", encoding="utf-8")
            with patch("eval.entity_linking_eval.LOG_DIR", root):
                cleanup_stale_baseline_outputs(executed_baselines=["hybrid_rerank"], memory_suffix="memory_off")
            self.assertTrue((eval_dir / "entity_linking_hybrid_rerank_memory_off.jsonl").exists())
            self.assertFalse((eval_dir / "entity_linking_exact_match_memory_off.jsonl").exists())
            self.assertFalse((eval_dir / "entity_linking_fulltext_memory_off.jsonl").exists())
            self.assertFalse((eval_dir / "entity_linking_hybrid_memory_off.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
