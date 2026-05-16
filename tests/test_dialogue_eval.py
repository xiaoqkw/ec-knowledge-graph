import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.dialogue_eval import (
    FixtureRetriever,
    RecordingRuntime,
    _is_runtime_success_turn,
    compute_summary,
    evaluate_retrieval_outcome,
    evaluate_state_transition,
    evaluate_tool_invocation,
    evaluate_turn,
    load_dataset,
)


class DialogueEvalTestCase(unittest.TestCase):
    def test_fixture_retriever_matches_slots_not_turn_index(self):
        fixture = {
            "search_cases": [
                {
                    "slots": {"budget_max": 5000, "use_case": "拍照"},
                    "results": [
                        {
                            "sku_id": 1,
                            "spu_id": 2,
                            "sku_name": "A",
                            "spu_name": "A",
                            "brand": "Apple",
                            "price": 4999,
                        }
                    ],
                }
            ]
        }
        retriever = FixtureRetriever(fixture)
        result = retriever.search({"use_case": "拍照", "budget_max": 5000, "brand": None})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].brand, "Apple")

    def test_state_transition_and_tool_invocation_are_separate(self):
        turn = {
            "expect_action": "recommend",
            "expect_mode": "dialogue",
            "expect_filled_slots": {"budget_max": 5000},
            "expected_tool_counts": {"product_search_tool": 1},
        }
        response = {
            "mode": "dialogue",
            "action": "recommend",
            "state": {"filled_slots": {"budget_max": 5000}, "pending_slots": []},
            "recommendations": [],
        }
        self.assertEqual(evaluate_state_transition(turn, response), [])
        self.assertEqual(evaluate_tool_invocation(turn, {"product_search_tool": 2}), ["tool_invocation_mismatch"])

    def test_compare_action_mismatch_is_classified_as_compare(self):
        response = {
            "mode": "dialogue",
            "action": "recommend",
            "state": {"filled_slots": {}, "pending_slots": []},
            "recommendations": [],
        }
        turn = {"expect_action": "compare", "expect_mode": "dialogue"}
        evidence, primary = evaluate_turn({}, turn, response, {})
        self.assertIn("compare", evidence)
        self.assertEqual(primary, "compare")

    def test_retrieval_outcome_detects_brand_mismatch(self):
        turn = {"expected_brands_in_top3": ["苹果"]}
        response = {
            "recommendations": [{"brand": "华为"}],
            "state": {"filled_slots": {}, "pending_slots": []},
            "mode": "dialogue",
            "action": "recommend",
        }
        self.assertEqual(evaluate_retrieval_outcome(turn, response), ["retrieval"])

    def test_compute_summary_uses_runtime_success_turn(self):
        tasks = [{"task_id": "t1"}]
        records = [
            {
                "task_id": "t1",
                "turn_index": 1,
                "passed": True,
                "state_transition_passed": True,
                "tool_invocation_passed": True,
                "success_turn_reached": False,
                "expected_action": "ask_slot",
                "recommendation_count": 0,
                "action": "ask_slot",
                "mode": "dialogue",
                "primary_cause": None,
            },
            {
                "task_id": "t1",
                "turn_index": 2,
                "passed": True,
                "state_transition_passed": True,
                "tool_invocation_passed": True,
                "success_turn_reached": True,
                "expected_action": "recommend",
                "recommendation_count": 1,
                "action": "recommend",
                "mode": "dialogue",
                "primary_cause": None,
            },
        ]
        summary = compute_summary(records, tasks)
        self.assertEqual(summary["task_success_rate"], 1.0)
        self.assertEqual(summary["avg_turns_to_success"], 2.0)
        self.assertIn("state_transition_correct_rate", summary)
        self.assertIn("tool_invocation_correct_rate", summary)

    def test_compute_summary_treats_all_passed_nonterminal_task_as_success(self):
        tasks = [{"task_id": "t1"}]
        records = [
            {
                "task_id": "t1",
                "turn_index": 1,
                "passed": True,
                "state_transition_passed": True,
                "tool_invocation_passed": True,
                "success_turn_reached": False,
                "expected_action": "ask_slot",
                "recommendation_count": 0,
                "action": "ask_slot",
                "mode": "dialogue",
                "primary_cause": None,
            },
            {
                "task_id": "t1",
                "turn_index": 2,
                "passed": True,
                "state_transition_passed": True,
                "tool_invocation_passed": True,
                "success_turn_reached": False,
                "expected_action": "ask_slot",
                "recommendation_count": 0,
                "action": "ask_slot",
                "mode": "dialogue",
                "primary_cause": None,
            },
        ]
        summary = compute_summary(records, tasks)
        self.assertEqual(summary["task_success_rate"], 1.0)
        self.assertEqual(summary["avg_turns_to_success"], 2.0)

    def test_runtime_success_turn_requires_terminal_action_and_no_evidence(self):
        turn = {"expect_action": "recommend"}
        response = {"action": "recommend"}
        self.assertTrue(_is_runtime_success_turn(turn, response, []))
        self.assertFalse(_is_runtime_success_turn(turn, response, ["retrieval"]))

    def test_recording_runtime_collects_tool_counts(self):
        runtime = RecordingRuntime()
        runtime.begin_turn()
        runtime._turn_tool_calls.extend(["product_search_tool", "product_search_tool", "price_floor_tool"])
        self.assertEqual(
            runtime.consume_turn_tool_counts(),
            {"product_search_tool": 2, "price_floor_tool": 1},
        )

    def test_dialogue_task_dataset_has_30_tasks_and_7_categories(self):
        dataset = load_dataset(ROOT_DIR / "data" / "eval" / "dialogue_tasks.jsonl")
        self.assertEqual(len(dataset), 30)
        categories = {}
        for item in dataset:
            categories[item["category"]] = categories.get(item["category"], 0) + 1
            for turn in item["turns"]:
                self.assertIn("expected_tool_counts", turn)
        self.assertEqual(
            categories,
            {
                "slot_filling": 5,
                "budget_expression": 4,
                "brand_storage": 5,
                "budget_confirmation": 5,
                "compare": 4,
                "fallback": 3,
                "reset": 4,
            },
        )

    def test_load_dataset_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tasks.jsonl"
            path.write_text(json.dumps({"task_id": "t1"}) + "\n", encoding="utf-8")
            dataset = load_dataset(path)
            self.assertEqual(dataset[0]["task_id"], "t1")


if __name__ == "__main__":
    unittest.main()
