import io
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

from eval import dialogue_nlu_eval
from eval.dialogue_nlu_eval import RecordingDialogueNLU, evaluate_dataset, load_dataset


class DialogueNLUEvalTestCase(unittest.TestCase):
    def test_load_dataset_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "samples.jsonl"
            path.write_text(json.dumps({"input": "a"}) + "\n", encoding="utf-8")
            dataset = load_dataset(path)
            self.assertEqual(dataset[0]["input"], "a")

    def test_recording_dialogue_nlu_marks_llm_fallback(self):
        nlu = RecordingDialogueNLU(llm_enabled=False)
        def mark_fallback(self, message, brand_vocabulary):
            self._used_llm_fallback = True
            return None

        with patch.object(RecordingDialogueNLU, "_parse_with_llm", new=mark_fallback):
            with patch.object(
                RecordingDialogueNLU,
                "_parse_rules",
                return_value=type("Result", (), {"intent": "fallback_qa", "slots": {}})(),
            ), patch.object(RecordingDialogueNLU, "_is_confident", return_value=False):
                _, used_llm_fallback = nlu.parse_with_trace("x", brand_vocabulary=[], state_has_context=False)
                self.assertTrue(used_llm_fallback)

    def test_evaluate_dataset_computes_summary(self):
        dataset = [
            {"input": "预算4k拍照", "expected_intent": "recommend", "expected_slots": {"budget_max": 4000}},
        ]
        nlu = type(
            "StubNLU",
            (),
            {
                "parse_with_trace": staticmethod(
                    lambda *args, **kwargs: (
                        type("Result", (), {"intent": "recommend", "slots": {"budget_max": 4000}})(),
                        False,
                    )
                )
            },
        )()
        summary, records = evaluate_dataset(nlu, dataset)
        self.assertEqual(summary["intent_accuracy"], 1.0)
        self.assertEqual(summary["slot_f1"], 1.0)
        self.assertEqual(summary["fallback_to_llm_rate"], 0.0)
        self.assertEqual(len(records), 1)

    def test_main_outputs_structured_eval_error(self):
        with patch.object(dialogue_nlu_eval, "RecordingDialogueNLU", return_value=object()), patch.object(
            dialogue_nlu_eval,
            "load_dataset",
            return_value=[{"input": "x", "expected_intent": "recommend", "expected_slots": {}}],
        ), patch.object(
            dialogue_nlu_eval,
            "evaluate_dataset",
            side_effect=RuntimeError("eval failed"),
        ), patch.object(
            sys,
            "argv",
            ["dialogue_nlu_eval.py"],
        ), patch(
            "sys.stdout",
            new_callable=io.StringIO,
        ) as stdout:
            with self.assertRaises(SystemExit):
                dialogue_nlu_eval.main()
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["stage"], "nlu_eval")


if __name__ == "__main__":
    unittest.main()
