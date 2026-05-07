import sys
import unittest
from pathlib import Path

from datasets import Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ner.error_analysis import (
    analyze_sample,
    build_summary,
    match_spans,
    sequence_to_spans,
    Span,
)
from ner.preprocess import clip_entities, encode_example


class DummyTokenizer:
    def __call__(self, tokens, is_split_into_words=True, truncation=True, max_length=128):
        class Result(dict):
            def word_ids(self_inner):
                return [None] + list(range(len(tokens))) + [None]

        return Result()


class NERErrorAnalysisTestCase(unittest.TestCase):
    def test_encode_example_keeps_id_and_clipped_fields(self):
        encoded = encode_example(
            {
                "id": 9001,
                "text": "abcd",
                "label": [{"start": 1, "end": 3, "text": "bc", "labels": ["ATTR"]}],
            },
            DummyTokenizer(),
        )
        self.assertEqual(encoded["id"], 9001)
        self.assertEqual(encoded["text"], "abcd")
        self.assertEqual(
            encoded["label"],
            [{"start": 1, "end": 3, "text": "bc", "labels": ["ATTR"]}],
        )
        self.assertIn("labels", encoded)

    def test_clip_entities_respects_window(self):
        text = "abcdef"
        entities = [
            {"start": 1, "end": 4, "text": "bcd", "labels": ["ATTR"]},
            {"start": 4, "end": 6, "text": "ef", "labels": ["SPEC"]},
        ]
        clipped = clip_entities(entities, text[:5], 5)
        self.assertEqual(
            clipped,
            [
                {"start": 1, "end": 4, "text": "bcd", "labels": ["ATTR"]},
                {"start": 4, "end": 5, "text": "e", "labels": ["SPEC"]},
            ],
        )

    def test_sequence_to_spans_handles_invalid_i_at_start(self):
        spans = sequence_to_spans("abc", ["I-ATTR", "I-ATTR", "O"])
        self.assertEqual(
            spans,
            [Span(start=0, end=2, entity_type="ATTR", text="ab")],
        )

    def test_sequence_to_spans_handles_invalid_i_after_other_type(self):
        spans = sequence_to_spans("abcd", ["B-ATTR", "I-ATTR", "I-SPEC", "O"])
        self.assertEqual(
            spans,
            [
                Span(start=0, end=2, entity_type="ATTR", text="ab"),
                Span(start=2, end=3, entity_type="SPEC", text="c"),
            ],
        )

    def test_match_spans_is_deterministic_for_overlap_tie_break(self):
        gold_spans = [
            Span(start=0, end=4, entity_type="ATTR", text="abcd"),
            Span(start=4, end=6, entity_type="ATTR", text="ef"),
        ]
        pred_spans = [
            Span(start=0, end=2, entity_type="ATTR", text="ab"),
            Span(start=1, end=4, entity_type="ATTR", text="bcd"),
            Span(start=4, end=6, entity_type="ATTR", text="ef"),
        ]

        matches = match_spans(gold_spans, pred_spans)
        boundary_matches = [
            match for match in matches
            if match["category"] == "boundary_mismatch"
        ]
        self.assertEqual(len(boundary_matches), 1)
        self.assertEqual(boundary_matches[0]["gold"], gold_spans[0])
        self.assertEqual(boundary_matches[0]["pred"], pred_spans[1])

        spurious_matches = [
            match for match in matches
            if match["category"] == "spurious"
        ]
        self.assertEqual(len(spurious_matches), 1)
        self.assertEqual(spurious_matches[0]["pred"], pred_spans[0])

    def test_analyze_sample_reports_type_mismatch(self):
        sample = analyze_sample(
            sample_id=1001,
            text="abc",
            gold_entities=[
                {"start": 0, "end": 2, "labels": ["ATTR"]},
            ],
            pred_labels=["B-SPEC", "I-SPEC", "O"],
        )
        self.assertEqual(sample["sample_id"], 1001)
        self.assertEqual(sample["split"], "test")
        self.assertEqual(sample["errors"][0]["category"], "type_mismatch")

    def test_build_summary_splits_gold_and_pred_views(self):
        sample_analyses = [
            analyze_sample(
                sample_id=1,
                text="abcd",
                gold_entities=[{"start": 0, "end": 2, "labels": ["ATTR"]}],
                pred_labels=["O", "O", "O", "O"],
            ),
            analyze_sample(
                sample_id=2,
                text="abcd",
                gold_entities=[],
                pred_labels=["B-SPEC", "I-SPEC", "O", "O"],
            ),
            analyze_sample(
                sample_id=3,
                text="abcd",
                gold_entities=[{"start": 0, "end": 2, "labels": ["ATTR"]}],
                pred_labels=["B-SPEC", "I-SPEC", "O", "O"],
            ),
        ]
        summary = build_summary({"overall_f1": 0.5}, sample_analyses)
        self.assertEqual(summary["per_gold_type_errors"]["ATTR"]["missing"], 1)
        self.assertEqual(summary["per_gold_type_errors"]["ATTR"]["type_mismatch"], 1)
        self.assertEqual(summary["per_pred_type_errors"]["SPEC"]["spurious"], 1)
        self.assertEqual(summary["per_pred_type_errors"]["SPEC"]["type_mismatch"], 1)
        self.assertEqual(summary["confusion_counts"][("ATTR", "SPEC")], 1)

    def test_predict_dataset_strips_analysis_only_columns(self):
        dataset = Dataset.from_list(
            [
                {
                    "id": 1,
                    "text": "abc",
                    "label": [],
                    "input_ids": [101, 102],
                    "token_type_ids": [0, 0],
                    "attention_mask": [1, 1],
                    "labels": [-100, 0],
                }
            ]
        )
        predict_dataset = dataset.remove_columns(["id", "text", "label"])
        self.assertEqual(
            predict_dataset.column_names,
            ["input_ids", "token_type_ids", "attention_mask", "labels"],
        )

    def test_train_dataset_strips_analysis_only_columns(self):
        dataset = Dataset.from_list(
            [
                {
                    "id": 1,
                    "text": "abc",
                    "label": [],
                    "input_ids": [101, 102],
                    "token_type_ids": [0, 0],
                    "attention_mask": [1, 1],
                    "labels": [-100, 0],
                }
            ]
        )
        train_features = dataset.remove_columns(["id", "text", "label"])
        self.assertEqual(
            train_features.column_names,
            ["input_ids", "token_type_ids", "attention_mask", "labels"],
        )


if __name__ == "__main__":
    unittest.main()
