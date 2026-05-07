import os
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
)

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import BEST_MODEL_DIR, ID_TO_LABEL, LOG_DIR, PROCESSED_DATA_DIR
from ner.error_analysis import analyze_sample, build_summary, write_bad_cases, write_confusion_csv, write_summary
from ner.metrics import collect_sequences, compute_metrics_from_sequences


def evaluate_model():
    model = AutoModelForTokenClassification.from_pretrained(str(BEST_MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(BEST_MODEL_DIR))
    test_dataset = load_from_disk(str(PROCESSED_DATA_DIR / "test"))
    predict_dataset = test_dataset.remove_columns(["id", "text", "label"])

    collater = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        eval_dataset=predict_dataset,
        data_collator=collater,
    )

    prediction_output = trainer.predict(predict_dataset)
    predictions = prediction_output.predictions.argmax(axis=-1)
    label_ids = prediction_output.label_ids
    sequences = collect_sequences(predictions, label_ids, ID_TO_LABEL)
    result = compute_metrics_from_sequences(*sequences)

    sample_analyses = []
    all_predictions = sequences[0]
    for index, sample in enumerate(test_dataset):
        sample_analyses.append(
            analyze_sample(
                sample_id=sample["id"],
                text=sample["text"],
                gold_entities=sample["label"],
                pred_labels=all_predictions[index],
            )
        )

    log_dir = LOG_DIR / "ner"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(result, sample_analyses)
    write_bad_cases(log_dir / "ner_bad_cases.jsonl", sample_analyses)
    write_confusion_csv(log_dir / "ner_confusion.csv", summary["confusion_counts"])
    write_summary(log_dir / "ner_error_summary.json", summary)
    print(result)


if __name__ == '__main__':
    evaluate_model()
