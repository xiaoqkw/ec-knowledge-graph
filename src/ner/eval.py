import os
import sys
from pathlib import Path

import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    Trainer,
)

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import BEST_MODEL_DIR, PROCESSED_DATA_DIR, ID_TO_LABEL


def compute_metrics_builder(id_to_label):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(prediction: EvalPrediction):
        # 与训练阶段保持同一套实体级评估逻辑，方便横向对比。
        logits = prediction.predictions
        preds = logits.argmax(axis=-1)
        labels = prediction.label_ids

        all_predictions = []
        all_labels = []
        for pred_ids, label_ids in zip(preds, labels):
            valid_mask = label_ids != -100
            pred_sequence = [id_to_label[int(item)] for item in pred_ids[valid_mask]]
            label_sequence = [id_to_label[int(item)] for item in label_ids[valid_mask]]
            all_predictions.append(pred_sequence)
            all_labels.append(label_sequence)

        return seqeval.compute(predictions=all_predictions, references=all_labels)

    return compute_metrics

def evaluate_model():
    model = AutoModelForTokenClassification.from_pretrained(str(BEST_MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(BEST_MODEL_DIR))
    test_dataset = load_from_disk(str(PROCESSED_DATA_DIR / "test"))

    collater = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        data_collator=collater,
        compute_metrics=compute_metrics_builder(ID_TO_LABEL),
    )

    result = trainer.evaluate()
    print(result)

if __name__ == '__main__':
    evaluate_model()