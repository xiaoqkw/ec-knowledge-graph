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

from configuration.config import BEST_MODEL_DIR, ID_TO_LABEL, PROCESSED_DATA_DIR
from ner.metrics import build_metrics


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
        compute_metrics=build_metrics(ID_TO_LABEL),
    )

    result = trainer.evaluate()
    print(result)


if __name__ == '__main__':
    evaluate_model()
