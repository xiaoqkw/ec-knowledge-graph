import inspect
import os
import sys
import time
from pathlib import Path

import evaluate
import torch

from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


from configuration.config import (
    BATCH_SIZE,
    BEST_MODEL_DIR,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
    LEARNING_RATE,
    LOG_DIR,
    MODEL_NAME,
    NER_DIR,
    PROCESSED_DATA_DIR,
    SAVE_STEPS,
    SEED,
    ensure_project_dirs,
)

def build_args():
    run_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = CHECKPOINT_DIR / NER_DIR
    logging_dir = LOG_DIR / NER_DIR / run_name
    logging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    kwargs = {
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "num_train_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": 3,
        "logging_strategy": "steps",
        "logging_steps": SAVE_STEPS,
        "metric_for_best_model": "eval_overall_f1",
        "greater_is_better": True,
        "load_best_model_at_end": True,
        "report_to": "tensorboard",
        "seed": SEED,
        "data_seed": SEED,
        "fp16": torch.cuda.is_available(),
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    if "eval_steps" in signature.parameters:
        kwargs["eval_steps"] = SAVE_STEPS

    return TrainingArguments(**kwargs)

def build_metrics(id_to_label):
    seqeval = evaluate.load("seqeval")
    def compute_metrics(prediction: EvalPrediction):
        logits = prediction.predictions
        label_ids = prediction.label_ids
        predictions = logits.argmax(axis=-1)

        all_predictions = []
        all_labels = []
        for pred, label_id in zip(predictions, label_ids):
            mask = label_id != -100
            pred = [id_to_label[int(pred)] for pred in pred[mask]]
            label_id = [id_to_label[int(label)] for label in label_id[mask]]
            all_predictions.append(pred)
            all_labels.append(label_id)

        return seqeval.compute(predictions=all_predictions, references=all_labels)

    return compute_metrics

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = load_from_disk(str(PROCESSED_DATA_DIR / "train"))
    valid_dataset = load_from_disk(str(PROCESSED_DATA_DIR / "valid"))

    collate_fn = DataCollatorForTokenClassification(
        tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=build_args(),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=build_metrics(ID_TO_LABEL),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        ],
    )

    trainer.train()
    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))

if __name__ == "__main__":
    train()
