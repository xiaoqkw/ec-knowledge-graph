import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import (
    ENTITY_TYPES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MAX_LENGTH,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_FILE,
    SEED,
    ensure_project_dirs,
)


def get_entity_type(entity: dict) -> str:
    labels = entity.get("labels", [])
    if not labels:
        raise ValueError(f"Found unlabeled entity span: {entity}")

    entity_type = str(labels[0]).strip().upper()
    if entity_type not in ENTITY_TYPES:
        raise ValueError(
            f"Unsupported entity label '{entity_type}'. "
            f"Please relabel data with ATTR / PEOPLE / SPEC."
        )
    return entity_type


def encode_example(example, tokenizer):
    tokens = list(example["text"])
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    char_labels = [LABEL_TO_ID["O"]] * len(tokens)
    entities = example.get("label") or []
    for entity in entities:
        start, end = entity["start"], entity["end"]
        if start >= len(tokens):
            continue

        entity_type = get_entity_type(entity)
        end = min(end, len(tokens))
        char_labels[start] = LABEL_TO_ID[f"B-{entity_type}"]
        for index in range(start + 1, end):
            char_labels[index] = LABEL_TO_ID[f"I-{entity_type}"]

    aligned_labels = []
    previous_word_id = None
    for word_id in tokenized.word_ids():
        if word_id is None:
            aligned_labels.append(-100)
            continue

        label_id = char_labels[word_id]
        if word_id == previous_word_id and label_id != LABEL_TO_ID["O"]:
            label_name = ID_TO_LABEL[label_id]
            if label_name.startswith("B-"):
                label_id = LABEL_TO_ID[f"I-{label_name[2:]}"]
        aligned_labels.append(label_id)
        previous_word_id = word_id

    tokenized["labels"] = aligned_labels
    return tokenized


def process():
    ensure_project_dirs()
    dataset = load_dataset("json", data_files=str(RAW_DATA_FILE), split="train")

    unused_columns = [
        "id",
        "annotator",
        "annotation_id",
        "created_at",
        "updated_at",
        "lead_time",
    ]
    dataset = dataset.remove_columns(unused_columns)

    dataset_dict = dataset.train_test_split(test_size=0.2, seed=SEED)
    dataset_test_valid = dataset_dict["test"].train_test_split(test_size=0.5, seed=SEED)
    dataset_dict["test"] = dataset_test_valid["train"]
    dataset_dict["valid"] = dataset_test_valid["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset_dict = dataset_dict.map(
        lambda example: encode_example(example, tokenizer),
        remove_columns=["text", "label"],
    )

    dataset_dict.save_to_disk(str(PROCESSED_DATA_DIR))


if __name__ == "__main__":
    process()
