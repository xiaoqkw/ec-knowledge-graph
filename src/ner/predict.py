import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import BEST_MODEL_DIR


class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, inputs: str | list[str]):
        is_single = isinstance(inputs, str)
        batch_inputs = [inputs] if is_single else inputs

        tokens_list = [list(text) for text in batch_inputs]
        tokenized = self.tokenizer(
            tokens_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}

        with torch.no_grad():
            logits = self.model(**tokenized).logits
            predictions = torch.argmax(logits, dim=-1).tolist()

        outputs = []
        for tokens, prediction in zip(tokens_list, predictions):
            valid_list = prediction[1 : len(tokens) + 1]
            outputs.append([self.model.config.id2label[index] for index in valid_list])

        return outputs[0] if is_single else outputs

    def extract(self, inputs: str | list[str]):
        is_single = isinstance(inputs, str)
        batch_inputs = [inputs] if is_single else inputs
        predictions = self.predict(batch_inputs)

        entities = []
        for text, prediction in zip(batch_inputs, predictions):
            entities.append(self._extract_entities(text, prediction))

        return entities[0] if is_single else entities

    def _extract_entities(self, text: str, prediction: list[str]) -> list[dict]:
        entities = []
        current_text = []
        current_start = None
        current_type = None

        def flush(end_index: int) -> None:
            nonlocal current_text, current_start, current_type
            if current_text and current_type is not None and current_start is not None:
                entities.append(
                    {
                        "text": "".join(current_text),
                        "entity_type": current_type,
                        "start": current_start,
                        "end": end_index,
                    }
                )
            current_text = []
            current_start = None
            current_type = None

        for index, (token, label) in enumerate(zip(text, prediction)):
            if label == "O":
                flush(index)
                continue

            prefix, entity_type = label.split("-", maxsplit=1)
            if prefix == "B":
                flush(index)
                current_text = [token]
                current_start = index
                current_type = entity_type
                continue

            if current_type == entity_type and current_text:
                current_text.append(token)
            else:
                flush(index)
                current_text = [token]
                current_start = index
                current_type = entity_type

        flush(len(text))
        return entities


def load_predictor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(str(BEST_MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(BEST_MODEL_DIR))
    return Predictor(model, tokenizer, device)


def demo() -> None:
    predictor = load_predictor()
    texts = [
        "麦德龙德国进口双心多维叶黄素护眼营养软胶囊60粒3盒眼干涩",
        "滋源无硅油无患子洗发水护发素女士2653控油清爽",
    ]
    print("BIO labels:")
    predictions = predictor.predict(texts)
    for text, labels in zip(texts, predictions):
        print(text)
        print(labels)

    print("\nExtracted entities:")
    print(predictor.extract(texts))


if __name__ == "__main__":
    demo()
