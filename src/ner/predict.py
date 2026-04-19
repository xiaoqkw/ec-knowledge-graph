import os
import sys
from pathlib import Path
from typing import List, Sequence

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

    def predict(self, inputs: str | List[str]):
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
        tokenized = { k:v.to(self.device) for k, v in tokenized.items() }

        with torch.no_grad():
            logits = self.model(**tokenized).logits
            predictions = torch.argmax(logits, dim=-1).tolist()

        outputs = []
        for tokens, prediction in zip(tokens_list, predictions):
            valid_list = prediction[1 : len(tokens)+1]
            entities = [self.model.config.id2label[index] for index in valid_list]
            outputs.append(entities)

        return outputs[0] if is_single else outputs

    def extract(self, inputs: str | List[str]):
        is_single = isinstance(inputs, str)
        batch_inputs = [inputs] if is_single else inputs
        predictions = self.predict(batch_inputs)

        entities = []
        for input, prediction in zip(batch_inputs, predictions):
            entities.append(self._extract_entity(input, prediction))

        return entities[0] if is_single else entities

    def _extract_entity(self, tokens, prediction):
        entity_list = []
        current = []

        tokens_list = list(tokens)
        for token, pred in zip(tokens_list, prediction):
            if pred == 'B':
                if current:
                    entity_list.append("".join(current))
                current = [token]
            elif pred == 'I':
                if current:
                    current.append(token)
            else:
                if current:
                    entity_list.append("".join(current))
                current = []

        if current:
            entity_list.append("".join(current))

        return entity_list

def load_predictor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(str(BEST_MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(BEST_MODEL_DIR))
    return Predictor(model, tokenizer, device)


def demo() -> None:
    predictor = load_predictor()
    texts = [
        "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩",
        "滋源无硅油无患子洗发水护发素女士2653控油清爽",
    ]
    print("BIO 标签结果:")
    predictions = predictor.predict(texts)
    for text, labels in zip(texts, predictions):
        print(text)
        print(labels)

    print("\n抽取出的实体:")
    print(predictor.extract(texts))

if __name__ == "__main__":
    demo()

