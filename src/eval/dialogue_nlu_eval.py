import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dialogue.nlu import DialogueNLU

BRAND_VOCABULARY = ["苹果", "华为", "小米", "OPPO", "VIVO", "红米"]


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class RecordingDialogueNLU(DialogueNLU):
    def __init__(self, llm_enabled: bool = True):
        super().__init__(llm_enabled=llm_enabled)
        self.llm_fallback_calls = 0
        self._used_llm_fallback = False

    def _parse_with_llm(self, message: str, brand_vocabulary: list[str]):
        self.llm_fallback_calls += 1
        self._used_llm_fallback = True
        return super()._parse_with_llm(message, brand_vocabulary)

    def parse_with_trace(self, message: str, *, brand_vocabulary: list[str], state_has_context: bool):
        self._used_llm_fallback = False
        result = self.parse(
            message,
            brand_vocabulary=brand_vocabulary,
            state_has_context=state_has_context,
        )
        return result, self._used_llm_fallback


def slot_f1(expected_slots: dict, predicted_slots: dict) -> tuple[int, int, int]:
    expected = {(key, value) for key, value in expected_slots.items()}
    predicted = {(key, value) for key, value in predicted_slots.items()}
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    return tp, fp, fn


def evaluate_dataset(nlu: RecordingDialogueNLU, dataset: list[dict]) -> tuple[dict, list[dict]]:
    records = []
    intent_hits = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    fallback_hits = 0

    for item in dataset:
        result, used_llm_fallback = nlu.parse_with_trace(
            item["input"],
            brand_vocabulary=BRAND_VOCABULARY,
            state_has_context=bool(item.get("state_has_context", False)),
        )
        predicted_slots = dict(result.slots)
        expected_slots = dict(item.get("expected_slots", {}))
        tp, fp, fn = slot_f1(expected_slots, predicted_slots)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        intent_hit = result.intent == item["expected_intent"]
        intent_hits += int(intent_hit)
        fallback_hits += int(used_llm_fallback)
        records.append(
            {
                "input": item["input"],
                "expected_intent": item["expected_intent"],
                "predicted_intent": result.intent,
                "expected_slots": expected_slots,
                "predicted_slots": predicted_slots,
                "intent_hit": intent_hit,
                "used_llm_fallback": used_llm_fallback,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {
        "total_samples": len(dataset),
        "intent_accuracy": intent_hits / len(dataset) if dataset else 0.0,
        "slot_f1": f1,
        "fallback_to_llm_rate": fallback_hits / len(dataset) if dataset else 0.0,
        "per_case_breakdown": records,
    }
    return summary, records


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_logs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DialogueNLU with real parsing and diagnostic logging.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "data" / "eval" / "dialogue_nlu_samples.jsonl"))
    args = parser.parse_args()

    try:
        nlu = RecordingDialogueNLU(llm_enabled=True)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": "nlu_init",
                    "message": f"Failed to initialize DialogueNLU with real LLM backend: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc
    dataset = load_dataset(Path(args.dataset))
    try:
        summary, records = evaluate_dataset(nlu, dataset)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": "nlu_eval",
                    "message": f"DialogueNLU evaluation failed during sample execution: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc
    write_summary(ROOT_DIR / "logs" / "eval" / "dialogue_nlu_diagnostic.json", summary)
    write_logs(ROOT_DIR / "logs" / "eval" / "dialogue_nlu_cases.jsonl", records)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_case_breakdown"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
