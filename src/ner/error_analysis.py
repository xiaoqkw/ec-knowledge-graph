import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from configuration.config import ENTITY_TYPES


MATCH_CATEGORIES = (
    "correct",
    "type_mismatch",
    "boundary_mismatch",
    "boundary_and_type_mismatch",
)
ERROR_CATEGORIES = (
    "type_mismatch",
    "boundary_mismatch",
    "boundary_and_type_mismatch",
    "missing",
    "spurious",
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity_type: str
    text: str

    def overlaps(self, other: "Span") -> bool:
        return min(self.end, other.end) > max(self.start, other.start)

    def overlap_length(self, other: "Span") -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def boundary_distance(self, other: "Span") -> int:
        return abs(self.start - other.start) + abs(self.end - other.end)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "text": self.text,
        }


def sequence_to_spans(text: str, labels: list[str]) -> list[Span]:
    spans = []
    current_start = None
    current_type = None

    def flush(end_index: int) -> None:
        nonlocal current_start, current_type
        if current_start is not None and current_type is not None:
            spans.append(
                Span(
                    start=current_start,
                    end=end_index,
                    entity_type=current_type,
                    text=text[current_start:end_index],
                )
            )
        current_start = None
        current_type = None

    for index, label in enumerate(labels):
        if label == "O":
            flush(index)
            continue

        prefix, entity_type = label.split("-", maxsplit=1)
        if prefix == "B":
            flush(index)
            current_start = index
            current_type = entity_type
            continue

        if current_type == entity_type and current_start is not None:
            continue

        flush(index)
        current_start = index
        current_type = entity_type

    flush(len(labels))
    return spans


def gold_entities_to_spans(text: str, gold_entities: list[dict]) -> list[Span]:
    spans = []
    for entity in gold_entities:
        labels = entity.get("labels", [])
        if not labels:
            continue
        start = int(entity["start"])
        end = int(entity["end"])
        spans.append(
            Span(
                start=start,
                end=end,
                entity_type=str(labels[0]).strip().upper(),
                text=text[start:end],
            )
        )
    return spans


def classify_pair(gold: Span, pred: Span) -> str | None:
    if gold.start == pred.start and gold.end == pred.end:
        if gold.entity_type == pred.entity_type:
            return "correct"
        return "type_mismatch"

    if not gold.overlaps(pred):
        return None

    if gold.entity_type == pred.entity_type:
        return "boundary_mismatch"
    return "boundary_and_type_mismatch"


def candidate_sort_key(gold: Span, pred: Span) -> tuple[int, int, int, int]:
    return (
        -gold.overlap_length(pred),
        gold.boundary_distance(pred),
        gold.start,
        pred.start,
    )


def match_spans(gold_spans: list[Span], pred_spans: list[Span]) -> list[dict]:
    gold_matched = set()
    pred_matched = set()
    matches = []

    for category in MATCH_CATEGORIES:
        candidates = []
        for gold_index, gold in enumerate(gold_spans):
            if gold_index in gold_matched:
                continue
            for pred_index, pred in enumerate(pred_spans):
                if pred_index in pred_matched:
                    continue
                if classify_pair(gold, pred) != category:
                    continue
                candidates.append(
                    (
                        candidate_sort_key(gold, pred),
                        gold_index,
                        pred_index,
                    )
                )

        candidates.sort()
        for _, gold_index, pred_index in candidates:
            if gold_index in gold_matched or pred_index in pred_matched:
                continue
            gold = gold_spans[gold_index]
            pred = pred_spans[pred_index]
            matches.append(
                {
                    "category": category,
                    "gold": gold,
                    "pred": pred,
                }
            )
            gold_matched.add(gold_index)
            pred_matched.add(pred_index)

    for gold_index, gold in enumerate(gold_spans):
        if gold_index not in gold_matched:
            matches.append(
                {
                    "category": "missing",
                    "gold": gold,
                    "pred": None,
                }
            )

    for pred_index, pred in enumerate(pred_spans):
        if pred_index not in pred_matched:
            matches.append(
                {
                    "category": "spurious",
                    "gold": None,
                    "pred": pred,
                }
            )

    return matches


def build_error_entry(match: dict) -> dict:
    return {
        "category": match["category"],
        "gold": match["gold"].to_dict() if match["gold"] is not None else None,
        "pred": match["pred"].to_dict() if match["pred"] is not None else None,
    }


def analyze_sample(sample_id, text: str, gold_entities: list[dict], pred_labels: list[str]) -> dict:
    gold_spans = gold_entities_to_spans(text, gold_entities)
    pred_spans = sequence_to_spans(text, pred_labels)
    matches = match_spans(gold_spans, pred_spans)
    errors = [build_error_entry(match) for match in matches if match["category"] != "correct"]
    return {
        "sample_id": sample_id,
        "split": "test",
        "text": text,
        "gold_spans": [span.to_dict() for span in gold_spans],
        "pred_spans": [span.to_dict() for span in pred_spans],
        "matches": matches,
        "errors": errors,
    }


def build_summary(metrics: dict, sample_analyses: list[dict]) -> dict:
    error_counts = Counter()
    per_gold_type_errors = {
        entity_type: Counter({category: 0 for category in ERROR_CATEGORIES})
        for entity_type in ENTITY_TYPES
    }
    per_pred_type_errors = {
        entity_type: Counter({category: 0 for category in ERROR_CATEGORIES})
        for entity_type in ENTITY_TYPES
    }
    confusion_counts = Counter()

    for sample in sample_analyses:
        for match in sample["matches"]:
            category = match["category"]
            if category == "correct":
                continue

            error_counts[category] += 1
            gold = match["gold"]
            pred = match["pred"]

            if gold is not None:
                per_gold_type_errors[gold.entity_type][category] += 1
            if pred is not None:
                per_pred_type_errors[pred.entity_type][category] += 1
            if category in {"type_mismatch", "boundary_and_type_mismatch"}:
                confusion_counts[(gold.entity_type, pred.entity_type)] += 1

    total_errors = sum(error_counts.values())
    error_distribution = {}
    for category in ERROR_CATEGORIES:
        count = error_counts[category]
        error_distribution[category] = {
            "count": count,
            "ratio": count / total_errors if total_errors else 0.0,
        }

    return {
        "metrics": metrics,
        "error_distribution": error_distribution,
        "per_gold_type_errors": {
            entity_type: dict(counter)
            for entity_type, counter in per_gold_type_errors.items()
        },
        "per_pred_type_errors": {
            entity_type: dict(counter)
            for entity_type, counter in per_pred_type_errors.items()
        },
        "confusion_counts": confusion_counts,
    }


def write_bad_cases(path: Path, sample_analyses: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in sample_analyses:
            if not sample["errors"]:
                continue
            payload = {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "text": sample["text"],
                "gold_spans": sample["gold_spans"],
                "pred_spans": sample["pred_spans"],
                "errors": sample["errors"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_confusion_csv(path: Path, confusion_counts: Counter) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_type", "pred_type", "count"])
        for (gold_type, pred_type), count in sorted(confusion_counts.items()):
            writer.writerow([gold_type, pred_type, count])


def write_summary(path: Path, summary: dict) -> None:
    payload = dict(summary)
    payload.pop("confusion_counts", None)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

