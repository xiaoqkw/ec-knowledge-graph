from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_LEFT = ROOT_DIR / "logs" / "eval" / "entity_linking_hybrid_memory_off.jsonl"
DEFAULT_RIGHT = ROOT_DIR / "logs" / "eval" / "entity_linking_hybrid_rerank_memory_off.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record.get("label", "")), str(record.get("query", "")), str(record.get("gold", "")))


def align_records(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    *,
    label: str | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    right_by_key = {record_key(item): item for item in right_records}
    pairs = []
    for left in left_records:
        if label is not None and str(left.get("label")) != label:
            continue
        key = record_key(left)
        right = right_by_key.get(key)
        if right is None:
            continue
        pairs.append((left, right))
    return pairs


def _metric_hit(record: dict[str, Any], metric: str) -> int:
    return int(bool(record.get(metric, False)))


def paired_accuracy_diff(pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, metric: str) -> float:
    if not pairs:
        return 0.0
    left_mean = sum(_metric_hit(left, metric) for left, _ in pairs) / len(pairs)
    right_mean = sum(_metric_hit(right, metric) for _, right in pairs) / len(pairs)
    return right_mean - left_mean


def bootstrap_confidence_interval(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    metric: str,
    num_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    observed = paired_accuracy_diff(pairs, metric=metric)
    if not pairs:
        return {
            "metric": metric,
            "num_pairs": 0,
            "observed_diff": observed,
            "confidence": confidence,
            "lower": 0.0,
            "upper": 0.0,
            "num_samples": num_samples,
            "seed": seed,
        }

    rng = random.Random(seed)
    diffs = []
    size = len(pairs)
    for _ in range(num_samples):
        sample = [pairs[rng.randrange(size)] for _ in range(size)]
        diffs.append(paired_accuracy_diff(sample, metric=metric))
    diffs.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, math.floor((alpha / 2.0) * len(diffs)))
    upper_index = min(len(diffs) - 1, math.ceil((1.0 - alpha / 2.0) * len(diffs)) - 1)
    return {
        "metric": metric,
        "num_pairs": size,
        "observed_diff": observed,
        "confidence": confidence,
        "lower": diffs[lower_index],
        "upper": diffs[upper_index],
        "num_samples": num_samples,
        "seed": seed,
    }


def mcnemar_table(pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, metric: str) -> dict[str, int]:
    both_correct = 0
    both_wrong = 0
    left_only = 0
    right_only = 0
    for left, right in pairs:
        left_hit = bool(left.get(metric, False))
        right_hit = bool(right.get(metric, False))
        if left_hit and right_hit:
            both_correct += 1
        elif left_hit and not right_hit:
            left_only += 1
        elif not left_hit and right_hit:
            right_only += 1
        else:
            both_wrong += 1
    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "left_only": left_only,
        "right_only": right_only,
    }


def exact_mcnemar_pvalue(*, left_only: int, right_only: int) -> float:
    total = left_only + right_only
    if total == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(total, i) for i in range(smaller + 1)) * (0.5 ** total)
    return min(1.0, 2.0 * tail)


def build_significance_report(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    *,
    label: str | None,
    metric: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    pairs = align_records(left_records, right_records, label=label)
    table = mcnemar_table(pairs, metric=metric)
    bootstrap = bootstrap_confidence_interval(
        pairs,
        metric=metric,
        num_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "label_filter": label,
        "metric": metric,
        "num_pairs": len(pairs),
        "mcnemar": {
            **table,
            "p_value": exact_mcnemar_pvalue(left_only=table["left_only"], right_only=table["right_only"]),
        },
        "bootstrap": bootstrap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired significance tests for entity linking baselines.")
    parser.add_argument("--left", default=str(DEFAULT_LEFT))
    parser.add_argument("--right", default=str(DEFAULT_RIGHT))
    parser.add_argument("--label", default="SKU")
    parser.add_argument("--metric", choices=["top1_hit", "topk_hit"], default="top1_hit")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_records = load_jsonl(Path(args.left))
    right_records = load_jsonl(Path(args.right))
    report = build_significance_report(
        left_records,
        right_records,
        label=args.label or None,
        metric=args.metric,
        bootstrap_samples=max(100, args.bootstrap_samples),
        confidence=args.confidence,
        seed=args.seed,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
