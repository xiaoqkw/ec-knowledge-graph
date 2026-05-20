from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.entity_linking_eval import load_dataset


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_by_label(dataset: list[dict], *, per_label: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in dataset:
        grouped.setdefault(item["label"], []).append(item)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for label in sorted(grouped):
        rows = list(grouped[label])
        if len(rows) < per_label:
            raise ValueError(f"Label {label} only has {len(rows)} rows, which is less than requested {per_label}.")
        sampled.extend(rng.sample(rows, per_label))
    rng.shuffle(sampled)
    return sampled


def build_summary(records: list[dict], *, input_path: str, output_path: str, per_label: int, seed: int) -> dict:
    label_counts = Counter(item["label"] for item in records)
    unique_gold_counts = Counter((item["label"], item["gold"]) for item in records)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "per_label": per_label,
        "seed": seed,
        "total": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "unique_golds": len(unique_gold_counts),
        "unique_golds_by_label": {
            label: len({item["gold"] for item in records if item["label"] == label})
            for label in sorted(label_counts)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a balanced entity linking dataset by label.")
    parser.add_argument(
        "--input",
        default=str(ROOT_DIR / "data" / "eval" / "entity_linking_v2_llm.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "data" / "eval" / "entity_linking_v2_llm_sampled_100_per_label.jsonl"),
    )
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(Path(args.input))
    sampled = sample_by_label(dataset, per_label=args.per_label, seed=args.seed)
    output_path = Path(args.output)
    write_jsonl(output_path, sampled)
    summary = build_summary(
        sampled,
        input_path=args.input,
        output_path=args.output,
        per_label=args.per_label,
        seed=args.seed,
    )
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
