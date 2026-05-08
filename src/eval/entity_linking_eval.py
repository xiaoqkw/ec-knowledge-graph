import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import LOG_DIR
from web.service import ChatService


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_metrics(records: list[dict], top_k: int = 3) -> dict:
    total = len(records)
    top1_hits = sum(1 for item in records if item["top1_hit"])
    topk_hits = sum(1 for item in records if item["topk_hit"])

    by_label = defaultdict(lambda: {"total": 0, "top1_hits": 0})
    for item in records:
        stats = by_label[item["label"]]
        stats["total"] += 1
        stats["top1_hits"] += int(item["top1_hit"])

    return {
        "total": total,
        "top1_accuracy": top1_hits / total if total else 0.0,
        "top_k": top_k,
        "topk_recall": topk_hits / total if total else 0.0,
        "by_label_accuracy": {
            label: stats["top1_hits"] / stats["total"] if stats["total"] else 0.0
            for label, stats in sorted(by_label.items())
        },
    }


def evaluate_baseline(service: ChatService, dataset: list[dict], baseline: str, top_k: int) -> tuple[dict, list[dict]]:
    records = []
    for item in dataset:
        candidates = service._search_entities(item["label"], item["query"], mode=baseline, k=top_k)
        record = {
            "query": item["query"],
            "label": item["label"],
            "gold": item["gold"],
            "candidates": candidates,
            "top1_hit": bool(candidates and candidates[0] == item["gold"]),
            "topk_hit": item["gold"] in candidates[:top_k],
        }
        records.append(record)
    return compute_metrics(records, top_k=top_k), records


def write_logs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        baseline: value["metrics"]
        for baseline, value in results.items()
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_report(results: dict) -> None:
    print("| baseline | total | top1_accuracy | topk_recall |")
    print("| --- | ---: | ---: | ---: |")
    for baseline, payload in results.items():
        metrics = payload["metrics"]
        print(
            f"| {baseline} | {metrics['total']} | {metrics['top1_accuracy']:.4f} | {metrics['topk_recall']:.4f} |"
        )
        for label, accuracy in metrics["by_label_accuracy"].items():
            print(f"| {baseline}:{label} | - | {accuracy:.4f} | - |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate entity linking baselines.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT_DIR / "data" / "eval" / "entity_linking.jsonl"),
    )
    parser.add_argument(
        "--baseline",
        choices=["exact_match", "fulltext", "hybrid", "all"],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    baselines = ["exact_match", "fulltext", "hybrid"] if args.baseline == "all" else [args.baseline]
    dataset = load_dataset(Path(args.dataset))
    service = ChatService(llm_enabled=False)
    try:
        results = {}
        for baseline in baselines:
            metrics, records = evaluate_baseline(service, dataset, baseline, args.top_k)
            write_logs(LOG_DIR / "eval" / f"entity_linking_{baseline}.jsonl", records)
            results[baseline] = {"metrics": metrics, "records": records}
        write_summary(LOG_DIR / "eval" / "entity_linking_summary.json", results)
        print_report(results)
    finally:
        service.close()


if __name__ == "__main__":
    main()
